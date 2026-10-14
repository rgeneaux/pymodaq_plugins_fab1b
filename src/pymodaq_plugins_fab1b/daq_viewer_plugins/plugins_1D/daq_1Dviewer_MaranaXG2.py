from __future__ import annotations

import copy
import threading
import time

import numpy as np
from qtpy import QtCore

from pymodaq.control_modules.viewer_utility_classes import main
from pymodaq.utils.data import Axis, DataFromPlugins
from pymodaq_data.data import DataToExport

from pymodaq_plugins_fab1b import g2_analysis
from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaX import (
    DAQ_2DViewer_MaranaX,
)

G2_MODE = "G2 Coincidence"
HARMONIC_NAMES = [f"h{i + 1}" for i in range(g2_analysis.N_HARMONICS)]


class MaranaXG2Worker(QtCore.QObject):
    """Persistent per-shot HBT coincidence-counting loop.

    Same skeleton as ``MaranaXAcquisitionWorker._run_ta_continuous`` in
    daq_2Dviewer_MaranaX.py: ``AcquisitionStart`` happens once and is only
    torn down on an explicit stop(), the loop polls ``wait_frame`` with a
    100 ms timeout, and every SDK buffer is requeued before the next
    iteration. Unlike that worker, every frame is always processed (there is
    no "discard vs accumulate" distinction to make - every extra shot only
    helps g2's statistics), and this class is not shared with the other
    plugins: its per-frame work (multi-area extraction, thresholding, a
    growing click-history buffer) is different enough from the shared
    worker's row-sum/chunk-fill logic to warrant its own class.

    Two things the loop can be doing with an incoming frame, decided fresh
    each iteration (never mid-frame):

    - Calibrating: accumulate the frame's per-pixel value into a running
      mean/std (:class:`g2_analysis.DarkCalibrationAccumulator`) over the
      *full* AOI, not yet split by area. Started by :meth:`request_calibration`
      (a plain int attribute, set from the GUI thread and read once when a
      *new* target lands - same "set-then-read, no lock needed" pattern
      already used for the TA worker's ``_requested_chunk_size``). On
      completion the accumulated mean/std are handed back to the plugin,
      which combines them with the checkerboard geometry and the noise/
      margin settings to build ``valid_area_masks``/``thresholds`` - see
      below.
    - Counting: only once ``valid_area_masks``/``thresholds`` are set (both
      start ``None`` and stay so until a calibration completes). Extracts
      one shot's 8-element click vector and appends it to a capacity-
      doubling buffer; every ``update_every_n_shots`` shots, recomputes the
      cumulative g2 matrices and per-area click counts from the *whole*
      history so far and emits them.

    ``valid_area_masks``/``thresholds`` are plain attributes the plugin can
    reassign directly from the GUI thread (e.g. when the detection margin
    changes) - the loop reads a fresh local reference each iteration, so a
    reassignment takes effect on the very next frame with no restart needed.
    This is safe because a Python attribute write is an atomic pointer swap;
    the loop never sees a partially-updated array, only the old or the new
    one.
    """

    STATS_INTERVAL_S = 0.5

    calibration_progress = QtCore.Signal(int, int)  # shots done, target
    calibration_done = QtCore.Signal(object, object)  # mean (H,W), std (H,W)
    g2_update = QtCore.Signal(object, object, int, float)  # g2 dict, counts(8,), total_shots, shot_rate_hz
    acquisition_error = QtCore.Signal(str)
    acquisition_stopped = QtCore.Signal()

    def __init__(self, camera, valid_area_masks, thresholds, update_every_n_shots):
        super().__init__()
        self.camera = camera
        self.valid_area_masks = valid_area_masks  # (8,H,W) bool, or None until calibrated
        self.thresholds = thresholds  # (H,W) float, or None until calibrated
        self.update_every_n_shots = int(update_every_n_shots)

        self.total_shots = 0
        self._stop_event = threading.Event()
        self._reset_requested = threading.Event()
        self._calibration_target = None  # int or None; see class docstring
        self._rng = np.random.default_rng()

        self._capacity = 100_000
        self._buffer = np.zeros((self._capacity, g2_analysis.N_AREAS), dtype=bool)
        self._filled = 0
        self._last_emit_t = None
        self._last_emit_shots = 0

    def request_calibration(self, n_shots):
        self._calibration_target = int(n_shots)

    def request_reset(self):
        self._reset_requested.set()

    def _append_click(self, clicks):
        if self._filled >= self._capacity:
            self._capacity *= 2
            grown = np.zeros((self._capacity, g2_analysis.N_AREAS), dtype=bool)
            grown[: self._filled] = self._buffer[: self._filled]
            self._buffer = grown
        self._buffer[self._filled] = clicks
        self._filled += 1

    def _reset_buffer(self):
        self._capacity = 100_000
        self._buffer = np.zeros((self._capacity, g2_analysis.N_AREAS), dtype=bool)
        self._filled = 0
        self.total_shots = 0
        self._last_emit_t = None
        self._last_emit_shots = 0

    @QtCore.Slot()
    def run(self):
        # See MaranaXAcquisitionWorker.run() for why there is no
        # self._stop_event.clear() here: each worker is a fresh instance
        # with an already-cleared Event, and clearing on entry would risk
        # erasing a stop() that landed before this slot actually started
        # (QThread.start() is async).
        try:
            self.camera.start()
            self._run_loop()
        except Exception as exc:
            self.acquisition_error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if self.camera.acquiring:
                    self.camera.stop()
            except Exception as exc:
                self.acquisition_error.emit(f"Error stopping camera: {exc}")
            self.acquisition_stopped.emit()

    def _run_loop(self):
        calib_accumulator = None
        calib_target = None
        calib_progress_t0 = 0.0

        while not self._stop_event.is_set():
            result = self.camera.wait_frame(timeout_ms=100)
            if result is None:
                continue

            buffer, _size = result
            try:
                # frame_view() is a view into the SDK's buffer; it must not
                # be retained past requeue() below, but every use of it here
                # (accumulator.add, extract_clicks) immediately produces a
                # new, independent array, so nothing outlives this iteration.
                frame = self.camera.frame_view(buffer)

                if self._reset_requested.is_set():
                    self._reset_buffer()
                    self._reset_requested.clear()

                requested = self._calibration_target
                if requested is not None:
                    if calib_accumulator is None or calib_target != requested:
                        calib_accumulator = g2_analysis.DarkCalibrationAccumulator(shape=frame.shape)
                        calib_target = requested
                        calib_progress_t0 = time.perf_counter()
                    calib_accumulator.add(frame)

                    now = time.perf_counter()
                    if now - calib_progress_t0 >= self.STATS_INTERVAL_S:
                        self.calibration_progress.emit(calib_accumulator.n, calib_target)
                        calib_progress_t0 = now

                    if calib_accumulator.n >= calib_target:
                        self.calibration_done.emit(calib_accumulator.mean.copy(), calib_accumulator.std.copy())
                        self._calibration_target = None
                        calib_accumulator = None
                        calib_target = None
                    # While calibrating, pixel values feed the accumulator
                    # only - click extraction needs thresholds that may not
                    # exist yet (first calibration) or no longer describe
                    # this run (a recalibration in progress).
                else:
                    calib_accumulator = None
                    masks = self.valid_area_masks
                    thresholds = self.thresholds
                    if masks is not None and thresholds is not None:
                        clicks = g2_analysis.extract_clicks(frame, masks, thresholds)
                        self._append_click(clicks)
                        self.total_shots += 1

                        if self.total_shots % self.update_every_n_shots == 0:
                            now = time.perf_counter()
                            if self._last_emit_t is not None and now > self._last_emit_t:
                                rate = (self.total_shots - self._last_emit_shots) / (now - self._last_emit_t)
                            else:
                                rate = 0.0
                            self._last_emit_t = now
                            self._last_emit_shots = self.total_shots

                            history = self._buffer[: self._filled]
                            g2_result = g2_analysis.g2_matrix(history, rng=self._rng)
                            counts = g2_analysis.click_counts(history)
                            self.g2_update.emit(g2_result, counts, self.total_shots, rate)
                    # else: no calibration yet - frames are drained (kept
                    # flowing so the SDK buffer pool never overflows) but
                    # nothing is counted until valid_area_masks/thresholds
                    # exist.
            finally:
                self.camera.requeue(buffer)

    def stop(self):
        self._stop_event.set()


def _build_params():
    # Deep copy: the nested dicts (e.g. acquisition_mode's own 'limits'
    # list) are shared object references with the base class's params, so
    # mutating them in place would also change DAQ_2DViewer_MaranaX's own
    # Preview/Chunked-only mode list.
    params = copy.deepcopy(DAQ_2DViewer_MaranaX.params)
    for entry in params:
        if entry.get("name") == "acquisition_mode":
            entry["limits"] = list(entry["limits"]) + [G2_MODE]
            entry["tip"] += (
                f" {G2_MODE}: shot-by-shot HBT photon-coincidence counting "
                "over 4 harmonics x 2 checkerboard areas each - see the "
                "Harmonics / Calibration / Detection / G2 groups below."
            )

    harmonics_children = []
    default_lefts = (200, 500, 800, 1100)
    for i, name in enumerate(HARMONIC_NAMES):
        harmonics_children.append(
            {
                "title": f"Harmonic {i + 1}:",
                "name": name,
                "type": "group",
                "children": [
                    {
                        "title": "Left:",
                        "name": "left",
                        "type": "int",
                        "value": default_lefts[i],
                        "min": 1,
                        "tip": "Absolute sensor column (same convention as "
                        "AOI > Left) where this harmonic's combined A+B "
                        "window begins. Placeholder default - set to the "
                        "real harmonic position.",
                    },
                    {
                        "title": "Width:",
                        "name": "width",
                        "type": "int",
                        "value": 30,
                        "min": 2,
                        "tip": "Width in columns of this harmonic's "
                        "combined A+B window.",
                    },
                ],
            }
        )

    click_count_children = [
        {
            "title": g2_analysis.area_label(i) + ":",
            "name": f"count_{i}",
            "type": "int",
            "value": 0,
            "readonly": True,
        }
        for i in range(g2_analysis.N_AREAS)
    ]
    click_prob_children = [
        {
            "title": g2_analysis.area_label(i) + ":",
            "name": f"prob_{i}",
            "type": "float",
            "value": 0.0,
            "decimals": 4,
            "readonly": True,
        }
        for i in range(g2_analysis.N_AREAS)
    ]

    params += [
        {
            "title": "Harmonics:",
            "name": "harmonics",
            "type": "group",
            "children": harmonics_children,
        },
        {
            "title": "Calibration:",
            "name": "calibration",
            "type": "group",
            "children": [
                {
                    "title": "Std threshold:",
                    "name": "std_threshold",
                    "type": "float",
                    "value": 10.0,
                    "min": 0.0,
                    "tip": "Pixels with dark standard deviation >= this are "
                    "excluded from detection. Change any time - takes "
                    "effect immediately from the stored calibration, no "
                    "need to re-run it. Resets the running g2 accumulation, "
                    "since past clicks were counted under the old rule.",
                },
                {
                    "title": "Calibration shots:",
                    "name": "calibration_shots",
                    "type": "int",
                    "value": 10000,
                    "min": 100,
                    "tip": "Number of dark frames averaged for the "
                    "per-pixel mean/std. Read once when a calibration is "
                    "started.",
                },
                {
                    "title": "Run dark calibration",
                    "name": "run_calibration",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Block all light first. Starts the acquisition "
                    "loop if needed, then accumulates 'Calibration shots' "
                    "dark frames and computes the per-pixel mean/std.",
                },
                {
                    "title": "Status:",
                    "name": "status",
                    "type": "str",
                    "value": "Not calibrated",
                    "readonly": True,
                },
            ],
        },
        {
            "title": "Detection:",
            "name": "detection",
            "type": "group",
            "children": [
                {
                    "title": "Margin:",
                    "name": "margin",
                    "type": "float",
                    "value": 10.0,
                    "tip": "Per-pixel click threshold = dark mean + this "
                    "margin. Change any time, instant, no recalibration "
                    "needed. Resets the running g2 accumulation, since past "
                    "clicks were counted under the old rule.",
                },
            ],
        },
        {
            "title": "G2 acquisition loop:",
            "name": "g2_control",
            "type": "group",
            "children": [
                {
                    "title": "Loop status:",
                    "name": "loop_status",
                    "type": "led",
                    "value": False,
                    "readonly": True,
                    "tip": "Lit whenever the persistent counting loop "
                    "(AcquisitionStart) is running. A single Grab does NOT "
                    "stop it afterwards - every additional shot only helps "
                    "g2's statistics. Use Stop loop to end it.",
                },
                {
                    "title": "Stop loop",
                    "name": "stop_loop",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Stops the persistent acquisition loop. Safe to "
                    "press any time, including when already stopped.",
                },
            ],
        },
        {
            "title": "G2 output:",
            "name": "g2_output",
            "type": "group",
            "children": [
                {
                    "title": "Update every:",
                    "name": "update_every_n_shots",
                    "type": "int",
                    "value": 500,
                    "min": 1,
                    "suffix": " shots",
                    "tip": "How often the live g2 matrix and click counts "
                    "are recomputed from the full accumulated history and "
                    "re-emitted.",
                },
                {
                    "title": "Total shots:",
                    "name": "total_shots",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                },
                {
                    "title": "Shot rate:",
                    "name": "shot_rate",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                    "suffix": " Hz",
                    "decimals": 1,
                    "tip": "Shots actually processed per second, measured "
                    "between updates - compare against the laser rep rate "
                    "to confirm no frames are being dropped.",
                },
                {
                    "title": "Click counts:",
                    "name": "click_counts",
                    "type": "group",
                    "children": click_count_children,
                },
                {
                    "title": "Click probability:",
                    "name": "click_probability",
                    "type": "group",
                    "children": click_prob_children,
                },
                {
                    "title": "Reset accumulation",
                    "name": "reset_accumulation",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Clears the click history and restarts g2 "
                    "statistics from zero, without stopping the "
                    "acquisition loop or losing calibration.",
                },
            ],
        },
    ]
    return params


class DAQ_1DViewer_MaranaXG2(DAQ_2DViewer_MaranaX):
    """
    Hanbury Brown-Twiss g2 (photon-coincidence) viewer for the Marana-X.

    4 harmonics, each split into two "single-photon-detector" areas A/B by a
    2D checkerboard (see g2_analysis.py); per shot, each area registers a
    click if any of its noise-mask-surviving pixels exceeds its own dark-
    mean-plus-margin threshold. g2 is built from the click statistics: each
    harmonic's own g2(0) from its A/B pair (avoiding the self-correlation
    bias a single detector would have), and cross-harmonic g2 from each
    harmonic's combined (A | B) click.

    Selected via Acquisition mode = "G2 Coincidence" (an explicit mode, not
    an implicit combination of existing settings the way the TA plugin uses
    Chunked+Vertical binning - there is no existing settings combination
    that would naturally mean "count coincidences"). Every other mode is
    exactly the inherited DAQ_2DViewer_MaranaX behaviour.

    Workflow: position the AOI and harmonic windows (Preview mode is
    unaffected, for alignment), switch to G2 Coincidence, block all light
    and run "Run dark calibration" once, then press Grab (Live) and leave it
    running - watch the g2 matrix and click counts refine over time, Stop
    loop when done.
    """

    live_mode_available = True
    hardware_averaging = False

    params = _build_params()

    @property
    def live_mode_available(self):
        # Preview/Chunked still run on the base class's own worker thread;
        # G2 Coincidence is a single persistent loop PyMoDAQ's grab loop
        # should not try to repeat itself - same technique already used in
        # daq_2Dviewer_MaranaXShutter.py for Shutter TA.
        try:
            return self.settings["acquisition_mode"] != G2_MODE
        except Exception:
            return True

    def ini_attributes(self):
        super().ini_attributes()
        self._last_mode = "Preview"
        self._calibration_mean = None  # (H, W) full-AOI dark mean, or None
        self._calibration_std = None  # (H, W) full-AOI dark std, or None
        self._calibration_shape = None  # (H, W) the AOI shape it was taken at
        self._area_masks = None  # (8, H, W) checkerboard geometry only
        self._valid_area_masks = None  # geometry AND noise mask
        self._thresholds = None  # (H, W) mean + margin

    # --- Mode -----------------------------------------------------------

    def _is_g2_mode(self):
        return self.settings["acquisition_mode"] == G2_MODE

    def _apply_mode_visibility(self):
        g2 = self._is_g2_mode()
        for group in ("harmonics", "calibration", "detection", "g2_control", "g2_output"):
            self.settings.child(group).show(g2)
        for name in ("chunk_size", "vertical_binning", "display"):
            self.settings.child(name).show(not g2)

    @staticmethod
    def _is_harmonic_param(param):
        parent = param.parent()
        return (
            parent is not None
            and parent.name() in HARMONIC_NAMES
            and parent.parent() is not None
            and parent.parent().name() == "harmonics"
        )

    # --- Setup ------------------------------------------------------------

    def ini_detector(self, controller=None):
        result = super().ini_detector(controller)
        if result[1]:
            self._last_mode = self.settings["acquisition_mode"]
            self._apply_mode_visibility()
            # emit=False: super().ini_detector() above already emitted one
            # dte_signal_temp (its own default placeholder image) as its
            # very last step. A second dte_signal_temp fired synchronously
            # in the same call chain - even a well-formed one - hits a
            # re-entrancy bug in PyMoDAQ's own viewer teardown/rebuild
            # (DAQ_Viewer.show_temp_data -> ... -> Viewer.remove_viewers:
            # "IndexError: pop from empty list"), reproduced against a real
            # Dashboard. Geometry is still built eagerly here so calibration
            # works immediately; only the *display* of the area map is
            # deferred to the next independent settings change (harmonics
            # edit, mode switch, threshold/margin change, or a completed
            # calibration), each of which emits on its own and is safe.
            self._rebuild_area_masks(emit=False)
        return result

    def _harmonics_from_settings(self):
        return [
            (
                int(self.settings["harmonics", name, "left"]),
                int(self.settings["harmonics", name, "width"]),
            )
            for name in HARMONIC_NAMES
        ]

    def _rebuild_area_masks(self, emit=True):
        harmonics = self._harmonics_from_settings()
        overlaps = g2_analysis.harmonics_overlap(harmonics)
        if overlaps:
            pairs = ", ".join(f"H{i + 1}/H{j + 1}" for i, j in overlaps)
            self._set_status(
                f"Harmonic windows overlap ({pairs}): a pixel can't belong "
                f"to two harmonics. Fix the Left/Width values."
            )
        self._area_masks = g2_analysis.build_area_masks(
            harmonics,
            aoi_left=self.settings["aoi", "left"],
            aoi_width=self.settings["aoi", "width"],
            aoi_height=self.settings["aoi", "height"],
        )
        self._recompute_valid_masks_and_thresholds(emit=emit)

    def _recompute_valid_masks_and_thresholds(self, emit=True):
        if self._area_masks is None:
            return

        if self._calibration_std is None:
            self._valid_area_masks = None
            self._thresholds = None
            if emit:
                self._emit_area_map()
            return

        if self._calibration_shape != self._area_masks.shape[1:]:
            # The AOI changed since calibration (e.g. stopped, reconfigured,
            # restarted without recalibrating): the stored mean/std no
            # longer matches the current frame shape.
            self._valid_area_masks = None
            self._thresholds = None
            self._set_status(
                "AOI changed since the last dark calibration; re-run "
                "'Run dark calibration'."
            )
            if emit:
                self._emit_area_map()
            return

        std_threshold = self.settings["calibration", "std_threshold"]
        margin = self.settings["detection", "margin"]
        noise_ok = g2_analysis.apply_noise_mask(self._calibration_std, std_threshold)
        valid = g2_analysis.valid_area_masks(self._area_masks, noise_ok)
        thresholds = g2_analysis.detection_thresholds(self._calibration_mean, margin)

        counts = g2_analysis.area_pixel_counts(valid)
        warnings = g2_analysis.calibration_warnings(counts)
        if warnings:
            status = "; ".join(warnings)
            self._set_status(status)
        else:
            status = "Calibrated - valid px per area: " + ", ".join(
                f"{g2_analysis.area_label(i)}={c}" for i, c in enumerate(counts)
            )
        self.settings.child("calibration", "status").setValue(status)

        self._valid_area_masks = valid
        self._thresholds = thresholds
        self._reset_accumulation()

        if self.acquisition_worker is not None:
            self.acquisition_worker.valid_area_masks = valid
            self.acquisition_worker.thresholds = thresholds

        if emit:
            self._emit_area_map()

    def _emit_area_map(self):
        if self._area_masks is None:
            return
        height, width = self._area_masks.shape[1:]
        area_map = np.zeros((height, width), dtype=np.float64)

        noise_ok = None
        if self._calibration_std is not None and self._calibration_shape == (height, width):
            noise_ok = g2_analysis.apply_noise_mask(
                self._calibration_std, self.settings["calibration", "std_threshold"]
            )
        for i in range(g2_analysis.N_AREAS):
            mask = self._area_masks[i]
            if noise_ok is not None:
                area_map[mask & noise_ok] = i + 1
                area_map[mask & ~noise_ok] = -(i + 1)
            else:
                area_map[mask] = i + 1

        y_axis = Axis(label="y", units="px", data=np.arange(height), index=0)
        x_axis = Axis(label="x", units="px", data=np.arange(width), index=1)
        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X G2",
                data=[
                    DataFromPlugins(
                        name="Area map (+n = area n valid, -n = area n excluded by noise)",
                        data=[area_map],
                        dim="Data2D",
                        axes=[y_axis, x_axis],
                    )
                ],
            )
        )

    # --- Settings -----------------------------------------------------------

    def commit_settings(self, param):
        if self.camera is None:
            return

        name = param.name()

        try:
            if self._is_harmonic_param(param):
                if self.running:
                    self._set_status("Stop the loop before changing harmonic geometry.")
                    return
                self._rebuild_area_masks()
                return

            if name == "acquisition_mode":
                new_mode = param.value()
                if self.running and (new_mode == G2_MODE) != (self._last_mode == G2_MODE):
                    self._set_status(
                        "Stop the loop before switching to or from G2 Coincidence."
                    )
                    self.settings.child("acquisition_mode").setValue(self._last_mode)
                    return
                self._last_mode = new_mode
                super().commit_settings(param)
                self._apply_mode_visibility()

            elif name == "stop_loop":
                if param.value():
                    self.stop()
                    self.settings.child("g2_control", "stop_loop").setValue(False)

            elif name == "run_calibration":
                if param.value():
                    self._start_calibration()
                    self.settings.child("calibration", "run_calibration").setValue(False)

            elif name == "reset_accumulation":
                if param.value():
                    self._reset_accumulation()
                    self.settings.child("g2_output", "reset_accumulation").setValue(False)

            elif name in ("std_threshold", "margin"):
                self._recompute_valid_masks_and_thresholds()

            elif name == "calibration_shots":
                pass  # read once when a calibration is started

            elif name == "update_every_n_shots":
                if self.acquisition_worker is not None:
                    self.acquisition_worker.update_every_n_shots = int(param.value())

            else:
                super().commit_settings(param)

        except Exception as exc:
            self._set_status(f"Setting {name!r} failed: {type(exc).__name__}: {exc}")

    # --- Acquisition loop lifecycle -------------------------------------

    def _ensure_g2_worker(self):
        """Create and start the persistent G2 worker if needed.

        Idempotent - a no-op if already running, matching the base plugin's
        _start_acquisition() guard and the TA plugin's _ensure_ta_worker().
        """
        if self.running:
            return

        self.camera.prepare()

        self.acquisition_thread = QtCore.QThread()
        self.acquisition_worker = MaranaXG2Worker(
            self.camera,
            valid_area_masks=self._valid_area_masks,
            thresholds=self._thresholds,
            update_every_n_shots=int(self.settings["g2_output", "update_every_n_shots"]),
        )
        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.calibration_progress.connect(self._calibration_progress)
        self.acquisition_worker.calibration_done.connect(self._calibration_done)
        self.acquisition_worker.g2_update.connect(self._g2_update)
        self.acquisition_worker.acquisition_error.connect(self._acquisition_error)
        self.acquisition_worker.acquisition_stopped.connect(self._acquisition_stopped)

        self.running = True
        self.settings.child("g2_control", "loop_status").setValue(True)
        self.acquisition_thread.start()

    @QtCore.Slot()
    def _acquisition_stopped(self):
        # Reached on an explicit Stop loop / DAQ_Viewer stop, and on a
        # worker crash (acquisition_error still flows into the same
        # run()/finally -> acquisition_stopped path).
        super()._acquisition_stopped()
        self.settings.child("g2_control", "loop_status").setValue(False)

    def grab_data(self, Naverage=1, **kwargs):
        if not self._is_g2_mode():
            super().grab_data(Naverage, **kwargs)
            return

        self._ensure_g2_worker()
        if self._valid_area_masks is None:
            self._set_status(
                "Loop running, but not calibrated yet: no clicks will be "
                "counted until you run Calibration > Run dark calibration."
            )

    def _start_calibration(self):
        self._ensure_g2_worker()
        n = int(self.settings["calibration", "calibration_shots"])
        self.settings.child("calibration", "status").setValue(f"Calibrating... (0 / {n})")
        self.acquisition_worker.request_calibration(n)

    def _reset_accumulation(self):
        if self.acquisition_worker is not None:
            self.acquisition_worker.request_reset()
        self.settings.child("g2_output", "total_shots").setValue(0)
        self.settings.child("g2_output", "shot_rate").setValue(0.0)
        for i in range(g2_analysis.N_AREAS):
            self.settings.child("g2_output", "click_counts", f"count_{i}").setValue(0)
            self.settings.child("g2_output", "click_probability", f"prob_{i}").setValue(0.0)

    # --- Data handling -------------------------------------------------------

    @QtCore.Slot(int, int)
    def _calibration_progress(self, done, total):
        self.settings.child("calibration", "status").setValue(f"Calibrating... ({done} / {total})")

    @QtCore.Slot(object, object)
    def _calibration_done(self, mean, std):
        self._calibration_mean = mean
        self._calibration_std = std
        self._calibration_shape = mean.shape
        self._recompute_valid_masks_and_thresholds()

    @QtCore.Slot(object, object, int, float)
    def _g2_update(self, g2_result, counts, total_shots, shot_rate_hz):
        self.settings.child("g2_output", "total_shots").setValue(total_shots)
        self.settings.child("g2_output", "shot_rate").setValue(shot_rate_hz)
        probs = counts / max(total_shots, 1)
        for i in range(g2_analysis.N_AREAS):
            self.settings.child("g2_output", "click_counts", f"count_{i}").setValue(int(counts[i]))
            self.settings.child("g2_output", "click_probability", f"prob_{i}").setValue(float(probs[i]))
        self._emit_g2_data(g2_result, counts)

    def _emit_g2_data(self, g2_result, counts):
        n = g2_analysis.N_HARMONICS
        row_axis = Axis(label="harmonic (row, H1..H4)", units="", data=np.arange(n), index=0)
        col_axis = Axis(label="harmonic (col, H1..H4)", units="", data=np.arange(n), index=1)
        g2_image = DataFromPlugins(
            name="g2 matrix",
            data=[g2_result["g2_corrected"], g2_result["g2_raw"], g2_result["g2_accidental"]],
            dim="Data2D",
            labels=["g2 corrected", "g2 raw", "g2 accidental"],
            axes=[row_axis, col_axis],
        )
        area_axis = Axis(
            label="area (H1-A,H1-B,H2-A,H2-B,H3-A,H3-B,H4-A,H4-B)",
            units="",
            data=np.arange(g2_analysis.N_AREAS),
            index=0,
        )
        counts_curve = DataFromPlugins(
            name="Click counts",
            data=[counts.astype(np.float64)],
            dim="Data1D",
            labels=["clicks"],
            axes=[area_axis],
        )
        self.dte_signal.emit(DataToExport("Marana-X G2", data=[g2_image, counts_curve]))


if __name__ == "__main__":
    main(__file__)
