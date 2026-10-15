from __future__ import annotations

import threading
import time

import numpy as np
from qtpy import QtCore

from pymodaq.control_modules.viewer_utility_classes import main
from pymodaq.utils.data import Axis, DataFromPlugins
from pymodaq_data.data import DataToExport

from pymodaq_plugins_fab1b.chopper_shots import SHOTS_NAME, STATE_NAME, TIMES_NAME
from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaX import (
    DAQ_2DViewer_MaranaX,
    MaranaXAcquisitionWorker,
)

# Below this ON-vs-OFF contrast significance (in standard errors) the fitted
# chopper phase is meaningless, so the chunk is discarded.
MIN_FIT_Z = 5.0


class MaranaXChopperWorker(MaranaXAcquisitionWorker):
    """Persistent single-shot loop that tags every kept shot with its time.

    Same persistent AcquisitionStart as the base TA-continuous loop, but no
    shot counting, parity or drop bookkeeping: the pump state of a shot is
    decided afterwards from its hardware timestamp (see
    :class:`DAQ_1DViewer_MaranaXChopperTA`), so a dropped frame or a stall
    simply shows up as a gap in the timestamps and costs nothing.
    """

    # (n_shots, width) vertically-binned shots, per-shot camera hardware timestamps (s, arbitrary
    # origin) and per-shot host arrival times (s, epoch)
    chopper_chunk_ready = QtCore.Signal(object, object, object)
    # (shots acquired so far, chunk size); (0, 0) = no chunk in progress
    chunk_progress = QtCore.Signal(int, int)

    PROGRESS_INTERVAL_S = 0.25

    def __init__(self, camera, chunk_size=None):
        super().__init__(
            camera,
            chunk_size=chunk_size,
            vertical_binning=True,
            ta_continuous=True,
        )
        self._finish_event = threading.Event()

    def finish_chunk(self):
        """Emit the shots acquired so far as the current chunk, right now."""
        self._finish_event.set()

    def _run_ta_continuous(self):
        width = self.camera.width
        ts_freq = float(self.camera.get_int("TimestampClockFrequency"))

        chunk = None
        chunk_ts = None
        chunk_host = None
        filled = 0
        chunk_size = 0
        last_progress = 0.0

        while not self._stop_event.is_set():
            # Checked before waiting for a frame so a finish request is
            # honoured even when no frame is arriving (e.g. no trigger).
            if self._finish_event.is_set():
                self._finish_event.clear()
                if self._accumulate_event.is_set():
                    if chunk is not None and filled > 0:
                        self.chopper_chunk_ready.emit(
                            chunk[:filled], chunk_ts[:filled], chunk_host[:filled]
                        )
                    else:
                        self.chunk_progress.emit(0, 0)
                    self._accumulate_event.clear()
                    chunk = None
                    chunk_ts = None
                    chunk_host = None

            result = self.camera.wait_frame(timeout_ms=100)
            if result is None:
                continue

            t_arrival = time.time()
            buffer, size = result
            try:
                self.shots_seen += 1

                # Not accumulating: the frame is just drained so the camera
                # keeps free-running between chunks (e.g. during a delay-stage
                # move) and the buffers never back up.
                if not self._accumulate_event.is_set():
                    continue

                if chunk is None:
                    chunk_size = self._requested_chunk_size or self.chunk_size
                    chunk = np.empty((chunk_size, width), dtype=self._sum_dtype)
                    chunk_ts = np.empty(chunk_size, dtype=np.float64)
                    chunk_host = np.empty(chunk_size, dtype=np.float64)
                    filled = 0

                view = self.camera.frame_view(buffer)
                chunk[filled] = view.sum(axis=0, dtype=self._sum_dtype)
                chunk_ts[filled] = self.camera.frame_timestamp(buffer, size) / ts_freq
                chunk_host[filled] = t_arrival
                filled += 1

                now = time.monotonic()
                if now - last_progress >= self.PROGRESS_INTERVAL_S:
                    last_progress = now
                    self.chunk_progress.emit(filled, chunk_size)

                if filled >= chunk_size:
                    self.chopper_chunk_ready.emit(chunk, chunk_ts, chunk_host)
                    self._accumulate_event.clear()
                    chunk = None
                    chunk_ts = None
                    chunk_host = None
            finally:
                self.camera.requeue(buffer)


class DAQ_1DViewer_MaranaXChopperTA(DAQ_2DViewer_MaranaX):
    """
    Chopped pump-probe (transient absorption/reflectivity) viewer for the
    Andor Marana-X, for a *slow* chopper: its period spans many laser shots
    (e.g. 10 Hz on a 1 kHz laser: 50 shots pumped, then 50 not).

    Shot parity says nothing about the pump state in that regime, so every
    single-shot frame is classified from its hardware timestamp and a chopper
    phase fitted on the data itself. For each chunk, the ON-window start is
    scanned over one chopper period to maximise the on/off contrast of the
    mean signal over the Reference pixels (where the pump leaves a signature,
    typically its scatter); shots within the Edge guard of a chopper edge are
    dropped. The chopper only has to run at the stated frequency: it need not
    be phase-locked to the camera trigger, and a dropped frame costs nothing.
    A chunk with no significant modulation is discarded with a status message
    rather than mislabelled.

    Use together with true single-shot acquisition: external trigger on the
    laser, exposure short enough for the camera to accept every trigger, and
    the trigger delayed (hardware) so the exposed rows catch the light.

    Same engagement rule as the other Marana-X TA plugin: the TA loop runs
    when Acquisition mode = Chunked and Vertical binning is on; any other
    combination falls through to the base plugin's plain camera output
    (Preview, to position the AOI).

    Workflow: position the AOI in Preview, switch on Chunked + Vertical
    binning, run "Check chopper phase" to see the fitted on/off traces, block
    the probe (pump on) and run "Measure background", then start the DAQ_Scan.
    """

    live_mode_available = True
    hardware_averaging = False

    params = DAQ_2DViewer_MaranaX.params + [
        {
            "title": "TA acquisition loop:",
            "name": "ta_control",
            "type": "group",
            "children": [
                {
                    "title": "Loop status:",
                    "name": "loop_status",
                    "type": "led",
                    "value": False,
                    "readonly": True,
                    "tip": "Lit whenever the persistent acquisition loop "
                    "(AcquisitionStart) is running. A single Grab does NOT "
                    "stop it afterwards: the camera keeps free-running "
                    "(pixel data discarded) between chunks, e.g. during "
                    "delay-stage moves. Use Stop loop to end it.",
                },
                {
                    "title": "Stop loop",
                    "name": "stop_loop",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Stops the persistent acquisition loop "
                    "(AcquisitionStop). Safe to press any time.",
                },
            ],
        },
        {
            "title": "Chopper:",
            "name": "chopper",
            "type": "group",
            "children": [
                {
                    "title": "Frequency:",
                    "name": "freq",
                    "type": "float",
                    "value": 10.0,
                    "min": 0.01,
                    "suffix": " Hz",
                    "tip": "Full on+off cycle frequency; 50% duty cycle assumed.",
                },
                {
                    "title": "Edge guard:",
                    "name": "edge_guard",
                    "type": "float",
                    "value": 8.0,
                    "min": 0.0,
                    "suffix": " ms",
                    "tip": "Shots closer than this to a chopper edge (either "
                    "side) are discarded: the blade is partly crossing the "
                    "beam there. The measured edge is roughly 10 ms wide on "
                    "a 10 Hz chopper.",
                },
                {
                    "title": "Reference pixels:",
                    "name": "ref_pixels",
                    "type": "group",
                    "children": [
                        {"title": "From:", "name": "start", "type": "int", "value": 200, "min": 0},
                        {"title": "To:", "name": "stop", "type": "int", "value": 300, "min": 1},
                        {
                            "title": "Brighter when pump:",
                            "name": "polarity",
                            "type": "list",
                            "limits": ["ON", "OFF"],
                            "value": "ON",
                            "tip": "In which pump state the reference region "
                            "is the brighter one. This fixes which half of the "
                            "chopper cycle is called ON.",
                        },
                    ],
                },
                {
                    "title": "Output single shots:",
                    "name": "output_shots",
                    "type": "bool",
                    "value": False,
                    "tip": "Also output, with each chunk, every classified shot "
                    "(background-subtracted) with its own time and pump state, "
                    "in addition to the chunk-averaged observable. The Delay "
                    "Binner extension switches it on by itself: it needs each "
                    "shot to bin it at its own delay. Leave it off otherwise "
                    "(the shot matrix is large and shows up in the viewer and "
                    "in saved data).",
                },
                {
                    "title": "Chunk progress:",
                    "name": "progress",
                    "type": "str",
                    "value": "-",
                    "readonly": True,
                    "tip": "Shots acquired in the chunk being filled, with an "
                    "estimated time left. A large Chunk size at a low shot "
                    "rate can take minutes.",
                },
                {
                    "title": "Finish chunk now",
                    "name": "finish_chunk",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Stop waiting: process the shots acquired so far "
                    "as this chunk (it must still span at least two chopper "
                    "periods). The loop keeps running. To end it, use the "
                    "viewer's stop button or Stop loop.",
                },
                {
                    "title": "Check chopper phase",
                    "name": "check_phase",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Acquire one chunk and display the fitted pump-ON "
                    "and pump-OFF mean traces, to confirm the classification "
                    "before measuring. Requires Acquisition mode = Chunked "
                    "and Vertical binning on.",
                },
                {
                    "title": "Last fit:",
                    "name": "fit_result",
                    "type": "str",
                    "value": "-",
                    "readonly": True,
                    "tip": "Pump-ON window start within the cycle, contrast "
                    "significance (z) and shots kept ON/OFF. A small z means "
                    "the reference pixels see no chopper modulation.",
                },
                {
                    "title": "Shots seen:",
                    "name": "shots_seen",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                    "tip": "Frames delivered by the camera since the "
                    "acquisition loop started.",
                },
            ],
        },
        {
            "title": "Background:",
            "name": "background",
            "type": "group",
            "children": [
                {
                    "title": "Measure background",
                    "name": "measure_background",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Block the probe beam, keep the pump on (the "
                    "chopper phase is fitted on the pump signature). "
                    "Acquires one chunk, splits it by fitted pump state and "
                    "stores the two averaged traces as the background.",
                },
                {
                    "title": "Clear background",
                    "name": "clear_background",
                    "type": "bool_push",
                    "value": False,
                },
                {
                    "title": "Status:",
                    "name": "background_status",
                    "type": "str",
                    "value": "Not measured",
                    "readonly": True,
                },
            ],
        },
        {
            "title": "Observable:",
            "name": "observable",
            "type": "list",
            "limits": ["ΔOD", "dR/R", "Raw on/off"],
            "value": "ΔOD",
            "tip": "ΔOD = -log10(on/off); dR/R = (on-off)/off; Raw on/off "
            "shows just the two background-subtracted averages. The on/off "
            "pair is always shown alongside ΔOD/dR/R too, as a diagnostic.",
        },
    ]

    def ini_attributes(self):
        super().ini_attributes()
        self._pending_action = None  # None | "check" | "background"
        self._chunk_t0 = time.monotonic()
        # Continuous grab: each processed chunk requests the next one.
        self._live = False
        self._background_on = None
        self._background_off = None
        self._background_tag = None

    # --- Setup / reconfigure ------------------------------------------------

    def _is_ta_configured(self):
        return self.settings["acquisition_mode"] == "Chunked" and self.settings["vertical_binning"]

    def _reconfigure_camera(self):
        # Metadata is always on: the hardware timestamp is what classifies
        # each shot's pump state.
        self.camera.configure(
            roi=self._roi_from_settings(),
            pixel_encoding=self.settings["pixel_encoding"],
            exposure_s=self.settings["timing", "exposure"] * 1e-3,
            frame_rate=self.settings["timing", "frame_rate"],
            trigger_mode=self.settings["trigger_mode"],
            gain_mode=self.settings["gain_mode"],
            enable_metadata=True,
        )
        self._sync_timing_settings()

    def commit_settings(self, param):
        if self.camera is None:
            return

        name = param.name()

        try:
            if name == "stop_loop":
                if param.value():
                    self.stop()
                    self.settings.child("ta_control", "stop_loop").setValue(False)

            elif name == "finish_chunk":
                if param.value():
                    if self.running and self.acquisition_worker is not None:
                        self.acquisition_worker.finish_chunk()
                    self.settings.child("chopper", "finish_chunk").setValue(False)

            elif name == "check_phase":
                if param.value():
                    self._start_phase_check()
                    self.settings.child("chopper", "check_phase").setValue(False)

            elif name == "measure_background":
                if param.value():
                    self._start_background_measurement()
                    self.settings.child("background", "measure_background").setValue(False)

            elif name == "clear_background":
                if param.value():
                    self._clear_background()
                    self.settings.child("background", "clear_background").setValue(False)

            else:
                super().commit_settings(param)

        except Exception as exc:
            self._set_status(f"Setting {name!r} failed: {type(exc).__name__}: {exc}")

    # --- Continuous worker lifecycle ----------------------------------------

    def _ensure_ta_worker(self):
        """Create and start the persistent loop if needed (idempotent)."""
        if self.running:
            return
        if not self._is_ta_configured():
            raise RuntimeError(
                "Set Acquisition mode to Chunked and switch on Vertical "
                "binning first."
            )

        self.camera.prepare()

        self.acquisition_thread = QtCore.QThread()
        self.acquisition_worker = MaranaXChopperWorker(
            self.camera, chunk_size=int(self.settings["chunk_size"])
        )
        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.chopper_chunk_ready.connect(self._chopper_chunk_received)
        self.acquisition_worker.chunk_progress.connect(self._chunk_progress)
        self.acquisition_worker.acquisition_error.connect(self._acquisition_error)
        self.acquisition_worker.acquisition_stopped.connect(self._acquisition_stopped)

        self.running = True
        self.settings.child("ta_control", "loop_status").setValue(True)
        self.acquisition_thread.start()

    @QtCore.Slot()
    def _acquisition_stopped(self):
        # Also reached on a worker crash: either way the loop is no longer
        # draining frames, so a leftover pending action would otherwise be
        # wrongly applied to the first plain chunk after the next start.
        super()._acquisition_stopped()
        self._pending_action = None
        self._live = False
        self.settings.child("ta_control", "loop_status").setValue(False)

    def grab_data(self, Naverage=1, **kwargs):
        if not self._is_ta_configured():
            super().grab_data(Naverage, **kwargs)
            return

        chunk_size = int(self.settings["chunk_size"])
        size = self._chunk_bytes(chunk_size=chunk_size, vertical_binning=True)
        limit = self._max_allocation_bytes()
        if size > limit:
            self._set_status(
                f"Refusing to start: a {chunk_size}-shot chunk would need "
                f"{self._format_bytes(size)}, above the {self._format_bytes(limit)} "
                f"safety limit (Safety > Max allocation)."
            )
            return

        self._pending_action = None
        self._live = bool(kwargs.get("live", False))
        self._ensure_ta_worker()
        self._request_chunk(chunk_size)

    def stop(self):
        self._live = False
        super().stop()

    def _request_chunk(self, chunk_size):
        self._chunk_t0 = time.monotonic()
        self.acquisition_worker.request_chunk(chunk_size=chunk_size)

    def _request_next_chunk(self, retry):
        """Re-request a chunk: always after a rejected one (so a scan point
        keeps trying instead of waiting forever for data that never comes),
        after a good one only in a continuous grab."""
        if (self._live or retry) and self.running and self.acquisition_worker is not None:
            self._request_chunk(int(self.settings["chunk_size"]))

    @QtCore.Slot(int, int)
    def _chunk_progress(self, done, total):
        if total <= 0:
            self.settings.child("chopper", "progress").setValue("-")
            self._set_status("Nothing to finish: no shots acquired yet (is the camera triggered?).")
            return
        elapsed = time.monotonic() - self._chunk_t0
        text = f"{done} / {total} shots ({100.0 * done / total:.0f} %)"
        if done > 0 and elapsed > 0.5:
            left = elapsed * (total - done) / done
            text += f", ~{left:.0f} s left" if left < 120 else f", ~{left / 60:.1f} min left"
        self.settings.child("chopper", "progress").setValue(text)

    def _start_phase_check(self):
        self._ensure_ta_worker()
        self._pending_action = "check"
        self._request_chunk(int(self.settings["chunk_size"]))

    def _start_background_measurement(self):
        self._ensure_ta_worker()
        self._pending_action = "background"
        self._request_chunk(int(self.settings["chunk_size"]))

    def _clear_background(self):
        self._background_on = None
        self._background_off = None
        self._background_tag = None
        self.settings.child("background", "background_status").setValue("Not measured")

    # --- Chopper phase classification ---------------------------------------

    @staticmethod
    def _chopper_masks(t_s, phase_s, period_s, guard_s):
        """(on, off) shot masks for a 50%-duty chopper.

        The ON window is [phase, phase + period/2) modulo the period; shots
        within ``guard_s`` of either edge belong to neither mask.
        """
        u = np.mod(t_s - phase_s, period_s)
        half = period_s / 2.0
        edge_dist = np.minimum.reduce([u, np.abs(u - half), period_s - u])
        keep = edge_dist >= guard_s
        return (u < half) & keep, (u >= half) & keep

    @classmethod
    def _fit_chopper_phase(cls, ref, t_s, period_s, guard_s, bright_is_on=True, step_s=0.5e-3):
        """Fit the chopper phase from a per-shot reference signal.

        Scans the ON-window start over one period and keeps the phase giving
        the largest signed (on - off) contrast of ``ref`` (sign flipped when
        the reference is brighter with the pump OFF). Returns
        ``(phase_s, z)``; z is that contrast in standard errors, so a small
        value means the reference sees no chopper modulation at all.
        """
        sign = 1.0 if bright_is_on else -1.0
        best_phase, best_z = 0.0, -np.inf
        for phase in np.arange(0.0, period_s, step_s):
            on, off = cls._chopper_masks(t_s, phase, period_s, guard_s)
            if on.sum() < 2 or off.sum() < 2:
                continue
            r_on, r_off = ref[on], ref[off]
            se = np.sqrt(r_on.var() / r_on.size + r_off.var() / r_off.size)
            if se == 0:
                continue
            z = sign * (r_on.mean() - r_off.mean()) / se
            if z > best_z:
                best_phase, best_z = phase, z
        return best_phase, best_z

    def _classify(self, chunk, shot_times):
        """Classify the shots of a chunk as pump-on / pump-off by fitted chopper phase.

        Returns ``(on, off)`` boolean shot masks (shots in the edge guard are
        in neither), or ``None`` when the chunk can't be classified (the
        status line says why).
        """
        period_s = 1.0 / float(self.settings["chopper", "freq"])
        guard_s = self.settings["chopper", "edge_guard"] * 1e-3
        a = int(self.settings["chopper", "ref_pixels", "start"])
        b = min(int(self.settings["chopper", "ref_pixels", "stop"]), chunk.shape[1])
        if b <= a:
            self._set_status("Reference pixels: 'To' must be greater than 'From' (and inside the AOI).")
            return None

        t_s = np.asarray(shot_times, dtype=np.float64)
        t_s = t_s - t_s[0]
        if t_s[-1] < 2 * period_s:
            self._set_status(
                f"Chunk spans only {t_s[-1] * 1e3:.0f} ms, less than two chopper "
                f"periods ({2 * period_s * 1e3:.0f} ms): increase Chunk size."
            )
            return None

        ref = chunk[:, a:b].astype(np.float64).mean(axis=1)
        bright_is_on = self.settings["chopper", "ref_pixels", "polarity"] == "ON"
        phase_s, z = self._fit_chopper_phase(ref, t_s, period_s, guard_s, bright_is_on)
        if not np.isfinite(z):
            self._set_status(
                f"Too few shots ({len(t_s)}) to fit the chopper phase: increase "
                f"Chunk size or the shot rate. Chunk discarded."
            )
            return None
        on, off = self._chopper_masks(t_s, phase_s, period_s, guard_s)
        self.settings.child("chopper", "fit_result").setValue(
            f"start {phase_s * 1e3:.1f} ms, z={z:.1f}, {int(on.sum())} on / {int(off.sum())} off"
        )
        if z < MIN_FIT_Z:
            self._set_status(
                f"No chopper modulation found on pixels {a}-{b} (z={z:.1f} < "
                f"{MIN_FIT_Z:.0f}): pump blocked? chopper frequency wrong? "
                f"reference pixels off the pump signature? Chunk discarded."
            )
            return None
        return on, off

    # --- Data handling -------------------------------------------------------

    @QtCore.Slot(object, object, object)
    def _chopper_chunk_received(self, chunk, shot_times, host_times):
        self.settings.child("chopper", "shots_seen").setValue(self.acquisition_worker.shots_seen)
        self.settings.child("chopper", "progress").setValue(f"done ({len(chunk)} shots)")
        accepted = False
        try:
            accepted = self._process_chunk(chunk, shot_times, host_times)
        finally:
            self._request_next_chunk(retry=not accepted)

    def _process_chunk(self, chunk, shot_times, host_times):
        """Classify and emit one chunk; False if it was rejected."""
        pending, self._pending_action = self._pending_action, None

        masks = self._classify(chunk, shot_times)
        if masks is None:
            return False
        on, off = masks
        on_shots, off_shots = chunk[on], chunk[off]
        mean_on = on_shots.mean(axis=0)
        mean_off = off_shots.mean(axis=0)

        if pending == "check":
            self._emit_check(mean_on, mean_off, len(on_shots), len(off_shots))
        elif pending == "background":
            self._store_background(mean_on, mean_off, len(on_shots), len(off_shots))
        else:
            shots = (chunk, shot_times, host_times, on, off) if self.settings["chopper", "output_shots"] else None
            self._emit_observable(mean_on, mean_off, shots)
        return True

    def _emit_check(self, mean_on, mean_off, n_on, n_off):
        x_axis = Axis(label="x", units="px", data=np.arange(mean_on.shape[0]), index=0)
        self._set_status(
            f"Chopper phase fitted: {n_on} pump-on and {n_off} pump-off shots "
            f"kept. Check the two traces."
        )
        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X chopper TA",
                data=[
                    DataFromPlugins(
                        name="Pump on/off (raw)",
                        data=[mean_on, mean_off],
                        dim="Data1D",
                        labels=["Pump on", "Pump off"],
                        axes=[x_axis],
                    ),
                ],
            )
        )

    def _background_tag_current(self):
        return (
            self.settings["aoi", "width"],
            self.settings["aoi", "height"],
            self.settings["timing", "exposure"],
            self.settings["pixel_encoding"],
        )

    def _store_background(self, mean_on, mean_off, n_on, n_off):
        self._background_on = mean_on
        self._background_off = mean_off
        self._background_tag = self._background_tag_current()
        stamp = time.strftime("%H:%M:%S")
        self.settings.child("background", "background_status").setValue(
            f"Measured at {stamp} ({n_on} on / {n_off} off shots)"
        )
        self._set_status(
            f"Background measured at {stamp} ({n_on} on / {n_off} off shots); "
            f"it is now subtracted from the data."
        )
        # Show the stored traces so the measurement is visible on the plot too.
        x_axis = Axis(label="x", units="px", data=np.arange(mean_on.shape[0]), index=0)
        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X chopper TA",
                data=[
                    DataFromPlugins(
                        name="Background (measured)",
                        data=[mean_on, mean_off],
                        dim="Data1D",
                        labels=["Background pump on", "Background pump off"],
                        axes=[x_axis],
                    ),
                ],
            )
        )

    def _background_stale(self):
        return (
            self._background_tag is not None
            and self._background_tag != self._background_tag_current()
        )

    def _active_background(self):
        """(background_on, background_off) to subtract, or None."""
        if self._background_on is not None and not self._background_stale():
            return self._background_on, self._background_off
        if self._background_tag is not None:
            self._set_status(
                "Stored background no longer matches the current AOI/exposure/"
                "encoding; not subtracting it. Re-run 'Measure background'."
            )
        return None

    def _shot_channels(self, chunk, cam_times, host_times, on, off, background):
        """Single-shot channels (see :mod:`pymodaq_plugins_fab1b.chopper_shots`)."""
        keep = on | off
        is_on = on[keep]
        spectra = chunk[keep].astype(np.float32)
        if background is not None:
            spectra -= np.where(is_on[:, None], background[0], background[1]).astype(np.float32)
        # Camera clock -> host wall clock. The offset is the median over the
        # chunk of (host arrival - camera timestamp): it rejects the per-frame
        # jitter of the host side, and is biased only by the constant
        # exposure + readout + transfer latency (milliseconds).
        offset = float(np.median(np.asarray(host_times) - np.asarray(cam_times)))
        times = np.asarray(cam_times)[keep] + offset
        n_shots, width = spectra.shape
        shot_axis = Axis(label="shot", units="", data=np.arange(n_shots), index=0)
        x_axis = Axis(label="x", units="px", data=np.arange(width), index=1)
        shot_index_axis = Axis(label="shot", units="", data=np.arange(n_shots), index=0)
        return [
            DataFromPlugins(
                name=SHOTS_NAME, data=[spectra], dim="Data2D", labels=["shots"], axes=[shot_axis, x_axis]
            ),
            DataFromPlugins(
                name=TIMES_NAME, data=[times], dim="Data1D", labels=["time"], axes=[shot_index_axis]
            ),
            DataFromPlugins(
                name=STATE_NAME,
                data=[is_on.astype(np.float64)],
                dim="Data1D",
                labels=["state"],
                axes=[shot_index_axis],
            ),
        ]

    def _emit_observable(self, mean_on, mean_off, shots=None):
        background = self._active_background()
        if background is not None:
            on_corr = mean_on - background[0]
            off_corr = mean_off - background[1]
        else:
            on_corr = mean_on
            off_corr = mean_off

        x_axis = Axis(label="x", units="px", data=np.arange(mean_on.shape[0]), index=0)
        observable = self.settings["observable"]

        on_off_channel = DataFromPlugins(
            name="Pump on/off",
            data=[on_corr, off_corr],
            dim="Data1D",
            labels=["Pump on", "Pump off"],
            axes=[x_axis],
        )

        if observable == "Raw on/off":
            data = [on_off_channel]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                if observable == "ΔOD":
                    result = -np.log10(on_corr / off_corr)
                    name = "ΔOD"
                else:
                    result = (on_corr - off_corr) / off_corr
                    name = "dR/R"
            data = [
                DataFromPlugins(name=name, data=[result], dim="Data1D", axes=[x_axis]),
                on_off_channel,
            ]

        if shots is not None:
            data = data + self._shot_channels(*shots, background)

        self.dte_signal.emit(DataToExport("Marana-X chopper TA", data=data))


if __name__ == "__main__":
    main(__file__)
