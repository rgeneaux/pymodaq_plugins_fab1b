from __future__ import annotations

import threading
import time

import numpy as np
from qtpy import QtCore

from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base,
    comon_parameters,
    main,
)
from pymodaq.utils.data import Axis, DataFromPlugins
from pymodaq_data.data import DataToExport
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_fab1b.hardware.andor_sdk3 import (
    AndorSDK3Camera,
    PIXEL_ENCODING_DTYPES,
    PIXEL_ENCODING_RAW_BYTES_PER_PIXEL,
)


class MaranaXAcquisitionWorker(QtCore.QObject):
    """Qt worker running the blocking SDK3 acquisition loop on its own thread.

    Two mutually exclusive modes, both decoupling the hardware acquisition
    rate from the GUI:

    - Preview (``chunk_size`` is None): every buffer SDK3 delivers is
      counted towards the real camera FPS, but only a throttled subset
      (``display_fps_max``) is copied and forwarded to PyMoDAQ. The camera
      keeps running at full speed regardless of how fast the GUI can
      redraw. That throttle is bypassed for a finite ``frame_limit`` (a
      single/averaged grab), where every frame is needed for correct
      averaging. For interactive alignment/focus, not for data collection.

    - Chunked (``chunk_size`` is an int): acquires ``chunk_size`` frames
      back-to-back into one preallocated array with zero throttling and
      zero drops *within* the chunk (every buffer is copied and requeued
      immediately), then emits the whole burst as a single 3D dataset. This
      is what the real high-speed experiment should use: PyMoDAQ's own
      per-frame continuous-saving path can't keep up with a 5 kHz camera,
      but appending one whole chunk a few times a second is trivial for it.
      With ``single_chunk=False`` this repeats indefinitely (Live); a few
      frames may be missed between chunks while the previous one is handed
      off and a fresh buffer allocated, never within one.

    - TA-continuous (``ta_continuous=True``): for chopped pump-probe
      acquisition driven by a DAQ_Scan stepping a delay stage between
      single-chunk grabs. AcquisitionStart happens once and is never torn
      down between grabs - only on an explicit stop() - because while
      stopped, the camera (and hence this code) has zero information about
      how many laser shots/chopper cycles elapsed during a motor move.
      Every single shot is counted via a running index derived from SDK3's
      free-running per-frame hardware timestamp (configure with
      enable_metadata=True), whether its pixel data is kept or discarded,
      so the shot's parity relative to the chopper is never lost - and any
      gap (a stall, an SDK3 buffer drop) is detected from a timestamp jump
      and self-corrects the running count rather than silently mislabeling
      every shot after it. See :meth:`_run_ta_continuous`.
    """

    STATS_INTERVAL_S = 0.5

    frame_ready = QtCore.Signal(object)
    chunk_ready = QtCore.Signal(object, int)  # (n_frames, height, width) array, chunk_index
    ta_chunk_ready = QtCore.Signal(object, int)  # (chunk_size, width) array, start_shot_parity
    stats_updated = QtCore.Signal(float, float, int)  # camera_fps, displayed_fps, frames_captured
    acquisition_error = QtCore.Signal(str)
    acquisition_stopped = QtCore.Signal()

    def __init__(
        self,
        camera,
        frame_limit=None,
        display_fps_max=20.0,
        chunk_size=None,
        single_chunk=False,
        vertical_binning=False,
        ta_continuous=False,
    ):
        super().__init__()
        self.camera = camera
        self.frame_limit = frame_limit
        self.display_interval = (
            0.0 if frame_limit is not None else 1.0 / max(display_fps_max, 1e-6)
        )
        self.chunk_size = chunk_size
        self.single_chunk = single_chunk
        # This sensor has no hardware full-vertical-binning mode (only fixed
        # NxN block binning up to 8x8), so a full-height sum has to be done
        # on the host, after each frame is read out.
        self.vertical_binning = vertical_binning
        # uint32 is plenty of headroom for a uint16 source (Mono16/Mono12/
        # Mono12Packed), but summing an already-uint32 source (Mono32) needs
        # a wider accumulator to avoid overflow.
        self._sum_dtype = np.uint64 if camera.pixel_dtype == np.uint32 else np.uint32
        self._stop_event = threading.Event()

        self.ta_continuous = ta_continuous
        # Set by request_chunk() (any thread) to ask the loop to start
        # filling a real chunk instead of just counting/discarding shots.
        # Plain ints below are read cross-thread by the plugin for status
        # display only (eventually-consistent is fine; CPython's GIL makes
        # individual int reads/writes safe, no lock needed).
        self._accumulate_event = threading.Event()
        # Set by request_chunk(), read once by the worker thread itself when
        # it starts a *new* chunk (chunk is None below). Never mutated by
        # the plugin thread after that point (it waits for ta_chunk_ready
        # before requesting again), so there's no race despite crossing
        # threads: it's set-then-read, not mutated concurrently.
        self._requested_chunk_size = None
        self.shots_seen = 0
        self.shots_dropped = 0
        self.chunks_aborted = 0

    def request_chunk(self, chunk_size=None):
        """Ask the persistent TA-continuous loop to fill and emit one chunk.

        chunk_size overrides self.chunk_size for just this one request (a
        smaller calibration/background burst) without touching the
        persistent default used for real acquisition chunks.
        """
        self._requested_chunk_size = chunk_size
        self._accumulate_event.set()

    @QtCore.Slot()
    def run(self):
        # No self._stop_event.clear() here: QThread.start() is async, so if
        # stop() (which calls _stop_event.set()) lands before this method
        # actually starts executing, clearing it here would silently erase
        # that stop request the moment the thread does start, and the loop
        # below would then run forever unaware it was ever asked to stop.
        # Each worker is a fresh instance with a fresh (already-cleared)
        # Event, so there's nothing to reset anyway.
        try:
            self.camera.start()
            if self.ta_continuous:
                self._run_ta_continuous()
            elif self.chunk_size:
                self._run_chunked()
            else:
                self._run_preview()
        except Exception as exc:
            self.acquisition_error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if self.camera.acquiring:
                    self.camera.stop()
            except Exception as exc:
                self.acquisition_error.emit(f"Error stopping camera: {exc}")

            self.acquisition_stopped.emit()

    def _run_preview(self):
        captured = displayed = 0
        last_display_t = 0.0
        stats_t0 = time.perf_counter()
        captured_at_stats0 = displayed_at_stats0 = 0

        while not self._stop_event.is_set():
            result = self.camera.wait_frame(timeout_ms=100)

            if result is None:
                continue

            buffer, _size = result
            now = time.perf_counter()

            try:
                captured += 1
                if now - last_display_t >= self.display_interval:
                    # Copy before requeueing (or, for vertical binning,
                    # sum(), which always allocates a new array anyway): the
                    # SDK3 buffer is handed back to the driver immediately
                    # after, so PyMoDAQ/Qt must never retain a reference to
                    # camera-owned memory.
                    view = self.camera.frame_view(buffer)
                    if self.vertical_binning:
                        image = view.sum(axis=0, dtype=self._sum_dtype)
                    else:
                        image = np.array(view, copy=True)
                    self.frame_ready.emit(image)
                    last_display_t = now
                    displayed += 1
            finally:
                self.camera.requeue(buffer)

            if now - stats_t0 >= self.STATS_INTERVAL_S:
                elapsed = now - stats_t0
                camera_fps = (captured - captured_at_stats0) / elapsed
                displayed_fps = (displayed - displayed_at_stats0) / elapsed
                self.stats_updated.emit(camera_fps, displayed_fps, captured)
                stats_t0 = now
                captured_at_stats0 = captured
                displayed_at_stats0 = displayed

            if self.frame_limit is not None and captured >= self.frame_limit:
                break

    def _run_chunked(self):
        height, width = self.camera.height, self.camera.width
        chunk_index = 0
        total_captured = 0

        if self.vertical_binning:
            chunk_shape, chunk_dtype = (self.chunk_size, width), self._sum_dtype
        else:
            chunk_shape, chunk_dtype = (self.chunk_size, height, width), self.camera.pixel_dtype

        while not self._stop_event.is_set():
            # A fresh array per chunk: the previous one is now owned by
            # whatever is handling chunk_ready (GUI/saver), which may still
            # be reading it when the next chunk starts filling.
            chunk = np.empty(chunk_shape, dtype=chunk_dtype)
            filled = 0
            chunk_t0 = time.perf_counter()
            stats_t0 = chunk_t0

            while filled < self.chunk_size:
                if self._stop_event.is_set():
                    return  # discard the partial chunk

                result = self.camera.wait_frame(timeout_ms=100)
                if result is None:
                    continue

                buffer, _size = result
                try:
                    view = self.camera.frame_view(buffer)
                    if self.vertical_binning:
                        chunk[filled] = view.sum(axis=0, dtype=self._sum_dtype)
                    else:
                        chunk[filled] = view
                    filled += 1
                finally:
                    self.camera.requeue(buffer)

                # Without this, a chunk_size large enough to take more than
                # STATS_INTERVAL_S to fill would leave "Frames captured"/
                # "Camera FPS" frozen for the whole burst - indistinguishable
                # from a hang - and only update once the chunk completes.
                now = time.perf_counter()
                if now - stats_t0 >= self.STATS_INTERVAL_S:
                    fps = filled / (now - chunk_t0)
                    self.stats_updated.emit(fps, fps, total_captured + filled)
                    stats_t0 = now

            elapsed = time.perf_counter() - chunk_t0
            total_captured += self.chunk_size
            self.chunk_ready.emit(chunk, chunk_index)
            fps = self.chunk_size / elapsed
            self.stats_updated.emit(fps, fps, total_captured)
            chunk_index += 1

            if self.single_chunk:
                break

    def _run_ta_continuous(self):
        """Persistent chopped-pump-probe loop; see the class docstring.

        Always vertically binned (one 1D trace per shot) - unlike
        _run_chunked, this mode's whole purpose is per-shot chopper phase
        tracking, which only makes sense on a single 1D trace per shot.
        """
        width = self.camera.width
        chunk_dtype = self._sum_dtype

        period_ticks = None
        last_ts = None
        shot_index = -1  # becomes 0 on the first frame ever seen

        chunk = None
        chunk_start_shot_index = None
        current_chunk_size = None
        filled = 0

        while not self._stop_event.is_set():
            result = self.camera.wait_frame(timeout_ms=100)
            if result is None:
                continue

            buffer, size = result
            try:
                ts = self.camera.frame_timestamp(buffer, size)

                if last_ts is None:
                    shot_index += 1
                    gap = 0
                else:
                    if period_ticks is None:
                        period_ticks = (
                            self.camera.get_int("TimestampClockFrequency")
                            / self.camera.get_float("FrameRate")
                        )
                    n_elapsed = max(1, round((ts - last_ts) / period_ticks))
                    shot_index += n_elapsed
                    gap = n_elapsed - 1
                last_ts = ts
                self.shots_seen = shot_index + 1

                if gap > 0:
                    self.shots_dropped += gap
                    if chunk is not None:
                        # The gap broke the simple even/odd alternation this
                        # chunk's rows were relying on - discard it rather
                        # than risk mislabeling shots after the gap. Matches
                        # _run_chunked's existing "partial chunk is
                        # discarded, not emitted" behaviour.
                        chunk = None
                        filled = 0
                        self.chunks_aborted += 1

                if self._accumulate_event.is_set():
                    if chunk is None:
                        current_chunk_size = self._requested_chunk_size or self.chunk_size
                        chunk = np.empty((current_chunk_size, width), dtype=chunk_dtype)
                        filled = 0
                        chunk_start_shot_index = shot_index

                    view = self.camera.frame_view(buffer)
                    chunk[filled] = view.sum(axis=0, dtype=self._sum_dtype)
                    filled += 1

                    if filled >= current_chunk_size:
                        self.ta_chunk_ready.emit(chunk, chunk_start_shot_index % 2)
                        self._accumulate_event.clear()
                        chunk = None
                # else: discard - shot_index bookkeeping above already ran,
                # only the pixel data is skipped.
            finally:
                self.camera.requeue(buffer)

    def stop(self):
        self._stop_event.set()


class DAQ_2DViewer_MaranaX(DAQ_Viewer_base):
    """
    Andor Marana-X 11 viewer for PyMoDAQ 5.2.

    Acquisition uses native Andor SDK3 buffers. Mono16 is currently the
    supported image encoding. ``live_mode_available = True``: the plugin
    manages its own continuous acquisition thread (see
    :class:`MaranaXAcquisitionWorker`) rather than being called repeatedly
    by PyMoDAQ's grab loop.

    Two acquisition modes (see the "Acquisition mode" setting):

    - Preview: one frame per emission, throttled to "Max display FPS" - for
      focusing/alignment while watching the live image.
    - Chunked: bursts of "Chunk size" frames acquired back-to-back at full
      camera speed with no drops within a burst, each burst emitted as one
      3D (frame, y, x) dataset - for the real experiment, paired with
      PyMoDAQ's continuous-saving feature (Detector Settings > main
      settings > continuous saving) to append one chunk at a time to the
      h5 file. A camera cropped for 5 kHz cannot be saved frame-by-frame
      through PyMoDAQ's normal per-frame save path, but a few chunks a
      second of it is trivial.
    """

    live_mode_available = True
    hardware_averaging = False

    params = comon_parameters + [
        {
            "title": "SDK3 library:",
            "name": "sdk3_library",
            "type": "browsepath",
            "value": "C:\\Program Files\\Andor SOLIS\\atcore.dll",
        },
        {
            "title": "Camera index:",
            "name": "camera_index",
            "type": "int",
            "value": 0,
            "min": 0,
        },
        {
            "title": "SDK buffers:",
            "name": "n_buffers",
            "type": "int",
            "value": 32,
            "min": 4,
            "max": 1024,
        },
        {
            "title": "Pixel encoding:",
            "name": "pixel_encoding",
            "type": "list",
            "limits": ["Mono16"],
            "value": "Mono16",
            "tip": "Limits are populated from the camera's actual supported "
            "encodings on init. Measured on this camera (benchmark_maranax.py "
            "--pixel-encoding all): Mono12Packed is the fastest, ~7% above "
            "Mono16, from its 25% smaller data volume; Mono12 (stored "
            "unpacked, same size as Mono16) makes no difference; Mono32 is "
            "dramatically slower, to ~25% of Mono16's throughput regardless "
            "of AOI size - SDK3's own ReadoutTime/FrameRate features report "
            "the same values for all encodings and do not predict this.",
        },
        {
            "title": "Trigger mode:",
            "name": "trigger_mode",
            "type": "list",
            "limits": ["Internal"],
            "value": "Internal",
            "tip": "Limits are populated from the camera's actual supported "
            "modes on init. Internal: camera free-runs at Frame rate below. "
            "External: an external TTL into the camera's trigger input (e.g. "
            "the laser sync, at the laser repetition rate) starts each "
            "exposure, with a fixed duration set by Exposure - Frame rate "
            "becomes read-only, since the TTL sets the rate, not the camera. "
            "Requires stopping any running acquisition first.",
        },
        {
            "title": "Acquisition mode:",
            "name": "acquisition_mode",
            "type": "list",
            "limits": ["Preview", "Chunked"],
            "value": "Preview",
            "tip": "Preview: throttled single frames, for focusing/alignment. "
            "Chunked: bursts of 'Chunk size' frames at full camera speed with no "
            "drops within a burst, each emitted as one 3D dataset - for the real "
            "experiment. Naverage averages whole chunks together in this mode, "
            "not individual frames.",
        },
        {
            "title": "Chunk size:",
            "name": "chunk_size",
            "type": "int",
            "value": 1000,
            "min": 1,
            "suffix": " frames",
            "tip": "Frames per burst in Chunked mode. chunk_size * height * "
            "width * 2 bytes must comfortably fit in RAM.",
        },
        {
            "title": "AOI:",
            "name": "aoi",
            "type": "group",
            "children": [
                {"title": "Left:", "name": "left", "type": "int", "value": 1, "min": 1},
                {"title": "Bottom:", "name": "top", "type": "int", "value": 1, "min": 1},
                {"title": "Width:", "name": "width", "type": "int", "value": 2048, "min": 1},
                {"title": "Height:", "name": "height", "type": "int", "value": 2048, "min": 1},
            ],
        },
        {
            "title": "Vertical binning:",
            "name": "vertical_binning",
            "type": "bool",
            "value": False,
            "tip": "Sum all AOI rows in software after readout, collapsing each "
            "frame to a single 1D trace (Preview) or a 2D frame-vs-column image "
            "(Chunked). The camera still reads out the full AOI height - this "
            "only reduces what gets displayed/saved, e.g. to save memory during "
            "scans. This sensor has no hardware full-vertical-binning mode "
            "(AOIBinning only offers fixed blocks up to 8x8), so it's done here "
            "instead.",
        },
        {
            "title": "Safety:",
            "name": "safety",
            "type": "group",
            "children": [
                {
                    "title": "Max allocation:",
                    "name": "max_allocation_mb",
                    "type": "int",
                    "value": 4096,
                    "min": 64,
                    "suffix": " MB",
                    "tip": "Acquisition is refused rather than started if the SDK3 "
                    "buffer pool (SDK buffers x AOI) or a single chunk (Chunk size "
                    "x AOI, less with vertical binning) would exceed this. Raise it "
                    "only if you know the host has enough free RAM.",
                },
                {
                    "title": "Buffer pool (est.):",
                    "name": "buffer_pool_estimate",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                    "tip": "SDK buffers x AOI width x AOI height x 2 bytes (Mono16).",
                },
                {
                    "title": "Chunk size (est.):",
                    "name": "chunk_size_estimate",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                    "tip": "Chunk size x AOI (x AOI height too, unless vertically binned).",
                },
            ],
        },
        {
            "title": "Timing:",
            "name": "timing",
            "type": "group",
            "children": [
                {
                    "title": "Exposure:",
                    "name": "exposure",
                    "type": "float",
                    "value": 10.0,
                    "min": 0.001,
                    "suffix": " ms",
                    "tip": "Can be changed while acquiring: SDK3 applies it to the next frame.",
                },
                {
                    "title": "Frame rate:",
                    "name": "frame_rate",
                    "type": "float",
                    "value": 1.0,
                    "min": 0.001,
                    "suffix": " Hz",
                    "tip": "Target readout rate. Can be changed while acquiring.",
                },
                {
                    "title": "Set to max",
                    "name": "set_max_frame_rate",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Push the frame rate to the ceiling allowed by the current "
                    "AOI and exposure.",
                },
            ],
        },
        {
            "title": "Display:",
            "name": "display",
            "type": "group",
            "children": [
                {
                    "title": "Max display FPS:",
                    "name": "max_display_fps",
                    "type": "float",
                    "value": 20.0,
                    "min": 1.0,
                    "max": 500.0,
                    "suffix": " Hz",
                    "tip": "Caps how often frames are forwarded to the PyMoDAQ viewer "
                    "during live acquisition. The camera keeps acquiring at full "
                    "speed regardless - this only throttles GUI redraw load. Does "
                    "not apply to single-grab acquisitions.",
                },
            ],
        },
        {
            "title": "Camera information:",
            "name": "camera_info",
            "type": "group",
            "children": [
                {"title": "Model:", "name": "model", "type": "str", "value": "", "readonly": True},
                {"title": "Serial:", "name": "serial", "type": "str", "value": "", "readonly": True},
                {
                    "title": "Camera FPS:",
                    "name": "camera_fps",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                    "suffix": " Hz",
                    "decimals": 2,
                    "tip": "True buffer-delivery rate from SDK3, independent of GUI display.",
                },
                {
                    "title": "Displayed FPS:",
                    "name": "displayed_fps",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                    "suffix": " Hz",
                    "decimals": 2,
                    "tip": "Rate at which frames are actually forwarded to the viewer.",
                },
                {
                    "title": "Frames captured:",
                    "name": "frames_captured",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                },
            ],
        },
    ]

    def ini_attributes(self):
        self.camera = None
        self.acquisition_thread = None
        self.acquisition_worker = None
        self.running = False

    def _set_status(self, message):
        self.emit_status(ThreadCommand("Update_Status", [message, "log"]))

    # --- Memory safety -----------------------------------------------------
    #
    # An oversized AOI/buffer-count/chunk-size combination (typically: full
    # chip x large chunk) can ask for tens of GB in a single allocation. The
    # methods below compute exactly what a given configuration would need and
    # refuse to allocate it above a user-adjustable limit ("Safety > Max
    # allocation"), rather than letting Python/numpy try and thrash or crash
    # the host. The "(est.)" fields mirror the same numbers back into the
    # settings tree so the size is visible before Grab/Live is even pressed.

    @staticmethod
    def _format_bytes(n_bytes):
        size = float(n_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _max_allocation_bytes(self):
        return int(self.settings["safety", "max_allocation_mb"]) * 1024**2

    def _buffer_pool_bytes(self, width=None, height=None, n_buffers=None, pixel_encoding=None):
        width = int(width if width is not None else self.settings["aoi", "width"])
        height = int(height if height is not None else self.settings["aoi", "height"])
        n_buffers = int(n_buffers if n_buffers is not None else self.settings["n_buffers"])
        pixel_encoding = pixel_encoding or self.settings["pixel_encoding"]
        # This is the raw SDK3 wire size (what ImageSizeBytes/AOIStride
        # actually are), not the decoded host-side size below - only
        # Mono12Packed differs between the two.
        bytes_per_px = PIXEL_ENCODING_RAW_BYTES_PER_PIXEL.get(pixel_encoding, 2.0)
        return int(n_buffers * width * height * bytes_per_px)

    def _chunk_bytes(
        self, chunk_size=None, width=None, height=None, vertical_binning=None, pixel_encoding=None
    ):
        chunk_size = int(chunk_size if chunk_size is not None else self.settings["chunk_size"])
        width = int(width if width is not None else self.settings["aoi", "width"])
        height = int(height if height is not None else self.settings["aoi", "height"])
        vertical_binning = (
            vertical_binning
            if vertical_binning is not None
            else self.settings["vertical_binning"]
        )
        pixel_encoding = pixel_encoding or self.settings["pixel_encoding"]
        decoded_dtype = PIXEL_ENCODING_DTYPES.get(pixel_encoding, np.uint16)
        if vertical_binning:
            # One binned trace per frame; the accumulator is uint32, except
            # for an already-uint32 source (Mono32), which sums into uint64
            # to avoid overflow.
            accum_itemsize = 8 if decoded_dtype == np.uint32 else 4
            return chunk_size * width * accum_itemsize
        return chunk_size * height * width * np.dtype(decoded_dtype).itemsize

    def _update_memory_estimates(self):
        if self.camera is None:
            return
        self.settings.child("safety", "buffer_pool_estimate").setValue(
            f"{self._format_bytes(self._buffer_pool_bytes())} "
            f"({self.settings['n_buffers']} buffers)"
        )
        self.settings.child("safety", "chunk_size_estimate").setValue(
            self._format_bytes(self._chunk_bytes())
        )

    def _check_buffer_pool_size(self, width, height, n_buffers, pixel_encoding=None):
        """Refuse rather than allocate an oversized SDK3 buffer pool.

        Returns True if within the safety limit; otherwise reports a clear
        status message and returns False, leaving the camera unconfigured.
        """
        size = self._buffer_pool_bytes(
            width=width, height=height, n_buffers=n_buffers, pixel_encoding=pixel_encoding
        )
        limit = self._max_allocation_bytes()
        if size > limit:
            self._set_status(
                f"Refusing to allocate the SDK3 buffer pool: {n_buffers} buffers of "
                f"{width}x{height} would need {self._format_bytes(size)}, above the "
                f"{self._format_bytes(limit)} safety limit (Safety > Max allocation). "
                f"Reduce the AOI or buffer count, or raise the limit if you're sure."
            )
            return False
        return True

    def ini_detector(self, controller=None):
        try:
            if self.is_master:
                camera = AndorSDK3Camera(
                    index=self.settings["camera_index"],
                    dll_path=self.settings["sdk3_library"],
                    n_buffers=self.settings["n_buffers"],
                )
                camera.open()
            else:
                camera = controller

            self.camera = self.ini_controller_init(
                old_controller=controller, new_controller=camera
            )

            info = self.camera.device_info()
            self.settings.child("camera_info", "model").setValue(info.get("CameraModel", ""))
            self.settings.child("camera_info", "serial").setValue(info.get("SerialNumber", ""))

            # Restrict to encodings this wrapper actually knows how to decode
            # (PIXEL_ENCODING_DTYPES) - in practice the same set the camera
            # reports, but this stays safe if a future camera adds one we
            # haven't implemented a decoder for.
            available_encodings = [
                enc
                for enc in self.camera.enum_values("PixelEncoding")
                if enc in PIXEL_ENCODING_DTYPES
            ]
            self.settings.child("pixel_encoding").setLimits(available_encodings)
            if self.settings["pixel_encoding"] not in available_encodings:
                self.settings.child("pixel_encoding").setValue(available_encodings[0])

            available_trigger_modes = self.camera.enum_values("TriggerMode")
            self.settings.child("trigger_mode").setLimits(available_trigger_modes)
            if self.settings["trigger_mode"] not in available_trigger_modes:
                self.settings.child("trigger_mode").setValue(available_trigger_modes[0])

            # Use the camera's current AOI as the initial GUI state, bounded
            # to the physical sensor.
            self.settings.child("aoi", "left").setOpts(max=self.camera.sensor_width)
            self.settings.child("aoi", "top").setOpts(max=self.camera.sensor_height)
            self.settings.child("aoi", "width").setOpts(max=self.camera.sensor_width)
            self.settings.child("aoi", "height").setOpts(max=self.camera.sensor_height)
            self.settings.child("aoi", "left").setValue(self.camera.left)
            self.settings.child("aoi", "top").setValue(self.camera.top)
            self.settings.child("aoi", "width").setValue(self.camera.width)
            self.settings.child("aoi", "height").setValue(self.camera.height)

            if not self._check_buffer_pool_size(
                self.camera.width, self.camera.height, self.settings["n_buffers"]
            ):
                return (
                    "Camera's current AOI/buffer count exceeds the safety limit; "
                    "crop the AOI, reduce buffers, or raise Safety > Max allocation, "
                    "then reinitialize.",
                    False,
                )

            self._reconfigure_camera()
            self._update_memory_estimates()

            self._emit_initial_data()

            return "", True

        except Exception as exc:
            self._set_status(f"Marana-X initialization failed: {type(exc).__name__}: {exc}")
            return str(exc), False

    def _roi_from_settings(self):
        return (
            int(self.settings["aoi", "left"]),
            int(self.settings["aoi", "top"]),
            int(self.settings["aoi", "width"]),
            int(self.settings["aoi", "height"]),
        )

    def _reconfigure_camera(self):
        """(Re)apply ROI/encoding/exposure/frame-rate to the camera and
        refresh the exposure/frame-rate GUI bounds to match.

        Subclasses that need extra configure() arguments (e.g.
        enable_metadata) should override this rather than duplicating the
        AOI/pixel_encoding/n_buffers commit_settings branch below.
        """
        self.camera.configure(
            roi=self._roi_from_settings(),
            pixel_encoding=self.settings["pixel_encoding"],
            exposure_s=self.settings["timing", "exposure"] * 1e-3,
            frame_rate=self.settings["timing", "frame_rate"],
            trigger_mode=self.settings["trigger_mode"],
        )
        self._sync_timing_settings()

    def _sync_timing_settings(self):
        """Refresh exposure/frame-rate GUI bounds, values, and read-only
        state (which depends on TriggerMode, e.g. FrameRate is read-only
        under "External") from hardware.

        ExposureTime and FrameRate bound each other, and both bounds depend
        on the current AOI, so this must be called after any AOI, pixel
        encoding, exposure, or trigger mode change.
        """
        handle = self.camera.handle

        exp_min = self.camera.sdk.get_float_min(handle, "ExposureTime")
        exp_max = self.camera.sdk.get_float_max(handle, "ExposureTime")
        self.settings.child("timing", "exposure").setOpts(
            min=exp_min * 1e3, max=exp_max * 1e3, readonly=not self.camera.is_writable("ExposureTime")
        )
        self.settings.child("timing", "exposure").setValue(
            self.camera.get_float("ExposureTime") * 1e3
        )

        fr_min = self.camera.sdk.get_float_min(handle, "FrameRate")
        fr_max = self.camera.sdk.get_float_max(handle, "FrameRate")
        self.settings.child("timing", "frame_rate").setOpts(
            min=fr_min, max=fr_max, readonly=not self.camera.is_writable("FrameRate")
        )
        self.settings.child("timing", "frame_rate").setValue(
            self.camera.get_float("FrameRate")
        )

    @staticmethod
    def _wrap_single(data):
        """Build the DataFromPlugins for one Preview-mode emission.

        ``data`` is either a plain 2D frame, or - with vertical binning - a
        1D trace. Shared between real acquisitions and the idle placeholder
        so the two can never disagree about how to represent a given shape.
        """
        if data.ndim == 1:
            width = data.shape[0]
            return DataFromPlugins(
                name="Marana-X",
                data=[data],
                dim="Data1D",
                axes=[Axis(label="x", units="px", data=np.arange(width), index=0)],
            )
        return DataFromPlugins(name="Marana-X", data=[data], dim="Data2D")

    @staticmethod
    def _wrap_chunk(chunk, chunk_index=0):
        """Build the DataFromPlugins for one Chunked-mode emission.

        ``chunk`` is either a (frame, y, x) stack, or - with vertical
        binning - a (frame, x) image where each row is one binned trace
        (a natural kymograph/waterfall view). Shared between real
        acquisitions and the idle placeholder, as in :meth:`_wrap_single`.
        """
        if chunk.ndim == 2:
            n_frames, width = chunk.shape
            axes = [
                Axis(label="frame", units="", data=np.arange(n_frames) + chunk_index * n_frames, index=0),
                Axis(label="x", units="px", data=np.arange(width), index=1),
            ]
            return DataFromPlugins(name="Marana-X", data=[chunk], dim="Data2D", axes=axes)

        n_frames, height, width = chunk.shape
        axes = [
            Axis(label="frame", units="", data=np.arange(n_frames) + chunk_index * n_frames, index=0),
            Axis(label="y", units="px", data=np.arange(height), index=1),
            Axis(label="x", units="px", data=np.arange(width), index=2),
        ]
        # nav_indexes=(0,) marks the frame axis as navigation, so ViewerND
        # shows a frame slider driving a 2D display of axes 1/2 - it
        # defaults to () (no nav axis), which leaves PyMoDAQ with no 2D
        # display to show at all for a 3D array.
        return DataFromPlugins(name="Marana-X", data=[chunk], dim="DataND", axes=axes, nav_indexes=(0,))

    def _emit_initial_data(self):
        # Also called on an AOI/mode/binning change while idle: without
        # this, switching modes leaves the viewer showing stale data (wrong
        # type entirely, for Preview<->Chunked) until the first real
        # acquisition arrives, and PyMoDAQ has no way to know it should
        # swap viewer types before then.
        vbin = self.settings["vertical_binning"]
        height, width = self.camera.height, self.camera.width
        dtype = self.camera.pixel_dtype
        sum_dtype = np.uint64 if dtype == np.uint32 else np.uint32

        if self.settings["acquisition_mode"] == "Chunked":
            placeholder = (
                np.zeros((1, width), dtype=sum_dtype)
                if vbin
                else np.zeros((1, height, width), dtype=dtype)
            )
            dwa = self._wrap_chunk(placeholder)
        else:
            placeholder = (
                np.zeros((width,), dtype=sum_dtype)
                if vbin
                else np.zeros((height, width), dtype=dtype)
            )
            dwa = self._wrap_single(placeholder)

        self.dte_signal_temp.emit(DataToExport("Marana-X", data=[dwa]))

    def commit_settings(self, param):
        if self.camera is None:
            return

        name = param.name()

        try:
            if name in {"acquisition_mode", "vertical_binning"}:
                self._emit_initial_data()
                self._update_memory_estimates()

            elif name == "chunk_size":
                self._update_memory_estimates()

            elif name == "exposure":
                self.camera.set_float("ExposureTime", param.value() * 1e-3)
                self._sync_timing_settings()

            elif name == "frame_rate":
                self.camera.set_float("FrameRate", param.value())
                self.settings.child("timing", "frame_rate").setValue(
                    self.camera.get_float("FrameRate")
                )

            elif name == "set_max_frame_rate":
                if param.value():
                    fr_max = self.camera.sdk.get_float_max(self.camera.handle, "FrameRate")
                    self.camera.set_float("FrameRate", fr_max)
                    self.settings.child("timing", "frame_rate").setValue(
                        self.camera.get_float("FrameRate")
                    )
                    self.settings.child("timing", "set_max_frame_rate").setValue(False)

            elif name in {"left", "top", "width", "height", "pixel_encoding", "n_buffers", "trigger_mode"}:
                if self.running:
                    self._set_status(
                        "Stop the live acquisition before changing AOI, pixel "
                        "encoding, buffer count, or trigger mode."
                    )
                    return

                width = self.settings["aoi", "width"]
                height = self.settings["aoi", "height"]
                n_buffers = int(param.value()) if name == "n_buffers" else self.settings["n_buffers"]
                if not self._check_buffer_pool_size(width, height, n_buffers):
                    return

                if name == "n_buffers":
                    self.camera.n_buffers = n_buffers

                self._reconfigure_camera()
                self._update_memory_estimates()
                self._emit_initial_data()

        except Exception as exc:
            self._set_status(f"Setting {name!r} failed: {type(exc).__name__}: {exc}")

    def grab_data(self, Naverage=1, **kwargs):
        live = kwargs.get("live", False)
        vertical_binning = self.settings["vertical_binning"]

        if self.settings["acquisition_mode"] == "Chunked":
            chunk_size = int(self.settings["chunk_size"])
            size = self._chunk_bytes(chunk_size=chunk_size, vertical_binning=vertical_binning)
            limit = self._max_allocation_bytes()
            if size > limit:
                width = self.settings["aoi", "width"]
                height = self.settings["aoi", "height"]
                self._set_status(
                    f"Refusing to start: a {chunk_size}-frame chunk at {width}x{height}"
                    f"{' (vertically binned)' if vertical_binning else ''} would need "
                    f"{self._format_bytes(size)}, above the {self._format_bytes(limit)} "
                    f"safety limit (Safety > Max allocation). Reduce the chunk size or "
                    f"AOI, enable vertical binning, or raise the limit if you're sure."
                )
                return

            self._start_acquisition(
                chunk_size=chunk_size,
                single_chunk=not live,
                vertical_binning=vertical_binning,
            )
        else:
            self._start_acquisition(
                frame_limit=None if live else max(1, int(Naverage)),
                vertical_binning=vertical_binning,
            )

    def _start_acquisition(
        self, frame_limit=None, chunk_size=None, single_chunk=False, vertical_binning=False
    ):
        if self.running:
            return

        self.camera.prepare()

        self.acquisition_thread = QtCore.QThread()
        self.acquisition_worker = MaranaXAcquisitionWorker(
            self.camera,
            frame_limit=frame_limit,
            display_fps_max=self.settings["display", "max_display_fps"],
            chunk_size=chunk_size,
            single_chunk=single_chunk,
            vertical_binning=vertical_binning,
        )
        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.frame_ready.connect(self._frame_received)
        self.acquisition_worker.chunk_ready.connect(self._chunk_received)
        self.acquisition_worker.stats_updated.connect(self._stats_updated)
        self.acquisition_worker.acquisition_error.connect(self._acquisition_error)
        self.acquisition_worker.acquisition_stopped.connect(self._acquisition_stopped)

        self.settings.child("camera_info", "frames_captured").setValue(0)
        self.running = True

        self.acquisition_thread.start()

    @QtCore.Slot(object)
    def _frame_received(self, image):
        self.dte_signal.emit(DataToExport("Marana-X", data=[self._wrap_single(image)]))

    @QtCore.Slot(object, int)
    def _chunk_received(self, chunk, chunk_index):
        self.dte_signal.emit(
            DataToExport("Marana-X", data=[self._wrap_chunk(chunk, chunk_index)])
        )

    @QtCore.Slot(float, float, int)
    def _stats_updated(self, camera_fps, displayed_fps, frames_captured):
        self.settings.child("camera_info", "camera_fps").setValue(camera_fps)
        self.settings.child("camera_info", "displayed_fps").setValue(displayed_fps)
        self.settings.child("camera_info", "frames_captured").setValue(frames_captured)

    @QtCore.Slot(str)
    def _acquisition_error(self, message):
        self._set_status(message)

    @QtCore.Slot()
    def _acquisition_stopped(self):
        # Reached whether the worker stopped because of an explicit stop()
        # or because a finite frame_limit (single/averaged grab) completed
        # on its own. Either way the QThread's own event loop is still
        # running at this point - our worker's run() slot returned, but
        # nothing has told the thread to quit() yet - so it must be torn
        # down here too, not just in stop(). Leaving it running and letting
        # the next _start_acquisition() overwrite self.acquisition_thread
        # would garbage-collect a QThread object while its underlying
        # thread is still alive, which crashes the process outright.
        self.running = False
        self._teardown_worker_thread()

    def _teardown_worker_thread(self):
        if self.acquisition_thread is not None:
            self.acquisition_thread.quit()
            if not self.acquisition_thread.wait(3000):
                self._set_status("Acquisition thread did not stop within 3 s")

        self.acquisition_worker = None
        self.acquisition_thread = None

    def stop(self):
        if not self.running:
            return

        if self.acquisition_worker is not None:
            self.acquisition_worker.stop()

        self._teardown_worker_thread()
        self.running = False

        try:
            if self.camera is not None and self.camera.acquiring:
                self.camera.stop()
        except Exception as exc:
            self._set_status(f"Error stopping Marana-X: {exc}")

    def close(self):
        self.stop()

        if self.camera is not None:
            try:
                if self.is_master:
                    self.camera.close()
            finally:
                self.camera = None


if __name__ == "__main__":
    main(__file__)
