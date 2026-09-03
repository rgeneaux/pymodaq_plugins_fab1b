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
from pymodaq.utils.data import DataFromPlugins
from pymodaq_data.data import DataToExport
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_maranax.hardware.andor_sdk3 import (
    AndorSDK3Camera,
)


class MaranaXAcquisitionWorker(QtCore.QObject):
    """Qt worker whose only blocking operation is native SDK3 AT_WaitBuffer."""

    frame_ready = QtCore.Signal(object)
    acquisition_error = QtCore.Signal(str)
    acquisition_stopped = QtCore.Signal()

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._stop_event = threading.Event()

    @QtCore.Slot()
    def run(self):
        self._stop_event.clear()

        try:
            self.camera.start()

            while not self._stop_event.is_set():
                result = self.camera.wait_frame(timeout_ms=100)

                if result is None:
                    continue

                buffer, _size = result

                try:
                    # The returned ndarray is a view on the SDK3 buffer.
                    # Copy it before requeueing the buffer: PyMoDAQ/Qt must
                    # never retain a reference to camera-owned memory.
                    image = np.array(
                        self.camera.frame_view(buffer),
                        copy=True,
                    )
                    self.frame_ready.emit(image)
                finally:
                    self.camera.requeue(buffer)

        except Exception as exc:
            self.acquisition_error.emit(
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            try:
                if self.camera.acquiring:
                    self.camera.stop()
            except Exception as exc:
                self.acquisition_error.emit(
                    f"Error stopping camera: {exc}"
                )

            self.acquisition_stopped.emit()

    def stop(self):
        self._stop_event.set()


class DAQ_2DViewer_MaranaX(DAQ_Viewer_base):
    """
    Andor Marana-X 11 viewer for PyMoDAQ 5.2.

    Acquisition uses native Andor SDK3 buffers. Mono16 is currently the
    supported image encoding.
    """

    live_mode_available = True
    hardware_averaging = False

    params = comon_parameters + [
        {
            "title": "SDK3 library:",
            "name": "sdk3_library",
            "type": "browsepath",
            "value": "atcore.dll",
        },
        {
            "title": "Camera index:",
            "name": "camera_index",
            "type": "int",
            "value": 0,
            "min": 0,
        },
        {
            "title": "Exposure:",
            "name": "exposure",
            "type": "float",
            "value": 10.0,
            "min": 0.001,
            "suffix": " ms",
        },
        {
            "title": "Pixel encoding:",
            "name": "pixel_encoding",
            "type": "list",
            "limits": ["Mono16"],
            "value": "Mono16",
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
            "title": "AOI:",
            "name": "aoi",
            "type": "group",
            "children": [
                {
                    "title": "Left:",
                    "name": "left",
                    "type": "int",
                    "value": 1,
                    "min": 1,
                },
                {
                    "title": "Top:",
                    "name": "top",
                    "type": "int",
                    "value": 1,
                    "min": 1,
                },
                {
                    "title": "Width:",
                    "name": "width",
                    "type": "int",
                    "value": 2048,
                    "min": 1,
                },
                {
                    "title": "Height:",
                    "name": "height",
                    "type": "int",
                    "value": 2048,
                    "min": 1,
                },
            ],
        },
        {
            "title": "Camera information:",
            "name": "camera_info",
            "type": "group",
            "children": [
                {
                    "title": "Model:",
                    "name": "model",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                },
                {
                    "title": "Serial:",
                    "name": "serial",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                },
                {
                    "title": "Frame rate:",
                    "name": "frame_rate",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                    "suffix": " Hz",
                },
                {
                    "title": "Dropped frames:",
                    "name": "dropped_frames",
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
        self.last_frame = None

        self.frame_count = 0
        self._rate_count = 0
        self._rate_time = None

    def _set_status(self, message):
        self.emit_status(
            ThreadCommand(
                "Update_Status",
                [message, "log"],
            )
        )

    def ini_detector(self, controller=None):
        try:
            self.camera = AndorSDK3Camera(
                index=self.settings["camera_index"],
                dll_path=self.settings["sdk3_library"],
                n_buffers=self.settings["n_buffers"],
            )
            self.camera.open()

            info = self.camera.device_info()

            self.settings.child(
                "camera_info", "model"
            ).setValue(
                info.get("CameraModel", "")
            )
            self.settings.child(
                "camera_info", "serial"
            ).setValue(
                info.get("SerialNumber", "")
            )

            # Use the camera's current AOI as the initial GUI state.
            self.settings.child(
                "aoi", "left"
            ).setValue(self.camera.left)
            self.settings.child(
                "aoi", "top"
            ).setValue(self.camera.top)
            self.settings.child(
                "aoi", "width"
            ).setValue(self.camera.width)
            self.settings.child(
                "aoi", "height"
            ).setValue(self.camera.height)

            # Configure with the current GUI settings.
            self.camera.configure(
                exposure_s=self.settings["exposure"] * 1e-3,
                roi=(
                    self.settings["aoi", "left"],
                    self.settings["aoi", "top"],
                    self.settings["aoi", "width"],
                    self.settings["aoi", "height"],
                ),
                pixel_encoding=self.settings["pixel_encoding"],
            )

            self._emit_initial_data()

            return "", True

        except Exception as exc:
            self._set_status(
                f"Marana-X initialization failed: {type(exc).__name__}: {exc}"
            )
            return str(exc), False

    def _emit_initial_data(self):
        image = np.zeros(
            (self.camera.height, self.camera.width),
            dtype=np.uint16,
        )

        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X",
                data=[
                    DataFromPlugins(
                        name="Marana-X",
                        data=[image],
                        dim="Data2D",
                    )
                ],
            )
        )

    def commit_settings(self, param):
        if self.camera is None:
            return

        name = param.name()

        try:
            if name == "exposure":
                if self.running:
                    return

                self.camera.set_float(
                    "ExposureTime",
                    param.value() * 1e-3,
                )

            elif name == "n_buffers":
                if self.running:
                    return

                self.camera.n_buffers = int(param.value())
                self.camera.configure(
                    exposure_s=self.settings["exposure"] * 1e-3,
                    roi=self._roi_from_settings(),
                    pixel_encoding=self.settings["pixel_encoding"],
                )

            elif name in {"left", "top", "width", "height"}:
                if self.running:
                    return

                self.camera.configure(
                    exposure_s=self.settings["exposure"] * 1e-3,
                    roi=self._roi_from_settings(),
                    pixel_encoding=self.settings["pixel_encoding"],
                )

                self._emit_initial_data()

        except Exception as exc:
            self._set_status(
                f"Setting {name!r} failed: {type(exc).__name__}: {exc}"
            )

    def _roi_from_settings(self):
        return (
            int(self.settings["aoi", "left"]),
            int(self.settings["aoi", "top"]),
            int(self.settings["aoi", "width"]),
            int(self.settings["aoi", "height"]),
        )

    def grab_data(self, Naverage=1, **kwargs):
        live = kwargs.get("live", False)

        if live:
            self.start_live()
        else:
            self._start_single(Naverage)

    def _start_single(self, Naverage=1):
        # First implementation: one-frame acquisition.
        # Software averaging can be added later without changing the SDK3
        # acquisition layer.
        self.start_live()

    def start_live(self):
        if self.running:
            return

        self.camera.prepare()

        self.acquisition_thread = QtCore.QThread()
        self.acquisition_worker = MaranaXAcquisitionWorker(
            self.camera
        )
        self.acquisition_worker.moveToThread(
            self.acquisition_thread
        )

        self.acquisition_thread.started.connect(
            self.acquisition_worker.run
        )
        self.acquisition_worker.frame_ready.connect(
            self._frame_received
        )
        self.acquisition_worker.acquisition_error.connect(
            self._acquisition_error
        )
        self.acquisition_worker.acquisition_stopped.connect(
            self._acquisition_stopped
        )

        self.frame_count = 0
        self._rate_count = 0
        self._rate_time = time.perf_counter()
        self.running = True

        self.emit_status(
            ThreadCommand("grab", True)
        )

        self.acquisition_thread.start()

    @QtCore.Slot(object)
    def _frame_received(self, image):
        self.last_frame = image
        self.frame_count += 1

        now = time.perf_counter()
        elapsed = now - self._rate_time

        if elapsed >= 1.0:
            fps = (
                self.frame_count - self._rate_count
            ) / elapsed

            self.settings.child(
                "camera_info", "frame_rate"
            ).setValue(fps)

            self._rate_time = now
            self._rate_count = self.frame_count

        self.dte_signal.emit(
            DataToExport(
                "Marana-X",
                data=[
                    DataFromPlugins(
                        name="Marana-X",
                        data=[image],
                        dim="Data2D",
                    )
                ],
            )
        )

    @QtCore.Slot(str)
    def _acquisition_error(self, message):
        self._set_status(message)

    @QtCore.Slot()
    def _acquisition_stopped(self):
        self.running = False
        self.emit_status(
            ThreadCommand("grab_stopped")
        )

    def stop(self):
        if not self.running:
            return

        if self.acquisition_worker is not None:
            self.acquisition_worker.stop()

        if self.acquisition_thread is not None:
            self.acquisition_thread.quit()
            self.acquisition_thread.wait(3000)

        self.acquisition_worker = None
        self.acquisition_thread = None
        self.running = False

        try:
            if self.camera is not None and self.camera.acquiring:
                self.camera.stop()
        except Exception as exc:
            self._set_status(
                f"Error stopping Marana-X: {exc}"
            )

    def close(self):
        self.stop()

        if self.camera is not None:
            try:
                self.camera.close()
            finally:
                self.camera = None


if __name__ == "__main__":
    main(__file__)
