from __future__ import annotations

import copy
import time

import numpy as np
from qtpy import QtCore

from pymodaq.control_modules.thread_commands import ThreadStatus
from pymodaq.control_modules.viewer_utility_classes import main
from pymodaq.utils.data import Axis, DataFromPlugins
from pymodaq_data.data import DataToExport
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaX import (
    DAQ_2DViewer_MaranaX,
)
from pymodaq_plugins_fab1b.hardware.uniblitz_vlm1 import (
    VLM1Shutter,
    available_ports,
    default_port,
)

SHUTTER_TA_MODE = "Shutter TA"

_PORTS = available_ports()


def shutter_observable(pon, poff, kind):
    """Differential signal from a pump-on / pump-off pair of spectra.

    Returns (name, data). NaN and inf (zero or negative counts) become 0, and a
    zero in ``poff`` is replaced by a tiny value, so a dead pixel never
    poisons the whole trace.
    """
    pon = np.asarray(pon, dtype=np.float64)
    poff = np.array(poff, dtype=np.float64)
    poff[poff == 0] = 1e-10
    with np.errstate(all="ignore"):
        if kind == "dR/R":
            name, data = "Differential Reflectivity", (pon - poff) / poff
        else:
            name, data = "Differential Optical Density", -np.log10(pon / poff)
    return name, np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _build_params():
    # Deep copy: the nested dicts are shared with the base class's params, and
    # changing the acquisition mode's limits in place would change its own.
    params = copy.deepcopy(DAQ_2DViewer_MaranaX.params)
    for entry in params:
        if entry.get("name") == "acquisition_mode":
            entry["limits"] = list(entry["limits"]) + [SHUTTER_TA_MODE]
            entry["tip"] += (
                " Shutter TA: one shutter-open frame and one shutter-closed "
                "frame per grab, combined into dR/R or ΔOD (rows are averaged "
                "in software into a 1D spectrum)."
            )
    params += [
        {
            "title": "Shutter (VLM1):",
            "name": "shutter",
            "type": "group",
            "children": [
                {
                    "title": "COM port:",
                    "name": "com_port",
                    "type": "list",
                    "limits": _PORTS or [""],
                    "value": default_port(_PORTS) or "",
                    "tip": "Serial port of the VLM1 shutter driver. Changing it "
                    "closes the shutter and reconnects on the new port.",
                },
                {
                    "title": "Settle time:",
                    "name": "settle_time",
                    "type": "float",
                    "value": 50.0,
                    "min": 0.0,
                    "suffix": " ms",
                    "tip": "Wait after each shutter move before the next frame "
                    "is taken (Shutter TA mode).",
                },
                {
                    "title": "Pump on before off",
                    "name": "on_before_off",
                    "type": "bool",
                    "value": True,
                    "tip": "Take the shutter-open (pump on) frame first. Untick "
                    "to take the shutter-closed frame first.",
                },
                {
                    "title": "Shutter (lit = open):",
                    "name": "shutter_bool",
                    "type": "led_push",
                    "value": False,
                    "tip": "Click to open / close the shutter by hand. The "
                    "shutter is on the pump arm: open = pump on.",
                },
            ],
        },
        {
            "title": "Shutter TA:",
            "name": "shutter_ta",
            "type": "group",
            "children": [
                {
                    "title": "Observable:",
                    "name": "diff_type",
                    "type": "list",
                    "limits": ["dR/R", "ΔOD"],
                    "value": "dR/R",
                    "tip": "dR/R = (on - off) / off;  ΔOD = -log10(on / off).",
                },
                {
                    "title": "Show pump on/off:",
                    "name": "show_pump_on_off",
                    "type": "bool",
                    "value": True,
                    "tip": "Also emit the two background-subtracted spectra as "
                    "a second panel.",
                },
                {
                    "title": "Measure background",
                    "name": "measure_background",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Block the probe beam first. Takes one on/off pair "
                    "and stores it as the background to subtract.",
                },
                {
                    "title": "Clear background",
                    "name": "clear_background",
                    "type": "bool_push",
                    "value": False,
                },
                {
                    "title": "Background:",
                    "name": "background_status",
                    "type": "str",
                    "value": "Not measured",
                    "readonly": True,
                },
            ],
        },
    ]
    return params


class DAQ_2DViewer_MaranaXShutter(DAQ_2DViewer_MaranaX):
    """
    Marana-X viewer that also drives a Uniblitz VLM1 shutter.

    Acquisition mode selects the behaviour:

    - Preview / Chunked: exactly :class:`DAQ_2DViewer_MaranaX`, with the shutter
      available for manual open / close (the "Shutter" toggle).
    - Shutter TA: slow, sequential transient reflectivity / absorption. Every
      grab opens the shutter (pump on), waits the settle time, takes one frame,
      closes it (pump off), waits, takes a second frame, and emits dR/R or ΔOD
      with the two spectra. PyMoDAQ's own loop repeats the grab in Live mode
      and averages for Naverage.

    The shutter is driven through :class:`VLM1Shutter` held by composition in
    ``self.shutter``: ``self.controller`` already holds the camera handle (see
    ``ini_controller_init``) and must not be shared with a second device. The
    shutter is connected (and closed) on initialisation and closed and
    disconnected on close.
    """

    params = _build_params()

    @property
    def live_mode_available(self):
        # Preview / Chunked run on the base class's own worker thread; Shutter
        # TA is one blocking grab per on/off pair, repeated by PyMoDAQ's loop.
        try:
            return not self._is_shutter_ta()
        except Exception:
            return True

    def ini_attributes(self):
        super().ini_attributes()
        self.shutter = None
        self._bkg_pon = None
        self._bkg_poff = None
        self._bkg_tag = None
        self._stale_warned_tag = None
        self._last_mode = "Preview"
        self._last_pair_t = None

    def _is_shutter_ta(self):
        return self.settings["acquisition_mode"] == SHUTTER_TA_MODE

    def _apply_mode_visibility(self):
        ta = self._is_shutter_ta()
        self.settings.child("shutter_ta").show(ta)
        self.settings.child("shutter", "on_before_off").show(ta)
        self.settings.child("shutter", "settle_time").show(ta)
        for name in ("chunk_size", "vertical_binning", "display"):
            self.settings.child(name).show(not ta)

    # --- Shutter -------------------------------------------------------------

    def _connect_shutter(self):
        """Open the serial port and start closed. Returns (info, initialized)."""
        port = self.settings["shutter", "com_port"]
        if not port:
            return "No COM port available for the VLM1 shutter", False
        shutter = VLM1Shutter(port)
        try:
            shutter.connect()
            shutter.close_shutter()
        except Exception as exc:
            try:
                shutter.disconnect()
            except Exception:
                pass
            return f"Shutter initialization failed on {port}: {type(exc).__name__}: {exc}", False
        self.shutter = shutter
        self.settings.child("shutter", "shutter_bool").setValue(False)
        return "", True

    def _disconnect_shutter(self):
        if self.shutter is None:
            return
        shutter, self.shutter = self.shutter, None
        try:
            shutter.close_shutter()
        except Exception as exc:
            self._set_status(f"Could not close the shutter: {exc}")
        finally:
            shutter.disconnect()
        self.settings.child("shutter", "shutter_bool").setValue(False)

    def _set_shutter(self, open_):
        """Move the shutter and mirror its state in the settings tree."""
        if self.shutter is None:
            raise RuntimeError("Shutter is not connected")
        self.shutter.set_open(open_)
        self.settings.child("shutter", "shutter_bool").setValue(bool(open_))

    # --- Lifecycle / settings ------------------------------------------------

    def ini_detector(self, controller=None):
        info, initialized = self._connect_shutter()
        if not initialized:
            self._set_status(info)
            return info, False

        result = super().ini_detector(controller)
        if result[1]:
            self._last_mode = self.settings["acquisition_mode"]
            self._apply_mode_visibility()
        else:
            self._disconnect_shutter()
        return result

    def _emit_initial_data(self):
        if not self._is_shutter_ta():
            super()._emit_initial_data()
            return
        width = self.camera.width
        x_axis = Axis(label="x", units="px", data=np.arange(width), index=0)
        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X Shutter TA",
                data=[
                    DataFromPlugins(
                        name="Differential Reflectivity",
                        data=[np.zeros(width)],
                        dim="Data1D",
                        axes=[x_axis],
                    )
                ],
            )
        )

    def commit_settings(self, param):
        name = param.name()

        try:
            if name == "com_port":
                self._disconnect_shutter()
                info, initialized = self._connect_shutter()
                if not initialized:
                    self._set_status(info)

            elif name == "shutter_bool":
                if self.shutter is None:
                    raise RuntimeError("Shutter is not connected")
                self.shutter.set_open(bool(param.value()))

            elif name == "acquisition_mode":
                new_mode = param.value()
                if self.running and (new_mode == SHUTTER_TA_MODE) != (
                    self._last_mode == SHUTTER_TA_MODE
                ):
                    self._set_status(
                        "Stop the live acquisition before switching to or from Shutter TA."
                    )
                    self.settings.child("acquisition_mode").setValue(self._last_mode)
                    return
                self._last_mode = new_mode
                super().commit_settings(param)
                self._apply_mode_visibility()

            elif name == "measure_background":
                if param.value():
                    self._measure_background()
                    self.settings.child("shutter_ta", "measure_background").setValue(False)

            elif name == "clear_background":
                if param.value():
                    self._clear_background()
                    self.settings.child("shutter_ta", "clear_background").setValue(False)

            elif name in ("settle_time", "diff_type", "show_pump_on_off", "on_before_off"):
                pass

            else:
                super().commit_settings(param)

        except Exception as exc:
            self._set_status(f"Setting {name!r} failed: {type(exc).__name__}: {exc}")

    def close(self):
        self._disconnect_shutter()
        super().close()

    # --- Shutter TA acquisition ---------------------------------------------

    def _snap_frame(self):
        """One frame, rows averaged into a 1D float spectrum.

        Starts and stops the camera around every frame: the frame is then
        guaranteed to begin after the shutter has settled, which a free-running
        acquisition could not promise.
        """
        camera = self.camera
        timeout_s = max(2.0, 3 * self.settings["timing", "exposure"] * 1e-3 + 1.0)
        camera.prepare()
        camera.start()
        try:
            deadline = time.perf_counter() + timeout_s
            while True:
                # Short polls, as the acquisition worker does: wait_frame holds
                # the SDK lock while it waits, and this keeps GUI calls
                # (exposure changes) from stalling behind it.
                result = camera.wait_frame(timeout_ms=100)
                if result is not None:
                    break
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"No frame within {timeout_s:.1f} s "
                        f"(trigger mode {self.settings['trigger_mode']!r})"
                    )
            buffer, _size = result
            try:
                return camera.frame_view(buffer).mean(axis=0, dtype=np.float64)
            finally:
                camera.requeue(buffer)
        finally:
            camera.stop()

    def _acquire_frame(self, open_):
        self._set_shutter(open_)
        QtCore.QThread.msleep(int(self.settings["shutter", "settle_time"]))
        return self._snap_frame()

    def _acquire_on_off_pair(self):
        if self.settings["shutter", "on_before_off"]:
            pon = self._acquire_frame(True)
            poff = self._acquire_frame(False)
        else:
            poff = self._acquire_frame(False)
            pon = self._acquire_frame(True)
        return pon, poff

    def grab_data(self, Naverage=1, **kwargs):
        if not self._is_shutter_ta():
            super().grab_data(Naverage, **kwargs)
            return

        try:
            pon, poff = self._acquire_on_off_pair()
            self._emit_shutter_ta(pon, poff)
            self._update_pair_stats()
        except Exception as exc:
            self._set_status(f"Shutter TA acquisition failed: {type(exc).__name__}: {exc}")
            # PyMoDAQ's loop waits for data that will never come; end the grab.
            self.emit_status(ThreadCommand(ThreadStatus.STOP))

    def _update_pair_stats(self):
        now = time.perf_counter()
        info = self.settings.child("camera_info")
        info.child("frames_captured").setValue(info["frames_captured"] + 2)
        if self._last_pair_t is not None and now > self._last_pair_t:
            pairs_per_s = 1.0 / (now - self._last_pair_t)
            info.child("displayed_fps").setValue(pairs_per_s)
            info.child("camera_fps").setValue(2 * pairs_per_s)
        self._last_pair_t = now

    # --- Background ----------------------------------------------------------

    def _background_tag_current(self):
        return (
            self.settings["aoi", "width"],
            self.settings["aoi", "height"],
            self.settings["timing", "exposure"],
            self.settings["pixel_encoding"],
            self.settings["gain_mode"],
        )

    def _measure_background(self):
        if self.camera is None:
            self._set_status("Initialize the detector first.")
            return
        if not self._is_shutter_ta():
            self._set_status("Set Acquisition mode to Shutter TA first.")
            return
        pon, poff = self._acquire_on_off_pair()
        self._bkg_pon, self._bkg_poff = pon, poff
        self._bkg_tag = self._background_tag_current()
        self.settings.child("shutter_ta", "background_status").setValue(
            f"Captured ({pon.shape[0]} px)"
        )

    def _clear_background(self):
        self._bkg_pon = self._bkg_poff = self._bkg_tag = None
        self.settings.child("shutter_ta", "background_status").setValue("Not measured")

    def _subtract_background(self, pon, poff):
        if self._bkg_tag is None:
            return pon, poff
        current = self._background_tag_current()
        if current != self._bkg_tag:
            if self._stale_warned_tag != current:
                self._stale_warned_tag = current
                self._set_status(
                    "Stored background no longer matches the current AOI / exposure / "
                    "encoding / gain mode; not subtracting it. Re-run 'Measure background'."
                )
            return pon, poff
        return pon - self._bkg_pon, poff - self._bkg_poff

    def _emit_shutter_ta(self, pon, poff):
        pon, poff = self._subtract_background(pon, poff)
        name, data = shutter_observable(pon, poff, self.settings["shutter_ta", "diff_type"])
        x_axis = Axis(label="x", units="px", data=np.arange(data.shape[0]), index=0)

        dwa_list = [
            DataFromPlugins(name=name, data=[data], dim="Data1D", labels=["Camera"], axes=[x_axis])
        ]
        if self.settings["shutter_ta", "show_pump_on_off"]:
            dwa_list.append(
                DataFromPlugins(
                    name="Pump On/Off",
                    data=[poff, pon],
                    dim="Data1D",
                    labels=["Pump Off", "Pump On"],
                    axes=[x_axis],
                )
            )
        self.dte_signal.emit(DataToExport("Marana-X Shutter TA", data=dwa_list))


if __name__ == "__main__":
    main(__file__)
