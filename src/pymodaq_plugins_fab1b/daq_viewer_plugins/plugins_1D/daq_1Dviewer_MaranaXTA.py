from __future__ import annotations

import numpy as np
from qtpy import QtCore

from pymodaq.control_modules.viewer_utility_classes import main
from pymodaq.utils.data import Axis, DataFromPlugins
from pymodaq_data.data import DataToExport

from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MaranaX import (
    DAQ_2DViewer_MaranaX,
    MaranaXAcquisitionWorker,
)

PHASE_LABEL_TO_OFFSET = {"Pump ON": 0, "Pump OFF": 1}


class DAQ_1DViewer_MaranaXTA(DAQ_2DViewer_MaranaX):
    """
    Chopped pump-probe (transient absorption/reflectivity) viewer for the
    Andor Marana-X, built on top of :class:`DAQ_2DViewer_MaranaX`.

    No separate "mode" setting: TA acquisition (continuous, phase-tracked,
    background-subtracted ΔOD/dR/R) engages automatically whenever
    Acquisition mode = Chunked *and* Vertical binning is on - the same two
    settings the base plugin already exposes, not a redundant third one.
    Any other combination (Preview, or Chunked without vertical binning)
    falls straight through to the base plugin's normal, always-working
    behaviour, so an unexpected settings combination never means "nothing
    happens" - it just means plain camera output instead of a physical
    observable. Use Preview (vertical binning off) day to day to position
    the AOI on the dispersed spectrum and check light source conditions;
    switch on Chunked + Vertical binning for the real measurement.

    Assumes the chopper runs at exactly half the laser repetition rate,
    with the AOI covering the vertically-focused dispersed spectrum
    (typically ~10 px tall before vertical binning collapses it to one
    trace per shot).

    Workflow: position the AOI in Preview, switch on Chunked + Vertical
    binning, run "Calibrate phase" once (visually confirm which shot parity
    is pump-on - defaults to "Pump ON" so there's always output even before
    calibrating; this is deliberately not automatic, matching how published
    shot-to-shot setups treat this determination as something to verify,
    not blindly trust), block the probe and run "Measure background", then
    start the real DAQ_Scan.
    """

    live_mode_available = True
    hardware_averaging = False

    params = DAQ_2DViewer_MaranaX.params + [
        {
            "title": "Pump phase:",
            "name": "pump_phase",
            "type": "group",
            "children": [
                {
                    "title": "Even shot index is:",
                    "name": "phase_state",
                    "type": "list",
                    "limits": ["Pump ON", "Pump OFF"],
                    "value": "Pump ON",
                    "tip": "Which pump state corresponds to an even running "
                    "shot count. Defaults to Pump ON so there's always an "
                    "output; set by inspecting the even/odd comparison "
                    "'Calibrate phase' shows you, or manually.",
                },
                {
                    "title": "Calibrate phase",
                    "name": "calibrate_phase",
                    "type": "bool_push",
                    "value": False,
                    "tip": "Acquire a short burst and display the even- vs "
                    "odd-shot sums so you can confirm which is pump-on. "
                    "Requires Acquisition mode = Chunked and Vertical "
                    "binning on.",
                },
                {
                    "title": "Calibration shots:",
                    "name": "calibration_shots",
                    "type": "int",
                    "value": 40,
                    "min": 4,
                    "suffix": " shots",
                },
                {
                    "title": "Shots seen:",
                    "name": "shots_seen",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                    "tip": "Running shot count since TA acquisition mode "
                    "was armed - stays valid across every chunk and every "
                    "pause (e.g. a delay-stage move) in between.",
                },
                {
                    "title": "Shots dropped:",
                    "name": "shots_dropped",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                    "tip": "Shots where a hardware-timestamp gap revealed a "
                    "missed trigger/buffer drop - self-corrected, not lost "
                    "track of.",
                },
                {
                    "title": "Chunks aborted:",
                    "name": "chunks_aborted",
                    "type": "int",
                    "value": 0,
                    "readonly": True,
                    "tip": "Chunks discarded because a shot was dropped "
                    "mid-accumulation, breaking its even/odd alternation.",
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
                    "tip": "Block the probe beam first. Acquires one chunk "
                    "(Chunk size), splits and averages it by tracked pump "
                    "phase, and stores the two traces as the background.",
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
            "shows just the two background-subtracted averages, without "
            "combining them into a single observable. The on/off pair is "
            "always shown alongside ΔOD/dR/R too, as a diagnostic.",
        },
    ]

    def ini_attributes(self):
        super().ini_attributes()
        # Defaults to matching the "Pump ON" default of the phase_state
        # setting - always a defined value, never None, so a chunk is never
        # silently dropped for lack of a phase assignment (see
        # _ta_chunk_received). Calibrate phase / manual selection corrects
        # it as needed; this is just a starting assumption.
        self._phase_offset = PHASE_LABEL_TO_OFFSET["Pump ON"]
        self._pending_action = None  # None | "calibrate" | "background"
        self._background_on = None
        self._background_off = None
        self._background_tag = None

    # --- Setup / reconfigure ------------------------------------------------

    def _is_ta_configured(self):
        """True when the current settings make TA acquisition apply.

        No separate "mode" setting: this is exactly Chunked + vertical
        binning, the same two settings the base plugin already exposes.
        Anything else (Preview, or Chunked without vertical binning) is
        left to the base plugin's own grab_data(), so there's no dead
        combination that silently produces nothing.
        """
        return self.settings["acquisition_mode"] == "Chunked" and self.settings["vertical_binning"]

    def _reconfigure_camera(self):
        # Metadata is always on here (unlike the base plugin): the tiny
        # per-frame overhead buys the hardware timestamp TA acquisition
        # relies on for phase tracking, and it's simply unused otherwise.
        self.camera.configure(
            roi=self._roi_from_settings(),
            pixel_encoding=self.settings["pixel_encoding"],
            exposure_s=self.settings["timing", "exposure"] * 1e-3,
            frame_rate=self.settings["timing", "frame_rate"],
            trigger_mode=self.settings["trigger_mode"],
            enable_metadata=True,
        )
        self._sync_timing_settings()

    def commit_settings(self, param):
        if self.camera is None:
            return

        name = param.name()

        try:
            if name == "phase_state":
                self._phase_offset = PHASE_LABEL_TO_OFFSET[param.value()]

            elif name == "calibrate_phase":
                if param.value():
                    self._start_calibration()
                    self.settings.child("pump_phase", "calibrate_phase").setValue(False)

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

    # --- Continuous TA worker lifecycle -------------------------------------

    def _ensure_ta_worker(self):
        """Create and start the persistent TA-continuous worker if needed.

        Idempotent: a no-op if it's already running, matching the base
        plugin's _start_acquisition() guard.
        """
        if self.running:
            return
        if not self._is_ta_configured():
            raise RuntimeError(
                "Set Acquisition mode to Chunked and switch on Vertical "
                "binning first."
            )

        self.camera.prepare()

        self.acquisition_thread = QtCore.QThread()
        self.acquisition_worker = MaranaXAcquisitionWorker(
            self.camera,
            vertical_binning=True,
            ta_continuous=True,
            chunk_size=int(self.settings["chunk_size"]),
        )
        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)
        self.acquisition_worker.ta_chunk_ready.connect(self._ta_chunk_received)
        self.acquisition_worker.acquisition_error.connect(self._acquisition_error)
        self.acquisition_worker.acquisition_stopped.connect(self._acquisition_stopped)

        self.running = True
        self.acquisition_thread.start()

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
        self._ensure_ta_worker()
        self.acquisition_worker.request_chunk(chunk_size=chunk_size)

    def _start_calibration(self):
        self._ensure_ta_worker()
        self._pending_action = "calibrate"
        n = int(self.settings["pump_phase", "calibration_shots"])
        self.acquisition_worker.request_chunk(chunk_size=n)

    def _start_background_measurement(self):
        self._ensure_ta_worker()
        self._pending_action = "background"
        self.acquisition_worker.request_chunk(chunk_size=int(self.settings["chunk_size"]))

    def _clear_background(self):
        self._background_on = None
        self._background_off = None
        self._background_tag = None
        self.settings.child("background", "background_status").setValue("Not measured")

    # --- Data handling -------------------------------------------------------

    @staticmethod
    def _split_on_off(chunk, start_parity):
        """Split a (n_shots, width) chunk by tracked shot-index parity.

        Oblivious to which physical pump state "even" is - that mapping is
        self._phase_offset, applied by the caller.
        """
        n_shots = chunk.shape[0]
        even_mask = (np.arange(n_shots) + start_parity) % 2 == 0
        return chunk[even_mask], chunk[~even_mask]

    @QtCore.Slot(object, int)
    def _ta_chunk_received(self, chunk, start_parity):
        self.settings.child("pump_phase", "shots_seen").setValue(self.acquisition_worker.shots_seen)
        self.settings.child("pump_phase", "shots_dropped").setValue(self.acquisition_worker.shots_dropped)
        self.settings.child("pump_phase", "chunks_aborted").setValue(self.acquisition_worker.chunks_aborted)

        even, odd = self._split_on_off(chunk, start_parity)

        if self._pending_action == "calibrate":
            self._pending_action = None
            self._emit_calibration(even, odd)
            return

        on_shots, off_shots = (even, odd) if self._phase_offset == 0 else (odd, even)
        mean_on = on_shots.mean(axis=0)
        mean_off = off_shots.mean(axis=0)

        if self._pending_action == "background":
            self._pending_action = None
            self._store_background(mean_on, mean_off)
            return

        self._emit_observable(mean_on, mean_off)

    def _emit_calibration(self, even, odd):
        width = even.shape[1]
        x_axis = Axis(label="x", units="px", data=np.arange(width), index=0)
        even_sum, odd_sum = float(even.sum()), float(odd.sum())
        self._set_status(
            f"Phase calibration: even-shot sum={even_sum:.4g}, "
            f"odd-shot sum={odd_sum:.4g}. Inspect the two traces, then set "
            f"'Even shot index is' to whichever is pump-on."
        )
        self.dte_signal_temp.emit(
            DataToExport(
                "Marana-X TA",
                data=[
                    DataFromPlugins(
                        name="Even/Odd shots",
                        data=[even.mean(axis=0), odd.mean(axis=0)],
                        dim="Data1D",
                        labels=["Even shots", "Odd shots"],
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

    def _store_background(self, mean_on, mean_off):
        self._background_on = mean_on
        self._background_off = mean_off
        self._background_tag = self._background_tag_current()
        self.settings.child("background", "background_status").setValue(
            f"Captured ({mean_on.shape[0]} px, {int(self.settings['chunk_size'])} shots)"
        )

    def _background_stale(self):
        return (
            self._background_tag is not None
            and self._background_tag != self._background_tag_current()
        )

    def _emit_observable(self, mean_on, mean_off):
        if self._background_on is not None and not self._background_stale():
            on_corr = mean_on - self._background_on
            off_corr = mean_off - self._background_off
        else:
            if self._background_tag is not None:
                self._set_status(
                    "Stored background no longer matches the current AOI/exposure/"
                    "encoding; not subtracting it. Re-run 'Measure background'."
                )
            on_corr = mean_on
            off_corr = mean_off

        width = mean_on.shape[0]
        x_axis = Axis(label="x", units="px", data=np.arange(width), index=0)
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

        self.dte_signal.emit(DataToExport("Marana-X TA", data=data))


if __name__ == "__main__":
    main(__file__)
