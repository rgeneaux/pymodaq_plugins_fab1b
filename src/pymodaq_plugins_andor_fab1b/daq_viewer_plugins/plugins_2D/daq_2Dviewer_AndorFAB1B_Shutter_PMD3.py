import copy

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins
from pymodaq.control_modules.viewer_utility_classes import main
from pymodaq.utils.parameter import Parameter

from qtpy import QtCore
import numpy as np

from pymodaq.utils.logger import set_logger, get_module_name
logger = set_logger(get_module_name(__file__))

from pymodaq_plugins_uniblitz.daq_move_plugins.daq_move_VLM1 import DAQ_Move_VLM1

from pymodaq_plugins_andor_fab1b.daq_viewer_plugins.plugins_2D.daq_2Dviewer_AndorFAB1B_PMD3 import (
    DAQ_2DViewer_AndorFAB1B_PMD3,
)


# Deep-copy the inherited params: the nested dicts (e.g. 'acq_mode' limits) are shared
# object references across a plain list '+', so mutating them in place would also
# change DAQ_2DViewer_AndorFAB1B_PMD3's own acq_mode options.
_base_params = copy.deepcopy(DAQ_2DViewer_AndorFAB1B_PMD3.params)
for _group in _base_params:
    if _group['name'] == 'camera_settings':
        for _child in _group['children']:
            if _child['name'] == 'acq':
                for _sub in _child['children']:
                    if _sub['name'] == 'acq_mode':
                        _sub['limits'] = ['Normal', 'Fast 1D', 'Shutter TA']
        _group['children'].append(
            {'title': 'Shutter (VLM1):', 'name': 'shutter', 'type': 'group', 'children': [
                {'title': 'COM Port:', 'name': 'com_port', 'type': 'list',
                 'limits': DAQ_Move_VLM1.COMports, 'value': DAQ_Move_VLM1.COMport},
                {'title': 'Settle time (ms):', 'name': 'settle_time', 'type': 'float', 'value': 50},
                {'title': 'Pump on before off', 'name': 'on_before_off', 'type': 'bool', 'value': True},
                {'title': 'Shutter:', 'name': 'shutter_bool', 'type': 'led_push', 'value': False},
            ]}
        )


class DAQ_2DViewer_AndorFAB1B_Shutter_PMD3(DAQ_2DViewer_AndorFAB1B_PMD3):
    """
    Marana-X viewer that also drives a VLM1 shutter for a slow, sequential
    transient-absorption mode: open shutter -> snap pump-on frame -> close shutter ->
    snap pump-off frame -> dR/R or dOD. Selected via the new 'Shutter TA' acquisition
    mode; the inherited 'Normal' and 'Fast 1D' (chopper-based) modes are untouched and
    delegate straight to the base plugin.

    Owns a separate DAQ_Move_VLM1 instance (composition, not multiple inheritance):
    that plugin keeps its serial handle in self.controller, the same attribute name
    DAQ_Viewer_base uses for the camera handle, so inheriting from both would make one
    silently clobber the other.
    """

    params = _base_params

    def ini_attributes(self):
        super().ini_attributes()
        self.shutter: DAQ_Move_VLM1 = None

    def ini_detector(self, controller=None):
        self.shutter = DAQ_Move_VLM1()
        self.shutter.settings.child('COM_port').setValue(
            self.settings["camera_settings", "shutter", "com_port"])
        status = self.shutter.ini_stage()
        if not status.initialized:
            return f"Shutter initialization failed: {status.info}", False
        self._set_shutter(False)  # start closed

        return super().ini_detector(controller)

    def commit_settings(self, param: Parameter):
        name = param.name()

        if name == 'com_port':
            self.shutter.close()
            self.shutter.settings.child('COM_port').setValue(param.value())
            self.shutter.ini_stage()

        elif name == 'shutter_bool':
            self._set_shutter(bool(param.value()))

        elif name in ('settle_time', 'on_before_off'):
            pass  # read directly at grab time, nothing to push to hardware now

        else:
            super().commit_settings(param)

    def set_acq_mode(self):
        super().set_acq_mode()
        is_shutter_ta = self.settings["camera_settings", "acq", "acq_mode"] == 'Shutter TA'
        self.settings.child("camera_settings", "shutter").show(is_shutter_ta)
        if is_shutter_ta:
            # Base class's 'else' branch (anything not 'Normal') already shows
            # dev/chunk_size/fast_mode/display for Fast 1D; only fast_mode/display/
            # chunk_size don't apply here.
            self.settings.child("camera_settings", "acq", "fast_mode").hide()
            self.settings.child("camera_settings", "acq", "display").hide()
            self.settings.child("camera_settings", "acq", "diff_type").show()
            self.settings.child("camera_settings", "timing_opts", "chunk_size").hide()

    def _set_shutter(self, open_):
        self.shutter.move_Abs(1 if open_ else 0)
        self.settings.child("camera_settings", "shutter", "shutter_bool").setValue(open_)

    def _acquire_frame(self, open_):
        """Move the shutter, wait for it to settle, snap one frame, vertically bin it
        in software into a 1D spectrum (same convention as the inherited Fast 1D mode)."""
        self._set_shutter(open_)
        QtCore.QThread.msleep(int(self.settings["camera_settings", "shutter", "settle_time"]))
        frame = np.squeeze(self.controller.snap())
        if frame.ndim == 2:
            frame = frame.mean(axis=0)
        return frame

    def _acquire_on_off_pair(self):
        if self.settings["camera_settings", "shutter", "on_before_off"]:
            pon = self._acquire_frame(True)
            poff = self._acquire_frame(False)
        else:
            poff = self._acquire_frame(False)
            pon = self._acquire_frame(True)
        return pon, poff

    def grab_data(self, Naverage=1, **kwargs):
        if self.settings["camera_settings", "acq", "acq_mode"] != 'Shutter TA':
            super().grab_data(Naverage, **kwargs)
            return

        if 'live' in kwargs:
            self.live = kwargs['live']

        self.temperature_timer.stop()
        try:
            pon, poff = self._acquire_on_off_pair()
            self._emit_shutter_ta(pon, poff)
            if self.settings["camera_settings", "timing_opts", "fps_on"]:
                self.update_fps()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))
        finally:
            self.temperature_timer.start(self.temp_freq)

    def _emit_shutter_ta(self, pon, poff):
        if self.bkg_pon is not None and self.bkg_poff is not None:
            pon = pon - self.bkg_pon
            poff = poff - self.bkg_poff

        poff = poff.copy()
        poff[poff == 0] = 1e-10

        if self.settings["camera_settings", "acq", "diff_type"] == 'dR/R':
            data = (pon - poff) / poff
            name = "Differential Reflectivity"
        else:
            data = -np.real(np.log(pon / poff))
            name = "Differential Optical Density"

        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0

        dfp_list = [DataFromPlugins(name=name, data=[np.squeeze(data)], dim='Data1D',
                                     labels=['Camera'], axes=self.axes)]

        if self.settings["camera_settings", "dev", "pumponoff_on"]:
            dfp_list.append(DataFromPlugins(name='Pump On/Off', data=[np.squeeze(poff), np.squeeze(pon)],
                                             dim='Data1D', labels=['Pump Off', 'Pump On'], axes=self.axes))

        self.data_grabed_signal.emit(dfp_list)

    def take_background(self):
        if self.settings["camera_settings", "acq", "acq_mode"] != 'Shutter TA':
            super().take_background()
            return
        self.bkg_pon, self.bkg_poff = self._acquire_on_off_pair()

    def close(self):
        if self.shutter is not None:
            self.shutter.move_Abs(0)
            self.shutter.close()
        super().close()


if __name__ == '__main__':
    main(__file__)
