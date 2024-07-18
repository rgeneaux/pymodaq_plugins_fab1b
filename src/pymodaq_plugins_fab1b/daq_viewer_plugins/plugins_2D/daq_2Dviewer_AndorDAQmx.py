from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from daq_2Dviewer_AndorFAB1B import DAQ_2DViewer_AndorFAB1B as DAQAndor
from ..plugins_1D.daq_1Dviewer_DAQmxFAB1B import DAQ_1DViewer_DAQmxFAB1B as DAQDaqmx

class QMultiWait(QObject):
    'Used to wait for acquisition of several dectector before processing'
    all_data_acquired = Signal()

    def __init__(self, parent=None):
        super(QMultiWait, self).__init__(parent)
        self._waitable = set()
        self._waitready = set()
    def addWaitableSignal(self, signal):
        if signal not in self._waitable:
            self._waitable.add(signal)
            signal.connect(self._checkSignal)
    @Slot()
    def _checkSignal(self):
        sender = self.sender()
        self._waitready.add(sender)
        if len(self._waitready) == len(self._waitable):
            self.all_data_acquired.emit()
    def clear(self):
        for signal in self._waitable:
            signal.disconnect(self._checkSignal)


class DAQ_2DViewer_AndorDAQmx(DAQAndor):
    """ Instrument plugin class for the Andor camera working with a DAQmx card.
    """
    params_camera = DAQAndor.params
    params_daqcard = DAQDaqmx.params

    params = params_camera + params_daqcard

    def ini_attributes(self):
        self.controller: DAQAndor = None
        self.daqcard_controller: DAQDaqmx = None

        super().ini_attributes()

    def commit_settings(self, param: Parameter):
        if 'camera_settings' in putils.get_param_path(param):
            super().commit_settings(param)
        elif 'daq_settings' in putils.get_param_path(param):
            self.daqcard_controller.commit_settings(param)
        QtWidgets.QApplication.processEvents()

    def ini_detector(self, controller=None):
        cam_status, cam_init = super().ini_detector(controller)
        QtWidgets.QApplication.processEvents()

        self.daqcard_controller = DAQDaqmx(None, self.settings.child('daqcard_settings').saveState())
        self.daqcard_controller.settings = self.settings.child('daqcard_settings')
        self.daqcard_controller.emit_status = self.emit_status
        daq_status, daq_init = self.daqcard_controller.ini_detector(controller)


        # Disconnect data signals and reconnect it here
        self.daqcard_controller.daqcard_ready_signal.disconnect()
        self.camera_ready_signal.disconnect()

        # Create a multiwait to synchronize the signals and then fire 'process_all_data'
        self.multiwait = QMultiWait()
        self.multiwait.addWaitableSignal(self.daqcard_controller.dte_ready_signal)
        self.multiwait.addWaitableSignal(self.camera_ready_signal)
        self.multiwait.all_data_acquired.connect(self.process_all_data)

        QtWidgets.QApplication.processEvents()

        initialized = daq_init and cam_init
        return daq_status + cam_status, initialized

    def close(self):
        """Terminate the communication protocol"""
        if self.daqcard_controller is not None:
            self.daqcard_controller.close()
        super().close()

    def grab_data(self, Naverage=1, **kwargs):
        self.daqcard_controller.grab_data(Naverage, **kwargs)
        super().grab_data(Naverage, **kwargs)

    def process_all_data(self):
        daq_data = self.daqcard_controller.daqcard_data
        camera_data = self.camera_data

        dte_total = DataToExport('Andor Camera DAQ', data=camera_data.data.append(daq_data.data))
        self.dte_signal.emit(dte_total)

    def stop(self):
        super().stop()
        if self.daqcard_controller is not None:
            self.daqcard_controller.stop()
        return ''


if __name__ == '__main__':
    main(__file__)
