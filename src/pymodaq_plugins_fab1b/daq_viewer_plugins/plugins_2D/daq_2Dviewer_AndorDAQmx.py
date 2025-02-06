from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from qtpy.QtCore import QObject, Signal, Slot
from pymodaq.utils.parameter import utils as putils
from qtpy import QtWidgets
from typing import List

from daq_2Dviewer_AndorFAB1B import DAQ_2DViewer_AndorFAB1B as Andor_Class
from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_1D.daq_1Dviewer_NIDAQmxFAB1B import DAQ_1DViewer_NIDAQmxFAB1B as NI_Class

class QMultiWait(QObject):
    'Used to wait for acquisition of several dectector before processing'
    all_data_acquired = Signal(list)
    all_data = []

    def __init__(self, parent=None):
        super(QMultiWait, self).__init__(parent)
        self._waitable = set()
        self._waitready = set()

    def addWaitableSignal(self, signal):
        if signal not in self._waitable:
            self._waitable.add(signal)
            signal.connect(self._checkSignal)

    @Slot(DataToExport)
    def _checkSignal(self, data):
        sender = self.sender()
        print(sender)

        self._waitready.add(sender)
        self.all_data.append(data)

        if len(self._waitready) == len(self._waitable):
            self.all_data_acquired.emit(self.all_data)
            self.all_data = []
            self._waitready = set()
    def clear(self):
        for signal in self._waitable:
            signal.disconnect(self._checkSignal)


class DAQ_2DViewer_AndorDAQmx(Andor_Class):
    """ Instrument plugin class for the Andor camera working with a DAQmx card.
    """
    params_camera = Andor_Class.params
    params_daqcard = NI_Class.params

    params = params_camera + [param for param in params_daqcard if param['name'] != 'controller_status']  #controller status already in Andor

    def __init__(self, parent=None, params_state=None):

        Andor_Class.__init__(self, parent, params_state)
        self.DAQ = NI_Class(parent, params_state=self.params_daqcard)


    def commit_settings(self, param: Parameter):
        path = putils.get_param_path(param)
        if 'camera_settings' in path:
            super().commit_settings(param)
        elif 'daq_settings' in path:
            # We have to set the param value because self.DAQ.update_settings is not triggered by the tree change
            self.DAQ.settings.child(*path[2:]).setValue(param.value())
            self.DAQ.commit_settings(param)
        QtWidgets.QApplication.processEvents()


    def ini_detector(self, controller=None):
        cam_status, cam_init = super().ini_detector(controller)
        QtWidgets.QApplication.processEvents()

        daq_status, daq_init = self.DAQ.ini_detector(controller)
        QtWidgets.QApplication.processEvents()

        self.dte_signal.disconnect()  # Disconnect camera dte signal

        # Create a multiwait to synchronize the signals, add the data signals and then fire 'process_all_data'
        self.multiwait = QMultiWait()
        self.multiwait.addWaitableSignal(self.DAQ.dte_signal)
        self.multiwait.addWaitableSignal(self.dte_signal)
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
        self.DAQ.grab_data(Naverage, **kwargs)
        super().grab_data(Naverage, **kwargs)

        if not self.live:
            super().stop()
            self.DAQ.stop()

    @Slot(list)
    def process_all_data(self, data):
        # daq_dfp = [dte.data for dte in data if dte.name =='DAQ Card'][0][0]
        # camera_dfp = [dte.data for dte in data if dte.name =='Andor'][0][0]
        # #
        # dte_total = DataToExport('Andor Camera DAQ', data=[camera_dfp])
        # self.dte_signal.emit(dte_total)r
        pass


    def stop(self):
        super().stop()
        self.DAQ.stop()


if __name__ == '__main__':
    main(__file__, init=False)
