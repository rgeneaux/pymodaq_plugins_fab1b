import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_stellarnet.daq_viewer_plugins.plugins_1D.daq_1Dviewer_Stellarnet import DAQ_1DViewer_Stellarnet
from pymodaq_plugins_oceaninsight.daq_viewer_plugins.plugins_1D.daq_1Dviewer_Omnidriver import DAQ_1DViewer_Omnidriver


class PythonWrapperOfYourInstrument:
    pass

class DAQ_1DViewer_OceanStellarSpectrometer(DAQ_1DViewer_Omnidriver, DAQ_1DViewer_Stellarnet):
    """ Instrument plugin class for a 1D viewer combining one Stellarnet spectrometer and one OceanInsight spectrometer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.
    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """
    ocean_params = DAQ_1DViewer_Omnidriver.params
    stellar_params = DAQ_1DViewer_Stellarnet.params

    params = comon_parameters+[
        {'title': 'Ocean:', 'name': 'ocean_settings', 'type': 'group', 'children': ocean_params},
        {'title': 'Stellarnet:', 'name': 'stellar_settings', 'type': 'group', 'children': stellar_params},
    ]

    def ini_attributes(self):
        self.controller.ocean_controller: DAQ_1DViewer_Omnidriver = None
        self.controller.stellar_controller: DAQ_1DViewer_Stellarnet = None
        self.x_axis: Axis = None
        super().ini_attributes()

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        ## TODO for your custom plugin
        if param.name() == "a_parameter_you've_added_in_self.params":
           self.controller.your_method_to_apply_this_param_change()
#        elif ...
        ##

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        ocean_status, ocean_initialized = DAQ_1DViewer_Omnidriver.ini_detector(self, controller.ocean_controller)
        QtWidgets.QApplication.processEvents()
        stellar_status, stellar_initialized = DAQ_1DViewer_Stellarnet.ini_detector(self, controller.stellar_controller)
        QtWidgets.QApplication.processEvents()

        initialized = ocean_initialized and stellar_initialized

        return ocean_status+stellar_status, initialized

    def close(self):
        """Terminate the communication protocol"""
        ## TODO for your custom plugin
        raise NotImplemented  # when writing your own plugin remove this line
        #  self.controller.your_method_to_terminate_the_communication()  # when writing your own plugin replace this line

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        ## TODO for your custom plugin: you should choose EITHER the synchrone or the asynchrone version following

        ##synchrone version (blocking function)
        data_tot = self.controller.your_method_to_start_a_grab_snap()
        self.dte_signal.emit(DataToExport('myplugin',
                                          data=[DataFromPlugins(name='Mock1', data=data_tot,
                                                                dim='Data1D', labels=['dat0', 'data1'],
                                                                axes=[self.x_axis])]))

        ##asynchrone version (non-blocking function with callback)
        self.controller.your_method_to_start_a_grab_snap(self.callback)
        #########################################################


    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        data_tot = self.controller.your_method_to_get_data_from_buffer()
        self.dte_signal.emit(DataToExport('myplugin',
                                          data=[DataFromPlugins(name='Mock1', data=data_tot,
                                                                dim='Data1D', labels=['dat0', 'data1'])]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        ## TODO for your custom plugin
        raise NotImplemented  # when writing your own plugin remove this line
        self.controller.your_method_to_stop_acquisition()  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['Some info you want to log']))
        ##############################
        return ''


if __name__ == '__main__':
    main(__file__)
