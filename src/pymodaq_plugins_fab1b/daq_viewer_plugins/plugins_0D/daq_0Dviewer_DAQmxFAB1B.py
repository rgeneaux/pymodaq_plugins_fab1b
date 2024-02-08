import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_daqmx.hardware.national_instruments.daqmx import DAQmx, \
    Edge, ClockSettings, Counter, ClockCounter,  TriggerSettings, AIChannel

from PyDAQmx import DAQmx_Val_ContSamps


class DAQ_0DViewer_DAQmxFAB1B(DAQ_Viewer_base):
    """ Instrument plugin class for a OD viewer.

    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    This is for FAB1B NI DAQ Card, used to read chopper status and other analog signals, when triggered by the camera

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """
    params = comon_parameters+[
        {'title': 'Display type:', 'name': 'display', 'type': 'list', 'limits': ['0D', '1D'], 'value':'1D'},
        {'title': 'Axis:', 'name': 'time_axis', 'type': 'list', 'limits': ['Time', 'Samples'], 'value':'Time'},
        {'title': 'Frequency Acq. (kS):', 'name': 'frequency', 'type': 'int', 'value': 500, 'min': 0.001, 'max': 500.0},
        {'title': 'Nsamples:', 'name': 'Nsamples', 'type': 'int', 'value': 1000, 'default': 100, 'min': 1},
        {'title': 'AI:', 'name': 'ai_channel', 'type': 'list',
         'limits': DAQmx.get_NIDAQ_channels(source_type='Analog_Input'),
         'value': DAQmx.get_NIDAQ_channels(source_type='Analog_Input')[0]},
        {'title': 'Trigger Settings:', 'name': 'trigger_settings', 'type': 'group', 'visible': True, 'children': [
            {'title': 'Enable?:', 'name': 'enable', 'type': 'bool', 'value': True, },
            {'title': 'Trigger Source:', 'name': 'trigger_channel', 'type': 'list',
             'limits': DAQmx.getTriggeringSources(), 'value': DAQmx.getTriggeringSources()[0]},
            {'title': 'Edge type:', 'name': 'edge', 'type': 'list', 'limits': Edge.names(), 'visible': True},
            {'title': 'Level:', 'name': 'level', 'type': 'float', 'value': 1., 'visible': True}
        ]}
    ]

    def ini_attributes(self):
        self.channels_ai = None
        self.clock_settings = None
        self.data_tot = None
        self.live = False
        self.Naverage = 1
        self.ind_average = 0
        self.axis = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        self.update_tasks()

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

        self.ini_detector_init(old_controller=controller,
                               new_controller=dict(ai=DAQmx()))
        self.update_tasks()

        info = "Analog measurement ready"
        initialized = True

        return info, initialized

    def update_tasks(self):
        # Create channels
        self.channels_ai = [AIChannel(name=self.settings.child('ai_channel').value(),
                                      source='Analog_Input', analog_type='Voltage',
                                      value_min=-10., value_max=10., termination='Diff', ),
                            ]

        self.clock_settings = ClockSettings(frequency=self.settings['frequency']*1000,
                                            Nsamples=self.settings['Nsamples'],
                                            repetition=self.live,)

        self.trigger_settings = \
            TriggerSettings(trig_source=self.settings['trigger_settings', 'trigger_channel'],
                            enable=self.settings['trigger_settings', 'enable'],
                            edge=self.settings['trigger_settings', 'edge'],
                            level=self.settings['trigger_settings', 'level'],)

        self.controller['ai'].update_task(self.channels_ai, self.clock_settings, trigger_settings=self.trigger_settings)

        if self.settings['time_axis'] == 'Time':
            self.axis = np.linspace(0, self.settings['Nsamples']/self.settings['frequency']*1000, num=self.settings['Nsamples'])
        elif self.settings['time_axis'] == 'Samples':

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
        update = False

        if 'live' in kwargs:
            if kwargs['live'] != self.live:
                update = True
            self.live = kwargs['live']

        if Naverage != self.Naverage:
            self.Naverage = Naverage
            update = True

        if update:
            self.update_tasks()

        self.ind_average = 0
        self.data_tot = np.zeros((len(self.channels_ai), self.clock_settings.Nsamples))

        # while not self.controller['ai'].isTaskDone():
        #     self.controller['ai'].stop()
        if self.controller['ai'].c_callback is None:
            self.controller['ai'].register_callback(self.read_data, 'Nsamples', self.clock_settings.Nsamples)

        if self.controller['ai'].isTaskDone():
            self.controller['ai'].task.StartTask()

    def read_data(self, taskhandle, status, samples=0, callbackdata=None):

        if not self.controller['ai'].isTaskDone():   # task is still running => we can read
            data = self.controller['ai'].readAnalog(len(self.channels_ai), self.clock_settings)
            self.ind_average += 1
            self.data_tot += 1 / self.Naverage * data

            if not self.live:   # if snap mode, we stop task
                self.stop()

            if self.ind_average == self.Naverage:   # Ok to emit
                self.emit_data(self.data_tot)
                self.ind_average = 0
                self.data_tot = np.zeros((len(self.channels_ai), self.clock_settings.Nsamples))

        else:   # task has been stopped externally (probably by user)
            self.stop()

        return 0  #mandatory for the PyDAQmx callback

    def emit_data(self, data):
        channels_name = [ch.name for ch in self.channels_ai]

        if self.settings.child('display').value() == '0D':
            data = np.mean(data, 1)

        #data = np.squeeze(data)
        # print(f'shape is {data.shape}')
        # print(data)
        if len(self.channels_ai) == 1 and data.size == 1:
            data_export = [np.array([data[0]])]
        else:
            data_export = [np.squeeze(data[ind, :]) for ind in range(len(self.channels_ai))]

        # print(f'data len is {len(data_export)} and shape is {data_export[0].shape}')
        self.data_grabed_signal.emit([DataFromPlugins(
            name='NI AI',
            data=data_export,
            dim=f'Data{self.settings.child("display").value()}', labels=channels_name)])


    def stop(self):
        try:
            self.controller['ai'].stop()
        except:
            pass
        ##############################

        return ''


if __name__ == '__main__':
    main(__file__)
