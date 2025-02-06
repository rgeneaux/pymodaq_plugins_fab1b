import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, DataToExport, Axis
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from qtpy.QtCore import Qt, QObject, Slot, QThread, Signal
from ctypes import c_ulong
import time

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
from nidaqmx.system.system import System
from nidaqmx.system.device import Device

# Read system properties: device, available AI channels and trigger sources
system = System()
device = Device(system.devices.device_names[0])  #Initialize first device found by System
ai_channels = device.ai_physical_chans.channel_names
triggers = []
if device.dig_trig_supported:
    triggers = [terminal for terminal in device.terminals if 'PFI' in terminal]
if device.anlg_trig_supported:
    triggers.extend(ai_channels)


class DAQ_1DViewer_NIDAQmxFAB1B(DAQ_Viewer_base):
    """
    This is for FAB1B NI DAQ Card, used to read chopper status and other analog signals, when triggered by the camera
    """
    params = comon_parameters+[
        {'title': 'DAQ Settings:', 'name': 'daq_settings', 'type': 'group', 'children':
            [#{'title': 'Display type:', 'name': 'display', 'type': 'list', 'limits': ['0D', '1D'], 'value':'1D'},
            #{'title': 'Axis:', 'name': 'time_axis', 'type': 'list', 'limits': ['Time', 'Samples'], 'value':'Time'},
            {'title': 'Frequency Acq. (kHz):', 'name': 'frequency', 'type': 'int', 'value': 500, 'min': 0.001, 'max': 500.0},
            {'title': 'Nsamples:', 'name': 'Nsamples', 'type': 'int', 'value': 1000, 'default': 100, 'min': 34},
            {'title': 'Chunk size:', 'name': 'chunk_size', 'type': 'int', 'value': 1},
            {'title': 'AI:', 'name': 'ai_channel', 'type': 'list',
             'limits': ai_channels,
             'value': ai_channels[0]},
            {'title': 'Trigger Settings:', 'name': 'trigger_settings', 'type': 'group', 'visible': True, 'children': [
                {'title': 'Enable?:', 'name': 'enable', 'type': 'bool', 'value': False, },
                {'title': 'Trigger Source:', 'name': 'trigger_channel', 'type': 'list',
                 'limits': triggers, 'value': triggers[0]},
                {'title': 'Edge type:', 'name': 'edge', 'type': 'list', 'limits': Edge, 'visible': True},
                #{'title': 'Level:', 'name': 'level', 'type': 'float', 'value': 1., 'visible': True}
            ]}
        ]}
    ]

    def ini_attributes(self):
        self.live = False
        self.data = []

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

        self.daq_controller = self.ini_detector_init(old_controller=controller,
                               new_controller=dict(master_task=nidaqmx.Task()))
        self.update_tasks()
        info = "Analog measurement ready"
        initialized = True

        return info, initialized

    def update_tasks(self):
        #Need to close and restart the task to change its properties
        self.daq_controller['master_task'].close()
        self.daq_controller['master_task']=nidaqmx.Task()

        #Add analog channel
        self.daq_controller['master_task'].ai_channels.add_ai_voltage_chan(self.settings['daq_settings','ai_channel'])

        if self.settings['daq_settings', 'trigger_settings', 'enable']:
            #At each trigger, we will acquire N samples with the chosen frequency
            self.daq_controller['master_task'].timing.cfg_samp_clk_timing(self.settings['daq_settings', 'frequency']*1000, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.settings['daq_settings', 'Nsamples'])
            self.daq_controller['master_task'].triggers.start_trigger.cfg_dig_edge_start_trig(
                self.settings['daq_settings', 'trigger_settings', 'trigger_channel'], trigger_edge=self.settings['daq_settings', 'trigger_settings', 'edge'])
            self.daq_controller['master_task'].triggers.start_trigger.retriggerable = True

        else:
            #We acquire samples continuously
            self.daq_controller['master_task'].timing.cfg_samp_clk_timing(self.settings['daq_settings', 'frequency']*1000, sample_mode=AcquisitionType.CONTINUOUS)

        #Register callback
        self.daq_controller['master_task'].register_every_n_samples_acquired_into_buffer_event(self.settings['daq_settings', 'Nsamples'], self.read_data)

        # Time domain not implemented for now
        # if self.settings['daq_settings','time_axis'] == 'Time':
        #     dt = 1/(self.settings['daq_settings','frequency']*1000)
        #     self.x_axis = Axis(data=np.linspace(0, self.settings['daq_settings','Nsamples'], self.settings['daq_settings','Nsamples'], endpoint=False)*dt,
        #                        label='Time',
        #                        units='s',
        #                        index=0)
        # elif self.settings['daq_settings','time_axis'] == 'Samples':
        #     self.x_axis = Axis(data=np.arange(self.settings['daq_settings','Nsamples']),
        #                        label='Sample',
        #                        units='',
        #                        index=0)


    def close(self):
        """Terminate the communication protocol"""
        self.daq_controller['master_task'].close()

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
        self.n_grabed_chunks = 0
        self.data = []

        if 'live' in kwargs:
            self.live = kwargs['live']

        if self.daq_controller['master_task'].is_task_done():
            self.daq_controller['master_task'].start()  # This will start the acquisition and wait for callback

    def read_data(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        """
        Function triggered by callback, when n_samples are in the buffer.
        Arguments and return follow definition by nidaqmx
        """
        if not self.daq_controller['master_task'].is_task_done():
            new_data = self.daq_controller['master_task'].read(number_of_samples_per_channel=self.settings['daq_settings','Nsamples'])
            if new_data is not []:
                self.data.extend(new_data)
                self.n_grabed_chunks += 1

        if self.n_grabed_chunks == self.settings['daq_settings', 'chunk_size']:
            # We emit the data
            daqcard_data = DataToExport('DAQ Card', data=[DataFromPlugins(name='NI AI',data=[np.asarray(self.data)],
                dim='Data1D')])#, axes=[self.x_axis])])
            self.dte_signal.emit(daqcard_data)
            # Reset counters
            self.n_grabed_chunks = 0
            self.data = []

            if not self.live:
                self.stop()

        return 0


    def stop(self):
        self.daq_controller['master_task'].stop()

if __name__ == '__main__':
    main(__file__)
