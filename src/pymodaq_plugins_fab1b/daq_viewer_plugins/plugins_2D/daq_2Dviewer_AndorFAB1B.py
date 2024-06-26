import time

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.parameter.utils import iter_children

from qtpy import QtWidgets, QtCore
from time import perf_counter
import numpy as np

from pymodaq.utils.logger import set_logger, get_module_name
logger = set_logger(get_module_name(__file__))

from pylablib.devices import Andor
camera_list = [*range(Andor.get_cameras_number_SDK3())]
camera_names_list = dict()
for camera in camera_list:
    cam = Andor.AndorSDK3Camera(idx=camera)
    camera_names_list.update({cam.get_device_info()[1]+' '+cam.get_device_info()[2]: camera})
    cam.close()

from pymodaq_plugins_daqmx.hardware.national_instruments.daqmx import DAQmx, \
    Edge, ClockSettings, Counter, ClockCounter,  TriggerSettings, AIChannel

from PyDAQmx import DAQmx_Val_ContSamps


class DAQ_2DViewer_AndorFAB1B(DAQ_Viewer_base):
    """
    """

    params = comon_parameters + [
        {'title': 'Camera Settings:', 'name': 'camera_settings', 'type': 'group', 'children':
            [{'title': 'Camera:', 'name': 'camera_list', 'type': 'list', 'limits': camera_names_list},
            {'title': 'Acquisition', 'name': 'acq', 'type': 'group', 'children':
                [{'title': 'Acquisition mode:', 'name': 'acq_mode', 'type': 'list', 'limits': ['Normal', 'Fast 1D']},#'Spectrum', 'Differential', 'Sequence'], 'value':'Spectrum'},
                 {'title': 'Fast mode:', 'name': 'fast_mode', 'type': 'list', 'limits': ['Spectrum', 'Differential']},
                 {'title': 'Display:', 'name': 'display', 'type': 'list', 'limits': ['Average', '2D'], 'value':'Average'},
                {'title': 'Differential type:', 'name': 'diff_type', 'type': 'list', 'limits': ['dR/R', 'dOD'], 'visible':False},
                {'title': 'Bit depth:', 'name': 'bit_depth', 'type': 'list', 'limits': []}]},
    
            {'title': 'Image', 'name': 'roi', 'type': 'group', 'children':
                [{'title': 'Height', 'name': 'height', 'type': 'int', 'value': 1},
                 {'title': 'Bottom', 'name': 'bottom', 'type': 'int', 'value': 0},
                 {'title': 'Width', 'name': 'width', 'type': 'int', 'value': 2048},
                 {'title': 'Left', 'name': 'left', 'type': 'int', 'value': 0},
                 {'title': 'Auto Vertical Centering', 'name': 'auto_vert', 'type': 'bool', 'value': False},
                 {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False},
                 {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False},
                 {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2]},]
             },
    
            {'title': 'Timing', 'name': 'timing_opts', 'type': 'group', 'children':
                [{'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 0.13},
                 {'title': 'Chunk size', 'name': 'chunk_size', 'type': 'int', 'value': 1000},
                 {'title': 'Compute FPS', 'name': 'fps_on', 'type': 'bool', 'value': True},
                 {'title': 'Actual FPS', 'name': 'fps', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 6},
                 {'title': 'Max FPS', 'name': 'fps2', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 6}]
             },
            {'title': 'Trigger Settings:', 'name': 'trigger', 'type': 'group', 'children': [
                {'title': 'Mode:', 'name': 'trigger_mode', 'type': 'list', 'limits': [], 'value': 'External'},
                {'title': 'Software Trigger:', 'name': 'soft_trigger', 'type': 'bool_push', 'value': False,
                 'label': 'Fire', 'visible': False},
                {'title': 'External Trigger delay (ms):', 'name': 'ext_trigger_delay', 'type': 'float', 'value': 0.,'visible': False},
            ]},
            {'title': 'Developer Settings:', 'name': 'dev', 'type': 'group', 'children': [
                {'title': 'Show Timestamps', 'name': 'timestamps_on', 'type': 'bool', 'value': False},
                {'title': 'Show Pump On/Off', 'name': 'pumponoff_on', 'type': 'bool', 'value': False},
            ]},
            # {'title': 'DAQCard Settings:', 'name': 'daqcard_settings', 'type': 'group', 'children': [
            #     {'title': 'Display type:', 'name': 'display', 'type': 'list', 'limits': ['0D', '1D'], 'value':'1D'},
            #     {'title': 'Axis:', 'name': 'time_axis', 'type': 'list', 'limits': ['Time', 'Samples'], 'value':'Time'},
            #     {'title': 'Frequency Acq. (kS):', 'name': 'frequency', 'type': 'int', 'value': 500, 'min': 0.001, 'max': 500.0},
            #     {'title': 'Nsamples:', 'name': 'Nsamples', 'type': 'int', 'value': 1000, 'default': 100, 'min': 1},
            #     {'title': 'AI:', 'name': 'ai_channel', 'type': 'list',
            #      'limits': DAQmx.get_NIDAQ_channels(source_type='Analog_Input'),
            #      'value': DAQmx.get_NIDAQ_channels(source_type='Analog_Input')[0]},
            #     {'title': 'Trigger Settings:', 'name': 'trigger_settings', 'type': 'group', 'visible': True, 'children': [
            #         {'title': 'Enable?:', 'name': 'enable', 'type': 'bool', 'value': True, },
            #         {'title': 'Trigger Source:', 'name': 'trigger_channel', 'type': 'list',
            #          'limits': DAQmx.getTriggeringSources(), 'value': DAQmx.getTriggeringSources()[0]},
            #         {'title': 'Edge type:', 'name': 'edge', 'type': 'list', 'limits': Edge.names(), 'visible': True},
            #         {'title': 'Level:', 'name': 'level', 'type': 'float', 'value': 1., 'visible': True}
            #     ]}
            ]}
    ]
    start_waitloop = QtCore.Signal()
    stop_waitloop = QtCore.Signal()
    roi_pos_size = QtCore.QRectF(0,0,10,10)
    axes = []
    live = False
    n_grabed_frames = 0
    data = None
    timestamps = []
    timestamp_frequency = 0

    def init_controller(self):
        return Andor.AndorSDK3Camera(idx=self.settings["camera_settings","camera_list"])

    def ini_attributes(self):
        self.controller: None

        self.x_axis = None
        self.y_axis = None
        self.last_tick = 0.0  # time counter used to compute FPS
        self.fps = 0.0

        self.data_shape = 'Data2D'
        self.callback_thread = None


    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "exposure_time":
            self.controller.set_attribute_value("ExposureTime", param.value() / 1000)
            self.settings.child("camera_settings",'timing_opts', 'exposure_time').setValue(self.controller.get_attribute_value("ExposureTime")*1000)
            self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.controller.get_attribute_value('FrameRate'))

        if param.name() == "bit_depth":
            self.controller.set_attribute_value("PixelEncoding",param.value())

        if param.name() in ['display', 'fast_mode']:
            self._prepare_view()

        if param.name() == "fps_on":
            self.settings.child("camera_settings",'timing_opts', 'fps').setOpts(visible=param.value())
            self.settings.child("camera_settings",'timing_opts', 'fps2').setOpts(visible=param.value())

        if param.name() == "update_roi":
            if param.value():  # Switching on ROI

                # We handle ROI and binning separately for clarity
                (old_x, _, old_y, _, xbin, ybin) = self.controller.get_roi()  # Get current binning

                x0 = self.roi_pos_size.x()
                y0 = self.roi_pos_size.y()
                width = self.roi_pos_size.width()
                height = self.roi_pos_size.height()

                # Values need to be rescaled by binning factor and shifted by current x0,y0 to be correct.
                new_x = (old_x + x0) * xbin
                new_y = (old_y + y0) * xbin
                new_width = width * ybin
                new_height = height * ybin

                new_roi = (new_x, new_width, xbin, new_y, new_height, ybin)
                self.update_rois(new_roi)

                param.setValue(False)

        if param.name() in iter_children(self.settings.child("camera_settings",'roi'), []):
            new_roi = self.get_roi_from_settings()
            self.update_rois(new_roi)

        if param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child("camera_settings",'roi','binning').value()
            ybin = self.settings.child("camera_settings",'roi','binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)

        if param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                self.clear_roi()
                param.setValue(False)

        if param.name() == 'timestamps_on':
            self._prepare_view()

        elif param.name() in iter_children(self.settings.child("camera_settings",'trigger'), []):
            self.set_trigger()

        if param.name() == 'pumponoff_on':
            self._prepare_view()

        if param.name() == 'acq_mode':
            self.set_acq_mode()

    def ROISelect(self, roi_pos_size):
        self.roi_pos_size = roi_pos_size

    def clear_roi(self):
        wdet, hdet = self.controller.get_detector_size()
        self.settings.child("camera_settings",'roi','binning').setValue(1)
        new_roi = (0, wdet, 1, 0, hdet, 1)
        self.update_rois(new_roi)

    def set_acq_mode(self):
        mode = self.settings["camera_settings",'acq','acq_mode']
        if mode == 'Normal':
            self.settings.child("camera_settings",'timing_opts', 'chunk_size').hide()
            self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('Internal')
            self.set_trigger()
            self.settings.child("camera_settings",'dev').hide()
            self.settings.child("camera_settings",'acq','fast_mode').hide()
            self.settings.child("camera_settings",'acq','diff_type').hide()
            self.settings.child("camera_settings",'acq','display').hide()

        else:
            self.settings.child("camera_settings",'acq','fast_mode').show()

            fast_mode = self.settings["camera_settings",'acq','fast_mode']

            self.settings.child("camera_settings",'timing_opts', 'chunk_size').show()
            self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('External')
            self.set_trigger()
            self.settings.child("camera_settings",'dev').show()
            self.settings.child("camera_settings",'acq','display').show()

            if fast_mode == 'Differential':
                self.settings.child("camera_settings",'acq','diff_type').show()
            else:
                self.settings.child("camera_settings",'acq','diff_type').hide()


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
        # Initialize camera class
        self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())

        # Choose data type
        # self.controller.set_frame_format("array")
        self.controller.set_frame_format("list")

        # Set bit depth
        self.settings.child("camera_settings",'acq','bit_depth').setOpts(limits=self.controller.get_attribute('PixelEncoding').values)
        self.settings.child("camera_settings",'acq','bit_depth').setOpts(value=self.controller.get_attribute_value('PixelEncoding'))

        # Set exposure time
        self.controller.set_exposure(self.settings.child("camera_settings",'timing_opts', 'exposure_time').value() / 1000)
        attr = self.controller.get_attribute('ExposureTime')
        self.settings.child("camera_settings",'timing_opts', 'exposure_time').setLimits((attr.min * 1000, attr.max * 1000))

        # FPS visibility
        self.settings.child("camera_settings",'timing_opts', 'fps').setOpts(visible=self.settings.child("camera_settings",'timing_opts', 'fps_on').value())

        # Update image parameters
        new_roi = self.get_roi_from_settings()
        self.update_rois(new_roi)

        # Enable Metadata in order to get Frame timestamps
        self.controller.enable_metadata()
        self.controller.call_command("TimestampClockReset")
        self.timestamp_frequency = self.controller.get_attribute_value("TimestampClockFrequency")
        # print(f'{self.controller.get_full_info("all")}')

        self.set_acq_mode()
        self.setup_callback()
        self._prepare_view()
        self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('External') # not very clean
        self.setup_trigger()

        info = "Initialized camera"
        initialized = True
        return info, initialized


    def setup_callback(self):

        if self.callback_thread is not None:
            if self.callback_thread.isRunning():
                self.callback_thread.terminate()

        callback = PylablibCallback(self.controller.wait_for_frame)
        self.callback_thread = QtCore.QThread()
        callback.moveToThread(self.callback_thread)
        callback.data_sig.connect(
            self.emit_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired

        self.start_waitloop.connect(callback.start)
        self.stop_waitloop.connect(callback.stop)
        self.callback_thread.callback = callback
        self.callback_thread.start()

    def setup_trigger(self):
        self.settings.child("camera_settings",'trigger', 'trigger_mode').setLimits(self.controller.get_attribute("TriggerMode").values)
        self.set_trigger()

    def set_trigger(self):
        self.controller.set_attribute_value("TriggerMode", self.settings.child("camera_settings",'trigger', 'trigger_mode').value())
        if self.settings["camera_settings",'trigger', 'trigger_mode'] == 'Software':
            self.settings.child("camera_settings",'trigger', 'soft_trigger').show()
        else:
            self.settings.child("camera_settings",'trigger', 'soft_trigger').hide()


    def _prepare_view(self):
        dte = self.generate_dte_temp()
        # init the viewers
        self.dte_signal_temp.emit(DataToExport('Camera', data=dte))
        QtWidgets.QApplication.processEvents()


    def generate_dte_temp(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs or acquisition modes are changed"""
        (hstart, hend, vstart, vend, *_) = self.controller.get_roi()
        height = vend - vstart
        width = hend - hstart

        self.settings.child("camera_settings",'roi','width').setValue(width)
        self.settings.child("camera_settings",'roi','height').setValue(height)
        self.settings.child("camera_settings",'roi', 'left').setValue(hstart)
        self.settings.child("camera_settings",'roi', 'bottom').setValue(vstart)
        mock_data = np.zeros((width, height))

        self.x_axis = Axis(data=np.linspace(0,width,width, endpoint=False), label='Pixels', index=1)

        if self.settings["camera_settings",'acq','acq_mode'] == 'Normal':   # Normal mode
            if height != 1: # we have a 2D image
                data_shape = 'Data2D'
                self.y_axis = Axis(data=np.linspace(0, height, height, endpoint=False), label='Pixels', index=0)
                self.axes = [self.x_axis, self.y_axis]
            else: # 1D spectrum
                data_shape = 'Data1D'
                self.x_axis.index = 0
                self.axes = [self.x_axis]

        else:  # FAST MODE
            if self.settings["camera_settings",'acq','display'] == '2D':   # spectra are shown in 2D
                data_shape = 'Data2D'
                nchunk = self.settings["camera_settings",'timing_opts','chunk_size']
                if self.settings["camera_settings",'acq','acq_mode'] == 'Differential':
                    nchunk = int(nchunk/2)
                self.y_axis = Axis(data=np.linspace(0, nchunk, nchunk, endpoint=False), label='Shot', index=0)
                self.axes = [self.x_axis, self.y_axis]
                mock_data = np.zeros((width, nchunk))

            else: # this is in 1D:
                data_shape = 'Data1D'
                self.x_axis.index = 0
                self.axes = [self.x_axis]

        self.data_shape = data_shape
        dte = [DataFromPlugins(name='Camera Image',
                               data=[np.squeeze(mock_data)],
                               dim=self.data_shape,
                               labels=[f'Camera_{data_shape}'],
                               axes=self.axes)]

        if self.settings["camera_settings",'acq','acq_mode'] == 'Fast 1D':   # in FAST MODE we can have additional plots
            # Extra plots:
            timestamps = self.settings["camera_settings",'dev', 'timestamps_on']
            ponoff = self.settings["camera_settings",'dev','pumponoff_on']

            if timestamps:
                taxis = Axis(data=np.arange((self.settings["camera_settings",'timing_opts', 'chunk_size'])), label='Time', index=0)
                timestamp_data = DataFromPlugins(name='Timestamps',
                                      data=[np.zeros((self.settings["camera_settings",'timing_opts', 'chunk_size']))],
                                      axes=[taxis],
                                      dim='Data1D')
                dte.append(timestamp_data)

            if ponoff:
                if self.settings["camera_settings",'acq','acq_mode'] == 'Differential' and self.settings["camera_settings",'acq','display'] == 'Average':
                    dte.append(DataFromPlugins(name='Pump Off/On',
                                           data=[np.squeeze(mock_data), np.squeeze(mock_data)],
                                           dim='Data1D',
                                           labels=['Pump Off', 'Pump On'],
                                           axes=self.axes))
        return dte


    def get_roi_from_settings(self):
        x0 = self.settings["camera_settings",'roi', 'left']
        y0 = self.settings["camera_settings",'roi', 'bottom']
        width = self.settings["camera_settings",'roi', 'width']
        height = self.settings["camera_settings",'roi', 'height']

        if self.settings["camera_settings",'roi', 'auto_vert']:
            (_, detector_height) = self.controller.get_detector_size()
            y0 = round(detector_height/2 - height/2)

        # We handle ROI and binning separately for clarity
        (*_, xbin, ybin) = self.controller.get_roi()  # Get current binning

        return x0, width, xbin, y0, height, ybin


    def update_rois(self, new_roi):
        # In pylablib, ROIs compare as tuples
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.controller.get_roi():
            # self.controller.set_attribute_value("ROIs",[new_roi])
            self.controller.set_roi(hstart=new_x, hend=new_x + new_width, vstart=new_y, vend=new_y + new_height,
                                    hbin=new_xbinning, vbin=new_ybinning)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.controller.clear_acquisition()
            self.controller.setup_acquisition()
            # Finally, prepare view for displaying the new data

            self.settings["camera_settings",'roi', 'left'] = new_x
            self.settings["camera_settings",'roi', 'bottom'] = new_y
            self.settings["camera_settings",'roi', 'width'] = new_width
            self.settings["camera_settings",'roi', 'height'] = new_height
            self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.controller.get_attribute_value('FrameRate'))
            self._prepare_view()

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data.
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """
        self.n_grabed_frames = 0
        self.data = None
        self.timestamps = []

        if 'live' in kwargs:
            self.live = kwargs['live']

        try:
            # Warning, acquisition_in_progress returns 1,0 and not a real bool
            if not self.controller.acquisition_in_progress():
                self.controller.clear_acquisition()
                self.controller.start_acquisition()

            # Then start the acquisition
            self.start_waitloop.emit()  # will trigger the wait for acquisition

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data(self):
        """
            Fonction used to emit data obtained by callback.
            We put the generate dte in separate function to help subclassing.
        """
        try:
            dte, do_emit = self.generate_dte_real()

            #SNAP MODE: when frame is ready, stop acquisition
            if not self.live:
                if do_emit:
                    self.stop()

            # Emit the frame.
            if do_emit:
                self.dte_signal.emit(DataToExport('Camera', data=dte))

                if self.settings.child("camera_settings",'timing_opts', 'fps_on').value():
                    self.update_fps()

            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def generate_dte_real(self):
        do_emit = False

        # CASE 1 : Normal acquision regardless of size
        if self.settings["camera_settings",'acq','acq_mode'] == 'Normal':
            # Trying to read and average several frames but it does not work:
            # in internal trigger, it just gets one frame,
            # in external trigger, it gets several but then the buffer overflows.
            frames = self.controller.read_multiple_images(return_info=False)
            if frames is not None:
                if len(frames)>0:
                    self.data = sum(frames)/len(frames)
                    do_emit = True

        # CASE 2 : Spectrum or Differential Acquisition
        elif self.settings["camera_settings",'acq','acq_mode'] in ['Spectrum', 'Differential']:
            # Read all frames in buffer together with timestamps
            frames, info = self.controller.read_multiple_images(return_info=True)
            if frames is not None:
                if len(frames)>0:    #happens sometimes for some reason
                    if np.squeeze(frames[0]).ndim ==2:       #if each frame is a 2D image
                        frames = [np.mean(frame, axis=0) for frame in frames]    # Software full vertical binning. frames size = [nframes, 2048]

                    remaining_frames = self.settings["camera_settings",'timing_opts', 'chunk_size'] - self.n_grabed_frames

                    # If we have more frames than chunk size we drop the extra
                    if len(frames) > remaining_frames:
                        frames = frames[:remaining_frames]
                        info = info[:remaining_frames]

                    if len(frames) == 0:    # if we already have everything
                        return

                    #Add frames to the list
                    if len(frames) > 1:
                        self.n_grabed_frames += len(frames)    # Increment number of read frames
                        if self.data is None:
                            self.data = frames
                        else:
                            self.data.append(frames)

                    # Store timestamps in ms
                    if self.settings["camera_settings",'dev', 'timestamps_on']:
                        # Save timestamps in ms:
                        self.timestamps.extend(info[:, 1]/self.timestamp_frequency*1000)

                    # If we have enough for the chunk,
                    if self.n_grabed_frames >= self.settings["camera_settings",'timing_opts', 'chunk_size']:
                        # Flatten the list of lists and convert to numpy
                        self.data = np.vstack([x for xs in self.data for x in xs])

                        if self.settings["camera_settings",'acq','acq_mode'] == 'Spectrum':
                            if self.settings["camera_settings",'acq','display'] == 'Average':
                                self.data = np.sum(self.data, axis=0) / self.n_grabed_frames   # divide for average

                        elif self.settings["camera_settings",'acq','acq_mode'] == 'Differential':
                            tmp = self.data
                            pon = tmp[0::2]
                            poff = tmp[1::2]
                            poff[poff==0] = 1e-10

                            if self.settings["camera_settings",'acq','diff_type'] == 'dR/R':
                                self.data = (pon-poff)/poff
                            elif self.settings["camera_settings",'acq','diff_type'] == 'dOD':
                                self.data = -np.real(np.log(pon/poff))

                            self.data[np.isnan(self.data)] = 0
                            self.data[np.isinf(self.data)] = 0

                            if self.settings["camera_settings",'acq','display'] == 'Average':
                                self.data = np.nanmean(self.data, axis=0)
                                pon = np.nanmean(pon, axis=0)
                                poff = np.nanmean(poff, axis=0)

                        do_emit = True


        dte = [DataFromPlugins(name='Camera Image',
                               data=[np.squeeze(self.data)],
                               dim=self.data_shape,
                               labels=[f'Camera'],
                               axes=self.axes)]

        if self.settings["camera_settings",'acq','acq_mode'] == 'Differential' and self.settings["camera_settings",'acq','display'] == 'Average' and self.settings["camera_settings",'dev','pumponoff_on']:
            dte.append(DataFromPlugins(name='Pump On/Off',
                                       data=[np.squeeze(poff), np.squeeze(pon)],
                                       dim=self.data_shape,
                                       labels=['Pump Off', 'Pump On'],
                                       axes=self.axes))

        if self.timestamps:
            taxis = Axis(data=np.arange(1,1+len(self.timestamps)), label="Frame", units="")
            taxis.index = 0
            dte.append(DataFromPlugins(name='Timestamps',
                                       data=[np.asarray(self.timestamps)-np.min(self.timestamps)],
                                       dim='Data1D',
                                       axes=[taxis],
                                       label='Timestamps (ms)'))
        return dte, do_emit

    def update_fps(self):
        current_tick = perf_counter()
        frame_time = current_tick - self.last_tick

        if self.last_tick != 0.0 and frame_time != 0.0:
            # We don't update FPS for the first frame, and we also avoid divisions by zero

            if self.fps == 0.0:
                self.fps = 1 / frame_time
            else:
                # If we already have an FPS calculated, we smooth its evolution
                self.fps = 0.7 * self.fps + 0.3 / frame_time

        self.last_tick = current_tick

        # Update reading
        if self.live and self.settings["camera_settings",'acq','acq_mode'] in ['Spectrum', 'Differential']:
            scaling = self.settings["camera_settings",'timing_opts', 'chunk_size']
        else:
            scaling = 1
        self.settings.child("camera_settings",'timing_opts', 'fps').setValue(round(self.fps * scaling, 1))
        self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.controller.get_attribute_value('FrameRate'))

    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        raise NotImplementedError


   ##############################################
    # DAQ CARD
   #################################################

    def init_daqcard(self):
        self.daqcontroller = dict(ai=DAQmx())

        # Create channels
        self.channels_ai = [AIChannel(name=self.settings.child("camera_settings",'ai_channel').value(),
                                      source='Analog_Input', analog_type='Voltage',
                                      value_min=-10., value_max=10., termination='Diff', ),
                            ]

        self.clock_settings = ClockSettings(frequency=self.settings["camera_settings",'frequency']*1000,
                                            Nsamples=self.settings["camera_settings",'Nsamples'],
                                            repetition=self.live,)

        self.trigger_settings = \
            TriggerSettings(trig_source=self.settings["camera_settings",'trigger_settings', 'trigger_channel'],
                            enable=self.settings["camera_settings",'trigger_settings', 'enable'],
                            edge=self.settings["camera_settings",'trigger_settings', 'edge'],
                            level=self.settings["camera_settings",'trigger_settings', 'level'],)

        self.controller['ai'].update_task(self.channels_ai, self.clock_settings, trigger_settings=self.trigger_settings)



    def close(self):
        """
        Terminate the communication protocol
        """
        # Terminate the communication
        self.controller.close()
        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""

    def stop(self):
        """Stop the acquisition."""
        self.stop_waitloop.emit()
        self.controller.stop_acquisition()
        self.controller.clear_acquisition()
        frames = self.controller.read_multiple_images() # read all images still in memory to remove them
        return ''


class PylablibCallback(QtCore.QObject):
    """Callback object """
    data_sig = QtCore.Signal()

    def __init__(self, wait_fn):
        super().__init__()
        # Set the wait function
        self.wait_fn = wait_fn
        self.running = False

    def start(self, nframes=1, wait_time=1):
        self.running = True
        self.wait_for_acquisition(nframes, wait_time)

    def stop(self):
        self.running = False

    def wait_for_acquisition(self, nframes, wait_time):
        while True:
            if not self.running:
                break
            new_data = self.wait_fn(nframes=nframes)
            if new_data is not False:
                self.data_sig.emit()
                QtCore.QThread.msleep(wait_time)


if __name__ == '__main__':
    main(__file__)