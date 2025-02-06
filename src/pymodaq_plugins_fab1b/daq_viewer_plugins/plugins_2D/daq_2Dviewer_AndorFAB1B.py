import time

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.parameter.utils import iter_children

from qtpy import QtWidgets
from qtpy.QtCore import Slot, Signal, QRectF, QObject, QThread, QTimer
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
                {'title': 'Differential type:', 'name': 'diff_type', 'type': 'list', 'limits': ['dR/R', 'dOD']},
                {'title': 'Bit depth:', 'name': 'bit_depth', 'type': 'list', 'limits': []}]},
    
            {'title': 'Image', 'name': 'roi', 'type': 'group', 'children':
                [{'title': 'Height', 'name': 'height', 'type': 'int', 'value': 2048},
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
                {'title': 'Mode:', 'name': 'trigger_mode', 'type': 'list', 'limits': [], 'value': 'Internal'},
                {'title': 'Software Trigger:', 'name': 'soft_trigger', 'type': 'bool_push', 'value': False,
                 'label': 'Fire', 'visible': False},
                {'title': 'External Trigger delay (ms):', 'name': 'ext_trigger_delay', 'type': 'float', 'value': 0.,'visible': False},
            ]},
            {'title': 'Developer Settings:', 'name': 'dev', 'type': 'group', 'children': [
                {'title': 'Show Timestamps', 'name': 'timestamps_on', 'type': 'bool', 'value': False},
                {'title': 'Show Pump On/Off', 'name': 'pumponoff_on', 'type': 'bool', 'value': False},
            ]},
             {'title': 'Temperature Settings:', 'name': 'temperature_settings', 'type': 'group', 'children': [
                 {'title': 'Enable Cooling:', 'name': 'enable_cooling', 'type': 'bool', 'value': False},
                 {'title': 'Set Point:', 'name': 'set_point', 'type': 'list', 'limits': []},
                 {'title': 'Current value:', 'name': 'current_value', 'type': 'float', 'value': 20, 'readonly': True},
             ]},
            ]}
    ]
    start_waitloop = Signal()
    stop_waitloop = Signal()
    roi_pos_size = QRectF(0,0,10,10)
    axes = []
    live = False
    n_grabed_frames = 0
    data = None
    timestamps = []
    timestamp_frequency = 0
    #live_mode_available = True
    #hardware_averaging = False

    def init_controller(self):
        return Andor.AndorSDK3Camera(idx=self.settings["camera_settings","camera_list"])

    def ini_attributes(self):
        self.camera_controller: None

        self.x_axis = None
        self.y_axis = None
        self.last_tick = 0.0  # time counter used to compute FPS
        self.fps = 0.0

        self.data_shape = 'Data2D'
        self.buffer_size = 500
        self.callback_thread = None

        self.temperature_timer = QTimer()
        self.temperature_timer.timeout.connect(self.update_temperature)
        self.temp_freq = 2000 # Frequency of temperature timer in ms

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == 'set_point':
            self.camera_controller.set_temperature(param.value(), enable_cooler=False)

        elif param.name() == "exposure_time":
            self.camera_controller.set_attribute_value("ExposureTime", param.value() / 1000)
            self.settings.child("camera_settings",'timing_opts', 'exposure_time').setValue(self.camera_controller.get_attribute_value("ExposureTime")*1000)
            self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.camera_controller.get_attribute_value('FrameRate'))

        elif param.name() == "bit_depth":
            self.camera_controller.set_attribute_value("PixelEncoding",param.value())

        elif param.name() in ['display', 'fast_mode']:
            self._prepare_view()

        elif param.name() == "fps_on":
            self.settings.child("camera_settings",'timing_opts', 'fps').setOpts(visible=param.value())
            self.settings.child("camera_settings",'timing_opts', 'fps2').setOpts(visible=param.value())

        elif param.name() == "update_roi":
            if param.value():  # Switching on ROI

                # We handle ROI and binning separately for clarity
                (old_x, _, old_y, _, xbin, ybin) = self.camera_controller.get_roi()  # Get current binning

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

        elif param.name() in iter_children(self.settings.child("camera_settings",'roi'), []):
            new_roi = self.get_roi_from_settings()
            self.update_rois(new_roi)

        elif param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.camera_controller.get_roi()  # Get current ROI
            xbin = self.settings.child("camera_settings",'roi','binning').value()
            ybin = self.settings.child("camera_settings",'roi','binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)

        elif param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                self.clear_roi()
                param.setValue(False)

        elif param.name() == 'timestamps_on':
            self._prepare_view()

        elif param.name() in iter_children(self.settings.child("camera_settings",'trigger'), []):
            self.set_trigger()

        elif param.name() in iter_children(self.settings.child('camera_settings', 'temperature_settings'), []):
            self.setup_temperature()

        elif param.name() == 'pumponoff_on':
            self._prepare_view()

        elif param.name() == 'acq_mode':
            self.set_acq_mode()

    def ROISelect(self, roi_pos_size):
        self.roi_pos_size = roi_pos_size

    def clear_roi(self):
        wdet, hdet = self.camera_controller.get_detector_size()
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
        self.camera_controller = self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())

        # Choose data type
        # self.camera_controller.set_frame_format("array")
        self.camera_controller.set_frame_format("list")

        #Ring buffer size
        self.camera_controller.setup_acquisition(mode="sequence", nframes=self.buffer_size)

        # Set bit depth
        self.settings.child("camera_settings",'acq','bit_depth').setOpts(limits=self.camera_controller.get_attribute('PixelEncoding').values)
        self.settings.child("camera_settings",'acq','bit_depth').setOpts(value=self.camera_controller.get_attribute_value('PixelEncoding'))

        # Set exposure time
        self.camera_controller.set_exposure(self.settings.child("camera_settings",'timing_opts', 'exposure_time').value() / 1000)
        attr = self.camera_controller.get_attribute('ExposureTime')
        self.settings.child("camera_settings",'timing_opts', 'exposure_time').setLimits((attr.min * 1000, attr.max * 1000))

        # FPS visibility
        self.settings.child("camera_settings",'timing_opts', 'fps').setOpts(visible=self.settings.child("camera_settings",'timing_opts', 'fps_on').value())

        # Update image parameters
        new_roi = self.get_roi_from_settings()
        self.update_rois(new_roi)

        # Enable Metadata in order to get Frame timestamps
        self.camera_controller.enable_metadata()
        self.camera_controller.call_command("TimestampClockReset")
        self.timestamp_frequency = self.camera_controller.get_attribute_value("TimestampClockFrequency")
        # print(f'{self.camera_controller.get_full_info("all")}')

        self.set_acq_mode()
        self.setup_callback()
        self._prepare_view()
        self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('Internal') # not very clean
        self.setup_trigger()
        self.setup_temperature()

        info = "Initialized camera"
        initialized = True
        return info, initialized


    def setup_callback(self):

        if self.callback_thread is not None:
            if self.callback_thread.isRunning():
                self.callback_thread.terminate()

        callback = PylablibCallback(self.camera_controller.wait_for_frame)
        self.callback_thread = QThread()
        callback.moveToThread(self.callback_thread)
        callback.data_sig.connect(
            self.emit_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired

        self.start_waitloop.connect(callback.start)
        self.stop_waitloop.connect(callback.stop)
        self.callback_thread.callback = callback
        self.callback_thread.start()

    def setup_trigger(self):
        self.settings.child("camera_settings",'trigger', 'trigger_mode').setLimits(self.camera_controller.get_attribute("TriggerMode").values)
        self.set_trigger()

    def set_trigger(self):
        self.camera_controller.set_attribute_value("TriggerMode", self.settings.child("camera_settings",'trigger', 'trigger_mode').value())
        if self.settings["camera_settings",'trigger', 'trigger_mode'] == 'Software':
            self.settings.child("camera_settings",'trigger', 'soft_trigger').show()
        else:
            self.settings.child("camera_settings",'trigger', 'soft_trigger').hide()

    def setup_temperature(self):
        enable = self.settings.child('camera_settings', 'temperature_settings', 'enable_cooling').value()
        self.camera_controller.set_cooler(on=enable)
        if not self.temperature_timer.isActive():
            self.temperature_timer.start(self.temp_freq)  # Timer event fired every 2s
        if enable:
            #if self.camera_controller.TemperatureControl.isWritable():
            #    self.camera_controller.TemperatureControl.setString(self.settings.child('camera_settings', 'temperature_settings', 'set_point').value())
            self.update_temperature()
            # set timer to update temperature info from controller

    def update_temperature(self):
        """
        update temperature status and value. Fired using the temperature_timer every 2s when not grabbing
        """
        temp = self.camera_controller.get_temperature()
        self.settings.child('camera_settings', 'temperature_settings', 'current_value').setValue(temp)


    def _prepare_view(self):
        dte = self.generate_dte_temp()
        # init the viewers
        self.dte_signal_temp.emit(DataToExport('Camera', data=dte))
        QtWidgets.QApplication.processEvents()


    def generate_dte_temp(self):
        # TODO: pon/off if required

        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs or acquisition modes are changed"""
        (hstart, hend, vstart, vend, *_) = self.camera_controller.get_roi()
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
            (_, detector_height) = self.camera_controller.get_detector_size()
            y0 = round(detector_height/2 - height/2)

        # We handle ROI and binning separately for clarity
        (*_, xbin, ybin) = self.camera_controller.get_roi()  # Get current binning

        return x0, width, xbin, y0, height, ybin


    def update_rois(self, new_roi):
        # In pylablib, ROIs compare as tuples
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.camera_controller.get_roi():
            # self.camera_controller.set_attribute_value("ROIs",[new_roi])
            self.camera_controller.set_roi(hstart=new_x, hend=new_x + new_width, vstart=new_y, vend=new_y + new_height,
                                    hbin=new_xbinning, vbin=new_ybinning)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.camera_controller.clear_acquisition()
            self.camera_controller.setup_acquisition()
            # Finally, prepare view for displaying the new data

            self.settings["camera_settings",'roi', 'left'] = new_x
            self.settings["camera_settings",'roi', 'bottom'] = new_y
            self.settings["camera_settings",'roi', 'width'] = new_width
            self.settings["camera_settings",'roi', 'height'] = new_height
            self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.camera_controller.get_attribute_value('FrameRate'))
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
        self.temperature_timer.stop() #Stop temperature reading during acquisition


        if 'live' in kwargs:
            self.live = kwargs['live']

        try:
            # Warning, acquisition_in_progress returns 1,0 and not a real bool
            if not self.camera_controller.acquisition_in_progress():
                self.camera_controller.clear_acquisition()
                self.camera_controller.setup_acquisition(mode="sequence", nframes=self.buffer_size)
                self.camera_controller.start_acquisition()

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
                self.dte_signal.emit(dte)

                if self.settings.child("camera_settings",'timing_opts', 'fps_on').value():
                    self.update_fps()

            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def generate_dte_real(self):
        dfp_list = []
        dte = DataToExport(name='Andor', data=[])
        do_emit = False

        # CASE 1 : Normal acquision regardless of size
        if self.settings["camera_settings",'acq','acq_mode'] == 'Normal':
            # Trying to read and average several frames but it does not work:
            # in internal trigger, it just gets one frame,
            # in external trigger, it gets several but then the buffer overflows.
            frames = self.camera_controller.read_multiple_images(return_info=False)
            if frames is not None:
                if len(frames)>0:
                    self.data = sum(frames)/len(frames)
                    do_emit = True

        # CASE 2 : Spectrum or Differential Acquisition
        elif self.settings["camera_settings",'acq','acq_mode'] == 'Fast 1D':
            # Read all frames in buffer together with timestamps
            frames, info = self.camera_controller.read_multiple_images(return_info=True)

            if frames is not None:
                if len(frames)>0:    # = 0 happens sometimes for some reason
                    if np.squeeze(frames[0]).ndim ==2:       #if each frame is a 2D image
                        frames = [np.mean(frame, axis=0) for frame in frames]    # Software full vertical binning. frames size = [nframes, 2048]

                    if len(frames) == self.buffer_size:
                        logger.warning("Frame buffer is full ("+str(len(frames))+" frames) - consider increasing its size")

                    remaining_frames = self.settings["camera_settings",'timing_opts', 'chunk_size'] - self.n_grabed_frames

                    # If we have more frames than chunk size we drop the extra
                    if len(frames) > remaining_frames:
                        frames = frames[:remaining_frames]
                        info = info[:remaining_frames]

                    #Add frames to the list
                    if len(frames) >= 1:
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
                        # Flatten the list of lists and convert to numpy. Convert to floats for divisions etc.
                        self.data = np.vstack([x for xs in self.data for x in xs]).astype(float)

                        if self.settings["camera_settings",'acq','fast_mode'] == 'Spectrum':
                            if self.settings["camera_settings",'acq','display'] == 'Average':
                                self.data = np.sum(self.data, axis=0) / self.n_grabed_frames   # divide for average

                        elif self.settings["camera_settings",'acq','fast_mode'] == 'Differential':
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

        if do_emit:
            dfp_list = [DataFromPlugins(name='Camera Image',
                                   data=[np.squeeze(self.data)],
                                   dim=self.data_shape,
                                   labels=[f'Camera'],
                                   axes=self.axes)]

            if self.settings["camera_settings",'acq','fast_mode'] == 'Differential' and self.settings["camera_settings",'dev','pumponoff_on']:
                     dfp_list.append(DataFromPlugins(name='Pump On/Off',
                                                data=[np.squeeze(poff), np.squeeze(pon)],
                                                dim=self.data_shape,
                                                labels=['Pump Off', 'Pump On'],
                                                axes=self.axes))

            if self.timestamps:
                taxis = Axis(data=np.arange(1,1+len(self.timestamps)), label="Frame", units="")
                taxis.index = 0
                dfp_list.append(DataFromPlugins(name='Timestamps',
                                           data=[np.asarray(self.timestamps)-np.min(self.timestamps)],
                                           dim='Data1D',
                                           axes=[taxis],
                                           label='Timestamps (ms)'))

            dte = DataToExport(name='Andor', data=dfp_list)
            # Reset counters and variables
            self.n_grabed_frames = 0
            self.data = None
            self.timestamps = []

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
        if self.live and self.settings["camera_settings",'acq','acq_mode'] == 'Fast 1D':
            scaling = self.settings["camera_settings",'timing_opts', 'chunk_size']
        else:
            scaling = 1
        self.settings.child("camera_settings",'timing_opts', 'fps').setValue(round(self.fps * scaling, 1))
        self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.camera_controller.get_attribute_value('FrameRate'))


    def close(self):
        """
        Terminate the communication protocol
        """
        # Terminate the communication
        self.temperature_timer.stop()
        self.camera_controller.close()
        self.camera_controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""

    def stop(self):
        """Stop the acquisition."""
        self.stop_waitloop.emit()
        self.camera_controller.stop_acquisition()
        self.camera_controller.clear_acquisition()
        frames = self.camera_controller.read_multiple_images() # read all images still in memory to remove them
        self.temperature_timer.start(self.temp_freq)

        return ''


class PylablibCallback(QObject):
    """Callback object """
    data_sig = Signal()

    def __init__(self, wait_fn):
        super().__init__()
        # Set the wait function
        self.wait_fn = wait_fn
        self.running = False

    def start(self, nframes=1, wait_time=10):
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
                QThread.msleep(wait_time)


if __name__ == '__main__':
    main(__file__, init=False)