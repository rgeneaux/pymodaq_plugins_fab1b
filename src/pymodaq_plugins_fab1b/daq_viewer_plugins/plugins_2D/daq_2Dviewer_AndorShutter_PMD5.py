import time

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.parameter.utils import iter_children
from pymodaq_gui.h5modules.saving import H5Saver
from pymodaq.utils.h5modules import module_saving

try:
    from pymodaq.utils.plotting.utils.plot_utils import RoiInfo
except:
    from pymodaq_gui.plotting.items.roi import RoiInfo

from qtpy import QtWidgets, QtCore
from time import perf_counter
import numpy as np

from pymodaq.utils.logger import set_logger, get_module_name
logger = set_logger(get_module_name(__file__))

from pylablib.devices import Andor
from pymodaq_plugins_fab1b.daq_move_plugins.daq_move_VLM1 import DAQ_Move_VLM1
from pymodaq.utils.parameter import utils as putils

camera_list = [*range(Andor.get_cameras_number_SDK3())]
camera_names_list = dict()
for camera in camera_list:
    cam = Andor.AndorSDK3Camera(idx=camera)
    camera_names_list.update({cam.get_device_info()[1]+' '+cam.get_device_info()[2]: camera})
    cam.close()


class DAQ_2DViewer_AndorShutter_PMD5(DAQ_Viewer_base, DAQ_Move_VLM1):
    """
    """
    params_shutter = DAQ_Move_VLM1.params
    axis_unit = ''
    d = putils.get_param_dict_from_name(params_shutter, 'multiaxes')
    if d is not None:
        d['visible'] = False

    params = comon_parameters + [
        {'title': 'Camera Settings:', 'name': 'camera_settings', 'type': 'group', 'children':
            [{'title': 'Camera:', 'name': 'camera_list', 'type': 'list', 'limits': camera_names_list},
             {'title': 'Acquisition', 'name': 'acq', 'type': 'group', 'children':
                 [{'title': 'Acquisition mode:', 'name': 'acq_mode', 'type': 'list', 'limits': ['Normal', 'Fast 1D']},#'Spectrum', 'Differential', 'Sequence'], 'value':'Spectrum'},
                  {'title': 'Fast mode:', 'name': 'fast_mode', 'type': 'list', 'limits': ['Spectrum', 'Differential']},
                  {'title': 'Display:', 'name': 'display', 'type': 'list', 'limits': ['Average', '2D'], 'value':'Average'},
                  {'title': 'Differential type:', 'name': 'diff_type', 'type': 'list', 'limits': ['dR/R', 'dOD'], 'visible':False},
                  {'title': 'Bit depth:', 'name': 'bit_depth', 'type': 'list', 'limits': ['Fastest frame rate (12-bit)', 'High dynamic range (16-bit)']}]
              },

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
                 [{'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 20},#0.13},
                  {'title': 'Chunk size', 'name': 'chunk_size', 'type': 'int', 'value': 1},
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
             {'title': 'Shutter:', 'name': 'shutter_bool', 'type': 'led_push', 'value': False},
             {'title': 'Pump-Probe Settings:', 'name': 'dev', 'type': 'group', 'children': [
                 {'title': 'Show Timestamps', 'name': 'timestamps_on', 'type': 'bool', 'value': False},
                 {'title': 'Show Pump On/Off', 'name': 'pumponoff_on', 'type': 'bool', 'value': True},
                 #{'title': 'N Average for backgrounds', 'name': 'navg_bkg', 'type': 'int', 'value': 1},
                 {'title': 'On before off', 'name': 'on_before_off', 'type': 'bool', 'value': False},
                 {'title': 'Take Backgrounds', 'name': 'take_bkg', 'type': 'bool_push', 'value': False},
                 {'title': 'Clear Backgrounds', 'name': 'clear_bkg', 'type': 'bool_push', 'value': False},
                 #{'title': 'Current background file', 'name': 'current_bkg_file', 'type': 'text', 'value': 'No background', 'readonly': True},
             ]},
             {'title': 'Temperature Settings:', 'name': 'temperature_settings', 'type': 'group', 'children': [
                 {'title': 'Enable Cooling:', 'name': 'enable_cooling', 'type': 'bool', 'value': False},
                 {'title': 'Set Point:', 'name': 'set_point', 'type': 'float', 'value': 20},
                 {'title': 'Current value:', 'name': 'current_value', 'type': 'float', 'value': 20, 'readonly': True},
             ]},
             ]}
    ] + params_shutter

    start_waitloop = QtCore.Signal()
    stop_waitloop = QtCore.Signal()
    roi_info = None
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

        # needed attributes for saving methods
        self.title = "Andor Camera"
        self.ui = None

        self.x_axis = None
        self.y_axis = None
        self.last_tick = 0.0  # time counter used to compute FPS
        self.fps = 0.0

        self.data_shape = 'Data2D'
        self.buffer_size = 500
        self.callback_thread = None

        self.temperature_timer = QtCore.QTimer()
        self.temperature_timer.timeout.connect(self.update_temperature)
        self.temp_freq = 2000 # Frequency of temperature timer in ms

        self.bkg_poff = None
        self.bkg_pon = None
        self.poff = None
        self.pon = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """

        # Temperature management
        # ----------------------
        if param.name() == 'set_point':
            self.controller.set_temperature(param.value(), enable_cooler=False)

        elif param.name() in iter_children(self.settings.child('camera_settings', 'temperature_settings'), []):
            self.setup_temperature()

        # Acquisition parameters
        # ----------------------
        elif param.name() == "exposure_time":
            self.controller.set_attribute_value("ExposureTime", param.value() / 1000)
            self.settings.child("camera_settings",'timing_opts', 'exposure_time').setValue(self.controller.get_attribute_value("ExposureTime")*1000)
            self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.controller.get_attribute_value('FrameRate'))

        elif param.name() == "bit_depth":
            # self.controller.set_attribute_value("PixelEncoding",param.value())
            self.controller.set_attribute_value("SimplePreAmpGainControl",param.value())
            self.settings.child("camera_settings", 'timing_opts', 'fps2').setValue(
                self.controller.get_attribute_value('FrameRate'))

        elif param.name() in iter_children(self.settings.child("camera_settings",'trigger'), []):
            self.set_trigger()

        # ROI
        # ---
        elif param.name() == "update_roi":
            if param.value():  # Switching on ROI
                # We handle ROI and binning separately for clarity
                (old_x, _, old_y, _, xbin, ybin) = self.controller.get_roi()  # Get current binning

                y0, x0 = self.roi_info.origin.coordinates
                height, width = self.roi_info.size.coordinates

                # Values need to be rescaled by binning factor and shifted by current x0,y0 to be correct.
                new_x = (old_x + x0) * xbin
                new_y = (old_y + y0) * xbin
                new_width = width * ybin
                new_height = height * ybin

                new_roi = (new_x, new_width, xbin, new_y, new_height, ybin)
                self.update_rois(new_roi)

                param.setValue(False)

        elif param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child("camera_settings",'roi','binning').value()
            ybin = self.settings.child("camera_settings",'roi','binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)

        elif param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                self.clear_roi()
                param.setValue(False)

        # Other ROI Parameters
        elif param.name() in iter_children(self.settings.child("camera_settings",'roi'), []):
            new_roi = self.get_roi_from_settings()
            self.update_rois(new_roi)

        # Options for Fast 1D Mode
        # ------------------------
        elif param.name() == "fps_on":
            self.settings.child("camera_settings",'timing_opts', 'fps').setOpts(visible=param.value())
            self.settings.child("camera_settings",'timing_opts', 'fps2').setOpts(visible=param.value())

        elif param.name() == 'timestamps_on':
            self._prepare_view()

        elif param.name() == 'pumponoff_on':
            self._prepare_view()

        elif param.name() == 'chunk_size':
            # if param.value() % 2:
            #     self.settings.child("camera_settings",'timing_opts', 'chunk_size').setValue(param.value()+1)
            self._prepare_view()

        elif param.name() == 'COM_port':
            DAQ_Move_VLM1.commit_settings(self, param)

        elif param.name() == 'shutter_bool':
            if param.value():
                self.move_abs(1)
            else:
                self.move_abs(0)

        # Switching between various modes
        # -------------------------------
        elif param.name() in ['display', 'fast_mode']:
            self.set_acq_mode()
            self._prepare_view()

        elif param.name() == 'acq_mode':
            self.set_acq_mode()
            self._prepare_view()

        # Backgrounds
        # -----------
        elif param.name() == "take_bkg":
            if param.value():
                self.take_background()
            param.setValue(False)

        elif param.name() == "clear_bkg":
            if param.value():
                self.clear_background()
            param.setValue(False)


    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info

    def clear_roi(self):
        wdet, hdet = self.controller.get_detector_size()
        self.settings.child("camera_settings",'roi','binning').setValue(1)
        new_roi = (0, wdet, 1, 0, hdet, 1)
        self.update_rois(new_roi)

    def set_acq_mode(self):
        mode = self.settings["camera_settings",'acq','acq_mode']
        if mode == 'Normal':
            self.settings.child("camera_settings",'timing_opts', 'chunk_size').hide()
            self.settings.child("camera_settings", "roi", "update_roi").show()
            #self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('Internal')
            self.settings.child("camera_settings",'dev').hide()
            self.settings.child("camera_settings",'acq','fast_mode').hide()
            self.settings.child("camera_settings",'acq','diff_type').hide()
            # self.settings.child("camera_settings",'acq','display').hide()

        else:
            self.settings.child("camera_settings",'acq','fast_mode').show()
            self.settings.child("camera_settings","roi","update_roi").hide()
            fast_mode = self.settings["camera_settings",'acq','fast_mode']
            self.settings.child("camera_settings",'timing_opts', 'chunk_size').show()
            #self.settings.child("camera_settings",'trigger', 'trigger_mode').setValue('External')
            self.settings.child("camera_settings",'dev').show()
            self.settings.child("camera_settings",'acq','display').show()

            if fast_mode == 'Differential':
                self.settings.child("camera_settings",'acq','diff_type').show()
            else:
                self.settings.child("camera_settings",'acq','diff_type').hide()

        self.settings.child("camera_settings", 'timing_opts', 'fps2').setValue(
            self.controller.get_attribute_value('FrameRate'))

    def update_temperature(self):
        """
        update temperature status and value. Fired using the temperature_timer every 2s when not grabbing
        """
        temp = self.controller.get_temperature()
        self.settings.child('camera_settings', 'temperature_settings', 'current_value').setValue(temp)

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
        shutter_initialized = DAQ_Move_VLM1.ini_stage(self, controller)
        if shutter_initialized[1]:
            self.move_home()  # close shutter

        self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())

        # Choose data type
        # self.controller.set_frame_format("array")
        self.controller.set_frame_format("list")
        self.controller.setup_acquisition(mode="sequence", nframes=self.buffer_size)

        # Set bit depth
        self.controller.set_attribute_value("SimplePreAmpGainControl", self.settings["camera_settings", "acq", "bit_depth"])
        # self.settings.child("camera_settings",'acq','bit_depth').setOpts(limits=self.controller.get_attribute('PixelEncoding').values)
        # self.settings.child("camera_settings",'acq','bit_depth').setOpts(value=self.controller.get_attribute_value('PixelEncoding'))

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
        self.settings.child("camera_settings", 'timing_opts', 'fps2').setValue(
            self.controller.get_attribute_value('FrameRate'))

    def setup_temperature(self):
        enable = self.settings.child('camera_settings', 'temperature_settings', 'enable_cooling').value()
        self.controller.set_cooler(on=enable)
        if not self.temperature_timer.isActive():
            self.temperature_timer.start(self.temp_freq)  # Timer event fired every 2s
        if enable:
            #if self.camera_controller.TemperatureControl.isWritable():
            #    self.camera_controller.TemperatureControl.setString(self.settings.child('camera_settings', 'temperature_settings', 'set_point').value())
            self.update_temperature()
            # set timer to update temperature info from controller

    def move_abs(self, position):
        DAQ_Move_VLM1.move_abs(self, position)
        if position == 0:
            self.settings.child('camera_settings', 'shutter_bool').setValue(False)
            self.shutter_status = False
        else:
            self.settings.child('camera_settings', 'shutter_bool').setValue(True)
            self.shutter_status = True

    def _prepare_view(self):
        self.settings.child("camera_settings", 'timing_opts', 'fps2').setValue(
            self.controller.get_attribute_value('FrameRate'))
        dte = self.generate_dte_temp()
        # init the viewers
        self.dte_signal_temp.emit(dte)
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
        mock_data = np.zeros((height, width))

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
                if self.settings["camera_settings",'acq','fast_mode'] == 'Differential':
                    nchunk = int(nchunk/2)
                self.y_axis = Axis(data=np.linspace(0, nchunk, nchunk, endpoint=False), label='Shot', index=0)
                self.axes = [self.x_axis, self.y_axis]
                mock_data = np.zeros((nchunk, width))

            else: # this is in 1D:
                data_shape = 'Data1D'
                self.x_axis.index = 0
                self.axes = [self.x_axis]
                mock_data = np.zeros((width,))

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

            if ponoff and self.settings["camera_settings",'acq','fast_mode'] == 'Differential' :
                dte.append(DataFromPlugins(name='Pump Off/On',
                                               data=[np.squeeze(mock_data), np.squeeze(mock_data)],
                                               dim=self.data_shape,
                                               labels=['Pump Off', 'Pump On'],
                                               axes=self.axes))

        return DataToExport("Andor", data=dte)


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
        self.data = []
        self.timestamps = []
        self.temperature_timer.stop() #Stop temperature reading during acquisition

        if 'live' in kwargs:
            self.live = kwargs['live']

        try:
            if self.settings["camera_settings",'acq','fast_mode'] == 'Differential':
                self.move_abs(int(self.settings["camera_settings", "dev", "on_before_off"]))  # Open or Close shutter depending on setting
                QtCore.QThread.msleep(16)

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

                self.dte_signal.emit(dte)

                if self.settings.child("camera_settings",'timing_opts', 'fps_on').value():
                    self.update_fps()

            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    # Might not be useful in the end
    # def save_background(self, dte):
    #     # Saving as viewer attribute
    #     ponoff = dte.get_data_from_name("Pump On/Off").data
    #
    #     avgs = [np.mean(spectrum[self.settings["camera_settings", "dev", "background_px1"]:self.settings["camera_settings", "dev", "background_px2"]]) for
    #             spectrum in ponoff]
    #     if avgs[1] > avgs[0]:
    #         self.bkg_poff, self.bkg_pon = ponoff
    #     else:
    #         self.bkg_pon, self.bkg_poff = ponoff
    #
    #     # Saving to h5 file for future post-processing
    #     h5saver = H5Saver(save_type='detector')
    #     h5saver.settings.child("base_name").setValue("Background")
    #     h5saver.init_file(update_h5=True, custom_naming=False)
    #
    #     self.settings["camera_settings", "dev", "current_bkg_file"] = h5saver.settings["current_h5_file"]
    #
    #     self.module_and_data_saver = module_saving.DetectorSaver(self)
    #     self.module_and_data_saver.h5saver = h5saver
    #
    #     detector_node = self.module_and_data_saver.get_set_node()
    #     self.module_and_data_saver.add_data(detector_node, dte)
    #
    #     h5saver.close_file()


    def generate_dte_real(self):
        dte = DataToExport(name='Andor', data=[])
        do_emit = False
        label='Image'

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
        elif self.settings["camera_settings",'acq','acq_mode'] == 'Fast 1D':
            # Read all frames in buffer together with timestamps
            frames, info = self.controller.read_multiple_images(return_info=True)

            if frames is not None:
                if len(frames) > 0:
                    if np.squeeze(frames[0]).ndim ==2:       #if each frame is a 2D image
                        frames = [np.sum(frame, axis=0) for frame in frames]    # Software full vertical binning. frames size = [nframes, 2048]
                    self.data = sum(frames) / len(frames)

                    if self.settings["camera_settings", 'acq', 'fast_mode'] == 'Spectrum':
                        do_emit = True

                    elif self.settings["camera_settings",'acq','fast_mode'] == 'Differential':
                        shutter_state = self.shutter_status #shutter state during this acquisition
                        if not shutter_state:  # This was pump off
                            self.poff = self.data
                            self.move_abs(1)  # Open shutter

                        else:  # This was pump on
                            self.pon = self.data
                            self.move_abs(0)  # Close shutter

                        # Differential acquisition is finished if:
                        # We are in "off before on" and this shot is pump on
                        # Or we are in "on before off" and this is pump off
                        # This is a XOR
                        acq_finished = self.settings["camera_settings", "dev", "on_before_off"] ^ shutter_state

                        #Depending on mode, clear data or process it
                        if not acq_finished:
                            self.data = []
                            do_emit = False

                        else:
                            if self.bkg_poff is not None and self.bkg_pon is not None:
                                self.pon -= self.bkg_pon
                                self.poff -= self.bkg_poff

                            self.poff[self.poff == 0] = 1e-10
                            self.pon[self.pon == 0] = 1e-10

                            if self.settings["camera_settings",'acq','diff_type'] == 'dR/R':
                                self.data = (self.pon-self.poff)/self.poff
                                name = "Differential Reflectivity"

                            elif self.settings["camera_settings",'acq','diff_type'] == 'dOD':
                                self.data = -np.real(np.log(self.pon/self.poff))
                                name = "Differential Optical Density"

                            self.data[np.isnan(self.data)] = 0
                            self.data[np.isinf(self.data)] = 0
                            #
                            # if self.settings["camera_settings",'acq','display'] == 'Average':
                            #     self.data = np.nanmean(self.data, axis=0)
                            #     pon = np.nanmean(pon, axis=0)
                            #     poff = np.nanmean(poff, axis=0)

                            do_emit = True

        if do_emit:
            dfp_list = [DataFromPlugins(name=label,
                                   data=[np.squeeze(self.data)],
                                   dim=self.data_shape,
                                   labels=[f'Camera'],
                                   axes=self.axes)]

            if self.settings["camera_settings",'acq','fast_mode'] == 'Differential' and self.settings["camera_settings",'dev','pumponoff_on']:
                dfp_list.append(DataFromPlugins(name='Pump On/Off',
                                           data=[np.squeeze(self.poff), np.squeeze(self.pon)],
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
            self.data = []  # Clear variables
            self.timestamps = []

        return dte, do_emit

    def take_background(self):
        self.move_abs(0)  # Close shutter
        QtCore.QThread.msleep(16)
        self.bkg_poff = np.sum(self.controller.snap(), axis=0)#np.mean([np.sum(frame, axis=0) for frame in self.controller.grab(nframes=self.settings["camera_settings", "dev", "navg_bkg"])])

        self.move_abs(1)  # Open shutter
        QtCore.QThread.msleep(16)
        self.bkg_pon = np.sum(self.controller.snap(), axis=0)#np.mean([np.sum(frame, axis=0) for frame in self.controller.grab(nframes=self.settings["camera_settings", "dev", "navg_bkg"])])
        self.move_abs(0)


    def clear_background(self):
        self.bkg_poff = None
        self.bkg_pon = None
        # self.settings["camera_settings", "dev", "current_bkg_file"] = "No background"

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
        if self.live and self.settings["camera_settings",'acq','acq_mode'] == "Fast 1D":
            scaling = self.settings["camera_settings",'timing_opts', 'chunk_size']
        else:
            scaling = 1
        self.settings.child("camera_settings",'timing_opts', 'fps').setValue(round(self.fps * scaling, 1))
        self.settings.child("camera_settings",'timing_opts', 'fps2').setValue(self.controller.get_attribute_value('FrameRate'))



    def close(self):
        """
        Terminate the communication protocol
        """
        # Terminate the communication
        self.temperature_timer.stop()
        self.controller.close()
        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""

    def stop(self):
        self.move_abs(int(self.settings["camera_settings", "dev", "on_before_off"])) # in this mode, we keep shutter open
        """Stop the acquisition."""
        self.stop_waitloop.emit()
        self.controller.stop_acquisition()
        self.controller.clear_acquisition()
        frames = self.controller.read_multiple_images() # read all images still in memory to remove them
        self.temperature_timer.start(self.temp_freq)
        return ''


class PylablibCallback(QtCore.QObject):
    """Callback object """
    data_sig = QtCore.Signal()

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
                QtCore.QThread.msleep(wait_time)


if __name__ == '__main__':
    main(__file__)