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


class DAQ_2DViewer_AndorFAB1B(DAQ_Viewer_base):
    """
    """

    params = comon_parameters + [
        {'title': 'Camera:', 'name': 'camera_list', 'type': 'list', 'limits': camera_list},
        {'title': 'Camera model:', 'name': 'camera_info', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False},
        {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False},
        {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2]},
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children':
            [{'title': 'Height', 'name': 'height', 'type': 'int', 'value': 1},
             {'title': 'Bottom', 'name': 'bottom', 'type': 'int', 'value': 0},
             {'title': 'Width', 'name': 'width', 'type': 'int', 'value': 2048},
             {'title': 'Left', 'name': 'left', 'type': 'int', 'value': 0},
             {'title': 'Auto Vertical Centering', 'name': 'auto_vert', 'type': 'bool', 'value': False},]
         },
        {'title': 'Timing', 'name': 'timing_opts', 'type': 'group', 'children':
            [{'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 0.13},
             {'title': 'Chunk size', 'name': 'chunk_size', 'type': 'int', 'value': 1000},
             {'title': 'Compute FPS', 'name': 'fps_on', 'type': 'bool', 'value': True},
             {'title': 'FPS', 'name': 'fps', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 6}]
         },
        {'title': 'Trigger Settings:', 'name': 'trigger', 'type': 'group', 'children': [
            {'title': 'Mode:', 'name': 'trigger_mode', 'type': 'list', 'limits': [], 'value': 'External'},
            {'title': 'Software Trigger:', 'name': 'soft_trigger', 'type': 'bool_push', 'value': False,
             'label': 'Fire'},
            #{'title': 'External Trigger delay (ms):', 'name': 'ext_trigger_delay', 'type': 'float', 'value': 0.},
        ]},
        {'title': 'Developer Settings:', 'name': 'dev', 'type': 'group', 'children': [
            {'title': 'Show Timestamps', 'name': 'timestamps_on', 'type': 'bool', 'value': False},
        ]},

    ]
    start_waitloop = QtCore.Signal()
    stop_waitloop = QtCore.Signal()
    roi_pos_size = QtCore.QRectF(0,0,10,10)
    axes = []
    live = False
    n_grabed_frames = 0
    data = None
    update_timestamp_plot = False
    timestamps = []
    timestamp_frequency = 0

    def init_controller(self):
        return Andor.AndorSDK3Camera(idx=self.settings["camera_list"])

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
            self.settings.child('timing_opts', 'exposure_time').setValue(self.controller.get_attribute_value("ExposureTime")*1000)

        if param.name() == "fps_on":
            self.settings.child('timing_opts', 'fps').setOpts(visible=param.value())

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

        if param.name() in iter_children(self.settings.child('roi'), []):
            new_roi = self.get_roi_from_settings()
            self.update_rois(new_roi)

        if param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child('binning').value()
            ybin = self.settings.child('binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)

        if param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                wdet, hdet = self.controller.get_detector_size()
                # self.settings.child('ROIselect', 'x0').setValue(0)
                # self.settings.child('ROIselect', 'width').setValue(wdet)
                self.settings.child('binning').setValue(1)
                #
                # self.settings.child('ROIselect', 'y0').setValue(0)
                # new_height = self.settings.child('ROIselect', 'height').setValue(hdet)

                new_roi = (0, wdet, 1, 0, hdet, 1)
                self.update_rois(new_roi)
                param.setValue(False)

        if param.name() == 'timestamps_on':
            self.update_timestamp_plot = param.value()

        elif param.name() in iter_children(self.settings.child('trigger'), []):
            self.set_trigger()

    def ROISelect(self, roi_pos_size):
        self.roi_pos_size = roi_pos_size

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

        # Get camera name
        self.settings.child('camera_info').setValue(self.controller.get_device_info()[1])

        # Set exposure time
        self.controller.set_exposure(self.settings.child('timing_opts', 'exposure_time').value() / 1000)
        attr = self.controller.get_attribute('ExposureTime')
        self.settings.child('timing_opts', 'exposure_time').setLimits((attr.min * 1000, attr.max * 1000))

        # FPS visibility
        self.settings.child('timing_opts', 'fps').setOpts(visible=self.settings.child('timing_opts', 'fps_on').value())

        # Update image parameters
        new_roi = self.get_roi_from_settings()
        self.update_rois(new_roi)

        # Enable Metadata in order to get Frame timestamps
        self.controller.enable_metadata()
        self.controller.call_command("TimestampClockReset")
        self.timestamp_frequency = self.controller.get_attribute_value("TimestampClockFrequency")

        self.setup_callback()
        self._prepare_view()
        self.settings.child('trigger', 'trigger_mode').setValue('External') # not very clean
        self.setup_trigger()

        info = "Initialized camera"
        initialized = True
        return info, initialized

    # def wait_func(since='lastread', nframes=1, timeout=20.0):
    #     return self.controller.wait_for_frame(since=since, nframes=nframes, timeout=timeout)
    #
    # callback = PylablibCallback(wait_func)
    #
    # self.callback_thread = QtCore.QThread()  # creation of a Qt5 thread
    # callback.moveToThread(self.callback_thread)  # callback object will live within this thread
    # callback.data_sig.connect(
    #     self.emit_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired
    #
    # self.callback_signal.connect(callback.wait_for_acquisition)
    # self.callback_thread.callback = callback
    # self.callback_thread.start()

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
        self.settings.child('trigger', 'trigger_mode').setLimits(self.controller.get_attribute("TriggerMode").values)
        self.set_trigger()

    def set_trigger(self):
        self.controller.set_attribute_value("TriggerMode", self.settings.child('trigger', 'trigger_mode').value())

    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""
        # wx = self.settings.child('rois', 'width').value()
        # wy = self.settings.child('rois', 'height').value()
        # bx = self.settings.child('rois', 'x_binning').value()
        # by = self.settings.child('rois', 'y_binning').value()
        #
        # sizex = wx // bx
        # sizey = wy // by
        (hstart, hend, vstart, vend, *_) = self.controller.get_roi()
        height = vend - vstart
        width = hend - hstart

        self.settings.child('roi','width').setValue(width)
        self.settings.child('roi','height').setValue(height)
        self.settings.child('roi', 'left').setValue(hstart)
        self.settings.child('roi', 'bottom').setValue(vstart)
        mock_data = np.zeros((width, height))
        self.data = mock_data.T

        self.x_axis = Axis(data=np.linspace(0,width,width, endpoint=False), label='Pixels', index=1)

        if height != 1:
            data_shape = 'Data2D'
            self.y_axis = Axis(data=np.linspace(0, height, height, endpoint=False), label='Pixels', index=0)
            self.axes = [self.x_axis, self.y_axis]

        else:
            data_shape = 'Data1D'
            self.x_axis.index = 0
            self.axes = [self.x_axis]

        timestamps = self.settings['dev', 'timestamps_on'] and data_shape == 'Data1D' and self.live

        if data_shape != self.data_shape or timestamps:

            dte = [DataFromPlugins(name='Camera Image',
                                      data=[np.squeeze(mock_data)],
                                      dim=self.data_shape,
                                      labels=[f'Camera_{self.data_shape}'],
                                      axes=self.axes)]

            if timestamps:
                taxis = Axis(data=np.arange((self.settings['timing_opts', 'chunk_size'])), label='Time', index=0)
                timestamp_data = DataFromPlugins(name='Timestamps',
                                      data=[np.zeros((self.settings['timing_opts', 'chunk_size']))],
                                      axes=[taxis],
                                      dim='Data1D')

                dte.append(timestamp_data)
                # dte.append(DataFromPlugins(name='Timestamps',
                #                            data=[timestamps],
                #                            dim='Data1D'))
            self.data_shape = data_shape
            # init the viewers
            self.dte_signal_temp.emit(DataToExport('Camera', data=dte))
            QtWidgets.QApplication.processEvents()

    def get_roi_from_settings(self):
        x0 = self.settings['roi', 'left']
        y0 = self.settings['roi', 'bottom']
        width = self.settings['roi', 'width']
        height = self.settings['roi', 'height']

        if self.settings['roi', 'auto_vert']:
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

            self.settings['roi', 'left'] = new_x
            self.settings['roi', 'bottom'] = new_y
            self.settings['roi', 'width'] = new_width
            self.settings['roi', 'height'] = new_height

            if new_height != 1:
                self.settings.child('timing_opts', 'chunk_size').setOpts(visible=False)
            else:
                self.settings.child('timing_opts', 'chunk_size').setOpts(visible=True)
            self._prepare_view()

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data.
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """
        self.n_grabed_frames = 0
        self.data = 0.0
        self.timestamps = []

        if 'live' in kwargs:
            self.live = kwargs['live']

        try:
            # Warning, acquisition_in_progress returns 1,0 and not a real bool
            if not self.controller.acquisition_in_progress():
                self._prepare_view()
                self.controller.clear_acquisition()

                if self.data_shape == 'Data1D' and self.live:
                    self.controller.set_frame_format("array")
                else:
                    self.controller.set_frame_format("list")

                self.controller.start_acquisition()

            # Then start the acquisition
            self.start_waitloop.emit()  # will trigger the wait for acquisition

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data(self):
        """
            Fonction used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            do_emit = False
            if self.live and self.data_shape == 'Data1D':
                remaining_frames = self.settings['timing_opts', 'chunk_size'] - self.n_grabed_frames

                # Read all frames in buffer together with timestamps
                frames, info = self.controller.read_multiple_images(return_info=True)

                # If we have more frames than chunk size we drop the extra
                if len(frames) > remaining_frames:
                    frames = frames[:remaining_frames]
                    info = info[:remaining_frames, :]

                # Increment number of read frames
                self.n_grabed_frames += len(frames)

                # Add frame data to current average
                if len(frames) > 1:
                    self.data += np.sum(frames, axis=0)
                elif len(frames) == 1:
                    self.data += np.squeeze(frames)

                # Store timestamps in ms
                if self.settings['dev', 'timestamps_on']:
                    # Save timestamps in ms:
                    self.timestamps.extend(info[:, 1]/self.timestamp_frequency*1000)

                # If we have enough for the chunk,
                if self.n_grabed_frames >= self.settings['timing_opts', 'chunk_size']:
                    self.data /= self.n_grabed_frames
                    do_emit = True

            else:
                self.data = self.controller.read_oldest_image()
                do_emit = True

            if not self.live:
                self.stop()

            # Emit the frame.
            if self.data is not None and do_emit:
                dte = [DataFromPlugins(name='Camera Image',
                                       data=[np.squeeze(self.data)],
                                       dim=self.data_shape,
                                       labels=[f'Camera'],
                                       axes=self.axes)]

                if self.timestamps:
                    taxis = Axis(data=np.arange(1,1+len(self.timestamps)), label="Frame", units="")
                    taxis.index = 0
                    dte.append(DataFromPlugins(name='Timestamps',
                                       data=[np.asarray(self.timestamps)-np.min(self.timestamps)],
                                       dim='Data1D',
                                       axes=[taxis],
                                        label='Timestamps (ms)'))
                self.dte_signal.emit(DataToExport('Camera', data=dte))


                if self.settings.child('timing_opts', 'fps_on').value():
                    self.update_fps()

            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def update_fps(self):
        current_tick = perf_counter()
        frame_time = current_tick - self.last_tick

        if self.last_tick != 0.0 and frame_time != 0.0:
            # We don't update FPS for the first frame, and we also avoid divisions by zero

            if self.fps == 0.0:
                self.fps = 1 / frame_time
            else:
                # If we already have an FPS calculated, we smooth its evolution
                self.fps = 0.9 * self.fps + 0.1 / frame_time

        self.last_tick = current_tick

        # Update reading
        if self.live and self.data_shape == 'Data1D':
            scaling = self.settings['timing_opts', 'chunk_size']
        else:
            scaling = 1
        self.settings.child('timing_opts', 'fps').setValue(round(self.fps * scaling, 1))

    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        raise NotImplementedError

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