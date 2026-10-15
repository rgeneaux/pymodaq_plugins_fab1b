"""Extension building a (delay x pixel) map of a chopped pump-probe measurement from an oscillating delay stage.

The delay stage oscillates slowly between two positions. The real delay is measured optically (e.g. fringe phase) by
one detector (0D, fs) at its own rate. The spectrometer (the Marana-X ChopperTA viewer) delivers *chunks* of single
shots, each shot with its own time and its pump on/off state. Both are continuously grabbed; the delay of every shot
is interpolated from the measured delays at the shot time, and the shot is added to the pump-on or pump-off average
of its delay bin. The map of dR/R (or ΔOD) is built from the two averages of each bin.

Shots, rather than the chunk-averaged spectrum, are binned because the stage keeps sweeping while a chunk is being
acquired: the average of a chunk would smear the delay.
The map is displayed in the extension and can also be exposed as a new DAQ_Viewer (2D) module of the dashboard.

Written for PyMoDAQ 5.2 (``Dashboard.add_det_from_extension``).
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.extensions.custom_ext import CustomExt
from pymodaq.utils.managers.modules import ModuleType
from pymodaq_data.data import DataToExport, DataCalculated, Axis
from pymodaq_gui import utils as gutils
from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq.control_modules.instruments import DET_TYPES
from pymodaq_plugins_fab1b.binning import ChunkDelayBinner as Binner, is_uniform, to_uniform_axis
from pymodaq_plugins_fab1b.chopper_shots import shots_from_dte

if TYPE_CHECKING:
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer

logger = set_logger(get_module_name(__file__))

EXTENSION_NAME = 'Delay Binner (chunks)'  # the name that will be displayed in the extension list in the dashboard
CLASS_NAME = 'ChunkDelayBinner'  # the name of the class defined below

DISPLAY_PERIOD_MS = 200  # the map is refreshed at most at this period

VIEWER_PLUGIN = 'ChunkDelayBinner'  # name of the DAQ_2DViewer_ChunkDelayBinner plugin, defined in this folder

# where the spectrometer's setting enabling its single-shot output lives, in the settings tree of its module
OUTPUT_SHOTS_PATH = ('detector_settings', 'chopper', 'output_shots')


def register_viewer_plugin():
    """Make the DAQ_2DViewer_ChunkDelayBinner plugin known to PyMoDAQ for the lifetime of the extension

    The plugin is only meant to be created by this extension: it is not in the plugins_2D folder (which would
    list it in the detectors available in any DAQ_Viewer or preset, where it can't work) but is added here to
    the list of the 2D plugins that PyMoDAQ uses to find the class of a module to create.
    """
    import importlib
    parent = importlib.import_module(__package__)
    importlib.import_module(f'{__package__}.daq_2Dviewer_{VIEWER_PLUGIN}')
    if not any(plugin['name'] == VIEWER_PLUGIN for plugin in DET_TYPES['DAQ2D']):
        DET_TYPES['DAQ2D'].append({'name': VIEWER_PLUGIN, 'module': parent, 'parent_module': parent,
                                   'type': 'daq_2Dviewer'})


def unregister_viewer_plugin():
    DET_TYPES['DAQ2D'][:] = [plugin for plugin in DET_TYPES['DAQ2D'] if plugin['name'] != VIEWER_PLUGIN]


class ChunkDelayBinner(CustomExt):
    """Bin the single shots of a chunked, chopped spectrometer according to the delay measured by a 0D detector"""
    settings_name = 'ChunkDelayBinnerSettings'
    params = [
        {'title': 'Detectors', 'name': 'detectors', 'type': 'group', 'children': [
            {'title': 'Spectrometer (Marana-X ChopperTA):', 'name': 'spectrum', 'type': 'list', 'limits': [],
             'tip': 'The extension switches on its "Output single shots" setting while it acquires'},
            {'title': 'Measured delay (0D, fs):', 'name': 'delay', 'type': 'list', 'limits': []},
        ]},
        {'title': 'Binning', 'name': 'binning', 'type': 'group', 'children': [
            {'title': 'Delay min (fs):', 'name': 'delay_min', 'type': 'float', 'value': -7.},
            {'title': 'Delay max (fs):', 'name': 'delay_max', 'type': 'float', 'value': 7.},
            {'title': 'Number of bins:', 'name': 'n_bins', 'type': 'int', 'value': 70, 'min': 1},
            {'title': 'Bin width (fs):', 'name': 'bin_width', 'type': 'float', 'value': 0.2, 'readonly': True},
            {'title': 'Observable:', 'name': 'observable', 'type': 'list', 'limits': ['dR/R', 'ΔOD'],
             'value': 'dR/R', 'tip': 'Built from the mean pump-on and pump-off spectra of each bin'},
        ]},
        {'title': 'Delay measurement', 'name': 'timing', 'type': 'group', 'children': [
            {'title': 'Max gap (ms):', 'name': 'max_gap', 'type': 'float', 'value': 500., 'min': 1.,
             'tip': 'Shots are given the delay interpolated between the two measurements around them, provided '
                    'these are closer than this. Shots in a longer gap (e.g. fringes lost) have no delay and are '
                    'not binned'},
            {'title': 'Latency (ms):', 'name': 'latency', 'type': 'float', 'value': 0.,
             'tip': 'Time between a delay measurement and the timestamp of its data (camera exposure and '
                    'processing of the delay reader): the delay measurements are moved this much back in time'},
            {'title': 'History (s):', 'name': 'history', 'type': 'float', 'value': 600., 'min': 1.,
             'tip': 'Duration of delay measurements kept. A chunk reaches the extension when it is complete, '
                    'so this must be longer than one chunk of the spectrometer (it is extended automatically '
                    'once longer chunks were seen, but shots of the first chunks are lost if it is too short)'},
        ]},
        {'title': 'Status', 'name': 'status', 'type': 'group', 'children': [
            {'title': 'Shots in map:', 'name': 'n_used', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Outside delay range:', 'name': 'n_out', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Without delay data:', 'name': 'n_no_delay', 'type': 'int', 'value': 0, 'readonly': True,
             'tip': 'Shots with no delay measurement around them: delay reader not running, fringes lost, '
                    'latency or history wrong'},
            {'title': 'Waiting for delays:', 'name': 'n_pending', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Empty bins:', 'name': 'n_empty', 'type': 'int', 'value': 0, 'readonly': True,
             'tip': 'Bins without at least one pump-on and one pump-off shot'},
        ]},
    ]

    dte_computed_signal = QtCore.Signal(DataToExport)  # the current map and counts, emitted periodically
    acquisition_done_signal = QtCore.Signal(DataToExport)  # the final map of a timed acquisition (viewer snap)
    request_signal = QtCore.Signal(str, float)  # ('start', duration in s, <= 0 for infinite) or ('stop', 0) from the viewer module

    def __init__(self, parent: gutils.DockArea, dashboard):
        register_viewer_plugin()
        self.binner: Optional[Binner] = None
        self._running = False
        self._timed = False
        self._dirty = False
        self._axis_label, self._axis_units = 'x', 'px'
        self._spectrum_det: Optional['DAQ_Viewer'] = None
        self._delay_det: Optional['DAQ_Viewer'] = None
        self._output_shots_before: Optional[bool] = None
        super().__init__(parent, dashboard, add_toolbar_break=False)

        self.binner = self._new_binner()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(DISPLAY_PERIOD_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._duration_timer = QtCore.QTimer(self)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self.stop)

        self.setup_ui()

    # ---- UI ----------------------------------------------------------------------------------------------------
    def setup_docks_and_widgets(self):
        self.docks['settings'] = gutils.Dock('Settings')
        self.dockarea.addDock(self.docks['settings'])
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.docks['settings'].addWidget(splitter)
        splitter.addWidget(self.settings_tree)

        self.docks['map'] = gutils.Dock('Delay map')
        self.dockarea.addDock(self.docks['map'], 'right')
        self.area_map = gutils.DockArea()
        self.docks['map'].addWidget(self.area_map)
        self.map_viewer = ViewerDispatcher(self.area_map)

    def setup_actions(self):
        self.add_action('start', 'Start', 'run2', 'Grab continuously both detectors and fill the map',
                        checkable=True, checked=False, auto_toolbar=True)
        self.add_action('reset', 'Reset map', 'reset', 'Empty the map', auto_toolbar=True)
        self.add_action('create_viewer', 'Create Viewer', 'Add_Step',
                        tip='Create a 2D DAQ_Viewer module of the dashboard giving the map')

    def setup_menus_and_toolbars(self, menubar: QtWidgets.QMenuBar = None):
        self.create_dashboard_toolbar(add_break=False)

    def connect_things(self):
        self.connect_action('start', self.start_acquisition)
        self.connect_action('reset', self.reset_map)
        self.connect_action('create_viewer', self.create_viewer)
        self.dte_computed_signal.connect(self.map_viewer.show_data)
        self.acquisition_done_signal.connect(self.map_viewer.show_data)
        self.request_signal.connect(self._on_request)

    def do_things_after_experiment_set(self, experiment_name: str, *args, **kwargs):
        super().do_things_after_experiment_set(experiment_name)
        self.update_detector_lists()

    def update_detector_lists(self):
        names = [n for n in self.modules_manager.detectors_name if n != VIEWER_PLUGIN]
        for child in ('spectrum', 'delay'):
            param = self.settings.child('detectors', child)
            param.setLimits(names)
        # guess: a detector with 'delay'/'interfero' in its name is the delay measurement
        for name in names:
            if 'delay' in name.lower() or 'interfero' in name.lower():
                self.settings.child('detectors', 'delay').setValue(name)
            elif 'chopper' in name.lower() or 'marana' in name.lower():
                self.settings.child('detectors', 'spectrum').setValue(name)

    def value_changed(self, param):
        name = param.name()
        if name in ('delay_min', 'delay_max', 'n_bins'):
            self._update_bin_width()
            if self.binner is not None:
                self.reset_map()
        elif name in ('max_gap', 'history') and self.binner is not None:
            self._apply_timing_settings()
        elif name == 'observable':
            self._dirty = True

    def _update_bin_width(self):
        b = self.settings.child('binning')
        b.child('bin_width').setValue((b['delay_max'] - b['delay_min']) / b['n_bins'])

    def _apply_timing_settings(self):
        t = self.settings.child('timing')
        self.binner.max_gap = t['max_gap'] / 1000
        self.binner.history = t['history']

    def _new_binner(self) -> Binner:
        b, t = self.settings.child('binning'), self.settings.child('timing')
        return Binner(b['delay_min'], b['delay_max'], b['n_bins'], max_gap=t['max_gap'] / 1000,
                      history=t['history'])

    # ---- acquisition -------------------------------------------------------------------------------------------
    def start_acquisition(self, start: bool):
        """Slot of the Start toolbar action: acquire for an infinite time, adding to the current map"""
        if start:
            self._start(duration=None, reset=False)
        else:
            self.stop()

    def _on_request(self, command: str, duration: float):
        """Slot of the requests of the viewer module (emitted from its thread):
        'start' with a duration in s (<= 0 for an infinite acquisition), or 'stop'"""
        if command == 'start':
            self._start(duration=duration if duration > 0 else None, reset=True)
        elif command == 'stop' and self._running:
            self.stop()

    def _set_start_checked(self, checked: bool):
        action = self.get_action('start')
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def _output_shots_param(self, det: 'DAQ_Viewer'):
        """The setting making the spectrometer output its single shots, None if it does not have it"""
        try:
            return det.settings.child(*OUTPUT_SHOTS_PATH)
        except Exception:
            return None

    def _start(self, duration: Optional[float], reset: bool):
        """Grab both detectors continuously, for ``duration`` seconds or until stopped

        Parameters
        ----------
        duration: float or None
            None to acquire until :meth:`stop` is called. Otherwise the acquisition stops by itself and
            ``acquisition_done_signal`` is emitted with the final map.
        reset: bool
            Empty the map before starting
        """
        if self._running:
            self.stop()
        spectrum_name, delay_name = self.settings['detectors', 'spectrum'], self.settings['detectors', 'delay']
        self._spectrum_det = self.modules_manager.get_mod_from_name(spectrum_name, ModuleType.Detector)
        self._delay_det = self.modules_manager.get_mod_from_name(delay_name, ModuleType.Detector)
        if self._spectrum_det is None or self._delay_det is None or self._spectrum_det is self._delay_det:
            logger.error('Select two different detectors (spectrometer and delay)')
            self._set_start_checked(False)
            return
        shots_param = self._output_shots_param(self._spectrum_det)
        if shots_param is None:
            logger.error(f'{spectrum_name} has no "Output single shots" setting: select a Marana-X ChopperTA viewer')
            self._set_start_checked(False)
            return
        self._output_shots_before = bool(shots_param.value())
        shots_param.setValue(True)
        if reset:
            self.reset_map()
        self._spectrum_det.grab_done_signal.connect(self._on_spectrum)
        self._delay_det.grab_done_signal.connect(self._on_delay)
        self._running = True
        self._timed = duration is not None
        self._set_start_checked(True)
        self.settings.child('detectors').setOpts(enabled=False)
        self._delay_det.grab_data(True)
        self._spectrum_det.grab_data(True)
        if duration is not None:
            self._duration_timer.start(int(duration * 1000))

    def stop(self):
        """Stop the detectors (the map is kept). After a timed acquisition, the map is sent with
        ``acquisition_done_signal``"""
        self._duration_timer.stop()
        was_timed = self._running and self._timed
        was_running = self._running
        self._running = False
        self._timed = False
        for det, slot in ((self._spectrum_det, self._on_spectrum), (self._delay_det, self._on_delay)):
            if det is not None:
                det.grab_data(False)
                try:
                    det.grab_done_signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        if was_running and self._spectrum_det is not None and self._output_shots_before is not None:
            shots_param = self._output_shots_param(self._spectrum_det)
            if shots_param is not None:
                shots_param.setValue(self._output_shots_before)
        self._output_shots_before = None
        if self.binner is not None:
            self.binner.flush()  # no more delays will come for the chunks still waiting
        self.settings.child('detectors').setOpts(enabled=True)
        self._set_start_checked(False)
        if was_timed:
            self._refresh(force=True)
            dte = self.current_dte()
            if dte is not None:
                self.acquisition_done_signal.emit(dte)

    def _on_delay(self, dte: DataToExport):
        latency = self.settings['timing', 'latency'] / 1000
        for dwa in dte.get_data_from_dim('Data0D'):  # the first one is the delay
            delay = float(np.ravel(dwa[0])[0])
            if np.isfinite(delay):  # NaN when the delay reader does not see its fringes
                self.binner.add_delay(dwa.timestamp - latency, delay)
                self._dirty = True
            return

    def _on_spectrum(self, dte: DataToExport):
        chunk = shots_from_dte(dte)
        if chunk is None:
            logger.warning('The spectrometer data has no single-shot channels (is its "Output single shots" '
                           'setting on?): ignored')
            return
        times, spectra, is_on, axis, label, units = chunk
        self._axis_label, self._axis_units = label, units
        self.binner.add_chunk(times, spectra, is_on, axis)
        self._dirty = True

    def reset_map(self):
        b = self.settings.child('binning')
        self.binner = self._new_binner() if self.binner is None else self.binner
        self._apply_timing_settings()
        self.binner.reset(b['delay_min'], b['delay_max'], b['n_bins'])
        self._dirty = True

    # ---- results -----------------------------------------------------------------------------------------------
    def current_dte(self) -> Optional[DataToExport]:
        """The map (delay x pixel, empty bins set to 0) and the number of shots per bin"""
        kind = self.settings['binning', 'observable']
        map_ = self.binner.observable(kind)
        if map_ is None:
            return None
        map_ = np.nan_to_num(map_)
        delay_axis = Axis('Delay', 'fs', data=self.binner.centers, index=0)
        axes = [delay_axis]
        spectral = self.binner.axis
        if spectral is not None and len(spectral) == map_.shape[1]:
            if not is_uniform(spectral):  # the 2D viewers need linear axes
                map_, spectral = to_uniform_axis(map_, spectral)
            axes.append(Axis(self._axis_label, self._axis_units, data=np.asarray(spectral, dtype=float), index=1))
        return DataToExport('DelayBinner', data=[
            DataCalculated(f'{kind} map', data=[map_], labels=[kind], axes=axes),
            DataCalculated('Shots per bin', data=[self.binner.counts_on.astype(float),
                                                  self.binner.counts_off.astype(float)],
                           labels=['pump on', 'pump off'], axes=[Axis('Delay', 'fs', data=self.binner.centers,
                                                                      index=0)]),
        ])

    def _refresh(self, force: bool = False):
        if not (self._dirty or force) or self.binner is None:
            return
        self._dirty = False
        st = self.settings.child('status')
        st.child('n_used').setValue(self.binner.n_used)
        st.child('n_out').setValue(self.binner.n_out_of_range)
        st.child('n_no_delay').setValue(self.binner.n_no_delay)
        st.child('n_pending').setValue(self.binner.n_pending_shots)
        st.child('n_empty').setValue(int(np.sum((self.binner.counts_on == 0) | (self.binner.counts_off == 0))))
        dte = self.current_dte()
        if dte is not None:
            self.dte_computed_signal.emit(dte)

    # ---- viewer module -----------------------------------------------------------------------------------------
    def create_viewer(self):
        """Create a 2D DAQ_Viewer module (plugin DAQ_2DViewer_ChunkDelayBinner) outputting the map"""
        try:
            try:  # PyMoDAQ >= 5.3: the module is described by a PluginInfo, created asynchronously
                from pymodaq.control_modules.enums import DAQTypesEnum
                from pymodaq.control_modules.utils import ControllerAndThread, QThreadProxy
                from pymodaq.utils.managers.modules.loader import PluginInfo
            except ImportError:  # PyMoDAQ 5.2: positional arguments
                self.dashboard.add_det_from_extension(VIEWER_PLUGIN, 'DAQ2D', VIEWER_PLUGIN, self)
                self._on_viewer_added([])
                return
            id = self.dashboard.modules_manager.get_random_id()
            plugin = PluginInfo(
                id=id,
                name=VIEWER_PLUGIN,
                class_name=VIEWER_PLUGIN,
                type=ModuleType.Detector,
                daq_type=DAQTypesEnum.DAQ2D,
                controller=ControllerAndThread(name=VIEWER_PLUGIN,
                                               thread=QThreadProxy(thread=self.thread()),
                                               controller=self,
                                               is_master=False,
                                               id=id))
            self.dashboard.module_creator.add_det_from_extension(modules=[plugin], callback=self._on_viewer_added)
        except Exception:
            logger.exception('Could not create the viewer module')

    def _on_viewer_added(self, modules: list['DAQ_Viewer']):
        try:
            self.dashboard.module_creator.forget_callback(self._on_viewer_added)
        except AttributeError:  # PyMoDAQ 5.2: no module_creator, no callback
            pass
        self.set_action_enabled('create_viewer', False)
        self.update_detector_lists()

    def quit_fun(self):
        self.stop()
        self._timer.stop()
        if self.dashboard is not None:
            viewer = self.dashboard.modules_manager.get_mod_from_name(VIEWER_PLUGIN, ModuleType.Detector)
            if viewer is not None:
                self.dashboard.remove_detectors([viewer])
        unregister_viewer_plugin()
        super().quit_fun()


def main():
    import sys
    from pymodaq_gui.qt_utils import mkQApp
    from pymodaq.dashboard import create_load_dashboard
    from pymodaq.utils.gui_utils.loader_utils import create_extension

    app = mkQApp('Delay Binner (chunks)')
    win, dashboard = create_load_dashboard()
    win.mainwindow.setVisible(False)
    win_ext, ext = create_extension(dashboard, ChunkDelayBinner)
    win_ext.show()  # (PyMoDAQ >= 5.3 already shows it, harmless)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
