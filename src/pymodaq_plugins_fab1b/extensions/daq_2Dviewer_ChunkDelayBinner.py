from typing import TYPE_CHECKING

from pymodaq_data.data import DataToExport
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

if TYPE_CHECKING:
    from pymodaq_plugins_fab1b.extensions.delay_binner import ChunkDelayBinner


class DAQ_2DViewer_ChunkDelayBinner(DAQ_Viewer_base):
    """ Viewer module created by the Delay Binner extension (not a hardware plugin).

    It gives, as a standard DAQ_Viewer, the map (delay bins x pixels) of the selected observable and the number of
    shots per bin built by the extension. It has two acquisition modes, both starting with an empty map:

    * *Snap*: the extension acquires during *Snap duration* then the final map is sent
    * *Grab* (continuous): the extension acquires for an infinite time, the map is updated regularly,
      until the grab is stopped

    Only the chunks completed within the acquisition time are in the map: set *Snap duration* to several chunks.
    """
    live_mode_available = True  # the plugin pushes data by itself in continuous mode
    params = comon_parameters + [
        {'title': 'Snap duration (s):', 'name': 'duration', 'type': 'float', 'value': 30., 'min': 0.1,
         'tip': 'Acquisition time of a snap (it should span several chunks of the spectrometer, the chunk being '
                'acquired when the time is over is lost). A continuous grab has no time limit.'},
    ]

    def ini_attributes(self):
        self.controller: 'ChunkDelayBinner' = None
        self._live = False

    def commit_settings(self, param: Parameter):
        pass

    def ini_detector(self, controller=None):
        self.controller = controller
        if self.controller is not None:
            self.controller.dte_computed_signal.connect(self._on_map_updated)
            self.controller.acquisition_done_signal.connect(self._on_acquisition_done)
        return "ChunkDelayBinner viewer initialized", self.controller is not None

    def close(self):
        if self.controller is not None:
            for signal, slot in ((self.controller.dte_computed_signal, self._on_map_updated),
                                 (self.controller.acquisition_done_signal, self._on_acquisition_done)):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass

    def _on_map_updated(self, dte: DataToExport):
        """Regular update of the map, only sent in continuous mode"""
        if self._live:
            self.dte_signal.emit(dte)

    def _on_acquisition_done(self, dte: DataToExport):
        """The timed acquisition of a snap is over"""
        self._live = False
        self.dte_signal.emit(dte)

    def grab_data(self, Naverage=1, live=False, **kwargs):
        """Ask the extension to start acquiring (the call returns immediately, data come later)

        Parameters
        ----------
        live: bool
            True for a continuous grab (infinite time), False for a snap (*Snap duration*)
        """
        self._live = live
        # the extension lives in the main thread: communicate with it through a (queued) signal
        self.controller.request_signal.emit('start', 0. if live else float(self.settings['duration']))

    def stop(self):
        """Stop the acquisition (the map acquired so far is kept in the extension)"""
        self._live = False
        if self.controller is not None:
            self.controller.request_signal.emit('stop', 0.)
        return ''


if __name__ == '__main__':
    main(__file__)
