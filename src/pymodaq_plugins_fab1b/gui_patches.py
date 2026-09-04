"""GUI-side defaults for this plugin package's viewers.

PyMoDAQ's Viewer2D has no config hook for a default colormap: a freshly
created Viewer2D (or the Viewer2D pymodaq_gui.plotting.data_viewers.viewerND
composes internally for its signal display and navigator - so this also
covers switching between Data2D and DataND) defaults every single-channel
image to the hardcoded 'grey' gradient until the user manually picks one in
that specific widget instance - and a new widget instance is created every
time the displayed data's dimensionality changes, silently reverting the
choice each time (pymodaq_gui/plotting/data_viewers/viewer2D.py,
Viewer2D._show_data: "if len(data) == 1 and not self._is_gradient_manually_set:
self.set_gradient('red', 'grey')").

Rather than a fragile full override of that large private method, this
patches the small public Viewer2D.set_gradient() convenience method to
substitute 'nipy_spectral' for 'grey' - covering every code path that would
otherwise silently fall back to grey, including across dimensionality
switches, without depending on _show_data's internals staying the same.

Imported once (Python caches it) from the daq_viewer_plugins __init__.py
files, which only run inside an actual PyMoDAQ GUI session - never from the
standalone hardware/benchmark scripts, so this carries no GUI dependency
into code that doesn't need one.
"""

from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))

# Andor's Marana-X output is always single-channel (raw sensor counts), so
# the "not self._is_gradient_manually_set" default-to-grey path in
# Viewer2D._show_data is exactly the one hit on every fresh viewer for this
# plugin's data - hence patching it here rather than leaving it to be
# rediscovered as a recurring annoyance.
NIPY_SPECTRAL_GRADIENT = {
    "ticks": [
        (0.000000, (0, 0, 0, 255)),
        (0.032258, (75, 0, 85, 255)),
        (0.064516, (123, 0, 140, 255)),
        (0.096774, (134, 0, 151, 255)),
        (0.129032, (56, 0, 163, 255)),
        (0.161290, (0, 0, 181, 255)),
        (0.193548, (0, 0, 213, 255)),
        (0.225806, (0, 56, 221, 255)),
        (0.258065, (0, 125, 221, 255)),
        (0.290323, (0, 146, 221, 255)),
        (0.322581, (0, 160, 199, 255)),
        (0.354839, (0, 170, 168, 255)),
        (0.387097, (0, 170, 144, 255)),
        (0.419355, (0, 163, 83, 255)),
        (0.451613, (0, 154, 0, 255)),
        (0.483871, (0, 175, 0, 255)),
        (0.516129, (0, 199, 0, 255)),
        (0.548387, (0, 220, 0, 255)),
        (0.580645, (0, 242, 0, 255)),
        (0.612903, (44, 255, 0, 255)),
        (0.645161, (176, 255, 0, 255)),
        (0.677419, (216, 245, 0, 255)),
        (0.709677, (241, 231, 0, 255)),
        (0.741935, (252, 210, 0, 255)),
        (0.774194, (255, 177, 0, 255)),
        (0.806452, (255, 129, 0, 255)),
        (0.838710, (255, 33, 0, 255)),
        (0.870968, (241, 0, 0, 255)),
        (0.903226, (219, 0, 0, 255)),
        (0.935484, (208, 0, 0, 255)),
        (0.967742, (204, 76, 76, 255)),
        (1.000000, (204, 204, 204, 255)),
    ],
    "mode": "rgb",
}
# Sampled from matplotlib 3.11's actual "nipy_spectral" colormap (32 evenly
# spaced stops) rather than transcribed by hand, then matplotlib was
# removed again - this module has no runtime dependency on it.

_patched = False


def apply():
    """Register the nipy_spectral gradient and make it the default.

    Idempotent and defensive: any failure here (e.g. a future pymodaq_gui
    version restructuring Viewer2D) is logged and swallowed rather than
    breaking plugin discovery for every other plugin.
    """
    global _patched
    if _patched:
        return
    _patched = True

    try:
        from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
        from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D

        Gradients.setdefault("nipy_spectral", NIPY_SPECTRAL_GRADIENT)

        original_set_gradient = Viewer2D.set_gradient

        def set_gradient(self, image_key, gradient):
            if gradient == "grey":
                gradient = "nipy_spectral"
            return original_set_gradient(self, image_key, gradient)

        Viewer2D.set_gradient = set_gradient

    except Exception as exc:
        logger.warning(f"Could not apply the default nipy_spectral colormap: {exc}")
