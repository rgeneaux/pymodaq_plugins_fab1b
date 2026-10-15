"""Names of the single-shot channels the ChopperTA viewer can add to its output.

Shared by the viewer plugin (producer) and the Delay Binner extension (consumer), which has to bin every shot at
its own delay: the chunk-averaged observable the viewer normally emits would smear the delay that the piezo stage
sweeps while a chunk is being acquired.

When the viewer's *Output single shots* setting is on, each chunk's output data additionally contains:

* ``SHOTS_NAME``: Data2D, shape (n_shots, n_pixels), the vertically binned shots with the stored background of
  their pump state subtracted (if a valid background is stored; raw otherwise). Axis 1 is the spectral axis.
* ``TIMES_NAME``: Data1D, shape (n_shots,), host wall-clock time (s, epoch, the same clock as the ``timestamp`` of
  the data of the other detectors) of each shot.
* ``STATE_NAME``: Data1D, shape (n_shots,), 1.0 for a pump-ON shot, 0.0 for a pump-OFF shot.

Shots in the chopper edge guard are not part of these channels.
"""

SHOTS_NAME = "Single shots"
TIMES_NAME = "Shot times (s, epoch)"
STATE_NAME = "Shot pump state (1=on, 0=off)"


def shots_from_dte(dte):
    """Extract the single-shot channels from the data a ChopperTA viewer emitted

    Returns
    -------
    None if the data has no (complete) single-shot channels, else a tuple
    ``(times, spectra, is_on, axis, axis_label, axis_units)``: the shot times (s, epoch), the (n_shots, n_pixels)
    spectra, the bool pump-on state of each shot, and the spectral axis (values, label, units).
    """
    import numpy as np

    by_name = {dwa.name: dwa for dwa in dte}
    if not all(name in by_name for name in (SHOTS_NAME, TIMES_NAME, STATE_NAME)):
        return None
    shots = by_name[SHOTS_NAME]
    spectra = np.asarray(shots[0])
    times = np.asarray(by_name[TIMES_NAME][0], dtype=float).ravel()
    is_on = np.asarray(by_name[STATE_NAME][0]).ravel() > 0.5
    if spectra.ndim != 2 or not (len(times) == len(is_on) == spectra.shape[0]):
        return None
    axes = [axis for axis in shots.axes if axis.index == 1]
    if axes:
        axis, label, units = axes[0].get_data(), axes[0].label, axes[0].units
    else:
        axis, label, units = np.arange(spectra.shape[1]), 'x', 'px'
    return times, spectra, is_on, axis, label, units
