import numpy as np
from qtpy import QtWidgets

from pymodaq_data.data import DataToExport
from pymodaq_plugins_fab1b.chopper_shots import SHOTS_NAME, STATE_NAME, TIMES_NAME, shots_from_dte
from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_1D.daq_1Dviewer_MaranaXChopperTA import (
    DAQ_1DViewer_MaranaXChopperTA as P,
)

N_SHOTS, WIDTH = 600, 64
PERIOD = 0.1


def _plugin(output_shots):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = P(None, None)
    p._set_status = lambda *a, **k: None
    p.settings.child("chopper", "output_shots").setValue(output_shots)
    p.settings.child("chopper", "ref_pixels", "start").setValue(0)
    p.settings.child("chopper", "ref_pixels", "stop").setValue(16)
    emitted = []
    p.dte_signal.connect(emitted.append)
    return p, emitted


def _chunk(seed=0, host_offset=1.7e9):
    """1 kHz shots, 10 Hz chopper starting its ON window at 37 ms; pump adds +40 on the 16 first pixels."""
    rng = np.random.default_rng(seed)
    cam_t = np.arange(N_SHOTS) / 1000.0
    on = ((cam_t - 0.037) % PERIOD) < PERIOD / 2
    chunk = 1000.0 + rng.normal(0, 5, (N_SHOTS, WIDTH))
    chunk[:, :16] += 40.0 * on[:, None]
    chunk[:, 32:] += 100.0  # a "probe" signal
    return chunk.astype(np.uint32), cam_t, cam_t + host_offset, on


def test_no_shot_channels_by_default():
    p, emitted = _plugin(output_shots=False)
    chunk, cam_t, host_t, _ = _chunk()
    assert p._process_chunk(chunk, cam_t, host_t)
    assert [d.name for d in emitted[0]] == ["ΔOD", "Pump on/off"]
    assert shots_from_dte(emitted[0]) is None


def test_shot_channels_have_each_shot_with_its_time_and_state():
    p, emitted = _plugin(output_shots=True)
    chunk, cam_t, host_t, truth = _chunk()
    assert p._process_chunk(chunk, cam_t, host_t)
    names = [d.name for d in emitted[0]]
    assert names[-3:] == [SHOTS_NAME, TIMES_NAME, STATE_NAME]
    times, spectra, is_on, axis, label, units = shots_from_dte(emitted[0])
    n = len(times)
    assert 0 < n < N_SHOTS  # the edge guard removed some shots
    assert spectra.shape == (n, WIDTH) and len(is_on) == n and len(axis) == WIDTH
    # times are on the host clock: they match the host arrival time of the shots they come from
    kept = np.round((times - 1.7e9) * 1000).astype(int)  # shot index = camera time in ms
    assert np.allclose(times, host_t[kept], atol=1e-6)
    # every reported state agrees with the simulated chopper
    assert np.array_equal(is_on, truth[kept])
    assert is_on.sum() > 100 and (~is_on).sum() > 100


def test_shots_have_the_background_of_their_pump_state_subtracted():
    p, emitted = _plugin(output_shots=True)
    chunk, cam_t, host_t, _ = _chunk()
    p._background_on = np.full(WIDTH, 300.0)
    p._background_off = np.full(WIDTH, 100.0)
    p._background_tag = p._background_tag_current()
    assert p._process_chunk(chunk, cam_t, host_t)
    _, spectra, is_on, *_ = shots_from_dte(emitted[0])
    probe_px = slice(40, 60)  # carries no pump signature: 1100 raw
    assert np.allclose(spectra[is_on][:, probe_px].mean(), 1100.0 - 300.0, atol=2)
    assert np.allclose(spectra[~is_on][:, probe_px].mean(), 1100.0 - 100.0, atol=2)


def test_a_rejected_chunk_emits_nothing():
    p, emitted = _plugin(output_shots=True)
    chunk, cam_t, host_t, _ = _chunk()
    chunk[:, :16] = 1000  # no pump signature on the reference pixels
    assert not p._process_chunk(chunk, cam_t, host_t)
    assert emitted == []


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as exc:
                failed += 1
                print("FAIL", name, exc)
    sys.exit(failed)
