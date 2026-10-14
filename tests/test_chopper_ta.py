import numpy as np

from pymodaq_plugins_fab1b.daq_viewer_plugins.plugins_1D.daq_1Dviewer_MaranaXChopperTA import (
    DAQ_1DViewer_MaranaXChopperTA as TA,
)

PERIOD = 0.1  # 10 Hz chopper
GUARD = 8e-3


def _synthetic(true_phase, n=1000, rate=1000.0, bright_on=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    on, _ = TA._chopper_masks(t, true_phase, PERIOD, 0.0)
    level = np.where(on, 40.0, 0.0) * (1 if bright_on else -1)
    return t, 4600 + level + rng.normal(0, 25, n), on


def test_fit_recovers_phase_and_pump_state():
    for true_phase in (0.0, 0.017, 0.0535, 0.0991):
        t, ref, truth = _synthetic(true_phase)
        phase, z = TA._fit_chopper_phase(ref, t, PERIOD, GUARD, bright_is_on=True)
        assert z > 10
        on, off = TA._chopper_masks(t, phase, PERIOD, GUARD)
        # every shot kept as ON really was pumped, and likewise for OFF
        assert truth[on].all()
        assert not truth[off].any()
        assert on.sum() > 300 and off.sum() > 300


def test_fit_polarity_flips_on_off():
    t, ref, truth = _synthetic(0.03, bright_on=False)
    phase, _ = TA._fit_chopper_phase(ref, t, PERIOD, GUARD, bright_is_on=False)
    on, off = TA._chopper_masks(t, phase, PERIOD, GUARD)
    # reference is *darker* when pumped, so ON shots are the low-level ones
    assert ref[on].mean() < ref[off].mean()


def test_fit_reports_low_z_without_modulation():
    rng = np.random.default_rng(1)
    t = np.arange(1000) / 1000.0
    ref = 4600 + rng.normal(0, 25, t.size)
    _, z = TA._fit_chopper_phase(ref, t, PERIOD, GUARD)
    assert z < 5


def test_masks_exclude_edges():
    t = np.arange(1000) / 1000.0
    on, off = TA._chopper_masks(t, 0.0, PERIOD, GUARD)
    assert not (on & off).any()
    assert (on | off).sum() < t.size  # guard shots dropped


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("ok  ", name)
            except AssertionError:
                failed += 1; print("FAIL", name)
    sys.exit(failed)
