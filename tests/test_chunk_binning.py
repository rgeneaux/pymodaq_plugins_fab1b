import numpy as np

from pymodaq_plugins_fab1b.binning import ChunkDelayBinner, is_uniform, to_uniform_axis

N_PIX = 5


def shots(values):
    """Spectra with a constant value across the pixels, one row per shot."""
    return np.repeat(np.asarray(values, dtype=float)[:, None], N_PIX, axis=1)


def ramp(binner, t_end, rate=100., slope=4.):
    """Delay measurements of a delay ramping linearly from 0 at t=0 (delay = slope * t)."""
    for t in np.arange(0., t_end + 1e-9, 1. / rate):
        binner.add_delay(t, slope * t)


def test_each_shot_of_a_chunk_goes_to_its_own_delay_bin():
    b = ChunkDelayBinner(0., 4., 4)
    ramp(b, 1.0)
    # one chunk spanning the whole sweep: delay = 4 t -> bin = int(4 t)
    t = np.array([0.05, 0.30, 0.55, 0.80])
    b.add_chunk(t, shots([1, 2, 3, 4]), np.ones(4, bool))
    assert list(b.counts_on) == [1, 1, 1, 1] and list(b.counts_off) == [0, 0, 0, 0]
    assert np.allclose(b.mean_on[:, 0], [1, 2, 3, 4])
    assert b.n_used == 4


def test_on_off_are_accumulated_separately_and_make_the_observable():
    b = ChunkDelayBinner(0., 2., 2)
    ramp(b, 1.0, slope=2.)
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])  # bin 0: t<0.5, bin 1: t>0.5
    is_on = np.array([1, 0, 1, 0, 1, 0, 1, 0], bool)
    on_value, off_value = [120., 100.], [100., 100.]
    values = [on_value[0], off_value[0], on_value[0], off_value[0], on_value[1], off_value[1], on_value[1], off_value[1]]
    b.add_chunk(t, shots(values), is_on)
    assert list(b.counts_on) == [2, 2] and list(b.counts_off) == [2, 2]
    drr = b.observable('dR/R')
    assert np.allclose(drr[0], 0.2) and np.allclose(drr[1], 0.0)
    assert np.allclose(b.observable('ΔOD')[0], -np.log10(1.2))


def test_chunk_arriving_after_its_delays():
    # the usual case: the delays were measured while the chunk was being acquired
    b = ChunkDelayBinner(0., 4., 4)
    ramp(b, 2.0, slope=2.)
    b.add_chunk(np.array([0.1, 1.9]), shots([1, 2]), np.ones(2, bool))
    assert b.n_used == 2 and b.n_pending_shots == 0
    assert b.counts_on[0] == 1 and b.counts_on[3] == 1


def test_chunk_arriving_before_its_delays_waits_for_them():
    b = ChunkDelayBinner(0., 4., 4)
    ramp(b, 0.5)
    b.add_chunk(np.array([0.1, 0.9]), shots([1, 2]), np.ones(2, bool))
    assert b.n_used == 0 and b.n_pending_shots == 2
    for t in np.arange(0.51, 0.89, 0.01):  # still short of the last shot
        b.add_delay(t, 4 * t)
    assert b.n_pending_shots == 2
    b.add_delay(0.95, 3.8)  # now the last shot is bracketed
    assert b.n_pending_shots == 0 and b.n_used == 2


def test_long_chunk_keeps_the_delays_of_its_first_shots():
    b = ChunkDelayBinner(0., 400., 4, history=150.)
    ramp(b, 100., rate=20., slope=4.)  # 100 s of delays, no chunk yet
    # a 100 s chunk arrives once complete: the delays of its first shots must still be there
    t = np.linspace(0.5, 99.5, 200)
    b.add_chunk(t, shots(np.ones(200)), np.ones(200, bool))
    assert b.n_used == 200 and b.n_no_delay == 0


def test_history_must_cover_the_first_chunk_and_then_adapts():
    # a history shorter than the chunk loses the delays of its first shots, and says so
    b = ChunkDelayBinner(0., 400., 4, history=10.)
    ramp(b, 100., rate=20., slope=4.)
    b.add_chunk(np.linspace(0.5, 99.5, 200), shots(np.ones(200)), np.ones(200, bool))
    assert b.n_used < 200 and b.n_no_delay > 0
    # after a long chunk was seen, the history is extended to twice its duration
    n_before = len(b._t_delay)
    for t in np.arange(100.05, 400., 0.05):
        b.add_delay(t, 0.)
    kept_span = b._t_delay[-1] - b._t_delay[0]
    assert kept_span > 150. and len(b._t_delay) < n_before + (400 - 100) / 0.05


def test_shots_in_a_delay_gap_have_no_delay():
    b = ChunkDelayBinner(0., 4., 4, max_gap=0.2)
    for t in (0.0, 0.1, 0.2, 2.0, 2.1, 5.0):  # measurements missing in 0.2-2.0 and 2.1-5.0 (NaN delays)
        b.add_delay(t, t)
    b.add_chunk(np.array([0.15, 1.0, 2.05, 4.0]), shots([1, 2, 3, 4]), np.ones(4, bool))
    assert b.n_used == 2 and b.counts_on[0] == 1 and b.counts_on[2] == 1
    assert b.n_no_delay == 2 and b.n_out_of_range == 0


def test_shots_outside_the_delay_range():
    b = ChunkDelayBinner(0., 4., 4)
    b.add_delay(10.0, 9.)
    b.add_delay(10.1, 9.)
    b.add_chunk(np.array([10.05]), shots([1]), np.ones(1, bool))
    assert b.n_out_of_range == 1 and b.n_used == 0


def test_flush_gives_up_on_missing_delays():
    b = ChunkDelayBinner(0., 4., 4)
    ramp(b, 0.5)
    b.add_chunk(np.array([0.1, 0.4, 0.9]), shots([1, 1, 1]), np.ones(3, bool))
    assert b.n_pending_shots == 3
    b.flush()
    assert b.n_pending_shots == 0 and b.n_used == 2 and b.n_no_delay == 1


def test_reset_keeps_delays_and_changes_bins():
    b = ChunkDelayBinner(0., 4., 4)
    ramp(b, 1.0)
    b.add_chunk(np.array([0.1]), shots([1]), np.ones(1, bool))
    b.reset(0., 4., 8)
    assert b.n_bins == 8 and b.n_used == 0 and b.mean_on is None
    b.add_chunk(np.array([0.1]), shots([1]), np.ones(1, bool))
    assert b.n_used == 1 and b.counts_on[0] == 1


def test_mismatched_inputs_are_rejected():
    b = ChunkDelayBinner(0., 4., 4)
    try:
        b.add_chunk(np.zeros(3), shots(np.zeros(2)), np.ones(3, bool))
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_uniform_axis_helpers():
    assert is_uniform(np.linspace(0, 1, 11)) and not is_uniform(np.array([0., 1., 3.]))
    m, ax = to_uniform_axis(np.array([[0., 1., 4.]]), np.array([0., 1., 3.]), n_points=4)
    assert np.allclose(ax, np.linspace(0, 3, 4)) and m.shape == (1, 4)


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
