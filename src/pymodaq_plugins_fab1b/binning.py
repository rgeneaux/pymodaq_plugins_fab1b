"""Core (Qt free) of the delay binning for a chunked, chopped pump-probe spectrometer.

The spectrometer delivers *chunks* of single shots, each shot with its own time and its pump state (on/off). A delay
reader delivers, independently and at its own rate, timestamped delay measurements. The delay of every shot is
obtained by interpolating the measured delay at the shot time, and the shot is added to the pump-on or pump-off sum
of the delay bin it falls in. The observable of a bin is built from the two mean spectra (ratio of means), as the
chunk-averaged observable of the viewer is.

A chunk can span a long time (many seconds), during which the delay stage sweeps: this is why the shots are binned
one by one rather than the chunk-averaged spectrum as a whole.
"""
import bisect
from typing import Optional

import numpy as np


class ChunkDelayBinner:
    """Average single shots into pre-defined delay bins, separately for pump on and off

    Parameters
    ----------
    delay_min, delay_max: float
        Edges (same unit as the measured delays, fs) of the first and last bins
    n_bins: int
    max_gap: float
        Largest time (s) between two delay measurements across which the delay is still interpolated. A shot that
        is not between two measurements closer than this has no delay (counted in ``n_no_delay``).
    history: float
        Minimum duration (s) of delay measurements kept. A chunk reaches the binner when it is complete, i.e. after
        the delays of its first shots were measured: the history is automatically extended to twice the longest
        chunk seen so that these delays are still there.

    Attributes
    ----------
    counts_on, counts_off: np.ndarray
        Number of shots in each bin
    n_used, n_out_of_range, n_no_delay: int
        Number of shots added to the map, outside the delay range, and without delay
    """

    def __init__(self, delay_min: float, delay_max: float, n_bins: int, max_gap: float = 0.5, history: float = 300.):
        self.max_gap = max_gap
        self.history = history
        self._set_bins(delay_min, delay_max, n_bins)
        self._t_delay: list = []
        self._delay: list = []
        self._pending: list = []  # (times, spectra, is_on) of chunks waiting for the delays covering them
        self._longest_chunk = 0.
        self._since_trim = 0
        self.axis: Optional[np.ndarray] = None
        self.reset()

    def _set_bins(self, delay_min: float, delay_max: float, n_bins: int):
        if n_bins < 1 or delay_max <= delay_min:
            raise ValueError('Need at least one bin and delay_max > delay_min')
        self.edges = np.linspace(delay_min, delay_max, int(n_bins) + 1)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def reset(self, delay_min: float = None, delay_max: float = None, n_bins: int = None):
        """Empty the map (and eventually change the bins). The delay measurements are kept."""
        if n_bins is not None:
            self._set_bins(self.edges[0] if delay_min is None else delay_min,
                           self.edges[-1] if delay_max is None else delay_max, n_bins)
        self._sum_on: Optional[np.ndarray] = None
        self._sum_off: Optional[np.ndarray] = None
        self.counts_on = np.zeros(self.n_bins, dtype=int)
        self.counts_off = np.zeros(self.n_bins, dtype=int)
        self._pending = []
        self.n_used = self.n_out_of_range = self.n_no_delay = 0

    # ---- inputs ------------------------------------------------------------------------------------------------
    def add_delay(self, timestamp: float, delay: float):
        """Add a delay measurement"""
        if self._t_delay and timestamp < self._t_delay[-1]:  # (should not happen) keep the times sorted
            i = bisect.bisect(self._t_delay, timestamp)
            self._t_delay.insert(i, timestamp)
            self._delay.insert(i, delay)
        else:
            self._t_delay.append(timestamp)
            self._delay.append(delay)
        self._process()
        # _process only trims the history while chunks are waiting: also do it now and then when idle
        self._since_trim += 1
        if self._since_trim >= 200:
            self._since_trim = 0
            self._forget_old_delays()

    def add_chunk(self, times: np.ndarray, spectra: np.ndarray, is_on: np.ndarray, axis: np.ndarray = None):
        """Add a chunk of single shots

        Parameters
        ----------
        times: (n_shots,) array
            time of each shot (s), same clock as the delay timestamps
        spectra: (n_shots, n_pixels) array
        is_on: (n_shots,) bool array
            pump state of each shot
        axis: (n_pixels,) array
            spectral axis, kept for display
        """
        times = np.asarray(times, dtype=float).ravel()
        spectra = np.asarray(spectra, dtype=float)
        is_on = np.asarray(is_on).ravel().astype(bool)
        if spectra.ndim != 2 or not (len(times) == len(is_on) == spectra.shape[0]):
            raise ValueError('times, is_on and spectra must have the same number of shots')
        if len(times) == 0:
            return
        if axis is not None:
            self.axis = np.asarray(axis)
        if self._sum_on is None or self._sum_on.shape[1] != spectra.shape[1]:
            self._sum_on = np.zeros((self.n_bins, spectra.shape[1]))
            self._sum_off = np.zeros((self.n_bins, spectra.shape[1]))
        order = np.argsort(times, kind='stable')
        times, spectra, is_on = times[order], spectra[order], is_on[order]
        self._longest_chunk = max(self._longest_chunk, float(times[-1] - times[0]))
        self._pending.append((times, spectra, is_on))
        self._process()

    # ---- processing --------------------------------------------------------------------------------------------
    def _process(self, flush: bool = False):
        """Bin the pending chunks whose last shot is covered by the delay measurements (all of them if ``flush``)"""
        if not self._pending or not self._t_delay:
            if flush and self._pending:
                self.n_no_delay += sum(len(t) for t, _, _ in self._pending)
                self._pending = []
            return
        t_delay, delay = np.asarray(self._t_delay), np.asarray(self._delay)
        still_pending = []
        for times, spectra, is_on in self._pending:
            if not flush and times[-1] > t_delay[-1]:
                still_pending.append((times, spectra, is_on))
                continue
            self._bin_chunk(times, spectra, is_on, t_delay, delay)
        self._pending = still_pending
        self._forget_old_delays()

    def _shot_delays(self, times, t_delay, delay):
        """Delay interpolated at each shot time, NaN where it can't be trusted"""
        right = np.searchsorted(t_delay, times, side='left')
        left = right - 1
        inside = (right > 0) & (right < len(t_delay))
        exact = (right < len(t_delay)) & (t_delay[np.minimum(right, len(t_delay) - 1)] == times)
        valid = inside | exact
        left_c, right_c = np.clip(left, 0, len(t_delay) - 1), np.clip(right, 0, len(t_delay) - 1)
        gap = t_delay[right_c] - t_delay[left_c]
        valid &= (gap <= self.max_gap) | exact
        out = np.full(times.shape, np.nan)
        out[valid] = np.interp(times[valid], t_delay, delay)
        return out

    def _bin_chunk(self, times, spectra, is_on, t_delay, delay):
        delays = self._shot_delays(times, t_delay, delay)
        has_delay = np.isfinite(delays)
        self.n_no_delay += int((~has_delay).sum())
        in_range = has_delay & (delays >= self.edges[0]) & (delays < self.edges[-1])
        self.n_out_of_range += int((has_delay & ~in_range).sum())
        if not in_range.any():
            return
        bins = np.searchsorted(self.edges, delays[in_range], side='right') - 1
        spectra, is_on = spectra[in_range], is_on[in_range]
        for state, sums, counts in ((True, self._sum_on, self.counts_on), (False, self._sum_off, self.counts_off)):
            sel = is_on == state
            if not sel.any():
                continue
            b = bins[sel]
            counts += np.bincount(b, minlength=self.n_bins)
            one_hot = np.zeros((self.n_bins, b.size))
            one_hot[b, np.arange(b.size)] = 1.
            sums += one_hot @ spectra[sel]
        self.n_used += int(in_range.sum())

    def _forget_old_delays(self):
        """Forget the delay measurements that cannot be needed anymore"""
        if not self._t_delay:
            return
        history = max(self.history, 2 * self._longest_chunk + 5.)
        oldest_needed = min([t[0] for t, _, _ in self._pending], default=self._t_delay[-1])
        n_old = int(np.searchsorted(self._t_delay, oldest_needed - history))
        if n_old > 0:
            del self._t_delay[:n_old]
            del self._delay[:n_old]

    def flush(self):
        """Bin the chunks still waiting for delay measurements that did not come (their shots after the last
        measurement count as without delay). To call when the acquisition stops."""
        self._process(flush=True)

    # ---- results -----------------------------------------------------------------------------------------------
    @property
    def n_pending_shots(self) -> int:
        return sum(len(t) for t, _, _ in self._pending)

    @property
    def mean_on(self) -> Optional[np.ndarray]:
        """Mean pump-on spectrum of each bin, shape (n_bins, n_pixels); NaN for the empty bins"""
        return self._mean(self._sum_on, self.counts_on)

    @property
    def mean_off(self) -> Optional[np.ndarray]:
        return self._mean(self._sum_off, self.counts_off)

    @staticmethod
    def _mean(sums, counts):
        if sums is None:
            return None
        with np.errstate(invalid='ignore', divide='ignore'):
            return sums / counts[:, None]

    def observable(self, kind: str = 'dR/R') -> Optional[np.ndarray]:
        """The map (n_bins, n_pixels): ``'dR/R'`` = (on - off) / off or ``'ΔOD'`` = -log10(on / off), from the mean
        spectra of the bins. NaN where a bin has no on or no off shot (or a non positive mean for the ΔOD)."""
        on, off = self.mean_on, self.mean_off
        if on is None:
            return None
        with np.errstate(invalid='ignore', divide='ignore'):
            if kind == 'ΔOD':
                return -np.log10(on / off)
            return (on - off) / off


def to_uniform_axis(map_: np.ndarray, axis: np.ndarray, n_points: int = None):
    """Resample the last axis of ``map_`` on a uniform grid (the 2D viewers need linear axes)

    Parameters
    ----------
    map_: np.ndarray
        shape (n_bins, len(axis))
    axis: np.ndarray
        monotonic (increasing or decreasing), possibly non uniform axis
    n_points: int
        number of points of the uniform axis, default twice the number of points of ``axis``

    Returns
    -------
    map, axis: uniformly sampled, with an increasing axis
    """
    axis = np.asarray(axis, dtype=float)
    if axis[0] > axis[-1]:
        axis, map_ = axis[::-1], map_[:, ::-1]
    n_points = n_points or 2 * len(axis)
    new_axis = np.linspace(axis[0], axis[-1], n_points)
    return np.array([np.interp(new_axis, axis, row) for row in map_]), new_axis


def is_uniform(axis: np.ndarray, rtol: float = 1e-6) -> bool:
    """True if ``axis`` is uniformly sampled"""
    steps = np.diff(np.asarray(axis, dtype=float))
    return len(steps) == 0 or bool(np.allclose(steps, steps[0], rtol=rtol, atol=0.))
