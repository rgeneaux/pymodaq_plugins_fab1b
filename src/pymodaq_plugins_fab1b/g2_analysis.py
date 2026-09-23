"""Pure-numpy math for the Marana-X HBT g2 (coincidence) plugin.

No Qt / PyMoDAQ imports (same ethos as ``hardware/andor_sdk3.py`` and
``hardware/uniblitz_vlm1.py``): independently unit-testable, and usable from
a stand-alone script for an on-bench feasibility check before the full
plugin is exercised.

Area layout: 4 harmonics, each split into two "single-photon-detector"
areas A and B by a 2D checkerboard - pixel (r, c) within a harmonic's own
window belongs to A if (r + c) is even, else B, with r/c counted from that
window's own top-left corner (row 0 = the AOI's own top row; col 0 = the
harmonic's own declared left edge), so every harmonic's split is in phase
regardless of where its window sits or how it is clipped by the AOI.
Areas are indexed 0..7 in the fixed order H1-A, H1-B, H2-A, H2-B, H3-A,
H3-B, H4-A, H4-B; :func:`area_label` is the single source of truth for that
naming, reused by the plugin for both settings-tree fields and plot labels.
"""

from __future__ import annotations

import numpy as np

N_HARMONICS = 4
N_AREAS = 2 * N_HARMONICS


def area_label(area_index):
    """'H<n>-A' / 'H<n>-B' for area_index in 0..7, in harmonic/A-B order."""
    harmonic = area_index // 2 + 1
    side = "A" if area_index % 2 == 0 else "B"
    return f"H{harmonic}-{side}"


def harmonics_overlap(harmonics):
    """Pairs of harmonic indices whose declared (left, width) windows overlap.

    A misconfiguration (not physically meaningful - two harmonics can't
    share sensor columns) that would otherwise double-assign pixels
    silently; the caller should warn rather than build masks from it.
    """
    overlaps = []
    for i in range(len(harmonics)):
        li, wi = harmonics[i]
        for j in range(i + 1, len(harmonics)):
            lj, wj = harmonics[j]
            if li < lj + wj and lj < li + wi:
                overlaps.append((i, j))
    return overlaps


def build_area_masks(harmonics, aoi_left, aoi_width, aoi_height):
    """Checkerboard A/B pixel masks for the 4 harmonics, in AOI-local pixels.

    Parameters
    ----------
    harmonics : sequence of 4 (left, width) pairs
        Each harmonic's window, in absolute sensor pixel columns (same
        convention as the base plugin's own ``aoi`` left/top), and its width
        in columns.
    aoi_left, aoi_width, aoi_height : int
        The camera's currently configured AOI (absolute left column, and its
        width/height), used to translate the harmonics' absolute-column
        windows into indices into an (aoi_height, aoi_width) frame array.

    Returns
    -------
    np.ndarray, shape (8, aoi_height, aoi_width), bool
        One geometry mask per area (see module docstring for the area
        order). A harmonic window that falls partially or fully outside the
        current AOI is silently clipped - the checkerboard phase is anchored
        to the harmonic's own declared left edge, not to whatever is
        visible, so clipping never shifts which pixels are A vs B.
    """
    if len(harmonics) != N_HARMONICS:
        raise ValueError(f"Expected {N_HARMONICS} harmonics, got {len(harmonics)}")

    masks = np.zeros((N_AREAS, aoi_height, aoi_width), dtype=bool)
    rows = np.arange(aoi_height)[:, None]

    for h, (h_left, h_width) in enumerate(harmonics):
        col_start = h_left - aoi_left  # AOI-local column where the window begins (may be < 0)
        col_end = col_start + h_width  # (may be > aoi_width)
        clipped_start = max(col_start, 0)
        clipped_end = min(col_end, aoi_width)
        if clipped_end <= clipped_start:
            continue  # window entirely outside the current AOI

        # Column index local to the harmonic's OWN window (0-based from its
        # declared left edge), not to the clipped/visible range - this is
        # what keeps the checkerboard phase fixed regardless of clipping.
        cols_local = np.arange(clipped_start, clipped_end) - col_start
        parity = (rows + cols_local[None, :]) % 2

        area_a, area_b = 2 * h, 2 * h + 1
        masks[area_a, :, clipped_start:clipped_end] = parity == 0
        masks[area_b, :, clipped_start:clipped_end] = parity == 1

    return masks


def apply_noise_mask(std, std_threshold):
    """Per-pixel "not too noisy" mask: True where dark std < std_threshold.

    Only std matters for this decision (per spec: "keep only the pixels for
    which the standard deviation is below 10") - the per-pixel dark mean is
    used separately, in :func:`detection_thresholds`.
    """
    return np.asarray(std) < std_threshold


def valid_area_masks(area_masks, noise_ok):
    """Combine checkerboard geometry with the noise mask: (8, H, W) bool."""
    return area_masks & noise_ok[None, :, :]


def area_pixel_counts(masks):
    """Per-area pixel counts from an (8, H, W) (or any per-area) bool array."""
    return masks.reshape(masks.shape[0], -1).sum(axis=1)


def calibration_warnings(counts):
    """Human-readable warnings for a (8,) array of per-area valid-pixel counts.

    Empty list if every area has at least one valid pixel. Two distinct
    warnings are possible: every area lost all its pixels (the whole
    calibration is unusable), or just some specific area(s) did (those
    areas can never click, but the rest of the measurement can proceed).
    """
    counts = np.asarray(counts)
    if counts.sum() == 0:
        return [
            "No valid pixels remain in ANY area: the Std threshold removed "
            "every pixel. Raise the threshold."
        ]
    empty = [i for i, c in enumerate(counts) if c == 0]
    if empty:
        labels = ", ".join(area_label(i) for i in empty)
        return [
            f"{len(empty)} area(s) have zero valid pixels: {labels}. Those "
            f"areas can never click; lower the Std threshold or check their "
            f"harmonic window."
        ]
    return []


def detection_thresholds(mean, margin):
    """Per-pixel click threshold = dark mean + margin.

    Cheap and instant: margin can be retuned live without recalibrating,
    since the raw per-pixel mean is kept from calibration regardless of the
    current margin value.
    """
    return np.asarray(mean) + margin


def extract_clicks(frame, valid_area_masks, thresholds):
    """One shot's (8,) bool click vector: any valid pixel above its threshold."""
    above = np.asarray(frame) > thresholds
    return (valid_area_masks & above[None, :, :]).any(axis=(1, 2))


def combine_harmonic_clicks(click_history):
    """(n_shots, 8) -> (n_shots, 4): OR of each harmonic's A and B click.

    Valid only for combining two *different* harmonics' channels (cross g2):
    a single harmonic's own A/B pair must stay split to avoid the
    self-correlation bias that split exists to prevent (see module
    docstring / plan).
    """
    ch = np.asarray(click_history, dtype=bool)
    return ch[:, 0::2] | ch[:, 1::2]


def click_counts(click_history):
    """(n_shots, 8) -> (8,) int: total clicks per area, since accumulation began."""
    return np.asarray(click_history, dtype=np.int64).sum(axis=0)


def _pair_g2(n_i, n_j, rng):
    """g2_raw/accidental/corrected for one pair of 0/1 click channels.

    g2_raw is the same-shot estimator <n_i n_j> / (<n_i><n_j>). g2_accidental
    is the same formula with n_j's shot order randomly shuffled - the
    standard HBT "accidental coincidence" baseline, robust against slow
    common-mode drift (unlike a fixed-lag shift). g2_corrected = raw /
    accidental cancels common multiplicative systematics. NaN wherever a
    channel has zero mean (no clicks yet, or a dead/fully-excluded area) -
    the ratio is undefined, not zero.
    """
    n_i = np.asarray(n_i, dtype=np.float64)
    n_j = np.asarray(n_j, dtype=np.float64)
    mean_i, mean_j = n_i.mean(), n_j.mean()
    if mean_i <= 0 or mean_j <= 0:
        return np.nan, np.nan, np.nan

    g2_raw = (n_i * n_j).mean() / (mean_i * mean_j)

    n_j_shuffled = n_j[rng.permutation(n_j.shape[0])]
    mean_j_shuffled = n_j_shuffled.mean()  # == mean_j; recomputed for self-consistency
    if mean_j_shuffled <= 0:
        return g2_raw, np.nan, np.nan
    g2_accidental = (n_i * n_j_shuffled).mean() / (mean_i * mean_j_shuffled)

    g2_corrected = g2_raw / g2_accidental if g2_accidental > 0 else np.nan
    return g2_raw, g2_accidental, g2_corrected


def g2_matrix(click_history, rng=None):
    """4x4 harmonic x harmonic g2 matrices from an (n_shots, 8) click history.

    Diagonal[h]: harmonic h's own g2(0), from its A vs B channels (same-shot
    correlation of two independent sub-detectors of the same beam - the
    reason the A/B split exists at all). Off-diagonal[i, j]: cross g2
    between harmonics i and j, from their OR-combined (A | B) click channels
    (no self-correlation risk between two already-independent physical
    channels, so combining is valid and doubles the statistics). Symmetric
    by construction.

    Returns a dict with 'g2_raw', 'g2_accidental', 'g2_corrected', each a
    (4, 4) float array (NaN where undefined, e.g. too few shots so far).
    """
    if rng is None:
        rng = np.random.default_rng()
    click_history = np.asarray(click_history)
    if click_history.ndim != 2 or click_history.shape[1] != N_AREAS:
        raise ValueError(f"click_history must have shape (n_shots, {N_AREAS})")

    combined = combine_harmonic_clicks(click_history)

    g2_raw = np.full((N_HARMONICS, N_HARMONICS), np.nan)
    g2_acc = np.full((N_HARMONICS, N_HARMONICS), np.nan)
    g2_corr = np.full((N_HARMONICS, N_HARMONICS), np.nan)

    for h in range(N_HARMONICS):
        raw, acc, corr = _pair_g2(click_history[:, 2 * h], click_history[:, 2 * h + 1], rng)
        g2_raw[h, h], g2_acc[h, h], g2_corr[h, h] = raw, acc, corr

    for i in range(N_HARMONICS):
        for j in range(i + 1, N_HARMONICS):
            raw, acc, corr = _pair_g2(combined[:, i], combined[:, j], rng)
            g2_raw[i, j] = g2_raw[j, i] = raw
            g2_acc[i, j] = g2_acc[j, i] = acc
            g2_corr[i, j] = g2_corr[j, i] = corr

    return {"g2_raw": g2_raw, "g2_accidental": g2_acc, "g2_corrected": g2_corr}


class DarkCalibrationAccumulator:
    """Incremental per-pixel mean/std over many dark frames.

    O(frame size) memory regardless of how many frames are added - the
    10,000-shot dark calibration never needs to hold more than one frame's
    worth of running sum/sum-of-squares at a time.
    """

    def __init__(self, shape):
        self.shape = shape
        self.n = 0
        self._sum = np.zeros(shape, dtype=np.float64)
        self._sumsq = np.zeros(shape, dtype=np.float64)

    def add(self, frame):
        frame = np.asarray(frame, dtype=np.float64)
        self._sum += frame
        self._sumsq += frame * frame
        self.n += 1

    @property
    def mean(self):
        if self.n == 0:
            raise RuntimeError("No frames accumulated yet")
        return self._sum / self.n

    @property
    def std(self):
        if self.n == 0:
            raise RuntimeError("No frames accumulated yet")
        mean = self.mean
        # Population variance; a tiny floating-point-cancellation negative
        # is possible for a near-constant pixel, hence the clip before sqrt.
        variance = np.clip(self._sumsq / self.n - mean * mean, 0.0, None)
        return np.sqrt(variance)
