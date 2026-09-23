"""g2_analysis: area geometry, noise masking, click extraction, and the g2 math."""
import unittest

import numpy as np

from pymodaq_plugins_fab1b import g2_analysis as g2a


class AreaLabelTest(unittest.TestCase):
    def test_order_and_naming(self):
        labels = [g2a.area_label(i) for i in range(8)]
        self.assertEqual(
            labels,
            ["H1-A", "H1-B", "H2-A", "H2-B", "H3-A", "H3-B", "H4-A", "H4-B"],
        )


class HarmonicsOverlapTest(unittest.TestCase):
    def test_no_overlap(self):
        harmonics = [(0, 10), (10, 10), (30, 10), (50, 10)]
        self.assertEqual(g2a.harmonics_overlap(harmonics), [])

    def test_detects_overlap(self):
        harmonics = [(0, 10), (5, 10), (30, 10), (50, 10)]
        self.assertEqual(g2a.harmonics_overlap(harmonics), [(0, 1)])


class BuildAreaMasksTest(unittest.TestCase):
    def test_checkerboard_parity_local_to_each_window(self):
        # Two harmonics, both fully inside a small AOI. Window-local (0,0) is
        # always area A, independent of the harmonic's absolute position.
        harmonics = [(2, 4), (10, 4), (100, 4), (200, 4)]
        masks = g2a.build_area_masks(harmonics, aoi_left=0, aoi_width=20, aoi_height=3)
        self.assertEqual(masks.shape, (8, 3, 20))

        # Harmonic 1 window: AOI-local columns 2..5, local col 0 at col 2.
        # (row, local_col) parity: row0,col0 -> A ; row0,col1 -> B ; etc.
        self.assertTrue(masks[0, 0, 2])   # H1-A: row0, local col0 (even)
        self.assertFalse(masks[1, 0, 2])
        self.assertTrue(masks[1, 0, 3])   # H1-B: row0, local col1 (odd)
        self.assertTrue(masks[0, 1, 3])   # H1-A: row1, local col1 -> sum=2, even

        # Harmonic 2 window starts at a DIFFERENT absolute column (10) but
        # its own local col0 is still area A - phase is anchored per-harmonic.
        self.assertTrue(masks[2, 0, 10])  # H2-A: row0, local col0
        self.assertTrue(masks[3, 0, 11])  # H2-B: row0, local col1

    def test_areas_within_one_harmonic_are_disjoint_and_cover_the_window(self):
        harmonics = [(0, 6), (20, 6), (40, 6), (60, 6)]
        masks = g2a.build_area_masks(harmonics, aoi_left=0, aoi_width=80, aoi_height=5)
        for h in range(4):
            a, b = masks[2 * h], masks[2 * h + 1]
            window = np.zeros_like(a)
            window[:, harmonics[h][0]:harmonics[h][0] + harmonics[h][1]] = True
            self.assertFalse(np.any(a & b), "A and B overlap")
            np.testing.assert_array_equal(a | b, window)

    def test_different_harmonics_do_not_overlap(self):
        harmonics = [(0, 10), (10, 10), (30, 10), (50, 10)]
        masks = g2a.build_area_masks(harmonics, aoi_left=0, aoi_width=70, aoi_height=4)
        combined = masks.any(axis=0)
        total_true = masks.sum()
        self.assertEqual(total_true, combined.sum(), "some pixel was claimed by >1 area")

    def test_window_partially_outside_aoi_is_clipped_without_shifting_phase(self):
        # Harmonic starts 3 columns before the AOI: only cols 3..width-1 of
        # its own window are visible, but they must keep their ORIGINAL
        # (unshifted) local parity, not restart from local col 0.
        harmonics = [(-3, 6), (20, 6), (40, 6), (60, 6)]
        masks = g2a.build_area_masks(harmonics, aoi_left=0, aoi_width=80, aoi_height=1)
        # Visible AOI-local columns 0..2 correspond to local cols 3..5 of H1.
        # local col3 (row0) -> parity odd -> B; local col4 -> even -> A; local col5 -> odd -> B
        self.assertTrue(masks[1, 0, 0])   # H1-B
        self.assertTrue(masks[0, 0, 1])   # H1-A
        self.assertTrue(masks[1, 0, 2])   # H1-B

    def test_window_fully_outside_aoi_yields_no_pixels(self):
        harmonics = [(-100, 6), (20, 6), (40, 6), (60, 6)]
        masks = g2a.build_area_masks(harmonics, aoi_left=0, aoi_width=80, aoi_height=1)
        self.assertFalse(masks[0].any())
        self.assertFalse(masks[1].any())

    def test_wrong_harmonic_count_raises(self):
        with self.assertRaises(ValueError):
            g2a.build_area_masks([(0, 6), (10, 6)], aoi_left=0, aoi_width=30, aoi_height=2)


class NoiseMaskTest(unittest.TestCase):
    def test_apply_noise_mask_thresholds_std(self):
        std = np.array([[1.0, 20.0], [9.9, 10.0]])
        mask = g2a.apply_noise_mask(std, std_threshold=10.0)
        np.testing.assert_array_equal(mask, [[True, False], [True, False]])

    def test_valid_area_masks_combines_geometry_and_noise(self):
        area_masks = np.zeros((8, 2, 2), dtype=bool)
        area_masks[0] = [[True, True], [False, False]]
        noise_ok = np.array([[True, False], [True, True]])
        valid = g2a.valid_area_masks(area_masks, noise_ok)
        np.testing.assert_array_equal(valid[0], [[True, False], [False, False]])

    def test_area_pixel_counts(self):
        masks = np.zeros((8, 3, 3), dtype=bool)
        masks[0, 0, :] = True  # 3 pixels
        masks[1, 0, 0] = True  # 1 pixel
        counts = g2a.area_pixel_counts(masks)
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 1)
        self.assertTrue(all(c == 0 for c in counts[2:]))


class CalibrationWarningsTest(unittest.TestCase):
    def test_no_warning_when_every_area_has_pixels(self):
        self.assertEqual(g2a.calibration_warnings(np.array([1] * 8)), [])

    def test_warns_when_one_area_is_empty(self):
        counts = np.array([5, 0, 5, 5, 5, 5, 5, 5])
        warnings = g2a.calibration_warnings(counts)
        self.assertEqual(len(warnings), 1)
        self.assertIn("H1-B", warnings[0])

    def test_warns_distinctly_when_every_area_is_empty(self):
        warnings = g2a.calibration_warnings(np.zeros(8, dtype=int))
        self.assertEqual(len(warnings), 1)
        self.assertIn("ANY area", warnings[0])

    def test_reports_all_empty_areas(self):
        counts = np.array([0, 0, 5, 5, 5, 5, 5, 5])
        warnings = g2a.calibration_warnings(counts)
        self.assertIn("H1-A", warnings[0])
        self.assertIn("H1-B", warnings[0])


class DetectionThresholdsTest(unittest.TestCase):
    def test_threshold_is_mean_plus_margin(self):
        mean = np.array([[100.0, 200.0]])
        thresholds = g2a.detection_thresholds(mean, margin=10.0)
        np.testing.assert_array_equal(thresholds, [[110.0, 210.0]])

    def test_margin_change_needs_no_recalibration(self):
        mean = np.array([100.0, 200.0])
        t1 = g2a.detection_thresholds(mean, margin=10.0)
        t2 = g2a.detection_thresholds(mean, margin=25.0)
        np.testing.assert_array_equal(t2 - t1, [15.0, 15.0])


class ExtractClicksTest(unittest.TestCase):
    def test_hot_pixel_in_area_a_only_registers_area_a(self):
        area_masks = np.zeros((8, 2, 2), dtype=bool)
        area_masks[0] = [[True, False], [False, False]]  # H1-A: one pixel
        area_masks[1] = [[False, True], [False, False]]  # H1-B: one pixel
        thresholds = np.full((2, 2), 100.0)
        frame = np.array([[150.0, 50.0], [0.0, 0.0]])  # only the H1-A pixel is hot
        clicks = g2a.extract_clicks(frame, area_masks, thresholds)
        self.assertTrue(clicks[0])
        self.assertFalse(clicks[1])
        self.assertFalse(clicks[2:].any())

    def test_masked_out_pixel_never_clicks(self):
        area_masks = np.zeros((8, 1, 1), dtype=bool)  # excluded everywhere
        thresholds = np.array([[0.0]])
        frame = np.array([[1000.0]])
        clicks = g2a.extract_clicks(frame, area_masks, thresholds)
        self.assertFalse(clicks.any())

    def test_click_requires_strictly_above_threshold(self):
        area_masks = np.zeros((8, 1, 1), dtype=bool)
        area_masks[0] = [[True]]
        thresholds = np.array([[10.0]])
        self.assertFalse(g2a.extract_clicks(np.array([[10.0]]), area_masks, thresholds)[0])
        self.assertTrue(g2a.extract_clicks(np.array([[10.1]]), area_masks, thresholds)[0])


class CombineHarmonicClicksTest(unittest.TestCase):
    def test_is_or_of_a_and_b(self):
        history = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        combined = g2a.combine_harmonic_clicks(history)
        np.testing.assert_array_equal(combined[:, 0], [True, True, True, False])
        np.testing.assert_array_equal(combined[:, 1:], np.zeros((4, 3), dtype=bool))


class ClickCountsTest(unittest.TestCase):
    def test_sums_over_shots(self):
        history = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0]])
        counts = g2a.click_counts(history)
        np.testing.assert_array_equal(counts, [2, 1, 1, 0, 2, 1, 1, 0])


class G2MatrixTest(unittest.TestCase):
    def test_uncorrelated_clicks_give_g2_near_one(self):
        rng = np.random.default_rng(0)
        n_shots = 2_000_000
        # Independent Bernoulli draws per area: no physical correlation at all.
        history = rng.random((n_shots, 8)) < 0.05
        result = g2a.g2_matrix(history, rng=np.random.default_rng(1))
        corr = result["g2_corrected"]
        self.assertTrue(np.all(np.isfinite(corr)))
        # 10 independent g2 estimates at p=0.05; loose enough to comfortably
        # cover shot-noise scatter across all of them, tight enough to catch
        # an actual formula bug (which tends to show up as a large offset).
        np.testing.assert_allclose(corr, np.ones((4, 4)), atol=0.05)

    def test_deliberately_bunched_clicks_give_g2_above_one(self):
        rng = np.random.default_rng(0)
        n_shots = 200_000
        # A shared "bright shot" latent variable correlates every area's click
        # probability shot-to-shot: classic bunching/super-Poissonian signature.
        bright = rng.random(n_shots) < 0.1
        p = np.where(bright, 0.4, 0.02)
        history = rng.random((n_shots, 8)) < p[:, None]
        result = g2a.g2_matrix(history, rng=np.random.default_rng(1))
        corr = result["g2_corrected"]
        self.assertTrue(np.all(corr > 1.2), corr)

    def test_matrix_is_symmetric(self):
        rng = np.random.default_rng(0)
        history = rng.random((5000, 8)) < 0.1
        result = g2a.g2_matrix(history, rng=np.random.default_rng(1))
        for key in ("g2_raw", "g2_accidental", "g2_corrected"):
            m = result[key]
            np.testing.assert_allclose(m, m.T, equal_nan=True)

    def test_dead_area_gives_nan_not_zero_or_crash(self):
        history = np.zeros((1000, 8), dtype=bool)
        history[:, 1:] = np.random.default_rng(0).random((1000, 7)) < 0.1  # area 0 (H1-A) never clicks
        result = g2a.g2_matrix(history, rng=np.random.default_rng(1))
        self.assertTrue(np.isnan(result["g2_corrected"][0, 0]))  # H1's own g2 needs H1-A
        self.assertFalse(np.isnan(result["g2_corrected"][1, 1]))  # H2's own g2 is unaffected

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            g2a.g2_matrix(np.zeros((10, 7)))


class DarkCalibrationAccumulatorTest(unittest.TestCase):
    def test_exact_mean_and_std_on_a_small_known_set(self):
        acc = g2a.DarkCalibrationAccumulator(shape=(2,))
        for frame in ([1.0, 10.0], [3.0, 10.0], [5.0, 10.0]):
            acc.add(frame)
        np.testing.assert_allclose(acc.mean, [3.0, 10.0])
        # population std of [1,3,5] is sqrt(mean((x-3)^2)) = sqrt(8/3)
        np.testing.assert_allclose(acc.std, [np.sqrt(8 / 3), 0.0], atol=1e-12)

    def test_converges_to_true_mean_and_std_for_random_data(self):
        rng = np.random.default_rng(0)
        true_mean, true_std = 100.0, 5.0
        acc = g2a.DarkCalibrationAccumulator(shape=(3, 3))
        for _ in range(20_000):
            acc.add(rng.normal(true_mean, true_std, size=(3, 3)))
        np.testing.assert_allclose(acc.mean, np.full((3, 3), true_mean), atol=0.3)
        np.testing.assert_allclose(acc.std, np.full((3, 3), true_std), atol=0.3)

    def test_access_before_any_frame_raises(self):
        acc = g2a.DarkCalibrationAccumulator(shape=(2,))
        with self.assertRaises(RuntimeError):
            _ = acc.mean
        with self.assertRaises(RuntimeError):
            _ = acc.std


if __name__ == "__main__":
    unittest.main()
