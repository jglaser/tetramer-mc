import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from assess_fft_resolution import accumulate, expanded_reference, nested_ranks, recommendation, angular_diagnostic


class ResolutionTest(unittest.TestCase):
    def test_nested_deterministic_all_rotations(self):
        a, b = nested_ranks(100, 71), nested_ranks(100, 71)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(100))
        self.assertEqual(sum(a < 7), 7)
        self.assertTrue(np.all((a < 7) <= (a < 21)))

    def test_exact_separate_tolerances_and_inverse(self):
        pose = dict(position=[3., 2., 1.], orientation=[1., 0., 0., 0.])
        features, owners, chord = expanded_reference([pose], 1., 10.)
        q8 = Rotation.from_euler('x', 8, degrees=True).as_quat()[[3, 0, 1, 2]]
        q11 = Rotation.from_euler('x', 11, degrees=True).as_quat()[[3, 0, 1, 2]]
        # Include valid inverse/sign, one angular false positive, one translation false positive.
        rows = np.array([[0, 0, -1, 0, 0, 0, -3, -2, -1, 5, 0],
                         [1, 0, *q8, 3.9, 2, 1, 6, 0],
                         [2, 0, *q11, 3, 2, 1, 100, 0],
                         [3, 0, 1, 0, 0, 0, 4.1, 2, 1, 200, 0]], float)
        best = np.full((3, 1), -np.inf)
        accumulate(rows, np.arange(4), [1, 2, 4], features, owners, cKDTree(features), best, 1., chord)
        np.testing.assert_array_equal(best[:, 0], [5., 6., 6.])

    def test_streaming_matches_one_pass_and_slot_overlaps(self):
        poses = [dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.]) for x in (3., 3.5)]
        features, owners, chord = expanded_reference(poses, 1., 10.)
        tree = cKDTree(features)
        rows = np.array([[i, 0, 1, 0, 0, 0, 3.2, 0, 0, i+1, 0] for i in range(4)], float)
        a, b = np.full((2, 2), -np.inf), np.full((2, 2), -np.inf)
        ranks = nested_ranks(4, 97)
        accumulate(rows, ranks, [2, 4], features, owners, tree, a, 1., chord)
        for r in rows:
            accumulate(r[None], ranks, [2, 4], features, owners, tree, b, 1., chord)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[-1], [4., 4.])

    def test_tracked_peak_uses_subset_and_preserves_missing(self):
        poses = [dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.]) for x in (3., 30.)]
        features, owners, chord = expanded_reference(poses, 1., 10.)
        tree = cKDTree(features)
        rows = np.array([[i, 0, 1, 0, 0, 0, 3.2, 0, 0, i+1, 0] for i in range(4)], float)
        best, tracked = np.full((2, 2), -np.inf), np.full((2, 11), np.nan)
        for row in rows:
            accumulate(row[None], np.arange(4), [2, 4], features, owners, tree, best, 1., chord, tracked, 0)
        self.assertEqual(tracked[0, 0], 1.)
        self.assertEqual(tracked[0, 9], 2.)
        self.assertTrue(np.all(np.isnan(tracked[1])))
        self.assertEqual(best[1, 0], 4.)

    def test_uncovered_reference_not_silently_counted_as_pass(self):
        best = np.array([[9., -np.inf, -np.inf], [10., 10., -np.inf]])
        choice, rows = recommendation(best, [10, 20])
        self.assertEqual(choice['budget'], 20)
        self.assertEqual(rows[-1]['reference_basins'], 3)
        self.assertEqual(rows[-1]['full_cache_nearby_basins'], 2)
        self.assertEqual(rows[0]['retained_score_fraction'], .5)

    def test_weak_stratum_prevents_selection(self):
        full = np.arange(1., 41.)
        low = full.copy(); low[:3] = -np.inf
        choice, _ = recommendation(np.stack((low, full)), [10, 20], required_fraction=.9, required_stratum=.9)
        self.assertEqual(choice['budget'], 20)

    def test_angle_diagnostic_monotonic_for_identical_probes(self):
        ranks = nested_ranks(100, 72)
        results = angular_diagnostic(100, ranks, [10, 100], 100, 12, 30.)
        self.assertGreaterEqual(results[0]['nearest_angle_degrees']['p95'], results[1]['nearest_angle_degrees']['p95'])
        self.assertEqual(results[0]['random_probes'], 100)


if __name__ == '__main__':
    unittest.main()
