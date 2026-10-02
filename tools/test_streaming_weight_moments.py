import math
import unittest

import numpy as np
from scipy.special import logsumexp

from analyze_basin_normalizers import moments, paired_noise
from streaming_weight_moments import PairedNoise, RegionMoments, WeightMoments


class StreamingMomentsTests(unittest.TestCase):
    def compare(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if isinstance(left[key], float):
                self.assertTrue(math.isclose(left[key], right[key], rel_tol=2e-11, abs_tol=2e-12),
                                (key, left[key], right[key]))
            else:
                self.assertEqual(left[key], right[key], key)

    def test_zero_sparse_and_extreme_weights_match_existing_estimator(self):
        rng = np.random.default_rng(1531)
        examples = [[None], [0.], [None]*213, [None]*999+[1200.],
                    [1000.+x if i % 7 else None for i, x in enumerate(rng.normal(0, 18, 1001))],
                    [-1000.+x for x in rng.normal(0, .1, 250)]]
        for values in examples:
            for ordered in (values, values[::-1]):
                with self.subTest(n=len(values), nonzero=sum(v is not None for v in values)):
                    reducer = WeightMoments(len(ordered))
                    for value in ordered: reducer.add(value)
                    expected = moments([-np.inf if v is None else v for v in ordered])
                    self.compare(reducer.result(), expected)
                    self.assertLessEqual(len(reducer.top), math.ceil(len(values)/100))

    def test_extreme_common_offset_does_not_overflow_square_weight(self):
        reducer = WeightMoments(4)
        for value in [1e308, None, 1e308, None]: reducer.add(value)
        result = reducer.result()
        self.assertEqual(result['logQ'], 1e308)
        self.assertEqual(result['ess'], 2.)
        self.assertEqual(result['max_fraction'], .5)

    def test_paired_cloud_noise_matches_array_oracle_with_unconditional_zeros(self):
        rng = np.random.default_rng(948)
        for offset in [-1000., 0., 1000.]:
            pairs = rng.normal(0, 2, size=(513, 2))+offset
            pairs[::3] = -np.inf
            logs = logsumexp(pairs, axis=1)-math.log(2)
            for order in [np.arange(len(pairs)), np.arange(len(pairs))[::-1]]:
                reducer = PairedNoise(len(pairs))
                for a, b in pairs[order]:
                    reducer.add(None if a == -np.inf else float(a), None if b == -np.inf else float(b))
                self.compare(reducer.result(), paired_noise(logs[order], pairs[order]))

    def test_negative_residual_is_retained_and_zero_populations_are_unresolved(self):
        reducer = PairedNoise(2)
        reducer.add(math.log(2), math.log(.25)); reducer.add(math.log(.25), math.log(2))
        result = reducer.result()
        self.assertEqual(result['scaled_total_variance'], 0.)
        self.assertLess(result['scaled_residual_pose_variance'], 0.)
        self.assertLess(result['relative_variance_of_mean_pose'], 0.)
        zero = PairedNoise(2); zero.add(None, None); zero.add(None, None)
        self.assertIsNone(zero.result())

    def test_attempt_accounting_and_nonfinite_values_fail(self):
        for cls in [WeightMoments, PairedNoise, RegionMoments]:
            with self.assertRaises(ValueError): cls(0)
        reducer = WeightMoments(2); reducer.add(0.)
        with self.assertRaises(ValueError): reducer.result()
        reducer.add(None)
        with self.assertRaises(ValueError): reducer.add(None)
        for value in [math.inf, -math.inf, math.nan, True, '0']:
            with self.subTest(value=value), self.assertRaises(ValueError): WeightMoments(1).add(value)
        with self.assertRaises(ValueError): PairedNoise(1).add(None, 0.)
        with self.assertRaises(ValueError): PairedNoise(2).result()

    def test_region_selection_retains_invalid_and_other_region_zeros(self):
        row = dict(hard_valid=True, log_importance_weight=math.log(3.), log_hard_weight=0.,
                   log_proposal_density=0., clouds=[dict(log_weight=math.log(2.)), dict(log_weight=math.log(4.))])
        reducer = RegionMoments(3)
        reducer.add(row, True); reducer.add(row, False); reducer.add(dict(hard_valid=False), False)
        result = reducer.result()
        self.assertAlmostEqual(result['Qz']['logQ'], 0.)
        self.assertAlmostEqual(result['Q0']['logQ'], -math.log(3.))
        self.assertEqual(result['Qz']['draws'], 3)
        self.assertEqual(result['Qz']['nonzero'], 1)
        self.assertEqual(result['Qz']['max_fraction'], 1.)
        with self.assertRaises(ValueError): RegionMoments(1).add(dict(row, hard_valid=False), True)


if __name__ == '__main__': unittest.main()
