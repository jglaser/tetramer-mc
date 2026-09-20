"""Independent-mean comparison laws, distinct from overlapping prefixes."""
import math
import unittest

from compare_expanded_contact_atlas_repeat import independent_difference


def row(mean, variance, draws):
    return dict(logQ=math.log(mean) if mean else None,
                log_variance_of_mean=math.log(variance) if variance else None,
                row_RSE=math.sqrt(variance)/mean if mean else None, draws=draws)


class ExpandedAtlasRepeatTests(unittest.TestCase):
    def test_independent_difference_adds_two_mean_variances(self):
        result = independent_difference(row(10., 4., 100), row(12., 1., 400))
        self.assertAlmostEqual(math.exp(result['estimated_log_variance_of_difference']), 5.)
        self.assertAlmostEqual(result['difference_over_original_mean'], .2)
        self.assertAlmostEqual(result['observed_SE_difference_over_original_mean'], math.sqrt(5)/10)
        self.assertAlmostEqual(result['observed_difference_in_SE_units'], 2/math.sqrt(5))
        self.assertAlmostEqual(result['N_scaled_observed_variance_ratio'], 1.)

    def test_log_variance_sum_stays_stable_for_large_log_weights(self):
        original = dict(logQ=900., log_variance_of_mean=1795., row_RSE=math.exp(-2.5), draws=100)
        repeat = dict(logQ=901., log_variance_of_mean=1794., row_RSE=math.exp(-4), draws=400)
        result = independent_difference(original, repeat)
        self.assertTrue(math.isfinite(result['observed_difference_in_SE_units']))
        self.assertAlmostEqual(result['estimated_log_variance_of_difference'], 1795+math.log1p(math.exp(-1)))

    def test_unobserved_region_remains_unresolved(self):
        result = independent_difference(row(0., 0., 100), row(12., 1., 400))
        self.assertIsNone(result['repeat_over_original'])
        self.assertIsNone(result['observed_difference_in_SE_units'])
        self.assertIn('Unobserved masks remain unresolved', result['qualification'])


if __name__ == '__main__': unittest.main()
