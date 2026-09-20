"""Population holdouts preserve source weights and exclude held-out geometry."""
import unittest

import numpy as np

from fit_peak_reference_proposal import population_diagnostics
from prepare_native_confirmation_atlas import geometric_floor, weighted_fit


class PeakReferenceFitTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(72, dtype=float).reshape(12, 6)/10
        self.logw = np.log(np.arange(1, 13, dtype=float))
        self.groups = np.repeat(['r00', 'r01', 'r02', 'r03'], 3)
        self.ell = 55.; self.floor = geometric_floor(self.ell)

    def diagnose(self, x=None):
        x = self.x if x is None else x
        fit = weighted_fit(x, self.logw, self.floor)
        return population_diagnostics(x, self.logw, self.groups, self.floor, self.ell,
            np.eye(6), fit, {'geometry_sd_0.2': np.arange(12, dtype=float)})

    def test_mass_fractions_and_scores_retain_original_unequal_weights(self):
        result = self.diagnose()
        np.testing.assert_allclose([r['original_total_weight_fraction'] for r in result],
                                   np.array([6., 15., 24., 33.])/78.)
        self.assertAlmostEqual(result[0]['conditional_expected_log_proposal_density']['geometry_sd_0.2'], 8/6)
        self.assertEqual(result[0]['heldout_nonzero_rows'], 3)
        self.assertEqual(result[0]['training_nonzero_rows'], 9)
        self.assertAlmostEqual(result[0]['heldout_weight_ESS'], 36/14)

    def test_heldout_geometry_does_not_enter_its_training_fit(self):
        original = self.diagnose()
        shifted = self.x.copy(); shifted[:3] += [3., 2., 1., .4, .3, .2]
        changed = self.diagnose(shifted)
        for key in ('mean', 'raw_covariance', 'covariance'):
            np.testing.assert_array_equal(original[0]['leave_one_population_out_fit'][key],
                                          changed[0]['leave_one_population_out_fit'][key])
        self.assertNotEqual(original[0]['conditional_expected_log_proposal_density']['weighted_leave_one_population_out'],
                            changed[0]['conditional_expected_log_proposal_density']['weighted_leave_one_population_out'])
        self.assertFalse(np.array_equal(original[1]['leave_one_population_out_fit']['mean'],
                                        changed[1]['leave_one_population_out_fit']['mean']))


if __name__ == '__main__': unittest.main()
