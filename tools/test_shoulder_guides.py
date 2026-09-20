#!/usr/bin/env python3
"""Controls for shoulder-guide weighting, frames, mixture and strict partition."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from analyze_shoulder_guides import statistics

from prepare_shoulder_guides import (
    AtomUnionAudit, Density, combine_models, fit_guides, geometry_probes,
    geometric_floor, relative_poses, shoulder,
)


def pose(position, rotation=None):
    rotation = Rotation.identity() if rotation is None else rotation
    return {'position': list(position), 'orientation': rotation.as_quat()[[3, 0, 1, 2]].tolist()}


class ShoulderGuidesTests(unittest.TestCase):
    def test_band_statistics_keep_unconditional_zero_draws(self):
        result = statistics(np.log([1., 3.]).tolist(), 4)
        self.assertAlmostEqual(result['logQ'], 0.)
        self.assertAlmostEqual(result['ESS'], 1.6)
        self.assertAlmostEqual(result['observed_RSE'], np.sqrt(.5))
        self.assertAlmostEqual(result['maximum_fraction'], .75)
        empty = statistics([], 4)
        self.assertIsNone(empty['logQ'])
        self.assertIsNone(empty['observed_RSE'])

    def test_weighted_fit_retains_low_weight_poses_and_cross_covariance(self):
        fixed = pose([3., -2., 1.], Rotation.from_rotvec([.1, -.4, .7]))
        native = pose([0., 0., 0.])
        poses = [pose([0., 0., 0.]), pose([1., 0., 0.], Rotation.from_rotvec([.02, 0., 0.])),
                 pose([2., 0., 0.], Rotation.from_rotvec([.04, 0., 0.]))]
        models, fits, audit = fit_guides(poses, np.log([1., 8., 1.]), fixed, native, 'fixture')
        self.assertAlmostEqual(fits['weighted']['normalized_weight_ESS'], 100/66)
        self.assertAlmostEqual(fits['geometry']['normalized_weight_ESS'], 3.)
        for fit in fits.values():
            np.testing.assert_allclose(fit['covariance']-fit['raw_covariance'], geometric_floor(), atol=1e-15)
            self.assertGreater(np.linalg.norm(fit['raw_covariance'][:3, 3:]), .01)
        self.assertGreater(np.trace(fits['geometry']['raw_covariance']), np.trace(fits['weighted']['raw_covariance']))
        relative = relative_poses(poses, fixed)
        mixture = Density(models['mixture']).evaluate(relative)[0]
        expected = np.logaddexp(Density(models['weighted']).evaluate(relative)[0],
                               Density(models['geometry']).evaluate(relative)[0])-np.log(2)
        np.testing.assert_allclose(mixture, expected, atol=1e-13)
        self.assertLess(audit['maximum_chart_angle_deg'], 3.)
        self.assertEqual(models['mixture']['weights'], [.5, .5])

    def test_large_chart_angles_fail_instead_of_dropping_rows(self):
        p = pose([0., 0., 0.])
        with self.assertRaises(AssertionError):
            fit_guides([p, pose([0., 0., 0.], Rotation.from_rotvec([1.3, 0, 0]))],
                       np.zeros(2), p, p, 'fixture')

    def test_mixture_copy_preserves_component_fits(self):
        p = pose([0., 0., 0.])
        models, _, _ = fit_guides([p], [0.], p, p, 'fixture')
        value = models['weighted']['covariances'][0][0][0]
        models['mixture']['covariances'][0][0][0] = 100.
        self.assertEqual(models['weighted']['covariances'][0][0][0], value)
        invalid = dict(models['geometry'], angular_length=1.)
        with self.assertRaises(AssertionError): combine_models(models['weighted'], invalid)

    def test_geometry_probe_keeps_invalid_draws_and_both_neighbors(self):
        native = pose([0., 0., 0.]); fixed = pose([20., 0., 0.])
        observed = [pose([3., 0., 0.]), pose([3.01, 0., 0.])]
        models, _, _ = fit_guides(observed, [0., 0.], fixed, native, 'fixture')
        cfg = {'fixed_poses': [fixed, pose([3., 0., 0.])], 'capture_center': [0., 0., 0.],
               'capture_radius': 18., 'depletant_radius': 1.5,
               'metadata': {'native_poses': [native], 'rigid_members': [native],
                            'member_error_scale': 2., 'angle_error_scale_deg': 15.}}
        audit = AtomUnionAudit({'atoms': [{'center': [0., 0., 0.], 'radius': .4}]}, cfg['fixed_poses'])
        result, rows = geometry_probes(models['mixture'], cfg, audit, 32, 98465)
        self.assertEqual(len(rows), 32)
        self.assertEqual(result['valid_shoulder'], 0)
        self.assertEqual(result['hard_valid'], 0)
        self.assertEqual(result['capture_valid'], 32)
        self.assertEqual(result['shoulder'], 32)
        self.assertTrue(all(r['minimum_gap_by_neighbor_A'][0] > 0 and
                            r['minimum_gap_by_neighbor_A'][1] < 0 for r in rows))
        self.assertEqual(sum(result['component_draws']), 32)
        self.assertTrue(all(n > 0 for n in result['component_draws']))

    def test_original_partition_is_strict(self):
        self.assertFalse(shoulder(1.))
        self.assertFalse(shoulder(2.))
        self.assertTrue(shoulder(np.nextafter(1., 2.)))
        self.assertTrue(shoulder(np.nextafter(2., 1.)))


if __name__ == '__main__':
    unittest.main()
