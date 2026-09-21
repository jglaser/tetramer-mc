"""Geometric containment and unchanged disjoint-stratum controls; no physical sampling."""
import copy
import unittest
import numpy as np
from prepare_mobile_competing_outer_weights import shell_region, containment, validate_partition


class OuterRegionTests(unittest.TestCase):
    def fixture(self):
        region = dict(minimum_mahalanobis_radius=3., mahalanobis_radius=5.,
                      minimum_original_q=1., minimum_original_q_inclusive=False,
                      fixed_neighbor=dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.]),
                      gaussian_chart=dict(anchors=[dict(position=[2., 0., 0.])],
                                          means=[[0.]*6], weights=[1.], angular_length=5.,
                                          covariances=[np.diag([.25**2]*3+[.12**2]*3).tolist()]))
        config = dict(capture_center=[0., 0., 0.], capture_radius=8.,
                      metadata=dict(physical_sphere_radius_A=12.))
        shape = dict(atoms=[dict(center=[1., 0., 0.], radius=1.)])
        return region, config, shape

    def test_region_extension_only_changes_declared_radial_bounds(self):
        old, _, _ = self.fixture()
        source = copy.deepcopy(old)
        new = shell_region(old, 5., 8.)
        self.assertEqual(old, source)
        for key in old.keys()-{'minimum_mahalanobis_radius', 'mahalanobis_radius', 'definition'}:
            self.assertEqual(new[key], old[key])
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            shell_region(old, 8., 5.)

    def test_partition_includes_boundaries_exactly_once(self):
        old, _, _ = self.fixture()
        regions = [shell_region(old, 5., 8.), shell_region(old, 8., 12.)]
        self.assertTrue(validate_partition(regions)['disjoint'])
        regions[1]['minimum_original_q'] = 2.
        with self.assertRaisesRegex(ValueError, 'changed more'):
            validate_partition(regions)

    def test_capture_and_wall_certificate_are_distinct(self):
        old, config, shape = self.fixture()
        region = shell_region(old, 8., 12.)
        result = containment(region, config, shape)
        self.assertAlmostEqual(result['entire_region_center_norm_upper_bound_A'], 6.)
        self.assertAlmostEqual(result['guaranteed_atomic_wall_clearance_A'], 2.)
        config['capture_radius'] = 5.
        with self.assertRaisesRegex(ValueError, 'inside capture'):
            containment(region, config, shape)
        config['capture_radius'] = 11.
        with self.assertRaisesRegex(ValueError, 'physical wall'):
            containment(region, config, shape)


if __name__ == '__main__':
    unittest.main()
