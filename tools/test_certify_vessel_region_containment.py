from fractions import Fraction as F
import unittest
from certify_vessel_region_containment import certify, rotation, sqrt_upper


class ContainmentTests(unittest.TestCase):
    def test_integer_square_root_enclosure_and_quaternion_normalization_are_exact(self):
        for value in (F(0), F(1), F(2), F(1, 3), F(10**80, 17), F(1, 10**99)):
            upper = sqrt_upper(value)
            self.assertGreaterEqual(upper*upper, value)
            if upper: self.assertLess((upper-F(1, 10**40))**2, value)
        self.assertEqual(rotation([1, 0, 0, 1]), [[0,-1,0],[1,0,0],[0,0,1]])
        self.assertEqual(rotation([2, 0, 0, 2]), rotation([1, 0, 0, 1]))

    def test_transformed_chart_with_cross_covariance_and_sphere_union_has_known_bound(self):
        cov = [[F(i == j) for j in range(6)] for i in range(6)]
        cov[0][3] = cov[3][0] = F(1,2)
        region = dict(mahalanobis_radius=2, fixed_neighbor=dict(position=[7,8,9], orientation=[1,0,0,1]),
            gaussian_chart=dict(means=[[1,0,0,0,0,0]], covariances=[cov], anchors=[dict(position=[1,0,0])]))
        shape = dict(atoms=[dict(center=[3,4,0],radius=1),dict(center=[0,0,0],radius=2)])
        value = certify(region, shape, [7,10,9], 10, [7,10,9], 4)
        self.assertTrue(value['wall_containment_proven']); self.assertTrue(value['capture_containment_proven'])
        self.assertEqual(F(value['bounds']['outer_atomic_radius']['exact_upper']), 6+2*sqrt_upper(F(3)))
        smaller = certify(region, shape, [7,10,9], 9, [7,10,9], 3)
        self.assertFalse(smaller['wall_containment_proven']); self.assertFalse(smaller['capture_containment_proven'])
        cov[0][3] = cov[3][0] = 2
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            certify(region, shape, [7,10,9], 10, [7,10,9], 4)


if __name__ == '__main__': unittest.main()
