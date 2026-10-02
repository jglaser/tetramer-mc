from fractions import Fraction as F
import unittest
from bound_vessel_unbound_mass import bound


class BoundTests(unittest.TestCase):
    def test_centered_sphere_and_offset_body_origin(self):
        shape = dict(atoms=[dict(center=[0,0,0],radius=2)])
        value = bound(shape,F(5),F(10))
        self.assertEqual(F(value['effective_radius_upper']),3)
        self.assertEqual(F(value['upper_bound_A3']),F(4,3)*F(22,7)*27)
        shape['atoms'][0]['center'] = [3,4,0]
        value = bound(shape,F(5),F(10))
        self.assertEqual(F(value['translation_radius_upper']),8)
        self.assertEqual(F(bound(shape,F(5),F(2))['effective_radius_upper']),2)

    def test_nonconvex_sphere_union_and_empty_domain_are_safe(self):
        shape = dict(atoms=[dict(center=[-4,0,0],radius=1),dict(center=[4,0,0],radius=3)])
        value = bound(shape,F(8),F(20))
        self.assertEqual(value['body_centroid'],['0','0','0'])
        self.assertEqual(F(value['translation_radius_upper']),6)
        impossible = bound(dict(atoms=[dict(center=[0,0,0],radius=9)]),F(8),F(20))
        self.assertTrue(impossible['exact_zero_bound']); self.assertIsNone(impossible['display_log_upper_bound'])
        with self.assertRaises(ValueError): bound(dict(atoms=[]),F(8),F(20))


if __name__ == '__main__': unittest.main()
