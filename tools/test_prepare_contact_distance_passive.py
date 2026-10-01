#!/usr/bin/env python3
"""Independent geometry checks for frozen component-mean label selection."""
import unittest
import numpy as np
from prepare_contact_distance_passive import NearestSurface, polygon_area


class ContactLabels(unittest.TestCase):
    def test_variable_radii_not_nearest_centers(self):
        fixed=np.array([[1.,0.,0.],[1.2,0.,0.]])
        result=NearestSurface(fixed,[.1,.8]).query([[0.,0.,0.]],[.2])
        self.assertEqual((result['moving_atom'],result['fixed_atom']),(0,1))
        self.assertAlmostEqual(result['minimum_surface_gap_A'],.2)

    def test_grouped_search_against_all_pairs(self):
        rng=np.random.default_rng(715015)
        for _ in range(16):
            moving=rng.normal(size=(23,3));fixed=rng.normal(size=(31,3))
            mr=rng.choice([.2,.6,1.1],len(moving));fr=rng.choice([.3,.9,1.2],len(fixed))
            gap=np.linalg.norm(moving[:,None,:]-fixed[None,:,:],axis=2)-mr[:,None]-fr[None,:]
            expected=np.unravel_index(np.argmin(gap),gap.shape)
            result=NearestSurface(fixed,fr).query(moving,mr)
            self.assertEqual((result['moving_atom'],result['fixed_atom']),expected)
            self.assertAlmostEqual(result['minimum_surface_gap_A'],gap[expected],places=13)

    def test_exact_and_near_ties_retained_without_changing_argmin(self):
        fixed=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.+2e-11,0.]])
        result=NearestSurface(fixed,[.3]*3).query([[0.,0.,0.]],[.2])
        self.assertEqual(result['fixed_atom'],0)
        self.assertEqual(result['exact_float_minimum_ties'],2)
        self.assertEqual(len(result['near_ties']),3)
        self.assertGreater(result['near_ties'][2]['gap_above_selected_A'],0)

    def test_polygon_area_empty_tangent_full_and_clipped(self):
        self.assertEqual(polygon_area([3.,3.],8.,.5),0)
        self.assertEqual(polygon_area([3.,3.],7.,.5),0)
        self.assertEqual(polygon_area([3.,3.],3.,.5),.25)
        self.assertAlmostEqual(polygon_area([3.,3.],6.5,.5),.125)
        self.assertAlmostEqual(polygon_area([3.,3.],.25,.5),.1875)
        self.assertAlmostEqual(polygon_area([2.,3.],1.,.5),.125)
        self.assertAlmostEqual(polygon_area([3.,2.],1.,.5),.125)


if __name__=='__main__':
    unittest.main()
