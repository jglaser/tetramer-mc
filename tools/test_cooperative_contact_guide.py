#!/usr/bin/env python3
"""Focused checks for the cooperative guide's eligibility and exact mixture law."""
import copy
import unittest

import numpy as np
from scipy.stats import multivariate_normal

from prepare_cooperative_contact_guide import cooperative
from prepare_contact_tail_expansion import expand_guide, allocation_control
from prepare_contact_bank_guides import log_proposal


class CooperativeGuideTests(unittest.TestCase):
    def setUp(self):
        self.label = dict(applicable=True, classification=dict(native_any=False),
            contact=dict(anchors=[dict(anchor_index=i, exclusion_contact=True) for i in range(2)]))

    def test_both_distinct_scaffolds_and_no_native_are_required(self):
        self.assertTrue(cooperative(self.label))
        for variant in ('missing', 'native', 'one', 'duplicate'):
            label = copy.deepcopy(self.label)
            if variant == 'missing': label['applicable'] = False
            if variant == 'native': label['classification']['native_any'] = True
            if variant == 'one': label['contact']['anchors'][1]['exclusion_contact'] = False
            if variant == 'duplicate': label['contact']['anchors'][1]['anchor_index'] = 0
            self.assertFalse(cooperative(label))

    def test_full_gaussian_density_and_exact_retention_identity(self):
        base = dict(defensive_uniform_shell_probability=.5, gaussian_components=[
            dict(weight=1., mean=[0.]*6, covariance=np.eye(6).tolist())])
        extra = [dict(weight=.25, mean=[float(i)]*6,
                      covariance=(np.eye(6)*.2).tolist()) for i in range(4)]
        candidate = expand_guide(base, extra, .1)
        points = np.array([[0.]*6, [.1]*6, [8.]*6])  # includes outside R4
        volume = np.pi**3*4**6/6
        uniform = .5*np.where(np.sum(points**2,axis=1)<=16, 1/volume, 0.)
        expected = uniform + .45*multivariate_normal.pdf(points,mean=np.zeros(6),cov=np.eye(6))
        expected += .05*sum(multivariate_normal.pdf(points,mean=x['mean'],cov=x['covariance']) for x in extra)/4
        actual = np.exp(log_proposal(points, candidate))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0)
        self.assertTrue(np.all(actual >= .9*np.exp(log_proposal(points,base))*(1-1e-12)))
        self.assertGreater(actual[-1],0.)

    def test_allocation_control_preserves_old_geometry(self):
        base = dict(defensive_uniform_shell_probability=.5, gaussian_components=[
            dict(weight=.5, mean=[float(i)]*6, covariance=np.eye(6).tolist()) for i in range(2)])
        control, assignments = allocation_control(base, [[0.]*6]*4,.1)
        self.assertEqual(assignments,[0]*4)
        self.assertEqual(len(control['gaussian_components']),2)
        for old,new in zip(base['gaussian_components'],control['gaussian_components']):
            self.assertEqual(old['mean'],new['mean'])
            self.assertEqual(old['covariance'],new['covariance'])
        self.assertAlmostEqual(sum(x['weight'] for x in control['gaussian_components']),1.)


if __name__ == '__main__':
    unittest.main()
