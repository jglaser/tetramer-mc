"""Synthetic exhaustive-support and population-statistics controls."""
import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from vessel_contact_partition import (PRIMARY,SUPPORTS,RegionSupport,Mass,partition,independent_mass,contrast)
from test_physical_latent_guide import fixture,independent_pose


def mass(weights,n=None):
    m=Mass()
    for w in weights:
        if w:m.add(math.log(w))
    return m.result(len(weights) if n is None else n)


class PartitionTests(unittest.TestCase):
    def test_union_counts_overlapping_witnesses_once_and_keeps_complement(self):
        for valid in (False,True):
            for mask in range(16):
                support={s:bool(mask&(1<<i)) for i,s in enumerate(SUPPORTS)}
                for chosen in PRIMARY:
                    result=partition(valid,{k:valid and k==chosen for k in PRIMARY},support)
                    self.assertEqual(result['inside_measured_pockets'],valid and mask!=0)
                    self.assertEqual(result['outside_measured_pockets'],valid and mask==0)
                    self.assertEqual(sum(result[k] for k in PRIMARY),int(valid))
                    self.assertEqual(result[chosen+':inside_measured_pockets']+result[chosen+':outside_measured_pockets'],int(valid))

    def test_invalid_primary_and_nonboolean_rejected(self):
        with self.assertRaises(ValueError):partition(True,{k:False for k in PRIMARY},{k:False for k in SUPPORTS})
        with self.assertRaises(ValueError):partition(False,{k:False for k in PRIMARY},{k:0 for k in SUPPORTS})

    def test_support_respects_source_capture_inner_radius_and_q_boundaries(self):
        region,_,lower=fixture(.5,coupled=True); pose=independent_pose([.5,0,0,0,0,0],region,lower)[0]
        region.update(capture_center=pose['position'],capture_radius=.1,physical_metric={},minimum_original_q=1.,minimum_original_q_inclusive=False)
        support=RegionSupport(region)
        with patch('vessel_contact_partition.registration',return_value=np.array([1.])):self.assertFalse(support.contains(pose))
        with patch('vessel_contact_partition.registration',return_value=np.array([1.1])):self.assertTrue(support.contains(pose))
        region['maximum_original_q']=1.1;region['maximum_original_q_inclusive']=False
        with patch('vessel_contact_partition.registration',return_value=np.array([1.1])):self.assertFalse(RegionSupport(region).contains(pose))
        region['maximum_original_q_inclusive']=True
        with patch('vessel_contact_partition.registration',return_value=np.array([1.1])):self.assertTrue(RegionSupport(region).contains(pose))
        region['capture_center']=[p+1 for p in pose['position']]
        with patch('vessel_contact_partition.registration',side_effect=AssertionError('must prune')):self.assertFalse(RegionSupport(region).contains(pose))
        region['capture_center']=pose['position'];region['minimum_mahalanobis_radius']=1.
        with patch('vessel_contact_partition.registration',side_effect=AssertionError('must prune')):self.assertFalse(RegionSupport(region).contains(pose))

    def test_mass_preserves_unconditional_zeros_and_large_logs(self):
        value=mass([0,2,0,4],8)
        self.assertAlmostEqual(value['logQ'],math.log(6/8))
        self.assertAlmostEqual(value['ess'],36/20)
        self.assertAlmostEqual(value['max_fraction'],4/6)
        m=Mass();m.add(1000.);m.add(1001.);value=m.result(4)
        self.assertTrue(math.isfinite(value['logQ']))
        self.assertGreater(value['ess'],1)
        with self.assertRaises(ValueError):m.result(1)
        with self.assertRaises(ValueError):m.add(None)

    def test_independent_linear_mass_includes_zero_populations(self):
        values=[mass([0,0]),mass([0,2]),mass([0,4]),mass([0,6])]
        result=independent_mass(values);x=np.array([0,1,2,3.])
        self.assertAlmostEqual(result['log_Q'],math.log(x.mean()))
        self.assertAlmostEqual(result['population_relative_SE'],x.std(ddof=1)/2/x.mean())
        self.assertAlmostEqual(result['importance_ESS'],12**2/(4+16+36))
        self.assertAlmostEqual(result['largest_contribution'],.5)
        self.assertEqual(result['nonzero_populations'],3)
        with self.assertRaises(ValueError):independent_mass(values[:3])
        values[0]['draws']=3
        with self.assertRaises(ValueError):independent_mass(values)

    def test_paired_covariance_cancels_common_population_fluctuation(self):
        populations=[]
        for value in (0,1,2,3):
            populations.append({PRIMARY[0]:{'Qz':mass([value,0])},PRIMARY[1]:{'Qz':mass([2*value,0])}})
        result=contrast(populations)
        self.assertAlmostEqual(result['beta_F_native_minus_competing'],math.log(2))
        self.assertAlmostEqual(result['population_SE'],0,places=14)
        zero=copy.deepcopy(populations)
        for p in zero:p[PRIMARY[1]]['Qz']=mass([0,0])
        self.assertIn('unresolved',contrast(zero))
        self.assertIn('unresolved',independent_mass([mass([0,0]) for _ in range(4)]))


if __name__=='__main__':unittest.main()
