"""Synthetic geometry bounds only; no protein atom/classifier queries."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import supplemental_native_candidates as a


class CenterCandidatesTests(unittest.TestCase):
    def test_boundary_and_outward_allowance(self):
        model=a.SupplementalCenterCandidates([[0,0,0]],[[7,0,0]],2.)
        result=model.candidate_pairs([[0,0,0],[9,0,0],[np.nextafter(9.,math.inf),0,0],[9.001,0,0]])
        self.assertEqual(result['nominal_bound_A'],9.)
        self.assertIn([0,1],result['candidate_pairs']);self.assertIn([0,2],result['candidate_pairs'])
        self.assertNotIn([0,3],result['candidate_pairs']);self.assertGreater(result['query_radius_A'],9.)
        self.assertEqual(result['total_pairs'],6)
        self.assertEqual(result['candidate_count']+result['pruned_pairs'],6)

    def test_nonzero_member_mean_attains_two_b_term(self):
        members=np.array([[5.,0,0]]*4);target=np.array([7.,0,0]);r=np.diag([-1.,-1.,1.]);d=np.array([19.,0,0])
        errors=np.linalg.norm(members@r.T+d-(members+target),axis=1)
        self.assertTrue(np.all(errors==2.))
        result=a.SupplementalCenterCandidates(members,[target],2.).candidate_pairs([[0,0,0],d])
        self.assertEqual(result['nominal_bound_A'],19.);self.assertEqual(result['candidate_pairs'],[[0,1]])

    def test_large_canceling_members_inflate_guard_not_nominal(self):
        model=a.SupplementalCenterCandidates([[1e15,0,0],[-1e15,0,0]],[[1,0,0]],2.)
        result=model.candidate_pairs([[0,0,0],[3.,0,0]])
        self.assertEqual(result['nominal_bound_A'],3.)
        self.assertGreater(result['roundoff_allowance_A'],100.)
        self.assertEqual(result['candidate_pairs'],[[0,1]])

    def test_large_common_translation_keeps_true_candidates(self):
        positions=np.array([[0.,0,0],[3.,0,0],[10.,0,0],[10.,2.,0]])
        model=a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.)
        original=model.candidate_pairs(positions);translated=model.candidate_pairs(positions+1e12)
        self.assertTrue(set(map(tuple,original['candidate_pairs']))<=set(map(tuple,translated['candidate_pairs'])))
        self.assertGreater(translated['roundoff_allowance_A'],original['roundoff_allowance_A'])

    def test_empty_singleton_and_no_motif_inventory(self):
        model=a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.)
        for positions in ([],[[0,0,0]]):
            result=model.candidate_pairs(positions);self.assertEqual(result['total_pairs'],0);self.assertEqual(result['tree_queries'],0)
        result=a.SupplementalCenterCandidates([[0,0,0]],[],2.).candidate_pairs([[0,0,0],[1,0,0],[2,0,0]])
        self.assertEqual(result['total_pairs'],3);self.assertEqual(result['pruned_pairs'],3)
        self.assertEqual(result['candidate_pairs'],[]);self.assertFalse(result['fallback'])

    def test_unsafe_arithmetic_falls_back_to_all_pairs(self):
        model=a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.)
        result=model.candidate_pairs([[1e300,0,0],[-1e300,0,0],[0,0,0]])
        self.assertTrue(result['fallback']);self.assertEqual(result['candidate_pairs'],[[0,1],[0,2],[1,2]])
        self.assertEqual(result['pruned_pairs'],0);self.assertEqual(result['tree_queries'],0)

    def test_numerical_tree_failure_retains_every_pair(self):
        model=a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.)
        with patch.object(a,'cKDTree',side_effect=ValueError('synthetic arithmetic failure')):
            result=model.candidate_pairs([[0,0,0],[10,0,0]])
        self.assertTrue(result['fallback']);self.assertEqual(result['candidate_pairs'],[[0,1]])
        self.assertEqual(result['tree_queries'],1)

    def test_canonical_pairs_and_direct_distance_oracle(self):
        positions=np.array([[0.,0,0],[4,0,0],[0,3,0],[2,2,2],[-3,-1,0],[100,0,0]])
        result=a.SupplementalCenterCandidates([[0,0,0]],[[2,0,0]],2.).candidate_pairs(positions)
        expected=[[i,j] for i in range(len(positions)) for j in range(i+1,len(positions))
            if np.linalg.norm(positions[i]-positions[j])<=result['query_radius_A']]
        self.assertEqual(result['candidate_pairs'],expected)
        self.assertEqual(result['total_pairs'],15)

    def test_invalid_inputs_are_fatal_and_periodic_is_rejected(self):
        for members,motifs,e in (([],[[1,0,0]],2.),([[0,0,0]],[[math.nan,0,0]],2.),
                                 ([[0,0,0]],[[1,0,0]],-1.),([[0,0,0]],[[1,0,0]],True)):
            with self.assertRaises(ValueError):a.SupplementalCenterCandidates(members,motifs,e)
        with self.assertRaisesRegex(ValueError,'open/spherical'):
            a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.,boundary='periodic')
        model=a.SupplementalCenterCandidates([[0,0,0]],[[1,0,0]],2.)
        for positions in ([[0,0]],[[math.inf,0,0]],[[math.nan,0,0]]):
            with self.assertRaises(ValueError):model.candidate_pairs(positions)


if __name__=='__main__':unittest.main()
