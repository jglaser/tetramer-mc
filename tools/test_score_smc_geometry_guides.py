import copy
import math
import unittest
import numpy as np
from score_smc_geometry_guides import (score_moments, combine_population_scores,
    candidate_densities, geometry_scores, saved_groups)
from prepare_contact_bank_guides import log_proposal


def discrete_fixture(offset=0):
    qs=[.2,.3,.5]; f=[2.,0.,5.]; noise=[(.5,1.5),(1.,1.),(.25,1.75)]
    pairs=[]; source=[]; bank=[]; new=[]
    for x,count in enumerate([2,3,5]):
        for a in noise[x]:
            for b in noise[x]:
                for _ in range(count):
                    pairs.append([-np.inf,-np.inf] if f[x]==0 else
                        [math.log(f[x]*a/qs[x])+offset, math.log(f[x]*b/qs[x])+offset])
                    source.append(math.log(qs[x]))
                    bank.append(math.log([.3,.4,.3][x]));new.append(math.log([.35,.3,.35][x]))
    pairs=np.asarray(pairs); z=np.logaddexp(pairs[:,0],pairs[:,1])-math.log(2)
    return z,pairs,np.asarray(source),dict(bank=np.asarray(bank),new=np.asarray(new))


class MomentTests(unittest.TestCase):
    def test_exact_independent_cloud_identity_with_invalid_zeros(self):
        z,pairs,source,targets=discrete_fixture()
        out=score_moments(z,pairs,source,targets,40)
        self.assertEqual(out['unconditional_attempts'],40)
        self.assertEqual(out['contributing_rows'],28)
        self.assertAlmostEqual(math.exp(out['original_log_Qz']),7.)
        for name,physical,noisy in [('bank',96.66666666666667,121.77083333333334),
                                    ('new',82.85714285714286,104.375)]:
            v=out['guides'][name]
            self.assertAlmostEqual(math.exp(v['paired_physical']['log_second_moment']),physical)
            self.assertAlmostEqual(math.exp(v['two_cloud_noisy']['log_second_moment']),noisy)
            self.assertGreaterEqual(noisy,physical)

    def test_masked_numerator_keeps_unconditional_denominator(self):
        z,p,s,q=discrete_fixture(); valid=np.isfinite(z)
        full=score_moments(z,p,s,q,40)
        masked=score_moments(z[valid],p[valid],s[valid],{k:v[valid] for k,v in q.items()},40)
        self.assertEqual(full,masked)

    def test_zero_region_remains_unobserved(self):
        out=score_moments(np.array([]),np.empty((0,2)),np.array([]),dict(bank=np.array([])),100)
        self.assertIsNone(out['original_log_Qz'])
        for v in out['guides']['bank'].values():
            self.assertIsNone(v['ratio_to_bank']);self.assertIsNone(v['log_second_moment'])

    def test_logspace_stress(self):
        baseline=score_moments(*discrete_fixture(),40)
        shifted=score_moments(*discrete_fixture(1000),40)
        for name in baseline['guides']:
            for kind in baseline['guides'][name]:
                a,b=baseline['guides'][name][kind],shifted['guides'][name][kind]
                self.assertAlmostEqual(b['log_second_moment']-a['log_second_moment'],2000)
                self.assertAlmostEqual(b['ratio_to_bank'],a['ratio_to_bank'])
                self.assertAlmostEqual(b['contribution_ESS'],a['contribution_ESS'])

    def test_defensive_violation_fails(self):
        z,p,s,q=discrete_fixture();q['bad']=q['bank']-math.log(3)
        with self.assertRaisesRegex(ValueError,'defensive'):score_moments(z,p,s,q,40)

    def test_independent_population_linear_pooling(self):
        populations=[]
        for i,offset in enumerate([0.,.2,-.1,.6]):
            score=score_moments(*discrete_fixture(offset),40)
            populations.append(dict(id=str(i),samples=40,source_arm='test',cloud_law='two-128',groups={'all':score}))
        out=combine_population_scores(populations)['groups']['all']
        for name in ['bank','new']:
            for kind in ['two_cloud_noisy','paired_physical']:
                expected=np.mean([math.exp(p['groups']['all']['guides'][name][kind]['log_second_moment']) for p in populations])
                self.assertAlmostEqual(math.exp(out['guides'][name][kind]['log_second_moment']),expected)
        other=copy.deepcopy(populations);other[0]['cloud_law']='two-256'
        with self.assertRaisesRegex(ValueError,'different cloud laws'):combine_population_scores(other)
        other=copy.deepcopy(populations);other[0]['source_arm']='other'
        with self.assertRaisesRegex(ValueError,'different source arms'):combine_population_scores(other)
        other=copy.deepcopy(populations);other[0]['samples']=20
        with self.assertRaisesRegex(ValueError,'Unequal-N'):combine_population_scores(other)


class DensityTests(unittest.TestCase):
    def setUp(self):
        self.bank=dict(region_sha256='test',defensive_uniform_shell_probability=.5,
            gaussian_components=[dict(weight=1.,mean=[0.]*6,covariance=np.eye(6).tolist())])
        self.guide=copy.deepcopy(self.bank);self.guide['gaussian_components'][0]['weight']=.5
        self.guide['gaussian_components'].append(dict(weight=.5,mean=[1.]*6,covariance=(2*np.eye(6)).tolist()))

    def test_factored_mixture_matches_direct_density_inside_and_outside(self):
        u=np.array([[0.]*6,[1.]*6,[10.]*6])
        out=candidate_densities(u,self.bank,dict(new=self.guide))
        np.testing.assert_allclose(out['new'],log_proposal(u,self.guide),rtol=0,atol=1e-12)
        self.assertTrue(np.all(out['new']>=out['bank']-math.log(2)-1e-12))
        self.assertTrue(np.isfinite(out['new']).all())

    def test_missing_or_changed_legacy_is_rejected(self):
        bad=copy.deepcopy(self.guide);bad['gaussian_components'][0]['mean'][0]=.01
        with self.assertRaisesRegex(ValueError,'Legacy component'):candidate_densities(np.zeros((1,6)),self.bank,dict(new=bad))

    def test_geometry_keeps_repeated_slots_without_native_labels(self):
        pop=dict(id='heldout',u=np.array([[0.]*6,[0.]*6,[1.]*6]),old_r5_inside=np.array([True,True,False]))
        result=geometry_scores([pop],self.bank,dict(new=self.guide))['populations'][0]
        self.assertEqual(result['groups']['all']['slots'],3)
        self.assertEqual(result['groups']['inside_old_R5_chart']['slots'],2)
        self.assertNotIn('standard_error',str(result))

    def test_region_masks_include_remaining_space_and_all_bins(self):
        a=dict(z=np.array([0.,0.,0.,-np.inf]),native=np.array([True,False,False,False]),
            old_R5_intersection_native=np.array([False]*4),remaining_R4_native=np.array([True,False,False,False]),
            contact=np.array([True,True,False,False]),bin_radial=np.array([0,1,2,-1]),
            bin_angular=np.array([1,2,0,-1]),bin_orthant=np.array([55,58,0,-1]))
        groups=saved_groups(a)
        self.assertEqual(groups['total'].sum(),3)
        self.assertEqual(groups['unbound'].sum(),1)
        self.assertEqual(groups['orthant:58:contact_no_native_entry'].sum(),1)
        self.assertIn('orthant:63:remaining_R4_native',groups)


if __name__=='__main__':unittest.main()
