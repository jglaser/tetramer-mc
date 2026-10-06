"""Analytic and synthetic controls only; no protein data or cloud draws."""
import math
import unittest

import context_multicage_noise_budget as m


def deterministic(weight):
    return m.moment_terms(0.,0,1.,0.,-math.log(weight))


class NoiseBudgetTests(unittest.TestCase):
    def test_independent_poisson_sum_first_second_pose_and_auxiliary_moments(self):
        z=.35;lam=2.;lower=.7;volume=1.3;log_q=-.4
        probability=math.exp(-2*lam*volume); first=second=pose=auxiliary=mass=0.
        for count in range(100):
            r=m.moment_terms(lower,count,lam,z,log_q)
            mass+=probability;first+=probability*math.exp(r['log_y'])
            second+=probability*math.exp(r['log_y2']);pose+=probability*math.exp(r['log_pose_second'])
            auxiliary+=probability*(math.exp(r['log_auxiliary_variance']) if r['log_auxiliary_variance'] is not None else 0.)
            probability*=2*lam*volume/(count+1)
        truth=math.exp(z*(lower+volume)-log_q)
        expected_second=truth*truth*math.exp(z*z*volume/(2*lam))
        self.assertAlmostEqual(mass,1.,places=14)
        self.assertAlmostEqual(first,truth,places=13)
        self.assertAlmostEqual(second,expected_second,places=12)
        self.assertAlmostEqual(pose,truth*truth,places=12)
        self.assertAlmostEqual(auxiliary,expected_second-truth*truth,places=12)

    def test_exact_intensity_forecasts_by_independent_poisson_sum(self):
        z=.8;lam=1.7;lower=.3;volume=1.1;q=.7
        probability=math.exp(-2*lam*volume); sums={k:0. for k in m.INTENSITY_MULTIPLIERS}
        for count in range(120):
            r=m.moment_terms(lower,count,lam,z,math.log(q))
            for label in sums:sums[label]+=probability*math.exp(r['log_forecast_second'][label])
            probability*=2*lam*volume/(count+1)
        truth2=math.exp(2*z*(lower+volume))/(q*q)
        for label,multiplier in m.INTENSITY_MULTIPLIERS.items():
            self.assertAlmostEqual(sums[label],truth2*math.exp(z*z*volume/(2*lam*multiplier)),places=11)

    def test_zero_count_and_zero_activity_give_exact_zero_auxiliary_term(self):
        for values in ((2.,0,3.,.5,-1.),(2.,37,3.,0.,-1.),(0.,0,3.,.5,0.)):
            r=m.moment_terms(*values)
            self.assertIsNone(r['log_auxiliary_variance']);self.assertEqual(r['realized_auxiliary_fraction'],0.)
            self.assertEqual(r['log_y2'],r['log_pose_second'])
            self.assertTrue(all(v==r['log_y2'] for v in r['log_forecast_second'].values()))

    def test_small_activity_stable_nonnegative_subtraction(self):
        r=m.moment_terms(0.,3,1.,1e-8,0.)
        expected=3*math.log1p((.5e-8)**2/(1+1e-8))
        self.assertGreater(r['realized_auxiliary_fraction'],0.)
        self.assertAlmostEqual(r['realized_auxiliary_fraction']/(-math.expm1(-expected)),1.,places=14)

    def test_large_logs_stay_finite_without_exp_overflow(self):
        terms=[m.moment_terms(2000.,10000,2.,1.,-1000.)]
        r=m.summarize(terms,4096)
        self.assertTrue(math.isfinite(r['log_second_moment']))
        self.assertEqual(r['observed_importance_ess'],1.)
        self.assertTrue(math.isfinite(r['noise_free_ess_plugin']))
        self.assertIsNone(m.linear(r['log_second_moment']))

    def test_all_logs_use_complete_q_and_certified_lower_volume(self):
        base=m.moment_terms(.4,13,2.,.35,-.7)
        shifted=m.moment_terms(1.4,13,2.,.35,1.3)
        self.assertAlmostEqual(shifted['log_y']-base['log_y'],.35-2.)
        for key in ('log_y2','log_pose_second','log_auxiliary_variance'):
            self.assertAlmostEqual(shifted[key]-base[key],2*(.35-2.))
        self.assertEqual(shifted['realized_auxiliary_fraction'],base['realized_auxiliary_fraction'])

    def test_invalid_inputs_and_unconditional_denominators(self):
        for values in ((-1.,0,1.,1.,0.),(0.,True,1.,1.,0.),(0.,2.,1.,1.,0.),
                       (0.,0,0.,1.,0.),(0.,0,1.,-1.,0.),(0.,0,1.,1.,math.nan)):
            with self.assertRaises(ValueError):m.moment_terms(*values)
        r=m.summarize([deterministic(2.)],4096)
        self.assertAlmostEqual(math.exp(r['log_mass']),2/4096)
        self.assertEqual(r['observed_importance_ess'],1.)
        self.assertEqual(m.summarize([],4096)['observed_importance_ess'],0.)
        with self.assertRaises(ValueError):m.summarize([deterministic(1.)]*2,1)

    def test_region_and_radial_second_moment_partitions(self):
        terms=[m.moment_terms(.2,k,2.,.4,-.1*k) for k in range(8)]
        whole=m.summarize(terms,4096)
        m.require_partition(whole,[m.summarize(terms[:3],4096),m.summarize(terms[3:],4096)])
        bins=[m.summarize([terms[i] for i in range(8) if i%6==j],4096) for j in range(6)]
        m.require_partition(whole,bins)
        self.assertEqual([m.radial_index(v) for v in (0.,6.,12.,24.,48.,96.,None)],[0,1,2,3,4,5,5])
        with self.assertRaises(ValueError):m.require_partition(whole,[m.summarize(terms[:3],4096)])

    def test_fixed_stratum_centering_removes_between_component_mean_differences(self):
        a=m.stratum_variance([deterministic(1.)]*3,3,6)
        b=m.stratum_variance([deterministic(9.)]*3,3,6)
        self.assertEqual(a['total']['value'],0.)
        self.assertEqual(b['total']['value'],0.)
        pooled=m.stratum_variance([deterministic(1.)]*3+[deterministic(9.)]*3,6,6)
        self.assertGreater(pooled['total']['value'],0.)

    def test_stratum_variance_retains_hard_zeros(self):
        r=m.stratum_variance([deterministic(2.)],4,8)
        # [2,0,0,0] has ordinary sample variance 1; n*s^2/N^2=4/64.
        self.assertAlmostEqual(r['total']['value'],1/16)
        self.assertAlmostEqual(r['pose']['value'],1/16)
        self.assertEqual(r['auxiliary']['value'],0.)

    def test_negative_pose_variance_and_super_N_proxy_are_not_clipped(self):
        rows=[m.moment_terms(0.,5,1.,1.,0.)]*2
        variance=m.stratum_variance(rows,2,2);summary=m.summarize(rows,2)
        self.assertEqual(variance['total']['value'],0.)
        self.assertLess(variance['pose']['value'],0.)
        self.assertAlmostEqual(variance['pose']['value']+variance['auxiliary']['value'],0.)
        self.assertGreater(summary['noise_free_ess_plugin'],2.)
        self.assertTrue(summary['noise_free_ess_plugin_exceeds_attempts'])

    def test_exact_two_point_stratum_expected_variance(self):
        total=pose=auxiliary=0.
        # Two independent draws from {1,3}; mean estimator variance is Var(Y)/2=.5.
        for x in (1.,3.):
            for y in (1.,3.):
                r=m.stratum_variance([deterministic(x),deterministic(y)],2,2)
                total+=r['total']['value']/4;pose+=r['pose']['value']/4;auxiliary+=r['auxiliary']['value']/4
        self.assertAlmostEqual(total,.5);self.assertAlmostEqual(pose,.5);self.assertEqual(auxiliary,0.)

    def test_fieller_bounded_unbounded_and_missing_denominator(self):
        bounded=m.fieller_ratio([1.,1.,1.,1.],[2.,2.,2.,2.],(0.,1.))
        self.assertEqual(bounded['confidence_set'],[[.5,.5]])
        self.assertFalse(bounded['upper_unbounded'])
        unbounded=m.fieller_ratio([2.,2.,2.,2.],[.001,.001,.001,1.],(1.,None))
        self.assertTrue(unbounded['upper_unbounded'])
        self.assertFalse(unbounded['denominator_separated_from_zero'])
        missing=m.fieller_ratio([0.]*4,[0.]*4,(0.,1.))
        self.assertFalse(missing['available']);self.assertIsNone(missing['point_estimate'])

    def test_current_intensity_self_ratio_is_exact_even_for_dominant_population(self):
        populations=[m.summarize([m.moment_terms(0.,count,2.24,.035,q)],4096)
                     for count,q in ((14000,20.),(14500,21.),(14545,17.),(14300,20.))]
        value=m.ratio_uncertainty(populations)['intensity_forecast_gain']['1']
        self.assertEqual(value['point_estimate'],1.)
        self.assertEqual(value['confidence_set'],[[1.,1.]])
        self.assertFalse(value['upper_unbounded']);self.assertTrue(value['guarantee'])
        empty=m.ratio_uncertainty([m.summarize([],4096)]*4)['intensity_forecast_gain']['1']
        self.assertFalse(empty['available']);self.assertIsNone(empty['point_estimate'])

    def test_source_contact_pattern_is_not_assumed_native_and_extra_neighbors_preserved(self):
        row=dict(physical_valid=True,region='other_contact',source_T_complete_all_regions=True)
        flags=m.memberships(row)
        self.assertFalse(flags['A_T']);self.assertTrue(flags['T_any']);self.assertTrue(flags['T_outside_A'])
        self.assertTrue(flags['remaining_without_A_T'])
        with self.assertRaises(ValueError):m.memberships(dict(row,physical_valid=False))


if __name__ == '__main__':
    unittest.main()
