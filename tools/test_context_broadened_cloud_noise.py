import math
import unittest
from analyze_context_broadened_cloud_noise import cloud_pair,summarize,memberships,rao_blackwell_log_weight


def row(a,b,ordinal=0):
    return dict(cloud_pair(math.log(a),math.log(b)),stratum_id='test',ordinal=ordinal,
        physical_valid=True,region='A_patch_complete',source_T_complete_all_regions=True,
        mahalanobis_squared={'full':0.,'diagonal':0.,'broad_full':0.})


class Tests(unittest.TestCase):
    def test_equal_clouds_add_no_auxiliary_noise(self):
        value=summarize([row(1,1),row(3,3,1)],4096)
        self.assertAlmostEqual(value['importance_ess'],1.6)
        self.assertAlmostEqual(value['conditional_product_ratio_not_ess'],1.6)
        self.assertEqual(value['cloud_noise_fraction_of_observed_second_moment'],0.)
        self.assertAlmostEqual(math.exp(value['log_mass']),4/4096)

    def test_unequal_cloud_split_is_exact_and_product_ratio_is_not_ess(self):
        value=summarize([row(1,3)],4096)
        self.assertAlmostEqual(value['importance_ess'],1.)
        self.assertAlmostEqual(value['conditional_product_ratio_not_ess'],4/3)
        self.assertAlmostEqual(value['cloud_noise_fraction_of_observed_second_moment'],.25)

    def test_poisson_pair_expectations(self):
        z=.35;lam=2.;volume=1.3;prob=math.exp(-lam*volume);m1=m2=0.
        for k in range(70):
            weight=math.exp(k*math.log1p(z/lam));m1+=prob*weight;m2+=prob*weight*weight;prob*=lam*volume/(k+1)
        expected_m1=math.exp(z*volume);variance=expected_m1**2*math.expm1(z*z*volume/lam)
        self.assertAlmostEqual(m1,expected_m1)
        self.assertAlmostEqual(.5*(m2-m1*m1),variance/2)

    def test_very_unequal_clouds_remain_finite(self):
        value=cloud_pair(1000.,0.)
        self.assertAlmostEqual(value['log_mean'],1000-math.log(2))
        self.assertEqual(value['pair_noise_fraction'],1.)
        self.assertTrue(math.isfinite(value['log_noise']))

    def test_T_any_keeps_extra_neighbor_contacts(self):
        r=row(1,1);r['region']='other_contact';flags=memberships(r)
        self.assertFalse(flags['A_T']);self.assertTrue(flags['T_any']);self.assertTrue(flags['T_outside_A'])
        self.assertTrue(flags['contact_without_A_T'])

    def test_Rao_Blackwell_binomial_conditional_expectation(self):
        z=.35;lam=2.;lower=1.3;q=-.7
        for total in range(20):
            conditional=sum(math.comb(total,k)/2**total *
                (math.exp(z*lower)*(1+z/lam)**k+math.exp(z*lower)*(1+z/lam)**(total-k))/2
                for k in range(total+1))
            self.assertAlmostEqual(math.log(conditional)-q,rao_blackwell_log_weight(lower,total,lam,z,q),places=12)

    def test_Rao_Blackwell_Poisson_expectation_variance_and_zero_limit(self):
        z=.35;lam=2.;volume=1.3;prob=math.exp(-2*lam*volume);m1=m2=0.
        for total in range(90):
            value=math.exp(rao_blackwell_log_weight(0.,total,lam,z,0.))
            m1+=prob*value;m2+=prob*value*value;prob*=2*lam*volume/(total+1)
        mean=math.exp(z*volume);relative_variance=math.expm1(z*z*volume/(2*lam))
        self.assertAlmostEqual(m1,mean);self.assertAlmostEqual(m2-mean*mean,mean*mean*relative_variance)
        self.assertLess(relative_variance,math.expm1(z*z*volume/lam)/2)
        self.assertEqual(rao_blackwell_log_weight(2.,0,lam,z,-1.),1.7)


if __name__=='__main__':unittest.main()
