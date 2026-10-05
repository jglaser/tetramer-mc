import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_context_covariance_balance import (SourceDensity, balanced_log_q, source_batch,
    importance_summary, across_populations, radial_bin, decomposition, close_logs)
from source_guide_reference import pose

class BalanceTests(unittest.TestCase):
    def test_discrete_normalization_and_balance_identity_with_disjoint_support(self):
        f=np.array([.8,.2,0]);d=np.array([0,.3,.7]);q=(f+d)/2;h=np.array([1.,0.,1.])
        self.assertAlmostEqual(q.sum(),1.)
        self.assertAlmostEqual(.5*np.sum(f*h/q)+.5*np.sum(d*h/q),h.sum())
        # A conditional valid-only mean would lose the correct all-attempt mass.
        s=importance_summary([math.log(4.),math.log(2.)],8)
        self.assertEqual(s['denominator'],8);self.assertAlmostEqual(s['mass'],.75)
        self.assertEqual(importance_summary([],8)['mass'],0.)
        self.assertEqual(importance_summary([],8)['importance_ess'],0.)
    def test_full_mixture_equals_equal_generating_laws(self):
        u=np.log([.1,.2]);g=np.log([.3,.4]);f=np.log([.7,.8]);d=np.log([.9,1.])
        actual=np.exp(balanced_log_q(u,g,f,d))
        expected=.5*(.5*np.exp(u)+.25*np.exp(g)+.25*np.exp(f))+.5*(.5*np.exp(u)+.25*np.exp(g)+.25*np.exp(d))
        np.testing.assert_allclose(actual,expected,rtol=2e-15)
    def test_vectorized_source_matches_scalar_noncommuting_full_chart(self):
        anchor=pose([2.,-3.,1.],Rotation.from_rotvec([.3,-.4,.1]).as_matrix())
        source=pose([1.,2.,3.],Rotation.from_rotvec([-.2,.5,.2]).as_matrix())
        lower=np.diag([.2,.3,.4,.1,.2,.3]);lower[4,1]=.09;lower[5,0]=-.08
        spec=dict(angular_length=2.,covariance=(lower@lower.T).tolist(),explicit_gaussian=dict(schema='source-gaussian-v1',mean=[.1,-.1,.2,.02,-.03,.01],provenance='synthetic'))
        den=SourceDensity(spec,source,anchor)
        latent=np.array([[0]*6,[1,-2,.3,.4,-.1,.2],[-3,1,2,.3,.5,-.9]])
        poses=[den.decode(z) for z in latent];logs,radius,seams=source_batch(den,poses)
        np.testing.assert_allclose(logs,[den.evaluate(p) for p in poses],atol=2e-12,rtol=2e-13)
        np.testing.assert_allclose(radius,np.sum(latent**2,axis=1),atol=2e-12)
        self.assertFalse(seams.any())
    def test_radial_boundaries_and_each_chart_partition(self):
        self.assertEqual([radial_bin(x) for x in [0,5.9,6,12,24,48,math.inf]],[0,0,1,2,3,4,4])
        records=[dict(arm='full',branch='source',log_weight=math.log(2.),mahalanobis_squared={'full':6.,'diagonal':48.}),
                 dict(arm='diagonal',branch='context',log_weight=math.log(3.),mahalanobis_squared={'full':0.,'diagonal':12.})]
        total=importance_summary([r['log_weight'] for r in records],8)
        partitions=decomposition(records,total)
        self.assertEqual(len(partitions),2)
        for p in partitions:self.assertAlmostEqual(p['total_fraction'],1.);self.assertEqual(sum(r['hits'] for r in p['entries']),2)
    def test_population_uncertainty_not_iid_weight_error(self):
        result=across_populations([{'mass':v} for v in [1.,2.,3.,4.]])
        self.assertAlmostEqual(result['standard_error'],np.std([1,2,3,4],ddof=1)/2)
        self.assertAlmostEqual(result['population_relative_standard_error'],result['standard_error']/2.5)
        with self.assertRaises(ValueError):across_populations([{'mass':1.}])
    def test_density_change_rejected(self):
        with self.assertRaises(ValueError):close_logs([0.],[.1])
        with self.assertRaises(ValueError):close_logs([-math.inf],[0.])

if __name__=='__main__':unittest.main()
