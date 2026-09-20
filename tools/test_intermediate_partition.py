"""Finite reference laws and unequal-budget disjoint-sum uncertainty."""
import math
import unittest
import numpy as np

from analyze_intermediate_partition import total_independent
from compare_intermediate_local_reference import PartitionMoments
from probe_intermediate_outer_geometry import affine_cover_bound


def summary(values):
    moments=PartitionMoments(4.)
    for value in values:
        if value: moments.add(1.,math.log(value),0.,[math.log(value)]*2)
        else: moments.add()
    return moments.report()['physical']['ball']


class PartitionTests(unittest.TestCase):
    def test_sum_keeps_unequal_denominators_and_independent_variances(self):
        x=np.array([0.,2.,4.,6.]);y=np.array([0.,0.,0.,3.,5.,12.])
        a,b=summary(x),summary(y); result=total_independent([a,b])
        mean=x.mean()+y.mean();variance=x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y)
        self.assertAlmostEqual(math.exp(result['logQ']),mean)
        self.assertAlmostEqual(math.exp(result['log_variance_of_mean']),variance)
        self.assertAlmostEqual(result['row_RSE'],math.sqrt(variance)/mean)
        self.assertAlmostEqual(result['largest_single_draw_fraction'],max(x.max()/len(x),y.max()/len(y))/mean)
        self.assertNotAlmostEqual(mean,np.r_[x,y].mean())

    def test_complete_partition_with_different_reference_laws_is_unbiased(self):
        f=np.array([.5,2.,7.,1.,4.]);partition=[np.array([1,1,0,0,0]),np.array([0,0,1,1,1])]
        proposals=[np.array([.1,.3,.2,.3,.1]),np.array([.2,.1,.3,.1,.3])]
        estimates=[]
        for mask,g in zip(partition,proposals):
            contribution=f*mask/g
            counts=np.rint(10*g).astype(int)
            estimates.append(summary(np.repeat(contribution,counts)))
        result=total_independent(estimates)
        self.assertAlmostEqual(math.exp(result['logQ']),f.sum())
        np.testing.assert_array_equal(sum(partition),np.ones(5))

    def test_unobserved_piece_is_not_a_zero_mass_bound(self):
        result=total_independent([summary([1.,3.]),summary([0.,0.,0.])])
        self.assertFalse(result['all_pieces_observed']);self.assertIsNone(result['logQ'])
        self.assertAlmostEqual(math.exp(result['observed_partial_logQ']),2.)

    def test_constant_limits_have_zero_variance_and_json_finite_fields(self):
        result=total_independent([summary([2.]*4),summary([3.]*6)])
        self.assertAlmostEqual(math.exp(result['logQ']),5.)
        # Log-moment subtraction can retain roundoff at a constant sample.
        self.assertLess(result['row_RSE'],1e-7)
        exact=[dict(logQ=math.log(value),log_variance_of_mean=None,maximum_fraction=1/count,
                    paired_cloud_variance_fraction=None) for value,count in [(2.,4),(3.,6)]]
        result=total_independent(exact)
        self.assertEqual(result['row_RSE'],0.);self.assertIsNone(result['log_variance_of_mean'])

    def test_complete_cover_affine_map_and_bound_for_coupled_charts(self):
        anchor=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]
        a=np.diag([1.,2.,3.,4.,5.,6.]);a[4,1]=.7
        b=np.diag([.5,.8,2.,3.,4.,5.]);b[5,2]=-.4
        mu=np.array([.2,-.3,.5,.1,.7,-.2]);cov=lambda l:(l@l.T).tolist()
        cover=dict(anchors=anchor,angular_length=5.,covariances=[cov(a)],means=[[0.]*6])
        target=dict(anchors=anchor,angular_length=5.,covariances=[cov(b)],means=[mu.tolist()])
        bound=affine_cover_bound(cover,target,2.)
        points=np.vstack([2*np.eye(6),-2*np.eye(6),np.ones((1,6))*2/math.sqrt(6)])
        mapped=points@np.array(bound['affine_matrix']).T+bound['affine_shift']
        expected=np.linalg.solve(b,(points@a.T-mu).T).T
        np.testing.assert_allclose(mapped,expected,atol=1e-14)
        self.assertLessEqual(np.linalg.norm(mapped,axis=1).max(),bound['complete_weighted_radius_bound'])
        changed=dict(target,angular_length=6.)
        with self.assertRaises(ValueError):affine_cover_bound(cover,changed,2.)


if __name__=='__main__':unittest.main()
