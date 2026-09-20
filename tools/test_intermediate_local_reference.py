"""Analytic partition, Jacobian and zero-denominator controls."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from audit_shoulder_mis_independently import chart_values
from compare_intermediate_local_reference import (KEYS,PartitionMoments,finish,partition,
    reference_weights,validate_region,WINDOW)


class LocalReferenceControls(unittest.TestCase):
    def test_closed_ball_and_shell_boundaries_are_an_exact_partition(self):
        for radius,shell in [(0.,0),(1.,0),(2.,1),(3.,2),(4.,3)]:
            self.assertEqual(partition(radius,4.),('full','ball',f'shell{shell}',
                                                 'inner_half' if radius<=2 else 'outer_half'))
        for i,boundary in enumerate((1.,2.,3.)):
            self.assertEqual(partition(math.nextafter(boundary,math.inf),4.)[2],f'shell{i+1}')
        self.assertEqual(partition(math.nextafter(4.,math.inf),4.),('full','complement'))
        self.assertEqual(partition(math.inf,4.),('full','complement'))
        self.assertEqual(partition(.25,1.),('full','ball','shell0','inner_half'))
        with self.assertRaises(ValueError):partition(math.nan,4.)

    def test_original_full_n_partition_moments_match_direct_linear_weights(self):
        radii=[.5,1.,2.,3.,4.,5.,math.inf,None]
        pair=np.array([[1.,3.],[2.,6.],[4.,4.],[3.,9.],[2.,2.],[5.,7.],[1.,5.],[0.,0.]])
        hard=np.array([.5,1.,2.,1.5,1.,3.,.2,0.]);n=len(radii);values=PartitionMoments(4.)
        for radius,clouds,h in zip(radii,pair,hard):
            if radius is None:values.add()
            else:values.add(radius,math.log(clouds.mean()),math.log(h),np.log(clouds).tolist())
        result=values.report()
        for key in KEYS:
            mask=np.array([r is not None and key in partition(r,4.) for r in radii])
            w=np.where(mask,pair.mean(axis=1),0.);h=np.where(mask,hard,0.)
            physical=result['physical'][key]
            self.assertEqual(physical['draws'],n)
            self.assertAlmostEqual(math.exp(physical['logQ']),w.mean())
            self.assertAlmostEqual(math.exp(result['hard'][key]['logQ']),h.mean())
            self.assertAlmostEqual(physical['row_RSE'],w.std(ddof=1)/math.sqrt(n)/w.mean())
            noise=np.sum(np.where(mask,(pair[:,0]-pair[:,1])**2,0.))/4/n**2
            variance=np.var(w,ddof=1)/n
            if noise:self.assertAlmostEqual(physical['paired_cloud_variance_fraction'],noise/variance)
        for kind in ('physical','hard'):
            mass=lambda key:math.exp(result[kind][key]['logQ'])
            self.assertAlmostEqual(mass('full'),mass('ball')+mass('complement'))
            self.assertAlmostEqual(mass('ball'),sum(mass(f'shell{i}') for i in range(4)))
            self.assertAlmostEqual(mass('ball'),mass('inner_half')+mass('outer_half'))
        # Shared-row shell sums carry covariance. The half estimator above is
        # checked against direct row variance, not independent-shell variances.
        inner=np.where([r is not None and r<=2 for r in radii],pair.mean(axis=1),0.)
        outer=np.where([r is not None and 2<r<=4 for r in radii],pair.mean(axis=1),0.)
        self.assertLess(float(np.cov(inner,outer,ddof=1)[0,1]),0.)

    def test_reference_report_does_not_turn_unsupported_complement_into_zero(self):
        aggregate=PartitionMoments(4.);pops=[];supported=('ball','shell0','shell1','shell2','shell3')
        for seed,value in enumerate((2.,4.)):
            local=PartitionMoments(4.);local.add(.5,math.log(value),0.,[math.log(value)]*2)
            for _ in range(7):local.add()
            pops.append(dict(seed=seed,samples=8,**local.report(supported)));aggregate.merge(local)
        result=finish(aggregate,pops,supported)
        self.assertNotIn('complement',result['physical']);self.assertNotIn('full',result['physical'])
        self.assertEqual(result['physical']['ball']['draws'],16)
        self.assertAlmostEqual(math.exp(result['physical']['ball']['logQ']),3/8)
        for key in ('shell1','shell2','shell3'):
            self.assertIsNone(result['physical'][key]['logQ'])
            self.assertIsNone(result['physical'][key]['row_RSE'])
        self.assertAlmostEqual(result['physical']['ball']['independent_population_RSE'],1/3)

    def test_coupled_chart_radius_and_physical_jacobian_from_known_forward_map(self):
        lower=np.diag([1.,2.,3.,.3,.4,.5]);lower[2,3]=.2;lower[4,1]=-.1
        mean=np.array([.3,-.2,.1,.4,-.2,.05]);ell=7.
        fixed_rotation=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();fixed_position=np.array([2.,-3.,4.])
        anchor_rotation=Rotation.from_rotvec([-.2,.4,.1]).as_matrix();anchor_position=np.array([.2,.3,-.1])
        model=dict(angular_length=ell,weights=[1.],means=[mean.tolist()],covariances=[(lower@lower.T).tolist()],
            anchors=[dict(position=anchor_position.tolist(),rotation=anchor_rotation.tolist())])
        latent=np.array([[0.,0.,0.,0.,0.,0.],[.2,-.3,.4,1.,-.7,.6],[3.,0.,0.,0.,0.,0.]])
        x=latent@lower.T+mean;c=x[:,3:]/ell
        relative_r=Rotation.from_quat(np.column_stack((c,np.ones(3)))).as_matrix()@anchor_rotation
        t=(x[:,:3]+anchor_position)@fixed_rotation.T+fixed_position
        r=fixed_rotation@relative_r
        fixed=dict(position=fixed_position.tolist(),orientation=Rotation.from_matrix(fixed_rotation).as_quat()[[3,0,1,2]].tolist())
        logs,norms,jac=chart_values(model,t,r,fixed)
        expected_j=np.log(np.diag(lower)).sum()-3*np.log(ell)-2*np.log(np.pi)-2*np.log1p((c*c).sum(axis=1))
        np.testing.assert_allclose(norms[:,0],np.linalg.norm(latent,axis=1),atol=1e-8)
        np.testing.assert_allclose(jac[:,0],expected_j,atol=1e-12)
        np.testing.assert_allclose(logs[:,0],-3*np.log(2*np.pi)-.5*(latent*latent).sum(axis=1)-expected_j,atol=1e-12)

    def test_reference_zero_masks_and_original_poisson_weight_cannot_be_changed(self):
        manifest=dict(activity=.035,lambda_ratio=64.,**{'lambda':2.24})
        logv=3.;logj=-2.;clouds=[dict(lower_volume=2.,overlap_points=k,
            log_weight=.07+k*math.log1p(1/64)) for k in (1,3)]
        row=dict(q=2.,region_valid=True,capture_valid=True,hard_valid=True,clouds=clouds,
            log_hard_weight=1.,log_importance_weight=1.+float(np.logaddexp(*[c['log_weight'] for c in clouds]))-math.log(2))
        weight,hard,pair=reference_weights(row,logj,logv,manifest)
        self.assertEqual(weight,row['log_importance_weight']);self.assertEqual(hard,1.)
        self.assertAlmostEqual(math.exp(weight),sum(math.exp(c) for c in pair)/2)
        bad=copy.deepcopy(row);bad['clouds'][0]['overlap_points']+=1
        with self.assertRaises(AssertionError):reference_weights(bad,logj,logv,manifest)
        for flag in ('capture_valid','hard_valid','region_valid'):
            invalid=copy.deepcopy(row);invalid[flag]=False
            if flag=='region_valid':invalid['q']=5.
            invalid.update(clouds=[],log_hard_weight=None,log_importance_weight=None)
            self.assertEqual(reference_weights(invalid,logj,logv,manifest),(None,None,None))
            invalid['log_importance_weight']=0.
            with self.assertRaises(ValueError):reference_weights(invalid,logj,logv,manifest)

    def test_region_chart_and_full_physical_neighborhood_are_immutable(self):
        fixed=[dict(position=[1.,0.,0.],orientation=[1.,0.,0.,0.]),dict(position=[0.,2.,0.],orientation=[1.,0.,0.,0.])]
        cfg=dict(fixed_poses=fixed,metadata={'original':'metric'},capture_center=[0.,0.,0.],capture_radius=18.,
            reservoir_density=.035,depletant_radius=1.5)
        model=dict(weights=[1.],shape_sha256='shape',covariances=[np.eye(6).tolist()])
        region=dict(minimum_original_q=2.,maximum_original_q=5.,minimum_original_q_inclusive=True,
            maximum_original_q_inclusive=False,mahalanobis_radius=4.,gaussian_chart=model,physical_metric=cfg['metadata'],
            physical_fixed_neighbors=fixed,fixed_neighbor=fixed[0],shape_sha256='shape',capture_center=cfg['capture_center'],
            capture_radius=18.,activity=.035,depletant_radius=1.5)
        validate_region(region,model,cfg,'shape')
        bad=copy.deepcopy(region);bad['physical_fixed_neighbors']=fixed[:1]
        with self.assertRaises(ValueError):validate_region(bad,model,cfg,'shape')
        bad=copy.deepcopy(region);bad['gaussian_chart']['covariances'][0][0][0]=2.
        with self.assertRaises(ValueError):validate_region(bad,model,cfg,'shape')


if __name__=='__main__':unittest.main()
