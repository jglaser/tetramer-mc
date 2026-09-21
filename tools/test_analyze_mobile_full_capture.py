"""Exact partition, shared-weight uncertainty, and archived-source controls."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from analyze_mobile_full_capture import (PRIMARY,chart_radii,load_native_classifier,
    moments,paired_noise,paired_ratio,partition_masks,region_stats,relative_covariance,
    sha,target_equal)


class FullCaptureTests(unittest.TestCase):
    def test_exhaustive_boundaries_and_invalid_zeros(self):
        rc=np.array([0.,3.,5.,8.,12.,np.nextafter(12.,np.inf),np.inf,0.])
        q=np.array([1.,2.,2.,2.,2.,2.,2.,np.inf]);valid=np.array([True]*7+[False])
        rn=np.array([4.,0.,0.,0.,0.,0.,0.,np.inf]);bound=np.array([True,False]*4)
        m=partition_masks(valid,q,rc,rn,bound)
        np.testing.assert_array_equal(sum(m[k].astype(int)for k in PRIMARY),valid.astype(int))
        for name,index in [('native_r4',0),('competitor_r3',1),('competitor_shell_3_5',2),
                           ('competitor_shell_5_8',3),('competitor_shell_8_12',4)]:
            self.assertEqual(np.flatnonzero(m[name]).tolist(),[index])
        self.assertEqual(np.flatnonzero(m['remaining_q_gt_1']).tolist(),[5,6])
        stats=region_stats(np.zeros(8),np.zeros(8),np.zeros((8,2)),m['native_r4'])
        self.assertEqual(stats['physical']['draws'],8)
        self.assertAlmostEqual(stats['physical']['logQ'],-math.log(8))
        rn[0]=np.nextafter(4.,np.inf)
        self.assertTrue(partition_masks(valid,q,rc,rn,bound)['native_outside_r4'][0])

    def test_nonfinite_valid_q_fails(self):
        with self.assertRaisesRegex(ValueError,'Invalid region coordinate'):
            partition_masks([True],[math.inf],[0.],[0.],[True])

    def test_chart_inverse_recovers_correlated_latents(self):
        rng=np.random.default_rng(137)
        lower=np.tril(rng.normal(0,.05,(6,6)))+np.diag([.4,.5,.6,.7,.8,.9])
        mean=np.arange(6)*.015;ell=3.
        f=Rotation.from_rotvec([.2,-.3,.4]).as_matrix()
        a=Rotation.from_rotvec([-.1,.5,.2]).as_matrix();ft=np.array([4.,-2.,1.]);at=np.array([1.,2.,3.])
        region=dict(fixed_neighbor=dict(position=ft.tolist(),orientation=Rotation.from_matrix(f).as_quat()[[3,0,1,2]].tolist()),
            gaussian_chart=dict(weights=[1.],angular_length=ell,means=[mean.tolist()],covariances=[(lower@lower.T).tolist()],
                anchors=[dict(position=at.tolist(),rotation=a.tolist())]))
        latents=rng.normal(size=(12,6));poses=[]
        for u in latents:
            x=mean+lower@u;c=x[3:]/ell
            delta=Rotation.from_quat(np.r_[c,1.]/np.linalg.norm(np.r_[c,1.])).as_matrix()
            poses.append(dict(position=(ft+f@(at+x[:3])).tolist(),
                orientation=Rotation.from_matrix(f@delta@a).as_quat()[[3,0,1,2]].tolist()))
        radii,seams=chart_radii(poses,region)
        np.testing.assert_allclose(radii,np.linalg.norm(latents,axis=1),rtol=2e-14,atol=2e-14)
        self.assertFalse(seams.any())

    def test_exact_cayley_seam_has_no_finite_radius(self):
        region=dict(fixed_neighbor=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),
            gaussian_chart=dict(weights=[1.],angular_length=1.,means=[[0.]*6],covariances=[np.eye(6).tolist()],
                anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]))
        radius,seam=chart_radii([dict(position=[0.,0.,0.],orientation=[0.,1.,0.,0.])],region)
        self.assertTrue(seam[0]);self.assertTrue(math.isinf(radius[0]))

    def test_ratio_covariance_distinguishes_shared_and_disjoint_support(self):
        a=np.array([0.,math.log(3.),-math.inf,-math.inf])
        ratio=paired_ratio(a+math.log(5.),a)
        self.assertAlmostEqual(ratio['log_ratio'],math.log(5.));self.assertAlmostEqual(ratio['log_ratio_SE'],0.)
        b=np.array([-math.inf,-math.inf,0.,math.log(3.)])
        ratio=paired_ratio(a,b)
        self.assertLess(ratio['relative_covariance'],0.)
        covariance=relative_covariance(dict(a=a,b=b))['covariance_relative']
        self.assertAlmostEqual(ratio['log_ratio_SE']**2,covariance[0][0]+covariance[1][1]-2*covariance[0][1])
        self.assertGreater(ratio['log_ratio_SE']**2,covariance[0][0]+covariance[1][1])

    def test_two_cloud_variance_and_allzero_mass(self):
        pair=np.array([[2.,4.],[0.,0.],[4.,8.],[0.,0.]])
        with np.errstate(divide='ignore'):
            logs=np.log(pair.mean(axis=1));clouds=np.log(pair)
        result=paired_noise(logs,clouds)
        expected_cloud=np.mean((pair[:,0]-pair[:,1])**2)/4
        self.assertAlmostEqual(result['cloud_fraction'],expected_cloud/np.var(pair.mean(axis=1),ddof=1))
        self.assertAlmostEqual(result['relative_variance_mean_cloud'],expected_cloud/4/np.mean(pair)**2)
        allzero=moments([-math.inf]*4)
        self.assertEqual(allzero['nonzero'],0);self.assertIsNone(allzero['logQ']);self.assertIsNone(allzero['relative_SE'])
        self.assertIsNone(paired_ratio([-math.inf]*4,[0.]*4)['log_ratio_SE'])

    def test_target_identity_rejects_bath_scaffold_or_shape_changes(self):
        config=dict(fixed_poses=[1,2],capture_center=[0,0,0],capture_radius=170.,reservoir_density=.035,
                    depletant_radius=1.5,metadata={'metric':'fixed'})
        region=dict(physical_fixed_neighbors=[1,2],capture_center=[0,0,0],capture_radius=170.,activity=.035,
                    depletant_radius=1.5,physical_metric={'metric':'fixed'},shape_sha256='shape')
        target_equal(config,region,'shape')
        for key,value in [('activity',.03),('physical_fixed_neighbors',[2,1]),('shape_sha256','other')]:
            changed=copy.deepcopy(region);changed[key]=value
            with self.assertRaisesRegex(ValueError,'differs'):target_equal(config,changed,'shape')

    def test_classifier_loads_only_hash_bound_archived_runtime(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);runtime=root/'inputs/source/native_contact_regions.py';runtime.parent.mkdir(parents=True)
            runtime.write_text('class NativeContactRegions:\n    def __init__(self,path):self.path=path\n')
            definition=root/'definition.json';definition.write_text(json.dumps(dict(input_sha256={'source/native_contact_regions.py':sha(runtime)})))
            classifier,binding=load_native_classifier(definition)
            self.assertEqual(classifier.path,definition);self.assertEqual(binding['runtime_sha256'],sha(runtime))
            runtime.write_text(runtime.read_text()+'# changed\n')
            with self.assertRaisesRegex(ValueError,'runtime hash'):load_native_classifier(definition)


if __name__=='__main__':unittest.main()
