"""Nonphysical analytic density/coordinate tests; no wall, clouds, or sampler."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from physical_latent_guide import PhysicalLatentGuide, half_mixture_log_density

SHAPE = 'a'*64
REGION = 'b'*64


def fixture(alpha=.5, coupled=False):
    lower = np.eye(6)
    if coupled:
        lower = np.array([[1.1,0,0,0,0,0],[.2,.8,0,0,0,0],[-.1,.3,1.2,0,0,0],
                          [.4,0,.1,.9,0,0],[0,-.2,.1,.2,1.3,0],[.1,.3,-.2,0,.4,.7]])
    fixed_r = Rotation.from_rotvec([.3,-.2,.5]).as_matrix() if coupled else np.eye(3)
    fixed_q = Rotation.from_matrix(fixed_r).as_quat()[[3,0,1,2]].tolist()
    anchor_r = Rotation.from_rotvec([-.1,.4,.2]).as_matrix() if coupled else np.eye(3)
    region = dict(shape_sha256=SHAPE, mahalanobis_radius=4., minimum_mahalanobis_radius=0.,
        minimum_original_q=0., fixed_neighbor=dict(position=[11.,-7.,3.] if coupled else [0.,0.,0.],orientation=fixed_q),
        capture_center=[0,0,0],capture_radius=.001,
        gaussian_chart=dict(shape_sha256=SHAPE,coordinate_convention='anchor-body-relative',angular_length=2.,
            anchors=[dict(position=[.4,-.7,.2] if coupled else [0.,0.,0.],rotation=anchor_r.tolist())],
            means=[[.1,-.2,.3,.4,-.5,.6] if coupled else [0.]*6],covariances=[(lower@lower.T).tolist()],weights=[1.]))
    guide = dict(schema='defensive-latent-shell-guide-v1', region_sha256=REGION,
        defensive_uniform_shell_probability=alpha,gaussian_components=[] if alpha==1. else
        [dict(weight=1.,mean=[0.]*6,covariance=np.eye(6).tolist())])
    return region, guide, lower


def independent_pose(u, region, lower):
    c = region['gaussian_chart']; x = np.asarray(c['means'][0])+lower@np.asarray(u)
    cayley = x[3:]/c['angular_length']; q = np.r_[cayley,1.]; q /= np.linalg.norm(q)
    fixed=region['fixed_neighbor']; fr=Rotation.from_quat(np.asarray(fixed['orientation'])[[1,2,3,0]]).as_matrix()
    rotation = fr@Rotation.from_quat(q).as_matrix()@np.asarray(c['anchors'][0]['rotation'])
    position = np.asarray(fixed['position'])+fr@(np.asarray(c['anchors'][0]['position'])+x[:3])
    pose=dict(position=position.tolist(),orientation=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]].tolist())
    logj=np.log(np.diag(lower)).sum()-3*math.log(c['angular_length'])-2*math.log(math.pi)-2*math.log1p(cayley@cayley)
    return pose, logj


class PhysicalDensityTests(unittest.TestCase):
    def build(self, alpha=.5, coupled=False):
        region,guide,lower=fixture(alpha,coupled)
        return PhysicalLatentGuide(region,guide,region_sha256=REGION,expected_shape_sha256=SHAPE),region,lower

    def test_simple_analytic_density_and_untruncated_exterior(self):
        obj,region,lower=self.build()
        volume=math.pi**3*4**6/6
        for u in ([0.]*6,[5.,0.,0.,0.,0.,0.],[1.,-.2,.3,.4,-.5,.6]):
            pose,logj=independent_pose(u,region,lower);result=obj.evaluate(pose)
            inside=np.linalg.norm(u)<=4
            gaussian=math.exp(-.5*np.dot(u,u))/(2*math.pi)**3
            q=.5*float(inside)/volume+.5*gaussian
            self.assertAlmostEqual(result.log_latent_density,math.log(q),places=12)
            self.assertAlmostEqual(result.log_physical_jacobian,logj,places=12)
            self.assertAlmostEqual(result.log_physical_density,math.log(q)-logj,places=12)
            self.assertEqual(result.in_reference_ball,inside)
        self.assertTrue(math.isfinite(obj.evaluate(independent_pose([5.,0,0,0,0,0],region,lower)[0]).log_physical_density))

    def test_coupled_chart_world_frame_full_jacobian(self):
        obj,region,lower=self.build(coupled=True)
        for u in ([.2,-.7,1.2,.3,-.6,.8],[4.3,.2,.1,.3,-.2,.1]):
            pose,logj=independent_pose(u,region,lower);result=obj.evaluate(pose)
            np.testing.assert_allclose(result.latent,u,rtol=0,atol=2e-14)
            self.assertAlmostEqual(result.log_physical_jacobian,logj,places=12)
            self.assertAlmostEqual(result.log_physical_density+logj,result.log_latent_density,places=12)
            # Source capture is deliberately tiny; it cannot filter the density.
            self.assertGreater(np.linalg.norm(pose['position']),region['capture_radius'])
            flipped=copy.deepcopy(pose);flipped['orientation']=[-x for x in pose['orientation']]
            self.assertAlmostEqual(result.log_physical_density,obj.evaluate(flipped).log_physical_density,places=12)
        poses=[independent_pose(u,region,lower)[0]for u in ([0.]*6,[5.,0,0,0,0,0])]
        np.testing.assert_allclose(obj.log_densities(poses),[obj.evaluate(p).log_physical_density for p in poses],rtol=0,atol=0)

    def test_near_symmetric_source_uses_same_chart_for_inverse_and_jacobian(self):
        region,guide,_=fixture(coupled=True)
        covariance=np.asarray(region['gaussian_chart']['covariances'][0])
        covariance[3,0]+=1e-13
        region['gaussian_chart']['covariances'][0]=covariance.tolist()
        lower=np.linalg.cholesky(.5*(covariance+covariance.T))
        obj=PhysicalLatentGuide(region,guide,region_sha256=REGION,expected_shape_sha256=SHAPE)
        u=[.3,-.2,.1,.4,-.1,.6]
        pose,logj=independent_pose(u,region,lower)
        result=obj.evaluate(pose)
        np.testing.assert_allclose(result.latent,u,rtol=0,atol=2e-14)
        self.assertAlmostEqual(result.log_physical_jacobian,logj,places=13)

    def test_pure_uniform_exterior_and_exact_seam_zero(self):
        obj,region,lower=self.build(alpha=1.)
        outside=obj.evaluate(independent_pose([5.,0,0,0,0,0],region,lower)[0])
        self.assertFalse(outside.in_reference_ball)
        self.assertEqual(outside.log_physical_density,-math.inf)
        for alpha in (1.,.5):
            evaluator,_,_=self.build(alpha=alpha)
            seam=evaluator.evaluate(dict(position=[0.,0.,0.],orientation=[0.,1.,0.,0.]))
            self.assertIsNone(seam.latent);self.assertIsNone(seam.log_latent_density)
            self.assertIsNone(seam.log_physical_jacobian);self.assertFalse(seam.in_reference_ball)
            self.assertEqual(seam.log_physical_density,-math.inf)

    def test_nonzero_near_seam_is_not_epsilon_filtered(self):
        obj,_,_=self.build()
        for w in (1e-6, 1e-13, 1e-50):
            result=obj.evaluate(dict(position=[0.,0.,0.],orientation=[w,math.sqrt(1-w*w),0.,0.]))
            self.assertIsNotNone(result.latent)
            self.assertTrue(math.isfinite(result.log_physical_density))
            self.assertAlmostEqual(result.log_physical_jacobian,
                -3*math.log(2.)-2*math.log(math.pi)+4*math.log(w), places=11)
            self.assertFalse(result.in_reference_ball)
        beyond=dict(position=[0.,0.,0.],orientation=[1e-200,1.,0.,0.])
        with self.assertRaisesRegex(ValueError, 'Unrepresentable positive Gaussian'):
            obj.evaluate(beyond)
        uniform,_,_=self.build(alpha=1.)
        result=uniform.evaluate(beyond)
        self.assertIsNotNone(result.latent)
        self.assertTrue(math.isfinite(result.log_physical_jacobian))
        self.assertEqual(result.log_physical_density,-math.inf)

    def test_outer_mixture_uses_both_complete_component_densities(self):
        p,q=math.log(.02),math.log(7.)
        self.assertAlmostEqual(half_mixture_log_density(p,q),math.log(.5*.02+.5*7.),places=14)
        self.assertEqual(half_mixture_log_density(-math.inf,-math.inf),-math.inf)
        self.assertAlmostEqual(half_mixture_log_density(p,-math.inf),p-math.log(2.))
        np.testing.assert_allclose(half_mixture_log_density([p,-math.inf],[q,q]),[math.log(3.51),q-math.log(2.)])
        for invalid in (math.nan,math.inf):
            with self.assertRaises(ValueError):half_mixture_log_density(invalid,q)

    def test_file_binding_and_bad_inputs_fail(self):
        region,guide,_=fixture()
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);raw=json.dumps(region).encode();(root/'region.json').write_bytes(raw)
            guide['region_sha256']=hashlib.sha256(raw).hexdigest();(root/'guide.json').write_text(json.dumps(guide))
            obj=PhysicalLatentGuide.from_files(root/'region.json',root/'guide.json',expected_shape_sha256=SHAPE)
            self.assertEqual(len(obj.sources),2)
            with self.assertRaises(ValueError):PhysicalLatentGuide.from_files(root/'region.json',root/'guide.json',expected_shape_sha256='c'*64)
        region,guide,_=fixture()
        region['minimum_original_q']=1.
        with self.assertRaisesRegex(ValueError, 'without a q filter'):
            PhysicalLatentGuide(region,guide,region_sha256=REGION,expected_shape_sha256=SHAPE)
        obj,_,_=self.build()
        with self.assertRaises(ValueError):obj.evaluate(dict(position=[0,0,0],orientation=[2,0,0,0]))
        self.assertEqual(obj.evaluate_many([]),[])
        self.assertEqual(obj.log_densities([]).shape,(0,))


if __name__=='__main__':unittest.main()
