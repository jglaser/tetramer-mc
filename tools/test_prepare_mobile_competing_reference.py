"""Synthetic finite-region geometry controls; no production or bath sampling."""
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_competing_reference import (
    capture_wall_certificate, competitor_model, decode_latent, encode_matrix,
    jacobian_check, native_frame, pose,
)


class MobileCompetingPreparationTests(unittest.TestCase):
    def fixture(self):
        fixed = pose([4., -2., 1.], Rotation.from_rotvec([.3, -.2, .7]).as_matrix())
        moving = pose([-3., 1., 2.], Rotation.from_rotvec([-.8, .4, -.3]).as_matrix())
        model = competitor_model(moving, fixed, dict(schema='weighted-pose-mixture-v1',
            shape_sha256='synthetic', angular_length=5.))
        return fixed, moving, model

    def test_center_and_geometric_scales_are_exact_without_fitting(self):
        fixed, moving, model = self.fixture()
        center, _ = decode_latent(np.zeros((1, 6)), model, fixed)
        np.testing.assert_allclose(center[0]['position'], moving['position'], atol=2e-14)
        actual = Rotation.from_quat(np.array(center[0]['orientation'])[[1, 2, 3, 0]])
        expected = Rotation.from_quat(np.array(moving['orientation'])[[1, 2, 3, 0]])
        self.assertLess((actual*expected.inv()).magnitude(), 2e-14)
        covariance = np.asarray(model['covariances'][0])
        np.testing.assert_array_equal(covariance[:3, :3], np.eye(3)*.25**2)
        self.assertAlmostEqual(covariance[3, 3], (5*math.tan(math.radians(.25)/2))**2)
        self.assertEqual(model['means'], [[0.]*6])

    def test_frame_transform_preserves_original_native_anchor_relation(self):
        old = pose([3., 7., -4.], Rotation.from_rotvec([.1, -.7, .2]).as_matrix())
        new = pose([-2., 1., 9.], Rotation.from_rotvec([-.3, .2, .5]).as_matrix())
        result = native_frame(old, new)
        r = Rotation.from_quat(np.array(result['orientation'])[[1, 2, 3, 0]])
        np.testing.assert_allclose(r.apply(old['position'])+result['position'], new['position'], atol=2e-14)

    def test_roundtrip_and_independent_normalized_haar_jacobian(self):
        fixed, _, model = self.fixture()
        latent = np.array([[.4, -.9, 1.2, -.7, .8, .3], [3., 0., 0., 0., 0., 0.]])
        poses, _ = decode_latent(latent, model, fixed)
        np.testing.assert_allclose(encode_matrix(poses, model, fixed), latent, atol=1e-12)
        self.assertLess(jacobian_check(model, fixed, latent[0]), 2e-7)

    def test_capture_certificate_checks_region_and_wall_separately(self):
        fixed, _, model = self.fixture()
        certificate = capture_wall_certificate(model, fixed, 5., 10., 14., 3.)
        self.assertEqual(certificate['guaranteed_atomic_wall_clearance_A'], 1.)
        with self.assertRaisesRegex(ValueError, 'physical wall'):
            capture_wall_certificate(model, fixed, 5., 10., 12., 3.)
        with self.assertRaisesRegex(ValueError, 'inside capture'):
            capture_wall_certificate(model, fixed, 5., 4., 14., 3.)


if __name__ == '__main__':
    unittest.main()
