"""Reciprocal chart construction controls; no physical or bath samples."""
import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_competing_atlas import (
    chart_coordinates, chart_pose, competitor_model, invert_pose, pose,
    reciprocal_jacobian, reciprocal_model, validate_reciprocal,
)
from prepare_mobile_posterior_pilot import combine_models, verify_density_identity


class ReciprocalContactChartTests(unittest.TestCase):
    def fixture(self):
        anchor=pose([4.,-3.,2.],Rotation.from_rotvec([.2,-.4,.8]).as_matrix())
        moving=pose([-8.,2.,9.],Rotation.from_rotvec([-.8,.6,.4]).as_matrix())
        return competitor_model(moving,anchor,dict(schema='weighted-pose-mixture-v1',
            shape_sha256='synthetic',angular_length=5.))

    def test_generic_proper_pose_reciprocal_derivative_and_centers(self):
        forward=self.fixture()
        inverse,j,error=reciprocal_model(forward)
        result=validate_reciprocal(forward,inverse,j)
        self.assertTrue(result['passed'])
        self.assertLess(error,1e-14)
        self.assertLess(result['reciprocal_derivative_product_error'],1e-12)

    def test_reciprocal_covariance_preserves_the_linearized_measure(self):
        forward=self.fixture()
        inverse,j,_=reciprocal_model(forward)
        sf=np.asarray(forward['covariances'][0]);si=np.asarray(inverse['covariances'][0])
        np.testing.assert_allclose(si,j@sf@j.T,atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(si).min(),0)
        self.assertAlmostEqual(np.linalg.det(si)/np.linalg.det(sf),1.,places=11)
        self.assertGreater(np.max(np.abs(si[:3,3:])),1e-4)

    def test_translation_is_nonlinear_while_reciprocal_rotation_is_exact(self):
        forward=self.fixture();inverse,j,_=reciprocal_model(forward)
        x=np.array([.2,-.1,.3,.15,-.2,.1])
        actual=chart_coordinates(inverse,invert_pose(chart_pose(forward,x)))
        self.assertGreater(np.linalg.norm((actual-j@x)[:3]),1e-4)
        np.testing.assert_allclose(actual[3:],(j@x)[3:],atol=1e-13)

    def test_density_mixture_is_normalized_by_components_not_inverse_pair_assumption(self):
        forward=self.fixture();inverse,_,_=reciprocal_model(forward)
        legacy=copy.deepcopy(forward)
        legacy['covariances']=(np.asarray(legacy['covariances'])*100).tolist()
        combined,groups=combine_models([legacy,forward,inverse],[.8,.1,.1],'synthetic')
        check=verify_density_identity(combined,[legacy,forward,inverse],[.8,.1,.1],seed=47)
        self.assertTrue(check['passed'])
        self.assertEqual([g['proposal_mass'] for g in groups],[.8,.1,.1])

    def test_invalid_angular_length_is_rejected(self):
        for ell in (0.,-1.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):
                reciprocal_jacobian(np.zeros(3),np.eye(3),ell)


if __name__=='__main__':
    unittest.main()
