"""Independent coordinates for frozen-region extraction from global rows."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_global_fixed_region import region_latent, summarize


class GlobalRegionCoordinates(unittest.TestCase):
    def test_rotated_coupled_chart_recovers_known_latent_and_region_membership(self):
        lower=np.diag([.7,.3,1.1,.4,.2,.5]);lower[3,0]=.2;lower[4,1]=-.1;lower[5,2]=.15
        mean=np.array([.2,-.1,.3,.1,-.2,.05]);ell=2.4
        anchor_t=np.array([.8,-.4,1.]);chart_r=Rotation.from_rotvec([.3,-.2,.1])
        fixed_r=Rotation.from_rotvec([-.4,.7,.2]);fixed_t=np.array([7,-4,3])
        latent=np.array([[0,0,0,0,0,0],[3,0,0,0,0,0],[0,0,0,0,0,4],[1,1,-1,1,-1,1]],dtype=float)
        x=latent@lower.T+mean
        relative_r=Rotation.from_quat(np.column_stack((x[:,3:]/ell,np.ones(len(x)))))*chart_r
        world_r=fixed_r*relative_r
        world_t=fixed_r.apply(x[:,:3]+anchor_t)+fixed_t
        poses=[{'position':p.tolist(),'orientation':q[[3,0,1,2]].tolist()} for p,q in zip(world_t,world_r.as_quat())]
        model={'coordinate_convention':'anchor-body-relative','angular_length':ell,'weights':[1.],
               'anchors':[{'position':anchor_t.tolist(),'rotation':chart_r.as_matrix().tolist()}],
               'means':[mean.tolist()],'covariances':[(lower@lower.T).tolist()]}
        region={'fixed_neighbor':{'position':fixed_t.tolist(),'orientation':fixed_r.as_quat()[[3,0,1,2]].tolist()},'gaussian_chart':model}
        actual,radius,audit=region_latent(poses,region)
        np.testing.assert_allclose(actual,latent,atol=2e-14)
        np.testing.assert_allclose(radius,np.linalg.norm(latent,axis=1),atol=2e-14)
        self.assertLess(audit['matrix_radius_relative_error'],1e-13)

    def test_fixed_denominator_and_zero_population_are_retained(self):
        result=summarize([np.log(8.),-np.inf,-np.inf,-np.inf],[[np.log(8.),np.log(8.)],[-np.inf,-np.inf],[-np.inf,-np.inf],[-np.inf,-np.inf]],[np.log(4.),None])
        self.assertAlmostEqual(result['logQ'],np.log(2.))
        self.assertEqual(result['draws'],4)
        self.assertEqual(result['nonzero'],1)
        self.assertAlmostEqual(result['independent_populations']['logQ'],np.log(2.))
        self.assertEqual(result['independent_populations']['draws'],2)


if __name__=='__main__':unittest.main()
