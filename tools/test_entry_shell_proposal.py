import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from entry_shell_proposal import EntryShellGuide
from analyze_latent_region import audit_importance_rows, validate_population_guide, DENSITY_MEASURE
from prepare_smc_normalizer_atlas import Density

DIGEST='a'*64

def fixture(alpha=.3):
    lower=np.eye(6);lower[3,0]=.8;lower[4,1]=-.5;lower[5,0]=.3;lower[5,2]=.4
    covariance=lower@lower.T
    region=dict(mahalanobis_radius=2.,minimum_mahalanobis_radius=0.,fixed_neighbor=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),
        gaussian_chart=dict(weights=[1.],coordinate_convention='anchor-body-relative',angular_length=3.,shape_sha256='toy',
            means=[[.1,-.2,.3,.15,-.1,.2]],covariances=[covariance.tolist()],
            anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]))
    guide=dict(schema='defensive-entry-shell-guide-v1',region_sha256=DIGEST,defensive_uniform_shell_probability=alpha,
        entries=[dict(moving_member=[1.2,.3,0.],target_world_member=[1.,0.,0.],inner_radius=.5,outer_radius=1.4,weight=1.)])
    return guide,region


class EntryShellTests(unittest.TestCase):
    def test_correlated_angular_marginal_normalization_and_moments(self):
        raw,region=fixture();g=EntryShellGuide(raw,region,DIGEST);rng=np.random.default_rng(813)
        n=30000;v=rng.normal(size=(n,3));v/=np.linalg.norm(v,axis=1)[:,None]
        v*=g.radius*rng.random(n)[:,None]**(1/3)
        angular=v@g.angular_lower.T+g.mean[3:]
        volume=(4*np.pi/3)*g.radius**3*np.exp(g.angular_log_det)
        self.assertLess(abs(np.exp(g.angular_log_density_raw(angular)).mean()*volume-1),.025)
        u,_=g.draw_for_validation(rng,n);x=g.coordinates(u)[:,3:]
        np.testing.assert_allclose(x.mean(axis=0),g.mean[3:],atol=.025)
        expected=g.radius**2*np.asarray(region['gaussian_chart']['covariances'][0])[3:,3:]/8
        np.testing.assert_allclose(np.cov(x.T),expected,atol=.035)
        self.assertGreater(expected[0,0],g.radius**2*g.lower[3,3]**2/8)

    def test_overlapping_shells_sum_and_representational_splitting_is_neutral(self):
        raw,region=fixture();g=EntryShellGuide(raw,region,DIGEST);u,_=g.draw_for_validation(np.random.default_rng(30),200)
        split=copy.deepcopy(raw);split['entries']*=2;split['entries'][1]=dict(split['entries'][1],weight=3.)
        h=EntryShellGuide(split,region,DIGEST);inside=np.linalg.norm(u,axis=1)<=g.radius
        np.testing.assert_allclose(g.log_density(u,inside,g.log_volume),h.log_density(u,inside,g.log_volume),atol=2e-14)

    def test_uniform_shell_radius_cubed_and_fixed_N_latent_volume(self):
        raw,region=fixture();g=EntryShellGuide(raw,region,DIGEST);u,k=g.draw_for_validation(np.random.default_rng(824),30000)
        t,r,_=g.world(u);sel=k>=0;centers=g.target[0]-np.einsum('nij,j->ni',r[sel],g.moving[0])
        distance=np.linalg.norm(t[sel]-centers,axis=1)
        cdf=(distance**3-g.inner[0]**3)/(g.outer[0]**3-g.inner[0]**3)
        self.assertGreaterEqual(cdf.min(),-1e-12);self.assertLessEqual(cdf.max(),1+1e-12)
        self.assertLess(abs(cdf.mean()-.5),.012)
        inside=np.linalg.norm(u,axis=1)<=g.radius;logq=g.log_density(u,inside,g.log_volume)
        ratio=np.where(inside,np.exp(-logq-g.log_volume),0.)
        self.assertGreater(np.sum(~inside),0)
        self.assertLessEqual(ratio.max(),1/g.alpha+1e-12)
        self.assertLess(abs(ratio.mean()-1),max(.03,6*ratio.std(ddof=1)/np.sqrt(len(ratio))))

    def test_rigid_world_transform_preserves_proposal_density(self):
        raw,region=fixture();g=EntryShellGuide(raw,region,DIGEST);u,_=g.draw_for_validation(np.random.default_rng(3),100)
        rot=Rotation.from_rotvec([.4,-.3,.6]);r=rot.as_matrix();t=np.array([13.,-9.,4.])
        shifted=copy.deepcopy(region);shifted['fixed_neighbor']=dict(position=t.tolist(),orientation=rot.as_quat()[[3,0,1,2]].tolist())
        guide=copy.deepcopy(raw)
        for e in guide['entries']:e['target_world_member']=(r@np.asarray(e['target_world_member'])+t).tolist()
        h=EntryShellGuide(guide,shifted,DIGEST);inside=np.linalg.norm(u,axis=1)<=g.radius
        np.testing.assert_allclose(g.log_density(u,inside,g.log_volume),h.log_density(u,inside,g.log_volume),atol=2e-12)

    def test_uniform_limit_and_invalid_guides_fail_closed(self):
        raw,region=fixture(1.);g=EntryShellGuide(raw,region,DIGEST);u,k=g.draw_for_validation(np.random.default_rng(4),500)
        self.assertTrue(np.all(k==-1));self.assertTrue(np.all(np.linalg.norm(u,axis=1)<=2))
        np.testing.assert_array_equal(g.log_density(u,np.ones(len(u),bool),g.log_volume),np.full(len(u),-g.log_volume))
        changes=[dict(defensive_uniform_shell_probability=0),dict(entries=[]),dict(region_sha256='b'*64),dict(unrecorded=3)]
        for patch in changes:
            with self.subTest(patch=patch),self.assertRaises(ValueError):EntryShellGuide(dict(raw,**patch),region,DIGEST)
        bad=copy.deepcopy(region);bad['minimum_mahalanobis_radius']=.5
        with self.assertRaises(ValueError):EntryShellGuide(raw,bad,DIGEST)

    def test_entry_branch_rows_reconstruct_and_detect_selected_shell_or_density_error(self):
        raw,region=fixture();g=EntryShellGuide(raw,region,DIGEST);u,k=g.draw_for_validation(np.random.default_rng(10),40)
        t,r,x=g.world(u);inside=np.linalg.norm(u,axis=1)<=g.radius;logq=g.log_density(u,inside,g.log_volume)
        logj=g.log_det-3*np.log(g.ell)-2*np.log(np.pi)-2*np.log1p(np.sum((x[:,3:]/g.ell)**2,axis=1))
        quats=Rotation.from_matrix(r).as_quat()[:,[3,0,1,2]];rows=[]
        for i in range(len(u)):
            rows.append(dict(latent=u[i].tolist(),latent_radius=float(np.linalg.norm(u[i])),backmapped_latent=u[i].tolist(),
                backmapped_radius=float(np.linalg.norm(u[i])),pose=dict(position=t[i].tolist(),orientation=quats[i].tolist()),
                log_physical_jacobian=float(logj[i]),log_proposal_density=float(logq[i]),shell_valid=bool(inside[i]),
                hard_valid=False,capture_valid=True,region_valid=True,log_importance_weight=None,log_hard_weight=None,clouds=[],
                proposal_branch='uniform-shell'if k[i]<0 else'entry-shell',proposal_component=None if k[i]<0 else int(k[i])))
        result=audit_importance_rows(rows,region,g,Density(region['gaussian_chart']))
        self.assertEqual(sum(result['branch_counts'].values()),40)
        self.assertGreater(result['branch_counts']['entry-shell'],0)
        changed=copy.deepcopy(rows);changed[0]['log_proposal_density']+=.1
        with self.assertRaises(AssertionError):audit_importance_rows(changed,region,g,Density(region['gaussian_chart']))


if __name__=='__main__':unittest.main()
