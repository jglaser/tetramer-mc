"""Deterministic arithmetic controls only; no source sampling or bath calls."""
import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from analyze_context_correlated_reference import AtlasReference, Audit, observations, paired_test, same_observations, RHO, NOISE_SCALE
from dimer_destination_density import inverse_pose


def model():
    lower=np.diag([.4,.35,.45,.6,.5,.65]);lower[3,0]=.12;lower[5,1]=-.1
    def r(c):
        q=np.r_[c,1.];q=q/np.linalg.norm(q);return Rotation.from_quat(q).as_matrix().tolist()
    return dict(schema='reciprocal-pose-mixture-v1',reciprocal_components=[True,False],base_model=dict(
        weights=[.35,.65],angular_length=1.,anchors=[dict(position=[.8,.1,-.05],rotation=r([.1,-.05,.15])),
        dict(position=[-.3,.8,.15],rotation=r([-.1,.15,.05]))],means=[[0.]*6,[0.]*6],covariances=[(lower@lower.T).tolist()]*2))


def proposal_fixture(i=1,j=2,prior_index=1,atypical=False):
    ref=AtlasReference(model())
    z=np.array([8.,-12.,6.,3.,-4.,2.] if atypical else [.2,-.3,.4,.1,.2,-.1]);noise=np.array([-.1,.5,-.2,.3,-.4,.2])
    a,b=ref.charts[ref.base_indices[i]],ref.charts[ref.base_indices[j]]
    target_z=RHO*z+NOISE_SCALE*noise;inverse_noise=NOISE_SCALE*z-RHO*noise
    old=a.decode(z);old=inverse_pose(old) if ref.flags[i] else old
    candidate=b.decode(target_z);candidate=inverse_pose(candidate) if ref.flags[j] else candidate
    ja,jb=a.log_volume(z),b.log_volume(target_z)
    logj=jb-ja;logn=.5*float(noise@noise-inverse_noise@inverse_noise)
    lp=np.log(ref.priors[prior_index]);lx=ref.component_logs(old);ly=ref.component_logs(candidate)
    gx,gy=float(logsumexp(lp+lx)),float(logsumexp(lp+ly))
    fs=lp[i]+lx[i]-gx;rs=lp[j]+ly[j]-gy;label=rs+lp[i]-fs-lp[j]
    info=dict(anchor_index=0,branch='involution',source_law='posterior',identity=False,
        trace=dict(source=i,target=j,noise=noise.tolist()),source_component_index=ref.base_indices[i],target_component_index=ref.base_indices[j],
        source_inverted=ref.flags[i],target_inverted=ref.flags[j],step=dict(pose=candidate,
        inverse_trace=dict(source=j,target=i,noise=inverse_noise.tolist()),source_latent=z.tolist(),target_latent=target_z.tolist(),
        log_extended_jacobian=logj,log_auxiliary_ratio=logn,log_correction=logj+logn),
        selected_source_log_density=float(lx[i]),selected_target_log_density=float(ly[j]),
        full_old_gaussian_log_density=gx,full_new_gaussian_log_density=gy,
        source_log_probability=fs,inverse_source_log_probability=rs,
        label_log_reverse_forward=label,expanded_log_reverse_forward=logj+logn+label,
        log_reverse_forward=gx-gy)
    return ref,old,candidate,info


class Controls(unittest.TestCase):
    def test_context_defensive_prior_complete_support(self):
        ref=AtlasReference(model())
        for index in (1,2):
            expected=.1*np.array(ref.priors[0]);expected[index-1]+=.9
            np.testing.assert_allclose(ref.priors[index],expected,atol=1e-16,rtol=0)
            self.assertAlmostEqual(sum(ref.priors[index]),1.)
            self.assertTrue(all(p>0 for p in ref.priors[index]))

    def test_density_against_scipy_and_reciprocal_measure(self):
        ref=AtlasReference(model());chart=ref.charts[0];z=np.array([.2,-.4,.6,-.1,.2,-.3])
        pose=chart.decode(z);x=chart.coordinates(z);c=x[3:]
        gaussian=multivariate_normal.logpdf(x,mean=chart.mean,cov=chart.lower@chart.lower.T)
        expected=gaussian+2*math.log(math.pi)+2*math.log1p(float(c@c))
        self.assertAlmostEqual(chart.log_density(pose),expected,places=11)
        self.assertAlmostEqual(ref.component_logs(inverse_pose(pose))[1],expected,places=11)

    def test_full_proposal_inverse_and_factors(self):
        ref,old,new,info=proposal_fixture();audit=Audit()
        value,_=ref.proposal(old,new,info,1,audit)
        self.assertAlmostEqual(value,info['log_reverse_forward'],places=11)
        self.assertGreater(audit.checked,15)

    def test_all_reciprocal_pairs_priors_and_atypical_latents(self):
        for i in range(3):
            for j in range(3):
                for prior in range(3):
                    for atypical in (False,True):
                        ref,old,new,info=proposal_fixture(i,j,prior,atypical)
                        value,_=ref.proposal(old,new,info,prior,Audit())
                        self.assertAlmostEqual(value,info['log_reverse_forward'],places=8)

    def test_independent_redraw_or_wrong_reverse_noise_is_rejected(self):
        ref,old,new,info=proposal_fixture()
        for field in ('target_latent','inverse_noise'):
            bad=copy.deepcopy(info)
            if field=='target_latent':bad['step'][field]=bad['trace']['noise']
            else:bad['step']['inverse_trace']['noise']=bad['step']['source_latent']
            with self.assertRaises(ValueError):ref.proposal(old,new,bad,1,Audit())

    def test_deliberate_shortcut_and_missing_jacobian_fail(self):
        ref,old,new,info=proposal_fixture()
        self.assertGreater(abs(info['log_reverse_forward']),.1)
        bad=copy.deepcopy(info);bad['log_reverse_forward']=0.
        with self.assertRaises(ValueError):ref.proposal(old,new,bad,1,Audit())
        bad=copy.deepcopy(info);bad['step']['log_extended_jacobian']=0.
        with self.assertRaises(ValueError):ref.proposal(old,new,bad,1,Audit())

    def test_binary_observations_and_types(self):
        pair=[dict(position=[-.5,.25,.25],orientation=[1.,0.,0.,0.]),
              dict(position=[.5,0.,0.],orientation=[1.,0.,0.,0.])]
        expected=[True,False,False,True,False,False,True,True,True,True,True,False]
        self.assertEqual(observations(pair),expected);same_observations(expected,expected)
        with self.assertRaises(ValueError):same_observations([int(v) for v in expected],expected)

    def test_exact_paired_zero_and_negative_controls(self):
        self.assertEqual(paired_test([8192,0,0,0],.05/24)['p_value'],1.)
        self.assertTrue(paired_test([4096,4096,100,100],.05/24)['passed'])
        self.assertTrue(paired_test([4096,4096,100,0],.05/12)['reject'])
        with self.assertRaises(ValueError):paired_test([100,0,0,0],.05/24)


if __name__=='__main__':unittest.main()
