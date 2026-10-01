import copy
import math
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal,norm
from analyze_contact_line_audit import Reconstructor,contains,interval_masses,normal_masses,read,validate_intervals


IDENTITY=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])


def setup(axes=(0,),beta=.5):
    L=np.eye(6);L[3,0]=.35;L[4,1]=-.2;L[1,0]=.4
    region=dict(gaussian_chart=dict(means=[[0.]*6],covariances=[(L@L.T).tolist()],angular_length=2.,
        anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]),fixed_neighbor=IDENTITY,
        mahalanobis_radius=4.)
    covariance=np.eye(6);covariance[0,3]=covariance[3,0]=.4
    guide=dict(schema='defensive-contact-line-guide-v1',defensive_uniform_shell_probability=.5,
        conditional_probability=beta,minimum_conditional_mass=1e-12,raw_translation_axes=list(axes),
        contact_widths_A=[.02,.1,.5],contact_neighbor_indices=[0,1],gaussian_components=[
        dict(weight=2.,mean=[0.]*6,covariance=covariance.tolist()),
        dict(weight=1.,mean=[20.,0.,0.,0.,0.,0.],covariance=(np.eye(6)*.2).tolist())])
    config=dict(capture_center=[0.,0.,0.],capture_radius=100.,fixed_poses=[
        dict(position=[-2.05,0.,0.],orientation=[1.,0.,0.,0.]),
        dict(position=[2.05,0.,0.],orientation=[1.,0.,0.,0.])])
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
    return region,guide,config,shape


def detail(recon,u,sets=None):
    x=recon.raw(u);axes=[];fallbacks=0
    for axis in recon.axes:
        raw=x.copy();raw[axis]=0
        u0=np.linalg.solve(recon.L0,raw-recon.m0);d=np.linalg.solve(recon.L0,np.eye(6)[axis])
        _,position,R,_=recon.decode(u0)
        from scipy.spatial.transform import Rotation
        q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        segment=recon.chord(u0,d,recon.radius)
        intervals=sets if sets is not None else [dict(lower=-.1,upper=.1,lower_closed=True,upper_closed=True)]
        widths=[dict(width_A=w,intervals=copy.deepcopy(intervals)) for w in recon.widths]
        axes.append(dict(axis=axis,segment=segment,origin=dict(position=position.tolist(),orientation=q.tolist()),direction=recon.Rf[:,axis].tolist(),widths=widths))
        means,sigmas=recon.conditional(x,axis)
        fallbacks+=len(widths)*int((interval_masses(intervals,means,sigmas)<=recon.floor).sum())
    baseguide=copy.deepcopy(recon.guide);baseguide['conditional_probability']=0
    base=Reconstructor(recon.region,baseguide,recon.config,dict(atoms=[dict(center=[0.,0.,0.],radius=1.)]))
    baseline=base.density(u,dict(conditioning_disabled=True))[0]
    return dict(raw_coordinates=x.tolist(),axes=axes,fallback_component_branches=fallbacks,
        component_branches=len(recon.weights)*len(axes)*len(recon.widths),baseline_log_density=baseline)


class AuditTests(unittest.TestCase):
    def test_independent_normal_integrals_against_decimal_reference(self):
        data=read(Path(__file__).parents[1]/'tests/data/contact_line_normal_reference.json')
        for row in data['cases']:
            got=float(normal_masses(row['lo'],row['hi']))
            self.assertLessEqual(abs(got-row['mass']),3e-12*row['mass']+1e-300,row['name'])
    def test_endpoint_algebra_and_empty_mass(self):
        intervals=[dict(lower=-1.,upper=0.,lower_closed=True,upper_closed=False),dict(lower=0.,upper=1.,lower_closed=False,upper_closed=True)]
        validate_intervals(intervals);self.assertFalse(contains(intervals,0.));self.assertTrue(contains(intervals,1.))
        self.assertEqual(float(interval_masses([],0.,1.)),0.)
        broken=copy.deepcopy(intervals);broken[0]['upper_closed']=True
        with self.assertRaisesRegex(ValueError,'merged'):validate_intervals(broken)
    def test_beta_zero_and_component_specific_fallback(self):
        region,guide,config,shape=setup();r=Reconstructor(region,guide,config,shape);u=np.zeros(6);d=detail(r,u)
        q,base,fallback=r.density(u,d)
        self.assertEqual(fallback,3)
        old=.5*math.exp(-r.logvolume)+.5*sum(w*multivariate_normal.pdf(u,mean=c['mean'],cov=c['covariance']) for w,c in zip(r.weights,guide['gaussian_components']))
        self.assertAlmostEqual(base,math.log(old),places=12)
        self.assertGreaterEqual(q,base+math.log(.5))
        bad=copy.deepcopy(d);bad['fallback_component_branches']=0
        with self.assertRaisesRegex(ValueError,'fallback'):r.density(u,bad)
        empty=detail(r,u,sets=[]);q2,b2,_=r.density(u,empty);self.assertAlmostEqual(q2,b2,places=12)
    def test_axis_mixture_uses_every_density_label(self):
        reg,guide,cfg,shape=setup((0,1,2));r=Reconstructor(reg,guide,cfg,shape);u=np.array([.02,.01,0.,.2,-.1,.1])
        full=r.density(u,detail(r,u))[0];parts=[]
        for a in guide['raw_translation_axes']:
            g=copy.deepcopy(guide);g['raw_translation_axes']=[a];single=Reconstructor(reg,g,cfg,shape)
            parts.append(math.exp(single.density(u,detail(single,u))[0]))
        self.assertAlmostEqual(math.exp(full),sum(parts)/3,places=12)
    def test_conditional_schur_law_and_normalization(self):
        reg,g,cfg,shape=setup();r=Reconstructor(reg,g,cfg,shape);x=np.array([0.,.1,.1,.15,.2,.3])
        means,sigmas=r.conditional(x,0);intervals=[dict(lower=-.3,upper=.2,lower_closed=True,upper_closed=True)]
        for m,s in zip(means,sigmas):
            mass=float(interval_masses(intervals,m,s))
            if mass<=r.floor:integral=quad(lambda z:norm.pdf(z,m,s),m-12*s,m+12*s)[0]
            else:integral=quad(lambda z:norm.pdf(z,m,s)/mass,-.3,.2)[0]
            self.assertAlmostEqual(integral,1.,places=11)
        # Changing raw x alone preserves angular coordinates despite correlated L0.
        x2=x.copy();x2[0]=.4
        first=r.decode(np.linalg.solve(r.L0,x-r.m0));second=r.decode(np.linalg.solve(r.L0,x2-r.m0))
        np.testing.assert_allclose(first[2],second[2],atol=1e-14)
    def test_empty_gaussian_pure_uniform_limit(self):
        reg,g,cfg,shape=setup();g['gaussian_components']=[];g['defensive_uniform_shell_probability']=1.
        r=Reconstructor(reg,g,cfg,shape);q,b,n=r.density(np.zeros(6),dict(conditioning_disabled=True))
        self.assertEqual(q,-r.logvolume);self.assertEqual(q,b);self.assertEqual(n,0)
    def test_independent_hard_and_two_contact_predicates(self):
        reg,g,cfg,shape=setup();r=Reconstructor(reg,g,cfg,shape)
        hard,contact=r.direct_geometry(np.zeros(3),np.eye(3));self.assertTrue(hard)
        self.assertEqual(contact,[[False,False],[True,True],[True,True]])
        hard,_=r.direct_geometry(np.array([.1,0,0]),np.eye(3));self.assertFalse(hard)


if __name__=='__main__':unittest.main()
