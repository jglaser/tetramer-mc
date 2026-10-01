"""Independent angular geometry, normalized conditional-law and row audit tests."""
import copy
import math
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.stats import norm,multivariate_normal
from hard_free_pose_line_reference import (Reconstructor,angular_hard_free_intervals,
    sphere_coefficients,quadratic_negative_intervals,contains,interval,interval_masses,
    subtract_closed_segment,compare_axis)
from test_hard_free_line_reference import setup as translation_setup


def pose(p,q=(1.,0.,0.,0.)):return dict(position=list(p),orientation=list(q))
def setup(axes=(3,4,5),beta=1.,anisotropic=False):
    r,g,c,s=translation_setup((0,),beta)
    g['schema']='defensive-hard-free-pose-line-guide-v1';g['raw_pose_axes']=list(axes);g.pop('raw_translation_axes')
    if anisotropic:
        s=dict(atoms=[dict(center=[-.65,0.,0.],radius=.42),dict(center=[.5,.15,.1],radius=.34)])
        c['fixed_poses']=[pose([1.55,.3,.2]),pose([-1.6,-.4,.1])]
    return r,g,c,s


class AngularGeometryTests(unittest.TestCase):
    def test_strict_quadratic_all_topologies_and_tangency(self):
        self.assertEqual(quadratic_negative_intervals([-1,0,1],[-2,2]),[interval(-1,1,False,False)])
        self.assertEqual(quadratic_negative_intervals([0,0,-1],[-2,2]),[interval(-2,0,True,False),interval(0,2,False,True)])
        self.assertEqual(subtract_closed_segment([-2,2],quadratic_negative_intervals([0,0,-1],[-2,2])),[interval(0,0)])
        self.assertEqual(quadratic_negative_intervals([0,0,1],[-2,2]),[])
        self.assertEqual(quadratic_negative_intervals([-1,0,0],[-2,2]),[interval(-2,2)])
        self.assertEqual(quadratic_negative_intervals([0,0,0],[-2,2]),[])
        self.assertEqual(quadratic_negative_intervals([-1,2,0],[-2,2]),[interval(-2,.5,True,False)])
        self.assertEqual(quadratic_negative_intervals([-1,-2,0],[-2,2]),[interval(-.5,2,False,True)])
        self.assertEqual(quadratic_negative_intervals([-1,0,1],[-1,1]),[interval(-1,1,False,False)])

    def test_known_circle_angular_scale_and_constant_axis_atom(self):
        p=sphere_coefficients([1,0,0],[1,0,0],[0,0,0],2,2.,1.)
        got=quadratic_negative_intervals(p,[-4.,4.]);self.assertEqual(len(got),1)
        np.testing.assert_allclose([got[0]['lower'],got[0]['upper']],[-2/math.sqrt(3),2/math.sqrt(3)],rtol=2e-15)
        self.assertFalse(got[0]['lower_closed'] or got[0]['upper_closed'])
        self.assertEqual(quadratic_negative_intervals(sphere_coefficients([1,0,0],[2,0,0],[0,0,0],2,1.,1.),[-4,4]),[])
        self.assertEqual(quadratic_negative_intervals(sphere_coefficients([0,0,1],[0,0,0],[0,0,0],2,1.,1.),[-4,4]),[])
        self.assertEqual(quadratic_negative_intervals(sphere_coefficients([0,0,1],[0,0,0],[0,0,0],2,1.,1.1),[-4,4]),[interval(-4,4)])

    def test_opposing_contacts_retain_single_feasible_tangent(self):
        shape=dict(atoms=[dict(center=[1.,0.,0.],radius=.5)])
        fixed=[pose([0,1.,0]),pose([0,-1.,0])]
        geometry=angular_hard_free_intervals(shape,fixed,np.zeros(3),np.eye(3),np.eye(3),np.zeros(3),2,1.,[-.1,.1])
        self.assertEqual(geometry['intervals'],[interval(0,0)])
        reg,g,cfg,_=setup((5,));reg['mahalanobis_radius']=.1;cfg['fixed_poses']=fixed
        r=Reconstructor(reg,g,cfg,shape);axis=r.reconstruct_axis(np.zeros(6),5)
        self.assertEqual(axis['empty_reason'],'no_positive_hard_free_length')
        self.assertEqual(axis['intervals'],[interval(0,0)])
        self.assertEqual(r.density(np.zeros(6))['log_density'],r.density(np.zeros(6))['baseline_log_density'])

    def test_coefficients_match_direct_quaternions_and_fixed_axis_circle(self):
        rng=np.random.default_rng(61005011)
        for _ in range(180):
            j=int(rng.integers(3));v=rng.normal(size=3);v[j]=0;a,b=rng.normal(size=(2,3));ell=math.exp(rng.uniform(-1,1));radius=rng.uniform(.1,2.)
            coefficients=sphere_coefficients(a,b,v,j,ell,radius);e=np.eye(3)[j];h=math.sqrt(1+v@v);k=(e+np.cross(v,e))/h
            R0=Rotation.from_quat(np.r_[v,1.]/h)
            for raw in [-4.,-.2,0.,.7,3.]:
                c=v.copy();c[j]=raw/ell;R=Rotation.from_quat(np.r_[c,1.]/np.linalg.norm(np.r_[c,1.]))
                expected=np.sum((R.apply(a)-b)**2)-radius**2
                got=float(np.polynomial.polynomial.polyval(np.longdouble(raw),coefficients))/(1+c@c)
                self.assertAlmostEqual(got,expected,delta=3e-13*(1+abs(expected)))
                circle=Rotation.from_rotvec(k*(2*math.atan(raw/(ell*h))))*R0
                np.testing.assert_allclose(circle.as_matrix(),R.as_matrix(),atol=9e-16)

    def test_swept_candidates_equal_exhaustive_and_direct_atom_witnesses(self):
        _,_,cfg,shape=setup(anisotropic=True);Rf=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();Ra=Rotation.from_rotvec([-.1,.3,.2]).as_matrix()
        center=np.array([.1,-.05,.03]);v=np.array([.15,-.2,0.]);segment=[-4.,3.]
        fast=angular_hard_free_intervals(shape,cfg['fixed_poses'],center,Rf,Ra,v,2,1.7,segment)
        slow=angular_hard_free_intervals(shape,cfg['fixed_poses'],center,Rf,Ra,v,2,1.7,segment,False)
        self.assertEqual(fast['intervals'],slow['intervals'])
        atoms=np.array([a['center']for a in shape['atoms']]);radii=np.array([a['radius']for a in shape['atoms']])
        for raw in np.linspace(*segment,251):
            c=v.copy();c[2]=raw/1.7;R=Rf@Rotation.from_quat(np.r_[c,1.]).as_matrix()@Ra
            moving=atoms@R.T+center;gaps=[]
            for fixed in cfg['fixed_poses']:
                points=atoms+fixed['position'];gaps.extend((np.linalg.norm(moving[:,None]-points[None,:],axis=-1)-radii[:,None]-radii[None,:]).ravel())
            if min(abs(np.array(gaps)))>1e-10:self.assertEqual(contains(fast['intervals'],raw),min(gaps)>=0)

    def test_global_rigid_frame_covariance(self):
        _,_,cfg,shape=setup(anisotropic=True);Rf=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();Ra=Rotation.from_rotvec([-.1,.3,.2]).as_matrix()
        center=np.array([.1,-.05,.03]);v=np.array([.15,0.,-.2]);Q=Rotation.from_rotvec([.4,.2,-.3]);shift=np.array([4.,-2.,3.])
        a=angular_hard_free_intervals(shape,cfg['fixed_poses'],center,Rf,Ra,v,1,1.7,[-4.,3.])
        fixed=[pose(Q.apply(f['position'])+shift,(Q*Rotation.from_quat(np.array(f['orientation'])[[1,2,3,0]])).as_quat()[[3,0,1,2]])for f in cfg['fixed_poses']]
        b=angular_hard_free_intervals(shape,fixed,Q.apply(center)+shift,Q.as_matrix()@Rf,Ra,v,1,1.7,[-4.,3.])
        self.assertEqual(len(a['intervals']),len(b['intervals']))
        for x,y in zip(a['intervals'],b['intervals']):np.testing.assert_allclose([x['lower'],x['upper']],[y['lower'],y['upper']],atol=4e-13)


class PoseGaussianLawTests(unittest.TestCase):
    def test_schur_normalization_without_haar_in_conditional_mass(self):
        recon=Reconstructor(*setup());u=np.zeros(6);u[0]=.01;x=recon.raw(u)
        rawcov=recon.L0@recon.guide['gaussian_components'][0]['covariance']@recon.L0.T
        for axis in recon.axes:
            geom=recon.reconstruct_axis(u,axis);means,sigmas=recon.conditional(x,axis)
            for mean,sigma in zip(means,sigmas):
                mass=float(interval_masses(geom['intervals'],mean,sigma))
                value=quad(lambda s:norm.pdf(s,mean,sigma),mean-12*sigma,mean+12*sigma)[0] if mass<=recon.floor else sum(quad(lambda s:norm.pdf(s,mean,sigma)/mass,i['lower'],i['upper'],epsabs=2e-12)[0]for i in geom['intervals'])
                self.assertAlmostEqual(value,1.,places=11)
            others=[j for j in range(6)if j!=axis];logmarg=multivariate_normal.logpdf(x[others],mean=recon.rawmeans[0,others],cov=rawcov[np.ix_(others,others)])
            changed=x.copy();changed[axis]=.7
            joint=multivariate_normal.logpdf(changed,mean=recon.rawmeans[0],cov=rawcov)
            self.assertAlmostEqual(joint-logmarg,norm.logpdf(.7,means[0],sigmas[0]),places=12)
            u2=np.linalg.solve(recon.L0,changed-recon.m0)
            np.testing.assert_allclose(recon.decode(u)[1],recon.decode(u2)[1],atol=1e-14)
            self.assertNotAlmostEqual(recon.decode(u)[3],recon.decode(u2)[3],places=4)

    def test_six_axis_full_density_arithmetic_mixture_and_dominance(self):
        region,guide,cfg,shape=setup(range(6),anisotropic=True);recon=Reconstructor(region,guide,cfg,shape)
        for u in [np.zeros(6),np.array([.1,.1,0.,.2,-.1,.3])]:
            full=recon.density(u);parts=[]
            for axis in recon.axes:
                g=copy.deepcopy(guide);g['raw_pose_axes']=[axis];r=Reconstructor(region,g,cfg,shape);part=r.density(u);parts.append(math.exp(part['log_density']))
                _,p,R,_=r.decode(u)
                if r.hard_valid(p,R):self.assertGreaterEqual(part['log_density'],part['baseline_log_density']-1e-13)
            self.assertAlmostEqual(math.exp(full['log_density']),sum(parts)/6,places=12)

    def test_zero_support_fallback_disabled_and_pure_uniform(self):
        reg,g,cfg,shape=setup((5,));r=Reconstructor(reg,g,cfg,shape);raw=np.zeros(6);raw[5]=10;u=np.linalg.solve(r.L0,raw-r.m0)
        result=r.density(u);self.assertEqual(result['log_density'],-math.inf);self.assertTrue(math.isfinite(result['baseline_log_density']))
        cfg['capture_radius']=.01;raw[0]=1.;u=np.linalg.solve(r.L0,raw-r.m0);r=Reconstructor(reg,g,cfg,shape)
        self.assertEqual(r.reconstruct_axis(u,5)['empty_reason'],'no_capture_at_fixed_center')
        self.assertEqual(r.density(u)['log_density'],r.density(u)['baseline_log_density'])
        for mode in ('beta0','alpha1'):
            reg,g,cfg,shape=setup(range(6))
            if mode=='beta0':g['conditional_probability']=0.
            else:g['defensive_uniform_shell_probability']=1.;g['gaussian_components']=[]
            r=Reconstructor(reg,g,cfg,shape);self.assertEqual(r.density(np.zeros(6))['log_density'],r.density(np.zeros(6))['baseline_log_density'])

    def test_component_specific_angular_floor_and_trace_tamper(self):
        reg,g,cfg,shape=setup((3,));g['gaussian_components'][1]['mean']=[0,0,0,30,0,0]
        r=Reconstructor(reg,g,cfg,shape);u=np.zeros(6);d=r.density(u);self.assertEqual(d['fallback_component_branches'],1)
        axis=d['axes'][0];actual={k:v for k,v in axis.items()if k not in ('intervals','conditional_means','conditional_sigmas','conditional_masses','component_fallbacks','hard_overlap','swept_circle_pair_candidates')};actual['hard_free_intervals']=axis['intervals']
        self.assertEqual(compare_axis(actual,axis),0.)
        bad=copy.deepcopy(actual);bad['fixed_cayley'][1]=1.
        with self.assertRaisesRegex(ValueError,'Different angular fixed_cayley'):compare_axis(bad,axis)

    def test_schema_cannot_silently_accept_old_axes_or_contact_labels(self):
        for key,value in [('raw_translation_axes',[0]),('contact_widths_A',[.1]),('contact_neighbor_indices',[0])]:
            r,g,c,s=setup();g[key]=value
            with self.assertRaisesRegex(ValueError,'Unexpected'):Reconstructor(r,g,c,s)
        r,g,c,s=setup();g['raw_pose_axes']=[3,3]
        with self.assertRaisesRegex(ValueError,'Bad pose axes'):Reconstructor(r,g,c,s)


if __name__=='__main__':unittest.main()
