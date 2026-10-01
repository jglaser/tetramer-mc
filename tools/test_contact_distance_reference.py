#!/usr/bin/env python3
"""Independent numerical checks; no physical sampling or cloud generation."""
import copy
from collections import Counter
import math
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from contact_distance_reference import (Reconstructor,azimuth_log_density,azimuth_parameters,check_draw,check_density_details,
    cartesian_from_radii,circle_coordinates,gaussian_conditionals,normal_log_density,
    polygon,radii_from_cartesian,radii_supported,transverse_basis,wrapped_cauchy_inverse,
    wrapped_cauchy_log_density)


CONTROLS=dict(localized_probability=.9,radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=math.pi)


def fixture():
    L=np.eye(6);L[3,0]=.13;L[4,2]=-.1;L[5,1]=.08
    region=dict(minimum_mahalanobis_radius=0.,mahalanobis_radius=4.,
        gaussian_chart=dict(means=[[0.]*6],covariances=[(L@L.T).tolist()],angular_length=2.,
            anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]),
        fixed_neighbor=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))
    B=np.eye(6)*.7;B[3,0]=.15;B[5,2]=-.13;B[4,1]=.1
    guide=dict(schema='defensive-contact-distance-guide-v1',defensive_uniform_shell_probability=.5,
        conditional_probability=.5,minimum_center_distance=1e-8,minimum_polygon_area=1e-16,
        contact_widths_A=[.02,.1,.5],contact_neighbor_indices=[0,1],azimuth=CONTROLS.copy(),
        gaussian_components=[dict(weight=.3,mean=[0.]*6,covariance=(B@B.T).tolist()),
            dict(weight=.7,mean=[.1,-.2,.3,.2,.1,-.1],covariance=(B@B.T*1.2).tolist())],
        component_contact_pairs=[[dict(neighbor_index=0,moving_atom=0,fixed_atom=0),
            dict(neighbor_index=1,moving_atom=0,fixed_atom=0)] for _ in range(2)])
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
    config=dict(fixed_poses=[dict(position=[-1.2,0.,0.],orientation=[1.,0.,0.,0.]),
        dict(position=[1.2,0.,0.],orientation=[1.,0.,0.,0.])],capture_center=[0.,0.,0.],capture_radius=10.)
    return region,guide,config,shape


class GeometryTests(unittest.TestCase):
    def test_exact_binary64_polygon_areas_at_fallback_floor(self):
        fixture=Path(__file__).resolve().parents[1]/'tests/data/contact_distance_polygon_reference.json'
        for case in json.loads(fixture.read_text())['cases']:
            with self.subTest(case=case['name']):
                bounds=case['bounds'];width=bounds[0][1]-bounds[0][0]
                _,area=polygon(bounds[0][0],bounds[1][0],case['distance'],width)
                if case['area']>0:self.assertAlmostEqual(area/case['area'],1.,delta=1e-12)
                else:self.assertEqual(area,0.)
                self.assertEqual(area<=1e-16,case['fallback_at_1e_16'])

    def test_polygon_area_against_independent_vertical_integral(self):
        cases=[(2.,2.,2.4,.02),(2.,2.,4.01,.1),(1.,3.,2.01,.1),(1.,1.,2.9,.5),
            (1.,1.,.02,.1),(1.,1.,3.,.5),(1.,1.,4.,.2),(1.,1.,2.,0.),
            (2.,2.,4.00000001,.00000003)]
        for d1,d2,D,w in cases:
            with self.subTest(case=(d1,d2,D,w)):
                vertices,area=polygon(d1,d2,D,w)
                def length(x):return max(0.,min(d2+w,x+D)-max(d2,D-x,x-D))
                points=sorted({d1,d1+w,*[p for p in [D-d2,D-d2-w,d2-D,d2+w-D,d2+D,d2+w+D,D/2] if d1<p<d1+w]})
                exact=sum((b-a)*(length(a)+length(b))/2 for a,b in zip(points[:-1],points[1:]))
                self.assertAlmostEqual(area,exact,delta=1e-13*max(1.,w*w))
                if len(vertices):
                    self.assertTrue(all(radii_supported(x,y,d1,d2,D,w) or min(abs(x-d1),abs(x-d1-w),abs(y-d2),abs(y-d2-w))<1e-12 for x,y in vertices))

    def test_coordinate_inverse_jacobian_and_label_exchange(self):
        c1=np.array([.3,-.2,.8]);c2=np.array([1.9,.7,-.1]);D=np.linalg.norm(c2-c1)
        for r1,r2,phi in [(2.,2.3,.1),(1.4,2.,2.1),(2.3,2.4,5.4)]:
            point=cartesian_from_radii(c1,c2,r1,r2,phi);inverse=radii_from_cartesian(c1,c2,point)
            self.assertTrue(np.allclose([inverse['r1'],inverse['r2'],inverse['phi']],[r1,r2,phi],atol=1e-13))
            v=np.array([r1,r2,phi]);eps=1e-5;columns=[]
            for j in range(3):
                delta=np.eye(3)[j]*eps
                columns.append((cartesian_from_radii(c1,c2,*(v+delta))-cartesian_from_radii(c1,c2,*(v-delta)))/(2*eps))
            self.assertAlmostEqual(abs(np.linalg.det(np.asarray(columns).T)),r1*r2/D,delta=2e-9)
            reversed=radii_from_cartesian(c2,c1,point)
            self.assertTrue(np.allclose(cartesian_from_radii(c2,c1,reversed['r1'],reversed['r2'],reversed['phi']),point,atol=2e-14))
            S=np.array([[.04,.002,0],[.002,.03,.001],[0,.001,.05]]);mean=np.array([.3,1.1,.7])
            p=azimuth_parameters(c1,c2,inverse['rho'],mean,S,CONTROLS)
            q=azimuth_parameters(c2,c1,reversed['rho'],mean,S,CONTROLS)
            self.assertAlmostEqual(azimuth_log_density(inverse['phi'],p,CONTROLS),azimuth_log_density(reversed['phi'],q,CONTROLS),places=12)

    def test_tangencies_and_declared_floors(self):
        for r1,r2,D in [(1.,1.,2.),(2.,1.,1.)]:
            _,rho=circle_coordinates(r1,r2,D);self.assertEqual(rho,0.)
        _,rho=circle_coordinates(1.,1.,2.-1e-10);self.assertGreater(rho,0)
        with self.assertRaises(ValueError):circle_coordinates(1.,1.,2.1)
        with self.assertRaises(ValueError):transverse_basis(np.zeros(3),np.zeros(3))
        p=azimuth_parameters(np.zeros(3),np.array([1.,0,0]),2.,np.array([2.,0,0]),np.eye(3),CONTROLS)
        self.assertFalse(p['localization_enabled'])
        for phi in [.1,2.,6.]:self.assertEqual(azimuth_log_density(phi,p,CONTROLS),-math.log(2*math.pi))
        p=azimuth_parameters(np.zeros(3),np.array([1.,0,0]),0.,np.array([0.,1,0]),np.eye(3),CONTROLS)
        self.assertEqual(p['gamma'],math.pi)

    def test_wrapped_cauchy_normalization_and_inverse(self):
        for gamma in [.01,.1,1.,math.pi]:
            # Centered integral, with the narrow peak exposed to quadrature.
            total=quad(lambda p:math.exp(wrapped_cauchy_log_density(p,0.,gamma)),-math.pi,math.pi,epsabs=2e-12,points=[0.])[0]
            self.assertAlmostEqual(total,1.,places=11)
            for u in [.00001,.01,.1,.5,.8,.99,.99999]:
                phi=float(wrapped_cauchy_inverse(u,0.,gamma));centered=(phi+math.pi)%(2*math.pi)-math.pi
                cdf=quad(lambda p:math.exp(wrapped_cauchy_log_density(p,0.,gamma)),-math.pi,centered,epsabs=2e-12,points=[0.] if centered>0 else None)[0]
                self.assertAlmostEqual(cdf,u,places=10)
            uniform=dict(CONTROLS,localized_probability=0.)
            self.assertEqual(azimuth_log_density(.7,dict(localization_enabled=True,phi_mean=.3,gamma=gamma),uniform),-math.log(2*math.pi))

    def test_translation_density_integrates_to_one(self):
        # Independent quadrature over physical cylindrical volume, dV=rho dρ dh dφ.
        # Two equal atomic-radius constraints give an explicit (h,rho) domain.
        D=2.4;d=2.;w=.2;c1=np.zeros(3);c2=np.array([D,0.,0.]);_,area=polygon(d,d,D,w)
        # Test both uniform and localized azimuth; gamma varies with circle radius.
        for b in [0.,.9]:
            controls=dict(CONTROLS,localized_probability=b)
            def radial(h):
                lo2=max(0.,d*d-h*h,d*d-(h-D)**2)
                hi2=min((d+w)**2-h*h,(d+w)**2-(h-D)**2)
                if hi2<=lo2:return 0.
                def integrand(rho):
                    p=azimuth_parameters(c1,c2,rho,np.array([1.1,.5,.8]),np.eye(3)*.09,controls)
                    angular=quad(lambda phi:math.exp(azimuth_log_density(phi,p,controls)),0.,2*math.pi,epsabs=2e-9,points=[p['phi_mean']])[0]
                    r1=math.hypot(h,rho);r2=math.hypot(h-D,rho)
                    return rho*D/(area*r1*r2)*angular
                return quad(integrand,math.sqrt(lo2),math.sqrt(hi2),epsabs=2e-8)[0]
            hmin=(d*d-(d+w)**2+D*D)/(2*D);hmax=((d+w)**2-d*d+D*D)/(2*D)
            integral=quad(radial,hmin,hmax,epsabs=2e-8,points=[D/2])[0]
            self.assertAlmostEqual(integral,1.,delta=2e-7)


class GaussianGuideTests(unittest.TestCase):
    def test_nonunit_chart_determinant_enters_conditioned_density(self):
        region,guide,config,shape=fixture();L=np.linalg.cholesky(np.asarray(region['gaussian_chart']['covariances'][0]))
        L=np.diag([.8,1.1,.9,.6,.7,.8])@L;region['gaussian_chart']['covariances']=[(L@L.T).tolist()]
        guide['azimuth']['localized_probability']=0.;recon=Reconstructor(region,guide,config,shape)
        raw=np.array([0.,math.sqrt(2.01**2-1.2**2),0.,.1,.2,-.1]);u=np.linalg.solve(L,raw);result=recon.density(u,True)
        self.assertGreater(abs(math.log(np.linalg.det(L))),.5)
        for branch in result['branches']:
            k=branch['component'];a=recon.rawcov[k,3:,3:];mean=recon.rawmeans[k,3:]
            angular=math.exp(normal_log_density(raw[3:],mean,a));width=recon.widths[branch['width_index']]
            # The entire small rectangle satisfies triangle inequalities here.
            area=((2.+width)-2.)**2
            expected=recon.weights[k]*np.linalg.det(L)*angular*2.4/(2*math.pi*area*2.01**2)
            self.assertAlmostEqual(math.exp(branch['weighted_latent_log_density'])/expected,1.,places=12)
        # A finite difference on the actual latent->raw map independently fixes
        # the determinant direction (it must multiply H, not divide it).
        eps=1e-5;jac=np.column_stack([(recon.raw(u+np.eye(6)[j]*eps)-recon.raw(u-np.eye(6)[j]*eps))/(2*eps) for j in range(6)])
        self.assertAlmostEqual(abs(np.linalg.det(jac)),math.exp(recon.logdet),delta=2e-11)

    def test_schur_density_factorization_with_correlations(self):
        generator=np.random.default_rng(941);B=generator.normal(size=(6,6));s=B@B.T+np.eye(6)*.4
        mean=generator.normal(size=6);x=generator.normal(size=6)
        m,c,angular=gaussian_conditionals(mean,s,x[3:])
        self.assertAlmostEqual(normal_log_density(x,mean,s),angular+normal_log_density(x[:3],m,c),places=12)
        self.assertGreater(np.linalg.norm(c-s[:3,:3]),.1)

    def test_mixture_density_all_widths_components_and_support(self):
        recon=Reconstructor(*fixture());raw=np.array([0.,math.sqrt(2.01**2-1.2**2),0.,.1,.2,-.1]);u=np.linalg.solve(recon.L0,raw-recon.m0)
        got=recon.density(u,True);self.assertEqual(got['component_branches'],6);self.assertEqual(got['fallback_component_branches'],0)
        h=sum(math.exp(b['weighted_latent_log_density']) for b in got['branches'])/3
        g=sum(math.exp(v) for v in recon.gaussian_logs(u));uniform=math.exp(-recon.logvolume)
        self.assertAlmostEqual(math.exp(got['log_density']),.5*uniform+.25*g+.25*h,places=12)
        self.assertGreaterEqual(got['log_density'],got['baseline_log_density']+math.log(.5)-1e-12)
        for branch in got['branches']:self.assertTrue(branch['inside'])
        # Raw-coordinate angular density * world translation * det(L0) is the
        # latent density, including correlated translation/angular whitening.
        self.assertNotEqual(recon.L0[3,0],0.)
        region,guide,config,shape=fixture();guide['conditional_probability']=0.
        baseline=Reconstructor(region,guide,config,shape).density(u)
        self.assertEqual(baseline['log_density'],got['baseline_log_density'])

    def test_fallback_reduces_exactly_to_old_density_without_endpoint_selection(self):
        for mode in ['coincident','empty_area','floor_area']:
            region,guide,config,shape=fixture()
            if mode=='coincident':config['fixed_poses'][1]['position']=config['fixed_poses'][0]['position'][:]
            if mode=='empty_area':config['fixed_poses'][1]['position']=[20.,0.,0.]
            if mode=='floor_area':guide['minimum_polygon_area']=1.
            recon=Reconstructor(region,guide,config,shape)
            for u in [np.zeros(6),np.array([9.,1.,2.,.3,-.2,.1])]:
                got=recon.density(u);self.assertEqual(got['fallback_component_branches'],6)
                self.assertAlmostEqual(got['log_density'],got['baseline_log_density'],places=12)
            self.assertTrue(all(b['fallback'] for b in got['branches']))

    def test_component_specific_fallback_and_empty_pure_uniform(self):
        region,guide,config,shape=fixture();shape['atoms'].append(dict(center=[10.,0.,0.],radius=1.))
        guide['component_contact_pairs'][1][1]['moving_atom']=1
        recon=Reconstructor(region,guide,config,shape);got=recon.density(np.zeros(6))
        self.assertEqual(got['fallback_component_branches'],3)
        self.assertTrue(all(b['fallback']==(b['component']==1) for b in got['branches']))
        guide['defensive_uniform_shell_probability']=1.;guide['gaussian_components']=[];guide['component_contact_pairs']=[]
        recon=Reconstructor(region,guide,config,shape);got=recon.density(np.zeros(6))
        self.assertEqual(got['log_density'],-recon.logvolume)
        self.assertEqual(recon.density(np.ones(6)*10)['log_density'],-math.inf)

    def test_recorded_draw_replay_and_orientation_tamper(self):
        recon=Reconstructor(*fixture());original=np.array([.1,-.2,.3,.4,-.3,.2]);raw,_,R,_=recon.decode(original)
        k,wi=0,1;centers,distances=recon.centers(R,k);D=np.linalg.norm(centers[1]-centers[0])
        vertices,area=polygon(*distances,D,recon.widths[wi]);ru=np.array([.2,.4,.6])
        a=vertices[1:-1]-vertices[0];b=vertices[2:]-vertices[0];areas=(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
        triangle=int(np.searchsorted(np.cumsum(areas),ru[0]*area));s=math.sqrt(ru[1]);radii=np.array([1-s,s*(1-ru[2]),s*ru[2]])@vertices[[0,triangle+1,triangle+2]]
        h,rho=circle_coordinates(*radii,D);_,worldmeans,angular=recon.conditionals(raw)
        params=azimuth_parameters(*centers,rho,worldmeans[k],recon.world_covariance[k],recon.controls)
        az=np.array([.2,.7]);phi=float(wrapped_cauchy_inverse(az[1],params['phi_mean'],params['gamma']))
        position=cartesian_from_radii(*centers,*radii,phi);newraw=raw.copy();newraw[:3]=recon.Rf.T@(position-recon.offset)
        u=np.linalg.solve(recon.L0,newraw-recon.m0);result=recon.density(u);_,axis,_,_=transverse_basis(*centers)
        draw=dict(conditional=True,component=k,width_index=wi,width_A=recon.widths[wi],original_latent=original.tolist(),
            retained_raw_angles=raw[3:].tolist(),fallback=False,fallback_reason=None,distance=D,radii=radii.tolist(),
            polygon_area=area,circle_center=(centers[0]+h*axis).tolist(),circle_radius=rho,phi=phi,
            world_translation=position.tolist(),angular_log_density=angular[k],conditional_world_mean=worldmeans[k].tolist(),
            azimuth_log_density=azimuth_log_density(phi,params,recon.controls),azimuth_uniforms=az.tolist(),
            radius_uniforms=ru.tolist(),polygon_vertices=vertices.tolist())
        row=dict(latent=u.tolist(),draw=draw);maxima=Counter()
        self.assertTrue(check_draw(recon,row,result,maxima)['radius_uniforms_replayed'])
        self.assertLess(maxima['retained_angles_absolute'],1e-14)
        bad=copy.deepcopy(row);bad['draw']['retained_raw_angles'][0]+=.001
        with self.assertRaisesRegex(ValueError,'retained angles'):check_draw(recon,bad,result,Counter())
        bad=copy.deepcopy(row);bad['draw']['azimuth_uniforms'][1]+=.1
        with self.assertRaisesRegex(ValueError,'Azimuth inverse'):check_draw(recon,bad,result,Counter())

    def test_full_density_record_rejects_omitted_width_and_wrong_fallback(self):
        recon=Reconstructor(*fixture());u=np.zeros(6);result=recon.density(u);_,means,angular=recon.conditionals(recon.raw(u))
        record=dict(raw_coordinates=recon.raw(u).tolist(),baseline_log_density=result['baseline_log_density'],
            component_branches=result['component_branches'],fallback_component_branches=result['fallback_component_branches'],components=[])
        for k in range(len(recon.weights)):
            widths=[]
            for wi,w in enumerate(recon.widths):
                b=result['branches'][k*len(recon.widths)+wi]
                widths.append(dict(width_index=wi,width_A=w,distance=b['D'],polygon_area=b['polygon_area'],fallback=b['fallback'],
                    translation_log_density=b['translation_log_density'] if math.isfinite(b['translation_log_density']) else None))
            record['components'].append(dict(component=k,angular_log_density=angular[k],conditional_world_mean=means[k].tolist(),widths=widths))
        check_density_details(recon,u,result,record,Counter())
        bad=copy.deepcopy(record);bad['components'][0]['widths'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete width'):check_density_details(recon,u,result,bad,Counter())
        bad=copy.deepcopy(record);bad['components'][0]['widths'][0]['fallback']=True
        with self.assertRaisesRegex(ValueError,'fallback differs'):check_density_details(recon,u,result,bad,Counter())


if __name__=='__main__':unittest.main()
