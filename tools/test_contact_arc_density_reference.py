#!/usr/bin/env python3
"""Normalized arc-law and complete synthetic-mixture references; no protein runs."""
import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from contact_arc_density_reference import ArcDensityReference,QueryGeometryCache,conditional_azimuth_factor,compare_rows,json_safe
from contact_circle_reference import TAU,azimuth_mass,interval,union
from contact_distance_reference import wrapped_cauchy_log_density


def fixture(localized=.9):
    L=np.eye(6);L[0,0]=.8;L[3,0]=.13;L[4,2]=-.1;L[5,1]=.08
    region=dict(minimum_mahalanobis_radius=0.,mahalanobis_radius=4.,
        gaussian_chart=dict(means=[[0.]*6],covariances=[(L@L.T).tolist()],angular_length=2.,
            anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]),
        fixed_neighbor=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))
    B=np.eye(6)*.4;B[3,0]=.08;B[5,2]=-.03;B[4,1]=.1
    guide=dict(schema='defensive-contact-distance-guide-v1',defensive_uniform_shell_probability=.5,
        conditional_probability=.5,minimum_center_distance=1e-8,minimum_polygon_area=1e-16,
        contact_widths_A=[.02,.1,.5],contact_neighbor_indices=[0,1],
        azimuth=dict(localized_probability=localized,radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=math.pi),
        gaussian_components=[dict(weight=.3,mean=[0.,.4,1.5,.1,-.1,.3],covariance=(B@B.T).tolist()),
            dict(weight=.7,mean=[0.,-1.4,-.2,-.2,.1,0.],covariance=(B@B.T*1.2).tolist())],
        component_contact_pairs=[[dict(neighbor_index=0,moving_atom=0,fixed_atom=0),dict(neighbor_index=1,moving_atom=0,fixed_atom=0)] for _ in range(2)])
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)]);pose=lambda p:dict(position=p,orientation=[1.,0.,0.,0.])
    config=dict(fixed_poses=[pose([-1.2,0.,0.]),pose([1.2,0.,0.]),pose([0.,0.,3.])],capture_center=[0.,0.,0.],capture_radius=10.)
    return region,guide,config,shape


def query(scorer,phi):
    rho=math.sqrt(2.01**2-1.2**2);raw=np.array([0.,-rho*math.sin(phi),rho*math.cos(phi),.1,.2,-.1])
    return np.linalg.solve(scorer.base.L0,raw-scorer.base.m0)


class ConditionalAzimuthTests(unittest.TestCase):
    def test_normalization_with_full_empty_partial_and_threshold_fallback(self):
        sets=[[],[interval(0,TAU,True,False)],[interval(.3,1.2,True,True),interval(5.4,TAU,True,False)],
            [interval(0,TAU*.5e-12,True,False)],[interval(0,TAU*2e-12,True,False)]]
        for b in [0.,.9]:
            law=dict(mode=.7,gamma=.1,localized_probability=b)
            for allowed in sets:
                correction=conditional_azimuth_factor(.2,allowed,law)
                def original(phi):return (1-b)/TAU+b*math.exp(wrapped_cauchy_log_density(phi,law['mode'],law['gamma']))
                if correction['arc_fallback']:
                    total=quad(original,0,TAU,points=[law['mode']],epsabs=1e-12)[0]
                    self.assertEqual(correction['log_correction'],0.)
                else:
                    total=math.fsum(quad(original,i['lower'],i['upper'],epsabs=1e-30,epsrel=1e-11,
                        points=[law['mode']] if i['lower']<law['mode']<i['upper'] else None)[0]/correction['arc_mass'] for i in allowed)
                self.assertAlmostEqual(total,1.,places=10)

    def test_component_specific_mass_floor_on_identical_arcs(self):
        allowed=[interval(0,5e-14,True,False)]
        near=dict(mode=0.,gamma=.01,localized_probability=.9);far=dict(near,mode=math.pi)
        a=conditional_azimuth_factor(1e-14,allowed,near);b=conditional_azimuth_factor(1e-14,allowed,far)
        self.assertGreater(a['arc_mass'],1e-12);self.assertFalse(a['arc_fallback'])
        self.assertLess(b['arc_mass'],1e-12);self.assertTrue(b['arc_fallback'])
        self.assertEqual(b['log_correction'],0.);self.assertAlmostEqual(a['log_correction'],-math.log(a['arc_mass']),places=13)

    def test_endpoint_membership_and_zero_probability_singleton(self):
        law=dict(mode=.3,gamma=.1,localized_probability=.9);arcs=[interval(.2,.8,True,False)]
        self.assertTrue(conditional_azimuth_factor(.2,arcs,law)['query_phi_allowed'])
        self.assertFalse(conditional_azimuth_factor(.8,arcs,law)['query_phi_allowed'])
        self.assertEqual(conditional_azimuth_factor(.8,arcs,law)['log_correction'],-math.inf)
        onlypoint=[interval(1.,1.,True,True)]
        result=conditional_azimuth_factor(1.,onlypoint,law)
        self.assertEqual(result['arc_mass'],0.);self.assertTrue(result['arc_fallback']);self.assertEqual(result['log_correction'],0.)


class CompleteMixtureTests(unittest.TestCase):
    def test_complete_weights_chart_jacobian_and_component_normalizers(self):
        scorer=ArcDensityReference(*fixture());u=query(scorer,2.1);cache=QueryGeometryCache();result=scorer.score(u,cache)
        self.assertEqual(result['component_branches'],6);self.assertEqual(result['arc_branches'],6)
        self.assertEqual(result['geometry_builds'],1);self.assertEqual(result['geometry_count'],1)
        self.assertTrue(all(b['query_phi_allowed'] for b in result['branches']))
        masses={k:next(b['arc_mass'] for b in result['branches'] if b['component']==k) for k in range(2)}
        self.assertGreater(abs(masses[0]-masses[1]),.05)
        old=scorer.base.density(u,True);G=sum(math.exp(x) for x in scorer.base.gaussian_logs(u))
        expectedH=sum(math.exp(b['weighted_latent_log_density'])/masses[b['component']] for b in old['branches'])/3
        expected=.5*math.exp(-scorer.base.logvolume)+.25*G+.25*expectedH
        self.assertAlmostEqual(math.exp(result['log_density'])/expected,1.,places=12)
        self.assertAlmostEqual(result['old_distance_log_density'],old['log_density'],places=13)
        self.assertAlmostEqual(result['log_physical_jacobian'],scorer.base.decode(u)[3],places=13)
        self.assertGreaterEqual(result['log_density'],result['baseline_log_density']+math.log(.5)-1e-12)

    def test_blocked_query_loses_only_nonfallback_azimuth_terms(self):
        scorer=ArcDensityReference(*fixture());u=query(scorer,0.);result=scorer.score(u)
        self.assertTrue(all(not b['query_phi_allowed'] and not b['arc_fallback'] for b in result['branches']))
        self.assertTrue(all(b['new_weighted_latent_log_density']==-math.inf for b in result['branches']))
        G=sum(math.exp(x) for x in scorer.base.gaussian_logs(u))
        self.assertAlmostEqual(math.exp(result['log_density']),.5*math.exp(-scorer.base.logvolume)+.25*G,places=13)

    def test_outer_fallbacks_remain_gaussian_and_unsupported_widths_stay_zero(self):
        region,guide,config,shape=fixture();guide['minimum_polygon_area']=1.
        scorer=ArcDensityReference(region,guide,config,shape);u=query(scorer,2.1);result=scorer.score(u)
        self.assertEqual(result['outer_fallback_branches'],6);self.assertEqual(result['geometry_count'],0)
        self.assertEqual(result['arc_branches'],0);self.assertAlmostEqual(result['log_density'],result['baseline_log_density'],places=13)
        scorer=ArcDensityReference(*fixture());raw=np.array([0.,math.sqrt(2.2**2-1.2**2),0.,.1,.2,-.1]);u=np.linalg.solve(scorer.base.L0,raw)
        result=scorer.score(u);self.assertEqual(result['arc_branches'],2)
        for b in result['branches']:
            if b['width_index']<2:self.assertIsNone(b['geometry_id']);self.assertEqual(b['new_weighted_latent_log_density'],-math.inf)

    def test_empty_and_small_mass_fallback_keep_original_law_everywhere(self):
        scorer=ArcDensityReference(*fixture());u=query(scorer,0.)
        old=scorer.base.density(u)['log_density']
        for allowed in [[],[interval(0,1e-20,True,False)]]:
            with patch.object(scorer,'_geometry',return_value=dict(allowed=allowed)):
                result=scorer.score(u)
            self.assertEqual(result['arc_fallback_branches'],6)
            self.assertAlmostEqual(result['log_density'],old,places=13)
            self.assertTrue(all(b['new_translation_log_density']==b['old_translation_log_density'] for b in result['branches']))

    def test_cache_across_widths_components_and_arms_preserves_scores(self):
        first=ArcDensityReference(*fixture(0.));second=ArcDensityReference(*fixture(.9));u=query(first,2.1);cache=QueryGeometryCache()
        a=first.score(u,cache);b=second.score(u,cache);standalone=second.score(u)
        self.assertEqual(a['geometry_builds'],1);self.assertEqual(b['geometry_builds'],0);self.assertEqual(b['geometry_cache_hits'],6)
        self.assertEqual(b['log_density'],standalone['log_density'])
        self.assertNotEqual(a['branches'][0]['arc_mass'],b['branches'][0]['arc_mass'])
        with self.assertRaisesRegex(ValueError,'different configuration/query'):second.score(u+np.eye(6)[0]*.001,cache)
        region,guide,config,shape=fixture(.9);config['fixed_poses'][2]['position'][2]+=.1
        changed=ArcDensityReference(region,guide,config,shape)
        with self.assertRaisesRegex(ValueError,'different configuration/query'):changed.score(u,cache)

    def test_disabled_conditioner_is_exact_old_density(self):
        for control in ['beta0','alpha1']:
            region,guide,config,shape=fixture()
            guide['conditional_probability' if control=='beta0' else 'defensive_uniform_shell_probability']=0. if control=='beta0' else 1.
            scorer=ArcDensityReference(region,guide,config,shape);u=np.zeros(6)
            got=scorer.score(u);self.assertTrue(got['conditioning_disabled']);self.assertEqual(got['geometry_count'],0)
            self.assertEqual(got['log_density'],scorer.base.density(u)['log_density'])

    def test_floor_one_restores_old_law_for_all_circle_topologies(self):
        for phi in [0.,1.3,3.4]:
            scorer=ArcDensityReference(*fixture(),minimum_arc_mass=1.)
            result=scorer.score(query(scorer,phi))
            self.assertEqual(result['arc_fallback_branches'],6)
            self.assertEqual(result['log_density'],result['old_distance_log_density'])


def comparison_fixture():
    """Schema-only Rust surrogate, used to test rejection of altered receipts."""
    scorer=ArcDensityReference(*fixture());u=query(scorer,2.1);cache=QueryGeometryCache();arm=scorer.score(u,cache)
    p=dict(id='probe',latent=u.tolist(),arms=[arm],geometries=cache.entries,hard_valid=True,shell_valid=True,capture_valid=True)
    components=[];requests=0
    for component in arm['components']:
        widths=[]
        for w in component['widths']:
            row=dict(width_index=w['width_index'],width_A=w['width_A'],fallback=w['outer_fallback'],distance=w['distance'],polygon_area=w['polygon_area'])
            if w['outer_fallback']:row['fallback_reason']=w['outer_fallback_reason']
            else:
                row.update(old_translation_log_density=w['old_translation_log_density'],translation_log_density=w['new_translation_log_density'])
                if w['geometry_id'] is not None:
                    for key in ['geometry_id','arc_mass','arc_fallback','query_phi_allowed','query_phi','azimuth_law']:row[key]=w[key]
                    requests+=1
            widths.append(row)
        components.append(dict(component=component['component'],angular_log_density=component['angular_log_density'],conditional_world_mean=component['conditional_world_mean'],widths=widths))
    circles=[dict(id=g['geometry_id'],circle=dict(center=g['circle_center'],radius=g['circle_radius'],radii=g['radii'],basis=g['basis']),
        geometry={name:dict(intervals=g[name]) for name in ['allowed','forbidden']}) for g in cache.entries]
    r=dict(ordinal=0,id=p['id'],latent=p['latent'],raw_coordinates=arm['raw_coordinates'],pose=arm['pose'],log_physical_jacobian=arm['log_physical_jacobian'],
        hard_valid=True,shell_valid=True,capture_valid=True,arms=[dict(arm_index=0,log_proposal_density=arm['log_density'],old_distance_log_density=arm['old_distance_log_density'],baseline_log_density=arm['baseline_log_density'],components=components)],
        geometry_cache=dict(requests=requests,distinct_circles=len(circles),circles=circles))
    return json_safe(r),json_safe(p)


class AuditReceiptTests(unittest.TestCase):
    def test_complete_saved_comparison_and_coverage_counts(self):
        r,p=comparison_fixture();receipt=compare_rows([r],[p],1)
        self.assertTrue(receipt['complete']);self.assertEqual(receipt['counts']['width_branches'],6)
        self.assertEqual(receipt['counts']['arc_branches'],6);self.assertEqual(receipt['counts']['rust_distinct_circles'],1)

    def test_missing_component_or_wrong_fallback_fails(self):
        for alteration in ['missing','fallback']:
            r,p=comparison_fixture()
            if alteration=='missing':r['arms'][0]['components'].pop()
            else:r['arms'][0]['components'][0]['widths'][0]['arc_fallback']=True
            with self.assertRaises(ValueError):compare_rows([r],[p],1)

    def test_changed_query_or_interval_endpoint_fails(self):
        for alteration in ['identity','geometry','density']:
            r,p=comparison_fixture()
            if alteration=='identity':r['id']='other'
            elif alteration=='geometry':r['geometry_cache']['circles'][0]['geometry']['allowed']['intervals'][0]['lower']+=1e-6
            else:r['arms'][0]['log_proposal_density']+=1e-4
            with self.assertRaises(ValueError):compare_rows([r],[p],1)


if __name__=='__main__':unittest.main()
