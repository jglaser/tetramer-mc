"""Synthetic density/record tests, without physics or historical reanalysis."""
import copy
import math
import unittest
from unittest.mock import patch
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.special import logsumexp

from audit_full_vessel_latent import (VesselDensity, check_rows, log_close,
    check_generation_metadata, check_cloud_envelopes_and_counts, PrunedExclusionContact)
from physical_latent_guide import PhysicalLatentGuide, half_mixture_log_density
from prepare_deep_far_normalizer_atlas import registration
from test_physical_latent_guide import fixture, independent_pose, SHAPE, REGION


def data(alpha=.5, reciprocal=False, selected=None):
    region,guide,lower=fixture(alpha,coupled=True)
    fixed=region['fixed_neighbor'];other=copy.deepcopy(fixed);other['position'][0]+=2.
    config=dict(fixed_poses=[fixed,other],capture_center=[4.,-2.,1.],capture_radius=100.,
        metadata=dict(native_poses=[fixed],rigid_members=[dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])],
                      member_error_scale=3.,angle_error_scale_deg=90.))
    base=copy.deepcopy(region['gaussian_chart'])
    raw=dict(schema='reciprocal-pose-mixture-v1',base_model=base,reciprocal_components=[True]) if reciprocal else base
    manifest=dict(samples=3,shape_sha256=SHAPE,pose_proposal_schema=3 if reciprocal else 2 if selected is not None else 1,
        covariance_scale=1. if reciprocal else 1.3,proposal_anchor_index=selected,physical_fixed_neighbor_count=2,
        uniform_probability=.4,reciprocal_components=[True],base_component_count=1,virtual_component_count=2,
        latent_gaussian_component_count=0 if alpha==1 else 1,latent_defensive_uniform_probability=alpha,
        cloud_replicates=2,activity=.035,**{'lambda':4.48})
    vessel=VesselDensity(config,manifest,raw)
    latent=PhysicalLatentGuide(region,guide,region_sha256=REGION,expected_shape_sha256=SHAPE)
    us=[[.2,-.1,.3,.1,.2,.4],[5.,0.,0.,.3,0.,0.],[.4,.2,-.1,.2,-.1,.3]]
    poses=[independent_pose(u,region,lower)[0] for u in us]
    config['initial_pose']=copy.deepcopy(poses[0])
    vl,_=vessel.evaluate(poses);dl=latent.evaluate_many(poses)
    mix=half_mixture_log_density(vl,[d.log_physical_density for d in dl])
    qs=registration(poses,config['metadata']);rows=[]
    for i,(pose,u,d) in enumerate(zip(poses,us,dl)):
        branch='vessel' if i==2 or alpha==1 and i==1 else 'latent'
        valid=i!=2;clouds=[dict(lower_volume=2.,uncertain_volume=10.,upper_volume=12.,overlap_points=k,
            raw_points=k+3,created_cells=3,retained_cells=1,certified_cells=1,
            log_weight=.07+k*math.log1p(.035/4.48)) for k in (2,4)] if valid else []
        row=dict(draw=i,pose=pose,outer_branch=branch,proposal={} if branch=='vessel' else None,
            latent_proposal=dict(latent=u,latent_radius=float(np.linalg.norm(u)),gaussian_component=None if i==0 else 0) if branch=='latent' else None,
            latent_density=dict(latent=d.latent,in_reference_ball=d.in_reference_ball,
                coordinate_chart_seam=False,log_latent_density=d.log_latent_density if math.isfinite(d.log_latent_density) else None,
                log_physical_jacobian=d.log_physical_jacobian),
            log_vessel_proposal_density=float(vl[i]),log_latent_physical_density=d.log_physical_density if math.isfinite(d.log_physical_density) else None,
            log_proposal_density=float(mix[i]),capture_valid=True,wall_valid=True,hard_valid=valid,
            q=float(qs[i]) if valid else None,region=(('native_core' if qs[i]<=.8 else 'native_shell' if qs[i]<=1
                else 'shoulder' if qs[i]<2 else 'intermediate' if qs[i]<5 else 'distant')+'_bound') if valid else None,
            depletion_contact=True if valid else None,clouds=clouds,
            log_hard_weight=float(-mix[i]) if valid else None,
            log_importance_weight=float(logsumexp([c['log_weight'] for c in clouds])-math.log(2)-mix[i]) if valid else None)
        rows.append(row)
    _,geometry=vessel.evaluate(poses);_,initial=vessel.evaluate([config['initial_pose']])
    for i,row in enumerate(rows):
        if row['outer_branch']!='vessel':continue
        new=float(geometry['anchor_log_densities'][0,i]);old=float(initial['anchor_log_densities'][0,0])
        row['proposal']=dict(moving_index=0,anchor_index=1,branch='uniform',component_index=None,null_reason=None,
            candidate=dict(position=(np.asarray(row['pose']['position'])-config['capture_center']).tolist(),orientation=row['pose']['orientation']),
            old_log_density=old,new_log_density=new,log_reverse_forward=old-new)
    return config,manifest,rows,vessel,latent


class MixtureAuditTests(unittest.TestCase):
    def test_both_complete_laws_shifted_frame_all_or_selected_anchors(self):
        for reciprocal in (False,True):
            for selected in (None,1):
                result=check_rows(*data(reciprocal=reciprocal,selected=selected))
                self.assertEqual(result['checked_attempts'],3)
                self.assertEqual(result['valid_outside_R4'],1)
                self.assertLess(result['maximum_log_density_error'],1e-12)

    def test_uniform_latent_zero_is_covered_by_vessel(self):
        result=check_rows(*data(alpha=1.))
        self.assertEqual(result['valid_outside_R4'],1)
        log_close(None,-math.inf,'zero')
        with self.assertRaises(ValueError):log_close(None,-1000.,'lost positive density')

    def test_missing_attempt_invalid_weight_and_changed_mixture_rejected(self):
        config,manifest,rows,vessel,latent=data()
        with self.assertRaisesRegex(ValueError,'Missing attempted draw'):
            check_rows(config,manifest,rows[:2],vessel,latent)
        changed=copy.deepcopy(rows);changed[2]['log_importance_weight']=0.
        with self.assertRaisesRegex(ValueError,'explicit zero'):
            check_rows(config,manifest,changed,vessel,latent)
        changed=copy.deepcopy(rows);changed[0]['log_proposal_density']=rows[0]['log_latent_physical_density']
        with self.assertRaisesRegex(ValueError,'Complete outer mixture'):
            check_rows(config,manifest,changed,vessel,latent)

    def test_no_extra_jacobian_or_geometric_mean_cloud_estimator(self):
        config,manifest,rows,vessel,latent=data()
        changed=copy.deepcopy(rows);changed[0]['log_hard_weight']+=rows[0]['latent_density']['log_physical_jacobian']
        with self.assertRaisesRegex(ValueError,'no extra J'):check_rows(config,manifest,changed,vessel,latent)
        changed=copy.deepcopy(rows);changed[0]['log_importance_weight']=sum(c['log_weight'] for c in rows[0]['clouds'])/2-rows[0]['log_proposal_density']
        with self.assertRaisesRegex(ValueError,'Arithmetic cloud-mean'):check_rows(config,manifest,changed,vessel,latent)

    def test_gaussian_exterior_cannot_be_relabeled_uniform(self):
        config,manifest,rows,vessel,latent=data()
        rows[1]['latent_proposal']['gaussian_component']=None
        with self.assertRaisesRegex(ValueError,'Uniform latent draw left R4'):check_rows(config,manifest,rows,vessel,latent)

    def test_generation_metadata_checked_for_legacy_and_reciprocal_laws(self):
        for reciprocal in (False,True):
            for selected in (None,1):
                cfg,manifest,rows,vessel,_=data(reciprocal=reciprocal,selected=selected)
                result=check_generation_metadata(cfg,manifest,rows,vessel)
                self.assertEqual(result['checked_vessel_generation_rows'],1)
                bad=copy.deepcopy(rows);bad[2]['proposal']['anchor_index']=100
                with self.assertRaisesRegex(ValueError,'anchor label'):
                    check_generation_metadata(cfg,manifest,bad,vessel)
                bad=copy.deepcopy(rows);bad[2]['proposal']['candidate']['position'][0]+=1.
                with self.assertRaisesRegex(ValueError,'translation differs'):
                    check_generation_metadata(cfg,manifest,bad,vessel)
                bad=copy.deepcopy(rows);bad[2]['proposal']['new_log_density']+=1.
                with self.assertRaisesRegex(ValueError,'generation density differs'):
                    check_generation_metadata(cfg,manifest,bad,vessel)

    def test_primitive_counts_regions_envelope_and_all_draw_summary(self):
        _,manifest,rows,_,_=data()
        summary=dict(samples=3,hard_valid=2,capture_rejected=0,wall_rejected=0,hard_rejected=1,raw_points=24)
        self.assertEqual(check_cloud_envelopes_and_counts(manifest,summary,rows)['raw_points'],24)
        bad=copy.deepcopy(rows);bad[0]['clouds'][0]['raw_points']=1
        with self.assertRaisesRegex(ValueError,'Thinned count exceeds'):
            check_cloud_envelopes_and_counts(manifest,summary,bad)
        bad=copy.deepcopy(rows);bad[0]['clouds'][0]['upper_volume']=13.
        with self.assertRaisesRegex(ValueError,'volume decomposition'):
            check_cloud_envelopes_and_counts(manifest,summary,bad)
        bad=copy.deepcopy(rows);bad[0]['region']='diagnostic'
        with self.assertRaisesRegex(ValueError,'region label'):
            check_cloud_envelopes_and_counts(manifest,summary,bad)
        with self.assertRaisesRegex(ValueError,'denominator'):
            check_cloud_envelopes_and_counts(manifest,dict(summary,samples=2),rows)

    def test_all_anchor_law_is_normalized_average(self):
        cfg,manifest,rows,_,_=data();region,_,_=fixture(.5,coupled=True);model=region['gaussian_chart']
        poses=[r['pose'] for r in rows]
        a=VesselDensity(cfg,dict(manifest,proposal_anchor_index=0),model).evaluate(poses)[0]
        b=VesselDensity(cfg,dict(manifest,proposal_anchor_index=1),model).evaluate(poses)[0]
        both=VesselDensity(cfg,manifest,model).evaluate(poses)[0]
        np.testing.assert_allclose(both,np.logaddexp(a,b)-math.log(2),rtol=0,atol=1e-14)
        self.assertGreater(np.max(abs(both-a)),.01)



class GeometryPruningTests(unittest.TestCase):
    @staticmethod
    def pose(t,angle=0.):
        return dict(position=list(t),orientation=[math.cos(angle/2),0.,0.,math.sin(angle/2)])

    @staticmethod
    def direct(shape,moving,fixed,rd):
        centers=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
        def placed(p):
            q=p['orientation'];matrix=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
            return centers@matrix.T+p['position']
        moved=placed(moving);gaps=[]
        for p in fixed:
            distances=np.linalg.norm(moved[:,None,:]-placed(p)[None,:,:],axis=2)
            gaps.append(float(np.min(distances-radii[:,None]-radii[None,:])))
        return gaps,all(g>=0 for g in gaps),any(g<2*rd for g in gaps)

    def test_domain_invalid_skips_core_and_far_certificate_never_calls_atom_search(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.5)])
        checker=PrunedExclusionContact(shape,[self.pose([0.,0.,0.])],.2)
        with patch.object(checker.exact,'placed',side_effect=AssertionError('unnecessary atom transform')):
            for capture,wall in [(False,False),(True,False)]:
                row=checker.classify(self.pose([0.,0.,0.]),capture_valid=capture,wall_valid=wall)
                self.assertFalse(row['applicable']);self.assertIsNone(row['exclusion_contact'])
                self.assertEqual(row['method'],'not-applicable-domain-invalid')
            row=checker.classify(self.pose([4.,0.,0.]),capture_valid=True,wall_valid=True)
            self.assertTrue(row['core_disjoint']);self.assertFalse(row['exclusion_contact'])
            self.assertIsNone(row['anchors'][0]['minimum_surface_gap_A'])
            self.assertGreater(row['anchors'][0]['surface_gap_lower_bound_A'],.4)
        counts=checker.report();self.assertEqual(counts['domain_invalid_pose_skips'],2)
        self.assertEqual(counts['certified_disjoint_pose_checks'],1);self.assertEqual(counts['exact_core_contact_pose_checks'],0)

    def test_sphere_and_rotated_variable_radius_dumbbell_match_direct_predicates(self):
        shapes=[dict(atoms=[dict(center=[0.,0.,0.],radius=.5)]),
            dict(atoms=[dict(center=[-.7,0.,0.],radius=.3),dict(center=[.5,.1,0.],radius=.2)])]
        fixed=[self.pose([0.,0.,0.]),self.pose([0.,6.,0.],math.pi/2)];rd=.2
        # Deterministic translations/rotations cover overlaps, one-far/one-near
        # mixtures, all-far certificates, and the exclusion-bound tangency.
        positions=[[0.,0.,0.],[1.,0.,0.],[1.4,0.,0.],[1.4+1e-14,0.,0.],
            [0.,2.8,0.],[0.,5.,0.],[5.,5.,0.],[-4.,-3.,0.]]
        for shape in shapes:
            checker=PrunedExclusionContact(shape,fixed,rd)
            for t in positions:
                for angle in [0.,.3,math.pi/2]:
                    pose=self.pose(t,angle);gaps,hard,contact=self.direct(shape,pose,fixed,rd)
                    result=checker.classify(pose,capture_valid=True,wall_valid=True)
                    self.assertEqual(result['core_disjoint'],hard);self.assertEqual(result['exclusion_contact'],contact)
                    for entry,gap in zip(result['anchors'],gaps):
                        if entry['method']=='certified-disjoint':
                            self.assertLessEqual(entry['surface_gap_lower_bound_A'],gap)
                        else:self.assertAlmostEqual(entry['minimum_surface_gap_A'],gap,places=12)
            counts=checker.report();self.assertGreater(counts['exact_core_contact_pose_checks'],0)
            self.assertGreater(counts['certified_disjoint_pose_checks'],0)
            self.assertGreater(counts['certified_disjoint_anchor_checks'],0)
        # At and infinitesimally outside the spherical exclusion-bound
        # tangency, the guard must retain the exact check, never reject a pose.
        checker=PrunedExclusionContact(shapes[0],[fixed[0]],rd)
        for radius in [1.4-1e-14,1.4,1.4+1e-14]:
            pose=self.pose([radius,0.,0.]);result=checker.classify(pose,capture_valid=True,wall_valid=True)
            self.assertEqual(result['method'],'exact-atom-gap')
            self.assertTrue(result['core_disjoint'])
            self.assertEqual(result['exclusion_contact'],self.direct(shapes[0],pose,[fixed[0]],rd)[2])


if __name__=='__main__':unittest.main()
