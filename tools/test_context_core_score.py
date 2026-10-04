import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from context_relaxed_atlas import OutsideContext
from context_core_score import ExactPointScorer,evaluate_records,pose_key,summarize


def center(t=(0.,0.,0.),rotvec=(0.,0.,0.)):
    return dict(position=list(t),rotation=Rotation.from_rotvec(rotvec).as_matrix().tolist())


def context(atoms,radii,fixed,fixed_radii):
    return OutsideContext(atoms,radii,fixed,fixed_radii,{})


def brute(ctx,pose,margin=.002):
    r=np.asarray(pose['rotation']);t=np.asarray(pose['position'])
    core=active=within=coincident=0;value=0.;g=np.zeros(6);minimum=None
    for a,ra in zip(ctx.atoms,ctx.radii):
        offset=r@a;p=offset+t
        for b,rb in zip(ctx.fixed,ctx.fixed_radii):
            delta=p-b;d=math.dist(p,b);rs=float(ra+rb);gap=d-rs;h=max(margin-gap,0.)
            core+=sum(float(v)*float(v) for v in delta)<rs*rs
            if d<=rs+margin:
                within+=1;minimum=gap if minimum is None else min(minimum,gap)
            if h>0:
                active+=1;coincident+=d==0;value+=h*h
                if d>0:
                    f=-2*h*delta/d;g[:3]+=f;g[3:]+=np.cross(offset,f)
    return dict(core_overlap_pairs=int(core),active_hinge_pairs=active,within_individual_margin_pairs=within,
        coincident_active_pairs=coincident,objective_A2=value,minimum_local_gap_A=minimum,gradient=g)


class PointwiseTests(unittest.TestCase):
    def assert_reference(self,ctx,p,**kwargs):
        got=ExactPointScorer(ctx,**kwargs).score(p);ref=brute(ctx,p,kwargs.get('margin',.002))
        self.assertTrue(got['complete_geometry'])
        for key in ('core_overlap_pairs','active_hinge_pairs','within_individual_margin_pairs','coincident_active_pairs'):
            self.assertEqual(got[key],ref[key])
        self.assertAlmostEqual(got['objective_A2'],ref['objective_A2'],places=11)
        if ref['minimum_local_gap_A'] is None:self.assertIsNone(got['minimum_local_gap_A'])
        else:self.assertAlmostEqual(got['minimum_local_gap_A'],ref['minimum_local_gap_A'],places=12)
        np.testing.assert_allclose(got['gradient_translation_A']+got['gradient_left_rotation_A2_per_rad'],ref['gradient'],atol=2e-11,rtol=2e-11)
        return got

    def test_analytic_two_spheres_overlap_and_gradient(self):
        ctx=context([[0,0,0]],[1],[[1.5,0,0]],[1]);got=self.assert_reference(ctx,center())
        self.assertAlmostEqual(got['objective_A2'],.502**2)
        self.assertEqual(got['core_overlap_pairs'],1)
        np.testing.assert_allclose(got['gradient_translation_A'],[1.004,0,0])

    def test_unequal_radius_filter(self):
        ctx=context([[0,0,0]],[1],[[3,0,0],[20,0,0]],[.1,3])
        got=self.assert_reference(ctx,center())
        self.assertEqual(got['candidates_materialized'],1)
        self.assertEqual(got['within_individual_margin_pairs'],0)
        self.assertTrue(got['minimum_gap_censored']);self.assertEqual(got['minimum_gap_censor_threshold_A'],.002)

    def test_empty_ray_no_infinite_margin_claim(self):
        ctx=context([[0,0,0]],[1],[[100,0,0]],[1]);got=self.assert_reference(ctx,center())
        self.assertTrue(got['hard_clear']);self.assertIsNone(got['minimum_local_gap_A'])
        self.assertTrue(got['minimum_gap_censored']);self.assertEqual(got['objective_A2'],0)

    def test_tangent_and_margin_boundaries(self):
        for d in [2.,np.nextafter(2.,0.),np.nextafter(2.,3.),2.002,np.nextafter(2.002,0.),np.nextafter(2.002,3.)]:
            with self.subTest(d=d):self.assert_reference(context([[0,0,0]],[1],[[d,0,0]],[1]),center())

    def test_coincidence_explicit(self):
        got=self.assert_reference(context([[0,0,0]],[1],[[0,0,0]],[1]),center())
        self.assertEqual(got['coincident_active_pairs'],1)
        self.assertEqual(got['gradient_translation_A'],[0.,0.,0.])

    def test_rotated_dumbbell_complete_over_synthetic_poses(self):
        ctx=context([[-1,0,0],[1,.2,.1]],[.8,.4],[[.2,.8,.3],[2,-.4,0],[-1.6,.1,.2]],[.6,.5,.8])
        rng=np.random.default_rng(76541)
        for _ in range(20):self.assert_reference(ctx,center(rng.uniform(-.4,.4,3),rng.uniform(-.8,.8,3)),batch_size=1)

    def test_left_rotation_and_translation_finite_difference(self):
        ctx=context([[-1,.2,.1],[.6,-.3,.7]],[.8,.6],[[.2,.8,.3],[1.1,-.4,.2],[-1.6,.1,.2]],[.6,.5,.8])
        p=center([.12,-.07,.04],[.4,-.2,.3]);score=ExactPointScorer(ctx)
        base=score.score(p);g=np.r_[base['gradient_translation_A'],base['gradient_left_rotation_A2_per_rad']]
        eps=1e-6;fd=[]
        for i in range(6):
            pairs=[]
            for sign in (1,-1):
                q=copy.deepcopy(p)
                if i<3:q['position'][i]+=sign*eps
                else:
                    v=np.zeros(3);v[i-3]=sign*eps
                    q['rotation']=(Rotation.from_rotvec(v).as_matrix()@np.asarray(p['rotation'])).tolist()
                pairs.append(score.score(q)['objective_A2'])
            fd.append((pairs[0]-pairs[1])/(2*eps))
        np.testing.assert_allclose(g,fd,atol=3e-8,rtol=3e-7)

    def test_batch_sizes_same_geometry(self):
        ctx=context([[0,0,0],[2,0,0],[4,0,0]],[1,.7,.3],[[.5,0,0],[2.4,0,0],[6,0,0]],[.5,.6,.2])
        for batch in (1,2,64):self.assert_reference(ctx,center(),batch_size=batch)

    def test_candidate_cap_before_materialization(self):
        ctx=context([[0,0,0]],[1],[[.1,0,0],[.2,0,0]],[1,1])
        got=ExactPointScorer(ctx,max_candidates=1).score(center())
        self.assertEqual(got['status'],'candidate_cap');self.assertFalse(got['complete_geometry'])
        self.assertEqual(got['candidates_materialized'],0);self.assertEqual(got['list_query_atom_points'],0)
        self.assertEqual(got['cap_batch']['batch_candidates'],2);self.assertIsNone(got['hard_clear'])
        self.assertIsNone(got['objective_A2']);self.assertFalse(got['minimum_gap_censored'])

    def test_cap_after_prefix_retains_partial_separately(self):
        ctx=context([[0,0,0],[3,0,0]],[1,1],[[.1,0,0],[3.1,0,0],[3.2,0,0]],[1,1,1])
        got=ExactPointScorer(ctx,max_candidates=1,batch_size=1).score(center())
        self.assertEqual(got['status'],'candidate_cap');self.assertEqual(got['atoms_processed'],1)
        self.assertGreater(got['partial_values']['objective_A2'],0);self.assertIsNone(got['objective_A2'])

    def test_identity_cache_preserves_all_weights_and_caps(self):
        p=center();records=[dict(identity='parent-0000',arm='parent',parent_virtual_label=0,source_weight=1.,center=p,cache_of=None),
            dict(identity='child-0000',arm='child',parent_virtual_label=0,source_weight=1.,center=p,cache_of='parent-0000')]
        class Score:
            calls=0
            def score(self,p):
                self.calls+=1
                return ExactPointScorer(context([[0,0,0]],[1],[[0,0,0]],[1]),max_candidates=0).score(p)
        score=Score();events=[];result=evaluate_records(records,score,events.append)
        self.assertEqual(score.calls,1);self.assertEqual(len(result),2);self.assertEqual(len(events),4)
        self.assertEqual(result[0]['score'],result[1]['score']);summary=summarize(result)
        self.assertEqual(summary['actual_pose_queries'],1);self.assertEqual(summary['cache_reuses'],1)
        self.assertEqual(summary['arms']['child']['capped']['source_probability'],1.)

    def test_failure_preserves_completed_parent_and_begun_child(self):
        records=[dict(identity='parent-0000',arm='parent',parent_virtual_label=0,source_weight=1.,center=center(),cache_of=None),
            dict(identity='child-0000',arm='child',parent_virtual_label=0,source_weight=1.,center=center([1,0,0]),cache_of=None)]
        class Score:
            calls=0
            def score(self,p):
                self.calls+=1
                if self.calls==2:raise RuntimeError('deliberate failure')
                return {'complete_geometry':True}
        events=[]
        with self.assertRaisesRegex(RuntimeError,'deliberate'):evaluate_records(records,Score(),events.append)
        self.assertEqual([e['kind'] for e in events],['pose_begun','pose_complete','pose_begun'])
        self.assertEqual(events[-1]['identity'],'child-0000')

    def test_pose_cache_is_exact_no_tolerance(self):
        p=center();q=copy.deepcopy(p);q['position'][0]=1e-20
        self.assertNotEqual(pose_key(p),pose_key(q))

    def test_context_anchor_frame_against_scipy_reference(self):
        atoms=np.array([[-1,.2,.1],[.7,.1,-.2]]);radii=[.3,.4]
        shape=dict(atoms=[dict(center=a.tolist(),radius=r) for a,r in zip(atoms,radii)])
        ar=Rotation.from_rotvec([.5,-.2,.1]);br=Rotation.from_rotvec([-.1,.4,.3]);at=np.array([10.,2.,-1.]);bt=np.array([12.,4.,1.])
        def q(rot):return np.roll(rot.as_quat(),1).tolist()
        data=dict(schema='fixed-outside-context-v1',anchor_label=7,excluded_moving_labels=[9],bodies=[
            dict(label=7,pose=dict(position=at.tolist(),orientation=q(ar))),
            dict(label=8,pose=dict(position=bt.tolist(),orientation=q(br)))])
        got=OutsideContext.from_records(shape,data)
        ref=np.concatenate([atoms,ar.inv().apply(br.apply(atoms)+bt-at)])
        np.testing.assert_allclose(got.fixed,ref,atol=3e-14,rtol=3e-14)

if __name__=='__main__':unittest.main()
