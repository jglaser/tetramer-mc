"""Synthetic-only tests: no archived protein configurations or scientific draws."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import context_relaxed_atlas as cra
from prepare_smc_normalizer_atlas import Density


def model():
    return dict(schema='reciprocal-pose-mixture-v1', reciprocal_components=[True, False],
        base_model=dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
            angular_length=cra.REFERENCE_ELL, shape_sha256='synthetic', provenance={'native_informed': True},
            anchors=[cra.pose_record(cra.cayley([.1,-.03,.07]),[1.8,.1,0]), cra.pose_record(np.eye(3),[0,4,0])],
            means=[[.1,.2,-.1,.5,-.8,.3],[0.]*6], covariances=[np.diag([.01]*6).tolist()]*2,
            weights=[.4,.6]))


def context():
    return cra.OutsideContext([[0,0,0]], [1.], [[0,0,0]], [1.], {'kind':'synthetic'})


class ContextRelaxedTests(unittest.TestCase):
    def test_plain_input_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError,'explicit reciprocal envelope'):
            cra.build_model(model()['base_model'],None,cra.Settings(child_mass=0))

    def test_exact_reciprocal_decoding(self):
        source=model(); b=source['base_model']; records=cra.branches(source)
        self.assertEqual([(r['component'],r['inverted']) for r in records],[(0,False),(0,True),(1,False)])
        self.assertEqual([r['weight'] for r in records],[.2,.2,.6])
        ar,at=cra.pose(b['anchors'][0]); m=np.array(b['means'][0]); r=cra.cayley(m[3:]/b['angular_length'])@ar; t=at+m[:3]
        pr,pt=cra.pose(records[0]['center']); ir,it=cra.pose(records[1]['center'])
        np.testing.assert_allclose(pr,r,atol=1e-15); np.testing.assert_allclose(pt,t,atol=1e-15)
        np.testing.assert_allclose(ir,r.T,atol=1e-15); np.testing.assert_allclose(it,-r.T@t,atol=1e-15)
        np.testing.assert_allclose(ir@pr,np.eye(3),atol=1e-15)
        np.testing.assert_allclose(ir@pt+it,0,atol=1e-15)

    def test_zero_child_mass_is_exact_baseline_without_context(self):
        source=model(); events=[]
        result,report=cra.build_model(source,None,cra.Settings(child_mass=0),events.append)
        self.assertEqual(result,source); self.assertIsNot(result,source)
        self.assertFalse(report['context_used']); self.assertEqual(events,[])

    def test_all_labels_weights_flags_and_source_provenance_preserved(self):
        source=model(); before=copy.deepcopy(source); events=[]
        result,report=cra.build_model(source,context(),cra.Settings(max_pairs=0),events.append)
        self.assertEqual(source,before)
        self.assertEqual(result['reciprocal_components'],[True,False,False,False,False])
        np.testing.assert_allclose(result['base_model']['weights'],[.2,.3,.1,.1,.3],rtol=0,atol=0)
        self.assertEqual(sum(result['base_model']['weights']),1.)
        for key in ('anchors','means','covariances'):
            self.assertEqual(result['base_model'][key][:2],source['base_model'][key])
        self.assertEqual(result['base_model']['provenance'],source['base_model']['provenance'])
        self.assertFalse(report['uses_native_classifier']); self.assertTrue(report['original_model_provenance_unmodified'])
        self.assertEqual([c['parent_virtual_label'] for c in report['children']],[0,1,2])
        self.assertEqual(len(cra.branches(result)),6)
        for child in report['children']:
            k=child['child_component']
            np.testing.assert_array_equal(result['base_model']['covariances'][k],np.diag(cra.CHILD_VARIANCES))
            self.assertEqual(result['base_model']['means'][k],[0.]*6)
        self.assertEqual(len([e for e in events if e['kind']=='child_begun']),3)
        self.assertEqual(len([e for e in events if e['kind']=='child_complete']),3)

    def test_angular_covariance_units_preserve_physical_width(self):
        source=model(); source['base_model']['angular_length']*=2
        result,_=cra.build_model(source,context(),cra.Settings(max_pairs=0))
        np.testing.assert_allclose(np.diag(result['base_model']['covariances'][2]),np.r_[cra.CHILD_VARIANCES[:3],np.array(cra.CHILD_VARIANCES[3:])*4])

    def test_exported_density_is_normalized_parent_plus_child_law(self):
        source=model(); mass=.37
        result,_=cra.build_model(source,context(),cra.Settings(child_mass=mass,max_pairs=0))
        child=copy.deepcopy(result['base_model'])
        for key in ('anchors','means','covariances'): child[key]=child[key][2:]
        child['weights']=[p['weight'] for p in cra.branches(source)]
        probes=[]
        for branch in cra.branches(source):
            r,t=cra.pose(branch['center'])
            for shift in ([0.,0,0],[.03,-.02,.01]):
                # Deliberately noncommuting with the stored anchor rotations.
                rr=cra.cayley(np.array([.001,-.002,.003]))@r
                q=Rotation.from_matrix(rr).as_quat()[[3,0,1,2]]
                probes.append(dict(position=(t+shift).tolist(),orientation=q.tolist()))
        observed=Density(result).evaluate(probes)[0]
        expected=np.logaddexp(math.log1p(-mass)+Density(source).evaluate(probes)[0],
            math.log(mass)+Density(child).evaluate(probes)[0])
        self.assertTrue(np.isfinite(observed).all())
        np.testing.assert_allclose(observed,expected,rtol=0,atol=2e-13)

    def test_fixed_context_ordering_and_moving_pose_rejection(self):
        shape={'atoms':[{'center':[0,0,0],'radius':1.}]}
        records=dict(schema='fixed-outside-context-v1',anchor_label=2,excluded_moving_labels=[99],bodies=[
            {'label':7,'pose':cra.pose_record(np.eye(3),[3,0,0])},
            {'label':2,'pose':cra.pose_record(np.eye(3),[1,0,0])}])
        first=cra.OutsideContext.from_records(shape,records)
        second=cra.OutsideContext.from_records(shape,{**records,'bodies':list(reversed(records['bodies']))})
        self.assertEqual(first.identity,second.identity); np.testing.assert_array_equal(first.fixed,second.fixed)
        self.assertEqual(first.identity['outside_labels'],[2,7]); np.testing.assert_array_equal(first.fixed,[[0,0,0],[2,0,0]])
        self.assertEqual(first.maximum_fixed_center_norm,2.)
        for arr in (first.atoms,first.radii,first.fixed,first.fixed_radii): self.assertFalse(arr.flags.writeable)
        with self.assertRaisesRegex(ValueError,'fixed outside bodies only'):
            cra.OutsideContext.from_records(shape,{**records,'moving_pose':cra.pose_record(np.eye(3),[4,0,0])})
        with self.assertRaisesRegex(ValueError,'Moving labels appear'):
            cra.OutsideContext.from_records(shape,{**records,'excluded_moving_labels':[7]})
        changed=copy.deepcopy(records); changed['bodies'][0]['pose']['position'][0]=4
        self.assertNotEqual(first.identity,cra.OutsideContext.from_records(shape,changed).identity)

    def test_one_shared_tree_and_deterministic_export(self):
        real_tree=cra.cKDTree
        with patch.object(cra,'cKDTree',wraps=real_tree) as factory:
            fixed=context(); a,ra=cra.build_model(model(),fixed,cra.Settings(max_pairs=0))
            b,rb=cra.build_model(model(),fixed,cra.Settings(max_pairs=0))
            self.assertEqual(factory.call_count,1)
        self.assertEqual(a,b); self.assertEqual(ra,rb)

    def test_pair_cap_retains_original_center_and_no_objective(self):
        center=cra.pose_record(np.eye(3),[1.8,0,0]); events=[]
        problem=cra.RigidIncomingProblem(context(),center,cra.Settings(max_pairs=0),events.append)
        result=problem.solve()
        self.assertEqual(result['fallback'],'pair_cap'); self.assertEqual(result['center'],center)
        self.assertEqual(result['evaluations'],[]); self.assertEqual(events,[])

    def test_evaluation_cap_retains_original_and_completed_prefix(self):
        center=cra.pose_record(np.eye(3),[1.8,0,0]); events=[]
        result=cra.RigidIncomingProblem(context(),center,cra.Settings(max_evaluations=1),events.append).solve()
        self.assertEqual(result['fallback'],'evaluation_cap'); self.assertEqual(result['center'],center)
        self.assertEqual([e['kind'] for e in events],['objective_begun','objective_complete'])
        self.assertEqual(len(result['evaluations']),1)

    def test_gradient_matches_independent_finite_differences(self):
        fixed=cra.OutsideContext([[.7,.2,-.1],[-.3,.8,.25]],[.8,.9],
            [[1.3,-.2,.2],[-.4,1.9,.1],[20,0,0]],[.8,.8,.5],{})
        p=cra.RigidIncomingProblem(fixed,cra.pose_record(cra.cayley([.1,-.03,.02]),[.1,.05,-.1]),cra.Settings(rotation_degrees=12))
        x=np.array([.07,-.03,.02,.1,-.15,.08]); _,g=p.objective(x); numeric=[]; h=1e-6
        for k in range(6):
            xp=x.copy(); xm=x.copy(); xp[k]+=h; xm[k]-=h
            numeric.append((p.objective(xp)[0]-p.objective(xm)[0])/(2*h))
        np.testing.assert_allclose(g,numeric,rtol=2e-7,atol=2e-8)

    def test_full_box_pruning_matches_all_pairs_even_outside_trust_balls(self):
        fixed=cra.OutsideContext([[0,0,0],[2,.2,0]],[.5,.6],
            [[1,1,1],[2.9,1,1],[-1.7,-1.7,-1.7],[100,0,0]],[.5,.7,.3,.2],{})
        p=cra.RigidIncomingProblem(fixed,cra.pose_record(np.eye(3),[0,0,0]),cra.Settings(rotation_degrees=20))
        self.assertLess(len(p.mi),len(fixed.atoms)*len(fixed.fixed))
        for x in [np.zeros(6),np.ones(6),-np.ones(6),np.array([1,-1,1,-1,1,-1]),np.array([.2,.1,-.3,.4,-.1,.5])]:
            points=p.geometry(x)[0]
            distances=np.linalg.norm(points[:,None,:]-fixed.fixed[None,:,:],axis=2)
            all_gaps=distances-fixed.radii[:,None]-fixed.fixed_radii[None,:]
            full=float(np.maximum(p.margin-all_gaps,0).ravel()@np.maximum(p.margin-all_gaps,0).ravel())
            self.assertAlmostEqual(p.objective(x)[0],full,places=13)

    def test_motion_bound_uses_actual_rotated_vectors_at_rotation_tolerance(self):
        # The admitted matrix stretches a long lever slightly. Nominal atom
        # norms would underbound its rotational displacement even if R is
        # sufficiently close to orthogonal to pass pose validation.
        r=(1+2e-11)*np.eye(3); fixed=cra.OutsideContext([[1e8,0,0]],[1.],[[0,0,0]],[1.],{})
        settings=cra.Settings(rotation_degrees=20)
        p=cra.RigidIncomingProblem(fixed,cra.pose_record(r,[0,0,0]),settings)
        sine=math.sin(math.sqrt(3)*math.radians(settings.rotation_degrees)/2)
        old=math.sqrt(3)+2*np.linalg.norm(fixed.atoms,axis=1)*sine
        expected=math.sqrt(3)+2*np.linalg.norm(p.base,axis=1)*sine
        np.testing.assert_array_equal(p.full_box_motion_bounds,expected)
        self.assertGreater(p.full_box_motion_bounds[0],old[0])

    def test_synthetic_relaxation_improves_core_objective(self):
        p=cra.RigidIncomingProblem(context(),cra.pose_record(np.eye(3),[1.8,0,0]),cra.Settings())
        result=p.solve()
        self.assertIsNone(result['fallback']); self.assertTrue(result['optimizer']['success'])
        self.assertLess(result['selected_objective_A2'],1e-20)
        self.assertGreaterEqual(np.linalg.norm(result['center']['position'])-2,.001999)

    def test_unexpected_failure_preserves_begun_record(self):
        events=[]; p=cra.RigidIncomingProblem(context(),cra.pose_record(np.eye(3),[1.8,0,0]),cra.Settings(),events.append)
        with patch.object(p,'geometry',side_effect=ArithmeticError('synthetic failure')):
            with self.assertRaises(ArithmeticError): p.objective(np.zeros(6))
        self.assertEqual([e['kind'] for e in events],['objective_begun'])
        self.assertEqual(p.evaluations,[])


if __name__=='__main__': unittest.main()
