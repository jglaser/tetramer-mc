"""Synthetic chart, residence, fitting and heldout controls; no protein queries."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from analyze_competing_cage_covariance import (coordinates,fit_gaussian,likelihood_diagnostics,
    movement_diagnostics,retained_states,validate_center,validate_cage_inventory,validate_patch_reference)


def pose(p,r=None):
    if r is None:r=np.eye(3)
    q=Rotation.from_matrix(r).as_quat()
    return dict(position=list(p),orientation=q[[3,0,1,2]].tolist())


def design():
    a=np.arange(48,dtype=float).reshape(8,6)/100
    a[:6]+=np.eye(6)*.1
    return a


class CompetingCageTests(unittest.TestCase):
    def test_actual_nested_source_guide_reference_schema(self):
        patch=dict(path='/frozen/patch_map.json',sha256='a'*64)
        regions=dict(a_neighbors=[16,217],b_neighbors=[16,56],source_secondary_tokens=[])
        cfg=dict(schema='context-source-guide-v1',inputs=dict(regions=regions,patch_map=patch))
        reference=dict(definitions=regions,patch_map=patch,unchanged_during_training=True)
        validate_patch_reference(cfg,reference)
        changed=copy.deepcopy(reference);changed['definitions']['a_neighbors']=[16]
        with self.assertRaises(ValueError):validate_patch_reference(cfg,changed)
        with self.assertRaises(KeyError):validate_patch_reference(dict(regions=regions,patch_map=patch),reference)

    def test_exact_inventory_and_independent_stream_labels(self):
        cages=[dict(cage_id=i,initial_pose=pose([float(i),0.,0.]),
            streams=[dict(stream=s,seed=4*i+s) for s in range(4)]) for i in range(2)]
        validate_cage_inventory(cages)
        for change in ('seed','identity','pose'):
            bad=copy.deepcopy(cages)
            if change=='seed':bad[1]['streams'][0]['seed']=0
            if change=='identity':bad[1]['cage_id']='../escape'
            if change=='pose':bad[1]['initial_pose']=bad[0]['initial_pose']
            with self.assertRaises(ValueError):validate_cage_inventory(bad)

    def test_chart_center_is_distinct_from_initial_and_original_reference(self):
        anchor=pose([1.,2.,3.],Rotation.from_rotvec([.3,-.2,.1]).as_matrix())
        original=pose([-2.,1.,4.]);initial=pose([4.,-1.,2.],Rotation.from_rotvec([-.2,.1,.4]).as_matrix())
        center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',pose=initial,provenance='frozen seed')
        validate_center(center);x,_=coordinates([initial],initial,anchor,3.)
        np.testing.assert_allclose(x,0,atol=1e-14)
        old,_=coordinates([initial],original,anchor,3.);self.assertGreater(np.linalg.norm(old),1.)
        bad=copy.deepcopy(center);bad['frame']='display-origin'
        with self.assertRaises(ValueError):validate_center(bad)
        self.assertNotEqual(original,center['pose'])

    def test_noncommuting_common_frame_and_chart_displacement(self):
        ra=Rotation.from_rotvec([.2,-.3,.4]).as_matrix();rc=Rotation.from_rotvec([-.1,.5,-.2]).as_matrix()
        anchor=pose([1.,2.,3.],ra);center=pose([3.,2.,1.],ra@rc)
        u=np.array([.02,-.03,.01]);dq=Rotation.from_quat(np.r_[u,1.]/np.linalg.norm(np.r_[u,1.])).as_matrix()
        change=np.array([.1,-.2,.3]);sample=pose(np.asarray(center['position'])+ra@change,ra@dq@rc)
        x,_=coordinates([sample],center,anchor,5.)
        np.testing.assert_allclose(x[0],np.r_[change,5*u],atol=2e-14)

    def test_rejected_residence_changes_fit_and_ridge_is_explicit(self):
        x=np.zeros((4,6));x[-1,0]=4.;f=fit_gaussian(x,np.ones(6))
        self.assertEqual(f['mean'][0],1.)
        self.assertEqual(f['unregularized_covariance'][0][0],3.)
        self.assertEqual(f['empirical_rank'],1);self.assertEqual(f['fit_status'],'rank_deficient_regularized')
        self.assertAlmostEqual(f['scaled_ridge'],.5e-6)
        self.assertGreater(np.linalg.eigvalsh(f['covariance'])[0],0.)
        self.assertNotEqual(f['mean'],fit_gaussian(np.unique(x,axis=0),np.ones(6))['mean'])

    def test_zero_motion_is_not_identified_by_positive_ridge(self):
        x=np.zeros((100,6));f=fit_gaussian(x,np.ones(6))
        self.assertTrue(f['normalized_gaussian']);self.assertEqual(f['empirical_rank'],0)
        self.assertEqual(f['fit_status'],'no_observed_motion_ridge_only')
        self.assertEqual(f['ridge_fraction_of_regularized_trace'],1.)
        rows=[dict(accepted=False) for _ in x];d=movement_diagnostics(x,rows,2.,np.ones(6))
        self.assertEqual(d['unique_retained_coordinate_states'],1);self.assertEqual(d['unique_accepted_coordinate_states'],0)
        self.assertIsNone(d['scaled_vector_autocorrelation']['apparent_ess'])
        self.assertTrue(all(r['apparent_ess'] is None for r in d['coordinate_autocorrelation']))

    def test_heldout_evaluation_never_changes_fit(self):
        x=design();f=fit_gaussian(x,np.ones(6));before=json.dumps(f,sort_keys=True)
        train=likelihood_diagnostics(x,f,3.)
        heldout=likelihood_diagnostics(x+5.,f,3.)
        self.assertEqual(json.dumps(f,sort_keys=True),before)
        self.assertGreater(train['mean_log_density'],heldout['mean_log_density'])
        self.assertGreater(heldout['mean_squared_mahalanobis'],train['mean_squared_mahalanobis'])
        self.assertEqual(heldout['nominal_gaussian_coverage'][-1]['count'],0)

    def test_haar_jacobian_once_and_coverage_denominator(self):
        x=design();f=fit_gaussian(x,np.ones(6));ell=3.
        r=likelihood_diagnostics(x,f,ell)
        c=x[:,3:]/ell;logj=-3*np.log(ell)-2*np.log(np.pi)-2*np.log1p(np.sum(c*c,axis=1))
        expected=multivariate_normal.logpdf(x,mean=f['mean'],cov=f['covariance'])-logj
        self.assertAlmostEqual(r['mean_log_density'],float(expected.mean()),places=10)
        for b in r['nominal_gaussian_coverage']:
            self.assertEqual(b['fraction'],b['count']/len(x))

    def test_persistent_trace_has_nontrivial_autocorrelation(self):
        y=np.repeat([0.,1.,-1.,.5],64);x=np.tile(y[:,None],(1,6))
        rows=[dict(accepted=i%64==0) for i in range(len(x))]
        d=movement_diagnostics(x,rows,2.,np.ones(6))
        self.assertGreater(d['scaled_vector_autocorrelation']['iact_samples'],2)
        self.assertLess(d['scaled_vector_autocorrelation']['apparent_ess'],len(x)/2)
        self.assertEqual(d['unique_accepted_coordinate_states'],4)

    def test_competing_initial_state_and_all_rejections_are_preserved(self):
        initial=pose([3.,2.,1.]);original=pose([0.,0.,0.]);identity={'stream':0};events=[]
        def add(kind,**kwargs):events.append(dict(event_index=len(events),kind=kind,**kwargs))
        for kind in ('invocation_begun','prepared','fixed_context_audit'):add(kind)
        add('initial_state',pose=initial,identity=identity)
        for cycle in (1,2):
            for slot in range(5):
                add('attempt_begun');add('attempt_decision')
                add('attempt_complete',attempt_index=(cycle-1)*5+slot,cycle=cycle,slot=slot,
                    identity=identity,production=cycle>1,old_pose=initial,proposed_pose=original,
                    retained_pose=initial,accepted=False,exclusion_contact_labels=[16,56],**{'global':False})
            add('cycle_complete',cycle=cycle,pose=initial)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'events.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in events))
            rows,audit=retained_states(p,identity,initial,cycles=2,warmup=1)
            self.assertEqual(audit['rejected_states'],10);self.assertEqual(len(rows),10)
            with self.assertRaises(ValueError):retained_states(p,identity,original,cycles=2,warmup=1)

if __name__=='__main__':unittest.main()
