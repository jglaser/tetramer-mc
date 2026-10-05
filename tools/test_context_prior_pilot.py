"""Synthetic observer/algebra tests; no protein or saved campaign queries."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analyze_context_prior_pilot import SingleMovingPatchObserver, audit_step, counts_zero, replay
from context_prior_metrics import set_ess, ab_diagnostics, persistent_neighbors, qualified_visits,contact_changes,visit_inventory,persistent_patches
from mobile_posterior_metrics import apparent_effective_count

POSE=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])


def local_step(zero=False):
    weight=math.log1p(1/64)*-100
    return dict(old_pose=POSE,proposed_pose=POSE,retained_pose=POSE,proposal=dict(branch='local',log_reverse_forward=0.),
        accepted=zero,status='accepted' if zero else 'bath_rejected',wall_valid=True,core_valid=True,
        gate=dict(gained=0,lost=100,raw_points=100,retained_points=100,log_weight=weight,
                  envelope_volume=1.,retained_cells=1,created_cells=1),
        log_proposal_reverse_forward=0.,raw_log_acceptance=weight,log_acceptance=weight,
        acceptance_uniform=0. if zero else .9,log_uniform=None if zero else math.log(.9),
        proposal_cpu_seconds=0.,geometry_cpu_seconds=0.,gate_cpu_seconds=0.)


class Metrics(unittest.TestCase):
    def test_chunked_ess_matches_dense_no_labels_dropped(self):
        sets=[{j for j in range(71) if (i+3*j)%(j%7+2)==0} for i in range(97)]
        dense=np.array([[j in row for j in range(71)] for row in sets],float)
        expected=apparent_effective_count(dense,10.)
        for size in (1,7,32):
            actual=set_ess(sets,10.,size)
            self.assertAlmostEqual(actual['apparent_ess'],expected['apparent_ess'],places=10)
            self.assertAlmostEqual(actual['iact_samples'],expected['iact_samples'],places=10)

    def test_constant_descriptors_remain_undefined(self):
        for values in ([set()]*20,[{16}]*20):self.assertIsNone(set_ess(values,3.)['apparent_ess'])

    def test_block_persistence_fft_is_not_merely_equal_floor(self):
        sets=[{j for j in range(65) if ((i+j*3)//41)%2==0} for i in range(600)]
        dense=np.array([[j in row for j in range(65)] for row in sets],float)
        expected=apparent_effective_count(dense,10.)
        self.assertGreater(expected['iact_samples'],2.)
        for size in (1,7,32):
            actual=set_ess(sets,10.,size)
            for key in ('iact_samples','unfloored_iact_samples','last_included_lag'):
                self.assertAlmostEqual(actual[key],expected[key],places=10)

    def test_ab_projection_counts_sequential_detachment_without_reweighting(self):
        neighbors=[[16,217]]*5+[[16]]*10+[[16,56]]*5+[[]]*10+[[16,217]]*5
        rows=[dict(cycle=i+1,neighbor_labels=n) for i,n in enumerate(neighbors)]
        report=ab_diagnostics(rows,rows)
        self.assertEqual(report['directional_counts'],{'A->B':1,'B->A':1})
        self.assertEqual(len(report['completed_returns']),1)
        self.assertEqual(len(report['nonoverlapping_roundtrips']),1)
        self.assertEqual(report['instantaneous_unconditional_occupancy']['Other']['count'],20)
        self.assertEqual(sum(v['fraction'] for v in report['instantaneous_unconditional_occupancy'].values()),1.)
        self.assertTrue(any(p['label']=='Other' for p in report['completed_passages'][0]['intermediate_path']))
        ordinary=persistent_neighbors(neighbors,[r['cycle'] for r in rows])
        self.assertEqual(ordinary['completed_partner_exchanges'],1) # B->A across empty; A->{16}->B is split.

    def test_short_flicker_does_not_create_confirmed_visit(self):
        labels=['A']*5+['B']*4+['A']*5
        self.assertEqual(len(qualified_visits(labels,list(range(14)))),1)
        inventory=visit_inventory(labels,list(range(14)))
        self.assertEqual(len(inventory),3);self.assertEqual(sum(v['qualifies'] for v in inventory),2)

    def test_patch_replacement_and_subset_changes_remain_distinct(self):
        a=(16,77,'a','x');b=(16,77,'b','x');c=(77,217,'y','z')
        tokens=[[a],[a,b],[b],[b,c],[b],[]]
        rows=[dict(attempt_index=i,cycle=i+1,slot=i%5,status='accepted',patch_tokens=t,
                   neighbor_labels=sorted({u if u!=77 else v for u,v,*_ in t})) for i,t in enumerate(tokens)]
        result=contact_changes(rows)
        self.assertEqual(result['consecutive_pairs'],5)
        changes={(r['label'],r['kind']):r['count'] for r in result['same_neighbor_patch_changes']}
        self.assertEqual(changes,{(16,'addition'):1,(16,'removal'):1})
        persistent=persistent_patches([dict(row,cycle=5*i+j) for i,row in enumerate(rows) for j in range(5)])
        self.assertEqual([t['kind'] for t in persistent['transitions']],['addition','removal','addition','removal','removal'])


class ObserverAndReplay(unittest.TestCase):
    def test_global_ids_and_patch_sides_are_canonical(self):
        shape=dict(atoms=[dict(center=[-1.,0.,0.],radius=.1),dict(center=[1.,0.,0.],radius=.1)])
        context=dict(bodies=[dict(label=16,pose=dict(POSE,position=[-2.3,0.,0.])),
                             dict(label=217,pose=dict(POSE,position=[2.3,0.,0.]))])
        observer=SingleMovingPatchObserver(shape,['left','right'],context,100.,.1)
        result=observer.classify(POSE)
        self.assertEqual(result['neighbor_labels'],[16,217])
        self.assertEqual(result['patch_tokens'],[(16,77,'right','left'),(77,217,'right','left')])
        calls=observer.calls
        self.assertIs(observer.classify(copy.deepcopy(POSE)),result)
        self.assertEqual(observer.calls,calls);self.assertEqual(observer.hits,1)

    def test_zero_uniform_and_rejected_residence(self):
        audit_step(local_step(True),False,0,[])
        audit_step(local_step(False),False,0,[])
        wrong=local_step(True);wrong['log_uniform']=-math.inf
        with self.assertRaises(ValueError):audit_step(wrong,False,0,[])

    def test_hard_rejections_retain_proposal_ratio_before_endpoint_geometry(self):
        for status in ('wall_rejected','core_rejected'):
            step=local_step(False)
            step.update(status=status,wall_valid=status!='wall_rejected',
                core_valid=False if status=='core_rejected' else None,gate=None,
                log_proposal_reverse_forward=0.,raw_log_acceptance=None,
                log_acceptance=None,acceptance_uniform=None,log_uniform=None)
            self.assertEqual(audit_step(step,False,0,[])['logq'],0.)
            wrong=copy.deepcopy(step);wrong['log_proposal_reverse_forward']=None
            with self.assertRaises(ValueError):audit_step(wrong,False,0,[])
        wrong=local_step(False);wrong['retained_pose']=dict(POSE,position=[.1,0.,0.])
        with self.assertRaises(ValueError):audit_step(wrong,False,0,[])

    def test_full_frozen_inventory_replay_and_missing_result(self):
        # This is a journal-only fixture; its declared fixed poses are never queried.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);identity=dict(arm='local',stream=0,start='synthetic')
            bindings=dict(config='x',shape='y',fixed_context='z',source_state='s')
            context=dict(bodies=[dict(label=i,pose=POSE) for i in range(264) if i!=77],excluded_moving_labels=[77],anchor_label=16)
            source=dict(pose=POSE,anchor_label=16,moving_label=77,boundary='spherical',
                coordinate_frame='Saved spherical-center frame; no display offset, wrapping or pose transform.')
            cfg=dict(warmup_cycles=256,local_attempts_per_cycle=4,depletant_radius=1.5,reservoir_density=.035,
                poisson_lambda_ratio=64.,translation_steps=[.2],rotation_steps_deg=[1.],rotation_probability=.5,
                uniform_probability=.5,endpoint_gate=dict(max_cells=255,max_depth=8,min_width=0.),identity=identity,initial_pose=None)
            model=dict(base_model=dict(weights=[1/1024]*1024),reciprocal_components=[True]*1024)
            counts=counts_zero();observation=dict(pose=POSE,wall_valid=True,core_overlap_labels=[],exclusion_contact_labels=[])
            event_index=0
            with (root/'events.jsonl').open('w') as stream:
                def event(kind,**kw):
                    nonlocal event_index
                    stream.write(json.dumps(dict(kind=kind,event_index=event_index,**kw))+'\n');event_index+=1
                event('invocation_begun',resume=None,certify=None,method='local')
                event('prepared',bindings=bindings,identity=identity,fixed_labels=[b['label'] for b in context['bodies']],
                    moving_label=77,moving_state_index=263,anchor_label=16,anchor_fixed_index=16)
                event('fixed_context_audit',audit=dict(physical_context_valid=True,fixed_fixed_core_overlaps=[],fixed_wall_invalid_labels=[]))
                event('initial_state',pose=POSE,counts=counts,completed_cycles=0,completed_attempts=0,resume_sha256=None,identity=identity,observation=observation)
                for cycle in range(1,2305):
                    for slot in range(5):
                        i=(cycle-1)*5+slot;step=local_step(False)
                        event('attempt_begun',cycle=cycle,slot=slot,attempt_index=i,old_pose=POSE,identity=identity,**{'global':False})
                        event('attempt_decision',cycle=cycle,slot=slot,attempt_index=i,step=step)
                        counts['attempted']+=1;counts['completed']+=1;counts['bath_rejected']+=1
                        body={k:v for k,v in step.items() if k not in ('status','proposal_cpu_seconds','geometry_cpu_seconds','gate_cpu_seconds')}
                        event('attempt_complete',cycle=cycle,slot=slot,attempt_index=i,production=cycle>256,identity=identity,
                            exclusion_contact_labels_before=[],exclusion_contact_labels=[],counts=counts,sampler_cpu_seconds=i+1.,invocation_cpu_seconds=i+2.,**body,**{'global':False})
                    event('cycle_complete',cycle=cycle,pose=POSE,counts=counts,completed_attempts=cycle*5,observation=observation,sampler_cpu_seconds=cycle*5.)
            checkpoint=dict(bindings=bindings,counts=counts,pose=POSE,method='local',correlation=0.,completed_cycles=2304,
                completed_attempts=11520,raw_points=1152000,retained_points=1152000,sample_every=2304)
            (root/'checkpoint.json').write_text(json.dumps(checkpoint))
            summary=dict(complete=True,passed=True,bindings=bindings,identity=identity,counts=counts,pose=POSE,method='local',correlation=0.,
                completed_cycles=2304,completed_attempts=11520,raw_points=1152000,retained_points=1152000,sampler_cpu_seconds=11521.,invocation_cpu_seconds=11522.,
                checkpoint_sha256=hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest())
            (root/'trajectory.jsonl').write_text(json.dumps(dict(cycle=2304,pose=POSE,counts=counts,exclusion_contact_labels=[]))+'\n')
            rows,_,_,audit=replay(root,summary,cfg,context,source,model,None)
            self.assertEqual(len(rows),11520);self.assertEqual(audit['events'],36868)
            self.assertTrue(all(r['pose']==POSE and not r['accepted'] for r in rows))
            data=(root/'events.jsonl').read_text().splitlines();del data[6]
            (root/'events.jsonl').write_text('\n'.join(data)+'\n')
            with self.assertRaises(ValueError):replay(root,summary,cfg,context,source,model,None)


if __name__=='__main__':unittest.main()
