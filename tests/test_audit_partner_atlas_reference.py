"""Synthetic saved-record mutations only; never open scientific journals."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import audit_partner_atlas_reference as audit

META=dict(activity=.5, **{'lambda':2.}, point_volume=.5, points_per_body=8,
          config=dict(translation_std=.25,rotation_std_degrees=30.),cube=[3.,3.,3.],center=[0.,0.,0.],
          branches=3,per_leg_cap=1000,per_outer_cap=1000)


def pose(x): return dict(position=[x,0.,0.],orientation=[1.,0.,0.,0.])


def score(internal,strength):
    units=2+internal
    return dict(points_per_body=8,spectator_covered=[1,0],internal_unshielded=[internal,0],
                nearby_spectators=[1,1],spectator_membership_queries=16,internal_membership_queries=15,
                twice_overlap_units=units,overlap_volume_estimate=.25*units,log_surrogate=.5*strength*.25*units)


def gate(gained=0,lost=0):
    active=gained+lost>0
    return dict(gained=gained,lost=lost,raw_points=gained+lost+2 if active else 0,
                retained_points=gained+lost,retained_cells=2 if active else 0,created_cells=3 if active else 0,
                envelope_volume=1. if active else 0.,log_weight=(gained-lost)*math.log1p(.25))


def learned(candidate,q):
    old_g=-3.; new_g=old_g-q; q=old_g-new_g
    return dict(branch='involution',charts='members',source_law='posterior',handle=deepcopy(candidate),identity=False,
                labels=dict(source=dict(member=0,anchor=0,branch=0),target=dict(member=0,anchor=0,branch=1)),
                full_old_member_log_density=old_g,full_new_member_log_density=new_g,
                log_reverse_forward=q,label_log_reverse_forward=0.,expanded_log_reverse_forward=q,
                step=dict(pose=pose(.1),inverse_trace=dict(source=1,target=0,noise=[0.]*6),
                          source_latent=[0.]*6,target_latent=[0.]*6,
                          log_extended_jacobian=q-.125,log_auxiliary_ratio=.125,log_correction=q))


def fixture(arm='m1_guided',mode='partner_atlas',null=False):
    direct=arm=='direct'; horizon=audit.HORIZONS[audit.ARMS.index(arm)]; strength=0. if arm=='flat8' else 1.
    old=[pose(-.5),pose(.5)]; current=deepcopy(old); current_score=score(2,strength)
    record=dict(kind='flexible_partner_atlas_direct' if direct else 'flexible_partner_atlas_chain',
                inner_filter=not direct,old=deepcopy(old),members=[0,1],selection_probabilities=[.5,.5],
                mode_probabilities=dict(partner_atlas=.25,local=.75),
                config=dict(META['config'],inner_steps=horizon,guidance_strength=strength),
                steps=[],budget_before=dict(raw=0,retained=0))
    counts=dict(attempted=horizon,hard_rejected=0,null_proposals=0,zero_reverse_support=0)
    if direct: counts['eligible']=0
    else: counts.update(accepted=0,mh_rejected=0); record['old_score']=deepcopy(current_score)
    proposal_q=0.
    for index in range(horizon):
        slot=index%2; target=deepcopy(current); target[slot]['position'][0]+=.125
        first=index==0
        trace=dict(index=index,old=deepcopy(current),selected_slot=slot,selected_label=slot,
                   proposed=target,accepted=False,mode=mode if first else 'local')
        if not direct: trace['old_score']=current_score['log_surrogate']
        if first and mode=='partner_atlas':
            trace.update(partner_slot=1-slot,partner_label=1-slot,partner_pose=deepcopy(current[1-slot]))
            if null:
                trace['proposal_trace']=dict(branch='involution',charts='members',null_reason='synthetic exact seam')
                trace['proposed']=deepcopy(current)
            else:
                trace['proposal_trace']=learned(target[slot],-.6 if direct else .4)
                proposal_q=trace['proposal_trace']['log_reverse_forward']
                trace['proposal_log_reverse_forward']=proposal_q
        else: trace['proposal_log_reverse_forward']=0.
        if first and null:
            trace['status']='null_proposal'; counts['null_proposals']+=1
        elif not first:
            trace['status']='hard_rejected'; counts['hard_rejected']+=1
        elif direct:
            trace.update(status='direct_candidate',accepted=None); counts['eligible']+=1; current=target
        else:
            proposed_score=score(0,strength); delta=proposed_score['log_surrogate']-current_score['log_surrogate']+proposal_q
            log_u=-1.; accepted=log_u<min(0.,delta)
            trace.update(status='completed',proposed_score=proposed_score,log_acceptance_ratio=delta,
                         log_u=log_u,accepted=accepted)
            counts['accepted' if accepted else 'mh_rejected']+=1
            if accepted: current,current_score=target,proposed_score
        trace['retained']=deepcopy(current)
        if not direct: trace['retained_score']=current_score['log_surrogate']
        record['steps'].append(trace)
    record['proposal_counts' if direct else 'inner_counts']=counts
    record['proposed']=deepcopy(current)
    if not direct: record['proposed_score']=deepcopy(current_score)
    identity=current==old
    correction=(0. if identity else proposal_q) if direct else record['old_score']['log_surrogate']-current_score['log_surrogate']
    record['complete_log_correction']=correction
    if identity:
        record.update(status='identity_self_loop',physical_decisions=0,accepted=False,budget_after=dict(raw=0,retained=0))
    else:
        legs=[gate(2 if direct else 1),gate()]
        aggregate={k:legs[0][k]+legs[1][k] for k in legs[0]}
        ratio=aggregate['log_weight']+correction; log_u=-.1
        record.update(status='completed',physical_decisions=1,accepted=log_u<min(0.,ratio),
                      log_acceptance_ratio=ratio,log_u=log_u,budget_after=dict(raw=aggregate['raw_points'],retained=aggregate['retained_points']),
                      bath=dict(order='first_then_second',ordered_members=[0,1],intermediate_selected=[current[0],old[1]],
                                legs=legs,aggregate=aggregate))
    retained=deepcopy(current if record['accepted'] else old); negative=None; drift=None
    if direct:
        negative={}; delta=0.
        if not identity and mode=='partner_atlas':
            info=record['steps'][0]['proposal_trace']; delta=(math.atan(info['full_new_member_log_density'])-math.atan(info['full_old_member_log_density']))/math.pi
        drift=[delta if record['accepted'] else 0.]
        for name in ('omitted','wrong_sign'):
            take=False if identity else record['log_u']<min(0.,record['bath']['aggregate']['log_weight']-(0. if name=='omitted' else proposal_q))
            negative[name]=dict(accepted=take,retained=deepcopy(current if take else old)); drift.append(delta if take else 0.)
    return record,old,retained,arm,deepcopy(META),negative,drift


def write(path,values): path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values))


def small_journal():
    sources={}; rows=[]; budget=dict(raw=0,retained=0)
    for index in range(2):
        for arm in audit.ARMS:
            record,old,retained,_,_,negative,drift=fixture(arm,mode='local' if index==1 else 'partner_atlas')
            sources[0,index]=dict(kind='cached_iid_source',stream=0,source_index=index,source_attempt=index+1,old=old)
            for key in budget: record['budget_after'][key]+=budget[key]
            record['budget_before']=deepcopy(budget); budget=record['budget_after']
            before=[0.]*38; before[0]=-.5; before[10]=1.
            after=deepcopy(before)
            if retained!=old: after[0]+=.125; after[10]=1.125
            ids=dict(stream=0,source_index=index,source_attempt=index+1,arm=arm)
            rows += [dict(kind='kernel_begin',old=old,**ids),
                     dict(kind='kernel_outcome',record=record,retained=retained,negative_controls=negative,
                          bounded_density_drift=drift,kernel_error=None,audit_error=None,
                          source_observables=before,retained_observables=after,
                          source_extra_observables=[0.]*21,retained_extra_observables=[0.]*21,
                          candidate_internal_lens=1.125,**ids)]
    return sources,rows


class AuditTests(unittest.TestCase):
    def test_direct_and_all_delayed_arms(self):
        for arm in audit.ARMS:
            work,_,drifts=audit.audit_record(*fixture(arm))
            self.assertEqual(work['candidate_attempts'],audit.HORIZONS[audit.ARMS.index(arm)])
            self.assertEqual(work['learned_physical_decisions'],1)
            self.assertEqual(drifts is not None,arm=='direct')

    def test_inner_omission_and_outer_double_q_rejected(self):
        args=list(fixture()); args[0]['steps'][0]['log_acceptance_ratio']-=args[0]['steps'][0]['proposal_log_reverse_forward']
        with self.assertRaises(ValueError): audit.audit_record(*args)
        args=list(fixture('direct')); q=args[0]['steps'][0]['proposal_log_reverse_forward']
        args[0]['complete_log_correction']+=q; args[0]['log_acceptance_ratio']+=q
        with self.assertRaises(ValueError): audit.audit_record(*args)

    def test_current_partner_locality_and_complete_density(self):
        for mutate in (
            lambda r:r['steps'][0]['partner_pose']['position'].__setitem__(0,100.),
            lambda r:r['steps'][0]['proposed'][1]['position'].__setitem__(0,100.),
            lambda r:r['steps'][0]['proposal_trace'].__setitem__('full_new_member_log_density',-10.),
            lambda r:r['steps'][0]['proposal_trace']['labels']['source'].__setitem__('member',False),
            lambda r:r['steps'][0]['proposal_trace']['labels']['target'].__setitem__('anchor',False),
            lambda r:r['steps'][0].__setitem__('partner_slot',True),
        ):
            args=list(fixture()); mutate(args[0])
            with self.assertRaises(ValueError): audit.audit_record(*args)

    def test_horizon_retention_and_bath_mutations(self):
        for mutate in (
            lambda r:r['steps'].pop(),
            lambda r:r['steps'][0]['retained'][0]['position'].__setitem__(0,100.),
            lambda r:r['bath']['aggregate'].__setitem__('gained',100),
            lambda r:r['inner_counts'].__setitem__('null_proposals',False),
            lambda r:r['members'].__setitem__(0,False),
        ):
            args=list(fixture('m8_guided')); mutate(args[0])
            with self.assertRaises(ValueError): audit.audit_record(*args)

    def test_nulls_preserve_all_residence_and_no_bath(self):
        for arm in audit.ARMS:
            args=fixture(arm,null=True); work,_,_=audit.audit_record(*args)
            self.assertEqual(work['identities'],1); self.assertEqual(work['null_proposals'],1)
            self.assertEqual(work['raw_bath_points'],0)
            args[0]['physical_decisions']=False
            with self.assertRaises(ValueError): audit.audit_record(*args)

    def test_negative_controls_and_zero_rows_are_retained(self):
        args=list(fixture('direct')); args[5]['omitted']['accepted']=False
        with self.assertRaises(ValueError): audit.audit_record(*args)
        args=list(fixture('direct')); args[5]['omitted']['accepted']=1
        with self.assertRaises(ValueError): audit.audit_record(*args)
        args=list(fixture('direct')); args[0]['proposal_counts']['null_proposals']=False
        with self.assertRaises(ValueError): audit.audit_record(*args)
        with tempfile.TemporaryDirectory() as tmp:
            sources,records=small_journal(); path=Path(tmp)/'journal.jsonl'; write(path,records)
            result=audit.reduce_journal(path,sources,1,2,META)
            self.assertEqual(result['calls'],8)
            self.assertEqual(sum(w['candidate_attempts'] for w in result['work'][0]),36)
            self.assertTrue(all(m.n==2 for m in result['total_drift']))
            self.assertTrue(all(m.n==2 for arm in result['total'] for m in arm))
            moments=result['total_drift'][1].report(); moments['n']=1
            with self.assertRaises(ValueError): audit.compare(moments,result['total_drift'][1].report())

    def test_source_filtering_or_reordering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources,records=small_journal(); root=Path(tmp); cache=root/'sources.jsonl'; journal=root/'journal.jsonl'
            write(cache,list(sources.values())); self.assertEqual(audit.read_sources(cache,1,2),sources)
            write(cache,[sources[0,1]])
            with self.assertRaises(ValueError): audit.read_sources(cache,1,2)
            write(journal,records[:-2])
            with self.assertRaises(ValueError): audit.reduce_journal(journal,sources,1,2,META)
            records[0]['source_attempt']=999; write(journal,records)
            with self.assertRaises(ValueError): audit.reduce_journal(journal,sources,1,2,META)

    def test_incomplete_runner_rejected_before_reference_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'validation.json'
            p.write_text(json.dumps(dict(schema='partner-atlas-reference-validation-v1',complete=False,
                                        child_started=True,child_drained=False,error=None)))
            with self.assertRaises(ValueError): audit.audit_completed(root,audit.sha(p))

    def test_complete_statistical_failure_is_auditable_but_killed_process_is_not(self):
        for passed in (False,True):
            code=0 if passed else 101
            validation=dict(numerical_allocation_complete=True,receipt_bindings_valid=True,
                            scientific_receipt_passed=passed,passed=passed,returncode=code,argv=['frozen-test'],
                            numerical_process=dict(returncode=code,error=None,timeout=False,child_started=True,
                                                   child_drained=True,argv=['frozen-test'],cleanup=dict(
                                                       child_drained=True,leader_reaped=True,process_group_absent=True)),
                            archive_before={'source':'hash'},archive_after={'source':'hash'},allocation=audit.allocation())
            audit.check_lifecycle(validation,passed)
            for mutate in (lambda d:d['numerical_process'].__setitem__('returncode',-15),
                           lambda d:d['numerical_process'].__setitem__('returncode',False),
                           lambda d:d['numerical_process'].__setitem__('timeout',True),
                           lambda d:d.__setitem__('scientific_receipt_passed',not passed)):
                wrong=deepcopy(validation); mutate(wrong)
                with self.assertRaises(ValueError): audit.check_lifecycle(wrong,passed)


if __name__=='__main__': unittest.main()
