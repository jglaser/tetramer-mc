"""Synthetic saved-journal tests only; no protein sampling or geometry queries."""
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parent)]
import analyze_partner_atlas_benchmark as a
import test_audit_partner_atlas_reference as fixture_ref


def fixture(arm=a.ARMS[1],start='source'):
    index=a.ARMS.index(arm); refarm=fixture_ref.audit.ARMS[index]
    raw,old,_,_,meta,_,_=fixture_ref.fixture(refarm)
    config=dict(contexts=[dict(root=9,child=24,anchor=71)],
        partner_atlas_policy=dict(schema='partner-atlas-policy-v1',proposal_scales=dict(source='local')),
        local=dict(translation_std_A=.25,rotation_std_degrees=30.),
        allocation=dict(warmup_blocks=1,production_blocks=3),physical=dict(activity=.5,lambda_ratio=4.),
        cloud=dict(raw_count=8),source_frame={'synthetic':'source'},
        limits=dict(raw_per_leg=1000,retained_per_leg=1000,raw_per_outer=1000,retained_per_outer=1000,
                    raw_campaign=10000,retained_campaign=10000,cpu_seconds=1000.))
    settings=dict(atlas={'synthetic':'atlas'},original_atlas=None,atlas_method='posterior_involution',
        atlas_correlation=0.,atlas_uniform_probability=.1,atlas_cube=[3.]*3,atlas_center=[0.]*3,
        atlas_chart_count=3,atlas_angular_length=1.,atlas_periodic=False)
    job=dict(id=0,context_index=0,arm=arm,initialization=start,stream=0)
    cloud={'synthetic':'cloud'}; metadata=dict(raw_count=8,low=[0.,0.,0.],high=[2.,2.,1.],kept_indices=list(range(8)),cpu_seconds=.01)
    prepared={'synthetic':'prepared'}; alternative=None if start=='source' else {'synthetic':'start'}
    contract=a.surrogate_contract(config,job,cloud,metadata,settings)
    rows=[dict(kind='initial',block=0,job=job,selected=copy.deepcopy(old),conditional_target=True,
        sampler_cpu_seconds=0.,fixed_source=config['source_frame'],cloud=cloud,cloud_cpu_seconds=.01,
        prepared_manifest=prepared,prepared_start=alternative,partner_atlas_contract=contract)]
    counts=dict(local_attempted=0,local_accepted=0,dimer_attempted=0,dimer_accepted=0,dimer_self_loop=0)
    nraw=nretained=0
    for block in range(1,5):
        for attempt,slot in enumerate([0,1,0,1]):
            rows.append(dict(kind='local',block=block,attempt=attempt,member=[9,24][slot],
                old=copy.deepcopy(old[slot]),proposed=fixture_ref.pose(100.),accepted=False,status='hard_rejected',
                retained=copy.deepcopy(old),sampler_cpu_seconds=float(len(rows))))
            counts['local_attempted']+=1
        row=copy.deepcopy(raw); row.update(block=block,members=[9,24],sampler_cpu_seconds=float(len(rows)+1))
        for step in row['steps']:
            step['selected_label']=[9,24][step['selected_slot']]
            if step['mode']=='partner_atlas':step['partner_label']=[9,24][step['partner_slot']]
        row['bath']['ordered_members']=[9,24]
        legs=[fixture_ref.gate(lost=4),fixture_ref.gate()]
        aggregate={k:legs[0][k]+legs[1][k] for k in legs[0]}
        row['bath'].update(legs=legs,aggregate=aggregate)
        row.update(log_acceptance_ratio=aggregate['log_weight']+row['complete_log_correction'],
                   log_u=-100. if block==4 else -.0001)
        row['accepted']=row['log_u']<min(0.,row['log_acceptance_ratio'])
        row['retained']=copy.deepcopy(row['proposed'] if row['accepted'] else old)
        row['budget_before']=dict(raw=nraw,retained=nretained)
        rows.append(dict(kind='partner_atlas_attempt_begun',status='begun',block=block,members=[9,24],
            config=contract['effective_config'],old=copy.deepcopy(old),raw=nraw,bath_retained=nretained,
            inner_filter=index!=0,sampler_cpu_seconds=float(len(rows))))
        nraw+=aggregate['raw_points'];nretained+=aggregate['retained_points']
        row['budget_after']=dict(raw=nraw,retained=nretained);rows.append(row)
        counts['dimer_attempted']+=1;counts['dimer_accepted']+=row['accepted']
        rows.append(dict(kind='retained_block',block=block,production=block>1,selected=copy.deepcopy(row['retained']),
            raw=nraw,retained=nretained,counts=copy.deepcopy(counts),sampler_cpu_seconds=float(len(rows))))
    terminal=dict(complete=True,conditional_target=True,job=job,blocks=4,counts=counts,raw=nraw,retained=nretained,
                  cpu_seconds=float(len(rows)))
    return config,job,old,rows,terminal,cloud,metadata,prepared,alternative,settings


def check(data,diagnostics=None):
    c,j,old,rows,terminal,cloud,meta,prepared,start,settings=data
    with mock.patch.object(a,'atlas_settings',return_value=settings),mock.patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
        return a.validate_journal(iter(rows),c,j,old,terminal,cloud,meta,prepared,start,diagnostics)


class PartnerObserverTests(unittest.TestCase):
    def test_all_arms_starts_preserve_rejected_residence_and_inputs(self):
        for arm,start in itertools.product(a.ARMS,a.STARTS):
            data=fixture(arm,start);frozen=copy.deepcopy(data);d=dict(warmup_blocks=1)
            points=check(data,d);self.assertEqual(data,frozen);self.assertEqual(len(points),5)
            self.assertEqual(points[0]['selected'],points[3]['selected'])
            self.assertNotEqual(points[3]['selected'],points[4]['selected'])
            reduced=a.finish_diagnostics(d);self.assertEqual(reduced['production']['outer_calls'],3)
            self.assertEqual(reduced['production']['candidates'],3*(1 if arm in a.ARMS[:2] else 8))

    def test_reject_wrong_corrections_partner_and_labels(self):
        changes=[lambda r:r['steps'][0].update(proposal_log_reverse_forward=0.),
            lambda r:r['steps'][0]['proposal_trace'].update(full_new_member_log_density=0.),
            lambda r:r['steps'][0]['proposal_trace']['step'].update(log_extended_jacobian=10.),
            lambda r:r['steps'][0].update(partner_label=9),
            lambda r:r['steps'][0].update(partner_pose=[0]),
            lambda r:r['steps'][0]['proposed'][1]['position'].__setitem__(0,88.),
            lambda r:r.update(complete_log_correction=r['complete_log_correction']+1.),lambda r:r.update(accepted=True),
            lambda r:r['steps'][0].update(selected_label=True),lambda r:r.update(inner_filter=0)]
        for arm,change in itertools.product(a.ARMS,changes):
            data=fixture(arm);change(data[3][6])
            with self.subTest(arm=arm,change=change),self.assertRaises((ValueError,KeyError,TypeError)):check(data)

    def test_direct_does_not_admit_score_or_inner_coin(self):
        for change in [lambda r:r.update(old_score={}),lambda r:r['steps'][0].update(log_u=-1.),
                       lambda r:r['steps'][0].update(proposed_score={})]:
            data=fixture(a.ARMS[0]);change(data[3][6])
            with self.assertRaises(ValueError):check(data)

    def test_wrong_order_incomplete_tail_and_caps(self):
        for change in [lambda d:d[3].pop(5),lambda d:d[3].append(copy.deepcopy(d[3][-1])),
            lambda d:d[3][6]['steps'].pop(),lambda d:d[3][6]['budget_after'].update(raw=0),
            lambda d:d[3][6]['bath'].update(ordered_members=[24,9]),
            lambda d:d[0]['limits'].update(raw_per_leg=1),
            lambda d:d[3][6].update(fatal_error='budget'),lambda d:d[3][7].update(selected=d[3][6]['proposed'])]:
            data=fixture();change(data)
            with self.assertRaises((ValueError,KeyError)):check(data)

    def test_source_score_counts_allow_multiple_spectators(self):
        data=fixture();row=data[3][6]
        for score in [row['old_score'],row['proposed_score'],row['steps'][0]['proposed_score']]:
            score['nearby_spectators']=[4,3];score['spectator_membership_queries']=40
        check(data)
        row['old_score']['nearby_spectators']=[263,3]
        with self.assertRaises(ValueError):check(data)

    def test_relative_pose_is_invariant_to_common_rigid_motion(self):
        data=fixture();points=check(data);before=a.relative_metrics(points,1,100.)
        moved=copy.deepcopy(points)
        # A common translation leaves the body-relative pose unchanged exactly.
        for row in moved:
            for pose in row['selected']:
                pose['position']=[x+y for x,y in zip(pose['position'],[8.,-4.,2.])]
        self.assertEqual(a.relative_metrics(moved,1,100.),before)
        constants=[x for x in before['components'] if x['minimum']==x['maximum']]
        self.assertTrue(constants)
        self.assertTrue(all(x['apparent_ess']['apparent_ess'] is None for x in constants))

    def test_analysis_allocation_and_policy(self):
        plan=a.analysis_plan();self.assertEqual(plan['new_chains'],32)
        self.assertEqual(plan['new_retained_initial_observations'],147488)
        self.assertEqual(plan['scalar_inner_candidates'],663552)
        self.assertFalse(plan['assembly_gate_open']);self.assertFalse(plan['native_observer'])
        for arm in a.ARMS:
            data=fixture(arm);self.assertEqual(a.effective_config(data[0],arm)['inner_steps'],1 if arm in a.ARMS[:2] else 8)
        data=fixture();data[0]['partner_atlas_policy']['bad']=True
        with self.assertRaises(ValueError):a.effective_config(data[0],a.ARMS[1])

    def test_metadata_reconstruction_binds_chart_count_cube_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shape=root/'shape.json';model=root/'atlas.json'
            shape.write_text(json.dumps(dict(atoms=[dict(center=[3.,4.,0.],radius=2.),
                                                    dict(center=[0.,0.,0.],radius=1.)])))
            model.write_text(json.dumps(dict(schema='reciprocal-pose-mixture-v1',reciprocal_components=[False,True],
                base_model=dict(shape_sha256=a.sha(shape),weights=[.3,.7],angular_length=4.))))
            config=fixture()[0];config['physical']['wall_radius']=10.
            config.update(shape=dict(path=str(shape),sha256=a.sha(shape)),atlas=dict(path=str(model),sha256=a.sha(model)))
            settings=a.atlas_settings(config)
            self.assertEqual(settings['atlas_cube'],[34.]*3);self.assertEqual(settings['atlas_chart_count'],3)
            self.assertEqual(settings['atlas_angular_length'],4.);self.assertFalse(settings['atlas_periodic'])
            data=fixture();data[0].update({k:config[k] for k in ('shape','atlas')})
            data[0]['physical']['wall_radius']=10.
            c,j,old,rows,terminal,cloud,meta,prepared,start,_=data
            rows[0]['partner_atlas_contract']=a.surrogate_contract(c,j,cloud,meta)
            a.validate_journal(iter(rows),c,j,old,terminal,cloud,meta,prepared,start)
            for key,value in [('atlas_chart_count',2),('atlas_cube',[33.]*3)]:
                bad=copy.deepcopy(rows);bad[0]['partner_atlas_contract'][key]=value
                with self.assertRaises(ValueError):a.validate_journal(iter(bad),c,j,old,terminal,cloud,meta,prepared,start)
            config['shape']['sha256']='0'*64
            with self.assertRaises(ValueError):a.atlas_settings(config)
            config['shape']['sha256']=a.sha(shape)
            payload=json.loads(model.read_text());payload['reciprocal_components']=[0,True];model.write_text(json.dumps(payload))
            config['atlas']['sha256']=a.sha(model)
            with self.assertRaises(ValueError):a.atlas_settings(config)

    def test_null_and_zero_reverse_support_are_recorded_self_loops(self):
        for index,arm in enumerate(a.ARMS):
            c,j,old,_,_,cloud,meta,_,_,settings=fixture(arm)
            contract=a.surrogate_contract(c,j,cloud,meta,settings)
            r,*_=fixture_ref.fixture(fixture_ref.audit.ARMS[index],null=True)
            r.update(block=1,members=[9,24],retained=copy.deepcopy(old))
            for step in r['steps']:
                step['selected_label']=[9,24][step['selected_slot']]
                if step['mode']=='partner_atlas':step['partner_label']=[9,24][step['partner_slot']]
            begun=dict(kind='partner_atlas_attempt_begun',status='begun',block=1,members=[9,24],
                config=contract['effective_config'],old=copy.deepcopy(old),raw=0,bath_retained=0,inner_filter=index!=0)
            a.validate_surrogate(r,begun,c,j,contract)
            self.assertEqual(r['physical_decisions'],0)
            # The selected source is outside the immutable cube; the encoded
            # uniform candidate is inside. No redrawing and no score/bath.
            uniform=copy.deepcopy(r);first=uniform['steps'][0]
            first.update(status='zero_reverse_support',proposal_trace=dict(branch='uniform',charts='members',log_reverse_forward=0.),
                         uniform_source_support=False,uniform_candidate_support=True)
            first['proposed'][0]['position'][0]=-.375
            counts=uniform['proposal_counts' if index==0 else 'inner_counts']
            counts.update(null_proposals=0,zero_reverse_support=1)
            restricted=dict(contract,atlas_cube=[.8]*3)
            a.validate_surrogate(uniform,begun,c,j,restricted)
            first['proposed'][0]['position'][0]=-.6
            with self.assertRaises(ValueError):a.validate_surrogate(uniform,begun,c,j,restricted)
            bad=copy.deepcopy(r);bad['physical_decisions']=1
            with self.assertRaises(ValueError):a.validate_surrogate(bad,begun,c,j,contract)

if __name__=='__main__':unittest.main()
