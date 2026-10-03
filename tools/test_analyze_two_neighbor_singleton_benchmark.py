"""Synthetic scalars, metadata and cached labels only; no physical queries."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_two_neighbor_singleton_benchmark as a


def pose(x): return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def policy(cap=2, width=4.):
    return dict(schema='two-neighbor-singleton-policy-v1', uniform_half_width=width,
        uniform_probability=.5, trial_cap=cap, member_schedule='alternating_0_first',
        oligomer=dict(multi_contact_mass=.8, max_mismatch=12., pair_distance_A=8.,
            pair_angle_degrees=60., max_candidates=4096, max_hard_checks=256, max_components=32))


def contract(config, job):
    context = config['contexts'][job['context_index']]
    return dict(schema='evolving-dimer-two-neighbor-singleton-v1', policy=config['singleton_policy'],
        effective_oligomer=a.effective_oligomer(config, job['arm']),
        member_labels=[context['root'], context['child']], anchor_label=context['anchor'],
        member_slot_schedule='(block-1)%2', neighbors='[other_mobile,fixed_anchor]',
        uniform_frame='fixed_anchor_body', catalogue_rebuild='every elementary attempt',
        proposal_rng_role='singleton_two_neighbor/proposal', bath_rng_role='singleton_two_neighbor/bath',
        accept_rng_role='singleton_two_neighbor/accept', local_rng_roles='unchanged shared local/{attempt}/{proposal,bath,accept}',
        physical_decisions_per_candidate=1, guidance_cloud_used=False)


def density(shift=0., zero=False):
    if zero: return dict(log_uniform='-inf', log_learned='-inf', log_full='-inf')
    u = -3*math.log(8.); learned = u+shift
    return dict(log_uniform=u, log_learned=learned,
                log_full=max(u, learned)+math.log(math.exp(u-max(u, learned))+math.exp(learned-max(u, learned)))-math.log(2.))


def bath(gained=2, lost=3):
    return dict(gained=gained, lost=lost, raw_points=10, retained_points=gained+lost,
        retained_cells=1, created_cells=1, envelope_volume=3.,
        log_weight=math.log1p(1/64)*(gained-lost))


def singleton(config, job, block, selected, status, accepted=False):
    slot=(block-1)%2; members=[1,3]; old=copy.deepcopy(selected[slot]); new=pose(old['position'][0]+.1)
    neighbors=[members[1-slot],2]; source=dict(spectator_core_collisions=[],wall_valid=True)
    counts=dict(uniform=0,learned=0,geometric_rejections=0,hard_rejections=0,wall_rejections=0,candidates=0,fatal_trials=0)
    p=dict(status=status,moving=members[slot],neighbors=neighbors,old=old,
        trial_cap=config['singleton_policy']['trial_cap'],uniform_frame=pose(0),uniform_center=[0.,0.,0.],
        uniform_half_width=4.,uniform_probability=.5,source_feasibility=source,
        old_density=density(),new_density=None,candidate=None,full_log_reverse_forward=None,
        counts=counts,trials=[])
    row=dict(kind='two_neighbor_singleton',block=block,member_slot=slot,member=members[slot],neighbors=neighbors,
        old=old,accepted=False,status='proposal_self_loop',proposal=p)
    if status=='source_zero_reverse_flow': p['old_density']=density(zero=True)
    elif status=='cap_zero': p['old_density']=None
    else:
        for index in range(1,3 if status=='cap_exhausted' else 2):
            valid=status=='candidate';branch='uniform' if index%2 else 'learned'
            p['trials'].append(dict(index=index,branch=branch,branch_uniform=.1 if branch=='uniform' else .9,
                trace={'synthetic':True},proposed_pose=new,density=density(shift=.7),
                feasibility=dict(spectator_core_collisions=[] if valid else [2],wall_valid=valid),
                disposition='candidate' if valid else 'geometric_rejection'))
            counts[branch]+=1
            if valid: counts['candidates']+=1
            else:
                for k in ['geometric_rejections','hard_rejections','wall_rejections']:counts[k]+=1
        if status=='candidate':
            p.update(candidate=new,new_density=p['trials'][-1]['density'])
            correction=p['old_density']['log_full']-p['new_density']['log_full'];p['full_log_reverse_forward']=correction
            physical=bath();ratio=correction+physical['log_weight']
            row.update(status='completed',accepted=accepted,proposed=new,complete_log_correction=correction,
                bath=physical,log_acceptance_ratio=ratio,log_u=-2. if accepted else -.01)
    return row


def fixture(arm=a.ARMS[0], initialization='source'):
    c=dict(contexts=[dict(root=1,child=3,anchor=2)],allocation=dict(warmup_blocks=1,production_blocks=3),
        physical=dict(a.PHYSICAL),singleton_policy=policy(),source_frame={'path':'synthetic-frame','sha256':'frame'},
        inherited_campaign={'prepared_manifest':{'path':'synthetic-prepared','sha256':'prepared'}})
    j=dict(id=0,context_index=0,initialization=initialization,stream=0,arm=arm)
    initial=[pose(1),pose(3)];selected=copy.deepcopy(initial)
    saved_start=dict(context_index=0,stream=0,record={'path':'start','sha256':'start'}) if initialization=='proposal_prepared' else None
    rows=[dict(kind='initial',block=0,job=j,selected=copy.deepcopy(initial),conditional_target=True,
        fixed_source=c['source_frame'],prepared_manifest=c['inherited_campaign']['prepared_manifest'],
        prepared_start=saved_start,singleton_contract=contract(c,j),sampler_cpu_seconds=0.)]
    counts=dict(local_attempted=0,local_accepted=0,singleton_attempted=0,singleton_accepted=0,singleton_self_loop=0)
    raw=retained=0
    for block, status in enumerate(['candidate','cap_exhausted','source_zero_reverse_flow','candidate'],1):
        for attempt,slot in enumerate([0,1,0,1]):
            rows.append(dict(kind='local',block=block,attempt=attempt,member=[1,3][slot],old=copy.deepcopy(selected[slot]),
                proposed=pose(999),accepted=False,status='hard_rejected',retained=copy.deepcopy(selected),sampler_cpu_seconds=float(len(rows))))
            counts['local_attempted']+=1
        row=singleton(c,j,block,selected,status,accepted=block==1)
        counts['singleton_attempted']+=1;counts['singleton_accepted']+=row['accepted'];counts['singleton_self_loop']+=row['status']=='proposal_self_loop'
        if row['accepted']:selected[(block-1)%2]=copy.deepcopy(row['proposed'])
        if 'bath' in row:raw+=row['bath']['raw_points'];retained+=row['bath']['retained_points']
        row.update(retained=copy.deepcopy(selected),sampler_cpu_seconds=float(len(rows)));rows.append(row)
        rows.append(dict(kind='retained_block',block=block,production=block>1,selected=copy.deepcopy(selected),raw=raw,
            retained=retained,counts=copy.deepcopy(counts),sampler_cpu_seconds=float(len(rows))))
    t=dict(complete=True,conditional_target=True,job=j,blocks=4,counts=counts,raw=raw,retained=retained,cpu_seconds=float(len(rows)))
    return c,j,initial,rows,t,saved_start


class ReplayTests(unittest.TestCase):
    def replay(self, fixture_value):
        c,j,s,rows,t,start=fixture_value
        with patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
            return a.validate_journal(rows,c,j,s,pose(0),t,start)

    def test_all_attempts_single_member_acceptance_and_rejected_residence(self):
        for arm in a.ARMS:
            for init in a.STARTS:
                f=fixture(arm,init);points=self.replay(f)
                self.assertEqual([p['block'] for p in points],list(range(5)))
                self.assertEqual(points[-1]['selected'],[pose(1.1),pose(3)])
                self.assertEqual(points[-1]['counts']['singleton_self_loop'],2)
                self.assertEqual(points[-1]['raw'],20)

    def test_complete_mixture_and_bath_correction_once(self):
        for defect in ('learned_only','double_correction','bath_sign','bath_count','accepted','candidate','null_bath'):
            f=fixture();row=f[3][5]
            if defect=='learned_only':row['proposal']['old_density']['log_full']+=.3
            elif defect=='double_correction':row['complete_log_correction']*=2
            elif defect=='bath_sign':row['bath']['log_weight']*=-1
            elif defect=='bath_count':row['bath']['retained_points']+=1
            elif defect=='accepted':row['log_u']=-.001
            elif defect=='candidate':row['proposal']['candidate']=pose(100)
            else:f[3][11]['bath']=bath()
            with self.subTest(defect=defect),self.assertRaises(ValueError):self.replay(f)

    def test_schedule_context_binding_and_state_defects_rejected(self):
        for defect in ('slot','neighbors','anchor_frame','other_member','missing','extra','fatal','source','prepared','contract','cpu','terminal'):
            f=fixture();rows=f[3];row=rows[5]
            if defect=='slot':row['member_slot']=1
            elif defect=='neighbors':row['neighbors']=[2,3]
            elif defect=='anchor_frame':row['proposal']['uniform_frame']=pose(7)
            elif defect=='other_member':row['retained'][1]=pose(6)
            elif defect=='missing':rows.pop(1)
            elif defect=='extra':rows.append(copy.deepcopy(rows[-1]))
            elif defect=='fatal':row['proposal_failure']={}
            elif defect=='source':rows[0]['fixed_source']={'path':'wrong','sha256':'wrong'}
            elif defect=='prepared':rows[0]['prepared_start']={}
            elif defect=='contract':rows[0]['singleton_contract']['effective_oligomer']['multi_contact_mass']=0.
            elif defect=='cpu':row['sampler_cpu_seconds']=-1.
            else:f[4]['counts']=dict(f[4]['counts'],singleton_accepted=2)
            with self.subTest(defect=defect),self.assertRaises(ValueError):self.replay(f)

    def test_first_success_full_branch_redraw_and_zero_flow_contracts(self):
        for defect in ('extra_after_success','premature_cap','branch','zero_flow_trials','fatal_disposition','counter'):
            f=fixture();row=f[3][5]
            if defect=='extra_after_success':row['proposal']['trials'].append(copy.deepcopy(row['proposal']['trials'][0]))
            elif defect=='premature_cap':f[3][11]['proposal']['trials'].pop()
            elif defect=='branch':row['proposal']['trials'][0]['branch_uniform']=.9
            elif defect=='zero_flow_trials':f[3][17]['proposal']['trials']=[copy.deepcopy(row['proposal']['trials'][0])]
            elif defect=='fatal_disposition':row['proposal']['trials'][0]['disposition']='fatal'
            else:row['proposal']['counts']['candidates']=True
            with self.subTest(defect=defect),self.assertRaises(ValueError):self.replay(f)

    def test_split_serialized_journal_replay_is_identical(self):
        f=fixture(initialization='proposal_prepared');expected=self.replay(f)
        rows=f[3];split=13
        restored=json.loads(json.dumps(rows[:split]))+json.loads(json.dumps(rows[split:]))
        continued=(*f[:3],restored,*f[4:])
        self.assertEqual(self.replay(continued),expected)

    def test_zero_cap_consumes_no_trial_or_bath(self):
        c,j,initial,_,_,_=fixture();c['singleton_policy']['trial_cap']=0
        row=singleton(c,j,1,initial,'cap_zero')
        self.assertIsNone(a.validate_singleton(row,c,j,initial,pose(0)))
        row['proposal']['old_density']=density()
        with self.assertRaises(ValueError):a.validate_singleton(row,c,j,initial,pose(0))

    def test_nonfinite_and_pure_missing_support_densities(self):
        for key in ('log_uniform','log_learned','log_full'):
            d=density();d[key]=float('nan')
            with self.assertRaises(ValueError):a.density(d,4.)
        self.assertEqual(a.density(density(zero=True),4.),-math.inf)
        d=density();d['log_learned']='-inf';d['log_full']=d['log_uniform']-math.log(2.)
        self.assertEqual(a.density(d,4.),d['log_full'])

    def test_finite_density_subtraction_overflow_is_fatal_not_mh_rejection(self):
        c,j,initial,rows,_,_=fixture();row=rows[5];p=row['proposal']
        p['old_density']=dict(log_uniform='-inf',log_learned=-1e308,log_full=-1e308)
        p['new_density']=dict(log_uniform='-inf',log_learned=1e308,log_full=1e308)
        p['trials'][0]['density']=copy.deepcopy(p['new_density'])
        p['full_log_reverse_forward']='-inf'
        row.update(complete_log_correction='-inf',log_acceptance_ratio='-inf',accepted=False)
        with self.assertRaisesRegex(ValueError,'Nonfinite singleton proposal correction is fatal'):
            a.validate_singleton(row,c,j,initial,pose(0))


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value)+'\n').encode())
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def metadata_fixture(root):
    base=root/'campaign';base.mkdir();old=root/'controls'
    jobs=[dict(id=n,context_index=c,arm=arm,initialization=init,stream=s)
          for n,(c,arm,init,s) in enumerate((c,arm,init,s) for c in range(4) for arm in a.ARMS for init in a.STARTS for s in range(4))]
    control_jobs=[dict(id=n,context_index=c,arm=arm,initialization=init,stream=s)
          for n,(c,arm,init,s) in enumerate((c,arm,init,s) for c in range(4) for arm in ('local','unguided','m4') for init in a.STARTS for s in range(4))]
    obs={str(j['id']):put(old/f"job-{j['id']:03}-observations.jsonl",b'inert cached bytes; no parse\n') for j in control_jobs if j['arm'] in a.CONTROLS}
    analysis=put(old/'analysis.json',dict(complete=True,chains=[dict(job=j,metrics={}) for j in control_jobs]))
    manifest=put(old/'manifest.json',dict(complete=True,files={Path(v['path']).name:v['sha256'] for v in obs.values()}))
    summary=put(old/'summary.json',dict(complete=True,passed=True,analysis=analysis,manifest_sha256=manifest['sha256']))
    config=dict(schema='evolving-dimer-benchmark-v1',jobs=jobs,contexts=[dict(root=1,child=3,anchor=2)]*4,
        physical=dict(a.PHYSICAL),allocation=dict(warmup_blocks=512,production_blocks=4096),
        singleton_policy=policy(32,160.),output=str(base/'execution'),
        control_analysis=dict(summary=summary,manifest=manifest,analysis=analysis,observations=obs))
    conf=put(base/'config.json',config);put(base/'protocol.json',{});binding=put(base/'run-binding.json',{})
    put(base/'analysis-plan.json',a.analysis_plan())
    dispatch=put(base/'dispatch/plan.json',dict(jobs=jobs,retries=False,replacements=False));completed=[]
    for job in jobs:
        directory=base/'execution'/f"job-{job['id']:03}";trajectory=put(directory/'trajectory.jsonl',b'inert raw bytes; metadata must not parse\n')
        terminal=put(directory/'terminal.json',dict(complete=True,conditional_target=True,job=job,blocks=4608,
            config_sha256=conf['sha256'],binding_sha256=binding['sha256'],trajectory=trajectory,
            counts=dict(local_attempted=18432,singleton_attempted=4608)))
        completed.append(dict(job=job,success=True,returncode=0,error=None,child_drained=True,terminal_sha256=terminal['sha256']))
    put(base/'dispatch/status.json',dict(complete=True,passed=True,failure=None,failure_draining=False,
        active=[],unstarted=[],plan_sha256=dispatch['sha256'],completed=completed))
    return base,config


class MetadataTests(unittest.TestCase):
    def test_exact_allocation_and_complete_inert_byte_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            base,config=metadata_fixture(Path(temp))
            with patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
                files,items,controls=a.bind_complete_inputs(base,config)
            self.assertEqual((len(items),len(controls)),(64,64))
            self.assertTrue(files)
            self.assertEqual({c['job']['arm'] for c,_ in controls},set(a.CONTROLS))
        p=a.analysis_plan();self.assertEqual(p['new_retained_initial_observations'],294976)
        self.assertEqual(p['new_production_observations'],262144)
        self.assertEqual(p['maximum_new_pair_classifications'],154862400)
        self.assertFalse(p['native_observer']);self.assertEqual(p['old_geometry_queries'],0)

    def test_missing_failed_or_repeated_inventory_rejected_before_geometry(self):
        for defect in ('partial','duplicate','undrained','changed_bytes','failed'):
            with tempfile.TemporaryDirectory() as temp:
                base,config=metadata_fixture(Path(temp));p=base/'dispatch/status.json';status=a.read(p)
                if defect=='partial':status['completed'].pop()
                elif defect=='duplicate':status['completed'][1]=status['completed'][0]
                elif defect=='undrained':status['completed'][0]['child_drained']=False
                elif defect=='changed_bytes':(base/'execution/job-000/trajectory.jsonl').write_bytes(b'changed')
                else:put(base/'execution/job-000/failure.json',{})
                put(p,status)
                with self.subTest(defect=defect),self.assertRaises(ValueError):a.bind_complete_inputs(base,config)

    def test_wrong_arm_target_or_scope_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base,original=metadata_fixture(Path(temp))
            for defect in ('arm','mass','density','schedule'):
                c=copy.deepcopy(original)
                if defect=='arm':c['jobs'][0]['arm']='root_m4'
                elif defect=='mass':c['singleton_policy']['oligomer']['multi_contact_mass']=.5
                elif defect=='density':c['physical']['activity']=.035
                else:c['allocation']['production_blocks']=4097
                with self.subTest(defect=defect),self.assertRaises(ValueError):a.validate_inventory(c)


class MetricTests(unittest.TestCase):
    def test_cached_empty_external_sets_and_internal_only_changes(self):
        trace=[dict(block=i,external_edges=[],partner_edges=[[1,3]] if i%2 else []) for i in range(7)]
        with patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
            m=a.external.external_metrics(trace,0,5.,[1,3])
        self.assertIsNone(m['ess']['apparent_ess']);self.assertEqual(m['environments']['completed_passages'],0)
        self.assertEqual(m['environments']['completed_returns'],[])

    def test_shared_singleton_roles_do_not_claim_global_control_pairing(self):
        from test_analyze_root_guided_dimer_benchmark import chain
        chains=[chain(arm,'source',[[],[(1,8)],[],[]],4.) for arm in ('local','m4',*a.ARMS)]
        summary=a.comparison_summaries(chains)
        for pair in summary['descriptive_paired_comparisons']:
            expected=pair['left']['arm'] in a.ARMS and pair['right']['arm'] in a.ARMS
            self.assertEqual(pair['control_global_rng_roles_paired'],expected)
        self.assertEqual(len(summary['groups']),4)


if __name__=='__main__':unittest.main()
