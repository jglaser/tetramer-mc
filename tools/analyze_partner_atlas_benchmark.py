#!/usr/bin/env python3
"""Completed partner-atlas-chain audit and cached-contact comparison.

No old geometry is replayed. Scalar audit cannot certify RNG, atomic hard
predicates, score membership or filesystem durability. Contact patches are
coarse threshold descriptors, not native registry or identified binding basins.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

import analyze_rigid_surrogate_benchmark as rigid
import analyze_flexible_surrogate_benchmark as flexible
import audit_partner_atlas_reference as atlas
import plot_rigid_surrogate_benchmark as authority

previous, external, scalar, shared = rigid.previous, rigid.external, rigid.scalar, rigid.shared
require, read, sha, canonical, write = rigid.require, rigid.read, rigid.sha, rigid.canonical, rigid.write
finite, close, identity, cloud_weight, validate_score, decision = (rigid.finite, rigid.close,
    rigid.identity, rigid.cloud_weight, rigid.validate_score, rigid.decision)
ARMS = ('partner_atlas_direct', 'partner_atlas_m1', 'partner_atlas_m8', 'partner_atlas_flat8')
STARTS = rigid.STARTS
CONTROLS = rigid.ARMS + ('local', 'm4')
SOURCE_FILES = list(dict.fromkeys(['tools/analyze_partner_atlas_benchmark.py',
    'tests/test_analyze_partner_atlas_benchmark.py',
    'tools/plot_rigid_surrogate_benchmark.py'] + flexible.SOURCE_FILES + ['tools/audit_partner_atlas_reference.py','tools/audit_flexible_surrogate_reference.py']))


def analysis_plan():
    plan = copy.deepcopy(rigid.analysis_plan())
    plan.update(schema='partner-atlas-analysis-plan-v1', source_files=SOURCE_FILES,
        new_chains=32, reused_control_chains=40, reused_control_arms=list(CONTROLS),
        primary_control_arms=['local'], contextual_control_arms=list(rigid.ARMS)+['m4'],
        new_retained_initial_observations=32*4609, new_production_observations=32*4096,
        scalar_local_attempts=32*4608*4, scalar_surrogate_attempts=32*4608,
        scalar_inner_candidates=8*4608*18, maximum_new_pair_classifications=32*4609*525,
        target='Context0 two-mobile conditional target; four unchanged locals then one partner-anchored update.',
        replay='All begun/outcome records, nulls, hard/MH rejections, complete learned q, inner score and final two-leg bath corrections.',
        internal_metrics='Internal contact and patch fingerprints, occupancy, completed nonempty exchanges, full CPU.',
        internal_interpretation='Threshold contact descriptors, not native registry or a complete relative-pose descriptor.',
        cached_control_identity='All40 authenticated rigid/local/m4 histories once; local8 primary, other32 contextual.',
        relative_pose_metrics=True, native_observer=False,
        relative_pose_scope='Body-frame translation, relative rotation matrix and distances from initial/last-warmup relative poses, '
                            'componentwise apparent ESS; not native labels or identified basins.',
        comparisons='Four streams per arm/start separately. Paired local/atlas RNG roles do not make different arms independent. '
                    'Full sampler CPU includes warmup, score construction and all rejected work. Undefined constant-series ESS stays undefined.',
        decision_rule='Conditional engineering pilot only; better acceptance alone is insufficient. Require broader contact/motif coverage, '
                      'initialization agreement and contact ESS per CPU before promotion. Physical-weight and native-label gates remain separate.')
    return plan



def effective_config(config, arm):
    require(arm in ARMS, 'Unknown partner-atlas arm')
    policy = config['partner_atlas_policy']
    require(set(policy) == {'schema', 'proposal_scales'} and policy['schema'] == 'partner-atlas-policy-v1',
            'Changed partner-atlas policy')
    proxy = dict(config, surrogate_policy=dict(policy, schema='rigid-surrogate-policy-v1'))
    equivalent = rigid.ARMS[(0,0,1,2)[ARMS.index(arm)]]
    return rigid.effective_config(proxy, equivalent)



def validate_inventory(config):
    jobs = config['jobs']
    require(len(jobs) == 32 and len({j['id'] for j in jobs}) == 32 and {identity(j) for j in jobs}
            == {(0, arm, start, stream) for arm in ARMS for start in STARTS for stream in range(4)},
            'Changed complete32 partner-atlas inventory')
    require(config['physical'] == scalar.PHYSICAL and config['allocation']['warmup_blocks'] == 512
            and config['allocation']['production_blocks'] == 4096, 'Changed target/schedule')
    require(not any(k in config for k in ('surrogate_policy', 'flexible_surrogate_policy', 'singleton_policy', 'two_root_policy')),
            'Another proposal policy present')
    require(config['partner_atlas_policy']['proposal_scales'] == {'source': 'local'}
            and config['local'] == dict(translation_std_A=.2, rotation_std_degrees=1., member_order=[0, 1, 0, 1],
                                        pair_contact_required=False),
            'Changed matched local scales/schedule')
    for arm in ARMS: effective_config(config, arm)


def surrogate_contract(config, job, cloud, meta, settings=None):
    proxy = dict(config, surrogate_policy=dict(config['partner_atlas_policy'], schema='rigid-surrogate-policy-v1'))
    old_job = dict(job, arm=rigid.ARMS[(0,0,1,2)[ARMS.index(job['arm'])]])
    contract = rigid.surrogate_contract(proxy, old_job, cloud, meta); contract.pop('handle')
    direct = job['arm'] == ARMS[0]
    contract.update(schema='evolving-dimer-partner-atlas-v1', policy=config['partner_atlas_policy'],
        effective_config=effective_config(config, job['arm']), selection_probabilities=[.5,.5],
        mode_probabilities=dict(partner_atlas=.25,local=.75),
        inner_selection='independent fair random scan each step',
        atlas_pool='current unselected member only; handle0 of the selected singleton',
        atlas_branch_correction='retained uniform/involution branch; complete learned member density and expanded-map correction',
        uniform_reverse_support='immutable source and generated encoded candidate checked by kernel',
        physical_path='fair-order two-singleton path; intermediate is not hard-filtered', inner_filter=not direct,
        inner_correction='none; no inner score or acceptance draw' if direct else 'S(new)-S(old)+proposal_log_reverse_forward',
        outer_correction='proposal_log_reverse_forward' if direct else 'S(old)-S(new)')
    contract.update(atlas_settings(config) if settings is None else settings)
    for key in ('proposal_rng_role','inner_accept_rng_role','bath_rng_role','accept_rng_role'):
        contract[key] = contract[key].replace('rigid_surrogate/','partner_atlas/')
    return contract



validate_path = flexible.validate_path


def validate_surrogate(row, begun, config, job, contract):
    members, settings = contract['members'], contract['effective_config']
    direct = job['arm'] == ARMS[0]
    require(row['kind'] == ('flexible_partner_atlas_direct' if direct else 'flexible_partner_atlas_chain')
        and begun['kind'] == 'partner_atlas_attempt_begun' and begun['status'] == 'begun'
        and row['inner_filter'] is begun['inner_filter'] is (not direct)
        and row['block'] == begun['block'] and row['old'] == begun['old'], 'Wrong kernel/begun contract')
    atlas.compare(row['members'], members); atlas.compare(begun['members'], members)
    atlas.compare(row['config'], settings); atlas.compare(begun['config'], settings)
    require(row['selection_probabilities'] == [.5,.5] and row['mode_probabilities'] == dict(partner_atlas=.25,local=.75)
            and 'handle' not in row and 'handle' not in begun, 'Wrong mixture')
    old = shared.poses(row['old']); current = copy.deepcopy(old)
    current_score = None; last_q = 0.
    if direct:
        require(not any(k in row for k in ('old_score','proposed_score','inner_counts')), 'Direct arm used score/filter')
        counts = dict(attempted=0,eligible=0,hard_rejected=0,null_proposals=0,zero_reverse_support=0)
    else:
        current_score = row['old_score']; validate_score(current_score,contract,config)
        counts = dict(attempted=0,accepted=0,hard_rejected=0,mh_rejected=0,null_proposals=0,zero_reverse_support=0)
    steps = row['steps']; require(type(steps) is list and len(steps) == settings['inner_steps'], 'Wrong fixed horizon')
    meta = dict(cube=contract['atlas_cube'],center=contract['atlas_center'],branches=contract['atlas_chart_count'])
    for index, step in enumerate(steps):
        require(scalar.nat(step['index'],'step index') == index and step['old'] == current, 'Broken inner residence')
        slot = scalar.nat(step['selected_slot'],'selected slot')
        require(slot in (0,1) and scalar.nat(step['selected_label'],'selected label') == members[slot], 'Wrong scan label')
        target = shared.poses(step['proposed']); require(target[1-slot] == current[1-slot], 'Partner moved')
        local_labels = copy.deepcopy(step)
        if step['mode'] == 'partner_atlas':
            require(scalar.nat(step['partner_label'],'partner label') == members[1-slot], 'Wrong partner label')
            local_labels['partner_label'] = 1-slot
        q, _ = atlas.proposal_correction(local_labels,current,slot,meta)
        counts['attempted'] += 1
        if direct:
            require(not any(k in step for k in ('old_score','proposed_score','retained_score','log_u','log_acceptance_ratio')),
                    'Direct arm used inner score/coin')
        else: close(step['old_score'],current_score['log_surrogate'],'Source score differs')
        status=step['status']
        if status in ('hard_rejected','null_proposal','zero_reverse_support'):
            require(step['accepted'] is False and not any(k in step for k in ('proposed_score','log_u','log_acceptance_ratio')),
                    'Rejection consumed inner decision')
            require((q is None) == (status != 'hard_rejected'), 'Proposal failure status differs')
            counts[{'hard_rejected':'hard_rejected','null_proposal':'null_proposals','zero_reverse_support':'zero_reverse_support'}[status]]+=1
        elif direct:
            require(status == 'direct_candidate' and step['accepted'] is None and q is not None, 'Wrong direct candidate')
            current=copy.deepcopy(target); last_q=q; counts['eligible']+=1
        else:
            require(status == 'completed' and q is not None, 'Fatal/incomplete step')
            new_score=validate_score(step['proposed_score'],contract,config)
            decision(step,new_score-current_score['log_surrogate']+q)
            counts['accepted' if step['accepted'] else 'mh_rejected']+=1
            if step['accepted']: current,current_score=copy.deepcopy(target),step['proposed_score']
        require(step['retained'] == current, 'Wrong retained inner state')
        if not direct:close(step['retained_score'],current_score['log_surrogate'],'Retained score differs')
    atlas.compare(row['proposal_counts' if direct else 'inner_counts'],counts,'attempt counters')
    require(row['proposed'] == current, 'Wrong final endpoint')
    if direct:correction=0. if current==old else last_q
    else:
        require(row['proposed_score'] == current_score,'Wrong final score')
        correction=row['old_score']['log_surrogate']-current_score['log_surrogate']
    close(row['complete_log_correction'],correction,'Wrong outer correction')
    before,after=row['budget_before'],row['budget_after']
    for budget in (before,after):
        require(set(budget)=={'raw','retained'},'Wrong budget fields')
        for k in budget:scalar.nat(budget[k],k)
        require(budget['raw']<=config['limits']['raw_campaign'] and budget['retained']<=config['limits']['retained_campaign'],
                'Campaign budget exceeded')
    require(before==dict(raw=begun['raw'],retained=begun['bath_retained']),'Begun budget differs')
    require(type(row['accepted']) is bool,'Invalid physical decision');scalar.nat(row['physical_decisions'],'physical decisions')
    if current==old:
        require(row['status']=='identity_self_loop' and not row['accepted'] and row['physical_decisions']==0 and before==after
                and not any(k in row for k in ('bath','log_u','log_acceptance_ratio')),'Identity spent physical work')
    else:
        require(row['status']=='completed' and row['physical_decisions']==1,'Repeated/incomplete physical gate')
        weight=validate_path(row['bath'],old,current,members,config);decision(row,weight+correction)
        bath=row['bath']['aggregate']
        require(after==dict(raw=before['raw']+bath['raw_points'],retained=before['retained']+bath['retained_points']),
                'Physical budget differs')
    require(row['retained']==(current if row['accepted'] else old),'Physical retained state differs')



def accumulate(diagnostics,row):
    phase='warmup' if row['block']<=diagnostics['warmup_blocks'] else 'production'
    d=diagnostics.setdefault(phase,dict(outer_calls=0,outer_accepted=0,physical_decisions=0,candidates=0,
        local=0,atlas=0,learned=0,uniform=0,nulls=0,zero_reverse_support=0,hard_rejected=0,mh_rejected=0,
        eligible_direct=0,inner_accepted=0,raw_bath_points=0,retained_bath_points=0,
        selected_slots=[0,0],physical_path_orders=[0,0],proposal_correction=[],score_change=[],outer_correction=[]))
    d['outer_calls']+=1;d['outer_accepted']+=row['accepted'];d['physical_decisions']+=row['physical_decisions']
    d['outer_correction'].append(row['complete_log_correction'])
    for step in row['steps']:
        d['candidates']+=1;d['selected_slots'][step['selected_slot']]+=1
        d['local' if step['mode']=='local' else 'atlas']+=1
        if step['mode']=='partner_atlas':d['uniform' if step['proposal_trace']['branch']=='uniform' else 'learned']+=1
        for status,name in [('null_proposal','nulls'),('zero_reverse_support','zero_reverse_support'),
                            ('hard_rejected','hard_rejected'),('direct_candidate','eligible_direct')]:
            d[name]+=step['status']==status
        if step['status']=='completed':d['inner_accepted' if step['accepted'] else 'mh_rejected']+=1
        if 'proposal_log_reverse_forward' in step:d['proposal_correction'].append(step['proposal_log_reverse_forward'])
        if 'proposed_score' in step:d['score_change'].append(step['proposed_score']['log_surrogate']-step['old_score'])
    if 'bath' in row:
        d['physical_path_orders'][int(row['bath']['order']=='second_then_first')]+=1
        for k in ('raw','retained'):d[k+'_bath_points']+=row['bath']['aggregate'][k+'_points']






validate_checkpoint = flexible.validate_checkpoint


internal_metrics = flexible.internal_metrics


comparison_summaries = flexible.comparison_summaries


bind_cached_controls = flexible.bind_cached_controls


def atlas_settings(config):
    # Metadata/pose algebra only, no protein overlap query. Inputs are also bound
    # in the completed-campaign admission. Keep this reconstruction independent
    # of the settings emitted by Rust.
    refs = [config['atlas'], config['shape']]
    for ref in refs: require(sha(ref['path']) == ref['sha256'], 'Changed atlas/shape input')
    model, shape = (read(ref['path']) for ref in refs)
    require(model['schema'] == 'reciprocal-pose-mixture-v1', 'Wrong frozen reciprocal atlas')
    base = model['base_model']; flags = model['reciprocal_components']
    require(base['shape_sha256'] == config['shape']['sha256'] and len(flags) == len(base['weights'])
            and all(type(f) is bool for f in flags), 'Wrong atlas identity/branches')
    bound = max(math.sqrt(sum(x*x for x in a['center']))+a['radius'] for a in shape['atoms'])
    return dict(atlas=config['atlas'], original_atlas=config.get('original_atlas'),
        atlas_method='posterior_involution', atlas_correlation=0., atlas_uniform_probability=.1,
        atlas_cube=[2*(config['physical']['wall_radius']+bound)]*3, atlas_center=[0.,0.,0.],
        atlas_chart_count=len(flags)+sum(flags), atlas_angular_length=base['angular_length'], atlas_periodic=False)



def finish_diagnostics(diagnostics):
    for phase in ('warmup','production'):
        for key in ('proposal_correction','score_change','outer_correction'):
            values=diagnostics[phase][key]
            diagnostics[phase][key]=dict(n=len(values),mean=math.fsum(values)/len(values) if values else None,
                                       minimum=min(values) if values else None,maximum=max(values) if values else None)
    return diagnostics



def relative_metrics(points,warmup,cpu):
    require(len(points)>warmup+1 and cpu>0,'Empty relative-pose record')
    def relative(point):
        a,b=point['selected'];r0=rigid.rotation(a);r1=rigid.rotation(b)
        return r0.T@(previous.np.asarray(b['position'])-previous.np.asarray(a['position'])),r0.T@r1
    references=[relative(points[i]) for i in (0,warmup)]
    values=[]
    for p in points[warmup+1:]:
        t,r=relative(p);distances=[]
        for t0,r0 in references:
            distances.extend([float(previous.np.linalg.norm(t-t0)),
                math.acos(max(-1.,min(1.,float((previous.np.trace(r0.T@r)-1.)/2.))))])
        values.append(list(t)+list(r.reshape(-1))+distances)
    arr=previous.np.asarray(values)
    names=['relative_t_'+s for s in 'xyz']+[f'relative_R_{i}{j}' for i in range(3) for j in range(3)]
    names += [f'{kind}_from_{reference}' for reference in ('initial','last_warmup')
              for kind in ('translation_distance_A','rotation_angle_radians')]
    return dict(scope=analysis_plan()['relative_pose_scope'],components=[dict(name=name,
        mean=float(arr[:,i].mean()),minimum=float(arr[:,i].min()),maximum=float(arr[:,i].max()),
        apparent_ess=previous.apparent_effective_count(arr[:,i:i+1],cpu)) for i,name in enumerate(names)])



def bind_complete_inputs(base, config):
    """Gate every terminal before opening scientific journals or old caches."""
    import run_native_class_physical_campaign as driver
    base = Path(base).resolve(); inputs = {}
    def bind(ref):
        path = Path(ref['path']).resolve()
        require(sha(path) == ref['sha256'], 'Changed input '+str(path))
        require(str(path) not in inputs or inputs[str(path)] == ref['sha256'], 'Conflicting input binding')
        inputs[str(path)] = ref['sha256']; return path
    def bind_path(path): return bind(dict(path=str(Path(path).resolve()), sha256=sha(path)))
    plan = read(base/'execution-plan.json'); status = read(base/'execution/status.json'); summary = read(base/'execution/summary.json')
    require(all(status[k] == summary[k] for k in ('complete', 'passed', 'failure', 'active', 'unstarted', 'completed', 'plan_sha256'))
            and status['complete'] is True and status['passed'] is True and status['failure'] is None
            and status['active'] is None and status['unstarted'] == [] and len(status['completed']) == len(plan['jobs']) == 32
            and not (base/'execution/failure.json').exists(), 'All32 partner-atlas jobs must complete cleanly')
    for name in ('config.json', 'run-binding.json', 'analysis-plan.json', 'protocol.json', 'execution-plan.json',
                 'execution/claim.json', 'execution/status.json', 'execution/summary.json'): bind_path(base/name)
    driver.verify_plan(base/'execution-plan.json', plan, inputs[str(base/'execution-plan.json')])
    for path, digest in plan['files'].items(): bind(dict(path=path, sha256=digest))
    binding = read(base/'run-binding.json')
    require(binding['config_sha256'] == inputs[str(base/'config.json')], 'Run config binding differs')
    prepared = read(bind(binding['prepared_manifest']))
    require(prepared['complete'] is True and prepared['passed'] is True, 'Incomplete inherited preparation')
    new = []
    for job in config['jobs']:
        terminal_path = Path(config['output'])/f"job-{job['id']:03}"/'terminal.json'
        matching = [x for x in plan['jobs'] if x['terminal']['path'] == str(terminal_path)]
        require(len(matching) == 1, 'Sampler terminal absent from frozen execution')
        execution_job = matching[0]
        terminal = read(bind(driver.completed_terminal(base, plan, execution_job['id'])))
        ordinal = plan['jobs'].index(execution_job); directory = driver.job_directory(base, ordinal, execution_job)
        require(read(directory/'success.json') == status['completed'][ordinal], 'Completed driver list differs')
        for name in ('attempt.json', 'process.json', 'exit.json', 'success.json'): bind_path(directory/name)
        require(terminal['complete'] is True and terminal['conditional_target'] is True and terminal['job'] == job
                and terminal['blocks'] == 4608 and terminal['config_sha256'] == inputs[str(base/'config.json')]
                and terminal['binding_sha256'] == inputs[str(base/'run-binding.json')]
                and not (terminal_path.parent/'failure.json').exists(), 'Incomplete/mismatched sampler terminal')
        require(finite(terminal['cpu_seconds'], 'sampler CPU') > 0, 'Zero sampler CPU')
        trajectory = bind(terminal['trajectory'])
        require(trajectory == terminal_path.parent/'trajectory.jsonl', 'Unexpected journal path')
        checkpoint = read(bind_path(terminal_path.parent/'checkpoint.json'))
        banks = [b for b in prepared['cloud_banks'] if all(b[k] == job[k] for k in ('context_index', 'initialization', 'stream'))]
        require(len(banks) == 1, 'Missing/ambiguous inherited cloud bank'); bank = banks[0]
        bind(bank['raw']); metadata = read(bind(bank['metadata'])); cloud_weight(metadata, config['cloud']['raw_count'])
        start = None
        if job['initialization'] == 'proposal_prepared':
            starts = [s for s in prepared['alternative_starts'] if all(s[k] == job[k] for k in ('context_index', 'stream'))]
            require(len(starts) == 1, 'Missing/ambiguous inherited start'); start = starts[0]
            bind(start['record'])
            if 'ledger' in start: bind(start['ledger'])
        new.append((job, terminal, trajectory, bank, metadata, start, checkpoint))
    cached = bind_cached_controls(config, bind)
    return inputs, new, cached, bind


def canonical_rows(rows, config, job, initial, cloud, metadata, prepared_manifest,
                   prepared_start=None, diagnostics=None):
    """Validate original rows then project only for legacy retained-state replay."""
    rows = iter(rows); first = next(rows, None)
    require(first is not None and first['kind'] == 'initial', 'Missing initial row')
    contract = surrogate_contract(config, job, cloud, metadata)
    require(first['partner_atlas_contract'] == contract and first['cloud'] == cloud
            and first['fixed_source'] == config['source_frame'] and first['prepared_manifest'] == prepared_manifest
            and first['prepared_start'] == prepared_start, 'Initial cloud/target/start contract differs')
    close(first['cloud_cpu_seconds'], metadata['cpu_seconds'], 'Cloud preparation timing differs')
    shared.poses(first['selected']); cpu = finite(first['sampler_cpu_seconds'], 'initial CPU', True)
    raw = retained = 0; yield first
    while True:
        row = next(rows, None)
        if row is None: return
        require(not any(k in row for k in ('fatal_error', 'error', 'bath_failure')), 'Fatal trace cannot be omitted')
        current_cpu = finite(row['sampler_cpu_seconds'], 'sampler CPU', True)
        require(current_cpu >= cpu, 'Nonmonotone sampler CPU'); cpu = current_cpu
        if row['kind'] == 'partner_atlas_attempt_begun':
            require(row['raw'] == raw and row['bath_retained'] == retained, 'Begun bath counters differ')
            begun = row; row = next(rows, None)
            require(row is not None and row['kind'] in ('flexible_partner_atlas_direct','flexible_partner_atlas_chain')
                    and not any(k in row for k in ('fatal_error', 'error', 'bath_failure')), 'Unmatched/fatal begun attempt')
            current_cpu = finite(row['sampler_cpu_seconds'], 'outcome CPU', True)
            require(current_cpu >= cpu, 'Nonmonotone outcome CPU'); cpu = current_cpu
            validate_surrogate(row, begun, config, job, contract)
            if diagnostics is not None: accumulate(diagnostics, row)
            projected = copy.deepcopy(row); projected['kind'] = 'factorized_dimer'
            projected['anchor'] = config['contexts'][job['context_index']]['anchor']
            if row['status'] == 'identity_self_loop':
                projected['status'] = 'proposal_self_loop'; projected['proposal'] = {'candidate': None}
            else:
                gate = row['bath']['aggregate']; raw += gate['raw_points']; retained += gate['retained_points']
            yield projected
        else:
            require(row['kind'] in ('local', 'retained_block'), 'Unexpected/missing begun journal row')
            if row['kind'] == 'local':
                shared.poses([row['old'], row['proposed']])
                if row['status'] == 'hard_rejected':
                    require(not any(k in row for k in ('bath', 'log_u', 'log_acceptance_ratio')), 'Hard local consumed bath')
                else:
                    require(row['status'] == 'completed', 'Incomplete local update')
                    decision(row, scalar.bath_weight(row['bath'], config))
                    require(row['bath']['raw_points'] <= min(config['limits']['raw_per_leg'], config['limits']['raw_per_outer'])
                        and row['bath']['retained_points'] <= min(config['limits']['retained_per_leg'], config['limits']['retained_per_outer']),
                        'Local completed bath exceeds cap')
                    raw += row['bath']['raw_points']; retained += row['bath']['retained_points']
                    require(raw <= config['limits']['raw_campaign'] and retained <= config['limits']['retained_campaign'],
                            'Local campaign budget exceeds cap')
            else: shared.poses(row['selected'])
            yield row


def validate_journal(rows, config, job, initial, terminal, cloud, metadata,
                     prepared_manifest, prepared_start=None, diagnostics=None):
    return previous.validate_journal(canonical_rows(rows, config, job, initial, cloud, metadata,
        prepared_manifest, prepared_start, diagnostics), config, job, initial, terminal)


def analyze(base, output):
    from prepare_partner_atlas_benchmark import verify
    base, output = Path(base).resolve(), Path(output).resolve()
    config = verify(base); validate_inventory(config)
    require(not output.exists(), 'Fresh analysis output required')
    require(read(base/'analysis-plan.json') == analysis_plan(), 'Frozen analysis plan differs')
    protocol = read(base/'protocol.json'); repo = Path(__file__).parents[1]
    for name in SOURCE_FILES: require(sha(repo/name) == protocol['source_files'][name], 'Observer source changed '+name)
    inputs, new, cached, bind = bind_complete_inputs(base, config)
    binding = read(base/'run-binding.json')
    shape = read(bind(config['shape'])); patch = read(bind(config['patch_map']))
    source = read(bind(config['source_frame']))['poses']
    require(len(source) == 264 and patch['shape_sha256'] == config['shape']['sha256']
            and len(patch['atom_patch_ids']) == len(shape['atoms']), 'Source/shape/patch identity differs')
    observer_config = dict(boundary=dict(kind='spherical', radius=config['physical']['wall_radius']),
        box_lengths=[2*config['physical']['wall_radius']]*3, depletant_radius=config['physical']['depletant_radius'])
    output.mkdir(); started = time.process_time(); chains = []; endpoints = 0
    write(output/'input-binding.json', dict(input_sha256=inputs, analysis_plan=analysis_plan(),
        source_sha256={n: protocol['source_files'][n] for n in SOURCE_FILES}))
    try:
        for job, terminal, path, cloud, metadata, start, checkpoint in new:
            tick = time.process_time(); case = config['contexts'][job['context_index']]
            members = [case['root'], case['child']]
            initial = [source[i] for i in members] if start is None else read(bind(start['record']))['selected']
            def rows(stream):
                for line in stream:
                    require(line.endswith('\n') and len(line) <= analysis_plan()['observer_limits']['max_record_bytes'], 'Truncated/oversized journal row')
                    yield json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
            diagnostics = dict(warmup_blocks=config['allocation']['warmup_blocks'])
            with path.open() as stream:
                points = validate_journal(rows(stream), config, job, initial, terminal, cloud, metadata,
                                          binding['prepared_manifest'], start, diagnostics)
            validate_checkpoint(checkpoint, terminal, points[-1], path, job)
            observer = previous.ConditionalObserver(shape, patch['atom_patch_ids'], observer_config, source, members)
            trace = []; prior = None; warmup = config['allocation']['warmup_blocks']
            with (output/f"job-{job['id']:03}-observations.jsonl").open('x') as handle:
                for point in points:
                    observation = observer.classify(point['selected']); tokens = set(observation['patch_tokens'])
                    old = set(prior['patch_tokens']) if prior else tokens; union = old | tokens
                    observation.update(block=point['block'], production=point['block'] > warmup,
                        sampler_cpu_seconds=point['sampler_cpu_seconds'], jaccard_from_previous=1-len(old & tokens)/len(union) if union else 0.)
                    handle.write(canonical(observation)+'\n'); trace.append(observation); prior = observation
            metrics = previous.summarize_trace(trace, warmup, terminal['cpu_seconds'])
            metrics['production_cpu_seconds'] = points[-1]['sampler_cpu_seconds']-points[warmup]['sampler_cpu_seconds']
            metrics['external_only'] = external.external_metrics(trace, warmup, terminal['cpu_seconds'], members)
            metrics['external_activity'] = shared.external_activity(trace, warmup, terminal['cpu_seconds'])
            metrics['internal_organization'] = internal_metrics(trace, warmup, terminal['cpu_seconds'], members)
            metrics['relative_pose'] = relative_metrics(points,warmup,terminal['cpu_seconds'])
            chains.append(dict(job=job, metrics=metrics, reused_control=False, trajectory=terminal['trajectory'],
                counts=terminal['counts'], surrogate_diagnostics=finish_diagnostics(diagnostics),
                observer_cpu_seconds=time.process_time()-tick, pair_classifications=observer.calls,
                exact_pair_cache_hits=observer.hits, geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds'),
                cloud_cpu_seconds=metadata.get('cpu_seconds')))
            endpoints += len(points)
        for control, path in cached:
            c = copy.deepcopy(control); job = c['job']; case = config['contexts'][job['context_index']]
            members = [case['root'], case['child']]
            trace = external.read_cached_trace(path, inputs[str(path)], 512, 4608)
            cpu = c['metrics']['full_sampler_cpu_seconds']
            c['metrics']['external_only'] = external.external_metrics(trace, 512, cpu, members)
            c['metrics']['external_activity'] = shared.external_activity(trace, 512, cpu)
            c['metrics']['internal_organization'] = internal_metrics(trace, 512, cpu, members)
            c.update(reused_control=True, new_geometry_queries=0); chains.append(c)
        require(endpoints == analysis_plan()['new_retained_initial_observations'] and len(chains) == 72, 'Incomplete analysis inventory')
        require(sum(c['pair_classifications'] for c in chains if not c['reused_control']) <= analysis_plan()['maximum_new_pair_classifications'],
                'Observer query allocation exceeded')
        for path, digest in inputs.items(): require(sha(path) == digest, 'Input changed during observation '+path)
        for name in SOURCE_FILES: require(sha(repo/name) == protocol['source_files'][name], 'Observer source changed '+name)
        comparisons = comparison_summaries(chains)
        for pair in comparisons['descriptive_paired_comparisons']:
            same_start = pair['left']['initialization'] == pair['right']['initialization']
            pair['local_rng_roles_paired'] = same_start
            pair['partner_atlas_rng_roles_paired'] = same_start and {pair['left']['arm'], pair['right']['arm']} <= set(ARMS)
            pair['rigid_surrogate_rng_roles_paired'] = same_start and {pair['left']['arm'], pair['right']['arm']} <= set(rigid.ARMS)
            pair['control_global_rng_roles_paired'] = False
        result = dict(schema='partner-atlas-analysis-v1', complete=True, new_chains=32, reused_control_chains=40,
            chains=chains, comparisons=comparisons, input_sha256=inputs,
            source_sha256={n: protocol['source_files'][n] for n in SOURCE_FILES}, analysis_plan=analysis_plan(),
            costs=dict(new_analysis_cpu_seconds=time.process_time()-started,
                new_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if not c['reused_control']),
                reused_control_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if c['reused_control']),
                shared_preparation_not_repeated=True), new_geometry_endpoints=endpoints,
            old_geometry_queries=0, new_physical_draws=0, native_observer=False)
        write(output/'analysis.json', result)
        write(output/'manifest.json', dict(complete=True, files={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
        return result
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, error=repr(error), completed_chains=len(chains),
            completed_endpoints=endpoints, input_sha256=inputs, retries=0)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.base, args.output)
    print(json.dumps({k: result[k] for k in ('complete', 'new_chains', 'reused_control_chains')}))
