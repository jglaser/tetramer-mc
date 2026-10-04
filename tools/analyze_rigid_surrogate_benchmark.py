#!/usr/bin/env python3
"""Completed-journal contact observer for the fixed rigid-surrogate controls.

The scalar audit checks every saved inner step and its outer correction. It does
not reproduce RNG, atom predicates, cloud membership or filesystem durability.
Only a transient projection is passed to the unchanged state replay. All new
retained endpoints are classified; cached controls are included exactly once.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time

import analyze_evolving_dimer_benchmark as previous
import analyze_root_guided_dimer_benchmark as external
import analyze_two_neighbor_singleton_benchmark as scalar
import analyze_two_root_dimer_benchmark as shared

require, read, sha, canonical, write = (previous.require, previous.read, previous.sha,
                                      previous.canonical, previous.write)
ARMS = ('rigid_surrogate_1', 'rigid_surrogate_8', 'rigid_surrogate_flat8')
STARTS = ('source', 'proposal_prepared')
CONTROLS = ('local', 'm4')
SOURCE_FILES = list(dict.fromkeys(['tools/analyze_rigid_surrogate_benchmark.py',
    'tools/test_analyze_rigid_surrogate_benchmark.py'] + shared.SOURCE_FILES))


def analysis_plan():
    return dict(schema='rigid-surrogate-analysis-plan-v1', source_files=SOURCE_FILES,
        new_chains=24, reused_control_chains=16, reused_control_arms=list(CONTROLS), context_indices=[0],
        new_retained_initial_observations=24*4609, new_production_observations=24*4096,
        scalar_local_attempts=24*4608*4, scalar_surrogate_attempts=24*4608,
        maximum_mobile_related_pairs_per_endpoint=525, maximum_new_pair_classifications=24*4609*525,
        observer_limits=dict(threads=1, cpu_limit_seconds=3600, wall_limit_seconds=7200,
                             address_space_limit_bytes=16*2**30, max_record_bytes=16*2**20),
        original_plan=previous.analysis_plan(), external_plan=external.analysis_plan(),
        target='Original context0 two-mobile conditional target. Four unchanged locals precede one rigid-fiber proposal. No fixed internal geometry is assumed between blocks.',
        replay='Every begun/outcome pair and fixed-length inner trace; saved score units, pose-only rigid reconstruction, scalar MH, physical counts and all rejected residence.',
        trust_boundary='No RNG, atom hard predicates, cloud membership or Poisson predicates reconstructed. Begun-row presence is checked; fsync durability remains a source/runtime obligation.',
        comparisons='Four streams per arm and start; no pooling starts. Shared local and surrogate role seeds do not make arms independent replicates. Cached controls included once.',
        metric_names='Patch/edge presence ESS is distinct from categorical fingerprint ESS. Constant/null ESS remains undefined.',
        new_geometry_only=True, old_geometry_queries=0, new_physical_draws=0, native_observer=False,
        complete_inventory_required=True, trajectories_concatenated=False, assembly_gate_open=False)


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), 'Invalid '+label)
    return value


def close(actual, expected, label):
    finite(actual, label); finite(expected, label)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), label)


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def effective_config(config, arm):
    require(arm in ARMS, 'Unknown rigid-surrogate arm')
    policy = config['surrogate_policy']
    require(set(policy) == {'schema', 'proposal_scales'}
            and policy['schema'] == 'rigid-surrogate-policy-v1', 'Changed surrogate policy')
    scales = policy['proposal_scales']
    if scales.get('source') == 'local':
        require(set(scales) == {'source'}, 'Unexpected local-scale override')
        dt, dr = (config['local'][k] for k in ('translation_std_A', 'rotation_std_degrees'))
    else:
        require(set(scales) == {'source', 'translation_std_A', 'rotation_std_degrees'}
                and scales['source'] == 'frozen_override', 'Malformed frozen scale policy')
        dt, dr = (scales[k] for k in ('translation_std_A', 'rotation_std_degrees'))
    finite(dt, 'translation scale', True); finite(dr, 'rotation scale', True)
    return dict(inner_steps=1 if arm == ARMS[0] else 8, translation_std=dt,
                rotation_std_degrees=dr, guidance_strength=0. if arm == ARMS[2] else 1.)


def validate_inventory(config):
    expected = {(0, a, i, s) for a in ARMS for i in STARTS for s in range(4)}
    jobs = config['jobs']
    require(len(jobs) == 24 and len({j['id'] for j in jobs}) == 24
            and {identity(j) for j in jobs} == expected, 'Changed complete 24-chain inventory')
    require(config['physical'] == scalar.PHYSICAL and config['allocation']['warmup_blocks'] == 512
            and config['allocation']['production_blocks'] == 4096, 'Changed target or schedule')
    require('singleton_policy' not in config and 'two_root_policy' not in config, 'Another proposal policy present')
    for arm in ARMS: effective_config(config, arm)


def cloud_weight(meta, raw_count):
    require(type(raw_count) is int and raw_count > 0 and meta['raw_count'] == raw_count,
            'Raw quadrature count differs')
    low, high = meta['low'], meta['high']
    require(len(low) == len(high) == 3, 'Malformed raw box')
    widths = []
    for a, b in zip(low, high):
        finite(a, 'box lower bound'); finite(b, 'box upper bound')
        require(b > a, 'Empty raw box'); widths.append(b-a)
    indices = meta['kept_indices']
    require(type(indices) is list and all(type(i) is int and 0 <= i < raw_count for i in indices)
            and indices == sorted(set(indices)), 'Invalid kept point indices')
    volume = math.prod(widths)/raw_count
    finite(volume, 'point volume'); require(volume > 0, 'Nonpositive point volume')
    return len(indices), volume


def surrogate_contract(config, job, cloud, meta):
    case = config['contexts'][job['context_index']]; members = [case['root'], case['child']]
    n, v = cloud_weight(meta, config['cloud']['raw_count'])
    return dict(schema='evolving-dimer-rigid-surrogate-v1', policy=config['surrogate_policy'],
        effective_config=effective_config(config, job['arm']), members=members, handle=members[0],
        fixed_spectators='all other labels', cloud=cloud,
        cloud_reuse='identical frozen body-frame points for both members; no extra draws',
        raw_count=meta['raw_count'], points_per_body=n, point_volume=v,
        point_weight='raw_box_volume/raw_count', wall_center=[0., 0., 0.],
        proposal_rng_role='rigid_surrogate/proposal', inner_accept_rng_role='rigid_surrogate/inner_accept',
        bath_rng_role='rigid_surrogate/bath', accept_rng_role='rigid_surrogate/accept',
        local_schedule='canonical members [0,1,0,1]; shared unchanged local RNG roles',
        collective_slots_per_block=1, physical_decisions_per_nonidentity_endpoint=1,
        identity_endpoint='retained self-loop without bath', inner_rejections='consume a step and retain current state',
        outer_correction='S(old)-S(new)', journal_rows_per_block=7)


def validate_score(value, contract, config, spectator_count=262):
    keys = {'points_per_body', 'spectator_covered', 'internal_unshielded', 'nearby_spectators',
            'spectator_membership_queries', 'internal_membership_queries', 'twice_overlap_units',
            'overlap_volume_estimate', 'log_surrogate'}
    require(set(value) == keys, 'Malformed score fields')
    n = scalar.nat(value['points_per_body'], 'points per body')
    require(n == contract['points_per_body'], 'Score cloud differs')
    for key in ('spectator_covered', 'internal_unshielded', 'nearby_spectators'):
        require(type(value[key]) is list and len(value[key]) == 2, 'Malformed score count pair')
        for count in value[key]: scalar.nat(count, key)
    covered, internal, nearby = (value[k] for k in ('spectator_covered', 'internal_unshielded', 'nearby_spectators'))
    require(all(a+b <= n and c <= spectator_count and (c != 0 or a == 0)
                for a, b, c in zip(covered, internal, nearby)), 'Impossible score membership counts')
    sq = scalar.nat(value['spectator_membership_queries'], 'spectator queries')
    iq = scalar.nat(value['internal_membership_queries'], 'internal queries')
    require(n*sum(c > 0 for c in nearby) <= sq <= n*sum(nearby)
            and iq == 2*n-sum(covered), 'Score query counts differ')
    units = 2*sum(covered)+sum(internal)
    require(scalar.nat(value['twice_overlap_units'], 'overlap units') == units, 'Score overlap units differ')
    volume = .5*contract['point_volume']*units
    close(value['overlap_volume_estimate'], volume, 'Score raw-count volume differs')
    strength = contract['effective_config']['guidance_strength']
    close(value['log_surrogate'], config['physical']['activity']*strength*volume, 'Score exponent differs')
    return value['log_surrogate']


def rotation(pose):
    w, x, y, z = previous.np.asarray(pose['orientation'], float)
    d = math.sqrt(w*w+x*x+y*y+z*z); w, x, y, z = (v/d for v in (w, x, y, z))
    return previous.np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def validate_rigid(old, proposed_handle, target):
    """Pose algebra only. Atomic hard and score membership remain trusted."""
    shared.poses(old); shared.poses(target); shared.poses([proposed_handle, proposed_handle])
    require(target[0] == proposed_handle, 'Rigid handle projection differs')
    delta = rotation(proposed_handle) @ rotation(old[0]).T
    expected = previous.np.asarray(proposed_handle['position']) + delta @ (
        previous.np.asarray(old[1]['position'])-previous.np.asarray(old[0]['position']))
    scale = 1+max(abs(x) for p in old+target for x in p['position'])
    require(previous.np.max(previous.np.abs(expected-target[1]['position'])) <= 1e-10*scale
            and previous.np.max(previous.np.abs(delta @ rotation(old[1])-rotation(target[1]))) <= 1e-10,
            'Inner endpoint violates fixed outer-source rigidity')


def decision(row, ratio):
    finite(ratio, 'MH log ratio'); finite(row['log_u'], 'MH log uniform')
    close(row['log_acceptance_ratio'], ratio, 'MH factor sum differs')
    require(row['log_u'] < 0 and type(row['accepted']) is bool
            and row['accepted'] == (row['log_u'] < min(0., ratio)), 'MH decision differs')


def validate_surrogate(row, begun, config, job, contract):
    members, settings = contract['members'], contract['effective_config']
    require(row['members'] == begun['members'] == members and row['handle'] == begun['handle'] == members[0]
            and row['config'] == begun['config'] == settings and row['old'] == begun['old']
            and row['block'] == begun['block'] and begun['kind'] == 'rigid_surrogate_attempt_begun'
            and begun['status'] == 'begun', 'Begun/outer identity differs')
    old = shared.poses(row['old']); current = copy.deepcopy(old)
    initial_score = row['old_score']; old_s = validate_score(initial_score, contract, config)
    current_score = initial_score; steps = row['steps']
    require(type(steps) is list and len(steps) == settings['inner_steps'], 'Fixed inner horizon differs')
    for index, step in enumerate(steps):
        require(type(step['index']) is int and step['index'] == index and step['old_handle'] == current[0]
                and type(step['accepted']) is bool, 'Inner source/order differs')
        close(step['old_score'], current_score['log_surrogate'], 'Inner source score differs')
        target = step['proposed']; validate_rigid(old, step['proposed_handle'], target)
        if step['status'] == 'hard_rejected':
            require(step['accepted'] is False and not any(k in step for k in
                ('proposed_score', 'log_u', 'log_acceptance_ratio')), 'Hard rejection consumed a score/MH decision')
        else:
            require(step['status'] == 'completed', 'Incomplete/fatal inner trace')
            new_s = validate_score(step['proposed_score'], contract, config)
            decision(step, new_s-current_score['log_surrogate'])
            if step['accepted']:
                current = copy.deepcopy(target); current_score = step['proposed_score']
        require(step['retained'] == current, 'Inner rejection/retained state differs')
    require(row['proposed'] == current and row['proposed_score'] == current_score, 'Final inner endpoint/score differs')
    correction = old_s-current_score['log_surrogate']
    close(row['complete_log_correction'], correction, 'Outer surrogate correction differs')
    require(type(row['accepted']) is bool and type(row['physical_decisions']) is int,
            'Invalid outer acceptance/decision count')
    if current == old:
        require(row['status'] == 'identity_self_loop' and row['accepted'] is False
                and row['physical_decisions'] == 0 and not any(k in row for k in
                    ('bath', 'log_u', 'log_acceptance_ratio')), 'Identity endpoint consumed a physical gate')
    else:
        require(row['status'] == 'completed' and row['physical_decisions'] == 1,
                'Nonidentity endpoint lacks exactly one physical decision')
        bath = scalar.bath_weight(row['bath'], config)
        decision(row, bath+correction)
    require(row['retained'] == (current if row['accepted'] else old), 'Outer rejection/retained state differs')


def canonical_rows(rows, config, job, initial, cloud, metadata, prepared_manifest,
                   prepared_start=None, diagnostics=None):
    """Validate original rows then project only for legacy retained-state replay."""
    rows = iter(rows); first = next(rows, None)
    require(first is not None and first['kind'] == 'initial', 'Missing initial row')
    contract = surrogate_contract(config, job, cloud, metadata)
    require(first['surrogate_contract'] == contract and first['cloud'] == cloud
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
        if row['kind'] == 'rigid_surrogate_attempt_begun':
            require(row['raw'] == raw and row['bath_retained'] == retained, 'Begun bath counters differ')
            begun = row; row = next(rows, None)
            require(row is not None and row['kind'] == 'rigid_surrogate_chain'
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
                gate = row['bath']; raw += gate['raw_points']; retained += gate['retained_points']
                projected['bath'] = {'aggregate': gate}
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
                    raw += row['bath']['raw_points']; retained += row['bath']['retained_points']
            else: shared.poses(row['selected'])
            yield row


def validate_journal(rows, config, job, initial, terminal, cloud, metadata,
                     prepared_manifest, prepared_start=None, diagnostics=None):
    return previous.validate_journal(canonical_rows(rows, config, job, initial, cloud, metadata,
        prepared_manifest, prepared_start, diagnostics), config, job, initial, terminal)


def accumulate(diagnostics, row):
    phase = 'warmup' if row['block'] <= diagnostics['warmup_blocks'] else 'production'
    target = diagnostics.setdefault(phase, dict(outer_attempts=0, outer_accepted=0, identity_self_loops=0,
        physical_decisions=0, inner_attempts=0, inner_accepted=0, inner_hard_rejected=0,
        score_spectator_queries=0, score_internal_queries=0, candidate_factors=[]))
    target['outer_attempts'] += 1; target['outer_accepted'] += row['accepted']
    target['identity_self_loops'] += row['status'] == 'identity_self_loop'
    target['physical_decisions'] += row['physical_decisions']
    scores = [row['old_score']]
    for step in row['steps']:
        target['inner_attempts'] += 1; target['inner_accepted'] += step['accepted']
        target['inner_hard_rejected'] += step['status'] == 'hard_rejected'
        if 'proposed_score' in step: scores.append(step['proposed_score'])
    for score in scores:
        target['score_spectator_queries'] += score['spectator_membership_queries']
        target['score_internal_queries'] += score['internal_membership_queries']
    if row['physical_decisions']:
        target['candidate_factors'].append(dict(correction=row['complete_log_correction'],
            bath=row['bath']['log_weight'], total=row['log_acceptance_ratio']))


def finish_diagnostics(diagnostics):
    result = copy.deepcopy(diagnostics)
    for phase in ('warmup', 'production'):
        target = result.get(phase)
        if target is None: continue
        values = target.pop('candidate_factors')
        target['candidate_factor_summary'] = {k: dict(count=len(values),
            mean=math.fsum(v[k] for v in values)/len(values) if values else None,
            minimum=min((v[k] for v in values), default=None),
            maximum=max((v[k] for v in values), default=None)) for k in ('correction', 'bath', 'total')}
    result['scope'] = 'Saved scalar counts and candidate-only factors; no physical free energy, counterfactual acceptance or geometry certification.'
    return result


def bind_complete_inputs(base, config):
    """All 24 terminals must close before journals or cached observations are read."""
    import run_native_class_physical_campaign as driver
    base = Path(base).resolve(); inputs = {}
    def bind(ref):
        path = Path(ref['path']).resolve()
        require(sha(path) == ref['sha256'], 'Changed input '+str(path))
        require(str(path) not in inputs or inputs[str(path)] == ref['sha256'], 'Conflicting binding')
        inputs[str(path)] = ref['sha256']; return path
    def bind_path(path):
        return bind(dict(path=str(Path(path).resolve()), sha256=sha(path)))
    # The cheap lifecycle gate precedes potentially large journal hashing.
    plan = read(base/'execution-plan.json'); status = read(base/'execution/status.json')
    summary = read(base/'execution/summary.json')
    require(all(status[k] == summary[k] for k in ('complete', 'passed', 'failure', 'active',
                'unstarted', 'completed', 'plan_sha256'))
            and status['complete'] is True and status['passed'] is True and status['failure'] is None
            and status['active'] is None and status['unstarted'] == []
            and len(status['completed']) == len(plan['jobs']) == 24
            and not (base/'execution/failure.json').exists(), 'All 24 sampler jobs must complete cleanly')
    for name in ('config.json', 'run-binding.json', 'analysis-plan.json', 'protocol.json', 'execution-plan.json',
                 'execution/claim.json', 'execution/status.json', 'execution/summary.json'):
        bind_path(base/name)
    driver.verify_plan(base/'execution-plan.json', plan, inputs[str(base/'execution-plan.json')])
    for path, digest in plan['files'].items(): bind(dict(path=path, sha256=digest))
    binding = read(base/'run-binding.json')
    require(binding['config_sha256'] == inputs[str(base/'config.json')], 'Run config binding differs')
    prepared = read(bind(binding['prepared_manifest']))
    require(prepared['complete'] is True and prepared['passed'] is True, 'Incomplete inherited preparation')
    controls = config['control_analysis']; old = read(bind(controls['analysis']))
    cs = read(bind(controls['summary'])); cm = read(bind(controls['manifest']))
    require(cs['complete'] is True and cs['passed'] is True and cm['complete'] is True
            and old['complete'] is True and cs['analysis']['sha256'] == controls['analysis']['sha256']
            and cs['manifest_sha256'] == controls['manifest']['sha256'], 'Cached control receipt chain differs')
    old_chains = {c['job']['id']: c for c in old['chains']}
    require(len(old_chains) == len(old['chains']), 'Duplicate historical identity')
    mapping = {m['job_id']: m for m in config['control_mapping']}
    require(len(mapping) == len(config['control_mapping']) == 24
            and set(mapping) == {j['id'] for j in config['jobs']}, 'Changed control mapping inventory')
    new, cached, used = [], [], {}
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
        finite(terminal['cpu_seconds'], 'full sampler CPU'); require(terminal['cpu_seconds'] > 0, 'Zero sampler CPU')
        trajectory = bind(terminal['trajectory'])
        require(trajectory == terminal_path.parent/'trajectory.jsonl', 'Unexpected journal path')
        banks = [b for b in prepared['cloud_banks'] if all(b[k] == job[k] for k in ('context_index', 'initialization', 'stream'))]
        require(len(banks) == 1, 'Missing/ambiguous inherited cloud bank'); bank = banks[0]
        bind(bank['raw']); metadata = read(bind(bank['metadata']))
        cloud_weight(metadata, config['cloud']['raw_count'])
        start = None
        if job['initialization'] == 'proposal_prepared':
            starts = [s for s in prepared['alternative_starts'] if all(s[k] == job[k] for k in ('context_index', 'stream'))]
            require(len(starts) == 1, 'Missing/ambiguous inherited start'); start = starts[0]
            bind(start['record'])
            if 'ledger' in start: bind(start['ledger'])
        new.append((job, terminal, trajectory, bank, metadata, start))
        link = mapping[job['id']]
        require(set(link['controls']) == set(CONTROLS) and all(link[k] == job[k] for k in ('context_index', 'initialization', 'stream')),
                'Matched control identity differs')
        for arm, old_id in link['controls'].items():
            control = old_chains[old_id]; cj = control['job']
            require(cj['arm'] == arm and all(cj[k] == job[k] for k in ('context_index', 'initialization', 'stream')),
                    'Wrong matched cached control')
            expected = identity(cj)
            require(old_id not in used or used[old_id] == expected, 'Control identity reused inconsistently')
            if old_id in used: continue  # Shared across arms, never three independent copies.
            used[old_id] = expected
            ref = controls['observations'][str(old_id)]; path = bind(ref)
            require(cm['files'][path.name] == ref['sha256'], 'Cached observations lack original binding')
            cached.append((control, path))
    require(len(cached) == 16 and {identity(c['job']) for c, _ in cached} ==
            {(0, a, i, s) for a in CONTROLS for i in STARTS for s in range(4)}, 'Incomplete unique cached control inventory')
    return inputs, new, cached, bind


def analyze(base, output):
    from prepare_rigid_surrogate_benchmark import verify
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
        for job, terminal, path, cloud, metadata, start in new:
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
            c.update(reused_control=True, new_geometry_queries=0); chains.append(c)
        require(endpoints == analysis_plan()['new_retained_initial_observations'] and len(chains) == 40, 'Incomplete analysis inventory')
        require(sum(c['pair_classifications'] for c in chains if not c['reused_control']) <= analysis_plan()['maximum_new_pair_classifications'],
                'Observer query allocation exceeded')
        for path, digest in inputs.items(): require(sha(path) == digest, 'Input changed during observation '+path)
        for name in SOURCE_FILES: require(sha(repo/name) == protocol['source_files'][name], 'Observer source changed '+name)
        comparisons = external.comparison_summaries(chains)
        for pair in comparisons['descriptive_paired_comparisons']:
            same_start = pair['left']['initialization'] == pair['right']['initialization']
            pair['local_rng_roles_paired'] = same_start
            pair['surrogate_rng_roles_paired'] = same_start and {pair['left']['arm'], pair['right']['arm']} <= set(ARMS)
            pair['control_global_rng_roles_paired'] = False
        result = dict(schema='rigid-surrogate-analysis-v1', complete=True, new_chains=24, reused_control_chains=16,
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
    parser.add_argument('--base', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.base, args.output)
    print(json.dumps({k: result[k] for k in ('complete', 'new_chains', 'reused_control_chains')}))
