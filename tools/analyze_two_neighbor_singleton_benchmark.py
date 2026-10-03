#!/usr/bin/env python3
"""Frozen contact observer for the two-neighbor singleton control.

Replay every elementary update before classifying retained endpoints. Only the
64 new chains reach atom geometry; the 64 local/m4 controls reuse their bound
cached observations. Scalar proposal checks do not independently reconstruct
the learned density, raw RNG, or physical Poisson predicates.
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

require, read, sha, canonical, write = (previous.require, previous.read,
    previous.sha, previous.canonical, previous.write)
ARMS = ('singleton_two_neighbor', 'singleton_two_neighbor_unfused')
CONTROLS = ('local', 'm4')
STARTS = ('source', 'proposal_prepared')
PHYSICAL = dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64.,
                wall_radius=593.742500239952)
SOURCE_FILES = list(dict.fromkeys([
    'tools/analyze_two_neighbor_singleton_benchmark.py',
    'tools/test_analyze_two_neighbor_singleton_benchmark.py',
    'tools/analyze_root_guided_dimer_benchmark.py',
    'tools/test_analyze_root_guided_dimer_benchmark.py'] + previous.SOURCE_FILES))


def analysis_plan():
    return dict(schema='two-neighbor-singleton-analysis-plan-v1', source_files=SOURCE_FILES,
        new_chains=64, reused_control_chains=64, reused_control_arms=list(CONTROLS),
        new_retained_initial_observations=294976, new_production_observations=262144,
        scalar_local_attempts=1179648, scalar_singleton_attempts=294912,
        maximum_mobile_related_pairs_per_endpoint=525,
        maximum_new_pair_classifications=154862400,
        original_plan=previous.analysis_plan(), external_plan=external.analysis_plan(),
        schedule='Four unchanged local updates in slots 0,1,0,1, then one singleton in slot (block-1)%2. Both labels remain mobile.',
        target='Two mobile labels and 262 fixed spectators; radius 1.4 A, activity .0275 A^-3. Contexts are different conditional targets.',
        observations='Initial and every retained block, including rejected residence, source-zero-flow and cap-exhausted self-loops. No accepted-only selection.',
        controls='Reuse exactly the completed local and m4 chains/cached observations; no old geometry and no root_m4 controls.',
        comparison='Per-context/per-initialization/per-stream metrics and descriptive paired differences. Shared local RNG roles; singleton global roles are shared only between the two new arms.',
        trust_boundary='All rows receive scalar state, complete-mixture arithmetic, first-success stopping, count and one-MH replay. Learned component densities, geometry predicates, RNG and physical Poisson predicates are not independently reconstructed here.',
        new_geometry_only=True, old_geometry_queries=0, new_physical_draws=0,
        native_observer=False, complete_inventory_required=True,
        trajectories_concatenated=False, assembly_gate_open=False)


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def validate_inventory(config):
    jobs = config['jobs']
    expected = {(c, a, i, s) for c in range(4) for a in ARMS for i in STARTS for s in range(4)}
    require(len(jobs) == 64 and len({j['id'] for j in jobs}) == 64
        and {identity(j) for j in jobs} == expected, 'Changed complete 64-chain inventory')
    require(len(config['contexts']) == 4 and config['physical'] == PHYSICAL,
            'Changed conditional physical target')
    require(config['allocation']['warmup_blocks'] == 512
        and config['allocation']['production_blocks'] == 4096, 'Changed production schedule')
    policy = config['singleton_policy']
    require(policy['schema'] == 'two-neighbor-singleton-policy-v1'
        and policy['uniform_probability'] == .5 and policy['uniform_half_width'] == 160.
        and policy['trial_cap'] == 32 and policy['member_schedule'] == 'alternating_0_first'
        and policy['oligomer']['multi_contact_mass'] == .8, 'Changed singleton proposal policy')


def effective_oligomer(config, arm):
    require(arm in ARMS, 'Unexpected singleton arm')
    value = copy.deepcopy(config['singleton_policy']['oligomer'])
    if arm == ARMS[1]: value['multi_contact_mass'] = 0.
    return value


def nat(value, label):
    require(type(value) is int and value >= 0, 'Invalid '+label)
    return value


def log_value(value):
    if value == '-inf': return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid saved log factor')
    return float(value)


def close(actual, expected, label):
    a = log_value(actual)
    require(a == expected or (math.isfinite(a) and math.isfinite(expected)
        and math.isclose(a, expected, rel_tol=1e-10, abs_tol=1e-8)), label)


def density(value, half_width):
    require(set(value) == {'log_uniform', 'log_learned', 'log_full'}, 'Incomplete density fields')
    u, learned, full = (log_value(value[k]) for k in ('log_uniform', 'log_learned', 'log_full'))
    require(u == -math.inf or math.isclose(u, -3*math.log(2*half_width), rel_tol=1e-10, abs_tol=1e-8),
            'Uniform density is not normalized translation times Haar')
    largest = max(u, learned)
    expected = -math.inf if largest == -math.inf else largest + math.log(math.exp(u-largest)+math.exp(learned-largest))-math.log(2.)
    close(value['log_full'], expected, 'Incomplete defensive mixture density')
    return full


def hard_valid(value):
    require(set(value) == {'spectator_core_collisions', 'wall_valid'}
        and type(value['wall_valid']) is bool and isinstance(value['spectator_core_collisions'], list),
        'Invalid scalar hard predicate')
    collisions = value['spectator_core_collisions']
    require(all(type(i) is int and i >= 0 for i in collisions) and collisions == sorted(set(collisions)),
            'Malformed spectator labels')
    return value['wall_valid'] and not collisions


def bath_weight(gate, config):
    for key in ('gained', 'lost', 'raw_points', 'retained_points', 'created_cells', 'retained_cells'):
        nat(gate[key], 'bath '+key)
    require(gate['retained_points'] == gate['gained']+gate['lost'] <= gate['raw_points']
        and gate['retained_cells'] <= gate['created_cells'], 'Physical bath accounting differs')
    require(type(gate['envelope_volume']) in (int, float) and math.isfinite(gate['envelope_volume'])
        and gate['envelope_volume'] >= 0, 'Invalid bath envelope volume')
    coefficient = math.log1p(1/config['physical']['lambda_ratio'])
    value = coefficient*(gate['gained']-gate['lost'])
    close(gate['log_weight'], value, 'Physical gained/lost correction differs')
    return value


def validate_singleton(row, config, job, selected, fixed_anchor):
    """Validate saved scalars only; no shape, proposal object or geometry call."""
    slot = (row['block']-1) % 2
    context = config['contexts'][job['context_index']]
    members = [context['root'], context['child']]
    neighbors = [members[1-slot], context['anchor']]
    require(row['member_slot'] == slot and row['member'] == members[slot]
        and row['neighbors'] == neighbors and row['old'] == selected[slot], 'Singleton context/state replay differs')
    p = row['proposal']; policy = config['singleton_policy']
    require(p['moving'] == members[slot] and p['neighbors'] == neighbors and p['old'] == selected[slot]
        and p['trial_cap'] == policy['trial_cap'] and p['uniform_probability'] == .5
        and p['uniform_half_width'] == policy['uniform_half_width']
        and p['uniform_frame'] == fixed_anchor and p['uniform_center'] == fixed_anchor['position'],
        'Singleton proposal frame/policy differs')
    require(hard_valid(p['source_feasibility']), 'Singleton source is hard invalid')
    trials = p['trials']; require(isinstance(trials, list) and len(trials) <= p['trial_cap'], 'Singleton cap exceeded')
    counts = dict(uniform=0, learned=0, geometric_rejections=0, hard_rejections=0,
                  wall_rejections=0, candidates=0, fatal_trials=0)
    found = None
    for index, trial in enumerate(trials, 1):
        require(found is None and trial['index'] == index, 'Hidden retry after first feasible singleton')
        coin = trial['branch_uniform']
        require(type(coin) in (int, float) and math.isfinite(coin) and 0 <= coin < 1,
                'Invalid defensive branch coin')
        branch = 'uniform' if coin < .5 else 'learned'
        require(trial['branch'] == branch, 'Defensive branch differs')
        counts[branch] += 1
        require(trial['proposed_pose'] is not None and isinstance(trial['trace'], dict), 'Missing begun trial/pose trace')
        require(math.isfinite(density(trial['density'], policy['uniform_half_width'])), 'Generated pose has zero proposal density')
        feasible = hard_valid(trial['feasibility'])
        if feasible:
            require(trial['disposition'] == 'candidate', 'Feasible singleton was retried')
            counts['candidates'] += 1; found = trial
        else:
            require(trial['disposition'] == 'geometric_rejection', 'Invalid singleton stopping disposition')
            counts['geometric_rejections'] += 1
            counts['hard_rejections'] += bool(trial['feasibility']['spectator_core_collisions'])
            counts['wall_rejections'] += not trial['feasibility']['wall_valid']
    require(p['counts'] == counts, 'Singleton proposal trial counters differ')
    for key, value in p['counts'].items(): nat(value, 'proposal counter '+key)
    if p['trial_cap'] == 0:
        require(p['status'] == 'cap_zero' and p['old_density'] is None and not trials, 'Zero-cap contract differs')
        old_density = None
    else:
        old_density = density(p['old_density'], policy['uniform_half_width'])
        if old_density == -math.inf:
            require(p['status'] == 'source_zero_reverse_flow' and not trials, 'Zero source flow consumed trials')
        elif found is None:
            require(p['status'] == 'cap_exhausted' and len(trials) == p['trial_cap'], 'Premature singleton exhaustion')
    if found is None:
        require(p['candidate'] is None and p['new_density'] is None and p['full_log_reverse_forward'] is None
            and row['status'] == 'proposal_self_loop' and row['accepted'] is False,
            'Null singleton supplied a candidate/acceptance')
        require(not any(k in row for k in ('bath', 'log_u', 'log_acceptance_ratio', 'complete_log_correction', 'proposed')),
                'Null singleton consumed physical acceptance')
        return None
    require(p['status'] == 'candidate' and row['status'] == 'completed'
        and p['candidate'] == found['proposed_pose'] == row['proposed']
        and p['new_density'] == found['density'], 'Singleton destination differs')
    require(old_density is not None and math.isfinite(old_density), 'Candidate from zero source density')
    correction = old_density-density(p['new_density'], policy['uniform_half_width'])
    require(math.isfinite(correction), 'Nonfinite singleton proposal correction is fatal')
    close(p['full_log_reverse_forward'], correction, 'Full proposal reverse/forward ratio differs')
    close(row['complete_log_correction'], correction, 'Proposal correction must enter once')
    value = bath_weight(row['bath'], config)+correction
    close(row['log_acceptance_ratio'], value, 'One physical MH sum differs')
    u = row['log_u']
    require(type(u) in (int, float) and math.isfinite(u) and u < 0
        and row['accepted'] == (u < min(0., value)), 'Singleton MH decision differs')
    return row['bath']


def validate_journal(rows, config, job, initial, fixed_anchor, terminal=None, prepared_start=None):
    """Stream all updates; return only fully replayed retained endpoints."""
    require(job['arm'] in ARMS, 'Unexpected observer arm')
    rows = iter(rows); first = next(rows, None)
    require(first is not None and first['kind'] == 'initial' and first['block'] == 0
        and first['job'] == job and first['selected'] == initial and first['conditional_target'] is True,
        'Initial singleton journal differs')
    # Exact contract is emitted by the example and is checked independently of
    # candidate observations; the two arms differ only in effective fused mass.
    validate_initial_contract(first, config, job)
    require(first['fixed_source'] == config['source_frame']
        and first['prepared_manifest'] == config['inherited_campaign']['prepared_manifest']
        and first['prepared_start'] == prepared_start, 'Initial source/preparation binding differs')
    require((prepared_start is not None) == (job['initialization'] == 'proposal_prepared'),
            'Initial preparation type differs')
    selected = copy.deepcopy(initial); points = [first]
    cpu = first['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu >= 0, 'Invalid initial CPU')
    counts = dict(local_attempted=0, local_accepted=0, singleton_attempted=0,
                  singleton_accepted=0, singleton_self_loop=0)
    raw = retained = 0; context = config['contexts'][job['context_index']]
    members = [context['root'], context['child']]
    total = config['allocation']['warmup_blocks']+config['allocation']['production_blocks']
    for block in range(1, total+1):
        for attempt in range(5):
            row = next(rows, None); kind = 'local' if attempt < 4 else 'two_neighbor_singleton'
            require(row is not None and row['kind'] == kind and row['block'] == block
                and not any(k in row for k in ('fatal_error', 'proposal_failure', 'bath_failure')),
                'Missing, reordered or failed elementary update')
            require(type(row['accepted']) is bool and math.isfinite(row['sampler_cpu_seconds'])
                and row['sampler_cpu_seconds'] >= cpu, 'Invalid acceptance/CPU record')
            cpu = row['sampler_cpu_seconds']; gate = None
            if kind == 'local':
                slot = attempt % 2
                require(row['attempt'] == attempt and row['member'] == members[slot]
                    and row['old'] == selected[slot], 'Local state replay differs')
                require(row['status'] in ('completed', 'hard_rejected'), 'Unknown local disposition')
                counts['local_attempted'] += 1; counts['local_accepted'] += row['accepted']
                if row['status'] == 'hard_rejected':
                    require(not row['accepted'] and not any(k in row for k in ('bath', 'log_u', 'log_acceptance_ratio')),
                            'Hard-rejected local consumed physical acceptance')
                else:
                    gate = row['bath']; value = bath_weight(gate, config)
                    close(row['log_acceptance_ratio'], value, 'Local bath correction differs')
                    u = row['log_u']
                    require(type(u) in (int, float) and math.isfinite(u) and u < 0
                        and row['accepted'] == (u < min(0., value)), 'Local MH decision differs')
            else:
                slot = (block-1) % 2
                gate = validate_singleton(row, config, job, selected, fixed_anchor)
                counts['singleton_attempted'] += 1; counts['singleton_accepted'] += row['accepted']
                counts['singleton_self_loop'] += row['status'] == 'proposal_self_loop'
            if row['accepted']: selected[slot] = copy.deepcopy(row['proposed'])
            if gate is not None:
                raw += gate['raw_points']; retained += gate['retained_points']
            require(row['retained'] == selected, 'Accepted/rejected residence changed wrong member')
        row = next(rows, None)
        require(row is not None and row['kind'] == 'retained_block' and row['block'] == block
            and row['production'] is (block > config['allocation']['warmup_blocks'])
            and row['selected'] == selected and row['counts'] == counts
            and row['raw'] == raw and row['retained'] == retained, 'Retained block/counter closure differs')
        require(math.isfinite(row['sampler_cpu_seconds']) and row['sampler_cpu_seconds'] >= cpu, 'Nonmonotone retained CPU')
        cpu = row['sampler_cpu_seconds']; points.append(row)
    require(next(rows, None) is None, 'Unexpected journal tail')
    if terminal is not None:
        require(terminal['complete'] is True and terminal['conditional_target'] is True
            and terminal['job'] == job and terminal['blocks'] == total and terminal['counts'] == counts
            and terminal['raw'] == raw and terminal['retained'] == retained
            and math.isfinite(terminal['cpu_seconds']) and terminal['cpu_seconds'] >= cpu,
            'Terminal state/counter closure differs')
    return points


def validate_initial_contract(first, config, job):
    context = config['contexts'][job['context_index']]
    expected = dict(schema='evolving-dimer-two-neighbor-singleton-v1',
        policy=config['singleton_policy'], effective_oligomer=effective_oligomer(config, job['arm']),
        member_labels=[context['root'], context['child']], anchor_label=context['anchor'],
        member_slot_schedule='(block-1)%2', neighbors='[other_mobile,fixed_anchor]',
        uniform_frame='fixed_anchor_body', catalogue_rebuild='every elementary attempt',
        proposal_rng_role='singleton_two_neighbor/proposal', bath_rng_role='singleton_two_neighbor/bath',
        accept_rng_role='singleton_two_neighbor/accept',
        local_rng_roles='unchanged shared local/{attempt}/{proposal,bath,accept}',
        physical_decisions_per_candidate=1, guidance_cloud_used=False)
    require(first['singleton_contract'] == expected, 'Initial singleton proposal contract differs')
    require('cloud' not in first and 'guidance_contract' not in first, 'Singleton unexpectedly used a guidance cloud')


def comparison_summaries(chains):
    result = external.comparison_summaries(chains)
    for pair in result['descriptive_paired_comparisons']:
        left, right = pair['left'], pair['right']
        pair['control_global_rng_roles_paired'] = (left['initialization'] == right['initialization']
            and left['arm'] in ARMS and right['arm'] in ARMS)
    result['warning'] += ' Only the two singleton arms share singleton proposal/bath/accept role names; cached local/m4 collective streams are not globally paired.'
    return result


def checked(record):
    require(set(record) == {'path', 'sha256'}, 'Invalid BoundFile')
    path = Path(record['path']).resolve()
    require(sha(path) == record['sha256'], 'Changed bound file '+str(path))
    return path


def bind_complete_inputs(base, config):
    """Metadata-only admission; complete all-new inventory before any geometry."""
    validate_inventory(config)
    inputs = {}; items = []
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Changed observer input '+str(path))
        require(str(path) not in inputs or inputs[str(path)] == digest, 'Input changed during binding')
        inputs[str(path)] = digest; return path
    for name in ('config.json', 'protocol.json', 'run-binding.json', 'analysis-plan.json'):
        bind(base/name)
    status = read(bind(base/'dispatch/status.json'))
    require(status['complete'] is True and status['passed'] is True and status['failure'] is None
        and status['failure_draining'] is False and status['active'] == [] and status['unstarted'] == [],
        'All 64 chains must complete before observation')
    dispatch = read(bind(base/'dispatch/plan.json', status['plan_sha256']))
    require(dispatch['jobs'] == config['jobs'] and dispatch['retries'] is False
        and dispatch['replacements'] is False, 'Changed dispatch inventory')
    completed = {c['job']['id']: c for c in status['completed']}
    require(len(completed) == len(status['completed']) == 64
        and set(completed) == {j['id'] for j in config['jobs']}, 'Missing/repeated completed chain')
    output = Path(config['output']).resolve()
    require(output == base/'execution', 'Unexpected trajectory output')
    for job in config['jobs']:
        done = completed[job['id']]
        require(done['job'] == job and done['success'] is True and done['returncode'] == 0
            and done['error'] is None and done['child_drained'] is True, 'Failed or mismatched completed chain')
        directory = output/f"job-{job['id']:03}"
        terminal_path = bind(directory/'terminal.json', done['terminal_sha256']); terminal = read(terminal_path)
        require(terminal['complete'] is True and terminal['conditional_target'] is True and terminal['job'] == job
            and terminal['blocks'] == 4608 and terminal['config_sha256'] == inputs[str(base/'config.json')]
            and terminal['binding_sha256'] == inputs[str(base/'run-binding.json')]
            and terminal['counts']['local_attempted'] == 18432 and terminal['counts']['singleton_attempted'] == 4608
            and not (directory/'failure.json').exists(), 'Terminal scope/provenance differs')
        trajectory = checked(terminal['trajectory'])
        require(trajectory == directory/'trajectory.jsonl', 'Trajectory escaped chain directory')
        bind(trajectory, terminal['trajectory']['sha256'])
        items.append((job, terminal, trajectory))
    controls = config['control_analysis']; old = {}
    for name in ('summary', 'manifest', 'analysis'):
        p = checked(controls[name]); bind(p, controls[name]['sha256']); old[name] = read(p)
    summary, manifest, result = (old[k] for k in ('summary', 'manifest', 'analysis'))
    require(summary['complete'] is True and summary['passed'] is True and manifest['complete'] is True
        and result['complete'] is True and len(result['chains']) == 96
        and summary['analysis']['sha256'] == controls['analysis']['sha256']
        and summary['manifest_sha256'] == controls['manifest']['sha256'], 'Incomplete control receipt chain')
    selected = [c for c in result['chains'] if c['job']['arm'] in CONTROLS]
    expected = {(c, a, i, s) for c in range(4) for a in CONTROLS for i in STARTS for s in range(4)}
    require(len(selected) == 64 and {identity(c['job']) for c in selected} == expected,
            'Changed 64-chain control inventory')
    cached = []
    for chain in selected:
        ref = controls['observations'][str(chain['job']['id'])]; path = checked(ref)
        require(manifest['files'][path.name] == ref['sha256'], 'Control cache lacks completed-manifest binding')
        bind(path, ref['sha256']); cached.append((chain, path))
    return inputs, items, cached


def analyze(base, output):
    from prepare_two_neighbor_singleton_benchmark import verify
    base, output = Path(base).resolve(), Path(output).resolve()
    require(not output.exists(), 'Fresh observer output required; no replay/retry')
    config = verify(base)
    require(read(base/'analysis-plan.json') == analysis_plan(), 'Changed frozen observer plan')
    protocol = read(base/'protocol.json')
    for name in SOURCE_FILES:
        require(sha(Path(__file__).parents[1]/name) == protocol['source_files'][name], 'Changed frozen observer source '+name)
    inputs, items, controls = bind_complete_inputs(base, config)
    def load(ref):
        p = checked(ref); inputs[str(p)] = ref['sha256']; return read(p)
    binding = read(base/'run-binding.json'); prepared = load(binding['prepared_manifest'])
    require(binding['config_sha256'] == inputs[str(base/'config.json')]
        and binding['protocol_sha256'] == inputs[str(base/'protocol.json')]
        and prepared['complete'] is True and prepared['passed'] is True, 'Prepared/run authority differs')
    shape, patches, frame = (load(config[k]) for k in ('shape', 'patch_map', 'source_frame'))
    source = frame['poses']
    require(len(source) == 264 and patches['shape_sha256'] == config['shape']['sha256']
        and len(patches['atom_patch_ids']) == len(shape['atoms'])
        and set(patches['atom_patch_ids']) == set(patches['patch_dictionary']), 'Shape/patch/source identity differs')
    starts = {(s['context_index'], s['stream']): s for s in prepared['alternative_starts']}
    require(len(starts) == len(prepared['alternative_starts']) == 16
        and set(starts) == {(c, s) for c in range(4) for s in range(4)}, 'Prepared-start inventory differs')
    prepared_poses = {key: load(value['record'])['selected'] for key, value in starts.items()}
    observer_config = dict(boundary=dict(kind='spherical', radius=config['physical']['wall_radius']),
        box_lengths=[2*config['physical']['wall_radius']]*3, depletant_radius=config['physical']['depletant_radius'])
    output.mkdir(); write(output/'input-binding.json', dict(input_sha256=inputs, plan=analysis_plan(),
        source_sha256={n: protocol['source_files'][n] for n in SOURCE_FILES}))
    started = time.process_time(); chains = []; endpoint_count = 0; warmup = 512
    for job, terminal, path in items:
        tick = time.process_time(); context = config['contexts'][job['context_index']]
        members = [context['root'], context['child']]; initial = [source[i] for i in members]
        saved_start = None
        if job['initialization'] == 'proposal_prepared':
            initial = prepared_poses[job['context_index'], job['stream']]
            saved_start = starts[job['context_index'], job['stream']]
        def rows(stream):
            for raw in stream:
                require(raw.endswith('\n'), 'Partial elementary journal line')
                yield json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))
        with path.open() as stream:
            points = validate_journal(rows(stream), config, job, initial, source[context['anchor']], terminal, saved_start)
        observer = previous.ConditionalObserver(shape, patches['atom_patch_ids'], observer_config, source, members)
        trace = []; prior = None
        with (output/f"job-{job['id']:03}-observations.jsonl").open('x') as handle:
            for point in points:
                observation = observer.classify(point['selected']); tokens = set(observation['patch_tokens'])
                old = set(prior['patch_tokens']) if prior else tokens; union = old|tokens
                observation.update(block=point['block'], production=point['block'] > warmup,
                    sampler_cpu_seconds=point['sampler_cpu_seconds'],
                    jaccard_from_previous=1-len(old&tokens)/len(union) if union else 0.)
                handle.write(canonical(observation)+'\n'); trace.append(observation); prior = observation
        endpoint_count += len(trace)
        metrics = previous.summarize_trace(trace, warmup, terminal['cpu_seconds'])
        metrics['production_cpu_seconds'] = terminal['cpu_seconds']-points[warmup]['sampler_cpu_seconds']
        metrics['external_only'] = external.external_metrics(trace, warmup, terminal['cpu_seconds'], members)
        chains.append(dict(job=job, metrics=metrics, reused_control=False,
            observer_cpu_seconds=time.process_time()-tick, pair_classifications=observer.calls,
            exact_pair_cache_hits=observer.hits, trajectory=terminal['trajectory'], counts=terminal['counts'],
            geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds')))
    for control, path in controls:
        reused = copy.deepcopy(control); job = reused['job']; context = config['contexts'][job['context_index']]
        members = [context['root'], context['child']]
        trace = external.read_cached_trace(path, inputs[str(path)], warmup)
        reused['metrics']['external_only'] = external.external_metrics(trace, warmup,
            reused['metrics']['full_sampler_cpu_seconds'], members)
        reused.update(reused_control=True, new_geometry_queries=0); chains.append(reused)
    require(endpoint_count == 294976 and len(chains) == 128, 'Incomplete new/control endpoint inventory')
    require(sum(c['pair_classifications'] for c in chains if not c['reused_control']) <= 154862400,
            'Observer pair allocation exceeded')
    for path, digest in inputs.items(): require(sha(path) == digest, 'Input changed during observation '+path)
    new = [c for c in chains if not c['reused_control']]
    costs = dict(new_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in new),
        reused_control_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if c['reused_control']),
        new_geometry_load_cpu_seconds=math.fsum(c['geometry_load_cpu_seconds'] for c in new)
            if all(c['geometry_load_cpu_seconds'] is not None for c in new) else None,
        new_analysis_cpu_seconds=time.process_time()-started,
        note='Every rate uses its actual full sampler CPU including warmup/rejections. Shared inherited preparation was not repeated; observer/setup costs stay separate.')
    result = dict(schema='two-neighbor-singleton-analysis-v1', complete=True,
        new_chains=64, reused_control_chains=64, chains=chains, comparisons=comparison_summaries(chains),
        costs=costs, input_sha256=inputs, new_geometry_endpoints=endpoint_count,
        new_physical_draws=0, old_geometry_queries=0, native_observer=False,
        analysis_plan=analysis_plan())
    write(output/'analysis.json', result)
    write(output/'manifest.json', dict(complete=True, files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    result = analyze(args.base, args.output)
    print(json.dumps(dict(complete=True, new_chains=result['new_chains'], reused_control_chains=result['reused_control_chains'])))
