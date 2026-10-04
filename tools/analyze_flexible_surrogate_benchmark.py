#!/usr/bin/env python3
"""Completed flexible-chain audit and cached-contact comparison.

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
import plot_rigid_surrogate_benchmark as authority

previous, external, scalar, shared = rigid.previous, rigid.external, rigid.scalar, rigid.shared
require, read, sha, canonical, write = rigid.require, rigid.read, rigid.sha, rigid.canonical, rigid.write
finite, close, identity, cloud_weight, validate_score, decision = (rigid.finite, rigid.close,
    rigid.identity, rigid.cloud_weight, rigid.validate_score, rigid.decision)
ARMS = ('flexible_m1', 'flexible_m8', 'flexible_flat8')
STARTS = rigid.STARTS
CONTROLS = rigid.ARMS + ('local', 'm4')
SOURCE_FILES = list(dict.fromkeys(['tools/analyze_flexible_surrogate_benchmark.py',
    'tests/test_analyze_flexible_surrogate_benchmark.py',
    'tools/plot_rigid_surrogate_benchmark.py'] + rigid.SOURCE_FILES))


def analysis_plan():
    plan = copy.deepcopy(rigid.analysis_plan())
    plan.update(schema='flexible-surrogate-analysis-plan-v1', source_files=SOURCE_FILES,
        reused_control_chains=40, reused_control_arms=list(CONTROLS),
        target='Original context0 two-mobile conditional target; four unchanged locals then one fixed-length flexible random-scan chain.',
        replay='Every begun/outcome and inner residence; nonselected pose unchanged, full score arithmetic, fair-order two-leg bath and one endpoint correction. No rigid reconstruction.',
        internal_metrics='Internal edge occupancy, attachments/detachments; internal patch-set occupancy, direct nonempty changes and completed nonempty returns. Block512 is the transition baseline.',
        internal_interpretation='Coarse contact-patch reorganization; threshold flicker can change a set. Not native registry, identified basins, or a complete relative-pose descriptor.',
        cached_control_identity='One completed rigid-analysis umbrella; each of40 full(context,arm,initialization,stream) identities once, with original observation/manifest bindings.',
        relative_pose_metrics=False, native_observer=False,
        comparisons='Four streams per arm/start, no pooling starts or concatenation. Full sampler CPU includes warmup and rejected work. Shared RNG roles do not make arms independent replicates.')
    return plan


def effective_config(config, arm):
    require(arm in ARMS, 'Unknown flexible arm')
    policy = config['flexible_surrogate_policy']
    require(set(policy) == {'schema', 'proposal_scales'} and policy['schema'] == 'flexible-surrogate-policy-v1',
            'Changed flexible policy')
    proxy = dict(config, surrogate_policy=dict(policy, schema='rigid-surrogate-policy-v1'))
    return rigid.effective_config(proxy, rigid.ARMS[ARMS.index(arm)])


def validate_inventory(config):
    jobs = config['jobs']
    require(len(jobs) == 24 and len({j['id'] for j in jobs}) == 24 and {identity(j) for j in jobs}
            == {(0, arm, start, stream) for arm in ARMS for start in STARTS for stream in range(4)},
            'Changed complete24 flexible inventory')
    require(config['physical'] == scalar.PHYSICAL and config['allocation']['warmup_blocks'] == 512
            and config['allocation']['production_blocks'] == 4096, 'Changed target/schedule')
    require(not any(k in config for k in ('surrogate_policy', 'singleton_policy', 'two_root_policy')),
            'Another proposal policy present')
    require(config['flexible_surrogate_policy']['proposal_scales'] == {'source': 'local'}
            and config['local'] == dict(translation_std_A=.2, rotation_std_degrees=1., member_order=[0, 1, 0, 1],
                                        pair_contact_required=False),
            'Changed matched local scales/schedule')
    for arm in ARMS: effective_config(config, arm)


def surrogate_contract(config, job, cloud, meta):
    settings = effective_config(config, job['arm'])
    proxy = dict(config, surrogate_policy=dict(config['flexible_surrogate_policy'], schema='rigid-surrogate-policy-v1'))
    old_job = dict(job, arm=rigid.ARMS[ARMS.index(job['arm'])])
    contract = rigid.surrogate_contract(proxy, old_job, cloud, meta)
    contract.pop('handle')
    contract.update(schema='evolving-dimer-flexible-surrogate-v1', policy=config['flexible_surrogate_policy'],
        effective_config=settings, selection_probabilities=[0.5, 0.5],
        inner_selection='independent fair random scan each step',
        physical_path='fair-order two-singleton path; intermediate is not hard-filtered')
    for key in ('proposal_rng_role', 'inner_accept_rng_role', 'bath_rng_role', 'accept_rng_role'):
        contract[key] = contract[key].replace('rigid_surrogate/', 'flexible_surrogate/')
    return contract


def validate_path(bath, old, proposed, members, config):
    require(bath['order'] in ('first_then_second', 'second_then_first'), 'Unknown bath order')
    slots = [0, 1] if bath['order'] == 'first_then_second' else [1, 0]
    require(bath['ordered_members'] == [members[i] for i in slots] and len(bath['legs']) == 2,
            'Physical leg order differs')
    middle = copy.deepcopy(old); middle[slots[0]] = proposed[slots[0]]
    require(bath['intermediate_selected'] == middle, 'Copied auxiliary midpoint differs')
    # No hard filter is applied to this algebraic midpoint.
    keys = ('gained', 'lost', 'raw_points', 'retained_points', 'created_cells', 'retained_cells')
    for slot, leg in zip(slots, bath['legs']):
        scalar.bath_weight(leg, config)
        require(leg['raw_points'] <= config['limits']['raw_per_leg']
                and leg['retained_points'] <= config['limits']['retained_per_leg'], 'Completed bath leg exceeds cap')
        if old[slot] == proposed[slot]:
            require(all(leg[k] == 0 for k in keys + ('envelope_volume', 'log_weight')),
                    'Identity singleton leg spent bath')
        if leg['envelope_volume'] == 0: require(leg['raw_points'] == 0, 'Cloud in empty envelope')
    total = bath['aggregate']; scalar.bath_weight(total, config)
    require(total['raw_points'] <= config['limits']['raw_per_outer']
            and total['retained_points'] <= config['limits']['retained_per_outer'], 'Completed path exceeds cap')
    for key in keys:
        require(total[key] == sum(leg[key] for leg in bath['legs']), 'Two-leg count aggregation differs')
    for key in ('envelope_volume', 'log_weight'):
        close(total[key], sum(leg[key] for leg in bath['legs']), 'Two-leg '+key+' aggregation differs')
    # The saved summed log weight is the actual MH factor; do not replace it
    # with a differently rounded coefficient times the aggregate counts.
    return total['log_weight']


def validate_surrogate(row, begun, config, job, contract):
    members, settings = contract['members'], contract['effective_config']
    require(row['kind'] == 'flexible_surrogate_chain' and begun['kind'] == 'flexible_surrogate_attempt_begun'
        and begun['status'] == 'begun' and row['block'] == begun['block']
        and row['members'] == begun['members'] == members and row['config'] == begun['config'] == settings
        and row['old'] == begun['old'] and row['selection_probabilities'] == [0.5, 0.5]
        and 'handle' not in row and 'handle' not in begun, 'Flexible begun/outer contract differs')
    old = shared.poses(row['old']); current = copy.deepcopy(old)
    current_score = row['old_score']; old_s = validate_score(current_score, contract, config)
    steps = row['steps']; require(type(steps) is list and len(steps) == settings['inner_steps'], 'Fixed horizon differs')
    counts = dict(attempted=0, accepted=0, hard_rejected=0, mh_rejected=0)
    for index, step in enumerate(steps):
        require(scalar.nat(step['index'], 'step index') == index and step['old'] == current
                and type(step['accepted']) is bool, 'Inner residence/order differs')
        close(step['old_score'], current_score['log_surrogate'], 'Inner source score differs')
        slot = scalar.nat(step['selected_slot'], 'selected slot')
        require(slot in (0, 1) and step['selected_label'] == members[slot], 'Selected slot/label differs')
        target = shared.poses(step['proposed'])
        require(target[1-slot] == current[1-slot], 'Unselected member moved')
        counts['attempted'] += 1
        if step['status'] == 'hard_rejected':
            require(step['accepted'] is False and not any(k in step for k in
                ('proposed_score', 'log_u', 'log_acceptance_ratio')), 'Hard rejection has score/decision')
            counts['hard_rejected'] += 1
        else:
            require(step['status'] == 'completed', 'Incomplete/fatal inner trace')
            new_s = validate_score(step['proposed_score'], contract, config)
            decision(step, new_s-current_score['log_surrogate'])
            counts['accepted' if step['accepted'] else 'mh_rejected'] += 1
            if step['accepted']: current, current_score = copy.deepcopy(target), step['proposed_score']
        require(step['retained'] == current, 'Inner retained state differs')
        close(step['retained_score'], current_score['log_surrogate'], 'Retained score differs')
    for key in counts: scalar.nat(row['inner_counts'][key], key)
    require(row['inner_counts'] == counts and row['proposed'] == current and row['proposed_score'] == current_score,
            'Inner counters/final endpoint differ')
    correction = old_s-current_score['log_surrogate']
    close(row['complete_log_correction'], correction, 'Outer correction differs')
    before, after = row['budget_before'], row['budget_after']
    for budget in (before, after):
        require(set(budget) == {'raw', 'retained'}, 'Malformed budget')
        for key in budget: scalar.nat(budget[key], key)
        require(budget['raw'] <= config['limits']['raw_campaign']
                and budget['retained'] <= config['limits']['retained_campaign'], 'Completed campaign budget exceeds cap')
    require(before == dict(raw=begun['raw'], retained=begun['bath_retained']), 'Begun/kernel budget differs')
    require(type(row['accepted']) is bool, 'Invalid outer acceptance')
    scalar.nat(row['physical_decisions'], 'physical decisions')
    if current == old:
        require(row['status'] == 'identity_self_loop' and row['accepted'] is False and row['physical_decisions'] == 0
                and before == after and not any(k in row for k in ('bath', 'log_u', 'log_acceptance_ratio')),
                'Identity endpoint spent physical work')
    else:
        require(row['status'] == 'completed' and row['physical_decisions'] == 1, 'Incomplete/repeated physical decision')
        weight = validate_path(row['bath'], old, current, members, config)
        decision(row, weight+correction)
        aggregate = row['bath']['aggregate']
        require(after == dict(raw=before['raw']+aggregate['raw_points'],
                              retained=before['retained']+aggregate['retained_points']), 'Completed path budget differs')
    require(row['retained'] == (current if row['accepted'] else old), 'Physical retained pose differs')


def accumulate(diagnostics, row):
    projected = dict(row)
    if 'bath' in row: projected['bath'] = row['bath']['aggregate']
    rigid.accumulate(diagnostics, projected)
    phase = 'warmup' if row['block'] <= diagnostics['warmup_blocks'] else 'production'
    counts = diagnostics[phase].setdefault('selected_slots', [0, 0])
    for step in row['steps']: counts[step['selected_slot']] += 1
    orders = diagnostics[phase].setdefault('physical_path_orders', [0, 0])
    if 'bath' in row: orders[int(row['bath']['order'] == 'second_then_first')] += 1


finish_diagnostics = rigid.finish_diagnostics


def validate_checkpoint(checkpoint, terminal, last, trajectory, job):
    require(checkpoint['schema'] == 'evolving-dimer-checkpoint-v1' and checkpoint['job'] == job
        and checkpoint['block'] == terminal['blocks'] == last['block']
        and checkpoint['selected'] == last['selected'] and checkpoint['counts'] == terminal['counts'] == last['counts']
        and checkpoint['raw'] == terminal['raw'] == last['raw']
        and checkpoint['retained'] == terminal['retained'] == last['retained']
        and checkpoint['config_sha256'] == terminal['config_sha256']
        and checkpoint['binding_sha256'] == terminal['binding_sha256']
        and checkpoint['journal_rows'] == 1+7*terminal['blocks']
        and checkpoint['journal_bytes'] == Path(trajectory).stat().st_size
        and checkpoint['journal_sha256'] == terminal['trajectory']['sha256'], 'Final checkpoint closure differs')
    require(last['sampler_cpu_seconds'] <= finite(checkpoint['cpu_seconds'], 'checkpoint CPU', True)
            <= terminal['cpu_seconds'], 'Checkpoint CPU closure differs')


def internal_metrics(trace, warmup, cpu, members):
    """Reduce cached token sets only; include the last warmup transition baseline."""
    require(math.isfinite(cpu) and cpu > 0 and len(set(members)) == 2, 'Invalid internal metric allocation')
    pair = tuple(sorted(members)); projected = []; values = []
    for index, row in enumerate(trace):
        require(row['block'] == index, 'Internal trace block sequence differs')
        tokens = [tuple(t) for t in row['patch_tokens']]
        require(len(tokens) == len(set(tokens)) and all(len(t) == 4 and type(t[0]) is int
            and type(t[1]) is int and t[0] < t[1] and type(t[2]) is str and type(t[3]) is str
            and bool(set(t[:2]) & set(members)) for t in tokens),
            'Malformed cached contact token')
        edges = sorted(set(t[:2] for t in tokens))
        require(edges == sorted(tuple(e) for e in row['partner_edges']), 'Token/edge partition differs')
        internal = sorted(t for t in tokens if t[:2] == pair)
        require(type(row['internal_contact']) is bool and row['internal_contact'] == bool(internal),
                'Internal edge/token occupancy differs')
        require(sorted(tuple(e) for e in row['external_edges']) == [e for e in edges if e != pair],
                'External/internal edge partition differs')
        projected.append(dict(block=index, external_edges=internal))
        if index > warmup: values.append(tuple(internal))
    require(values and len(trace) > warmup, 'No internal production observations')
    counts = Counter(values); activity = shared.external_activity(projected, warmup, cpu)
    activity['scope'] = analysis_plan()['internal_interpretation']
    for event in activity['completed_nonempty_returns']:
        event['patch_tokens'] = event.pop('environment')
    labels = [previous.key(v) for v in values]
    matrix, _ = previous.presence_matrix([[(label,)] for label in labels])
    return dict(production_samples=len(values), full_sampler_cpu_seconds=cpu,
        internal_contact_fraction=sum(bool(v) for v in values)/len(values),
        attachments=activity['entering_nonempty'], detachments=activity['leaving_nonempty'],
        patch_set_occupancy=[dict(fingerprint=previous.key(v), patch_tokens=list(v), fraction=n/len(values))
            for v, n in sorted(counts.items())],
        patch_set_ess=previous.apparent_effective_count(matrix, cpu), patch_set_activity=activity,
        transition_baseline_block=warmup, scope=analysis_plan()['internal_interpretation'])


def comparison_summaries(chains):
    result = external.comparison_summaries(chains)
    indexed = {identity(c['job']): c['metrics']['internal_organization'] for c in chains}
    def compact(value):
        activity = value['patch_set_activity']; ess = value['patch_set_ess']
        return dict(internal_contact_fraction=value['internal_contact_fraction'],
            unique_patch_sets=len(value['patch_set_occupancy']), attachments=value['attachments'], detachments=value['detachments'],
            apparent_ess=ess['apparent_ess'], apparent_ess_per_sampling_CPU_second=ess['apparent_ess_per_sampling_CPU_second'],
            full_sampler_cpu_seconds=value['full_sampler_cpu_seconds'],
            direct_nonempty_changes=activity['direct_nonempty_changes'],
            completed_nonempty_returns=len(activity['completed_nonempty_returns']),
            direct_nonempty_changes_per_full_CPU_second=activity['direct_nonempty_changes_per_full_CPU_second'],
            completed_nonempty_returns_per_full_CPU_second=activity['completed_nonempty_returns_per_full_CPU_second'])
    for group in result['groups']:
        for row in group['independent_stream_metrics']:
            row['internal_organization'] = compact(indexed[identity(dict(group, stream=row['stream']))])
    for pair in result['descriptive_paired_comparisons']:
        a, b = (indexed[identity(pair[side])] for side in ('left', 'right'))
        occupancy = lambda value: {row['fingerprint']: row['fraction'] for row in value['patch_set_occupancy']}
        pair['internal_organization'] = dict(left=compact(a), right=compact(b),
            patch_set_occupancy_total_variation=previous.total_variation(occupancy(a), occupancy(b)),
            contact_fraction_difference=b['internal_contact_fraction']-a['internal_contact_fraction'],
            direct_nonempty_changes_per_CPU_difference=b['patch_set_activity']['direct_nonempty_changes_per_full_CPU_second']
                -a['patch_set_activity']['direct_nonempty_changes_per_full_CPU_second'],
            nonempty_returns_per_CPU_difference=b['patch_set_activity']['completed_nonempty_returns_per_full_CPU_second']
                -a['patch_set_activity']['completed_nonempty_returns_per_full_CPU_second'], difference_direction='right minus left')
    return result


def bind_cached_controls(config, bind):
    controls = config['control_analysis']
    analysis_path = bind(controls['analysis'])
    root = analysis_path.parent.parent
    umbrella, authenticated = authority.authenticate(root)
    for path, digest in authenticated.items(): bind(dict(path=path, sha256=digest))
    require(analysis_path == root/'analysis/analysis.json'
            and bind(controls['summary']) == root/'execution/summary.json'
            and bind(controls['manifest']) == root/'analysis/manifest.json', 'Wrong control umbrella authority')
    input_path = bind(controls['input_binding']); input_binding = read(input_path)
    manifest = read(Path(controls['manifest']['path']))
    require(manifest['files'][input_path.name] == controls['input_binding']['sha256']
            and input_binding['input_sha256'] == umbrella['input_sha256'], 'Umbrella input closure differs')
    for ref in controls['authority'].values(): bind(ref)
    indexed = {identity(chain['job']): chain for chain in umbrella['chains']}
    expected = {(0, arm, start, stream) for arm in CONTROLS for start in STARTS for stream in range(4)}
    require(len(indexed) == len(umbrella['chains']) == 40 and set(indexed) == expected,
            'Incomplete/duplicate umbrella control inventory')
    entries = controls['observations']
    require(len(entries) == 40 and len({identity(entry['job']) for entry in entries}) == 40
            and {identity(entry['job']) for entry in entries} == expected, 'Control observation inventory differs')
    cached = []; origins = {}
    for entry in entries:
        chain = indexed[identity(entry['job'])]
        require(chain['job'] == entry['job'], 'Control full job identity differs')
        path = bind(entry['observation']); origin_path = bind(entry['analysis']); manifest_path = bind(entry['manifest'])
        require(path == origin_path.parent/f"job-{entry['job']['id']:03}-observations.jsonl",
                'Observation filename does not match originating job identity')
        origin_key = (str(origin_path), str(manifest_path))
        if origin_key not in origins:
            origin, original_manifest = read(origin_path), read(manifest_path)
            require(origin['complete'] is True and original_manifest['complete'] is True
                    and original_manifest['files'][origin_path.name] == entry['analysis']['sha256'],
                    'Originating cached analysis not complete/authenticated')
            origins[origin_key] = origin, original_manifest
        origin, original_manifest = origins[origin_key]
        require(original_manifest['files'][path.name] == entry['observation']['sha256'],
                'Cached observations not bound by originating manifest')
        matches = [c for c in origin['chains'] if identity(c['job']) == identity(entry['job'])]
        require(len(matches) == 1 and matches[0]['job'] == entry['job'], 'Ambiguous originating control identity')
        if chain['job']['arm'] in ('local', 'm4'):
            for ref in (entry['observation'], entry['analysis'], entry['manifest']):
                require(umbrella['input_sha256'].get(str(Path(ref['path']).resolve())) == ref['sha256'],
                        'Inherited control absent from umbrella input closure')
        else:
            require(origin_path == analysis_path and manifest_path == Path(controls['manifest']['path']).resolve(),
                    'Rigid control has wrong originating analysis')
        terminal_path = bind(entry['terminal']); terminal = read(terminal_path)
        origin_inputs = origin['input_files'] if origin.get('schema') == 'conditional-dimer-analysis-v1' else origin['input_sha256']
        require(origin_inputs.get(str(terminal_path)) == entry['terminal']['sha256'],
                'Control terminal absent from originating analysis closure')
        require(terminal['complete'] is True and terminal['conditional_target'] is True
                and terminal['job'] == chain['job'] and terminal['counts'] == chain['counts']
                and terminal['trajectory'] == chain['trajectory'] and terminal['blocks'] == 4608
                and terminal['cpu_seconds'] == chain['metrics']['full_sampler_cpu_seconds'],
                'Control terminal/cost provenance differs')
        cached.append((chain, path))
    return cached


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
            and status['active'] is None and status['unstarted'] == [] and len(status['completed']) == len(plan['jobs']) == 24
            and not (base/'execution/failure.json').exists(), 'All24 flexible jobs must complete cleanly')
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
    require(first['flexible_surrogate_contract'] == contract and first['cloud'] == cloud
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
        if row['kind'] == 'flexible_surrogate_attempt_begun':
            require(row['raw'] == raw and row['bath_retained'] == retained, 'Begun bath counters differ')
            begun = row; row = next(rows, None)
            require(row is not None and row['kind'] == 'flexible_surrogate_chain'
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
    from prepare_flexible_surrogate_benchmark import verify
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
        require(endpoints == analysis_plan()['new_retained_initial_observations'] and len(chains) == 64, 'Incomplete analysis inventory')
        require(sum(c['pair_classifications'] for c in chains if not c['reused_control']) <= analysis_plan()['maximum_new_pair_classifications'],
                'Observer query allocation exceeded')
        for path, digest in inputs.items(): require(sha(path) == digest, 'Input changed during observation '+path)
        for name in SOURCE_FILES: require(sha(repo/name) == protocol['source_files'][name], 'Observer source changed '+name)
        comparisons = comparison_summaries(chains)
        for pair in comparisons['descriptive_paired_comparisons']:
            same_start = pair['left']['initialization'] == pair['right']['initialization']
            pair['local_rng_roles_paired'] = same_start
            pair['flexible_surrogate_rng_roles_paired'] = same_start and {pair['left']['arm'], pair['right']['arm']} <= set(ARMS)
            pair['control_global_rng_roles_paired'] = False
        result = dict(schema='flexible-surrogate-analysis-v1', complete=True, new_chains=24, reused_control_chains=40,
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
