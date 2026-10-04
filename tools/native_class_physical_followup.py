#!/usr/bin/env python3
"""Three fresh sensitivity blocks, with the completed primary kept external.

The regional producer, audit, label, selection, geometry, statistics and
fail/drain controller are reused unchanged. Their protocol wire schema remains
v1; this module owns the explicit ``sensitivity_followup_only`` scope. No
completed primary job is manufactured, copied into the new lifecycle or rerun.
The final cross-stage comparison consumes admitted JSON summaries only.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sys

import prepare_native_class_physical_campaign as base
import native_class_physical_stage as stage
import admit_native_class_physical_campaign as admission
import analyze_native_class_physical_populations as statistics
from analyze_mobile_native_pocket import local_sources

require, read, sha, write = base.require, base.read, base.sha, base.write
SCOPE = 'sensitivity_followup_only'
INPUT_SCHEMA = 'native-class-physical-followup-inputs-v1'
REVIEW_SCHEMA = 'native-class-physical-followup-review-v1'
ADMISSION_SCHEMA = 'native-class-physical-followup-admission-v1'
COMPARISON_SCHEMA = 'native-class-physical-followup-comparison-v1'
ARMS = (
    dict(id='class_large', samples=65536, alpha=.5, lambda_ratio=128.),
    dict(id='class_defensive', samples=16384, alpha=.2, lambda_ratio=128.),
    dict(id='class_intensity', samples=16384, alpha=.5, lambda_ratio=64.),
)
TOTAL_ATTEMPTS = 786432
PRIMARY_PINS = dict(
    protocol='780dad504535f67184939367d35769cda5a0feeab47cce989949a0138339de92',
    execution_plan='3d7f6e61f6b77d4b08b73659fca5b3ef9f07cd721361edec3c152cd68985c94a',
    execution_summary='1c4e6d19361a5cc292588089eba284a868a14fd96bf3bfca8d2ba896a7b6df40',
    admission='7a15e180263b8bf5d19b5f9056a74a149b16e128d37e3ab7bfafe6c64ec70613',
    statistics='942b4f7fbe8bc8e163eb8c88a342c2e98432edfcece282a8e63374c87a4fb72c',
)


def arm_design():
    return copy.deepcopy(list(ARMS))


def primary_evidence(refs, bindings):
    """Reuse a pinned completed admission, without decoding any old row stream."""
    require(type(refs) is dict and set(refs) == set(PRIMARY_PINS), 'Exact external primary references required')
    values = {}
    for name, digest in PRIMARY_PINS.items():
        require(refs[name]['sha256'] == digest, 'Changed external primary '+name)
        values[name] = read(bindings.reference(refs[name]))
    protocol, execution, done, admitted, report = [values[k] for k in PRIMARY_PINS]
    require(protocol['schema'] == base.SCHEMA and protocol['study_scope'] == 'primary_only'
            and protocol['total_attempts'] == 262144 and protocol['cloud_replicates'] == 2,
            'Wrong completed primary allocation')
    original = admission.protocol_inventory(protocol)
    require(len(original) == 16 and protocol['class_guide_profile'] == base.REVISED_PROFILE,
            'Wrong completed primary proposal profile')
    root = Path(protocol['root']).resolve()
    for name, relative in [('protocol', 'protocol.json'), ('execution_plan', 'execution-plan.json'),
                           ('execution_summary', 'execution/summary.json'), ('statistics', 'analysis/statistics.json')]:
        require(Path(refs[name]['path']).resolve() == root/relative, 'External primary path differs: '+name)
    require(execution['root'] == str(root) and execution['files'].get(refs['protocol']['path']) == refs['protocol']['sha256']
            and done['complete'] is True and done['passed'] is True and done['failure'] is None
            and done['active'] is None and done['unstarted'] == []
            and done['plan_sha256'] == refs['execution_plan']['sha256']
            and len(done['completed']) == len(execution['jobs']) == 81,
            'External primary lifecycle is incomplete')
    require(admitted['schema'] == admission.SCHEMA and admitted['complete'] is True
            and admitted['implementation_admission_passed'] is True
            and admitted['execution_lifecycle_passed'] is True
            and admitted['selected_full_geometry_gate_satisfied'] is True
            and admitted['protocol'] == refs['protocol'] and admitted['statistics_result'] == refs['statistics']
            and admitted['total_unconditional_attempts'] == 262144,
            'Completed external primary admission required')
    require(report['schema'] == statistics.SCHEMA and report['complete'] is True
            and report['target_and_regions_sha256'] == protocol['target_and_regions_sha256']
            == admitted['target_and_regions_sha256'] and report['gates'] == statistics.GATES,
            'External primary statistical target differs')
    check_report_inventory(report, protocol['arms'], admitted['populations'])
    require(all(admitted[k] is False for k in ('physical_campaign_gate_open', 'full_vessel_gate_open', 'assembly_gate_open')),
            'External implementation admission cannot imply scientific convergence')
    return values


def check_report_inventory(report, arms, accepted=None):
    require(set(report['arms']) == {a['id'] for a in arms}, 'Report arm inventory differs')
    expected = {}
    seeds = []; roots = []
    for arm in arms:
        rows = report['arms'][arm['id']]['populations']
        require(len(rows) == 8 and [p['id'] for p in rows] == [p['id'] for p in arm['populations']],
                'Report population order differs')
        for row, pop in zip(rows, arm['populations']):
            require(row['seed'] == pop['seed'] and row['source_schema'] == arm['source_schema']
                    and row['draws'] == row['unconditional_denominator'] == arm['samples'],
                    'Report lost original attempted denominator or seed')
            expected[(arm['id'], pop['id'])] = (arm['samples'], arm['source_schema'])
            seeds.append(pop['seed']); roots.append(str(Path(pop['directory']).resolve()))
    require(len(set(seeds)) == len(seeds) and len(set(roots)) == len(roots), 'Repeated physical population')
    if accepted is not None:
        keys = [(p['arm'], p['id']) for p in accepted]
        require(len(keys) == len(set(keys)) and set(keys) == set(expected), 'Admitted population inventory differs')
        require(all((p['samples'], p['source_schema']) == expected[(p['arm'], p['id'])] for p in accepted),
                'Admitted population denominator differs')


def external_comparisons(history):
    return [dict(left='class', right=a['id'], historical_failed_strata=copy.deepcopy(history)) for a in ARMS]


def history_union(history, primary, history_ref, primary_ref):
    """Retain every old/current failed stratum, including now-small Q0/Qz bins."""
    require(history['complete'] is True and primary['complete'] is True
            and history['target_and_regions_sha256'] == primary['target_and_regions_sha256'],
            'Historical union target differs')
    keys = set()
    def add(row):
        key = row['family'], row['bin'], row['region']
        require(key[0] in statistics.BINS and type(key[1]) is int
                and 0 <= key[1] < statistics.BINS[key[0]] and key[2] in statistics.DECISIONS,
                'Invalid historical stratum key')
        keys.add(key)
    for row in history['entries']: add(row)
    original = set(keys); retained = []
    for row in primary['statistical_diagnostics']['unstable_stratum_comparisons']:
        require(row['kind'] in ('Qz','Q0'), 'Unknown primary stratum mass kind')
        add(row)
        retained.append({k: copy.deepcopy(row[k]) for k in (
            'left','right','kind','family','bin','region','material','historical_failure',
            'left_quality','right_quality','left_observed','right_observed')})
    return dict(schema='native-class-followup-failed-strata-union-v1', complete=True,
        target_and_regions_sha256=history['target_and_regions_sha256'],
        policy='union-of-original-history-and-all-primary-unstable-strata-v1',
        original_history=history_ref, completed_primary_admission=primary_ref,
        entries=[dict(family=f,bin=i,region=r) for f,i,r in sorted(keys)],
        added_keys=[dict(family=f,bin=i,region=r) for f,i,r in sorted(keys-original)],
        primary_unstable_provenance=retained,
        scope='Metadata-only prospective review policy. Both Qz and Q0 provenance retained; all union keys remain reportable regardless of later observed mass fraction. No region, target, label or proposal changes.')


def protocol_inventory(protocol):
    require(protocol['schema'] == base.SCHEMA and protocol['study_scope'] == SCOPE,
            'Wrong follow-up protocol scope')
    require(protocol['strata'] == statistics.STRATA and protocol['gates'] == statistics.GATES
            and protocol['cloud_replicates'] == 2 and protocol['total_attempts'] == TOTAL_ATTEMPTS
            and protocol['maximum_clouds'] == 2*TOTAL_ATTEMPTS,
            'Changed target diagnostics or fixed follow-up allocation')
    require(protocol['comparisons'] == [] and protocol['primary_is_external_control'] is True,
            'Primary controls must remain outside new row reduction and lifecycle')
    arms = protocol['arms']
    require(len(arms) == 3 and [a['id'] for a in arms] == [a['id'] for a in ARMS],
            'Missing, repeated or extra follow-up arm')
    inventory = {}; seeds = []; roots = []
    for arm, specification in zip(arms, ARMS):
        require(all(arm[k] == v for k, v in specification.items())
                and arm['producer'] == 'class' and arm['source_schema'] == admission.V7,
                'Follow-up proposal, schema or allocation differs')
        require([p['id'] for p in arm['populations']] == [f'r{i:02}' for i in range(8)],
                'Exactly eight ordered fresh populations required')
        for pop in arm['populations']:
            for name in ('seed', 'audit_seed'):
                require(type(pop[name]) is int and 0 <= pop[name] < 2**64, 'Invalid follow-up seed')
                seeds.append(pop[name])
            root = Path(pop['directory']); require(root.is_absolute(), 'Population root must be absolute')
            roots.append(str(root.resolve())); inventory[(arm['id'], pop['id'])] = (arm, pop)
    require(len(set(seeds)) == 48 and len(set(roots)) == 24, 'Repeated follow-up seed or population root')
    require(type(protocol['external_comparisons']) is list and len(protocol['external_comparisons']) == 3,
            'Exactly three external comparisons required')
    history = protocol['external_comparisons'][0]['historical_failed_strata']
    require(protocol['external_comparisons'] == external_comparisons(history), 'Changed external comparison inventory')
    base.validate_limits(protocol['phase_limits'], protocol['resources'])
    return inventory


def source_closure():
    sources = local_sources(__file__)
    for name in ('prepare_native_class_physical_followup.py', 'postrun_native_class_physical_followup.py'):
        path = Path(__file__).resolve().parent/name
        require(path.is_file(), 'Missing follow-up entry point '+name)
        sources[name] = path
    return sources


def validate(inputs_path, review_path):
    bindings = statistics.Bindings()
    inputs_path = bindings.bind(Path(inputs_path).resolve()); review_path = bindings.bind(Path(review_path).resolve())
    inputs, review = read(inputs_path), read(review_path)
    require(inputs['schema'] == INPUT_SCHEMA and review['schema'] == REVIEW_SCHEMA,
            'Wrong follow-up input/review schema')
    require(review['complete'] is True and review['passed'] is True
            and review['inputs'] == base.bound(inputs_path) and review['study_scope'] == SCOPE
            and review['physical_gates_remain_closed'] is True
            and review['historical_inventory_complete'] is True
            and review['stage_materialization_policy'] == base.POLICY
            and review['deterministic_materialization_authorized'] is True,
            'Matching follow-up freeze review required')
    base.validate_limits(inputs['phase_limits'], inputs['resources'])
    require(review['phase_limits'] == inputs['phase_limits'] and review['resources'] == inputs['resources'],
            'Unreviewed follow-up resource budgets')
    require(type(inputs['seed_namespace']) is str and inputs['seed_namespace'].strip(), 'Fresh seed namespace required')
    primary = primary_evidence(inputs['primary'], bindings); original = primary['protocol']
    region, descriptor, target_id = statistics.target_identity(original['target'], bindings)
    require(target_id == original['target_and_regions_sha256'] and descriptor == primary['statistics']['target_descriptor'],
            'Follow-up target differs from completed primary')
    require(original['class_guide_profile'] == base.REVISED_PROFILE, 'The frozen 116-component guide is required')
    for path, digest in original['files'].items(): bindings.bind(path, digest)
    producer = original['producers']['class']
    for key in ('executable', 'source_bundle'):
        require(producer[key]['sha256'] == base.PRODUCERS['class'][key], 'Changed validated class producer')
        bindings.reference(producer[key])
    guide_ref = next(a['guide'] for a in original['arms'] if a['id'] == 'class')
    guide = read(bindings.reference(guide_ref))
    require(guide['defensive_uniform_shell_probability'] == .5 and len(guide['gaussian_components']) == 116
            and guide['class_channels'] == base.CHANNELS, 'Changed inherited proposal law')
    old_history = read(bindings.reference(original['historical_failed_strata']))
    history = history_union(old_history, primary['admission'], original['historical_failed_strata'], inputs['primary']['admission'])
    sources = source_closure()
    for name, digest in original['source_sha256'].items():
        path = Path(__file__).resolve().parent/name
        require(path.is_file() and sha(path) == digest, 'Reused pipeline source changed: '+name)
        sources[name] = path
    for path in sources.values(): bindings.bind(path)
    runtime = base.streaming.runtime_identity()
    require(runtime == original['runtime'], 'Archived physical pipeline runtime changed')
    for path, digest in runtime['file_sha256'].items(): bindings.bind(path, digest)
    validation = read(bindings.reference(review['validation']))
    require(validation['complete'] is True and validation['passed'] is True
            and validation['returncode'] == 0, 'Passing follow-up validation receipt required')
    for name in ('native_class_physical_followup.py', 'prepare_native_class_physical_followup.py',
                 'postrun_native_class_physical_followup.py'):
        path = str(sources[name].resolve())
        require(validation['source_before'].get(path) == validation['source_after'].get(path) == sha(path),
                'Follow-up validation does not bind current source '+name)
    bindings.recheck()
    return dict(inputs=inputs, review=review, bindings=bindings, primary=primary, guide=guide,
                guide_ref=guide_ref, target_id=target_id, descriptor=descriptor, sources=sources,
                runtime=runtime, history=history, inputs_path=inputs_path, review_path=review_path)


def prepare(out, inputs_path, review_path):
    out = Path(out).resolve()
    require(not out.exists() and out.parent.is_dir(), 'Fresh follow-up root required')
    context = validate(inputs_path, review_path); inputs = context['inputs']
    require(shutil.disk_usage(out.parent).free >= inputs['resources']['minimum_free_bytes'], 'Storage reserve unavailable')
    seeds = [base.stream_seed(inputs['seed_namespace'], a['id'], f'r{i:02}', role)
             for a in ARMS for i in range(8) for role in ('physical', 'audit')]
    require(len(set(seeds)) == 48, 'Follow-up seed collision')
    repo = Path(__file__).resolve().parents[1]
    roots = list(dict.fromkeys([str(repo/'runs'), str(repo/'results'), *base.ROOTS]))
    require('/vast/xvg/tetramer-mc-runs' in roots, 'Global /vast seed inventory is required')
    inventory = base.seed_inventory(roots)
    require(not set(seeds).intersection(inventory['seeds']), 'Follow-up seed collides with historical declaration')
    context['bindings'].recheck(); out.mkdir()
    try:
        write(out/'preparation-claim.json', dict(schema=INPUT_SCHEMA, root=str(out),
            inputs=base.bound(context['inputs_path']), review=base.bound(context['review_path']),
            study_scope=SCOPE, launched=False, retries=0))
        return materialize(out, context, inventory)
    except BaseException as error:
        write(out/'preparation-failure.json', dict(schema=INPUT_SCHEMA, complete=False, launched=False,
            error_type=type(error).__name__, error=str(error), retries=0,
            scope='Preserved failed preparation; never run, resume or replace this directory.'))
        raise


def materialize(out, context, inventory):
    """Copy orchestration only; use authenticated immutable physical assets."""
    out = Path(out).resolve(); code = out/'code'; common = out/'common'
    code.mkdir(); common.mkdir(); (out/'analysis').mkdir(); (common/'preselections').mkdir()
    inputs, original = context['inputs'], context['primary']['protocol']
    files = dict(original['files'])
    for name, source in context['sources'].items():
        require(context['bindings'].files.get(str(source.resolve())) == sha(source), 'Unadmitted source copy')
        shutil.copy2(source, code/name); files[str(code/name)] = sha(code/name)
    for name, path in [('inputs.json', context['inputs_path']), ('review.json', context['review_path'])]:
        shutil.copy2(path, common/name); files[str(common/name)] = sha(common/name)
    write(common/'seed-inventory.json', inventory)
    files[str(common/'seed-inventory.json')] = sha(common/'seed-inventory.json')
    history_path = common/'historical-failed-strata-union.json'; write(history_path,context['history'])
    files[str(history_path)] = sha(history_path)
    arms = []
    for declaration in arm_design():
        guide = copy.deepcopy(context['guide']); guide['defensive_uniform_shell_probability'] = declaration['alpha']
        guide_path = common/(declaration['id']+'-guide.json'); write(guide_path, guide)
        arm = dict(declaration, source_schema=admission.V7, producer='class', guide=base.bound(guide_path), populations=[])
        files[str(guide_path)] = sha(guide_path)
        for i in range(8):
            ident = f'r{i:02}'; seed = base.stream_seed(inputs['seed_namespace'], arm['id'], ident, 'physical')
            audit_seed = base.stream_seed(inputs['seed_namespace'], arm['id'], ident, 'audit')
            pre = common/'preselections'/f'{arm["id"]}-{ident}.json'
            write(pre, base.selected.preselection(ident, arm['samples'], audit_seed)); files[str(pre)] = sha(pre)
            directory = out/'populations'/arm['id']/ident; directory.parent.mkdir(parents=True, exist_ok=True)
            (out/'analysis'/f'{arm["id"]}-{ident}').mkdir()
            arm['populations'].append(dict(id=ident, seed=seed, directory=str(directory), audit_seed=audit_seed,
                                          preselection=base.bound(pre)))
        arms.append(arm)
    files.update(context['runtime']['file_sha256'])
    for reference in inputs['primary'].values(): files[reference['path']] = reference['sha256']
    protocol = {key: copy.deepcopy(original[key]) for key in (
        'schema', 'stage_materialization_policy', 'runtime', 'target', 'target_and_regions_sha256',
        'strata', 'strata_file', 'gates', 'config', 'shape', 'native_identity', 'observer_setup',
        'historical_failed_strata', 'historical_failed_strata_policy', 'prerequisites', 'class_guide_profile')}
    protocol.update(root=str(out), code_directory=str(code), study_scope=SCOPE,
        materialization_review=base.bound(common/'review.json'), files=files,
        source_sha256={name: sha(code/name) for name in context['sources']},
        python=str(Path(sys.executable).absolute()), producers={'class': original['producers']['class']},
        arms=arms, comparisons=[], external_comparisons=external_comparisons(context['history']['entries']),
        external_primary=copy.deepcopy(inputs['primary']), primary_is_external_control=True,
        inherited_class_guide=context['guide_ref'], phase_limits=inputs['phase_limits'], resources=inputs['resources'],
        historical_failed_strata=base.bound(history_path),historical_failed_strata_policy=context['history']['policy'],
        cloud_replicates=2, total_attempts=TOTAL_ATTEMPTS, maximum_clouds=2*TOTAL_ATTEMPTS,
        selected_geometry_max_rows=480, selected_geometry_max_axis_queries=1440,
        prior_inputs=context['bindings'].files, launched=False,
        physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        scope='Three fixed fresh sensitivity blocks. Completed primary controls remain external; no refit, pooling, retries or assembly inference.')
    protocol_inventory(protocol); context['bindings'].recheck()
    write(out/'protocol.json', protocol)
    execution = base.execution_plan(protocol, base.bound(out/'protocol.json'))
    require(len(execution['jobs']) == 121, 'Wrong follow-up stage allocation')
    for path, digest in files.items(): require(sha(path) == digest, 'Frozen input changed before publication')
    write(out/'execution-plan.json', execution)
    result = dict(schema='native-class-physical-followup-preparation-v1', complete=True, root=str(out),
        protocol=base.bound(out/'protocol.json'), execution_plan=base.bound(out/'execution-plan.json'),
        study_scope=SCOPE, populations=24, total_attempts=TOTAL_ATTEMPTS, maximum_clouds=2*TOTAL_ATTEMPTS,
        external_primary=inputs['primary'], primary_jobs_reexecuted=0, launched=False)
    write(out/'preparation.json', result)
    return result


def comparison_pair(left, right, arms, historical):
    require(left in arms and right in arms and left != right, 'Invalid independent comparison')
    history = set()
    for item in historical:
        family, index, name = item['family'], item['bin'], item['region']
        require(family in statistics.BINS and type(index) is int and 0 <= index < statistics.BINS[family]
                and name in statistics.DECISIONS, 'Invalid historical stratum')
        history.add((family, index, name))
    result = dict(left=left, right=right, kinds={})
    for kind in ('Qz', 'Q0'):
        a, b = arms[left]['estimates'][kind], arms[right]['estimates'][kind]; strata = []
        for family, count in statistics.BINS.items():
            for index in range(count):
                x, y = a['strata'][family][index], b['strata'][family][index]
                compared = statistics.compare_blocks(x, y)
                for name in statistics.DECISIONS:
                    material = max(x['observed_parent_mass_fraction'][name] or 0.,
                                   y['observed_parent_mass_fraction'][name] or 0.) >= statistics.GATES['significant_stratum_mass_fraction']
                    strata.append(dict(family=family, bin=index, region=name, material=material,
                        historical_failure=(family, index, name) in history, comparison=compared[name]))
        result['kinds'][kind] = dict(regions=statistics.compare_blocks(a['primary'], b['primary']), strata=strata,
            free_energy_contrast=statistics.compare_free_energy(a['primary'], b['primary']))
    return result


def compare(protocol_ref, followup_admission_ref):
    """No old/new row replay, refitting, renormalization or population pooling."""
    bindings = statistics.Bindings(); protocol = read(bindings.reference(protocol_ref)); protocol_inventory(protocol)
    primary = primary_evidence(protocol['external_primary'], bindings)
    accepted = read(bindings.reference(followup_admission_ref))
    require(accepted['schema'] == ADMISSION_SCHEMA and accepted['complete'] is True
            and accepted['implementation_admission_passed'] is True and accepted['protocol'] == protocol_ref
            and accepted['total_unconditional_attempts'] == TOTAL_ATTEMPTS
            and accepted['selected_full_geometry_gate_satisfied'] is True,
            'Completed follow-up admission required before comparison')
    fresh = read(bindings.reference(accepted['statistics_result']))
    require(fresh['schema'] == statistics.SCHEMA and fresh['complete'] is True and fresh['comparisons'] == [],
            'Complete separate follow-up reduction required')
    check_report_inventory(fresh, protocol['arms'], accepted['populations'])
    old = primary['statistics']
    require(fresh['target_and_regions_sha256'] == old['target_and_regions_sha256'] == protocol['target_and_regions_sha256']
            and fresh['target_descriptor'] == old['target_descriptor']
            and fresh['gates'] == old['gates'] == statistics.GATES,
            'Cannot compare unmatched target, measure, classifier or gates')
    all_arms = primary['protocol']['arms']+protocol['arms']
    combined = dict(old['arms'], **fresh['arms'])
    check_report_inventory(dict(arms=combined), all_arms)
    comparisons = [comparison_pair(c['left'], c['right'], combined, c['historical_failed_strata'])
                   for c in protocol['external_comparisons']]
    diagnostics = admission.summarize_diagnostics(dict(arms=combined, comparisons=comparisons))
    bindings.recheck()
    return dict(schema=COMPARISON_SCHEMA, complete=True, protocol=protocol_ref,
        external_primary=protocol['external_primary'], followup_admission=followup_admission_ref,
        statistics_result=accepted['statistics_result'], target_and_regions_sha256=protocol['target_and_regions_sha256'],
        target_descriptor=old['target_descriptor'], gates=statistics.GATES,
        arms=combined, comparisons=comparisons, retained_primary_comparisons=old['comparisons'],
        statistical_diagnostics=diagnostics, original_primary_attempts=262144,
        fresh_followup_attempts=TOTAL_ATTEMPTS, new_populations=24, reused_primary_populations=16,
        missing_studies=[], input_sha256=bindings.files,
        source_sha256={str(p): sha(p) for p in local_sources(__file__).values()},
        regional_convergence_established=False, unseen_support_verdict='unresolved',
        physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        new_pose_draws=0, new_Poisson_clouds=0, new_atom_geometry_queries=0, parsed_scientific_rows=0,
        scope='Independent linear population comparisons, stages retained separately. Missing studies cleared means executed, not passed. Unseen/full-vessel mass and finite assembly remain separate obligations.')


def join(plan_path):
    bindings = admission.Bindings(); path = Path(plan_path).resolve()
    plan = bindings.read(dict(path=str(path),sha256=sha(path)))
    require(plan['schema'] == admission.PLAN_SCHEMA,'Wrong admission plan')
    protocol, execution, status, stat_plan, report = [bindings.read(plan[k]) for k in
        ('protocol','execution_plan','execution_status','statistics_plan','statistics_result')]
    inventory = protocol_inventory(protocol)
    primary = primary_evidence(protocol['external_primary'], bindings)
    require(protocol['target'] == primary['protocol']['target']
            and protocol['target_and_regions_sha256'] == primary['protocol']['target_and_regions_sha256'],
            'Follow-up physical target changed')
    old_seeds = {p[k] for a in primary['protocol']['arms'] for p in a['populations'] for k in ('seed','audit_seed')}
    require(not old_seeds.intersection(p[k] for a in protocol['arms'] for p in a['populations'] for k in ('seed','audit_seed')),
            'Follow-up reused primary physical/audit seed')
    expected_jobs={(f'{arm}-{population}',phase) for arm,population in inventory
                   for phase in ('producer','algebra','labels','selection','geometry')} | {('all','statistics')}
    jobs=[(j['population'],j['phase']) for j in execution['jobs']]
    require(len(jobs)==len(expected_jobs) and set(jobs)==expected_jobs,'Execution phase inventory differs')
    require(Path(plan['execution_status']['path']).resolve() == Path(execution['root']).resolve()/'execution/summary.json',
            'Admission needs the final controller summary')
    terminals = admission.completed_execution(execution,status,plan['execution_plan'],plan['protocol'],bindings)
    source_files = {}
    for source in local_sources(__file__).values():
        source = Path(source).resolve(); digest = sha(source)
        require(execution['files'].get(str(source)) == digest, 'Admission source is outside frozen execution closure')
        bindings.bind(source,digest); source_files[str(source)] = digest
    admission.require_ref_in_map(plan['statistics_result'],terminals,'Statistics terminal')
    require(stat_plan['schema'] == statistics.PLAN_SCHEMA and report['schema'] == statistics.SCHEMA
            and report['complete'] is True and report['plan_sha256'] == plan['statistics_plan']['sha256'],
            'Wrong completed statistics artifact')
    for name in ('target','target_and_regions_sha256','strata','gates','comparisons'):
        require(stat_plan[name] == protocol[name], 'Statistics declaration differs: '+name)
    target_id = protocol['target_and_regions_sha256']
    require(report['target_and_regions_sha256'] == target_id and report['gates'] == protocol['gates'],
            'Statistics target or gates differ')
    admission.sources_and_inputs(report,execution,bindings)
    admission.require_ref_in_map(plan['statistics_plan'],report['input_sha256'],'Statistics plan')
    require({a['id'] for a in stat_plan['arms']} == {a['id'] for a in protocol['arms']}
            and len(stat_plan['arms']) == len(protocol['arms']) == len(report['arms']), 'Statistics arm inventory differs')
    slots = {(s['arm'],s['id']):s for s in plan['populations']}
    require(len(slots) == len(plan['populations']) and set(slots) == set(inventory), 'Admission population inventory differs')
    stats = {a['id']:a for a in stat_plan['arms']}; accepted = []
    for arm in protocol['arms']:
        stat_arm = stats[arm['id']]; records = report['arms'][arm['id']]['populations']
        require(stat_arm['samples'] == arm['samples'] and len(stat_arm['populations']) == len(records) == 8,
                'Statistics population allocation differs')
        require([p['id'] for p in stat_arm['populations']] == [p['id'] for p in arm['populations']]
                == [p['id'] for p in records], 'Statistics population order/inventory differs')
        for population, stat_slot, record in zip(arm['populations'],stat_arm['populations'],records):
            accepted.append(admission.join_population(arm,population,slots[(arm['id'],population['id'])],stat_slot,record,
                protocol['target'],target_id,report['target_descriptor'],execution,terminals,bindings))
    require([(c['left'],c['right']) for c in report['comparisons']] ==
            [(c['left'],c['right']) for c in protocol['comparisons']], 'Statistics comparison inventory differs')
    diagnostics=admission.summarize_diagnostics(report)
    bindings.finish()
    return dict(schema=ADMISSION_SCHEMA,complete=True,implementation_admission_passed=True,
        execution_lifecycle_passed=True,selected_full_geometry_gate_satisfied=True,
        plan_sha256=bindings.files[str(path)],protocol=plan['protocol'],statistics_result=plan['statistics_result'],
        target_and_regions_sha256=target_id,study_scope=protocol['study_scope'],populations=accepted,
        total_unconditional_attempts=sum(p['samples'] for p in accepted),
        selected_full_geometry_rows=sum(p['selected_rows'] for p in accepted),input_sha256=bindings.files,
        source_sha256=source_files,
        statistical_diagnostics=dict(source=plan['statistics_result'],**diagnostics),
        missing_studies=['external_primary_comparison'], external_primary=protocol['external_primary'],
        primary_is_external_control=True,
        regional_convergence_established=False,unseen_support_verdict='unresolved',
        physical_campaign_gate_open=False,full_vessel_gate_open=False,assembly_gate_open=False,
        new_pose_draws=0,new_Poisson_clouds=0,new_atom_geometry_queries=0,parsed_scientific_rows=0,
        scope='Bound implementation evidence only. Selected maxima rely on the frozen worker all-attempt scan; '
              'pre-draw chronology relies on external review. Neither sampling convergence, unseen mass nor assembly follows.')
