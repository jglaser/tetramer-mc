#!/usr/bin/env python3
"""Freeze a reviewed mixed v6/v7 campaign; never run science or read row streams.

Inputs and a separate review are explicit JSON files. Admission reuses pinned
completed reference receipts, authenticates both embedded producer bundles,
checks all 18 original diagnostic terminals and the once-only cost summary,
and freezes fresh streams plus selected-audit IDs before the first draw.
No observer, shape matcher, proposal law, sampler or geometry audit is called.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys

import analyze_native_class_physical_populations as statistics
import native_class_line_physical_algebra_audit as streaming
import native_class_physical_labels as labels
import native_class_selected_geometry as selected
from analyze_mobile_native_pocket import local_sources
from prepare_hard_free_line_fresh import ROOTS, DECLARATIONS, seeds_in
from run_mobile_posterior_pilot import verify_bundle
from run_native_class_line_probe import validate_terminal

require, read, sha, write = streaming.require, streaming.read, streaming.sha, streaming._write
SCHEMA = 'native-class-physical-protocol-v1'
INPUT_SCHEMA = 'native-class-physical-inputs-v1'
REVIEW_SCHEMA = 'native-class-physical-freeze-review-v1'
POLICY = 'bound-predecessors-v1'
PHASES = ('producer', 'algebra', 'labels', 'selection', 'geometry', 'statistics')
PRODUCERS = {
    'hard_free': dict(schema='importance-latent-region-normalizer-v6',
        executable='752c5d7aa36249cb923ca0959b0872b4f96204101090ac1cfca4606732554c7b',
        source_bundle='30f7234102da852263c2853d3d32a885f9913553f3c632086913da6cb23f3ed3'),
    'class': dict(schema='importance-latent-region-normalizer-v7',
        executable='88c7341cfb080c0d5c6fa68c6062428eb19d938fd7cdc2755af52f5fc06c428f',
        source_bundle='dcf06c0d6dd5b4e94e8f11db91dc09662ed9b1df09019423a5d8fd429e4f47d5'),
}
GUIDES = {'hard_free': '085e5e9d348698802becc11c1a4b9a5ff5cbc9894e8942e73df8fb15ba48a439',
          'class': 'f14837081dc2fb08873d4617b2d9bac3b56cf55640242a77fe8b5cd968cb329e'}
COMPILED_SHA = 'dbb3c3e32259f506b9c979ebb9d1fd773cb78507c1f808fd8dd0ed319e2eade4'
DEFINITION_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
OLD_R5_SHA = '76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae'
WITNESS_MANIFEST_SHA = 'ef956cda08b762fb3d398e5446b9db8d547152e394072b1f28dae04c59f70301'
CHANNELS = [dict(**{'class': name}, probability=.2, **({} if orthant is None else dict(orthant=orthant)))
            for name, orthant in [('hard_free', None), ('contact_without_native', None),
                                 ('contact_without_native', 22), ('contact_without_native', 62), ('native', 55)]]
PINS = {
    'proposal_plan': '7f1bfeb01f8c0d371a5e1f14fbc66371b68fb485452be5fe25f5a2ca72fc246e',
    'hard_prerequisites': 'b4efafd69f620f54f03357dc85f2ef3b89ec9f9292fa717c3f67157a01dcb527',
    'hard_reference': '83a48a1408c913396c18e7e83b748770042e3b756b30a53ec5b02894ace95e9a',
    'hard_independent': '71c220275fa48c554b4e5ee74100ffe2a64017318b86af0298ffc1c3cb0c5912',
    'class_rust': '11582f92046963fe8e688fcac7001baa5d832b9457d0cb797aa49cd20ed8873a',
    'class_sphere_allocation': 'cb34b3f19c9b10ad7d570a54d3edacee9bb9fcdbb580ee13a2c85ae4237ed8c0',
    'class_sphere_binding': '7132bd5bd42318a2bcb2c529bc01f56dcaf41572a97f6406f2606bc7b2cd29e9',
    'class_sphere_execution': '583e9b855284beb2f9b480be26826933f6c1902e03628ebccf6d5fe785762e63',
    'class_sphere_analytic': '150ca537cefb9a925d1b41d4b9f707e6bda114bd8cdb3808b2fc8c9682c5ab9f',
    'sphere_algebra': '8d719ff6cf8fbedb1a20f9be31e78d4a9ba3bb8899fd992ec41fb4376758e389',
    'bridge_validation': '267485b1821528bad100c4a98d64cb6559b3076f77a3c97dd0153aeb6f14e4cc',
    'bridge_review': '43b9676f7405b3a4535a67436b85f662e1a0e9ea4d9b4b36b3a133f0cea19fe5',
}
CODE_ENTRIES = ('prepare_native_class_physical_campaign.py', 'native_class_physical_stage.py',
                'run_native_class_physical_campaign.py', 'native_class_line_physical_algebra_audit.py',
                'native_class_physical_labels.py', 'native_class_selected_geometry.py',
                'analyze_native_class_physical_populations.py')


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and '..' not in path.parts, 'Unsafe relative archive member')
    return path


def arm_design(scope):
    require(scope in ('primary_only', 'full_declared_campaign'), 'Unknown study scope')
    arms = [dict(id=name, samples=16384, alpha=.5, lambda_ratio=128.) for name in ('hard_free', 'class')]
    if scope == 'full_declared_campaign':
        arms += [dict(id='class_large', samples=65536, alpha=.5, lambda_ratio=128.),
                 dict(id='class_defensive', samples=16384, alpha=.2, lambda_ratio=128.),
                 dict(id='class_intensity', samples=16384, alpha=.5, lambda_ratio=64.)]
    return arms


def stream_seed(namespace, arm, population, role):
    require(type(namespace) is str and namespace and role in ('physical', 'audit'), 'Invalid seed namespace/role')
    return int.from_bytes(hashlib.sha256(f'{namespace}/{arm}/{population}/{role}'.encode()).digest()[:8], 'little')


def seed_inventory(roots):
    """Search declaration filenames only; never inspect samples/trajectories."""
    require(all(Path(path).is_dir() for path in roots), 'Missing seed inventory root')
    argv = ['rg', '--files', '--hidden', '--no-ignore', *roots]
    for name in [*DECLARATIONS, 'execution-plan.json']: argv += ['-g', name]
    result = subprocess.run(argv, text=True, capture_output=True)
    require(result.returncode in (0, 1), 'Seed declaration search failed')
    files = {}; seen = set()
    for path in sorted({str(Path(v).resolve()) for v in result.stdout.splitlines()}):
        values = seeds_in(read(path)); seen.update(values)
        files[path] = dict(sha256=sha(path), declared_seeds=sorted(values))
    require(files, 'Empty seed declaration inventory')
    return dict(roots=roots, filename_patterns=[*DECLARATIONS, 'execution-plan.json'], files=files,
                seeds=sorted(seen), scope='Declaration metadata only; no sample or trajectory streams.')


def validate_limits(limits, resources):
    require(set(limits) == set(PHASES), 'All six phase budgets required')
    for phase, spec in limits.items():
        required = {'cpu_limit_seconds', 'wall_limit_seconds', 'address_space_limit_bytes'}
        if phase in ('labels', 'geometry'): required.add('max_record_bytes')
        require(set(spec) == required, 'Unexpected/missing phase budget fields: '+phase)
        require(all(type(v) is int and v > 0 for v in spec.values()), 'Phase budgets must be positive integers')
        require(spec['wall_limit_seconds'] >= spec['cpu_limit_seconds'], 'Wall budget is below CPU budget')
    require(all(type(resources[k]) is int for k in ('maximum_workers', 'threads_per_worker',
                                                  'global_scientific_slots', 'global_scientific_threads'))
            and resources['maximum_workers'] == resources['threads_per_worker'] == 1
            and resources['global_scientific_slots'] == 8 and resources['global_scientific_threads'] == 32,
            'One-worker workflow and original global capacity limits required')
    for key in ('estimated_output_bytes', 'minimum_free_bytes'):
        require(type(resources[key]) is int and resources[key] > 0, 'Missing positive storage budget')
    require(resources['minimum_free_bytes'] >= resources['estimated_output_bytes'], 'Storage reserve is below estimated outputs')


def _proposal_evidence(evidence, bindings):
    plan, done, summary = [evidence[k] for k in ('proposal_plan', 'proposal_execution', 'proposal_summary')]
    plan_ref, done_ref, summary_ref = [evidence['_refs'][k] for k in ('proposal_plan', 'proposal_execution', 'proposal_summary')]
    root = Path(plan_ref['path']).parent
    require(plan['schema'] == 'native-class-line-reviewed-execution-v1' and plan['ready'] is True
            and plan['physical_clouds'] == 0 and len(plan['jobs']) == 18, 'Wrong original diagnostic plan')
    require(Path(done_ref['path']) == root/'execution/summary.json' and done['complete'] is True
            and done['passed'] is True and len(done['completed']) == 18 and done['plan_sha256'] == plan_ref['sha256']
            and done['fresh_draws'] == 1024 and done['saved_development_probes'] == 40
            and done['new_physical_clouds'] == 0 and not (root/'execution/failure.json').exists(),
            'All 18 original proposal jobs must finish cleanly before preparation')
    for ordinal, (job, complete) in enumerate(zip(plan['jobs'], done['completed'])):
        require(complete['id'] == job['id'] and complete['terminal'] == job['terminal'], 'Original job order/terminal differs')
        result = read(bindings.bind(complete['terminal'], complete['sha256']))
        validate_terminal(job, result, plan)
        folder = root/'execution'/f'{ordinal:02}-{job["id"]}'
        exit_record = read(bindings.bind(folder/'exit.json'))
        require(type(exit_record['returncode']) is int and exit_record['returncode'] == 0
                and exit_record['error'] is None and exit_record['child_started'] is True
                and exit_record['child_drained'] is True, 'Original diagnostic has an unsuccessful exit')
        require(read(bindings.bind(folder/'success.json')) == complete, 'Original success binding differs')
    require(summary['schema'] == 'native-class-line-proposal-diagnostic-summary-v1' and summary['complete'] is True
            and summary['execution_plan_sha256'] == plan_ref['sha256']
            and summary['execution_summary_sha256'] == done_ref['sha256'] and summary['new_Poisson_clouds'] == 0
            and summary['full_vessel_gate_open'] is False and summary['assembly_gate_open'] is False,
            'Completed one-time proposal/cost summary required')
    require(set(summary['arms']) == {'hard_free', 'class'} and len(summary['saved_development_probes']) == 40,
            'Proposal summary allocation differs')
    for arm in summary['arms'].values():
        require(len(arm['populations']) == 4 and all(p['attempted'] == 128 for p in arm['populations']),
                'Proposal summary lost unconditional draws')
        for population in arm['populations']:
            for key in ('draw_cpu_seconds', 'density_cpu_seconds', 'observer_cpu_seconds'):
                value = population['cpu_seconds'][key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid proposal cost evidence')
    return summary_ref


def _reference_evidence(evidence):
    h = evidence['hard_prerequisites']; hr = evidence['hard_reference']; hi = evidence['hard_independent']
    require(h['schema'] == 'hard-free-line-physical-prerequisites-v1' and h['complete'] is True
            and h['all_checks_passed'] is True and h['fresh_proposal_validated'] is True, 'Incomplete optimized-H prerequisites')
    for obj in (h, hr):
        require(obj['complete'] is True and obj['binary_sha256'] == PRODUCERS['hard_free']['executable']
                and obj['source_bundle_sha256'] == PRODUCERS['hard_free']['source_bundle'], 'H exact-binary evidence differs')
    require(hr['reference']['complete'] is True and hr['reference']['primary_attempts'] == 16384
            and len(hr['reference']['jobs']) == 2, 'Incomplete H sphere references')
    for job in hr['reference']['jobs']:
        require(job['samples'] == 8192 and job['manifest']['schema'] == PRODUCERS['hard_free']['schema']
                and all(job[k]['passed'] is True and job[k]['attempted_denominator'] == 8192
                        for k in ('hard', 'depletion', 'latent_volume')), 'H analytic reference failed')
    require(hi['complete'] is True and hi['total_audited_rows'] == 16704
            and hi['fresh_independent_reference_draws'] == 16384, 'Incomplete independent H reference')
    cr, allocation, binding, execution, analytic = [evidence[k] for k in
        ('class_rust', 'class_sphere_allocation', 'class_sphere_binding', 'class_sphere_execution', 'class_sphere_analytic')]
    require(cr['complete'] is True and cr['passed'] is True and cr['test_count'] == 94
            and cr['executable_sha256'] == PRODUCERS['class']['executable']
            and cr['source_bundle_sha256'] == PRODUCERS['class']['source_bundle'], 'Class exact-binary Rust validation differs')
    require(all(obj['executable_sha256'] == PRODUCERS['class']['executable'] for obj in (binding, execution, analytic))
            and all(obj['allocation_sha256'] == evidence['_refs']['class_sphere_allocation']['sha256'] for obj in (binding, execution, analytic)),
            'Class sphere binary/allocation differs')
    require(all(obj['complete'] is True and obj['passed'] is True for obj in (execution, analytic))
            and execution['failure'] is None and execution['remaining_jobs_not_started'] == execution['retries'] == 0
            and len(execution['completed']) == 32 and analytic['attempts'] == allocation['total_attempts'] == 32768
            and len(analytic['populations']) == 16 and analytic['failed_tests'] == [], 'Incomplete class sphere reference')
    require(execution['binding_sha256'] == evidence['_refs']['class_sphere_binding']['sha256'], 'Sphere execution binding differs')
    replay = evidence['sphere_algebra']
    require(replay['complete'] is True and replay['passed'] is True and replay['all_rows_algebra'] == 32768
            and len(replay['populations']) == 16 and replay['retries'] == 0, 'Sphere algebra compatibility is incomplete')
    for key in ('bridge_validation', 'bridge_review'):
        require(evidence[key]['complete'] is True and evidence[key]['passed'] is True, 'Mixed-schema implementation validation failed')
    bridge = evidence['bridge_validation']
    require(bridge['returncode'] == 0 and bridge['source_before'] == bridge['source_after'], 'Mixed bridge source changed during validation')


def _sources():
    base = Path(__file__).resolve().parent; result = {}
    for name in CODE_ENTRIES:
        require((base/name).is_file(), 'Missing campaign implementation '+name)
        for key, path in local_sources(base/name).items():
            require(key not in result or result[key] == path, 'Ambiguous frozen source name')
            result[key] = path
    return result


def _historical_evidence(history, target_id, bindings):
    """Retain the expanded diagnostic watchlist without adding pass criteria."""
    require(history.get('complete') is True, 'Historical inventory must be complete')
    require(history.get('policy') == 'retention_only', 'Historical inventory must have retention_only policy')
    require(history['schema'] == 'native-class-historical-failed-strata-v1' and history['target_and_regions_sha256'] == target_id
            and history['entries'] and history['source_receipts'], 'Missing target-bound historical failure inventory')
    for ref in history['source_receipts']: bindings.reference(ref)
    for key in ('derived_from', 'provenance_review'):
        if key in history: bindings.reference(history[key])
    seen = set()
    for entry in history['entries']:
        require(set(entry) == {'family', 'bin', 'region'} and entry['family'] in statistics.BINS
                and type(entry['bin']) is int and 0 <= entry['bin'] < statistics.BINS[entry['family']]
                and entry['region'] in statistics.DECISIONS, 'Invalid historical failed stratum')
        key = (entry['family'], entry['bin'], entry['region']); require(key not in seen, 'Duplicate historical failure'); seen.add(key)
    require({('orthant', 22, 'competing'), ('orthant', 62, 'competing'), ('orthant', 55, 'native'),
             ('orthant', 55, 'native_remainder')} <= seen, 'Known original failures omitted')


def validate(inputs_path, review_path):
    """Read metadata/source only. Incomplete proposal evidence stops here."""
    bindings = statistics.Bindings()
    inputs_path, review_path = bindings.bind(Path(inputs_path).resolve()), bindings.bind(Path(review_path).resolve())
    inputs, review = read(inputs_path), read(review_path)
    require(inputs['schema'] == INPUT_SCHEMA and review['schema'] == REVIEW_SCHEMA, 'Wrong preparation input/review schema')
    require(review['complete'] is True and review['passed'] is True
            and review['inputs'] == bound(inputs_path) and review['study_scope'] == inputs['study_scope']
            and review['stage_materialization_policy'] == POLICY and review['deterministic_materialization_authorized'] is True
            and review['historical_inventory_complete'] is True and review['physical_gates_remain_closed'] is True,
            'Explicit matching scope/materialization review required')
    arms = arm_design(inputs['study_scope']); validate_limits(inputs['phase_limits'], inputs['resources'])
    require(review['phase_limits'] == inputs['phase_limits'] and review['resources'] == inputs['resources'], 'Unreviewed resource budgets')
    refs = inputs['prerequisites']; require(set(refs) == set(PINS) | {'proposal_execution', 'proposal_summary'}, 'Incomplete prerequisite inventory')
    evidence = {'_refs': refs}
    for name, ref in refs.items():
        if name in PINS: require(ref['sha256'] == PINS[name], 'Unreviewed prerequisite identity '+name)
        evidence[name] = read(bindings.reference(ref))
    _proposal_evidence(evidence, bindings); _reference_evidence(evidence)
    cost = review['cost_review']
    require(cost['proposal_summary_sha256'] == refs['proposal_summary']['sha256']
            and cost['class_physical_cost_unknown'] is True and cost['endpoint_cost_unknown'] is True
            and type(cost['rationale']) is str and cost['rationale'].strip(), 'Missing explicit cost decision and uncertainty')
    region, descriptor, target_id = statistics.target_identity(inputs['target'], bindings)
    require(inputs['target']['native_definition']['sha256'] == DEFINITION_SHA
            and inputs['target']['old_r5_region']['sha256'] == OLD_R5_SHA, 'Changed original classifier/old-R5 definition')
    config = read(bindings.reference(inputs['config'])); bindings.reference(inputs['shape'])
    require(inputs['shape']['sha256'] == region['shape_sha256'] and sha(config['shape']) == region['shape_sha256'], 'Physical shape differs')
    for key, other in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                       ('depletant_radius', 'depletant_radius'), ('activity', 'reservoir_density'), ('physical_metric', 'metadata')]:
        require(region[key] == config[other], 'Changed physical config '+key)
    require(config['fixed_poses'] == region['physical_fixed_neighbors'] and len(config['fixed_poses']) == 2, 'Changed two-neighbor scaffold')
    definition_path = Path(inputs['target']['native_definition']['path']); definition = read(definition_path)
    for name, digest in definition['input_sha256'].items(): bindings.bind(definition_path.parent/'inputs'/safe_relative(name), digest)
    labels.validate_classifier_target(config, region['shape_sha256'], definition, definition_path)
    require(inputs['compiled_native']['sha256'] == COMPILED_SHA, 'Compiled native identity differs')
    compiled = read(bindings.reference(inputs['compiled_native']))
    require(compiled['source_definition_sha256'] == DEFINITION_SHA and compiled['source_input_sha256'] == definition['input_sha256']
            and compiled['fixed_poses'] == config['fixed_poses'], 'Compiled definition/scaffold closure differs')
    witness_source = inputs['shape_compatibility_source']
    require(witness_source['field'] == 'native_shape_compatibility'
            and witness_source['manifest']['sha256'] == WITNESS_MANIFEST_SHA, 'Unreviewed shape witness source')
    wm = read(bindings.reference(witness_source['manifest'])); witness = wm[witness_source['field']]
    require(wm['compiled_native_sha256'] == COMPILED_SHA and wm['shape_sha256'] == region['shape_sha256']
            and witness['compatible'] is True and witness['compiled_sha256'] == COMPILED_SHA
            and witness['expected_shape_sha256'] == region['shape_sha256']
            and witness['hard_valid_implication_within_tolerance'] is True,
            'Existing static shape compatibility witness differs')
    guides = {name: read(bindings.reference(ref)) for name, ref in inputs['guides'].items()}
    require(set(guides) == {'hard_free', 'class'} and all(inputs['guides'][name]['sha256'] == GUIDES[name] for name in GUIDES),
            'Unvalidated guide identity')
    for name, guide in guides.items():
        require(guide['schema'] == ('defensive-hard-free-line-guide-v1' if name == 'hard_free' else 'defensive-native-class-line-guide-v1')
                and guide['region_sha256'] == inputs['target']['region']['sha256']
                and guide['raw_translation_axes'] == [0, 1, 2] and guide['conditional_probability'] == 1.
                and guide['minimum_conditional_mass'] == 1e-12 and guide['defensive_uniform_shell_probability'] == .5
                and len(guide['gaussian_components']) == 92, 'Original guide law changed')
    require(guides['hard_free']['gaussian_components'] == guides['class']['gaussian_components']
            and guides['class']['class_channels'] == CHANNELS
            and guides['class']['compiled_native']['sha256'] == COMPILED_SHA, 'Changed Gaussian mixture/class channels')
    producers = {}
    for name in PRODUCERS:
        item = inputs['producers'][name]
        for key in ('executable', 'source_bundle'):
            require(item[key]['sha256'] == PRODUCERS[name][key], 'Unvalidated producer identity '+name+'/'+key)
            bindings.reference(item[key])
        bundle, hashes = verify_bundle(item['executable']['path'], item['source_bundle']['path'], item['source_archive'])
        for member, digest in hashes.items(): bindings.bind(Path(item['source_archive'])/safe_relative(member), digest)
        producers[name] = dict(input=item, bundle=bundle)
    history = read(bindings.reference(inputs['historical_failed_strata']))
    _historical_evidence(history, target_id, bindings)
    source = _sources()
    for name, path in source.items(): bindings.bind(path)
    # New orchestration code is reviewed separately. Every reused bridge source
    # covered by its passing suite must still be those exact validated bytes.
    for name, digest in evidence['bridge_validation']['source_after'].items():
        basename = Path(name).name
        if basename in source: require(sha(source[basename]) == digest, 'Validated bridge source changed: '+basename)
    runtime = streaming.runtime_identity()
    for path, digest in runtime['file_sha256'].items(): bindings.bind(path, digest)
    bindings.recheck()
    return dict(inputs=inputs, review=review, bindings=bindings, region=region, target_id=target_id,
        descriptor=descriptor, config=config, definition=definition, definition_path=definition_path,
        witness=witness, guides=guides, producers=producers, history=history, source=source, runtime=runtime,
        arms=arms, inputs_path=inputs_path, review_path=review_path)


def prepare(out, inputs_path, review_path):
    out = Path(out).resolve()
    require(not out.exists() and out.parent.is_dir(), 'Fresh campaign root with existing parent required')
    context = validate(inputs_path, review_path); inputs = context['inputs']; bindings = context['bindings']
    require(shutil.disk_usage(out.parent).free >= inputs['resources']['minimum_free_bytes'], 'Reviewed storage reserve is unavailable')
    all_seeds = [stream_seed(inputs['seed_namespace'], arm['id'], f'r{i:02}', role)
                 for arm in context['arms'] for i in range(8) for role in ('physical', 'audit')]
    require(len(set(all_seeds)) == len(all_seeds), 'New population/audit seed collision')
    inventory = seed_inventory([str(Path(__file__).resolve().parents[1]/'runs'),
                                str(Path(__file__).resolve().parents[1]/'results'), *ROOTS])
    require(not set(all_seeds).intersection(inventory['seeds']), 'Seed collision with existing scientific declaration')
    bindings.recheck()
    out.mkdir()
    try:
        write(out/'preparation-claim.json', dict(schema=SCHEMA, root=str(out),
            inputs=bound(context['inputs_path']), review=bound(context['review_path']),
            study_scope=inputs['study_scope'], launched=False, retries=0))
        return _materialize(out, context, inventory)
    except BaseException as error:
        write(out/'preparation-failure.json', dict(schema=SCHEMA, complete=False, launched=False,
            phase='preparation', error_type=type(error).__name__, error=str(error),
            input_sha256=bindings.files, retries=0,
            scope='Preserved partial preparation; never execute or resume this directory.'))
        raise


def _materialize(out, context, inventory):
    inputs, bindings = context['inputs'], context['bindings']
    common = out/'common'; code = out/'code'
    common.mkdir(); code.mkdir(); (out/'analysis').mkdir(); (common/'preselections').mkdir()
    copied = {}
    def copy_file(source, destination):
        source = Path(source).resolve()
        require(str(source) in bindings.files and sha(source) == bindings.files[str(source)], 'Copy source was not admitted')
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        reference = bound(destination)
        require(reference['sha256'] == bindings.files[str(source)], 'Copied bytes differ from admitted source')
        copied[reference['path']] = reference['sha256']
        return reference
    copy_file(context['inputs_path'], common/'inputs.json'); review_ref = copy_file(context['review_path'], common/'review.json')
    for name, source in context['source'].items(): copy_file(source, code/name)
    target = {}
    for name in ('region', 'old_r5_region'):
        target[name] = copy_file(inputs['target'][name]['path'], common/(name+'.json'))
    target['native_definition'] = copy_file(context['definition_path'], out/'native-definition/definition.json')
    for name in context['definition']['input_sha256']:
        copy_file(context['definition_path'].parent/'inputs'/name, out/'native-definition/inputs'/name)
    shape_ref = copy_file(inputs['shape']['path'], common/'shape.json')
    compiled_ref = copy_file(inputs['compiled_native']['path'], common/'compiled-native.json')
    write(common/'shape-compatibility.json', context['witness']); witness_ref = bound(common/'shape-compatibility.json')
    config = copy.deepcopy(context['config']); config['shape'] = shape_ref['path']; write(common/'config.json', config)
    write(common/'strata.json', statistics.STRATA); write(common/'seed-inventory.json', inventory)
    history_ref = copy_file(inputs['historical_failed_strata']['path'], common/'historical-failed-strata.json')
    prerequisite_refs = {name: copy_file(ref['path'], common/'evidence'/(name+'.json')) for name, ref in inputs['prerequisites'].items()}
    producers = {}
    for name, value in context['producers'].items():
        folder = common/'producers'/name
        exe = copy_file(value['input']['executable']['path'], folder/'latent-region-normalizer')
        bundle = copy_file(value['input']['source_bundle']['path'], folder/'source-bundle.json')
        for member in value['bundle']['files']:
            copy_file(Path(value['input']['source_archive'])/member, folder/'rust-source'/member)
        producers[name] = dict(executable=exe, source_bundle=bundle, source_archive=str(folder/'rust-source'), source_schema=PRODUCERS[name]['schema'])
    arms = []
    for declaration in context['arms']:
        arm = copy.deepcopy(declaration); name = 'hard_free' if arm['id'] == 'hard_free' else 'class'
        guide = copy.deepcopy(context['guides'][name]); guide['defensive_uniform_shell_probability'] = arm['alpha']
        if name == 'class': guide['compiled_native']['path'] = compiled_ref['path']
        guide_path = common/(arm['id']+'-guide.json'); write(guide_path, guide)
        arm.update(source_schema=PRODUCERS[name]['schema'], guide=bound(guide_path), producer=name, populations=[])
        for i in range(8):
            ident = f'r{i:02}'; seed = stream_seed(inputs['seed_namespace'], arm['id'], ident, 'physical')
            audit_seed = stream_seed(inputs['seed_namespace'], arm['id'], ident, 'audit')
            preselection_path = common/'preselections'/f'{arm["id"]}-{ident}.json'
            write(preselection_path, selected.preselection(ident, arm['samples'], audit_seed))
            directory = out/'populations'/arm['id']/ident
            (out/'analysis'/f'{arm["id"]}-{ident}').mkdir(); directory.parent.mkdir(parents=True, exist_ok=True)
            arm['populations'].append(dict(id=ident, seed=seed, directory=str(directory), audit_seed=audit_seed, preselection=bound(preselection_path)))
        arms.append(arm)
    comparisons = [dict(left='hard_free', right='class', historical_failed_strata=context['history']['entries'])]
    if inputs['study_scope'] == 'full_declared_campaign':
        comparisons += [dict(left='class', right=name, historical_failed_strata=context['history']['entries'])
                        for name in ('class_large', 'class_defensive', 'class_intensity')]
    for path, digest in copied.items(): require(sha(path) == digest, 'Copied closure changed before publication')
    files = {str(path.resolve()): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    files.update(context['runtime']['file_sha256'])
    bindings.recheck()
    protocol = dict(schema=SCHEMA, root=str(out), study_scope=inputs['study_scope'], code_directory=str(code),
        stage_materialization_policy=POLICY, materialization_review=review_ref, files=files,
        source_sha256={name: sha(path) for name, path in context['source'].items()}, runtime=context['runtime'],
        python=str(Path(sys.executable).absolute()), target=target, target_and_regions_sha256=context['target_id'],
        strata=statistics.STRATA, strata_file=bound(common/'strata.json'), gates=statistics.GATES,
        config=bound(common/'config.json'), shape=shape_ref, native_identity=dict(compiled_native=compiled_ref, shape_compatibility=witness_ref),
        observer_setup=labels.observer_setup_inventory(out/'native-definition/definition.json'), producers=producers,
        arms=arms, comparisons=comparisons, historical_failed_strata=history_ref, prerequisites=prerequisite_refs,
        historical_failed_strata_policy=context['history']['policy'],
        phase_limits=inputs['phase_limits'], resources=inputs['resources'], cloud_replicates=2,
        total_attempts=sum(8*arm['samples'] for arm in arms), maximum_clouds=2*sum(8*arm['samples'] for arm in arms),
        selected_geometry_max_rows=20*8*len(arms), selected_geometry_max_axis_queries=60*8*len(arms),
        prior_inputs=bindings.files, launched=False, physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        scope='Fresh fixed regional comparison. Every attempted draw retained; no retry, extension, refit, pooling or assembly inference.')
    write(out/'protocol.json', protocol)
    execution = execution_plan(protocol, bound(out/'protocol.json'))
    for path, digest in files.items(): require(sha(path) == digest, 'Frozen closure changed before execution plan publication')
    write(out/'execution-plan.json', execution)
    receipt = dict(schema='native-class-physical-preparation-v1', complete=True,
                root=str(out), protocol=bound(out/'protocol.json'), study_scope=inputs['study_scope'],
                execution_plan=bound(out/'execution-plan.json'), populations=8*len(arms),
                total_attempts=protocol['total_attempts'], launched=False)
    write(out/'preparation.json', receipt)
    return receipt


def execution_plan(protocol, protocol_ref):
    """Exact frozen argv; the generic driver owns only lifecycle execution."""
    from native_class_physical_stage import paths
    root, code = Path(protocol['root']), Path(protocol['code_directory'])
    python = protocol['python']; stage = str(code/'native_class_physical_stage.py'); jobs = []
    def job(identity, population, phase, argv, terminal, contract):
        budget = protocol['phase_limits'][phase]
        jobs.append(dict(id=identity, population=population, phase=phase, argv=argv,
            terminal=dict(path=str(terminal), success_contract=contract),
            **{k: budget[k] for k in ('cpu_limit_seconds', 'wall_limit_seconds', 'address_space_limit_bytes')}))
    for arm in protocol['arms']:
        producer = protocol['producers'][arm['producer']]['executable']['path']
        for pop in arm['populations']:
            key = arm['id']+'-'+pop['id']; output = paths(root, arm['id'], pop['id'])
            job(key+'-producer', key, 'producer', [producer, '--config', protocol['config']['path'],
                '--region', protocol['target']['region']['path'], '--importance-guide', arm['guide']['path'],
                '--out', pop['directory'], '--samples', str(arm['samples']), '--seed', str(pop['seed']),
                '--cloud-replicates', '2', '--lambda-ratio', str(arm['lambda_ratio'])],
                Path(pop['directory'])/'summary.json', 'complete')
            job(key+'-algebra', key, 'algebra', [python, '-B', str(code/'native_class_line_physical_algebra_audit.py'),
                '--root', pop['directory'], '--out', str(output['algebra'])], output['algebra'], 'complete_and_passed')
            for phase, terminal in [('labels', 'labels'), ('selection', 'selection_review'), ('geometry', 'geometry')]:
                job(key+'-'+phase, key, phase, [python, '-B', stage, '--protocol', protocol_ref['path'],
                    '--protocol-sha256', protocol_ref['sha256'], '--population', key, '--phase', phase],
                    output[terminal], 'complete_and_passed')
    output = paths(root)
    job('statistics', 'all', 'statistics', [python, '-B', stage, '--protocol', protocol_ref['path'],
        '--protocol-sha256', protocol_ref['sha256'], '--population', 'all', '--phase', 'statistics'], output['statistics'], 'complete')
    files = dict(protocol['files']); files[protocol_ref['path']] = protocol_ref['sha256']
    resolutions = {entry['argv'][0]: str(Path(entry['argv'][0]).resolve()) for entry in jobs}
    for path in resolutions.values(): files[path] = sha(path)
    return dict(schema='native-class-physical-execution-v1', root=str(root), maximum_workers=1, threads=1,
                files=files, executable_resolutions=resolutions, jobs=jobs,
                preparation_receipt=str(root/'preparation.json'),
                scope='Exactly the frozen scientific populations and analyses; no retries or admission stage in this lifecycle.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.out, args.inputs, args.review), indent=2))
