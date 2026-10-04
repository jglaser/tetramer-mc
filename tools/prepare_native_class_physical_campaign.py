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
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys

import analyze_native_class_physical_populations as statistics
import native_class_line_physical_algebra_audit as streaming
import native_class_physical_labels as labels
import native_class_selected_geometry as selected
import admit_native_class_physical_campaign as admission
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
ORIGINAL_PROFILE = 'original_92'
REVISED_PROFILE = 'revised_support_pilot_116'
REVISED_PINS = dict(
    guide='13e5c31774c77169d7f4604b2a46b80e84b366ff2172fcef3911fab93286a179',
    protocol='2cc3e8ae27566b273e8568b7a7bc463bb56c6a842bf7cc6eef959f5a8e9887ef',
    execution_plan='a847304317aae29967c1aaacc04a12388b34ecbb41af5d4da304e5a8c6d19cee',
    original_plan='61613b40549091e46d5496a9e72559841f613753b34fc6b937419cd212e0be76',
    original_failure='4b2f4f311b231bf9dad2d12bafef2cd7d79c6416c03efd4bb7cc76887bc2b855',
    original_algebra_failure='648cfd1b7856bae27321a907116e33b02c8f201a9ecfc903c523c35599a850e8',
    retained_summary='6063a2e21f1515a82aa97a788f5c997bdb31a0844c9f259c16fcf834faf913a8')
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


def _guide_profile(inputs, guides):
    """Keep the original law intact; the candidate is one explicitly pinned design."""
    profile = inputs.get('class_guide_profile', ORIGINAL_PROFILE)
    require(profile in (ORIGINAL_PROFILE, REVISED_PROFILE), 'Unknown class guide profile')
    expected = dict(GUIDES)
    if profile == REVISED_PROFILE: expected['class'] = REVISED_PINS['guide']
    require(set(guides) == set(expected)
            and all(inputs['guides'][name]['sha256'] == digest for name, digest in expected.items()),
            'Unvalidated guide identity')
    for name, guide in guides.items():
        require(guide['schema'] == ('defensive-hard-free-line-guide-v1' if name == 'hard_free' else 'defensive-native-class-line-guide-v1')
                and guide['region_sha256'] == inputs['target']['region']['sha256']
                and guide['raw_translation_axes'] == [0, 1, 2] and guide['conditional_probability'] == 1.
                and guide['minimum_conditional_mass'] == 1e-12 and guide['defensive_uniform_shell_probability'] == .5,
                'Original guide law changed')
    original, candidate = [guides[name]['gaussian_components'] for name in ('hard_free', 'class')]
    require(len(original) == 92 and guides['class']['class_channels'] == CHANNELS
            and guides['class']['compiled_native']['sha256'] == COMPILED_SHA, 'Changed Gaussian mixture/class channels')
    if profile == ORIGINAL_PROFILE:
        require(original == candidate, 'Changed Gaussian mixture/class channels')
        require('candidate_pilot' not in inputs, 'Candidate evidence requires the explicit revised profile')
        return profile
    require(len(candidate) == 116, 'Revised guide must have exactly 116 components')
    total = math.fsum(c['weight'] for c in original)
    require(math.isfinite(total) and total > 0 and all(math.isfinite(c['weight']) and c['weight'] > 0 for c in original),
            'Invalid original Gaussian weights')
    retained = copy.deepcopy(original)
    for component in retained: component['weight'] = .75*(component['weight']/total)
    require(candidate[:92] == retained, 'Revised guide changed the exact retained 0.75 design')
    centers = []
    for center in range(8):
        additions = candidate[92+3*center:95+3*center]; mean = additions[0]['mean']; centers.append(tuple(mean))
        require(len(mean) == 6 and all(type(v) in (int, float) and math.isfinite(v) for v in mean), 'Invalid training center')
        for addition, width in zip(additions, (.05, .15, .45)):
            require(addition == dict(weight=.25/24, mean=mean,
                    covariance=[[width**2 if a == b else 0. for b in range(6)] for a in range(6)]),
                    'Revised guide changed a fixed center/scale/weight')
    require(len(set(centers)) == 8, 'Repeated revised training center')
    return profile


def _candidate_recovery(protocol, adoption, terminals, proof):
    """Authenticate the failed stop and byte adoption, without decoding raw rows."""
    recovery = protocol['recovery']; root = Path(protocol['root']); old_root = Path(recovery['original_root'])
    require(recovery['schema'] == 'native-class-support-format-recovery-v1'
            and recovery['retained_attempts'] == 128 and recovery['new_draws'] == 384
            and recovery['total_attempts'] == 512 and recovery['scientific_replacements'] == 0
            and recovery['no_outcome_adaptation'] is True and recovery['allocation_changed'] is False,
            'Candidate recovery changed the original allocation')
    old = proof.read(recovery['original_protocol'])
    originals = {}
    for key in ('original_plan', 'original_failure', 'original_algebra_failure', 'retained_summary'):
        require(recovery[key]['sha256'] == REVISED_PINS[key], 'Changed original recovery evidence '+key)
        originals[key] = proof.read(recovery[key])
    plan = originals['original_plan']; failure = originals['original_failure']; stopped = originals['original_algebra_failure']
    require(Path(plan['root']) == old_root and recovery['original_plan']['path'] == str(old_root/'execution-plan.json')
            and failure['plan_sha256'] == recovery['original_plan']['sha256'] and failure['complete'] is False
            and failure['failed_job'] == dict(ordinal=1, id='r00-algebra', population='r00', phase='algebra')
            and failure['unstarted'] == plan['jobs'][2:] and len(failure['completed']) == 1
            and stopped['active_id'] == 0 and stopped['completed_ids'] == []
            and stopped['counts'] == stopped['setup_counts'] == {}
            and stopped['error'] == 'Wrong proposal density/contact format', 'Different original format failure')
    # Use the original helper at its bound self-path, as required by verify_plan.
    path = old_root/'code/run_native_class_physical_campaign.py'
    proof.add_map(plan['files'])
    require(sha(path) == plan['files'][str(path)], 'Original driver changed')
    spec = importlib.util.spec_from_file_location('physical_candidate_original_driver', path)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    require(driver.completed_terminal(old_root, plan, 'r00-producer') == recovery['retained_summary'],
            'Original retained producer is not authenticated')
    for key in ('guide', 'config', 'shape', 'region', 'compiled_native', 'definition'):
        require(protocol[key] == old[key], 'Recovery changed a scientific input '+key)
    for current, previous in zip(protocol['populations'], old['populations']):
        require({k:v for k,v in current.items() if k != 'directory'} == {k:v for k,v in previous.items() if k != 'directory'},
                'Recovery changed seeds or preselected IDs')
    require(adoption['schema'] == recovery['schema'] and adoption['complete'] is True and adoption['passed'] is True
            and adoption['protocol_sha256'] == REVISED_PINS['protocol'] and adoption['new_draws'] == 0
            and adoption['retained_attempts'] == 128 and adoption['sample_rows_parsed'] == adoption['retries'] == 0
            and adoption['retained_files'] == recovery['retained_files']
            and adoption['original_summary'] == recovery['retained_summary']
            and adoption['adopted_summary'] == dict(path=str(root/'queries/r00/summary.json'), sha256=REVISED_PINS['retained_summary'])
            and terminals.get(adoption['adopted_summary']['path']) == REVISED_PINS['retained_summary'],
            'Incomplete or changed retained-r00 adoption')
    proof.add_map(adoption['input_sha256']); proof.reference(adoption['journal'])
    for name, identity in recovery['retained_files'].items():
        relative = safe_relative(name)
        for directory in (Path(recovery['source_directory']), root/'queries/r00'):
            path = proof.bind(directory/relative, identity['sha256'])
            require(path.stat().st_size == identity['bytes'] and not path.is_symlink(), 'Retained file size/type changed')


def _candidate_access(summary):
    require(summary['schema'] == 'native-class-support-pilot-summary-v1' and summary['complete'] is True
            and summary['passed'] is True and summary['protocol_sha256'] == REVISED_PINS['protocol']
            and summary['new_Poisson_clouds'] == summary['raw_rows_read'] == summary['new_geometry_queries'] == summary['retries'] == 0,
            'Incomplete candidate proposal reduction')
    combined = summary['combined']; reports = summary['populations']
    require([r['id'] for r in reports] == [f'r{i:02}' for i in range(4)]
            and all(r['attempted'] == 128 and r['selected_reference_rows'] == 16 for r in reports)
            and combined['attempted_denominator'] == 512 and combined['selected_reference_rows'] == 64,
            'Candidate screen lost planned unconditional denominators')
    for region in ('competing22', 'competing62', 'native55'):
        counts = [r['critical_endpoints'][region] for r in reports]
        require(all(type(n) is int and 0 <= n <= 128 for n in counts)
                and combined['critical_endpoints'][region] == sum(counts) > 0,
                'Candidate has no authenticated critical endpoint access: '+region)
    for channel in (2, 3, 4):
        values = [r['target_line_hit_rates'].get(f'channel:{channel}', dict(selected=0, hits=0)) for r in reports]
        require(all(type(v['selected']) is int and type(v['hits']) is int and 0 <= v['hits'] <= v['selected'] <= 128 for v in values),
                'Invalid candidate selected-line counts')
        value = combined['selected_target_line_access'][str(channel)]
        require(value['selected'] == sum(v['selected'] for v in values)
                and value['hits'] == sum(v['hits'] for v in values) > 0, 'Candidate critical selected lines remain empty')
    require(combined['access_screen'] == 'access_observed_requires_review'
            and all(combined[key] is False for key in ('physical_campaign_gate_open','full_vessel_gate_open','assembly_gate_open'))
            and combined['training_critical_points'] == 8 and combined['nontraining_critical_holdouts'] == 0,
            'Proposal access cannot replace physical convergence or claim independent holdouts')


def _candidate_evidence(inputs, review, bindings):
    refs = inputs['candidate_pilot']
    require(set(refs) == {'protocol','execution_plan','execution_summary','statistics','adoption'}, 'Incomplete candidate evidence inventory')
    proof = admission.Bindings()
    for key in ('protocol', 'execution_plan'):
        require(refs[key]['sha256'] == REVISED_PINS[key], 'Unreviewed candidate '+key)
    protocol, plan, done, summary, adoption = [proof.read(refs[k]) for k in
        ('protocol','execution_plan','execution_summary','statistics','adoption')]
    root = Path(protocol['root'])
    for key, relative in [('protocol','protocol.json'), ('execution_plan','execution-plan.json'),
                          ('execution_summary','execution/summary.json'), ('statistics','analysis/statistics.json'),
                          ('adoption','analysis/r00/adoption.json')]:
        require(refs[key]['path'] == str(root/relative), 'Candidate evidence path differs: '+key)
    require(Path(plan['root']) == root and protocol['schema'] == 'native-class-support-pilot-v1'
            and protocol['guide'] == inputs['guides']['class'] and protocol['guide']['sha256'] == REVISED_PINS['guide']
            and protocol['total_attempts'] == 512 and protocol['new_draws'] == 384 and protocol['retained_attempts'] == 128
            and protocol['new_Poisson_clouds'] == protocol['physical_weight_estimates'] == 0
            and protocol['selected_geometry_rows'] == 64, 'Candidate scope/guide differs')
    for key, ref in [('region',inputs['target']['region']), ('definition',inputs['target']['native_definition']),
                     ('shape',inputs['shape']), ('compiled_native',inputs['compiled_native'])]:
        require(protocol[key]['sha256'] == ref['sha256'], 'Candidate physical target differs: '+key)
    expected = [f'r{i:02}-{phase}' for i in range(4) for phase in ('producer','algebra','labels','geometry')]+['statistics']
    require([job['id'] for job in plan['jobs']] == expected, 'Candidate lifecycle allocation differs')
    terminals = admission.completed_execution(plan, done, refs['execution_plan'], refs['protocol'], proof)
    require(terminals.get(refs['statistics']['path']) == refs['statistics']['sha256'], 'Unbound candidate statistics terminal')
    _candidate_recovery(protocol, adoption, terminals, proof)
    predecessor_refs = {}
    for population in protocol['populations']:
        require(population['samples'] == 128, 'Candidate population size differs')
        hashes = None
        for phase in ('algebra','labels','geometry'):
            identity = population['id']+'-'+phase; path = str(root/'analysis'/population['id']/(phase+'.json'))
            ref = dict(path=path, sha256=terminals[path]); receipt = proof.read(ref); predecessor_refs[identity] = ref
            ids = population['selected_ids'] if phase == 'geometry' else list(range(128))
            require(receipt['schema'] == 'native-class-support-pilot-'+phase+'-v1' and receipt['phase'] == phase
                    and receipt['population'] == population['id'] and receipt['protocol_sha256'] == refs['protocol']['sha256']
                    and receipt['attempted_records'] == 128 and [r['id'] for r in receipt['rows']] == ids
                    and receipt['source_sha256'] == protocol['source_sha256'] and receipt['runtime'] == protocol['runtime']
                    and receipt['new_pose_draws'] == receipt['new_Poisson_clouds'] == receipt['physical_weight_estimates'] == 0,
                    'Incomplete or mismatched candidate '+phase)
            proof.add_map(receipt['input_sha256']); proof.reference(receipt['journal'])
            if phase == 'algebra': hashes = {r['id']:r['sample_record_sha256'] for r in receipt['rows']}
            else: require(all(r['sample_record_sha256'] == hashes[r['id']] for r in receipt['rows']), 'Candidate row-byte identities differ')
    require(summary['predecessor_receipts'] == predecessor_refs, 'Candidate reduction omitted an audit')
    proof.add_map(summary['input_sha256']); _candidate_access(summary)
    require(review.get('class_guide_profile') == REVISED_PROFILE
            and review['cost_review'].get('candidate_statistics_sha256') == refs['statistics']['sha256'],
            'Revised guide/access cost review required')
    proof.finish()
    for path, digest in proof.files.items(): bindings.bind(path, digest)
    return dict(profile=REVISED_PROFILE, evidence=refs, retained_attempts=128, new_draws=384,
        attempted_denominator=512, selected_reference_rows=64, access_screen='access_observed_requires_review',
        binary64_weight_retention=protocol['binary64_weight_retention'],
        physical_campaign_gate_open=False, regional_convergence_established=False,
        scope='Proposal access permits review only; every fresh physical audit, precision, support and sensitivity obligation remains.')


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
    profile = _guide_profile(inputs, guides)
    candidate = _candidate_evidence(inputs, review, bindings) if profile == REVISED_PROFILE else None
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
        arms=arms, inputs_path=inputs_path, review_path=review_path, candidate=candidate)


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
    candidate = copy.deepcopy(context.get('candidate'))
    if candidate is not None:
        candidate['evidence'] = {name:copy_file(ref['path'], common/'evidence'/('candidate-'+name+'.json'))
                                 for name,ref in candidate['evidence'].items()}
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
    if candidate is not None:
        protocol.update(class_guide_profile=REVISED_PROFILE, candidate_pilot=candidate,
                        class_guide_source=inputs['guides']['class'])
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
