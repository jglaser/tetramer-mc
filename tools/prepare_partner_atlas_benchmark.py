#!/usr/bin/env python3
"""Freeze 32 partner-atlas context0 chains; reuse all 40 completed controls.

Metadata and hash operations only. Historical journals are neither scanned nor
replayed; the completed rigid observer authenticates its 24 chains and 16 cached
controls. No source/cloud generation, geometry query, build, or launch occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil

import prepare_evolving_dimer_benchmark as old
import prepare_rigid_surrogate_benchmark as rigid
import prepare_flexible_surrogate_benchmark as flexible
import prepare_two_root_dimer_benchmark as prior
import plot_rigid_surrogate_benchmark as completed
import plot_surrogate_internal_native as native_completed
import run_native_class_physical_campaign as driver

ROOT = Path(__file__).resolve().parents[1]
require, read, sha, record, write = prior.require, prior.read, prior.sha, prior.record, prior.write
Inputs = prior.Inputs
SELF = 'tools/prepare_partner_atlas_benchmark.py'
TEST = 'tests/test_prepare_partner_atlas_benchmark.py'
DRIVER = 'tools/run_native_class_physical_campaign.py'
ARMS = ['partner_atlas_direct', 'partner_atlas_m1', 'partner_atlas_m8', 'partner_atlas_flat8']
CONTROL_ARMS = ['local', 'm4', *rigid.ARMS]
RUST_TESTS = ['tests/partner_atlas.rs', 'tests/flexible_surrogate_chain.rs', 'tests/partner_atlas_stationarity.rs']
LIMITS = dict(cpu_limit_seconds=1800, wall_limit_seconds=3600, address_space_limit_bytes=8*2**30)
ROLES = dict(proposal='partner_atlas/proposal', inner_accept='partner_atlas/inner_accept',
             bath='partner_atlas/bath', accept='partner_atlas/accept',
             local='local/{attempt}/{proposal,bath,accept}')
REQUIRED_COMMANDS = [
    ['cargo', 'test', '--offline', '--locked', '--release', '--test',
     'partner_atlas', '--', '--test-threads=1'],
    ['cargo', 'test', '--offline', '--locked', '--release', '--test',
     'flexible_surrogate_chain', '--', '--test-threads=1'],
    ['cargo', 'test', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark', '--', '--test-threads=1'],
    ['cargo', 'build', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark'],
]
SCOPE = ('Context0 conditional target at rd1.4,z.0275: labels 27/132 mobile, 262 spectators fixed. '
         'Four streams per start and arm; shared local and partner-atlas roles make comparisons paired. '
         'Local8 is the primary control; completed m4x8 and rigid24 are contextual only. '
         'No assembly, physical-kinetics or original-condition thermodynamic conclusion; '
         'no control resampling or old geometry queries.')


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def policy():
    return dict(schema='partner-atlas-policy-v1', proposal_scales=dict(source='local'))


def jobs():
    result = []
    for initialization in old.INITIALIZATIONS:
        for stream in range(4):
            for arm in ARMS:
                key = f'{old.MASTER}|partner-atlas|0|{initialization}|{stream}|{arm}'
                result.append(dict(id=len(result), context_index=0, initialization=initialization,
                    stream=stream, arm=arm,
                    seed_family=dict(context_index=0, initialization=initialization, stream=stream),
                    queue_key=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')))
    return result


def allocation():
    value = rigid.allocation()
    value.update(schema='partner-atlas-allocation-v1', arms=ARMS[:], chains=32,
        extra_dimer_per_block={a: 1 for a in ARMS}, inner_steps_per_collective=dict(zip(ARMS, [1, 1, 8, 8])),
        guidance_strength=dict(zip(ARMS, [1., 1., 1., 0.])), reused_control_chains=40,
        reused_rigid_chains=24, reused_local_chains=8, reused_m4_chains=8,
        combined_analysis_chains=72, primary_control_arm='local', contextual_control_arms=['m4', *rigid.ARMS],
        local_attempts=589824, dimer_attempts=147456, maximum_inner_steps=663552,
        retained_block_observations=147456, initial_observations=32, retained_initial_observations=147488,
        production_observations=131072, maximum_journal_rows=1032224,
        maximum_new_pair_classifications=77431200,
        per_chain_limits=LIMITS.copy())
    return value


def native_plan(authority):
    return dict(schema='partner-atlas-native-observer-plan-v1', dispatched=False,
        authority=authority,
        after='All 32 chains and arithmetic/contact observer complete and authenticated',
        new_chains=32, new_endpoints=147488, queries_per_retained_endpoint=1,
        target='Instantaneous native labels of the mobile pair; canonical fixed native definition',
        cached_controls='Reuse completed native-observer caches only when full identities and definitions match; no replay here.',
        observables=['internal native occupancy', 'per-motif occupancy', 'motif-presence ESS/CPU',
                     'nonempty motif returns and completed changes', 'occupied frames and return counts'],
        denominators='All retained endpoints including rejections; production blocks 513..4608; block512 transition baseline',
        timing='Full sampler CPU including warmup/rejected work; observer CPU separately and setup-inclusive cost reported',
        comparison='Four streams per start/arm separately, matched local8 primary; no pooling starts or filtering native labels',
        constant_series='Undefined ESS remains null',
        scope='Conditional native-registry diagnostic, not equilibrium or assembly stability')


inherited = flexible.inherited


def validate_native_authority(inputs, root, summary_sha256, original):
    """Reuse completed classifier identity; never load a classifier or query poses."""
    root = Path(root).resolve()
    require(isinstance(summary_sha256, str) and re.fullmatch('[0-9a-f]{64}', summary_sha256),
            'Explicit completed native summary SHA required')
    report, lifecycle = native_completed.authenticate(root)
    summary = inputs.bind(root/'analysis/summary.json', summary_sha256)
    for path, digest in lifecycle.items(): inputs.bind(path, digest)
    plans = []
    for path, digest in report['input_sha256'].items():
        if path.endswith('/audit-plan.json'):
            ref = inputs.bind(path, digest); value = read(ref['path'])
            if value.get('schema') == 'native-pair-initial-audit-plan-v1': plans.append((ref, value))
    require(len(plans) == 1, 'Missing unique native-definition authority')
    plan_ref, plan = plans[0]
    refs = {k:plan[k] for k in ('definition', 'compiled_native', 'witness', 'shape', 'source_frame')}
    for ref in refs.values():
        require(report['input_sha256'].get(ref['path']) == ref['sha256'], 'Native input not execution-bound')
        inputs.bind(ref['path'], ref['sha256'])
    require(refs['shape'] == original['shape'] and refs['source_frame'] == original['source_frame'],
            'Native authority and benchmark shape/source differ')
    definition, compiled, witness = (read(refs[k]['path']) for k in ('definition','compiled_native','witness'))
    classifier = report['classifier']
    require(classifier['definition'] == refs['definition']['path']
            and classifier['definition_sha256'] == refs['definition']['sha256']
            and classifier['input_sha256'] == definition['input_sha256']
            and classifier['criteria'] == definition['criteria']
            and classifier['runtime_sha256'] == definition['input_sha256']['source/native_contact_regions.py']
            and definition['shape_sha256'] == refs['shape']['sha256']
            and compiled['source_definition_sha256'] == refs['definition']['sha256']
            and compiled['source_input_sha256'] == definition['input_sha256']
            and compiled['criteria'] == definition['criteria']
            and witness['native_definition_sha256'] == refs['definition']['sha256']
            and witness['shape_sha256'] == refs['shape']['sha256']
            and witness['compiled_native_sha256'] == refs['compiled_native']['sha256'], 'Classifier identity differs')
    match = witness['native_shape_compatibility']
    require(match['compatible'] is True and match['hard_valid_implication_within_tolerance'] is True
            and match['native_atoms'] == match['physical_atoms'] == match['matched_atoms'] == 4004
            and match['physical_index_by_native_atom'] == list(range(4004))
            and match['matched_max_center_error_a'] == match['matched_max_radius_error_a'] == match['pair_overlap_slack_bound_a'] == 0.,
            'Native atom-identity witness differs')
    require(len(definition['input_sha256']) == 15, 'Incomplete native classifier source closure')
    for name, digest in definition['input_sha256'].items():
        part = Path(name); require(not part.is_absolute() and '..' not in part.parts, 'Unsafe native input path')
        path = Path(refs['definition']['path']).parent/'inputs'/part
        require(report['input_sha256'].get(str(path)) == digest, 'Unbound native classifier dependency')
        inputs.bind(path, digest)
    return dict(root=str(root), summary=summary, original_native_plan=plan_ref, refs=refs,
        classifier=classifier, completed_lifecycle_sha256=lifecycle, no_geometry_queries=True,
        no_trajectory_reads=True, definition_frozen=True)


def validate_build(inputs, path, source_bundle=None):
    ref = inputs.bind(path); receipt = read(ref['path']); tested = receipt.get('source_after')
    require(receipt.get('schema') == 'partner-atlas-benchmark-validation-v1'
            and receipt.get('complete') is True and receipt.get('passed') is True and receipt.get('error') is None
            and isinstance(tested, dict) and tested and receipt.get('source_before') == tested
            and isinstance(receipt.get('production_before'), str)
            and re.fullmatch('[0-9a-f]{64}', receipt['production_before'])
            and receipt['production_before'] == receipt.get('production_after'), 'Incomplete or changed build validation')
    runs = receipt.get('runs', [])
    require(all(command in [r.get('argv') for r in runs] for command in REQUIRED_COMMANDS)
            and all(r.get('returncode') == 0 and r.get('child_drained') is True for r in runs),
            'Required isolated validations were not completed and drained')
    for run in runs:
        inputs.bind(run['log']['path'], run['log']['sha256'])
    executable = inputs.bind(receipt['executable']['path'], receipt['executable']['sha256'])
    executable_bytes = Path(executable['path']).read_bytes(); tested_sources = {}
    for absolute, digest in tested.items():
        source = Path(absolute)
        require(source.is_absolute() and source.resolve() == source and source.is_relative_to(ROOT),
                'Validation source outside project or noncanonical')
        name = str(source.relative_to(ROOT))
        require(name not in tested_sources and sha(source) == digest, 'Tested Rust source changed: '+name)
        tested_sources[name] = digest
    rust = {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.rs')}
    require(rust == {n for n in tested_sources if n.startswith('src/') and n.endswith('.rs')}
            and {'Cargo.toml', 'Cargo.lock', 'build.rs', old.EXAMPLE, *RUST_TESTS, 'src/flexible_surrogate_chain.rs'} <= set(tested_sources),
            'Validation source inventory omits required code/tests')
    bundle_names = rust | {'Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md'}
    candidates = [Path(source_bundle)] if source_bundle else sorted(
        (Path(executable['path']).parent.parent/'build').glob('tetramer-mc-*/out/source-bundle.json'))
    matches = []
    for candidate in candidates:
        bundle = read(candidate)
        if bundle.get('schema') != 1 or set(bundle.get('files', {})) != bundle_names:
            continue
        if all(hashlib.sha256(v['text'].encode()).hexdigest() == v['sha256']
               and v['sha256'] == sha(ROOT/n)
               and (n not in tested_sources or tested_sources[n] == v['sha256'])
               for n, v in bundle['files'].items()) and candidate.read_bytes() in executable_bytes:
            matches.append(candidate)
    require(matches and len({sha(p) for p in matches}) == 1, 'No unique tested source bundle embedded in executable')
    bundle_ref = inputs.bind(matches[0]); bundle = read(bundle_ref['path']); example = record(ROOT/old.EXAMPLE)
    require(example['sha256'] == tested_sources[old.EXAMPLE]
            and Path(example['path']).read_bytes() in executable_bytes, 'Example differs from embedded executable source')
    witness = dict(schema='partner-atlas-example-build-witness-v1', complete=True, passed=True,
        new_arms=ARMS[:], validation=ref, executable=executable, source_bundle=bundle_ref,
        example_source=example, tested_sources=tested_sources,
        production_executable_sha256=receipt['production_after'], production_executable_unchanged=True,
        bundle_embedded_in_executable=True, example_embedded_in_executable=True)
    return ref, witness, {n: v['sha256'] for n, v in bundle['files'].items()}


def validate_reference(inputs, validation, validation_sha256, audit, audit_sha256,
                       audit_validation, audit_validation_sha256, tested_sources, bundle_sha256):
    """Bind externally pinned completed reference receipts, without rescanning journals."""
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                for h in (validation_sha256, audit_sha256, audit_validation_sha256)), 'Explicit reference receipt SHA256s required')
    vr, ar = inputs.bind(validation, validation_sha256), inputs.bind(audit, audit_sha256)
    avr = inputs.bind(audit_validation, audit_validation_sha256); av = read(avr['path'])
    v, a = read(vr['path']), read(ar['path'])
    require(v.get('schema') == 'partner-atlas-reference-validation-v1'
            and v.get('complete') is True and v.get('passed') is True and v.get('returncode') == 0
            and v.get('child_started') is True and v.get('child_drained') is True and v.get('error') is None
            and v.get('source_before') == v.get('source_after') and isinstance(v.get('source_before'), dict)
            and isinstance(v.get('production_before'), str) and re.fullmatch('[0-9a-f]{64}', v['production_before'])
            and v.get('production_before') == v.get('production_after')
            and all(v.get(k) is True for k in ('numerical_allocation_complete', 'scientific_receipt_passed', 'receipt_bindings_valid')),
            'Independent reference execution incomplete or changed')
    require(av.get('schema') == 'partner-atlas-reference-audit-execution-v1'
            and all(av.get(k) is True for k in ('complete', 'passed', 'child_started', 'child_drained'))
            and av.get('returncode') == 0 and av.get('error') is None
            and isinstance(av.get('source_before'), dict) and av['source_before'] == av.get('source_after')
            and av.get('audit_sha256') == ar['sha256'] and av.get('new_physical_samples') == 0,
            'Reference audit execution incomplete, undrained or changed')
    require(a.get('schema') == 'partner-atlas-reference-journal-audit-v1'
            and a.get('complete') is True and a.get('passed') is True
            and a.get('reference_statistical_checks_passed') is True
            and a.get('validation_sha256') == vr['sha256']
            and a.get('inventory') == {k: 8192 for k in ('direct', 'm1_guided', 'm8_guided', 'flat8')}
            and a.get('candidate_attempts') == 147456 and a.get('new_physical_samples') == 0,
            'Independent reference journal audit incomplete or mismatched')
    root = Path(vr['path']).parent/'reference'
    expected = {n: v['outputs']['reference/'+n] for n in
                ('protocol.json', 'kernel-attempts.jsonl', 'summary.json', 'receipt.json')}
    require(a['input_sha256'].get(vr['path']) == vr['sha256']
            and all(a['input_sha256'].get(str(root/n)) == h for n,h in expected.items()),
            'Reference audit output bindings differ')
    refs = {n: inputs.bind(root/n, expected[n]) for n in ('protocol.json', 'summary.json', 'receipt.json')}
    p, s, r = [read(refs[n]['path']) for n in ('protocol.json', 'summary.json', 'receipt.json')]
    require(r.get('schema') == 'partner-atlas-cached-reference-receipt-v1'
            and all(r.get(k) is True for k in ('complete', 'passed', 'numerical_allocation_complete', 'statistical_checks_passed'))
            and r.get('error') is None and r.get('retries') == r.get('replacement_draws') == r.get('new_source_draws') == 0
            and r['output_sha256'] == {n: h for n, h in expected.items() if n != 'receipt.json'}
            and s.get('schema') == 'partner-atlas-cached-reference-summary-v1'
            and s.get('complete') is True and s.get('passed') is True and s.get('statistical_checks_passed') is True
            and s.get('completed_sources') == 8192 and len(s.get('checks', [])) == 292
            and all(c.get('passed') is True for c in s['checks']),
            'Independent reference scientific checks did not pass')
    require(p.get('schema') == 'partner-atlas-cached-reference-protocol-v1'
            and p['streams'] == 4 and p['sources_per_stream'] == 2048
            and p['arms'] == [[1, 1., True, 'direct'], [1, 1., False, 'm1_guided'],
                              [8, 1., False, 'm8_guided'], [8, 0., False, 'flat8']]
            and p.get('total_outer_calls') == 32768 and p.get('total_candidate_attempts') == 147456
            and p.get('new_source_draws') == 0,
            'Independent reference allocation differs')
    required = {n for n in tested_sources if n.startswith('src/') or n in ('Cargo.toml', 'Cargo.lock', 'build.rs', RUST_TESTS[-1])}
    require(required and all(v['source_before'].get(n) == tested_sources[n] for n in required)
            and p['reference_source_sha256'] == tested_sources[RUST_TESTS[-1]]
            and p['compiled_library_bundle_sha256'] == bundle_sha256,
            'Independent reference runtime source differs from tested build')
    return dict(validation=vr, journal_audit=ar, audit_validation=avr, outputs=refs,
        inherited_journal_sha256={'kernel-attempts.jsonl': expected['kernel-attempts.jsonl']},
        source_bound_to_build=True, journal_rescan=False, reference_rerun=False,
        profile_scope='Completed release reference and benchmark build share the full library source bundle; '
                      'the extended example is separately tested. No physical reference rerun.')


def source_paths():
    from analyze_partner_atlas_benchmark import SOURCE_FILES
    return {str(p.relative_to(ROOT)): p for p in old.source_dependencies(
        [ROOT/n for n in [SELF, TEST, DRIVER]+SOURCE_FILES])}


validate_tests = prior.validate_tests
recheck_sources = prior.recheck_sources


def configuration(original, root, authority, controls, compiled):
    config = copy.deepcopy(original)
    config.update(scientific_allocation=record(root/'common/scientific-allocation.json'),
        protocol=record(root/'protocol.json'), jobs=jobs(), allocation=allocation(),
        compiled_source_sha256=compiled, output=str(root/'chains'), inherited_campaign=authority,
        control_analysis=controls, partner_atlas_policy=policy())
    config['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    return config


def verify(base):
    base = Path(base).resolve(); frozen = read(base/'freeze.json')
    require(frozen['schema'] == 'partner-atlas-freeze-v1' and frozen['complete'] is True
            and frozen['scientific_execution_started'] is False, 'Invalid partner-atlas freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base) and sha(path) == digest, 'Frozen artifact changed: '+name)
    for path, digest in frozen['input_sha256'].items():
        require(sha(path) == digest, 'Inherited input changed: '+path)
    config = read(base/'config.json'); expected = read(old.checked_file(config['inherited_campaign']['config']))
    for key in ('scientific_allocation', 'protocol', 'jobs', 'allocation', 'compiled_source_sha256',
                'output', 'inherited_campaign', 'control_analysis', 'partner_atlas_policy'):
        expected[key] = config[key]
    expected['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    require(config == expected and config['jobs'] == jobs() and config['allocation'] == allocation()
            and config['partner_atlas_policy'] == policy() and config['output'] == str(base/'chains'),
            'Undeclared matched-control change')
    protocol = read(old.checked_file(config['protocol']))
    require(protocol['roles'] == ROLES and protocol['partner_atlas_policy'] == policy()
            and protocol['allocation'] == allocation() and protocol['assembly_gate_open'] is False
            and protocol['control_analysis'] == config['control_analysis']
            and read(old.checked_file(protocol['native_observer_plan'])) == native_plan(protocol['native_authority']),
            'Changed partner-atlas policy, observer plan or control authority')
    entries = config['control_analysis']['observations']
    require(len(entries) == 40 and {identity(e['job']) for e in entries}
            == {(0, a, i, s) for a in CONTROL_ARMS for i in old.INITIALIZATIONS for s in range(4)},
            'Changed complete cached-control inventory')
    return config


def prepare(base, rigid_root, rigid_analysis, build_validation, validation,
            reference_validation, reference_validation_sha256, reference_audit, reference_audit_sha256,
            reference_audit_validation, reference_audit_validation_sha256,
            native_authority_root, native_summary_sha256,
            source_bundle=None):
    base = Path(base).resolve(); require(not base.exists(), 'Fresh campaign required; no overwrite/retry')
    inputs = Inputs()
    original, authority, controls, prepared = inherited(inputs, rigid_root, rigid_analysis)
    native = validate_native_authority(inputs, native_authority_root, native_summary_sha256, original)
    build_ref, witness, compiled = validate_build(inputs, build_validation, source_bundle)
    reference = validate_reference(inputs, reference_validation, reference_validation_sha256,
        reference_audit, reference_audit_sha256, reference_audit_validation, reference_audit_validation_sha256,
        witness['tested_sources'], witness['source_bundle']['sha256'])
    sources = source_paths(); validation_ref, admitted = validate_tests(inputs, validation, sources)
    from analyze_partner_atlas_benchmark import analysis_plan
    analysis = analysis_plan()
    require(analysis['schema'] == 'partner-atlas-analysis-plan-v1', 'Wrong observer analysis plan')
    for value in original.values():
        if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
            inputs.bind(value['path'], value['sha256'])
    inputs.recheck(); base.mkdir(parents=True)
    try:
        (base/'common').mkdir(); source_hashes = dict(admitted)
        for name, path in sources.items():
            target = base/'common/source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == admitted[name], 'Tested source changed during copy: '+name)
        for name, item in read(witness['source_bundle']['path'])['files'].items():
            target = base/'common/source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x', encoding='utf-8', newline='') as handle:
                handle.write(item['text'])
            require(sha(target) == item['sha256'], 'Compiled archive differs: '+name)
            source_hashes[name] = item['sha256']
        for name in [old.EXAMPLE]+RUST_TESTS:
            target = base/'common/source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
            require(sha(target) == witness['tested_sources'][name], 'Tested Rust source changed during copy: '+name)
            source_hashes[name] = witness['tested_sources'][name]
        for key, name in (('executable', 'evolving_dimer_benchmark'), ('source_bundle', 'source-bundle.json')):
            shutil.copyfile(witness[key]['path'], base/'common'/name)
            require(sha(base/'common'/name) == witness[key]['sha256'], 'Compiled archive differs')
        executable = base/'common/evolving_dimer_benchmark'; executable.chmod(0o755)
        write(base/'common/build-witness.json', witness)
        write(base/'common/scientific-allocation.json', allocation()); write(base/'analysis-plan.json', analysis)
        write(base/'native-observer-plan.json', native_plan(native))
        protocol = dict(schema='partner-atlas-protocol-v1', allocation=allocation(),
            partner_atlas_policy=policy(), roles=ROLES, inherited_campaign=authority, control_analysis=controls,
            build_witness=record(base/'common/build-witness.json'), build_validation=build_ref,
            validation=validation_ref, independent_reference=reference, analysis_plan=record(base/'analysis-plan.json'),
            native_observer_plan=record(base/'native-observer-plan.json'), native_authority=native,
            source_files=source_hashes, scope=SCOPE, all_attempts_retained=True,
            analysis_after_complete_inventory=True, retry=False, replacement=False, assembly_gate_open=False,
            members=[27, 132], spectators=262,
            surrogate_cloud='Original family body cloud, raw box volume/raw_count, shared by both labels; no refresh.',
            decisions='Four unchanged canonical locals, then one fair-scan partner-atlas slot. '
                      'Direct: one proposal, no score/filter, helper correction at physical gate. '
                      'Guided: fixed horizon with proposal correction in each inner MH; one endpoint S(old)-S(new) correction. '
                      'Flat8: same inner proposal correction with zero score. One fair-order two-singleton bath; all rejections retained.',
            auxiliary_midpoint='Copied midpoint may violate cores or wall; never hard-filtered.',
            controls='Completed local8 primary; rigid24 + m4x8 contextual. Every full identity included once, independent of outcomes; no old geometry.',
            interruption='Begun/outcome journals; fatal caps or partial tails stop, never retry or replace.')
        write(base/'protocol.json', protocol)
        write(base/'config.json', configuration(original, base, authority, controls, compiled))
        write(base/'freeze.json', dict(schema='partner-atlas-freeze-v1', complete=True,
            scientific_execution_started=False, files={str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()},
            input_sha256=dict(inputs.files)))
        write(base/'binding.json', dict(schema='evolving-dimer-binding-v1', config_sha256=sha(base/'config.json'),
            protocol_sha256=sha(base/'protocol.json'), freeze_sha256=sha(base/'freeze.json'),
            example_source_sha256=source_hashes[old.EXAMPLE], compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'),
            executable_sha256=sha(executable), isolated_build_witness=record(base/'common/build-witness.json')))
        write(base/'run-binding.json', dict(schema='evolving-dimer-run-binding-v1', complete=True,
            config_sha256=sha(base/'config.json'), protocol_sha256=sha(base/'protocol.json'),
            prelaunch_binding=record(base/'binding.json'), prepared_manifest=authority['prepared_manifest'],
            prepared_files=prepared, cloud_banks=32, alternative_starts=16, inherited_preparation_authority=authority))
        execution_jobs = [dict(id=f"job-{j['id']:03}", population=f"context0-{j['initialization']}-{j['stream']}",
            phase='producer', argv=[str(executable), '--mode', 'run', '--config', str(base/'config.json'),
            '--binding', str(base/'run-binding.json'), '--job', str(j['id'])],
            terminal=dict(path=str(base/'chains'/f"job-{j['id']:03}"/'terminal.json'), success_contract='complete'),
            **LIMITS) for j in sorted(jobs(), key=lambda j: (j['queue_key'], j['id']))]
        consumed = dict(inputs.files); consumed.update({str(p): sha(p) for p in base.rglob('*') if p.is_file()})
        execution = dict(schema=driver.SCHEMA, root=str(base), maximum_workers=1, threads=1,
            files=consumed, jobs=execution_jobs, executable_resolutions={str(executable): str(executable)},
            preparation_receipt=str(base/'preparation.json'), scope=SCOPE)
        write(base/'execution-plan.json', execution); inputs.recheck(); verify(base); recheck_sources(sources, admitted)
        receipt = dict(schema='partner-atlas-preparation-v1', complete=True, launched=False,
            protocol=record(base/'protocol.json'), execution_plan=record(base/'execution-plan.json'),
            config=record(base/'config.json'), allocation=allocation(), new_physical_draws=0, new_geometry_queries=0,
            historical_journals_scanned=0, reference_rerun=False, source_archived=True, assembly_gate_open=False)
        write(base/'preparation.json', receipt)
        spec = importlib.util.spec_from_file_location('partner_atlas_frozen_driver', base/'common/source'/DRIVER)
        frozen_driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen_driver)
        frozen_driver.verify_plan(base/'execution-plan.json', execution, receipt['execution_plan']['sha256'], fresh=True)
        return receipt
    except BaseException as error:
        write(base/'preparation-failure.json', dict(complete=False, launched=False, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'rigid-root', 'rigid-analysis', 'build-validation', 'validation',
                 'reference-validation', 'reference-audit', 'reference-audit-validation', 'native-authority-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('reference-validation-sha256', 'reference-audit-sha256', 'reference-audit-validation-sha256', 'native-summary-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-bundle', type=Path)
    args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.root, args.rigid_root, args.rigid_analysis,
        args.build_validation, args.validation, args.reference_validation, args.reference_validation_sha256,
        args.reference_audit, args.reference_audit_sha256,
        args.reference_audit_validation, args.reference_audit_validation_sha256,
        args.native_authority_root, args.native_summary_sha256, args.source_bundle), indent=2))
