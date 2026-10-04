#!/usr/bin/env python3
"""Freeze 24 flexible context0 chains; reuse completed controls and preparation.

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
import prepare_two_root_dimer_benchmark as prior
import plot_rigid_surrogate_benchmark as completed
import run_native_class_physical_campaign as driver

ROOT = Path(__file__).resolve().parents[1]
require, read, sha, record, write = prior.require, prior.read, prior.sha, prior.record, prior.write
Inputs = prior.Inputs
SELF = 'tools/prepare_flexible_surrogate_benchmark.py'
TEST = 'tests/test_prepare_flexible_surrogate_benchmark.py'
DRIVER = 'tools/run_native_class_physical_campaign.py'
ARMS = ['flexible_m1', 'flexible_m8', 'flexible_flat8']
CONTROL_ARMS = ['local', 'm4', *rigid.ARMS]
RUST_TESTS = ['tests/flexible_surrogate_chain.rs', 'tests/flexible_surrogate_stationarity.rs']
LIMITS = dict(cpu_limit_seconds=1800, wall_limit_seconds=3600, address_space_limit_bytes=8*2**30)
ROLES = dict(proposal='flexible_surrogate/proposal', inner_accept='flexible_surrogate/inner_accept',
             bath='flexible_surrogate/bath', accept='flexible_surrogate/accept',
             local='local/{attempt}/{proposal,bath,accept}')
REQUIRED_COMMANDS = [
    ['cargo', 'test', '--offline', '--locked', '--release', '--test',
     'flexible_surrogate_chain', '--', '--test-threads=1'],
    ['cargo', 'test', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark', '--', '--test-threads=1'],
    ['cargo', 'build', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark'],
]
SCOPE = ('Context0 conditional target: labels 27/132 mobile, 262 spectators fixed. '
         'Four streams per start and arm; shared local and flexible role seeds do not make arms independent. '
         'Contact-patch reorganization and new pose metrics are not native registry, physical kinetics, '
         'equilibrium certification or assembly. No control resampling or old geometry queries.')


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def policy():
    return dict(schema='flexible-surrogate-policy-v1', proposal_scales=dict(source='local'))


def jobs():
    result = []
    for initialization in old.INITIALIZATIONS:
        for stream in range(4):
            for arm in ARMS:
                key = f'{old.MASTER}|flexible-surrogate|0|{initialization}|{stream}|{arm}'
                result.append(dict(id=len(result), context_index=0, initialization=initialization,
                    stream=stream, arm=arm,
                    seed_family=dict(context_index=0, initialization=initialization, stream=stream),
                    queue_key=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')))
    return result


def allocation():
    value = rigid.allocation()
    value.update(schema='flexible-surrogate-allocation-v1', arms=ARMS[:],
        extra_dimer_per_block={a: 1 for a in ARMS}, inner_steps_per_collective=dict(zip(ARMS, [1, 8, 8])),
        guidance_strength=dict(zip(ARMS, [1., 1., 0.])), reused_control_chains=40,
        reused_rigid_chains=24, reused_local_chains=8, reused_m4_chains=8,
        combined_analysis_chains=64, maximum_new_pair_classifications=58073400,
        per_chain_limits=LIMITS.copy())
    return value


def inherited(inputs, rigid_root, analysis_root):
    """Authenticate completed cached evidence, without opening any trajectory."""
    rigid_root, analysis_root = Path(rigid_root).resolve(), Path(analysis_root).resolve()
    # This cheap lifecycle gate precedes cache hashing or campaign inspection.
    report, lifecycle = completed.authenticate(analysis_root)
    for path, digest in lifecycle.items():
        inputs.bind(path, digest)
    config = rigid.verify(rigid_root)
    original_root = Path(config['inherited_campaign']['config']['path']).parent
    old_analysis = Path(config['control_analysis']['summary']['path']).parent
    original, authority, original_controls, prepared, _ = rigid.inherited(inputs, original_root, old_analysis)
    require(authority == config['inherited_campaign'] and original_controls == config['control_analysis'],
            'Rigid campaign inherited authority differs')
    controls = {k: inputs.bind(analysis_root/v) for k, v in dict(
        analysis='analysis/analysis.json', summary='execution/summary.json',
        manifest='analysis/manifest.json', input_binding='analysis/input-binding.json').items()}
    manifest, ib = [read(controls[k]['path']) for k in ('manifest', 'input_binding')]
    require(manifest['files']['analysis.json'] == controls['analysis']['sha256']
            and manifest['files']['input-binding.json'] == controls['input_binding']['sha256']
            and ib['input_sha256'] == report['input_sha256']
            and ib['source_sha256'] == report['source_sha256'], 'Rigid observer input closure differs')
    campaign = {}
    for name in ('config.json', 'run-binding.json', 'protocol.json', 'freeze.json', 'execution-plan.json',
                 'execution/status.json', 'execution/summary.json'):
        path = str(rigid_root/name)
        require(path in report['input_sha256'], 'Rigid observer did not bind campaign '+name)
        campaign[name] = inputs.bind(path, report['input_sha256'][path])
    rprotocol = read(campaign['protocol.json']['path'])
    require(all(rprotocol['source_files'].get(n) == h for n, h in report['source_sha256'].items()),
            'Completed observer source archive differs')
    for name, digest in report['source_sha256'].items():
        inputs.bind(rigid_root/'common/source'/name, digest)
    status, summary = [read(campaign[k]['path']) for k in ('execution/status.json', 'execution/summary.json')]
    require(all(status[k] == summary[k] for k in ('complete', 'passed', 'failure', 'active',
                'unstarted', 'completed', 'plan_sha256'))
            and status['complete'] is True and status['passed'] is True and status['failure'] is None
            and status['active'] is None and status['unstarted'] == [] and len(status['completed']) == 24
            and status['plan_sha256'] == campaign['execution-plan.json']['sha256']
            and not (rigid_root/'execution/failure.json').exists(), 'Rigid sampler incomplete or undrained')
    controls['authority'] = {'campaign/'+name: ref for name, ref in campaign.items()}
    # The frozen declaration belongs to the sampler campaign, outside the
    # separate observer root. Preserve every authenticated absolute reference.
    controls['authority'].update({'analysis/'+path: dict(path=path, sha256=digest)
                                  for path, digest in lifecycle.items()})
    expected = {(0, a, i, s) for a in CONTROL_ARMS for i in old.INITIALIZATIONS for s in range(4)}
    chains = {identity(c['job']): c for c in report['chains']}
    require(len(chains) == len(report['chains']) == 40 and set(chains) == expected,
            'Completed controls must contain each full identity exactly once')
    require(config['jobs'] == rigid.jobs() and config['allocation'] == rigid.allocation(),
            'Rigid allocation changed')
    controls['observations'] = []
    for key in sorted(chains):
        chain = chains[key]; job = chain['job']; ident = str(job['id'])
        cached = job['arm'] in ('local', 'm4')
        require(chain['reused_control'] is cached, 'Cached control origin differs')
        if cached:
            obs = original_controls['observations'][ident]
            terminal = original_controls['terminals'][ident]
            origin, origin_manifest = original_controls['analysis'], original_controls['manifest']
        else:
            require(job == config['jobs'][job['id']], 'Rigid control job differs')
            name = f"job-{job['id']:03}-observations.jsonl"
            obs = inputs.bind(analysis_root/'analysis'/name, manifest['files'][name])
            path = str(Path(config['output'])/f"job-{job['id']:03}"/'terminal.json')
            require(path in report['input_sha256'], 'Unbound rigid terminal')
            terminal = inputs.bind(path, report['input_sha256'][path])
            origin, origin_manifest = controls['analysis'], controls['manifest']
        for ref in (terminal, origin, origin_manifest):
            inputs.bind(ref['path'], ref['sha256'])
        if cached:
            require(all(report['input_sha256'].get(ref['path']) == ref['sha256']
                        for ref in (obs, terminal, origin, origin_manifest)),
                    'Inherited cache absent from completed umbrella binding')
        t = read(terminal['path']); om = read(origin_manifest['path']); origin_report = read(origin['path'])
        origin_inputs = origin_report['input_files' if cached else 'input_sha256']
        require(t['complete'] is True and t['conditional_target'] is True and t['job'] == job
                and t['blocks'] == 4608 and t['trajectory'] == chain['trajectory']
                and t['cpu_seconds'] == chain['metrics']['full_sampler_cpu_seconds']
                and origin_inputs.get(terminal['path']) == terminal['sha256']
                and origin_inputs.get(t['trajectory']['path']) == t['trajectory']['sha256']
                and not (Path(terminal['path']).parent/'failure.json').exists(),
                'Cached terminal/replay identity differs')
        require(om['complete'] is True and om['files']['analysis.json'] == origin['sha256']
                and om['files'][Path(obs['path']).name] == obs['sha256'], 'Unbound observation cache')
        controls['observations'].append(dict(job=job, observation=obs, manifest=origin_manifest,
                                             terminal=terminal, analysis=origin))
    return original, authority, controls, prepared


def validate_build(inputs, path, source_bundle=None):
    ref = inputs.bind(path); receipt = read(ref['path']); tested = receipt.get('source_after')
    require(receipt.get('schema') == 'flexible-surrogate-benchmark-validation-v1'
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
            and {'Cargo.toml', 'Cargo.lock', old.EXAMPLE, *RUST_TESTS, 'src/flexible_surrogate_chain.rs'} <= set(tested_sources),
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
    witness = dict(schema='flexible-surrogate-example-build-witness-v1', complete=True, passed=True,
        new_arms=ARMS[:], validation=ref, executable=executable, source_bundle=bundle_ref,
        example_source=example, tested_sources=tested_sources,
        production_executable_sha256=receipt['production_after'], production_executable_unchanged=True,
        bundle_embedded_in_executable=True, example_embedded_in_executable=True)
    return ref, witness, {n: v['sha256'] for n, v in bundle['files'].items()}


def validate_reference(inputs, validation, validation_sha256, audit, audit_sha256, tested_sources):
    """Bind externally pinned completed reference receipts, without rescanning journals."""
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                for h in (validation_sha256, audit_sha256)), 'Explicit reference receipt SHA256s required')
    vr, ar = inputs.bind(validation, validation_sha256), inputs.bind(audit, audit_sha256)
    v, a = read(vr['path']), read(ar['path'])
    require(v.get('schema') == 'flexible-surrogate-reference-validation-v1'
            and v.get('complete') is True and v.get('passed') is True and v.get('returncode') == 0
            and v.get('child_started') is True and v.get('child_drained') is True and v.get('error') is None
            and v.get('source_before') == v.get('source_after') and isinstance(v.get('source_before'), dict)
            and isinstance(v.get('production_before'), str) and re.fullmatch('[0-9a-f]{64}', v['production_before'])
            and v.get('production_before') == v.get('production_after'), 'Independent reference execution incomplete or changed')
    require(a.get('schema') == 'flexible-surrogate-reference-journal-audit-v1'
            and a.get('complete') is True and a.get('passed') is True
            and a.get('reference_scientific_checks_passed') is True
            and a.get('validation_sha256') == vr['sha256']
            and a.get('inventory') == {k: 8192 for k in ('m1_guided', 'm8_guided', 'm8_zero')},
            'Independent reference journal audit incomplete or mismatched')
    root = Path(vr['path']).parent/'reference'
    expected = {n: v['outputs']['reference/'+n] for n in
                ('protocol.json', 'source-attempts.jsonl', 'kernel-attempts.jsonl', 'summary.json', 'receipt.json')}
    require(a['input_sha256'] == expected, 'Reference audit output bindings differ')
    refs = {n: inputs.bind(root/n, expected[n]) for n in ('protocol.json', 'summary.json', 'receipt.json')}
    p, s, r = [read(refs[n]['path']) for n in ('protocol.json', 'summary.json', 'receipt.json')]
    require(r.get('schema') == 'flexible-surrogate-independent-reference-receipt-v1'
            and all(r.get(k) is True for k in ('complete', 'passed', 'numerical_allocation_complete', 'scientific_checks_passed'))
            and r.get('error') is None and r.get('retries') == r.get('replacement_draws') == 0
            and r['output_sha256'] == {n: h for n, h in expected.items() if n != 'receipt.json'}
            and s.get('complete') is True and s['source_counts'] == r['source_counts'] == a['source_counts'],
            'Independent reference scientific checks did not pass')
    require(p.get('schema') == 'flexible-surrogate-independent-reference-protocol-v1'
            and p['streams'] == 4 and p['sources_per_stream'] == 2048
            and p['arms'] == [[1, 1., 'm1_guided'], [8, 1., 'm8_guided'], [8, 0., 'm8_zero']],
            'Independent reference allocation differs')
    required = {n for n in tested_sources if n.startswith('src/') or n in ('Cargo.toml', 'Cargo.lock', RUST_TESTS[1])}
    require(required and all(v['source_before'].get(n) == tested_sources[n] for n in required)
            and p['reference_source_sha256'] == tested_sources[RUST_TESTS[1]],
            'Independent reference runtime source differs from tested build')
    return dict(validation=vr, journal_audit=ar, outputs=refs,
        inherited_journal_sha256={n: expected[n] for n in ('source-attempts.jsonl', 'kernel-attempts.jsonl')},
        source_bound_to_build=True, journal_rescan=False, reference_rerun=False,
        profile_scope='The independent reference used debug; the matched release source is validated '
                      'by focused kernel and actual example tests, without rerunning the physical reference.')


def source_paths():
    from analyze_flexible_surrogate_benchmark import SOURCE_FILES
    return {str(p.relative_to(ROOT)): p for p in old.source_dependencies(
        [ROOT/n for n in [SELF, TEST, DRIVER]+SOURCE_FILES])}


validate_tests = prior.validate_tests
recheck_sources = prior.recheck_sources


def configuration(original, root, authority, controls, compiled):
    config = copy.deepcopy(original)
    config.update(scientific_allocation=record(root/'common/scientific-allocation.json'),
        protocol=record(root/'protocol.json'), jobs=jobs(), allocation=allocation(),
        compiled_source_sha256=compiled, output=str(root/'chains'), inherited_campaign=authority,
        control_analysis=controls, flexible_surrogate_policy=policy())
    config['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    return config


def verify(base):
    base = Path(base).resolve(); frozen = read(base/'freeze.json')
    require(frozen['schema'] == 'flexible-surrogate-freeze-v1' and frozen['complete'] is True
            and frozen['scientific_execution_started'] is False, 'Invalid flexible surrogate freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base) and sha(path) == digest, 'Frozen artifact changed: '+name)
    for path, digest in frozen['input_sha256'].items():
        require(sha(path) == digest, 'Inherited input changed: '+path)
    config = read(base/'config.json'); expected = read(old.checked_file(config['inherited_campaign']['config']))
    for key in ('scientific_allocation', 'protocol', 'jobs', 'allocation', 'compiled_source_sha256',
                'output', 'inherited_campaign', 'control_analysis', 'flexible_surrogate_policy'):
        expected[key] = config[key]
    expected['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    require(config == expected and config['jobs'] == jobs() and config['allocation'] == allocation()
            and config['flexible_surrogate_policy'] == policy() and config['output'] == str(base/'chains'),
            'Undeclared matched-control change')
    protocol = read(old.checked_file(config['protocol']))
    require(protocol['roles'] == ROLES and protocol['flexible_surrogate_policy'] == policy()
            and protocol['allocation'] == allocation() and protocol['assembly_gate_open'] is False
            and protocol['control_analysis'] == config['control_analysis'], 'Changed flexible policy or control authority')
    entries = config['control_analysis']['observations']
    require(len(entries) == 40 and {identity(e['job']) for e in entries}
            == {(0, a, i, s) for a in CONTROL_ARMS for i in old.INITIALIZATIONS for s in range(4)},
            'Changed complete cached-control inventory')
    return config


def prepare(base, rigid_root, rigid_analysis, build_validation, validation,
            reference_validation, reference_validation_sha256, reference_audit, reference_audit_sha256,
            source_bundle=None):
    base = Path(base).resolve(); require(not base.exists(), 'Fresh campaign required; no overwrite/retry')
    inputs = Inputs()
    original, authority, controls, prepared = inherited(inputs, rigid_root, rigid_analysis)
    build_ref, witness, compiled = validate_build(inputs, build_validation, source_bundle)
    reference = validate_reference(inputs, reference_validation, reference_validation_sha256,
                                   reference_audit, reference_audit_sha256, witness['tested_sources'])
    sources = source_paths(); validation_ref, admitted = validate_tests(inputs, validation, sources)
    from analyze_flexible_surrogate_benchmark import analysis_plan
    analysis = analysis_plan()
    require(analysis['schema'] == 'flexible-surrogate-analysis-plan-v1', 'Wrong observer analysis plan')
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
        protocol = dict(schema='flexible-surrogate-protocol-v1', allocation=allocation(),
            flexible_surrogate_policy=policy(), roles=ROLES, inherited_campaign=authority, control_analysis=controls,
            build_witness=record(base/'common/build-witness.json'), build_validation=build_ref,
            validation=validation_ref, independent_reference=reference, analysis_plan=record(base/'analysis-plan.json'),
            source_files=source_hashes, scope=SCOPE, all_attempts_retained=True,
            analysis_after_complete_inventory=True, retry=False, replacement=False, assembly_gate_open=False,
            members=[27, 132], spectators=262,
            surrogate_cloud='Original family body cloud, raw box volume/raw_count, shared by both labels; no refresh.',
            decisions='Four unchanged canonical local attempts, then one fixed-horizon fair-scan flexible attempt. '
                      'Hard/MH rejections count. One fair-order two-singleton bath and one S(old)-S(new) correction.',
            auxiliary_midpoint='Copied midpoint may violate cores or wall; never hard-filtered.',
            controls='Completed rigid24 + local8 + secondary m4x8, each full identity included once; no old geometry.',
            interruption='Begun/outcome journals; fatal caps or partial tails stop, never retry or replace.')
        write(base/'protocol.json', protocol)
        write(base/'config.json', configuration(original, base, authority, controls, compiled))
        write(base/'freeze.json', dict(schema='flexible-surrogate-freeze-v1', complete=True,
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
        receipt = dict(schema='flexible-surrogate-preparation-v1', complete=True, launched=False,
            protocol=record(base/'protocol.json'), execution_plan=record(base/'execution-plan.json'),
            config=record(base/'config.json'), allocation=allocation(), new_physical_draws=0, new_geometry_queries=0,
            historical_journals_scanned=0, reference_rerun=False, source_archived=True, assembly_gate_open=False)
        write(base/'preparation.json', receipt)
        spec = importlib.util.spec_from_file_location('flexible_surrogate_frozen_driver', base/'common/source'/DRIVER)
        frozen_driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen_driver)
        frozen_driver.verify_plan(base/'execution-plan.json', execution, receipt['execution_plan']['sha256'], fresh=True)
        return receipt
    except BaseException as error:
        write(base/'preparation-failure.json', dict(complete=False, launched=False, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'rigid-root', 'rigid-analysis', 'build-validation', 'validation',
                 'reference-validation', 'reference-audit'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('reference-validation-sha256', 'reference-audit-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-bundle', type=Path)
    args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.root, args.rigid_root, args.rigid_analysis,
        args.build_validation, args.validation, args.reference_validation, args.reference_validation_sha256,
        args.reference_audit, args.reference_audit_sha256, args.source_bundle), indent=2))
