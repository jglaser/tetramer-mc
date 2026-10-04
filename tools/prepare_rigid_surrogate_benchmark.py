#!/usr/bin/env python3
"""Freeze 24 matched conditional rigid-surrogate chains without launching them.

Reuse the original context0 preparation and sixteen completed local/m4 caches.
Only metadata and hashes are read: no new starts, clouds, physical draws, or
geometry queries. The copied executable and observer closure are bound before
the single-worker producer plan is published; every allocation is exclusive.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil

import prepare_evolving_dimer_benchmark as old
import prepare_two_root_dimer_benchmark as prior
import run_native_class_physical_campaign as driver

ROOT = Path(__file__).resolve().parents[1]
require, read, sha, record, write = prior.require, prior.read, prior.sha, prior.record, prior.write
Inputs = prior.Inputs
ARMS = ['rigid_surrogate_1', 'rigid_surrogate_8', 'rigid_surrogate_flat8']
SELF = 'tools/prepare_rigid_surrogate_benchmark.py'
TEST = 'tests/test_prepare_rigid_surrogate_benchmark.py'
DRIVER = 'tools/run_native_class_physical_campaign.py'
RUST_TESTS = ['tests/rigid_surrogate_chain.rs', 'tests/rigid_surrogate_stationarity.rs']
LIMITS = dict(cpu_limit_seconds=1800, wall_limit_seconds=3600,
              address_space_limit_bytes=8*2**30)
ROLES = dict(proposal='rigid_surrogate/proposal', inner_accept='rigid_surrogate/inner_accept',
             bath='rigid_surrogate/bath', accept='rigid_surrogate/accept',
             local='local/{attempt}/{proposal,bath,accept}')
SCOPE = ('Context0 conditional two-mobile contact sampling at rd1.4,z.0275. '
         'Four independent streams per start; all three surrogate arms share local and collective RNG roles. '
         'No finite-system assembly or native-registry conclusion; the assembly gate remains closed.')
REQUIRED_COMMANDS = [
    ['cargo', 'test', '--offline', '--locked', '--release', '--test',
     'rigid_surrogate_chain', '--', '--test-threads=1'],
    ['cargo', 'test', '--offline', '--locked', '--release', '--test',
     'rigid_surrogate_stationarity', '--', '--test-threads=1', '--nocapture'],
    ['cargo', 'test', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark', '--', '--test-threads=1'],
    ['cargo', 'build', '--offline', '--locked', '--release', '--example',
     'evolving_dimer_benchmark'],
]


def policy():
    return dict(schema='rigid-surrogate-policy-v1', proposal_scales=dict(source='local'))


def jobs():
    result = []
    for initialization in old.INITIALIZATIONS:
        for stream in range(4):
            for arm in ARMS:
                key = f'{old.MASTER}|rigid-surrogate|0|{initialization}|{stream}|{arm}'
                result.append(dict(id=len(result), context_index=0, initialization=initialization,
                    stream=stream, arm=arm,
                    seed_family=dict(context_index=0, initialization=initialization, stream=stream),
                    queue_key=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')))
    return result


def allocation():
    return dict(schema='rigid-surrogate-allocation-v1', contexts=1, context_indices=[0], arms=ARMS[:],
        initializations=old.INITIALIZATIONS[:], streams=4, chains=24,
        warmup_blocks=512, production_blocks=4096, blocks_per_chain=4608,
        local_attempts_per_block=4, local_member_order=[0, 1, 0, 1],
        extra_dimer_per_block={arm: 1 for arm in ARMS},
        inner_steps_per_collective=dict(zip(ARMS, [1, 8, 8])),
        guidance_strength=dict(zip(ARMS, [1., 1., 0.])),
        local_attempts=442368, dimer_attempts=110592, maximum_inner_steps=626688,
        retained_block_observations=110592, initial_observations=24,
        retained_initial_observations=110616, production_observations=98304,
        journal_rows_per_block=7, maximum_journal_rows=774168,
        raw_points_per_cloud=16384, inherited_cloud_banks_used=8,
        inherited_alternative_starts_used=4, reused_control_chains=16,
        new_cloud_banks=0, new_preparation_attempts=0, new_raw_cloud_points=0,
        master_seed=old.MASTER, replacement=False, retry=False, native_classification=False,
        python_geometry_evaluations=0, physical_draws_during_python_preparation=0,
        maximum_workers=1, threads=1, per_chain_limits=LIMITS,
        assembly_gate_open=False)


def inherited(inputs, original, analysis_root):
    # The existing authenticator checks immutable historical archives rather
    # than comparing historical Rust with the current development workspace.
    config, authority, controls, prepared, _ = prior.inherited(inputs, original, analysis_root)
    require(config['jobs'] == old.jobs() and config['allocation'] == old.allocation(),
            'Original allocation differs')
    require(config['master_seed'] == old.MASTER and config['cloud']['raw_count'] == 16384
            and config['local']['translation_std_A'] == .2
            and config['local']['rotation_std_degrees'] == 1.
            and config['local']['member_order'] == [0, 1, 0, 1]
            and [config['contexts'][0][k] for k in ('root', 'child')] == old.EXPECTED_PAIRS[0],
            'Original matched scales, cloud or context differs')
    require(not any(k in config for k in ('surrogate_policy', 'singleton_policy', 'two_root_policy')),
            'Original config contains a later experimental policy')
    selected = [j for j in config['jobs'] if j['context_index'] == 0 and j['arm'] in ('local', 'm4')]
    mapping = []
    for job in jobs():
        controls_by_arm = {j['arm']: j['id'] for j in selected
            if j['initialization'] == job['initialization'] and j['stream'] == job['stream']}
        require(set(controls_by_arm) == {'local', 'm4'}, 'Incomplete matched control family')
        mapping.append(dict(job_id=job['id'], context_index=0,
            initialization=job['initialization'], stream=job['stream'], controls=controls_by_arm))
    return config, authority, controls, prepared, mapping


def validate_build(inputs, path, source_bundle=None):
    """Authenticate the supplied completed validation; never run Rust again.

    Exact embedded bundle/example bytes tie the archived closure to the tested
    executable. Build.rs and vendor/README.md are authenticated by that bundle;
    every compiled src/Cargo input and both new tests must also match the receipt.
    """
    ref = inputs.bind(path)
    receipt = read(ref['path'])
    tested = receipt.get('source_after')
    require(receipt.get('schema') == 'rigid-surrogate-validation-v1'
            and receipt.get('complete') is True and receipt.get('passed') is True
            and receipt.get('error') is None and isinstance(tested, dict) and tested
            and receipt.get('source_before') == tested
            and receipt.get('production_before') == receipt.get('production_after')
            and isinstance(receipt.get('production_before'), str)
            and len(receipt['production_before']) == 64,
            'Incomplete or changed rigid-surrogate validation')
    runs = receipt.get('runs', [])
    require([r.get('argv') for r in runs] == REQUIRED_COMMANDS
            and all(r.get('returncode') == 0 and r.get('child_drained') is True for r in runs),
            'Required isolated validations were not completed and drained')
    for run in runs:
        inputs.bind(run['log']['path'], run['log']['sha256'])
    executable = inputs.bind(receipt['executable']['path'], receipt['executable']['sha256'])
    executable_bytes = Path(executable['path']).read_bytes()
    source_hashes = {}
    for absolute, digest in tested.items():
        source = Path(absolute)
        require(source.is_absolute() and source.resolve() == source and source.is_relative_to(ROOT),
                'Validation source outside project or noncanonical')
        name = str(source.relative_to(ROOT))
        require(name not in source_hashes and sha(source) == digest, 'Tested Rust source changed: '+name)
        source_hashes[name] = digest
    required = {'Cargo.toml', 'Cargo.lock', old.EXAMPLE, *RUST_TESTS, 'src/rigid_surrogate_chain.rs'}
    require(required <= set(source_hashes), 'Validation source inventory omits required code/tests')
    rust_sources = {n for n in source_hashes if n.startswith('src/') and n.endswith('.rs')}
    require(rust_sources == {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.rs')},
            'Compiled Rust source inventory differs from validation')
    bundle_names = rust_sources | {'Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md'}
    paths = [Path(source_bundle)] if source_bundle else sorted(
        (Path(executable['path']).parent.parent/'build').glob('tetramer-mc-*/out/source-bundle.json'))
    matches = []
    for candidate in paths:
        bundle = read(candidate)
        if bundle.get('schema') != 1 or set(bundle.get('files', {})) != bundle_names:
            continue
        files = bundle['files']
        if not all(hashlib.sha256(v['text'].encode()).hexdigest() == v['sha256']
                   and (n not in source_hashes or source_hashes[n] == v['sha256'])
                   for n, v in files.items()):
            continue
        if candidate.read_bytes() in executable_bytes:
            matches.append(candidate)
    require(matches and len({sha(p) for p in matches}) == 1,
            'No unique tested source bundle embedded in executable')
    bundle_ref = inputs.bind(matches[0])
    bundle = read(bundle_ref['path'])
    example = record(ROOT/old.EXAMPLE)
    require(example['sha256'] == source_hashes[old.EXAMPLE]
            and Path(example['path']).read_bytes() in executable_bytes,
            'Tested example source differs from embedded executable source')
    sources = {name: item['sha256'] for name, item in bundle['files'].items()}
    witness = dict(schema='rigid-surrogate-example-build-witness-v1', complete=True, passed=True,
        new_arms=ARMS[:], validation=ref, executable=executable, source_bundle=bundle_ref,
        example_source=example, tested_sources=source_hashes,
        production_executable_sha256=receipt['production_after'], production_executable_unchanged=True,
        bundle_embedded_in_executable=True, example_embedded_in_executable=True)
    return ref, witness, sources


def source_paths():
    from analyze_rigid_surrogate_benchmark import SOURCE_FILES
    return {str(p.relative_to(ROOT)): p for p in old.source_dependencies(
        [ROOT/n for n in [SELF, TEST, DRIVER]+SOURCE_FILES])}


validate_tests = prior.validate_tests
recheck_sources = prior.recheck_sources


def configuration(original, root, authority, controls, mapping, compiled):
    config = copy.deepcopy(original)
    config.update(scientific_allocation=record(root/'common/scientific-allocation.json'),
        protocol=record(root/'protocol.json'), jobs=jobs(), allocation=allocation(),
        compiled_source_sha256=compiled, output=str(root/'chains'),
        inherited_campaign=authority, control_analysis=controls, control_mapping=mapping,
        surrogate_policy=policy())
    config['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    return config


def verify(base):
    base = Path(base).resolve()
    frozen = read(base/'freeze.json')
    require(frozen['schema'] == 'rigid-surrogate-freeze-v1' and frozen['complete'] is True
            and frozen['scientific_execution_started'] is False, 'Invalid surrogate freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base) and sha(path) == digest, 'Frozen artifact changed: '+name)
    for path, digest in frozen['input_sha256'].items():
        require(sha(path) == digest, 'Inherited input changed: '+path)
    config = read(base/'config.json')
    original = read(old.checked_file(config['inherited_campaign']['config']))
    expected = copy.deepcopy(original)
    for key in ('scientific_allocation', 'protocol', 'jobs', 'allocation', 'compiled_source_sha256',
                'output', 'inherited_campaign', 'control_analysis', 'control_mapping', 'surrogate_policy'):
        expected[key] = config[key]
    expected['limits']['cpu_seconds'] = LIMITS['cpu_limit_seconds']
    require(config == expected and config['jobs'] == jobs() and config['allocation'] == allocation()
            and config['surrogate_policy'] == policy() and config['output'] == str(base/'chains'),
            'Undeclared matched-control change')
    protocol = read(old.checked_file(config['protocol']))
    require(protocol['roles'] == ROLES and protocol['surrogate_policy'] == policy()
            and protocol['allocation'] == allocation() and protocol['assembly_gate_open'] is False,
            'Changed surrogate/RNG/schedule policy')
    return config


def prepare(base, original, control_analysis, build_validation, validation, source_bundle=None):
    base = Path(base).resolve()
    require(not base.exists(), 'Fresh campaign required; no overwrite/retry')
    inputs = Inputs()
    original_config, authority, controls, prepared, mapping = inherited(inputs, original, control_analysis)
    build_ref, witness, compiled = validate_build(inputs, build_validation, source_bundle)
    sources = source_paths()
    validation_ref, admitted = validate_tests(inputs, validation, sources)
    from analyze_rigid_surrogate_benchmark import analysis_plan
    analysis = analysis_plan()
    require(analysis['schema'] == 'rigid-surrogate-analysis-plan-v1', 'Wrong observer analysis plan')
    for value in original_config.values():
        if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
            inputs.bind(value['path'], value['sha256'])
    inputs.recheck()
    base.mkdir(parents=True)
    try:
        (base/'common').mkdir()
        for name, path in sources.items():
            target = base/'common/source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == admitted[name], 'Tested source changed during copy: '+name)
        bundle = read(witness['source_bundle']['path'])
        source_hashes = dict(admitted)
        for name, item in bundle['files'].items():
            target = base/'common/source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x', encoding='utf-8', newline='') as handle:
                handle.write(item['text'])
            require(sha(target) == item['sha256'], 'Compiled source archive differs: '+name)
            source_hashes[name] = item['sha256']
        for name in [old.EXAMPLE]+RUST_TESTS:
            target = base/'common/source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
            require(sha(target) == witness['tested_sources'][name], 'Tested Rust source changed during copy: '+name)
            source_hashes[name] = witness['tested_sources'][name]
        for key, name in (('executable', 'evolving_dimer_benchmark'), ('source_bundle', 'source-bundle.json')):
            shutil.copyfile(witness[key]['path'], base/'common'/name)
            require(sha(base/'common'/name) == witness[key]['sha256'], 'Compiled archive differs')
        executable = base/'common/evolving_dimer_benchmark'
        executable.chmod(0o755)
        write(base/'common/build-witness.json', witness)
        write(base/'common/scientific-allocation.json', allocation())
        write(base/'analysis-plan.json', analysis)
        protocol = dict(schema='rigid-surrogate-protocol-v1', allocation=allocation(),
            surrogate_policy=policy(), roles=ROLES, inherited_campaign=authority,
            control_analysis=controls, control_mapping=mapping,
            build_witness=record(base/'common/build-witness.json'), build_validation=build_ref,
            validation=validation_ref, analysis_plan=record(base/'analysis-plan.json'), source_files=source_hashes,
            scope=SCOPE, all_attempts_retained=True, analysis_after_complete_inventory=True,
            retry=False, replacement=False, assembly_gate_open=False,
            rigid_members='Fixed context0 root/child labels; first member is the fixed handle for every inner step.',
            surrogate_cloud='Reuse each original family body cloud for both mobile bodies; raw box volume/raw_count; no new draws.',
            decisions='Four unchanged canonical local updates and one rigid collective per block; one physical gate for each nonidentity endpoint.',
            interruption='Durable begun/outcome rows; partial tails and fatal budgets stop the allocation and require audit, never automatic retry.',
            inherited_evidence='Completed original all-row/contact observer plus independent preparation audit. Old raw trajectories are not reread.',
            selection='Context0 and all eight start/stream families fixed after development; no trajectory-outcome filtering among the 24 new chains.')
        write(base/'protocol.json', protocol)
        config = configuration(original_config, base, authority, controls, mapping, compiled)
        write(base/'config.json', config)
        files = {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}
        write(base/'freeze.json', dict(schema='rigid-surrogate-freeze-v1', complete=True,
            scientific_execution_started=False, files=files, input_sha256=dict(inputs.files)))
        write(base/'binding.json', dict(schema='evolving-dimer-binding-v1',
            config_sha256=sha(base/'config.json'), protocol_sha256=sha(base/'protocol.json'),
            freeze_sha256=sha(base/'freeze.json'), example_source_sha256=source_hashes[old.EXAMPLE],
            compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'),
            executable_sha256=sha(executable), isolated_build_witness=record(base/'common/build-witness.json')))
        write(base/'run-binding.json', dict(schema='evolving-dimer-run-binding-v1', complete=True,
            config_sha256=sha(base/'config.json'), protocol_sha256=sha(base/'protocol.json'),
            prelaunch_binding=record(base/'binding.json'), prepared_manifest=authority['prepared_manifest'],
            prepared_files=prepared, cloud_banks=32, alternative_starts=16,
            inherited_preparation_authority=authority))
        execution_jobs = []
        for job in sorted(jobs(), key=lambda j: (j['queue_key'], j['id'])):
            execution_jobs.append(dict(id=f"job-{job['id']:03}",
                population=f"context0-{job['initialization']}-{job['stream']}", phase='producer',
                argv=[str(executable), '--mode', 'run', '--config', str(base/'config.json'),
                      '--binding', str(base/'run-binding.json'), '--job', str(job['id'])],
                terminal=dict(path=str(base/'chains'/f"job-{job['id']:03}"/'terminal.json'),
                              success_contract='complete'), **LIMITS))
        consumed = dict(inputs.files)
        consumed.update({str(p): sha(p) for p in base.rglob('*') if p.is_file()})
        execution = dict(schema=driver.SCHEMA, root=str(base), maximum_workers=1, threads=1,
            files=consumed, jobs=execution_jobs, executable_resolutions={str(executable): str(executable)},
            preparation_receipt=str(base/'preparation.json'), scope=SCOPE)
        write(base/'execution-plan.json', execution)
        inputs.recheck()
        verify(base)
        recheck_sources(sources, admitted)
        receipt = dict(schema='rigid-surrogate-preparation-v1', complete=True, launched=False,
            protocol=record(base/'protocol.json'), execution_plan=record(base/'execution-plan.json'),
            config=record(base/'config.json'), allocation=allocation(), new_physical_draws=0,
            new_geometry_queries=0, source_archived=True, assembly_gate_open=False)
        write(base/'preparation.json', receipt)
        spec = importlib.util.spec_from_file_location('rigid_surrogate_frozen_driver', base/'common/source'/DRIVER)
        frozen_driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen_driver)
        frozen_driver.verify_plan(base/'execution-plan.json', execution,
                                  receipt['execution_plan']['sha256'], fresh=True)
        return receipt
    except BaseException as error:
        write(base/'preparation-failure.json', dict(complete=False, launched=False,
            error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'original', 'control-analysis', 'build-validation', 'validation'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-bundle', type=Path)
    args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.root, args.original, args.control_analysis,
        args.build_validation, args.validation, args.source_bundle), indent=2))
