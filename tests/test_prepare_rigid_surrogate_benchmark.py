#!/usr/bin/env python3
"""Synthetic metadata only: no Rust execution, geometry or physical draws."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_rigid_surrogate_benchmark as p


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    p.write(path, value)
    return p.record(path)


def opaque(path, text='opaque synthetic input; never a scientific query'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return p.record(path)


def inherited_fixture(root):
    original, analysis = root/'original', root/'control'
    original.mkdir(); (analysis/'analysis').mkdir(parents=True)
    physical = opaque(original/'shape.json')
    contexts = [dict(root=a, child=b, anchor=228+c) for c, (a, b) in enumerate(p.old.EXPECTED_PAIRS)]
    config = dict(schema='evolving-dimer-benchmark-v1', jobs=p.old.jobs(), allocation=p.old.allocation(),
        master_seed=p.old.MASTER, contexts=contexts, shape=physical, source_frame=physical,
        source_config=physical, physical=dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64.,
        wall_radius=593.742500239952), limits=dict(cpu_seconds=1800., raw_per_leg=20000000),
        local=dict(member_order=[0, 1, 0, 1], translation_std_A=.2, rotation_std_degrees=1.),
        factorized=dict(root_cap=32, internal_cap=32, joint_cap=1, order='root_first'),
        cloud=dict(raw_count=16384), compiled_source_sha256={'src/old.rs': 'f'*64},
        output=str(original/'execution'), preparation_output=str(original/'prepared'))
    config_ref = save(original/'config.json', config)
    source = opaque(original/'common/source/tools/historical.py', 'historical = True\n')
    protocol = save(original/'protocol.json', {'synthetic': True})
    freeze = save(original/'freeze.json', dict(schema='evolving-dimer-preparation-freeze-v1',
        complete=True, scientific_execution_started=False,
        files={'config.json': config_ref['sha256'], 'protocol.json': protocol['sha256'],
               'common/source/tools/historical.py': source['sha256']},
        original_input_bindings={'shape': physical}))
    binding = save(original/'binding.json', dict(config_sha256=config_ref['sha256'],
        freeze_sha256=freeze['sha256'], protocol_sha256=protocol['sha256']))
    files, banks, starts = [], [], []
    for context in range(4):
        for initialization in p.old.INITIALIZATIONS:
            for stream in range(4):
                raw = opaque(original/'prepared'/f'cloud-{context}-{initialization}-{stream}.bin')
                metadata = opaque(original/'prepared'/f'cloud-{context}-{initialization}-{stream}.json')
                files.extend([raw, metadata])
                banks.append(dict(context_index=context, initialization=initialization,
                                  stream=stream, raw=raw, metadata=metadata))
        for stream in range(4):
            start = opaque(original/'prepared'/f'start-{context}-{stream}.json')
            ledger = opaque(original/'prepared'/f'start-{context}-{stream}.jsonl')
            files.extend([start, ledger])
            starts.append(dict(context_index=context, stream=stream, record=start, ledger=ledger))
    prepared = save(original/'prepared/manifest.json', dict(complete=True, passed=True,
        all_attempts_retained=True, config_sha256=config_ref['sha256'], binding_sha256=binding['sha256'],
        files=files, cloud_banks=banks, alternative_starts=starts))
    run = save(original/'run-binding.json', dict(complete=True, config_sha256=config_ref['sha256'],
        prelaunch_binding=binding, prepared_manifest=prepared,
        prepared_files={f['path']: f['sha256'] for f in files}))
    audit_protocol = save(original/'preparation-audit-v2/protocol.json', {'synthetic': True})
    save(original/'preparation-audit-v2/result.json', dict(
        schema='evolving-dimer-preparation-independent-audit-v1', complete=True, passed=True,
        physical=config['physical'], source_contexts=contexts, protocol=audit_protocol,
        input_sha256={r['path']: r['sha256'] for r in (config_ref, binding, prepared)}))
    completed, chains, cached = [], [], {}
    consumed = {r['path']: r['sha256'] for r in (config_ref, run)}
    for job in config['jobs']:
        ident = job['id']; directory = original/'execution'/f'job-{ident:03}'
        # Intentionally nonexistent; preparation must consume cached rows only.
        trajectory = dict(path=str(directory/'trajectory.jsonl'), sha256='a'*64)
        terminal = save(directory/'terminal.json', dict(complete=True, conditional_target=True,
            job=job, blocks=4608, config_sha256=config_ref['sha256'], binding_sha256=run['sha256'],
            trajectory=trajectory, cpu_seconds=10.))
        consumed[terminal['path']] = terminal['sha256']; consumed[trajectory['path']] = trajectory['sha256']
        completed.append(dict(job=job, success=True, terminal_sha256=terminal['sha256']))
        chains.append(dict(job=job, trajectory=trajectory, metrics=dict(full_sampler_cpu_seconds=10.)))
        if job['context_index'] == 0 and job['arm'] in ('local', 'm4'):
            name = f'job-{ident:03}-observations.jsonl'
            cached[name] = opaque(analysis/'analysis'/name)['sha256']
    save(original/'dispatch/status.json', dict(complete=True, passed=True, active=[], unstarted=[], completed=completed))
    report = save(analysis/'analysis/analysis.json', dict(complete=True, chains=chains))
    ib = save(analysis/'analysis/input-binding.json', dict(config_sha256=config_ref['sha256'],
        run_binding_sha256=run['sha256'], source_files={'tools/historical.py': source['sha256']}))
    manifest = save(analysis/'analysis/manifest.json', dict(complete=True,
        files={'analysis.json': report['sha256'], 'input-binding.json': ib['sha256'], **cached}))
    execution = save(analysis/'execution-plan.json', dict(files=consumed))
    save(analysis/'summary.json', dict(complete=True, passed=True, chains=96, analysis=report,
        manifest_sha256=manifest['sha256'], plan_sha256=execution['sha256']))
    save(analysis/'exit.json', dict(returncode=0, child_started=True, child_drained=True, error=None))
    return original, analysis


def build_fixture(root):
    project = root/'project'
    names = ['Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md',
             'src/lib.rs', 'src/rigid_surrogate_chain.rs', p.old.EXAMPLE]+p.RUST_TESTS
    refs = {name: opaque(project/name, '// synthetic '+name+'\n') for name in names}
    compiled = {name: dict(text=(project/name).read_text(), sha256=refs[name]['sha256'])
                for name in names if name not in [p.old.EXAMPLE]+p.RUST_TESTS}
    bundle = save(root/'build/source-bundle.json', dict(schema=1, files=compiled))
    executable_path = root/'build/executable'
    # Opaque bytes represent an executable embedding; this file is never run.
    executable_path.write_bytes(b'FAKE EXECUTABLE\n'+Path(bundle['path']).read_bytes()
                                +b'\n'+(project/p.old.EXAMPLE).read_bytes())
    sources = {r['path']: r['sha256'] for name, r in refs.items()
               if name not in ('build.rs', 'vendor/README.md')}
    runs = [dict(argv=argv, returncode=0, child_drained=True,
                 log=opaque(root/'build'/f'command-{i}.log'))
            for i, argv in enumerate(p.REQUIRED_COMMANDS)]
    receipt = save(root/'build/validation.json', dict(schema='rigid-surrogate-validation-v1',
        complete=True, passed=True, error=None, source_before=sources, source_after=copy.deepcopy(sources),
        production_before='0'*64, production_after='0'*64, runs=runs,
        executable=p.record(executable_path)))
    return project, receipt, bundle


def python_receipt(root, sources):
    hashes = {str(path): p.sha(path) for path in sources.values()}
    return save(root/'python-validation.json', dict(complete=True, passed=True,
        source_before=hashes, source_after=copy.deepcopy(hashes)))


class PreparationTests(unittest.TestCase):
    def test_allocation_controls_reused_three_times_without_old_journal_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            original, analysis = inherited_fixture(Path(temp))
            inputs = p.Inputs()
            config, _, controls, _, mapping = p.inherited(inputs, original, analysis)
            self.assertEqual(len(p.jobs()), 24)
            self.assertEqual({j['id'] for j in p.jobs()}, set(range(24)))
            self.assertEqual({(j['arm'], j['initialization'], j['stream']) for j in p.jobs()},
                {(a, i, s) for a in p.ARMS for i in p.old.INITIALIZATIONS for s in range(4)})
            self.assertEqual(len(controls['observations']), 16)
            self.assertEqual(len(mapping), 24)
            for init in p.old.INITIALIZATIONS:
                for stream in range(4):
                    family = [m for m in mapping if m['initialization'] == init and m['stream'] == stream]
                    self.assertEqual(len(family), 3)
                    self.assertTrue(all(m['controls'] == family[0]['controls'] for m in family))
                    self.assertEqual(set(family[0]['controls']), {'local', 'm4'})
            self.assertFalse(any(Path(path).name == 'trajectory.jsonl' for path in inputs.files))
            a = p.allocation()
            self.assertEqual((a['local_attempts'], a['dimer_attempts'], a['maximum_inner_steps']),
                             (442368, 110592, 626688))
            self.assertEqual(a['maximum_journal_rows'], 24*(1+4608*7))
            self.assertEqual(a['raw_points_per_cloud'], config['cloud']['raw_count'])
            self.assertEqual(p.policy(), {'schema': 'rigid-surrogate-policy-v1',
                                         'proposal_scales': {'source': 'local'}})
            self.assertEqual(a['per_chain_limits'], dict(cpu_limit_seconds=1800,
                wall_limit_seconds=3600, address_space_limit_bytes=8*2**30))

    def test_cache_preparation_and_archive_tampering_rejected(self):
        for choice in ('cache', 'cloud', 'archive'):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as temp:
                original, analysis = inherited_fixture(Path(temp))
                path = (next((analysis/'analysis').glob('*observations.jsonl')) if choice == 'cache'
                        else original/('prepared/cloud-0-source-0.bin' if choice == 'cloud'
                                       else 'common/source/tools/historical.py'))
                path.write_text('changed')
                with self.assertRaises(ValueError):
                    p.inherited(p.Inputs(), original, analysis)

    def test_build_requires_tested_sources_drained_commands_and_embedded_bundle(self):
        for change in (None, 'sources', 'production', 'drain', 'command', 'example', 'bundle', 'executable'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); project, ref, bundle = build_fixture(root)
                value = p.read(ref['path'])
                if change in ('sources', 'production', 'drain', 'command'):
                    if change == 'sources': value['source_after'].pop(next(iter(value['source_after'])))
                    elif change == 'production': value['production_after'] = '1'*64
                    elif change == 'drain': value['runs'][0]['child_drained'] = False
                    else: value['runs'][0]['argv'] = ['true']
                    Path(ref['path']).unlink(); p.write(ref['path'], value)
                elif change == 'example': (project/p.old.EXAMPLE).write_text('// untested change\n')
                elif change == 'bundle':
                    content = p.read(bundle['path']); content['files']['build.rs']['text'] += '// altered\n'
                    content['files']['build.rs']['sha256'] = hashlib.sha256(content['files']['build.rs']['text'].encode()).hexdigest()
                    Path(bundle['path']).unlink(); p.write(bundle['path'], content)
                elif change == 'executable': Path(value['executable']['path']).write_bytes(b'changed')
                with patch.object(p, 'ROOT', project):
                    if change is None:
                        _, witness, compiled = p.validate_build(p.Inputs(), ref['path'], bundle['path'])
                        self.assertEqual(witness['new_arms'], p.ARMS)
                        self.assertIn('src/rigid_surrogate_chain.rs', compiled)
                    else:
                        with self.assertRaises(ValueError):
                            p.validate_build(p.Inputs(), ref['path'], bundle['path'])

    def test_freeze_roundtrip_no_launch_and_exact_matched_configuration(self):
        sources = p.source_paths()
        self.assertIn(p.SELF, sources); self.assertIn(p.TEST, sources)
        self.assertIn('tools/analyze_rigid_surrogate_benchmark.py', sources)
        self.assertIn('tools/test_analyze_rigid_surrogate_benchmark.py', sources)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); original, analysis = inherited_fixture(root)
            project, build, bundle = build_fixture(root)
            tested = python_receipt(root, sources); destination = root/'new'
            before = {str(f): p.sha(f) for f in original.rglob('*') if f.is_file()}
            with patch.object(p, 'ROOT', project), patch.object(p, 'source_paths', return_value=sources):
                receipt = p.prepare(destination, original, analysis, build['path'], tested['path'], bundle['path'])
                self.assertTrue(receipt['complete']); self.assertFalse(receipt['launched'])
                config = p.verify(destination); execution = p.read(destination/'execution-plan.json')
                self.assertEqual(len(execution['jobs']), 24)
                self.assertEqual((execution['maximum_workers'], execution['threads']), (1, 1))
                self.assertTrue(all(j['phase'] == 'producer' and j['cpu_limit_seconds'] == 1800
                    and j['wall_limit_seconds'] == 3600 and j['address_space_limit_bytes'] == 8*2**30
                    for j in execution['jobs']))
                old = p.read(original/'config.json')
                for key in ('physical', 'local', 'factorized', 'cloud', 'contexts', 'source_frame',
                            'source_config', 'master_seed', 'preparation_output'):
                    self.assertEqual(config[key], old[key])
                self.assertEqual(config['surrogate_policy'], p.policy())
                self.assertEqual(p.read(destination/'run-binding.json')['prepared_manifest'],
                                 p.record(original/'prepared/manifest.json'))
                self.assertFalse((destination/'execution').exists()); self.assertFalse((destination/'chains').exists())
                self.assertFalse(any(Path(path).name == 'trajectory.jsonl' for path in execution['files']))
                self.assertEqual(before, {str(f): p.sha(f) for f in original.rglob('*') if f.is_file()})
                with self.assertRaises(ValueError):
                    p.prepare(destination, original, analysis, build['path'], tested['path'], bundle['path'])
                (destination/'common/source/src/rigid_surrogate_chain.rs').write_text('// changed\n')
                with self.assertRaises(ValueError): p.verify(destination)

    def test_incomplete_python_receipt_and_source_changes_during_copy_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); original, analysis = inherited_fixture(root)
            project, build, bundle = build_fixture(root)
            probe = root/'probe.py'; probe.write_text('TESTED = True\n')
            sources = {'tools/probe.py': probe, p.DRIVER: p.ROOT/p.DRIVER}
            tested = python_receipt(root, sources)
            invalid = p.read(tested['path']); invalid['source_after'].pop(str(probe))
            invalid_ref = save(root/'invalid-validation.json', invalid)
            with self.assertRaises(ValueError): p.validate_tests(p.Inputs(), invalid_ref['path'], sources)
            destination = root/'failed'; copyfile = p.shutil.copyfile
            def changing_copy(source, target):
                if Path(source) == probe: probe.write_text('TESTED = False\n')
                return copyfile(source, target)
            with patch.object(p, 'ROOT', project), patch.object(p, 'source_paths', return_value=sources), \
                    patch.object(p.shutil, 'copyfile', side_effect=changing_copy):
                with self.assertRaisesRegex(ValueError, 'Tested source changed'):
                    p.prepare(destination, original, analysis, build['path'], tested['path'], bundle['path'])
            self.assertFalse((destination/'preparation.json').exists())
            self.assertFalse(p.read(destination/'preparation-failure.json')['complete'])
            with self.assertRaises(ValueError):
                p.prepare(destination, original, analysis, build['path'], tested['path'], bundle['path'])


if __name__ == '__main__':
    unittest.main()
