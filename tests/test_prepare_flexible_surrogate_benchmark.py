#!/usr/bin/env python3
"""Synthetic provenance only: no Rust, geometry, journals, or physical draws."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_flexible_surrogate_benchmark as p
import test_prepare_rigid_surrogate_benchmark as fixtures

save, opaque, python_receipt = fixtures.save, fixtures.opaque, fixtures.python_receipt


def replace(path, value):
    Path(path).unlink()
    return save(Path(path), value)


def inherited_fixture(root):
    original, old_analysis = fixtures.inherited_fixture(root)
    # Match the original observer's actual historical schema. Its raw journals
    # deliberately do not exist, while their completed observer hashes do.
    report = p.read(old_analysis/'analysis/analysis.json')
    report.update(schema='conditional-dimer-analysis-v1',
                  input_files=p.read(old_analysis/'execution-plan.json')['files'])
    ref = replace(old_analysis/'analysis/analysis.json', report)
    manifest = p.read(old_analysis/'analysis/manifest.json')
    manifest['files']['analysis.json'] = ref['sha256']
    mref = replace(old_analysis/'analysis/manifest.json', manifest)
    summary = p.read(old_analysis/'summary.json')
    summary.update(analysis=ref, manifest_sha256=mref['sha256'])
    replace(old_analysis/'summary.json', summary)
    inputs = p.Inputs()
    config, authority, controls, prepared, mapping = p.rigid.inherited(inputs, original, old_analysis)
    campaign, analysis = root/'rigid', root/'rigid-analysis'
    campaign.mkdir(); (analysis/'analysis').mkdir(parents=True)
    archive = opaque(campaign/'common/source/tools/historical.py', 'historical = True\n')
    source_hashes = {'tools/historical.py': archive['sha256']}
    save(campaign/'common/scientific-allocation.json', p.rigid.allocation())
    save(campaign/'protocol.json', dict(roles=p.rigid.ROLES, surrogate_policy=p.rigid.policy(),
        allocation=p.rigid.allocation(), assembly_gate_open=False, source_files=source_hashes))
    config = p.rigid.configuration(config, campaign, authority, controls, mapping, {'src/old.rs': 'f'*64})
    save(campaign/'config.json', config)
    save(campaign/'run-binding.json', dict(complete=True))
    save(campaign/'freeze.json', dict(schema='rigid-surrogate-freeze-v1', complete=True,
        scientific_execution_started=False,
        files={str(f.relative_to(campaign)): p.sha(f) for f in campaign.rglob('*') if f.is_file()},
        input_sha256=inputs.files))
    execution = save(campaign/'execution-plan.json', {'synthetic': True})
    status = dict(complete=True, passed=True, failure=None, active=None, unstarted=[],
                  completed=[{'success': True} for _ in range(24)], plan_sha256=execution['sha256'])
    save(campaign/'execution/status.json', status); save(campaign/'execution/summary.json', status)
    consumed = {str(campaign/n): p.sha(campaign/n) for n in ('config.json', 'run-binding.json',
        'protocol.json', 'freeze.json', 'execution-plan.json', 'execution/status.json', 'execution/summary.json')}
    chains, caches = [], {}
    for job in config['jobs']:
        trajectory = dict(path=str(campaign/'chains'/f"job-{job['id']:03}"/'trajectory.jsonl'), sha256='b'*64)
        terminal = save(campaign/'chains'/f"job-{job['id']:03}"/'terminal.json',
            dict(complete=True, conditional_target=True, job=job, blocks=4608, trajectory=trajectory, cpu_seconds=20.))
        for item in (terminal, trajectory): consumed[item['path']] = item['sha256']
        chains.append(dict(job=job, trajectory=trajectory, reused_control=False,
                           metrics=dict(full_sampler_cpu_seconds=20.)))
        name = f"job-{job['id']:03}-observations.jsonl"
        caches[name] = opaque(analysis/'analysis'/name)['sha256']
    for chain in report['chains']:
        job = chain['job']
        if job['context_index'] != 0 or job['arm'] not in ('local', 'm4'): continue
        chains.append(dict(chain, reused_control=True))
        for item in (controls['observations'][str(job['id'])], controls['terminals'][str(job['id'])],
                     controls['analysis'], controls['manifest']):
            consumed[item['path']] = item['sha256']
    declaration = save(campaign/'analysis-plan.json', dict(schema='synthetic-plan-v1'))
    result = save(analysis/'analysis/analysis.json', dict(schema='rigid-surrogate-analysis-v1',
        complete=True, new_chains=24, reused_control_chains=16, chains=chains,
        new_geometry_endpoints=110616, old_geometry_queries=0, new_physical_draws=0, native_observer=False,
        input_sha256=consumed, source_sha256=source_hashes, analysis_plan=p.read(declaration['path'])))
    binding = save(analysis/'analysis/input-binding.json', dict(input_sha256=consumed, source_sha256=source_hashes))
    save(analysis/'analysis/manifest.json', dict(complete=True,
        files={'analysis.json': result['sha256'], 'input-binding.json': binding['sha256'], **caches}))
    protocol = save(analysis/'protocol.json', dict(schema='rigid-surrogate-analysis-dispatch-v1',
        new_chains=24, cached_control_chains=16, analysis_plan=declaration))
    job = dict(id='whole-contact-analysis', population='whole', phase='statistics', argv=['never-run'],
               terminal=dict(path=result['path'], success_contract='complete'))
    plan = save(analysis/'execution-plan.json', dict(schema=p.driver.SCHEMA, root=str(analysis),
        maximum_workers=1, threads=1, jobs=[job],
        files={ref['path']: ref['sha256'] for ref in (protocol, declaration)}))
    preparation = save(analysis/'preparation.json', dict(complete=True, launched=False,
        execution_plan=plan, protocol=protocol))
    save(analysis/'execution/claim.json', dict(schema=p.driver.SCHEMA, plan_sha256=plan['sha256'],
        maximum_workers=1, threads=1, retries=0, replacements=0, preparation_receipt=preparation))
    done = dict(job, success=True, child_started=True, child_drained=True, returncode=0,
        error=None, timeout=False, retries=0, replacements=0, success_contract='complete',
        pid=123, birth_ticks=12, terminal=result)
    directory = analysis/'execution/jobs/000-whole-contact-analysis'
    save(directory/'attempt.json', dict(job=job))
    save(directory/'process.json', {k: done[k] for k in ('id', 'pid', 'birth_ticks', 'argv')})
    save(directory/'success.json', done); save(directory/'exit.json', done)
    status = dict(complete=True, passed=True, failure=None, active=None, unstarted=[],
                  completed=[done], plan_sha256=plan['sha256'])
    save(analysis/'execution/status.json', status); save(analysis/'execution/summary.json', status)
    return original, campaign, analysis


def build_fixture(root):
    with patch.object(fixtures.p, 'RUST_TESTS', p.RUST_TESTS), \
            patch.object(fixtures.p, 'REQUIRED_COMMANDS', p.REQUIRED_COMMANDS):
        project, ref, bundle = fixtures.build_fixture(root)
    old = project/'src/rigid_surrogate_chain.rs'
    old.rename(project/'src/flexible_surrogate_chain.rs')
    value = p.read(ref['path']); value['schema'] = 'flexible-surrogate-benchmark-validation-v1'
    for k in ('source_before', 'source_after'):
        value[k][str(project/'src/flexible_surrogate_chain.rs')] = value[k].pop(str(old))
    content = p.read(bundle['path'])
    content['files']['src/flexible_surrogate_chain.rs'] = content['files'].pop('src/rigid_surrogate_chain.rs')
    bundle = replace(bundle['path'], content)
    exe = Path(value['executable']['path'])
    exe.write_bytes(b'FAKE, NEVER RUN\n'+Path(bundle['path']).read_bytes()+(project/p.old.EXAMPLE).read_bytes())
    value['executable'] = p.record(exe)
    ref = replace(ref['path'], value)
    return project, ref, bundle


def reference_fixture(root, project, build):
    directory = root/'reference-validation'; output = directory/'reference'
    sources = {str(Path(n).relative_to(project)): h for n, h in p.read(build['path'])['source_after'].items()}
    protocol = save(output/'protocol.json', dict(schema='flexible-surrogate-independent-reference-protocol-v1',
        streams=4, sources_per_stream=2048, arms=[[1, 1., 'm1_guided'], [8, 1., 'm8_guided'], [8, 0., 'm8_zero']],
        reference_source_sha256=sources[p.RUST_TESTS[1]]))
    counts = dict(attempted=9000, accepted=8192)
    summary = save(output/'summary.json', dict(complete=True, source_counts=counts))
    # Completed audit pins these nonexistent journals. The preparer must not rescan.
    hashes = {'protocol.json': protocol['sha256'], 'summary.json': summary['sha256'],
              'source-attempts.jsonl': 'c'*64, 'kernel-attempts.jsonl': 'd'*64}
    receipt = save(output/'receipt.json', dict(schema='flexible-surrogate-independent-reference-receipt-v1',
        complete=True, passed=True, numerical_allocation_complete=True, scientific_checks_passed=True,
        error=None, retries=0, replacement_draws=0, output_sha256=hashes.copy(), source_counts=counts))
    hashes['receipt.json'] = receipt['sha256']
    validation = save(directory/'validation.json', dict(schema='flexible-surrogate-reference-validation-v1',
        complete=True, passed=True, returncode=0, child_started=True, child_drained=True, error=None,
        source_before=sources, source_after=copy.deepcopy(sources), production_before='0'*64, production_after='0'*64,
        outputs={'reference/'+n: h for n, h in hashes.items()}))
    audit = save(directory/'audit.json', dict(schema='flexible-surrogate-reference-journal-audit-v1',
        complete=True, passed=True, reference_scientific_checks_passed=True,
        validation_sha256=validation['sha256'], inventory={k: 8192 for k in ('m1_guided', 'm8_guided', 'm8_zero')},
        input_sha256=hashes, source_counts=counts))
    return validation, audit, sources


class PreparationTests(unittest.TestCase):
    def test_all_forty_controls_use_full_identity_and_original_authority_without_journals(self):
        with tempfile.TemporaryDirectory() as temp:
            original, campaign, analysis = inherited_fixture(Path(temp))
            inputs = p.Inputs()
            config, authority, controls, prepared = p.inherited(inputs, campaign, analysis)
            entries = controls['observations']
            self.assertEqual(len(entries), 40)
            self.assertEqual(len({p.identity(e['job']) for e in entries}), 40)
            self.assertLess(len({e['job']['id'] for e in entries}), 40)
            for entry in entries:
                inherited = entry['job']['arm'] in ('local', 'm4')
                self.assertEqual(Path(entry['analysis']['path']).is_relative_to(analysis), not inherited)
                self.assertEqual(p.read(entry['terminal']['path'])['job'], entry['job'])
            self.assertFalse(any(Path(n).name == 'trajectory.jsonl' for n in inputs.files))
            self.assertEqual(authority['config'], p.record(original/'config.json'))
            self.assertEqual(prepared, p.read(original/'run-binding.json')['prepared_files'])
            self.assertEqual(config, p.read(original/'config.json'))
            self.assertTrue(all(set(ref) == {'path', 'sha256'} for ref in controls['authority'].values()))
            self.assertEqual(len(p.jobs()), 24)
            a = p.allocation()
            self.assertEqual((a['reused_control_chains'], a['combined_analysis_chains']), (40, 64))
            self.assertEqual(a['maximum_journal_rows'], 24*(1+4608*7))
            self.assertEqual((a['new_raw_cloud_points'], a['new_preparation_attempts']), (0, 0))
            self.assertEqual(a['per_chain_limits'], p.LIMITS)

    def test_incomplete_lifecycle_and_changed_inherited_evidence_are_rejected(self):
        for change in ('live', 'failed', 'cache', 'cloud', 'archive', 'original_trajectory_binding'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                original, campaign, analysis = inherited_fixture(Path(temp))
                if change in ('live', 'failed'):
                    for name in ('status', 'summary'):
                        path = analysis/f'execution/{name}.json'; value = p.read(path)
                        value['active' if change == 'live' else 'passed'] = {'pid': 123} if change == 'live' else False
                        replace(path, value)
                elif change == 'cache': next((analysis/'analysis').glob('*observations.jsonl')).write_text('changed')
                elif change == 'cloud': (original/'prepared/cloud-0-source-0.bin').write_text('changed')
                elif change == 'archive': (campaign/'common/source/tools/historical.py').write_text('changed')
                else:
                    path = original.parent/'control/analysis/analysis.json'
                    value = p.read(path); value['input_files'] = {}; replace(path, value)
                with self.assertRaises(ValueError): p.inherited(p.Inputs(), campaign, analysis)

    def test_tested_build_commands_sources_bundle_and_production_must_match(self):
        for change in (None, 'sources', 'production', 'drain', 'command', 'example', 'bundle', 'executable'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                project, ref, bundle = build_fixture(Path(temp)); value = p.read(ref['path'])
                if change in ('sources', 'production', 'drain', 'command'):
                    if change == 'sources': value['source_after'].pop(next(iter(value['source_after'])))
                    elif change == 'production': value['production_after'] = '1'*64
                    elif change == 'drain': value['runs'][0]['child_drained'] = False
                    else: value['runs'][0]['argv'] = ['true']
                    replace(ref['path'], value)
                elif change == 'example': (project/p.old.EXAMPLE).write_text('// untested\n')
                elif change == 'bundle':
                    content = p.read(bundle['path']); content['files']['build.rs']['text'] += '// changed\n'
                    content['files']['build.rs']['sha256'] = hashlib.sha256(content['files']['build.rs']['text'].encode()).hexdigest()
                    replace(bundle['path'], content)
                elif change == 'executable': Path(value['executable']['path']).write_bytes(b'changed')
                with patch.object(p, 'ROOT', project):
                    if change is None:
                        _, witness, compiled = p.validate_build(p.Inputs(), ref['path'], bundle['path'])
                        self.assertEqual(witness['new_arms'], p.ARMS)
                        self.assertIn('src/flexible_surrogate_chain.rs', compiled)
                    else:
                        with self.assertRaises(ValueError): p.validate_build(p.Inputs(), ref['path'], bundle['path'])

    def test_reference_requires_successful_pinned_runner_audit_and_matched_runtime(self):
        for change in (None, 'pin', 'passed', 'returncode', 'child_started', 'child_drained', 'inventory', 'scientific', 'source'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); project, build, _ = build_fixture(root)
                validation, audit, sources = reference_fixture(root, project, build)
                if change in ('passed', 'returncode', 'child_started', 'child_drained'):
                    value = p.read(validation['path']); value[change] = 1 if change == 'returncode' else False
                    validation = replace(validation['path'], value)
                    value = p.read(audit['path']); value['validation_sha256'] = validation['sha256']
                    audit = replace(audit['path'], value)
                elif change in ('inventory', 'scientific'):
                    value = p.read(audit['path'])
                    if change == 'inventory': value['inventory']['m8_guided'] -= 1
                    else: value['reference_scientific_checks_passed'] = False
                    audit = replace(audit['path'], value)
                elif change == 'source': sources['src/flexible_surrogate_chain.rs'] = 'e'*64
                if change == 'pin': validation['sha256'] = 'f'*64
                inputs = p.Inputs()
                args = (inputs, validation['path'], validation['sha256'], audit['path'], audit['sha256'], sources)
                if change is None:
                    result = p.validate_reference(*args)
                    self.assertFalse(result['reference_rerun']); self.assertFalse(result['journal_rescan'])
                    self.assertFalse(any(n.endswith('-attempts.jsonl') for n in inputs.files))
                else:
                    with self.assertRaises(ValueError): p.validate_reference(*args)

    def test_freeze_no_launch_preserves_original_starts_clouds_and_single_worker_caps(self):
        sources = p.source_paths()
        for name in (p.SELF, p.TEST, 'tools/analyze_flexible_surrogate_benchmark.py',
                     'tests/test_analyze_flexible_surrogate_benchmark.py', 'tools/plot_rigid_surrogate_benchmark.py',
                     'tests/test_prepare_rigid_surrogate_benchmark.py'):
            self.assertIn(name, sources)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); original, campaign, analysis = inherited_fixture(root)
            project, build, bundle = build_fixture(root)
            reference, audit, _ = reference_fixture(root, project, build)
            tested = python_receipt(root, sources); destination = root/'new'
            before = {str(f): p.sha(f) for f in original.rglob('*') if f.is_file()}
            args = (destination, campaign, analysis, build['path'], tested['path'], reference['path'],
                    reference['sha256'], audit['path'], audit['sha256'], bundle['path'])
            with patch.object(p, 'ROOT', project), patch.object(p, 'source_paths', return_value=sources):
                receipt = p.prepare(*args)
                self.assertTrue(receipt['complete']); self.assertFalse(receipt['launched'])
                config = p.verify(destination); execution = p.read(destination/'execution-plan.json')
                self.assertEqual(len(execution['jobs']), 24)
                self.assertEqual((execution['maximum_workers'], execution['threads']), (1, 1))
                for job in execution['jobs']:
                    self.assertEqual({k: job[k] for k in p.LIMITS}, p.LIMITS)
                    self.assertEqual(job['phase'], 'producer')
                for key in ('physical', 'local', 'factorized', 'cloud', 'contexts', 'source_frame',
                            'source_config', 'master_seed', 'preparation_output'):
                    self.assertEqual(config[key], p.read(original/'config.json')[key])
                self.assertEqual(config['flexible_surrogate_policy'], p.policy())
                self.assertFalse(any(k in config for k in ('surrogate_policy', 'two_root_policy', 'singleton_policy')))
                self.assertEqual(p.read(destination/'run-binding.json')['prepared_manifest'],
                                 p.record(original/'prepared/manifest.json'))
                self.assertFalse((destination/'execution').exists()); self.assertFalse((destination/'chains').exists())
                self.assertFalse(any(Path(n).name == 'trajectory.jsonl' for n in execution['files']))
                self.assertEqual(before, {str(f): p.sha(f) for f in original.rglob('*') if f.is_file()})
                with self.assertRaises(ValueError): p.prepare(*args)
                (destination/'common/source/src/flexible_surrogate_chain.rs').write_text('// changed\n')
                with self.assertRaises(ValueError): p.verify(destination)

    def test_source_changes_during_archive_leave_failed_fresh_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, campaign, analysis = inherited_fixture(root)
            project, build, bundle = build_fixture(root)
            reference, audit, _ = reference_fixture(root, project, build)
            probe = root/'probe.py'; probe.write_text('TESTED = True\n')
            sources = {'tools/probe.py': probe, p.DRIVER: p.ROOT/p.DRIVER}
            tested = python_receipt(root, sources); destination = root/'failed'
            invalid = p.read(tested['path']); invalid['source_after'].pop(str(probe))
            invalid_ref = save(root/'invalid.json', invalid)
            with self.assertRaises(ValueError): p.validate_tests(p.Inputs(), invalid_ref['path'], sources)
            copyfile = p.shutil.copyfile
            def changing_copy(source, target):
                if Path(source) == probe: probe.write_text('TESTED = False\n')
                return copyfile(source, target)
            args = (destination, campaign, analysis, build['path'], tested['path'], reference['path'],
                    reference['sha256'], audit['path'], audit['sha256'], bundle['path'])
            with patch.object(p, 'ROOT', project), patch.object(p, 'source_paths', return_value=sources), \
                    patch.object(p.shutil, 'copyfile', side_effect=changing_copy):
                with self.assertRaisesRegex(ValueError, 'Tested source changed'): p.prepare(*args)
            self.assertFalse((destination/'preparation.json').exists())
            self.assertFalse(p.read(destination/'preparation-failure.json')['complete'])
            with self.assertRaises(ValueError): p.prepare(*args)


if __name__ == '__main__':
    unittest.main()
