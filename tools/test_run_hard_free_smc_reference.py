"""Synthetic reference-runner tests; every child launch is mocked."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_hard_free_smc_reference as runner


class ReferenceMathTests(unittest.TestCase):
    def test_four_independent_linear_masses_retain_zero(self):
        result = runner.check_mean([1., 3., 0., 4.], 2.)
        self.assertEqual(result['population_masses'], [1., 3., 0., 4.])
        self.assertEqual(result['mean'], 2.)
        self.assertAlmostEqual(result['standard_error'], math.sqrt(10./12))
        self.assertTrue(result['passed'])
        with self.assertRaises(ValueError): runner.check_mean([1., 3., 4.], 2.)

    def test_terminal_indicators_multiply_Z_and_boundary_is_unbound(self):
        summary = dict(zero_estimate=False, log_Z=math.log(8.), terminal_particles=[
            dict(pose=dict(position=[radius, 0., 0.])) for radius in [.7, 1., 1.399, 1.4]])
        masses = runner.population_masses(summary)
        for name, expected in [('total', 8.), ('contact', 6.), ('unbound', 2.)]:
            self.assertAlmostEqual(masses[name], expected)
        self.assertEqual(runner.population_masses(dict(zero_estimate=True, log_Z=None,
            terminal_particles=[])), dict(total=0., contact=0., unbound=0.))

    def test_failed_tolerance_is_a_reported_scientific_result(self):
        result = runner.check_mean([1.]*4, 2.)
        self.assertFalse(result['passed']); self.assertEqual(result['reference'], 2.)
        self.assertEqual(result['standard_error'], 0.)
        self.assertTrue(runner.check_mean([1.]*4, 1.+.5e-6)['passed'])

    def test_uniform_parity_exact_fields_and_distinct_numeric_tolerances(self):
        left = dict(pose=dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.]),
            parents=[0, 2], accepted=True, count=2, log_Z=3., log_physical_density=-2.,
            log_physical_jacobian=.4, latent=[.2, .3], initial_weight_ESS=12.,
            terminal_particles=[dict(pose=dict(position=[1., 0., 0.]))])
        right = copy.deepcopy(left)
        for field in ['log_Z', 'log_physical_density', 'log_physical_jacobian', 'initial_weight_ESS']:
            right[field] += 1e-10
        right['latent'][0] += 1e-8
        runner.parity_record(left, right)
        changes = [('pose', dict(left['pose'], position=[1.+1e-12, 0., 0.])),
            ('parents', [0, 1]), ('accepted', False), ('count', 3),
            ('log_Z', 3.+4e-10), ('log_physical_density', -2.+4e-10),
            ('log_physical_jacobian', .4+4e-10), ('latent', [.2+4e-8, .3]),
            ('initial_weight_ESS', 12.+4e-10),
            ('terminal_particles', [dict(pose=dict(position=[1.+1e-12, 0., 0.]))])]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.parity_record(left, dict(left, **{key: value}))


class ReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.prep = self.root/'preparation'; self.prep.mkdir()
        self.out = self.root/'execution'; self.out.mkdir()
        self.ep = dict(preparation=str(self.prep), runtime={})
        self.plan = dict(jobs=[dict(id=str(i), directory=str(self.prep/str(i)),
            argv=['inert-never-execute', str(i)], guide=None) for i in range(2)])
        for name, result in [('validate', (self.ep, self.plan)), ('runtime', {}), ('process_token', 'mock-birth')]:
            mocked = patch.object(runner, name, return_value=result); mocked.start(); self.addCleanup(mocked.stop)
        self.capacity = dict(private_pid_namespace=False, physical_pids=[], workers=0)

    def test_unobservable_or_full_capacity_refuses_without_claim_status_or_child(self):
        for change in [dict(private_pid_namespace=True), dict(physical_pids=list(range(8))), dict(workers=32)]:
            with self.subTest(change=change), patch.object(runner, 'capacity', return_value=dict(self.capacity, **change)), \
                    patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(ValueError): runner.run(self.out, 'frozen')
                launch.assert_not_called()
                self.assertFalse((self.prep/'reference-launch-claim.json').exists())
                self.assertFalse((self.out/'status.json').exists())

    def test_first_child_failure_halts_and_global_claim_prevents_every_retry(self):
        with patch.object(runner, 'capacity', return_value=self.capacity), \
                patch.object(runner.subprocess, 'Popen') as launch, patch.object(runner, 'check_output') as accept:
            launch.return_value.pid = 12345; launch.return_value.wait.return_value = 7
            with self.assertRaisesRegex(ValueError, 'Child failed'): runner.run(self.out, 'frozen')
            state = runner.read(self.out/'status.json'); self.assertEqual(state['phase'], 'failed')
            self.assertEqual(len(state['jobs']), 1); self.assertEqual(state['jobs'][0]['steps'][0]['returncode'], 7)
            claim = (self.prep/'reference-launch-claim.json').read_bytes()
            with self.assertRaises(ValueError): runner.run(self.out, 'frozen')
            other = self.root/'other-execution'; other.mkdir()
            with self.assertRaises(FileExistsError): runner.run(other, 'frozen')
            self.assertEqual(claim, (self.prep/'reference-launch-claim.json').read_bytes())
            self.assertFalse((other/'status.json').exists()); launch.assert_called_once(); accept.assert_not_called()

    def test_guided_audit_failure_prevents_next_physical_job(self):
        self.plan['jobs'][0]['guide'] = 'guide.json'
        with patch.object(runner, 'capacity', return_value=self.capacity), \
                patch.object(runner.subprocess, 'Popen') as launch, \
                patch.object(runner, 'check_output', return_value=({}, dict(bound=True))) as accept:
            launch.return_value.pid = 12345; launch.return_value.wait.side_effect = [0, 9]
            with self.assertRaisesRegex(ValueError, 'Child failed'): runner.run(self.out, 'frozen')
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(launch.call_args_list[0].args[0], self.plan['jobs'][0]['argv'])
            self.assertEqual(Path(launch.call_args_list[1].args[0][1]).name, 'audit_hard_free_smc.py')
            accept.assert_called_once_with(self.prep, self.plan, self.plan['jobs'][0], None)
            state = runner.read(self.out/'status.json'); self.assertEqual(state['phase'], 'failed')
            self.assertEqual(len(state['jobs']), 1)
            self.assertEqual([(s['kind'], s['returncode']) for s in state['jobs'][0]['steps']], [('physical', 0), ('audit', 9)])
            self.assertFalse((self.out/'1-physical.log').exists())

    def test_sigterm_during_wait_finishes_child_then_prevents_audit_and_next_job(self):
        self.plan['jobs'][0]['guide'] = 'guide.json'; handlers = {}; finished = []
        def register(sig, callback):
            previous = handlers.get(sig, runner.signal.SIG_DFL); handlers[sig] = callback; return previous
        def finish_child():
            handlers[runner.signal.SIGTERM](runner.signal.SIGTERM, None)
            self.assertEqual(runner.read(self.out/'status.json')['jobs'][0]['steps'][0]['phase'], 'running')
            finished.append(True); return 0
        with patch.object(runner, 'capacity', return_value=self.capacity), \
                patch.object(runner.signal, 'signal', side_effect=register), \
                patch.object(runner.subprocess, 'Popen') as launch, patch.object(runner, 'check_output') as accept:
            launch.return_value.pid = 12345; launch.return_value.wait.side_effect = finish_child
            with self.assertRaisesRegex(ValueError, 'active child finished'): runner.run(self.out, 'frozen')
            launch.assert_called_once(); launch.return_value.wait.assert_called_once(); accept.assert_not_called()
            self.assertEqual(finished, [True]); state = runner.read(self.out/'status.json')
            self.assertEqual(state['phase'], 'failed'); self.assertFalse(state['complete'])
            self.assertEqual(len(state['jobs']), 1); self.assertEqual(len(state['jobs'][0]['steps']), 1)
            self.assertEqual(state['jobs'][0]['steps'][0]['returncode'], 0)
            self.assertFalse((self.out/'0-audit.log').exists()); self.assertFalse((self.out/'1-physical.log').exists())
            self.assertTrue(all(callback == runner.signal.SIG_DFL for callback in handlers.values()))


class ReferenceReceiptTests(unittest.TestCase):
    def test_accepted_receipt_binds_outputs_sources_and_audit_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            prep = Path(temporary); inputs = prep/'inputs/zero'; inputs.mkdir(parents=True)
            common = prep/'common'; common.mkdir(); root = prep/'population'; (root/'provenance').mkdir(parents=True)
            job = dict(id='zero', fixture='zero', directory=str(root), guide='guide.json', initial_draws=4,
                population=2, seed=123, stages=1, sweeps=0, purpose='zero-hit')
            for name in ['config', 'region', 'shape', 'guide']: runner.write(inputs/(name+'.json'), dict(kind=name))
            runner.write(common/'source-bundle.json', dict(kind='source'))
            (common/'latent-region-smc').write_bytes(b'inert binary')
            source = common/'audit_hard_free_smc.py'; source.write_text('# inert audit source\n')
            plan = dict(cloud_replicates=2, lambda_ratio=8., binary_sha256=runner.sha(common/'latent-region-smc'),
                source_bundle_sha256=runner.sha(common/'source-bundle.json'),
                files_sha256={'common/'+source.name: runner.sha(source)})
            options = dict(config=str(inputs/'config.json'), region=str(inputs/'region.json'), out=str(root),
                initial_reference_region=None, initial_current_probability=1., initial_draws=4, population=2,
                seed=123, bridge='proposal_density', schedule=[0., 1.], sweeps_per_stage=0,
                cloud_replicates=2, lambda_ratio=8.)
            manifest = dict(schema='latent-region-smc-hard-free-initial-guide-v1', options=options,
                executable_sha256=plan['binary_sha256'], initial_guide=dict(path=str(inputs/'guide.json'),
                sha256=runner.sha(inputs/'guide.json')))
            for name in ['config', 'region', 'shape', 'source-bundle', 'initial-guide']:
                original = (common/'source-bundle.json' if name == 'source-bundle' else
                    inputs/('guide.json' if name == 'initial-guide' else name+'.json'))
                (root/'provenance'/(name+'.json')).write_bytes(original.read_bytes())
                if name != 'initial-guide': manifest[name.replace('-', '_')+'_sha256'] = runner.sha(original)
            runner.write(root/'manifest.json', manifest)
            summary = dict(schema='latent-region-smc-hard-free-initial-guide-summary-v1', complete=True,
                manifest_sha256=runner.sha(root/'manifest.json'), initial_draws=4, initial_hits=0,
                completed_stage=0, zero_estimate=True, log_Z=None, terminal_particles=[])
            for name in ['initialization', 'stages', 'attempts']:
                path = root/(name+'.jsonl'); path.write_text(''); summary[name+'_sha256'] = runner.sha(path)
            runner.write(root/'summary.json', summary)
            runner.write(root/'status.json', dict(complete=True, phase='complete', completed_stage=0,
                summary_sha256=runner.sha(root/'summary.json')))
            _, receipt = runner.check_output(prep, plan, job)
            ledger = dict(receipt['input_sha256'], **{str(common/'latent-region-smc'): plan['binary_sha256'],
                str(source): runner.sha(source)})
            audit = dict(schema='hard-free-initial-guide-smc-independent-audit-v1', complete=True, directory=str(root),
                initial_draws=4, initial_hits=0, completed_stage=0, log_Z=None, terminal_particles=0, input_sha256=ledger)
            audit_path = prep/'audit.json'; runner.write(audit_path, audit)
            _, accepted = runner.check_output(prep, plan, job, audit_path)
            self.assertTrue(accepted['independently_audited'])
            self.assertEqual(accepted['input_sha256'], dict(ledger, **{str(audit_path): runner.sha(audit_path)}))
            for i, missing in enumerate([root/'initialization.jsonl', common/'latent-region-smc', source]):
                bad = prep/f'bad-{i}.json'; altered = dict(ledger); del altered[str(missing)]
                runner.write(bad, dict(audit, input_sha256=altered))
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'required bound'):
                    runner.check_output(prep, plan, job, bad)
            (root/'stages.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Output binding differs'):
                runner.check_output(prep, plan, job, audit_path)


if __name__ == '__main__': unittest.main()
