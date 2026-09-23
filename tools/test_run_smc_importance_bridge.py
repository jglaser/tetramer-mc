"""Synthetic workflow tests: no protein rows, classifiers, audits, or samplers."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_smc_importance_bridge as bridge
from analyze_r4_smc_control import read, sha, write


def freeze_tree(root):
    write(root / 'freeze.json', dict(files={str(p.relative_to(root)): sha(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'freeze.json'}))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.dependency = self.root / 'dependency'
        self.dependency.mkdir(); self.importance = self.root / 'importance'; self.importance.mkdir()
        campaign = self.root / 'campaign'; (campaign / 'bank/provenance').mkdir(parents=True)
        shape = campaign / 'shape.json'; write(shape, {})
        region = campaign / 'bank/provenance/region.json'; write(region, {})
        write(campaign / 'bank/provenance/config.json', dict(shape=str(shape)))
        write(campaign / 'status.json', dict(complete=True))
        (self.importance / 'provenance').mkdir()
        protocol = self.importance / 'provenance/campaign-protocol.json'; write(protocol, {})
        write(self.importance / 'analysis.json', dict(schema='contact-confirmation-comparison-v1',
            complete=True, convergence=dict(confirmation_passed=False), campaign=str(campaign),
            protocol_sha256=sha(protocol), status_sha256=sha(campaign / 'status.json'),
            region_sha256=sha(region), shape_sha256=sha(shape)))
        self.importance_sha = sha(self.importance / 'analysis.json')
        write(self.importance / 'status.json', dict(complete=True, analysis_sha256=self.importance_sha))
        freeze_tree(self.importance)
        paths = [str(self.dependency / (arm + '-analysis')) for arm in bridge.ARMS]
        self.source_plan = dict(schema=bridge.DEPENDENCY_SCHEMA, python=sys.executable,
            repository=str(self.root), controls=[dict(protocol='/inert/' + arm, protocol_sha256=arm)
                for arm in bridge.ARMS], analysis_workers=4, stage_stride=16, analysis_paths=paths,
            jobs=[dict(id=arm + '-r' + str(i), arm=arm, seed=100 * a + i,
                command=['inert-physical-command', arm, str(i)], directory='/inert/' + arm + str(i),
                log='/inert/' + arm + str(i) + '.log')
                for i in range(4) for a, arm in enumerate(bridge.ARMS)])
        write(self.dependency / 'plan.json', self.source_plan); freeze_tree(self.dependency)
        self.dep_sha = sha(self.dependency / 'plan.json')
        self.dep = dict(directory=str(self.dependency), plan_sha256=self.dep_sha,
            pid=os.getpid(), process_birth=bridge.process_token(os.getpid()),
            contract=bridge.dependency_contract(self.dependency, self.source_plan))
        self.state = dict(schema=bridge.DEPENDENCY_SCHEMA, plan_sha256=self.dep_sha, pid=os.getpid(),
            complete=False, phase='physical', jobs=[dict(job, status='running', returncode=None)
                for job in self.source_plan['jobs']], analyses=[])
        write(self.dependency / 'status.json', self.state)
        self.out = self.root / 'bridge'

    def freeze(self):
        self.plan = bridge.freeze(self.out, self.dependency, self.dep_sha, os.getpid(),
            self.dep['process_birth'], self.importance / 'analysis.json', self.importance_sha)
        self.plan_sha = sha(self.out / 'plan.json')
        return self.plan

    def complete_dependency(self):
        analyses = {}
        for arm in bridge.ARMS:
            target = self.dependency / (arm + '-analysis'); target.mkdir()
            write(target / 'analysis.json', dict(schema='smc-r4-control-analysis-v1',
                complete=True, protocol_sha256=arm))
            digest = sha(target / 'analysis.json'); analyses[str(target / 'analysis.json')] = digest
            write(target / 'status.json', dict(complete=True, analysis_sha256=digest)); freeze_tree(target)
        target = self.dependency / 'comparison'; target.mkdir()
        write(target / 'comparison.json', dict(schema='smc-r4-control-comparison-v1',
            source_populations_kept_separate=True, source_sha256=analyses, primary_mass_agreement=False))
        freeze_tree(target)
        self.state.update(complete=True, phase='complete', confirmation_passed=False,
            jobs=[dict(job, status='complete', returncode=0) for job in self.source_plan['jobs']],
            analyses=[dict(command=command, returncode=0) for command in self.dep['contract']['analyses']],
            comparison=dict(command=self.dep['contract']['comparison'], returncode=0))
        write(self.dependency / 'status.json', self.state)

    def fake_comparator(self, command, **kwargs):
        """Emit a minimal frozen report, checking that no physical command is used."""
        self.assertEqual(Path(command[3]).name, 'compare_r4_smc_importance.py')
        self.assertEqual(command[1:3], ['-B', '-E'])
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
        target = Path(command[command.index('--out') + 1]); target.mkdir()
        smc = command[command.index('--smc') + 1]
        if target.name == 'narrow-comparison':
            prior = read(self.out / 'status.json')['steps'][0]
            self.assertEqual(prior['returncode'], 0); self.assertIn('analysis_sha256', prior)
        write(target / 'analysis.json', dict(schema='r4-smc-importance-comparison-v1', complete=True,
            populations_pooled=False,
            inputs=dict(smc=smc, importance=str(self.importance / 'analysis.json')),
            input_sha256={smc: sha(smc), str(self.importance / 'analysis.json'): self.importance_sha},
            source_sha256={str(self.out / 'common' / name): digest
                for name, digest in self.plan['comparator_sources'].items()},
            arms=dict(bank=dict(primary_Qz_mass_agreement=False, significant_primary_strata_agree=False))))
        freeze_tree(target)
        return subprocess.CompletedProcess(command, 0)

    def run_bridge(self, launch=None):
        with patch.object(bridge, '__file__', str(self.out / 'common/run_smc_importance_bridge.py')):
            with patch.object(bridge.subprocess, 'run', side_effect=launch or self.fake_comparator) as dispatch:
                result = bridge.run(self.out, self.plan_sha)
        return result, dispatch

    def test_waiting_and_original_process_required(self):
        self.assertEqual(bridge.prerequisite_state(self.state, self.dep, True), 'waiting')
        with self.assertRaisesRegex(ValueError, 'original process is absent'):
            bridge.prerequisite_state(self.state, self.dep, False)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            bridge.prerequisite_state(dict(self.state, pid=os.getpid() + 1), self.dep, True)
        self.assertIsNone(bridge.process_token(2147483647))

    def test_terminal_ledger_required_but_scientific_pass_is_not(self):
        self.complete_dependency()
        self.assertEqual(bridge.prerequisite_state(self.state, self.dep, False), 'ready')
        variants = [dict(self.state, comparison=None), dict(self.state, analyses=[]),
            dict(self.state, error='failure'), dict(self.state, phase='failed')]
        changed = copy.deepcopy(self.state); changed['jobs'][0]['command'] = ['changed']; variants.append(changed)
        changed = copy.deepcopy(self.state); changed['analyses'][0]['returncode'] = 1; variants.append(changed)
        changed = copy.deepcopy(self.state); changed['comparison']['command'] = ['changed']; variants.append(changed)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(ValueError):
                bridge.prerequisite_state(state, self.dep, True)

    def test_auditing_phase_retains_only_successful_completed_entries(self):
        self.state.update(phase='auditing_and_classifying',
            jobs=[dict(job, status='complete', returncode=0) for job in self.source_plan['jobs']],
            analyses=[dict(command=self.dep['contract']['analyses'][0], returncode=0)])
        self.assertEqual(bridge.prerequisite_state(self.state, self.dep, True), 'waiting')
        with self.assertRaisesRegex(ValueError, 'original process is absent'):
            bridge.prerequisite_state(self.state, self.dep, False)

    def test_freeze_pins_sources_commands_runtime_and_hashes(self):
        plan = self.freeze()
        self.assertEqual(plan['runtime'], bridge.runtime())
        self.assertEqual(plan['dependency']['process_birth'], self.dep['process_birth'])
        self.assertEqual(plan['input_bindings'][str(self.importance / 'analysis.json')], self.importance_sha)
        self.assertEqual(plan['input_bindings'][str(self.dependency / 'plan.json')], self.dep_sha)
        self.assertIn('compare_r4_smc_importance.py', plan['comparator_sources'])
        self.assertEqual(plan['physical_workers'], 0)
        self.assertEqual([step['id'] for step in plan['steps']], ['broad', 'narrow'])
        self.assertFalse((self.out / 'status.json').exists())

    def test_wrong_expected_hash_or_birth_refused_before_freeze(self):
        arguments = [self.out, self.dependency, self.dep_sha, os.getpid(), self.dep['process_birth'],
            self.importance / 'analysis.json', self.importance_sha]
        for index, bad in [(2, '0' * 64), (4, 'wrong-birth'), (6, '0' * 64)]:
            args = arguments.copy(); args[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError): bridge.freeze(*args)
            self.assertFalse(self.out.exists())

    def test_successful_sequential_comparisons_preserve_disagreements(self):
        self.freeze(); self.complete_dependency()
        result, dispatch = self.run_bridge()
        self.assertTrue(result['complete']); self.assertEqual(dispatch.call_count, 2)
        self.assertEqual([item['id'] for item in result['steps']], ['broad', 'narrow'])
        self.assertFalse(result['steps'][0]['arms']['bank']['primary_Qz_mass_agreement'])
        before = (self.out / 'status.json').read_bytes()
        with self.assertRaises(FileExistsError): self.run_bridge()
        self.assertEqual(before, (self.out / 'status.json').read_bytes())

    def test_wait_only_dispatches_after_complete_terminal_artifacts(self):
        self.freeze()
        with patch.object(bridge.time, 'sleep', side_effect=lambda _: self.complete_dependency()) as wait:
            result, dispatch = self.run_bridge()
        wait.assert_called_once_with(30); self.assertTrue(result['complete']); self.assertEqual(dispatch.call_count, 2)

    def test_dead_or_recycled_source_process_fails_without_dispatch(self):
        self.freeze()
        with patch.object(bridge, 'process_token', return_value='recycled'):
            with self.assertRaisesRegex(ValueError, 'original process is absent'): self.run_bridge()
        self.assertEqual(read(self.out / 'status.json')['phase'], 'failed')
        self.assertFalse((self.out / 'broad.log').exists())

    def test_first_failure_stops_and_preserves_partial_output(self):
        self.freeze(); self.complete_dependency(); calls = []
        def fail(command, **kwargs):
            calls.append(command); target = Path(command[-1]); target.mkdir()
            (target / 'partial').write_text('preserve me')
            return subprocess.CompletedProcess(command, 7)
        with self.assertRaises(subprocess.CalledProcessError): self.run_bridge(fail)
        state = read(self.out / 'status.json')
        self.assertEqual(len(calls), 1); self.assertEqual(state['phase'], 'failed')
        self.assertEqual(state['steps'][0]['returncode'], 7)
        self.assertEqual((self.out / 'broad-comparison/partial').read_text(), 'preserve me')
        self.assertFalse((self.out / 'narrow-comparison').exists())

    def test_existing_output_and_changed_source_refused(self):
        self.freeze(); self.complete_dependency()
        (self.out / 'broad-comparison').mkdir()
        with self.assertRaisesRegex(ValueError, 'Existing comparison output'): self.run_bridge()
        self.assertEqual(read(self.out / 'status.json')['phase'], 'failed')
        self.assertFalse((self.out / 'broad.log').exists())

    def test_tampered_frozen_source_reports_failure(self):
        self.freeze()
        with (self.out / 'common/compare_r4_smc_importance.py').open('a') as stream: stream.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'): self.run_bridge()
        self.assertEqual(read(self.out / 'status.json')['phase'], 'failed')

    def test_changed_completed_input_during_wait_stops_before_dispatch(self):
        self.freeze()
        def finish_and_change(_):
            self.complete_dependency()
            with (self.importance / 'analysis.json').open('a') as stream: stream.write('\n')
        with patch.object(bridge.time, 'sleep', side_effect=finish_and_change):
            with self.assertRaisesRegex(ValueError, 'Source changed'): self.run_bridge()
        self.assertFalse((self.out / 'broad.log').exists())

    def test_wrong_runtime_reports_failure(self):
        self.freeze()
        with patch.object(bridge, 'runtime', return_value={}):
            with self.assertRaisesRegex(ValueError, 'runtime differs'): self.run_bridge()
        self.assertEqual(read(self.out / 'status.json')['phase'], 'failed')

    def test_wrong_plan_checksum_reports_failure(self):
        self.freeze(); self.plan_sha = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'): self.run_bridge()
        self.assertEqual(read(self.out / 'status.json')['phase'], 'failed')

    def test_corrupt_terminal_binding_prevents_comparison(self):
        self.freeze(); self.complete_dependency()
        target = self.dependency / 'broad-analysis'
        write(target / 'status.json', dict(complete=True, analysis_sha256='wrong')); freeze_tree(target)
        with self.assertRaisesRegex(ValueError, 'terminal analysis binding differs'): self.run_bridge()
        self.assertFalse((self.out / 'broad.log').exists())

    def test_terminal_comparison_must_keep_source_populations_separate(self):
        self.freeze(); self.complete_dependency()
        target = self.dependency / 'comparison'; value = read(target / 'comparison.json')
        value['source_populations_kept_separate'] = False
        write(target / 'comparison.json', value); freeze_tree(target)
        with self.assertRaisesRegex(ValueError, 'does not bind both completed analyses'): self.run_bridge()
        self.assertFalse((self.out / 'broad.log').exists())

    def test_bad_comparator_result_preserved_and_second_not_launched(self):
        self.freeze(); self.complete_dependency(); calls = []
        def bad_result(command, **kwargs):
            calls.append(command); result = self.fake_comparator(command, **kwargs)
            target = Path(command[-1]); value = read(target / 'analysis.json')
            value['input_sha256'][value['inputs']['importance']] = 'changed'
            write(target / 'analysis.json', value); freeze_tree(target)
            return result
        with self.assertRaisesRegex(ValueError, 'output input bindings differ'): self.run_bridge(bad_result)
        self.assertEqual(len(calls), 1); self.assertTrue((self.out / 'broad-comparison/analysis.json').exists())
        self.assertFalse((self.out / 'narrow-comparison').exists())


if __name__ == '__main__': unittest.main()
