"""Bounded-reference numerical and launch-safety regression tests; no simulation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import prepare_hard_free_line_sensitivity_reference as reference


class SensitivityReferenceTests(unittest.TestCase):
    def test_oracle_matches_archived_hard_and_depletion_quadrature(self):
        for activity, expected in ((0., 16.69111845989073), (2., 18.55934473899714)):
            self.assertAlmostEqual(reference.analytic_reference(activity), expected, places=11)

    def test_child_failure_drains_peer_without_starting_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('activity0', 'activity2'): (root/name).mkdir()
            jobs = [dict(id=name, seed=seed) for name, seed in zip(('activity0', 'activity2'), reference.SEEDS)]
            first = Mock(pid=10); first.poll.return_value = None; first.wait.return_value = 0
            second = Mock(pid=11); second.poll.return_value = 1; second.wait.return_value = 1
            with patch.object(reference, 'capacity', return_value=dict(workers=0, physical_pids=[], private_pid_namespace=False)), \
                 patch.object(reference.subprocess, 'Popen', side_effect=[first, second]) as launch:
                with self.assertRaisesRegex(ValueError, 'Reference child failed'):
                    reference.execute_group(root, jobs, 'physical')
            self.assertEqual(launch.call_count, 2)
            self.assertTrue(all(call.kwargs['env']['PYTHONOPTIMIZE'] == '0' for call in launch.call_args_list))
            first.wait.assert_called_once_with(timeout=None)
            first.terminate.assert_not_called()
            second.terminate.assert_not_called()
            result = reference.read(root/'physical-execution.json')
            self.assertEqual([job['returncode'] for job in result['jobs']], [0, 1])
            self.assertTrue(result['failure_draining'])
            self.assertFalse(any(root.glob('*/audit*')))

    def test_private_namespace_cannot_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(reference, 'capacity', return_value=dict(workers=0, physical_pids=[], private_pid_namespace=True)), \
                 patch.object(reference.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'private sandbox PID namespace'):
                    reference.execute_group(root, [dict(id='activity0', seed=1)], 'physical')
            launch.assert_not_called()

    def test_one_shot_claim_blocks_relaunch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(schema=reference.SCHEMA, target_arms=reference.ARMS,
                        total_unconditional_draws=2*reference.SAMPLES, preparer_sha256=reference.sha(reference.__file__),
                        runtime=reference.runtime(),
                        jobs=[dict(seed=seed) for seed in reference.SEEDS])
            for name, value in [('allocation.json', plan), ('freeze.json', dict(files={})),
                                ('reused-references.json', dict(files={})), ('launch-claim.json', dict(pid=1))]:
                reference.write(root/name, value)
            with patch.object(reference, 'capacity', return_value=dict(workers=0, physical_pids=[], private_pid_namespace=False)), \
                 patch.object(reference.subprocess, 'Popen') as launch:
                with self.assertRaises(FileExistsError): reference.run(root)
            launch.assert_not_called()

    def test_optimized_parent_or_changed_runtime_cannot_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(reference.sys, 'flags', SimpleNamespace(optimize=1)), \
                 patch.object(reference.subprocess, 'Popen') as launch:
                for action in (lambda: reference.prepare(root/'new'), lambda: reference.run(root),
                               lambda: reference.execute_group(root, [], 'physical')):
                    with self.assertRaisesRegex(ValueError, 'Unoptimized Python'):
                        action()
                launch.assert_not_called()
            reference.write(root/'allocation.json', dict(runtime=dict(reference.runtime(), scipy_version='changed')))
            with patch.object(reference.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'runtime differs'):
                    reference.run(root)
                launch.assert_not_called()
            self.assertFalse((root/'launch-claim.json').exists())


if __name__ == '__main__': unittest.main()
