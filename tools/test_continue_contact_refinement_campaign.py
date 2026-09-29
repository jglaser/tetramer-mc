"""No physical protein jobs: bounded continuation lifecycle and gate controls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import continue_contact_refinement_campaign as controller


class ContinuationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_python_command_preserves_virtualenv_symlink(self):
        base = self.root/'base-python'
        base.write_text('base installation placeholder')
        venv = self.root/'venv/bin'
        venv.mkdir(parents=True)
        link = venv/'python'
        link.symlink_to(base)
        with patch.object(controller.sys, 'executable', str(link)):
            self.assertEqual(controller.python_command(), str(link.absolute()))
            self.assertNotEqual(controller.python_command(), str(link.resolve()))

    def test_wait_requires_valid_completion_and_no_running_workers(self):
        controller.write(self.root/'status.json', dict(jobs=[dict(id='one', status='complete')]))
        controller.write(self.root/'completion.json', dict(complete=True, plan_sha256='frozen'))
        plan = dict(main_campaign=str(self.root), main_plan_sha256='frozen', maximum_wait_seconds=5)
        self.assertTrue(controller.wait_for_main(plan, lambda row:None, pause=lambda seconds:None))
        controller.write(self.root/'completion.json', dict(complete=True, plan_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'Invalid main completion'):
            controller.wait_for_main(plan, lambda row:None, pause=lambda seconds:None)

    def test_failed_allocation_never_gets_rerun(self):
        controller.write(self.root/'status.json', dict(jobs=[dict(id='one', status='failed'),
                                                            dict(id='two', status='not_started')]))
        plan = dict(main_campaign=str(self.root), main_plan_sha256='frozen', maximum_wait_seconds=5)
        self.assertFalse(controller.wait_for_main(plan, lambda row:None, pause=lambda seconds:None))
        self.assertFalse((self.root/'completion.json').exists())

    def test_live_worker_timeout_does_not_release_capacity(self):
        controller.write(self.root/'status.json', dict(jobs=[dict(id='live', status='running', pid=os.getpid())]))
        plan = dict(main_campaign=str(self.root), main_plan_sha256='frozen', maximum_wait_seconds=5)
        ticks = iter([0., 0., 6.])
        with self.assertRaisesRegex(TimeoutError, 'did not release'):
            controller.wait_for_main(plan, lambda row:None, pause=lambda seconds:None,
                                     monotonic=lambda:next(ticks))

    def make_rejection(self):
        options = dict(initial_weight=0, shrinkage=.02, covariance_floor=1e-6,
                       minimum_unique_poses=32, minimum_empirical_rank=6, minimum_ess=0)
        controller.write(self.root/'freeze-rejected.json', dict(model_written=False,
                         all_slots_preserved=True, options=options))
        controller.write(self.root/'fit-metrics.json', dict(independent_slots=2, slots=[{}, {}],
                         exploration_gate_passed=False))

    def test_only_explicit_complete_whole_fit_failure_is_expected(self):
        self.make_rejection()
        self.assertTrue(controller.gate_rejection(self.root, 2))
        with self.assertRaisesRegex(ValueError, 'lost slots'):
            controller.gate_rejection(self.root, 3)
        (self.root/'model.json').write_text('{}')
        self.assertFalse(controller.gate_rejection(self.root, 2))

    def test_changed_gate_is_not_an_allowed_failure(self):
        self.make_rejection()
        rejected = controller.read(self.root/'freeze-rejected.json')
        rejected['options']['minimum_unique_poses'] = 1
        controller.write(self.root/'freeze-rejected.json', rejected)
        with self.assertRaisesRegex(ValueError, 'thresholds changed'):
            controller.gate_rejection(self.root, 2)

    def test_failed_stage_records_receipt_and_cannot_be_retried(self):
        (self.root/'logs').mkdir()
        (self.root/'stages').mkdir()
        stages = controller.Stages(self.root, dict(workers=1), lambda row:None)
        with self.assertRaisesRegex(ValueError, 'failed; no retry'):
            stages.run('failed', [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(controller.read(self.root/'stages/failed.json')['returncode'], 7)
        with self.assertRaises(FileExistsError):
            stages.run('failed', [sys.executable, '-c', 'pass'])

    def test_audit_batch_drains_then_reports_failure(self):
        (self.root/'logs').mkdir()
        (self.root/'stages').mkdir()
        stages = controller.Stages(self.root, dict(workers=2), lambda row:None)
        jobs = [dict(id='bad', directory=str(self.root/'bad'), log=str(self.root/'logs/bad.log'),
                     command=[sys.executable, '-c', 'raise SystemExit(3)'], status='pending'),
                dict(id='drained', directory=str(self.root/'drained'), log=str(self.root/'logs/drained.log'),
                     command=[sys.executable, '-c', f'import time; from pathlib import Path; time.sleep(.1); Path({str(self.root/"drained.txt")!r}).write_text("finished")'],
                     status='pending'),
                dict(id='not-started', directory=str(self.root/'unused'), log=str(self.root/'logs/unused.log'),
                     command=[sys.executable, '-c', 'raise AssertionError("must not run")'], status='pending')]
        with self.assertRaisesRegex(RuntimeError, 'drain children'):
            stages.audit_batch(jobs)
        self.assertTrue((self.root/'drained.txt').exists())
        self.assertEqual([job['status'] for job in jobs], ['failed', 'complete', 'not_started'])
        self.assertEqual(controller.read(self.root/'stages/geometry-audits.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
