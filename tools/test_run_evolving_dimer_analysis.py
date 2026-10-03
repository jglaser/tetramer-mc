import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_evolving_dimer_analysis as runner


def inventory():
    jobs = [dict(id=k, context_index=c, arm=a, initialization=i, stream=s)
        for k, (c, a, i, s) in enumerate((c, a, i, s) for c in range(4)
            for a in ('local', 'unguided', 'm4') for i in ('source', 'proposal_prepared') for s in range(4))]
    config = dict(jobs=jobs)
    status = dict(complete=True, passed=True, failure_draining=False, active=[], unstarted=[],
        completed=[dict(job=j, success=True, returncode=0, terminal_error=None) for j in jobs])
    return config, status


class CompleteInventoryTests(unittest.TestCase):
    def test_complete_exact_inventory(self):
        config, status = inventory()
        status['completed'].reverse()
        self.assertEqual([r['job']['id'] for r in runner.complete_jobs(config, status)], list(range(96)))

    def test_partial_failure_active_and_duplicate_refused(self):
        config, original = inventory()
        changes = [dict(complete=False), dict(passed=False), dict(failure_draining=True),
                   dict(active=[dict(pid=123)]), dict(unstarted=[config['jobs'][0]]),
                   dict(completed=original['completed'][:-1]),
                   dict(completed=original['completed'][:-1]+[original['completed'][0]])]
        for change in changes:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    runner.complete_jobs(config, dict(original, **change))
        for change in [dict(success=False), dict(returncode=1), dict(terminal_error='bad terminal')]:
            with self.subTest(change=change):
                status = copy.deepcopy(original)
                status['completed'][0].update(change)
                with self.assertRaises(ValueError):
                    runner.complete_jobs(config, status)

    def test_changed_job_and_lost_stream_refused(self):
        config, status = inventory()
        status = copy.deepcopy(status)
        status['completed'][0]['job']['stream'] = 999
        with self.assertRaises(ValueError):
            runner.complete_jobs(config, status)
        config['jobs'][0]['stream'] = 999
        with self.assertRaises(ValueError):
            runner.complete_jobs(config, status)

    def test_partial_prepare_does_not_create_output_or_start_child(self):
        config, status = inventory()
        status['complete'] = False
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/'campaign'
            (base/'dispatch').mkdir(parents=True)
            runner.write(base/'config.json', config)
            runner.write(base/'dispatch/status.json', status)
            output = Path(temp)/'analysis'
            with self.assertRaises(ValueError):
                runner.prepare(base, output)
            self.assertFalse(output.exists())


class ChildLifecycleTests(unittest.TestCase):
    def test_success_records_one_thread_environment_and_reaped_child(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            code = "import os; print(','.join(os.environ[x] for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']))"
            runner.owned_child([sys.executable, '-c', code], temp, path, 5, 5)
            self.assertEqual((path/'output.log').read_text().strip(), '1,1,1,1')
            result = runner.read(path/'exit.json')
            self.assertEqual(result['returncode'], 0)
            self.assertTrue(result['child_started'] and result['child_drained'])
            with self.assertRaises(FileExistsError):
                runner.owned_child([sys.executable, '-c', code], temp, path, 5, 5)

    def test_timeout_kills_reaps_and_retains_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.owned_child([sys.executable, '-u', '-c',
                    "import time; print('partial output', flush=True); time.sleep(30)"], temp, path, 5, .2)
            result = runner.read(path/'exit.json')
            self.assertTrue(result['child_started'] and result['child_drained'])
            self.assertLess(result['returncode'], 0)
            self.assertIn('partial output', (path/'output.log').read_text())

    def test_nonzero_exit_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaisesRegex(ValueError, 'failed with code 7'):
                runner.owned_child([sys.executable, '-c', 'raise SystemExit(7)'], temp, path, 5, 5)
            self.assertEqual(runner.read(path/'exit.json')['returncode'], 7)

    def test_failed_launch_has_terminal_without_started_child(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(FileNotFoundError):
                runner.owned_child([str(path/'does-not-exist')], temp, path, 5, 5)
            result = runner.read(path/'exit.json')
            self.assertFalse(result['child_started'])
            self.assertTrue(result['child_drained'])
            self.assertIsNone(result['returncode'])


if __name__ == '__main__':
    unittest.main()
