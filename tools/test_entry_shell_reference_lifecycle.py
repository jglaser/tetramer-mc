"""Synthetic controller lifecycle checks; never launch a physical executable."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_entry_shell_reference_campaign as runner


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeChildren:
    """Record starts/reaps and optionally leave recognizable synthetic output."""
    def __init__(self, jobs, fail_exit=None, fail_launch=None, save_outputs=False):
        self.jobs = jobs
        self.fail_exit = fail_exit
        self.fail_launch = fail_launch
        self.save_outputs = save_outputs
        self.attempted = []
        self.started = []
        self.waited = []
        self.events = []
        self.peak = 0

    def popen(self, argv, **kwargs):
        index = int(argv[1])
        self.attempted.append(index)
        if index == self.fail_launch:
            raise OSError('synthetic launch failure')
        self.started.append(index)
        self.events.append(('start', index))
        self.peak = max(self.peak, len(self.started) - len(self.waited))
        if self.save_outputs:
            directory = Path(self.jobs[index]['directory'])
            directory.mkdir()
            (directory / 'samples.jsonl').write_text('synthetic saved row ' + str(index) + '\n')
        owner = self

        class Child:
            pid = 2000 + index

            def wait(self):
                owner.waited.append(index)
                owner.events.append(('wait', index))
                return 7 if index == owner.fail_exit else 0

        return Child()


class EntryShellReferenceLifecycle(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def jobs(self, root=None, count=4):
        root = self.root if root is None else root
        (root / 'runs').mkdir(exist_ok=True)
        (root / 'logs').mkdir(exist_ok=True)
        return [dict(id=f'r{i:02d}', seed=134101010 + 1009 * i, samples=8,
                     directory=str(root / 'runs' / f'r{i:02d}'),
                     log=str(root / 'logs' / f'r{i:02d}.log'),
                     command=['synthetic-no-executable', str(i)], status='pending')
                for i in range(count)]

    def campaign(self):
        out = self.root / 'campaign'
        out.mkdir()
        (out / 'provenance').mkdir()
        jobs = self.jobs(out)
        manifest = dict(jobs=copy.deepcopy(jobs), workers=4, physical_activity=.035,
                        lambda_ratio=64., cloud_replicates=2)
        write_json(out / 'manifest.json', manifest)
        protocol = dict(controller_sha256=digest(Path(runner.__file__)),
                        populations=4, samples_per_population=8,
                        total_unconditional_draws=32, maximum_physical_workers=4,
                        manifest_sha256=digest(out / 'manifest.json'))
        write_json(out / 'protocol.json', protocol)
        return out, protocol, jobs

    def assert_terminal_failure(self, out, phase):
        state = read_json(out / 'status.json')
        self.assertIs(state['complete'], False)
        self.assertEqual(state['phase'], phase)
        self.assertIn('finished', state)
        self.assertIn('exception', state)
        self.assertFalse(any(j['status'] in ('pending', 'running') for j in state['jobs']))
        return state

    def assert_refuses_restart(self, out, protocol):
        before = (out / 'status.json').read_bytes()
        # Exercise the real lifecycle check while keeping source/hash binding
        # validation out of these tests. This is deliberately not a freeze test.
        with patch.object(runner, 'validate', return_value=protocol), \
                patch.object(runner, 'execute_jobs') as execute, \
                patch.object(runner.subprocess, 'run') as audit:
            with self.assertRaises(ValueError):
                runner.run(out)
            execute.assert_not_called()
            audit.assert_not_called()
        self.assertEqual((out / 'status.json').read_bytes(), before)

    def test_more_than_four_refused_before_any_launch(self):
        jobs = self.jobs(count=5)
        children = FakeChildren(jobs)
        with self.assertRaises(ValueError):
            runner.execute_jobs(jobs, lambda: None, popen=children.popen)
        self.assertEqual(children.attempted, [])
        self.assertEqual(children.waited, [])
        self.assertFalse(any((self.root / 'logs').iterdir()))

    def test_four_launch_once_before_wait_and_finish(self):
        jobs = self.jobs()
        children = FakeChildren(jobs)
        snapshots = []
        runner.execute_jobs(jobs, lambda: snapshots.append(copy.deepcopy(jobs)), popen=children.popen)
        self.assertEqual(children.attempted, list(range(4)))
        self.assertEqual(children.started, children.waited)
        self.assertEqual(children.peak, 4)
        self.assertEqual(children.events[:4], [('start', i) for i in range(4)])
        self.assertEqual([j['status'] for j in jobs], ['complete'] * 4)
        self.assertEqual([j['returncode'] for j in jobs], [0] * 4)
        self.assertEqual(snapshots[-1], jobs)

    def test_failed_child_drains_all_four_without_retry(self):
        jobs = self.jobs()
        children = FakeChildren(jobs, fail_exit=1)
        snapshots = []
        with self.assertRaises(RuntimeError):
            runner.execute_jobs(jobs, lambda: snapshots.append(copy.deepcopy(jobs)), popen=children.popen)
        self.assertEqual(children.attempted, list(range(4)))
        self.assertEqual(children.started, list(range(4)))
        self.assertEqual(children.waited, list(range(4)))
        self.assertEqual(children.events[:4], [('start', i) for i in range(4)])
        self.assertEqual([j['returncode'] for j in jobs], [0, 7, 0, 0])
        self.assertEqual([j['status'] for j in jobs], ['complete', 'failed', 'complete', 'complete'])
        self.assertEqual(snapshots[-1], jobs)

    def test_launch_failure_drains_started_and_marks_unstarted(self):
        jobs = self.jobs()
        children = FakeChildren(jobs, fail_launch=2)
        snapshots = []
        with self.assertRaisesRegex(OSError, 'synthetic launch'):
            runner.execute_jobs(jobs, lambda: snapshots.append(copy.deepcopy(jobs)), popen=children.popen)
        self.assertEqual(children.attempted, [0, 1, 2])
        self.assertEqual(children.started, [0, 1])
        self.assertEqual(children.waited, [0, 1])
        self.assertEqual([j['status'] for j in jobs], ['complete', 'complete', 'not_started', 'not_started'])
        self.assertTrue(all('returncode' not in j for j in jobs[2:]))
        self.assertEqual(snapshots[-1], jobs)

    def test_run_physical_failure_is_terminal_and_skips_audit(self):
        out, protocol, jobs = self.campaign()
        children = FakeChildren(jobs, fail_exit=1, save_outputs=True)
        execute = runner.execute_jobs

        def physical(records, snapshot):
            return execute(records, snapshot, popen=children.popen)

        with patch.object(runner, 'check', return_value=protocol), \
                patch.object(runner, 'validate', return_value=protocol), \
                patch.object(runner, 'execute_jobs', side_effect=physical), \
                patch.object(runner, 'verify_output') as output, \
                patch.object(runner, 'verify_assessment') as assessment, \
                patch.object(runner.subprocess, 'run') as audit:
            with self.assertRaises(RuntimeError):
                runner.run(out)
            output.assert_not_called()
            assessment.assert_not_called()
            audit.assert_not_called()
        self.assertEqual(children.started, children.waited)
        self.assertEqual(children.attempted, list(range(4)))
        state = self.assert_terminal_failure(out, 'physical_failed')
        self.assertEqual([j['returncode'] for j in state['jobs']], [0, 7, 0, 0])
        for i, job in enumerate(jobs):
            self.assertEqual((Path(job['directory']) / 'samples.jsonl').read_text(), f'synthetic saved row {i}\n')
        self.assert_refuses_restart(out, protocol)

    def test_audit_failure_follows_all_output_validations_and_preserves_outputs(self):
        out, protocol, jobs = self.campaign()
        children = FakeChildren(jobs, save_outputs=True)
        execute = runner.execute_jobs
        sequence = []
        verified = {}

        def physical(records, snapshot):
            execute(records, snapshot, popen=children.popen)
            sequence.append('all_drained')

        def validate(_):
            sequence.append('validate_inputs')
            return protocol

        def output(folder, manifest, job):
            self.assertEqual(children.waited, list(range(4)))
            sequence.append('output:' + job['id'])
            value = dict(samples_sha256=digest(Path(job['directory']) / 'samples.jsonl'))
            verified[job['id']] = value
            return value

        def audit(argv, **kwargs):
            sequence.append('audit')
            self.assertEqual(len(verified), 4)
            self.assertIn(str(out / 'provenance' / 'analyze_latent_region.py'), argv)
            return subprocess.CompletedProcess(argv, 9)

        with patch.object(runner, 'check', return_value=protocol), \
                patch.object(runner, 'validate', side_effect=validate), \
                patch.object(runner, 'execute_jobs', side_effect=physical), \
                patch.object(runner, 'verify_output', side_effect=output), \
                patch.object(runner, 'verify_assessment') as assessment, \
                patch.object(runner.subprocess, 'run', side_effect=audit) as audit_mock:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run(out)
            assessment.assert_not_called()
            self.assertEqual(audit_mock.call_count, 1)
        self.assertEqual(sequence, ['all_drained', 'validate_inputs'] +
                         [f'output:r{i:02d}' for i in range(4)] + ['audit'])
        state = self.assert_terminal_failure(out, 'audit_failed')
        self.assertEqual(state['audit']['returncode'], 9)
        for job in state['jobs']:
            self.assertEqual(job['status'], 'complete')
            self.assertEqual(job['output'], verified[job['id']])
            self.assertEqual(digest(Path(job['directory']) / 'samples.jsonl'), job['output']['samples_sha256'])
        self.assert_refuses_restart(out, protocol)

    def test_invalid_output_prevents_audit_after_all_children_drained(self):
        out, protocol, jobs = self.campaign()
        children = FakeChildren(jobs, save_outputs=True)
        execute = runner.execute_jobs
        verified = []

        def physical(records, snapshot):
            execute(records, snapshot, popen=children.popen)

        def output(folder, manifest, job):
            self.assertEqual(children.waited, list(range(4)))
            verified.append(job['id'])
            if job['id'] == 'r02':
                raise ValueError('synthetic invalid physical output')
            return dict(samples_sha256=digest(Path(job['directory']) / 'samples.jsonl'))

        with patch.object(runner, 'check', return_value=protocol), \
                patch.object(runner, 'validate', return_value=protocol), \
                patch.object(runner, 'execute_jobs', side_effect=physical), \
                patch.object(runner, 'verify_output', side_effect=output), \
                patch.object(runner, 'verify_assessment') as assessment, \
                patch.object(runner.subprocess, 'run') as audit:
            with self.assertRaisesRegex(ValueError, 'synthetic invalid physical output'):
                runner.run(out)
            assessment.assert_not_called()
            audit.assert_not_called()
        self.assertEqual(verified, ['r00', 'r01', 'r02'])
        state = self.assert_terminal_failure(out, 'physical_validation_failed')
        self.assertTrue(all(job['status'] == 'complete' for job in state['jobs']))
        self.assert_refuses_restart(out, protocol)


if __name__ == '__main__':
    unittest.main()
