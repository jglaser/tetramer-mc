"""Procedural interruption/failure tests; fake children never generate poses/clouds."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import run_native_class_line_physical_toy as runner


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, mode='complete'):
    save(root / 'mode.json', dict(mode=mode))
    executable = root / 'fake-normalizer'
    script = ('#!' + sys.executable + '\n' +
        'import json, os, pathlib, subprocess, sys, time\n'
        f'root=pathlib.Path({str(root)!r})\n'
        'value=lambda flag: sys.argv[sys.argv.index(flag)+1]\n'
        'mode=json.loads((root/"mode.json").read_text())["mode"]\n'
        'physical="--config" in sys.argv\n'
        'audit="--synthetic" in sys.argv\n'
        'phase="physical" if physical else "audit" if audit else "analysis"\n'
        'destination=pathlib.Path(value("--out"))\n'
        'if physical: destination=destination/"summary.json"\n'
        'if physical and mode=="sleep": time.sleep(30)\n'
        'if physical and mode=="descendant":\n'
        ' child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\n'
        ' (root/"descendant.json").write_text(json.dumps(dict(pid=child.pid)))\n'
        ' time.sleep(30)\n'
        'if mode==phase+"-exit": sys.exit(7)\n'
        'if mode==phase+"-missing": sys.exit(0)\n'
        'destination.parent.mkdir(parents=True,exist_ok=True)\n'
        'destination.write_text(json.dumps(dict(complete=mode!=phase+"-incomplete",passed=mode!=phase+"-failed")))\n')
    executable.write_text(script)
    executable.chmod(0o755)
    scripts = root / 'tools'
    scripts.mkdir()
    for name in ('native_class_line_physical_reference.py',
                 'analyze_native_class_line_physical_toy.py'):
        (scripts / name).write_text(script)
    jobs = [dict(id='toy-' + str(i), out=str(root / ('population-' + str(i))),
                 config='unused', region='unused', guide='unused', samples=0,
                 seed=i, lambda_ratio=64, cpu_limit_seconds=10, wall_limit_seconds=30)
            for i in range(2)]
    save(root / 'allocation.json', dict(jobs=jobs))
    files = [root / 'allocation.json', root / 'mode.json', executable, *scripts.iterdir()]
    save(root / 'execution-binding.json', dict(
        maximum_workers=1, allocation_sha256=runner.sha(root / 'allocation.json'),
        files={str(p): runner.sha(p) for p in files}, executable=str(executable),
        executable_sha256=runner.sha(executable), python=sys.executable, tools=str(scripts)))


class ToyControllerTest(unittest.TestCase):
    def run_quiet(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(root)

    def assert_failed_and_reaped(self, root, phase='physical'):
        failure = runner.read(root / 'failure.json')
        self.assertFalse(failure['complete'])
        self.assertFalse(runner.read(root / 'completion.json')['passed'])
        self.assertTrue(failure['child_drained'])
        self.assertEqual(failure['retries'], 0)
        self.assertEqual(failure['protein_queries'], 0)
        if phase != 'analysis':
            self.assertFalse((root / 'population-1').exists())
        terminal = runner.read(root / 'logs' / ('toy-0-' + phase + '.terminal.json'))
        self.assertTrue(terminal['child_drained'])
        with self.assertRaises(ProcessLookupError):
            os.kill(terminal['child_pid'], 0)
        before = (root / 'failure.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_quiet(root)
        self.assertEqual(before, (root / 'failure.json').read_bytes())
        return failure, terminal

    def test_success_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            self.run_quiet(root)
            self.assertTrue(runner.read(root / 'completion.json')['passed'])
            self.assertEqual(len(runner.read(root / 'execution-summary.json')['completed']), 4)
            self.assertEqual(len(list((root / 'logs').glob('*.terminal.json'))), 5)
            self.assertFalse((root / 'failure.json').exists())
            with self.assertRaises(FileExistsError):
                self.run_quiet(root)

    def test_nonzero_missing_incomplete_failed_receipts_stop(self):
        for mode in ('physical-exit', 'physical-missing', 'physical-incomplete',
                     'audit-failed', 'analysis-failed'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root, mode)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.run_quiet(root)
                phase = mode.split('-')[0]
                if phase == 'analysis':
                    failure = runner.read(root / 'failure.json')
                    self.assertTrue(failure['child_drained'])
                    self.assertEqual(len(failure['completed']), 4)
                    self.assertFalse(runner.read(root / 'completion.json')['passed'])
                else:
                    failure, _ = self.assert_failed_and_reaped(root, phase)
                    self.assertEqual(len(failure['completed']), int(phase == 'audit'))

    def test_keyboard_systemexit_timeout_and_sigterm_drain(self):
        original_wait = subprocess.Popen.wait
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        for kind in ('keyboard', 'systemexit', 'timeout', 'sigterm'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root, 'sleep')
                def interrupted_wait(child, timeout=None):
                    if timeout is not None and not getattr(child, '_injected', False):
                        child._injected = True
                        if kind == 'keyboard':
                            raise KeyboardInterrupt('synthetic interruption')
                        if kind == 'systemexit':
                            raise SystemExit('synthetic exit')
                        if kind == 'sigterm':
                            os.kill(os.getpid(), signal.SIGTERM)
                        return original_wait(child, timeout=.02)
                    return original_wait(child, timeout=timeout)
                expected = dict(keyboard=KeyboardInterrupt, systemexit=SystemExit,
                                timeout=subprocess.TimeoutExpired, sigterm=SystemExit)[kind]
                with mock.patch.object(subprocess.Popen, 'wait', interrupted_wait):
                    with self.assertRaises(expected):
                        self.run_quiet(root)
                _, terminal = self.assert_failed_and_reaped(root)
                self.assertEqual(terminal['timeout'], kind == 'timeout')
                self.assertEqual(signal.getsignal(signal.SIGTERM), previous_sigterm)

    def test_exception_while_recording_launch_still_drains(self):
        original_write = runner.write
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, 'sleep')
            def failed_write(path, value):
                if str(path).endswith('.launch.json'):
                    raise RuntimeError('synthetic serialization failure after Popen')
                return original_write(path, value)
            with mock.patch.object(runner, 'write', failed_write):
                with self.assertRaisesRegex(RuntimeError, 'serialization failure'):
                    self.run_quiet(root)
            failure, terminal = self.assert_failed_and_reaped(root)
            self.assertIn('serialization failure', failure['error'])
            self.assertTrue(terminal['child_started'])
            self.assertFalse((root / 'logs/toy-0-physical.launch.json').exists())

    def test_sigterm_inside_popen_before_handle_assignment_is_drained(self):
        original_popen = subprocess.Popen
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, 'sleep')
            def interrupted_popen(*args, **kwargs):
                child = original_popen(*args, **kwargs)
                os.kill(os.getpid(), signal.SIGTERM)
                return child
            with mock.patch.object(subprocess, 'Popen', interrupted_popen):
                with self.assertRaisesRegex(SystemExit, 'received signal'):
                    self.run_quiet(root)
            self.assert_failed_and_reaped(root)
            self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), prior_mask)

    def test_binding_changed_during_child_is_rejected_after_child(self):
        original_wait = subprocess.Popen.wait
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            def tampered_wait(child, timeout=None):
                code = original_wait(child, timeout=timeout)
                if timeout is not None:
                    path = root / 'execution-binding.json'
                    path.write_text(path.read_text() + '\n')
                return code
            with mock.patch.object(subprocess.Popen, 'wait', tampered_wait):
                with self.assertRaisesRegex(ValueError, 'Frozen execution binding changed'):
                    self.run_quiet(root)
            failure = runner.read(root / 'failure.json')
            self.assertTrue(failure['child_drained'])
            self.assertEqual(failure['completed'], [])
            self.assertFalse((root / 'population-1').exists())
            self.assertFalse((root / 'execution-summary.json').exists())

    def test_timeout_drains_descendant_group(self):
        original_wait = subprocess.Popen.wait
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, 'descendant')
            def interrupted_wait(child, timeout=None):
                if timeout is not None:
                    deadline = time.monotonic() + 5
                    while not (root / 'descendant.json').exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError('fixture did not launch descendant')
                        time.sleep(.01)
                    return original_wait(child, timeout=.02)
                return original_wait(child, timeout=timeout)
            with mock.patch.object(subprocess.Popen, 'wait', interrupted_wait):
                with self.assertRaises(subprocess.TimeoutExpired):
                    self.run_quiet(root)
            self.assert_failed_and_reaped(root)
            pid = runner.read(root / 'descendant.json')['pid']
            status = Path(f'/proc/{pid}/stat')
            deadline = time.monotonic() + 2
            while status.exists():
                # An orphan zombie is dead and awaits the OS subreaper, which
                # the controller cannot reap; it is not running scientific work.
                if status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    break
                self.assertLess(time.monotonic(), deadline, 'descendant still alive')
                time.sleep(.01)

    def test_changed_input_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            (root / 'fake-normalizer').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen execution file changed'):
                self.run_quiet(root)
            self.assertFalse((root / 'execution-claim.json').exists())


if __name__ == '__main__':
    unittest.main()
