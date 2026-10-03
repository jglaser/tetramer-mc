import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from prepare_native_class_line_probe import jobs
import run_native_class_line_probe as runner
from run_native_class_line_probe import canonical_jobs, read, run, sha, verify


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, mode='complete'):
    """Canonical argv and frozen artifacts, with executable toy children only."""
    allocation = dict(jobs=jobs(), fresh_draws=1024, development_probes=40, minimum_conditional_mass=1e-12)
    save(root/'allocation.json', allocation)
    for name in ('common/config.json', 'common/region.json', 'common/shape.json', 'common/compiled-native.json',
                 'guides/hard_free.json', 'guides/class.json', 'execution-code/native-definition/definition.json'):
        save(root/name, {})
    save(root/'selected-probes.json', [])
    (root/'probes.jsonl').write_text('')
    save(root/'mode.json', {'mode': mode})
    executable = root/'execution-code/contact-line-guide-audit'
    script = ('#!'+runner.PYTHON+'\n'+
        'import hashlib, json, pathlib, sys, time\n'
        f'root=pathlib.Path({str(root)!r})\n'
        'value=lambda flag: sys.argv[sys.argv.index(flag)+1]\n'
        'query="--out" in sys.argv\n'
        'destination=pathlib.Path(value("--out")) / "summary.json" if query else pathlib.Path(value("--output"))\n'
        'first=query and destination.parent.name=="hard_free-r00"\n'
        'mode=json.loads((root/"mode.json").read_text())["mode"]\n'
        'if first and mode=="sleep": time.sleep(30)\n'
        'if first and mode=="fail": sys.exit(4)\n'
        'if first and mode=="missing": sys.exit(0)\n'
        'if query:\n'
        ' samples=int(value("--samples")); probes=40 if "--probes" in sys.argv else 0\n'
        ' digest=hashlib.sha256((root/"execution-code/contact-line-guide-audit").read_bytes()).hexdigest()\n'
        ' result=dict(complete=True,samples=samples+(1 if first and mode=="badcounts" else 0),probes=probes,manifest=dict(executable_sha256=digest,seed=int(value("--seed")),samples=samples))\n'
        'else: result=dict(complete=True,passed=mode!="audit_failure")\n'
        'destination.parent.mkdir(parents=True,exist_ok=True)\n'
        'destination.write_text(json.dumps(result))\n')
    executable.write_text(script); executable.chmod(0o755)
    audit = root/'execution-code/tools/native_class_line_reference.py'
    audit.parent.mkdir(); audit.write_text(script)
    frozen_names = ['allocation.json', 'selected-probes.json', 'probes.jsonl', 'mode.json'] + [
        str(p.relative_to(root)) for folder in ('common', 'guides') for p in sorted((root/folder).iterdir())]
    save(root/'freeze.json', {'files': {name:sha(root/name) for name in frozen_names}})
    paths = [root/name for name in frozen_names] + [root/'freeze.json', executable, audit,
        root/'execution-code/native-definition/definition.json', Path(runner.PYTHON)]
    plan = dict(schema='native-class-line-reviewed-execution-v1', ready=True, maximum_workers=1,
                physical_clouds=0, files={str(p):sha(p) for p in paths}, jobs=canonical_jobs(root,allocation),
                repository=str(root), allocation_sha256=sha(root/'allocation.json'),
                initial_freeze_sha256=sha(root/'freeze.json'), executable_sha256=sha(executable))
    save(root/'execution-plan.json', plan)
    return plan


class DispatcherTest(unittest.TestCase):
    def test_success_once_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); run(root)
            result = read(root/'execution/summary.json')
            self.assertTrue(result['passed']); self.assertEqual(len(result['completed']), 18)
            self.assertEqual(result['plan_sha256'], sha(root/'execution-plan.json'))
            with self.assertRaisesRegex(ValueError, 'already claimed'): run(root)

    def test_failures_stop_and_leave_queued_work_unstarted(self):
        for mode in ('fail', 'missing', 'badcounts', 'audit_failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d); plan = fixture(root, mode)
                with self.assertRaises(Exception): run(root)
                result = read(root/'execution/failure.json')
                self.assertTrue(result['child_drained'])
                expected = 9 if mode == 'audit_failure' else 0
                self.assertEqual(len(result['completed']), expected)
                self.assertFalse(Path(plan['jobs'][expected+1]['terminal']).exists())
                self.assertFalse((root/'execution/summary.json').exists())

    def test_timeout_and_keyboard_interrupt_reap_the_child(self):
        original_wait = subprocess.Popen.wait
        for interruption in ('timeout', 'keyboard'):
            with self.subTest(interruption=interruption), tempfile.TemporaryDirectory() as d:
                root = Path(d); fixture(root, 'sleep')
                def interrupted_wait(child, timeout=None):
                    if timeout is not None and not getattr(child, '_injected', False):
                        child._injected = True
                        if interruption == 'keyboard': raise KeyboardInterrupt('synthetic interrupt')
                        return original_wait(child, timeout=.02)
                    return original_wait(child, timeout=timeout)
                expected = KeyboardInterrupt if interruption == 'keyboard' else subprocess.TimeoutExpired
                with mock.patch.object(subprocess.Popen, 'wait', interrupted_wait):
                    with self.assertRaises(expected): run(root)
                failure = read(root/'execution/failure.json')
                self.assertTrue(failure['child_drained']); self.assertEqual(failure['completed'], [])
                directory = root/'execution/00-hard_free-r00'
                pid = read(directory/'process.json')['pid']
                with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
                self.assertTrue(read(directory/'exit.json')['child_drained'])
                self.assertFalse((root/'execution/01-class-r00').exists())

    def test_sigterm_terminates_and_reaps_child_and_restores_handler(self):
        original_wait = subprocess.Popen.wait
        old_handler = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root, 'sleep')
            def interrupted_wait(child, timeout=None):
                if timeout is not None and not getattr(child, '_injected', False):
                    child._injected = True
                    os.kill(os.getpid(), signal.SIGTERM)
                return original_wait(child, timeout=timeout)
            with mock.patch.object(subprocess.Popen, 'wait', interrupted_wait):
                with self.assertRaisesRegex(SystemExit, 'received signal'): run(root)
            self.assertEqual(signal.getsignal(signal.SIGTERM), old_handler)
            failure = read(root/'execution/failure.json')
            self.assertTrue(failure['child_drained']); self.assertEqual(failure['completed'], [])
            pid = read(root/'execution/00-hard_free-r00/process.json')['pid']
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
            self.assertFalse((root/'execution/01-class-r00').exists())

    def test_changed_inputs_or_allocation_fail_before_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root)
            (root/'execution-code/contact-line-guide-audit').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'): run(root)
            self.assertFalse((root/'execution').exists())
        for mutation in ('seed', 'samples_argument', 'guide_argument', 'terminal', 'order', 'resource', 'missing_binding'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d); plan = fixture(root)
                if mutation == 'seed': plan['jobs'][0]['seed'] += 1
                if mutation == 'samples_argument':
                    a=plan['jobs'][0]['argv'];a[a.index('--samples')+1]='129'
                if mutation == 'guide_argument':
                    a=plan['jobs'][0]['argv'];a[a.index('--importance-guide')+1]=str(root/'guides/class.json')
                if mutation == 'terminal': plan['jobs'][0]['terminal']=str(root/'different.json')
                if mutation == 'order': plan['jobs'][:2]=reversed(plan['jobs'][:2])
                if mutation == 'resource': plan['jobs'][0]['cpu_limit_seconds'] += 1
                if mutation == 'missing_binding': del plan['files'][str(root/'common/config.json')]
                with self.assertRaises(ValueError): verify(root,plan)
                self.assertFalse((root/'execution').exists())


if __name__ == '__main__': unittest.main()
