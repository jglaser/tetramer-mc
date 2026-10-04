"""Tiny subprocess lifecycle tests only; no scientific inputs or allocations."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_native_class_physical_campaign as runner


CHILD = '''import json, os, pathlib, resource, signal, sys, time
out, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
if mode == 'ignore': signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == 'graceful':
    def stop(signum, frame):
        out.write_text(json.dumps(dict(complete=False, stopped=True)))
        raise SystemExit(3)
    signal.signal(signal.SIGTERM, stop)
if mode in ('sleep', 'ignore', 'graceful'):
    out.with_suffix('.ready').write_text(str(os.getpid()))
    time.sleep(60)
if mode == 'change_input': pathlib.Path(sys.argv[3]).write_text('changed input')
if mode == 'change_terminal': pathlib.Path(sys.argv[3]).write_text('changed terminal')
if mode == 'nonzero':
    out.write_text(json.dumps(dict(complete=False)))
    raise SystemExit(7)
if mode == 'missing': raise SystemExit(0)
out.write_text(json.dumps(dict(complete=mode != 'incomplete', passed=mode != 'unresolved',
    argv=sys.argv, executable=sys.executable, pid=os.getpid(),
    threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
    cpu=list(resource.getrlimit(resource.RLIMIT_CPU)), memory=list(resource.getrlimit(resource.RLIMIT_AS)))))
'''


class PhysicalCampaignLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root/'outputs').mkdir()
        self.script = self.root/'child.py'; self.script.write_text(CHILD)
        self.input = self.root/'frozen.txt'; self.input.write_text('frozen input')
        self.python = str(Path(sys.executable).absolute())
        self.plan_path = self.root/'execution-plan.json'
        self.plan = dict(schema=runner.SCHEMA, root=str(self.root), maximum_workers=1, threads=1,
            executable_resolutions={self.python: str(Path(self.python).resolve())},
            files={str(p.resolve()): runner.sha(p) for p in
                   (Path(runner.__file__), Path(self.python), self.script, self.input)}, jobs=[])

    def tearDown(self): self.temporary.cleanup()

    def add(self, identity, mode='success', *, phase='producer', contract='complete_and_passed', extra=()):
        terminal = self.root/'outputs'/(identity+'.json')
        self.plan['jobs'].append(dict(id=identity, population='toy-r00', phase=phase,
            argv=[self.python, '-B', str(self.script), str(terminal), mode, *extra],
            terminal=dict(path=str(terminal), success_contract=contract),
            cpu_limit_seconds=5, wall_limit_seconds=5, address_space_limit_bytes=256*1024**2))
        return terminal

    def freeze(self):
        self.plan_path.write_text(json.dumps(self.plan, allow_nan=False)+'\n')
        return runner.sha(self.plan_path)

    def execute(self): return runner.run(self.plan_path, plan_sha256=self.freeze())

    def test_success_exact_argv_limits_threads_and_predecessor_bindings(self):
        first = self.add('produce', extra=('literal $HOME; `false`',))
        self.add('statistics', 'unresolved', phase='statistics', contract='complete')
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        result = self.execute()
        self.assertTrue(result['complete']); self.assertTrue(result['passed'])
        self.assertFalse(result['scientific_admission_asserted'])
        self.assertEqual([v['id'] for v in result['completed']], ['produce', 'statistics'])
        self.assertEqual(result['active'], None); self.assertEqual(result['unstarted'], [])
        report = runner.read(first)
        self.assertEqual(report['argv'][-1], 'literal $HOME; `false`')
        self.assertEqual(report['threads'], dict.fromkeys(runner.THREAD_KEYS, '1'))
        self.assertEqual(report['cpu'], [5, 6]); self.assertEqual(report['memory'], [256*1024**2]*2)
        self.assertEqual(runner.completed_terminal(self.root, self.plan, 'produce'),
                         dict(path=str(first), sha256=runner.sha(first)))
        for ordinal, item in enumerate(result['completed']):
            self.assertEqual(item['argv'], self.plan['jobs'][ordinal]['argv'])
            self.assertTrue(item['child_started'] and item['child_drained'])
            self.assertEqual(item['returncode'], 0)
            self.assertIsInstance(item['pid'], int)
            self.assertTrue(item['birth_ticks'] is None or type(item['birth_ticks']) is int)
            directory = runner.job_directory(self.root, ordinal, item)
            self.assertEqual(runner.read(directory/'exit.json'), runner.read(directory/'success.json'))
        self.assertEqual({sig: signal.getsignal(sig) for sig in before}, before)
        with self.assertRaisesRegex(ValueError, 'no resume'):
            runner.run(self.plan_path, plan_sha256=runner.sha(self.plan_path))

    def test_failure_stops_queue_and_keeps_terminal_hash(self):
        first = self.add('failed', 'unresolved')
        second = self.add('unstarted')
        with self.assertRaisesRegex(ValueError, 'did not pass'): self.execute()
        self.assertFalse(second.exists())
        directory = runner.job_directory(self.root, 0, self.plan['jobs'][0])
        record = runner.read(directory/'exit.json')
        self.assertTrue(record['child_drained']); self.assertFalse(record['success'])
        self.assertEqual(record['terminal']['sha256'], runner.sha(first))
        failure = runner.read(self.root/'execution/failure.json')
        self.assertEqual(failure['failed_job']['id'], 'failed')
        self.assertEqual([v['id'] for v in failure['unstarted']], ['unstarted'])
        self.assertIsNone(runner.read(self.root/'execution/status.json')['active'])
        self.assertFalse((self.root/'execution/summary.json').exists())

    def test_invalid_frozen_contract_never_launches(self):
        self.add('one'); original = copy.deepcopy(self.plan)
        mutations = [lambda p: p.update(maximum_workers=2), lambda p: p.update(threads=True),
            lambda p: p['jobs'][0].update(cpu_limit_seconds=True),
            lambda p: p['jobs'][0].update(wall_limit_seconds=0),
            lambda p: p['jobs'][0].update(address_space_limit_bytes=1.5),
            lambda p: p['jobs'][0].update(phase='unknown'),
            lambda p: p['jobs'][0].update(id='../escape'),
            lambda p: p['jobs'].append(copy.deepcopy(p['jobs'][0])),
            lambda p: p['jobs'][0]['terminal'].update(success_contract='successful'),
            lambda p: p['jobs'][0]['terminal'].update(path=str(self.input)),
            lambda p: p['jobs'][0]['terminal'].update(path=str(self.root.parent/'escape.json')),
            lambda p: p['jobs'][0]['terminal'].update(path=str(self.root/'execution/summary.json')),
            lambda p: p['files'].pop(str(Path(runner.__file__).resolve()))]
        with mock.patch.object(runner.subprocess, 'Popen') as child:
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    self.plan = copy.deepcopy(original); mutation(self.plan)
                    with self.assertRaises(ValueError): self.execute()
                    self.assertFalse((self.root/'execution').exists())
            child.assert_not_called()

    def test_hash_and_terminal_existence_fail_before_claim(self):
        output = self.add('one'); digest = self.freeze()
        self.input.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            runner.run(self.plan_path, plan_sha256=digest)
        self.input.write_text('frozen input'); output.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'terminal already exists'):
            runner.run(self.plan_path, plan_sha256=digest)
        output.unlink(); self.plan_path.write_text(self.plan_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'execution plan changed'):
            runner.run(self.plan_path, plan_sha256=digest)

    def test_optional_preparation_receipt_requires_finished_unfailed_matching_freeze(self):
        self.add('prepared')
        protocol = self.root/'protocol.json'; protocol.write_text('{"frozen":true}\n')
        self.plan['files'][str(protocol)] = runner.sha(protocol)
        preparation = self.root/'preparation.json'
        self.plan['preparation_receipt'] = str(preparation)
        digest = self.freeze()
        value = dict(complete=True, launched=False,
            protocol=dict(path=str(protocol), sha256=runner.sha(protocol)),
            execution_plan=dict(path=str(self.plan_path), sha256=digest))
        with mock.patch.object(runner.subprocess, 'Popen') as child:
            with self.assertRaises(FileNotFoundError): runner.run(self.plan_path, plan_sha256=digest)
            for key, changed in [('complete', False), ('launched', True),
                                 ('execution_plan', dict(path=str(self.plan_path), sha256='0'*64)),
                                 ('protocol', dict(path=str(protocol), sha256='0'*64))]:
                preparation.write_text(json.dumps(dict(value, **{key: changed})))
                with self.assertRaises(ValueError): runner.run(self.plan_path, plan_sha256=digest)
                self.assertFalse((self.root/'execution').exists())
            preparation.write_text(json.dumps(value))
            failure = self.root/'preparation-failure.json'; failure.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Preparation failure'):
                runner.run(self.plan_path, plan_sha256=digest)
            failure.unlink(); child.assert_not_called()
        runner.run(self.plan_path, plan_sha256=digest)
        claim = runner.read(self.root/'execution/claim.json')
        self.assertEqual(claim['preparation_receipt'], dict(path=str(preparation), sha256=runner.sha(preparation)))
        preparation.write_text(json.dumps(dict(value, changed=True)))
        with self.assertRaisesRegex(ValueError, 'changed after claim'):
            runner.verify_plan(self.plan_path, self.plan, digest)

    def test_symlink_executable_argv_preserved_and_resolution_drift_rejected(self):
        link = self.root/'venv-python'; link.symlink_to(Path(self.python).resolve())
        self.python = str(link)
        self.plan['executable_resolutions'] = {str(link): str(link.resolve())}
        output = self.add('linked')
        result = self.execute()
        self.assertEqual(result['completed'][0]['argv'][0], str(link))
        self.assertEqual(runner.read(output)['executable'], str(link))
        link.unlink(); link.symlink_to(self.script)
        with self.assertRaisesRegex(ValueError, 'resolution changed'):
            runner.verify_plan(self.plan_path, self.plan, runner.sha(self.plan_path))

    def test_changed_frozen_input_during_child_stops_later_jobs(self):
        self.add('mutator', 'change_input', extra=(str(self.input),))
        later = self.add('later')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'): self.execute()
        self.assertFalse(later.exists())
        record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
        self.assertTrue(record['child_drained']); self.assertEqual(record['returncode'], 0)

    def test_predecessor_record_or_terminal_tampering_rejected(self):
        path = self.add('one'); self.execute()
        record = runner.job_directory(self.root, 0, self.plan['jobs'][0])/'success.json'
        original = record.read_bytes(); data = runner.read(record); data['pid'] += 1
        record.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'lifecycle records differ'):
            runner.completed_terminal(self.root, self.plan, 'one')
        record.write_bytes(original); path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'terminal bytes changed'):
            runner.completed_terminal(self.root, self.plan, 'one')

    def test_missing_incomplete_nonzero_and_launch_failure_have_no_success(self):
        for mode in ('missing', 'incomplete', 'nonzero', 'launch'):
            with self.subTest(mode=mode):
                # Each attempted execution uses a separately claimed toy root.
                self.tearDown(); self.setUp(); self.add('bad', mode)
                patch = mock.patch.object(runner.subprocess, 'Popen', side_effect=OSError('synthetic launch failure'))
                if mode == 'launch':
                    with patch, self.assertRaises(OSError): self.execute()
                else:
                    with self.assertRaises(ValueError): self.execute()
                record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
                self.assertFalse(record['success']); self.assertTrue(record['child_drained'])
                self.assertEqual(record['child_started'], mode != 'launch')
                self.assertFalse((self.root/'execution/summary.json').exists())

    def test_timeout_term_allows_child_cleanup_then_reaps(self):
        output = self.add('graceful', 'graceful'); self.plan['jobs'][0]['wall_limit_seconds'] = 1
        with self.assertRaises(subprocess.TimeoutExpired): self.execute()
        self.assertTrue(runner.read(output)['stopped'])
        record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
        self.assertTrue(record['child_drained'] and record['timeout'])
        self.assertTrue(record['cleanup']['term_sent']); self.assertFalse(record['cleanup']['kill_sent'])

    def test_timeout_ignoring_term_is_killed_and_reaped_without_retry(self):
        self.add('ignores', 'ignore'); self.plan['jobs'][0]['wall_limit_seconds'] = 1
        with mock.patch.object(runner, 'TERM_GRACE_SECONDS', .05), self.assertRaises(subprocess.TimeoutExpired):
            self.execute()
        record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
        self.assertTrue(record['child_drained'] and record['cleanup']['kill_sent'])
        self.assertEqual(record['returncode'], -signal.SIGKILL)

    def test_signal_between_fork_and_publication_retains_owned_child(self):
        self.add('signal', 'sleep'); real_popen = subprocess.Popen; children = []
        def interrupted(*args, **kwargs):
            child = real_popen(*args, **kwargs); children.append(child)
            os.kill(os.getpid(), signal.SIGTERM)
            return child
        with mock.patch.object(runner.subprocess, 'Popen', side_effect=interrupted), self.assertRaises(SystemExit):
            self.execute()
        self.assertEqual(len(children), 1); self.assertIsNotNone(children[0].poll())
        record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
        self.assertTrue(record['child_drained']); self.assertEqual(record['pid'], children[0].pid)

    def test_process_publication_failure_drains_child_before_failure_receipt(self):
        self.add('publication', 'sleep'); real_write = runner.write
        def publication(path, value):
            if Path(path).name == 'process.json': raise OSError('synthetic publication failure')
            return real_write(path, value)
        with mock.patch.object(runner, 'write', side_effect=publication), self.assertRaisesRegex(OSError, 'publication'):
            self.execute()
        record = runner.read(runner.job_directory(self.root, 0, self.plan['jobs'][0])/'exit.json')
        self.assertTrue(record['child_drained']); self.assertIsNotNone(record['returncode'])


if __name__ == '__main__': unittest.main()
