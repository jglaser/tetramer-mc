#!/usr/bin/env python3
"""Run a frozen, bounded, single-worker toy allocation; never retry a failed job.

Each launched process owns a new process group. Any BaseException after launch
(including SIGTERM converted to SystemExit) kills that group and reaps its leader
before the immutable failure ledger is written. This also covers errors while
recording a launch, reading a terminal receipt, or validating frozen inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _termination_requested(signum, _frame):
    raise SystemExit('Controller received signal ' + str(signum))


def _drain(child):
    """Terminate the owned group and reap despite repeated stop signals."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:
                continue
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(root):
    root = Path(root).resolve()
    allocation_bytes = (root / 'allocation.json').read_bytes()
    binding_bytes = (root / 'execution-binding.json').read_bytes()
    plan = json.loads(allocation_bytes)
    binding = json.loads(binding_bytes)
    allocation_sha = hashlib.sha256(allocation_bytes).hexdigest()
    binding_sha = hashlib.sha256(binding_bytes).hexdigest()
    require(binding['maximum_workers'] == 1 and
            binding['allocation_sha256'] == allocation_sha,
            'Changed frozen execution allocation')

    def verify():
        require(sha(root / 'allocation.json') == allocation_sha,
                'Frozen allocation changed')
        require(sha(root / 'execution-binding.json') == binding_sha,
                'Frozen execution binding changed')
        for p, h in binding['files'].items():
            require(sha(p) == h, 'Frozen execution file changed ' + p)

    verify()
    # Exclusive claim makes even interrupted attempts non-restartable.
    write(root / 'execution-claim.json', dict(
        pid=os.getpid(), started_unix=time.time(),
        binding_sha256=binding_sha))
    completed = []
    attempts = []
    current = None
    child_drained = True
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')

    def execute(job, phase, argv, cpu, wall, terminal):
        nonlocal current, child_drained
        current = dict(id=job['id'], phase=phase)
        verify()
        started = time.monotonic()
        label = job['id'] + '-' + phase
        child = None
        error = None
        result = None

        def limits():
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))

        try:
            with (root / 'logs' / (label + '.log')).open('x') as log:
                # A stop arriving inside Popen is delivered only after the
                # returned handle is owned. The child restores the old mask
                # before exec, so its stop signals remain enabled.
                prior_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                try:
                    child = subprocess.Popen(
                        argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                        preexec_fn=limits, start_new_session=True)
                    child_drained = False
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                write(root / 'logs' / (label + '.launch.json'), dict(
                    argv=argv, pid=child.pid, cpu_limit_seconds=cpu,
                    wall_limit_seconds=wall, started_unix=time.time()))
                child.wait(timeout=wall)
                require(child.returncode == 0, 'Child command failed: ' + label)
                receipt = read(terminal)
                require(receipt.get('complete') is True,
                        'Successful command lacks complete receipt')
                if phase in ('audit', 'analysis'):
                    require(receipt.get('passed') is True,
                            'Independent validation failed: ' + label)
                verify()
                child_drained = True
        except BaseException as caught:
            error = caught
            if child is not None:
                _drain(child)
                child_drained = True
            raise
        finally:
            result = dict(
                id=job['id'], phase=phase,
                returncode=None if child is None else child.returncode,
                timeout=isinstance(error, subprocess.TimeoutExpired),
                wall_seconds=time.monotonic() - started,
                terminal=str(terminal),
                terminal_sha256=sha(terminal) if terminal.exists() else None,
                child_pid=None if child is None else child.pid,
                child_started=child is not None, child_drained=child_drained,
                error=None if error is None else repr(error))
            attempts.append(result)
            write(root / 'logs' / (label + '.terminal.json'), result)
            print(json.dumps(result), flush=True)
        return result

    previous_sigterm = signal.signal(signal.SIGTERM, _termination_requested)
    try:
        (root / 'audits').mkdir()
        (root / 'logs').mkdir()
        for job in plan['jobs']:
            out = Path(job['out'])
            argv = [binding['executable'], '--config', job['config'],
                    '--region', job['region'], '--importance-guide', job['guide'],
                    '--out', str(out), '--samples', str(job['samples']),
                    '--seed', str(job['seed']), '--cloud-replicates', '2',
                    '--lambda-ratio', str(job['lambda_ratio'])]
            result = execute(job, 'physical', argv, job['cpu_limit_seconds'],
                             job['wall_limit_seconds'], out / 'summary.json')
            completed.append(result)
            destination = root / 'audits' / (job['id'] + '.json')
            argv = [binding['python'], str(Path(binding['tools']) /
                    'native_class_line_physical_reference.py'), '--root', str(out),
                    '--synthetic', '--out', str(destination), '--journal',
                    str(root / 'audits' / (job['id'] + '.journal.jsonl'))]
            completed.append(execute(job, 'audit', argv, 1200, 2400, destination))
        verify()
        summary = dict(
            complete=True, passed=True, executable_sha256=binding['executable_sha256'],
            allocation_sha256=allocation_sha, binding_sha256=binding_sha,
            completed=completed, failure=None, remaining_jobs_not_started=0,
            retries=0, protein_queries=0)
        write(root / 'execution-summary.json', summary)
        argv = [binding['python'], str(Path(binding['tools']) /
                'analyze_native_class_line_physical_toy.py'), '--root', str(root),
                '--out', str(root / 'analytic-validation.json')]
        result = execute(dict(id='analytic-validation'), 'analysis', argv, 600,
                         1200, root / 'analytic-validation.json')
        verify()
        write(root / 'completion.json', dict(
            complete=True, passed=True, analysis=result,
            execution_summary_sha256=sha(root / 'execution-summary.json')))
    except BaseException as error:
        # No retry and no queued work after any failure, including launch-record
        # serialization, KeyboardInterrupt, SystemExit or input revalidation.
        previous = {sig: signal.signal(sig, signal.SIG_IGN)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            write(root / 'failure.json', dict(
                complete=False, passed=False, current=current,
                error=repr(error), traceback=traceback.format_exc(),
                child_drained=child_drained, completed=completed, attempts=attempts,
                remaining_jobs_not_started=len(plan['jobs']) - len({
                    r['id'] for r in attempts if r['phase'] == 'physical'}),
                retries=0, protein_queries=0))
            write(root / 'completion.json', dict(
                complete=False, passed=False, failure_sha256=sha(root / 'failure.json')))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
