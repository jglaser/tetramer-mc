#!/usr/bin/env python3
"""Execute an externally frozen physical workflow once, one owned child at a time.

This module makes no plans, changes no argv, and interprets no scientific gate.
The preparer owns all commands, stage materialization and scientific limits.
Every begun stage is retained; failure stops the queue without resume or retry.
Driver output occupies root/execution; declared terminals occupy other paths
under root. A lifecycle success only means every explicit terminal contract
passed. Statistical reports may be complete while scientific gates stay closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time


SCHEMA = 'native-class-physical-execution-v1'
PHASES = frozenset(('producer', 'algebra', 'labels', 'selection', 'geometry', 'statistics', 'admission'))
CONTRACTS = frozenset(('complete', 'complete_and_passed'))
TERM_GRACE_SECONDS = 2.
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key '+key)
        result[key] = value
    return result


def read(path):
    def invalid(value): raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=invalid)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write(path, value):
    """Exclusive durable receipt; never overwrite an attempt or result."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _status(path, value):
    temporary = path.with_name(path.name+'.tmp')
    try:
        write(temporary, value)
        os.replace(temporary, path); _sync_directory(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def _absolute(value, label):
    require(type(value) is str and Path(value).is_absolute(), 'Absolute '+label+' required')
    path = Path(value)
    require(str(path.resolve()) == value, 'Canonical '+label+' required')
    return path


def _digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')


def job_directory(root, ordinal, job):
    return Path(root)/'execution'/'jobs'/f'{ordinal:03d}-{job["id"]}'


def completed_terminal(root, plan, identity):
    """Authenticate a predecessor's published lifecycle records, without rows.

    This binds evidence to the claim and plan. The stage worker separately
    verifies that its actual controller is live and that its own stage is active.
    """
    root = Path(root).resolve(); plan_path = root/'execution-plan.json'
    require(read(plan_path) == plan, 'Predecessor execution plan differs')
    digest = sha(plan_path); verify_plan(plan_path, plan, digest)
    claim = read(root/'execution/claim.json')
    require(claim['schema'] == SCHEMA and claim['plan_sha256'] == digest
            and claim['maximum_workers'] == claim['threads'] == 1
            and claim['retries'] == claim['replacements'] == 0, 'Predecessor controller claim differs')
    matches = [(ordinal, job) for ordinal, job in enumerate(plan['jobs']) if job['id'] == identity]
    require(len(matches) == 1, 'Unknown predecessor job')
    ordinal, job = matches[0]; directory = job_directory(root, ordinal, job)
    result = read(directory/'success.json')
    require(read(directory/'exit.json') == result and read(directory/'attempt.json')['job'] == job,
            'Predecessor lifecycle records differ')
    for key in ('id', 'population', 'phase', 'argv'):
        require(result[key] == job[key], 'Predecessor job identity differs: '+key)
    require(result['success'] is True and result['error'] is None and result['timeout'] is False
            and result['child_started'] is True and result['child_drained'] is True
            and type(result['returncode']) is int and result['returncode'] == 0
            and result['retries'] == result['replacements'] == 0, 'Predecessor did not complete cleanly')
    process = read(directory/'process.json')
    require(type(result['pid']) is int and result['pid'] > 0
            and all(process[key] == result[key] for key in ('id', 'pid', 'birth_ticks', 'argv')),
            'Predecessor process identity differs')
    terminal = result['terminal']
    require(set(terminal) == {'path', 'sha256'} and terminal['path'] == job['terminal']['path']
            and result['success_contract'] == job['terminal']['success_contract'], 'Predecessor terminal identity differs')
    _digest(terminal['sha256'])
    require(sha(terminal['path']) == terminal['sha256'], 'Predecessor terminal bytes changed')
    value = read(terminal['path'])
    require(value.get('complete') is True and
            (result['success_contract'] == 'complete' or value.get('passed') is True),
            'Predecessor terminal contract differs')
    return dict(terminal)


def verify_plan(plan_path, plan, plan_sha256, *, fresh=False):
    """Metadata/hash validation only; never execute or interpret a job."""
    _digest(plan_sha256)
    plan_path = Path(plan_path).resolve()
    require(sha(plan_path) == plan_sha256, 'Frozen execution plan changed')
    require(plan['schema'] == SCHEMA and type(plan['maximum_workers']) is int
            and plan['maximum_workers'] == 1 and type(plan['threads']) is int
            and plan['threads'] == 1, 'One worker and one thread required')
    root = _absolute(plan['root'], 'execution root')
    require(plan_path == root/'execution-plan.json', 'Execution plan path differs from root')
    files = plan['files']
    require(type(files) is dict and files, 'Frozen input closure required')
    for value, digest in files.items():
        path = _absolute(value, 'input path'); _digest(digest)
        require(not path.is_relative_to(root/'execution'), 'Input aliases driver output namespace')
        require(sha(path) == digest, 'Frozen input changed: '+value)
    if 'preparation_receipt' in plan:
        path = _absolute(plan['preparation_receipt'], 'preparation receipt')
        require(path == root/'preparation.json' and str(path) not in files,
                'Preparation receipt path or circular binding differs')
        require(not (root/'preparation-failure.json').exists(), 'Preparation failure is present')
        prepared = read(path)
        require(prepared['complete'] is True and prepared['launched'] is False
                and prepared['execution_plan'] == dict(path=str(plan_path), sha256=plan_sha256),
                'Preparation is incomplete or binds another execution plan')
        protocol = prepared['protocol']
        require(set(protocol) == {'path', 'sha256'} and protocol['path'] == str(root/'protocol.json')
                and files.get(protocol['path']) == protocol['sha256'], 'Prepared protocol is not frozen')
        claim_path = root/'execution/claim.json'
        if claim_path.exists():
            require(read(claim_path)['preparation_receipt'] == dict(path=str(path), sha256=sha(path)),
                    'Preparation receipt changed after claim')
    require(str(Path(__file__).resolve()) in files, 'Executed driver source is not bound')
    require(type(plan['jobs']) is list and plan['jobs'], 'Ordered jobs required')
    resolutions = plan['executable_resolutions']
    require(type(resolutions) is dict and resolutions, 'Executable resolution bindings required')
    for lexical, target in resolutions.items():
        require(type(lexical) is str and Path(lexical).is_absolute()
                and os.path.normpath(lexical) == lexical, 'Absolute executable argv required')
        canonical = _absolute(target, 'executable target')
        require(str(canonical) in files and Path(lexical).resolve() == canonical,
                'Frozen executable resolution changed')
    ids, terminals, executables = set(), set(), set()
    for job in plan['jobs']:
        name = job['id']
        require(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                and name not in ids, 'Unsafe or duplicate job ID')
        ids.add(name)
        require(type(job['population']) is str and job['population'], 'Population identity required')
        require(job['phase'] in PHASES, 'Unknown workflow phase')
        argv = job['argv']
        require(type(argv) is list and argv and all(type(v) is str and '\0' not in v for v in argv),
                'An exact argument vector is required')
        require(argv[0] in resolutions, 'Executable is outside frozen resolution map')
        executables.add(argv[0])
        for key in ('cpu_limit_seconds', 'wall_limit_seconds', 'address_space_limit_bytes'):
            require(type(job[key]) is int and job[key] > 0, 'Positive integer limit required: '+key)
        terminal = job['terminal']
        require(set(terminal) == {'path', 'success_contract'} and terminal['success_contract'] in CONTRACTS,
                'Explicit terminal success contract required')
        path = _absolute(terminal['path'], 'terminal path')
        require(path.is_relative_to(root) and path != root and not path.is_relative_to(root/'execution'),
                'Terminal must be under root outside driver output namespace')
        require(str(path) not in files and path != plan_path and path not in terminals,
                'Terminal aliases an input or another output')
        require(not any(path.is_relative_to(Path(p)) or Path(p).is_relative_to(path) for p in files),
                'Terminal and input paths overlap')
        require(not any(p.is_relative_to(path) or path.is_relative_to(p) for p in terminals),
                'Terminal paths overlap')
        terminals.add(path)
        if fresh: require(not path.exists(), 'Declared terminal already exists; no resume')
    require(executables == set(resolutions), 'Executable resolution inventory differs')
    if fresh: require(not (root/'execution').exists(), 'Execution already claimed; no resume')
    return root


def _birth(pid):
    try: return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError: return None


def _stop(signum, _frame):
    raise SystemExit('Controller received signal '+str(signum))


def _group_exists(pid):
    try: os.killpg(pid, 0)
    except ProcessLookupError: return False
    return True


def drain(child):
    """Allow a bounded TERM cleanup, then KILL the owned group and reap."""
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    term = killed = False
    try:
        try: os.killpg(child.pid, signal.SIGTERM); term = True
        except ProcessLookupError: pass
        deadline = time.monotonic()+TERM_GRACE_SECONDS
        while _group_exists(child.pid) and time.monotonic() < deadline:
            child.poll(); time.sleep(.02)
        try: os.killpg(child.pid, signal.SIGKILL); killed = True
        except ProcessLookupError: pass
        while True:
            try: child.wait(); break
            except (KeyboardInterrupt, InterruptedError): continue
        return dict(term_sent=term, kill_sent=killed, grace_seconds=TERM_GRACE_SECONDS,
                    child_drained=child.returncode is not None)
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def execute_job(job, directory, root, verify):
    """Own the child before unmasking signals, including publication failures."""
    require(not Path(job['terminal']['path']).exists(), 'Terminal already exists; no repeated stage')
    verify()
    started = time.time(); clock = time.monotonic()
    write(directory/'attempt.json', dict(job=job, started=started, retries=0, replacements=0))
    child = None; birth = None; error = None; terminal_sha = None; prior_mask = None
    terminal_hash_error = None
    cleanup = dict(term_sent=False, kill_sent=False, grace_seconds=TERM_GRACE_SECONDS, child_drained=True)
    env = dict(os.environ, **{key: '1' for key in THREAD_KEYS})
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (job['cpu_limit_seconds'], job['cpu_limit_seconds']+1))
        resource.setrlimit(resource.RLIMIT_AS, (job['address_space_limit_bytes'], job['address_space_limit_bytes']))
        os.nice(10); signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    try:
        with (directory/'output.log').open('x') as log:
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                child = subprocess.Popen(job['argv'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         shell=False, start_new_session=True, preexec_fn=limits)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            birth = _birth(child.pid)
            write(directory/'process.json', dict(id=job['id'], pid=child.pid, birth_ticks=birth,
                                                started=started, argv=job['argv']))
            child.wait(timeout=job['wall_limit_seconds'])
            log.flush(); os.fsync(log.fileno())
        require(child.returncode == 0, 'Child failed with code '+str(child.returncode))
        # A successful leader may not leave background descendants behind.
        if _group_exists(child.pid): cleanup = drain(child)
        verify()
        terminal = Path(job['terminal']['path'])
        require(terminal.is_file(), 'Missing terminal receipt')
        terminal_sha = sha(terminal)
        result = read(terminal)
        require(type(result) is dict and result.get('complete') is True, 'Incomplete terminal receipt')
        if job['terminal']['success_contract'] == 'complete_and_passed':
            require(result.get('passed') is True, 'Terminal did not pass')
    except BaseException as caught:
        error = caught
        if child is not None: cleanup = drain(child)
        raise
    finally:
        if terminal_sha is None:
            terminal = Path(job['terminal']['path'])
            try:
                if terminal.is_file(): terminal_sha = sha(terminal)
            except Exception as binding_error:
                terminal_hash_error = repr(binding_error)
        record = dict(id=job['id'], population=job['population'], phase=job['phase'], argv=job['argv'],
            started=started, finished=time.time(), pid=None if child is None else child.pid, birth_ticks=birth,
            child_started=child is not None, child_drained=cleanup['child_drained'],
            returncode=None if child is None else child.returncode,
            terminal=dict(path=job['terminal']['path'], sha256=terminal_sha),
            success_contract=job['terminal']['success_contract'],
            terminal_hash_error=terminal_hash_error,
            success=error is None, error=None if error is None else repr(error),
            timeout=isinstance(error, subprocess.TimeoutExpired), cleanup=cleanup,
            wall_seconds=time.monotonic()-clock, retries=0, replacements=0)
        write(directory/'exit.json', record)
    write(directory/'success.json', record)
    return record


def run(plan_path, *, plan_sha256):
    plan_path = Path(plan_path).resolve(); plan = read(plan_path)
    root = verify_plan(plan_path, plan, plan_sha256, fresh=True)
    out = root/'execution'; out.mkdir(); (out/'jobs').mkdir()
    claim = dict(schema=SCHEMA, pid=os.getpid(), birth_ticks=_birth(os.getpid()),
        started=time.time(), plan_sha256=plan_sha256, maximum_workers=1, threads=1, retries=0, replacements=0)
    if 'preparation_receipt' in plan:
        claim['preparation_receipt'] = dict(path=plan['preparation_receipt'], sha256=sha(plan['preparation_receipt']))
    write(out/'claim.json', claim)
    completed = []; current = None; next_ordinal = 0; failure = None
    handlers = {s: signal.signal(s, _stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def verify(): verify_plan(plan_path, plan, plan_sha256)
    def snapshot(complete=False):
        _status(out/'status.json', dict(schema=SCHEMA, complete=complete,
            passed=complete and failure is None, plan_sha256=plan_sha256,
            completed=completed, active=current, unstarted=plan['jobs'][next_ordinal:],
            failure=failure, retries=0, replacements=0, maximum_workers=1, threads=1))
    try:
        snapshot()
        for ordinal, job in enumerate(plan['jobs']):
            current = dict(ordinal=ordinal, id=job['id'], population=job['population'], phase=job['phase'])
            next_ordinal = ordinal+1
            directory = job_directory(root, ordinal, job); directory.mkdir()
            snapshot()
            result = execute_job(job, directory, root, verify)
            completed.append(result); current = None; snapshot()
        verify()
        for result in completed:
            require(completed_terminal(root, plan, result['id']) == result['terminal'], 'Completed terminal changed')
        summary = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=plan_sha256,
            files=plan['files'], completed=completed, active=None, unstarted=[], failure=None,
            maximum_workers=1, threads=1, retries=0, replacements=0,
            scientific_admission_asserted=False,
            scope='Lifecycle completion of explicit terminal contracts; no independent scientific admission.')
        snapshot(complete=True); write(out/'summary.json', summary)
        return summary
    except BaseException as caught:
        # execute_job has already drained any owned group before this boundary.
        failure = repr(caught); failed_job = current; current = None
        previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            status_error = None
            try: snapshot()
            except BaseException as publication_error: status_error = repr(publication_error)
            write(out/'failure.json', dict(schema=SCHEMA, complete=False, passed=False,
                plan_sha256=plan_sha256, failed_job=failed_job, completed=completed,
                unstarted=plan['jobs'][next_ordinal:], error=failure, partial_outputs_retained=True,
                status_publication_error=status_error,
                retries=0, replacements=0, scientific_admission_asserted=False))
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    result = run(args.plan, plan_sha256=args.plan_sha256)
    print(json.dumps(dict(complete=result['complete'], jobs=len(result['completed']))))


if __name__ == '__main__': main()
