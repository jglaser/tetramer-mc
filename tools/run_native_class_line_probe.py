#!/usr/bin/env python3
"""Run an already reviewed, frozen class-line diagnostic once, one child at a time.

The execution plan binds every input, executable, source and validation receipt.
This command never makes a plan, retries a failed query, or extends its allocation.
"""
from __future__ import annotations
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


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)
def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


PYTHON = '/home/xvg/protein-nucleation/.venv/bin/python'


def canonical_jobs(root, allocation):
    """Actual argv, order and resource budgets are part of the frozen allocation."""
    root = Path(root).absolute()
    output = root/'execution-code'
    queries = []
    for item in allocation['jobs'] + [dict(id='saved', arm='class', samples=0, seed=0)]:
        destination = root/'queries'/item['id']
        probes = 40 if item['id'] == 'saved' else 0
        argv = [str(output/'contact-line-guide-audit'), '--config', str(root/'common/config.json'),
                '--region', str(root/'common/region.json'), '--importance-guide', str(root/'guides'/f'{item["arm"]}.json'),
                '--out', str(destination), '--samples', str(item['samples']), '--seed', str(item['seed'])]
        if probes: argv += ['--probes', str(root/'probes.jsonl')]
        queries.append(dict(item, phase='query', probes=probes, argv=argv,
                            terminal=str(destination/'summary.json'), cpu_limit_seconds=1800, wall_limit_seconds=3600))
    audits = []
    for query in queries:
        ident = query['id']; destination = root/'audits'/f'{ident}.json'
        audits.append(dict(id=ident+'-audit', query_id=ident, phase='audit',
            argv=[PYTHON, str(output/'tools/native_class_line_reference.py'), str(root/'queries'/ident),
                  '--definition', str(output/'native-definition/definition.json'), '--output', str(destination),
                  '--journal', str(root/'audits'/f'{ident}.journal.jsonl')],
            terminal=str(destination), cpu_limit_seconds=7200, wall_limit_seconds=14400))
    return queries+audits


def verify(root, plan):
    root = Path(root).absolute()
    require(plan['schema'] == 'native-class-line-reviewed-execution-v1' and plan['ready'] is True,
            'Missing reviewed execution plan')
    require(plan['maximum_workers'] == 1 and plan['physical_clouds'] == 0, 'Diagnostic scope changed')
    require(Path(plan['repository']).is_absolute(), 'Relative repository path')
    for path, digest in plan['files'].items():
        require(Path(path).is_absolute(), 'Relative frozen input path')
        require(sha(path) == digest, 'Frozen file changed: '+path)
    allocation = read(root/'allocation.json')
    from prepare_native_class_line_probe import jobs
    require(allocation['jobs'] == jobs(), 'Population binding changed')
    require(allocation['fresh_draws'] == 1024 and allocation['development_probes'] == 40,
            'Fixed query allocation changed')
    require(sha(root/'allocation.json') == plan['allocation_sha256'] and
            sha(root/'freeze.json') == plan['initial_freeze_sha256'], 'Preparation binding changed')
    for name, digest in read(root/'freeze.json')['files'].items():
        require(plan['files'].get(str(root/name)) == digest, 'Unbound frozen input: '+name)
    required = [root/'allocation.json', root/'freeze.json', root/'common/config.json', root/'common/region.json',
                root/'common/compiled-native.json', root/'common/shape.json', root/'guides/hard_free.json',
                root/'guides/class.json', root/'probes.jsonl', root/'selected-probes.json',
                root/'execution-code/contact-line-guide-audit', root/'execution-code/tools/native_class_line_reference.py',
                root/'execution-code/native-definition/definition.json', Path(PYTHON)]
    for path in required: require(str(path) in plan['files'], 'Unbound required execution input: '+str(path))
    require(plan['files'][str(root/'execution-code/contact-line-guide-audit')] == plan['executable_sha256'],
            'Copied executable differs from validated executable')
    require(plan['jobs'] == canonical_jobs(root, allocation), 'Executable command/allocation binding changed')


def _termination_requested(signum, _frame):
    raise SystemExit('Controller received signal '+str(signum))


def _drain(child):
    """Kill the owned process group and reap despite repeated stop signals."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        while True:
            try: return child.wait()
            except KeyboardInterrupt: continue
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def validate_terminal(job, result, plan):
    require(result.get('complete') is True, 'Incomplete child result: '+job['id'])
    if job['phase'] == 'query':
        require(result.get('samples') == job['samples'] and result.get('probes') == job['probes'],
                'Query attempt allocation differs: '+job['id'])
        manifest = result['manifest']
        require(manifest['executable_sha256'] == plan['executable_sha256'] and
                manifest['seed'] == job['seed'] and manifest['samples'] == job['samples'],
                'Query execution binding differs: '+job['id'])
    else:
        require(result.get('passed') is True, 'Failed audit: '+job['id'])


def run(root):
    root = Path(root).absolute()
    plan = read(root/'execution-plan.json'); verify(root, plan)
    output = root/'execution'
    require(not output.exists(), 'Execution already claimed; no automatic retry')
    output.mkdir()
    birth = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    write(output/'claim.json', dict(pid=os.getpid(), birth_ticks=birth,
        plan_sha256=sha(root/'execution-plan.json'), started=time.time(), maximum_workers=1))
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): env[name] = '1'
    completed = []
    current = None
    child_drained = True
    previous_sigterm = signal.signal(signal.SIGTERM, _termination_requested)
    try:
        for ordinal, job in enumerate(plan['jobs']):
            current = job['id']; verify(root, plan)
            terminal = Path(job['terminal'])
            require(not terminal.exists(), 'Terminal result already exists')
            directory = output/f'{ordinal:02}-{current}'; directory.mkdir()
            write(directory/'attempt.json', dict(job=job, started=time.time(), retry=False))
            def child_limits():
                resource.setrlimit(resource.RLIMIT_CPU, (job['cpu_limit_seconds'], job['cpu_limit_seconds']+1))
                os.nice(10)
            started = time.monotonic(); child = None; error = None
            try:
                with (directory/'output.log').open('x') as log:
                    child = subprocess.Popen(job['argv'], cwd=plan['repository'], env=env,
                        stdout=log, stderr=subprocess.STDOUT, preexec_fn=child_limits, start_new_session=True)
                    child_drained = False
                    try: birth = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                    except FileNotFoundError: birth = None
                    write(directory/'process.json', dict(pid=child.pid, birth_ticks=birth, started=time.time()))
                    child.wait(timeout=job['wall_limit_seconds'])
                    child_drained = True
            except BaseException as caught:
                error = caught
                if child is not None:
                    _drain(child); child_drained = True
                raise
            finally:
                write(directory/'exit.json', dict(returncode=None if child is None else child.returncode,
                    wall_seconds=time.monotonic()-started, error=None if error is None else repr(error),
                    child_started=child is not None, child_drained=child_drained))
            require(child.returncode == 0, 'Child failed: '+current)
            require(terminal.is_file(), 'Missing terminal result: '+current)
            result = read(terminal); validate_terminal(job, result, plan)
            completed.append(dict(id=current, terminal=str(terminal), sha256=sha(terminal)))
            write(directory/'success.json', completed[-1])
        verify(root, plan)
        write(output/'summary.json', dict(complete=True, passed=True, completed=completed,
            plan_sha256=sha(root/'execution-plan.json'), fresh_draws=1024, saved_development_probes=40,
            new_physical_clouds=0, scope='Proposal geometry/density diagnostic only; no equilibrium weight or assembly conclusion'))
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, passed=False, failed_job=current,
            completed=completed, error=repr(error), traceback=traceback.format_exc(),
            child_drained=child_drained, remaining_jobs_not_started=True, retries=0))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
