#!/usr/bin/env python3
"""Dispatch a fixed validated conditional benchmark; no retries or replacements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def checked(binding):
    path = Path(binding['path'])
    if sha(path) != binding['sha256']:
        raise ValueError('Changed bound file: '+str(path))
    return path


def launch(base, review_path, workers):
    base, review_path = base.resolve(), review_path.resolve()
    review = read(review_path)
    if not review.get('passed') or not review.get('complete'):
        raise ValueError('Prelaunch review did not pass')
    required = [base/'config.json', base/'run-binding.json', base/'binding.json',
                base/'common/evolving_dimer_benchmark', Path(__file__).resolve()]
    if any(str(path.resolve()) not in review['input_sha256'] for path in required):
        raise ValueError('Prelaunch review omits an execution closure input')
    for path, digest in review['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed reviewed input: '+path)
    if not 1 <= workers <= 3:
        raise ValueError('At most three new physical workers')
    config = read(base/'config.json')
    binding = read(base/'run-binding.json')
    checked(binding['prepared_manifest'])
    checked(binding['prelaunch_binding'])
    if binding['config_sha256'] != sha(base/'config.json'):
        raise ValueError('Config/run binding mismatch')
    controller = base/'dispatch'
    controller.mkdir()
    plan = dict(schema='evolving-dimer-dispatch-v1', base=str(base), workers=workers,
        review=dict(path=str(review_path), sha256=sha(review_path)),
        config_sha256=sha(base/'config.json'), binding_sha256=sha(base/'run-binding.json'),
        executable_sha256=sha(base/'common/evolving_dimer_benchmark'),
        source_sha256=sha(Path(__file__)), jobs=config['jobs'], retry=False,
        failure='Stop launching and drain every already-running child. No replacements.',
        physical_scope='Conditional two-mobile-body relaxation, not assembly.')
    save(controller/'plan.json', plan)
    frozen = controller/'run.py'
    frozen.write_bytes(Path(__file__).read_bytes())
    with (controller/'controller.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', str(frozen), '--run', '--base', str(base)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(controller/'launch.json', dict(pid=process.pid, launched=time.time(), plan_sha256=sha(controller/'plan.json')))
    print(json.dumps(dict(pid=process.pid, jobs=len(plan['jobs']), workers=workers, directory=str(controller))))


def run(base):
    base = base.resolve()
    out = base/'dispatch'
    plan = read(out/'plan.json')
    if plan['source_sha256'] != sha(Path(__file__)):
        raise ValueError('Controller source changed')
    for key, path in [('config_sha256',base/'config.json'),('binding_sha256',base/'run-binding.json'),
                      ('executable_sha256',base/'common/evolving_dimer_benchmark')]:
        if sha(path) != plan[key]:
            raise ValueError('Execution closure changed')
    checked(plan['review'])
    execution_output = Path(read(base/'config.json')['output'])
    claim = out/'claimed.json'
    with claim.open('x') as handle:
        json.dump(dict(pid=os.getpid(), started=time.time()), handle)
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:
        env[key] = '1'
    queue = list(plan['jobs'])
    active = {}
    completed = []
    failed = False
    while queue or active:
        while queue and len(active) < plan['workers'] and not failed:
            job = queue.pop(0)
            started = time.time()
            log = None
            try:
                log = (out/f"job-{job['id']:03}.log").open('xb')
                command = [str(base/'common/evolving_dimer_benchmark'), '--mode','run',
                    '--config',str(base/'config.json'),'--binding',str(base/'run-binding.json'),
                    '--job',str(job['id'])]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[job['id']] = (process, log, job, started)
            except Exception as error:
                if log is not None:
                    log.close()
                completed.append(dict(job=job,pid=None,started=started,finished=time.time(),
                    returncode=None,success=False,launch_failure=f'{type(error).__name__}: {error}'))
                failed = True
        for identity, (process, log, job, started) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            terminal = execution_output/f'job-{identity:03}'/'terminal.json'
            terminal_error = None
            terminal_sha = None
            try:
                terminal_sha = sha(terminal) if terminal.exists() else None
                success = code == 0 and terminal.is_file() and read(terminal).get('complete') is True
            except Exception as error:
                success = False
                terminal_error = f'{type(error).__name__}: {error}'
            completed.append(dict(job=job, pid=process.pid, started=started, finished=time.time(),
                returncode=code, success=success, terminal_sha256=terminal_sha,
                terminal_error=terminal_error))
            failed |= not success
            del active[identity]
        done = not active and (failed or not queue)
        save(out/'status.json', dict(schema='evolving-dimer-dispatch-status-v1', complete=done,
            passed=done and not failed, failure_draining=failed and bool(active),
            completed=completed, active=[dict(job=item[2],pid=item[0].pid,started=item[3]) for item in active.values()],
            unstarted=queue, updated=time.time(), plan_sha256=sha(out/'plan.json')))
        if done:
            break
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:
        run(args.base)
    else:
        if args.review is None:
            parser.error('--review required for launch')
        launch(args.base,args.review,args.workers)


if __name__=='__main__':
    main()
