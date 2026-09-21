#!/usr/bin/env python3
"""Run one frozen shoulder benchmark exactly once, then its archived observer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare_shoulder_docking_benchmark import validate_plan


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def check(campaign, journal, assessment):
    manifest = read(campaign/'manifest.json')
    assert manifest['schema'] == 'conditional-shoulder-docking-v1'
    assert manifest['binary_supplied']
    validate_plan(manifest, campaign)
    assert not journal.exists() and not assessment.exists() and not (campaign/'status.json').exists(), 'No restart or overwrite'
    for name, digest in manifest['input_sha256'].items():
        assert sha(campaign/'provenance'/name) == digest, name
    for job in manifest['jobs']:
        assert sha(job['config']) == job['config_sha256']
        assert read(job['config'])['seed'] == job['seed']
        assert not Path(job['directory']).exists() and not Path(job['log']).exists(), 'No replay of existing job'
        assert job['command'][0] == str(campaign/'provenance/docking-mc')
    assert len({job['id'] for job in manifest['jobs']}) == 12
    assert [job['seed'] for job in manifest['jobs']] == manifest['seeds']
    return manifest


def run(campaign, journal, assessment):
    manifest = check(campaign, journal, assessment)
    journal.mkdir(parents=True)
    shutil.copy2(__file__, journal/'runner.py')
    state = dict(complete=False, phase='physical', started=time.time(),
                 manifest_sha256=sha(campaign/'manifest.json'), runner_sha256=sha(__file__))
    status = dict(running=True, complete=False, jobs=[dict(id=j['id'], status='pending') for j in manifest['jobs']])
    write(journal/'status.json', state)
    write(campaign/'status.json', status)
    active, index, failed = {}, 0, False
    try:
        while active or (index < len(manifest['jobs']) and not failed):
            while not failed and index < len(manifest['jobs']) and len(active) < manifest['workers']:
                job = manifest['jobs'][index]
                # Recheck the exact frozen executable and per-job input at launch.
                assert sha(job['command'][0]) == manifest['binary_sha256']
                assert sha(job['config']) == job['config_sha256']
                log = open(job['log'], 'xb')
                try:
                    child = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                status['jobs'][index].update(status='running', pid=child.pid, started=time.time(), argv=job['command'], log=job['log'])
                active[index] = child
                print('Started', job['id'], 'PID', child.pid, flush=True)
                index += 1
                write(campaign/'status.json', status)
            for i, child in list(active.items()):
                result = child.poll()
                if result is None:
                    continue
                child.wait()
                status['jobs'][i].update(status='complete' if result == 0 else 'failed', exit_code=result, finished=time.time())
                print('Finished', status['jobs'][i]['id'], 'exit', result, flush=True)
                failed = failed or result != 0
                del active[i]
                write(campaign/'status.json', status)
            if active:
                time.sleep(.5)
    except BaseException as error:
        state['exception'] = repr(error)
        failed = True
        # Preserve launched children and drain them; never replay a population.
        for i, child in active.items():
            result = child.wait()
            status['jobs'][i].update(status='complete' if result == 0 else 'failed', exit_code=result, finished=time.time())
        state.update(phase='physical_failed', finished=time.time())
        write(journal/'status.json', state)
        raise
    finally:
        status.update(running=False, complete=not failed and index == len(manifest['jobs']))
        write(campaign/'status.json', status)
    if not status['complete']:
        state.update(phase='physical_failed', finished=time.time())
        write(journal/'status.json', state)
        raise RuntimeError('A frozen job failed; preserved outputs, no retry, no observer')
    command = [sys.executable, str(campaign/'provenance/analyze_shoulder_docking_benchmark.py'),
               str(campaign), '--out', str(assessment), '--workers', str(min(manifest['workers'], len(manifest['jobs'])))]
    state.update(phase='audit', physical_status_sha256=sha(campaign/'status.json'), analysis_command=command)
    write(journal/'status.json', state)
    with (journal/'analysis.log').open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    state.update(analysis_exit_code=result.returncode, complete=result.returncode == 0,
                 phase='complete' if result.returncode == 0 else 'audit_failed', finished=time.time())
    write(journal/'status.json', state)
    print('Observer exit', result.returncode, flush=True)
    result.check_returncode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.campaign, args.journal, args.assessment)]
    if args.preflight:
        manifest = check(*paths)
        print(json.dumps(dict(preflight=True, actual_kernel_executions=0, jobs=len(manifest['jobs']), workers=manifest['workers'])))
    else:
        run(*paths)
