#!/usr/bin/env python3
"""Freeze and dispatch the two existing independent R4 SMC controls.

The prerequisite is successful execution/auditing of the confirmation workflow,
not passage of its convergence tests. No frozen allocation is edited or retried.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from analyze_r4_smc_control import Ledger, parse_job_options, read, require, sha, validate_control, write
from prepare_shoulder_docking_benchmark import local_dependencies
from run_conditional_ray_campaign import execute_jobs

SCHEMA = 'dependent-r4-smc-controls-v1'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')}


def process_token(pid):
    """Linux process birth identity, avoiding a recycled-PID dependency."""
    try:
        value = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return None if value[0] == 'Z' else value[19]
    except FileNotFoundError:
        return None


def prerequisite_state(status, alive):
    """Convergence flags are deliberately not consulted here."""
    require(not status.get('error'), 'Prerequisite workflow failed')
    steps = status.get('steps', [])
    require(all(s.get('returncode') in (None, 0) for s in steps), 'Prerequisite command failed')
    if status.get('complete') is True:
        require(len(steps) == 2 and all(s.get('returncode') == 0 for s in steps),
                'Incomplete prerequisite terminal command ledger')
        return 'ready'
    require(alive, 'Prerequisite is incomplete and its original process is absent; no restart or launch')
    return 'waiting'


def control_jobs(contexts, out):
    """Interleave control labels, retaining each protocol's original population order."""
    jobs = []
    for index in range(4):
        for arm, ctx in zip(('broad', 'narrow'), contexts):
            job = ctx['protocol']['jobs'][index]
            jobs.append(dict(id=arm+'-'+job['id'], arm=arm, seed=job['seed'],
                command=job['argv'], directory=job['output'],
                log=str(Path(out)/('physical-'+arm+'-'+job['id']+'.log')), status='pending'))
    require(len({j['seed'] for j in jobs}) == 8 and len({j['directory'] for j in jobs}) == 8,
            'Controls overlap seeds or outputs')
    return jobs


def freeze(out, protocols, dependency, dependency_pid):
    out = Path(out).resolve(); dependency = Path(dependency).resolve()
    require(not out.exists(), 'Fresh workflow directory required')
    contexts = [validate_control(p) for p in protocols]
    require(len(contexts) == 2, 'Exactly two frozen controls required')
    jobs = control_jobs(contexts, out)
    require(all(not Path(j['directory']).exists() for j in jobs), 'A physical output already exists')
    birth = process_token(dependency_pid)
    require(birth is not None, 'A live prerequisite workflow is required at freeze')
    ledger = Ledger(); ledger.frozen(dependency)
    require(read(dependency/'status.json')['schema'] == 'contact-confirmation-workflow-status-v1',
            'Wrong prerequisite workflow')
    manifest = read(dependency/'manifest.json')
    require(len(manifest['steps']) == 2, 'Expected physical/audit plus classification prerequisite')
    out.mkdir(parents=True); common = out/'common'; common.mkdir()
    closure = local_dependencies([Path(__file__), Path(__file__).with_name('analyze_r4_smc_control.py')])
    for name, source in closure.items(): shutil.copy2(source, common/name)
    plan = dict(schema=SCHEMA, created=time.time(), repository=manifest['repository'],
        python=sys.executable, python_version=sys.version, python_sha256=sha(sys.executable),
        controls=[dict(protocol=c['protocol_path'], protocol_sha256=sha(c['protocol_path'])) for c in contexts],
        dependency=dict(directory=str(dependency), pid=dependency_pid, process_birth=birth,
                        source_bindings=ledger.files, steps=manifest['steps']),
        maximum_physical_workers=4, all_physical_cap=8, all_worker_cap=32, analysis_workers=4,
        thread_environment=THREAD_ENV, jobs=jobs, sources={name:sha(common/name) for name in closure},
        analysis_paths=[str(out/(arm+'-analysis')) for arm in ('broad','narrow')],
        stage_stride=16, dispatch='Run only after prerequisite commands complete successfully; scientific convergence may fail.',
        failure='No retry or output overwrite; stop new launches and drain active children.',
        scope='Independent R4 controls only. No full-vessel, assembly, or convergence authorization.')
    write(out/'plan.json', plan)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return plan


def validate_workflow(out):
    out = Path(out).resolve(); ledger = Ledger(); ledger.frozen(out)
    plan = read(out/'plan.json')
    require(plan['schema'] == SCHEMA and plan['maximum_physical_workers'] == 4
        and plan['all_physical_cap'] == 8 and plan['all_worker_cap'] == 32
        and plan['analysis_workers'] == 4 and plan['stage_stride'] == 16
        and plan['thread_environment'] == THREAD_ENV, 'Workflow allocation changed')
    require(sys.flags.optimize == 0 and sys.version == plan['python_version']
        and sha(sys.executable) == plan['python_sha256'], 'Auditor Python runtime differs')
    require(sha(__file__) == plan['sources']['run_r4_smc_controls.py'], 'Controller source changed')
    for name, digest in plan['sources'].items(): ledger.bind(out/'common'/name, digest)
    for name, digest in plan['dependency']['source_bindings'].items(): ledger.bind(name, digest)
    contexts = [validate_control(c['protocol'], c['protocol_sha256']) for c in plan['controls']]
    require(plan['jobs'] == control_jobs(contexts, out), 'Job allocation changed')
    return plan, contexts, ledger


def physical_preflight(plan, contexts):
    """Refuse overwrite and verify every byte again immediately before dispatch."""
    for context in contexts:
        for name, digest in context['bindings'].items(): require(sha(name) == digest, 'Control changed before launch')
    require(all(not Path(j['directory']).exists() for j in plan['jobs']), 'Existing SMC output; no retry')
    # Dependency completion accounts for its children. Additionally refuse any
    # other same-user physical executable left running, rather than guessing at
    # uncoordinated external worker capacity.
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            name = Path(os.fsdecode(argv[0])).name if argv and argv[0] else ''
            if name in ('latent-region-normalizer','latent-region-smc','basin-normalizer','tetramer-mc'):
                active.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    require(not active, 'Other physical workers are active: '+str(active))


def run(out):
    out = Path(out).resolve(); plan, contexts, ledger = validate_workflow(out)
    with (out/'status.json').open('x') as stream: stream.write('{}\n')
    state = dict(schema=SCHEMA, complete=False, phase='waiting_for_prerequisite',
        started=time.time(), plan_sha256=sha(out/'plan.json'), pid=os.getpid(), jobs=plan['jobs'], analyses=[])
    def snapshot():
        write(out/'status.tmp', state); (out/'status.tmp').replace(out/'status.json')
    snapshot()
    env = dict(os.environ, **THREAD_ENV, PYTHONOPTIMIZE='0')
    try:
        dep = plan['dependency']; status_path = Path(dep['directory'])/'status.json'
        while prerequisite_state(read(status_path), process_token(dep['pid']) == dep['process_birth']) == 'waiting':
            time.sleep(30)
        require(read(status_path)['steps'] == [dict(s, **{k:v for k,v in actual.items() if k not in s})
            for s, actual in zip(dep['steps'], read(status_path)['steps'])], 'Prerequisite command identity changed')
        ledger.recheck(); physical_preflight(plan, contexts)
        state.update(phase='physical', prerequisite_status_sha256=sha(status_path)); snapshot()
        execute_jobs(state['jobs'], snapshot, workers=4,
            popen=lambda *a, **k: subprocess.Popen(*a, **k, cwd=plan['repository'], env=env))
        state.update(phase='auditing_and_classifying'); snapshot()
        for control, target in zip(plan['controls'], plan['analysis_paths']):
            command = [plan['python'], str(out/'common/analyze_r4_smc_control.py'), 'analyze',
                '--protocol', control['protocol'], '--expected-protocol-sha256', control['protocol_sha256'],
                '--out', target, '--workers','4','--stage-stride','16']
            with (out/(Path(target).name+'.log')).open('xb') as log:
                result = subprocess.run(command, cwd=plan['repository'], env=env, stdout=log, stderr=subprocess.STDOUT)
            state['analyses'].append(dict(command=command, returncode=result.returncode)); snapshot(); result.check_returncode()
        state.update(phase='comparing'); snapshot()
        command = [plan['python'], str(out/'common/analyze_r4_smc_control.py'), 'compare',
            '--left', str(Path(plan['analysis_paths'][0])/'analysis.json'),
            '--right', str(Path(plan['analysis_paths'][1])/'analysis.json'), '--out', str(out/'comparison')]
        with (out/'comparison.log').open('xb') as log:
            result = subprocess.run(command, cwd=plan['repository'], env=env, stdout=log, stderr=subprocess.STDOUT)
        state['comparison'] = dict(command=command, returncode=result.returncode); snapshot(); result.check_returncode()
        ledger.recheck(); state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(complete=False, phase='failed', error=repr(error), finished=time.time()); snapshot(); raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze'); f.add_argument('--out', type=Path, required=True)
    f.add_argument('--protocol', type=Path, action='append', required=True)
    f.add_argument('--dependency', type=Path, required=True); f.add_argument('--dependency-pid', type=int, required=True)
    r = commands.add_parser('run'); r.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.out,args.protocol,args.dependency,args.dependency_pid) if args.command == 'freeze' else run(args.out)
    print(result.get('phase', result['schema']), flush=True)


if __name__ == '__main__': main()
