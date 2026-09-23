#!/usr/bin/env python3
"""Freeze two read-only comparisons after a completed independent R4 SMC workflow.

Only the existing SMC-versus-importance comparator is dispatched. Scientific
agreement and confirmation_passed are results, never prerequisites for diagnosis.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy
from analyze_r4_smc_control import Ledger, read, require, sha, write
from prepare_shoulder_docking_benchmark import local_dependencies


SCHEMA = 'smc-importance-bridge-v1'
DEPENDENCY_SCHEMA = 'dependent-r4-smc-controls-v1'
ARMS = ('broad', 'narrow')
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')}


def process_token(pid):
    """Linux start ticks distinguish the original process from a recycled PID."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def runtime():
    return dict(python=sys.executable, python_version=sys.version,
        python_sha256=sha(sys.executable), numpy=np.__version__, scipy=scipy.__version__)


def dependency_contract(directory, plan):
    """Reconstruct the scheduler's exact two audits and terminal comparison."""
    directory = Path(directory).resolve()
    require(plan['schema'] == DEPENDENCY_SCHEMA and len(plan['controls']) == 2
        and len(plan['jobs']) == 8 and plan['analysis_workers'] == 4
        and plan['stage_stride'] == 16, 'Unexpected SMC workflow allocation')
    paths = [str(directory / (arm + '-analysis')) for arm in ARMS]
    require(plan['analysis_paths'] == paths, 'Unexpected SMC analysis destinations')
    fields = ('id', 'arm', 'seed', 'command', 'directory', 'log')
    jobs = [{key: job[key] for key in fields} for job in plan['jobs']]
    require(len({job['id'] for job in jobs}) == 8
        and len({job['seed'] for job in jobs}) == 8
        and all(sum(job['arm'] == arm for job in jobs) == 4 for arm in ARMS),
        'Incomplete independent SMC allocation')
    analyzer = str(directory / 'common/analyze_r4_smc_control.py')
    analyses = [[plan['python'], analyzer, 'analyze', '--protocol', control['protocol'],
        '--expected-protocol-sha256', control['protocol_sha256'], '--out', target,
        '--workers', '4', '--stage-stride', '16']
        for control, target in zip(plan['controls'], paths)]
    comparison = [plan['python'], analyzer, 'compare', '--left', str(Path(paths[0]) / 'analysis.json'),
        '--right', str(Path(paths[1]) / 'analysis.json'), '--out', str(directory / 'comparison')]
    return dict(jobs=jobs, analyses=analyses, comparison=comparison)


def prerequisite_state(status, dependency, alive):
    """Require successful execution, irrespective of scientific pass flags."""
    require(status.get('schema') == DEPENDENCY_SCHEMA
        and status.get('plan_sha256') == dependency['plan_sha256']
        and status.get('pid') == dependency['pid'], 'Prerequisite workflow identity changed')
    require(not status.get('error') and status.get('phase') != 'failed', 'Prerequisite workflow failed')
    expected = dependency['contract']
    jobs = status.get('jobs', [])
    require(len(jobs) == len(expected['jobs']) and all(
        all(actual.get(key) == value for key, value in wanted.items())
        for actual, wanted in zip(jobs, expected['jobs'])), 'Prerequisite job identities changed')
    require(all(job.get('returncode') in (None, 0)
        and job.get('status') in ('pending', 'running', 'complete') for job in jobs),
        'Prerequisite physical command failed')
    analyses = status.get('analyses', [])
    require(len(analyses) <= 2 and all(item.get('command') == command
        and item.get('returncode') == 0 for item, command in zip(analyses, expected['analyses'])),
        'Prerequisite audit failed or changed command')
    comparison = status.get('comparison')
    require(comparison is None or (comparison.get('command') == expected['comparison']
        and comparison.get('returncode') == 0), 'Prerequisite comparison failed or changed command')
    if status.get('complete') is True:
        require(status.get('phase') == 'complete' and len(analyses) == 2 and comparison is not None
            and all(job.get('status') == 'complete' and job.get('returncode') == 0 for job in jobs),
            'Incomplete prerequisite terminal ledger')
        return 'ready'
    require(alive, 'Prerequisite is unfinished and its original process is absent; no restart')
    return 'waiting'


def comparison_steps(out, dependency, importance, python):
    return [dict(id=arm, smc=str(Path(dependency) / (arm + '-analysis/analysis.json')),
        output=str(Path(out) / (arm + '-comparison')), command=[python, '-B', '-E',
            str(Path(out) / 'common/compare_r4_smc_importance.py'), '--smc',
            str(Path(dependency) / (arm + '-analysis/analysis.json')), '--importance', str(importance),
            '--out', str(Path(out) / (arm + '-comparison'))]) for arm in ARMS]


def bind_importance(ledger, path, expected):
    path = Path(path).resolve()
    ledger.frozen(path.parent)
    analysis = read(ledger.bind(path, expected))
    state = read(ledger.bind(path.parent / 'status.json'))
    require(analysis['schema'] == 'contact-confirmation-comparison-v1'
        and analysis['complete'] is True and state['complete'] is True
        and state['analysis_sha256'] == expected, 'Completed importance analysis required')
    ledger.bind(path.parent / 'provenance/campaign-protocol.json', analysis['protocol_sha256'])
    campaign = Path(analysis['campaign'])
    ledger.bind(campaign / 'status.json', analysis['status_sha256'])
    config = read(ledger.bind(campaign / 'bank/provenance/config.json'))
    ledger.bind(campaign / 'bank/provenance/region.json', analysis['region_sha256'])
    ledger.bind(config['shape'], analysis['shape_sha256'])


def freeze(out, dependency, dependency_sha256, dependency_pid, dependency_birth,
           importance, importance_sha256):
    out = Path(out).resolve(); dependency = Path(dependency).resolve(); importance = Path(importance).resolve()
    require(not out.exists(), 'Fresh bridge directory required')
    require(sys.flags.optimize == 0, 'Python optimization must be disabled')
    ledger = Ledger(); ledger.frozen(dependency)
    source_plan = read(ledger.bind(dependency / 'plan.json', dependency_sha256))
    dep = dict(directory=str(dependency), plan_sha256=dependency_sha256, pid=dependency_pid,
        process_birth=str(dependency_birth), contract=dependency_contract(dependency, source_plan))
    state_path = dependency / 'status.json'
    prerequisite_state(read(state_path), dep, process_token(dependency_pid) == dep['process_birth'])
    dep['status_at_freeze_sha256'] = sha(state_path)  # Mutable until successful completion.
    bind_importance(ledger, importance, importance_sha256)
    source = Path(__file__).resolve(); comparator = source.with_name('compare_r4_smc_importance.py')
    closure = local_dependencies([source, comparator])
    original_sources = {str(path): sha(path) for path in closure.values()}
    out.mkdir(parents=True); common = out / 'common'; common.mkdir()
    for name, path in closure.items(): shutil.copy2(path, common / name)
    sources = {name: sha(common / name) for name in closure}
    require(all(sources[Path(path).name] == digest for path, digest in original_sources.items()),
        'Source changed while archiving')
    plan = dict(schema=SCHEMA, created=time.time(), repository=source_plan['repository'],
        runtime=runtime(), thread_environment=THREAD_ENV, physical_workers=0, comparison_workers=1,
        dependency=dep, importance=dict(path=str(importance), sha256=importance_sha256),
        input_bindings=ledger.files, original_sources=original_sources, sources=sources,
        comparator_sources={name: sources[name] for name in local_dependencies([comparator])},
        steps=comparison_steps(out, dependency, importance, sys.executable), poll_seconds=30,
        scope='Independent completed-artifact diagnosis only; no physics, reclassification, audit replay, '
            'pooling, full-vessel dispatch, or assembly conclusion. Scientific pass flags never gate dispatch.',
        failure='Stop on failed or abandoned prerequisite, changed bytes, or failed comparison; no retry or overwrite.')
    ledger.recheck()
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', dict(files={str(path.relative_to(out)): sha(path)
        for path in out.rglob('*') if path.is_file()}))
    return plan


def validate_bridge(out, expected_plan_sha256):
    ledger = Ledger(); ledger.frozen(out)
    plan = read(ledger.bind(out / 'plan.json', expected_plan_sha256))
    require(plan['schema'] == SCHEMA and plan['physical_workers'] == 0
        and plan['comparison_workers'] == 1 and plan['poll_seconds'] == 30
        and plan['thread_environment'] == THREAD_ENV, 'Bridge allocation changed')
    require(sys.flags.optimize == 0 and plan['runtime'] == runtime(), 'Frozen comparison runtime differs')
    require(Path(__file__).resolve() == out / 'common/run_smc_importance_bridge.py',
        'Run the frozen bridge source in common/')
    for name, digest in plan['sources'].items(): ledger.bind(out / 'common' / name, digest)
    for name, digest in plan['input_bindings'].items(): ledger.bind(name, digest)
    dep = plan['dependency']; dependency = Path(dep['directory'])
    source_plan = read(ledger.bind(dependency / 'plan.json', dep['plan_sha256']))
    require(dep['contract'] == dependency_contract(dependency, source_plan), 'Dependency command contract changed')
    require(plan['steps'] == comparison_steps(out, dependency, plan['importance']['path'], sys.executable),
        'Comparison commands changed')
    require(plan['comparator_sources'] == {name: sha(path) for name, path in
        local_dependencies([out / 'common/compare_r4_smc_importance.py']).items()},
        'Comparator source closure changed')
    return plan, source_plan, ledger


def bind_terminal(dependency, source_plan, ledger):
    """Bind completed analyses and their comparison; never inspect partial weights."""
    root = Path(dependency['directory'])
    state_path = ledger.bind(root / 'status.json')
    require(prerequisite_state(read(state_path), dependency, False) == 'ready', 'Terminal dependency required')
    analyses = {}
    for arm, control in zip(ARMS, source_plan['controls']):
        target = root / (arm + '-analysis'); ledger.frozen(target)
        path = ledger.bind(target / 'analysis.json'); value = read(path)
        state = read(ledger.bind(target / 'status.json'))
        require(value['schema'] == 'smc-r4-control-analysis-v1' and value['complete'] is True
            and state['complete'] is True and state['analysis_sha256'] == sha(path)
            and value['protocol_sha256'] == control['protocol_sha256'], 'SMC terminal analysis binding differs')
        analyses[str(path)] = sha(path)
    ledger.frozen(root / 'comparison')
    comparison = read(ledger.bind(root / 'comparison/comparison.json'))
    require(comparison['schema'] == 'smc-r4-control-comparison-v1'
        and comparison['source_populations_kept_separate'] is True
        and all(comparison['source_sha256'].get(path) == digest for path, digest in analyses.items()),
        'Terminal SMC comparison does not bind both completed analyses')
    ledger.recheck()
    return dict(status_sha256=sha(state_path), analyses=analyses,
        comparison_sha256=sha(root / 'comparison/comparison.json'))


def validate_result(step, plan, ledger):
    target = Path(step['output']); ledger.frozen(target)
    path = ledger.bind(target / 'analysis.json'); value = read(path)
    require(value['schema'] == 'r4-smc-importance-comparison-v1' and value['complete'] is True
        and value['populations_pooled'] is False
        and value['inputs'] == dict(smc=step['smc'], importance=plan['importance']['path'])
        and value['input_sha256'].get(step['smc']) == ledger.files[step['smc']]
        and value['input_sha256'].get(plan['importance']['path']) == plan['importance']['sha256'],
        'Comparison output input bindings differ')
    expected_sources = {str(target.parent / 'common' / name): digest
        for name, digest in plan['comparator_sources'].items()}
    require(value['source_sha256'] == expected_sources, 'Comparison used an unexpected source closure')
    ledger.recheck()
    return dict(analysis_sha256=sha(path), arms={name: dict(
        primary_Qz_mass_agreement=arm['primary_Qz_mass_agreement'],
        significant_primary_strata_agree=arm['significant_primary_strata_agree'])
        for name, arm in value['arms'].items()})


def run(out, expected_plan_sha256):
    out = Path(out).resolve()
    with (out / 'status.json').open('x') as stream: stream.write('{}\n')
    state = dict(schema=SCHEMA, complete=False, phase='validating', pid=os.getpid(),
        started=time.time(), plan_sha256=expected_plan_sha256, physical_workers=0, steps=[])
    def snapshot():
        write(out / 'status.tmp', state); (out / 'status.tmp').replace(out / 'status.json')
    snapshot()
    try:
        plan, source_plan, ledger = validate_bridge(out, expected_plan_sha256)
        for step in plan['steps']:
            require(not Path(step['output']).exists() and not (out / (step['id'] + '.log')).exists(),
                'Existing comparison output or log; no retry or overwrite')
        dep = plan['dependency']; state.update(phase='waiting_for_prerequisite'); snapshot()
        while prerequisite_state(read(Path(dep['directory']) / 'status.json'), dep,
                process_token(dep['pid']) == dep['process_birth']) == 'waiting':
            time.sleep(plan['poll_seconds'])
        ledger.recheck()
        state.update(prerequisite=bind_terminal(dep, source_plan, ledger), phase='comparing'); snapshot()
        env = dict(os.environ, **THREAD_ENV, PYTHONOPTIMIZE='0', PYTHONDONTWRITEBYTECODE='1')
        for step in plan['steps']:
            ledger.recheck()
            require(not Path(step['output']).exists(), 'Existing comparison output; no retry or overwrite')
            item = dict(step, started=time.time(), returncode=None); state['steps'].append(item); snapshot()
            with (out / (step['id'] + '.log')).open('xb') as log:
                result = subprocess.run(step['command'], cwd=plan['repository'], env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            item.update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
            item.update(validate_result(step, plan, ledger)); snapshot()
        ledger.recheck(); state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(complete=False, phase='failed', error=repr(error), finished=time.time()); snapshot(); raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze'); f.add_argument('--out', type=Path, required=True)
    f.add_argument('--dependency', type=Path, required=True)
    f.add_argument('--expected-dependency-plan-sha256', required=True)
    f.add_argument('--dependency-pid', type=int, required=True); f.add_argument('--dependency-birth', required=True)
    f.add_argument('--importance', type=Path, required=True); f.add_argument('--expected-importance-sha256', required=True)
    r = commands.add_parser('run'); r.add_argument('--out', type=Path, required=True)
    r.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.out, args.dependency, args.expected_dependency_plan_sha256,
            args.dependency_pid, args.dependency_birth, args.importance, args.expected_importance_sha256)
        print(dict(schema=result['schema'], plan_sha256=sha(args.out / 'plan.json'), physics_launched=False), flush=True)
    else:
        print(run(args.out, args.expected_plan_sha256)['phase'], flush=True)


if __name__ == '__main__': main()
