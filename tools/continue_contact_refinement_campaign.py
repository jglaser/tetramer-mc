#!/usr/bin/env python3
"""Frozen, singleton continuation of the paired contact-refinement pilot.

Wait for the already running allocation; never rerun it. Analyze all slots,
fit both arms with fixed gates, conditionally replace covariances and benchmark,
then run the independent predeclared FFT-repair audit. No equilibrium claim.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from merge_refined_contact_atlas import frozen_input
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_refinement_campaign import (ARMS, checked_path, read, require, sha,
                                             validate as validate_main, write)
from run_conditional_ray_campaign import execute_jobs

SCHEMA = 'contact-refinement-frozen-continuation-v1'
TOOLS = ('analyze_contact_refinement_campaign.py', 'fit_depletion_contact_atlas.py',
         'merge_refined_contact_atlas.py', 'benchmark_depletion_contact_atlas.py')
FIT_FLAGS = ['--initial-weight', '0', '--shrinkage', '.02', '--covariance-floor', '1e-6',
             '--minimum-unique-poses', '32', '--minimum-empirical-rank', '6', '--minimum-ess', '0']


def python_command():
    # A venv's interpreter is commonly a symlink to its base installation.
    # Resolving it drops the neighboring pyvenv.cfg and its scientific packages.
    return os.path.abspath(sys.executable)


def verify_fft(root):
    root = Path(root).resolve()
    plan = read(root/'plan.json')
    require(plan.get('schema') == 'fft-subset-exact-recovery-v1'
            and plan.get('native_information') is False, 'Unexpected FFT repair allocation')
    require(1 <= plan['worker_limit'] <= 4, 'FFT repair worker limit exceeds continuation cap')
    for row in plan['frozen'].values():
        require(sha(row['path']) == row['sha256'], 'Frozen FFT input changed: '+row['path'])
    return plan


def prepare(args):
    out, main, base, fft = map(lambda p: Path(p).resolve(),
                             (args.out, args.campaign, args.base_fit, args.fft_prepared))
    require(not out.exists(), 'Fresh continuation directory required; no overwrite')
    require(1 <= args.workers <= 4 and args.wait_seconds > 0, 'Invalid bounded continuation allocation')
    plan = validate_main(main)
    require(plan['replicates'] == 2, 'This continuation uses two frozen independent streams')
    model, _, base_payload = frozen_input(base)
    fft_plan = verify_fft(fft)
    require(not (fft/'slots').exists(), 'FFT repair already started; do not schedule a duplicate')
    config_path = Path(args.config).resolve()
    cfg = read(config_path)
    require(not cfg.get('contact_memory') and not cfg.get('reversible_jump') and not cfg.get('conditional_closure'),
            'Require a frozen-proposal assembly benchmark config')
    bath = plan['bath']
    require(cfg['depletant_radius'] == bath['radius'] and cfg['reservoir_density'] == bath['activity'],
            'Assembly config and refinement bath differ')
    shape_path = Path(cfg['shape'])
    shape_path = (shape_path if shape_path.is_absolute() else config_path.parent/shape_path).resolve()
    require(sha(shape_path) == model['base_model']['shape_sha256'] == sha(main/'inputs/shape.json'),
            'Shape mismatch among old atlas, campaign and assembly config')
    binary_inputs = dict(audit=Path(args.audit_binary).resolve(), sampler=Path(args.sampler_binary).resolve())
    for name, path in binary_inputs.items():
        require(path.is_file() and os.access(path, os.X_OK), 'Missing executable: '+name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    inputs = out/'inputs'
    inputs.mkdir()
    (out/'logs').mkdir()
    (out/'stages').mkdir()
    (inputs/'base-fit-receipt').mkdir()
    for name, raw in base_payload.items():
        (inputs/'base-fit-receipt'/name).write_bytes(raw)
    shutil.copy2(main/'plan.json', inputs/'main-plan.json')
    shutil.copy2(main/'freeze.json', inputs/'main-freeze.json')
    shutil.copy2(fft/'plan.json', inputs/'fft-plan.json')
    shutil.copy2(config_path, inputs/'source-config.json')
    for key in ('shape', 'monomer_shape', 'native_pair_motifs'):
        if not cfg.get(key):
            continue
        path = Path(cfg[key])
        path = (path if path.is_absolute() else config_path.parent/path).resolve()
        destination = inputs/(key+'.json')
        shutil.copy2(path, destination)
        cfg[key] = str(destination)
    write(inputs/'benchmark-config.json', cfg, exclusive=True)
    for name, path in binary_inputs.items():
        shutil.copy2(path, inputs/(name+'-binary'))
    tool_root = Path(__file__).resolve().parent
    dependencies = local_dependencies([Path(__file__), *[tool_root/name for name in TOOLS]])
    (inputs/'python').mkdir()
    for name, path in dependencies.items():
        shutil.copy2(path, inputs/'python'/name)
    fixtures = []
    fixed_pose_sources = [Path(path).resolve() for path in args.fixed_poses]
    if args.neighborhoods:
        neighborhood_root = Path(args.neighborhoods).resolve()
        receipt = read(neighborhood_root/'manifest.json')
        require(receipt.get('complete') is True and receipt.get('evaluation_only') is True
                and receipt['shape_sha256'] == sha(shape_path), 'Invalid evaluation-neighborhood receipt')
        require(receipt['source_config_sha256'] == sha(config_path), 'Neighborhood and benchmark source configs differ')
        for name, digest in receipt['files'].items():
            require(sha(checked_path(neighborhood_root, name)) == digest, 'Frozen neighborhood changed: '+name)
        shutil.copy2(neighborhood_root/'manifest.json', inputs/'neighborhoods-manifest.json')
        fixed_pose_sources.extend(checked_path(neighborhood_root, row['file']) for row in receipt['neighborhoods'])
    for index, source in enumerate(fixed_pose_sources):
        destination = inputs/f'fixed-poses-{index:02d}.json'
        shutil.copy2(source, destination)
        fixtures.append(dict(name=f'scaffold-{index:02d}', path=str(destination), source=str(source), sha256=sha(destination)))
    mappings = []
    for replicate in range(plan['replicates']):
        for group in range(plan['groups']):
            job = next(j for j in plan['jobs'] if j['arm'] == 'adaptive-pivot'
                       and j['replicate'] == replicate and j['group'] == group)
            mappings.extend(row['original_global_slot'] for row in job['slot_map'])
    write(inputs/'replacement-mapping.json', mappings, exclusive=True)
    protocol = dict(schema=SCHEMA, main_campaign=str(main), base_fit=str(base), fft_prepared=str(fft),
                    main_plan_sha256=sha(main/'plan.json'), main_freeze_sha256=sha(main/'freeze.json'),
                    base_fit_freeze_sha256=sha(base/'freeze.json'), base_model_sha256=sha(base/'model.json'),
                    fft_plan_sha256=sha(fft/'plan.json'), workers=args.workers,
                    maximum_wait_seconds=args.wait_seconds, python=python_command(),
                    python_environment=dict(prefix=sys.prefix, base_prefix=sys.base_prefix, version=sys.version),
                    fit_flags=FIT_FLAGS, source_slot_mapping=mappings,
                    fit_order='replicate then group, within-job recorded slot order; both arms use every chain',
                    main_failure='Never rerun main. Skip dependent stages; only run independent FFT repair after all tracked main workers have exited.',
                    gate_failure='Record whole-fit rejection; no dropped slots, retries, threshold changes, or production model.',
                    audit=dict(draws_per_component=256, rd=bath['radius'], score_points=0,
                               seed=202609283101, fixtures=fixtures),
                    benchmark=dict(sweeps=200, sample_every=100, streams=2, seed=202609283701),
                    fft=dict(slots=fft_plan['all_slot_denominator'], command=[python_command(),
                        fft_plan['frozen']['controller']['path'], 'run', '--prepared', str(fft), '--workers', str(args.workers)]),
                    input_sha256={str(p.relative_to(out)): sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()},
                    scope='Frozen proposal validation and bounded accessibility benchmark. Native labels are absent from refinement and fitting; '
                          'the supplied assembly config and optional fixed scaffolds are evaluation contexts and may contain a native seed. '
                          'No physical stability, equilibrium assembly or kinetic-rate conclusion follows from this allocation.')
    write(out/'plan.json', protocol, exclusive=True)
    write(out/'freeze.json', dict(plan_sha256=sha(out/'plan.json'), files=protocol['input_sha256']), exclusive=True)
    validate(out)
    return dict(prepared=True, launched=False, out=str(out), main_campaign=str(main), fit_sources_per_arm=len(mappings),
                conditional_production_streams=4, fft_slots=fft_plan['all_slot_denominator'])


def validate(out):
    out = Path(out).resolve()
    plan, freeze = read(out/'plan.json'), read(out/'freeze.json')
    require(plan['schema'] == SCHEMA and sha(out/'plan.json') == freeze['plan_sha256'], 'Continuation plan changed')
    require(plan['input_sha256'] == freeze['files'], 'Continuation input list changed')
    for name, digest in freeze['files'].items():
        require(sha(checked_path(out, name)) == digest, 'Frozen continuation input changed: '+name)
    main, base, fft = Path(plan['main_campaign']), Path(plan['base_fit']), Path(plan['fft_prepared'])
    require(sha(main/'plan.json') == plan['main_plan_sha256'] and sha(main/'freeze.json') == plan['main_freeze_sha256'],
            'Main campaign allocation changed')
    require(sha(base/'freeze.json') == plan['base_fit_freeze_sha256'] and sha(base/'model.json') == plan['base_model_sha256'],
            'Original frozen atlas changed')
    require(sha(fft/'plan.json') == plan['fft_plan_sha256'], 'FFT repair allocation changed')
    return plan


def process_alive(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rfind(')')+2:].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_for_main(plan, snapshot, pause=time.sleep, monotonic=time.monotonic):
    main = Path(plan['main_campaign'])
    begin, idle_since = monotonic(), None
    while monotonic()-begin <= plan['maximum_wait_seconds']:
        try:
            status = read(main/'status.json')
        except (FileNotFoundError, json.JSONDecodeError):
            pause(2.)
            continue
        jobs = status['jobs']
        active = [job for job in jobs if job['status'] == 'running' and job.get('pid') and process_alive(job['pid'])]
        snapshot(dict(phase='waiting_for_main', active_main_jobs=[job['id'] for job in active],
                      completed_main_jobs=sum(job['status'] == 'complete' for job in jobs),
                      allocated_main_jobs=len(jobs)))
        if (main/'completion.json').exists() and not active:
            completion = read(main/'completion.json')
            require(completion.get('complete') is True
                    and completion['plan_sha256'] == plan['main_plan_sha256']
                    and all(job['status'] == 'complete' for job in jobs), 'Invalid main completion receipt')
            return True
        if not active:
            if any(job['status'] in ('failed', 'not_started') for job in jobs):
                return False
            idle_since = monotonic() if idle_since is None else idle_since
            if monotonic()-idle_since >= 60.:
                return False
        else:
            idle_since = None
        pause(5.)
    raise TimeoutError('Main campaign did not release all workers before the frozen wait limit')


def gate_rejection(directory, expected_count):
    marker, metrics = directory/'freeze-rejected.json', directory/'fit-metrics.json'
    if not marker.exists() or not metrics.exists() or (directory/'model.json').exists():
        return False
    rejected, report = read(marker), read(metrics)
    require(rejected.get('model_written') is False and rejected.get('all_slots_preserved') is True,
            'Fit failure is not a preserved whole-fit rejection')
    require(report.get('independent_slots') == expected_count and len(report.get('slots', [])) == expected_count,
            'Gate failure lost slots')
    require(report.get('exploration_gate_passed') is False, 'Missing failed fit gate')
    requested = rejected.get('options', {})
    require(requested.get('initial_weight') == 0 and requested.get('shrinkage') == .02
            and requested.get('covariance_floor') == 1e-6 and requested.get('minimum_unique_poses') == 32
            and requested.get('minimum_empirical_rank') == 6 and requested.get('minimum_ess') == 0,
            'Fit rejection thresholds changed')
    return True


class Stages:
    def __init__(self, out, plan, snapshot):
        self.out, self.plan, self.snapshot = out, plan, snapshot
        self.records = []

    def run(self, name, command, outputs=(), allowed_gate=None):
        log = self.out/'logs'/(name+'.log')
        record = dict(name=name, command=command, status='running', started=time.time())
        self.records.append(record)
        self.snapshot(dict(phase=name, stages=self.records))
        child, interruption = None, None
        try:
            with log.open('xb') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, **{k:'1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                                            'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')}))
                record['pid'] = child.pid
                self.snapshot(dict(phase=name, stages=self.records))
                try:
                    code = child.wait()
                except BaseException as error:
                    interruption = error
                    while True:
                        try:
                            code = child.wait()
                            break
                        except KeyboardInterrupt:
                            pass
            if interruption is not None:
                raise interruption
            rejected = code != 0 and allowed_gate is not None and allowed_gate()
            record.update(status='gate_rejected' if rejected else 'complete' if code == 0 else 'failed',
                          returncode=code, finished=time.time(), log_sha256=sha(log),
                          output_sha256={str(path):sha(path) for path in outputs if path.is_file()})
            write(self.out/'stages'/(name+'.json'), record, exclusive=True)
            self.snapshot(dict(phase=name, stages=self.records))
            require(code == 0 or rejected, f'Continuation stage {name} failed; no retry')
            return not rejected
        except BaseException as error:
            record.update(status='failed', finished=time.time(), error=repr(error))
            receipt = self.out/'stages'/(name+'.json')
            if not receipt.exists():
                write(receipt, record, exclusive=True)
            self.snapshot(dict(phase=name, stages=self.records))
            raise

    def audit_batch(self, jobs):
        """Bounded independent audits; stop starting after failure and drain."""
        name = 'geometry-audits'
        record = dict(name=name, jobs=jobs, status='running', started=time.time())
        self.records.append(record)

        def snapshot():
            self.snapshot(dict(phase=name, stages=self.records))

        snapshot()
        try:
            execute_jobs(jobs, snapshot, workers=self.plan['workers'])
            for job in jobs:
                root = Path(job['directory'])
                summary = read(root/'summary.json')
                require(summary.get('complete') is True, 'Geometry audit incomplete')
                require(sha(root/'draws.jsonl') == summary['draws_sha256'], 'Geometry audit draws changed')
                require(all(row['draws'] == self.plan['audit']['draws_per_component'] for row in summary['components']),
                        'Geometry audit denominator changed')
                job['summary_sha256'] = sha(root/'summary.json')
            record.update(status='complete', finished=time.time())
        except BaseException as error:
            record.update(status='failed', error=repr(error), finished=time.time())
            raise
        finally:
            write(self.out/'stages'/(name+'.json'), record, exclusive=True)
            snapshot()


def run(args):
    out = Path(args.out).resolve()
    plan = validate(out)
    # Persistent exclusive marker refuses a second controller or an automatic
    # restart after interruption. Existing populations are never retried.
    write(out/'run-lock.json', dict(pid=os.getpid(), started=time.time(), plan_sha256=sha(out/'plan.json')), exclusive=True)
    require(not (out/'status.json').exists(), 'Continuation already started')
    state = dict(complete=False, started=time.time(), plan_sha256=sha(out/'plan.json'), stages=[])

    def snapshot(update):
        state.update(update, updated=time.time())
        # Atomic replacement lets observers distinguish complete status updates.
        temporary = out/'status.tmp.json'
        write(temporary, state)
        temporary.replace(out/'status.json')

    stages = Stages(out, plan, snapshot)
    snapshot(dict(phase='starting'))
    main_released, primary_error, fft_error, interrupted = False, None, None, False
    main_complete, adaptive_passed = False, False
    python, tools = plan['python'], out/'inputs/python'
    main, fft = Path(plan['main_campaign']), Path(plan['fft_prepared'])
    try:
        main_complete = wait_for_main(plan, snapshot)
        main_released = True
        if main_complete:
            validate(out)
            stages.run('analysis', [python, str(tools/'analyze_contact_refinement_campaign.py'),
                       '--campaign', str(main), '--out', str(out/'analysis')], [out/'analysis/manifest.json'])
            main_plan = read(main/'plan.json')
            fit_paths = {}
            for arm in ARMS:
                jobs = sorted((job for job in main_plan['jobs'] if job['arm'] == arm),
                              key=lambda job:(job['replicate'], job['group']))
                paths = [job['directory'] for job in jobs]
                destination = out/('fit-'+arm)
                command = [python, str(tools/'fit_depletion_contact_atlas.py'), '--shape', str(out/'inputs/shape.json'),
                           '--discovery', *paths, '--out', str(destination), *plan['fit_flags']]
                passed = stages.run('fit-'+arm, command,
                    [destination/'manifest.json', destination/'freeze.json', destination/'freeze-rejected.json', destination/'fit-metrics.json'],
                    allowed_gate=lambda destination=destination:gate_rejection(destination, len(plan['source_slot_mapping'])))
                fit_paths[arm] = destination
                if arm == 'adaptive-pivot':
                    adaptive_passed = passed
            if adaptive_passed:
                merged = out/'merged-atlas'
                stages.run('merge', [python, str(tools/'merge_refined_contact_atlas.py'), '--base-fit', plan['base_fit'],
                    '--refined-fit', str(fit_paths['adaptive-pivot']), '--mapping', str(out/'inputs/replacement-mapping.json'),
                    '--out', str(merged)], [merged/'manifest.json', merged/'freeze.json'])
                fixtures = [dict(name='pair', path=None)] + plan['audit']['fixtures']
                audit_jobs = []
                for fixture_index, fixture in enumerate(fixtures):
                    for arm, filename in [('old', 'old-updated-slots-only.json'), ('new', 'new-updated-slots-only.json')]:
                        name = 'audit-'+fixture['name']+'-'+arm
                        destination = out/name
                        command = [str(out/'inputs/audit-binary'), '--shape', str(out/'inputs/shape.json'),
                                   '--model', str(merged/filename), '--out', str(destination),
                                   '--draws-per-component', str(plan['audit']['draws_per_component']),
                                   '--rd', str(plan['audit']['rd']), '--score-points', str(plan['audit']['score_points']),
                                   '--seed', str(plan['audit']['seed']+1009*fixture_index)]
                        if fixture['path']:
                            command += ['--fixed-poses', fixture['path']]
                        audit_jobs.append(dict(id=name, directory=str(destination), log=str(out/'logs'/(name+'.log')),
                                               command=command, status='pending'))
                stages.audit_batch(audit_jobs)
                benchmark = plan['benchmark']
                stages.run('assembly-benchmark', [python, str(tools/'benchmark_depletion_contact_atlas.py'),
                           '--config', str(out/'inputs/benchmark-config.json'), '--initial-model', str(out/'inputs/base-fit-receipt/model.json'),
                           '--discovered-model', str(merged/'model.json'), '--binary', str(out/'inputs/sampler-binary'),
                           '--out', str(out/'assembly-benchmark'), '--streams', str(benchmark['streams']),
                           '--sweeps', str(benchmark['sweeps']), '--sample-every', str(benchmark['sample_every']),
                           '--workers', str(plan['workers']), '--seed', str(benchmark['seed'])],
                           [out/'assembly-benchmark/plan.json', out/'assembly-benchmark/status.json'])
            else:
                snapshot(dict(dependent_production='skipped: complete adaptive fit did not pass the frozen gates; no slot filtering'))
        else:
            snapshot(dict(primary_result='Main campaign incomplete/failed after workers exited; no rerun or dependent physical work'))
    except BaseException as error:
        primary_error = repr(error)
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        snapshot(dict(primary_error=primary_error))
    if main_released and not interrupted:
        try:
            verify_fft(fft)
            if (fft/'report.json').exists():
                report = read(fft/'report.json')
                require(report.get('complete') is True, 'Existing FFT repair did not complete; no retry')
                record = dict(name='fft-repair', status='reused_complete', report_sha256=sha(fft/'report.json'), completed=time.time())
                stages.records.append(record)
                write(out/'stages/fft-repair.json', record, exclusive=True)
            else:
                require(not (fft/'slots').exists(), 'FFT repair already started elsewhere; refuse a duplicate')
                stages.run('fft-repair', plan['fft']['command'], [fft/'report.json'])
                require(read(fft/'report.json').get('complete') is True, 'FFT repair report is incomplete')
        except BaseException as error:
            fft_error = repr(error)
            snapshot(dict(fft_error=fft_error))
    success = main_complete and primary_error is None and fft_error is None
    receipt = dict(complete=success, main_complete=main_complete, main_workers_released=main_released,
                   adaptive_fit_passed=adaptive_passed, primary_error=primary_error, fft_error=fft_error,
                   all_stages=stages.records, plan_sha256=sha(out/'plan.json'), finished=time.time(), scope=plan['scope'])
    write(out/'completion.json', receipt, exclusive=True)
    snapshot(dict(phase='finished', complete=success, stages=stages.records))
    require(success, 'Continuation ended with recorded failure; no automatic retries')
    return dict(complete=True, out=str(out), adaptive_fit_passed=adaptive_passed,
                production_benchmark_ran=adaptive_passed, stages=len(stages.records))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    for name in ('campaign', 'base-fit', 'fft-prepared', 'config', 'out'):
        q.add_argument('--'+name, required=True)
    q.add_argument('--audit-binary', default='target/release/contact-proposal-audit')
    q.add_argument('--sampler-binary', default='target/release/tetramer-mc')
    q.add_argument('--fixed-poses', action='append', default=[])
    q.add_argument('--neighborhoods', help='Frozen evaluation-only neighborhood package; every listed fixture is audited')
    q.add_argument('--workers', type=int, default=4)
    q.add_argument('--wait-seconds', type=int, default=86400)
    q = sub.add_parser('run')
    q.add_argument('--out', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    print(json.dumps(prepare(args) if args.action == 'prepare' else run(args), indent=2, allow_nan=False))
