#!/usr/bin/env python3
"""Freeze and run matched legacy versus pivot/adaptive contact refinement.

Selection is the first N rows of the immutable native-blind source fit, in its
recorded order. Every selected row, replicate, and rejected state is retained.
Preparation never launches a physical job. Launch is a separate explicit step.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare_shoulder_docking_benchmark import local_dependencies
from run_conditional_ray_campaign import execute_jobs

SCHEMA = 'paired-contact-refinement-campaign-v1'
ARMS = ('legacy', 'adaptive-pivot')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Frozen path escapes its directory')
    return path


def validate_source(source, shape):
    manifest = read(source / 'manifest.json')
    require(manifest.get('complete') is True, 'Source fit is not complete')
    require(manifest.get('native_informed_proposal') is False
            and manifest.get('native_geometry_used_in_preparation') is False,
            'Require a declared native-blind source fit')
    require(manifest['shape_sha256'] == sha(shape), 'Source fit/shape mismatch')
    for name, digest in read(source / 'freeze.json')['files'].items():
        require(sha(checked_path(source, name)) == digest, 'Source fit changed: ' + name)
    metrics = read(source / 'fit-metrics.json')
    require(metrics['independent_slots'] == len(metrics['slots']), 'Source slot count mismatch')
    return metrics


def allocations(args):
    require(type(args.slots) is int and args.slots > 0, 'Positive slot count required')
    require(type(args.replicates) is int and args.replicates > 0, 'Positive replicate count required')
    require(type(args.slots_per_job) is int and 1 <= args.slots_per_job <= 1024,
            'Require 1..1024 slots per job')
    require(1 <= args.workers <= 4, 'At most four new physical jobs')
    require(args.refine_steps > args.burn > 0 and 0 < args.save_every <= args.refine_steps - args.burn,
            'Positive burn and retained refinement allocations required')
    require(args.search_points > 0 and args.validation_points > 0, 'Positive score budgets required')
    require(math.isfinite(args.rd) and args.rd >= 0
            and math.isfinite(args.activity) and args.activity >= 0, 'Invalid bath')
    groups = (args.slots + args.slots_per_job - 1) // args.slots_per_job
    require(0 <= args.seed <= 2**64 - 1 - 1009 * (args.replicates * groups - 1),
            'Seed plan does not fit u64')
    return groups


def prepare(args):
    groups = allocations(args)
    out, source, shape, binary = map(lambda p: Path(p).resolve(),
                                   (args.out, args.source_fit, args.shape, args.binary))
    require(not out.exists(), 'Fresh campaign directory required; no overwrite or retry')
    metrics = validate_source(source, shape)
    require(args.slots <= len(metrics['slots']), 'Requested more slots than the source contains')
    help_text = subprocess.run([str(binary), '--help'], check=True, capture_output=True, text=True).stdout
    require('--refine-adapt' in help_text and '--refine-contact-pivot' in help_text,
            'Discovery executable lacks the opt-in refinement kernels')
    selected = metrics['slots'][:args.slots]
    require(len({row['global_slot'] for row in selected}) == len(selected), 'Duplicate source slots')
    for row in selected:
        pose = row['reference_pose']
        require(len(pose['position']) == 3 and len(pose['orientation']) == 4
                and all(math.isfinite(x) for x in pose['position'] + pose['orientation']),
                'Invalid frozen reference pose')
    estimated_bytes = 2 * args.replicates * args.slots * args.refine_steps * 8192 + 1024**3
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    require(shutil.disk_usage(parent).free >= estimated_bytes,
            'Insufficient space for conservative full-trace storage budget')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    archive = out / 'inputs'
    archive.mkdir()
    (out / 'logs').mkdir()
    (out / 'populations').mkdir()
    (archive / 'source-fit').mkdir()
    for name in ('fit-metrics.json', 'manifest.json', 'freeze.json'):
        shutil.copy2(source / name, archive / 'source-fit' / name)
    shutil.copy2(shape, archive / 'shape.json')
    shutil.copy2(binary, archive / 'depletion-contact-discovery')
    python_sources = local_dependencies([Path(__file__)])
    (archive / 'python').mkdir()
    for name, path in python_sources.items():
        shutil.copy2(path, archive / 'python' / name)
    jobs = []
    mapping = []
    for replicate in range(args.replicates):
        for group in range(groups):
            start = group * args.slots_per_job
            rows = selected[start:start + args.slots_per_job]
            poses = [row['reference_pose'] for row in rows]
            poses_path = archive / f'starts-r{replicate:02d}-g{group:02d}.json'
            write(poses_path, poses, exclusive=True)
            seed = args.seed + 1009 * (replicate * groups + group)
            slot_map = [dict(local_slot=i, new_global_slot=replicate * args.slots + start + i,
                             replicate=replicate, source_selection_index=start + i,
                             original_global_slot=row['global_slot'],
                             original_source_index=row['source_index'],
                             original_local_slot=row['local_slot'],
                             original_source_seed=row.get('source_seed'))
                        for i, row in enumerate(rows)]
            mapping.extend(slot_map)
            for arm in ARMS:
                name = f'{arm}-r{replicate:02d}-g{group:02d}'
                directory = out / 'populations' / name
                command = [str(archive / 'depletion-contact-discovery'),
                           '--shape', str(archive / 'shape.json'), '--out', str(directory),
                           '--initial-poses', str(poses_path), '--starts', str(len(rows)),
                           '--search-steps', '0', '--search-points', str(args.search_points),
                           '--validation-points', str(args.validation_points),
                           '--refine-steps', str(args.refine_steps), '--burn', str(args.burn),
                           '--save-every', str(args.save_every), '--rd', str(args.rd),
                           '--activity', str(args.activity), '--seed', str(seed)]
                if arm == 'adaptive-pivot':
                    command += ['--refine-adapt', '--refine-contact-pivot']
                jobs.append(dict(id=name, arm=arm, replicate=replicate, group=group, seed=seed,
                                 slots=len(rows), slot_map=slot_map, initial_poses=str(poses_path),
                                 directory=str(directory), log=str(out / 'logs' / (name + '.log')),
                                 command=command, status='pending'))
    plan = dict(schema=SCHEMA, source_fit=str(source), source_shape=str(shape),
                selection='First slots in frozen fit-metrics recorded order; no native labels, ranking, acceptance or score filter.',
                source_slots=args.slots, source_slot_count=len(metrics['slots']),
                replicates=args.replicates, slots_per_job=args.slots_per_job, groups=groups,
                source_slot_map=mapping, jobs=jobs, workers=args.workers, seed=args.seed,
                bath=dict(radius=args.rd, activity=args.activity),
                allocation=dict(refine_steps=args.refine_steps, burn=args.burn, save_every=args.save_every,
                                search_steps=0, search_points=args.search_points,
                                validation_points=args.validation_points,
                                retained_per_slot=(args.refine_steps - args.burn) // args.save_every),
                total_attempts=2 * args.replicates * args.slots * args.refine_steps,
                storage_estimate_bytes=estimated_bytes,
                input_sha256={str(p.relative_to(out)): sha(p) for p in sorted(archive.rglob('*')) if p.is_file()},
                scope='Fixed pair-refinement proposal-design comparison, not basin masses, stationarity or assembly. '
                      'Same master seed across matched arms is deliberate; their RNG consumption diverges. '
                      'Independent replicates are distinct streams, not proof of equilibration. '
                      'Every attempt and source slot is preserved; adaptation is discarded burn only.')
    write(out / 'plan.json', plan, exclusive=True)
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), files=plan['input_sha256']), exclusive=True)
    validate(out)
    return dict(prepared=True, launched=False, out=str(out), jobs=len(jobs), attempts=plan['total_attempts'],
                source_slots=args.slots, replicates=args.replicates)


def validate(out):
    out = Path(out).resolve()
    plan, freeze = read(out / 'plan.json'), read(out / 'freeze.json')
    require(plan['schema'] == SCHEMA, 'Unexpected campaign schema')
    require(sha(out / 'plan.json') == freeze['plan_sha256'], 'Frozen plan changed')
    require(freeze['files'] == plan['input_sha256'], 'Frozen file list changed')
    for name, digest in freeze['files'].items():
        require(sha(checked_path(out, name)) == digest, 'Frozen input changed: ' + name)
    require(1 <= plan['workers'] <= 4, 'Invalid worker cap')
    require(len(plan['jobs']) == 2 * plan['replicates'] * plan['groups'], 'Incomplete job allocation')
    require(len({j['id'] for j in plan['jobs']}) == len(plan['jobs']), 'Duplicate jobs')
    return plan


def runtime_summary(out, jobs):
    rows = []
    for job in jobs:
        log = Path(job['log'])
        slots = []
        if log.is_file():
            for line in log.read_text(errors='replace').splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and 'slot' in value and 'wall_seconds' in value:
                    slots.append(value)
        measured = sum(row['wall_seconds'] for row in slots)
        rows.append(dict(id=job['id'], arm=job['arm'], status=job['status'],
                         completed_slots=len(slots), slots=job['slots'],
                         measured_slot_seconds=measured,
                         mean_seconds_per_completed_slot=measured / len(slots) if slots else None,
                         extrapolated_job_seconds=measured * job['slots'] / len(slots) if slots else None,
                         slot_reports=slots))
    return dict(out=str(Path(out).resolve()), complete=all(j['status'] == 'complete' for j in jobs), jobs=rows)


def verify_output(job, plan):
    root = Path(job['directory'])
    manifest = read(root / 'manifest.json')
    require(manifest.get('complete') is True and manifest['slots'] == job['slots'], 'Incomplete output')
    for name, digest in manifest['outputs_sha256'].items():
        require(sha(checked_path(root, name)) == digest, 'Discovery output changed: ' + name)
    config = manifest['config']
    for key in ('refine_steps', 'burn', 'save_every', 'search_steps', 'search_points', 'validation_points'):
        require(config[key] == plan['allocation'][key], 'Output allocation mismatch: ' + key)
    require(config['seed'] == job['seed'] and config['rd'] == plan['bath']['radius']
            and config['activity'] == plan['bath']['activity'], 'Output stream/bath mismatch')
    adaptive = job['arm'] == 'adaptive-pivot'
    require(config['refine_adapt'] == adaptive and config['refine_contact_pivot'] == adaptive,
            'Output kernel mismatch')
    require(manifest['retained_samples'] == job['slots'] * plan['allocation']['retained_per_slot'],
            'Lost retained observations')


def run(args):
    out = Path(args.out).resolve()
    plan = validate(out)
    require(not (out / 'status.json').exists(), 'Campaign already launched; no retries')
    require(not any((out / 'populations').iterdir()) and not any((out / 'logs').iterdir()),
            'Existing outputs; no overwrite or continuation')
    jobs = copy.deepcopy(plan['jobs'])
    env = dict(os.environ, **{key: '1' for key in THREAD_VARIABLES})
    started = time.time()

    def snapshot():
        write(out / 'status.json', dict(complete=all(j['status'] == 'complete' for j in jobs),
              jobs=jobs, started=started, updated=time.time(), plan_sha256=sha(out / 'plan.json')))
        write(out / 'runtime-summary.json', runtime_summary(out, jobs))

    snapshot()
    execute_jobs(jobs, snapshot, workers=plan['workers'],
                 popen=lambda *a, **kw: subprocess.Popen(*a, **kw, env=env))
    for job in jobs:
        verify_output(job, plan)
    write(out / 'completion.json', dict(complete=True, jobs=len(jobs),
          total_attempts=plan['total_attempts'], wall_seconds=time.time() - started,
          plan_sha256=sha(out / 'plan.json')), exclusive=True)
    return runtime_summary(out, jobs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    q = commands.add_parser('prepare', help='Freeze inputs and allocations; do not launch')
    q.add_argument('--source-fit', required=True)
    q.add_argument('--shape', required=True)
    q.add_argument('--binary', default='target/release/depletion-contact-discovery')
    q.add_argument('--out', required=True)
    q.add_argument('--slots', type=int, default=64)
    q.add_argument('--replicates', type=int, default=2)
    q.add_argument('--slots-per-job', type=int, default=8)
    q.add_argument('--workers', type=int, default=4)
    q.add_argument('--seed', type=int, default=202609280701)
    q.add_argument('--refine-steps', type=int, default=4608)
    q.add_argument('--burn', type=int, default=512)
    q.add_argument('--save-every', type=int, default=8)
    q.add_argument('--search-points', type=int, default=2048)
    q.add_argument('--validation-points', type=int, default=8192)
    q.add_argument('--rd', type=float, default=1.4)
    q.add_argument('--activity', type=float, default=0.0275)
    for action in ('run', 'status'):
        q = commands.add_parser(action)
        q.add_argument('--out', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'prepare':
        result = prepare(args)
    elif args.action == 'run':
        result = run(args)
    else:
        out = Path(args.out).resolve()
        jobs = read(out / 'status.json')['jobs'] if (out / 'status.json').exists() else validate(out)['jobs']
        result = runtime_summary(out, jobs)
    print(json.dumps(result, indent=2, allow_nan=False))
