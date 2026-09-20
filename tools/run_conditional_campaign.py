#!/usr/bin/env python3
"""Matched free-start pilot: conditional s=0/s=6, local and uniform controls.

No frozen model or native docking data enter production. The native catalogue
is archived solely for optional post-hoc classification. This short campaign
measures feasibility and cost, not equilibrium or mixing efficiency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_free_tetramer_campaign import ROOT, prepare_poses, seed, sha, write
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT/'target/release/tetramer-mc')
    parser.add_argument('--conditional-json', type=Path)
    parser.add_argument('--sweeps', type=int, default=200)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--master-seed', type=int, default=2026092517)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gsd', action='store_true')
    args = parser.parse_args()
    if min(args.sweeps, args.sample_every, args.replicates, args.workers) < 1:
        parser.error('Run sizes and worker count must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Output directory must be empty')
    for name in ('configs', 'logs', 'runs', 'provenance'):
        (out/name).mkdir()
    archive = out/'provenance'
    inputs = {'tetramer-mc': args.binary.resolve(),
              'shape.json': ROOT/'examples/tetramer-shape.json',
              'monomer-shape.json': ROOT/'examples/monomer-shape.json',
              'native-pair-motifs.json': ROOT/'examples/native-pair-motifs.json',
              'launcher.py': Path(__file__).resolve(),
              'run_free_tetramer_campaign.py': ROOT/'tools/run_free_tetramer_campaign.py'}
    if args.conditional_json:
        inputs['conditional-options.json'] = args.conditional_json.resolve()
    for name, source in inputs.items():
        shutil.copy2(source, archive/name)
    binary = archive/'tetramer-mc'
    options = json.loads(args.conditional_json.read_text()) if args.conditional_json else {}
    shape = json.loads((archive/'shape.json').read_text())
    radius, rd, activity = 354.50820786337056, 1.5, .035
    jobs, preparations = [], []
    for replicate in range(args.replicates):
        prepare_seed = seed(args.master_seed, 'prepare', replicate)
        mc_seed = seed(args.master_seed, 'mc', replicate)
        poses, certificate = prepare_poses(np.random.default_rng(prepare_seed), shape, radius, rd)
        pose_hash = hashlib.sha256(json.dumps(poses, sort_keys=True).encode()).hexdigest()
        preparations.append(dict(replicate=replicate, preparation_seed=prepare_seed,
                                 mc_seed=mc_seed, initial_poses_sha256=pose_hash,
                                 certificate=certificate))
        for variant in ('conditional-s0', 'conditional-s6', 'local', 'uniform'):
            identifier = f'free-r{replicate:02d}-{variant}'
            conditional = variant.startswith('conditional')
            cfg = dict(shape=str(archive/'shape.json'),
                       monomer_shape=str(archive/'monomer-shape.json'),
                       initial_poses=copy.deepcopy(poses), box_lengths=[2*radius]*3,
                       boundary=dict(kind='spherical', radius=radius),
                       depletant_radius=rd, reservoir_density=activity,
                       poisson_lambda_ratio=16.,
                       endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5),
                       global_probability=0. if variant == 'local' else .5,
                       learned_uniform_weight=.1, local_translation_std_A=.2,
                       local_small_angle_std_degrees=1., gca_probability=1.,
                       center_shift_probability=1., fixed_body_indices=[], seed_labels=[],
                       seed=mc_seed, sweeps=args.sweeps, sample_every=args.sample_every,
                       metadata=dict(arm='dispersed-free', replicate=replicate, variant=variant,
                                     production_model='geometry-only instantaneous conditional closure' if conditional else variant,
                                     external_native_proposal=False, training_feedback=False,
                                     native_pair_motifs=str(archive/'native-pair-motifs.json'),
                                     preparation_seed=prepare_seed, preparation_equilibrated=False,
                                     initial_poses_sha256=pose_hash,
                                     preparation_certificate=certificate))
            if conditional:
                cfg['conditional_closure'] = dict(options, covariance_exponent=0. if variant.endswith('s0') else 6.)
            path = out/'configs'/f'{identifier}.json'
            write(path, cfg)
            jobs.append(dict(id=identifier, replicate=replicate, variant=variant,
                             method='learned' if conditional else 'local-uniform',
                             config=str(path), config_sha256=sha(path),
                             directory=str(out/'runs'/identifier)))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    manifest = dict(schema=1, created_unix_time=time.time(), jobs=jobs, preparations=preparations,
                    sweeps=args.sweeps, sample_every=args.sample_every, workers=args.workers,
                    master_seed=args.master_seed, git_head=head, binary_sha256=sha(binary),
                    input_sha256={name: sha(archive/name) for name in inputs},
                    scope='Four paired controls from each independent dispersed start. Conditional closure has no frozen atlas or native docking input. Native data are used only in post-hoc analysis. No equilibrium, crystal growth or efficiency claim follows from this short pilot.')
    write(out/'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, out=str(out), jobs=len(jobs))), flush=True)
    if args.prepare_only:
        return

    def run(job):
        command = [str(binary), 'run', '--config', job['config'], '--method', job['method'],
                   '--out', job['directory'], '--sweeps', str(args.sweeps),
                   '--sample-every', str(args.sample_every)]
        if not args.gsd:
            command.append('--no-gsd')
        started = time.monotonic()
        with (out/'logs'/f'{job["id"]}.log').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            write(out/'logs'/f'{job["id"]}-process.json', dict(pid=process.pid, command=command))
            code = process.wait()
        path = Path(job['directory'])/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        return dict(id=job['id'], returncode=code, complete=summary.get('complete', False),
                    sampler_cpu_seconds=summary.get('sampler_cpu_seconds'),
                    wall_seconds=time.monotonic()-started)

    records = []
    started = time.monotonic()
    write(out/'status.json', dict(running=True, completed=0, total=len(jobs), supervisor_pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            write(out/'status.json', dict(running=True, completed=len(records), total=len(jobs), records=records))
    for name, digest in manifest['input_sha256'].items():
        assert sha(archive/name) == digest
    complete = all(r['returncode'] == 0 and r['complete'] for r in records)
    result = dict(complete=complete, records=records, wall_seconds=time.monotonic()-started,
                  total_sampler_cpu_seconds=sum(r['sampler_cpu_seconds'] or 0 for r in records))
    write(out/'summary.json', result)
    write(out/'status.json', dict(result, running=False, completed=len(records), total=len(jobs)))
    if not complete:
        raise SystemExit('At least one campaign run failed')


if __name__ == '__main__':
    main()
