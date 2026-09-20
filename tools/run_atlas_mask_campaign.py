#!/usr/bin/env python3
"""Matched active-mask compression around the same transported atlas.

Reuse the exact covariance-arm starts, MC seeds, parameter transport, and
native-informed atlas. Masks reduce active density evaluation while retaining
all reference components and Gaussian auxiliary coordinates in memory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_free_tetramer_campaign import ROOT, sha, write


ARMS = ('full', 'fixed16', 'fixed8', 'poisson8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-campaign', type=Path,
                        default=ROOT/'runs/atlas-transport-400')
    parser.add_argument('--binary', type=Path, default=ROOT/'target/release/tetramer-mc')
    parser.add_argument('--mask-json', type=Path,
                        help='Common atlas_mask overrides applied before fixed-count arm bounds')
    parser.add_argument('--sweeps', type=int, default=400)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gsd', action='store_true')
    args = parser.parse_args()
    if min(args.sweeps, args.sample_every, args.workers) < 1:
        parser.error('Run sizes and worker count must be positive')
    source = args.source_campaign.resolve()
    previous = json.loads((source/'manifest.json').read_text())
    previous_jobs = [j for j in previous['jobs'] if j.get('variant', j.get('mode')) == 'covariance']
    if not previous_jobs:
        parser.error('Source campaign needs covariance-arm configurations')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Output directory must be empty')
    for name in ('configs', 'logs', 'runs', 'provenance'):
        (out/name).mkdir()
    archive = out/'provenance'
    inputs = {'tetramer-mc': args.binary.resolve(),
              'source-manifest.json': source/'manifest.json',
              'model.json': Path(previous_jobs[0]['model']),
              'shape.json': source/'provenance/shape.json',
              'monomer-shape.json': source/'provenance/monomer-shape.json',
              'native-pair-motifs.json': source/'provenance/native-pair-motifs.json',
              'launcher.py': Path(__file__).resolve(),
              'run_free_tetramer_campaign.py': ROOT/'tools/run_free_tetramer_campaign.py'}
    if args.mask_json:
        inputs['mask-options.json'] = args.mask_json.resolve()
    for job in previous_jobs:
        inputs[f"source-r{job['replicate']:02d}.json"] = Path(job['config'])
    for name, path in inputs.items():
        shutil.copy2(path, archive/name)
    binary, model = archive/'tetramer-mc', archive/'model.json'
    base_options = json.loads(args.mask_json.read_text()) if args.mask_json else {}
    jobs = []
    for old_job in previous_jobs:
        replicate = old_job['replicate']
        original = json.loads((archive/f'source-r{replicate:02d}.json').read_text())
        assert sha(old_job['model']) == sha(model)
        assert not original['fixed_body_indices'] and not original['seed_labels']
        assert original['boundary']['kind'] == 'spherical'
        for arm in ARMS:
            cfg = copy.deepcopy(original)
            cfg.update(shape=str(archive/'shape.json'),
                       monomer_shape=str(archive/'monomer-shape.json'),
                       sweeps=args.sweeps, sample_every=args.sample_every)
            for key in ('auxiliary_transport', 'conditional_closure', 'contact_memory', 'reversible_jump'):
                cfg.pop(key, None)
            cfg['metadata'].update(
                variant=arm, proposal_mode=arm,
                native_informed_atlas=True, model_sha256=sha(model),
                native_pair_motifs=str(archive/'native-pair-motifs.json'),
                source_campaign=str(source), source_config_sha256=old_job['config_sha256'],
                frozen_atlas_component_count=len(json.loads(model.read_text())['anchors']),
                training_feedback=True,
                feedback_scope='Current pair geometry only; no accumulated training archive')
            assert original['atlas_transport']['weight_gain'] == 0.
            assert original['atlas_transport']['weight_noise'] == 0.
            mask = dict(activity=8., min_components=0, max_components=None,
                        label_law='atlas_weight', refresh_probability=1., initial_full=True)
            mask.update(base_options)
            if arm in ('full', 'fixed16', 'fixed8'):
                k = {'full': len(json.loads(model.read_text())['anchors']), 'fixed16': 16, 'fixed8': 8}[arm]
                mask.update(min_components=k, max_components=k)
            cfg['atlas_mask'] = mask
            identifier = f'free-r{replicate:02d}-{arm}'
            path = out/'configs'/f'{identifier}.json'
            write(path, cfg)
            jobs.append(dict(id=identifier, replicate=replicate, variant=arm,
                             mode=arm, config=str(path), config_sha256=sha(path),
                             model=str(model), model_sha256=sha(model),
                             directory=str(out/'runs'/identifier)))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    manifest = dict(schema=1, created_unix_time=time.time(), jobs=jobs,
                    preparations=previous['preparations'], source_campaign=str(source),
                    sweeps=args.sweeps, sample_every=args.sample_every, workers=args.workers,
                    git_head=head, binary_sha256=sha(binary), model_sha256=sha(model),
                    model_label='native-informed fixed atlas', arms=list(ARMS),
                    input_sha256={name: sha(archive/name) for name in inputs},
                    scope='Reused independent starts and paired MC seeds; all arms retain the same native-informed full atlas and Gaussian auxiliary state, with identical mean/covariance transport, physical target and local/collective kernels. Masks only change active proposal density evaluation. The count prior is separately normalized from each conditional subset law. This is an accessibility and runtime pilot, not an equilibrium or speedup estimate.')
    write(out/'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, out=str(out), jobs=len(jobs))), flush=True)
    if args.prepare_only:
        return

    def run(job):
        command = [str(binary), 'run', '--config', job['config'], '--model', job['model'],
                   '--method', 'learned', '--out', job['directory'],
                   '--sweeps', str(args.sweeps), '--sample-every', str(args.sample_every)]
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

    started = time.monotonic()
    records = []
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
