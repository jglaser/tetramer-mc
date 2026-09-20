#!/usr/bin/env python3
"""Freeze and run the paired posterior-chart docking control, with arbitrary K."""
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

from prepare_smc_normalizer_atlas import ROOT, Density, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'runs/smc-normalizer-mis-refined/site0/model.json')
    parser.add_argument('--binary', type=Path, default=ROOT/'target/release/docking-mc')
    parser.add_argument('--cycles', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.cycles < 1 or not 1 <= args.workers <= 4:
        parser.error('Positive cycles and one to four workers required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Output directory must be empty')
    for name in ('configs', 'logs', 'runs', 'provenance'):
        (out/name).mkdir()
    archive = out/'provenance'
    native_root = ROOT/'runs/involution-docking-conditional-5000'
    deep_root = ROOT/'runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r1'
    originals = [native_root/'configs'/f'site0-m1-native-r{rep:02d}-mixture.json' for rep in range(2)]
    cfgs = [read(p) for p in originals]
    inputs = {'docking-mc': args.binary.resolve(), 'model.json': args.model.resolve(),
              'shape.json': Path(cfgs[0]['shape']), 'environment.json': Path(cfgs[0]['metadata']['environment']),
              'deep-summary.json': deep_root/'summary.json', 'deep-config.json': deep_root/'config.json',
              'launcher.py': Path(__file__).resolve(),
              'density-reference.py': ROOT/'tools/prepare_smc_normalizer_atlas.py',
              'registration-reference.py': ROOT/'tools/prepare_deep_far_normalizer_atlas.py',
              'chart-reference.py': ROOT/'tools/analyze_involution_docking_campaign.py'}
    inputs.update({f'native-config-r{rep:02d}.json': p for rep, p in enumerate(originals)})
    for name, path in inputs.items():
        shutil.copy2(path, archive/name)
    model = read(archive/'model.json')
    assert model['coordinate_convention'] == 'anchor-body-relative'
    assert model['shape_sha256'] == sha(archive/'shape.json')
    density = Density(model)
    deep = read(archive/'deep-summary.json')
    assert deep['complete']
    shared = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius',
              'reservoir_density', 'translation_steps', 'rotation_steps_deg',
              'rotation_probability', 'local_attempts_per_cycle', 'uniform_probability')
    assert all(cfgs[0][key] == cfgs[1][key] for key in shared)
    assert len(cfgs[0]['fixed_poses']) == 1
    jobs, starts = [], []
    for startindex, start in enumerate(('native', 'deep')):
        for rep in range(2):
            pose = cfgs[rep]['initial_pose'] if start == 'native' else deep['final_particles'][256*rep]['pose']
            seed = 98591010 + 1009*rep + 100000*startindex
            source = dict(archive='native-config-r%02d.json' % rep) if start == 'native' else dict(archive='deep-summary.json', endpoint_index=256*rep)
            q = float(registration([pose], cfgs[rep]['metadata'])[0])
            logg, norms, logs = density.evaluate(relative_poses([pose], cfgs[0]['fixed_poses'][0]))
            starts.append(dict(start=start, replicate=rep, seed=seed, pose=pose, source=source,
                original_q=q, gaussian_log_density=float(logg[0]),
                minimum_latent_radius=float(norms.min()), dominant_component=int(logs[0].argmax())))
            for mode, correlation in (('c0', 0.), ('c09', .9)):
                identifier = f'site0-m1-{start}-r{rep:02d}-{mode}'
                cfg = copy.deepcopy(cfgs[rep])
                cfg.update(shape=str(archive/'shape.json'), initial_pose=pose, seed=seed,
                    poisson_lambda_ratio=16., local_attempts_per_cycle=2,
                    endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5))
                cfg['metadata'].update(start_basin=start, mode=mode, replicate=rep,
                    environment=str(archive/'environment.json'), initial_pose_source=source,
                    source_smc=None, source_config_sha256=sha(archive/f'native-config-r{rep:02d}.json'),
                    atlas_frozen=True, native_informed_atlas=True, conditional_atlas=False,
                    proposal_source='posterior chart responsibilities', component_count=len(model['weights']),
                    initialization_equilibrated=False)
                path = out/'configs'/f'{identifier}.json'
                write(path, cfg)
                jobs.append(dict(id=identifier, start=start, replicate=rep, mode=mode, seed=seed,
                    correlation=correlation, method='posterior-involution', config=str(path),
                    config_sha256=sha(path), directory=str(out/'runs'/identifier)))
    write(archive/'selected-starts.json', starts)
    # Archive the actual source state, including uncommitted parent implementation.
    for name in ('docking.rs', 'proposal.rs', 'involution.rs', 'depletion.rs'):
        path = ROOT/'src'/name
        if path.exists():
            shutil.copy2(path, archive/name)
    manifest = dict(schema=1, created_unix_time=time.time(), jobs=jobs, starts=starts,
        cycles=args.cycles, sample_every=1, workers=args.workers,
        component_count=len(model['weights']), model_sha256=sha(archive/'model.json'),
        binary_sha256=sha(archive/'docking-mc'),
        input_sha256={p.name: sha(p) for p in archive.iterdir() if p.is_file()},
        source_paths={name: str(path) for name, path in inputs.items()},
        target='Site0: one mobile rigid tetramer, one fixed neighbor, radius18 capture ball, all proper rotations, rd1.5, z0.035.',
        kernel='Two identical local attempts per cycle plus one posterior-source involution. Separate uniform branch epsilon0.1. c0 is matched independent redraw within the Gaussian branch, not the old full-mixture kernel.',
        prior_information='Frozen native-informed parent atlas retained. This is a controlled sampler benchmark, not template-free learning.',
        inference_scope='Selected non-equilibrated starts; MC passages are sampling diagnostics, not physical kinetic rates. No long-run or equilibrium claim follows from this bounded pilot.')
    write(out/'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, out=str(out), jobs=len(jobs), starts=starts)), flush=True)
    if args.prepare_only:
        return

    def run(job):
        command = [str(archive/'docking-mc'), '--config', job['config'], '--model', str(archive/'model.json'),
            '--out', job['directory'], '--cycles', str(args.cycles), '--sample-every', '1',
            '--method', 'posterior-involution', '--correlation', str(job['correlation'])]
        started = time.monotonic()
        with (out/'logs'/f'{job["id"]}.log').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            write(out/'logs'/f'{job["id"]}-process.json', dict(pid=process.pid, command=command))
            code = process.wait()
        path = Path(job['directory'])/'summary.json'
        summary = read(path) if path.exists() else {}
        return dict(id=job['id'], returncode=code, complete=summary.get('complete', False),
            cpu_seconds=summary.get('sampler_cpu_seconds'), wall_seconds=time.monotonic()-started)

    records, begun = [], time.monotonic()
    write(out/'status.json', dict(running=True, completed=0, total=len(jobs), supervisor_pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            records.append(future.result())
            print(json.dumps(records[-1]), flush=True)
            write(out/'status.json', dict(running=True, completed=len(records), total=len(jobs), records=records))
    assert all(sha(archive/name) == digest for name, digest in manifest['input_sha256'].items())
    result = dict(complete=all(r['complete'] and r['returncode'] == 0 for r in records), records=records,
        wall_seconds=time.monotonic()-begun, total_sampler_cpu_seconds=sum(r['cpu_seconds'] or 0. for r in records))
    write(out/'summary.json', result)
    write(out/'status.json', dict(result, running=False, completed=len(records), total=len(jobs)))
    if not result['complete']:
        raise SystemExit('At least one pilot run failed')


if __name__ == '__main__':
    main()
