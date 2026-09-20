#!/usr/bin/env python3
"""Freeze and run independent fixed-budget contact-region importance populations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output must be empty; use --run-prepared to launch an existing preparation')
    for name in ['provenance', 'runs', 'logs']:
        (out / name).mkdir()
    archive = out / 'provenance'
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    shape = Path(cfg['shape'])
    if not shape.is_absolute():
        shape = config_path.parent / shape
    inputs = {'basin-normalizer': args.binary.resolve(), 'model.json': args.model.resolve(),
              'input-config.json': config_path, 'shape.json': shape,
              'launcher.py': Path(__file__).resolve(),
              'analyze_basin_normalizers.py': ROOT / 'tools/analyze_basin_normalizers.py',
              'normalizer.rs': ROOT / 'src/normalizer.rs',
              'overlap_weight.rs': ROOT / 'src/overlap_weight.rs'}
    for name, path in inputs.items():
        shutil.copy2(path, archive / name)
    cfg['shape'] = str(archive / 'shape.json')
    if args.lambda_ratio is not None:
        if args.lambda_ratio <= 0:
            raise ValueError('Poisson intensity ratio must be positive')
        cfg['poisson_lambda_ratio'] = args.lambda_ratio
    save(archive / 'config.json', cfg)
    component_count = len(json.loads((archive / 'model.json').read_text())['weights'])
    assert component_count > 0
    assert cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
    assert cfg['capture_radius'] == 18 and cfg['poisson_lambda_ratio'] > 0
    assert len(cfg['fixed_poses']) == 1
    assert all(k in cfg['metadata'] for k in ['native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg'])
    jobs = []
    for scale_index, scale in enumerate(args.scales):
        for replicate in range(args.replicates):
            seed = args.seed + scale_index * 100003 + replicate * 1009
            label = f'scale-{scale:g}-r{replicate:02d}'
            cmd = [str(archive / 'basin-normalizer'), '--config', str(archive / 'config.json'),
                   '--model', str(archive / 'model.json'), '--out', str(out / 'runs' / label),
                   '--samples', str(args.samples), '--seed', str(seed),
                   '--covariance-scale', str(scale), '--cloud-replicates', str(args.cloud_replicates)]
            jobs.append(dict(id=label, covariance_std_scale=scale, replicate=replicate, seed=seed,
                             samples=args.samples, cloud_replicates=args.cloud_replicates, command=cmd,
                             directory=str(out / 'runs' / label)))
    assert len({j['seed'] for j in jobs}) == len(jobs)
    manifest = dict(schema=1, source_inputs={name: dict(path=str(path), sha256=sha(path)) for name, path in inputs.items()},
        archive_sha256={p.name: sha(p) for p in archive.iterdir()}, jobs=jobs,
        workers=args.workers, created_unix=time.time(),
        design='Independent seeds per scale/population. All unconditional draws, including invalid zeros, are retained. Frozen conditional atlas; covariance matrices multiplied by scale^2; same means, weights, capture, physical target, and defensive uniform probability.',
        model_components=component_count,
        physical=dict(depletant_radius=cfg['depletant_radius'], activity=cfg['reservoir_density'],
                      capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
                      lambda_ratio=cfg['poisson_lambda_ratio'], uniform_probability=cfg['uniform_probability']),
        inference='Exploratory importance pilot; observed ESS and four-population intervals cannot certify missing-mode coverage. No native template-free claim.')
    save(out / 'manifest.json', manifest)
    return out


def run(out, workers):
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['archive_sha256'].items():
        if sha(out / 'provenance' / name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    for job in manifest['jobs']:
        if Path(job['directory']).exists():
            raise ValueError(f'Job already exists; refusing to overwrite/restart: {job["id"]}')
    started = time.time()
    save(out / 'runner-status.json', dict(status='running', started_unix=started, jobs=len(manifest['jobs'])))
    def execute(job):
        with (out / 'logs' / (job['id'] + '.log')).open('w') as log:
            result = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT)
        record = dict(id=job['id'], returncode=result.returncode, completed_unix=time.time())
        save(out / 'logs' / (job['id'] + '-status.json'), record)
        return record
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(execute, j) for j in manifest['jobs']]):
            record = future.result()
            results.append(record)
            print(json.dumps(record), flush=True)
    failed = [r['id'] for r in results if r['returncode'] != 0]
    save(out / 'runner-status.json', dict(status='failed' if failed else 'complete',
        started_unix=started, completed_unix=time.time(), wall_seconds=time.time()-started,
        results=results, failed=failed))
    if failed:
        raise RuntimeError(f'Failed jobs: {failed}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--run-prepared', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--binary', type=Path, default=ROOT / 'target/release/basin-normalizer')
    p.add_argument('--config', type=Path, default=ROOT / 'runs/involution-docking-conditional-5000/configs/site0-m1-native-r00-mixture.json')
    p.add_argument('--model', type=Path, default=ROOT / 'runs/involution-docking-conditional-5000/provenance/model.json')
    p.add_argument('--scales', type=float, nargs='+', default=[.5, 1., 2., 4.])
    p.add_argument('--replicates', type=int, default=4)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--cloud-replicates', type=int, default=2)
    p.add_argument('--lambda-ratio', type=float, help='Override auxiliary Poisson intensity/activity ratio; physical target stays fixed')
    p.add_argument('--seed', type=int, default=249103771)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if min(args.replicates, args.samples, args.cloud_replicates, args.workers) < 1 or min(args.scales) <= 0:
        p.error('All sample counts and scales must be positive')
    if (args.out is None) == (args.run_prepared is None):
        p.error('Provide exactly one of --out or --run-prepared')
    out = args.run_prepared.resolve() if args.run_prepared else prepare(args)
    if not args.prepare_only:
        run(out, args.workers)
    print(json.dumps(dict(output=str(out), prepared_only=args.prepare_only)))


if __name__ == '__main__':
    main()
