#!/usr/bin/env python3
"""Freeze and run independent fixed-budget contact-region importance populations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def validate_inputs(cfg, model, shape, proposal_anchor_index):
    """Validate every physical neighbor, independent of proposal-anchor choice.

    Exact hard nonoverlap among fixed sphere unions remains an executable
    precondition checked before any draw, not a bounding-sphere approximation.
    """
    def vector(value, length):
        if len(value) != length or not all(math.isfinite(x) for x in value):
            raise ValueError('Nonfinite or incorrectly sized pose vector')
    def pose(value):
        vector(value['position'], 3); vector(value['orientation'], 4)
        if abs(sum(x*x for x in value['orientation'])-1) > 1e-8:
            raise ValueError('Physical pose quaternion must be normalized')
    if not cfg['fixed_poses']:
        raise ValueError('At least one physical fixed neighbor is required')
    for p in cfg['fixed_poses']:
        pose(p)
    pose(cfg['initial_pose'])
    vector(cfg['capture_center'], 3)
    if proposal_anchor_index is not None and (type(proposal_anchor_index) is not int or not 0 <= proposal_anchor_index < len(cfg['fixed_poses'])):
        raise ValueError('Proposal anchor index outside fixed-neighbor list')
    if cfg['depletant_radius'] != 1.5 or cfg['reservoir_density'] != .035 or cfg['capture_radius'] != 18:
        raise ValueError('Campaign requires rd=1.5, activity=.035 and capture radius18')
    if not math.isfinite(cfg['poisson_lambda_ratio']) or cfg['poisson_lambda_ratio'] <= 0:
        raise ValueError('Positive finite Poisson intensity ratio required')
    if not math.isfinite(cfg['uniform_probability']) or not 0 < cfg['uniform_probability'] < 1:
        raise ValueError('Positive defensive uniform probability below one required')
    metric = cfg['metadata']
    for key in ('native_poses', 'rigid_members'):
        if not metric[key]:
            raise ValueError('Complete native-region metadata is required')
        for p in metric[key]:
            pose(p)
    for key in ('member_error_scale', 'angle_error_scale_deg'):
        if not math.isfinite(metric[key]) or metric[key] <= 0:
            raise ValueError('Positive finite native-region scales required')
    if model['shape_sha256'] != sha(shape):
        raise ValueError('Model and physical shape hashes differ')
    if model['coordinate_convention'] != 'anchor-body-relative':
        raise ValueError('Expected anchor-body-relative atlas')
    count = len(model['weights'])
    if count == 0 or any(len(model[k]) != count for k in ('anchors', 'means', 'covariances')):
        raise ValueError('Inconsistent Gaussian atlas arrays')
    if any(not math.isfinite(w) or w <= 0 for w in model['weights']) or abs(sum(model['weights'])-1) > 1e-10:
        raise ValueError('Atlas weights must form a positive normalized probability')


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
    if args.lambda_ratio is not None:
        cfg['poisson_lambda_ratio'] = args.lambda_ratio
    model = json.loads(args.model.resolve().read_text())
    anchor_index = args.proposal_anchor_index
    validate_inputs(cfg, model, shape, anchor_index)
    inputs = {'basin-normalizer': args.binary.resolve(), 'model.json': args.model.resolve(),
              'input-config.json': config_path, 'shape.json': shape,
              'launcher.py': Path(__file__).resolve(),
              'analyze_basin_normalizers.py': ROOT / 'tools/analyze_basin_normalizers.py',
              'prepare_smc_normalizer_atlas.py': ROOT / 'tools/prepare_smc_normalizer_atlas.py',
              'prepare_deep_far_normalizer_atlas.py': ROOT / 'tools/prepare_deep_far_normalizer_atlas.py',
              'normalizer.rs': ROOT / 'src/normalizer.rs',
              'basin-normalizer-cli.rs': ROOT / 'src/bin/basin-normalizer.rs',
              'proposal.rs': ROOT / 'src/proposal.rs',
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
    assert len(cfg['fixed_poses']) >= 1
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
            if anchor_index is not None:
                cmd.extend(['--proposal-anchor-index', str(anchor_index)])
            jobs.append(dict(id=label, covariance_std_scale=scale, replicate=replicate, seed=seed,
                             samples=args.samples, cloud_replicates=args.cloud_replicates, command=cmd,
                             proposal_anchor_index=anchor_index,
                             directory=str(out / 'runs' / label)))
    assert len({j['seed'] for j in jobs}) == len(jobs)
    manifest = dict(schema=1, source_inputs={name: dict(path=str(path), sha256=sha(path)) for name, path in inputs.items()},
        archive_sha256={p.name: sha(p) for p in archive.iterdir()}, jobs=jobs,
        workers=args.workers, created_unix=time.time(),
        design='Independent seeds per scale/population. All unconditional draws, including invalid zeros, are retained. Frozen conditional atlas; covariance matrices multiplied by scale^2; same means, weights, capture, physical target, and defensive uniform probability.',
        model_components=component_count,
        proposal_anchor_index=anchor_index,
        physical=dict(depletant_radius=cfg['depletant_radius'], activity=cfg['reservoir_density'],
                      capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
                      fixed_poses=cfg['fixed_poses'], fixed_neighbor_count=len(cfg['fixed_poses']),
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
    p.add_argument('--proposal-anchor-index', type=int, help='Use one fixed neighbor as proposal frame; all remain physical (default: average all anchors)')
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
