#!/usr/bin/env python3
"""Freeze uniform latent-shell reference targets and audit their geometry.

The original Gaussian supplies a coordinate chart only. This preparer draws
uniform six-dimensional shell probes, never Gaussian or depletant samples.
"""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import beta

from prepare_native_confirmation_atlas import AtomUnionAudit, arrays, native_q, relative_poses
from prepare_smc_normalizer_atlas import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODEL_SHA = '86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667'
SHELLS = [(4., 5.), (5., 8.), (8., 12.)]


def validate_shell(lo, hi):
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
        raise ValueError('Require finite 0 <= inner < outer radius')


def radial_quantile(u, lo, hi):
    validate_shell(lo, hi)
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('Quantiles must lie in [0,1]')
    # Direct inverse of P(r<=s)=(s^6-lo^6)/(hi^6-lo^6).
    sixth = lo**6 + u * (hi**6 - lo**6)
    if not np.isfinite(sixth).all():
        raise ValueError('Unresolved shell scale')
    return sixth**(1. / 6.)


def shell_volume(lo, hi):
    validate_shell(lo, hi)
    difference = hi**6 if lo == 0 else hi**6 * (-math.expm1(6 * math.log1p((lo-hi)/hi)))
    volume = math.pi**3 * difference / 6
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('Unresolved positive shell volume')
    return volume


def sample_shell(rng, count, lo, hi):
    if count <= 0:
        raise ValueError('Positive probe count required')
    direction = rng.normal(size=(count, 6))
    lengths = np.linalg.norm(direction, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError('Invalid Gaussian direction; stop rather than resample')
    radius = radial_quantile(rng.random(count), lo, hi)
    return direction * (radius / lengths)[:, None]


def decode_latent(latent, model, fixed):
    latent = np.asarray(latent, dtype=float)
    assert latent.ndim == 2 and latent.shape[1] == 6 and np.isfinite(latent).all()
    assert model['coordinate_convention'] == 'anchor-body-relative'
    assert model['weights'] == [1.] and len(model['anchors']) == 1
    lower = np.linalg.cholesky(model['covariances'][0])
    x = model['means'][0] + latent @ lower.T
    ell = model['angular_length']
    assert ell > 0 and math.isfinite(ell)
    c = x[:, 3:] / ell
    relative_r = Rotation.from_quat(np.column_stack((c, np.ones(len(c))))).as_matrix() @ np.array(model['anchors'][0]['rotation'])
    relative_t = x[:, :3] + model['anchors'][0]['position']
    ft, _, fr = arrays([fixed])
    world_t = relative_t @ fr[0].T + ft[0]
    world_q = Rotation.from_matrix(fr[0] @ relative_r).as_quat()[:, [3, 0, 1, 2]]
    poses = [{'position': t.tolist(), 'orientation': q.tolist()} for t, q in zip(world_t, world_q)]
    log_j = np.log(np.diag(lower)).sum() - 3 * math.log(ell) - 2 * math.log(math.pi) - 2 * np.log1p(np.sum(c*c, axis=1))
    return poses, log_j


def encode_matrix(poses, model, fixed):
    relative = relative_poses(poses, fixed)
    t, _, r = arrays(relative)
    delta = r @ np.array(model['anchors'][0]['rotation']).T
    skew = (delta - np.eye(3)) @ np.linalg.inv(delta + np.eye(3))
    c = skew[:, [2, 0, 1], [1, 2, 0]]
    x = np.column_stack((t - model['anchors'][0]['position'], model['angular_length'] * c))
    return np.linalg.solve(np.linalg.cholesky(model['covariances'][0]), (x - model['means'][0]).T).T


def validate_physics(region, cfg, shape_hash):
    assert region['fixed_neighbor'] == cfg['fixed_poses'][0]
    assert region['physical_fixed_neighbors'] == cfg['fixed_poses']
    for field in ['capture_center', 'capture_radius', 'depletant_radius']:
        assert region[field] == cfg[field]
    assert region['activity'] == cfg['reservoir_density']
    assert region['physical_metric'] == cfg['metadata']
    assert region['shape_sha256'] == shape_hash == region['gaussian_chart']['shape_sha256']


def geometry_probe(region, cfg, audit, count, seed):
    lo, hi = region.get('minimum_mahalanobis_radius', 0.), region['mahalanobis_radius']
    latent = sample_shell(np.random.default_rng(seed), count, lo, hi)
    poses, log_j = decode_latent(latent, region['gaussian_chart'], region['fixed_neighbor'])
    back = encode_matrix(poses, region['gaussian_chart'], region['fixed_neighbor'])
    error = float(np.max(np.abs(back-latent)))
    assert error < 1e-8
    q = native_q(poses, cfg['metadata'])
    qmin, qmax = region['minimum_original_q'], region.get('maximum_original_q', math.inf)
    rows = []
    for i, pose in enumerate(poses):
        gaps = audit.gaps(pose)
        hard = all(gap >= 0 for gap in gaps)
        capture = bool(np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center']) <= cfg['capture_radius'])
        original_q_valid = qmin <= q[i] <= qmax
        rows.append({'draw': i, 'latent': latent[i].tolist(), 'latent_radius': float(np.linalg.norm(latent[i])),
                     'pose': pose, 'log_pose_jacobian': float(log_j[i]), 'original_q': float(q[i]),
                     'original_q_valid': bool(original_q_valid), 'hard_valid_all_neighbors': hard,
                     'capture_valid': capture, 'minimum_gap_by_neighbor_A': gaps,
                     'target_geometry_valid': bool(hard and capture and original_q_valid)})
    k = sum(row['target_geometry_valid'] for row in rows)
    interval = [0. if k == 0 else float(beta.ppf(.025, k, count-k+1)),
                1. if k == count else float(beta.ppf(.975, k+1, count-k))]
    return {'draws': count, 'valid': k, 'fraction': k/count, 'clopper_pearson_95_fraction_interval': interval,
            'hard_valid': sum(row['hard_valid_all_neighbors'] for row in rows),
            'original_q_valid': sum(row['original_q_valid'] for row in rows),
            'capture_valid': sum(row['capture_valid'] for row in rows), 'seed': seed,
            'latent_shell_volume': shell_volume(lo, hi), 'maximum_matrix_backmap_error': error,
            'log_pose_jacobian_range': [float(log_j.min()), float(log_j.max())]}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'runs/native-cover-mixture-4x16384-l64-20260920/AB/provenance/config.json')
    parser.add_argument('--model', type=Path, default=ROOT/'runs/native-ab-covariance-guide-20260920/model.json')
    parser.add_argument('--cost-source', type=Path, default=ROOT/'runs/native-ab-refined-selected-8x16384-l64-20260920/assessment-streaming.json')
    parser.add_argument('--competitor-region', type=Path)
    parser.add_argument('--probes', type=int, default=256)
    args = parser.parse_args()
    started = time.monotonic()
    out = args.out.resolve()
    if out.exists():
        parser.error('Use a fresh immutable preparation directory')
    assert args.probes > 0 and sha(args.model) == ORIGINAL_MODEL_SHA
    cfg, model, cost = map(read, [args.config, args.model, args.cost_source])
    assert len(cfg['fixed_poses']) == 2
    assert cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
    assert cost['all_rows_and_hashes_validated'] and cost['complete']
    assert cost['populations'][0]['physical_signature']['config_sha256'] == sha(args.config)
    shape_path = Path(cfg['shape'])
    if not shape_path.is_absolute():
        shape_path = args.config.parent / shape_path
    shape_hash = sha(shape_path)
    assert model['shape_sha256'] == shape_hash
    archive = out/'provenance'
    archive.mkdir(parents=True)
    inputs = {'input-config.json': args.config, 'original-model.json': args.model,
              'shape.json': shape_path, 'cost-source-assessment.json': args.cost_source,
              'prepare_native_tail_regions.py': Path(__file__),
              'prepare_native_confirmation_atlas.py': ROOT/'tools/prepare_native_confirmation_atlas.py',
              'prepare_smc_normalizer_atlas.py': ROOT/'tools/prepare_smc_normalizer_atlas.py'}
    if args.competitor_region:
        inputs['original-competitor-region.json'] = args.competitor_region
    for name, source in inputs.items():
        shutil.copy2(source, archive/name)
    frozen_cfg = copy.deepcopy(cfg)
    frozen_cfg['shape'] = str(archive/'shape.json')
    write(out/'config.json', frozen_cfg)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'original_model_sha256': ORIGINAL_MODEL_SHA, 'original_config_sha256': sha(args.config),
                'config_sha256': sha(out/'config.json'), 'shape_sha256': shape_hash,
                'native_shells': SHELLS, 'native_original_q_bounds_inclusive': [0., 1.],
                'probes_per_region': args.probes, 'probe_seeds': [99271010+1009*i for i in range(4 if args.competitor_region else 3)],
                'proposal_law': 'Independent standard-normal direction in R6; r=[lo^6+u*(hi^6-lo^6)]^(1/6), u uniform[0,1). No rejection/resampling.',
                'jacobian': 'det(L)/(ell^3*pi^2*(1+|Cayley|^2)^2), normalized SO(3) Haar',
                'proposed_physical_budget_per_native_shell': {'populations': 4, 'draws_per_population': 8192, 'lambda_ratio': 64., 'cloud_replicates': 2},
                'geometry_gate': 'Recommend fixed pilot if at least one static target-valid probe is observed; zero hits leaves cost/support unresolved and is not grounds to claim zero physical mass.',
                'physical_production': False, 'no_beyond_12_claim': 'The reference targets only their stated shells; native mass beyond original-chart radius12 remains outside this preparation.',
                'input_sha256': {name: sha(archive/name) for name in inputs}}
    write(out/'preparation-protocol.json', protocol)
    regions = []
    for lo, hi in SHELLS:
        name = f'native-shell-{int(lo)}-{int(hi)}'
        region = {'frozen_at_utc': protocol['created_utc'], 'source_model_sha256': ORIGINAL_MODEL_SHA,
                  'fixed_neighbor': cfg['fixed_poses'][0], 'physical_fixed_neighbors': cfg['fixed_poses'],
                  'capture_center': cfg['capture_center'], 'capture_radius': cfg['capture_radius'],
                  'activity': cfg['reservoir_density'], 'depletant_radius': cfg['depletant_radius'],
                  'shape_sha256': shape_hash, 'physical_metric': cfg['metadata'], 'gaussian_chart': model,
                  'minimum_mahalanobis_radius': lo, 'mahalanobis_radius': hi,
                  'minimum_original_q': 0., 'maximum_original_q': 1.,
                  'definition': f'Original-chart {lo}<r<={hi}, original0<=q<=1, capture and fullAB hard-valid. Boundaries of zero measure do not alter target mass.',
                  'selection': 'Frozen original coordinate chart; uniform shell reference, no Gaussian proposal or newly fitted weights.'}
        validate_physics(region, cfg, shape_hash)
        write(out/f'{name}.json', region)
        regions.append((name, region))
    if args.competitor_region:
        competitor = read(args.competitor_region)
        validate_physics(competitor, cfg, shape_hash)
        assert competitor.get('minimum_mahalanobis_radius', 0.) == 0 and competitor['mahalanobis_radius'] == 3
        assert competitor['minimum_original_q'] == 5 and 'maximum_original_q' not in competitor
        shutil.copy2(args.competitor_region, out/'competitor-r3.json')
        assert sha(args.competitor_region) == sha(out/'competitor-r3.json')
        regions.append(('competitor-r3', competitor))
    audit = AtomUnionAudit(read(shape_path), cfg['fixed_poses'])
    fixed_pair_gaps = audit.gaps(cfg['fixed_poses'][0])
    assert fixed_pair_gaps[1] >= 0, 'The physical fixed neighbors must not clash'
    cpu_per_valid = cost['cpu_seconds'] / cost['regions']['native']['nonzero']
    reports = []
    for i, (name, region) in enumerate(regions):
        probe, rows = geometry_probe(region, cfg, audit, args.probes, protocol['probe_seeds'][i])
        probe.update(name=name, region_file=str(out/f'{name}.json'), region_sha256=sha(out/f'{name}.json'))
        probe['forecast_CPU_seconds_for_4x8192'] = 32768 * probe['fraction'] * cpu_per_valid
        probe['geometry_interval_CPU_seconds_for_4x8192'] = [32768*p*cpu_per_valid for p in probe['clopper_pearson_95_fraction_interval']]
        probe['fixed_pilot_geometry_reasonable'] = probe['valid'] > 0
        with (out/f'{name}-geometry-probes.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        reports.append(probe)
    result = {'complete': True, 'protocol_sha256': sha(out/'preparation-protocol.json'), 'regions': reports,
              'cost_source_CPU_seconds_per_valid': cpu_per_valid, 'fixed_neighbors_pair_gap_A': fixed_pair_gaps[1],
              'wall_seconds': time.monotonic()-started,
              'cost_scope': 'Historical sampling CPU per valid pose includes its rejected-draw overhead; multiplied by new static success fractions. Geometry intervals omit changes in cloud cost, overhead and physical-weight variance; not runtime or precision guarantees.',
              'scope': 'Static independent geometry and immutable physical regions only. No clouds, new physical masses, trajectory, or global native/competing coverage claim.',
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'report.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
