#!/usr/bin/env python3
"""Freeze a complete Cayley RMS cover and probe its geometry without clouds.

This finite-chart construction requires one reference, zero member centroid,
positive definite guarded rotational moment, and a complete angular cap below
pi. It preserves the original q window and every physical neighbor. The matrix
guard is a floating-point margin, not a formal interval certificate.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_native_region_reference import native_q
from prepare_native_confirmation_atlas import AtomUnionAudit, ELL, coordinates, laboratory_model
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive_model(metric, fixed, shape_sha256, q_max=2., ell=ELL):
    """Construct the cover from metric geometry; no observed pose enters it."""
    require(math.isfinite(q_max) and q_max > 0, 'Positive finite q upper bound required')
    require(math.isfinite(ell) and ell > 0, 'Positive finite angular length required')
    require(len(metric['native_poses']) == 1, 'Exactly one native reference required')
    require(bool(metric['rigid_members']), 'Nonempty rigid-member list required')
    delta, alpha = metric['member_error_scale'], metric['angle_error_scale_deg']
    require(math.isfinite(delta) and delta > 0 and math.isfinite(alpha) and 0 < alpha <= 180,
            'Invalid original registration scales')
    members, _, _ = arrays(metric['rigid_members'])
    centroid = members.mean(axis=0)
    require(np.array_equal(centroid, np.zeros(3)), 'This construction requires exactly zero member centroid')
    covariance = members.T @ members / len(members)
    magnitude = float(np.mean(np.sum(members*members, axis=1)))
    guard = 4096*np.finfo(float).eps*(1+magnitude)*len(members)
    trace = float(np.trace(covariance))
    upper = min(float(np.linalg.norm(covariance)), float(np.abs(covariance).sum(axis=1).max())) + guard
    bound_lower = max(0., trace-guard-upper)
    radius = q_max*delta
    require(math.isfinite(radius) and radius > 0, 'Unrepresentable cover displacement')
    nominal = min(math.pi, math.radians(alpha)*q_max)
    geometric = 2*math.asin(min(1., radius/(2*math.sqrt(bound_lower)))) if bound_lower > 0 else math.pi
    cap = min(nominal, geometric)
    require(math.isfinite(cap) and 0 < cap < math.pi,
            'Complete angular cap must be below pi; use the product-cover fallback')
    matrix = trace*np.eye(3)-covariance
    guarded = matrix-guard*np.eye(3)
    eigenvalues = np.linalg.eigvalsh(guarded)
    require(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0,
            'Rotational moment is singular or unresolved; use the product-cover fallback')
    f_t, _, f_r = arrays([fixed])
    n_t, _, n_r = arrays(metric['native_poses'])
    to_chart = f_r[0].T @ n_r[0]
    chart_matrix = to_chart @ guarded @ to_chart.T
    anchor = {'position': (f_r[0].T @ (n_t[0]-f_t[0])).tolist(),
              'rotation': (f_r[0].T @ n_r[0]).tolist()}
    k = math.cos(cap/2)**2
    sigma = np.zeros((6, 6)); sigma[:3, :3] = np.eye(3)
    sigma[3:, 3:] = ell**2/(4*k)*np.linalg.inv(chart_matrix)
    sigma = .5*(sigma+sigma.T)
    require(np.isfinite(sigma).all(), 'Unrepresentable chart covariance')
    lower = np.linalg.cholesky(sigma)
    require(np.isfinite(lower).all() and (np.diag(lower) > 0).all(), 'Invalid chart factor')
    model = {'schema': 'weighted-pose-mixture-v1', 'angular_length': ell,
             'shape_sha256': shape_sha256, 'coordinate_convention': 'anchor-body-relative',
             'anchors': [anchor], 'means': [[0.]*6], 'covariances': [sigma.tolist()], 'weights': [1.]}
    volume6 = math.pi**3*radius**6/6
    factor = 1/(8*math.pi**2*k**1.5*math.sqrt(float(np.linalg.det(guarded))))
    cmax = radius/(2*math.sqrt(k*eigenvalues.min()))
    require(math.isfinite(volume6*factor) and volume6*factor > 0, 'Invalid mapped cover volume')
    proof = {'q_max': q_max, 'mahalanobis_radius': radius, 'angular_length_A': ell,
             'centroid_A': centroid.tolist(), 'member_covariance_A2': covariance.tolist(),
             'A_A2': matrix.tolist(), 'guard_kappa_A2': guard, 'guarded_A_A2': guarded.tolist(),
             'guarded_A_eigenvalues_A2': eigenvalues.tolist(), 'left_chart_A_A2': chart_matrix.tolist(),
             'spectral_lambda_upper': upper, 'spectral_l_lower': bound_lower,
             'full_product_cover_angle_cap': cap, 'cosine_square_factor': k,
             'physical_cover_volume_bounds_A3': [volume6*factor/(1+cmax*cmax)**2, volume6*factor],
             'maximum_relative_Cayley_norm': cmax,
             'maximum_possible_rotation_angle_deg': math.degrees(2*math.atan(cmax)),
             'complete_cover_derivation': 'On q<=q_max, RMS^2>=|t-t0|^2+4*cos(h/2)^2*c^T*A*c; subtracting guard*I broadens the ellipsoid.',
             'outward_guard_scope': 'FP64 numerical margin, not a formal interval certificate',
             'sampling_law': 'Uniform latent six-ball; Gaussian parameters specify coordinates only'}
    return model, proof


def uniform_draws(model, fixed, radius, count, seed):
    """Sample the actual uniform six-ball law, preserving unconditional draws."""
    require(isinstance(count, int) and count > 0, 'Positive integer draw count required')
    require(math.isfinite(radius) and radius > 0, 'Positive finite latent radius required')
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 6))
    norms = np.linalg.norm(directions, axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), 'Numerical null in latent directions; no retry')
    directions /= norms[:, None]
    latent = directions*(radius*rng.random(count)**(1/6))[:, None]
    lower = np.linalg.cholesky(np.asarray(model['covariances'][0]))
    x = latent @ lower.T + np.asarray(model['means'][0])
    ell = model['angular_length']; cayley = x[:, 3:]/ell
    anchor = model['anchors'][0]
    relative_rotation = Rotation.from_quat(np.c_[cayley, np.ones(count)]).as_matrix() @ np.asarray(anchor['rotation'])
    relative_position = x[:, :3]+np.asarray(anchor['position'])
    f_t, _, f_r = arrays([fixed])
    world_rotation = f_r[0] @ relative_rotation
    world_position = relative_position @ f_r[0].T + f_t[0]
    quaternion = Rotation.from_matrix(world_rotation).as_quat()[:, [3, 0, 1, 2]]
    poses = [{'position': t.tolist(), 'orientation': q.tolist()} for t, q in zip(world_position, quaternion)]
    log_det = float(np.log(np.diag(lower)).sum())
    log_jacobian = log_det-3*math.log(ell)-2*math.log(math.pi)-2*np.log1p(np.sum(cayley*cayley, axis=1))
    return poses, latent, x, log_jacobian


def check_coordinates(model, fixed, poses, latent, x, log_jacobian):
    relative = relative_poses(poses, fixed)
    recovered = coordinates(relative, model['anchors'][0], model['angular_length'])
    density = Density(model).evaluate(relative)[0]
    normal_log = -3*math.log(2*math.pi)-.5*np.sum(latent*latent, axis=1)
    world = Density(laboratory_model(model, fixed)).evaluate(poses)[0]
    checks = {'maximum_latent_reconstruction_error': float(np.max(np.abs(recovered-x))),
              'maximum_density_reconstruction_error': float(np.max(np.abs(density-(normal_log-log_jacobian)))),
              'maximum_world_body_density_error': float(np.max(np.abs(world-density)))}
    require(max(checks.values()) < 2e-10, 'Independent coordinate or physical-density check failed')
    return checks


def support_audit(model, fixed, metric, radius, paths, q_min, q_max):
    """Check existing poses without fitting or changing the geometric proposal."""
    reports = []
    for path in paths:
        path = Path(path)
        rows = read(path) if path.suffix == '.json' else [json.loads(line) for line in path.read_text().splitlines()]
        selected = [r['pose'] for r in rows if q_min < native_q(metric, r['pose']) < q_max]
        require(bool(selected), f'No original-window poses in {path}')
        norms = Density(model).evaluate(relative_poses(selected, fixed))[1][:, 0]
        require(float(norms.max()) <= radius*(1+1e-12), 'Known target pose lies outside complete cover')
        reports.append({'source': str(path.resolve()), 'sha256': sha(path), 'selected_original_window_poses': len(selected),
                        'maximum_latent_radius': float(norms.max()), 'all_inside_complete_cover': True})
    return reports


def prepare(config_path, output, count, seed, q_min=1., q_max=2., anchor_index=0, source_poses=()):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    require(not output.exists(), 'Use a fresh output directory')
    require(math.isfinite(q_min) and 0 <= q_min < q_max, 'Require 0<=q_min<q_max')
    cfg = read(config_path)
    require(0 <= anchor_index < len(cfg['fixed_poses']), 'Anchor outside physical neighbor list')
    shape_path = Path(cfg['shape'])
    if not shape_path.is_absolute(): shape_path = config_path.parent/shape_path
    shape_hash = sha(shape_path); fixed = cfg['fixed_poses'][anchor_index]
    model, proof = derive_model(cfg['metadata'], fixed, shape_hash, q_max)
    poses, latent, x, log_jacobian = uniform_draws(model, fixed, proof['mahalanobis_radius'], count, seed)
    checks = check_coordinates(model, fixed, poses, latent, x, log_jacobian)
    support = support_audit(model, fixed, cfg['metadata'], proof['mahalanobis_radius'], source_poses, q_min, q_max)
    (output/'provenance').mkdir(parents=True)
    inputs = {'input-config.json': config_path, 'shape.json': shape_path,
              'prepare_cayley_rms_cover.py': Path(__file__)}
    for name in ('prepare_native_confirmation_atlas.py', 'prepare_smc_normalizer_atlas.py', 'analyze_native_region_reference.py'):
        inputs[name] = ROOT/'tools'/name
    for name, path in inputs.items(): shutil.copy2(path, output/'provenance'/name)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'draws': count, 'seed': seed,
                'q_min': q_min, 'q_max': q_max, 'interval_endpoints': 'Both open',
                'physical_fixed_neighbors': cfg['fixed_poses'], 'proposal_anchor_index': anchor_index,
                'no_clouds': True, 'no_fitted_data': True, 'proof': proof,
                'archived_sha256': {name: sha(output/'provenance'/name) for name in inputs}}
    write(output/'protocol.json', protocol); write(output/'model.json', model)
    archived_config = copy.deepcopy(cfg); archived_config['shape'] = str(output/'provenance/shape.json')
    write(output/'config.json', archived_config)
    region = {'frozen_at_utc': protocol['created_utc'], 'fixed_neighbor': copy.deepcopy(fixed),
              'physical_fixed_neighbors': copy.deepcopy(cfg['fixed_poses']),
              'capture_center': cfg['capture_center'], 'capture_radius': cfg['capture_radius'],
              'shape_sha256': shape_hash, 'activity': cfg['reservoir_density'], 'depletant_radius': cfg['depletant_radius'],
              'physical_metric': copy.deepcopy(cfg['metadata']), 'gaussian_chart': model,
              'mahalanobis_radius': proof['mahalanobis_radius'], 'minimum_original_q': q_min, 'maximum_original_q': q_max,
              'minimum_original_q_inclusive': False, 'maximum_original_q_inclusive': False,
              'definition': 'Complete geometric outer Cayley RMS cover intersected with original q window, capture, and all physical hard exclusions.',
              'complete_coverage_proof': proof, 'protocol_sha256': sha(output/'protocol.json')}
    write(output/'region.json', region)
    atom_audit = AtomUnionAudit(read(shape_path), cfg['fixed_poses'])
    counts = {k: 0 for k in ('capture_valid', 'hard_valid', 'q_window', 'valid_window', 'valid_window_all_depletion_contacts')}
    volume6 = math.pi**3*proof['mahalanobis_radius']**6/6
    start, cpu = time.monotonic(), time.process_time()
    with (output/'geometry-probes.jsonl').open('w') as handle:
        for i, pose in enumerate(poses):
            q = native_q(cfg['metadata'], pose); gaps = atom_audit.gaps(pose)
            capture = bool(np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center']) <= cfg['capture_radius'])
            hard = all(g >= 0 for g in gaps); inside = q_min < q < q_max
            contacts = [g < 2*cfg['depletant_radius'] for g in gaps]
            flags = {'capture_valid': capture, 'hard_valid': hard, 'q_window': inside,
                     'valid_window': capture and hard and inside,
                     'valid_window_all_depletion_contacts': capture and hard and inside and all(contacts)}
            for key, value in flags.items(): counts[key] += int(value)
            row = {'draw': i, 'pose': pose, 'original_q': q, 'latent_radius': float(np.linalg.norm(latent[i])),
                   'log_jacobian': float(log_jacobian[i]), 'log_uniform_pose_density': -math.log(volume6)-float(log_jacobian[i]),
                   'minimum_gap_by_neighbor_A': gaps, 'depletion_contact_by_neighbor': contacts, **flags}
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            if (i+1) % 512 == 0:
                write(output/'progress.json', {'completed': i+1, 'requested': count, 'counts': counts})
    report = {'complete': True, 'draws': count, 'seed': seed, 'counts': counts, 'checks': checks,
              'observed_pose_support': support, 'proof': proof, 'CPU_seconds': time.process_time()-cpu,
              'wall_seconds': time.monotonic()-start, 'model_sha256': sha(output/'model.json'),
              'region_sha256': sha(output/'region.json'), 'config_sha256': sha(output/'config.json'),
              'protocol_sha256': sha(output/'protocol.json'), 'probe_sha256': sha(output/'geometry-probes.jsonl'),
              'scope': 'Fixed-count independent geometry only; no clouds or physical statistical weights.'}
    write(output/'report.json', report)
    return report


def audit_preparation(preparation, output, source_poses=()):
    """Replay frozen coordinates and verify provenance without fresh atom queries."""
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    require(not output.exists(), 'Use a fresh audit directory; frozen inputs remain immutable')
    report = read(preparation/'report.json'); protocol = read(preparation/'protocol.json')
    cfg = read(preparation/'config.json'); model = read(preparation/'model.json')
    region_paths = [p for p in (preparation/'region.json', preparation/'region-shoulder.json') if p.exists()]
    require(len(region_paths) == 1, 'Expected one frozen region definition')
    region_path = region_paths[0]; region = read(region_path)
    require(report['complete'], 'Frozen preparation did not complete')
    require(report['probe_sha256'] == sha(preparation/'geometry-probes.jsonl'), 'Probe hash mismatch')
    for key, name in [('model_sha256', 'model.json'), ('config_sha256', 'config.json'), ('protocol_sha256', 'protocol.json')]:
        require(report[key] == sha(preparation/name), f'{name} hash mismatch')
    require(report['region_sha256'] == sha(region_path), 'Region hash mismatch')
    for name, digest in protocol['archived_sha256'].items():
        require(sha(preparation/'provenance'/name) == digest, f'Archived source changed: {name}')
    require(region['gaussian_chart'] == model, 'Model differs from the chart embedded in region')
    require(region['physical_metric'] == cfg['metadata'], 'Physical registration metric changed')
    require(region['physical_fixed_neighbors'] == cfg['fixed_poses'], 'Physical neighbor list changed')
    for rkey, ckey in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                      ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius')]:
        require(region[rkey] == cfg[ckey], f'Physical field changed: {rkey}')
    require(not region['minimum_original_q_inclusive'] and not region['maximum_original_q_inclusive'],
            'This preparation tool uses an open original-q interval')
    fixed = region['fixed_neighbor']; require(fixed in cfg['fixed_poses'], 'Chart anchor is not a physical neighbor')
    rebuilt, proof = derive_model(cfg['metadata'], fixed, model['shape_sha256'],
                                   region['maximum_original_q'], model['angular_length'])
    require(proof['mahalanobis_radius'] == region['mahalanobis_radius'], 'Cover radius differs')
    numerical_error = 0.
    for key in ('anchors', 'means', 'covariances', 'weights'):
        if key == 'anchors':
            pairs = [(rebuilt[key][0][name], model[key][0][name]) for name in ('position', 'rotation')]
        else:
            pairs = [(rebuilt[key], model[key])]
        for expected, observed in pairs:
            error = float(np.max(np.abs(np.asarray(expected)-np.asarray(observed))))
            numerical_error = max(numerical_error, error)
            require(error < 2e-11, 'Frozen chart differs from geometric construction')
    poses, latent, x, log_j = uniform_draws(model, fixed, region['mahalanobis_radius'], report['draws'], report['seed'])
    checks = check_coordinates(model, fixed, poses, latent, x, log_j)
    rows = [json.loads(line) for line in (preparation/'geometry-probes.jsonl').read_text().splitlines()]
    require(len(rows) == report['draws'], 'Incomplete fixed-count geometry output')
    maximum_pose_error = maximum_q_error = maximum_jacobian_error = 0.
    count_window = count_valid = 0
    for i, (row, pose) in enumerate(zip(rows, poses)):
        require(row['draw'] == i, 'Nonsequential geometry draw indices')
        pose_error = max(float(np.max(np.abs(np.asarray(row['pose'][key])-np.asarray(pose[key]))))
                         for key in ('position', 'orientation'))
        q = native_q(cfg['metadata'], pose)
        maximum_pose_error = max(maximum_pose_error, pose_error)
        maximum_q_error = max(maximum_q_error, abs(q-row['original_q']))
        maximum_jacobian_error = max(maximum_jacobian_error, abs(float(log_j[i])-row['log_jacobian']))
        window = region['minimum_original_q'] < q < region['maximum_original_q']
        capture = np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center']) <= cfg['capture_radius']
        require(bool(window) == row['q_window'] and bool(capture) == row['capture_valid'], 'Geometry label changed')
        # Hard labels are checked against the archived gaps, not fresh atoms.
        hard = all(g >= 0 for g in row['minimum_gap_by_neighbor_A'])
        require(hard == row['hard_valid'], 'Stored gaps disagree with hard label')
        count_window += int(window); count_valid += int(window and capture and hard)
    require(max(maximum_pose_error, maximum_q_error, maximum_jacobian_error) < 2e-10, 'Frozen coordinate replay failed')
    target_name = 'valid_shoulder' if 'valid_shoulder' in report['counts'] else 'valid_window'
    require(report['counts']['q_window'] == count_window and report['counts'][target_name] == count_valid,
            'Reported geometry counts differ from frozen rows')
    support = support_audit(model, fixed, cfg['metadata'], region['mahalanobis_radius'], source_poses,
                            region['minimum_original_q'], region['maximum_original_q'])
    (output/'provenance').mkdir(parents=True)
    shutil.copy2(__file__, output/'provenance/prepare_cayley_rms_cover.py')
    result = {'complete': True, 'preparation': str(preparation), 'replayed_draws': len(rows),
              'maximum_model_reconstruction_error': numerical_error, 'maximum_pose_replay_error': maximum_pose_error,
              'maximum_original_q_replay_error': maximum_q_error, 'maximum_jacobian_replay_error': maximum_jacobian_error,
              'checks': checks, 'observed_pose_support': support, 'proof': proof,
              'input_sha256': {str(preparation/name): sha(preparation/name) for name in
                              ('report.json', 'protocol.json', 'config.json', 'model.json', 'geometry-probes.jsonl')},
              'auditor_sha256': sha(__file__),
              'scope': 'All frozen coordinates, q/capture labels, densities and source hashes verified. Hard gaps reused; no fresh atomic or Poisson queries.'}
    write(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit-preparation', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probes', type=int, default=4096)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--q-min', type=float, default=1.)
    parser.add_argument('--q-max', type=float, default=2.)
    parser.add_argument('--anchor-index', type=int, default=0)
    parser.add_argument('--source-poses', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.audit_preparation:
        require(args.config is None and args.seed is None, 'Audit uses the frozen configuration and seed')
        report = audit_preparation(args.audit_preparation, args.out, args.source_poses)
    else:
        require(args.config is not None and args.seed is not None, 'Preparation requires --config and --seed')
        report = prepare(args.config, args.out, args.probes, args.seed, args.q_min, args.q_max,
                         args.anchor_index, args.source_poses)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
