#!/usr/bin/env python3
"""Freeze pose-centered, geometry-scaled references without fitting their widths."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time

import numpy as np

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import uniform_draws, check_coordinates, read, write, sha, require
from prepare_intermediate_local_region import ROOT, GUIDES, BINARY, BINARY_SHA
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import arrays

RADII = (.5, 1., 2.)
PROBE_SEED = 100801010
PROBE_COUNT = 512
PRODUCTION_SEEDS = (100901010, 101001010, 101101010)
SAMPLES = 16384
REPLICATES = 4
EXTREMES = ROOT/'runs/ab-intermediate-partition-extremes-20260920/analysis.json'


def geometry_model(metric, fixed, peak, shape_hash, ell):
    """A chart ball is contained in the member-RMS ball, not a cover of it."""
    require(math.isfinite(ell) and ell > 0, 'Positive angular length required')
    members, _, _ = arrays(metric['rigid_members'])
    require(len(members) > 0 and np.array_equal(members.mean(axis=0), np.zeros(3)),
            'Exactly centered, nonempty rigid-member geometry required')
    moment = members.T@members/len(members)
    a = np.trace(moment)*np.eye(3)-moment
    eigenvalues = np.linalg.eigvalsh(a)
    require(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0,
            'Rotational moment must be positive definite')
    ft, _, fr = arrays([fixed]); pt, _, pr = arrays([peak])
    relative = fr[0].T@pr[0]
    a_anchor = relative@a@relative.T
    sigma = np.zeros((6, 6)); sigma[:3, :3] = np.eye(3)
    sigma[3:, 3:] = ell**2/4*np.linalg.inv(a_anchor)
    sigma = .5*(sigma+sigma.T)
    np.linalg.cholesky(sigma)
    model = dict(schema='weighted-pose-mixture-v1', angular_length=ell,
                 coordinate_convention='anchor-body-relative', shape_sha256=shape_hash,
                 anchors=[dict(position=(fr[0].T@(pt[0]-ft[0])).tolist(), rotation=relative.tolist())],
                 means=[[0.]*6], covariances=[sigma.tolist()], weights=[1.])
    proof = dict(member_moment_A2=moment.tolist(), A_A2=a.tolist(),
                 A_anchor_A2=a_anchor.tolist(), A_eigenvalues_A2=eigenvalues.tolist(),
                 radius_units='Angstrom-equivalent rigid-member displacement',
                 exact_RMS_squared='|dt|^2 + 4*c^T*A_anchor*c/(1+|c|^2)',
                 chart_radius_squared='|dt|^2 + 4*c^T*A_anchor*c',
                 jacobian='1/(8*pi^2*sqrt(det(A))*(1+|c|^2)^2)',
                 support='Each chart ball is contained in the member-RMS ball; not a complete RMS cover. Original q, capture and both AB hard masks are retained.',
                 construction='No weight fit, covariance fit, or eigenvalue guard; this defines a local chart, not an enclosing cover.',
                 maximum_rotation_angle_deg_by_radius={str(r):math.degrees(2*math.atan(r/(2*math.sqrt(eigenvalues.min())))) for r in RADII})
    return model, proof


def verify_member_geometry(metric, fixed, peak, poses, latent, x, model, proof, log_j):
    members, _, _ = arrays(metric['rigid_members'])
    pt, _, pr = arrays([peak]); positions, _, rotations = arrays(poses)
    world_peak = members@pr[0].T+pt[0]
    world = np.einsum('nij,mj->nmi', rotations, members)+positions[:, None, :]
    actual = np.mean(np.sum((world-world_peak)**2, axis=2), axis=1)
    c = x[:, 3:]/model['angular_length']; c2 = np.sum(c*c, axis=1)
    quadratic = np.einsum('ni,ij,nj->n', c, proof['A_anchor_A2'], c)
    predicted = np.sum(x[:, :3]**2, axis=1)+4*quadratic/(1+c2)
    radius2 = np.sum(latent*latent, axis=1)
    expected_j = -math.log(8*math.pi**2)-.5*np.linalg.slogdet(proof['A_A2'])[1]-2*np.log1p(c2)
    error = max(float(np.max(np.abs(actual-predicted))), float(np.max(np.abs(log_j-expected_j))))
    require(error < 2e-10 and np.max(actual-radius2) < 2e-10, 'Member geometry or Jacobian identity failed')
    return dict(maximum_RMS_identity_error=float(np.max(np.abs(actual-predicted))),
                maximum_Jacobian_identity_error=float(np.max(np.abs(log_j-expected_j))),
                maximum_RMS_squared_minus_chart_radius_squared=float(np.max(actual-radius2)))


def prepare(out, campaign_root=None, seed_base=None):
    require(not out.exists(), 'Use a fresh output directory')
    if campaign_root is not None:
        campaign_root = Path(campaign_root).resolve()
        require(not campaign_root.exists(), 'Campaign root must be fresh')
    if seed_base is not None:
        require(isinstance(seed_base, int) and seed_base >= 0, 'Nonnegative integer seed required')
    production_seeds = PRODUCTION_SEEDS if seed_base is None else tuple(seed_base+100000*i for i in range(len(RADII)))
    freeze = read(GUIDES/'freeze.json')
    require(sha(GUIDES/'config.json') == freeze['config_sha256'], 'Changed config')
    require(sha(GUIDES/'protocol.json') == freeze['protocol_sha256'], 'Changed guide protocol')
    require(sha(GUIDES/'model-weighted.json') == freeze['model_sha256']['weighted'], 'Changed angular-length source')
    for name, digest in freeze['archived_sha256'].items():
        require(sha(GUIDES/'provenance'/name) == digest, 'Changed guide provenance')
    require(sha(BINARY) == BINARY_SHA, 'Changed physical kernel')
    config = read(GUIDES/'config.json'); source = read(GUIDES/'provenance/source-region.json')
    require(source['physical_metric'] == config['metadata'] and source['physical_fixed_neighbors'] == config['fixed_poses'], 'Changed physical target')
    require((source['minimum_original_q'], source['maximum_original_q'], source['minimum_original_q_inclusive'], source['maximum_original_q_inclusive']) == (2., 5., True, False), 'Changed q window')
    require(source['activity'] == config['reservoir_density'] == .035 and source['depletant_radius'] == config['depletant_radius'] == 1.5, 'Changed depletion parameters')
    require(source['capture_center'] == config['capture_center'] and source['capture_radius'] == config['capture_radius'] == 18., 'Changed capture domain')
    shape = Path(config['shape']); require(sha(shape) == source['shape_sha256'], 'Changed shape')
    extremes = read(EXTREMES)
    selected = next(p for p in extremes['pieces'] if p['name'] == 'remainder')['top_poses'][0]
    require(selected['draw'] == 151795 and selected['seed'] == 100601010, 'Unexpected selected point')
    peak = selected['pose']; fixed = source['fixed_neighbor']
    require(fixed in config['fixed_poses'], 'Keep the actual physical neighbor as chart frame')
    ell = read(GUIDES/'model-weighted.json')['angular_length']
    model, proof = geometry_model(config['metadata'], fixed, peak, source['shape_sha256'], ell)
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {'config.json':GUIDES/'config.json', 'source-region.json':GUIDES/'provenance/source-region.json',
               'guide-protocol.json':GUIDES/'protocol.json', 'angular-length-source.json':GUIDES/'model-weighted.json',
               'shape.json':shape, 'extremes.json':EXTREMES, 'guide-freeze.json':GUIDES/'freeze.json'}
    for name in ('prepare_peak_neighborhood.py', 'prepare_cayley_rms_cover.py',
                 'prepare_native_confirmation_atlas.py', 'prepare_smc_normalizer_atlas.py',
                 'analyze_native_region_reference.py', 'prepare_intermediate_local_region.py'):
        sources[name] = ROOT/'tools'/name
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    frozen_config = copy.deepcopy(config); frozen_config['shape'] = str(archive/'shape.json')
    write(out/'config.json', frozen_config); write(out/'model.json', model)
    write(out/'selected-pose.json', selected); write(out/'geometry.json', proof)
    campaigns = []
    for radius, seed in zip(RADII, production_seeds):
        label = f'{radius:g}'.replace('.', 'p')
        region = {key:copy.deepcopy(source[key]) for key in (
            'fixed_neighbor','physical_fixed_neighbors','capture_center','capture_radius',
            'shape_sha256','activity','depletant_radius','physical_metric',
            'minimum_original_q','maximum_original_q','minimum_original_q_inclusive','maximum_original_q_inclusive')}
        region.update(gaussian_chart=model, mahalanobis_radius=radius, minimum_mahalanobis_radius=0.,
                      definition='Pose-centered geometry-scaled local ball intersected with unchanged original intermediate domain; not a basin or complete-domain definition.')
        name = f'region-r{label}.json'; write(out/name, region)
        campaigns.append(dict(radius=radius, region=str(out/name), region_sha256=sha(out/name),
                              output=str(campaign_root/f'r{label}' if campaign_root is not None else ROOT/f'runs/ab-intermediate-new-peak-r{label}-reference-4x16384-l64-20260920'),
                              samples=SAMPLES, replicates=REPLICATES, seed_base=seed,
                              seeds=[seed+1009*i for i in range(REPLICATES)]))
    protocol = dict(schema='geometry-scaled-peak-reference-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    selection='Center selected from historical remainder maximum. Geometry and radii now fixed before fresh probes or physical draws; center selection is not held-out basin discovery.',
                    sampling='Uniform latent six-ball with exact physical Jacobian, original masks, fixed unconditional N, two independent clouds averaged linearly; no retry or adaptive stop.',
                    uncertainty='Row and independent-population estimates; observed errors cannot bound unseen tails. The new nested balls overlap and cannot be added to one another or to the old complete partition.',
                    analysis='Full-N radial and old-weighted-radius <=32 versus >32 poststratification. Each campaign is analyzed separately; retain complete remainder support.',
                    executable=str(BINARY), executable_sha256=BINARY_SHA,
                    lambda_ratio=64., cloud_replicates=2, campaigns=campaigns,
                    probe_count_per_radius=PROBE_COUNT, probe_seeds=[PROBE_SEED+1009*i for i in range(len(RADII))],
                    config_sha256=sha(out/'config.json'), model_sha256=sha(out/'model.json'),
                    selected_pose_sha256=sha(out/'selected-pose.json'), geometry_sha256=sha(out/'geometry.json'),
                    archived_sha256={name:sha(archive/name) for name in sources})
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', {p.name:sha(p) for p in out.glob('*.json')})
    atom = AtomUnionAudit(read(shape), config['fixed_poses']); reports=[]
    start = time.process_time()
    for index, radius in enumerate(RADII):
        poses, latent, x, log_j = uniform_draws(model, fixed, radius, PROBE_COUNT, PROBE_SEED+1009*index)
        checks = check_coordinates(model, fixed, poses, latent, x, log_j)
        checks.update(verify_member_geometry(config['metadata'], fixed, peak, poses, latent, x, model, proof, log_j))
        valid=0; q_valid=0; minimum_q=math.inf; maximum_q=-math.inf
        label = f'{radius:g}'.replace('.', 'p'); path=out/f'probes-r{label}.jsonl'
        with path.open('w') as handle:
            import json
            for i, pose in enumerate(poses):
                q=native_q(config['metadata'], pose); gaps=atom.gaps(pose)
                inside=2<=q<5; capture=np.linalg.norm(np.asarray(pose['position'])-config['capture_center'])<=config['capture_radius']
                allowed=bool(inside and capture and min(gaps)>=0); valid+=allowed; q_valid+=inside
                minimum_q=min(minimum_q,q); maximum_q=max(maximum_q,q)
                handle.write(json.dumps(dict(draw=i,pose=pose,q=q,valid=allowed,gaps_A=gaps,latent_radius=float(np.linalg.norm(latent[i])),log_jacobian=float(log_j[i])),allow_nan=False)+'\n')
        reports.append(dict(radius=radius,draws=PROBE_COUNT,valid=valid,q_valid=q_valid,
                            minimum_observed_q=minimum_q,maximum_observed_q=maximum_q,
                            checks=checks,probe_sha256=sha(path)))
        print(json.dumps(reports[-1]),flush=True)
    write(out/'report.json',dict(complete=True,probes=reports,CPU_seconds=time.process_time()-start,
                                protocol_sha256=sha(out/'protocol.json'),scope='Cloud-free geometry only; no measured physical statistical weights.'))
    return protocol


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--campaign-root',type=Path,help='Fresh parent directory for the three declared campaign outputs')
    parser.add_argument('--seed-base',type=int,help='First population seed; radii offset by 100000 and populations by 1009')
    args=parser.parse_args(); prepare(args.out.resolve(), args.campaign_root, args.seed_base)
