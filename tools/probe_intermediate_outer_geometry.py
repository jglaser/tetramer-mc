#!/usr/bin/env python3
"""Freeze cloud-free outer weighted-chart probes; never estimate physical mass."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import norm

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import uniform_draws, check_coordinates, read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_intermediate_local_region import ROOT, GUIDES

RADII = (8., 16., 32.)
SEED = 100201010
COUNT = 2048


def affine_cover_bound(cover, weighted, radius):
    require(cover['anchors'] == weighted['anchors'], 'Charts require identical anchors')
    require(cover['angular_length'] == weighted['angular_length'], 'Angular lengths differ')
    lc = np.linalg.cholesky(cover['covariances'][0])
    lw = np.linalg.cholesky(weighted['covariances'][0])
    matrix = np.linalg.solve(lw, lc)
    shift = np.linalg.solve(lw, np.array(cover['means'][0])-weighted['means'][0])
    frobenius = np.linalg.norm(matrix)
    raw = float(np.linalg.norm(shift)+radius*frobenius)
    margin = 1e-10*(1+raw)
    return dict(affine_matrix=matrix.tolist(), affine_shift=shift.tolist(),
                cover_radius=radius, complete_weighted_radius_bound=raw+margin,
                numerical_outward_margin=margin,
                formula='u_weighted = shift + matrix*u_cover; norm <= norm(shift)+R_cover*Frobenius(matrix).',
                scope='Analytic enclosing bound with an FP64 outward margin, not formal interval arithmetic. Complete original q<=5 support inherits the existing geometric cover proof; this does not bound physical mass.')


def probe(out):
    require(not out.exists(), 'Use a fresh output directory')
    model = read(GUIDES/'model-weighted.json'); config = read(GUIDES/'config.json')
    freeze = read(GUIDES/'freeze.json')
    require(sha(GUIDES/'model-weighted.json') == freeze['model_sha256']['weighted'], 'Changed model')
    require(sha(GUIDES/'config.json') == freeze['config_sha256'], 'Changed config')
    shape = Path(config['shape']); require(sha(shape) == model['shape_sha256'], 'Changed shape')
    cover_path = GUIDES/'provenance/source-region.json'; cover = read(cover_path)
    require(sha(cover_path) == freeze['archived_sha256']['source-region.json'], 'Changed complete cover')
    bound = affine_cover_bound(cover['gaussian_chart'], model, cover['mahalanobis_radius'])
    diagnostic_path = ROOT/'runs/ab-intermediate-guided-tail-diagnostic-qualified-20260920/analysis.json'
    diagnostic = read(diagnostic_path)
    extrema = [dict(source=p['source_kind'], population=p['population'], draw=p['draw'], q=p['q'],
                    weighted_radius=p['guide_densities']['mixture']['mahalanobis_radii'][0],
                    original_log_importance_weight=p['original_log_importance_weight'])
               for p in diagnostic['top_pose_density_checks']]
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {'model-weighted.json': GUIDES/'model-weighted.json', 'config.json': GUIDES/'config.json', 'shape.json': shape,
               'source-region.json': cover_path, 'diagnostic.json': diagnostic_path,
               'probe_intermediate_outer_geometry.py': Path(__file__)}
    for name in ['prepare_cayley_rms_cover.py', 'prepare_native_confirmation_atlas.py',
                 'prepare_smc_normalizer_atlas.py', 'analyze_native_region_reference.py']:
        sources[name] = ROOT/'tools'/name
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), radii=list(RADII),
                    draws_per_radius=COUNT, seeds=[SEED+1009*i for i in range(len(RADII))],
                    model_sha256=sha(GUIDES/'model-weighted.json'), shape_sha256=sha(shape),
                    archive_sha256={name: sha(archive/name) for name in sources},
                    no_depletant_clouds=True, no_refitting=True,
                    selection='Radii chosen from previous local mass growth and observed outer guide contacts; all probes precede selection of physical campaign counts.',
                    sampling='Independent fixed-N uniform six-balls with every rejection retained. Atomic gaps checked only after original q and capture pass.',
                    q_window='2<=q<5; unchanged AB hard geometry and capture18',
                    complete_cover_bound=bound)
    write(out/'protocol.json', protocol)
    atom = AtomUnionAudit(read(shape), config['fixed_poses']); reports = []
    for index, radius in enumerate(RADII):
        tick = time.process_time(); seed = SEED+1009*index
        poses, latent, x, jac = uniform_draws(model, config['fixed_poses'][0], radius, COUNT, seed)
        checks = check_coordinates(model, config['fixed_poses'][0], poses, latent, x, jac)
        valid = capture_count = q_count = tested = 0
        path = out/f'r{int(radius)}-geometry.jsonl'
        with path.open('w') as handle:
            for i, pose in enumerate(poses):
                q = native_q(config['metadata'], pose)
                capture = math.dist(pose['position'], config['capture_center']) <= config['capture_radius']
                iq = 2 <= q < 5; capture_count += capture; q_count += iq
                gaps = atom.gaps(pose) if capture and iq else None
                if gaps is not None: tested += 1
                ok = gaps is not None and all(gap >= 0 for gap in gaps); valid += ok
                record = dict(draw=i, pose=pose, latent_radius=float(np.linalg.norm(latent[i])),
                              original_q=q, capture_valid=capture, q_valid=iq,
                              atomic_gaps_A=gaps, target_valid=ok, log_jacobian=float(jac[i]))
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        p = valid/COUNT; z = float(norm.ppf(.975)); denom = 1+z*z/COUNT
        mid = (p+z*z/(2*COUNT))/denom
        half = z/denom*math.sqrt(p*(1-p)/COUNT+z*z/(4*COUNT*COUNT))
        reports.append(dict(radius=radius, seed=seed, draws=COUNT, valid=valid,
                            capture_valid=capture_count, q_valid=q_count, atomic_checks=tested,
                            valid_fraction=p, Wilson_95_interval=[mid-half, mid+half],
                            reconstruction=checks, samples_sha256=sha(path), CPU_seconds=time.process_time()-tick))
        print(json.dumps(reports[-1]), flush=True)
    write(out/'analysis.json', dict(complete=True, protocol_sha256=sha(out/'protocol.json'),
                                   probes=reports, complete_cover_bound=bound, historical_extrema=extrema,
                                   scope='Geometry and observed-pose diagnostics only. No physical overlap weights, mass estimates or convergence claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); probe(args.out.resolve())
