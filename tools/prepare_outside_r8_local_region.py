#!/usr/bin/env python3
"""Freeze local integration regions around a chronological, unequilibrated cohort.

This command only fits geometry, validates coordinates, and writes a plan.
It never launches a physical normalizer or uses cohort occupancy as a weight.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from prepare_smc_normalizer_atlas import ROOT, Density, arrays, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration


def chart_coordinates(poses, chart, fixed):
    t, _, r = arrays(relative_poses(poses, fixed))
    anchor = chart['anchors'][0]
    q = Rotation.from_matrix(r@np.asarray(anchor['rotation']).T).as_quat()
    if np.any(q[:, 3] == 0):
        raise ValueError('Cayley seam: cannot fit a finite coordinate cloud')
    x = np.column_stack((t-anchor['position'], chart['angular_length']*q[:, :3]/q[:, 3, None]))
    assert np.isfinite(x).all()
    return x


def decode(u, chart, fixed):
    x = np.asarray(u)@np.linalg.cholesky(chart['covariances'][0]).T+chart['means'][0]
    ell, anchor = chart['angular_length'], chart['anchors'][0]
    cayley = x[:, 3:]/ell
    relative_r = Rotation.from_quat(np.column_stack((cayley, np.ones(len(x))))).as_matrix()@anchor['rotation']
    ft, _, fr = arrays([fixed])
    t = (x[:, :3]+anchor['position'])@fr[0].T+ft[0]
    r = fr[0]@relative_r
    quat = Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]
    poses = [dict(position=p.tolist(), orientation=q.tolist()) for p, q in zip(t, quat)]
    logdet = np.linalg.slogdet(np.linalg.cholesky(chart['covariances'][0]))[1]
    logj = logdet-3*np.log(ell)-2*np.log(np.pi)-2*np.log1p(np.sum(cayley*cayley, axis=1))
    return poses, logj


def uniform_ball(rng, radius, n):
    direction = rng.normal(size=(n, 6))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return radius*rng.random(n)[:, None]**(1/6)*direction


def geometry_probe(poses, config, shape):
    centers = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    ft, _, fr = arrays(config['fixed_poses'])
    assert len(ft) == 1
    fixed = centers@fr[0].T+ft[0]
    tree = cKDTree(fixed)
    t, _, r = arrays(poses)
    capture = np.linalg.norm(t-config['capture_center'], axis=1) <= config['capture_radius']
    hard = np.zeros(len(poses), dtype=bool)
    for k in np.flatnonzero(capture):
        moved = centers@r[k].T+t[k]
        neighbors = tree.query_ball_point(moved, radii+radii.max(), workers=1)
        collision = False
        for i, js in enumerate(neighbors):
            if not js:
                continue
            js = np.asarray(js)
            if np.any(np.sum((fixed[js]-moved[i])**2, axis=1) < (radii[i]+radii[js])**2):
                collision = True
                break
        hard[k] = not collision
    return capture, hard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/outside-r8-local-region-20260920/site0')
    parser.add_argument('--cohort', type=Path, default=ROOT/'runs/posterior-docking-mis-5000/assessment/outside-R8-cohort.json')
    parser.add_argument('--old-region', type=Path, default=ROOT/'runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json')
    parser.add_argument('--config', type=Path, default=ROOT/'runs/latent-region-uniform-ball-16384-l64/provenance/config.json')
    parser.add_argument('--binary', type=Path, default=ROOT/'runs/latent-region-radius5-16384-l64/provenance/latent-region-normalizer')
    parser.add_argument('--geometry-probes', type=int, default=1024)
    args = parser.parse_args()
    if args.geometry_probes < 1:
        parser.error('Require positive geometry probe count')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use a fresh output directory')
    archive = out/'provenance'
    archive.mkdir(parents=True)
    cohort, old, cfg = read(args.cohort), read(args.old_region), read(args.config)
    campaign = ROOT/'runs/posterior-docking-mis-5000'
    source_dir = campaign/'runs'/cohort['source_job']
    assert sha(args.old_region) == cohort['frozen_region_sha256']
    assert old['shape_sha256'] == sha(cfg['shape'])
    assert cfg['fixed_poses'] == [old['fixed_neighbor']] and cfg['metadata'] == old['physical_metric']
    assert cfg['capture_center'] == old['capture_center'] and cfg['capture_radius'] == old['capture_radius']
    assert cfg['depletant_radius'] == old['depletant_radius'] and cfg['reservoir_density'] == old['activity']
    assert all(sha(source_dir/name) == digest for name, digest in cohort['source_sha256'].items())
    frames = [json.loads(line) for line in (source_dir/'trajectory.jsonl').open()]
    old_chart, fixed = old['gaussian_chart'], old['fixed_neighbor']
    frame_q = registration([f['pose'] for f in frames], cfg['metadata'])
    frame_rad = Density(old_chart).evaluate(relative_poses([f['pose'] for f in frames], fixed))[1][:, 0]
    outside = np.flatnonzero((frame_q >= 5.) & (frame_rad > 8.))
    expected = outside[np.linspace(0, len(outside)-1, min(32, len(outside)), dtype=int)].tolist()
    assert [p['cycle'] for p in cohort['poses']] == expected
    assert all(p['pose'] == frames[p['cycle']]['pose'] for p in cohort['poses'])
    inputs = {'cohort.json': args.cohort, 'old-region.json': args.old_region, 'input-config.json': args.config,
        'shape.json': Path(cfg['shape']), 'latent-region-normalizer': args.binary,
        'source-trajectory.jsonl': source_dir/'trajectory.jsonl', 'source-summary.json': source_dir/'summary.json',
        'source-config.json': campaign/'configs'/f"{cohort['source_job']}.json",
        'prepare_outside_r8_local_region.py': Path(__file__),
        'prepare_smc_normalizer_atlas.py': ROOT/'tools/prepare_smc_normalizer_atlas.py',
        'prepare_deep_far_normalizer_atlas.py': ROOT/'tools/prepare_deep_far_normalizer_atlas.py',
        'run_latent_region_campaign.py': ROOT/'tools/run_latent_region_campaign.py'}
    for name, path in inputs.items():
        shutil.copy2(path, archive/name)
    cfg['shape'] = str(archive/'shape.json')
    write(out/'physical-config.json', cfg)
    poses = [p['pose'] for p in cohort['poses']]
    x = chart_coordinates(poses, old_chart, fixed)
    mean = x.mean(axis=0)
    centered = x-mean
    raw = centered.T@centered/len(x)
    raw = .5*(raw+raw.T)
    sigma0 = np.asarray(old_chart['covariances'][0])
    covariance = raw+.25*sigma0
    covariance = .5*(covariance+covariance.T)
    lower = np.linalg.cholesky(covariance)
    eig, relative_eig = np.linalg.eigvalsh(covariance), eigh(covariance, sigma0, eigvals_only=True)
    assert eig[0] > 0 and relative_eig[0] >= .25-1e-10
    chart = copy.deepcopy(old_chart)
    chart['means'] = [mean.tolist()]
    chart['covariances'] = [covariance.tolist()]
    assert chart['anchors'] == old_chart['anchors'] and chart['angular_length'] == old_chart['angular_length']
    cohort_latent = np.linalg.solve(lower, (x-mean).T).T
    decoded, _ = decode(cohort_latent, chart, fixed)
    p, _, r = arrays(poses)
    pp, _, rr = arrays(decoded)
    position_error = float(np.max(np.abs(p-pp)))
    rotation_error = float(np.max(np.abs(r-rr)))
    assert position_error < 2e-10 and rotation_error < 2e-12
    stamp = datetime.now(timezone.utc).isoformat()
    regions = {}
    for radius in (3., 5.):
        region = {key: copy.deepcopy(old[key]) for key in ('fixed_neighbor', 'capture_center', 'capture_radius',
            'shape_sha256', 'activity', 'depletant_radius', 'physical_metric', 'minimum_original_q')}
        region.update(frozen_at_utc=stamp, physical_config_sha256=sha(out/'physical-config.json'),
            gaussian_chart=chart, mahalanobis_radius=radius,
            definition=f'Capture/hard-valid pose with original q>=5 and cohort-chart latent radius<={radius:g}. Regional target only.',
            selection='Equal-count fit to32 chronological outside-old-R8 retained poses; duplicates retained. This unequilibrated cohort defines integration geometry only. Covariance regularization adds0.25*original-deep covariance.',
            cohort_sha256=sha(args.cohort), old_region_sha256=sha(args.old_region),
            old_chart_anchor_unchanged=True, physical_parameters_unchanged=True,
            preparer_sha256=sha(__file__))
        path = out/f'region-r{radius:g}.json'
        write(path, region)
        regions[str(int(radius))] = dict(path=str(path), sha256=sha(path))

    probes, checks = [], []
    shape = read(archive/'shape.json')
    for index, radius in enumerate((3., 5.)):
        seed = 98601010+1009*index
        u = uniform_ball(np.random.default_rng(seed), radius, args.geometry_probes)
        candidates, logj = decode(u, chart, fixed)
        back = np.linalg.solve(lower, (chart_coordinates(candidates, chart, fixed)-mean).T).T
        back_error = float(np.max(np.abs(back-u)))
        logg = Density(chart).evaluate(relative_poses(candidates, fixed))[0]
        jac_error = float(np.max(np.abs(logg+logj+3*np.log(2*np.pi)+.5*np.sum(u*u, axis=1))))
        assert back_error < 2e-9 and jac_error < 2e-9
        # Independent physical-Jacobian check: numerical derivatives of center
        # and local rotation-vector increments. Normalized Haar at identity is
        # d^3theta/(8*pi^2), rather than Euclidean quaternion measure.
        finite_errors = []
        for point, target in zip(u[:16], logj[:16]):
            jac = np.empty((6, 6))
            h = 1e-4
            for axis in range(6):
                delta = np.zeros(6); delta[axis] = h
                pair, _ = decode(np.array([point+delta, point-delta]), chart, fixed)
                tp, _, rp = arrays(pair)
                jac[:3, axis] = (tp[0]-tp[1])/(2*h)
                jac[3:, axis] = Rotation.from_matrix(rp[0]@rp[1].T).as_rotvec()/(2*h)
            numeric = np.linalg.slogdet(jac)[1]-np.log(8*np.pi**2)
            finite_errors.append(abs(float(numeric-target)))
        assert max(finite_errors) < 2e-6
        old_radius = Density(old_chart).evaluate(relative_poses(candidates, fixed))[1][:, 0]
        q = registration(candidates, cfg['metadata'])
        capture, hard = geometry_probe(candidates, cfg, shape)
        valid = capture & hard & (q >= old['minimum_original_q'])
        inside = old_radius <= 8.
        counts = dict(draws=len(u), capture_valid=int(capture.sum()), hard_and_capture_valid=int((capture&hard).sum()),
            physical_region_valid=int(valid.sum()), inside_old_R8_geometry=int(inside.sum()),
            outside_old_R8_geometry=int((~inside).sum()), valid_inside_old_R8=int((valid&inside).sum()),
            valid_outside_old_R8=int((valid&~inside).sum()))
        probes.append(dict(radius=radius, seed=seed, counts=counts,
            fractions_per_unconditional_probe={key: value/len(u) for key, value in counts.items() if key != 'draws'},
            valid_fraction_outside_old_R8=float(np.mean(~inside[valid])) if valid.any() else None,
            original_q_range=[float(q.min()), float(q.max())], old_latent_radius_range=[float(old_radius.min()), float(old_radius.max())],
            scope='Geometry-only probes with zero physical depletion evaluation. These counts do not estimate equilibrium mass or validate discovery frequency.'))
        checks.append(dict(radius=radius, backmap_max_error=back_error, density_jacobian_max_log_error=jac_error,
            finite_difference_jacobian_max_log_error=max(finite_errors), finite_difference_probes=len(finite_errors), step=h))
    # The fixed proper frame change has unit joint translation/Haar Jacobian.
    ft, _, fr = arrays([fixed])
    block = np.zeros((6, 6)); block[:3, :3] = block[3:, 3:] = fr[0]
    lab = copy.deepcopy(chart)
    lab['coordinate_convention'] = 'laboratory'
    lab['anchors'] = [dict(position=(fr[0]@np.asarray(chart['anchors'][0]['position'])+ft[0]).tolist(),
        rotation=(fr[0]@np.asarray(chart['anchors'][0]['rotation'])).tolist())]
    lab['means'] = [list(block@mean)]
    lab['covariances'] = [(block@covariance@block.T).tolist()]
    frame_error = float(np.max(np.abs(Density(lab).evaluate(poses)[0]-Density(chart).evaluate(relative_poses(poses, fixed))[0])))
    assert frame_error < 2e-9
    report = dict(frozen_at_utc=stamp, cohort_count=len(x), distinct_pose_count=cohort['unique_selected_poses'],
        weighting='Equal32 count weights including repeated poses; no likelihood, SMC, physical, or dwell reweighting.',
        mean=mean.tolist(), raw_covariance=raw.tolist(), additive_floor=(.25*sigma0).tolist(), covariance=covariance.tolist(),
        covariance_eigenvalues=eig.tolist(), covariance_condition_number=float(eig[-1]/eig[0]),
        covariance_eigenvalues_relative_to_original=relative_eig.tolist(),
        cohort_radius_range=[float(np.linalg.norm(cohort_latent, axis=1).min()), float(np.linalg.norm(cohort_latent, axis=1).max())],
        cohort_inside_new_R3=int((np.linalg.norm(cohort_latent, axis=1)<=3).sum()),
        cohort_inside_new_R5=int((np.linalg.norm(cohort_latent, axis=1)<=5).sum()),
        cohort_backmap_position_error=position_error, cohort_backmap_rotation_error=rotation_error,
        body_lab_log_density_error=frame_error, coordinate_checks=checks, geometry_probes=probes,
        regions=regions, source_paths={name: str(path.resolve()) for name, path in inputs.items()},
        input_sha256={name: sha(archive/name) for name in inputs},
        scope='Frozen geometry, not an equilibrium fit. No production draws generated. The old chart/regions and physical target are unchanged.')
    write(out/'report.json', report)
    plan = dict(frozen_at_utc=stamp, status='prepared_not_launched', regions=regions,
        physical_config=str(out/'physical-config.json'), physical_config_sha256=sha(out/'physical-config.json'),
        executable=str(archive/'latent-region-normalizer'), executable_sha256=sha(archive/'latent-region-normalizer'),
        seeds={'3':98621010, '5':98631010}, seed_increment=1009, replicates_per_radius=4,
        samples_per_population=16384, lambda_ratio=64., cloud_replicates=2,
        workers_per_campaign=2, total_workers=4,
        poststratification=dict(old_region=str(archive/'old-region.json'), old_region_sha256=sha(archive/'old-region.json'), old_radius=8.,
            groups=['inside_or_on_old_R8', 'outside_old_R8'],
            estimator='For each group, indicator times V6*J*H_capture*H_hard*I(q>=5)*mean_cloud_W, averaged over EVERY unconditional draw of that campaign. Zero rows remain; never divide by the selected count.',
            uncertainty='Compute each group directly from nonnegative weights; no subtraction of close total estimates. Keep R3 and R5 estimates separate. Report paired-cloud noise, ESS/max weight, population agreement, and Q0 alongside Qz.'),
        suggested_outputs={'3':str(ROOT/'runs/outside-r8-region-r3-16384-l64'), '5':str(ROOT/'runs/outside-r8-region-r5-16384-l64')},
        estimated_budget='131072 total fresh draws. Prior original-R3 uniform integration cost599CPU seconds/65536 draws; provision roughly1200CPU seconds (~5min ideal wall on4workers), with uncertain geometry-dependent gate costs. Stop/report above4x prior cost rather than expand budget.',
        independence='All32 cohort observations and geometry probes are preparation only. Fresh seeds and fixedN confirmation; no refitting or selected-data reuse during production. Regional masses only; no global coverage claim.')
    write(out/'validation-plan.json', plan)
    print(json.dumps(dict(output=str(out), report=report, plan=plan), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
