#!/usr/bin/env python3
"""Freeze two finite conditional pose regions; no bath draws or MC execution."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation

from prepare_native_confirmation_atlas import AtomUnionAudit, native_q
from prepare_native_tail_regions import decode_latent, encode_matrix, validate_physics
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT/'runs/mobile-posterior-pilot-12x2000-20260921'
ASSESSMENT = ROOT/'runs/mobile-posterior-pilot-assessment-v2-20260921'
JOB = 'mobile3-dispersed-r01-c09'
NATIVE_MODEL = ROOT/'runs/native-ab-covariance-guide-20260920/model.json'
NATIVE_SHA = '86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667'
OLD_CONFIG = ROOT/'runs/ab-shoulder-contact-atlas-preparation-20260921/config.json'
BINARY = ROOT/'target/mobile-posterior-review/release/latent-region-normalizer'
BINARY_SHA = 'db4dfbbf23e594ccc49e829cd47436e8e39e33dd9d13e8f0eac22e103d2b45df'
BUNDLE = ROOT/'target/mobile-posterior-review/release/build/tetramer-mc-356a8e0b913327fe/out/source-bundle.json'
CAPTURE, SEED = 170., 116601010


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose(position, matrix):
    return dict(position=np.asarray(position).tolist(),
                orientation=Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]].tolist())


def native_frame(old_anchor, new_anchor):
    old_t, _, old_r = arrays([old_anchor])
    new_t, _, new_r = arrays([new_anchor])
    rotation = new_r[0]@old_r[0].T
    return pose(new_t[0]-rotation@old_t[0], rotation)


def competitor_model(competitor, fixed, template, translation_std=.25, angular_std_degrees=.25):
    require(translation_std > 0 and 0 < angular_std_degrees < 180, 'Invalid geometric chart scales')
    relative = relative_poses([competitor], fixed)
    t, _, r = arrays(relative)
    angular = template['angular_length']*math.tan(math.radians(angular_std_degrees)/2)
    return dict(schema=template['schema'], coordinate_convention='anchor-body-relative',
                shape_sha256=template['shape_sha256'], angular_length=template['angular_length'],
                anchors=[dict(position=t[0].tolist(), rotation=r[0].tolist())], means=[[0.]*6],
                covariances=[np.diag([translation_std**2]*3+[angular**2]*3).tolist()], weights=[1.])


def capture_wall_certificate(model, fixed, radius, capture, wall_radius, shape_bound):
    require(0 < radius and 0 < capture < wall_radius-shape_bound, 'Capture sphere does not certify the physical wall')
    centers, _ = decode_latent(np.zeros((1, 6)), model, fixed)
    lower = np.linalg.cholesky(model['covariances'][0])
    displacement = radius*float(np.linalg.norm(lower[:3], ord=2))
    center_norm = float(np.linalg.norm(centers[0]['position']))
    require(center_norm+displacement < capture, 'Finite chart region is not entirely inside capture')
    return dict(chart_center_norm_A=center_norm, maximum_translation_from_chart_center_A=displacement,
                entire_region_center_norm_upper_bound_A=center_norm+displacement,
                capture_radius_A=capture, shape_bound_A=shape_bound, physical_sphere_radius_A=wall_radius,
                all_capture_poses_atom_radius_upper_bound_A=capture+shape_bound,
                guaranteed_atomic_wall_clearance_A=wall_radius-capture-shape_bound,
                argument='For every chart draw ||t|| <= ||t_center|| + R ||L_translation||_2 < capture. For every capture-valid pose and every orientation, ||world_atom||+atom_radius <= capture+shape_bound < original sphere radius.')


def jacobian_check(model, fixed, latent):
    latent = np.asarray(latent, dtype=float)
    base, logj = decode_latent(latent[None], model, fixed)
    base_r = Rotation.from_quat(np.array(base[0]['orientation'])[[1, 2, 3, 0]])
    differential, eps = np.zeros((6, 6)), 2e-4
    for axis in range(6):
        shift = np.eye(6)[axis]*eps
        neighbors, _ = decode_latent(np.stack([latent+shift, latent-shift]), model, fixed)
        differential[:3, axis] = (np.array(neighbors[0]['position'])-neighbors[1]['position'])/(2*eps)
        rotations = Rotation.from_quat(np.array([p['orientation'] for p in neighbors])[:, [1, 2, 3, 0]])
        angular = (rotations*base_r.inv()).as_rotvec()
        differential[3:, axis] = (angular[0]-angular[1])/(2*eps)
    independent = abs(np.linalg.det(differential))/(8*math.pi**2)
    relative_error = abs(independent/math.exp(float(logj[0]))-1)
    require(relative_error < 2e-7, 'Independent physical Haar differential failed')
    return relative_error


def geometry_record(value, cfg, audit, shape):
    t, _, r = arrays([value])
    atoms = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    wall = cfg['metadata']['physical_sphere_radius_A']-np.max(np.linalg.norm(atoms@r[0].T+t[0], axis=1)+radii)
    gaps = audit.gaps(value)
    return dict(pose=value, q=float(native_q([value], cfg['metadata'])[0]),
                minimum_core_gap_by_fixed_body_A=gaps, hard_valid=all(g >= 0 for g in gaps),
                minimum_atomic_wall_clearance_A=float(wall), wall_valid=bool(wall >= 0))


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh immutable preparation directory')
    manifest, completed = read(CAMPAIGN/'manifest.json'), read(ASSESSMENT/'analysis.json')
    require(completed['complete'] and completed['manifest_sha256'] == sha(CAMPAIGN/'manifest.json'), 'Source campaign/audit changed')
    job = next(j for j in manifest['jobs'] if j['id'] == JOB)
    directory = Path(job['directory'])
    audited = read(ASSESSMENT/'runs'/JOB/'analysis.json')
    require(audited == next(r for r in completed['runs'] if r['id'] == JOB) and audited['passed'], 'Source audit identity differs')
    for name, digest in audited['source_sha256'].items():
        require(sha(directory/name) == digest, 'Source run file changed: '+name)
    frames = [json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    final, mobile_cfg = frames[-1], read(directory/'config.json')
    require(final['sweep'] == 2000 and final['poses'] == read(directory/'checkpoint.json')['poses'], 'Wrong final scaffold')
    require(len(final['poses']) == 3 and mobile_cfg['boundary']['kind'] == 'spherical', 'Wrong mobile geometry')
    require(sha(NATIVE_MODEL) == NATIVE_SHA and sha(BINARY) == BINARY_SHA, 'Legacy native chart or reference binary changed')
    require(BUNDLE.read_bytes() in BINARY.read_bytes(), 'Reference binary does not embed the supplied source bundle')
    for name, entry in read(BUNDLE)['files'].items():
        require(sha(ROOT/name) == entry['sha256'] and (ROOT/name).read_text() == entry['text'], 'Reviewed source differs from binary bundle: '+name)
    original, old = read(NATIVE_MODEL), read(OLD_CONFIG)
    shape_path = directory/'provenance/shape.json'
    shape = read(shape_path)
    require(original['shape_sha256'] == manifest['shape_sha256'] == sha(shape_path), 'Shape changed')
    require(original['weights'] == [1.] and old['metadata']['native_poses'] == [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])], 'Expected the original single native chart')
    moving, fixed = final['poses'][0], [final['poses'][2], final['poses'][1]]
    native = native_frame(old['fixed_poses'][0], fixed[0])
    cfg = copy.deepcopy(old)
    cfg.update(shape=str(out/'provenance/shape.json'), fixed_poses=copy.deepcopy(fixed),
               initial_pose=copy.deepcopy(moving), capture_center=[0., 0., 0.], capture_radius=CAPTURE, seed=SEED)
    cfg.pop('target_region', None)
    cfg['metadata'] = dict(native_poses=[native], rigid_members=shape['rigid_members'], member_error_scale=2.,
        angle_error_scale_deg=15., physical_sphere_radius_A=mobile_cfg['boundary']['radius'],
        physical_wall_scope='The capture sphere is wholly inside the original atomic wall for every orientation; the finite regions are wholly inside capture. Bath remains unbounded/permeable.',
        native_reference='Original AB identity transformed by exact body2=A frame, without repairing body1=B.',
        initial_mobile_job=JOB, source_sweep=2000, fixed_body_ids=[2, 1], moving_body_id=0)
    require(cfg['depletant_radius'] == mobile_cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == mobile_cfg['reservoir_density'] == .035, 'Bath changed')
    cfg['endpoint_gate'] = copy.deepcopy(mobile_cfg['endpoint_gate'])
    cfg['poisson_lambda_ratio'] = 64.
    competitor = competitor_model(moving, fixed[0], original)
    shape_bound = float(max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms']))
    require(all(np.linalg.norm(p['position'])+shape_bound < mobile_cfg['boundary']['radius'] for p in fixed), 'Fixed body wall proof failed')
    audit = AtomUnionAudit(shape, fixed)
    centers = {name: geometry_record(value, cfg, audit, shape) for name, value in [('native_reference', native), ('competitor_reference', moving)]}
    require(all(value['hard_valid'] and value['wall_valid'] for value in centers.values()), 'Selected exact reference pose is physically invalid; no repair permitted')
    fixed_audit = AtomUnionAudit(shape, [fixed[0]])
    require(min(fixed_audit.gaps(fixed[1])) >= 0, 'Fixed scaffold overlaps')
    regions, checks, probe_rows = {}, {}, []
    specifications = [('native-r4', original, 0., 4., 0., 1.), ('competitor-r3', competitor, 0., 3., 1., None),
                      ('competitor-shell-3-5', competitor, 3., 5., 1., None), ('competitor-r5', competitor, 0., 5., 1., None)]
    for name, model, lo, hi, qlo, qhi in specifications:
        region = dict(fixed_neighbor=copy.deepcopy(fixed[0]), physical_fixed_neighbors=copy.deepcopy(fixed),
            capture_center=cfg['capture_center'], capture_radius=CAPTURE, shape_sha256=sha(shape_path),
            activity=.035, depletant_radius=1.5, physical_metric=copy.deepcopy(cfg['metadata']), gaussian_chart=copy.deepcopy(model),
            minimum_mahalanobis_radius=lo, mahalanobis_radius=hi, minimum_original_q=qlo,
            minimum_original_q_inclusive=qlo == 0., maximum_original_q_inclusive=True,
            definition=f'Finite {name}: frozen chart {lo}<rho<={hi} (center included when inner=0), original native q'+('<=1' if qhi else '>1')+', exact final-scaffold hard geometry, and technical capture170. No native residue-patch or contact mask is implicit.')
        if qhi is not None:
            region['maximum_original_q'] = qhi
        validate_physics(region, cfg, sha(shape_path))
        certificate = capture_wall_certificate(model, fixed[0], hi, CAPTURE, mobile_cfg['boundary']['radius'], shape_bound)
        # Fixed deterministic map/geometry probes are never integration samples.
        latent = np.concatenate([np.zeros((1, 6)), hi*np.eye(6), -hi*np.eye(6), np.full((1, 6), hi/(2*math.sqrt(6)))])
        probes, logj = decode_latent(latent, model, fixed[0])
        back = encode_matrix(probes, model, fixed[0])
        error = float(np.max(np.abs(back-latent)))
        require(error < 1e-8, 'Pose-map inverse failed')
        density = Density(model).evaluate(relative_poses(probes, fixed[0]))[0]
        identity_error = float(np.max(np.abs(logj+density+3*math.log(2*math.pi)+.5*np.sum(latent**2, axis=1))))
        require(identity_error < 2e-8, 'Gaussian/Haar Jacobian identity failed')
        differential_error = max(jacobian_check(model, fixed[0], u) for u in latent[[0, -1]])
        for i, value in enumerate(probes):
            probe_rows.append(dict(region=name, probe=i, latent=latent[i].tolist(), **geometry_record(value, cfg, audit, shape)))
        checks[name] = dict(capture_and_wall=certificate, deterministic_probes=len(probes), maximum_backmap_error=error,
            maximum_log_density_jacobian_identity_error=identity_error, maximum_independent_differential_relative_error=differential_error)
        regions[name] = region
    archive = out/'provenance'
    archive.mkdir(parents=True)
    sources = {'shape.json': shape_path, 'native-original-model.json': NATIVE_MODEL, 'original-ab-config.json': OLD_CONFIG,
        'source-campaign-manifest.json': CAMPAIGN/'manifest.json', 'source-effective-config.json': directory/'config.json',
        'source-checkpoint.json': directory/'checkpoint.json', 'source-trajectory.jsonl': directory/'trajectory.jsonl',
        'source-run-audit.json': ASSESSMENT/'runs'/JOB/'analysis.json', 'reference-binary-source-bundle.json': BUNDLE,
        'independent-geometry-analysis.json': ROOT/'runs/mobile-competing-geometry-20260921/analysis.json',
        'independent-fixed-snapshot.json': ROOT/'runs/mobile-competing-geometry-20260921/fixed-snapshot.json'}
    sources.update(local_dependencies([Path(__file__)]))
    for name, source in sources.items():
        shutil.copy2(source, archive/name)
    shutil.copy2(NATIVE_MODEL, out/'model-native-original.json')
    write(out/'model-competitor-geometric.json', competitor)
    write(out/'config.json', cfg)
    for name, value in regions.items():
        write(out/f'region-{name}.json', value)
    write(out/'geometry-probes.json', probe_rows)
    write(out/'preflight.json', dict(passed=True, centers=centers, regions=checks,
        fixed_scaffold_body_ids=[2, 1], no_repairs=True, Poisson_clouds=0, physical_MC_updates=0,
        note='Deterministic probes check geometry/maps, not region volume, physical weight or statistical coverage.'))
    jobs = []
    for name in ('native-r4', 'competitor-r3'):
        for replicate in range(4):
            jobs.append(dict(id=f'{name}-r{replicate:02}', region=name, replicate=replicate, seed=SEED+1009*len(jobs),
                samples=8192, config=str(out/'config.json'), config_sha256=sha(out/'config.json'),
                region_file=str(out/f'region-{name}.json'), region_sha256=sha(out/f'region-{name}.json')))
    plan = dict(schema='mobile-competing-finite-region-preparation-v1', production_launched=False,
        source_campaign=str(CAMPAIGN), source_assessment=str(ASSESSMENT), source_job=JOB, source_sweep=2000,
        source_assessment_sha256=sha(ASSESSMENT/'analysis.json'), physical_scaffold=cfg['fixed_poses'], fixed_body_ids=[2, 1],
        moving_body_id=0, native_original_model_sha256=NATIVE_SHA, shape_sha256=sha(shape_path),
        reference_binary=str(BINARY), reference_binary_sha256=BINARY_SHA, reference_source_bundle_sha256=sha(BUNDLE),
        executable_action='Recorded existing compatible executable only; parent must freeze/review the campaign before launch.',
        samples_per_population=8192, populations_per_region=4, total_unconditional_draws=65536,
        workers=8, cloud_replicates=2, lambda_ratio=64., jobs=jobs,
        future_regions=['competitor-shell-3-5', 'competitor-r5'], future_regions_authorized_for_sampling=False,
        command_template=['FROZEN_LATENT_BINARY', '--config', 'CONFIG', '--region', 'REGION_FILE', '--out', 'FRESH_POPULATION_OUT', '--samples', '8192', '--seed', 'SEED', '--cloud-replicates', '2', '--lambda-ratio', '64'],
        competitor_chart_geometry=dict(translation_std_A=.25, angular_coordinate_scale_degrees=.25,
            angular_coordinate_std_A=original['angular_length']*math.tan(math.radians(.25)/2),
            angular_convention='Each Cayley coordinate SD = ell*tan(0.25 degrees/2); exact maximum rotation at radius R is 2*atan(R*tan(0.25 degrees/2)).',
            fitted=False, region_selection='Final observed pose supplies the center only. Isotropic scales fixed by explicit geometry, without bath weights or covariance fitting.'),
        estimands='For each finite region: Qz=integral H_hard exp[z |E(moving) intersect union E(fixed)|] d3t dHaar; Q0=integral H_hard d3t dHaar. Common fixed-body and moving self-volume constants cancel in regional ratios.',
        analysis='Keep every unconditional draw including invalid zeros. Report Qz and Q0 separately, Qz/Q0 with correlated paired uncertainty, independent-population dispersion, row SE, ESS, maximum share, and two-cloud noise. Retain native and competitor populations separately. Compare finite Qz ratio to accessible-volume Q0 ratio; atlas density is absent from direct integration weights.',
        scope='Conditional finite-region comparison on the exact observed slightly deformed scaffold. Native original-chart R4 is only a subset of q<=1; do not transfer old scaffold coverage fractions. Competitor R3 excludes its frozen outer shell and all other poses. No global equilibrium population, scaffold assembly cost, equilibrium efficiency, physical residence rate or full competitor/native free-energy claim. q is one rigid-reference geometric metric, distinct from hysteretic native residue-patch graphs.',
        input_sha256={name: sha(archive/name) for name in sources})
    write(out/'plan.json', plan)
    write(out/'freeze.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(out=str(out), plan_sha256=sha(out/'plan.json'), jobs=len(jobs), total_draws=65536, preflight_passed=True)))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/mobile-competing-reference-preparation-20260921')
    prepare(parser.parse_args().out)
