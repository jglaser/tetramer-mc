#!/usr/bin/env python3
"""Retrospectively mask byte-verified contact-atlas draws around a new peak.

The new center was selected from the broad campaign. These are historical
diagnostics, not held-out validation or conditionally unbiased estimates after
center selection. Original densities and full unconditional N are retained;
complements are measured directly, and no sources or nested balls are pooled.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_native_region_reference import native_q
from analyze_peak_neighborhood import geometry_model_audit, partition_check
from audit_shoulder_mis_independently import chart_values, near, rotation
from compare_intermediate_local_reference import PartitionMoments, finish, read_batches
from prepare_cayley_rms_cover import read, require, sha, write
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
RADII = (.5, 1., 2.)
WINDOW = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
PHYSICAL_KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius',
                 'reservoir_density', 'metadata')
THREEWAY_KEYS = dict(full='full', old_ball='inner_half', new_ball='outer_half',
                    outside_both='complement', union='ball')


def threeway_add(moment, new_radius=None, old_radius=None, physical=None, hard=None, cloud_pair=None):
    """Reuse full-N moments with categorical labels; no invented pose distance.

    The internal half/shell coordinate encodes membership only. Its report is
    renamed to the actual masks, and is never presented as a physical radius.
    """
    if physical is None:
        moment.add(); return
    require(new_radius >= 0 and old_radius >= 0, 'Invalid geometry-chart radius')
    require(not (new_radius <= 2 and old_radius <= 2), 'Claimed disjoint radius-2 balls overlap')
    label = .5 if old_radius <= 2 else 1.5 if new_radius <= 2 else 3.
    moment.add(label, physical, hard, cloud_pair)


def threeway_report(aggregate, populations):
    result = finish(aggregate, populations)
    for source in (result, *result['populations']):
        for kind in ('physical', 'hard'):
            source[kind] = {key: source[kind][previous] for key, previous in THREEWAY_KEYS.items()}
    result['partition_checks'] = {kind: partition_check(values, ('inner_half', 'outer_half', 'complement'))
        for kind, values in aggregate.values.items()}
    result['definition'] = 'Old geometry-chart r<=2; new geometry-chart r<=2; outside both. Every mask retains the full original denominator; union is the same-row sum.'
    return result


def center_separation(cfg, new_pose, old_pose):
    """Euclidean distance in the rigid-member embedding gives a triangle bound."""
    members = np.asarray([pose['position'] for pose in cfg['metadata']['rigid_members']])
    new_members = members@rotation(new_pose).T+np.asarray(new_pose['position'])
    old_members = members@rotation(old_pose).T+np.asarray(old_pose['position'])
    rms = float(np.sqrt(np.mean(np.sum((new_members-old_members)**2, axis=1))))
    return dict(member_RMS_distance_A=rms,
        two_radius_2_balls_disjoint=rms > 4.,
        positive_RMS_separation_lower_bound_A=max(0., rms-4.),
        argument='Each geometry-chart radius-2 ball is contained in its rigid-member RMS ball. The RMS embedding obeys the triangle inequality; center distance > 4 therefore proves the two balls disjoint.',
        qualification='Numerical geometry with a macroscopic separation margin, not interval arithmetic.')


def validate_hashes(mapping, skip_samples=False):
    for name, digest in mapping.items():
        if skip_samples and Path(name).name == 'samples.jsonl':
            continue
        require(sha(Path(name)) == digest, f'Previously audited input changed: {name}')


def analyze(source_audit, preparation, old_preparation, out):
    source_audit, preparation, old_preparation, out = map(Path, (source_audit, preparation, old_preparation, out))
    source_audit, preparation, old_preparation, out = [p.resolve() for p in (source_audit, preparation, old_preparation, out)]
    require(not out.exists(), 'Use a fresh historical-mask output directory')
    started = time.monotonic()
    audit = read(source_audit); require(audit['complete'] and audit['original_q_window'] == WINDOW, 'Wrong or incomplete source audit')
    validate_hashes(audit['input_sha256'])
    for name, digest in audit['archived_sha256'].items():
        require(sha(source_audit.parent/'provenance'/name) == digest, 'Source audit archive changed')
    cfg = read(preparation/'config.json'); protocol = read(preparation/'protocol.json')
    model = read(preparation/'model.json'); selected = read(preparation/'selected-pose.json')
    require(sha(preparation/'config.json') == protocol['config_sha256'], 'New physical config changed')
    require(sha(preparation/'model.json') == protocol['model_sha256'], 'New chart changed')
    require(sha(preparation/'selected-pose.json') == protocol['selected_pose_sha256'], 'Selected new peak changed')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035,
            'Physical baseline changed')
    broad = next(c for c in audit['campaigns'] if c['arm'] == 'broad')
    require(selected == broad['top_poses'][0], 'New center is not the audited broad maximum')
    near(native_q(cfg['metadata'], selected['pose']), selected['q'])
    require(2 <= selected['q'] < 5, 'Selected center outside original q window')
    geometry, _ = geometry_model_audit(model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, selected['pose'])
    old_protocol = read(old_preparation/'protocol.json'); old_model = read(old_preparation/'model.json')
    old_selected = read(old_preparation/'selected-pose.json'); old_cfg = read(old_preparation/'config.json')
    require(sha(old_preparation/'model.json') == old_protocol['model_sha256'], 'Prior geometry chart changed')
    require(sha(old_preparation/'selected-pose.json') == old_protocol['selected_pose_sha256'], 'Prior selected pose changed')
    require(all(cfg[key] == old_cfg[key] for key in PHYSICAL_KEYS), 'New/prior chart physical targets differ')
    old_geometry, _ = geometry_model_audit(old_model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, old_selected['pose'])
    require(model['shape_sha256'] == old_model['shape_sha256'], 'Geometry chart shape differs')
    separation = center_separation(cfg, selected['pose'], old_selected['pose'])
    require(separation['two_radius_2_balls_disjoint'], 'Three-way partition requires disjoint enclosing RMS balls')
    sources = {'source-audit.json': source_audit, 'new-model.json': preparation/'model.json',
        'new-protocol.json': preparation/'protocol.json', 'new-selected-pose.json': preparation/'selected-pose.json',
        'new-config.json': preparation/'config.json', 'old-model.json': old_preparation/'model.json',
        'old-protocol.json': old_preparation/'protocol.json', 'old-selected-pose.json': old_preparation/'selected-pose.json'}
    input_hashes = {str(path): sha(path) for path in sources.values()}
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in {**local_dependencies([Path(__file__)]), **sources}.items():
        (archive/name).write_bytes(path.read_bytes())
    campaigns = []
    for original in audit['campaigns']:
        root = Path(original['root']); prior = original['original_full_window_audit']
        master = read(root/'manifest.json'); current_cfg = read(root/'provenance/config.json')
        require(sha(root/'manifest.json') == prior['manifest_sha256'], 'Original campaign manifest changed')
        require(master['q_window'] == WINDOW and prior['original_integration_window'] == WINDOW, 'Original window changed')
        require(all(current_cfg[key] == cfg[key] for key in PHYSICAL_KEYS), 'Original physical target differs')
        require(prior['shape_sha256'] == model['shape_sha256'], 'Original shape differs')
        for name, digest in master['archive_sha256'].items():
            require(sha(root/'provenance'/name) == digest, 'Original frozen campaign input changed')
        validate_hashes(original['input_sha256'], skip_samples=True)
        aggregate = {r: PartitionMoments(r) for r in RADII}; populations = {r: [] for r in RADII}
        threeway = PartitionMoments(2.); threeway_populations = []
        matched_peak = None; sample_hashes = {}
        for job in master['jobs']:
            path = Path(job['output']); local = {r: PartitionMoments(r) for r in RADII}
            local_threeway = PartitionMoments(2.)
            digest = hashlib.sha256(); count = 0
            for lines, rows in read_batches(path/'samples.jsonl'):
                for line in lines: digest.update(line)
                positions = np.asarray([row['pose']['position'] for row in rows])
                quats = np.asarray([row['pose']['orientation'] for row in rows])
                rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
                _, radii, _ = chart_values(model, positions, rotations, cfg['fixed_poses'][0])
                _, old_radii, _ = chart_values(old_model, positions, rotations, cfg['fixed_poses'][0])
                for i, row in enumerate(rows):
                    require(row['draw'] == count, 'Original draw order changed'); count += 1
                    if 'zero' in row:
                        for moment in local.values(): moment.add()
                        threeway_add(local_threeway)
                        continue
                    require(2 <= row['q'] < 5 and math.dist(row['pose']['position'], cfg['capture_center']) <= 18.,
                            'Audited nonzero row violates original domain')
                    physical, hard = row['log_importance_weight'], row['log_hard_weight']
                    pair = [value+hard for value in row['cloud_log_weights']]
                    near(float(np.logaddexp(*pair))-math.log(2), physical)
                    for moment in local.values(): moment.add(float(radii[i, 0]), physical, hard, pair)
                    threeway_add(local_threeway, float(radii[i, 0]), float(old_radii[i, 0]), physical, hard, pair)
                    if original['arm'] == 'broad' and job['seed'] == selected['seed'] and row['draw'] == selected['draw']:
                        require(row['pose'] == selected['pose'], 'Selected draw pose changed')
                        near(physical, selected['original_log_importance_weight'])
                        matched_peak = dict(radius_A=float(radii[i, 0]),
                            original_log_importance_weight=physical,
                            log_contribution_to_full_N_mean=physical-math.log(master['total_unconditional_draws']),
                            original_population=path.name, original_seed=job['seed'], original_draw=row['draw'])
            manifest = read(path/'manifest.json')
            require(count == manifest['samples'], 'Original unconditional count changed')
            value = digest.hexdigest(); sample_path = str(path/'samples.jsonl')
            require(value == prior['sample_sha256'][sample_path] == original['input_sha256'][sample_path], 'Previously audited sample bytes changed')
            sample_hashes[sample_path] = value
            for radius in RADII:
                populations[radius].append(dict(id=path.name, seed=job['seed'], samples=count, **local[radius].report()))
                aggregate[radius].merge(local[radius])
            threeway_populations.append(dict(id=path.name, seed=job['seed'], samples=count, **local_threeway.report()))
            threeway.merge(local_threeway)
        regions = []
        for radius in RADII:
            report = finish(aggregate[radius], populations[radius])
            for kind in ('physical', 'hard'):
                near(report[kind]['full']['logQ'], original[kind]['full']['logQ'])
                near(report[kind]['full']['row_RSE'], original[kind]['full']['row_RSE'])
            report['partition_checks'] = {kind: partition_check(values, ('ball', 'complement'))
                for kind, values in aggregate[radius].values.items()}
            if matched_peak:
                report['selected_point_fraction_of_ball_weight'] = math.exp(matched_peak['log_contribution_to_full_N_mean']-report['physical']['ball']['logQ'])
            regions.append(dict(radius_A=radius, **report))
        if original['arm'] == 'broad': require(matched_peak is not None, 'Selected historical maximum missing')
        threeway_result = threeway_report(threeway, threeway_populations)
        near(threeway_result['physical']['old_ball']['logQ'], original['physical']['peak_r_le_2']['logQ'])
        record = dict(arm=original['arm'], root=str(root), regions=regions, selected_point=matched_peak,
            disjoint_threeway=threeway_result,
            samples_sha256=sample_hashes, CPU_seconds=original['CPU_seconds'],
            previous_peak_balls={key: original['physical'][key] for key in ('peak_r_le_0p5', 'peak_r_le_1', 'peak_r_le_2')})
        campaigns.append(record)
        print(f'{original["arm"]}: local ball logQ {[r["physical"]["ball"]["logQ"] for r in regions]}', flush=True)
    validate_hashes(input_hashes)
    result = dict(complete=True, campaigns=campaigns, radii_A=RADII, original_q_window=WINDOW,
        new_geometry_reconstruction=geometry, old_geometry_reconstruction=old_geometry,
        center_separation=separation, input_sha256=input_hashes,
        archived_sha256={p.name: sha(p) for p in archive.iterdir()}, elapsed_seconds=time.monotonic()-started,
        estimator='Original frozen density and full unconditional N in every mask, with invalid/off-mask rows retained as zeros. Complements are direct positive masked means, not subtractions.',
        selection='New center selected from the broad source maximum: retrospective, data-dependent region diagnostics, not held-out validation or claims of conditional unbiasedness after selection. Fresh local references provide independent measurements of the frozen regions.',
        coverage='Full-window source support is retained and each ball/complement partition reconstructs its source. Nested balls and historical/new sources must not be summed or pooled. Zero observations and empirical errors do not bound unseen mass.',
        scope='Fixed AB contact-domain diagnostics only; no assembly or equilibrium-mixing conclusion.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--old-preparation', type=Path, default=ROOT/'runs/ab-intermediate-new-peak-geometry-20260920')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); analyze(args.source_audit, args.preparation, args.old_preparation, args.out)
