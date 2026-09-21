#!/usr/bin/env python3
"""Freeze an observed-pose member-RMS cover without physical resampling.

The historical peak is first. Farthest-first additions use only rigid-member
geometry, not importance weights or distorted fitted-chart distances.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_far_peak_reference import PHYSICAL, WINDOW
from analyze_native_region_reference import native_q
from analyze_peak_neighborhood import geometry_model_audit, geometry_rows
from audit_shoulder_mis_independently import near
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_far_contact_candidates import member_distances
from prepare_native_confirmation_atlas import AtomUnionAudit, to_world
from prepare_peak_neighborhood import geometry_model
from prepare_smc_normalizer_atlas import arrays, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'runs/ab-far-contact-candidates-20260920/analysis.json'
LOCAL = ROOT/'runs/ab-far-peak-reference-assessment-20260920/analysis.json'
FLAT = ROOT/'runs/ab-far-capture-control-20260920/AB-analysis.json'
PREPARATION = ROOT/'runs/ab-far-peak-reference-preparation-20260920'
COVER_RADIUS = .5


def farthest_first(distances, first, radius):
    """Closed radius cover; first index breaks exact distance ties."""
    distances = np.asarray(distances, float)
    require(distances.ndim == 2 and distances.shape[0] == distances.shape[1] and len(distances) > 0, 'Need a nonempty square distance matrix')
    require(np.isfinite(distances).all() and np.min(distances) >= 0, 'Invalid geometric distances')
    require(np.max(abs(distances-distances.T)) < 1e-12 and np.array_equal(np.diag(distances), np.zeros(len(distances))), 'Distance matrix is not symmetric with zero diagonal')
    require(type(first) is int and 0 <= first < len(distances) and math.isfinite(radius) and radius >= 0, 'Invalid fixed start or cover radius')
    centers = [first]; nearest = distances[:, first].copy(); stages = []
    while True:
        farthest = int(np.argmax(nearest)); maximum = float(nearest[farthest])
        stages.append(dict(center_count=len(centers), maximum_nearest_member_RMS_A=maximum, farthest_candidate_index=farthest))
        if maximum <= radius: break
        require(farthest not in centers, 'Farthest-first failed to make progress')
        centers.append(farthest); nearest = np.minimum(nearest, distances[:, farthest])
    chosen = np.argmin(distances[:, centers], axis=1)
    return dict(center_candidate_indices=centers, nearest_center_indices=chosen.tolist(),
        nearest_member_RMS_A=distances[np.arange(len(distances)), np.asarray(centers)[chosen]].tolist(), stages=stages)


def checked_snapshot(path, source_hashes):
    value = read(path); require(value['complete'], f'Incomplete source: {path}')
    source_hashes[str(path)] = sha(path)
    return value


def build_records(history, local, flat, hashes):
    records = []
    def add(kind, source_path, source_index, row, sample_path, sample_sha, manifest_path, config_path, originals=None):
        manifest = read(manifest_path); samples = manifest['samples']
        record = dict(source_kind=kind, source_analysis_path=str(source_path), source_analysis_sha256=sha(source_path),
            source_record_index=source_index, pose=row['pose'], q=row['q'], draw=row['draw'],
            seed=manifest['seed'], population=row.get('population', row.get('replicate')),
            unconditional_population_draws=samples, original_log_importance_weight=row.get('original_log_importance_weight', row.get('log_importance_weight')),
            source_samples_path=str(sample_path), source_samples_sha256=sample_sha,
            source_manifest_path=str(manifest_path), source_manifest_sha256=sha(manifest_path),
            source_config_path=str(config_path), source_config_sha256=sha(config_path),
            source_record=row, original_row=originals)
        require(record['population'] is not None and record['original_log_importance_weight'] is not None, 'Missing source identity or original weight')
        records.append(record)
    for index, row in enumerate(history['candidates']):
        for kind in ('samples', 'manifest', 'config'):
            path = Path(row[f'source_{kind}_path']); require(sha(path) == row[f'source_{kind}_sha256'], 'Historical selected source changed')
        add('historical_width', HISTORY, index, row, Path(row['source_samples_path']), row['source_samples_sha256'],
            Path(row['source_manifest_path']), Path(row['source_config_path']), row['original_row'])
        require(records[-1]['seed'] == row['seed'], 'Historical selected seed changed')
        records[-1]['width'] = row['width']
    for campaign_index, campaign in enumerate(local['campaigns']):
        root = Path(campaign['root']); require(sha(root/'manifest.json') == campaign['manifest_sha256'], 'Local master manifest changed')
        hashes[str(root/'manifest.json')] = sha(root/'manifest.json')
        for index, row in enumerate(campaign['top8_atomic_checks']):
            sample = Path(row['source_samples_path']); require(row['source_samples_sha256'] == campaign['sample_sha256'][str(sample)], 'Local selected source hash differs')
            add('local_reference', LOCAL, [campaign_index, index], row, sample, row['source_samples_sha256'],
                sample.parent/'manifest.json', root/'provenance/config.json', row['original_row'])
            require(records[-1]['seed'] == row['seed'], 'Local selected seed changed')
            records[-1]['local_radius_A'] = campaign['radius_A']
    flat_audit_path = Path(flat['full_density_audit_path']); require(sha(flat_audit_path) == flat['full_density_audit_sha256'], 'Flat density audit changed')
    hashes[str(flat_audit_path)] = sha(flat_audit_path); flat_root = flat_audit_path.parent
    for index, row in enumerate(flat['top_poses']):
        sample = flat_root/'runs'/row['replicate']/'samples.jsonl'
        add('flat_capture', FLAT, index, row, sample, flat['sample_sha256'][str(sample)],
            sample.parent/'manifest.json', flat_root/'provenance/config.json')
    return records


def verify_original_rows(records, cfg, shape_sha, hashes):
    """Read every selected source once; preserve exact rows and original masks."""
    by_source = defaultdict(list)
    for i, record in enumerate(records): by_source[record['source_samples_path']].append(i)
    checked_rows = 0
    for sample_name, indices in by_source.items():
        first = records[indices[0]]; sample = Path(sample_name); manifest_path = Path(first['source_manifest_path'])
        manifest = read(manifest_path); config_path = Path(first['source_config_path']); config = read(config_path)
        require(all(config[k] == cfg[k] for k in PHYSICAL), 'Selected row has different AB physical geometry/bath/metric')
        require(sha(config_path) == manifest['config_sha256'], 'Selected runtime config differs from source config')
        require(manifest['shape_sha256'] == shape_sha == sha(Path(config['shape'])), 'Selected source shape differs')
        require(manifest['activity'] == cfg['reservoir_density'] and manifest['cloud_replicates'] == 2 and manifest['lambda'] == 64*manifest['activity'], 'Selected source cloud law differs')
        kind = first['source_kind']; summary_path = sample.parent/'summary.json'; summary = read(summary_path)
        require(summary['complete'], 'Selected population incomplete')
        if kind != 'flat_capture': require(summary['manifest'] == manifest, 'Selected summary and runtime manifest differ')
        if kind == 'historical_width': require(manifest['proposal_anchor_index'] == 0, 'Historical proposal frame differs')
        if kind == 'local_reference':
            require(manifest['chart_anchor'] == cfg['fixed_poses'][0] and manifest['physical_fixed_neighbors'] == cfg['fixed_poses'], 'Local selected chart or hard neighbors differ')
        if kind == 'flat_capture': require(manifest['guide']['anchor_index'] == 0 and manifest['q_window'] == WINDOW, 'Flat frame or far mask differs')
        wanted = defaultdict(list)
        for index in indices:
            row = records[index]; require(row['source_samples_sha256'] == first['source_samples_sha256'] and row['source_manifest_sha256'] == sha(manifest_path), 'Source hash inconsistency')
            require(row['seed'] == manifest['seed'] and row['unconditional_population_draws'] == manifest['samples'], 'Original seed or budget mismatch')
            wanted[row['draw']].append(index)
        found = set(); count = 0; digest = hashlib.sha256()
        with sample.open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line); require(row['draw'] == count, 'Original source row ordering changed'); count += 1
                if row['draw'] not in wanted: continue
                for index in wanted[row['draw']]:
                    record = records[index]
                    require(row['pose'] == record['pose'] and row['q'] == record['q'] and row['log_importance_weight'] == record['original_log_importance_weight'], 'Original selected pose, q or weight differs')
                    require(5 <= row['q'] < 37 and math.isfinite(row['log_importance_weight']), 'Selected row is not a finite far contribution')
                    if kind == 'flat_capture':
                        require(row.get('zero') is None, 'Selected flat row was rejected')
                        for key, value in row.items(): require(record['source_record'].get(key) == value, 'Flat original field changed')
                        record['original_row'] = row
                    else:
                        require(row == record['original_row'] and row['capture_valid'] and row['hard_valid'], 'Selected original mask changed')
                        if kind == 'local_reference': require(row['region_valid'], 'Local selected row outside declared q window')
                    record['original_masks'] = dict(q_window=WINDOW, original_q=row['q'], capture_valid=True, hard_valid=True,
                        region_valid=True if kind != 'historical_width' else None,
                        original_region=row.get('region'), original_zero=row.get('zero'))
                    near(native_q(cfg['metadata'], row['pose']), row['q'])
                    found.add(index)
        require(count == manifest['samples'] and digest.hexdigest() == first['source_samples_sha256'], 'Original source budget/hash changed')
        if 'samples_sha256' in summary: require(digest.hexdigest() == summary['samples_sha256'], 'Original summary sample hash differs')
        require(found == set(indices), 'Selected source draws missing')
        checked_rows += count
        for path in (sample, manifest_path, config_path, summary_path): hashes[str(path)] = sha(path)
    return dict(unique_source_populations=len(by_source), original_source_rows_hash_checked=checked_rows,
        selected_rows_checked=len(records), no_new_physical_draws=True)


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Use a fresh output directory')
    hashes = {}; history = checked_snapshot(HISTORY, hashes); local = checked_snapshot(LOCAL, hashes); flat = checked_snapshot(FLAT, hashes)
    require(len(history['candidates']) == 96 and sum(len(c['top8_atomic_checks']) for c in local['campaigns']) == 24 and len(flat['top_poses']) == 8, 'Frozen candidate source counts differ')
    require(local['original_q_window'] == flat['q_window'] == WINDOW, 'Frozen far masks differ')
    require(local['selection_audit']['candidate_sha256'] == sha(HISTORY), 'Local peak selected from different historical candidates')
    require(history['protocol_sha256'] == sha(HISTORY.parent/'protocol.json'), 'Historical selection protocol changed')
    for result, path in ((history, HISTORY), (local, LOCAL)):
        for name, digest in result['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Source analysis archive changed')
    for path, digest in local['input_sha256'].items(): require(sha(Path(path)) == digest, 'Local source input changed')
    cfg = read(PREPARATION/'config.json'); peak_model = read(PREPARATION/'model.json'); shape = Path(cfg['shape']); shape_sha = sha(shape)
    require(shape_sha == peak_model['shape_sha256'], 'Physical protein shape changed')
    require(len(cfg['fixed_poses']) == 2 and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and cfg['capture_radius'] == 18., 'Frozen AB target differs')
    first = history['selected_peak']; require(type(first) is int and read(PREPARATION/'selected-pose.json')['pose'] == history['candidates'][first]['pose'], 'Frozen first peak changed')
    records = build_records(history, local, flat, hashes)
    out.mkdir(parents=True)
    protocol = dict(schema='observed-far-member-RMS-centers-v1', frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_analyses={str(path): sha(path) for path in (HISTORY, LOCAL, FLAT)}, source_counts=dict(historical_width=96, local_reference=24, flat_capture=8),
        candidate_order='96 historical rows in archived order, then local campaigns and top8 lists in archived order, then eight flat top_poses in archived order',
        first_candidate_index=first, first_selection='The previously frozen historical width-four maximum, unchanged',
        cover_member_RMS_A=COVER_RADIUS, rule='Farthest-first by minimum rigid-member RMS distance to selected centers. Lowest candidate index breaks exact ties. Stop only once every recorded candidate is within the closed0.5Å radius.',
        coordinate_convention='anchor-body-relative', proposal_anchor_index=0, fixed_neighbor=cfg['fixed_poses'][0], physical={k: cfg[k] for k in PHYSICAL},
        angular_length=peak_model['angular_length'], shape_sha256=shape_sha,
        scope='Finite observed candidate cover only. No physical draws, refitting or equilibrium mixture weights. No full far/contact-space coverage claim.')
    write(out/'protocol.json', protocol)
    original_audit = verify_original_rows(records, cfg, shape_sha, hashes)
    members = np.asarray([p['position'] for p in cfg['metadata']['rigid_members']]); poses = [r['pose'] for r in records]
    distances = member_distances(poses, members); cover = farthest_first(distances, first, COVER_RADIUS)
    relative = relative_poses(poses, cfg['fixed_poses'][0]); reconstructed = to_world(relative, cfg['fixed_poses'][0])
    t, _, rotations = arrays(poses); rt, _, rr = arrays(reconstructed)
    frame_errors = dict(position_A=near(t, rt), rotation=near(rotations, rr))
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses']); centers = []; assigned_chart_radius = np.zeros(len(records))
    for center_index, candidate_index in enumerate(cover['center_candidate_indices']):
        selected = records[candidate_index]; model, construction = geometry_model(cfg['metadata'], cfg['fixed_poses'][0], selected['pose'], shape_sha, peak_model['angular_length'])
        model_audit, geometry = geometry_model_audit(model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, selected['pose'])
        gaps = atom.gaps(selected['pose']); require(min(gaps) >= 0, 'Selected geometric center clashes atomic AB')
        assigned = [i for i, center in enumerate(cover['nearest_center_indices']) if center == center_index]
        radii, _, errors, _ = geometry_rows(t[assigned], rotations[assigned], geometry)
        assigned_chart_radius[assigned] = radii
        centers.append(dict(center_index=center_index, candidate_index=candidate_index, pose=selected['pose'], relative_pose=relative[candidate_index],
            q=selected['q'], source_kind=selected['source_kind'], seed=selected['seed'], draw=selected['draw'], population=selected['population'],
            original_log_importance_weight=selected['original_log_importance_weight'], source_samples_path=selected['source_samples_path'],
            source_samples_sha256=selected['source_samples_sha256'], assigned_candidate_indices=assigned,
            maximum_assigned_member_RMS_A=max(cover['nearest_member_RMS_A'][i] for i in assigned),
            maximum_assigned_geometry_chart_radius_A=float(max(radii)), minimum_AB_atomic_gaps_A=gaps,
            geometric_model=model, geometry_construction=construction, geometry_audit=model_audit, assigned_geometry_audit=errors))
    for i, row in enumerate(records):
        row.update(candidate_index=i, relative_pose=relative[i], nearest_center_index=cover['nearest_center_indices'][i],
            nearest_member_RMS_A=cover['nearest_member_RMS_A'][i], assigned_geometry_chart_radius_A=float(assigned_chart_radius[i]))
    center_indices = cover['center_candidate_indices']; center_distances = distances[np.ix_(center_indices, center_indices)]
    separation = min(center_distances[i, j] for i in range(len(centers)) for j in range(i)) if len(centers) > 1 else None
    require(max(cover['nearest_member_RMS_A']) <= COVER_RADIUS and (separation is None or separation > COVER_RADIUS), 'Observed cover or separation failed')
    for path, digest in hashes.items(): require(sha(Path(path)) == digest, 'Source changed during center preparation')
    archive = out/'provenance'; archive.mkdir()
    archive_sources = {**local_dependencies([Path(__file__)]), 'source-historical.json': HISTORY, 'source-local.json': LOCAL, 'source-flat.json': FLAT,
        'physical-config.json': PREPARATION/'config.json', 'shape.json': shape, 'original-peak-model.json': PREPARATION/'model.json'}
    for name, path in archive_sources.items(): (archive/name).write_bytes(path.read_bytes())
    result = dict(complete=True, protocol_sha256=sha(out/'protocol.json'), candidates=records, centers=centers,
        candidate_count=len(records), center_count=len(centers), selected_peak_candidate_index=first,
        center_candidate_indices=center_indices, nearest_center_indices=cover['nearest_center_indices'],
        maximum_nearest_member_RMS_A=max(cover['nearest_member_RMS_A']), minimum_center_separation_member_RMS_A=separation,
        maximum_assigned_geometry_chart_radius_A=float(max(assigned_chart_radius)), member_RMS_distance_matrix_A=distances.tolist(),
        farthest_first_stages=cover['stages'], coordinate_frame_audit=frame_errors, original_rows_audit=original_audit,
        fixed_neighbor=cfg['fixed_poses'][0], proposal_anchor_index=0, angular_length=peak_model['angular_length'],
        shape_sha256=shape_sha, original_q_window=WINDOW, source_sha256=hashes,
        archived_sha256={p.name: sha(p) for p in archive.iterdir()},
        scope='Every listed source pose has a selected center within0.5Å rigid-member RMS. This is an observed geometry cover, not a complete pose-space cover or equilibrium weighting. '
              'The supplied deterministic geometric charts are convenient proposal coordinates; a chart ball is contained in its RMS ball, so RMS coverage alone is not a uniform-chart-support certificate. '
              'Original importance weights and source selection masks are preserved solely as provenance; no physical draws, Poisson clouds, weight fit or covariance fit occurred.')
    write(out/'analysis.json', result)
    lines = ['# Frozen observed far-contact centers', '',
        f'{len(records)} archived candidates yield {len(centers)} deterministic centers. Maximum nearest member RMS: {result["maximum_nearest_member_RMS_A"]:.9g}Å.',
        '', '| Center | Candidate | Source | q | Assigned | Max member RMS Å | Minimum AB atomic gap Å |', '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for c in centers: lines.append(f"| {c['center_index']} | {c['candidate_index']} | {c['source_kind']} | {c['q']:.6g} | {len(c['assigned_candidate_indices'])} | {c['maximum_assigned_member_RMS_A']:.6g} | {min(c['minimum_AB_atomic_gaps_A']):.6g} |")
    lines += ['', result['scope'], '']; (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=True, output=str(out/'analysis.json'), candidates=len(records), centers=len(centers),
        maximum_nearest_member_RMS_A=result['maximum_nearest_member_RMS_A'], maximum_assigned_geometry_chart_radius_A=result['maximum_assigned_geometry_chart_radius_A'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args(); prepare(args.out)
