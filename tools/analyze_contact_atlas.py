#!/usr/bin/env python3
"""Audit independent full-window contact-atlas campaigns and fixed spatial masks.

Each arm retains its original frozen hybrid proposal density and full number of
unconditional draws. Historical reference regions are calibration regions, not
held-out proposal-selection data. No historical or cross-arm pooling is used.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import heapq
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_native_region_reference import logadd
from analyze_peak_neighborhood import geometry_model_audit, partition_check
from audit_shoulder_mis_independently import chart_values, near
from compare_intermediate_local_reference import read_batches
from compare_intermediate_reference import load_guided, population_rse
from prepare_cayley_rms_cover import read, require, sha, write
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments, quota_summary

ROOT = Path(__file__).resolve().parents[1]
PEAK = ROOT/'runs/ab-intermediate-new-peak-geometry-20260920'
OLD = ROOT/'runs/ab-intermediate-guide-preparation-20260920'
PHYSICAL_KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius',
                 'reservoir_density', 'metadata')
WINDOW = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
PEAK_BALLS = ('peak_r_le_0p5', 'peak_r_le_1', 'peak_r_le_2')
PEAK_PARTITION = ('peak_r_le_0p5', 'peak_0p5_lt_r_le_1', 'peak_1_lt_r_le_2', 'peak_r_gt_2')
OLD_PARTITION = ('old_r_le_4', 'old_4_lt_r_le_8', 'old_8_lt_r_le_16',
                 'old_16_lt_r_le_32', 'old_r_gt_32')
KEYS = ('full', *PEAK_BALLS, *PEAK_PARTITION[1:], *OLD_PARTITION)


def selected_keys(radius, old_radius):
    require(not math.isnan(radius) and radius >= 0, 'Invalid peak radius')
    require(not math.isnan(old_radius) and old_radius >= 0, 'Invalid old-chart radius')
    selected = ['full']
    for key, upper in zip(PEAK_BALLS, (.5, 1., 2.)):
        if radius <= upper: selected.append(key)
    if .5 < radius <= 1: selected.append(PEAK_PARTITION[1])
    elif 1 < radius <= 2: selected.append(PEAK_PARTITION[2])
    elif radius > 2: selected.append(PEAK_PARTITION[3])
    selected.append(next(key for key, upper in zip(OLD_PARTITION, (4., 8., 16., 32., math.inf))
                         if old_radius <= upper))
    return tuple(selected)


class AtlasMoments:
    """Direct full-N masked estimators, with same-row covariance retained."""
    def __init__(self):
        self.values = {kind: {key: LogMoments() for key in KEYS} for kind in ('physical', 'hard')}
        self.physical_hard_cross = {key: -math.inf for key in KEYS}

    def add(self, radius=None, old_radius=None, physical=None, hard=None, cloud_pair=None):
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        selected = ()
        if physical is not None:
            require(math.isfinite(physical) and math.isfinite(hard), 'Invalid original row weight')
            require(cloud_pair is not None and len(cloud_pair) == 2, 'Two original independent clouds required')
            near(float(np.logaddexp(*cloud_pair))-math.log(2), physical)
            selected = selected_keys(radius, old_radius)
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in KEYS:
                use = key in selected
                self.values[kind][key].add(value if use else -math.inf,
                    cloud_pair if use and kind == 'physical' else None)
        for key in selected:
            self.physical_hard_cross[key] = logadd(self.physical_hard_cross[key], physical+hard)

    def merge(self, other):
        for kind in self.values:
            for key in KEYS: self.values[kind][key].merge(other.values[kind][key])
        for key in KEYS:
            self.physical_hard_cross[key] = logadd(self.physical_hard_cross[key], other.physical_hard_cross[key])

    def report(self):
        result = {}
        for kind, values in self.values.items():
            result[kind] = {}
            for key, moment in values.items():
                row = quota_summary([moment]); row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = 'IID original-density row variance over full unconditional N, including all invalid and off-mask zeros'
                result[kind][key] = row
        paired = {}
        for key in KEYS:
            physical, hard = self.values['physical'][key], self.values['hard'][key]
            covariance = (math.expm1(math.log(physical.count)+self.physical_hard_cross[key]-physical.total-hard.total)
                          /(physical.count-1)) if physical.nonzero else None
            paired[key] = dict(relative_covariance_of_means=covariance,
                observed_SE_log_depletion_enhancement=(math.sqrt(max(0., result['physical'][key]['row_RSE']**2+
                    result['hard'][key]['row_RSE']**2-2*covariance)) if covariance is not None else None),
                qualification='Delta-method log-ratio diagnostic using the shared-row covariance; neither log nor ratio is unbiased')
        result['physical_hard_paired_statistics'] = paired
        result['partition_checks'] = {kind: {
            'peak_neighborhood_and_remaining_space': partition_check(values, PEAK_PARTITION),
            'old_weighted_chart': partition_check(values, OLD_PARTITION)} for kind, values in self.values.items()}
        return result


def audit_campaign(campaign, arm, preparation, peak_model, old_model, peak_pose):
    root = Path(campaign).resolve(); cfg = read(root/'provenance/config.json'); master = read(root/'manifest.json')
    protocol = read(preparation/'protocol.json')
    declared = next(item for item in protocol['arms'] if item['name'] == arm)
    require(Path(declared['output']).resolve() == root, 'Campaign differs from frozen output declaration')
    require(master['q_window'] == WINDOW, 'Require the original complete 2 <= q < 5 window')
    prepared_cfg = read(preparation/'config.json')
    require(all(cfg[key] == prepared_cfg[key] for key in PHYSICAL_KEYS), 'Physical target differs from preparation')
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and cfg['capture_radius'] == 18.,
            'Original physical baseline changed')
    require(sha(root/'provenance/guide-model.json') == sha(preparation/f'model-{arm}.json'), 'Frozen arm model changed')
    require(len(master['jobs']) == declared['populations'] and [job['seed'] for job in master['jobs']] == declared['seeds'],
            'Fixed population allocation changed')
    require(len({job['seed'] for job in master['jobs']}) == len(master['jobs']), 'Repeated population seed')
    require(master['total_unconditional_draws'] == declared['populations']*declared['samples_per_population'], 'Fixed total denominator changed')
    for job in master['jobs']:
        manifest = read(Path(job['output'])/'manifest.json')
        summary = read(Path(job['output'])/'summary.json')
        require(summary['complete'] and manifest['samples'] == summary['samples'] == declared['samples_per_population'],
                'Incomplete or changed fixed-N population')
        require(manifest['cloud_replicates'] == 2 and manifest['lambda_ratio'] == 64., 'Cloud law changed')
        require(manifest['executable_sha256'] == protocol['executable_sha256'], 'Frozen physical executable changed')
        require(manifest['guide']['model_sha256'] == sha(preparation/f'model-{arm}.json'), 'Population model changed')
        require(manifest['guide']['weight'] == .75 and manifest['guide']['uniform_probability'] == .05,
                'Unconditional defensive hybrid probabilities changed')
    print(f'{arm}: reconstructing every original q, hybrid density, cloud weight and denominator', flush=True)
    audited = load_guided(root)
    require(audited['shape_sha256'] == peak_model['shape_sha256'] == old_model['shape_sha256'], 'Chart/physical shapes differ')
    geometry, _ = geometry_model_audit(peak_model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, peak_pose)
    atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses'])
    aggregate = AtlasMoments(); populations = []; heap = []; ordinal = 0; input_hashes = {}
    for job in master['jobs']:
        path = Path(job['output']); local = AtlasMoments(); digest = hashlib.sha256(); n = 0
        for lines, rows in read_batches(path/'samples.jsonl'):
            for line in lines: digest.update(line)
            poses = [row['pose'] for row in rows]
            t = np.asarray([pose['position'] for pose in poses])
            q = np.asarray([pose['orientation'] for pose in poses])
            r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
            _, radius, _ = chart_values(peak_model, t, r, cfg['fixed_poses'][0])
            _, old_radius, _ = chart_values(old_model, t, r, cfg['fixed_poses'][0])
            for i, row in enumerate(rows):
                require(row['draw'] == n, 'Missing or reordered draw'); n += 1
                if 'zero' in row: local.add(); continue
                physical = row['log_importance_weight']; hard = row['log_hard_weight']
                pair = [value+hard for value in row['cloud_log_weights']]
                local.add(float(radius[i, 0]), float(old_radius[i, 0]), physical, hard, pair)
                point = dict(population=path.name, seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'],
                    original_log_importance_weight=physical, original_log_proposal_density=row['log_proposal_density'],
                    log_boltzmann_mean=row['log_boltzmann_mean'], cloud_log_weights=row['cloud_log_weights'],
                    cloud_overlap_counts=row['cloud_overlap_counts'], lower_volume=row['lower_volume'],
                    peak_radius_A=float(radius[i, 0]), old_weighted_radius=float(old_radius[i, 0]),
                    proposal_family=row['proposal_family'], guide_branch=row['guide_branch'], guide_component=row['guide_component'])
                entry = (physical, ordinal, point); ordinal += 1
                if len(heap) < 8: heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)
        require(n == read(path/'manifest.json')['samples'] and digest.hexdigest() == audited['sample_sha256'][str(path/'samples.jsonl')],
                'Audited sample bytes or unconditional count changed')
        input_hashes[str(path/'samples.jsonl')] = digest.hexdigest()
        for filename in ('manifest.json', 'summary.json'): input_hashes[str(path/filename)] = sha(path/filename)
        populations.append(dict(id=path.name, seed=job['seed'], samples=n, **local.report()))
        aggregate.merge(local)
    result = aggregate.report()
    for kind in ('physical', 'hard'):
        value, original = result[kind]['full']['logQ'], audited[kind]['all']['logQ']
        if value is None: require(original is None, 'Full zero estimate changed')
        else: near(value, original)
        for key in KEYS:
            result[kind][key]['independent_population_RSE'] = population_rse([p[kind][key]['logQ'] for p in populations])
    points = []
    for _, _, point in sorted(heap, reverse=True):
        gaps = atom.gaps(point['pose']); require(min(gaps) >= 0, 'Extreme fails independent atom-union overlap check')
        point['minimum_atomic_gap_by_neighbor_A'] = gaps
        point['fraction_of_full_weight'] = math.exp(point['original_log_importance_weight']-aggregate.values['physical']['full'].total)
        points.append(point)
    print(f'{arm}: logQ={result["physical"]["full"]["logQ"]}, rowRSE={result["physical"]["full"]["row_RSE"]}', flush=True)
    return dict(root=str(root), arm=arm, **result, populations=populations, top_poses=points,
        independent_atom_check_count=len(points), peak_geometry_reconstruction=geometry,
        CPU_seconds=audited['CPU_seconds'], original_full_window_audit=audited, input_sha256=input_hashes,
        qualification='Known original full hybrid density and full unconditional N. Per-arm estimates only; no historical or cross-arm pooling. Previous selected references are historical calibration, not untouched holdout validation.')


def analyze(preparation, campaigns, out):
    preparation, out = Path(preparation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use a fresh audit output directory')
    started = time.monotonic(); protocol = read(preparation/'protocol.json'); freeze = read(preparation/'freeze.json')
    require(sha(preparation/'protocol.json') == freeze['protocol_sha256'], 'Preparation protocol changed')
    require(sha(preparation/'config.json') == freeze['config_sha256'], 'Preparation config changed')
    for arm in ('narrow', 'broad'):
        require(sha(preparation/f'model-{arm}.json') == freeze['model_sha256'][arm], 'Preparation arm changed')
    for name, digest in freeze['archived_sha256'].items():
        require(sha(preparation/'provenance'/name) == digest, f'Preparation archive changed: {name}')
    for name, path in (('peak', PEAK/'model.json'), ('old_weighted', OLD/'model-weighted.json')):
        declared = protocol['analysis_charts'][name]
        require(Path(declared['path']).resolve() == path.resolve() and sha(path) == declared['sha256'], 'Analysis chart differs from frozen protocol')
    peak_protocol = read(PEAK/'protocol.json'); old_freeze = read(OLD/'freeze.json')
    require(sha(PEAK/'model.json') == peak_protocol['model_sha256'], 'Original peak chart changed')
    require(sha(PEAK/'selected-pose.json') == peak_protocol['selected_pose_sha256'], 'Selected peak changed')
    require(sha(OLD/'model-weighted.json') == old_freeze['model_sha256']['weighted'], 'Old partition chart changed')
    peak_model, old_model = read(PEAK/'model.json'), read(OLD/'model-weighted.json')
    peak_pose = read(PEAK/'selected-pose.json')['pose']
    sources = {'protocol.json': preparation/'protocol.json', 'freeze.json': preparation/'freeze.json', 'config.json': preparation/'config.json',
        'model-narrow.json': preparation/'model-narrow.json', 'model-broad.json': preparation/'model-broad.json',
        'peak-model.json': PEAK/'model.json', 'peak-protocol.json': PEAK/'protocol.json',
        'selected-pose.json': PEAK/'selected-pose.json', 'old-weighted-model.json': OLD/'model-weighted.json',
        'old-chart-freeze.json': OLD/'freeze.json'}
    input_hashes = {str(path): sha(path) for path in sources.values()}
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in {**local_dependencies([Path(__file__)]), **sources}.items():
        (archive/name).write_bytes(path.read_bytes())
    records = []; arms = set(); seeds = set()
    for campaign in campaigns:
        root = Path(campaign).resolve(); digest = sha(root/'provenance/guide-model.json')
        matching = [arm for arm in ('narrow', 'broad') if digest == sha(preparation/f'model-{arm}.json')]
        require(len(matching) == 1 and matching[0] not in arms, 'Unknown or duplicate frozen arm')
        arm = matching[0]; arms.add(arm)
        master = read(root/'manifest.json')
        current_seeds = {job['seed'] for job in master['jobs']}
        require(not current_seeds.intersection(seeds), 'Arms must use independent population seeds')
        seeds.update(current_seeds)
        (archive/f'{arm}-campaign-manifest.json').write_bytes((root/'manifest.json').read_bytes())
        record = audit_campaign(root, arm, preparation, peak_model, old_model, peak_pose)
        write(out/f'{arm}.json', record); records.append(record)
    for path, digest in input_hashes.items(): require(sha(path) == digest, 'Input changed during analysis')
    result = dict(complete=True, protocol_sha256=sha(preparation/'protocol.json'), campaigns=records,
        original_q_window=WINDOW, peak_ball_radii_A=[.5, 1., 2.], peak_partition=PEAK_PARTITION,
        old_chart_partition=OLD_PARTITION, elapsed_seconds=time.monotonic()-started,
        input_sha256=input_hashes, archived_sha256={p.name: sha(p) for p in archive.iterdir()},
        estimator='Each full-N masked positive original-density importance mean is unbiased conditional on its frozen proposal and numerical geometry. Logarithms and ratios are not unbiased.',
        coverage='Both arms retain complete q-window support. Zero observed contribution is not a mass upper bound; observed RSE and ESS do not certify absence of unseen tails.',
        calibration='Selected historical extremes and the small-ball reference were used to prepare the atlas. Their local integrals remain useful calibration, but are not untouched holdout validation.',
        variance='Direct full-row totals retain covariance of all masks. Disjoint mask sums are checked against the same-row covariance, never independent-bin variances.',
        scope='Conditional fixed-AB contact weights only; excludes neighbor assembly cost, kinetic mixing, and the rest of native/shoulder/distant configuration space.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); analyze(args.preparation, args.campaign, args.out)
