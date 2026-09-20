#!/usr/bin/env python3
"""Direct nonnegative fixed-N masses inside/outside a previously frozen region."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import Density, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration
from analyze_basin_normalizers import moments, paired_noise


def split_fixed_n(logs, hard_logs, pairs, inside):
    logs, hard_logs, pairs, inside = map(np.asarray, (logs, hard_logs, pairs, inside))
    assert len(logs) == len(hard_logs) == len(pairs) == len(inside)
    output = {}
    for name, mask in [('total', np.ones(len(logs), dtype=bool)), ('inside_old_R8', inside), ('outside_old_R8', ~inside)]:
        selected = np.where(mask, logs, -np.inf)
        selected_hard = np.where(mask, hard_logs, -np.inf)
        selected_pairs = np.where(mask[:, None], pairs, -np.inf)
        estimate = moments(selected)
        estimate['hard_region'] = moments(selected_hard)
        estimate['paired_noise'] = paired_noise(selected, selected_pairs)
        estimate['log_regional_depletion_enhancement'] = (estimate['logQ']-estimate['hard_region']['logQ']
            if estimate['logQ'] is not None and estimate['hard_region']['logQ'] is not None else None)
        output[name] = estimate
    parts = [output[name]['logQ'] for name in ('inside_old_R8', 'outside_old_R8') if output[name]['logQ'] is not None]
    if parts:
        assert abs(logsumexp(parts)-output['total']['logQ']) < 1e-10
    else:
        assert output['total']['logQ'] is None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    root, plan_path = args.root.resolve(), args.plan.resolve()
    plan, manifest = read(plan_path), read(root/'manifest.json')
    region = read(root/'provenance/region.json')
    old_path = Path(plan['poststratification']['old_region'])
    assert sha(old_path) == plan['poststratification']['old_region_sha256']
    old = read(old_path)
    assert plan['poststratification']['old_radius'] == 8.
    assert region['minimum_original_q'] == old['minimum_original_q'] == 5.
    key = str(int(region['mahalanobis_radius']))
    assert region['mahalanobis_radius'] in (3., 5.)
    assert manifest['region_sha256'] == plan['regions'][key]['sha256'] == sha(root/'provenance/region.json')
    for field in ('fixed_neighbor', 'capture_center', 'capture_radius', 'shape_sha256', 'activity', 'depletant_radius', 'physical_metric'):
        assert region[field] == old[field]
    old_density = Density(old['gaussian_chart'])
    logs, hard_logs, pairs, inside_flags = [], [], [], []
    populations, source_hashes = [], []
    log_volume = 3*np.log(np.pi)+6*np.log(region['mahalanobis_radius'])-np.log(6.)
    for job in manifest['jobs']:
        directory = Path(job['directory'])
        summary = read(directory/'summary.json')
        assert summary['complete'] and summary['samples'] == job['samples']
        assert summary['manifest']['region_sha256'] == manifest['region_sha256']
        assert summary['manifest']['seed'] == job['seed']
        rows = [json.loads(line) for line in (directory/'samples.jsonl').open()]
        assert [row['draw'] for row in rows] == list(range(job['samples']))
        poses = [row['pose'] for row in rows]
        radii = old_density.evaluate(relative_poses(poses, region['fixed_neighbor']))[1][:, 0]
        q = registration(poses, region['physical_metric'])
        assert np.max(np.abs(q-np.asarray([row['q'] for row in rows]))) < 2e-8
        inside = radii <= 8.
        local_logs, local_hard, local_pairs = [], [], []
        for row in rows:
            valid = row['capture_valid'] and row['hard_valid'] and row['region_valid']
            if valid:
                assert row['q'] >= 5. and len(row['clouds']) == 2
                hard = log_volume+row['log_physical_jacobian']
                assert abs(hard-row['log_hard_weight']) < 1e-10
                p = [hard+cloud['log_weight'] for cloud in row['clouds']]
                assert abs(logsumexp(p)-np.log(2)-row['log_importance_weight']) < 1e-10
                local_logs.append(row['log_importance_weight']); local_hard.append(hard); local_pairs.append(p)
            else:
                assert row['log_importance_weight'] is None and row['log_hard_weight'] is None and not row['clouds']
                local_logs.append(-np.inf); local_hard.append(-np.inf); local_pairs.append([-np.inf, -np.inf])
        split = split_fixed_n(local_logs, local_hard, local_pairs, inside)
        recorded = summary['estimates']['region']['logQ']
        assert recorded is None if split['total']['logQ'] is None else abs(recorded-split['total']['logQ']) < 1e-10
        populations.append(dict(id=job['id'], seed=job['seed'], estimates=split))
        logs.extend(local_logs); hard_logs.extend(local_hard); pairs.extend(local_pairs); inside_flags.extend(inside)
        source_hashes.append(dict(id=job['id'], samples_sha256=sha(directory/'samples.jsonl'), summary_sha256=sha(directory/'summary.json')))
    result = split_fixed_n(logs, hard_logs, pairs, inside_flags)
    for name, estimate in result.items():
        poplogs = [p['estimates'][name]['logQ'] for p in populations]
        estimate['independent_populations'] = moments([-np.inf if p is None else p for p in poplogs])
        estimate['independent_populations']['logQ_values'] = poplogs
    out = root/'assessment'
    out.mkdir(exist_ok=True)
    report = dict(plan_sha256=sha(plan_path), new_region_sha256=manifest['region_sha256'], old_region_sha256=sha(old_path),
        new_radius=region['mahalanobis_radius'], old_radius=8., estimates=result, populations=populations,
        source_hashes=source_hashes, analyzer_sha256=sha(__file__),
        scope='Each nonnegative subgroup integral retains every unconditional draw of this campaign, including invalid and opposite-group zeros. Groups sum to this NEW regional mass, not the old R8 mass or a global normalizer. No subtraction or pooling across newR3/newR5.',
        enhancement='Qz/Q0 is a correlated same-pose ratio; point value only. Zero observed mass is unresolved, not a physical zero or upper bound.')
    write(out/'old-region-overlap.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
