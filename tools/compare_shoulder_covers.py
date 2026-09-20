#!/usr/bin/env python3
"""Compare physical and hard shoulder weights under independent complete covers.

Coverage is an external geometric proof obligation, not inferred from agreement.
All estimates retain unconditional sample counts and exclude old fitting poses.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp

from analyze_native_region_reference import native_q
from analyze_shoulder_guides import BANDS, ROOT, statistics
from prepare_smc_normalizer_atlas import read, sha, write

WINDOW = {'minimum': 1., 'maximum': 2., 'lower_inclusive': False, 'upper_inclusive': False}
PHYSICAL_KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def population_error(populations, key):
    logs = [p[key]['logQ'] for p in populations]
    finite = [x for x in logs if x is not None]
    if not finite:
        return None
    weights = np.array([0. if x is None else math.exp(x-max(finite)) for x in logs])
    return float(np.std(weights, ddof=1)/math.sqrt(len(weights))/weights.mean())


def analyze(campaign, kind, expected, excluded_poses, seen_seeds):
    manifest = read(campaign/'manifest.json')
    for name, digest in manifest['archive_sha256'].items():
        assert sha(campaign/'provenance'/name) == digest
    cfg = read(campaign/'provenance/config.json')
    assert {k: cfg[k] for k in PHYSICAL_KEYS} == expected
    region = read(campaign/'provenance/region.json') if kind == 'cayley' else None
    if region:
        assessment_path = campaign/'assessment/analysis.json'; assessed = read(assessment_path)
        assert assessed['original_q_window'] == WINDOW
        assert region['physical_fixed_neighbors'] == expected['fixed_poses']
        assert region['physical_metric'] == expected['metadata']
        assert assessed['region_sha256'] == sha(campaign/'provenance/region.json')
        summaries = {p['id']: p for p in assessed['populations']}
    else:
        assessment_path = campaign/'assessment-streaming.json'; assessed = read(assessment_path)
        assert assessed['complete'] and assessed['all_rows_and_hashes_validated']
        assert assessed['q_window'] == WINDOW
        summaries = {p['replicate']: p for p in assessed['populations']}
    physical, hard = [], []
    bands = defaultdict(list); hard_bands = defaultdict(list)
    half_physical = [[], []]; half_hard = [[], []]
    count = 0; populations = []; cpu = 0.; shape_hash = None; population_counts = set()
    hashes = {}; q_error = 0.; top = []; cloud_pairs = []
    for job in manifest['jobs']:
        directory = Path(job['directory'] if region else job['output'])
        summary = read(directory/'summary.json'); run = read(directory/'manifest.json')
        assert summary['complete'] and run['seed'] == job['seed']
        assert job['seed'] not in seen_seeds; seen_seeds.add(job['seed'])
        assert run['config_sha256'] == sha(campaign/'provenance/config.json')
        shape_hash = sha(campaign/'provenance/shape.json')
        assert shape_hash == run['shape_sha256']
        assert run['cloud_replicates'] == 2 and run['lambda_ratio'] == 64
        assert run['activity'] == expected['reservoir_density']
        if region:
            assert run['region_sha256'] == sha(campaign/'provenance/region.json')
            assert region['depletant_radius'] == expected['depletant_radius']
        else:
            assert run['depletant_radius'] == expected['depletant_radius']
        metric = {k: cfg['metadata'][k] for k in ('angle_error_scale_deg', 'member_error_scale', 'native_poses', 'rigid_members')}
        if not region: assert run['metric'] == metric and run['q_window'] == WINDOW
        n = run['samples']; assert n%2 == 0
        population_counts.add(n)
        local, local_hard = [], []; local_bands = defaultdict(list); local_hard_bands = defaultdict(list)
        digest = hashlib.sha256(); actual_count = 0
        with (directory/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                assert row['draw'] == actual_count; actual_count += 1
                if region:
                    valid = row['region_valid'] and row['hard_valid'] and row['capture_valid']
                    assert valid == (row['log_importance_weight'] is not None)
                else:
                    valid = 'zero' not in row
                if not valid: continue
                assert json.dumps(row['pose'], sort_keys=True) not in excluded_poses
                q = native_q(metric, row['pose']); q_error = max(q_error, abs(q-row['q']))
                assert 1 < q < 2 and abs(q-row['q']) < 2e-8
                assert math.dist(row['pose']['position'], cfg['capture_center']) <= cfg['capture_radius']
                logz, logh = row['log_importance_weight'], row['log_hard_weight']
                clouds = [c['log_weight'] for c in row['clouds']] if region else row['cloud_log_weights']
                assert len(clouds) == 2 and math.isfinite(logz) and math.isfinite(logh)
                assert abs(float(logsumexp(clouds))-math.log(2)+logh-logz) < 2e-9
                cloud_pairs.append([logh+cloud for cloud in clouds])
                band = next(i for i, (a, b) in enumerate(BANDS) if a <= q < b)
                physical.append(logz); hard.append(logh); local.append(logz); local_hard.append(logh)
                bands[band].append(logz); hard_bands[band].append(logh)
                local_bands[band].append(logz); local_hard_bands[band].append(logh)
                half = int(row['draw'] >= n//2)
                half_physical[half].append(logz); half_hard[half].append(logh)
                top.append(dict(row, population=directory.name))
        assert actual_count == n == summary['samples']
        sample_hash = digest.hexdigest(); assert sample_hash == summary['samples_sha256']
        audit_key = job['id'] if region else directory.name
        assert sample_hash == summaries[audit_key]['samples_sha256' if region else 'sample_sha256']
        hashes[str(directory/'samples.jsonl')] = sample_hash; count += n
        cpu += summary['sampler_cpu_seconds'] if region else summary['cpu_seconds']
        populations.append({'id': directory.name, 'seed': job['seed'], 'physical': statistics(local, n),
            'hard': statistics(local_hard, n), 'bands': {str(i): statistics(local_bands[i], n) for i in range(len(BANDS))},
            'hard_bands': {str(i): statistics(local_hard_bands[i], n) for i in range(len(BANDS))}})
    assert len(population_counts) == 1, 'Independent-population errors require equal fixed budgets here'
    total = statistics(physical, count); hard_total = statistics(hard, count)
    prior_physical = assessed['estimate'] if region else assessed['regions']['region']
    prior_hard = assessed['hard_region'] if region else assessed['hard_regions']['region']
    field = 'logQ' if region else 'log_normalizer'
    assert abs(total['logQ']-prior_physical[field]) < 2e-8
    assert abs(hard_total['logQ']-prior_hard[field]) < 2e-8
    cloud_array = np.asarray(cloud_pairs)
    offset = float(cloud_array.max()); cloud_weights = np.exp(cloud_array-offset)
    average = cloud_weights.mean(axis=1); full_mean = float(average.sum()/count)
    variance = (float(average@average)-count*full_mean**2)/(count-1)
    cloud_variance = float(np.sum((cloud_weights[:, 0]-cloud_weights[:, 1])**2)/(4*count))
    output = {'root': str(campaign), 'kind': kind, 'physical': total, 'hard': hard_total,
        'physical_population_RSE': population_error(populations, 'physical'),
        'hard_population_RSE': population_error(populations, 'hard'),
        'hard_volume_A3': math.exp(hard_total['logQ']), 'CPU_seconds': cpu,
        'paired_cloud_variance_fraction': cloud_variance/variance,
        'physical_importance_ESS_per_CPU': total['ESS']/cpu, 'hard_importance_ESS_per_CPU': hard_total['ESS']/cpu,
        'bands': {str(i): statistics(bands[i], count) for i in range(len(BANDS))},
        'hard_bands': {str(i): statistics(hard_bands[i], count) for i in range(len(BANDS))},
        'populations': populations, 'complementary_halves': [{'physical': statistics(z, count//2), 'hard': statistics(h, count//2)}
            for z, h in zip(half_physical, half_hard)],
        'half_scope': 'Retrospective complementary halves of every fixed population; independent draws under the same frozen proposal, not extra independent campaigns.',
        'sample_sha256': hashes, 'shape_sha256': shape_hash, 'config_sha256': sha(campaign/'provenance/config.json'),
        'manifest_sha256': sha(campaign/'manifest.json'), 'assessment_sha256': sha(assessment_path),
        'maximum_original_q_error': q_error,
        'top_16': sorted(top, key=lambda r: r['log_importance_weight'], reverse=True)[:16]}
    for i in range(len(BANDS)):
        item = output['bands'][str(i)]; hard_item = output['hard_bands'][str(i)]
        item['fraction_of_observed_total'] = 0. if item['logQ'] is None else math.exp(item['logQ']-total['logQ'])
        hard_item['fraction_of_observed_hard_volume'] = 0. if hard_item['logQ'] is None else math.exp(hard_item['logQ']-hard_total['logQ'])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cayley-root', type=Path, default=ROOT/'runs/ab-shoulder-cayley-reference-4x262144-l64-20260920')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-native-campaign', nargs=2, action='append', default=[],
                        metavar=('LABEL', 'PATH'), help='Keep each fresh native-format confirmation as a separate estimate')
    args = parser.parse_args(); out = args.out.resolve()
    assert not out.exists(), 'Use a fresh comparison output directory'
    source = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
    cfg = read(source/'config.json'); expected = {k: cfg[k] for k in PHYSICAL_KEYS}
    old = read(source/'provenance/input-shoulder-poses.json')
    poses = {json.dumps(r['pose'], sort_keys=True) for r in old}; seeds = {r['seed'] for r in old}
    plan = read(ROOT/'runs/ab-shoulder-window-pilot-20260920/protocol.json')
    jobs = [(j['name'], Path(j['root']), 'product_or_guide') for j in plan['jobs']]
    jobs.append(('cayley', args.cayley_root.resolve(), 'cayley'))
    jobs.extend((name, Path(path).resolve(), 'product_or_guide') for name, path in args.extra_native_campaign)
    assert len({name for name, _, _ in jobs}) == len(jobs), 'Campaign labels must be distinct'
    results = {name: analyze(path, kind, expected, poses, seeds) for name, path, kind in jobs}
    assert len({r['shape_sha256'] for r in results.values()}) == 1
    report = {'physical': expected, 'q_window': WINDOW, 'bands': BANDS, 'campaigns': results,
        'analysis_sha256': sha(__file__),
        'scope': 'Original strict shoulder, fixed full AB, unconditional zeros retained. Fitting and geometry probes excluded. Completeness is supported by separate geometry/proof checks, never inferred from agreement or finite samples. Importance ESS is not trajectory contact ESS.'}
    out.mkdir(parents=True); write(out/'comparison.json', report)
    for file in ('compare_shoulder_covers.py', 'analyze_shoulder_guides.py', 'analyze_native_region_reference.py', 'prepare_smc_normalizer_atlas.py'):
        shutil.copy2(ROOT/'tools'/file, out/file)
    print(json.dumps({name: {key: r[key] for key in ('physical', 'hard', 'physical_population_RSE', 'hard_population_RSE',
        'hard_volume_A3', 'CPU_seconds', 'physical_importance_ESS_per_CPU', 'hard_importance_ESS_per_CPU', 'bands', 'hard_bands', 'complementary_halves')}
        for name, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
