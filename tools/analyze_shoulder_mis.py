#!/usr/bin/env python3
"""Audit a fresh fixed-quota MIS campaign with one full physical denominator."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from analyze_latent_region import original_q_window, validate_manifest_q_window
from analyze_native_region_reference import (DEFAULT_Q_WINDOW, GaussianGuide, hybrid_log_density, native_q,
    proposal_model, q_in_window, validate_window_cover, window_cover_metric)
from prepare_smc_normalizer_atlas import read, sha, write
from run_shoulder_mis_campaign import verify_cover, verify_archive
from shoulder_mis import (LogMoments, ProposalDensities, mixture_log_density,
                          quota_summary, require, signed_quota_difference)

FAMILIES = ('guide', 'cover')
PHYSICAL_KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def near(a, b, tolerance=2e-8, label='Numerical mismatch'):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, label)


def check_summary(moment, recorded, key):
    ours = quota_summary([moment])
    if ours['logQ'] is None:
        require(recorded[key] is None, 'Recorded nonzero normalizer with all-zero rows')
    else:
        near(ours['logQ'], recorded[key], label='Original component normalizer changed')


def population_rse(values):
    finite = [v for v in values if v is not None]
    if len(values) < 2 or not finite:
        return None
    offset = max(finite)
    means = np.asarray([0. if v is None else math.exp(v-offset) for v in values])
    return float(means.std(ddof=1)/math.sqrt(len(means))/means.mean())


def groups(keys):
    return {key: {family: LogMoments() for family in FAMILIES} for key in keys}


def process_population(population, master, archive, config, model, region, edges):
    keys = ['all']+[str(i) for i in range(len(edges)-1)]
    physical, hard, own, own_hard = (groups(keys) for _ in range(4))
    differences = {baseline: groups(keys) for baseline in FAMILIES}
    outputs = {family: Path(population[family]['output']) for family in FAMILIES}
    manifests = {family: read(path/'manifest.json') for family, path in outputs.items()}
    summaries = {family: read(path/'summary.json') for family, path in outputs.items()}
    gm = manifests['guide']
    cover_base = read(outputs['guide']/'cover.json')
    require(gm['schema'] in (3, 4), 'MIS guide requires a detailed frozen-guide manifest')
    window_manifest = gm
    if gm['schema'] == 3:
        require(master['q_window'] == DEFAULT_Q_WINDOW, 'Legacy guide is only the inclusive native window')
        window_manifest = dict(gm, q_window=DEFAULT_Q_WINDOW,
                               cover_metric=window_cover_metric(gm['metric'], DEFAULT_Q_WINDOW))
    require(validate_window_cover(window_manifest, cover_base, config) == master['q_window'], 'Guide target window changed')
    pm = proposal_model(gm, cover_base)
    description = gm['guide']
    require(description['model_sha256'] == master['model_sha256'], 'Different frozen guide model')
    for field, expected in [('weight', master['guide']['model_weight']),
                            ('uniform_probability', master['guide']['uniform_probability']),
                            ('anchor_index', master['guide']['anchor_index'])]:
        require(description[field] == expected, 'Guide allocation changed')
    densities = ProposalDensities(config, model, description, pm, region)
    scalar_guide = GaussianGuide(description, model, config, master['shape_sha256'])
    allocation = master['allocation']
    rows_audited = 0
    source_density_error = 0.
    cross_density_error = 0.
    q_error = 0.
    density_cpu = 0.
    cpu = {}
    hashes = {}
    corrected = []
    for family in FAMILIES:
        path, manifest, summary = outputs[family], manifests[family], summaries[family]
        job = population[family]
        n = allocation[family]
        require(n == job['samples'] == manifest['samples'] == summary['samples'], 'Fixed quota changed')
        require(summary['complete'] and manifest['seed'] == job['seed'], 'Incomplete or changed population')
        require(manifest['config_sha256'] == master['config_sha256'], 'Different physical configuration bytes')
        require(manifest['shape_sha256'] == master['shape_sha256'], 'Different physical shape bytes')
        binary = 'native-region-normalizer' if family == 'guide' else 'latent-region-normalizer'
        require(manifest['executable_sha256'] == master['archive_sha256'][binary], 'Different executable')
        require(manifest['cloud_replicates'] == master['cloud_replicates'] == 2, 'Different cloud count')
        require(manifest['activity'] == config['reservoir_density'], 'Different activity')
        require(manifest['lambda_ratio'] == master['lambda_ratio'], 'Different cloud intensity')
        z = config['reservoir_density']
        lam = z*master['lambda_ratio'] if z > 0 else 1.
        require(manifest['lambda'] == lam, 'Different cloud law')
        if family == 'guide':
            require(gm['depletant_radius'] == config['depletant_radius'], 'Different depletant radius')
            for filename, field in [('config.json', 'config_sha256'), ('shape.json', 'shape_sha256'),
                                    ('source-bundle.json', 'source_bundle_sha256')]:
                require(sha(path/'provenance'/filename) == manifest[field], 'Changed native provenance')
            require(sha(path/'provenance/guide-model.json') == master['model_sha256'], 'Changed model provenance')
        else:
            require(manifest == summary['manifest'], 'Latent summary and manifest differ')
            require(manifest['region_sha256'] == master['region_sha256'], 'Changed latent region')
            require(manifest['physical_fixed_neighbors'] == config['fixed_poses'], 'Lost physical neighbor')
            require(manifest['chart_anchor'] == region['fixed_neighbor'], 'Changed chart anchor')
            validate_manifest_q_window(manifest, region, master['q_window'])
            near(manifest['log_latent_shell_volume'], densities.log_volume)
            for filename, field in [('input-config.json', 'config_sha256'), ('shape.json', 'shape_sha256'),
                                    ('region.json', 'region_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
                require(sha(path/'provenance'/filename) == manifest[field], 'Changed latent provenance')
        digest = hashlib.sha256()
        count = 0
        with (path/'samples.jsonl').open('rb') as handle:
            while True:
                raw = []
                for _ in range(4096):
                    line = handle.readline()
                    if not line:
                        break
                    digest.update(line)
                    raw.append(json.loads(line))
                if not raw:
                    break
                density_started = time.process_time()
                guide_logs, cover_logs, radii = densities.evaluate([row['pose'] for row in raw])
                mixed_logs = mixture_log_density(guide_logs, cover_logs, allocation['guide'], allocation['cover'])
                density_cpu += time.process_time()-density_started
                for row, lg, lc, lm, radius in zip(raw, guide_logs, cover_logs, mixed_logs, radii):
                    require(row['draw'] == count, 'Missing or repeated unconditional draw')
                    count += 1
                    q = native_q(config['metadata'], row['pose'])
                    q_error = max(q_error, abs(q-row['q']))
                    near(q, row['q'], label='Original q changed')
                    window_ok = q_in_window(q, master['q_window'])
                    capture_ok = math.dist(row['pose']['position'], config['capture_center']) <= config['capture_radius']
                    own_log = lg if family == 'guide' else lc
                    require(math.isfinite(own_log) and math.isfinite(lm), 'Generated draw lacks unconditional density')
                    if family == 'guide':
                        recorded_log = row['log_proposal_density']
                        reason = row.get('zero')
                        require((reason == 'q') == (not window_ok), 'q rejection mismatch')
                        if window_ok:
                            require((reason == 'capture') == (not capture_ok), 'Capture rejection mismatch')
                        require(reason in (None, 'q', 'capture', 'hard'), 'Unknown rejection')
                        valid = reason is None
                    else:
                        recorded_log = -densities.log_volume-row['log_physical_jacobian']
                        near(radius, row['latent_radius'], label='Latent radius reconstruction mismatch')
                        require(row['region_valid'] == window_ok and row['capture_valid'] == capture_ok,
                                'Latent target predicate changed')
                        valid = row['region_valid'] and row['capture_valid'] and row['hard_valid']
                    error = abs(own_log-recorded_log)
                    source_density_error = max(source_density_error, error)
                    near(own_log, recorded_log, label='Source proposal density mismatch')
                    if not valid:
                        require(row.get('log_importance_weight') is None and row.get('log_hard_weight') is None,
                                'Invalid draw retained a nonzero contribution')
                        if family == 'cover':
                            require(not row['clouds'], 'Invalid latent draw unexpectedly carries clouds')
                        continue
                    require(window_ok and capture_ok, 'A nonzero pose lies outside target')
                    require(math.isfinite(lc), 'Complete moment cover misses a target pose')
                    scalar = hybrid_log_density(pm, scalar_guide, row['pose'])[0]
                    cross_density_error = max(cross_density_error, abs(lg-scalar))
                    near(lg, scalar, label='Independent scalar cross-density mismatch')
                    if family == 'guide':
                        cloud_logs = row['cloud_log_weights']
                        require(len(cloud_logs) == len(row['cloud_overlap_counts']) == len(row['cloud_raw_points']) == 2, 'Wrong cloud count')
                        require(all(type(k) is int and type(raw_count) is int and 0 <= k <= raw_count
                            for k, raw_count in zip(row['cloud_overlap_counts'], row['cloud_raw_points'])), 'Invalid Poisson counts')
                        for value, k in zip(cloud_logs, row['cloud_overlap_counts']):
                            near(value, z*row['lower_volume']+k*math.log1p(z/lam), label='Poisson weight changed')
                    else:
                        require(len(row['clouds']) == 2, 'Wrong latent cloud count')
                        cloud_logs = [c['log_weight'] for c in row['clouds']]
                        for c in row['clouds']:
                            require(type(c['overlap_points']) is int and type(c['raw_points']) is int
                                and 0 <= c['overlap_points'] <= c['raw_points'], 'Invalid latent Poisson counts')
                            near(c['log_weight'], z*c['lower_volume']+c['overlap_points']*math.log1p(z/lam),
                                 label='Poisson weight changed')
                    mean = float(logsumexp(cloud_logs))-math.log(2)
                    original = mean-own_log
                    corrected_log = mean-lm
                    near(original, row['log_importance_weight'], label='Original importance numerator changed')
                    near(-own_log, row['log_hard_weight'], label='Original hard-volume weight changed')
                    band = min(len(edges)-2, int(np.searchsorted(edges, q, side='right'))-1)
                    require(band >= 0, 'Missing target band')
                    for key in ('all', str(band)):
                        physical[key][family].add(corrected_log, [value-lm for value in cloud_logs])
                        hard[key][family].add(-lm)
                        own[key][family].add(original, [value-own_log for value in cloud_logs])
                        own_hard[key][family].add(-own_log)
                        for baseline in FAMILIES:
                            # The source-only estimator divides by n_baseline,
                            # whereas MIS divides by N. Its own-stratum
                            # difference is negative with this exact magnitude.
                            if baseline == family:
                                other = 'cover' if family == 'guide' else 'guide'
                                log_other = lc if other == 'cover' else lg
                                adjustment = (math.log(allocation[other]/allocation[family])
                                              +log_other-own_log)
                            else:
                                adjustment = 0.
                            differences[baseline][key][family].add(corrected_log+adjustment,
                                [value-lm+adjustment for value in cloud_logs])
                    corrected.append({'population': population['id'], 'family': family, 'draw': row['draw'], 'q': q,
                        'pose': row['pose'], 'log_guide_density': float(lg), 'log_cover_density': float(lc),
                        'log_mixture_density': float(lm), 'log_boltzmann_mean': mean,
                        'log_importance_weight': corrected_log, 'log_hard_weight': -float(lm),
                        'source_log_importance_weight': original, 'cloud_log_weights': cloud_logs})
        require(count == n, 'Prescribed unconditional count not reached')
        require(digest.hexdigest() == summary['samples_sha256'], 'Raw sample hash changed')
        hashes[family] = digest.hexdigest()
        rows_audited += count
        for table in (physical, hard, own, own_hard, *differences.values()):
            for key in keys:
                require(table[key][family].count <= n, 'Overcounted valid rows')
                table[key][family].count = n  # All other prescribed draws are exact zeros.
        if family == 'guide':
            key = 'region' if gm['schema'] == 4 else 'native'
            check_summary(own['all'][family], summary[key], 'log_normalizer')
            check_summary(own_hard['all'][family], summary['hard_'+key], 'log_normalizer')
        else:
            check_summary(own['all'][family], summary['estimates']['region'], 'logQ')
            check_summary(own_hard['all'][family], summary['estimates']['hard_region'], 'logQ')
        cpu[family] = summary['cpu_seconds'] if family == 'guide' else summary['sampler_cpu_seconds']
    result = {'id': population['id'], 'seeds': {f: population[f]['seed'] for f in FAMILIES},
              'physical': {key: quota_summary(list(physical[key].values())) for key in keys},
              'hard': {key: quota_summary(list(hard[key].values())) for key in keys},
              'component_only': {f: {key: quota_summary([own[key][f]]) for key in keys} for f in FAMILIES},
              'component_only_hard': {f: {key: quota_summary([own_hard[key][f]]) for key in keys} for f in FAMILIES},
              'CPU_seconds': cpu, 'sample_sha256': hashes, 'rows_audited': rows_audited,
              'vector_density_CPU_seconds': density_cpu,
              'maximum_source_density_error': source_density_error, 'maximum_cross_density_error': cross_density_error,
              'maximum_original_q_error': q_error}
    result['paired_differences'] = {baseline: {key: signed_quota_difference(
        differences[baseline][key][baseline], differences[baseline][key]['cover' if baseline == 'guide' else 'guide'],
        result['component_only'][baseline][key]['logQ']) for key in keys} for baseline in FAMILIES}
    return result, physical, hard, own, own_hard, differences, corrected


def analyze(root, output=None):
    root = Path(root).resolve()
    started = time.monotonic()
    cpu_started = time.process_time()
    master = read(root/'manifest.json')
    verify_archive(root, master)
    status = read(root/'runner-status.json')
    require(status.get('complete') and status.get('success'), 'Campaign is not completely successful')
    archive = root/'provenance'
    config, model, region = (read(archive/name) for name in ('config.json', 'guide-model.json', 'region.json'))
    require({k: config[k] for k in PHYSICAL_KEYS} == master['physical'], 'Physical fields differ')
    require(sha(archive/'config.json') == master['config_sha256'], 'Configuration identity mismatch')
    require(sha(archive/'shape.json') == master['shape_sha256'] == region['shape_sha256'], 'Shape identity mismatch')
    require(sha(archive/'guide-model.json') == master['model_sha256'], 'Guide identity mismatch')
    require(sha(archive/'region.json') == master['region_sha256'], 'Cover identity mismatch')
    require(original_q_window(region) == master['q_window'], 'Original window changed')
    verify_cover(region, config, master['shape_sha256'], master['q_window'])
    allocation = master['allocation']
    require(min(allocation['guide'], allocation['cover']) >= 2, 'At least two samples per fixed quota required')
    require(allocation['total'] == allocation['guide']+allocation['cover'], 'Quota sum changed')
    for family in FAMILIES:
        require(allocation['alpha_'+family] == allocation[family]/allocation['total'], 'Mixture allocation changed')
    seeds = [p[f]['seed'] for p in master['populations'] for f in FAMILIES]
    require(len(seeds) == len(set(seeds)), 'Repeated family or population seed')
    lower, upper = master['q_window']['minimum'], master['q_window']['maximum']
    edges = [lower]+[b for b in (1.1, 1.25, 1.5) if lower < b < upper]+[upper]
    keys = ['all']+[str(i) for i in range(len(edges)-1)]
    merged = [groups(keys) for _ in range(4)]
    difference_merged = {baseline: groups(keys) for baseline in FAMILIES}
    populations, corrected = [], []
    for population in master['populations']:
        for family in FAMILIES:
            executed = status['jobs'][population['id']+'-'+family]
            require(executed['status'] == 'complete' and executed['exit_code'] == 0, 'A prescribed component did not complete')
            require(sha(Path(population[family]['output'])/'summary.json') == executed['summary_sha256'],
                    'Component summary changed after execution')
        result, part_p, part_h, part_o, part_oh, diff, records = process_population(population, master, archive, config, model, region, edges)
        populations.append(result)
        corrected.extend(records)
        for pooled, part in zip(merged, (part_p, part_h, part_o, part_oh)):
            for key in keys:
                for family in FAMILIES:
                    pooled[key][family].merge(part[key][family])
        for baseline in FAMILIES:
            for key in keys:
                for family in FAMILIES:
                    difference_merged[baseline][key][family].merge(diff[baseline][key][family])
    require(len(populations) == master['population_count'], 'Population count changed')
    require(sum(p['rows_audited'] for p in populations) == master['total_unconditional_draws'], 'Unconditional count changed')
    physical, hard, own, own_hard = merged
    total_cpu = sum(sum(p['CPU_seconds'].values()) for p in populations)
    estimates = {key: quota_summary(list(physical[key].values())) for key in keys}
    for key, estimate in estimates.items():
        estimate['independent_population_RSE'] = population_rse([p['physical'][key]['logQ'] for p in populations])
        estimate['relative_variance_times_CPU'] = (None if estimate['stratified_RSE'] is None else estimate['stratified_RSE']**2*total_cpu)
    result = {'complete': True, 'all_rows_audited': True, 'manifest_sha256': sha(root/'manifest.json'),
        'physical': master['physical'], 'shape_sha256': master['shape_sha256'], 'q_window': master['q_window'],
        'allocation': allocation, 'bands': list(zip(edges[:-1], edges[1:])), 'estimates': estimates,
        'hard_estimates': {key: quota_summary(list(hard[key].values())) for key in keys},
        'component_only': {f: {key: quota_summary([own[key][f]]) for key in keys} for f in FAMILIES},
        'component_only_hard': {f: {key: quota_summary([own_hard[key][f]]) for key in keys} for f in FAMILIES},
        'CPU_seconds': {f: sum(p['CPU_seconds'][f] for p in populations) for f in FAMILIES},
        'total_sampler_CPU_seconds': total_cpu, 'populations': populations,
        'top_16': sorted(corrected, key=lambda r: r['log_importance_weight'], reverse=True)[:16],
        'analysis_wall_seconds': time.monotonic()-started, 'analyzer_sha256': sha(__file__),
        'analysis_CPU_seconds': time.process_time()-cpu_started,
        'vector_density_CPU_seconds': sum(p['vector_density_CPU_seconds'] for p in populations),
        'density_and_moments_sha256': sha(Path(__file__).with_name('shoulder_mis.py')),
        'scope': 'Fresh fixed-quota MIS on the declared region. Full unconditional density and all zeros retained. Component controls share their rows with MIS and are correlated with it. Weight ESS is descriptive; fixed-stratum variance is authoritative. No historical samples pooled, adaptive learning, MCMC mixing or global equilibrium claim.'}
    for family in FAMILIES:
        for estimate in result['component_only'][family].values():
            estimate['relative_variance_times_CPU'] = (None if estimate['stratified_RSE'] is None
                else estimate['stratified_RSE']**2*result['CPU_seconds'][family])
    result['cost_scope'] = ('Sampler costs are generation only. Offline Python cross-density and validation CPU are reported separately; '
        'they are additional costs of this implementation. A lower statistical variance need not improve precision per CPU.')
    result['paired_differences'] = {baseline: {key: signed_quota_difference(
        difference_merged[baseline][key][baseline],
        difference_merged[baseline][key]['cover' if baseline == 'guide' else 'guide'],
        result['component_only'][baseline][key]['logQ']) for key in keys} for baseline in FAMILIES}
    out = Path(output).resolve() if output is not None else root/'assessment'
    require(not out.exists(), 'Use a fresh derived assessment; preserve earlier analysis')
    out.mkdir()
    with (out/'corrected-nonzero.jsonl').open('w') as handle:
        for row in corrected:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    result['corrected_nonzero_sha256'] = sha(out/'corrected-nonzero.jsonl')
    write(out/'analysis.json', result)
    summary = estimates['all']
    lines = ['# Fixed-allocation contact-region MIS', '',
        f"{master['total_unconditional_draws']:,} unconditional draws; quotas {allocation['guide']} guide + {allocation['cover']} cover per population.", '',
        f"log Q={summary['logQ']}; stratified RSE={summary['stratified_RSE']}; independent population RSE={summary['independent_population_RSE']}.", '',
        f"Sampler CPU {total_cpu:.2f} s. All rows, source/cross densities, original q, cloud numerators and frozen hashes audited.", '', result['scope'], '']
    (out/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Fresh derived output directory; raw campaign stays unchanged')
    args = parser.parse_args()
    analyze(args.root, args.out)
