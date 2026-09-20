#!/usr/bin/env python3
"""Independent full-denominator comparison of frozen shoulder-guide campaigns."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analyze_native_region_reference import native_q
from prepare_smc_normalizer_atlas import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((1., 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2.))


def statistics(logs, n):
    if not logs:
        return {'draws': n, 'nonzero': 0, 'logQ': None, 'ESS': 0., 'observed_RSE': None,
                'maximum_fraction': None, 'coverage': 'No observations are not a mass upper bound'}
    x = np.asarray(logs); total = float(logsumexp(x)); square = float(logsumexp(2*x))
    ess = math.exp(2*total-square)
    return {'draws': n, 'nonzero': len(logs), 'logQ': total-math.log(n), 'ESS': ess,
            'observed_RSE': math.sqrt(max(0., (n/ess-1)/(n-1))),
            'maximum_fraction': math.exp(float(x.max())-total),
            'coverage': 'Observed rows only; unseen importance mass can invalidate apparent precision'}


def label(row):
    if row['proposal_family'] == 'guide':
        return f"guide/{row['guide_branch']}/{row['guide_component']}"
    return f"{row['proposal_family']}/{row['proposal_component']}"


def compare(root, out):
    assert not out.exists(), 'Preserve earlier analyses with a fresh output directory'
    protocol = read(root/'protocol.json')
    source = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
    training = read(source/'provenance/input-shoulder-poses.json')
    training_poses = {json.dumps(row['pose'], sort_keys=True) for row in training}
    training_seeds = {row['seed'] for row in training}
    physical_cfg = read(source/'config.json')
    keys = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')
    physical = {k: physical_cfg[k] for k in keys}
    signature = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
    report = {'physical': physical, 'physical_signature_sha256': signature,
              'protocol_sha256': sha(root/'protocol.json'), 'bands': BANDS,
              'band_rule': '1<q<1.1; 1.1<=q<1.25; 1.25<=q<1.5; 1.5<=q<2. Boundaries have zero continuous measure.',
              'campaigns': {}, 'analysis_sha256': sha(__file__)}
    global_seeds = set(training_seeds)
    for job in protocol['jobs']:
        name = job['name']; campaign = Path(job['root'])
        manifest = read(campaign/'manifest.json'); assessment = read(campaign/'assessment-streaming.json')
        assert assessment['complete'] and assessment['all_rows_and_hashes_validated']
        assert assessment['independent_complete_cover_validated']
        assert manifest['q_window'] == protocol['q_window'] == assessment['q_window']
        config_path = Path(manifest['frozen_config_path']); cfg = read(config_path)
        assert sha(config_path) == manifest['config_sha256']
        assert {k: cfg[k] for k in keys} == physical
        assert sha(Path(manifest['frozen_shape_path'])) == manifest['shape_sha256'] == sha(source/'provenance/shape.json')
        for file, expected in manifest['archive_sha256'].items():
            assert sha(campaign/'provenance'/file) == expected, (name, file)
        audited = {p['replicate']: p for p in assessment['populations']}
        full, bands, components = [], defaultdict(list), defaultdict(list)
        selected, valid_selected = Counter(), Counter()
        populations, top, sample_hashes = [], [], {}
        cloud_pairs = []; n = 0; maximum_q_error = 0.
        for population in manifest['jobs']:
            path = Path(population['output']); summary = read(path/'summary.json'); run = read(path/'manifest.json')
            assert summary['complete'] and run['seed'] == population['seed']
            assert population['seed'] not in global_seeds; global_seeds.add(population['seed'])
            assert run['config_sha256'] == manifest['config_sha256']
            assert run['metric'] == {key: physical['metadata'][key] for key in
                                     ('angle_error_scale_deg', 'member_error_scale', 'native_poses', 'rigid_members')}
            assert run['q_window'] == protocol['q_window']
            local, local_bands, local_components = [], defaultdict(list), defaultdict(list)
            digest = hashlib.sha256(); count = 0
            with (path/'samples.jsonl').open('rb') as handle:
                for line in handle:
                    digest.update(line); row = json.loads(line)
                    assert row['draw'] == count; count += 1
                    component = label(row); selected[component] += 1
                    if 'zero' in row:
                        continue
                    assert json.dumps(row['pose'], sort_keys=True) not in training_poses
                    q = native_q(cfg['metadata'], row['pose'])
                    maximum_q_error = max(maximum_q_error, abs(q-row['q']))
                    assert 1 < q < 2 and abs(q-row['q']) < 2e-8
                    assert np.linalg.norm(np.asarray(row['pose']['position'])-cfg['capture_center']) <= cfg['capture_radius']
                    logw = row['log_importance_weight']; assert math.isfinite(logw)
                    assert abs(logw-row['log_boltzmann_mean']+row['log_proposal_density']) < 2e-10
                    assert abs(float(logsumexp(row['cloud_log_weights']))-math.log(2)-row['log_boltzmann_mean']) < 2e-10
                    index = next(i for i, (a, b) in enumerate(BANDS) if a <= q < b)
                    full.append(logw); local.append(logw); bands[index].append(logw); local_bands[index].append(logw)
                    components[component].append(logw); local_components[component].append(logw); valid_selected[component] += 1
                    cloud_pairs.append([w-row['log_proposal_density'] for w in row['cloud_log_weights']])
                    top.append(dict(row, population=path.name))
            assert count == run['samples'] == summary['samples']
            assert digest.hexdigest() == summary['samples_sha256'] == audited[path.name]['sample_sha256']
            sample_hashes[str(path/'samples.jsonl')] = digest.hexdigest(); n += count
            populations.append({'population': path.name, 'seed': population['seed'], 'total': statistics(local, count),
                                'bands': {str(i): statistics(local_bands[i], count) for i in range(len(BANDS))},
                                'components': {key: statistics(value, count) for key, value in local_components.items()}})
        assert n == assessment['samples'] == job['samples_per_population']*job['populations']
        total = statistics(full, n); wanted = assessment['regions']['region']
        assert abs(total['logQ']-wanted['log_normalizer']) < 2e-10
        assert abs(total['ESS']/wanted['ess']-1) < 2e-9
        totals = np.exp(np.array([p['total']['logQ'] for p in populations])-total['logQ'])
        population_rse = float(np.std(totals, ddof=1)/np.sqrt(len(totals))/totals.mean())
        assert abs(population_rse-assessment['independent_population_relative_SE']) < 2e-9
        band_stats = {str(i): statistics(bands[i], n) for i in range(len(BANDS))}
        for item in band_stats.values():
            item['fraction_of_observed_total'] = 0. if item['logQ'] is None else math.exp(item['logQ']-total['logQ'])
        component_stats = {}
        for key in selected:
            item = statistics(components[key], n)
            item.update(selected_draws=selected[key], valid_draws=valid_selected[key],
                        fraction_of_observed_total=0. if item['logQ'] is None else math.exp(item['logQ']-total['logQ']))
            component_stats[key] = item
        pair = np.asarray(cloud_pairs); offset = float(pair.max()); weights = np.exp(pair-offset)
        mean = weights.mean(axis=1); mean_full = float(mean.sum()/n)
        variance = (float(mean@mean)-n*mean_full**2)/(n-1)
        cloud_variance = float(np.sum((weights[:, 0]-weights[:, 1])**2)/(4*n))
        report['campaigns'][name] = {'root': str(campaign), 'total': total, 'bands': band_stats,
            'source_components': component_stats, 'populations': populations, 'population_RSE': population_rse,
            'CPU_seconds': assessment['cpu_seconds'], 'importance_ESS_per_CPU_second': total['ESS']/assessment['cpu_seconds'],
            'paired_cloud_variance_fraction': cloud_variance/variance,
            'maximum_original_q_error': maximum_q_error, 'sample_sha256': sample_hashes,
            'config_sha256': manifest['config_sha256'], 'shape_sha256': manifest['shape_sha256'],
            'model_sha256': assessment['guide']['model_sha256'] if assessment.get('guide') else None,
            'manifest_sha256': sha(campaign/'manifest.json'), 'assessment_sha256': sha(campaign/'assessment-streaming.json'),
            'top_16': sorted(top, key=lambda r: r['log_importance_weight'], reverse=True)[:16]}
    report['scope'] = ('New independent integration only; all 66 training poses excluded. Band and source-component estimates retain full unconditional denominators. Source-component contributions are sampling diagnostics, not physical basin probabilities. Observed errors and importance ESS are not proofs of convergence or MCMC contact decorrelation.')
    out.mkdir(parents=True)
    write(out/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-root', type=Path, default=ROOT/'runs/ab-shoulder-window-pilot-20260920')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); report = compare(args.protocol_root.resolve(), args.out.resolve())
    print(json.dumps({k: {f: v[f] for f in ('total', 'population_RSE', 'CPU_seconds', 'importance_ESS_per_CPU_second', 'bands', 'source_components')}
                      for k, v in report['campaigns'].items()}, indent=2))


if __name__ == '__main__':
    main()
