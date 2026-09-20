#!/usr/bin/env python3
"""Compare independent latent campaigns on exactly the same original-q band."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp

from analyze_native_region_reference import native_q
from analyze_shoulder_guides import statistics
from prepare_cayley_rms_cover import derive_model, read, require, sha, write

PHYSICAL_KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def population_rse(populations, key):
    logs = [p[key]['logQ'] for p in populations]
    finite = [x for x in logs if x is not None]
    if len(logs) < 2 or not finite:
        return None
    offset = max(finite)
    means = np.asarray([math.exp(x-offset) if x is not None else 0. for x in logs])
    return float(means.std(ddof=1)/math.sqrt(len(means))/means.mean())


def load_latent(campaign, lower, upper):
    campaign = Path(campaign).resolve()
    manifest = read(campaign/'manifest.json'); analysis = read(campaign/'assessment/analysis.json')
    for name, digest in manifest['archive_sha256'].items():
        require(sha(campaign/'provenance'/name) == digest, f'Archived campaign input changed: {name}')
    require(len({job['samples'] for job in manifest['jobs']}) == 1,
            'Unweighted population RSE requires equal unconditional population budgets')
    cfg = read(campaign/'provenance/config.json'); region = read(campaign/'provenance/region.json')
    require(analysis['region_sha256'] == manifest['region_sha256'] == sha(campaign/'provenance/region.json'), 'Region hash mismatch')
    require(region['minimum_original_q'] <= lower < upper <= region['maximum_original_q'], 'Requested band is outside this reference')
    require(region['physical_metric'] == cfg['metadata'] and region['physical_fixed_neighbors'] == cfg['fixed_poses'], 'Physical definition mismatch')
    shape_hash = sha(campaign/'provenance/shape.json')
    require(shape_hash == region['shape_sha256'] == manifest['archive_sha256']['shape.json'], 'Physical shape changed')
    require(analysis['original_q_window']['minimum'] == region['minimum_original_q'] and
            analysis['original_q_window']['maximum'] == region['maximum_original_q'], 'Audited q window differs')
    expected_hashes = {p['id']: p['samples_sha256'] for p in analysis['populations']}
    full_physical, full_hard, populations, sample_hashes, top = [], [], [], {}, []
    maximum_q_error = 0.; n = 0
    for job in manifest['jobs']:
        directory = Path(job['directory']); summary = read(directory/'summary.json')
        require(summary['complete'], 'Incomplete population')
        logs, hard_logs = [], []; digest = hashlib.sha256(); count = 0
        with (directory/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                require(row['draw'] == count, 'Missing or repeated draw'); count += 1
                if not row.get('region_valid', False) or not row.get('capture_valid', False) or not row.get('hard_valid', False):
                    continue
                if not lower < row['q'] < upper:
                    continue
                q = native_q(cfg['metadata'], row['pose'])
                maximum_q_error = max(maximum_q_error, abs(q-row['q']))
                require(abs(q-row['q']) < 2e-8 and lower < q < upper, 'Original-q band mismatch')
                log_j = row['log_physical_jacobian']
                hard = analysis['log_latent_volume']+log_j
                require(abs(hard-row['log_hard_weight']) < 2e-10, 'Hard-volume weight mismatch')
                mean = float(logsumexp([c['log_weight'] for c in row['clouds']]))-math.log(len(row['clouds']))
                require(abs(hard+mean-row['log_importance_weight']) < 2e-10, 'Physical importance weight mismatch')
                logs.append(row['log_importance_weight']); hard_logs.append(hard)
                top.append(dict(row, population=job['id'], seed=job['seed']))
        require(count == job['samples'] == summary['samples'], 'Fixed unconditional denominator mismatch')
        require(digest.hexdigest() == expected_hashes[job['id']] == summary['samples_sha256'], 'Sample hash mismatch')
        sample_hashes[str(directory/'samples.jsonl')] = digest.hexdigest()
        full_physical.extend(logs); full_hard.extend(hard_logs); n += count
        populations.append({'id': job['id'], 'seed': job['seed'], 'physical': statistics(logs, count),
                            'hard': statistics(hard_logs, count)})
    require(n == analysis['estimate']['draws'] == analysis['independently_reconstructed_poses'], 'Incomplete independent pose audit')
    physical, hard = statistics(full_physical, n), statistics(full_hard, n)
    cpu = analysis['sampler_cpu_seconds']
    return {'root': str(campaign), 'physical_signature': {k: cfg[k] for k in PHYSICAL_KEYS},
            'shape_sha256': shape_hash,
            'cloud_law': {'lambda_ratio': manifest['lambda_ratio'], 'cloud_replicates': manifest['cloud_replicates'],
                          'endpoint_gate': cfg.get('endpoint_gate')},
            'integration_window': analysis['original_q_window'], 'comparison_window': [lower, upper],
            'physical': physical, 'hard': hard, 'populations': populations,
            'physical_population_RSE': population_rse(populations, 'physical'),
            'hard_population_RSE': population_rse(populations, 'hard'),
            'CPU_seconds': cpu, 'physical_ESS_per_total_CPU': physical['ESS']/cpu,
            'hard_ESS_per_total_CPU': hard['ESS']/cpu, 'maximum_original_q_error': maximum_q_error,
            'sample_sha256': sample_hashes,
            'top_16': sorted(top, key=lambda row: row['log_importance_weight'], reverse=True)[:16],
            'input_sha256': {str(campaign/name): sha(campaign/name) for name in
                            ('manifest.json', 'assessment/analysis.json', 'provenance/config.json', 'provenance/region.json')}}


def compare(reference, focused, output, lower=1., upper=1.1, guide_comparisons=()):
    output = Path(output).resolve(); require(not output.exists(), 'Use a fresh comparison directory')
    require(math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper, 'Invalid original-q comparison band')
    old = load_latent(reference, lower, upper); new = load_latent(focused, lower, upper)
    require(old['physical_signature'] == new['physical_signature'], 'Different physical targets')
    require(old['shape_sha256'] == new['shape_sha256'], 'Different physical protein shapes')
    old_seeds = {p['seed'] for p in old['populations']}; new_seeds = {p['seed'] for p in new['populations']}
    require(len(old_seeds) == len(old['populations']) and len(new_seeds) == len(new['populations']) and not old_seeds & new_seeds,
            'Comparison requires independent populations')
    cfg = read(Path(reference)/'provenance/config.json')
    old_region = read(Path(reference)/'provenance/region.json'); new_region = read(Path(focused)/'provenance/region.json')
    old_b = old_region['maximum_original_q']; new_b = new_region['maximum_original_q']
    proofs = []
    for region in (old_region, new_region):
        observed = region['gaussian_chart']
        expected, proof = derive_model(cfg['metadata'], region['fixed_neighbor'], region['shape_sha256'],
                                        region['maximum_original_q'], observed['angular_length'])
        require(proof['mahalanobis_radius'] == region['mahalanobis_radius'], 'Reference is not the claimed complete geometric cover')
        for key in ('means', 'covariances', 'weights'):
            require(np.max(np.abs(np.asarray(expected[key])-np.asarray(observed[key]))) < 2e-11,
                    'Exact density-ratio formula requires the geometrically derived chart')
        for key in ('position', 'rotation'):
            require(np.max(np.abs(np.asarray(expected['anchors'][0][key])-np.asarray(observed['anchors'][0][key]))) < 2e-11,
                    'Geometric chart reference differs')
        proofs.append(proof)
    old_proof, new_proof = proofs
    density_gain = (old_b/new_b)**6*(new_proof['cosine_square_factor']/old_proof['cosine_square_factor'])**1.5
    differences = {}
    for key in ('physical', 'hard'):
        a, b = old[key], new[key]
        if a['logQ'] is None or b['logQ'] is None:
            differences[key] = {'log_difference': None, 'ratio': None}
            continue
        ratio = math.exp(b['logQ']-a['logQ'])
        observed_se_in_old_units = math.hypot(a['observed_RSE'], ratio*b['observed_RSE'])
        differences[key] = {'log_difference': b['logQ']-a['logQ'], 'ratio': ratio,
                            'linear_difference_in_combined_observed_standard_errors': (ratio-1)/observed_se_in_old_units,
                            'scope': 'Observed error diagnostic; weight concentration can invalidate nominal significance.'}
    guide_rows = []
    for path in guide_comparisons:
        path = Path(path); cached = read(path)
        require('campaigns' in cached and isinstance(cached['bands'], list),
                'Use the combined matching-band guide comparison, with physical definitions and a band table')
        require(cached['physical'] == old['physical_signature'], 'Cached guide uses a different physical target')
        band = next((i for i, pair in enumerate(cached['bands']) if list(pair) == [lower, upper]), None)
        require(band is not None, 'Cached guide has no matching q band')
        for name, campaign in cached['campaigns'].items():
            if name == 'cayley': continue
            if campaign.get('shape_sha256'):
                require(campaign['shape_sha256'] == old['shape_sha256'], 'Cached guide uses a different physical shape')
            physical = campaign['bands'][str(band)]
            row = {'label': name, 'source': str(path.resolve()), 'source_sha256': sha(path),
                   'physical': physical, 'CPU_seconds': campaign['CPU_seconds'],
                   'physical_ESS_per_total_CPU': physical['ESS']/campaign['CPU_seconds'],
                   'scope': 'Matching-band cached guide estimate; original dedicated campaign audit remains authoritative.'}
            if 'hard_bands' in campaign: row['hard'] = campaign['hard_bands'][str(band)]
            guide_rows.append(row)
    (output/'provenance').mkdir(parents=True)
    shutil.copy2(__file__, output/'provenance/compare_cayley_qwindow.py')
    result = {'complete': True, 'q_window': {'minimum': lower, 'maximum': upper, 'lower_inclusive': False, 'upper_inclusive': False},
              'reference_band': old, 'focused_band': new, 'matching_guide_estimates': guide_rows,
              'differences': differences, 'exact_shared_band_proposal_density_gain': density_gain,
              'identical_recorded_cloud_law': old['cloud_law'] == new['cloud_law'],
              'observed_physical_ESS_gain': new['physical']['ESS']/old['physical']['ESS'],
              'observed_physical_ESS_per_CPU_gain': new['physical_ESS_per_total_CPU']/old['physical_ESS_per_total_CPU'],
              'scope': 'Same physical q band and full unconditional denominators. Reference CPU includes its wider-window work. Proposal-density gain is exact; observed importance ESS and means do not certify unseen-tail convergence.'}
    write(output/'comparison.json', result)
    lines = ['# Matching original-q band comparison', '', f'Only `{lower}<q<{upper}` contributes to every estimate below.', '',
             '| Sampling campaign | Draws | Valid in band | log Q | Physical ESS | Row / population RSE | Total CPU s |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, item in [('Wider reference, selected band', old), ('Focused reference, same band', new)]:
        s = item['physical']
        lines.append(f"| {label} | {s['draws']} | {s['nonzero']} | {s['logQ']:.6f} | {s['ESS']:.2f} | {s['observed_RSE']:.1%} / {item['physical_population_RSE']:.1%} | {item['CPU_seconds']:.2f} |")
    lines += ['', f'The focused proposal has exactly {density_gain:.8f} times the physical density throughout this band.', '',
              'The hard and physical mean comparisons and matching guide estimates are in comparison.json. The wider run retains its full CPU cost and all unconditional zero draws. Observed ESS is importance-weight ESS, not Markov-chain contact mixing.', '']
    (output/'report.md').write_text('\n'.join(lines))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True); p.add_argument('--focused', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--q-min', type=float, default=1.)
    p.add_argument('--q-max', type=float, default=1.1); p.add_argument('--guide-comparison', type=Path, action='append', default=[])
    a = p.parse_args(); result = compare(a.reference, a.focused, a.out, a.q_min, a.q_max, a.guide_comparison)
    print(json.dumps({'complete': True, 'differences': result['differences'],
                      'exact_shared_band_proposal_density_gain': result['exact_shared_band_proposal_density_gain'],
                      'observed_physical_ESS_gain': result['observed_physical_ESS_gain']}, indent=2))


if __name__ == '__main__': main()
