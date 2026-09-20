#!/usr/bin/env python3
"""Retrospective variance/cost forecast for fixed-quota shoulder MIS.

This uses existing independent g- and c-pilot draws to estimate stratum
variances under prospective balance-heuristic denominators. It does not
report a new pooled physical normalizer, refit, or create physical samples.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp

from analyze_native_region_reference import GaussianGuide, native_q, proposal_log_density
from prepare_smc_normalizer_atlas import Density, ROOT, read, relative_poses, sha, write

RATIOS = (0, 1, 4, 16, 64)
KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')
WINDOW = {'minimum': 1., 'maximum': 2., 'lower_inclusive': False, 'upper_inclusive': False}


def zero_inclusive_variance(values, n):
    """Unbiased sample variance; omitted draws are exact target zeros."""
    values = np.asarray(values)
    assert n > 1 and len(values) <= n and np.isfinite(values).all()
    total = float(values.sum()); squares = float(values@values)
    return (squares-total*total/n)/(n-1)


def fixed_quota_cost_variance(vg, vc, ratio, cost_g, cost_c):
    """Var(Qhat)*CPU for fixed n_c/n_g and independent stratum samples."""
    assert ratio >= 0 and vg >= 0 and vc >= 0 and cost_g > 0 and cost_c > 0
    return (vg+ratio*vc)/(1+ratio)**2*(cost_g+ratio*cost_c)


def load_pilot(root, kind, physical, seen_seeds):
    manifest = read(root/'manifest.json'); cfg = read(root/'provenance/config.json')
    assert {k: cfg[k] for k in KEYS} == physical
    for name, digest in manifest['archive_sha256'].items():
        assert sha(root/'provenance'/name) == digest
    if kind == 'g':
        assessed = read(root/'assessment-streaming.json')
        assert assessed['complete'] and assessed['all_rows_and_hashes_validated'] and assessed['q_window'] == WINDOW
        by_name = {p['replicate']: p for p in assessed['populations']}
    else:
        assessed = read(root/'assessment/analysis.json')
        assert assessed['original_q_window'] == WINDOW
        by_name = {p['id']: p for p in assessed['populations']}
    rows = []; sources = []; counts = set(); total_cpu = 0.; total_n = 0
    for group, job in enumerate(manifest['jobs']):
        path = Path(job['output'] if kind == 'g' else job['directory'])
        summary = read(path/'summary.json'); run = read(path/'manifest.json')
        assert summary['complete'] and run['seed'] == job['seed'] and run['seed'] not in seen_seeds
        seen_seeds.add(run['seed'])
        assert run['config_sha256'] == sha(root/'provenance/config.json')
        assert run['shape_sha256'] == sha(root/'provenance/shape.json')
        assert run['lambda_ratio'] == 64 and run['cloud_replicates'] == 2 and run['activity'] == .035
        digest = hashlib.sha256(); n = 0; positive = 0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                assert row['draw'] == n; n += 1
                valid = 'zero' not in row if kind == 'g' else row['log_importance_weight'] is not None
                if not valid: continue
                q = native_q(cfg['metadata'], row['pose'])
                assert 1 < q < 2 and abs(q-row['q']) < 2e-8
                clouds = row['cloud_log_weights'] if kind == 'g' else [c['log_weight'] for c in row['clouds']]
                logf = float(logsumexp(clouds)-math.log(2))
                assert abs(logf+row['log_hard_weight']-row['log_importance_weight']) < 2e-9
                rows.append({'pose': row['pose'], 'q': q, 'source_population': group, 'draw': row['draw'],
                    'log_source_density': -row['log_hard_weight'], 'log_f_estimator': logf,
                    'log_original_importance_weight': row['log_importance_weight']})
                positive += 1
        assert n == summary['samples'] == run['samples']
        assert digest.hexdigest() == summary['samples_sha256']
        reference = by_name[path.name if kind == 'g' else job['id']]
        assert digest.hexdigest() == reference['sample_sha256' if kind == 'g' else 'samples_sha256']
        cpu = summary['cpu_seconds'] if kind == 'g' else summary['sampler_cpu_seconds']
        total_n += n; total_cpu += cpu; counts.add(n)
        sources.append({'population': group, 'seed': run['seed'], 'samples': n, 'valid': positive,
                        'CPU_seconds': cpu, 'samples_path': str(path/'samples.jsonl'), 'samples_sha256': digest.hexdigest()})
    assert len(sources) == 4 and len(counts) == 1
    return {'rows': rows, 'populations': sources, 'samples': total_n, 'CPU_seconds': total_cpu,
            'CPU_per_unconditional_draw': total_cpu/total_n, 'assessment': assessed,
            'root': str(root), 'config_sha256': sha(root/'provenance/config.json')}


def cross_densities(pilot, cfg, guide_assessment, model, region, kind):
    poses = [row['pose'] for row in pilot['rows']]
    relative = relative_poses(poses, cfg['fixed_poses'][0])
    gaussian = Density(model).evaluate(relative)[0]
    cover_logs = np.array([proposal_log_density(guide_assessment['cover_mixture'], p)[0] for p in poses])
    assert np.isfinite(cover_logs).all(), 'Every target-valid pose must lie in the complete product cover'
    cube_logs = np.array([-3*math.log(2*cfg['capture_radius']) if all(abs(p['position'][i]-cfg['capture_center'][i]) <= cfg['capture_radius'] for i in range(3)) else -math.inf for p in poses])
    description = guide_assessment['guide']; beta = description['weight']; epsilon = description['uniform_probability']
    inner = np.logaddexp(math.log1p(-epsilon)+gaussian, math.log(epsilon)+cube_logs)
    g = np.logaddexp(math.log1p(-beta)+cover_logs, math.log(beta)+inner)
    chart = Density(region['gaussian_chart'])
    log_chart, norms, _ = chart.evaluate(relative_poses(poses, region['fixed_neighbor']))
    radius = region['mahalanobis_radius']; assert np.all(norms[:, 0] <= radius*(1+2e-12))
    assert region.get('minimum_mahalanobis_radius', 0.) == 0.
    log_volume = 3*math.log(math.pi)+6*math.log(radius)-math.log(6)
    logj = -3*math.log(2*math.pi)-.5*norms[:, 0]**2-log_chart
    c = -log_volume-logj
    expected = np.array([row['log_source_density'] for row in pilot['rows']])
    source_error = float(np.max(np.abs((g if kind == 'g' else c)-expected)))
    assert source_error < 2e-7
    # Independent scalar guide algebra on a fixed spread plus all largest
    # original weights, including the known rare contacts when present.
    scalar = GaussianGuide(description, model, cfg, model['shape_sha256'])
    chosen = np.unique(np.concatenate((np.linspace(0, len(poses)-1, 64, dtype=int),
                                      np.argsort([r['log_original_importance_weight'] for r in pilot['rows']])[-16:])))
    scalar_error = 0.
    for i in chosen:
        value, _, _ = scalar.log_density(poses[i])
        expected_g = float(np.logaddexp(math.log1p(-beta)+cover_logs[i], math.log(beta)+value))
        scalar_error = max(scalar_error, abs(expected_g-g[i]))
    assert scalar_error < 2e-7
    return g, c, {'source_density_max_error': source_error, 'scalar_guide_max_error': scalar_error,
                  'scalar_checks': len(chosen), 'maximum_cayley_cover_radius': float(norms[:, 0].max()),
                  'valid_rows_cross_evaluated': len(poses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    assert not out.exists(), 'Use a fresh diagnostic output directory'
    guide_root = ROOT/'runs/ab-shoulder-mixture-confirmation-4x65536-l64-20260920'
    cover_root = ROOT/'runs/ab-shoulder-cayley-reference-4x262144-l64-20260920'
    cfg = read(guide_root/'provenance/config.json'); physical = {k: cfg[k] for k in KEYS}
    model = read(guide_root/'provenance/guide-model.json'); region = read(cover_root/'provenance/region.json')
    assert region['physical_fixed_neighbors'] == cfg['fixed_poses'] and region['physical_metric'] == cfg['metadata']
    assert region['shape_sha256'] == model['shape_sha256'] == sha(guide_root/'provenance/shape.json') == sha(cover_root/'provenance/shape.json')
    seeds = set(); pilots = {kind: load_pilot(root, kind, physical, seeds) for kind, root in [('g', guide_root), ('c', cover_root)]}
    assessment = pilots['g']['assessment']; reference_logQ = assessment['regions']['region']['log_normalizer']
    assert assessment['guide']['weight'] == .75 and assessment['guide']['uniform_probability'] == .05
    densities = {}; audits = {}
    for kind, pilot in pilots.items():
        g, c, audit = cross_densities(pilot, cfg, assessment, model, region, kind)
        densities[kind] = (g, c); audits[kind] = audit
    costs = {kind: pilot['CPU_per_unconditional_draw'] for kind, pilot in pilots.items()}
    reports = []; baseline = None
    for ratio in RATIOS:
        alpha = 1/(1+ratio); variances = {}; per_population = {}; leave_one_out = {}
        for kind, pilot in pilots.items():
            g, c = densities[kind]
            logm = g if ratio == 0 else np.logaddexp(math.log(alpha)+g, math.log1p(-alpha)+c)
            logf = np.array([row['log_f_estimator'] for row in pilot['rows']])
            values = np.exp(logf-logm-reference_logQ)
            groups = np.array([row['source_population'] for row in pilot['rows']])
            variances[kind] = zero_inclusive_variance(values, pilot['samples'])
            per_population[kind] = [zero_inclusive_variance(values[groups == p['population']], p['samples']) for p in pilot['populations']]
            leave_one_out[kind] = [zero_inclusive_variance(values[groups != p['population']], pilot['samples']-p['samples']) for p in pilot['populations']]
        objective = fixed_quota_cost_variance(variances['g'], variances['c'], ratio, costs['g'], costs['c'])
        if baseline is None:
            baseline = objective
            implied_rse = math.sqrt(variances['g']/pilots['g']['samples'])
            assert abs(implied_rse-assessment['regions']['region']['relative_SE']) < 2e-9
        leave_costs = [fixed_quota_cost_variance(a, b, ratio, costs['g'], costs['c'])
                      for a in leave_one_out['g'] for b in leave_one_out['c']]
        future = []
        for ng_per_population in (4096, 16384):
            ng = 4*ng_per_population; nc = ratio*ng
            future.append({'populations': 4, 'guide_draws_per_population': ng_per_population,
                'cover_draws_per_population': ratio*ng_per_population,
                'CPU_seconds': ng*costs['g']+nc*costs['c'],
                'nominal_RSE_on_existing_guide_mean_scale': math.sqrt((variances['g']+ratio*variances['c'])/(ng*(1+ratio)**2)),
                'scope': 'Plug-in pilot forecast, not a confidence interval or bound on unseen weights'})
        reports.append({'cover_to_guide_quota_ratio': ratio, 'guide_fraction': alpha,
            'stratum_variances_on_existing_guide_mean_scale': variances,
            'variance_times_CPU_on_same_scale': objective,
            'relative_efficiency_vs_guide_only': baseline/objective,
            'cost_ratio_at_fixed_guide_draws': (costs['g']+ratio*costs['c'])/costs['g'],
            'leave_one_population_out_cost_variance_range': [min(leave_costs), max(leave_costs)],
            'stratum_population_variances': per_population, 'future_fixed_budget_forecasts': future})
    best = min(reports, key=lambda r: r['variance_times_CPU_on_same_scale'])
    out.mkdir(parents=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'physical': physical, 'q_window': WINDOW,
        'g_law': '0.25 complete product cover + 0.75*(0.95 frozen two-component Gaussian mixture + 0.05 cube/Haar)',
        'c_law': 'Uniform latent Cayley RMS ball, exact physical Jacobian, unchanged original1<q<2 target mask',
        'candidate_quota_ratios': RATIOS, 'density_audits': audits,
        'source_pilots': {k: {f: p[f] for f in ('root', 'populations', 'samples', 'CPU_seconds', 'CPU_per_unconditional_draw', 'config_sha256')} for k, p in pilots.items()},
        'model_sha256': sha(guide_root/'provenance/guide-model.json'), 'region_sha256': sha(cover_root/'provenance/region.json'),
        'existing_guide_logQ_used_only_as_common_variance_scale': reference_logQ,
        'formula': 'm=(n_g*g+n_c*c)/(n_g+n_c); V_k=Var_k(Wbar/m), zeros included; Var(Qhat)=(n_g V_g+n_c V_c)/(n_g+n_c)^2. Optimize Var(Qhat)*CPU.',
        'allocations': reports, 'lowest_observed_cost_variance_ratio': best['cover_to_guide_quota_ratio'],
        'scope': 'Retrospective pilot allocation and cost diagnostic only. No new samples, fitting, pooled physical normalizer, replacement of original estimates, or convergence claim. Unknown high-weight tails can change the optimum. Leave-one-out ranges are dependent sensitivity diagnostics, not confidence intervals.',
        'analysis_sha256': sha(__file__)}
    write(out/'forecast.json', result)
    for file in ('forecast_shoulder_mis.py', 'analyze_native_region_reference.py', 'prepare_smc_normalizer_atlas.py'):
        shutil.copy2(ROOT/'tools'/file, out/file)
    print(json.dumps({'best_observed_ratio': best['cover_to_guide_quota_ratio'], 'costs': costs,
        'allocations': [{k: r[k] for k in ('cover_to_guide_quota_ratio', 'relative_efficiency_vs_guide_only', 'cost_ratio_at_fixed_guide_draws', 'future_fixed_budget_forecasts')} for r in reports],
        'density_audits': audits}, indent=2))


if __name__ == '__main__':
    main()
