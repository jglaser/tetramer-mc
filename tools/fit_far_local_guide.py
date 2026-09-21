#!/usr/bin/env python3
"""Fit a far-contact proposal from an independently audited finite neighborhood."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import logsumexp

from analyze_far_peak_reference import PHYSICAL, WINDOW, audit_selection
from analyze_peak_neighborhood import geometry_model_audit
from audit_shoulder_mis_independently import near
from compare_intermediate_local_reference import read_batches
from fit_peak_reference_proposal import population_diagnostics, serial
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import (coordinates, weighted_fit, geometric_floor,
    model_from_fit, chart_log_density, laboratory_model)
from prepare_smc_normalizer_atlas import Density, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'runs/ab-far-peak-reference-preparation-20260920'
AUDIT = ROOT/'runs/ab-far-peak-reference-assessment-20260920/analysis.json'


def fit(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh fit output required')
    started = time.process_time(); protocol = read(PREPARATION/'protocol.json')
    for name, digest in read(PREPARATION/'freeze.json').items():
        require(sha(PREPARATION/name) == digest, 'Frozen geometry changed: '+name)
    for name, digest in protocol['archived_sha256'].items():
        require(sha(PREPARATION/'provenance'/name) == digest, 'Frozen input changed: '+name)
    audited = read(AUDIT); require(audited['complete'] and audited['original_q_window'] == WINDOW, 'Completed far audit required')
    for path, digest in audited['input_sha256'].items(): require(sha(path) == digest, 'Local audit input changed')
    for name, digest in audited['archived_sha256'].items(): require(sha(AUDIT.parent/'provenance'/name) == digest, 'Local audit source changed')
    record = next(c for c in audited['campaigns'] if c['radius_A'] == .5)
    declared = next(c for c in protocol['campaigns'] if c['radius_A'] == .5)
    reference = Path(record['root']); require(reference == Path(declared['output']), 'Wrong fit campaign')
    require(sha(reference/'manifest.json') == record['manifest_sha256'], 'Reference manifest changed')
    master = read(reference/'manifest.json'); region = read(reference/'provenance/region.json')
    cfg = read(PREPARATION/'config.json'); source_cfg = read(reference/'provenance/config.json')
    model = read(PREPARATION/'model.json')
    require(all(cfg[k] == source_cfg[k] == protocol['physical'][k] for k in PHYSICAL), 'Different AB physical target')
    require(sha(reference/'provenance/shape.json') == model['shape_sha256'], 'Different protein shape')
    require(region['gaussian_chart'] == model and region['mahalanobis_radius'] == .5 and region['minimum_mahalanobis_radius'] == 0., 'Different local support')
    require(master['region_sha256'] == declared['region_sha256'] == sha(reference/'provenance/region.json'), 'Different far region')
    require(master['archive_sha256']['latent-region-normalizer'] == protocol['executable_sha256'], 'Executable changed')
    require(master['lambda_ratio'] == 64. and master['cloud_replicates'] == 2, 'Cloud allocation changed')
    selected, selection = audit_selection(PREPARATION, protocol)
    geometry, _ = geometry_model_audit(model, region, cfg, selected['pose'])
    require(len(master['jobs']) == 4 and [j['seed'] for j in master['jobs']] == declared['seeds'], 'Wrong source populations')
    rows = []; labels = []; populations = []; hashes = {}; draws = 0
    for job in master['jobs']:
        require(job['samples'] == declared['samples_per_population'], 'Source budget changed')
        path = Path(job['directory'])/'samples.jsonl'; digest = hashlib.sha256(); count = 0; logs = []
        for lines, batch in read_batches(path):
            for line in lines: digest.update(line)
            for row in batch:
                require(row['draw'] == count, 'Source row order changed'); count += 1
                if row['log_importance_weight'] is None: continue
                require(row['hard_valid'] and row['capture_valid'] and row['region_valid'] and 5 <= row['q'] < 37, 'Invalid fitting point')
                rows.append(row); labels.append(job['id']); logs.append(row['log_importance_weight'])
        require(count == job['samples'] and logs, 'Truncated or empty source population')
        require(digest.hexdigest() == record['sample_sha256'][str(path)], 'Physical rows changed since audit')
        hashes[str(path)] = digest.hexdigest(); draws += count
        populations.append(dict(id=job['id'], seed=job['seed'], unconditional_draws=count, positive_rows=len(logs),
            source_logQ=float(logsumexp(logs)-math.log(count)), samples_sha256=digest.hexdigest()))
    source = record['physical']['full']
    require(draws == source['draws'] and len(rows) == source['nonzero'], 'Source normalization counts differ')
    logw = np.array([r['log_importance_weight'] for r in rows]); logq = float(logsumexp(logw)-math.log(draws))
    near(logq, source['logQ'])
    poses = [r['pose'] for r in rows]; relative = relative_poses(poses, cfg['fixed_poses'][0])
    ell = model['angular_length']; anchor = model['anchors'][0]; x = coordinates(relative, anchor, ell)
    floor = geometric_floor(ell); fitted = weighted_fit(x, logw, floor)
    near(fitted['normalized_weight_ESS'], source['weight_ESS']); near(fitted['largest_normalized_weight'], source['maximum_fraction'])
    models = {'weighted': model_from_fit(fitted, anchor, model['shape_sha256'], 'single', ell)}
    for sd in (.2, .4):
        m = copy.deepcopy(model); m['covariances'] = (np.asarray(model['covariances'])*sd**2).tolist()
        models[f'geometry_sd_{sd}'] = m
    densities = {name: Density(m).evaluate(relative)[0] for name, m in models.items()}
    independent = chart_log_density(x, fitted, 'single', ell)
    errors = dict(chart=near(independent, densities['weighted']),
        laboratory=near(independent, Density(laboratory_model(models['weighted'], cfg['fixed_poses'][0])).evaluate(poses)[0]))
    robustness = population_diagnostics(x, logw, labels, floor, ell, np.asarray(model['covariances'][0]), fitted, densities)
    inverse = np.linalg.inv(np.linalg.cholesky(model['covariances'][0])); weights = np.exp(logw-logsumexp(logw))
    inputs = {'source-audit.json': AUDIT, 'source-config.json': reference/'provenance/config.json',
        'source-region.json': reference/'provenance/region.json', 'shape.json': reference/'provenance/shape.json',
        'source-manifest.json': reference/'manifest.json', 'peak-model.json': PREPARATION/'model.json',
        'peak-protocol.json': PREPARATION/'protocol.json', 'peak-freeze.json': PREPARATION/'freeze.json',
        'selected-pose.json': PREPARATION/'selected-pose.json', **local_dependencies([Path(__file__)])}
    hashes.update({str(p): sha(p) for p in inputs.values()})
    (out/'provenance').mkdir(parents=True)
    for name, path in inputs.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    for name, m in models.items(): write(out/f'model-{name}.json', m)
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(), no_new_geometry_or_physical_draws=True,
        physical={k: cfg[k] for k in PHYSICAL}, original_q_window=WINDOW, source_ball_radius_A=.5,
        source=str(reference), original_source_audit=record, selected_source_audit=selection, chart_geometry=geometry,
        source_full_unconditional_N=draws, source_valid_fit_rows=len(rows), source_populations=populations,
        original_logQ_reconstruction=logq, shape_sha256=model['shape_sha256'], angular_length=ell,
        fit_rule='Original importance weights pooled once across all equal-budget populations, with no clipping or equal-population renormalization; weighted scatter plus additive floor.',
        floor_rule='diag(.05^2 I3, [ell*tan(.1 degree/2)]^2 I3)', floor=floor, fit=fitted,
        geometry_scaled_mean=inverse@fitted['mean'], geometry_scaled_covariance=inverse@fitted['covariance']@inverse.T,
        geometry_scaled_covariance_eigenvalues=eigvalsh(fitted['covariance'], np.asarray(model['covariances'][0])),
        pooled_conditional_expected_log_proposal_density={k: float(weights@d) for k,d in densities.items()},
        population_robustness=robustness, independent_density_errors=errors,
        predictive_scope='Held-out weighted log g is conditional on this truncated region. The Gaussian g is normalized globally. Folds have overlapping training sets; scores are ratio diagnostics, not unbiased likelihood estimates.',
        scope='Only source R<=.5 moments; not an entire basin, assembly result or improved physical normalizer. Future complete-support proposal validation needs independent populations after freezing.',
        input_sha256=hashes, archived_sha256={name:sha(out/'provenance'/name) for name in inputs},
        output_model_sha256={name:sha(out/f'model-{name}.json') for name in models}, fitting_CPU_seconds=time.process_time()-started)
    for path,digest in hashes.items(): require(sha(path) == digest, 'Input changed during fit')
    write(out/'report.json', serial(report))
    write(out/'freeze.json', dict(report_sha256=sha(out/'report.json'), model_sha256=report['output_model_sha256'],
        archived_sha256=report['archived_sha256'], source_sample_sha256=record['sample_sha256']))
    print(dict(output=str(out), source_ESS=fitted['normalized_weight_ESS'], maximum_fraction=fitted['largest_normalized_weight']))
    for p in robustness:
        print(p['population'], p['leave_one_population_out_mean_displacement_geometry_radius'],
            p['leave_one_population_out_covariance_eigenvalues_relative_to_pooled'][[0,-1]], p['leave_one_population_out_gain_vs_geometry'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    fit(p.parse_args().out)
