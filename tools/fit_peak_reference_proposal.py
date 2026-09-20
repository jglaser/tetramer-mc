#!/usr/bin/env python3
"""Fit and cross-check one conditional peak guide without new physical draws.

The fit uses every positive original importance weight, not equal-population
renormalization. Source zeros retain their original denominator when the
physical normalizer is reconstructed. The Gaussian itself has global support;
its learned moments describe only the source ball intersected with the
original physical masks.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
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

from analyze_peak_neighborhood import audit_peak_selection, geometry_model_audit
from audit_shoulder_mis_independently import near
from compare_intermediate_local_reference import load_reference, read_batches
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_intermediate_guides import PHYSICAL
from prepare_native_confirmation_atlas import (coordinates, weighted_fit, geometric_floor,
    model_from_fit, chart_log_density, laboratory_model)
from prepare_smc_normalizer_atlas import Density, relative_poses
from run_shoulder_mis_campaign import local_dependencies


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [serial(item) for item in value]
    return value


def population_diagnostics(x, logw, groups, floor, ell, geometric_covariance,
                           fit, comparison_densities):
    """Train on other populations; score each held-out population once."""
    groups = np.asarray(groups); labels = np.unique(groups)
    require(len(labels) >= 2, 'Cross-validation requires independent populations')
    require(len(x) == len(logw) == len(groups), 'Mismatched fitting rows')
    require(all(len(d) == len(x) for d in comparison_densities.values()), 'Mismatched density rows')
    inverse = np.linalg.inv(np.linalg.cholesky(geometric_covariance))
    total = float(logsumexp(logw)); reports = []
    for label in labels:
        held = groups == label; train = ~held
        one = weighted_fit(x[held], logw[held], floor)
        loo = weighted_fit(x[train], logw[train], floor)
        held_weight = np.exp(logw[held]-logsumexp(logw[held]))
        score = {name: float(held_weight@values[held]) for name, values in comparison_densities.items()}
        score['weighted_leave_one_population_out'] = float(held_weight@chart_log_density(x[held], loo, 'single', ell))
        reports.append(dict(population=str(label), heldout_nonzero_rows=int(held.sum()),
            training_nonzero_rows=int(train.sum()), heldout_weight_ESS=float(1/(held_weight@held_weight)),
            heldout_largest_normalized_weight=float(held_weight.max()),
            original_total_weight_fraction=float(math.exp(logsumexp(logw[held])-total)),
            population_fit=one, leave_one_population_out_fit=loo,
            population_mean_displacement_geometry_radius=float(np.linalg.norm(inverse@(one['mean']-fit['mean']))),
            population_covariance_eigenvalues_relative_to_pooled=eigvalsh(one['covariance'], fit['covariance']),
            leave_one_population_out_mean_displacement_geometry_radius=float(np.linalg.norm(inverse@(loo['mean']-fit['mean']))),
            leave_one_population_out_covariance_eigenvalues_relative_to_pooled=eigvalsh(loo['covariance'], fit['covariance']),
            conditional_expected_log_proposal_density=score,
            leave_one_population_out_gain_vs_geometry={name: score['weighted_leave_one_population_out']-value
                for name, value in score.items() if name.startswith('geometry_sd_')}))
    return reports


def fit_reference(reference, preparation, out):
    reference, preparation, out = (Path(p).resolve() for p in (reference, preparation, out))
    require(not out.exists(), 'Use a fresh fit directory')
    started = time.process_time(); protocol = read(preparation/'protocol.json')
    freeze = read(preparation/'freeze.json')
    for name, digest in freeze.items(): require(sha(preparation/name) == digest, f'Changed peak preparation: {name}')
    for name, digest in protocol['archived_sha256'].items():
        require(sha(preparation/'provenance'/name) == digest, f'Changed peak archive: {name}')
    model = read(preparation/'model.json'); cfg = read(preparation/'config.json')
    require(sha(preparation/'model.json') == protocol['model_sha256'], 'Changed geometric chart')
    selected = read(preparation/'selected-pose.json')
    require(sha(preparation/'selected-pose.json') == protocol['selected_pose_sha256'], 'Changed selected pose')
    selection = audit_peak_selection(protocol, preparation/'provenance/extremes.json', selected, cfg, model['shape_sha256'])
    audited, region, source_cfg = load_reference(reference, model)
    for key in PHYSICAL: require(cfg[key] == source_cfg[key], f'Source physical target changed: {key}')
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and
            cfg['capture_radius'] == 18. and len(cfg['fixed_poses']) == 2, 'Unexpected physical baseline')
    require(region['mahalanobis_radius'] == .5, 'Fit source must be the fixed radius-.5 ball')
    require(region['minimum_original_q'] == 2. and region['maximum_original_q'] == 5. and
            region['minimum_original_q_inclusive'] and not region['maximum_original_q_inclusive'], 'Changed original q window')
    geometry, _ = geometry_model_audit(model, region, cfg, selected['pose'])
    master = read(reference/'manifest.json')
    declared = [c for c in protocol['campaigns'] if Path(c['output']).resolve() == reference]
    require(len(declared) == 1, 'Source not declared in peak protocol')
    require(master['region_sha256'] == declared[0]['region_sha256'], 'Declared source region changed')
    require([j['seed'] for j in master['jobs']] == declared[0]['seeds'], 'Declared source random streams changed')
    require(all(j['samples'] == declared[0]['samples'] for j in master['jobs']), 'Declared source budgets changed')
    rows, labels, populations = [], [], []; draws = 0; seen = set()
    for job in master['jobs']:
        require(job['seed'] not in seen, 'Source random streams repeat'); seen.add(job['seed'])
        path = Path(job['directory'])/'samples.jsonl'; digest = hashlib.sha256(); count = nonzero = 0; logs = []
        for lines, batch in read_batches(path):
            for line in lines: digest.update(line)
            for row in batch:
                require(row['draw'] == count, 'Source row ordering changed'); count += 1
                if row['log_importance_weight'] is None: continue
                rows.append(row); labels.append(job['id']); logs.append(row['log_importance_weight']); nonzero += 1
        require(count == job['samples'], 'Source unconditional count changed')
        require(digest.hexdigest() == audited['sample_sha256'][str(path)], 'Source rows changed since all-row audit')
        require(nonzero > 0, 'Each held-out population needs positive source weights')
        populations.append(dict(population=job['id'], seed=job['seed'], unconditional_draws=count,
            nonzero_rows=nonzero, samples_path=str(path), samples_sha256=digest.hexdigest(),
            source_logQ=float(logsumexp(logs)-math.log(count))))
        draws += count
    require(len(populations) == 4 and len({p['unconditional_draws'] for p in populations}) == 1,
            'Require four equal-budget independent source populations')
    require(draws == audited['physical']['ball']['draws'] and len(rows) == audited['physical']['ball']['nonzero'], 'Source audit counts changed')
    logw = np.asarray([r['log_importance_weight'] for r in rows]); labels = np.asarray(labels)
    physical_logQ = float(logsumexp(logw)-math.log(draws)); near(physical_logQ, audited['physical']['ball']['logQ'])
    poses = [row['pose'] for row in rows]; relative = relative_poses(poses, cfg['fixed_poses'][0])
    ell = model['angular_length']; anchor = model['anchors'][0]; x = coordinates(relative, anchor, ell)
    floor = geometric_floor(ell); fit = weighted_fit(x, logw, floor)
    near(fit['normalized_weight_ESS'], audited['physical']['ball']['weight_ESS'])
    near(fit['largest_normalized_weight'], audited['physical']['ball']['maximum_fraction'])
    fitted = model_from_fit(fit, anchor, model['shape_sha256'], 'single', ell)
    models = {'weighted': fitted}
    for sd in (.2, .4):
        control = copy.deepcopy(model); control['covariances'] = (np.asarray(model['covariances'])*sd**2).tolist()
        models[f'geometry_sd_{sd}'] = control
    densities = {name: Density(m).evaluate(relative)[0] for name, m in models.items()}
    independent = chart_log_density(x, fit, 'single', ell)
    density_errors = dict(chart=near(independent, densities['weighted']),
                         laboratory=near(independent, Density(laboratory_model(fitted, cfg['fixed_poses'][0])).evaluate(poses)[0]))
    robustness = population_diagnostics(x, logw, labels, floor, ell, np.asarray(model['covariances'][0]), fit, densities)
    inverse = np.linalg.inv(np.linalg.cholesky(model['covariances'][0])); weights = np.exp(logw-logsumexp(logw))
    sources = {'source-config.json': reference/'provenance/config.json', 'source-region.json': reference/'provenance/region.json',
        'source-shape.json': reference/'provenance/shape.json', 'source-manifest.json': reference/'manifest.json',
        'peak-model.json': preparation/'model.json', 'peak-protocol.json': preparation/'protocol.json',
        'peak-freeze.json': preparation/'freeze.json', 'selected-pose.json': preparation/'selected-pose.json',
        'selection-source.json': preparation/'provenance/extremes.json', **local_dependencies([Path(__file__)])}
    source_hashes = {**{str(p): sha(p) for p in sources.values()}, **audited['sample_sha256']}
    (out/'provenance').mkdir(parents=True)
    for name, path in sources.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    for name, m in models.items(): write(out/f'model-{name}.json', m)
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(), no_new_geometry_or_physical_draws=True,
        source=str(reference), preparation=str(preparation), source_full_unconditional_N=draws, source_valid_fit_rows=len(rows),
        source_populations=populations, original_source_audit=audited, selected_source_audit=selection,
        original_logQ_reconstruction=physical_logQ, physical={key: cfg[key] for key in PHYSICAL}, shape_sha256=model['shape_sha256'],
        original_q_window=dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False),
        source_ball_radius_A=.5, chart_geometry=geometry, angular_length=ell,
        coordinate_convention='A-body translation and ell times left Cayley residual about the selected peak anchor',
        fit_rule='Normalize original positive importance weights once across all equal-budget populations. No clipping, trimming, or equal-population renormalization. Weighted maximum-likelihood scatter plus additive floor.',
        floor_rule='diag(.05^2 I3, [ell*tan(.1 degree/2)]^2 I3)', floor=floor, fit=fit,
        geometry_scaled_mean=inverse@fit['mean'], geometry_scaled_covariance=inverse@fit['covariance']@inverse.T,
        geometry_scaled_covariance_eigenvalues=eigvalsh(fit['covariance'], np.asarray(model['covariances'][0])),
        pooled_conditional_expected_log_proposal_density={name: float(weights@d) for name, d in densities.items()},
        population_robustness=robustness, independent_density_errors=density_errors,
        predictive_scope='Each held-out score is an original-weight normalized average of log g under the physical distribution conditional on the source ball and hard/q/capture masks. g is normalized on full pose space, not renormalized to this truncation. Scores are ratio diagnostics, not unbiased likelihoods; overlapping leave-one-population-out training sets make folds dependent.',
        scope='A conditional truncated-region proposal fit, not a complete physical basin covariance, region occupancy, or a new free-energy estimate. Its globally supported Gaussian may propose outside the training region, where fresh physical validation and a complete-support mixture remain necessary. Original source estimates and uncertainty are retained, not replaced by the normalized training weights.',
        input_sha256=source_hashes, archived_sha256={name: sha(out/'provenance'/name) for name in sources},
        output_model_sha256={name: sha(out/f'model-{name}.json') for name in models}, fitting_CPU_seconds=time.process_time()-started)
    for path, digest in source_hashes.items(): require(sha(path) == digest, 'Fit input changed during calculation')
    write(out/'report.json', serial(report))
    write(out/'freeze.json', dict(report_sha256=sha(out/'report.json'), model_sha256=report['output_model_sha256'],
        archived_sha256=report['archived_sha256'], source_sample_sha256=audited['sample_sha256']))
    print(out/'report.json')
    print('source ESS', fit['normalized_weight_ESS'], 'largest weight', fit['largest_normalized_weight'])
    for row in robustness:
        print(row['population'], 'LOO mean shift', row['leave_one_population_out_mean_displacement_geometry_radius'],
              'LOO covariance range', row['leave_one_population_out_covariance_eigenvalues_relative_to_pooled'][[0, -1]],
              'held-out gains', row['leave_one_population_out_gain_vs_geometry'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); fit_reference(args.reference, args.preparation, args.out)
