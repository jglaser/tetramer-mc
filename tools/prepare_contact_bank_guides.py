#!/usr/bin/env python3
"""Freeze population-stratified six-dimensional importance guides; no physical draws.

Physical source weights are never clipped in held-out diagnostics. Capping belongs
only to local covariance fitting. Both declared covariance widths are retained;
leave-one-population-out diagnostics do not choose a winner or prove convergence.
"""
from __future__ import annotations

import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from prepare_native_confirmation_atlas import geometric_floor, weighted_fit
from diagnose_conditional_ray_extremes import chart_coordinates

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('native', 'no-entry')
WIDTHS = {'bank': 1.0, 'wide': 4.0}
PINS = {
    'protocol_sha256': '69c15c1c501b12c825495e5267cc81684ddabbe825d1cc951289acaef179c891',
    'status_sha256': '4a56e27e40234d29cf3c344a4619fd98e5eadbad5f5a57148cbf29ba3e7374d6',
    'freeze_sha256': 'a20e2cac291a2a6c2c47e3feb95f00d36b22dcc5463cf9591788133337588db7',
    'region_sha256': '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    'shape_sha256': 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    'native_definition_sha256': '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Source hash mismatch: {path}: {actual} != {expected}')
    return actual


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def capped_weights(log_weights, cap=1 / 16):
    """Waterfill min(cap, c exp(logw)); stable even for extreme log-weight gaps."""
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1 or not np.isfinite(log_weights).all():
        raise ValueError('Finite one-dimensional fitting log weights required')
    n = len(log_weights)
    if not 0 < cap <= 1 or n * cap < 1 - 1e-14:
        raise ValueError('Not enough distinct neighbors for the normalized-weight cap')
    raw = np.exp(log_weights - logsumexp(log_weights))
    result = np.zeros(n)
    active = np.ones(n, dtype=bool)
    remaining = 1.0
    iterations = 0
    while active.any():
        indices = np.flatnonzero(active)
        candidate = remaining * np.exp(log_weights[indices] - logsumexp(log_weights[indices]))
        above = candidate > cap + 1e-15
        iterations += 1
        if not above.any():
            result[indices] = candidate
            break
        saturated = indices[above]
        result[saturated] = cap
        active[saturated] = False
        remaining = 1.0 - float(result.sum())
        if remaining <= 1e-15:
            break
    if not np.isclose(result.sum(), 1.0, rtol=0, atol=2e-13) or result.max() > cap + 2e-13:
        raise ValueError('Waterfill normalization failed')
    return result, {
        'raw_weight_ESS': float(1 / (raw @ raw)),
        'raw_largest_weight': float(raw.max()),
        'capped_weight_ESS': float(1 / (result @ result)),
        'capped_largest_weight': float(result.max()),
        'saturated_weights': int(np.sum(np.isclose(result, cap, rtol=0, atol=2e-14))),
        'cap': cap, 'iterations': iterations,
    }


def latent_geometric_floor(chart):
    """Convert the existing physical raw-chart floor to the archived u chart."""
    lower = np.linalg.cholesky(np.asarray(chart['covariances'][0], dtype=float))
    inverse = np.linalg.solve(lower, np.eye(6))
    raw = geometric_floor(ell=chart['angular_length'], translation_std=.05, angle_axis_scale_deg=.1)
    transformed = inverse @ raw @ inverse.T
    return .5 * (transformed + transformed.T), raw


def source_identity(data, index, populations):
    pop = populations[int(data['population'][index])]
    return {'arm': pop['arm'], 'population_id': pop['id'], 'seed': pop['seed'],
            'draw': int(data['draw'][index]), 'source_samples': pop['samples'],
            'source_population_index': int(data['population'][index])}


def nearest_distinct(data, eligible, tree, anchor, count):
    """Nearest original-u poses; ties use the stable source population/draw order.

    Querying the complete boundary ball makes tie handling independent of KD-tree
    internals. Exact repeated coordinates contribute only once to the fit.
    """
    if len(eligible) < count:
        raise ValueError('Not enough same-class source poses')
    ask = min(len(eligible), max(count * 2, 128))
    while True:
        distances, locations = tree.query(anchor, k=ask, workers=1)
        radius = float(np.max(np.atleast_1d(distances)))
        local = np.asarray(tree.query_ball_point(anchor, np.nextafter(radius, np.inf)), dtype=int)
        indices = eligible[local]
        squared = np.sum((data['u'][indices] - anchor) ** 2, axis=1)
        indices = indices[np.lexsort((indices, squared))]
        chosen, seen = [], set()
        for index in indices:
            key = tuple(data['u'][index])
            if key not in seen:
                seen.add(key)
                chosen.append(int(index))
                if len(chosen) == count:
                    return np.asarray(chosen)
        if ask == len(eligible):
            raise ValueError('Not enough distinct same-class source poses')
        ask = min(len(eligible), ask * 2)


def fit_bank(data, populations, floor, excluded_population=None, neighbors=64, cap=1 / 16):
    """One immutable anchor for every retained population and definition class."""
    keep = data['population'] != excluded_population if excluded_population is not None else np.ones(len(data['u']), bool)
    training_populations = [i for i in range(len(populations)) if i != excluded_population]
    components, metadata = [], []
    trees = {}
    for class_id, name in enumerate(CLASSES):
        eligible = np.flatnonzero(keep & (data['class_id'] == class_id))
        if len(eligible) < neighbors:
            raise ValueError(f'Insufficient distinct geometry for class {name}')
        trees[class_id] = (eligible, cKDTree(data['u'][eligible]))
    for population in training_populations:
        for class_id, name in enumerate(CLASSES):
            rows = np.flatnonzero(keep & (data['population'] == population) & (data['class_id'] == class_id))
            if not len(rows):
                raise ValueError(f'Missing {name} anchor in source population {population}')
            anchor_index = int(rows[np.argmax(data['log_weight'][rows])])
            anchor = data['u'][anchor_index]
            eligible, tree = trees[class_id]
            indices = nearest_distinct(data, eligible, tree, anchor, neighbors)
            source_n = np.asarray([populations[int(i)]['samples'] for i in data['population'][indices]])
            fitting_logs = data['log_weight'][indices] - np.log(source_n)
            weights, cap_report = capped_weights(fitting_logs, cap)
            # Underflowed residual weights are legitimate zero fit contributions.
            nonzero = weights > 0
            fit = weighted_fit(data['u'][indices][nonzero], np.log(weights[nonzero]), floor)
            offset = fit['mean'] - anchor
            covariance = fit['covariance'] + np.outer(offset, offset)
            covariance = .5 * (covariance + covariance.T)
            np.linalg.cholesky(covariance)
            components.append({'weight': 1 / (2 * len(training_populations)),
                               'mean': anchor.tolist(), 'covariance': covariance.tolist()})
            entry = source_identity(data, anchor_index, populations)
            entry.update({'component_index': len(components) - 1, 'class': name,
                'original_log_importance_weight': float(data['log_weight'][anchor_index]),
                'mean': anchor.tolist(), 'neighbors': [source_identity(data, int(i), populations) for i in indices],
                'neighbor_count': len(indices), 'distinct_neighbor_count': len({tuple(data['u'][i]) for i in indices}),
                'neighbor_max_distance_u': float(np.linalg.norm(data['u'][indices] - anchor, axis=1).max()),
                'fitting_weights': weights.tolist(), 'weight_cap_diagnostic': cap_report,
                'weighted_neighbor_mean': fit['mean'].tolist(), 'anchored_covariance': covariance.tolist(),
                'covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist()})
            if excluded_population is not None and any(i['source_population_index'] == excluded_population for i in entry['neighbors']):
                raise ValueError('Held-out population leaked into covariance fitting')
            metadata.append(entry)
    return components, metadata


def guide_from_components(components, region_sha256, multiplier=1.0):
    return {'schema': 'defensive-latent-shell-guide-v1', 'region_sha256': region_sha256,
            'defensive_uniform_shell_probability': .5,
            'gaussian_components': [{'weight': c['weight'], 'mean': c['mean'],
                'covariance': (np.asarray(c['covariance']) * multiplier).tolist()} for c in components]}


def log_proposal(u, guide, radius=4.0):
    """Density of the full, untruncated mixture relative to d^6u."""
    u = np.asarray(u, dtype=float).reshape((-1, 6))
    alpha = guide['defensive_uniform_shell_probability']
    log_volume = 3 * math.log(math.pi) - math.lgamma(4) + 6 * math.log(radius)
    answer = np.where(np.sum(u * u, axis=1) <= radius * radius,
                      math.log(alpha) - log_volume, -np.inf)
    total = sum(c['weight'] for c in guide['gaussian_components'])
    for component in guide['gaussian_components']:
        lower = np.linalg.cholesky(np.asarray(component['covariance']))
        inverse = np.linalg.solve(lower, np.eye(6))
        whitened = (u - np.asarray(component['mean'])) @ inverse.T
        log_normal = -3 * math.log(2 * math.pi) - np.log(np.diag(lower)).sum() - .5 * np.sum(whitened ** 2, axis=1)
        answer = np.logaddexp(answer, math.log(1 - alpha) + math.log(component['weight'] / total) + log_normal)
    return answer


def cross_validate(data, populations, floor, region_sha256, neighbors=64, cap=1 / 16):
    folds = []
    uniform_log_q = -(3 * math.log(math.pi) - math.lgamma(4) + 6 * math.log(4))
    for heldout, population in enumerate(populations):
        components, metadata = fit_bank(data, populations, floor, heldout, neighbors, cap)
        held = data['population'] == heldout
        indices = np.flatnonzero(held)
        density = {name: log_proposal(data['u'][indices], guide_from_components(components, region_sha256, width))
                   for name, width in WIDTHS.items()}
        by_class = {}
        for class_id, name in enumerate(CLASSES):
            select = data['class_id'][indices] == class_id
            rows = indices[select]
            logw = data['log_weight'][rows]
            weights = np.exp(logw - logsumexp(logw))
            scores = {}
            for width in WIDTHS:
                logq = density[width][select]
                # pairs already include J/q_old. Product removes independent
                # Poisson-cloud inflation from the second physical moment.
                log_terms = data['pairs'][rows].sum(axis=1) + data['log_q'][rows] - logq
                normalized = np.exp(log_terms - logsumexp(log_terms))
                scores[width] = {
                    'weighted_log_q': float(weights @ logq),
                    'gain_vs_uniform_nat': float(weights @ (logq - uniform_log_q)),
                    'gain_vs_original_row_q_nat': float(weights @ (logq - data['log_q'][rows])),
                    'weighted_fraction_q_exceeds_original': float(weights @ (logq > data['log_q'][rows])),
                    'paired_cloud_log_secondmoment_estimate': float(logsumexp(log_terms) - math.log(population['samples'])),
                    'paired_cloud_observed_contribution_ESS': float(1 / (normalized @ normalized)),
                    'paired_cloud_largest_contribution': float(normalized.max()),
                }
            by_class[name] = {'saved_class_rows': len(rows), 'source_unconditional_samples': population['samples'],
                'original_weight_ESS': float(1 / (weights @ weights)),
                'original_largest_weight': float(weights.max()),
                'weighted_original_log_q': float(weights @ data['log_q'][rows]), 'guides': scores}
        folds.append({'heldout_population_index': heldout, 'heldout_population': population,
            'training_population_indices': [i for i in range(len(populations)) if i != heldout],
            'components': len(components), 'anchor_metadata': metadata, 'classes': by_class})
        print(f'held-out fold {heldout + 1}/{len(populations)} complete', flush=True)
    summary = {}
    for name in CLASSES:
        summary[name] = {}
        for width in WIDTHS:
            values = np.asarray([fold['classes'][name]['guides'][width]['gain_vs_original_row_q_nat'] for fold in folds])
            summary[name][width] = {'positive_folds': int(sum(values > 0)),
                'equal_population_mean_gain_vs_original_nat': float(values.mean()),
                'median_gain_vs_original_nat': float(np.median(values)),
                'drop_largest_gain_mean_vs_original_nat': float((values.sum() - values.max()) / (len(values) - 1))}
    return {'schema': 'contact-bank-guide-heldout-v1', 'folds': folds, 'summary': summary,
        'selection': 'None: bank and wide are both frozen regardless of these diagnostics.',
        'weighted_log_density_scope': 'Original-weight normalized predictive scores; finite-sample ratio diagnostics, not unbiased free energies. All class rows and original unconditional population sizes are retained.',
        'paired_cloud_scope': 'For held-out X~q_old and independent W1,W2, mean[1_B J(X)^2 W1 W2/(q_old(X) q_new(X))] estimates integral_B J^2 exp(2 z overlap)/q_new du. The saved pairs are log[J Wi/q_old]. Invalid and other-class rows are zero with original N in the denominator. This is a physical second-moment proxy excluding new-cloud noise, not a complete sampling-variance prediction.',
        'caveat': 'Source importance ESS can be near one. Conditional unbiasedness of the paired estimate does not imply a reliable realized estimate or honest narrow error bars. Training folds overlap. These scores neither establish coverage of unseen modes nor physical convergence.'}



def chart_log_jacobian(u, region):
    chart = region['gaussian_chart']
    lower = np.linalg.cholesky(chart['covariances'][0])
    raw = np.asarray(u) @ lower.T + np.asarray(chart['means'][0])
    ell = chart['angular_length']
    c2 = np.sum((raw[:, 3:] / ell) ** 2, axis=1)
    return np.log(np.diag(lower)).sum() - 3 * math.log(ell) - 2 * math.log(math.pi) - 2 * np.log1p(c2)


def reference_coverage(components, region, source_ledger):
    """Score previously saved R5-native contributors; no new validation draws."""
    diagnostic_path = ROOT / 'runs/mobile-conditional-ray-extremes-20260922/analysis.json'
    expected = 'ec4e6b76fa2c27079f7357116157f7df393f0af1eefb4a411ce48ca9e6004f31'
    check_hash(diagnostic_path, expected)
    diagnostic = read(diagnostic_path)
    source_ledger[str(diagnostic_path)] = expected
    reference = Path(diagnostic['reference']['source'])
    reference_path = reference / 'analysis.json'
    check_hash(reference_path, diagnostic['reference']['source_sha256'])
    source_ledger[str(reference_path)] = diagnostic['reference']['source_sha256']
    core = next(s for s in read(reference_path)['strata'] if s['name'] == 'r5repeat')
    labels_path = reference / core['native_labels']['path']
    check_hash(labels_path, core['native_labels']['sha256'])
    source_ledger[str(labels_path)] = core['native_labels']['sha256']
    labels = {}
    with labels_path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['applicable']:
                labels[(item['population'], item['draw'])] = item['label']['native_any']
    guide_models = {name: guide_from_components(components, PINS['region_sha256'], multiplier) for name, multiplier in WIDTHS.items()}
    batches, populations, reports = [], [], []
    uniform_log_q = -(3 * math.log(math.pi) - math.lgamma(4) + 6 * math.log(4))
    for population_index, population in enumerate(core['populations']):
        path = Path(core['campaign']) / 'runs' / population['id'] / 'samples.jsonl'
        expected_hash = core['source_sha256'][str(path)]
        saved, digest, n = [], hashlib.sha256(), 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if row['draw'] != n:
                    raise ValueError('Independent reference draw order changed')
                if row['log_importance_weight'] is not None and labels.get((population['id'], n), False):
                    saved.append(row)
                n += 1
        if digest.hexdigest() != expected_hash or n != population['draws']:
            raise ValueError('Independent reference source identity changed')
        source_ledger[str(path)] = expected_hash
        u = chart_coordinates([row['pose'] for row in saved], region)
        inside = np.linalg.norm(u, axis=1) <= 4
        u = u[inside]
        saved = [row for row, keep in zip(saved, inside) if keep]
        log_j = chart_log_jacobian(u, region)
        log_hard = np.asarray([row['log_hard_weight'] for row in saved])
        # q_old, expressed in current u, follows from the physical measure:
        # J_old/q_old_old == J_current/q_old_current.
        old_current_log_q = log_j - log_hard
        logw = np.asarray([row['log_importance_weight'] for row in saved])
        pairs = np.asarray([[row['log_hard_weight'] + c['log_weight'] for c in row['clouds']] for row in saved])
        if pairs.shape != (len(saved), 2):
            raise ValueError('Independent reference must retain two cloud replicates')
        batch = {'u': u, 'log_weight': logw, 'log_q': old_current_log_q, 'pairs': pairs,
            'population': np.full(len(saved), population_index, dtype=int),
            'draw': np.asarray([row['draw'] for row in saved], dtype=int)}
        for name, guide in guide_models.items():
            batch['log_q_' + name] = log_proposal(u, guide)
        batches.append(batch)
        populations.append({'id': population['id'], 'seed': population['seed'], 'samples': n, 'contributing_rows': len(saved)})
    data = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
    for population_index in list(range(len(populations))) + [None]:
        selected = data['population'] == population_index if population_index is not None else np.ones(len(data['u']), bool)
        denominator = populations[population_index]['samples'] if population_index is not None else sum(p['samples'] for p in populations)
        logw = data['log_weight'][selected]
        normalized = np.exp(logw - logsumexp(logw))
        log_qz = float(logsumexp(logw) - math.log(denominator))
        scores = {}
        for name in ('uniform',) + tuple(WIDTHS):
            qnew = np.full(len(logw), uniform_log_q) if name == 'uniform' else data['log_q_' + name][selected]
            paired = data['pairs'][selected].sum(axis=1) + data['log_q'][selected] - qnew
            normalized_moment = np.exp(paired - logsumexp(paired))
            log_moment = float(logsumexp(paired) - math.log(denominator))
            scores[name] = {
                'weighted_gain_vs_current_uniform_nat': float(normalized @ (qnew - uniform_log_q)),
                'weighted_gain_vs_reference_density_in_current_u_nat': float(normalized @ (qnew - data['log_q'][selected])),
                'weighted_fraction_above_current_uniform': float(normalized @ (qnew > uniform_log_q)),
                'paired_cloud_log_secondmoment_estimate': log_moment,
                'paired_cloud_observed_contribution_ESS': float(1 / (normalized_moment @ normalized_moment)),
                'paired_cloud_largest_contribution': float(normalized_moment.max()),
                'physical_moment_native_ESS_proxy_per_100000_attempts': float(math.exp(math.log(100000) + 2 * log_qz - log_moment)),
            }
        reports.append({'population': populations[population_index]['id'] if population_index is not None else 'pooled',
            'source_unconditional_samples': denominator, 'contributing_rows': int(sum(selected)),
            'original_weight_ESS': float(1 / (normalized @ normalized)),
            'original_largest_weight': float(normalized.max()), 'log_Q_intersection_native': log_qz,
            'guides': scores})
    report = {'schema': 'contact-bank-independent-reference-coverage-v1',
        'source_diagnostic_sha256': expected, 'source_reference': str(reference), 'populations': populations,
        'reports': reports,
        'scope': 'Previously generated R5 native-reference rows falling in the unchanged current R4; no new draws, no reclassification, no fitting, and no model selection from this diagnostic. Independent of bank training data, but already examined evidence, not a fresh pilot.',
        'measure': 'Reference q is converted to current d6u by log q_old_current = log J_current - log_hard_weight_old. All original unconditional reference draws remain in the denominator.',
        'moment_caveat': 'Only the R5 intersection is measured. Paired clouds estimate the physical second moment without cloud inflation. The reported ESS per 100000 is a noisy optimistic moment proxy, ignores new cloud variance and unseen/outside-intersection mass, and is not an effective-sampling-time measurement.'}
    print('saved independent reference coverage: ' + json.dumps(reports[-1]), flush=True)
    return report, data


def load_sources(campaign, comparison):
    protocol = read(campaign / 'protocol.json')
    status = read(campaign / 'status.json')
    analysis = read(comparison / 'analysis.json')
    for name, filename in [('protocol_sha256', 'protocol.json'), ('status_sha256', 'status.json'), ('freeze_sha256', 'freeze.json')]:
        check_hash(campaign / filename, PINS[name])
        if analysis[name] != PINS[name]:
            raise ValueError(f'Comparison source binding mismatch: {name}')
    if not status['complete'] or status['phase'] != 'complete' or not analysis['complete']:
        raise ValueError('Only the completed frozen campaign and comparison may train this guide')
    if len(protocol['jobs']) != 24 or analysis['total_unconditional_draws'] != 786432:
        raise ValueError('Unexpected source population allocation')
    frozen = read(comparison / 'freeze.json')
    check_hash(comparison / 'analysis.json', frozen['analysis.json'])
    bound = {(arm, p['id']): p for arm, a in analysis['arms'].items() for p in a['populations']}
    complete = {(p['arm'], p['id']): p for p in status['jobs']}
    source_ledger, populations, batches = {}, [], []
    def bind(path, expected=None):
        digest = check_hash(path, expected) if expected else sha(path)
        source_ledger[str(path.resolve())] = digest
        return digest
    for filename in ('protocol.json', 'status.json', 'freeze.json'):
        bind(campaign / filename)
    for filename in ('analysis.json', 'freeze.json'):
        bind(comparison / filename)
    for population_id, job in enumerate(protocol['jobs']):
        population = bound[(job['arm'], job['id'])]
        completed = complete[(job['arm'], job['id'])]
        if completed['status'] != 'complete' or completed['returncode'] != 0 or completed['output'] != population['raw_output']:
            raise ValueError('Source population completion or comparison binding mismatch')
        n = job['samples']
        if population['samples'] != n or population['seed'] != job['seed']:
            raise ValueError('Source sample allocation mismatch')
        records_path = comparison / population['records']
        bind(records_path, population['records_sha256'])
        if frozen[population['records']] != population['records_sha256']:
            raise ValueError('Comparison record freeze mismatch')
        labels_path = comparison / population['labels']
        bind(labels_path, population['labels_sha256'])
        classification = comparison / job['arm'] / job['id'] / 'classification.json'
        bind(classification, frozen[str(classification.relative_to(comparison))])
        records = np.load(records_path, allow_pickle=False)
        valid = np.isfinite(records['z'])
        if len(valid) != n or records['pairs'].shape != (n, 2):
            raise ValueError('Saved record count mismatch')
        # Full definition classes only: there is no q, stratum, weight, or
        # observer refinement filter. Remaining space is retained by defense.
        class_id = np.where(records['native'], 0, np.where(records['contact'], 1, 2))
        selected = valid & (class_id < 2)
        saved = np.flatnonzero(selected)
        u, old_q, draw_ids = [], [], []
        raw_path = Path(job['directory']) / 'samples.jsonl'
        digest = hashlib.sha256()
        rows = 0
        with raw_path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if row['draw'] != rows or rows >= n:
                    raise ValueError('Raw draw ordering mismatch')
                if selected[rows]:
                    if not all(row[key] for key in ('hard_valid', 'region_valid', 'shell_valid', 'capture_valid')):
                        raise ValueError('Saved positive class row is not hard/region valid')
                    if not math.isclose(row['log_importance_weight'], float(records['z'][rows]), rel_tol=0, abs_tol=2e-10):
                        raise ValueError('Saved original importance weight mismatch')
                    if len(row['clouds']) != 2:
                        raise ValueError('Two independent archived clouds are required')
                    expected_pairs = [row['log_physical_jacobian'] - row['log_proposal_density'] + c['log_weight'] for c in row['clouds']]
                    if not np.allclose(expected_pairs, records['pairs'][rows], rtol=0, atol=2e-10):
                        raise ValueError('Archived independent cloud-weight alignment mismatch')
                    u.append(row['latent']); old_q.append(row['log_proposal_density']); draw_ids.append(rows)
                rows += 1
        if rows != n or digest.hexdigest() != population['raw_output']['samples_sha256']:
            raise ValueError('Raw saved draw count or hash mismatch')
        source_ledger[str(raw_path.resolve())] = digest.hexdigest()
        if not np.array_equal(draw_ids, saved):
            raise ValueError('Source record/raw alignment mismatch')
        for filename, key in [('summary.json', 'summary_sha256'), ('manifest.json', 'manifest_sha256')]:
            bind(Path(job['directory']) / filename, population['raw_output'][key])
        provenance = campaign / job['arm'] / 'provenance'
        for filename in ('config.json', 'region.json', 'shape.json', 'importance-guide.json'):
            bind(provenance / filename)
        populations.append({'arm': job['arm'], 'id': job['id'], 'seed': job['seed'], 'samples': n,
            'raw_directory': job['directory'], 'comparison_records': str(records_path.resolve()),
            'class_rows': {name: int(sum(valid & (class_id == i))) for i, name in enumerate(CLASSES)},
            'other_valid_rows': int(sum(valid & (class_id == 2))), 'zero_weight_rows': int(sum(~valid)),
            'raw_output': population['raw_output']})
        batches.append({'u': np.asarray(u, dtype=float), 'log_weight': records['z'][saved],
            'log_q': np.asarray(old_q), 'pairs': records['pairs'][saved], 'draw': saved,
            'class_id': class_id[saved], 'population': np.full(len(saved), population_id, dtype=int)})
        print(f'loaded source population {population_id + 1}/24: {len(saved)} class poses', flush=True)
    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    for key, values in data.items():
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite retained source field: {key}')
    if np.any(np.linalg.norm(data['u'], axis=1) > 4 + 1e-10):
        raise ValueError('Source poses escaped unchanged R4 region')
    return data, populations, source_ledger, analysis


def local_source_closure(paths):
    """Archive every local Python import recursively, including test imports."""
    found, pending = {}, [Path(p).resolve() for p in paths]
    while pending:
        path = pending.pop()
        if path.name in found:
            if found[path.name] != path:
                raise ValueError('Source closure basename collision')
            continue
        found[path.name] = path
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                candidate = path.parent / (module.split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate.resolve())
    return found


def prepare(campaign, comparison, out):
    start = time.monotonic()
    if out.exists():
        raise ValueError(f'Refusing to overwrite preparation: {out}')
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools'),
                           '-p', 'test_contact_bank_guides.py', '-v'], capture_output=True, text=True)
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    data, populations, source_ledger, analysis = load_sources(campaign, comparison)
    provenance = campaign / 'pilot_uniform' / 'provenance'
    region = read(provenance / 'region.json')
    for key, filename in [('region_sha256', 'region.json'), ('shape_sha256', 'shape.json')]:
        check_hash(provenance / filename, PINS[key])
    if region['activity'] != .035 or region['depletant_radius'] != 1.5 or region['minimum_mahalanobis_radius'] != 0 or region['mahalanobis_radius'] != 4:
        raise ValueError('Physical region pins changed')
    floor, raw_floor = latent_geometric_floor(region['gaussian_chart'])
    components, anchors = fit_bank(data, populations, floor)
    coverage, coverage_data = reference_coverage(components, region, source_ledger)
    cross = cross_validate(data, populations, floor, PINS['region_sha256'])
    out.mkdir(parents=True)
    for filename in ('config.json', 'region.json', 'shape.json'):
        shutil.copyfile(provenance / filename, out / filename)
    native = Path(analysis['native_definition']['definition'])
    check_hash(native, PINS['native_definition_sha256'])
    shutil.copyfile(native, out / 'native-definition.json')
    (out / 'native-region').mkdir()
    shutil.copyfile(native, out / 'native-region' / 'definition.json')
    for relative, expected in analysis['native_definition']['input_sha256'].items():
        source = native.parent / 'inputs' / relative
        check_hash(source, expected)
        source_ledger[str(source.resolve())] = expected
        target = out / 'native-region' / 'inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    source_ledger[str(native.resolve())] = PINS['native_definition_sha256']
    (out / 'provenance').mkdir()
    for base, prefix, files in [(campaign, 'source-campaign', ('protocol.json', 'status.json', 'freeze.json')),
                                (comparison, 'source-comparison', ('analysis.json', 'freeze.json'))]:
        for filename in files:
            shutil.copyfile(base / filename, out / 'provenance' / f'{prefix}-{filename}')
    for population in populations:
        arm = population['arm']
        relative = f'{arm}/{population["id"]}/classification.json'
        target = out / 'provenance' / 'classifications' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(comparison / relative, target)
    source_closure = local_source_closure([Path(__file__), ROOT / 'tools' / 'test_contact_bank_guides.py'])
    for name, source in source_closure.items():
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_ledger[str(source)] = sha(source)
    guides = {name: guide_from_components(components, PINS['region_sha256'], width) for name, width in WIDTHS.items()}
    for name, guide in guides.items():
        write(out / f'guide-{name}.json', guide)
    write(out / 'cross-validation.json', cross)
    write(out / 'independent-reference-coverage.json', coverage)
    np.savez_compressed(out / 'independent-reference-poses.npz', **coverage_data)
    write(out / 'source-closure.json', {'schema': 'contact-bank-source-closure-v1', 'files': source_ledger,
        'scope': 'Immutable source identities and aligned saved records. Geometry, proposal, and Poisson audits already performed by the bound campaign/comparison were not rerun.'})
    np.savez_compressed(out / 'training-poses.npz', **data)
    (out / 'tests.txt').write_text(test.stdout + test.stderr)
    plan = {'schema': 'contact-bank-guide-preparation-v1', 'complete': True, 'production_launched': False,
        'created': datetime.now(timezone.utc).isoformat(), 'source_campaign': str(campaign),
        'source_comparison': str(comparison), **PINS,
        'config_sha256': sha(out / 'config.json'),
        'guide_sha256': {name: sha(out / f'guide-{name}.json') for name in WIDTHS},
        'guides': {name: f'guide-{name}.json' for name in WIDTHS},
        'covariance_multipliers': WIDTHS, 'anchor_metadata': anchors,
        'source_populations': populations, 'source_seeds': [p['seed'] for p in populations],
        'source_unconditional_draws': sum(p['samples'] for p in populations),
        'retained_training_class_rows': len(data['u']),
        'source_comparison_analysis_sha256': sha(comparison / 'analysis.json'),
        'native_definition': 'native-region/definition.json',
        'native_input_sha256': analysis['native_definition']['input_sha256'],
        'activity_A^-3': .035, 'depletant_radius_A': 1.5,
        'fitting_rule': {'anchors': 'Highest ORIGINAL saved importance-weight pose per source population and full definition class; all 48 means remain anchored.',
            'classes': {'native': 'registered_native_entry', 'no-entry': 'contact_no_native_entry'},
            'neighbors': 64, 'distance': 'Euclidean original six-dimensional u; distinct coordinates; stable population/draw tie order.',
            'fitting_log_weight': 'original_log_importance_weight - log(source_unconditional_samples)',
            'normalized_weight_cap': 1 / 16,
            'covariance': 'Capped weighted second moment about the mandatory anchor + transformed physical geometric floor.',
            'translation_floor_std_A': .05, 'angle_axis_floor_scale_deg': .1,
            'raw_geometric_floor': raw_floor, 'latent_geometric_floor': floor,
            'allocation': {'uniform_R4': .5, 'native_Gaussians': .25, 'no_entry_Gaussians': .25},
            'selection': 'Both base and four-times-covariance guides frozen; no held-out winner selection.'},
        'heldout_diagnostics': 'cross-validation.json',
        'independent_reference_coverage': 'independent-reference-coverage.json',
        'estimator': 'For NEW independent X~q_g, retain every draw, including zero-weight invalid/out-of-region poses. Estimate Z_B = mean[1_B J(X) W(X)/q_g(X)], using independent nonnegative unbiased Poisson W for exp(z overlap); q_g is the full untruncated Gaussian/uniform mixture. Old training rows are not new physical evidence.',
        'scope': 'Native-informed proposal reference for unchanged finite R4, exact repaired shape and original native definition. The uniform floor covers all remaining R4 configurations. This is not a full-vessel, assembly, geometry-only discovery, or thermodynamic-stability result.',
        'config_path_note': 'config.json is an exact byte copy and retains the frozen source shape path. A runner may change only its launched copy of that path to the copied shape.json, preserving this source file and recording both hashes.',
        'tests': {'returncode': test.returncode, 'report': 'tests.txt', 'source_sha256': sha(ROOT / 'tools' / 'test_contact_bank_guides.py')},
        'preparation_wall_seconds': time.monotonic() - start}
    write(out / 'plan.json', plan)
    (out / 'README.md').write_text('Frozen contact-bank importance guides\n\n'
        '48 mandatory source-population/class anchors, capped local anchored covariance and a physical geometric floor. '
        'The bank and wide proposals retain the same centers and allocations; wide covariance is exactly four times bank. '
        'Both use a 50% uniform R4 defense. Gaussian tails are not truncated or renormalized.\n\n'
        'cross-validation.json excludes each entire held-out population from both anchors and fitting neighbors. '
        'Original source weights are preserved for scoring and often have very low ESS. These predictive diagnostics are not physical convergence evidence. '
        'No physical samples were generated. New independent, unconditional importance populations are required.\n\n'
        'plan.json records the estimator and construction; source-closure.json binds the immutable original records and code. '
        'The compressed training rows retain all hard-valid full-class rows with original weights, original proposal densities and two independent cloud weights. '
        'Other classes and invalid rows remain in each population allocation; their space is covered by the uniform defense.\n')
    frozen = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out / 'freeze.json', {'files': frozen})
    print(json.dumps({'out': str(out), 'components': len(components), 'class_rows': len(data['u']),
        'guide_sha256': plan['guide_sha256'], 'cross_validation_summary': cross['summary']}, indent=2), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs/mobile-conditional-ray-campaign-20260921')
    parser.add_argument('--comparison', type=Path, default=ROOT / 'runs/mobile-conditional-ray-comparison-20260921')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/contact-bank-guide-preparation-20260922')
    args = parser.parse_args()
    prepare(args.campaign.resolve(), args.comparison.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
