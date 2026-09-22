#!/usr/bin/env python3
"""Score frozen Gaussian refinements on saved pilot populations; no physics.

The pilot is training evidence. Twelve declared hyperparameter combinations use
whole-population holdouts, unclipped physical scoring weights and independent
cloud products. The previous 56-component bank remains half the Gaussian guide.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from prepare_contact_bank_guides import (ROOT, capped_weights, guide_from_components,
    local_source_closure, log_proposal, nearest_distinct, read, sha, source_identity, write)
from analyze_contact_bank_reference_partition import (HashLedger, partition_masks,
    r5_contains, validate_regions)
from diagnose_conditional_ray_extremes import chart_coordinates

STRATA = ('old_R5_intersection_native', 'remaining_R4_native', 'contact_no_native_entry')
PARTITION_SHA256 = '4602f6255473a2957412e9a4bc79209f435692d6806727e40124d8d65b62bb4f'
OLD_BANK_SHA256 = '5bc58a35d8a87c8677f73d997860595790cb7fe0c875414ce17a59dc8184002c'
PILOT_PROTOCOL_SHA256 = '98ca05e13dc211542204a072d4e07c89b18bb9709baee52688588eff74896a8d'
CANDIDATES = tuple({'name': f'{center}-n{neighbors}-f{factor:g}', 'center': center,
    'neighbors': neighbors, 'floor_multiplier': factor, 'cap': 1 / 16}
    for center in ('anchor', 'weighted-mean') for neighbors in (64, 128)
    for factor in (1., .25, .0625))
SCOPE = ('Training-only proposal design diagnostic from all eight completed pilot populations. '
    'No new physical samples, classifier calls or audit replay. Whole-population holdouts remove '
    'that pilot population from candidate anchors and covariance neighbors. Old-bank components '
    'were fixed before the pilot and remain unchanged. Hyperparameters may be selected from these '
    'scores, so these scores are not fresh validation of the selected or full-data proposal. '
    'Physical moment estimates use independent cloud products and omit new-cloud noise. '
    'Sparse contribution ESS and unseen support can invalidate apparent gains. No convergence, '
    'mixing rate, full-vessel or assembly conclusion follows.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def mixture_guide(old_guide, candidate_components):
    """q_new=.5 U_R4+.25 G_old+.25 G_candidate >= .5 q_old globally."""
    require(old_guide['defensive_uniform_shell_probability'] == .5, 'Old uniform allocation changed')
    old = copy.deepcopy(old_guide['gaussian_components'])
    new = copy.deepcopy(candidate_components)
    require(old and new, 'Both Gaussian groups are mandatory')
    for group in (old, new):
        total = sum(c['weight'] for c in group)
        require(total > 0 and all(c['weight'] > 0 for c in group), 'Invalid component allocation')
        for component in group:
            component['weight'] = .5 * component['weight'] / total
    return guide_from_components(old + new, old_guide['region_sha256'])


def fit_candidates(data, populations, floor, specification, excluded_population=None):
    """One same-stratum neighborhood per retained population; cap only fit weights."""
    neighbors, cap = specification['neighbors'], specification['cap']
    center, multiplier = specification['center'], specification['floor_multiplier']
    require(center in ('anchor', 'weighted-mean') and multiplier > 0, 'Unknown fitting rule')
    floor = np.asarray(floor, float)
    require(floor.shape == (6, 6) and np.allclose(floor, floor.T), 'Invalid geometric floor')
    np.linalg.cholesky(floor)
    retained = [i for i in range(len(populations)) if i != excluded_population]
    keep = np.isin(data['population'], retained)
    trees = {}
    for class_id, name in enumerate(STRATA):
        eligible = np.flatnonzero(keep & (data['class_id'] == class_id))
        require(len(eligible) >= neighbors, 'Insufficient neighbors for ' + name)
        trees[class_id] = (eligible, cKDTree(data['u'][eligible]))
    components, metadata = [], []
    for population in retained:
        for class_id, name in enumerate(STRATA):
            rows = np.flatnonzero((data['population'] == population) & (data['class_id'] == class_id))
            require(len(rows) > 0, f'Missing {name} anchor in population {population}')
            anchor_index = int(rows[np.argmax(data['log_weight'][rows])])
            anchor = data['u'][anchor_index]
            eligible, tree = trees[class_id]
            indices = nearest_distinct(data, eligible, tree, anchor, neighbors)
            source_n = np.asarray([populations[int(p)]['samples'] for p in data['population'][indices]])
            weights, cap_report = capped_weights(data['log_weight'][indices] - np.log(source_n), cap)
            weighted_mean = weights @ data['u'][indices]
            mean = anchor if center == 'anchor' else weighted_mean
            offsets = data['u'][indices] - mean
            empirical = offsets.T @ (weights[:, None] * offsets)
            empirical = .5 * (empirical + empirical.T)
            covariance = empirical + multiplier * floor
            covariance = .5 * (covariance + covariance.T)
            np.linalg.cholesky(covariance)
            components.append({'weight': 1 / (len(STRATA) * len(retained)),
                'mean': mean.tolist(), 'covariance': covariance.tolist()})
            metadata.append({**source_identity(data, anchor_index, populations),
                'component_index': len(components) - 1, 'class_id': class_id, 'class': name,
                'center_rule': center, 'anchor': anchor.tolist(), 'mean': mean.tolist(),
                'weighted_neighbor_mean': weighted_mean.tolist(),
                'original_anchor_log_importance_weight': float(data['log_weight'][anchor_index]),
                'neighbors': [source_identity(data, int(index), populations) for index in indices],
                'neighbor_count': len(indices), 'fitting_weights': weights.tolist(),
                'fitting_rule': 'original log importance weight minus log source unconditional N',
                'weight_cap_diagnostic': cap_report, 'empirical_covariance': empirical.tolist(),
                'floor_multiplier': multiplier, 'covariance': covariance.tolist(),
                'covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
                'floor_trace_fraction_u': float(multiplier * np.trace(floor) / np.trace(covariance)),
                'maximum_neighbor_distance_u': float(np.linalg.norm(data['u'][indices] - anchor, axis=1).max())})
            require(excluded_population is None or all(n['source_population_index'] != excluded_population
                for n in metadata[-1]['neighbors']), 'Held-out population leaked into fit')
    return components, metadata


def score_rows(data, selected, n, target_log_q, old_bank_log_q):
    """Original J/q_source weights, all-N denominators, physical cloud-product M2."""
    indices = np.flatnonzero(selected)
    require(n >= len(indices) and n > 0, 'Unconditional denominator lost zero rows')
    if not len(indices):
        return {'source_unconditional_samples': n, 'contributing_rows': 0,
                'log_Qz': None, 'original_weight_ESS': 0., 'guides': {}}
    logs = data['log_weight'][indices]
    require(np.isfinite(logs).all() and np.isfinite(data['pairs'][indices]).all(), 'Invalid contributing weights')
    normalized = np.exp(logs - logsumexp(logs))
    logqz = float(logsumexp(logs) - math.log(n))
    # Each pair already includes J/q_source. Multiplication by q_source/q_target
    # cancels one source density after integrating under the actual source law.
    physical_terms = data['pairs'][indices].sum(axis=1) + data['log_q'][indices]
    old_terms = physical_terms - old_bank_log_q[indices]
    old_logmoment = float(logsumexp(old_terms) - math.log(n))
    scores = {}
    for name, values in target_log_q.items():
        logq = values[indices]
        terms = physical_terms - logq
        moments = np.exp(terms - logsumexp(terms))
        logmoment = float(logsumexp(terms) - math.log(n))
        scores[name] = {'weighted_log_q': float(normalized @ logq),
            'weighted_gain_vs_old_bank_nat': float(normalized @ (logq - old_bank_log_q[indices])),
            'weighted_gain_vs_actual_source_q_nat': float(normalized @ (logq - data['log_q'][indices])),
            'weighted_fraction_q_above_old_bank': float(normalized @ (logq > old_bank_log_q[indices])),
            'paired_cloud_log_secondmoment': logmoment,
            'paired_cloud_secondmoment_ratio_to_old_bank': float(math.exp(logmoment - old_logmoment)),
            'paired_cloud_contribution_ESS': float(1 / (moments @ moments)),
            'paired_cloud_largest_contribution': float(moments.max()),
            'paired_cloud_top10_contribution_fraction': float(np.sort(moments)[-10:].sum()),
            'observed_row_RSE_of_cloudproduct_mean': float(math.sqrt(max(0., n / (n - 1) * (moments @ moments - 1 / n)))) if n > 1 else None,
            'physical_moment_ESS_projections': {str(attempts): float(math.exp(math.log(attempts) + 2 * logqz - logmoment)) for attempts in (524288, 1048576)},
            'physical_moment_ESS_proxy_per_100000_attempts': float(math.exp(math.log(100000) + 2 * logqz - logmoment))}
    return {'source_unconditional_samples': n, 'contributing_rows': len(indices), 'log_Qz': logqz,
        'original_weight_ESS': float(1 / (normalized @ normalized)),
        'original_largest_weight': float(normalized.max()), 'guides': scores}


def training_report(data, populations, densities, old_bank_log_q):
    reports = {}
    groups = [('population-' + str(i), [i]) for i in range(len(populations))]
    groups += [(arm, [i for i, p in enumerate(populations) if p['arm'] == arm])
               for arm in sorted({p['arm'] for p in populations})]
    groups += [('pooled', list(range(len(populations))))]
    for group, pops in groups:
        selected = np.isin(data['population'], pops)
        n = sum(populations[i]['samples'] for i in pops)
        masks = {name: selected & (data['class_id'] == i) for i, name in enumerate(STRATA)}
        masks['all_native'] = selected & np.isin(data['class_id'], [0, 1])
        reports[group] = {'population_indices': pops, 'classes': {name:
            score_rows(data, mask, n, densities, old_bank_log_q) for name, mask in masks.items()}}
    return reports


def cross_validate(data, populations, floor, old_guide, specifications=CANDIDATES):
    densities = {s['name']: np.full(len(data['u']), np.nan) for s in specifications}
    old_bank_log_q = log_proposal(data['u'], old_guide)
    folds = []
    for heldout, population in enumerate(populations):
        selected = data['population'] == heldout
        scored = selected & (data['class_id'] >= 0)
        fold = {'heldout_population_index': heldout, 'heldout_population': population,
            'training_population_indices': [i for i in range(len(populations)) if i != heldout], 'candidates': {}}
        for specification in specifications:
            components, metadata = fit_candidates(data, populations, floor, specification, heldout)
            guide = mixture_guide(old_guide, components)
            values = log_proposal(data['u'][scored], guide)
            require(np.all(values >= old_bank_log_q[scored] - math.log(2) - 1e-12), 'Defensive old-bank bound violated')
            densities[specification['name']][scored] = values
            fold['candidates'][specification['name']] = {'guide_canonical_sha256': canonical_sha(guide),
                'component_count': len(guide['gaussian_components']), 'anchor_metadata': metadata}
        folds.append(fold)
        print(f'whole-pilot-population holdout {heldout + 1}/{len(populations)} complete', flush=True)
    densities['old-bank'] = old_bank_log_q
    scores = training_report(data, populations, densities, old_bank_log_q)
    summaries = {}
    for specification in specifications:
        name = specification['name']; strata = {}
        for stratum in STRATA:
            fold_scores = [scores[f'population-{i}']['classes'][stratum]['guides'][name] for i in range(len(populations))]
            gains = [s['weighted_gain_vs_old_bank_nat'] for s in fold_scores]
            ratios = [s['paired_cloud_secondmoment_ratio_to_old_bank'] for s in fold_scores]
            strata[stratum] = {'pooled': scores['pooled']['classes'][stratum]['guides'][name],
                'equal_population_mean_log_density_gain': float(np.mean(gains)),
                'minimum_population_log_density_gain': float(min(gains)),
                'positive_log_density_gain_folds': sum(g > 0 for g in gains),
                'secondmoment_improvement_folds': sum(r < 1 for r in ratios),
                'worst_population_secondmoment_ratio': max(ratios),
                'minimum_population_secondmoment_ESS': min(s['paired_cloud_contribution_ESS'] for s in fold_scores),
                'by_source_arm': {arm: scores[arm]['classes'][stratum]['guides'][name]
                    for arm in sorted({p['arm'] for p in populations})}}
        # Diagnostic sensitivity to removing a scored cloud, holding its already
        # fitted fold proposal fixed; overlapping CV fits prohibit an independent-population CI.
        for stratum in STRATA:
            class_id = STRATA.index(stratum)
            leave_one = []
            for dropped in range(len(populations)):
                mask = (data['class_id'] == class_id) & (data['population'] != dropped)
                n = sum(p['samples'] for i, p in enumerate(populations) if i != dropped)
                score = score_rows(data, mask, n, {name: densities[name]}, old_bank_log_q)['guides'][name]
                leave_one.append(score['paired_cloud_secondmoment_ratio_to_old_bank'])
            strata[stratum]['drop_one_scored_population_ratio_range'] = [min(leave_one), max(leave_one)]
            strata[stratum]['drop_one_scored_population_ratios'] = leave_one
        summaries[name] = {'specification': specification, 'strata': strata,
            'maximum_pooled_stratum_secondmoment_ratio': max(s['pooled']['paired_cloud_secondmoment_ratio_to_old_bank'] for s in strata.values())}
    ranking = sorted(summaries, key=lambda name: (summaries[name]['maximum_pooled_stratum_secondmoment_ratio'], name))
    return {'schema': 'contact-refinement-whole-population-cv-v1', 'folds': folds,
        'scores': scores, 'candidate_summaries': summaries, 'training_ranking': ranking,
        'ranking_rule': 'Minimize maximum of the three pooled held-out paired-cloud second-moment ratios to fixed old bank; equal named-stratum protection. Diagnostics, not validation.',
        'pooled_moment_scope': 'Pooled averages eight different fold-proposal moment estimates with all original attempt counts. It is not the moment of any final all-data fitted guide.',
        'scope': SCOPE}, densities


def load_pilot(campaign, comparison, partition, old_preparation):
    """Reuse frozen numeric labels/weights; parse coordinates and original q only."""
    ledger = HashLedger()
    ledger.frozen(partition)
    partition_result = read(ledger.bind(partition / 'analysis.json', PARTITION_SHA256))
    require(partition_result['complete'] and partition_result['new_physical_samples'] == 0
        and partition_result['classifiers_rerun'] == 0, 'Partition is not completed saved evidence')
    partition_plan = read(ledger.bind(partition / 'supplemental-plan.json', partition_result['plan_sha256']))
    ledger.frozen(comparison)
    primary = read(ledger.bind(comparison / 'analysis.json'))
    require(primary['complete'] and primary['total_unconditional_draws'] == 131072
        and Path(primary['campaign']).resolve() == campaign, 'Completed pilot comparison differs')
    require(Path(partition_result['primary_comparison']).resolve() == comparison, 'Partition belongs to another comparison')
    state = read(ledger.bind(comparison / 'status.json'))
    require(state['complete'] and state['phase'] == 'complete' and state['analysis_sha256'] == sha(comparison / 'analysis.json'), 'Comparison status changed')
    protocol = read(ledger.bind(campaign / 'protocol.json', PILOT_PROTOCOL_SHA256))
    require(primary['protocol_sha256'] == PILOT_PROTOCOL_SHA256, 'Pilot protocol changed')
    status = read(ledger.bind(campaign / 'status.json', primary['status_sha256']))
    ledger.bind(campaign / 'freeze.json', primary['freeze_sha256'])
    require(status['complete'] and status['phase'] == 'complete', 'Pilot did not finish')
    require(len(protocol['jobs']) == 8 and sum(j['samples'] for j in protocol['jobs']) == 131072, 'Pilot attempt counts changed')
    require(len({j['seed'] for j in protocol['jobs']}) == 8, 'Pilot seeds repeat')
    reference = read(ledger.bind(partition_plan['reference_region'], partition_plan['reference_region_sha256']))
    old_guide = read(ledger.bind(old_preparation / 'guide-bank.json', OLD_BANK_SHA256))
    old_plan = read(ledger.bind(old_preparation / 'plan.json'))
    require(old_plan['complete'] and old_plan['guide_sha256']['bank'] == OLD_BANK_SHA256
        and len(old_guide['gaussian_components']) == 56, 'Wrong fixed previous guide')
    floor = np.asarray(old_plan['fitting_rule']['latent_geometric_floor'])
    expected_sources = partition_result['source_sha256']
    buffers, populations = [], []
    for index, job in enumerate(sorted(protocol['jobs'], key=lambda j: (j['arm'], j['id']))):
        arm, name, n = job['arm'], job['id'], job['samples']
        require(n == 16384, 'Unequal pilot populations require a revised pooling rule')
        terminal = next(j for j in status['jobs'] if (j['arm'], j['id']) == (arm, name))
        require(terminal['status'] == 'complete' and terminal['returncode'] == 0, 'Incomplete source population')
        record = next(p for p in primary['arms'][arm]['populations'] if p['id'] == name)
        require((record['seed'], record['samples'], record['raw_output']) == (job['seed'], n, terminal['output']), 'Source population binding differs')
        current = read(ledger.bind(campaign / arm / 'provenance/region.json', primary['region_sha256']))
        validate_regions(current, reference)
        source_guide_path = ledger.bind(campaign / arm / 'provenance/importance-guide.json')
        require(sha(source_guide_path) == old_plan['guide_sha256'][arm], 'Actual source guide differs')
        archive = ledger.bind(comparison / record['records'], record['records_sha256'])
        require(expected_sources[str(archive)] == record['records_sha256'], 'Partition used another numeric archive')
        with np.load(archive, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        raw = ledger.bind(Path(job['directory']) / 'samples.jsonl', record['raw_output']['samples_sha256'])
        require(expected_sources[str(raw)] == record['raw_output']['samples_sha256'], 'Partition used another pose stream')
        with raw.open() as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == n and [r['draw'] for r in rows] == list(range(n)), 'Original attempted rows changed')
        u = np.asarray([r['latent'] for r in rows], float)
        logq = np.asarray([r['log_proposal_density'] for r in rows], float)
        original_q = np.asarray([r['q'] for r in rows], float)
        require(u.shape == (n, 6) and np.isfinite(u).all() and np.isfinite(logq).all(), 'Invalid chart or original density')
        valid = np.isfinite(arrays['z'])
        require(np.array_equal(valid, [r['log_importance_weight'] is not None for r in rows]), 'Saved zero rows changed')
        require(np.array_equal(arrays['z'][valid], [r['log_importance_weight'] for r in rows if r['log_importance_weight'] is not None]), 'Original physical weights changed')
        poses = [r['pose'] for r in rows]
        old_radius = np.linalg.norm(chart_coordinates(poses, reference), axis=1)
        captured = np.asarray([r['capture_valid'] for r in rows], bool)
        in_r5 = r5_contains(old_radius, original_q, captured)
        masks = partition_masks(valid, arrays['native'], in_r5)
        masks[STRATA[2]] = valid & arrays['contact'] & ~arrays['native']
        class_id = np.full(n, -2, np.int8)
        class_id[valid] = -1
        for stratum_index, stratum in enumerate(STRATA):
            class_id[masks[stratum]] = stratum_index
            expected = (partition_result['arms'][arm]['estimates'][stratum] if stratum_index < 2
                        else primary['arms'][arm]['estimates'][stratum])
            expected = next(p for p in expected['populations'] if p['id'] == name)
            logs = arrays['z'][masks[stratum]]
            require(len(logs) == expected['nonzero'] and expected['draws'] == n
                and abs(float(logsumexp(logs) - math.log(n)) - expected['log_Qz']) < 2e-10,
                'Exact saved stratum physical mass changed: ' + stratum)
        require(np.isfinite(arrays['pairs'][valid]).all()
            and np.max(abs(logsumexp(arrays['pairs'][valid], axis=1) - math.log(2) - arrays['z'][valid])) < 2e-10,
            'Saved independent-cloud mean differs')
        # Verify full normalized source density, including exterior Gaussian support.
        density_error = float(np.max(abs(log_proposal(u, read(source_guide_path)) - logq)))
        require(density_error < 2e-8, 'Saved source density changed')
        populations.append({'arm': arm, 'id': name, 'seed': job['seed'], 'samples': n,
            'source_population_index': index, 'training_counts': {s: int(masks[s].sum()) for s in STRATA},
            'invalid_or_exterior_attempts': int((~valid).sum()), 'exterior_attempts': int((~arrays['support']).sum()),
            'valid_unbound_or_other_attempts': int((class_id == -1).sum()),
            'maximum_original_density_error': density_error,
            'records_sha256': record['records_sha256'], 'raw_samples_sha256': record['raw_output']['samples_sha256']})
        buffers.append({'u': u, 'population': np.full(n, index, np.int16), 'draw': np.arange(n),
            'source_n': np.full(n, n, np.int32), 'class_id': class_id,
            'log_weight': arrays['z'], 'h': arrays['h'], 'pairs': arrays['pairs'], 'log_q': logq,
            'native': arrays['native'], 'contact': arrays['contact'], 'support': arrays['support'],
            'old_r5_radius': old_radius, 'original_q': original_q, 'capture': captured})
        print(f'loaded frozen training population {arm}/{name}: {populations[-1]["training_counts"]}', flush=True)
    data = {key: np.concatenate([buffer[key] for buffer in buffers]) for key in buffers[0]}
    ledger.recheck()
    return data, populations, floor, old_guide, ledger


def render_report(result):
    lines = ['# Contact proposal refinement: training scores', '', SCOPE, '',
        'Every candidate retains 50% uniform R4, 25% previous bank Gaussians and 25% new Gaussians. '
        'The new Gaussian group is divided equally among the three named strata. '
        'The exact globally normalized density obeys q_new ≥ 0.5 q_old_bank, including outside R4. '
        'Floor factors multiply the geometric floor covariance only; empirical full covariance is unchanged.', '',
        'Eight complete pilot populations supply 131,072 original attempts. Each holdout uses seven population anchors per stratum. '
        'Covariance neighbors are same-stratum distinct original-u poses; fitting weights use original log weight minus log source N, '
        'with normalized cap 1/16. Only fitting weights are capped. The weighted-mean option uses the capped neighborhood mean.', '',
        '| Candidate | R5 M2 ratio | Complement M2 ratio | No-entry M2 ratio | Worst stratum |',
        '|---|---:|---:|---:|---:|']
    for name in result['training_ranking']:
        summary = result['candidate_summaries'][name]
        values = [summary['strata'][s]['pooled']['paired_cloud_secondmoment_ratio_to_old_bank'] for s in STRATA]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.4f}' for v in values + [max(values)]) + ' |')
    lines += ['', 'Ratios below one predict smaller physical second moments than the unchanged previous bank on the observed held-out poses. '
        'The pooling averages different fold proposals and is not a final-guide variance estimate. '
        'Selection minimizes the largest of the three ratios so an abundant native subset cannot hide deterioration of no-entry or the native complement.', '',
        '| Training stratum | Original importance ESS | Original largest draw |', '|---|---:|---:|']
    for stratum in STRATA:
        score = result['scores']['pooled']['classes'][stratum]
        lines.append(f'| {stratum} | {score["original_weight_ESS"]:.2f} | {score["original_largest_weight"]:.2%} |')
    lines += ['', 'All per-population and per-arm scores, predictive density gains, original mass estimates, '
        'cloud-product contribution ESS and largest terms remain in cross-validation.json. '
        'A favorable ranking is a design prospect requiring separately frozen fresh physical validation.']
    return '\n'.join(lines) + '\n'


def run(campaign, comparison, partition, old_preparation, out):
    start = time.monotonic()
    require(not out.exists(), 'Refuse to overwrite a scoring artifact')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools'),
        '-p', 'test_score_contact_refinements.py', '-v'], capture_output=True, text=True)
    require(tests.returncode == 0, tests.stdout + tests.stderr)
    out.mkdir(parents=True)
    (out / 'tests.txt').write_text(tests.stdout + tests.stderr)
    write(out / 'declared-candidates.json', {'created': datetime.now(timezone.utc).isoformat(),
        'candidates': CANDIDATES, 'strata': STRATA, 'selection': 'minimax pooled held-out secondmoment ratio across three named strata',
        'scope': SCOPE, 'physical_samples_authorized': False})
    data, populations, floor, old_guide, ledger = load_pilot(campaign, comparison, partition, old_preparation)
    np.savez_compressed(out / 'training-poses.npz', **data)
    write(out / 'training-metadata.json', {'schema': 'contact-refinement-training-data-v1',
        'all_original_attempts_retained': True, 'total_unconditional_draws': len(data['u']),
        'class_ids': {'-2': 'invalid_or_exterior', '-1': 'valid_unbound_or_other', **{str(i): s for i, s in enumerate(STRATA)}},
        'populations': populations, 'latent_geometric_floor': floor,
        'region_sha256': old_guide['region_sha256'], 'old_bank_sha256': OLD_BANK_SHA256,
        'partition_sha256': PARTITION_SHA256, 'source_sha256': ledger.files, 'scope': SCOPE})
    result, densities = cross_validate(data, populations, floor, old_guide)
    write(out / 'cross-validation.json', result)
    np.savez_compressed(out / 'heldout-log-densities.npz', **densities)
    (out / 'report.md').write_text(render_report(result))
    closure = local_source_closure([Path(__file__), ROOT / 'tools/test_score_contact_refinements.py'])
    for filename, source in closure.items():
        destination = out / 'source' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        ledger.bind(source)
    ledger.recheck()
    write(out / 'source-closure.json', {'source_sha256': ledger.files, 'scope': SCOPE})
    write(out / 'status.json', {'schema': 'contact-refinement-score-v1', 'complete': True,
        'new_physical_samples': 0, 'classifiers_rerun': 0, 'audits_replayed': 0,
        'candidates_scored': len(CANDIDATES), 'whole_population_folds': len(populations),
        'training_ranking': result['training_ranking'], 'scoring_wall_seconds': time.monotonic() - start,
        'training_archive_sha256': sha(out / 'training-poses.npz'),
        'scope': SCOPE})
    write(out / 'freeze.json', {'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'out': str(out), 'ranking': result['training_ranking'], 'wall_seconds': time.monotonic() - start}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs/contact-bank-pilot-20260922')
    parser.add_argument('--comparison', type=Path, default=ROOT / 'runs/contact-bank-pilot-comparison-20260922')
    parser.add_argument('--partition', type=Path, default=ROOT / 'runs/contact-bank-reference-partition-20260922')
    parser.add_argument('--old-preparation', type=Path, default=ROOT / 'runs/reference-contact-bank-preparation-20260922')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/contact-refinement-score-20260922')
    args = parser.parse_args()
    run(*(getattr(args, key).resolve() for key in ('campaign', 'comparison', 'partition', 'old_preparation', 'out')))
