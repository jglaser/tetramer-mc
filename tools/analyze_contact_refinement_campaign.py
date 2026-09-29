#!/usr/bin/env python3
"""Analyze every declared slot/replicate of the paired refinement campaign.

No native labels are read. Autocorrelation, split-Rhat and covariance agreement
are diagnostics, not claims of equilibrium or complete basin coverage. Timings
are observed single-thread elapsed seconds, not OS-measured process CPU time.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import solve_triangular

from fit_depletion_contact_atlas import (FitOptions, chart_coordinates,
    exploration_metrics, rolling_covariance)
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_refinement_campaign import (ARMS, read, require, runtime_summary,
    sha, validate, verify_output, write)


def summary(values):
    data = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return dict(count=len(data), minimum=float(data.min()) if len(data) else None,
                median=float(np.median(data)) if len(data) else None,
                maximum=float(data.max()) if len(data) else None,
                quartiles=np.quantile(data, [.25, .75]).tolist() if len(data) else None)


def scatter(values):
    shifted = np.asarray(values) - values[0]
    centered = shifted - shifted.mean(axis=0)
    return centered.T @ centered / len(values)


def drift(first, second):
    s1, s2 = scatter(first), scatter(second)
    pooled = (s1 + s2) / 2
    eigenvalues = np.linalg.eigvalsh(pooled)
    floor = max(1e-14, float(eigenvalues.max()) * 1e-12)
    inverse = np.linalg.inv(pooled + floor * np.eye(pooled.shape[0]))
    delta = first.mean(axis=0) - second.mean(axis=0)
    denominator = .5 * (np.linalg.norm(s1) + np.linalg.norm(s2))
    return dict(mean_difference=delta.tolist(), mean_distance_baseline_units=float(np.linalg.norm(delta)),
                mean_distance_pooled_scatter_units=float(np.sqrt(max(0., delta @ inverse @ delta))),
                pooled_scatter_rank=int(np.linalg.matrix_rank(pooled)),
                mean_distance_covariance_floor=floor,
                covariance_relative_frobenius=float(np.linalg.norm(s1-s2) / denominator) if denominator else None,
                covariance_eigenvalues_first=np.linalg.eigvalsh(s1).tolist(),
                covariance_eigenvalues_second=np.linalg.eigvalsh(s2).tolist())


def replicate_diagnostics(chains):
    require(len(chains) >= 2, 'At least two independent streams needed for replicate diagnostics')
    lengths = {len(chain) for chain in chains}
    require(len(lengths) == 1, 'Unequal retained allocations')
    count = next(iter(lengths))
    combined = np.concatenate(chains)
    centered = combined - combined.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    half = count // 2
    split = np.stack([part @ axes.T for chain in chains
                      for part in (chain[:half], chain[-half:])])
    require(half > 1, 'Too few observations for split-chain diagnostics')
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * split.mean(axis=1).var(axis=0, ddof=1)
    rhat = []
    for w, b in zip(within, between):
        rhat.append(float(np.sqrt(((half-1)*w/half + b/half)/w)) if w > 0 else None)
    pairwise = [dict(replicates=[a, b], **drift(chains[a], chains[b]))
                for a in range(len(chains)) for b in range(a+1, len(chains))]
    return dict(replicates=len(chains), retained_per_replicate=count,
                pooled_pca_split_rhat=rhat,
                maximum_pooled_pca_split_rhat=max((v for v in rhat if v is not None), default=None),
                undefined_split_rhat_axes=sum(v is None for v in rhat),
                rhat_method='Ordinary split-chain variance ratio in pooled PCA directions, not rank normalized. '
                    'Identical starts and missed common modes can make this optimistic. Constants are undefined.',
                pairwise=pairwise,
                maximum_mean_distance_pooled_scatter_units=max(row['mean_distance_pooled_scatter_units'] for row in pairwise),
                maximum_covariance_relative_frobenius=max((row['covariance_relative_frobenius'] for row in pairwise
                                                           if row['covariance_relative_frobenius'] is not None), default=None))


def collect(campaign, options, allow_partial):
    campaign = Path(campaign).resolve()
    plan = validate(campaign)
    status = read(campaign / 'status.json')
    complete = (campaign / 'completion.json').exists() and all(j['status'] == 'complete' for j in status['jobs'])
    require(complete or allow_partial, 'Campaign incomplete; use --allow-partial only for explicitly partial diagnostics')
    source = read(campaign / 'inputs/source-fit/fit-metrics.json')
    originals = {row['global_slot']: row for row in source['slots']}
    ell = float(source['angular_length'])
    times = {job['id']: {row['slot']: row['wall_seconds'] for row in job['slot_reports']}
             for job in runtime_summary(campaign, status['jobs'])['jobs']}
    rows, coordinates = [], {}
    hashes = {'plan.json': sha(campaign / 'plan.json'), 'freeze.json': sha(campaign / 'freeze.json'),
              'status.json': sha(campaign / 'status.json')}
    missing = []
    for job in status['jobs']:
        if job['status'] != 'complete':
            missing.append(dict(id=job['id'], status=job['status']))
            continue
        verify_output(job, plan)
        directory = Path(job['directory'])
        for name in ('manifest.json', 'discovery.json'):
            hashes[str((directory/name).relative_to(campaign))] = sha(directory/name)
        result = read(directory / 'discovery.json')
        require(len(result['slots']) == job['slots'], 'Lost source slots')
        for slot, mapping in zip(result['slots'], job['slot_map']):
            require(slot['slot'] == mapping['local_slot'], 'Discovery source mapping changed')
            original = originals[mapping['original_global_slot']]
            reference = original['reference_pose']
            values, seam = chart_coordinates(reference, [sample['pose'] for sample in slot['refinement_samples']], ell)
            baseline = rolling_covariance(np.array(original['refinement_contact_lever']), ell, FitOptions())
            diagnostic = exploration_metrics(values, baseline, slot, options)
            lower = np.linalg.cholesky(baseline)
            whitened = solve_triangular(lower, values.T, lower=True).T
            production = slot['refinement_production_counts']
            hard_valid = production['attempted'] - production['hard_rejected'] - production['outside_ball']
            elapsed = times[job['id']].get(slot['slot'])
            require(elapsed is not None and elapsed > 0, 'Missing complete slot runtime')
            minimum_ess = diagnostic['minimum_retained_pca_ess_estimate']
            key = (job['arm'], mapping['original_global_slot'], job['replicate'])
            require(key not in coordinates, 'Duplicate independent stream')
            coordinates[key] = whitened
            protocol = slot['refinement_protocol']
            row = dict(job=job['id'], arm=job['arm'], replicate=job['replicate'], mapping=mapping,
                       retained=len(values), warmup_counts=slot['refinement_warmup_counts'],
                       production_counts=production, production_hard_valid=hard_valid,
                       production_acceptance=production['accepted']/production['attempted'],
                       production_hard_valid_fraction=hard_valid/production['attempted'],
                       accepted_given_hard_valid=production['accepted']/hard_valid if hard_valid else None,
                       distinct_retained_poses=diagnostic['unique_physical_chart_poses'],
                       minimum_reference_quaternion_scalar=seam,
                       final_scale=protocol['final_scale'],
                       adaptation_lower_bound_windows=sum(w['at_lower_bound'] for w in protocol['updates']),
                       adaptation_upper_bound_windows=sum(w['at_upper_bound'] for w in protocol['updates']),
                       frozen_translation_std_a=protocol['frozen_memory_config']['local_translation_std_A'],
                       frozen_angle_std_degrees=protocol['frozen_memory_config']['local_small_angle_std_degrees'],
                       slot_elapsed_seconds=elapsed,
                       minimum_pca_ess_per_elapsed_second=minimum_ess/elapsed,
                       chronological_half_drift=drift(whitened[:len(values)//2], whitened[len(values)//2:]),
                       exploration=diagnostic)
            rows.append(row)
    cross = []
    for arm in ARMS:
        for source_slot in [row['global_slot'] for row in source['slots'][:plan['source_slots']]]:
            keys = [(arm, source_slot, replicate) for replicate in range(plan['replicates'])]
            if len(keys) >= 2 and all(key in coordinates for key in keys):
                cross.append(dict(arm=arm, original_global_slot=source_slot,
                                  **replicate_diagnostics([coordinates[key] for key in keys])))
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row['arm'] == arm]
        replicate_rows = [row for row in cross if row['arm'] == arm]
        attempts = sum(row['production_counts']['attempted'] for row in selected)
        accepted = sum(row['production_counts']['accepted'] for row in selected)
        elapsed = sum(row['slot_elapsed_seconds'] for row in selected)
        total_ess = sum(row['exploration']['minimum_retained_pca_ess_estimate'] for row in selected)
        arms[arm] = dict(completed_source_replicates=len(selected), expected_source_replicates=plan['source_slots']*plan['replicates'],
                        production_attempts=attempts, production_accepted=accepted,
                        production_acceptance=accepted/attempts if attempts else None,
                        production_hard_valid_fraction=sum(row['production_hard_valid'] for row in selected)/attempts if attempts else None,
                        distinct_retained_poses=summary([row['distinct_retained_poses'] for row in selected]),
                        zero_empirical_rank=sum(row['exploration']['empirical_rank'] == 0 for row in selected),
                        full_empirical_rank=sum(row['exploration']['empirical_rank'] == 6 for row in selected),
                        minimum_pca_ess=summary([row['exploration']['minimum_retained_pca_ess_estimate'] for row in selected]),
                        minimum_pca_ess_per_elapsed_second=summary([row['minimum_pca_ess_per_elapsed_second'] for row in selected]),
                        total_slot_elapsed_seconds=elapsed,
                        summed_minimum_pca_ess_per_total_elapsed_second=total_ess/elapsed if elapsed else None,
                        gate_passing=sum(row['exploration']['exploration_gate_passed'] for row in selected),
                        diagnostic_minimum_pca_ess_at_least_20=sum(row['exploration']['minimum_retained_pca_ess_estimate'] >= 20 for row in selected),
                        final_scale=summary([row['final_scale'] for row in selected]),
                        replicate_sources=len(replicate_rows),
                        maximum_split_rhat=summary([row['maximum_pooled_pca_split_rhat'] for row in replicate_rows]),
                        replicate_mean_distance=summary([row['maximum_mean_distance_pooled_scatter_units'] for row in replicate_rows]),
                        replicate_covariance_difference=summary([row['maximum_covariance_relative_frobenius'] for row in replicate_rows]))
    return dict(schema='paired-contact-refinement-assessment-v1', complete=complete,
                native_label_inputs=0, campaign=str(campaign), source_sha256=hashes,
                options=asdict(options), source_slots=plan['source_slots'], replicates=plan['replicates'],
                descriptive_ess_threshold=dict(value=20., used_as_fit_gate=False),
                rows=rows, cross_replicate=cross, arms=arms, missing_jobs=missing,
                timing='Observed single-thread slot elapsed seconds includes validation and discarded warmup. '
                       'Not measured process CPU seconds; no exact CPU-efficiency claim.',
                scope='All declared source slots and rejected observations are retained. '
                      'Fit adequacy gates and within-trace ESS/Rhat are diagnostics, not stationarity, '
                      'basin-mass, important-mode coverage or physical assembly guarantees.')


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    specs = [('distinct_retained_poses', 'Distinct retained poses'),
             ('minimum_pca_ess_per_elapsed_second', 'Within-trace minimum PCA ESS / elapsed second'),
             ('final_scale', 'Frozen local step multiplier')]
    for axis, (key, title) in zip(axes.flat, specs):
        for index, arm in enumerate(ARMS):
            data = [row[key] for row in report['rows'] if row['arm'] == arm]
            axis.scatter(np.full(len(data), index), data, s=9, alpha=.45)
            if data:
                axis.plot([index-.18, index+.18], [np.median(data)]*2, color='black', lw=2)
        axis.set(xticks=[0, 1], xticklabels=['Legacy', 'Pivot + warmup'], title=title)
        axis.grid(axis='y', alpha=.2)
    for axis, (key, title) in zip(list(axes.flat)[3:], [
            ('maximum_covariance_relative_frobenius', 'Cross-replicate covariance difference')]):
        for index, arm in enumerate(ARMS):
            data = [row[key] for row in report['cross_replicate'] if row['arm'] == arm and row[key] is not None]
            axis.scatter(np.full(len(data), index), data, s=9, alpha=.45)
            if data:
                axis.plot([index-.18, index+.18], [np.median(data)]*2, color='black', lw=2)
        axis.set(xticks=[0, 1], xticklabels=['Legacy', 'Pivot + warmup'], title=title)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle(('Complete' if report['complete'] else 'PARTIAL') +
                 ' paired refinement: all declared contact slots; no native selection\n'
                 'Exploration diagnostics only; no mixing or equilibrium claim')
    fig.savefig(out/'comparison.png', dpi=180)
    fig.savefig(out/'comparison.svg')


def markdown(report):
    lines = ['# Paired contact refinement', '',
             '**' + ('Complete allocation.' if report['complete'] else 'Partial allocation; missing jobs remain listed below.') + '**', '',
             '| Arm | Completed chains | Acceptance | Median distinct poses | Median minimum PCA ESS | Full rank | Fit gate passing |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in report['arms'].items():
        acceptance = f"{100*row['production_acceptance']:.2f}%" if row['production_acceptance'] is not None else '—'
        unique, ess = row['distinct_retained_poses']['median'], row['minimum_pca_ess']['median']
        lines.append(f"| {arm} | {row['completed_source_replicates']}/{row['expected_source_replicates']} | {acceptance} | "
                     f"{unique if unique is not None else '—'} | {f'{ess:.2f}' if ess is not None else '—'} | "
                     f"{row['full_empirical_rank']} | {row['gate_passing']} |")
    lines += ['', report['timing'], '', report['scope'], '',
              'Autocorrelation estimates use every fixed-stride observation, including rejected repeats. '
              'Cross-replicate comparisons use the same original pose chart and geometric covariance scaling; '
              'their streams are independent but their starting pose is shared. '
              'Undefined constant modes are retained as diagnostic failures, not replaced by an IID estimate.', '',
              'Every chain, adaptation bound hit, chronological drift, cross-replicate comparison and source hash '
              'is recorded in `assessment.json`. No native classifier enters this report.', '']
    if report['missing_jobs']:
        lines += ['Missing jobs: ' + ', '.join(row['id']+' ('+row['status']+')' for row in report['missing_jobs']), '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--minimum-unique-poses', type=int, default=32)
    p.add_argument('--minimum-empirical-rank', type=int, default=6)
    p.add_argument('--minimum-ess', type=float, default=0.)
    args = p.parse_args()
    out = Path(args.out).resolve()
    require(not out.exists(), 'Fresh analysis output required')
    options = FitOptions(minimum_unique_poses=args.minimum_unique_poses,
                         minimum_empirical_rank=args.minimum_empirical_rank, minimum_ess=args.minimum_ess)
    options.validate()
    report = collect(args.campaign, options, args.allow_partial)
    out.mkdir(parents=True)
    (out/'source').mkdir()
    for name, path in local_dependencies([Path(__file__)]).items():
        shutil.copy2(path, out/'source'/name)
    write(out/'assessment.json', report, exclusive=True)
    (out/'report.md').write_text(markdown(report))
    plot(report, out)
    write(out/'manifest.json', dict(complete=True, campaign_complete=report['complete'],
          files={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}), exclusive=True)
    print(json.dumps(dict(out=str(out), complete=report['complete'], arms=report['arms']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
