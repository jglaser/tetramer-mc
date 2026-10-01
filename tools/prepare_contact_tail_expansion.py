#!/usr/bin/env python3
"""Freeze eight tail-centered R4 integration components; never launch physics.

Only bank/protected r00/r01 coordinates fit the new components. Previously
inspected r02/r03 are retrospective holdouts, opened after the model freeze.
Native labels belong to integration-guide design, not blind assembly.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal

from analyze_latent_region import LatentImportanceGuide
from diagnose_fitted_kernel_shear import (bind, combine_group, evaluate_group,
    group_masks, load_rows, read, require, sha, write_new)
from prepare_contact_bank_guides import (capped_weights, latent_geometric_floor,
    log_proposal, nearest_distinct)
from prepare_native_confirmation_atlas import weighted_fit
from prepare_shoulder_docking_benchmark import local_dependencies
from run_protected_guide_validation import declared_seeds

SCHEMA = 'contact-tail-eight-component-preparation-v1'
PINS = dict(
    analysis='471f61cacf7253bf5746d8fc04765ea71ddfacfb07e205e12e50bf43742737dd',
    guide='867a3be62ec834ee96cf5cd3149dfa82f06158986b892f2dfccd83bbe91d1f48',
    region='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    shape='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9')
FIT = dict(neighbors=128, weight_cap=1/16, new_gaussian_fraction=.2,
    regions=[dict(name='native_complement', class_id=1, orthant=55),
             dict(name='contact_no_native_entry', class_id=2, orthant=63)],
    anchor='largest paired-cloud physical-second-moment contribution against frozen84',
    neighbors_scope='same class, all four training populations; no orthant truncation',
    covariance='capped-weight covariance about selected anchor plus original geometric floor',
    duplicates='distinct exact u coordinates only, stable population/draw tie order')
PILOT_SEEDS = [610011001 + 1009*i for i in range(8)]
SCOPE = ('Native-label-aware integration proposal, separate from blind assembly. '
    'No new physical draws, classifiers, geometry calls, previous audit replay, '
    'equilibrium assertion, or modification of historical convergence gates. '
    'Holdout summaries were inspected historically; evaluation is retrospective. '
    'Full original R4, complete native classifier, Haar Jacobian, invalid zeros '
    'and unconditional attempted denominators remain unchanged.')


def expand_guide(baseline, components, fraction=.2):
    require(0 <= fraction < 1 and components, 'Invalid expansion allocation')
    result = copy.deepcopy(baseline)
    total = sum(c['weight'] for c in result['gaussian_components'])
    for component in result['gaussian_components']:
        component['weight'] *= (1-fraction)/total
    for component in components:
        entry = copy.deepcopy(component)
        entry['weight'] = fraction/len(components)
        result['gaussian_components'].append(entry)
    return result


def allocation_control(baseline, anchors, fraction=.2):
    """Same new allocation, assigned to existing best-posterior components."""
    result = copy.deepcopy(baseline)
    components = result['gaussian_components']
    original = np.asarray([c['weight'] for c in components], float)
    original /= original.sum()
    extra = np.zeros(len(components)); assignments = []
    for anchor in anchors:
        score = [math.log(w)+float(multivariate_normal.logpdf(anchor,
            mean=c['mean'], cov=c['covariance'])) for c, w in zip(components, original)]
        index = int(np.argmax(score)); extra[index] += 1/len(anchors)
        assignments.append(index)
    for c, w in zip(components, (1-fraction)*original+fraction*extra):
        c['weight'] = float(w)
    return result, assignments


def independently_score(u, guide):
    """SciPy normalized densities, independent of the existing Cholesky scorer."""
    u = np.asarray(u)
    alpha = guide['defensive_uniform_shell_probability']
    volume = math.pi**3 * 4**6 / 6
    answer = np.where(np.sum(u*u, axis=1) <= 16, math.log(alpha/volume), -np.inf)
    total = sum(c['weight'] for c in guide['gaussian_components'])
    for c in guide['gaussian_components']:
        answer = np.logaddexp(answer, math.log((1-alpha)*c['weight']/total)
            + multivariate_normal.logpdf(u, mean=c['mean'], cov=c['covariance']))
    return answer


def validate_guide(guide, baseline, anchors):
    components = guide['gaussian_components']
    require(guide['region_sha256'] == PINS['region'] and
            guide['defensive_uniform_shell_probability'] == .5, 'Target/support changed')
    require(all(c['weight'] > 0 for c in components) and
            abs(sum(c['weight'] for c in components)-1) < 1e-12, 'Unnormalized weights')
    for c in components: np.linalg.cholesky(c['covariance'])
    rng = np.random.default_rng(610010001)
    probes = np.vstack([np.asarray(anchors), np.zeros((1, 6)),
                        rng.normal(size=(256, 6))*2, rng.normal(size=(64, 6))*8])
    actual = log_proposal(probes, guide)
    independent = independently_score(probes, guide)
    error = float(np.max(np.abs(actual-independent)))
    require(error < 1e-7, 'Independent normalized-density disagreement')
    observer = LatentImportanceGuide(guide, PINS['region'])
    audited = observer.log_density(probes, np.sum(probes*probes, axis=1) <= 16,
                                  math.log(math.pi**3 * 4**6 / 6))
    audit_error = float(np.max(np.abs(actual-audited)))
    require(audit_error < 1e-7, 'Existing independent audit-density disagreement')
    lower_margin = float(np.min(actual-log_proposal(probes, baseline)-math.log(.8)))
    require(lower_margin >= -1e-9, 'Defensive old-guide lower bound violated')
    return dict(components=len(components), normalized_component_mass=1., uniform_mass=.5,
        untruncated_gaussian_mass=.5, support='uniform R4 plus untruncated Gaussians',
        independent_density_max_abs_log_error=error, probe_count=len(probes),
        existing_auditor_max_abs_log_error=audit_error,
        old_proposal_retention_lower_bound=.8, smallest_observed_log_bound_margin=lower_margin,
        normalization='Each Gaussian integrates to one in all R6; uniform integrates to one in R4; sum=.5+.5=1',
        physical_reference_tests_reused=True, new_Rust_execution=False)


def fit_components(data, populations, baseline, floor):
    components, details = [], []
    eligible = {g['class_id']: np.flatnonzero(data['class_id'] == g['class_id']) for g in FIT['regions']}
    trees = {key: cKDTree(data['u'][indices]) for key, indices in eligible.items()}
    for population, record in enumerate(populations):
        for group in FIT['regions']:
            rows = np.flatnonzero((data['population'] == population)
                & (data['class_id'] == group['class_id']) & (data['bin_orthant'] == group['orthant']))
            require(len(rows) > 0, 'Missing prescribed training region')
            score = data['pairs'][rows].sum(axis=1)+data['log_q'][rows]-log_proposal(data['u'][rows], baseline)
            anchor_index = int(rows[np.argmax(score)]); anchor = data['u'][anchor_index]
            indices = nearest_distinct(data, eligible[group['class_id']], trees[group['class_id']],
                                       anchor, FIT['neighbors'])
            ns = np.asarray([populations[int(p)]['samples'] for p in data['population'][indices]])
            weights, cap = capped_weights(data['z'][indices]-np.log(ns), FIT['weight_cap'])
            keep = weights > 0
            fit = weighted_fit(data['u'][indices][keep], np.log(weights[keep]), floor)
            offset = fit['mean']-anchor
            covariance = fit['covariance']+np.outer(offset, offset)
            covariance = .5*(covariance+covariance.T)
            np.linalg.cholesky(covariance)
            component = dict(weight=1/8, mean=anchor.tolist(), covariance=covariance.tolist())
            components.append(component)
            details.append(dict(group=group, source=dict(arm=record['arm'], id=record['id'],
                seed=record['seed'], draw=int(data['draw'][anchor_index])),
                training_group_rows=len(rows), selected_log_paired_moment=float(score.max()),
                anchor=anchor.tolist(), neighborhood_weight_diagnostic=cap,
                covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
                weighted_neighbor_mean=fit['mean'].tolist(),
                largest_neighbor_distance_u=float(np.linalg.norm(data['u'][indices]-anchor, axis=1).max()),
                neighbors=[dict(arm=populations[int(data['population'][i])]['arm'],
                    id=populations[int(data['population'][i])]['id'], draw=int(data['draw'][i]),
                    weight=float(w)) for i, w in zip(indices, weights)]))
    require(len(components) == 8, 'Expected eight fixed components')
    return components, details


def prepare(out, repository):
    out, repository = Path(out).resolve(), Path(repository).resolve()
    require(not out.exists(), 'Fresh output directory required; no overwrite')
    bindings = {}; comparison = repository/'runs/protected-guide-validation-20260923/comparison'
    analysis = read(bind(comparison/'analysis.json', bindings, PINS['analysis']))
    require(analysis['complete'] and analysis['region_sha256'] == PINS['region']
            and analysis['shape_sha256'] == PINS['shape'], 'Physical target differs')
    preparation = repository/'runs/protected-guide-preparation-20260923'
    guide_path = bind(preparation/'guide-protected.json', bindings, PINS['guide'])
    region_path = bind(preparation/'region.json', bindings, PINS['region'])
    datasets = []
    for arm in ('bank', 'protected'):
        for row in sorted(analysis['arms'][arm]['populations'], key=lambda r: r['id']):
            require(row['samples'] == 1048576, 'Original population size differs')
            path = bind(comparison/row['records'], bindings, row['records_sha256'])
            datasets.append(dict(arm=arm, id=row['id'], seed=row['seed'], samples=row['samples'],
                role='training' if row['id'] in ('r00', 'r01') else 'heldout',
                records=str(path), records_sha256=row['records_sha256']))
    require(len(datasets) == 8 and len({r['seed'] for r in datasets}) == 8, 'Repeated datasets')
    sources = local_dependencies([Path(__file__)])
    for source in sources.values(): bind(source, bindings)
    out.mkdir(parents=True); (out/'source').mkdir()
    for name, source in sources.items(): shutil.copy2(source, out/'source'/name)
    shutil.copy2(guide_path, out/'baseline.json'); shutil.copy2(region_path, out/'region.json')
    plan = dict(schema=SCHEMA, scope=SCOPE, fit=FIT, datasets=datasets,
        source_and_input_sha256=bindings, baseline_sha256=sha(out/'baseline.json'),
        physical_target=dict(region_sha256=PINS['region'], shape_sha256=PINS['shape'],
            depletant_radius_A=1.5, activity_A_minus3=.035,
            native_definition_sha256=analysis['native_definition']['definition_sha256']),
        python=sys.executable, python_sha256=sha(sys.executable), fit_calls=1,
        heldout_rows_before_model_freeze=0, prospective_pilot_draws=131072,
        pilot_launch=False, no_hyperparameter_search=True)
    write_new(out/'plan.json', plan)
    training = [r for r in datasets if r['role'] == 'training']
    pieces = []; training_counts = []
    start = time.monotonic()
    for index, record in enumerate(training):
        arrays = load_rows(record); valid = np.isfinite(arrays['z'])
        piece = {key: arrays[key][valid] for key in
                 ('u', 'z', 'pairs', 'log_q', 'class_id', 'bin_orthant', 'draw')}
        piece['population'] = np.full(valid.sum(), index, int)
        pieces.append(piece)
        training_counts.append(dict(arm=record['arm'], id=record['id'], attempts=record['samples'],
            nonzero_rows=int(valid.sum()), invalid_zeros=int((~valid).sum())))
        del arrays
    data = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    floor, raw_floor = latent_geometric_floor(read(out/'region.json')['gaussian_chart'])
    baseline = read(out/'baseline.json')
    components, details = fit_components(data, training, baseline, floor)
    expanded = expand_guide(baseline, components, FIT['new_gaussian_fraction'])
    allocation, assignments = allocation_control(baseline, [c['mean'] for c in components])
    write_new(out/'expanded-guide.json', expanded)
    write_new(out/'allocation-control.json', allocation)
    write_new(out/'fit.json', dict(components=details, counts=training_counts,
        training_attempts=sum(r['samples'] for r in training), holdout_rows_read=0,
        floor_u=floor.tolist(), floor_raw_chart=raw_floor.tolist(),
        allocation_control_assignments=assignments, fit_wall_seconds=time.monotonic()-start))
    write_new(out/'guide-validation.json', dict(expanded=validate_guide(expanded, baseline, [c['mean'] for c in components]),
        allocation_control=validate_guide(allocation, baseline, [c['mean'] for c in components])))
    write_new(out/'model-freeze.json', dict(files={name: sha(out/name) for name in
        ('plan.json', 'baseline.json', 'expanded-guide.json', 'allocation-control.json', 'fit.json', 'guide-validation.json')},
        heldout_rows_read=0, full_R4_and_native_definition_unchanged=True))
    del data, pieces
    print('Candidate and allocation-only control frozen; now reading retrospective holdouts', flush=True)
    populations = []
    for record in [r for r in datasets if r['role'] == 'heldout']:
        arrays = load_rows(record); valid = np.isfinite(arrays['z'])
        densities = {}
        for name, guide in [('baseline', baseline), ('expanded', expanded), ('allocation', allocation)]:
            values = np.zeros(record['samples']); values[valid] = log_proposal(arrays['u'][valid], guide)
            densities[name] = values
        controls = {}
        for name in ('expanded', 'allocation'):
            controls[name] = {key: evaluate_group(arrays, densities['baseline'], densities[name], mask)
                              for key, mask in group_masks(arrays)}
        populations.append(dict(arm=record['arm'], id=record['id'], role=record['role'],
            attempts=record['samples'], controls=controls))
        print('Retrospective evaluation complete', record['arm'], record['id'], flush=True)
    combined = {}
    for arm in ('bank', 'protected'):
        rows = [r for r in populations if r['arm'] == arm]
        combined[arm] = {control: {key: combine_group([r['controls'][control][key] for r in rows])
            for key in rows[0]['controls'][control]} for control in ('expanded', 'allocation')}
    write_new(out/'retrospective-analysis.json', dict(schema=SCHEMA, complete=True,
        scope=SCOPE, populations=populations, combined_within_source_only=combined,
        attempts=sum(r['samples'] for r in datasets if r['role']=='heldout'),
        model_freeze_sha256=sha(out/'model-freeze.json'), physical_samples_launched=0))
    # Prepare commands only; a fresh audited dispatcher/reference receipt is required before execution.
    protocol_path = repository/'runs/protected-guide-validation-20260923/protocol.json'
    protocol = read(protocol_path); base = protocol['jobs'][0]['command']
    seed_paths = sorted((repository/'runs').rglob('protocol.json'))
    previous = set(); inventory = {}
    for path in seed_paths:
        previous.update(declared_seeds(read(path))); inventory[str(path.resolve())] = sha(path)
    previous.update(r['seed'] for r in datasets)
    require(not previous.intersection(PILOT_SEEDS), 'Proposed pilot seed collides with existing protocol')
    jobs = []
    for arm in ('baseline', 'expanded'):
        for index in range(4):
            seed = PILOT_SEEDS[len(jobs)]; command = list(base)
            changes = {'--importance-guide': str(out/('baseline.json' if arm=='baseline' else 'expanded-guide.json')),
                '--region': str(out/'region.json'), '--out': str(out/'pilot'/arm/f'r{index:02}'),
                '--samples': '16384', '--seed': str(seed)}
            for flag, value in changes.items(): command[command.index(flag)+1] = value
            jobs.append(dict(arm=arm, id=f'r{index:02}', seed=seed, samples=16384, command=command))
    write_new(out/'prospective-pilot-plan.json', dict(schema='contact-tail-fresh-pilot-preparation-v1',
        preparation_only=True, physical_jobs_launched=0, jobs=jobs, total_attempted_draws=131072,
        maximum_physical_jobs=2, maximum_overall_physical_jobs=8, maximum_total_workers=32,
        cloud_replicates=2, auxiliary_intensity_ratio=128, defensive_uniform_probability=.5,
        seeds=PILOT_SEEDS, earlier_protocol_sha256=inventory, base_protocol_sha256=sha(protocol_path),
        source_binary_sha256=protocol['binary_sha256'], source_bundle_sha256=protocol['source_bundle_sha256'],
        model_freeze_sha256=sha(out/'model-freeze.json'),
        physical_target=plan['physical_target'], fixed_region_and_all_strata=True,
        estimator='J times arithmetic mean of two independent Poisson factors / complete q; invalid attempts zero; divide by original N',
        retain_all_attempts=True, retry_or_extension=False, old_samples_pooled=False,
        prerequisite='Bind candidate Rust density reconstruction and sphere reference, archived config/scaffold/source and complete classifier; authenticated failure-draining dispatcher and independent output audit before launch',
        analysis='Separate four-population linear masses; all original strata and cloud/contribution diagnostics; archived baseline comparisons separate, not pooled',
        scope='Prospective feasibility pilot, not replacement for frozen convergence confirmation; no thermodynamic assembly conclusion'))
    write_new(out/'completion.json', dict(schema=SCHEMA, complete=True, physical_jobs_launched=0,
        elapsed_seconds=time.monotonic()-start, files={str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob('*')) if p.is_file()}))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(prepare(args.out, args.repository))


if __name__ == '__main__': main()
