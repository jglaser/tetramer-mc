#!/usr/bin/env python3
"""Freeze the declared reference-pocket augmentation; no new physical sampling.

The first bank is immutable. Eight observed native-reference anchors augment its
48 components. Held-out reference populations never contribute either anchors
or fitting neighbors to their own predictive density. Both widths are retained.
"""
from __future__ import annotations
import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from prepare_contact_bank_guides import (
    ROOT, PINS, WIDTHS, capped_weights, check_hash, fit_bank, guide_from_components,
    local_source_closure, log_proposal, nearest_distinct, read, sha,
    source_identity, weighted_fit, write,
)

BASE_FREEZE_SHA256 = '33365ccf697dce596efe1b35486fc01bcd3176f01eb2ed01f299db2c509a9798'


def validate_base(base):
    check_hash(base / 'freeze.json', BASE_FREEZE_SHA256)
    frozen = read(base / 'freeze.json')['files']
    for filename, expected in frozen.items():
        path = (base / filename).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError('Base freeze escapes the package')
        check_hash(path, expected)
    plan = read(base / 'plan.json')
    if plan['schema'] != 'contact-bank-guide-preparation-v1' or not plan['complete'] or plan['production_launched']:
        raise ValueError('Require the inert completed original bank')
    for key, expected in PINS.items():
        if plan[key] != expected:
            raise ValueError(f'Base physical/source binding changed: {key}')
    # Imported construction helpers must be exactly those archived in the base.
    for filename in ('prepare_contact_bank_guides.py', 'prepare_native_confirmation_atlas.py'):
        if sha(ROOT / 'tools' / filename) != sha(base / 'source' / filename):
            raise ValueError('Imported construction helper changed from the frozen base: ' + filename)
    return plan


def load_reference(base):
    coverage = read(base / 'independent-reference-coverage.json')
    populations = []
    for item in coverage['populations']:
        populations.append({'arm': 'independent_R5', 'id': item['id'], 'seed': item['seed'],
            'samples': item['samples'], 'contributing_rows': item['contributing_rows']})
    source = np.load(base / 'independent-reference-poses.npz', allow_pickle=False)
    data = {key: source[key].copy() for key in source.files}
    data['class_id'] = np.zeros(len(data['u']), dtype=int)
    if len(populations) != 8 or len(data['u']) != 5064 or sum(p['samples'] for p in populations) != 131072:
        raise ValueError('Reference population allocation changed')
    for name, values in data.items():
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite reference field: ' + name)
    if np.max(np.linalg.norm(data['u'], axis=1)) > 4:
        raise ValueError('Reference contributors escaped current R4')
    return data, populations


def fit_reference(data, populations, floor, excluded_population=None, neighbors=64, cap=1 / 16):
    """Local anchored covariance for one native component per R5 population."""
    keep = data['population'] != excluded_population if excluded_population is not None else np.ones(len(data['u']), bool)
    eligible = np.flatnonzero(keep)
    tree = cKDTree(data['u'][eligible])
    retained = [i for i in range(len(populations)) if i != excluded_population]
    if not retained:
        raise ValueError('At least one training reference population is required')
    components, metadata = [], []
    for population in retained:
        rows = np.flatnonzero(keep & (data['population'] == population))
        if not len(rows):
            raise ValueError('Missing mandatory reference anchor')
        anchor_index = int(rows[np.argmax(data['log_weight'][rows])])
        anchor = data['u'][anchor_index]
        indices = nearest_distinct(data, eligible, tree, anchor, neighbors)
        source_n = np.asarray([populations[int(i)]['samples'] for i in data['population'][indices]])
        weights, cap_report = capped_weights(data['log_weight'][indices] - np.log(source_n), cap)
        nonzero = weights > 0
        fit = weighted_fit(data['u'][indices][nonzero], np.log(weights[nonzero]), floor)
        offset = fit['mean'] - anchor
        raw = fit['raw_covariance'] + np.outer(offset, offset)
        covariance = .5 * (raw + raw.T) + floor
        covariance = .5 * (covariance + covariance.T)
        np.linalg.cholesky(covariance)
        components.append({'weight': .25 / len(retained), 'mean': anchor.tolist(), 'covariance': covariance.tolist()})
        metadata.append({**source_identity(data, anchor_index, populations),
            'component_index': 48 + len(components) - 1, 'class': 'native', 'source_group': 'R5-reference',
            'mean': anchor.tolist(), 'original_log_importance_weight': float(data['log_weight'][anchor_index]),
            'neighbors': [source_identity(data, int(i), populations) for i in indices],
            'neighbor_count': len(indices), 'distinct_neighbor_count': len({tuple(data['u'][i]) for i in indices}),
            'neighbor_max_distance_u': float(np.linalg.norm(data['u'][indices] - anchor, axis=1).max()),
            'fitting_weights': weights.tolist(), 'weight_cap_diagnostic': cap_report,
            'weighted_neighbor_mean': fit['mean'].tolist(), 'anchored_empirical_covariance': raw.tolist(),
            'anchored_covariance': covariance.tolist(), 'covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
            'geometric_floor_trace_fraction_u': float(np.trace(floor) / np.trace(covariance)),
            'scope': 'Observed full-classifier native contributor in R5 intersection with unchanged R4; no synthetic native center.'})
        if excluded_population is not None and any(n['source_population_index'] == excluded_population for n in metadata[-1]['neighbors']):
            raise ValueError('Held-out reference population leaked into fitting')
    return components, metadata


def augmented_components(old_components, old_metadata, reference_components, reference_metadata):
    if len(old_components) != 48 or len(old_metadata) != 48:
        raise ValueError('The original bank must contain exactly 48 components')
    components, metadata = copy.deepcopy(old_components), copy.deepcopy(old_metadata)
    for component, item in zip(components, metadata):
        if item['class'] not in ('native', 'no-entry'):
            raise ValueError('Unexpected original definition class')
        component['weight'] = 1 / 96 if item['class'] == 'native' else 1 / 48
        item['source_group'] = 'conditional-ray'
    if sum(item['class'] == 'native' for item in metadata) != 24:
        raise ValueError('The original class allocation changed')
    components.extend(copy.deepcopy(reference_components))
    metadata.extend(copy.deepcopy(reference_metadata))
    for index, item in enumerate(metadata):
        item['component_index'] = index
    if not math.isclose(sum(c['weight'] for c in components), 1., rel_tol=0, abs_tol=1e-13):
        raise ValueError('Augmented Gaussian allocation must normalize')
    return components, metadata


def predictive_report(data, populations, log_densities, mode):
    """Original-weight scores and paired moments, including original zero rows."""
    reports = []
    uniform_q = -(3 * math.log(math.pi) - math.lgamma(4) + 6 * math.log(4))
    for population in list(range(len(populations))) + [None]:
        selected = data['population'] == population if population is not None else np.ones(len(data['u']), bool)
        n = populations[population]['samples'] if population is not None else sum(p['samples'] for p in populations)
        logw = data['log_weight'][selected]
        normalized = np.exp(logw - logsumexp(logw))
        log_qz = float(logsumexp(logw) - math.log(n))
        scores = {}
        for name, density in log_densities.items():
            qnew = density[selected]
            terms = data['pairs'][selected].sum(axis=1) + data['log_q'][selected] - qnew
            moment_weights = np.exp(terms - logsumexp(terms))
            log_moment = float(logsumexp(terms) - math.log(n))
            scores[name] = {'weighted_log_q': float(normalized @ qnew),
                'weighted_gain_vs_current_uniform_nat': float(normalized @ (qnew - uniform_q)),
                'weighted_gain_vs_reference_density_in_current_u_nat': float(normalized @ (qnew - data['log_q'][selected])),
                'paired_cloud_log_secondmoment_estimate': log_moment,
                'paired_cloud_observed_contribution_ESS': float(1 / (moment_weights @ moment_weights)),
                'paired_cloud_largest_contribution': float(moment_weights.max()),
                'physical_moment_native_ESS_proxy_per_100000_attempts': float(math.exp(math.log(100000) + 2 * log_qz - log_moment))}
        reports.append({'population': populations[population]['id'] if population is not None else 'pooled',
            'source_unconditional_samples': n, 'contributing_rows': int(sum(selected)),
            'original_weight_ESS': float(1 / (normalized @ normalized)),
            'original_largest_weight': float(normalized.max()), 'log_Q_intersection_native': log_qz,
            'guides': scores})
    return {'prediction_mode': mode, 'reports': reports}


def validate_reference(data, populations, floor, old_components, old_metadata, region_hash):
    heldout_density = {name: np.empty(len(data['u'])) for name in WIDTHS}
    folds = []
    for heldout in range(len(populations)):
        reference, reference_meta = fit_reference(data, populations, floor, heldout)
        components, metadata = augmented_components(old_components, old_metadata, reference, reference_meta)
        selected = data['population'] == heldout
        for name, width in WIDTHS.items():
            heldout_density[name][selected] = log_proposal(data['u'][selected], guide_from_components(components, region_hash, width))
        folds.append({'heldout_population_index': heldout, 'heldout_population': populations[heldout],
            'training_reference_population_indices': [i for i in range(len(populations)) if i != heldout],
            'old_components_preserved': 48, 'reference_components': len(reference),
            'reference_anchor_metadata': metadata[48:],
            'reference_conditional_component_weight': .25 / (len(populations) - 1)})
        print(f'reference held-out fold {heldout + 1}/{len(populations)} complete', flush=True)
    result = predictive_report(data, populations, heldout_density, 'leave-one-reference-population-out')
    result.update({'schema': 'reference-contact-bank-heldout-v1', 'folds': folds,
        'selection': 'None: both bank and wide are frozen; held-out scores do not select components, widths or penalties.',
        'scope': 'Each R5 reference population is excluded from all reference anchors and neighbors. All original 48 means/covariances are held fixed and use no R5 reference rows. Their revised group allocations are also fixed.',
        'moment_scope': 'Each per-population paired-cloud estimate concerns that fold\'s proposal. The pooled estimate averages different held-out proposal second moments; it is not the second moment of the final full-data guide. Original reference denominators, including zero rows, are retained.',
        'caveat': 'Previously examined reference data are not fresh validation. Scores concern only known R5-intersection native mass, not all R4 native/no-entry weight. Paired moments remove old cloud inflation but omit new cloud noise, and their realized estimates remain noisy. The ESS proxy is not measured mixing or independent samples per CPU.'})
    return result



def validate_original(base, base_plan, floor, reference_components):
    """Rebuild original population holdouts with the independent reference bank fixed."""
    archive = np.load(base / 'training-poses.npz', allow_pickle=False)
    data = {key: archive[key].copy() for key in archive.files}
    populations = base_plan['source_populations']
    densities = {name: np.empty(len(data['u'])) for name in WIDTHS}
    folds = []
    for heldout in range(len(populations)):
        old, metadata = fit_bank(data, populations, floor, heldout)
        counts = {name: sum(m['class'] == name for m in metadata) for name in ('native', 'no-entry')}
        for component, item in zip(old, metadata):
            component['weight'] = (.25 if item['class'] == 'native' else .5) / counts[item['class']]
        combined = old + copy.deepcopy(reference_components)
        selected = data['population'] == heldout
        identities = {}
        for name, width in WIDTHS.items():
            guide = guide_from_components(combined, base_plan['region_sha256'], width)
            densities[name][selected] = log_proposal(data['u'][selected], guide)
            identities[name] = hashlib.sha256(json.dumps(guide, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        folds.append({'heldout_population_index': heldout, 'heldout_population': populations[heldout],
            'old_components': len(old), 'fixed_reference_components': len(reference_components),
            'training_original_population_indices': [i for i in range(len(populations)) if i != heldout],
            'guide_canonical_json_sha256': identities,
            'old_anchor_identities': [{key: value for key, value in m.items() if key in
                ('component_index', 'class', 'arm', 'population_id', 'source_population_index', 'draw', 'seed')} for m in metadata]})
        print(f'original held-out fold with references {heldout + 1}/{len(populations)} complete', flush=True)
    classes = {}
    summary = {}
    old_validation = read(base / 'cross-validation.json')
    for class_id, class_name in enumerate(('native', 'no-entry')):
        selected = data['class_id'] == class_id
        subdata = {key: value[selected] for key, value in data.items()}
        subdensities = {key: value[selected] for key, value in densities.items()}
        report = predictive_report(subdata, populations, subdensities, 'leave-one-original-population-out-with-fixed-reference-bank')
        changes = {name: [] for name in WIDTHS}
        for index, record in enumerate(report['reports'][:-1]):
            for name in WIDTHS:
                previous = old_validation['folds'][index]['classes'][class_name]['guides'][name]['weighted_log_q']
                gain = record['guides'][name]['weighted_log_q'] - previous
                record['guides'][name]['weighted_gain_vs_original_bank_heldout_nat'] = gain
                changes[name].append(gain)
        summary[class_name] = {name: {'equal_population_mean_gain_vs_original_bank_nat': float(np.mean(values)),
            'minimum_gain_vs_original_bank_nat': float(np.min(values)),
            'maximum_gain_vs_original_bank_nat': float(np.max(values)),
            'positive_folds': int(sum(v > 0 for v in values))} for name, values in changes.items()}
        classes[class_name] = report
    return {'schema': 'reference-contact-bank-original-heldout-v1', 'folds': folds, 'classes': classes,
        'summary': summary,
        'scope': 'Entire original population excluded from original anchors and neighbors. All eight independently generated reference components remain fixed. Native original-group allocation changes from0.25 to0.125; no-entry remains0.25; references add0.125. Both widths retained.',
        'caveat': 'Original importance ESS is often near one. Predictive scores and averaged fold-specific paired moments do not establish physical convergence or the variance of the final full-data guide.'}


def prepare(base, out):
    start = time.monotonic()
    if out.exists():
        raise ValueError('Refusing to overwrite a preparation: ' + str(out))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools'),
                            '-p', 'test_reference_contact_bank.py', '-v'], capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    base_plan = validate_base(base)
    data, populations = load_reference(base)
    floor = np.asarray(base_plan['fitting_rule']['latent_geometric_floor'])
    old = read(base / 'guide-bank.json')['gaussian_components']
    old_metadata = base_plan['anchor_metadata']
    reference, reference_metadata = fit_reference(data, populations, floor)
    components, metadata = augmented_components(old, old_metadata, reference, reference_metadata)
    heldout = validate_reference(data, populations, floor, old, old_metadata, base_plan['region_sha256'])
    original_heldout = validate_original(base, base_plan, floor, reference)
    guides = {name: guide_from_components(components, base_plan['region_sha256'], width) for name, width in WIDTHS.items()}
    full_density = {name: log_proposal(data['u'], guide) for name, guide in guides.items()}
    fitted = predictive_report(data, populations, full_density, 'full-data-fitted-density-not-held-out')
    fitted['caveat'] = 'In-sample coverage diagnostic only. This guide used these reference rows; scores cannot validate the final guide or predict physical convergence.'
    out.mkdir(parents=True)
    # Reuse the complete frozen preparation; do not reread any physical raw stream.
    shutil.copytree(base, out / 'base-preparation')
    for filename in ('config.json', 'region.json', 'shape.json', 'native-definition.json'):
        shutil.copyfile(base / filename, out / filename)
    shutil.copytree(base / 'native-region', out / 'native-region')
    for name, guide in guides.items():
        write(out / f'guide-{name}.json', guide)
    write(out / 'reference-cross-validation.json', heldout)
    write(out / 'original-cross-validation.json', original_heldout)
    write(out / 'reference-fitted-coverage.json', fitted)
    np.savez_compressed(out / 'reference-training-poses.npz', **data)
    closure = local_source_closure([Path(__file__), ROOT / 'tools/test_reference_contact_bank.py'])
    source_sha = {}
    for name, source in closure.items():
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_sha[str(source)] = sha(source)
    write(out / 'source-closure.json', {'schema': 'reference-contact-bank-source-closure-v1',
        'base_preparation': str(base), 'base_freeze_sha256': BASE_FREEZE_SHA256,
        'source_files': source_sha,
        'scope': 'Reuses the frozen 100408-row original bank and 5064-row independent-reference archive. No old physical raw rows, overlap predicates, classifier calls or Poisson audits were rerun.'})
    (out / 'tests.txt').write_text(tests.stdout + tests.stderr)
    plan = copy.deepcopy(base_plan)
    plan.update({'schema': 'reference-contact-bank-preparation-v1', 'complete': True, 'production_launched': False,
        'created': datetime.now(timezone.utc).isoformat(), 'base_preparation': 'base-preparation',
        'base_preparation_source': str(base), 'base_preparation_freeze_sha256': BASE_FREEZE_SHA256,
        'component_count': len(components), 'anchor_metadata': metadata,
        'component_groups': [
            {'source_group': 'conditional-ray', 'class_name': 'native', 'count': 24, 'component_weight': 1 / 96, 'full_probability': .125},
            {'source_group': 'conditional-ray', 'class_name': 'no-entry', 'count': 24, 'component_weight': 1 / 48, 'full_probability': .25},
            {'source_group': 'R5-reference', 'class_name': 'native', 'count': 8, 'component_weight': 1 / 32, 'full_probability': .125}],
        'guide_sha256': {name: sha(out / f'guide-{name}.json') for name in WIDTHS},
        'source_reference_populations': populations,
        'source_seeds': base_plan['source_seeds'] + [p['seed'] for p in populations],
        'source_unconditional_draws': base_plan['source_unconditional_draws'] + sum(p['samples'] for p in populations),
        'retained_training_class_rows': base_plan['retained_training_class_rows'] + len(data['u']),
        'heldout_diagnostics': 'reference-cross-validation.json', 'independent_reference_coverage': 'reference-fitted-coverage.json',
        'original_heldout_diagnostics': 'original-cross-validation.json',
        'reference_coverage_kind': 'Reference rows are now training evidence; full-fit scoring is in-sample. Whole-reference-population held-out scores are stored separately. Neither is a new physical population.',
        'tests': {'returncode': tests.returncode, 'report': 'tests.txt', 'source_sha256': sha(ROOT / 'tools/test_reference_contact_bank.py')},
        'preparation_wall_seconds': time.monotonic() - start})
    plan['fitting_rule'] = {'old_components': 'Original48 means/covariances exactly retained from base-preparation/guide-bank.json; only group allocation changes.',
        'new_components': 'One highest ORIGINAL-weight observed native contributor per independent R5 population, eight mandatory means.',
        'neighbors': 64, 'distance': 'Euclidean original current-R4 u, distinct R5 contributor coordinates, stable population/draw tie order.',
        'fitting_log_weight': 'original_log_importance_weight - log(source_unconditional_samples)',
        'normalized_weight_cap': 1 / 16, 'covariance': 'Capped weighted second moment about the mandatory reference anchor plus the same transformed physical geometric floor.',
        'translation_floor_std_A': .05, 'angle_axis_floor_scale_deg': .1,
        'raw_geometric_floor': base_plan['fitting_rule']['raw_geometric_floor'], 'latent_geometric_floor': floor,
        'allocation': {'uniform_R4': .5, 'conditional_ray_native_Gaussians': .125,
                       'conditional_ray_no_entry_Gaussians': .25, 'R5_reference_native_Gaussians': .125},
        'conditional_gaussian_component_weights': {'conditional_ray_native': 1 / 96, 'conditional_ray_no_entry': 1 / 48, 'R5_reference_native': 1 / 32},
        'selection': 'Both base and four-times-covariance widths are frozen. Construction declared after first bank coverage failure, before any new physical draws.'}
    write(out / 'plan.json', plan)
    (out / 'README.md').write_text('Reference-augmented contact bank\n\n'
        'This separate frozen design adds eight observed native-reference anchors to the unchanged means and covariances of the original48-component bank. '
        'The original bank remains an inert record of its coverage limitation. Full allocations are50%uniform R4,12.5%old native,25%no-entry and12.5%reference native. '
        'Guide-wide multiplies every covariance by four. Both are retained.\n\n'
        'reference-cross-validation.json excludes each reference population from both reference anchors and neighbors. '
        'reference-fitted-coverage.json uses all reference rows and is explicitly in-sample. Neither supplies new physical evidence. '
        'All original unconditional denominators are retained in the diagnostics. '
        'No physical samples were generated, and no old geometry or Poisson audits were replayed.\n')
    freeze = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out / 'freeze.json', {'files': freeze})
    print(json.dumps({'out': str(out), 'components': len(components), 'guide_sha256': plan['guide_sha256'],
                      'heldout_pooled': heldout['reports'][-1], 'fitted_pooled': fitted['reports'][-1]}, indent=2), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'runs/contact-bank-guide-preparation-20260922')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/reference-contact-bank-preparation-20260922')
    args = parser.parse_args()
    prepare(args.base.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
