#!/usr/bin/env python3
"""Declare and freeze geometry-only additions to an existing defensive R4 guide.

Completed terminal SMC slots are correlated fitting geometry, never independent
importance draws. This tool launches no physics, replays no audits, and uses no
native labels. Historical native-informed charts/old guide remain provenance.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import scipy
from analyze_r4_smc_control import Chart, Ledger, validate_config_identity
from prepare_contact_bank_guides import (guide_from_components, latent_geometric_floor,
    local_source_closure, log_proposal, read, sha, write)

ROOT = Path(__file__).resolve().parents[1]
AUTHENTICATION = ROOT / 'runs/smc-completed-evidence-review-20260923/authentication.json'
AUTHENTICATION_SHA256 = 'b5e23f739358bac7b0234d629a6744ee08b9c7c37f8ffe43dfc5c71d05b33fc7'
ANALYSIS = ROOT / 'runs/smc-r4-controls-workflow-20260922/narrow-analysis/analysis.json'
OLD_PREPARATION = ROOT / 'runs/refined-contact-bank-preparation-20260922'
FIT_IDS = ('r00', 'r01')
HELDOUT_IDS = ('r02', 'r03')
CANDIDATES = (
    {'id': 'r5-cov1', 'split_sign': False, 'covariance_multiplier': 1.0},
    {'id': 'r5-cov4', 'split_sign': False, 'covariance_multiplier': 4.0},
    {'id': 'r5-sign-cov1', 'split_sign': True, 'covariance_multiplier': 1.0},
    {'id': 'r5-sign-cov4', 'split_sign': True, 'covariance_multiplier': 4.0},
)
SCOPE = ('Preparation and proposal-density geometry only. No new physical draws, audit replay, '
         'physical normalizer, descendant IID ESS, row-level standard error, or convergence claim. '
         'All terminal slots including repeated descendants enter geometric fitting. The new '
         'components use no native/classifier labels; historical charts and the unchanged old '
         '80-component bank are native-informed. This is not template-free assembly proof.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_identity():
    return {'python': str(Path(sys.executable).resolve()), 'python_sha256': sha(sys.executable),
            'python_version': sys.version, 'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
            'thread_environment': {k: os.environ[k] for k in
                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}


def arithmetic_geometry_fit(u, floor):
    """Population second moment, retaining multiplicity; positive geometric floor."""
    u = np.asarray(u, dtype=float)
    floor = np.asarray(floor, dtype=float)
    require(u.ndim == 2 and u.shape[1] == 6 and len(u) > 0 and np.isfinite(u).all(),
            'Expected nonempty finite six-dimensional terminal geometry')
    require(floor.shape == (6, 6) and np.isfinite(floor).all() and
            np.allclose(floor, floor.T, rtol=0, atol=1e-14), 'Invalid geometric floor')
    np.linalg.cholesky(floor)
    mean = u.mean(axis=0)
    centered = u - mean
    covariance = centered.T @ centered / len(u) + floor
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    return mean, covariance


def fit_population_geometry(populations, floor, split_sign=False):
    """Equal population/group mass; empty groups fit their full population.

    Each input has only id, u, and old_r5_inside fields relevant to this fit.
    old_r5_inside is the UNFILTERED historical chart-ball membership of each pose.
    No slot is deduplicated, weighted by physical labels, or discarded.
    """
    require(len(populations) > 0, 'No fitting populations')
    require(len({p['id'] for p in populations}) == len(populations), 'Duplicate population ids')
    components, metadata = [], []
    group_count = 4 if split_sign else 2
    for population in populations:
        u = np.asarray(population['u'], dtype=float)
        inside = np.asarray(population['old_r5_inside'])
        # Validates all slots, even if their geometric group is empty.
        arithmetic_geometry_fit(u, floor)
        require(inside.shape == (len(u),) and inside.dtype.kind == 'b', 'Invalid R5 ball memberships')
        covered = np.zeros(len(u), dtype=int)
        for in_ball in (True, False):
            for positive in ((False, True) if split_sign else (None,)):
                mask = inside == in_ball
                if positive is not None:
                    mask &= (u[:, 0] >= 0) == positive
                slots = np.flatnonzero(mask)
                covered[slots] += 1
                selected = slots if len(slots) else np.arange(len(u))
                mean, covariance = arithmetic_geometry_fit(u[selected], floor)
                components.append({'weight': 1 / (len(populations) * group_count),
                                   'mean': mean.tolist(), 'covariance': covariance.tolist()})
                metadata.append({'component_index': len(components) - 1,
                    'population_id': population['id'], 'inside_unfiltered_historical_R5_ball': in_ball,
                    'original_u0_nonnegative': positive, 'group_slots': slots.tolist(),
                    'group_slot_count': len(slots), 'fitting_slots': selected.tolist(),
                    'fitting_slot_count': len(selected), 'empty_group_full_population_fallback': not len(slots),
                    'population_slot_count': len(u),
                    'covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist()})
        require(np.all(covered == 1), 'Geometry partition omitted or duplicated terminal slots')
    return components, metadata


def validate_guide(guide, expected_components=None):
    require(guide['schema'] == 'defensive-latent-shell-guide-v1' and
            guide['defensive_uniform_shell_probability'] == .5, 'Guide must have alpha=.5')
    components = guide['gaussian_components']
    require(len(components) > 0 and (expected_components is None or len(components) == expected_components),
            'Unexpected Gaussian component count')
    total = 0.
    for c in components:
        mean, covariance, weight = np.asarray(c['mean']), np.asarray(c['covariance']), c['weight']
        require(mean.shape == (6,) and np.isfinite(mean).all() and covariance.shape == (6, 6) and
                np.isfinite(covariance).all() and np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
                and math.isfinite(weight) and weight > 0, 'Invalid Gaussian component')
        np.linalg.cholesky(covariance)
        total += weight
    require(math.isclose(total, 1., rel_tol=0, abs_tol=2e-13), 'Gaussian weights must sum to one')


def augment_guide(old_guide, smc_components, covariance_multiplier=1.):
    """q_new=.5 U_R4+.25 G_old+.25 G_smc, hence q_new>=.5 q_old."""
    validate_guide(old_guide)
    require(covariance_multiplier in (1., 4.), 'Undeclared covariance multiplier')
    smc = guide_from_components(smc_components, old_guide['region_sha256'], covariance_multiplier)
    validate_guide(smc)
    combined = copy.deepcopy(old_guide['gaussian_components']) + smc['gaussian_components']
    for component in combined:
        component['weight'] *= .5
    guide = guide_from_components(combined, old_guide['region_sha256'])
    validate_guide(guide)
    return guide


def authenticate_inputs(authentication=AUTHENTICATION, analysis_path=ANALYSIS, old_preparation=OLD_PREPARATION):
    """Authenticate small manifests/summaries; do not read terminal coordinates.

    The previous completed audit is reused. Its large initialization/stage stream
    hashes are recorded through the authenticated audit, not replayed or reread.
    """
    authentication, analysis_path, old_preparation = map(lambda p: Path(p).resolve(),
        (authentication, analysis_path, old_preparation))
    ledger = Ledger()
    auth = read(ledger.bind(authentication, AUTHENTICATION_SHA256))
    require(auth['schema'] == 'completed-smc-bridge-evidence-authentication-v1' and auth['complete'],
            'Completed SMC evidence authentication required')
    analysis = read(ledger.bind(analysis_path, auth['input_sha256'][str(analysis_path)]))
    require(analysis['complete'] and analysis['schema'] == 'smc-r4-control-analysis-v1',
            'Completed narrow SMC audit required')
    protocol_path = ledger.bind(analysis['protocol'], analysis['protocol_sha256'])
    protocol = read(protocol_path)
    require(protocol == analysis['protocol_snapshot'], 'Audited protocol snapshot differs')
    require('narrow' in protocol_path.parent.name, 'Only the narrow SMC arm is authorized')
    source_hashes = protocol['source_and_input_sha256']
    old_freeze_path = old_preparation / 'freeze.json'
    old_freeze = read(ledger.bind(old_freeze_path, source_hashes[str(old_freeze_path)]))['files']
    for filename in ('plan.json', 'guide-bank.json', 'config.json', 'shape.json', 'region.json', 'old-r5-region.json'):
        ledger.bind(old_preparation / filename, old_freeze[filename])
    old_plan = read(old_preparation / 'plan.json')
    require(old_plan['complete'] and old_plan['guide_sha256']['bank'] == old_freeze['guide-bank.json'],
            'Incomplete or inconsistent old bank preparation')
    guide = read(old_preparation / 'guide-bank.json')
    validate_guide(guide, 80)
    region_path, reference_path = old_preparation / 'region.json', old_preparation / 'old-r5-region.json'
    region, reference = read(region_path), read(reference_path)
    require(region['mahalanobis_radius'] == 4 and region.get('minimum_mahalanobis_radius', 0) == 0 and
            region['minimum_original_q'] == 0 and 'maximum_original_q' not in region, 'Expected full current R4')
    require(reference['mahalanobis_radius'] == 5, 'Expected historical R5 chart ball')
    require(guide['region_sha256'] == sha(region_path) == protocol['physical_target']['region_sha256'],
            'Guide/current SMC region identity differs')
    ledger.bind(protocol['physical_target']['region'], sha(region_path))
    ledger.bind(protocol['proposal']['reference_region'], sha(reference_path))
    narrow_config_path = ledger.bind(protocol['physical_target']['config'],
        source_hashes[protocol['physical_target']['config']])
    validate_config_identity(read(old_preparation / 'config.json'), read(narrow_config_path),
                             allowed=('translation_steps', 'rotation_steps_deg'))
    require(sha(old_preparation / 'shape.json') == protocol['physical_target']['shape_sha256'],
            'Physical shape differs')
    populations = []
    for job in protocol['jobs']:
        require(job['id'] in FIT_IDS + HELDOUT_IDS, 'Unexpected narrow population')
        audit_path = analysis_path.parent / job['id'] / 'population.json'
        audit = read(ledger.bind(audit_path, auth['input_sha256'][str(audit_path)]))
        require(audit['id'] == job['id'] and audit['seed'] == job['seed'], 'Audited population identity mismatch')
        hashes = audit['source_sha256']
        directory = Path(job['output'])
        # Hashing the summary does not evaluate its coordinates.
        for filename in ('summary.json', 'status.json', 'manifest.json', 'provenance/config.json',
                         'provenance/region.json', 'provenance/shape.json',
                         'provenance/initial-reference-region.json', 'provenance/source-bundle.json'):
            path = directory / filename
            ledger.bind(path, hashes[str(path)])
        status, manifest = read(directory / 'status.json'), read(directory / 'manifest.json')
        require(status['complete'] and status['phase'] == 'complete' and
                status['summary_sha256'] == hashes[str(directory / 'summary.json')], 'SMC population incomplete')
        require(manifest['options']['seed'] == job['seed'] and manifest['region_sha256'] == sha(region_path)
                and manifest['config_sha256'] == sha(narrow_config_path)
                and manifest['initial_reference_sha256'] == sha(reference_path)
                and manifest['shape_sha256'] == sha(old_preparation / 'shape.json')
                and manifest['executable_sha256'] == protocol['binary_sha256']
                and manifest['source_bundle_sha256'] == protocol['source_bundle_sha256'],
                'SMC manifest physical/source identity mismatch')
        require(status['completed_stage'] == len(manifest['options']['schedule']) - 1,
                'SMC schedule did not finish')
        populations.append({'id': job['id'], 'seed': job['seed'], 'directory': str(directory),
            'summary': str(directory / 'summary.json'), 'summary_sha256': hashes[str(directory / 'summary.json')],
            'manifest_sha256': hashes[str(directory / 'manifest.json')],
            'expected_slots': manifest['options']['population'], 'completed_stage': status['completed_stage'],
            'audited_source_sha256': hashes, 'audit_population': str(audit_path),
            'role': 'fit' if job['id'] in FIT_IDS else 'heldout_geometry_only'})
    require(tuple(p['id'] for p in populations) == FIT_IDS + HELDOUT_IDS, 'Narrow population allocation differs')
    require(len({p['seed'] for p in populations}) == 4, 'Population seeds are not independent allocations')
    ledger.recheck()
    return {'authentication': str(authentication), 'analysis': str(analysis_path),
            'protocol': str(protocol_path), 'old_preparation': str(old_preparation),
            'current_region': str(region_path), 'historical_r5_region': str(reference_path),
            'old_guide': str(old_preparation / 'guide-bank.json'), 'populations': populations,
            'input_sha256': ledger.files}


def load_authenticated_terminal_geometry(plan, population_ids):
    """Load complete slots only for explicitly requested declared populations."""
    require(tuple(population_ids) in (FIT_IDS, HELDOUT_IDS), 'Load either declared fit or held-out populations')
    ledger = Ledger()
    for key in ('current_region', 'historical_r5_region'):
        ledger.bind(plan[key], plan['input_sha256'][plan[key]])
    current, historical = Chart(read(plan['current_region'])), Chart(read(plan['historical_r5_region']))
    populations, validations = [], []
    for identity in plan['populations']:
        if identity['id'] not in population_ids:
            continue
        summary = read(ledger.bind(identity['summary'], identity['summary_sha256']))
        require(summary['complete'] and summary['completed_stage'] == identity['completed_stage'] and
                summary['manifest_sha256'] == identity['manifest_sha256'], 'Terminal summary identity differs')
        particles = summary['terminal_particles']
        require(len(particles) == identity['expected_slots'] > 0, 'Missing terminal slots; never drop a population')
        u, inside, ancestors, maximum_error = [], [], [], 0.
        for particle in particles:
            latent = np.asarray(particle['latent'], dtype=float)
            mapped = current.evaluate(particle['pose'])
            require(latent.shape == (6,) and np.isfinite(latent).all() and mapped['latent'] is not None,
                    'Invalid terminal chart geometry')
            error = float(np.max(np.abs(latent - mapped['latent'])))
            require(error <= 2e-8 and np.linalg.norm(latent) <= current.radius + 2e-8,
                    'Stored latent/pose mapping mismatch or terminal pose outside current R4')
            maximum_error = max(maximum_error, error)
            u.append(latent)
            inside.append(historical.evaluate(particle['pose'])['inside'])
            ancestors.append(particle['initial_ancestor'])
        populations.append({'id': identity['id'], 'u': np.asarray(u), 'old_r5_inside': np.asarray(inside, dtype=bool),
                            'initial_ancestor': np.asarray(ancestors, dtype=np.int64), 'seed': identity['seed']})
        validations.append({'population_id': identity['id'], 'slots': len(particles),
            'maximum_pose_to_latent_absolute_error': maximum_error,
            'distinct_initial_ancestors': len(set(ancestors)),
            'distinct_latent_coordinates': len({tuple(row) for row in u}),
            'all_slots_retained': True, 'summary_sha256': identity['summary_sha256']})
    require(tuple(p['id'] for p in populations) == tuple(population_ids),
            'Missing or reordered declared terminal populations')
    ledger.recheck()
    return populations, validations


def declare(out, authentication=AUTHENTICATION, analysis=ANALYSIS, old_preparation=OLD_PREPARATION):
    out = Path(out).resolve()
    require(not out.exists(), 'Refusing to overwrite an existing declaration')
    # Fixed declarations precede any terminal-coordinate evaluation. Authentication
    # reads identities only and hashes summaries without interpreting their slots.
    choices = copy.deepcopy(list(CANDIDATES))
    inputs = authenticate_inputs(authentication, analysis, old_preparation)
    source_paths = local_source_closure([Path(__file__),
        Path(__file__).with_name('test_prepare_smc_geometry_guides.py'),
        Path(__file__).with_name('score_smc_geometry_guides.py'),
        Path(__file__).with_name('test_score_smc_geometry_guides.py')])
    out.mkdir(parents=True)
    (out / 'source').mkdir()
    code_hashes = {}
    for name, path in source_paths.items():
        shutil.copyfile(path, out / 'source' / name)
        code_hashes[str(path)] = sha(path)
    plan = {'schema': 'smc-terminal-geometry-guide-declaration-v1', 'declared_at': time.time(),
        **inputs, 'fit_population_ids': list(FIT_IDS), 'heldout_population_ids': list(HELDOUT_IDS),
        'candidates': choices, 'runtime': runtime_identity(), 'code_sha256': code_hashes,
        'fitting_rule': 'Arithmetic mean and 1/N centered second moment plus the existing positive physical geometric floor; include repeated terminal slots.',
        'grouping': 'Inside/outside full unfiltered historical R5 chart ball; optional original current-u coordinate 0 sign with zero nonnegative.',
        'empty_group': 'Keep every declared component; deterministic full-population fallback, never redraw or filter by native labels.',
        'allocation': 'Each independent fit population has equal Gsmc mass; its two or four geometric groups have equal mass.',
        'mixture': {'uniform_R4': .5, 'unchanged_old_80_component_Gaussian_bank': .25, 'SMC_geometry_Gaussians': .25},
        'density_bound': 'q_new >= 0.5 q_old pointwise; full normalized untruncated Gaussian densities in original d6u.',
        'covariance_floor': {'translation_std_A': .05, 'angle_axis_scale_deg': .1},
        'heldout_use': 'r02/r03 are excluded from fitting. Aggregate geometry was already inspected during proposal design, so later density diagnostics are not pristine validation or evidence of unseen coverage.',
        'diagnostics': {
            'selection_rule': 'Report all four candidates; no automatic winner or post-score refitting.',
            'heldout_population_ids': list(HELDOUT_IDS),
            'heldout_geometry_groups': ['all', 'inside_old_R5_chart', 'outside_old_R5_chart'],
            'heldout_statistics': ['mean_log_density_gain_vs_bank', 'median_log_density_gain_vs_bank', 'fraction_density_above_bank'],
            'importance_source': str(ROOT / 'runs/contact-confirmation-comparison-20260922'),
            'importance_classes': ['total', 'registered_native_entry', 'old_R5_intersection_native',
                'remaining_R4_native', 'contact_no_native_entry', 'unbound'],
            'stratified_classes': ['registered_native_entry', 'old_R5_intersection_native',
                'remaining_R4_native', 'contact_no_native_entry'],
            'strata': {'radial': 3, 'angular': 3, 'orthant': 64},
            'moment_laws': ['two_cloud_noisy', 'paired_physical'],
            'moment_rule': 'E_source[w^2 q_source/q_target] and paired-cloud product counterpart; retain all attempted zeros in original unconditional N; separate source arms/cloud laws.',
            'interpretation': 'Reused inspected archives and correlated terminal geometry provide design diagnostics only; no independent physical validation, automatic winner, variance/speedup claim, or unseen-mode coverage claim.'},
        'physical_draws_launched': 0, 'audits_replayed': 0, 'scope': SCOPE}
    write(out / 'declaration.json', plan)
    write(out / 'declaration-freeze.json', {'files': {str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}})
    return plan


def prepare(out):
    out = Path(out).resolve()
    require(not (out / 'freeze.json').exists() and not (out / 'preparation.json').exists(),
            'Refusing to overwrite a completed preparation')
    frozen = read(out / 'declaration-freeze.json')['files']
    require('declaration.json' in frozen, 'Unfrozen declaration')
    for name, expected in frozen.items():
        path = (out / name).resolve()
        require(path.is_relative_to(out) and sha(path) == expected, 'Declaration/source snapshot changed')
    plan = read(out / 'declaration.json')
    require(plan['schema'] == 'smc-terminal-geometry-guide-declaration-v1' and
            plan['candidates'] == list(CANDIDATES) and plan['fit_population_ids'] == list(FIT_IDS)
            and plan['heldout_population_ids'] == list(HELDOUT_IDS), 'Predeclared choices changed')
    require(plan['runtime'] == runtime_identity(), 'Declared runtime changed')
    ledger = Ledger()
    for path, expected in {**plan['input_sha256'], **plan['code_sha256']}.items():
        ledger.bind(path, expected)
    populations, validations = load_authenticated_terminal_geometry(plan, FIT_IDS)
    region, old_guide = read(plan['current_region']), read(plan['old_guide'])
    floor, raw_floor = latent_geometric_floor(region['gaussian_chart'])
    fits, reports, guides = {}, {}, {}
    for candidate in plan['candidates']:
        split = candidate['split_sign']
        if split not in fits:
            fits[split] = fit_population_geometry(populations, floor, split)
        components, metadata = fits[split]
        guide = augment_guide(old_guide, components, candidate['covariance_multiplier'])
        target = out / ('guide-' + candidate['id'] + '.json')
        require(not target.exists(), 'Refusing to overwrite a partial guide')
        write(target, guide)
        guides[candidate['id']] = {'path': str(target), 'sha256': sha(target),
            'old_components': 80, 'new_components': len(components),
            'covariance_multiplier': candidate['covariance_multiplier']}
        reports[candidate['id']] = metadata
    np.savez_compressed(out / 'fitting-geometry.npz', **{p['id'] + '_' + k: p[k]
        for p in populations for k in ('u', 'old_r5_inside', 'initial_ancestor')})
    write(out / 'component-metadata.json', reports)
    ledger.recheck()
    result = {'schema': 'smc-terminal-geometry-guide-preparation-v1', 'complete': True,
        'declaration_sha256': sha(out / 'declaration.json'), 'guides': guides,
        'pose_to_latent_validation': validations, 'latent_geometric_floor': floor,
        'raw_geometric_floor': raw_floor, 'all_terminal_slots_retained': True,
        'native_classifier_calls': 0, 'physical_draws_launched': 0, 'audits_replayed': 0,
        'heldout_terminal_geometry_read': False, 'scope': SCOPE}
    write(out / 'preparation.json', result)
    write(out / 'plan.json', {**plan, **result})
    write(out / 'status.json', {'complete': True, 'physical_draws_launched': 0,
        'audits_replayed': 0, 'preparation_sha256': sha(out / 'preparation.json'),
        'plan_sha256': sha(out / 'plan.json')})
    write(out / 'freeze.json', {'files': {str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'freeze.json'}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    declaration = sub.add_parser('declare', help='Freeze deterministic candidates and source identities before fitting')
    declaration.add_argument('--out', type=Path, required=True)
    declaration.add_argument('--authentication', type=Path, default=AUTHENTICATION)
    declaration.add_argument('--analysis', type=Path, default=ANALYSIS)
    declaration.add_argument('--old-preparation', type=Path, default=OLD_PREPARATION)
    preparation = sub.add_parser('prepare', help='Fit only r00/r01 using an unchanged prior declaration')
    preparation.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = declare(args.out, args.authentication, args.analysis, args.old_preparation) if args.command == 'declare' else prepare(args.out)
    print(json.dumps({'command': args.command, 'out': str(args.out.resolve()),
                      'schema': result['schema'], 'physical_draws_launched': 0, 'audits_replayed': 0}))


if __name__ == '__main__':
    main()
