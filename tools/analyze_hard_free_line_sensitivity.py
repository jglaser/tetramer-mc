#!/usr/bin/env python3
"""Separate, fixed guide/cloud controls; reuse completed pilot summaries only."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import math
from pathlib import Path
import numpy as np
from analyze_hard_free_line_physical import (
    classify_population, summarize_arm, relabel_nonuniform, load_classifier,
    DECISION_CLASSES, quality_gate, free_energy_interval, unbound_volume_bound,
    paired_moments, read, sha, require, write)
from analyze_hard_free_line_population_size import population_logs
from analyze_contact_bank_pilot import compare_free_energy_intervals
from run_smc_guide_pilot import CONVERGENCE, STRATA

REFERENCE = dict(id='conditioned', beta=1., alpha=.5, component_count=92,
                 samples=16384, lambda_ratio=128.)
CONTROLS = [dict(REFERENCE, id='alpha02', alpha=.2),
            dict(REFERENCE, id='lambda64', lambda_ratio=64.)]
PLAN = dict(schema='hard-free-line-sensitivity-comparison-plan-v1',
    reference_arm='conditioned', reference_allocation=copy.deepcopy(REFERENCE),
    control_arms=[a['id'] for a in CONTROLS], control_allocations=copy.deepcopy(CONTROLS),
    samples_per_population=16384, populations_per_arm=4,
    regions=list(DECISION_CLASSES), quantities=['Qz', 'Q0'], total_quantities=['Qz', 'Q0'],
    total_mass_policy='Total Q0 is required; total Qz is reported. Each decision region requires both Qz and Q0.',
    stratum_families=['radial', 'angular', 'orthant'], strata=copy.deepcopy(STRATA),
    convergence=copy.deepcopy(CONVERGENCE), stages_pooled=False,
    old_poses_replayed=False, old_native_classifier_calls=0,
    comparisons='Each control against the completed conditioned pilot; no direct '
                'alpha02-versus-lambda64 comparison that changes two factors at once.',
    material_stratum_rule='Observed within-region Qz fraction >=0.01 in either '
                         'arm; all strata and both physical and hard-only comparisons retained.',
    free_energy_contrast='Compare the native-minus-competing contrast itself using '
                         'independent, paired-within-population delta SE; require '
                         '<=3 combined SE and <=0.2 kBT, in addition to each 95% interval halfwidth <=0.5.',
    full_vessel_gate_open=False, assembly_gate_open=False)
SCHEMA = 'hard-free-line-sensitivity-analysis-v1'


def compare_linear(reference, control, kind='Qz'):
    """Compare linear population means, retaining finite log masses and zeros."""
    a = population_logs(reference, kind, 16384)
    b = population_logs(control, kind, 16384)
    require({p['seed'] for p in reference['populations']}.isdisjoint(
        p['seed'] for p in control['populations']), 'Reference and control streams must be independent')
    logs = dict(reference_population_log_masses=[None if not math.isfinite(x) else float(x) for x in a],
                control_population_log_masses=[None if not math.isfinite(x) else float(x) for x in b])
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return dict(observed=False, passed=False, **logs,
            reason='Unobserved mass is neither a physical zero nor an upper bound.')
    la, lb = float(a.max()), float(b.max())
    x, y = np.exp(a-la), np.exp(b-lb)
    mx, my = float(x.mean()), float(y.mean())
    sx, sy = float(x.std(ddof=1)/2), float(y.std(ddof=1)/2)
    lma, lmb = la+math.log(mx), lb+math.log(my)
    delta = lmb-lma
    require(math.isfinite(delta), 'Unrepresentable log mass difference')
    scale = max(la, lb)
    fa, fb = math.exp(la-scale), math.exp(lb-scale)
    ma, mb, sa, sb = mx*fa, my*fb, sx*fa, sy*fb
    se = math.hypot(sa, sb)
    tolerance = 32*np.finfo(float).eps*max(ma, mb)
    absolute = abs(delta) <= CONVERGENCE['log_agreement_absolute_max']
    statistical = abs(mb-ma) <= CONVERGENCE['log_agreement_combined_SE_max']*se+tolerance
    return dict(observed=True, passed=bool(absolute and statistical), **logs,
        log_scale=scale, scaled_reference_linear_mean=ma, scaled_control_linear_mean=mb,
        log_reference_linear_mean=lma, log_control_linear_mean=lmb,
        shared_display_scale_underflow=dict(reference=ma == 0., control=mb == 0.),
        scaled_reference_population_SE=sa, scaled_control_population_SE=sb,
        scaled_control_minus_reference_linear_difference=mb-ma,
        scaled_independent_difference_SE=se, scaled_comparison_roundoff_tolerance=tolerance,
        log_control_minus_reference=delta, combined_population_log_delta_SE=math.hypot(sx/mx, sy/my),
        absolute_passed=bool(absolute), SE_passed=bool(statistical),
        absolute_limit=CONVERGENCE['log_agreement_absolute_max'],
        SE_multiplier=CONVERGENCE['log_agreement_combined_SE_max'],
        SE_test='Independent linear difference; log-ratio SE is diagnostic only.')


def validate_summary(data, schema, allocations):
    require(data['schema'] == schema and data['complete'] is True
        and data['total_unconditional_draws'] == 131072
        and set(data['arms']) == {a['id'] for a in allocations}, 'Incomplete or mismatched stage summary')
    require(not data['diagnostics']['full_vessel_gate_open']
        and not data['diagnostics']['assembly_gate_open'], 'A regional stage cannot decide assembly')
    seeds = set()
    for allocation in allocations:
        arm = data['arms'][allocation['id']]
        require(arm['allocation'] == allocation, 'Changed physical proposal/cloud allocation')
        require(arm['native_partition_sum_verified'] is True, 'Native partition sum not verified')
        records = arm['populations']
        require(len(records) == 4 and {p['id'] for p in records} == {f'r{i:02}' for i in range(4)}
            and all(p['samples'] == 16384 and p['arm'] == allocation['id'] for p in records),
            'Missing unconditional population identities')
        for record in records:
            require(record['seed'] not in seeds, 'Repeated stream within stage')
            seeds.add(record['seed'])
        identity = {(p['id'], p['seed']) for p in records}
        entries = list(arm['estimates'].values())
        for family in PLAN['stratum_families']:
            for region in PLAN['regions']:
                group = arm['strata'][family][region]
                require([s['bin'] for s in group] == list(range(64 if family == 'orthant' else 3)),
                        'Changed stratum identity or order')
                entries.extend(group)
        for entry in entries:
            require({(p['id'], p['seed']) for p in entry['populations']} == identity,
                    'Estimate population identity differs from saved stage')
            for kind in PLAN['quantities']:
                population_logs(entry, kind, 16384)
    return seeds


def arm_diagnostics(arms):
    return dict(quality={name: {region: quality_gate(arm['estimates'][region], CONVERGENCE)
        for region in DECISION_CLASSES} for name, arm in arms.items()},
        free_energy_intervals={name: free_energy_interval(arm, CONVERGENCE) for name, arm in arms.items()},
        full_vessel_gate_open=False, assembly_gate_open=False)


def compare_sensitivity(pilot, current, plan):
    require(plan == PLAN, 'Sensitivity comparison plan changed')
    old_allocations = [dict(REFERENCE, id='baseline', beta=0.), REFERENCE]
    old_seeds = validate_summary(pilot, 'hard-free-line-physical-comparison-v1', old_allocations)
    new_seeds = validate_summary(current, SCHEMA, CONTROLS)
    require(old_seeds.isdisjoint(new_seeds), 'Control streams reuse completed pilot streams')
    identity = ('definition_sha256', 'runtime_sha256', 'input_sha256', 'criteria', 'scope')
    require(all(pilot['native_definition'][k] == current['native_definition'][k] for k in identity)
        and pilot['unbound_R4_bound'] == current['unbound_R4_bound'], 'Classifier or target bound changed')
    reference = pilot['arms']['conditioned']
    selected = dict(reference=reference, **current['arms'])
    diagnostics = arm_diagnostics(selected)
    regions, strata, contrasts = {}, [], {}
    for name, control in current['arms'].items():
        regions[name] = {r: {kind: compare_linear(reference['estimates'][r], control['estimates'][r], kind)
            for kind in PLAN['quantities']} for r in ['total', *DECISION_CLASSES]}
        contrasts[name] = compare_free_energy_intervals(
            diagnostics['free_energy_intervals'][name], diagnostics['free_energy_intervals']['reference'], CONVERGENCE)
        for family in PLAN['stratum_families']:
            for region in PLAN['regions']:
                for x, y in zip(reference['strata'][family][region], control['strata'][family][region]):
                    material = max(x['observed_class_fraction']['Qz'] or 0., y['observed_class_fraction']['Qz'] or 0.) >= CONVERGENCE['significant_stratum_mass_fraction']
                    strata.append(dict(control=name, family=family, region=region, bin=x['bin'],
                        reference_fraction=x['observed_class_fraction']['Qz'], control_fraction=y['observed_class_fraction']['Qz'],
                        material=material, comparisons={kind: compare_linear(x, y, kind) for kind in PLAN['quantities']}))
    failed = [s for s in strata if s['material'] and not s['comparisons']['Qz']['passed']]
    checks = dict(regional_mass_agreement=all(e['passed'] for arm in regions.values() for r in DECISION_CLASSES
            for e in arm[r].values()), total_hard_mass_agreement=all(a['total']['Q0']['passed'] for a in regions.values()),
        observed_regional_quality=all(e['passed'] for a in diagnostics['quality'].values() for e in a.values()),
        free_energy_contrast_agreement=all(e['passed'] for e in contrasts.values()),
        free_energy_precision=all(e['passed'] for e in diagnostics['free_energy_intervals'].values()),
        material_physical_strata_agreement=not failed,
        classifier_contact_consistency=all(a['native_entry_unbound_anomaly_count'] == 0 for a in selected.values()))
    return dict(schema='hard-free-line-sensitivity-comparison-v1', complete=True, plan=copy.deepcopy(PLAN),
        regional_comparisons=regions, free_energy_contrast_comparisons=contrasts,
        all_stratum_comparisons=strata, failed_material_strata=failed,
        stage_quality=diagnostics['quality'], stage_free_energy_intervals=diagnostics['free_energy_intervals'],
        checks=checks, sensitivity_checks_passed=all(checks.values()),
        total_attempts_by_stage=dict(reference_pilot=131072, controls=131072),
        stages_pooled=False, old_poses_replayed=False, old_native_classifier_calls=0,
        full_vessel_gate_open=False, assembly_gate_open=False,
        scope='One-factor independent guide/cloud controls against completed conditioned pilot. '
              'No stage pooling or old geometry/classifier replay. Population-size evidence and '
              'full-vessel coverage remain separate; even passing these checks is not an unseen-mode '
              'bound, an equilibrium mixing result or a finite-system assembly conclusion.')


def analyze(root):
    """Classify only the new audited controls once, with unchanged physical weights."""
    root = Path(root).resolve()
    protocol = read(root/'protocol.json')
    require(protocol['arms'] == CONTROLS and protocol['strata'] == STRATA
        and protocol['convergence'] == CONVERGENCE and protocol['total_unconditional_draws'] == 131072,
        'Unexpected fixed sensitivity allocation')
    destination = root/'comparison'
    require(not destination.exists(), 'No repeated sensitivity classification')
    destination.mkdir()
    classifier, binding = load_classifier(root/'common/reference-package/native-region/definition.json')
    arms = {}
    for arm in protocol['arms']:
        records, arrays = [], []
        for job in (j for j in protocol['jobs'] if j['arm'] == arm['id']):
            record, values = classify_population(root, destination, arm, job, protocol, classifier, binding)
            records.append(record)
            arrays.append(values)
        pooled = paired_moments(np.concatenate([a['z'] for a in arrays]), np.concatenate([a['h'] for a in arrays]))
        assessment = dict(estimate=dict(logQ=pooled['log_Qz'], draws=pooled['draws']),
            hard_region=dict(logQ=pooled['log_Q0'], draws=pooled['draws']),
            importance_sampling=dict(schema='defensive-hard-free-line-guide-v1', beta=arm['beta'],
                alpha=arm['alpha'], lambda_ratio=arm['lambda_ratio'], independent_geometry='full',
                preserved_attempts=pooled['draws']))
        arms[arm['id']] = relabel_nonuniform(summarize_arm(destination, arm, records, protocol['strata'], assessment))
    result = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(root/'protocol.json'), arms=arms,
        diagnostics=arm_diagnostics(arms), total_unconditional_draws=protocol['total_unconditional_draws'],
        native_definition=binding, old_samples_pooled=False, old_classifier_calls=0,
        unbound_R4_bound=unbound_volume_bound(read(root/'common/reference-package/region.json')))
    validate_summary(result, SCHEMA, CONTROLS)
    write(destination/'analysis.json', result)
    return result
