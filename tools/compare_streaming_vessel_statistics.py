#!/usr/bin/env python3
"""Stage arithmetic for the full-vessel streaming partition.

Consumes authenticated population summaries in a future dispatcher; this module
does not authenticate files, execute physics, or open an assembly gate. Existing
global covariance/contrast code is reused; all new regional strata are retained.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from scipy.stats import t as student_t

import compare_full_vessel_stage as previous
from analyze_r4_smc_control import close, require
from partition_vessel_streaming import SCHEMA as PARTITION_SCHEMA, REGIONAL, STRATA, BIN_COUNTS
from vessel_contact_partition import independent_mass

SCHEMA = 'full-vessel-streaming-stage-statistics-v1'
DECISION_REGIONS = REGIONAL[:-1]
MATERIAL_FRACTION = .01


def same_mass(left, right):
    require(left['draws'] == right['draws'] and left['nonzero'] == right['nonzero'], 'Changed regional denominator/support')
    for field in ('logQ', 'log_sum_squared_weights', 'log_max_weight'):
        require((left[field] is None) == (right[field] is None), 'Changed regional zero/nonzero mass')
        if left[field] is not None: close(left[field], right[field], 'Changed regional moments')


def validate_partition(value, n):
    require(value['schema'] == PARTITION_SCHEMA and value['complete'] is True, 'Completed streaming partition required')
    # The global mass-statistic contract is unchanged. This view reuses its
    # algebraic checks only, not the historical schema's provenance claims.
    previous.validate_partition(dict(value, schema='full-vessel-contact-partition-v1'), n)
    require(value['stratum_definition'] == STRATA and set(value['regional_estimates']) == set(REGIONAL)
            and set(value['strata']) == set(BIN_COUNTS), 'Changed reporting regions or fixed strata')
    require(type(value['new_native_classifier_calls']) is int
            and value['new_native_classifier_calls'] == n-value['invalid_draws']
            and type(value['native_unbound_anomalies']) is int
            and 0 <= value['native_unbound_anomalies'] <= value['new_native_classifier_calls'], 'Classifier accounting differs')
    for name, kinds in value['regional_estimates'].items():
        require(set(kinds) == set(previous.KINDS), 'Missing hard or physical mass')
        for kind, mass in kinds.items():
            previous.check_mass(mass, n)
            if name in previous.PRIMARY:
                same_mass(mass, value['estimates'][name+':inside_current_R4'][kind])
        require(kinds['Qz']['nonzero'] == kinds['Q0']['nonzero'], 'Hard and physical regional support differs')
    previous.partition_sum(value['regional_estimates'], REGIONAL[0], REGIONAL[2:4])
    for family, size in BIN_COUNTS.items():
        groups = value['strata'][family]
        require(set(groups) == set(REGIONAL), 'Missing class in fixed stratum family')
        for name, bins in groups.items():
            require(len(bins) == size, 'Dropped or added stratum')
            for kinds in bins:
                require(set(kinds) == set(previous.KINDS), 'Missing stratum mass kind')
                for mass in kinds.values(): previous.check_mass(mass, n)
                require(kinds['Qz']['nonzero'] == kinds['Q0']['nonzero'], 'Stratum physical/hard support differs')
            previous.partition_sum(dict(parent=value['regional_estimates'][name], **{str(i): b for i,b in enumerate(bins)}),
                                   'parent', [str(i) for i in range(size)])


def linear_agreement(left, right):
    """Difference of independent linear means, with a separate log-size limit."""
    if left['log_Q'] is None or right['log_Q'] is None:
        return dict(passed=False, observed=False, state='unresolved',
                    unresolved='Unobserved contribution is not a zero-mass result or an upper bound.')
    la, lb = left['log_Q'], right['log_Q']
    ra, rb = left['population_relative_SE'], right['population_relative_SE']
    require(all(math.isfinite(x) for x in (la,lb,ra,rb)) and min(ra,rb) >= 0, 'Invalid population mean or SE')
    scale = max(la,lb); a,b = math.exp(la-scale),math.exp(lb-scale)
    error = math.hypot(a*ra,b*rb); difference = a-b
    roundoff = 32*np.finfo(float).eps*max(a,b)
    absolute = abs(la-lb) <= previous.THRESHOLDS['between_arm_absolute_max']
    statistical = abs(difference) <= previous.THRESHOLDS['between_arm_SE_multiple']*error+roundoff
    return dict(passed=bool(absolute and statistical), observed=True,
        state='corroborated' if absolute and statistical else 'material_disagreement' if not absolute and not statistical else 'unresolved',
        log_half_mixture_minus_vessel=la-lb, log_display_scale=scale,
        scaled_half_mixture_mean=a, scaled_vessel_mean=b,
        scaled_linear_difference=difference, scaled_independent_population_SE=error,
        scaled_roundoff_tolerance=roundoff, absolute_passed=bool(absolute), three_linear_SE_passed=bool(statistical),
        comparison='Independent linear population means; finite log ratio supplies the separate 0.2 size check.')


def mass_statistics(populations, values):
    result = independent_mass(values)
    checks = dict(population_RSE=result['population_relative_SE'] is not None and result['population_relative_SE'] <= .1,
        importance_ESS=result['importance_ESS'] >= 200,
        largest_draw=result['largest_contribution'] is not None and result['largest_contribution'] <= .02)
    result.update(quality_checks=checks, passed=all(checks.values()),
        populations=[dict(id=p['id'], seed=p['seed'], **v) for p,v in zip(populations,values)])
    if result['log_Q'] is not None:
        half = float(student_t.ppf(.975,3))*result['population_relative_SE']
        result.update(log_mass_halfwidth_95=half, log_mass_interval_95=[result['log_Q']-half,result['log_Q']+half])
    return result


def fraction(child, parent):
    if child['log_Q'] is None: return None
    require(parent['log_Q'] is not None, 'Positive child has unobserved parent')
    answer = math.exp(child['log_Q']-parent['log_Q'])
    require(answer <= 1+1e-8, 'Observed child exceeds its parent')
    return answer


def compare(populations, stage):
    require(stage in dict(previous.STAGES), 'Select one frozen stage')
    n = dict(previous.STAGES)[stage]
    require(len(populations) == 8 and len({p['id'] for p in populations}) == 8
            and len({p['seed'] for p in populations}) == 8, 'Eight independent populations required')
    for population in populations:
        require(population['samples'] == n and population['stage'] == stage, 'Mixed stages or changed denominator')
        validate_partition(population['partition_data'], n)
        require(population['estimates'] == population['partition_data']['estimates'], 'Different global mass view')
    # Keep the previous joint covariance and native/contact and native/all-noentry
    # contrasts. Do not duplicate the new per-stratum records in this global view.
    view = [{k:v for k,v in p.items() if k != 'partition_data'} for p in populations]
    result = previous.compare(view,stage); result['schema'] = SCHEMA
    arms = result['arms']
    result['between_arms'] = {kind:{name:linear_agreement(arms['half_mixture'][kind]['estimates'][name],
            arms['vessel'][kind]['estimates'][name]) for name in previous.CLASSES} for kind in previous.KINDS}
    regional = {}; strata = {}
    for arm in previous.ARMS:
        selected = sorted([p for p in populations if p['arm'] == arm],key=lambda p:p['population'])
        regional[arm] = {kind:{name:mass_statistics(selected,[p['partition_data']['regional_estimates'][name][kind] for p in selected])
                              for name in REGIONAL} for kind in previous.KINDS}
        strata[arm] = {family:{name:[{kind:mass_statistics(selected,
            [p['partition_data']['strata'][family][name][i][kind] for p in selected]) for kind in previous.KINDS}
            for i in range(size)] for name in REGIONAL} for family,size in BIN_COUNTS.items()}
    regional_comparison = {kind:{name:linear_agreement(regional['half_mixture'][kind][name],regional['vessel'][kind][name])
                                for name in REGIONAL} for kind in previous.KINDS}
    stratum_comparisons = []
    for family,size in BIN_COUNTS.items():
        for name in REGIONAL:
            for i in range(size):
                fractions = {arm:fraction(strata[arm][family][name][i]['Qz'],regional[arm]['Qz'][name]) for arm in previous.ARMS}
                material = any(v is not None and v >= MATERIAL_FRACTION for v in fractions.values())
                stratum_comparisons.append(dict(family=family, region=name, bin=i, observed_regional_fractions=fractions,
                    material=material, decision_region=name in DECISION_REGIONS,
                    comparisons={kind:linear_agreement(strata['half_mixture'][family][name][i][kind],
                        strata['vessel'][family][name][i][kind]) for kind in previous.KINDS}))
    remainder = []
    for parent in previous.PRIMARY:
        for part in ('outside_current_R4','outside_measured_pockets'):
            name = parent+':'+part
            fractions = {arm:fraction(arms[arm]['Qz']['estimates'][name],arms[arm]['Qz']['estimates'][parent]) for arm in previous.ARMS}
            material = any(v is not None and v >= MATERIAL_FRACTION for v in fractions.values())
            remainder.append(dict(region=name, observed_primary_fractions=fractions, material=material,
                qualities={arm:arms[arm]['Qz']['estimates'][name]['passed'] for arm in previous.ARMS},
                comparison=result['between_arms']['Qz'][name],
                unresolved_unobserved=any(v is None for v in fractions.values())))
    checks = result['diagnostics']['checks']
    checks['primary_Qz_and_Q0_agreement'] = all(result['between_arms'][k][c]['passed']
        for k in previous.KINDS for c in ('total',*previous.PRIMARY))
    checks.update(regional_Qz_quality=all(regional[a]['Qz'][name]['passed'] for a in previous.ARMS for name in DECISION_REGIONS),
        regional_Qz_and_Q0_agreement=all(regional_comparison[k][name]['passed'] for k in previous.KINDS for name in DECISION_REGIONS),
        material_regional_strata_agreement=all(s['comparisons']['Qz']['passed'] for s in stratum_comparisons
                                               if s['material'] and s['decision_region']),
        material_remainder_quality_and_agreement=all(all(s['qualities'].values()) and s['comparison']['passed']
                                                     for s in remainder if s['material']),
        classifier_contact_consistency=all(p['partition_data']['native_unbound_anomalies'] == 0 for p in populations))
    result['diagnostics'] = dict(checks=checks, declared_observed_diagnostics_passed=all(checks.values()),
        unresolved_unobserved_remainders=[r['region'] for r in remainder if r['unresolved_unobserved']],
        scope='Observed moment and sensitivity diagnostics only. Unobserved remainder is not certified absent; '
              'no physical conclusion or assembly gate follows even when these checks pass.')
    result.update(regional_estimates=regional, regional_comparisons=regional_comparison, strata=strata,
        stratum_definition=copy.deepcopy(STRATA), stratum_comparisons=stratum_comparisons,
        remainder_diagnostics=remainder, material_fraction=MATERIAL_FRACTION,
        material_rule='Observed Qz child/its regional or primary parent >=0.01 in either arm. '
                      'A within-R4 stratum never uses full-vessel native mass as its denominator.',
        input_authentication_performed=False,
        scope=result['scope']+' This arithmetic layer expects an independently authenticated campaign loader; '
              'no file provenance or dispatch authorization is supplied by these calculations.')
    return result
