#!/usr/bin/env python3
"""Predeclared aggregate-only population-size comparison; never pool stages."""
from __future__ import annotations
import copy
import math
import numpy as np
from scipy.special import logsumexp
from analyze_hard_free_line_physical import DECISION_CLASSES
from run_smc_guide_pilot import CONVERGENCE
from prepare_hard_free_line_score import require

PLAN = dict(schema='hard-free-line-population-size-comparison-plan-v1',
    samples_per_population=dict(pilot=16384, population_size=65536), populations_per_arm=4,
    arms=['baseline','conditioned'], regions=list(DECISION_CLASSES), quantities=['Qz','Q0'],
    stratum_families=['radial','angular','orthant'], convergence=copy.deepcopy(CONVERGENCE),
    stages_pooled=False, old_poses_replayed=False, old_native_classifier_calls=0,
    estimator='Independent linear means of four unconditional population masses per stage; '
              'SE of their difference combines independent population variances. '
              'Log ratios and relative errors are finite-sample diagnostics, not unbiased estimators.',
    material_stratum_rule='Observed within-region Qz fraction >=0.01 in either stage of the same arm; every stratum retained.',
    full_vessel_gate_open=False, assembly_gate_open=False)


def population_logs(estimate, kind, n):
    populations=estimate['populations']
    require(len(populations)==4 and len({p['id'] for p in populations})==4
        and len({p['seed'] for p in populations})==4,'Need four independent population means')
    require(all(p['draws']==n for p in populations) and estimate['row_uncertainty']['draws']==4*n,
            'Lost unconditional attempted denominator')
    logs=np.array([-math.inf if p['log_'+kind] is None else p['log_'+kind] for p in populations])
    require(not np.isnan(logs).any() and not np.isposinf(logs).any(),'Invalid population log mass')
    expected=estimate['population_uncertainty']['log_'+kind]
    mean=float(logsumexp(logs)-math.log(4)) if np.isfinite(logs).any() else None
    require(mean is None and expected is None or mean is not None and expected is not None
        and abs(mean-expected)<2e-10,'Saved mean is not the mean of linear population masses')
    return logs


def compare_linear(pilot, larger, kind='Qz'):
    a=population_logs(pilot,kind,16384);b=population_logs(larger,kind,65536)
    require({p['seed'] for p in pilot['populations']}.isdisjoint(p['seed']for p in larger['populations']),
            'Pilot and larger populations are not independent')
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return dict(observed=False,passed=False,pilot_population_log_masses=[None if not math.isfinite(x)else float(x)for x in a],
            larger_population_log_masses=[None if not math.isfinite(x)else float(x)for x in b],
            reason='A stage has unobserved mass: this is neither physical zero nor an upper bound.')
    # Independent local scales preserve finite log ratios even when one stage
    # is too small to represent on the shared linear display scale.
    local_a,local_b=float(a.max()),float(b.max())
    x,y=np.exp(a-local_a),np.exp(b-local_b)
    mx,my=float(x.mean()),float(y.mean());sx,sy=float(x.std(ddof=1)/2),float(y.std(ddof=1)/2)
    logma=local_a+math.log(mx);logmb=local_b+math.log(my)
    delta=logmb-logma;se=math.hypot(sx/mx,sy/my)
    require(math.isfinite(delta),'Unrepresentable log mass difference')
    scale=max(local_a,local_b);fa,fb=math.exp(local_a-scale),math.exp(local_b-scale)
    ma,mb=mx*fa,my*fb;sa,sb=sx*fa,sy*fb
    absolute=abs(delta)<=CONVERGENCE['log_agreement_absolute_max']
    linear_se=math.hypot(sa,sb)
    tolerance=32.*np.finfo(float).eps*max(ma,mb)
    statistical=abs(mb-ma)<=CONVERGENCE['log_agreement_combined_SE_max']*linear_se+tolerance
    return dict(observed=True,passed=bool(absolute and statistical),log_scale=scale,
        scaled_pilot_linear_mean=ma,scaled_larger_linear_mean=mb,
        log_pilot_linear_mean=logma,log_larger_linear_mean=logmb,
        shared_display_scale_underflow=dict(pilot=ma==0.,population_size=mb==0.),
        scaled_pilot_population_SE=sa,scaled_larger_population_SE=sb,
        scaled_larger_minus_pilot_linear_difference=mb-ma,
        scaled_independent_difference_SE=linear_se,scaled_comparison_roundoff_tolerance=tolerance,
        log_larger_minus_pilot=delta,combined_population_log_delta_SE=se,
        absolute_passed=bool(absolute),SE_passed=bool(statistical),
        absolute_limit=CONVERGENCE['log_agreement_absolute_max'],
        SE_multiplier=CONVERGENCE['log_agreement_combined_SE_max'],
        SE_test='abs(linear larger mean - pilot mean) <= multiplier * independent linear SE '
                '+ 32 machine eps * max(scaled means); log SE is diagnostic only',
        pilot_population_log_masses=[None if not math.isfinite(x)else float(x)for x in a],
        larger_population_log_masses=[None if not math.isfinite(x)else float(x)for x in b])


def compare_population_size(pilot, larger, plan):
    require(plan==PLAN,'Population-size analysis plan changed')
    for data,total in ((pilot,131072),(larger,524288)):
        require(data['schema']=='hard-free-line-physical-comparison-v1' and data['complete']
            and data['total_unconditional_draws']==total and set(data['arms'])==set(PLAN['arms']),
            'Incomplete or unexpected stage summary')
        require(not data['diagnostics']['full_vessel_gate_open'] and not data['diagnostics']['assembly_gate_open'],
                'Regional comparison cannot open physical conclusion gates')
    comparisons={};strata=[]
    for arm in PLAN['arms']:
        a,b=pilot['arms'][arm],larger['arms'][arm]
        comparisons[arm]={r:{kind:compare_linear(a['estimates'][r],b['estimates'][r],kind)
            for kind in PLAN['quantities']}for r in PLAN['regions']}
        for family in PLAN['stratum_families']:
            for region in PLAN['regions']:
                left,right=a['strata'][family][region],b['strata'][family][region]
                count=64 if family=='orthant' else 3
                require([s['bin']for s in left]==[s['bin']for s in right]==list(range(count)),
                        'Stratum allocation changed')
                for x,y in zip(left,right):
                    require(x['bin']==y['bin'],'Stratum identity changed')
                    material=max(x['observed_class_fraction']['Qz']or 0.,y['observed_class_fraction']['Qz']or 0.)>=CONVERGENCE['significant_stratum_mass_fraction']
                    strata.append(dict(arm=arm,family=family,region=region,bin=x['bin'],material=material,
                        comparison=compare_linear(x,y)))
    return dict(schema='hard-free-line-population-size-comparison-v1',complete=True,plan=copy.deepcopy(PLAN),
        regional_comparisons=comparisons,all_stratum_comparisons=strata,
        failed_material_strata=[s for s in strata if s['material'] and not s['comparison']['passed']],
        stage_quality={name:data['diagnostics']['quality']for name,data in [('pilot',pilot),('population_size',larger)]},
        stage_free_energy_intervals={name:data['diagnostics']['free_energy_intervals']for name,data in [('pilot',pilot),('population_size',larger)]},
        total_attempts_by_stage=dict(pilot=131072,population_size=524288),
        stages_pooled=False,old_poses_replayed=False,old_native_classifier_calls=0,
        full_vessel_gate_open=False,assembly_gate_open=False,
        scope='A fresh fixed fourfold population-size sensitivity check, with the same guide and cloud law. '
              'Every comparison is between independent population means; no historical rows, fits or classifiers '
              'are replayed. Existing within-stage diagnostics and all between-stage strata remain visible. '
              'Neither agreement nor failed sampling establishes finite-system assembly or instability.')
