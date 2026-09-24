#!/usr/bin/env python3
"""Evaluate the exact frozen pilot shear on eight separate archived populations.

No fitting, sampling, geometry/classification, old audit replay, or candidate
selection is performed.  This is a retrospective independent-population check;
the physical summaries have already been inspected and are not pristine data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
from scipy.special import logsumexp
from fit_kernel_shear import KernelMixture
from diagnose_fitted_kernel_shear import (
    CLASSES, STRATA, bind, combine_group, evaluate_group, group_masks,
    load_rows, read, recheck, require, sha, status, write_new,
)
from prepare_shoulder_docking_benchmark import local_dependencies

SCHEMA = 'frozen-kernel-shear-independent-evaluation-v1'
ARMS = ('bank', 'protected')
POPULATIONS = ('r00', 'r01', 'r02', 'r03')
N = 1048576
IMPORTANT_FRACTION = .01
PINS = dict(
    pilot_plan='c5465750e853f31d61a0f5eb141186e9561d36475f85fe7edaaab0e222c1ad80',
    pilot_analysis='8ade00d34dffcfc5beee68d0fc1fdac774f478a8720d20e544cd26e89f900992',
    model_freeze='1dae8533228cc4650b5a745817c5da5881318b4ecfb281442a7f633b5eea11f4',
    baseline='21b1c387a36d97fb73c28c6c6cb7d772c0500628a4e5836b66a804411362576c',
    fitted='d780ad5eccb1c7078d43bf1367d80fb400fd7d9dcd038e84c05b1e5bc4689268',
    evaluation_analysis='471f61cacf7253bf5746d8fc04765ea71ddfacfb07e205e12e50bf43742737dd',
    region='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    shape='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9')
SCOPE = ('Retrospective evaluation of the unchanged frozen kernel shear and '
    'affine control on eight populations independent of fitting/tuning. Existing '
    'physical summaries were inspected previously; this is not pristine '
    'prospective validation. Native-informed fixed84 geometry/weights and '
    'class-balanced training are unchanged. No fitting, candidate selection, '
    'new physical draws, geometry/native classifiers, old audit replays, '
    'operational guide replacement, gate promotion, sampling-speedup or '
    'finite-system assembly claim. Density-only CPU cost is descriptive and '
    'does not include a physical Monte Carlo kernel.')


def training_importance(pilot):
    """Select important groups using ONLY saved fitting-population mass sums."""
    training = [p for p in pilot['populations'] if p['role'] == 'training']
    require(len(training) == 4 and {(p['arm'], p['id']) for p in training} ==
            {(a, r) for a in ('bank', 'smc') for r in ('r00', 'r01')}, 'Pilot fitting split differs')
    result = {}
    for name in CLASSES:
        totals = [p['groups'][name]['likelihood']['log_weight_sum'] for p in training]
        require(all(v is not None and math.isfinite(v) for v in totals), 'Training class unobserved')
        total = float(logsumexp(totals))
        result[name] = dict(class_name=name, family=None, index=None, training_class_fraction=1.,
                            important=True, reason='Predeclared aggregate class')
        for family, size in STRATA.items():
            for index in range(size):
                key = f'{name}:{family}:{index}'
                logs = [p['groups'][key]['likelihood']['log_weight_sum'] for p in training]
                finite = [v for v in logs if v is not None]
                fraction = math.exp(float(logsumexp(finite))-total) if finite else 0.
                require(0 <= fraction <= 1.+1e-12, 'Training stratum mass exceeds its class')
                result[key] = dict(class_name=name, family=family, index=index,
                    training_class_fraction=fraction, important=fraction >= IMPORTANT_FRACTION,
                    reason='Training-only class fraction >=1%; every other group still retained')
    return result


def declared_datasets(comparison, directory, pilot):
    require(comparison['complete'] is True and comparison['region_sha256'] == PINS['region'] and
            comparison['shape_sha256'] == PINS['shape'], 'Evaluation physical target differs')
    prior_seeds = {p['seed'] for p in pilot['populations']}
    rows = []
    for arm in ARMS:
        records = comparison['arms'][arm]['populations']
        require(len(records) == 4 and {r['id'] for r in records} == set(POPULATIONS),
                'Exactly four independent evaluation populations per arm required')
        allocation = comparison['arms'][arm]['allocation']
        require(allocation['samples'] == N and allocation['alpha'] == .5 and allocation['lambda_ratio'] == 128.,
                'Evaluation cloud/proposal allocation differs')
        for record in sorted(records, key=lambda p: p['id']):
            require(record['samples'] == N and record['seed'] not in prior_seeds,
                    'Evaluation population allocation changed or seed overlaps fitting/tuning')
            rows.append(dict(arm=arm, id=record['id'], seed=record['seed'], samples=N,
                role='independent_retrospective', records=str((Path(directory)/record['records']).resolve()),
                records_sha256=record['records_sha256']))
    require(len({p['seed'] for p in rows}) == 8 and len({p['records'] for p in rows}) == 8,
            'Evaluation duplicates a seed or record archive')
    return rows


def same_frozen_affine(baseline, fitted):
    """Only additive warp parameters may differ; no updated mixture weights."""
    require(baseline['schema'] == fitted['schema'] == 'frozen-kernel-shear-mixture-v1', 'Unknown frozen model')
    for key in ('alpha', 'radius', 'transform'):
        require(baseline[key] == fitted[key], 'Frozen model support/transform differs')
    require(baseline['alpha'] == .5 and baseline['radius'] == 4 and
            len(baseline['charts']) == len(fitted['charts']) == len(baseline['weights']) == len(fitted['weights']) == 84,
            'Expected fixed84 defensive R4 models')
    require(np.allclose(baseline['weights'], fitted['weights'], atol=2e-15, rtol=0),
            'Frozen mixture weights differ')
    for a, b in zip(baseline['charts'], fitted['charts']):
        require(a['mean'] == b['mean'] and a['lower'] == b['lower'] and not a['shear']['centers'],
                'Frozen affine chart parameters differ or baseline is already warped')
        for key in ('dimension', 'conditioning', 'shifted'):
            require(a['shear'][key] == b['shear'][key], 'Frozen shear split differs')


def freeze(out, repository):
    out = Path(out).resolve(); repository = Path(repository).resolve()
    require(not out.exists(), 'Fresh independent evaluation directory required')
    require(sys.flags.optimize == 0, 'Disable Python optimization')
    bindings = {}; pilot_root = repository/'runs/fitted-kernel-shear-pilot-20260924'
    pilot_plan = read(bind(pilot_root/'plan.json', bindings, PINS['pilot_plan']))
    pilot = read(bind(pilot_root/'analysis.json', bindings, PINS['pilot_analysis']))
    pilot_status = read(bind(pilot_root/'status.json', bindings))
    require(pilot['complete'] is True and pilot_status['complete'] is True and
            pilot_status['analysis_sha256'] == PINS['pilot_analysis'] and
            pilot['plan_sha256'] == PINS['pilot_plan'] and pilot['fit']['fit_calls'] == 1 and
            pilot['fit']['tuning'] is False and pilot['fit']['holdout_rows_read'] == 0,
            'Pilot model was not frozen before its single evaluation')
    model_freeze = read(bind(pilot_root/'model-freeze.json', bindings, PINS['model_freeze']))
    for name, digest in model_freeze['files'].items(): bind(pilot_root/name, bindings, digest)
    baseline_path = bind(pilot_root/'baseline-model.json', bindings, PINS['baseline'])
    fitted_path = bind(pilot_root/'fitted-model.json', bindings, PINS['fitted'])
    same_frozen_affine(read(baseline_path), read(fitted_path))
    comparison = repository/'runs/protected-guide-validation-20260923/comparison'
    summary = read(bind(comparison/'analysis.json', bindings, PINS['evaluation_analysis']))
    datasets = declared_datasets(summary, comparison, pilot)
    for record in datasets: bind(record['records'], bindings, record['records_sha256'])
    important = training_importance(pilot)
    sources = local_dependencies([Path(__file__), Path(__file__).with_name('test_evaluate_frozen_kernel_shear.py')])
    # No updated scoring or model implementation can enter this comparison.
    for name, digest in pilot_plan['sources'].items():
        if name in sources:
            require(sha(sources[name]) == digest and sha(pilot_root/'source'/name) == digest,
                    'Scoring/model dependency differs from the frozen pilot: '+name)
            bind(pilot_root/'source'/name, bindings, digest)
    for name in ('diagnose_fitted_kernel_shear.py', 'fit_kernel_shear.py', 'kernel_shear.py'):
        require(name in sources and name in pilot_plan['sources'], 'Incomplete immutable scoring closure')
    for path in sources.values(): bind(path, bindings)
    out.mkdir(parents=True); (out/'source').mkdir(); (out/'pilot').mkdir()
    for name, path in sources.items(): shutil.copy2(path, out/'source'/name)
    for name in ('plan.json', 'analysis.json', 'model-freeze.json', 'fit.json',
                 'baseline-model.json', 'fitted-model.json'):
        shutil.copy2(pilot_root/name, out/'pilot'/name)
    plan = dict(schema=SCHEMA, scope=SCOPE, repository=str(repository), python=sys.executable,
        python_version=sys.version, python_sha256=sha(sys.executable),
        sources={name: sha(out/'source'/name) for name in sources},
        input_and_source_sha256=bindings, datasets=datasets, total_attempts=8*N,
        populations_per_source_arm=4, source_arms=list(ARMS), fit_calls=0,
        physical_draws=0, candidate_selection=False, fixed_candidate_sha256=PINS['fitted'],
        fixed_baseline_sha256=PINS['baseline'], fixed_model_freeze_sha256=PINS['model_freeze'],
        pilot_plan_sha256=PINS['pilot_plan'], pilot_analysis_sha256=PINS['pilot_analysis'],
        evaluation_analysis_sha256=PINS['evaluation_analysis'],
        important_group_fraction=IMPORTANT_FRACTION, training_only_group_declaration=important,
        all_groups_per_population=5*(1+sum(STRATA.values())),
        uncertainty='Every population retained separately; no IID row-level claim.',
        output_rows='Original cache arrays plus both complete density arrays; invalid and residual rows preserved.',
        density_timing='Single serial pass per population; process CPU seconds for baseline/warp density only. Descriptive, not a physical-kernel speedup.',
        workers=1)
    write_new(out/'plan.json', plan)
    write_new(out/'declaration-freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    recheck(bindings)
    return plan


def validate(out, expected):
    out = Path(out).resolve(); require(sha(out/'plan.json') == expected, 'Evaluation plan hash differs')
    plan = read(out/'plan.json')
    require(plan['schema'] == SCHEMA and plan['fit_calls'] == plan['physical_draws'] == 0 and
            plan['candidate_selection'] is False and plan['total_attempts'] == 8*N and plan['workers'] == 1,
            'Frozen no-fit evaluation allocation changed')
    require(plan['fixed_candidate_sha256'] == PINS['fitted'] and plan['fixed_baseline_sha256'] == PINS['baseline'] and
            plan['fixed_model_freeze_sha256'] == PINS['model_freeze'], 'Frozen candidate identity changed')
    require(sys.flags.optimize == 0 and sys.executable == plan['python'] and
            sys.version == plan['python_version'] and sha(sys.executable) == plan['python_sha256'],
            'Frozen evaluation runtime differs')
    for name, digest in read(out/'declaration-freeze.json')['files'].items():
        path = (out/name).resolve()
        require(path.is_relative_to(out) and sha(path) == digest, 'Frozen declaration/artifact changed')
    require(sha(__file__) == plan['sources'][Path(__file__).name], 'Evaluator source changed')
    recheck(plan['input_and_source_sha256'])
    same_frozen_affine(read(out/'pilot/baseline-model.json'), read(out/'pilot/fitted-model.json'))
    return plan


def group_diagnostics(groups, declaration):
    result = {}; newly_observed = []
    for name, rule in declaration.items():
        group = groups[name]; aggregate = groups[rule['class_name']]
        total = aggregate['likelihood']['log_weight_sum']; mass = group['likelihood']['log_weight_sum']
        fraction = math.exp(mass-total) if mass is not None and total is not None else 0.
        ratio = group['second_moments']['paired_physical']['log_warp_to_baseline_ratio']
        result[name] = dict(**rule, observed_class_fraction=fraction,
            mean_log_density_gain=group['likelihood']['mean_log_density_gain'],
            log_paired_M2_warp_to_baseline=ratio, paired_M2_improved=None if ratio is None else ratio < 0.)
        if rule['family'] is not None and rule['training_class_fraction'] == 0 and fraction >= IMPORTANT_FRACTION:
            newly_observed.append(name)
    return dict(groups=result, training_unobserved_but_evaluation_fraction_ge_1percent=newly_observed,
        important_regressions=[name for name, row in result.items() if row['important'] and
            row['paired_M2_improved'] is False],
        important_unobserved=[name for name, row in result.items() if row['important'] and
            row['paired_M2_improved'] is None],
        interpretation='Observed proposal tradeoffs only; these are not extra physical convergence gates.')


def comparison_plot(path, populations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4)); names = list(CLASSES)
    colors = ['#3465a4', '#75507b', '#c17d11']
    for index, name in enumerate(names):
        for pop in populations:
            replicate = POPULATIONS.index(pop['id']); arm_offset = -.11 if pop['arm'] == 'bank' else .11
            x = index+arm_offset+(replicate-1.5)*.028
            marker = 'o' if pop['arm'] == 'bank' else '^'; group = pop['groups'][name]
            gain = group['likelihood']['mean_log_density_gain']
            ratio = group['second_moments']['paired_physical']['log_warp_to_baseline_ratio']
            if gain is not None: axes[0].scatter(x, gain, marker=marker, color=colors[index], alpha=.8)
            if ratio is not None: axes[1].scatter(x, ratio, marker=marker, color=colors[index], alpha=.8)
    for ax in axes:
        ax.axhline(0., color='.5', lw=1); ax.set_xticks(range(3), ['Native\ninside R5', 'Native\nremainder', 'Contact\nno entry'])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Physical-weighted log-density gain\npositive favors frozen shear')
    axes[1].set_ylabel('log(paired M₂ shear / affine)\nnegative favors frozen shear')
    fig.suptitle('Eight independent populations · retrospective check\nCircles: bank source; triangles: protected source', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def run(out, expected):
    out = Path(out).resolve(); plan = validate(out, expected)
    require(Path(__file__).resolve() == out/'source'/Path(__file__).name, 'Run the archived evaluator')
    with (out/'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), plan_sha256=expected), stream)
    state = dict(complete=False, phase='evaluating', fit_calls=0, completed_populations=[])
    status(out/'status.json', state); started = time.monotonic()
    try:
        baseline = KernelMixture.from_dict(read(out/'pilot/baseline-model.json'))
        fitted = KernelMixture.from_dict(read(out/'pilot/fitted-model.json'))
        (out/'evaluations').mkdir(); populations = []
        for record in plan['datasets']:
            arrays = load_rows(record)
            before = time.process_time(); baseq = baseline.log_density(arrays['u']); baseline_cpu = time.process_time()-before
            before = time.process_time(); fitq = fitted.log_density(arrays['u']); fitted_cpu = time.process_time()-before
            require(baseq.shape == fitq.shape == (record['samples'],) and np.isfinite(baseq).all() and np.isfinite(fitq).all(),
                    'Nonfinite/incomplete full-mixture density')
            groups = {name: evaluate_group(arrays, baseq, fitq, mask) for name, mask in group_masks(arrays)}
            require(len(groups) == plan['all_groups_per_population'], 'A requested diagnostic group was dropped')
            target = out/'evaluations'/f"{record['arm']}-{record['id']}.npz"
            np.savez_compressed(target, **arrays, log_baseline_density=baseq, log_warped_density=fitq)
            populations.append(dict(**record, groups=groups,
                important_group_diagnostics=group_diagnostics(groups, plan['training_only_group_declaration']),
                attempted_rows_preserved=len(arrays['draw']),
                density_cpu_seconds=dict(baseline=baseline_cpu, warped=fitted_cpu),
                evaluated_rows=str(target.relative_to(out)), evaluated_rows_sha256=sha(target)))
            state['completed_populations'].append(record['arm']+'/'+record['id']); status(out/'status.json', state)
        aggregates = {}
        for arm in ARMS:
            chosen = [p for p in populations if p['arm'] == arm]
            require(len(chosen) == 4, 'Independent evaluation population was omitted')
            groups = {name: combine_group([p['groups'][name] for p in chosen]) for name in chosen[0]['groups']}
            aggregates[arm] = dict(groups=groups,
                important_group_diagnostics=group_diagnostics(groups, plan['training_only_group_declaration']),
                density_cpu_seconds={name: math.fsum(p['density_cpu_seconds'][name] for p in chosen)
                                     for name in ('baseline', 'warped')})
        # These copies are checked after all rows have been scored too: no tuning
        # or replacement is permitted even when a held-out tradeoff looks bad.
        require(sha(out/'pilot/baseline-model.json') == PINS['baseline'] and
                sha(out/'pilot/fitted-model.json') == PINS['fitted'], 'Frozen candidate changed during evaluation')
        recheck(plan['input_and_source_sha256'])
        comparison_plot(out/'independent-comparison.png', populations)
        result = dict(schema=SCHEMA, complete=True, scope=SCOPE, plan_sha256=expected,
            frozen_candidate_sha256=PINS['fitted'], frozen_baseline_sha256=PINS['baseline'],
            populations=populations, aggregates=aggregates,
            important_group_declaration=plan['training_only_group_declaration'],
            all_unconditional_draws_preserved=sum(p['attempted_rows_preserved'] for p in populations),
            fit_calls=0, candidate_selection=False, new_physical_draws=0, classifier_calls=0,
            geometry_predicate_calls=0, old_audits_replayed=0, operational_guide_written=False,
            gate_promotion=False, density_timing_scope=plan['density_timing'],
            source_and_input_sha256=plan['input_and_source_sha256'],
            plot_sha256=sha(out/'independent-comparison.png'), wall_seconds=time.monotonic()-started)
        require(result['all_unconditional_draws_preserved'] == plan['total_attempts'], 'Evaluation lost attempted rows')
        write_new(out/'analysis.json', result)
        state.update(complete=True, phase='complete', analysis_sha256=sha(out/'analysis.json')); status(out/'status.json', state)
        write_new(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob('*')) if p.is_file()}))
        return result
    except BaseException as error:
        state.update(complete=False, phase='failed', error=repr(error)); status(out/'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--out', type=Path, required=True); p.add_argument('--repository', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--out', type=Path, required=True); p.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.out, args.repository)
    else: run(args.out, args.expected_plan_sha256)


if __name__ == '__main__': main()
