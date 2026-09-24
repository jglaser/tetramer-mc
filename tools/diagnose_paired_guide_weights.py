#!/usr/bin/env python3
"""One retrospective paired-cloud allocation fit; no physical sampling.

Reuse the frozen 84-component dictionary and the earlier 21 protected groups.
The previously completed noisy-moment optimum is a fixed control, not refitted.
Only the original four training arrays are opened before the new weight freeze.
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

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import scipy

from diagnose_fitted_kernel_shear import (PINS as PILOT_PINS, bind, combine_group,
    evaluate_group, group_masks, load_rows, read, recheck, require, sha, status, write_new)
from prepare_contact_bank_guides import log_proposal
from prepare_shoulder_docking_benchmark import local_dependencies

SCHEMA = 'paired-physical-guide-allocation-diagnostic-v1'
PINS = dict(
    old_plan='76a3dffc63911fe82d349d735533d42d352fc79ff08a9fef84898efaff998b03',
    old_fit='4fae6ef241d87d155a07c6d72396d7be427bd03d63730e9e39d805ea762205c6',
    bank='d0e62d1b23d627582ca5f393af515a1ca10b8ecc01ce1a0e75c40e3e18885929',
    dictionary='20cbf3cc6081d5ce2f434d4189dc890fa2f3c1d1a4fcad320317bea39e8142e8')
FIT = dict(moment='paired', alpha=.5, floor=1e-5, maxiter=300, ftol=1e-9)
SCOPE = ('One fixed-dictionary physical-paired-cloud minimax allocation fit. '
    'The historical noisy-moment optimum and all21 protected groups are reused, '
    'not refitted or reselected. Previously inspected pilot holdouts provide '
    'retrospective diagnostics only. Native-informed proposal design, no new '
    'physical samples, geometry/classifier calls, old audit replay, production '
    'guide, convergence promotion, or assembly/stability inference. Both noisy '
    'and paired physical moments retain unconditional attempted denominators.')


def geometry(guide):
    return [{key: value for key, value in item.items() if key != 'weight'}
            for item in guide['gaussian_components']]


def with_weights(guide, weights):
    weights = np.asarray(weights, float)
    require(weights.shape == (len(guide['gaussian_components']),) and
            np.isfinite(weights).all() and (weights > 0).all() and
            abs(float(weights.sum())-1) <= 1e-10, 'Invalid frozen candidate weights')
    result = copy.deepcopy(guide)
    for component, weight in zip(result['gaussian_components'], weights):
        component['weight'] = float(weight)
    return result


def component_logs(u, guide):
    """Normalized, untruncated Gaussian densities and original R4 uniform law."""
    u = np.asarray(u, float)
    require(u.ndim == 2 and u.shape[1] == 6 and np.isfinite(u).all(), 'Invalid chart rows')
    columns = []
    for component in guide['gaussian_components']:
        lower = np.linalg.cholesky(np.asarray(component['covariance'], float))
        v = np.linalg.solve(lower, (u-np.asarray(component['mean'], float)).T).T
        columns.append(-3*math.log(2*math.pi)-np.log(np.diag(lower)).sum()-.5*np.sum(v*v, axis=1))
    log_volume = 3*math.log(math.pi)-math.lgamma(4)+6*math.log(4.)
    uniform = np.where(np.sum(u*u, axis=1) <= 16., -log_volume, -np.inf)
    return np.column_stack(columns), uniform


def protected_masks(arrays, groups):
    masks = {}
    for group in groups:
        mask = np.isfinite(arrays['z']) & (arrays['class_id'] == group['class_id'])
        if group['orthant'] is not None:
            mask &= arrays['bin_orthant'] == group['orthant']
        require(group['name'] not in masks and mask.any(), 'Missing/duplicate protected training group')
        masks[group['name']] = mask
    return masks


def freeze(out, repository):
    out, repository = Path(out).resolve(), Path(repository).resolve()
    require(not out.exists(), 'Fresh output required; no overwrite or retry')
    require(sys.flags.optimize == 0, 'Run without Python optimization')
    bindings = {}
    pilot = repository/'runs/smc-geometry-guide-pilot-20260923'
    analysis = read(bind(pilot/'comparison/analysis.json', bindings, PILOT_PINS['analysis']))
    require(analysis['complete'] is True and analysis['region_sha256'] == PILOT_PINS['region']
            and analysis['shape_sha256'] == PILOT_PINS['shape'], 'Original pilot target differs')
    previous = repository/'runs/contact-guide-stratum-reweighting-20260923'
    old_plan_path = bind(previous/'plan.json', bindings, PINS['old_plan'])
    old_fit_path = bind(previous/'analysis.json', bindings, PINS['old_fit'])
    old_plan, old_fit = read(old_plan_path), read(old_fit_path)
    require(old_fit['complete'] is True and old_fit['optimizer']['success'] is True and
            old_plan['protected_groups'] == old_fit['protected_groups'] and
            len(old_plan['protected_groups']) == 21 and not old_plan['missing_training_mandatory_groups'],
            'Frozen earlier fit/groups differ')
    bank_path = bind(pilot/'bank/provenance/importance-guide.json', bindings, PINS['bank'])
    dictionary_path = bind(pilot/'smc/provenance/importance-guide.json', bindings, PINS['dictionary'])
    bank, dictionary = read(bank_path), read(dictionary_path)
    require(len(geometry(bank)) == 80 and len(geometry(dictionary)) == 84 and
            geometry(dictionary)[:80] == geometry(bank), 'Fixed 80/84 dictionary order differs')
    require(bank['defensive_uniform_shell_probability'] == dictionary['defensive_uniform_shell_probability'] == .5
            and bank['region_sha256'] == dictionary['region_sha256'] == PILOT_PINS['region'],
            'Original region or defensive probability differs')
    old_weights = np.asarray(old_fit['candidate_weights']['protected_minimax'], float)
    with_weights(dictionary, old_weights)
    records = []
    for arm in ('bank', 'smc'):
        rows = analysis['arms'][arm]['populations']
        require(len(rows) == 4 and {r['id'] for r in rows} == {'r00', 'r01', 'r02', 'r03'},
                'Four independent populations required per source')
        for record in sorted(rows, key=lambda row: row['id']):
            path = bind(pilot/'comparison'/record['records'], bindings, record['records_sha256'])
            require(record['samples'] == 65536, 'Pilot allocation changed')
            records.append(dict(arm=arm, id=record['id'], seed=record['seed'], samples=65536,
                role='training' if record['id'] in ('r00', 'r01') else 'heldout',
                records=str(path), records_sha256=record['records_sha256']))
    require(len({r['seed'] for r in records}) == 8, 'Population seed reuse')
    sources = local_dependencies([Path(__file__), Path(__file__).with_name('fit_paired_guide_weights.py'),
        Path(__file__).with_name('test_fit_paired_guide_weights.py'),
        Path(__file__).with_name('test_diagnose_paired_guide_weights.py')])
    require('fit_paired_guide_weights.py' in sources, 'Missing optimizer source')
    for path in sources.values(): bind(path, bindings)
    out.mkdir(parents=True); (out/'source').mkdir()
    for name, path in sources.items(): shutil.copy2(path, out/'source'/name)
    shutil.copy2(bank_path, out/'bank.json'); shutil.copy2(dictionary_path, out/'dictionary.json')
    plan = dict(schema=SCHEMA, scope=SCOPE, fit=FIT, fit_calls=1, tuning=False,
        python=sys.executable, python_sha256=sha(sys.executable), python_version=sys.version,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        training_attempts=262144, heldout_attempts=262144, datasets=records,
        protected_groups=old_plan['protected_groups'], groups_reselected=False,
        reference_weights=old_fit['candidate_weights']['q80_reference'],
        historical_noisy_weights=old_weights.tolist(), historical_fit_repeated=False,
        objective='max_R [sum_R exp(logIW1+logIW2+log_qsource-log_qcandidate)] / '
                  '[sum_R exp(logIW1+logIW2+log_qsource-log_q80)]; each mass uses original N',
        all_strata_reported=True, source_and_input_sha256=bindings,
        sources={name: sha(out/'source'/name) for name in sources})
    write_new(out/'plan.json', plan)
    write_new(out/'declaration-freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    recheck(bindings)
    return plan


def validate(out, expected):
    out = Path(out).resolve()
    require(sha(out/'plan.json') == expected, 'Frozen plan differs')
    plan = read(out/'plan.json')
    require(plan['schema'] == SCHEMA and plan['fit'] == FIT and plan['fit_calls'] == 1 and
            plan['tuning'] is False and plan['historical_fit_repeated'] is False and
            plan['groups_reselected'] is False and
            plan['training_attempts'] == plan['heldout_attempts'] == 262144, 'Design differs')
    require(sys.flags.optimize == 0 and sys.executable == plan['python'] and
            sha(sys.executable) == plan['python_sha256'] and sys.version == plan['python_version'] and
            np.__version__ == plan['numpy_version'] and scipy.__version__ == plan['scipy_version'],
            'Frozen Python differs')
    recheck(plan['source_and_input_sha256'])
    for name, digest in read(out/'declaration-freeze.json')['files'].items():
        path = (out/name).resolve()
        require(path.is_relative_to(out) and sha(path) == digest, 'Declaration archive differs')
    require(sha(__file__) == plan['sources'][Path(__file__).name], 'Controller source differs')
    return plan


def run(out, expected):
    from fit_paired_guide_weights import build_problem, fit_weights
    out = Path(out).resolve(); plan = validate(out, expected)
    require(Path(__file__).resolve() == out/'source'/Path(__file__).name, 'Run archived controller')
    write_new(out/'claim.json', dict(pid=os.getpid(), plan_sha256=expected))
    state = dict(complete=False, phase='training', fit_calls=0, holdout_rows_before_freeze=0)
    status(out/'status.json', state); start = time.monotonic()
    try:
        dictionary = read(out/'dictionary.json'); bank = read(out/'bank.json')
        pieces = [load_rows(record) for record in plan['datasets'] if record['role'] == 'training']
        arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        attempts = len(arrays['draw']); require(attempts == plan['training_attempts'], 'Training allocation changed')
        # Physical zeros can be omitted from the optimizer's arithmetic only;
        # their original attempted denominator remains explicit and unchanged.
        valid = np.isfinite(arrays['z']); counts = dict(attempts=attempts,
            contributing_rows=int(valid.sum()), invalid_zero_rows=int((~valid).sum()))
        logs, uniform = component_logs(arrays['u'][valid], dictionary)
        masks = {name: mask[valid] for name, mask in protected_masks(arrays, plan['protected_groups']).items()}
        problem = build_problem(log_component_density=logs, log_uniform_density=uniform,
            log_source_density=arrays['log_q'][valid], log_cloud_importance=arrays['pairs'][valid],
            groups=masks, total_attempts=attempts, reference_weights=plan['reference_weights'],
            moment='paired', alpha=.5, floor=1e-5)
        state['fit_calls'] = 1; status(out/'status.json', state)
        fitted = fit_weights(problem, maxiter=FIT['maxiter'], ftol=FIT['ftol'])
        weights = list(fitted.weights)
        with_weights(dictionary, weights)
        write_new(out/'candidate-weights.json', dict(schema='diagnostic-frozen-weight-vector-v1',
            dictionary_sha256=sha(out/'dictionary.json'), weights=weights,
            operational_guide=False, fitted_moment='paired_physical'))
        write_new(out/'fit.json', dict(training=counts, fit_calls=1, holdout_rows_read=0,
            optimizer=fitted.to_dict(), objective=problem.metadata(), groups=plan['protected_groups'],
            historical_noisy_weights=plan['historical_noisy_weights']))
        write_new(out/'model-freeze.json', dict(files={name: sha(out/name) for name in
            ('plan.json', 'candidate-weights.json', 'fit.json')}))
        del arrays, pieces, logs, uniform, masks, problem
        candidate = with_weights(dictionary, read(out/'candidate-weights.json')['weights'])
        baseline = with_weights(dictionary, plan['historical_noisy_weights'])
        require(geometry(candidate) == geometry(baseline) == geometry(dictionary), 'Geometry was modified')
        state['phase'] = 'frozen_evaluation'; status(out/'status.json', state)
        (out/'evaluations').mkdir(); populations = []
        for record in plan['datasets']:
            arrays = load_rows(record)
            source = log_proposal(arrays['u'], bank if record['arm'] == 'bank' else dictionary)
            error = float(np.max(np.abs(source-arrays['log_q'])))
            require(error < 2e-8, 'Archived complete source density differs')
            before = time.process_time(); q0 = log_proposal(arrays['u'], baseline); baseline_cpu = time.process_time()-before
            before = time.process_time(); q1 = log_proposal(arrays['u'], candidate); candidate_cpu = time.process_time()-before
            require(np.isfinite(q0).all() and np.isfinite(q1).all(), 'Invalid complete density')
            groups = {name: evaluate_group(arrays, q0, q1, mask) for name, mask in group_masks(arrays)}
            path = out/'evaluations'/f"{record['arm']}-{record['id']}.npz"
            np.savez_compressed(path, **arrays, log_noisy_fit_density=q0, log_paired_fit_density=q1)
            populations.append(dict(**record, groups=groups, attempts_preserved=len(arrays['draw']),
                source_density_max_error=error, baseline_density_cpu_seconds=baseline_cpu,
                candidate_density_cpu_seconds=candidate_cpu, rows=str(path.relative_to(out)), rows_sha256=sha(path)))
        aggregates = {}
        for role in ('training', 'heldout'):
            for arm in ('bank', 'smc'):
                chosen = [p for p in populations if p['role'] == role and p['arm'] == arm]
                require(len(chosen) == 2, 'Original split differs')
                aggregates[role+'_'+arm] = {name: combine_group([p['groups'][name] for p in chosen])
                    for name in chosen[0]['groups']}
        recheck(plan['source_and_input_sha256'])
        for name, digest in read(out/'model-freeze.json')['files'].items():
            require(sha(out/name) == digest, 'Candidate changed during evaluation')
        result = dict(schema=SCHEMA, complete=True, scope=SCOPE, plan_sha256=expected,
            fit=read(out/'fit.json'), populations=populations, aggregates=aggregates,
            original_attempts_preserved=sum(p['attempts_preserved'] for p in populations),
            new_physical_draws=0, classifier_calls=0, geometry_calls=0, audits_replayed=0,
            protected_validation_rows_read=0, historical_fit_repeated=False, operational_guide_written=False,
            gate_promotion=False, wall_seconds=time.monotonic()-start,
            limitations='Retrospective second moments; two heldout populations/source; unstable or unseen tails '
                'remain unresolved. No Monte Carlo CPU efficiency or physical convergence conclusion.')
        write_new(out/'analysis.json', result)
        state.update(complete=True, phase='complete', analysis_sha256=sha(out/'analysis.json'))
        status(out/'status.json', state)
        write_new(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob('*')) if p.is_file()}))
        return result
    except BaseException as error:
        state.update(complete=False, phase='failed', error=repr(error)); status(out/'status.json', state)
        write_new(out/'failure.json', state)
        write_new(out/'failure-freeze.json', dict(files={str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob('*')) if p.is_file()}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--repository', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--out', type=Path, required=True); p.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.out, args.repository)
    else: run(args.out, args.expected_plan_sha256)


if __name__ == '__main__': main()
