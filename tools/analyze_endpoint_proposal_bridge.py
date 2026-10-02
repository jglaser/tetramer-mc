#!/usr/bin/env python3
"""Retrospective alternative bridge from authenticated, already derived arrays.

Each arm uses its own frozen IID proposal g=q/J as its initial law. The arms
therefore have different intermediate targets. No raw poses, clouds, audits or
classifiers are replayed, and no new physical or SMC samples are generated.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from analyze_smc_bridge_bottleneck import (BETAS, REGIONS, DECISION, PLAN as OLD_PLAN,
    Inputs, masked_moments, aggregate_iid, fraction, population_statistics,
    local_dependencies, read, write, sha, require)

PLAN = dict(schema='endpoint-proposal-bridge-plan-v1', betas=BETAS,
    arms=['baseline', 'conditioned'], classes=REGIONS, strata=OLD_PLAN['strata'],
    populations_per_arm=4, attempts_per_population=16384, physical_activity=.035,
    lambda_ratio=128., clouds_per_valid_pose=2,
    formula='H * (J/q_IID)^beta * mean_r[exp(beta*z*L_r)*(1+beta*z/lambda)^K_r]',
    intermediate_target='H * (q_IID/J)^(1-beta) * exp(beta*z*C)',
    common_target_between_arms_only_at_beta1=True, exploratory=True,
    new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
    physical_gate_open=False)


def change_bridge(weights, log_hard, log_g, valid, betas=BETAS):
    """Change proposal factor only, retaining all attempted rows and endpoints."""
    weights, log_hard, log_g = map(lambda x: np.asarray(x, float), (weights, log_hard, log_g))
    valid, beta = np.asarray(valid), np.asarray(betas, float)
    require(valid.dtype == np.bool_ and valid.ndim == 1, 'Invalid validity mask')
    n = len(valid)
    require(weights.shape == (n, len(beta)) and log_hard.shape == log_g.shape == (n,)
        and np.isfinite(beta).all() and beta[0] == 0 and beta[-1] == 1
        and (np.diff(beta) > 0).all(), 'Unexpected bridge dimensions/grid')
    require(np.isfinite(weights[valid]).all() and np.isneginf(weights[~valid]).all()
        and np.isfinite(log_hard[valid]).all() and np.isfinite(log_g[valid]).all(),
        'Valid weights or invalid attempted zeros changed')
    result = np.full_like(weights, -np.inf)
    # Exclude invalid rows before arithmetic: 0*(-inf) must never be evaluated.
    result[valid] = weights[valid] - (1-beta)*(log_hard[valid]+log_g[valid])[:, None]
    cancellation = float(np.max(np.abs(result[valid, 0]), initial=0.))
    require(cancellation < 2e-10, 'Beta-zero cloud cancellation failed')
    result[valid, 0] = 0.
    result[:, -1] = weights[:, -1]  # identical physical endpoint, including zeros
    require(np.isfinite(result[valid]).all(), 'Nonfinite transformed valid weight')
    return result, cancellation


def paired_fraction_change(old_rows, new_rows):
    """Delta-method SE for log(new region fraction / old region fraction)."""
    def scaled(rows, key):
        a = np.array([-np.inf if r[key] is None else r[key] for r in rows])
        if not np.isfinite(a).any():
            return None
        return np.exp(a-float(a.max()))
    arrays = [scaled(rows, key) for rows in (new_rows, old_rows)
              for key in ('remaining_R4_native', 'total')]
    if any(a is None for a in arrays):
        return dict(observed=False, reason='Missing numerator/denominator, not a physical zero.')
    # Four measurements from the same population are correlated; preserve this.
    influence = sum(sign*a/a.mean() for sign, a in zip((1, -1, -1, 1), arrays))
    old = fraction(population_statistics(old_rows))
    new = fraction(population_statistics(new_rows))
    return dict(observed=True, log_new_over_old=new['log_ratio']-old['log_ratio'],
        fraction_ratio=new['fraction']/old['fraction'],
        paired_population_log_ratio_SE=float(influence.std(ddof=1)/2),
        scope='Paired descriptive comparison of different bridge targets, not an agreement test.')


def masks_from_records(a):
    valid = np.isfinite(a['z'])
    masks = dict(total=valid, registered_native_entry=valid & a['native'],
        old_R5_intersection_native=a['old_R5_intersection_native'],
        remaining_R4_native=a['remaining_R4_native'],
        contact_no_native_entry=valid & a['contact'] & ~a['native'],
        unbound_no_native_entry=valid & ~a['contact'] & ~a['native'])
    require(all(m.dtype == np.bool_ and not np.any(m & ~valid) for m in masks.values()),
        'Invalid class masks')
    require(np.array_equal(masks['registered_native_entry'],
        masks['old_R5_intersection_native'] | masks['remaining_R4_native'])
        and not np.any(masks['old_R5_intersection_native'] & masks['remaining_R4_native']),
        'Native partition changed')
    return masks


def analyze(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    require(sys.flags.optimize == 0 and not out.exists(), 'Assertions and fresh destination required')
    ledger = Inputs()
    old = ledger.json(source/'analysis.json')
    old_plan = ledger.json(source/'plan.json')
    require(old['schema'] == 'smc-bridge-bottleneck-retrospective-v1' and old['complete']
        and old['plan'] == OLD_PLAN and old_plan['betas'] == BETAS,
        'Completed compatible old-bridge retrospective required')
    pilot = Path(old_plan['pilot'])
    out.mkdir(parents=True)
    (out/'source').mkdir()
    for name, path in local_dependencies([Path(__file__)]).items():
        ledger.bind(path)
        shutil.copy2(path, out/'source'/name)
    write(out/'plan.json', dict(PLAN, source=str(source), input_sha256=copy.deepcopy(ledger.files)))
    started = time.process_time()
    iid = {}
    for arm in PLAN['arms']:
        populations = []
        originals = old['iid'][arm]['populations']
        require(len(originals) == 4 and len({p['seed'] for p in originals}) == 4,
            'Four independent source populations required')
        for p in originals:
            require(p['samples'] == 16384 and p['source_record']['arm'] == arm,
                'Source population identity changed')
            weight_path = ledger.bind(source/p['derived_weights'], p['derived_weights_sha256'])
            r = p['source_record']
            records_path = ledger.bind(pilot/'comparison'/r['records'], r['records_sha256'])
            require(old['source_sha256'][str(records_path)] == r['records_sha256'],
                'Classification records were not bound to old analysis')
            with np.load(weight_path, allow_pickle=False) as data:
                w = {k: data[k] for k in data.files}
            with np.load(records_path, allow_pickle=False) as data:
                a = {k: data[k] for k in data.files}
            require(np.array_equal(w['draw'], np.arange(16384))
                and np.array_equal(a['draw'], w['draw']) and np.all(a['source_n'] == 16384)
                and np.array_equal(w['betas'], BETAS), 'Attempt identities or beta grid changed')
            masks = masks_from_records(a)
            require(np.array_equal(np.isfinite(w['log_weights'][:, -1]), masks['total'])
                and np.max(np.abs(w['log_weights'][masks['total'], -1]-a['z'][masks['total']]), initial=0.) < 2e-10,
                'Physical endpoint differs from saved classification')
            weights, cancellation = change_bridge(w['log_weights'], a['h'], w['log_g'], masks['total'])
            profiles = []
            for bi, beta in enumerate(BETAS):
                classes = {name: masked_moments(weights[:, bi], m) for name, m in masks.items()}
                strata = {family: {name: [masked_moments(weights[:, bi],
                    masks[name] & (a['bin_'+family] == k)) for k in range(size)]
                    for name in DECISION} for family, size in PLAN['strata'].items()}
                profiles.append(dict(beta=beta, classes=classes, strata=strata))
            for name in REGIONS:
                require(profiles[-1]['classes'][name] == p['profiles'][-1]['classes'][name],
                    'Beta-one class moments changed')
                count = int(np.count_nonzero(masks[name]))
                actual = profiles[0]['classes'][name]
                require(actual['nonzero'] == count and (actual['logQ'] is None if count == 0
                    else abs(actual['logQ']-math.log(count/16384)) < 1e-12),
                    'Beta-zero mass is not unconditional proposal feasibility')
            require(profiles[-1]['strata'] == p['profiles'][-1]['strata'], 'Beta-one strata changed')
            populations.append(dict(id=p['id'], seed=p['seed'], samples=p['samples'], profiles=profiles,
                beta0_maximum_cancellation_error=cancellation, endpoint_moments_identical=True))
        profiles = aggregate_iid(populations)
        changes = []
        for bi, beta in enumerate(BETAS):
            rows = [[{name: p['profiles'][bi]['classes'][name]['logQ'] for name in REGIONS}
                for p in collection] for collection in (originals, populations)]
            changes.append(dict(beta=beta, **paired_fraction_change(*rows)))
        iid[arm] = dict(populations=populations, profiles=profiles, old_to_new_fraction=changes,
            old_profiles=old['iid'][arm]['profiles'])
    ledger.recheck()
    result = dict(schema='endpoint-proposal-bridge-retrospective-v1', complete=True, plan=PLAN,
        iid=iid, source_sha256=ledger.files, analysis_cpu_seconds=time.process_time()-started,
        new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
        raw_pose_or_cloud_rows_replayed=False, physical_gate_open=False,
        finite_system_conclusion='unresolved',
        scope='Each arm defines a different bridge below beta1. Old/new changes are paired '
            'descriptive comparisons, not common-target agreement tests. Beta0 estimates '
            'proposal feasibility, not hard-fluid equilibrium. The native-informed frozen '
            'guide remains native-informed. All beta points share poses/clouds. No SMC '
            'implementation or speedup, unseen-mode coverage, or assembly stability is established.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.source, args.out)['complete'])
