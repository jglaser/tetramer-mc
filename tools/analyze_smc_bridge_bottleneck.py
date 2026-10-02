#!/usr/bin/env python3
"""Exploratory bridge integrals from completed IID clouds, without new sampling.

Recompute the Poisson PGF at beta*z; never raise a saved weight to beta.
All beta values share poses/clouds and are correlated. Old labels and audits
are reused, not rerun. Intermediate integrals describe the old SMC bridge.
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
from scipy.special import logsumexp
from scipy.stats import t as student_t
from analyze_r4_smc_control import Proposal, CLASSES, population_statistics
from compare_hard_free_line_smc import Inputs, target, validate_unrestricted
from run_contact_tail_pilot import verify_frozen
from prepare_hard_free_line_score import read, sha, require, write
from prepare_shoulder_docking_benchmark import local_dependencies
from analyze_basin_normalizers import moments
from run_smc_guide_pilot import CONVERGENCE

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path('/vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001')
BRIDGE = Path('/vast/xvg/tetramer-mc-runs/hard-free-line-smc-bridge-preparation-v2-20261001')
BETAS = [i/8 for i in range(9)]
REGIONS = list(CLASSES)
DECISION = ['registered_native_entry', 'contact_no_native_entry', 'old_R5_intersection_native', 'remaining_R4_native']
PLAN = dict(schema='smc-bridge-bottleneck-retrospective-plan-v1', betas=BETAS,
    iid_arms=['baseline', 'conditioned'], smc_controls=['unrestricted_broad', 'unrestricted_narrow'],
    classes=REGIONS, strata={'radial': 3, 'angular': 3, 'orthant': 64},
    populations_per_arm=4, attempts_per_iid_population=16384, clouds_per_valid_pose=2,
    physical_activity=.035, lambda_ratio=128., current_bridge_probability=.5,
    formula='H * J/q_IID * g_SMC^(1-beta) * mean_r[exp(beta*z*L_r)*(1+beta*z/lambda)^K_r]',
    purpose='Distinguish an intermediate bridge bottleneck from SMC under-exploration of that bridge.',
    exploratory=True, new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
    old_geometry_audits_replayed=False, physical_gate_open=False)


def bridge_log_weights(log_hard, log_g, lower, counts, activity, intensity, betas=BETAS):
    """Two independent PGF estimators, averaged before multiplying J/q and g."""
    beta = np.asarray(betas, float)
    lower, counts = np.asarray(lower, float), np.asarray(counts)
    require(lower.shape == counts.shape == (2,) and np.isfinite(lower).all() and (lower >= 0).all()
        and np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all(), 'Invalid saved cloud pair')
    require(math.isfinite(log_hard) and math.isfinite(log_g) and activity >= 0 and math.isfinite(activity)
        and intensity > 0 and math.isfinite(intensity) and np.isfinite(beta).all()
        and ((beta >= 0) & (beta <= 1)).all(), 'Invalid bridge inputs')
    logs = beta[:, None]*activity*lower + np.log1p(beta[:, None]*activity/intensity)*counts
    return log_hard+(1-beta)*log_g+logsumexp(logs, axis=1)-math.log(2)


def masked_moments(weights, selected):
    return moments(np.where(selected, weights, -np.inf))


def aggregate_moments(populations):
    """Four independent population means and pooled IID-row concentration."""
    require(len(populations) == 4 and all(p['draws'] == 16384 for p in populations), 'Attempted denominator changed')
    logs = np.array([-np.inf if p['logQ'] is None else p['logQ'] for p in populations])
    if not np.isfinite(logs).any():
        return dict(observed=False, log_mass=None, population_relative_SE=None, importance_ESS=0.,
            largest_draw_fraction=None, draws=65536, passed=False, population_log_masses=[None]*4)
    scale = float(logs.max()); values = np.exp(logs-scale)
    log_mean = float(logsumexp(logs)-math.log(4))
    rse = float(values.std(ddof=1)/2/values.mean())
    sums = logs+math.log(16384)
    sum_total = float(logsumexp(sums))
    second = [2*sums[i]-math.log(p['ess']) for i, p in enumerate(populations) if p['nonzero']]
    ess = math.exp(2*sum_total-float(logsumexp(second)))
    maximum = max(math.exp(sums[i]-sum_total)*p['max_fraction'] for i, p in enumerate(populations) if p['nonzero'])
    return dict(observed=True, log_mass=log_mean, population_relative_SE=rse, importance_ESS=ess,
        largest_draw_fraction=maximum, draws=65536,
        passed=bool(rse <= .1 and ess >= 200 and maximum <= .02),
        population_log_masses=[None if not math.isfinite(v) else float(v) for v in logs])


def fraction(statistics, numerator='remaining_R4_native', denominator='total'):
    estimates, cov = statistics['estimates'], statistics['covariance_of_mean']
    a, b = estimates[numerator]['log_Q'], estimates[denominator]['log_Q']
    if a is None or b is None:
        return dict(observed=False, reason='Unobserved numerator/denominator is not a physical zero.')
    i, j = cov['class_order'].index(numerator), cov['class_order'].index(denominator)
    relative = cov['relative_matrix']
    variance = relative[i][i]+relative[j][j]-2*relative[i][j]
    require(variance >= -1e-12, 'Negative paired ratio variance')
    se = math.sqrt(max(0., variance)); ratio = math.exp(a-b)
    half = float(student_t.ppf(.975, 3))*se
    return dict(observed=True, fraction=ratio, log_ratio=a-b, paired_population_log_ratio_SE=se,
        approximate_interval_95=[math.exp(a-b-half), math.exp(a-b+half)],
        scope='Ratio of independent linear population means, with paired within-population covariance. '
              'Consistent ratio estimator, not finite-sample unbiased; all beta points are correlated.')


def compare_logs(left, right):
    """Independent four-population linear comparison at one common beta."""
    a = np.array([-np.inf if v is None else v for v in left], float)
    b = np.array([-np.inf if v is None else v for v in right], float)
    require(a.shape == b.shape == (4,), 'Four populations required')
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return dict(observed=False, passed=False, reason='Unobserved mass, not zero.')
    scale = max(float(a.max()), float(b.max()))
    x, y = np.exp(a-scale), np.exp(b-scale)
    mx, my = float(x.mean()), float(y.mean())
    se = math.hypot(float(x.std(ddof=1)/2), float(y.std(ddof=1)/2))
    delta = float(logsumexp(a)-logsumexp(b))
    abs_pass = abs(delta) <= .2
    se_pass = abs(mx-my) <= 3*se+32*np.finfo(float).eps*max(mx, my)
    return dict(observed=True, log_left_over_right=delta, ratio_left_over_right=math.exp(delta),
        scaled_linear_difference=mx-my, scaled_independent_SE=se, log_scale=scale,
        absolute_passed=bool(abs_pass), SE_passed=bool(se_pass), passed=bool(abs_pass and se_pass))


def bind_inputs(pilot, bridge, ledger):
    require(sys.flags.optimize == 0, 'Assertions must remain enabled')
    frozen = verify_frozen(bridge)
    for name, digest in frozen.items(): ledger.bind(bridge/name, digest)
    history = ledger.json(bridge/'historical.json')
    t = target(pilot, ledger)
    require(t == history['target'], 'Retrospective target differs from authenticated matching bridge')
    state = ledger.json(pilot/'status.json')
    require(state['complete'] and state['phase'] == 'complete'
        and len(state['jobs']) == len(state['audits']) == 8
        and all(j['status'] == 'complete' and j['returncode'] == 0 for j in state['jobs']+state['audits']),
        'Completed eight-population pilot/audits required')
    summary = ledger.json(pilot/'comparison/analysis.json', state['comparison_sha256'])
    require(summary['protocol_sha256'] == state['protocol_sha256'] == t['protocol_sha256']
        and summary['complete'] and summary['total_unconditional_draws'] == 131072
        and set(summary['arms']) == set(PLAN['iid_arms']), 'Incomplete/mismatched pilot summary')
    for name in PLAN['smc_controls']:
        control = history['controls'][name]
        validate_unrestricted(control['analysis'], t)
        require(control['analysis']['populations'] == control['populations'], 'Historical population copies differ')
        prior = control['analysis']['protocol_snapshot']['proposal']
        require(prior['current_ball_probability'] == prior['reference_ball_probability'] == .5
            and prior['reference_is_full_unfiltered_ball'] is True, 'Old SMC bridge mixture differs')
        for p in control['populations']:
            require(list(p['profiles']) == [str(i*16) for i in range(9)]
                and [p['stages'][i*16]['beta'] for i in range(9)] == BETAS, 'Saved bridge profile grid differs')
    return summary, history


def read_population(root, out, arm, record, proposal, summary, ledger):
    n = record['samples']; require(n == 16384, 'Unexpected pilot size')
    base = root/arm/record['id']
    manifest = ledger.json(base/'manifest.json')
    require(manifest['activity'] == .035 and manifest['lambda_ratio'] == 128
        and manifest['cloud_replicates'] == 2 and manifest['samples'] == n and manifest['seed'] == record['seed'],
        'Saved physical cloud/allocation differs')
    records_path = ledger.bind(root/'comparison'/record['records'], record['records_sha256'])
    audit = ledger.json(root/'audits'/arm/(record['id']+'.json'), record['independent_audit_sha256'])
    require(audit['complete'] and audit['geometry_mode'] == 'full' and audit['samples_sha256'] == record['samples_sha256'],
        'Saved full independent audit missing')
    require(audit['input_sha256'][str((base/'manifest.json').resolve())] == sha(base/'manifest.json')
        and manifest['lambda'] == manifest['activity']*manifest['lambda_ratio'],
        'Manifest or auxiliary intensity differs from the completed audit')
    with np.load(records_path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    require(np.array_equal(arrays['draw'], np.arange(n)) and np.all(arrays['source_n'] == n), 'Lost attempted draw identities')
    valid = np.isfinite(arrays['z'])
    masks = dict(total=valid, registered_native_entry=valid & arrays['native'],
        old_R5_intersection_native=arrays['old_R5_intersection_native'], remaining_R4_native=arrays['remaining_R4_native'],
        contact_no_native_entry=valid & arrays['contact'] & ~arrays['native'],
        unbound_no_native_entry=valid & ~arrays['contact'] & ~arrays['native'])
    require(np.array_equal(masks['registered_native_entry'], masks['old_R5_intersection_native'] | masks['remaining_R4_native'])
        and not np.any(masks['old_R5_intersection_native'] & masks['remaining_R4_native']), 'Native partition changed')
    weights = np.full((n, len(BETAS)), -np.inf); logg = np.full(n, -np.inf)
    endpoint_max = 0.; current_j_max = 0.; count = 0
    rows = ledger.bind(base/'samples.jsonl', record['samples_sha256'])
    with rows.open() as stream:
        for i, line in enumerate(stream):
            row = json.loads(line); count += 1
            require(i < n and row['draw'] == i, 'Missing/reordered attempted row')
            if not valid[i]:
                require(row['log_importance_weight'] is None, 'Invalid zero changed'); continue
            current, _, g = proposal.evaluate(row['pose'])
            require(current['inside'] and math.isfinite(g), 'Current target lost old SMC proposal support')
            require(len(row['clouds']) == 2, 'Missing pair of saved clouds')
            current_j_max = max(current_j_max, abs(current['log_jacobian']-row['log_physical_jacobian']))
            require(current_j_max < 2e-8 and abs(arrays['h'][i]-row['log_hard_weight']) < 1e-11,
                    'Coordinate measure or hard weight changed')
            logg[i] = g
            weights[i] = bridge_log_weights(arrays['h'][i], g,
                [c['lower_volume'] for c in row['clouds']], [c['overlap_points'] for c in row['clouds']],
                manifest['activity'], manifest['lambda'])
            endpoint_max = max(endpoint_max, abs(weights[i, -1]-arrays['z'][i]))
            require(abs(weights[i, 0]-(arrays['h'][i]+g)) < 1e-11, 'Beta zero did not eliminate cloud factors')
    require(count == n and endpoint_max < 2e-10, 'Endpoint weight or unconditional count changed')
    destination = out/arm/record['id']; destination.mkdir(parents=True)
    np.savez_compressed(destination/'bridge-weights.npz', draw=arrays['draw'], log_g=logg, log_weights=weights, betas=BETAS)
    profiles = []
    for bi, beta in enumerate(BETAS):
        class_moments = {name: masked_moments(weights[:, bi], mask) for name, mask in masks.items()}
        strata = {family: {name: [masked_moments(weights[:, bi], masks[name] & (arrays['bin_'+family] == k))
                    for k in range(size)] for name in DECISION} for family, size in PLAN['strata'].items()}
        profiles.append(dict(beta=beta, classes=class_moments, strata=strata))
    for name in REGIONS:
        expected = next(p for p in summary['estimates'][name]['populations'] if p['id'] == record['id'])['log_Qz']
        actual = profiles[-1]['classes'][name]['logQ']
        require(actual is None if expected is None else actual is not None and abs(actual-expected) < 2e-10,
                'Beta-one region mass does not reproduce saved complete classification')
    for family, size in PLAN['strata'].items():
        for name in DECISION:
            for k in range(size):
                expected = next(p for p in summary['strata'][family][name][k]['populations']
                                if p['id'] == record['id'])['log_Qz']
                actual = profiles[-1]['strata'][family][name][k]['logQ']
                require(actual is None if expected is None else actual is not None and abs(actual-expected) < 2e-10,
                        'Beta-one stratum mass does not reproduce saved classification')
    result = dict(id=record['id'], seed=record['seed'], samples=n, profiles=profiles,
        maximum_beta1_weight_error=endpoint_max, maximum_current_jacobian_error=current_j_max,
        derived_weights=str((destination/'bridge-weights.npz').relative_to(out)),
        derived_weights_sha256=sha(destination/'bridge-weights.npz'), source_record=record,
        new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0)
    write(destination/'population.json', result)
    return result


def aggregate_iid(populations):
    profiles = []
    for bi, beta in enumerate(BETAS):
        rows = [{name: p['profiles'][bi]['classes'][name]['logQ'] for name in REGIONS} for p in populations]
        statistics = population_statistics(rows)
        classes = {name: aggregate_moments([p['profiles'][bi]['classes'][name] for p in populations]) for name in REGIONS}
        strata = {family: {name: [aggregate_moments([p['profiles'][bi]['strata'][family][name][k] for p in populations])
            for k in range(size)] for name in DECISION} for family, size in PLAN['strata'].items()}
        profiles.append(dict(beta=beta, classes=classes, statistics=statistics,
            remainder_fraction=fraction(statistics), strata=strata))
    return profiles


def analyze(out, pilot=PILOT, bridge=BRIDGE):
    out, pilot, bridge = map(lambda p: Path(p).resolve(), (out, pilot, bridge))
    require(not out.exists(), 'Fresh one-pass retrospective destination required')
    ledger = Inputs(); completed, history = bind_inputs(pilot, bridge, ledger)
    common = pilot/'common/reference-package'
    current, reference = ledger.json(common/'region.json'), ledger.json(common/'old-r5-region.json')
    proposal = Proposal(current, reference, .5)
    out.mkdir(parents=True)
    sources = local_dependencies([Path(__file__)])
    (out/'source').mkdir()
    for name, path in sources.items(): ledger.bind(path); shutil.copy2(path, out/'source'/name)
    write(out/'plan.json', dict(PLAN, pilot=str(pilot), bridge=str(bridge), input_sha256=copy.deepcopy(ledger.files)))
    started = time.process_time(); iid = {}
    for arm in PLAN['iid_arms']:
        records = completed['arms'][arm]['populations']
        require(len(records) == 4 and len({r['seed'] for r in records}) == 4, 'Four IID populations required')
        populations = [read_population(pilot, out, arm, record, proposal, completed['arms'][arm], ledger) for record in records]
        iid[arm] = dict(populations=populations, profiles=aggregate_iid(populations))
    smc = {}
    for name in PLAN['smc_controls']:
        populations = history['controls'][name]['populations']; profiles = []
        for bi, beta in enumerate(BETAS):
            key = str(16*bi)
            rows = [p['profiles'][key]['log_masses'] for p in populations]
            statistics = population_statistics(rows)
            profiles.append(dict(beta=beta, statistics=statistics, remainder_fraction=fraction(statistics),
                populations=[dict(id=p['id'], seed=p['seed'], **p['profiles'][key],
                    ancestry=p['stages'][16*bi]['ancestry']) for p in populations]))
        smc[name] = dict(profiles=profiles)
    comparisons = []
    for bi, beta in enumerate(BETAS):
        a, b = iid['baseline']['profiles'][bi], iid['conditioned']['profiles'][bi]
        regions = {name: compare_logs(a['classes'][name]['population_log_masses'], b['classes'][name]['population_log_masses']) for name in REGIONS}
        strata = []
        for family, size in PLAN['strata'].items():
            for name in DECISION:
                for k in range(size):
                    x, y = a['strata'][family][name][k], b['strata'][family][name][k]
                    fractions = [0. if not child['observed'] else math.exp(child['log_mass']-parent['classes'][name]['log_mass'])
                                 for child, parent in [(x, a), (y, b)]]
                    strata.append(dict(family=family, region=name, bin=k, observed_fractions=fractions,
                        material=max(fractions) >= .01, comparison=compare_logs(x['population_log_masses'], y['population_log_masses'])))
        independent = {control: {arm: {region: compare_logs(
            [p['log_masses'][region] for p in smc[control]['profiles'][bi]['populations']],
            iid[arm]['profiles'][bi]['classes'][region]['population_log_masses'])
            for region in REGIONS} for arm in PLAN['iid_arms']} for control in PLAN['smc_controls']}
        comparisons.append(dict(beta=beta, iid_regions=regions, all_iid_strata=strata,
            failed_material_strata=[s for s in strata if s['material'] and not s['comparison']['passed']], smc_over_iid=independent))
    ledger.recheck()
    result = dict(schema='smc-bridge-bottleneck-retrospective-v1', complete=True, plan=PLAN,
        iid=iid, smc=smc, comparisons=comparisons, source_sha256=ledger.files,
        analysis_cpu_seconds=time.process_time()-started, new_pose_draws=0, new_Poisson_clouds=0,
        new_classifier_calls=0, old_geometry_audits_replayed=False, reused_saved_clouds=True,
        physical_gate_open=False, finite_system_conclusion='unresolved',
        scope='Exploratory reuse of completed IID poses, Poisson counts and fixed class masks. '
              'All beta points within an arm share their draws/clouds and are correlated. '
              'Only independent population replicates quantify uncertainty. Intermediate targets '
              'include g^(1-beta); they are not pure physical baths or assembly free energies. '
              'Endpoint agreement and observed concentration do not guarantee unseen-mode coverage.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); print(analyze(args.out)['complete'])
