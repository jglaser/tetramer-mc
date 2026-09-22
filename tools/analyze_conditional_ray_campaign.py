#!/usr/bin/env python3
"""Classify completed conditional-ray references; never launch physics or raw audits.

All six arms remain separate estimators. Every attempted pose, including zeros,
retains the original denominator. Passing finite-region numerical diagnostics
would not establish full-wall coverage, physical kinetics, or assembly stability.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import t as student_t
from analyze_mobile_native_pocket import require, read, write, sha, inside, load_classifier, local_sources
from analyze_mobile_competing_reference import paired_moments, check_estimate
from analyze_mobile_threshold_reference import ExclusionContact, validate_classifier_target
from analyze_mobile_wall_contacts import PRIMARY, summarize_regions
from run_conditional_ray_campaign import validate as validate_frozen, verify_output

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('total',)+PRIMARY
DECISION_CLASSES = PRIMARY[:2]
COMPARISONS = (
    ('pilot_ray', 'pilot_uniform'), ('large_ray', 'large_uniform'),
    ('large_ray', 'pilot_ray'), ('large_uniform', 'pilot_uniform'),
    ('alpha02', 'pilot_ray'), ('lambda128', 'pilot_ray'))
SCOPE = ('Six separate unconditional estimates on the unchanged finite R4 and two-neighbor scaffold. '
         'Native entry, contact without native entry, and unbound without entry partition every hard-valid pose. '
         'No-entry is a catalogue/threshold class, not an identified competing basin. '
         'The guide changes only proposal allocation, never physical weights. '
         'Observed errors and importance ESS are not tail bounds or MCMC mixing measurements; '
         'failed convergence is unresolved sampling, not evidence against assembly. '
         'The full-wall complement and finite assembly stability remain separate unmeasured targets.')


def validate_terminal(root):
    root = Path(root).resolve(); protocol = validate_frozen(root); status = read(root/'status.json')
    require(status.get('schema') == 'conditional-ray-reference-status-v1' and status.get('complete') is True
            and status.get('phase') == 'complete', 'Campaign has not completed physics and all raw audits')
    require(status.get('protocol_sha256') == sha(root/'protocol.json'), 'Terminal protocol binding changed')
    expected = {(j['arm'], j['id']): j for j in protocol['jobs']}
    terminal = {(j['arm'], j['id']): j for j in status['jobs']}
    require(len(terminal) == len(status['jobs']) == len(expected) and set(terminal) == set(expected), 'Missing/duplicate terminal population')
    require(set(status['audits']) == {a['id'] for a in protocol['arms']}, 'Missing/extra arm audit')
    require(len({j['seed'] for j in protocol['jobs']}) == len(protocol['jobs']), 'Independent population seeds reused')
    for key, job in expected.items():
        term = terminal[key]
        require(term['status'] == 'complete' and term['returncode'] == 0 and all(term[k] == v for k, v in job.items()), 'Population not terminal or allocation changed')
        require(verify_output(root, protocol, term) == term['output'], 'Terminal output binding changed')
    assessments = {}
    for arm in protocol['arms']:
        name = arm['id']; base = root/name; record = status['audits'][name]; assessment = read(base/'assessment/analysis.json')
        require(record['returncode'] == 0 and record['analysis_sha256'] == sha(base/'assessment/analysis.json'), 'Raw audit failed or changed')
        jobs = [j for j in protocol['jobs'] if j['arm'] == name]; n = sum(j['samples'] for j in jobs)
        require(assessment['estimate']['draws'] == assessment['hard_region']['draws'] == assessment['independently_reconstructed_poses'] == n,
                'Raw audit discarded attempts')
        require(assessment['region_sha256'] == protocol['region_sha256'] and len(assessment['populations']) == len(jobs)
                and {p['id'] for p in assessment['populations']} == {j['id'] for j in jobs}, 'Audit target/populations differ')
        for job in jobs:
            population = next(p for p in assessment['populations'] if p['id'] == job['id'])
            require(population['seed'] == job['seed'] and population['samples_sha256'] == terminal[(name, job['id'])]['output']['samples_sha256']
                    and population['estimate']['draws'] == population['hard_region']['draws'] == job['samples'], 'Audit row binding changed')
        guide = assessment['importance_sampling']
        require(guide['uniform_shell_probability'] == arm['alpha'] and guide['conditional_ray_component_count'] == 3
                and guide['guide_schema'] == 'defensive-conditional-ray-guide-v1' and guide['shell_rejected'] == 0, 'Audited guide differs')
        assessments[name] = assessment
    definition = inside(root, protocol['native_definition'])
    require(sha(definition) == protocol['native_definition_sha256'], 'Frozen native definition changed')
    for arm in protocol['arms']:
        validate_classifier_target(read(root/arm['id']/'provenance/config.json'), protocol['shape_sha256'], read(definition), definition)
    return protocol, status, assessments, definition


def read_weights(rows, samples, region, arm):
    require(len(rows) == samples and [r['draw'] for r in rows] == list(range(samples)), 'Missing/repeated unconditional draws')
    z = np.full(samples, -np.inf); h = z.copy(); pairs = np.full((samples, 2), -np.inf)
    branch = np.zeros(samples, dtype=np.int8); fallback = np.full(samples, -1, dtype=np.int8)
    component = np.full(samples, -1, dtype=np.int8); latents = np.empty((samples, 6))
    for i, row in enumerate(rows):
        require(all(type(row[k]) is bool for k in ('hard_valid', 'capture_valid', 'region_valid', 'shell_valid')), 'Missing Boolean support flags')
        require(row['shell_valid'] and row['capture_valid'] and row['region_valid'], 'Conditional reference left R4 or changed capture/q target')
        require(math.isfinite(row['latent_radius']) and 0 <= row['latent_radius'] <= region['mahalanobis_radius'], 'Saved radius outside R4')
        require(math.isfinite(row['q']) and row['q'] >= 0 and all(math.isfinite(row[k]) for k in ('log_physical_jacobian', 'log_proposal_density')), 'Nonfinite metric/density')
        latents[i] = row['latent']; require(np.isfinite(latents[i]).all(), 'Nonfinite latent row')
        if row['proposal_branch'] == 'uniform-shell':
            require(row['proposal_component'] is None and row['selected_ray_fallback'] is None, 'Invalid uniform branch metadata')
        else:
            require(row['proposal_branch'] == 'conditional-ray' and arm['alpha'] < 1 and type(row['proposal_component']) is int
                    and 0 <= row['proposal_component'] < 3 and type(row['selected_ray_fallback']) is bool, 'Invalid conditional-ray branch metadata')
            branch[i] = 1; fallback[i] = int(row['selected_ray_fallback']); component[i] = row['proposal_component']
        if not row['hard_valid']:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'], 'Invalid draw must retain a zero')
            continue
        require(len(row['clouds']) == 2 and math.isfinite(row['log_hard_weight']) and math.isfinite(row['log_importance_weight']), 'Missing two-cloud weight')
        h[i] = row['log_hard_weight']; z[i] = row['log_importance_weight']
        require(abs(h[i]-row['log_physical_jacobian']+row['log_proposal_density']) < 2e-10, 'Saved weight is not J/q')
        pairs[i] = [h[i]+cloud['log_weight'] for cloud in row['clouds']]
        require(np.isfinite(pairs[i]).all() and abs(float(logsumexp(pairs[i])-math.log(2))-z[i]) < 2e-10, 'Two-cloud mean changed')
    return dict(z=z, h=h, pairs=pairs, branch=branch, fallback=fallback, component=component, latents=latents)


def stratify(latents, region, specification):
    """Fixed original-chart bins; final radial/angular bin includes its endpoint."""
    u = np.asarray(latents, float); require(u.ndim == 2 and u.shape[1] == 6 and np.isfinite(u).all(), 'Bad stratification coordinates')
    chart = region['gaussian_chart']; covariance = np.asarray(chart['covariances'][0], float)
    lower = np.linalg.cholesky(covariance); angular_lower = np.linalg.cholesky(covariance[3:, 3:])
    angular = u@lower[3:, :].T
    a = solve_triangular(angular_lower, angular.T, lower=True).T
    radius = np.linalg.norm(u, axis=1); angular_squared = np.sum(a*a, axis=1)
    result = {}
    for name, values, edges in [('radial', radius, specification['radial_edges']),
                              ('angular', angular_squared, specification['angular_projection_squared_edges'])]:
        edges = np.asarray(edges, float)
        require(np.all(np.diff(edges) > 0) and np.all(values >= edges[0]-1e-10) and np.all(values <= edges[-1]+1e-10), 'Stratum support differs')
        result[name] = np.searchsorted(edges[1:-1], values, side='right').astype(np.int8)
    result['orthant'] = ((u >= 0).astype(np.uint8)*(1 << np.arange(6))).sum(axis=1).astype(np.int8)
    return result


def primary_masks(valid, contact, native, anchors, triangle):
    valid, contact, native, triangle = [np.asarray(v, bool) for v in (valid, contact, native, triangle)]
    anchors = np.asarray(anchors, int)
    require(all(v.shape == valid.shape for v in (contact, native, anchors, triangle)), 'Class vectors differ')
    require(np.array_equal(native[valid], anchors[valid] > 0) and np.isin(anchors, [0, 1, 2]).all(), 'Native anchors disagree')
    require(not np.any(valid & triangle & (anchors != 2)), 'Triangle requires both anchors')
    result = dict(total=valid, registered_native_entry=valid & native,
                  contact_no_native_entry=valid & contact & ~native, unbound_no_native_entry=valid & ~contact & ~native,
                  native_one_anchor=valid & (anchors == 1), native_both_anchors=valid & (anchors == 2),
                  registry_triangle=valid & triangle)
    require(np.array_equal(sum(result[k].astype(int) for k in PRIMARY), valid.astype(int)), 'Primary classes overlap or leave gaps')
    return result


def classify_population(task):
    """One analysis worker; no subprocesses and no calls to any physical sampler."""
    root, out = Path(task['root']), Path(task['out']); job, arm = task['job'], task['arm']
    base = root/arm['id']; directory = Path(job['directory']); destination = out/arm['id']/job['id']
    destination.mkdir(parents=True, exist_ok=False)
    region, config = read(base/'provenance/region.json'), read(base/'provenance/config.json')
    classifier, _ = load_classifier(task['definition'])
    contact = ExclusionContact(read(base/'provenance/shape.json'), config['fixed_poses'], config['depletant_radius'])
    rows_path = directory/'samples.jsonl'
    require(sha(rows_path) == task['output']['samples_sha256'], 'Classified rows changed after validation')
    with rows_path.open() as stream: rows = [json.loads(line) for line in stream]
    arrays = read_weights(rows, job['samples'], region, arm); valid = np.isfinite(arrays['z'])
    summary = read(directory/'summary.json'); audit = task['audit']
    check_estimate(paired_moments(arrays['z'], arrays['h']), audit['estimate'], audit['hard_region'])
    require(int(valid.sum()) == summary['estimates']['region']['nonzero'] == summary['estimates']['hard_region']['nonzero'], 'Contributing count changed')
    native, contacts, triangle = [np.zeros(len(rows), bool) for _ in range(3)]; anchors = np.zeros(len(rows), np.int8)
    negative = 0; anomalies = []; labels = destination/'labels.jsonl.gz'
    with labels.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
        for i, row in enumerate(rows):
            label = classifier.classify(row['pose']) if valid[i] else None
            measured = contact.classify(row['pose']) if valid[i] else None
            if valid[i]:
                native[i] = label['native_any']; contacts[i] = measured['exclusion_contact']
                triangle[i] = label['registry_consistent_triangle']; anchors[i] = label['native_anchor_count']
                negative += int(measured['near_zero_negative_gap'])
                if native[i] and not contacts[i]: anomalies.append(dict(draw=i, pose=row['pose'], classification=label, contact=measured))
            stream.write((json.dumps(dict(draw=i, applicable=bool(valid[i]), classification=label, contact=measured),
                                     separators=(',', ':'), allow_nan=False)+'\n').encode())
    bins = stratify(arrays.pop('latents'), region, task['strata'])
    archive = destination/'records.npz'
    np.savez_compressed(archive, **arrays, native=native, contact=contacts, anchors=anchors, triangle=triangle,
                        **{'bin_'+k: v for k, v in bins.items()})
    require(sha(rows_path) == task['output']['samples_sha256'], 'Raw rows changed during classification')
    cpu = summary['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu >= 0, 'Invalid physical CPU')
    result = dict(id=job['id'], arm=arm['id'], seed=job['seed'], samples=job['samples'],
                  sampler_cpu_seconds=cpu, raw_output=task['output'], records=str(archive.relative_to(out)), records_sha256=sha(archive),
                  labels=str(labels.relative_to(out)), labels_sha256=sha(labels), near_zero_negative_core_gaps=negative,
                  native_entry_unbound_anomalies=anomalies)
    write(destination/'classification.json', result)
    return result


def summarize_arm(out, arm, records, specification, assessment):
    populations = []; counts = []; cpu = 0.
    bin_counts = dict(radial=len(specification['radial_edges'])-1, angular=len(specification['angular_projection_squared_edges'])-1, orthant=64)
    for record in sorted(records, key=lambda p: p['id']):
        path = inside(out, record['records']); require(sha(path) == record['records_sha256'], 'Classification archive changed')
        with np.load(path, allow_pickle=False) as stored: arrays = {k: stored[k] for k in stored.files}
        valid = np.isfinite(arrays['z']); masks = primary_masks(valid, arrays['contact'], arrays['native'], arrays['anchors'], arrays['triangle'])
        dispositions = {}
        branch_masks = {'uniform-shell': arrays['branch'] == 0, 'conditional-ray': arrays['branch'] == 1,
                        'ray-fallback': arrays['fallback'] == 1, 'ray-interval': arrays['fallback'] == 0}
        for branch, selected in branch_masks.items():
            dispositions[branch] = dict(attempted=int(selected.sum()), contributing=int((selected & valid).sum()),
                                       **{name: int((selected & masks[name]).sum()) for name in PRIMARY})
            for name in CLASSES: masks['branch:'+branch+':'+name] = selected & masks[name]
        for family, count in bin_counts.items():
            bins = arrays['bin_'+family]
            require(np.all((bins >= 0) & (bins < count)), 'Invalid saved stratum')
            for index in range(count):
                for name in CLASSES: masks[f'stratum:{family}:{index}:{name}'] = masks[name] & (bins == index)
        populations.append(dict(id=record['id'], seed=record['seed'], z=arrays['z'], h=arrays['h'], pairs=arrays['pairs'], masks=masks))
        counts.append(dict(id=record['id'], seed=record['seed'], branches=dispositions,
                           strata_counts={family: dict(attempted=np.bincount(arrays['bin_'+family], minlength=count).tolist(),
                                                        contributing=np.bincount(arrays['bin_'+family][valid], minlength=count).tolist())
                                          for family, count in bin_counts.items()},
                           selected_width_counts=[int((arrays['component'] == i).sum()) for i in range(3)],
                           per_width_fallback_counts=[int(((arrays['component'] == i) & (arrays['fallback'] == 1)).sum()) for i in range(3)]))
        cpu += record['sampler_cpu_seconds']
    estimates, covariance, ratios = summarize_regions(populations)
    check_estimate(estimates['total']['row_uncertainty'], assessment['estimate'], assessment['hard_region'])
    strata = {}; branch_weights = {}
    for family, count in bin_counts.items():
        strata[family] = {}
        for name in CLASSES:
            entries = []
            for index in range(count):
                item = estimates.pop(f'stratum:{family}:{index}:{name}')
                item.update(bin=index, observed_class_fraction={kind: contribution_fraction(item, estimates[name], kind) for kind in ('Qz', 'Q0')})
                entries.append(item)
            strata[family][name] = entries
            for kind in ('Qz', 'Q0'):
                actual = [item['row_uncertainty']['log_'+kind] for item in entries]
                actual = [v for v in actual if v is not None]; expected = estimates[name]['row_uncertainty']['log_'+kind]
                require(not actual and expected is None or actual and expected is not None and abs(float(logsumexp(actual))-expected) < 2e-10,
                        'Strata fail to partition class mass')
    for branch in ('uniform-shell', 'conditional-ray', 'ray-fallback', 'ray-interval'):
        branch_weights[branch] = {}
        for name in CLASSES:
            item = estimates.pop('branch:'+branch+':'+name)
            branch_weights[branch][name] = dict(estimate=item, observed_class_fraction={kind: contribution_fraction(item, estimates[name], kind) for kind in ('Qz', 'Q0')})
    return dict(allocation=arm, estimates=estimates, primary_covariance=covariance, primary_ratios=ratios,
                sampler_cpu_seconds=cpu, populations=records, strata=strata, branch_dispositions=counts, branch_weight_contributions=branch_weights,
                observed_importance_ESS_per_cpu_second={name: estimates[name]['row_uncertainty'].get('Qz_ESS', 0)/cpu if cpu > 0 else None for name in CLASSES},
                efficiency_scope='Descriptive importance ESS and CPU, not a stationary contact-sampling speedup; requires convergence and coverage.',
                native_entry_unbound_anomaly_count=sum(len(r['native_entry_unbound_anomalies']) for r in records),
                near_zero_negative_core_gap_count=sum(r['near_zero_negative_core_gaps'] for r in records),
                proposal_audit=assessment['importance_sampling'])


def contribution_fraction(child, parent, kind='Qz'):
    a, b = child['row_uncertainty']['log_'+kind], parent['row_uncertainty']['log_'+kind]
    if b is None: return None
    return 0. if a is None else math.exp(a-b)


def compare_mass(left, right, kind, gates, absolute=True):
    a, b = left['population_uncertainty'], right['population_uncertainty']
    if a['log_'+kind] is None or b['log_'+kind] is None:
        return dict(observed=False, passed=False, reason='Unobserved regional mass is neither physical zero nor an upper bound.')
    delta = a['log_'+kind]-b['log_'+kind]; se = math.hypot(a[kind+'_relative_SE'], b[kind+'_relative_SE'])
    abs_pass = not absolute or abs(delta) <= gates['log_agreement_absolute_max']
    se_pass = abs(delta) <= gates['log_agreement_combined_SE_max']*se+1e-12
    return dict(observed=True, log_left_minus_right=delta, combined_population_log_delta_SE=se,
                absolute_limit=gates['log_agreement_absolute_max'] if absolute else None,
                SE_multiplier=gates['log_agreement_combined_SE_max'], absolute_passed=abs_pass, SE_passed=se_pass,
                passed=bool(abs_pass and se_pass))


def quality_gate(estimate, gates):
    row, pop = estimate['row_uncertainty'], estimate['population_uncertainty']
    if row['log_Qz'] is None: return dict(passed=False, observed=False, reason='Unobserved is not a zero mass estimate.')
    checks = dict(population_RSE=pop['Qz_relative_SE'] <= gates['population_relative_SE_max'],
                  importance_ESS=row['Qz_ESS'] >= gates['importance_ESS_min'], largest_draw=row['largest_Qz_fraction'] <= gates['largest_draw_max'])
    return dict(passed=all(checks.values()), observed=True, checks=checks,
                values=dict(population_RSE=pop['Qz_relative_SE'], importance_ESS=row['Qz_ESS'], largest_draw=row['largest_Qz_fraction']))


def free_energy_interval(arm, gates):
    ratio = arm['primary_ratios'][PRIMARY[0]+'/'+PRIMARY[1]]['population']
    n = arm['estimates'][PRIMARY[0]]['population_count']; require(n == 4, 'Predeclared Student-t interval requires four paired populations')
    if ratio['log_ratio'] is None:
        return dict(passed=False, observed=False, reason='Native or no-entry mass unobserved; no free-energy interval.')
    quantile = float(student_t.ppf(.975, n-1)); half = quantile*ratio['log_ratio_SE']; mean = -ratio['log_ratio']
    return dict(observed=True, beta_F_native_minus_noentry=mean, paired_population_log_ratio_SE=ratio['log_ratio_SE'],
                degrees_of_freedom=n-1, Student_t_quantile=quantile, halfwidth_95=half,
                interval_95=[mean-half, mean+half], passed=half <= gates['deltaF_95_halfwidth_max'],
                scope='Approximate paired-population delta interval using Student-t3, not a rigorous tail-coverage bound.')


def evaluate(arms, protocol):
    gates = protocol['convergence']; comparisons = {}; all_comparisons = True; significant_disagreements = []
    for left_name, right_name in COMPARISONS:
        left, right = arms[left_name], arms[right_name]; entry = {}; strata = []
        for name in CLASSES:
            entry[name] = {kind: compare_mass(left['estimates'][name], right['estimates'][name], kind, gates, absolute=kind == 'Qz')
                           for kind in ('Qz', 'Q0')}
            for level in ('row_uncertainty', 'population_uncertainty'):
                a, b = left['estimates'][name][level], right['estimates'][name][level]
                if a['log_Qz'] is None or b['log_Qz'] is None: continue
                ca = a['Qz_relative_SE']**2*left['sampler_cpu_seconds']; cb = b['Qz_relative_SE']**2*right['sampler_cpu_seconds']
                entry[name].setdefault('descriptive_cost_variance', {})[level] = dict(left=ca, right=cb, right_over_left=cb/ca if ca > 0 else None)
            if name in DECISION_CLASSES:
                all_comparisons &= entry[name]['Qz']['passed'] and entry[name]['Q0']['passed']
        all_comparisons &= entry['total']['Q0']['passed']
        for family in ('radial', 'angular', 'orthant'):
            for name in DECISION_CLASSES:
                for a, b in zip(left['strata'][family][name], right['strata'][family][name]):
                    fa, fb = a['observed_class_fraction']['Qz'], b['observed_class_fraction']['Qz']
                    if max(fa or 0., fb or 0.) < gates['significant_stratum_mass_fraction']: continue
                    limits = dict(gates, log_agreement_absolute_max=gates['stratum_log_agreement_absolute_max'])
                    comparison = compare_mass(a, b, 'Qz', limits)
                    record = dict(family=family, region=name, bin=a['bin'], left_fraction=fa, right_fraction=fb, **comparison)
                    strata.append(record)
                    if not comparison['passed']: significant_disagreements.append(dict(left=left_name, right=right_name, **record))
        comparisons[left_name+'__'+right_name] = dict(left=left_name, right=right_name, classes=entry, significant_strata=strata)
    quality = {a: {name: quality_gate(arms[a]['estimates'][name], gates) for name in DECISION_CLASSES} for a in ('large_uniform', 'large_ray')}
    intervals = {a: free_energy_interval(arms[a], gates) for a in ('large_uniform', 'large_ray')}
    checks = dict(large_population_quality=all(v['passed'] for a in quality.values() for v in a.values()),
                  specified_mass_comparisons=bool(all_comparisons), significant_strata_agreement=not significant_disagreements,
                  paired_free_energy_precision=all(v['passed'] for v in intervals.values()),
                  classifier_contact_consistency=all(a['native_entry_unbound_anomaly_count'] == 0 for a in arms.values()))
    passed = all(checks.values())
    return dict(passed=passed, checks=checks, decision_classes=list(DECISION_CLASSES), quality=quality, free_energy_intervals=intervals,
                comparisons=comparisons, significant_stratum_disagreements=significant_disagreements,
                verdict='finite-region numerical gates passed' if passed else 'unresolved finite-region sampling',
                full_wall_coverage_established=False, assembly_stability_established=False,
                scope='Failed gates do not imply the model prevents assembly. Passing these finite diagnostics does not bound undiscovered full-wall contacts.')


def unbound_volume_bound(region):
    """An actual geometric bound: no exclusion overlap implies depletion C=0."""
    chart = region['gaussian_chart']; lower = np.linalg.cholesky(np.asarray(chart['covariances'][0], float))
    r = region['mahalanobis_radius']; log_volume = 3*math.log(math.pi)+6*math.log(r)-math.log(6)
    log_max_j = float(np.log(np.diag(lower)).sum()-3*math.log(chart['angular_length'])-2*math.log(math.pi))
    return dict(log_Qz_upper=log_volume+log_max_j, log_Q0_upper=log_volume+log_max_j,
                log_latent_volume=log_volume, log_maximum_physical_J=log_max_j,
                derivation='Q_unbound = integral H_unbound J dxi <= V6(R) det(L)/(pi^2 ell^3), because no exclusion contact gives C=0 and (1+|Cayley|^2)^-2<=1.',
                scope='Deterministic upper bound inside this finite R4 only; unrelated to whether unbound poses happened to be observed.')


def previous_uniform(path, protocol):
    if path is None: return None
    path = Path(path).resolve(); frozen = read(path/'freeze.json'); frozen = frozen.get('files', frozen)
    require(frozen and 'analysis.json' in frozen, 'Missing previous comparison freeze')
    for name, digest in frozen.items(): require(sha(inside(path, name)) == digest, 'Previous comparison changed')
    data = read(path/'analysis.json')
    require(data['complete'] and data['schema'] == 'mobile-threshold-reference-comparison-v1'
            and data['region_sha256'] == protocol['region_sha256'] and data['shape_sha256'] == protocol['shape_sha256']
            and data['native_definition']['definition_sha256'] == protocol['native_definition_sha256'], 'Previous comparison uses another target')
    old_seeds = {p['seed'] for p in data['reference']['estimates']['total']['populations']}
    require(old_seeds.isdisjoint(j['seed'] for j in protocol['jobs']), 'Previous seeds overlap fresh validation')
    return dict(path=str(path), analysis_sha256=sha(path/'analysis.json'), freeze_sha256=sha(path/'freeze.json'),
                reference=data['reference'], scope='Previously completed, byte-bound comparison; no raw audit replay and no pooling or decision gate.')


def report(result):
    g = result['convergence']; lines = ['# Conditional-ray finite-region reference', '', g['verdict']+'.', '', SCOPE, '',
        '![Population estimates, free-energy contrasts, and tail diagnostics](conditional-ray-reference.png)', '',
        '![Observed radial and angular class-weight coverage](conditional-ray-strata.png)', '',
        '| Arm | Class | log Qz | population RSE | ESS | largest draw | log Q0 | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm, source in result['arms'].items():
        for name in PRIMARY:
            row = source['estimates'][name]['row_uncertainty']; pop = source['estimates'][name]['population_uncertainty']
            values = ['unobserved', '—', '—', '—', '—'] if row['log_Qz'] is None else [f"{row['log_Qz']:.6f}", f"{pop['Qz_relative_SE']:.2%}", f"{row['Qz_ESS']:.1f}", f"{row['largest_Qz_fraction']:.2%}", f"{row['log_Q0']:.6f}"]
            lines.append('| '+arm+' | '+name+' | '+' | '.join(values)+f" | {source['sampler_cpu_seconds']:.1f} |")
    lines += ['', 'Gate results: '+', '.join(k+'='+str(v) for k, v in g['checks'].items())+'.', '',
              'Each arm retains four independent populations. Pilot, larger-N, defensive-floor, and cloud-intensity estimates are never pooled.',
              'The Student-t3 intervals use paired population means. Unobserved regional mass is not zero; the separately derived unbound bound is geometric.', '',
              f"Significant stratum disagreements: {len(g['significant_stratum_disagreements'])} (at least 1% of a compared class in either arm).", '',
              '| Comparison | Class | Radial bins failing | Angular bins failing | Latent orthants failing |',
              '|---|---|---|---|---|']
    for key, comp in g['comparisons'].items():
        for name in DECISION_CLASSES:
            bins = [[str(r['bin']) for r in comp['significant_strata'] if r['region'] == name and r['family'] == family and not r['passed']]
                    for family in ('radial', 'angular', 'orthant')]
            lines.append('| '+key+' | '+name+' | '+' | '.join(', '.join(b) or 'none' for b in bins)+' |')
    lines += ['', 'Branch counts, fallback fractions, class-weight contributions, all64 orthants, paired cloud variances, and each population remain in analysis.json.',
              'ESS/CPU and variance-times-CPU ratios are descriptive; a smaller runtime or more accepted guide draws is not a demonstrated physical sampling speedup.',
              'No full-wall calculation or assembly trajectory was launched by this analyzer.']
    return '\n'.join(lines)+'\n'


def plot(result, out):
    """Derived diagnostics only: never average logarithms or pool different arms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names = list(result['arms']); x = np.arange(len(names)); colors = ['#2563a6', '#cf6b22']
    short = ['native entry', 'contact without entry']; offsets = np.linspace(-.18, .18, 4)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, region in enumerate(DECISION_CLASSES):
        axis = axes[0, index]
        for column, name in enumerate(names):
            estimate = result['arms'][name]['estimates'][region]
            for offset, population in zip(offsets, estimate['populations']):
                if population['log_Qz'] is not None:
                    axis.scatter(column+offset, population['log_Qz'], color=colors[index], alpha=.45, s=17)
            population = estimate['population_uncertainty']
            if population['log_Qz'] is not None:
                axis.errorbar(column, population['log_Qz'], yerr=population['Qz_relative_SE'],
                              fmt='D', color=colors[index], capsize=4, markersize=5)
            else:
                axis.text(column, .04, 'unobserved', rotation=90, fontsize=7, ha='center', transform=axis.get_xaxis_transform())
        axis.set_title(short[index].capitalize()); axis.set_ylabel('log Qz (arm mean ± population delta SE)')
    axis = axes[0, 2]
    for column, name in enumerate(names):
        source = result['arms'][name]
        native = source['estimates'][PRIMARY[0]]; other = source['estimates'][PRIMARY[1]]
        by_id = {p['id']: p for p in other['populations']}
        for offset, population in zip(offsets, native['populations']):
            competing = by_id[population['id']]
            if population['log_Qz'] is not None and competing['log_Qz'] is not None:
                axis.scatter(column+offset, competing['log_Qz']-population['log_Qz'], color='#686080', alpha=.5, s=17)
        interval = free_energy_interval(source, result['protocol_convergence'])
        if interval['observed']:
            axis.errorbar(column, interval['beta_F_native_minus_noentry'], yerr=interval['halfwidth_95'],
                          fmt='D', color='#443a60', capsize=4, markersize=5)
    axis.set_title('Native versus no-entry weight'); axis.set_ylabel('β(F_native − F_no-entry), paired t3 95% interval')
    axis.axhline(0., color='gray', linewidth=.8)
    for index, region in enumerate(DECISION_CLASSES):
        for axis, key, title in [(axes[1, 0], 'Qz_ESS', 'Observed importance ESS'),
                                 (axes[1, 1], 'largest_Qz_fraction', 'Largest single-draw share')]:
            xs, ys = [], []
            for column, name in enumerate(names):
                row = result['arms'][name]['estimates'][region]['row_uncertainty']
                if row['log_Qz'] is not None:
                    xs.append(column+.08*(2*index-1)); ys.append(row[key])
            axis.plot(xs, ys, 'o-', linewidth=.8, color=colors[index], label=short[index]); axis.set_title(title)
        xs, ys = [], []
        for column, name in enumerate(names):
            noise = result['arms'][name]['estimates'][region]['paired_cloud_noise']
            if noise is not None and noise['cloud_fraction'] is not None:
                xs.append(column+.08*(2*index-1)); ys.append(noise['cloud_fraction'])
        axes[1, 2].plot(xs, ys, 'o-', linewidth=.8, color=colors[index], label=short[index])
    axes[1, 0].set_yscale('log'); axes[1, 0].set_ylabel('ESS'); axes[1, 0].axhline(result['protocol_convergence']['importance_ESS_min'], color='gray', ls='--', linewidth=.8)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.)); axes[1, 1].set_ylabel('Fraction of estimated class weight')
    axes[1, 1].axhline(result['protocol_convergence']['largest_draw_max'], color='gray', ls='--', linewidth=.8)
    axes[1, 2].set_title('Paired-cloud variance contribution'); axes[1, 2].set_ylabel('Observed cloud / total variance')
    axes[1, 2].yaxis.set_major_formatter(PercentFormatter(1.)); axes[1, 0].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, names, rotation=28, ha='right'); axis.grid(axis='y', alpha=.18)
    verdict = 'Finite-region numerical gates passed' if result['convergence']['passed'] else 'Convergence unresolved'
    fig.suptitle(verdict+' — fixed R4, six separate estimates', fontsize=15)
    fig.supxlabel('Small dots: independent populations. Diamonds: logarithms of arithmetic mean weights (ΔF uses their ratio); no means of logs.', fontsize=9)
    fig.savefig(out/'conditional-ray-reference.png', dpi=170); fig.savefig(out/'conditional-ray-reference.svg'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    labels = {'radial': ['[0, 2)', '[2, 3)', '[3, 4]'], 'angular': ['[0, 4)', '[4, 9)', '[9, 16]']}
    for row_index, region in enumerate(DECISION_CLASSES):
        for column, family in enumerate(('radial', 'angular')):
            axis = axes[row_index, column]
            for offset, name in enumerate(('large_uniform', 'large_ray')):
                entries = result['arms'][name]['strata'][family][region]
                fractions = [np.nan if e['observed_class_fraction']['Qz'] is None else e['observed_class_fraction']['Qz'] for e in entries]
                axis.bar(np.arange(len(entries))+.32*(offset-.5), fractions, width=.3,
                         label=name, color=('#707c89', '#259c91')[offset])
            axis.set_xticks(range(3), labels[family]); axis.yaxis.set_major_formatter(PercentFormatter(1.))
            axis.set_ylabel(short[row_index]+': class weight fraction')
            axis.set_xlabel('Original latent radius' if family == 'radial' else 'Angular marginal squared radius a²')
            axis.grid(axis='y', alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(verdict+' — radial and angular coverage in larger populations', fontsize=12)
    fig.supxlabel('Observed weight fractions only. Strata are unchanged; all64 latent orthants and population uncertainties remain in analysis.json.', fontsize=8)
    fig.savefig(out/'conditional-ray-strata.png', dpi=170); fig.savefig(out/'conditional-ray-strata.svg'); plt.close(fig)


def analyze(campaign, out, workers=4, old_uniform=None):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(type(workers) is int and 1 <= workers <= 4, 'At most four analysis workers')
    require(not out.exists(), 'Fresh downstream output required; no retry/overwrite')
    protocol, status, assessments, definition = validate_terminal(root)
    _, classifier_binding = load_classifier(definition)
    old = previous_uniform(old_uniform, protocol)
    out.mkdir(parents=True); start = time.time()
    state = dict(complete=False, phase='classifying_saved_rows', completed_populations=[], started=start)
    write(out/'status.json', state)
    try:
        tasks = []
        for job in protocol['jobs']:
            arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
            term = next(t for t in status['jobs'] if (t['arm'], t['id']) == (job['arm'], job['id']))
            audit = next(a for a in assessments[job['arm']]['populations'] if a['id'] == job['id'])
            tasks.append(dict(root=str(root), out=str(out), job=job, arm=arm, output=term['output'], audit=audit,
                              definition=str(definition), strata=protocol['strata']))
        records = []
        if workers == 1:
            for task in tasks:
                record = classify_population(task); records.append(record)
                state['completed_populations'].append(record['arm']+'/'+record['id']); write(out/'status.json', state)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(classify_population, task) for task in tasks]
                for future in as_completed(futures):
                    record = future.result(); records.append(record)
                    state['completed_populations'].append(record['arm']+'/'+record['id']); write(out/'status.json', state)
                    print('Classified saved '+record['arm']+'/'+record['id'], flush=True)
        state['phase'] = 'statistics'; write(out/'status.json', state)
        arms = {}
        for arm in protocol['arms']:
            arms[arm['id']] = summarize_arm(out, arm, [r for r in records if r['arm'] == arm['id']], protocol['strata'], assessments[arm['id']])
        result = dict(schema='conditional-ray-reference-comparison-v1', complete=True, campaign=str(root),
                      protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'), freeze_sha256=sha(root/'freeze.json'),
                      region_sha256=protocol['region_sha256'], shape_sha256=protocol['shape_sha256'], native_definition=classifier_binding,
                      arms=arms, convergence=evaluate(arms, protocol), protocol_convergence=protocol['convergence'], strata_definition=protocol['strata'],
                      unbound_finite_region_bound=unbound_volume_bound(read(root/protocol['arms'][0]['id']/'provenance/region.json')),
                      previous_uniform=old, primary_partition=list(PRIMARY), total_unconditional_draws=protocol['total_unconditional_draws'], scope=SCOPE,
                      analysis_wall_seconds=time.time()-start, analysis_workers=workers)
        sources = out/'provenance'; sources.mkdir()
        for name, path in local_sources(__file__).items(): shutil.copy2(path, sources/name)
        shutil.copy2(root/'protocol.json', sources/'campaign-protocol.json')
        result['analyzer_source_sha256'] = {p.name: sha(p) for p in sources.iterdir()}
        write(out/'analysis.json', result); (out/'report.md').write_text(report(result)); plot(result, out)
        state.update(complete=True, phase='complete', finished=time.time(), analysis_sha256=sha(out/'analysis.json')); write(out/'status.json', state)
        write(out/'freeze.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return result
    except BaseException as error:
        state.update(complete=False, phase=state['phase']+'_failed', error=repr(error), finished=time.time()); write(out/'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--previous-uniform', type=Path)
    args = parser.parse_args(); result = analyze(args.campaign, args.out, args.workers, args.previous_uniform)
    print(json.dumps(dict(complete=True, verdict=result['convergence']['verdict'], analysis_sha256=sha(args.out/'analysis.json'))))
