#!/usr/bin/env python3
"""Separate fresh Gaussian-guide screening; never pool training rows or run physics.

All Gaussian draws outside the frozen region have zero physical weight and remain
in the unconditional denominator. This pilot alone cannot pass the larger-N,
cloud-intensity, full-vessel, or finite-assembly decision gates.
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
from scipy.special import logsumexp
from analyze_mobile_native_pocket import require, read, write, sha, inside, load_classifier, local_sources
from analyze_mobile_competing_reference import paired_moments, check_estimate
from analyze_mobile_threshold_reference import ExclusionContact, validate_classifier_target
from analyze_mobile_wall_contacts import PRIMARY, summarize_regions
from analyze_conditional_ray_campaign import (stratify, primary_masks, contribution_fraction,
    compare_mass, quality_gate, free_energy_interval, unbound_volume_bound)

CLASSES = ('total',)+PRIMARY
DECISION_CLASSES = PRIMARY[:2]
SCOPE = ('Independent screening of two frozen full-six-dimensional proposal widths on the unchanged R4. '
    'All attempted draws, including exterior Gaussian draws and hard-invalid poses, retain zero weights in '
    'the original denominators. The native, contact-without-entry, and unbound classes are exhaustive inside '
    'the hard-valid target. Old training estimates remain separate. Passing this pilot is not a population-size '
    'or intensity validation, a coverage certificate, a physical mixing measurement, or an assembly result.')


def read_weights(rows, samples, region, arm):
    require(len(rows) == samples and [r['draw'] for r in rows] == list(range(samples)), 'Missing/repeated unconditional draws')
    z = np.full(samples, -np.inf); h = z.copy(); pairs = np.full((samples, 2), -np.inf)
    branch = np.zeros(samples, np.int8); component = np.full(samples, -1, np.int16)
    latents = np.empty((samples, 6)); support = np.zeros(samples, bool)
    for i, row in enumerate(rows):
        require(all(type(row[k]) is bool for k in ('hard_valid', 'capture_valid', 'region_valid', 'shell_valid')), 'Missing support flags')
        require(math.isfinite(row['latent_radius']) and row['latent_radius'] >= 0, 'Invalid saved radius')
        latents[i] = row['latent']; require(np.isfinite(latents[i]).all(), 'Nonfinite latent row')
        require(abs(np.linalg.norm(latents[i])-row['latent_radius']) < 2e-8, 'Latent radius differs')
        inner = region.get('minimum_mahalanobis_radius', 0.)
        shell = (row['latent_radius'] > inner if inner else row['latent_radius'] >= 0.) and row['latent_radius'] <= region['mahalanobis_radius']
        require(row['shell_valid'] == shell, 'Saved shell predicate differs')
        require(math.isfinite(row['q']) and row['q'] >= 0 and all(math.isfinite(row[k]) for k in ('log_physical_jacobian', 'log_proposal_density')), 'Nonfinite metric/density')
        support[i] = shell
        if row['proposal_branch'] == 'uniform-shell':
            require(shell and row['proposal_component'] is None, 'Invalid uniform branch')
        else:
            require(row['proposal_branch'] == 'gaussian' and arm['alpha'] < 1 and type(row['proposal_component']) is int
                    and 0 <= row['proposal_component'] < arm['component_count'], 'Invalid Gaussian branch')
            branch[i] = 1; component[i] = row['proposal_component']
        require(row.get('selected_ray_fallback') is None, 'Unexpected ray metadata in Gaussian guide')
        valid = row['hard_valid'] and row['capture_valid'] and row['region_valid'] and shell
        if not valid:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'], 'Every invalid or exterior draw must retain zero weight')
            continue
        require(len(row['clouds']) == 2 and math.isfinite(row['log_hard_weight']) and math.isfinite(row['log_importance_weight']), 'Missing two-cloud weight')
        h[i] = row['log_hard_weight']; z[i] = row['log_importance_weight']
        require(abs(h[i]-row['log_physical_jacobian']+row['log_proposal_density']) < 2e-10, 'Saved hard weight is not full J/q')
        pairs[i] = [h[i]+cloud['log_weight'] for cloud in row['clouds']]
        require(np.isfinite(pairs[i]).all() and abs(float(logsumexp(pairs[i])-math.log(2))-z[i]) < 2e-10, 'Two-cloud mean changed')
    return dict(z=z, h=h, pairs=pairs, branch=branch, component=component, latents=latents, support=support)


def stratify_with_exterior(latents, region, specification, support):
    """The old fixed strata apply on R4; all exterior attempts have explicit bin -1."""
    support = np.asarray(support, bool)
    result = {name: np.full(len(latents), -1, np.int8) for name in ('radial', 'angular', 'orthant')}
    if support.any():
        for name, values in stratify(np.asarray(latents)[support], region, specification).items():
            result[name][support] = values
    return result


def classify_population(task):
    root, out = Path(task['root']), Path(task['out']); job, arm = task['job'], task['arm']
    base = root/arm['id']; directory = Path(job['directory']); destination = out/arm['id']/job['id']
    destination.mkdir(parents=True, exist_ok=False)
    region, config = read(base/'provenance/region.json'), read(base/'provenance/config.json')
    classifier, _ = load_classifier(task['definition'])
    contact = ExclusionContact(read(base/'provenance/shape.json'), config['fixed_poses'], config['depletant_radius'])
    rows_path = directory/'samples.jsonl'
    require(sha(rows_path) == task['output']['samples_sha256'], 'Raw rows changed after completed audit')
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
    bins = stratify_with_exterior(arrays.pop('latents'), region, task['strata'], arrays['support'])
    archive = destination/'records.npz'
    np.savez_compressed(archive, **arrays, native=native, contact=contacts, anchors=anchors, triangle=triangle,
                        **{'bin_'+k: v for k, v in bins.items()})
    require(sha(rows_path) == task['output']['samples_sha256'], 'Rows changed during classification')
    cpu = summary['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu >= 0, 'Invalid physical CPU')
    result = dict(id=job['id'], arm=arm['id'], seed=job['seed'], samples=job['samples'], sampler_cpu_seconds=cpu,
        raw_output=task['output'], records=str(archive.relative_to(out)), records_sha256=sha(archive),
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
        dispositions = {}; branch_masks = {'uniform-shell': arrays['branch'] == 0, 'gaussian': arrays['branch'] == 1}
        require(not np.any(valid & ~arrays['support']), 'Exterior physical contribution')
        for branch, selected in branch_masks.items():
            dispositions[branch] = dict(attempted=int(selected.sum()), exterior=int((selected & ~arrays['support']).sum()),
                contributing=int((selected & valid).sum()), **{name: int((selected & masks[name]).sum()) for name in PRIMARY})
            for name in CLASSES: masks['branch:'+branch+':'+name] = selected & masks[name]
        for family, count in bin_counts.items():
            bins = arrays['bin_'+family]
            require(np.all((bins >= -1) & (bins < count)) and np.array_equal(bins == -1, ~arrays['support']), 'Invalid saved stratum')
            for index in range(count):
                for name in CLASSES: masks[f'stratum:{family}:{index}:{name}'] = masks[name] & (bins == index)
        populations.append(dict(id=record['id'], seed=record['seed'], z=arrays['z'], h=arrays['h'], pairs=arrays['pairs'], masks=masks))
        counts.append(dict(id=record['id'], seed=record['seed'], branches=dispositions,
            exterior_attempts=int((~arrays['support']).sum()),
            strata_counts={family: dict(attempted=np.bincount(arrays['bin_'+family][arrays['support']], minlength=count).tolist(),
                contributing=np.bincount(arrays['bin_'+family][valid], minlength=count).tolist()) for family, count in bin_counts.items()},
            selected_component_counts=[int((arrays['component'] == i).sum()) for i in range(arm['component_count'])]))
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
                actual = [e['row_uncertainty']['log_'+kind] for e in entries]; actual = [v for v in actual if v is not None]
                expected = estimates[name]['row_uncertainty']['log_'+kind]
                require(not actual and expected is None or actual and expected is not None and abs(float(logsumexp(actual))-expected) < 2e-10, 'Strata do not partition class mass')
    for branch in ('uniform-shell', 'gaussian'):
        branch_weights[branch] = {}
        for name in CLASSES:
            item = estimates.pop('branch:'+branch+':'+name)
            branch_weights[branch][name] = dict(estimate=item, observed_class_fraction={kind: contribution_fraction(item, estimates[name], kind) for kind in ('Qz', 'Q0')})
    return dict(allocation=arm, estimates=estimates, primary_covariance=covariance, primary_ratios=ratios,
        sampler_cpu_seconds=cpu, populations=records, strata=strata, branch_dispositions=counts, branch_weight_contributions=branch_weights,
        observed_importance_ESS_per_cpu_second={name: estimates[name]['row_uncertainty'].get('Qz_ESS', 0)/cpu if cpu > 0 else None for name in CLASSES},
        efficiency_scope='Descriptive importance ESS/CPU; not stationary contact-sampling ESS or a converged speedup.',
        native_entry_unbound_anomaly_count=sum(len(r['native_entry_unbound_anomalies']) for r in records),
        near_zero_negative_core_gap_count=sum(r['near_zero_negative_core_gaps'] for r in records), proposal_audit=assessment['importance_sampling'])



def compare_free_energy_intervals(left, right, gates):
    """Check the contrast itself; opposite small mass shifts can add up."""
    if not left['observed'] or not right['observed']:
        return dict(observed=False, passed=False, reason='Both regional contrasts must be observed.')
    delta = left['beta_F_native_minus_noentry'] - right['beta_F_native_minus_noentry']
    se = math.hypot(left['paired_population_log_ratio_SE'], right['paired_population_log_ratio_SE'])
    absolute = abs(delta) <= gates['log_agreement_absolute_max']
    statistical = abs(delta) <= gates['log_agreement_combined_SE_max']*se + 1e-12
    return dict(observed=True, beta_deltaF_left_minus_right=delta, combined_population_SE=se,
        absolute_passed=absolute, SE_passed=statistical, passed=bool(absolute and statistical),
        absolute_limit=gates['log_agreement_absolute_max'], SE_multiplier=gates['log_agreement_combined_SE_max'])


def evaluate(arms, gates):
    require(set(arms) == {'bank', 'wide'}, 'Two separately frozen arms required')
    quality = {a: {name: quality_gate(arms[a]['estimates'][name], gates) for name in DECISION_CLASSES} for a in arms}
    intervals = {a: free_energy_interval(arms[a], gates) for a in arms}
    comparisons = {name: {kind: compare_mass(arms['bank']['estimates'][name], arms['wide']['estimates'][name], kind, gates, absolute=kind == 'Qz')
                         for kind in ('Qz', 'Q0')} for name in CLASSES}
    significant = []
    for family in ('radial', 'angular', 'orthant'):
        for name in DECISION_CLASSES:
            for a, b in zip(arms['bank']['strata'][family][name], arms['wide']['strata'][family][name]):
                fa, fb = a['observed_class_fraction']['Qz'], b['observed_class_fraction']['Qz']
                if max(fa or 0., fb or 0.) < gates['significant_stratum_mass_fraction']: continue
                limit = dict(gates, log_agreement_absolute_max=gates['stratum_log_agreement_absolute_max'])
                significant.append(dict(family=family, region=name, bin=a['bin'], bank_fraction=fa, wide_fraction=fb, **compare_mass(a, b, 'Qz', limit)))
    contrast_agreement = compare_free_energy_intervals(intervals['bank'], intervals['wide'], gates)
    checks = dict(width_free_energy_contrast_agreement=contrast_agreement['passed'], observed_regional_quality=all(v['passed'] for arm in quality.values() for v in arm.values()),
        width_agreement=all(comparisons[name][kind]['passed'] for name in DECISION_CLASSES for kind in ('Qz', 'Q0')) and comparisons['total']['Q0']['passed'],
        paired_free_energy_precision=all(v['passed'] for v in intervals.values()),
        significant_strata_agreement=all(v['passed'] for v in significant),
        classifier_contact_consistency=all(arm['native_entry_unbound_anomaly_count'] == 0 for arm in arms.values()))
    passed = all(checks.values())
    return dict(screening_passed=passed, passed=False, checks=checks, quality=quality, free_energy_intervals=intervals,
        comparisons=comparisons, free_energy_contrast_comparison=contrast_agreement, significant_strata=significant, significant_stratum_disagreements=[v for v in significant if not v['passed']],
        verdict='screening diagnostics passed; fresh size/intensity and coverage checks remain' if passed else 'unresolved finite-region sampling',
        missing_confirmatory_checks=['fresh larger populations', 'cloud-intensity sensitivity', 'defensive probability sensitivity', 'remaining full-vessel contribution'],
        full_wall_coverage_established=False, assembly_stability_established=False)


def previous_training(path, protocol):
    """Bind the old completed comparison without replaying its physical audits."""
    path = Path(path).resolve(); frozen = read(path/'freeze.json'); frozen = frozen.get('files', frozen)
    require('analysis.json' in frozen and all(sha(inside(path, name)) == digest for name, digest in frozen.items()), 'Training comparison changed')
    data = read(path/'analysis.json')
    require(data['complete'] and data['schema'] == 'conditional-ray-reference-comparison-v1'
        and data['region_sha256'] == protocol['region_sha256'] and data['shape_sha256'] == protocol['shape_sha256']
        and data['native_definition']['definition_sha256'] == protocol['native_definition_sha256'], 'Training target differs')
    old_seeds = {p['seed'] for a in data['arms'].values() for p in a['estimates']['total']['populations']}
    require(old_seeds.isdisjoint(j['seed'] for j in protocol['jobs']), 'Training and validation seeds overlap')
    return dict(path=str(path), analysis_sha256=sha(path/'analysis.json'), freeze_sha256=sha(path/'freeze.json'),
        arms={k: dict(estimates=v['estimates'], sampler_cpu_seconds=v['sampler_cpu_seconds']) for k, v in data['arms'].items()},
        scope='Completed training-source estimates are discovery evidence only, never pooled or treated as an independent validation of their fitted guide.')


def validate_terminal(root):
    from run_contact_bank_pilot import validate_completed
    protocol, status = validate_completed(root)
    assessments = {}
    for arm in protocol['arms']:
        name = arm['id']; assessment = read(root/name/'assessment/analysis.json')
        jobs = [j for j in protocol['jobs'] if j['arm'] == name]; n = sum(j['samples'] for j in jobs)
        require(assessment['estimate']['draws'] == assessment['hard_region']['draws'] == assessment['independently_reconstructed_poses'] == n, 'Audit discarded attempts')
        require(assessment['region_sha256'] == protocol['region_sha256'] and len(assessment['populations']) == len(jobs)
                and {p['id'] for p in assessment['populations']} == {j['id'] for j in jobs}, 'Audit populations/target differ')
        for job in jobs:
            population = next(p for p in assessment['populations'] if p['id'] == job['id'])
            term = next(t for t in status['jobs'] if (t['arm'], t['id']) == (name, job['id']))
            require(population['seed'] == job['seed'] and population['samples_sha256'] == term['output']['samples_sha256']
                and population['estimate']['draws'] == population['hard_region']['draws'] == job['samples'], 'Audit population binding changed')
        guide = assessment['importance_sampling']
        require(guide['uniform_shell_probability'] == arm['alpha'], 'Audited uniform floor differs')
        assessments[name] = assessment
    definition = inside(root, protocol['native_definition'])
    require(sha(definition) == protocol['native_definition_sha256'], 'Native definition changed')
    for arm in protocol['arms']:
        validate_classifier_target(read(root/arm['id']/'provenance/config.json'), protocol['shape_sha256'], read(definition), definition)
    return protocol, status, assessments, definition


def report(result):
    lines = ['# Fresh six-dimensional contact-guide pilot', '', result['convergence']['verdict']+'.', '', SCOPE, '',
        '![Fresh guide screening](contact-bank-pilot.png)', '',
        '| Arm | Class | log Qz | population RSE | ESS | largest draw | log Q0 | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm, source in result['arms'].items():
        for name in PRIMARY:
            row = source['estimates'][name]['row_uncertainty']; pop = source['estimates'][name]['population_uncertainty']
            values = ['unobserved', '—', '—', '—', '—'] if row['log_Qz'] is None else [f"{row['log_Qz']:.6f}", f"{pop['Qz_relative_SE']:.2%}", f"{row['Qz_ESS']:.1f}", f"{row['largest_Qz_fraction']:.2%}", f"{row['log_Q0']:.6f}"]
            lines.append('| '+arm+' | '+name+' | '+' | '.join(values)+f" | {source['sampler_cpu_seconds']:.1f} |")
    lines += ['', 'Screening checks: '+', '.join(k+'='+str(v) for k, v in result['convergence']['checks'].items())+'.', '',
        'The widths are two separate estimators. Exterior and hard-invalid draws remain in every denominator; class fractions are never used as normalizers.',
        'All radial/angular strata and 64 orthants are retained in analysis.json. Paired-population Student-t3 intervals are approximate, not unseen-mode bounds.',
        'No production full-wall or assembly job was launched. The independent Poisson-cloud pair diagnoses estimator noise, not independent physical contact samples.']
    return '\n'.join(lines)+'\n'


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names = ['bank', 'wide']; colors = ['#286eaa', '#bd6a2a']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), constrained_layout=True)
    for k, name in enumerate(DECISION_CLASSES):
        for x, arm in enumerate(names):
            e = result['arms'][arm]['estimates'][name]; row, pop = e['row_uncertainty'], e['population_uncertainty']
            for offset, p in zip(np.linspace(-.14, .14, 4), e['populations']):
                if p['log_Qz'] is not None: axes[0, k].scatter(x+offset, p['log_Qz'], s=15, color=colors[k], alpha=.5)
            if pop['log_Qz'] is not None:
                axes[0, k].errorbar(x, pop['log_Qz'], yerr=pop['Qz_relative_SE'], fmt='D', color=colors[k], capsize=4)
                axes[1, 0].scatter(x+.07*(2*k-1), row['Qz_ESS'], color=colors[k], label=['native', 'no-entry'][k] if x == 0 else None)
                axes[1, 1].scatter(x+.07*(2*k-1), row['largest_Qz_fraction'], color=colors[k])
            noise = e['paired_cloud_noise']
            if noise is not None and noise['cloud_fraction'] is not None: axes[1, 2].scatter(x+.07*(2*k-1), noise['cloud_fraction'], color=colors[k])
        axes[0, k].set_title(['Native entry', 'Contact without entry'][k]); axes[0, k].set_ylabel('log Qz ± population delta SE')
    for x, arm in enumerate(names):
        i = result['convergence']['free_energy_intervals'][arm]
        if i['observed']: axes[0, 2].errorbar(x, i['beta_F_native_minus_noentry'], yerr=i['halfwidth_95'], fmt='D', color='#675080', capsize=4)
    axes[0, 2].set_title('Native versus contact without entry'); axes[0, 2].set_ylabel('βΔF, approximate paired t3 95% interval')
    axes[1, 0].set_title('Observed importance ESS'); axes[1, 0].set_yscale('log'); axes[1, 0].axhline(200, ls='--', color='gray', lw=.7)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title('Largest contribution'); axes[1, 1].axhline(.02, ls='--', color='gray', lw=.7)
    axes[1, 2].set_title('Paired-cloud share of variance')
    for axis in axes[1, 1:]: axis.yaxis.set_major_formatter(PercentFormatter(1.))
    for axis in axes.flat: axis.set_xticks(range(2), names); axis.grid(axis='y', alpha=.2)
    fig.suptitle('Fresh fixed-region screening — assembly remains unresolved', fontsize=15)
    fig.supxlabel('Dots in top panels: four independent population estimates. No training rows pooled. Importance ESS is not MCMC mixing.', fontsize=9)
    fig.savefig(out/'contact-bank-pilot.png', dpi=170); fig.savefig(out/'contact-bank-pilot.svg'); plt.close(fig)


def analyze(campaign, out, workers=4, training=None):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(type(workers) is int and 1 <= workers <= 4, 'At most four analysis workers')
    require(not out.exists(), 'Fresh analysis output required')
    protocol, status, assessments, definition = validate_terminal(root)
    _, classifier_binding = load_classifier(definition)
    old = previous_training(training, protocol) if training is not None else None
    out.mkdir(parents=True); start = time.time()
    state = dict(complete=False, phase='classifying_saved_rows', completed_populations=[], started=start); write(out/'status.json', state)
    try:
        tasks = []
        for job in protocol['jobs']:
            arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
            term = next(t for t in status['jobs'] if (t['arm'], t['id']) == (job['arm'], job['id']))
            audit = next(a for a in assessments[job['arm']]['populations'] if a['id'] == job['id'])
            tasks.append(dict(root=str(root), out=str(out), job=job, arm=arm, output=term['output'], audit=audit, definition=str(definition), strata=protocol['strata']))
        records = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(classify_population, task) for task in tasks]
            for future in as_completed(futures):
                record = future.result(); records.append(record); state['completed_populations'].append(record['arm']+'/'+record['id'])
                write(out/'status.json', state); print('Classified saved '+record['arm']+'/'+record['id'], flush=True)
        state['phase'] = 'statistics'; write(out/'status.json', state)
        arms = {a['id']: summarize_arm(out, a, [r for r in records if r['arm'] == a['id']], protocol['strata'], assessments[a['id']]) for a in protocol['arms']}
        result = dict(schema='contact-bank-pilot-comparison-v2', complete=True, campaign=str(root),
            protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'), freeze_sha256=sha(root/'freeze.json'),
            region_sha256=protocol['region_sha256'], shape_sha256=protocol['shape_sha256'], native_definition=classifier_binding,
            arms=arms, convergence=evaluate(arms, protocol['convergence']), protocol_convergence=protocol['convergence'], strata_definition=protocol['strata'],
            proposal_training=dict(preparation_plan_sha256=protocol['preparation_plan_sha256'],
                preparation_freeze_sha256=protocol['preparation_freeze_sha256'], component_groups=protocol['component_groups'],
                scope='The earlier conditional-ray rows and independent R5 pocket rows were both used for design. Neither is fresh validation of the augmented guide; all new pilot populations use disjoint seeds and unconditional denominators.'),
            unbound_finite_region_bound=unbound_volume_bound(read(root/'bank/provenance/region.json')),
            previous_training=old, primary_partition=list(PRIMARY), total_unconditional_draws=protocol['total_unconditional_draws'],
            scope=SCOPE, analysis_wall_seconds=time.time()-start, analysis_workers=workers)
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
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--training-comparison', type=Path)
    args = parser.parse_args(); result = analyze(args.campaign, args.out, args.workers, args.training_comparison)
    print(json.dumps(dict(complete=True, verdict=result['convergence']['verdict'], analysis_sha256=sha(args.out/'analysis.json'))))
