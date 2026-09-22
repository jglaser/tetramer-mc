#!/usr/bin/env python3
"""Analyze fresh five-arm contact confirmation with one native-classifier pass.

Every original attempt remains in its denominator. The old-R5 intersection and
remaining full native R4 are classified during the same first pass and receive
unchanged physical weights. No second chart Jacobian or proposal reweighting.
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
from analyze_contact_bank_pilot import (read_weights, stratify_with_exterior,
    compare_free_energy_intervals)
from analyze_contact_bank_reference_partition import (r5_contains, partition_masks,
    validate_regions, assert_sum, PARTS, NATIVE)
from diagnose_conditional_ray_extremes import chart_coordinates
from analyze_mobile_native_pocket import require, read, write, sha, inside, load_classifier, local_sources
from analyze_mobile_competing_reference import paired_moments, check_estimate
from analyze_mobile_threshold_reference import ExclusionContact, validate_classifier_target
from analyze_mobile_wall_contacts import PRIMARY, summarize_regions
from analyze_conditional_ray_campaign import (primary_masks, contribution_fraction,
    compare_mass, quality_gate, free_energy_interval, unbound_volume_bound)

ARM_NAMES = ('bank', 'wide', 'small', 'alpha02', 'intensity256')
MAIN_ARMS = ('bank', 'wide', 'alpha02', 'intensity256')
DECISION_CLASSES = PRIMARY[:2] + PARTS
CLASSES = ('total',) + PRIMARY + PARTS
SUPPLEMENTAL_SCHEMA = 'contact-confirmation-native-partition-v1'
SCOPE = ('Fresh independent five-arm importance confirmation on unchanged full R4, separate from all '
    'training estimates. Every exterior or hard-invalid zero retains its original attempted-draw '
    'denominator. Each hard-valid pose receives one full native classifier call and one contact '
    'classification. The exact old-R5 intersection and complementary native mask are computed '
    'during that same pass from the archived original chart and metric. No second Jacobian, '
    'native filter on the target, dropped modes, retraining or adaptive stopping. Passing these '
    'finite-region importance checks is not an unseen-mass guarantee, independent SMC coverage '
    'result, stationary mixing measurement, full-vessel result or assembly conclusion.')


def supplemental_masks(valid, native, old_radius, original_q, capture):
    """Closed old-chart radius, strict original metric, old capture, full native."""
    return partition_masks(np.asarray(valid, bool), np.asarray(native, bool),
        r5_contains(np.asarray(old_radius, float), np.asarray(original_q, float), np.asarray(capture, bool)))


def classify_rows_once(rows, arrays, classifier, contact, reference, label_stream):
    """Adapt the v2 first classification pass; never reread or reclassify a pose."""
    n = len(rows); valid = np.isfinite(arrays['z'])
    native, contacts, triangle = [np.zeros(n, bool) for _ in range(3)]
    anchors = np.zeros(n, np.int8)
    old_radius = np.linalg.norm(chart_coordinates([r['pose'] for r in rows], reference), axis=1)
    original_q = np.asarray([r['q'] for r in rows], float)
    capture = np.asarray([r['capture_valid'] for r in rows], bool)
    old_support = r5_contains(old_radius, original_q, capture)
    negative = 0; anomalies = []; calls = 0
    for i, row in enumerate(rows):
        label = classifier.classify(row['pose']) if valid[i] else None
        measured = contact.classify(row['pose']) if valid[i] else None
        if valid[i]:
            calls += 1
            native[i] = label['native_any']; contacts[i] = measured['exclusion_contact']
            triangle[i] = label['registry_consistent_triangle']; anchors[i] = label['native_anchor_count']
            negative += int(measured['near_zero_negative_gap'])
            if native[i] and not contacts[i]:
                anomalies.append(dict(draw=i, pose=row['pose'], classification=label, contact=measured))
        supplemental = None if not valid[i] else {
            PARTS[0]: bool(native[i] and old_support[i]), PARTS[1]: bool(native[i] and not old_support[i])}
        item = dict(draw=i, applicable=bool(valid[i]), classification=label, contact=measured,
            native_partition=supplemental)
        label_stream.write((json.dumps(item, separators=(',', ':'), allow_nan=False) + '\n').encode())
    masks = supplemental_masks(valid, native, old_radius, original_q, capture)
    class_id = np.full(n, -2, np.int8); class_id[valid] = -1
    class_id[masks[PARTS[0]]] = 0; class_id[masks[PARTS[1]]] = 1
    class_id[valid & contacts & ~native] = 2
    return dict(native=native, contact=contacts, triangle=triangle, anchors=anchors,
        old_r5_radius=old_radius, original_q=original_q, capture=capture,
        old_R5_intersection_native=masks[PARTS[0]], remaining_R4_native=masks[PARTS[1]],
        class_id=class_id), dict(near_zero_negative_core_gaps=negative,
        native_entry_unbound_anomalies=anomalies, full_native_classifier_calls=calls,
        contact_classifier_calls=calls, partition_counts={k: int(v.sum()) for k, v in masks.items()})


def classify_population(task):
    """One bounded worker reuses v2 weight validation and performs one classifier pass."""
    root, out = Path(task['root']), Path(task['out']); job, arm = task['job'], task['arm']
    base = root / arm['id']; directory = Path(job['directory']); destination = out / arm['id'] / job['id']
    destination.mkdir(parents=True, exist_ok=False)
    region, config = read(base / 'provenance/region.json'), read(base / 'provenance/config.json')
    reference = read(task['reference_region']); validate_regions(region, reference)
    require(sha(task['reference_region']) == task['reference_region_sha256'], 'Old R5 chart changed before classification')
    classifier, binding = load_classifier(task['definition'])
    require(binding == task['classifier_binding'], 'Full native classifier binding changed')
    contact = ExclusionContact(read(base / 'provenance/shape.json'), config['fixed_poses'], config['depletant_radius'])
    rows_path = directory / 'samples.jsonl'
    require(sha(rows_path) == task['output']['samples_sha256'], 'Rows changed after completed physical audit')
    with rows_path.open() as stream:
        rows = [json.loads(line) for line in stream]
    arrays = read_weights(rows, job['samples'], region, arm)
    valid = np.isfinite(arrays['z']); summary = read(directory / 'summary.json')
    check_estimate(paired_moments(arrays['z'], arrays['h']), task['audit']['estimate'], task['audit']['hard_region'])
    require(int(valid.sum()) == summary['estimates']['region']['nonzero'] == summary['estimates']['hard_region']['nonzero'], 'Contributing count changed')
    labels = destination / 'labels.jsonl.gz'
    with labels.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
        classified, diagnostics = classify_rows_once(rows, arrays, classifier, contact, reference, stream)
    latents = arrays.pop('latents')
    bins = stratify_with_exterior(latents, region, task['strata'], arrays['support'])
    archive = destination / 'records.npz'
    np.savez_compressed(archive, **arrays, **classified, u=latents,
        log_q=np.asarray([r['log_proposal_density'] for r in rows]),
        log_physical_jacobian=np.asarray([r['log_physical_jacobian'] for r in rows]),
        draw=np.arange(len(rows)), source_n=np.full(len(rows), job['samples'], np.int32),
        **{'bin_' + k: v for k, v in bins.items()})
    require(sha(rows_path) == task['output']['samples_sha256'], 'Rows changed during classification')
    require(sha(task['reference_region']) == task['reference_region_sha256'], 'R5 chart changed during classification')
    cpu = summary['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu >= 0., 'Invalid physical CPU time')
    result = dict(id=job['id'], arm=arm['id'], seed=job['seed'], samples=job['samples'],
        sampler_cpu_seconds=cpu, raw_output=task['output'], records=str(archive.relative_to(out)),
        records_sha256=sha(archive), labels=str(labels.relative_to(out)), labels_sha256=sha(labels),
        full_native_definition_sha256=task['classifier_binding']['definition_sha256'],
        reference_region_sha256=task['reference_region_sha256'],
        supplemental_definition_sha256=task['supplemental_definition_sha256'], **diagnostics)
    write(destination / 'classification.json', result)
    return result


def load_population_masks(arrays, specification):
    valid = np.isfinite(arrays['z'])
    require(not np.any(valid & ~arrays['support']), 'Exterior physical contribution')
    masks = primary_masks(valid, arrays['contact'], arrays['native'], arrays['anchors'], arrays['triangle'])
    partition = supplemental_masks(valid, arrays['native'], arrays['old_r5_radius'], arrays['original_q'], arrays['capture'])
    for name in PARTS:
        require(np.array_equal(arrays[name], partition[name]), 'Saved supplemental mask changed')
        masks[name] = partition[name]
    require(len(arrays['u']) == len(valid) and arrays['u'].shape == (len(valid), 6)
        and np.isfinite(arrays['u']).all() and np.isfinite(arrays['log_q']).all(), 'Original chart/density archive changed')
    require(np.array_equal(arrays['draw'], np.arange(len(valid))) and np.all(arrays['source_n'] == len(valid)), 'Archive lost attempted-draw identities')
    bin_counts = dict(radial=len(specification['radial_edges']) - 1,
        angular=len(specification['angular_projection_squared_edges']) - 1, orthant=64)
    for family, count in bin_counts.items():
        bins = arrays['bin_' + family]
        require(np.all((bins >= -1) & (bins < count)) and np.array_equal(bins == -1, ~arrays['support']), 'Invalid saved stratum')
        for index in range(count):
            for name in CLASSES:
                masks[f'stratum:{family}:{index}:{name}'] = masks[name] & (bins == index)
    for branch, selected in (('uniform-shell', arrays['branch'] == 0), ('gaussian', arrays['branch'] == 1)):
        for name in CLASSES:
            masks['branch:' + branch + ':' + name] = selected & masks[name]
    return masks, bin_counts


def summarize_arm(out, arm, records, specification, assessment):
    require(len(records) == 4 and len({r['id'] for r in records}) == 4 and len({r['seed'] for r in records}) == 4,
        'Exactly four independent populations required per arm')
    populations, counts = [], []; cpu = 0.
    for record in sorted(records, key=lambda p: p['id']):
        path = inside(out, record['records']); require(sha(path) == record['records_sha256'], 'Classification archive changed')
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in stored.files}
        require(len(arrays['z']) == record['samples'] == arm['samples'], 'Population denominator differs')
        masks, bin_counts = load_population_masks(arrays, specification)
        valid = np.isfinite(arrays['z'])
        require(record['full_native_classifier_calls'] == record['contact_classifier_calls'] == int(valid.sum()), 'First-pass classifier call count differs')
        dispositions = {}
        for branch, selected in (('uniform-shell', arrays['branch'] == 0), ('gaussian', arrays['branch'] == 1)):
            dispositions[branch] = dict(attempted=int(selected.sum()), exterior=int((selected & ~arrays['support']).sum()),
                contributing=int((selected & valid).sum()), **{name: int((selected & masks[name]).sum()) for name in CLASSES})
        populations.append(dict(id=record['id'], seed=record['seed'], z=arrays['z'], h=arrays['h'], pairs=arrays['pairs'], masks=masks))
        counts.append(dict(id=record['id'], seed=record['seed'], branches=dispositions,
            exterior_attempts=int((~arrays['support']).sum()), invalid_or_exterior_attempts=int((~valid).sum()),
            selected_component_counts=[int((arrays['component'] == i).sum()) for i in range(arm['component_count'])]))
        cpu += record['sampler_cpu_seconds']
    estimates, covariance, ratios = summarize_regions(populations)
    check_estimate(estimates['total']['row_uncertainty'], assessment['estimate'], assessment['hard_region'])
    assert_sum({name: estimates[name] for name in (NATIVE,) + PARTS})
    for name in PARTS:
        estimates[name]['observed_fraction_of_native'] = {kind: contribution_fraction(estimates[name], estimates[NATIVE], kind) for kind in ('Qz', 'Q0')}
    strata, branches = {}, {}
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
                actual = [e['row_uncertainty']['log_' + kind] for e in entries]
                actual = [v for v in actual if v is not None]; expected = estimates[name]['row_uncertainty']['log_' + kind]
                require(not actual and expected is None or actual and expected is not None
                    and abs(float(logsumexp(actual)) - expected) < 2e-10, 'Strata do not partition class mass')
    for branch in ('uniform-shell', 'gaussian'):
        branches[branch] = {}
        for name in CLASSES:
            item = estimates.pop('branch:' + branch + ':' + name)
            branches[branch][name] = dict(estimate=item, observed_class_fraction={kind: contribution_fraction(item, estimates[name], kind) for kind in ('Qz', 'Q0')})
    return dict(allocation=arm, estimates=estimates, primary_covariance=covariance, primary_ratios=ratios,
        sampler_cpu_seconds=cpu, populations=records, strata=strata, branch_dispositions=counts,
        branch_weight_contributions=branches, native_partition_sum_verified=True,
        observed_importance_ESS_per_cpu_second={name: estimates[name]['row_uncertainty'].get('Qz_ESS', 0) / cpu if cpu > 0 else None for name in CLASSES},
        efficiency_scope='Descriptive importance ESS/CPU; not contact-mixing ESS or a converged speedup.',
        native_entry_unbound_anomaly_count=sum(len(r['native_entry_unbound_anomalies']) for r in records),
        near_zero_negative_core_gap_count=sum(r['near_zero_negative_core_gaps'] for r in records),
        proposal_audit=assessment['importance_sampling'])


def evaluate(arms, protocol):
    require(set(arms) == set(ARM_NAMES), 'Five frozen confirmation arms required')
    require(protocol['decision_arms'] == list(MAIN_ARMS), 'Main decision arms differ')
    expected_comparisons = [['bank', name] for name in ARM_NAMES if name != 'bank']
    require(protocol['comparisons'] == expected_comparisons, 'Confirmation comparison pairs differ')
    gates = protocol['convergence']
    quality = {arm: {name: quality_gate(arms[arm]['estimates'][name], gates) for name in DECISION_CLASSES} for arm in ARM_NAMES}
    intervals = {arm: free_energy_interval(arms[arm], gates) for arm in ARM_NAMES}
    comparisons, significant = {}, []
    for left_name, right_name in protocol['comparisons']:
        left, right = arms[left_name], arms[right_name]
        masses = {name: {kind: compare_mass(left['estimates'][name], right['estimates'][name], kind, gates, absolute=True)
            for kind in ('Qz', 'Q0')} for name in CLASSES}
        contrast = compare_free_energy_intervals(intervals[left_name], intervals[right_name], gates)
        for family in ('radial', 'angular', 'orthant'):
            for name in DECISION_CLASSES:
                a_bins, b_bins = left['strata'][family][name], right['strata'][family][name]
                require(len(a_bins) == len(b_bins), 'Comparison stratum counts differ')
                for a, b in zip(a_bins, b_bins):
                    require(a['bin'] == b['bin'], 'Comparison stratum ordering differs')
                    fa, fb = a['observed_class_fraction']['Qz'], b['observed_class_fraction']['Qz']
                    if max(fa or 0., fb or 0.) < gates['significant_stratum_mass_fraction']:
                        continue
                    limit = dict(gates, log_agreement_absolute_max=gates['stratum_log_agreement_absolute_max'])
                    significant.append(dict(left_arm=left_name, right_arm=right_name, family=family, region=name,
                        bin=a['bin'], left_fraction=fa, right_fraction=fb, **compare_mass(a, b, 'Qz', limit)))
        comparisons[right_name] = dict(left_arm=left_name, right_arm=right_name, masses=masses,
            free_energy_contrast=contrast, region_agreement=all(masses[name][kind]['passed'] for name in DECISION_CLASSES for kind in ('Qz', 'Q0')),
            total_hard_mass_agreement=masses['total']['Q0']['passed'])
    checks = dict(main_regional_quality=all(quality[a][name]['passed'] for a in MAIN_ARMS for name in DECISION_CLASSES),
        main_paired_free_energy_precision=all(intervals[a]['passed'] for a in MAIN_ARMS),
        every_control_regional_agreement=all(c['region_agreement'] and c['total_hard_mass_agreement'] for c in comparisons.values()),
        every_control_direct_free_energy_agreement=all(c['free_energy_contrast']['passed'] for c in comparisons.values()),
        significant_original_strata_agreement=all(s['passed'] for s in significant),
        classifier_contact_consistency=all(arms[a]['native_entry_unbound_anomaly_count'] == 0 for a in ARM_NAMES),
        native_partition_sum=all(arms[a]['native_partition_sum_verified'] for a in ARM_NAMES))
    passed = all(checks.values())
    return dict(confirmation_passed=passed, passed=False, checks=checks, quality=quality,
        free_energy_intervals=intervals, comparisons=comparisons, significant_strata=significant,
        significant_stratum_disagreements=[s for s in significant if not s['passed']],
        decision_arms=list(MAIN_ARMS), diagnostic_precision_arms=['small'],
        small_arm_scope='Standalone regional ESS/RSE/largest draw and deltaF interval precision are diagnostic. Every bank-versus-small mass, direct deltaF and significant-stratum agreement check remains mandatory.',
        verdict='declared finite-R4 importance confirmation checks passed; independent coverage evidence remains required' if passed else 'unresolved finite-R4 importance confirmation',
        missing_confirmatory_checks=['independent full-R4 SMC and coverage comparison', 'remaining full-vessel contribution', 'finite-system assembly controls'],
        full_wall_coverage_established=False, assembly_stability_established=False)


def validate_terminal(root):
    from run_contact_confirmation import validate_completed
    protocol, status = validate_completed(root)
    require('analyze_contact_confirmation.py' in protocol['python_sources']
        and sha(__file__) == protocol['python_sources']['analyze_contact_confirmation.py'], 'Use the exactly frozen confirmation analyzer')
    require(protocol['decision_arms'] == list(MAIN_ARMS), 'Precision gate population policy changed')
    definition = inside(root, protocol['native_definition'])
    reference_path = inside(root, protocol['reference_region'])
    supplemental_path = inside(root, protocol['supplemental_definition'])
    require(sha(definition) == protocol['native_definition_sha256'], 'Full native definition changed')
    require(sha(reference_path) == protocol['reference_region_sha256'], 'Old R5 chart changed')
    require(sha(supplemental_path) == protocol['supplemental_definition_sha256'], 'Supplemental definition changed')
    supplemental = read(supplemental_path)
    require(supplemental['schema'] == SUPPLEMENTAL_SCHEMA, 'Unexpected supplemental definition')
    for name in ('reference_region_sha256', 'region_sha256', 'native_definition_sha256'):
        require(supplemental[name] == protocol[name], 'Supplemental target binding differs: ' + name)
    reference = read(reference_path)
    classifier, classifier_binding = load_classifier(definition)
    assessments = {}
    for arm in protocol['arms']:
        base = root / arm['id']; current = read(base / 'provenance/region.json')
        validate_regions(current, reference)
        validate_classifier_target(read(base / 'provenance/config.json'), protocol['shape_sha256'], classifier.definition, definition)
        assessment = read(base / 'assessment/analysis.json')
        n = sum(j['samples'] for j in protocol['jobs'] if j['arm'] == arm['id'])
        require(assessment['estimate']['draws'] == assessment['hard_region']['draws'] == assessment['independently_reconstructed_poses'] == n,
            'Completed audit discarded attempts')
        assessments[arm['id']] = assessment
    return protocol, status, assessments, definition, classifier_binding, reference_path, supplemental


def report(result):
    lines = ['# Fresh contact-weight confirmation', '', result['convergence']['verdict'] + '.', '', SCOPE, '',
        '| Arm | Region | log Qz | Population RSE | ESS | Largest draw |', '|---|---|---:|---:|---:|---:|']
    for name in ARM_NAMES:
        for region in DECISION_CLASSES:
            estimate = result['arms'][name]['estimates'][region]
            row, population = estimate['row_uncertainty'], estimate['population_uncertainty']
            values = ['unobserved', '—', '—', '—'] if row['log_Qz'] is None else [f'{row["log_Qz"]:.6f}',
                f'{population["Qz_relative_SE"]:.2%}', f'{row["Qz_ESS"]:.1f}', f'{row["largest_Qz_fraction"]:.2%}']
            lines.append('| ' + name + ' | ' + region + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '| Bank compared with | Regional masses | Direct ΔF |', '|---|---|---|']
    for name, comparison in result['convergence']['comparisons'].items():
        lines.append(f'| {name} | {comparison["region_agreement"]} | {comparison["free_energy_contrast"]["passed"]} |')
    lines += ['', result['convergence']['small_arm_scope'], '',
        'Agreement requires both absolute difference ≤0.2 and difference ≤3 combined observed population standard errors. '
        'Native/no-entry and both native subsets receive the same checks; every original radial, angular and orthant bin is retained. '
        'Unobserved regions remain unresolved. Approximate paired Student-t3 intervals are not unseen-tail bounds.', '',
        'The records archive retains all original u coordinates, proposal densities, metric q, cloud pairs and attempted-draw identities. '
        'Training estimates are not pooled. The old-R5 split is saved during the first full native classification pass.']
    return '\n'.join(lines) + '\n'


def analyze(campaign, out, workers=4):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(type(workers) is int and 1 <= workers <= 4, 'At most four classification workers')
    require(not out.exists(), 'Fresh analysis output required; no reclassification retries')
    protocol, status, assessments, definition, binding, reference_path, supplemental = validate_terminal(root)
    out.mkdir(parents=True); start = time.time()
    state = dict(complete=False, phase='first_classification_pass', completed_populations=[], started=start)
    write(out / 'status.json', state)
    try:
        tasks = []
        for job in protocol['jobs']:
            arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
            term = next(t for t in status['jobs'] if (t['arm'], t['id']) == (job['arm'], job['id']))
            audit = next(a for a in assessments[job['arm']]['populations'] if a['id'] == job['id'])
            tasks.append(dict(root=str(root), out=str(out), job=job, arm=arm, output=term['output'], audit=audit,
                definition=str(definition), classifier_binding=binding, strata=protocol['strata'],
                reference_region=str(reference_path), reference_region_sha256=protocol['reference_region_sha256'],
                supplemental_definition_sha256=protocol['supplemental_definition_sha256']))
        records = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(classify_population, task) for task in tasks]
            for future in as_completed(futures):
                record = future.result(); records.append(record)
                state['completed_populations'].append(record['arm'] + '/' + record['id'])
                write(out / 'status.json', state)
                print('First classification completed: ' + record['arm'] + '/' + record['id'], flush=True)
        state['phase'] = 'statistics'; write(out / 'status.json', state)
        arms = {}
        for arm in protocol['arms']:
            arms[arm['id']] = summarize_arm(out, arm, [r for r in records if r['arm'] == arm['id']], protocol['strata'], assessments[arm['id']])
            print('Summarized confirmation arm ' + arm['id'], flush=True)
        # Revalidate immutable inputs and all completed raw/audit bindings without replaying an audit.
        checked, checked_status, _, _, checked_binding, _, _ = validate_terminal(root)
        require(checked == protocol and checked_status == status and checked_binding == binding, 'Sources changed during analysis')
        result = dict(schema='contact-confirmation-comparison-v1', complete=True, campaign=str(root),
            protocol_sha256=sha(root / 'protocol.json'), status_sha256=sha(root / 'status.json'), freeze_sha256=sha(root / 'freeze.json'),
            region_sha256=protocol['region_sha256'], shape_sha256=protocol['shape_sha256'], native_definition=binding,
            supplemental_definition=supplemental, supplemental_definition_sha256=protocol['supplemental_definition_sha256'],
            reference_region_sha256=protocol['reference_region_sha256'], arms=arms,
            convergence=evaluate(arms, protocol), protocol_convergence=protocol['convergence'], strata_definition=protocol['strata'],
            primary_partition=list(PRIMARY), decision_regions=list(DECISION_CLASSES),
            total_unconditional_draws=protocol['total_unconditional_draws'],
            proposal_training=dict(preparation_plan_sha256=protocol['preparation_plan_sha256'],
                preparation_freeze_sha256=protocol['preparation_freeze_sha256'], component_groups=protocol['component_groups'],
                scope='All earlier reference, ray and fresh-pilot rows are training evidence only. These confirmation populations use disjoint frozen seeds and are never pooled with training estimates.'),
            unbound_finite_region_bound=unbound_volume_bound(read(root / 'bank/provenance/region.json')),
            classifier_passes_per_population=1, old_classifications_rerun=0, old_audits_replayed=0,
            scope=SCOPE, analysis_wall_seconds=time.time() - start, analysis_workers=workers)
        sources = out / 'provenance'; sources.mkdir()
        for name, path in local_sources(__file__).items():
            shutil.copy2(path, sources / name)
        for source, name in [(root / 'protocol.json', 'campaign-protocol.json'), (reference_path, 'old-r5-region.json'),
            (inside(root, protocol['supplemental_definition']), 'native-partition-definition.json')]:
            shutil.copy2(source, sources / name)
        result['analyzer_source_sha256'] = {p.name: sha(p) for p in sources.iterdir()}
        write(out / 'analysis.json', result); (out / 'report.md').write_text(report(result))
        state.update(complete=True, phase='complete', finished=time.time(), analysis_sha256=sha(out / 'analysis.json'))
        write(out / 'status.json', state)
        write(out / 'freeze.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return result
    except BaseException as error:
        state.update(phase=state['phase'] + '_failed', exception=repr(error), finished=time.time())
        write(out / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()
    analyze(args.campaign, args.out, args.workers)
    print(args.out.resolve(), flush=True)
