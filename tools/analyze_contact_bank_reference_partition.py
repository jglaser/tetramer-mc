#!/usr/bin/env python3
"""Apply the separately frozen R5-intersection diagnostic to completed fresh rows.

Never runs physics, a classifier, or either earlier audit. Original importance
weights and unconditional denominators are retained in both complementary masks.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from analyze_contact_bank_pilot import read_weights
from analyze_mobile_native_pocket import require, read, write, sha, inside, summarize, local_sources
from analyze_native_region_reference import native_q
from diagnose_conditional_ray_extremes import chart_coordinates
from review_conditional_ray_extremes import decode_block, jacobian

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = '497fa30e7389c8baf3c458060397f13a31f0227e1241f7dd38a9fff761010e50'
DEFAULT_PLAN = ROOT/'runs/contact-bank-reference-partition-plan-20260922/protocol.json'
NATIVE = 'registered_native_entry'
PARTS = ('old_R5_intersection_native', 'remaining_R4_native')
TARGET_FIELDS = ('physical_fixed_neighbors', 'capture_center', 'capture_radius', 'shape_sha256',
                 'activity', 'depletant_radius', 'physical_metric')
OBSERVER_FIELDS = ('definition_sha256', 'runtime_sha256', 'input_sha256', 'criteria', 'scope')
SCOPE = ('Additional named coverage diagnostic fixed before fresh weights were examined; the frozen primary '
    'convergence gates are unchanged. Each arm retains four separate populations and every original attempt, '
    'including exterior and hard-invalid zeros. The old R5 data trained the guide and are not fresh proposal '
    'validation. No source/pilot pooling, new Jacobian multiplier, new sampling, classifier evaluation, or '
    'audit replay. Observed agreement and dispersion cannot certify unseen mass or authorize production.')


class HashLedger:
    """Record only files actually verified, and detect edits before publication."""
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'Hash changed: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual, 'Source changed during analysis: '+str(path))
        self.files[str(path)] = actual
        return path

    def frozen(self, root):
        root = Path(root).resolve()
        manifest = read(self.bind(root/'freeze.json'))
        entries = manifest.get('files', manifest)
        require(isinstance(entries, dict) and entries, 'Empty freeze manifest')
        for name, digest in entries.items():
            require(name != 'freeze.json', 'Freeze cannot hash itself')
            self.bind(inside(root, name), digest)
        return entries

    def recheck(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, 'Source changed during analysis: '+path)


def r5_contains(radius, q, capture):
    """Exact reporting boundaries: rho<=5, original q>1, and original D170."""
    radius, q, capture = np.asarray(radius, float), np.asarray(q, float), np.asarray(capture)
    require(radius.ndim == 1 and radius.shape == q.shape == capture.shape and capture.dtype == bool,
            'R5 predicate arrays differ')
    require(np.isfinite(radius).all() and np.all(radius >= 0) and np.isfinite(q).all(), 'Invalid R5 coordinates')
    return (radius <= 5.) & (q > 1.) & capture


def partition_masks(valid, native, in_r5):
    valid, native, in_r5 = map(np.asarray, (valid, native, in_r5))
    require(valid.ndim == 1 and valid.shape == native.shape == in_r5.shape
            and all(a.dtype == bool for a in (valid, native, in_r5)), 'Partition arrays differ')
    require(not np.any(native & ~valid), 'Invalid zero row gained native classification')
    full = valid & native
    first, second = full & in_r5, full & ~in_r5
    require(not np.any(first & second) and np.array_equal(first | second, full), 'Native partition is not complementary')
    return {NATIVE: full, PARTS[0]: first, PARTS[1]: second}


def same_estimate(actual, expected):
    for name in ('draws', 'nonzero'):
        require(actual[name] == expected[name], 'Original '+name+' changed')
    for name in ('log_Qz', 'log_Q0'):
        a, b = actual[name], expected[name]
        require(a is None and b is None or a is not None and b is not None and abs(a-b) < 2e-10,
                'Original native '+name+' changed')


def assert_sum(estimates):
    """Check row and population means, preserving unobserved all-zero subsets."""
    levels = [('row_uncertainty', None), ('population_uncertainty', None)]
    levels += [('populations', i) for i in range(len(estimates[NATIVE]['populations']))]
    for level, index in levels:
        values = {k: v[level] if index is None else v[level][index] for k, v in estimates.items()}
        require(len({v['draws'] for v in values.values()}) == 1, 'Partition denominator changed')
        for field in ('log_Qz', 'log_Q0'):
            parts = [values[k][field] for k in PARTS if values[k][field] is not None]
            target = values[NATIVE][field]
            require(not parts and target is None or parts and target is not None
                    and abs(float(logsumexp(parts))-target) < 2e-10, 'Partition does not sum to native '+field)


def summarize_partition(populations, expected=None):
    require(len(populations) == 4 and len({p['id'] for p in populations}) == 4
            and len({p['seed'] for p in populations}) == 4, 'Four distinct fresh populations required')
    estimates = {name: summarize(populations, name, [p['masks'][name] for p in populations]) for name in (NATIVE,)+PARTS}
    assert_sum(estimates)
    if expected is not None:
        same_estimate(estimates[NATIVE]['row_uncertainty'], expected['row_uncertainty'])
        same_estimate(estimates[NATIVE]['population_uncertainty'], expected['population_uncertainty'])
        saved = {p['id']: p for p in expected['populations']}
        require(set(saved) == {p['id'] for p in populations}, 'Primary native populations differ')
        for population in estimates[NATIVE]['populations']:
            require(population['seed'] == saved[population['id']]['seed'], 'Primary native seed changed')
            same_estimate(population, saved[population['id']])
    for name in PARTS:
        estimates[name]['observed_fraction_of_native'] = {}
        for field in ('log_Qz', 'log_Q0'):
            child, parent = estimates[name]['row_uncertainty'][field], estimates[NATIVE]['row_uncertainty'][field]
            estimates[name]['observed_fraction_of_native'][field[4:]] = (None if parent is None else 0. if child is None else math.exp(child-parent))
    return estimates


def validate_regions(current, reference):
    require(all(current[k] == reference[k] for k in TARGET_FIELDS), 'Shape/scaffold/physical metric or measure target differs')
    require(current.get('minimum_mahalanobis_radius', 0.) == 0. and current['mahalanobis_radius'] == 4.
            and current['minimum_original_q'] == 0. and current.get('minimum_original_q_inclusive', True) is True
            and 'maximum_original_q' not in current, 'Fresh target is not the full R4')
    require(reference.get('minimum_mahalanobis_radius', 0.) == 0. and reference['mahalanobis_radius'] == 5.
            and reference['minimum_original_q'] == 1. and reference['minimum_original_q_inclusive'] is False
            and 'maximum_original_q' not in reference, 'Old R5 must retain strict original q>1')
    require(reference['capture_radius'] == 170. and reference['capture_center'] == [0., 0., 0.]
            and reference['activity'] == .035 and reference['depletant_radius'] == 1.5
            and len(reference['physical_fixed_neighbors']) == 2, 'Original D170/scaffold/bath changed')


def validate_sources(plan_path, ledger):
    plan_path = ledger.bind(plan_path, PLAN_SHA256); plan = read(plan_path)
    require(plan['schema'] == 'contact-bank-reference-partition-plan-v1', 'Wrong supplemental plan')
    comparison = Path(plan['primary_comparison']).resolve()
    frozen = ledger.frozen(comparison)
    require({'analysis.json', 'status.json'} <= set(frozen), 'Unbound primary comparison')
    data = read(comparison/'analysis.json'); state = read(comparison/'status.json')
    require(data['schema'] == 'contact-bank-pilot-comparison-v1' and data['complete'] is True
            and state['complete'] is True and state['phase'] == 'complete'
            and state['analysis_sha256'] == sha(comparison/'analysis.json'), 'Primary comparison is not complete')
    campaign = Path(data['campaign']).resolve()
    protocol = read(ledger.bind(campaign/'protocol.json', plan['pilot_protocol_sha256']))
    require(data['protocol_sha256'] == plan['pilot_protocol_sha256'], 'Comparison uses another pilot')
    ledger.bind(campaign/'freeze.json', data['freeze_sha256']); ledger.frozen(campaign)
    status = read(ledger.bind(campaign/'status.json', data['status_sha256']))
    require(status['complete'] is True and status['phase'] == 'complete'
            and status['protocol_sha256'] == plan['pilot_protocol_sha256'], 'Physical pilot is not complete')
    require(set(data['arms']) == {'bank', 'wide'} and len(protocol['arms']) == 2
            and {a['id'] for a in protocol['arms']} == {'bank', 'wide'}
            and len(protocol['jobs']) == len(status['jobs']) == 8
            and len({j['seed'] for j in protocol['jobs']}) == 8
            and data['total_unconditional_draws'] == protocol['total_unconditional_draws'] == 131072,
            'Fresh allocation changed')
    for job, terminal in zip(protocol['jobs'], status['jobs']):
        require({k: terminal[k] for k in job} == job and terminal['status'] == 'complete'
                and terminal['returncode'] == 0 and job['samples'] == 16384, 'Physical job did not finish its fixed allocation')
    reference_path = ledger.bind(plan['reference_comparison'], plan['reference_comparison_sha256'])
    reference = read(reference_path)
    require(reference['complete'] is True, 'Old reference incomplete')
    core = next(s for s in reference['strata'] if s['name'] == 'r5repeat')
    rr_path = ledger.bind(plan['reference_region'], plan['reference_region_sha256'])
    require(core['region_sha256'] == plan['reference_region_sha256']
            and rr_path == Path(core['campaign']).resolve()/'provenance/region.json', 'Wrong old reference chart')
    rr = read(rr_path); rp = rr_path.parent
    review_path = ledger.bind(plan['identity_review'], plan['identity_review_sha256']); review = read(review_path)
    require(review['complete'] is True and review['new_physical_samples'] == review['classifiers_rerun'] == 0,
            'Identity review is not completed saved evidence')
    require(review['source_sha256'][str(reference_path)] == plan['reference_comparison_sha256']
            and review['source_sha256'][str(rr_path)] == plan['reference_region_sha256'], 'Identity review references another R5')
    for path, digest in core['source_sha256'].items():
        ledger.bind(path, digest)
    ledger.bind(inside(reference_path.parent, core['native_labels']['path']), core['native_labels']['sha256'])
    require(len(core['populations']) == 8 and core['samples_per_population'] == 16384, 'Old reference denominator changed')
    old_seeds = set()
    for p in core['populations']:
        require(p['draws'] == 16384 and review['reference_denominators'][p['id']] ==
                dict(seed=p['seed'], unconditional_draws=16384), 'Old reference attempt count or seed changed')
        old_seeds.add(p['seed'])
    require(old_seeds.isdisjoint(j['seed'] for j in protocol['jobs']), 'Old/fresh seeds overlap')
    require(review['subset_estimate']['row_uncertainty']['draws'] == 131072
            and review['subset_estimate']['population_count'] == 8
            and review['observed_reference_current_radius_range'][1] <= 4., 'Old matching intersection lost its denominator or R4 support')
    same_estimate(review['subset_estimate']['row_uncertainty'], core['row_uncertainty'])
    same_estimate(review['subset_estimate']['population_uncertainty'], core['population_uncertainty'])
    old_config = read(ledger.bind(rp/'config.json')); old_bundle = read(ledger.bind(rp/'source-bundle.json'))
    for path in (rp/'config.json', rp/'source-bundle.json'):
        require(review['source_sha256'][str(path)] == ledger.files[str(path)], 'Old identity-reviewed input changed')
    shape = sha(ledger.bind(rp/'shape.json', rr['shape_sha256']))
    require(shape == reference['shape_sha256'] == data['shape_sha256'] == protocol['shape_sha256'] == review['shape_sha256'], 'Shape identity changed')
    for field in OBSERVER_FIELDS:
        require(data['native_definition'][field] == reference['native_definition'][field], 'Full native observer differs: '+field)
    regions = {}
    for arm in protocol['arms']:
        name = arm['id']; cp = campaign/name/'provenance'
        current = read(ledger.bind(cp/'region.json', protocol['region_sha256']))
        config = read(ledger.bind(cp/'config.json')); bundle = read(ledger.bind(cp/'source-bundle.json'))
        ledger.bind(cp/'shape.json', shape); validate_regions(current, rr)
        require({k: v for k, v in config.items() if k != 'shape'} == {k: v for k, v in old_config.items() if k != 'shape'},
                'Configuration differs beyond shape archive path')
        require(decode_block(bundle) == decode_block(old_bundle), 'Physical chart decoder/Jacobian changed')
        for key, digest in review['common_physical_source_sha256'].items():
            require(bundle['files'][key]['sha256'] == old_bundle['files'][key]['sha256'] == digest, 'Physical source changed: '+key)
        audit = status['audits'][name]
        ledger.bind(campaign/name/'assessment/analysis.json', audit['analysis_sha256'])
        require(audit['returncode'] == 0 and data['arms'][name]['allocation'] == arm, 'Fresh audit/allocation changed')
        regions[name] = current
    require(data['region_sha256'] == protocol['region_sha256'], 'Fresh comparison region changed')
    return plan, comparison, data, campaign, protocol, status, rr, regions, review


def read_population(comparison, campaign, record, job, terminal, arm, current, reference, ledger):
    require((record['arm'], record['id'], record['seed'], record['samples']) ==
            (job['arm'], job['id'], job['seed'], job['samples']) and record['raw_output'] == terminal['output'], 'Classification/raw job binding changed')
    directory = Path(job['directory']).resolve()
    require(directory == campaign/job['arm']/'runs'/job['id'], 'Raw population path changed')
    for name, key in (('samples.jsonl', 'samples_sha256'), ('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')):
        ledger.bind(directory/name, record['raw_output'][key])
    pm, summary = read(directory/'manifest.json'), read(directory/'summary.json')
    require(pm['samples'] == summary['samples'] == job['samples'] and pm['seed'] == job['seed']
            and summary['complete'] is True and summary['manifest'] == pm
            and pm['region_sha256'] == sha(campaign/job['arm']/'provenance/region.json')
            and pm['shape_sha256'] == current['shape_sha256'] and pm['cloud_replicates'] == 2
            and pm['physical_fixed_neighbors'] == current['physical_fixed_neighbors']
            and pm['minimum_original_q'] == 0. and pm['minimum_original_q_inclusive'] is True
            and pm['maximum_original_q'] is None and pm['minimum_latent_radius'] == 0. and pm['latent_radius'] == 4.
            and pm['proposal_density_measure'] == 'Lebesgue measure in the original six-dimensional whitened region chart', 'Fresh manifest target/measure changed')
    with (directory/'samples.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    fresh = read_weights(rows, job['samples'], current, arm)
    archive = ledger.bind(inside(comparison, record['records']), record['records_sha256'])
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    for key in ('z', 'h', 'pairs', 'branch', 'component', 'support'):
        require(np.array_equal(arrays[key], fresh[key]), 'Classifier records differ from raw '+key)
    require(arrays['native'].dtype == bool and arrays['native'].shape == (job['samples'],), 'Wrong full-native record shape')
    labels = ledger.bind(inside(comparison, record['labels']), record['labels_sha256'])
    valid = np.isfinite(fresh['z']); seen = 0
    with gzip.open(labels, 'rt') as stream:
        for i, line in enumerate(stream):
            require(i < len(rows), 'Too many classifier labels')
            label = json.loads(line)
            require(label['draw'] == i and type(label['applicable']) is bool and label['applicable'] == bool(valid[i]), 'Classifier label applicability changed')
            if valid[i]:
                classification, contact = label['classification'], label['contact']
                require(type(classification['native_any']) is bool and classification['native_any'] == bool(arrays['native'][i])
                        and contact['exclusion_contact'] == bool(arrays['contact'][i]), 'Full native/contact record differs from saved labels')
            else:
                require(label['classification'] is None and label['contact'] is None
                        and not arrays['native'][i] and not arrays['contact'][i], 'Invalid zero row gained classification')
            seen += 1
    require(seen == len(rows), 'Missing classifier labels')
    poses = [r['pose'] for r in rows]
    current_u = chart_coordinates(poses, current); old_u = chart_coordinates(poses, reference)
    inverse_error = float(np.max(abs(current_u-fresh['latents'])))
    require(inverse_error < 2e-8, 'Fresh physical pose/chart mismatch')
    logj_error = float(np.max(abs(jacobian(current_u, current)-[r['log_physical_jacobian'] for r in rows])))
    require(logj_error < 2e-8, 'Fresh physical Jacobian mismatch')
    radii = np.linalg.norm(old_u, axis=1); q = np.asarray([r['q'] for r in rows])
    positions = np.asarray([r['pose']['position'] for r in rows])
    captured = np.linalg.norm(positions-reference['capture_center'], axis=1) <= reference['capture_radius']
    require(np.array_equal(captured, [r['capture_valid'] for r in rows]), 'Original D170 capture predicate differs')
    # The q scalar was already independently audited for every attempted row.
    # Recheck native reporting rows against that same original physical metric.
    q_error = 0.
    for i in np.flatnonzero(arrays['native']):
        actual = native_q(reference['physical_metric'], rows[i]['pose'])
        q_error = max(q_error, abs(actual-q[i]))
        require(abs(actual-q[i]) < 2e-8 and (actual > 1.) == (q[i] > 1.), 'Original strict q>1 predicate differs')
    masks = partition_masks(valid, arrays['native'], r5_contains(radii, q, captured))
    population = dict(id=job['id'], seed=job['seed'], z=fresh['z'], h=fresh['h'], pairs=fresh['pairs'], masks=masks)
    diagnostics = dict(id=job['id'], seed=job['seed'], unconditional_draws=len(rows),
        exterior_attempts=int((~fresh['support']).sum()), invalid_or_exterior_attempts=int((~valid).sum()),
        counts={k: int(v.sum()) for k, v in masks.items()}, maximum_current_chart_inverse_error=inverse_error,
        maximum_current_log_jacobian_error=logj_error, maximum_native_original_q_error=q_error)
    return population, diagnostics


def analyze(plan_path, out):
    out = Path(out).resolve(); require(not out.exists(), 'Refuse to overwrite a completed or partial diagnostic')
    ledger = HashLedger()
    plan, comparison, primary, campaign, protocol, status, reference, regions, review = validate_sources(plan_path, ledger)
    arms = {}
    for arm in protocol['arms']:
        name = arm['id']; jobs = [j for j in protocol['jobs'] if j['arm'] == name]
        records = primary['arms'][name]['populations']
        require(len(jobs) == len(records) == 4 and {r['id'] for r in records} == {j['id'] for j in jobs}, 'Fresh arm lost a population')
        populations, diagnostics = [], []
        for job in sorted(jobs, key=lambda p: p['id']):
            record = next(r for r in records if r['id'] == job['id'])
            terminal = next(t for t in status['jobs'] if (t['arm'], t['id']) == (name, job['id']))
            population, diagnostic = read_population(comparison, campaign, record, job, terminal, arm, regions[name], reference, ledger)
            populations.append(population); diagnostics.append(diagnostic)
        estimates = summarize_partition(populations, primary['arms'][name]['estimates'][NATIVE])
        old = review['subset_estimate']['row_uncertainty']['log_Qz']
        fresh = estimates[PARTS[0]]['row_uncertainty']['log_Qz']
        arms[name] = dict(estimates=estimates, populations=diagnostics, native_partition_sum_verified=True,
            old_matching_reference_log_Qz_difference=None if fresh is None or old is None else fresh-old)
    result = dict(schema='contact-bank-reference-partition-v1', complete=True, plan_sha256=PLAN_SHA256,
        primary_comparison=str(comparison), pilot_protocol_sha256=plan['pilot_protocol_sha256'],
        supplemental_predicate=plan['predicate'], arms=arms, total_fresh_unconditional_draws=131072,
        old_matching_reference=dict(estimate=review['subset_estimate'],
            reference_comparison_sha256=plan['reference_comparison_sha256'], reference_region_sha256=plan['reference_region_sha256'],
            identity_review_sha256=plan['identity_review_sha256'],
            scope='Saved R5/native intersection with current R4, retaining all eight original populations and old weights. '
                  'Observed contributor inclusion does not prove geometric containment of the whole R5. These data trained the guide; not fresh validation.'),
        shape_sha256=primary['shape_sha256'], native_definition=primary['native_definition'], physical_measure='d^3t times normalized SO(3) Haar',
        estimator='For each complementary native mask: mean over all N original attempts of mask times the saved J/q times (W1+W2)/2; no coordinate-change reweighting.',
        new_physical_samples=0, classifiers_rerun=0, audits_replayed=0, primary_gates_changed=False,
        full_vessel_or_assembly_authorized=False, scope=SCOPE)
    sources = local_sources(__file__)
    for path in sources.values():
        ledger.bind(path)
    ledger.recheck(); result['source_sha256'] = ledger.files
    out.mkdir(parents=True); provenance = out/'provenance'; provenance.mkdir()
    for name, path in sources.items():
        shutil.copy2(path, provenance/name)
    shutil.copy2(plan_path, out/'supplemental-plan.json')
    write(out/'analysis.json', result)
    lines = ['# Separately declared native R5 intersection', '', SCOPE, '',
        '| Arm | Native reporting region | log Qz | Population RSE | ESS |', '|---|---|---:|---:|---:|']
    for name, source in arms.items():
        for key, estimate in source['estimates'].items():
            row, pop = estimate['row_uncertainty'], estimate['population_uncertainty']
            values = ['unobserved', '—', '—'] if row['log_Qz'] is None else [f"{row['log_Qz']:.6f}", f"{pop['Qz_relative_SE']:.2%}", f"{row['Qz_ESS']:.1f}"]
            lines.append('| '+name+' | '+key+' | '+' | '.join(values)+' |')
    lines += ['', 'R5 intersection uses the old chart radius ≤5, strict original q>1, and D170 within the complete fresh native R4 class. The complement and intersection sum to the unchanged native Qz and Q0 for every population and each arm. Zero observations mean unresolved mass, not proven zero mass.', '',
        'The saved old matching-reference estimate uses eight original populations. It remains separate because its poses were used to fit the guide. The fresh native weights receive no second Jacobian multiplier.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out/'freeze.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.plan, args.out)
    print(args.out.resolve())
