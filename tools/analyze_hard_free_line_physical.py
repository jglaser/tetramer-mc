#!/usr/bin/env python3
"""Classify freshly audited line-guide populations once; preserve all zeros.

The historical classifier, strata and statistical summaries are reused. No
training population enters this fresh comparison, and no assembly claim follows.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_contact_confirmation import (classify_rows_once, summarize_arm,
    load_classifier, validate_classifier_target, validate_regions, DECISION_CLASSES,
    compare_mass, quality_gate, free_energy_interval, unbound_volume_bound)
from analyze_contact_bank_pilot import stratify_with_exterior
from analyze_mobile_threshold_reference import ExclusionContact
from analyze_mobile_competing_reference import paired_moments, check_estimate
from prepare_hard_free_line_score import read, sha, require, write
from hard_free_line_physical_reference import read_weights


def relabel_nonuniform(value):
    """Old numeric branch 1 means Gaussian lineage, including conditioning.

    Relabel presentation only. Masks, estimates, arrays and original row labels
    remain unchanged; this must not describe a conditioned law as Gaussian.
    """
    if isinstance(value, dict):
        return {('nonuniform_guide' if k == 'gaussian' else k): relabel_nonuniform(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [relabel_nonuniform(v) for v in value]
    return value


def classify_population(root, destination, arm, job, protocol, classifier, binding):
    directory = Path(job['directory']); target = destination/arm['id']/job['id']
    require(not target.exists(), 'No repeated classifier pass')
    audit_path = root/'audits'/arm['id']/(job['id']+'.json')
    audit = read(audit_path)
    require(audit['complete'] and audit['geometry_mode'] == 'full', 'Complete independent geometry audit required')
    provenance = directory/'provenance'
    region, config, shape = [read(provenance/name) for name in ('region.json', 'input-config.json', 'shape.json')]
    reference_path = root/'common/reference-package/old-r5-region.json'
    reference = read(reference_path); validate_regions(region, reference)
    definition = root/'common/reference-package/native-region/definition.json'
    validate_classifier_target(config, sha(provenance/'shape.json'), classifier.definition, definition)
    rows_path = directory/'samples.jsonl'; row_hash = sha(rows_path)
    # Bind the immutable job and independent audit, then retain all attempted rows.
    summary = read(directory/'summary.json')
    require(row_hash == summary['samples_sha256'], 'Raw population changed')
    require(audit['input_sha256'][str(rows_path.resolve())] == row_hash
        and audit['input_sha256'][str((directory/'summary.json').resolve())] == sha(directory/'summary.json'),
        'Independent audit refers to different physical outputs')
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()]
    arrays = read_weights(rows, job['samples'], region, arm)
    check_estimate(paired_moments(arrays['z'], arrays['h']), audit['estimate'], audit['hard_region'])
    contact = ExclusionContact(shape, config['fixed_poses'], config['depletant_radius'])
    target.mkdir(parents=True)
    labels = target/'labels.jsonl.gz'
    with labels.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
        classified, diagnostics = classify_rows_once(rows, arrays, classifier, contact, reference, stream)
    latents = arrays.pop('latents')
    bins = stratify_with_exterior(latents, region, protocol['strata'], arrays['support'])
    records = target/'records.npz'
    np.savez_compressed(records, **arrays, **classified, u=latents,
        log_q=np.asarray([r['log_proposal_density'] for r in rows]),
        log_physical_jacobian=np.asarray([r['log_physical_jacobian'] for r in rows]),
        draw=np.arange(len(rows)), source_n=np.full(len(rows), job['samples'], np.int32),
        **{'bin_'+k: v for k, v in bins.items()})
    require(sha(rows_path) == row_hash, 'Population changed during classification')
    record = dict(id=job['id'], arm=arm['id'], seed=job['seed'], samples=job['samples'],
        sampler_cpu_seconds=summary['sampler_cpu_seconds'],
        records=str(records.relative_to(destination)), records_sha256=sha(records),
        labels=str(labels.relative_to(destination)), labels_sha256=sha(labels),
        independent_audit_sha256=sha(audit_path), samples_sha256=row_hash,
        full_native_definition_sha256=binding['definition_sha256'],
        reference_region_sha256=sha(reference_path),
        supplemental_definition_sha256=protocol['supplemental_definition_sha256'], **diagnostics)
    write(target/'classification.json', record)
    return record, arrays


def compare(arms, gates):
    baseline, candidate = arms['baseline'], arms['conditioned']
    strata = []
    for family in ('radial', 'angular', 'orthant'):
        for region in DECISION_CLASSES:
            for a, b in zip(baseline['strata'][family][region], candidate['strata'][family][region]):
                significant = max(a['observed_class_fraction']['Qz'] or 0., b['observed_class_fraction']['Qz'] or 0.) >= gates['significant_stratum_mass_fraction']
                strata.append(dict(family=family, region=region, bin=a['bin'], significant=significant,
                    comparison=compare_mass(a, b, 'Qz', gates)))
    return dict(regional_comparisons={region: compare_mass(baseline['estimates'][region],
        candidate['estimates'][region], 'Qz', gates) for region in DECISION_CLASSES},
        quality={name: {region: quality_gate(data['estimates'][region], gates)
            for region in DECISION_CLASSES} for name, data in arms.items()},
        free_energy_intervals={name: free_energy_interval(data, gates) for name, data in arms.items()},
        all_stratum_comparisons=strata,
        full_vessel_gate_open=False, assembly_gate_open=False,
        limitation='Fixed R4 pilot only; population-size, intensity, full-domain and unseen-mode checks remain outstanding.')


def analyze(root):
    root = Path(root).resolve(); protocol = read(root/'protocol.json')
    destination = root/'comparison'; require(not destination.exists(), 'No repeated analysis/classification')
    destination.mkdir()
    classifier, binding = load_classifier(root/'common/reference-package/native-region/definition.json')
    arms = {}
    for arm in protocol['arms']:
        records, arrays = [], []
        for job in (j for j in protocol['jobs'] if j['arm'] == arm['id']):
            record, values = classify_population(root, destination, arm, job, protocol, classifier, binding)
            records.append(record); arrays.append(values)
        pooled = paired_moments(np.concatenate([a['z'] for a in arrays]), np.concatenate([a['h'] for a in arrays]))
        assessment = dict(estimate=dict(logQ=pooled['log_Qz'], draws=pooled['draws']),
            hard_region=dict(logQ=pooled['log_Q0'], draws=pooled['draws']),
            importance_sampling=dict(schema='defensive-hard-free-line-guide-v1', beta=arm['beta'],
                alpha=.5, independent_geometry='full', preserved_attempts=pooled['draws']))
        arms[arm['id']] = relabel_nonuniform(summarize_arm(destination, arm, records, protocol['strata'], assessment))
    result = dict(schema='hard-free-line-physical-comparison-v1', complete=True,
        protocol_sha256=sha(root/'protocol.json'), arms=arms, diagnostics=compare(arms, protocol['convergence']),
        total_unconditional_draws=protocol['total_unconditional_draws'],
        native_definition=binding, old_samples_pooled=False, old_classifier_calls=0,
        unbound_R4_bound=unbound_volume_bound(read(root/'common/reference-package/region.json')))
    write(destination/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); print(analyze(args.root)['complete'])
