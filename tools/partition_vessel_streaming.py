#!/usr/bin/env python3
"""Classify native registry once, reusing a completed streaming geometry audit.

All masses retain the full attempted-draw denominator. Historical pockets and
the R4 chart are reporting predicates, never filters on the vessel target.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil
import sys
import time

from scipy.special import logsumexp
from analyze_mobile_native_pocket import load_classifier, local_sources
from analyze_r4_smc_control import Ledger, close, read, require, sha, write
from vessel_contact_partition import Mass, PRIMARY, SUPPORTS, RegionSupport, partition

SCHEMA = 'full-vessel-streaming-native-partition-v1'
AUDITS = ('full-vessel-baseline-streaming-audit-v1', 'full-vessel-hard-free-line-streaming-audit-v1')
KINDS = ('Qz', 'Q0')
REGIONAL = ('registered_native_entry', 'contact_no_native_entry',
            'old_R5_intersection_native', 'remaining_R4_native', 'unbound_no_native_entry')
STRATA = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
              orthants=64, zero_sign='positive', outside_current_R4_bin=-1)
BIN_COUNTS = dict(radial=3, angular=3, orthant=64)


def classify(valid, contact, native, supports):
    require(all(type(x) is bool for x in (valid, contact, native)), 'Non-Boolean primary input')
    require(valid or not (contact or native or any(supports.values())), 'Invalid pose acquired a label')
    primary = dict(registered_native_entry=valid and native,
        contact_no_native_entry=valid and contact and not native,
        unbound_no_native_entry=valid and not contact and not native)
    classes = partition(valid, primary, supports)
    inside = supports['current_R4']; old = supports['old_alternative_R5']
    regional = {k: v and inside for k, v in primary.items()}
    regional.update(old_R5_intersection_native=valid and native and inside and old,
                    remaining_R4_native=valid and native and inside and not old)
    require(regional['old_R5_intersection_native'] + regional['remaining_R4_native']
            == regional['registered_native_entry'], 'Regional native split differs')
    return classes, regional


def validate_supports(regions, config, manifest):
    require(set(regions) == set(SUPPORTS), 'All four frozen pocket definitions required')
    for region in regions.values():
        require(region['shape_sha256'] == manifest['shape_sha256']
                and region['physical_fixed_neighbors'] == config['fixed_poses']
                and region['fixed_neighbor'] in config['fixed_poses'], 'Reporting shape/scaffold differs')
        require(region['physical_metric'] == config['metadata'], 'Reporting registration metric differs')
        for rk, ck in [('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius')]:
            require(region[rk] == config[ck], 'Reporting bath differs: ' + rk)
    current, old = regions['current_R4'], regions['old_alternative_R5']
    require(current['mahalanobis_radius'] == 4. and current.get('minimum_mahalanobis_radius', 0.) == 0.
            and current['minimum_original_q'] == 0. and current['minimum_original_q_inclusive'] is True
            and current.get('maximum_original_q') is None, 'Current full R4 definition differs')
    require(old['mahalanobis_radius'] == 5. and old.get('minimum_mahalanobis_radius', 0.) == 0.
            and old['minimum_original_q'] == 1. and old['minimum_original_q_inclusive'] is False
            and old.get('maximum_original_q') is None, 'Old R5 must retain radius <=5 and original q>1')
    require(old['capture_center'] == current['capture_center'] and old['capture_radius'] == current['capture_radius'],
            'Old R5 capture differs from the frozen regional comparison')
    return {k: RegionSupport(r) for k, r in regions.items()}


def bind_native(ledger, definition_path, config, manifest):
    definition_path = ledger.bind(definition_path); definition = read(definition_path)
    require(definition['shape_sha256'] == manifest['shape_sha256']
            and definition['fixed_poses'] == config['fixed_poses'], 'Native geometry identity differs')
    input_root = (definition_path.parent/'inputs').resolve()
    for name, digest in definition['input_sha256'].items():
        path = (input_root/name).resolve()
        require(path.is_relative_to(input_root), 'Native input escapes frozen directory')
        ledger.bind(path, digest)
    native_config_path = ledger.bind(input_root/'physical-config.json', definition['physical_config_sha256'])
    native_config = read(native_config_path)
    for key in ('fixed_poses', 'depletant_radius', 'reservoir_density', 'metadata'):
        require(config[key] == native_config[key], 'Native observer physical law differs: ' + key)
    # Capture radii intentionally differ: source-region support versus full vessel.
    return load_classifier(definition_path)


def pair():
    return {kind: Mass() for kind in KINDS}


def add(reducers, row, selected):
    if selected:
        reducers['Qz'].add(row['log_importance_weight'])
        reducers['Q0'].add(row['log_hard_weight'])


def check_sum(parent, children):
    for kind in KINDS:
        a, bs = parent[kind], [b[kind] for b in children]
        require(a['nonzero'] == sum(b['nonzero'] for b in bs), 'Partition count differs')
        for field in ('logQ', 'log_sum_squared_weights', 'log_max_weight'):
            values = [b[field] for b in bs if b[field] is not None]
            require((a[field] is None) == (not values), 'Partition support differs')
            if values: close(a[field], max(values) if field == 'log_max_weight' else float(logsumexp(values)),
                             'Partition sufficient statistics differ')


def analyze(audit_path, region_paths, definition_path, out):
    require(sys.flags.optimize == 0, 'Unoptimized Python required')
    audit_path, out = Path(audit_path).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh native partition destination required')
    ledger = Ledger(); frozen = ledger.frozen(audit_path.parent)
    require({'analysis.json', 'status.json', 'geometry.jsonl'} <= set(frozen), 'Incomplete frozen audit')
    audit = read(ledger.bind(audit_path)); status = read(audit_path.parent/'status.json')
    require(audit['schema'] in AUDITS and audit['complete'] is True and status['complete'] is True
            and status['phase'] == 'complete' and status['analysis_sha256'] == sha(audit_path), 'Incomplete streaming audit')
    root = Path(audit['population']).resolve()
    manifest = read(ledger.bind(root/'manifest.json')); require(manifest == audit['manifest'], 'Manifest changed')
    n = manifest['samples']; require(type(n) is int and n > 0, 'Positive attempted-draw allocation required')
    summary = read(ledger.bind(root/'summary.json'))
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['samples'] == n
            and not (root/'failure.json').exists(), 'Physical population incomplete')
    raw = ledger.bind(root/'samples.jsonl', audit['samples_sha256'])
    ledger.bind(root/'attempts.jsonl', audit['attempts_sha256'])
    require(summary['samples_sha256'] == audit['samples_sha256']
            and summary['attempts_sha256'] == audit['attempts_sha256'], 'Audited completion hashes differ')
    geometry = ledger.bind(audit_path.parent/'geometry.jsonl', audit['geometry_sha256'])
    config_path = (root/'config.json').resolve()
    config = read(ledger.bind(config_path, audit['source_sha256'][str(config_path)]))
    current_sha = (audit['reporting_region_binding']['sha256'] if audit['schema'] == AUDITS[0]
                   else manifest['latent_region_sha256'])
    require(set(region_paths) == set(SUPPORTS), 'Freeze all current and historical supports')
    ledger.bind(region_paths['current_R4'], current_sha)
    regions = {k: read(ledger.bind(v)) for k, v in region_paths.items()}
    supports = validate_supports(regions, config, manifest)
    classifier, native_binding = bind_native(ledger, definition_path, config, manifest)
    sources = local_sources(__file__)
    for path in sources.values(): ledger.bind(path)
    estimates = {k: pair() for k in partition(False, dict.fromkeys(PRIMARY, False), dict.fromkeys(SUPPORTS, False))}
    regional = {k: pair() for k in REGIONAL}
    strata = {family: {k: [pair() for _ in range(size)] for k in REGIONAL} for family, size in BIN_COUNTS.items()}
    processed = invalid = calls = anomalies = 0; started = time.process_time()
    out.mkdir(parents=True)
    def save_status(phase, **extra):
        temporary = out/'status.json.tmp'
        write(temporary, dict(phase=phase, complete=phase == 'complete', processed_attempts=processed,
                              new_native_classifier_calls=calls, **extra)); temporary.replace(out/'status.json')
    save_status('classification')
    try:
        with raw.open() as raws, geometry.open() as geometries, (out/'labels.jsonl').open('x') as output:
            for rtext, gtext in zip_longest(raws, geometries):
                require(rtext is not None and gtext is not None, 'Lost raw or geometry draw')
                row, geo = json.loads(rtext), json.loads(gtext)
                require(type(row['draw']) is int and type(geo['draw']) is int
                        and row['draw'] == geo['draw'] == processed < n, 'Lost/reordered attempted draw')
                valid = row['hard_valid']; require(type(valid) is bool and geo['classes']['total'] is valid,
                                                  'Audited validity differs')
                contact = geo['geometry']['exclusion_contact'] if valid else False
                label = None
                if valid:
                    require(type(contact) is bool and contact is row['depletion_contact'], 'Audited contact differs')
                    calls += 1; label = classifier.classify(row['pose'])
                    require(type(label['native_any']) is bool, 'Non-Boolean native classification')
                else:
                    require(row['log_importance_weight'] is None and row['log_hard_weight'] is None,
                            'Hard-invalid draw acquired weight')
                native = label['native_any'] if valid else False
                witnesses = {k: bool(s.contains(row['pose'])) for k, s in supports.items()} if valid else dict.fromkeys(SUPPORTS, False)
                classes, local = classify(valid, contact, native, witnesses)
                inside = witnesses['current_R4']
                bins = supports['current_R4'].chart.bins(supports['current_R4'].chart.evaluate(row['pose'])['latent']) if inside else dict.fromkeys(BIN_COUNTS, -1)
                for name, selected in classes.items(): add(estimates[name], row, selected)
                for name, selected in local.items():
                    add(regional[name], row, selected)
                    if selected:
                        for family, index in bins.items():
                            require(0 <= index < BIN_COUNTS[family], 'Selected regional stratum outside R4')
                            add(strata[family][name][index], row, True)
                anomaly = native and not contact; anomalies += int(anomaly)
                output.write(json.dumps(dict(draw=processed, classes=classes, regional_classes=local,
                    witnesses=witnesses, strata=bins, native=label, native_unbound_anomaly=anomaly), allow_nan=False)+'\n')
                processed += 1; invalid += int(not valid)
                if processed % 256 == 0: output.flush(); save_status('classification')
        require(processed == n and calls == n-invalid, 'Unconditional denominator or classifier calls differ')
        finish = lambda values: {k: v.result(n) for k, v in values.items()}
        estimates = {k: finish(v) for k, v in estimates.items()}
        regional = {k: finish(v) for k, v in regional.items()}
        strata = {f: {k: [finish(v) for v in bs] for k, bs in by_class.items()} for f, by_class in strata.items()}
        for kind in KINDS:
            old, new = audit['estimates']['total'][kind], estimates['total'][kind]
            require(old['nonzero'] == new['nonzero'] and (old['logQ'] is None) == (new['logQ'] is None), 'Audited total support changed')
            if old['logQ'] is not None: close(old['logQ'], new['logQ'], 'Audited total mass changed')
        check_sum(estimates['total'], [estimates[k] for k in PRIMARY])
        check_sum(regional['registered_native_entry'], [regional[k] for k in REGIONAL[2:4]])
        for by_class in strata.values():
            for name, bins in by_class.items(): check_sum(regional[name], bins)
        ledger.recheck(); (out/'provenance').mkdir()
        for name, path in sources.items():
            shutil.copy2(path, out/'provenance'/name)
            require(sha(out/'provenance'/name) == ledger.files[str(Path(path).resolve())], 'Source changed while archiving')
        ledger.recheck()
        result = dict(schema=SCHEMA, complete=True, population=str(root), audit=str(audit_path), manifest=manifest,
            samples=n, invalid_draws=invalid, estimates=estimates, regional_estimates=regional, strata=strata,
            stratum_definition=STRATA, region_paths={k: str(Path(v).resolve()) for k, v in region_paths.items()},
            native_binding=native_binding, native_definition_sha256=native_binding['definition_sha256'],
            new_native_classifier_calls=calls, native_unbound_anomalies=anomalies,
            new_geometry_queries=0, new_pose_draws=0, new_clouds=0,
            input_sha256=ledger.files, labels_sha256=sha(out/'labels.jsonl'), analysis_CPU_seconds=time.process_time()-started,
            scope='All-attempt vessel masses, complete native observer, and Boolean pocket union. '
                  'Regional strata use full source-region support, not the chart-ball-only audit label. '
                  'No new density or Jacobian factor. Unobserved classes are unresolved, not zero-mass bounds. '
                  'These conditional scaffold data alone do not decide finite-system assembly.')
        write(out/'analysis.json', result); save_status('complete', analysis_sha256=sha(out/'analysis.json'))
        write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    except BaseException as error:
        save_status('failed', error=str(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'native-definition', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    for name in SUPPORTS: parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.audit, {k: getattr(args, k) for k in SUPPORTS}, args.native_definition, args.out)
    print(json.dumps(dict(complete=result['complete'], samples=result['samples'], native_calls=result['new_native_classifier_calls'])))
