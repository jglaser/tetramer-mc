#!/usr/bin/env python3
"""Arithmetic-only regional statistics for separately labelled v6/v7 rows.

This is a downstream adapter, not a label producer, geometry auditor, allocation
or launcher. ``analyze(plan_path)`` consumes a hash-bound JSON plan with ``target``
references (region, old_r5_region, native_definition), ``arms`` (id, samples,
populations), and explicit ``comparisons``. Each population supplies id, seed,
directory and hash-bound ``manifest``, ``summary``, ``algebra`` and ``labels``
receipt references. Every reference is {path, sha256}; paths are absolute.

The new label receipt contract is ``native-class-independent-physical-labels-v1``:
complete/passed, seed, samples, manifest_sha256, samples_sha256, attempts_sha256,
target_and_regions_sha256, labels:{path,sha256}, source_sha256, input_sha256,
runtime, physical_hard_validity_scope='every_attempt_capture_and_atomic', and
physical_native_contact_scope='every_contributing_pose'. It attests independent
physical observations; this adapter only verifies its bindings and projections.
The separate physical-label producer must be validated and independently run
before production use. Source hashes are provenance, not proof of execution/correctness.

Each label line carries draw, sample_record_sha256 (including newline), hard_valid
(capture AND strict atomic validity), applicable, native, exclusion_contact,
old_r5_radius, old_capture_valid, classification, contact. The last six fields
are null on noncontributing attempts. Applicable labels preserve full observer
objects; native/contact projections must agree with them. The old-R5 partition
is radius <= 5 AND original q > 1 AND old capture, computed here.

All attempted denominators survive, and each producer's validator verifies J/q and
the LINEAR two-cloud mean. No saved guide interval is a physical label. All-row
algebra and labels still do not discharge selected full-interval geometry,
spatial-Poisson/envelope, unseen-mass, full-vessel or assembly obligations. This
adapter always leaves those production gates closed. No real inputs are needed
for its tests, and no atom/tree/observer constructor is called.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import contact_population_statistics as population
import native_class_line_physical_algebra_audit as streaming
import native_class_line_weight_row as weights
from analyze_conditional_ray_campaign import stratify
from analyze_contact_bank_reference_partition import validate_regions

require, close = streaming.require, streaming.close
SCHEMA = 'native-class-physical-population-analysis-v1'
PLAN_SCHEMA = 'native-class-physical-population-analysis-plan-v1'
LABEL_SCHEMA = 'native-class-independent-physical-labels-v1'
ORIGINAL_REGION_SHA = '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
ORIGINAL_SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
REGIONS = ('total', 'native', 'competing', 'unbound', 'native_old_r5', 'native_remainder')
DECISIONS = ('native', 'competing', 'native_old_r5', 'native_remainder')
STRATA = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
              latent_orthants='six signs, zero assigned positive; all 64 bins retained')
BINS = dict(radial=3, angular=3, orthant=64)
GATES = dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
             combined_linear_SE_max=3., log_agreement_absolute_max=.2,
             deltaF_95_halfwidth_max=.5, significant_stratum_mass_fraction=.01)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class Bindings:
    def __init__(self): self.files = {}

    def bind(self, path, digest=None):
        path = Path(path)
        require(path.is_absolute(), 'Bound paths must be absolute')
        path = path.resolve(); actual = streaming.sha(path)
        if digest is not None:
            streaming._digest(digest, str(path))
            require(actual == digest, 'Changed bound file '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual, 'Changed input during analysis')
        self.files[str(path)] = actual
        return path

    def reference(self, value):
        require(set(value) == {'path', 'sha256'}, 'A reference must bind path and SHA256')
        return self.bind(value['path'], value['sha256'])

    def recheck(self):
        for path, digest in self.files.items():
            require(streaming.sha(path) == digest, 'Bound input changed during analysis '+path)


def target_identity(target, bindings):
    """Authenticate the original region and exact classifier/R5 file identities."""
    require(set(target) == {'region', 'old_r5_region', 'native_definition'}, 'Incomplete target references')
    paths = {name: bindings.reference(ref) for name, ref in target.items()}
    region, old = (streaming.read(paths[name]) for name in ('region', 'old_r5_region'))
    require(streaming.sha(paths['region']) == ORIGINAL_REGION_SHA
            and region['shape_sha256'] == ORIGINAL_SHAPE_SHA, 'Original R4 target/domain changed')
    require(region['activity'] == .035 and region['depletant_radius'] == 1.5
            and region['mahalanobis_radius'] == 4. and region.get('minimum_mahalanobis_radius', 0.) == 0.,
            'Original bath or full R4 changed')
    validate_regions(region, old)
    descriptor = dict(region_sha256=streaming.sha(paths['region']), shape_sha256=region['shape_sha256'],
        old_r5_region_sha256=streaming.sha(paths['old_r5_region']),
        native_definition_sha256=streaming.sha(paths['native_definition']),
        proposal_measure=weights.MEASURE, physical_measure='translation volume times normalized Haar orientation',
        activity=region['activity'], depletant_radius=region['depletant_radius'],
        regions=list(REGIONS), primary_partition=['native', 'competing', 'unbound'],
        native_partition=['native_old_r5', 'native_remainder'],
        native_old_r5_predicate='old_r5_radius <= 5 AND original_q > 1 AND old_capture_valid AND native',
        competing_predicate='exclusion_contact AND NOT native',
        strata=STRATA, strata_sha256=fingerprint(STRATA),
        unconditional_denominator='Every originally declared attempted draw, including all zeros')
    return region, descriptor, fingerprint(descriptor)


class Moment:
    """Sparse positive observations, with an explicit *unconditional* final N.

    Zeros are omitted from sums only, never from N. No row/trace arrays or top-N
    heaps are retained. A maximum location is deterministic under lowest draw ID.
    """
    def __init__(self):
        self.offset = -math.inf; self.sum1 = self.sum2 = 0.; self.nonzero = 0
        self.maximum = None

    def add(self, log_weight, location):
        require(math.isfinite(log_weight), 'Nonfinite positive log weight')
        self.nonzero += 1
        if log_weight > self.offset:
            scale = math.exp(self.offset-log_weight)
            self.sum1 *= scale; self.sum2 *= scale*scale
            self.offset = log_weight; self.maximum = location
        x = math.exp(log_weight-self.offset)
        self.sum1 += x; self.sum2 += x*x

    def merge(self, other):
        if not other.nonzero: return
        if other.offset > self.offset:
            scale = math.exp(self.offset-other.offset)
            self.sum1 *= scale; self.sum2 *= scale*scale
            self.offset = other.offset; self.maximum = other.maximum
        x = math.exp(other.offset-self.offset)
        self.sum1 += other.sum1*x; self.sum2 += other.sum2*x*x
        self.nonzero += other.nonzero

    def result(self, draws):
        require(type(draws) is int and draws >= self.nonzero and draws > 0, 'Invalid unconditional denominator')
        if not self.nonzero:
            return dict(draws=draws, nonzero=0, logQ=None, ess=0., max_fraction=None,
                        relative_se=None, maximum=None, observed=False,
                        limitation='No observations: neither physical zero nor an upper bound.')
        ess = self.sum1*self.sum1/self.sum2
        return dict(draws=draws, nonzero=self.nonzero, logQ=self.offset+math.log(self.sum1/draws),
                    ess=ess, max_fraction=1./self.sum1, maximum=self.maximum, observed=True,
                    relative_se=math.sqrt(max(0., (draws/ess-1)/(draws-1))) if draws > 1 else None)


class PairMoment:
    """Scaled central moments of positive pairs; zeros enter final unconditional N."""
    def __init__(self):
        self.offset = -math.inf; self.nonzero = 0
        self.mean = self.m2 = self.difference2 = self.product = 0.

    def rescale(self, offset):
        if offset > self.offset:
            factor = math.exp(self.offset-offset)
            self.mean *= factor; self.m2 *= factor*factor
            self.difference2 *= factor*factor; self.product *= factor*factor
            self.offset = offset

    def add(self, pairs):
        require(len(pairs) == 2 and all(math.isfinite(v) for v in pairs), 'Finite weighted cloud pair required')
        self.rescale(max(pairs))
        a, b = (math.exp(v-self.offset) for v in pairs)
        y = (a+b)/2; self.nonzero += 1
        delta = y-self.mean; self.mean += delta/self.nonzero; self.m2 += delta*(y-self.mean)
        # expm1 avoids cancellation for nearly equal clouds on a large log scale.
        high, low = max(pairs), min(pairs)
        difference = math.exp(high-self.offset)*(-math.expm1(low-high))
        self.difference2 += difference*difference; self.product += a*b

    def merge(self, other):
        if not other.nonzero: return
        self.rescale(other.offset); factor = math.exp(other.offset-self.offset)
        n = self.nonzero+other.nonzero; delta = other.mean*factor-self.mean
        self.m2 += other.m2*factor*factor + delta*delta*self.nonzero*other.nonzero/n
        self.mean += delta*other.nonzero/n
        self.difference2 += other.difference2*factor*factor
        self.product += other.product*factor*factor; self.nonzero = n

    def result(self, draws):
        require(draws >= self.nonzero, 'Pair denominator lost attempts')
        if draws < 2 or not self.nonzero: return None
        mean = self.mean*self.nonzero/draws
        # Merge N-positive_count exact zero observations using central moments.
        total = (self.m2+self.mean*self.mean*self.nonzero*(draws-self.nonzero)/draws)/(draws-1)
        cloud = self.difference2/(4*draws); residual = total-cloud
        return dict(unconditional_denominator=draws, log_weight_offset=self.offset,
            scaled_total_variance=total, scaled_paired_cloud_variance=cloud,
            scaled_residual_pose_variance=residual, paired_cloud_variance_fraction=cloud/total if total > 0 else None,
            relative_variance_of_mean_cloud=cloud/draws/mean**2,
            relative_variance_of_mean_pose=residual/draws/mean**2,
            scaled_mean_W1W2=self.product/draws,
            scaled_mean_squared_cloud_difference=self.difference2/draws,
            scope='J/q weighted independent-cloud arithmetic, all attempted zeros retained; negative residual pose variance is not clipped; independence is an external obligation.')


def keys_for(row, selected, region):
    """Use the existing original-chart stratum function, including seam rules."""
    bins = {name: -1 for name in BINS}
    if row['shell_valid']:
        bins = {name: int(values[0]) for name, values in stratify([row['latent']], region, STRATA).items()}
    keys = list(selected)
    for family, index in bins.items():
        if index >= 0:
            keys.extend(f'{family}:{index}:{name}' for name in selected)
    return keys, bins


def all_keys():
    return list(REGIONS) + [f'{family}:{index}:{name}' for family, count in BINS.items()
                           for index in range(count) for name in REGIONS]


def label_regions(label, row, accounting, sample_hash):
    require(type(label['draw']) is int and label['draw'] == row['draw']
            and label['sample_record_sha256'] == sample_hash, 'Label/sample row identity differs')
    for key in ('hard_valid', 'applicable'):
        require(type(label[key]) is bool, 'Missing Boolean label '+key)
    require(label['hard_valid'] == row['hard_valid'], 'Independent physical hard flag differs')
    valid = math.isfinite(accounting['z'])
    require(label['applicable'] == valid, 'Label applicability differs from unconditional weight support')
    fields = ('native', 'exclusion_contact', 'old_r5_radius', 'old_capture_valid', 'classification', 'contact')
    if not valid:
        require(all(label[key] is None for key in fields), 'Invalid/exterior zero acquired physical labels')
        return []
    for key in ('native', 'exclusion_contact', 'old_capture_valid'):
        require(type(label[key]) is bool, 'Missing Boolean physical label '+key)
    radius = weights._finite(label['old_r5_radius'], 'old R5 radius', 0.)
    classification, contact = label['classification'], label['contact']
    require(type(classification) is dict and type(contact) is dict, 'Complete observer objects required')
    require(type(classification['native_any']) is bool and classification['native_any'] == label['native']
            and type(contact['exclusion_contact']) is bool
            and contact['exclusion_contact'] == label['exclusion_contact'], 'Observer projection differs')
    anchors = classification['native_anchor_count']; triangle = classification['registry_consistent_triangle']
    require(type(anchors) is int and 0 <= anchors <= 2 and label['native'] == (anchors > 0)
            and type(triangle) is bool and (not triangle or anchors == 2), 'Native observer consistency differs')
    if label['native']:
        old = radius <= 5. and row['q'] > 1. and label['old_capture_valid']
        return ['total', 'native', 'native_old_r5' if old else 'native_remainder']
    return ['total', 'competing' if label['exclusion_contact'] else 'unbound']


def check_receipt(receipt, root, manifest, summary, target_id, kind, bindings, descriptor):
    require(receipt['complete'] is True and receipt['passed'] is True, 'Incomplete '+kind+' receipt')
    require(receipt.get('source_schema', streaming.full.SCHEMA) == manifest['schema'],
            'Receipt producer schema differs')
    require(type(receipt['samples']) is int and receipt['samples'] == manifest['samples']
            and type(receipt['seed']) is int and receipt['seed'] == manifest['seed'], 'Receipt allocation differs')
    for key in ('samples_sha256', 'attempts_sha256'):
        require(receipt[key] == summary[key], 'Receipt consumed a different stream')
    require(type(receipt['input_sha256']) is dict and receipt['input_sha256'], 'Missing receipt input closure')
    for path, digest in receipt['input_sha256'].items(): bindings.bind(path, digest)
    for path in (root/'manifest.json', root/'summary.json', root/'samples.jsonl', root/'attempts.jsonl',
                 root/'provenance/region.json'):
        require(receipt['input_sha256'].get(str(path)) == bindings.files[str(path)],
                'Receipt lacks population input binding '+str(path))
    require(type(receipt['source_sha256']) is dict and receipt['source_sha256'], 'Missing receipt source closure')
    for path, digest in receipt['source_sha256'].items(): bindings.bind(path, digest)
    require(type(receipt['runtime']) is dict and receipt['runtime'], 'Missing receipt runtime identity')
    if kind == 'algebra':
        require(receipt['schema'] == streaming.SCHEMA and receipt['all_rows_algebra'] == manifest['samples']
                and receipt['geometry_certified'] is False and receipt['physical_contact_labels_certified'] is False,
                'Wrong all-row algebra trust boundary')
        require(Path(receipt['root']).resolve() == root, 'Algebra population root differs')
    else:
        require(receipt['schema'] == LABEL_SCHEMA and receipt['target_and_regions_sha256'] == target_id
                and receipt['manifest_sha256'] == bindings.files[str(root/'manifest.json')], 'Label target/manifest differs')
        require(receipt['physical_hard_validity_scope'] == 'every_attempt_capture_and_atomic'
                and receipt['physical_native_contact_scope'] == 'every_contributing_pose',
                'Separate independent physical-label certification is required')
        for key, expected in [('region_sha256', descriptor['region_sha256']),
                              ('reference_region_sha256', descriptor['old_r5_region_sha256']),
                              ('strata_sha256', descriptor['strata_sha256']),
                              ('definition_sha256', descriptor['native_definition_sha256']),
                              ('shape_sha256', descriptor['shape_sha256'])]:
            require(receipt[key] == expected, 'Independent label scope differs '+key)
        if manifest['schema'] == streaming.hard_input.SCHEMA:
            require(receipt.get('native_identity_origin') == 'external_analysis_plan',
                    'v6 physical labels need independently bound analysis-native provenance')
            reference = receipt['compiled_native_binding']
            path = bindings.reference(reference)
            require(receipt['input_sha256'].get(str(path)) == reference['sha256'],
                    'External native geometry was not bound by the endpoint pass')
            compiled = streaming.read(path)
            require(compiled['source_definition_sha256'] == descriptor['native_definition_sha256']
                    and compiled['source_input_sha256']['tetramer-shape.json'] == descriptor['shape_sha256'],
                    'External native geometry targets different physical labels')
            report_ref = receipt.get('shape_compatibility_binding')
            require(type(report_ref) is dict, 'Missing external shape-compatibility binding')
            report_path = bindings.reference(report_ref)
            require(receipt['input_sha256'].get(str(report_path)) == report_ref['sha256'],
                    'External shape witness was not bound by the endpoint pass')
            report = streaming.read(report_path)
            require(report['compiled_sha256'] == reference['sha256']
                    and report['expected_shape_sha256'] == descriptor['shape_sha256']
                    and report['compatible'] is True, 'External shape witness identity differs')
            config_path = root/'provenance/input-config.json'
            config = streaming.read(bindings.bind(config_path, manifest['config_sha256']))
            require(compiled['fixed_poses'] == config['fixed_poses'] == manifest['physical_fixed_neighbors'],
                    'External native observer scaffold differs')


def consume_population(slot, samples, region, descriptor, target_id, bindings):
    root = Path(slot['directory']).resolve()
    manifest_path, summary_path = (bindings.reference(slot[key]) for key in ('manifest', 'summary'))
    require(manifest_path == root/'manifest.json' and summary_path == root/'summary.json', 'Population file identity differs')
    manifest, summary = streaming.read(manifest_path), streaming.read(summary_path)
    require(summary['complete'] is True and summary['manifest'] == manifest
            and not (root/'failure.json').exists(), 'Incomplete physical population')
    require(type(manifest['samples']) is int and manifest['samples'] == samples
            and type(slot['seed']) is int and manifest['seed'] == slot['seed'], 'Population allocation differs')
    require(manifest['region_sha256'] == descriptor['region_sha256']
            and manifest['shape_sha256'] == descriptor['shape_sha256'],
            'Population target/classifier identity differs')
    streaming.validate_manifest(manifest, region)
    if manifest['schema'] == streaming.full.SCHEMA:
        require(manifest['compiled_native']['source_definition_sha256'] == descriptor['native_definition_sha256'],
                'Producer native definition differs')
    for filename, key in [('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                          ('input-config.json', 'config_sha256'), ('importance-guide.json', 'importance_guide_sha256'),
                          ('source-bundle.json', 'source_bundle_sha256')]:
        bindings.bind(root/'provenance'/filename, manifest[key])
    require(streaming.read(root/'provenance/region.json') == region, 'Population domain differs')
    for filename, key in [('samples.jsonl', 'samples_sha256'), ('attempts.jsonl', 'attempts_sha256')]:
        bindings.bind(root/filename, summary[key])
    algebra, labels = (streaming.read(bindings.reference(slot[key])) for key in ('algebra', 'labels'))
    check_receipt(algebra, root, manifest, summary, target_id, 'algebra', bindings, descriptor)
    check_receipt(labels, root, manifest, summary, target_id, 'labels', bindings, descriptor)
    label_path = bindings.reference(labels['labels'])
    moments = {kind: {key: Moment() for key in all_keys()} for kind in ('Qz', 'Q0')}
    moments['pairs'] = {key: PairMoment() for key in all_keys()}
    counters, branches, bin_counts = Counter(), Counter(), Counter()
    anomalies = 0
    streams = [streaming.JsonLines(root/name) for name in ('samples.jsonl', 'attempts.jsonl')]
    streams.append(streaming.JsonLines(label_path))
    try:
        for index in range(samples):
            row, attempt, label = (stream.next() for stream in streams)
            require(row is not None and attempt is not None and label is not None, 'Missing unconditional stream row')
            require(type(attempt.get('draw')) is int and attempt == dict(draw=index, state='begin'),
                    'Missing/reordered attempted draw journal')
            value = streaming.validate_row(row, expected_draw=index, manifest=manifest, region=region)
            selected = label_regions(label, row, value, streams[0].last['sha256'])
            anomalies += int(label['applicable'] and label['native'] and not label['exclusion_contact'])
            keys, bins = keys_for(row, selected, region)
            location = dict(population=slot['id'], draw=index)
            for kind, log_weight in (('Qz', value['z']), ('Q0', value['h'])):
                for key in keys: moments[kind][key].add(log_weight, location)
            for key in keys: moments['pairs'][key].add(value['pairs'])
            counters.update(value['counters']); branches[row['proposal_branch']] += 1
            for family, bin_id in bins.items(): bin_counts[f'{family}:{bin_id}'] += 1
        for stream, expected in zip(streams, [summary['samples_sha256'], summary['attempts_sha256'], labels['labels']['sha256']]):
            require(stream.next() is None and stream.lines == samples and stream.hash.hexdigest() == expected,
                    'Extra/reordered/changed unconditional stream')
    finally:
        for stream in streams: stream.close()
    require(summary['samples'] == samples and algebra['counts'] == dict(counters), 'Audit denominator/counters differ')
    for kind, key in [('Qz', 'estimate'), ('Q0', 'hard_region')]:
        observed, audited = moments[kind]['total'].result(samples), algebra[key]
        require(observed['nonzero'] == audited['nonzero'] and audited['draws'] == samples, 'Audit total differs')
        require((observed['logQ'] is None) == (audited['logQ'] is None), 'Audit zero mass differs')
        if observed['logQ'] is not None: close(observed['logQ'], audited['logQ'], 'Audit linear mean differs')
    cpu = weights._finite(summary['sampler_cpu_seconds'], 'sampler CPU', 0.)
    return moments, dict(id=slot['id'], source_schema=manifest['schema'], seed=slot['seed'], draws=samples,
        unconditional_denominator=samples, counters=dict(counters), branches=dict(branches),
        stratum_attempts=dict(bin_counts), sampler_cpu_seconds=cpu,
        native_entry_unbound_anomalies=anomalies, classifier_contact_consistency_passed=anomalies == 0,
        algebra_audit_cpu_seconds=algebra.get('analysis_cpu_seconds'),
        physical_labels_cpu_seconds=labels.get('analysis_cpu_seconds'),
        algebra_receipt_sha256=slot['algebra']['sha256'], label_receipt_sha256=slot['labels']['sha256'])


def quality(row, estimate):
    rse = estimate['population_relative_SE']
    return dict(passed=row['observed'] and row['ess'] >= 200 and row['max_fraction'] <= .02
                and rse is not None and rse <= .1,
                importance_ESS_passed=row['ess'] >= 200,
                largest_contribution_passed=row['max_fraction'] is not None and row['max_fraction'] <= .02,
                population_RSE_passed=rse is not None and rse <= .1)


def statistics_block(arm, records, moments, samples, target_id, prefix='', pairs=None):
    block_id = fingerprint(dict(target_and_regions_sha256=target_id, stratum=prefix)) if prefix else target_id
    declaration = dict(arm_id=arm, population_count=len(records), draws_per_population=samples,
        total_unconditional_draws=len(records)*samples, target_and_regions_sha256=block_id,
        regions=list(REGIONS), populations=[dict(id=r['id'], seed=r['seed']) for r in records])
    columns = [{**{k: r[k] for k in ('id', 'seed', 'draws', 'unconditional_denominator')},
                'log_masses': {name: m[prefix+name].result(samples)['logQ'] for name in REGIONS}}
               for r, m in zip(records, moments)]
    summarized = population.summarize_populations(declaration, columns)
    rows = {}
    for name in REGIONS:
        combined = Moment()
        for m in moments: combined.merge(m[prefix+name])
        rows[name] = combined.result(samples*len(records))
        close_optional = summarized['estimates'][name]['log_linear_mean']
        if rows[name]['logQ'] is not None: close(rows[name]['logQ'], close_optional, 'Pooled/population means differ')
    for parent, children in [('total', ('native', 'competing', 'unbound')),
                             ('native', ('native_old_r5', 'native_remainder'))]:
        logs = [rows[name]['logQ'] for name in children if rows[name]['logQ'] is not None]
        require(bool(logs) == (rows[parent]['logQ'] is not None), 'Partition zero mass differs')
        if logs:
            offset = max(logs); value = offset+math.log(math.fsum(math.exp(v-offset) for v in logs))
            close(value, rows[parent]['logQ'], 'Exhaustive physical partition differs')
    noise = {}
    if pairs is not None:
        for name in REGIONS:
            combined = PairMoment()
            for values in pairs: combined.merge(values[prefix+name])
            noise[name] = combined.result(samples*len(records))
    return dict(declaration=declaration, parent_target_and_regions_sha256=target_id, stratum=prefix or None,
                population_records=columns, population_statistics=summarized, paired_noise=noise,
                row_diagnostics=rows,
                quality={name: quality(rows[name], summarized['estimates'][name]) for name in DECISIONS})


def compare_blocks(left, right):
    return {name: population.compare_population_masses(left['declaration'], left['population_records'],
                right['declaration'], right['population_records'], name) for name in REGIONS}


def compare_free_energy(left, right):
    """Compare the paired-population log ratio itself across independent arms.

    Matching marginal masses can conceal opposite shifts of a much more precise
    ratio. Each within-arm SE already includes native/competing covariance.
    """
    a_decl, b_decl = left['declaration'], right['declaration']
    require(a_decl['target_and_regions_sha256'] == b_decl['target_and_regions_sha256'],
            'Free-energy target or stratum differs')
    require({p['seed'] for p in a_decl['populations']}.isdisjoint(p['seed'] for p in b_decl['populations']),
            'Free-energy comparison requires independent population seeds')
    a = left['population_statistics']['free_energy_contrast']
    b = right['population_statistics']['free_energy_contrast']
    require((a['native_region'], a['competing_region']) == (b['native_region'], b['competing_region']),
            'Free-energy regional definitions differ')
    result = dict(observed=False, passed=False, left_arm=a_decl['arm_id'], right_arm=b_decl['arm_id'],
        target_and_regions_sha256=a_decl['target_and_regions_sha256'],
        beta_deltaF_left_minus_right=None, combined_population_SE=None,
        SE_passed=None, absolute_passed=None, SE_multiplier=3., absolute_limit=.2,
        reason='Native or competing mass is unobserved in an arm; no finite contrast or upper bound.')
    if not (a['observed'] and b['observed']): return result
    delta = a['beta_F_native_minus_competing']-b['beta_F_native_minus_competing']
    se = math.hypot(a['population_SE'], b['population_SE'])
    require(math.isfinite(delta) and math.isfinite(se), 'Nonfinite free-energy comparison')
    tolerance = 32*sys.float_info.epsilon*max(1., abs(a['beta_F_native_minus_competing']),
                                           abs(b['beta_F_native_minus_competing']))
    statistical, absolute = abs(delta) <= 3*se+tolerance, abs(delta) <= .2
    result.update(observed=True, passed=statistical and absolute, SE_passed=statistical,
        absolute_passed=absolute, beta_deltaF_left_minus_right=delta, combined_population_SE=se,
        comparison_roundoff_tolerance=tolerance, reason=None,
        within_arm_uncertainty='Paired native/competing covariance from independent population linear means; cross-arm SEs combined in quadrature.')
    return result


def analyze(plan_path):
    """Read completed streams only; return a report without writing or observing geometry."""
    started = time.process_time(); bindings = Bindings()
    source_files = {}
    names = set(streaming.SOURCE_NAMES) | {'contact_population_statistics.py', Path(__file__).name,
        'analyze_conditional_ray_campaign.py', 'analyze_contact_bank_reference_partition.py'}
    for name in sorted(names):
        path = Path(__file__).resolve() if name == Path(__file__).name else Path(sys.modules[Path(name).stem].__file__).resolve()
        require(path == Path(__file__).resolve().parent/name, 'Imported implementation outside source closure')
        source_files[str(path)] = streaming.sha(bindings.bind(path))
    runtime = streaming.runtime_identity()
    for path, digest in runtime['file_sha256'].items(): bindings.bind(path, digest)
    plan_path = bindings.bind(Path(plan_path).resolve()); plan = streaming.read(plan_path)
    require(plan['schema'] == PLAN_SCHEMA and plan['strata'] == STRATA and plan['gates'] == GATES,
            'Wrong plan or changed fixed strata/gates')
    region, descriptor, target_id = target_identity(plan['target'], bindings)
    require(plan['target_and_regions_sha256'] == target_id, 'Target/regions fingerprint differs')
    require(plan['arms'] and len({a['id'] for a in plan['arms']}) == len(plan['arms']), 'Missing/duplicate arms')
    all_seeds = [p['seed'] for a in plan['arms'] for p in a['populations']]
    require(len(all_seeds) == len(set(all_seeds)), 'Reused independent population seed')
    roots = [str(Path(p['directory']).resolve()) for a in plan['arms'] for p in a['populations']]
    require(len(roots) == len(set(roots)), 'Repeated physical population')
    arms = {}
    for arm in plan['arms']:
        slots = arm['populations']; samples = weights._integer(arm['samples'], 'samples', 1)
        require(len(slots) >= 4 and len({p['id'] for p in slots}) == len(slots), 'At least four distinct populations required')
        require([p['id'] for p in slots] == sorted(p['id'] for p in slots), 'Population order must be canonical')
        values = [consume_population(slot, samples, region, descriptor, target_id, bindings) for slot in slots]
        records = [item[1] for item in values]
        blocks = {}
        for kind in ('Qz', 'Q0'):
            moments = [item[0][kind] for item in values]
            pairs = [item[0]['pairs'] for item in values] if kind == 'Qz' else None
            primary = statistics_block(arm['id'], records, moments, samples, target_id, pairs=pairs)
            strata = {family: [statistics_block(arm['id'], records, moments, samples, target_id,
                                f'{family}:{index}:', pairs=pairs) for index in range(count)] for family, count in BINS.items()}
            for family, bins in strata.items():
                for entry in bins:
                    fractions = {}
                    for name in REGIONS:
                        part, whole = entry['row_diagnostics'][name]['logQ'], primary['row_diagnostics'][name]['logQ']
                        fractions[name] = None if whole is None else 0. if part is None else math.exp(part-whole)
                    entry['observed_parent_mass_fraction'] = fractions
            blocks[kind] = dict(primary=primary, strata=strata)
        arms[arm['id']] = dict(populations=records, estimates=blocks,
            sampler_cpu_seconds=math.fsum(r['sampler_cpu_seconds'] for r in records),
            classifier_contact_consistency_passed=all(r['classifier_contact_consistency_passed'] for r in records))
        for name, row in blocks['Qz']['primary']['row_diagnostics'].items():
            row['importance_ESS_per_sampler_CPU_second'] = row['ess']/arms[arm['id']]['sampler_cpu_seconds'] if arms[arm['id']]['sampler_cpu_seconds'] else None
        contrast = blocks['Qz']['primary']['population_statistics']['free_energy_contrast']
        arms[arm['id']]['free_energy_precision_passed'] = contrast['observed'] and contrast['halfwidth_95'] <= .5
    comparisons = []
    seen = set()
    for comparison in plan['comparisons']:
        left, right = comparison['left'], comparison['right']
        require(left in arms and right in arms and left != right and (left, right) not in seen, 'Invalid/repeated comparison')
        seen.add((left, right))
        require(type(comparison['historical_failed_strata']) is list, 'Historical failure inventory required')
        historical = set()
        for item in comparison['historical_failed_strata']:
            family, index, name = item['family'], item['bin'], item['region']
            require(family in BINS and type(index) is int and 0 <= index < BINS[family] and name in DECISIONS,
                    'Invalid historical failed stratum')
            historical.add((family, index, name))
        result = dict(left=left, right=right, kinds={})
        for kind in ('Qz', 'Q0'):
            a, b = arms[left]['estimates'][kind], arms[right]['estimates'][kind]
            strata = []
            for family, count in BINS.items():
                for index in range(count):
                    x, y = a['strata'][family][index], b['strata'][family][index]
                    compared = compare_blocks(x, y)
                    for name in DECISIONS:
                        material = max(x['observed_parent_mass_fraction'][name] or 0.,
                                       y['observed_parent_mass_fraction'][name] or 0.) >= .01
                        strata.append(dict(family=family, bin=index, region=name, material=material,
                            historical_failure=(family, index, name) in historical, comparison=compared[name]))
            result['kinds'][kind] = dict(regions=compare_blocks(a['primary'], b['primary']), strata=strata,
                free_energy_contrast=compare_free_energy(a['primary'], b['primary']))
        comparisons.append(result)
    bindings.recheck()
    return dict(schema=SCHEMA, complete=True, plan_sha256=streaming.sha(plan_path),
        target_and_regions_sha256=target_id, target_descriptor=descriptor, arms=arms, comparisons=comparisons,
        gates=GATES, input_sha256=bindings.files, source_sha256=source_files,
        runtime=runtime, analysis_cpu_seconds=time.process_time()-started,
        new_pose_draws=0, new_Poisson_clouds=0, new_atom_geometry_queries=0, new_native_classifier_calls=0,
        label_evidence='Authenticated independent label receipt attestations and exact per-sample bindings; physical labels are not recomputed here.',
        independent_geometry_rows_reconstructed_here=0, selected_full_geometry_gate_satisfied=False,
        physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        remaining_obligations=['Validate and bind the separate exact physical-label producer.',
            'Bind and complete the separately declared selected full-interval geometry audit.',
            'Retain spatial-Poisson/envelope/RNG implementation evidence.',
            'Resolve all material strata, historical discrepancies, population-size/intensity sensitivities and unseen-contact remainder.'],
        scope='Explicit v6/v7 arithmetic adapter; all attempted denominators and stages remain separate. No full physical convergence or assembly certification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Output already exists')
    # No scientific observation occurs; failed arithmetic writes no success file.
    result = analyze(args.plan)
    streaming._write(args.out, result)


if __name__ == '__main__': main()
