#!/usr/bin/env python3
"""Compare saved full atomic-wall contact integrals under separate proposal laws.

This is downstream aggregation and stateless classification only. It never runs
an atomic/cloud/density audit or a physical sampler. Every attempted draw stays
in its original denominator, including hard/wall invalid zeros.
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

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_mobile_native_pocket import (
    require, read, sha, write, inside, load_classifier, local_sources,
    summarize, label_keys, paired_noise)
from analyze_mobile_competing_reference import paired_moments, contrast
from analyze_mobile_full_capture import chart_radii, relative_covariance, paired_ratio

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('original', 'coverage')
PRIMARY = ('registered_native_entry', 'contact_no_native_entry', 'unbound_no_native_entry')
WALL = dict(center=[0., 0., 0.], radius=223.32617672378387)
SAMPLES = 16384
SEEDS = [126101010+1009*i for i in range(8)]
REFERENCE_SHA = 'e252600e2498b1fc437406b417d7b1053089833c10a2cd23202a6d5cbbcc0aa1'
SCOPE = ('Full original atomic-wall conditional integrals for one exact fixed scaffold and an unbounded ideal bath. '
    'The three primary classes are exhaustive on valid poses: registered native entry; exclusion contact with no native entry; '
    'and unbound with no native entry. No-entry is a threshold/catalogue result, not proof of nonnative structure. '
    'The original q<=1 metric and finite chart witnesses are separate historical observables. '
    'The original and coverage proposal laws are never pooled. Full support makes the target complete but does not establish '
    'numerical tail coverage, mobile equilibrium, physical kinetics, or stationary sampling efficiency.')


def partition_masks(valid, contact, native, anchors, triangle, d170, q, native_radius, alternative_radius):
    valid = np.asarray(valid, bool)
    vectors = [np.asarray(x) for x in (contact, native, anchors, triangle, d170, q, native_radius, alternative_radius)]
    require(valid.ndim == 1 and all(v.shape == valid.shape for v in vectors), 'Partition vector sizes differ')
    contact, native, anchors, triangle, d170, q, rn, ra = vectors
    contact, native, triangle, d170 = [x.astype(bool) for x in (contact, native, triangle, d170)]
    require(np.isfinite(q[valid]).all() and not np.isnan(rn[valid]).any() and not np.isnan(ra[valid]).any(), 'Invalid valid-pose region coordinates')
    require(np.all(np.isin(anchors, [0, 1, 2])) and np.array_equal(native[valid], anchors[valid] > 0), 'Native anchor labels disagree')
    require(not np.any(triangle & valid & (anchors != 2)), 'A registry triangle requires both anchors')
    masks = dict(total=valid, registered_native_entry=valid & native,
        contact_no_native_entry=valid & contact & ~native,
        unbound_no_native_entry=valid & ~contact & ~native,
        exclusion_contact=valid & contact, unbound=valid & ~contact,
        native_one_anchor=valid & (anchors == 1), native_both_anchors=valid & (anchors == 2),
        registry_triangle=valid & triangle, native_entry_unbound=valid & native & ~contact,
        inside_d170=valid & d170, outside_d170=valid & ~d170,
        original_q_le_1=valid & (q <= 1.), original_q_gt_1=valid & (q > 1.),
        original_native_r4=valid & d170 & (q <= 1.) & (rn <= 4.),
        alternative_native_r5=valid & d170 & (q > 1.) & (ra <= 5.),
        alternative_native_r32=valid & d170 & (q > 1.) & (ra <= 32.))
    require(np.array_equal(sum(masks[key].astype(int) for key in PRIMARY), valid.astype(int)), 'Primary partition has a gap or overlap')
    for name in PRIMARY:
        masks[name+'_inside_d170'] = masks[name] & d170
        masks[name+'_outside_d170'] = masks[name] & ~d170
    return masks


def wall_target(config, shape, wall):
    require(wall == WALL, 'Original atomic wall changed')
    require(config['capture_center'] == [0., 0., 0.] and config['capture_radius'] == 273., 'Full-wall support capture changed')
    require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035 and len(config['fixed_poses']) == 2,
            'Fixed-scaffold bath target changed')
    bound = max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms'])
    required = math.dist(config['capture_center'], wall['center'])+wall['radius']+bound
    require(config['capture_radius'] >= required, 'Capture fails to enclose every wall-valid body center')
    require(170.+bound < wall['radius'], 'Historical finite witnesses lack an all-orientation atomic-wall certificate')
    return dict(shape_bound_A=float(bound), required_capture_radius_A=float(required), capture_radius_A=config['capture_radius'],
        atomic_wall=wall, bath_wall_permeable=True,
        scope='Conservative enclosure of all atomic-wall-valid body centers. The capture sphere is proposal support, not the physical wall.')


def reference_target(config, region, shape_sha, radius):
    require(region['shape_sha256'] == shape_sha, 'Reference hard shape differs')
    for a, b in [('physical_fixed_neighbors', 'fixed_poses'), ('physical_metric', 'metadata'),
                 ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius')]:
        require(region[a] == config[b], 'Reference physical target differs: '+a)
    require(region['capture_center'] == [0., 0., 0.] and region['capture_radius'] == 170., 'Unexpected historical capture')
    chart, fixed = region['gaussian_chart'], region['fixed_neighbor']
    require(len(chart['weights']) == 1 and 'base_model' not in chart, 'Finite witness requires one ordinary chart')
    rotation = Rotation.from_quat(np.asarray(fixed['orientation'])[[1, 2, 3, 0]]).as_matrix()
    center = np.asarray(fixed['position'])+rotation@(np.asarray(chart['anchors'][0]['position'])+np.asarray(chart['means'][0][:3]))
    upper = float(np.linalg.norm(center)+radius*np.linalg.norm(np.linalg.cholesky(chart['covariances'][0])[:3], ord=2))
    require(upper < 170., 'Finite witness extends beyond historical D170 target')
    return upper


def validate_parent(root):
    protocol, status = read(root/'protocol.json'), read(root/'status.json')
    require(protocol['schema'] == 'mobile-wall-contact-controller-v1', 'Unexpected wall campaign schema')
    require(status['complete'] and status['phase'] == 'complete' and not status.get('running', False), 'Wall campaign is incomplete')
    require(status['protocol_sha256'] == sha(root/'protocol.json'), 'Terminal protocol binding changed')
    files = read(root/'freeze.json')['files']
    require(files, 'Missing frozen input map')
    for name, digest in files.items():
        require(sha(inside(root, name)) == digest, 'Frozen wall input changed: '+name)
    require(len(protocol['campaigns']) == 2 and {c['arm'] for c in protocol['campaigns']} == set(ARMS), 'Missing or duplicate proposal arm')
    require(len(status['jobs']) == 8 and len({(j['arm'], j['id']) for j in status['jobs']}) == 8, 'Missing or repeated physical population')
    require(set(status['audits']) == set(ARMS), 'Missing completed raw audit')
    require(protocol['populations_per_arm'] == 4 and protocol['samples_per_population'] == SAMPLES
            and protocol['seed_base'] == SEEDS[0] and protocol['total_jobs'] == 8
            and protocol['total_unconditional_draws'] == 8*SAMPLES and protocol['cloud_replicates'] == 2
            and protocol['lambda_ratio'] == 64. and protocol['uniform_probability'] == .1, 'Frozen wall allocation changed')
    require(protocol['physical_wall'] == WALL and protocol['capture_radius'] == 273., 'Protocol wall domain changed')
    require(sha(root/'provenance/source-config.json') == protocol['original_config_sha256'], 'Original source config changed')
    for entry in protocol['campaigns']:
        folder = Path(entry['path']).resolve()
        require(folder == root/entry['arm'], 'Frozen arm path differs')
        manifest = read(folder/'manifest.json')
        require(manifest['archive_sha256']['basin-normalizer'] == protocol['physical_executable_sha256']
                and manifest['archive_sha256']['source-bundle.json'] == protocol['source_bundle_sha256'], 'Reviewed executable/source identity differs')
    require(Path(protocol['native_definition']) == Path('provenance/native-region/definition.json'), 'Unexpected native definition path')
    return protocol, status


def validate_jobs(jobs, terminal, arm):
    require(arm in ARMS and len(jobs) == len(terminal) == 4, 'Require four independent populations per arm')
    ids = {f'r{i:02d}' for i in range(4)}
    require({j['id'] for j in jobs} == ids and {j['id'] for j in terminal} == ids, 'Population IDs differ or repeat')
    for job in jobs:
        index = int(job['id'][1:])+4*ARMS.index(arm)
        require(job['seed'] == SEEDS[index] and job['samples'] == SAMPLES and job['covariance_std_scale'] == 1.,
                'Seed, fixed-N budget, or covariance scale changed')
        require(job.get('proposal_anchor_index') is None and job['cloud_replicates'] == 2, 'Proposal-anchor/cloud allocation changed')
        term = next(t for t in terminal if t['id'] == job['id'])
        require(term['status'] == 'complete' and term['returncode'] == 0 and
                all(term[k] == job[k] for k in ('id', 'seed', 'samples')), 'Population not successfully complete')


def validate_model_manifest(model, manifest):
    require(model.get('schema') == 'reciprocal-pose-mixture-v1' and set(model) == {'schema', 'base_model', 'reciprocal_components'}, 'Expected explicit reciprocal envelope')
    base, flags = model['base_model'], model['reciprocal_components']
    count = len(base['weights'])
    require(len(flags) == count and all(type(f) is bool for f in flags) and any(flags), 'Missing active reciprocal branch')
    require(manifest['schema'] == 4 and manifest['pose_proposal_schema'] == 3 and manifest['proposal_model_kind'] == model['schema'],
            'Expected full-wall schema4 with active reciprocal proposal schema3')
    require(manifest['base_component_count'] == count and manifest['virtual_component_count'] == count+sum(flags)
            and manifest['reciprocal_components'] == flags, 'Saved reciprocal branch semantics changed')
    require(manifest['atomic_wall'] == WALL and manifest['bath_wall_permeable'] is True, 'Wall or bath permeability changed')
    require(manifest['covariance_scale'] == 1. and manifest['uniform_probability'] == .1
            and manifest.get('proposal_anchor_index') is None and manifest['physical_fixed_neighbor_count'] == 2,
            'Whole-support proposal law or anchor marginalization changed')


def read_weights(rows, samples):
    require(len(rows) == samples and [r['draw'] for r in rows] == list(range(samples)), 'Unconditional draws missing or repeated')
    z, h, pairs = [], [], []
    for row in rows:
        require(row['pose'] is not None and all(type(row[k]) is bool for k in ('hard_valid', 'wall_valid', 'capture_valid')), 'Missing pose/support flag')
        valid = row['hard_valid']
        if not valid:
            require(row['log_importance_weight'] is None and row['log_hard_weight'] is None and
                    row['q'] is None and row['depletion_contact'] is None and not row['clouds'], 'Invalid poses must retain unconditional zeros')
            z.append(-math.inf);h.append(-math.inf);pairs.append([-math.inf, -math.inf]);continue
        require(row['wall_valid'] and row['capture_valid'] and type(row['depletion_contact']) is bool and math.isfinite(row['q']),
                'A contributing row is not valid in the full-wall domain')
        require(len(row['clouds']) == 2 and all(math.isfinite(row[k]) for k in ('log_importance_weight', 'log_hard_weight', 'log_proposal_density')),
                'Missing finite two-cloud weight')
        require(abs(row['log_hard_weight']+row['log_proposal_density']) < 2e-10, 'Hard-volume weight differs from full proposal reciprocal')
        p = [c['log_weight']-row['log_proposal_density'] for c in row['clouds']]
        require(all(math.isfinite(v) for v in p) and abs(float(logsumexp(p)-math.log(2))-row['log_importance_weight']) < 2e-10,
                'Saved physical weight differs from two-cloud mean')
        z.append(row['log_importance_weight']);h.append(row['log_hard_weight']);pairs.append(p)
    return dict(z=np.asarray(z), h=np.asarray(h), pairs=np.asarray(pairs))


def same_log(actual, expected):
    require((actual is None and expected is None) or (actual is not None and expected is not None and abs(actual-expected) < 2e-10),
            'Downstream integral differs from saved completed audit')


def load_references(root, protocol, config, shape_sha):
    mapping = protocol['references']
    path = inside(root, mapping['comparison'])
    require(sha(path) == mapping['comparison_sha256'] == REFERENCE_SHA, 'Independent finite reference identity changed')
    reference = read(path)
    require(reference['complete'] and reference['schema'] == 'mobile-native-pocket-coverage-comparison-v1', 'Incomplete independent finite reference')
    require(reference['shape_sha256'] == shape_sha, 'Independent finite reference shape differs')
    native = read(inside(root, mapping['native_r4']));alternative = read(inside(root, mapping['alternative_r5']))
    alternative32 = read(inside(root, mapping['alternative_r32']))
    normalized32 = copy.deepcopy(alternative32);normalized32['mahalanobis_radius'] = alternative['mahalanobis_radius']
    normalized32['definition'] = alternative.get('definition')
    require(alternative32['mahalanobis_radius'] == 32. and alternative32.get('minimum_mahalanobis_radius', 0.) == 0.
            and normalized32 == alternative, 'R32 witness differs from the same full-ball chart/target')
    require(native['mahalanobis_radius'] == 4. and native.get('maximum_original_q') == 1., 'Original native R4 target changed')
    require(alternative['mahalanobis_radius'] == 5. and alternative['minimum_original_q'] == 1.
            and alternative.get('minimum_original_q_inclusive', True) is False and 'maximum_original_q' not in alternative,
            'Alternative finite target changed')
    upper_native = reference_target(config, native, shape_sha, 4.)
    upper_alternative = reference_target(config, alternative, shape_sha, 32.)
    records = dict(original_native_r4=reference['original_native_r4'],
        alternative_native_r5=next(r for r in reference['strata'] if r['name'] == 'r5repeat'),
        alternative_native_r32=reference['coverage']['cumulative_r32'])
    require(sha(inside(root, mapping['native_r4'])) == records['original_native_r4']['region_sha256'], 'Original native witness bytes differ')
    require(sha(inside(root, mapping['alternative_r5'])) == records['alternative_native_r5']['region_sha256'], 'Alternative witness bytes differ')
    def seeds(value):
        if isinstance(value, dict):
            return ({value['seed']} if 'seed' in value else set()).union(*(seeds(v) for v in value.values()))
        if isinstance(value, list):return set().union(*(seeds(v) for v in value))
        return set()
    require(seeds(reference).isdisjoint(SEEDS), 'Wall and finite-reference seeds overlap')
    return dict(native=native, alternative=alternative, records=records,
        binding=dict(path=str(path), sha256=sha(path), region_sha256={k:sha(inside(root, mapping[k])) for k in ('native_r4', 'alternative_r5', 'alternative_r32')},
            maximum_native_R4_center_radius_A=upper_native, maximum_alternative_R32_center_radius_A=upper_alternative,
            scope='Archived completed independent finite-region estimates; no old audit or classifier replay.'))


def native_unbound_diagnostic(row, label, population):
    gaps = [bond['minimum_gap_A'] for match in label['matches'] for bond in match['supporting_member_bonds']
            if bond['minimum_gap_A'] is not None]
    return dict(population=population, draw=row['draw'], pose=row['pose'], q=row['q'], classification=label,
        supporting_entry_gaps_A=gaps, exclusion_contact=row['depletion_contact'],
        interpretation='Native entry requires a supporting gap<=2 Å; rd=1.5 makes exclusion contact at gap<3 Å. '
                       'An entry/unbound observation is retained in the native primary class and flagged as an inconsistency to investigate.')


def summarize_regions(populations):
    names = list(populations[0]['masks'])
    require(all(set(p['masks']) == set(names) for p in populations), 'Population diagnostic masks differ')
    estimates = {name:summarize(populations, name, [p['masks'][name] for p in populations]) for name in names}
    covariances = {}
    for level in ('row', 'population'):
        columns = {}
        for name in PRIMARY:
            for kind, field in [('Qz', 'z'), ('Q0', 'h')]:
                if level == 'row':
                    values = np.concatenate([np.where(p['masks'][name], p[field], -np.inf) for p in populations])
                else:
                    values = [v['log_'+kind] for v in estimates[name]['populations']]
                    values = [-np.inf if v is None else v for v in values]
                columns[name+':'+kind] = values
        covariances[level] = relative_covariance(columns)
    ratios = {}
    for numerator, denominator in [(PRIMARY[0], PRIMARY[1]), (PRIMARY[0], PRIMARY[2]), (PRIMARY[1], PRIMARY[2])]:
        item = {}
        for level in ('row', 'population'):
            columns = []
            for name in (numerator, denominator):
                if level == 'row':values = np.concatenate([np.where(p['masks'][name], p['z'], -np.inf) for p in populations])
                else:values = [-np.inf if v['log_Qz'] is None else v['log_Qz'] for v in estimates[name]['populations']]
                columns.append(values)
            item[level] = paired_ratio(*columns)
        ratios[numerator+'/'+denominator] = item
    return estimates, covariances, ratios


def load_arm(root, entry, status, classifier, references, output):
    arm = entry['arm'];folder = Path(entry['path']).resolve()
    require(folder == root/arm, 'Arm directory differs from frozen campaign')
    require(sha(folder/'manifest.json') == entry['manifest_sha256'], 'Arm manifest changed')
    manifest = read(folder/'manifest.json')
    require(manifest['schema'] == 1 and manifest['experiment_schema'] == 'mobile-wall-contact-arm-v1', 'Unexpected wall arm schema')
    for name, digest in manifest['archive_sha256'].items():
        require(sha(inside(folder/'provenance', name)) == digest, 'Archived arm input changed: '+name)
    require(manifest['archive_sha256']['model.json'] == entry['model_sha256'], 'Arm model identity changed')
    config = read(folder/'provenance/config.json');shape_sha = sha(folder/'provenance/shape.json')
    original_config = read(root/'provenance/source-config.json')
    expected_config = copy.deepcopy(original_config)
    expected_config.update(shape=str(folder/'provenance/shape.json'), capture_radius=273.)
    require(config == expected_config, 'Wall config changed beyond shape relocation and support expansion')
    require(classifier.fixed_poses == config['fixed_poses'] and classifier.shape_sha256 == shape_sha, 'Native classifier target differs')
    physical = manifest['physical']
    require(physical['atomic_wall'] == WALL and physical['bath_wall_permeable'] is True and physical['uniform_probability'] == .1
            and physical['lambda_ratio'] == 64., 'Physical wall/bath or proposal floor changed')
    for key, cfgkey in [('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius'), ('fixed_poses', 'fixed_poses'),
                        ('capture_center', 'capture_center'), ('capture_radius', 'capture_radius')]:
        require(physical[key] == config[cfgkey], 'Arm physical manifest differs from config')
    support = wall_target(config, read(folder/'provenance/shape.json'), physical['atomic_wall'])
    terminal = [j for j in status['jobs'] if j['arm'] == arm]
    validate_jobs(manifest['jobs'], terminal, arm)
    audit_record = status['audits'][arm]
    assessment_path = folder/'assessment/analysis.json'
    require(audit_record['returncode'] == 0 and sha(assessment_path) == audit_record['analysis_sha256'], 'Completed wall audit changed or failed')
    assessment = read(assessment_path)
    require(not assessment['pending'] and len(assessment['populations']) == 4
            and {p['job']['id'] for p in assessment['populations']} == {j['id'] for j in manifest['jobs']}, 'Saved audit omitted populations')
    require(assessment['provenance'][str(folder/'manifest.json')] == entry['manifest_sha256'], 'Audit manifest binding differs')
    require(assessment['provenance'][str(folder/'provenance/analyze_basin_normalizers.py')]
            == manifest['archive_sha256']['analyze_basin_normalizers.py'], 'Audit source binding differs')
    model = read(folder/'provenance/model.json')
    populations, files, anomalies, cpu = [], {}, [], 0.
    labels_path = output/f'{arm}-native-labels.jsonl'
    all_motif_keys = {f'anchor{a}_motif{m}' for a in (0, 1) for m in classifier.motif_by_id}
    with labels_path.open('x') as label_file:
        for job in manifest['jobs']:
            directory = Path(job['directory']).resolve()
            require(directory == folder/'runs'/job['id'], 'Population directory differs')
            term = next(j for j in terminal if j['id'] == job['id'])
            for name, key in [('samples.jsonl', 'samples_sha256'), ('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')]:
                digest = sha(directory/name)
                require(digest == term['output'][key] == assessment['provenance'][str(directory/name)], 'Terminal/audit output binding changed')
                files[str(directory/name)] = digest
            pm, summary = read(directory/'manifest.json'), read(directory/'summary.json')
            validate_model_manifest(model, pm)
            require(summary['complete'] and summary['manifest'] == pm and summary['samples'] == SAMPLES and summary['numerical_nulls'] == 0,
                    'Population incomplete or numerical proposals omitted')
            require(pm['seed'] == job['seed'] and pm['samples'] == SAMPLES and pm['cloud_replicates'] == 2
                    and pm['activity'] == .035 and pm['lambda'] == .035*64., 'Population allocation/bath changed')
            for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                              ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
                require(sha(directory/'provenance'/name) == pm[key], 'Population archived source changed: '+name)
            for name, key in [('config.json', 'config_sha256'), ('model.json', 'model_sha256'), ('shape.json', 'shape_sha256'),
                              ('source-bundle.json', 'source_bundle_sha256'), ('basin-normalizer', 'executable_sha256')]:
                require(manifest['archive_sha256'][name] == pm[key], 'Population/campaign source identity differs')
            audited = next(p for p in assessment['populations'] if p['job']['id'] == job['id'])
            require(audited['job']['seed'] == job['seed'] and audited['summary'] == summary, 'Completed audit population differs')
            pa = audited['proposal_audit']
            require(pa['checked_actual_poses'] == pa['checked_proposal_labels'] == SAMPLES
                    and pa['wall_domain']['checked_poses'] == SAMPLES and pa['wall_domain']['atomic_wall'] == WALL
                    and pa['wall_domain']['bath_wall_permeable'] is True and pa.get('factor_audit'), 'Incomplete schema4/active reciprocal audit')
            with (directory/'samples.jsonl').open() as stream:rows = [json.loads(line) for line in stream]
            weights = read_weights(rows, SAMPLES)
            valid = np.isfinite(weights['z']);indices = np.flatnonzero(valid)
            poses = [rows[i]['pose'] for i in indices]
            rn, ra = np.full(SAMPLES, np.inf), np.full(SAMPLES, np.inf)
            if poses:
                rn[indices], _ = chart_radii(poses, references['native'])
                ra[indices], _ = chart_radii(poses, references['alternative'])
            native = np.zeros(SAMPLES, bool);anchors = np.zeros(SAMPLES, int);triangle = native.copy()
            motif_masks = {key:np.zeros(SAMPLES, bool) for key in all_motif_keys | {'motif7_4_triangle'}}
            for i in indices:
                row = rows[i];label = classifier.classify(row['pose'])
                native[i] = label['native_any'];anchors[i] = label['native_anchor_count'];triangle[i] = label['registry_consistent_triangle']
                for key in label_keys(label):
                    if key in motif_masks:motif_masks[key][i] = True
                label_file.write(json.dumps(dict(population=job['id'], seed=job['seed'], draw=int(i), classification=label),
                                           separators=(',', ':'), allow_nan=False)+'\n')
                if label['native_any'] and row['depletion_contact'] is False:
                    anomalies.append(native_unbound_diagnostic(row, label, job['id']))
            contact = np.array([r['depletion_contact'] is True for r in rows])
            q = np.array([r['q'] if r['q'] is not None else np.inf for r in rows])
            d170 = np.array([math.dist(r['pose']['position'], [0., 0., 0.]) <= 170. for r in rows])
            masks = partition_masks(valid, contact, native, anchors, triangle, d170, q, rn, ra)
            masks.update(motif_masks)
            for ours, theirs in [('total', 'total'), ('original_q_le_1', 'native'), ('exclusion_contact', 'bound'), ('unbound', 'unbound')]:
                value = paired_moments(np.where(masks[ours], weights['z'], -np.inf), np.where(masks[ours], weights['h'], -np.inf))
                same_log(value['log_Qz'], audited['estimates'][theirs]['logQ'])
            population = dict(id=job['id'], seed=job['seed'], masks=masks, **weights)
            populations.append(population)
            require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds'] >= 0., 'Invalid sampler CPU time')
            cpu += summary['sampler_cpu_seconds']
    estimates, covariances, ratios = summarize_regions(populations)
    require(len(assessment['groups']) == 1, 'Audit mixed proposal scales')
    group = next(iter(assessment['groups'].values()))
    require(group['covariance_std_scale'] == 1., 'Audit proposal scale differs')
    pooled_audit = group['estimates']
    for ours, theirs in [('total', 'total'), ('original_q_le_1', 'native'), ('exclusion_contact', 'bound'), ('unbound', 'unbound')]:
        same_log(estimates[ours]['row_uncertainty']['log_Qz'], pooled_audit[theirs]['logQ'])
    return dict(arm=arm, campaign=str(folder), model_sha256=entry['model_sha256'], manifest_sha256=entry['manifest_sha256'],
        assessment_sha256=sha(assessment_path), source_sha256=files, physical_config=config, shape_sha256=shape_sha,
        proposal=dict(uniform_probability=.1, base_components=len(model['base_model']['weights']),
                      reciprocal_components=model['reciprocal_components'], virtual_components=len(model['base_model']['weights'])+sum(model['reciprocal_components'])),
        samples=4*SAMPLES, sampler_cpu_seconds=cpu, estimates=estimates, primary_covariance=covariances,
        primary_physical_ratios=ratios, support_certificate=support,
        native_labels=dict(path=labels_path.name, sha256=sha(labels_path)), native_entry_unbound=dict(count=len(anomalies), rows=anomalies,
            interpretation='Expected zero: native entry has gap<=2 Å and exclusion contact reaches gap<3 Å. Any discrepancy remains in the native primary class.'),
        scope=SCOPE)


def compare_references(arms, references):
    output = []
    for arm in arms:
        for name, reference in references['records'].items():
            value = contrast(arm['estimates'][name], reference)
            value.update(arm=arm['arm'], witness=name, reference_binding=references['binding'],
                scope='Identical named finite-region integral inside D170 and the atomic wall; independent wall-proposal and direct-region estimates. '
                      'No pooling and no whole-basin inference. A missing wall-proposal witness is unresolved, not zero mass.')
            output.append(value)
    return output


def report(result):
    lines = ['Full original-wall contact comparison: separate proposal laws.', '',
        '| Proposal | Class | log Qz | row RSE | population RSE | log Q0 | log(Qz/Q0) ± paired row SE | ESS | max weight |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in result['arms']:
        for name in ('total',)+PRIMARY:
            r = arm['estimates'][name];a, b = r['row_uncertainty'], r['population_uncertainty']
            if a['log_Qz'] is None:
                lines.append(f"| {arm['arm']} | {name} | unresolved | — | — | — | — | — | — |")
            else:
                lines.append(f"| {arm['arm']} | {name} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {a['log_Q0']:.6f} | {a['log_enhancement']:.6f} ± {a['log_enhancement_SE']:.6f} | {a['Qz_ESS']:.1f} | {a['largest_Qz_fraction']:.2%} |")
        lines += ['', f"{arm['arm']}: native entry/unbound observations = {arm['native_entry_unbound']['count']}; all remain in the native primary class.",
            'Population log Qz values (total): '+', '.join('unresolved' if p['log_Qz'] is None else f"{p['log_Qz']:.5f}" for p in arm['estimates']['total']['populations'])+'.']
        for name in ('outside_d170', 'native_one_anchor', 'native_both_anchors', 'registry_triangle', 'original_q_le_1'):
            value = arm['estimates'][name]['fraction_of_parent']['Qz']['row_uncertainty']
            lines.append(f"{name}: observed total-weight fraction "+('unresolved' if value['observed_fraction'] is None else f"{value['observed_fraction']:.2%}")+
                         ('; no resolved SE.' if value['SE'] is None else f" ± {value['SE']:.2%} paired row SE."))
    lines += ['', '| Proposal | Finite witness | wall log Qz | independent regional log Qz | log difference ± independent row SE |',
              '|---|---|---:|---:|---:|']
    for item in result['finite_reference_comparisons']:
        reference = result['finite_references']['records'][item['witness']]['row_uncertainty']['log_Qz']
        arm = next(a for a in result['arms'] if a['arm'] == item['arm'])
        observed = arm['estimates'][item['witness']]['row_uncertainty']['log_Qz']
        contrast_value = item['row_uncertainty']
        difference = 'unresolved' if 'unresolved' in contrast_value else f"{contrast_value['log_Qz_ratio']:.6f} ± {contrast_value['log_Qz_ratio_SE']:.6f}"
        value = 'unresolved' if observed is None else f'{observed:.6f}'
        lines.append(f"| {item['arm']} | {item['witness']} | {value} | {reference:.6f} | {difference} |")
    lines += ['', 'All masks retain every original attempted draw. Paired Qz/Q0 covariance, two-cloud noise, primary cross-covariance, '
              'four individual population estimates, motif IDs and all native support matches are saved in analysis.json and label files. '
              'Native entry has precedence in the exhaustive partition, including any diagnosed entry/unbound discrepancy. '
              'No-entry contact is not automatically nonnative adsorption; q<=1 is one historical reference, not all native sites.', '', SCOPE, '']
    return '\n'.join(lines)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('total',)+PRIMARY
    labels = ['total', 'native entry', 'contact, no entry', 'unbound, no entry']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for row, arm in enumerate(result['arms']):
        for column, (key, error, title) in enumerate([('log_Qz', 'Qz_relative_SE', 'log Qz'),
                                                     ('log_enhancement', 'log_enhancement_SE', 'log(Qz/Q0)')]):
            ax = axes[row, column]
            for i, name in enumerate(names):
                region = arm['estimates'][name]
                for j, pop in enumerate(region['populations']):
                    if pop[key] is not None:ax.plot(i+(j-1.5)*.09, pop[key], '.', color='#777777')
                for dx, field, marker, color in [(-.12, 'row_uncertainty', 'o', '#2459a6'), (.12, 'population_uncertainty', 's', '#c66223')]:
                    value = region[field]
                    if value[key] is not None:ax.errorbar(i+dx, value[key], yerr=value[error], fmt=marker, color=color, capsize=3,
                                                         label=field.replace('_', ' ') if i == 0 else None)
            ax.set_xticks(range(4), labels, rotation=15);ax.set_ylabel(arm['arm']+': '+title);ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Full original atomic wall; proposal laws separate; observed errors do not certify tails', fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'wall-contact-comparison.png', dpi=170);fig.savefig(output/'wall-contact-comparison.svg')
    plt.close(fig)


def analyze(campaign, out):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh downstream output required')
    protocol, status = validate_parent(root)
    classifier, classifier_binding = load_classifier(inside(root, protocol['native_definition']))
    entries = {e['arm']:e for e in protocol['campaigns']}
    config = read(root/'original/provenance/config.json');shape_sha = sha(root/'original/provenance/shape.json')
    references = load_references(root, protocol, config, shape_sha)
    require(classifier_binding['definition_sha256'] == read(inside(root, protocol['references']['comparison']))['native_definition']['definition_sha256'],
            'Wall/reference native criteria differ')
    out.mkdir(parents=True)
    try:
        arms = []
        for arm in ARMS:
            print('Aggregating saved wall poses and native labels: '+arm, flush=True)
            arms.append(load_arm(root, entries[arm], status, classifier, references, out))
        normalized = copy.deepcopy(arms[1]['physical_config']);normalized['shape'] = arms[0]['physical_config']['shape']
        require(normalized == arms[0]['physical_config'] and arms[1]['shape_sha256'] == arms[0]['shape_sha256'], 'Proposal arms have different physical targets')
        result = dict(schema='mobile-wall-contact-comparison-v1', complete=True, campaign=str(root),
            protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'), freeze_sha256=sha(root/'freeze.json'),
            analyzer_sha256=sha(__file__), primary_partition=list(PRIMARY), atomic_wall=WALL, bath_wall_permeable=True,
            native_definition=classifier_binding, finite_references=dict(records=references['records'], binding=references['binding']),
            arms=arms, finite_reference_comparisons=compare_references(arms, references),
            independent_proposal_comparisons={name:contrast(arms[1]['estimates'][name], arms[0]['estimates'][name]) for name in ('total',)+PRIMARY},
            scope=SCOPE)
        sources = out/'provenance';sources.mkdir()
        for name, source in local_sources(__file__).items():shutil.copy2(source, sources/name)
        result['analyzer_source_sha256'] = {p.name:sha(p) for p in sources.iterdir()}
        write(out/'analysis.json', result);(out/'report.md').write_text(report(result));plot(result, out)
        write(out/'freeze.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        print(json.dumps(dict(complete=True, out=str(out), analysis_sha256=sha(out/'analysis.json'))))
        return result
    except Exception as error:
        write(out/'failure.json', dict(complete=False, error=str(error), scope='No sampler or raw audit invoked; partial outputs preserved.'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'runs/mobile-wall-contact-campaign-20260921')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();analyze(args.campaign, args.out)
