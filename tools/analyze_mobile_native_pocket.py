#!/usr/bin/env python3
"""Reaggregate completed independent finite-pocket integrals; never launch an audit.

Saved uniform latent weights keep their original attempted-draw denominator.
The native entry catalogue is a separate instantaneous reporting observable.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_mobile_competing_reference import (
    PREPARATION, PILOT, paired_moments, check_estimate, contrast,
    verify_preparation, load_region)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('r5', 'shell5to8')
SEEDS = [124101010 + 1009*i for i in range(8)]
INITIAL_ALLOCATION = {
    'r5': dict(inner=0., outer=5., replicates=4, samples=8192),
    'shell5to8': dict(inner=5., outer=8., replicates=4, samples=8192)}
COVERAGE_ALLOCATION = {
    'r5repeat': dict(inner=0., outer=5., replicates=8, samples=16384),
    'shell8to12': dict(inner=8., outer=12., replicates=4, samples=8192),
    'shell12to16': dict(inner=12., outer=16., replicates=4, samples=8192),
    'shell16to24': dict(inner=16., outer=24., replicates=4, samples=8192),
    'shell24to32': dict(inner=24., outer=32., replicates=4, samples=8192)}
INITIAL_COMPARISON_SHA = '0dc4d265f16161ba548e46b1ce9b7b366836ad26ae657a947136612c7dd4f700'
COVERAGE_SEED = 125101010
SCOPE = ('Independent conditional finite regions on the exact observed two-neighbor scaffold. '
    'Original native R4 is R4 intersect q<=1 in its original chart; alternative R5 and R8 '
    'use another frozen chart and q>1. Neither is an entire native basin. Native entry labels '
    'do not alter the physical target. No overall basin ratio, mobile equilibrium, stationary '
    'efficiency, or bound on unseen outer/high-weight mass follows from these estimates. '
    'Weights use uniform latent volume and its physical Jacobian, with no global atlas denominator.')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inside(root, name):
    path = (Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Archived path escapes its root')
    return path


def paired_noise(logs, pairs):
    """Observed two-cloud variance decomposition with unconditional zeros."""
    logs, pairs = np.asarray(logs, float), np.asarray(pairs, float)
    require(pairs.shape == (len(logs), 2), 'Require two clouds per attempted draw')
    require(not np.isnan(pairs).any() and not np.isposinf(pairs).any(), 'Invalid cloud weights')
    finite = np.isfinite(logs)
    require(np.array_equal(np.isfinite(pairs).all(axis=1), finite), 'Cloud support differs')
    require(np.isneginf(pairs[~finite]).all(), 'Invalid draws require two cloud zeros')
    if not finite.any():
        return None
    require(np.max(abs(logsumexp(pairs[finite], axis=1)-math.log(2)-logs[finite])) < 2e-10,
            'Two-cloud mean differs from saved physical weight')
    offset = float(max(logs.max(), pairs.max()))
    a, b = np.exp(pairs-offset).T
    y = (a+b)/2
    total = float(np.var(y, ddof=1))
    cloud = float(np.mean((a-b)**2)/4)
    pose = total-cloud
    denominator = len(y)*float(y.mean())**2
    return dict(log_weight_offset=offset, scaled_total_variance=total,
        scaled_cloud_variance=cloud, scaled_residual_pose_variance=pose,
        cloud_fraction=cloud/total if total > 0 else None,
        relative_variance_mean_cloud=cloud/denominator,
        relative_variance_mean_pose=pose/denominator,
        scope='E[(W1-W2)^2/4] estimates the noise of the two-cloud average, with all invalid zeros. '
              'Residual pose variance is not clipped; finite-sample cloud fraction may exceed one. '
              'The cloud term also contributes to log(Qz/Q0) variance; Q0 has no cloud noise.')


def ratio(log_numerator, log_denominator):
    """Paired delta variance allowing a strict subset in the numerator."""
    a, b = np.asarray(log_numerator, float), np.asarray(log_denominator, float)
    require(a.shape == b.shape and a.ndim == 1 and len(a) >= 2, 'Ratio records differ')
    require(not np.isnan(a).any() and not np.isnan(b).any() and
            not np.isposinf(a).any() and not np.isposinf(b).any(), 'Invalid ratio weights')
    if not np.isfinite(b).any():
        return dict(observed_fraction=None, log_ratio=None, SE=None, log_ratio_SE=None,
                    unresolved='Denominator unobserved; no zero mass or upper bound established.')
    if not np.isfinite(a).any():
        return dict(observed_fraction=0., log_ratio=None, SE=None, log_ratio_SE=None,
                    unresolved='No observed numerator contribution; zero is a sample fraction, not physical zero or an upper bound.')
    wa, wb = np.exp(a-logsumexp(a)), np.exp(b-logsumexp(b))
    n = len(a)
    variance = float(n/(n-1)*np.sum((wa-wb)**2))
    log_ratio = float(logsumexp(a)-logsumexp(b))
    fraction = math.exp(log_ratio)
    return dict(observed_fraction=fraction, log_ratio=log_ratio,
        SE=fraction*math.sqrt(variance), log_ratio_SE=math.sqrt(variance),
        covariance_relative=float(n/(n-1)*((wa-1/n)@(wb-1/n))),
        scope='Shared-draw numerator/denominator covariance; observed delta uncertainty only.')


def summarize(populations, name, masks=None):
    """Every mask replaces nonmembers by zero; it never shortens a population."""
    values, arrays = [], []
    for i, population in enumerate(populations):
        z, h, p = population['z'], population['h'], population['pairs']
        mask = np.ones(len(z), bool) if masks is None else np.asarray(masks[i], bool)
        require(mask.shape == z.shape, 'Subset mask has wrong unconditional denominator')
        z, h, p = np.where(mask, z, -np.inf), np.where(mask, h, -np.inf), np.where(mask[:, None], p, -np.inf)
        entry = paired_moments(z, h)
        entry.update(id=population['id'], seed=population['seed'], paired_cloud_noise=paired_noise(z, p))
        values.append(entry)
        arrays.append((z, h, p))
    require(len(set(len(a[0]) for a in arrays)) == 1 and len(arrays) >= 2, 'Require equal-budget independent populations')
    z, h, p = [np.concatenate([a[i] for a in arrays]) for i in range(3)]
    means = [[-math.inf if v[field] is None else v[field] for v in values] for field in ('log_Qz', 'log_Q0')]
    result = dict(name=name, row_uncertainty=paired_moments(z, h),
        population_uncertainty=paired_moments(*means), populations=values,
        paired_cloud_noise=paired_noise(z, p),
        uncertainty_scope='Row delta uncertainty and dispersion of independent equal-budget population means are both retained. '
                          f'{len(values)} population means; agreement is not a tail-coverage guarantee.',
        population_count=len(values), samples_per_population=len(arrays[0][0]))
    if masks is not None:
        parent_z = np.concatenate([v['z'] for v in populations])
        parent_h = np.concatenate([v['h'] for v in populations])
        result['fraction_of_parent'] = {}
        for k, child, parent in [('Qz', z, parent_z), ('Q0', h, parent_h)]:
            index = 0 if k == 'Qz' else 1
            parent_means = [paired_moments(v['z'], v['h'])['log_'+k] for v in populations]
            result['fraction_of_parent'][k] = dict(row_uncertainty=ratio(child, parent),
                population_uncertainty=ratio(means[index], [-math.inf if x is None else x for x in parent_means]))
    return result


def sum_regions(regions, name):
    """Independent strata add integrals/covariance, never concatenate their rows."""
    require(len(regions) >= 2, 'Need independent disjoint strata')
    result = dict(name=name, source_regions=[r['name'] for r in regions],
                  scope='Sum of independent disjoint region integrals, not an average over pooled stratum draws.')
    if any(r['row_uncertainty']['log_Qz'] is None for r in regions):
        result.update(row_uncertainty=dict(log_Qz=None, log_Q0=None, log_enhancement=None),
            population_uncertainty=dict(log_Qz=None, log_Q0=None, log_enhancement=None),
            unresolved='An unobserved stratum cannot be silently identified as zero physical mass.')
        return result
    lz = float(logsumexp([r['row_uncertainty']['log_Qz'] for r in regions]))
    lh = float(logsumexp([r['row_uncertainty']['log_Q0'] for r in regions]))
    fz = np.exp([r['row_uncertainty']['log_Qz']-lz for r in regions])
    fh = np.exp([r['row_uncertainty']['log_Q0']-lh for r in regions])
    for field in ('row_uncertainty', 'population_uncertainty'):
        covariance = sum(np.asarray(r[field]['covariance_relative'])*np.outer([a, b], [a, b])
                         for r, a, b in zip(regions, fz, fh))
        result[field] = dict(log_Qz=lz, log_Q0=lh, log_enhancement=lz-lh,
            covariance_relative=covariance.tolist(), Qz_relative_SE=math.sqrt(max(0., covariance[0, 0])),
            Q0_relative_SE=math.sqrt(max(0., covariance[1, 1])),
            log_enhancement_SE=math.sqrt(max(0., covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1])))
    result['paired_cloud_noise'] = dict(relative_variance_mean_cloud=sum(
        a*a*r['paired_cloud_noise']['relative_variance_mean_cloud'] for r, a in zip(regions, fz)),
        scope='Independent stratum cloud variances added with squared physical-mass fractions.')
    return result


def independent_shell_fraction(core, shell):
    result = {}
    for field in ('row_uncertainty', 'population_uncertainty'):
        result[field] = {}
        for k in ('Qz', 'Q0'):
            a, b = core[field], shell[field]
            if a['log_'+k] is None or b['log_'+k] is None:
                result[field][k] = dict(observed_fraction=None, SE=None, unresolved='An independent stratum is unobserved.')
                continue
            fraction = math.exp(b['log_'+k]-float(np.logaddexp(a['log_'+k], b['log_'+k])))
            error = fraction*(1-fraction)*math.hypot(a[k+'_relative_SE'], b[k+'_relative_SE'])
            result[field][k] = dict(observed_fraction=fraction, SE=error)
    result['scope'] = 'Shell/(core+shell), accounting for the same shell integral in numerator and denominator. '
    return result


def nested_masks(radius):
    radius = np.asarray(radius, float)
    require(np.isfinite(radius).all() and (radius >= 0).all() and (radius <= 5*(1+1e-12)).all(), 'Unexpected R5 radius')
    return dict(r3=radius <= 3., r4=radius <= 4., shell4to5=radius > 4.)


def target_equal(config, region, shape_sha):
    require(region['shape_sha256'] == shape_sha, 'Physical shape differs')
    for region_key, config_key in [('physical_fixed_neighbors', 'fixed_poses'), ('capture_center', 'capture_center'),
            ('capture_radius', 'capture_radius'), ('activity', 'reservoir_density'),
            ('depletant_radius', 'depletant_radius'), ('physical_metric', 'metadata')]:
        require(region[region_key] == config[config_key], 'Physical target differs: '+region_key)


def validate_regions(regions, config, shape, shape_sha, allocation=None):
    allocation = INITIAL_ALLOCATION if allocation is None else allocation
    require(set(regions) == set(allocation), 'Region set differs from frozen allocation')
    for arm, region in regions.items():
        target_equal(config, region, shape_sha)
        require(region['fixed_neighbor'] == config['fixed_poses'][0], 'Chart must be anchored at exact fixed body 2')
        require(region['minimum_original_q'] == 1. and region.get('minimum_original_q_inclusive', True) is False
                and 'maximum_original_q' not in region, 'Alternative pocket requires original q>1')
        inner, outer = allocation[arm]['inner'], allocation[arm]['outer']
        require(region.get('minimum_mahalanobis_radius', 0.) == inner and region['mahalanobis_radius'] == outer,
                'Unexpected disjoint radii')
        chart = region['gaussian_chart']
        require('base_model' not in chart and 'reciprocal_components' not in chart and chart['weights'] == [1.]
                and chart['means'] == [[0.]*6] and len(chart['anchors']) == len(chart['covariances']) == 1,
                'Require a single ordinary ideal-centered geometric chart')
        require(chart['coordinate_convention'] == 'anchor-body-relative' and chart['angular_length'] > 0,
                'Unknown chart convention')
        covariance = np.asarray(chart['covariances'][0], float)
        require(covariance.shape == (6, 6) and np.isfinite(covariance).all()
                and np.max(abs(covariance-covariance.T)) < 1e-12, 'Invalid chart covariance')
        np.linalg.cholesky(covariance)
    reference = next(iter(regions.values()))
    require(all(r['gaussian_chart'] == reference['gaussian_chart'] for r in regions.values()), 'Region charts differ; no nested sum')
    require(config['capture_center'] == [0., 0., 0.] and config['capture_radius'] == 170.
            and config['reservoir_density'] == .035 and config['depletant_radius'] == 1.5,
            'Unexpected D170 physical target')
    shape_bound = max(np.linalg.norm(atom['center'])+atom['radius'] for atom in shape['atoms'])
    wall = config['metadata']['physical_sphere_radius_A']
    require(config['capture_radius']+shape_bound < wall, 'D170 does not certify the original atomic wall')
    region = reference;fixed = region['fixed_neighbor'];chart = region['gaussian_chart']
    rotation = Rotation.from_quat(np.asarray(fixed['orientation'])[[1, 2, 3, 0]]).as_matrix()
    center = np.asarray(fixed['position'])+rotation@np.asarray(chart['anchors'][0]['position'])
    radius = max(a['outer'] for a in allocation.values())
    upper = float(np.linalg.norm(center)+radius*np.linalg.norm(np.linalg.cholesky(chart['covariances'][0])[:3], ord=2))
    require(upper < config['capture_radius'], 'Finite chart region leaves the certified capture domain')
    return dict(shape_bound_A=float(shape_bound), maximum_region_center_radius_A=upper, maximum_latent_radius=radius,
        minimum_capture_wall_clearance_A=float(wall-config['capture_radius']-shape_bound),
        scope='All orientations in D170 obey the original wall; this is not whole-wall coverage. Bath remains unbounded.')


def load_classifier(definition):
    definition = Path(definition).resolve()
    data = read(definition)
    runtime = definition.parent/'inputs/source/native_contact_regions.py'
    require(sha(runtime) == data['input_sha256']['source/native_contact_regions.py'], 'Native runtime changed')
    spec = importlib.util.spec_from_file_location('frozen_pocket_native_regions', runtime)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    classifier = module.NativeContactRegions(definition)
    return classifier, dict(definition=str(definition), definition_sha256=sha(definition),
        runtime_sha256=sha(runtime), input_sha256=data['input_sha256'], criteria=data['criteria'], scope=data['scope'])


def validate_parent(root):
    protocol, status = read(root/'protocol.json'), read(root/'status.json')
    require(protocol['schema'] == 'mobile-native-pocket-controller-v1', 'Unexpected pocket controller schema')
    require(status['complete'] and status['phase'] == 'complete' and not status.get('running', False), 'Campaign is incomplete')
    require(status['protocol_sha256'] == sha(root/'protocol.json'), 'Terminal protocol binding differs')
    freeze = read(root/'freeze.json');files = freeze.get('files', freeze)
    require(files, 'Missing frozen input bindings')
    for name, digest in files.items():
        require(sha(inside(root, name)) == digest, 'Frozen campaign input changed: '+name)
    allocation, seed_base = protocol_design(protocol)
    campaigns = protocol['campaigns']
    jobs = sum(a['replicates'] for a in allocation.values())
    require(len(campaigns) == len(allocation) and {c['arm'] for c in campaigns} == set(allocation), 'Missing/duplicate campaign arm')
    require(len(status['jobs']) == jobs and len({(j['arm'], j['id']) for j in status['jobs']}) == jobs, 'Missing/duplicate terminal population')
    require(set(status['audits']) == set(allocation), 'Missing terminal audit')
    require(protocol['cloud_replicates'] == 2 and protocol['lambda_ratio'] == 64. and protocol['total_jobs'] == jobs,
            'Unexpected frozen integration budget')
    return protocol, status


def protocol_design(protocol):
    design = protocol.get('design', 'initial')
    require(design in ('initial', 'coverage'), 'Unknown pocket campaign design')
    allocation = INITIAL_ALLOCATION if design == 'initial' else COVERAGE_ALLOCATION
    seed_base = SEEDS[0] if design == 'initial' else COVERAGE_SEED
    if design == 'coverage':
        require(protocol['allocation'] == allocation, 'Coverage allocation changed')
        require(protocol['samples_per_population'] is None and protocol['populations_per_region'] is None,
                'Mixed-budget campaign must not claim one uniform population size')
        require(protocol['seed_base'] == seed_base and protocol['total_unconditional_draws'] == 262144,
                'Coverage seeds or attempted-draw budget changed')
    else:
        require(protocol['samples_per_population'] == 8192 and protocol['populations_per_region'] == 4,
                'Initial allocation changed')
        if 'allocation' in protocol:require(protocol['allocation'] == allocation, 'Initial region allocation changed')
    return allocation, seed_base


def validate_jobs(jobs, terminal, arm, allocation=None, seed_base=SEEDS[0]):
    allocation = INITIAL_ALLOCATION if allocation is None else allocation
    require(arm in allocation, 'Unknown campaign arm')
    design = allocation[arm];count = design['replicates']
    ids = {f'r{i:02d}' for i in range(count)}
    require(len(jobs) == len(terminal) == count and {j['id'] for j in jobs} == ids, 'Missing/duplicate populations')
    require(len({j['id'] for j in terminal}) == count and {j['id'] for j in terminal} == ids, 'Terminal population set differs')
    order = list(allocation)
    offset = sum(allocation[a]['replicates'] for a in order[:order.index(arm)])
    for j in jobs:
        index = int(j['id'][1:])
        require(j['samples'] == design['samples'] and j['seed'] == seed_base+1009*(offset+index),
                'Population sample count or independent seed differs')
        term = next(t for t in terminal if t['id'] == j['id'])
        require(term.get('status', 'complete') == 'complete', 'Population is not terminal')
        require(term['arm'] == arm and term['seed'] == j['seed'] and term['samples'] == j['samples'] and term['returncode'] == 0,
                'Population did not complete its frozen allocation')


def read_weights(rows, samples, region):
    require(len(rows) == samples and [r['draw'] for r in rows] == list(range(samples)), 'Unconditional draws missing/repeated')
    z, h, pairs, radii = [], [], [], []
    inner, outer = region.get('minimum_mahalanobis_radius', 0.), region['mahalanobis_radius']
    for row in rows:
        radius = row['latent_radius']
        require(math.isfinite(radius) and (radius > inner if inner else radius >= 0.) and radius <= outer*(1+1e-12), 'Draw outside uniform shell')
        require(all(type(row[k]) is bool for k in ('hard_valid', 'region_valid', 'capture_valid')), 'Invalid support flags')
        require(math.isfinite(row['q']) and row['region_valid'] == (row['q'] > 1.), 'Original q partition differs')
        valid = row['hard_valid'] and row['region_valid']
        radii.append(radius)
        if not valid:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'], 'Invalid draws must remain zeros')
            z.append(-math.inf);h.append(-math.inf);pairs.append([-math.inf, -math.inf])
            continue
        require(row['capture_valid'] and len(row['clouds']) == 2, 'Valid draw lacks capture or two independent clouds')
        require(math.isfinite(row['log_hard_weight']) and math.isfinite(row['log_importance_weight']), 'Nonfinite physical weights')
        cloud = [row['log_hard_weight']+c['log_weight'] for c in row['clouds']]
        require(all(math.isfinite(x) for x in cloud), 'Nonfinite cloud')
        require(abs(float(logsumexp(cloud)-math.log(2))-row['log_importance_weight']) < 2e-10, 'Cloud mean differs from saved weight')
        z.append(row['log_importance_weight']);h.append(row['log_hard_weight']);pairs.append(cloud)
    return dict(z=np.asarray(z), h=np.asarray(h), pairs=np.asarray(pairs), radius=np.asarray(radii))


def label_keys(label):
    keys = set()
    if label['native_any']:keys.add('native_any')
    if label['cooperative_entry']:keys.add('both_anchor_entry')
    if label['registry_consistent_triangle']:keys.add('registry_triangle')
    for match in label['matches']:
        keys.add(f"anchor{match['anchor_index']}_motif{match['motif_id']}")
    if any(t['anchor0_to_moving_motif'] == 7 and t['anchor1_to_moving_motif'] == 4
           for t in label['registry_consistent_triangles']):
        keys.add('motif7_4_triangle')
    return keys


def load_arm(root, entry, status, config, shape_sha, classifier, output, allocation=None, seed_base=SEEDS[0]):
    allocation = INITIAL_ALLOCATION if allocation is None else allocation
    arm = entry['arm'];folder = Path(entry['path']).resolve()
    require(folder == root/arm, 'Arm path differs')
    require(sha(folder/'manifest.json') == entry['manifest_sha256'], 'Arm manifest changed')
    manifest = read(folder/'manifest.json');region = read(folder/'provenance/region.json')
    require(manifest['schema'] == 'uniform-latent-region-campaign-v1' and 'importance_guide_sha256' not in manifest
            and 'importance-guide.json' not in manifest['archive_sha256'], 'Expected uniform latent integration without guide')
    require(manifest['region_sha256'] == entry['region_sha256'] == sha(folder/'provenance/region.json'), 'Region identity changed')
    require(manifest['lambda_ratio'] == 64. and manifest['cloud_replicates'] == 2, 'Cloud law differs')
    for name, digest in manifest['archive_sha256'].items():
        require(sha(inside(folder/'provenance', name)) == digest, 'Arm archive changed: '+name)
    cfg = read(folder/'provenance/config.json')
    normalized = copy.deepcopy(cfg);normalized['shape'] = config['shape']
    require(normalized == config and Path(cfg['shape']).resolve() == folder/'provenance/shape.json', 'Config changed beyond shape relocation')
    target_equal(cfg, region, shape_sha)
    require(sha(folder/'provenance/shape.json') == shape_sha, 'Arm shape differs')
    terminal = [j for j in status['jobs'] if j['arm'] == arm]
    validate_jobs(manifest['jobs'], terminal, arm, allocation, seed_base)
    expected_populations = allocation[arm]['replicates']
    expected_samples = allocation[arm]['samples']
    record = status['audits'][arm]
    assessment_path = folder/'assessment/analysis.json'
    require(record['returncode'] == 0 and sha(assessment_path) == record['analysis_sha256'], 'Completed audit changed or failed')
    assessment = read(assessment_path)
    require(assessment['region_sha256'] == entry['region_sha256'] and len(assessment['populations']) == expected_populations
            and {p['id'] for p in assessment['populations']} == {j['id'] for j in manifest['jobs']}, 'Audit population/region set differs')
    require(assessment['independently_reconstructed_poses'] == expected_populations*expected_samples, 'Audit did not reconstruct every unconditional pose')
    populations, binding, cpu = [], {}, 0.
    labels_path = output/f'{arm}-native-labels.jsonl'
    with labels_path.open('x') as labels_file:
        for job in manifest['jobs']:
            directory = Path(job['directory']).resolve()
            require(directory == folder/'runs'/job['id'], 'Population directory differs')
            term = next(j for j in terminal if j['id'] == job['id'])
            for name, key in [('samples.jsonl', 'samples_sha256'), ('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')]:
                digest = sha(directory/name)
                require(term['output'][key] == digest, 'Terminal output changed: '+name)
                binding[str(directory/name)] = digest
            summary, pm = read(directory/'summary.json'), read(directory/'manifest.json')
            require(summary['complete'] and summary['samples'] == job['samples'] and summary['manifest'] == pm
                    and pm['schema'] == 'uniform-latent-region-normalizer-v2', 'Population incomplete or unexpected law')
            require(pm['seed'] == job['seed'] and pm['samples'] == job['samples'] and pm['cloud_replicates'] == 2,
                    'Population allocation differs')
            require(pm['physical_fixed_neighbors'] == cfg['fixed_poses'] and pm['chart_anchor'] == region['fixed_neighbor'], 'Population scaffold differs')
            require(pm['region_sha256'] == entry['region_sha256'] and pm['config_sha256'] == sha(folder/'provenance/config.json')
                    and pm['shape_sha256'] == shape_sha and pm['executable_sha256'] == manifest['archive_sha256']['latent-region-normalizer'],
                    'Population physical/archive identity differs')
            require(pm['lambda_ratio'] == 64. and pm['activity'] == .035 and pm['lambda'] == .035*64., 'Population bath law differs')
            require(pm['minimum_original_q'] == 1. and pm['minimum_original_q_inclusive'] is False
                    and pm.get('maximum_original_q') is None and pm['minimum_latent_radius'] == region.get('minimum_mahalanobis_radius', 0.),
                    'Population region bounds differ')
            for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                              ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
                require(sha(directory/'provenance'/name) == pm[key], 'Population archived input changed: '+name)
            if 'source-bundle.json' in manifest['archive_sha256']:
                require(pm['source_bundle_sha256'] == manifest['archive_sha256']['source-bundle.json'], 'Compiled source binding differs')
            audited = next(p for p in assessment['populations'] if p['id'] == job['id'])
            require(audited['seed'] == job['seed'] and audited['samples_sha256'] == summary['samples_sha256']
                    == sha(directory/'samples.jsonl'), 'Previously audited raw rows changed')
            with (directory/'samples.jsonl').open() as stream:
                rows = [json.loads(line) for line in stream]
            arrays = read_weights(rows, job['samples'], region)
            check_estimate(paired_moments(arrays['z'], arrays['h']), audited['estimate'], audited['hard_region'])
            keys = []
            for row, valid in zip(rows, np.isfinite(arrays['z'])):
                label = classifier.classify(row['pose']) if valid else None
                keys.append(label_keys(label) if label is not None else set())
                labels_file.write(json.dumps(dict(population=job['id'], seed=job['seed'], draw=row['draw'],
                    applicable=bool(valid), label=label), separators=(',', ':'), allow_nan=False)+'\n')
            populations.append(dict(id=job['id'], seed=job['seed'], label_keys=keys, **arrays))
            require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds'] >= 0., 'Invalid CPU total')
            cpu += summary['sampler_cpu_seconds']
    result = summarize(populations, arm)
    check_estimate(result['row_uncertainty'], assessment['estimate'], assessment['hard_region'])
    result.update(campaign=str(folder), region_sha256=entry['region_sha256'], manifest_sha256=entry['manifest_sha256'],
        assessment_sha256=sha(assessment_path), source_sha256=binding, sampler_cpu_seconds=cpu,
        native_labels=dict(path=labels_path.name, sha256=sha(labels_path)), region_definition=region.get('definition'))
    labels = {'native_any', 'both_anchor_entry', 'registry_triangle', 'motif7_4_triangle'}
    labels.update(key for p in populations for keys in p['label_keys'] for key in keys)
    result['native_diagnostics'] = {key:summarize(populations, key, [[key in keys for keys in p['label_keys']] for p in populations])
                                    for key in sorted(labels)}
    if allocation[arm]['inner'] == 0. and allocation[arm]['outer'] == 5.:
        result['nested_regions'] = {key:summarize(populations, key, [nested_masks(p['radius'])[key] for p in populations])
                                    for key in ('r3', 'r4', 'shell4to5')}
    return result, populations


def local_sources(path):
    pending, found = [Path(path).resolve()], {}
    while pending:
        source = pending.pop()
        if source.name in found:
            require(found[source.name] == source, 'Ambiguous source closure')
            continue
        found[source.name] = source
        for node in ast.walk(ast.parse(source.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                candidate = source.parent/(name.split('.')[0]+'.py')
                if candidate.is_file():pending.append(candidate)
    return found


def report(result):
    lines = ['Independent finite-pocket reference. Invalid draws remain in every original sample denominator.', '',
        '| Region | log Qz | row RSE | population RSE | log Q0 | log(Qz/Q0) ± paired row SE |',
        '|---|---:|---:|---:|---:|---:|']
    regions = [result['original_native_r4']]+result['regions']+[result['cumulative_r8']]
    for r in regions:
        a, b = r['row_uncertainty'], r['population_uncertainty']
        if a['log_Qz'] is None:
            lines.append(f"| {r['name']} | unresolved | — | — | — | — |")
        else:
            lines.append(f"| {r['name']} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {a['log_Q0']:.6f} | {a['log_enhancement']:.6f} ± {a['log_enhancement_SE']:.6f} |")
    shell = result['shell_fraction_of_r8']
    if shell['row_uncertainty']['Qz']['observed_fraction'] is not None:
        value, error = shell['row_uncertainty']['Qz']['observed_fraction'], shell['row_uncertainty']['Qz']['SE']
        lines += ['', f'The independent 5–8 shell contributes {value:.2%} ± {error:.2%} observed row SE of R8; '
                    f"population SE is {shell['population_uncertainty']['Qz']['SE']:.2%}. This does not bound mass outside R8."]
    for r in result['regions']:
        a = r['row_uncertainty'];noise = r['paired_cloud_noise']
        lines += ['', f"{r['name']}: per-population log Qz = "+', '.join('unresolved' if p['log_Qz'] is None else f"{p['log_Qz']:.5f}" for p in r['populations'])+'.']
        if a['log_Qz'] is not None:
            lines += [f"Observed row ESS {a['Qz_ESS']:.1f}; largest weight {a['largest_Qz_fraction']:.2%}."]
            lines += ([f"Paired-cloud variance fraction {noise['cloud_fraction']:.2%}."]
                      if noise['cloud_fraction'] is not None else ['Cloud fraction undefined for constant observed weights.'])
        for name in ('native_any', 'both_anchor_entry', 'registry_triangle', 'motif7_4_triangle'):
            fractions = r['native_diagnostics'][name]['fraction_of_parent']['Qz']
            f = fractions['row_uncertainty']
            lines += [f"{name}: observed physical-weight fraction "+('unresolved' if f['observed_fraction'] is None else f"{f['observed_fraction']:.2%}")+
                      (' (no resolved SE).' if f['SE'] is None else f" ± {f['SE']:.2%} paired row SE.")]
    lines += ['', 'R5 nested estimates use the same unconditional R5 rows and their original R5-volume weights:', '',
              '| Subregion | log Qz | row RSE | population RSE |', '|---|---:|---:|---:|']
    for r in result['regions'][0]['nested_regions'].values():
        a, b = r['row_uncertainty'], r['population_uncertainty']
        lines.append(f"| {r['name']} | unresolved | — | — |" if a['log_Qz'] is None else
                     f"| {r['name']} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} |")
    lines += ['', 'Comparisons to original native R4 concern two named finite regions only:']
    for c in result['finite_region_contrasts']:
        a = c['row_uncertainty']
        if 'unresolved' not in a:
            lines += [f"- Original native R4 / {c['denominator_region']}: log physical-mass ratio {a['log_Qz_ratio']:.6f} ± {a['log_Qz_ratio_SE']:.6f} row SE; "
                      f"log hard-volume ratio {a['log_Q0_ratio']:.6f}; enhancement difference {a['log_enhancement_difference']:.6f}."]
    lines += ['', 'Qz/Q0 is a ratio of paired means. Cloud noise and population scatter remain explicit; neither substitutes for tail coverage. '
              'Native entry means the frozen stateless contact criterion, not the original q<=1 label or persistent attachment. '
              'All motif IDs and supporting triangle matches are saved. No-entry is not proof of nonnative geometry.', '', SCOPE, '']
    return '\n'.join(lines)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    regions = [result['original_native_r4']]+result['regions']+[result['cumulative_r8']]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, field, se, title in [(axes[0], 'log_Qz', 'Qz_relative_SE', 'Finite physical mass'),
                                (axes[1], 'log_enhancement', 'log_enhancement_SE', 'Regional depletion enhancement')]:
        for i, r in enumerate(regions):
            a, b = r['row_uncertainty'], r['population_uncertainty']
            if a[field] is None:
                continue
            ax.errorbar(i-.05, a[field], yerr=a[se], fmt='o', color='#2459a6', capsize=3, label='row delta SE' if i == 0 else None)
            ax.errorbar(i+.05, b[field], yerr=b[se], fmt='s', color='#c66223', capsize=3, label='population SE' if i == 0 else None)
            for p in r.get('populations', []):
                if p[field] is not None:ax.plot(i, p[field], '.', color='#626262', alpha=.55)
        ax.set_xticks(range(len(regions)), ['original R4', 'alternative R5', 'shell 5–8', 'alternative R8'], rotation=15)
        ax.set_title(title);ax.set_ylabel('log Qz' if field == 'log_Qz' else 'log(Qz/Q0)')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Named finite regions; observed errors do not establish basin coverage', fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'finite-pocket-reference.png', dpi=170)
    fig.savefig(output/'finite-pocket-reference.svg')
    plt.close(fig)


def analyze(campaign, out, original_preparation=PREPARATION, original_native=PILOT/'native-r4'):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh downstream output directory required')
    protocol, status = validate_parent(root)
    require(protocol['seed_base'] == SEEDS[0] and protocol['total_unconditional_draws'] == 65536, 'Unexpected seed base or total budget')
    require(sha(root/'provenance/source-config.json') == protocol['original_config_sha256']
            and sha(root/'provenance/source-model.json') == protocol['model_sha256'], 'Original frozen source identity differs')
    classifier, classifier_binding = load_classifier(inside(root, protocol['native_definition']))
    entries = {c['arm']:c for c in protocol['campaigns']}
    regions = {name:read(Path(entry['path'])/'provenance/region.json') for name, entry in entries.items()}
    config = read(Path(entries['r5']['path'])/'provenance/config.json')
    shape_path = Path(entries['r5']['path'])/'provenance/shape.json'
    shape_sha = sha(shape_path)
    require(classifier.fixed_poses == config['fixed_poses'] and classifier.shape_sha256 == shape_sha, 'Classifier physical target differs')
    old_plan, old_config, old_bound = verify_preparation(Path(original_preparation).resolve())
    normalized = copy.deepcopy(config);normalized['shape'] = old_config['shape']
    require(normalized == old_config, 'Physical target differs from original native finite reference')
    wall = validate_regions(regions, config, read(shape_path), shape_sha)
    source_model = read(root/'provenance/source-model.json')['base_model']
    chart = regions['r5']['gaussian_chart']
    motif = classifier.motif_by_id[7]
    ideal_rotation = Rotation.from_quat(np.asarray(motif['relative_orientation'])[[1, 2, 3, 0]]).as_matrix()
    require(chart['covariances'][0] == source_model['covariances'][19]
            and chart['angular_length'] == source_model['angular_length'], 'Frozen base-19 geometric covariance changed')
    require(chart['anchors'][0]['position'] == motif['relative_position']
            and np.max(abs(np.asarray(chart['anchors'][0]['rotation'])-ideal_rotation)) < 1e-14,
            'Chart center is not the frozen ideal motif 7')
    for entry in entries.values():
        manifest = read(Path(entry['path'])/'manifest.json')
        require(manifest['archive_sha256']['latent-region-normalizer'] == protocol['physical_executable_sha256']
                and manifest['archive_sha256']['source-bundle.json'] == protocol['source_bundle_sha256'],
                'Arm executable or source bundle differs from reviewed protocol')
    original = load_region(Path(original_native).resolve(), Path(original_preparation).resolve(), old_plan, old_config, old_bound)
    require(original['name'] == 'native-r4', 'Require the original native R4 finite reference')
    original['name'] = 'original-native-r4'
    require(set(p['seed'] for p in original['populations']).isdisjoint(SEEDS), 'New and original reference seeds overlap')
    out.mkdir(parents=True)
    try:
        results = []
        for arm in ARMS:
            print(f'Reaggregating saved {arm} weights and stateless native labels', flush=True)
            value, _ = load_arm(root, entries[arm], status, config, shape_sha, classifier, out)
            results.append(value)
        cumulative = sum_regions(results, 'alternative-r8')
        result = dict(schema='mobile-native-pocket-finite-comparison-v1', complete=True,
            campaign=str(root), protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'),
            freeze_sha256=sha(root/'freeze.json'), analyzer_sha256=sha(__file__),
            physical_config=config, shape_sha256=shape_sha, wall_certificate=wall,
            native_definition=classifier_binding, original_native_r4=original, regions=results,
            cumulative_r8=cumulative, shell_fraction_of_r8=independent_shell_fraction(*results),
            finite_region_contrasts=[contrast(original, r) for r in results+[cumulative]], scope=SCOPE)
        sources = out/'provenance';sources.mkdir()
        for name, source in local_sources(__file__).items():shutil.copy2(source, sources/name)
        result['analyzer_source_sha256'] = {p.name:sha(p) for p in sources.iterdir()}
        write(out/'analysis.json', result)
        (out/'report.md').write_text(report(result))
        plot(result, out)
        write(out/'freeze.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        print(json.dumps(dict(complete=True, analysis_sha256=sha(out/'analysis.json'), out=str(out))))
        return result
    except Exception as error:
        write(out/'failure.json', dict(complete=False, error=str(error), scope='No simulation or raw audit launched; partial outputs preserved.'))
        raise


def verify_initial_comparison(path):
    """Reuse immutable saved estimates/labels, without replaying old observers."""
    path = Path(path).resolve()
    if path.is_dir():path = path/'analysis.json'
    require(sha(path) == INITIAL_COMPARISON_SHA, 'Initial completed comparison identity changed')
    root = path.parent
    for name, digest in read(root/'freeze.json').items():
        require(sha(inside(root, name)) == digest, 'Initial comparison artifact changed: '+name)
    result = read(path)
    require(result['complete'] and result['schema'] == 'mobile-native-pocket-finite-comparison-v1', 'Initial comparison is incomplete')
    campaign = Path(result['campaign']).resolve()
    protocol, status = validate_parent(campaign)
    require(protocol.get('design', 'initial') == 'initial', 'Expected the initial two-region campaign')
    for name, key in [('protocol.json', 'protocol_sha256'), ('status.json', 'status_sha256'), ('freeze.json', 'freeze_sha256')]:
        require(sha(campaign/name) == result[key], 'Initial campaign binding changed')
    require({r['name'] for r in result['regions']} == set(ARMS) and len(result['regions']) == 2, 'Initial regions differ')
    for region in result['regions']:
        folder = Path(region['campaign']).resolve()
        require(folder == campaign/region['name'], 'Initial arm path differs')
        require(sha(folder/'manifest.json') == region['manifest_sha256']
                and sha(folder/'assessment/analysis.json') == region['assessment_sha256']
                == status['audits'][region['name']]['analysis_sha256'], 'Initial completed audit/manifest changed')
        require(sha(folder/'provenance/region.json') == region['region_sha256'], 'Initial finite region changed')
        require(sha(inside(root, region['native_labels']['path'])) == region['native_labels']['sha256'], 'Initial saved native labels changed')
        for source, digest in region['source_sha256'].items():
            require(sha(source) == digest, 'Initial hash-bound saved population changed')
    original = result['original_native_r4']
    for source, digest in original['source_sha256'].items():
        require(sha(source) == digest, 'Original finite native reference changed')
    return result, dict(path=str(path), analysis_sha256=sha(path), freeze_sha256=sha(root/'freeze.json'),
        reuse='Saved initial core, independent 5–8 shell, original native R4 and native labels. No old raw audit or classifier replay.')


def validate_disjoint_strata(regions):
    """Require an exact radial partition in one chart and physical target."""
    require(len(regions) >= 2, 'A radial partition requires at least two regions')
    reference = regions[0]
    end = 0.
    same = ('gaussian_chart', 'fixed_neighbor', 'physical_fixed_neighbors', 'capture_center', 'capture_radius',
            'shape_sha256', 'activity', 'depletant_radius', 'physical_metric', 'minimum_original_q')
    for region in regions:
        require(all(region[key] == reference[key] for key in same), 'Disjoint strata have different physical charts/targets')
        require(region.get('minimum_original_q_inclusive', True) is False and 'maximum_original_q' not in region,
                'Strata differ from original q>1')
        require(region.get('minimum_mahalanobis_radius', 0.) == end and region['mahalanobis_radius'] > end,
                'Radial strata overlap or leave a gap')
        end = region['mahalanobis_radius']
    return end


def coverage_totals(strata):
    """Add independently estimated disjoint integrals with unequal budgets."""
    require([r['name'] for r in strata] == ['r5repeat', 'shell5to8', 'shell8to12', 'shell12to16', 'shell16to24', 'shell24to32'],
            'Coverage sum must use the new core exactly once, with the independent old 5–8 shell')
    all_seeds = [p['seed'] for r in strata for p in r['populations']]
    require(len(all_seeds) == len(set(all_seeds)), 'Stratum populations are not independent')
    radii = [8, 12, 16, 24, 32]
    cumulative = []
    for stop, radius in enumerate(radii, 2):
        total = sum_regions(strata[:stop], f'alternative-r{radius}')
        inner = strata[0] if stop == 2 else cumulative[-1]
        total['new_outer_shell_fraction'] = independent_shell_fraction(inner, strata[stop-1])
        total['population_counts_by_region'] = {r['name']:len(r['populations']) for r in strata[:stop]}
        total['attempted_draws_by_region'] = {r['name']:r['row_uncertainty']['draws'] for r in strata[:stop]}
        cumulative.append(total)
    by_stratum = {}
    for i, region in enumerate(strata):
        rest = sum_regions([r for j, r in enumerate(strata) if j != i], 'remaining-disjoint-strata')
        by_stratum[region['name']] = independent_shell_fraction(rest, region)
    outside_r8 = sum_regions(strata[2:], 'shell8to32')
    return dict(cumulative_regions=cumulative, cumulative_r32=cumulative[-1],
        stratum_fractions_of_r32=by_stratum, outside_r8=outside_r8,
        outside_r8_fraction_of_r32=independent_shell_fraction(cumulative[0], outside_r8),
        scope='New R5 repeat + old independent 5–8 shell + four new disjoint shells. Initial R5 is never pooled or added. '
              'Each stratum retains its own row denominator and its own four or eight population means.')


def coverage_report(result):
    lines = ['Independent finite-pocket coverage follow-up.', '',
        'The cumulative estimate uses the new core once, the old independent 5–8 shell, and four new outer shells. '
        'The initial core remains a separate repeat comparison.', '',
        '| Region | populations × draws | log Qz | row RSE | population RSE | log Q0 | log(Qz/Q0) ± paired row SE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for region in [result['initial_core']]+result['strata']:
        a, b = region['row_uncertainty'], region['population_uncertainty']
        count = len(region['populations']);budget = a['draws']//count
        if a['log_Qz'] is None:
            lines.append(f"| {region['name']} | {count} × {budget} | unresolved | — | — | — | — |")
        else:
            lines.append(f"| {region['name']} | {count} × {budget} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {a['log_Q0']:.6f} | {a['log_enhancement']:.6f} ± {a['log_enhancement_SE']:.6f} |")
    c = result['independent_core_repeat_comparison']
    if 'unresolved' not in c['row_uncertainty']:
        a, b = c['row_uncertainty'], c['population_uncertainty']
        lines += ['', f"New / initial R5: log ratio {a['log_Qz_ratio']:.6f} ± {a['log_Qz_ratio_SE']:.6f} row SE "
                    f"or ± {b['log_Qz_ratio_SE']:.6f} population SE. This repeat comparison uses disjoint seeds; agreement does not prove tail coverage."]
    lines += ['', '| Cumulative finite region | log Qz | row RSE | population RSE | newest shell fraction ± row SE |',
              '|---|---:|---:|---:|---:|']
    for region in result['coverage']['cumulative_regions']:
        a, b = region['row_uncertainty'], region['population_uncertainty']
        f = region['new_outer_shell_fraction']['row_uncertainty']['Qz']
        if a['log_Qz'] is None:
            lines.append(f"| {region['name']} | unresolved | — | — | — |")
        else:
            lines.append(f"| {region['name']} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {f['observed_fraction']:.2%} ± {f['SE']:.2%} |")
    f = result['coverage']['outside_r8_fraction_of_r32']
    if f['row_uncertainty']['Qz']['observed_fraction'] is not None:
        lines += ['', f"R8–32 contributes {f['row_uncertainty']['Qz']['observed_fraction']:.2%} of observed R32 mass "
                    f"(row SE {f['row_uncertainty']['Qz']['SE']:.2%}; population SE {f['population_uncertainty']['Qz']['SE']:.2%})."]
    lines += ['', '| Region | native-entry weight | both-anchor weight | registry-triangle weight | motif 7/4 triangle weight |',
              '|---|---:|---:|---:|---:|']
    for region in result['strata']:
        columns = []
        for name in ('native_any', 'both_anchor_entry', 'registry_triangle', 'motif7_4_triangle'):
            value = region['native_diagnostics'][name]['fraction_of_parent']['Qz']['row_uncertainty']['observed_fraction']
            columns.append('unresolved' if value is None else f'{value:.2%}')
        lines.append('| '+region['name']+' | '+' | '.join(columns)+' |')
    lines += ['', 'Native labels are stateless reporting observables, not additional target restrictions. All motif IDs and '
              'supporting matches are saved per applicable pose. Fraction covariance, paired-cloud variance, individual population '
              'estimates, and nested new-core R3/R4/4–5 estimates are retained in analysis.json.', '',
              'Each shell has its own unconditional sample denominator. Small observed outer contributions with large errors '
              'do not certify missing-tail mass, and relative-error agreement does not establish thermodynamic convergence. '
              'Original-native comparisons remain ratios of named finite regions.', '', result['scope'], '']
    return '\n'.join(lines)


def plot_coverage(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    regions = [result['initial_core']]+result['strata']
    cumulative = result['coverage']['cumulative_regions']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, values, title in [(axes[0], regions, 'Independently sampled finite strata'),
                              (axes[1], cumulative, 'Disjoint cumulative finite mass')]:
        for i, region in enumerate(values):
            for shift, field, marker, color in [(-.07, 'row_uncertainty', 'o', '#2459a6'),
                                               (.07, 'population_uncertainty', 's', '#c66223')]:
                value = region[field]
                if value['log_Qz'] is not None:
                    ax.errorbar(i+shift, value['log_Qz'], yerr=value['Qz_relative_SE'], fmt=marker,
                                color=color, capsize=3, label=field.replace('_', ' ') if i == 0 else None)
        ax.set_xticks(range(len(values)), [r['name'] for r in values], rotation=30, ha='right')
        ax.set_ylabel('log Qz');ax.set_title(title);ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Observed delta errors; finite R32 does not establish whole-basin coverage', fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'finite-pocket-coverage.png', dpi=170)
    fig.savefig(output/'finite-pocket-coverage.svg')
    plt.close(fig)


def analyze_coverage(campaign, out, initial_comparison):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh downstream output directory required')
    protocol, status = validate_parent(root)
    allocation, seed_base = protocol_design(protocol)
    require(protocol.get('design') == 'coverage', 'Expected coverage follow-up campaign')
    prior, prior_binding = verify_initial_comparison(initial_comparison)
    initial_root = Path(prior['campaign']).resolve()
    initial_protocol = read(initial_root/'protocol.json')
    for key in ('original_config_sha256', 'model_sha256', 'physical_executable_sha256', 'source_bundle_sha256'):
        require(protocol[key] == initial_protocol[key], 'Follow-up physical/source identity changed: '+key)
    require(sha(root/'provenance/source-config.json') == protocol['original_config_sha256']
            and sha(root/'provenance/source-model.json') == protocol['model_sha256'], 'Follow-up frozen physical input changed')
    classifier, classifier_binding = load_classifier(inside(root, protocol['native_definition']))
    require(classifier_binding['definition_sha256'] == prior['native_definition']['definition_sha256'], 'Native definition changed')
    entries = {c['arm']:c for c in protocol['campaigns']}
    definitions = {arm:read(Path(entry['path'])/'provenance/region.json') for arm, entry in entries.items()}
    config = read(root/'r5repeat/provenance/config.json')
    shape_path = root/'r5repeat/provenance/shape.json';shape_sha = sha(shape_path)
    require(classifier.fixed_poses == config['fixed_poses'] and classifier.shape_sha256 == shape_sha == prior['shape_sha256'],
            'Classifier/follow-up physical target changed')
    normalized = copy.deepcopy(config);normalized['shape'] = prior['physical_config']['shape']
    require(normalized == prior['physical_config'], 'Follow-up physical config changed beyond shape relocation')
    wall = validate_regions(definitions, config, read(shape_path), shape_sha, allocation)
    by_name = {r['name']:r for r in prior['regions']}
    require(entries['r5repeat']['region_sha256'] == by_name['r5']['region_sha256'], 'Core repeat target is not byte-identical')
    old_shell_definition = read(initial_root/'shell5to8/provenance/region.json')
    ordered_definitions = [definitions['r5repeat'], old_shell_definition]+[definitions[a] for a in list(allocation)[1:]]
    require(validate_disjoint_strata(ordered_definitions) == 32., 'Cumulative region does not terminate at R32')
    prior_seeds = [p['seed'] for r in prior['regions']+[prior['original_native_r4']] for p in r['populations']]
    new_seeds = [seed_base+1009*i for i in range(24)]
    require(len(set(prior_seeds+new_seeds)) == len(prior_seeds)+len(new_seeds), 'New/old reference populations share seeds')
    for entry in entries.values():
        manifest = read(Path(entry['path'])/'manifest.json')
        require(manifest['archive_sha256']['latent-region-normalizer'] == protocol['physical_executable_sha256']
                and manifest['archive_sha256']['source-bundle.json'] == protocol['source_bundle_sha256'],
                'Follow-up executable or source bundle changed')
    out.mkdir(parents=True)
    try:
        values = []
        for arm in allocation:
            print(f'Reaggregating saved {arm} weights and stateless native labels', flush=True)
            value, _ = load_arm(root, entries[arm], status, config, shape_sha, classifier, out, allocation, seed_base)
            values.append(value)
        strata = [values[0], by_name['shell5to8']]+values[1:]
        initial_core = copy.deepcopy(by_name['r5']);initial_core['name'] = 'initial-r5'
        strata[1] = copy.deepcopy(strata[1])
        for reused in (initial_core, strata[1]):
            label = reused['native_labels']
            source = Path(prior_binding['path']).parent/label['path']
            target = out/('reused-'+Path(label['path']).name)
            shutil.copy2(source, target)
            require(sha(target) == label['sha256'], 'Reused native label copy changed')
            label.update(path=target.name, source_path=str(source))
        comparison = contrast(values[0], initial_core)
        comparison['scope'] = 'Independent new versus initial estimates of the byte-identical finite R5 target. '
        comparison['scope'] += 'Eight new populations of 16384 versus four old populations of 8192; no pooling. Observed agreement is not a tail bound.'
        totals = coverage_totals(strata)
        result = dict(schema='mobile-native-pocket-coverage-comparison-v1', complete=True,
            campaign=str(root), protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'),
            freeze_sha256=sha(root/'freeze.json'), analyzer_sha256=sha(__file__),
            allocation=allocation, original_comparison=prior_binding, physical_config=config,
            shape_sha256=shape_sha, wall_certificate=wall, native_definition=classifier_binding,
            initial_core=initial_core, independent_core_repeat_comparison=comparison,
            original_native_r4=prior['original_native_r4'], strata=strata, coverage=totals,
            finite_region_contrasts=[contrast(prior['original_native_r4'], r) for r in [values[0], totals['cumulative_r32']]],
            scope=SCOPE.replace('alternative R5 and R8', 'alternative R5 and R32')+
                ' The old 5–8 shell is independently reused, not redrawn; the initial core is a separate comparison only.')
        sources = out/'provenance';sources.mkdir()
        for name, source in local_sources(__file__).items():shutil.copy2(source, sources/name)
        shutil.copy2(prior_binding['path'], sources/'initial-comparison.json')
        result['analyzer_source_sha256'] = {p.name:sha(p) for p in sources.iterdir()}
        write(out/'analysis.json', result)
        (out/'report.md').write_text(coverage_report(result))
        plot_coverage(result, out)
        write(out/'freeze.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        print(json.dumps(dict(complete=True, analysis_sha256=sha(out/'analysis.json'), out=str(out))))
        return result
    except Exception as error:
        write(out/'failure.json', dict(complete=False, error=str(error), scope='No simulation or raw audit launched; partial outputs preserved.'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'runs/mobile-native-pocket-campaign-20260921')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--original-preparation', type=Path, default=PREPARATION)
    parser.add_argument('--original-native', type=Path, default=PILOT/'native-r4')
    parser.add_argument('--initial-comparison', type=Path, default=ROOT/'runs/mobile-native-pocket-comparison-20260921/analysis.json')
    args = parser.parse_args()
    if read(args.campaign/'protocol.json').get('design') == 'coverage':
        analyze_coverage(args.campaign, args.out, args.initial_comparison)
    else:
        analyze(args.campaign, args.out, args.original_preparation, args.original_native)
