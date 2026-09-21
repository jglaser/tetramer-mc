#!/usr/bin/env python3
"""Remask historical AB far draws into the newly frozen local peak geometry.

No physical draws, Poisson points, refits, or changed importance densities.
Historical selection makes these comparisons exploratory even though fresh
reference random streams are distinct.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_basin_normalizers import population as audit_population
from analyze_far_peak_reference import PHYSICAL, WINDOW, add_population_errors
from compare_intermediate_local_reference import read_batches
from analyze_global_fixed_region import region_latent
from analyze_peak_neighborhood import geometry_model_audit, geometry_rows, partition_check
from audit_shoulder_mis_independently import near
from prepare_cayley_rms_cover import read, write, sha, require
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments, quota_summary

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'runs/ab-far-peak-reference-preparation-20260920'
REFERENCE = ROOT/'runs/ab-far-peak-reference-assessment-20260920/analysis.json'
FLAT_ANALYSIS = ROOT/'runs/ab-far-capture-control-20260920/AB-analysis.json'
HISTORIES = (ROOT/'runs/ab-global-contact-4x8192-l64-20260920',
             ROOT/'runs/ab-global-widths-2-4-4x16384-l64-20260920')
ATOMS = ('radial_0', 'radial_1', 'radial_2', 'outside2')
MASKS = dict(full=ATOMS, ball0p5=ATOMS[:1], ball1=ATOMS[:2], ball2=ATOMS[:3],
             **{key: (key,) for key in ATOMS})
METRIC_KEYS = ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg')


def validate_runtime_metric(runtime_metric, metadata):
    """Runtime metric omits unrelated initialization/reporting metadata."""
    require(runtime_metric == {key: metadata[key] for key in METRIC_KEYS}, 'Physical original metric differs')


def selected_keys(radius):
    """Closed upper shell edges; the Cayley seam belongs to outside2."""
    require(radius >= 0 and not math.isnan(radius), 'Invalid geometric radius')
    atom = 'radial_0' if radius <= .5 else 'radial_1' if radius <= 1. else 'radial_2' if radius <= 2. else 'outside2'
    return tuple(key for key, atoms in MASKS.items() if atom in atoms)


def covariance_of_means(a, b, log_intersection_square):
    """Cov(mean A,mean B) = (sum AB - sum A sum B/N)/(N(N-1))."""
    require(a.count == b.count and a.count >= 2, 'Covariance requires original matching N')
    product = a.total+b.total-math.log(a.count)
    if log_intersection_square == product:
        return dict(sign=0, log_absolute_covariance=None)
    sign = 1 if log_intersection_square > product else -1
    high, low = max(log_intersection_square, product), min(log_intersection_square, product)
    value = high+(math.log(-math.expm1(low-high)) if low != -math.inf else 0.)
    return dict(sign=sign, log_absolute_covariance=value-math.log(a.count)-math.log(a.count-1))


class HistoryMoments:
    def __init__(self):
        self.keys = tuple(MASKS)
        self.values = {kind: {key: LogMoments() for key in MASKS} for kind in ('physical', 'hard')}

    def add(self, radius=None, physical=None, hard=None, cloud_pair=None):
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        keys = () if physical is None else selected_keys(radius)
        if physical is not None:
            require(math.isfinite(physical) and math.isfinite(hard), 'Finite original weights required')
            require(cloud_pair is not None and len(cloud_pair) == 2 and all(math.isfinite(x) for x in cloud_pair), 'Original cloud pair required')
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in MASKS:
                include = key in keys
                self.values[kind][key].add(value if include else -math.inf, cloud_pair if include and kind == 'physical' else None)

    def merge(self, other):
        for kind in self.values:
            for key in MASKS: self.values[kind][key].merge(other.values[kind][key])

    def report(self):
        result = {}; covariance = {}; checks = {}
        for kind, values in self.values.items():
            result[kind] = {}
            for key, value in values.items():
                row = quota_summary([value]); row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = 'Original-density IID row variance / full unconditional N, retaining all non-far and invalid zeros'
                result[kind][key] = row
            covariance[kind] = {}
            for left in MASKS:
                covariance[kind][left] = {}
                for right in MASKS:
                    shared = set(MASKS[left]).intersection(MASKS[right])
                    log_square = float(logsumexp([values[key].square for key in shared])) if shared else -math.inf
                    covariance[kind][left][right] = covariance_of_means(values[left], values[right], log_square)
            checks[kind] = dict(full=partition_check(values, ATOMS),
                ball2=partition_check(dict(full=values['ball2'], **{k: values[k] for k in ATOMS[:3]}), ATOMS[:3]),
                ball1=partition_check(dict(full=values['ball1'], **{k: values[k] for k in ATOMS[:2]}), ATOMS[:2]),
                ball2_vs_outside=partition_check(values, ('ball2', 'outside2')))
        return dict(**result, same_row_covariances=covariance, partition_checks=checks)


def independent_comparison(historical, fresh):
    """Observed errors for distinct draws; selection precludes nominal testing."""
    a, b = historical['logQ'], fresh['logQ']
    result = dict(historical=historical, fresh=fresh, historical_minus_fresh_logQ=None,
        historical_to_fresh_ratio=None, linear_difference_in_combined_row_SE=None,
        linear_difference_in_combined_population_SE=None,
        scope='Distinct random streams, but the chart was chosen from these historical data. Observed-error comparison is exploratory, not a nominal significance test or tail certificate.')
    if a is None or b is None: return result
    offset = max(a, b); ha, fb = math.exp(a-offset), math.exp(b-offset)
    result['historical_minus_fresh_logQ'] = a-b; result['historical_to_fresh_ratio'] = math.exp(a-b)
    for source, destination in [('row_RSE', 'linear_difference_in_combined_row_SE'),
                                 ('independent_population_RSE', 'linear_difference_in_combined_population_SE')]:
        if historical[source] is None or fresh[source] is None: continue
        error = math.hypot(ha*historical[source], fb*fresh[source])
        result[destination] = (ha-fb)/error if error else None
    return result


def flat_visit_rate(draws, density, hard, moment=None):
    """Plug-in independent-proposal hit rate, not a physical mixing time."""
    require(type(draws) is int and draws > 0 and math.isfinite(density) and density > 0, 'Invalid independent proposal budget/law')
    require(hard['logQ'] is not None, 'Local hard volume unresolved')
    volume = math.exp(hard['logQ']); p = density*volume
    require(0 < p < 1, 'Invalid estimated hit probability')
    pzero = math.exp(draws*math.log1p(-p)); rse = hard['row_RSE']
    result = dict(unconditional_draws=draws, constant_physical_proposal_density=density,
        local_hard_volume_A3=volume, local_hard_volume_observed_SE_A3=volume*rse,
        local_hard_volume_row_RSE=rse, local_hard_volume_population_RSE=hard['independent_population_RSE'],
        estimated_probability_per_draw=p, expected_hits=draws*p, expected_hits_observed_SE=draws*p*rse,
        probability_zero_hits=pzero, zero_probability_delta_method_SE=pzero*draws*p*rse/(1-p),
        mean_independent_draws_to_hit=1/p, mean_wait_delta_method_SE=rse/p,
        scope='Plug-in estimate from the independent local hard-volume shell sum. Errors propagate observed hard-volume uncertainty; no rigorous unseen-tail bound or MCMC mixing-time interpretation.')
    if moment is not None:
        require(np.linalg.eigvalsh(moment).min() > 0, 'Positive rotational moment required')
        logj0 = -math.log(8*math.pi**2)-.5*np.linalg.slogdet(moment)[1]
        volume_bound = math.exp(logj0)*math.pi**3/6*2**6
        pbound = density*volume_bound; require(0 < pbound < 1, 'Invalid geometric probability bound')
        result['geometric_upper_bound'] = dict(radius_A=2., maximum_physical_jacobian=math.exp(logj0),
            physical_volume_A3=volume_bound, probability_per_independent_draw=pbound,
            expected_hits=draws*pbound, probability_zero_hits_lower_bound=math.exp(draws*math.log1p(-pbound)),
            derivation='J(c)=J0/(1+|c|^2)^2 <= J0; multiply J0 by volume of the latent six-ball of radius2. This bounds the whole chart ball, before hard/q restrictions.',
            numerical_scope='Analytic inequality evaluated in FP64; not a formal interval-arithmetic certificate. No physical-weight sampling assumption.')
    return result


def load_flat(cfg, model, geometry, reference):
    """Hash-verified prior density audit plus new local masks; no replay."""
    record = read(FLAT_ANALYSIS); audit_path = Path(record['full_density_audit_path']); audited = read(audit_path); root = audit_path.parent
    require(record['complete'] and record['q_window'] == WINDOW and sha(audit_path) == record['full_density_audit_sha256'], 'Flat reference audit changed')
    require(audited['complete'] and audited['all_rows_and_hashes_validated'] and audited['independent_complete_cover_validated'] and audited['q_window'] == WINDOW, 'Flat original audit incomplete')
    master = read(root/'manifest.json'); config = read(root/'provenance/config.json')
    flat_protocol_path = FLAT_ANALYSIS.parent/'protocol.json'; flat_protocol = read(flat_protocol_path)
    require(sha(flat_protocol_path) == record['protocol_sha256'], 'Flat frozen protocol differs')
    require(flat_protocol['protein']['root'] == str(root) and [j['seed'] for j in master['jobs']] == flat_protocol['protein']['seeds'], 'Flat frozen streams differ')
    for name, digest in master['archive_sha256'].items(): require(sha(root/'provenance'/name) == digest, 'Flat archived input changed')
    require(all(config[k] == cfg[k] for k in PHYSICAL) and master['shape_sha256'] == model['shape_sha256'], 'Flat physical target differs')
    require(master['q_window'] == WINDOW and master['total_unconditional_draws'] == 262144, 'Flat frozen window or budget changed')
    guide = audited['guide']; cover = audited['cover_mixture']
    require(guide['weight'] == .99 and guide['uniform_probability'] == 1. and guide['cube_lengths'] == [36.]*3 and guide['capture_center'] == cfg['capture_center'], 'Flat cube law changed')
    require(cover['weights'] == [1.] and cover['scales'] == [1.] and len(cover['covers']) == 1, 'Flat cover law changed')
    body = cover['covers'][0]
    require(body['ball_radius'] == 74. and body['angle_cap'] == math.pi and body['centroid'] == [0.]*3 and body['reference'] == dict(position=[0.]*3, orientation=[1., 0., 0., 0.]), 'Flat complete ball/Haar cover changed')
    near(body['volume'], 4*math.pi/3*74**3, 1e-8)
    # r<=2 bounds translation displacement by 2Å, so the entire frozen local
    # ball lies in both flat branches, not just its observed valid samples.
    center = geometry[2]; require(np.max(abs(center))+2 < 18 and np.linalg.norm(center)+2 < 74, 'Local support not inside both flat branches')
    density = .99/36**3+.01/(4*math.pi/3*74**3)
    sources = {str(p): sha(p) for p in [FLAT_ANALYSIS, audit_path, root/'manifest.json', flat_protocol_path]}
    aggregate = HistoryMoments(); populations = []; geometry_counts = {k: 0 for k in ('ball0p5', 'ball1', 'ball2')}
    inside_q_capture_counts = dict(geometry_counts); errors = dict(matrix_radius_relative_error=0., geometric_radius_relative_error=0., member_rms2=0.)
    for job in master['jobs']:
        directory = Path(job['output']); path = directory/'samples.jsonl'; run = read(directory/'manifest.json')
        population_audit = next(p for p in audited['populations'] if p['replicate'] == directory.name)
        summary_path = directory/'summary.json'; summary = read(summary_path)
        require(sha(summary_path) == population_audit['summary_sha256'] and summary['complete'], 'Flat original summary differs')
        require(summary['samples_sha256'] == population_audit['sample_sha256'] == record['sample_sha256'][str(path)], 'Flat audited row hashes differ')
        sources[str(summary_path)] = sha(summary_path)
        require(run['seed'] == job['seed'] and run['samples'] == 65536 and run['q_window'] == WINDOW, 'Flat population allocation differs')
        require(run['guide'] == guide and run['cover_mixture'] == cover, 'Flat population geometry/law differs')
        validate_runtime_metric(run['metric'], cfg['metadata'])
        require(run['config_sha256'] == master['config_sha256'] and run['shape_sha256'] == model['shape_sha256'] and run['executable_sha256'] == master['binary_sha256'], 'Flat runtime provenance differs')
        sources[str(directory/'manifest.json')] = sha(directory/'manifest.json')
        local = HistoryMoments(); digest = hashlib.sha256(); n = 0; local_geometry = dict.fromkeys(geometry_counts, 0); local_capture = dict(local_geometry)
        for lines, rows in read_batches(path):
            for line in lines: digest.update(line)
            poses = [r['pose'] for r in rows]
            _, radii, diagnostic = region_latent(poses, dict(gaussian_chart=model, fixed_neighbor=cfg['fixed_poses'][0]))
            positions = np.asarray([p['position'] for p in poses]); quats = np.asarray([p['orientation'] for p in poses])
            independent_radii, _, current, _ = geometry_rows(positions, Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix(), geometry)
            finite = np.isfinite(radii) & np.isfinite(independent_radii)
            require(np.array_equal(np.isfinite(radii), np.isfinite(independent_radii)), 'Flat geometry seam mismatch')
            error = float(np.max(abs(radii[finite]-independent_radii[finite])/(1+radii[finite]))) if finite.any() else 0.
            require(error < 2e-7, 'Flat independent geometric radius mismatch')
            errors['matrix_radius_relative_error'] = max(errors['matrix_radius_relative_error'], diagnostic['matrix_radius_relative_error'])
            errors['geometric_radius_relative_error'] = max(errors['geometric_radius_relative_error'], error)
            errors['member_rms2'] = max(errors['member_rms2'], current['member_rms2'])
            for key, boundary in [('ball0p5', .5), ('ball1', 1.), ('ball2', 2.)]:
                require(np.array_equal(radii <= boundary, independent_radii <= boundary), 'Flat local masks disagree')
            for i, row in enumerate(rows):
                require(row['draw'] == n, 'Flat original ordering changed'); n += 1
                keys = selected_keys(float(independent_radii[i]))
                for key in local_geometry:
                    if key in keys:
                        local_geometry[key] += 1
                        if row.get('zero') not in ('q', 'capture'): local_capture[key] += 1
                if row.get('zero') is not None:
                    require(row.get('log_importance_weight') is None, 'Flat zero row carries weight'); local.add(); continue
                require(5 <= row['q'] < 37, 'Flat positive target differs')
                near(row['log_proposal_density'], math.log(density)); near(row['log_hard_weight'], -row['log_proposal_density'])
                pair = [value+row['log_hard_weight'] for value in row['cloud_log_weights']]
                require(len(pair) == 2, 'Flat paired-cloud record changed')
                near(float(logsumexp(pair))-math.log(2), row['log_importance_weight'])
                local.add(float(independent_radii[i]), row['log_importance_weight'], row['log_hard_weight'], pair)
        require(n == run['samples'] and digest.hexdigest() == record['sample_sha256'][str(path)], 'Flat original rows or full N changed')
        sources[str(path)] = digest.hexdigest(); aggregate.merge(local)
        for key in geometry_counts: geometry_counts[key] += local_geometry[key]; inside_q_capture_counts[key] += local_capture[key]
        populations.append(dict(id=directory.name, seed=run['seed'], samples=n, **local.report(), geometric_hits=local_geometry, q_capture_geometric_hits=local_capture))
    result = add_population_errors(aggregate.report(), populations, aggregate.keys)
    require(result['physical']['full']['draws'] == audited['samples'] == record['independent_q_and_density_rows'], 'Flat total denominator changed')
    for kind in ('physical', 'hard'):
        near(result[kind]['full']['logQ'], record[kind]['log_normalizer']); near(result[kind]['full']['row_RSE'], record[kind]['relative_SE'])
        full = result[kind]['full']['logQ']
        for item in result[kind].values(): item['fraction_of_observed_far_mass'] = math.exp(item['logQ']-full) if item['logQ'] is not None else None
    return dict(root=str(root), label='flat_complete_capture', **result, populations=populations,
        geometric_hits=geometry_counts, q_capture_geometric_hits=inside_q_capture_counts,
        hard_valid_geometric_hits={key: result['hard'][key]['nonzero'] for key in geometry_counts},
        geometry_audit_errors=errors, original_full_density_audit_sha256=record['full_density_audit_sha256'],
        local_R2_visit_rate=flat_visit_rate(audited['samples'], density, reference['independent_shell_sum']['hard']['full'], geometry[4]),
        CPU_seconds=record['CPU_seconds'], prior_audit_reused_without_kernel_replay=True), sources


def load_history(root, model, cfg, geometry, shape_sha, candidate_record):
    master = read(root/'manifest.json'); original = read(root/'assessment/analysis.json')
    require(not original['pending'], 'Historical campaign incomplete')
    for name, digest in master['archive_sha256'].items(): require(sha(root/'provenance'/name) == digest, f'Historical archive changed: {name}')
    require(master['proposal_anchor_index'] == 0, 'Historical proposal does not use physical A frame')
    require(master['archive_sha256']['shape.json'] == shape_sha, 'Historical shape differs')
    config = read(root/'provenance/config.json')
    require(all(config[k] == cfg[k] for k in PHYSICAL), 'Historical physical AB target differs')
    region = dict(gaussian_chart=model, fixed_neighbor=cfg['fixed_poses'][0])
    grouped = {}; hashes = {str(root/'manifest.json'): sha(root/'manifest.json'), str(root/'assessment/analysis.json'): sha(root/'assessment/analysis.json')}
    reference_model = candidate_record['archived_sha256']['old-mixture-model.json']
    require(master['archive_sha256']['model.json'] == reference_model, 'Historical frozen 119-component proposal changed')
    for job in master['jobs']:
        directory = Path(job['directory']); sample_path = directory/'samples.jsonl'
        for name in ('samples.jsonl', 'manifest.json', 'summary.json'):
            path = directory/name; hashes[str(path)] = sha(path)
            require(hashes[str(path)] == original['provenance'][str(path)], 'Historical audited source changed')
        audited = audit_population(job); rows = audited.pop('rows'); run = audited['summary']['manifest']
        for field, name in [('model_sha256', 'model.json'), ('config_sha256', 'config.json'), ('shape_sha256', 'shape.json'), ('executable_sha256', 'basin-normalizer')]:
            require(run[field] == master['archive_sha256'][name], f'Historical population input changed: {field}')
        require(run['activity'] == cfg['reservoir_density'] and run['lambda'] == 64*run['activity'] and run['cloud_replicates'] == 2, 'Historical cloud law differs')
        require(run['schema'] == 2 and run['proposal_anchor_index'] == 0 and run['uniform_probability'] == .1, 'Historical proposal law differs')
        width = job['covariance_std_scale']; local = HistoryMoments()
        group = grouped.setdefault(width, dict(aggregate=HistoryMoments(), populations=[], CPU_seconds=0., top=[],
            maximum_member_rms2_error=0., maximum_radius_relative_error=0., checked_far_poses=0,
            matrix_radius_relative_error=0., quaternion_seam_fallback_count=0, maximum_original_q=None))
        far_indices = [i for i, row in enumerate(rows) if row['hard_valid'] and row['q'] >= 5.]
        radii = {}
        if far_indices:
            poses = [rows[i]['pose'] for i in far_indices]
            _, chart_radii, diagnostics = region_latent(poses, region)
            positions = np.asarray([p['position'] for p in poses]); quats = np.asarray([p['orientation'] for p in poses])
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            geometric_radii, _, errors, _ = geometry_rows(positions, rotations, geometry)
            finite = np.isfinite(geometric_radii) & np.isfinite(chart_radii)
            require(np.array_equal(np.isfinite(geometric_radii), np.isfinite(chart_radii)), 'Chart seam mismatch')
            relative_error = float(np.max(np.abs(geometric_radii[finite]-chart_radii[finite])/(1+chart_radii[finite]))) if finite.any() else 0.
            require(relative_error < 2e-7, 'Independent geometric radius mismatch')
            # No tolerance changes the frozen ball masks.
            for boundary in (.5, 1., 2.): require(np.array_equal(geometric_radii <= boundary, chart_radii <= boundary), 'Independent geometric masks disagree')
            radii = dict(zip(far_indices, map(float, geometric_radii)))
            group['maximum_member_rms2_error'] = max(group['maximum_member_rms2_error'], errors['member_rms2'])
            group['maximum_radius_relative_error'] = max(group['maximum_radius_relative_error'], relative_error)
            group['matrix_radius_relative_error'] = max(group['matrix_radius_relative_error'], diagnostics['matrix_radius_relative_error'])
            group['quaternion_seam_fallback_count'] += diagnostics['quaternion_seam_fallback_count']; group['checked_far_poses'] += len(poses)
        for i, row in enumerate(rows):
            if row['hard_valid']:
                require(row['q'] < 37., 'Observed hard pose exceeds frozen complete-capture q bound')
                if group['maximum_original_q'] is None or row['q'] > group['maximum_original_q']: group['maximum_original_q'] = row['q']
            if i not in radii:
                local.add(); continue
            require(row['capture_valid'] and 5 <= row['q'] < 37, 'Historical far target changed')
            near(row['log_hard_weight'], -row['log_proposal_density'])
            pair = [c['log_weight']-row['log_proposal_density'] for c in row['clouds']]
            local.add(radii[i], row['log_importance_weight'], row['log_hard_weight'], pair)
        local_result = local.report(); near(local_result['physical']['full']['logQ'], audited['estimates']['distant']['logQ'])
        for i in sorted(radii, key=lambda i: rows[i]['log_importance_weight'], reverse=True)[:8]:
            row = rows[i]
            group['top'].append(dict(population=job['id'], seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'],
                geometric_radius_A=radii[i], original_log_importance_weight=row['log_importance_weight'],
                component_index=row['proposal'].get('component_index'), source_samples_path=str(sample_path), source_samples_sha256=hashes[str(sample_path)]))
        group['aggregate'].merge(local)
        group['populations'].append(dict(id=job['id'], seed=job['seed'], samples=job['samples'], **local_result, original_proposal_audit=audited['proposal_audit']))
        group['CPU_seconds'] += audited['summary']['sampler_cpu_seconds']
        require(sha(sample_path) == hashes[str(sample_path)], 'Historical rows changed during reconstruction')
    groups = []
    for width, group in sorted(grouped.items()):
        aggregate = group.pop('aggregate'); top = group.pop('top'); result = add_population_errors(aggregate.report(), group['populations'], aggregate.keys)
        expected = original['groups'][str(width)]['estimates']['distant']; actual = result['physical']['full']
        near(actual['logQ'], expected['logQ']); near(actual['row_RSE'], expected['relative_se'])
        require(actual['draws'] == original['groups'][str(width)]['unconditional_draws'], 'Historical full N differs')
        for kind in ('physical', 'hard'):
            total = result[kind]['full']['logQ']
            for item in result[kind].values(): item['fraction_of_observed_far_mass'] = math.exp(item['logQ']-total) if item['logQ'] is not None and total is not None else None
        groups.append(dict(root=str(root), width=width, **group, **result,
            top8_far_contributors=sorted(top, key=lambda r: r['original_log_importance_weight'], reverse=True)[:8],
            original_far_estimate=expected))
    return groups, hashes


def compare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Use a fresh comparison directory')
    preparation = PREPARATION; protocol = read(preparation/'protocol.json'); frozen = read(preparation/'freeze.json')
    for name, digest in frozen.items(): require(sha(preparation/name) == digest, 'Frozen peak preparation changed')
    for name, digest in protocol['archived_sha256'].items(): require(sha(preparation/'provenance'/name) == digest, 'Peak preparation archive changed')
    reference = read(REFERENCE); require(reference['complete'] and reference['original_q_window'] == WINDOW, 'Fresh local audit incomplete or wrong q mask')
    for path, digest in reference['input_sha256'].items(): require(sha(Path(path)) == digest, 'Fresh reference audit input changed')
    for name, digest in reference['archived_sha256'].items(): require(sha(REFERENCE.parent/'provenance'/name) == digest, 'Fresh audit source archive changed')
    for campaign in reference['campaigns']:
        for path, digest in campaign['sample_sha256'].items(): require(sha(Path(path)) == digest, 'Fresh reference samples changed')
        require(sha(Path(campaign['root'])/'manifest.json') == campaign['manifest_sha256'], 'Fresh manifest changed')
    cfg = read(preparation/'config.json'); model = read(preparation/'model.json'); selected = read(preparation/'selected-pose.json')
    candidate_record = read(preparation/'provenance/candidates.json')
    require(sha(preparation/'provenance/candidates.json') == reference['selection_audit']['candidate_sha256'], 'Frozen historical selection differs')
    require(candidate_record['candidates'][candidate_record['selected_peak']] == selected, 'Frozen historical center changed')
    report, geometry = geometry_model_audit(model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, selected['pose'])
    sources = {str(p): sha(p) for p in [REFERENCE, preparation/'protocol.json', preparation/'model.json', preparation/'config.json', preparation/'selected-pose.json', preparation/'provenance/candidates.json']}
    groups = []; seeds = {p['seed'] for c in reference['campaigns'] for p in c['populations']}
    for root in HISTORIES:
        historical, hashes = load_history(root, model, cfg, geometry, model['shape_sha256'], candidate_record)
        sources.update(hashes)
        for group in historical:
            for population in group['populations']:
                require(population['seed'] not in seeds, 'Fresh/historical random stream collision'); seeds.add(population['seed'])
            group['matched_reference_comparisons'] = {}
            for i, radius in enumerate((.5, 1., 2.)):
                ref = next(c for c in reference['campaigns'] if c['radius_A'] == radius)
                key = ('ball0p5', 'ball1', 'ball2')[i]
                group['matched_reference_comparisons'][key] = {kind: independent_comparison(group[kind][key], ref[kind]['full']) for kind in ('physical', 'hard')}
                shell = f'radial_{i}'
                group['matched_reference_comparisons'][shell] = {kind: independent_comparison(group[kind][shell], ref[kind][shell]) for kind in ('physical', 'hard')}
            group['matched_reference_comparisons']['ball2_independent_shell_sum'] = {kind: independent_comparison(
                group[kind]['ball2'], reference['independent_shell_sum'][kind]['full']) for kind in ('physical', 'hard')}
            groups.append(group)
    require([g['width'] for g in groups] == [1., 2., 4.], 'Unexpected historical width selection')
    flat, flat_hashes = load_flat(cfg, model, geometry, reference); sources.update(flat_hashes)
    for population in flat['populations']:
        require(population['seed'] not in seeds, 'Flat random stream collision'); seeds.add(population['seed'])
    flat['matched_reference_comparisons'] = {}
    for i, radius in enumerate((.5, 1., 2.)):
        ref = next(c for c in reference['campaigns'] if c['radius_A'] == radius); key = ('ball0p5', 'ball1', 'ball2')[i]
        flat['matched_reference_comparisons'][key] = {kind: independent_comparison(flat[kind][key], ref[kind]['full']) for kind in ('physical', 'hard')}
    for path, digest in sources.items(): require(sha(Path(path)) == digest, 'Input changed during comparison')
    (out/'provenance').mkdir(parents=True)
    inputs = {'reference-analysis.json': REFERENCE, 'peak-protocol.json': preparation/'protocol.json', 'peak-model.json': preparation/'model.json',
        'physical-config.json': preparation/'config.json', 'selected-pose.json': preparation/'selected-pose.json'}
    for name, path in {**local_dependencies([Path(__file__)]), **inputs}.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    scope = ('Historical q>=5 and fresh 5<=q<37 masks refer to the same complete capture domain; all observed hard historical poses satisfy q<37. '
        'Each historical width retains its original full 119-Gaussian-plus-cube density and every unconditional zero. No widths or nested balls are pooled. '
        'The geometric chart was selected from historical width-four extremes, so fresh random streams are distinct but comparisons remain exploratory after selection. '
        'Linear differences use observed independent-stream variance, never formal significance or an unseen-tail bound. Outside2 is a positive far-only mask, not generic nonnative mass. '
        'No refitting, new physical draws or new Poisson clouds. Importance ESS and original CPU do not measure equilibrium mixing.')
    result = dict(complete=True, original_q_window=WINDOW, radial_intervals_A=[[0., .5], [.5, 1.], [1., 2.], [2., None]],
        boundary_rule='First lower edge closed; finite upper edges closed; subsequent lower edges open. Positive infinite Cayley radius is outside2.',
        geometric_model_sha256=sha(preparation/'model.json'), source_sha256=sources,
        archived_sha256={p.name: sha(p) for p in (out/'provenance').iterdir()}, geometry_reconstruction=report,
        selection_audit=reference['selection_audit'], fresh_reference_analysis_sha256=sha(REFERENCE), groups=groups, flat_control=flat, scope=scope)
    write(out/'analysis.json', result)
    def num(x): return 'unresolved' if x is None else f'{x:.6g}'
    lines = ['# Historical far mass in identical frozen local masks', '', '| Width | Mask | Full N | Positive | log Q | Row / population RSE | Fraction of observed far |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for group in groups:
        for key, row in group['physical'].items():
            lines.append(f"| {group['width']:g} | {key} | {row['draws']} | {row['nonzero']} | {num(row['logQ'])} | {num(row['row_RSE'])} / {num(row['independent_population_RSE'])} | {num(row['fraction_of_observed_far_mass'])} |")
    lines += ['', '| Width | Same ball | Historical log Q | Fresh log Q | Historical / fresh | Difference / combined row SE |', '| ---: | --- | ---: | ---: | ---: | ---: |']
    for group in groups:
        for key in ('ball0p5', 'ball1', 'ball2', 'ball2_independent_shell_sum'):
            c = group['matched_reference_comparisons'][key]['physical']
            lines.append(f"| {group['width']:g} | {key} | {num(c['historical']['logQ'])} | {num(c['fresh']['logQ'])} | {num(c['historical_to_fresh_ratio'])} | {num(c['linear_difference_in_combined_row_SE'])} |")
    lines += ['', 'The separate flat control retains all 262,144 unconditional draws.', '',
              '| Frozen ball | Geometric hits | q/capture hits | Hard-valid hits | log Q |', '| --- | ---: | ---: | ---: | ---: |']
    for key in ('ball0p5', 'ball1', 'ball2'):
        lines.append(f"| {key} | {flat['geometric_hits'][key]} | {flat['q_capture_geometric_hits'][key]} | {flat['hard_valid_geometric_hits'][key]} | {num(flat['physical'][key]['logQ'])} |")
    rate = flat['local_R2_visit_rate']; bound = rate['geometric_upper_bound']
    lines += ['', f"Local hard-volume plug-in predicts {rate['expected_hits']:.8g} flat valid R2 hits, with P(no hit)={rate['probability_zero_hits']:.8g}; mean independent wait {rate['mean_independent_draws_to_hit']:.8g} draws.",
              f"The analytic J<=J0 bound gives at most {bound['expected_hits']:.8g} expected hits of the entire R2 chart ball and P(no hit)>={bound['probability_zero_hits_lower_bound']:.8g}. Numerical values use FP64, not interval arithmetic.",
              'The plug-in rate propagates observed hard-volume uncertainty; neither rate is an MCMC mixing-time estimate.']
    lines += ['', scope, '', 'All positive partitions, full same-row covariance matrices, population results and source hashes are in analysis.json.', '']
    (out/'report.md').write_text('\n'.join(lines)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = compare(args.out)
    print(dict(complete=result['complete'], output=str(args.out/'analysis.json'),
        ball0p5={str(g['width']): g['matched_reference_comparisons']['ball0p5']['physical'] for g in result['groups']}))
