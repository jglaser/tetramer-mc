#!/usr/bin/env python3
"""Audit fixed AB far peak references and the prespecified disjoint-shell sum.

All masks retain the original unconditional uniform-ball denominator. The old
R3 chart is the unchanged fitted competitor chart, not a geometric RMS chart.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import hashlib
import heapq
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_latent_region import analyze as audit_original, original_q_window
from analyze_peak_neighborhood import geometry_model_audit, geometry_rows, partition_check
from audit_shoulder_mis_independently import chart_values, near
from compare_intermediate_local_reference import read_batches
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments, quota_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION = ROOT/'runs/ab-far-peak-reference-preparation-20260920'
WINDOW = dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False)
EDGES = (0., .5, 1., 2.)
OLD_KEYS = ('old_r_le_3', 'old_r_gt_3')
PHYSICAL = ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density')


def selected_keys(radius, old_radius, limit):
    require(limit in EDGES[1:] and math.isfinite(radius) and 0 <= radius <= limit, 'Outside frozen ball')
    require(math.isfinite(old_radius) and old_radius >= 0, 'Invalid old fitted-chart radius')
    radial = next(f'radial_{i}' for i, upper in enumerate(EDGES[1:]) if radius <= upper)
    old = OLD_KEYS[int(old_radius > 3.)]
    return ('full', old, radial, f'{radial}__{old}')


class MaskedMoments:
    def __init__(self, limit):
        require(limit in EDGES[1:], 'Unplanned ball radius')
        self.limit = limit; self.radial = tuple(f'radial_{i}' for i in range(EDGES.index(limit)))
        self.cross = tuple(f'{r}__{old}' for r in self.radial for old in OLD_KEYS)
        self.keys = ('full', *OLD_KEYS, *self.radial, *self.cross)
        self.values = {kind: {key: LogMoments() for key in self.keys} for kind in ('physical', 'hard')}

    def add(self, radius, old_radius, physical=None, hard=None, cloud_pair=None):
        selected = selected_keys(radius, old_radius, self.limit)
        require((physical is None) == (hard is None), 'Physical and hard zero masks differ')
        if physical is not None:
            require(math.isfinite(physical) and math.isfinite(hard), 'Invalid original weight')
            require(cloud_pair is not None and len(cloud_pair) == 2 and all(math.isfinite(x) for x in cloud_pair), 'Need original cloud pair')
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in self.keys:
                use = value is not None and key in selected
                self.values[kind][key].add(value if use else -math.inf, cloud_pair if use and kind == 'physical' else None)

    def merge(self, other):
        require(self.limit == other.limit, 'Do not pool different proposal radii')
        for kind in self.values:
            for key in self.keys: self.values[kind][key].merge(other.values[kind][key])

    def report(self):
        result = {}
        for kind, values in self.values.items():
            result[kind] = {}
            for key, value in values.items():
                row = quota_summary([value]); row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = 'Original IID row variance / full unconditional N; all masked and invalid zeros retained'
                result[kind][key] = row
        result['partition_checks'] = {kind: dict(old_chart=partition_check(values, OLD_KEYS),
            radial=partition_check(values, self.radial), cross_product=partition_check(values, self.cross),
            radial_old_chart={r: partition_check(dict(full=values[r], **{old: values[f'{r}__{old}'] for old in OLD_KEYS}), OLD_KEYS)
                              for r in self.radial}) for kind, values in self.values.items()}
        return result


def add_population_errors(result, populations, keys):
    require(len({p['samples'] for p in populations}) == 1, 'Population errors require equal fixed budgets')
    for kind in ('physical', 'hard'):
        for key in keys:
            row = result[kind][key]
            row['population_logQ_values'] = [p[kind][key]['logQ'] for p in populations]
            row['independent_population_RSE'] = population_rse(row['population_logQ_values'])
    return result


def sum_independent_estimates(rows):
    """Sum disjoint-region estimators from independent fixed random streams.

    An all-zero piece contributes an observed zero, never an upper bound.
    Row and population variances are independently summed, not averaged.
    """
    require(rows, 'Need prespecified shell estimates')
    finite = [r['logQ'] for r in rows if r['logQ'] is not None]
    logq = float(logsumexp(finite)) if finite else None
    variances = [r['log_variance_of_mean'] for r in rows if r['log_variance_of_mean'] is not None]
    logv = float(logsumexp(variances)) if variances else None
    pop_variances = [2*r['logQ']+2*math.log(r['independent_population_RSE']) for r in rows
                     if r['logQ'] is not None and r['independent_population_RSE'] is not None and r['independent_population_RSE'] > 0]
    popv = float(logsumexp(pop_variances)) if pop_variances else None
    return dict(logQ=logq, log_variance_of_mean=logv,
        row_RSE=math.exp(.5*logv-logq) if logq is not None and logv is not None else (0. if logq is not None else None),
        log_independent_population_variance=popv,
        independent_population_RSE=math.exp(.5*popv-logq) if logq is not None and popv is not None else (0. if logq is not None else None),
        total_unconditional_draws=sum(r['draws'] for r in rows), nonzero=sum(r['nonzero'] for r in rows),
        unresolved_zero_pieces=sum(r['logQ'] is None for r in rows),
        variance_rule='Sum independent shell means and their independent variances; not a pooled draw mean',
        coverage='Observed zero pieces are unresolved. Observed errors do not bound unseen weight.')


def combine_shells(campaigns):
    require(sorted(c['radius_A'] for c in campaigns) == list(EDGES[1:]), 'Need all three planned independent campaigns')
    ordered = sorted(campaigns, key=lambda c: c['radius_A']); result = {}; pieces = []
    for i, campaign in enumerate(ordered):
        radial = f'radial_{i}'
        pieces.append(dict(root=campaign['root'], radius_A=campaign['radius_A'], selected_shell=radial,
            interval=list(EDGES[i:i+2]), lower_inclusive=i == 0, upper_inclusive=True))
    for kind in ('physical', 'hard'):
        result[kind] = {}
        for key in ('full', *OLD_KEYS):
            rows = [c[kind][f'radial_{i}' if key == 'full' else f'radial_{i}__{key}'] for i, c in enumerate(ordered)]
            result[kind][key] = sum_independent_estimates(rows)
        covs = [c['partition_checks'][kind]['radial_old_chart'][f'radial_{i}']['covariances'][0] for i, c in enumerate(ordered)]
        logs = [r['log_absolute_covariance'] for r in covs if r['sign']]
        logcov = float(logsumexp(logs)) if logs else None
        covariance = dict(sign=-1 if logs else 0, log_absolute_covariance=logcov,
                         rule='Sum same-row covariances within independently sampled shells')
        full = result[kind]['full']; left, right = [result[kind][k] for k in OLD_KEYS]
        mass = [r['logQ'] for r in (left, right) if r['logQ'] is not None]
        if full['logQ'] is not None: near(float(logsumexp(mass)), full['logQ'])
        variance_logs = [r['log_variance_of_mean'] for r in (left, right, full) if r['log_variance_of_mean'] is not None]
        if logcov is not None: variance_logs.append(logcov)
        variance_error = None
        if variance_logs:
            offset = max(variance_logs)
            scaled = lambda v: 0. if v is None else math.exp(v-offset)
            reconstructed = sum(scaled(r['log_variance_of_mean']) for r in (left, right))-2*scaled(logcov)
            variance_error = abs(reconstructed-scaled(full['log_variance_of_mean']))
            require(variance_error < 2e-9, 'Combined old-chart covariance reconstruction failed')
        result[kind]['old_partition_audit'] = dict(covariance=covariance, scaled_variance_error=variance_error,
            mass_partition_checked=True, old_boundary=3.)
    return dict(**result, pieces=pieces, scope='Only the frozen geometric ball r<=2 intersected with 5<=q<37, capture and both hard neighbors. No old-R3 total or full-far estimate is added.')


def audit_selection(preparation, protocol):
    source = protocol['selection_source']; candidate_path = preparation/'provenance/candidates.json'
    require(sha(candidate_path) == source['sha256'] == sha(Path(source['path'])), 'Historical candidate audit changed')
    candidates = read(candidate_path); require(candidates['complete'], 'Candidate audit incomplete')
    index = source['selected_peak']; require(index == candidates['selected_peak'], 'Frozen peak index changed')
    selected = read(preparation/'selected-pose.json'); require(selected == candidates['candidates'][index], 'Peak record changed')
    require(selected['width'] == 4. and selected['original_log_importance_weight'] == max(
        r['original_log_importance_weight'] for r in candidates['candidates'] if r['width'] == 4.), 'Peak selection changed')
    original = read(preparation/'selected-original-row.json'); require(original == selected['original_row'], 'Original row snapshot changed')
    for key in ('samples', 'manifest', 'config'):
        require(sha(Path(selected[f'source_{key}_path'])) == selected[f'source_{key}_sha256'], 'Selected source hash changed')
    found = False
    for _, rows in read_batches(Path(selected['source_samples_path'])):
        for row in rows:
            if row['draw'] == selected['draw']:
                require(row == original, 'Selected original draw changed'); found = True
    require(found and original['hard_valid'] and original['capture_valid'] and 5 <= original['q'] < 37, 'Selected far pose missing or invalid')
    return selected, dict(candidate_sha256=source['sha256'], selected_peak=index,
        selected_original_row_sha256=sha(preparation/'selected-original-row.json'),
        original_samples_sha256=selected['source_samples_sha256'], original_manifest_sha256=selected['source_manifest_sha256'],
        seed=selected['seed'], draw=selected['draw'], population=selected['population'],
        scope='Frozen historical selection only; no historical row is reused as an independent local draw')


def audit_campaign(root, declared, model, old_region, cfg, selected_pose):
    root = Path(root).resolve(); manifest = read(root/'manifest.json'); region = read(root/'provenance/region.json')
    require(manifest['region_sha256'] == declared['region_sha256'] == sha(root/'provenance/region.json'), 'Frozen local region changed')
    require(original_q_window(region) == WINDOW and region['gaussian_chart'] == model, 'Far mask or geometric chart changed')
    require(region['minimum_mahalanobis_radius'] == 0. and region['mahalanobis_radius'] == declared['radius_A'], 'Frozen support changed')
    local_cfg = read(root/'provenance/config.json')
    require(all(local_cfg[k] == cfg[k] for k in PHYSICAL), 'Physical target differs')
    require(region['fixed_neighbor'] == cfg['fixed_poses'][0] == old_region['fixed_neighbor'], 'Physical A chart frame differs')
    require(len(manifest['jobs']) == declared['populations'] and [j['seed'] for j in manifest['jobs']] == declared['seeds'], 'Fixed seeds changed')
    require(all(j['samples'] == declared['samples_per_population'] for j in manifest['jobs']), 'Fixed budgets changed')
    # This generic auditor checks every original q, denominator, Poisson factor,
    # cloud pair, zero, inverse coordinate and manifest/hash using 5<=q<37.
    original = audit_original(root)
    geometry_report, geometry = geometry_model_audit(model, region, cfg, selected_pose)
    aggregate = MaskedMoments(declared['radius_A']); populations = []; hashes = {}; top = []; errors = dict(member_rms2=0., radius=0., jacobian=0.)
    atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses'])
    for pi, job in enumerate(manifest['jobs']):
        path = Path(job['directory'])/'samples.jsonl'; digest = hashlib.sha256(); count = 0
        local = MaskedMoments(declared['radius_A'])
        for lines, rows in read_batches(path):
            for line in lines: digest.update(line)
            positions = np.asarray([r['pose']['position'] for r in rows]); quats = np.asarray([r['pose']['orientation'] for r in rows])
            near(np.sum(quats*quats, axis=1), np.ones(len(rows)))
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            radii, jac, current, _ = geometry_rows(positions, rotations, geometry)
            _, old_radii, _ = chart_values(old_region['gaussian_chart'], positions, rotations, old_region['fixed_neighbor'])
            current['radius'] = near(radii, [r['latent_radius'] for r in rows]); current['jacobian'] = near(jac, [r['log_physical_jacobian'] for r in rows])
            for key in errors: errors[key] = max(errors[key], current[key])
            for i, row in enumerate(rows):
                require(row['draw'] == count, 'Row ordering changed'); count += 1
                weight, hard = row['log_importance_weight'], row['log_hard_weight']
                require((weight is None) == (not(row['capture_valid'] and row['hard_valid'] and row['region_valid'])), 'Invalid original positive/zero mask')
                pair = [hard+c['log_weight'] for c in row['clouds']] if weight is not None else None
                local.add(float(radii[i]), float(old_radii[i, 0]), weight, hard, pair)
                if weight is not None:
                    record = dict(population=job['id'], seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'],
                        geometric_radius_A=float(radii[i]), old_competitor_radius=float(old_radii[i, 0]),
                        original_log_importance_weight=weight, original_row=row, source_samples_path=str(path))
                    entry = (weight, -pi, -row['draw'], record)
                    if len(top) < 8: heapq.heappush(top, entry)
                    elif entry[:3] > top[0][:3]: heapq.heapreplace(top, entry)
        require(count == job['samples'], 'Unconditional denominator changed')
        require(digest.hexdigest() == original['populations'][pi]['samples_sha256'], 'Samples changed between audits')
        hashes[str(path)] = digest.hexdigest(); aggregate.merge(local)
        populations.append(dict(id=job['id'], seed=job['seed'], samples=count, samples_sha256=digest.hexdigest(), **local.report()))
    result = add_population_errors(aggregate.report(), populations, aggregate.keys)
    for kind, original_key in (('physical', 'estimate'), ('hard', 'hard_region')):
        actual, expected = result[kind]['full']['logQ'], original[original_key]['logQ']
        require(actual == expected if actual is None else expected is not None and abs(actual-expected) < 2e-8, 'Independent full-ball mass mismatch')
    top_rows = [item[-1] for item in sorted(top, key=lambda e: e[:3], reverse=True)]
    for row in top_rows:
        gaps = atom.gaps(row['pose']); require(min(gaps) >= 0, 'Largest positive row clashes atomic AB')
        row['minimum_AB_gaps_A'] = gaps; row['source_samples_sha256'] = hashes[row['source_samples_path']]
    return dict(root=str(root), radius_A=declared['radius_A'], **result, populations=populations,
        original_reference_audit=original, row_geometry_max_errors=errors, geometry_reconstruction=geometry_report,
        sample_sha256=hashes, manifest_sha256=sha(root/'manifest.json'), top8_atomic_checks=top_rows,
        CPU_seconds=original['sampler_cpu_seconds'])


def analyze(preparation, out):
    preparation = Path(preparation).resolve(); out = Path(out).resolve(); require(not out.exists(), 'Fresh audit directory required')
    protocol = read(preparation/'protocol.json'); freeze = read(preparation/'freeze.json'); report = read(preparation/'report.json')
    require(report['complete'] and report['freeze_sha256'] == sha(preparation/'freeze.json'), 'Preparation incomplete or changed')
    for name, digest in freeze.items(): require(sha(preparation/name) == digest, f'Frozen preparation changed: {name}')
    for name, digest in protocol['archived_sha256'].items(): require(sha(preparation/'provenance'/name) == digest, f'Preparation archive changed: {name}')
    plan = read(preparation/'analysis-plan.json')
    require(sha(preparation/'analysis-plan.json') == read(preparation/'analysis-plan-freeze.json')['analysis_plan_sha256'], 'Analysis plan changed')
    require(plan['preparation_protocol_sha256'] == sha(preparation/'protocol.json') and plan['frozen_before_physical_sampling'], 'Analysis plan not tied to frozen protocol')
    require(protocol['original_q_window'] == plan['physical_window'] == WINDOW and plan['radial_edges_A'] == list(EDGES), 'Planned physical window or shells changed')
    require(plan['shell_estimator_sources'] == [dict(region=list(EDGES[i:i+2]), campaign_radius_A=EDGES[i+1]) for i in range(3)], 'Shell source allocation changed')
    cfg = read(preparation/'config.json'); model = read(preparation/'model.json')
    old_path = preparation/'provenance/old-competitor-region.json'; old_region = read(old_path)
    require(sha(old_path) == protocol['old_region_sha256'], 'Old R3 partition changed')
    require(old_region['mahalanobis_radius'] == 3. and old_region['gaussian_chart'] == read(preparation/'provenance/old-competitor-model.json'), 'Old non-geometric chart changed')
    require(all(cfg[k] == protocol['physical'][k] for k in PHYSICAL), 'Protocol physical target changed')
    require(old_region['shape_sha256'] == model['shape_sha256'] == sha(preparation/'provenance/shape.json'), 'Old/new shape differs')
    require(old_region['physical_fixed_neighbors'] == cfg['fixed_poses'] and old_region['physical_metric'] == cfg['metadata'], 'Old partition physical geometry differs')
    for field, key in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                       ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius')]:
        require(old_region[field] == cfg[key], f'Old partition physical target differs: {field}')
    selected, selection = audit_selection(preparation, protocol)
    sources = {p.name: p for p in [preparation/'protocol.json', preparation/'freeze.json', preparation/'analysis-plan.json', preparation/'analysis-plan-freeze.json',
        preparation/'model.json', preparation/'config.json', preparation/'geometry.json', preparation/'report.json', preparation/'selected-pose.json',
        preparation/'selected-original-row.json', old_path, preparation/'provenance/candidates.json']}
    hashes = {str(p): sha(p) for p in sources.values()}; campaigns = []; seeds = set()
    for declared in protocol['campaigns']:
        require(not seeds.intersection(declared['seeds']), 'Independent shell random streams overlap'); seeds.update(declared['seeds'])
        master = read(Path(declared['output'])/'manifest.json')
        require(master['archive_sha256']['latent-region-normalizer'] == protocol['executable_sha256'], 'Frozen physical executable differs')
        require(master['lambda_ratio'] == protocol['lambda_ratio'] and master['cloud_replicates'] == protocol['cloud_replicates'], 'Frozen Poisson allocation differs')
        campaigns.append(audit_campaign(declared['output'], declared, model, old_region, cfg, selected['pose']))
    combined = combine_shells(campaigns)
    for path, digest in hashes.items(): require(sha(Path(path)) == digest, 'Analysis input changed during audit')
    (out/'provenance').mkdir(parents=True)
    for name, path in {**local_dependencies([Path(__file__)]), **sources}.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    scope = ('The three whole-ball estimates remain separate. Only prespecified disjoint shells from matching-radius independent campaigns are summed. '
        'All estimates retain original 5<=q<37, capture, both fixed hard neighbors, full unconditional N and physical center/Haar measure. '
        'Old fitted-chart R3 intersections and complements use positive same-row masks and covariance. Neither nested balls nor the old R3 total are added. '
        'Selected historical poses define the frozen region but are never reused as fresh observations. No global far coverage or equilibrium mixing is certified. '
        'Empty supported pieces are unresolved; row and population errors do not bound unseen tails. Top-eight all-atom checks supplement original hard flags.')
    result = dict(complete=True, original_q_window=WINDOW, radial_edges_A=list(EDGES), old_chart_split=3.,
        input_sha256=hashes, archived_sha256={p.name: sha(p) for p in (out/'provenance').iterdir()},
        selection_audit=selection, campaigns=campaigns, independent_shell_sum=combined, scope=scope)
    write(out/'analysis.json', result)
    def number(x): return 'unresolved' if x is None else f'{x:.6g}'
    lines = ['# Frozen AB far peak reference', '', '| Source | Mask | N | Nonzero | log Q | Row / population RSE |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in campaigns:
        for key, row in c['physical'].items():
            lines.append(f"| R={c['radius_A']:g} | {key} | {row['draws']} | {row['nonzero']} | {number(row['logQ'])} | {number(row['row_RSE'])} / {number(row['independent_population_RSE'])} |")
    for key in ('full', *OLD_KEYS):
        row = combined['physical'][key]
        lines.append(f"| Independent shell sum | {key} | {row['total_unconditional_draws']} | {row['nonzero']} | {number(row['logQ'])} | {number(row['row_RSE'])} / {number(row['independent_population_RSE'])} |")
    lines += ['', scope, '', 'Full hard-volume, cloud-noise, population, covariance, member-geometry, atomic-gap and provenance audits are in analysis.json.', '']
    (out/'report.md').write_text('\n'.join(lines)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = analyze(args.preparation, args.out)
    print(dict(output=str(args.out/'analysis.json'), complete=result['complete'], combined=result['independent_shell_sum']['physical']))
