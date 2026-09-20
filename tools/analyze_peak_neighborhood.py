#!/usr/bin/env python3
"""Audit selected-pose local references without pooling nested support.

All masks use the original uniform-ball density and unconditional draw count.
The old weighted-chart split is a direct positive poststratification, never a
subtraction from a separately estimated full-window total.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from audit_shoulder_mis_independently import chart_values, near, rotation
from compare_intermediate_local_reference import load_reference, read_batches
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import read, require, sha, write
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments, quota_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_CHART = ROOT/'runs/ab-intermediate-guide-preparation-20260920/model-weighted.json'
RADIAL_EDGES = (0., .5, 1., 2.)
OLD_KEYS = ('old_r_le_32', 'old_r_gt_32')


def radial_keys(limit):
    require(limit in RADIAL_EDGES[1:], 'This diagnostic uses frozen radii .5, 1, or 2')
    return tuple(f'radial_{i}' for i in range(RADIAL_EDGES.index(limit)))


def selected_keys(radius, old_radius, limit):
    require(math.isfinite(radius) and 0 <= radius <= limit, 'Pose outside frozen ball')
    require(math.isfinite(old_radius) and old_radius >= 0, 'Invalid old chart radius')
    radial = next(f'radial_{i}' for i, upper in enumerate(RADIAL_EDGES[1:]) if radius <= upper)
    old = OLD_KEYS[int(old_radius > 32.)]
    return ('full', old, radial, f'{radial}__{old}')


def disjoint_covariance(a, b):
    """Unbiased sample covariance of two disjoint masked sample means.

    Since every row has a_i*b_i=0, this is -sum(a)*sum(b)/(N^2*(N-1)).
    A zero observation gives an observed zero covariance, not a tail bound.
    """
    require(a.count == b.count and a.count >= 2, 'Covariance needs matching original N')
    if not a.nonzero or not b.nonzero:
        return dict(sign=0, log_absolute_covariance=None)
    return dict(sign=-1, log_absolute_covariance=a.total+b.total-2*math.log(a.count)-math.log(a.count-1))


def partition_check(values, keys):
    """Reconstruct full mass and row variance, including cross-bin covariance."""
    full = values['full']; bins = [values[key] for key in keys]
    require(all(part.count == full.count for part in bins), 'Masked denominator changed')
    require(sum(part.nonzero for part in bins) == full.nonzero, 'Masks fail disjoint row coverage')
    covariances = [dict(left=left, right=right, **disjoint_covariance(values[left], values[right]))
                   for i, left in enumerate(keys) for right in keys[i+1:]]
    if not full.nonzero:
        return dict(complete_row_partition=True, logQ=None, log_variance_of_mean=None,
                    mass_error=0., relative_variance_error=None, covariances=covariances)
    log_total = float(logsumexp([part.total for part in bins]))
    log_square = float(logsumexp([part.square for part in bins]))
    error = max(near(log_total, full.total), near(log_square, full.square))
    # Scale by the largest sum of squared weights; cancellation is then bounded
    # FP64 arithmetic. The aggregate variance always comes from original rows.
    offset = full.square
    diagonals = sum(math.exp(part.centered_square()-offset) for part in bins)
    cross = sum(2*math.exp(a.total+b.total-math.log(full.count)-offset)
                for i, a in enumerate(bins) for b in bins[i+1:] if a.nonzero and b.nonzero)
    reconstructed = diagonals-cross
    direct = math.exp(full.centered_square()-offset)
    require(abs(reconstructed-direct) < 2e-10, 'Partition covariance reconstruction failed')
    result = quota_summary([full])
    return dict(complete_row_partition=True, logQ=result['logQ'],
                log_variance_of_mean=result['log_variance_of_mean'], mass_error=error,
                scaled_variance_error=abs(reconstructed-direct),
                relative_variance_error=abs(reconstructed-direct)/direct if direct else None,
                covariances=covariances,
                variance_rule='sum bin variances plus twice all same-row covariances; full original N')


class MaskedMoments:
    def __init__(self, limit):
        self.limit = limit; self.radial = radial_keys(limit)
        self.cross = tuple(f'{r}__{old}' for r in self.radial for old in OLD_KEYS)
        self.keys = ('full', *OLD_KEYS, *self.radial, *self.cross)
        self.values = {kind: {key: LogMoments() for key in self.keys} for kind in ('physical', 'hard')}

    def add(self, radius, old_radius, physical=None, hard=None, cloud_pair=None):
        selected = selected_keys(radius, old_radius, self.limit)
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        if physical is not None:
            require(math.isfinite(physical) and math.isfinite(hard), 'Invalid original nonzero weight')
            require(cloud_pair is not None and len(cloud_pair) == 2, 'Need original independent cloud pair')
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in self.keys:
                use = value is not None and key in selected
                self.values[kind][key].add(value if use else -math.inf,
                    cloud_pair if use and kind == 'physical' else None)

    def merge(self, other):
        require(self.limit == other.limit, 'Cannot pool different proposal radii')
        for kind in self.values:
            for key in self.keys: self.values[kind][key].merge(other.values[kind][key])

    def report(self):
        result = {}
        for kind, values in self.values.items():
            result[kind] = {}
            for key, moments in values.items():
                row = quota_summary([moments]); row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = 'IID original-density row variance / full unconditional N; every masked row stays zero'
                result[kind][key] = row
        result['partition_checks'] = {kind: {
            'old_chart': partition_check(values, OLD_KEYS),
            'radial': partition_check(values, self.radial),
            'cross_product': partition_check(values, self.cross)} for kind, values in self.values.items()}
        return result


def geometry_model_audit(model, region, cfg, selected_peak=None):
    """Rebuild the chart metric from rigid-member positions, not fitted weights."""
    require(model['weights'] == [1.] and model['means'] == [[0.]*6], 'Require one zero-mean geometric chart')
    require(model['coordinate_convention'] == 'anchor-body-relative', 'Unexpected chart convention')
    members = np.asarray([pose['position'] for pose in cfg['metadata']['rigid_members']])
    require(members.shape[1] == 3 and np.array_equal(members.mean(axis=0), np.zeros(3)),
            'RMS identity requires exactly centered rigid members')
    covariance = members.T@members/len(members)
    moment = np.trace(covariance)*np.eye(3)-covariance
    require(np.linalg.eigvalsh(moment).min() > 0, 'Degenerate member geometry')
    fixed = region['fixed_neighbor']; rf = rotation(fixed)
    anchor = model['anchors'][0]; relative_rotation = np.asarray(anchor['rotation'])
    near(relative_rotation.T@relative_rotation, np.eye(3)); near(np.linalg.det(relative_rotation), 1.)
    center = np.asarray(fixed['position'])+rf@np.asarray(anchor['position'])
    peak_rotation = rf@relative_rotation
    left_moment = relative_rotation@moment@relative_rotation.T
    sigma = np.zeros((6, 6)); sigma[:3, :3] = np.eye(3)
    sigma[3:, 3:] = model['angular_length']**2/4*np.linalg.inv(left_moment)
    errors = dict(covariance=near(sigma, model['covariances'][0]))
    if selected_peak is not None:
        errors['selected_peak_position'] = near(center, selected_peak['position'])
        errors['selected_peak_rotation'] = near(peak_rotation, rotation(selected_peak))
    report = dict(member_centroid_A=members.mean(axis=0).tolist(), member_covariance_A2=covariance.tolist(),
        rotational_moment_A2=moment.tolist(), rotational_moment_eigenvalues_A2=np.linalg.eigvalsh(moment).tolist(),
        left_chart_moment_A2=left_moment.tolist(), world_center_A=center.tolist(),
        world_rotation=peak_rotation.tolist(), reconstruction_errors=errors,
        selected_peak_checked=selected_peak is not None,
        meaning='r^2 = |dt|^2 + 4 c^T A c; exact member RMS^2 = |dt|^2 + 4 c^T A c/(1+|c|^2) <= r^2',
        scope='Numerical reconstruction from unchanged rigid members; no interval-arithmetic certificate')
    return report, (members, rf, center, peak_rotation, left_moment)


def geometry_rows(positions, rotations, geometry):
    members, rf, center, peak_rotation, moment = geometry
    delta = positions-center
    member_delta = delta[:, None, :]+np.einsum('nij,kj->nki', rotations-peak_rotation, members)
    direct_rms2 = np.mean(np.sum(member_delta*member_delta, axis=2), axis=1)
    relative = rf.T@rotations@peak_rotation.T@rf
    quaternion = Rotation.from_matrix(relative).as_quat()
    cayley = quaternion[:, :3]/quaternion[:, 3, None]
    norm2 = np.sum(cayley*cayley, axis=1)
    translation2 = np.sum(delta*delta, axis=1)
    angular2 = 4*np.einsum('ni,ij,nj->n', cayley, moment, cayley)
    rms2 = translation2+angular2/(1+norm2)
    radius2 = translation2+angular2
    _, logdet = np.linalg.slogdet(moment)
    logj = -math.log(8*math.pi**2)-.5*logdet-2*np.log1p(norm2)
    require(np.all(rms2 <= radius2+2e-10), 'RMS bound violated')
    return np.sqrt(radius2), logj, dict(member_rms2=near(direct_rms2, rms2)), direct_rms2


def audit_campaign(reference, model, old_model, selected_peak):
    audited, region, cfg = load_reference(reference, model)
    require(old_model['shape_sha256'] == audited['shape_sha256'] and old_model['weights'] == [1.],
            'Old partition chart shape/components changed')
    require(region['fixed_neighbor'] == cfg['fixed_poses'][0], 'Both chart models must use physical A frame')
    geometry_report, geometry = geometry_model_audit(model, region, cfg, selected_peak)
    radius = region['mahalanobis_radius']; aggregate = MaskedMoments(radius)
    master = read(Path(reference)/'manifest.json'); populations = []; errors = dict(member_rms2=0., radius=0., jacobian=0.)
    rms_range = [math.inf, -math.inf]; old_range = [math.inf, -math.inf]
    for job in master['jobs']:
        path = Path(job['directory']); local = MaskedMoments(radius); digest = hashlib.sha256(); n = 0
        for lines, rows in read_batches(path/'samples.jsonl'):
            for line in lines: digest.update(line)
            t = np.asarray([row['pose']['position'] for row in rows])
            q = np.asarray([row['pose']['orientation'] for row in rows])
            r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
            _, old_radii, _ = chart_values(old_model, t, r, cfg['fixed_poses'][0])
            radii, jac, current_errors, rms2 = geometry_rows(t, r, geometry)
            current_errors['radius'] = near(radii, [row['latent_radius'] for row in rows])
            current_errors['jacobian'] = near(jac, [row['log_physical_jacobian'] for row in rows])
            for key in errors: errors[key] = max(errors[key], current_errors[key])
            rms_range = [min(rms_range[0], float(np.sqrt(rms2.min()))), max(rms_range[1], float(np.sqrt(rms2.max())))]
            old_range = [min(old_range[0], float(old_radii.min())), max(old_range[1], float(old_radii.max()))]
            for i, row in enumerate(rows):
                require(row['draw'] == n, 'Changed row ordering'); n += 1
                weight = row['log_importance_weight']; hard = row['log_hard_weight']
                pair = [c['log_weight']+hard for c in row['clouds']] if weight is not None else None
                # Membership uses independent pose reconstruction. No tolerance
                # enlarges the masks, and no selected-row normalization is used.
                local.add(float(radii[i]), float(old_radii[i, 0]), weight, hard, pair)
        require(n == job['samples'], 'Unconditional sample count changed')
        require(digest.hexdigest() == audited['sample_sha256'][str(path/'samples.jsonl')], 'Rows changed between audits')
        populations.append(dict(id=job['id'], seed=job['seed'], samples=n, **local.report()))
        aggregate.merge(local)
    result = aggregate.report()
    for kind in ('physical', 'hard'):
        actual, original = result[kind]['full']['logQ'], audited[kind]['ball']['logQ']
        if actual is None: require(original is None, 'Full zero estimate changed')
        else: near(actual, original)
        for key in aggregate.keys:
            result[kind][key]['independent_population_RSE'] = population_rse([p[kind][key]['logQ'] for p in populations])
    return dict(root=str(Path(reference).resolve()), radius_A=radius, **result, populations=populations,
        geometry_reconstruction=geometry_report, row_geometry_max_errors=errors,
        observed_member_RMS_range_A=rms_range, observed_old_chart_radius_range=old_range,
        CPU_seconds=audited['CPU_seconds'], original_reference_audit=audited,
        scope='One fresh frozen local region only. Its old-chart remainder intersection overlaps the previous remainder estimate and must not be added to it.')


def analyze(references, chart_model, old_chart_model, out, peak_protocol=None):
    out = Path(out).resolve(); require(not out.exists(), 'Use a fresh derived output directory')
    chart_model = Path(chart_model).resolve(); old_chart_model = Path(old_chart_model).resolve()
    model, old_model = read(chart_model), read(old_chart_model)
    protocol_path = Path(peak_protocol).resolve() if peak_protocol else chart_model.parent/'protocol.json'
    require(protocol_path.is_file(), 'Need frozen selected-peak protocol')
    protocol = read(protocol_path); selected_path = protocol_path.parent/'selected-pose.json'
    require(sha(selected_path) == protocol['selected_pose_sha256'], 'Selected historical pose changed')
    require(sha(chart_model) == protocol['model_sha256'], 'Frozen geometry chart changed')
    for name, digest in protocol['archived_sha256'].items():
        require(sha(protocol_path.parent/'provenance'/name) == digest, f'Changed preparation archive: {name}')
    require(sha(protocol_path.parent/'config.json') == protocol['config_sha256'], 'Frozen physical config changed')
    require(sha(protocol_path.parent/'geometry.json') == protocol['geometry_sha256'], 'Frozen geometry report changed')
    preparation_cfg = read(protocol_path.parent/'config.json')
    require(preparation_cfg['capture_radius'] == 18. and preparation_cfg['depletant_radius'] == 1.5 and
            preparation_cfg['reservoir_density'] == .035, 'Physical baseline changed')
    selected = read(selected_path); selected_peak = selected['pose']
    historical = read(protocol_path.parent/'provenance/extremes.json')
    require(selected == next(p for p in historical['pieces'] if p['name'] == 'remainder')['top_poses'][0],
            'Center no longer matches the selected historical remainder maximum')
    # The old chart is the already-frozen partition definition, not refitted
    # from the new local samples. Check its parent freeze when available.
    old_freeze_path = old_chart_model.parent/'freeze.json'
    require(old_freeze_path.is_file(), 'Need old partition chart freeze')
    old_freeze = read(old_freeze_path)
    require(sha(old_chart_model) == old_freeze['model_sha256']['weighted'], 'Old weighted partition chart changed')
    require(sha(old_chart_model.parent/'protocol.json') == old_freeze['protocol_sha256'], 'Old chart protocol changed')
    for name, digest in old_freeze['archived_sha256'].items():
        require(sha(old_chart_model.parent/'provenance'/name) == digest, f'Changed old chart archive: {name}')
    require(sha(old_chart_model.parent/'config.json') == old_freeze['config_sha256'], 'Old chart config changed')
    old_cfg = read(old_chart_model.parent/'config.json')
    for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata'):
        require(old_cfg[key] == preparation_cfg[key], f'Old partition physical target differs: {key}')
    sources = {'chart-model.json': chart_model, 'old-chart-model.json': old_chart_model,
               'old-chart-freeze.json': old_freeze_path, 'peak-protocol.json': protocol_path,
               'selected-pose.json': selected_path}
    dependencies = local_dependencies([Path(__file__)])
    input_hashes = {str(path): sha(path) for path in sources.values()}
    campaigns = []; seeds = set()
    for path in references:
        declared = next((c for c in protocol['campaigns'] if Path(c['output']).resolve() == Path(path).resolve()), None)
        require(declared is not None, 'Reference is not in the frozen peak protocol')
        master = read(Path(path)/'manifest.json')
        require(master['region_sha256'] == declared['region_sha256'], 'Declared local region changed')
        require([job['seed'] for job in master['jobs']] == declared['seeds'] and
                all(job['samples'] == declared['samples'] for job in master['jobs']), 'Frozen population allocation changed')
        result = audit_campaign(Path(path).resolve(), model, old_model, selected_peak)
        for population in result['populations']:
            require(population['seed'] not in seeds, 'Reference random streams must be distinct'); seeds.add(population['seed'])
        if campaigns:
            require(result['original_reference_audit']['physical_signature'] == campaigns[0]['original_reference_audit']['physical_signature'],
                    'Reference physical targets differ')
        campaigns.append(result)
    require(campaigns, 'Need at least one reference')
    for path, digest in input_hashes.items(): require(sha(path) == digest, 'Analysis input changed')
    (out/'provenance').mkdir(parents=True)
    for name, path in {**dependencies, **sources}.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    scope = ('Each campaign remains an independent estimate with its original unconditional N and density. Nested balls are not summed or pooled. '
        'All masks retain original 2<=q<5, capture and AB hard constraints. Old weighted r<=32 and r>32 split each local ball directly using positive weights. '
        'A local remainder estimate may replace a portion of the previous remainder only with a complementary reference masked by exactly the same new ball; it cannot be added to the previous total. '
        'Peak selection used historical data; these fresh draws estimate fixed selected neighborhoods conditionally. Empty supported pieces are unresolved, not zero masses or bounds. '
        'Observed errors and ESS do not certify unseen weight, equilibrium mixing, association of the fixed neighbors, or assembly. Hard flags are from the frozen physical kernel.')
    result = dict(complete=True, radial_edges_A=list(RADIAL_EDGES), radial_endpoint_rule='First bin closed at zero; each upper edge closed and other lower edges open',
        old_chart_split=32., input_sha256=input_hashes, archived_sha256={p.name: sha(p) for p in (out/'provenance').iterdir()},
        campaigns=campaigns, scope=scope)
    write(out/'analysis.json', result)
    def number(value): return 'unresolved' if value is None else f'{value:.6g}'
    lines = ['# Selected-pose local neighborhoods', '', '| Radius (Å) | Mask | N | Nonzero | log Q | Row / population RSE | Largest weight |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for campaign in campaigns:
        for key, row in campaign['physical'].items():
            lines.append(f"| {campaign['radius_A']:g} | {key} | {row['draws']} | {row['nonzero']} | {number(row['logQ'])} | {number(row['row_RSE'])} / {number(row['independent_population_RSE'])} | {number(row['maximum_fraction'])} |")
    lines += ['', scope, '', 'Hard volumes, same-row covariances, geometry identities, original audits and source hashes are in analysis.json.', '']
    (out/'report.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, action='append', required=True)
    parser.add_argument('--chart-model', type=Path, required=True)
    parser.add_argument('--old-chart-model', type=Path, default=DEFAULT_OLD_CHART)
    parser.add_argument('--peak-protocol', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reference, args.chart_model, args.old_chart_model, args.out, args.peak_protocol)
    print(args.out/'analysis.json')
    for campaign in result['campaigns']:
        print(campaign['radius_A'], campaign['physical']['full']['logQ'], campaign['physical']['full']['row_RSE'])


if __name__ == '__main__': main()
