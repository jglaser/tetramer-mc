#!/usr/bin/env python3
"""Independent per-population audit of hard-free-line physical integration.

No new poses or Poisson clouds are drawn. Full geometry reconstruction is the
default. ``saved-intervals`` explicitly leaves interval completeness unaudited;
it is useful for arithmetic checks, never a substitute for the full reference.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
import hard_free_line_reference as line
import analyze_native_region_reference as native
import analyze_basin_normalizers as statistics

require, read, sha, close = line.require, line.read, line.sha, line.close
SCHEMA = 'importance-latent-region-normalizer-v6'
GUIDE_SCHEMA = 'defensive-hard-free-line-guide-v1'
PROPOSAL_KIND = 'raw-translation-hard-free-line-conditioned-Gaussian-mixture'
MEASURE = 'Lebesgue measure in the original six-dimensional whitened region chart'


def shell_contains(radius, region):
    inner = region.get('minimum_mahalanobis_radius', 0.)
    return (radius > inner if inner else radius >= 0.) and radius <= region['mahalanobis_radius']


def q_contains(q, region):
    lo, hi = region['minimum_original_q'], region.get('maximum_original_q', math.inf)
    return ((q >= lo if region.get('minimum_original_q_inclusive', True) else q > lo) and
            (q <= hi if region.get('maximum_original_q_inclusive', True) else q < hi))


def cloud_log_weight(cloud, activity, intensity):
    """Check the positive Poisson PGF estimator, including z=0 and zero U."""
    require(math.isfinite(activity) and activity >= 0 and math.isfinite(intensity) and intensity > 0,
            'Invalid physical or auxiliary intensity')
    for key in ('lower_volume', 'upper_volume', 'uncertain_volume'):
        require(type(cloud[key]) in (int, float) and math.isfinite(cloud[key]) and cloud[key] >= 0,
                'Invalid envelope volume ' + key)
    for key in ('raw_points', 'overlap_points', 'retained_cells', 'created_cells', 'certified_cells'):
        require(type(cloud[key]) is int and cloud[key] >= 0, 'Invalid cloud count ' + key)
    close(cloud['upper_volume'], cloud['lower_volume'] + cloud['uncertain_volume'],
          'Envelope upper volume differs', atol=1e-10, rtol=1e-12)
    require(cloud['overlap_points'] <= cloud['raw_points'], 'Overlap count exceeds raw Poisson count')
    if activity == 0 or cloud['uncertain_volume'] == 0:
        require(cloud['raw_points'] == cloud['overlap_points'] == 0, 'Zero-intensity/volume cloud has points')
    if cloud['uncertain_volume'] > 0:
        require(cloud['retained_cells'] > 0, 'Positive uncertain volume without retained cells')
    expected = activity*cloud['lower_volume'] + cloud['overlap_points']*math.log1p(activity/intensity)
    close(cloud['log_weight'], expected, 'Poisson count weight differs', atol=2e-10, rtol=2e-12)
    return expected


def read_weights(rows, samples, region, arm):
    """Array contract shared with contact classification, with all attempted N.

    ``branch == 1`` denotes hard-free-line lineage, including an unconditioned
    or fallback Gaussian draw. It must never be reported as a Gaussian-only arm.
    This lightweight adapter assumes the separate population audit succeeded.
    """
    require(len(rows) == samples and all(type(r['draw']) is int for r in rows)
            and [r['draw'] for r in rows] == list(range(samples)),
            'Missing/repeated unconditional draws')
    z = np.full(samples, -np.inf); h = z.copy(); pairs = np.full((samples, 2), -np.inf)
    branch = np.zeros(samples, np.int8); component = np.full(samples, -1, np.int32)
    latents = np.empty((samples, 6)); support = np.zeros(samples, bool)
    for i, row in enumerate(rows):
        require(all(type(row[k]) is bool for k in ('hard_valid', 'capture_valid', 'region_valid', 'shell_valid')),
                'Missing support flags')
        latents[i] = row['latent']; require(np.isfinite(latents[i]).all(), 'Nonfinite latent row')
        close(row['latent_radius'], np.linalg.norm(latents[i]), 'Latent radius differs')
        support[i] = shell_contains(row['latent_radius'], region)
        require(row['shell_valid'] == support[i], 'Saved shell predicate differs')
        require(math.isfinite(row['q']) and row['q'] >= 0 and q_contains(row['q'], region) == row['region_valid'],
                'Saved original-q predicate differs')
        require(all(math.isfinite(row[k]) for k in ('log_physical_jacobian', 'log_proposal_density')),
                'Nonfinite Jacobian or generated-point density')
        if row['proposal_branch'] == 'uniform-shell':
            require(support[i] and row['proposal_component'] is None, 'Invalid uniform branch')
        else:
            k = row['proposal_component']
            require(row['proposal_branch'] == 'hard-free-line' and arm['alpha'] < 1 and type(k) is int
                    and 0 <= k < arm['component_count'], 'Invalid hard-free-line branch')
            branch[i] = 1; component[i] = k
        require(row.get('selected_ray_fallback') is None, 'Unexpected ray metadata')
        valid = row['hard_valid'] and row['capture_valid'] and row['region_valid'] and support[i]
        if not valid:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and row['clouds'] == [],
                    'Every invalid/exterior attempted draw must retain zero weight')
            continue
        require(len(row['clouds']) == 2, 'Expected two independent cloud records per valid pose')
        h[i] = row['log_hard_weight']; z[i] = row['log_importance_weight']
        require(math.isfinite(h[i]) and math.isfinite(z[i]), 'Missing finite physical weights')
        close(h[i], row['log_physical_jacobian'] - row['log_proposal_density'], 'Hard weight is not full J/q')
        pairs[i] = [h[i] + c['log_weight'] for c in row['clouds']]
        close(z[i], float(logsumexp(pairs[i]) - math.log(2)), 'Two-cloud arithmetic mean differs')
    return dict(z=z, h=h, pairs=pairs, branch=branch, component=component, latents=latents, support=support)


def compact_density(recon, u, details, geometry='full'):
    """Reconstruct every Gaussian conditional mass and complete mixture q."""
    require(geometry in ('full', 'saved-intervals'), 'Unknown geometry audit mode')
    x = recon.raw(u); g = recon.gaussian_logs(u)
    uniform = math.log(recon.alpha)-recon.logvolume if np.linalg.norm(u) <= recon.radius else -math.inf
    baseline = float(np.logaddexp(uniform, math.log1p(-recon.alpha)+logsumexp(g))) if recon.alpha < 1 else uniform
    if recon.beta == 0 or recon.alpha == 1:
        require(details.get('conditioning_disabled') is True, 'Missing disabled-conditioning trace')
        return dict(log_density=baseline, baseline_log_density=baseline, axes=[], interval_error=0.)
    close(details['raw_coordinates'], x, 'Compact trace raw coordinates differ')
    close(details['baseline_log_density'], baseline, 'Compact trace baseline differs')
    require([a['axis'] for a in details['axes']] == recon.axes, 'Missing/reordered density axes')
    factors = np.zeros(len(g)); fallback_count = 0; endpoints = 0.; axes = []
    for actual in details['axes']:
        axis = actual['axis']; line.normal.validate_intervals(actual['hard_free_intervals'], actual.get('segment'))
        if geometry == 'full':
            expected = recon.reconstruct_axis(u, axis)
            endpoints = max(endpoints, line.compare_axis(actual, expected))
        else:
            expected = dict(actual, intervals=actual['hard_free_intervals'])
        means, sigmas = recon.conditional(x, axis)
        masses = line.interval_masses(expected['intervals'], means, sigmas)
        fallback = masses <= recon.floor; allowed = line.contains(expected['intervals'], x[axis])
        multipliers = np.ones(len(g)); multipliers[~fallback] = 1/masses[~fallback] if allowed else 0.
        factors += multipliers; fallback_count += int(fallback.sum())
        correction = 1-recon.beta+recon.beta*multipliers; positive = correction > 0
        axis_q = float(np.logaddexp(uniform, math.log1p(-recon.alpha)+logsumexp(g[positive]+np.log(correction[positive]))))
        if math.isfinite(axis_q): close(actual['axis_log_proposal_density'], axis_q, 'Complete axis density differs', atol=2e-7, rtol=1e-11)
        else: require(actual['axis_log_proposal_density'] is None, 'Unsupported axis has nonzero density')
        axes.append(dict(**expected, conditional_means=means, conditional_sigmas=sigmas, conditional_masses=masses))
    require(details['component_branches'] == len(g)*len(recon.axes) and details['fallback_component_branches'] == fallback_count,
            'Compact conditional branch counts differ')
    correction = 1-recon.beta+recon.beta*factors/len(recon.axes); positive = correction > 0
    q = float(np.logaddexp(uniform, math.log1p(-recon.alpha)+logsumexp(g[positive]+np.log(correction[positive]))))
    return dict(log_density=q, baseline_log_density=baseline, axes=axes, interval_error=endpoints)


def audit_draw(row, recon, result):
    trace = row['hard_free_line_draw']; u = np.asarray(row['latent']); raw = recon.raw(u)
    require(type(trace['conditional']) is bool, 'Invalid conditional branch flag')
    original = recon.raw(trace['original_latent'])
    if not trace['conditional']:
        close(trace['original_latent'], u, 'Unconditioned draw changed original latent')
        return 0.
    require(row['proposal_branch'] == 'hard-free-line' and recon.beta > 0, 'Conditional draw on disabled/uniform branch')
    axis, k = trace['axis'], trace['component']
    require(axis in recon.axes and k == row['proposal_component'], 'Selected conditional component/axis differs')
    require('width_index' not in trace and 'width_A' not in trace, 'Hard-free guide acquired contact conditioning')
    others = [a for a in range(6) if a != axis]
    close(original[others], raw[others], 'Conditioner changed retained five raw coordinates')
    geom = next(a for a in result['axes'] if a['axis'] == axis)
    line.compare_axis(trace['geometry'], geom)
    mean, sigma, mass = (float(geom[key][k]) for key in ('conditional_means', 'conditional_sigmas', 'conditional_masses'))
    close(trace['conditional_mean'], mean, 'Selected mean differs'); close(trace['conditional_sigma'], sigma, 'Selected sigma differs')
    close(trace['conditional_mass'], mass, 'Selected mass differs', atol=2e-15, rtol=2e-7)
    require(type(trace['fallback']) is bool and trace['fallback'] == (mass <= recon.floor), 'Selected fallback differs')
    if trace['fallback']:
        close(trace['original_latent'], u, 'Fallback redrew outer coordinates'); return 0.
    require(line.contains(geom['intervals'], raw[axis]), 'Successful draw is outside feasible intervals')
    selected = trace['selected_interval']
    index = next((i for i, a in enumerate(geom['intervals']) if all(abs(selected[key]-a[key]) < 1e-9 for key in ('lower', 'upper'))), None)
    require(index is not None, 'Selected interval absent')
    interval_mass = float(line.interval_masses([selected], mean, sigma))
    close(trace['selected_interval_mass'], interval_mass, 'Selected interval mass differs', atol=2e-15, rtol=2e-7)
    lower_mass = float(line.interval_masses(geom['intervals'][:index], mean, sigma))
    selection = trace['uniform_interval_selection']*mass
    require(0 < trace['uniform_interval_selection'] < 1 and lower_mass-2e-14 <= selection <= lower_mass+interval_mass+2e-14,
            'Gaussian interval selection differs')
    close(trace['returned_raw_coordinate'], raw[axis], 'Returned coordinate differs')
    require(0 < trace['uniform_within_interval'] < 1, 'Invalid inverse-CDF uniform')
    cdf = float(line.normal.normal_masses((selected['lower']-mean)/sigma, (raw[axis]-mean)/sigma))/interval_mass
    error = abs(cdf-trace['uniform_within_interval']); require(error < 2e-6, 'Inverse conditional CDF differs')
    return error


def check_estimate(actual, expected):
    for key in ('draws', 'nonzero'): require(actual[key] == expected[key], 'Estimate denominator/count differs')
    for key in ('logQ', 'ess', 'relative_se', 'max_fraction'):
        if key not in actual: require(expected['nonzero'] == 0, 'Missing nonzero moment'); continue
        if expected[key] is None: require(actual[key] is None, 'Unresolved mass replaced by number')
        else: close(actual[key], expected[key], 'Saved estimate differs: '+key, atol=2e-9, rtol=2e-9)


def audit(directory, geometry='full'):
    root = Path(directory).resolve(); started = time.process_time()
    manifest, summary = read(root/'manifest.json'), read(root/'summary.json')
    require(manifest['schema'] == SCHEMA and manifest['guide_schema'] == GUIDE_SCHEMA and manifest['proposal_kind'] == PROPOSAL_KIND,
            'Wrong physical guide/schema')
    require(summary['complete'] is True and summary['manifest'] == manifest and not (root/'failure.json').exists(),
            'Incomplete or failed physical population')
    require(manifest['cloud_replicates'] == 2 and type(manifest['samples']) is int and manifest['samples'] > 0,
            'Expected positive fixed allocation and two independent clouds')
    require(type(manifest['seed']) is int and 0 <= manifest['seed'] < 2**64, 'Invalid population seed')
    bindings = {}
    def bind(path, digest=None):
        path = Path(path); value = sha(path)
        if digest is not None: require(value == digest, 'Changed provenance/output '+str(path))
        bindings[str(path.resolve())] = value
    for p in (root/'manifest.json', root/'summary.json', Path(__file__), Path(line.__file__), Path(line.normal.__file__), Path(native.__file__), Path(statistics.__file__)):
        bind(p)
    for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                      ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        bind(root/'provenance'/name, manifest[key])
    region, guide, config, shape = [read(root/'provenance'/n) for n in ('region.json', 'importance-guide.json', 'input-config.json', 'shape.json')]
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256']
            and region['gaussian_chart']['shape_sha256'] == manifest['shape_sha256'], 'Guide/chart/region/shape binding differs')
    fixed = region.get('physical_fixed_neighbors', [region['fixed_neighbor']])
    require(fixed == config['fixed_poses'] == manifest['physical_fixed_neighbors'] and region['fixed_neighbor'] == manifest['chart_anchor'], 'Changed physical scaffold/chart anchor')
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'), ('depletant_radius', 'depletant_radius'),
                   ('activity', 'reservoir_density'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Changed physical field '+rk)
    require(manifest['activity'] == region['activity'], 'Changed physical activity')
    require(type(manifest['lambda_ratio']) in (float, int) and math.isfinite(manifest['lambda_ratio'])
            and manifest['lambda_ratio'] > 0, 'Invalid auxiliary intensity ratio')
    close(manifest['lambda'], manifest['activity']*manifest['lambda_ratio'] if manifest['activity'] > 0 else 1., 'Auxiliary intensity differs')
    recon = line.Reconstructor(region, guide, config, shape)
    require(manifest['importance_uniform_probability'] == recon.alpha and manifest['importance_component_count'] == len(recon.weights)
            and manifest['proposal_density_measure'] == MEASURE, 'Changed guide mixture/measure')
    require(manifest['latent_radius'] == recon.radius and manifest['minimum_latent_radius'] == 0., 'Changed latent target')
    close(manifest['log_latent_ball_volume'], recon.logvolume, 'Ball volume differs'); close(manifest['log_latent_shell_volume'], recon.logvolume, 'Shell volume differs')
    for key, default in [('minimum_original_q', None), ('maximum_original_q', None), ('minimum_original_q_inclusive', True), ('maximum_original_q_inclusive', True)]:
        require(manifest.get(key, default) == region.get(key, default), 'Changed q-window '+key)
    for key in ('minimum_original_q_inclusive', 'maximum_original_q_inclusive'):
        require(type(manifest.get(key, True)) is bool and type(region.get(key, True)) is bool,
                'Original-q endpoint inclusion must be boolean')
    bind(root/'samples.jsonl', summary['samples_sha256']); bind(root/'attempts.jsonl', summary['attempts_sha256'])
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    attempts = [json.loads(s) for s in (root/'attempts.jsonl').read_text().splitlines()]
    require(attempts == [dict(draw=i, state='begin') for i in range(manifest['samples'])], 'Missing/duplicated/reordered attempted draw journal')
    require(summary['samples'] == manifest['samples'], 'Changed completed allocation')
    arrays = read_weights(rows, manifest['samples'], region, dict(alpha=recon.alpha, component_count=len(recon.weights)))
    maxima = Counter(); counts = Counter(); raw_points = 0
    for row in rows:
        u = np.asarray(row['latent']); raw, position, rotation, logj = recon.decode(u)
        close(row['pose']['position'], position, 'Position backmap differs'); close(line.normal.rotation(row['pose']['orientation']), rotation, 'Orientation backmap differs')
        relative = recon.Rf.T @ line.normal.rotation(row['pose']['orientation']) @ np.asarray(recon.anchor['rotation']).T
        quat = Rotation.from_matrix(relative).as_quat(); require(quat[3] != 0, 'Chart inverse singularity')
        recovered = np.r_[recon.Rf.T@(np.asarray(row['pose']['position'])-recon.fixed['position'])-recon.anchor['position'], recon.ell*quat[:3]/quat[3]]
        backmap = solve_triangular(recon.L0, recovered-recon.m0, lower=True)
        close(backmap, u, 'Independent latent inverse differs'); close(row['backmapped_latent'], backmap, 'Saved latent inverse differs')
        close(row['backmapped_radius'], np.linalg.norm(backmap), 'Saved inverse radius differs')
        close(row['log_physical_jacobian'], logj, 'Physical Jacobian differs'); close(row['physical_jacobian'], math.exp(logj), 'Exponentiated Jacobian differs')
        result = compact_density(recon, u, row['hard_free_line_density'], geometry)
        close(row['log_proposal_density'], result['log_density'], 'Complete proposal density differs', atol=2e-7, rtol=1e-11)
        maxima['log_proposal_density'] = max(maxima['log_proposal_density'], abs(row['log_proposal_density']-result['log_density']))
        maxima['log_physical_jacobian'] = max(maxima['log_physical_jacobian'], abs(row['log_physical_jacobian']-logj))
        maxima['latent_backmap'] = max(maxima['latent_backmap'], float(np.max(abs(backmap-u))))
        maxima['interval_endpoints'] = max(maxima['interval_endpoints'], result['interval_error'])
        maxima['inverse_cdf'] = max(maxima['inverse_cdf'], audit_draw(row, recon, result))
        capture = np.linalg.norm(position-np.asarray(config['capture_center'])) <= config['capture_radius']
        hard = bool(capture and recon.hard_valid(position, rotation))
        require(row['capture_valid'] == capture and row['hard_valid'] == hard, 'Independent physical indicator differs')
        physical_q = native.native_q(region['physical_metric'], row['pose']); close(row['q'], physical_q, 'Original physical metric differs')
        require(row['region_valid'] == q_contains(physical_q, region), 'Original-q inclusion differs')
        if capture and hard and row['shell_valid']:
            require(result['log_density'] >= result['baseline_log_density']-2e-9, 'Hard-free conditioning reduced valid-support density')
        counts.update(attempted=1, capture_rejected=int(not capture), hard_rejected=int(capture and not hard),
                      region_rejected=int(hard and not row['region_valid']), shell_rejected=int(not row['shell_valid']),
                      contributing=int(math.isfinite(arrays['z'][row['draw']])), conditioned_draws=int(row['hard_free_line_draw']['conditional']),
                      fallback_draws=int(row['hard_free_line_draw'].get('fallback', False)))
        if row['clouds']:
            for cloud in row['clouds']:
                cloud_log_weight(cloud, manifest['activity'], manifest['lambda']); raw_points += cloud['raw_points']
            for key in ('lower_volume', 'upper_volume', 'uncertain_volume', 'retained_cells', 'created_cells', 'certified_cells'):
                require(row['clouds'][0][key] == row['clouds'][1][key], 'Replicate clouds do not share fixed geometric envelope')
    for key in ('capture_rejected', 'hard_rejected', 'region_rejected', 'shell_rejected'):
        require(summary[key] == counts[key], 'Summary rejection count differs: '+key)
    require(summary['raw_points'] == raw_points, 'Raw cloud count summary differs')
    estimate, hard = statistics.moments(arrays['z']), statistics.moments(arrays['h'])
    check_estimate(summary['estimates']['region'], estimate); check_estimate(summary['estimates']['hard_region'], hard)
    require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds'] >= 0, 'Invalid sampler CPU')
    for p, digest in bindings.items(): require(sha(p) == digest, 'Input changed during audit')
    return dict(schema='independent-hard-free-line-physical-audit-v1', complete=True, root=str(root), seed=manifest['seed'],
        samples=manifest['samples'], finite_count=estimate['nonzero'], estimate=estimate, hard_region=hard, paired_noise=statistics.paired_noise(arrays['z'], arrays['pairs']),
        counts=dict(counts), maximum_errors=dict(maxima), input_sha256=bindings, files=bindings,
        samples_sha256=summary['samples_sha256'], attempts_sha256=summary['attempts_sha256'], geometry=geometry, geometry_mode=geometry,
        independently_reconstructed_geometry_rows=len(rows) if geometry == 'full' else 0,
        sampler_cpu_seconds=summary['sampler_cpu_seconds'], analysis_cpu_seconds=time.process_time()-started,
        executable_sha256=manifest['executable_sha256'], source_bundle_sha256=manifest['source_bundle_sha256'],
        new_pose_draws=0, new_Poisson_clouds=0,
        branch_labels={'0': 'uniform-shell', '1': 'hard-free-line'},
        scope='Fixed regional physical mass with every attempted draw in N. Complete q and J, physical indicators, conditional draw trace, two-cloud arithmetic mean and Poisson count formula checked. No native-classifier or global-coverage claim. Distinct replica RNG streams, exact thinning and envelope certification remain pinned implementation obligations; cloud counts alone cannot establish independence.',
        geometry_scope='Every whole-line interval independently reconstructed from atoms.' if geometry == 'full' else 'Stored intervals only: interval completeness is NOT independently audited.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', '--root', dest='root', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--geometry', choices=('full', 'saved-intervals'), default='full'); args = parser.parse_args()
    require(not args.out.exists(), 'New receipt path required')
    result = audit(args.root, geometry=args.geometry); args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.out)
