#!/usr/bin/env python3
"""Read-only raw-row audit of a frozen, fresh fixed-quota shoulder campaign.

This deliberately does not import shoulder_mis or its analyzer. Rotations,
covariance quadratics, physical Jacobians and zero-inclusive NumPy variances
are reconstructed independently. Hard-overlap flags are the archived kernel's;
this audit does not rerun atomic geometry, insert depletants, or refit guides.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

OFFSET = 25.
EDGES = (1., 1.1, 1.25, 1.5, 2.)
FAMILIES = ('guide', 'cover')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose['orientation'])[[1, 2, 3, 0]]).as_matrix()


def near(a, b, tolerance=2e-8):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert error < tolerance, (error, tolerance)
    return error


def chart_values(model, t, r, fixed):
    """Return each component's log density, radius and direct physical log J."""
    fr = rotation(fixed)
    tr = (t-np.asarray(fixed['position'])) @ fr
    rr = fr.T @ r
    ell = model['angular_length']
    logs, radii, jacobians = [], [], []
    for anchor, mean, covariance, weight in zip(model['anchors'], model['means'], model['covariances'], model['weights']):
        quaternion = Rotation.from_matrix(rr @ np.asarray(anchor['rotation']).T).as_quat()
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            cayley = quaternion[:, :3]/quaternion[:, 3, None]
            x = np.column_stack((tr-anchor['position'], ell*cayley))-mean
            sign, logdet = np.linalg.slogdet(covariance)
            assert sign == 1
            quadratic = np.einsum('ni,ij,nj->n', x, np.linalg.inv(covariance), x)
            logj = .5*logdet-3*math.log(ell)-2*math.log(math.pi)-2*np.log1p(np.sum(cayley*cayley, axis=1))
            logpdf = math.log(weight)-3*math.log(2*math.pi)-.5*quadratic-logj
        finite = np.isfinite(x).all(axis=1)
        logs.append(np.where(finite, logpdf, -np.inf))
        radii.append(np.where(finite, np.sqrt(quadratic), np.inf))
        jacobians.append(np.where(finite, logj, np.inf))
    return np.array(logs).T, np.array(radii).T, np.array(jacobians).T


def independent_densities(raw, cfg, model, region, guide, product):
    t = np.asarray([row['pose']['position'] for row in raw])
    q = np.asarray([row['pose']['orientation'] for row in raw])
    assert np.isfinite(t).all() and np.isfinite(q).all()
    assert np.max(abs(np.sum(q*q, axis=1)-1)) < 1e-8
    q /= np.linalg.norm(q, axis=1)[:, None]
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    logs, _, _ = chart_values(model, t, r, cfg['fixed_poses'][guide['anchor_index']])
    gaussian = logsumexp(logs, axis=1)
    product_logs = []
    for weight, node in zip(product['weights'], product['covers']):
        ref = node['reference']; rr = rotation(ref)
        centroid = np.asarray(node['centroid'])
        compensated = t-ref['position']+r@centroid-rr@centroid
        distance = np.linalg.norm(compensated, axis=1)
        angles = Rotation.from_matrix(rr.T @ r).magnitude()
        volume = 4*node['ball_radius']**3/3*(node['angle_cap']-math.sin(node['angle_cap']))
        near(volume, node['volume'], 2e-11)
        product_logs.append(np.where((distance <= node['ball_radius']) & (angles <= node['angle_cap']), math.log(weight/volume), -np.inf))
    logproduct = logsumexp(product_logs, axis=0)
    lengths = np.asarray(guide['cube_lengths'])
    delta = t-guide['capture_center']
    cube = np.where(np.all((delta >= -.5*lengths) & (delta < .5*lengths), axis=1), -np.log(lengths).sum(), -np.inf)
    beta, epsilon = guide['weight'], guide['uniform_probability']
    g = np.logaddexp(math.log1p(-beta)+logproduct,
                    math.log(beta)+np.logaddexp(math.log1p(-epsilon)+gaussian, math.log(epsilon)+cube))
    _, radii, jac = chart_values(region['gaussian_chart'], t, r, region['fixed_neighbor'])
    assert radii.shape[1] == 1
    lo, hi = region.get('minimum_mahalanobis_radius', 0.), region['mahalanobis_radius']
    logvolume = math.log(math.pi**3/6*(hi**6-lo**6))
    c = np.where((radii[:, 0] >= lo) & (radii[:, 0] <= hi), -logvolume-jac[:, 0], -np.inf)
    metric = cfg['metadata']; scores = []
    for reference in metric['native_poses']:
        rr = rotation(reference)
        displacements = [np.linalg.norm(t-reference['position']+r@member['position']-rr@member['position'], axis=1)
                         for member in metric['rigid_members']]
        rq = np.asarray(reference['orientation']); rq = rq/np.linalg.norm(rq)
        angles = 2*np.arccos(np.minimum(1., np.abs(q@rq)))
        scores.append(np.maximum(np.max(displacements, axis=0)/metric['member_error_scale'],
                                 angles/math.radians(metric['angle_error_scale_deg'])))
    original_q = np.min(scores, axis=0)
    capture = np.linalg.norm(t-cfg['capture_center'], axis=1) <= cfg['capture_radius']
    return g, c, radii[:, 0], jac[:, 0], logvolume, original_q, capture


def estimate(strata, offset=OFFSET):
    n = sum(len(a) for a in strata)
    total = sum(a.sum() for a in strata)
    mean = total/n
    variance = sum(len(a)*np.var(a, ddof=1) for a in strata)/n**2
    return {'draws': n, 'nonzero': int(sum(np.count_nonzero(a) for a in strata)),
            'logQ': float(math.log(mean)+offset), 'stratified_RSE': float(math.sqrt(variance)/mean),
            'log_variance_of_mean': float(math.log(variance)+2*offset),
            'maximum_fraction': float(max(a.max() for a in strata)/total),
            'weight_ESS': float(total**2/sum(a@a for a in strata))}


def paired(mis, own, baseline):
    n = sum(len(a) for a in mis); alpha = len(mis[baseline])/n
    h = [a.copy() for a in mis]
    h[baseline] -= own[baseline]/alpha
    difference = sum(a.sum() for a in h)/n
    variance = sum(len(a)*np.var(a, ddof=1) for a in h)/n**2
    vm = sum(len(a)*np.var(a, ddof=1) for a in mis)/n**2
    vu = np.var(own[baseline], ddof=1)/len(own[baseline])
    covariance = np.cov(mis[baseline], own[baseline], ddof=1)[0, 1]/n
    near(variance, vm+vu-2*covariance, 2e-13)
    return {'relative_difference_to_component': float(difference/own[baseline].mean()),
            'difference_in_observed_standard_errors': float(difference/math.sqrt(variance)),
            'log_variance_of_difference': float(math.log(variance)+2*OFFSET),
            'log_absolute_difference': float(math.log(abs(difference))+OFFSET),
            'sign': int(np.sign(difference)),
            'MIS_component_correlation': float(covariance/math.sqrt(vm*vu))}


def run(root, out):
    started = time.process_time(); assert not out.exists(), 'Use a fresh derived directory'
    master = read(root/'manifest.json'); archive = root/'provenance'
    assert sha(root/'manifest.json') == (root/'manifest.sha256').read_text().split()[0]
    for filename, digest in master['archive_sha256'].items():
        assert sha(archive/filename) == digest, filename
    cfg, model, region = [read(archive/name) for name in ('config.json', 'guide-model.json', 'region.json')]
    for key, value in master['physical'].items(): assert cfg[key] == value, key
    for filename, key in [('config.json', 'config_sha256'), ('shape.json', 'shape_sha256'), ('guide-model.json', 'model_sha256'), ('region.json', 'region_sha256')]:
        assert sha(archive/filename) == master[key]
    assert region['physical_fixed_neighbors'] == cfg['fixed_poses'] and region['physical_metric'] == cfg['metadata']
    assert region['depletant_radius'] == cfg['depletant_radius']
    assert master['q_window'] == dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False)
    assert master['lambda_ratio'] == 64 and master['cloud_replicates'] == 2
    assert master['population_count'] == len(master['populations']) == 4
    quotas = [master['allocation'][name] for name in FAMILIES]; alpha = quotas[0]/sum(quotas)
    accumulated = {name: [] for name in FAMILIES}; provenance = []; seeds = set(); top = []
    errors = dict(source_log_density=0., original_q=0., latent_radius=0., direct_log_jacobian=0., numerator=0.)
    density_cpu = 0.
    for pindex, population in enumerate(master['populations']):
        gm = read(Path(population['guide']['output'])/'manifest.json')
        assert gm['guide'] == master['guide']['description']
        for family, quota in zip(FAMILIES, quotas):
            job = population[family]; path = Path(job['output'])
            runmanifest, summary = read(path/'manifest.json'), read(path/'summary.json')
            assert summary['complete'] and quota == job['samples'] == runmanifest['samples'] == summary['samples']
            assert job['seed'] == runmanifest['seed'] and job['seed'] not in seeds
            seeds.add(job['seed'])
            for key in ('config_sha256', 'shape_sha256'): assert runmanifest[key] == master[key]
            binary = 'native-region-normalizer' if family == 'guide' else 'latent-region-normalizer'
            assert runmanifest['executable_sha256'] == master['archive_sha256'][binary]
            assert runmanifest['activity'] == cfg['reservoir_density'] and runmanifest['cloud_replicates'] == 2
            assert runmanifest['lambda_ratio'] == 64 and runmanifest['lambda'] == 64*cfg['reservoir_density']
            if family == 'cover':
                assert runmanifest == summary['manifest']
                assert runmanifest['physical_fixed_neighbors'] == cfg['fixed_poses']
                assert runmanifest['region_sha256'] == master['region_sha256']
                assert runmanifest['minimum_original_q'] == 1 and runmanifest['maximum_original_q'] == 2
                assert not runmanifest['minimum_original_q_inclusive'] and not runmanifest['maximum_original_q_inclusive']
            else:
                assert runmanifest['q_window'] == master['q_window']
                assert runmanifest['depletant_radius'] == cfg['depletant_radius']
            for filename, field in [('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256'),
                                    ('config.json' if family == 'guide' else 'input-config.json', 'config_sha256'),
                                    ('guide-model.json' if family == 'guide' else 'region.json', 'model_sha256' if family == 'guide' else 'region_sha256')]:
                expected = master[field] if field == 'model_sha256' else runmanifest[field]
                assert sha(path/'provenance'/filename) == expected
            # Dense arrays retain every invalid draw as an exact zero in its original quota.
            data = {name: np.zeros(quota) for name in ('mis', 'own', 'hard', 'own_hard', 'cloud_variance', 'q')}
            digest = hashlib.sha256(); count = 0
            with (path/'samples.jsonl').open('rb') as handle:
                while True:
                    raw = []
                    for _ in range(4096):
                        line = handle.readline()
                        if not line: break
                        digest.update(line); raw.append(json.loads(line))
                    if not raw: break
                    tick = time.process_time()
                    lg, lc, radii, jac, logvolume, q, capture = independent_densities(raw, cfg, model, region, gm['guide'], gm['cover_mixture'])
                    lm = np.logaddexp(math.log(alpha)+lg, math.log1p(-alpha)+lc)
                    density_cpu += time.process_time()-tick
                    for i, row in enumerate(raw):
                        assert row['draw'] == count
                        errors['original_q'] = max(errors['original_q'], near(q[i], row['q']))
                        data['q'][count] = q[i]
                        window = 1 < q[i] < 2
                        source = lg[i] if family == 'guide' else lc[i]
                        if family == 'guide':
                            recorded = row['log_proposal_density']; reason = row.get('zero')
                            assert reason in (None, 'q', 'capture', 'hard')
                            assert (reason == 'q') == (not window)
                            if window: assert (reason == 'capture') == (not capture[i])
                            valid = reason is None
                        else:
                            recorded = -logvolume-row['log_physical_jacobian']
                            errors['latent_radius'] = max(errors['latent_radius'], near(radii[i], row['latent_radius']))
                            errors['direct_log_jacobian'] = max(errors['direct_log_jacobian'], near(jac[i], row['log_physical_jacobian']))
                            assert row['region_valid'] == window and row['capture_valid'] == capture[i]
                            valid = row['region_valid'] and row['capture_valid'] and row['hard_valid']
                        errors['source_log_density'] = max(errors['source_log_density'], near(source, recorded))
                        assert math.isfinite(lm[i])
                        if valid:
                            assert window and capture[i] and math.isfinite(lc[i])
                            if family == 'guide':
                                clouds = row['cloud_log_weights']
                                counts = row['cloud_overlap_counts']; rawcounts = row['cloud_raw_points']
                                lowers = [row['lower_volume']]*2
                            else:
                                clouds = [c['log_weight'] for c in row['clouds']]
                                counts = [c['overlap_points'] for c in row['clouds']]
                                rawcounts = [c['raw_points'] for c in row['clouds']]
                                lowers = [c['lower_volume'] for c in row['clouds']]
                            assert len(clouds) == len(counts) == len(rawcounts) == 2
                            assert all(type(k) is int and type(n) is int and 0 <= k <= n for k, n in zip(counts, rawcounts))
                            near(clouds, cfg['reservoir_density']*np.array(lowers)+np.array(counts)*math.log1p(1/64))
                            logf = float(np.logaddexp(*clouds)-math.log(2))
                            errors['numerator'] = max(errors['numerator'], near(logf-source, row['log_importance_weight']))
                            near(-source, row['log_hard_weight'])
                            data['mis'][count] = math.exp(logf-lm[i]-OFFSET)
                            data['own'][count] = math.exp(logf-source-OFFSET)
                            data['hard'][count] = math.exp(-lm[i])
                            data['own_hard'][count] = math.exp(-source)
                            zcloud = np.exp(np.array(clouds)-lm[i]-OFFSET)
                            data['cloud_variance'][count] = float((zcloud[0]-zcloud[1])**2/4)
                            top.append(dict(population=population['id'], source=family, draw=count, q=float(q[i]),
                                            log_mis_weight=float(logf-lm[i]), log_own_weight=float(logf-source),
                                            log_guide_to_cover_density=float(lg[i]-lc[i]), cloud_log_weights=clouds))
                        else:
                            assert row.get('log_importance_weight') is None and row.get('log_hard_weight') is None
                        count += 1
            assert count == quota and digest.hexdigest() == summary['samples_sha256']
            accumulated[family].append(data)
            provenance.append(dict(population=population['id'], family=family, seed=job['seed'], samples=count,
                                   samples_path=str(path/'samples.jsonl'), samples_sha256=digest.hexdigest(),
                                   manifest_sha256=sha(path/'manifest.json'), summary_sha256=sha(path/'summary.json'),
                                   CPU_seconds=summary['cpu_seconds' if family == 'guide' else 'sampler_cpu_seconds']))
    assert sum(p['samples'] for p in provenance) == master['total_unconditional_draws']
    pooled = [{name: np.concatenate([p[name] for p in accumulated[f]]) for name in accumulated[f][0]} for f in FAMILIES]
    mis, own = [[p[name] for p in pooled] for name in ('mis', 'own')]
    result = dict(MIS=estimate(mis), guide=estimate([own[0]]), cover=estimate([own[1]]),
                  hard_MIS=estimate([p['hard'] for p in pooled], 0.),
                  paired_MIS_minus_guide=paired(mis, own, 0), paired_MIS_minus_cover=paired(mis, own, 1))
    n = sum(map(len, mis)); v = sum(len(a)*np.var(a, ddof=1) for a in mis)/n**2
    result['MIS']['paired_cloud_variance_fraction'] = float(sum(p['cloud_variance'].sum() for p in pooled)/n**2/v)
    result['q_bands'] = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        arrays = [np.where((p['q'] >= lo) & (p['q'] < hi), p['mis'], 0.) for p in pooled]
        result['q_bands'].append(dict(lower=lo, upper=hi, **estimate(arrays)))
    result['populations'] = []
    for pindex, population in enumerate(master['populations']):
        parts = [accumulated[f][pindex] for f in FAMILIES]
        result['populations'].append(dict(id=population['id'], MIS=estimate([p['mis'] for p in parts]),
                                         guide=estimate([parts[0]['own']]), cover=estimate([parts[1]['own']]),
                                         paired_MIS_minus_guide=paired([p['mis'] for p in parts], [p['own'] for p in parts], 0)))
    popvalues = np.array([math.exp(p['MIS']['logQ']-OFFSET) for p in result['populations']])
    result['MIS']['independent_population_RSE'] = float(popvalues.std(ddof=1)/math.sqrt(len(popvalues))/popvalues.mean())
    costs = {f: sum(p['CPU_seconds'] for p in provenance if p['family'] == f) for f in FAMILIES}
    costs['MIS'] = sum(costs.values())
    for name in ('MIS', 'guide', 'cover'):
        result[name]['generation_CPU_seconds'] = costs[name]
        result[name]['relative_variance_times_generation_CPU'] = costs[name]*result[name]['stratified_RSE']**2
    result['cost_comparison'] = dict(extra_generation_cost_fraction=costs['cover']/costs['guide'],
        MIS_efficiency_vs_guide_own_mean_scales=result['guide']['relative_variance_times_generation_CPU']/result['MIS']['relative_variance_times_generation_CPU'],
        MIS_efficiency_vs_guide_common_mean_scale=math.exp(result['guide']['log_variance_of_mean']-result['MIS']['log_variance_of_mean'])*costs['guide']/costs['MIS'],
        fixed_guide_count_variance_reduction=1-math.exp(result['MIS']['log_variance_of_mean']-result['guide']['log_variance_of_mean']))
    # Only now read the primary output; it did not supply means, weights or densities.
    primary_path = root/'assessment/analysis.json'; primary = read(primary_path)
    comparisons = []
    for label, ours, theirs in [('MIS', result['MIS'], primary['estimates']['all']),
                                ('guide', result['guide'], primary['component_only']['guide']['all']),
                                ('cover', result['cover'], primary['component_only']['cover']['all']),
                                ('hard_MIS', result['hard_MIS'], primary['hard_estimates']['all'])]:
        comparisons.append(dict(estimate=label, maximum_numeric_difference=max(near(ours[k], theirs[k]) for k in ('logQ', 'stratified_RSE', 'log_variance_of_mean', 'maximum_fraction', 'weight_ESS'))))
    for f in FAMILIES:
        theirs = primary['paired_differences'][f]['all']; ours = result['paired_MIS_minus_'+f]
        for key in ('relative_difference_to_component', 'difference_in_observed_standard_errors', 'log_variance_of_difference', 'log_absolute_difference'): near(ours[key], theirs[key])
    for index, band in enumerate(result['q_bands']): near(band['logQ'], primary['estimates'][str(index)]['logQ'])
    result['top_16_MIS'] = sorted(top, key=lambda p: p['log_mis_weight'], reverse=True)[:16]
    result['top_16_cover_control'] = sorted([p for p in top if p['source'] == 'cover'], key=lambda p: p['log_own_weight'], reverse=True)[:16]
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), campaign=str(root),
                  manifest_sha256=sha(root/'manifest.json'), physical=master['physical'], q_window=master['q_window'],
                  model_sha256=master['model_sha256'], region_sha256=master['region_sha256'], shape_sha256=master['shape_sha256'],
                  config_sha256=master['config_sha256'], allocation=master['allocation'], raw_sources=provenance,
                  all_unconditional_rows_audited=n, density_max_errors=errors, independent_density_CPU_seconds=density_cpu,
                  audit_CPU_seconds=time.process_time()-started, audit_sha256=sha(__file__),
                  primary_assessment_sha256=sha(primary_path), final_primary_comparisons=comparisons,
                  primary_analysis_CPU_seconds=primary['analysis_CPU_seconds'],
                  primary_vector_density_CPU_seconds_subset_of_analysis=primary['vector_density_CPU_seconds'],
                  arithmetic='Direct covariance inverse quadratic and direct Haar/Cayley Jacobian; dense zero-padded NumPy arrays, common exp(25) scale; pooled within-source sample variances. No shoulder_mis imports.',
                  scope='Read-only audit of existing fresh draws. Kernel hard-overlap flags retained; no atomic clash recalculation, physical insertions, fitting, retrospective allocation change, or pooling with historical campaigns. Reported errors and costs describe observed samples and do not bound unvisited tails.')
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    shutil.copy2(__file__, out/Path(__file__).name)
    print(json.dumps({key: result[key] for key in ('MIS', 'guide', 'cover', 'hard_MIS', 'paired_MIS_minus_guide', 'cost_comparison', 'density_max_errors', 'audit_CPU_seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())
