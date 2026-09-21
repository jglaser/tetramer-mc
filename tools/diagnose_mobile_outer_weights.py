#!/usr/bin/env python3
"""Read saved outer-shell rows, characterize concentration, and freeze guide-only centers."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(row, center, members, angular_length, chart_sd):
    p, q = row['pose'], center
    r = Rotation.from_quat(np.array(p['orientation'])[[1, 2, 3, 0]])
    s = Rotation.from_quat(np.array(q['orientation'])[[1, 2, 3, 0]])
    dt = np.asarray(p['position'])-q['position']
    displacement = r.apply(members)+p['position']-s.apply(members)-q['position']
    u = np.asarray(row['latent'])
    cloud_logs = np.array([c['log_weight'] for c in row['clouds']])
    return dict(latent=row['latent'], radius=row['latent_radius'], q=row['q'],
                translation_from_chart_center_A=float(np.linalg.norm(dt)),
                rotation_from_chart_center_deg=math.degrees((r*s.inv()).magnitude()),
                maximum_member_displacement_A=float(np.linalg.norm(displacement, axis=1).max()),
                rotational_fraction_of_squared_latent_radius=float(np.sum(u[3:]**2)/np.sum(u**2)),
                cloud_log_factors=cloud_logs.tolist(),
                absolute_cloud_log_difference=float(abs(cloud_logs[0]-cloud_logs[1])),
                larger_cloud_fraction=float(np.exp(cloud_logs.max()-logsumexp(cloud_logs))),
                overlap_volume_point_estimates_A3=[c['lower_volume']+c['overlap_points']/(64*.035) for c in row['clouds']],
                overlap_estimate_scope='Noisy Poisson point-count estimates, not exact overlap volumes or energies.')


def choose_centers(populations, count=4, separation=1.):
    """Equal influence per population, ranking by the smaller of its two noisy bath factors."""
    selected = []
    for population in populations:
        valid = [r for r in population if r['log_importance_weight'] is not None]
        ordered = sorted(valid, key=lambda r: min(c['log_weight'] for c in r['clouds']), reverse=True)
        group = []
        for row in ordered:
            if all(np.linalg.norm(np.array(row['latent'])-other['latent']) >= separation for other in group):
                group.append(row)
            if len(group) == count:
                break
        if len(group) != count:
            raise ValueError('Insufficient distinct guide centers; do not silently alter guide protocol')
        selected.extend(group)
    return selected


def diagnose(campaign, out):
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Use fresh immutable output')
    protocol, status = read(campaign/'protocol.json'), read(campaign/'status.json')
    assert status['complete'] and not status['running']
    assert status['protocol_sha256'] == sha(campaign/'protocol.json')
    results, input_hashes = [], {str(campaign/'protocol.json'): sha(campaign/'protocol.json'),
                               str(campaign/'status.json'): sha(campaign/'status.json')}
    guides = {}
    for spec in protocol['commands']:
        root = campaign/spec['region']
        terminal = next(x for x in status['jobs'] if x['region'] == spec['region'])
        assessment_path = root/'assessment/analysis.json'
        assert sha(assessment_path) == terminal['assessment_sha256']
        assessment, manifest = read(assessment_path), read(root/'manifest.json')
        region, config = read(root/'provenance/region.json'), read(root/'provenance/config.json')
        input_hashes[str(assessment_path)] = sha(assessment_path)
        input_hashes[str(root/'provenance/region.json')] = sha(root/'provenance/region.json')
        members = np.array([m['position'] for m in config['metadata']['rigid_members']])
        populations, combined = [], []
        for job in manifest['jobs']:
            sample_path = Path(job['directory'])/'samples.jsonl'
            audited = next(x for x in assessment['populations'] if x['id'] == job['id'])
            assert sha(sample_path) == audited['samples_sha256']
            input_hashes[str(sample_path)] = sha(sample_path)
            rows = [dict(json.loads(line), population=job['id']) for line in sample_path.read_text().splitlines()]
            assert len(rows) == job['samples']
            populations.append(rows); combined.extend(rows)
        valid = [r for r in combined if r['log_importance_weight'] is not None]
        ordered = sorted(valid, key=lambda r: r['log_importance_weight'], reverse=True)
        logs = np.array([r['log_importance_weight'] for r in ordered])
        weights = np.exp(logs-logsumexp(logs))
        points = np.array([r['latent'] for r in ordered])
        average = weights@points
        covariance = ((points-average)*weights[:, None]).T@(points-average)
        cloud_noise = assessment['estimate']['paired_noise']
        alpha = cloud_noise['paired_cloud_variance_fraction']
        row_rse = assessment['estimate']['relative_se']
        cpu = assessment['sampler_cpu_seconds']
        guide = choose_centers(populations)
        components = [dict(weight=1/(2*len(guide)), mean=r['latent'],
                           covariance=(np.eye(6)*sd**2).tolist(),
                           source_population=r['population'], source_draw=r['draw'], latent_sd=sd)
                      for r in guide for sd in (.5, 1.5)]
        guides[spec['region']] = dict(schema='defensive-latent-shell-guide-design-v1',
            implemented=False, region_sha256=sha(root/'provenance/region.json'),
            defensive_uniform_shell_probability=.5, gaussian_components=components,
            guide_rule='Four centers per independent completed population, ranked by min(logW1,logW2), '
                       'greedily separated by at least one latent unit within each population; '
                       'equal masses and fixed isotropic latent SD0.5/1.5. Selection weights are never physical estimators.',
            support='Uniform shell with probability0.5, otherwise an untruncated Gaussian mixture in R6. '
                    'Gaussian draws outside the fixed shell contribute zero; no rejection/resampling conditioning.',
            density='q(u)=0.5*I_shell/V_shell+0.5*sum_k a_k Normal6(u;m_k,Sigma_k)',
            estimator='I_shell H_capture H_hard I_q J(u) [mean of two independent new Poisson W]/q(u)',
            fresh_data_required=True, existing_samples_used_for_estimate=False,
            scope='Frozen pilot guide only. This proposal has not been implemented or tested for variance reduction.')
        top = []
        for row, fraction in zip(ordered[:12], weights[:12]):
            top.append(dict(population=row['population'], draw=row['draw'], weight_fraction=float(fraction),
                            **geometry(row, config['initial_pose'], members,
                                       region['gaussian_chart']['angular_length'], .1200413863268778)))
        # With independent clouds, doubling their count halves the conditional-cloud contribution.
        ideal_cloud_rse_floor = row_rse*math.sqrt(max(0., 1-alpha))
        cost_target = {}
        for target in (.2, .1):
            multiplier = (row_rse/target)**2
            cost_target[str(target)] = dict(observed_N_multiplier=multiplier,
                approximate_total_draws=math.ceil(multiplier*len(combined)),
                approximate_total_CPU_seconds=multiplier*cpu,
                warning='Planning extrapolation from unresolved observed moments, not a reliable cost guarantee or convergence criterion.')
        results.append(dict(region=spec['region'], samples=len(combined), nonzero=len(valid),
            largest_fraction=float(weights[0]), top4_fraction=float(weights[:4].sum()),
            top12_fraction=float(weights[:12].sum()), top64_fraction=float(weights[:64].sum()),
            top_contributions=top, weighted_latent_mean=average.tolist(),
            weighted_latent_covariance=covariance.tolist(),
            weighted_latent_covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
            guide_caveat='Low-ESS weighted geometry is descriptive and cannot identify a converged basin shape.',
            paired_noise=cloud_noise,
            observed_cloud_variance_fraction=alpha,
            observed_pose_variance_fraction=1-alpha,
            current_row_relative_SE=row_rse,
            ideal_zero_cloud_noise_relative_SE_floor=ideal_cloud_rse_floor,
            four_clouds_same_pose_count_relative_SE=row_rse*math.sqrt(1-alpha/2),
            cloud_optimization_scope='At fixed number of poses. Extra clouds have positive cost; these are observed variance decompositions, not exact unknown variances.',
            brute_force_planning=cost_target, sampler_cpu_seconds=cpu))
    out.mkdir()
    for name, guide in guides.items():
        (out/f'guide-{name}.json').write_text(json.dumps(guide, indent=2)+'\n')
    result = dict(schema='mobile-outer-weight-diagnostic-v1', complete=True,
                  physical_calculations=0, audits_rerun=0, regions=results,
                  input_sha256=input_hashes,
                  guide_validation='Implementation remains necessary. Use independent new seeds after the guide is frozen; '
                  'none of these fitting rows may enter the new importance estimate.',
                  outside_coverage='No information here bounds mass beyond rho12 or other basins.')
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    shutil.copy2(Path(__file__), out/'analyzer.py')
    (out/'freeze.json').write_text(json.dumps({p.name: sha(p) for p in out.iterdir()}, indent=2)+'\n')
    print(json.dumps({r['region']: {k: r[k] for k in ['samples', 'top4_fraction', 'top12_fraction',
        'observed_cloud_variance_fraction', 'current_row_relative_SE',
        'ideal_zero_cloud_noise_relative_SE_floor', 'brute_force_planning']} for r in results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.campaign, args.out)
