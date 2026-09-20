#!/usr/bin/env python3
"""Compare exact fixed-region membership with independently estimated mass bounds."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from prepare_smc_normalizer_atlas import ROOT, Density, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration
from analyze_local_region_overlap import split_fixed_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'runs/outside-r8-local-region-20260920/site0/validation-plan.json')
    parser.add_argument('--trajectories', type=Path, default=ROOT/'runs/posterior-docking-mis-5000')
    parser.add_argument('--old-mass', type=Path, default=ROOT/'runs/basin-normalizer-mis-refined-16384-l64/assessment/deep-shells.json')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/outside-r8-production-20260920/assessment')
    args = parser.parse_args()
    plan, old_mass = read(args.plan), read(args.old_mass)
    old = read(plan['poststratification']['old_region'])
    assert old_mass['chart_region_sha256'] == sha(plan['poststratification']['old_region'])
    reference_config_path = args.old_mass.resolve().parent.parent/'provenance/config.json'
    reference_config = read(reference_config_path)
    assert reference_config['fixed_poses'] == [old['fixed_neighbor']]
    assert reference_config['capture_center'] == old['capture_center'] and reference_config['capture_radius'] == old['capture_radius']
    assert reference_config['reservoir_density'] == old['activity'] and reference_config['depletant_radius'] == old['depletant_radius']
    assert sha(reference_config['shape']) == old['shape_sha256']
    for field in ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg'):
        assert reference_config['metadata'][field] == old['physical_metric'][field]
    regions = {key:read(value['path']) for key,value in plan['regions'].items()}
    assert regions['3']['gaussian_chart'] == regions['5']['gaussian_chart']
    new_density, old_density = Density(regions['3']['gaussian_chart']), Density(old['gaussian_chart'])
    bounds = {}
    ref = old_mass['estimates']['r_le_8']
    old_pop_logs = [p['estimates']['r_le_8']['logQ'] for p in old_mass['populations']]
    for key in ('3', '5'):
        root = Path(plan['suggested_outputs'][key])
        part = read(root/'assessment/old-region-overlap.json')
        estimate = part['estimates']['outside_old_R8']
        poplogs = estimate['independent_populations']['logQ_values']
        central = float(expit(estimate['logQ']-ref['logQ']))
        spread = float(expit(max(poplogs)-min(old_pop_logs)))
        numerator = estimate['logQ']+np.log1p(2*estimate['relative_se'])
        denominator = ref['logQ']+np.log1p(-2*ref['relative_se'])
        bounds[key] = dict(new_radius=float(key), B_logQ=estimate['logQ'], B_observed_relative_se=estimate['relative_se'],
            B_ess=estimate['ess'], B_max_fraction=estimate['max_fraction'], B_population_logQ=poplogs,
            old_R8_logQ=ref['logQ'], old_R8_observed_relative_se=ref['relative_se'],
            estimated_population_bound=central,
            empirical_max_B_min_A_population_ratio=spread,
            observed_two_SE_sensitivity_ratio=float(expit(numerator-denominator)),
            exact_inequality='For disjoint A=oldR8 and B=newR intersect outside oldR8, pi(B)=Q_B/Q_total <= Q_B/(Q_B+Q_A). No global normalizer is needed.',
            caveat='The inequality is exact for true masses. Plug-in ratios, population extremes, and observed2SE sensitivities are estimated diagnostics, not certified upper confidence bounds; unseen high weights in B remain possible.',
            mass_report_sha256=sha(root/'assessment/old-region-overlap.json'))

    trajectories = []
    manifest = read(args.trajectories/'manifest.json')
    for job in manifest['jobs']:
        directory = Path(job['directory'])
        cfg = read(job['config'])
        assert cfg['fixed_poses'] == [old['fixed_neighbor']]
        assert cfg['capture_center'] == old['capture_center'] and cfg['capture_radius'] == old['capture_radius']
        assert cfg['reservoir_density'] == old['activity'] and cfg['depletant_radius'] == old['depletant_radius']
        for field in ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg'):
            assert cfg['metadata'][field] == old['physical_metric'][field]
        frames = [json.loads(line) for line in (directory/'trajectory.jsonl').open()]
        assert [f['cycle'] for f in frames] == list(range(manifest['cycles']+1))
        poses = [f['pose'] for f in frames]
        rel = relative_poses(poses, old['fixed_neighbor'])
        q = registration(poses, old['physical_metric'])
        nr, ar = new_density.evaluate(rel)[1][:,0], old_density.evaluate(rel)[1][:,0]
        outside = (q >= 5.) & (ar > 8.)
        subsets = {}
        for key in ('3', '5'):
            selected = outside & (nr <= float(key))
            counts = dict(all_retained_frames=len(frames), selected=int(selected.sum()),
                fraction_all_retained_frames=float(selected.mean()), outside_old_R8_frames=int(outside.sum()),
                fraction_of_outside_old_R8=float(selected.sum()/outside.sum()) if outside.any() else None,
                fraction_excluding_initial=float(selected[1:].mean()),
                first_half_fraction=float(selected[1:manifest['cycles']//2+1].mean()),
                last_half_fraction=float(selected[manifest['cycles']//2+1:].mean()))
            runs, start = [], None
            for cycle, value in enumerate(np.r_[selected, False]):
                if value and start is None:
                    start = cycle
                elif not value and start is not None:
                    runs.append(dict(first_cycle=start, last_cycle=cycle-1, retained_frames=cycle-start))
                    start = None
            counts['dwell_intervals'] = runs
            subsets[key] = counts
        trajectories.append(dict(id=job['id'], start=job['start'], replicate=job['replicate'], mode=job['mode'],
            source_trajectory_sha256=sha(directory/'trajectory.jsonl'), subsets=subsets))

    # An independent common-subregion comparison: retain all R5 denominator
    # draws while restricting their weights to the already frozen new R3.
    common = {}
    max_cloud_error = 0.
    for key in ('3', '5'):
        root = Path(plan['suggested_outputs'][key]); campaign = read(root/'manifest.json')
        all_logs, all_hard, all_pairs, all_inside, populations = [], [], [], [], []
        for job in campaign['jobs']:
            directory = Path(job['directory']); summary = read(directory/'summary.json')
            assert summary['complete'] and summary['samples'] == 16384
            physical = summary['manifest']
            assert physical['activity'] == old['activity'] and physical['lambda_ratio'] == 64. and physical['cloud_replicates'] == 2
            rows = [json.loads(line) for line in (directory/'samples.jsonl').open()]
            assert len(rows) == 16384
            r_old = old_density.evaluate(relative_poses([row['pose'] for row in rows], old['fixed_neighbor']))[1][:,0]
            logs, hard, pairs = [], [], []
            for row in rows:
                for cloud in row['clouds']:
                    expected = physical['activity']*cloud['lower_volume']+cloud['overlap_points']*np.log1p(physical['activity']/physical['lambda'])
                    error = abs(expected-cloud['log_weight'])
                    assert error < 1e-10
                    max_cloud_error = max(max_cloud_error,error)
                selected = row['log_importance_weight'] is not None and np.linalg.norm(row['latent']) <= 3.
                if selected:
                    logs.append(row['log_importance_weight']);hard.append(row['log_hard_weight'])
                    pairs.append([row['log_hard_weight']+c['log_weight'] for c in row['clouds']])
                else:
                    logs.append(-np.inf);hard.append(-np.inf);pairs.append([-np.inf,-np.inf])
            inside = r_old <= 8.
            populations.append(dict(id=job['id'], estimates=split_fixed_n(logs,hard,pairs,inside)))
            all_logs.extend(logs);all_hard.extend(hard);all_pairs.extend(pairs);all_inside.extend(inside)
        common[key] = dict(estimates=split_fixed_n(all_logs,all_hard,all_pairs,all_inside), populations=populations,
            scope='Same fixed newR3 physical subset under an independent original-newR3 or original-newR5 uniform-ball sampling campaign. All65536 draws remain in each denominator.')
    comparisons = {}
    for key in ('total','inside_old_R8','outside_old_R8'):
        a,b = common['3']['estimates'][key],common['5']['estimates'][key]
        scale = max(a['logQ'],b['logQ'])
        va,vb = np.exp(a['logQ']-scale),np.exp(b['logQ']-scale)
        se = np.hypot(va*a['relative_se'],vb*b['relative_se'])
        comparisons[key] = dict(logQ_from_R3=a['logQ'],logQ_from_R5=b['logQ'],difference_in_observed_combined_SE=float((va-vb)/se),
            caveat='Observed-sample variance only; this agreement test does not certify unseen weights.')
    out = args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    result = dict(plan_sha256=sha(args.plan),old_mass_sha256=sha(args.old_mass),old_mass_config_sha256=sha(reference_config_path),old_R8_estimate=ref,
        region_bounds=bounds,trajectories=trajectories,common_new_R3_integrals=common,common_region_comparisons=comparisons,
        positive_Poisson_weight_max_log_error=max_cloud_error,analyzer_sha256=sha(__file__),
        interpretation='These selected finite records can be compared with independently estimated fixed-region masses. Their cohort-defined regions are data-dependent; the result diagnoses empirical nonrepresentativeness and long residence, not a calibrated test rejecting a stationary correlated process. No new simulation or model fit.')
    write(out/'mass-and-residence.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4.5))
    x=np.arange(len(trajectories))
    for key,offset,color in [('3',-.18,'#4794a4'),('5',.18,'#da8b4b')]:
        values=[100*t['subsets'][key]['fraction_all_retained_frames'] for t in trajectories]
        ax.bar(x+offset,values,width=.36,label=f'New R{key} outside old R8',color=color)
        ax.axhline(100*bounds[key]['estimated_population_bound'],color=color,ls='--',lw=1.3,
            label=f'R{key}: exact bound evaluated at estimated masses ({100*bounds[key]["estimated_population_bound"]:.2f}%)')
    ax.set_xticks(x,[f"{t['start']} {t['replicate']}\n{'redraw' if t['mode']=='c0' else 'transport'}" for t in trajectories])
    ax.set_ylabel('Fraction of all retained frames / %');ax.set_ylim(0,70)
    ax.set_title('Long residence in a region with independently estimated small physical mass')
    ax.legend(fontsize=8,loc='upper left');fig.tight_layout()
    fig.savefig(out/'residence-versus-regional-mass.png',dpi=170)
    fig.savefig(out/'residence-versus-regional-mass.svg')
    plt.close(fig)
    print(json.dumps(dict(region_bounds=bounds,common_region_comparisons=comparisons,
        residence=[dict(id=t['id'],R3=t['subsets']['3']['fraction_all_retained_frames'],R5=t['subsets']['5']['fraction_all_retained_frames']) for t in trajectories],
        max_cloud_error=max_cloud_error),indent=2))


if __name__=='__main__':main()
