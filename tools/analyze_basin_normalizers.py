#!/usr/bin/env python3
"""Independent fixed-N aggregation and paired-Poisson diagnostics for normalizers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

QBINS = ['native_core', 'native_shell', 'shoulder', 'intermediate', 'distant']
REGIONS = [q + '_' + bound for q in QBINS for bound in ['bound', 'unbound']]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return float(x) if np.isfinite(x) else None


def moments(logs):
    logs = np.asarray(logs)
    n = len(logs)
    nonzero = int(np.isfinite(logs).sum())
    if not nonzero:
        return dict(draws=n, nonzero=0, logQ=None, ess=0., relative_se=None,
                    max_fraction=None, top_one_percent_fraction=None,
                    coverage='Unresolved: no nonzero observations, not a physical zero or upper bound')
    total = logsumexp(logs)
    ess = float(np.exp(2*total-logsumexp(2*logs)))
    top = max(1, math.ceil(.01*n))
    return dict(draws=n, nonzero=nonzero, logQ=float(total-np.log(n)), ess=ess,
                relative_se=float(np.sqrt(max(0., (n/ess-1)/(n-1)))) if n > 1 else None,
                max_fraction=float(np.exp(max(logs)-total)),
                top_one_percent_fraction=float(np.exp(logsumexp(np.sort(logs)[-top:])-total)),
                top_one_percent_draw_count=top,
                coverage='Observed-sample uncertainty; unseen high-weight regions remain possible')


def paired_noise(logs, pairs):
    """Exact algebraic decomposition of observed per-draw variance, with zeros."""
    if len(logs) < 2 or not np.isfinite(logs).any():
        return None
    offset = max(np.max(logs), np.max(pairs))
    a, b = np.exp(pairs[:, 0]-offset), np.exp(pairs[:, 1]-offset)
    y = (a+b)/2
    total_var = float(np.var(y, ddof=1))
    # E[(W1-W2)^2 / 4] is the noise variance of the two-cloud average.
    cloud_var = float(np.mean((a-b)**2)/4)
    pose_var = total_var-cloud_var
    return dict(log_weight_offset=float(offset), scaled_total_variance=total_var,
        scaled_paired_cloud_variance=cloud_var, scaled_residual_pose_variance=pose_var,
        paired_cloud_variance_fraction=cloud_var/total_var if total_var > 0 else None,
        relative_variance_of_mean_cloud=cloud_var/len(y)/np.mean(y)**2,
        relative_variance_of_mean_pose=pose_var/len(y)/np.mean(y)**2,
        scaled_mean_W1W2=float(np.mean(a*b)), scaled_mean_squared_cloud_difference=float(np.mean((a-b)**2)),
        scope='Paired independent clouds at each pose, with zero weights for invalid draws. Residual pose variance may be negative due to finite sampling; it is not clipped. This diagnostic cannot certify unseen tails.')


def selection(row, name):
    if not row['hard_valid']:
        return False
    if name == 'total':
        return True
    if name == 'native':
        return row['q'] <= 1
    if name == 'other':
        return row['q'] > 1
    if name in QBINS:
        return row['region'].rsplit('_', 1)[0] == name
    if name in ['bound', 'unbound']:
        return row['depletion_contact'] == (name == 'bound')
    return row['region'] == name


def population(job):
    root = Path(job['directory'])
    summary, manifest = read(root/'summary.json'), read(root/'manifest.json')
    assert summary['complete'] and manifest['seed'] == job['seed']
    assert manifest['covariance_scale'] == job['covariance_std_scale']
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    assert [r['draw'] for r in rows] == list(range(job['samples']))
    assert manifest['samples'] == len(rows) and manifest['cloud_replicates'] == 2
    for row in rows:
        if row['hard_valid']:
            assert row['capture_valid'] and row['region'] in REGIONS
            expected = ('native_core' if row['q'] <= .8 else 'native_shell' if row['q'] <= 1
                        else 'shoulder' if row['q'] < 2 else 'intermediate' if row['q'] < 5 else 'distant')
            assert row['region'] == expected + ('_bound' if row['depletion_contact'] else '_unbound')
            assert len(row['clouds']) == 2
            lw = logsumexp([c['log_weight'] for c in row['clouds']])-np.log(2)-row['log_proposal_density']
            assert abs(lw-row['log_importance_weight']) < 1e-10
            for cloud in row['clouds']:
                expected_weight = manifest['activity']*cloud['lower_volume'] + cloud['overlap_points']*np.log1p(manifest['activity']/manifest['lambda'])
                assert abs(expected_weight-cloud['log_weight']) < 1e-10
        else:
            assert row['log_importance_weight'] is None and not row['clouds']
    names = ['total', 'native', 'other', 'bound', 'unbound'] + QBINS + REGIONS
    estimates = {}
    for name in names:
        logs = np.array([r['log_importance_weight'] if selection(r, name) else -np.inf for r in rows])
        pairs = np.array([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']]
                          if selection(r, name) else [-np.inf, -np.inf] for r in rows])
        estimates[name] = moments(logs)
        estimates[name]['paired_noise'] = paired_noise(logs, pairs)
        if name in summary['estimates']:
            recorded = summary['estimates'][name]['log_normalizer']
            assert recorded is None if estimates[name]['logQ'] is None else abs(recorded-estimates[name]['logQ']) < 1e-10
    region_logs = [estimates[k]['logQ'] for k in REGIONS if estimates[k]['logQ'] is not None]
    if region_logs:
        assert abs(logsumexp(region_logs)-estimates['total']['logQ']) < 1e-10
    else:
        assert estimates['total']['logQ'] is None
    components = {}
    labels = sorted({str(r['proposal'].get('component_index')) for r in rows})
    for label in labels:
        selected = [r for r in rows if str(r['proposal'].get('component_index')) == label]
        logs = np.array([r['log_importance_weight'] if r['hard_valid'] and str(r['proposal'].get('component_index')) == label else -np.inf for r in rows])
        components['uniform' if label == 'None' else label] = dict(draws=len(selected),
            hard_valid=sum(r['hard_valid'] for r in selected), contribution=moments(logs))
    bounds = [manifest['activity']**2*c['uncertain_volume']/manifest['lambda'] for r in rows if r['hard_valid'] for c in r['clouds'][:1]]
    return dict(job=job, estimates=estimates, components=components, summary=summary,
                conditional_cloud_log_relative_second_moment_upper_quantiles=dict(zip(['min','median','p90','max'], np.quantile(bounds,[0,.5,.9,1]).tolist())) if bounds else None,
                rows=rows)


def analyze(root):
    manifest = read(root/'manifest.json')
    complete, pending = [], []
    for job in manifest['jobs']:
        path = Path(job['directory'])/'summary.json'
        (complete if path.exists() and read(path).get('complete') else pending).append(job)
    populations = [population(j) for j in complete]
    for p in populations:
        m = p['summary']['manifest']
        assert m['model_sha256'] == manifest['archive_sha256']['model.json']
        assert m['config_sha256'] == manifest['archive_sha256']['config.json']
        assert m['shape_sha256'] == manifest['archive_sha256']['shape.json']
        assert m['executable_sha256'] == manifest['archive_sha256']['basin-normalizer']
        assert m['activity'] == manifest['physical']['activity']
        assert math.isclose(m['lambda'], m['activity']*manifest['physical']['lambda_ratio'], rel_tol=1e-14)
    groups = {}
    for scale in sorted({j['covariance_std_scale'] for j in complete}):
        pops = [p for p in populations if p['job']['covariance_std_scale'] == scale]
        rows = sum([p['rows'] for p in pops], [])
        estimates = {}
        for name in pops[0]['estimates']:
            logs = np.array([r['log_importance_weight'] if selection(r,name) else -np.inf for r in rows])
            pairs = np.array([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']]
                if selection(r,name) else [-np.inf,-np.inf] for r in rows])
            estimates[name] = moments(logs)
            estimates[name]['paired_noise'] = paired_noise(logs,pairs)
            population_logs = np.array([p['estimates'][name]['logQ'] if p['estimates'][name]['logQ'] is not None else -np.inf for p in pops])
            estimates[name]['independent_population_estimates'] = moments(population_logs)
            estimates[name]['independent_population_estimates']['logQ_values'] = [finite(x) for x in population_logs]
            assert estimates[name]['logQ'] is None if not np.isfinite(population_logs).any() else abs(logsumexp(population_logs)-np.log(len(pops))-estimates[name]['logQ']) < 1e-10
        total = estimates['total']['logQ']
        fractions = {name: np.exp(v['logQ']-total) if v['logQ'] is not None and total is not None else None for name,v in estimates.items()}
        def delta(a,b):
            return estimates[b]['logQ']-estimates[a]['logQ'] if estimates[b]['logQ'] is not None and estimates[a]['logQ'] is not None else None
        components = {}
        for label in sorted({str(r['proposal'].get('component_index')) for r in rows}):
            logs = np.array([r['log_importance_weight'] if r['hard_valid'] and str(r['proposal'].get('component_index')) == label else -np.inf for r in rows])
            c = moments(logs)
            components['uniform' if label == 'None' else label] = dict(estimates=c,
                fraction_of_observed_total=np.exp(c['logQ']-total) if c['logQ'] is not None else None,
                selected_draws=sum(str(r['proposal'].get('component_index')) == label for r in rows))
        groups[str(scale)] = dict(covariance_std_scale=scale, populations=len(pops), unconditional_draws=len(rows),
            model_sha256=pops[0]['summary']['manifest']['model_sha256'],
            config_sha256=pops[0]['summary']['manifest']['config_sha256'],
            poisson_lambda=pops[0]['summary']['manifest']['lambda'],
            poisson_lambda_ratio=manifest['physical']['lambda_ratio'], activity=manifest['physical']['activity'],
            estimates=estimates, fraction_of_observed_total=fractions, selected_component_contributions=components,
            F_native_minus_other_kBT=delta('native','other'), F_native_minus_distant_kBT=delta('native','distant'),
            sampler_cpu_seconds=sum(p['summary']['sampler_cpu_seconds'] for p in pops))
    for p in populations:
        p.pop('rows')
    out = root/'assessment';out.mkdir(exist_ok=True)
    provenance = {str(Path(j['directory'])/name):sha(Path(j['directory'])/name) for j in complete for name in ['summary.json','samples.jsonl','manifest.json']}
    provenance[str(Path(__file__).resolve())]=sha(__file__)
    provenance[str(root/'manifest.json')]=sha(root/'manifest.json')
    result = dict(groups=groups, populations=populations, pending=pending, provenance=provenance,
        limitation='Independent populations per scale are exploratory. Fixed-N invalid zeros are retained. Missing bins remain unresolved, ratios are plug-in estimates, and observed ESS cannot establish tail coverage. Selected proposal components are auxiliary labels, not physical basins.',
        cloud_bound='For one cloud, E[W^2|pose]/E[W|pose]^2 <= exp(z^2 uncertain_volume/lambda). Quantiles report the logarithm of this conditional geometric upper bound, not a measured RSE.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Independent normalizer pilot','',result['limitation'],'',f'Completed {len(complete)}/{len(manifest["jobs"])} populations.',
        f'Frozen model SHA-256: `{manifest["archive_sha256"]["model.json"]}`. Auxiliary lambda/z: {manifest["physical"]["lambda_ratio"]:g}. Physical activity: {manifest["physical"]["activity"]:g} Å⁻³. No pooling across models or auxiliary intensities.','',
        '| Std scale | Draws | log Q native | log Q other | F native−other / kBT | Native fraction | Shoulder fraction | Distant fraction | Total ESS | Total top 1% share | CPU s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def f(value):return 'unresolved' if value is None else f'{value:.4g}'
    for key,g in groups.items():
        e=g['estimates'];r=g['fraction_of_observed_total']
        lines.append('| '+' | '.join([key,str(g['unconditional_draws']),f(e['native']['logQ']),f(e['other']['logQ']),f(g['F_native_minus_other_kBT']),f(r['native']),f(r['shoulder']),f(r['distant']),f(e['total']['ess']),f(e['total']['top_one_percent_fraction']),f(g['sampler_cpu_seconds'])])+' |')
    lines += ['', 'All ten exhaustive contact/registration region estimates, per-population values, paired-cloud variance decomposition, and selected proposal-component contributions are in [analysis.json](analysis.json).', '',
        'The q partitions are q≤0.8, 0.8<q≤1, 1<q<2, 2≤q<5, q≥5, each split by exact exclusion contact. The old SMC native partition is q≤1 and its other partition is all q>1. Zero observations provide no upper bound on missing physical mass.', '',
        '![Observed normalizers and cloud/pose variance](normalizer-pilot.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    plot(result, out)
    print('\n'.join(lines))


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    styles = [('native','#087f8c','o'), ('shoulder','#e9a038','s'), ('distant','#4c5a89','^')]
    for name,color,marker in styles:
        xy = [(g['covariance_std_scale'],g['estimates'][name]['logQ'])
              for g in result['groups'].values() if g['estimates'][name]['logQ'] is not None]
        if xy:
            x,y=zip(*xy);axes[0].plot(x,y,marker=marker,color=color,label=name)
    axes[0].set(xscale='log',xlabel='Gaussian standard-deviation multiplier',ylabel='Observed log Q',
                title='Proposal-width sensitivity')
    axes[0].set_xticks([g['covariance_std_scale'] for g in result['groups'].values()],
                      labels=[str(g['covariance_std_scale']) for g in result['groups'].values()])
    axes[0].legend()
    groups=list(result['groups'].values())
    x=np.arange(len(groups))
    cloud=[g['estimates']['total']['paired_noise']['paired_cloud_variance_fraction']
           if g['estimates']['total']['paired_noise'] else np.nan for g in groups]
    axes[1].bar(x,cloud,color='#87b8af',label='Paired-cloud noise')
    axes[1].bar(x,1-np.array(cloud),bottom=cloud,color='#ccd5e1',label='Residual pose variation')
    for i,g in enumerate(groups):
        axes[1].text(i,.5,f"ESS {g['estimates']['total']['ess']:.1f}",ha='center',fontsize=9)
    axes[1].set(xticks=x,xticklabels=[str(g['covariance_std_scale']) for g in groups],
        xlabel='Gaussian standard-deviation multiplier',ylabel='Fraction of observed weight variance',
        title='Separating observed cloud and pose variance')
    axes[1].legend(fontsize=8)
    sizes = {(g['populations'],g['unconditional_draws']//g['populations']) for g in groups}
    size = f'{next(iter(sizes))[0]} × {next(iter(sizes))[1]} draws per width' if len(sizes) == 1 else 'population sizes listed in report'
    fig.suptitle('Exploratory independent importance pilot: '+size+'\nZero observations are unresolved; these normalizers are not converged',fontsize=10)
    for extension in ['png','svg','pdf']:
        fig.savefig(out/f'normalizer-pilot.{extension}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root.resolve())
