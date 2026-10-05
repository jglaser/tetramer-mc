"""Reduce and plot audited covariance-guide coverage; no new physical queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_bytes())
    assert report['schema'] == 'context-covariance-guide-independent-audit-v1'
    assert report['complete'] and report['passed'] and report['decoded_candidates'] == 16384
    assert report['independent_panel_size'] == 512 and report['physical_weight_status'] == 'not_estimated'
    assert len(report['populations']) == 8
    results = []
    for item in report['populations']:
        candidates = [Path(p) for p in report['input_sha256']
                      if Path(p).name == 'rows.jsonl' and Path(p).parent.name == item['id']]
        assert len(candidates) == 1
        path = candidates[0]; summary_path = path.parent/'summary.json'
        assert sha(path) == report['input_sha256'][str(path)]
        assert sha(summary_path) == report['input_sha256'][str(summary_path)]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        summary = json.loads(summary_path.read_text())
        assert len(rows) == item['denominator'] == 2048
        source = [r for r in rows if r['input']['branch'] == 'source']
        weights = np.array([-r['density']['log_q'] for r in rows
                            if r['region'] == 'A_patch_complete'])
        hits = len(weights)
        assert hits == item['full_source217_inclusion_A_hits']
        if hits:
            total = float(logsumexp(weights))
            logmass = total-np.log(len(rows))
            ess = float(np.exp(2*total-logsumexp(2*weights)))
            largest = float(np.exp(max(weights)-total))
        else:
            logmass = None; ess = 0.; largest = None
        results.append(dict(id=item['id'], arm=item['arm'], stream=item['stream'], attempts=len(rows),
            source_attempts=len(source), source_valid=sum(r['actual']['physical_valid'] for r in source),
            source_complete_hits=sum(r['region'] == 'A_patch_complete' for r in source),
            complete_hits_all_branches=hits, hard_only_log_mass=logmass,
            hard_only_importance_ess=ess, largest_hard_only_contribution=largest,
            geometry_cpu_seconds=summary['cpu_seconds'],
            prospective_cloud_points=item['expected_complete_allocation_raw_points']))
    args.out.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for index, (arm, color) in enumerate([('full', '#087786'), ('diagonal', '#ba8735')]):
        group = sorted([r for r in results if r['arm'] == arm], key=lambda r:r['stream'])
        assert len(group) == 4
        x = index+np.linspace(-.12, .12, 4)
        coverage = np.array([r['source_valid']/r['source_attempts'] for r in group])
        axes[0].scatter(x, 100*coverage, color=color, s=40)
        axes[0].plot([index-.16,index+.16], [100*coverage.mean()]*2, color=color, lw=2)
        for pos, row in zip(x, group):
            if row['hard_only_log_mass'] is not None:
                axes[1].scatter(pos, np.exp(row['hard_only_log_mass']), color=color, s=40)
            else:
                axes[1].text(pos, .03, '×', ha='center', color=color,
                             transform=axes[1].get_xaxis_transform())
        axes[2].scatter(x, [r['hard_only_importance_ess'] for r in group], color=color, s=40)
    axes[0].set_title('Hard feasibility of the source Gaussian')
    axes[0].set_ylabel('Hard-valid / source-branch draws (%)')
    axes[0].set_ylim(bottom=0)
    axes[1].set_title('Complete source-pattern region')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Hard-only mass (Å³ × normalized Haar)')
    axes[2].set_title('Concentration of its geometric weights')
    axes[2].set_ylabel('Importance ESS per population')
    axes[2].axhline(200, color='#777777', ls='--', lw=1, label='Required regional ESS')
    axes[2].set_ylim(bottom=0); axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks([0,1], ['Full covariance', 'Diagonal control'])
        ax.set_xlim(-.4,1.4); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Can correlated translation and rotation recover the occupied contact pocket?', fontsize=14)
    fig.text(.5,.875, 'Same trained mean and marginal widths · ½ uniform + ¼ context atlas + ¼ source guide · four fresh populations per arm', ha='center',fontsize=10)
    fig.text(.5,.05, 'Each dot: 2,048 unconditional proposals; hard-invalid draws remain zero. T16 pattern is not a native-registry classifier.\n'
             'One mobile tetramer with 263 fixed spectators at 500 μM; no depletant clouds, equilibrium sampling or assembly conclusion.', ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.76,wspace=.35)
    for suffix in ['png','svg']:
        fig.savefig(args.out/f'covariance-guide-coverage.{suffix}',dpi=180)
    plt.close(fig)
    result = dict(complete=True,passed=True,source_sha256=sha(__file__),report_sha256=sha(args.report),
        populations=results,new_geometry_queries=0,clouds_sampled=0,
        scope='Descriptive fresh importance-proposal geometry, not physical weights or Markov-chain ESS.')
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
