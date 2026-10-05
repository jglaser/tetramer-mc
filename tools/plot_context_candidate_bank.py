#!/usr/bin/env python3
"""Plot a completed candidate-bank audit; no geometry queries or sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    assert report['schema'] == 'context-candidate-bank-independent-audit-v1'
    assert report['complete'] and report['passed']
    assert report['audited_candidates'] == 36864
    assert report['physical_weight_status'] == 'not_estimated'
    populations = [p for p in report['populations'] if p['phase'] == 'all']
    assert len(populations) == 16
    assert all(p['attempted'] == p['denominator'] == 2304 for p in populations)
    arms = ('original', 'context')
    by_arm = {arm: sorted([p for p in populations if p['arm'] == arm],
                          key=lambda p: (p['start'], p['stream'])) for arm in arms}
    assert all(len(ps) == 8 for ps in by_arm.values())
    assert not args.out_dir.exists(), 'Fresh figure output required'
    args.out_dir.mkdir(parents=True)
    colors = ('#b57625', '#087786')
    regions = ('A_patch_0_0.25', 'A_patch_0.25_0.5', 'A_patch_0.5_0.75',
               'A_patch_0.75_1', 'A_patch_complete')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4),
                             gridspec_kw={'width_ratios': [1.05, 1.4, 1]})
    partition_names = ('Invalid', 'A', 'B', 'Other contact', 'Unbound')
    partition_colors = ('#dadde0', '#526eab', '#aa6793', '#70b9a7', '#eee8bc')
    counts = {}
    for a, (arm, color) in enumerate(zip(arms, colors)):
        ps = by_arm[arm]
        totals = {}
        for p in ps:
            assert sum(r['attempted_count'] for r in p['partitions']) == 2304
            for r in p['partitions']:
                totals[r['region']] = totals.get(r['region'], 0) + r['attempted_count']
        counts[arm] = totals
        groups = (totals['hard_invalid'], sum(totals[r] for r in regions),
                  totals['B'], totals['other_contact'], totals['unbound'])
        assert sum(groups) == 18432
        left = 0.
        for k, (value, fill) in enumerate(zip(groups, partition_colors)):
            fraction = value / 18432
            axes[0].barh(a, fraction, left=left, color=fill, height=.55,
                         label=partition_names[k] if a == 0 else None)
            left += fraction
        for k, region in enumerate(regions):
            x = k + (a - .5) * .29
            values = [next(r['attempted_count'] for r in p['partitions']
                           if r['region'] == region) for p in ps]
            axes[1].scatter(x + np.linspace(-.06, .06, 8), values,
                            color=color, s=24, alpha=.7,
                            label=arm.title() if k == 0 else None)
            axes[1].plot([x-.08, x+.08], [np.mean(values)]*2, color=color, lw=2)
        widths = [p['log_interval_width']['mean'] for p in ps]
        assert all(w is not None and w >= 0 for w in widths)
        axes[2].scatter(a + np.linspace(-.12, .12, 8), widths, color=color, s=30)
    axes[0].set_yticks([0, 1], ['Original', 'Context'])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('Fraction of every attempted candidate')
    axes[0].set_title('Complete proposal allocation')
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, -.47),
                   ncol=2, frameon=False, fontsize=9)
    axes[1].set_xticks(range(5), ['[0, .25)', '[.25, .5)', '[.5, .75)', '[.75, 1)', 'All 16'])
    axes[1].set_yscale('symlog', linthresh=1)
    axes[1].set_ylim(bottom=-.12)
    axes[1].set_xlabel('Fraction of source secondary-contact tokens retained')
    axes[1].set_ylabel('Candidate hits per 2,304-draw population')
    axes[1].set_title('Patch coverage within neighbor set A')
    axes[1].legend(frameon=False)
    axes[2].set_xticks([0, 1], ['Original', 'Context'])
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel('Mean z × (upper − lower overlap volume)')
    axes[2].set_title('Geometric weight-bound uncertainty')
    for ax in axes[1:]:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Saved independent candidates: coverage before physical-weight sampling', fontsize=15)
    fig.text(.5, .89, '36,864 proposals · 8 paired populations per arm · all rejections and warmup retained', ha='center')
    fig.text(.5, .035,
             'One mobile tetramer, 263 fixed spectators at 500 μM; rd = 1.5 Å, z = 0.035 Å⁻³.\n'
             'A = neighbors {16, 217}; B = {16, 56}. Patch inclusion is not native registry or equilibrium weight. No Poisson clouds.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.79, bottom=.31, left=.07, right=.98, wspace=.4)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out_dir/f'candidate-bank-coverage.{suffix}', dpi=180)
    plt.close(fig)
    (args.out_dir/'provenance.json').write_text(json.dumps({
        'report': str(args.report.resolve()), 'report_sha256': hashlib.sha256(raw).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts': counts, 'new_draws': 0, 'new_geometry_queries': 0,
        'scope': 'Descriptive candidate coverage, no physical-weight inference.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
