#!/usr/bin/env python3
"""Plot saved all-stream summaries without geometry or physical sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_bytes())
    assert data['complete'] and data['passed'] and len(data['streams']) == 24
    assert not args.out_dir.exists(), 'Fresh figure destination required'
    args.out_dir.mkdir(parents=True)
    arms = ('local', 'original', 'context')
    labels = ('Local', 'Original atlas', 'Context prior')
    colors = ('#687886', '#d48728', '#16899a')
    starts = ('saved_body77', 'highest_original_prior_valid_neighbor_distinct_center')
    start_labels = ('Start A: saved contact', 'Start B: alternative contact')
    by_key = {(r['identity']['start'], r['identity']['arm'], r['identity']['stream']): r
              for r in data['streams']}
    assert len(by_key) == 24
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for row, start in enumerate(starts):
        ax = axes[row, 0]
        for arm_i, (arm, color) in enumerate(zip(arms, colors)):
            for stream in range(4):
                record = by_key[start, arm, stream]
                value = record['patch_ess']['apparent_ess_per_sampling_CPU_second']
                x = arm_i + (stream - 1.5) * .09
                if value is None:
                    ax.text(x, .04, '×', color=color, transform=ax.get_xaxis_transform(), ha='center')
                else:
                    ax.scatter(x, value, s=40, color=color, edgecolors='white', linewidth=.5, zorder=3)
        ax.set_yscale('log')
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylabel('Apparent patch ESS / full sampler CPU s')
        ax.set_title(start_labels[row])
        ax.grid(axis='y', alpha=.2)

        ax = axes[row, 1]
        for arm_i, arm in enumerate(arms):
            for stream in range(4):
                record = by_key[start, arm, stream]
                y = arm_i * 5 + stream
                left = 0.
                for env, color in zip(('A', 'B', 'Other'), ('#5072b0', '#b367aa', '#d5d8dd')):
                    fraction = record['ABOther'][env]['fraction']
                    ax.barh(y, fraction, left=left, height=.78, color=color,
                            label=env if arm_i == stream == 0 else None)
                    left += fraction
                assert abs(left - 1) < 1e-10
        ax.set_yticks([1.5, 6.5, 11.5], labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel('Unconditional retained-state occupancy')
        ax.set_title('Each thin bar is one independent stream')
        if row == 0:
            handles, env_labels = ax.get_legend_handles_labels()
            fig.legend(handles, env_labels, loc='upper center', bbox_to_anchor=(.5, .915), ncol=3, frameon=False)

        ax = axes[row, 2]
        # Timers need not be disjoint, so plot the measured full cost only.
        for arm_i, (arm, color) in enumerate(zip(arms, colors)):
            for stream in range(4):
                record = by_key[start, arm, stream]
                ax.scatter(arm_i + (stream - 1.5) * .09,
                           record['costs']['invocation_cpu_seconds'], s=40, color=color,
                           edgecolors='white', linewidth=.5)
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylabel('Full sampler CPU seconds / chain')
        ax.set_ylim(bottom=0)
        ax.set_title('Matched 11,520 attempts / chain')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed-neighborhood proposal comparison', fontsize=16, y=.99)
    fig.text(.5, .935, 'One moving tetramer, 263 fixed spectators · depletant radius 1.5 Å · activity 0.035 Å⁻³', ha='center')
    tv = {r['arm']: r['fingerprint']['total_variation']
          for r in data['equal_stream_average_start_comparisons'] if r['cadence'] == 'all_attempts'}
    support = '; '.join(f'{label} {tv[arm]:.2f}' for arm, label in zip(arms, labels))
    fig.text(.04, .025, 'A = {16,217}; B = {16,56}; matching neighbors need not mean matching surface patches. All rejections retain residence weight.\n'
             'Between-start fingerprint TV (0 = agreement, 1 = disjoint observed support): ' + support + '.\n'
             '× = constant descriptor / undefined ESS. Short-record ESS is diagnostic; these conditional runs do not determine assembly stability.',
             fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .89), h_pad=2.5, w_pad=2.5)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out_dir / ('contact-prior-comparison.' + suffix), dpi=180)
    plt.close(fig)
    receipt = dict(input=str(args.comparison.resolve()),
                   input_sha256=hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   stream_records=24, geometry_queries=0, physical_draws=0,
                   outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(args.out_dir.iterdir())})
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
