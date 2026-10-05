#!/usr/bin/env python3
"""Plot completed matched rho=0 and rho=.95 summaries; no geometry or sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--correlated', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    old, new = [json.loads(p.read_bytes()) for p in (args.baseline, args.correlated)]
    assert old['complete'] and old['passed'] and new['complete'] and new['passed']
    assert new['schema'] == 'context-correlated095-pilot-comparison-v1'
    assert new['new_chains'] == 16 and new['reused_local_controls'] == 8
    assert new['input_sha256'][str(args.baseline.resolve())] == hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    assert not args.out_dir.exists(), 'Fresh figure destination required'
    args.out_dir.mkdir(parents=True)
    arms = [('local', 0), ('original', 0), ('context', 0), ('original', .95), ('context', .95)]
    labels = ['Local', 'Original\nρ = 0', 'Context\nρ = 0', 'Original\nρ = .95', 'Context\nρ = .95']
    colors = ['#687886', '#dca760', '#70b9c2', '#a86613', '#087786']
    starts = ('saved_body77', 'highest_original_prior_valid_neighbor_distinct_center')
    title_starts = ('Start A: saved contact', 'Start B: alternative contact')
    records = {}
    for report, rho in ((old, 0), (new, .95)):
        for r in report['streams']:
            ident = r['identity']
            if rho and ident['arm'] == 'local':
                continue  # Reused controls are counted only once.
            records[ident['start'], ident['arm'], rho, ident['stream']] = r
    assert len(records) == 40
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
    for row, start in enumerate(starts):
        for k, ((arm, rho), color) in enumerate(zip(arms, colors)):
            for stream in range(4):
                r = records[start, arm, rho, stream]
                x = k + (stream - 1.5) * .10
                value = r['patch_ess']['apparent_ess_per_sampling_CPU_second']
                if value is None:
                    axes[row, 0].text(x, .04, '×', color=color, ha='center',
                                      transform=axes[row, 0].get_xaxis_transform())
                else:
                    axes[row, 0].scatter(x, value, color=color, s=40, edgecolors='white', linewidth=.5)
                axes[row, 2].scatter(x, r['costs']['invocation_cpu_seconds'], color=color,
                                     s=40, edgecolors='white', linewidth=.5)
                left = 0.
                for env, fill in zip(('A', 'B', 'Other'), ('#5072b0', '#b367aa', '#d5d8dd')):
                    fraction = r['ABOther'][env]['fraction']
                    axes[row, 1].barh(k * 5 + stream, fraction, left=left, height=.8,
                                      color=fill, label=env if row == k == stream == 0 else None)
                    left += fraction
                assert abs(left - 1.) < 1e-10
        for col in (0, 2):
            axes[row, col].set_xticks(range(5), labels)
            axes[row, col].set_xlim(-.5, 4.5)
            axes[row, col].grid(axis='y', alpha=.2)
        axes[row, 0].set_yscale('log')
        axes[row, 0].set_ylabel('Apparent patch ESS / full sampler CPU s')
        axes[row, 0].set_title(title_starts[row])
        axes[row, 1].set_yticks([k * 5 + 1.5 for k in range(5)], labels)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_title('Each thin bar is one independent stream')
        axes[row, 1].set_xlabel('Unconditional retained-state occupancy')
        axes[row, 2].set_title('Matched 11,520 attempts / chain')
        axes[row, 2].set_ylabel('Full sampler CPU seconds / chain')
        axes[row, 2].set_ylim(bottom=0)
    handles, names = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .915), ncol=3, frameon=False)
    fig.suptitle('Matched contact transport: independent redraw versus correlated transfer', fontsize=16, y=.99)
    fig.text(.5, .948, 'One moving tetramer, 263 fixed spectators · depletant radius 1.5 Å · activity 0.035 Å⁻³', ha='center')
    tv = {}
    for report, rho in ((old, 0), (new, .95)):
        for r in report['equal_stream_average_start_comparisons']:
            if r['cadence'] == 'all_attempts':
                tv[r['arm'], rho] = r['fingerprint']['total_variation']
    support = '; '.join(f'{arm} ρ={rho:g}: {tv[arm, rho]:.2f}' for arm, rho in arms)
    fig.text(.035, .025,
             'A = {16,217}; B = {16,56}; matching neighbors need not mean matching surface patches. All rejections retain residence weight.\n'
             'Between-start fingerprint TV (0 = agreement, 1 = disjoint observed support): ' + support + '.\n'
             '× = constant descriptor / undefined ESS. Controls are reused once; finite-record ESS does not establish equilibrium or assembly stability.',
             fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .89), h_pad=2.5, w_pad=2.0)
    for extension in ('png', 'svg'):
        fig.savefig(args.out_dir / ('correlated-contact-comparison.' + extension), dpi=180)
    plt.close(fig)
    receipt = dict(inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.baseline, args.correlated)},
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   stream_records=40, reused_controls=24, new_chains=16,
                   geometry_queries=0, physical_draws=0,
                   outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(args.out_dir.iterdir())})
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
