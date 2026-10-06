"""Plot a completed geometry-only bridge comparison without new queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('baseline', 'bridge')
COLORS = ('#4477aa', '#cc6677')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Changed completed report')
    report = json.loads(args.report.read_bytes())
    if not (report['schema'] == 'context-bridge-guide-independent-audit-v1'
            and report['complete'] and report['passed']
            and report['all_attempts'] == 16384
            and report['physical_weight_status'] == 'not_estimated'
            and report['clouds_generated'] == 0):
        raise ValueError('Completed geometry-only comparison required')
    path = args.report.parent/'contributions.jsonl'
    if sha(path) != report['contributions_sha256']:
        raise ValueError('Changed all-attempt records')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 16384 or len({(r['stratum_id'], r['ordinal']) for r in rows}) != 16384:
        raise ValueError('Incomplete or duplicate attempt inventory')
    for row in rows:
        row['token_count'] = None
        if row['physical_valid']:
            count = round(16*row['source_fraction'])
            if count not in range(17) or abs(row['source_fraction']-count/16) > 1e-13:
                raise ValueError('Changed source-token denominator')
            row['token_count'] = count
    for arm in ARMS:
        for population in range(4):
            if sum(r['comparison_arm'] == arm and r['population_index'] == population for r in rows) != 2048:
                raise ValueError('Changed unconditional population denominator')
    if args.out.exists():
        raise ValueError('Fresh output directory required')
    args.out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    counts_by_arm = {}
    for index, arm in enumerate(ARMS):
        selected = [r for r in rows if r['comparison_arm'] == arm]
        counts_by_arm[arm] = {}
        for ax, key, condition in [
                (axes[0, 0], 'all_neighbors', lambda r: True),
                (axes[0, 1], 'exact_A', lambda r: r['neighbor_labels'] == [16, 217])]:
            counts = np.bincount([r['token_count'] for r in selected
                                 if r['physical_valid'] and condition(r)], minlength=17)
            counts_by_arm[arm][key] = counts.tolist()
            ax.plot(range(17), counts, 'o-', color=COLORS[index], markersize=4,
                    label=('Existing guides', '+ Coupled bridge guides')[index])
            ax.set_yscale('symlog', linthresh=1)
            ax.set_xticks(range(0, 17, 2))
            ax.set_xlim(-.5, 16.5)
            ax.set_ylabel('Hard-valid proposals / 8,192 attempted poses')
            ax.set_xlabel('Retained source-contact tokens (out of 16)')
            ax.axvspan(7.5, 15.5, color='#cccccc', alpha=.18)
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].set_title('All neighbor environments')
    axes[0, 1].set_title('Unchanged neighbor set A = {16, 217}')
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].text(.03, .97, 'Shading: predeclared target, 8–15 tokens',
                    transform=axes[0, 1].transAxes, va='top', fontsize=8)
    diagnostics = []
    for width_index, width in enumerate((1, 4)):
        color = ('#228833', '#aa3377')[width_index]
        for alpha_index in range(4):
            name = f'bridge_a{alpha_index}_b{width}'
            group = [r for r in rows if r['comparison_arm'] == 'bridge'
                     and r['component'] == name and r['branch'] == 'source']
            point = alpha_index+(-.06 if width == 1 else .06)
            metrics = []
            for population in range(4):
                sample = [r for r in group if r['population_index'] == population]
                metrics.append(dict(population_index=population, attempts=len(sample),
                    valid=sum(r['physical_valid'] for r in sample),
                    intermediate=sum(r['physical_valid'] and 8 <= r['token_count'] <= 15 for r in sample)))
            diagnostics.append(dict(component=name, populations=metrics))
            for column, metric in enumerate(('valid', 'intermediate')):
                ax = axes[1, column]
                for population, item in enumerate(metrics):
                    if item['attempts']:
                        ax.scatter(point+(population-1.5)*.012,
                                   100*item[metric]/item['attempts'], color=color, s=17, alpha=.5)
                if group:
                    ratio = 100*sum(item[metric] for item in metrics)/len(group)
                    ax.plot([point-.055, point+.055], [ratio]*2, color=color, linewidth=2.5,
                            label=f'Width b = {width}' if alpha_index == 0 else None)
    for ax, title in zip(axes[1], ('Direct bridge draws: hard validity',
                                  'Direct bridge draws: 8–15 source tokens')):
        ax.set_title(title)
        ax.set_xticks(range(4), ['0 (source)', '1/3', '2/3', '1 (competing)'])
        ax.set_xlabel('Bridge-center position along pose interpolation')
        ax.set_ylabel('Percent of direct bridge Gaussian attempts')
        ax.set_ylim(-2, 102)
        ax.grid(axis='y', alpha=.2)
        ax.text(.02, .97, 'Dots: populations; bars: pooled proportions',
                transform=ax.transAxes, va='top', fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Can coupled translation–rotation proposals reach intermediate contacts?\n'
                 'Geometry only · 4 × 2,048 attempts per arm · 50% uniform coverage · no depletion weights', fontsize=13)
    fig.savefig(args.out/'comparison.png', dpi=180)
    fig.savefig(args.out/'comparison.pdf')
    plt.close(fig)
    if sha(args.report) != args.expected_sha256 or sha(path) != report['contributions_sha256']:
        raise ValueError('Audited inputs changed during rendering')
    receipt = dict(complete=True, passed=True, source_sha256=sha(__file__),
        source_report_sha256=args.expected_sha256, contributions_sha256=sha(path),
        counts=counts_by_arm, direct_bridge_diagnostics=diagnostics,
        outputs={name: sha(args.out/name) for name in ('comparison.png', 'comparison.pdf')},
        new_geometry_queries=0, new_clouds=0, new_poses=0,
        scope='One mobile tetramer in a historical 500 μM frozen environment. '
              'Source tokens are not native registry. Raw counts measure proposal coverage, '
              'not equilibrium contact weights, path connectivity, or assembly stability.')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
