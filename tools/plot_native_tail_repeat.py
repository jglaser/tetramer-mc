#!/usr/bin/env python3
"""Plot independent native-cover controls without replacing full-native weights."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_expanded_contact_atlas import point, limits
from prepare_cayley_rms_cover import read, write, sha, require


def plot(path, out):
    require(not out.exists(), 'Fresh figure directory required')
    result = read(path)
    require(result['complete'] and result['no_pooling'], 'Completed independent comparison required')
    for name, digest in result['archived_sha256'].items():
        require(sha(path.parent/'provenance'/name) == digest, 'Comparison archive changed')
    colors = {'pilot': '#8b94a2', 'repeat': '#136d88', 'reference': '#a45739'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), gridspec_kw={'width_ratios': [1.45, 1.]})
    ax = axes[0]
    keys = result['masks'][1:]
    refs = result['targeted_uniform']
    rows = [result[name]['physical'][key] for name in ['pilot', 'repeat'] for key in keys]
    rows += [r['physical'] for r in refs.values()]
    ax.set_ylim(*limits(rows))
    for name, offset in [('pilot', -.15), ('repeat', 0.)]:
        for i, key in enumerate(keys):
            point(ax, i+offset, result[name]['physical'][key], colors[name], ax.get_ylim()[0])
    for i, key in enumerate(keys):
        if key in refs:
            point(ax, i+.15, refs[key]['physical'], colors['reference'], ax.get_ylim()[0], marker='D')
    ax.set_xticks(range(len(keys)), ['r ≤ 4', '4 < r ≤ 5', '5 < r ≤ 8', '8 < r ≤ 12', 'r > 12'])
    ax.set_xlabel('Diagnostic radius in the original fitted chart; native q ≤ 1 throughout')
    ax.set_title('Independent controls of the same native pieces')
    ax.text(.5, .93, 'Core remains poorly sampled: only 6 repeat rows at r ≤ 4.\n'
            'Targeted references remain separate; pieces are not added across runs.',
            ha='center', va='top', transform=ax.transAxes, fontsize=8.4, color='#56606d')

    ax = axes[1]
    prefixes = result['repeat_prefixes']
    rows = [p['physical']['old_r_gt_12'] for p in prefixes]
    population_rows = [dict(r, row_RSE=r['independent_population_RSE']) for r in rows]
    ax.set_ylim(*limits(rows+population_rows))
    for i, (row, population) in enumerate(zip(rows, population_rows)):
        point(ax, i-.06, row, colors['repeat'], ax.get_ylim()[0])
        point(ax, i+.06, population, '#82adb7', ax.get_ylim()[0], marker='s', markersize=4)
    ax.plot([i-.06 for i in range(len(rows))], [r['logQ'] for r in rows],
            color=colors['repeat'], lw=1, alpha=.35)
    ax.set_xticks(range(len(rows)), [f"{r['draws']:,}" for r in rows])
    ax.set_xlabel('Original unconditional draws; fixed 4 / 8 / 16 populations')
    ax.set_title('Remote tail: r > 12, nested repeat prefixes')
    final = rows[-1]
    ax.text(.5, .06, f"Final: row RSE {final['row_RSE']:.1%}; population RSE "
            f"{final['independent_population_RSE']:.1%}\n"
            f"Weight ESS {final['weight_ESS']:.1f}; largest row {final['maximum_fraction']:.1%}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
    ax.legend(handles=[Line2D([], [], color=colors['repeat'], marker='o', lw=1, label='Row SE'),
                       Line2D([], [], color='#82adb7', marker='s', lw=1, label='Population SE')],
              loc='upper left', frameon=False, fontsize=8.5)
    for ax in axes:
        ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$')
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Native tail coverage improves; concentrated weights remain', y=.99, fontsize=15)
    fig.legend(handles=[Line2D([], [], marker=marker, lw=0, color=colors[key], label=label)
        for key, marker, label in [('pilot','o','Pilot: 131,072 draws'),
                                  ('repeat','o','Repeat: 4,194,304 draws'),
                                  ('reference','D','Targeted finite-shell reference')]],
        loc='upper center', bbox_to_anchor=(.5,.945), frameon=False, ncol=3)
    fig.text(.5,.084,'AB fixed • depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture 18 Å • inclusive native 0 ≤ q ≤ 1',
             ha='center',fontsize=9)
    fig.text(.5,.045,'Bars transform Q ± one observed SE. Prefixes share draws; they are not independent replications.',
             ha='center',fontsize=8.5)
    fig.text(.5,.013,'Observed errors do not bound unseen weight. This cover does not replace the fitted full-native estimate; no MCMC mixing claim.',
             ha='center',fontsize=8.3)
    fig.tight_layout(rect=(0,.12,1,.88),w_pad=2.5)
    out.mkdir(parents=True)
    for ext in ['png','svg']:
        fig.savefig(out/f'native-tail-repeat.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    sources={'analysis.json':path,'plotter.py':Path(__file__),
             'plot_helpers.py':Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    for name,source in sources.items():shutil.copy2(source,out/name)
    write(out/'provenance.json',dict(input_sha256={str(p):sha(p) for p in sources.values()},
        archived_sha256={name:sha(out/name) for name in sources},
        output_sha256={f'native-tail-repeat.{ext}':sha(out/f'native-tail-repeat.{ext}') for ext in ['png','svg']},
        scope=result['scope']))
    print(out/'native-tail-repeat.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plot(args.analysis.resolve(),args.out.resolve())
