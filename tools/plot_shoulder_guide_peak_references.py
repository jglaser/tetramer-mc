#!/usr/bin/env python3
"""Compare identical three-center masks and retain the unmeasured complement."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_expanded_contact_atlas import point, limits
from prepare_cayley_rms_cover import read, write, sha, require

CENTERS = ('direct', 'mixture', 'geometry')
HISTORY = ('direct-inner', 'mixture-confirmation', 'geometry-confirmation')


def plot(path, out):
    require(not out.exists(), 'Fresh figure output required')
    result = read(path)
    require(result['complete'], 'Completed independent audit required')
    require(result['original_q_window'] == dict(minimum=1., maximum=1.1,
            lower_inclusive=False, upper_inclusive=False), 'Original inner window differs')
    for name, digest in result['archived_sha256'].items():
        require(sha(path.parent/'provenance'/name) == digest, 'Executed archive changed')
    for name, digest in result['source_sha256'].items():
        require(sha(name) == digest, 'Analyzed source changed')
    histories = {h['name']: h for h in result['histories']}
    require(set(histories) == set(HISTORY), 'Historical sources differ')
    campaigns = {(c['owner'], c['radius_A']): c for c in result['campaigns']}
    direct_path = path.parent/'provenance/direct-analysis.json'
    require(sha(direct_path) == result['direct_reference']['sha256'], 'Direct reference changed')
    direct = {c['radius_A']: c for c in read(direct_path)['campaigns']}
    # Whole balls compare precisely the same target. Assigned shell sums are
    # shown separately below; estimates sharing rows are never added here.
    fresh_local = {
        radius: [direct[radius]['physical']['full']] +
        [campaigns[name, radius]['physical']['full'] for name in CENTERS[1:]]
        for radius in (.25, .5)
    }
    fresh_assigned = [result['independent_assigned_region_sums']['physical'][name] for name in CENTERS]
    fresh_union = result['independent_union_sum']['physical']
    labels = ('Direct peak', 'Mixture peak', 'Geometry peak')
    colors = dict(zip(HISTORY, ('#647080', '#16735b', '#bc7627')))
    blue = '#226bb1'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.8))
    axes = axes.ravel()
    masks = [[f'{n}_ball0p25' for n in CENTERS],
             [f'{n}_ball0p5' for n in CENTERS],
             [f'{n}_assigned_ball0p5' for n in CENTERS],
             ['union', 'outside_union']]
    references = [fresh_local[.25], fresh_local[.5], fresh_assigned, [fresh_union]]
    for ax, keys, refs in zip(axes, masks, references):
        rows = [histories[n]['physical'][key] for n in HISTORY for key in keys] + refs
        ax.set_ylim(*limits(rows))
        ax.set_xlim(-.42, len(keys)-.58)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$')
        for name, offset in zip(HISTORY, (-.21, -.07, .07)):
            for i, key in enumerate(keys):
                point(ax, i+offset, histories[name]['physical'][key], colors[name], ax.get_ylim()[0])
        for i, row in enumerate(refs):
            point(ax, i+.21, row, blue, ax.get_ylim()[0], marker='D', markersize=5.5)
    for ax in axes[:3]:
        ax.set_xticks(range(3), labels)
    axes[3].set_xticks(range(2), ('Union of three balls', 'Outside all three'))
    titles = ('Same radius-0.25 Å balls', 'Same radius-0.5 Å balls',
              'Disjoint assigned regions, radius 0.5 Å', 'Finite union and remaining inner shoulder')
    notes = ('Local whole-ball proposals; centers selected from historical data.',
             'Nested balls overlap: these are separate estimates.',
             'Priority: direct → mixture → geometry. Blue uses independent shells.',
             'No fresh remainder estimate: the missing blue point is not zero.')
    for ax, title, note in zip(axes, titles, notes):
        ax.set_title(title, fontsize=11.5, pad=26)
        ax.text(.5, 1.025, note, transform=ax.transAxes, ha='center', fontsize=8.1, color='#657181')
    legend = [Line2D([], [], marker='o', lw=0, color=colors[n], label=label)
              for n, label in zip(HISTORY, ('Historical direct cover', 'Historical mixture guide', 'Historical geometry guide'))]
    legend.append(Line2D([], [], marker='D', lw=0, color=blue, label='Independent geometric reference'))
    fig.suptitle('Calibrating three inner-shoulder contact neighborhoods', fontsize=15, y=.985)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .954), ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .105, 'Both AB neighbors fixed • depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture 18 Å • original 1 < q < 1.1', ha='center', fontsize=9)
    fig.text(.5, .067, 'Bars: log(Q ± one observed row SE). Every method retains its original full N and proposal density.', ha='center', fontsize=8.7)
    fig.text(.5, .028, 'Same-row covariances are retained in the audit. Local and historical remainder estimates are not stitched into a new whole normalizer.', ha='center', fontsize=8.3)
    fig.tight_layout(rect=(0, .135, 1, .899), h_pad=3.3, w_pad=2.6)
    out.mkdir(parents=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'shoulder-guide-peak-references.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    sources = {'analysis.json': path, 'plotter.py': Path(__file__),
               'plot_helpers.py': Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    for name, source in sources.items():
        shutil.copy2(source, out/name)
    write(out/'provenance.json', dict(input_sha256={str(p): sha(p) for p in sources.values()},
        archived_sha256={name: sha(out/name) for name in sources},
        output_sha256={f'shoulder-guide-peak-references.{ext}': sha(out/f'shoulder-guide-peak-references.{ext}') for ext in ('png', 'svg')},
        plotted=dict(fresh_local=fresh_local, fresh_assigned=fresh_assigned, fresh_union=fresh_union,
                     history={name: {key: histories[name]['physical'][key] for keys in masks for key in keys} for name in HISTORY}),
        scope='Data-selected finite regions, original full N and densities. Independent finite-union reference; no fresh complement or whole-window convergence claim.'))
    print(out/'shoulder-guide-peak-references.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis.resolve(), args.out.resolve())
