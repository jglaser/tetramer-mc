#!/usr/bin/env python3
"""Compare fixed-model independent campaigns and correlated sample prefixes."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_expanded_contact_atlas import limits, point, read, sha


def verify_archive(path):
    data = read(path)
    assert data['complete'], 'Analysis must be terminal'
    for name, digest in data['archived_sha256'].items():
        assert sha(path.parent/'provenance'/name) == digest, 'Archive changed'
    for name, digest in data['input_sha256'].items():
        assert sha(Path(name)) == digest, 'Analysis input changed'
    return data


def plot(comparison_path, out):
    assert not out.exists(), 'Use a fresh figure directory'
    comparison = verify_archive(comparison_path)
    sources = {}
    for label in ('original', 'repeat'):
        # Comparison archives preserve the audited summaries used for plotting.
        sources[label] = read(comparison_path.parent/'provenance'/f'{label}-audit.json')
        assert sources[label]['complete']
        assert sources[label]['original_q_window'] == dict(
            minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
    physical = comparison['unchanged_physical_signature']
    assert physical['depletant_radius'] == 1.5 and physical['reservoir_density'] == .035
    assert physical['capture_radius'] == 18. and len(physical['fixed_poses']) == 2
    arms = {label: {c['arm']: c for c in source['campaigns']} for label, source in sources.items()}
    assert all(set(c) == {'narrow', 'broad'} for c in arms.values())
    compared = {a['arm']: a for a in comparison['arms']}
    for name in compared:
        for label in sources:
            assert compared[name]['physical']['full'][label] == arms[label][name]['physical']['full']
            assert arms[label][name]['disjoint_threeway']['partition_checks']['physical']['complete_row_partition']
    prefix_rows = {name: [p['physical']['full'] for p in c['prefixes']]
                   for name, c in arms['repeat'].items()}
    budgets = [r['draws'] for r in prefix_rows['narrow']]
    assert len(budgets) >= 2 and budgets == sorted(set(budgets))
    assert budgets == [r['draws'] for r in prefix_rows['broad']]
    original_n = arms['original']['narrow']['physical']['full']['draws']
    assert original_n == arms['original']['broad']['physical']['full']['draws'] == budgets[0]
    assert all(prefix_rows[name][-1] == arms['repeat'][name]['physical']['full'] for name in prefix_rows)
    parts = ('old_ball', 'new_ball', 'outside_both')
    plotted = dict(prefixes=prefix_rows,
        original_full={n: c['physical']['full'] for n, c in arms['original'].items()},
        disjoint={label: {n: {key: c['disjoint_threeway']['physical'][key] for key in parts}
                          for n, c in campaigns.items()} for label, campaigns in arms.items()})
    colors = {'narrow': '#14795c', 'broad': '#bc722a'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7), gridspec_kw={'width_ratios': [1.05, 1.2]})
    left = [r for rows in prefix_rows.values() for r in rows] + list(plotted['original_full'].values())
    right = [r for campaigns in plotted['disjoint'].values() for rows in campaigns.values() for r in rows.values()]
    for ax, rows in zip(axes, (left, right)):
        ax.set_ylim(*limits(rows)); ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylabel(r'$\log Q$')
    for name, offset in (('narrow', -.045), ('broad', .045)):
        x = np.arange(len(budgets)) + offset
        axes[0].plot(x, [r['logQ'] for r in prefix_rows[name]], color=colors[name], alpha=.6, lw=1.4)
        for xx, row in zip(x, prefix_rows[name]):
            point(axes[0], xx, row, colors[name], axes[0].get_ylim()[0])
        point(axes[0], -.22 + offset, plotted['original_full'][name], colors[name], axes[0].get_ylim()[0], marker='s')
    axes[0].set_xticks(range(len(budgets)), [f'{n:,}' for n in budgets])
    axes[0].set_xlim(-.55, len(budgets)-.6)
    axes[0].set_xlabel('Total unconditional draws per proposal arm')
    axes[0].set_title('Four times the sample count; unchanged proposals', fontsize=11.5, pad=25)
    axes[0].text(.5, 1.02, 'Squares: original. Connected circles: correlated repeat prefixes.',
                 transform=axes[0].transAxes, ha='center', fontsize=8.5, color='#4b5560')
    for name, base in (('narrow', -.19), ('broad', .19)):
        for label, offset, marker in (('original', -.075, 's'), ('repeat', .075, 'o')):
            for x, key in enumerate(parts):
                point(axes[1], x+base+offset, plotted['disjoint'][label][name][key],
                      colors[name], axes[1].get_ylim()[0], marker=marker, markersize=5.5)
    axes[1].set_xticks(range(3), ['Old contact\nρ ≤ 2 Å', 'New contact\nρ ≤ 2 Å', 'Outside both\nremaining poses'])
    axes[1].set_xlim(-.5, 2.5)
    axes[1].set_title('Where does the independently measured weight lie?', fontsize=11.5, pad=25)
    axes[1].text(.5, 1.02, 'Old + new + outside = the full 2 ≤ q < 5 window',
                 transform=axes[1].transAxes, ha='center', fontsize=8.5, color='#4b5560')
    handles = [Line2D([], [], color=colors[n], marker='o', lw=1.4, label=f'{n.capitalize()} geometry, σ = {s} Å')
               for n, s in (('narrow', '.2'), ('broad', '.4'))]
    handles += [Line2D([], [], color='#515964', marker=marker, lw=0, label=label)
                for marker, label in [('s', 'Original: 8 × 32,768'), ('o', 'Repeat: 16 × 65,536')]]
    fig.suptitle('Frozen contact atlas: independent larger-sample repeat', fontsize=14, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=4, frameon=False, fontsize=8.7)
    fig.text(.5, .163, 'Fixed AB neighbors • depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture radius 18 Å • normalized Haar measure',
             ha='center', fontsize=9)
    fig.text(.5, .114, 'Bars: log(Q ± one observed row SE). Separate campaigns have disjoint seeds; prefixes within a campaign share samples.',
             ha='center', fontsize=8.6)
    fig.text(.5, .068, 'Model bytes, integration window and neighborhood definitions are unchanged. No refitting or pooling of campaigns.',
             ha='center', fontsize=8.6)
    fig.text(.5, .024, 'Agreement and observed errors do not exclude unseen high-weight contacts. This measures statistical weight, not trajectory mixing.',
             ha='center', fontsize=8.6)
    fig.tight_layout(rect=(0., .215, 1., .86), w_pad=3.)
    out.mkdir(parents=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'contact-atlas-repeat.{ext}', dpi=180, bbox_inches='tight')
    files = {'input-comparison.json': comparison_path, 'plotter.py': Path(__file__),
             'plot_expanded_contact_atlas.py': Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    files.update({f'{label}-audit.json': comparison_path.parent/'provenance'/f'{label}-audit.json' for label in sources})
    for name, path in files.items(): shutil.copy2(path, out/name)
    provenance = dict(input_sha256={str(p): sha(p) for p in files.values()},
        archived_sha256={n: sha(out/n) for n in files},
        output_sha256={f'contact-atlas-repeat.{ext}': sha(out/f'contact-atlas-repeat.{ext}') for ext in ('png', 'svg')},
        plotted_rows=plotted, total_draw_prefixes=budgets,
        errors='One observed row SE on linear Q, displayed on log scale; not confidence bounds on unseen mass.',
        scope='Separate fixed-model importance campaigns. No pooling, model refitting, or dynamical mixing claim.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    plt.close(fig)
    print(out/'contact-atlas-repeat.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); plot(args.comparison.resolve(), args.out.resolve())
