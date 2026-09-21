#!/usr/bin/env python3
"""Whole-shoulder width controls, matched finite masks and nested prefixes."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_expanded_contact_atlas import point, limits
from prepare_cayley_rms_cover import read, write, sha, require

COLORS = {'narrow': '#176a83', 'broad': '#c07125'}


def plot(path, out):
    require(not out.exists(), 'Fresh plot directory required')
    result = read(path)
    require(result['complete'], 'Completed audit required')
    for name, digest in result['archived_sha256'].items():
        require(sha(path.parent/'provenance'/name) == digest, 'Executed source changed')
    campaigns = {c['arm']: c for c in result['campaigns']}
    require(list(campaigns) == ['narrow', 'broad'], 'Fixed width arms required')
    reference = result['finite_reference']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.8))
    ax = axes[0, 0]
    keys = ['full', 'inner', 'outer']
    rows = [c['physical'][k] for c in campaigns.values() for k in keys]
    ax.set_ylim(*limits(rows))
    for name, offset in [('narrow', -.08), ('broad', .08)]:
        for i, key in enumerate(keys):
            point(ax, i+offset, campaigns[name]['physical'][key], COLORS[name], ax.get_ylim()[0])
    ax.set_xticks(range(3), ['Full shoulder', 'Inner: 1 < q < 1.1', 'Outer: 1.1 ≤ q < 2'])
    ax.set_title('Same physical shoulder, different fitted widths')

    ax = axes[0, 1]
    keys = ['inner_union', 'inner_outside_union']
    union = reference['independent_union_sum']['physical']
    rows = [c['physical'][k] for c in campaigns.values() for k in keys]+[union]
    ax.set_ylim(*limits(rows))
    for name, offset in [('narrow', -.13), ('broad', 0.)]:
        for i, key in enumerate(keys):
            point(ax, i+offset, campaigns[name]['physical'][key], COLORS[name], ax.get_ylim()[0])
    point(ax, .13, union, '#6c5a96', ax.get_ylim()[0], marker='D')
    ax.set_xticks(range(2), ['Three-ball union', 'Remaining inner shoulder'])
    ax.set_title('Fresh full-support estimates of both parts')
    ax.text(.5, .04, 'Reference measures only the finite union.', transform=ax.transAxes,
            ha='center', fontsize=8.5, color='#606a76')

    ax = axes[1, 0]
    centers = ['direct', 'mixture', 'geometry']
    keys = [f'inner_{c}_assigned_ball0p5' for c in centers]
    refs = reference['independent_assigned_region_sums']['physical']
    rows = [c['physical'][k] for c in campaigns.values() for k in keys]+[refs[c] for c in centers]
    ax.set_ylim(*limits(rows))
    for name, offset in [('narrow', -.13), ('broad', 0.)]:
        for i, key in enumerate(keys):
            point(ax, i+offset, campaigns[name]['physical'][key], COLORS[name], ax.get_ylim()[0])
    for i, center in enumerate(centers):
        point(ax, i+.13, refs[center], '#6c5a96', ax.get_ylim()[0], marker='D')
    ax.set_xticks(range(3), ['Direct contact', 'Mixture contact', 'Geometry contact'])
    ax.set_title('Matching disjoint inner regions')
    ax.text(.5, .04, 'Priority assignment; radius 0.5 Å. No overlap double counting.',
            transform=ax.transAxes, ha='center', fontsize=8.2, color='#606a76')

    ax = axes[1, 1]
    rows = [p['physical']['full'] for c in campaigns.values() for p in c['prefixes']]
    ax.set_ylim(*limits(rows))
    for name, offset in [('narrow', -.05), ('broad', .05)]:
        ps = campaigns[name]['prefixes']
        for i, prefix in enumerate(ps):
            point(ax, i+offset, prefix['physical']['full'], COLORS[name], ax.get_ylim()[0])
        ax.plot([i+offset for i in range(len(ps))],
                [p['physical']['full']['logQ'] for p in ps], color=COLORS[name], alpha=.4, lw=1)
    ax.set_xticks(range(3), [f"{16*n:,}" for n in result['prefix_counts']])
    ax.set_xlabel('Unconditional draws across 16 populations')
    ax.set_title('Nested prefixes: full shoulder')
    ax.text(.5, .04, 'Prefixes share rows; comparison errors retain covariance.',
            transform=ax.transAxes, ha='center', fontsize=8.2, color='#606a76')

    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$')
    legend = [Line2D([], [], marker='o', lw=0, color=COLORS[n], label=label)
              for n, label in [('narrow', 'Fitted widths'), ('broad', 'Doubled fitted widths')]]
    legend.append(Line2D([], [], marker='D', lw=0, color='#6c5a96', label='Independent finite reference'))
    fig.suptitle('Calibrated contacts in a complete shoulder proposal', fontsize=15, y=.99)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.text(.5, .075, 'AB fixed • radius 1.5 Å • activity 0.035 Å⁻³ • capture 18 Å • original 1 < q < 2', ha='center', fontsize=9)
    fig.text(.5, .040, 'Bars: log(Q ± one observed row SE). Full N and full hybrid density retained. Width arms are never added.', ha='center', fontsize=8.5)
    fig.text(.5, .012, 'Finite references informed the guide; their agreement is calibration. These are importance weights, not MCMC mixing measurements.', ha='center', fontsize=8.1)
    fig.tight_layout(rect=(0, .10, 1, .905), h_pad=3., w_pad=2.7)
    out.mkdir(parents=True)
    for ext in ['png', 'svg']:
        fig.savefig(out/f'shoulder-contact-atlas.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    sources = {'analysis.json': path, 'plotter.py': Path(__file__),
               'plot_helpers.py': Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    for name, source in sources.items(): shutil.copy2(source, out/name)
    write(out/'provenance.json', dict(input_sha256={str(p): sha(p) for p in sources.values()},
        archived_sha256={name: sha(out/name) for name in sources},
        output_sha256={f'shoulder-contact-atlas.{ext}': sha(out/f'shoulder-contact-atlas.{ext}') for ext in ['png', 'svg']},
        scope=result['scope']))
    print(out/'shoulder-contact-atlas.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis.resolve(), args.out.resolve())
