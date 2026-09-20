#!/usr/bin/env python3
"""Compare new fixed-region references with historical atlas masks.

Uniform-ball and multiscale estimates share samples and are alternative
estimators, not independent confirmations. Historical masks reuse campaigns
from which the new center was selected. No samples or nested means are pooled.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from prepare_cayley_rms_cover import read, require, sha, write

RADII = (.5, 1., 2.)
WINDOW = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
SERIES = {
    'reference': dict(color='#2369a1', marker='o', filled=True, offset=-.27,
                      label='Fresh uniform ball'),
    'multiscale': dict(color='#14836b', marker='D', filled=True, offset=-.09,
                       label='Fresh disjoint-shell sum'),
    'narrow': dict(color='#bf831a', marker='^', filled=False, offset=.09,
                   label='Historical narrow atlas'),
    'broad': dict(color='#b55269', marker='v', filled=False, offset=.27,
                  label='Historical broad atlas'),
}


def plot_mass(ax, x, row, style):
    y, rse = row['logQ'], row['row_RSE']
    if y is None:
        require(rse is None and row['nonzero'] == 0, 'Inconsistent unobserved mask')
        return None
    require(math.isfinite(y) and rse is not None and math.isfinite(rse) and rse >= 0,
            'Invalid estimate or observed relative SE')
    lower_zero = rse >= 1.-1e-12
    lower = 1.25 if lower_zero else -math.log1p(-rse)
    upper = math.log1p(rse)
    ax.errorbar(x, y, yerr=np.array([[lower], [upper]]), fmt=style['marker'],
                markersize=7, capsize=3.5, color=style['color'], linewidth=1.2,
                markerfacecolor=style['color'] if style['filled'] else 'white',
                markeredgewidth=1.2, zorder=3)
    if lower_zero:
        ax.annotate('', (x, y-lower-.3), (x, y-lower+.06),
                    arrowprops=dict(arrowstyle='->', color=style['color'], lw=1.1))
    return y-lower-(.3 if lower_zero else 0), y+upper


def plot(analysis_path, multiscale_path, historical_path, out):
    analysis_path, multiscale_path, historical_path, out = [Path(p).resolve()
        for p in (analysis_path, multiscale_path, historical_path, out)]
    require(not out.exists(), 'Use a fresh figure output directory')
    audit, multiscale, historical = map(read, (analysis_path, multiscale_path, historical_path))
    require(audit['complete'] and multiscale['complete'] and historical['complete'], 'Need completed analyses')
    require(audit['radial_edges_A'] == [0., *RADII], 'Fresh radial masks changed')
    require(historical['radii_A'] == list(RADII) and historical['original_q_window'] == WINDOW,
            'Historical region definitions differ')
    require(Path(multiscale['source']).resolve() == analysis_path and multiscale['source_sha256'] == sha(analysis_path),
            'Multiscale sum is not derived from this exact fresh audit')
    fresh = {c['radius_A']: c for c in audit['campaigns']}
    summed = {c['radius_A']: c for c in multiscale['totals']}
    old = {c['arm']: c for c in historical['campaigns']}
    require(set(fresh) == set(RADII) and set(summed) == {1., 2.} and set(old) == {'narrow', 'broad'},
            'Missing or duplicated comparison arms')
    new_model_path = historical_path.parent/'provenance/new-model.json'
    new_model = read(new_model_path)
    require(sha(new_model_path) == historical['archived_sha256']['new-model.json'], 'Historical frozen geometry chart changed')
    source_paths = {'input-reference-analysis.json': analysis_path,
        'input-multiscale-analysis.json': multiscale_path, 'input-historical-analysis.json': historical_path,
        'input-new-model.json': new_model_path}
    for i, radius in enumerate(RADII):
        region_path = Path(fresh[radius]['root'])/'provenance/region.json'; region = read(region_path)
        require(region['gaussian_chart'] == new_model and region['mahalanobis_radius'] == radius,
                'Fresh and historical neighborhoods differ')
        require((region['minimum_original_q'], region['maximum_original_q'],
                 region['minimum_original_q_inclusive'], region['maximum_original_q_inclusive']) == (2., 5., True, False),
                'Original q window changed')
        require(region['capture_radius'] == 18. and region['activity'] == .035 and region['depletant_radius'] == 1.5,
                'Original physical baseline changed')
        require(len(region['physical_fixed_neighbors']) == 2 and region['minimum_mahalanobis_radius'] == 0.,
                'Physical AB neighbors or complete local ball changed')
        source_paths[f'input-region-{i}.json'] = region_path
    input_hashes = {str(path): sha(path) for path in source_paths.values()}
    rows = {'reference': {r: fresh[r]['physical']['full'] for r in RADII},
            'multiscale': {r: summed[r]['physical'] for r in (1., 2.)}}
    for arm in ('narrow', 'broad'):
        rows[arm] = {region['radius_A']: region['physical']['ball'] for region in old[arm]['regions']}
        require(set(rows[arm]) == set(RADII), 'Missing historical radius')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 12})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.1), gridspec_kw={'width_ratios': [1.55, 1.]})
    mass_ax, error_ax = axes
    limits = []; errors = []
    for name, series in rows.items():
        style = SERIES[name]
        for radius, row in series.items():
            x = RADII.index(radius)+style['offset']
            extent = plot_mass(mass_ax, x, row, style)
            if extent is not None:
                limits.extend(extent); errors.append(100*row['row_RSE'])
                error_ax.plot(x, max(100*row['row_RSE'], 1e-5), marker=style['marker'],
                              color=style['color'], markersize=7,
                              markerfacecolor=style['color'] if style['filled'] else 'white',
                              markeredgewidth=1.2, linestyle='none')
    lower, upper = min(limits), max(limits)
    pad = max(.6, .08*(upper-lower)); mass_ax.set_ylim(lower-pad, upper+2*pad)
    for name, series in rows.items():
        for radius, row in series.items():
            if row['logQ'] is None:
                x = RADII.index(radius)+SERIES[name]['offset']
                mass_ax.text(x, lower-.7*pad, 'No hits', color=SERIES[name]['color'], fontsize=8,
                             ha='center', va='bottom', rotation=90)
                error_ax.text(x, .03, 'No hits', transform=error_ax.get_xaxis_transform(),
                              color=SERIES[name]['color'], fontsize=8, ha='center', va='bottom', rotation=90)
    selected = old['broad']['selected_point']
    require(selected is not None and selected['radius_A'] < 1e-8, 'Selected broad maximum not centered')
    require(rows['broad'][.5]['nonzero'] == 1, 'Annotation requires one historical broad hit in smallest ball')
    mass_ax.annotate('Selected pose:\nonly historical broad hit',
        xy=(SERIES['broad']['offset'], selected['log_contribution_to_full_N_mean']),
        xytext=(.38, .98), textcoords='axes fraction', fontsize=8.5,
        color=SERIES['broad']['color'], va='top', ha='left',
        arrowprops=dict(arrowstyle='->', color=SERIES['broad']['color'], lw=.9,
                        connectionstyle='arc3,rad=.12'))
    mass_ax.set_title('(a) Integrated neighborhood weight', loc='left', pad=12)
    mass_ax.set_ylabel(r'$\log Q$')
    error_ax.set_title('(b) Observed relative standard error', loc='left', pad=12)
    error_ax.set_ylabel('Observed SE / estimated Q (%)')
    error_ax.set_yscale('log')
    positive_errors = [v for v in errors if v > 0]
    error_ax.set_ylim(max(.05, min(positive_errors)/1.8), max(140., max(errors)*1.4))
    for ax in axes:
        ax.set_xticks(range(3), ['0.5', '1', '2'])
        ax.set_xlim(-.5, 2.5)
        ax.set_xlabel('Geometry-chart radius R (Å-equivalent)')
        ax.grid(axis='y', alpha=.2, which='major')
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], color=style['color'], marker=style['marker'], linestyle='none',
                      markerfacecolor=style['color'] if style['filled'] else 'white', markersize=7,
                      label=style['label']) for style in SERIES.values()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .205), ncol=2,
               frameon=False, fontsize=9.5, columnspacing=2.4)
    fig.suptitle('Statistical weight around a newly discovered competing contact', fontsize=14, y=.98)
    fig.text(.5, .16, 'Fixed AB; original 2 ≤ q < 5; depletant radius 1.5 Å, activity 0.035 Å⁻³.',
             ha='center', fontsize=9)
    fig.text(.5, .115, 'Center selected from the broad atlas: historical markers are retrospective, not held-out validation.',
             ha='center', fontsize=8.7)
    fig.text(.5, .072, 'Shell sums use independent disjoint bands; they share data with the ball estimates. At R = 0.5 they coincide.',
             ha='center', fontsize=8.5)
    fig.text(.5, .03, 'Bars: Q ± observed SE transformed to log; downward arrow: zero lower endpoint. Unseen weight remains unbounded.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .30, 1, .94))
    out.mkdir(parents=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'atlas-peak-neighborhood.{extension}', dpi=190, bbox_inches='tight')
    for name, path in source_paths.items(): shutil.copy2(path, out/name)
    shutil.copy2(__file__, out/'plotter.py')
    for path, digest in input_hashes.items(): require(sha(Path(path)) == digest, 'Input changed during plotting')
    write(out/'provenance.json', dict(input_sha256=input_hashes, plotter_sha256=sha(Path(__file__)),
        figure_sha256={suffix: sha(out/f'atlas-peak-neighborhood.{suffix}') for suffix in ('png', 'svg')},
        plotted_estimates=rows,
        error_bars='One observed linear-mean standard error transformed to log, not a log standard error or unseen-tail confidence bound.',
        qualification='Historical center selection is data dependent. Fresh disjoint-shell and full-ball estimates share samples; no pooling or summing of nested full-ball estimates.',
        scope='Only the frozen local chart balls, intersected with unchanged q, capture and AB masks. Does not estimate the whole intermediate region or imply an assembly conclusion.'))
    plt.close(fig)
    print(out/'atlas-peak-neighborhood.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--multiscale', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plot(args.analysis, args.multiscale, args.historical, args.out)
