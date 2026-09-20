#!/usr/bin/env python3
"""Plot local reference masses and matched regions, without pooling estimates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_estimate(ax, x, row, color, unresolved_length=1.15):
    y, se = row['logQ'], row['row_RSE']
    assert y is not None and math.isfinite(y) and se is not None and 0 <= se <= 1+1e-10
    unbounded = se >= 1-1e-12
    lower = unresolved_length if unbounded else -math.log1p(-se)
    upper = math.log1p(se)
    ax.errorbar(x, y, yerr=np.array([[lower], [upper]]), fmt='o', color=color,
                capsize=4, markersize=7, markerfacecolor='white' if unbounded else color,
                zorder=3)
    if unbounded:
        ax.annotate('', xy=(x, y-lower-.28), xytext=(x, y-lower+.03),
                    arrowprops=dict(arrowstyle='->', color=color))
    return y, upper


def plot(path, out):
    assert not out.exists(), 'Use a fresh output directory'
    data = json.loads(path.read_text())
    assert data['complete'] is True and data['radial_edges_A'] == [0., .5, 1., 2.]
    campaigns = sorted(data['campaigns'], key=lambda c: c['radius_A'])
    assert [c['radius_A'] for c in campaigns] == [.5, 1., 2.]
    inputs = {str(path.resolve()): sha(path)}
    regions = []
    for campaign in campaigns:
        region_path = Path(campaign['root'])/'provenance/region.json'
        region = json.loads(region_path.read_text())
        assert (region['minimum_original_q'], region['maximum_original_q'],
                region['minimum_original_q_inclusive'], region['maximum_original_q_inclusive']) == (2., 5., True, False)
        assert region['depletant_radius'] == 1.5 and region['activity'] == .035
        assert region['capture_radius'] == 18.
        assert region['mahalanobis_radius'] == campaign['radius_A']
        assert region.get('minimum_mahalanobis_radius', 0.) == 0.
        if regions:
            for field in ('gaussian_chart', 'physical_fixed_neighbors', 'physical_metric', 'capture_center', 'shape_sha256'):
                assert region[field] == regions[0][1][field]
        inputs[str(region_path.resolve())] = sha(region_path)
        regions.append((region_path, region))

    colors = ['#167a55', '#3677b5', '#b86b26']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={'width_ratios': [1, 1.12]})
    for i, (campaign, color) in enumerate(zip(campaigns, colors)):
        row = campaign['physical']['full']
        y, upper = draw_estimate(axes[0], i, row, color)
        population_y = [p['physical']['full']['logQ'] for p in campaign['populations']]
        assert len(population_y) == 4 and all(v is not None for v in population_y)
        axes[0].scatter(i+np.array([-.16, -.09, .09, .16]), population_y,
                        s=18, facecolor='white', edgecolor=color, linewidth=.9, zorder=2)
        axes[0].annotate(f"{row['row_RSE']:.1%} SE", (i, max(y+upper, *population_y)),
                         xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=color)
    axes[0].set_title('Each complete local ball', fontsize=12)
    axes[0].set_xticks(range(3), ['0.5', '1', '2'])
    axes[0].set_xlabel('Local chart radius R (Å-equivalent)', fontsize=10)
    axes[0].set_xlim(-.4, 2.4)
    axes[0].set_ylim(11.7, 14.02)
    axes[0].text(.02, .97, 'Small open dots: four independent populations',
                 transform=axes[0].transAxes, ha='left', va='top', fontsize=8.5, color='#444444')

    for i, (campaign, color) in enumerate(zip(campaigns, colors)):
        for group, key in enumerate(('radial_0', 'radial_1')):
            if key not in campaign['physical']:
                continue  # Region lies outside this proposal's support.
            x = group+(i-1)*.20
            row = campaign['physical'][key]
            y, upper = draw_estimate(axes[1], x, row, color)
            if row['nonzero'] == 1:
                axes[1].annotate('1 nonzero draw', (x, y), xytext=(8, -5),
                                 textcoords='offset points', fontsize=8, color=color)
    axes[1].set_title('Same regions, different proposal sizes', fontsize=12)
    axes[1].set_xticks([0, 1], [r'$0\leq\rho\leq0.5$', r'$0.5<\rho\leq1$'])
    axes[1].set_xlabel('Fixed subregion in the same local chart (Å-equivalent)', fontsize=10)
    axes[1].set_xlim(-.45, 1.45)
    axes[1].set_ylim(8.95, 13.5)
    for ax in axes:
        ax.set_ylabel(r'$\log Q$')
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)

    handles = [Line2D([], [], color=c, marker='o', linestyle='none', label=f'Uniform source ball R = {r:g}')
               for c, r in zip(colors, [.5, 1., 2.])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .15), ncol=3, frameon=False)
    fig.suptitle('Independent integration around the newly found contact pose', fontsize=14, y=.98)
    fig.text(.5, .105, 'Fixed AB neighbors; original 2 ≤ q < 5; depletant radius 1.5 Å, activity 0.035 Å⁻³.',
             ha='center', fontsize=9)
    fig.text(.5, .065, 'Local regions only. Nested estimates are not added; matched-region estimates retain every invalid zero.',
             ha='center', fontsize=8.5)
    fig.text(.5, .025, 'Bars: Q ± observed SE, transformed to log. Downward arrow: lower endpoint is zero. No bound on unseen weight.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .225, 1, .94))
    out.mkdir(parents=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'peak-neighborhood.{extension}', dpi=180, bbox_inches='tight')
    shutil.copy2(path, out/'input-analysis.json')
    for i, (region_path, _) in enumerate(regions):
        shutil.copy2(region_path, out/f'input-region-{i}.json')
    shutil.copy2(__file__, out/'plotter.py')
    provenance = {'inputs': inputs, 'plotter_sha256': sha(Path(__file__)),
                  'scope': 'Separate fixed local and matched-subregion estimates. No pooling, global normalizer, unseen-tail confidence bounds or mixing claims.',
                  'error_bars': 'Q plus or minus one observed row SE, transformed to log. Zero lower endpoint shown with a downward arrow.',
                  'population_points': 'Four independent populations within each source-ball campaign; not independent of their pooled estimate.'}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    plt.close(fig)
    print(out/'peak-neighborhood.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis.resolve(), args.out.resolve())
