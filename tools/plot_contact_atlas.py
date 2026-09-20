#!/usr/bin/env python3
"""Plot separately estimated intermediate weights and a matched local calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plot_point(ax, x, row, color):
    y, relative_se = row['logQ'], row['row_RSE']
    assert math.isfinite(y) and 0 <= relative_se < 1
    # The uncertainty is on the linear mean Q, not an additive error on log Q.
    errors = np.array([[-math.log1p(-relative_se)], [math.log1p(relative_se)]])
    ax.errorbar(x, y, yerr=errors, fmt='o', markersize=8, color=color,
                capsize=5, elinewidth=1.7, zorder=3)
    return y+errors[1, 0]


def plot(analysis_path, calibration_path, out):
    assert not out.exists(), 'Use a fresh figure directory'
    analysis, historical = read(analysis_path), read(calibration_path)
    assert analysis['complete'] and historical['complete']
    q_window = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
    assert analysis['original_q_window'] == q_window
    assert analysis['peak_ball_radii_A'] == [.5, 1., 2.]
    campaigns = {campaign['arm']: campaign for campaign in analysis['campaigns']}
    assert set(campaigns) == {'narrow', 'broad'}
    protocol_paths = [Path(path) for path, digest in analysis['input_sha256'].items()
                      if digest == analysis['protocol_sha256']]
    assert len(protocol_paths) == 1
    protocol_path = protocol_paths[0]
    assert sha(protocol_path) == analysis['protocol_sha256']
    protocol = read(protocol_path)
    assert protocol['q_window'] == q_window
    physical = protocol['physical']
    assert physical['depletant_radius'] == 1.5 and physical['reservoir_density'] == .035
    assert physical['capture_radius'] == 18. and len(physical['fixed_poses']) == 2
    arms = {arm['name']: arm for arm in protocol['arms']}
    assert {name: arm['geometric_latent_SD_A'] for name, arm in arms.items()} == {'narrow': .2, 'broad': .4}
    source_analysis = Path(historical['source'])
    assert sha(source_analysis) == historical['source_sha256']
    inputs = {str(path.resolve()): sha(path) for path in
              (analysis_path, calibration_path, protocol_path, source_analysis)}
    archive_paths = {'input-atlas-analysis.json': analysis_path,
                     'input-local-calibration.json': calibration_path,
                     'input-atlas-protocol.json': protocol_path,
                     'input-local-source-analysis.json': source_analysis}
    for i, piece in enumerate(historical['pieces']):
        region_path = Path(piece['source'])/'provenance/region.json'
        region = read(region_path)
        assert (region['minimum_original_q'], region['maximum_original_q'],
                region['minimum_original_q_inclusive'], region['maximum_original_q_inclusive']) == (2., 5., True, False)
        assert region['physical_metric'] == physical['metadata']
        assert region['physical_fixed_neighbors'] == physical['fixed_poses']
        assert region['capture_center'] == physical['capture_center']
        assert region['capture_radius'] == physical['capture_radius']
        assert region['depletant_radius'] == 1.5 and region['activity'] == .035
        assert region['mahalanobis_radius'] == piece['maximum']
        inputs[str(region_path.resolve())] = sha(region_path)
        archive_paths[f'input-local-region-{i}.json'] = region_path
    reference = next(total for total in historical['totals'] if total['radius_A'] == 2.)
    assert reference['physical']['all_pieces_observed']
    colors = {'historical': '#69717a', 'narrow': '#167a55', 'broad': '#c57a25'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 6.0),
                             gridspec_kw={'width_ratios': [1, 1.2]})
    names = ('narrow', 'broad')
    for x, name in enumerate(names):
        campaign = campaigns[name]
        assert campaign['root'] == arms[name]['output']
        assert campaign['physical']['full']['draws'] == 65536
        upper = plot_point(axes[0], x, campaign['physical']['full'], colors[name])
        row = campaign['physical']['full']
        axes[0].annotate(f"{row['row_RSE']:.1%} relative SE\nLargest draw: {row['maximum_fraction']:.1%}",
                         (x, upper), xytext=(0, 11), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9, color=colors[name])
    axes[0].set_title('Full intermediate region: 2 ≤ q < 5', fontsize=12, pad=15)
    axes[0].set_xticks([0, 1], ['Narrow atlas\nσ = 0.2 Å', 'Broad atlas\nσ = 0.4 Å'])
    axes[0].set_xlim(-.55, 1.55)
    axes[0].set_ylim(14.55, 19.7)
    axes[0].text(.04, .965, '4 × 16,384 fresh draws per atlas', transform=axes[0].transAxes,
                 va='top', fontsize=9, color='#444444')

    local_rows = [reference['physical']]+[campaigns[name]['physical']['peak_r_le_2'] for name in names]
    cpu = [reference['CPU_seconds']]+[campaigns[name]['CPU_seconds'] for name in names]
    for x, (row, color) in enumerate(zip(local_rows, [colors['historical']]+[colors[name] for name in names])):
        plot_point(axes[1], x, row, color)
        axes[1].text(x, 13.85, f"{row['row_RSE']:.1%} relative SE", ha='center', va='bottom',
                     fontsize=9, color=color)
    axes[1].set_title('Same local neighborhood: ρ ≤ 2 Å-equivalent', fontsize=12, pad=15)
    axes[1].set_xticks([0, 1, 2], [f'Historical local\nreference\n{cpu[0]:.0f} CPU s',
                                  f'Narrow atlas\nσ = 0.2 Å\n{cpu[1]:.0f} CPU s',
                                  f'Broad atlas\nσ = 0.4 Å\n{cpu[2]:.0f} CPU s'])
    axes[1].set_xlim(-.5, 2.5)
    axes[1].set_ylim(13.24, 13.98)
    axes[1].text(.5, .05, 'Matched physical region; estimates are not pooled',
                 transform=axes[1].transAxes, ha='center', fontsize=9, color='#444444')
    for ax in axes:
        ax.set_ylabel(r'$\log Q$')
        ax.grid(axis='y', alpha=.19)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Contact atlas: local calibration agrees; full-region weight remains unresolved',
                 fontsize=14, y=.98)
    fig.text(.5, .175, 'Fixed AB neighbors; depletant radius 1.5 Å, activity 0.035 Å⁻³; capture radius 18 Å; normalized Haar measure.',
             ha='center', fontsize=9)
    fig.text(.5, .127, 'Bars: Q ± one observed SE, transformed to log. Full proposal support does not establish convergence or bound unseen weight.',
             ha='center', fontsize=8.6)
    fig.text(.5, .079, 'Historical local samples supplied training poses: this comparison is calibration, not an untouched holdout.',
             ha='center', fontsize=8.6)
    fig.text(.5, .031, 'CPU labels count sampler time only; training and audits are excluded. These are integration estimates, not MCMC mixing measurements.',
             ha='center', fontsize=8.6)
    fig.tight_layout(rect=(0, .235, 1, .945), w_pad=3)
    out.mkdir(parents=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'contact-atlas.{extension}', dpi=180, bbox_inches='tight')
    for name, path in archive_paths.items():
        shutil.copy2(path, out/name)
    shutil.copy2(__file__, out/'plotter.py')
    reference_log_cost = reference['physical']['log_variance_of_mean']+math.log(reference['CPU_seconds'])
    local_efficiency = {name: math.exp(reference_log_cost-
                                     campaigns[name]['physical']['peak_r_le_2']['log_variance_of_mean']-
                                     math.log(campaigns[name]['CPU_seconds'])) for name in names}
    provenance = dict(inputs=inputs, plotter_sha256=sha(Path(__file__)),
                      outputs_sha256={f'contact-atlas.{ext}':sha(out/f'contact-atlas.{ext}') for ext in ('png', 'svg')},
                      archived_sha256={name:sha(out/name) for name in archive_paths},
                      error_bars='Q plus or minus one observed row SE mapped to log; not tail confidence bounds.',
                      comparisons='Separate original-density estimates; no historical or cross-arm pooling. Historical local samples informed atlas training. Full support does not prove convergence.',
                      local_variance_times_CPU_reduction=local_efficiency,
                      efficiency_scope='Observed absolute variance of Q times sampler CPU for the matched radius-two local region only. Excludes training and audits; does not establish whole-target or MCMC mixing speedup. Ratios are diagnostic, not plotted.',
                      local_plotted_rows=local_rows, CPU_seconds=cpu,
                      full_plotted_rows={name:campaigns[name]['physical']['full'] for name in names})
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    plt.close(fig)
    print(out/'contact-atlas.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis.resolve(), args.calibration.resolve(), args.out.resolve())
