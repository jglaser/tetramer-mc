#!/usr/bin/env python3
"""Plot completed source-guide coverage; no new geometry or physical weights."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes(); report = json.loads(raw)
    assert report['schema'] == 'context-source-guide-independent-audit-v1'
    assert report['complete'] and report['passed'] and report['physical_weight_status'] == 'not_estimated'
    assert report['decoded_candidates'] == 8192 and report['independent_panel_size'] == 512
    assert not args.out_dir.exists(), 'Fresh figure output required'
    args.out_dir.mkdir(parents=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    colors = ['#294f89', '#087786', '#ba8735', '#a54f63']
    summaries = []
    for width, color in enumerate(colors):
        ps = sorted([p for p in report['populations'] if p['width_index'] == width], key=lambda p:p['stream'])
        assert len(ps) == 4 and all(p['denominator'] == 512 for p in ps)
        hits = [p['full_source217_inclusion_A_hits'] for p in ps]
        any_hits = [p['full_source217_inclusion_hits'] for p in ps]
        masses = [next(r['hard_only_importance_mass'] for r in p['partitions'] if r['region'] == 'A_patch_complete') for p in ps]
        points = [p['expected_complete_allocation_raw_points']/1e6 for p in ps]
        offsets = np.linspace(-.12, .12, 4)
        axes[0].scatter(width+offsets, np.asarray(hits)/512, color=color, s=40)
        axes[0].plot([width-.14,width+.14], [np.mean(hits)/512]*2, color=color, lw=2)
        for x, mass in zip(width+offsets, masses):
            if mass > 0: axes[1].scatter(x, mass, color=color, s=40)
            else: axes[1].text(x,.02,'×', color=color, ha='center', transform=axes[1].get_xaxis_transform())
        axes[2].scatter(width+offsets, points, color=color, s=40)
        summaries.append(dict(width_index=width, source_A_hits=hits, any_source_hits=any_hits,
                              hard_only_source_A_masses=masses, expected_raw_points_millions=points))
    axes[0].set_ylabel('Complete source-pattern hits / every attempted draw')
    axes[0].set_ylim(bottom=0)
    axes[0].set_title('Does the source guide cover the missing pocket?')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Hard-only source-region volume (Å³ × normalized Haar)')
    axes[1].set_title('Width sensitivity of the geometric mass estimate')
    axes[2].set_ylabel('Expected raw points for two clouds (millions)')
    axes[2].set_ylim(bottom=0)
    axes[2].set_title('Prospective cost; no clouds sampled')
    labels = ['0.1 Å\n0.5°', '0.2 Å\n1°', '0.4 Å\n2°', '0.8 Å\n4°']
    for ax in axes:
        ax.set_xticks(range(4), labels);ax.set_xlim(-.4, 3.4);ax.grid(axis='y', alpha=.2)
        ax.set_xlabel('Fixed source-chart width')
    fig.suptitle('Source-informed conditional proposal: coverage before depletion weights', fontsize=15)
    fig.text(.5,.895,'q = ½ uniform + ¼ context atlas + ¼ source Gaussian · 4 independent populations × 512 draws per width', ha='center')
    fig.text(.5,.045,'Source region = neighbors {16,217} with all 16 source secondary-contact tokens; additional tokens allowed.\n'
             'One mobile tetramer, 263 fixed spectators at 500 μM · rd = 1.5 Å, z = 0.035 Å⁻³ · No assembly or physical free-energy conclusion.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,bottom=.29,top=.78,wspace=.38)
    for suffix in ('png','svg'):fig.savefig(args.out_dir/f'source-guide-coverage.{suffix}',dpi=180)
    plt.close(fig)
    (args.out_dir/'provenance.json').write_text(json.dumps(dict(report=str(args.report.resolve()),
        report_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summaries=summaries, new_geometry_queries=0, clouds_sampled=0),indent=2)+'\n')


if __name__ == '__main__':main()
