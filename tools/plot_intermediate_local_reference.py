#!/usr/bin/env python3
"""Plot observed local physical/hard estimates without pooling campaigns."""
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


def plot(paths, out):
    assert not out.exists(), 'Use a fresh output directory'
    records = sorted([(p, json.loads(p.read_text())) for p in paths], key=lambda t: t[1]['radius'])
    assert len({d['radius'] for _, d in records}) == len(records)
    assert len({d['chart_model_sha256'] for _, d in records}) == 1
    sources = [('Uniform reference', '#167a55', 'o', -.18),
               ('Historical mixture', '#c57722', '^', 0.),
               ('Historical geometry', '#4770aa', 's', .18)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharex=True)
    for i, (_, data) in enumerate(records):
        guides = data['guided_campaigns']
        mixture = [v for k, v in guides.items() if 'mixture' in k]
        geometry = [v for k, v in guides.items() if 'geometry' in k]
        assert len(mixture) == len(geometry) == 1
        for item, (label, color, marker, dx) in zip([data['reference'], mixture[0], geometry[0]], sources):
            for ax, kind in zip(axes, ['physical', 'hard']):
                row = item[kind]['ball']
                if row['logQ'] is None:
                    ax.text(i+dx, .03, 'no hits', color=color, rotation=90, ha='center', va='bottom',
                            transform=ax.get_xaxis_transform(), fontsize=8)
                    continue
                y, se = row['logQ'], row['row_RSE']
                assert math.isfinite(y) and se is not None and 0 <= se <= 1+1e-9
                upper = math.log1p(se)
                lower = -math.log1p(-se) if se < 1-1e-12 else 2.5
                ax.errorbar(i+dx, y, yerr=np.array([[lower], [upper]]),
                            fmt=marker, color=color, capsize=4, markersize=7,
                            markerfacecolor=color if label == 'Uniform reference' else 'white',
                            label=label if i == 0 else None)
                if se >= 1-1e-12:
                    ax.annotate('', xy=(i+dx, y-lower-.35), xytext=(i+dx, y-lower),
                                arrowprops=dict(arrowstyle='->', color=color))
    for ax, title, ylabel in zip(axes, ['Depletion-weighted mass', 'Hard accessible pose volume'],
                                 [r'$\log Q$', r'$\log Q_0$']):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(records)), [f"Radius {d['radius']:g}" for _, d in records])
        ax.set_xlabel('Weighted-chart radius (dimensionless)', fontsize=9)
        ax.set_xlim(-.5, len(records)-.5)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none',
                      markerfacecolor=color if label == 'Uniform reference' else 'white',
                      label=label) for label, color, marker, _ in sources]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
    fig.suptitle('Independent integration around the intermediate-contact outlier', fontsize=14)
    fig.text(.5, .045, 'Same chart and AB neighbors; 2 ≤ q < 5; depletant radius 1.5 Å, activity 0.035 Å⁻³.',
             ha='center', fontsize=9)
    fig.text(.5, .01, 'Bars: Q ± observed SE, transformed to log scale. Historical restrictions are exploratory; unseen tails remain possible.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .92))
    out.mkdir(parents=True)
    for extension in ['png', 'svg']:
        fig.savefig(out/f'local-reference.{extension}', dpi=180, bbox_inches='tight')
    shutil.copy2(__file__, out/'plotter.py')
    provenance = dict(inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in records},
                      plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      scope='Fixed-region estimates only; no pooled normalizers or unseen-tail confidence bounds. Missing supported estimates shown as no hits.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    plt.close(fig)
    print(out/'local-reference.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.comparison, args.out)
