#!/usr/bin/env python3
"""Plot observed band weights and their concentration; no convergence inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a fresh plot directory')
    data = json.loads(source.read_text())
    rows = [('Geometric reference (training data)', data['reference'], '#777777')]
    for name, color in [('mixture', '#1769aa'), ('geometry', '#ca6821')]:
        matches = [r for key, r in data['guided_campaigns'].items() if f'-{name}-' in key]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name} campaign')
        rows.append((f'Fresh {name} guide', matches[0], color))
    keys, labels = ['0', '1', '2', '3', 'all'], ['2–2.5', '2.5–3', '3–4', '4–5', 'All: 2–5']
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for offset, (label, row, color) in zip([-.19, 0, .19], rows):
        values = [row['physical'][key] for key in keys]
        y = np.asarray([v['logQ'] for v in values], dtype=float)
        rse = np.asarray([v['row_RSE'] for v in values], dtype=float)
        if not np.isfinite(y).all() or not np.all((rse >= 0) & (rse < 1)):
            raise ValueError('This plot requires nonzero bands with finite Q ± one observed SE')
        x = np.arange(len(keys)) + offset
        errors = np.vstack([-np.log1p(-rse), np.log1p(rse)])
        axes[0].errorbar(x, y, yerr=errors, fmt='o', color=color, capsize=3, label=label, markersize=6)
        axes[1].plot(x, [100*v['maximum_fraction'] for v in values], 'o', color=color, markersize=6)
    axes[0].set_ylabel('log(Q / Å³)')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    axes[0].set_title('Intermediate contacts: more valid poses have not resolved the weight', loc='left', pad=15)
    axes[1].set_ylabel('Largest draw\n(% of band weight)')
    axes[1].set_ylim(-3, 105)
    axes[1].set_yticks([0, 50, 100])
    axes[1].set_xticks(np.arange(len(keys)), labels)
    axes[1].set_xlabel('Original registration coordinate q; lower boundaries included')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(3.5, color='#dddddd', linewidth=1)
    fig.text(.09, .015, 'Bars: Q ± one observed standard error, mapped to log scale; not tail bounds or confidence intervals.\n'
             'Both guides were fitted from the geometric reference. It is not held-out validation.', fontsize=9, color='#444444')
    fig.tight_layout(rect=[0, .085, 1, 1])
    output.mkdir(parents=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'intermediate-weights.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (output/'plotter.py').write_bytes(Path(__file__).read_bytes())
    (output/'provenance.json').write_text(json.dumps({'input': str(source),
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Original separate estimates and observed concentration; no pooled estimate or convergence claim.'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.comparison, args.out)
