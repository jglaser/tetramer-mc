#!/usr/bin/env python3
"""Plot frozen physical count diagnostics; no sampling, geometry or audit rerun."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / 'results/auxiliary-overlap-physical-20261003'
DIAGNOSTIC_SHA = 'e15155fe54af0139393b9da12270bd9d15debae77f789da82af7821866955754'
BINDING_SHA = '13f1fc6627deaf8e65bca6f40da90a51b1335806c4ad14a219d7b2ee1f186f4d'
COLORS = {'baseline': '#778491', 'm1': '#237ba2', 'm4': '#b57626'}
ROOT, INTERNAL, BATH, JENSEN = '#47939c', '#8a4e99', '#235f9a', '#c7852c'
ORDER = ['blind_memory_allslot64', 'blind_fft512slots', 'native_informed178']
LABELS = {'blind_memory_allslot64': 'Blind memory', 'blind_fft512slots': 'Blind FFT',
          'native_informed178': 'Native informed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError('Changed frozen input: ' + str(path))
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def style(ax, labels):
    ax.set_yticks(range(9), labels)
    ax.set_ylim(8.6, -.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#e5e8ec', linewidth=.7)
    for y in (2.5, 5.5):
        ax.axhline(y, color='#dfe3e8', linewidth=.8)


def points(ax, entries, stratum, metric, color, offset, marker):
    for y, item in enumerate(entries):
        stat = item[stratum][metric]
        if not stat['count']:
            continue
        ax.errorbar(stat['median'], y + offset,
                    xerr=[[stat['median'] - stat['p05']], [stat['p95'] - stat['median']]],
                    fmt=marker, color=color, capsize=2.5, markersize=5, linewidth=1.4, zorder=3)


def draw(report, output):
    groups = {(c['atlas'], c['method']): c for c in report['comparisons']}
    entries = [groups[atlas, method] for atlas in ORDER for method in ('baseline', 'm1', 'm4')]
    labels = [f"{LABELS[c['atlas']]} · {c['method']}" for c in entries]
    internal_labels = [f"{label}  (n={c['internal_only']['candidates']})" for label, c in zip(labels, entries)]
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10}):
        fig = plt.figure(figsize=(15, 11.5), facecolor='white')
        grid = fig.add_gridspec(2, 2, left=.19, right=.965, bottom=.23, top=.82,
                               wspace=.68, hspace=.62)
        ax = fig.add_subplot(grid[0, 0])
        style(ax, labels)
        for y, c in enumerate(entries):
            n, outer = c['summary']['candidates'], c['summary']['outer_attempts']
            ax.barh(y, 100 * n / outer, height=.55, color=COLORS[c['method']], alpha=.94)
            ax.text(100 * n / outer + 2, y, f'{n}/{outer}', va='center', fontsize=9)
        ax.set(xlim=(0, 100), xticks=(0, 25, 50, 75, 100), xlabel='Candidates / all outer attempts (%)')
        ax.set_title('A   Unconditional candidate yield', loc='left', pad=28)
        ax.text(0, 1.035, 'Nulls stay in the denominator; zero accepted in every arm.',
                transform=ax.transAxes, fontsize=9, color='#495464')

        ax = fig.add_subplot(grid[0, 1])
        style(ax, labels)
        points(ax, entries, 'summary', 'log_root_f_correction', ROOT, -.13, 'o')
        points(ax, entries, 'summary', 'log_internal_f_correction', INTERNAL, .13, 's')
        ax.axvline(0, color='#66717e', linewidth=.9, linestyle='--')
        ax.set(xlim=(-47, 45), xlabel='Saved old − new full-mixture log density')
        ax.set_title('B   Full-F terms · all candidates', loc='left', pad=28)
        ax.legend([Line2D([], [], color=ROOT, marker='o'), Line2D([], [], color=INTERNAL, marker='s')],
                  ['Anchor–root', 'Root–child'], frameon=False, ncol=2,
                  loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=9)

        ax = fig.add_subplot(grid[1, 0])
        style(ax, internal_labels)
        points(ax, entries, 'internal_only', 'log_physical_bath_ratio_estimate', BATH, -.13, 'o')
        points(ax, entries, 'internal_only', 'estimated_auxiliary_log_penalty', JENSEN, .13, 'D')
        ax.axvline(0, color='#66717e', linewidth=.9, linestyle='--')
        ax.set(xlim=(-46, 20), xlabel='Saved-count logratio estimate / logweight penalty')
        ax.set_title('C   Bath terms · both endpoints internal only', loc='left', pad=46)
        ax.legend([Line2D([], [], color=BATH, marker='o'), Line2D([], [], color=JENSEN, marker='D')],
                  ['Physical logratio estimate ℓ', 'Bath Jensen penalty estimate'], frameon=False, ncol=1,
                  loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=9)

        ax = fig.add_subplot(grid[1, 1])
        style(ax, internal_labels)
        points(ax, entries, 'internal_only', 'log_root_f_correction', ROOT, -.13, 'o')
        points(ax, entries, 'internal_only', 'log_internal_f_correction', INTERNAL, .13, 's')
        ax.axvline(0, color='#66717e', linewidth=.9, linestyle='--')
        ax.set(xlim=(-47, 17), xlabel='Saved old − new full-mixture log density')
        ax.set_title('D   Full-F terms · both endpoints internal only', loc='left', pad=46)
        ax.legend([Line2D([], [], color=ROOT, marker='o'), Line2D([], [], color=INTERNAL, marker='s')],
                  ['Anchor–root', 'Root–child'], frameon=False, ncol=2,
                  loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=9)

        fig.suptitle('Physical overlap guide: rejection terms from saved counts',
                     x=.055, y=.966, ha='left', fontsize=18, fontweight='bold')
        fig.text(.055, .924,
                 '1,536 guided + 768 cached baseline outers; 324 + 511 candidates; zero accepts. '
                 'Radius 1.4 Å · activity 0.0275 Å⁻³ · 500 μM.', fontsize=11)
        fig.text(.055, .89,
                 'Full MH logratio = root F + internal F + auxiliary-target correction + realized bath logweight.\n'
                 'Guided auxiliary-target correction: median +0.358 overall, +0.237 in the internal-only stratum; included once.',
                 fontsize=10, color='#394451', linespacing=1.5)
        top = report['top_two_guided_conditional_alpha']
        fig.text(.055, .085,
                 'Points: candidate medians; bars: 5th–95th percentiles across saved candidates (not confidence intervals). '
                 'Internal only: old contacts = 1 and new external contacts = 0.\n'
                 f"The two largest guided conditional α values supply {100 * top['fraction_of_all_guided_conditional_alpha']:.2f}% "
                 'of Σα; both detach, losing three external edges and gaining none.\n'
                 'ℓ is an exploratory conditional count estimate, not an exactly known bath free energy; no exp(ℓ) or new MH test. '
                 'Fixed reset contexts do not establish equilibrium or refute the model.',
                 fontsize=9.4, color='#394451', linespacing=1.65)
        fig.savefig(output, dpi=180, metadata={'Title': 'Physical auxiliary overlap saved-count decomposition',
                                             'Diagnostic SHA256': DIAGNOSTIC_SHA,
                                             'Preanalysis binding SHA256': BINDING_SHA})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, default=RUN / 'saved-count-decomposition.json')
    parser.add_argument('--output', type=Path, default=REPO / 'docs/figures/auxiliary-overlap-physical.png')
    parser.add_argument('--binding', type=Path, default=RUN / 'saved-count-decomposition-plot-binding.json')
    parser.add_argument('--receipt', type=Path, default=RUN / 'saved-count-decomposition-plot-receipt.json')
    args = parser.parse_args()
    if args.output.exists() or args.binding.exists() or args.receipt.exists():
        raise ValueError('Refusing to overwrite a plot attempt')
    report = read_bound(args.diagnostic, DIAGNOSTIC_SHA)
    binding_path = args.diagnostic.parent / 'saved-count-decomposition-binding.json'
    read_bound(binding_path, BINDING_SHA)
    if not report['complete'] or not report['passed'] or report['binding_sha256'] != BINDING_SHA:
        raise ValueError('Incomplete diagnostic')
    files = {str(path.resolve()): sha(path) for path in (args.diagnostic, binding_path, Path(__file__))}
    for module in sys.modules.values():
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            files[str(path)] = sha(path)
    write_new(args.binding, dict(schema='auxiliary-overlap-physical-plot-binding-v1', complete=True,
                                 recorded_utc=datetime.now(timezone.utc).isoformat(), input_hashes=files,
                                 matplotlib_version=matplotlib.__version__, python=sys.version))
    receipt = dict(schema='auxiliary-overlap-physical-plot-receipt-v1', complete=False, passed=False,
                   binding_sha256=sha(args.binding), failures=[])
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        draw(report, args.output)
        if not all(sha(path) == digest for path, digest in files.items()):
            raise ValueError('Plot input changed during rendering')
        receipt.update(complete=True, passed=True, output=str(args.output.resolve()), output_sha256=sha(args.output))
    except Exception as error:
        receipt['failures'].append(str(error))
    write_new(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
