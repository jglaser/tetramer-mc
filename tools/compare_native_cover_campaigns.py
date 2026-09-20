#!/usr/bin/env python3
"""Compare completed, independently audited uniform and mixture integrations.

No estimates are pooled and no sample weights are replaced. The two-cloud
variance decomposition is descriptive; it cannot detect unobserved tails.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_target(path):
    source = path / 'assessment-streaming.json'
    assessment = json.loads(source.read_text())
    assert assessment['complete'] and assessment['all_rows_and_hashes_validated']
    region = assessment['regions']['native']
    return dict(path=str(path), assessment_sha256=sha(source),
                cpu_seconds=assessment['cpu_seconds'],
                hard_volume=assessment['zero_activity_volume_estimate_A3'],
                population_logs=[r['regions']['native']['log_normalizer']
                                 for r in assessment['populations']], **region)


def ab_concentration(path):
    data = json.loads((path / 'assessment-streaming.json').read_text())
    logq = data['regions']['native']['log_normalizer']
    n = data['samples']
    bins = [0., .1, .2, .4, .6, .8, 1.]
    mass = [0.] * (len(bins) - 1)
    total = square = cloud_variance = 0.
    component_mass = {}
    for pop in sorted((path / 'runs').glob('r*')):
        manifest = json.loads((pop / 'manifest.json').read_text())
        assert manifest['cloud_replicates'] == 2
        cover = json.loads((pop / 'cover.json').read_text())
        with (pop / 'samples.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if 'zero' in row:
                    continue
                y = math.exp(row['log_importance_weight'] - logq)
                total += y
                square += y * y
                log_inverse_g = row.get('log_hard_weight', math.log(cover['volume']))
                clouds = [math.exp(w + log_inverse_g - logq) for w in row['cloud_log_weights']]
                # E[(W1-W2)^2/4 | x] = Var[(W1+W2)/2 | x].
                cloud_variance += (clouds[0] - clouds[1]) ** 2 / 4.
                index = min(len(mass) - 1, int(np.searchsorted(bins, row['q'], side='right')) - 1)
                assert index >= 0 and 0 <= row['q'] <= 1
                mass[index] += y
                component = str(row.get('proposal_component', 'uniform'))
                component_mass[component] = component_mass.get(component, 0.) + y
    assert abs(total / n - 1.) < 1e-9
    sample_square = square - total * total / n
    return dict(q_bin_edges=bins, q_bin_mass_fractions=[x / total for x in mass],
                proposal_component_mass_fractions={k: v / total for k, v in component_mass.items()},
                paired_cloud_fraction_of_observed_variance=cloud_variance / sample_square,
                paired_cloud_formula='sum ((W1-W2)/(2g))^2 / [sum Y^2-(sum Y)^2/N], Y=(W1+W2)/(2g)',
                interpretation='Conditional cloud-noise diagnostic from the original two clouds; not a tail-coverage bound.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uniform-A', type=Path, default=Path('runs/native-region-reference-8x4194304-l64-20260920'))
    p.add_argument('--uniform-factorial', type=Path, default=Path('runs/native-factorial-screen-20260920'))
    p.add_argument('--mixture', type=Path, default=Path('runs/native-cover-mixture-4x16384-l64-20260920'))
    p.add_argument('--AB-confirmation', type=Path)
    args = p.parse_args()
    labels = ['empty', 'A', 'B', 'AB']
    paths = {
        'uniform': {label: args.uniform_A if label == 'A' else args.uniform_factorial / label for label in labels},
        'mixture': {label: args.mixture / label for label in labels},
    }
    # Reuse the independently validated physical-identity accessor, excluding
    # proposal parameters from identity but retaining exact native/capture/bath.
    from analyze_native_factorial_screen import physical_signature
    for label in labels:
        assert physical_signature(paths['uniform'][label]) == physical_signature(paths['mixture'][label])
    reports = {method: {label: read_target(path) for label, path in targets.items()}
               for method, targets in paths.items()}
    concentration = {method: ab_concentration(targets['AB']) for method, targets in paths.items()}
    result = dict(same_physical_targets_verified=True, campaigns=reports, AB=concentration,
                  limitation='Different fixed budgets. Observed ESS, weight concentration and errors do not certify unsampled-tail coverage or a converged speedup.',
                  script_sha256=sha(Path(__file__)))
    if args.AB_confirmation:
        assert physical_signature(args.AB_confirmation) == physical_signature(paths['mixture']['AB'])
        result['AB_confirmation'] = read_target(args.AB_confirmation)
        concentration['confirmation'] = ab_concentration(args.AB_confirmation)
    output = args.mixture / 'comparison'
    output.mkdir(exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    colors = {'uniform': '#ad6b2d', 'mixture': '#26758e'}
    names = {'uniform': 'Uniform outer cover', 'mixture': 'Four nested covers'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    for method, shift in [('uniform', -.15), ('mixture', .15)]:
        rows = [reports[method][label] for label in labels]
        xs = np.arange(4) + shift
        for x, row in zip(xs, rows):
            logs = [v for v in row['population_logs'] if v is not None]
            axes[0].scatter(x + np.linspace(-.05, .05, len(logs)), logs,
                            s=16, alpha=.55, color=colors[method])
        axes[0].scatter(xs, [r['log_normalizer'] for r in rows], color=colors[method],
                        marker='D', s=42, label=names[method])
        axes[1].bar(xs, [100 * r['maximum_point_fraction'] for r in rows],
                    width=.28, color=colors[method])
        axes[2].bar(xs, [r['cpu_seconds'] for r in rows], width=.28, color=colors[method])
    axes[0].set_ylabel('log Q (translation Å³ × normalized Haar)')
    axes[0].set_title('Dots: independent populations\nDiamonds: pooled mass estimate')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].set_ylabel('Largest individual contribution / %')
    axes[1].set_title('Weight concentration\nLower is better; unseen tails remain possible')
    axes[2].set_ylabel('Summed sampling CPU seconds')
    axes[2].set_title('Cost at the declared budgets\nMixture: 65,536 draws per target')
    for ax in axes:
        ax.set_xticks(range(4), ['No neighbor', 'Neighbor A', 'Neighbor B', 'Both'])
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Native-region integration: same physical target, different proposal coverage', fontsize=14)
    fig.savefig(output / 'native-cover-comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    colors['confirmation'] = '#694b91'
    names['confirmation'] = 'Nested covers: fresh larger run'
    bars = [('uniform', -.24), ('mixture', 0.), ('confirmation', .24)] if args.AB_confirmation else [('uniform', -.18), ('mixture', .18)]
    for method, shift in bars:
        c = concentration[method]
        ax.bar(np.arange(6) + shift, np.array(c['q_bin_mass_fractions']) * 100,
               width=.22 if args.AB_confirmation else .34, color=colors[method], label=names[method])
    ax.set_xticks(range(6), ['0–0.1', '0.1–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1'])
    ax.set_xlabel('Original native registration error q')
    ax.set_ylabel('Fraction of estimated native-region mass / %')
    ax.set_title('Two neighbors: where the original importance weights lie')
    ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(output / 'AB-registration-weight.png', dpi=160)
    plt.close(fig)
    if args.AB_confirmation:
        rows = [reports['uniform']['AB'], reports['mixture']['AB'], result['AB_confirmation']]
        methods = ['uniform', 'mixture', 'confirmation']
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
        for x, (method, row) in enumerate(zip(methods, rows)):
            values = [v for v in row['population_logs'] if v is not None]
            axes[0].scatter(x + np.linspace(-.08, .08, len(values)), values, color=colors[method], s=20, alpha=.55)
            axes[0].scatter([x], [row['log_normalizer']], color=colors[method], s=65, marker='D')
            axes[1].bar(x, 100 * row['maximum_point_fraction'], color=colors[method], width=.6)
            axes[1].text(x, 100 * row['maximum_point_fraction'] + 2, f"ESS {row['ess']:.1f}", ha='center', fontsize=9)
            axes[2].bar(x, row['cpu_seconds'], color=colors[method], width=.6)
        captions = ['Uniform\n4 × 1,048,576', 'Nested pilot\n4 × 16,384', 'Fresh nested\n8 × 131,072']
        for ax in axes:
            ax.set_xticks(range(3), captions)
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        axes[0].set_ylabel('log Q for the same native region')
        axes[0].set_title('Diamonds: pooled estimates\nDots: independent populations')
        axes[1].set_ylabel('Largest sample contribution / %')
        axes[1].set_ylim(0, 110)
        axes[1].set_title('Concentration persists\nObserved ESS does not establish convergence')
        axes[2].set_ylabel('Sampling CPU seconds')
        axes[2].set_title('Sixteen times more mixture draws\nHigher cost did not resolve the mass')
        fig.suptitle('Two neighbors: centered geometric covers remain inadequate', fontsize=14)
        fig.savefig(output / 'AB-native-confirmation.png', dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
