"""Render an audited saved-bank depletion diagnostic without new queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('baseline', 'bridge')
COLORS = ('#4477aa', '#cc6677')
LABELS = ('Existing guides', '+ Bridge guides')
REGIONS = ('A_T', 'contact_without_A_T', 'unbound')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Changed completed audit')
    report = json.loads(args.report.read_bytes())
    if not (report['schema'] == 'context-bridge-physical-report-v1'
            and report['complete'] and report['passed']
            and report['all_attempts'] == 16384
            and report['population_denominator'] == 2048
            and report['confirmatory_gates_admitted'] is False
            and report['extension_scope']['fresh_confirmation'] is False):
        raise ValueError('Expected complete exploratory saved-bank audit')
    if args.out.exists():
        raise ValueError('Fresh plot directory required')
    args.out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    missing_log_points = 0
    for arm_index, arm in enumerate(ARMS):
        populations = sorted([p for p in report['populations'] if p['comparison_arm'] == arm],
                             key=lambda p: p['population_index'])
        if [p['population_index'] for p in populations] != list(range(4)):
            raise ValueError('Incomplete population inventory')
        tail = report['pooled_A_T_tail'][arm]
        bins = tail['bins']
        if [b['lower'] for b in bins] != [0., 6., 12., 24., 48., 96.]:
            raise ValueError('Changed original-chart radial partition')
        ax = axes[0, arm_index]
        for measure_index, measure in enumerate(('hard', 'physical')):
            values = np.array([b[measure+'_fraction_of_A_T'] for b in bins])
            if not np.isfinite(values).all() or not np.isclose(values.sum(), 1., atol=2e-12):
                raise ValueError('Source-tail fractions do not close')
            ax.bar(np.arange(6)+(-.18, .18)[measure_index], 100*values,
                   width=.34, color=('#888888', COLORS[arm_index])[measure_index],
                   label=('Geometric volume', 'Depletion weight')[measure_index])
        ax.set(title=LABELS[arm_index]+': source-contact tail',
               ylabel='Fraction of pooled A_T estimate [%]', ylim=(0, 105),
               xlabel='Original Gaussian squared Mahalanobis radius')
        ax.set_xticks(range(6), ['0–6', '6–12', '12–24', '24–48', '48–96', '96+'])
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.2)
        for region_index, region in enumerate(REGIONS):
            x = region_index+(-.13, .13)[arm_index]
            reference_mass = report['arm_summaries']['baseline'][region]['mean_mass']
            if not np.isfinite(reference_mass) or reference_mass <= 0:
                raise ValueError('Positive finite baseline mean required for regional normalization')
            for population_index, population in enumerate(populations):
                info = population['groups'][region]['physical']
                position = x+(population_index-1.5)*.03
                if info['mass'] > 0:
                    axes[1, 0].scatter(position, info['mass']/reference_mass, color=COLORS[arm_index], s=23)
                else:
                    missing_log_points += 1
                if info['largest_fraction'] is not None:
                    axes[1, 1].scatter(position, 100*info['largest_fraction'],
                                       color=COLORS[arm_index], s=23)
            mean = report['arm_summaries'][arm][region]['mean_mass']
            if mean > 0:
                axes[1, 0].plot([x-.075, x+.075], [mean/reference_mass]*2, color=COLORS[arm_index],
                                linewidth=2.5, label=LABELS[arm_index] if region_index == 0 else None)
    axes[1, 0].set(title='Physical masses: population disagreement',
                   ylabel='Mass / baseline mean of the same region', yscale='log')
    axes[1, 0].axhline(1., color='#555555', linestyle='--', linewidth=1)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].text(.02, .02, 'Dots: populations; bars: arithmetic means',
                    transform=axes[1, 0].transAxes, fontsize=8)
    axes[1, 1].set(title='Largest contribution within each population',
                   ylabel='Fraction of regional weight [%]', ylim=(-2, 102))
    axes[1, 1].axhline(2, color='#555555', linestyle='--', linewidth=1)
    for ax in axes[1]:
        ax.set_xticks(range(3), ['Source-complete A_T', 'Other contacts', 'Unbound'])
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Does the geometrically large tail carry depletion weight?\n'
                 'Exploratory saved bank · rd = 1.5 Å, z = 0.035 Å⁻³ · frozen 500 μM neighborhood', fontsize=13)
    fig.savefig(args.out/'comparison.png', dpi=180)
    fig.savefig(args.out/'comparison.pdf')
    plt.close(fig)
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Audit changed during rendering')
    receipt = dict(complete=True, passed=True, source_sha256=sha(__file__),
        source_report_sha256=args.expected_sha256,
        outputs={name: sha(args.out/name) for name in ('comparison.png', 'comparison.pdf')},
        missing_zero_mass_log_points=missing_log_points,
        mass_axis_normalization='Each region divided by its baseline arithmetic population mean; no cross-region physical ratio implied.',
        new_poses=0, new_clouds=0, new_geometry_queries=0,
        scope='Exploratory selected-bank diagnosis only. Source-contact tokens are not '
              'independent native registry; this does not establish equilibrium assembly or instability.')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
