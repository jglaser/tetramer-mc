"""Render completed fixed-context physical weights; no fitting or new queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('baseline', 'multicage')
COLORS = {'baseline': '#4477aa', 'multicage': '#cc6677'}
LABELS = {'baseline': 'Existing F/D/B guides', 'multicage': '+ Competing-contact guides'}
GROUPS = ('A_T', 'contact_without_A_T', 'unbound', 'full_domain')
TICKS = ('Source-complete A', 'Other contacts', 'Unbound', 'Full domain')


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
        raise ValueError('Changed physical report')
    report = json.loads(args.report.read_bytes())
    if not (report['schema'] == 'context-multicage-physical-report-v1'
            and report['complete'] and report['passed']
            and report['all_attempts'] == 32768
            and report['population_denominator'] == 4096
            and report['estimator']['primary_estimator'] == 'pooled-count-rao-blackwell'):
        raise ValueError('Completed fixed-allocation primary report required')
    if args.out.exists():
        raise ValueError('Fresh render directory required')
    args.out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    mass, efficiency = axes[0]
    fractions, largest = axes[1]
    omitted = {'mass': 0, 'efficiency': 0, 'largest': 0}
    missing = {}
    for arm_index, arm in enumerate(ARMS):
        populations = [p for p in report['populations'] if p['comparison_arm'] == arm]
        if sorted(p['population_index'] for p in populations) != list(range(4)):
            raise ValueError('Exactly four independent populations per arm required')
        offset = (-.14, .14)[arm_index]
        jitter = np.linspace(-.045, .045, 4)
        for group_index, group in enumerate(GROUPS):
            values = [p['groups'][group]['physical']['mass'] for p in populations]
            eff = [p['importance_ess_per_cpu'][group] for p in populations]
            max_fraction = [p['groups'][group]['physical']['largest_fraction'] for p in populations]
            for index, (v, e, f) in enumerate(zip(values, eff, max_fraction)):
                x = group_index+offset+jitter[index]
                for ax, name, value in [(mass, 'mass', v), (efficiency, 'efficiency', e),
                                        (largest, 'largest', None if f is None else 100*f)]:
                    if value is not None and value > 0:
                        ax.scatter(x, value, color=COLORS[arm], s=27, alpha=.8)
                    else:
                        omitted[name] += 1
            for ax, mean in [(mass, report['arm_summaries'][arm][group]['mean_mass']),
                             (efficiency, sum(eff)/4)]:
                if mean > 0:
                    ax.plot([group_index+offset-.085, group_index+offset+.085],
                            [mean]*2, color=COLORS[arm], linewidth=2.5)
        categories = ('A_partial', 'B', 'other_contact')
        weights = np.asarray([[p['groups'][g]['physical']['mass'] for g in categories]
                              for p in populations])
        totals = np.asarray([p['groups']['contact_without_A_T']['physical']['mass']
                             for p in populations])
        if not np.allclose(weights.sum(axis=1), totals, rtol=2e-10, atol=0):
            raise ValueError('Competing-contact mass partition does not close')
        for row, total in zip(weights, totals):
            if total > 0:
                fractions.plot(range(3), row/total, color=COLORS[arm], alpha=.22, linewidth=.9)
        if totals.sum() > 0:
            fractions.plot(range(3), weights.sum(axis=0)/totals.sum(), 'o-',
                           color=COLORS[arm], linewidth=2, label=LABELS[arm])
        missing[arm] = {region: sum(p['regions'][region]['attempted_count'] for p in populations)
                       for region in ('A_patch_0.5_0.75', 'A_patch_0.75_1')}
    for ax, title, ylabel in [
            (mass, 'Physical regional masses', 'Importance mass [Å³ × normalized Haar]'),
            (efficiency, 'Importance concentration per CPU', 'Importance ESS / sampling CPU second'),
            (largest, 'Largest contribution in each population', 'Largest fraction of region weight [%]')]:
        ax.set_yscale('log')
        ax.set_xticks(range(4), TICKS, rotation=12)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
    largest.axhline(2., color='#555555', linestyle='--', linewidth=1, label='2% diagnostic threshold')
    largest.legend(frameon=False, fontsize=8)
    mass.text(.02, .02, 'Dots: four populations; bars: arithmetic means',
              transform=mass.transAxes, fontsize=8)
    efficiency.text(.02, .02, 'Weight ESS, not trajectory mixing ESS',
                    transform=efficiency.transAxes, fontsize=8)
    fractions.set(title='Composition of estimated competing-contact mass',
                  ylabel='Fraction of contact mass outside source-complete A',
                  ylim=(-.02, 1.02))
    fractions.set_xticks(range(3), ['Partial A contacts', 'Neighbor environment B', 'Other contacts'], rotation=10)
    fractions.grid(axis='y', alpha=.2)
    fractions.legend(frameon=False, fontsize=8)
    fractions.text(.02, .98, 'Bold: pooled mass fractions; faint: individual populations',
                   transform=fractions.transAxes, va='top', fontsize=8)
    fig.suptitle('Fixed protein environment: do competing-contact guides improve physical weights?\n'
                 'rd = 1.5 Å, z = 0.035 Å⁻³ · 4 × 4,096 attempts per arm · pooled-count Poisson estimator', fontsize=13)
    fig.savefig(args.out/'comparison.png', dpi=180)
    fig.savefig(args.out/'comparison.pdf')
    plt.close(fig)
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Source report changed during rendering')
    receipt = dict(complete=True, passed=True, source_sha256=sha(__file__),
        source_report=str(args.report.resolve()), source_report_sha256=args.expected_sha256,
        outputs={name: sha(args.out/name) for name in ('comparison.png', 'comparison.pdf')},
        zeros_omitted_from_log_axes=omitted, intermediate_partial_contact_hits=missing,
        new_fits=0, new_geometry_queries=0, new_clouds=0,
        scope='Conditional historical 500 μM frozen environment only. Source-complete patch labels '
              'are not native registry. Unseen contact regions remain unresolved. Weight ESS is '
              'importance concentration, not contact-fingerprint mixing or evidence of finite-system stability.')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
