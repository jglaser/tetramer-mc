#!/usr/bin/env python3
"""Plot an authenticated retrospective bridge analysis; no physical data are rerun."""
from __future__ import annotations
import argparse
import csv
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

REGION = 'remaining_R4_native'
STYLES = {
    'conditioned': ('IID conditioned', '#087f8c', 'o'),
    'baseline': ('IID baseline', '#7564a7', 's'),
    'unrestricted_broad': ('SMC broad', '#d46b27', '^'),
    'unrestricted_narrow': ('SMC narrow', '#3c873f', 'D'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, digits=4):
    return 'unobserved' if value is None else f'{value:.{digits}g}'


def profiles(data, name):
    return data['iid' if name in data['iid'] else 'smc'][name]['profiles']


def mean_mass(profile, region=REGION):
    estimate = profile['statistics']['estimates'][region]
    return estimate['log_Q'], estimate['population_relative_SE']


def mass_ratio(left, right, region=REGION):
    a, sa = mean_mass(left, region)
    b, sb = mean_mass(right, region)
    if a is None or b is None:
        return np.nan, np.nan
    return math.exp(a-b), math.hypot(sa, sb)


def validate(data):
    require(data.get('schema') == 'smc-bridge-bottleneck-retrospective-v1' and data.get('complete'),
            'Expected completed retrospective bridge analysis')
    for name in ('new_pose_draws', 'new_Poisson_clouds', 'new_classifier_calls'):
        require(data[name] == 0, 'Expected saved-data analysis without new sampling/classification')
    require(data['old_geometry_audits_replayed'] is False and data['physical_gate_open'] is False,
            'Unexpected audit replay or physical conclusion')
    beta = data['plan']['betas']
    require(beta == [i/8 for i in range(9)], 'Expected the nine saved profile points')
    for name in STYLES:
        rows = profiles(data, name)
        require([p['beta'] for p in rows] == beta, 'Inconsistent profile grid')
        require(all(p['statistics']['population_count'] == 4 for p in rows), 'Four population replicates required')
        if name in data['smc']:
            for p in rows:
                require(len(p['populations']) == 4, 'Missing SMC population')
                for pop in p['populations']:
                    n = pop['particles']
                    for region, count in pop['counts'].items():
                        stored = pop['log_masses'][region]
                        if count == 0:
                            require(stored is None, 'Zero class count has nonzero saved mass')
                        else:
                            expected = pop['log_masses']['total']+math.log(count/n)
                            require(stored is not None and abs(stored-expected) < 1e-10,
                                    'SMC class mass is not normalizer times indicator fraction')
    require(len(data['comparisons']) == len(beta), 'Missing comparisons')


def log_errorbar(ax, x, y, se, **kwargs):
    y, se = np.asarray(y, float), np.asarray(se, float)
    valid = np.isfinite(y) & (y > 0) & np.isfinite(se)
    low, high = y[valid]*np.exp(-se[valid]), y[valid]*np.exp(se[valid])
    ax.errorbar(np.asarray(x)[valid], y[valid], yerr=[y[valid]-low, high-y[valid]],
                capsize=2.2, elinewidth=.85, **kwargs)


def plot(data, out):
    beta = np.array(data['plan']['betas'])
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.9))
    fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.73, wspace=.32)
    for name, (label, color, marker) in STYLES.items():
        rows = profiles(data, name)
        y = [p['remainder_fraction'].get('fraction', np.nan) for p in rows]
        se = [p['remainder_fraction'].get('paired_population_log_ratio_SE', np.nan) for p in rows]
        log_errorbar(axes[0], beta, y, se, color=color, fmt=marker+'-', markersize=4.3, lw=1.5,
                     label=label, zorder=4 if name in data['iid'] else 3)
        if name in data['iid']:
            failed = [i for i, p in enumerate(rows) if not all(p['classes'][r]['passed'] for r in ['total', REGION])]
            axes[0].scatter(beta[failed], np.asarray(y)[failed], marker=marker, s=29,
                            facecolor='white', edgecolor=color, linewidth=1.2, zorder=6)
    axes[0].set(title='A  Remaining-native fraction', ylabel='Remaining-native / total bridge mass', yscale='log')
    axes[0].set_yticks([.001, .003, .01, .03, .1, .3, 1], labels=['.001', '.003', '.01', '.03', '.1', '.3', '1'])

    markers = ['o', 's', '^', 'D']
    for name in ['unrestricted_broad', 'unrestricted_narrow']:
        _, color, _ = STYLES[name]
        rows = profiles(data, name)
        for pi in range(4):
            values = [p['populations'][pi]['class_distinct_initial_families'][REGION] for p in rows]
            axes[1].plot(beta, values, color=color, marker=markers[pi], ms=4,
                         linestyle='-' if name == 'unrestricted_broad' else '--', lw=1.1, alpha=.88)
    axes[1].set(title='B  Families represented in the class', ylabel='Distinct original-draw families', yscale='log')
    axes[1].set_yticks([1, 2, 5, 10, 20, 50, 100, 200], labels=['1', '2', '5', '10', '20', '50', '100', '200'])
    axes[1].legend(handles=[Line2D([], [], color='#777', marker=m, ms=4, lw=0, label=f'r0{i}')
                           for i, m in enumerate(markers)], title='Independent SMC populations',
                   title_fontsize=8, fontsize=8, ncol=4, loc='upper center', frameon=False,
                   handletextpad=.3, columnspacing=.7)

    reference = profiles(data, 'conditioned')
    ratio_limits = []
    for name in ['baseline', 'unrestricted_broad', 'unrestricted_narrow']:
        label, color, marker = STYLES[name]
        values = [mass_ratio(p, q) for p, q in zip(profiles(data, name), reference)]
        ratio_limits.extend((value*math.exp(-se), value*math.exp(se))
                            for value, se in values if math.isfinite(value) and math.isfinite(se))
        log_errorbar(axes[2], beta, [x[0] for x in values], [x[1] for x in values],
                     color=color, fmt=marker+'-', ms=4.3, lw=1.5)
    axes[2].axhline(1, color='#444', ls=':', lw=1.2)
    axes[2].set(title='C  Remaining-native mass recovery', ylabel='Bridge mass / IID conditioned mass', yscale='log')
    axes[2].set_yticks([.05, .1, .2, .5, 1, 2, 5, 10], labels=['.05', '.1', '.2', '.5', '1', '2', '5', '10'])
    axes[2].set_ylim(min(v[0] for v in ratio_limits)*.8, max(v[1] for v in ratio_limits)*1.2)
    for ax in axes:
        ax.set_xlabel('Bridge coordinate β')
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlim(-.025, 1.025)
        ax.grid(axis='y', alpha=.18)
        ax.minorticks_off()
    fig.suptitle('Does the old SMC bridge suppress native configurations outside old R5?',
                 y=.985, fontsize=15)
    fig.text(.5, .922, r'Fixed bridge: $H\,g^{1-\beta}\exp(\beta z C)$   •   Saved IID poses and Poisson clouds; no new physics',
             ha='center', fontsize=11, color='#444')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .887), ncol=4, frameon=False,
               fontsize=10, columnspacing=2.1)
    fig.text(.065, .125, 'Bars: ±1 SE in log ratio from four independent whole-population estimates per arm. '
             'Fractions retain paired numerator–denominator covariance.', fontsize=9, color='#444')
    fig.text(.065, .09, 'β points share saved data and are correlated; connecting lines only guide the eye. '
             'Open IID markers fail one or more observed concentration checks.', fontsize=9, color='#444')
    fig.text(.065, .055, 'Family counts are genealogy diagnostics, not IID sample sizes. '
             'Intermediate masses describe the bridge, not physical bath free energies; unseen mass remains unresolved.',
             fontsize=9, color='#444')
    for ext in ['png', 'svg']:
        fig.savefig(out/f'smc-bridge-bottleneck.{ext}', dpi=200)
    plt.close(fig)


def write_tsv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(data, out):
    quality = []
    for arm, record in data['iid'].items():
        for profile in record['profiles']:
            for region, stats in profile['classes'].items():
                quality.append(dict(beta=profile['beta'], arm=arm, region=region,
                    observed=stats['observed'], log_mass=stats['log_mass'],
                    population_relative_SE=stats['population_relative_SE'], importance_ESS=stats['importance_ESS'],
                    largest_draw_fraction=stats['largest_draw_fraction'], passed=stats['passed']))
    write_tsv(out/'iid-region-quality.tsv', quality, list(quality[0]))
    strata = []
    for profile in data['comparisons']:
        for row in profile['all_iid_strata']:
            comp = row['comparison']
            strata.append(dict(beta=profile['beta'], family=row['family'], region=row['region'], bin=row['bin'],
                baseline_fraction=row['observed_fractions'][0], conditioned_fraction=row['observed_fractions'][1],
                material=row['material'], observed=comp['observed'], passed=comp['passed'],
                ratio_baseline_over_conditioned=comp.get('ratio_left_over_right'),
                absolute_passed=comp.get('absolute_passed'), SE_passed=comp.get('SE_passed'), reason=comp.get('reason')))
    write_tsv(out/'all-iid-stratum-comparisons.tsv', strata, list(strata[0]))
    write_tsv(out/'failed-material-iid-strata.tsv', [x for x in strata if x['material'] and not x['passed']], list(strata[0]))
    return quality


def report(data, source, digest, out, quality):
    lines = ['# Retrospective SMC bridge bottleneck', '',
        '![SMC bridge comparison](smc-bridge-bottleneck.png)', '',
        'The region is registered native entry within R4, excluding its exact old-R5 intersection. '
        'The same fixed class masks are used at every β. This exploratory analysis reuses the completed '
        'IID pilot and the previously audited broad/narrow SMC profiles; it creates no poses, clouds or classifications.', '',
        'The original R4 geometry, two-neighbor scaffold, full native definition, activity '
        'z = 0.035 Å⁻³, and depletant radius 1.5 Å remain fixed. The bridge coordinate changes '
        'the proposal-density factor and depletion exponent; it does not introduce a new physical target.', '',
        '## What the saved evidence shows', '',
        '| Arm | Fraction at β=0 | Smallest sampled fraction | β at minimum | Fraction at β=1 |',
        '|---|---:|---:|---:|---:|']
    for name, (label, _, _) in STYLES.items():
        rows = profiles(data, name)
        observed = [p for p in rows if p['remainder_fraction'].get('observed')]
        if not observed:
            lines.append(f'| {label} | unobserved | unobserved | — | unobserved |')
            continue
        low = min(observed, key=lambda p: p['remainder_fraction']['fraction'])
        lines.append(f"| {label} | {number(rows[0]['remainder_fraction'].get('fraction'))} | "
                     f"{number(low['remainder_fraction']['fraction'])} | {low['beta']:g} | "
                     f"{number(rows[-1]['remainder_fraction'].get('fraction'))} |")
    iid_lows = {arm: min(profiles(data, arm), key=lambda p: p['remainder_fraction'].get('fraction', math.inf))
                for arm in ['baseline', 'conditioned']}
    if all(0 < p['beta'] < 1 for p in iid_lows.values()):
        low = iid_lows['conditioned']
        first, last = profiles(data, 'conditioned')[0], profiles(data, 'conditioned')[-1]
        ratio = first['remainder_fraction']['fraction']/low['remainder_fraction']['fraction']
        rebound = last['remainder_fraction']['fraction']/low['remainder_fraction']['fraction']
        lines += ['', f'Both independent IID arms show an interior dip in the observed bridge fraction. '
                  f'The conditioned estimate falls {ratio:.1f}-fold from β=0 to its sampled minimum and '
                  f'then rises {rebound:.1f}-fold by β=1. This supports a bottleneck in the specified '
                  f'bridge itself, in addition to any SMC exploration failure. It is not solely an '
                  f'artifact of counting SMC descendants.']
        if not all(p['classes'][REGION]['passed'] for p in iid_lows.values()):
            lines += ['', 'The depth of that dip remains uncertain: the IID remaining-native concentration '
                      'checks fail at one or both sampled minima. Small four-population SEs do not remove '
                      'this limitation. The full quality and stratum failures are retained below.']
    lines += ['', 'These fractions are ratios of arithmetic mean **unnormalized population masses**. '
              'They are not arithmetic means of population fractions. The SMC numerator at each saved stage '
              'is the stage normalizer multiplied by the remaining-native particle fraction.', '',
              '| SMC arm / population | Initial families in class | Minimum families in class | β at minimum | '
              'Final families in class | Final descendants from initially other classes / all in class |',
              '|---|---:|---:|---|---:|---:|']
    for name in ['unrestricted_broad', 'unrestricted_narrow']:
        rows = profiles(data, name)
        for pi in range(4):
            counts = [p['populations'][pi]['class_distinct_initial_families'][REGION] for p in rows]
            last = rows[-1]['populations'][pi]
            locations = ', '.join(f"{p['beta']:g}" for p, c in zip(rows, counts) if c == min(counts))
            lines.append(f"| {STYLES[name][0]} / {last['id']} | {counts[0]} | {min(counts)} | {locations} | {counts[-1]} | "
                f"{last['descendants_from_initially_other_class'][REGION]} / {last['counts'][REGION]} |")
    lines += ['', 'The fall in class-specific family representation is a direct genealogy bottleneck. '
              'A later increase means surviving initial lineages have descendants entering the class; '
              'it does not restore extinct initial lineages. Family counts and descendant counts provide no IID standard errors.', '',
              '## Matched bridge mass comparison', '',
              'All ratios below compare arithmetic mean masses at the **same β** with conditioned IID. '
              'The plotted bars use propagated independent whole-population relative SEs in log-ratio space. '
              'They are pointwise descriptive one-SE intervals, not simultaneous bands.', '',
              '| β | IID baseline / conditioned | SMC broad / conditioned | SMC narrow / conditioned |',
              '|---:|---:|---:|---:|']
    for bi, beta in enumerate(data['plan']['betas']):
        values = [mass_ratio(profiles(data, name)[bi], profiles(data, 'conditioned')[bi])[0]
                  for name in ['baseline', 'unrestricted_broad', 'unrestricted_narrow']]
        lines.append(f'| {beta:g} | '+' | '.join(number(None if np.isnan(x) else x) for x in values)+' |')
    reference = profiles(data, 'conditioned')[-1]
    broad = mass_ratio(profiles(data, 'unrestricted_broad')[-1], reference)[0]
    narrow = mass_ratio(profiles(data, 'unrestricted_narrow')[-1], reference)[0]
    total = mass_ratio(profiles(data, 'unrestricted_narrow')[-1], reference, 'total')[0]
    lines += ['', f'At the endpoint, broad SMC recovers {broad:.3f}× and narrow SMC {narrow:.3f}× '
              f'the conditioned IID remaining-native mass. Narrow SMC has much stronger class-family '
              f'repopulation, but its total bridge normalizer is also {total:.3f}× conditioned IID. '
              'A recovered descendant fraction alone therefore cannot establish recovered class mass. '
              'The source comparison retains the separate absolute-difference and population-SE checks; '
              'a visually similar fraction or overlapping error bar does not override either check.']
    lines += ['', '## Observed quality and disagreements', '',
              'The IID concentration checks require population relative SE ≤ 0.10, importance ESS ≥ 200, '
              'and largest individual draw contribution ≤ 0.02. An unobserved class remains unresolved. '
              'These are observed diagnostics and do not certify coverage of unseen modes. '
              'The figure leaves an IID fraction marker open if either total or remaining-native mass fails these checks.', '',
              'A baseline/conditioned comparison passes only if the absolute log-mass difference is ≤ 0.2 '
              'and the linear-mass difference is within three combined population SEs. A stratum is material '
              'if it carries at least 1% of the observed class mass in either IID arm. These comparisons were '
              'made after sampling; none is a new physical validation gate.', '',
              '| β | Failed IID class checks: baseline | Failed IID class checks: conditioned | '
              'Failed IID region comparisons | Failed material IID strata |',
              '|---:|---|---|---|---:|']
    for comp in data['comparisons']:
        beta = comp['beta']
        failed = {a: [q['region'] for q in quality if q['arm'] == a and q['beta'] == beta and not q['passed']]
                  for a in ['baseline', 'conditioned']}
        regions = [name for name, c in comp['iid_regions'].items() if not c['passed']]
        lines.append(f"| {beta:g} | {', '.join(failed['baseline']) or 'none'} | "
                     f"{', '.join(failed['conditioned']) or 'none'} | {', '.join(regions) or 'none'} | "
                     f"{len(comp['failed_material_strata'])} |")
    lines += ['', 'Every per-class quality value is retained in [iid-region-quality.tsv](iid-region-quality.tsv). '
              'Every bin comparison, including empty and immaterial bins, is retained in '
              '[all-iid-stratum-comparisons.tsv](all-iid-stratum-comparisons.tsv). '
              'The complete material-failure list is [failed-material-iid-strata.tsv](failed-material-iid-strata.tsv). '
              'All original physical comparison failures remain in the source analysis.', '',
              '## Estimator and limits', '',
              'For an IID draw from latent proposal density q, the contribution is', '',
              '```text',
              'H × (J/q) × g^(1−β) × mean over two clouds [exp(β z L) × (1+β z/λ)^K]',
              '```', '',
              'Here g is the original normalized 50/50 current-ball plus unfiltered old-chart-ball SMC density '
              'in physical measure. The Poisson intensity λ stays at its saved value. The cloud factor is '
              'recomputed through its probability-generating function at βz; raising the noisy endpoint weight '
              'to β would produce a different estimator. All 16,384 attempted draws per population, including '
              'explicit zeros, remain in the denominator. The two clouds are averaged arithmetically before '
              'integration. Four populations per arm remain separate.', '',
              'Every β shares the same IID poses/clouds and the same SMC trajectory for its population, so '
              'the plotted β points are correlated. The minimum in the table is the smallest of nine saved '
              'grid points, not an estimate of the exact path minimum. At β=0 the bridge is H g; at β=1 it '
              'is the physical endpoint H exp(z C). Intermediate integrals include g^(1−β) and therefore '
              'are not physical-bath or finite-assembly free energies. Four populations give limited '
              'information about tails. Neither an observed bottleneck nor endpoint agreement establishes '
              'full-vessel coverage or finite-system stability.', '',
              '## Provenance', '', f'- Source: `{source}`', f'- Source SHA-256: `{digest}`',
              f'- Plot source SHA-256: `{sha(Path(__file__))}`',
              '- Numerical inputs are checked before plotting and rehashed before report completion.',
              '- New physical draws, new Poisson clouds, and new classifier calls: **0**.',
              '- Physical gate: **closed**. Finite-system conclusion: **unresolved**.', '']
    (out/'README.md').write_text('\n'.join(lines))


def run(source, out):
    require(not out.exists(), 'Preserve existing figure/report directory')
    digest = sha(source)
    data = json.loads(source.read_text())
    validate(data)
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out/'plot_smc_bridge_bottleneck.py')
    plot(data, out)
    quality = diagnostics(data, out)
    report(data, source, digest, out, quality)
    require(sha(source) == digest, 'Source analysis changed during plotting')
    receipt = dict(schema='smc-bridge-bottleneck-figure-v1', complete=True,
        source=str(source), source_sha256=digest, plot_source_sha256=sha(Path(__file__)),
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
        uncertainty='Pointwise delta-method log-ratio one-SE bars using independent whole-population means; '
                    'within-population numerator/denominator covariance retained for fractions. Correlated beta points.',
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'figure.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(out/'smc-bridge-bottleneck.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve())
