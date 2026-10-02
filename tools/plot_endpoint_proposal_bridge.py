#!/usr/bin/env python3
"""Plot old/new bridge designs from completed retrospective summaries only."""
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
BRIDGES = [('old_profiles', 'Original SMC proposal g', '#7b7d83', 's'),
           ('profiles', 'This IID arm\'s frozen proposal g = q/J', '#087f8c', 'o')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def num(value, digits=4):
    return 'unobserved' if value is None else f'{value:.{digits}g}'


def validate(data):
    require(data.get('schema') == 'endpoint-proposal-bridge-retrospective-v1' and data.get('complete'),
            'Expected a completed endpoint-proposal bridge summary')
    require(data['plan']['common_target_between_arms_only_at_beta1'] is True,
            'Distinct intermediate targets must remain explicit')
    require(data['plan']['betas'] == [i/8 for i in range(9)], 'Unexpected beta grid')
    require(all(data[k] == 0 for k in ['new_pose_draws', 'new_Poisson_clouds', 'new_classifier_calls'])
            and data['raw_pose_or_cloud_rows_replayed'] is False and data['physical_gate_open'] is False,
            'Expected an offline design comparison')
    for arm in ['baseline', 'conditioned']:
        record = data['iid'][arm]
        for key, _, _, _ in BRIDGES:
            rows = record[key]
            require([r['beta'] for r in rows] == data['plan']['betas']
                    and all(r['statistics']['population_count'] == 4 for r in rows),
                    'Four population estimates required on the common beta grid')
        require(record['profiles'][-1]['classes'] == record['old_profiles'][-1]['classes']
                and record['profiles'][-1]['strata'] == record['old_profiles'][-1]['strata'],
                'Physical endpoint moments changed')
        require([r['beta'] for r in record['old_to_new_fraction']] == data['plan']['betas'],
                'Missing paired bridge comparisons')
        endpoint = record['old_to_new_fraction'][-1]
        if endpoint['observed']:
            require(abs(endpoint['fraction_ratio']-1) < 1e-12
                    and abs(endpoint['paired_population_log_ratio_SE']) < 1e-12,
                    'Endpoint equality must be by construction')


def figure(data, out):
    beta = np.array(data['plan']['betas'])
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.7), sharex=True)
    fig.subplots_adjust(left=.1, right=.97, top=.77, bottom=.225, wspace=.27, hspace=.28)
    frac_limits, ess_limits = [], []
    for col, arm in enumerate(['baseline', 'conditioned']):
        record = data['iid'][arm]
        for key, label, color, marker in BRIDGES:
            rows = record[key]
            y = np.array([r['remainder_fraction'].get('fraction', np.nan) for r in rows])
            se = np.array([r['remainder_fraction'].get('paired_population_log_ratio_SE', np.nan) for r in rows])
            positive = np.isfinite(y) & (y > 0) & np.isfinite(se)
            lower, upper = y[positive]*np.exp(-se[positive]), y[positive]*np.exp(se[positive])
            frac_limits.extend(zip(lower, upper))
            axes[0, col].errorbar(beta[positive], y[positive], yerr=[y[positive]-lower, upper-y[positive]],
                fmt=marker+'-', ms=4.4, lw=1.5, capsize=2.3, elinewidth=.85, color=color, label=label)
            failed = [i for i, r in enumerate(rows) if not all(r['classes'][k]['passed'] for k in ['total', REGION])]
            axes[0, col].scatter(beta[failed], y[failed], marker=marker, s=31,
                                facecolor='white', edgecolor=color, linewidth=1.1, zorder=5)
            ess = np.array([r['classes'][REGION]['importance_ESS'] for r in rows])
            ess_limits.extend(ess[ess > 0])
            axes[1, col].plot(beta, np.where(ess > 0, ess, np.nan), marker=marker, ms=4.4, lw=1.5, color=color)
            failed_ess = [i for i, r in enumerate(rows) if not r['classes'][REGION]['passed']]
            axes[1, col].scatter(beta[failed_ess], ess[failed_ess], marker=marker, s=31,
                                facecolor='white', edgecolor=color, linewidth=1.1, zorder=5)
        axes[0, col].set_title(f"{'A' if col == 0 else 'B'}  {arm.capitalize()} IID arm: bridge fractions", loc='left')
        axes[1, col].set_title(f"{'C' if col == 0 else 'D'}  {arm.capitalize()} IID arm: observed concentration", loc='left')
        axes[1, col].axhline(200, color='#555', linestyle=':', linewidth=1)
        axes[1, col].text(.03, 200, 'ESS = 200', fontsize=8, ha='left', va='bottom',
                           color='#555', transform=axes[1, col].get_yaxis_transform())
        axes[1, col].set_xlabel('Bridge coordinate β')
    for ax in axes.flat:
        ax.set_yscale('log')
        ax.set_xlim(-.025, 1.025)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(axis='y', alpha=.18)
        ax.minorticks_off()
    for ax in axes[0]:
        ax.set_yticks([.001, .003, .01, .03, .1, .3, 1], labels=['.001', '.003', '.01', '.03', '.1', '.3', '1'])
        ax.set_ylim(min(v[0] for v in frac_limits)*.78, max(v[1] for v in frac_limits)*1.2)
    for ax in axes[1]:
        ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
                     labels=['20', '50', '100', '200', '500', '1,000', '2,000', '5,000', '10,000'])
        ax.set_ylim(min(ess_limits)*.78, max(ess_limits)*1.2)
    axes[0, 0].set_ylabel('Remaining-native / total bridge mass')
    axes[1, 0].set_ylabel('Remaining-native importance ESS\n(weight concentration, not SMC mixing)')
    fig.suptitle('Changing the initial proposal changes the intermediate bridge', fontsize=15, y=.98)
    fig.text(.5, .937, r'Old and new: $H\,g^{1-\beta}\exp(\beta z C)$; only the proposal factor $g$ changes',
             ha='center', fontsize=11, color='#444')
    fig.text(.5, .905, 'Each new bridge uses its own arm\'s frozen g = q/J. '
             'The arms have different targets below β = 1.', ha='center', fontsize=10, color='#444')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .875), ncol=2,
               frameon=False, fontsize=10)
    fig.text(.1, .125, 'Bars: ±1 SE in log fraction from four independent population means, with paired numerator–denominator covariance.', fontsize=8.5, color='#444')
    fig.text(.1, .095, 'Open markers fail an observed concentration check. β points and old/new curves reuse the same draws and clouds.', fontsize=8.5, color='#444')
    fig.text(.1, .065, 'Design evidence only: no new SMC populations, mixing result, or speedup estimate. Endpoint equality is built into the transform.', fontsize=8.5, color='#444')
    for ext in ['png', 'svg']:
        fig.savefig(out/f'endpoint-proposal-bridge.{ext}', dpi=200)
    plt.close(fig)


def tsv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(data, out):
    quality, strata, changes = [], [], []
    for arm, record in data['iid'].items():
        for key, _, _, _ in BRIDGES:
            bridge = 'original' if key == 'old_profiles' else 'own_IID_proposal'
            for profile in record[key]:
                for region, r in profile['classes'].items():
                    quality.append(dict(arm=arm, bridge=bridge, beta=profile['beta'], region=region,
                        observed=r['observed'], log_mass=r['log_mass'], population_relative_SE=r['population_relative_SE'],
                        importance_ESS=r['importance_ESS'], largest_draw_fraction=r['largest_draw_fraction'], passed=r['passed']))
                for family, classes in profile['strata'].items():
                    for region, bins in classes.items():
                        parent = profile['classes'][region]
                        for k, r in enumerate(bins):
                            fraction = (math.exp(r['log_mass']-parent['log_mass'])
                                        if r['observed'] and parent['observed'] else None)
                            strata.append(dict(arm=arm, bridge=bridge, beta=profile['beta'], family=family, region=region, bin=k,
                                observed=r['observed'], class_fraction=fraction, material=fraction is not None and fraction >= .01,
                                log_mass=r['log_mass'], population_relative_SE=r['population_relative_SE'],
                                importance_ESS=r['importance_ESS'], largest_draw_fraction=r['largest_draw_fraction'], passed=r['passed']))
        for row in record['old_to_new_fraction']:
            changes.append(dict(arm=arm, beta=row['beta'], observed=row['observed'],
                fraction_ratio_new_over_old=row.get('fraction_ratio'),
                paired_population_log_ratio_SE=row.get('paired_population_log_ratio_SE')))
    tsv(out/'all-region-quality.tsv', quality, list(quality[0]))
    tsv(out/'all-stratum-quality.tsv', strata, list(strata[0]))
    tsv(out/'failed-material-stratum-quality.tsv', [r for r in strata if r['material'] and not r['passed']], list(strata[0]))
    tsv(out/'paired-fraction-changes.tsv', changes, list(changes[0]))
    return quality, strata


def report(data, source, digest, out, quality, strata):
    lines = ['# Alternative bridge based on each frozen IID proposal', '',
        '![Alternative proposal bridge](endpoint-proposal-bridge.png)', '',
        'This is a retrospective bridge-design comparison using completed IID poses, fixed native masks, '
        'and already derived Poisson-cloud weights. No raw pose or cloud rows are reopened. '
        'No new SMC populations, physical draws, clouds, classifications, or geometry audits are generated.', '',
        '**Each arm has a different alternative target below β=1.** Baseline uses its own frozen proposal '
        'density; conditioned uses its own. The old bridge uses the same historical SMC proposal in both arms. '
        'Old/new changes are paired descriptive changes of target, not agreement tests. The common physical '
        'endpoint is identical by construction, so endpoint equality supplies no new validation.', '',
        '## Observed fractions and quality', '']
    conditioned = data['iid']['conditioned']['profiles']
    fractions = [r['remainder_fraction']['fraction'] for r in conditioned]
    if min(fractions) == fractions[0]:
        all_primary_pass = all(r['classes'][k]['passed'] for r in conditioned for k in ['total', REGION])
        ess = [r['classes'][REGION]['importance_ESS'] for r in conditioned]
        lines += [f'The conditioned arm\'s own-proposal bridge has no observed interior dip below its '
            f'initial remaining-native fraction on this grid: it starts at {100*fractions[0]:.3f}%, '
            f'reaches {100*max(fractions):.3f}%, and ends at {100*fractions[-1]:.3f}%. '
            f'Remaining-native importance ESS ranges from {min(ess):.1f} to {max(ess):.0f}. '
            f'Total and remaining-native aggregate concentration checks '
            f'{"pass at all nine β values" if all_primary_pass else "do not pass at every β value"}. '
            'This is favorable evidence about this specified alternative bridge, with the limits below.', '']
    decision = ['registered_native_entry', 'old_R5_intersection_native', REGION, 'contact_no_native_entry']
    if all(r['classes'][k]['passed'] for r in conditioned for k in decision):
        lines += ['All four nonempty decision classes in the conditioned alternative also pass their '
                  'aggregate concentration checks at every saved β. This aggregate result does not '
                  'remove individual stratum failures, which are preserved in the complete tables.', '']
    for arm in ['baseline', 'conditioned']:
        record = data['iid'][arm]
        lines += [f'### {arm.capitalize()} IID arm', '',
            '| β | Original fraction | Own-proposal fraction | New / old fraction | Paired SE of log(new / old) | '
            'New remainder ESS | New largest draw | New total + remainder quality |',
            '|---:|---:|---:|---:|---:|---:|---:|---|']
        for old, new, change in zip(record['old_profiles'], record['profiles'], record['old_to_new_fraction']):
            r = new['classes'][REGION]
            passed = all(new['classes'][k]['passed'] for k in ['total', REGION])
            lines.append(f"| {new['beta']:g} | {num(old['remainder_fraction'].get('fraction'))} | "
                f"{num(new['remainder_fraction'].get('fraction'))} | {num(change.get('fraction_ratio'))} | "
                f"{num(change.get('paired_population_log_ratio_SE'))} | {num(r['importance_ESS'])} | "
                f"{num(r['largest_draw_fraction'])} | {'pass' if passed else 'fail'} |")
        old_low = min(record['old_profiles'], key=lambda r: r['remainder_fraction'].get('fraction', math.inf))
        new_low = min(record['profiles'], key=lambda r: r['remainder_fraction'].get('fraction', math.inf))
        lines += ['', f"The smallest sampled original fraction is {num(old_low['remainder_fraction'].get('fraction'))} "
            f"at β={old_low['beta']:g}; the smallest own-proposal fraction is "
            f"{num(new_low['remainder_fraction'].get('fraction'))} at β={new_low['beta']:g}. "
            'These minima concern nine correlated grid points, not an exact minimum over a continuous path.', '']
    lines += ['The native-informed frozen guide remains native-informed. A favorable bridge profile supports '
        'considering that design; it does not demonstrate easy mutation, preserved genealogy, a good SMC '
        'normalizer, reduced runtime, or coverage of unseen modes. The old-bridge concentration failures '
        'remain part of the evidence.', '', '## Complete diagnostics', '',
        'For every observed class or bin, the diagnostic passes only if population relative SE ≤ 0.10, '
        'importance ESS ≥ 200, and largest draw contribution ≤ 0.02. Importance ESS describes concentration '
        'of direct IID importance weights. It is not an SMC particle/family ESS or a mixing measurement. '
        'The plotted fraction marker is open if either total or remaining-native fails; the ESS marker is '
        'open if remaining-native fails any of the three checks.', '',
        '| Arm | Bridge | β | Failed class checks | Material stratum quality failures |',
        '|---|---|---:|---|---:|']
    for arm in ['baseline', 'conditioned']:
        for bridge in ['original', 'own_IID_proposal']:
            for beta in data['plan']['betas']:
                failed = [r['region'] for r in quality if r['arm'] == arm and r['bridge'] == bridge
                          and r['beta'] == beta and not r['passed']]
                count = sum(r['arm'] == arm and r['bridge'] == bridge and r['beta'] == beta
                            and r['material'] and not r['passed'] for r in strata)
                lines.append(f"| {arm} | {bridge} | {beta:g} | {', '.join(failed) or 'none'} | {count} |")
    lines += ['', 'Material means at least 1% of that observed class mass within the same arm, bridge and β. '
        'These are individual stratum concentration checks, not old/new or cross-arm agreement checks. '
        'Unobserved bins remain explicitly unobserved; a zero observation supplies no upper bound.', '',
        '- [All region quality values](all-region-quality.tsv)',
        '- [All stratum quality values, including unobserved and immaterial bins](all-stratum-quality.tsv)',
        '- [All failed material stratum checks](failed-material-stratum-quality.tsv)',
        '- [Paired old/new fraction changes](paired-fraction-changes.tsv)', '',
        '## Construction and interpretation', '',
        'For each arm separately, replace the old SMC density by its own frozen physical proposal density '
        'g_new = q_IID/J. The alternative target is H g_new^(1−β) exp(βzC), whose direct IID contribution is', '',
        '```text', 'H × (J/q_IID)^β × mean over two clouds [exp(β z L) × (1+β z/λ)^K]', '```', '',
        'The Poisson probability-generating function is evaluated at βz with the original saved intensity λ. '
        'This changes only the proposal-density factor in the previously derived weights; it does not raise '
        'the noisy endpoint cloud factor to β. All 16,384 attempted draws per population, including explicit '
        'invalid zeros, remain in the denominator. Each arm retains four independent populations.', '',
        'At β=0 every valid draw contributes one and every invalid draw contributes zero. Thus the total '
        'mass estimates unconditional proposal feasibility, while the plotted region fraction is the '
        'fraction of feasible proposal draws in remaining native. It is not hard-fluid equilibrium. '
        'At β=1 all old/new moments, including the fixed strata, are identical by construction.', '',
        'Fractions are ratios of arithmetic mean unnormalized masses. Their uncertainty retains paired '
        'within-population numerator/denominator covariance. The SE of the new/old fraction change retains '
        'all four paired mass measurements from each population. β points share saved poses and clouds '
        'and are correlated. The plotted one-SE bars are pointwise descriptive errors, not simultaneous '
        'coverage or uncertainty about unseen modes. Intermediate targets are not physical bath free energies. '
        'Full-vessel coverage and finite-system stability remain unresolved.', '', '## Provenance', '',
        f'- Analysis: `{source}`', f'- Analysis SHA-256: `{digest}`',
        f'- Plot source SHA-256: `{sha(Path(__file__))}`',
        '- Physical gate remains closed; no new physical or SMC samples were generated.', '']
    (out/'README.md').write_text('\n'.join(lines))


def run(source, out):
    require(not out.exists(), 'Preserve existing report directory')
    digest = sha(source)
    data = json.loads(source.read_text())
    validate(data)
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out/'plot_endpoint_proposal_bridge.py')
    figure(data, out)
    quality, strata = diagnostics(data, out)
    report(data, source, digest, out, quality, strata)
    require(sha(source) == digest, 'Source analysis changed during plotting')
    receipt = dict(schema='endpoint-proposal-bridge-figure-v1', complete=True, source=str(source), source_sha256=digest,
        plot_source_sha256=sha(Path(__file__)), matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        new_physical_or_SMC_draws=0, targets_differ_below_beta1=True,
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'figure.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(out/'endpoint-proposal-bridge.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve())
