#!/usr/bin/env python3
"""Summarize the completed frozen pilot without replaying physical/classifier work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REGIONS = {
    'registered_native_entry': 'All native',
    'contact_no_native_entry': 'All competing',
    'old_R5_intersection_native': 'Native inside R5',
    'remaining_R4_native': 'Native outside R5',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def extract(estimate, cpu):
    row, pop = estimate['row_uncertainty'], estimate['population_uncertainty']
    return dict(log_Qz=pop['log_Qz'], log_Q0=pop['log_Q0'],
                population_relative_SE=pop['Qz_relative_SE'],
                importance_ESS=row['Qz_ESS'], largest_draw_fraction=row['largest_Qz_fraction'],
                nonzero=row['nonzero'], importance_ESS_per_cpu_second=row['Qz_ESS']/cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pilot, out = args.pilot.resolve(), args.out.resolve()
    check(not out.exists(), 'Refuse to overwrite a completed review')
    status, protocol = read(pilot/'status.json'), read(pilot/'protocol.json')
    a = read(pilot/'comparison/analysis.json')
    check(status['complete'] and a['complete'], 'Incomplete pilot')
    check(sha(pilot/'protocol.json') == status['protocol_sha256'] == a['protocol_sha256'],
          'Protocol binding changed')
    check(sha(pilot/'comparison/analysis.json') == status['comparison_sha256'],
          'Comparison binding changed')
    check(len(status['jobs']) == 8 and a['total_unconditional_draws'] == 131072,
          'Unexpected allocation')
    verified = {}
    for name in ('status.json', 'protocol.json', 'freeze.json', 'comparison/analysis.json'):
        verified[str(pilot/name)] = sha(pilot/name)
    rows = 0
    for job in status['jobs']:
        check(job['status'] == 'complete' and job['returncode'] == 0, 'Failed job')
        directory = Path(job['directory'])
        for name, key in [('samples.jsonl', 'samples_sha256'), ('summary.json', 'summary_sha256'),
                          ('manifest.json', 'manifest_sha256')]:
            value = sha(directory/name)
            check(value == job['output'][key], 'Saved physical output changed: '+str(directory/name))
            verified[str(directory/name)] = value
        with (directory/'samples.jsonl').open('rb') as stream:
            count = sum(1 for _ in stream)
        check(count == job['samples'] == 16384, 'Attempted draw lost')
        rows += count
    check(rows == 131072, 'Wrong unconditional denominator')
    for arm, audit in status['audits'].items():
        path = Path(audit['directory'])/'analysis.json'
        check(audit['status'] == 'complete' and sha(path) == audit['analysis_sha256'], 'Audit binding changed')
        verified[str(path)] = sha(path)
    check(len(status['classified']) == 8, 'Missing classifier pass')
    check(not a['old_samples_pooled'] and a['old_audits_replayed'] == 0, 'Historical work repeated/pooled')

    arms = {}
    for arm, data in a['arms'].items():
        cpu = data['sampler_cpu_seconds']
        arms[arm] = dict(cpu_seconds=cpu, regions={r: extract(data['estimates'][r], cpu) for r in REGIONS},
                         critical_strata={}, proposal_audit=data['proposal_audit'])
        for region, index in [('remaining_R4_native', 55), ('contact_no_native_entry', 63),
                              ('contact_no_native_entry', 58), ('contact_no_native_entry', 55)]:
            source = data['strata']['orthant'][region][index]
            estimate = extract(source, cpu)
            estimate['fraction_of_own_region'] = source['observed_class_fraction']['Qz']
            arms[arm]['critical_strata'][f'{region}:{index}'] = estimate

    diagnostics = a['diagnostics']
    significant = [x for x in diagnostics['all_stratum_comparisons'] if x['significant']]
    failed = [x for x in significant if not x['comparison']['passed']]
    efficiency = []
    for r, label in REGIONS.items():
        b, e = (arms[x]['regions'][r] for x in ('baseline', 'expanded'))
        efficiency.append(dict(label=label, baseline=b['importance_ESS_per_cpu_second'],
                               expanded=e['importance_ESS_per_cpu_second'], ratio=
                               e['importance_ESS_per_cpu_second']/b['importance_ESS_per_cpu_second']))
    for r, i, label in [('remaining_R4_native', 55, 'Native outside R5: orthant 55'),
                        ('contact_no_native_entry', 63, 'Competing: orthant 63'),
                        ('contact_no_native_entry', 58, 'Competing: orthant 58')]:
        b, e = (arms[x]['critical_strata'][f'{r}:{i}'] for x in ('baseline', 'expanded'))
        efficiency.append(dict(label=label, baseline=b['importance_ESS_per_cpu_second'],
                               expanded=e['importance_ESS_per_cpu_second'], ratio=
                               e['importance_ESS_per_cpu_second']/b['importance_ESS_per_cpu_second']))
    summary = dict(schema='contact-tail-pilot-completed-review-v1', source=verified,
        attempted_draws=rows, wall_seconds=status['finished']-status['started'], arms=arms,
        efficiency=efficiency, diagnostics=diagnostics,
        significant_stratum_count=len(significant), failed_significant_stratum_count=len(failed),
        failed_significant_strata=failed, unbound_R4_bound=a['unbound_R4_bound'],
        interpretation='Useful targeted integration efficiency, unresolved region coverage; no assembly verdict.',
        efficiency_scope='Observed importance-weight ESS / physical sampler CPU; no Markov-chain mixing claim; excludes fitting, auditing and classification CPU.')

    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    colors = {'baseline': '#43556e', 'expanded': '#0a8f86'}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={'width_ratios': [0.95, 1.45, 1.4]})
    for y, arm in enumerate(('baseline', 'expanded')):
        f = diagnostics['free_energy_intervals'][arm]
        ax[0].errorbar(f['beta_F_native_minus_noentry'], y, xerr=f['halfwidth_95'], fmt='o',
                       capsize=4, color=colors[arm], markersize=7)
    ax[0].set_yticks([0, 1], ['84 components', '92 components'])
    ax[0].set_ylim(1.8, -.8)
    ax[0].set_xlabel(r'$(F_{native}-F_{competing})/k_BT$')
    ax[0].set_title('Aggregate conditional weights agree\nPopulation 95% delta intervals', fontsize=11)
    y = np.arange(len(efficiency))
    ax[1].barh(y, [e['ratio'] for e in efficiency], color=['#8b9da8']*4+['#0a8f86']*2+['#8b9da8'])
    ax[1].set_yticks(y, [e['label'] for e in efficiency], fontsize=9)
    ax[1].invert_yaxis(); ax[1].axvline(1, color='#17212b', ls='--', lw=1)
    for i, e in enumerate(efficiency):
        ax[1].text(e['ratio']+.04, i, f"{e['ratio']:.2f}×", va='center', fontsize=9)
    ax[1].set_xlim(0, 4.4)
    ax[1].set_xlabel('Expanded / baseline importance ESS per CPU')
    ax[1].set_title('Two targeted tails improve\nObserved efficiency, not trajectory mixing', fontsize=11)
    selected = [('remaining_R4_native',55,'Native outside R5: 55'),
                ('contact_no_native_entry',63,'Competing: 63'),
                ('contact_no_native_entry',58,'Competing: 58'),
                ('contact_no_native_entry',55,'Competing: 55')]
    for y, (region, index, _) in enumerate(selected):
        record = next(x['comparison'] for x in diagnostics['all_stratum_comparisons']
                      if x['family']=='orthant' and x['region']==region and x['bin']==index)
        ax[2].errorbar(record['log_left_minus_right'], y,
                       xerr=3*record['combined_population_log_delta_SE'], fmt='o', capsize=4,
                       color='#0a8f86' if record['passed'] else '#bf6639')
    ax[2].axvspan(-.2, .2, color='#a4cbbd', alpha=.35)
    ax[2].axvline(0, color='#17212b', lw=.8)
    ax[2].set_yticks(range(4), [x[2] for x in selected], fontsize=9)
    ax[2].set_ylim(3.7, -.7)
    ax[2].set_xlabel('log mass: baseline − expanded (bars: 3 SE)')
    ax[2].set_title('Coverage still fails\n14 / 34 material stratum comparisons', fontsize=11)
    for axis in ax:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.18)
    fig.suptitle('Fresh fixed pilot: 4 × 16,384 attempts per arm; 1.5 Å, 0.035 Å⁻³', fontsize=13)
    fig.text(.5, .015, 'Original R4 / fixed scaffold only. No vessel remainder bound or finite-system assembly conclusion.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .93))
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'contact-tail-pilot.{suffix}', dpi=180)
    plt.close(fig)

    lines = ['# Fresh contact-tail pilot: useful coverage, unresolved thermodynamics', '',
      f'All **{rows:,} attempted draws** completed and were independently audited and classified. ',
      f'Wall time: {summary["wall_seconds"]/60:.2f} minutes; physical sampler CPU: '+
      f'{sum(x["cpu_seconds"] for x in arms.values()):.2f} seconds. '+
      'The 84- and 92-component arms each contain four independent 16,384-draw populations, '+
      'two clouds per valid pose, intensity ratio 128 and 50% uniform defensive support. ',
      'The unchanged model is the repaired rigid tetramer, 1.5 Å depletants and activity 0.035 Å⁻³. '+
      'The region is the original finite R4 around a prescribed fixed scaffold, not a mobile assembly ensemble.', '',
      '![Completed pilot](contact-tail-pilot.png)', '',
      '## Aggregate weights', '',
      '| Region | Baseline log mass | Expanded log mass | Baseline / expanded importance ESS |',
      '|---|---:|---:|---:|']
    for r, label in REGIONS.items():
        b, e = (arms[x]['regions'][r] for x in ('baseline','expanded'))
        lines.append(f'| {label} | {b["log_Qz"]:.6f} | {e["log_Qz"]:.6f} | {b["importance_ESS"]:.1f} / {e["importance_ESS"]:.1f} |')
    lines += ['', 'All four aggregate proposal comparisons pass the original 0.2-log-unit and three-combined-SE criteria. '+
       'The native-minus-competing free energies are '+ '; '.join(
           f'**{arm}: {f["beta_F_native_minus_noentry"]:.4f} ± {f["halfwidth_95"]:.4f} kBT**'
           for arm,f in diagnostics['free_energy_intervals'].items()) +
       ' (paired population delta intervals with Student-t₃ 95% half-widths). '+
       'These are observed precision intervals, not missing-mode bounds.', '',
       '## Efficiency and remaining failures', '',
       '| Region | Expanded / baseline importance ESS per physical CPU |', '|---|---:|']
    for item in efficiency:
        lines.append(f'| {item["label"]} | {item["ratio"]:.3f} |')
    lines += ['', 'The two prespecified difficult tails improve approximately 3.6–3.8-fold in this importance-weight metric. '+
       'Whole competing-contact efficiency is lower, so this is not a general speedup. '+
       'No trajectory mixing or equilibrium independent-sample claim follows from these weights.', '',
       'The native-complement ESS remains below 200 in both arms (126 and 166); its largest contributions are '+
       '4.09% and 4.71%. Expanded all-native and R5-native largest contributions also exceed the 2% limit '+
       '(2.13% and 2.72%). No full regional convergence gate passes.', '',
       f'**{len(failed)} of {len(significant)} material stratum comparisons fail.** Material means '+
       'at least 1% of its own contact-region mass in either arm, as fixed before sampling. '+
       'This diagnostic is not a multiple-testing-adjusted hypothesis test. All failures are listed below; '+
       'bins overlap across the separate radial, angular and orthant partitions and must not be summed.', '',
       '| Family / bin | Region | Baseline − expanded log mass | Combined population SE | >3 SE? |',
       '|---|---|---:|---:|---:|']
    for item in failed:
        c = item['comparison']
        lines.append(f'| {item["family"]} {item["bin"]} | {REGIONS[item["region"]]} | '+
                     f'{c["log_left_minus_right"]:.4f} | {c["combined_population_log_delta_SE"]:.4f} | '+
                     ('Yes' if not c['SE_passed'] else 'No')+' |')
    lines += ['', 'Competing orthant 55 is distinct from the targeted native-complement orthant 55. '+
       'The expanded guide finds 74 nonzero competing rows there versus four for the baseline; '+
       'its observed mass is 18.7 times larger, but its ESS is still only 7.16 with a 26.9% largest term. '+
       'This is a concrete remaining coverage issue, not a converged new basin weight.', '',
       'The unchanged deterministic no-contact bound inside R4 is log Q ≤ −9.807904. '+
       'It says nothing about mass outside R4 or the full atomic-wall vessel. '+
       'The previous failed population-size, cloud-intensity and matching-SMC comparisons remain unresolved. '+
       'These small new populations do not replace the earlier independent controls or cure their failed gates.', '',
       '## Integrity and decision', '',
       'Every raw population, summary and manifest hash was rechecked against the completed controller receipt; '+
       'all 131,072 original raw lines remain. Both independent density audits and all eight complete-native '+
       'classifications completed. Maximum reconstructed log-density discrepancy is 1.42×10⁻¹⁴; '+
       'maximum latent-coordinate discrepancy is 1.31×10⁻¹³. Hard-invalid and out-of-region draws retain '+
       'zero weight in unconditional denominators. No historical classifier pass, physical draw or audit '+
       'was rerun, and no old sample entered these fresh estimates. '+
       'The earlier sphere-reference wrapper schema failure remains archived separately; its original zero-activity '+
       'draws were reused, and both actual activities 0 and 2 passed the corrected audit.', '',
       '**The finite-system assembly verdict remains unresolved.** The expanded integration guide is useful '+
       'for further coverage work, but the next action should inspect the newly exposed competing-55 tail '+
       'along with original 55/63/58 strata before declaring any larger independent confirmation. '+
       'The current pilot has ended at its fixed allocation; no extension, vessel calculation or assembly '+
       'campaign is triggered. Integration fitting uses native labels and is not a blind assembly proposal.', '',
       f'- [Frozen protocol]({pilot}/protocol.json)',
       f'- [Complete physical comparison]({pilot}/comparison/analysis.json)',
       f'- [Completion and draw receipts]({pilot}/status.json)',
       '- [Bound review data](summary.json)',
       '- [Vector figure](contact-tail-pilot.svg)', '']
    (out/'report.md').write_text('\n'.join(lines))
    (out/'report_contact_tail_pilot.py').write_bytes(Path(__file__).read_bytes())
    manifest = {x.name: sha(x) for x in sorted(out.iterdir()) if x.is_file()}
    (out/'manifest.json').write_text(json.dumps(dict(schema='completed-review-files-v1', files=manifest), indent=2)+'\n')
    print(json.dumps(dict(out=str(out), complete=True, verified_raw_draws=rows,
                         significant_strata=len(significant), failed_significant_strata=len(failed),
                         efficiency=efficiency), indent=2))


if __name__ == '__main__':
    main()
