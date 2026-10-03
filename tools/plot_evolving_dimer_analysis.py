#!/usr/bin/env python3
"""Plot completed frozen dimer summaries; no geometry, new samples, or refits."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.analysis_root.resolve(), args.output.resolve()
    summary = json.loads((root/'summary.json').read_text())
    path = Path(summary['analysis']['path'])
    assert summary['complete'] and summary['passed'] and summary['chains'] == 96
    assert sha(path) == summary['analysis']['sha256']
    assert sha(root/'analysis/manifest.json') == summary['manifest_sha256']
    assert sha(root/'execution-plan.json') == summary['plan_sha256']
    data = json.loads(path.read_text())
    assert data['complete'] and len(data['chains']) == 96
    rows = []
    for chain in data['chains']:
        job, m = chain['job'], chain['metrics']
        row = {k: job[k] for k in ('id', 'context_index', 'initialization', 'arm', 'stream')}
        row['sampler_cpu_seconds'] = m['full_sampler_cpu_seconds']
        for kind in ('fingerprint', 'patch', 'partner'):
            row[kind+'_ess_per_cpu'] = m[kind+'_ess']['apparent_ess_per_sampling_CPU_second']
        for kind, key in [('fingerprint', 'fingerprint'), ('partner', 'partner_environments')]:
            row[kind+'_passages'] = m[key]['completed_passages']
            row[kind+'_returns'] = len(m[key]['completed_returns'])
            for metric in ('passages', 'returns'):
                row[kind+'_'+metric+'_per_1000_cpu'] = 1000*row[kind+'_'+metric]/row['sampler_cpu_seconds']
        row['internal_contact_fraction'] = m['internal_contact_fraction']
        row.update(chain['counts'])
        rows.append(row)
    rows.sort(key=lambda r: r['id'])
    assert {r['id'] for r in rows} == set(range(96))
    arms = ('local', 'unguided', 'm4')
    starts = ('source', 'proposal_prepared')
    identities = {(r['context_index'], r['initialization'], r['arm'], r['stream']) for r in rows}
    assert identities == {(c, s, a, i) for c in range(4) for s in starts for a in arms for i in range(4)}
    agreement = [r for r in data['comparisons']['descriptive_paired_comparisons']
                 if r['left']['arm'] == r['right']['arm']]
    assert len(agreement) == 48
    output.mkdir(parents=True, exist_ok=False)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output/'chains.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output/'initialization-agreement.json').write_text(json.dumps(agreement, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(local='#2563a6', unguided='#ad7930', m4='#168474')
    names = dict(local='Local', unguided='Local + unguided dimer', m4='Local + guided dimer (m = 4)')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics = [
        ('fingerprint_ess_per_cpu', 'Whole contact fingerprint', True),
        ('patch_ess_per_cpu', 'Surface-patch contacts', True),
        ('partner_ess_per_cpu', 'Contact partners', True),
        ('partner_passages_per_1000_cpu', 'Changes of partner-contact set', False),
        ('partner_returns_per_1000_cpu', 'Returns to a partner-contact set', False),
    ]
    labels = [f'{c+1}\n{s}' for c in range(4) for s in ('S', 'P')]
    for ax, (metric, title, logarithmic) in zip(axes.flat, metrics):
        nonnull = [r[metric] for r in rows if r[metric] is not None and r[metric] > 0]
        null_level = min(nonnull)*.45 if logarithmic else None
        for c in range(4):
            for j, start in enumerate(starts):
                for ai, arm in enumerate(arms):
                    group = sorted((r for r in rows if (r['context_index'], r['initialization'], r['arm']) == (c, start, arm)), key=lambda r:r['stream'])
                    x = c*2+j+(ai-1)*.23
                    values = []
                    for r in group:
                        value = r[metric]
                        if value is None:
                            ax.scatter(x+(r['stream']-1.5)*.026, null_level, marker='x', c=colors[arm], s=23, linewidths=1)
                        else:
                            assert math.isfinite(value)
                            values.append(value)
                            ax.scatter(x+(r['stream']-1.5)*.026, value, c=colors[arm], s=24, alpha=.8, edgecolors='white', linewidths=.4)
                    if values:
                        ax.plot([x-.075,x+.075], [statistics.median(values)]*2, color=colors[arm], linewidth=2.2)
        if logarithmic:
            ax.set_yscale('log')
            ax.set_ylabel('Apparent ESS / sampler CPU second')
            ax.axhline(null_level*1.35, color='#999999', ls=':', lw=.7)
            ax.text(.01, .1, '× below dotted line = constant; ESS undefined', transform=ax.transAxes, fontsize=8, color='#555555')
        else:
            ax.set_yscale('symlog', linthresh=1.)
            ax.set_ylabel('Observed events / 1,000 sampler CPU seconds')
            ax.set_ylim(bottom=-.05)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_xticks(range(8), labels)
        for v in (1.5,3.5,5.5): ax.axvline(v, color='#dddddd', lw=.8)
        ax.grid(axis='y', color='#eeeeee', zorder=0)
        ax.set_xlabel('Conditional context; S = source, P = proposal-prepared start')
    ax = axes.flat[-1]
    for c in range(4):
        for ai, arm in enumerate(arms):
            group = sorted((r for r in agreement if r['context_index']==c and r['left']['arm']==arm), key=lambda r:r['left']['stream'])
            vals = []
            for r in group:
                x = c+(ai-1)*.23+(r['left']['stream']-1.5)*.032
                value = r['partner_occupancy_max_difference']; vals.append(value)
                ax.scatter(x, value, c=colors[arm], s=30, alpha=.8, edgecolors='white', linewidths=.4)
            x=c+(ai-1)*.23
            ax.plot([x-.075,x+.075], [statistics.median(vals)]*2, color=colors[arm], lw=2.2)
    ax.set(title='Do the two starts agree?', ylabel='Largest partner-occupancy difference', xlabel='Conditional context')
    ax.set_xticks(range(4), ['1\n27–132','2\n32–110','3\n9–24','4\n11–246'])
    ax.set_ylim(-.03,1.07); ax.grid(axis='y', color='#eeeeee')
    ax.text(.03,.05,'0 = agreement; 1 = completely different for at least one contact', transform=ax.transAxes, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle('Guided dimer moves: cheaper than unguided, but contact equilibration remains unresolved', x=.05, ha='left', fontsize=16)
    fig.legend(handles=[Line2D([0],[0],marker='o',lw=2,color=colors[a],label=names[a]) for a in arms],
               loc='upper left',bbox_to_anchor=(.045,.955),ncol=3,frameon=False)
    fig.text(.05,.018,'96 chains; 4 independent streams per group (dots), median across defined values (bars). No trajectory concatenation.\n'
             '2 mobile tetramers + 262 fixed spectators; depletant radius = 1.4 Å, z = 0.0275 Å⁻³. Partner-set changes/returns are threshold observations, not proven basin exchange.\n'
             'ESS is a finite-record descriptor, not an equilibrium certification. Sampler CPU includes warmup and rejected/null moves; geometry observer CPU is separate.', fontsize=9)
    fig.subplots_adjust(top=.87,bottom=.18,wspace=.3,hspace=.47,left=.06,right=.985)
    fig.savefig(output/'contact-efficiency.png',dpi=160)
    fig.savefig(output/'contact-efficiency.svg')
    plt.close(fig)
    receipt=dict(complete=True, source_sha256=sha(__file__), analysis_sha256=sha(path),
        analysis_manifest_sha256=sha(root/'analysis/manifest.json'), execution_plan_sha256=sha(root/'execution-plan.json'),
        new_geometry_queries=0, new_sampling_attempts=0, new_fits=0, trajectories_concatenated=False,
        scope='Descriptive visualization of previously frozen complete 96-chain analysis. No new hypothesis test or equilibrium conclusion.',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(complete=True,output=str(output),chains=len(rows))))


if __name__ == '__main__':
    main()
