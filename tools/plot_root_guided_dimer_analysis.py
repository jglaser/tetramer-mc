#!/usr/bin/env python3
"""Plot a completed, audited matched comparison; no trajectory or geometry reads."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


ARMS = ('m4', 'root_m4')
STARTS = ('source', 'proposal_prepared')


def reduce(data):
    require(data['schema'] == 'root-guided-dimer-analysis-v1' and data['complete'] is True
            and data['new_chains'] == data['reused_control_chains'] == 32
            and data['new_geometry_endpoints'] == 147488
            and data['new_physical_draws'] == 0 and data['native_observer'] is False,
            'Completed matched analysis required')
    rows = []
    for chain in data['chains']:
        job, metrics = chain['job'], chain['metrics']
        require(job['arm'] in ARMS and chain['reused_control'] is (job['arm'] == 'm4'),
                'Wrong control identity')
        require(metrics['production_samples'] == 4096, 'Missing rejected residence or production endpoints')
        cpu = metrics['full_sampler_cpu_seconds']
        require(math.isfinite(cpu) and cpu > 0, 'Invalid sampler CPU')
        external = metrics['external_only']
        row = {k: job[k] for k in ('id', 'context_index', 'arm', 'initialization', 'stream')}
        row['sampler_cpu_seconds'] = cpu
        for kind, values in [('external', external['ess']), ('patch', metrics['patch_ess']),
                             ('fingerprint', metrics['fingerprint_ess'])]:
            value = values['apparent_ess_per_sampling_CPU_second']
            require(value is None or (math.isfinite(value) and value > 0), 'Invalid or fabricated ESS')
            row[kind+'_ess_per_cpu'] = value
        row['external_contact_fraction'] = external['any_contact_fraction']
        row['external_passages'] = external['environments']['completed_passages']
        row['external_returns'] = len(external['environments']['completed_returns'])
        for name in ('passages', 'returns'):
            row['external_'+name+'_per_1000_cpu'] = 1000*row['external_'+name]/cpu
        rows.append(row)
    identities = {(r['context_index'], r['arm'], r['initialization'], r['stream']) for r in rows}
    require(len(rows) == len(identities) == 64 and identities ==
            {(c, a, s, i) for c in range(4) for a in ARMS for s in STARTS for i in range(4)},
            'Changed matched inventory')
    agreement = [p for p in data['comparisons']['descriptive_paired_comparisons']
                 if p['left']['arm'] == p['right']['arm']]
    require(len(agreement) == 32, 'Missing initialization comparisons')
    pairs = set()
    for p in agreement:
        a, b = p['left'], p['right']
        require(a['context_index'] == b['context_index'] == p['context_index']
                and a['stream'] == b['stream'] and {a['initialization'], b['initialization']} == set(STARTS),
                'Unmatched initializations')
        pairs.add((p['context_index'], a['arm'], a['stream']))
        difference = p['external_only']['edge_occupancy_max_difference']
        require(math.isfinite(difference) and 0 <= difference <= 1, 'Invalid occupancy comparison')
    require(len(pairs) == 32, 'Duplicate initialization comparisons')
    return rows, agreement


def render(rows, agreement, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(m4='#2876a5', root_m4='#c4682e')
    labels = dict(m4='Internal guide', root_m4='Anchor + internal guides')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    definitions = [('external_ess_per_cpu', 'External-contact edges', True),
        ('patch_ess_per_cpu', 'Surface-patch contacts', True),
        ('fingerprint_ess_per_cpu', 'Whole contact fingerprint', True),
        ('external_passages_per_1000_cpu', 'External contact-set changes', False),
        ('external_returns_per_1000_cpu', 'External contact-set returns', False)]
    for ax, (key, title, logarithmic) in zip(axes.flat, definitions):
        positive = [r[key] for r in rows if r[key] is not None and r[key] > 0]
        null_level = .4*min(positive) if positive else 1.
        for context in range(4):
            for si, start in enumerate(STARTS):
                for ai, arm in enumerate(ARMS):
                    group = [r for r in rows if (r['context_index'], r['initialization'], r['arm']) ==
                             (context, start, arm)]
                    x = 2*context+si+(ai-.5)*.3
                    values = []
                    for row in group:
                        value = row[key]
                        ax.scatter(x+(row['stream']-1.5)*.035, null_level if value is None else value,
                            marker='x' if value is None else 'o', color=colors[arm], s=26, alpha=.8)
                        if value is not None:
                            values.append(value)
                    if values:
                        ax.plot([x-.065, x+.065], [statistics.median(values)]*2, color=colors[arm], lw=2)
        if logarithmic:
            ax.set_yscale('log')
            ax.set_ylabel('Apparent ESS / full sampler CPU second')
            ax.axhline(1.35*null_level, color='#999999', ls=':', lw=.7)
            ax.text(.01, .03, '× below dotted line: constant; ESS undefined', transform=ax.transAxes, fontsize=8)
        else:
            ax.set_yscale('symlog', linthresh=1.)
            ax.set_ylim(bottom=-.05)
            ax.set_ylabel('Observed events / 1,000 sampler CPU seconds')
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_xticks(range(8), [f'{c+1}\n{s}' for c in range(4) for s in ('S', 'P')])
        ax.set_xlabel('Conditional context; S = source, P = displaced start')
        for x in (1.5, 3.5, 5.5):
            ax.axvline(x, color='#dddddd', lw=.8)
    ax = axes.flat[-1]
    for context in range(4):
        for ai, arm in enumerate(ARMS):
            group = [p for p in agreement if p['context_index'] == context and p['left']['arm'] == arm]
            values = [p['external_only']['edge_occupancy_max_difference'] for p in group]
            x = context+(ai-.5)*.3
            for p, value in zip(group, values):
                ax.scatter(x+(p['left']['stream']-1.5)*.035, value, color=colors[arm], s=26)
            ax.plot([x-.065, x+.065], [statistics.median(values)]*2, color=colors[arm], lw=2)
    ax.set(title='Agreement between the two starts', ylabel='Largest external-edge occupancy difference',
           xlabel='Conditional context', ylim=(-.03, 1.05))
    ax.set_xticks(range(4), ['1\n27–132', '2\n32–110', '3\n9–24', '4\n11–246'])
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#eeeeee')
    fig.suptitle('Matched anchor-guided dimer comparison', x=.05, ha='left', fontsize=17)
    fig.legend(handles=[Line2D([0], [0], marker='o', color=colors[a], label=labels[a]) for a in ARMS],
        loc='upper left', bbox_to_anchor=(.045, .951), frameon=False, ncol=2)
    fig.text(.05, .025, '32 new chains + 32 reused controls; four streams per group (dots), medians of defined values (bars). Both arms include identical local schedules.\n'
        '2 mobile tetramers + 262 fixed spectators; radius 1.4 Å, activity 0.0275 Å⁻³. External sets exclude the moving pair’s internal contact.\n'
        'Set changes can be contact gain/loss. Finite-record ESS and occupancy agreement do not certify equilibrium, native registry or finite-system stability.', fontsize=9)
    fig.subplots_adjust(left=.06, right=.985, bottom=.18, top=.86, hspace=.47, wspace=.32)
    fig.savefig(output/'contact-efficiency.png', dpi=160)
    fig.savefig(output/'contact-efficiency.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root, output = args.analysis_root.resolve(), args.output.resolve()
    summary = read(root/'summary.json')
    require(summary['schema'] == 'root-guided-dimer-observer-completion-v1'
            and summary['complete'] is True and summary['passed'] is True, 'Passed independent analysis required')
    path, manifest = Path(summary['analysis']['path']), Path(summary['manifest']['path'])
    require(sha(path) == summary['analysis']['sha256'] and sha(manifest) == summary['manifest']['sha256']
            and sha(root/'execution-plan.json') == summary['plan_sha256'], 'Analysis binding differs')
    m = read(manifest)
    require(m['complete'] is True and m['files']['analysis.json'] == sha(path), 'Manifest differs')
    rows, agreement = reduce(read(path)); output.mkdir(parents=True, exist_ok=False)
    with (output/'chains.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'initialization-agreement.json').write_text(json.dumps(agreement, indent=2)+'\n')
    render(rows, agreement, output)
    receipt = dict(complete=True, source_sha256=sha(__file__), summary_sha256=sha(root/'summary.json'),
        analysis_sha256=sha(path), manifest_sha256=sha(manifest), plan_sha256=summary['plan_sha256'],
        chains=64, new_geometry_queries=0, new_sampling_attempts=0, new_fits=0,
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(complete=True, output=str(output), chains=64)))


if __name__ == '__main__':
    main()
