#!/usr/bin/env python3
"""Render the frozen paired-allocation diagnostic; never decode sample arrays.

All numbers come from analysis.json. Other archived files are read only as bytes
for SHA256 authentication. No fitting, geometry, classification, or MC runs occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

PLAN_SHA = 'f4d1b7722b8390bc6c853d8b5996b85caf857ec282edc11da8ff05c2f4d9f79a'
ANALYSIS_SHA = 'e3d83b46749d6b7ae3031d85f7e97777aacfa684f0a782c523ae7436dcf4cf64'
FREEZE_SHA = '984d83c763f0ac49299a88c333e06742304ac01d9e09761d06722e17ebaeb14f'
GROUPS = (
    ('native_inside_R5', 'Native inside R5'),
    ('native_complement', 'Native outside R5'),
    ('contact_no_native_entry', 'Contact without native entry'),
    ('native_complement:orthant:55', 'Native outside R5: orthant 55'),
    ('contact_no_native_entry:orthant:63', 'No-entry contact: orthant 63'),
)
MOMENTS = ('paired_physical', 'two_cloud_noisy')
SCOPE = ('Retrospective reweighting of existing samples. Ratios compare the new '
    'paired-cloud physical-moment fit with the frozen historical noisy-moment fit. '
    'They are second-moment ratios, not sampling speedups or convergence evidence. '
    'Two heldout populations per source are shown separately without error bars. '
    'The full original class, radial, angular and orthant strata are retained.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def authenticate(source):
    source = source.resolve()
    require(sha(source/'plan.json') == PLAN_SHA, 'Wrong frozen plan')
    require(sha(source/'analysis.json') == ANALYSIS_SHA, 'Wrong frozen analysis')
    require(sha(source/'freeze.json') == FREEZE_SHA, 'Wrong final archive inventory')
    checked = {}
    for name, digest in read(source/'freeze.json')['files'].items():
        path = (source/name).resolve()
        require(path.is_relative_to(source), 'Archive path escapes source directory')
        require(sha(path) == digest, 'Archive file differs: '+name)
        checked[name] = digest
    analysis = read(source/'analysis.json')
    require(analysis['complete'] and analysis['plan_sha256'] == PLAN_SHA,
            'Incomplete or unmatched analysis')
    require(analysis['original_attempts_preserved'] == 524288 and
            analysis['fit']['fit_calls'] == 1 and
            analysis['fit']['holdout_rows_read'] == 0 and
            not analysis['historical_fit_repeated'] and not analysis['gate_promotion'],
            'Diagnostic scope or accounting differs')
    require(all(analysis[k] == 0 for k in ('new_physical_draws', 'classifier_calls',
            'geometry_calls', 'audits_replayed', 'protected_validation_rows_read')),
            'Unexpected physical work')
    populations = analysis['populations']
    require(len(populations) == 8 and len({p['seed'] for p in populations}) == 8,
            'Original independent populations differ')
    require({(p['arm'], p['id']) for p in populations} ==
            {(arm, f'r{i:02d}') for arm in ('bank', 'smc') for i in range(4)},
            'Original population identities differ')
    classes = ('native_inside_R5', 'native_complement', 'contact_no_native_entry',
               'residual_valid', 'all_valid')
    expected = set(classes)
    expected.update(f'{name}:{family}:{index}' for name in classes
        for family, size in (('radial', 3), ('angular', 3), ('orthant', 64))
        for index in range(size))
    for population in populations:
        require(population['samples'] == population['attempts_preserved'] == 65536,
                'Attempt denominator changed')
        require(set(population['groups']) == expected, 'Original strata missing')
        require(population['role'] == ('training' if population['id'] in ('r00', 'r01')
                                      else 'heldout'), 'Original split changed')
    return analysis, checked


def row(group, name, identity):
    result = dict(**identity, group=name, attempts=group['attempts'],
                  contributing_rows=group['contributing_rows'],
                  source_importance_ESS=group['likelihood']['weight_ESS'])
    for moment in MOMENTS:
        values = group['second_moments'][moment]
        log_ratio = values['log_warp_to_baseline_ratio']
        if log_ratio is not None:
            direct = values['warped']['log_M2']-values['baseline']['log_M2']
            require(abs(log_ratio-direct) < 1e-12, 'Saved ratio does not match moments')
        result[moment+'_ratio'] = None if log_ratio is None else math.exp(log_ratio)
        for key in ('contribution_ESS', 'largest_fraction'):
            result[moment+'_new_'+key] = values['warped'][key]
    return result


def plot(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    matplotlib.rcParams.update({'font.size': 10, 'svg.hashsalt': 'paired-guide-allocation-v1'})
    lookup = {(r['arm'], r['id'], r['group']): r for r in rows if r['role'] == 'heldout'}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharey=True)
    series = [('bank', 'r02', -.18, '#1769aa', 'o', True),
              ('bank', 'r03', -.06, '#1769aa', 'o', False),
              ('smc', 'r02', .06, '#cf621b', '^', True),
              ('smc', 'r03', .18, '#cf621b', '^', False)]
    for ax, moment, title in zip(axes, MOMENTS,
            ('Physical second moment: independent cloud product',
             'Noisy second moment: implemented cloud average')):
        ax.axvline(1., color='#666666', linewidth=1., linestyle='--')
        for arm, rid, offset, color, marker, filled in series:
            ax.scatter([lookup[(arm, rid, name)][moment+'_ratio'] for name, _ in GROUPS],
                       [i+offset for i in range(len(GROUPS))], s=52, marker=marker,
                       facecolors=color if filled else 'none', edgecolors=color,
                       linewidths=1.3, label=f'{arm} {rid}')
        ax.set_xlim(.93, 1.035)
        ax.set_xticks([.94, .96, .98, 1., 1.02])
        ax.set_title(title, fontsize=10.5, pad=13)
        ax.set_xlabel('New paired fit / historical noisy fit\nSecond-moment ratio; lower is better')
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(GROUPS)), [label for _, label in GROUPS])
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=9, frameon=False)
    fig.suptitle('Changing the allocation objective yields only modest heldout gains', fontsize=13)
    fig.text(.5, .02, 'Four independent heldout populations; no error bars from two populations/source. '
             'Important tail moments remain poorly sampled.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(path/'heldout-moment-ratios.png', dpi=170,
                metadata={'Software': 'report_paired_guide_weights.py'})
    fig.savefig(path/'heldout-moment-ratios.svg', metadata={'Date': None})
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(not out.exists(), 'Fresh output directory required')
    analysis, checked = authenticate(args.source)
    rows = [row(group, name, dict(arm=p['arm'], id=p['id'], role=p['role'], seed=p['seed']))
            for p in analysis['populations'] for name, group in p['groups'].items()]
    aggregate_rows = [row(group, name, dict(aggregate=identity))
        for identity, groups in analysis['aggregates'].items() for name, group in groups.items()]
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'report_paired_guide_weights.py')
    with (out/'all-population-strata.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matplotlib_version = plot(out, rows)
    report = dict(schema='paired-guide-allocation-report-v1', scope=SCOPE,
        analysis_path=str(args.source.resolve()/'analysis.json'), analysis_sha256=ANALYSIS_SHA,
        plan_sha256=PLAN_SHA, source_freeze_sha256=FREEZE_SHA, authenticated_files=checked,
        python=sys.version, matplotlib_version=matplotlib_version,
        source_sha256=sha(__file__), samples_decoded=0, fits_run=0, physical_draws=0,
        fit_iterations=analysis['fit']['optimizer']['iterations'],
        diagnostic_wall_seconds=analysis['wall_seconds'],
        original_attempts=analysis['original_attempts_preserved'],
        training_attempts=analysis['fit']['training']['attempts'],
        training_invalid_zero_rows=analysis['fit']['training']['invalid_zero_rows'],
        selected_groups=[name for name, _ in GROUPS],
        selected_independent_heldout_rows=[r for r in rows if r['role'] == 'heldout'
                                         and r['group'] in dict(GROUPS)],
        aggregate_rows=aggregate_rows)
    write_new(out/'report.json', report)
    lines = [SCOPE, '', 'Physical / noisy second-moment ratios for each independent heldout population:', '',
        '| Region | bank r02 | bank r03 | smc r02 | smc r03 |',
        '|---|---:|---:|---:|---:|']
    selected = {(r['arm'], r['id'], r['group']): r for r in report['selected_independent_heldout_rows']}
    for name, label in GROUPS:
        values = [selected[(arm, rid, name)] for arm in ('bank', 'smc') for rid in ('r02', 'r03')]
        lines.append('| '+label+' | '+' | '.join(
            f"{r['paired_physical_ratio']:.4f} / {r['two_cloud_noisy_ratio']:.4f}" for r in values)+' |')
    lines.extend(['', 'All 355 original strata for all eight populations are preserved in '
        '`all-population-strata.csv`; empty groups remain present with empty ratio cells.', '',
        'The ESS columns describe contributions to the saved importance or second-moment '
        'estimators. They are not Markov-chain effective sample sizes or evidence that '
        'unseen contact modes have been covered.', ''])
    (out/'report.md').write_text('\n'.join(lines))
    write_new(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(output=str(out), original_attempts=report['original_attempts'],
                         population_strata_rows=len(rows), authenticated_files=len(checked))))


if __name__ == '__main__':
    main()
