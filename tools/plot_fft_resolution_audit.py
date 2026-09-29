#!/usr/bin/env python3
"""Plot a completed retrospective FFT angular-thinning audit, without sampling."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate-budget', type=int, default=75000)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    if not report.get('complete'):
        raise ValueError('requires a completed resolution audit')
    table = report['candidate_table']
    budgets = np.array([r['rotations'] for r in table])
    if args.candidate_budget not in budgets:
        raise ValueError('candidate budget must be one of the completed budgets')
    references = np.array([r['reference_basins'] for r in table])
    if len(set(references)) != 1 or references[0] <= 0:
        raise ValueError('inconsistent reference allocation')
    coverage = 100*np.array([r['nearby_basins'] for r in table])/references
    score = 100*np.array([r['retained_score_fraction'] for r in table])
    chosen = report['recommendation']['budget']
    gate = 100*report['recommendation']['required_fraction']
    source_plan = args.report.parent/'plan.json'
    plan = json.loads(source_plan.read_text()) if source_plan.exists() else None
    params = plan['arguments'] if plan else {}
    translation = params.get('translation_tolerance', 'recorded')
    angle = params.get('angle_tolerance', 'recorded')
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.report, args.out/'source-report.json')
    if source_plan.exists():
        shutil.copy2(source_plan, args.out/'source-plan.json')
    shutil.copy2(__file__, args.out/Path(__file__).name)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    x = np.arange(len(budgets))
    ax.plot(x, coverage, 'o-', color='#217A76', lw=2.3, ms=7,
            label='Reference neighborhoods with a nearby raw peak')
    ax.plot(x, score, 's-', color='#B85A26', lw=2.3, ms=6.5,
            label='Retain ≥90% of full-cache best grid score')
    ax.axhline(gate, color='#B85A26', lw=1., ls=':', alpha=.55)
    ax.text(.015, gate-5, '95% score-retention gate; each score quartile must also retain ≥90%',
            transform=ax.get_yaxis_transform(), color='#865234', fontsize=9)
    i = int(np.flatnonzero(budgets == args.candidate_budget)[0])
    ax.axvline(i, color='#217A76', lw=1., ls=':', alpha=.4)
    ax.annotate(f'{args.candidate_budget//1000:,}k: {table[i]["nearby_basins"]}/{references[i]} raw neighborhoods\n'
                'Candidate budget only; exact repair not established',
                xy=(i, coverage[i]), xytext=(.43, .47), textcoords='axes fraction',
                ha='center', va='center', fontsize=10, color='#175D59',
                bbox=dict(boxstyle='round,pad=.45', fc='#EFF7F6', ec='#AED3CE'),
                arrowprops=dict(arrowstyle='->', color='#217A76', connectionstyle='arc3,rad=.12'))
    if chosen in budgets:
        j = int(np.flatnonzero(budgets == chosen)[0])
        ax.annotate(f'{chosen//1000:,}k: first tested budget passing\nall predeclared score-retention criteria',
                    xy=(j, score[j]), xytext=(.98, .69), textcoords='axes fraction',
                    ha='right', va='center', fontsize=10, color='#8D441C',
                    bbox=dict(boxstyle='round,pad=.45', fc='#FCF3EC', ec='#DDB391'),
                    arrowprops=dict(arrowstyle='->', color='#B85A26', connectionstyle='arc3,rad=-.15'))
    ax.set_xticks(x, [f'{b//1000:,}k' for b in budgets])
    ax.set_xlabel('Orientations retained from the existing 300k grid (nested subsets)')
    ax.set_ylabel('References retained (%)')
    ax.set_ylim(0, 108)
    ax.set_xlim(-.15, len(budgets)-.85)
    ax.grid(axis='y', alpha=.16)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    fig.suptitle('Coarse angular sampling retains candidate neighborhoods sooner than peak scores',
                 x=.075, ha='left', y=.965, fontsize=14, weight='bold')
    fig.text(.075, .905,
             f'{references[0]} frozen blind-reference contacts · {report["cached_spacing_A"]:g} Å cached grid · '
             f'{translation:g} Å / {angle:g}° neighborhoods, including pose inverses', fontsize=10.5)
    fig.text(.075, .075,
             'Retrospective, reference-conditioned candidate search: select the highest grid-score peak near each existing reference.\n'
             'No new FFT or hard-valid recovery was measured. Nearby peaks do not establish basin recovery or native assembly.',
             fontsize=9.5, color='#444444', va='top')
    fig.subplots_adjust(left=.075, right=.975, bottom=.23, top=.86)
    for ext in ('png', 'svg'):
        fig.savefig(args.out/f'fft-resolution.{ext}', dpi=180)
    plt.close(fig)
    metadata = dict(schema='fft-resolution-figure-v1', source_report=str(args.report.resolve()),
                    source_report_sha256=sha(args.report), source_plan_sha256=sha(source_plan) if source_plan.exists() else None,
                    script_sha256=sha(__file__), python=platform.python_version(), matplotlib=matplotlib.__version__,
                    candidate_budget=args.candidate_budget, strict_gate_budget=chosen,
                    files={p.name:sha(p) for p in args.out.glob('fft-resolution.*')},
                    native_labels_used=False, physical_jobs=0,
                    scope='Reference-conditioned raw candidate retention, not end-to-end hard-valid basin recovery.')
    (args.out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
