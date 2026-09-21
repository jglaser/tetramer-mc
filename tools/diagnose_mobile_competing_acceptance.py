#!/usr/bin/env python3
"""Inspect all audited native candidates for the persistent competing body.

Stored Poisson log factors are random auxiliary factors, not exact energies.
This diagnostic does not alter proposals, rerun bath clouds, or estimate a
basin free energy. Selection is the fixed postburn interval and moving body.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    campaign, assessment, out = [p.resolve() for p in (args.campaign, args.assessment, args.out)]
    assert not out.exists()
    manifest = read(campaign/'manifest.json')
    job = next(j for j in manifest['jobs'] if j['id'] == 'mobile3-dispersed-r01-c09')
    path = assessment/'runs'/job['id']/'analysis.json'
    audited = read(path)
    summary = read(assessment/'analysis.json')
    assert summary['complete'] and summary['manifest_sha256'] == sha(campaign/'manifest.json')
    assert summary['terminal_status_sha256'] == sha(campaign/'status.json')
    assert next(r for r in summary['runs'] if r['id'] == job['id']) == audited
    assert audited['passed']
    directory = Path(job['directory'])
    for name, digest in audited['source_sha256'].items():
        assert sha(directory/name) == digest
    selected = [c for c in audited['native_candidates']
                if c['moving'] == 0 and c['sweep'] > manifest['burn_sweeps']]
    by_serial = {r['serial']: r for r in selected}
    assert len(by_serial) == len(selected)
    rows = []
    for serial, line in enumerate((directory/'moves.jsonl').open()):
        if serial not in by_serial:
            continue
        move, candidate = json.loads(line), by_serial[serial]
        assert move['hard_valid'] and move['kind'] == 'global' and move['moving_index'] == 0
        info = move['proposal']
        assert move['sweep'] == candidate['sweep']
        assert info['kernel']+':'+info['branch'] == candidate['source']
        assert move['accepted'] == candidate['accepted']
        assert info['log_reverse_forward'] == candidate['proposal_correction']
        assert move['gate']['log_weight'] == candidate['gate_log_weight']
        alpha = min(0., info['log_reverse_forward']+move['gate']['log_weight'])
        assert math.isclose(alpha, candidate['log_acceptance'], rel_tol=1e-12, abs_tol=1e-12)
        rows.append(dict(candidate, anchor=info['anchor_index'], old_pose=move['old_pose'],
                         proposed_pose=move['proposed_pose'], alpha=math.exp(alpha),
                         native_body_pair_count=len({tuple(k[:2]) for k in candidate['new_registered_keys']})))
    assert len(rows) == len(selected) > 0
    report = dict(complete=True, job=job['id'], moving_body=0,
        burn_sweeps=manifest['burn_sweeps'], candidates=rows,
        counts=dict(total=len(rows), by_kernel=dict(Counter(r['source'] for r in rows)),
                    positive_stored_bath_factor=sum(r['gate_log_weight'] > 0 for r in rows),
                    accepted=sum(r['accepted'] for r in rows),
                    two_native_body_pairs=sum(r['native_body_pair_count'] == 2 for r in rows)),
        maximum_recorded_acceptance_probability=max(r['alpha'] for r in rows),
        sum_recorded_acceptance_probabilities=sum(r['alpha'] for r in rows),
        input_sha256={str(p): sha(p) for p in (path, assessment/'analysis.json', campaign/'manifest.json', directory/'moves.jsonl')},
        scope='All postburn hard-valid new-native candidates for body0, retaining both one- and two-native-edge candidates. Each bath factor is the actual sampled auxiliary log factor, not an exact physical energy or basin weight. Sum of conditional acceptance probabilities describes this finite candidate list; it is not a stationary transition rate or a counterfactual speedup.')
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    colors = {'full-mixture-capture:learned': '#507ca7', 'frozen-posterior:involution': '#cf7339'}
    for source, color in colors.items():
        for count, marker in ((1, 'o'), (2, '^')):
            group = [r for r in rows if r['source'] == source and r['native_body_pair_count'] == count]
            if group:
                ax.scatter([-r['proposal_correction'] for r in group], [r['gate_log_weight'] for r in group],
                    color=color, marker=marker, s=48, alpha=.8,
                    label=f"{'Capture' if source.startswith('full') else 'Transport'} · {count} native pair{'s' if count == 2 else ''}")
    xlim = [min(-r['proposal_correction'] for r in rows)-3, max(-r['proposal_correction'] for r in rows)+3]
    ax.plot(xlim, xlim, '--', color='#555555', label='Unit-acceptance threshold')
    ax.axhline(0., color='#aaaaaa', linewidth=.8)
    ax.set(xlim=xlim, xlabel='Proposal penalty: −log(reverse / forward)',
           ylabel='Stored Poisson bath log factor',
           title='Valid native proposals lose to the reverse-density penalty\n'
                 'Persistent competing attachment · body 0 · sweeps 401–2000')
    ax.legend(fontsize=9, loc='lower right')
    ax.text(.03, .97, f"{len(rows)} valid native candidates; {report['counts']['accepted']} accepted\n"
            f"{report['counts']['positive_stored_bath_factor']} had a positive bath factor",
            transform=ax.transAxes, va='top', fontsize=10)
    out.mkdir(parents=True)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'competing-acceptance.{suffix}', dpi=180)
    plt.close(fig)
    shutil.copy2(__file__, out/Path(__file__).name)
    report['script_sha256'] = sha(__file__)
    (out/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('counts', 'maximum_recorded_acceptance_probability', 'sum_recorded_acceptance_probabilities')}, indent=2))


if __name__ == '__main__':
    main()
