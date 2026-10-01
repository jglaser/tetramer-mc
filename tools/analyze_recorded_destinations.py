#!/usr/bin/env python3
"""Summarize every recorded destination; no physical sampling or filtering."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def stats(xs):
    if not xs:
        return {"count": 0}
    a = np.asarray(xs, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "q10": float(np.quantile(a, .1)), "q90": float(np.quantile(a, .9))}


def summarize(rows):
    learned = [r for r in rows if r['branch'] == 'involution']
    subsets = {'all': rows, 'learned': learned,
               'source_norm_le_6': [r for r in learned if r['source_norm'] <= 6],
               'source_norm_gt_6': [r for r in learned if r['source_norm'] > 6]}
    out = {}
    for label, rs in subsets.items():
        d = {'attempted': len(rs), 'accepted': sum(r['accepted'] for r in rs),
             'hard_valid': sum(r.get('candidate', {}).get('hard_valid', False) for r in rs),
             'two_current_neighbors': sum(len(r['old']['contact_bodies']) >= 2 for r in rs),
             'two_current_neighbors_in_pool': sum(len(set(r['old']['contact_bodies']) & set(r.get('anchor_pool', []))) >= 2 for r in rs),
             'any_fused_available': sum((r.get('fused_available') or 0) > 0 for r in rs)}
        ms = [r for r in rs if 'mean' in r]
        d['has_mean'] = len(ms)
        d['mean_sample_validity'] = dict(Counter(f"mean_{r['mean']['hard_valid']}_sample_{r['candidate']['hard_valid']}" for r in ms))
        d['sample_clash_types'] = dict(Counter(
            f"target_anchor_{bool(set(r['candidate']['hard_clash_bodies']) & set(r.get('anchor_pool', [])))}_other_{bool(set(r['candidate']['hard_clash_bodies']) - set(r.get('anchor_pool', [])))}_wall_{not r['candidate']['wall_valid']}"
            for r in rs if 'candidate' in r and not r['candidate']['hard_valid']))
        d['target_fused'] = summarize_validity([r for r in ms if r['target_label'].get('kind') == 'fused'])
        d['target_single'] = summarize_validity([r for r in ms if r['target_label'].get('kind') != 'fused'])
        valid = [r for r in rs if r.get('candidate', {}).get('hard_valid')]
        env = {}
        for r in valid:
            old, new = set(r['old']['contact_bodies']), set(r['candidate']['contact_bodies'])
            key = f"+{len(new-old)}/-{len(old-new)}"
            env.setdefault(key, []).append(r)
        d['contact_change'] = {k: {'count': len(v), 'accepted': sum(r['accepted'] for r in v),
                                       'gate': stats([r['gate']['log_weight'] for r in v if r['gate']]),
                                       'log_acceptance': stats([r['log_acceptance'] for r in v if r['log_acceptance'] is not None])}
                               for k, v in env.items()}
        d['candidate_motion'] = {'translation_a': stats([r['translation_a'] for r in rs if 'translation_a' in r]),
                                 'rotation_degrees': stats([r['rotation_degrees'] for r in rs if 'rotation_degrees' in r])}
        d['accepted_motion'] = {'translation_a': stats([r['translation_a'] for r in rs if r['accepted']]),
                                'rotation_degrees': stats([r['rotation_degrees'] for r in rs if r['accepted']])}
        d['selected_mean_rotation_relative_anchor_degrees'] = stats([r['target_mean_relative_rotation_degrees'] for r in ms if 'target_mean_relative_rotation_degrees' in r])
        d['contracted_target_displacement'] = {}
        for factor in (0., .125, .25, .5, 1.):
            counter = Counter()
            for r in ms:
                old = set(r['old']['contact_bodies'])
                for v in r.get('contracted_target_displacement', []):
                    if v['factor'] != factor:
                        continue
                    counter['attempted'] += 1
                    counter['hard_valid'] += v['hard_valid']
                    if v['hard_valid']:
                        new = set(v['contact_bodies'])
                        counter['contact_change'] += (new != old)
                        counter['contact_exchange'] += bool(new - old) and bool(old - new)
                        counter['lost_any_old_contact'] += bool(old - new)
                        counter['retained_all_old_contacts'] += old <= new
                        counter['gained_any_contact'] += bool(new - old)
                        counter['has_two_or_more_contacts'] += len(new) >= 2
            d['contracted_target_displacement'][str(factor)] = dict(counter)
        out[label] = d
    return out


def summarize_validity(rs):
    return {'attempted': len(rs), 'mean_valid': sum(r['mean']['hard_valid'] for r in rs),
            'sample_valid': sum(r['candidate']['hard_valid'] for r in rs), 'accepted': sum(r['accepted'] for r in rs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = {'scope': 'Posthoc deterministic geometry replay of every archived attempt, retaining all nulls and hard failures; reset-phase conditional diagnostics only. Source norm threshold is descriptive. Realized Poisson gates are not free-energy estimates.',
           'populations': {}, 'arms': {}, 'sources': {}}
    arms = {}
    for path in sorted(args.directory.glob('*-p*/audit.jsonl')):
        raw = path.read_bytes()
        rows = [json.loads(x) for x in raw.splitlines()]
        out['sources'][str(path)] = hashlib.sha256(raw).hexdigest()
        out['populations'][path.parent.name] = summarize(rows)
        arm = path.parent.name.rsplit('-p', 1)[0]
        arms.setdefault(arm, []).extend(rows)
    for arm, rows in arms.items():
        out['arms'][arm] = summarize(rows)
        pops = [v for k, v in out['populations'].items() if k.startswith(arm+'-p')]
        out['arms'][arm]['between_population_descriptive_rates'] = {}
        for metric in ('hard_valid', 'accepted', 'two_current_neighbors_in_pool'):
            x = [p['all'][metric]/p['all']['attempted'] for p in pops]
            out['arms'][arm]['between_population_descriptive_rates'][metric] = {'values': x, 'mean': float(np.mean(x)), 'standard_error': float(np.std(x, ddof=1)/np.sqrt(len(x)))}
    (args.directory/'analysis.json').write_text(json.dumps(out, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {'unfused': '#3366aa', 'fused': '#cc7733'}
    for arm, armrows in out['arms'].items():
        for ax, metric in zip(axes, ('hard_valid', 'contact_exchange')):
            factors = np.array([.0, .125, .25, .5, 1.])
            pops = [v['source_norm_le_6'] for k, v in out['populations'].items() if k.startswith(arm+'-p')]
            values = np.array([[p['contracted_target_displacement'][str(f)][metric] / p['attempted'] for f in factors] for p in pops])
            ax.errorbar(factors, 100 * values.mean(axis=0), yerr=100 * values.std(axis=0, ddof=1)/np.sqrt(len(pops)), color=colors[arm], marker='o', label=arm.capitalize(), capsize=3)
            ax.grid(alpha=.2)
            ax.set_xlabel('Fraction of recorded target latent displacement')
            ax.set_ylabel('Percent of represented-source attempts')
            ax.set_xlim(-.035, 1.035)
        axes[0].legend(frameon=False)
    axes[0].set_title('Hard-valid destinations')
    axes[1].set_title('Hard-valid neighbor exchanges')
    fig.suptitle('Passive geometry replay: useful destinations exist near rejected candidates', fontsize=13)
    fig.text(.5, .005, 'Same archived candidates; source norm ≤6. Four populations/arm, bars ±1 SE.\nCounterfactual contractions are not acceptance rates or a validated Markov proposal.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .94])
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.directory/f'destination-width.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps({a: {s: {k: v for k, v in r[s].items() if k in ('attempted', 'accepted', 'hard_valid', 'mean_sample_validity', 'target_single', 'target_fused', 'two_current_neighbors', 'two_current_neighbors_in_pool', 'sample_clash_types')} for s in ('all','source_norm_le_6')} for a, r in out['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
