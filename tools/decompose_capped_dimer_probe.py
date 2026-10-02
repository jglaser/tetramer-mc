#!/usr/bin/env python3
"""Exploratory decomposition of an already audited passive screen; no new draws.

Only the maximum-cap stream is used, because shorter caps repeat its prefixes.
Counts are from first-success-stopped proposal sequences in fixed contexts,
not an equilibrium sample or a new predeclared statistical test.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def classify(f):
    internal_core=bool(f['internal_core_overlap'])
    internal_contact=bool(f['internal_exclusion_contact'])
    pair_ok=not internal_core and internal_contact
    root_ok=not f['spectator_core_collisions'][0] and bool(f['wall_valid'][0])
    child_ok=not f['spectator_core_collisions'][1] and bool(f['wall_valid'][1])
    stage=('internal_pair_failed' if not pair_ok else 'root_fixed_failed_after_pair' if not root_ok
           else 'child_fixed_failed_after_pair_and_root' if not child_ok else 'feasible')
    return dict(raw_trials=1,internal_core_overlap=int(internal_core),internal_no_contact=int(not internal_contact),
                internal_pair_valid=int(pair_ok),root_fixed_hard_failed=int(not root_ok),child_fixed_hard_failed=int(not child_ok),
                **{stage:1})


def analyze(base):
    analysis=read(base/'analysis.json');assert analysis['complete'] and analysis['passed'] and not analysis['failures']
    ledger=base/'execution/attempts.jsonl'
    assert sha(ledger)==analysis['input_hashes']['attempts.jsonl']
    groups=defaultdict(Counter);contexts=defaultdict(Counter);branches=defaultdict(Counter);slots=set();raw=0
    with ledger.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['cap']!=32:continue
            key=(row['atlas'],row['case']['name'],row['attempt']);assert key not in slots;slots.add(key)
            assert row['status']=='completed'
            for trial in row['outcome']['trials']:
                counts=classify(trial['feasibility']);raw+=1
                groups[row['atlas']].update(counts);contexts[key[:2]].update(counts)
                child_branch=trial['draw']['edges'][1]['branch'];branches[(row['atlas'],child_branch)].update(counts)
    assert len(slots)==768 and raw==analysis['unique_raw_trials_independently_reconstructed']
    for counts in groups.values():
        assert counts['raw_trials']==sum(counts[k] for k in ['internal_pair_failed','root_fixed_failed_after_pair','child_fixed_failed_after_pair_and_root','feasible'])
    return dict(schema='capped-dimer-saved-row-exploratory-decomposition-v1',complete=True,
        computed_utc=datetime.now(timezone.utc).isoformat(),allocation_changed=False,new_draws=0,
        exploratory=True,selection='All cap32 raw prefixes; shorter caps excluded only as previously verified bitwise duplicates.',
        input_hashes={'analysis.json':sha(base/'analysis.json'),'execution/attempts.jsonl':sha(ledger),'script':sha(__file__)},
        unique_source_seed_slots=len(slots),unique_raw_trials=raw,
        predicates={'internal_pair_valid':'not internal_core_overlap AND internal_exclusion_contact',
                    'root_fixed_hard_valid':'no root spectator_core_collisions AND root wall_valid',
                    'child_fixed_hard_valid':'no child spectator_core_collisions AND child wall_valid'},
        exclusive_order=['internal_pair_failed','root_fixed_failed_after_pair','child_fixed_failed_after_pair_and_root','feasible'],
        by_atlas={name:dict(counts) for name,counts in groups.items()},
        by_context=[dict(atlas=key[0],case=key[1],counts=dict(counts)) for key,counts in contexts.items()],
        by_child_branch=[dict(atlas=key[0],child_branch=key[1],counts=dict(counts)) for key,counts in branches.items()],
        limitations=['Posthoc description of frozen saved predicates; not a confirmatory test or measured speedup.',
                     'Nonexclusive failure counts overlap; the separately reported ordered partition is disjoint.',
                     'Raw sequences stop on first feasible endpoint; pooled contexts have different exposure and are not independent equilibrium populations.',
                     'World endpoint predicates support cost planning; body-frame prefilter equivalence and a staged generator still require implementation validation.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as output:
        report=analyze(args.base);json.dump(report,output,indent=2,sort_keys=True,allow_nan=False);output.write('\n')
    print(json.dumps(dict(complete=report['complete'],unique_raw_trials=report['unique_raw_trials'],by_atlas=report['by_atlas']),indent=2))
