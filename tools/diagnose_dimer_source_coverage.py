#!/usr/bin/env python3
"""Describe the learned/uniform mixture contributions at saved source poses.

No density is reevaluated, and no pose, cloud, or acceptance decision is drawn.
This is an exploratory inspection of already audited scores, not a coverage
integral or an estimate of equilibrium basin weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_SHA = '0dfcf4ddd6d2a8df568688fe80e8ef41c0558dc8835642f9624701f3bdfddae6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(value):
    if value == '-inf':
        return -math.inf
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Invalid saved log score')
    return value


def analyze(cache):
    if sha(cache) != CACHE_SHA:
        raise ValueError('Changed authenticated baseline cache')
    groups = {}
    attempts = candidates = 0
    for line in cache.open():
        raw = json.loads(line)
        row = raw['cached']
        attempts += 1
        if row['candidate'] is None:
            continue
        candidates += 1
        edge = row['candidate']['diagnostics']['old_edges'][1]
        full, learned, uniform = [score(edge[k]) for k in ('log_full','log_learned','log_uniform')]
        lp, up = math.log(.5)+learned-full, math.log(.5)+uniform-full
        if not math.isfinite(full) or abs(math.exp(lp)+math.exp(up)-1) > 1e-10:
            raise ValueError('Scores do not form the declared half mixture')
        key = row['atlas'], row['case']['root'], row['case']['child']
        value = groups.setdefault(key, dict(atlas=key[0], root=key[1], child=key[2],
            candidate_rows=0, contexts=set(), old_internal_full_log_density=full,
            learned_log_posterior=lp, uniform_log_posterior=up))
        for name, observed in [('old_internal_full_log_density',full),('learned_log_posterior',lp),('uniform_log_posterior',up)]:
            expected=value[name]
            if expected != observed and not math.isclose(expected,observed,rel_tol=2e-12,abs_tol=2e-10):
                raise ValueError('Same source pair acquired different saved scores')
        value['candidate_rows'] += 1
        value['contexts'].add(row['case']['name'])
    if attempts != 768 or candidates != 511 or len(groups) != 12:
        raise ValueError('Incomplete archived source coverage')
    rows=[]
    for key,value in sorted(groups.items()):
        value['contexts']=sorted(value['contexts'])
        value['learned_posterior']=math.exp(value['learned_log_posterior'])
        value['uniform_posterior']=math.exp(value['uniform_log_posterior'])
        for name in ('learned_log_posterior','uniform_log_posterior'):
            if value[name] == -math.inf:
                value[name]='-inf'
        rows.append(value)
    return dict(schema='saved-dimer-source-mixture-coverage-v1',complete=True,exploratory=True,
        input=dict(path=str(cache.resolve()),sha256=CACHE_SHA),script_sha256=sha(__file__),
        outer_attempts=attempts,candidates=candidates,source_pairs_per_atlas=4,
        formula='p_learned=.5*exp(log_learned-log_full); p_uniform=.5*exp(log_uniform-log_full)',
        uniform_probability=.5,rows=rows,new_draws=0,new_density_evaluations=0,
        limitations=['Four prescribed source pairs, not independent equilibrium configurations or coverage of other basins.',
            'Repeated candidate rows verify the same saved source score; they are not extra independent observations.',
            'Underflowed posterior values do not prove mathematically zero Gaussian support; log scores are retained.',
            'No acceptance rescore, overlap volume, physical free energy, or kernel-width benefit is inferred.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'results/auxiliary-overlap-physical-20261003/common/baseline-physical.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.cache)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=True,source_rows=len(result['rows']),output_sha256=sha(args.output))))
