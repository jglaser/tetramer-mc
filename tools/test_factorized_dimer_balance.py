#!/usr/bin/env python3
"""Exact rational finite-state validation of independently capped dimer edges.

Enumerates finite random tapes, including failed trials and null mass. This is
not a protein sampler, a numerical simulation, or an instantiated SE(3) proof.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import unittest


ZERO, ONE = F(0), F(1)
ROOT = Path(__file__).resolve().parents[1]


def normalized(weights):
    assert weights and all(w >= 0 for w in weights)
    assert sum(weights, ZERO) == ONE


def retry_factor(p, cap):
    """Closed expression under test; enumeration below does not call this."""
    return sum(((ONE-p)**j for j in range(cap)), ZERO)


def edge_tapes(weights, allowed, cap):
    """Enumerate all cap-length tapes, then integrate unused suffixes exactly.

    Keys record first success (or None) and the actually consumed prefix. A
    length-zero tape is a probability-one exhausted edge block.
    """
    normalized(weights)
    assert type(cap) is int and cap >= 0
    outcomes = defaultdict(F)
    for tape in product(range(len(weights)), repeat=cap):
        probability = ONE
        for item in tape:
            probability *= weights[item]
        index = next((j for j, item in enumerate(tape) if item in allowed), None)
        selected = None if index is None else tape[index]
        prefix = tape if index is None else tape[:index+1]
        outcomes[(selected, prefix)] += probability
    assert sum(outcomes.values(), ZERO) == ONE
    return dict(outcomes)


def joint_trial(weights0, weights1, allowed0, allowed1, final_mask,
                cap0, cap1, order='root_first', short_circuit=True):
    """Marginalize every edge-block trace in one whole joint attempt.

    Failure of the first stage can skip or consume the second stage. The
    policy is fixed independently of old state and reported separately.
    """
    assert order in ('root_first', 'child_first')
    root = edge_tapes(weights0, allowed0, cap0)
    child = edge_tapes(weights1, allowed1, cap1)
    first, second = (root, child) if order == 'root_first' else (child, root)
    law = defaultdict(F)
    skipped = ZERO
    for (a, _), pa in first.items():
        if a is None and short_circuit:
            law[None] += pa
            skipped += pa
            continue
        for (b, _), pb in second.items():
            pair = (a, b) if order == 'root_first' else (b, a)
            endpoint = pair if a is not None and b is not None and pair in final_mask else None
            law[endpoint] += pa*pb
    assert sum(law.values(), ZERO) == ONE
    return dict(law), skipped


def outer_tapes(trial_law, cap):
    """Enumerate all outer tapes after exact marginalization of each trial.

    The first non-null joint endpoint terminates the outer attempt. All failed
    edge/joint histories are contained in the independently enumerated trial
    law's null mass. No closed retry formula is used here.
    """
    assert sum(trial_law.values(), ZERO) == ONE
    assert type(cap) is int and cap >= 0
    law = defaultdict(F)
    choices = tuple(trial_law)
    for tape in product(choices, repeat=cap):
        probability = ONE
        for item in tape:
            probability *= trial_law[item]
        endpoint = next((item for item in tape if item is not None), None)
        law[endpoint] += probability
    assert sum(law.values(), ZERO) == ONE
    return dict(law)


def mh_acceptance(numerator, denominator):
    assert numerator >= 0 and denominator >= 0
    if denominator == 0:
        return ONE if numerator > 0 else ZERO
    return min(ONE, numerator/denominator)


def physical_kernel(states, domain, candidate_law, guide, physical,
                    corrected_guide=None):
    """One decision after the first feasible endpoint, with all rejection mass.

    Source states outside the conditioning domain receive identity. The default
    correction uses the original complete product guide, not an estimated
    conditioned normalizer. A supplied corrected_guide is used only to expose
    the missing factor in the intentionally invalid retained-root variant.
    """
    correction = guide if corrected_guide is None else corrected_guide
    kernel = {x: {y: ZERO for y in states} for x in states}
    for x in states:
        if x not in domain:
            kernel[x][x] = ONE
            continue
        kernel[x][x] += candidate_law.get(None, ZERO)
        for y, q in candidate_law.items():
            if y is None:
                continue
            assert y in domain
            alpha = mh_acceptance(physical[y]*correction[x], physical[x]*correction[y])
            kernel[x][y] += q*alpha
            kernel[x][x] += q*(ONE-alpha)
    return kernel


def check_balance(states, kernel, physical):
    assert all(sum(kernel[x].values(), ZERO) == ONE for x in states)
    assert all(kernel[x][y] >= 0 for x in states for y in states)
    assert all(physical[x]*kernel[x][y] == physical[y]*kernel[y][x]
               for x in states for y in states)
    assert all(sum((physical[x]*kernel[x][y] for x in states), ZERO) == physical[y]
               for y in states)


def root_retaining_trial(weights0, weights1, allowed0, allowed1, final_mask,
                         cap0, cap1, child_retries):
    """Intentionally different generator: retain root after final joint failure.

    This is state-independent but its endpoint multiplier depends on the root.
    Its guide correction cannot generally remain the original F0*F1 ratio.
    """
    root_law = edge_tapes(weights0, allowed0, cap0)
    child_law = edge_tapes(weights1, allowed1, cap1)
    result = defaultdict(F)
    for (r, _), pr in root_law.items():
        if r is None:
            result[None] += pr
            continue
        one = defaultdict(F)
        for (c, _), pc in child_law.items():
            endpoint = (r,c) if c is not None and (r,c) in final_mask else None
            one[endpoint] += pc
        for endpoint, probability in outer_tapes(dict(one), child_retries).items():
            result[endpoint] += pr*probability
    assert sum(result.values(), ZERO) == ONE
    return dict(result)


def counterexample():
    w0, w1 = (F(1,2),F(1,2)), (F(1,3),F(2,3))
    states = tuple(product(range(2),repeat=2))
    domain = {(0,0),(1,0),(1,1)}
    guide = {(r,c):w0[r]*w1[c] for r,c in states}
    physical = {x:F(1,4) for x in states}
    law = root_retaining_trial(w0,w1,{0,1},{0,1},domain,1,1,2)
    wrong = physical_kernel(states,domain,law,guide,physical)
    corrected = physical_kernel(states,domain,law,guide,physical,
                                corrected_guide={x:law.get(x,ZERO) for x in states})
    check_balance(states,corrected,physical)
    x,y=(0,0),(1,0)
    return dict(law=law,states=states,domain=domain,guide=guide,physical=physical,
                wrong=wrong,forward_flow=physical[x]*wrong[x][y],reverse_flow=physical[y]*wrong[y][x],
                after={y:sum((physical[x]*wrong[x][y] for x in states),ZERO) for y in states})


class FactorizedDimerBalanceTests(unittest.TestCase):
    def test_edge_tapes_match_capped_subdensity_and_null_including_zero_support(self):
        for weights in [(F(1,2),F(1,3),F(1,6)),(ZERO,F(1,4),F(3,4))]:
            for allowed in [set(),{0},{0,2},{0,1,2}]:
                p=sum((weights[i] for i in allowed),ZERO)
                for cap in range(4):
                    traces=edge_tapes(weights,allowed,cap)
                    law=defaultdict(F)
                    for (selected,prefix),probability in traces.items():
                        self.assertLessEqual(len(prefix),cap)
                        if selected is None:self.assertEqual(len(prefix),cap)
                        else:
                            self.assertIn(prefix[-1],allowed)
                            self.assertTrue(all(v not in allowed for v in prefix[:-1]))
                        law[selected]+=probability
                    self.assertEqual(law[None],(ONE-p)**cap)
                    for i,w in enumerate(weights):
                        self.assertEqual(law[i],retry_factor(p,cap)*w if i in allowed else ZERO)

    def test_all_caps_orders_and_short_circuit_laws_with_nonuniform_physical_target(self):
        states=tuple(product(range(3),repeat=2))
        cases=[((F(1,2),F(1,3),F(1,6)),(F(1,7),F(2,7),F(4,7))),
               ((ZERO,F(1,4),F(3,4)),(F(2,5),ZERO,F(3,5)))]
        allowed0,allowed1={0,1},{0,2}
        mask={(0,0),(1,0),(1,2),(2,2)} # (2,2) excluded only by B0.
        domain={(r,c) for r,c in states if r in allowed0 and c in allowed1 and (r,c) in mask}
        for w0,w1 in cases:
            guide={(r,c):w0[r]*w1[c] for r,c in states}
            p0=sum((w0[i] for i in allowed0),ZERO);p1=sum((w1[i] for i in allowed1),ZERO)
            d=sum((guide[x] for x in domain),ZERO)
            for k0,k1,j in product(range(4),repeat=3):
                a=retry_factor(p0,k0)*retry_factor(p1,k1);q=a*d
                self.assertLessEqual(q,ONE)
                expected={x:a*retry_factor(q,j)*guide[x] for x in domain}
                expected[None]=(ONE-q)**j
                for order in ['root_first','child_first']:
                    for short in [False,True]:
                        trial,_=joint_trial(w0,w1,allowed0,allowed1,mask,k0,k1,order,short)
                        law=outer_tapes(trial,j)
                        for x in (*states,None):self.assertEqual(law.get(x,ZERO),expected.get(x,ZERO))
                        for target_kind in ['positive','zero_source','zero_outside_domain']:
                            masses=[F(v) for v in [2,3,5,7,11,13,17,19,23]]
                            if target_kind=='zero_source':masses[0]=ZERO
                            if target_kind=='zero_outside_domain':
                                masses=[m if x in domain else ZERO for x,m in zip(states,masses)]
                            physical={x:m/sum(masses,ZERO) for x,m in zip(states,masses)}
                            kernel=physical_kernel(states,domain,law,guide,physical)
                            check_balance(states,kernel,physical)

    def test_empty_and_full_preliminary_and_final_masks(self):
        w0,w1=(F(1,3),F(2,3)),(F(3,4),F(1,4))
        states=tuple(product(range(2),repeat=2));guide={(r,c):w0[r]*w1[c] for r,c in states}
        for allowed0,allowed1,mask in [(set(),{0,1},set(states)),({0,1},set(),set(states)),({0,1},{0,1},set()),({0,1},{0,1},set(states))]:
            domain={(r,c) for r,c in states if r in allowed0 and c in allowed1 and (r,c) in mask}
            for k0,k1,j in product([0,1,3],repeat=3):
                trial,_=joint_trial(w0,w1,allowed0,allowed1,mask,k0,k1)
                law=outer_tapes(trial,j)
                success=bool(domain and k0 and k1 and j)
                self.assertEqual(law.get(None,ZERO),ZERO if success else ONE)
                for x in states:self.assertEqual(law.get(x,ZERO),guide[x] if success else ZERO)
                physical={x:F(i+1,10) for i,x in enumerate(states)}
                check_balance(states,physical_kernel(states,domain,law,guide,physical),physical)

    def test_stage_short_circuit_changes_trace_consumption_not_endpoint_law(self):
        args=((F(1,2),F(1,2)),(F(1,3),F(2,3)),{0},{0,1},{(0,0),(0,1)},1,2)
        short,skipped=joint_trial(*args,short_circuit=True)
        full,not_skipped=joint_trial(*args,short_circuit=False)
        self.assertEqual(short,full);self.assertEqual(skipped,F(1,2));self.assertEqual(not_skipped,ZERO)

    def test_root_retention_counterexample_is_stochastic_but_not_balanced(self):
        c=counterexample();law=c['law']
        self.assertEqual(law.get(None,ZERO),F(2,9))
        self.assertEqual(law[(0,0)],F(5,18));self.assertEqual(law[(1,0)],F(1,6));self.assertEqual(law[(1,1)],F(1,3))
        self.assertTrue(all(sum(row.values(),ZERO)==ONE for row in c['wrong'].values()))
        self.assertEqual(c['forward_flow'],F(1,24));self.assertEqual(c['reverse_flow'],F(5,72))
        self.assertNotEqual(c['forward_flow'],c['reverse_flow'])
        self.assertEqual(c['after'][(0,0)],F(11,36));self.assertEqual(c['after'][(1,0)],F(2,9));self.assertEqual(c['after'][(1,1)],F(2,9))
        with self.assertRaises(AssertionError):check_balance(c['states'],c['wrong'],c['physical'])


def serial(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(set,tuple,list)):return [serial(v) for v in value]
    return value


def receipt(result):
    paths=['tools/test_factorized_dimer_balance.py','docs/factorized-dimer-conditioning.md',
           'formal/ReversibleSampling/CappedIndependent.lean','formal/capped-independent-validation.json',
           'formal/capped-independent-audit.txt']
    c=counterexample()
    return dict(schema='factorized-dimer-exact-finite-check-v1',complete=True,passed=result.wasSuccessful(),
        checked_at_utc=datetime.now(timezone.utc).isoformat(),arithmetic='fractions.Fraction; exact equality; no tolerance or RNG',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        main_joint_laws_enumerated=2*64*2*2,main_physical_kernels_checked=2*64*2*2*3,
        edge_cap_values=[0,1,2,3],joint_cap_values=[0,1,2,3],stage_orders=['root_first','child_first'],
        short_circuit_policies=[False,True],physical_target_zero_weights_tested=True,proposal_zero_weights_tested=True,
        additional_limits='32 edge controls,108 empty/full domain controls,stage-consumption control and exact retained-root counterexample',
        counterexample={k:c[k] for k in ['law','physical','forward_flow','reverse_flow','after']},
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        scope='Finite random-tape marginalization and detailed balance for specified toy matrices, not a general Lean/SE(3)/Rust implementation theorem.',
        physical_jobs=0,new_protein_proposals=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FactorizedDimerBalanceTests))
    if args.receipt:
        with args.receipt.open('x') as out:json.dump(serial(receipt(result)),out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
