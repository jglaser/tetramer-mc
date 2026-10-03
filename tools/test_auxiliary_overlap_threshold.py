#!/usr/bin/env python3
"""Exact finite auxiliary-threshold checks; no poses, clouds, RNG, or physics jobs."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import unittest

from test_factorized_dimer_balance import (
    ZERO, ONE, check_balance, joint_trial, outer_tapes, physical_kernel, retry_factor,
)


def auxiliary_mass(count, threshold, exponent):
    assert type(count) is int and count >= 0
    assert type(threshold) is int
    assert type(exponent) is int and exponent >= 1
    if threshold < 0 or threshold > count:
        return ZERO
    return F((threshold+1)**exponent-threshold**exponent, (count+1)**exponent)


def fixture():
    states = tuple(product(range(3), repeat=2))
    root = (F(1,2), F(1,3), F(1,6))
    child = (F(1,6), F(1,3), F(1,2))
    # These are abstract channel restrictions, not a simulated hard geometry.
    allowed_root, allowed_child = {0,1}, {0,1}
    final = {(0,0), (0,1), (1,1)}
    clouds = ((F(2,5), (0,1,0)), (F(3,5), (2,3,0)))
    physical = {x:F(i+1,45) for i,x in enumerate(states)}
    return states, root, child, allowed_root, allowed_child, final, clouds, physical


def laws(model, exponent, caps, physical=None, omit_auxiliary_ratio=False):
    states,w0,w1,b0,b1,final,clouds,original_physical = model
    physical = original_physical if physical is None else physical
    k0,k1,j = caps
    guide = {(r,h):w0[r]*w1[h] for r,h in states}
    blocks = {}
    marginal = {x:{y:ZERO for y in states} for x in states}
    for p,(rho,counts) in enumerate(clouds):
        for k in range(max(counts)+1):
            selected = {h for h in b1 if counts[h] >= k}
            domain = {(r,h) for r,h in final if r in b0 and h in selected}
            trial,_ = joint_trial(w0,w1,b0,selected,final,k0,k1)
            candidate = outer_tapes(trial,j)
            joint_mass = {x:physical[x]*rho*auxiliary_mass(counts[x[1]],k,exponent) for x in states}
            conditional = physical_kernel(states,domain,candidate,guide,
                                          physical if omit_auxiliary_ratio else joint_mass)
            blocks[(p,k)] = dict(domain=domain,candidate=candidate,kernel=conditional,target=joint_mass)
            for x in states:
                refresh = rho*auxiliary_mass(counts[x[1]],k,exponent)
                for y in states:
                    marginal[x][y] += refresh*conditional[x][y]
    return blocks,marginal


def missing_ratio_counterexample(exponent=1):
    states=((0,0),(0,1))
    model=(states,(ONE,),(F(1,2),F(1,2)),{0},{0,1},set(states),
           ((ONE,(0,1)),),{x:F(1,2) for x in states})
    _,wrong=laws(model,exponent,(1,1,1),omit_auxiliary_ratio=True)
    _,correct=laws(model,exponent,(1,1,1))
    check_balance(states,correct,model[-1])
    x,y=states
    return dict(exponent=exponent,forward_flow=model[-1][x]*wrong[x][y],
                reverse_flow=model[-1][y]*wrong[y][x],
                after={b:sum((model[-1][a]*wrong[a][b] for a in states),ZERO) for b in states})


class AuxiliaryOverlapThresholdTests(unittest.TestCase):
    def test_maximum_of_uniform_integers_realizes_exact_threshold_law(self):
        for m in (1,4):
            for count in range(4):
                frequencies=defaultdict(int)
                for tape in product(range(count+1),repeat=m):
                    frequencies[max(tape)]+=1
                denominator=(count+1)**m
                self.assertEqual(sum(frequencies.values()),denominator)
                for k in range(count+1):
                    self.assertEqual(F(frequencies[k],denominator),auxiliary_mass(count,k,m))

    def test_auxiliary_normalization_zero_count_and_cdf(self):
        for m in (1,4):
            for count in range(6):
                self.assertEqual(sum((auxiliary_mass(count,k,m) for k in range(count+1)),ZERO),ONE)
                self.assertEqual(auxiliary_mass(count,count+1,m),ZERO)
                for k in range(count+1):
                    self.assertEqual(sum((auxiliary_mass(count,i,m) for i in range(k+1)),ZERO),
                                     F((k+1)**m,(count+1)**m))
            self.assertEqual(auxiliary_mass(0,0,m),ONE)

    def test_all_finite_caps_and_joint_and_marginal_balance(self):
        model=fixture();states,w0,w1,b0,b1,final,clouds,physical=model
        targets=[physical,{x:F(1,len(states)) for x in states}]
        guide={(r,h):w0[r]*w1[h] for r,h in states}
        for m in (1,4):
            for caps in product(range(3),repeat=3):
                for target in targets:
                    blocks,marginal=laws(model,m,caps,target)
                    check_balance(states,marginal,target)
                    for (p,k),block in blocks.items():
                        check_balance(states,block['kernel'],block['target'])
                        _,counts=clouds[p]
                        p0=sum((w0[r] for r in b0),ZERO)
                        p1=sum((w1[h] for h in b1 if counts[h]>=k),ZERO)
                        a=retry_factor(p0,caps[0])*retry_factor(p1,caps[1])
                        d=sum((guide[x] for x in block['domain']),ZERO);q=a*d
                        self.assertLessEqual(q,ONE)
                        self.assertEqual(block['candidate'].get(None,ZERO),(ONE-q)**caps[2])
                        for x in states:
                            expected=a*retry_factor(q,caps[2])*guide[x] if x in block['domain'] else ZERO
                            self.assertEqual(block['candidate'].get(x,ZERO),expected)
                    if 0 in caps:
                        self.assertEqual(marginal,{x:{y:ONE if x==y else ZERO for y in states} for x in states})

    def test_auxiliary_ratio_cancels_threshold_numerator(self):
        for m in (1,4):
            for old,new in product(range(5),repeat=2):
                for k in range(min(old,new)+1):
                    self.assertEqual(auxiliary_mass(new,k,m)/auxiliary_mass(old,k,m),
                                     F((old+1)**m,(new+1)**m))

    def test_refresh_then_pose_preserves_joint_but_need_not_be_jointly_reversible(self):
        model=fixture();states,_,_,_,_,_,clouds,physical=model
        m=4;blocks,marginal=laws(model,m,(2,2,1))
        augmented=[(x,p,k) for x in states for p,(_,counts) in enumerate(clouds)
                   for k in range(counts[x[1]]+1)]
        mu={(x,p,k):physical[x]*clouds[p][0]*auxiliary_mass(clouds[p][1][x[1]],k,m)
            for x,p,k in augmented}
        def composed(a,b):
            x,_,_=a;y,p,k=b
            return (clouds[p][0]*auxiliary_mass(clouds[p][1][x[1]],k,m)
                    *blocks[(p,k)]['kernel'][x][y])
        for a in augmented:
            self.assertEqual(sum((composed(a,b) for b in augmented),ZERO),ONE)
        for b in augmented:
            self.assertEqual(sum((mu[a]*composed(a,b) for a in augmented),ZERO),mu[b])
        self.assertTrue(any(mu[a]*composed(a,b)!=mu[b]*composed(b,a)
                            for a in augmented for b in augmented))
        check_balance(states,marginal,physical)

    def test_uncompensated_threshold_changes_the_physical_marginal(self):
        for m in (1,4):
            result=missing_ratio_counterexample(m)
            self.assertEqual(result['forward_flow'],F(1,4))
            self.assertEqual(result['reverse_flow'],F(1,2**(m+2)))
            self.assertNotEqual(result['forward_flow'],result['reverse_flow'])
            self.assertEqual(result['after'][(0,0)],F(1,4)+F(1,2**(m+2)))


if __name__=='__main__':
    unittest.main()
