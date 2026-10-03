"""Exact finite-state oracle for two independent capped edge stages.

Enumerates successful draw probabilities directly, then the auxiliary refresh
and a single MH decision. This checks accepted flow for a finite analogue; it
does not prove continuous geometry, floating-point or Rust execution correct.
"""
from fractions import Fraction as F
from itertools import product
import unittest


ROOT = (F(1, 6), F(1, 3), F(1, 2))
CHILD = (F(1, 2), F(1, 3), F(1, 6))
STATES = tuple(product(range(3), repeat=2))
HARD = frozenset(x for x in STATES if x not in ((0, 2), (2, 0)))
KR = (0, 1, 3)
KI = (2, 0, 1)
PHYSICAL = {x: F(1+2*x[0]+3*x[1]+x[0]*x[1], 1) for x in HARD}


def thresholds(count, multiplicity):
    if multiplicity is None:
        return {None: F(1)}
    return {k: F((k+1)**multiplicity-k**multiplicity, (count+1)**multiplicity)
            for k in range(count+1)}


def score_allows(count, threshold):
    return threshold is None or count >= threshold


def cap_success(weights, allowed, cap):
    """First qualifying outcome, explicitly summing all prior-failure counts."""
    p = sum((weights[i] for i in allowed), F(0))
    remaining = F(1)
    result = [F(0)]*len(weights)
    for _ in range(cap):
        for i in allowed:
            result[i] += remaining*weights[i]
        remaining *= 1-p
    assert sum(result)+remaining == 1
    return result


def joint_success(kr, ki, caps, order):
    r = cap_success(ROOT, [i for i in range(3) if score_allows(KR[i], kr)], caps[0])
    c = cap_success(CHILD, [i for i in range(3) if score_allows(KI[i], ki)], caps[1])
    # An exhausted first stage contributes to failure, without running stage2.
    # Enumerating either independent order gives the same successful pairs.
    one = {x: F(0) for x in HARD}
    first, second = (r, c) if order == 'root_first' else (c, r)
    for i, wi in enumerate(first):
        for j, wj in enumerate(second):
            x = (i, j) if order == 'root_first' else (j, i)
            if x in HARD:
                one[x] += wi*wj
    remaining, success = F(1), {x: F(0) for x in HARD}
    for _ in range(caps[2]):
        for x in HARD:
            success[x] += remaining*one[x]
        remaining *= 1-sum(one.values())
    assert sum(success.values())+remaining == 1
    return success


def transition(caps, mroot, mint, order='root_first'):
    flow = {}
    for old in HARD:
        rates = {new: F(0) for new in HARD}
        for kr, ar in thresholds(KR[old[0]], mroot).items():
            for ki, ai in thresholds(KI[old[1]], mint).items():
                q = joint_success(kr, ki, caps, order)
                for new in HARD:
                    if new == old or q[new] == 0:
                        continue
                    br = thresholds(KR[new[0]], mroot).get(kr, F(0))
                    bi = thresholds(KI[new[1]], mint).get(ki, F(0))
                    ratio = PHYSICAL[new]/PHYSICAL[old]
                    ratio *= ROOT[old[0]]*CHILD[old[1]]/(ROOT[new[0]]*CHILD[new[1]])
                    ratio *= br*bi/(ar*ai)
                    rates[new] += ar*ai*q[new]*min(F(1), ratio)
        rates[old] = 1-sum(rates.values())
        assert rates[old] >= 0
        flow[old] = rates
    return flow


class ExternalThresholdBalance(unittest.TestCase):
    def test_exact_invariance_and_flow_with_both_thresholds(self):
        for caps in ((1, 1, 1), (3, 2, 1), (2, 3, 4)):
            for mr, mi in ((1, 1), (2, 4), (4, 2)):
                with self.subTest(caps=caps, multiplicities=(mr, mi)):
                    flow = transition(caps, mr, mi)
                    self.assertEqual(flow, transition(caps, mr, mi, 'child_first'))
                    for x in HARD:
                        self.assertEqual(sum(flow[x].values()), 1)
                        self.assertEqual(sum(PHYSICAL[y]*flow[y][x] for y in HARD), PHYSICAL[x])
                        for y in HARD:
                            self.assertEqual(PHYSICAL[x]*flow[x][y], PHYSICAL[y]*flow[y][x])

    def test_disabled_one_guide_and_empty_cloud_limits(self):
        for mr, mi in ((None, None), (4, None), (None, 4)):
            flow = transition((2, 3, 2), mr, mi)
            for x, y in product(HARD, repeat=2):
                self.assertEqual(PHYSICAL[x]*flow[x][y], PHYSICAL[y]*flow[y][x])
        for m in (1, 4, 20):
            self.assertEqual(thresholds(0, m), {0: F(1)})
        # Threshold zero passes every candidate; its target ratio still matters.
        self.assertEqual(thresholds(3, 4)[0]/thresholds(0, 4)[0], F(1, 256))

    def test_zero_cap_is_identity(self):
        for caps in ((0, 2, 3), (3, 0, 2), (2, 3, 0)):
            self.assertEqual(transition(caps, 4, 4),
                             {x: {y: F(x == y) for y in HARD} for x in HARD})

    def test_retaining_root_during_coupled_child_retries_breaks_full_f_ratio(self):
        # Actual first-success distribution when a child depends on the sampled
        # root. Unequal feasible masses create root-dependent cap factors.
        q = {}
        for r in range(3):
            children = cap_success(CHILD, [c for c in range(3) if (r, c) in HARD], 3)
            for c in range(3):
                if (r, c) in HARD:
                    q[r, c] = ROOT[r]*children[c]
        witnesses = []
        for x, y in product(HARD, repeat=2):
            ratio = PHYSICAL[y]/PHYSICAL[x]*ROOT[x[0]]*CHILD[x[1]]/(ROOT[y[0]]*CHILD[y[1]])
            forward = PHYSICAL[x]*q[y]*min(F(1), ratio)
            reverse = PHYSICAL[y]*q[x]*min(F(1), 1/ratio)
            if forward != reverse:
                witnesses.append((x, y, forward, reverse))
            # Scoring the actual proposal would restore balance; omitting its
            # unknown conditional-cap normalizer is precisely the problem.
            actual = PHYSICAL[y]*q[x]/(PHYSICAL[x]*q[y])
            self.assertEqual(PHYSICAL[x]*q[y]*min(F(1), actual),
                             PHYSICAL[y]*q[x]*min(F(1), 1/actual))
        self.assertTrue(witnesses)


if __name__ == '__main__':
    unittest.main()
