#!/usr/bin/env python3
"""Exact path-enumeration checks for prospective context selectors.

Independent of production proposal, geometry, bath, and arithmetic code. The
fourth toy state is a hard-invalid raw draw. All arithmetic uses Fraction; every
raw history up to the declared (at most three) attempts is explicitly visited.
This validates finite-state balance identities, not sampler efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import unittest


ZERO, ONE = F(0), F(1)
HARD = (True, True, True, False)
TARGET = (F(1, 7), F(2, 7), F(4, 7), ZERO)
DENSITIES = (
    (F(1, 2), F(1, 8), F(1, 8), F(1, 4)),
    (F(1, 16), F(3, 16), F(1, 4), F(1, 2)),
    (ZERO, ZERO, ZERO, ONE),  # Z=0; no hard-valid source support.
    (F(1, 3), F(1, 3), F(1, 3), ZERO),  # Zero prior weight.
)
WEIGHTS = (F(1, 2), F(1, 3), F(1, 6), ZERO)
SCORES = ((8, 1, 1, 1), (1, 7, 2, 1), (2, 1, 4, 1), (1, 1, 1, 1))
CAPS = (2, 3, 3, 0)


@dataclass(frozen=True)
class Path:
    context: int | None
    draws: tuple[tuple[int, int], ...]
    candidate: int | None
    probability: F


def enumerate_raw(distribution, cap, hard=HARD, *, context=None):
    """Enumerate histories, without a geometric-series success shortcut.

    distribution lists (context, raw_state, probability). For retained context
    there is one context in that law. For independent-Q or the wrong reselection
    rule, every trial redraws its context as part of this joint raw law.
    """
    assert type(cap) is int and 0 <= cap <= 3
    assert all(isinstance(p, F) and p >= 0 for _, _, p in distribution)
    assert sum((p for _, _, p in distribution), ZERO) == ONE
    paths = []

    def visit(history, probability):
        if len(history) == cap:
            paths.append(Path(context, history, None, probability))
            return
        for selected, state, chance in distribution:
            if chance == 0:
                continue
            next_history = history + ((selected, state),)
            next_probability = probability * chance
            if hard[state]:
                paths.append(Path(context, next_history, state, next_probability))
            else:
                visit(next_history, next_probability)

    visit((), ONE)
    return paths


def mixture(densities, weights):
    return tuple(sum((w * g[x] for w, g in zip(weights, densities)), ZERO)
                 for x in range(len(densities[0])))


def selector(kind, x, densities, weights, scores=SCORES):
    if kind == 'prior':
        return weights
    if kind == 'local':
        # A positive prior floor, including the complete source normalization.
        norm = sum((w * F(s[x]) for w, s in zip(weights, scores)), ZERO)
        assert norm > 0
        epsilon = F(1, 10)
        return tuple(epsilon*w + (ONE-epsilon)*w*F(s[x])/norm
                     for w, s in zip(weights, scores))
    assert kind == 'posterior'
    q = mixture(densities, weights)[x]
    assert q > 0
    return tuple(w * g[x] / q for w, g in zip(weights, densities))


def proposals(kind, x, densities=DENSITIES, weights=WEIGHTS, caps=CAPS,
              scores=SCORES, hard=HARD, global_cap=3):
    """Keep failed paths and source-zero self-loops, matching the cap contract."""
    q = mixture(densities, weights)
    # The invalid-source row only completes the toy matrix outside target
    # support. Production instead rejects an invalid source as a fatal input.
    if not hard[x] or (kind in ('independent_q', 'posterior', 'redrawn_posterior') and q[x] == 0):
        return [Path(None, (), None, ONE)]
    if kind in ('independent_q', 'redrawn_posterior'):
        selected = weights if kind == 'independent_q' else selector('posterior', x, densities, weights)
        raw = [(s, y, selected[s]*g[y]) for s, g in enumerate(densities)
               for y in range(len(hard))]
        return enumerate_raw(raw, global_cap, hard)
    selected = selector(kind, x, densities, weights, scores)
    paths = []
    for s, chance in enumerate(selected):
        if chance == 0:
            continue
        if densities[s][x] == 0:
            # Rust's SourceZeroReverseFlow does not consume raw draws.
            paths.append(Path(s, (), None, chance))
            continue
        raw = [(s, y, value) for y, value in enumerate(densities[s])]
        paths.extend(Path(s, p.draws, p.candidate, chance*p.probability)
                     for p in enumerate_raw(raw, caps[s], hard, context=s))
    return paths


def accept_ratio(kind, x, y, context, target, densities, weights, scores,
                 omit_local_selection=False):
    if kind in ('posterior', 'independent_q', 'redrawn_posterior'):
        q = mixture(densities, weights)
        numerator, denominator = target[y]*q[x], target[x]*q[y]
    else:
        g = densities[context]
        numerator, denominator = target[y]*g[x], target[x]*g[y]
        if kind == 'local' and not omit_local_selection:
            numerator *= selector('local', y, densities, weights, scores)[context]
            denominator *= selector('local', x, densities, weights, scores)[context]
    # Only zero-target rows can have zero denominator on a generated path.
    if denominator == 0:
        return ONE if numerator > 0 else ZERO
    return min(ONE, numerator/denominator)


def transition(kind, *, target=TARGET, densities=DENSITIES, weights=WEIGHTS,
               caps=CAPS, scores=SCORES, hard=HARD, global_cap=3,
               omit_local_selection=False):
    n = len(hard)
    assert len(target) == n and sum(target, ZERO) == sum(weights, ZERO) == ONE
    assert len(densities) == len(weights) == len(caps)
    assert all(len(g) == n and sum(g, ZERO) == ONE and min(g) >= 0 for g in densities)
    assert all(target[x] >= 0 and (hard[x] or target[x] == 0) for x in range(n))
    matrix = [[ZERO for _ in range(n)] for _ in range(n)]
    accounting = []
    for x in range(n):
        paths = proposals(kind, x, densities, weights, caps, scores, hard, global_cap)
        assert sum((p.probability for p in paths), ZERO) == ONE
        failed = rejected = accepted = ZERO
        for path in paths:
            if path.candidate is None:
                matrix[x][x] += path.probability
                failed += path.probability
                continue
            y = path.candidate
            alpha = accept_ratio(kind, x, y, path.context, target, densities, weights,
                                 scores, omit_local_selection)
            matrix[x][y] += path.probability*alpha
            matrix[x][x] += path.probability*(ONE-alpha)
            accepted += path.probability*alpha
            rejected += path.probability*(ONE-alpha)
        accounting.append(dict(paths=paths, failed=failed, rejected=rejected, accepted=accepted))
        assert failed+rejected+accepted == ONE
    return matrix, accounting


def stationary(matrix, target):
    return tuple(sum((target[x]*matrix[x][y] for x in range(len(target))), ZERO)
                 for y in range(len(target)))


def flow_defects(matrix, target):
    return {(x, y): target[x]*matrix[x][y]-target[y]*matrix[y][x]
            for x in range(len(target)) for y in range(x+1, len(target))
            if target[x]*matrix[x][y] != target[y]*matrix[y][x]}


class ContextSelectorBalance(unittest.TestCase):
    def assert_invariant(self, matrix, target=TARGET):
        self.assertTrue(all(all(p >= 0 for p in row) and sum(row, ZERO) == ONE for row in matrix))
        self.assertEqual(flow_defects(matrix, target), {})
        self.assertEqual(stationary(matrix, target), target)

    def test_explicit_raw_histories_include_cap_failure_and_zero_mass(self):
        for s, g in enumerate(DENSITIES):
            z = sum((p for p, valid in zip(g, HARD) if valid), ZERO)
            for cap in range(4):
                paths = enumerate_raw([(s, y, p) for y, p in enumerate(g)], cap, context=s)
                self.assertEqual(sum((p.probability for p in paths), ZERO), ONE)
                self.assertEqual(sum((p.probability for p in paths if p.candidate is None), ZERO), (ONE-z)**cap)
                for p in paths:
                    self.assertLessEqual(len(p.draws), cap)
                    self.assertTrue(all(not HARD[y] for _, y in p.draws[:-1]))
                    if p.candidate is None:
                        self.assertEqual(len(p.draws), cap)
                        self.assertTrue(all(not HARD[y] for _, y in p.draws))
                    else:
                        self.assertTrue(HARD[p.draws[-1][1]])
                        self.assertEqual(p.candidate, p.draws[-1][1])
                # Independent closed-form cross-check of the enumeration.
                factor = sum(((ONE-z)**r for r in range(cap)), ZERO)
                for y in range(3):
                    self.assertEqual(sum((p.probability for p in paths if p.candidate == y), ZERO), g[y]*factor)

    def test_four_correct_kernels_unequal_masses_occupancies_and_rejections(self):
        self.assertNotEqual(sum(DENSITIES[0][:3], ZERO), sum(DENSITIES[1][:3], ZERO))
        for kind in ('prior', 'local', 'posterior', 'independent_q'):
            with self.subTest(kind=kind):
                matrix, rows = transition(kind)
                self.assert_invariant(matrix)
                self.assertTrue(any(row['failed'] > 0 for row in rows[:3]))
                self.assertTrue(any(row['rejected'] > 0 for row in rows[:3]))
                self.assertGreater(matrix[0][2], 0)
                self.assertEqual(matrix[0][2], 4*matrix[2][0])
                self.assertNotEqual(matrix[0][2], matrix[2][0])

    def test_retained_context_zero_cap_zero_weight_and_zero_source_support(self):
        for kind in ('prior', 'local', 'posterior'):
            matrix, rows = transition(kind, caps=(0, 3, 3, 0))
            self.assert_invariant(matrix)
            for row in rows[:3]:
                for path in row['paths']:
                    self.assertNotEqual(path.context, 3)  # w=0 never selected.
                    if path.context in (0, 2):
                        self.assertEqual(path.draws, ())
                        self.assertIsNone(path.candidate)
                    else:
                        self.assertTrue(all(s == path.context for s, _ in path.draws))
        for kind in ('prior', 'local', 'posterior', 'independent_q'):
            matrix, _ = transition(kind, caps=(0, 0, 0, 0), global_cap=0)
            self.assertEqual(matrix, [[ONE if x == y else ZERO for y in range(4)] for x in range(4)])

    def test_zero_total_source_density_is_a_self_loop_not_conditioning(self):
        densities = ((F(1, 2), ZERO, ZERO, F(1, 2)), (ZERO, ZERO, ZERO, ONE))
        for kind in ('prior', 'local', 'posterior', 'independent_q'):
            matrix, rows = transition(kind, densities=densities, weights=(F(1, 2), F(1, 2)),
                                      caps=(3, 3), scores=SCORES[:2])
            self.assert_invariant(matrix)
            for x in (1, 2):
                self.assertEqual(matrix[x][x], ONE)
                self.assertTrue(all(p.draws == () for p in rows[x]['paths']))

    def test_zero_target_probability_does_not_require_strict_positivity(self):
        target = (F(1, 3), ZERO, F(2, 3), ZERO)
        for kind in ('prior', 'local', 'posterior', 'independent_q'):
            matrix, _ = transition(kind, target=target)
            self.assert_invariant(matrix, target)
            self.assertEqual(matrix[0][1], ZERO)
            self.assertEqual(matrix[2][1], ZERO)

    def test_omitting_local_selection_ratio_breaks_balance(self):
        correct, _ = transition('local')
        wrong, _ = transition('local', omit_local_selection=True)
        self.assert_invariant(correct)
        self.assertTrue(all(sum(row, ZERO) == ONE for row in wrong))
        self.assertTrue(flow_defects(wrong, TARGET))
        self.assertNotEqual(stationary(wrong, TARGET), TARGET)

    def test_redrawing_posterior_context_after_failure_needs_another_factor(self):
        correct, _ = transition('posterior', caps=(3, 3, 3, 3))
        wrong, _ = transition('redrawn_posterior', global_cap=3)
        self.assert_invariant(correct)
        self.assertTrue(all(sum(row, ZERO) == ONE for row in wrong))
        self.assertTrue(flow_defects(wrong, TARGET))
        self.assertNotEqual(stationary(wrong, TARGET), TARGET)
        # At one raw draw there has been no context reselection, so both laws agree.
        retained_one, _ = transition('posterior', caps=(1, 1, 1, 1))
        redrawn_one, _ = transition('redrawn_posterior', global_cap=1)
        self.assertEqual(retained_one, redrawn_one)
        self.assert_invariant(redrawn_one)

    def test_alternating_coordinate_like_kernels_preserve_stationarity(self):
        first, _ = transition('prior')
        second, _ = transition('local')
        product = [[sum((first[x][z]*second[z][y] for z in range(4)), ZERO)
                    for y in range(4)] for x in range(4)]
        self.assertEqual(stationary(product, TARGET), TARGET)
        self.assertTrue(all(sum(row, ZERO) == ONE for row in product))
        self.assertTrue(flow_defects(product, TARGET))  # Sweep need not be reversible.


if __name__ == '__main__':
    unittest.main()
