#!/usr/bin/env python3
"""Exact finite-state checks for a prospective frozen-surrogate inner chain.

No production sampler, geometry, cloud generator, or physical bath is invoked.
Positive Fraction weights stand for exp(S_C); the last toy state is hard-invalid.
The outer physical ratio is exact here. Reusing the existing conditional Poisson
gate still requires its separate augmented-variable balance argument. These
checks establish neither floating-point correctness nor sampling efficiency.
"""
from fractions import Fraction as F
from itertools import product
import unittest


ZERO, ONE = F(0), F(1)
SURROGATE = tuple(map(F, (1, 2, 5, 3, 0)))
PHYSICAL = tuple(map(F, (3, 1, 7, 2, 0)))
N = len(SURROGATE)


def identity():
    return [[F(i == j) for j in range(N)] for i in range(N)]


def permutation(order):
    return [[F(j == order[i]) for j in range(N)] for i in range(N)]


def mixture(kernels, weights):
    assert sum(weights, ZERO) == ONE and all(w >= 0 for w in weights)
    return [[sum((w * k[i][j] for w, k in zip(weights, kernels)), ZERO)
             for j in range(N)] for i in range(N)]


def multiply(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(N)), ZERO)
             for j in range(N)] for i in range(N)]


def power(kernel, steps):
    assert type(steps) is int and steps >= 0
    result = identity()
    for _ in range(steps):
        result = multiply(result, kernel)
    return result


def inner_mh(proposal, weights=SURROGATE, *, omit_proposal_ratio=False):
    """Every raw draw, invalid proposal, and MH rejection consumes one step."""
    kernel = [[ZERO for _ in range(N)] for _ in range(N)]
    for i in range(N):
        if weights[i] == 0:
            kernel[i][i] = ONE  # Outside the channel; never an admitted source.
            continue
        for j, chance in enumerate(proposal[i]):
            if chance == 0:
                continue
            ratio = weights[j] / weights[i]
            if not omit_proposal_ratio:
                ratio *= proposal[j][i] / chance
            accepted = chance * min(ONE, ratio)
            kernel[i][j] += accepted
            kernel[i][i] += chance - accepted
    return kernel


def outer_gate(kernel, surrogate=SURROGATE, physical=PHYSICAL, *, correction="correct",
               extra_density=None):
    result = [[ZERO for _ in range(N)] for _ in range(N)]
    for i in range(N):
        if physical[i] == 0:
            result[i][i] = ONE
            continue
        assert surrogate[i] > 0
        for j, chance in enumerate(kernel[i]):
            if chance == 0:
                continue
            if physical[j] == 0:
                accepted = ZERO
            else:
                ratio = physical[j] / physical[i]
                if correction == "correct":
                    ratio *= surrogate[i] / surrogate[j]
                elif correction == "wrong_sign":
                    ratio *= surrogate[j] / surrogate[i]
                else:
                    assert correction == "omitted"
                if extra_density is not None:
                    ratio *= extra_density[i] / extra_density[j]
                accepted = chance * min(ONE, ratio)
            result[i][j] += accepted
            result[i][i] += chance - accepted
    return result


def histories(kernel, start, steps):
    """Enumerate retained-state histories, including every repeated state."""
    if steps == 0:
        yield (start,), ONE
        return
    for history, probability in histories(kernel, start, steps - 1):
        for end, chance in enumerate(kernel[history[-1]]):
            if chance:
                yield history + (end,), probability * chance


def stopped_on_first_change(kernel, cap):
    """Deliberately wrong replacement for a fixed-length inner chain."""
    result = [[ZERO for _ in range(N)] for _ in range(N)]
    for start in range(N):
        def visit(probability, remaining):
            if remaining == 0:
                result[start][start] += probability
                return
            for end, chance in enumerate(kernel[start]):
                if end == start:
                    visit(probability * chance, remaining - 1)
                else:
                    result[start][end] += probability * chance
        visit(ONE, cap)
    return result


def balanced(kernel, weights):
    return (all(sum(row, ZERO) == ONE and all(p >= 0 for p in row) for row in kernel)
            and all(weights[i] * kernel[i][j] == weights[j] * kernel[j][i]
                    for i in range(N) for j in range(N)))


class SurrogateFlowTests(unittest.TestCase):
    def setUp(self):
        self.a = inner_mh(permutation((1, 0, 3, 2, 4)))
        self.b = inner_mh(permutation((2, 3, 0, 1, 4)))
        self.invalid = inner_mh(permutation((4, 1, 2, 3, 0)))
        self.kernel = mixture((self.a, self.b, self.invalid), (F(1, 3),) * 3)

    def test_random_scan_and_fixed_powers_preserve_all_null_residence(self):
        for k in (self.a, self.b, self.invalid, self.kernel):
            self.assertTrue(balanced(k, SURROGATE))
        self.assertEqual(self.invalid[0][0], ONE)
        self.assertEqual(self.invalid[0][4], ZERO)
        for steps in range(5):
            expected = power(self.kernel, steps)
            self.assertTrue(balanced(expected, SURROGATE))
            for start in range(4):
                actual = [ZERO] * N
                for history, probability in histories(self.kernel, start, steps):
                    self.assertEqual(len(history), steps + 1)
                    self.assertNotIn(4, history)
                    actual[history[-1]] += probability
                self.assertEqual(actual, expected[start])
        self.assertGreater(self.kernel[0][0], ZERO)

    def test_one_outer_correction_after_every_fixed_horizon(self):
        for steps in range(5):
            self.assertTrue(balanced(outer_gate(power(self.kernel, steps)), PHYSICAL))
        self.assertEqual(outer_gate(power(self.kernel, 0)), identity())

    def test_missing_or_wrong_sign_correction_is_detected(self):
        endpoint = power(self.kernel, 3)
        for correction in ("omitted", "wrong_sign"):
            self.assertFalse(balanced(outer_gate(endpoint, correction=correction), PHYSICAL))

    def test_ordered_sweep_fails_but_reversal_symmetric_schedule_passes(self):
        forward = multiply(self.a, self.b)
        reverse = multiply(self.b, self.a)
        self.assertFalse(balanced(forward, SURROGATE))
        self.assertFalse(balanced(outer_gate(forward), PHYSICAL))
        randomized = mixture((forward, reverse), (F(1, 2), F(1, 2)))
        self.assertTrue(balanced(randomized, SURROGATE))
        self.assertTrue(balanced(outer_gate(randomized), PHYSICAL))

    def test_first_change_stopping_and_source_dependent_horizon_fail(self):
        stopped = stopped_on_first_change(self.kernel, 3)
        self.assertFalse(balanced(stopped, SURROGATE))
        self.assertFalse(balanced(outer_gate(stopped), PHYSICAL))
        variable = [power(self.kernel, 1 + i % 2)[i] for i in range(N)]
        self.assertFalse(balanced(outer_gate(variable), PHYSICAL))

    def test_reducible_rigid_fiber_needs_no_full_dimensional_density(self):
        endpoint = power(self.a, 3)
        self.assertEqual(endpoint[0][2], ZERO)
        self.assertEqual(endpoint[0][3], ZERO)
        self.assertTrue(balanced(outer_gate(endpoint), PHYSICAL))

    def test_atlas_ratio_belongs_inside_and_is_not_added_twice(self):
        density = tuple(F(x, 16) for x in (1, 2, 4, 8, 1))
        proposal = [list(density) for _ in range(N)]
        inner = inner_mh(proposal)
        self.assertTrue(balanced(inner, SURROGATE))
        self.assertFalse(balanced(inner_mh(proposal, omit_proposal_ratio=True), SURROGATE))
        endpoint = power(inner, 3)
        self.assertTrue(balanced(outer_gate(endpoint), PHYSICAL))
        self.assertFalse(balanced(outer_gate(endpoint, extra_density=density), PHYSICAL))

    def test_endpoint_independent_cloud_mixture_and_horizon_mixture(self):
        other = tuple(map(F, (7, 1, 2, 4, 0)))
        proposal = permutation((1, 0, 3, 2, 4))
        a = outer_gate(power(inner_mh(proposal), 2))
        b = outer_gate(power(inner_mh(proposal, other), 2), surrogate=other)
        self.assertTrue(balanced(mixture((a, b), (F(2, 7), F(5, 7))), PHYSICAL))
        horizon = mixture([power(self.kernel, n) for n in (0, 1, 3)],
                          (F(1, 4), F(1, 2), F(1, 4)))
        self.assertTrue(balanced(outer_gate(horizon), PHYSICAL))

    def test_two_body_overlap_score_pointwise_union_identity(self):
        # Common integration point: this is the volume identity, not a claim
        # that two arbitrary realized body-frame quadratures give exact volumes.
        for first, second, spectator_a, spectator_b in product((False, True), repeat=4):
            spectator = spectator_a or spectator_b
            units = (int(first) * (2 * int(spectator) + int(not spectator and second))
                     + int(second) * (2 * int(spectator) + int(not spectator and first)))
            internal = int(first and second)
            external = int(spectator and (first or second))
            self.assertEqual(F(units, 2), internal + external)
            self.assertEqual(internal + external,
                             int(first) + int(second) - int((first or second) and not spectator))
        # Monte Carlo normalization retains the raw box count, not the number
        # retained inside the exclusion shape; the latter changes the score.
        volume, raw, kept, units = F(12), 8, 3, 5
        self.assertNotEqual(volume / raw * F(units, 2), volume / kept * F(units, 2))


if __name__ == "__main__":
    unittest.main()
