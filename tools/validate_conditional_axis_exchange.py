#!/usr/bin/env python3
"""Exact finite-state audit of the conditional-axis exchange correction.

This checks the auxiliary-MH algebra, including retained capped-rejection
failures. It does not validate the protein geometry, the physical GCA kernel,
the contact score, floating-point execution, or production mixing efficiency.
Run directly or with --self-test; no third-party dependencies are required.
"""

import argparse
from fractions import Fraction as F
import json


ZERO = F(0)
ONE = F(1)
PI = (F(1, 2), F(1, 3), F(1, 6))
BASE = ((F(1, 5), F(3, 10), F(1, 2)),) * len(PI)
POSITIVE = (
    (F(1, 8), F(3, 4), F(1, 3)),
    (F(4, 5), F(1, 5), F(3, 4)),
    (F(2, 5), F(7, 8), F(1, 6)),
)
BINARY = ((ONE, ZERO, ONE), (ONE, ONE, ZERO), (ZERO, ONE, ONE))


def zero_matrix(n):
    return [[ZERO for _ in range(n)] for _ in range(n)]


def fixed_axis_kernel(i, j, flow):
    """Set one symmetric physical flow and retain the rest as self-loops."""
    matrix = zero_matrix(len(PI))
    for x in range(len(PI)):
        matrix[x][x] = ONE
    matrix[i][j] = flow / PI[i]
    matrix[j][i] = flow / PI[j]
    matrix[i][i] -= matrix[i][j]
    matrix[j][j] -= matrix[j][i]
    return matrix


KERNELS = (
    fixed_axis_kernel(0, 1, F(1, 8)),
    fixed_axis_kernel(1, 2, F(1, 12)),
    fixed_axis_kernel(0, 2, F(1, 10)),
)


def errors(matrix):
    n = len(PI)
    return {
        "row_sum": max(abs(sum(row, ZERO) - ONE) for row in matrix),
        "stationarity": max(
            abs(sum((PI[x] * matrix[x][y] for x in range(n)), ZERO) - PI[y])
            for y in range(n)
        ),
        "detailed_balance": max(
            abs(PI[x] * matrix[x][y] - PI[y] * matrix[y][x])
            for x in range(n) for y in range(n)
        ),
    }


def assert_reversible(matrix):
    assert all(p >= ZERO for row in matrix for p in row)
    result = errors(matrix)
    assert all(value == ZERO for value in result.values()), result
    return result


def conditional_laws(scores, base, caps=None):
    """Return the exact successful draw subdensity and success probability.

    With caps=None the rejection sampler runs to acceptance. With caps[x]=M,
    it returns its first acceptance within M trials or an explicit failure.
    The same state-local routine must be used for forward and reverse draws.
    """
    probabilities = []
    success = []
    normalizers = []
    for x, row in enumerate(scores):
        assert sum(base[x], ZERO) == ONE
        assert all(p >= ZERO for p in base[x])
        assert all(ZERO <= h <= ONE for h in row)
        weights = [base[x][u] * row[u] for u in range(len(KERNELS))]
        z = sum(weights, ZERO)
        assert z > ZERO, "An exact conditional draw requires nonzero support"
        q = [w / z for w in weights]
        assert sum(q, ZERO) == ONE
        if caps is None:
            s = ONE
            subdensity = q
        else:
            attempts = caps[x]
            assert isinstance(attempts, int) and attempts >= 1
            s = ONE - (ONE - z) ** attempts
            # Independent enumeration of all possible first-success positions.
            subdensity = [
                sum(((ONE - z) ** k * w for k in range(attempts)), ZERO)
                for w in weights
            ]
            assert subdensity == [s * p for p in q]
        assert sum(subdensity, ZERO) == s
        probabilities.append(subdensity)
        success.append(s)
        normalizers.append(z)
    return probabilities, success, normalizers


def exchange_kernel(scores, base=BASE, caps=None):
    """Enumerate X, chosen axis u, GCA endpoint Y, and exchange auxiliary v."""
    draws, success, normalizers = conditional_laws(scores, base, caps)
    matrix = zero_matrix(len(PI))
    for x in range(len(PI)):
        matrix[x][x] += ONE - success[x]  # Failed forward draw is a self-loop.
        for u, q_x_u in enumerate(draws[x]):
            if q_x_u == ZERO:
                continue
            for y, k_xy in enumerate(KERNELS[u][x]):
                prefix = q_x_u * k_xy
                if prefix == ZERO:
                    continue
                # A failed draw at the proposed endpoint also retains X.
                matrix[x][x] += prefix * (ONE - success[y])
                for v, q_y_v in enumerate(draws[y]):
                    if q_y_v == ZERO:
                        continue
                    denominator = (
                        base[x][u] * scores[x][u]
                        * base[y][v] * scores[y][v]
                    )
                    numerator = (
                        base[y][u] * scores[y][u]
                        * base[x][v] * scores[x][v]
                    )
                    assert denominator > ZERO
                    acceptance = min(ONE, numerator / denominator)
                    probability = prefix * q_y_v
                    matrix[x][y] += probability * acceptance
                    matrix[x][x] += probability * (ONE - acceptance)

                    # Stronger local check: accepted extended flows agree
                    # with the SAME u and SAME v when X and Y are exchanged.
                    forward = PI[x] * probability
                    reverse = (
                        PI[y] * draws[y][u] * KERNELS[u][y][x]
                        * draws[x][v]
                    )
                    assert forward * acceptance == min(forward, reverse)
    return matrix, success, normalizers


def naive_conditional_kernel(scores):
    draws, _, _ = conditional_laws(scores, BASE)
    return [
        [sum((draws[x][u] * KERNELS[u][x][y]
              for u in range(len(KERNELS))), ZERO)
         for y in range(len(PI))]
        for x in range(len(PI))
    ]


def summary(name, matrix, success=None, normalizers=None):
    result = {
        "name": name,
        "exact_errors": {k: str(v) for k, v in errors(matrix).items()},
        "transition_matrix": [[str(p) for p in row] for row in matrix],
        "target_averaged_change_probability": str(sum(
            (PI[x] * (ONE - matrix[x][x]) for x in range(len(PI))), ZERO
        )),
    }
    if success is not None:
        result["auxiliary_draw_success_probabilities"] = [str(p) for p in success]
    if normalizers is not None:
        result["score_normalizers_for_validation_only"] = [str(z) for z in normalizers]
    return result


def self_test():
    checks = []
    for u, kernel in enumerate(KERNELS):
        assert_reversible(kernel)
        assert all(kernel[x][x] > ZERO for x in range(len(PI)))
        checks.append(summary("fixed_axis_%d" % u, kernel))

    state_dependent_base = (
        (F(1, 2), F(1, 3), F(1, 6)),
        (F(1, 6), F(1, 2), F(1, 3)),
        (F(1, 3), F(1, 6), F(1, 2)),
    )
    cases = (
        ("positive_exact", POSITIVE, BASE, None),
        ("positive_cap_1", POSITIVE, BASE, (1, 1, 1)),
        ("positive_cap_3", POSITIVE, BASE, (3, 3, 3)),
        ("positive_state_local_caps", POSITIVE, BASE, (1, 2, 4)),
        ("binary_exact", BINARY, BASE, None),
        ("binary_cap_1", BINARY, BASE, (1, 1, 1)),
        ("binary_cap_3", BINARY, BASE, (3, 3, 3)),
        ("state_dependent_base_exact", POSITIVE, state_dependent_base, None),
        ("state_dependent_base_caps", POSITIVE, state_dependent_base, (1, 2, 4)),
    )
    for name, scores, base, caps in cases:
        matrix, success, normalizers = exchange_kernel(scores, base, caps)
        assert_reversible(matrix)
        assert any(matrix[x][y] > ZERO for x in range(len(PI))
                   for y in range(len(PI)) if x != y), "Nontrivial transitions required"
        checks.append(summary(name, matrix, success, normalizers))

    controls = []
    for name, scores in (("positive", POSITIVE), ("binary", BINARY)):
        naive = naive_conditional_kernel(scores)
        residual = errors(naive)
        assert residual["row_sum"] == ZERO
        assert residual["stationarity"] > ZERO
        assert residual["detailed_balance"] > ZERO
        controls.append(summary("naive_uncorrected_" + name, naive))

    return {
        "passed": True,
        "arithmetic": "exact fractions throughout enumeration and assertions",
        "physical_target": [str(p) for p in PI],
        "reversible_cases": checks,
        "expected_failure_controls": controls,
        "limitations": [
            "Finite-state algebra audit; not a proof of the physical GCA kernel.",
            "Does not test geometry, floating-point execution, or protein mixing.",
            "Capped failures must remain self-loops; both auxiliary draws use the same state-local routine.",
            "The proposed endpoint auxiliary is independent conditional on that endpoint.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run the exact enumeration (also the default)")
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))


if __name__ == "__main__":
    main()
