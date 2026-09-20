#!/usr/bin/env python3
"""Independent algebra checks for importing a Gaussian conditional initial state.

Standard library only. This checks the initialization construction, not the
Rust runner or convergence of a protein simulation. No production files change.
"""
from __future__ import annotations

import json
import math
import random


def lower_product(lower):
    return [[sum(lower[i][k] * lower[j][k] for k in range(min(i, j) + 1))
             for j in range(len(lower))] for i in range(len(lower))]


def cholesky(covariance):
    n = len(covariance)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = covariance[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not math.isfinite(value):
                    raise ValueError('covariance must be strictly positive definite')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def solve(lower, residual):
    result = []
    for i, value in enumerate(residual):
        result.append((value - sum(lower[i][j] * result[j] for j in range(i))) / lower[i][i])
    return result


def matvec(lower, vector):
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in lower]


def normal_log(mean, covariance, value):
    lower = cholesky(covariance)
    residual = solve(lower, [x - mu for x, mu in zip(value, mean)])
    return (-len(mean) / 2 * math.log(2 * math.pi)
            - sum(math.log(lower[i][i]) for i in range(len(lower)))
            - sum(x * x for x in residual) / 2)


def logsumexp(values):
    high = max(values)
    return high + math.log(sum(math.exp(x - high) for x in values))


def mixture_log(components, value):
    return logsumexp([math.log(c['weight']) + normal_log(c['mean'], c['covariance'], value)
                     for c in components])


class BoundedCoordinates:
    """An independently specified version of the closure's parameter map."""
    def __init__(self, translation_scale=3., covariance_floor=.04, covariance_ceiling=4.):
        self.scale = translation_scale
        log_low = math.log(math.sqrt(covariance_floor) / 2)
        log_high = math.log(2 * math.sqrt(covariance_ceiling))
        self.log_mid = (log_high + log_low) / 2
        self.log_span = (log_high - log_low) / 2
        self.offdiag_bound = 2 * math.sqrt(covariance_ceiling)
        self.ridge = covariance_floor / 16

    @staticmethod
    def inverse(value):
        if not math.isfinite(value) or abs(value) >= 1:
            raise ValueError('parameter is outside the strict interior of the coordinate bounds')
        return math.atanh(value)

    def encode(self, components):
        result = []
        for component in components:
            result.extend(64 * self.inverse(x / self.scale / 64) for x in component['mean'])
            residual_covariance = [[component['covariance'][i][j] / self.scale**2
                                    - (self.ridge if i == j else 0.) for j in range(6)] for i in range(6)]
            lower = cholesky(residual_covariance)
            for i in range(6):
                for j in range(i + 1):
                    value = lower[i][j]
                    result.append(self.log_span * self.inverse((math.log(value) - self.log_mid) / self.log_span)
                                  if i == j else self.offdiag_bound * self.inverse(value / self.offdiag_bound))
        baseline = components[-1]['weight']
        result.extend(24 * self.inverse(math.log(c['weight'] / baseline) / 24) for c in components[:-1])
        return result

    def decode(self, values, count):
        assert len(values) == 28 * count - 1
        components, offset = [], 0
        for _ in range(count):
            mean = [self.scale * 64 * math.tanh(v / 64) for v in values[offset:offset + 6]]
            offset += 6
            lower = [[0.] * 6 for _ in range(6)]
            for i in range(6):
                for j in range(i + 1):
                    value = values[offset]
                    lower[i][j] = self.scale * (math.exp(self.log_mid + self.log_span * math.tanh(value / self.log_span))
                                               if i == j else self.offdiag_bound * math.tanh(value / self.offdiag_bound))
                    offset += 1
            covariance = lower_product(lower)
            for i in range(6):
                covariance[i][i] += self.scale**2 * self.ridge
            components.append(dict(mean=mean, covariance=covariance))
        logits = [24 * math.tanh(v / 24) for v in values[offset:]] + [0.]
        normalizer = logsumexp(logits)
        for component, logit in zip(components, logits):
            component['weight'] = math.exp(logit - normalizer)
        return components


def initialization_checks(rng):
    mapping = BoundedCoordinates()
    coordinate_error, parameter_error, density_error, transport_error = 0., 0., 0., 0.
    cases = 0
    for count in (1, 2, 4):
        for _ in range(20):
            masses = [.1 + rng.random() for _ in range(count)]
            components = []
            for mass in masses:
                lower = [[0.] * 6 for _ in range(6)]
                for i in range(6):
                    lower[i][i] = 1. + rng.random()
                    for j in range(i):
                        lower[i][j] = rng.uniform(-.3, .3)
                components.append(dict(mean=[rng.uniform(-2., 2.) for _ in range(6)],
                                       covariance=lower_product(lower), weight=mass / sum(masses)))
            imported = mapping.encode(components)
            dimension = len(imported)
            center = [rng.uniform(-.2, .2) for _ in imported]
            next_center = [rng.uniform(-.2, .2) for _ in imported]
            # Include non-diagonal L: the initialization formula is a triangular
            # solve, not elementwise division except for a diagonal closure.
            scale = [[0.] * dimension for _ in imported]
            next_scale = [[0.] * dimension for _ in imported]
            for i in range(dimension):
                scale[i][i], next_scale[i][i] = rng.uniform(.2, .7), rng.uniform(.2, .7)
                for j in range(max(0, i - 2), i):
                    scale[i][j], next_scale[i][j] = rng.uniform(-.03, .03), rng.uniform(-.03, .03)
            eta = solve(scale, [v - f for v, f in zip(imported, center)])
            recovered = [f + residual for f, residual in zip(center, matvec(scale, eta))]
            coordinate_error = max(coordinate_error, max(abs(a - b) for a, b in zip(imported, recovered)))
            restored = mapping.decode(recovered, count)
            for old, new in zip(components, restored):
                pairs = list(zip(old['mean'], new['mean'])) + [(old['weight'], new['weight'])]
                pairs += [(old['covariance'][i][j], new['covariance'][i][j])
                          for i in range(6) for j in range(6)]
                parameter_error = max(parameter_error, max(abs(a - b) for a, b in pairs))
            for _ in range(10):
                point = [rng.uniform(-4., 4.) for _ in range(6)]
                density_error = max(density_error, abs(mixture_log(components, point) - mixture_log(restored, point)))
            transported = [f + residual for f, residual in zip(next_center, matvec(next_scale, eta))]
            roundtrip = solve(next_scale, [v - f for v, f in zip(transported, next_center)])
            transport_error = max(transport_error, max(abs(a - b) for a, b in zip(eta, roundtrip)))
            cases += 1
    assert coordinate_error < 2e-14 and parameter_error < 2e-12
    assert density_error < 2e-11 and transport_error < 2e-13
    return dict(cases=cases, max_coordinate_error=coordinate_error, max_parameter_error=parameter_error,
                max_mixture_log_density_error=density_error, max_transport_residual_error=transport_error)


def angular_metric_check():
    mean = [.4, -.2, .1, .3, -.1, .2]
    lower = [[0.] * 6 for _ in range(6)]
    for i in range(6):
        lower[i][i] = .7 + .1 * i
        for j in range(i):
            lower[i][j] = .03 * (i - j)
    covariance = lower_product(lower)
    old_length, new_length = 5., 11.
    affine = [1.] * 3 + [new_length / old_length] * 3
    scaled_mean = [x * a for x, a in zip(mean, affine)]
    scaled_covariance = [[covariance[i][j] * affine[i] * affine[j] for j in range(6)] for i in range(6)]
    cayley = [.2, -.15, .1]
    old_point = [.8, -.3, .7] + [old_length * x for x in cayley]
    new_point = [x * a for x, a in zip(old_point, affine)]
    # The same Haar factor cancels; include the angular-length cubed factor.
    old_density = normal_log(mean, covariance, old_point) + 3 * math.log(old_length)
    new_density = normal_log(scaled_mean, scaled_covariance, new_point) + 3 * math.log(new_length)
    error = abs(old_density - new_density)
    assert error < 2e-13
    return dict(physical_log_density_error=error)


def nonlinear_chart_check():
    # Reanchoring a full three-dimensional N(0,I) Cayley law by an axial
    # rotation. Along c'=(0,0,y), c_z=(y+a)/(1-a*y), and the full 3D
    # coordinate Jacobian is (1+a*a)^2/(1-a*y)^4. A Gaussian log density
    # restricted to a line is quadratic, hence has zero third differences.
    a = .7
    values = [-.5 * ((y + a) / (1 - a * y)) ** 2
              + 2 * math.log1p(a * a) - 4 * math.log(abs(1 - a * y))
              for y in (0., .1, .2, .3)]
    third_difference = values[3] - 3 * values[2] + 3 * values[1] - values[0]
    assert abs(third_difference) > .001
    return dict(nonzero_log_density_third_difference=third_difference,
                conclusion='Rotational Cayley reanchoring need not preserve Gaussianity.')


def invalid_import_checks():
    mapping = BoundedCoordinates()
    rejected = 0
    for value in (-1., 1., float('inf'), float('nan')):
        try:
            mapping.inverse(value)
        except ValueError:
            rejected += 1
    try:
        cholesky([[1., 1.], [1., 1.]])
    except ValueError:
        rejected += 1
    assert rejected == 5
    return dict(rejected_invalid_coordinate_or_singular_covariance_cases=rejected)


def main():
    result = dict(initialization=initialization_checks(random.Random(19741)),
                  angular_metric=angular_metric_check(), nonlinear_chart=nonlinear_chart_check(),
                  invalid_imports=invalid_import_checks(),
                  scope='Independent formula checks; not a Rust import/runner integration test or convergence result.')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
