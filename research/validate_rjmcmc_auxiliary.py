#!/usr/bin/env python3
"""Exact finite-state auxiliary-target checks and continuous RJ map audits.

This is a correctness calculation, not a protein simulation or a mixing
benchmark. It requires only Python's standard library. The finite model-index
chain tests joint-target normalization; changing a finite label does not by
itself test a continuous dimension-changing Gaussian-mixture implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def normalize(values):
    total = sum(values)
    return [value/total for value in values]


def stationary(matrix):
    n = len(matrix)
    vector = [1/n]*n
    for iteration in range(100000):
        new = [sum(vector[i]*matrix[i][j] for i in range(n)) for j in range(n)]
        if max(abs(a-b) for a, b in zip(new, vector)) < 2e-16:
            return new, iteration+1
        vector = new
    raise AssertionError('Power iteration did not converge')


def finite_chain(target, physical, proposal, model_proposal, omit_auxiliary=False):
    nx, nt = len(physical), len(proposal)
    matrix = [[0. for _ in range(nx*nt)] for _ in range(nx*nt)]
    for x in range(nx):
        for t in range(nt):
            i = x*nt+t
            for y in range(nx):
                j = y*nt+t
                target_ratio = physical[y]/physical[x] if omit_auxiliary else target[j]/target[i]
                acceptance = min(1., target_ratio*proposal[t][x]/proposal[t][y])
                probability = .6*proposal[t][y]
                matrix[i][j] += probability*acceptance
                matrix[i][i] += probability*(1-acceptance)
            for u in range(nt):
                probability = .4*model_proposal[x][t][u]
                if probability == 0:
                    continue
                j = x*nt+u
                acceptance = min(1., target[j]/target[i]*model_proposal[x][u][t]/model_proposal[x][t][u])
                matrix[i][j] += probability*acceptance
                matrix[i][i] += probability*(1-acceptance)
    return matrix


def assess(matrix, intended, nx, nt):
    observed, iterations = stationary(matrix)
    marginal = [sum(observed[x*nt:(x+1)*nt]) for x in range(nx)]
    return dict(stationary_distribution=observed, physical_marginal=marginal,
        power_iterations=iterations,
        maximum_row_sum_error=max(abs(sum(row)-1) for row in matrix),
        maximum_stationarity_error=max(abs(sum(intended[i]*matrix[i][j] for i in range(nx*nt))-intended[j]) for j in range(nx*nt)),
        maximum_detailed_balance_error=max(abs(intended[i]*matrix[i][j]-intended[j]*matrix[j][i]) for i in range(nx*nt) for j in range(nx*nt)),
        transition_matrix=matrix)


def finite_checks():
    physical = [.2, .3, .5]
    scores = [[8., 1., 1., .5], [1., 2., 1., .5], [1., 1., 6., .5]]
    conditional = [normalize(row) for row in scores]
    nt, nx = 4, 3
    components = [1, 1, 2, 3]
    proposal = [[.9, .08, .02], [.02, .95, .03], [.15, .2, .65], [.33, .34, .33]]
    # X-guided, asymmetric model proposals: valid only with their reverse ratio.
    model_proposal = [[normalize([scores[x][u]**.7 if t != u and abs(components[t]-components[u]) <= 1 else 0.
                                   for u in range(nt)]) for t in range(nt)] for x in range(nx)]
    normalized = [physical[x]*conditional[x][t] for x in range(nx) for t in range(nt)]
    wrong = normalize([physical[x]*scores[x][t] for x in range(nx) for t in range(nt)])
    independent = [physical[x]/nt for x in range(nx) for t in range(nt)]
    cases = {}
    for name, target, omission in [('normalized_conditional', normalized, False),
                                    ('unnormalized_fit_factor', wrong, False),
                                    ('independent_model_target', independent, False),
                                    ('omitted_auxiliary_ratio', normalized, True)]:
        result = assess(finite_chain(target, physical, proposal, model_proposal, omission), target, nx, nt)
        result['maximum_physical_marginal_error'] = max(abs(a-b) for a, b in zip(result['physical_marginal'], physical))
        cases[name] = result
    for name in ('normalized_conditional', 'independent_model_target'):
        assert cases[name]['maximum_physical_marginal_error'] < 1e-12
        assert cases[name]['maximum_detailed_balance_error'] < 1e-14
    predicted_wrong = normalize([physical[x]*sum(scores[x]) for x in range(nx)])
    assert max(abs(a-b) for a, b in zip(predicted_wrong, cases['unnormalized_fit_factor']['physical_marginal'])) < 1e-12
    assert cases['omitted_auxiliary_ratio']['maximum_physical_marginal_error'] > .01
    return dict(physical_target=physical, model_component_counts=components,
        conditional_model_probabilities=conditional, unnormalized_row_integrals=[sum(row) for row in scores],
        predicted_wrong_marginal=predicted_wrong, cases=cases,
        scope='Exact finite-state balance/marginal calculation; finite K-labelled models are not a continuous Gaussian RJ implementation.')


def determinant(matrix):
    data = [row[:] for row in matrix]
    value = 1.
    n = len(data)
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(data[row][column]))
        if data[pivot][column] == 0:
            return 0.
        if pivot != column:
            data[pivot], data[column] = data[column], data[pivot]
            value = -value
        diagonal = data[column][column]
        value *= diagonal
        for row in range(column+1, n):
            ratio = data[row][column]/diagonal
            for j in range(column+1, n):
                data[row][j] -= ratio*data[column][j]
    return value


def numerical_jacobian(function, values, step=1e-6):
    dimension = len(values)
    matrix = [[0.]*dimension for _ in values]
    for column in range(dimension):
        plus, minus = list(values), list(values)
        plus[column] += step
        minus[column] -= step
        high, low = function(plus), function(minus)
        for row in range(dimension):
            matrix[row][column] = (high[row]-low[row])/(2*step)
    return matrix


def birth_checks(rng):
    maximum_error, maximum_inverse_error, count = 0., 0., 0
    for k in range(1, 8):
        weights = normalize([.2+rng.random() for _ in range(k)])
        alpha = .1+.5*rng.random()
        for insert in range(k+1):
            def transform(values):
                old = list(values[:-1])+[1-sum(values[:-1])]
                new = [(1-values[-1])*weight for weight in old]
                new.insert(insert, values[-1])
                return new[:-1]
            coordinates = weights[:-1]+[alpha]
            predicted = (1-alpha)**(k-1)
            measured = abs(determinant(numerical_jacobian(transform, coordinates)))
            maximum_error = max(maximum_error, abs(measured-predicted))
            new = transform(coordinates)
            new += [1-sum(new)]
            recovered_alpha = new.pop(insert)
            recovered = [weight/(1-recovered_alpha) for weight in new]
            maximum_inverse_error = max(maximum_inverse_error, abs(recovered_alpha-alpha),
                                        max(abs(a-b) for a, b in zip(recovered, weights)))
            count += 1
    assert maximum_error < 2e-8
    assert maximum_inverse_error < 1e-14
    return dict(cases=count, maximum_weight_jacobian_absolute_error=maximum_error,
        maximum_inverse_error=maximum_inverse_error,
        formula='abs J_birth = (1-alpha)^(K-1), relative to simplex free coordinates; new mean/covariance coordinate blocks are identities')


def mean(x):
    return math.sin(x)+.2*x


def scale(x):
    return .5+.2*math.cos(x)**2


def normal_log(x, mu, sigma):
    return -.5*((x-mu)/sigma)**2-math.log(sigma)-.5*math.log(2*math.pi)


def transport_checks(rng):
    max_cancel, max_jacobian, max_flux, max_inverse = 0., 0., 0., 0.
    for _ in range(200):
        x, y, eta = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        theta = mean(x)+scale(x)*eta
        next_theta = mean(y)+scale(y)*eta
        log_jacobian = math.log(scale(y)/scale(x))
        cancellation = normal_log(next_theta, mean(y), scale(y))-normal_log(theta, mean(x), scale(x))+log_jacobian
        max_cancel = max(max_cancel, abs(cancellation))
        ratio = normal_log(y, 0, 1)-normal_log(x, 0, 1)+normal_log(x, next_theta, .7)-normal_log(y, theta, .7)
        forward = normal_log(x, 0, 1)+normal_log(theta, mean(x), scale(x))+normal_log(y, theta, .7)+min(0., ratio)
        reverse = normal_log(y, 0, 1)+normal_log(next_theta, mean(y), scale(y))+normal_log(x, next_theta, .7)+min(0., -ratio)+log_jacobian
        max_flux = max(max_flux, abs(forward-reverse))
        def transform(values):
            old_x, old_theta, new_x = values
            return [new_x, mean(new_x)+scale(new_x)*(old_theta-mean(old_x))/scale(old_x), old_x]
        mapped = transform([x, theta, y])
        recovered = transform(mapped)
        max_inverse = max(max_inverse, max(abs(a-b) for a, b in zip(recovered, [x, theta, y])))
        actual_jacobian = abs(determinant(numerical_jacobian(transform, [x, theta, y])))
        max_jacobian = max(max_jacobian, abs(actual_jacobian-math.exp(log_jacobian)))
    assert max_cancel < 1e-12 and max_flux < 1e-12 and max_inverse < 1e-12
    assert max_jacobian < 2e-8
    return dict(cases=200, maximum_auxiliary_log_density_jacobian_cancellation_error=max_cancel,
        maximum_log_flux_error=max_flux, maximum_inverse_error=max_inverse,
        maximum_full_map_jacobian_absolute_error=max_jacobian,
        scope='Continuous nonlinear state-conditioned Gaussian auxiliary; joint pose/model proposal retains the latent eta and uses the proposed endpoint model in the reverse density.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('rjmcmc-validation.json'))
    args = parser.parse_args()
    rng = random.Random(20260919)
    result = dict(schema='rjmcmc-auxiliary-proof-check-v1', finite=finite_checks(),
                  birth_map=birth_checks(rng), state_correlated_transport=transport_checks(rng), passed=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, output=str(args.out.resolve()),
        physical_marginals={name: value['physical_marginal'] for name, value in result['finite']['cases'].items()},
        birth_map=result['birth_map'], state_correlated_transport=result['state_correlated_transport']), indent=2))


if __name__ == '__main__':
    main()
