#!/usr/bin/env python3
"""Deterministic, standard-library checks of a conditional-parameter bridge.

This is an independent scalar mathematical reference, NOT a production sampler
validation. The physical state has two labels, and the retained real parameter
has a normalized, state-dependent Gaussian law. No molecular geometry, implicit
depletion gate, Rust execution, or empirical mixing claim is covered here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path


PI = (0.23, 0.77)
MEAN = (-0.8, 1.15)
SIGMA = (0.65, 1.3)
LOG_2PI = math.log(2.0 * math.pi)


def normal_logpdf(value: float, mean: float, sigma: float) -> float:
    return -0.5 * ((value - mean) / sigma) ** 2 - math.log(sigma) - 0.5 * LOG_2PI


def log_r(state: int, parameter: float) -> float:
    return normal_logpdf(parameter, MEAN[state], SIGMA[state])


def log_g(parameter: float) -> tuple[float, float]:
    """Unequal, parameter-dependent proposal weights with stable logarithms."""
    logit = 0.6 + 0.7 * parameter
    log_normalizer = max(0.0, logit) + math.log1p(math.exp(-abs(logit)))
    return logit - log_normalizer, -log_normalizer


def simpson(function, low: float, high: float, panels: int = 20000) -> float:
    assert panels > 0 and panels % 2 == 0
    width = (high - low) / panels
    middle = math.fsum(
        (4.0 if index % 2 else 2.0) * function(low + width * index)
        for index in range(1, panels)
    )
    return width * (function(low) + middle + function(high)) / 3.0


def require_close(actual: float, expected: float, tolerance: float = 2e-11) -> float:
    error = abs(actual - expected)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(f"{actual!r} != {expected!r}: error {error} > {tolerance}")
    return error


def fixed_parameter_log_flow(x: int, y: int, parameter: float, corrected: bool) -> float:
    """Accepted off-diagonal flux for a lazy, G_u-reversible proposal."""
    assert x != y
    weights = log_g(parameter)
    # K_u(x,y) = (1-c) G_u(y), with fixed c. Its diagonal also includes c.
    log_proposal = math.log1p(-0.63) + weights[y]
    log_ratio = math.log(PI[y] / PI[x]) + weights[x] - weights[y]
    if corrected:
        log_ratio += log_r(y, parameter) - log_r(x, parameter)
    return math.log(PI[x]) + log_r(x, parameter) + log_proposal + min(0.0, log_ratio)


def check_normalized_joint() -> dict:
    low = min(m - 12.0 * s for m, s in zip(MEAN, SIGMA))
    high = max(m + 12.0 * s for m, s in zip(MEAN, SIGMA))
    conditional_integrals = [simpson(lambda u: math.exp(log_r(x, u)), low, high) for x in (0, 1)]
    errors = [require_close(value, 1.0) for value in conditional_integrals]
    marginals = [PI[x] * conditional_integrals[x] for x in (0, 1)]
    errors.extend(require_close(marginals[x], PI[x]) for x in (0, 1))
    require_close(sum(marginals), 1.0)
    return {
        "passed": True,
        "target": "Pi(x,u) = pi(x) Normal(u; mean[x], sigma[x]^2)",
        "physical_probabilities": PI,
        "conditional_integrals": conditional_integrals,
        "recovered_physical_marginals": marginals,
        "quadrature_domain": [low, high],
        "omitted_tail_bound_per_conditional": math.erfc(12.0 / math.sqrt(2.0)),
        "max_absolute_error": max(errors),
    }


def check_fixed_parameter_balance() -> dict:
    parameters = [-8.0 + index / 6.0 for index in range(97)]
    errors = [
        require_close(fixed_parameter_log_flow(0, 1, u, True), fixed_parameter_log_flow(1, 0, u, True))
        for u in parameters
    ]
    negative_errors = [
        abs(fixed_parameter_log_flow(0, 1, u, False) - fixed_parameter_log_flow(1, 0, u, False))
        for u in parameters
    ]
    low = min(m - 12.0 * s for m, s in zip(MEAN, SIGMA))
    high = max(m + 12.0 * s for m, s in zip(MEAN, SIGMA))
    corrected_flows = [
        simpson(lambda u: math.exp(fixed_parameter_log_flow(x, 1 - x, u, True)), low, high)
        for x in (0, 1)
    ]
    uncorrected_flows = [
        simpson(lambda u: math.exp(fixed_parameter_log_flow(x, 1 - x, u, False)), low, high)
        for x in (0, 1)
    ]
    # MH has kinked integrands; pointwise balance is the primary equality check.
    require_close(corrected_flows[0], corrected_flows[1])
    imbalance = abs(uncorrected_flows[0] - uncorrected_flows[1])
    if max(negative_errors) <= 1.0 or imbalance <= 1e-3:
        raise AssertionError("Omitting the auxiliary density ratio did not fail the negative control")
    return {
        "passed": True,
        "parameter_grid_size": len(parameters),
        "proposal": "K_u(x,y) = c delta_x(y) + (1-c) G_u(y), c=0.63",
        "log_acceptance_ratio": "log pi(y)/pi(x) + log G_u(x)/G_u(y) + log r(u|y)/r(u|x)",
        "max_pointwise_log_flux_error": max(errors),
        "integrated_forward_reverse_flows": corrected_flows,
        "omitted_auxiliary_factor_negative_control": {
            "expected_failure_observed": True,
            "max_pointwise_log_flux_discrepancy": max(negative_errors),
            "integrated_forward_reverse_flows": uncorrected_flows,
            "integrated_absolute_imbalance": imbalance,
        },
    }


def check_transported_parameter_balance() -> dict:
    errors, inverse_errors, flow_errors = [], [], []
    for x in (0, 1):
        y = 1 - x
        physical_forward = min(1.0, PI[y] / PI[x])
        physical_reverse = min(1.0, PI[x] / PI[y])
        for residual in (-5.0, -2.1, -0.2, 0.0, 0.75, 3.1, 5.0):
            old = MEAN[x] + SIGMA[x] * residual
            new = MEAN[y] + SIGMA[y] / SIGMA[x] * (old - MEAN[x])
            reverse = MEAN[x] + SIGMA[x] / SIGMA[y] * (new - MEAN[y])
            log_jacobian = math.log(SIGMA[y] / SIGMA[x])
            errors.append(require_close(log_r(y, new) - log_r(x, old) + log_jacobian, 0.0))
            inverse_errors.append(require_close(reverse, old))
            forward_flow = math.log(PI[x]) + log_r(x, old) + math.log(physical_forward)
            reverse_flow = math.log(PI[y]) + log_r(y, new) + math.log(physical_reverse) + log_jacobian
            flow_errors.append(require_close(forward_flow, reverse_flow))
    return {
        "passed": True,
        "transport": "u' = m(y) + sigma(y)/sigma(x) * (u-m(x))",
        "conditional_standard_deviations": SIGMA,
        "nonunit_transport_jacobians": [SIGMA[1] / SIGMA[0], SIGMA[0] / SIGMA[1]],
        "max_conditional_density_plus_jacobian_log_error": max(errors),
        "max_inverse_absolute_error": max(inverse_errors),
        "max_joint_log_flux_error": max(flow_errors),
        "physical_kernel": "symmetric two-state flip proposal with ordinary pi-Metropolis acceptance",
    }


def check_conditional_pcn_balance() -> dict:
    errors, variance_errors = [], []
    correlations = (-0.8, 0.0, 0.75, 0.99)
    residuals = (-3.1, -0.7, 0.3, 2.7)
    for x in (0, 1):
        for correlation in correlations:
            innovation_sigma = SIGMA[x] * math.sqrt(1.0 - correlation**2)
            variance_errors.append(require_close(correlation**2 * SIGMA[x]**2 + innovation_sigma**2, SIGMA[x]**2))
            for z in residuals:
                for w in residuals:
                    old = MEAN[x] + SIGMA[x] * z
                    new = MEAN[x] + SIGMA[x] * w
                    forward = log_r(x, old) + normal_logpdf(new, MEAN[x] + correlation * (old - MEAN[x]), innovation_sigma)
                    reverse = log_r(x, new) + normal_logpdf(old, MEAN[x] + correlation * (new - MEAN[x]), innovation_sigma)
                    errors.append(require_close(forward, reverse))
    return {
        "passed": True,
        "update": "u' = m(x) + c*(u-m(x)) + sigma(x)*sqrt(1-c^2)*xi; xi~N(0,1)",
        "correlations": correlations,
        "pairwise_flux_checks": len(errors),
        "max_log_flux_error": max(errors),
        "max_stationary_variance_error": max(variance_errors),
        "gibbs_refresh_included": 0.0 in correlations,
    }


def check_reference_decoder() -> dict:
    reference = {"mean": 37.25, "variance": 0.25**2, "weight": 1.0, "reciprocal": True}
    log_bound = math.log(2.0)

    def decode(mean_coordinate: float, covariance_coordinate: float) -> dict:
        if mean_coordinate == 0.0 and covariance_coordinate == 0.0:
            return reference
        result = reference.copy()
        result["mean"] += math.sqrt(reference["variance"]) * 3.0 * math.tanh(mean_coordinate / 3.0)
        transform = math.exp(log_bound * math.tanh(covariance_coordinate / log_bound))
        result["variance"] *= transform**2
        return result

    decoded = decode(0.0, 0.0)
    assert decoded is reference
    assert decoded["variance"].hex() == reference["variance"].hex()
    assert decoded["variance"] != 1.0
    modified = decode(0.2, 0.1)
    assert modified["variance"] > reference["variance"]
    assert modified["reciprocal"] and modified["weight"] == reference["weight"]
    # A prepared reference start under N(m(x),sigma(x)^2) need not be a Gibbs draw.
    prepared_residuals = [-MEAN[x] / SIGMA[x] for x in (0, 1)]
    start_errors = [require_close(MEAN[x] + SIGMA[x] * prepared_residuals[x], 0.0) for x in (0, 1)]
    return {
        "passed": True,
        "reference": reference,
        "zero_coordinates_return_same_reference_object": decoded is reference,
        "reference_variance_hex": reference["variance"].hex(),
        "nonzero_coordinate_example": modified,
        "prepared_zero_parameter_latent_residuals": prepared_residuals,
        "max_reference_initialization_error": max(start_errors),
        "prepared_reference_start_is_stationary_auxiliary_draw": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runs/conditional-parameter-bridge-20260925/validation.json")
    args = parser.parse_args()
    checks = {
        "normalized_conditional_and_physical_marginal": check_normalized_joint(),
        "held_parameter_learned_kernel": check_fixed_parameter_balance(),
        "model_independent_kernel_with_parameter_transport": check_transported_parameter_balance(),
        "conditional_parameter_refresh": check_conditional_pcn_balance(),
        "calibrated_reference_limit": check_reference_decoder(),
    }
    result = {
        "schema": "conditional-parameter-bridge-mathematical-reference-v1",
        "passed": all(check["passed"] for check in checks.values()),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deterministic": True,
        "random_draws": 0,
        "dependencies": "Python standard library only",
        "scope": "Independent scalar reference identities and negative controls; not production integration or molecular sampling validation.",
        "not_validated": ["Rust proposal implementation", "protein geometry and Jacobians", "implicit Poisson depletion gate", "cluster selection clock", "checkpoint integration", "oligomer fitting quality", "mixing or assembly"],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
