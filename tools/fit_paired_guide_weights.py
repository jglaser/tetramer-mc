"""Pure fixed-dictionary minimax allocation of a defensive Gaussian proposal.

This module reads no files and performs no physical sampling or classification.
Callers supply complete component densities, original source densities, explicit
protected groups, and logs of the TWO cloud weights J W_k / q_source.  Missing
attempts may only be known zero contributions: total_attempts is the ORIGINAL
unconditional denominator, never the number of valid or selected rows.

The paired objective estimates the physical second moment, assuming conditional
independence and unbiasedness of the supplied cloud estimators.  The noisy
objective estimates the second moment of their implemented arithmetic average.
Neither an optimizer success nor these saved-sample moments establish unseen
mode coverage, physical convergence, or Markov-chain sampling efficiency.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _log_array(value, shape, name, *, finite=False):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape, f"Wrong {name} shape")
    good = np.isfinite(result) if finite else ~(np.isnan(result) | np.isposinf(result))
    _require(np.all(good), f"Invalid {name}: require finite logs or exact negative infinity zeros")
    return result


@dataclass(frozen=True)
class FixedDictionaryProblem:
    """Row-scaled positive affine densities and normalized moment coefficients.

    objective_and_gradient returns one M2(candidate)/M2(reference) ratio per
    explicitly supplied protected group and its derivative with respect to ALL
    Gaussian weights. The simplex constraint is imposed by fit_weights, not by
    this derivative. Row scaling cancels analytically and avoids enormous
    density ratios when reference weights happen to be zero.
    """
    component_density: np.ndarray
    uniform_density: np.ndarray
    coefficients: np.ndarray
    reference_weights: np.ndarray
    group_names: tuple[str, ...]
    log_reference_moments: tuple[float, ...]
    group_rows: tuple[int, ...]
    contributing_rows: tuple[int, ...]
    total_attempts: int
    supplied_rows: int
    moment: str
    alpha: float
    floor: float

    @property
    def component_count(self):
        return self.component_density.shape[1]

    def objective_and_gradient(self, weights):
        weights = np.asarray(weights, dtype=float)
        _require(weights.shape == (self.component_count,) and np.isfinite(weights).all(),
                 "Invalid candidate weight vector")
        q = self.uniform_density + self.component_density @ weights
        active = np.any(self.coefficients > 0., axis=0)
        _require(np.all(np.isfinite(q)) and np.all(q[active] > 0.),
                 "Candidate density must be positive on every contributing row")
        inverse = np.zeros_like(q)
        inverse[active] = 1. / q[active]
        ratios = self.coefficients @ inverse
        gradient = -(self.coefficients * inverse**2) @ self.component_density
        _require(np.isfinite(ratios).all() and np.isfinite(gradient).all(),
                 "Unrepresentable objective or gradient")
        return ratios, gradient

    def metadata(self):
        return dict(moment=self.moment, alpha=self.alpha, floor=self.floor,
                    component_count=self.component_count, total_attempts=self.total_attempts,
                    supplied_rows=self.supplied_rows,
                    omitted_zero_attempts=self.total_attempts-self.supplied_rows,
                    groups=[dict(name=name, rows=rows, contributing_rows=contributing,
                                 log_reference_M2=log_moment)
                            for name, rows, contributing, log_moment in zip(
                                self.group_names, self.group_rows, self.contributing_rows,
                                self.log_reference_moments)])


def build_problem(*, log_component_density, log_uniform_density, log_source_density,
                  log_cloud_importance, groups: Mapping[str, np.ndarray], total_attempts,
                  reference_weights, moment="paired", alpha=.5, floor=1e-5):
    """Build an immutable objective; does not infer or select protected groups.

    Component density inputs are G_j and U BEFORE multiplying by 1-alpha/alpha.
    log_cloud_importance has shape (rows, 2) and is log(J W_k / q_source).
    All inputs use the same measure; rows may have different actual q_source.
    Duplicate rows remain separate attempts. Exact zero cloud weights use -inf.
    An entirely unobserved protected group is rejected, NEVER silently dropped.
    """
    components = np.asarray(log_component_density, dtype=float)
    _require(components.ndim == 2 and components.shape[0] > 0 and components.shape[1] > 0,
             "Need a nonempty row by component density matrix")
    n, m = components.shape
    _require(type(total_attempts) is int and total_attempts >= n,
             "total_attempts must retain every supplied and omitted zero attempt")
    _require(moment in ("paired", "noisy"), "Unknown moment; expected paired or noisy")
    _require(alpha == .5, "This fixed comparison retains exactly 50% uniform probability")
    _require(math.isfinite(floor) and floor > 0. and m*floor < 1., "Invalid component floor")
    components = _log_array(components, (n, m), "component densities")
    uniform = _log_array(log_uniform_density, (n,), "uniform density")
    source = _log_array(log_source_density, (n,), "source density", finite=True)
    clouds = _log_array(log_cloud_importance, (n, 2), "cloud importance")
    reference = np.asarray(reference_weights, dtype=float)
    _require(reference.shape == (m,) and np.isfinite(reference).all()
             and np.all(reference >= 0.) and abs(float(reference.sum())-1.) <= 1e-12,
             "Reference weights must be finite, nonnegative and normalized")
    _require(isinstance(groups, Mapping) and len(groups) > 0, "Missing explicit protected groups")
    masks = []
    for name, mask in groups.items():
        mask = np.asarray(mask)
        _require(isinstance(name, str) and bool(name) and mask.shape == (n,)
                 and mask.dtype.kind == "b", "Protected groups need named boolean row masks")
        masks.append(mask.copy())

    log_component_terms = math.log1p(-alpha) + components
    log_uniform_term = math.log(alpha) + uniform
    scale = np.maximum(np.max(log_component_terms, axis=1), log_uniform_term)
    # A density-free row can represent a physical zero outside this proposal's
    # domain. It is retained; any nonzero protected contribution there is rejected.
    empty_density = np.isneginf(scale)
    scale = np.where(empty_density, 0., scale)
    density = np.exp(log_component_terms-scale[:, None])
    constant = np.exp(log_uniform_term-scale)
    reference_density = constant+density@reference
    log_reference_density = np.full(n, -np.inf)
    positive_reference = reference_density > 0.
    log_reference_density[positive_reference] = np.log(reference_density[positive_reference])
    with np.errstate(over="ignore", invalid="ignore"):
        log_cloud_moment = (clouds.sum(axis=1) if moment == "paired" else
                            2.*(np.logaddexp(clouds[:, 0], clouds[:, 1])-math.log(2.)))
        terms = log_cloud_moment+source-scale
    _require(not (np.isnan(terms) | np.isposinf(terms)).any(), "Unrepresentable cloud moment")

    coefficients, log_moments, counts, contributing = [], [], [], []
    for name, mask in zip(groups, masks):
        active = mask & np.isfinite(terms)
        _require(active.any(), f"Unobserved protected group: {name}; cannot normalize or omit it")
        _require(np.all(positive_reference[active]),
                 f"Reference density has no support for protected group {name}")
        normalizer = float(logsumexp(terms[active]-log_reference_density[active]))
        coefficient = np.zeros(n)
        coefficient[active] = np.exp(terms[active]-normalizer)
        _require(np.isfinite(coefficient).all() and np.any(coefficient > 0.),
                 f"Unrepresentable normalized coefficients for {name}")
        coefficients.append(coefficient)
        log_moments.append(normalizer-math.log(total_attempts))
        counts.append(int(mask.sum()))
        contributing.append(int(active.sum()))
    problem = FixedDictionaryProblem(_readonly(density), _readonly(constant),
        _readonly(coefficients), _readonly(reference), tuple(groups), tuple(log_moments),
        tuple(counts), tuple(contributing), total_attempts, n, moment, alpha, float(floor))
    reference_ratios, _ = problem.objective_and_gradient(reference)
    _require(np.allclose(reference_ratios, 1., rtol=1e-11, atol=1e-13),
             "Reference moment reconstruction failed")
    return problem


def floor_simplex_roundoff(weights, *, floor=1e-5, tolerance=1e-8):
    """Correct only an already feasible numerical result, not a failed fit."""
    weights = np.asarray(weights, dtype=float)
    _require(weights.ndim == 1 and len(weights) > 0 and np.isfinite(weights).all(),
             "Invalid fitted weights")
    _require(math.isfinite(floor) and floor > 0. and len(weights)*floor < 1.
             and math.isfinite(tolerance) and tolerance > 0., "Invalid correction parameters")
    _require(abs(float(weights.sum())-1.) <= tolerance
             and float(weights.min()) >= floor-tolerance,
             "Fit is infeasible; floor-simplex correction only handles roundoff")
    excess = np.maximum(weights-floor, 0.)
    _require(float(excess.sum()) > 0., "No free simplex mass")
    corrected = floor+(1.-len(weights)*floor)*excess/float(excess.sum())
    _require(np.isfinite(corrected).all() and np.all(corrected >= floor)
             and abs(float(corrected.sum())-1.) <= 2e-13, "Correction failed")
    return corrected


@dataclass(frozen=True)
class FitResult:
    weights: tuple[float, ...]
    raw_weights: tuple[float, ...]
    initial_weights: tuple[float, ...]
    ratios: tuple[float, ...]
    group_names: tuple[str, ...]
    moment: str
    alpha: float
    floor: float
    iterations: int
    optimizer_message: str
    raw_epigraph: float

    def to_dict(self):
        return dict(success=True, weights=list(self.weights), raw_weights=list(self.raw_weights),
                    initial_weights=list(self.initial_weights), moment=self.moment,
                    alpha=self.alpha, floor=self.floor, iterations=self.iterations,
                    optimizer_message=self.optimizer_message, raw_epigraph=self.raw_epigraph,
                    normalized_maximum_ratio=max(self.ratios),
                    group_ratios=dict(zip(self.group_names, self.ratios)),
                    raw_weight_sum=sum(self.raw_weights), normalized_weight_sum=sum(self.weights),
                    minimum_weight=min(self.weights),
                    roundoff_L1_change=sum(abs(a-b) for a, b in zip(self.weights, self.raw_weights)))


def fit_weights(problem: FixedDictionaryProblem, *, initial_weights=None, maxiter=300,
                ftol=1e-9):
    """One deterministic epigraph SLSQP solve; no retries or failed-fit fallback."""
    _require(type(maxiter) is int and maxiter > 0 and math.isfinite(ftol) and ftol > 0.,
             "Invalid optimizer budget or tolerance")
    if initial_weights is None:
        # The old bank may have zero weights on added components. This is a
        # declared starting point, not repair of an optimizer's failed output.
        initial = problem.floor+(1.-problem.component_count*problem.floor)*problem.reference_weights
    else:
        initial = floor_simplex_roundoff(initial_weights, floor=problem.floor)
    ratios, _ = problem.objective_and_gradient(initial)
    x0 = np.r_[initial, float(max(ratios))+1e-9]
    m = problem.component_count
    cache = {}

    def fg(x):
        if "x" not in cache or not np.array_equal(cache["x"], x):
            values, jacobian = problem.objective_and_gradient(x[:m])
            cache.update(x=x.copy(), values=values, jacobian=jacobian)
        return cache["values"], cache["jacobian"]

    result = minimize(lambda x: x[-1], x0,
        jac=lambda x: np.r_[np.zeros(m), 1.], method="SLSQP",
        bounds=[(problem.floor, 1.)]*m+[(0., None)],
        constraints=[dict(type="eq", fun=lambda x: x[:m].sum()-1.,
                          jac=lambda x: np.r_[np.ones(m), 0.]),
                     dict(type="ineq", fun=lambda x: x[-1]-fg(x)[0],
                          jac=lambda x: np.column_stack((-fg(x)[1], np.ones(len(problem.group_names)))))],
        options=dict(maxiter=maxiter, ftol=ftol, disp=False))
    _require(bool(result.success), f"Optimizer failed; no candidate produced: {result.message}")
    raw = np.asarray(result.x, dtype=float)
    _require(raw.shape == (m+1,) and np.isfinite(raw).all(), "Optimizer returned nonfinite result")
    raw_ratios, _ = problem.objective_and_gradient(raw[:m])
    _require(raw[-1] >= 0. and max(raw_ratios)-raw[-1] <= max(1e-8, 10.*ftol),
             "Optimizer claimed success with infeasible epigraph")
    weights = floor_simplex_roundoff(raw[:m], floor=problem.floor)
    ratios, _ = problem.objective_and_gradient(weights)
    return FitResult(tuple(map(float, weights)), tuple(map(float, raw[:m])),
        tuple(map(float, initial)), tuple(map(float, ratios)), problem.group_names, problem.moment,
        problem.alpha, problem.floor, int(result.nit), str(result.message), float(raw[-1]))


def render_guide(guide, result: FitResult):
    """Return a copied legacy guide with weights changed and geometry untouched.

    This helper performs no output/promotion. The caller remains responsible for
    frozen provenance, independent validation and conditional-cloud obligations.
    """
    _require(isinstance(result, FitResult), "Need a successful FitResult")
    _require(guide.get("schema") == "defensive-latent-shell-guide-v1"
             and guide.get("defensive_uniform_shell_probability") == result.alpha == .5,
             "Guide schema or defensive probability differs")
    components = guide.get("gaussian_components")
    _require(isinstance(components, list) and len(components) == len(result.weights),
             "Guide dictionary size differs")
    checked = floor_simplex_roundoff(result.weights, floor=result.floor)
    _require(np.allclose(checked, result.weights, rtol=0., atol=2e-13), "Invalid result weights")
    copied = deepcopy(guide)
    for component, weight in zip(copied["gaussian_components"], result.weights):
        component["weight"] = weight
    return copied
