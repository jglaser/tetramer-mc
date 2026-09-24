"""Analytic Gaussian controls for frozen additive RBF mixture components.

No data, refitting, quadrature, or random samples enter construction. Each
replacement Gaussian has exactly the mathematical first two moments of its
nonlinear component. This is the forward-KL best Gaussian for that component,
not the best Gaussian for a physical target or the best Gaussian mixture.
"""
from __future__ import annotations

import math
import numpy as np

from fit_kernel_shear import KernelMixture
from kernel_shear import KernelShear, WarpedGaussianChart


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _shear_moments(shear):
    """Return E[S(Z)] and Cov[S(Z)] for independent standard normal Z."""
    d = shear.dimension
    a, b = list(shear.conditioning), list(shear.shifted)
    p, q, count = len(a), len(b), len(shear.centers)
    mean, covariance = np.zeros(d), np.eye(d)
    if not count or not q:
        return mean, covariance
    centers = np.asarray(shear.centers).reshape(count, p)
    coefficients = np.asarray(shear.coefficients).reshape(count, q)
    h2 = shear.bandwidth*shear.bandwidth
    require(math.isfinite(h2) and h2 > 0., 'Squared bandwidth is not representable')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        first = (h2/(h2+1.))**(.5*p)*np.exp(-np.sum(centers*centers, axis=1)/(2.*(h2+1.)))
        differences = centers[:, None, :]-centers[None, :, :]
        midpoints = .5*centers[:, None, :]+.5*centers[None, :, :]
        second = (h2/(h2+2.))**(.5*p)*np.exp(
            -np.sum(differences*differences, axis=2)/(4.*h2)
            -np.sum(midpoints*midpoints, axis=2)/(h2+2.))
        mean[b] = first@coefficients
        kernel_covariance = second-np.outer(first, first)
        covariance[np.ix_(b, b)] += coefficients.T@kernel_covariance@coefficients
        cross = (centers.T*(first/(h2+1.))[None, :])@coefficients
        covariance[np.ix_(a, b)] = cross
        covariance[np.ix_(b, a)] = cross.T
    require(np.isfinite(mean).all() and np.isfinite(covariance).all(),
            'Analytic moments are not representable')
    # Matrix products may differ by a few ulps across symmetric entries. This
    # symmetrization adds no regularizer, eigenvalue clipping, or fitted jitter.
    covariance = .5*(covariance+covariance.T)
    return mean, covariance


def moment_matched_mixture(model):
    """Replace each frozen shear by its analytic moment-matched Gaussian.

    For S(A,B)=(A,B+f(A)), E[S]=(0,E[f]); covariance blocks are I,
    E[A f(A)^T], and I+Cov(f). Existing component L maps these moments into
    y=T*u. The original-u defensive uniform component, T and weights are retained.
    The returned charts have zero shear. Non-PD numerical results fail explicitly.
    """
    require(isinstance(model, KernelMixture), 'Frozen KernelMixture required')
    charts, components = [], []
    transform = np.asarray(model.transform)
    inverse = np.linalg.solve(transform, np.eye(model.dimension))
    for index, chart in enumerate(model.charts):
        mean_s, covariance_s = _shear_moments(chart.shear)
        lower = np.asarray(chart.lower)
        mean_y = np.asarray(chart.mean)+lower@mean_s
        covariance_y = lower@covariance_s@lower.T
        covariance_y = .5*(covariance_y+covariance_y.T)
        require(np.isfinite(mean_y).all() and np.isfinite(covariance_y).all(),
                'Transformed component moments are not representable')
        zero_shear = KernelShear(chart.dimension, chart.shear.conditioning,
                                chart.shear.shifted, (), (), chart.shear.bandwidth)
        identity_covariance = np.array_equal(covariance_s, np.eye(chart.dimension))
        if identity_covariance:
            # Includes zero shear and a constant displacement. Preserve the exact
            # original factor instead of round-tripping it through Cholesky.
            matched_lower = lower
        else:
            try:
                matched_lower = np.linalg.cholesky(covariance_y)
            except np.linalg.LinAlgError as error:
                raise ValueError('Moment covariance is not numerically positive definite; no repair applied') from error
        matched = WarpedGaussianChart(mean_y, matched_lower, zero_shear)
        kl = matched.log_abs_determinant-chart.log_abs_determinant
        require(math.isfinite(kl) and kl >= -1e-9,
                'Moment-matched Gaussian entropy violates the analytic nonnegative KL identity')
        mean_u = inverse@mean_y
        covariance_u = inverse@covariance_y@inverse.T
        require(np.isfinite(mean_u).all() and np.isfinite(covariance_u).all(),
                'Original-coordinate moments are not representable')
        components.append(dict(index=index,
            conditioning_dimension=len(chart.shear.conditioning), centers=len(chart.shear.centers),
            shear_mean=mean_s.tolist(), shear_covariance=covariance_s.tolist(),
            chart_mean=mean_y.tolist(), chart_covariance=covariance_y.tolist(),
            original_coordinate_mean=mean_u.tolist(), original_coordinate_covariance=covariance_u.tolist(),
            covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(covariance_y).min()),
            forward_KL_component_to_matched_Gaussian=max(0., float(kl)),
            forward_KL_raw_roundoff=float(kl), covariance_regularization=0.))
        charts.append(matched)
    result = KernelMixture(model.alpha, model.radius, model.transform, model.weights, tuple(charts))
    return result, dict(schema='analytic-kernel-shear-moment-control-v1', components=components,
        means_and_covariances='Exact Gaussian expectations of finite RBF expansion; floating-point evaluation.',
        construction_data_rows=0, random_draws=0, refit_calls=0,
        uniform_component_unchanged=True, transform_unchanged=True,
        maximum_weight_normalization_roundoff=float(np.max(np.abs(np.asarray(result.weights)-model.weights))),
        component_forward_KL_identity='KL(component || its moment Gaussian) = logdet(L_matched)-logdet(L_original).',
        limitation='Forward-KL best Gaussian per frozen component only; not an optimized physical-target Gaussian, '
                   'not the best complete mixture, and not a sampling or physical-convergence result.')
