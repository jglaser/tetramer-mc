"""Physical proposal densities and fixed-quota multiple importance moments.

The two strata are the complete native-format hybrid and a uniform latent
six-ball/shell. Internal GMM/cube/cover labels remain random within the hybrid.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_latent_region import shell_log_volume
from analyze_native_region_reference import logadd
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProposalDensities:
    """Evaluate both unconditional densities relative to d^3t times Haar.

    No hard/capture/q conditioning enters these densities. The caller validates
    that the latent region is a complete cover of the chosen physical target.
    """
    def __init__(self, config, guide_model, guide_description, cover_mixture, region):
        self.config = config
        self.description = guide_description
        self.beta = guide_description['weight']
        self.epsilon = guide_description['uniform_probability']
        require(0 < self.beta < 1 and 0 < self.epsilon <= 1, 'Invalid hybrid probabilities')
        self.fixed = config['fixed_poses'][guide_description['anchor_index']]
        require(self.fixed == guide_description['anchor_pose'], 'Guide coordinate anchor mismatch')
        self.guide = Density(guide_model)
        self.product = cover_mixture
        self.region = region
        self.chart = Density(region['gaussian_chart'])
        require(len(region['gaussian_chart']['anchors']) == 1, 'One uniform latent chart required')
        require(region['gaussian_chart']['weights'] == [1.], 'Uniform chart must have one unit weight')
        self.log_volume = shell_log_volume(region)
        self.lower = region.get('minimum_mahalanobis_radius', 0.)
        self.upper = region['mahalanobis_radius']
        self.center = np.asarray(config['capture_center'])
        self.lengths = np.asarray(guide_description['cube_lengths'])
        require(np.array_equal(self.lengths, [2*config['capture_radius']]*3), 'Cube dimensions changed')
        require(np.array_equal(self.center, guide_description['capture_center']), 'Cube center changed')
        require(abs(sum(self.product['weights'])-1) < 2e-12, 'Product mixture is not normalized')

    def evaluate(self, poses):
        """Return (log guide density, log latent-cover density, latent radius)."""
        require(len(poses) > 0, 'Empty pose batch')
        position, _, rotations = arrays(poses)
        product_terms = []
        for weight, cover in zip(self.product['weights'], self.product['covers']):
            require(weight > 0 and cover['volume'] > 0, 'Invalid product cover')
            rt, _, rr = arrays([cover['reference']])
            centroid = np.asarray(cover['centroid'])
            compensated = position-rt[0]+rotations@centroid-rr[0]@centroid
            angle = Rotation.from_matrix(rr[0].T@rotations).magnitude()
            inside = (np.linalg.norm(compensated, axis=1) <= cover['ball_radius']) & (angle <= cover['angle_cap'])
            product_terms.append(np.where(inside, math.log(weight/cover['volume']), -np.inf))
        product_log = logsumexp(np.asarray(product_terms), axis=0)
        gaussian, _, _ = self.guide.evaluate(relative_poses(poses, self.fixed))
        delta = position-self.center
        cube = np.all((delta >= -self.lengths/2) & (delta < self.lengths/2), axis=1)
        cube_log = np.where(cube, math.log(self.epsilon)-float(np.log(self.lengths).sum()), -np.inf)
        learned_log = gaussian+math.log1p(-self.epsilon) if self.epsilon < 1 else np.full(len(poses), -np.inf)
        inner = np.logaddexp(cube_log, learned_log)
        guide_log = np.logaddexp(math.log1p(-self.beta)+product_log, math.log(self.beta)+inner)

        chart_log, radii, _ = self.chart.evaluate(relative_poses(poses, self.region['fixed_neighbor']))
        radius = radii[:, 0]
        inside = np.isfinite(radius) & (radius >= self.lower) & (radius <= self.upper)
        cover_log = np.full(len(poses), -np.inf)
        # Only evaluate the cancellation on bounded chart coordinates. Far-away
        # or Cayley-seam points have zero cover density and need no Jacobian.
        log_j = -3*math.log(2*math.pi)-.5*radius[inside]**2-chart_log[inside]
        cover_log[inside] = -self.log_volume-log_j
        require(not np.isnan(guide_log).any() and not np.isnan(cover_log).any(), 'Nonfinite density arithmetic')
        return guide_log, cover_log, radius


def mixture_log_density(guide, cover, guide_count, cover_count):
    require(type(guide_count) is int and type(cover_count) is int and min(guide_count, cover_count) > 0,
            'Positive integer fixed quotas required')
    total = guide_count+cover_count
    return np.logaddexp(np.asarray(guide)+math.log(guide_count/total),
                        np.asarray(cover)+math.log(cover_count/total))


def log_difference_square(a, b):
    """log((exp(a)-exp(b))^2), including exact zero and zero numerators."""
    if a == b:
        return -math.inf
    if min(a, b) == -math.inf:
        return 2*max(a, b)
    return 2*(max(a, b)+math.log(-math.expm1(min(a, b)-max(a, b))))


@dataclass
class LogMoments:
    count: int = 0
    nonzero: int = 0
    total: float = -math.inf
    square: float = -math.inf
    maximum: float = -math.inf
    noise: float = -math.inf

    def add(self, value=-math.inf, cloud_pair=None):
        self.count += 1
        require(value == -math.inf or math.isfinite(value), 'Invalid log contribution')
        if value != -math.inf:
            self.nonzero += 1
            self.total = logadd(self.total, value)
            self.square = logadd(self.square, 2*value)
            self.maximum = max(self.maximum, value)
        if cloud_pair is not None:
            require(len(cloud_pair) == 2, 'Paired-cloud diagnostic requires exactly two clouds')
            self.noise = logadd(self.noise, log_difference_square(*cloud_pair)-math.log(4))

    def merge(self, other):
        self.count += other.count
        self.nonzero += other.nonzero
        self.total = logadd(self.total, other.total)
        self.square = logadd(self.square, other.square)
        self.maximum = max(self.maximum, other.maximum)
        self.noise = logadd(self.noise, other.noise)
        return self

    def centered_square(self):
        if self.nonzero == 0:
            return -math.inf
        ratio = min(0., 2*self.total-math.log(self.count)-self.square)
        if ratio == 0.:
            return -math.inf
        return self.square+math.log(-math.expm1(ratio))


def quota_summary(strata):
    """Stratified variance for fixed quotas; pooled weight ESS is only descriptive."""
    require(bool(strata) and all(s.count >= 2 for s in strata), 'At least two unconditional draws per stratum required')
    combined = LogMoments()
    variance_numerator = -math.inf
    for s in strata:
        combined.merge(s)
        centered = s.centered_square()
        variance_numerator = logadd(variance_numerator, centered+math.log(s.count/(s.count-1)))
    if combined.nonzero == 0:
        return {'draws': combined.count, 'nonzero': 0, 'logQ': None, 'weight_ESS': 0.,
                'stratified_RSE': None, 'log_variance_of_mean': None, 'maximum_fraction': None,
                'paired_cloud_variance_fraction': None, 'coverage': 'No observations are not a mass upper bound'}
    log_q = combined.total-math.log(combined.count)
    log_variance = variance_numerator-2*math.log(combined.count)
    noise_fraction = (math.exp(combined.noise-variance_numerator)
                      if math.isfinite(variance_numerator) and math.isfinite(combined.noise) else None)
    return {'draws': combined.count, 'nonzero': combined.nonzero, 'logQ': log_q,
            'weight_ESS': math.exp(2*combined.total-combined.square),
            'stratified_RSE': math.exp(.5*log_variance-log_q),
            'log_variance_of_mean': log_variance if math.isfinite(log_variance) else None,
            'maximum_fraction': math.exp(combined.maximum-combined.total),
            'paired_cloud_variance_fraction': noise_fraction,
            'variance_rule': 'sum_j n_j s_j^2 / N^2; invalid zeros retained within each fixed stratum',
            'ESS_scope': 'Pooled weight concentration only; not a stratified variance or MCMC mixing estimate',
            'coverage': 'Observed moments do not bound unseen physical weight'}


def signed_quota_difference(negative, positive, baseline_log_q=None):
    """MIS minus one source-only estimate, with uniform sign within each stratum.

    On baseline-family rows the difference contribution is nonpositive;
    on the other family's rows it is nonnegative. `negative` stores absolute
    contributions. Signs do not change either within-stratum variance.
    """
    variance = quota_summary([negative, positive])
    n = negative.count+positive.count
    sign = 0 if positive.total == negative.total else (1 if positive.total > negative.total else -1)
    log_abs = (.5*log_difference_square(positive.total, negative.total)-math.log(n)
               if sign else None)
    log_var = variance['log_variance_of_mean']
    return {'sign': sign, 'log_absolute_difference': log_abs,
            'log_variance_of_difference': log_var,
            'difference_in_observed_standard_errors': (sign*math.exp(log_abs-.5*log_var)
                if log_abs is not None and log_var is not None else (0. if not sign else None)),
            'relative_difference_to_component': (sign*math.exp(log_abs-baseline_log_q)
                if log_abs is not None and baseline_log_q is not None else (0. if not sign else None)),
            'scope': 'Paired MIS minus component-only estimate on shared rows; fixed-stratum covariance included. Observed variance is not a tail bound.'}
