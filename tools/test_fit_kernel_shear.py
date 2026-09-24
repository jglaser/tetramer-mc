"""Independent fitting checks; toy coordinates only, no physical sampling."""
import copy
import itertools
import json
import math
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from fit_kernel_shear import KernelMixture, fit_shears


def guide(alpha=0., means=None, covariances=None, weights=None):
    means = [np.zeros(6)] if means is None else means
    covariances = [np.eye(6) for _ in means] if covariances is None else covariances
    weights = [1./len(means)]*len(means) if weights is None else weights
    return dict(schema='defensive-latent-shell-guide-v1', region_sha256='0'*64,
        defensive_uniform_shell_probability=alpha,
        gaussian_components=[dict(weight=float(w), mean=np.asarray(m).tolist(),
                                  covariance=np.asarray(c).tolist())
                             for w, m, c in zip(weights, means, covariances)])


def curve_data(n, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 6))
    values[:, 3] += 2.*np.exp(-.5*np.sum(values[:, :3]**2, axis=1))
    return values


def curved_baseline():
    # Exact first two moments of B0 = independent N(0,1) + 2 exp(-|A|²/2).
    mean = np.zeros(6); mean[3] = 2./(2.**1.5)
    covariance = np.eye(6)
    covariance[3, 3] += 4.*(3.**(-1.5)-2.**(-3.))
    return KernelMixture.from_guide(guide(means=[mean], covariances=[covariance]), np.eye(6))


def independent_kernel(a, centers, bandwidth):
    return np.exp(-.5*np.sum(((a[:, None, :]-np.asarray(centers)[None, :, :])/bandwidth)**2, axis=2))


class KernelFitTests(unittest.TestCase):
    def test_affine_density_matches_original_guide_under_nonorthogonal_transform(self):
        lower = np.diag([.8, 1.1, .7, 1.4, .9, 1.2])
        lower[3, 0] = .4; lower[4, 2] = -.2; lower[5, 1] = .3
        means = [np.arange(6)/10., -np.arange(6)/8.]
        covariances = [lower@lower.T, np.diag([1.2, .9, .6, .8, 1.3, 1.1])]
        source = guide(.5, means, covariances, [.3, .7])
        transform = np.eye(6)[[3, 4, 5, 0, 1, 2]] @ lower
        model = KernelMixture.from_guide(source, transform)
        values = np.random.default_rng(5101).normal(size=(70, 6))
        values = np.vstack([values, np.zeros(6), np.ones(6)*8])
        volume = math.pi**3*4.**6/6.
        terms = np.column_stack([math.log(.5*w)+multivariate_normal.logpdf(values, m, c)
                                 for w, m, c in zip([.3, .7], means, covariances)])
        uniform = np.where(np.linalg.norm(values, axis=1) <= 4., math.log(.5/volume), -np.inf)
        expected = logsumexp(np.column_stack([uniform, terms]), axis=1)
        np.testing.assert_allclose(model.log_density(values), expected, atol=2e-12, rtol=0)
        self.assertTrue(np.isfinite(model.log_density(values[-1:]))[0],
                        'Gaussian tails must remain untruncated outside original R4')

    def test_weighted_regression_solves_the_stated_objective(self):
        model = KernelMixture.from_guide(guide(), np.eye(6))
        values = curve_data(600, 5201)
        log_weights = np.linspace(-2., 1., len(values)); log_weights[::17] = -np.inf
        ridge = .03
        fitted, diagnostics = fit_shears(model, values, log_weights, centers=11,
            bandwidth=.9, ridge=ridge, minimum_ess=64., seed=5202)
        chart = fitted.charts[0]; shear = chart.shear
        a, b = values[:, :3], values[:, 3:]
        kernel = independent_kernel(a, shear.centers, shear.bandwidth)
        w = np.exp(log_weights-logsumexp(log_weights))
        coefficients = np.asarray(shear.coefficients)
        expected = np.linalg.solve(kernel.T@(w[:, None]*kernel)+ridge*np.eye(kernel.shape[1]),
                                   kernel.T@(w[:, None]*b))
        np.testing.assert_allclose(coefficients, expected, atol=2e-11, rtol=2e-11)
        before = .5*np.sum(w*np.sum(b*b, axis=1))
        after = .5*np.sum(w*np.sum((b-kernel@coefficients)**2, axis=1))
        penalty = .5*ridge*np.sum(coefficients**2)
        self.assertLess(after+penalty, before)
        record = diagnostics['components'][0]
        self.assertTrue(record['fitted'])
        self.assertAlmostEqual(record['conditional_objective_before'], before, places=11)
        self.assertAlmostEqual(record['conditional_objective_after'], after, places=11)
        self.assertAlmostEqual(record['coefficient_penalty'], penalty, places=11)
        self.assertLess(record['normal_equation_max_residual'], 1e-10)

    def test_complete_mixture_responsibilities_enter_component_regression(self):
        means = [np.zeros(6), np.r_[np.ones(3)*.3, np.zeros(3)]]
        baseline = KernelMixture.from_guide(guide(.5, means=means, weights=[.7, .3]), np.eye(6))
        values = curve_data(900, 5301)
        log_weights = -.2*values[:, 0]**2
        fitted, diagnostics = fit_shears(baseline, values, log_weights, centers=9,
            bandwidth=1., ridge=.02, minimum_ess=64., seed=5302)
        log_q = baseline.log_density(values)
        weight = np.exp(log_weights-logsumexp(log_weights))
        for index, (base_chart, chart) in enumerate(zip(baseline.charts, fitted.charts)):
            component_logq = multivariate_normal.logpdf(values, means[index], np.eye(6))
            responsibility = np.exp(math.log(.5*baseline.weights[index])+component_logq-log_q)
            tau = np.sum(weight*responsibility)
            w = weight*responsibility/tau
            whitened = values-np.asarray(means[index])
            k = independent_kernel(whitened[:, :3], chart.shear.centers, chart.shear.bandwidth)
            expected = np.linalg.solve(k.T@(w[:, None]*k)+.02*np.eye(k.shape[1]),
                                       k.T@(w[:, None]*whitened[:, 3:]))
            np.testing.assert_allclose(chart.shear.coefficients, expected, atol=3e-11, rtol=3e-11)
            self.assertAlmostEqual(diagnostics['components'][index]['responsibility_mass'], tau, places=12)

    def test_curved_holdout_gain_over_exact_affine_moments(self):
        baseline = curved_baseline()
        training = curve_data(3000, 5401)
        # A distinct random stream is used only for scoring after fitting returns.
        fitted, _ = fit_shears(baseline, training, np.zeros(len(training)),
            centers=16, bandwidth=1., ridge=.01, minimum_ess=64., seed=5402)
        holdout = curve_data(6000, 5403)
        gain = fitted.log_density(holdout)-baseline.log_density(holdout)
        self.assertGreater(float(gain.mean()), .025)
        self.assertGreater(float(gain.mean()), 3*float(gain.std(ddof=1))/math.sqrt(len(gain)))

    def test_rotation_first_split_and_nonorthogonal_physical_jacobian(self):
        lower = np.diag([.9, 1.2, .8, 1.1, .7, 1.3]); lower[3, 0] = .4; lower[5, 1] = -.3
        transform = np.eye(6)[[3, 4, 5, 0, 1, 2]]@lower
        baseline = KernelMixture.from_guide(guide(covariances=[np.linalg.inv(transform)@np.linalg.inv(transform).T]), transform)
        y = curve_data(700, 5501); u = np.linalg.solve(transform, y.T).T
        fitted, _ = fit_shears(baseline, u, np.zeros(len(u)), centers=12,
            bandwidth=1., ridge=.01, minimum_ess=64., seed=5502)
        chart = fitted.charts[0]; base = baseline.charts[0]
        self.assertEqual(chart.shear.conditioning, (0, 1, 2))
        self.assertEqual(chart.shear.shifted, (3, 4, 5))
        z = np.array([.2, -.4, .1, .8, -.3, .5])
        old_u = np.linalg.solve(transform, base.decode(z))
        new_u = np.linalg.solve(transform, chart.decode(z))
        np.testing.assert_allclose(transform[:3]@new_u, transform[:3]@old_u, atol=2e-14)
        self.assertGreater(np.linalg.norm(new_u-old_u), 1e-3)
        epsilon = 2e-6
        def decode(point): return np.linalg.solve(transform, chart.decode(point))
        derivative = np.column_stack([(decode(z+epsilon*e)-decode(z-epsilon*e))/(2*epsilon)
                                      for e in np.eye(6)])
        expected = chart.log_abs_determinant-np.linalg.slogdet(transform)[1]
        self.assertAlmostEqual(np.linalg.slogdet(derivative)[1], expected, places=8)
        expected_logq = -.5*(6*math.log(2*math.pi)+z@z)-expected
        self.assertAlmostEqual(fitted.log_density(new_u[None, :])[0], expected_logq, places=11)

    def test_gaussian_null_has_exact_nonpositive_quadrature_gain(self):
        baseline = KernelMixture.from_guide(guide(), np.eye(6))
        training = np.random.default_rng(5601).normal(size=(512, 6))
        fitted, _ = fit_shears(baseline, training, np.zeros(len(training)), centers=12,
            bandwidth=1., ridge=.01, minimum_ess=64., seed=5602)
        nodes, weights = np.polynomial.hermite.hermgauss(7)
        triples = list(itertools.product(range(len(nodes)), repeat=3))
        a = np.array([[math.sqrt(2)*nodes[i] for i in triple] for triple in triples])
        wa = np.array([math.prod(weights[i]/math.sqrt(math.pi) for i in triple) for triple in triples])
        # Conditional log-density difference is affine in B, so any symmetric
        # cubature with exactly zero B mean integrates it exactly.
        b = np.vstack([np.eye(3)*math.sqrt(3), -np.eye(3)*math.sqrt(3)])
        values = np.hstack([np.repeat(a, 6, axis=0), np.tile(b, (len(a), 1))])
        observed = float(np.sum(np.repeat(wa/6, 6)*(fitted.log_density(values)-baseline.log_density(values))))
        shear = fitted.charts[0].shear
        displacement = independent_kernel(a, shear.centers, shear.bandwidth)@np.asarray(shear.coefficients)
        expected = -.5*float(np.sum(wa*np.sum(displacement**2, axis=1)))
        self.assertLessEqual(observed, 1e-13)
        self.assertLess(expected, -1e-6, 'Null fit should retain a detectable amount of finite-training noise')
        self.assertAlmostEqual(observed, expected, places=12)

    def test_low_effective_sample_count_keeps_affine_component(self):
        baseline = KernelMixture.from_guide(guide(), np.eye(6))
        values = curve_data(100, 5701)
        log_weights = np.full(100, -100.); log_weights[0] = 0.
        fitted, diagnostics = fit_shears(baseline, values, log_weights,
            centers=16, minimum_ess=64., seed=5702)
        np.testing.assert_array_equal(fitted.log_density(values), baseline.log_density(values))
        self.assertFalse(diagnostics['components'][0]['fitted'])

    def test_pure_uniform_noop_preserves_zero_weight_exterior_rows(self):
        baseline = KernelMixture.from_guide(guide(1.), np.eye(6))
        values = np.zeros((70, 6)); values[-1] = 8.
        log_weights = np.zeros(70); log_weights[-1] = -np.inf
        fitted, diagnostics = fit_shears(baseline, values, log_weights)
        np.testing.assert_array_equal(fitted.log_density(values), baseline.log_density(values))
        self.assertEqual(diagnostics['training_weighted_log_density_gain'], 0.)
        self.assertEqual(diagnostics['training_rows'], 70)
        self.assertEqual(diagnostics['training_positive_weight_rows'], 69)
        self.assertFalse(diagnostics['components'][0]['fitted'])

    def test_weight_sum_overflow_cannot_create_subnormalized_proposal(self):
        source = guide(.5, means=[np.zeros(6), np.ones(6)], weights=[1e308, 1e308])
        try:
            model = KernelMixture.from_guide(source, np.eye(6))
        except ValueError:
            return  # An explicit invalid-normalization failure is acceptable.
        self.assertAlmostEqual(sum(model.weights), 1., places=14)
        self.assertTrue(all(math.isfinite(w) and w > 0. for w in model.weights))
        self.assertTrue(np.isfinite(model.log_density(np.ones((1, 6))*8))[0])

    def test_zero_weight_rows_do_not_train_the_map(self):
        baseline = KernelMixture.from_guide(guide(.5), np.eye(6))
        values = curve_data(400, 5751)
        fitted, _ = fit_shears(baseline, values, np.zeros(len(values)), centers=8, seed=5752)
        augmented = np.vstack([values, np.full((20, 6), 1e6)])
        weights = np.r_[np.zeros(len(values)), np.full(20, -np.inf)]
        second, diagnostics = fit_shears(baseline, augmented, weights, centers=8, seed=5752)
        scoring = curve_data(30, 5753)
        np.testing.assert_allclose(second.log_density(scoring), fitted.log_density(scoring), atol=2e-12, rtol=0)
        self.assertEqual(diagnostics['training_rows'], 420)
        self.assertEqual(diagnostics['training_positive_weight_rows'], 400)

    def test_serialization_determinism_and_no_input_memory(self):
        source = guide(.5); saved_source = copy.deepcopy(source)
        baseline = KernelMixture.from_guide(source, np.eye(6))
        training = curve_data(600, 5801); log_weights = np.zeros(600)
        original_training = training.copy()
        fitted, _ = fit_shears(baseline, training, log_weights, centers=8, seed=5802)
        second, _ = fit_shears(baseline, original_training, log_weights.copy(), centers=8, seed=5802)
        self.assertEqual(fitted.to_dict(), second.to_dict())
        self.assertEqual(source, saved_source)
        serialized = json.loads(json.dumps(fitted.to_dict(), allow_nan=False))
        recovered = KernelMixture.from_dict(serialized)
        scoring = curve_data(30, 5803)
        expected = fitted.log_density(scoring).copy()
        np.testing.assert_array_equal(recovered.log_density(scoring), expected)
        training[:] = 1e5; log_weights[:] = -100.; source['gaussian_components'][0]['mean'][0] = 999.
        np.testing.assert_array_equal(fitted.log_density(scoring), expected)
        np.testing.assert_array_equal(baseline.log_density(scoring),
            KernelMixture.from_guide(saved_source, np.eye(6)).log_density(scoring))


if __name__ == '__main__':
    unittest.main()
