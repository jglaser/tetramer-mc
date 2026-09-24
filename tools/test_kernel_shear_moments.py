"""Independent quadrature and coordinate checks of analytic moment controls."""
import itertools
import math
import unittest

import numpy as np

from fit_kernel_shear import KernelMixture
from kernel_shear import KernelShear, WarpedGaussianChart
from kernel_shear_moments import moment_matched_mixture


def quadrature_moments(chart, order=48):
    # Full standard-normal integration; conditional B moments are polynomial
    # of degree two, hence two-point Gauss-Hermite is exact in those dimensions.
    one_dimensional = []
    for axis in range(chart.dimension):
        n = order if axis in chart.shear.conditioning else 2
        nodes, weights = np.polynomial.hermite.hermgauss(n)
        one_dimensional.append((math.sqrt(2.)*nodes, weights/math.sqrt(math.pi)))
    indices = list(itertools.product(*(range(len(x[0])) for x in one_dimensional)))
    probabilities = np.array([math.prod(one_dimensional[j][1][i] for j, i in enumerate(row)) for row in indices])
    values = np.array([chart.decode([one_dimensional[j][0][i] for j, i in enumerate(row)]) for row in indices])
    mean = probabilities@values
    centered = values-mean
    covariance = centered.T@(probabilities[:, None]*centered)
    return mean, covariance


def example_chart():
    return WarpedGaussianChart([.4, -.7, .2], [[1.2, 0., 0.], [-.3, .8, 0.], [.2, .1, 1.1]],
        KernelShear(3, (2, 0), (1,), [[.1, -.7], [1., .2], [-.8, .9]],
                    [[1.7], [-.6], [.8]], 1.1))


class MomentTests(unittest.TestCase):
    def test_exact_moments_against_independent_permuted_quadrature(self):
        chart = example_chart(); model = KernelMixture(.5, 4., np.eye(3), [1.], [chart])
        matched, diagnostics = moment_matched_mixture(model)
        mean, covariance = quadrature_moments(chart)
        gaussian = matched.charts[0]; lower = np.asarray(gaussian.lower)
        np.testing.assert_allclose(gaussian.mean, mean, atol=2e-12, rtol=0)
        np.testing.assert_allclose(lower@lower.T, covariance, atol=2e-11, rtol=0)
        self.assertEqual(gaussian.shear.centers, ())
        self.assertGreater(diagnostics['components'][0]['forward_KL_component_to_matched_Gaussian'], 0.)
        self.assertEqual(diagnostics['refit_calls'], 0)
        self.assertEqual(diagnostics['random_draws'], 0)

    def test_zero_shear_preserves_exact_chart_and_density(self):
        chart = WarpedGaussianChart([1., -.2], [[.7, 0.], [.3, 1.2]], KernelShear(2, [0], [1], [], []))
        model = KernelMixture(.5, 4., [[1.2, .1], [-.3, .8]], [1.], [chart])
        matched, diagnostics = moment_matched_mixture(model)
        self.assertEqual(matched.charts[0].mean, chart.mean)
        self.assertEqual(matched.charts[0].lower, chart.lower)
        points = np.array([[0., 0.], [1., 2.], [-3., .2], [10., -8.]])
        np.testing.assert_array_equal(matched.log_density(points), model.log_density(points))
        self.assertEqual(diagnostics['components'][0]['forward_KL_component_to_matched_Gaussian'], 0.)

    def test_zero_coefficients_and_constant_displacement(self):
        zero = WarpedGaussianChart([0., 0.], np.eye(2), KernelShear(2, [0], [1], [[.4]], [[0.]], .8))
        constant = WarpedGaussianChart([1., 2.], [[1.2, 0.], [.3, .7]],
            KernelShear(2, [], [1, 0], [[], []], [[1., -.2], [-.3, .9]], 1.))
        model = KernelMixture(.5, 4., np.eye(2), [.4, .6], [zero, constant])
        matched, diagnostics = moment_matched_mixture(model)
        np.testing.assert_array_equal(matched.charts[0].mean, zero.mean)
        np.testing.assert_array_equal(matched.charts[0].lower, zero.lower)
        expected_mean = np.asarray(constant.mean)+np.asarray(constant.lower)@np.array([.7, .7])
        np.testing.assert_allclose(matched.charts[1].mean, expected_mean, atol=2e-15)
        self.assertEqual(matched.charts[1].lower, constant.lower)
        self.assertEqual(diagnostics['components'][1]['forward_KL_component_to_matched_Gaussian'], 0.)

    def test_nonorthogonal_transform_and_component_translation(self):
        chart = example_chart(); transform = np.array([[1.1, .2, -.1], [.3, .8, .1], [0., -.2, 1.4]])
        model = KernelMixture(.37, 3.2, transform, [1.], [chart])
        matched, diagnostics = moment_matched_mixture(model)
        mean_y, covariance_y = quadrature_moments(chart)
        inverse = np.linalg.inv(transform)
        expected_mean, expected_covariance = inverse@mean_y, inverse@covariance_y@inverse.T
        record = diagnostics['components'][0]
        np.testing.assert_allclose(record['original_coordinate_mean'], expected_mean, atol=2e-12, rtol=0)
        np.testing.assert_allclose(record['original_coordinate_covariance'], expected_covariance, atol=2e-11, rtol=0)
        self.assertEqual(matched.transform, model.transform)
        self.assertEqual(matched.alpha, model.alpha)
        self.assertEqual(matched.radius, model.radius)
        shift = np.array([.7, -1.1, 2.3])
        shifted_chart = WarpedGaussianChart(np.asarray(chart.mean)+transform@shift, chart.lower, chart.shear)
        shifted, shifted_diagnostics = moment_matched_mixture(KernelMixture(.37, 3.2, transform, [1.], [shifted_chart]))
        np.testing.assert_allclose(np.asarray(shifted_diagnostics['components'][0]['original_coordinate_mean'])-record['original_coordinate_mean'], shift, atol=2e-14)
        np.testing.assert_allclose(shifted_diagnostics['components'][0]['original_coordinate_covariance'], record['original_coordinate_covariance'], atol=0, rtol=0)

    def test_covariance_positive_definite_and_shear_entropy_identity(self):
        chart = example_chart(); matched, diagnostics = moment_matched_mixture(KernelMixture(0., 4., np.eye(3), [1.], [chart]))
        record = diagnostics['components'][0]
        covariance_s = np.asarray(record['shear_covariance'])
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance_s) > 0))
        self.assertGreaterEqual(np.linalg.slogdet(covariance_s)[1], -1e-13)
        self.assertAlmostEqual(record['forward_KL_component_to_matched_Gaussian'],
            .5*np.linalg.slogdet(covariance_s)[1], places=12)
        self.assertEqual(record['covariance_regularization'], 0.)
        self.assertAlmostEqual(matched.charts[0].log_abs_determinant-chart.log_abs_determinant,
            record['forward_KL_component_to_matched_Gaussian'], places=12)

    def test_mixture_global_first_two_moments_are_preserved(self):
        first = example_chart()
        second = WarpedGaussianChart([-1., .3, .8], [[.8, 0., 0.], [.1, 1.3, 0.], [0., -.2, .7]],
            KernelShear(3, [0, 1], [2], [[.3, -.2], [-.5, .4]], [[.9], [-.4]], 1.2))
        model = KernelMixture(.5, 4., np.eye(3), [.25, .75], [first, second])
        matched, diagnostics = moment_matched_mixture(model)
        original_moments = [quadrature_moments(c, 40) for c in model.charts]
        original_mean = sum(w*m for w, (m, c) in zip(model.weights, original_moments))
        original_second = sum(w*(c+np.outer(m, m)) for w, (m, c) in zip(model.weights, original_moments))
        matched_mean = sum(w*np.asarray(c.mean) for w, c in zip(matched.weights, matched.charts))
        matched_second = sum(w*(np.asarray(c.lower)@np.asarray(c.lower).T+np.outer(c.mean, c.mean)) for w, c in zip(matched.weights, matched.charts))
        np.testing.assert_allclose(matched_mean, original_mean, atol=2e-11, rtol=0)
        np.testing.assert_allclose(matched_second, original_second, atol=2e-11, rtol=0)
        self.assertEqual(matched.alpha, model.alpha)
        self.assertEqual(matched.radius, model.radius)

    def test_unrepresentable_bandwidth_fails_without_covariance_repair(self):
        chart = WarpedGaussianChart([0., 0.], np.eye(2), KernelShear(2, [0], [1], [[0.]], [[1.]], 1e308))
        with self.assertRaisesRegex(ValueError, 'bandwidth'):
            moment_matched_mixture(KernelMixture(0., 4., np.eye(2), [1.], [chart]))


if __name__ == '__main__':
    unittest.main()
