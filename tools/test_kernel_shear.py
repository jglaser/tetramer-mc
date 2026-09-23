"""Independent density and extended-map checks for the Euclidean prototype."""
from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from kernel_shear import KernelShear, WarpedGaussianChart, PairTrace, pair_move


def derivative(function, x, epsilon=2e-6):
    x = np.asarray(x, float)
    return np.column_stack([(np.asarray(function(x + epsilon * e)) -
                              np.asarray(function(x - epsilon * e))) / (2 * epsilon)
                             for e in np.eye(len(x))])


def charts():
    return [WarpedGaussianChart([1., -2.], [[1.2, 0.], [.3, .8]],
                KernelShear(2, [0], [1], [[-1.], [.7]], [[1.3], [-.9]], .8)),
            WarpedGaussianChart([-3., 2.], [[.7, 0.], [-.4, 1.8]],
                KernelShear(2, [1], [0], [[-.4], [1.4]], [[-.8], [1.5]], 1.1))]


class ShearTests(unittest.TestCase):
    def test_permuted_split_inverse_and_unit_jacobian(self):
        shear = KernelShear(5, [3, 0], [4, 1, 2], [[.1, -.7], [1., .2]],
                            [[1., -.5, .2], [.3, .8, -.4]], .7)
        for z in np.random.default_rng(22017).normal(size=(20, 5)):
            np.testing.assert_allclose(shear.inverse(shear.forward(z)), z, atol=1e-14)
            np.testing.assert_allclose(shear.forward(shear.inverse(z)), z, atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(derivative(shear.forward, z)), 1., places=8)
        np.testing.assert_array_equal(shear.encode(z), shear.forward(z))

    def test_empty_expansion_and_empty_partitions(self):
        z = [1., 2.]
        for shear in [KernelShear(2, [0], [1], [], []),
                      KernelShear(2, [1, 0], [], [[0., 1.]], [[]])]:
            np.testing.assert_array_equal(shear.forward(z), z)
        constant = KernelShear(2, [], [1, 0], [[], []], [[2., 1.], [-.5, 3.]])
        np.testing.assert_array_equal(constant.forward(z), [5., 3.5])
        np.testing.assert_array_equal(constant.inverse(constant.forward(z)), z)

    def test_frozen_parameters_copy_inputs(self):
        centers, coefficients = np.array([[.2]]), np.array([[.8]])
        mean, lower = np.array([1., 2.]), np.eye(2)
        shear = KernelShear(2, [0], [1], centers, coefficients)
        chart = WarpedGaussianChart(mean, lower, shear)
        expected = chart.decode([.4, .7])
        centers[:] = 100; coefficients[:] = 100; mean[:] = 100; lower[:] = 100
        np.testing.assert_array_equal(chart.decode([.4, .7]), expected)
        with self.assertRaises(FrozenInstanceError): shear.bandwidth = 4
        with self.assertRaises(TypeError): shear.coefficients[0][0] = 4
        with self.assertRaises(FrozenInstanceError): chart.shear = shear

    def test_invalid_parameters_and_inputs_fail(self):
        for args in [(2, [0], [0], [[0]], [[1]], 1),
                     (2, [.0], [1], [[0]], [[1]], 1),
                     (2, [0], [1], [[0]], [[1]], 0),
                     (2, [0], [1], [[0]], [[1]], math.inf),
                     (2, [0], [1], [[math.nan]], [[1]], 1),
                     (2, [0], [1], [[0]], [[1, 2]], 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError): KernelShear(*args)
        shear = charts()[0].shear
        for vector in [[0.], [0., math.inf], [[0., 0.]]]:
            with self.assertRaises(ValueError): shear.forward(vector)
        for lower in [[[1., 1.], [0., 1.]], [[1., 0.], [0., 0.]], [[1., 0.], [0., -1.]]]:
            with self.assertRaises(ValueError): WarpedGaussianChart([0., 0.], lower, shear)


class ChartTests(unittest.TestCase):
    def test_chart_inverse_and_jacobian(self):
        for chart in charts():
            for z in np.random.default_rng(8901).normal(size=(10, 2)):
                np.testing.assert_allclose(chart.encode(chart.decode(z)), z, atol=2e-14)
                det = np.linalg.det(derivative(chart.decode, z))
                self.assertAlmostEqual(math.log(abs(det)), chart.log_abs_determinant, places=8)

    def test_zero_coefficients_recover_full_gaussian(self):
        mean = [1., -2.]; lower = np.array([[1.2, 0.], [.7, .3]])
        chart = WarpedGaussianChart(mean, lower, KernelShear(2, [0], [1], [[1.]], [[0.]]))
        for x in np.random.default_rng(994).normal(size=(30, 2)):
            self.assertAlmostEqual(chart.log_density(x), multivariate_normal.logpdf(x, mean, lower @ lower.T), places=11)

    def test_analytic_curved_banana_density(self):
        # Exact finite-RBF curved ridge, not an approximate fitted quadratic.
        # X~N(0,1), Y|X~N(2 exp(-X²/2),1).
        chart = WarpedGaussianChart([0., 0.], np.eye(2),
                                   KernelShear(2, [0], [1], [[0.]], [[2.]]))
        for x, y in [(-2., 0.), (0., 2.), (.7, -.3), (1.2, 1.4)]:
            expected = -math.log(2 * math.pi) - .5 * (x*x + (y - 2*math.exp(-x*x/2))**2)
            self.assertAlmostEqual(chart.log_density([x, y]), expected, places=13)

    def test_independent_quadrature_normalizes_curved_density(self):
        chart = WarpedGaussianChart([.4, -.2], [[1.3, 0.], [.25, .7]],
                                   KernelShear(2, [0], [1], [[0.]], [[2.]]))
        # Integrate in physical x,y, rather than feeding transformed normal draws
        # back to the density. Gaussian tails beyond eight x standard deviations
        # have mass below 2e-15; all y is integrated adaptively.
        def marginal(x):
            return quad(lambda y: math.exp(chart.log_density([x, y])),
                        -np.inf, np.inf, epsabs=2e-9, limit=150)[0]
        mass = quad(marginal, .4-8*1.3, .4+8*1.3, epsabs=2e-8, limit=150)[0]
        self.assertAlmostEqual(mass, 1., places=7)


class InvolutionTests(unittest.TestCase):
    def test_stochastic_extended_involution_and_antisymmetry(self):
        bank = charts(); rng = np.random.default_rng(314159)
        for c in [0., -1., 1., -.6, .83]:
            for source, target in [(0, 1), (1, 0), (0, 0)]:
                for _ in range(12):
                    x = bank[source].decode(rng.normal(size=2)); noise = rng.normal(size=2)
                    step = pair_move(bank, x, PairTrace(source, target, noise), c)
                    reverse = pair_move(bank, step.position, step.inverse_trace, c)
                    np.testing.assert_allclose(reverse.position, x, atol=2e-13)
                    np.testing.assert_allclose(reverse.inverse_trace.noise, noise, atol=2e-13)
                    self.assertEqual((reverse.inverse_trace.source, reverse.inverse_trace.target), (source, target))
                    self.assertAlmostEqual(step.log_extended_jacobian, -reverse.log_extended_jacobian, places=12)
                    self.assertAlmostEqual(step.log_auxiliary_ratio, -reverse.log_auxiliary_ratio, places=11)
                    self.assertAlmostEqual(step.log_correction, -reverse.log_correction, places=11)

    def test_full_extended_jacobian_including_independent_limit(self):
        bank = charts(); v = np.array([.7, -.9, .2, 1.1])
        for c in [0., -1., 1., .45]:
            def update(q):
                s = pair_move(bank, q[:2], PairTrace(0, 1, q[2:]), c)
                return np.r_[s.position, s.inverse_trace.noise]
            computed = math.log(abs(np.linalg.det(derivative(update, v))))
            expected = pair_move(bank, v[:2], PairTrace(0, 1, v[2:]), c).log_extended_jacobian
            self.assertAlmostEqual(computed, expected, places=8)

    def test_independent_redraw_limit_matches_density_ratio(self):
        bank = charts(); x = [.8, -1.2]; noise = [.2, -.6]
        step = pair_move(bank, x, PairTrace(0, 1, noise), 0.)
        np.testing.assert_allclose(step.position, bank[1].decode(noise))
        self.assertAlmostEqual(step.log_correction,
                               bank[0].log_density(x) - bank[1].log_density(step.position), places=12)

    def test_transport_of_exact_chart_targets_has_unit_acceptance(self):
        bank = charts()
        for c in [0., -.9, 1., -1., .7]:
            step = pair_move(bank, [.8, -1.2], PairTrace(0, 1, [.2, -.6]), c)
            # Expanded target with equal chart-label mass and exact chart
            # densities; this is not a claim about an arbitrary physical target.
            correction = bank[1].log_density(step.position) - bank[0].log_density([.8, -1.2]) + step.log_correction
            self.assertAlmostEqual(correction, 0., places=12)

    def test_invalid_move_arguments_and_frozen_trace(self):
        bank = charts()
        for c in [1.1, math.nan, math.inf]:
            with self.assertRaises(ValueError): pair_move(bank, [0., 0.], PairTrace(0, 1, [0., 0.]), c)
        with self.assertRaises(ValueError): pair_move(bank, [0., 0.], PairTrace(0, 2, [0., 0.]), 0.)
        with self.assertRaises(ValueError): pair_move(bank, [0., 0.], PairTrace(0, 1, [0.]), 0.)
        noise = np.array([.2, -.6]); trace = PairTrace(0, 1, noise); noise[:] = 10
        self.assertEqual(trace.noise, (.2, -.6))
        with self.assertRaises(FrozenInstanceError): trace.source = 1


if __name__ == '__main__':
    unittest.main()
