"""Independent finite-space controls for fixed-dictionary proposal allocation."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fit_paired_guide_weights import (
    build_problem, fit_weights, floor_simplex_roundoff, render_guide,
)


def logs(values):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(values, float))


def discrete_inputs():
    # Exactly enumerate a two-point space under uniform source density. Target
    # integral is one; the attainable optimal proposal is target itself.
    return dict(log_component_density=logs([[1., 0.], [0., 1.]]),
                log_uniform_density=logs([.5, .5]), log_source_density=logs([.5, .5]),
                log_cloud_importance=logs([[1.2, 1.2], [.8, .8]]),
                groups={"whole": np.ones(2, bool)}, total_attempts=2,
                reference_weights=[.5, .5])


class FixedAllocationTests(unittest.TestCase):
    def test_known_finite_integral_optimum_and_cauchy_schwarz_bound(self):
        problem = build_problem(**discrete_inputs())
        self.assertAlmostEqual(math.exp(problem.log_reference_moments[0]), 1.04, places=14)
        result = fit_weights(problem)
        np.testing.assert_allclose(result.weights, [.7, .3], atol=3e-6, rtol=0.)
        self.assertAlmostEqual(result.ratios[0]*1.04, 1., places=10)
        for a in np.linspace(1e-5, 1.-1e-5, 31):
            q = np.array([.25+.5*a, .75-.5*a])
            independently_summed = np.sum(np.array([.6, .4])**2/q)
            ratios, _ = problem.objective_and_gradient([a, 1.-a])
            self.assertAlmostEqual(ratios[0]*1.04, independently_summed, places=13)
            self.assertGreaterEqual(independently_summed, 1.-1e-14)
        self.assertTrue(result.to_dict()["success"])

    def test_gradient_and_convexity_with_overlapping_explicit_groups(self):
        args = discrete_inputs()
        args["groups"] = {"whole": np.array([True, True]), "first": np.array([True, False])}
        p = build_problem(**args)
        x = np.array([.4, .6]); f, gradient = p.objective_and_gradient(x)
        for j in range(2):
            delta = np.zeros(2); delta[j] = 1e-6
            numerical = (p.objective_and_gradient(x+delta)[0]-p.objective_and_gradient(x-delta)[0])/2e-6
            np.testing.assert_allclose(gradient[:, j], numerical, rtol=2e-9, atol=2e-10)
        a, b = np.array([.13, .87]), np.array([.85, .15])
        for theta in [.1, .4, .9]:
            mid = p.objective_and_gradient(theta*a+(1-theta)*b)[0]
            chord = theta*p.objective_and_gradient(a)[0]+(1-theta)*p.objective_and_gradient(b)[0]
            self.assertTrue(np.all(mid <= chord+2e-14))
        result = fit_weights(p)
        self.assertLessEqual(max(result.ratios), max(f))

    def test_independent_cloud_noise_changes_noisy_but_not_physical_optimum(self):
        # Enumerate all four equally likely pairs of independent {0,2} clouds
        # at site 0, and four repetitions of deterministic clouds at site 1.
        cloud = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]]+[[1., 1.]]*4)
        args = dict(log_component_density=logs([[1., 0.]]*4+[[0., 1.]]*4),
                    log_uniform_density=logs([.5]*8), log_source_density=logs([.5]*8),
                    log_cloud_importance=logs(cloud), groups={"whole": np.ones(8, bool)},
                    total_attempts=8, reference_weights=[.5, .5])
        paired = build_problem(**args, moment="paired")
        noisy = build_problem(**args, moment="noisy")
        self.assertAlmostEqual(math.exp(paired.log_reference_moments[0]), 1., places=14)
        self.assertAlmostEqual(math.exp(noisy.log_reference_moments[0]), 1.25, places=14)
        rp, rn = fit_weights(paired), fit_weights(noisy)
        np.testing.assert_allclose(rp.weights, [.5, .5], atol=1e-9, rtol=0.)
        q0 = math.sqrt(1.5)/(math.sqrt(1.5)+1.)
        self.assertAlmostEqual(rn.weights[0], 2.*(q0-.25), places=5)
        self.assertGreater(rn.weights[0], rp.weights[0]+.09)

    def test_noiseless_objectives_are_identical(self):
        args = discrete_inputs()
        paired, noisy = build_problem(**args), build_problem(**args, moment="noisy")
        for weights in ([.1, .9], [.5, .5], [.7, .3]):
            fp, gp = paired.objective_and_gradient(weights)
            fn, gn = noisy.objective_and_gradient(weights)
            np.testing.assert_allclose(fp, fn, rtol=0., atol=3e-16)
            np.testing.assert_allclose(gp, gn, rtol=0., atol=4e-16)

    def test_duplicate_rows_and_zero_attempt_denominators(self):
        args = discrete_inputs(); original = build_problem(**args)
        duplicated = deepcopy(args)
        for name in ("log_component_density", "log_uniform_density", "log_source_density", "log_cloud_importance"):
            duplicated[name] = np.repeat(args[name], 2, axis=0)
        duplicated.update(groups={"whole": np.ones(4, bool)}, total_attempts=4)
        doubled = build_problem(**duplicated)
        np.testing.assert_allclose(original.log_reference_moments, doubled.log_reference_moments, atol=3e-16)
        explicit_zeros = deepcopy(args)
        explicit_zeros["log_component_density"] = np.r_[args["log_component_density"], [[-np.inf, -np.inf]]*2]
        explicit_zeros["log_uniform_density"] = np.r_[args["log_uniform_density"], [-np.inf]*2]
        explicit_zeros["log_source_density"] = np.r_[args["log_source_density"], [0.]*2]
        explicit_zeros["log_cloud_importance"] = np.r_[args["log_cloud_importance"], [[-np.inf, -np.inf]]*2]
        explicit_zeros.update(groups={"whole": np.ones(4, bool)}, total_attempts=4)
        all_rows = build_problem(**explicit_zeros)
        omitted_zeros = deepcopy(args); omitted_zeros["total_attempts"] = 4
        compact = build_problem(**omitted_zeros)
        self.assertAlmostEqual(all_rows.log_reference_moments[0], original.log_reference_moments[0]-math.log(2.), places=14)
        np.testing.assert_allclose(all_rows.log_reference_moments, compact.log_reference_moments, atol=0.)
        np.testing.assert_allclose(all_rows.objective_and_gradient([.3, .7])[0],
                                   compact.objective_and_gradient([.3, .7])[0], atol=0.)
        self.assertEqual(compact.metadata()["omitted_zero_attempts"], 2)

    def test_actual_source_density_is_used_for_each_row(self):
        # Enumerate source p=(1/4,3/4) with deterministic frequencies, not an
        # unweighted two-row shortcut. Its exact second moment is unchanged.
        args = discrete_inputs()
        args.update(log_component_density=logs([[1., 0.]]+[[0., 1.]]*3),
                    log_uniform_density=logs([.5]*4), log_source_density=logs([.25]+[.75]*3),
                    log_cloud_importance=logs([[.6/.25]*2]+[[.4/.75]*2]*3),
                    groups={"whole": np.ones(4, bool)}, total_attempts=4)
        p = build_problem(**args)
        for weights in ([.3, .7], [.7, .3]):
            q = .25+.5*np.array(weights)
            exact = np.sum(np.array([.6, .4])**2/q)
            self.assertAlmostEqual(p.objective_and_gradient(weights)[0][0]*1.04, exact, places=13)

    def test_density_row_scaling_handles_large_logs_and_zero_reference_components(self):
        args = discrete_inputs(); args["reference_weights"] = [1., 0.]
        base = build_problem(**args)
        shift = np.array([10000., -10000.])
        args["log_component_density"] += shift[:, None]
        args["log_uniform_density"] += shift
        args["log_source_density"] += shift
        scaled = build_problem(**args)
        np.testing.assert_allclose(scaled.log_reference_moments, base.log_reference_moments, atol=2e-12)
        np.testing.assert_allclose(scaled.objective_and_gradient([.2, .8])[0],
                                   base.objective_and_gradient([.2, .8])[0], atol=2e-12)
        result = fit_weights(scaled)
        self.assertGreaterEqual(min(result.weights), 1e-5)
        self.assertAlmostEqual(result.initial_weights[1], 1e-5)

    def test_arrays_and_result_are_independent_immutable_copies(self):
        args = discrete_inputs(); p = build_problem(**args)
        expected = p.objective_and_gradient([.3, .7])[0].copy()
        args["log_component_density"][:] = 0.
        np.testing.assert_array_equal(p.objective_and_gradient([.3, .7])[0], expected)
        with self.assertRaises(ValueError):
            p.coefficients[0, 0] = 0.
        result = fit_weights(p)
        with self.assertRaises(AttributeError):
            result.weights = (.5, .5)

    def test_exact_floor_and_roundoff_correction(self):
        corrected = floor_simplex_roundoff([.2-2e-12, .8+1e-12])
        self.assertAlmostEqual(float(corrected.sum()), 1., places=14)
        self.assertGreaterEqual(float(corrected.min()), 1e-5)
        boundary = floor_simplex_roundoff([1e-5, 1.-1e-5])
        self.assertEqual(boundary[0], 1e-5)
        np.testing.assert_allclose(boundary, [1e-5, 1.-1e-5], atol=2e-16, rtol=0.)
        for weights in ([.2, .7], [-.01, 1.01], [math.nan, 1.], [0., 1.]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                floor_simplex_roundoff(weights)

    def test_inactive_duplicate_components_retain_floor_and_geometry(self):
        args = discrete_inputs()
        args["log_component_density"] = np.column_stack((args["log_component_density"], [-np.inf]*2))
        args["reference_weights"] = [.5, .5, 0.]
        result = fit_weights(build_problem(**args))
        self.assertGreaterEqual(result.weights[2], 1e-5)
        self.assertLess(result.weights[2], 1.01e-5)
        guide = dict(schema="defensive-latent-shell-guide-v1", region_sha256="frozen",
                     defensive_uniform_shell_probability=.5,
                     gaussian_components=[dict(weight=1/3, mean=[i, 2.], covariance=[[2., .1], [.1, 1.]])
                                          for i in range(3)])
        snapshot = deepcopy(guide); rendered = render_guide(guide, result)
        self.assertEqual(guide, snapshot)
        self.assertEqual(rendered["region_sha256"], "frozen")
        for original, changed in zip(guide["gaussian_components"], rendered["gaussian_components"]):
            self.assertEqual(original["mean"], changed["mean"])
            self.assertEqual(original["covariance"], changed["covariance"])
        self.assertEqual(len(rendered["gaussian_components"]), 3)
        rendered["gaussian_components"][0]["mean"][0] = 999.
        self.assertEqual(guide, snapshot)

    def test_identical_dictionary_components_are_not_removed(self):
        args = discrete_inputs()
        args["log_component_density"] = logs([[.5, .5], [.5, .5]])
        result = fit_weights(build_problem(**args), initial_weights=[.3, .7])
        self.assertEqual(len(result.weights), 2)
        np.testing.assert_allclose(result.weights, [.3, .7], atol=1e-12)

    def test_all_84_components_survive_serialization_without_geometry_changes(self):
        args = discrete_inputs()
        args["log_component_density"] = np.tile(args["log_component_density"], (1, 42))
        args["reference_weights"] = [1./84]*84
        result = fit_weights(build_problem(**args))
        guide = dict(schema="defensive-latent-shell-guide-v1", region_sha256="fixed-region",
                     defensive_uniform_shell_probability=.5,
                     gaussian_components=[dict(weight=1./84, mean=[float(i), 0., 0., 0., 0., 0.],
                                               covariance=np.eye(6).tolist()) for i in range(84)])
        before = deepcopy(guide)
        rendered = render_guide(guide, result)
        self.assertEqual(guide, before)
        self.assertEqual(len(rendered["gaussian_components"]), 84)
        for original, changed in zip(guide["gaussian_components"], rendered["gaussian_components"]):
            self.assertEqual({k: v for k, v in original.items() if k != "weight"},
                             {k: v for k, v in changed.items() if k != "weight"})
            self.assertGreaterEqual(changed["weight"], 1e-5)
        self.assertAlmostEqual(sum(c["weight"] for c in rendered["gaussian_components"]), 1., places=13)
        for bad in (guide | {"defensive_uniform_shell_probability": .2},
                    guide | {"gaussian_components": guide["gaussian_components"][:-1]}):
            with self.assertRaises(ValueError):
                render_guide(bad, result)

    def test_failed_optimizer_never_returns_a_candidate(self):
        p = build_problem(**discrete_inputs())
        failed = SimpleNamespace(success=False, message="declared limit", x=np.array([.7, .3, 1.]))
        with patch("fit_paired_guide_weights.minimize", return_value=failed):
            with self.assertRaisesRegex(ValueError, "Optimizer failed"):
                fit_weights(p)
        with self.assertRaisesRegex(ValueError, "Optimizer failed"):
            fit_weights(p, maxiter=1)

    def test_false_success_infeasibility_and_nonfinite_output_rejected(self):
        p = build_problem(**discrete_inputs())
        for values in ([.5, .5, .1], [.2, .3, 10.], [math.nan, .5, 1.]):
            fake = SimpleNamespace(success=True, message="claimed success", nit=2, x=np.asarray(values))
            with self.subTest(values=values), patch("fit_paired_guide_weights.minimize", return_value=fake):
                with self.assertRaises(ValueError):
                    fit_weights(p)

    def test_unobserved_groups_are_rejected_not_omitted(self):
        args = discrete_inputs(); args["groups"]["unobserved"] = np.zeros(2, bool)
        with self.assertRaisesRegex(ValueError, "Unobserved protected group: unobserved"):
            build_problem(**args)
        args = discrete_inputs(); args["log_cloud_importance"][:] = -np.inf
        with self.assertRaisesRegex(ValueError, "Unobserved protected group"):
            build_problem(**args)

    def test_invalid_inputs_and_support_fail_closed(self):
        edits = [dict(total_attempts=1), dict(total_attempts=2.), dict(alpha=.2), dict(floor=0.),
                 dict(floor=.5), dict(moment="unknown"), dict(reference_weights=[.4, .4]),
                 dict(reference_weights=[-.1, 1.1]), dict(log_cloud_importance=np.zeros((2, 3))),
                 dict(groups={"integer": np.ones(2, int)}), dict(groups={}),
                 dict(log_uniform_density=[math.nan, 0.]), dict(log_source_density=[-np.inf, 0.]),
                 dict(log_component_density=[[math.inf, 0.], [0., 0.]]),
                 dict(log_cloud_importance=[[math.nan, 0.], [0., 0.]])]
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                build_problem(**(discrete_inputs() | edit))
        args = discrete_inputs(); args["log_component_density"][0] = -np.inf
        args["log_uniform_density"][0] = -np.inf
        with self.assertRaisesRegex(ValueError, "no support"):
            build_problem(**args)


if __name__ == "__main__":
    unittest.main()
