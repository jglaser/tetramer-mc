import copy
import math
import unittest
import numpy as np
from analyze_dimer_tree_equilibrium import (
    angular_log_factor, audit_row, joint_log_density, overlap, reference,
    series_diagnostics,
)


CONFIG = {"core_radius": .2, "exclusion_radius": 1., "activity": 1.5,
          "auxiliary_intensity": 24., "root_radius": 2.,
          "min_separation": .4, "max_separation": 2.5}


def pose(x):
    return {"position": x, "orientation": [1., 0., 0., 0.]}


class ReferenceTests(unittest.TestCase):
    def test_hard_only_radial_integrals_and_lens_limits(self):
        config = {**CONFIG, "activity": 0.}
        result = reference(config)
        a, b = config["min_separation"], config["max_separation"]
        self.assertAlmostEqual(result["radial_mass"], (b**3-a**3)/3, places=12)
        self.assertAlmostEqual(result["d"], .75*(b**4-a**4)/(b**3-a**3), places=12)
        self.assertAlmostEqual(result["contact"], (1-a**3)/(b**3-a**3), places=12)
        self.assertAlmostEqual(overlap(0., 1.), 4*math.pi/3, places=12)
        self.assertEqual(overlap(2., 1.), 0.)
        self.assertEqual(overlap(3., 1.), 0.)

    def test_four_chart_density_sign_basis_symmetry_and_translation(self):
        q = np.array([.2, .3, .4, .5]); q /= np.linalg.norm(q)
        expected = angular_log_factor(q)
        self.assertAlmostEqual(expected, angular_log_factor(-q), places=14)
        self.assertAlmostEqual(expected, angular_log_factor(q[::-1]), places=14)
        root, child = pose([0., 0., 0.]), pose([1., 0., 0.])
        old = joint_log_density([root, child])
        translated = [pose([0., 2., 0.]), pose([1., 2., 0.])]
        self.assertAlmostEqual(joint_log_density(translated)-old, -2., places=13)

    def test_complete_mh_reconstruction_detects_wrong_sign_and_missing_rejection(self):
        old = [pose([0., 0., 0.]), pose([1., 0., 0.])]
        new = [pose([0., .5, 0.]), pose([2., .5, 0.])]
        q = joint_log_density(old) - joint_log_density(new)
        bath = CONFIG["activity"] * (overlap(2., 1.) - overlap(1., 1.))
        row = {"old_state": old, "proposed_state": new, "mh_state": new, "state": new,
               "old_d": 1., "proposed_d": 2., "d": 2., "proposal_null": False,
               "domain_valid": True, "hard_valid": True, "q_correction": q,
               "analytic_log_weight": bath, "gate": None, "log_ratio": q+bath,
               "log_uniform": -100., "accepted": True, "orientation_refresh_count": 2}
        self.assertLess(audit_row(row, CONFIG, "analytic", old), 1e-12)
        bad = copy.deepcopy(row); bad["q_correction"] *= -1
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "analytic")
        bad = copy.deepcopy(row); bad["accepted"] = False
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "analytic")
        with self.assertRaises(ValueError): audit_row(row, CONFIG, "analytic", new)
        bad = copy.deepcopy(row); bad["state"][0]["position"][0] = .01
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "analytic")

    def test_batch_ess_includes_repeated_states_and_rejects_partial_batches(self):
        values = np.repeat([0., 1., 0., 1.], 1024)
        result = series_diagnostics(values)
        self.assertLess(result["batch_1024"]["ess"], 5)
        self.assertEqual(result["n"], 4096)
        with self.assertRaises(ValueError): series_diagnostics(values[:-1])

    def test_defensive_density_uses_relative_cube_and_complete_mixture(self):
        root = pose([0., 0., 0.])
        root["orientation"] = [math.cos(math.pi/8), 0., 0., math.sin(math.pi/8)]
        state = [root, pose([3., 0., 0.])]
        config = {**CONFIG, "uniform_probability": 1., "uniform_half_width": 2.5}
        # World displacement is outside the cube, root-body displacement inside.
        self.assertAlmostEqual(joint_log_density(state, config, "defensive_independent"),
                               -6*math.log(5.), places=12)
        config["uniform_probability"] = 0.
        self.assertAlmostEqual(joint_log_density(state, config, "defensive_independent"),
                               joint_log_density(state), places=12)
        config["uniform_probability"] = .5
        self.assertGreaterEqual(joint_log_density(state, config, "defensive_independent"),
                                2*math.log(.5)-6*math.log(5.))
        config["uniform_probability"] = 1.
        outside = [root, pose([3.6, 0., 0.])]
        self.assertEqual(joint_log_density(outside, config, "defensive_independent"), -math.inf)

    def test_poisson_and_path_arithmetic_tampering_is_detected(self):
        old = [pose([0., 0., 0.]), pose([1., 0., 0.])]
        new = [pose([0., .5, 0.]), pose([2., .5, 0.])]
        q = joint_log_density(old) - joint_log_density(new)
        coefficient = math.log1p(CONFIG["activity"]/CONFIG["auxiliary_intensity"])
        leg = {"gained": 3, "lost": 2, "raw_points": 10, "retained_points": 5,
               "retained_cells": 1, "created_cells": 1, "envelope_volume": 32.,
               "log_weight": coefficient}
        row = {"old_state": old, "proposed_state": new, "mh_state": new, "state": new,
               "old_d": 1., "proposed_d": 2., "d": 2., "proposal_null": False,
               "domain_valid": True, "hard_valid": True, "q_correction": q,
               "analytic_log_weight": -CONFIG["activity"]*overlap(1., 1.),
               "gate": leg, "log_ratio": q+coefficient,
               "log_uniform": -100., "accepted": True, "orientation_refresh_count": 2}
        audit_row(row, CONFIG, "poisson")
        bad = copy.deepcopy(row); bad["gate"]["gained"] += 1
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "poisson")
        bad = copy.deepcopy(row); bad["gate"]["log_weight"] *= -1
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "poisson")
        row["gate"] = {k: v*2 for k, v in leg.items()}
        row["log_ratio"] = q+2*coefficient
        row["path_gate"] = {"order": "first_then_second", "ordered_members": [0, 1],
                            "intermediate_selected": [new[0], old[1]], "legs": [leg, leg],
                            "aggregate": row["gate"]}
        audit_row(row, CONFIG, "singleton_path")
        bad = copy.deepcopy(row); bad["path_gate"]["intermediate_selected"] = old
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "singleton_path")
        bad = copy.deepcopy(row); bad["path_gate"]["legs"][0]["lost"] += 1
        with self.assertRaises(ValueError): audit_row(bad, CONFIG, "singleton_path")


if __name__ == "__main__": unittest.main()
