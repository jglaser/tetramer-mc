import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from analyze_dimer_one_step_stationarity import (
    OBSERVABLES, audit_row, audit_source, expected_seed, merge, moments,
    observables, primary_contact, reference, validate_config,
)


CONFIG = json.loads((Path(__file__).resolve().parents[1]/"examples/dimer-one-step-stationarity.json").read_text())


def identity_row(arm="analytic"):
    spec = next(p for p in CONFIG["populations"] if p["arm"] == arm and p["proposal_kind"] == "tree")
    old = [{"position": [1., 0., 0.], "orientation": [1., 0., 0., 0.]},
           {"position": [1., 1.1, 0.], "orientation": [1., 0., 0., 0.]}]
    from analyze_dimer_tree_equilibrium import overlap
    log_accept = CONFIG["activity"]*(overlap(1.1, 1.)-overlap(.4, 1.))
    source = {"radial_trials": 1, "radial_uniform": (1.1**3-.4**3)/(2.5**3-.4**3),
              "radial_log_uniform": -100., "radial_log_acceptance": log_accept,
              "sampled_distance": 1.1, "root_radius_uniform": .125,
              "root_direction_normals": [1., 0., 0.], "relative_direction_normals": [0., 1., 0.],
              "quaternion_normals": [[1., 0., 0., 0.], [1., 0., 0., 0.]]}
    proposal = {"spectator": {"position": [0., 0., 0.], "orientation": [1., 0., 0., 0.]},
                "old_root": old[0], "old_child": old[1],
                "candidate": {"root": old[0], "child": old[1], "diagnostics": {"log_reverse_forward": 0.}}}
    gate = None if arm == "analytic" else {
        "gained": 0, "lost": 0, "raw_points": 0, "retained_points": 0,
        "log_weight": 0., "envelope_volume": 0., "retained_cells": 0, "created_cells": 0}
    path = None if arm != "singleton_path" else {
        "order": "first_then_second", "ordered_members": [0, 1], "intermediate_selected": old,
        "legs": [copy.deepcopy(gate), copy.deepcopy(gate)], "aggregate": gate}
    return spec, {"population_id": spec["id"], "step": 0,
        "source_seed": expected_seed(spec["seed"], 0, "source"), "source": source,
        "old_state": old, "proposed_state": old, "state": old,
        "old_d": 1.1, "proposed_d": 1.1, "d": 1.1,
        "proposal": proposal, "defensive_proposal": None, "proposal_null": False,
        "domain_valid": True, "hard_valid": True, "q_correction": 0.,
        "analytic_log_weight": 0., "gate": gate, "path_gate": path,
        "log_ratio": 0., "log_uniform": -1., "accepted": True,
        "orientation_refresh_count": 0, "cpu_seconds": 1., "attempt_cpu_seconds": .1, "error": None}


class OneStepTests(unittest.TestCase):
    def test_frozen_five_arm_allocation_and_source_reference_limits(self):
        validate_config(CONFIG)
        wrong = copy.deepcopy(CONFIG); wrong["populations"][1]["seed"] = wrong["populations"][0]["seed"]
        with self.assertRaises(ValueError): validate_config(wrong)
        result = reference({**CONFIG, "activity": 0.})
        self.assertAlmostEqual(result["source_radial_acceptance"], 1., places=14)
        self.assertAlmostEqual(result["source_radial_trials_mean"], 1., places=14)
        self.assertAlmostEqual(result["radial_mass"], (2.5**3-.4**3)/3, places=12)
        self.assertAlmostEqual(result["observables"]["root_quartic_sum"], .5)
        self.assertEqual(result["observables"]["root_child_q0_squared_product"], 1/16)
        self.assertAlmostEqual(sum(result["radial_bin_probabilities"]), 1., places=12)
        self.assertAlmostEqual(reference(CONFIG)["observables"]["contact"], .48404710905886494, places=12)

    def test_saved_source_trace_reconstructs_and_rejects_tampering(self):
        spec, row = identity_row()
        audit_source(row, CONFIG, spec)
        for edit in [lambda r:r["source_seed"].__setitem__(0, r["source_seed"][0]^1),
                     lambda r:r["source"].__setitem__("radial_trials", 0),
                     lambda r:r["source"].__setitem__("radial_log_uniform", 0.),
                     lambda r:r["source"].__setitem__("sampled_distance", 1.2),
                     lambda r:r["source"].__setitem__("root_direction_normals", [1., 0., 0., 0.]),
                     lambda r:r["source"].__setitem__("quaternion_normals", [[0., 1., 0., 0.]]*2)]:
            bad = copy.deepcopy(row); edit(bad)
            with self.assertRaises(ValueError): audit_source(bad, CONFIG, spec)

    def test_no_refresh_mh_and_both_baths_are_independently_checked(self):
        for arm in ["analytic", "poisson", "singleton_path"]:
            spec, row = identity_row(arm)
            self.assertLess(audit_row(row, CONFIG, spec), 1e-12)
            for key, value in [("q_correction", .1), ("log_ratio", .1),
                               ("orientation_refresh_count", 2), ("accepted", False)]:
                bad = copy.deepcopy(row); bad[key] = value
                with self.assertRaises(ValueError): audit_row(bad, CONFIG, spec)
            if arm != "analytic":
                bad = copy.deepcopy(row); bad["gate"]["gained"] = -1
                with self.assertRaises(ValueError): audit_row(bad, CONFIG, spec)
            if arm == "singleton_path":
                bad = copy.deepcopy(row); bad["path_gate"]["intermediate_selected"][0]["position"][0] = .5
                with self.assertRaises(ValueError): audit_row(bad, CONFIG, spec)

    def test_all_declared_observables_are_quaternion_sign_invariant(self):
        _, row = identity_row(); state = copy.deepcopy(row["old_state"])
        state[0]["orientation"] = [.5, .5, .5, .5]
        values = observables(state)
        self.assertEqual(set(values), set(OBSERVABLES))
        self.assertEqual(values["root_quartic_sum"], .25)
        self.assertEqual(values["root_child_q0_squared_product"], .25)
        self.assertAlmostEqual(values["root_axis_alignment_squared"], 0., places=14)
        for member in range(2):
            flipped = copy.deepcopy(state)
            flipped[member]["orientation"] = [-v for v in flipped[member]["orientation"]]
            for key, value in observables(flipped).items(): self.assertAlmostEqual(value, values[key], places=14)

    def test_paired_stats_merge_and_exact_primary_include_zero_changes(self):
        values = np.array([1.]*30+[-1.]*30+[0.]*940)
        whole = moments(values); combined = merge([moments(values[:500]), moments(values[500:])])
        for key in whole: self.assertAlmostEqual(whole[key], combined[key], places=14)
        result = primary_contact(whole, 30, 30)
        self.assertEqual(result["exact_conditional_binomial_p"], 1.)
        self.assertFalse(result["primary_reject_stationarity"])
        biased = primary_contact(moments(np.array([1.]*60+[0.]*940)), 60, 0)
        self.assertTrue(biased["primary_reject_stationarity"])
        # SciPy may return np.float64; its comparison produces np.bool_, which
        # must be converted without changing the test statistic or decision.
        self.assertIs(type(biased["exact_conditional_binomial_p"]), float)
        self.assertIs(type(biased["primary_reject_stationarity"]), bool)
        self.assertEqual(json.loads(json.dumps(biased, allow_nan=False)), biased)
        static = primary_contact(moments(np.zeros(1000)), 0, 0)
        self.assertEqual(static["exact_conditional_binomial_p"], 1.)
        self.assertGreater(static["simultaneous_hoeffding_interval"][1], 0.)


if __name__ == "__main__": unittest.main()
