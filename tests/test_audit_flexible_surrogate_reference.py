"""Synthetic mutation tests only. No actual reference output is read."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "tools/audit_flexible_surrogate_reference.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
META = dict(activity=0.5, **{"lambda": 2.}, point_volume=0.5, points_per_body=8,
            config={"translation_std": 0.25, "rotation_std_degrees": 30.},
            per_leg_cap=1000, per_outer_cap=1000)


def pose(x):
    return {"position": [x, 0., 0.], "orientation": [1., 0., 0., 0.]}


def score(internal, guidance):
    units = 2+internal
    return dict(points_per_body=8, spectator_covered=[1, 0], internal_unshielded=[internal, 0],
                nearby_spectators=[1, 1], spectator_membership_queries=16,
                internal_membership_queries=15, twice_overlap_units=units,
                overlap_volume_estimate=0.25*units, log_surrogate=0.5*guidance*0.25*units)


def gate(gained=0, lost=0):
    active = gained+lost > 0
    return dict(gained=gained, lost=lost, raw_points=gained+lost+2 if active else 0,
                retained_points=gained+lost, retained_cells=2 if active else 0,
                created_cells=3 if active else 0, envelope_volume=1. if active else 0.,
                log_weight=(gained-lost)*math.log1p(0.25))


def fixture(arm="m8_guided", identity=False, old=None):
    old = deepcopy(old or [pose(-0.5), pose(0.5)])
    horizon = 1 if arm == "m1_guided" else 8
    guidance = 0. if arm == "m8_zero" else 1.
    record = dict(kind="flexible_surrogate_chain", status="in_progress", members=[0, 1],
        selection_probabilities=[0.5, 0.5], config=dict(META["config"], inner_steps=horizon, guidance_strength=guidance),
        old=old, old_score=score(4, guidance), steps=[],
        inner_counts=dict(attempted=horizon, accepted=0, hard_rejected=0, mh_rejected=0),
        budget_before={"raw": 0, "retained": 0})
    current, current_score = deepcopy(old), record["old_score"]
    actions = [(0, 2, -1.), (1, None, None), (1, 0, -0.1), (1, 3, -1.)]
    actions += [(i % 2, None, None) for i in range(4, 8)]
    if identity:
        actions = [(i % 2, None, None) for i in range(horizon)]
    for index, (slot, units, log_u) in enumerate(actions[:horizon]):
        proposed = deepcopy(current)
        proposed[slot]["position"][0] += 0.125
        trace = dict(index=index, selected_slot=slot, selected_label=slot, old=deepcopy(current),
                     old_score=current_score["log_surrogate"], proposed=proposed, accepted=False)
        if units is None:
            trace["status"] = "hard_rejected"
            record["inner_counts"]["hard_rejected"] += 1
        else:
            candidate_score = score(units, guidance)
            delta = candidate_score["log_surrogate"]-current_score["log_surrogate"]
            accepted = log_u < min(0., delta)
            trace.update(status="completed", proposed_score=candidate_score,
                         log_acceptance_ratio=delta, log_u=log_u, accepted=accepted)
            record["inner_counts"]["accepted" if accepted else "mh_rejected"] += 1
            if accepted:
                current, current_score = proposed, candidate_score
        trace.update(retained=deepcopy(current), retained_score=current_score["log_surrogate"])
        record["steps"].append(trace)
    correction = record["old_score"]["log_surrogate"]-current_score["log_surrogate"]
    record.update(proposed=current, proposed_score=current_score, complete_log_correction=correction)
    if identity:
        record.update(status="identity_self_loop", accepted=False, physical_decisions=0,
                      budget_after=deepcopy(record["budget_before"]))
    else:
        # Explicit finite trace fixture, not a sampled physical trajectory.
        legs = [gate(1, 3), gate(1, 2) if horizon == 8 else gate()]
        aggregate = {name: legs[0][name]+legs[1][name] for name in legs[0]}
        ratio = aggregate["log_weight"]+correction
        log_u = -0.6 if horizon == 8 else -0.3
        record.update(status="completed", physical_decisions=1, accepted=log_u < min(0., ratio),
            log_u=log_u, log_acceptance_ratio=ratio,
            budget_after={"raw": aggregate["raw_points"], "retained": aggregate["retained_points"]},
            bath=dict(order="first_then_second", ordered_members=[0, 1],
                      intermediate_selected=[current[0], old[1]], legs=legs, aggregate=aggregate))
    retained = current if record["accepted"] else old
    negative = None
    if arm == "m8_guided":
        negative = {}
        for name in ("omitted", "wrong_sign"):
            accepted = False if identity else record["log_u"] < min(0., record["bath"]["aggregate"]["log_weight"]
                + (0. if name == "omitted" else -correction))
            negative[name] = dict(accepted=accepted, retained=current if accepted else old)
    return deepcopy((record, old, retained, arm, META, negative))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))


def source_rows(per_stream=2):
    rows = []
    attempt = 0
    for index in range(per_stream):
        old = [pose(-0.5-index*0.125), pose(0.5+index*0.125)]
        for keep in ([False, True] if index == 0 else [True]):
            attempt += 1
            ids = dict(stream=0, source_index=index, attempt=attempt)
            rows += [dict(kind="source_begin", **ids), dict(kind="source_outcome", **ids,
                positions=[p["position"] for p in old], haar_normals=[[1., 0., 0., 0.]]*2,
                poses=old, hard_valid=keep, void_accepted=keep, status="completed",
                clouds=[dict(owner=0, planned_points=1, points=[[3., 0., 0.]]),
                        dict(owner=1, planned_points=0, points=[])] if keep else [])]
    return rows


class ArithmeticTests(unittest.TestCase):
    def test_all_arms_and_identity_controls_pass(self):
        for arm in audit.ARMS:
            for identity in (False, True):
                with self.subTest(arm=arm, identity=identity):
                    result = audit.audit_record(*fixture(arm, identity))
                    self.assertEqual(result["identity"], identity)

    def test_inner_mutations_are_rejected(self):
        mutations = {
            "wrong label": lambda r: r["steps"][0].update(selected_label=1),
            "wrong slot": lambda r: r["steps"][0].update(selected_slot=3),
            "biased scan": lambda r: r.update(selection_probabilities=[0.6, 0.4]),
            "short horizon": lambda r: r["steps"].pop(),
            "changed horizon": lambda r: r["config"].update(inner_steps=7),
            "residence": lambda r: r["steps"][1]["old"][0]["position"].__setitem__(0, 7.),
            "unselected moved": lambda r: r["steps"][0]["proposed"][1]["position"].__setitem__(0, 7.),
            "inner ratio": lambda r: r["steps"][0].update(log_acceptance_ratio=1.),
            "inner decision": lambda r: r["steps"][0].update(accepted=False),
            "rejection retention": lambda r: r["steps"][2]["retained"][0]["position"].__setitem__(0, 7.),
            "hard rejection scored": lambda r: r["steps"][1].update(log_u=-1.),
            "hard rejection accepted": lambda r: r["steps"][1].update(accepted=True),
            "counter": lambda r: r["inner_counts"].update(accepted=7),
            "score count": lambda r: r["old_score"].update(twice_overlap_units=100),
            "stale score": lambda r: r["steps"][0].update(old_score=99.),
            "nonfinite score": lambda r: r["old_score"].update(log_surrogate=float("nan")),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                args = list(fixture())
                mutate(args[0])
                with self.assertRaises(ValueError):
                    audit.audit_record(*args)

    def test_outer_path_and_control_mutations_are_rejected(self):
        mutations = {
            "correction": lambda r: r.update(complete_log_correction=-r["complete_log_correction"]),
            "outer ratio": lambda r: r.update(log_acceptance_ratio=0.),
            "outer decision": lambda r: r.update(accepted=not r["accepted"]),
            "coin": lambda r: r.update(log_u=1.),
            "extra decision": lambda r: r.update(physical_decisions=2),
            "wrong order labels": lambda r: r["bath"].update(ordered_members=[1, 0]),
            "wrong midpoint": lambda r: r["bath"]["intermediate_selected"][1]["position"].__setitem__(0, 9.),
            "leg log": lambda r: r["bath"]["legs"][0].update(log_weight=0.),
            "leg count": lambda r: r["bath"]["legs"][0].update(retained_points=99),
            "aggregate": lambda r: r["bath"]["aggregate"].update(raw_points=99),
            "missing leg": lambda r: r["bath"]["legs"].pop(),
            "budget": lambda r: r["budget_after"].update(raw=99),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                args = list(fixture())
                mutate(args[0])
                with self.assertRaises(ValueError):
                    audit.audit_record(*args)
        for name in ("omitted", "wrong_sign"):
            args = list(fixture())
            args[5][name]["accepted"] = not args[5][name]["accepted"]
            with self.assertRaises(ValueError):
                audit.audit_record(*args)
        args = list(fixture(identity=True))
        args[0]["bath"] = {}
        with self.assertRaises(ValueError):
            audit.audit_record(*args)
        args = list(fixture())
        args[2][1]["position"][0] = 9.
        with self.assertRaises(ValueError):
            audit.audit_record(*args)

    def test_reversed_path_order_passes_with_correct_midpoint(self):
        args = list(fixture())
        record = args[0]
        record["bath"].update(order="second_then_first", ordered_members=[1, 0],
                              intermediate_selected=[args[1][0], record["proposed"][1]])
        audit.audit_record(*args)


class InventoryTests(unittest.TestCase):
    def test_complete_small_synthetic_inventory_and_missing_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path, kernel_path = Path(tmp)/"source.jsonl", Path(tmp)/"kernel.jsonl"
            write_rows(source_path, source_rows())
            sources, counts = audit.audit_sources(source_path, 1, 2)
            self.assertEqual(counts, [dict(attempts=3, hard_valid=2, raw_points=2, void_accepted=2)])
            rows = []
            budget = {"raw": 0, "retained": 0}
            for index in range(2):
                for arm in audit.ARMS:
                    record, old, retained, _, _, negatives = fixture(arm, old=sources[(0, index)]["poses"])
                    delta = record["budget_after"]
                    record["budget_before"] = budget
                    budget = {name: budget[name]+delta[name] for name in budget}
                    record["budget_after"] = budget
                    ids = dict(stream=0, source_index=index, arm=arm, source_attempt=sources[(0, index)]["attempt"])
                    rows += [dict(kind="kernel_begin", old=old, **ids),
                             dict(kind="kernel_outcome", record=record, retained=retained,
                                  negative_controls=negatives, negative_control_error=None, **ids)]
            write_rows(kernel_path, rows)
            work, end = audit.audit_kernels(kernel_path, sources, 1, 2, META)
            self.assertEqual([w["attempts"] for w in work[0]], [2, 2, 2])
            self.assertEqual(end, budget)
            for mutation in (rows[:-2], rows+rows[-2:], rows[2:]):
                write_rows(kernel_path, mutation)
                with self.assertRaises(ValueError):
                    audit.audit_kernels(kernel_path, sources, 1, 2, META)
            changed = deepcopy(rows)
            changed[2]["source_attempt"] += 1
            write_rows(kernel_path, changed)
            with self.assertRaises(ValueError):
                audit.audit_kernels(kernel_path, sources, 1, 2, META)

    def test_source_omission_truncated_cloud_and_partial_rows_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp)/"source.jsonl"
            for name in ("missing attempt", "truncated cloud", "replacement", "missing outcome"):
                rows = source_rows()
                if name == "missing attempt": rows = rows[2:]
                if name == "truncated cloud": rows[3]["clouds"][0]["points"] = []
                if name == "replacement": rows[4]["source_index"] = 0
                if name == "missing outcome": rows.pop()
                write_rows(filename, rows)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    audit.audit_sources(filename, 1, 2)
            filename.write_text(json.dumps(source_rows()[0]))
            with self.assertRaises(ValueError):
                list(audit.paired_rows(filename, "source_begin", "source_outcome"))

    def test_duplicate_json_nan_and_wrong_validation_hash_fail(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                audit.strict_json(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"validation.json"
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "validation hash mismatch"):
                audit.audit_completed(tmp, "0"*64)

    def test_protocol_fixed_inventory_and_arm_contract(self):
        protocol = dict(schema="flexible-surrogate-independent-reference-protocol-v1",
            streams=4, sources_per_stream=2048, arms=[[1, 1., "m1_guided"], [8, 1., "m8_guided"], [8, 0., "m8_zero"]],
            members=[0, 1], negative_control_arm="m8_guided", negative_controls=["omitted", "wrong_sign"],
            activity=0.5, **{"lambda": 2.}, point_volume=(2./8.)**3, body_raw_count=512, body_retained_count=280,
            translation_std=0.25, rotation_std_degrees=30., bath_per_leg_cap=100000, bath_per_outer_cap=100000)
        self.assertEqual(audit.protocol_meta(protocol)["points_per_body"], 280)
        for field, wrong in (("sources_per_stream", 2047), ("streams", 3), ("members", [1, 0]),
                             ("point_volume", 8./280), ("translation_std", 0.3)):
            changed = deepcopy(protocol)
            changed[field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.protocol_meta(changed)


if __name__ == "__main__":
    unittest.main()
