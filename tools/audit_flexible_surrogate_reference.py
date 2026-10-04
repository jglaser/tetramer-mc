"""Independent arithmetic/inventory audit; never recompute or sample geometry.

The receipt hash is supplied externally. Only complete, authenticated reference
outputs may be read by the CLI. Selection probabilities are checked against the
fixed contract; a finite journal cannot establish RNG fairness statistically.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ARMS = ("m1_guided", "m8_guided", "m8_zero")
GATE_INTS = ("gained", "lost", "raw_points", "retained_points", "retained_cells", "created_cells")
GATE_FLOATS = ("log_weight", "envelope_volume")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and value >= 0, f"invalid counter {name}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite {name}")
    return value


def close(actual, expected, name):
    finite(actual, name)
    require(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13), f"wrong {name}")


def boolean(value, name):
    require(type(value) is bool, f"invalid boolean {name}")
    return value


def vector(value, count, name):
    require(type(value) is list and len(value) == count, f"wrong {name} shape")
    for number in value:
        finite(number, name)


def poses(value, name):
    require(type(value) is list and len(value) == 2, f"wrong {name} pair")
    for pose in value:
        require(set(pose) == {"position", "orientation"}, f"wrong {name} pose fields")
        vector(pose["position"], 3, name)
        vector(pose["orientation"], 4, name)
        close(sum(x*x for x in pose["orientation"]), 1., f"{name} quaternion norm")


def score(value, meta, guidance):
    n = integer(value["points_per_body"], "points_per_body")
    require(n == meta["points_per_body"], "changed score cloud size")
    for name in ("spectator_covered", "internal_unshielded", "nearby_spectators"):
        require(type(value[name]) is list and len(value[name]) == 2, f"wrong score {name}")
        for count in value[name]:
            integer(count, name)
    require(all(x <= 1 for x in value["nearby_spectators"]), "too many reference spectators")
    require(integer(value["spectator_membership_queries"], "spectator queries")
            == n * sum(value["nearby_spectators"]), "wrong single-spectator query count")
    require(integer(value["internal_membership_queries"], "internal queries")
            == 2*n - sum(value["spectator_covered"]), "wrong internal query count")
    require(all(a+b <= n for a, b in zip(value["spectator_covered"], value["internal_unshielded"])),
            "score coverage exceeds cloud")
    require(all(near or covered == 0 for near, covered in zip(value["nearby_spectators"], value["spectator_covered"])),
            "coverage without nearby spectator")
    units = 2 * sum(value["spectator_covered"]) + sum(value["internal_unshielded"])
    require(integer(value["twice_overlap_units"], "twice_overlap_units") == units, "wrong score units")
    volume = 0.5 * meta["point_volume"] * units
    require(value["overlap_volume_estimate"] == volume, "wrong surrogate volume arithmetic")
    expected = (meta["activity"] * guidance) * volume
    require(finite(value["log_surrogate"], "score") == expected, "wrong surrogate log score")
    return expected


def gate(value, meta, aggregate=False):
    require(set(value) == set(GATE_INTS + GATE_FLOATS), "wrong bath gate fields")
    for name in GATE_INTS:
        integer(value[name], name)
    for name in GATE_FLOATS:
        finite(value[name], name)
    require(value["retained_points"] == value["gained"] + value["lost"], "wrong retained bath count")
    require(value["retained_points"] <= value["raw_points"], "bath retained exceeds raw")
    require(value["retained_cells"] <= value["created_cells"], "bath cells exceed created")
    require(value["envelope_volume"] >= 0., "negative bath volume")
    if value["envelope_volume"] == 0.:
        require(value["raw_points"] == 0, "cloud drawn in zero envelope")
    if not aggregate:
        coefficient = math.log1p(meta["activity"] / meta["lambda"])
        close(value["log_weight"], coefficient * (value["gained"] - value["lost"]), "leg log weight")


def path(value, old, proposed, members, meta):
    order = value["order"]
    require(order in ("first_then_second", "second_then_first"), "unknown path order")
    slots = [0, 1] if order == "first_then_second" else [1, 0]
    require(value["ordered_members"] == [members[i] for i in slots], "wrong ordered labels")
    middle = list(old)
    middle[slots[0]] = proposed[slots[0]]
    require(value["intermediate_selected"] == middle, "wrong copied path intermediate")
    require(len(value["legs"]) == 2, "path does not have two legs")
    for slot, leg in zip(slots, value["legs"]):
        gate(leg, meta)
        if old[slot] == proposed[slot]:
            require(all(leg[name] == 0 for name in GATE_INTS + GATE_FLOATS),
                    "unchanged singleton leg consumed bath")
    aggregate = value["aggregate"]
    gate(aggregate, meta, aggregate=True)
    for name in GATE_INTS + GATE_FLOATS:
        require(aggregate[name] == value["legs"][0][name] + value["legs"][1][name],
                f"wrong two-leg aggregate {name}")
    return aggregate


def audit_record(record, old, retained, arm, meta, negative_controls):
    require(arm in ARMS, "unknown arm")
    require(record["kind"] == "flexible_surrogate_chain", "wrong kernel kind")
    require(record["status"] in ("completed", "identity_self_loop"), "incomplete/fatal kernel")
    require(record["old"] == old, "wrong outer source")
    poses(old, "old")
    poses(retained, "retained")
    members = record["members"]
    require(members == [0, 1], "reference labels changed")
    require(record["selection_probabilities"] == [0.5, 0.5], "scan is not fixed fair selection")
    expected_config = dict(meta["config"], inner_steps=1 if arm == "m1_guided" else 8,
                           guidance_strength=0. if arm == "m8_zero" else 1.)
    require(record["config"] == expected_config, "wrong fixed arm config")
    guidance = expected_config["guidance_strength"]
    current = old
    current_score = record["old_score"]
    old_score = score(current_score, meta, guidance)
    require(len(record["steps"]) == expected_config["inner_steps"], "wrong fixed horizon")
    counts = {"attempted": 0, "accepted": 0, "hard_rejected": 0, "mh_rejected": 0}
    selected = [0, 0]
    for index, step in enumerate(record["steps"]):
        require(integer(step["index"], "inner index") == index, "wrong inner index")
        require(step["old"] == current, "broken inner residence chain")
        current_log_score = current_score["log_surrogate"]
        require(step["old_score"] == current_log_score, "wrong old inner score")
        slot = integer(step["selected_slot"], "selected_slot")
        require(slot in (0, 1) and integer(step["selected_label"], "selected label") == members[slot], "wrong scan slot/label")
        selected[slot] += 1
        proposed = step["proposed"]
        poses(proposed, "inner proposal")
        require(proposed[1-slot] == current[1-slot], "unselected member moved")
        counts["attempted"] += 1
        accepted = boolean(step["accepted"], "inner accepted")
        if step["status"] == "hard_rejected":
            require(not accepted, "hard rejected step accepted")
            require(not any(k in step for k in ("proposed_score", "log_u", "log_acceptance_ratio")),
                    "hard rejection consumed/scored MH decision")
            counts["hard_rejected"] += 1
        else:
            require(step["status"] == "completed", "incomplete inner step")
            proposed_score = score(step["proposed_score"], meta, guidance)
            delta = proposed_score - current_log_score
            require(step["log_acceptance_ratio"] == delta, "wrong inner score difference")
            log_u = finite(step["log_u"], "inner log coin")
            require(log_u < 0., "inner coin outside Open01")
            require(accepted == (log_u < min(0., delta)), "wrong inner MH decision")
            counts["accepted" if accepted else "mh_rejected"] += 1
            if accepted:
                current, current_score = proposed, step["proposed_score"]
        require(step["retained"] == current, "wrong retained inner pose")
        require(step["retained_score"] == current_score["log_surrogate"], "wrong retained inner score")
    for name in counts:
        integer(record["inner_counts"][name], name)
    require(record["inner_counts"] == counts, "wrong inner counters")
    require(record["proposed"] == current and record["proposed_score"] == current_score,
            "wrong final surrogate endpoint/score")
    correction = old_score - current_score["log_surrogate"]
    require(record["complete_log_correction"] == correction, "wrong endpoint correction")
    before, after = record["budget_before"], record["budget_after"]
    for budget in (before, after):
        require(set(budget) == {"raw", "retained"}, "wrong budget fields")
        for name in budget:
            integer(budget[name], name)
    accepted = boolean(record["accepted"], "outer accepted")
    integer(record["physical_decisions"], "physical decisions")
    identity = current == old
    if identity:
        require(record["status"] == "identity_self_loop" and not accepted, "wrong identity outcome")
        require(record["physical_decisions"] == 0 and before == after, "identity spent physical budget")
        require(not any(k in record for k in ("bath", "log_u", "log_acceptance_ratio")),
                "identity has bath/outer decision")
    else:
        require(record["status"] == "completed" and record["physical_decisions"] == 1, "wrong outer decision count")
        bath = path(record["bath"], old, current, members, meta)
        require(after["raw"] == before["raw"] + bath["raw_points"]
                and after["retained"] == before["retained"] + bath["retained_points"], "wrong completed bath budget")
        ratio = bath["log_weight"] + correction
        require(record["log_acceptance_ratio"] == ratio, "wrong complete physical log ratio")
        log_u = finite(record["log_u"], "outer log coin")
        require(log_u < 0., "outer coin outside Open01")
        require(accepted == (log_u < min(0., ratio)), "wrong outer MH decision")
    require(retained == (current if accepted else old), "wrong final retained physical pose")
    if arm == "m8_guided":
        require(type(negative_controls) is dict and set(negative_controls) == {"omitted", "wrong_sign"},
                "missing negative controls")
        for name, control in negative_controls.items():
            decision = False if identity else record["log_u"] < min(0., record["bath"]["aggregate"]["log_weight"]
                + (0. if name == "omitted" else -correction))
            require(boolean(control["accepted"], f"{name} accepted") == decision, f"wrong {name} decision")
            require(control["retained"] == (current if decision else old), f"wrong {name} retained pose")
    else:
        require(negative_controls is None, "unexpected negative controls")
    return {"inner_counts": counts, "selected": selected, "accepted": accepted, "identity": identity}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_rows(filename, begin_kind, outcome_kind):
    with Path(filename).open("rb") as stream:
        number = 0
        while True:
            raw = stream.readline()
            if not raw:
                break
            number += 1
            require(raw.endswith(b"\n"), f"unterminated row {number}")
            begin = strict_json(raw)
            raw = stream.readline()
            number += 1
            require(raw and raw.endswith(b"\n"), f"missing complete outcome row {number}")
            outcome = strict_json(raw)
            require(begin["kind"] == begin_kind and outcome["kind"] == outcome_kind, "wrong journal pairing")
            yield begin, outcome


def audit_sources(filename, streams, per_stream):
    accepted = {}
    counts = [dict(attempts=0, hard_valid=0, raw_points=0, void_accepted=0) for _ in range(streams)]
    for begin, row in paired_rows(filename, "source_begin", "source_outcome"):
        stream = integer(begin["stream"], "source stream")
        index = integer(begin["source_index"], "source index")
        require(stream < streams and stream == len(accepted)//per_stream, "wrong source stream order/inventory")
        count = counts[stream]
        require(index == count["void_accepted"] and index < per_stream, "wrong source index/replacement")
        count["attempts"] += 1
        require(begin["attempt"] == count["attempts"], "source attempt omitted/repeated")
        for key in ("stream", "source_index", "attempt"):
            require(row[key] == begin[key], "source begin/outcome mismatch")
        require(row["status"] == "completed", "incomplete/fatal source")
        poses(row["poses"], "source")
        require(type(row["positions"]) is list and len(row["positions"]) == 2
                and type(row["haar_normals"]) is list and len(row["haar_normals"]) == 2, "wrong source draw shape")
        for i in range(2):
            vector(row["positions"][i], 3, "source center draw")
            vector(row["haar_normals"][i], 4, "source Haar draw")
            require(row["poses"][i]["position"] == row["positions"][i], "wrong source center copy")
            length = math.sqrt(sum(x*x for x in row["haar_normals"][i]))
            require(math.isfinite(length) and length > 0., "invalid source Haar length")
            for actual, normal in zip(row["poses"][i]["orientation"], row["haar_normals"][i]):
                close(actual, normal/length, "normalized Haar coordinate")
        hard = boolean(row["hard_valid"], "source hard flag")
        keep = boolean(row["void_accepted"], "source void flag")
        if not hard:
            require(not keep and row["clouds"] == [], "hard-invalid source used PPP/was accepted")
        else:
            count["hard_valid"] += 1
            require(len(row["clouds"]) == 2, "missing source cloud")
            for owner, cloud in enumerate(row["clouds"]):
                require(cloud["owner"] == owner, "wrong source cloud owner")
                n = integer(cloud["planned_points"], "source planned points")
                require(len(cloud["points"]) == n, "source cloud truncated after hit")
                for point in cloud["points"]:
                    vector(point, 3, "source PPP point")
                count["raw_points"] += n
        if keep:
            accepted[(stream, index)] = {"poses": row["poses"], "attempt": row["attempt"]}
            count["void_accepted"] += 1
    require(len(accepted) == streams*per_stream, "incomplete source inventory")
    return accepted, counts


def empty_work():
    return dict(attempts=0, accepted=0, physical_decisions=0, identities=0, hard_rejections=0,
                inner_rejections=0, selected_slots=[0, 0], path_orders=[0, 0],
                raw_bath_points=0, retained_bath_points=0)


def audit_kernels(filename, sources, streams, per_stream, meta):
    work = [[empty_work() for _ in ARMS] for _ in range(streams)]
    budget = {"raw": 0, "retained": 0}
    seen = 0
    for begin, row in paired_rows(filename, "kernel_begin", "kernel_outcome"):
        stream = seen//(per_stream*len(ARMS))
        index = (seen//len(ARMS)) % per_stream
        arm_index = seen % len(ARMS)
        arm = ARMS[arm_index]
        require(stream < streams, "too many kernel outcomes")
        require((begin["stream"], begin["source_index"], begin["arm"]) == (stream, index, arm),
                "missing/duplicate/reordered kernel family")
        for key in ("stream", "source_index", "arm", "source_attempt"):
            require(row[key] == begin[key], "kernel begin/outcome mismatch")
        source = sources[(stream, index)]
        require(begin["old"] == source["poses"] and begin["source_attempt"] == source["attempt"],
                "arms did not share accepted source")
        require(row.get("negative_control_error") is None, "negative-control computation failed")
        record = row["record"]
        require(record["budget_before"] == budget, "broken campaign budget continuity")
        result = audit_record(record, begin["old"], row["retained"], arm, meta, row["negative_controls"])
        budget = record["budget_after"]
        current = work[stream][arm_index]
        current["attempts"] += 1
        current["accepted"] += int(result["accepted"])
        current["identities"] += int(result["identity"])
        current["physical_decisions"] += record["physical_decisions"]
        current["hard_rejections"] += result["inner_counts"]["hard_rejected"]
        current["inner_rejections"] += result["inner_counts"]["mh_rejected"]
        for i in range(2):
            current["selected_slots"][i] += result["selected"][i]
        if not result["identity"]:
            bath = record["bath"]
            current["path_orders"][int(bath["order"] == "second_then_first")] += 1
            current["raw_bath_points"] += bath["aggregate"]["raw_points"]
            current["retained_bath_points"] += bath["aggregate"]["retained_points"]
            for leg in bath["legs"]:
                require(leg["raw_points"] <= meta["per_leg_cap"] and leg["retained_points"] <= meta["per_leg_cap"],
                        "completed leg exceeds allocation")
            for name in ("raw_points", "retained_points"):
                require(bath["aggregate"][name] <= meta["per_outer_cap"], "completed path exceeds allocation")
        seen += 1
    require(seen == streams*per_stream*len(ARMS), "incomplete kernel inventory")
    return work, budget


def protocol_meta(protocol):
    require(protocol["schema"] == "flexible-surrogate-independent-reference-protocol-v1", "wrong protocol")
    require(protocol["streams"] == 4 and protocol["sources_per_stream"] == 2048, "changed fixed allocation")
    require(protocol["arms"] == [[1, 1., "m1_guided"], [8, 1., "m8_guided"], [8, 0., "m8_zero"]],
            "changed fixed arms")
    require(protocol["members"] == [0, 1] and protocol["negative_control_arm"] == "m8_guided"
            and protocol["negative_controls"] == ["omitted", "wrong_sign"], "changed control contract")
    for name in ("activity", "lambda", "point_volume", "translation_std", "rotation_std_degrees"):
        finite(protocol[name], name)
    require(protocol["activity"] == 0.5 and protocol["lambda"] == 2., "changed reference bath")
    require(protocol["body_raw_count"] == 512 and protocol["point_volume"] == (2./8.)**3,
            "wrong frozen cloud weight")
    require(protocol["translation_std"] == 0.25 and protocol["rotation_std_degrees"] == 30.,
            "changed reference local scales")
    return dict(activity=protocol["activity"], **{"lambda": protocol["lambda"]},
                point_volume=protocol["point_volume"], points_per_body=protocol["body_retained_count"],
                config={"translation_std": protocol["translation_std"],
                        "rotation_std_degrees": protocol["rotation_std_degrees"]},
                per_leg_cap=protocol["bath_per_leg_cap"], per_outer_cap=protocol["bath_per_outer_cap"])


def audit_completed(base, validation_sha256):
    base = Path(base).resolve()
    validation_file = base/"validation.json"
    require(sha(validation_file) == validation_sha256, "runner validation hash mismatch")
    validation = strict_json(validation_file.read_bytes())
    require(validation["schema"] == "flexible-surrogate-reference-validation-v1"
            and validation["complete"] is True and validation["child_drained"] is True,
            "runner has not drained a completed execution")
    require(validation["source_before"] == validation["source_after"]
            and validation["production_before"] == validation["production_after"], "execution source/production changed")
    # Allow an inferential check to fail after a complete allocation: arithmetic
    # audit status and scientific reference status are separate conclusions.
    require(validation["error"] is None, "runner infrastructure failure")
    files = ("protocol.json", "source-attempts.jsonl", "kernel-attempts.jsonl", "summary.json", "receipt.json")
    hashes = {}
    for name in files:
        path = base/"reference"/name
        expected = validation["outputs"]["reference/"+name]
        require(sha(path) == expected, f"unauthenticated {name}")
        hashes[name] = expected
    receipt = strict_json((base/"reference/receipt.json").read_bytes())
    require(receipt["schema"] == "flexible-surrogate-independent-reference-receipt-v1"
            and receipt["complete"] is True and receipt["numerical_allocation_complete"] is True,
            "reference allocation incomplete")
    require(receipt["retries"] == 0 and receipt["replacement_draws"] == 0, "unplanned replacement/retry")
    require(receipt["output_sha256"] == {name: hashes[name] for name in files if name != "receipt.json"},
            "reference receipt output hashes disagree")
    require(receipt["passed"] == receipt["scientific_checks_passed"], "inconsistent scientific status")
    protocol = strict_json((base/"reference/protocol.json").read_bytes())
    require(protocol["reference_source_sha256"] == validation["source_before"]["tests/flexible_surrogate_stationarity.rs"],
            "compiled reference source differs from frozen source")
    meta = protocol_meta(protocol)
    sources, counts = audit_sources(base/"reference/source-attempts.jsonl", 4, 2048)
    work, budget = audit_kernels(base/"reference/kernel-attempts.jsonl", sources, 4, 2048, meta)
    summary = strict_json((base/"reference/summary.json").read_bytes())
    require(summary["complete"] is True and summary["source_counts"] == counts == receipt["source_counts"],
            "source totals disagree with summary/receipt")
    require(len(summary["stream_panel"]) == 4, "incomplete summary panels")
    for stream, panel in enumerate(summary["stream_panel"]):
        require(panel["stream"] == stream and panel["source_counts"] == counts[stream], "wrong summary source panel")
        require(len(panel["work"]) == 3, "missing summary arm")
        for arm in range(3):
            for name, value in work[stream][arm].items():
                require(panel["work"][arm][name] == value, f"wrong summary work {stream}/{ARMS[arm]}/{name}")
        require(counts[stream]["attempts"] <= protocol["source_attempt_cap_per_stream"]
                and counts[stream]["raw_points"] <= protocol["source_point_cap_per_stream"], "source allocation exceeded")
    require(summary["raw_bath_points"] == budget["raw"] <= protocol["bath_raw_cap"]
            and summary["retained_bath_points"] == budget["retained"] <= protocol["bath_retained_cap"],
            "wrong final campaign budget")
    return dict(schema="flexible-surrogate-reference-journal-audit-v1", complete=True, passed=True,
                validation_sha256=validation_sha256, input_sha256=hashes, source_counts=counts,
                inventory={arm: 4*2048 for arm in ARMS}, work=work, budget=budget,
                reference_scientific_checks_passed=receipt["scientific_checks_passed"],
                scope="Saved arithmetic, decisions, retention, counters, and inventory only; no geometry reclassification, RNG replay, or statistical inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "audit output must be fresh")
    result = audit_completed(args.input, args.validation_sha256)
    result["auditor_sha256"] = sha(__file__)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "inventory": result["inventory"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
