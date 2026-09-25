#!/usr/bin/env python3
"""Analyze independent frozen-attempt exchange rates, never equilibrium mixing."""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ARMS = ("isotropic", "conditioned_uniform", "conditioned_bands")


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - width), min(1., center + width)]


def summarize(rows):
    n = len(rows)
    exchanges = sum(bool(r.get("tagged_exchange", False)) for r in rows)
    cpu = sum(r["sampler_cpu_seconds"] for r in rows)
    diagnostics = sum(r.get("diagnostics_cpu_seconds", 0.) for r in rows)
    selector = [r.get("selector", {}) for r in rows]
    histogram = Counter(s.get("failure_reason") or "none" for s in selector)
    forward_attempts = sum(s.get("forward", {}).get("attempts", 0) for s in selector)
    reverse_attempts = sum(s.get("reverse", {}).get("attempts", 0) for s in selector)
    forward_predicates = sum(s.get("forward", {}).get("predicate_candidates", 0) for s in selector)
    reverse_predicates = sum(s.get("reverse", {}).get("predicate_candidates", 0) for s in selector)
    crossed = Counter()
    for s in selector:
        for name in ("cross_score_y_u", "cross_score_x_v"):
            value = s.get(name)
            if value is not None:
                crossed[name + ("_favorable" if value == 1 else "_floor")] += 1
    gcas = [r["gca"] for r in rows if r.get("gca") is not None]
    qualified = [r for r in rows if r.get("selector", {}).get("forward", {}).get("selected_score") == 1]
    def partners_share_flip_status(row):
        flipped = set(row["gca"]["flipped_indices"])
        tag_flipped = row["tag"] in flipped
        return all((j in flipped) == tag_flipped for j in row["selector"]["initial_contacts"])
    proposed_exchanges = [r for r in rows if bool(r.get("selector", {}).get("proposed_lost") and r.get("selector", {}).get("proposed_gained")) or (r["arm"] == "isotropic" and r.get("tagged_exchange", False))]
    return {
        "attempts": n,
        "errors": sum(r["status"] != "ok" for r in rows),
        "accepted": sum(bool(r.get("accepted", False)) for r in rows),
        "gca_executed": sum(r.get("gca") is not None for r in rows),
        "tag_flipped": sum(bool(r.get("tag_flipped", False)) for r in rows),
        "tagged_exchanges": exchanges,
        "probability_per_attempt": exchanges / n if n else None,
        "probability_wilson_95": wilson(exchanges, n),
        "sampler_cpu_seconds": cpu,
        "diagnostics_cpu_seconds": diagnostics,
        "exchanges_per_sampler_cpu_second": exchanges / cpu if cpu else None,
        "exchanges_per_cpu_second_including_diagnostics": exchanges / (cpu + diagnostics) if cpu + diagnostics else None,
        "contact_changed_attempts": sum(bool(r.get("pairs_lost") or r.get("pairs_gained")) for r in rows),
        "lost_pairs": sum(len(r.get("pairs_lost", [])) for r in rows),
        "gained_pairs": sum(len(r.get("pairs_gained", [])) for r in rows),
        "selector_failure_reasons": dict(histogram),
        "forward_candidates": forward_attempts,
        "reverse_candidates": reverse_attempts,
        "forward_favorable_candidates": forward_predicates,
        "reverse_favorable_candidates": reverse_predicates,
        "cross_scores": dict(crossed),
        "forward_qualifying_candidate_fraction": forward_predicates / forward_attempts if forward_attempts else None,
        "reverse_qualifying_candidate_fraction": reverse_predicates / reverse_attempts if reverse_attempts else None,
        "forward_cap_failures": sum(s.get("forward", {}).get("capped_failure", False) for s in selector),
        "reverse_cap_failures": sum(s.get("reverse", {}).get("capped_failure", False) for s in selector),
        "forward_qualified_selected_axes": len(qualified),
        "qualified_selected_axes_with_tag_flipped": sum(r.get("selector", {}).get("proposed_tag_flipped", False) for r in qualified),
        "qualified_selected_axes_with_proposed_exchange": sum(bool(r.get("selector", {}).get("proposed_lost") and r.get("selector", {}).get("proposed_gained")) for r in qualified),
        "qualified_selected_axes_with_accepted_exchange": sum(r.get("tagged_exchange", False) for r in qualified),
        "qualified_selected_axes_all_initial_partners_share_tag_flip_status": sum(partners_share_flip_status(r) for r in qualified),
        "qualified_flipped_tags_all_initial_partners_share_tag_flip_status": sum(partners_share_flip_status(r) and r["selector"]["proposed_tag_flipped"] for r in qualified),
        "qualified_selected_axes_mean_largest_component_fraction": statistics.mean(r["gca"]["largest_component"] / sum(r["gca"]["component_sizes"]) for r in qualified) if qualified else None,
        "qualified_selected_axes_initial_contact_preserved": sum(r["selector"]["initial_contacts"] == r["selector"]["proposed_contacts"] for r in qualified),
        "proposed_tagged_exchanges": len(proposed_exchanges),
        "proposed_exchanges_with_reverse_cap_failure": sum(r.get("selector", {}).get("reverse", {}).get("capped_failure", False) for r in proposed_exchanges),
        "proposed_exchanges_selector_rejected": sum(not r.get("accepted", False) for r in proposed_exchanges),
        "selector_acceptance_given_executed_gca": sum(bool(r.get("accepted", False)) for r in rows) / len(gcas) if gcas else None,
        "gca_single_component": sum(g["components"] == 1 for g in gcas),
        "gca_hard_single_component": sum(g["hard_components"] == 1 for g in gcas),
        "gca_mean_largest_component_fraction": statistics.mean(g["largest_component"] / sum(g["component_sizes"]) for g in gcas) if gcas else None,
        "gca_mean_components": statistics.mean(g["components"] for g in gcas) if gcas else None,
    }


def analyze(path):
    report = json.loads(path.read_text())
    if report.get("schema") != "conditional-axis-frozen-attempt-benchmark-v1":
        raise ValueError("unknown benchmark schema")
    if not report.get("finished_allocation") or not report.get("completed"):
        raise ValueError("incomplete/error-bearing benchmark; not a comparison result")
    stream_path = Path(report["attempts_path"])
    if not stream_path.exists():
        stream_path = path.with_suffix(".attempts.jsonl")
    raw = stream_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["attempts_sha256"]:
        raise ValueError("attempt stream hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = {(arm, pop, attempt) for arm in ARMS
                for pop in range(report["args"]["populations"])
                for attempt in range(report["args"]["attempts"])}
    observed = [(r["arm"], r["population"], r["attempt"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("attempts differ from frozen allocation")
    if any(r["status"] != "ok" for r in rows):
        raise ValueError("error-bearing stream")
    for r in rows:
        if r["tagged_exchange"] != bool(r["tag_contacts_lost"] and r["tag_contacts_gained"]):
            raise ValueError("exchange indicator disagrees with contact sets")
        if not r["accepted"] and (r["tagged_exchange"] or r["pairs_lost"] or r["pairs_gained"]):
            raise ValueError("rejected state changed contacts")
    arms = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        populations = [summarize([r for r in arm_rows if r["population"] == p])
                       for p in range(report["args"]["populations"])]
        probabilities = [p["probability_per_attempt"] for p in populations]
        rates = [p["exchanges_per_sampler_cpu_second"] for p in populations]
        item = {"arm": arm, "pooled": summarize(arm_rows), "populations": populations,
                "population_probability_mean": statistics.mean(probabilities),
                "population_probability_standard_error": statistics.stdev(probabilities) / math.sqrt(len(populations)) if len(populations) > 1 else None,
                "population_cpu_rate_mean": statistics.mean(rates),
                "population_cpu_rate_standard_error": statistics.stdev(rates) / math.sqrt(len(populations)) if len(populations) > 1 else None}
        arms.append(item)
    baseline = arms[0]["pooled"]["exchanges_per_sampler_cpu_second"]
    for arm in arms:
        arm["cpu_rate_relative_to_isotropic"] = arm["pooled"]["exchanges_per_sampler_cpu_second"] / baseline if baseline else None
    return {"schema": "conditional-axis-frozen-attempt-analysis-v1", "input": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "completed": True, "arms": arms,
            "interval_scope": "Wilson 95% intervals describe conditional frozen-state exchange probabilities, not equilibrium or timing uncertainty; no multiple-comparison adjustment. Zero between-population variance with zero events does not imply a zero upper probability.",
            "limitations": "Independent one-step attempts only. These rates do not establish contact ESS, equilibrium occupancy, assembly stability or trajectory mixing. Kernel CPU excludes separately reported contact diagnostics, setup and stream serialization."}


def self_test():
    low, high = wilson(0, 128)
    assert low < 1e-15 and 0.02 < high < 0.04
    assert wilson(128, 128)[0] > 0.97
    assert wilson(0, 0) is None
    rows = [{"arm": "isotropic", "status": "ok", "sampler_cpu_seconds": 2., "accepted": False},
            {"arm": "isotropic", "status": "ok", "sampler_cpu_seconds": 3., "accepted": True,
             "tagged_exchange": True, "pairs_lost": [[0, 1]], "pairs_gained": [[0, 2]]}]
    summary = summarize(rows)
    assert summary["probability_per_attempt"] == .5
    assert summary["exchanges_per_sampler_cpu_second"] == .2
    assert summary["attempts"] == 2
    print("conditional-axis benchmark analysis self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input or not args.out:
        parser.error("--input and --out are required")
    result = analyze(args.input)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for arm in result["arms"]:
        p = arm["pooled"]
        print(f'{arm["arm"]}: {p["tagged_exchanges"]}/{p["attempts"]} exchanges; '
              f'{p["exchanges_per_sampler_cpu_second"]:.6g}/CPU s; '
              f'{p["probability_wilson_95"]} probability 95% interval')


if __name__ == "__main__":
    main()
