#!/usr/bin/env python3
"""Read-only analysis of fixed-configuration GCA pair-overlap diagnostics.

Uses unconditional draws, including all misses. Intervals are conservative
binomial KL-Chernoff bounds for cloud integration conditional on the selected
axes and pairs. They are not equilibrium or pair-subsampling error bars.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

COUNTS = {
    "old": "old_hits",
    "cross": "cross_hits",
    "lost": "lost_hits",
    "reverse": "reverse_hits",
    "shielded_lost": "shielded_lost_hits",
    "triple_old": "triple_old_hits",
}
STRATA = ("cross_possible", "cross_disjoint")
STATUSES = ("hard_connected",) + STRATA
ALPHA = 0.05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, label):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label}: expected a nonnegative integer")
    return value


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: expected a finite number")
    return float(value)


def binary_kl(p, q):
    if p == 0.0:
        return -math.log1p(-q) if q < 1.0 else math.inf
    if p == 1.0:
        return -math.log(q) if q > 0.0 else math.inf
    if q <= 0.0 or q >= 1.0:
        return math.inf
    return p * math.log(p / q) + (1.0-p) * (math.log1p(-p)-math.log1p(-q))


def binomial_interval(k, n, alpha=ALPHA):
    """Invert n KL(k/n || p) <= log(2/alpha), with outward brackets.

Each tail has failure probability at most alpha/2 by the Chernoff bound.
No normal approximation or zero-hit substitution is used.
"""
    require(n > 0 and 0 <= k <= n and 0.0 < alpha < 1.0,
            "invalid binomial interval arguments")
    h = math.log(2.0 / alpha) / n
    if k == 0:
        return [0.0, min(1.0, math.nextafter(-math.expm1(-h), math.inf))]
    if k == n:
        return [max(0.0, math.nextafter(math.exp(-h), -math.inf)), 1.0]
    p = k / n
    lo, hi = 0.0, p
    for _ in range(96):
        mid = (lo + hi) / 2.0
        if binary_kl(p, mid) > h:
            lo = mid
        else:
            hi = mid
    lower = max(0.0, math.nextafter(lo, -math.inf))
    lo, hi = p, 1.0
    for _ in range(96):
        mid = (lo + hi) / 2.0
        if binary_kl(p, mid) > h:
            hi = mid
        else:
            lo = mid
    upper = min(1.0, math.nextafter(hi, math.inf))
    return [lower, upper]


def bond(z, volume):
    require(volume >= 0, "negative bond volume")
    return -math.expm1(-z * volume)


def exp_difference(a, b):
    """Compute exp(-a)-exp(-b) without subtracting nearly equal survivors."""
    if a <= b:
        return math.exp(-a) * (-math.expm1(-(b-a)))
    return -math.exp(-b) * (-math.expm1(-(a-b)))


def recruitment_reduction(z, lost, reverse):
    """Stable f(L)-f(max(L-R,0)); positive rare escape gains stay visible."""
    return math.exp(-z * max(lost-reverse, 0.0)) * (-math.expm1(-z * min(lost, reverse)))


def log_no_link(volumes, z, disjoint=False):
    lost = volumes["lost"]
    reverse = 0.0 if disjoint else volumes["reverse"]
    delta = max(lost-reverse, 0.0)
    compensation = min(lost, reverse)
    return {
        "two_body_poisson": -z * lost,
        "pair_integrated": -z * delta,
        "current_full_pair": -z * volumes["shielded_lost"],
        "pair_over_two_body_enhancement": z * compensation,
        "two_body_recruitment_reduction": (
            -z * delta + math.log(-math.expm1(-z * compensation))
            if z > 0.0 and compensation > 0.0 else None),
        "zero_reduction_log_convention": "null means exact zero; finite log values are retained even when the probability underflows",
    }


def probability_estimates(volumes, z, disjoint=False):
    lost, reverse = volumes["lost"], volumes["reverse"]
    delta = lost if disjoint else max(0.0, lost-reverse)
    p_two = bond(z, lost)
    p_soft = bond(z, delta)
    p_full = bond(z, volumes["shielded_lost"])
    return {
        "p_two_body_poisson": p_two,
        "p_pair_integrated": p_soft,
        "p_current_full_pair": p_full,
        "two_body_recruitment_reduction": 0.0 if disjoint else recruitment_reduction(z, lost, reverse),
        "full_pair_recruitment_reduction": exp_difference(z*delta, z*volumes["shielded_lost"]),
        "no_link_two_body_poisson": math.exp(-z*lost),
        "no_link_pair_integrated": math.exp(-z*delta),
        "no_link_current_full_pair": math.exp(-z*volumes["shielded_lost"]),
    }


def probability_intervals(intervals, z, disjoint=False):
    ll, lu = intervals["lost"]
    rl, ru = intervals["reverse"]
    fl, fu = intervals["shielded_lost"]
    # Marginal count intervals need not respect known geometric subset relations.
    # Tightening full lost <= lost is safe and improves the nonlinear bounds.
    fu = min(fu, lu)
    two = [bond(z, ll), bond(z, lu)]
    dl, du = (ll, lu) if disjoint else (max(0.0, ll-ru), max(0.0, lu-rl))
    soft = [bond(z, dl), bond(z, du)]
    full = [bond(z, fl), bond(z, fu)]
    gain_full = [exp_difference(z*du, z*fl), exp_difference(z*dl, z*fu)]
    if disjoint:
        gain_two = [0.0, 0.0]
        gain_full = [min(0.0, x) for x in gain_full]
    else:
        # The gain f(L)-f(max(L-R,0)) is increasing in R, and increases
        # up to L=R before decreasing in L. Its extrema over a rectangle
        # therefore occur at the endpoints or at L=R.
        gain_two = [min(recruitment_reduction(z, ll, rl), recruitment_reduction(z, lu, rl)),
                    recruitment_reduction(z, min(max(ru, ll), lu), ru)]
    return {
        "p_two_body_poisson": two,
        "p_pair_integrated": soft,
        "p_current_full_pair": full,
        "two_body_recruitment_reduction": gain_two,
        "full_pair_recruitment_reduction": gain_full,
        "no_link_two_body_poisson": [math.exp(-z*lu), math.exp(-z*ll)],
        "no_link_pair_integrated": [math.exp(-z*du), math.exp(-z*dl)],
        "no_link_current_full_pair": [math.exp(-z*fu), math.exp(-z*fl)],
    }


def validate_population(pop, disjoint, volume):
    n = count(pop["draws"], "draws")
    require(n > 0, "each population must retain its positive unconditional draw count")
    values = {name: count(pop[field], field) for name, field in COUNTS.items()}
    require(all(k <= n for k in values.values()), "hit count exceeds unconditional draws")
    old, cross, lost, reverse, full, triple = (values[k] for k in COUNTS)
    require(lost <= old and reverse <= cross and full <= lost and triple <= old,
            "geometry mask subset identity failed")
    require(lost + cross <= n and reverse + old <= n,
            "geometry mask disjointness identity failed")
    if disjoint:
        require(cross == reverse == 0, "cross-disjoint certificate conflicts with observed hits")
    if volume == 0.0:
        require(not any(values.values()), "zero-volume envelope contains nonzero hit counts")
    return n, values


def analyze_probe(probe, z, family_alpha):
    estimate = probe["estimate"]
    volume = finite(estimate["envelope_volume"], "envelope_volume")
    require(volume >= 0, "negative envelope volume")
    disjoint = probe["stratum"] == "cross_disjoint"
    populations = estimate["populations"]
    require(len(populations) > 0, "probe has no populations")
    require(len({pop["seed"] for pop in populations}) == len(populations),
            "duplicate population seeds within a probe")
    totals = dict.fromkeys(COUNTS, 0)
    n_total, per_population = 0, []
    for pop in populations:
        n, values = validate_population(pop, disjoint, volume)
        n_total += n
        for name, k in values.items():
            totals[name] += k
        volumes = {name: volume * k / n for name, k in values.items()}
        per_population.append({
            "seed": pop["seed"], "draws": n, "counts": values,
            "volumes_A3": volumes,
            "probabilities_plugin_diagnostic_only": probability_estimates(volumes, z, disjoint),
            "log_no_link_probabilities": log_no_link(volumes, z, disjoint),
        })
    volumes = {name: volume * k / n_total for name, k in totals.items()}
    intervals = {}
    for label, alpha in (("pointwise_95", ALPHA), ("per_probe_joint_95", ALPHA/len(COUNTS)),
                         ("familywise_95", family_alpha)):
        intervals[label] = {name: [volume * p for p in binomial_interval(k, n_total, alpha)]
                            for name, k in totals.items()}
    # The two differences agree after integration under the common involution,
    # but are not pointwise equal; this independent difference is a useful check.
    difference = {
        "old_minus_cross_A3": volumes["old"]-volumes["cross"],
        "lost_minus_reverse_A3": volumes["lost"]-volumes["reverse"],
        "residual_A3": volumes["old"]-volumes["cross"]-volumes["lost"]+volumes["reverse"],
    }
    fb = intervals["familywise_95"]
    difference["residual_familywise_95_interval_A3"] = [
        fb["old"][0]-fb["cross"][1]-fb["lost"][1]+fb["reverse"][0],
        fb["old"][1]-fb["cross"][0]-fb["lost"][0]+fb["reverse"][1],
    ]
    difference["zero_in_familywise_interval"] = (
        difference["residual_familywise_95_interval_A3"][0] <= 0.0 <=
        difference["residual_familywise_95_interval_A3"][1])
    return {
        **{key: probe[key] for key in ("axis_index", "pair_index", "i", "j", "stratum", "selection_probability")},
        "envelope_volume_A3": volume,
        "retained_cells": estimate.get("retained_cells"),
        "created_cells": estimate.get("created_cells"),
        "populations": per_population, "draws": n_total, "counts": totals,
        "volumes_A3": volumes, "volume_intervals_A3": intervals,
        "probabilities_plugin_diagnostic_only": probability_estimates(volumes, z, disjoint),
        "log_no_link_probabilities": log_no_link(volumes, z, disjoint),
        "probability_intervals": {
            label: probability_intervals(bounds, z, disjoint) for label, bounds in intervals.items()
            if label != "pointwise_95"},
        "common_involution_volume_identity": difference,
    }


def summarize_selected(rows):
    if not rows:
        return {"selected": 0, "mean_probabilities_plugin_diagnostic_only": None,
                "mean_probability_familywise_intervals": None}
    keys = rows[0]["probabilities_plugin_diagnostic_only"]
    return {
        "selected": len(rows),
        "mean_probabilities_plugin_diagnostic_only": {
            k: math.fsum(r["probabilities_plugin_diagnostic_only"][k] for r in rows)/len(rows)
            for k in keys},
        "mean_probability_familywise_intervals": {
            k: [math.fsum(r["probability_intervals"]["familywise_95"][k][side] for r in rows)/len(rows)
                for side in (0, 1)] for k in keys},
        "draws": sum(r["draws"] for r in rows),
    }


def analyze(report):
    if "completed" in report:
        require(report["completed"] is True, "input is an unfinished census; analyze the completed report")
    z = finite(report["config"]["reservoir_density"], "reservoir_density")
    require(z >= 0, "negative activity")
    raw_probes = report["probes"]
    cases = {}
    candidate_pairs = report["old_candidate_pairs"]
    axis_counts = {}
    for axis in report["axes"]:
        axis_id = axis["axis_index"]
        require(axis_id not in axis_counts, "duplicate axis index")
        counts = Counter()
        for pair in axis["pairs"]:
            pair_id, status = pair["pair_index"], pair["status"]
            require(isinstance(pair_id, int) and 0 <= pair_id < len(candidate_pairs),
                    "axis pair index outside old-candidate array")
            require(status in STATUSES, f"unknown geometry status {status}")
            require((axis_id, pair_id) not in cases, "duplicate axis/pair case")
            cases[axis_id, pair_id] = status
            counts[status] += 1
        require(len(axis["pairs"]) == len(candidate_pairs),
                "axis does not preserve every old-candidate pair")
        axis_counts[axis_id] = dict(counts)
    require(len(cases) == len(report["axes"]) * len(candidate_pairs), "incomplete case coverage")
    selected_keys = set()
    for probe in raw_probes:
        key = probe["axis_index"], probe["pair_index"]
        require(key in cases and cases[key] == probe["stratum"] and probe["stratum"] in STRATA,
                "probe does not match its predeclared geometry stratum")
        require(key not in selected_keys, "duplicate selected axis/pair case")
        selected_keys.add(key)
        candidate = candidate_pairs[probe["pair_index"]]
        require(probe["i"] == candidate["i"] and probe["j"] == candidate["j"],
                "probe particle labels differ from candidate pair")
        selection_probability = finite(probe["selection_probability"], "selection_probability")
        require(0.0 < selection_probability <= 1.0, "invalid selection probability")
    if "selected_cases" in report:
        declared = {(p["axis_index"], p["pair_index"]): p for p in report["selected_cases"]}
        require(len(declared) == len(report["selected_cases"]) and set(declared) == selected_keys,
                "completed probes differ from frozen selected cases")
        for probe in raw_probes:
            expected = declared[probe["axis_index"], probe["pair_index"]]
            require(expected["stratum"] == probe["stratum"]
                    and expected["selection_probability"] == probe["selection_probability"],
                    "probe allocation differs from frozen selection")
    family_alpha = ALPHA / (len(COUNTS) * max(1, len(raw_probes)))
    rows = [analyze_probe(p, z, family_alpha) for p in raw_probes]
    counts = Counter(cases.values())
    strata = {}
    for stratum in STRATA:
        selected = [r for r in rows if r["stratum"] == stratum]
        n_cases = counts[stratum]
        fraction = len(selected)/n_cases if n_cases else 0.0
        require(all(math.isclose(r["selection_probability"], fraction, rel_tol=1e-10, abs_tol=1e-15)
                    for r in selected), "selection probabilities do not match uniform global stratum sampling")
        strata[stratum] = {
            "cases": n_cases, "unsampled": n_cases-len(selected),
            "selection_fraction": fraction, **summarize_selected(selected),
            "interpretation": "Uniform random sample of this fixed axis/pair stratum; interval covers cloud error only, not pair subsampling.",
        }
    per_axis = []
    for axis_id, coarse in axis_counts.items():
        per_axis.append({"axis_index": axis_id, "case_counts": coarse,
                         "selected_by_stratum": {
                             s: summarize_selected([r for r in rows if r["axis_index"] == axis_id and r["stratum"] == s])
                             for s in STRATA},
                         "interpretation": "Descriptive: pair selection was global within each stratum, not balanced within each axis."})
    possible_rows = [r for r in rows if r["stratum"] == "cross_possible"]
    unmeasured = counts["cross_possible"]-len(possible_rows)
    total_bounds = [math.fsum(r["probability_intervals"]["familywise_95"]["two_body_recruitment_reduction"][side]
                             for r in possible_rows) + (unmeasured if side else 0)
                    for side in (0, 1)]
    n_cases = len(cases)
    n_nonhard = counts["cross_possible"]+counts["cross_disjoint"]
    return {
        "schema": "gca-pair-overlap-analysis-v1",
        "input_schema": report.get("schema"),
        "physical_activity_A_minus3": z,
        "physical_depletant_radius_A": report["config"].get("depletant_radius"),
        "volume_identity_diagnostic": {
            "compared": "old-cross = lost-reverse after integration under the common involution; not a pointwise mask identity",
            "probes_excluding_zero_in_familywise_interval": [
                {"axis_index": r["axis_index"], "pair_index": r["pair_index"]} for r in rows
                if not r["common_involution_volume_identity"]["zero_in_familywise_interval"]],
        },
        "axes": len(report["axes"]), "old_candidate_pairs": len(candidate_pairs),
        "case_counts": {s: counts[s] for s in STATUSES},
        "selected_probes": len(rows),
        "draws": sum(r["draws"] for r in rows),
        "interval_method": {
            "name": "two-sided binomial KL-Chernoff inversion",
            "pointwise_alpha": ALPHA, "familywise_alpha": ALPHA,
            "per_pooled_count_familywise_alpha": family_alpha,
            "nonlinear_interval_note": "Per-probe probability intervals use six-way Bonferroni; whole-report probability intervals use simultaneous bounds across every probe. Marginal volume intervals alone are not treated as joint 95% intervals.",
            "simultaneous_pooled_counts": len(COUNTS)*len(rows),
            "denominator": "Every attempted uniform envelope point; no conditioning on hard-valid or overlap hits.",
            "assumptions": "Independent uniform points within each fixed conservative envelope and independent populations; exact envelope volume and predicates are implementation obligations.",
            "scope": "Conditional on the selected finite axes and pairs. Excludes pair-subsampling and equilibrium/trajectory uncertainty. Plug-in exponentials are diagnostic estimates, not unbiased bond decisions.",
        },
        "strata": strata,
        "per_axis": per_axis,
        "finite_case_two_body_reduction_bound": {
            "total_reduction_sum_familywise_interval": total_bounds,
            "mean_over_all_old_candidate_axis_pairs_familywise_interval":
                [v/n_cases for v in total_bounds] if n_cases else None,
            "mean_over_nonhard_pairs_familywise_interval":
                [v/n_nonhard for v in total_bounds] if n_nonhard else None,
            "unsampled_cross_possible_cases": unmeasured,
            "rule": "Hard-connected and certified cross-disjoint pairs contribute exactly zero additional ability to separate. Every unmeasured cross-possible pair contributes [0,1]. Selected intervals are simultaneous cloud-error bounds. No subsampling extrapolation is used in this bound.",
        },
        "probes": rows,
        "limitations": [
            "Only pairs declared by the input old-overlap bounding geometry are counted; certified omitted old-disjoint pairs cannot lose an existing pair attraction.",
            "The current-full-pair probability records isolated pair recruitment after all other transformed shadows are excluded. It is not a complete cluster-link probability or a prediction of cluster sizes.",
            "A negative full-pair reduction means the pair reference recruits more strongly than the current shielded many-body construction for that pair.",
            "Reduced recruitment need not produce contact exchange, faster mixing, or improved physical sampling; no acceptance rule is changed by this diagnostic.",
            "The conservative finite-case bound does not extrapolate to unseen axes or other configurations.",
        ],
    }


def self_test():
    for n in (1, 2, 10, 1000):
        for k in sorted({0, n//2, n}):
            lo, hi = binomial_interval(k, n)
            assert 0 <= lo <= k/n <= hi <= 1
            broad = binomial_interval(k, n, 0.0001)
            assert broad[0] <= lo and broad[1] >= hi
    assert binomial_interval(0, 100)[1] > 0
    assert binomial_interval(100, 100)[0] < 1
    # Check nonlinear interval containment over a deterministic interior grid.
    for z in (0.0, 0.035, 2.0):
        for bounds in ((0.0, 10.0, 0.0, 20.0), (10.0, 20.0, 1.0, 5.0), (1.0, 5.0, 2.0, 3.0)):
            ll, lu, rl, ru = bounds
            iv = {"lost": [ll, lu], "reverse": [rl, ru], "shielded_lost": [0, lu]}
            result = probability_intervals(iv, z)
            lo, hi = result["two_body_recruitment_reduction"]
            for a in range(11):
                for b in range(11):
                    l, r = ll+(lu-ll)*a/10, rl+(ru-rl)*b/10
                    value = bond(z, l)-bond(z, max(0, l-r))
                    assert lo-1e-14 <= value <= hi+1e-14
    extreme = {"lost": 1451.0, "reverse": 86.0, "shielded_lost": 1451.0}
    value = probability_estimates(extreme, 0.025)
    assert 0.0 < value["two_body_recruitment_reduction"] < 1e-14
    assert math.isclose(value["two_body_recruitment_reduction"],
                        math.exp(-0.025*1365.0)*(-math.expm1(-0.025*86.0)), rel_tol=1e-15)
    assert value["full_pair_recruitment_reduction"] > 0.0
    reverse_value = exp_difference(0.025*1451.0, 0.025*1365.0)
    assert reverse_value < 0.0 and math.isclose(-reverse_value, value["full_pair_recruitment_reduction"], rel_tol=1e-15)
    tiny = {"lost": 100000.0, "reverse": 100.0, "shielded_lost": 100000.0}
    assert probability_estimates(tiny, 1.0)["no_link_pair_integrated"] == 0.0
    assert log_no_link(tiny, 1.0)["pair_integrated"] == -99900.0
    assert math.isfinite(log_no_link(tiny, 1.0)["two_body_recruitment_reduction"])
    population = dict(seed=1, draws=100, old_hits=20, cross_hits=0, lost_hits=10,
                      reverse_hits=0, shielded_lost_hits=5, triple_old_hits=2)
    fixture = {
        "schema": "self-test", "config": {"reservoir_density": 0.035},
        "old_candidate_pairs": [{"i": 0, "j": 1}, {"i": 0, "j": 2}],
        "axes": [{"axis_index": 0, "pairs": [
            {"pair_index": 0, "status": "cross_disjoint"},
            {"pair_index": 1, "status": "cross_possible"}]}],
        "probes": [{"axis_index": 0, "pair_index": 0, "i": 0, "j": 1,
                    "stratum": "cross_disjoint", "selection_probability": 1.0,
                    "estimate": {"envelope_volume": 1000.0, "populations": [population]}}],
    }
    result = analyze(fixture)
    assert result["probes"][0]["probabilities_plugin_diagnostic_only"]["two_body_recruitment_reduction"] == 0.0
    assert result["finite_case_two_body_reduction_bound"]["total_reduction_sum_familywise_interval"] == [0.0, 1.0]
    assert result["draws"] == 100
    population["shielded_lost_hits"] = 11
    try:
        analyze(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("subset error was not detected")
    # Other OLD exclusions may cover all of the overlap, while every other
    # transformed SHADOW is far away. Full lost and triple-old can coincide.
    complete_overlap = dict(seed=7, draws=10, old_hits=10, cross_hits=0,
                            lost_hits=10, reverse_hits=0,
                            shielded_lost_hits=10, triple_old_hits=10)
    validate_population(complete_overlap, True, 1000.0)
    fixture["probes"] = []
    for pair in fixture["axes"][0]["pairs"]:
        pair["status"] = "hard_connected"
    empty = analyze(fixture)
    assert empty["draws"] == 0 and empty["selected_probes"] == 0
    assert empty["finite_case_two_body_reduction_bound"]["total_reduction_sum_familywise_interval"] == [0.0, 0.0]
    assert empty["finite_case_two_body_reduction_bound"]["mean_over_nonhard_pairs_familywise_interval"] is None
    fixture["old_candidate_pairs"] = []
    fixture["axes"][0]["pairs"] = []
    assert analyze(fixture)["finite_case_two_body_reduction_bound"]["mean_over_all_old_candidate_axis_pairs_familywise_interval"] is None
    print("GCA overlap analyzer self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.input is None or args.out is None:
        parser.error("--input and --out are required unless --self-test is used")
    require(not args.out.exists(), "output already exists; choose a fresh analysis path")
    raw = args.input.read_bytes()
    result = analyze(json.loads(raw))
    result["provenance"] = {
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['axes']} axes; {result['old_candidate_pairs']} old candidate pairs; "
          f"{result['selected_probes']} probes; {result['draws']} unconditional draws")
    for name, info in result["strata"].items():
        mean = info["mean_probabilities_plugin_diagnostic_only"]
        suffix = "" if mean is None else (
            f"; sampled mean two-body reduction {mean['two_body_recruitment_reduction']:.6g}"
            f"; full-pair reduction {mean['full_pair_recruitment_reduction']:.6g}")
        print(f"{name}: selected {info['selected']}/{info['cases']}{suffix}")
    bound = result["finite_case_two_body_reduction_bound"]["mean_over_all_old_candidate_axis_pairs_familywise_interval"]
    print(f"Finite-case mean two-body reduction bound: {bound}")
    print("Bounds cover cloud error for these finite cases, not equilibrium or unobserved configurations.")
    print(args.out)


if __name__ == "__main__":
    main()
