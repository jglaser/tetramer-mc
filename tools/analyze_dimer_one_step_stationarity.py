#!/usr/bin/env python3
"""Frozen, independent-source paired stationarity analysis.

Five primary binary-contact tests use exact conditional binomial tests with a
Bonferroni family alpha .05. Other moments are fully reported, without selecting
favorable diagnostics. Normal intervals are estimates; a simultaneous Hoeffding
interval is also reported for bounded paired contact changes. Source checks are
a separately declared family. No evolving-chain data enter this calculation.

The imported quaternion/density/lens routines are an explicit dependency that
must be archived and hash-bound alongside this file before physical sampling.
Source seeds and accepted raw variates are checked, but this Python analyzer does
not replay Rust's RNG or every rejected source trial / auxiliary point cloud.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import binomtest, chi2, norm
from analyze_dimer_tree_equilibrium import (
    close, distance, joint_log_density, overlap, quaternion_product, validate_state,
)

PROTOCOL = "iid-paired-contact-bonferroni5-v1"
RNG_DOMAIN = b"dimer-one-step-stationarity-v1"
FAMILY_ALPHA = .05
GROUPS = ["tree:analytic", "tree:poisson", "tree:singleton_path",
          "defensive_independent:poisson", "defensive_independent:singleton_path"]
OBSERVABLES = (["contact", "d", "root_r2"]
               + [f"{body}_q{k}_squared" for body in ["root", "child"] for k in range(4)]
               + ["root_quartic_sum", "child_quartic_sum", "relative_quartic_sum",
                  "root_child_q0_squared_product", "contact_root_quartic_sum",
                  "root_axis_alignment_squared"])
RADIAL_BINS = [.4, .6, .8, 1., 1.25, 1.5, 2., 2.5]


def validate_config(config):
    if config["schema"] != 1 or config["analysis_protocol"] != PROTOCOL:
        raise ValueError("unknown one-step protocol")
    if config["draws_per_population"] != 16384 or len(config["populations"]) != 20:
        raise ValueError("allocation differs from frozen one-step design")
    if (config["min_separation"], config["max_separation"]) != (RADIAL_BINS[0], RADIAL_BINS[-1]):
        raise ValueError("radial source histogram domain changed")
    counts = {g: 0 for g in GROUPS}
    ids, seeds = set(), set()
    for spec in config["populations"]:
        group = f'{spec["proposal_kind"]}:{spec["arm"]}'
        if group not in counts or spec["id"] in ids or spec["seed"] in seeds:
            raise ValueError("unknown arm or duplicate id/seed")
        counts[group] += 1
        ids.add(spec["id"]); seeds.add(spec["seed"])
    if any(n != 4 for n in counts.values()):
        raise ValueError("each of five arms requires four independent populations")


def expected_seed(master, step, role):
    return list(hashlib.sha256(RNG_DOMAIN + struct.pack("<QQ", master, step)
                               + role.encode("ascii")).digest())


def unit(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite saved normal variates")
    length = math.sqrt(math.fsum(v*v for v in values))
    if not length > 0:
        raise ValueError("zero normal vector")
    return [v/length for v in values]


def reference(config):
    lo, hi, z, radius = (config[k] for k in
                        ["min_separation", "max_separation", "activity", "exclusion_radius"])
    weight = lambda d: d*d*math.exp(z*overlap(d, radius))
    def integrate(f, left=lo, right=hi):
        return quad(f, left, right, epsabs=1e-11, epsrel=1e-11,
                    points=[p for p in [1., 2*radius] if left < p < right])[0]
    mass = integrate(weight)
    contact = integrate(weight, lo, 1.)/mass
    values = {"contact": contact, "d": integrate(lambda d:d*weight(d))/mass,
              "root_r2": 3*config["root_radius"]**2/5,
              **{f"{body}_q{k}_squared": .25 for body in ["root", "child"] for k in range(4)},
              "root_quartic_sum": .5, "child_quartic_sum": .5, "relative_quartic_sum": .5,
              "root_child_q0_squared_product": 1/16,
              "contact_root_quartic_sum": contact/2,
              "root_axis_alignment_squared": 1/3}
    acceptance = mass / ((hi**3-lo**3)/3 * math.exp(z*overlap(lo, radius)))
    return {"observables": values, "radial_mass": mass,
            "radial_bin_probabilities": [integrate(weight, a, b)/mass
                                         for a, b in zip(RADIAL_BINS[:-1], RADIAL_BINS[1:])],
            "source_radial_acceptance": acceptance,
            "source_radial_trials_mean": 1/acceptance,
            "source_radial_trials_variance": (1-acceptance)/acceptance**2}


def observables(state):
    root, child = state
    qr, qc = root["orientation"], child["orientation"]
    relative = quaternion_product([qr[0], -qr[1], -qr[2], -qr[3]], qc)
    d = distance(state)
    contact = float(d < 1.)
    root4 = math.fsum(q**4 for q in qr)
    # R(q_root) e_z from quaternion conjugation, independent of Rust matrices.
    axis = quaternion_product(quaternion_product(qr, [0., 0., 0., 1.]),
                              [qr[0], -qr[1], -qr[2], -qr[3]])[1:]
    direction = [(a-b)/d for a, b in zip(child["position"], root["position"])]
    return {"contact": contact, "d": d,
            "root_r2": math.fsum(v*v for v in root["position"]),
            **{f"{body}_q{k}_squared": q[k]**2 for body, q in [("root", qr), ("child", qc)] for k in range(4)},
            "root_quartic_sum": root4, "child_quartic_sum": math.fsum(q**4 for q in qc),
            "relative_quartic_sum": math.fsum(q**4 for q in relative),
            "root_child_q0_squared_product": qr[0]**2*qc[0]**2,
            "contact_root_quartic_sum": contact*root4,
            "root_axis_alignment_squared": math.fsum(a*b for a, b in zip(axis, direction))**2}


def audit_source(row, config, spec):
    if row["source_seed"] != expected_seed(spec["seed"], row["step"], "source"):
        raise ValueError("source RNG binding differs")
    source = row["source"]
    if (len(source["root_direction_normals"]) != 3
            or len(source["relative_direction_normals"]) != 3
            or len(source["quaternion_normals"]) != 2
            or any(len(q) != 4 for q in source["quaternion_normals"])):
        raise ValueError("source variate dimensions differ")
    if type(source["radial_trials"]) is not int or source["radial_trials"] < 1:
        raise ValueError("invalid source rejection count")
    for key in ["radial_uniform", "root_radius_uniform"]:
        if not 0 < source[key] < 1:
            raise ValueError("source open uniform outside support")
    lo, hi = config["min_separation"], config["max_separation"]
    d = (lo**3 + source["radial_uniform"]*(hi**3-lo**3))**(1/3)
    close(source["sampled_distance"], d, 2e-12)
    log_accept = config["activity"]*(overlap(d, config["exclusion_radius"])
                                    - overlap(lo, config["exclusion_radius"]))
    close(source["radial_log_acceptance"], log_accept, 2e-12)
    threshold = source["radial_log_uniform"]
    if not math.isfinite(threshold) or not threshold < min(0., log_accept):
        raise ValueError("accepted radial rejection trial violates its envelope")
    root = [v*config["root_radius"]*source["root_radius_uniform"]**(1/3)
            for v in unit(source["root_direction_normals"])]
    child = [r+d*v for r, v in zip(root, unit(source["relative_direction_normals"]))]
    for i, (position, qnormals) in enumerate(zip([root, child], source["quaternion_normals"])):
        for actual, expected in zip(row["old_state"][i]["position"], position):
            close(actual, expected, 2e-12)
        for actual, expected in zip(row["old_state"][i]["orientation"], unit(qnormals)):
            close(actual, expected, 2e-12)


def audit_gate(gate, config):
    if gate is None:
        raise ValueError("valid bath trial lacks a gate")
    for key in ["gained", "lost", "raw_points", "retained_points", "retained_cells", "created_cells"]:
        if type(gate[key]) is not int or gate[key] < 0:
            raise ValueError("invalid gate integer")
    if not (gate["gained"]+gate["lost"] == gate["retained_points"] <= gate["raw_points"]):
        raise ValueError("gate count relation differs")
    if not math.isfinite(gate["envelope_volume"]) or gate["envelope_volume"] < 0:
        raise ValueError("invalid gate envelope")
    physical = (gate["gained"]-gate["lost"])*math.log1p(config["activity"]/config["auxiliary_intensity"])
    close(gate["log_weight"], physical)
    return physical


def audit_row(row, config, spec):
    if row.get("error") is not None:
        raise ValueError("failed draw cannot enter stationarity statistics")
    if row["population_id"] != spec["id"] or row["orientation_refresh_count"] != 0:
        raise ValueError("wrong population or forbidden post-move refresh")
    if not math.isfinite(row["log_uniform"]) or row["log_uniform"] >= 0:
        raise ValueError("invalid Metropolis threshold")
    for key in ["cpu_seconds", "attempt_cpu_seconds"]:
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError("invalid CPU clock")
    old, trial, final = row["old_state"], row["proposed_state"], row["state"]
    validate_state(old); validate_state(final)
    audit_source(row, config, spec)
    close(row["old_d"], distance(old))
    if not (sum(x*x for x in old[0]["position"]) <= config["root_radius"]**2
            and config["min_separation"] <= distance(old) <= config["max_separation"]):
        raise ValueError("source outside physical domain")
    kind, arm = spec["proposal_kind"], spec["arm"]
    field = "proposal" if kind == "tree" else "defensive_proposal"
    other = "defensive_proposal" if kind == "tree" else "proposal"
    record = row[field]
    if record is None or row[other] is not None or record["old_root"] != old[0] or record["old_child"] != old[1]:
        raise ValueError("wrong proposal record or source")
    if record["spectator"] != {"position": [0., 0., 0.], "orientation": [1., 0., 0., 0.]}:
        raise ValueError("virtual proposal frame changed")
    if row["proposal_null"] != (trial is None) or (record["candidate"] is None) != (trial is None):
        raise ValueError("proposal null/endpoint mismatch")
    allowed = False
    max_error = 0.
    if trial is not None:
        validate_state(trial)
        if [record["candidate"]["root"], record["candidate"]["child"]] != trial:
            raise ValueError("logged proposed endpoint differs from proposal result")
        d = distance(trial); close(row["proposed_d"], d)
        domain = (sum(x*x for x in trial[0]["position"]) <= config["root_radius"]**2
                  and config["min_separation"] <= d <= config["max_separation"])
        hard = d >= 2*config["core_radius"]
        if domain != row["domain_valid"] or hard != row["hard_valid"]:
            raise ValueError("physical endpoint predicate differs")
        q = joint_log_density(old, config, kind)-joint_log_density(trial, config, kind)
        close(row["q_correction"], q)
        close(record["candidate"]["diagnostics"]["log_reverse_forward"], q)
        max_error = abs(row["q_correction"]-q)
        analytic = config["activity"]*(overlap(d, config["exclusion_radius"])
                                       - overlap(distance(old), config["exclusion_radius"]))
        close(row["analytic_log_weight"], analytic)
        allowed = domain and hard
        if allowed:
            if arm == "analytic":
                if row["gate"] is not None or row["path_gate"] is not None:
                    raise ValueError("analytic arm generated a cloud")
                physical = analytic
            else:
                physical = audit_gate(row["gate"], config)
                if arm == "singleton_path":
                    path = row["path_gate"]
                    order = {"first_then_second": [0, 1], "second_then_first": [1, 0]}[path["order"]]
                    middle = list(old); middle[order[0]] = trial[order[0]]
                    if path["ordered_members"] != order or path["intermediate_selected"] != middle or path["aggregate"] != row["gate"]:
                        raise ValueError("path copy/order/aggregate differs")
                    if len(path["legs"]) != 2:
                        raise ValueError("two auxiliary legs required")
                    close(physical, sum(audit_gate(leg, config) for leg in path["legs"]))
                    for key in ["gained", "lost", "raw_points", "retained_points", "retained_cells", "created_cells"]:
                        if row["gate"][key] != sum(leg[key] for leg in path["legs"]):
                            raise ValueError("path count sum differs")
                    close(row["gate"]["envelope_volume"], sum(leg["envelope_volume"] for leg in path["legs"]))
                elif row["path_gate"] is not None:
                    raise ValueError("world gate contains a path record")
            close(row["log_ratio"], q+physical)
    if not allowed and any(row[key] is not None for key in ["log_ratio", "gate", "path_gate"]):
        raise ValueError("invalid endpoint nevertheless entered the physical gate")
    accepted = allowed and row["log_uniform"] < min(0., row["log_ratio"])
    if row["accepted"] != accepted or final != (trial if accepted else old):
        raise ValueError("MH decision/rejected state/no-refresh endpoint differs")
    close(row["d"], distance(final))
    return max_error


def moments(values):
    data = np.asarray(values, dtype=float)
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("invalid IID moment sample")
    n, mean, variance = len(data), float(data.mean()), float(data.var(ddof=1))
    return {"n": n, "mean": mean, "variance": variance, "se": math.sqrt(variance/n)}


def merge(parts):
    n = sum(p["n"] for p in parts)
    mean = math.fsum(p["n"]*p["mean"] for p in parts)/n
    m2 = math.fsum((p["n"]-1)*p["variance"]+p["n"]*(p["mean"]-mean)**2 for p in parts)
    return {"n": n, "mean": mean, "variance": m2/(n-1), "se": math.sqrt(m2/(n-1)/n)}


def primary_contact(stats, positive, negative):
    changes = positive+negative
    p = float(binomtest(positive, changes, .5).pvalue) if changes else 1.
    critical = float(norm.ppf(1-FAMILY_ALPHA/(2*len(GROUPS))))
    half = critical*stats["se"]
    bound = math.sqrt(2*math.log(2*len(GROUPS)/FAMILY_ALPHA)/stats["n"])
    return {**stats, "positive_changes": positive, "negative_changes": negative,
            "exact_conditional_binomial_p": p, "bonferroni_adjusted_p": min(1., len(GROUPS)*p),
            "primary_reject_stationarity": p <= FAMILY_ALPHA/len(GROUPS),
            "estimated_bonferroni_normal_interval": [stats["mean"]-half, stats["mean"]+half],
            "simultaneous_hoeffding_interval": [max(-1., stats["mean"]-bound), min(1., stats["mean"]+bound)],
            "scope": "Failure to reject does not establish stationarity or rule out small amplified occupancy bias."}


def analyze(config, directory):
    validate_config(config)
    refs = reference(config)
    populations, sources = [], {}
    for spec in config["populations"]:
        path = directory/f'{spec["id"]}.jsonl'; digest = hashlib.sha256()
        xs = {k: [] for k in OBSERVABLES}; ys = {k: [] for k in OBSERVABLES}; ds = {k: [] for k in OBSERVABLES}
        trials, radial_counts = [], np.zeros(len(RADIAL_BINS)-1, dtype=int)
        positives = negatives = accepted = raw_points = 0
        max_q_error = cpu = 0.
        with path.open("rb") as stream:
            for step, raw in enumerate(stream):
                digest.update(raw); row = json.loads(raw)
                if row["step"] != step:
                    raise ValueError("missing/reordered IID attempt")
                max_q_error = max(max_q_error, audit_row(row, config, spec))
                if row["cpu_seconds"] < cpu:
                    raise ValueError("CPU clock decreased")
                cpu = row["cpu_seconds"]
                x, y = observables(row["old_state"]), observables(row["state"])
                for key in OBSERVABLES:
                    xs[key].append(x[key]); ys[key].append(y[key]); ds[key].append(y[key]-x[key])
                delta = y["contact"]-x["contact"]
                positives += int(delta == 1); negatives += int(delta == -1)
                trials.append(row["source"]["radial_trials"])
                index = int(np.searchsorted(RADIAL_BINS, x["d"], side="right")-1)
                radial_counts[min(index, len(radial_counts)-1)] += 1
                accepted += int(row["accepted"]); raw_points += (row["gate"] or {}).get("raw_points", 0)
        if len(trials) != config["draws_per_population"]:
            raise ValueError("incomplete or excess population allocation")
        sources[str(path)] = digest.hexdigest()
        populations.append({**spec, "n": len(trials), "accepted": accepted, "raw_points": raw_points,
            "cpu_seconds": cpu, "max_logq_error": max_q_error,
            "source": {k: moments(xs[k]) for k in OBSERVABLES},
            "endpoint": {k: moments(ys[k]) for k in OBSERVABLES},
            "paired_change": {k: moments(ds[k]) for k in OBSERVABLES},
            "positive_contact_changes": positives, "negative_contact_changes": negatives,
            "source_radial_trials": moments(trials), "source_radial_counts": radial_counts.tolist()})
    arms = {}
    for name in GROUPS:
        group = [p for p in populations if f'{p["proposal_kind"]}:{p["arm"]}' == name]
        paired = {k: merge([p["paired_change"][k] for p in group]) for k in OBSERVABLES}
        secondary = {}
        for key, stats in paired.items():
            if key == "contact": continue
            secondary[key] = {**stats, "normal_95_interval": [stats["mean"]-1.96*stats["se"], stats["mean"]+1.96*stats["se"]],
                              "mean_over_se": stats["mean"]/stats["se"] if stats["se"] else None}
        arms[name] = {"primary_contact": primary_contact(paired["contact"],
            sum(p["positive_contact_changes"] for p in group), sum(p["negative_contact_changes"] for p in group)),
            "secondary_paired_changes": secondary,
            "source": {k: merge([p["source"][k] for p in group]) for k in OBSERVABLES},
            "endpoint": {k: merge([p["endpoint"][k] for p in group]) for k in OBSERVABLES},
            "population_paired_contact_means": [p["paired_change"]["contact"]["mean"] for p in group]}
    source = {}
    # Separately declared source-check family: 17 moments + histogram + trials.
    source_family_size = len(OBSERVABLES)+2
    critical = float(norm.ppf(1-FAMILY_ALPHA/(2*source_family_size)))
    for key in OBSERVABLES:
        stats = merge([p["source"][key] for p in populations]); target = refs["observables"][key]
        source[key] = {**stats, "reference": target, "error_over_se": (stats["mean"]-target)/stats["se"],
                       "estimated_family_interval_contains_reference": abs(stats["mean"]-target) <= critical*stats["se"]}
    hist = np.sum([p["source_radial_counts"] for p in populations], axis=0)
    expected = np.array(refs["radial_bin_probabilities"])*sum(hist)
    statistic = float(np.sum((hist-expected)**2/expected)); hist_p = float(chi2.sf(statistic, len(hist)-1))
    trial_stats = merge([p["source_radial_trials"] for p in populations])
    trial_se = math.sqrt(refs["source_radial_trials_variance"]/trial_stats["n"])
    trial_error = (trial_stats["mean"]-refs["source_radial_trials_mean"])/trial_se
    return {"schema": PROTOCOL, "references": refs, "observables": OBSERVABLES,
        "primary_family": {"alpha": FAMILY_ALPHA, "groups": GROUPS, "exact_test": "binomial sign conditional on nonzero paired contact changes"},
        "arms": arms, "populations": populations,
        "source_checks": {"family_size": source_family_size, "moments": source,
            "radial_histogram": {"bins": RADIAL_BINS, "counts": hist.tolist(), "expected_counts": expected.tolist(),
                "pearson_statistic": statistic, "asymptotic_p": hist_p, "bonferroni_adjusted_p": min(1., hist_p*source_family_size)},
            "radial_rejection_trials": {**trial_stats, "reference": refs["source_radial_trials_mean"],
                "known_geometric_se": trial_se, "error_over_known_se": trial_error,
                "estimated_family_interval_contains_reference": abs(trial_error) <= critical}},
        "source_sha256": sources,
        "interpretation": "Independent one-step paired test, not an evolving-chain convergence certificate. Source moment/histogram tests are asymptotic; contact paired test is exact under IID assumptions. Secondary diagnostics fully reported without selection. No physical protein inference."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = analyze(config, args.directory)
    result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__": main()
