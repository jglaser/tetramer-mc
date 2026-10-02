#!/usr/bin/env python3
"""Independent reference and attempted-state audit for the two-sphere control.

No protein classifier, trajectory resampling, or missing-row filtering. Errors
stop analysis. Batch and between-chain diagnostics do not certify convergence.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad


def overlap(d, a):
    return math.pi * (4 * a + d) * (2 * a - d)**2 / 12 if d < 2 * a else 0.0


def reference(config):
    a, z = config["exclusion_radius"], config["activity"]
    lo, hi = config["min_separation"], config["max_separation"]
    weight = lambda d: d*d * math.exp(z * overlap(d, a))
    def integral(f, left=lo, right=hi):
        return quad(f, left, right, epsabs=1e-11, epsrel=1e-11,
                    points=[v for v in [1., 2*a] if left < v < right])
    mass, error = integral(weight)
    return {"radial_mass": mass, "quadrature_absolute_error": error,
            "d": integral(lambda d: d * weight(d))[0] / mass,
            "contact": integral(weight, lo, min(1., hi))[0] / mass,
            "root_r2": 3 * config["root_radius"]**2 / 5,
            **{f"{stage}_{member}_q{k}2": .25 for stage in ["mh", "refreshed"]
               for member in ["root", "child"] for k in range(4)}}


def quaternion_product(a, b):
    w, x, y, z = a
    s, u, v, t = b
    return [w*s-x*u-y*v-z*t, w*u+x*s+y*t-z*v,
            w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s]


def angular_log_factor(q):
    # Four equally weighted identity-covariance Cayley charts centered on the
    # quaternion basis. In chart i, |c|²=(1-q_i²)/q_i². This reconstructs density
    # directly from quaternions, independently of the Rust matrix inversion.
    terms = []
    for value in q:
        square = value*value
        if square > 0:
            square = min(square, 1.)
            terms.append(-math.log(4) - .5*(1-square)/square
                         + 2*math.log(math.pi) - 2*math.log(square))
    maximum = max(terms)
    return maximum + math.log(math.fsum(math.exp(v-maximum) for v in terms))


def joint_log_density(state, config=None, proposal_kind="tree"):
    root, child = state
    qr, qc = root["orientation"], child["orientation"]
    relative = quaternion_product([qr[0], -qr[1], -qr[2], -qr[3]], qc)
    radial2 = math.fsum(x*x for x in root["position"])
    distance2 = math.fsum((x-y)**2 for x, y in zip(root["position"], child["position"]))
    root_g = -3*math.log(2*math.pi) - .5*radial2 + angular_log_factor(qr)
    child_g = -3*math.log(2*math.pi) - .5*distance2 + angular_log_factor(relative)
    if proposal_kind == "tree":
        return root_g + child_g
    if proposal_kind != "defensive_independent":
        raise ValueError("unknown frozen proposal kind")
    alpha, width = config["uniform_probability"], config["uniform_half_width"]
    # Quaternion conjugation computes the child's translation in the OLD root
    # frame, independently of the Rust matrix utility.
    delta = [x-y for x, y in zip(child["position"], root["position"])]
    inverse = [qr[0], -qr[1], -qr[2], -qr[3]]
    child_t = quaternion_product(quaternion_product(inverse, [0.]+delta), qr)[1:]
    def mixture(g, t):
        learned = math.log1p(-alpha) + g if alpha < 1 else -math.inf
        uniform = (math.log(alpha)-3*math.log(2*width)
                   if alpha > 0 and all(-width <= v <= width for v in t) else -math.inf)
        if learned == -math.inf: return uniform
        if uniform == -math.inf: return learned
        hi, lo = max(learned, uniform), min(learned, uniform)
        return hi + math.log1p(math.exp(lo-hi))
    return mixture(root_g, root["position"]) + mixture(child_g, child_t)


def close(a, b, tolerance=2e-8):
    if not (math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance*(1+abs(b))):
        raise ValueError(f"independent reconstruction differs: {a} versus {b}")


def distance(state):
    return math.sqrt(math.fsum((x-y)**2 for x, y in
                     zip(state[0]["position"], state[1]["position"])))


def validate_state(state):
    if len(state) != 2:
        raise ValueError("expected exactly two physical spheres")
    for pose in state:
        if len(pose["position"]) != 3 or len(pose["orientation"]) != 4:
            raise ValueError("invalid pose size")
        if not all(math.isfinite(v) for v in pose["position"] + pose["orientation"]):
            raise ValueError("nonfinite pose")
        close(sum(x*x for x in pose["orientation"]), 1., 2e-10)


def observables(row):
    values = {"d": distance(row["mh_state"]),
              "contact": float(distance(row["mh_state"]) < 1.),
              "root_r2": sum(x*x for x in row["mh_state"][0]["position"])}
    for prefix, field in [("mh", "mh_state"), ("refreshed", "state")]:
        for index, member in enumerate(["root", "child"]):
            for k, v in enumerate(row[field][index]["orientation"]):
                values[f"{prefix}_{member}_q{k}2"] = v*v
    return values


def audit_row(row, config, arm, previous=None, proposal_kind="tree"):
    if arm not in ["analytic", "poisson", "singleton_path"]:
        raise ValueError("unknown physical gate arm")
    if row.get("error") is not None:
        raise ValueError("failed attempt cannot enter the equilibrium analysis")
    if not math.isfinite(row["log_uniform"]) or row["log_uniform"] >= 0:
        raise ValueError("invalid Metropolis random threshold")
    old, trial = row["old_state"], row["proposed_state"]
    validate_state(old)
    validate_state(row["mh_state"])
    validate_state(row["state"])
    if previous is not None and old != previous:
        raise ValueError("chain continuity broken (a rejected state may be missing)")
    close(row["old_d"], distance(old))
    if not (sum(v*v for v in old[0]["position"]) <= config["root_radius"]**2
            and config["min_separation"] <= distance(old) <= config["max_separation"]):
        raise ValueError("old physical state is outside the frozen domain")
    if row["proposal_null"] != (trial is None):
        raise ValueError("null flag differs from endpoint")
    allowed = False
    max_density_error = 0.
    if trial is not None:
        validate_state(trial)
        d = distance(trial)
        close(row["proposed_d"], d)
        domain = (sum(v*v for v in trial[0]["position"]) <= config["root_radius"]**2
                  and config["min_separation"] <= d <= config["max_separation"])
        hard = d >= 2*config["core_radius"]
        if row["domain_valid"] != domain:
            raise ValueError("root-ball/relative-shell predicate differs")
        # Implementations may skip a hard query outside the integration domain.
        if domain and row["hard_valid"] != hard:
            raise ValueError("hard-sphere predicate differs")
        correction = (joint_log_density(old, config, proposal_kind)
                      - joint_log_density(trial, config, proposal_kind))
        close(row["q_correction"], correction)
        max_density_error = abs(row["q_correction"] - correction)
        analytic = config["activity"] * (overlap(d, config["exclusion_radius"])
                                         - overlap(distance(old), config["exclusion_radius"]))
        close(row["analytic_log_weight"], analytic)
        allowed = domain and hard
        if allowed:
            if arm in ["poisson", "singleton_path"]:
                gate = row["gate"]
                if gate is None:
                    raise ValueError("valid Poisson trial lost its gate")
                if any(type(gate[k]) is not int or gate[k] < 0
                       for k in ["gained", "lost", "retained_points", "raw_points"]):
                    raise ValueError("Poisson counters must be nonnegative integers")
                if not (gate["gained"] + gate["lost"] == gate["retained_points"] <= gate["raw_points"]):
                    raise ValueError("invalid Poisson counters")
                physical = ((gate["gained"] - gate["lost"])
                            * math.log1p(config["activity"] / config["auxiliary_intensity"]))
                close(gate["log_weight"], physical)
                if arm == "singleton_path":
                    path = row["path_gate"]
                    order = {"first_then_second": [0, 1], "second_then_first": [1, 0]}[path["order"]]
                    if path["ordered_members"] != order or path["aggregate"] != gate:
                        raise ValueError("path order or aggregate differs")
                    middle = list(old)
                    middle[order[0]] = trial[order[0]]
                    if path["intermediate_selected"] != middle:
                        raise ValueError("path intermediate is not an exact endpoint copy")
                    legs = path["legs"]
                    if len(legs) != 2:
                        raise ValueError("path must have two auxiliary legs")
                    for key in ["gained", "lost", "raw_points", "retained_points",
                                "retained_cells", "created_cells"]:
                        if gate[key] != sum(leg[key] for leg in legs):
                            raise ValueError("path counter sum differs")
                    for key in ["envelope_volume", "log_weight"]:
                        close(gate[key], sum(leg[key] for leg in legs))
                    for leg in legs:
                        if any(type(leg[k]) is not int or leg[k] < 0
                               for k in ["gained", "lost", "raw_points", "retained_points"]):
                            raise ValueError("invalid path leg counters")
                        if not (leg["gained"]+leg["lost"] == leg["retained_points"] <= leg["raw_points"]):
                            raise ValueError("path leg counter relationship differs")
                        close(leg["log_weight"], (leg["gained"] - leg["lost"])
                              * math.log1p(config["activity"] / config["auxiliary_intensity"]))
            else:
                if row["gate"] is not None:
                    raise ValueError("analytic arm unexpectedly sampled a cloud")
                physical = analytic
            close(row["log_ratio"], correction + physical)
    accept = allowed and row["log_uniform"] < min(0., row["log_ratio"])
    if row["accepted"] != accept:
        raise ValueError("MH decision differs from logged random threshold")
    expected = trial if accept else old
    if row["mh_state"] != expected:
        raise ValueError("accepted/rejected endpoint differs")
    if row["orientation_refresh_count"] != 2:
        raise ValueError("unexpected angular refresh schedule")
    if any(a["position"] != b["position"] for a, b in zip(row["mh_state"], row["state"])):
        raise ValueError("Haar refresh moved centers")
    close(row["d"], distance(row["state"]))
    return max_density_error


def series_diagnostics(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    result = {"mean": float(x.mean()), "n": n, "variance": float(x.var(ddof=1))}
    for batch in [256, 1024]:
        if n % batch or n < 4*batch:
            raise ValueError("frozen batching requires four or more complete batches")
        means = x.reshape(-1, batch).mean(axis=1)
        se = float(means.std(ddof=1) / math.sqrt(len(means)))
        ess = min(n, result["variance"] / (se*se)) if se > 0 and result["variance"] > 0 else None
        result[f"batch_{batch}"] = {"se": se, "ess": ess, "batches": len(means)}
    return result


def analyze(config, directory):
    reference_values = reference(config)
    chains = []
    source_hashes = {}
    for spec in config["chains"]:
        path = directory / f'{spec["id"]}.jsonl'
        digest = hashlib.sha256()
        values = {k: [] for k in reference_values if k not in ["radial_mass", "quadrature_absolute_error"]}
        previous = None
        n = accepted = raw = changes = exchanges = 0
        phase = None
        max_error = cpu = 0.
        first_cpu = None
        for raw_row in path.open("rb"):
            digest.update(raw_row)
            row = json.loads(raw_row)
            if row["chain_id"] != spec["id"] or row["step"] != n:
                raise ValueError("missing, reordered or wrong-chain attempt")
            burn = n < config["burn_in"]
            if row["burn_in"] != burn:
                raise ValueError("burn-in selection differs from frozen allocation")
            max_error = max(max_error, audit_row(row, config, spec["arm"], previous,
                                               spec.get("proposal_kind", "tree")))
            previous = row["state"]
            n += 1
            if row["cpu_seconds"] < cpu:
                raise ValueError("CPU clock went backwards")
            cpu = row["cpu_seconds"]
            if burn:
                first_cpu = cpu
                continue
            observations = observables(row)
            for key in values:
                values[key].append(observations[key])
            accepted += row["accepted"]
            raw += (row["gate"] or {}).get("raw_points", 0)
            new_phase = int(observations["contact"])
            if phase is not None and new_phase != phase:
                changes += 1
                exchanges += int(changes % 2 == 0)
            phase = new_phase
        if n != config["burn_in"] + config["production_steps"]:
            raise ValueError(f"incomplete chain {spec['id']}: {n}")
        source_hashes[str(path)] = digest.hexdigest()
        diagnostics = {k: series_diagnostics(v) for k, v in values.items()}
        production_cpu = cpu - (first_cpu or 0.)
        ess = diagnostics["contact"]["batch_1024"]["ess"]
        group = (f'{spec["proposal_kind"]}:{spec["arm"]}'
                 if "proposal_kind" in spec else spec["arm"])
        chains.append({**spec, "analysis_group": group, "attempts": n, "accepted_production": accepted,
                       "raw_points_production": raw, "production_cpu_seconds": production_cpu,
                       "contact_changes": changes, "completed_contact_exchanges": exchanges,
                       "contact_ess_per_cpu": ess / production_cpu if ess is not None else None,
                       "max_log_density_error": max_error, "observables": diagnostics})
    arms = {}
    for arm in sorted({c["analysis_group"] for c in chains}):
        group = [c for c in chains if c["analysis_group"] == arm]
        summary = {}
        for key, target in reference_values.items():
            if key in ["radial_mass", "quadrature_absolute_error"]:
                continue
            means = np.array([c["observables"][key]["mean"] for c in group])
            mean = float(means.mean())
            se = float(means.std(ddof=1) / math.sqrt(len(means)))
            within = math.sqrt(sum(c["observables"][key]["batch_1024"]["se"]**2 for c in group))/len(group)
            variances = [c["observables"][key]["variance"] for c in group]
            w = float(np.mean(variances))
            n = config["production_steps"]
            rhat = math.sqrt(((n-1)/n*w + float(means.var(ddof=1)))/w) if w > 0 else None
            summary[key] = {"mean": mean, "reference": target, "population_se": se,
                            "batch_se": within, "raw_rhat": rhat,
                            "error_over_population_se": (mean-target)/se if se > 0 else None,
                            "within_three_population_se": abs(mean-target) <= 3*se}
        arms[arm] = summary
    return {"schema": "dimer-tree-equilibrium-review-v1", "reference": reference_values,
            "chains": chains, "arms": arms, "source_sha256": source_hashes,
            "interpretation": "Toy equilibrium control only. No protein sampling or assembly inference. "
            "All attempted states retained; 256/1024 batch and four-chain diagnostics are finite-sample checks."}


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
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
