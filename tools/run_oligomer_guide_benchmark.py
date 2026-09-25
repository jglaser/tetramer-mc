#!/usr/bin/env python3
"""Freeze, drain, and analyze a bounded 512-phase oligomer-guide comparison.

No simulation starts unless the explicit `run` action is supplied. The preparation
copies both original configurations and their provenance before materializing two
common frozen-state inputs. All counts use complete fixed-duration phase probes,
not a success threshold or CPU-limited stopping rule.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

ARMS = (0, 1, 4, 16)
POPULATIONS = 4
PHASES = 16
MAX_JOBS = 2
MASTER_SEED = 202609251400
SCOPE = ("Frozen whole-system phase transitions, not equilibrium trajectories, "
         "contact ESS, assembly stability, or physical kinetics.")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    raw = source.read_bytes()
    destination.write_bytes(raw)
    return {"source": str(source), "copy": str(destination.resolve()),
            "sha256": digest(raw), "bytes": len(raw)}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(repo, out):
    check(not out.exists(), "Output exists; choose a fresh directory")
    binary = repo / "target/release/oligomer-guide-benchmark"
    check(binary.is_file(), "Build the oligomer-guide-benchmark binary first")
    embedded = subprocess.run([str(binary), "--source-manifest"], cwd=repo, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    bundle = json.loads(embedded)
    for relative, source in bundle["files"].items():
        check(file_digest(repo / relative) == source["sha256"],
              f"Binary does not match current source: {relative}")
    early = repo / "runs/cluster-selection-diagnostic-20260925/input"
    late = repo / "runs/cluster-phase-protein-pilot-20260925"
    early_manifest = read(early / "freeze-manifest.json")
    late_manifest = read(late / "manifest.json")
    sources = {
        "early-config.json": early / "config.json",
        "early-trajectory.jsonl": early / "trajectory.jsonl",
        "early-freeze-manifest.json": early / "freeze-manifest.json",
        "early-run-manifest.json": early / "manifest.json",
        "late-config.json": late / "cluster-p0.json",
        "late-pilot-manifest.json": late / "manifest.json",
    }
    for name in ("config.json", "trajectory.jsonl"):
        check(file_digest(early / name) == early_manifest["inputs"][name]["sha256"],
              f"Early frozen {name} hash mismatch")
    late_entry = next(c for c in late_manifest["configs"]
                      if c["arm"] == "cluster" and c["population"] == 0)
    check(file_digest(sources["late-config.json"]) == late_entry["sha256"],
          "Late frozen config hash mismatch")
    frames = [json.loads(line) for line in sources["early-trajectory.jsonl"].read_text().splitlines()]
    endpoint = next(f for f in frames if f["sweep"] == 600)
    check(frames[-1]["sweep"] == 600, "Early prefix must end at frozen sweep 600")
    early_config = read(sources["early-config.json"])
    late_config = read(sources["late-config.json"])
    check(len(endpoint["poses"]) == len(late_config["initial_poses"]) == 264,
          "Frozen allocation expects 264 tetramers")
    model_source = Path(late_manifest["model"])
    model_hash = file_digest(model_source)
    check(model_hash == late_manifest["model_sha256"], "Frozen model hash mismatch")
    check(model_hash == read(early / "manifest.json")["model_sha256"],
          "Frozen preparations used different models")
    shape_source = Path(early_config["shape"])
    shape_hash = file_digest(shape_source)
    check(shape_hash == early_manifest["config_shape_sha256"], "Early shape hash mismatch")
    check(file_digest(late_config["shape"]) == shape_hash, "Frozen shapes differ")
    for field in ("boundary", "box_lengths", "depletant_radius", "reservoir_density",
                  "poisson_lambda_ratio", "learned_uniform_weight", "endpoint_gate", "seed_labels"):
        check(early_config[field] == late_config[field], f"Preparations differ in {field}")
    check(early_config["depletant_radius"] == 1.4 and early_config["reservoir_density"] == 0.0275,
          "This allocation is for growth bath 1.4 A / 0.0275 A^-3")
    for config in (early_config, late_config):
        for key in ("assembly_bias", "auxiliary_transport", "reversible_jump", "contact_memory",
                    "conditional_closure", "atlas_transport", "atlas_mask"):
            check(config.get(key) is None, f"Unsupported biased/adaptive configuration: {key}")
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    originals = inputs / "originals"
    originals.mkdir()
    archived = {name: archive(path, originals / name) for name, path in sources.items()}
    (inputs / "source-bundle.json").write_bytes(embedded)
    archived["source-bundle.json"] = {"source": "compiled executable --source-manifest",
                                      "copy": str((inputs / "source-bundle.json").resolve()),
                                      "sha256": digest(embedded), "bytes": len(embedded)}
    archived["runner.py"] = archive(Path(__file__), inputs / "runner.py")
    archived["shape.json"] = archive(shape_source, inputs / "shape.json")
    archived["model.json"] = archive(model_source, inputs / "model.json")
    common_phase = {
        "duration": 0.01, "dimer_rate": 1.0, "trimer_rate": 0.25,
        "transport_probability": 0.5, "correlation": 0.9,
        "local_translation_std_A": 0.2, "local_small_angle_std_degrees": 1.0,
    }
    preparations = []
    for name, source_sweep, config in (("early600", 600, early_config),
                                        ("stalled27000", 27000, late_config)):
        config = json.loads(json.dumps(config))
        if name == "early600":
            config["initial_poses"] = endpoint["poses"]
            config["seed_labels"] = endpoint["seed_labels"]
        config["shape"] = str((inputs / "shape.json").resolve())
        config["cluster_phase"] = common_phase.copy()
        config["metadata"] = {
            "protocol": "oligomer-guide-frozen-phase-allocation-v1", "preparation": name,
            "source_sweep": source_sweep, "scope": SCOPE,
            "source_metadata": config.get("metadata"),
            "poses_are_in_sphere_center_frame": True,
            "native_informed_proposal": True, "preparation_equilibrated": False,
        }
        path = inputs / f"{name}.json"
        save(path, config)
        preparations.append({"name": name, "source_sweep": source_sweep,
                             "config": str(path.resolve()), "sha256": file_digest(path)})
    jobs = []
    # Deterministic rotation distributes any order/thermal effects across arms.
    for population in range(POPULATIONS):
        for preparation in preparations:
            order = ARMS[population % 4:] + ARMS[:population % 4]
            for steps in order:
                name = f"{preparation['name']}-m{steps}-p{population}"
                jobs.append({"name": name, "preparation": preparation["name"],
                             "steps": steps, "population": population,
                             "config": preparation["config"],
                             "out": str((out / name).resolve())})
    manifest = {
        "schema": "oligomer-guide-frozen-phase-allocation-v1", "scope": SCOPE,
        "repo": str(repo), "binary": str(binary.resolve()),
        "executable_sha256": file_digest(binary), "inputs": archived,
        "preparations": preparations, "arms": list(ARMS), "populations": POPULATIONS,
        "phases_per_job": PHASES, "total_phase_allocation": len(jobs) * PHASES,
        "maximum_simultaneous_jobs": MAX_JOBS, "master_seed": MASTER_SEED,
        "guide": {"anchor_count": 4, "score_power": 1.0},
        "cluster_phase": common_phase, "model": str((inputs / "model.json").resolve()),
        "jobs": jobs,
        "analysis": {
            "strata": ["preparation", "subset_size", "whole_or_embedded", "seed_connectivity", "proposal_branch"],
            "primary": ["completed_exchanges_per_sampler_cpu_second", "accepted_contact_changes_per_sampler_cpu_second"],
            "secondary": ["hard_valid_per_attempt", "accepted_nonidentity_per_attempt", "proposal_correction_distribution",
                          "retained_contact_fingerprint_distance", "conditional_contact_change_acceptance_sum"],
            "uncertainty": "Four independent population values and paired differences; Student-t interval descriptive only at n=4",
            "zero_events": "Report count and exact attempted exposure; no thermodynamic or equilibrium conclusion",
            "one_step": "Delayed acceptance cannot improve per-candidate acceptance, but can avoid depletion work; exclude inner self-loops from actual moved-endpoint counts",
            "failures": "Drain independent jobs and fixed phase allocations; any failed phase makes campaign incomplete",
        },
    }
    save(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "jobs": len(jobs), "phases": len(jobs)*PHASES}))


def run(out):
    manifest = read(out / "manifest.json")
    check(not (out / "execution.json").exists(), "Execution already recorded; use fresh allocation")
    check(file_digest(manifest["binary"]) == manifest["executable_sha256"], "Executable changed after allocation freeze")
    for source in manifest["inputs"].values():
        check(file_digest(source["copy"]) == source["sha256"], f"Archived input changed: {source['copy']}")
    check(file_digest(Path(__file__)) == manifest["inputs"]["runner.py"]["sha256"], "Runner changed after allocation freeze")
    for preparation in manifest["preparations"]:
        check(file_digest(preparation["config"]) == preparation["sha256"], "Materialized preparation changed")
    for job in manifest["jobs"]:
        check(not Path(job["out"]).exists(), f"Job already exists: {job['out']}")
        check(not (out / f"{job['name']}.log").exists(), "Job log exists; refuse implicit retry")
    def execute(job):
        command = [manifest["binary"], "--config", job["config"], "--model", manifest["model"],
                   "--out", job["out"], "--preparation", job["preparation"],
                   "--steps", str(job["steps"]), "--population", str(job["population"]),
                   "--phases", str(manifest["phases_per_job"]), "--seed", str(manifest["master_seed"]),
                   "--anchor-count", str(manifest["guide"]["anchor_count"]),
                   "--score-power", str(manifest["guide"]["score_power"])]
        begin = time.monotonic()
        result = {**job, "command": command}
        try:
            with (out / f"{job['name']}.log").open("x") as log:
                process = subprocess.run(command, cwd=manifest["repo"], stdout=log, stderr=subprocess.STDOUT)
            result["returncode"] = process.returncode
        except Exception as error:
            result.update(returncode=-1, error=repr(error))
        result["wall_seconds"] = time.monotonic() - begin
        save(out / f"{job['name']}-status.json", result)
        print(json.dumps({k: result[k] for k in ("name", "returncode", "wall_seconds")}), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=manifest["maximum_simultaneous_jobs"]) as pool:
        futures = [pool.submit(execute, job) for job in manifest["jobs"]]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda row: row["name"])
    save(out / "execution.json", results)
    analyze(out)
    return 0 if all(row["returncode"] == 0 for row in results) else 1


def describe(values):
    n = len(values)
    if not n:
        return {"n": 0, "values": []}
    result = {"n": n, "values": values, "mean": statistics.mean(values)}
    if n > 1:
        se = statistics.stdev(values) / math.sqrt(n)
        result["population_SE"] = se
        if n == 4:
            result["population_t95_interval"] = [result["mean"] - 3.182446305*se,
                                                   result["mean"] + 3.182446305*se]
    return result


def analyze(out):
    manifest = read(out / "manifest.json")
    populations = []
    failed = []
    for job in manifest["jobs"]:
        path = Path(job["out"]) / "summary.json"
        if not path.exists():
            failed.append({"name": job["name"], "reason": "summary missing"})
            continue
        summary = read(path)
        populations.append(summary)
        if not summary["complete"]:
            failed.append({"name": job["name"], "reason": "phase failure"})
    fields = ("completed_exchanges_per_sampler_cpu_second", "accepted_contact_changes_per_sampler_cpu_second",
              "sampler_cpu_seconds", "diagnostics_cpu_seconds", "guide_cpu_seconds", "guide_cpu_fraction")
    reports = []
    paired = []
    for preparation in manifest["preparations"]:
        name = preparation["name"]
        baseline = {p["population"]: p for p in populations if p["preparation"] == name and p["steps"] == 0 and p["complete"]}
        for steps in ARMS:
            selected = [p for p in populations if p["preparation"] == name and p["steps"] == steps and p["complete"]]
            selected.sort(key=lambda row: row["population"])
            report = {"preparation": name, "steps": steps,
                      "populations": [p["population"] for p in selected],
                      "population_statistics": {field: describe([p[field] for p in selected]) for field in fields},
                      "attempted": sum(p["metrics"]["attempted"] for p in selected),
                      "accepted_nonidentity": sum(p["metrics"]["accepted_pose_changes"] for p in selected),
                      "hard_valid": sum(p["metrics"]["hard_valid"] for p in selected),
                      "attachments": sum(p["metrics"]["accepted_attachments"] for p in selected),
                      "detachments": sum(p["metrics"]["accepted_detachments"] for p in selected),
                      "exchanges": sum(p["metrics"]["completed_exchanges"] for p in selected),
                      "counts": [p["counts"] for p in selected],
                      "group_reports": [{"population": p["population"], "groups": p["groups"]} for p in selected]}
            reports.append(report)
            if steps:
                comparable = [p for p in selected if p["population"] in baseline]
                paired.append({"preparation": name, "steps": steps,
                               "differences_vs_baseline": {field: describe([p[field] - baseline[p["population"]][field]
                                                                             for p in comparable]) for field in fields}})
    population_rows = []
    for p in sorted(populations, key=lambda x: (x["preparation"], x["steps"], x["population"])):
        metrics, counts = p["metrics"], p["counts"]
        population_rows.append({
            "preparation": p["preparation"], "steps": p["steps"], "population": p["population"],
            "complete": p["complete"], "attempted": metrics["attempted"],
            "hard_valid": metrics["hard_valid"], "physical_accepted": metrics["physical_accepted"],
            "accepted_pose_changes": metrics["accepted_pose_changes"],
            "proposed_attachments": metrics["proposed_attachments"],
            "proposed_detachments": metrics["proposed_detachments"],
            "proposed_exchanges": metrics["proposed_exchanges"],
            "accepted_attachments": metrics["accepted_attachments"],
            "accepted_detachments": metrics["accepted_detachments"],
            "completed_exchanges": metrics["completed_exchanges"],
            "contact_change_acceptance_sum": metrics["contact_change_acceptance_sum"],
            "contact_change_acceptance_sum_per_cpu_second": metrics["contact_change_acceptance_sum"] / p["sampler_cpu_seconds"],
            "gate_raw_points": counts["gate_raw_points"],
            "guide_counts": metrics["guide_counts"],
            "guide_changed_endpoints": metrics["guide_changed_endpoints"],
            "sampler_cpu_seconds": p["sampler_cpu_seconds"],
            "guide_cpu_seconds": p["guide_cpu_seconds"], "guide_cpu_fraction": p["guide_cpu_fraction"],
            "guide_density_cpu_seconds": p["guide_density_cpu_seconds"],
            "guide_geometry_cpu_seconds": p["guide_geometry_cpu_seconds"],
            "proposal_correction_contact_changing": metrics["proposal_correction_contact_changing"],
            "sampled_depletion_hard_valid": metrics["sampled_depletion_hard_valid"],
            "completed_exchanges_per_sampler_cpu_second": p["completed_exchanges_per_sampler_cpu_second"],
            "accepted_contact_changes_per_sampler_cpu_second": p["accepted_contact_changes_per_sampler_cpu_second"],
        })
    complete = not failed and len(populations) == len(manifest["jobs"])
    save(out / "analysis.json", {"complete": complete, "scope": SCOPE, "failures": failed,
                                 "reports": reports, "paired": paired, "population_rows": population_rows})
    lines = ["# Frozen-state oligomer guide benchmark", "", SCOPE, "",
             "All events, including rejected and unchanged endpoints, remain in attempted denominators. "
             "The four populations share each preparation but have independent random streams; arms are paired by stream. "
             "This is not a physical equilibrium or mixing-time assessment.", "",
             f"Allocation complete: **{complete}**. {len(populations)}/{len(manifest['jobs'])} job summaries.", "",
             "| Preparation | Inner steps | Attempts | Hard-valid | Accepted changed poses | Attach / detach / exchange | Sampler CPU s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        cpu = sum(p["sampler_cpu_seconds"] for p in populations if p["preparation"] == r["preparation"] and p["steps"] == r["steps"] and p["complete"])
        lines.append(f"| {r['preparation']} | {r['steps']} | {r['attempted']} | {r['hard_valid']} | {r['accepted_nonidentity']} | "
                     f"{r['attachments']} / {r['detachments']} / {r['exchanges']} | {cpu:.3f} |")
    lines += ["", "Inner-step count zero is the existing handle-transport phase. One inner step is a delayed-acceptance control: "
              "it can save physical-gate work, but cannot improve acceptance of an otherwise identical candidate. "
              "Accepted guide self-loops are excluded from changed-pose counts.", "",
              "Population values, paired differences, proposal-correction distributions and context strata are in `analysis.json` "
              "and each job's `summary.json`. Raw production events and complete phase endpoints are retained per job. "
              "Zero observed exchanges do not establish that an arrangement is thermodynamically unstable.", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--repo", type=Path, default=Path("/home/xvg/tetramer-mc"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    if args.action == "prepare":
        prepare(repo, out)
    elif args.action == "run":
        return run(out)
    else:
        analyze(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
