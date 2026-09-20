#!/usr/bin/env python3
"""Stream every native-reference row, retaining only moments and a small top-weight heap."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import time


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(min(a, b)-max(a, b)))


def fresh():
    return {"sum": -math.inf, "square": -math.inf, "max": -math.inf, "nonzero": 0}


def add(moment, weight):
    moment["sum"] = logadd(moment["sum"], weight)
    moment["square"] = logadd(moment["square"], 2*weight)
    moment["max"] = max(moment["max"], weight)
    moment["nonzero"] += 1


def present(moment, count):
    if moment["nonzero"] == 0:
        return {"samples": count, "nonzero": 0, "log_normalizer": None, "ess": 0,
                "relative_SE": None, "maximum_point_fraction": None}
    ess = math.exp(2*moment["sum"]-moment["square"])
    return {"samples": count, "nonzero": moment["nonzero"],
            "log_normalizer": moment["sum"]-math.log(count), "ess": ess,
            "relative_SE": math.sqrt(max(0.0, (count/ess-1)/(count-1))) if count > 1 else None,
            "maximum_point_fraction": math.exp(moment["max"]-moment["sum"])}


def analyze(path_string):
    path = Path(path_string)
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    cover = json.loads((path / "cover.json").read_text())
    z, lam = manifest["activity"], manifest["lambda"]
    log_volume = math.log(cover["volume"])
    counts = {"q": 0, "capture": 0, "hard": 0, "valid": 0}
    moments = {name: fresh() for name in ["native", "native_core", "native_shell"]}
    top = []
    digest = hashlib.sha256()
    byte_count = 0
    n = 0
    for line in (path / "samples.jsonl").open("rb"):
        digest.update(line)
        byte_count += len(line)
        row = json.loads(line)
        assert row["draw"] == n
        n += 1
        if "zero" in row:
            counts[row["zero"]] += 1
            assert (row["q"] > 1) == (row["zero"] == "q")
            continue
        assert row["q"] <= 1
        counts["valid"] += 1
        logs = row["cloud_log_weights"]
        assert len(logs) == manifest["cloud_replicates"]
        mean = -math.inf
        for k, w in zip(row["cloud_overlap_counts"], logs):
            expected = z*row["lower_volume"] + k*math.log1p(z/lam)
            assert abs(w-expected) < 1e-10
            mean = logadd(mean, w)
        mean -= math.log(len(logs))
        assert abs(mean-row["log_boltzmann_mean"]) < 1e-10
        weight = mean+log_volume
        assert abs(weight-row["log_importance_weight"]) < 1e-10
        add(moments["native"], weight)
        add(moments["native_core" if row["q"] <= .8 else "native_shell"], weight)
        item = (weight, row["draw"], row)
        if len(top) < 12:
            heapq.heappush(top, item)
        elif item[:2] > top[0][:2]:
            heapq.heapreplace(top, item)
    assert n == summary["samples"] == manifest["samples"]
    assert digest.hexdigest() == summary["samples_sha256"]
    for key in ["q", "capture", "hard"]:
        assert counts[key] == summary[f"{key}_rejected"]
    for key, m in moments.items():
        assert m["nonzero"] == summary[key]["nonzero"]
        if m["nonzero"]:
            assert abs(m["sum"]-summary[key]["log_sum_weights"]) < 1e-8
            assert abs(m["square"]-summary[key]["log_sum_squared_weights"]) < 1e-8
    return {"replicate": path.name, "samples": n, "counts": counts, "moments": moments,
            "regions": {key: present(m, n) for key, m in moments.items()},
            "sample_bytes": byte_count, "sample_sha256": digest.hexdigest(),
            "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
            "physical_signature": {key: manifest[key] for key in ["config_sha256", "shape_sha256",
                                       "activity", "depletant_radius", "metric"]},
            "cover": cover,
            "top_weights": [dict(item[2], replicate=path.name) for item in sorted(top, reverse=True)],
            "cover_volume": cover["volume"], "cpu_seconds": summary["cpu_seconds"],
            "wall_seconds": summary["wall_seconds"], "raw_cloud_points": summary["raw_cloud_points"]}


def require_matching_targets(results):
    assert results, "No populations to combine"
    reference = results[0]
    for result in results[1:]:
        assert result["physical_signature"] == reference["physical_signature"], "Different physical targets"
        assert result["cover"] == reference["cover"], "Different native covers"
        assert result["cover_volume"] == reference["cover_volume"], "Different cover volumes"


def replicate_relative_se(regional_results, combined_log_q):
    if combined_log_q is None or len(regional_results) <= 1:
        return None
    ratios = [math.exp(result["log_normalizer"]-combined_log_q)
              if result["log_normalizer"] is not None else 0.0 for result in regional_results]
    k = len(ratios)
    return math.sqrt(sum((x-1)**2 for x in ratios)/(k*(k-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert all((Path(job["output"]) / "summary.json").exists() for job in manifest["jobs"])
    paths = [job["output"] for job in manifest["jobs"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, paths))
    require_matching_targets(results)
    n = sum(r["samples"] for r in results)
    assert n == manifest["total_unconditional_draws"]
    combined = {key: fresh() for key in ["native", "native_core", "native_shell"]}
    for result in results:
        for key, m in result["moments"].items():
            c = combined[key]
            c["sum"] = logadd(c["sum"], m["sum"])
            c["square"] = logadd(c["square"], m["square"])
            c["max"] = max(c["max"], m["max"])
            c["nonzero"] += m["nonzero"]
    regions = {key: present(m, n) for key, m in combined.items()}
    # Independent equal-size population means provide a second observed error diagnostic.
    assert len({r["samples"] for r in results}) == 1
    qlog = regions["native"]["log_normalizer"]
    rep_se = replicate_relative_se([r["regions"]["native"] for r in results], qlog)
    nonzero = combined["native"]["nonzero"]
    cover_volume = results[0]["cover_volume"]
    result = {"complete": True, "all_rows_and_hashes_validated": True,
              "analyzed_utc": datetime.now(timezone.utc).isoformat(), "samples": n,
              "regions": regions, "independent_population_relative_SE": rep_se,
              "cpu_seconds": sum(r["cpu_seconds"] for r in results),
              "max_population_wall_seconds": max(r["wall_seconds"] for r in results),
              "raw_cloud_points": sum(r["raw_cloud_points"] for r in results),
              "sample_bytes": sum(r["sample_bytes"] for r in results),
              "zero_activity_volume_estimate_A3": cover_volume*nonzero/n,
              "zero_activity_volume_SE_A3": cover_volume*math.sqrt((nonzero/n)*(1-nonzero/n)/n),
              "populations": results,
              "top_weights": sorted([row for r in results for row in r["top_weights"]],
                                    key=lambda row: row["log_importance_weight"], reverse=True)[:20],
              "analysis_wall_seconds": time.monotonic()-started,
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Complete geometric proposal support; fixed-budget independent estimate. "
                       "Observed errors and ESS do not certify unobserved weight concentration. "
                       "Previous campaign not pooled."}
    (args.root / "assessment-streaming.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ["populations", "top_weights"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
