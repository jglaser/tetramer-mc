#!/usr/bin/env python3
"""Audit a frozen native hybrid protocol and estimate its paired-cloud noise.

The two clouds share a pose. Their difference estimates conditional cloud
variance; it does not provide two independent pose samples or a tail bound.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from analyze_native_region_reference import add, fresh, logadd, present


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty():
    return {"mean": fresh(), "clouds": [fresh(), fresh()], "log_noise_sum": -math.inf}


def accumulate(moment, log_a, log_b):
    add(moment["clouds"][0], log_a)
    add(moment["clouds"][1], log_b)
    add(moment["mean"], logadd(log_a, log_b)-math.log(2))
    if log_a != log_b:
        # expm1 preserves an accurate difference for nearly identical weights.
        log_difference = max(log_a, log_b)+math.log(-math.expm1(-abs(log_a-log_b)))
        moment["log_noise_sum"] = logadd(moment["log_noise_sum"], 2*log_difference-math.log(4))


def describe(moment, count):
    result = {"mean": present(moment["mean"], count),
              "individual_clouds": [present(m, count) for m in moment["clouds"]]}
    m = moment["mean"]
    if not m["nonzero"]:
        return dict(result, cloud_only_relative_SE=None, observed_cloud_variance_fraction=None)
    noise = moment["log_noise_sum"]
    result["cloud_only_relative_SE"] = math.exp(.5*noise-m["sum"]) if noise != -math.inf else 0.
    delta = 2*m["sum"]-math.log(count)-m["square"]
    # Constant weights have no observed total variance; a fraction is undefined.
    if delta >= 0 or count < 2:
        result["observed_cloud_variance_fraction"] = None
    elif noise == -math.inf:
        result["observed_cloud_variance_fraction"] = 0.
    else:
        centered = m["square"]+math.log(-math.expm1(delta))
        result["observed_cloud_variance_fraction"] = (count-1)/count*math.exp(noise-centered)
    return result


def analyze(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    campaigns = []
    physical_signatures = []
    planned_jobs = protocol.get("jobs") or [dict(protocol, name="confirmation")]
    for planned in planned_jobs:
        command = planned.get("command", [])
        def expected_guide(key):
            if "guide_"+key in protocol:
                return protocol["guide_"+key]
            flag = "--model-"+key.replace("_", "-")
            value = command[command.index(flag)+1]
            return int(value) if key == "anchor_index" else float(value)
        root = Path(planned["root"])
        campaign = json.loads((root/"manifest.json").read_text())
        audited = json.loads((root/"assessment-streaming.json").read_text())
        assert audited["all_rows_and_hashes_validated"] and audited["complete"]
        audited_populations = {item["replicate"]: item for item in audited["populations"]}
        assert sha(root/"provenance/native-region-normalizer") == protocol["binary_sha256"] == campaign["binary_sha256"]
        assert sha(root/"provenance/guide-model.json") == planned["model_sha256"]
        assert [job["seed"] for job in campaign["jobs"]] == planned["seeds"]
        pooled = {region: empty() for region in ["native", "native_core", "native_shell"]}
        families = {label: fresh() for label in ["cover", "learned", "uniform"]}
        count = 0
        populations = []
        input_hashes = []
        for job in campaign["jobs"]:
            out = Path(job["output"])
            manifest = json.loads((out/"manifest.json").read_text())
            summary = json.loads((out/"summary.json").read_text())
            assert summary["complete"]
            assert manifest["schema"] == 3
            assert manifest["samples"] == protocol["samples_per_population"]
            assert manifest["seed"] == job["seed"]
            assert manifest["lambda_ratio"] == protocol["lambda_ratio"]
            assert manifest["cloud_replicates"] == protocol["cloud_replicates"] == 2
            assert manifest["cover_mixture"]["scales"] == protocol.get("cover_scales", [1])
            assert manifest["executable_sha256"] == protocol["binary_sha256"]
            assert sha(out/"provenance/config.json") == protocol["config_sha256"] == manifest["config_sha256"]
            guide = manifest["guide"]
            assert guide["model_sha256"] == sha(out/"provenance/guide-model.json") == planned["model_sha256"]
            for key in ["weight", "uniform_probability", "anchor_index"]:
                assert guide[key] == expected_guide(key)
            physical = {key: manifest[key] for key in ["config_sha256", "shape_sha256", "activity", "depletant_radius", "metric"]}
            physical_signatures.append(physical)
            local = {region: empty() for region in pooled}
            rows = 0
            with (out/"samples.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row["draw"] == rows
                    rows += 1
                    if "zero" in row:
                        continue
                    logs = [x-row["log_proposal_density"] for x in row["cloud_log_weights"]]
                    assert len(logs) == 2 and all(math.isfinite(x) for x in logs)
                    mean = logadd(*logs)-math.log(2)
                    assert abs(mean-row["log_importance_weight"]) < 1e-10
                    regions = ["native", "native_core" if row["q"] <= .8 else "native_shell"]
                    for region in regions:
                        accumulate(local[region], *logs)
                        accumulate(pooled[region], *logs)
                    source = "cover" if row["proposal_family"] == "cover" else row["guide_branch"]
                    add(families[source], mean)
            assert rows == manifest["samples"]
            count += rows
            populations.append({"replicate": out.name, "seed": job["seed"],
                                "regions": {r: describe(m, rows) for r, m in local.items()}})
            sample_sha256 = sha(out/"samples.jsonl")
            summary_sha256 = sha(out/"summary.json")
            assert sample_sha256 == summary["samples_sha256"] == audited_populations[out.name]["sample_sha256"]
            assert summary_sha256 == audited_populations[out.name]["summary_sha256"]
            input_hashes.append({"replicate": out.name, "samples_sha256": sample_sha256,
                                 "manifest_sha256": sha(out/"manifest.json"), "summary_sha256": summary_sha256})
        for region, moment in pooled.items():
            result = describe(moment, count)["mean"]
            target = audited["regions"][region]
            assert result["samples"] == target["samples"] and result["nonzero"] == target["nonzero"]
            if result["log_normalizer"] is not None:
                assert abs(result["log_normalizer"]-target["log_normalizer"]) < 1e-10
        campaigns.append({"name": planned["name"], "root": str(root), "samples": count,
                          "regions": {r: describe(m, count) for r, m in pooled.items()},
                          "source_weight_fractions": {label: math.exp(m["sum"]-pooled["native"]["mean"]["sum"]) if m["nonzero"] else 0.
                                                      for label, m in families.items()},
                          "populations": populations, "input_hashes": input_hashes,
                          "assessment_sha256": sha(root/"assessment-streaming.json")})
    assert all(s == physical_signatures[0] for s in physical_signatures)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": str(protocol_path),
            "protocol_sha256": sha(protocol_path), "script_sha256": sha(__file__),
            "all_frozen_inputs_and_budgets_match": True, "common_physical_signature": physical_signatures[0],
            "campaigns": campaigns,
            "formula": "noise_sum=sum((a-b)^2/4); cloud_RSE=sqrt(noise_sum)/sum(m); fraction=(N-1)/N*noise_sum/(sum(m^2)-sum(m)^2/N), m=(a+b)/2, a=W1/g, b=W2/g",
            "scope": "Conditional cloud-noise diagnostic from paired independent clouds sharing each pose. Includes every unconditional draw. No pooling across guides; no tail certification. Fractions may exceed one in finite samples and are not clamped. Source weight fractions are contributions to an estimator, not thermodynamic mixture fractions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.protocol)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for campaign in result["campaigns"]:
        print(campaign["name"], json.dumps(campaign["regions"]["native"], indent=2))


if __name__ == "__main__":
    main()
