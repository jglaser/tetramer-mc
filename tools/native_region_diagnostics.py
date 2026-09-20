#!/usr/bin/env python3
"""Extract valid reference poses and summarize concentration without retaining zero rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def bins(edges):
    return [{"lo": a, "hi": b, "count": 0, "scaled_weight": 0.0} for a, b in zip(edges, edges[1:])]


def add_bin(groups, x, weight):
    for i, group in enumerate(groups):
        if group["lo"] <= x < group["hi"] or (i == len(groups)-1 and x == group["hi"]):
            group["count"] += 1
            group["scaled_weight"] += weight
            return
    raise ValueError(f"Outside diagnostic bins: {x}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    assessment = json.loads((args.root / "assessment-streaming.json").read_text())
    manifest = json.loads((args.root / "manifest.json").read_text())
    cover = json.loads((Path(manifest["jobs"][0]["output"]) / "cover.json").read_text())
    assert cover["centroid"] == [0.0, 0.0, 0.0]
    ref = cover["reference"]
    offset = max(row["log_importance_weight"] for row in assessment["top_weights"])
    groups = {"q": bins([0,.2,.4,.6,.8,1.]),
              "translation_radius_A": bins([0,.25,.5,.75,1.,1.25,1.5,1.75,2.00000001]),
              "proper_angle_degrees": bins([0,1,2,3,4,5,7.5,11])}
    count, total = 0, 0.0
    extraction = args.root / "valid-native-poses.jsonl"
    with extraction.open("w") as output:
        for job in manifest["jobs"]:
            source = Path(job["output"]) / "samples.jsonl"
            with source.open("rb") as samples:
                for line in samples:
                    if b'"pose":' not in line:
                        continue
                    row = json.loads(line)
                    assert "zero" not in row
                    count += 1
                    w = math.exp(row["log_importance_weight"]-offset)
                    total += w
                    t = row["pose"]["position"]
                    radius = math.sqrt(sum((x-y)**2 for x,y in zip(t,ref["position"])))
                    q = row["pose"]["orientation"]
                    q0 = ref["orientation"]
                    dot = abs(sum(x*y for x,y in zip(q,q0)))/math.sqrt(sum(x*x for x in q)*sum(y*y for y in q0))
                    angle = math.degrees(2*math.acos(min(1.,dot)))
                    add_bin(groups["q"], row["q"], w)
                    add_bin(groups["translation_radius_A"], radius, w)
                    add_bin(groups["proper_angle_degrees"], angle, w)
                    row.update(replicate=f"r{job['replicate']:02d}", source_file=str(source),
                               translation_radius_A=radius, proper_angle_degrees=angle)
                    output.write(json.dumps(row,separators=(",",":"))+"\n")
    assert count == assessment["regions"]["native"]["nonzero"]
    assert abs(offset+math.log(total)-math.log(assessment["samples"])
               -assessment["regions"]["native"]["log_normalizer"]) < 1e-9
    for values in groups.values():
        for group in values:
            group["observed_weight_fraction"] = group.pop("scaled_weight")/total
            group["unweighted_valid_fraction"] = group["count"]/count
    top = []
    for row in assessment["top_weights"]:
        q = row["pose"]["orientation"]
        q0 = ref["orientation"]
        dot = abs(sum(x*y for x,y in zip(q,q0)))/math.sqrt(sum(x*x for x in q)*sum(y*y for y in q0))
        r = dict(row)
        r["translation_radius_A"] = math.sqrt(sum((x-y)**2 for x,y in zip(row["pose"]["position"],ref["position"])))
        r["proper_angle_degrees"] = math.degrees(2*math.acos(min(1.,dot)))
        r["observed_weight_fraction"] = math.exp(row["log_importance_weight"]-offset)/total
        top.append(r)
    result = {"valid_rows": count, "bins": groups, "leading_native_poses": top,
              "top_1_weight_fraction": top[0]["observed_weight_fraction"],
              "top_5_weight_fraction": sum(row["observed_weight_fraction"] for row in top[:5]),
              "top_20_weight_fraction": sum(row["observed_weight_fraction"] for row in top),
              "valid_pose_file": str(extraction),
              "valid_pose_file_sha256": hashlib.sha256(extraction.read_bytes()).hexdigest(),
              "valid_pose_file_bytes": extraction.stat().st_size,
              "scope": "Descriptive concentration and a derived valid-pose index, not a fitted proposal or replacement normalizer. Full unconditional zero accounting remains in original samples and manifest."}
    (args.root / "concentration.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!='leading_native_poses'},indent=2))


if __name__ == "__main__":
    main()
