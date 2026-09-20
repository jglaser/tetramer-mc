#!/usr/bin/env python3
"""Check snapshot hard validity as prescribed native neighbors are added.

Snapshot fractions are geometric diagnostics, not physical basin occupations.
No rejection sampling, relaxation, or inference of zero region mass is used.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose["orientation"])[[1, 2, 3, 0]])


def placed(centers, pose):
    return rotation(pose).apply(centers) + pose["position"]


def first_clash(centers, radii, tree, fixed_centers):
    neighbors = tree.query_ball_point(centers, radii + radii.max() + 1e-10)
    for i, indices in enumerate(neighbors):
        if not indices:
            continue
        distances = np.linalg.norm(centers[i] - fixed_centers[indices], axis=1)
        gaps = distances - radii[i] - radii[indices]
        bad = np.flatnonzero(gaps < 0.)
        if len(bad):
            k = int(bad[0])
            return dict(moving_atom=i, fixed_atom=int(indices[k]), surface_gap_A=float(gaps[k]))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deep-summary", type=Path, default=ROOT / "runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r1/summary.json")
    parser.add_argument("--guided-campaign", type=Path, default=ROOT / "runs/basin-normalizer-importance-guided-16384-l64")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "provenance"
    archive.mkdir()
    inputs = {Path(__file__).resolve(), args.deep_summary.resolve()}
    shape_path = OLD / "inputs/tetramer-shape.json"
    inputs.add(shape_path)
    shape = read(shape_path)
    centers = np.asarray([atom["center"] for atom in shape["atoms"]])
    radii = np.asarray([atom["radius"] for atom in shape["atoms"]])
    environments = {}
    for count in (1, 2, 3):
        path = OLD / f"inputs/geometry/environment-site0-m{count}.json"
        inputs.add(path)
        environments[count] = read(path)
        shutil.copy2(path, archive / path.name)
        assert len(environments[count]["fixed_poses"]) == count
        assert environments[count]["fixed_poses"][:1] == environments[1]["fixed_poses"]
        assert environments[count]["native_poses"] == environments[1]["native_poses"]
        assert environments[count]["capture_center"] == environments[1]["capture_center"]
        assert environments[count]["capture_radius"] == environments[1]["capture_radius"]
    assert environments[3]["fixed_poses"][:2] == environments[2]["fixed_poses"]
    fixed = [placed(centers, pose) for pose in environments[3]["fixed_poses"]]
    trees = [cKDTree(points) for points in fixed]
    for i in range(3):
        for j in range(i):
            assert first_clash(fixed[i], radii, trees[j], fixed[j]) is None, "Prescribed fixed neighbors clash"

    deep = read(args.deep_summary)
    assert deep["complete"] and not deep["zero_estimate"]
    cohorts = {"deep_far": [dict(id=f"endpoint-{i}", pose=row["pose"], original_q=5 * row["q"])
                            for i, row in enumerate(deep["final_particles"])], "native": [], "shoulder": []}
    native_sources = sorted((OLD / "runs").glob("site0-m1-r1.5-z0.035-native-*/summary.json"))
    assert len(native_sources) == 8
    for path in native_sources:
        inputs.add(path)
        population = read(path)
        assert population["complete"]
        indices = np.linspace(0, len(population["final_particles"]) - 1, 64, dtype=int)
        for i in indices:
            row = population["final_particles"][int(i)]
            cohorts["native"].append(dict(id=f"{path.parent.name}-{i}", pose=row["pose"], original_q=row["q"]))
    guided = read(args.guided_campaign / "manifest.json")
    inputs.add(args.guided_campaign / "manifest.json")
    for job in guided["jobs"]:
        path = Path(job["directory"]) / "samples.jsonl"
        inputs.add(path)
        candidates = []
        for line in path.open():
            row = json.loads(line)
            if row["hard_valid"] and 1 < row["q"] < 2:
                candidates.append(row)
        for i in np.linspace(0, len(candidates) - 1, min(128, len(candidates)), dtype=int):
            row = candidates[int(i)]
            cohorts["shoulder"].append(dict(id=f"{job['id']}-{row['draw']}", pose=row["pose"], original_q=row["q"]))
    assert len(cohorts["native"]) == len(cohorts["deep_far"]) == len(cohorts["shoulder"]) == 512

    results = {}
    for name, rows in cohorts.items():
        records = []
        for row in rows:
            points = placed(centers, row["pose"])
            witnesses = [first_clash(points, radii, trees[j], fixed[j]) for j in range(3)]
            assert witnesses[0] is None, f"Source snapshot fails original hard constraint: {name}/{row['id']}"
            assert np.linalg.norm(np.asarray(row["pose"]["position"]) - environments[1]["capture_center"]) <= environments[1]["capture_radius"]
            records.append(dict(row, clash_witness_by_neighbor=witnesses,
                                hard_valid_by_neighbor_count={str(n): all(w is None for w in witnesses[:n]) for n in (1, 2, 3)}))
        results[name] = dict(snapshot_count=len(records),
            hard_valid_counts={str(n): sum(row["hard_valid_by_neighbor_count"][str(n)] for row in records) for n in (1, 2, 3)},
            snapshots=records)
        print(json.dumps(dict(cohort=name, counts=results[name]["hard_valid_counts"])), flush=True)
    shutil.copy2(shape_path, archive / "shape.json")
    shutil.copy2(__file__, archive / Path(__file__).name)
    payload = dict(results=results, inputs={str(path): sha(path) for path in sorted(inputs)},
        method="Direct atomic core distances; KD tree only enumerates all possible overlapping pairs. No depletant-pair approximation.",
        selection="All512 deep discovery endpoints;64 evenly spaced endpoints from each of8 old native populations;128 evenly spaced valid shoulder draws from each of4 fresh importance populations. No statistical weighting.",
        scope="Compatibility of inspected snapshots with prescribed neighbors. Snapshot fractions are not equilibrium weights. Zero survivors do not bound the entire continuous region or exclude relaxation routes.")
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Geometric survival under added native neighbors", "", payload["scope"], "",
             "| Snapshot cohort | Inspected | One neighbor | Two neighbors | Three neighbors |", "|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        counts = result["hard_valid_counts"]
        lines.append(f"| {name} | {result['snapshot_count']} | {counts['1']} | {counts['2']} | {counts['3']} |")
    lines += ["", payload["selection"], "", "Every rejected snapshot records an explicit clashing atomic pair and its negative core gap. Positive survivors still require a physical free-energy calculation.", ""]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
