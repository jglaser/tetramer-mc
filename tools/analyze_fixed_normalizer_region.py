#!/usr/bin/env python3
"""Freeze a discovered chart ellipsoid, then independently measure its mass."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from prepare_smc_normalizer_atlas import Density, read, relative_poses, sha, write
from analyze_basin_normalizers import moments, paired_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--model", type=Path, required=True)
    freeze.add_argument("--component", type=int, default=-1)
    freeze.add_argument("--physical-config", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--mahalanobis-radius", type=float, default=3.)
    analyze = subs.add_parser("analyze")
    analyze.add_argument("--region", type=Path, required=True)
    analyze.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.out.exists():
            parser.error("Frozen region already exists; do not overwrite")
        if not 0 < args.mahalanobis_radius < float("inf"):
            parser.error("Require positive finite radius")
        m, cfg = read(args.model), read(args.physical_config)
        index = args.component % len(m["weights"])
        guide = {k:m[k] for k in ["schema", "angular_length", "coordinate_convention", "shape_sha256"]}
        for key in ["anchors", "means", "covariances"]:
            guide[key] = [m[key][index]]
        guide["weights"] = [1.]
        definition = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            source_model=str(args.model.resolve()), source_model_sha256=sha(args.model), component_index=index,
            physical_config_sha256=sha(args.physical_config), fixed_neighbor=cfg["fixed_poses"][0],
            capture_center=cfg["capture_center"], capture_radius=cfg["capture_radius"],
            shape_sha256=m["shape_sha256"], activity=cfg["reservoir_density"], depletant_radius=cfg["depletant_radius"],
            physical_metric=cfg["metadata"], gaussian_chart=guide,
            mahalanobis_radius=args.mahalanobis_radius, minimum_original_q=5.,
            definition=f"hard-valid capture pose with original q>=5 and frozen chart Mahalanobis radius<={args.mahalanobis_radius:g}; boundary has zero measure",
            selection="Frozen from discovery endpoints before fresh importance production. Ellipsoid unchanged across proposal widths.",
            freezing_script_sha256=sha(__file__))
        write(args.out, definition)
        print(json.dumps(dict(region=str(args.out.resolve()),sha256=sha(args.out)),indent=2))
        return
    region, manifest = read(args.region), read(args.root / "manifest.json")
    cfg = read(args.root / "provenance/config.json")
    assert cfg["fixed_poses"] == [region["fixed_neighbor"]]
    assert cfg["metadata"] == region["physical_metric"]
    assert cfg["reservoir_density"] == region["activity"] and cfg["depletant_radius"] == region["depletant_radius"]
    assert sha(args.root / "provenance/shape.json") == region["shape_sha256"]
    density = Density(region["gaussian_chart"])
    all_logs, all_pairs, per_population = {}, {}, []
    names = ["discovered_ellipsoid", "far_complement", "capture_complement", "all_far", "total"]
    for name in names:
        all_logs[name], all_pairs[name] = [], []
    for job in manifest["jobs"]:
        directory = Path(job["directory"])
        assert read(directory / "summary.json")["complete"]
        rows = [json.loads(line) for line in (directory / "samples.jsonl").open()]
        assert len(rows) == job["samples"] and [r["draw"] for r in rows] == list(range(len(rows)))
        indices = [i for i,r in enumerate(rows) if r["hard_valid"]]
        distances = density.evaluate(relative_poses([rows[i]["pose"] for i in indices], region["fixed_neighbor"]))[1][:,0]
        valid = np.zeros(len(rows), bool); valid[indices] = True
        far = np.zeros(len(rows), bool)
        far[indices] = [rows[i]["q"] >= region["minimum_original_q"] for i in indices]
        inside = np.zeros(len(rows), bool)
        inside[indices] = distances <= region["mahalanobis_radius"]
        inside &= far
        masks = dict(discovered_ellipsoid=inside, far_complement=far & ~inside,
                     capture_complement=valid & ~inside, all_far=far, total=valid)
        stats = {}
        for name, mask in masks.items():
            logs = np.array([r["log_importance_weight"] if hit else -np.inf for hit,r in zip(mask,rows)])
            pairs = np.array([[c["log_weight"] - r["log_proposal_density"] for c in r["clouds"]]
                              if hit else [-np.inf,-np.inf] for hit,r in zip(mask,rows)])
            stats[name] = moments(logs)
            all_logs[name].append(logs); all_pairs[name].append(pairs)
        per_population.append(dict(id=job["id"],seed=job["seed"],estimates=stats))
    estimates = {}
    for name in names:
        logs, pairs = np.concatenate(all_logs[name]), np.concatenate(all_pairs[name])
        estimates[name] = moments(logs)
        estimates[name]["paired_noise"] = paired_noise(logs,pairs)
        poplogs = [p["estimates"][name]["logQ"] for p in per_population]
        estimates[name]["independent_population_estimates"] = moments([-np.inf if x is None else x for x in poplogs])
        estimates[name]["independent_population_estimates"]["logQ_values"] = poplogs
    result = dict(region=str(args.region.resolve()),region_sha256=sha(args.region),
        manifest_sha256=sha(args.root / "manifest.json"),estimates=estimates,populations=per_population,
        scope="Fresh fixed-N importance estimates with full proposal density and unconditional zero draws. Fixed discovered-region mass does not certify all far mass.")
    out = args.root / "assessment/fixed-region.json"
    out.parent.mkdir(exist_ok=True)
    write(out,result)
    print(json.dumps(dict(output=str(out),estimates=estimates),indent=2))


if __name__ == "__main__":
    main()
