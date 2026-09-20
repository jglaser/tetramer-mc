#!/usr/bin/env python3
"""Freeze native-informed shoulder guides using independent earlier IS data.

Importance weights and their tempered versions choose a proposal fit only.
They are not physical mixture weights and must not be reused as fresh evidence.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from prepare_smc_normalizer_atlas import ROOT, arrays, candidate_audit, read, relative_poses, sha, write


def physical_signature(root):
    cfg = read(root / "provenance/config.json")
    result = {key: cfg[key] for key in ["capture_center", "capture_radius", "fixed_poses", "depletant_radius", "reservoir_density"]}
    result["shape_sha256"] = sha(root / "provenance/shape.json")
    result["metric"] = {key: cfg["metadata"][key] for key in ["native_poses", "rigid_members", "member_error_scale", "angle_error_scale_deg"]}
    return cfg, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path,
                        default=ROOT / "runs/smc-normalizer-local-nativewide/site0/model.json")
    parser.add_argument("--campaign", type=Path, action="append")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/smc-normalizer-importance-guided/site0")
    parser.add_argument("--chart-component", type=int, default=0)
    parser.add_argument("--retained-weight", type=float, default=.5)
    parser.add_argument("--weight-powers", type=float, nargs="+", default=[1., .5])
    parser.add_argument("--covariance-floor-fraction", type=float, default=.25)
    parser.add_argument("--candidate-probes-per-component", type=int, default=128)
    args = parser.parse_args()
    if not 0 < args.retained_weight < 1 or not args.weight_powers or any(not 0 < p < float("inf") for p in args.weight_powers):
        parser.error("retained-weight must be in (0,1); weight powers must be positive finite")
    if not 0 < args.covariance_floor_fraction < float("inf") or args.candidate_probes_per_component < 1:
        parser.error("positive finite covariance floor and candidate count required")
    campaigns = args.campaign or [ROOT / f"runs/basin-normalizer-local-width{s}-8192-l64" for s in [1, 2, 4]] + [ROOT / "runs/basin-normalizer-local-nativewide-8192-l64"]
    campaigns = [path.resolve() for path in campaigns]
    original = read(args.source_model)
    count = len(original["weights"])
    k = args.chart_component
    assert 0 <= k < count and abs(sum(original["weights"]) - 1) < 1e-12
    assert original["coordinate_convention"] == "anchor-body-relative"
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "provenance"
    archive.mkdir()
    shutil.copy2(args.source_model, archive / "source-model.json")
    shutil.copy2(__file__, archive / "prepare_importance_guided_normalizer_atlas.py")
    shutil.copy2(Path(__file__).with_name("prepare_smc_normalizer_atlas.py"), archive / "prepare_smc_normalizer_atlas.py")
    target, cfg, selected, sources, seeds = None, None, [], [], set()
    total_draws = 0
    for index, root in enumerate(campaigns):
        manifest = read(root / "manifest.json")
        this_cfg, signature = physical_signature(root)
        if target is None:
            target, cfg = signature, this_cfg
            assert signature["shape_sha256"] == original["shape_sha256"]
            shutil.copy2(root / "provenance/shape.json", archive / "shape.json")
            shutil.copy2(root / "provenance/config.json", archive / "physical-config.json")
        assert signature == target, "Training campaigns must have exactly the same physical target and regions"
        shutil.copy2(root / "manifest.json", archive / f"campaign-{index}-manifest.json")
        expected_n = sum(j["samples"] for j in manifest["jobs"])
        used = 0
        for job in manifest["jobs"]:
            assert job["seed"] not in seeds, "Require distinct training population seeds"
            seeds.add(job["seed"])
            directory = Path(job["directory"])
            summary = read(directory / "summary.json")
            assert summary["complete"] and summary["samples"] == job["samples"]
            path = directory / "samples.jsonl"
            draws = 0
            for line in path.open():
                row = json.loads(line)
                draws += 1
                if row["hard_valid"] and 1 < row["q"] < 5:
                    assert row["log_importance_weight"] is not None
                    selected.append(dict(campaign=index, population=job["id"], seed=job["seed"],
                                         draw=row["draw"], pose=row["pose"], q=row["q"],
                                         original_region=row["region"], log_importance_weight=row["log_importance_weight"],
                                         log_proposal_density=row["log_proposal_density"], clouds=row["clouds"],
                                         campaign_draws=expected_n))
                    used += 1
            assert draws == job["samples"]
            total_draws += draws
            sources.append(dict(campaign=index, population=job["id"], path=str(path), sha256=sha(path),
                                unconditional_draws=draws, seed=job["seed"]))
        if used == 0:
            raise ValueError(f"No training-region observations in {root}")
    if len(selected) < 8:
        raise ValueError("Insufficient training data for a six-dimensional covariance")
    assert len(cfg["fixed_poses"]) == 1
    poses = relative_poses([r["pose"] for r in selected], cfg["fixed_poses"][0])
    position, _, rotation = arrays(poses)
    anchor = original["anchors"][k]
    delta = Rotation.from_matrix(rotation @ np.asarray(anchor["rotation"]).T).as_quat()
    if np.any(delta[:, 3] == 0):
        raise ValueError("Training region intersects the exact fixed Cayley chart seam")
    cayley = delta[:, :3] / delta[:, 3, None]
    x = np.column_stack((position - anchor["position"], original["angular_length"] * cayley))
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite fixed-chart training coordinate")
    # Each completed campaign has equal fit influence before its normalized
    # importance weights. Equal original budgets make this common factor
    # irrelevant, while unequal future budgets remain explicitly accounted for.
    log_weights = np.asarray([r["log_importance_weight"] - np.log(r["campaign_draws"]) for r in selected])
    model = {field: copy.deepcopy(original[field]) for field in [
        "schema", "angular_length", "coordinate_convention", "shape_sha256",
        "anchors", "means", "covariances", "weights"]}
    model["weights"] = [args.retained_weight * w for w in original["weights"]]
    sigma0 = np.asarray(original["covariances"][k])
    floor = args.covariance_floor_fraction * sigma0
    reports = []
    for power in args.weight_powers:
        normalized = np.exp(power * log_weights - logsumexp(power * log_weights))
        mean = normalized @ x
        centered = x - mean
        raw = centered.T @ (normalized[:, None] * centered)
        raw = .5 * (raw + raw.T)
        covariance = raw + floor
        covariance = .5 * (covariance + covariance.T)
        lower = np.linalg.cholesky(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert np.isfinite(lower).all() and eigenvalues[0] > 0
        model["anchors"].append(copy.deepcopy(anchor))
        model["means"].append(mean.tolist())
        model["covariances"].append(covariance.tolist())
        model["weights"].append((1 - args.retained_weight) / len(args.weight_powers))
        reports.append(dict(weight_power=power, training_points=len(x),
            normalized_training_weight_ess=float(1 / np.sum(normalized**2)), largest_training_weight=float(normalized.max()),
            mean=mean.tolist(), raw_weighted_covariance=raw.tolist(), additive_covariance_floor=floor.tolist(),
            final_covariance=covariance.tolist(), final_eigenvalues=eigenvalues.tolist(), condition_number=float(eigenvalues[-1] / eigenvalues[0]),
            training_use="Proposal fit only; neither tempered fit weight nor component weight is a physical basin probability"))
    assert abs(sum(model["weights"]) - 1) < 1e-12
    for field in ["anchors", "means", "covariances"]:
        assert model[field][:count] == original[field]
    provenance = dict(kind="explicit native-informed importance-guided proposal control; frozen before fresh normalization",
        source_model=str(args.source_model.resolve()), source_model_sha256=sha(args.source_model), retained_components=count,
        retained_total_weight=args.retained_weight, original_chart_component=k, original_chart_anchor=anchor,
        original_region="hard-valid and 1<original registration q<5; no new basin definition",
        training_campaigns=[str(p) for p in campaigns], training_unconditional_draws=total_draws,
        training_selected_poses=len(x), weight_powers=args.weight_powers,
        additive_covariance_floor_fraction=args.covariance_floor_fraction,
        target=target, maximum_chart_angle_deg=float(np.degrees(2 * np.arccos(np.min(np.abs(delta[:, 3]))))),
        maximum_cayley_radius=float(np.linalg.norm(cayley, axis=1).max()),
        weighting="Within each campaign log importance weight minus log unconditional draw count; pool contributions, apply stated power, normalize for fitting only",
        physical_inference="Training observations are not fresh evidence. Correct future importance weights divide by the complete113-component (or configured size) pose density with positive uniform defense.")
    model["proposal_provenance"] = provenance
    write(out / "model.json", model)
    with (archive / "training-poses.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    new_model = {field: copy.deepcopy(model[field]) for field in ["schema", "angular_length", "coordinate_convention", "shape_sha256"]}
    for field in ["anchors", "means", "covariances"]:
        new_model[field] = copy.deepcopy(model[field][count:])
    new_model["weights"] = [1 / len(reports)] * len(reports)
    geometry = candidate_audit(new_model, cfg, read(archive / "shape.json"), args.candidate_probes_per_component, 34689210)
    write(out / "report.json", dict(provenance=provenance, fit_reports=reports, sources=sources,
        independent_new_component_geometry=geometry, model_sha256=sha(out / "model.json"), output_components=len(model["weights"]),
        archived_files={p.name: sha(p) for p in sorted(archive.iterdir())},
        requirement="Use only independent new production draws to assess this guide; fixed physical target and exact full-density weighting remain unchanged"))
    print(json.dumps(dict(out=str(out), components=len(model["weights"]), selected_training_poses=len(x),
        fits=[{k:r[k] for k in ["weight_power", "normalized_training_weight_ess", "largest_training_weight", "condition_number"]} for r in reports],
        candidate_geometry=geometry, model_sha256=sha(out / "model.json")), indent=2))


if __name__ == "__main__":
    main()
