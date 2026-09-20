#!/usr/bin/env python3
"""Freeze one compact SMC-discovered far-contact cloud as importance guides.

Endpoint counts define the fit only. SMC normalizers and ancestry do not set
mixture weights; only fresh independent normalizer draws assess physical mass.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_smc_normalizer_atlas import (
    ROOT, Density, arrays, candidate_audit, convert_model, fit_population,
    model_from_components, quantiles, read, relative_poses, sha, write,
)


def registration(poses, metadata):
    t, q, r = arrays(poses)
    ft, fq, fr = arrays(metadata["native_poses"])
    members = np.asarray([m["position"] for m in metadata["rigid_members"]])
    moved = np.einsum("nij,mj->nmi", r, members) + t[:, None, :]
    scores = []
    for nt, nq, nr in zip(ft, fq, fr):
        target = members @ nr.T + nt
        displacement = np.linalg.norm(moved - target[None], axis=2).max(axis=1)
        angle = 2 * np.arccos(np.clip(np.abs(q @ nq), 0., 1.))
        scores.append(np.maximum(displacement / metadata["member_error_scale"],
                                 angle / np.deg2rad(metadata["angle_error_scale_deg"])))
    return np.min(scores, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r1")
    parser.add_argument("--parent", type=Path, default=ROOT / "runs/smc-normalizer-importance-guided/site0/model.json")
    parser.add_argument("--physical-config", type=Path, default=ROOT / "runs/basin-normalizer-importance-guided-16384-l64/provenance/config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/smc-normalizer-deep-far/site0")
    parser.add_argument("--std-scales", nargs="+", type=float, default=[1., 2., 4.])
    parser.add_argument("--coordinate-std-floor", type=float, default=.02)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1e-8)
    parser.add_argument("--maximum-chart-angle-deg", type=float, default=30.)
    parser.add_argument("--candidate-probes", type=int, default=256)
    args = parser.parse_args()
    if any(not 0 < s < float("inf") for s in args.std_scales):
        parser.error("Standard deviation scales must be finite and positive")
    if len(set(args.std_scales)) != len(args.std_scales):
        parser.error("Duplicate scales would overwrite a model")
    if args.coordinate_std_floor <= 0 or args.relative_eigenvalue_floor <= 0 or args.candidate_probes < 1:
        parser.error("Floors and candidate count must be positive")
    if not 0 < args.maximum_chart_angle_deg < 180:
        parser.error("Chart-extent guard must be in (0,180) degrees")
    source = args.source.resolve()
    summary, smc_cfg = read(source / "summary.json"), read(source / "config.json")
    cfg, original = read(args.physical_config), read(args.parent)
    environment = read(smc_cfg["environment"])
    shape_path = Path(smc_cfg["shape"])
    assert summary["complete"] and summary["completed_stage"] == len(smc_cfg["schedule"]) - 1
    assert cfg["depletant_radius"] == smc_cfg["depletant_radius"]
    assert cfg["reservoir_density"] == smc_cfg["reservoir_density"]
    assert len(cfg["fixed_poses"]) == 1 and cfg["fixed_poses"] == environment["fixed_poses"]
    assert cfg["capture_center"] == environment["capture_center"] and cfg["capture_radius"] == environment["capture_radius"]
    assert sha(shape_path) == original["shape_sha256"] == sha(Path(cfg["shape"]))
    assert original["coordinate_convention"] == "anchor-body-relative"
    poses = [row["pose"] for row in summary["final_particles"]]
    component, fit = fit_population(poses, smc_cfg["metadata"]["replica"], original["angular_length"],
                                    args.coordinate_std_floor, args.relative_eigenvalue_floor, 1.)
    if fit["maximum_chart_angle_deg"] > args.maximum_chart_angle_deg:
        raise ValueError("Cloud exceeds the single-chart compactness guard; use explicit localized groups before freezing")
    lab = model_from_components([component], original["angular_length"], original["shape_sha256"])
    body = convert_model(lab, cfg["fixed_poses"][0])
    relative = relative_poses(poses, cfg["fixed_poses"][0])
    lab_logq = Density(lab).evaluate(poses)[0]
    body_logq = Density(body).evaluate(relative)[0]
    frame_error = float(np.max(np.abs(lab_logq - body_logq)))
    assert frame_error < 2e-8
    physical_q = registration(poses, cfg["metadata"])
    stored_q = np.asarray([row["q"] for row in summary["final_particles"]])
    stored_factor = smc_cfg["metadata"]["region"]["stored_q_to_physical_q"]
    assert np.max(np.abs(physical_q - stored_factor * stored_q)) < 1e-9
    families = Counter(row["family_id"] for row in summary["final_particles"])
    parent_logg = Density(original).evaluate(relative)[0]
    eps = cfg["uniform_probability"]
    uniform_log = np.log(eps) - 3 * np.log(2 * cfg["capture_radius"])
    parent_logq = np.logaddexp(uniform_log, np.log1p(-eps) + parent_logg)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh output directory")
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir()
    archive = out / "provenance"
    sources = {"source-summary.json": source / "summary.json", "source-config.json": source / "config.json",
               "source-environment.json": Path(smc_cfg["environment"]), "shape.json": shape_path,
               "parent-model.json": args.parent, "physical-config.json": args.physical_config,
               "prepare_deep_far_normalizer_atlas.py": Path(__file__),
               "prepare_smc_normalizer_atlas.py": Path(__file__).with_name("prepare_smc_normalizer_atlas.py")}
    for name, path in sources.items():
        shutil.copy2(path, archive / name)
    provenance = dict(source_summary=str(source / "summary.json"), source_summary_sha256=sha(source / "summary.json"),
        smc_seed=smc_cfg["seed"], endpoint_count=len(poses), unique_pose_count=len({str(p) for p in poses}),
        family_counts=dict(families), family_ess=len(poses)**2 / sum(n*n for n in families.values()),
        source_smc_logZ_diagnostic_only=summary["logZ"],
        physical_q=quantiles(physical_q), stored_internal_q=quantiles(stored_q),
        stored_internal_q_to_physical_q=stored_factor,
        physical_metric={k: cfg["metadata"][k] for k in ["member_error_scale", "angle_error_scale_deg"]},
        physical_q_recomputation_max_error=float(np.max(np.abs(physical_q - stored_factor * stored_q))),
        translation_span_A=np.ptp(arrays(poses)[0], axis=0).tolist(),
        parent_model_sha256=sha(args.parent), parent_components=len(original["weights"]),
        fixed_neighbor=cfg["fixed_poses"][0], frame_conversion_log_density_max_error=frame_error,
        construction="One equal-endpoint-weight full covariance in compact maximin Cayley chart; no SMC logZ or ancestry weighting; parent weight0.5 and guide weight0.5",
        inference="Correlated single-family discovery cloud, not an equilibrium sample. Only fresh independent normalizer draws evaluate physical mass; parent retains native-informed components.",
        parent_full_proposal_log_density_on_endpoints=quantiles(parent_logq),
        parent_gaussian_log_density_on_endpoints=quantiles(parent_logg),
        full_proposal_uniform_probability=eps,
        uniform_branch_log_density=uniform_log)
    reports = []
    for scale in args.std_scales:
        guide = copy.deepcopy(body)
        guide["covariances"] = (np.asarray(body["covariances"]) * scale**2).tolist()
        model = {k: copy.deepcopy(original[k]) for k in ["schema", "angular_length", "coordinate_convention", "shape_sha256", "weights", "anchors", "means", "covariances"]}
        model["weights"] = [.5 * w for w in model["weights"]] + [.5]
        for field in ["anchors", "means", "covariances"]:
            model[field].extend(copy.deepcopy(guide[field]))
            assert model[field][:-1] == original[field]
        assert abs(sum(model["weights"]) - 1) < 1e-12
        model["proposal_provenance"] = dict(provenance, new_guide_standard_deviation_scale=scale)
        path = out / "models" / f"std{scale:g}.json"
        write(path, model)
        new_logg = Density(model).evaluate(relative)[0]
        new_logq = np.logaddexp(uniform_log, np.log1p(-eps) + new_logg)
        geometry = candidate_audit(guide, cfg, read(shape_path), args.candidate_probes, 73891210)
        reports.append(dict(standard_deviation_scale=scale, model=str(path), model_sha256=sha(path),
            guide_gaussian_log_density_on_endpoints=quantiles(Density(guide).evaluate(relative)[0]),
            complete_new_proposal_log_density_on_endpoints=quantiles(new_logq),
            complete_new_vs_parent_log_density_gain_on_endpoints=quantiles(new_logq - parent_logq),
            candidate_geometry=geometry))
    write(out / "report.json", dict(provenance=provenance, fit=fit, scale_reports=reports,
        raw_lab_component=component, body_component=body,
        archived_files={name: dict(source=str(path.resolve()), sha256=sha(archive/name)) for name,path in sources.items()}))
    print(__import__("json").dumps(dict(out=str(out), provenance=provenance, fit=fit, scale_reports=reports), indent=2))


if __name__ == "__main__":
    main()
