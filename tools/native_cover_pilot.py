#!/usr/bin/env python3
"""Geometry-only independent uniform cover of the complete native pose region.

No Gaussian atlas, Poisson cloud, or physical Boltzmann weight is used. All
unconditional draws are retained. Hard overlap is checked with independent
atom-radius-group scipy KD-trees; output is an efficiency screen, not a
depletion normalizer.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import kstest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(pose):
    q = np.asarray(pose["orientation"], dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def theta_minus_sin(theta):
    theta = np.asarray(theta)
    squared = theta * theta
    series = theta * squared * (1 / 6 - squared / 120 + squared**2 / 5040 - squared**3 / 362880)
    return np.where(np.abs(theta) < 0.01, series, theta - np.sin(theta))


def inverse_cap_cdf(u, cap):
    target = u * theta_minus_sin(cap)
    low = np.zeros_like(u)
    high = np.full_like(u, cap)
    for _ in range(58):
        mid = (low + high) * 0.5
        left = theta_minus_sin(mid) < target
        low = np.where(left, mid, low)
        high = np.where(left, high, mid)
    return (low + high) * 0.5


def unit_vectors(rng, n):
    vectors = rng.normal(size=(n, 3))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=202609201700)
    args = parser.parse_args()
    assert args.draws > 0
    args.config = args.config.resolve()
    assert not args.out.exists(), "Refuse to overwrite a pilot"
    args.out.mkdir(parents=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    config = json.loads(args.config.read_text())
    metric = config["metadata"]
    refs = metric["native_poses"]
    assert len(refs) > 0
    members = np.asarray([m["position"] for m in metric["rigid_members"]])
    centroid = members.mean(axis=0)
    centered = members - centroid
    covariance = centered.T @ centered / len(members)
    eigenvalues = np.linalg.eigvalsh(covariance)
    lower = max(0.0, float(eigenvalues[0] + eigenvalues[1]))
    # This is ordinary FP64, not an interval-arithmetic certificate. Enlarge
    # the exact-arithmetic cover by a negligible outward numerical margin.
    guarded_lower = max(0.0, lower - 1e-12 * max(1.0, float(np.trace(covariance))))
    a = float(metric["member_error_scale"])
    nominal_cap = math.radians(metric["angle_error_scale_deg"])
    assert 0 < nominal_cap <= math.pi
    geometric_cap = math.pi if guarded_lower == 0 else 2 * math.asin(min(1.0, a / (2 * math.sqrt(guarded_lower))))
    cap = min(nominal_cap, geometric_cap)
    ball_volume = 4 * math.pi * a**3 / 3
    cap_probability = float(theta_minus_sin(cap)) / math.pi
    cover_volume = ball_volume * cap_probability
    shape_path = Path(config["shape"])
    if not shape_path.is_absolute():
        shape_path = args.config.parent / shape_path
    shape = json.loads(shape_path.read_text())
    atoms = np.asarray([atom["center"] for atom in shape["atoms"]])
    radii = np.asarray([atom["radius"] for atom in shape["atoms"]])
    atom_bound = float(np.max(np.linalg.norm(atoms, axis=1) + radii))

    old_comparison = None
    source = metric.get("source_smc", {})
    if source.get("config"):
        old_path = Path(source["config"])
        old = json.loads(old_path.read_text())
        old_env_path = Path(old["environment"])
        old_env = json.loads(old_env_path.read_text())
        agreements = {
            "member_error_scale": old["member_error_scale"] == a,
            "angle_error_scale_deg": old["angle_error_scale_deg"] == metric["angle_error_scale_deg"],
            "native_poses": old_env["native_poses"] == refs,
            "rigid_members": old_env["rigid_members"] == metric["rigid_members"],
            "fixed_poses": old_env["fixed_poses"] == config["fixed_poses"],
            "capture_center": old_env["capture_center"] == config["capture_center"],
            "capture_radius": old_env["capture_radius"] == config["capture_radius"],
            "shape_sha256": digest(old["shape"]) == digest(shape_path),
            "bath_radius": old["depletant_radius"] == config["depletant_radius"],
            "bath_activity": old["reservoir_density"] == config["reservoir_density"],
        }
        assert all(agreements.values()), agreements
        old_comparison = {"config": str(old_path), "config_sha256": digest(old_path),
                          "environment": str(old_env_path), "environment_sha256": digest(old_env_path),
                          "agreements": agreements}

    # Verify that nonzero periodic images cannot overlap any captured pose.
    box = metric.get("source_periodic_box")
    image_gap_lower = None
    if box is not None:
        box = np.asarray(box)
        capture_center = np.asarray(config["capture_center"])
        gaps = []
        for fixed in config["fixed_poses"]:
            for image in itertools.product((-1, 0, 1), repeat=3):
                if image != (0, 0, 0):
                    delta = np.asarray(fixed["position"]) + np.asarray(image) * box - capture_center
                    gaps.append(np.linalg.norm(delta) - config["capture_radius"] - 2 * atom_bound)
        image_gap_lower = float(min(gaps))
        assert image_gap_lower > 0, "Pilot requires an image-safe finite capture region"
        assert np.min(box) > 2 * (config["capture_radius"] + atom_bound)

    fixed_atoms = np.concatenate([
        atoms @ matrix(fixed).T + fixed["position"] for fixed in config["fixed_poses"]
    ])
    fixed_radii = np.tile(radii, len(config["fixed_poses"]))
    groups = [(float(radius), cKDTree(fixed_atoms[fixed_radii == radius]))
              for radius in np.unique(fixed_radii)]
    # Largest atom population first commonly finds an overlap early.
    groups.sort(key=lambda item: item[1].n, reverse=True)

    rng = np.random.default_rng(args.seed)
    components = rng.integers(len(refs), size=args.draws)
    angular_uniforms = rng.random(args.draws)
    angles = inverse_cap_cdf(angular_uniforms, cap)
    axes = unit_vectors(rng, args.draws)
    relative = Rotation.from_rotvec(axes * angles[:, None]).as_matrix()
    ball_uniforms = rng.random(args.draws)
    w = unit_vectors(rng, args.draws) * (a * np.cbrt(ball_uniforms))[:, None]
    rotations = np.empty((args.draws, 3, 3))
    positions = np.empty((args.draws, 3))
    for j, ref in enumerate(refs):
        selected = components == j
        r0 = matrix(ref)
        rotations[selected] = r0 @ relative[selected]
        positions[selected] = (np.asarray(ref["position"])
                               - (rotations[selected] - r0) @ centroid + w[selected])
    q = np.full(args.draws, math.inf)
    multiplicity = np.zeros(args.draws, dtype=np.int32)
    mean_square_min = np.full(args.draws, math.inf)
    transformed_members = np.einsum("nij,kj->nki", rotations, members) + positions[:, None, :]
    quaternions = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    for ref in refs:
        r0 = matrix(ref)
        ref_members = members @ r0.T + ref["position"]
        errors = np.linalg.norm(transformed_members - ref_members, axis=2)
        qref = np.asarray(ref["orientation"])
        dots = np.abs(quaternions @ (qref / np.linalg.norm(qref)))
        theta = 2 * np.arccos(np.clip(dots, 0, 1))
        q = np.minimum(q, np.maximum(np.max(errors, axis=1) / a, theta / nominal_cap))
        mean_square_min = np.minimum(mean_square_min, np.mean(errors**2, axis=1))
        delta = positions - ref["position"] + (rotations - r0) @ centroid
        multiplicity += ((np.linalg.norm(delta, axis=1) <= a * (1 + 1e-12))
                         & (theta <= cap + 1e-12))
    assert np.all(multiplicity >= 1)
    capture = np.linalg.norm(positions - config["capture_center"], axis=1) <= config["capture_radius"]
    native = q <= 1
    hard = np.zeros(args.draws, dtype=bool)
    hard_checked = native & capture
    hard_gaps = np.full(args.draws, np.nan)
    hard_started = time.monotonic()
    checked_indices = np.flatnonzero(hard_checked)
    for step, i in enumerate(checked_indices):
        mobile = atoms @ rotations[i].T + positions[i]
        minimum = math.inf
        for radius, tree in groups:
            distances = tree.query(mobile, k=1, workers=1)[0]
            minimum = min(minimum, float(np.min(distances - radii - radius)))
            if minimum < 0:
                break
        hard[i] = minimum >= 0
        hard_gaps[i] = minimum
        if (step + 1) % 1024 == 0:
            print(json.dumps({"hard_checks": step + 1, "of": len(checked_indices),
                              "hard_native": int(hard.sum()),
                              "wall_seconds": time.monotonic() - started}), flush=True)
    hard_seconds = time.monotonic() - hard_started

    # This importance correction also supports overlapping equal-volume covers.
    geometric_weights = len(refs) * cover_volume / multiplicity * native * capture * hard
    mean = float(geometric_weights.mean())
    stderr = float(geometric_weights.std(ddof=1) / math.sqrt(args.draws))
    cap_cdf = theta_minus_sin(angles) / theta_minus_sin(cap)
    # Save every draw, including zero-weight samples, for independent audit.
    np.savez_compressed(args.out / "draws.npz", position=positions, orientation=quaternions,
                        component=components, q=q, cover_multiplicity=multiplicity,
                        capture=capture, native=native, hard_checked=hard_checked,
                        hard_valid=hard, hard_gap_A=hard_gaps,
                        uniform_angular=angular_uniforms, uniform_ball=ball_uniforms)
    summary = {
        "complete": True, "scope": "Geometry-only complete-native-cover efficiency pilot; no physical depletion weights",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "unconditional_draws": args.draws, "native_references": len(refs),
        "native_reference_poses": refs, "member_centroid_A": centroid.tolist(),
        "member_covariance_A2": covariance.tolist(), "member_covariance_eigenvalues_A2": eigenvalues.tolist(),
        "L_A2": lower, "guarded_L_A2": guarded_lower, "member_tolerance_A": a,
        "nominal_angle_cap_degrees": math.degrees(nominal_cap), "cover_angle_cap_degrees": math.degrees(cap),
        "translation_ball_volume_A3": ball_volume, "normalized_Haar_cap_probability": cap_probability,
        "single_cover_volume_A3": cover_volume,
        "cover_proposal_density": "sum_j 1[p in cover_j] / (number_of_refs * single_cover_volume)",
        "physical_estimator_if_added": "mean[hard * capture * (q<=1) * unbiased_overlap_Boltzmann_weight / full_cover_proposal_density]; never discard zeros",
        "counts": {"mean_square_necessary_pass": int((mean_square_min <= a*a).sum()),
                   "q_native": int(native.sum()), "capture": int(capture.sum()),
                   "q_native_capture": int(hard_checked.sum()), "q_native_capture_hard": int(hard.sum()),
                   "native_core_hard": int((hard & (q <= .8)).sum()),
                   "native_shell_hard": int((hard & (q > .8)).sum())},
        "hard_native_fraction_all_draws": float(hard.mean()),
        "hard_fraction_given_native_capture": float(hard.sum() / max(1, hard_checked.sum())),
        "native_zero_activity_volume_A3": mean, "native_zero_activity_volume_SE_A3": stderr,
        "geometric_relative_standard_error": stderr / mean if mean > 0 else None,
        "angular_inverse_CDF_max_residual": float(np.max(np.abs(cap_cdf - angular_uniforms))),
        "angular_CDF_uniform_KS_pvalue": float(kstest(cap_cdf, "uniform").pvalue),
        "ball_r3_uniform_KS_pvalue": float(kstest((np.linalg.norm(w, axis=1) / a)**3, "uniform").pvalue),
        "atom_count": len(atoms), "distinct_atom_radii_A": sorted(float(r) for r in np.unique(radii)),
        "hard_predicate": "Exact nearest-center query separately for each equal-radius fixed-atom group; strict distance < radius_sum rejects. All group nearest distances checked for accepted poses.",
        "hard_gap_note": "Valid-pose gap is global atom-pair minimum; invalid-pose negative gap may stop at first conflicting radius group.",
        "body_bound_A": atom_bound, "periodic_nonzero_image_core_gap_lower_A": image_gap_lower,
        "bound_numerics": "FP64 eigensystem with outward cap guard; mathematical cover proved in exact arithmetic, not interval-certified floating point.",
        "old_metadata_comparison": old_comparison,
        "hard_check_wall_seconds": hard_seconds, "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "config": str(args.config), "config_sha256": digest(args.config),
        "shape": str(shape_path), "shape_sha256": digest(shape_path),
        "script_sha256": digest(__file__), "draws_sha256": digest(args.out / "draws.npz"),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "python_version": platform.python_version(),
        "environment_thread_caps": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
