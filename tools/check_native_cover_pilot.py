#!/usr/bin/env python3
"""Independent algebra/atomic checks of an existing geometry-only cover pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert sha(args.run / "draws.npz") == summary["draws_sha256"]
    config = json.loads(Path(summary["config"]).read_text())
    draws = np.load(args.run / "draws.npz")
    metric = config["metadata"]
    members = np.array([m["position"] for m in metric["rigid_members"]])
    centroid = members.mean(0)
    centered = members-centroid
    S = centered.T @ centered / len(members)
    matrices = np.array([rotation(q) for q in draws["orientation"]])
    moved = np.einsum("nij,kj->nki", matrices, members)+draws["position"][:, None, :]
    independent_q = np.full(len(matrices), np.inf)
    max_identity_error = 0.0
    for ref in metric["native_poses"]:
        r0 = rotation(ref["orientation"])
        member_errors = moved - (members @ r0.T+ref["position"])
        A = matrices-r0
        w = draws["position"]-ref["position"]+A @ centroid
        left = np.mean(np.sum(member_errors**2, axis=2), axis=1)
        right = np.sum(w*w, axis=1)+np.einsum("nij,jk,nik->n", A, S, A)
        max_identity_error = max(max_identity_error, float(np.max(np.abs(left-right))))
        dots = np.abs(draws["orientation"] @ (np.array(ref["orientation"])/np.linalg.norm(ref["orientation"])))
        angle = 2*np.arccos(np.clip(dots, 0, 1))
        independent_q = np.minimum(independent_q, np.maximum(
            np.max(np.linalg.norm(member_errors, axis=2), axis=1)/metric["member_error_scale"],
            angle/np.deg2rad(metric["angle_error_scale_deg"])))
    q_error = float(np.max(np.abs(independent_q-draws["q"])))
    assert q_error < 1e-12 and max_identity_error < 1e-10
    assert np.array_equal(independent_q <= 1, draws["native"])

    shape = json.loads(Path(summary["shape"]).read_text())
    atoms = np.array([atom["center"] for atom in shape["atoms"]])
    radii = np.array([atom["radius"] for atom in shape["atoms"]])
    fixed = np.concatenate([atoms @ rotation(p["orientation"]).T+p["position"] for p in config["fixed_poses"]])
    fixed_radii = np.tile(radii, len(config["fixed_poses"]))
    chosen = list(np.flatnonzero(draws["hard_valid"])[:3])+list(np.flatnonzero(draws["hard_checked"] & ~draws["hard_valid"])[:3])
    gap_checks = []
    for i in chosen:
        mobile = atoms @ matrices[i].T+draws["position"][i]
        gap = np.inf
        for begin in range(0, len(atoms), 128):
            block = mobile[begin:begin+128]
            distances = np.sqrt(np.sum((block[:, None, :]-fixed[None, :, :])**2, axis=2))
            gap = min(gap, np.min(distances-radii[begin:begin+128, None]-fixed_radii[None, :]))
        assert bool(gap >= 0) == bool(draws["hard_valid"][i])
        if gap >= 0:
            assert abs(gap-draws["hard_gap_A"][i]) < 1e-12
        gap_checks.append({"draw": int(i), "direct_all_atom_pair_gap_A": float(gap),
                           "kdtree_gap_A": float(draws["hard_gap_A"][i]),
                           "valid": bool(gap >= 0), "pairs_checked": len(atoms)*len(fixed)})
    result = {"passed": True, "summary_sha256": sha(summary_path),
              "script_sha256": sha(__file__), "mean_square_identity_max_error_A2": max_identity_error,
              "independent_quaternion_formula_q_max_error": q_error,
              "all_draw_classifications_agree": True,
              "brute_force_atom_pair_checks": gap_checks,
              "wall_seconds": time.monotonic()-started}
    (args.run / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
