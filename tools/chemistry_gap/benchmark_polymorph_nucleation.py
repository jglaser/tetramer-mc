#!/usr/bin/env python3
"""
Parallel nucleation benchmark launcher and trajectory analyzer for HEWL polymorphs.

Usage examples:
  # 1. Dry-run all 10 validated polymorph parameterizations (shows assigned nuc-<N> IDs):
  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py launch --preset all10 --dry-run

  # 2. Build morph-mc and launch all 10 polymorph simulations in parallel in the background:
  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py launch --preset all10

  # 3. Launch only the core 4 benchmark set (1lyz-monomer, 1lyz-nacl, 3wl2-nano3, 2d4i-nano3):
  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py launch --preset core4

  # 4. Launch specific targets:
  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py launch --targets 1lyz-nacl,3wl2-nano3,2d4i-nano3

  # 5. Summarize / monitor running or completed nuc-<N> simulations:
  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py status --runs 102..105
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = Path("/vast/xvg/tetramer-mc-runs/morph-growth-20261012")


BENCHMARK_TARGETS: List[Dict[str, object]] = [
    {
        "key": "1lyz-monomer",
        "label": "1LYZ + monomer.json",
        "shape": "examples/monomer-shape.json",
        "seed_config": "examples/spherical-cluster-monomer.json",
        "chemistry_model": "examples/chemistry-model-monomer.json",
        "salt_type": "nacl",
        "salt_conc_m": 0.85,
    },
    {
        "key": "1lyz-nacl",
        "label": "1LYZ + hofmeister-nacl",
        "shape": "examples/monomer-shape.json",
        "seed_config": "examples/spherical-cluster-monomer.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-nacl.json",
        "salt_type": "nacl",
        "salt_conc_m": 0.85,
    },
    {
        "key": "1iee-nacl",
        "label": "tetragonal-1IEE (NaCl)",
        "shape": "examples/polymorphs/monomer-shape-tetragonal-1IEE.json",
        "seed_config": "examples/polymorphs/seeded-tetragonal-1IEE.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-nacl.json",
        "salt_type": "nacl",
        "salt_conc_m": 0.85,
    },
    {
        "key": "2d4i-nano3",
        "label": "monoclinic-2D4I (NaNO3)",
        "shape": "examples/polymorphs/monomer-shape-monoclinic-2D4I.json",
        "seed_config": "examples/polymorphs/seeded-monoclinic-2D4I.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-nano3.json",
        "salt_type": "nano3",
        "salt_conc_m": 0.30,
    },
    {
        "key": "3wl2-nano3",
        "label": "monoclinic-3WL2 (NaNO3)",
        "shape": "examples/polymorphs/monomer-shape-monoclinic-3WL2.json",
        "seed_config": "examples/polymorphs/seeded-monoclinic-3WL2.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-nano3.json",
        "salt_type": "nano3",
        "salt_conc_m": 0.40,
    },
    {
        "key": "2z12-nacl",
        "label": "monoclinic-2Z12 (NaCl)",
        "shape": "examples/polymorphs/monomer-shape-monoclinic-2Z12.json",
        "seed_config": "examples/polymorphs/seeded-monoclinic-2Z12.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-monoclinic-2Z12.json",
        "salt_type": "nacl",
        "salt_conc_m": 1.50,
    },
    {
        "key": "1wtn-nacl",
        "label": "orthorhombic-1WTN (NaCl)",
        "shape": "examples/polymorphs/monomer-shape-orthorhombic-1WTN.json",
        "seed_config": "examples/polymorphs/seeded-orthorhombic-1WTN.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-orthorhombic-1WTN.json",
        "salt_type": "nacl",
        "salt_conc_m": 0.50,
    },
    {
        "key": "2vb1-nano3",
        "label": "triclinic-2VB1 (NaNO3)",
        "shape": "examples/polymorphs/monomer-shape-triclinic-2VB1.json",
        "seed_config": "examples/polymorphs/seeded-triclinic-2VB1.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-triclinic-2VB1.json",
        "salt_type": "nano3",
        "salt_conc_m": 0.30,
    },
    {
        "key": "2fbb-nano3",
        "label": "hexagonal-2FBB (NaNO3)",
        "shape": "examples/polymorphs/monomer-shape-hexagonal-2FBB.json",
        "seed_config": "examples/polymorphs/seeded-hexagonal-2FBB.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-hexagonal-2FBB.json",
        "salt_type": "nano3",
        "salt_conc_m": 0.40,
    },
    {
        "key": "9hru-so4",
        "label": "trigonal-9HRU (SO4)",
        "shape": "examples/polymorphs/monomer-shape-trigonal-9HRU.json",
        "seed_config": "examples/polymorphs/seeded-trigonal-9HRU.json",
        "chemistry_model": "examples/polymorphs/chemistry-model-hofmeister-so4.json",
        "salt_type": "so4",
        "salt_conc_m": 0.20,
    },
]

PRESETS: Dict[str, List[str]] = {
    "core4": ["1lyz-nacl", "3wl2-nano3", "2d4i-nano3", "1lyz-monomer"],
    "all10": [str(t["key"]) for t in BENCHMARK_TARGETS],
}


def _register_30pdb_targets() -> None:
    lib_path = REPO_ROOT / "examples" / "polymorphs" / "polymorph-contact-library.json"
    if not lib_path.exists():
        return
    lib = json.loads(lib_path.read_text())
    existing_by_tag: Dict[str, str] = {
        "tetragonal-1IEE": "1iee-nacl",
        "monoclinic-2D4I": "2d4i-nano3",
        "monoclinic-3WL2": "3wl2-nano3",
        "monoclinic-2Z12": "2z12-nacl",
        "orthorhombic-1WTN": "1wtn-nacl",
        "triclinic-2VB1": "2vb1-nano3",
        "hexagonal-2FBB": "2fbb-nano3",
        "trigonal-9HRU": "9hru-so4",
    }
    hewl_keys: List[str] = []
    nonhewl_keys: List[str] = []
    train_keys: List[str] = []
    heldout_keys: List[str] = []
    all30_keys: List[str] = []
    for p in lib.get("polymorphs", []):
        tag = str(p["polymorph"])
        pdb_id = str(p["pdb_id"]).lower()
        split = str(p.get("split", "train"))
        is_hewl = bool(p.get("is_hewl", False))
        prot = str(p.get("protein_family", "HEWL"))
        sc = p.get("salt_concs", [0.5, 0.0, 0.0, 0.5])
        if sc[1] > 0.0:
            s_type, s_conc = "nano3", float(sc[1])
        elif sc[2] > 0.0:
            s_type, s_conc = "so4", float(sc[2])
        else:
            s_type, s_conc = "nacl", float(max(sc[0], 0.05))
        if tag in existing_by_tag:
            key = existing_by_tag[tag]
        else:
            key = f"{pdb_id}-{s_type}"
            BENCHMARK_TARGETS.append(
                {
                    "key": key,
                    "label": f"{tag} ({prot})",
                    "shape": f"examples/polymorphs/monomer-shape-{tag}.json",
                    "seed_config": f"examples/polymorphs/seeded-{tag}.json",
                    "chemistry_model": f"examples/polymorphs/chemistry-model-hofmeister-{tag}.json",
                    "salt_type": s_type,
                    "salt_conc_m": s_conc,
                }
            )
        all30_keys.append(key)
        if is_hewl:
            hewl_keys.append(key)
        else:
            nonhewl_keys.append(key)
        if split == "train":
            train_keys.append(key)
        else:
            heldout_keys.append(key)
    PRESETS["hewl14"] = hewl_keys
    PRESETS["nonhewl16"] = nonhewl_keys
    PRESETS["train18"] = train_keys
    PRESETS["heldout12"] = heldout_keys
    PRESETS["all30"] = all30_keys


_register_30pdb_targets()


def next_nuc_id(runs_root: Path) -> int:
    max_id = 0
    if runs_root.exists():
        for entry in runs_root.iterdir():
            m = re.fullmatch(r"nuc-(\d+)", entry.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def build_morph_mc_cmd(
    binary: Path,
    repo_root: Path,
    target: Dict[str, object],
    out_dir: Path,
    free_bodies: int,
    concentration_um: float,
    sweeps: int,
    sample_every: int,
    seed: int,
) -> List[str]:
    shape_path = repo_root / str(target["shape"])
    seed_cfg_path = repo_root / str(target["seed_config"])
    chem_path = repo_root / str(target["chemistry_model"])
    return [
        str(binary),
        "run",
        "--shape",
        str(shape_path),
        "--seed-config",
        str(seed_cfg_path),
        "--seed-count",
        "0",
        "--free-bodies",
        str(free_bodies),
        "--concentration-um",
        str(concentration_um),
        "--no-solvation",
        "--epsilon",
        "0.0",
        "--width",
        "0.0",
        "--chemistry-model",
        str(chem_path),
        "--salt-type",
        str(target["salt_type"]),
        "--salt-concentration-m",
        str(target["salt_conc_m"]),
        "--dock-probability",
        "0.25",
        "--dock-sigma-translation",
        "0.25",
        "--dock-sigma-rotation-degrees",
        "2.5",
        "--dock-uniform-weight",
        "0.5",
        "--dock-library-gap",
        "2.8",
        "--dock-library-beta",
        "0.15",
        "--surface-prior-samples",
        "64",
        "--surface-prior-beta",
        "0.3",
        "--memory-only",
        "--memory-lambda-N",
        "32.0",
        "--memory-lambda-floor",
        "1.0",
        "--memory-lambda-prior",
        "16.0",
        "--memory-death",
        "0.01",
        "--memory-sigma-translation",
        "0.25",
        "--memory-sigma-rotation-degrees",
        "2.5",
        "--memory-floor-sigma",
        "30.0",
        "--memory-refresh-per-sweep",
        "2",
        "--sweeps",
        str(sweeps),
        "--sample-every",
        str(sample_every),
        "--seed",
        str(seed),
        "--out",
        str(out_dir),
    ]


def quat_to_rot(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def monomer_bounding_radius(shape_path: Path) -> float:
    if not shape_path.exists():
        return 22.6
    sh = json.loads(shape_path.read_text())
    atoms = sh.get("atoms", [])
    if not atoms:
        return 22.6
    return float(
        max(
            math.sqrt(sum(float(x) * float(x) for x in a["center"])) + float(a.get("radius", 1.7))
            for a in atoms
        )
    )


def extract_native_modes(
    seed_cfg_path: Path, r_bound: float = 22.6
) -> List[Tuple[np.ndarray, np.ndarray]]:
    modes: List[Tuple[np.ndarray, np.ndarray]] = []
    if seed_cfg_path.name.startswith("seeded-"):
        tag = seed_cfg_path.stem.replace("seeded-", "")
        contacts_path = seed_cfg_path.parent / f"contacts-{tag}.json"
        if contacts_path.exists():
            cdata = json.loads(contacts_path.read_text())
            for c in cdata.get("contacts", []):
                t_rel = np.array(c["t_retracted"], dtype=float)
                r_rel = np.array(c["R_rel"], dtype=float)
                if not any(
                    np.linalg.norm(t_rel - tm) < 2.0 and np.linalg.norm(r_rel - rm, "fro") < 0.25
                    for tm, rm in modes
                ):
                    modes.append((t_rel, r_rel))
    if not seed_cfg_path.exists():
        return modes
    cfg = json.loads(seed_cfg_path.read_text())
    poses = cfg.get("initial_poses", [])
    max_seed_dist = max(2.12 * r_bound, 28.0)
    for i in range(len(poses)):
        ri = quat_to_rot(poses[i]["orientation"])
        ti = np.array(poses[i]["position"], dtype=float)
        for j in range(len(poses)):
            if i == j:
                continue
            rj = quat_to_rot(poses[j]["orientation"])
            tj = np.array(poses[j]["position"], dtype=float)
            dt = tj - ti
            dist = float(np.linalg.norm(dt))
            if dist > max_seed_dist:
                continue
            t_rel = ri.T @ dt
            r_rel = ri.T @ rj
            if not any(
                np.linalg.norm(t_rel - tm) < 2.0 and np.linalg.norm(r_rel - rm, "fro") < 0.25
                for tm, rm in modes
            ):
                modes.append((t_rel, r_rel))
    return modes


def read_last_jsonl(path: Path, max_bytes: int = 262144) -> Optional[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_bytes), os.SEEK_SET)
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    for raw in reversed(lines):
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
    return None


def supercell_percolation_dim(
    seed_cfg_path: Path, modes: List[Tuple[np.ndarray, np.ndarray]], active_modes: List[int]
) -> str:
    if not active_modes or not seed_cfg_path.exists():
        return "0D"
    data = json.loads(seed_cfg_path.read_text())
    sp = data.get("initial_poses", data.get("poses", []))
    n_sp = len(sp)
    if n_sp < 4:
        return "0D"
    s_pos = np.array([p["position"] for p in sp], dtype=float)
    s_rot = [quat_to_rot(p["orientation"]) for p in sp]
    mode_norms = [float(np.linalg.norm(modes[k][0])) for k in active_modes]
    max_mode_d = max(mode_norms) + 4.0 if mode_norms else 42.0
    min_mode_d = min(mode_norms) if mode_norms else 30.0
    adj_s: List[List[int]] = [[] for _ in range(n_sp)]
    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            dv = s_pos[j] - s_pos[i]
            if float(np.linalg.norm(dv)) > max_mode_d:
                continue
            t_ij = s_rot[i].T @ dv
            r_ij = s_rot[i].T @ s_rot[j]
            t_ji = s_rot[j].T @ (-dv)
            r_ji = s_rot[j].T @ s_rot[i]
            for k in active_modes:
                tm, rm = modes[k]
                for tr, rr in ((t_ij, r_ij), (t_ji, r_ji)):
                    dt = float(np.linalg.norm(tr - tm))
                    tr_v = (np.trace(rr.T @ rm) - 1.0) * 0.5
                    dth = math.degrees(math.acos(max(-1.0, min(1.0, float(tr_v)))))
                    if dt < 2.5 and dth < 15.0:
                        adj_s[i].append(j)
                        adj_s[j].append(i)
                        break
    seen = [False] * n_sp
    comps: List[List[int]] = []
    for i in range(n_sp):
        if not seen[i]:
            comp = []
            q = [i]
            seen[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in adj_s[u]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    top = comps[0]
    if len(top) <= 2:
        return "0D"
    lat_vecs = []
    for a_i in range(len(top)):
        u = top[a_i]
        for b_i in range(a_i + 1, len(top)):
            v = top[b_i]
            if float(np.linalg.norm(s_rot[u] - s_rot[v], "fro")) < 0.25:
                lat_vecs.append(s_pos[v] - s_pos[u])
    if not lat_vecs:
        return "1D" if len(top) >= 4 else "0D"
    lv = np.array(lat_vecs, dtype=float)
    evals = np.sort(np.linalg.eigvalsh(lv.T @ lv / len(lv)))[::-1]
    r_lat = [math.sqrt(max(0.0, float(ev))) for ev in evals]
    lat_thresh = max(8.0, 0.48 * min_mode_d)
    n_lat_dim = sum(1 for r in r_lat if r > lat_thresh)
    return f"{n_lat_dim}D"


def connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if not seen[i]:
            comp = []
            q = [i]
            seen[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
            clusters.append(comp)
    clusters.sort(key=len, reverse=True)
    return clusters


def multi_nucleus_rg_axes(
    pos: np.ndarray, clusters: List[List[int]], min_size: int = 8
) -> Tuple[float, float, float]:
    qual = [c for c in clusters if len(c) >= min_size]
    if not qual and clusters and len(clusters[0]) >= 3:
        qual = [clusters[0]]
    if not qual:
        return (0.0, 0.0, 0.0)
    w_sum = 0.0
    rg_acc = np.zeros(3, dtype=float)
    for c in qual:
        pts = pos[c] - np.mean(pos[c], axis=0)
        evals = np.sort(np.linalg.eigvalsh(pts.T @ pts / len(c)))[::-1]
        r_vec = np.array([math.sqrt(max(0.0, float(ev))) for ev in evals], dtype=float)
        w = float(len(c))
        rg_acc += w * r_vec
        w_sum += w
    rg_mean = rg_acc / max(w_sum, 1.0)
    return (round(float(rg_mean[0]), 1), round(float(rg_mean[1]), 1), round(float(rg_mean[2]), 1))


def summarize_run(run_dir: Path) -> Dict[str, object]:
    cfg_path = run_dir / "config.json"
    prog_path = run_dir / "progress.jsonl"
    traj_path = run_dir / "trajectory.jsonl"
    if not cfg_path.exists():
        return {"run": run_dir.name, "status": "MISSING_CONFIG"}
    cfg = json.loads(cfg_path.read_text())
    shape_name = Path(str(cfg.get("shape", ""))).name
    chem_name = Path(str(cfg.get("chemistry_model", ""))).name
    salt = f"{cfg.get('salt_type', 'none')}:{cfg.get('salt_concentration_m', 0.0)}M"
    prog = read_last_jsonl(prog_path)
    traj = read_last_jsonl(traj_path)
    if prog is None:
        return {
            "run": run_dir.name,
            "shape": shape_name,
            "chem": chem_name,
            "salt": salt,
            "status": "INITIALIZING",
        }
    sweep = int(prog.get("sweep", 0))
    target_sweeps = int(cfg.get("sweeps", 1000000))
    chem_e = float(prog.get("chemistry", prog.get("chem", 0.0)))
    rt_contacts = int(prog.get("contacts", 0))
    if traj is None or "poses" not in traj:
        return {
            "run": run_dir.name,
            "shape": shape_name,
            "chem": chem_name,
            "salt": salt,
            "sweep": f"{sweep}/{target_sweeps}",
            "status": "NO_TRAJ",
        }
    poses = traj["poses"]
    n_bodies = len(poses)
    pos = np.array([p["position"] for p in poses], dtype=float)
    rots = [quat_to_rot(p["orientation"]) for p in poses]

    shape_raw = cfg.get("shape")
    shape_path = Path(str(shape_raw)) if shape_raw else Path()
    if not shape_path.exists():
        shape_path = REPO_ROOT / "examples" / "polymorphs" / shape_path.name
    if not shape_path.exists():
        shape_path = REPO_ROOT / "examples" / shape_path.name
    r_bound = monomer_bounding_radius(shape_path)

    seed_cfg_raw = cfg.get("seed_config")
    seed_cfg_path = Path(seed_cfg_raw) if seed_cfg_raw else Path()
    if not seed_cfg_path.exists():
        seed_cfg_path = REPO_ROOT / "examples" / "polymorphs" / seed_cfg_path.name
    modes = extract_native_modes(seed_cfg_path, r_bound=r_bound)
    d_geom_cut = (
        max(float(np.linalg.norm(tm)) for tm, _ in modes) + 3.7
        if modes
        else 1.75 * r_bound
    )

    adj_geom: List[List[int]] = [[] for _ in range(n_bodies)]
    adj_nat: List[List[int]] = [[] for _ in range(n_bodies)]
    n_geom_pairs = 0
    n_native = 0
    mode_counts: Dict[int, int] = {}
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dv = pos[j] - pos[i]
            d = float(np.linalg.norm(dv))
            if d < d_geom_cut:
                adj_geom[i].append(j)
                adj_geom[j].append(i)
                n_geom_pairs += 1
                t_ij = rots[i].T @ dv
                r_ij = rots[i].T @ rots[j]
                t_ji = rots[j].T @ (-dv)
                r_ji = rots[j].T @ rots[i]
                best_k = -1
                best_sc = 999.0
                for k, (tm, rm) in enumerate(modes):
                    for tr, rr in ((t_ij, r_ij), (t_ji, r_ji)):
                        dt = float(np.linalg.norm(tr - tm))
                        tr_v = (np.trace(rr.T @ rm) - 1.0) * 0.5
                        dth = math.degrees(math.acos(max(-1.0, min(1.0, float(tr_v)))))
                        sc = math.hypot(dt / 3.5, dth / 20.0)
                        if sc < best_sc:
                            best_sc = sc
                            best_k = k
                if best_sc < 1.35 and best_k >= 0:
                    n_native += 1
                    adj_nat[i].append(j)
                    adj_nat[j].append(i)
                    mode_counts[best_k] = mode_counts.get(best_k, 0) + 1

    geom_clusters = connected_components(adj_geom)
    nat_clusters = connected_components(adj_nat)
    geom_sizes = [len(c) for c in geom_clusters]
    nat_sizes = [len(c) for c in nat_clusters]
    free_monomers = sum(1 for s in geom_sizes if s == 1)

    nat_max = nat_sizes[0] if nat_sizes else 1
    geom_max = geom_sizes[0] if geom_sizes else 1
    use_clusters = nat_clusters if nat_max >= 8 else geom_clusters
    rg_axes = multi_nucleus_rg_axes(pos, use_clusters, min_size=8)

    nat_nodes_ge8 = [u for c in nat_clusters if len(c) >= 8 for u in c]
    z_nat = (
        float(np.mean([len(adj_nat[u]) for u in nat_nodes_ge8])) if nat_nodes_ge8 else 0.0
    )

    n_contacts = rt_contacts if rt_contacts > 0 else n_geom_pairs
    nat_frac = (100.0 * min(n_native, n_contacts) / n_contacts) if n_contacts > 0 else 0.0
    e_per_c = (chem_e / n_contacts) if n_contacts > 0 else 0.0

    active_modes = [k for k, cnt in mode_counts.items() if cnt >= 10]
    if not active_modes:
        active_modes = [k for k, cnt in mode_counts.items() if cnt >= 5]
    lat_dim = supercell_percolation_dim(seed_cfg_path, modes, active_modes)

    n_nat_ge10 = sum(1 for s in nat_sizes if s >= 10)
    r1_r2 = rg_axes[0] / max(rg_axes[1], 1.0) if rg_axes[0] > 0 else 1.0
    r2_r3 = rg_axes[1] / max(rg_axes[2], 1.0) if rg_axes[1] > 0 else 1.0
    if n_contacts < 100 or (nat_max < 20 and geom_max < 20):
        morph = "no nucleation"
    elif nat_frac < 25.0 or nat_max < 12:
        morph = "no nucleation (amorphous)" if geom_max >= 20 else "no nucleation"
    elif nat_max < 20:
        morph = "no nucleation"
    elif lat_dim == "1D" or (r1_r2 >= 3.5 and z_nat < 3.5):
        morph = "1D filaments"
    elif lat_dim == "2D" or (z_nat < 4.2 and r2_r3 >= 1.9):
        morph = "2D sheets"
    elif n_nat_ge10 >= 8 and nat_max <= 60 and nat_frac >= 70.0 and r1_r2 < 2.2:
        morph = "homogeneous 3D nuclei"
    elif r1_r2 > 2.5 and r2_r3 < 1.6:
        morph = "filament + 3D lateral"
    else:
        morph = "3D crystal"

    display_sizes = nat_sizes[:5] if nat_max >= 8 else geom_sizes[:5]
    return {
        "run": run_dir.name,
        "shape": shape_name.replace("monomer-shape-", "").replace(".json", ""),
        "chem": chem_name.replace("chemistry-model-", "").replace(".json", ""),
        "salt": salt,
        "sweep": f"{sweep // 1000}k/{target_sweeps // 1000}k",
        "contacts": n_contacts,
        "E_c": f"{e_per_c:+.2f}",
        "nat%": f"{nat_frac:.1f}% ({n_native}/{n_contacts})",
        "free": free_monomers,
        "top5": display_sizes,
        "rg": rg_axes,
        "morph": morph,
    }


def parse_run_spec(runs_root: Path, spec: str) -> List[Path]:
    out: List[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m_range = re.fullmatch(r"(?:nuc-)?(\d+)\.\.(?:nuc-)?(\d+)", part)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            for k in range(a, b + 1):
                out.append(runs_root / f"nuc-{k}")
        elif re.fullmatch(r"\d+", part):
            out.append(runs_root / f"nuc-{int(part)}")
        else:
            p = Path(part)
            out.append(p if p.exists() else (runs_root / part))
    return out


def cmd_launch(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    target_map = {str(t["key"]): t for t in BENCHMARK_TARGETS}

    if args.targets:
        keys = [k.strip() for k in args.targets.split(",") if k.strip()]
    else:
        keys = PRESETS[args.preset]

    selected: List[Dict[str, object]] = []
    for k in keys:
        if k not in target_map:
            sys.stderr.write(f"Unknown target '{k}'. Valid targets: {', '.join(target_map)}\n")
            return 1
        selected.append(target_map[k])

    if not selected:
        sys.stderr.write(f"No targets selected for preset='{args.preset}' / targets='{args.targets}'.\n")
        return 1

    binary = REPO_ROOT / "target" / "release" / "morph-mc"
    if not args.no_build and not args.dry_run:
        print(f"[build] Compiling {binary} ...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "morph-mc"],
            cwd=str(REPO_ROOT),
            check=True,
        )

    start_id = args.start_id if args.start_id is not None else next_nuc_id(runs_root)
    n_cpus = os.cpu_count() or 16
    n_jobs = max(1, min(len(selected), max(1, args.jobs)))
    threads_per_job = args.threads_per_job or max(1, min(12, n_cpus // n_jobs))

    manifest_rows = []
    procs: List[Tuple[str, str, subprocess.Popen, Path]] = []

    print(
        f"=== Launching {len(selected)} nucleation benchmark simulations "
        f"(nuc-{start_id}..nuc-{start_id + len(selected) - 1}, "
        f"jobs={n_jobs}, RAYON_NUM_THREADS={threads_per_job}) ==="
    )

    for idx, t in enumerate(selected):
        nuc_id = start_id + idx
        run_name = f"nuc-{nuc_id}"
        out_dir = runs_root / run_name
        cmd = build_morph_mc_cmd(
            binary=binary,
            repo_root=REPO_ROOT,
            target=t,
            out_dir=out_dir,
            free_bodies=args.free_bodies,
            concentration_um=args.concentration_um,
            sweeps=args.sweeps,
            sample_every=args.sample_every,
            seed=args.seed,
        )
        row = {
            "run_name": run_name,
            "target_key": t["key"],
            "label": t["label"],
            "out_dir": str(out_dir),
            "cmd": cmd,
        }
        manifest_rows.append(row)
        if args.dry_run:
            print(f"  [DRY-RUN] {run_name:8s} | {str(t['label']):26s} | {' '.join(cmd)}")
            continue

        while len([p for p in procs if p[2].poll() is None]) >= n_jobs:
            time.sleep(1.0)

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads_per_job)
        log_f = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        row["pid"] = proc.pid
        row["log"] = str(log_path)
        procs.append((run_name, str(t["label"]), proc, log_path))
        print(f"  [STARTED] {run_name:8s} | {str(t['label']):26s} | pid={proc.pid} | log={log_path}")

    if not args.dry_run:
        manifest_path = (
            runs_root / f"benchmark-manifest-nuc-{start_id}-{start_id + len(selected) - 1}.json"
        )
        manifest_path.write_text(json.dumps(manifest_rows, indent=2))
        print(f"\nSaved batch manifest to {manifest_path}")
        print(
            f"Monitor progress with:\n"
            f"  python3 tools/chemistry_gap/benchmark_polymorph_nucleation.py status "
            f"--runs {start_id}..{start_id + len(selected) - 1}"
        )
        if args.wait:
            print("Waiting for all jobs to complete...")
            for run_name, label, proc, _ in procs:
                rc = proc.wait()
                print(f"  [DONE] {run_name} ({label}) exited with code {rc}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).resolve()
    run_dirs = parse_run_spec(runs_root, args.runs)
    print(
        f"{'Run':8s} | {'Polymorph':17s} | {'Model':24s} | {'Salt':11s} | {'Sweeps':9s} | "
        f"{'E/c':6s} | {'Native Contacts':16s} | {'Free':4s} | {'Top Clusters':18s} | "
        f"{'Rg Axes (A)':17s} | {'Morphology'}"
    )
    print("-" * 165)
    for d in run_dirs:
        s = summarize_run(d)
        if "contacts" not in s:
            print(f"{s['run']:8s} | status={s.get('status')}")
            continue
        print(
            f"{s['run']:8s} | {s['shape'][:17]:17s} | {s['chem'][:24]:24s} | {s['salt']:11s} | "
            f"{s['sweep']:9s} | {s['E_c']:6s} | {s['nat%']:16s} | {s['free']:4d} | "
            f"{str(s['top5']):18s} | {str(s['rg']):17s} | {s['morph']}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch and summarize parallel HEWL polymorph nucleation benchmarks."
    )
    sub = parser.add_subparsers(dest="subcmd")

    p_launch = sub.add_parser("launch", help="Launch parallel morph-mc nucleation runs")
    p_launch.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="all10",
        help="Target preset: 'core4' (1lyz-nacl, 3wl2-nano3, 2d4i-nano3, 1lyz-monomer) or 'all10'",
    )
    p_launch.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated list of target keys (overrides --preset)",
    )
    p_launch.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory where nuc-<N> run folders are created",
    )
    p_launch.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Starting nuc-<N> integer ID (default: auto-detect next available ID)",
    )
    p_launch.add_argument("--free-bodies", type=int, default=256)
    p_launch.add_argument("--concentration-um", type=float, default=2800.0)
    p_launch.add_argument("--sweeps", type=int, default=1000000)
    p_launch.add_argument("--sample-every", type=int, default=50)
    p_launch.add_argument("--seed", type=int, default=1)
    p_launch.add_argument("--jobs", type=int, default=10, help="Max concurrent morph-mc processes")
    p_launch.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help="RAYON_NUM_THREADS per process (default: auto-partition across CPU cores)",
    )
    p_launch.add_argument("--no-build", action="store_true", help="Skip cargo build --release")
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.add_argument("--wait", action="store_true", help="Wait for all launched runs to finish")

    p_status = sub.add_parser("status", help="Summarize progress and crystal morphology of nuc-<N> runs")
    p_status.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Run range or comma list, e.g. '102..105' or 'nuc-102,nuc-103'",
    )
    p_status.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory containing nuc-<N> run folders",
    )

    args = parser.parse_args()
    if args.subcmd == "status":
        return cmd_status(args)
    if args.subcmd == "launch":
        return cmd_launch(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
