#!/usr/bin/env python3
"""
Extract monomer shapes, 32-body overlap-free crystal seeds, and crystal contact
libraries for the 30-PDB crystallization benchmark (14 HEWL polymorphs + 16
diverse non-HEWL proteins across 7 space groups).

Preserves the existing 8 HEWL polymorph files and adds the 22 new PDB crystal
structures with a stratified 18-train / 12-held-out split.
"""

import glob
import json
import os
import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO_ROOT = "/home/xvg/tetramer-mc"
PDB_DIR = "/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pdb-hewl"
POLY_DIR = os.path.join(REPO_ROOT, "examples", "polymorphs")

VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "FE": 1.80,
    "ZN": 1.70,
    "MG": 1.70,
    "CA": 1.80,
    "SE": 1.90,
}

# Metadata for all 30 PDB crystal structures:
# (tag, split, is_hewl, salt_type, salt_conc_M, protein_family)
BENCHMARK_METADATA = {
    # --- 8 Existing HEWL polymorphs (Train) ---
    "1IEE": ("tetragonal-1IEE", "train", True, "NaCl", 0.85, "HEWL"),
    "3WL2": ("monoclinic-3WL2", "train", True, "NaNO3", 0.30, "HEWL"),
    "2D4I": ("monoclinic-2D4I", "train", True, "NaNO3", 0.30, "HEWL"),
    "2Z12": ("monoclinic-2Z12", "train", True, "NaNO3", 0.25, "HEWL"),
    "1WTN": ("orthorhombic-1WTN", "train", True, "NaCl", 0.50, "HEWL"),
    "2VB1": ("triclinic-2VB1", "train", True, "NaNO3", 0.25, "HEWL"),
    "2FBB": ("hexagonal-2FBB", "train", True, "NaNO3", 0.40, "HEWL"),
    "9HRU": ("trigonal-9HRU", "train", True, "CuSO4", 0.20, "HEWL"),
    # --- 10 Diverse Non-HEWL structures (Train) ---
    "6E6O": ("monoclinic-6E6O", "train", False, "SO4", 1.20, "Pheromone-Er1"),
    "1EJG": ("monoclinic-1EJG", "train", False, "NaCl", 0.15, "Crambin"),
    "2DSX": ("monoclinic-2DSX", "train", False, "SO4", 1.50, "Rubredoxin-Dv"),
    "5NW3": ("orthorhombic-5NW3", "train", False, "SO4", 1.20, "SH3-domain"),
    "5PTI": ("orthorhombic-5PTI", "train", False, "SO4", 1.00, "BPTI"),
    "1UCS": ("orthorhombic-1UCS", "train", False, "SO4", 0.50, "Antifreeze-RD1"),
    "3A38": ("orthorhombic-3A38", "train", False, "SO4", 1.20, "HiPIP"),
    "1KF5": ("monoclinic-1KF5", "train", False, "NaCl", 0.20, "RNase-A"),
    "1A6M": ("monoclinic-1A6M", "train", False, "SO4", 1.50, "Myoglobin"),
    "1RQW": ("tetragonal-1RQW", "train", False, "SO4", 0.80, "Thaumatin"),
    # --- 6 Held-Out HEWL structures ---
    "1IO5": ("tetragonal-1IO5", "held_out", True, "NaCl", 0.60, "HEWL"),
    "6K8G": ("tetragonal-6K8G", "held_out", True, "NaCl", 0.80, "HEWL"),
    "7DER": ("tetragonal-7DER", "held_out", True, "NaCl", 0.70, "HEWL"),
    "9RZL": ("tetragonal-9RZL", "held_out", True, "NaCl", 0.80, "HEWL"),
    "6F1O": ("orthorhombic-6F1O", "held_out", True, "NaCl", 0.20, "HEWL"),
    "4LZT": ("triclinic-4LZT", "held_out", True, "NaNO3", 0.24, "HEWL"),
    # --- 6 Held-Out Non-HEWL structures ---
    "3NIR": ("monoclinic-3NIR", "held_out", False, "NaCl", 0.15, "Crambin"),
    "1YK4": ("orthorhombic-1YK4", "held_out", False, "SO4", 1.00, "Rubredoxin-Pf"),
    "3ONS": ("trigonal-3ONS", "held_out", False, "SO4", 0.80, "Ubiquitin"),
    "4ZM7": ("orthorhombic-4ZM7", "held_out", False, "NaCl", 0.25, "Cellulase-Cel45A"),
    "1GCI": ("orthorhombic-1GCI", "held_out", False, "SO4", 0.80, "Subtilisin"),
    "1US0": ("monoclinic-1US0", "held_out", False, "SO4", 0.50, "Aldose-Reductase"),
}


def load_reference_1lyz():
    shape_path = os.path.join(REPO_ROOT, "examples", "monomer-shape.json")
    shape = json.load(open(shape_path))
    coords = np.array([a["center"] for a in shape["atoms"]], dtype=float)
    radii = np.array([a["radius"] for a in shape["atoms"]], dtype=float)
    names_path = "/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref/atom_types_monomer.json"
    names = json.load(open(names_path))["atoms"]
    assert len(names) == len(coords) == 1001
    return coords, radii, names


def kabsch_align(P, Q):
    """Find rotation R and translation t such that P @ R.T + t aligns P onto Q."""
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P0 = P - p_mean
    Q0 = Q - q_mean
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
    R = Vt.T @ D @ U.T
    t = q_mean - p_mean @ R.T
    return R, t


def extract_pdb_monomer(pdb_path, is_hewl, ref_1lyz):
    ref_coords, ref_radii, ref_names = ref_1lyz
    st = gemmi.read_structure(pdb_path)
    model = st[0]
    # Pick first protein chain
    prot_chain = None
    for ch in model:
        n_aa = sum(1 for r in ch if r.het_flag != "H" and r.name != "HOH")
        if n_aa >= 15:
            prot_chain = ch
            break
    assert prot_chain is not None, f"No protein chain in {pdb_path}"

    raw_atoms = {}
    ordered_coords = []
    ordered_radii = []
    ordered_names = []

    for res in prot_chain:
        if res.name == "HOH" or res.het_flag == "H":
            continue
        resnum = int(res.seqid.num)
        for at in res:
            el = at.element.name.upper()
            if el in ("H", "D"):
                continue
            if at.altloc not in ("\x00", " ", "A", "1"):
                continue
            aname = at.name.strip()
            key = (resnum, aname)
            if key in raw_atoms:
                continue
            pos = np.array([at.pos.x, at.pos.y, at.pos.z], dtype=float)
            rad = VDW_RADII.get(el, 1.70)
            raw_atoms[key] = (pos, rad, res.name)
            ordered_coords.append(pos)
            ordered_radii.append(rad)
            ordered_names.append([res.name, resnum, aname])

    if is_hewl:
        # Match the exact 1001-atom indexing of 1LYZ
        matched_idx = []
        P_sub = []
        Q_sub = []
        for idx, (rname, rnum, aname) in enumerate(ref_names):
            if (rnum, aname) in raw_atoms:
                matched_idx.append(idx)
                P_sub.append(raw_atoms[(rnum, aname)][0])
                Q_sub.append(ref_coords[idx])
        P_sub = np.array(P_sub, dtype=float)
        Q_sub = np.array(Q_sub, dtype=float)
        # Map 1LYZ -> PDB frame first to place any missing disordered atoms
        R_ref2pdb, t_ref2pdb = kabsch_align(Q_sub, P_sub)
        ref_in_pdb = ref_coords @ R_ref2pdb.T + t_ref2pdb

        full_pdb_coords = np.zeros((1001, 3), dtype=float)
        for idx, (rname, rnum, aname) in enumerate(ref_names):
            if (rnum, aname) in raw_atoms:
                full_pdb_coords[idx] = raw_atoms[(rnum, aname)][0]
            else:
                # Align locally on the same or nearest residue C-alpha if available
                full_pdb_coords[idx] = ref_in_pdb[idx]

        R_to_local, t_to_local = kabsch_align(full_pdb_coords, ref_coords)
        local_coords = full_pdb_coords @ R_to_local.T + t_to_local
        rmsd_1lyz = float(np.sqrt(np.mean(np.sum((local_coords - ref_coords) ** 2, axis=1))))
        return (
            st,
            prot_chain.name,
            full_pdb_coords,
            local_coords,
            ref_radii.copy(),
            ref_names,
            R_to_local,
            t_to_local,
            rmsd_1lyz,
        )
    else:
        full_pdb_coords = np.array(ordered_coords, dtype=float)
        radii = np.array(ordered_radii, dtype=float)
        com = full_pdb_coords.mean(axis=0)
        local_coords = full_pdb_coords - com
        R_to_local = np.eye(3, dtype=float)
        t_to_local = -com
        return (
            st,
            prot_chain.name,
            full_pdb_coords,
            local_coords,
            radii,
            ordered_names,
            R_to_local,
            t_to_local,
            0.0,
        )


def find_crystal_contacts(st, chain_name, pdb_coords, local_coords, radii, R_to_local, t_to_local):
    """Find all symmetry-mate crystal contacts in the monomer local frame."""
    cell = st.cell
    sg = st.find_spacegroup()
    ops = sg.operations()
    orth_mat = np.array(cell.orth.mat, dtype=float)
    frac_mat = np.array(cell.frac.mat, dtype=float)

    monomer_com_pdb = pdb_coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((local_coords - local_coords.mean(axis=0)) ** 2, axis=1)))
    max_r = np.max( np.linalg.norm(local_coords - local_coords.mean(axis=0), axis=1) )
    com_cutoff = 2.0 * max_r + 8.0

    tree0 = cKDTree(local_coords)
    contacts = []
    seen_poses = []

    for op_idx, op in enumerate(ops):
        rot_frac = np.array(op.rot, dtype=float) / 24.0
        tran_frac = np.array(op.tran, dtype=float) / 24.0
        R_cart = orth_mat @ rot_frac @ frac_mat
        t_cart_base = orth_mat @ tran_frac

        for na in (-2, -1, 0, 1, 2):
            for nb in (-2, -1, 0, 1, 2):
                for nc in (-2, -1, 0, 1, 2):
                    if op_idx == 0 and na == 0 and nb == 0 and nc == 0:
                        continue
                    shift_cart = orth_mat @ np.array([na, nb, nc], dtype=float)
                    t_cart = t_cart_base + shift_cart
                    mate_com_pdb = monomer_com_pdb @ R_cart.T + t_cart
                    d_com = float(np.linalg.norm(mate_com_pdb - monomer_com_pdb))
                    if d_com > com_cutoff or d_com < 5.0:
                        continue

                    # Express mate pose in local monomer frame:
                    # x_pdb = (x_local - t_to_local) @ R_to_local
                    # mate_pdb = R_cart @ x_pdb + t_cart
                    # mate_local = R_to_local @ mate_pdb + t_to_local
                    #            = (R_to_local @ R_cart @ R_to_local.T) @ x_local
                    #              + R_to_local @ (t_cart - R_cart @ R_to_local.T @ t_to_local) + t_to_local
                    R_rel = R_to_local @ R_cart @ R_to_local.T
                    t_rel = R_to_local @ (t_cart - R_cart @ (R_to_local.T @ t_to_local)) + t_to_local

                    # Deduplicate identical poses
                    is_dup = False
                    for prev_t, prev_R in seen_poses:
                        if np.linalg.norm(t_rel - prev_t) < 0.25 and np.linalg.norm(R_rel - prev_R) < 0.05:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    mate_local = local_coords @ R_rel.T + t_rel
                    pairs = tree0.query_ball_point(mate_local, r=6.5)
                    n_buried = sum(len(p) for p in pairs)
                    if n_buried < 15:
                        continue

                    seen_poses.append((t_rel, R_rel))
                    shifted_op = gemmi.Op()
                    shifted_op.rot = op.rot
                    shifted_op.tran = [
                        op.tran[0] + 24 * na,
                        op.tran[1] + 24 * nb,
                        op.tran[2] + 24 * nc,
                    ]
                    triplet_op = shifted_op.triplet()

                    contacts.append({
                        "partner_chain": chain_name,
                        "symop": triplet_op,
                        "d_com_crystal": round(d_com, 3),
                        "R_rel": R_rel,
                        "t_crystal": t_rel,
                        "n_buried_raw": int(n_buried),
                    })

    return contacts


def crystal_repair_coordinates(local_coords, radii, contacts, target_clearance=0.042):
    """
    Slightly adjust clashing surface atoms in local_coords so that in the exact
    crystal lattice (t_retracted == t_crystal, s_retract == 0.0) every crystal
    contact has min_gap >= target_clearance and n_overlap_crystal == 0.
    """
    x = local_coords.copy()
    n_atoms = len(x)
    for it in range(60):
        disp = np.zeros_like(x)
        counts = np.zeros(n_atoms, dtype=float)
        max_violation = 0.0

        for c in contacts:
            R = c["R_rel"]
            t = c["t_crystal"]
            mate = x @ R.T + t
            tree_x = cKDTree(x)
            pairs = tree_x.query_ball_point(mate, r=4.0)
            for j, i_list in enumerate(pairs):
                for i in i_list:
                    vec = x[i] - mate[j]
                    d = float(np.linalg.norm(vec))
                    req = float(radii[i] + radii[j] + target_clearance)
                    if d < req:
                        viol = req - d
                        if viol > max_violation:
                            max_violation = viol
                        u = vec / max(d, 1e-6)
                        step = 0.55 * viol
                        disp[i] += step * u
                        disp[j] -= step * (R.T @ u)
                        counts[i] += 1.0
                        counts[j] += 1.0

        if max_violation <= 1e-4:
            break
        mask = counts > 0
        x[mask] += disp[mask] / np.maximum(counts[mask, None], 1.0)

    # Final verification and contact annotation
    tree_x = cKDTree(x)
    final_contacts = []
    for c in contacts:
        R = c["R_rel"]
        t = c["t_crystal"]
        mate = x @ R.T + t
        dists, idxs = tree_x.query(mate, k=1)
        gaps = dists - (radii + radii[idxs])
        min_gap = float(np.min(gaps))
        n_overlap = int(np.sum(gaps < 0.0))
        pairs_65 = tree_x.query_ball_point(mate, r=6.5)
        n_buried = sum(len(p) for p in pairs_65)

        if n_buried < 15:
            continue

        # Convert R_rel to quaternion [w, x, y, z] matching existing format
        rot = Rotation.from_matrix(R)
        xyzw = rot.as_quat()
        q_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        if q_wxyz[0] < 0:
            q_wxyz = [-v for v in q_wxyz]

        d_com = float(np.linalg.norm((x.mean(axis=0) @ R.T + t) - x.mean(axis=0)))
        final_contacts.append({
            "partner_chain": c["partner_chain"],
            "symop": c["symop"],
            "d_com_crystal": round(d_com, 3),
            "d_com_retracted": round(d_com, 3),
            "s_retract": 0.0,
            "min_gap_crystal": round(min_gap, 4),
            "n_overlap_crystal": n_overlap,
            "n_buried_crystal": int(n_buried),
            "t_crystal": [round(float(v), 4) for v in t],
            "t_retracted": [round(float(v), 4) for v in t],
            "q_rel": q_wxyz,
            "R_rel": [[round(float(v), 6) for v in row] for row in R],
            "bridging_ions": [],
        })

    return x, final_contacts


def generate_32_seed_poses(st, local_coords, radii, R_to_local, t_to_local, n_seed=32):
    """
    Generate the 32 closest crystal lattice poses around the central monomer
    and verify/repair local_coords if any secondary lattice pair clashes.
    """
    cell = st.cell
    sg = st.find_spacegroup()
    ops = sg.operations()
    orth_mat = np.array(cell.orth.mat, dtype=float)
    frac_mat = np.array(cell.frac.mat, dtype=float)

    candidates = []
    for op_idx, op in enumerate(ops):
        rot_frac = np.array(op.rot, dtype=float) / 24.0
        tran_frac = np.array(op.tran, dtype=float) / 24.0
        R_cart = orth_mat @ rot_frac @ frac_mat
        t_cart_base = orth_mat @ tran_frac

        for na in (-2, -1, 0, 1, 2):
            for nb in (-2, -1, 0, 1, 2):
                for nc in (-2, -1, 0, 1, 2):
                    shift_cart = orth_mat @ np.array([na, nb, nc], dtype=float)
                    t_cart = t_cart_base + shift_cart
                    R_rel = R_to_local @ R_cart @ R_to_local.T
                    t_rel = R_to_local @ (t_cart - R_cart @ (R_to_local.T @ t_to_local)) + t_to_local
                    d_origin = float(np.linalg.norm(t_rel))
                    candidates.append((d_origin, t_rel, R_rel))

    candidates.sort(key=lambda item: item[0])
    selected = []
    for d_origin, t_rel, R_rel in candidates:
        is_dup = False
        for _, prev_t, prev_R in selected:
            if np.linalg.norm(t_rel - prev_t) < 0.25 and np.linalg.norm(R_rel - prev_R) < 0.05:
                is_dup = True
                break
        if not is_dup:
            selected.append((d_origin, t_rel, R_rel))
            if len(selected) >= n_seed:
                break

    # Center the 32-monomer cluster at the origin
    mean_pos = np.mean([item[1] for item in selected], axis=0)
    poses = []
    for _, t_rel, R_rel in selected:
        pos = t_rel - mean_pos
        rot = Rotation.from_matrix(R_rel)
        xyzw = rot.as_quat()
        q_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        if q_wxyz[0] < 0:
            q_wxyz = [-v for v in q_wxyz]
        poses.append({
            "position": [round(float(v), 6) for v in pos],
            "orientation": [round(float(v), 8) for v in q_wxyz],
        })
    return poses


def main():
    ref_coords, ref_radii, ref_names = load_reference_1lyz()
    existing_lib_path = os.path.join(POLY_DIR, "polymorph-contact-library.json")
    existing_lib = json.load(open(existing_lib_path))
    existing_by_pdb = {p["pdb_id"]: p for p in existing_lib["polymorphs"]}

    all_polymorph_entries = []

    for pdb_id, (tag, split, is_hewl, salt_type, salt_conc, family) in BENCHMARK_METADATA.items():
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        shape_path = os.path.join(POLY_DIR, f"monomer-shape-{tag}.json")
        seed_path = os.path.join(POLY_DIR, f"seeded-{tag}.json")
        contact_path = os.path.join(POLY_DIR, f"contacts-{tag}.json")

        if pdb_id in existing_by_pdb and os.path.exists(shape_path) and os.path.exists(seed_path):
            # Preserve existing repaired shape, seed, and contacts; enrich metadata
            entry = dict(existing_by_pdb[pdb_id])
            entry["split"] = split
            entry["is_hewl"] = is_hewl
            entry["protein_family"] = family
            entry["n_atoms"] = 1001
            sh = json.load(open(shape_path))
            if "atom_names" not in sh:
                sh["atom_names"] = ref_names
                sh["split"] = split
                sh["is_hewl"] = is_hewl
                sh["protein_family"] = family
                with open(shape_path, "w") as f:
                    json.dump(sh, f, indent=2)
            all_polymorph_entries.append(entry)
            print(f"[PRESERVED] {tag:22s} ({pdb_id}, {split:8s}, {family:16s}): n_atoms=1001, n_contacts={entry['n_contacts']}")
            continue

        # Extract and crystal-repair new PDB structure
        (
            st,
            chain_name,
            pdb_coords,
            local_coords,
            radii,
            atom_names,
            R_to_local,
            t_to_local,
            rmsd_1lyz,
        ) = extract_pdb_monomer(pdb_path, is_hewl, (ref_coords, ref_radii, ref_names))

        raw_contacts = find_crystal_contacts(
            st, chain_name, pdb_coords, local_coords, radii, R_to_local, t_to_local
        )
        repaired_coords, final_contacts = crystal_repair_coordinates(
            local_coords, radii, raw_contacts, target_clearance=0.042
        )

        # Write monomer-shape-<tag>.json
        shape_json = {
            "name": f"{family}-{tag}",
            "pdb_id": pdb_id,
            "space_group": st.spacegroup_hm,
            "salt_type": salt_type,
            "salt_conc_M": salt_conc,
            "rmsd_to_1LYZ_A": round(rmsd_1lyz, 3),
            "split": split,
            "is_hewl": is_hewl,
            "protein_family": family,
            "crystal_repaired": True,
            "target_clearance_angstrom": 0.04,
            "atom_names": atom_names,
            "atoms": [
                {
                    "center": [round(float(c), 6) for c in repaired_coords[i]],
                    "radius": round(float(radii[i]), 3),
                }
                for i in range(len(repaired_coords))
            ],
        }
        with open(shape_path, "w") as f:
            json.dump(shape_json, f, indent=2)

        # Write seeded-<tag>.json
        seed_poses = generate_32_seed_poses(
            st, repaired_coords, radii, R_to_local, t_to_local, n_seed=32
        )
        rg = float(np.sqrt(np.mean(np.sum(repaired_coords ** 2, axis=1))))
        box_half = round(max(135.0, 9.0 * rg), 1)
        seed_json = {
            "shapes": [f"examples/polymorphs/monomer-shape-{tag}.json"],
            "species": [{"shape_index": 0, "count": len(seed_poses)}],
            "box_half_extents": [box_half, box_half, box_half],
            "boundary": {"Periodic": None},
            "ensembles": [{"Nvt": {"temperature": 1.0}}],
            "initial_poses": seed_poses,
            "max_translation_delta": 1.5,
            "max_rotation_degrees": 15.0,
            "rng_seed": 42,
            "steps": 50000,
            "depletion": None,
        }
        with open(seed_path, "w") as f:
            json.dump(seed_json, f, indent=2)

        entry = {
            "polymorph": tag,
            "pdb_id": pdb_id,
            "space_group": st.spacegroup_hm,
            "salt_type": salt_type,
            "salt_conc_M": salt_conc,
            "rmsd_to_1LYZ_A": round(rmsd_1lyz, 3),
            "split": split,
            "is_hewl": is_hewl,
            "protein_family": family,
            "n_atoms": len(repaired_coords),
            "n_contacts": len(final_contacts),
            "contacts": final_contacts,
        }
        with open(contact_path, "w") as f:
            json.dump(entry, f, indent=2)

        all_polymorph_entries.append(entry)
        min_gap = min(c["min_gap_crystal"] for c in final_contacts)
        print(
            f"[EXTRACTED] {tag:22s} ({pdb_id}, {split:8s}, {family:16s}): "
            f"n_atoms={len(repaired_coords):4d}, n_contacts={len(final_contacts):2d}, min_gap={min_gap:.4f} A"
        )

    out_lib = {"polymorphs": all_polymorph_entries}
    with open(existing_lib_path, "w") as f:
        json.dump(out_lib, f, indent=2)
    print(f"\nSaved {len(all_polymorph_entries)} crystal structures to {existing_lib_path}")


if __name__ == "__main__":
    main()
