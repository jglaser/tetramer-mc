#!/usr/bin/env python3
"""Generate two publication-quality figures:
1. docs/figures/energy_model_shape_solvation_chemistry.{png,pdf,svg}
   Full breakdown of the Hierarchical Shape + Morphometric Solvation + SE(3) Chemistry
   Residual Energy functional (exact 1L2Y evaluations across all terms, radial basis
   and link functions, 2-body + 3-body surface-field junction sealing, and 2-body/multi-body
   basin free energies ln K).
2. docs/figures/chemistry_model_training_procedure.{png,pdf,svg}
   Full illustration of the Interface-Agnostic Contrastive & Energy-Balance Training
   Procedure (datasets: PDB corpus + 8 monomer crystal poses + 225 self-generated slide/twist
   perturbations; Stage A InfoNCE + Laplacian loss on P x P = 376 x 376 polar matrix;
   Stage B universal SE(3) involution vs open-generator energy-balance link function;
   held-out decoy discrimination and nuc-55 3D crystal nucleation).
"""

import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch, Rectangle
from scipy.spatial import cKDTree

SHAPE_PATH = "/home/xvg/tetramer-mc/examples/monomer-shape.json"
MODEL_PATH = "/home/xvg/tetramer-mc/examples/chemistry-model-monomer.json"
D_DIR = "/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pair-binding"
NUC55_DIR = "/vast/xvg/tetramer-mc-runs/morph-growth-20261012/nuc-55"

CHEM_CAT, CHEM_ANI, CHEM_NDON, CHEM_NACC = 1, 2, 4, 8
CHEM_TYR_OH, CHEM_AMIDE_ND, CHEM_PURE_ACC, CHEM_PURE_DON = 16, 32, 64, 128


def hat_b1(u):
    return np.maximum(0.0, 4.0 * u - 3.0)


def hat_b2(u):
    return np.maximum(0.0, np.minimum((4.0 / 3.0) * u, 4.0 * (1.0 - u)))


def hat_b12(u):
    return np.clip((4.0 / 3.0) * u, 0.0, 1.0)


def hat_hb(u):
    return hat_b1(u) + 0.25 * hat_b2(u)


def hat_wb(u):
    return hat_b2(u) * (1.0 - hat_b1(u))


def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def comp(fi, fj):
    return bool(
        ((fi & CHEM_NDON) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_NDON))
        or ((fi & CHEM_CAT) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_CAT))
        or ((fi & CHEM_CAT) and (fj & CHEM_ANI)) or ((fi & CHEM_ANI) and (fj & CHEM_CAT))
        or ((fi & CHEM_ANI) and (fj & CHEM_NDON)) or ((fi & CHEM_NDON) and (fj & CHEM_ANI))
        or ((fi & CHEM_TYR_OH) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_TYR_OH))
    )


def compute_model_diagnostics():
    """Compute exact energy decompositions, embedding matrices, and link function points."""
    shape = json.load(open(SHAPE_PATH))
    X = np.array([a["center"] for a in shape["atoms"]], dtype=float)
    R_vdw = np.array([a["radius"] for a in shape["atoms"]], dtype=float)
    model = json.load(open(MODEL_PATH))
    flags = np.array(model["flags"], dtype=int)
    weights = model["weights"]
    cos_weights = model["cos_weights"]
    U = np.array(model["cos_u"], dtype=float)
    V = np.array(model["cos_v"], dtype=float)
    a_all = np.array(model["cos_activity"], dtype=float)
    N = len(X)
    tree_x = cKDTree(X)

    polar_idx = np.where(flags > 0)[0]
    P = len(polar_idx)
    full_to_p = {int(idx): p for p, idx in enumerate(polar_idx)}
    Un = U[polar_idx]
    Vn = V[polar_idx]
    a_pol = a_all[polar_idx]
    S_mat = 0.5 * (Un @ Vn.T + Vn @ Un.T)

    nat_modes = {
        "m0": ([0.0, 0.0, 37.9], [1.0, 0.0, 0.0, 0.0]),
        "m1": ([19.47624995357571, 18.42510338365495, -9.475000000000003], [0.7071067811865477, 0.0, 0.0, 0.7071067811865475]),
        "m2": ([19.47624995357571, 18.42510338365495, 28.424999999999994], [0.7071067811865477, 0.0, 0.0, 0.7071067811865475]),
        "m3": ([-39.54999999999998, -1.6486466627693304, 8.973438122994171], [0.0, 1.0, 0.0, 0.0]),
        "m4": ([-20.073750046424273, -20.07375004642428, -19.451561877005823], [0.0, 0.7071067811865477, -0.7071067811865475, 0.0]),
        "m5": ([21.124896616345055, -21.124896616345048, 37.39843812299417], [0.0, 0.7071067811865475, 0.7071067811865477, 0.0]),
        "m6": ([21.124896616345055, -21.124896616345048, -0.5015618770058232], [0.0, 0.7071067811865475, 0.7071067811865477, 0.0]),
    }

    slid_modes = {
        "m2_slid #1": (
            [25.558208661897677, 17.14431684558781, 27.425798416524085],
            [0.8881644264175561, -0.11571731266034693, 0.0167161775235159, 0.4444029979714845],
        ),
        "m2_slid #2": (
            [16.96629413861771, 10.981842636940902, 33.51412525311606],
            [0.744444897770184, 0.05517118714756371, -0.19213065071077906, 0.6370586686876818],
        ),
        "m4_slid #0": (
            [-15.655601083935212, -22.281587149634607, -20.170399307969525],
            [-0.014648641698265356, 0.7143710530540851, -0.6708376476383641, 0.19858566505621555],
        ),
    }

    def decompose_pose(t, R):
        t = np.asarray(t, float)
        YB = (X @ R.T) + t
        pairs = tree_x.query_ball_tree(cKDTree(YB), 6.8)
        hb_cands, chb_cands, clash_cands = [], [], []
        max_u_a = np.zeros(N)
        max_u_b = np.zeros(N)
        max_comp_a = np.zeros(N)
        max_comp_b = np.zeros(N)
        ii_p, jj_p, psi_p, u_p = [], [], [], []
        for i, js in enumerate(pairs):
            fi = int(flags[i])
            for j in js:
                fj = int(flags[j])
                d = float(np.linalg.norm(X[i] - YB[j]))
                g = d - (R_vdw[i] + R_vdw[j])
                if g >= 2.8:
                    continue
                u = float(min(max((2.8 - g) / 2.8, 0.0), 1.0))
                if u <= 0.0:
                    continue
                if u > max_u_a[i]:
                    max_u_a[i] = u
                if u > max_u_b[j]:
                    max_u_b[j] = u
                if fi == 0 or fj == 0:
                    continue
                ii_p.append(full_to_p[i])
                jj_p.append(full_to_p[j])
                u_p.append(u)
                psi_p.append(float(hat_b12(u)))
                if comp(fi, fj):
                    if u > max_comp_a[i]:
                        max_comp_a[i] = u
                    if u > max_comp_b[j]:
                        max_comp_b[j] = u
                gh = float(hat_hb(u))
                if gh > 0.0:
                    if ((fi & CHEM_NDON) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_NDON)):
                        hb_cands.append((gh, min(i, j), max(i, j), i, j))
                    if (((fi & CHEM_CAT) and (fj & (CHEM_NACC | CHEM_ANI)))
                            or ((fj & CHEM_CAT) and (fi & (CHEM_NACC | CHEM_ANI)))
                            or ((fi & CHEM_ANI) and (fj & CHEM_NDON))
                            or ((fj & CHEM_ANI) and (fi & CHEM_NDON))):
                        chb_cands.append((gh, min(i, j), max(i, j), i, j))
                    if ((fi & CHEM_PURE_ACC) and (fj & CHEM_PURE_ACC)) or ((fi & CHEM_PURE_DON) and (fj & CHEM_PURE_DON)):
                        clash_cands.append((gh, i, j))

        hb_cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        ua, ub = set(), set()
        s_hb = 0.0
        for gh, _, _, i, j in hb_cands:
            if i not in ua and j not in ub:
                ua.add(i)
                ub.add(j)
                s_hb += gh
        chb_cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        ua, ub = set(), set()
        s_chb = 0.0
        for gh, _, _, i, j in chb_cands:
            if i not in ua and j not in ub:
                ua.add(i)
                ub.add(j)
                s_chb += gh
        s_clash = sum(gh * float(1.0 - hat_b12(max_comp_a[i])) * float(1.0 - hat_b12(max_comp_b[j])) for gh, i, j in clash_cands)
        s_wb_raw = 0.5 * (
            sum(float(hat_wb(u)) for u in max_comp_a if u > 0.0)
            + sum(float(hat_wb(u)) for u in max_comp_b if u > 0.0)
        )
        s_wb = s_wb_raw * float(np.exp(-1.5 * s_clash))
        unp_pol = sum(
            max(0.0, float(hat_hb(max_u_a[i]) - hat_b12(max_comp_a[i])))
            for i in range(N)
            if flags[i] and not (flags[i] & (CHEM_CAT | CHEM_ANI)) and (flags[i] & (CHEM_NDON | CHEM_NACC))
        )
        unp_pol += sum(
            max(0.0, float(hat_hb(max_u_b[j]) - hat_b12(max_comp_b[j])))
            for j in range(N)
            if flags[j] and not (flags[j] & (CHEM_CAT | CHEM_ANI)) and (flags[j] & (CHEM_NDON | CHEM_NACC))
        )
        unp_chg = sum(
            max(0.0, float(hat_hb(max_u_a[i]) - hat_b12(max_comp_a[i])))
            for i in range(N)
            if (flags[i] & (CHEM_CAT | CHEM_ANI))
        )
        unp_chg += sum(
            max(0.0, float(hat_hb(max_u_b[j]) - hat_b12(max_comp_b[j])))
            for j in range(N)
            if (flags[j] & (CHEM_CAT | CHEM_ANI))
        )

        S_cos = 0.0
        psi_sum = 0.0
        if ii_p:
            ii_a = np.array(ii_p, int)
            jj_a = np.array(jj_p, int)
            psi_a = np.array(psi_p, float)
            psi_sum = float(np.sum(psi_a))
            s_c = S_mat[ii_a, jj_a]
            wi = psi_a * a_pol[ii_a]
            wj = psi_a * a_pol[jj_a]
            num_a = np.bincount(ii_a, weights=wj * s_c, minlength=P)
            den_a = np.bincount(ii_a, weights=wj, minlength=P)
            num_b = np.bincount(jj_a, weights=wi * s_c, minlength=P)
            den_b = np.bincount(jj_a, weights=wi, minlength=P)
            S_cos = 0.5 * float(np.sum(a_pol * num_a / (den_a + 0.25)) + np.sum(a_pol * num_b / (den_b + 0.25)))

        w = weights
        e_hb = -w[0] * s_hb
        e_chb = -w[1] * s_chb
        e_wb = -w[3] * s_wb
        e_clash = +w[4] * s_clash + w[5] * unp_pol + w[6] * unp_chg
        e_pol0 = e_hb + e_chb + e_wb + e_clash
        e_cos = -cos_weights[0] * S_cos if S_cos >= 0 else -cos_weights[1] * S_cos
        is_inv = bool((np.linalg.norm(R @ t + t) < 3.0) and (np.linalg.norm(R @ R - np.eye(3)) < 0.25))
        return {
            "e_hb": e_hb + e_chb,
            "e_wb": e_wb,
            "e_desolv": e_clash,
            "e_pol0": e_pol0,
            "e_cos": e_cos,
            "e_tot": e_pol0 + e_cos,
            "S_cos": S_cos,
            "psi_sum": psi_sum,
            "is_inv": is_inv,
            "ii": np.array(ii_p, int),
            "jj": np.array(jj_p, int),
            "u": np.array(u_p, float),
        }

    decomps = {}
    for k, (t0, q0) in nat_modes.items():
        decomps[f"nat_{k}"] = decompose_pose(t0, quat_to_rot(q0))
    for k, (t0, q0) in slid_modes.items():
        decomps[k] = decompose_pose(t0, quat_to_rot(q0))

    # Held-out decoy scores
    neg_scores = []
    c_mon = json.load(open(f"{D_DIR}/contacts_monomer.json"))
    for c in c_mon:
        lbl = c.get("label", "")
        if not lbl.startswith("native") and lbl != "capture #9":
            p = c["pose"] if "pose" in c else c
            d = decompose_pose(p["position"], quat_to_rot(p["orientation"]))
            if d["psi_sum"] > 0:
                neg_scores.append(d["S_cos"])
    d41_path = "/home/xvg/tetramer-mc/tools/chemistry_gap/contacts_nuc41_decoys.json"
    if os.path.exists(d41_path):
        for c in json.load(open(d41_path)):
            p = c["pose"] if "pose" in c else c
            d = decompose_pose(p["position"], quat_to_rot(p["orientation"]))
            if d["psi_sum"] > 0:
                neg_scores.append(d["S_cos"])

    # Pos mask for off-diagonal vs positive pair similarity distributions
    pos_mask = np.zeros((P, P), dtype=bool)
    for k in [f"nat_m{m}" for m in range(7)]:
        d = decomps[k]
        ii, jj, u = d["ii"], d["jj"], d["u"]
        if len(ii) == 0:
            continue
        max_ui = np.zeros(P)
        max_uj = np.zeros(P)
        for pi, pj, uv in zip(ii, jj, u):
            if uv > max_ui[pi]:
                max_ui[pi] = uv
            if uv > max_uj[pj]:
                max_uj[pj] = uv
        for pi, pj, uv in zip(ii, jj, u):
            if uv >= 0.25 or uv >= 0.70 * max_ui[pi] or uv >= 0.70 * max_uj[pj]:
                pos_mask[pi, pj] = True
                pos_mask[pj, pi] = True

    s_pos_pairs = S_mat[pos_mask]
    s_neg_pairs = S_mat[~pos_mask]

    return decomps, S_mat, pos_mask, s_pos_pairs, s_neg_pairs, neg_scores


def make_figure_1_energy(decomps):
    """Figure 1: Hierarchical Shape + Morphometric Solvation + Chemistry Residual Energy."""
    fig = plt.figure(figsize=(16.6, 11.6), dpi=300, facecolor="white")
    gs_top = fig.add_gridspec(
        1, 2, left=0.04, right=0.975, bottom=0.525, top=0.880, wspace=0.18, width_ratios=[1.08, 1.0]
    )
    gs_bot = fig.add_gridspec(
        1, 2, left=0.04, right=0.975, bottom=0.065, top=0.440, wspace=0.35, width_ratios=[1.02, 1.04]
    )

    fig.text(
        0.5, 0.962,
        "Hierarchical Shape, Morphometric Solvation, and SE(3)-Invariant Chemistry Residual Energy",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0f172a"
    )
    fig.text(
        0.5, 0.928,
        r"$E_{\mathrm{total}}(\{T_A\}) = E_{\mathrm{shape}}(\mathrm{hard\;vdW}) + E_{\mathrm{morph}}(V,A,C,X) + E_{\mathrm{chem}}^{(2\mathrm{B})}(\mathrm{HB,\;WB,\;Cl^-,\;desolv,\;cosine}) + E_{\mathrm{junc}}^{(3\mathrm{B})}(B_1\;\mathrm{core},\;B_2\;\mathrm{collar})$",
        ha="center", va="center", fontsize=10.8, color="#334155"
    )

    # =========================================================================
    # PANEL A: 3-Layer Physical Architecture Schematic
    # =========================================================================
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_xlim(0, 10.5)
    ax_a.set_ylim(0, 7.5)
    ax_a.axis("off")
    ax_a.text(
        0.0, 7.25,
        "a   Three-Tier Physical Decomposition on Monomer Heavy-Atom Geometry (1001 Atoms)",
        fontsize=12.0, fontweight="bold", color="#0f172a", ha="left", va="center"
    )

    tiers = [
        (
            4.85, 2.02, "#f8fafc", "#64748b", "#1e293b",
            "1. Hard-Core Steric & Morphometric Solvation Baseline",
            [
                ("Hard vdW Exclusion:", r"$E_{\mathrm{shape}} = +\infty$ if $d_{ij} < r_i + r_j$ ($g_{ij} < 0$)"),
                ("Hadwiger WBII Morphometry:", r"$\Omega_{\mathrm{wb2}} = -pV + \sigma A + \kappa C + \bar{\kappa} X$ ($r_s = 1.4$ Å, $\eta = 0.383$)"),
                ("Typed Gaussian Burial:", r"$E_{\mathrm{solv}} = g_{\mathrm{burial}}(V_{\mathrm{ov}})\,\sum_{i}\tilde{w}_i\,\Delta V_i^{\mathrm{buried}}$ (bounded at $\pm 12.5\,k_BT$)"),
            ]
        ),
        (
            2.48, 2.18, "#f0fdf4", "#16a34a", "#14532d",
            "2. Two-Body Polar, Water-Bridge, Anion & Contrastive Cosine Residual",
            [
                ("1-to-1 H-Bonds & Salt Bridges:", r"$-w_{\mathrm{hb}}S_{\mathrm{hb}} - w_{\mathrm{chb}}S_{\mathrm{chb}} - w_{\mathrm{Cl}}S_{\mathrm{Cl^-}}$ ($g_{ij} < 2.8$ Å)"),
                ("Per-Atom Saturated Water Bridge:", r"$S_{\mathrm{wb}} = \frac{1}{2}\sum_{i}\hat{w}_{\mathrm{wb}}(u_{i,\max}^{\mathrm{comp}})\,e^{-1.5 S_{\mathrm{clash}}}$ (no wb_cap)"),
                ("Unpaired Burial & Clash Penalty:", r"$+w_{\mathrm{clash}}S_{\mathrm{clash}} + w_{\mathrm{unp,pol}}U_{\mathrm{pol}} + w_{\mathrm{unp,chg}}U_{\mathrm{chg}}$"),
                ("Contrastive Cosine Match:", r"$E_{\cos} = -w_{\cos}^{\pm}S_{\cos}^{AB},\;\;s_{ij} = \frac{1}{2}(\mathbf{u}_i\cdot\mathbf{v}_j + \mathbf{v}_i\cdot\mathbf{u}_j) \in [-1,1]$"),
            ]
        ),
        (
            0.12, 2.18, "#eff6ff", "#2563eb", "#1e3a8a",
            "3. Cutoff-Free Three-Body Vertex Surface-Field Hamiltonian (Core + Collar)",
            [
                ("11D Surface Feature Field on A:", r"$\boldsymbol{\psi}_i = \mathbf{n}_i \oplus \gamma_c a_i(\mathbf{u}_i - \mathbf{v}_i) \in \mathbb{R}^{11}$"),
                ("Inner-Core Steric Crowding (B1):", r"$E_{\mathrm{crowd}}^A = +w_{\mathrm{sat}}\sum_{B<C}\|\sum_{i\in A} B_1(u_i^B)B_1(u_i^C)\boldsymbol{\psi}_i\|$"),
                ("Hydration-Collar Seam Sealing (B2):", r"$E_{\mathrm{seal}}^A = -w_{\mathrm{seal}}\sum_{B<C} g_{AB}g_{AC}\frac{\|\mathbf{V}_{\mathrm{col},BC}^A\|^2}{\|\mathbf{V}_{\mathrm{col},BC}^A\| + w_0}$"),
                ("Desolvation Sharing (Ci):", r"$E_{\mathrm{coop}}^A = -\sum_{i\in A} w_{\mathrm{coop}}(f_i)\max(0,\,S_{i,\mathrm{comp}}^{\mathrm{tot}} - S_{i,\mathrm{comp}}^{\max})$"),
            ]
        ),
    ]

    for y0, h, fc, ec, tc, title, bullets in tiers:
        box = FancyBboxPatch(
            (0.08, y0), 10.30, h,
            boxstyle="round,pad=0.04,rounding_size=0.16",
            facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=1
        )
        ax_a.add_patch(box)
        ax_a.text(0.30, y0 + h - 0.30, title, ha="left", va="center", fontsize=10.2, fontweight="bold", color=tc, zorder=3)
        n_b = len(bullets)
        for idx, (b_head, b_math) in enumerate(bullets):
            by = y0 + h - 0.72 - idx * ((h - 0.82) / max(1, n_b - 1 if n_b > 1 else 1))
            ax_a.text(0.36, by, f"• {b_head}  {b_math}", ha="left", va="center", fontsize=8.45, color="#1e293b", zorder=3)

    # =========================================================================
    # PANEL B: Radial Basis & Saturating Link Functions (Cutoff-Free)
    # =========================================================================
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.set_title(
        "b   Cutoff-Free Radial Basis Functions on Gap $g_{ij} = d_{ij} - (r_i + r_j)$",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    g = np.linspace(0.0, 3.05, 600)
    u = np.clip((2.8 - g) / 2.8, 0.0, 1.0)
    b1 = hat_b1(u)
    b2 = hat_b2(u)
    b12 = hat_b12(u)
    ghb = hat_hb(u)
    gwb = hat_wb(u)

    ax_b.axvspan(0.0, 0.7, color="#fee2e2", alpha=0.40, label=r"Inner contact core ($0 \leq g \leq 0.7$ Å)")
    ax_b.axvspan(0.7, 2.8, color="#ccfbf1", alpha=0.35, label="Hydration / water-bridge collar ($0.7 < g < 2.8$ Å)")

    ax_b.plot(g, b1, color="#dc2626", linewidth=2.6, label=r"Core hat $\hat{B}_1(u) = \max(0, 4u-3)$")
    ax_b.plot(g, b2, color="#0d9488", linewidth=2.6, label=r"Collar hat $\hat{B}_2(u) = \min(\frac{4}{3}u, 4(1-u))$")
    ax_b.plot(g, ghb, color="#7c3aed", linewidth=2.2, linestyle="--", label=r"Direct H-bond $\hat{w}_{\mathrm{hb}}(u) = \hat{B}_1 + \frac{1}{4}\hat{B}_2$")
    ax_b.plot(g, gwb, color="#0284c7", linewidth=2.2, linestyle="-.", label=r"Water bridge $\hat{w}_{\mathrm{wb}}(u) = \hat{B}_2(1 - \hat{B}_1)$")
    ax_b.plot(g, b12, color="#475569", linewidth=1.8, linestyle=":", label=r"Cosine support $\psi(u) = \hat{B}_{12}(u) = \min(\frac{4}{3}u, 1)$")

    for gx, lbl_k, ha_k, dx_k in [
        (0.0, "vdW contact\n$g=0$ Å", "left", 0.04),
        (0.7, "Solvent knot\n$g=0.7$ Å", "left", 0.04),
        (2.8, "Cutoff\n$g=2.8$ Å", "right", -0.05),
    ]:
        ax_b.axvline(gx, color="#475569", linestyle="--", linewidth=0.9, alpha=0.7)
        ax_b.text(gx + dx_k, 1.13, lbl_k, fontsize=7.8, color="#334155", va="top", ha=ha_k,
                  bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))

    ax_b.set_xlim(0.0, 3.05)
    ax_b.set_ylim(-0.04, 1.18)
    ax_b.set_xlabel("Atom-Atom Surface Gap $g_{ij} = d_{ij} - (r_i + r_j)$ (Å)", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_b.set_ylabel("Kernel Amplitude", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_b.legend(loc="center right", bbox_to_anchor=(0.98, 0.56), fontsize=8.0, frameon=True, facecolor="white", framealpha=0.94)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL C: Exact 2-Body Energy Decomposition Across Native & Slid Poses
    # =========================================================================
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_c.set_title(
        "c   Exact 2-Body Energy Decomposition on Native & Slid Poses ($k_BT$)",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    pose_keys = ["nat_m1", "nat_m6", "nat_m2", "nat_m4", "nat_m0", "m2_slid #1", "m4_slid #0", "m2_slid #2"]
    pose_labels = [
        "nat m1\n($4_3$ screw)",
        "nat m6\n($C_2$ dimer)",
        "nat m2\n($4_3+c$)",
        "nat m4\n($C_2$ lat.)",
        "nat m0\n($c$-trans.)",
        "m2_slid #1\n(nuc-53)",
        "m4_slid #0\n(nuc-54)",
        "m2_slid #2\n(nuc-54)",
    ]
    x_c = np.arange(len(pose_keys))
    e_hb = np.array([decomps[k]["e_hb"] for k in pose_keys])
    e_wb = np.array([decomps[k]["e_wb"] for k in pose_keys])
    e_des = np.array([decomps[k]["e_desolv"] for k in pose_keys])
    e_cos = np.array([decomps[k]["e_cos"] for k in pose_keys])
    e_tot = np.array([decomps[k]["e_tot"] for k in pose_keys])

    w_c = 0.56
    neg_base = np.zeros_like(e_hb)
    pos_base = np.zeros_like(e_hb)

    components_c = [
        (e_hb, "#2563eb", "Direct H-bond + salt bridge ($-w_{\\mathrm{hb}}S_{\\mathrm{hb}} - w_{\\mathrm{chb}}S_{\\mathrm{chb}}$)"),
        (e_wb, "#0d9488", "Per-atom saturated water bridge ($-w_{\\mathrm{wb}}S_{\\mathrm{wb}}$)"),
        (e_des, "#f59e0b", "Unpaired burial + like-polar clash ($+E_{\\mathrm{desolv}} + E_{\\mathrm{clash}}$)"),
        (e_cos, "#7c3aed", "Contrastive cosine complementarity ($-w_{\\cos}^{\\pm}S_{\\cos}$)"),
    ]
    for vals, col, lbl in components_c:
        bot = np.where(vals < 0, neg_base, pos_base)
        ax_c.bar(x_c, vals, w_c, bottom=bot, color=col, edgecolor="#1e293b", linewidth=0.6, alpha=0.90, label=lbl, zorder=3)
        neg_base = np.where(vals < 0, neg_base + vals, neg_base)
        pos_base = np.where(vals >= 0, pos_base + vals, pos_base)

    ax_c.plot(x_c, e_tot, "o-", color="#0f172a", linewidth=2.0, markersize=6.5, label="Net 2-body energy $E_{2\\mathrm{B}}^{\\mathrm{tot}}$", zorder=5)
    for xi, et, nb, pb in zip(x_c, e_tot, neg_base, pos_base):
        y_lbl = (nb - 1.55) if et < 0 else (pb + 1.45)
        ax_c.text(
            xi, y_lbl, f"{et:+.1f}",
            ha="center", va="center", fontsize=8.1, fontweight="bold",
            color="#0f172a" if et < 0 else "#b91c1c",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=6
        )

    ax_c.axhline(0.0, color="#334155", linewidth=1.1, zorder=2)
    ax_c.axvline(4.5, color="#dc2626", linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)
    ax_c.text(2.0, 24.8, "Native Crystal Contacts", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#065f46")
    ax_c.text(5.85, 24.8, "Slid Decoy Polymorphs", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#991b1b")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(pose_labels, fontsize=8.0, color="#1e293b")
    ax_c.set_ylim(-29.5, 27.2)
    ax_c.set_ylabel("2-Body Contact Energy ($k_BT$)", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.01, 0.90), fontsize=7.4, frameon=True, facecolor="white", framealpha=0.94)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL D: Basin Free Energies ln K (2-Body & Multi-Body Growth Sites)
    # =========================================================================
    ax_d = fig.add_subplot(gs_bot[0, 1])
    ax_d.set_title(
        "d   Basin Association Constants $\\ln(K / \\mathrm{\\AA}^3)$ from `pair-binding`",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    site_names = [
        "Slid decoy m2 #2 (2B)",
        "Slid decoy m2 #1 (2B)",
        "Slid decoy m4 #0 (2B)",
        "Native m0 $c$-trans. (2B)",
        "Native m4 $C_2$ lat. (2B)",
        "Native m2 $4_3+c$ (2B)",
        "Native m6 $C_2$ dimer (2B)",
        "Native m1 $4_3$ screw (2B)",
        "1D Helix Tip (m1+m2+m0)",
        "3D 4-Cycle Corner (m2+m4)",
        "3D Lateral Turn (m1+m2+m4)",
    ]
    lnk_vals = [-15.13, -3.39, -1.38, +0.61, +2.08, +3.07, +3.41, +5.06, +14.34, +14.74, +31.01]
    colors_d = [
        "#ef4444", "#ef4444", "#ef4444",
        "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af",
        "#10b981", "#059669", "#047857",
    ]
    y_d = np.arange(len(site_names))
    ax_d.axvspan(13.29, 35.5, color="#ecfdf5", alpha=0.65, zorder=0)
    ax_d.axvline(0.0, color="#64748b", linewidth=1.0, zorder=1)
    ax_d.axvline(13.29, color="#059669", linestyle="--", linewidth=1.8, zorder=2)
    ax_d.text(
        13.8, 2.2,
        "Supercritical growth threshold\n" + r"$\ln K_{\mathrm{crit}} = -\ln \rho_0 = +13.29$" + "\n(at $c_0 = 2800\\;\\mu$M)",
        fontsize=8.2, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#059669", alpha=0.95),
        zorder=5
    )

    bars_d = ax_d.barh(y_d, lnk_vals, height=0.66, color=colors_d, edgecolor="#0f172a", linewidth=0.8, zorder=3)
    for bar, v in zip(bars_d, lnk_vals):
        yc = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            ax_d.text(v + 0.55, yc, f"{v:+.2f}", va="center", ha="left", fontsize=8.2, fontweight="bold", color="#0f172a")
        else:
            ax_d.text(v - 0.55, yc, f"{v:+.2f}", va="center", ha="right", fontsize=8.2, fontweight="bold", color="#991b1b")

    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(site_names, fontsize=8.4, fontweight="bold", color="#1e293b")
    ax_d.set_xlim(-19.5, 35.5)
    ax_d.set_xlabel(r"Basin Association Constant $\ln(K / \mathrm{\AA}^3)$", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    return fig


def make_figure_2_training(decomps, S_mat, pos_mask, s_pos_pairs, s_neg_pairs, neg_scores):
    """Figure 2: Interface-Agnostic Training Procedure (Datasets, Losses, Link Function, Nucleation)."""
    fig = plt.figure(figsize=(16.6, 11.6), dpi=300, facecolor="white")
    gs_top = fig.add_gridspec(
        1, 2, left=0.04, right=0.975, bottom=0.525, top=0.880, wspace=0.18, width_ratios=[1.10, 0.98]
    )
    gs_bot = fig.add_gridspec(
        1, 2, left=0.04, right=0.975, bottom=0.065, top=0.440, wspace=0.22, width_ratios=[1.02, 1.06]
    )

    fig.text(
        0.5, 0.962,
        "Interface-Agnostic Contrastive & Energy-Balance Training Procedure for Monomer Crystallization",
        ha="center", va="center", fontsize=15.2, fontweight="bold", color="#0f172a"
    )
    fig.text(
        0.5, 0.928,
        r"Self-supervised $376\times 376$ polar partition function + local slide/twist perturbations + universal SE(3) involution vs. open-generator link function",
        ha="center", va="center", fontsize=10.8, color="#334155"
    )

    # =========================================================================
    # PANEL A: Datasets & Two-Stage Optimization Pipeline Flowchart
    # =========================================================================
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_xlim(0, 10.6)
    ax_a.set_ylim(0, 7.5)
    ax_a.axis("off")
    ax_a.text(
        0.0, 7.25,
        "a   Training Datasets, Monomer Feature Graph & Two-Stage Loss Functions",
        fontsize=12.0, fontweight="bold", color="#0f172a", ha="left", va="center"
    )

    # Left column: Datasets
    ds_boxes = [
        (
            (0.06, 5.05), (4.58, 1.88), "#f8fafc", "#475569", "#0f172a",
            "Dataset 1: PDB Corpus & Monomer Graph",
            [
                "• 20 PDB Structures: 125 symmetry interfaces (H2O, Cl-)",
                "• Monomer (193L): N = 1001 heavy atoms, P = 376 polar",
                r"• 91D 3D Env. Descriptors $\mathbf{f}_i \to$ Laplacian $L_{\mathrm{chem}}$",
            ]
        ),
        (
            (0.06, 2.65), (4.58, 2.15), "#f0fdf4", "#16a34a", "#14532d",
            "Dataset 2: Positive Crystal Contacts (8 Poses)",
            [
                r"• 7 Unique $P4_32_12$ Lattice Modes ($\mathrm{m0\dots m6}$) + capture",
                r"• Primary Partner Filter: $u_{ij} \geq 0.25$ or $\geq 0.70\,u_{i,\max}$",
                r"• Support-Balanced Weight: $w_{\mathrm{norm}}(m) = 200 / \max(\Psi_m, 1)$",
                "• Universal support weighting (no hand-tuned weights)",
            ]
        ),
        (
            (0.06, 0.15), (4.58, 2.25), "#fff7ed", "#ea580c", "#9a3412",
            "Dataset 3: Perturbations & Held-Out Decoys",
            [
                r"• 225 Local Slide/Twist Poses around each crystal $x_m$:",
                r"   $\delta t_\parallel \in \{4.5,\,7.0\}$ Å, $\delta\theta \in \{-24^\circ,0^\circ,+24^\circ\}$",
                "• 61 Held-Out MC Decoys: never seen in training,",
                "   used strictly for out-of-sample validation (61/61 < 0)",
            ]
        ),
    ]
    for (x0, y0), (w, h), fc, ec, tc, title, lines in ds_boxes:
        b = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.14", facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=1)
        ax_a.add_patch(b)
        ax_a.text(x0 + 0.15, y0 + h - 0.26, title, ha="left", va="center", fontsize=8.8, fontweight="bold", color=tc, zorder=3)
        for idx, l in enumerate(lines):
            ax_a.text(x0 + 0.16, y0 + h - 0.64 - idx * 0.39, l, ha="left", va="center", fontsize=7.65, color="#1e293b", zorder=3)

    # Right column: Stage A and Stage B Losses
    stage_boxes = [
        (
            (5.05, 3.70), (5.45, 3.23), "#eff6ff", "#2563eb", "#1e3a8a",
            r"Stage A: Contrastive Embedding Loss ($\mathbf{u}_i,\mathbf{v}_i \in S^7$)",
            [
                r"$s_{ij} = \frac{1}{2}(\mathbf{u}_i\cdot\mathbf{v}_j + \mathbf{v}_i\cdot\mathbf{u}_j)$ on $P\times P = 376\times 376$ matrix",
                r"$\mathcal{L}_{\mathrm{Stage\;A}}(\mathbf{U},\mathbf{V}) = -\sum_{i,j} W_{ij}^{\mathrm{pos}}\ln\frac{\exp(s_{ij}/\tau)}{\sum_k \pi_{ik}\exp(s_{ik}/\tau)}$",
                r"$\quad + \frac{1}{2}\sum_{i,k\notin\mathcal{P}} W_{ik}^{\mathrm{neg}}[s_{ik} + 0.52]_+^2 + 5\sum_{i,j\in\mathcal{P}} W_{ij}^{\mathrm{pos}}[0.85 - s_{ij}]_+^2$",
                r"$\quad + \lambda_{\mathrm{lap}}\,\mathrm{Tr}(\mathbf{U}^\top L_{\mathrm{chem}}\mathbf{U} + \mathbf{V}^\top L_{\mathrm{chem}}\mathbf{V})\;\;\;(\tau = 0.15)$",
                r"Prior: $W_{ik}^{\mathrm{neg}} = (16\,I_{\mathrm{act}} + 1.2\,I_{\mathrm{bg}} + W_{ik}^{\mathrm{pert}})\,I_{(i,k)\notin\mathcal{P}}$",
            ]
        ),
        (
            (5.05, 0.15), (5.45, 3.25), "#faf5ff", "#9333ea", "#581c87",
            r"Stage B: Energy-Balance Activity Loss ($a_i \in [0.20, 1.65]$)",
            [
                r"Universal Target Link: $S_{\cos}^*(m) = \max\!\left(0.55,\;\frac{E_{\mathrm{pol}}^{(0)}(m) - E_{2\mathrm{B}}^*(m)}{w_{\cos}^+}\right)$",
                r"$E_{2\mathrm{B}}^*(m) = E_{\mathrm{asymp}}(\mathrm{is\_inv}(m))\,\frac{\Psi_m}{\Psi_m + 4.2}$",
                r"$E_{\mathrm{asymp}} = -18.2\;k_BT$ (open screw) vs. $-15.5\;k_BT$ ($C_2$ involution)",
                r"$\mathcal{L}_{\mathrm{Stage\;B}}(\mathbf{a}) = w_{\mathrm{pose}}\sum_{m=1}^{8}\mathrm{band}(S_{\cos}(x_m;\mathbf{a}),\,S_{\cos}^*(m))$",
                r"$\quad + w_{\mathrm{pert}}\sum_{p\in\mathcal{D}_{\mathrm{pert}}}[S_{\cos}(x_p;\mathbf{a}) + 0.35]_+^2 + \lambda_a\|\mathbf{a} - a_{\mathrm{bg}}\|^2$",
            ]
        ),
    ]
    for (x0, y0), (w, h), fc, ec, tc, title, lines in stage_boxes:
        b = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.14", facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=1)
        ax_a.add_patch(b)
        ax_a.text(x0 + 0.16, y0 + h - 0.28, title, ha="left", va="center", fontsize=8.9, fontweight="bold", color=tc, zorder=3)
        for idx, l in enumerate(lines):
            ax_a.text(x0 + 0.18, y0 + h - 0.72 - idx * 0.53, l, ha="left", va="center", fontsize=7.9, color="#1e293b", zorder=3)

    # Arrows connecting datasets to stages
    for y_arr in [5.6, 4.2]:
        arr = FancyArrowPatch((4.64, y_arr), (5.02, y_arr), arrowstyle="-|>,head_length=5,head_width=3", color="#2563eb", linewidth=1.6, zorder=4)
        ax_a.add_patch(arr)
    for y_arr in [3.1, 1.4]:
        arr = FancyArrowPatch((4.64, y_arr), (5.02, y_arr), arrowstyle="-|>,head_length=5,head_width=3", color="#9333ea", linewidth=1.6, zorder=4)
        ax_a.add_patch(arr)

    # =========================================================================
    # PANEL B: Universal Stage B Energy-Balance Link Function Curve
    # =========================================================================
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.set_title(
        "b   Universal Stage B Energy-Balance Link Function $E_{2\\mathrm{B}}^*(\\Psi_m)$",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    psi_grid = np.linspace(0.0, 65.0, 400)
    e_curve_open = -18.2 * (psi_grid / (psi_grid + 4.2))
    e_curve_inv = -15.5 * (psi_grid / (psi_grid + 4.2))

    ax_b.plot(psi_grid, e_curve_open, color="#15803d", linewidth=2.5, label=r"Open generator curve ($R^2 \neq I$): $E_{\mathrm{asymp}} = -18.2\,k_BT$")
    ax_b.plot(psi_grid, e_curve_inv, color="#2563eb", linewidth=2.5, linestyle="--", label=r"Closed $C_2$ involution ($R^2 = I,\,Rt+t=0$): $E_{\mathrm{asymp}} = -15.5\,k_BT$")

    label_offsets = {
        "nat_m0": (1.4, -0.45, "left", "m0 (-4.0)"),
        "nat_m1": (1.4, -0.95, "left", "m1 (-15.5)"),
        "nat_m2": (1.4, -0.75, "left", "m2 (-11.8)"),
        "nat_m3": (1.4, 0.90, "left", "m3 (+0.1)"),
        "nat_m4": (-1.4, 0.85, "right", "m4 (-13.2)"),
        "nat_m5": (1.4, 0.70, "left", "m5 (-2.7)"),
        "nat_m6": (-1.5, -1.25, "right", "m6 (-15.2)"),
    }
    for k, (dx, dy, ha, lbl) in label_offsets.items():
        d = decomps[k]
        psi_m = d["psi_sum"]
        e0 = d["e_pol0"]
        etot = d["e_tot"]
        col = "#2563eb" if d["is_inv"] else "#15803d"
        # Plot baseline E_pol^(0) as open square and final E_tot as filled circle connected by vertical arrow
        ax_b.plot([psi_m], [e0], "s", color="#94a3b8", markersize=7.0, markeredgecolor="#334155", zorder=4)
        ax_b.plot([psi_m], [etot], "o", color=col, markersize=8.5, markeredgecolor="#0f172a", zorder=6)
        if abs(etot - e0) > 1.2:
            ax_b.annotate(
                "", xy=(psi_m, etot + 0.35), xytext=(psi_m, e0 - 0.35),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.5, ls="-"),
                zorder=5
            )
        ax_b.text(psi_m + dx, etot + dy, lbl, fontsize=7.9, fontweight="bold", color=col, ha=ha, va="center", zorder=7)

    # Dummy legend entries for baseline vs learned
    ax_b.plot([], [], "s", color="#94a3b8", markeredgecolor="#334155", label=r"Baseline polar energy $E_{\mathrm{pol}}^{(0)}(m)$ (before cosine)")
    ax_b.plot([], [], "o", color="#0f172a", label=r"Calibrated 2-body energy $E_{\mathrm{pol}}^{(0)}(m) - w_{\cos}^+ S_{\cos}(m)$")

    ax_b.axhline(0.0, color="#64748b", linewidth=0.9, linestyle=":")
    ax_b.set_xlim(-1.0, 65.0)
    ax_b.set_ylim(-19.5, 4.0)
    ax_b.set_xlabel(r"Polar Contact Support $\Psi_m = \sum_{(i,j)\in m} \psi(u_{ij})$", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_b.set_ylabel("2-Body Contact Energy ($k_BT$)", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_b.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.94)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL C: Contrastive Pair & Held-Out Pose Discrimination
    # =========================================================================
    gs_c = gs_bot[0, 0].subgridspec(2, 1, hspace=0.45, height_ratios=[1.0, 1.05])
    ax_c1 = fig.add_subplot(gs_c[0])
    ax_c2 = fig.add_subplot(gs_c[1])

    ax_c1.set_title(
        "c   Stage A Atom-Pair Partition Separation & Held-Out Pose Scores",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    bins_s = np.linspace(-1.0, 1.0, 45)
    ax_c1.hist(s_neg_pairs, bins=bins_s, density=True, color="#94a3b8", alpha=0.65, edgecolor="#475569", linewidth=0.6, label=f"Off-diagonal non-partner pairs ($N={len(s_neg_pairs):,}$, mean {s_neg_pairs.mean():+.2f})")
    ax_c1.hist(s_pos_pairs, bins=bins_s, density=True, color="#10b981", alpha=0.75, edgecolor="#047857", linewidth=0.8, label=f"Primary crystal contact pairs ($N={len(s_pos_pairs)}$, max {s_pos_pairs.max():+.2f})")
    ax_c1.axvline(-0.52, color="#dc2626", linestyle="--", linewidth=1.3, label="Negative hinge $(-0.52)$")
    ax_c1.axvline(+0.85, color="#059669", linestyle="--", linewidth=1.3, label="Positive hinge $(+0.85)$")
    ax_c1.set_xlim(-1.02, 1.02)
    ax_c1.set_xlabel(r"Atom-Atom Cosine Similarity $s_{ij} = \frac{1}{2}(\mathbf{u}_i\cdot\mathbf{v}_j + \mathbf{v}_i\cdot\mathbf{u}_j)$", fontsize=8.8, fontweight="bold", color="#1e293b")
    ax_c1.set_ylabel("Density", fontsize=8.8, fontweight="bold", color="#1e293b")
    ax_c1.legend(loc="upper right", fontsize=7.3, frameon=True, facecolor="white", ncol=1)
    ax_c1.spines["top"].set_visible(False)
    ax_c1.spines["right"].set_visible(False)

    # Bottom subplot: Pose-level cosine scores S_cos across Native (7/7 > 0), Held-Out Decoys (61/61 < 0), Slid Decoys (3/3 < 0)
    nat_s = [decomps[f"nat_m{m}"]["S_cos"] for m in [1, 2, 4, 6, 0, 5, 3]]
    slid_s = [decomps[k]["S_cos"] for k in ["m2_slid #1", "m4_slid #0", "m2_slid #2"]]
    neg_clipped = np.clip(neg_scores, -4.8, 3.6)

    bins_p = np.linspace(-5.0, 3.6, 40)
    ax_c2.hist(neg_clipped, bins=bins_p, color="#fca5a5", edgecolor="#b91c1c", linewidth=0.8, alpha=0.85, label="Held-out non-native MC decoys (61/61 $< 0$, never in training)")
    ax_c2.hist(slid_s, bins=bins_p, color="#dc2626", edgecolor="#7f1d1d", linewidth=1.0, alpha=0.95, label="Trajectory slid polymorphs m2_slid & m4_slid (3/3 $< 0$)")
    ax_c2.hist(nat_s, bins=bins_p, color="#10b981", edgecolor="#065f46", linewidth=1.0, alpha=0.90, label="Native crystal contacts m0..m6 (7/7 $> 0$)")
    ax_c2.axvline(0.0, color="#0f172a", linewidth=1.4, linestyle="--")
    ax_c2.set_xlim(-5.1, 3.6)
    ax_c2.set_ylim(0, 26)
    ax_c2.set_xlabel(r"Pose Cosine Complementarity Score $S_{\cos}(A,B)$ (clipped at $-4.8$)", fontsize=8.8, fontweight="bold", color="#1e293b")
    ax_c2.set_ylabel("Pose Count", fontsize=8.8, fontweight="bold", color="#1e293b")
    ax_c2.legend(loc="upper right", fontsize=7.4, frameon=True, facecolor="white")
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL D: Spontaneous Unseeded 3D Nucleation in nuc-55 (256 Monomers)
    # =========================================================================
    ax_d = fig.add_subplot(gs_bot[0, 1])
    ax_d.set_title(
        "d   Unseeded 256-Monomer 3D $P4_32_12$ Nucleation (`nuc-55`, $10^6$ Sweeps)",
        loc="left", fontsize=12.0, fontweight="bold", color="#0f172a", pad=10
    )

    sweeps_k = np.array([0, 200, 400, 600, 800, 1000], dtype=float)
    proto_contacts = np.array([0, 32, 47, 137, 233, 373], dtype=float)  # m0 + m1 + m2
    lateral_contacts = np.array([0, 0, 0, 0, 9, 176], dtype=float)       # m3 + m4 + m5 + m6
    nucleus_size = np.array([1, 13, 18, 34, 51, 100], dtype=float)

    ax_d.fill_between(sweeps_k, 0, proto_contacts, color="#bbf7d0", alpha=0.55)
    ax_d.plot(sweeps_k, proto_contacts, "o-", color="#15803d", linewidth=2.5, markersize=6.5, label=r"1D $4_3$ helix contacts ($\mathrm{m0}+\mathrm{m1}+\mathrm{m2} = 373$)")
    ax_d.fill_between(sweeps_k, 0, lateral_contacts, color="#bfdbfe", alpha=0.65)
    ax_d.plot(sweeps_k, lateral_contacts, "s-", color="#1d4ed8", linewidth=2.5, markersize=6.5, label=r"3D lateral locking contacts ($\mathrm{m3}+\mathrm{m4}+\mathrm{m5}+\mathrm{m6} = 176$)")

    ax_d2 = ax_d.twinx()
    ax_d2.plot(sweeps_k, nucleus_size, "^--", color="#7c3aed", linewidth=2.3, markersize=7.0, label="Largest single crystal nucleus ($N_{\\mathrm{lattice}} = 100$, #2 $= 79$)")
    ax_d2.set_ylabel("Monomers in Largest Native Crystal", fontsize=9.5, fontweight="bold", color="#6d28d9")
    ax_d2.set_ylim(0, 125)
    ax_d2.spines["top"].set_visible(False)

    ax_d.annotate(
        "Stage 1: $4_3$ protofilament\nassembly (m0, m1, m2)",
        xy=(450, 80), xytext=(180, 210),
        fontsize=8.3, fontweight="bold", color="#14532d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor="#16a34a"),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.4),
    )
    ax_d.annotate(
        "Stage 2: 3D lateral locking\nAll 7 crystal modes active!\n(549 native vs. 141 non-native)",
        xy=(960, 176), xytext=(520, 310),
        fontsize=8.3, fontweight="bold", color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#2563eb"),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5),
    )

    ax_d.set_xlim(0, 1030)
    ax_d.set_ylim(0, 420)
    ax_d.set_xlabel(r"Monte Carlo Sweeps ($\times 10^3$)", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_d.set_ylabel("Native Crystal Contacts in Vessel", fontsize=9.8, fontweight="bold", color="#1e293b")

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.95)
    ax_d.spines["top"].set_visible(False)

    return fig


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    decomps, S_mat, pos_mask, s_pos_pairs, s_neg_pairs, neg_scores = compute_model_diagnostics()
    fig1 = make_figure_1_energy(decomps)
    fig2 = make_figure_2_training(decomps, S_mat, pos_mask, s_pos_pairs, s_neg_pairs, neg_scores)

    out_dirs = [
        "/home/xvg/tetramer-mc/docs/figures",
        "/home/xvg/tetramer-mc/.claude/worktrees/oligomer-capture/docs/figures",
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig1.savefig(os.path.join(d, f"energy_model_shape_solvation_chemistry.{ext}"), dpi=300)
            fig2.savefig(os.path.join(d, f"chemistry_model_training_procedure.{ext}"), dpi=300)

    viewer_dir = "/vast/xvg/tetramer-mc-runs/morph-growth-20261012/viewer"
    if os.path.exists(viewer_dir):
        shutil.copyfile(
            "/home/xvg/tetramer-mc/docs/figures/energy_model_shape_solvation_chemistry.png",
            os.path.join(viewer_dir, "energy_model_shape_solvation_chemistry.png"),
        )
        shutil.copyfile(
            "/home/xvg/tetramer-mc/docs/figures/chemistry_model_training_procedure.png",
            os.path.join(viewer_dir, "chemistry_model_training_procedure.png"),
        )
    print("Generated both publication figures in docs/figures/ (.png, .pdf, .svg)")


if __name__ == "__main__":
    main()
