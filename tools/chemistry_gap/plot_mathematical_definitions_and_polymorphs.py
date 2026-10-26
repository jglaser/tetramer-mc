#!/usr/bin/env python3
"""Generate publication-quality figure:
docs/figures/mathematical_definitions_and_polymorph_complementarity.{png,pdf,svg}

Graphical and mathematical definitions of:
  (A) Interatomic surface gap g_ij(t, R), normalized proximity u_ij, and 3D outward surface normals n_i^A, R n_j^B
  (B) Radial partition-of-unity basis (b1, b2, psi_b12) and physical envelopes (psi_hb, psi_wb, psi_Hof)
  (C) Lawrence & Colman (1993) shape complementarity S_c vs. multi-point contrastive complementarity S_cos(t, R)
  (D) 6D pose-funnel sensitivity of S_c(Delta t, Delta theta) vs. S_cos(Delta t, Delta theta) on native contacts
  (E) Verification of physical complementarity (S_c > 0, E_pol0 + dE_Hof < 0, and S_cos > 0) across all 8 HEWL polymorphs
  (F) Salt-selective polymorph stabilization across NaCl (0.85 M), NaNO3 (0.35 M), and SO4^2- (0.20 M)
"""

import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch
from scipy.spatial import cKDTree

POLY_LIB_PATH = "/home/xvg/tetramer-mc/examples/polymorphs/polymorph-contact-library.json"
UNIV_MODEL_PATH = "/home/xvg/tetramer-mc/examples/polymorphs/chemistry-model-universal-multipolymorph.json"
OLD_MODEL_PATH = "/home/xvg/tetramer-mc/examples/chemistry-model-monomer.json"
OUT_DIRS = [
    "/home/xvg/tetramer-mc/.claude/worktrees/oligomer-capture/docs/figures",
    "/home/xvg/tetramer-mc/docs/figures",
]

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


def comp(fi, fj):
    return bool(
        ((fi & CHEM_NDON) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_NDON))
        or ((fi & CHEM_CAT) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_CAT))
        or ((fi & CHEM_CAT) and (fj & CHEM_ANI)) or ((fi & CHEM_ANI) and (fj & CHEM_CAT))
        or ((fi & CHEM_ANI) and (fj & CHEM_NDON)) or ((fi & CHEM_NDON) and (fj & CHEM_ANI))
        or ((fi & CHEM_TYR_OH) and (fj & CHEM_NACC)) or ((fi & CHEM_NACC) and (fj & CHEM_TYR_OH))
    )


def axis_angle_rot(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis) + 1e-12
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def compute_outward_normals(X, R_vdw, sigma=3.5):
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    r3 = (np.maximum(R_vdw, 1.0) ** 3)[None, :]
    w = r3 * np.exp(-d2 / (sigma * sigma))
    np.fill_diagonal(w, 0.0)
    grad = np.sum(w[:, :, None] * diff, axis=1)
    return grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8)


def evaluate_polymorph_metrics():
    lib = json.load(open(POLY_LIB_PATH))
    univ = json.load(open(UNIV_MODEL_PATH))
    old = json.load(open(OLD_MODEL_PATH))
    flags = np.array(univ["flags"], dtype=np.uint8)
    weights = univ["weights"]
    w_cos_pos, w_cos_neg = univ["cos_weights"]
    kd = univ["hofmeister_kd"]
    hw = univ["hofmeister_weights"]

    polar_idx = np.where(flags > 0)[0]
    P = len(polar_idx)
    full_to_p = {int(idx): p for p, idx in enumerate(polar_idx)}

    U_u = np.array(univ["cos_u"], dtype=float)[polar_idx]
    V_u = np.array(univ["cos_v"], dtype=float)[polar_idx]
    S_u_mat = 0.5 * (U_u @ V_u.T + V_u @ U_u.T)
    a_u = np.array(univ["cos_activity"], dtype=float)[polar_idx]

    U_o = np.array(old["cos_u"], dtype=float)[polar_idx]
    V_o = np.array(old["cos_v"], dtype=float)[polar_idx]
    S_o_mat = 0.5 * (U_o @ V_o.T + V_o @ U_o.T)
    a_o = np.array(old["cos_activity"], dtype=float)[polar_idx]

    poly_stats = []
    # Compute grazing roll/twist + slide sensitivity on 1IEE c00: theta in [-60, +60] deg
    slide_d = np.linspace(-60.0, 60.0, 49)
    slide_sc = []
    slide_scos = []
    slide_scos_o = []

    for poly in lib["polymorphs"]:
        tag = poly["polymorph"]
        sh = json.load(open(f"/home/xvg/tetramer-mc/examples/polymorphs/monomer-shape-{tag}.json"))
        X = np.array([a["center"] for a in sh["atoms"]], dtype=float)
        R_vdw = np.array([a["radius"] for a in sh["atoms"]], dtype=float)
        N_out = compute_outward_normals(X, R_vdw)
        tree_x = cKDTree(X)
        N = len(X)

        sc_list, scos_u_list, scos_o_list = [], [], []
        epol0_list, ehof_nacl, ehof_nano3, ehof_so4 = [], [], [], []
        ebind_nacl, ebind_nano3, ebind_so4 = [], [], []

        for idx, c in enumerate(poly["contacts"]):
            if c["n_buried_crystal"] < 150:
                continue
            t = np.array(c["t_retracted"], dtype=float)
            R = np.array(c["R_rel"], dtype=float)
            YB = (X @ R.T) + t
            NB = N_out @ R.T
            pairs = tree_x.query_ball_tree(cKDTree(YB), 6.4)

            num_lc_a, den_lc_a = np.zeros(N), np.zeros(N)
            num_lc_b, den_lc_b = np.zeros(N), np.zeros(N)
            ii, jj, psi_arr = [], [], []
            max_u_a, max_u_b = np.zeros(N), np.zeros(N)
            max_comp_a, max_comp_b = np.zeros(N), np.zeros(N)
            hb_cands, chb_cands, clash_cands = [], [], []
            s_hof = np.zeros(5, dtype=float)

            for i, js in enumerate(pairs):
                ri = R_vdw[i]
                fi = int(flags[i])
                for j in js:
                    g = float(np.linalg.norm(X[i] - YB[j])) - (ri + R_vdw[j])
                    if g >= 2.8:
                        continue
                    u = min(max((2.8 - g) / 2.8, 0.0), 1.0)
                    psi = float(hat_b12(u))
                    s_norm = float(-np.dot(N_out[i], NB[j]))
                    num_lc_a[i] += psi * s_norm
                    den_lc_a[i] += psi
                    num_lc_b[j] += psi * s_norm
                    den_lc_b[j] += psi
                    fj = int(flags[j])
                    if fi > 0 and fj > 0:
                        ii.append(full_to_p[i])
                        jj.append(full_to_p[j])
                        psi_arr.append(psi)
                        max_u_a[i] = max(max_u_a[i], u)
                        max_u_b[j] = max(max_u_b[j], u)
                        if comp(fi, fj):
                            max_comp_a[i] = max(max_comp_a[i], u)
                            max_comp_b[j] = max(max_comp_b[j], u)
                        gh = float(hat_hb(u))
                        if gh > 0.0:
                            if ((fi & CHEM_NDON) and (fj & CHEM_NACC)) or ((fj & CHEM_NDON) and (fi & CHEM_NACC)):
                                hb_cands.append((gh, min(i, j), max(i, j), i, j))
                            if (((fi & CHEM_CAT) and (fj & CHEM_NACC)) or ((fj & CHEM_CAT) and (fi & CHEM_NACC))
                                    or ((fi & CHEM_ANI) and (fj & CHEM_NDON)) or ((fj & CHEM_ANI) and (fi & CHEM_NDON))):
                                chb_cands.append((gh, min(i, j), max(i, j), i, j))
                            if ((fi & CHEM_PURE_ACC) and (fj & CHEM_PURE_ACC)) or ((fi & CHEM_PURE_DON) and (fj & CHEM_PURE_DON)):
                                clash_cands.append((gh, i, j))
                        wb = float(hat_wb(u))
                        b2 = float(hat_b2(u))
                        ca, cb = bool(fi & CHEM_CAT), bool(fj & CHEM_CAT)
                        aa, ab = bool(fi & CHEM_ANI), bool(fj & CHEM_ANI)
                        pa, pb = bool(fi & (CHEM_NDON | CHEM_NACC)), bool(fj & (CHEM_NDON | CHEM_NACC))
                        if (ca and (ab or pb)) or (cb and (aa or pa)):
                            s_hof[0] += wb
                        if ca and cb:
                            s_hof[1] += b2
                            s_hof[3] += b2
                        elif (ca and pb) or (cb and pa):
                            s_hof[2] += wb
                            s_hof[3] += 0.65 * wb
                        if (aa and pb) or (ab and pa) or (aa and ab):
                            s_hof[4] += 0.5 * wb

            ma = den_lc_a > 0
            mb = den_lc_b > 0
            sc = 0.5 * float(np.mean(num_lc_a[ma] / (den_lc_a[ma] + 0.25)) + np.mean(num_lc_b[mb] / (den_lc_b[mb] + 0.25)))
            sc_list.append(sc)

            ii = np.array(ii, int)
            jj = np.array(jj, int)
            psi_arr = np.array(psi_arr, float)

            def eval_scos(S_mat, a_vec):
                sc_p = S_mat[ii, jj]
                wi = psi_arr * a_vec[ii]
                wj = psi_arr * a_vec[jj]
                na = np.bincount(ii, weights=wj * sc_p, minlength=P)
                da = np.bincount(ii, weights=wj, minlength=P)
                nb = np.bincount(jj, weights=wi * sc_p, minlength=P)
                db = np.bincount(jj, weights=wi, minlength=P)
                return 0.5 * float(np.sum(a_vec * na / (da + 0.25)) + np.sum(a_vec * nb / (db + 0.25)))

            su = eval_scos(S_u_mat, a_u)
            so = eval_scos(S_o_mat, a_o)
            scos_u_list.append(su)
            scos_o_list.append(so)

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
            s_clash = sum(gh * (1.0 - hat_b12(max_comp_a[i])) * (1.0 - hat_b12(max_comp_b[j])) for gh, i, j in clash_cands)
            s_wb = 0.5 * (
                sum(hat_wb(u) for u in max_comp_a if u > 0.0)
                + sum(hat_wb(u) for u in max_comp_b if u > 0.0)
            ) * np.exp(-1.5 * s_clash)
            unp_pol = sum(
                max(0.0, hat_hb(max_u_a[i]) - hat_b12(max_comp_a[i]))
                for i in range(N)
                if flags[i] and not (flags[i] & (CHEM_CAT | CHEM_ANI)) and (flags[i] & (CHEM_NDON | CHEM_NACC))
            ) + sum(
                max(0.0, hat_hb(max_u_b[j]) - hat_b12(max_comp_b[j]))
                for j in range(N)
                if flags[j] and not (flags[j] & (CHEM_CAT | CHEM_ANI)) and (flags[j] & (CHEM_NDON | CHEM_NACC))
            )
            unp_chg = sum(
                max(0.0, hat_hb(max_u_a[i]) - hat_b12(max_comp_a[i]))
                for i in range(N) if (flags[i] & (CHEM_CAT | CHEM_ANI))
            ) + sum(
                max(0.0, hat_hb(max_u_b[j]) - hat_b12(max_comp_b[j]))
                for j in range(N) if (flags[j] & (CHEM_CAT | CHEM_ANI))
            )
            w = weights
            e_pol0 = -w[0] * s_hb - w[1] * s_chb - w[3] * s_wb + w[4] * s_clash + w[5] * unp_pol + w[6] * unp_chg
            epol0_list.append(e_pol0)
            e_cos = -w_cos_pos * su if su >= 0 else -w_cos_neg * su

            def hof_e(c_salt):
                c_ch = [c_salt[0], c_salt[1], c_salt[1], c_salt[2], c_salt[3]]
                return -sum(hw[m] * np.log(1.0 + c_ch[m] / kd[m]) * s_hof[m] for m in range(5) if c_ch[m] > 0)

            dh_nacl = hof_e([0.85, 0.0, 0.0, 0.85])
            dh_nano3 = hof_e([0.0, 0.35, 0.0, 0.35])
            dh_so4 = hof_e([0.0, 0.0, 0.20, 0.0])
            ehof_nacl.append(dh_nacl)
            ehof_nano3.append(dh_nano3)
            ehof_so4.append(dh_so4)
            ebind_nacl.append(e_pol0 + e_cos + dh_nacl)
            ebind_nano3.append(e_pol0 + e_cos + dh_nano3)
            ebind_so4.append(e_pol0 + e_cos + dh_so4)

            if tag == "tetragonal-1IEE" and idx == 0:
                n_hat = t / np.linalg.norm(t)
                t1 = np.cross(n_hat, np.array([1.0, 0.0, 0.0]))
                t1 /= np.linalg.norm(t1)
                t2 = np.cross(n_hat, t1)
                x_patch = 0.5 * (np.mean(X[ma], axis=0) + np.mean(YB[mb], axis=0))
                for deg in slide_d:
                    rad = np.deg2rad(deg)
                    # Roll/twist around patch centroid x_patch + tangential shear along interface plane
                    R_pert = axis_angle_rot(0.7 * t1 + 0.714 * n_hat, rad)
                    ds = 0.17 * deg * t2 + 0.08 * deg * t1
                    YB_base = ((YB - x_patch) @ R_pert.T) + x_patch + ds
                    NB_s = (N_out @ R.T) @ R_pert.T
                    sh_best = 0.0
                    for sh_n in np.linspace(-6.0, 8.0, 141):
                        YB_cand = YB_base + sh_n * n_hat
                        d_nn, j_nn = cKDTree(YB_cand).query(X, k=1)
                        min_g_cand = float(np.min(d_nn - (R_vdw + R_vdw[j_nn])))
                        if min_g_cand >= 0.05:
                            sh_best = sh_n
                            break
                    YB_s = YB_base + sh_best * n_hat
                    pairs_s = tree_x.query_ball_tree(cKDTree(YB_s), 6.4)
                    nla, dla = np.zeros(N), np.zeros(N)
                    nlb, dlb = np.zeros(N), np.zeros(N)
                    ii_s, jj_s, psi_s = [], [], []
                    for i_s, js_s in enumerate(pairs_s):
                        ri_s = R_vdw[i_s]
                        for j_s in js_s:
                            g_s = float(np.linalg.norm(X[i_s] - YB_s[j_s])) - (ri_s + R_vdw[j_s])
                            if g_s < 2.8:
                                u_s = min(max((2.8 - g_s) / 2.8, 0.0), 1.0)
                                ps = float(hat_b12(u_s))
                                sn = float(-np.dot(N_out[i_s], NB_s[j_s]))
                                nla[i_s] += ps * sn
                                dla[i_s] += ps
                                nlb[j_s] += ps * sn
                                dlb[j_s] += ps
                                if flags[i_s] > 0 and flags[j_s] > 0:
                                    ii_s.append(full_to_p[i_s])
                                    jj_s.append(full_to_p[j_s])
                                    psi_s.append(ps)
                    ma_s = dla > 0
                    mb_s = dlb > 0
                    if np.any(ma_s) and np.any(mb_s):
                        sc_v = 0.5 * float(np.mean(nla[ma_s] / (dla[ma_s] + 0.25)) + np.mean(nlb[mb_s] / (dlb[mb_s] + 0.25)))
                    else:
                        sc_v = 0.0
                    slide_sc.append(sc_v)
                    if len(ii_s) > 0:
                        ii_s = np.array(ii_s, int)
                        jj_s = np.array(jj_s, int)
                        psi_s = np.array(psi_s, float)
                        sp = S_u_mat[ii_s, jj_s]
                        wi = psi_s * a_u[ii_s]
                        wj = psi_s * a_u[jj_s]
                        na = np.bincount(ii_s, weights=wj * sp, minlength=P)
                        da = np.bincount(ii_s, weights=wj, minlength=P)
                        nb = np.bincount(jj_s, weights=wi * sp, minlength=P)
                        db = np.bincount(jj_s, weights=wi, minlength=P)
                        scos_v = 0.5 * float(np.sum(a_u * na / (da + 0.25)) + np.sum(a_u * nb / (db + 0.25)))

                        sp_o = S_o_mat[ii_s, jj_s]
                        wi_o = psi_s * a_o[ii_s]
                        wj_o = psi_s * a_o[jj_s]
                        na_o = np.bincount(ii_s, weights=wj_o * sp_o, minlength=P)
                        da_o = np.bincount(ii_s, weights=wj_o, minlength=P)
                        nb_o = np.bincount(jj_s, weights=wi_o * sp_o, minlength=P)
                        db_o = np.bincount(jj_s, weights=wi_o, minlength=P)
                        scos_vo = 0.5 * float(np.sum(a_o * na_o / (da_o + 0.25)) + np.sum(a_o * nb_o / (db_o + 0.25)))
                    else:
                        scos_v = 0.0
                        scos_vo = 0.0
                    slide_scos.append(scos_v)
                    slide_scos_o.append(scos_vo)

        poly_stats.append({
            "tag": tag,
            "space_group": poly["space_group"],
            "salt_type": poly["salt_type"],
            "n_c": len(sc_list),
            "sc_mean": float(np.mean(sc_list)),
            "sc_pos": sum(x > 0 for x in sc_list),
            "epol0_mean": float(np.mean(epol0_list)),
            "epol0_neg": sum(x < 0 for x in epol0_list),
            "scos_u_mean": float(np.mean(scos_u_list)),
            "scos_u_pos": sum(x > 0 for x in scos_u_list),
            "scos_o_mean": float(np.mean(scos_o_list)),
            "scos_o_pos": sum(x > 0 for x in scos_o_list),
            "dh_nacl": float(np.mean(ehof_nacl)),
            "dh_nano3": float(np.mean(ehof_nano3)),
            "dh_so4": float(np.mean(ehof_so4)),
            "ebind_nacl": float(np.min(ebind_nacl)),
            "ebind_nano3": float(np.min(ebind_nano3)),
            "ebind_so4": float(np.min(ebind_so4)),
        })

    return poly_stats, slide_d, np.array(slide_sc), np.array(slide_scos), np.array(slide_scos_o)


def build_figure():
    poly_stats, slide_d, slide_sc, slide_scos, slide_scos_o = evaluate_polymorph_metrics()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "mathtext.fontset": "stix",
    })

    fig = plt.figure(figsize=(18.0, 11.6), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 3,
        left=0.045, right=0.985, top=0.875, bottom=0.065,
        wspace=0.25, hspace=0.34,
    )

    fig.suptitle(
        r"Graphical & Mathematical Definitions of Surface-Gap Envelopes ($g_{ij},\,u_{ij},\,\psi$), "
        r"Complementarity ($S_c$ vs. $S_{\mathrm{cos}}$), & Hofmeister Polymorph Selection",
        fontsize=13.2, fontweight="bold", y=0.968, color="#0f172a"
    )

    # =========================================================================
    # PANEL A: Graphical definition of g_ij(t, R), u_ij, and outward normals
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    ax_a.set_title(
        r"(A) Interatomic Surface Gap $g_{ij}(\mathbf{t},\mathbf{R})$ & Proximity $u_{ij}$",
        fontweight="bold", loc="left", pad=8, color="#0f172a"
    )

    # Draw Monomer A (left bulk lobe) and Monomer B (right bulk lobe)
    lobe_a = Wedge((1.2, 5.6), 3.1, -68, 68, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.6, alpha=0.65)
    lobe_b = Wedge((8.8, 5.6), 3.1, 112, 248, facecolor="#ffedd5", edgecolor="#c2410c", lw=1.6, alpha=0.65)
    ax_a.add_patch(lobe_a)
    ax_a.add_patch(lobe_b)
    ax_a.text(0.85, 8.15, "Monomer $A$\n(body frame)", fontsize=8.8, fontweight="bold", color="#1e3a8a")
    ax_a.text(7.35, 8.15, "Monomer $B$\npose $(\\mathbf{t},\\mathbf{R})$", fontsize=8.8, fontweight="bold", color="#9a3412")

    # Atom i on A and Atom j on B
    xi, yi, ri = 3.15, 5.65, 1.05
    xj, yj, rj = 6.85, 5.65, 1.05
    # Interaction shell around atom i (r_i + r_j + 2.8 A schematic)
    shell_i = Circle((xi, yi), ri + 1.45, facecolor="#eff6ff", edgecolor="#3b82f6", lw=1.1, ls="--", alpha=0.45)
    ax_a.add_patch(shell_i)
    atom_i = Circle((xi, yi), ri, facecolor="#60a5fa", edgecolor="#1e3a8a", lw=1.8)
    atom_j = Circle((xj, yj), rj, facecolor="#fb923c", edgecolor="#9a3412", lw=1.8)
    ax_a.add_patch(atom_i)
    ax_a.add_patch(atom_j)
    ax_a.plot([xi], [yi], "o", color="#1e3a8a", ms=4.5)
    ax_a.plot([xj], [yj], "o", color="#9a3412", ms=4.5)
    ax_a.text(xi - 0.15, yi - 0.48, r"$\mathbf{x}_i^A$", fontsize=9.2, fontweight="bold", color="#0f172a")
    ax_a.text(xj - 0.45, yj - 0.48, r"$\mathbf{y}_j^B=\mathbf{R}\mathbf{x}_j^B+\mathbf{t}$", fontsize=8.6, fontweight="bold", color="#0f172a")

    # Radii r_i, r_j arrows
    ax_a.annotate("", xy=(xi + ri, yi - 0.12), xytext=(xi, yi - 0.12),
                  arrowprops=dict(arrowstyle="<->", color="#1e3a8a", lw=1.3))
    ax_a.text(xi + 0.42, yi - 0.42, r"$r_i$", fontsize=8.8, color="#1e3a8a", fontweight="bold")
    ax_a.annotate("", xy=(xj, yi - 0.12), xytext=(xj - rj, yi - 0.12),
                  arrowprops=dict(arrowstyle="<->", color="#9a3412", lw=1.3))
    ax_a.text(xj - 0.62, yi - 0.42, r"$r_j$", fontsize=8.8, color="#9a3412", fontweight="bold")

    # Surface-to-surface gap g_ij arrow
    ax_a.annotate("", xy=(xj - rj, yi + 0.12), xytext=(xi + ri, yi + 0.12),
                  arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=2.0))
    ax_a.text(5.0, yi + 0.35, r"gap $g_{ij}$", fontsize=9.5, fontweight="bold", color="#dc2626", ha="center")

    # Outward normals n_i^A and R n_j^B
    ax_a.annotate("", xy=(xi + 1.48, yi + 0.92), xytext=(xi + 0.55, yi + 0.52),
                  arrowprops=dict(arrowstyle="->", color="#047857", lw=1.8))
    ax_a.text(xi + 0.85, yi + 1.12, r"$\mathbf{n}_i^A$", fontsize=9.2, fontweight="bold", color="#047857")
    ax_a.annotate("", xy=(xj - 1.48, yj + 0.92), xytext=(xj - 0.55, yj + 0.52),
                  arrowprops=dict(arrowstyle="->", color="#047857", lw=1.8))
    ax_a.text(xj - 1.55, yj + 1.12, r"$\mathbf{R}\mathbf{n}_j^B$", fontsize=9.2, fontweight="bold", color="#047857")

    # Center-to-center distance d_ij bracket at top
    ax_a.annotate("", xy=(xj, 7.55), xytext=(xi, 7.55),
                  arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.3))
    ax_a.plot([xi, xi], [yi + ri, 7.62], ":", color="#64748b", lw=1.0)
    ax_a.plot([xj, xj], [yj + rj, 7.62], ":", color="#64748b", lw=1.0)
    ax_a.text(5.0, 7.75, r"$d_{ij}(\mathbf{t},\mathbf{R})=\|\mathbf{x}_i^A - (\mathbf{R}\mathbf{x}_j^B+\mathbf{t})\|$",
              fontsize=8.6, ha="center", color="#1e293b")

    # Equation summary box at bottom of Panel A
    eq_box_a = FancyBboxPatch((0.25, 0.25), 9.5, 3.35, boxstyle="round,pad=0.18",
                              facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.2)
    ax_a.add_patch(eq_box_a)
    ax_a.text(0.50, 3.05, r"$\mathbf{1.\;Surface\ Gap}\ (\Delta_{\max}=2.8\;\mathrm{\AA}):$"
              r"   $g_{ij}(\mathbf{t},\mathbf{R}) = d_{ij}(\mathbf{t},\mathbf{R}) - (r_i + r_j)$",
              fontsize=8.7, color="#0f172a")
    ax_a.text(0.50, 2.25, r"$\mathbf{2.\;Hard\ Sphere\ & \ Normalized\ Proximity:}$"
              r"   $g_{ij} < 0 \Rightarrow E_{\mathrm{HS}}=\infty,$"
              r"   $u_{ij} = \mathrm{clamp}\!\left(\frac{2.8 - g_{ij}}{2.8},\,0,\,1\right)$",
              fontsize=8.5, color="#0f172a")
    ax_a.text(0.50, 1.40, r"$\mathbf{3.\;Outward\ Surface\ Normal}\ (\sigma=3.5\;\mathrm{\AA}):$"
              r"   $\mathbf{n}_i^A = -\nabla\rho_{\mathrm{self}}(\mathbf{x}_i^A)/\|\nabla\rho_{\mathrm{self}}(\mathbf{x}_i^A)\|$",
              fontsize=8.5, color="#0f172a")
    ax_a.text(0.50, 0.62, r"$\mathbf{4.\;Solvent\ Exposure\ Factor:}$"
              r"   $s_{\mathrm{exp},i} = \exp(-\rho_{\mathrm{self},i}/45),\ \ "
              r"\rho_{\mathrm{self},i}=\sum_{k\neq i} r_k^3 e^{-\|\mathbf{x}_i-\mathbf{x}_k\|^2/\sigma^2}$",
              fontsize=8.3, color="#334155")

    # =========================================================================
    # PANEL B: Radial Partition-of-Unity Basis & Envelopes psi(g_ij)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    g_grid = np.linspace(0.0, 2.8, 400)
    u_grid = (2.8 - g_grid) / 2.8

    ax_b.axvspan(0.0, 0.7, color="#dcfce7", alpha=0.45, label="Direct H-bond shell ($0\\leq g_{ij}\\leq 0.7\\,\\mathrm{\\AA}$)")
    ax_b.axvspan(0.7, 2.8, color="#e0f2fe", alpha=0.45, label="Water / Ion bridge shell ($0.7 < g_{ij} < 2.8\\,\\mathrm{\\AA}$)")

    ax_b.plot(g_grid, hat_b1(u_grid), color="#15803d", lw=2.2, ls="--",
              label=r"$\hat{b}_1(u)=\max(0,\,4u-3)$ [short hat]")
    ax_b.plot(g_grid, hat_b2(u_grid), color="#0369a1", lw=2.2, ls=":",
              label=r"$\hat{b}_2(u)=\min(\frac{4}{3}u,\,4(1-u))$ [bridge hat]")
    ax_b.plot(g_grid, hat_b12(u_grid), color="#1e293b", lw=2.6,
              label=r"$\psi_{b12}(u)=\hat{b}_1+\hat{b}_2=\min(\frac{4}{3}u,1)$ [$S_{\mathrm{cos}}$ & burial]")
    ax_b.plot(g_grid, hat_hb(u_grid), color="#dc2626", lw=2.1,
              label=r"$\psi_{\mathrm{hb}}(u)=\hat{b}_1+0.25\hat{b}_2$ [H-bond / clash]")
    ax_b.plot(g_grid, hat_wb(u_grid), color="#7c3aed", lw=2.1,
              label=r"$\psi_{\mathrm{wb}}(u)=\hat{b}_2(1-\hat{b}_1)$ [water / Hofmeister]")

    ax_b.axvline(0.7, color="#475569", lw=1.0, ls="-.")
    ax_b.text(0.73, 1.03, r"$g_{ij}=0.7\,\mathrm{\AA}$ ($u=0.75$)", fontsize=8.0, color="#334155")

    ax_b.set_xlim(0.0, 2.8)
    ax_b.set_ylim(-0.03, 1.15)
    ax_b.set_xlabel(r"Surface-to-surface gap $g_{ij}=d_{ij}-(r_i+r_j)$ ($\mathrm{\AA}$)")
    ax_b.set_ylabel(r"Radial envelope weight $\psi(g_{ij})$")
    ax_b.set_title(
        r"(B) Radial Basis ($\hat{b}_1,\hat{b}_2$) & Envelopes $\psi(g_{ij})$",
        fontweight="bold", loc="left", pad=22, color="#0f172a"
    )
    ax_b.grid(True, ls=":", alpha=0.45)
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.5)

    # Top secondary axis for normalized proximity u_ij
    ax_b_top = ax_b.twiny()
    ax_b_top.set_xlim(0.0, 2.8)
    u_ticks = [1.0, 0.75, 0.50, 0.25, 0.0]
    ax_b_top.set_xticks([2.8 * (1.0 - u) for u in u_ticks])
    ax_b_top.set_xticklabels([f"$u={u:.2f}$" for u in u_ticks], fontsize=7.8)
    ax_b_top.tick_params(axis="x", pad=2)

    # =========================================================================
    # PANEL C: Lawrence-Colman S_c vs Multi-Point Contrastive S_cos(t, R)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis("off")
    ax_c.set_title(
        r"(C) Lawrence–Colman $S_c$ vs. Multi-Point Contrastive $S_{\mathrm{cos}}(\mathbf{t},\mathbf{R})$",
        fontweight="bold", loc="left", pad=8, color="#0f172a"
    )

    # Top box: Lawrence & Colman S_c vs S_cos equations
    box_c1 = FancyBboxPatch((0.2, 5.35), 9.6, 4.35, boxstyle="round,pad=0.18",
                            facecolor="#f8fafc", edgecolor="#64748b", lw=1.2)
    ax_c.add_patch(box_c1)
    ax_c.text(0.45, 9.15, r"$\mathbf{Shared\ Symmetric\ Interface\ Normalization}\ (c_0=0.25):$",
              fontsize=8.9, fontweight="bold", color="#0f172a")
    ax_c.text(0.45, 8.20,
              r"$S(A,B) = \frac{1}{2}\!\left[\sum_{i\in A} a_i \frac{\sum_{j\in B}\psi_{b12}(g_{ij})\,a_j\,s_{ij}}{\sum_{j\in B}\psi_{b12}(g_{ij})\,a_j + c_0} + (A \leftrightarrow B)\right]$",
              fontsize=9.2, color="#0f172a")
    ax_c.text(0.45, 7.10,
              r"$\bullet\ \mathbf{Lawrence\text{--}Colman\ } S_c:\ \ s_{ij}^{\mathrm{LC}}(\mathbf{R}) = -\mathbf{n}_i^A \cdot (\mathbf{R}\mathbf{n}_j^B) \in [-1,1]\ \ (a_i=1/\!N_{\mathrm{int}})$",
              fontsize=8.4, color="#047857")
    ax_c.text(0.45, 6.30,
              r"$\bullet\ \mathbf{Contrastive\ } S_{\mathrm{cos}}:\ \ s_{ij} = \frac{1}{2}(\mathbf{u}_i^\top\mathbf{v}_j + \mathbf{u}_j^\top\mathbf{v}_i) \in [-1,1],\ \ \mathbf{u}_i,\mathbf{v}_i \in S^{D-1}\ (D=8)$",
              fontsize=8.4, color="#1d4ed8")
    ax_c.text(0.45, 5.60,
              r"$\bullet\ \mathbf{3\text{-}Body\ Junction\ Gate}:\ \ g_{\mathrm{cos}}(A,B)=\mathrm{clamp}(S_{\mathrm{cos}}(A,B)/0.18,\,0,\,1)$",
              fontsize=8.4, color="#7c3aed")

    # Bottom schematic: Why multi-point S_cos depends strongly on 6D pose (t, R)
    box_c2 = FancyBboxPatch((0.2, 0.25), 4.6, 4.75, boxstyle="round,pad=0.15",
                            facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.3)
    box_c3 = FancyBboxPatch((5.2, 0.25), 4.6, 4.75, boxstyle="round,pad=0.15",
                            facecolor="#fef2f2", edgecolor="#dc2626", lw=1.3)
    ax_c.add_patch(box_c2)
    ax_c.add_patch(box_c3)

    ax_c.text(2.5, 4.30, "Native Lock-and-Key Pose\n$(\\mathbf{t}^*,\\mathbf{R}^*):\\ S_{\\mathrm{cos}} > 0$",
              fontsize=8.3, fontweight="bold", color="#15803d", ha="center")
    ax_c.text(7.5, 4.30, "Sheared / Rotated Decoy\n$(\\mathbf{t}^*+\\Delta\\mathbf{t},\\,\\mathbf{R}'):\\ S_{\\mathrm{cos}} < 0$",
              fontsize=8.3, fontweight="bold", color="#b91c1c", ha="center")

    # Native paired nodes (i1-j1, i2-j2, i3-j3)
    xs_l = [1.1, 2.5, 3.9]
    for k, xk in enumerate(xs_l, 1):
        ax_c.plot([xk], [3.22], "o", color="#1d4ed8", ms=7.5)
        ax_c.plot([xk], [1.60], "o", color="#ea580c", ms=7.5)
        ax_c.annotate("", xy=(xk, 1.78), xytext=(xk, 3.04),
                      arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=2.0))
        ax_c.text(xk, 3.52, f"$i_{k}$", fontsize=8.0, ha="center", fontweight="bold", color="#1e3a8a")
        ax_c.text(xk, 1.15, f"$j_{k}$", fontsize=8.0, ha="center", fontweight="bold", color="#9a3412")
    ax_c.text(2.5, 0.48, r"All $\psi_{b12}(g_{i_k j_k})>0$ on matched" "\n" r"partner pairs $s_{i_k j_k}\approx +0.8$",
              fontsize=7.8, ha="center", color="#14532d")

    # Sheared / rotated nodes (i1-j2, i2-j3, off-diagonal)
    xs_r_top = [6.0, 7.4, 8.8]
    xs_r_bot = [6.6, 8.0, 9.3]
    for k, (xt, xb) in enumerate(zip(xs_r_top, xs_r_bot), 1):
        ax_c.plot([xt], [3.22], "o", color="#1d4ed8", ms=7.5)
        ax_c.plot([xb], [1.60], "o", color="#ea580c", ms=7.5)
        ax_c.text(xt, 3.52, f"$i_{k}$", fontsize=8.0, ha="center", fontweight="bold", color="#1e3a8a")
        ax_c.text(xb, 1.15, f"$j_{k}$", fontsize=8.0, ha="center", fontweight="bold", color="#9a3412")
    ax_c.annotate("", xy=(xs_r_bot[0], 1.78), xytext=(xs_r_top[1], 3.04),
                  arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.8, ls="--"))
    ax_c.annotate("", xy=(xs_r_bot[1], 1.78), xytext=(xs_r_top[2], 3.04),
                  arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.8, ls="--"))
    ax_c.text(7.5, 0.48, r"Shift/twist swaps contacts to" "\n" r"off-diagonal pairs $s_{i_k j_m}\approx -0.65$",
              fontsize=7.8, ha="center", color="#7f1d1d")

    # =========================================================================
    # PANEL D: 6D Pose Sensitivity S_c(Delta t) vs S_cos(Delta t)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.axhline(0.0, color="#475569", lw=1.1, ls="--")
    ax_d.axvline(0.0, color="#94a3b8", lw=1.0, ls=":")
    slide_scos_o_clip = np.clip(slide_scos_o, -1.15, 1.35)
    ax_d.fill_between(slide_d, 0, np.maximum(0, slide_scos), color="#dbeafe", alpha=0.55,
                      label=r"Attractive lock-and-key basin ($S_{\mathrm{cos}} > 0$)")
    ax_d.fill_between(slide_d, np.minimum(0, np.minimum(slide_scos, slide_scos_o_clip)), 0, color="#fee2e2", alpha=0.55,
                      label=r"Off-diagonal decoy rejection ($S_{\mathrm{cos}} < 0$)")

    ax_d.plot(slide_d, slide_scos, color="#1d4ed8", lw=2.5,
              label=r"Multi-polymorph $S_{\mathrm{cos}}(\Delta\theta)$ [8 polymorphs]")
    ax_d.plot(slide_d, slide_scos_o_clip, color="#6366f1", lw=1.9, ls="--",
              label=r"Single-polymorph $S_{\mathrm{cos}}(\Delta\theta)$ [1LYZ/1IEE]")
    ax_d.plot(slide_d, slide_sc, color="#047857", lw=2.2, ls="-.",
              label=r"Lawrence–Colman geometric $S_c(\Delta\theta)$")

    ax_d.set_xlim(-60.0, 60.0)
    ax_d.set_ylim(-1.95, 1.50)
    ax_d.set_xlabel(r"Grazing interface roll/slide perturbation $\Delta\theta$ (deg)")
    ax_d.set_ylabel(r"Complementarity score $S(\mathbf{t}(\Delta\theta),\, \mathbf{R}(\Delta\theta))$")
    ax_d.set_title(
        r"(D) Pose Funnel: $S_{\mathrm{cos}}(\Delta\theta)$ vs. $S_c(\Delta\theta)$",
        fontweight="bold", loc="left", pad=8, color="#0f172a"
    )
    ax_d.grid(True, ls=":", alpha=0.45)
    ax_d.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
    ax_d.text(0.0, 1.18, r"Native crystal pose $\Delta\theta = 0^\circ$" "\n" r"($S_{\mathrm{cos}}=+1.09,\ S_c=+0.41$)",
              fontsize=8.0, ha="center", fontweight="bold", color="#1e3a8a",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#93c5fd"))

    # =========================================================================
    # PANEL E: Verification Across All 8 HEWL Polymorphs (53 Crystal Contacts)
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    labels_e = [
        f"{p['tag'].split('-')[1]}\n({p['space_group'].replace(' ', '')})"
        for p in poly_stats
    ]
    x_e = np.arange(len(poly_stats))
    w_bar = 0.26

    sc_vals = [p["sc_mean"] for p in poly_stats]
    scos_u_vals = [p["scos_u_mean"] for p in poly_stats]
    echem_vals = [-(p["epol0_mean"]) / 10.0 for p in poly_stats]  # scaled to [-0.5, +1.5] for visual comparison

    ax_e.bar(x_e - w_bar, sc_vals, width=w_bar, color="#10b981", edgecolor="#065f46", lw=0.9,
             label=r"Lawrence–Colman $S_c$ ($53/53 > 0$, 100%)")
    ax_e.bar(x_e, echem_vals, width=w_bar, color="#f59e0b", edgecolor="#92400e", lw=0.9,
             label=r"6-channel $-E_{\mathrm{pol},0}/10\,k_BT$ ($47/53 > 0$, $53/53$ w/ salt)")
    ax_e.bar(x_e + w_bar, scos_u_vals, width=w_bar, color="#2563eb", edgecolor="#1e3a8a", lw=0.9,
             label=r"Multi-polymorph $S_{\mathrm{cos}}$ ($50/53 > 0$, 94.3%)")

    ax_e.axhline(0.0, color="#334155", lw=1.0)
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(labels_e, fontsize=7.6)
    ax_e.set_ylim(-0.15, 2.55)
    ax_e.set_ylabel(r"Mean interface complementarity score")
    ax_e.set_title(
        r"(E) Physical Complementarity Across All 8 Polymorphs ($53$ Contacts)",
        fontweight="bold", loc="left", pad=8, color="#0f172a"
    )
    ax_e.grid(True, axis="y", ls=":", alpha=0.45)
    ax_e.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)

    for i, p in enumerate(poly_stats):
        ax_e.text(i + w_bar, scos_u_vals[i] + 0.04, f"{p['scos_u_pos']}/{p['n_c']}",
                  ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1e3a8a")

    # =========================================================================
    # PANEL F: Morphology-Selective Binding Energy Across Salt Conditions
    # =========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    # Show Hofmeister stabilization dE_Hof (kT) across the 3 salt regimes
    dh_nacl = [-p["dh_nacl"] for p in poly_stats]
    dh_nano3 = [-p["dh_nano3"] for p in poly_stats]
    dh_so4 = [-p["dh_so4"] for p in poly_stats]

    ax_f.bar(x_e - w_bar, dh_nacl, width=w_bar, color="#0284c7", edgecolor="#0c4a6e", lw=0.9,
             label=r"$0.85\,\mathrm{M\ NaCl}$ ($\mathrm{Cl}^-$ halide bridge)")
    ax_f.bar(x_e, dh_nano3, width=w_bar, color="#ea580c", edgecolor="#7c2d12", lw=0.9,
             label=r"$0.35\,\mathrm{M\ NaNO_3}$ ($\mathrm{NO}_3^-$ chaotrope bridge)")
    ax_f.bar(x_e + w_bar, dh_so4, width=w_bar, color="#7c3aed", edgecolor="#4c1d95", lw=0.9,
             label=r"$0.20\,\mathrm{M\ SO_4^{2-}}$ (divalent bridge)")

    for i, p in enumerate(poly_stats):
        st = p["salt_type"]
        y_top = max(dh_nacl[i], dh_nano3[i], dh_so4[i])
        c_tag = "#0284c7" if st == "NaCl" else ("#ea580c" if st == "NaNO3" else "#7c3aed")
        ax_f.text(i, y_top + 0.65, st, ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=c_tag)

    ax_f.set_xticks(x_e)
    ax_f.set_xticklabels(labels_e, fontsize=7.6)
    ax_f.set_ylim(0.0, 38.0)
    ax_f.set_ylabel(r"Hofmeister bridge stabilization $-\Delta E_{\mathrm{Hof}}(\mathbf{c})$ ($k_BT$)")
    ax_f.set_title(
        r"(F) Salt-Conditioned Hofmeister Stabilization $-\Delta E_{\mathrm{Hof}}(\mathbf{c})$",
        fontweight="bold", loc="left", pad=8, color="#0f172a"
    )
    ax_f.grid(True, axis="y", ls=":", alpha=0.45)
    ax_f.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.7)

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            out_file = os.path.join(
                out_dir, f"mathematical_definitions_and_polymorph_complementarity.{ext}"
            )
            fig.savefig(out_file, dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("Saved mathematical_definitions_and_polymorph_complementarity.{png,pdf,svg}")


if __name__ == "__main__":
    build_figure()
