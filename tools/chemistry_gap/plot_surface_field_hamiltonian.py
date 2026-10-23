#!/usr/bin/env python3
"""Generate publication-ready figure illustrating the cutoff-free piecewise-linear
surface-field many-body Hamiltonian and its resolution of 1D ribbon vs. 3D crystal growth."""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch, Polygon
import matplotlib.patheffects as pe


def draw_monomer_pill(ax, center, width, height, angle_deg, facecolor, edgecolor, label=None, sublabel=None, fontsize=10, z=4):
    """Draw a clean rotated rounded capsule representing a rigid protein monomer."""
    cx, cy = center
    t = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + ax.transData
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={min(width, height)*0.38:.3f}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.0,
        transform=t,
        zorder=z,
    )
    ax.add_patch(box)
    if label:
        ax.text(
            cx,
            cy + (0.08 * height if sublabel else 0.0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#111827",
            zorder=z + 2,
        )
    if sublabel:
        ax.text(
            cx,
            cy - 0.24 * height,
            sublabel,
            ha="center",
            va="center",
            fontsize=fontsize - 2.0,
            color="#374151",
            zorder=z + 2,
        )


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    fig = plt.figure(figsize=(16.2, 11.4), dpi=300, facecolor="white")
    gs_top = fig.add_gridspec(
        1, 2,
        left=0.035, right=0.975,
        bottom=0.515, top=0.875,
        wspace=0.16,
        width_ratios=[1.04, 1.04],
    )
    gs_bot = fig.add_gridspec(
        1, 2,
        left=0.035, right=0.975,
        bottom=0.065, top=0.440,
        wspace=0.31,
        width_ratios=[1.08, 0.96],
    )

    # Overall Title & Subtitle
    fig.text(
        0.5, 0.962,
        "Cutoff-Free Piecewise-Linear Surface-Field Hamiltonian for 3D Protein Crystallization",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0f172a"
    )
    fig.text(
        0.5, 0.928,
        r"Unifying 2-body contrastive cosine embeddings $\boldsymbol{\psi}_i = (\mathbf{n}_i,\; \gamma_c a_i(\mathbf{u}_i - \mathbf{v}_i))$ with radial hat functions $B_1(g), B_2(g)$ eliminates $s>0.5$ cutoffs and triangle rules",
        ha="center", va="center", fontsize=10.8, color="#334155"
    )

    # =========================================================================
    # PANEL A: Schematic of the Surface Feature Field & Radial Hat Basis on Monomer A
    # =========================================================================
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 7.4)
    ax_a.axis("off")

    ax_a.text(
        0.0, 7.15,
        "a   Piecewise-Linear Radial Surface Field on Vertex Monomer $A$",
        fontsize=12.2, fontweight="bold", color="#0f172a", ha="left", va="center"
    )

    # Background card
    card_a = FancyBboxPatch(
        (0.1, 0.15), 9.8, 6.65,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2, zorder=0
    )
    ax_a.add_patch(card_a)

    # Draw vertex monomer A at bottom-center, and neighbors B (left-top) and C (right-top)
    cA = (3.45, 2.25)
    # Outer hydration collar shell B2 (0 < g < 2.8 A, peaking at 0.7 A)
    collar_b = Wedge(cA, 2.52, 82, 165, width=0.58, facecolor="#0d9488", alpha=0.26, edgecolor="#0d9488", linestyle="--", linewidth=1.3, zorder=2)
    collar_c = Wedge(cA, 2.52, 18, 104, width=0.58, facecolor="#0284c7", alpha=0.26, edgecolor="#0284c7", linestyle="--", linewidth=1.3, zorder=2)
    ax_a.add_patch(collar_b)
    ax_a.add_patch(collar_c)

    # Seam overlap region (82..104 deg in outer collar B2)
    seam_wedge = Wedge(cA, 2.52, 82, 104, width=0.58, facecolor="#10b981", alpha=0.65, edgecolor="#047857", linewidth=1.8, zorder=3)
    ax_a.add_patch(seam_wedge)

    # Inner contact core shell B1 (0 <= g <= 0.7 A)
    core_b = Wedge(cA, 1.94, 98, 152, width=0.28, facecolor="#ef4444", alpha=0.55, edgecolor="#b91c1c", linewidth=1.4, zorder=3)
    core_c = Wedge(cA, 1.94, 30, 80, width=0.28, facecolor="#ef4444", alpha=0.55, edgecolor="#b91c1c", linewidth=1.4, zorder=3)
    ax_a.add_patch(core_b)
    ax_a.add_patch(core_c)

    # Hard body of Monomer A
    body_a = Circle(cA, 1.66, facecolor="#e2e8f0", edgecolor="#1e293b", linewidth=2.2, zorder=4)
    ax_a.add_patch(body_a)
    ax_a.text(cA[0], cA[1] - 0.22, "Vertex Monomer $A$", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a", zorder=6)
    ax_a.text(cA[0], cA[1] - 0.62, r"Surface features $\boldsymbol{\psi}_i \in \mathbb{R}^{11}$", ha="center", va="center", fontsize=8.8, color="#334155", zorder=6)

    # Surface feature vectors psi_i = (n_i, a_i(u_i - v_i)) along rim of A
    for ang_deg in [42, 62, 86, 96, 118, 138]:
        rad = np.deg2rad(ang_deg)
        x0 = cA[0] + 1.66 * np.cos(rad)
        y0 = cA[1] + 1.66 * np.sin(rad)
        x1 = cA[0] + 2.08 * np.cos(rad)
        y1 = cA[1] + 2.08 * np.sin(rad)
        ax_a.plot([x0], [y0], "o", color="#1e293b", markersize=4.5, zorder=6)
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4,head_width=2.5", color="#334155", linewidth=1.2, zorder=6)
        ax_a.add_patch(arr)

    # Neighbor B and Neighbor C shapes
    draw_monomer_pill(ax_a, (1.45, 4.95), 2.15, 1.25, -32, "#dbeafe", "#1d4ed8", "Neighbor $B$", r"patch $(A \leftarrow B)$", fontsize=9.8, z=5)
    draw_monomer_pill(ax_a, (5.45, 4.95), 2.15, 1.25, 32, "#fef3c7", "#b45309", "Neighbor $C$", r"patch $(A \leftarrow C)$", fontsize=9.8, z=5)

    # Callout annotations on diagram
    ax_a.annotate(
        "Hydration-collar seam\n" + r"$B_2(u_i^B)\,B_2(u_i^C) > 0$" + "\n(shared dewetting rim)",
        xy=(3.38, 4.52), xytext=(3.45, 6.08),
        ha="center", va="center", fontsize=8.7, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#d1fae5", edgecolor="#059669", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
        zorder=8
    )
    ax_a.annotate(
        "Inner contact core\n" + r"$B_1(u_i) > 0$" + "\n" + r"($g_i \in [0, 0.7]$ Å)",
        xy=(2.20, 3.32), xytext=(0.98, 2.75),
        ha="center", va="center", fontsize=8.3, color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#dc2626", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3),
        zorder=8
    )

    # Right-hand formula box inside Panel A explaining the unified math
    eq_box = FancyBboxPatch(
        (6.45, 0.42), 3.25, 6.12,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        facecolor="#ffffff", edgecolor="#94a3b8", linewidth=1.2, zorder=4
    )
    ax_a.add_patch(eq_box)
    ax_a.text(8.08, 6.22, "Unified Hamiltonian", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0f172a", zorder=6)

    lines_a = [
        (5.68, "#1e293b", r"$\mathbf{1.\;Learned\;Atom\;Feature}$"),
        (5.23, "#334155", r"$\boldsymbol{\psi}_i = \mathbf{n}_i \oplus \gamma_c a_i(\mathbf{u}_i - \mathbf{v}_i)$"),
        (4.78, "#475569", r"($\mathbf{u}_i,\mathbf{v}_i \in S^7$ from InfoNCE)"),
        (4.15, "#1e293b", r"$\mathbf{2.\;2\text{-}Body\;Cosine\;Match}$"),
        (3.68, "#334155", r"$S_{\cos}^{AB} = \sum_{i \in A} a_i \frac{\sum_j w_{ij} s_{ij}}{\sum_j w_{ij} + c_0}$"),
        (3.05, "#991b1b", r"$\mathbf{3.\;Core\;Crowding\;(B_1)}$"),
        (2.58, "#991b1b", r"$E_{\mathrm{crowd}} = +w_{\mathrm{sat}}\|\mathbf{V}_{\mathrm{core}}^A\|$"),
        (2.14, "#475569", r"$\mathbf{V}_{\mathrm{core}}^A = \sum_i B_1(u_i^B) B_1(u_i^C)\boldsymbol{\psi}_i$"),
        (1.52, "#065f46", r"$\mathbf{4.\;Collar\;Sealing\;(B_2)}$"),
        (1.04, "#065f46", r"$E_{\mathrm{seal}} = -w_{\mathrm{seal}} g_{AB} g_{AC} \frac{\|\mathbf{V}_{\mathrm{col}}^A\|^2}{\|\mathbf{V}_{\mathrm{col}}^A\|+w_0}$"),
        (0.58, "#475569", r"$\mathbf{V}_{\mathrm{col}}^A = \sum_i B_2(u_i^B) B_2(u_i^C)\boldsymbol{\psi}_i$"),
    ]
    for y_pos, col, txt in lines_a:
        ax_a.text(6.62, y_pos, txt, ha="left", va="center", fontsize=8.6, color=col, zorder=6)

    # =========================================================================
    # PANEL B: 2-Knot Piecewise-Linear Hat Basis & 1L2Y Surface-Field Norms
    # =========================================================================
    gs_b = gs_top[0, 1].subgridspec(2, 1, hspace=0.45, height_ratios=[1.05, 1.0])
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])

    ax_b1.set_title(
        "b   2-Knot Piecewise-Linear Radial Hat Basis & 1L2Y Vertex Overlaps",
        loc="left", fontsize=12.2, fontweight="bold", color="#0f172a", pad=10
    )

    g = np.linspace(0.0, 3.2, 500)
    u = np.clip(1.0 - g / 2.8, 0.0, 1.0)
    b1 = np.maximum(0.0, 4.0 * u - 3.0)
    b2 = np.minimum((4.0 / 3.0) * u, 4.0 * (1.0 - u))
    b2 = np.maximum(0.0, b2)

    ax_b1.axvspan(0.0, 0.7, color="#fee2e2", alpha=0.45, label="Inner contact core ($0 - 0.7$ Å)")
    ax_b1.axvspan(0.7, 2.8, color="#ccfbf1", alpha=0.40, label="Outer hydration collar ($0.7 - 2.8$ Å)")
    ax_b1.plot(g, u, color="#64748b", linestyle=":", linewidth=2.0, label=r"Ramp $u(g) = B_1(g) + \frac{3}{4}B_2(g)$")
    ax_b1.plot(g, b1, color="#dc2626", linewidth=2.6, label=r"Core hat $B_1(u) = \max(0, 4u-3)$")
    ax_b1.plot(g, b2, color="#0d9488", linewidth=2.6, label=r"Collar hat $B_2(u) = \min(\frac{4}{3}u, 4(1-u))$")

    for knot_g in [0.0, 0.7, 2.8]:
        ax_b1.axvline(knot_g, color="#475569", linestyle="--", linewidth=0.9, alpha=0.7)

    ax_b1.set_xlim(0.0, 3.15)
    ax_b1.set_ylim(-0.04, 1.15)
    ax_b1.set_xlabel("Atom-Atom Surface Gap $g_{ij} = d_{ij} - (r_i + r_j)$ (Å)", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_b1.set_ylabel("Basis Weight", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_b1.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.92, ncol=2)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)

    # Bottom subplot of B: Real 1L2Y norms ||V_core|| vs ||V_collar|| across patch pairs
    categories = [
        "Same-site crowding\n(decoy $\\times$ m4)",
        "1D $4_3$ helix wedge\n($\\mathrm{m1}^+ \\times \\mathrm{m2}^-$)",
        "1D $4_3$ tip wedge\n($\\mathrm{m1}^+ \\times \\mathrm{m0}^+$)",
        "3D 4-cycle corner\n($\\mathrm{m2}^- \\times \\mathrm{m4}^+$)",
    ]
    v_core_vals = [2.970, 0.000, 0.000, 0.176]
    v_collar_vals = [0.000, 0.562, 0.305, 1.412]

    x_b = np.arange(len(categories))
    w_bar = 0.34
    rects1 = ax_b2.bar(x_b - w_bar/2, v_core_vals, w_bar, color="#dc2626", edgecolor="#991b1b", linewidth=1.0, label=r"Repulsive Core Crowding $\|\mathbf{V}_{\mathrm{core}}^A\|$")
    rects2 = ax_b2.bar(x_b + w_bar/2, v_collar_vals, w_bar, color="#0d9488", edgecolor="#115e59", linewidth=1.0, label=r"Attractive Collar Seam $g_{AB}g_{AC}\|\mathbf{V}_{\mathrm{collar}}^A\|$")

    for r in rects1:
        val = r.get_height()
        if val > 0.01:
            ax_b2.text(r.get_x() + r.get_width()/2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#991b1b")
    for r in rects2:
        val = r.get_height()
        if val > 0.01:
            ax_b2.text(r.get_x() + r.get_width()/2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#0f766e")

    ax_b2.set_xticks(x_b)
    ax_b2.set_xticklabels(categories, fontsize=8.6, color="#1e293b")
    ax_b2.set_ylim(0, 3.45)
    ax_b2.set_ylabel("11D Field Norm on $A$", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_b2.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="white")
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL C: Architectural Comparison — 1D 3-Cycle vs 3D Inter-Helix 4-Cycle
    # =========================================================================
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 7.6)
    ax_c.axis("off")

    ax_c.text(
        0.0, 7.38,
        "c   Why Pure-Vertex Surface Fields Resolve Both 1D 3-Cycles & 3D 4-Cycles",
        fontsize=12.2, fontweight="bold", color="#0f172a", ha="left", va="center"
    )

    # Left sub-card: 1D 4_3 Screw Ribbon (3-cycle m1 + m2 = m0)
    card_c1 = FancyBboxPatch(
        (0.08, 0.15), 4.65, 6.80,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.3, zorder=0
    )
    ax_c.add_patch(card_c1)
    ax_c.text(2.40, 6.62, "1D $4_3$ Screw Ribbon (3-Cycle)", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#14532d")
    ax_c.text(2.40, 6.22, r"Closed triangle: $\mathrm{m1} \circ \mathrm{m2} = \mathrm{m0}$ along $c$-axis", ha="center", va="center", fontsize=8.5, color="#166534")

    p_h0 = (1.32, 5.02)
    p_h1 = (3.62, 3.65)
    p_h4 = (1.32, 2.22)

    # Edges of 3-cycle
    ax_c.plot([p_h0[0], p_h1[0]], [p_h0[1], p_h1[1]], color="#15803d", linewidth=3.0, zorder=2)
    ax_c.plot([p_h1[0], p_h4[0]], [p_h1[1], p_h4[1]], color="#15803d", linewidth=3.0, zorder=2)
    ax_c.plot([p_h0[0], p_h4[0]], [p_h0[1], p_h4[1]], color="#0d9488", linewidth=2.4, linestyle="--", zorder=2)

    ax_c.text(2.85, 4.68, "m1\n($-9.5$ Å, $90^\\circ$)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#14532d", bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#86efac"), zorder=5)
    ax_c.text(2.85, 2.58, "m2\n($+28.4$ Å, $90^\\circ$)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#14532d", bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#86efac"), zorder=5)
    ax_c.text(0.80, 3.62, "m0\n($+37.9$ Å)\n$g_{\\min}=1.24$ Å", ha="center", va="center", fontsize=7.8, color="#0f766e", bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#5eead4"), zorder=5)

    draw_monomer_pill(ax_c, p_h0, 1.38, 0.78, 0, "#dcfce7", "#15803d", "$H_1[0]$", "$z=0$", fontsize=9.0, z=4)
    draw_monomer_pill(ax_c, p_h1, 1.38, 0.78, 0, "#dcfce7", "#15803d", "$H_1[1]$", "$z=-c/4$", fontsize=9.0, z=4)
    draw_monomer_pill(ax_c, p_h4, 1.38, 0.78, 0, "#bbf7d0", "#166534", "$H_1[4]$", "$z=-c$", fontsize=9.0, z=4)

    ax_c.text(
        2.40, 0.88,
        "Both old & new models stabilize 1D tip:\n"
        r"$\ln K_{\mathrm{tip}} = +14.34 > \ln K_{\mathrm{crit}} (+13.29)$",
        ha="center", va="center", fontsize=8.3, fontweight="bold", color="#14532d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#dcfce7", edgecolor="#22c55e", lw=1.1)
    )

    # Right sub-card: 3D Inter-Helix Lateral 4-Cycle (m1 - m4 - m2 - m6)
    card_c2 = FancyBboxPatch(
        (5.00, 0.15), 4.90, 6.80,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.3, zorder=0
    )
    ax_c.add_patch(card_c2)
    ax_c.text(7.45, 6.62, "3D Lateral Packing (Inter-Helix 4-Cycle)", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#1e3a8a")
    ax_c.text(7.45, 6.22, r"Antiparallel helices $H_1 \leftrightarrow H_2$: diagonal $\mathrm{m3}$ is an open gap", ha="center", va="center", fontsize=8.5, color="#1d4ed8")

    q00 = (6.02, 5.02)  # H1[0]
    q01 = (6.02, 2.32)  # H1[1]
    q10 = (8.88, 5.02)  # H2[0]
    q11 = (8.88, 2.32)  # H2[-3] (target corner)

    # 4-cycle edges
    ax_c.plot([q00[0], q01[0]], [q00[1], q01[1]], color="#15803d", linewidth=3.0, zorder=2)  # m1
    ax_c.plot([q00[0], q10[0]], [q00[1], q10[1]], color="#2563eb", linewidth=3.0, zorder=2)  # m6
    ax_c.plot([q01[0], q11[0]], [q01[1], q11[1]], color="#d97706", linewidth=3.2, zorder=2)  # m4
    ax_c.plot([q10[0], q11[0]], [q10[1], q11[1]], color="#15803d", linewidth=3.0, zorder=2)  # m2
    # Open diagonal m3
    ax_c.plot([q01[0], q10[0]], [q01[1], q10[1]], color="#94a3b8", linewidth=1.6, linestyle=":", zorder=1)

    ax_c.text(5.52, 3.68, "m1", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#15803d")
    ax_c.text(9.38, 3.68, "m2", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#15803d")
    ax_c.text(7.45, 5.32, "m6 ($C_2$ bridge)", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1d4ed8")
    ax_c.text(7.45, 1.98, "m4 ($C_2$ bridge)", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#b45309")
    ax_c.text(
        7.45, 3.72,
        "Open diagonal m3\n$g_{\\min}=2.45$ Å ($g_{\\cos}=0$)\nNo 3-body triangle!",
        ha="center", va="center", fontsize=7.8, color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0),
        zorder=3
    )

    # Highlight hydration-collar seam on vertices H1[1] and H2[-3]
    seam_c1 = Wedge(q01, 0.90, 0, 90, width=0.26, facecolor="#10b981", alpha=0.80, edgecolor="#047857", linewidth=1.4, zorder=5)
    seam_c2 = Wedge(q11, 0.90, 90, 180, width=0.26, facecolor="#10b981", alpha=0.80, edgecolor="#047857", linewidth=1.4, zorder=5)
    ax_c.add_patch(seam_c1)
    ax_c.add_patch(seam_c2)

    draw_monomer_pill(ax_c, q00, 1.35, 0.76, 0, "#dcfce7", "#15803d", "$H_1[0]$", "Helix 1", fontsize=8.6, z=4)
    draw_monomer_pill(ax_c, q01, 1.35, 0.76, 0, "#dcfce7", "#15803d", "$H_1[1]$", "Helix 1", fontsize=8.6, z=4)
    draw_monomer_pill(ax_c, q10, 1.35, 0.76, 0, "#dbeafe", "#1d4ed8", "$H_2[0]$", "Helix 2", fontsize=8.6, z=4)
    draw_monomer_pill(ax_c, q11, 1.35, 0.76, 0, "#fef3c7", "#d97706", "$H_2[-3]$", "4-Cycle Corner", fontsize=8.6, z=4)

    ax_c.text(
        7.45, 0.88,
        "Collar seam $B_2(\\mathrm{m2}^-)B_2(\\mathrm{m4}^+)$ seals directly on vertex:\n"
        r"$\ln K_{\mathrm{4\text{-}cycle}}: -11.14 \longrightarrow \mathbf{+14.74 > \ln K_{\mathrm{crit}}}$",
        ha="center", va="center", fontsize=8.3, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#d1fae5", edgecolor="#059669", lw=1.2)
    )

    # =========================================================================
    # PANEL D: Exact Multi-Body pair-binding Site Constants ln K_site
    # =========================================================================
    ax_d = fig.add_subplot(gs_bot[0, 1])
    ax_d.set_title(
        "d   Multi-Body Crystal Growth Site Constants $\\ln K_{\\mathrm{site}}$ (`pair-binding`)",
        loc="left", fontsize=12.2, fontweight="bold", color="#0f172a", pad=10
    )

    sites_lbl = [
        "1. 1D Helix Tip\n($\\mathrm{m1}+\\mathrm{m2}+\\mathrm{m0}$)",
        "2. Lateral 1st (m6)\n($C_2$ bridge)",
        "3. Lateral 1st (m4)\n($C_2$ bridge)",
        "4. Lateral 2nd\n4-Cycle ($\\mathrm{m2}+\\mathrm{m4}$)",
        "5. Lateral 4th\n3D Turn ($\\mathrm{m1}+\\mathrm{m2}+\\mathrm{m4}$)",
    ]
    # Exact pair-binding ln K values
    lnk_old = [14.41, 5.31, -3.27, -11.14, 8.46]
    lnk_nojunc = [13.87, 5.31, 4.08, 5.41, 26.08]
    lnk_new = [14.34, 6.02, 6.97, 14.74, 31.01]

    y_pos = np.arange(len(sites_lbl))[::-1]
    h_bar = 0.25

    ax_d.axvspan(13.29, 36.5, color="#ecfdf5", alpha=0.55, zorder=0)
    ax_d.axvline(0.0, color="#64748b", linewidth=1.0, zorder=1)
    ax_d.axvline(13.29, color="#059669", linestyle="--", linewidth=1.8, zorder=2)
    ax_d.text(
        14.2, 2.55,
        r"Supercritical growth $\ln K_{\mathrm{crit}} = +13.29$" + "\n(at $c_0 = 2800\\;\\mu$M)",
        fontsize=8.4, fontweight="bold", color="#065f46", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#059669", alpha=0.95),
        zorder=5
    )

    b_old = ax_d.barh(y_pos + h_bar, lnk_old, h_bar, color="#f87171", edgecolor="#b91c1c", linewidth=0.9, label="Old cutoff + triangle rule ($s>0.5$, 3-cycle cancel)", zorder=3)
    b_noj = ax_d.barh(y_pos, lnk_nojunc, h_bar, color="#94a3b8", edgecolor="#475569", linewidth=0.9, label="No junction field ($w_{\\mathrm{seal}}=w_{\\mathrm{sat}}=0$)", zorder=3)
    b_new = ax_d.barh(y_pos - h_bar, lnk_new, h_bar, color="#0d9488", edgecolor="#115e59", linewidth=1.1, label="New piecewise-linear surface field ($B_1$ core + $B_2$ collar)", zorder=4)

    for bars, col in [(b_old, "#991b1b"), (b_noj, "#334155"), (b_new, "#065f46")]:
        for bar in bars:
            w = bar.get_width()
            y_c = bar.get_y() + bar.get_height() / 2
            if w >= 0:
                ax_d.text(w + 0.55, y_c, f"{w:+.1f}", va="center", ha="left", fontsize=8.1, fontweight="bold", color=col)
            else:
                ax_d.text(w - 0.55, y_c, f"{w:+.1f}", va="center", ha="right", fontsize=8.1, fontweight="bold", color=col)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(sites_lbl, fontsize=8.7, fontweight="bold", color="#1e293b")
    ax_d.set_xlim(-16.5, 36.5)
    ax_d.set_ylim(-0.55, 5.25)
    ax_d.set_xlabel(r"Site Association Constant $\ln(K_{\mathrm{site}} / \mathrm{\AA}^3)$", fontsize=9.8, fontweight="bold", color="#1e293b")
    ax_d.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.95)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # Save to all output locations
    out_dirs = [
        "/home/xvg/tetramer-mc/docs/figures",
        "/home/xvg/tetramer-mc/.claude/worktrees/oligomer-capture/docs/figures",
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            p1 = os.path.join(d, f"surface_field_many_body_hamiltonian.{ext}")
            p2 = os.path.join(d, f"ribbon_1d_vs_3d_lateral_frustration.{ext}")
            fig.savefig(p1, dpi=300)
            shutil.copyfile(p1, p2)

    viewer_dir = "/vast/xvg/tetramer-mc-runs/morph-growth-20261012/viewer"
    if os.path.exists(viewer_dir):
        shutil.copyfile(
            "/home/xvg/tetramer-mc/docs/figures/surface_field_many_body_hamiltonian.png",
            os.path.join(viewer_dir, "surface_field_many_body_hamiltonian.png"),
        )
        shutil.copyfile(
            "/home/xvg/tetramer-mc/docs/figures/surface_field_many_body_hamiltonian.png",
            os.path.join(viewer_dir, "ribbon_1d_vs_3d_lateral_frustration.png"),
        )
    print("Saved figure to /home/xvg/tetramer-mc/docs/figures/surface_field_many_body_hamiltonian.{png,pdf,svg}")


if __name__ == "__main__":
    main()
