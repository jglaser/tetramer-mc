#!/usr/bin/env python3
"""Explain the implemented two-contact proposal with synthetic schematic geometry."""

from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
INK = "#203342"
MUTED = "#536574"
ORANGE = "#d9792c"
BLUE = "#337eae"
GREEN = "#157f72"
GRAY = "#b7c0c7"
PALE = "#f2f6f8"


def label(ax, x, y, text, size=11, color=INK, **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, **kwargs)


def arrow(ax, a, b, color=MUTED, lw=1.6, style="-|>", curve=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14,
                               linewidth=lw, color=color,
                               connectionstyle=f"arc3,rad={curve}"))


def body(ax, center, radius, color, name="", halo=True, alpha=1):
    if halo:
        ax.add_patch(Circle(center, radius + .20, facecolor=color, alpha=.13 * alpha,
                            edgecolor=color, linewidth=1))
    ax.add_patch(Circle(center, radius, facecolor=color, alpha=alpha,
                        edgecolor="white", linewidth=1.8, zorder=4))
    if name:
        label(ax, *center, name, size=12, color="white" if color != GRAY else INK,
              ha="center", va="center", weight="bold", zorder=5)


def contact(ax, moving, spectator, moving_radius, spectator_radius, color):
    a, b = np.asarray(moving), np.asarray(spectator)
    unit = (b - a) / np.linalg.norm(b - a)
    a = a + moving_radius * unit
    b = b - spectator_radius * unit
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=3,
            solid_capstyle="round", zorder=6)


def ellipse(ax, mean, covariance, color, scale=2, filled=False):
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    patch = Ellipse(mean, 2 * scale * np.sqrt(values[-1]),
                    2 * scale * np.sqrt(values[0]),
                    angle=np.degrees(np.arctan2(direction[1], direction[0])),
                    facecolor=color if filled else "none", edgecolor=color,
                    linewidth=2.1, alpha=.22 if filled else .95)
    ax.add_patch(patch)
    if filled:
        ax.add_patch(Ellipse(mean, patch.width, patch.height, angle=patch.angle,
                            facecolor="none", edgecolor=color, linewidth=2.3))
    ax.plot(*mean, "o", color=color, markersize=4)


def box(ax, xy, width, height, text, color=PALE, size=11, edge="#d8e1e6"):
    ax.add_patch(FancyBboxPatch(xy, width, height,
                               boxstyle="round,pad=0.012,rounding_size=0.025",
                               facecolor=color, edgecolor=edge, linewidth=1.1))
    label(ax, xy[0] + width / 2, xy[1] + height / 2, text,
          size=size, ha="center", va="center", linespacing=1.45)


def panel_title(fig, x, y, number, title, subtitle):
    fig.text(x, y, f"{number}   {title}", fontsize=16, color=INK, weight="bold")
    fig.text(x, y - .027, subtitle, fontsize=11, color=MUTED)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "mathtext.fontset": "dejavusans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    fig.text(.055, .958, "The two-contact update", fontsize=26, weight="bold", color=INK)
    fig.text(.055, .925,
             "Combine compatible docking preferences into one rigid-oligomer proposal; then test the full physical move.",
             fontsize=13, color=MUTED)

    panel_title(fig, .055, .870, "1", "Two interfaces, one rigid pose",
                "Colored bodies move together; gray spectators stay fixed during this trial.")
    ax = fig.add_axes([.055, .565, .43, .257])
    ax.set(xlim=(0, 8), ylim=(0, 4.2), aspect="equal")
    ax.axis("off")
    first, second, anchor = (1.65, 1.4), (3.35, 1.4), (2.5, 2.78)
    ax.plot([first[0], second[0]], [first[1], second[1]], color=INK, lw=4, zorder=2)
    body(ax, anchor, .73, GRAY, "A")
    body(ax, first, .66, ORANGE, "1")
    body(ax, second, .66, BLUE, "2")
    contact(ax, first, anchor, .66, .73, ORANGE)
    contact(ax, second, anchor, .66, .73, BLUE)
    label(ax, 1.15, 2.40, "contact 1", color=ORANGE, size=10, ha="center")
    label(ax, 3.88, 2.40, "contact 2", color=BLUE, size=10, ha="center")
    label(ax, 2.5, .34, "2 carried members → 1 spectator", size=11, ha="center")
    # Alternative interface pattern: one member can satisfy both contacts.
    gray1, gray2, member, carried = (5.61, 2.77), (7.19, 2.77), (6.4, 1.64), (6.4, .50)
    ax.plot([member[0], carried[0]], [member[1], carried[1]], color=INK, lw=3, zorder=2)
    body(ax, gray1, .51, GRAY, "A", halo=False)
    body(ax, gray2, .51, GRAY, "B", halo=False)
    body(ax, member, .57, ORANGE, "1", halo=False)
    body(ax, carried, .43, BLUE, "2", halo=False)
    contact(ax, member, gray1, .57, .51, ORANGE)
    contact(ax, member, gray2, .57, .51, ORANGE)
    label(ax, 6.4, 3.68, "Also allowed", size=11, ha="center", color=MUTED)
    label(ax, 6.4, -.36, "1 member → 2 spectators", size=10, ha="center", color=MUTED)
    fig.text(.055, .527, "Each interface constrains the same collective pose H:  "
             r"$A_a^{-1} H U_i$", fontsize=12, color=INK)
    fig.text(.055, .502, "Uᵢ fixes member i inside the oligomer. Halos represent depletant-exclusion regions.",
             fontsize=10.5, color=MUTED)

    panel_title(fig, .55, .870, "2", "Fuse compatible constraints",
                "A two-dimensional slice of the six-dimensional collective pose space.")
    ax = fig.add_axes([.585, .590, .35, .230])
    m1, m2 = np.array([-.34, .12]), np.array([.35, -.10])
    c1 = np.array([[1.20, .82], [.82, .68]])
    c2 = np.array([[.88, -.64], [-.64, .66]])
    precision1, precision2 = np.linalg.inv(c1), np.linalg.inv(c2)
    fused = np.linalg.inv(precision1 + precision2)
    mean = fused @ (precision1 @ m1 + precision2 @ m2)
    ellipse(ax, m1, c1, ORANGE)
    ellipse(ax, m2, c2, BLUE)
    ellipse(ax, mean, fused, GREEN, filled=True)
    ax.set(xlim=(-2.7, 2.7), ylim=(-1.85, 1.85),
           xlabel="Collective translation", ylabel="Collective rotation")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#becbd3")
    label(ax, 1.00, 1.44, "Contact 1", size=10, color=ORANGE)
    label(ax, -2.48, 1.44, "Contact 2", size=10, color=BLUE)
    label(ax, .75, -.20, "Joint basin", size=11, color=GREEN, weight="bold")
    arrow(ax, (.80, -.03), (.27, .09), GREEN, lw=1)
    fig.text(.55, .535, r"Fit:  $H_* = \arg\min_H\,[\|z_1(H)\|^2+\|z_2(H)\|^2]$",
             fontsize=12, color=INK)
    fig.text(.55, .508, r"Local precision:  $\Sigma_*^{-1}=J_1^{\mathsf{T}}J_1+J_2^{\mathsf{T}}J_2$",
             fontsize=12, color=INK)
    fig.text(.55, .484, "The Gaussian approximates a proposal basin, not its physical statistical weight.",
             fontsize=10.5, color=MUTED)

    # Separator establishes a reading order without suggesting two independent body moves.
    fig.add_artist(plt.Line2D([.055, .95], [.460, .460], transform=fig.transFigure,
                             color="#dbe3e8", lw=1))
    panel_title(fig, .055, .428, "3", "Use the reversible basin map",
                "Single-contact and fused basins share one normalized catalogue G.")
    ax = fig.add_axes([.055, .130, .43, .248])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, (.012, .775), .966, .20,
        r"$G=(1-\rho)G_{\rm single}+\rho G_{\rm fused}$" + "\n"
        "Default ρ = 0.8 when fused basins exist; original mixture retained.", size=11)
    box(ax, (.012, .325), .345, .265,
        "Source basin\nChoose by responsibility\nWhiten the old pose", size=11)
    box(ax, (.635, .325), .345, .265,
        "Destination basin\nChoose by mixture weight\nDecode the new pose", size=11)
    arrow(ax, (.360, .450), (.630, .450), GREEN, lw=2.2, style="<->")
    label(ax, .497, .675, "Correlated latent map", size=11, color=GREEN, ha="center")
    label(ax, .497, .195, "Reverse: swap basin labels and invert the auxiliary-noise map.",
          size=10.5, ha="center", color=MUTED)
    fig.text(.055, .126, "Carry every selected member with the same rigid transformation.",
             fontsize=12, color=INK, weight="bold")
    fig.text(.055, .103, "Source and destination can each be single or fused; keep the defensive uniform branch.",
             fontsize=10.5, color=MUTED)

    panel_title(fig, .55, .428, "4", "Check the complete physical move",
                "A promising fitted center does not guarantee a valid or favorable trial.")
    ax = fig.add_axes([.55, .127, .40, .250])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, (.012, .772), .966, .198,
        "Check all hard cores and the wall.\nRequire unchanged internal contacts and rounded internal key.", size=11)
    box(ax, (.012, .266), .966, .344,
        r"$\log\alpha=\min\{0,\ \log W_{\rm depl}$" + "\n"
        r"$+\log G(X)-\log G(Y)+\log q_a(Y)-\log q_a(X)\}$",
        color="#e9f5f1", edge="#acd4c7", size=11.5)
    arrow(ax, (.495, .767), (.495, .617), GREEN)
    label(ax, .50, .120, "Accept the whole oligomer, or retain its old pose.",
          size=11.5, ha="center", weight="bold")
    fig.text(.55, .125, "Wdepl: full many-body Poisson depletion factor; qₐ: primary-anchor selection.",
             fontsize=10.5, color=MUTED)
    fig.text(.55, .103, "Use the full catalogue density G, including its pose-coordinate Jacobians.",
             fontsize=10.5, color=MUTED)

    fig.text(.055, .048,
             "Same catalogue on reversal: fixed spectators + wall + rounded internal geometry; reject trials that change that key.",
             fontsize=11, color=INK)
    fig.text(.055, .024,
             "Schematic geometry and Gaussian slice only. Two distinct member–spectator interfaces guide the proposal; binding and native registry are not guaranteed.",
             fontsize=10.5, color=MUTED)
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"two-contact-update.{extension}", dpi=180, facecolor="white")
    provenance = {
        "kind": "explanatory schematic; synthetic geometry, no simulation data",
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in ["src/oligomer_proposal.rs", "src/cluster_phase.rs", "src/basin_involution.rs"]},
        "illustrative_gaussians": {"means": [m1.tolist(), m2.tolist()],
                                   "covariances": [c1.tolist(), c2.tolist()],
                                   "fused_mean": mean.tolist(), "fused_covariance": fused.tolist()},
    }
    (OUT / "two-contact-update-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(OUT / "two-contact-update.png")


if __name__ == "__main__":
    main()
