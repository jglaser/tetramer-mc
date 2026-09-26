#!/usr/bin/env python3
"""Illustrate cooperative proposal construction; synthetic coordinates only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch


def ellipse(ax, mean, cov, color, label, fill=False):
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, -1]
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    ax.add_patch(Ellipse(mean, 4*np.sqrt(values[-1]), 4*np.sqrt(values[0]),
                         angle=angle, edgecolor=color,
                         facecolor=color if fill else "none", alpha=.22 if fill else 1,
                         linewidth=2.4, label=label))
    ax.plot(*mean, "o", color=color, markersize=5)


def box(ax, xy, width, height, text, color="#eaf1f5"):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012",
                              facecolor=color, edgecolor="#8093a1", linewidth=1.1))
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha="center", va="center", fontsize=11)


def main():
    out = Path(__file__).resolve().parents[1]/"docs"/"assets"
    out.mkdir(exist_ok=True)
    mean1 = np.array([-.35, .10]); mean2 = np.array([.45, -.15])
    cov1 = np.array([[1.2, .82], [.82, .68]])
    cov2 = np.array([[.88, -.64], [-.64, .66]])
    p1, p2 = np.linalg.inv(cov1), np.linalg.inv(cov2)
    fused = np.linalg.inv(p1+p2)
    center = fused@(p1@mean1+p2@mean2)
    fig = plt.figure(figsize=(14, 6.8), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.45])
    ax = fig.add_subplot(grid[0])
    ellipse(ax, mean1, cov1, "#c97735", "Member 1 contact")
    ellipse(ax, mean2, cov2, "#426da1", "Member 2 contact")
    ellipse(ax, center, fused, "#147f70", "Fused cooperative basin", True)
    ax.set(xlim=(-3, 3), ylim=(-2.5, 2.5), xlabel="Collective translation (illustrative)",
           ylabel="Collective rotation (illustrative)",
           title="A   Constraints on the same rigid pose")
    ax.set_aspect("equal"); ax.grid(alpha=.13)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(.5, -.23, "Gaussian fusion is exact for affine factors.\nRigid-pose pullbacks are locally approximated.",
            transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax = fig.add_subplot(grid[1]); ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    ax.set_title("B   One reversible map for the whole oligomer", loc="left")
    box(ax, (.08,.80), .84,.14, "Fixed internal member geometry + stationary spectators")
    box(ax, (.08,.57), .84,.15, "Exact original charts + compatible two-contact Gaussians\nOne normalized catalogue, identical on reversal")
    box(ax, (.03,.29), .40,.15, "Source basin\nWhiten collective pose")
    box(ax, (.57,.29), .40,.15, "Destination basin\nDecode collective pose")
    box(ax, (.08,.03), .84,.16, "Carry all members rigidly\nFull catalogue + anchor correction + many-body depletion", "#e7f3ee")
    arrow = dict(arrowstyle="->", color="#526575", lw=1.8)
    ax.annotate("", (.50,.73), (.50,.80), arrowprops=arrow)
    ax.annotate("", (.23,.45), (.36,.57), arrowprops=arrow)
    ax.annotate("", (.77,.45), (.64,.57), arrowprops=arrow)
    ax.annotate("", (.565,.36), (.435,.36), arrowprops=dict(arrowstyle="<->", color="#147f70", lw=2))
    ax.text(.5,.25,"Invertible latent update",ha="center",fontsize=10,color="#147f70")
    ax.annotate("", (.50,.20), (.77,.29), arrowprops=arrow)
    fig.suptitle("Merge contact basins, then transport a single six-dimensional rigid pose", fontsize=18)
    for suffix in ["png","svg"]:
        fig.savefig(out/f"oligomer-fusion-schematic.{suffix}", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
