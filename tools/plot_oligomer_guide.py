#!/usr/bin/env python3
"""Plot the frozen-state guide pilot; no new sampling or inferred speedup."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("runs/oligomer-guide-pilot-20260925/analysis.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/oligomer-guide-pilot"))
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete"] or data["failures"]:
        raise ValueError("An incomplete allocation must not be plotted as the completed pilot")
    rows = {(r["preparation"], r["steps"]): r for r in data["reports"]}
    steps = [0, 1, 4, 16]
    colors = ["#286e9b", "#cb7637"]
    x = np.arange(4)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    for state, offset, color, label in zip(
        ["early600", "stalled27000"], [-.19, .19], colors, ["Early snapshot", "Stalled snapshot"]
    ):
        rs = [rows[state, s] for s in steps]
        cpu = [sum(r["population_statistics"]["sampler_cpu_seconds"]["values"]) for r in rs]
        points = [sum(c.get("gate_raw_points", 0) for c in r["counts"]) for r in rs]
        ratios = [v / cpu[0] for v in cpu]
        for ax, values in zip(axes[:2], [ratios, [v/1e6 for v in points]]):
            bars = ax.bar(x+offset, values, width=.36, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=8, padding=3)
        bars = axes[2].bar(x+offset, [r["attachments"] for r in rs], width=.36, color=color)
        axes[2].bar_label(bars, fontsize=9, padding=3)
        if any(r["detachments"] or r["exchanges"] for r in rs):
            raise ValueError("Update the event panel for a campaign containing detachments or exchanges")
    for ax in axes:
        ax.set_xticks(x, ["Baseline", "1", "4", "16"])
        ax.set_xlabel("Guide inner steps")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
        ax.margins(y=.2)
    axes[0].set_title("Cost of complete phases", loc="left", fontweight="bold")
    axes[0].set_ylabel("CPU relative to own baseline")
    axes[0].axhline(1, color=".3", lw=.8, ls="--")
    axes[1].set_title("Physical depletion work", loc="left", fontweight="bold")
    axes[1].set_ylabel("Generated Poisson points (millions)")
    axes[2].set_title("Realized contact changes", loc="left", fontweight="bold")
    axes[2].set_ylabel("Accepted attachments per 64 phases")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_ylim(0, 3.1)
    axes[2].text(.5, .92, "No detachments or exchanges", transform=axes[2].transAxes,
                 ha="center", fontsize=9, color=".3")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Joint oligomer guide: cheaper rejection, no demonstrated docking gain", x=.065,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(.065, .10, "4 streams × 16 reset phases per arm and snapshot; 512 phases total. Cost ratios are descriptive, not mixing speedups.", fontsize=9)
    fig.text(.065, .045, "The extra attachment at 1 and 4 steps is the same paired uniform-branch event. No learned-map attachment was accepted.", fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, bottom=.27, top=.79, wspace=.34)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(args.out.with_suffix(suffix), dpi=180, facecolor="white")
    svg = args.out.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
