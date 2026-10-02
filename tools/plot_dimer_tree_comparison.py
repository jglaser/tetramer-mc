#!/usr/bin/env python3
"""Plot the frozen sphere controls; no protein-assembly inference."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = json.loads((ROOT/"results/dimer-tree-equilibrium-20261002/analysis.json").read_text())
    comparison = json.loads((ROOT/"results/dimer-tree-comparison-20261002/analysis.json").read_text())
    groups = [
        (original, "poisson", "Involutive\nworld bath", "#7e8796"),
        (comparison, "tree:singleton_path", "Involutive\ntwo-leg bath", "#237faa"),
        (comparison, "defensive_independent:poisson", "Defensive redraw\nworld bath", "#cf7938"),
        (comparison, "defensive_independent:singleton_path", "Defensive redraw\ntwo-leg bath", "#3b8b64"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
    target = original["reference"]["contact"]
    axes[0].axhline(target, color="#384355", ls="--", lw=1.2,
                   label=f"Exact reference: {target:.3f}")
    for x, (data, key, _, color) in enumerate(groups):
        stats = data["arms"][key]["contact"]
        axes[0].errorbar(x, stats["mean"], yerr=3.182446305*stats["population_se"],
                        fmt="o", color=color, capsize=4, ms=7)
        chains = [c for c in data["chains"] if c.get("analysis_group", c["arm"]) == key]
        values = [c["contact_ess_per_cpu"] for c in chains]
        axes[1].scatter(x+np.linspace(-.12, .12, len(values)), values, color=color, s=37)
        total = (sum(c["observables"]["contact"]["batch_1024"]["ess"] for c in chains)
                 / sum(c["production_cpu_seconds"] for c in chains))
        axes[1].plot([x-.22, x+.22], [total, total], color=color, lw=2)
    for ax in axes:
        ax.set_xticks(range(4), [g[2] for g in groups], fontsize=9)
        ax.set_xlim(-.5, 3.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Equilibrium occupancy", loc="left", weight="bold")
    axes[0].set_ylabel("Probability of d < 1")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_title("Threshold-contact sampling efficiency", loc="left", weight="bold")
    axes[1].set_ylabel("Estimated contact ESS / CPU second")
    axes[1].set_yscale("log")
    fig.suptitle("Two-sphere control: same target, four independent streams per method", fontsize=13)
    fig.supxlabel("Left: population 95% t intervals. Right: individual streams and summed ESS / summed CPU.\n"
                  "All rejected states retained; d = 1 crossings are not full binding/unbinding events.", fontsize=9)
    for suffix in ["svg", "png"]:
        fig.savefig(ROOT/f"docs/assets/dimer-tree-equilibrium-comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
