#!/usr/bin/env python3
"""Plot independently estimated region weights without implying global coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    uniform = read("runs/native-region-reference-8x4194304-l64-20260920/assessment-streaming.json")
    guided = read("runs/basin-normalizer-importance-guided-16384-l64/assessment/analysis.json")
    ball = read("runs/latent-region-uniform-ball-16384-l64/assessment/analysis.json")
    shells = read("runs/basin-normalizer-mis-refined-16384-l64/assessment/deep-shells.json")
    certificate = read("runs/frozen-deep-region-neighbor2-certificate-v2/results.json")
    native_manifest = read("runs/native-region-reference-8x4194304-l64-20260920/runs/r00/manifest.json")
    region = read("runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json")
    guided_group = guided["groups"]["1.0"]
    assert native_manifest["activity"] == guided_group["activity"] == region["activity"] == .035
    assert native_manifest["depletant_radius"] == region["depletant_radius"] == 1.5
    assert native_manifest["shape_sha256"] == region["shape_sha256"]
    assert uniform["complete"] and uniform["all_rows_and_hashes_validated"]
    assert not guided["pending"]
    assert all(r["certified_entire_ball"] and r["unresolved_cells"] == 0
               for r in certificate["results"])
    assert {r["radius"] for r in certificate["results"]} == {3., 5., 8.}

    rows = []

    def row(label, group, estimate, populations, method):
        logq = estimate.get("logQ", estimate.get("log_normalizer"))
        rse = estimate.get("relative_se", estimate.get("relative_SE"))
        assert logq is not None and 0 <= rse < 1
        rows.append(dict(label=label, group=group, method=method, logQ=logq,
                         relative_SE=rse, ess=estimate["ess"], populations=populations))

    row("Complete native region\nUniform geometric cover", "native", uniform["regions"]["native"],
        [p["regions"]["native"]["log_normalizer"] for p in uniform["populations"]], "uniform")
    row("Complete native region\nFrozen Gaussian guide", "native", guided_group["estimates"]["native"],
        [p["estimates"]["native"]["logQ"] for p in guided["populations"]], "guided")
    row("Competing region, radius 3\nUniform latent ball", "deep", ball["estimate"],
        [p["estimate"]["logQ"] for p in ball["populations"]], "uniform")
    for radius in [3, 8]:
        key = f"r_le_{radius}"
        row(f"Competing region, radius {radius}\nFrozen MIS-refined guide", "deep", shells["estimates"][key],
            [p["estimates"][key]["logQ"] for p in shells["populations"]], "guided")

    fig, ax = plt.subplots(figsize=(10.7, 6.7))
    fig.subplots_adjust(left=.31, right=.96, top=.84, bottom=.23)
    colors = {"native": "#246a9a", "deep": "#b86721"}
    for index, item in enumerate(rows):
        y = len(rows) - 1 - index
        # Bars are Q +/- one observed SE, transformed onto the logarithmic axis.
        rse = item["relative_SE"]
        error = [[-math.log1p(-rse)], [math.log1p(rse)]]
        ax.scatter(item["populations"], y + np.linspace(-.10, .10, len(item["populations"])),
                   c="#9a9a9a", s=21, alpha=.75, zorder=2)
        ax.errorbar(item["logQ"], y, xerr=error, marker="D" if item["method"] == "uniform" else "o",
                    color=colors[item["group"]], mfc=colors[item["group"]], ms=7,
                    capsize=4, lw=1.9, zorder=3)
        ax.text(22.22, y, f'{item["logQ"]:.3f}\nESS {item["ess"]:.0f}',
                fontsize=9, va="center", color=colors[item["group"]])
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]], fontsize=10)
    ax.set_xlim(13.55, 23.02)
    ax.set_ylim(-.48, len(rows) - .52)
    ax.set_xticks(np.arange(14, 23))
    ax.set_xlabel(r"Regional statistical weight, $\log(Q/\mathrm{\AA}^3)$", fontsize=11)
    ax.grid(axis="x", color="#e6e6e6", lw=.7)
    ax.axhline(2.5, color="#d7d7d7", lw=.8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.suptitle("Strong isolated contacts can be incompatible with a second neighbor", x=.04,
                 ha="left", y=.975, fontsize=15, weight="bold")
    fig.text(.04, .919, "One fixed neighbor · depletant radius 1.5 Å · activity 0.035 Å⁻³ · normalized rotational measure",
             fontsize=10, color="#444444")
    fig.text(.04, .115, "Adding the second prescribed native neighbor hard-excludes the entire competing radius-8 region.",
             fontsize=10, weight="bold")
    fig.text(.04, .074, "Bars: observed ±1 SE in Q. Gray points: independent population estimates. ESS describes integration weights.\n"
             "The native uniform estimate remains concentrated in rare poses. Other competing regions and the cost of forming\n"
             "the prescribed neighbors remain unresolved; this is not a global equilibrium or assembly conclusion.",
             fontsize=9, color="#444444", va="top", linespacing=1.3)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / f"regional-contact-evidence.{suffix}", dpi=180)
    plt.close(fig)
    gap = rows[-1]["logQ"] - rows[0]["logQ"]
    (out / "data.json").write_text(json.dumps(dict(
        sources=sources, rows=rows, log_competing_R8_over_native=gap,
        competing_R8_over_native=math.exp(gap),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Fixed regions only; finite-sample errors are not bounds on unseen weight. No global native probability."
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
