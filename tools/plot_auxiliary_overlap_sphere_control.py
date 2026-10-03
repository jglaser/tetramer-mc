#!/usr/bin/env python3
"""Render the frozen one-step sphere control; never sample or rerun its audit."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ANALYSIS_SHA256 = "c96a6d32f00fe2d6e23034049b48899f94442f3e2db29e5995d3b269512b52cd"
RECEIPT_SHA256 = "491242b900cba99df10ecc877694eefa43cc339bcac6ce3821958b47bf652df9"
REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results/auxiliary-overlap-sphere-control-20261003"


def read_bound_json(path, expected_sha256):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen input SHA256 mismatch: {path}")
    return json.loads(raw)


def style_axis(ax):
    ax.axvline(0, color="#5b6470", linewidth=1, linestyle="--", zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e5e8ec", linewidth=0.7)


def plot(analysis_path, output_path):
    report = read_bound_json(analysis_path, ANALYSIS_SHA256)
    receipt = read_bound_json(analysis_path.parent / "completed-review.json", RECEIPT_SHA256)
    if not (report["complete"] and report["passed"] and receipt["complete"] and receipt["passed"]):
        raise ValueError("The saved analysis and receipt must be complete and passed")
    if receipt["output_hashes"]["analysis.json"] != ANALYSIS_SHA256:
        raise ValueError("Receipt does not bind the plotted analysis")
    summary = report["summary"]
    primary = {(item["m"], item["observable"]): item for item in summary["primary_tests"]}
    colors = {1: "#226b9b", 4: "#bd6a16"}
    metrics = [
        ("separation", "A   Root–child separation", 1000, "Mean shift (10⁻³ length units)"),
        ("close_contact", "B   Close-contact probability", 100, "Mean shift (percentage points)"),
        ("overlap_count", "C   Guidance overlap count", 1, "Mean shift (cloud points)"),
        ("root_w2", "D   Root quaternion w²", 1000, "Mean shift (10⁻³; dimensionless)"),
    ]
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10}):
        fig = plt.figure(figsize=(12, 7), facecolor="white")
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.18),
                               left=0.07, right=0.97, bottom=0.25, top=0.79,
                               wspace=0.42, hspace=0.92)
        for index, (observable, title, scale, xlabel) in enumerate(metrics):
            ax = fig.add_subplot(grid[index // 2, index % 2])
            limit = 0
            for m, y in ((1, 1), (4, 0)):
                item = primary[m, observable]
                mean, half_width = scale * item["mean"], scale * 1.96 * item["se"]
                ax.errorbar(mean, y, xerr=half_width, fmt="o", color=colors[m],
                            capsize=4, markersize=6, linewidth=1.7, zorder=3)
                limit = max(limit, abs(mean) + half_width)
            style_axis(ax)
            ax.set(xlim=(-1.15 * limit, 1.15 * limit), ylim=(-0.6, 1.6),
                   yticks=[1, 0], yticklabels=["m = 1", "m = 4"], xlabel=xlabel)
            ax.set_title(title, loc="left", pad=10)

        ax = fig.add_subplot(grid[:, 2])
        negative = summary["negative_control"]
        mean, half_width = 100 * negative["mean"], 100 * 1.96 * negative["se"]
        ax.errorbar(mean, 0, xerr=half_width, fmt="o", color="#b63b47",
                    capsize=5, markersize=7, linewidth=2, zorder=3)
        style_axis(ax)
        ax.set(xlim=(-4, 34), ylim=(-1.4, 1.4), yticks=[],
               xlabel="Mean close-contact shift\n(percentage points; separate scale)")
        ax.set_title("E   Omitted-correction control", loc="left", pad=10)
        ax.text(0.03, 0.84, "Wrong m = 4\nSame candidate and MH uniform;\nauxiliary ratio omitted.",
                transform=ax.transAxes, va="top", fontsize=10, linespacing=1.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        ax.text(0.5, 0.31, f"+{mean:.2f} percentage points\nz = {negative['z']:.2f}",
                transform=ax.transAxes, ha="center", va="top", color="#a6313d",
                fontsize=12, linespacing=1.6)
        fig.suptitle("Auxiliary overlap threshold: one-step hard-sphere control",
                     x=0.07, y=0.96, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.07, 0.89,
                 f"Corrected arms: {summary['independent_sources']:,} independent sources each; "
                 "fixed guidance cloud; zero bath activity.", fontsize=11)
        max_z = max(abs(item["z"]) for item in primary.values())
        fig.text(0.07, 0.085,
                 "Points: paired retained − source means. Bars: mean ± 1.96 SE "
                 "(unadjusted normal intervals).\n"
                 f"All 8 corrected tests passed the frozen Bonferroni gate "
                 f"(family α = 0.01; max |z| = {max_z:.3f}). No trajectory ESS or protein inference.",
                 fontsize=10, color="#38414b", linespacing=1.6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, metadata={
            "Title": "Auxiliary overlap sphere control",
            "Description": "Frozen paired summary statistics; bars are mean +/- 1.96 SE.",
            "Analysis SHA256": ANALYSIS_SHA256,
            "Completed receipt SHA256": RECEIPT_SHA256,
        })
        plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=RUN / "analysis.json")
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/figures/auxiliary-overlap-sphere-control.png")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must name a PNG file")
    plot(args.analysis, args.output)
