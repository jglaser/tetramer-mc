#!/usr/bin/env python3
"""Compare separately analyzed importance proposals on the same physical target.

No samples are pooled and no missing region is assigned zero physical mass.
Error bars are observed-sample log-scale standard errors, not coverage bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_signature(root):
    config = read(root / "provenance/config.json")
    keys = ("capture_center", "capture_radius", "fixed_poses", "depletant_radius", "reservoir_density")
    target = {key: config[key] for key in keys}
    target["shape_sha256"] = digest(root / "provenance/shape.json")
    target["regions"] = {key: config["metadata"][key] for key in
                         ("native_poses", "rigid_members", "member_error_scale", "angle_error_scale_deg")}
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", nargs=2, metavar=("LABEL", "DIRECTORY"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smc-audit", type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    target = None
    for label, directory in args.campaign:
        root = Path(directory).resolve()
        current = physical_signature(root)
        if target is None:
            target = current
        if current != target:
            raise ValueError(f"Physical target or region definition differs: {root}")
        analysis = read(root / "assessment/analysis.json")
        if analysis["pending"]:
            raise ValueError(f"Incomplete campaign: {root}")
        for scale, group in sorted(analysis["groups"].items(), key=lambda item: float(item[0])):
            rows.append(dict(label=label if len(analysis["groups"]) == 1 else f"{label}, s={scale}",
                             root=str(root), analysis_sha256=digest(root / "assessment/analysis.json"),
                             manifest_sha256=digest(root / "manifest.json"),
                             draws=group["unconditional_draws"], populations=group["populations"],
                             model_sha256=group["model_sha256"], estimates=group["estimates"]))

    smc = read(args.smc_audit) if args.smc_audit else None
    if smc:
        # This reference is specifically the archived site0 calculation. The
        # separate audit verifies all other endpoints lie in the distant bin.
        site = smc["sites"]["site0"]
        if not np.isclose(site["other_adsorbed"]["normalizer_weighted_q_mass"][-1], 1., rtol=0, atol=1e-14):
            raise ValueError("Cannot label the archived other normalizer as terminal-filtered distant mass")
        config_path = Path(smc["source"]) / "configs/site0-m1-r1.5-z0.035-native-00.json"
        if digest(config_path) != smc["provenance"][str(config_path)]:
            raise ValueError("Archived SMC reference config changed")
        config = read(config_path)
        environment = read(config["environment"])
        reference_target = {key: environment[key] for key in ("capture_center", "capture_radius", "fixed_poses")}
        reference_target.update({key: config[key] for key in ("depletant_radius", "reservoir_density")})
        reference_target["shape_sha256"] = digest(config["shape"])
        reference_target["regions"] = {key: environment[key] for key in ("native_poses", "rigid_members")}
        reference_target["regions"].update({key: config[key] for key in ("member_error_scale", "angle_error_scale_deg")})
        if reference_target != target:
            raise ValueError("Archived SMC reference differs from importance target")
    colors = ["#177e89", "#c97820", "#6750a4", "#666666"]
    region_keys = ["native", "shoulder", "distant", "intermediate"]
    region_titles = ["Native: q ≤ 1", "Shoulder: 1 < q < 2", "Distant: q ≥ 5", "Intermediate: 2 ≤ q < 5"]
    fig, axes = plt.subplots(1, 4, figsize=(17, max(4.8, .57 * len(rows) + 1.6)), sharey=True)
    for ax, key, title, color in zip(axes, region_keys, region_titles, colors):
        for i, row in enumerate(rows):
            value = row["estimates"][key]
            if value["logQ"] is None:
                ax.text(.02, i, "unobserved", transform=ax.get_yaxis_transform(), va="center", color="#777777", fontsize=9)
                continue
            error = value["relative_se"]
            # Delta-method error of log Q. At large RSE this is only a visual
            # diagnostic; the table retains the original-scale RSE explicitly.
            ax.errorbar(value["logQ"], i, xerr=error, fmt="o", color=color, capsize=3)
        if smc and key in ("native", "distant"):
            reference = site["native" if key == "native" else "other_adsorbed"]
            ax.axvline(reference["logQ"], color="#444444", linestyle="--", linewidth=1.1)
            ax.axvspan(reference["logQ"] - reference["relative_standard_error"],
                       reference["logQ"] + reference["relative_standard_error"], color="#777777", alpha=.12)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("log(Q / Å³), normalized Haar")
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(rows)), [f"{row['label']}\nN={row['draws']:,}" for row in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Independent physical weights remain proposal-sensitive", fontsize=15)
    fig.text(.5, .01, "Bars: observed ±1 SE on log scale (delta approximation), not missing-mode bounds. Dashed: archived SMC; no samples pooled.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"guide-comparison.{extension}", dpi=180)
    plt.close(fig)
    payload = dict(target=target, campaigns=rows,
                   comparison_script_sha256=digest(__file__),
                   reference_audit=str(args.smc_audit.resolve()) if args.smc_audit else None,
                   reference_sha256=digest(args.smc_audit) if args.smc_audit else None,
                   uncertainty="Observed-sample SE does not cover unobserved modes. No pooling or equilibrium convergence claim.")
    (args.out / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen proposal sensitivity", "", payload["uncertainty"], "",
             "| Proposal | Draws | Region | log Q | Observed relative SE | ESS | Largest weight share |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        for key in region_keys:
            value = row["estimates"][key]
            fields = ["unobserved", "unresolved", "0", "—"] if value["logQ"] is None else [
                f"{value['logQ']:.4f}", f"{value['relative_se']:.1%}", f"{value['ess']:.2f}", f"{value['max_fraction']:.1%}"]
            lines.append(f"| {row['label']} | {row['draws']} | {key} | " + " | ".join(fields) + " |")
    lines += ["", "![Independent guide comparison](guide-comparison.png)", "",
              "All four registration regions include bound and unbound configurations. The archived distant reference is explicitly terminal-filtered q≥5, not a redefinition of the old q>1 target.", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(args.out.resolve()), campaigns=len(rows)), indent=2))


if __name__ == "__main__":
    main()
