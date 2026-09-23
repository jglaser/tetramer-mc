#!/usr/bin/env python3
"""Summarize authenticated completed guide comparisons without new sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGIONS = (
    "old_R5_intersection_native",
    "remaining_R4_native",
    "contact_no_native_entry",
)
LABELS = ("Native inside R5", "Native outside R5", "Competing contact")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(comparison, diagnostic, authentication, out):
    if out.exists():
        raise ValueError("Refuse to overwrite a completed or partial report")
    auth, pilot, noise = map(read, (authentication, comparison, diagnostic))
    verification_path = diagnostic.parent / "verification.json"
    verification = read(verification_path)
    if not all(x["complete"] for x in (auth, pilot, noise, verification)):
        raise ValueError("Expected completed source artifacts")
    if digest(comparison) != auth["analysis_sha256"]:
        raise ValueError("Completed-pilot authentication does not match analysis")
    if digest(comparison) != verification["comparison_sha256"]:
        raise ValueError("Moment reconstruction used a different comparison")
    if digest(diagnostic) != verification["diagnostic_sha256"]:
        raise ValueError("Moment diagnostic changed after reconstruction check")
    if not verification["ordinary_Q0_reproduces_archived_results"]:
        raise ValueError("Hard-only reconstruction failed")
    if not verification["all_original_denominators_retained"]:
        raise ValueError("Diagnostic did not retain the original denominators")
    if any(noise[k] != 0 for k in ("new_physical_draws", "classifier_calls", "audits_replayed")):
        raise ValueError("Expected read-only diagnostic of saved results")
    for path, expected in noise["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Changed diagnostic input: {path}")
    if pilot["total_unconditional_draws"] != auth["unconditional_attempts"]:
        raise ValueError("Attempt accounting disagrees with authentication")

    ratios = {}
    for region in REGIONS:
        rates = {arm: pilot["arms"][arm]["observed_importance_ESS_per_cpu_second"][region]
                 for arm in ("bank", "smc")}
        ratios[region] = {
            "observed_importance_ESS_per_cpu_ratio": rates["smc"] / rates["bank"],
            "common_target_second_moment_ratios": {
                arm: noise["arms"][arm]["aggregate"]["groups"][region]["guides"]["smc"]
                for arm in ("bank", "smc")},
        }
    failed_strata = [s for s in pilot["convergence"]["significant_strata"] if not s["passed"]]
    summary = {
        "schema": "completed-smc-guide-pilot-report-v1", "complete": True,
        "scope": "Fixed R4/scaffold importance sampling; not trajectory mixing or assembly evidence.",
        "new_physical_draws": 0, "audits_replayed": 0, "classifiers_rerun": 0,
        "source_attempts": pilot["total_unconditional_draws"],
        "pilot_diagnostics_passed": pilot["convergence"]["pilot_diagnostics_passed"],
        "full_vessel_gate_open": False, "assembly_stability_established": False,
        "failed_checks": [k for k, v in pilot["convergence"]["checks"].items() if not v],
        "failed_significant_strata": failed_strata,
        "significant_strata_count": len(pilot["convergence"]["significant_strata"]),
        "efficiency_diagnostics": ratios,
        "hard_only_native_comparison": noise["hard_only_arm_comparisons"]["registered_native_entry"],
        "conditional_free_energy_intervals": pilot["convergence"]["free_energy_intervals"],
        "input_sha256": {str(p): digest(p) for p in
                         (comparison, diagnostic, authentication, verification_path, Path(__file__).resolve())},
        "verified_diagnostic_inputs": len(noise["input_sha256"]),
    }
    out.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4))
    x = np.arange(len(REGIONS))
    observed = [ratios[r]["observed_importance_ESS_per_cpu_ratio"] for r in REGIONS]
    axes[0].bar(x, observed, color=["#3078a6", "#3078a6", "#bb845b"], width=.56)
    for i, value in enumerate(observed):
        axes[0].text(i, value + .065, f"{value:.2f}×", ha="center", fontsize=11)
    axes[0].set_title("Observed importance ESS per CPU\nHigher is favorable", fontsize=11)
    axes[0].set_ylabel("84-component / 80-component guide")
    for arm, dx, color, label in (("bank", -.10, "#3078a6", "80-component source"),
                                 ("smc", .10, "#bd5a30", "84-component source")):
        values = [ratios[r]["common_target_second_moment_ratios"][arm]
                  ["two_cloud_noisy"]["ratio_to_bank"] for r in REGIONS]
        axes[1].scatter(x + dx, values, s=65, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            axes[1].annotate(f"{value:.2f}", (i + dx, value), xytext=(0, 10 if dx < 0 else -18),
                             textcoords="offset points", ha="center", fontsize=10, color=color)
    axes[1].set_title("Second moments evaluated on the same poses\nLower is favorable", fontsize=11)
    axes[1].set_ylabel("M₂(84-component) / M₂(80-component)")
    axes[1].legend(loc="lower left", fontsize=9)
    for ax in axes:
        ax.axhline(1, color="#444444", ls="--", lw=1)
        ax.set_xticks(x, ["Native\ninside R5", "Native\noutside R5", "Competing\ncontact"])
        ax.set_ylim(0, 2.5)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Native coverage improves; the competing-contact bottleneck remains", fontsize=14)
    fig.text(.5, .04, "Four fresh populations × 65,536 attempts per arm. Fixed region and physical measure.\n"
             "No confidence intervals shown: rare contributions still destabilize moment estimates.\n"
             "The observed competing-contact ESS gain is not supported by the paired proposal comparison.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.79, bottom=.27, left=.08, right=.98, wspace=.28)
    for extension in ("png", "svg"):
        fig.savefig(out / f"guide-efficiency-diagnostic.{extension}", dpi=180)
    plt.close(fig)

    lines = ["# Completed frozen guide pilot", "",
             "All 524,288 attempts, eight populations, density/Jacobian audits and first-pass classifications completed.",
             "The prescribed convergence checks **failed**. Full-vessel and assembly production remain gated.", "",
             "| Region | Bank log Qz | New guide log Qz | Observed ESS/CPU ratio |",
             "|---|---:|---:|---:|"]
    for region, label in zip(REGIONS, LABELS):
        masses = [pilot["arms"][a]["estimates"][region]["population_uncertainty"]["log_Qz"]
                  for a in ("bank", "smc")]
        lines.append(f"| {label} | {masses[0]:.6f} | {masses[1]:.6f} | "
                     f"{ratios[region]['observed_importance_ESS_per_cpu_ratio']:.3f} |")
    lines += ["", "These ratios describe importance weights, not independent trajectory contacts or mixing times.", "",
              "For the same source poses, the new/old second-moment ratio is about 1.97 for competing contacts in both arms. "
              "The apparent competing-contact ESS gain therefore cannot establish an efficiency gain. "
              "The absolute second moments are themselves unstable. Native ratios are favorable in both source arms.", "",
              "Failed checks: " + ", ".join(summary["failed_checks"]) + ".", "",
              "| Unstable region | Stratum | Bank − new log mass | Combined population SE |",
              "|---|---|---:|---:|"]
    for s in failed_strata:
        lines.append(f"| {s['region']} | {s['family']} {s['bin']} | "
                     f"{s['log_left_minus_right']:.6f} | {s['combined_population_log_delta_SE']:.6f} |")
    q0 = summary["hard_only_native_comparison"]
    lines += ["", f"Hard-only native mass differs by {q0['ordinary']['smc_minus_bank_log_mean']:.6f} log units: "
              f"{q0['ordinary']['population_standard_errors']:.2f} combined population SE, versus "
              f"{q0['ordinary']['row_standard_errors']:.2f} row SE. "
              "The uniform-branch Horvitz–Thompson control reproduces the difference without Gaussian density weights. "
              "Four-population scatter may be understating uncertainty; this diagnosis does not override the failed check.", "",
              "The conditional native/competing estimates remain near −18.7 kBT. "
              "Their small population intervals do not establish convergence, complete vessel coverage or finite-system assembly.", "",
              "No physical sampling, classification or density audit was rerun to produce this report."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    summary["files"] = {p.name: digest(p) for p in sorted(out.iterdir())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "diagnostic", "authentication", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(*(getattr(args, name).resolve() for name in ("comparison", "diagnostic", "authentication", "out")))
