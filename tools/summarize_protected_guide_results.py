#!/usr/bin/env python3
"""Render the completed, authenticated protected-guide results; no physical calls.

Only small cached JSON summaries are read.  This report does not rerun audits,
classifiers, geometry, or fitting, and cannot change any campaign gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "authentication": ("runs/protected-guide-completed-review-20260924/authentication.json", "fcaad551949e38d19f104853e8c8abd3ac527c88532c77e881341be290071d99"),
    "comparison": ("runs/protected-guide-validation-20260923/comparison/analysis.json", "471f61cacf7253bf5746d8fc04765ea71ddfacfb07e205e12e50bf43742737dd"),
    "protocol": ("runs/protected-guide-validation-20260923/protocol.json", "cc924806e65e1fc702e7058403edd43ae102c7e9d51f11bbf04a9311d148aead"),
    "matching": ("runs/protected-guide-matching-review-20260924/review.json", "9aa835f9bc6bbe807e29d560655dea896bffce5899125e56be8751c7313930d4"),
    "noise": ("runs/protected-guide-completed-noise-diagnostic-20260923/analysis.json", "829e4e9fe209d7a01d9d7677127b77092327c91cc82c010c021c95a2b2d1d060"),
}
ARMS = ("bank", "protected", "small", "intensity256")
NATIVE = "registered_native_entry"
INSIDE = "old_R5_intersection_native"
COMPLEMENT = "remaining_R4_native"
NOENTRY = "contact_no_native_entry"
REGIONS = (NATIVE, INSIDE, COMPLEMENT, NOENTRY)
LABEL = {NATIVE: "All native", INSIDE: "Native inside R5", COMPLEMENT: "Native complement", NOENTRY: "Contact without entry"}
ARM_LABEL = {"bank": "Bank 80", "protected": "Protected 84", "small": "Protected 84, smaller N", "intensity256": "Protected 84, intensity 256"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, message: str) -> None:
    check(math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-12), f"{message}: {a} != {b}")


def dump(path: Path, obj: object) -> None:
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_and_validate(root: Path) -> tuple[dict, dict]:
    data, hashes = {}, {}
    for name, (relative, expected) in INPUTS.items():
        path = root / relative
        actual = sha(path)
        check(actual == expected, f"Changed frozen {name}: {path}")
        hashes[str(path)] = actual
        data[name] = json.loads(path.read_text())
    auth, comparison = data["authentication"], data["comparison"]
    check(auth["complete"] and auth["files_verified"] == 682, "Missing completed authentication")
    check(auth["unconditional_attempts"] == 10485760 and auth["populations"] == 16, "Authentication allocation changed")
    check(all(row["terminal"] for row in auth["terminal_handles"]), "Authentication has nonterminal handles")
    for name in ("comparison", "protocol"):
        path = root / INPUTS[name][0]
        check(auth["files"][str(path)] == hashes[str(path)], f"Unbound authentication input {name}")
    check(auth["analysis_sha256"] == INPUTS["comparison"][1], "Analysis authentication mismatch")
    check(comparison["protocol_sha256"] == INPUTS["protocol"][1], "Comparison protocol mismatch")
    for name, field in (("matching", "input_sha256"), ("noise", "source_sha256")):
        check(data[name]["complete"], f"Incomplete {name}")
        check(data[name][field][str(root / INPUTS["comparison"][0])] == INPUTS["comparison"][1], f"Unbound {name} comparison")
    check(comparison["complete"] and comparison["total_unconditional_draws"] == 10485760, "Incomplete comparison")
    check(not comparison["convergence"]["regional_diagnostics_passed"], "Frozen verdict changed")
    check(not auth["full_vessel_gate_open"] and not auth["assembly_stability_established"], "Unexpected promotion")
    return data, hashes


def summarize(data: dict) -> dict:
    comparison, noise, matching = data["comparison"], data["noise"], data["matching"]
    arms, convergence = comparison["arms"], comparison["convergence"]
    out = {
        "schema": "protected-guide-completed-report-v1", "complete": True,
        "attempts": comparison["total_unconditional_draws"], "arms": {},
        "regional_gate": "FAIL", "checks": convergence["checks"],
        "failed_strata": convergence["significant_stratum_disagreements"],
        "significant_stratum_comparisons": len(convergence["significant_strata"]),
        "free_energy_intervals": convergence["free_energy_intervals"],
        "direct_free_energy_comparisons": {key: value["free_energy_contrast"] for key, value in convergence["comparisons"].items()},
        "importance_efficiency": convergence["observed_importance_efficiency"],
        "matching_SMC": matching["matching_mass_comparisons"],
        "matching_SMC_conclusions": matching["conclusions"],
        "matching_SMC_terminal_noentry": matching["terminal_noentry_observations"],
        "unbound_finite_R4_bound": comparison["unbound_finite_region_bound"],
        "selected_cached_tail_rows": noise["selected_tail_rows"],
        "row_SE_diagnostic_only": noise["failed_strata"],
        "new_physical_draws": 0, "new_classifier_calls": 0, "new_geometry_calls": 0,
        "audits_replayed": 0, "fits_performed": 0, "gate_changes": 0,
        "scope": "Fixed scaffold and finite R4. Importance ESS/CPU is not trajectory mixing. No full-vessel or mobile finite-system assembly conclusion.",
        "numeric_checks": [],
    }
    total = 0
    for name in ARMS:
        arm = arms[name]
        n = 1048576 if name in ("bank", "protected") else 262144
        check(arm["allocation"]["samples"] == n and len(arm["populations"]) == 4, f"Allocation {name}")
        check(all(p["samples"] == n for p in arm["populations"]), f"Population N {name}")
        close(arm["sampler_cpu_seconds"], sum(p["sampler_cpu_seconds"] for p in arm["populations"]), f"CPU {name}")
        total += 4 * n
        groups = {}
        for region in REGIONS:
            estimate = arm["estimates"][region]
            rows, pops = estimate["row_uncertainty"], estimate["population_uncertainty"]
            check(rows["draws"] == 4 * n and estimate["population_count"] == 4, "Unconditional denominators")
            close(rows["log_Qz"], pops["log_Qz"], "Row/population linear mean mass")
            close(rows["Qz_ESS"] / arm["sampler_cpu_seconds"], arm["observed_importance_ESS_per_cpu_second"][region], "ESS/CPU reconstruction")
            groups[region] = {
                "log_Qz": rows["log_Qz"], "importance_ESS": rows["Qz_ESS"],
                "largest_contribution_fraction": rows["largest_Qz_fraction"],
                "row_relative_SE": rows["Qz_relative_SE"], "population_relative_SE": pops["Qz_relative_SE"],
                "cloud_variance_fraction": estimate["paired_cloud_noise"]["cloud_fraction"],
                "importance_ESS_per_cpu_second": rows["Qz_ESS"] / arm["sampler_cpu_seconds"],
            }
        fe = convergence["free_energy_intervals"][name]
        delta = -(groups[NATIVE]["log_Qz"] - groups[NOENTRY]["log_Qz"])
        close(delta, fe["beta_F_native_minus_noentry"], "Plotted free energy from masses")
        population = {r: {p["id"]: p["log_Qz"] for p in arm["estimates"][r]["populations"]} for r in (NATIVE, NOENTRY)}
        residuals = [math.exp(population[NATIVE][pid] - groups[NATIVE]["log_Qz"]) - math.exp(population[NOENTRY][pid] - groups[NOENTRY]["log_Qz"]) for pid in sorted(population[NATIVE])]
        se = math.sqrt(sum(x * x for x in residuals) / 12)
        close(se, fe["paired_population_log_ratio_SE"], "Paired population SE reconstruction")
        close(fe["halfwidth_95"], se * fe["Student_t_quantile"], "Population t3 halfwidth")
        for endpoint, sign in zip(fe["interval_95"], (-1, 1)):
            close(endpoint, delta + sign * fe["halfwidth_95"], "Plotted interval endpoint")
        check(fe["degrees_of_freedom"] == 3, "Population interval df")
        strata = {}
        for region, index in ((COMPLEMENT, 55), (NOENTRY, 63), (NOENTRY, 42), (NOENTRY, 62)):
            item = arm["strata"]["orthant"][region][index]
            strata[f"{region}:orthant{index}"] = {"row": item["row_uncertainty"], "population": item["population_uncertainty"], "fraction": math.exp(item["row_uncertainty"]["log_Qz"] - groups[region]["log_Qz"])}
        out["arms"][name] = {"allocation": arm["allocation"], "sampler_cpu_seconds": arm["sampler_cpu_seconds"], "regions": groups, "selected_strata": strata}
    check(total == out["attempts"], "Total original attempts")
    check(len(out["failed_strata"]) == 7 and out["significant_stratum_comparisons"] == 132, "Frozen stratum result changed")
    for comparison_name, regions in out["importance_efficiency"].items():
        left, right = comparison_name.split("/")
        for region in REGIONS:
            ratio = out["arms"][right]["regions"][region]["importance_ESS_per_cpu_second"] / out["arms"][left]["regions"][region]["importance_ESS_per_cpu_second"]
            close(ratio, regions[region]["right_over_left"], "Plotted importance-efficiency ratio")
    for row in out["failed_strata"]:
        check(not row["passed"], "Failure list contains passing comparison")
    contrast = out["direct_free_energy_comparisons"]["protected/small"]
    contrast["standard_errors"] = abs(contrast["beta_deltaF_left_minus_right"]) / contrast["combined_population_SE"]
    out["numeric_checks"] = ["Original allocations and all attempted-draw denominators", "Linear mass agreement between row and population estimates", "DeltaF reconstructed from regional log masses", "Paired population SE reconstructed from four linear population means", "Student-t3 interval endpoints", "Importance ESS/CPU reconstructed from ESS and summed sampler CPU", "All plotted efficiency ratios reconstructed", "Seven frozen failures among 132 stratum comparisons"]
    return out


def plot_data(summary: dict) -> dict:
    return {
        "schema": "protected-guide-plot-data-v1", "regional_gate": "FAIL",
        "free_energy": [{"arm": name, "label": ARM_LABEL[name], **summary["free_energy_intervals"][name]} for name in ARMS],
        "importance_efficiency": [{"region": r, "label": LABEL[r], **summary["importance_efficiency"]["bank/protected"][r]} for r in (INSIDE, COMPLEMENT, NOENTRY)],
        "efficiency_uncertainty": "Descriptive realized importance ESS/CPU; no confidence interval estimated.",
        "scope": summary["scope"],
    }


def draw_figure(data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "protected-guide-20260924"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.2, 1], "wspace": .69})
    fig.subplots_adjust(left=.205, right=.965, top=.73, bottom=.27)
    fig.suptitle("Protected contact guide: mixed efficiency gains, unresolved regional weights", x=.5, y=.975, fontsize=16, fontweight="bold")
    fig.text(.5, .892, "REGIONAL CONVERGENCE GATE: FAIL", ha="center", fontsize=13, color="#a32932", fontweight="bold", bbox={"facecolor": "#fbecee", "edgecolor": "none", "pad": 9})
    fig.text(.5, .82, "10,485,760 attempts · 4 independent populations per arm · unchanged physical model", ha="center", color="#45515c")
    colors = ("#687785", "#187f86", "#578cab", "#995d9b")
    ys = list(range(4))[::-1]
    for row, y, color in zip(data["free_energy"], ys, colors):
        ax.errorbar(row["beta_F_native_minus_noentry"], y, xerr=row["halfwidth_95"], fmt="o", ms=8, color=color, capsize=5, lw=2)
    ax.set_yticks(ys, ["Bank 80\n4 × 1,048,576; λ/z = 128", "Protected 84\n4 × 1,048,576; λ/z = 128", "Protected 84, smaller N\n4 × 262,144; λ/z = 128", "Protected 84, more cloud points\n4 × 262,144; λ/z = 256"], fontsize=10)
    ax.set_ylim(-.6, 3.6)
    ax.set_xlim(-18.81, -18.645)
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.set_xlabel(r"$\Delta F_{\rm native-noentry}/k_BT$", labelpad=10)
    ax.set_title("A  Conditional free-energy difference", loc="left", fontsize=12, pad=15)
    ax.grid(axis="x", color="#e5e9ec", zorder=0)
    ax.text(.5, -.31, "Finite R4 + fixed scaffold only\n95% paired-population t₃ intervals", transform=ax.transAxes, ha="center", fontsize=10, color="#45515c")
    bx.axvline(1, ls="--", lw=1.2, color="#929ca4", zorder=0)
    for row, y, color in zip(data["importance_efficiency"], (2, 1, 0), ("#187f86", "#b96640", "#187f86")):
        value = row["right_over_left"]
        bx.barh(y, value, height=.43, color=color)
        bx.text(value + .035, y, f"{value:.3f}×", va="center", fontweight="bold", color=color)
    bx.set_yticks((2, 1, 0), ["Native\ninside R5", "Native\ncomplement", "Contact\nwithout entry"], fontsize=10)
    bx.set_ylim(-.6, 2.6)
    bx.set_xlim(0, 1.58)
    bx.set_xlabel("Protected / bank importance ESS per CPU", labelpad=10)
    bx.set_title("B  Allocation tradeoffs at matched N", loc="left", fontsize=12, pad=15)
    bx.text(.5, -.31, "Observed importance-sampling efficiency\nNot a trajectory mixing speedup", transform=bx.transAxes, ha="center", fontsize=10, color="#45515c")
    fig.text(.5, .043, "rᵈ = 1.5 Å; z = 0.035 Å⁻³. Seven stratum failures and one population-size ΔF failure remain. No assembly conclusion.", ha="center", fontsize=10, color="#45515c")
    fig.savefig(out / "protected-guide-results.png", dpi=180, facecolor="white", metadata={"Software": "Matplotlib"})
    fig.savefig(out / "protected-guide-results.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def markdown(summary: dict, out: Path) -> str:
    lines = ["# Protected-guide validation: completed results", "", "**The regional convergence gate remains FAIL.** The completed allocation improves importance-sampling efficiency inside R5 and for contact without entry, but reduces it for the native complement. It does not establish finite-system native assembly or instability.", "", f"![Completed regional comparison]({out / 'protected-guide-results.png'})", "", "## Allocation and scope", "", "All 10,485,760 attempts are retained, including hard-invalid zeros and rejected proposal branches. Each arm has four independent populations and two independent Poisson clouds per valid pose. The repaired rigid shape, fixed scaffold, finite R4 region, native classifier, physical measure, depletant radius 1.5 Å, activity 0.035 Å⁻³, and 50% uniform defensive probability are unchanged. Bank uses the old 80-component guide; the other arms use the frozen protected 84-component guide.", "", "| Arm | Draws per population | Auxiliary intensity λ/z | Sampler CPU seconds |", "|---|---:|---:|---:|"]
    for name in ARMS:
        arm = summary["arms"][name]
        lines.append(f"| {ARM_LABEL[name]} | {arm['allocation']['samples']:,} | {arm['allocation']['lambda_ratio']:g} | {arm['sampler_cpu_seconds']:,.3f} |")
    lines += ["", "CPU is summed sampler CPU, excluding this reporting step and downstream analysis. These are importance estimates, not a Markov-chain mixing benchmark.", "", "## Aggregate weights and precision", "", "ΔF = −log(Q_native/Q_contact-without-entry) within the same frozen R4 and scaffold. It is not a mobile-system association or crystal free energy. Linear population masses are averaged before taking logs. Intervals use the paired population delta method and Student t with three degrees of freedom; they are not rigorous guarantees of tail coverage.", "", "| Arm | ΔF / kBT | Population SE | Population 95% interval |", "|---|---:|---:|---:|"]
    for name in ARMS:
        row = summary["free_energy_intervals"][name]
        lines.append(f"| {ARM_LABEL[name]} | {row['beta_F_native_minus_noentry']:.6f} | {row['paired_population_log_ratio_SE']:.6f} | [{row['interval_95'][0]:.6f}, {row['interval_95'][1]:.6f}] |")
    contrast = summary["direct_free_energy_comparisons"]["protected/small"]
    lines += ["", "Aggregate decision-region quality, aggregate mass agreement, hard-only mass agreement, paired free-energy precision, native partition sums, and classifier/contact consistency pass. Standalone aggregate qualities also pass in the diagnostic small arm. These successes do not override the failed comparisons.", "", f"The protected–small ΔF difference is {contrast['beta_deltaF_left_minus_right']:+.9f} kBT with combined population SE {contrast['combined_population_SE']:.9f}: **{contrast['standard_errors']:.3f} SE**. It passes the 0.2 kBT tolerance but fails the frozen three-SE criterion.", "", "## All seven failed stratum comparisons", "", "Seven of 132 significant-stratum comparisons fail. Signs below are log Q_left − log Q_right. The full-precision entries and original flags are retained in summary.json; no criteria have been relaxed.", "", "| Left − right | Stratum | Region | Log-mass difference | Combined population SE | Difference / SE | Failed criterion |", "|---|---|---|---:|---:|---:|---|"]
    for row in summary["failed_strata"]:
        value, se = row["log_left_minus_right"], row["combined_population_log_delta_SE"]
        criterion = "0.2 absolute" if not row["absolute_passed"] else "3 SE"
        lines.append(f"| {row['left_arm']} − {row['right_arm']} | {row['family']} {row['bin']} | {LABEL[row['region']]} | {value:+.9f} | {se:.9f} | {abs(value)/se:.3f} | {criterion} |")
    lines += ["", "The previously weak no-entry orthants 30, 34, 50 and 58 pass the four comparisons. Orthants 42 and 62 are below the 1% parent-mass significance threshold in every arm and therefore were not tested by this gate; their absence from the failure list does not establish convergence.", "", "## Efficiency and remaining concentration", "", "| Region | Protected / bank importance ESS per CPU | Intensity 256 / 128 at smaller N |", "|---|---:|---:|"]
    for region in REGIONS:
        lines.append(f"| {LABEL[region]} | {summary['importance_efficiency']['bank/protected'][region]['right_over_left']:.6f} | {summary['importance_efficiency']['small/intensity256'][region]['right_over_left']:.6f} |")
    lines += ["", "These ratios describe realized importance weights. They have no estimated uncertainty bars and may change when rare tails are encountered; they do not establish faster independent contact sampling along a trajectory. A ratio below one is an allocation tradeoff, not an additional thermodynamic convergence criterion.", "", "The protected guide's native-complement orthant 55 carries 20.076% of that region's mass, with importance ESS 255.73 and a largest single contribution of 4.353% of the stratum. In the bank arm the corresponding values are 17.555%, ESS 955.98 and 1.090%. Ten cached tail rows (the five largest in each of protected r01 and r02) account for 12.997% of the protected stratum mass but 85.796% of its squared-weight sum. This distinction matters: second-moment concentration is much greater than mass concentration.", "", "No-entry orthant 63 remains poorly determined in the smaller controls. Its importance ESS is 220.42 for protected, 29.33 for small, and 35.73 for intensity256. The largest contributions are respectively 2.816%, 12.542% and 9.291% of the stratum. Two dominant intensity256 rows have cloud log-weight differences only 0.339 and 0.257, so their importance cannot be attributed to large disagreement between their two clouds alone.", "", "Doubling auxiliary intensity costs 1.676× as much CPU at matched N. The no-entry cloud fraction of observed variance falls from 30.22% to 14.00%, but importance ESS per CPU falls to 0.708×. More cloud intensity alone does not address the observed pose-coverage cost.", "", "A diagnostic using row-based instead of four-population SE would put the first five stratum differences below three row SE. That diagnoses uncertainty in an SE estimated from only four populations; it does not replace the prescribed population-based checks or turn their failures into passes.", "", "## Matching SMC evidence", "", "The independent review matches shape, scaffold, R4 domain, native definitions and measure. It compares linear SMC normalizer × terminal-region-indicator estimates, not endpoint fractions alone. Aggregate native and native-inside-R5 masses agree with the narrow SMC control, but the native complement remains unresolved:", "", "| Fresh arm | log Q_complement − matching narrow SMC | Combined population SE |", "|---|---:|---:|"]
    for name in ARMS:
        row = summary["matching_SMC"]["narrow"][name]["Qz"][COMPLEMENT]
        lines.append(f"| {ARM_LABEL[name]} | {row['log_new_minus_reference']:+.9f} | {row['combined_population_log_delta_SE']:.9f} |")
    lines += ["", "Each complement difference passes three combined SE but fails the original 0.2 absolute tolerance. Broad SMC also remains inconsistent in aggregate native and complement mass. All 40 observed matching hard-only Q0 comparisons pass. Both historical SMC controls have **zero no-entry particles among 8,192 terminal particles**; this is an unobserved contribution, not evidence of zero equilibrium mass or an independent confirmation of ΔF.", "", "## Consequence and next calculation", "", "Keep full-vessel and assembly-production gates closed. The immediate target is independent coverage of native-complement orthant 55 and no-entry orthant 63, plus reconciliation of the small-arm ΔF and matching SMC complement discrepancy. A separately frozen proposal/control that adds geometric support to those tails is more informative than merely increasing cloud intensity. Any new pilot or allocation requires a new protocol; this report launches none and recommends no post-hoc threshold changes.", "", "The existing deterministic unbound contribution bound applies only inside R4. Unmeasured full-vessel configurations and cooperative, mobile finite systems remain outside this calculation. The physical assembly question is **unresolved**.", "", "## Reproduction and provenance", "", "The report script checks pinned hashes for the completed authentication receipt, comparison, protocol, matching SMC review and bounded cached-noise diagnostic. It reuses the receipt's 682-file authentication and terminal-handle audit; it does not replay it. It reconstructs ΔF, paired population SE, interval endpoints and ESS/CPU ratios numerically before plotting. summary.json preserves all seven failures and the selected cached-tail metadata; plot-data.json is the exact plotted data; freeze.json binds the report source, inputs and outputs.", "", "```bash", "python tools/summarize_protected_guide_results.py --output /tmp/protected-guide-report-reproduction", "```", "", "Use the repository's Python environment with Matplotlib. The output must not exist, preventing overwrites of previous evidence. This reporting run performs zero physical draws, geometry calls, classifier calls, audit replays or fits, and changes no gates.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/protected-guide-completed-review-20260924/report")
    parser.add_argument("--doc-output", type=Path, help="Optional new documentation file; refuses existing paths")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    check(not out.exists(), f"Refusing existing output {out}")
    if args.doc_output:
        check(not args.doc_output.exists(), "Refusing existing documentation file")
    data, inputs = load_and_validate(root)
    summary = summarize(data)
    pdata = plot_data(summary)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "summary.json", summary)
    dump(out / "plot-data.json", pdata)
    draw_figure(pdata, out)
    report = markdown(summary, out)
    (out / "report.md").write_text(report)
    if args.doc_output:
        with args.doc_output.open("x") as handle:
            handle.write(report)
    shutil.copyfile(__file__, out / "summarize_protected_guide_results.py")
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    dump(out / "freeze.json", {"schema": "protected-guide-report-freeze-v1", "complete": True, "input_sha256": inputs, "source_sha256": sha(Path(__file__)), "files": files, "documentation": {"path": str(args.doc_output.resolve()), "sha256": sha(args.doc_output)} if args.doc_output else None, "numeric_checks": summary["numeric_checks"], "new_physical_draws": 0, "new_classifier_calls": 0, "new_geometry_calls": 0, "audits_replayed": 0, "fits_performed": 0, "gate_changes": 0})
    print(json.dumps({"output": str(out), "freeze_sha256": sha(out / "freeze.json"), "summary_sha256": sha(out / "summary.json"), "plot_data_sha256": sha(out / "plot-data.json"), "regional_gate": "FAIL"}, indent=2))


if __name__ == "__main__":
    main()
