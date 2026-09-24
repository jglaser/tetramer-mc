# Protected-guide validation: completed results

**The regional convergence gate remains FAIL.** The completed allocation improves importance-sampling efficiency inside R5 and for contact without entry, but reduces it for the native complement. It does not establish finite-system native assembly or instability.

![Completed regional comparison](/home/xvg/tetramer-mc/runs/protected-guide-completed-review-20260924/report/protected-guide-results.png)

## Allocation and scope

All 10,485,760 attempts are retained, including hard-invalid zeros and rejected proposal branches. Each arm has four independent populations and two independent Poisson clouds per valid pose. The repaired rigid shape, fixed scaffold, finite R4 region, native classifier, physical measure, depletant radius 1.5 Å, activity 0.035 Å⁻³, and 50% uniform defensive probability are unchanged. Bank uses the old 80-component guide; the other arms use the frozen protected 84-component guide.

| Arm | Draws per population | Auxiliary intensity λ/z | Sampler CPU seconds |
|---|---:|---:|---:|
| Bank 80 | 1,048,576 | 128 | 79,865.961 |
| Protected 84 | 1,048,576 | 128 | 80,142.310 |
| Protected 84, smaller N | 262,144 | 128 | 20,143.669 |
| Protected 84, intensity 256 | 262,144 | 256 | 33,752.779 |

CPU is summed sampler CPU, excluding this reporting step and downstream analysis. These are importance estimates, not a Markov-chain mixing benchmark.

## Aggregate weights and precision

ΔF = −log(Q_native/Q_contact-without-entry) within the same frozen R4 and scaffold. It is not a mobile-system association or crystal free energy. Linear population masses are averaged before taking logs. Intervals use the paired population delta method and Student t with three degrees of freedom; they are not rigorous guarantees of tail coverage.

| Arm | ΔF / kBT | Population SE | Population 95% interval |
|---|---:|---:|---:|
| Bank 80 | -18.726368 | 0.009627 | [-18.757004, -18.695732] |
| Protected 84 | -18.745986 | 0.012145 | [-18.784638, -18.707334] |
| Protected 84, smaller N | -18.707017 | 0.003348 | [-18.717672, -18.696363] |
| Protected 84, intensity 256 | -18.713895 | 0.016431 | [-18.766186, -18.661604] |

Aggregate decision-region quality, aggregate mass agreement, hard-only mass agreement, paired free-energy precision, native partition sums, and classifier/contact consistency pass. Standalone aggregate qualities also pass in the diagnostic small arm. These successes do not override the failed comparisons.

The protected–small ΔF difference is -0.038968981 kBT with combined population SE 0.012598360: **3.093 SE**. It passes the 0.2 kBT tolerance but fails the frozen three-SE criterion.

## All seven failed stratum comparisons

Seven of 132 significant-stratum comparisons fail. Signs below are log Q_left − log Q_right. The full-precision entries and original flags are retained in summary.json; no criteria have been relaxed.

| Left − right | Stratum | Region | Log-mass difference | Combined population SE | Difference / SE | Failed criterion |
|---|---|---|---:|---:|---:|---|
| bank − protected | radial 0 | Contact without entry | +0.016768006 | 0.004404810 | 3.807 | 3 SE |
| bank − protected | radial 2 | Native complement | -0.058946853 | 0.019535861 | 3.017 | 3 SE |
| bank − protected | orthant 55 | All native | -0.132535130 | 0.024085460 | 5.503 | 3 SE |
| bank − protected | orthant 23 | Native inside R5 | +0.015508011 | 0.004447245 | 3.487 | 3 SE |
| bank − protected | orthant 55 | Native complement | -0.189025366 | 0.044849265 | 4.215 | 3 SE |
| protected − intensity256 | orthant 63 | Contact without entry | -0.361845182 | 0.192814982 | 1.877 | 0.2 absolute |
| small − intensity256 | orthant 63 | Contact without entry | -0.253780804 | 0.214676780 | 1.182 | 0.2 absolute |

The previously weak no-entry orthants 30, 34, 50 and 58 pass the four comparisons. Orthants 42 and 62 are below the 1% parent-mass significance threshold in every arm and therefore were not tested by this gate; their absence from the failure list does not establish convergence.

## Efficiency and remaining concentration

| Region | Protected / bank importance ESS per CPU | Intensity 256 / 128 at smaller N |
|---|---:|---:|
| All native | 0.665785 | 0.695174 |
| Native inside R5 | 1.260557 | 0.713719 |
| Native complement | 0.578293 | 0.687103 |
| Contact without entry | 1.141477 | 0.708326 |

These ratios describe realized importance weights. They have no estimated uncertainty bars and may change when rare tails are encountered; they do not establish faster independent contact sampling along a trajectory. A ratio below one is an allocation tradeoff, not an additional thermodynamic convergence criterion.

The protected guide's native-complement orthant 55 carries 20.076% of that region's mass, with importance ESS 255.73 and a largest single contribution of 4.353% of the stratum. In the bank arm the corresponding values are 17.555%, ESS 955.98 and 1.090%. Ten cached tail rows (the five largest in each of protected r01 and r02) account for 12.997% of the protected stratum mass but 85.796% of its squared-weight sum. This distinction matters: second-moment concentration is much greater than mass concentration.

No-entry orthant 63 remains poorly determined in the smaller controls. Its importance ESS is 220.42 for protected, 29.33 for small, and 35.73 for intensity256. The largest contributions are respectively 2.816%, 12.542% and 9.291% of the stratum. Two dominant intensity256 rows have cloud log-weight differences only 0.339 and 0.257, so their importance cannot be attributed to large disagreement between their two clouds alone.

Doubling auxiliary intensity costs 1.676× as much CPU at matched N. The no-entry cloud fraction of observed variance falls from 30.22% to 14.00%, but importance ESS per CPU falls to 0.708×. More cloud intensity alone does not address the observed pose-coverage cost.

A diagnostic using row-based instead of four-population SE would put the first five stratum differences below three row SE. That diagnoses uncertainty in an SE estimated from only four populations; it does not replace the prescribed population-based checks or turn their failures into passes.

## Matching SMC evidence

The independent review matches shape, scaffold, R4 domain, native definitions and measure. It compares linear SMC normalizer × terminal-region-indicator estimates, not endpoint fractions alone. Aggregate native and native-inside-R5 masses agree with the narrow SMC control, but the native complement remains unresolved:

| Fresh arm | log Q_complement − matching narrow SMC | Combined population SE |
|---|---:|---:|
| Bank 80 | +0.225623844 | 0.159402456 |
| Protected 84 | +0.280456934 | 0.160236687 |
| Protected 84, smaller N | +0.247987656 | 0.159872743 |
| Protected 84, intensity 256 | +0.252769356 | 0.159998304 |

Each complement difference passes three combined SE but fails the original 0.2 absolute tolerance. Broad SMC also remains inconsistent in aggregate native and complement mass. All 40 observed matching hard-only Q0 comparisons pass. Both historical SMC controls have **zero no-entry particles among 8,192 terminal particles**; this is an unobserved contribution, not evidence of zero equilibrium mass or an independent confirmation of ΔF.

## Consequence and next calculation

Keep full-vessel and assembly-production gates closed. The immediate target is independent coverage of native-complement orthant 55 and no-entry orthant 63, plus reconciliation of the small-arm ΔF and matching SMC complement discrepancy. A separately frozen proposal/control that adds geometric support to those tails is more informative than merely increasing cloud intensity. Any new pilot or allocation requires a new protocol; this report launches none and recommends no post-hoc threshold changes.

The existing deterministic unbound contribution bound applies only inside R4. Unmeasured full-vessel configurations and cooperative, mobile finite systems remain outside this calculation. The physical assembly question is **unresolved**.

## Reproduction and provenance

The report script checks pinned hashes for the completed authentication receipt, comparison, protocol, matching SMC review and bounded cached-noise diagnostic. It reuses the receipt's 682-file authentication and terminal-handle audit; it does not replay it. It reconstructs ΔF, paired population SE, interval endpoints and ESS/CPU ratios numerically before plotting. summary.json preserves all seven failures and the selected cached-tail metadata; plot-data.json is the exact plotted data; freeze.json binds the report source, inputs and outputs.

```bash
python tools/summarize_protected_guide_results.py --output /tmp/protected-guide-report-reproduction
```

Use the repository's Python environment with Matplotlib. The output must not exist, preventing overwrites of previous evidence. This reporting run performs zero physical draws, geometry calls, classifier calls, audit replays or fits, and changes no gates.
