# Review before the larger-population comparison

The completed 4 × 16,384 pilot supports a fresh population-size comparison of the **same frozen proposal**, before using the new angular conditioner on proteins. Its four aggregate region comparisons pass; the conditioned arm passes the aggregate RSE, ESS and largest-contribution checks. Eight material strata still fail the separate 0.2 kBT agreement limit, although all pass three combined standard errors. Full-vessel and assembly gates remain closed.

## Matching SMC evidence

Ratios below are SMC linear population means divided by the conditioned IID estimate. The standardized difference uses **combined linear-mass standard errors**, not a log-space z-score. Native and total are numerically indistinguishable at the displayed precision.

| SMC control | Region | SMC / IID | Linear difference / SE | Outcome |
|---|---|---:|---:|---|
| unrestricted_broad | total / native | 0.623 | -3.48 | fails both limits |
| unrestricted_broad | old-R5 native intersection | 0.985 | -0.13 | passes |
| unrestricted_broad | remaining-R4 native | 0.216 | -5.34 | fails both limits |
| unrestricted_narrow | total / native | 0.888 | -1.90 | passes |
| unrestricted_narrow | old-R5 native intersection | 0.989 | -0.54 | passes |
| unrestricted_narrow | remaining-R4 native | 0.773 | -1.82 | fails 0.2 only |
| excluded_narrow | contact without native entry | 1.001 | +0.03 | passes |
| excluded_large | contact without native entry | 0.886 | -1.47 | passes |
| excluded_broad | contact without native entry | 1.585 | +1.98 | fails 0.2 only |

All observed hard-only `Q0` comparisons pass for all five controls, against both fresh arms. The full JSON receipt retains every available region/measure/control comparison for both arms. Unrestricted SMC has no observed terminal nonnative-contact contribution; this is missing coverage, not zero mass. No control observes the unbound region, but the independent deterministic bound inside R4 is `log Q_unbound <= -9.807904`, far below the measured contact masses. This bound says nothing about the full vessel.

The old-R5 native intersection agrees across methods. The remaining native region contributes about **47.0%** of the current native mass and accounts for the broad-SMC shortfall. Narrow SMC is closer but still misses the predeclared absolute tolerance there. The excluded broad control has excess competing-contact mass relative to the narrow/large controls and fresh estimates, with substantial population uncertainty. These observations favor incomplete exploration of high-weight subregions as the remaining problem. They do not identify which method has fully covered all modes.

## Why increase population size now?

Keeping the guide and executable fixed isolates population-size sensitivity. The current baseline has a 3.57% largest remaining-native draw and ESS160, while conditioning reduces these to 0.97% and ESS642. The remaining cross-proposal stratum differences are statistically compatible but too large under the independent absolute criterion. Four times as many fresh draws per population is a direct test of whether these contributions stabilize or expose heavier tails. Finite-variance SE would shrink roughly twofold; that is an expectation to test, not a convergence guarantee.

Five failing material bins concern nonnative contacts (radial bin2 and orthants22,30,39,63); one concerns the old-R5 native intersection (orthant55), and two concern the remaining native region (orthants19,51). In particular orthant39 still has nearly unit combined log-delta SE, so the planned increase may remain insufficient. Preserve all original bins, all attempts/zeros, all independent populations and both clouds.

The proposed continuation is four **fresh** populations of 65,536 per arm, using the exact archived `752c…` normalizer, the unchanged beta=0/xyz-beta=1 guide pair, alpha=0.5, lambda/activity=128, two independent clouds per valid pose, and the original repaired shape, radius 1.5 Å, activity 0.035 Å⁻³, region and classifier definitions. Freeze seeds, inputs and job limits before execution; compare stages separately and retain the same convergence gates. Do not pool old and new populations to manufacture precision or substitute the exploratory retained-index weights for the primary full-mixture estimates.

This is a regional integration check, not an assembly experiment. Failure would leave the physical question unresolved. Angular-guide toy validation can proceed independently; its protein test would introduce another proposal variable before this existing sensitivity question is answered.

## Sources and scope

- Matching bridge: `/vast/xvg/tetramer-mc-runs/hard-free-line-smc-bridge-20261001/analysis.json`.
- Completed regional pilot: `/vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001/comparison/analysis.json`.
- Source hashes and all matching comparison records: `results/hard-free-line-population-size-review-20261001/review.json`.

This review read completed summaries only. It performed no new geometry, classification, sampling, or historical trajectory replay.
