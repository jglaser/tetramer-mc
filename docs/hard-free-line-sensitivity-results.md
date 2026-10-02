# Completed guide-probability and cloud-intensity controls

The fixed sensitivity campaign is complete: **131,072 new attempted draws**,
eight full independent audits and one native-classification pass per valid pose.
All populations retain the repaired shape, original two-neighbor R4, depletant
radius 1.5 Å and activity 0.035 Å⁻³. Each control has four fresh 16,384-draw
populations, compared against the completed conditioned pilot without pooling.

**The sensitivity checks fail.** Aggregate physical weights and free-energy
differences agree, but a hard-volume check, weight-tail checks and 11 material
strata do not. These are unresolved sampling diagnostics, not evidence against
assembly. The full-vessel preparation remains unlaunched.

| Arm | β(F_native − F_competing) | Population 95% half-width |
|---|---:|---:|
| Reference: α=.5, λ/z=128 | −18.6950 | 0.1237 |
| α=.2, λ/z=128 | −18.7776 | 0.1024 |
| α=.5, λ/z=64 | −18.7993 | 0.0631 |

All eight between-control regional **Qz** comparisons pass the predeclared
absolute 0.2 and three combined linear-SE checks. Both direct free-energy
comparisons also pass. These are conditional fixed-scaffold integrals, not
finite-system association or assembly free energies.
Native entry uses the complete frozen contact classifier, not a mobile-cluster
registry or assembly observable.

The smaller cloud intensity fails the old-R5 native **Q0** comparison: its log
mass is 0.10061 higher, or 3.364 combined linear population SE. Q0 contains no
Poisson weight, so this discrepancy cannot be a direct consequence of cloud
noise at fixed poses. Finite pose-sampling error or underestimated uncertainty
remains a candidate explanation; the frozen failure is retained. At λ/z=64,
the largest individual contributions to old-R5 and remaining-native Qz are
3.157% and 2.682%, exceeding the 2% threshold despite the narrow ΔF interval.

The material-stratum failures are:

| Control | Region | Failed bins |
|---|---|---|
| α=.2 | Native total | orthant 55 |
| α=.2 | Competing | orthants 22, 38, 39 |
| α=.2 | Remaining native | orthant 55 |
| λ/z=64 | Native total | radial 1 (2≤ρ<3) |
| λ/z=64 | Remaining native | radial 1 (2≤ρ<3) |
| λ/z=64 | Competing | orthants 22, 26, 39 |
| λ/z=64 | Old-R5 native | orthant 51 |

Every original stratum is retained in the
[complete report](../results/hard-free-line-sensitivity-review-20261002/report.md)
and its JSON. The failures have different significance: some exceed only the
absolute criterion, while others fail the linear-SE criterion as well. Competing
orthants 22 and 39 also failed in the earlier proposal pilot. Their recurrence
directs attention to local weight concentration; it does not by itself prove a
missing physical basin.

![Completed sensitivity controls](../results/hard-free-line-sensitivity-review-20261002/sensitivity.png)

## Efficiency and the two-cloud diagnostic

Reducing α from .5 to .2 improves observed importance ESS per sampler CPU by only
1.03–1.10× across the four decision regions. Reducing λ/z to 64 improves competing
contact efficiency by 1.38×, but lowers native-total efficiency to 0.71×, old-R5
native efficiency to 0.65× and remaining-native efficiency to 0.78×. Neither
result warrants replacing the primary α=.5, λ/z=128 setting. These are integration
weight diagnostics, not an assembly mixing benchmark.

The saved two-cloud decomposition estimates the cloud contribution to observed
native-total row variance at 37.2% for the reference, 36.9% for α=.2 and 65.8%
for λ/z=64. The corresponding old-R5 values are 31.6%, 31.7% and 68.4%.
This is consistent with the smaller intensity giving noisier native weights.
Other failed strata retain substantial residual pose variance, so increasing
cloud intensity alone would not resolve all observed instabilities. These are
finite-sample decompositions, not exact causal fractions; negative residuals
would be retained. The supplemental calculation only reuses saved aggregate
statistics at
`results/hard-free-line-sensitivity-noise-20261002/analysis.json`.

The native-remainder mass shifts by only +0.0611 and −0.00928 in log units under
these two controls. Neither explains the historical broad-SMC estimate of only
0.216× the pilot's remainder. This strengthens the case for investigating SMC
coverage and its bridge, but does not independently demonstrate that a modified
SMC run resolves the discrepancy.

The separately frozen [larger-population stage](hard-free-line-population-size-results.md)
has completed. Aggregate masses agree, while material-stratum failures remain.
Its populations are not pooled with these controls and none of these failures
is waived. A fresh independent SMC bridge control is the next measurement.

## Provenance

Campaign:
`/vast/xvg/tetramer-mc-runs/hard-free-line-sensitivity-20261001`.
Protocol SHA-256:
`100fdb3b68b2a1bfdf9de6fdce61a7759894d0818f7affa6a9b9e7d0ba977cc6`.
Physical classification summary:
`641eac49e9cc8a8e87fa0b9375514104af058f20d94de769f0d1182fdd882406`.
Sensitivity comparison:
`7445d6e8dddadba426a6024ac3967d1603c9801bfe5eeccd3728326720b1fef8`.
The plot/report binds the terminal controller and completed summaries; no poses,
clouds or classifiers were replayed.
