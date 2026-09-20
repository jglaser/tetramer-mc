# Complete intermediate coverage with independent radial references

The original **2 ≤ q < 5** integral now has a disjoint reference allocation
covering its whole pose domain. The physical estimate is still **not
converged**: one new remainder draw supplies 88.85% of the combined estimate.
Hard accessible volume is much better resolved and agrees with earlier
calculations. The physical model remains rd = 1.5 Å, z = 0.035 Å⁻³, both
fixed AB neighbors, capture radius 18 Å, and normalized SO(3) Haar measure.
This is a conditional contact calculation, not an assembly free energy.

## Allocation and complete support

The frozen weighted chart is unchanged from the
[local controls](intermediate-local-reference.md). Its radius r is
dimensionless and distinct from the depletant radius. Three cloud-free
uniform-ball probes, 2,048 draws each, gave valid-pose fractions:

| Chart radius | Valid / draws | Valid fraction | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| 8 | 515 / 2,048 | 25.15% | 23.32–27.07% |
| 16 | 109 / 2,048 | 5.32% | 4.43–6.38% |
| 32 | 28 / 2,048 | 1.37% | 0.95–1.97% |

These probe rates determine useful fixed production budgets, not physical
weight variances. The successful probe artifact is
`runs/ab-intermediate-outer-geometry-v2-20260920`. An earlier constructor
call failed before any probe draws; its frozen, incomplete directory was
preserved. The corrected run used the declared independent seeds.

The source complete geometric chart and weighted chart have identical
anchors and angular scale. Their coordinates therefore obey the affine map

`u_weighted = L_weighted⁻¹(μ_cover − μ_weighted) + L_weighted⁻¹ L_cover u_cover`.

The original geometric cover has |u_cover| ≤ 10. The triangle inequality
and matrix Frobenius norm bound every corresponding weighted radius by
323.388218, including a numerical outward margin. This inherits the existing
complete q≤5 cover construction. It is an analytic enclosure evaluated in
FP64, not a formal interval-arithmetic certificate. Radius 32 alone is not
complete support.

The allocation was frozen before new physical draws in
`runs/ab-intermediate-outer-plan-20260920/protocol.json`, SHA-256
`472bac0a554fb887414171c22f51fe009363b71042940bd764927a840d093e93`.
It uses five disjoint pieces:

| Selected piece | Draw proposal | Populations × unconditional draws | Seed base |
| --- | --- | ---: | ---: |
| 0 ≤ r ≤ 4 | Existing uniform R4 reference | 4 × 16,384 | 99971010 |
| 4 < r ≤ 8 | Fresh uniform R8 ball | 4 × 32,768 | 100301010 |
| 8 < r ≤ 16 | Fresh uniform R16 ball | 4 × 131,072 | 100401010 |
| 16 < r ≤ 32 | Fresh uniform R32 ball | 4 × 524,288 | 100501010 |
| r > 32 | Fresh complete geometric q≤5 cover | 4 × 262,144 | 100601010 |

Population seeds increment by 1009. Every source retains all its invalid
and off-piece zeros in its own unconditional N. Each valid pose uses two
independent positive Poisson weights at λ/z=64. The selected outer half of
each new ball is accumulated directly from rows. The final remainder is
also a direct positive masked integral; it is never estimated by subtracting
two noisy totals. The earlier radius-one control is not added again.

The four new campaigns contain **3,801,088** unconditional draws. Including
the retained R4 reference, the partition uses 3,866,624 draws. The total
sampler cost is 3,759.75 CPU seconds, of which 954.18 seconds belong to the
previous R4 campaign. Production uses the unchanged archived latent kernel,
SHA `d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.

## Estimation and audit

For each piece Aⱼ, its original proposal gⱼ produces

\[
\widehat Q_j=N_j^{-1}\sum_i I_{A_j}(X_i)H(X_i)\overline W_i/g_j(X_i),
\qquad \widehat Q=\sum_j\widehat Q_j.
\]

For a uniform latent ball, 1/g is its latent volume times the exact pose
Jacobian. The complete remainder uses its own geometric chart's Jacobian;
the weighted chart determines only membership. No samples are pooled under
a common density. The analyzed physical domain includes every original-q,
capture and AB hard constraint.

New reference populations are independent conditional on the frozen
allocation. The earlier inner region and its estimator remain unchanged.
Even when the choice of outer partitions uses earlier information, their
conditional total expectation is the same fixed outside-R4 mass. Thus the
new outside sum has zero covariance with the retained inner estimator;
regional sampling variances can be added for the total under this conditional
unbiasedness argument. This does not require treating data-selected
individual regions as unconditionally independent. The archived analysis
uses this variance sum and also forms four population-group totals.

Within any one source, disjoint pieces share rows and have covariance.
Their combined half-ball variance is reconstructed from per-row totals;
quarter-bin variances are not added as if independent. An unobserved
supported piece is explicitly unresolved. No observed error estimate bounds
unseen physical weight.

All rows pass independent reconstruction of original q, pose Jacobian,
source-chart radius, capture mask, two-cloud numerator and full-N weights.
Shape, configuration, binary, region, source and raw-data hashes are checked.
The complete remainder also reconstructs the geometric cover exactly from
the original rigid-member geometry, and checks its weighted radii by both
the world-pose map and the affine latent map. Hard flags in this all-row
audit come from the archived kernel; selected extremes receive a separate
atomic audit below.

`tools/analyze_intermediate_partition.py` supports independent component
audits and a final disjoint sum. Its five tests cover unequal sample counts,
finite exact importance laws, unobserved pieces, constant limits, and coupled
chart enclosures. Six updated local-comparison tests cover exact boundaries,
shared-row covariance, Jacobians and masks. All 11 pass. The physical kernel
and its previously validated analytic sphere/union controls are unchanged.

## Physical result remains concentrated

| Piece | log Q | Row / population relative SE | Weight ESS | Largest draw |
| --- | ---: | ---: | ---: | ---: |
| r ≤ 4 | 4.733195 | 3.52% / 3.82% | 798.05 | 1.08% |
| 4 < r ≤ 8 | 7.836672 | 8.13% / 6.37% | 151.10 | 3.41% |
| 8 < r ≤ 16 | 11.787731 | 20.08% / 30.30% | 24.80 | 12.36% |
| 16 < r ≤ 32 | 14.034795 | 36.18% / 36.91% | 7.64 | 21.34% |
| r > 32 | 16.214232 | 99.99% / 99.99% | 1.00 | 99.99% |

The combined point estimate is **log Q = 16.332283**, with **88.93%** row
relative error and **86.15%** population-group error. The four group log
estimates are 17.608634, 14.304018, 14.722941 and 13.865754. A single remainder
draw supplies **88.85%** of the combined estimate; the observed paired-cloud
variance fraction is 26.84%. These numbers do not establish the true total.

The hard-region integral is **log Q₀ = −2.350480**, with **1.18%** row error
and **1.01%** population-group error. Previous geometric/mixture/full-cover
hard estimates were −2.352069, −2.385626 and −2.364224 with percent-level
errors. Geometry and measure normalization agree much better than the
depletion-weighted integral.

![Partition contributions and historical full-window estimates](../runs/ab-intermediate-partition-figure-qualified-20260920/intermediate-partition.png)

The figure transforms Q±one observed standard error to log coordinates.
Historical controls retain their original estimates and labels. The earlier
cover was used for training. None is retrospectively pooled or reweighted.
Large differences in point estimates are evidence of unresolved sampling,
not by themselves evidence of different physical targets or a model verdict.

## Where the new weight lies

`tools/diagnose_intermediate_partition_extremes.py` audits the eight largest
contributions from each of the five pieces. All **40** poses pass independent
atom-union hard checks. The smallest checked gap is 0.001321 Å. Original
cloud weights and proposal denominators are retained. Original-q bins
[2,2.25), [2.25,2.5), [2.5,3), [3,4), [4,5) reconstruct every piece's mass
with its full original N; these bins are descriptive after sampling.

The dominant remainder pose is population r00, draw 151795, at weighted
radius **39.245926** and **q = 3.277846**, with hard gaps **0.120054 and
0.115103 Å** to AB. It is not a q=2 or q=5 boundary event. Relative to the
old training maximum (q=3.267810), its center differs by **2.6917 Å** and
its proper orientation by **12.3924°**. Similar scalar registration scores
therefore do not identify the same pose neighborhood. These point differences
do not prove separate metastable basins or their integrated occupancies.

A further 256 independent clouds at each of the new R16, R32 and remainder
maxima give:

| Selected maximum | Fresh log(mean W) | Relative SE | Original two-cloud mean / fresh mean |
| --- | ---: | ---: | ---: |
| R16 outer half | 28.33588 | 3.91% | 0.843 |
| R32 outer half | 28.65507 | 4.53% | 0.624 |
| Complete remainder | 27.32489 | 4.54% | 1.058 |

The large pointwise weights reproduce under the frozen original kernel.
The remainder's original weight is only about 6% above its new conditional
mean. This rules out cloud noise as the main explanation of that outlier's
large contribution, but says nothing by itself about its neighborhood volume.
The arithmetic mean W is unbiased conditional on the fixed numerical
geometry; its logarithm is not. Original extreme selection is acknowledged.

## Artifacts and next calculation

The main results are in
`runs/ab-intermediate-partition-total-20260920/analysis.json`, with component
audits under `runs/ab-intermediate-partition-{r4,r8,r16,r32,remainder}-audit-20260920`.
Extreme-pose and cloud checks are in
`runs/ab-intermediate-partition-extremes-20260920` and
`runs/ab-intermediate-partition-peak-clouds-20260920`. The plan records all
campaign paths, budgets, seeds and hashes. `tools/prepare_intermediate_outer_campaign.py`
and the existing `tools/run_latent_region_campaign.py` reproduce the allocation
in fresh directories. Component analysis, for example:

```bash
python tools/analyze_intermediate_partition.py \
  --plan runs/ab-intermediate-outer-plan-20260920/protocol.json \
  --component remainder --out runs/fresh-remainder-audit
```

The next useful reference is a fixed neighborhood of the new dominant pose,
with broader proposal support for the additional observed contact poses and
the complete remainder retained. More clouds at the old peak, or a scalar-q
split alone, would leave these geometrically different neighborhoods unresolved.
The complete intermediate mass, q≥5 remainder, earlier SMC discrepancies,
trajectory mixing, and assembly thermodynamics remain open requirements.
