# Independent checks around the guided shoulder peaks

The [first local shoulder reference](inner-shoulder-local-reference.md)
calibrates the three observations that dominate the old direct estimate.
Most guided inner-shoulder weight remains outside that checked region.
This calculation measures two additional finite neighborhoods, centered on
the largest recorded importance contribution from each guided confirmation.
Selecting those observations identifies where to check; it does not establish
their neighborhoods' integrated weight.

The physical target remains one mobile rigid tetramer with **both AB neighbors
fixed**, depletant radius **1.5 Å**, activity **0.035 Å⁻³**, capture radius
**18 Å**, and the original 2 Å maximum-member-displacement / 15° proper-rotation
metric. Every local region is intersected with **strict 1<q<1.1**, capture,
and full AB hard validity. Orientation measure is normalized proper SO(3)
Haar measure, so Q has units Å³.

## Fixed references

The centers are exact stored observations; their poses, original proposal
densities, physical cloud weights, seeds and source hashes are preserved.

| Center source | Population / seed / draw | Original q |
| --- | --- | ---: |
| Mixture confirmation | r03 / 99604037 / 30939 | 1.0382368971 |
| Geometry confirmation | r01 / 99612019 / 33280 | 1.0232210223 |

At each center we construct the same geometric chart used in the first
reference. For zero-centered member positions with second moment M, set
A=tr(M)I−M. Relative translation and a left-Cayley vector define

\[
\rho^2=|\delta t|^2+4c^T A_{\rm anchor}c,\qquad
J=\frac{1}{8\pi^2\sqrt{\det A}(1+|c|^2)^2}.
\]

The chart ball lies inside the corresponding rigid-member RMS ball; it is
not a complete RMS cover. No likelihood, physical weights, widths or
covariance are fitted. A uniform six-ball draw contributes
`V6(R) J I_inner I_capture I_hard Wbar`, retaining every rejected draw as
zero. `Wbar` is the linear mean of two independent positive Poisson factors
with intensity/activity ratio 64; both fixed neighbors enter the union
geometry. The reviewed physical executable is unchanged.

| Center | Radius | Populations × draws | Seed schedule |
| --- | ---: | ---: | --- |
| Mixture | 0.25 Å | 16 × 16,384 | 110001010+1009i |
| Mixture | 0.5 Å | 16 × 16,384 | 110101010+1009i |
| Geometry | 0.25 Å | 16 × 16,384 | 111001010+1009i |
| Geometry | 0.5 Å | 16 × 16,384 | 111101010+1009i |

The allocation totals **1,048,576 new unconditional draws**, eight workers
per campaign and 32 total. Budgets, streams, boundaries and combination
rules were frozen before physical outcomes. Separate 256-pose geometric
probes found 19, seven, 39 and four valid poses, respectively. Coordinate,
member-RMS and Haar-Jacobian checks passed; these probes did not change the
production allocation.

## A disjoint union, with the remainder retained

The direct and mixture centers are 0.6970 Å apart in actual member RMS;
the geometry center is 1.6206 Å from direct and 1.5465 Å from mixture.
Membership nevertheless uses each exact frozen chart, not a distance
between centers. Whole radius-0.5 balls cannot simply be added.

The prespecified priority assigns a pose to the first ball containing it:

1. Direct: `rho_direct <= 0.5`.
2. Mixture: `rho_mixture <= 0.5` and `rho_direct > 0.5`.
3. Geometry: `rho_geometry <= 0.5` and both earlier radii exceed 0.5.
4. Remainder: all three radii exceed 0.5.

The direct contribution uses its previously frozen independent [0,0.25]
and (0.25,0.5] shell estimates. Each new contribution uses the same
independent-shell allocation around its own center, **after excluding
earlier balls**. The three assigned means and independent variances can
then be added. This produces a finite-union estimate; overlapping whole
balls and samples from different proposal laws are never pooled.

The historical direct and guided streams are also classified by these
same masks. They retain their original proposal densities and full
unconditional counts: 1,048,576 for direct and 262,144 for each guide.
Guided observations with q≥1.1 remain zeros for this comparison. Same-row
covariance is required when reconstructing totals or comparing overlapping
masks. Historical observations contributed to center selection, so their
contrasts with the new references are calibration diagnostics rather than
nominal independent significance tests.

The historical outside-union estimate stays separate. Combining it with
the fresh finite-region estimates would not supply a new independent
whole-shoulder reference. A small observed error on this union cannot bound
unseen weight elsewhere, prove trajectory mixing, or establish assembly.

## Reproducible preparation

`tools/prepare_shoulder_guide_peak_references.py` verifies original selected
rows, source audits, physical definitions and geometry before writing the
frozen preparation and four exact launch commands to
`runs/ab-shoulder-guide-peak-reference-preparation-20260921`.
The protocol SHA-256 is
`64f5a355b11d81f49015f5379f3fc4332ca8649e6974d18b8c93e28e9f7ba11b`;
the commands SHA-256 is
`7fd91ebb04d0e1881ba35a4e53bf2348174aafbe11ee63ac297cb53dd6b57339`.

Production outputs use separate center/radius directories under
`runs/ab-shoulder-guide-peak-reference-20260921`. Process identities,
commands and exit statuses are in
`runs/ab-shoulder-guide-peak-reference-launch-20260921`. All source and
executed dependency archives are retained independently of later tool edits.

## Completed calibration

All 64 new physical populations and four original density/Jacobian auditors
finished successfully. The three-chart analysis classifies **1,048,576 new
rows plus 1,572,864 historical rows**, with original proposal densities and
denominators intact. Fifty-six extreme poses pass separate atomic checks
against both neighbors. All executing sources match their frozen archives.
Fifty tests cover selection, geometry, strict boundaries, full-denominator
moments, overlap covariance, independent sums and failed-auditor handling.

| Fresh whole-ball reference | log Q | Row / population RSE | Weight ESS | Largest row | CPU s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixture, radius 0.25 | 21.492252 | 2.62% / 1.68% | 1451.2 | 1.11% | 745.1 |
| Mixture, radius 0.5 | 22.882839 | 6.84% / 5.84% | 213.4 | 2.37% | 268.7 |
| Geometry, radius 0.25 | 22.971191 | 1.41% / 1.76% | 4928.9 | 0.39% | 1445.0 |
| Geometry, radius 0.5 | 23.838476 | 7.21% / 7.52% | 192.3 | 3.44% | 260.0 |

The larger proposals independently reproduce their respective radius-0.25
integrals: log Q=21.57103 with 17.52% row error for mixture, and 23.08650
with 12.77% error for geometry. Their differences from the dedicated smaller
references are 0.43 and 0.85 combined observed standard errors. These
comparisons use identical regions and independent streams, although finite
agreement still cannot exclude unseen high-weight poses.

The paired-cloud diagnostic attributes 25–44% of the observed whole-ball
row variance to conditional Poisson noise. Weight ESS measures the
concentration of importance contributions; it is not trajectory mixing ESS.
The four new campaigns cost 2718.8 sampler CPU seconds.

### Selected extremes versus integrated neighborhoods

| Radius-0.25 region | Historical mixture log Q (row RSE) | Historical geometry log Q (row RSE) | Fresh geometric log Q (row RSE) |
| --- | ---: | ---: | ---: |
| Mixture peak | 22.71151 (76.03%) | 21.84628 (51.83%) | 21.49225 (2.62%) |
| Geometry peak | 22.77203 (12.39%) | 23.96358 (91.86%) | 22.97119 (1.41%) |

Each guide's own selected maximum exaggerates the corresponding small
region's weight in its historical sample: factors **3.38** and **2.70**
relative to the new references. The original direct inner cover recorded
no hits in either small ball, and only two and one hits in their respective
radius-0.5 balls. It therefore overrepresented its own selected neighborhood
while missing important weight at these other locations. The original rows
and their uncertainties remain in the comparison; data-dependent selection
precludes treating these ratios as independent significance tests.

The mixture guide's estimate of the geometry-centered small ball is about
18% lower than the fresh reference, despite its smaller observed error than
the other historical controls. This discrepancy remains visible; agreement
of a larger union does not establish precision of every part.

### Removing overlaps and adding independent pieces

Excluding the direct ball removes **20.05% ± 2.30 percentage points** of
the observed whole radius-0.5 mixture-ball weight. This uncertainty includes
the same-row covariance. The removal is concentrated in its outer shell;
only 0.82% of the dedicated small-ball weight is excluded. No geometry-ball
samples overlap either earlier ball. Its center's member-RMS distances
also exceed the sums of the relevant RMS radii, consistent with separation.
The analysis still uses explicit membership masks.

| Assigned region, independent shell sum | log Q | Row / population RSE |
| --- | ---: | ---: |
| Direct, first | 23.238833 | 5.36% / 4.82% |
| Mixture, excluding direct | 22.633189 | 5.64% / 4.60% |
| Geometry, excluding both | 23.785766 | 4.23% / 5.43% |
| **Union of the three** | **24.424744** | **2.92% / 3.31%** |

The union uses 1,572,864 independent source draws across six proposal laws:
the two previously completed direct references and four new references.
Each law retains its own denominator. The union is a sum of means and
independent variances, not a pooled row mean. The assigned two-shell sum
shares its outer-shell samples with the corresponding whole radius-0.5
control, so these two estimators are not independent of one another.

| Historical source | Same union log Q (row RSE) | Outside-union log Q (row RSE) | Outside fraction of its inner estimate |
| --- | ---: | ---: | ---: |
| Direct inner cover | 26.06339 (77.50%) | 22.60058 (66.49%) | 3.04% |
| Mixture guide | 24.36147 (15.61%) | 23.33294 (8.58%) | 26.34% |
| Geometry guide | 24.61746 (48.55%) | 23.79311 (22.97%) | 30.48% |

Both guided union estimates agree in scale with the independently measured
finite union. Their old data assign about 70–74% of inner-shoulder weight
to that union. The remaining 26–30% is substantial, and the two guided
outside estimates differ by a factor 1.58. Their largest outside rows carry
4.65% and 14.60% of those estimates, with importance ESS 135.6 and 18.9.
The geometry guide's outside variance is especially sensitive to Poisson
noise: the paired-cloud fraction is 62.75%.

![Finite neighborhood comparisons and the separate remainder](../runs/ab-shoulder-guide-peak-reference-figure-20260921/shoulder-guide-peak-references.png)

This controls the three selected neighborhoods without substituting a new
whole-shoulder normalizer into the [AB ledger](ab-regional-weight-status.md).
A fresh proposal with complete shoulder support should retain coverage of
the remainder while concentrating useful effort around these calibrated
contacts. Native-tail and far-contact uncertainties also remain; neither
the finite union nor its importance ESS establishes assembly or an MCMC
speedup.

The completed analysis and executed dependencies are in
`runs/ab-shoulder-guide-peak-reference-assessment-20260921/analysis.json`,
SHA-256 `100f67ef74289bee1c3f0a38651c2b1c211e83afb9e50d0c9c516521fd44e3ef`.
The figure and its source data are in
`runs/ab-shoulder-guide-peak-reference-figure-20260921`.
