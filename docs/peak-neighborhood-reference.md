# A geometric neighborhood around the new intermediate contact

The dominant draw in the complete intermediate remainder has a reproducible
pointwise depletion weight. A new reference measures its **neighborhood
volume and physical weight**, using a chart whose widths follow from rigid
tetramer geometry. Its center is selected from earlier data; fresh independent
draws integrate the now-fixed region. This is not an independently discovered
basin, and the finite neighborhood is not the complete intermediate region.

The physical target remains depletant radius 1.5 Å, activity 0.035 Å⁻³,
both fixed AB neighbors, capture radius 18 Å, original 2 ≤ q < 5, and
normalized SO(3) Haar measure. In particular, recentering the proposal does
not change the native reference used to calculate q. These are conditional
contact weights; they exclude the cost of assembling the fixed neighbors.

## Geometry and exact measure

Let rᵢ be the centered tetramer member centers, M = mean(rᵢrᵢᵀ), and
A = tr(M)I − M. The eigenvalues of A are approximately 161.383, 399.866,
and 561.249 Å². Rotate A into neighbor A's frame at the selected contact
pose to obtain Aₐ. The pose is represented by translation δt and a left
Cayley rotation c relative to this selected pose. Define

\[
\rho^2=|\delta t|^2+4c^\mathsf{T}A_a c.
\]

The single Gaussian file describes this coordinate system with zero mean
and covariance diag(I, ℓ²Aₐ⁻¹/4). **Production draws are uniform in the
latent six-dimensional ball**, not Gaussian. The angular length ℓ is a
coordinate convention; it cancels from the actual proposal when the
covariance changes consistently.

The actual member-center RMS displacement from the selected pose obeys

\[
\mathrm{RMS}^2=|\delta t|^2+
\frac{4c^\mathsf{T}A_a c}{1+|c|^2}\le\rho^2.
\]

Consequently, a chart ball lies inside the corresponding member-RMS ball;
it is **not a complete cover** of that RMS ball. The exact physical Jacobian
from unit latent coordinates is

\[
J=\frac{1}{8\pi^2\sqrt{\det A}(1+|c|^2)^2}.
\]

For a uniform ball of radius R, V₆(R) = π³R⁶/6. Each unconditional draw
contributes V₆(R) J H Iq Ic W̄, with the original AB hard, q, and capture
indicators. W̄ averages two independent positive Poisson estimators at
λ/z = 64. Invalid draws remain zeros in the original fixed N. Means of
these weights are unbiased conditional on the frozen region and numerical
geometry; their logarithms are not.

The selected point is historical remainder population r00, draw 151795,
seed 100601010, at original q = 3.277845574. For R = 1, geometry bounds
the original q between 2.27785 and 4.27785; the angular contribution is
smaller. R = 2 can leave the original interval. Every campaign retains
the original masks, including where this particular bound makes them
redundant. The maximum relative rotation angles at R = 0.5, 1, and 2
are approximately 2.255°, 4.508°, and 9.002°.

## Frozen independent calculation

`tools/prepare_peak_neighborhood.py` freezes the selected center, geometric
chart, original physical constraints, three radii, fixed budgets and seeds
before geometry probes or physical draws. It performs no fit to the selected
weight. Its preparation is
`runs/ab-intermediate-new-peak-geometry-20260920`.

| Radius R (Å-equivalent) | Cloud-free valid / 512 | Production populations × draws | Seed base |
| --- | ---: | ---: | ---: |
| 0.5 | 123 | 4 × 16,384 | 100901010 |
| 1 | 37 | 4 × 16,384 | 101001010 |
| 2 | 21 | 4 × 16,384 | 101101010 |

The source selection and covariance geometry are the same for all three
campaigns. Population seeds increment by 1009. The 1,536 geometry probes
check coordinate inversion, world/body density agreement, the exact member
RMS identity and Jacobian to below 1.5 × 10⁻¹⁴ in their respective numerical
units. Probe acceptance measures hard geometry, not physical weight variance.

All physical draws use the unchanged archived Rust executable with SHA-256
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.
Later fail-fast provenance checks in the reusable preparer additionally
check the original guide protocol, angular-length source hash, and capture
domain. Those inputs also pass independent checks for this frozen
preparation; its archived preparer and all physical inputs remain unchanged.

## Comparison and coverage

The three balls are nested. Their estimates must not be added. Direct
poststratification with each source's full N estimates matched smaller
balls, radial shells and intersections with the previous weighted-chart
r ≤ 32 or r > 32 pieces. Here ρ is the new geometry-scaled radius and r
is the old dimensionless weighted-chart radius; they are different maps.

The radius-one and radius-two neighborhoods cross the old partition boundary. Adding their whole
weight to the previous complete-partition estimate would double count.
Replacing a piece requires the same new mask and a separately sampled
complement. The complete reference remains available, and the earlier
sample used to select this center is a historical diagnostic rather than
held-out validation. Same-source disjoint regions have covariance; their
combined variance must be computed from per-row totals, not from an
independence assumption. Observed error bars do not bound unseen weight.

## Results

All 196,608 physical draws completed. Independent reconstruction checks every
row's original q, source density, Jacobian, hard/capture masks, Poisson
numerator and unconditional denominator. A second geometric reconstruction
checks the actual transformed member positions, chart radius and Jacobian.
The hard flags in this audit come from the original Rust kernel. Separate
atom-union checks pass at all 24 selected highest-weight poses, with the
smallest gap 0.001964 Å.

| Source ball R | log Q | Row / population relative SE | Largest draw | log hard volume | Sampler CPU seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 12.054798 | 4.16% / 5.50% | 1.28% | −17.093233 | 480.75 |
| 1 | 13.293247 | 14.56% / 12.57% | 7.18% | −14.008428 | 158.60 |
| 2 | 12.918826 | 40.59% / 34.02% | 35.11% | −10.672196 | 62.18 |

The hard-volume row errors are 0.73%, 1.36% and 2.10%, respectively.
CPU costs decrease with radius because most wider proposals clash and do
not require clouds. A lower cost per trial therefore does not mean better
physical sampling.

The exact integral is nondecreasing with region size. The radius-two point
estimate lying below radius one's is sampling noise, not a physical decrease.
The shared smaller regions expose the loss of effective sampling directly:

| Selected region | R = 0.5 source | R = 1 source | R = 2 source |
| --- | ---: | ---: | ---: |
| ρ ≤ 0.5 | 12.054798 ± 4.16% | 12.293098 ± 28.27% | 10.577835 ± 100% |
| 0.5 < ρ ≤ 1 | Outside support | 12.834658 ± 16.12% | 11.810005 ± 49.56% |
| 1 < ρ ≤ 2 | Outside support | Outside support | 12.363387 ± 62.54% |

Entries are log Q with the **relative standard error of Q**, not an additive
error on log Q. In the smallest region, the respective numbers of valid
weighted draws are 14,547, 207 and **one**. The large-source estimates are
weak controls, not evidence against the better-sampled small region.

![Local and matched-region estimates](../runs/ab-intermediate-new-peak-figure-20260920/peak-neighborhood.png)

All observed radius-0.5 poses lie at old weighted radius above 32; this is
an observation, not a certified bound on that ball. The radius-one estimate
in the old r > 32 piece is log Q = 13.284844 with 14.68% row error. The
radius-two remainder intersection is 12.344899 with 35.91% error. These
local pieces overlap the historical complete reference and are not added
to it.

### A disjoint sum at several scales

Use the radius-0.5 reference for [0,0.5], radius-one reference for (0.5,1],
and radius-two reference for (1,2]. These pieces are disjoint, use independent
source campaigns, and retain their original full N. Their **linear means**
and independent sampling variances can be added. This gives:

| Covered local ball | log Q | Row / population-group relative SE | Largest draw | Total sampler CPU seconds |
| ---: | ---: | ---: | ---: | ---: |
| ρ ≤ 1 | 13.212046 | 11.13% / 5.79% | 7.79% | 639.35 |
| ρ ≤ 2 | 13.568313 | 20.30% / 20.29% | 18.34% | 701.53 |

The three pieces contribute 22.01%, 48.02% and 29.97% of the radius-two
point estimate. The outer shell remains poorly resolved, and no saturation
with radius has been established. This derived sum shares data with the
original full-ball controls; comparison between them is not an independent
validation. The reduction in observed radius-two error costs additional
CPU time and is **not a demonstrated speedup per CPU**. It demonstrates
why preserving measurements at smaller spatial scales matters.
For radius two, the relative error also falls because the combined estimate
is larger: its observed absolute variance is only about 8% smaller, while
its CPU cost is 11.3 times higher. Observed variance times CPU is about
10.3 times worse than the direct radius-two reference. Neither noisy
variance estimate establishes a general efficiency ranking.

The neighborhood now has measured integrated weight, rather than only a
large selected point value. It is still not a complete basin or a global
intermediate-region normalizer. Further coverage of the rest of 2 ≤ q < 5,
q ≥ 5, native and shoulder tails, SMC discrepancies and reversible
trajectory mixing remains necessary before judging assembly thermodynamics.

## Reproduction and audit artifacts

The reusable preparer and unchanged launcher reproduce these references in
fresh output directories. For the radius-one campaign:

```bash
python tools/prepare_peak_neighborhood.py --out runs/fresh-peak-preparation \
  --campaign-root runs/fresh-peak-reference --seed-base 101201010
python tools/run_latent_region_campaign.py \
  --out runs/fresh-peak-reference/r1 \
  --config runs/fresh-peak-preparation/config.json \
  --region runs/fresh-peak-preparation/region-r1.json \
  --binary runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer \
  --samples 16384 --replicates 4 --workers 4 --seed 101301010 \
  --lambda-ratio 64 --cloud-replicates 2
```

The preparer records all three campaign paths and seed sets before any
draws. The example uses new physical streams; reusing a seed reproduces a
stream rather than providing an independent replication. Geometry probes
retain their original reproducible seeds. The other two declared source
balls must also be run for the full three-campaign audit.

`tools/analyze_peak_neighborhood.py` performs the complete row audit and
cross-chart poststratification. `tools/audit_peak_neighborhood_extremes.py`
performs selected atom checks. `tools/combine_peak_neighborhood.py` derives
the independent disjoint-shell totals from the immutable audit, without
resampling or changing an original estimator. Main artifacts:

- `runs/ab-intermediate-new-peak-audit-20260920/analysis.json`
- `runs/ab-intermediate-new-peak-extremes-20260920/analysis.json`
- `runs/ab-intermediate-new-peak-multiscale-20260920/analysis.json`

The figure's input hashes, archived data and plotter are stored alongside
its PNG and SVG. Nineteen relevant tests pass: four independent geometric
and preparation tests, four masked-estimator tests, and eleven existing
partition/local-reference checks. The geometric tests cover finite
rotations, noncommuting frames, angular-length invariance and freezing
before draws; estimator tests include original-N normalization and
same-row covariance. These validate the implemented constructions and
audits, not unobserved statistical tails.
