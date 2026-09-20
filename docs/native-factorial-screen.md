# Native factorial screen: the two-neighbor weight remains unresolved

The fixed-budget screen completed, but it does **not** establish thermodynamic
cooperativity. One near-native AB pose contributes 98.52% of the entire AB
estimate; its effective sample size is 1.03. The hard-support volumes are much
better determined and provide a useful geometric control.

All calculations integrate over the same complete native region Ω, with the
original 2 Å / 15° metric, repaired shape, capture sphere, normalized Haar
measure, and bath radius 1.5 Å / activity 0.035 Å⁻³. The conditional mass is
`Q_S(z)=∫Ω H_S exp[z |E_m ∩ union_j∈S E_j|] dx dR`. This uses the full
exclusion union for AB. Fixed-neighbor background terms are outside this
conditional insertion normalizer.

## Results and concentration

| Fixed neighbors | Unconditional draws | Hard-native draws | log Q(z) | Weighted ESS | Largest weight | Q(0), Å³ |
|---|---:|---:|---:|---:|---:|---:|
| Empty | 1,048,576 | 18,029 | −8.70790 | 18,029 | 0.0055% | 1.65275×10⁻⁴ |
| A, reused independent reference | 33,554,432 | 20,041 | 15.26037 | 11.32 | 22.43% | 5.74122×10⁻⁶ |
| B | 4,194,304 | 4,897 | 12.27129 | 80.99 | 4.39% | 1.12229×10⁻⁵ |
| AB | 4,194,304 | 872 | 35.84604 | 1.03 | 98.52% | 1.99844×10⁻⁶ |

The factorial point estimate is

`log C(z) = log[Q_AB(z) Q_empty / (Q_A(z) Q_B(z))] = −0.39352`.

Its observed independent-population delta-method SE is **1.0151**. This SE
does not certify tail coverage, and the AB concentration precludes a
thermodynamic conclusion about the sign or magnitude. The four AB population
log masses are 30.5946, 37.2209, 32.6204 and 28.2878. Replacing the exceptional
draw or dropping that population would invalidate the original estimator.

The hard-only comparison gives `log C(0)=1.63435`, with observed population
SE 0.03393, or `C(0)=5.1261`. Thus the same-pose hard-validity indicators are
positively associated within Ω. The hard volume ratio `Q_AB(0)/Q_A(0)` is
about 34.8%, consistent with the earlier 35.4% geometry screen. The observed
activity-induced change `log C(z)−log C(0)=−2.02787` has SE 1.0169 and inherits
the unresolved AB weight. No equilibrium, global native-probability, or
assembly conclusion follows from these factorial point estimates.

These factorial contrasts include shared-pose correlations and hard support.
They do not isolate the many-body depletion penalty. That requires the
pair-additive reference `Q_AB^pair=∫Ω H_A H_B exp[z(C_A+C_B)]` on the same hard
support. Since `C_AB=C_A+C_B−|E_m∩E_A∩E_B|`, the isolated penalty
`−log(Q_AB/Q_AB^pair)` is nonnegative. No such counterfactual was estimated in
this screen.

## The dominant original draw

The largest AB contribution is population **r01, draw 1030265**, with
`q=0.14934976`, center `(0.066414, 0.075854, −0.054411) Å`, and orientation
quaternion `(0.99998470, 0.00002589, 0.00513557, −0.00205324)` in the archived
scalar-first convention. Its two independent cloud counts are 3635 and 3456;
their log weights are 56.35772 and 53.58247. The original linear-average
weight has log 55.72504.

Using only those same selected clouds, `Ĉ=(K₁+K₂)/(2λ)=1582.81 Å³`, where
`λ=64z=2.24 Å⁻³`, gives `zĈ=55.39844`. Its plug-in Poisson SE is 0.65788 in
`zC` units. The original log mean is 0.32660 above this plug-in value. The
second-ranked pose, at q=0.37899, has counts 3246 and 3295 and `zĈ=51.10156`.
Both first-pose cloud counts exceed both second-pose counts, which is
consistent with a geometric contribution as well as cloud noise. This is
a **selected-cloud diagnostic**, not an independent determination of either
overlap volume or a post-selection confidence interval.

For a fixed pose, `K~Poisson[λ(C−L)]` and the positive weight is
`W=exp(zL)(1+z/λ)^K`. Its conditional squared coefficient of variation is
`exp[z²(C−L)/λ]−1`, divided by the cloud count when independent clouds are
averaged. The selected-data plug-in relative SD of this two-cloud weight is
about 83%. Cloud noise is therefore material. If the concentration reflects a narrow
physical peak, cloud refinement alone would not resolve its small pose volume. No
resampling or estimator-weight replacement was performed.

The exact two leading original rows, original sample-file hashes and line
numbers, manifests, counts and diagnostic formulas are preserved in
[dominant-AB-original-diagnostic.json](/home/xvg/tetramer-mc/runs/native-factorial-screen-20260920/dominant-AB-original-diagnostic.json).

## Reproducibility and uncertainty calculation

The execution record is separate from the earlier prepared-only plan:

- [Execution manifest](/home/xvg/tetramer-mc/runs/native-factorial-screen-20260920/manifest.json)
- [Preflight geometry and identity checks](/home/xvg/tetramer-mc/runs/native-factorial-screen-20260920/preflight.json)
- [Final streamed factorial assessment](/home/xvg/tetramer-mc/runs/native-factorial-screen-20260920/factorial-assessment.json)
- [Analyzer controls](/home/xvg/tetramer-mc/runs/native-factorial-screen-20260920/analysis-controls.json)

Empty used 4×262,144 draws; B and AB each used 4×1,048,576. Seeds were
98781010, 98791010 and 98801010 respectively, plus 1009 times the replicate
index. A reuses the separate eight-population reference without pooling its
earlier small pilot. All targets use two clouds and λ/z=64. A single pool
enforced at most eight workers; all twelve new populations exited normally.
New simulation cost was 152.85 CPU seconds and the longest individual
population took 26.89 s. The new unconditional sample archives total about
477 MB. Every row, zero, sample-file hash, and physical signature was audited
by the streaming analyzer.

The frozen executable SHA-256 is
`f039c771388bb1a9e36e5632300459a5ff45e60c4972a5b8c88d16437705ad7d`.
The preflight compared all non-neighbor configuration fields exactly, allowing
only the `cooperative_plan` metadata annotation. An independent atomic
radius-group KD-tree check found minimum native–A, native–B and fixed A–B
gaps of 0.03713, 0.03692 and 0.03741 Å. The reference pose and fixed assembly
are hard-valid; distant periodic images were excluded by bounding spheres.
The recorded core-gap lower bound is 204.69 Å, exceeding the 3 Å required
to exclude inflated-sphere interactions as well. The runner now enforces
that stronger condition; `preflight-exclusion-check.json` records the check
on the completed campaign without changing its estimates.

Within each target, let population estimates be `X_k=Q̂(z)` and `Y_k=Q̂(0)`.
All population sizes within a target are equal, and targets use independent
random streams. For a target with K populations, the observed variances of
`log Q̂(z)`, `log Q̂(0)` and `log[Q̂(z)/Q̂(0)]` are respectively

`sampleVar(X_k/X̄)/K`, `sampleVar(Y_k/Ȳ)/K`, and
`sampleVar(X_k/X̄−Y_k/Ȳ)/K`.

The last expression retains the physical/hard covariance. Target variances
add in the factorial log contrast; the empty term cancels exactly from its
activity-induced change. Four numerical controls verify proportional-weight
cancellation, anticorrelation, all-zero estimates and nonestimable
single-population uncertainty. Physical mass estimates are unbiased in the
stated idealized estimator model; finite-sample logarithms and ratios are
not. Observed replicate errors cannot establish rare-tail convergence.

Analysis and diagnostic commands, after the completed production run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/test_native_factorial_analysis.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/analyze_native_factorial_screen.py runs/native-factorial-screen-20260920
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/diagnose_native_factorial_weights.py runs/native-factorial-screen-20260920
```

Exact executed analyzer sources are archived under the run's `provenance/`;
the earlier analyzer and assessment were retained before the null-uncertainty
guard was added. Nonzero campaign statistics are unchanged by that guard.

## A bounded next proposal control, not yet implemented

The very small q of the dominant event motivates a normalized mixture of
nested geometric covers at prespecified scales such as 0.1, 0.2, 0.4 and 1.
Each scaled cover shrinks the proposal translation radius and angular cap.
The target remains the original Ω, with its original q, bath and hard
predicate. Running the existing executable with scaled target metadata
would instead change Ω and is **not** this control.

For fixed mixture probabilities α_i and exact cover volumes V_i, draw a
cover then a uniform pose in it. Every pose uses the complete mixture density
`g(x)=Σ_i α_i 1_(cover_i)(x)/V_i`, and the physical estimator is
`1_Ω H W/g(x)`. The scale-1 component must have positive probability to
preserve coverage of all Ω. This follows directly by importance sampling;
neither Gaussian fitting nor selecting only the most favorable cover is
required. The smaller covers need not themselves equal native subregions.

Before a fresh fixed-budget comparison, controls must verify normalized
Haar/translation measures, nested-cover membership, overlap in the full
mixture denominator, and identical original target labels. Zero-activity
sphere/known-volume cases, rotated frames, off-center member geometry, and
the existing complete-cover reference provide independent checks. Proposal
probabilities and budgets must be frozen in advance. This control addresses
geometric concentration; cloud-count changes, if tested, are a separate
variance/cost parameter. No implementation or new production is included
in the completed screen.
