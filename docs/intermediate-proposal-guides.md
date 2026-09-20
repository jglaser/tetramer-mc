# Frozen proposals for the intermediate contact region

The new guides target the original interval **2 ≤ q < 5**, with both AB
neighbors, capture radius 18 Å, depletant radius 1.5 Å, and activity
0.035 Å⁻³. They address a demonstrated mismatch between the earlier
shoulder guides and the intermediate contacts. Fitting and geometry probes
produce proposals; fresh independent physical integration is still required.

## Why the existing shoulder guides were unsuitable

`tools/diagnose_intermediate_frozen_guides.py` evaluated the previously
frozen shoulder mixture and equal-geometry Gaussian on 16 historical
intermediate poses and the 16 largest contributions from the new direct
intermediate reference. For this comparison each model used the full law

`0.25 product(q_max=5) + 0.75 [0.95 frozen_model + 0.05 cube/Haar]`.

At every one of the direct reference's 16 largest contributions, either
old guide supplied only 0.02246–0.02335 times the complete Cayley cover
density. Their learned Gaussian contributions were negligible; nearly all
probability came from the product/cube defense. The largest reference
contribution, at q = 3.26781, lay 75.26 Mahalanobis units from the weighted
component and 9.16 from the geometry component. Its guide/cover ratio was
0.0230193. The geometry-component distances over all 16 direct poses ranged
from 7.82 to 37.98. Only one of the 16 historical poses had a guide/cover
ratio above one: 1.509 for the old mixture and 2.994 for its geometry model.

Independent atom-union checks found positive hard gaps for all 32 poses.
The minimum gap was 0.00058237 Å; the largest reference contribution had
AB gaps of 0.140696 and 0.00641024 Å. The large weights therefore do not
come from a hard clash at the reported precision. The source's two cloud
weights remain separate in the diagnostic artifact.

A geometric-only importance diagnostic using all 3,335 valid poses among
524,288 direct cover draws estimated an old full-guide hit probability of
about 0.00014745. Its observed relative error was 1.73%, with weight ESS
about 3,335. Extrapolation gives only about 9.7 or 19.3 valid target poses
for 4×16,384 or 4×32,768 old-guide trials, respectively. These are observed
geometric diagnostics, not physical normalizer estimates or guarantees
against unvisited regions. Together with the pointwise density checks they
give a clear reason to skip the unchanged-guide campaigns.

The complete diagnostic is archived at
`runs/ab-intermediate-frozen-guide-diagnostic-20260920/analysis.json`.

## Fitting inputs and convention

`tools/prepare_intermediate_guides.py` freezes three new models in
`runs/ab-intermediate-guide-preparation-20260920`:

- **Weighted:** one full-covariance Gaussian fitted with the original full
  importance weights of all 3,335 valid reference poses.
- **Geometry:** one full-covariance Gaussian giving every one of those
  poses equal weight.
- **Mixture:** the two fitted Gaussians with fixed probabilities 0.5/0.5,
  selected as the primary model before geometry probes.

The source is the completed four-population campaign
`runs/ab-intermediate-cayley-reference-4x131072-l64-20260920`. Each population
contains 131,072 unconditional draws. Every valid original-window pose is
included; no historical poses are added, and no observations or weights are
clipped, trimmed, or normalized separately by population. The original
importance weights already contain the physical Jacobian. It is not applied
a second time during fitting.

The weighted fit has source weight ESS **1.10990** and largest normalized
weight **94.7912%**. Its covariance is therefore strongly influenced by one
noisy physical observation. The equal-row geometry covariance describes the
valid source proposal cloud; it is neither a uniform-hard-volume covariance
nor an equilibrium covariance. Its count of 3,335 must not be interpreted as
3,335 effective physical samples. Neither fitted model determines a new
physical normalizer or establishes physical convergence.

Both fits reuse `prepare_shoulder_guides.fit_guides` with the native
reference expressed relative to the A body frame, left Cayley rotations,
and angular length ℓ = 55.022831 Å. Translation–rotation covariance is
retained. The additive covariance floor is

`diag(0.05² I₃, [ℓ tan(0.1°/2)]² I₃)`.

The maximum source chart angle is 19.646°, safely below the declared
60° input guard. That guard checks the fitting data; subsequent Gaussian
draws are untruncated. The weighted covariance eigenvalues range from
0.003848 to 6.271822 Å², and the geometry covariance from 1.133711 to
10.087790 Å². Independent chart-density and laboratory/body-frame checks
agree to at most 5.08×10⁻¹⁰ in log density.

Every one of the 3,335 training poses was independently checked against both
AB atom unions. All gaps are positive; the minimum is 0.000191686 Å.
These results are retained in `source-atomic-checks.json`.

## Frozen geometry probes

The source file hashes, exact fit rules, model hashes, probe counts, and
seeds are recorded in `protocol.json` and `freeze.json` before probe RNGs
are initialized. Each model gets exactly 256 independent raw-model draws.
All failed draws remain in the denominator; there are no retries or
truncation, and no depletant clouds are generated. Validity means the
original interval 2 ≤ q < 5, capture, and hard compatibility with both AB
neighbors.

| Raw model | Seed | Valid / 256 | Valid fraction | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: |
| Weighted | 99941010 | 127 | 49.61% | 43.54–55.69% |
| Geometry | 99942019 | 137 | 53.52% | 47.40–59.53% |
| 50/50 mixture | 99943028 | 137 | 53.52% | 47.40–59.53% |

The mixture drew 121 weighted and 135 geometry components. Among valid
poses, both-neighbor exclusion contacts occurred for 127/127 weighted,
85/137 geometry, and 112/137 mixture draws. These contact predicates use
the 3 Å sum of depletant radii added to the hard gap; they do not quantify
overlap volume or statistical weight. The broad geometry model reaches
more weakly coordinated configurations, while the weighted model focuses
on the observed strong contact. Similar geometric hit rates do not imply
similar importance-weight variance.

The original source cost 80.75 CPU seconds, or 0.024214 seconds per valid
pose when its entire generation cost is divided by valid rows. Applying
that crude cost scale to 65,536 **pure raw-model** draws predicts roughly
787 seconds for weighted and 849 seconds for geometry or mixture. Doubling
the draw count doubles those figures. Different overlap envelopes, cloud
work and rejection overhead can change the cost substantially; this is a
planning estimate, not a runtime prediction from new physical measurements.

The future physical campaign's outer defense is separate from the 50/50
Gaussian model. Under the proposed existing hybrid law, a raw-Gaussian
draw has total probability 0.75×0.95 = 0.7125, with additional q_max=5
product and cube/Haar branches. The above probe fractions apply to that
raw-model branch only. Every physical weight must use the full summed
hybrid density, including the 0.5/0.5 component probabilities when the
mixture is selected. A complete geometric defense preserves support but
does not guarantee that a finite run samples every important contact.

## Reproduction and validation scope

```bash
python tools/prepare_intermediate_guides.py \
  --out runs/fresh-intermediate-guide-preparation
```

The completed `report.json` records all source and probe hashes, fit
parameters, independent density checks, geometry outcomes and cost caveats.
Archived source configuration, shape, region, raw file hashes and original
two-cloud numerators are checked before fitting. The separate read-only
`source-cover-external-audit.json` checks every source executable hash and
the archived region hash and exactly reconstructs the complete Cayley
chart. This external check was added after the preparation; it does not
change the frozen protocol, models, source poses, or probe results.

Fresh physical sampling from the frozen mixture and geometry controls has
completed: four populations of 16,384 draws per model under
`runs/ab-intermediate-guided-pilot-plan-20260920/protocol.json`, using seed
bases 99951010 and 99961010, respectively. The campaign directories are
`runs/ab-intermediate-mixture-4x16384-l64-20260920` and
`runs/ab-intermediate-geometry-4x16384-l64-20260920`. They use the full hybrid
law; their full-N results, weight concentration, population scatter and
CPU costs are reported below.

The new physical draws are independent conditional on the frozen proposals.
The original direct reference was used for fitting and is therefore
training data, not a separate held-out validation campaign. Its estimates
must not be retrospectively pooled with the guided results. A fresh draw
from the unchanged reference law would provide a separate validation if
needed. The fit and the 768 cloud-free geometry probes justify trying
these proposals; they establish neither a thermodynamic answer nor
convergence.

## Completed guided pilots

Both four-population pilots have now completed, with every raw row audited.
The qualified comparison is
`runs/ab-intermediate-guided-comparison-qualified-20260920/comparison.json`.
The target, physical model and full unconditional denominators are unchanged.

| Campaign | Draws | Valid poses | log Q | Observed relative SE | Weight ESS | Largest weight | CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct reference, used for training | 524,288 | 3,335 | 12.118911 | 94.92% | 1.11 | 94.79% | 80.75 s |
| Fresh mixture | 65,536 | 25,639 | 7.759717 | 69.87% | 2.05 | 67.85% | 600.38 s |
| Fresh geometry | 65,536 | 26,768 | 6.628094 | 32.50% | 9.47 | 19.08% | 447.36 s |

The geometric improvement is substantial, but the physical estimates remain
dominated by rare contributions. The mixture population log estimates are
9.00607, 6.17228, 6.18665 and 5.55709; geometry gives 6.74630, 6.85467,
7.03468 and 4.49563. Neither the lower geometry error nor the discrepancy
from the training reference establishes a converged physical answer.
The training reference is not held-out validation, and these campaigns
are not pooled.

The hard-region integrals agree much better: reference log Q₀ = −2.364224
(1.73% observed relative SE), mixture −2.385626 (1.73%), and geometry
−2.352069 (1.12%). This supports consistent geometry and normalization on
the observed samples, while leaving the depletion-weighted integral
unresolved.

![Intermediate estimates and largest-weight fractions](../runs/ab-intermediate-guided-figure-qualified-20260920/intermediate-weights.png)

The figure shows observed Q ± SE transformed to the log axis, rather than
assuming symmetric Gaussian uncertainty in log Q. Its lower panel shows
the largest observed weight fractions. These error bars do not bound
unvisited weight.

## Branch and extreme-pose diagnosis

In the mixture, 23,531 weighted-Gaussian draws produced 12,323 valid poses
but only **7.669%** of the observed estimator sum. The 23,335 geometry draws
produced 13,307 valid poses and **92.331%** of the sum. Product supplied
0.000152%, and cube zero. These are estimator-source contributions,
`αⱼ ∫ I exp[z C] gⱼ/g_full dx`, not physical basin occupancies or masses
of the Gaussians. Every weight uses the full hybrid density and original
N; random branch labels are not fixed-quota strata. Detailed branch counts
and q-band contributions remain in the diagnostic JSON.

The mixture's two largest contributions, 67.85% and 16.37% of its sum,
were geometry draws at q = 3.21151 and 2.19110. Their geometry radii were
3.207 and 3.101, versus weighted radii 10.326 and 17.303. About 99% of the
full proposal density at both poses comes from geometry. All 16 largest
geometry-only contributions likewise came from its Gaussian, at radii
2.924–3.777; product/cube accounts for at most 3.74% of their density.
Thus the fresh extremes are poorly represented by the weighted component
but are reachable within the broad geometry Gaussian, without relying
primarily on fallback proposals. Gaussian radius is not a local integrated
sampling probability.

Conversely, the old training maximum is at weighted radius 0.234, where
the full mixture density is 3.59×10⁸ times the complete reference density.
Fresh weighted draws reach a maximum log two-cloud Boltzmann mean of
24.652, above the training maximum's approximately 22.549. These checks
argue against simple inaccessibility of strong contacts under the weighted
proposal. They do not determine the mass or visitation rate of a fixed
neighborhood; the old maximum is an in-sample fitting point.

Paired independent clouds account for **2.858%** of observed mixture
variance and **15.317%** of geometry variance. Pose variation therefore
dominates this diagnostic. A weighted fit focused on training extremes
and a broad Gaussian with moderate distances still leave rare, dominant
physical weights. This is consistent with narrow regions of large weight
inside a broad pose distribution, but does not establish their number or
topology. A prespecified local reference integral in the weighted chart,
with a complementary guide check, would distinguish a noisy direct-reference
outlier from unresolved local mass more directly than another fit.

The read-only audit at
`runs/ab-intermediate-guided-tail-diagnostic-qualified-20260920/analysis.json`
recomputes all 131,072 new raw-row densities, q predicates, cloud numerators
and source moments. It also checks the 16 largest contributions from each
fresh guide and 16 from training under both full proposal laws. All 48
poses pass independent AB atomic checks (minimum gap 0.00058237 Å), and
selected-pose log densities agree to 1.12×10⁻¹³. No samples, fits, replacement
weights or pooled normalizers are produced. Reproduce it with
`python tools/diagnose_intermediate_guided_tails.py --out runs/fresh-tail-audit`.
