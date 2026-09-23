# Bounded SMC terminal geometry guide preparation

`tools/prepare_smc_geometry_guides.py` creates four fixed normalized importance
proposals from already completed and authenticated **narrow** SMC terminal
geometry. It performs no physical sampling, classification, or audit replay.

The preparation tool declares candidates before it loads terminal coordinates: one Gaussian
per independent fitting population and each inside/outside group of the full,
unfiltered historical R5 chart ball, optionally subdivided by the sign of
original current-chart coordinate `u[0]`. Zero belongs to the nonnegative group.
Both covariance multipliers 1 and 4 are retained, producing `r5-cov1`, `r5-cov4`,
`r5-sign-cov1`, and `r5-sign-cov4`. There is no automatic winner selection.

Only narrow populations `r00` and `r01` are fitted. All of their terminal slots,
including identical repeated descendants, contribute to the arithmetic mean and
`1/N` centered second moment of their geometric group. A strictly positive
covariance floor uses the existing physical translation scale 0.05 Å and angular
scale 0.1 degrees transformed into the current latent chart. Every population
gets equal total Gaussian mass, and its two or four groups get equal mass.
An empty group remains as a component fitted to its entire source population.
Native indicators, classifier labels, overlap weights, and ancestor frequencies
never select or weight fit rows. Correlated descendants are fitting geometry;
they are not IID importance samples or independent evidence of convergence.

The saved old 80-component bank keeps every original mean and covariance. In the
new guide its Gaussian-component weights are halved; new SMC components share
the other half, and the guide's defensive uniform probability remains 0.5:

```
q_old = 0.5 U_R4 + 0.5 G_old
q_new = 0.5 U_R4 + 0.25 G_old + 0.25 G_smc >= 0.5 q_old
```

Gaussians are normalized over all six-dimensional latent space and are never
truncated at R4. Future independent importance draws must retain invalid or
out-of-region proposals as zero-weight draws. Only the new Gaussian covariances
receive the declared multiplier; the old bank remains unchanged.

The authentication chain is the completed evidence authentication, the completed
narrow audit and population records, then each terminal summary, status,
manifest, and input provenance. The declaration binds source/protocol/config,
region, old-guide, all four terminal summary identities, code closure, and
runtime. Large stage and initialization stream hashes remain bound through the
completed audit; those audits are not replayed. Each loaded terminal pose is
independently mapped through `Chart` and checked against its recorded latent
coordinate. Historical R5 membership uses its chart ball only, ignoring native
q, capture, and inner-radius filters.

Use the scientific environment, choosing a new output directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_smc_geometry_guides.py declare --out runs/NEW-PREPARATION
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_smc_geometry_guides.py prepare --out runs/NEW-PREPARATION
```

`declare` freezes all deterministic choices and authenticates identities without
evaluating terminal coordinates. `prepare` verifies the unchanged declaration,
fitting-source identities and runtime, then exports all four guides, component
slot metadata, mapping validation, fitting geometry, and a final freeze.
Existing declarations and completed preparations are never overwritten.
Populations `r02` and `r03` remain excluded from fitting; the fit does not read
their terminal coordinates. Aggregate terminal geometry had already been inspected
during proposal design, so their later descriptive density diagnostics are not
pristine validation. The declaration also fixes all four candidates, density
statistics, original importance classes and radial/angular/orthant strata, and
two-cloud/paired-cloud second-moment diagnostics. All candidates are reported;
there is no automatic winner, post-score refitting, or measured speedup claim.
The preparation freezes the scoring code and tests alongside the fitting code.

The historical chart and unchanged bank retain native-informed provenance, so
this geometric addition is **not template-free assembly proof**. Proposal-density
or reused-archive diagnostics alone cannot establish a new physical normalizer,
repair an existing convergence failure, or authorize larger physical production.


## Completed offline diagnostic

The declared preparation completed on 2026-09-23. Four guides contain 84 or 88
Gaussians: the unchanged 80 old components plus four or eight new components.
The score reuses all **2,228,224 attempted importance draws** from the five
completed confirmation arms. No protein simulation, classifier or audit was
rerun. The 25 focused tests passed, including independent-cloud finite-state
identities with hard-invalid zeros, complete mixture density, arithmetic
population pooling, and source-arm/cloud-law separation.

For an archived source-proposal draw, let `z` be the log of the mean of its two
full importance weights, and `p0,p1` the two individual log weights. The target
proposal's noisy second moment is estimated by

```
log M2(q_target) = logsumexp(2*z + log q_source - log q_target) - log N.
```

Replacing `2*z` by `p0+p1` estimates the physical second moment without cloud
noise inflation. All original attempted rows remain in N; region masks affect
only the numerator. Every source arm evaluates the same unchanged-bank target
density in its denominator comparison. The four populations are combined in
linear moment space within their source arm only. Intensity 256 is reported
separately and never used as a forecast for the intensity-128 noise law.

![Offline proposal tradeoff](../runs/smc-geometry-guide-report-20260923/second-moment-tradeoff.png)

For the four-new-component, covariance-times-one candidate, the observed noisy
second-moment ratios to the unchanged bank are:

| Saved source arm | Native inside old R5 | Native outside old R5 | Contact without native entry |
|---|---:|---:|---:|
| bank | 0.7640 | 0.0746 | 1.8244 |
| wide | 0.7450 | 0.0901 | 1.9282 |
| small | 0.7604 | 0.8056 | 1.9837 |
| defensive 0.2 | 0.7158 | 0.6093 | 1.8523 |
| intensity 256 | 0.7287 | 0.4528 | 1.9140 |

The paired-cloud physical moments show the same tradeoff. These are retrospective
**second-moment ratios, not measured variance reductions or speedups**. The
strongest apparent gain is particularly uncertain: one saved draw contributes
93.37% of the bank-source baseline complement second moment, and one population
contributes 95.30%. With the new candidate, the largest draw still contributes
42.90%, and second-moment contribution ESS is only 5.30. This ESS measures
concentration of the diagnostic, not independent physical contacts or prospective
importance ESS.

The two populations excluded from fitting show mean log-density gains of +0.333
outside the historical R5 chart for the four-component addition, and +0.379 for
the sign-split addition, giving each population equal weight. Yet the sign split
generally has worse native moment scores. Covariance-times-four loses held-out
density and worsens the old-R5 moment. Thus a geometric likelihood score alone
would not select a good importance proposal. These populations had undergone
aggregate inspection before this exercise; none of these results is pristine
post-selection validation.

The addition is useful as a **native-targeted candidate**. It does not supply
competing-contact geometry: all saved terminal SMC slots happen to be native,
although the fit never reads their labels. Every candidate diverts probability
from the old no-entry proposal and nearly doubles its estimated second moment.
Future independently frozen validation must preserve explicit no-entry coverage,
for example by keeping an unchanged-bank estimator alongside a native-targeted
one. No fresh allocation or automatic candidate selection is made here.

The fixed contact convergence gate remains failed. No physical mass estimate,
region definition, threshold, old allocation, or assembly kernel changed.
Full-vessel and finite-system assembly production remain gated; the physical
question is still an **unresolved sampling limitation**.

Artifacts:
[declaration and guide preparation](../runs/smc-geometry-guide-preparation-20260923/plan.json),
[all scores and held-out geometry diagnostics](../runs/smc-geometry-guide-diagnostics-20260923/summary.json),
[figure provenance](../runs/smc-geometry-guide-report-20260923/figure.json).

To score a completed preparation into a fresh directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/score_smc_geometry_guides.py --preparation runs/NEW-PREPARATION --out runs/NEW-DIAGNOSTICS
/home/xvg/protein-nucleation/.venv/bin/python tools/plot_smc_guide_diagnostics.py --source runs/NEW-DIAGNOSTICS/summary.json --out runs/NEW-REPORT
```
