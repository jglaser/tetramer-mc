# Separately centered competing-contact guides

The [broader-source comparison](context-broadened-guide.md) left competing
contact weights concentrated in approximately one to four effective
importance contributions per population. Pooling the existing Poisson
counts removed some auxiliary noise but did not resolve this coverage
problem. Important competing poses lie far outside the fitted source
Gaussian; increasing that Gaussian's width is an inefficient way to
describe another basin.

This stage constructs local proposals around two competing contact
environments. The physical model, 263 fixed tetramers, mobile label 77,
anchor 16 and original source-contact definitions remain unchanged. It
uses radius 1.5 Å and activity 0.035 Å⁻³ in the historical 500 μM frozen
environment. It is a conditional proposal experiment. Finite-system
assembly near 106.8 μM requires the separate physical checks in the main
plan.

## Three distinct pose roles

The original source pose defines the contact reference. The training
initial pose determines where a local chain starts. The Gaussian chart
center determines its coordinate system. Changing either of the latter
must not redefine the contact regions or the physical target.

```mermaid
flowchart LR
    R[Original source and fixed environment] --> C[Unchanged contact classifier]
    R --> P[Unchanged physical target]
    I[Competing initial pose] --> T[Four local training trajectories]
    P --> T
    T --> D[Retained poses including rejections]
    O[Frozen local chart center] --> E[Encode poses in six coordinates]
    D --> E
    E --> F[Fit mean and covariance on streams 0 and 1]
    E --> V[Validate on streams 2 and 3]
    F --> Q[Frozen normalized Gaussian component]
    O --> Q
    Q --> N[Fresh importance comparison]
    C --> N
```

For poses relative to the same unchanged anchor, write the chosen center
as (t_c,R_c). Coordinates are ξ=(t−t_c,ℓu), where the Cayley rotation of u
is RR_c⁻¹. Sampling ξ=μ+Lη with η standard Gaussian and LLᵀ=Σ gives

\[
t=t_c+\xi_t,\qquad R=\operatorname{Cayley}(\xi_r/\ell)R_c.
\]

Relative to translation volume and normalized Haar measure, the density is
the six-dimensional Gaussian density divided by

\[
J(\xi)=\frac{1}{\pi^2\ell^3(1+|\xi_r/\ell|^2)^2}.
\]

Relocating the chart is a proper rigid-coordinate transformation and adds
no further volume factor. A Gaussian mean shift within the original
Cayley chart is a different operation; it cannot relocate that chart's
rotation singularity. The Gaussian map and physical kernels are reused.

## Configuration and reference preservation

`examples/context_source_guide.rs` accepts an optional field:

```json
{
  "chart_center": {
    "schema": "source-chart-center-v1",
    "frame": "saved-spherical-center",
    "pose": {
      "position": [0.0, 0.0, 0.0],
      "orientation": [1.0, 0.0, 0.0, 0.0]
    },
    "provenance": "Frozen selected-pose artifact and row identity"
  }
}
```

This object belongs inside `source_chart`; the numbers above are only a
format example. Actual positions and quaternions use the saved physical
sphere-center frame. Display-origin offsets are not applied. The field
is included in configuration hashes and echoed in the proposal manifest.
When absent, serialization and the original source-centered behavior are
preserved. The independently implemented Python density resolves the
same chart center. Original source validity and patch-reference checks
continue to use the original source metadata.

## Training allocation

Seeds are selected from the completed pilot, using the pooled-count
importance contribution outside A_T and unbound regions. Selection is
pilot-informed training, not fresh validation. The highest contribution
starts the first cage. The second is the highest remaining contribution
whose full contact-token Jaccard distance from the first is at least 0.5;
stable stratum and ordinal identifiers break ties.

The selected poses are `baseline-pop1-diagonal/1815`, a partial source
contact with neighbors {16,217}, and `baseline-pop1-full/1483`, with
neighbors {16,56}. Their contact-token sets are disjoint. The first is
8.28° from the original source orientation, so its large standardized
distance is not evidence for a Cayley singularity. The second is about
94.17 Å and 70.16° from the first.

Each seed gets four independent local-only trajectories, each with 2,304
cycles of five elementary attempts. The first 256 cycles are designated
warmup and the remaining 2,048 supply 10,240 retained states per stream.
All rejected states remain in the trajectories. Translation and rotation
proposal sizes are 0.2 Å and 1°, matching the prior local controls.
Streams 0 and 1 train one Gaussian per cage; streams 2 and 3 are held out.
The predeclared covariance regularization is reported explicitly.

The total allocation is eight chains and 92,160 attempted moves, with at
most four new workers and one thread per worker. Per-chain limits are
600 CPU seconds, 1,200 wall seconds, 4 GiB and two billion test points.
Failed prefixes are retained without replacement. Training does not
establish equilibrium or a region normalizer. Movement, contact residence,
autocorrelation, between-stream differences and held-out likelihood must
be reported; a constant observable cannot establish an independent-sample
count.

The fitted proposal must be frozen before a fresh importance comparison.
That comparison retains the uniform defensive component and evaluates
the complete mixture density. The physical target and region definitions
must be the same in every arm. Improved training likelihood or accepted
move throughput alone cannot establish physical convergence or assembly.

## Correctness of the frozen guide

Training changes a proposal, not the physical distribution. Once a chart's
center, mean and covariance are frozen, it is a normalized density on
translation times normalized rotational Haar measure. The omitted Cayley
seam has Haar measure zero. The physical wall and hard-core predicates
still contribute zeros to the importance estimator; draws are not
conditioned on passing them.

For fixed stratum fractions a_j and normalized runner densities q_j,
use q(x)=sum_j a_j q_j(x). With n_j=N a_j independent draws from each
q_j and an unbiased nonnegative depletion estimator W_hat(x),

\[
\mathbb E\!\left[\frac1N\sum_j\sum_{i=1}^{n_j}
 \frac{\mathbf1_R(X_{ji})\mathbf1_{\rm valid}(X_{ji})
       \widehat W(X_{ji})}{q(X_{ji})}\right]
=\int_R\mathbf1_{\rm valid}(x)W(x)\,dx.
\]

This identity uses the complete mixture density and the unconditional
draw count. Separate chart Jacobians are already included in each q_j;
there is no further Jacobian at the mixture level. Freezing a guide
allows this argument conditional on the training record. Reusing the
same record as evaluation draws would require a different argument.

The new chart-center implementation passed 16 Rust tests and 13 Python
tests covering independent coordinate/density reconstruction, a
noncommuting rotation, the Haar Jacobian, invalid centers and legacy
behavior. The isolated build is
`results/source-chart-center-guide-build-20261005`; its example binary
SHA-256 is
`2dacc7ebe2530b2c0cac94dbbaeb339afd40bea4c87c6e3c45545fb68b948549`.
The production assembly executable remains unchanged.

## Completed local training

All eight predeclared chains completed without replacement: 92,160 attempts,
1,108.33 total sampling CPU seconds and 297.86 seconds controller elapsed
time. All 13 owned controller, driver and sampler process groups were
confirmed drained. The existing growth and contact-weight campaigns were
left running.

The partial-source-contact chains accepted 135–187 of 11,520 attempts
each; the different-neighbor chains accepted 504–547. These counts show
that the training record contains motion; they do not measure equilibrium
contact exchanges or justify an independent-sample count. Fitting and
contact diagnostics retain every rejection.

The frozen training manifest is
`results/context-competing-cage-local-training-20261005/controller-manifest.json`,
SHA-256 `f2951a70ca0bf2b6f4af9104e11c8476e97e4c5248fd20ebf3a083065f6386c8`.
Its controller receipt is
`6f96b08eb947e221e57b79bfb7562cbb49d6e89bfafa19860954bd567d48c815`;
the host drain receipt is
`5dfa13595cb6a5dedbe5bf4fed66c73b9f83b87b791e8ddbc6e9afefe61036f5`.

## Held-out results

The [fit report and method](competing-cage-covariance.md) retain all 92,160
states and pass ten additional synthetic controls. Both empirical
covariances have rank six, with only about one millionth of their scaled
variance introduced by regularization. However, the nominal 95% Gaussian
region covers 39.3% of held-out residence states for the partial-contact
cage and 86.0% for the different-neighbor cage. Apparent per-stream vector
ESS is only 4.5–9.5 and 7.9–16.3, respectively. The guide is normalized;
the short local record is not a converged equilibrium basin model.

![Training and held-out covariance and coverage](../results/competing-cage-covariance-plot-20261005/result/comparison.png)

The top panels project the fixed training covariance and each held-out
stream's moments onto the two leading scaled training directions. The
ellipses are Gaussian moment summaries, not measured density contours.
The bottom panels evaluate all six coordinates. They expose between-stream
disagreement that a training likelihood alone would miss. These held-out
results do not change the fitted parameters.

The next proposal comparison keeps the old broadened law Q and adds the
two frozen cage runner laws r_0 and r_1 as
Q_new = 0.5 Q + 0.25 r_0 + 0.25 r_1. Each runner already contains 50%
uniform draws, so the full proposal retains 50% uniform coverage and
Q_new >= 0.5 Q pointwise. This bounds the importance-weight inflation of
an already covered pose by two; it does not guarantee improved realized
ESS or CPU efficiency. Fresh physical evaluation, including the complete
remaining contact region, must establish any improvement.

## Contact audit

All eight independent observers completed and their 13 owned process
groups drained. They audited all 92,160 attempts, including 81,920
production endpoints, using a cache of 2,761 distinct poses. Classification
used 18,168 body-pair queries and 57.08 observer CPU seconds. No additional
poses or depletant clouds were generated.

| Production contact diagnostic | Partial-contact start | Different-neighbor start |
|---|---:|---:|
| Distinct full fingerprints per stream | 2–4 | 8–15 |
| Dominant fingerprint occupancy | 85.85–99.92% | 44.07–61.51% |
| Instantaneous fingerprint passages | 4–23 | 94–154 |
| Changes of neighbor environment | 0 | 0 |

Neighbor ESS is undefined because the observed neighbor sets are constant.
Fingerprint ESS is especially unreliable for rare minority observations:
one stream has only eight minority endpoints and four passages but a
sample-centered apparent ESS of about 1,640. That number cannot establish
contact-sampling efficiency. Conversely, no observed environment exchange
does not establish that exchanges are physically absent or that both
environments should have comparable equilibrium occupancy.

The observer's neighbor-defined A/B labels are not the finer A_T mass
regions or a native-registry classifier. Full patch tokens are archived;
source-patch completeness was not separately reduced in this audit.
The exact metrics are
`results/context-competing-cage-observers-20261005/contact-summary.json`,
SHA-256 `138258bada2425794d0d601b686b070bd9f28e0806e9bccb3da6473db04c594e`.
All observer inputs and the protected assembly binary remain unchanged.

## Frozen importance comparison

The next comparison fixes four independent populations of 4,096 attempts
per arm, with the following deterministic stratum counts in each
population. F, D and B are the existing full-covariance, diagonal and
four-times-covariance guides; C0 and C1 are the two new frozen fits.

| Runner law | Baseline | With competing cages |
|---|---:|---:|
| F | 1,536 | 768 |
| D | 1,536 | 768 |
| B | 1,024 | 512 |
| C0 | 0 | 1,024 |
| C1 | 0 | 1,024 |
| Total attempted poses | 4,096 | 4,096 |

Each runner law is r_j=0.5 U+0.25 G+0.25 g_j. Thus 1,024 C0-runner
attempts supply 256 expected direct C0 Gaussian draws, not 1,024. The
comparison contains 32,768 fresh poses across 32 strata. Its independent
audit reconstructs every pose and source density, and checks a fixed
stride-64 panel of 512 poses against independent atlas and atomic geometry
calculations. The complete remaining contact and unbound regions remain
in the integration domain.

Physical scoring is predeclared to use two independent clouds of intensity
lambda=2.24 Å⁻³ for every valid pose. With a common certified overlap-volume
lower bound L and cloud counts K0,K1, the primary estimator is

\[
\log\widehat W_{\rm RB}=zL+(K_0+K_1)\log(1+z/(2\lambda)).
\]

It is the conditional expectation of the earlier arithmetic-pair estimator
given the pooled count; the arithmetic-pair result remains a separately
labeled control. The [existing Poisson argument](context-broadened-guide.md)
establishes unbiasedness and variance reduction at fixed pose. It cannot
repair missing pose-space coverage. Zero-volume cloud records and every
hard-invalid zero are retained. No outcome from this comparison has been
used to choose its mixture weights, Gaussian widths or estimator.

The five-guide manifest is
`results/context-multicage-frozen-guides-20261005/result/mixture-manifest.json`,
SHA-256 `7e54d2b6140806d8f917818e2072391df038e39d524346f8996aeb9527be0563`.
Its 11 arithmetic and metadata controls passed; freezing it generated no
poses or depletant clouds.

## Completed prospective comparison, 6 October

All 32 strata completed, retaining every one of the 32,768 attempted
poses. Independent reconstruction checked the complete five-chart
mixture density at every draw, and all 512 predeclared independent
geometry checks passed. There were 7,984 hard-valid poses, each scored
with two fresh independent clouds. The 15,968 clouds processed
1,935,881,890 points. No failed draws were replaced or omitted.
Geometry cost 498.50 CPU seconds; physical scoring cost 612.09 CPU
seconds. All 37 owned producer process groups and both audit groups
were confirmed drained. The assembly binary and ongoing campaigns were
unchanged.

**The competing guides improve coverage, but neither arm converges.**
The table uses the primary pooled-count estimator and four independent
population estimates. ESS/CPU is the arithmetic mean of each population's
importance ESS divided by its geometry-plus-scoring CPU time. It is not
trajectory contact ESS or an assembly speedup.

| Conditional diagnostic | Existing guides | With competing cages |
|---|---:|---:|
| Contact mass outside source-complete A | 1.05 × 10¹⁴ | 2.35 × 10¹⁶ |
| Relative population SE of that mass | 96.8% | 20.9% |
| Importance ESS per population, that region | 1.00–3.18 | 4.14–17.63 |
| Importance ESS/CPU, that region | 0.0139 | 0.0798 |
| Importance ESS/CPU, neighbor environment B | 0.0103 | 0.1534 |
| Source-complete A mass | 1.67 × 10³⁷ | 3.18 × 10³⁷ |
| Relative population SE, source-complete A | 11.8% | 36.6% |
| Importance ESS/CPU, source-complete A | 0.1835 | 0.0649 |

Masses use translation volume times normalized Haar measure, with the
same depletion reference in both arms. They are conditional integrals
in this frozen environment, not chemical potentials or native assembly
probabilities. Source-complete A means neighbors {16,217} with all
sixteen original source patch tokens; it is not independently determined
native registry.

![Completed physical comparison](../results/context-multicage-physical-plot-20261006/result/comparison.png)

The competing-contact estimate increases by approximately 224-fold
(5.41 log units, 4.76 combined population SE), almost entirely from
partial-A configurations reached by the new cage0 component. The largest
competing contribution in each new population lies within that chart at
squared Mahalanobis distance 2.92–10.27. The gain therefore comes from
sampling a separately located contact basin. It cannot be interpreted
as an equilibrium change: both arms integrate the same physical target.
The baseline demonstrably lacks adequate coverage in this comparison.

Source-complete A still has a serious tail problem. One broad-guide draw
supplies 84.8% of its population's weight in the multicage arm. Its
squared distances are 26.18 in the original full-covariance chart and
6.55 in the broad chart; it is not a Cayley-seam excursion. Two auxiliary
clouds at that pose do not by themselves identify whether cloud noise
or pose-weight mismatch dominates its contribution. Saved pooled counts
permit a separate second-moment diagnostic before spending on more
clouds or changing the guide again.

Both arms leave the 50–75% and 75–100% source-patch-completeness bins
without observations, and each has only one observation in the 25–50%
bin. Other-contact mass remains unstable (about 90% relative SE in the
new arm). Unbound mass agrees within 0.0083 log units and passes the
concentration/error checks, but that does not bound missed contact
weight. Every contact-region concentration check fails. The approximate
population-based free-energy interval half-widths are 2.75 and 1.67 kBT,
above the required 0.5 kBT; their central values are not admitted as
physical conclusions.

The [completed saved-count diagnostic](context-poisson-noise-budget.md)
separates auxiliary-count variance from pose-weight variance, retaining
fixed proposal strata, all zeros and the complete population denominators.
Its descriptive estimates favor improving pose coverage before buying
substantially more clouds, although ratio uncertainties remain unbounded.
Neither that diagnostic nor another conditional guide comparison can
decide finite-system assembly. The original-condition contact campaign
and subsequent finite-system checks remain necessary.

### Reproducibility

Fifteen additional physical-preparation and reduction controls passed.
An earlier synthetic-test prefix contains two NumPy-scalar serialization
fixture failures; the corrected fixtures were run under a fresh prefix.
No physical job was launched from the failed prefix. The rendering
preparation initially encountered the system interpreter's missing
`hashlib.file_digest`, before copying inputs or launching any job; the
project interpreter completed the unchanged plot preparation.

- Geometry audit:
  `results/context-multicage-guide-independent-audit-20261005/result/report.json`,
  SHA-256 `c69a028219799009d79ef9355d2aa05a7e96e1fa795ff0b8a5852cd8a7232168`.
- Physical manifest:
  `results/context-multicage-guide-physical-20261005/controller-manifest.json`,
  SHA-256 `a4d58b4fc628e1c42b3c96eba0bfa01538e26c61adb1103bfaf4f5096a4af822`.
- Physical audit:
  `results/context-multicage-physical-independent-audit-20261005/result/report.json`,
  SHA-256 `952a1ca4e5325ee44c1ca81af2753e3cba456fcbc91050c4be325c54edd24dac`.
- Producer drain receipt:
  SHA-256 `3af03b4974d84b13aefd25401c920d31457684f4d3092c0e7d9581a019421386`.
- Plot:
  `results/context-multicage-physical-plot-20261006/result/comparison.png`,
  SHA-256 `e00b07f599abe0599ec97a00cfbee8a0a006a2ad9f1c31fc6d3db5683846d877`.

An audit's `passed` field means its reconstruction and accounting checks
passed. Its separately reported physical convergence gates remain failed;
the finite-system stability conclusion remains unresolved.
