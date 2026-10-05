# Broader source guide: conditional physical comparison

The fixed-pose [overlap diagnostic](context-overlap-panel.md) found two
low-probability source-contact poses whose depletion-corrected importance
scores exceed those of the tested core poses. Depletion did not eliminate
those particular tails. This motivates a fresh comparison of proposal
coverage, with physical weights evaluated at every hard-valid draw.

This calculation holds 263 tetramers fixed in the historical 500 μM
environment and moves tetramer 77 relative to anchor 16. The depletant radius
is 1.5 Å and its activity is 0.035 Å⁻³. It is a conditional sampler diagnostic,
not a native-registry or finite-system assembly test. The assembly decision
near 106.8 μM remains separate.

## Frozen proposals and allocation

Let F and D denote the previously frozen full and diagonal source-covariance
guides. Each generating distribution is

\[
q_j=\tfrac12 U+\tfrac14 G_{\rm context}+\tfrac14 g_j.
\]

The added guide B has the same source frame and fitted mean as F and four
times its covariance, hence twice its standard deviation. It involves no
new fit and does not change the physical model. The comparison is

\[
q_{\rm baseline}=\tfrac12q_F+\tfrac12q_D,\qquad
q_{\rm broad}=\tfrac38q_F+\tfrac38q_D+\tfrac14q_B.
\]

Both retain a 50% uniform defensive component. Pointwise,
\(q_{\rm broad}\geq 3q_{\rm baseline}/4\), which limits inverse-density
inflation for existing poses to 4/3. It does not guarantee improved ESS.

Four independent populations of 4,096 draws are generated in each arm.
Each baseline population allocates 2,048 draws to F and 2,048 to D;
each broadened population allocates 1,536 to F, 1,536 to D and 1,024 to B.
The total is 32,768 fresh draws in 20 strata. Samples from the earlier pilot
are not pooled into this comparison. The predetermined audit panel takes
every 64th draw within each stratum, totaling 512 poses.

The Python audit reconstructs all generating densities, latent-coordinate
decodes and Jacobians, evaluates all three source densities, and evaluates
the complete comparison-arm density. It independently checks atomic and
patch geometry and the full atlas density on the fixed panel. Radial
partitions use squared chart radii 0, 6, 12, 24, 48, 96 and infinity.

## Physical estimator

Every hard-valid fresh pose receives two independent Poisson clouds at
intensity λ=2.24 Å⁻³. If the certified overlap lower bound is L and K is the
number of uncertain-overlap hits, the positive estimator is

\[
W=e^{zL}(1+z/\lambda)^K,\qquad E[W]=e^{zO}.
\]

For region R and a population with N=4,096 attempted draws,

\[
\widehat Z_R=\frac1N\sum_i
\mathbf1_{\rm valid}(x_i)\mathbf1_R(x_i)
\frac{W_{i0}+W_{i1}}{2q_{\rm arm}(x_i)}.
\]

The generating-stratum density must not replace the complete arm density.
The coordinate Jacobian is already included in q. Invalid draws contribute
zero; denominators never condition on validity or region membership.
Exponentiating an unbiased estimate of zO would give a different, biased
weight and is not used here.

A_T means exact exclusion-neighbor set {16,217} with all 16 source patch
tokens. T-any also allows additional neighbors. Neither is a native
classifier. The region partition includes partial source contacts, the
competing B environment, other contacts and unbound poses throughout the
full atomic-wall domain. Radial partitions under different charts are
reported separately and are not added together.

## Execution and interpretation

There are at most four new scientific workers, each using one thread.
Every attempted draw is retained. Every valid pose is scored, without
selection by region or estimated weight. An all-invalid stratum uses an
explicit zero-work receipt, preserving its full attempted-draw denominator.

Cloud caps are one million raw and processed points per cloud, two million
per pose, and two million times the original stratum allocation in total.
The root-envelope maximum Poisson mean is 740,601.7; a Chernoff union bound
over at most 65,536 clouds has log exceedance probability below −40,882.
This controls count-cap risk, not runtime. Physical jobs have 600 CPU and
1,200 wall seconds each. A failed or timed-out stratum leaves the campaign
incomplete; its prefix is never clipped, replaced or treated as zero.

The reducer reports linear physical and hard-only masses, between-population
uncertainty, contribution ESS, largest contribution, radial contributions
and ESS per geometry-plus-scoring CPU. These are independent-importance
diagnostics, not Markov-chain mixing times. Proposal agreement and
concentration checks cannot rule out unseen modes. This pilot cannot by
itself support either assembly or instability.

Frozen geometry data live under
`results/context-broadened-guide-geometry-20261005`; the mixture recipe is
`results/context-broadened-frozen-guides-20261005/result/mixture-manifest.json`.
The predeclared physical policy is
`results/context-broadened-guide-preparation-20261005/physical-policy.json`.

## Completed execution

All 32,768 geometry draws completed in 348.1 CPU seconds. The independent
audit passed all density/coordinate checks and its 512-pose geometry panel
in 28.0 CPU seconds. Ten synthetic audit controls passed.

There were 7,966 hard-valid draws. All received both clouds, including the
deterministic zero-envelope cases: 15,932 clouds and 1,993,451,940 raw points,
all processed. Scoring took 651.8 CPU seconds and 194.6 seconds elapsed with
four workers. Every scientific job completed, all owned process groups
drained, and the production binary was unchanged.

The complete physical bank is
`results/context-broadened-guide-physical-20261005-v2`. Initial metadata-only
preparations exposed snapshot-path, syntax/import and exact executable-list
issues; these were repaired before scientific sampling. Their failed
preparation records remain intact. Fourteen estimator/preparer controls
passed, followed by seven preparer controls after the executable-list fix.
No scientific allocation was replaced or repeated.

For the actual saved bank, the maximum uncertain-envelope Poisson mean was
297,976.5. Its count-cap union-bound log probability is below −508,707;
`saved-envelope-cap-check.json` binds this calculation to all 20 panels.
It does not bound runtime or diagnose unobserved pose-space contributions.

## Physical result: broadening did not resolve concentration

The independent physical reduction passed all count, weight, density and
attempt-accounting checks. Its completion is distinct from statistical
convergence: the decision-driving contact estimates do not pass the gates.

| A_T diagnostic | Baseline | Broadened |
|---|---:|---:|
| Between-population relative standard error | 24.65% | 24.59% |
| Importance ESS range across four populations | 5.31–32.09 | 5.06–23.18 |
| Largest single contribution range | 12.2–31.1% | 13.8–41.6% |
| Mean population ESS / sampling CPU second | 0.133 | 0.094 |
| Mass fraction at full-chart squared radius 24–48 | 6.16% | 18.97% |

The broader guide samples more of the outer source-contact region, but this
does not improve the physical importance concentration in this comparison.
Its average reported efficiency is 29% lower. This is a descriptive result
from four populations, not a precise speedup estimate. Most estimated A_T
physical weight lies at squared full-chart radii 12–24 in both arms.

The two A_T mean masses differ by 0.285 in log units and 0.81 combined standard
errors. Thus the three-standard-error check passes, but the separate 0.2 kBT
agreement check fails. Neither arm reaches 10% population relative error,
ESS 200, or a 2% largest contribution. The complement is worse: its population
relative errors are 84.6% and 37.7%, respectively. The approximate
A_T-versus-complement free-energy interval half-widths are 2.00 and 1.68 kBT,
above the required 0.5 kBT.

No sampled pose preserved all source tokens while adding an extra exclusion
neighbor: T-any and A_T coincide in this bank, and T-outside-A has zero hits.
That is an observation, not a bound on the missing contribution. The very
large apparent A_T/complement contrast is not admitted as a physical
free-energy conclusion. It cannot establish rare-return rates, native
registry preference or assembly stability.

The completed report is
`results/context-broadened-physical-independent-audit-20261005/result/report.json`
(SHA256 `e0c144d7` prefix). The
[comparison figure](../results/context-broadened-physical-plot-20261005/result/comparison.png)
shows the four populations and separate radial partitions. The primary
estimator and results remain frozen while a separate saved-data diagnostic
examines the contribution of independent cloud noise.

## Separating cloud noise and pooling the counts

For importance-weighted cloud values w₀=W₀/q and w₁=W₁/q, let
A=(w₀+w₁)/2 and P=w₀w₁. Then

\[
A^2-P=\frac{(w_0-w_1)^2}{4},\qquad
E[P\mid x]=\left(\frac{e^{zO}}{q}\right)^2.
\]

Consequently E[A²−P | x] is the conditional variance of the two-cloud mean.
The saved-data diagnostic reports the realized sum of A²−P divided by the
sum of A². This ratio is noisy; it is not an unbiased fraction of total
sampling variance. Similarly, (ΣA)²/ΣP is not a corrected or true ESS and
can exceed the number of sampled poses.

There is also an exact improvement requiring no additional test points.
Write M=O−L and S=K₀+K₁. Conditional on S, the split K₀ is Binomial(S,1/2),
so

\[
E\left[\frac{W_0+W_1}{2}\middle|S,x\right]
=e^{zL}\left(1+\frac{z}{2\lambda}\right)^S
\equiv W_{\rm RB}.
\]

Since S is Poisson(2λM), E[W_RB | x]=e^{zO}. This is the estimator for one
cloud at combined intensity 2λ. Its relative conditional variance is
exp(z²M/(2λ))−1, whereas the original arithmetic mean has relative variance
[exp(z²M/λ)−1]/2. The reduction is

\[
\frac12\left[e^{z^2M/(2\lambda)}-1\right]^2\geq0.
\]

This variance ordering also holds for linear region-mass estimators with
the same sampled poses and densities. Realized ESS and uncertainty across
only four populations need not improve in a particular bank. The identity
is for the uncapped Poisson law, with the same implementation and finite-cap
obligations as the original estimator.

The secondary analysis reads the independently audited per-cloud counts and
weights, checks their common L, retains q_arm and all 4,096 attempted draws,
and leaves the primary analysis unchanged. It cannot repair unobserved
pose-space contributions. Binomial conditional-expectation, Poisson moment,
zero-volume and numerical-stability tests check the new arithmetic.

The seven diagnostic controls and saved-data analysis passed. The analysis
used no new points or geometry queries and took 2.14 CPU seconds. Observed
cloud-noise shares of the A_T second moment were 43.1% in the baseline arm
and 86.3% in the broadened arm.

| Secondary pooled-count diagnostic for A_T | Baseline | Broadened |
|---|---:|---:|
| Population relative standard error | 27.8% | 18.5% |
| Importance ESS range across four populations | 4.46–23.17 | 8.33–30.60 |
| Mean population ESS / sampling CPU second | 0.128 | 0.150 |
| ESS pooling all 16,384 arm attempts | 19.4 | 39.2 |

The secondary arm means differ by about 3.5%. Baseline realized ESS worsened,
which does not contradict conditional variance reduction. The broadened arm
looks better with this estimator, but neither arm meets the convergence
criteria; a reliable proposal speedup is not established. Pooled-arm ESS
and mean population ESS per CPU are different diagnostics and are not
interchangeable.

The report is `results/context-broadened-cloud-noise-20261005/report.json`,
SHA256 `974be3b69e0da52a108ef57dea40299f38d161fc6bc000e54292a190ef50c7be`.

## Next coverage problem

Pooling the cloud counts leaves competing-region ESS at approximately 1–4
per population. In the sampled bank, every non-A_T contribution is at
squared radius at least 96 under all three source charts; leading partial
contact poses have full-chart squared radii around 19,000–37,000. Increasing
the width of the same source-centered Gaussian is poorly suited to these
distinct contact environments. The partial-contact intervals [0.5,0.75)
and [0.75,1) also have no sampled hits in either arm; their contribution
remains unresolved.

The next controlled geometric change is a
[separately centered competing-contact guide](context-competing-cage.md),
trained on completed pilot evidence and frozen before fresh
evaluation. It should retain existing components and the 50% uniform
defense, use the complete mixture density, and predeclare pooled-count
weights with arithmetic-pair weights as a control. It must preserve the
physical environment and original region definitions. This is a coverage
experiment; the present conditional estimates do not decide finite-system
assembly or instability.
