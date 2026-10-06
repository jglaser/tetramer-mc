# Separating depletant noise from pose coverage

The [competing-cage comparison](context-competing-cage.md) improved coverage
of competing contacts but left physical weights concentrated in a few
draws. More depletant points and a better pose proposal address different
sources of that concentration. The existing saved counts allow an exact
second-moment diagnostic before allocating either kind of new work.

## Fixed-pose identity

At a fixed pose, let O be the true many-body overlap volume, L its
certified lower bound, and V=O−L. Both clouds have intensity λ and are
independent conditional on the pose. Their pooled count is
S=K₀+K₁ ∼ Poisson(rV), where r=2λ. Write c=z/r and
b=exp(zL)/q(x), using the **complete** frozen mixture density q.
The primary importance contribution and a same-count mean-square
estimator are

\[
Y=b(1+c)^S,\qquad P=b^2(1+2c)^S.
\]

The Poisson generating function gives

\[
\begin{aligned}
\mathbb E[Y\mid x]&=e^{zO}/q(x),\\
\mathbb E[P\mid x]&=(e^{zO}/q(x))^2,\\
\mathbb E[Y^2\mid x]&=(e^{zO}/q(x))^2 e^{z^2V/r}.
\end{aligned}
\]

Consequently D=Y²−P is a nonnegative unbiased estimator of
Var(Y|x). Its nonnegativity holds for every saved count because
(1+c)²≥1+2c. Zero activity, zero count, zero uncertain volume and
hard-invalid contributions are included. A region indicator multiplies
Y and P; it must be determined by the pose, independently of the clouds.
The same holds for each fixed radial subdivision.

This construction does not insert an estimated overlap volume into an
exponential. It estimates the required moments directly under the
Poisson law. It also does not change any previously reported mass.

## Fixed allocations matter

Each population has N=4,096 unconditional attempts, divided into fixed
proposal strata j with n_j draws. Thus

\[
\widehat I=\frac1N\sum_j\sum_iY_{ji},\qquad
\widehat V_{\rm aux}=\frac1{N^2}\sum_j\sum_iD_{ji}.
\]

The second expression is unbiased for the auxiliary part of the
normalizer estimator's variance. An unbiased estimate of the total
variance is Σ_j n_j s²_j/N², where s²_j is the ordinary sample variance
of **all** n_j contributions in stratum j, including zeros. Centering
all draws about one mixture-wide mean would give the wrong variance
for these fixed allocations.

Subtracting the estimated auxiliary variance from the estimated total
variance estimates the pose contribution. That difference can be
negative in a finite record even though its expectation is nonnegative.
It must be retained and flagged, not clipped to a favorable result.
Population disagreement and dominant contributions still need to be
reported.

The ratio ΣD/ΣY² describes an estimated share of **uncentered second
moments**; it is not the share of total estimator variance. Similarly,
(ΣY)²/ΣP is only a noisy estimate of an effective sample count with the
cloud noise removed. It is neither unbiased nor a rigorous bound and
may exceed the attempted sample count. None of these ratios detects an
unvisited pose basin.

## Hypothetical intensity changes

For a positive intensity multiplier m, the same stored S yields

\[
H_m=b^2(1+2c+c^2/m)^S.
\]

Its expectation is the importance second moment that would result from
fresh pooled clouds with intensity mr at the same pose. H₁=Y² and
H_∞=P. This provides a diagnostic for m=2 and 4 without generating
new clouds or using a biased exponential of a volume estimate.
It is not an observed speedup. A runtime projection that multiplies
the scoring cost by m assumes linear point-processing cost and leaves
pose-generation cost fixed; both assumptions must be labeled.

Increasing intensity cannot discover a missing contact basin. Even
perfect evaluation of every saved pose leaves proposal coverage,
hard-invalid draws and fluctuations of the physical weight over poses.
Conversely, large cloud variance should not be blamed entirely on the
Gaussian map. The purpose of this diagnostic is to choose the next
experiment, not to relax the physical convergence requirements.

## Formal and implementation boundary

The Lean bridge in `formal/ReversibleSampling/PoissonNoise.lean` uses the
existing Poisson generating-function proof to express both moments and
the centered conditional variance as convergent real series. It allows
a count-independent prefactor, instantiated here as exp(zL)/q(x).
`formal/PoissonNoiseAudit.lean` audits its theorem axioms separately.

Exact independent thinning, correct geometry and lower bounds, complete
proposal densities and Jacobians, and floating-point evaluation remain
implementation obligations. The bridge proves a conditional statistical
identity; it does not certify a finite protein sample, convergence of
the contact integral, or finite-system assembly.

Seven Lean theorems passed with pinned Lean 4.24.0 and the existing
mathlib dependency. Every audited axiom set is exactly
`[propext, Classical.choice, Quot.sound]`; there are no `sorry` or custom
axioms. The successful compile and audit consumed 11.82 CPU seconds
and 12.02 wall seconds with one thread and peak process-group RSS
3.183 GiB. The receipt is
`results/poisson-noise-lean-check-20261006-v3/receipt.json`, SHA-256
`bbd92f86f3a3ca0d2f88485494fc71477d371c55e5c37b43acba8213eea43764`.

Two earlier check prefixes are retained: an address-space-cap failure
during cached-import loading, then proof-draft errors. The successful
check used Lean's 4 GiB memory setting plus a 4 GiB process-group RSS
watchdog, with the same 120 CPU / 240 wall-second caps. No dependency
was upgraded and no physical calculation was rerun.

## Completed saved-count diagnostic

All 32,768 attempted poses and 15,968 cloud records were retained. The
diagnostic independently reconstructs the primary weights, checks every
region/radial second-moment partition and preserves the original mass
estimates. Fifteen synthetic controls pass, including analytic Poisson
moments, zero cases, fixed-stratum centering, signed variance differences,
intensity forecasts and unbounded ratio confidence sets. No new poses,
geometry queries or clouds were generated.

The following ranges span the four independent populations of the
**multicage** arm. They describe the observed record; they are not
confidence bounds or guaranteed gains.

| Region | Auxiliary share of uncentered second moment | Gain with noise removed, plug-in | Noise-free ESS, plug-in | 2× intensity ESS/CPU forecast |
|---|---:|---:|---:|---:|
| Source-complete A | 58.0–58.3% | 2.38–2.40× | 3.33–42.24 | 1.020–1.039× |
| Other contacts combined | 37.5–38.1% | 1.60–1.62× | 6.68–28.47 | 0.838–0.850× |
| Neighbor environment B | 26.4–27.8% | 1.36–1.38× | 23.19–38.79 | 0.769–0.786× |

The baseline source-complete A forecasts 0.946–0.966 times its current
efficiency at twice the cloud intensity. At four times the intensity,
all these contact-region forecasts are below 0.79 times current
efficiency. These projections use the stated linear scoring-cost
approximation. They have not been measured in new runs.

**Every contact-region confidence set for the noise-free second-moment
gain remains upper-unbounded.** Four populations with dominant weights
do not support a precise efficiency prediction. Nevertheless, the
observed budget does not justify substantially increasing cloud work
as the next step: pose concentration and the empty intermediate contact
regions remain even in the noise-free diagnostic. Keep the present
cloud intensity while investigating proposal coverage. This is a
choice of the next experiment, not an assertion that more clouds can
never help.

The first saved-output report is preserved. Its moment estimates were
correct, but a generic Fieller calculation produced numerical artifacts
for the structurally exact intensity-1 self-ratio. The corrected
version explicitly returns that identity and adds a synthetic control;
it does not infer exact identities merely from equal observed arrays.
No physical weights or primary conclusions changed.

The corrected reduction used 6.79 CPU seconds. Report:
`results/context-multicage-noise-budget-20261005-v2/report.json`, SHA-256
`eef18d9a2b066b3006db72c6591adb22f5556314e9a1f0604f5d6d40b522bc77`.
Its two owned process groups are drained; the receipt SHA-256 is
`da126297cf1998fef66d61ebaeff760b5faea81ecb188f8b8428042a8148dde9`.
