# Previous SMC region coverage

The earlier SMC campaign assigns roughly half the captured weight to native
registration, but its final populations leave **the entire shoulder
`1 < q < 5` unrepresented**. This is a concrete limitation of that reference.
It does not establish that this shoulder has zero weight, nor resolve the
comparison with recent docking trajectories. An independent normalizer
calculation must include this region and the rest of the capture domain.

The source is
`/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity`.
The audit reads the archived configurations, final populations and independent
Poisson overlap counts. It changes no source datasets. Machine-readable
[results and input hashes](../runs/normalizer-reference-audit/audit.json)
retain every population rather than replacing them with a pooled trajectory.

## Physical target and estimator

At depletant radius 1.5 Å and reservoir activity 0.035 Å⁻³, a mobile rigid
tetramer moves around one fixed tetramer in an 18 Å center-capture ball.
The reference measure is Lebesgue center volume in Å³ and normalized proper
SO(3) Haar measure. For a fixed neighborhood and region `b`,

\[
 Q_b=\int_{\mathrm{capture}}H(x)1_b(x)e^{zC(x)}\,dx,
 \qquad C(x)=|E(x)\cap\cup_j E_j|.
\]

Here `H` enforces hard nonoverlap. Native registration is `q <= 1`, where
`q` is the maximum of corresponding member-center displacement divided by
2 Å and proper angular error divided by 15°, minimized over native references.
The archived executable's label `other_adsorbed` means **all `q > 1` poses**;
it imposes no adsorption requirement. Both regions use the same capture ball.
The newer roundtrip diagnostic instead uses contact cores `q <= 0.8` and
`q >= 2` (or the stricter `q >= 5`). These partitions should not be equated.

Each basin has eight independent populations of 512 particles, 1,048,576
fixed-budget unconditional proposal draws for initialization, 128 annealing
stages, and two rejuvenation sweeps per stage. The initial valid-hit fraction
is retained. Zero-hit populations remain zero estimates. Native initialization
uses a normalized mixture including a native COM-ball/Haar-cap envelope;
other initialization uses uniform capture positions and full Haar rotations.
The proposal density is constant within each target hard basin.

The archived path uses `lambda > 0`, `c = 1 + z/lambda`, and marginal activity
`z_t = lambda (c^t - 1)`. Before a stage increment, a fresh conditional count
is drawn as `K ~ Poisson((lambda + z_t) C(x))`. The stage factor is

\[
 w=\exp[\Delta t(K\log c-\log g(x))].
\]

Its Poisson expectation supplies the correct physical increment;
`g^(1-t)` supplies the known proposal flattening. Physical mutation kernels
are followed by basin and proposal-tilt corrections. Stage normalizers are
accumulated using log-mean-exp, and independent population normalizers are
averaged on the **linear Q scale** before taking a logarithm. This bounded
audit found no concrete normalization defect in those formulas or their
implementation. It does not replace a full implementation proof.

## What the archived final populations contain

Every final endpoint has equal weight within its population following
systematic resampling and target-invariant rejuvenation. To combine independent
populations, the audit weights their endpoint histograms by their linear
normalizer estimates. Correlated descendants are not treated as independent
replicates. All 16,384 stored registration errors were independently recomputed
from poses and rigid-member geometry; maximum disagreement was
`1.4210854715202004e-14`, with identical bin assignments.

| Registration region | Site 0 combined endpoint weight | Site 1 combined endpoint weight |
|---|---:|---:|
| `q <= 0.8` | 0.280768 | 0.347206 |
| `0.8 < q <= 1` | 0.213427 | 0.262590 |
| `1 < q < 2` | 0 observed | 0 observed |
| `2 <= q < 5` | 0 observed | 0 observed |
| `q >= 5` | 0.505805 | 0.390204 |

All 4,096 other endpoints at each site have `q >= 5`; their minimum `q` is
7.638 at site 0 and 6.222 at site 1. This statement concerns final endpoints,
not every transient move during annealing. It is an observed coverage gap,
not a rigorous zero-mass conclusion. Native endpoint populations divide
approximately 57%/43% between the core and its `0.8 < q <= 1` shell.

The normalizer-weighted locally unbound fractions within other are 0.000367
and 0.003349. The contact classifications here are those archived with the
independent final overlap counts; this audit does not rerun atomic contact
queries. Consequently, including unbound configurations in the old definition
does not by itself explain its estimated other weight.

![Archived SMC population coverage and normalizers](../runs/normalizer-reference-audit/previous-smc-region-audit.png)

The [SVG](../runs/normalizer-reference-audit/previous-smc-region-audit.svg) and
[PDF](../runs/normalizer-reference-audit/previous-smc-region-audit.pdf) are
available for export. Observed zero shoulder mass must not be interpreted as
an equilibrated physical probability.

| Diagnostic | Site 0 native | Site 0 other | Site 1 native | Site 1 other |
|---|---:|---:|---:|---:|
| log(mean Q) | 15.079408 | 15.102628 | 15.347374 | 14.900917 |
| Independent-population Q ESS / 8 | 7.853 | 4.310 | 7.861 | 2.979 |
| Largest population contribution | 15.85% | 41.76% | 14.47% | 55.23% |
| Relative standard error | 5.16% | 34.97% | 5.02% | 49.07% |
| Final family ESS range / 512 | 28.36–44.80 | 1.345–10.73 | 52.60–57.59 | 1.844–6.826 |

The corresponding native probabilities are 0.494195 and 0.609796. Their
reported bootstrap intervals condition on the populations that were explored;
they cannot expose missing regions. Large stage-weight ESS does not repair
small lineage diversity or certify mixing. `diagnostics.md` in the source
contains stale prose saying four populations; the actual datasets and
computed statistics contain eight per basin.

## Why unfavorable return factors are not a free-energy contradiction

The archived endpoint overlap-volume estimates are:

| Quantity | Site 0 | Site 1 |
|---|---:|---:|
| Mean C, native / Å³ | 937.52 | 882.55 |
| Mean C, other / Å³ | 608.58 | 570.47 |
| Native mean depletion advantage / kBT | 11.51 | 10.92 |
| Other minus native differential entropy / kB | 11.54 | 10.48 |

For a normalized conditional basin distribution `p_b(x)=exp(z C(x))/Q_b`
on its hard-valid support, differential entropy in the common reference
measure obeys

\[
 h_b=-\langle\log p_b\rangle_b=\log Q_b-z\langle C\rangle_b.
\]

Thus

\[
 \beta(F_N-F_O)=(h_O-h_N)-z(\langle C\rangle_N-\langle C\rangle_O).
\]

The estimates already contain an approximately 11 kBT native depletion
advantage compensated by configurational entropy. This shows how unfavorable
pointwise return factors can coexist with comparable integrated weights.
The entropy values above are derived from the same potentially incomplete
populations, not an independent thermodynamic test. Likewise, medians of
sampled Poisson acceptance factors for selected return candidates are neither
mean physical energy differences nor basin normalizer ratios. The apparent
disagreement remains unresolved pending independent region-weight estimates.

## Validation scope and reproduction

The archived `independent-audit/audit.json` reports all 64 campaign jobs passing
saved stage-weight, normalizer, ancestry, proposal-density, coordinate, and
selected atomic-geometry checks. Its limitation explicitly excludes basin
coverage. `implementation/kernel-validation.json` also records zero-activity
and positive-activity analytic sphere controls, deterministic restart/thread
checks, and zero-hit retention. The present script verifies that these
artifacts exist, records their hashes, and independently recomputes the final
region assignment; it does not rerun the original simulations or all their
validation tests.

Relevant source locations are archived
`implementation/src/bin/coordination_smc.rs` (initialization near line 711,
mutation correction near 793, path near 1165, and stage weights/resampling
near 1305–1367) and `implementation/src/single_body_depletion.rs` (exact
Poisson overlap-count thinning near 254).

From `/home/xvg/tetramer-mc`:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/audit_previous_smc_regions.py
```

The command writes `runs/normalizer-reference-audit/audit.json`, the figure
exports, and output hashes. The default source and output directories can
be overridden with `--source` and `--out`. This is a retrospective diagnostic;
the next normalizer calculation must independently cover native, near-native
shoulder, distant competing contacts, and locally unbound configurations.

## What terminal filtering of the archived SMC can estimate

There is a useful distinction between an unbiased **unnormalized region-mass
estimator** and an equilibrated endpoint histogram. For this fixed-schedule
SMC construction, let `Zhat_T` be one population's complete normalizer estimate
and let its `N` final particles have equal weight after the final resampling
and mutation. For any fixed measurable region `B` within that run's target,

\[
 \widehat Q_B=\widehat Z_T\frac1N\sum_{i=1}^N1_B(X_T^i)
 \quad\text{satisfies}\quad
 \mathbb E\widehat Q_B=\int_B\rho_T(x)\,dx.
\]

This identity holds under the sampler's exact-count and invariant-kernel
assumptions; it does not require independent final descendants or completely
mixed mutation steps. It is an identity for `Zhat * empirical_fraction`, not
for the empirical fraction alone, and it does not make ratios or logarithms
of estimated masses unbiased.

The source-specific argument is as follows. The
[fixed initialization budget](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src/bin/coordination_smc.rs:710)
uses `M` unconditional draws from normalized `g`, retains every hard/capture/
basin hit, initializes `Zhat_0=hits/M`, and resamples uniformly among hits.
Conditional expectation over that resampling turns `Zhat_0 * mean(f)` into
`sum_draws(H I_capture I_basin f)/M`. Its expectation is the unnormalized
initial measure applied to `f`. Zero-hit populations are explicitly retained
as zero estimators (source lines 1233–1242), rather than retried.

At a stage from `s` to `t`, define

\[
 \rho_t(x)=H(x)I_{\rm capture}(x)I_{\rm basin}(x)
           g(x)^{1-t}\exp[\lambda(c^t-1)C(x)],
 \qquad c=1+z/\lambda.
\]

The [archived stage loop](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src/bin/coordination_smc.rs:1304)
draws `K|x ~ Poisson(lambda*c^s*C(x))` and uses
`W=g(x)^(-(t-s))*c^((t-s)*K)`. Hence

\[
 \mathbb E[W\mid x]
 =g(x)^{-(t-s)}\exp[\lambda(c^t-c^s)C(x)]
 =\rho_t(x)/\rho_s(x).
\]

The [single-offset systematic resampler](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src/bin/coordination_smc.rs:682)
has the required conditional expected offspring counts. If the subsequent
fixed mutation kernel `M_t` preserves `rho_t`, induction gives
`E[Zhat_t mean(f)] = integral rho_s E[W|x] M_t f = integral rho_t f`.
The fresh count is generated by
[body-local Poisson thinning against the fixed exclusion union](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src/single_body_depletion.rs:254).
The physical kernels and their basin/proposal-tilt corrections still require
their separate balance and geometry validation; the induction cannot repair
a faulty mutation kernel.

The archived configurations use 128 fixed stage increments, a fixed particle
count, two fixed mutation sweeps per stage, and fixed proposal parameters.
No population-dependent schedule or online fitting was found. For the
single-reference, uniform-bank other runs, `g` is constant on the target and
the proposal-tilt filter is therefore trivial. A kernel or schedule selected
from the current population would require a new argument: invariance of
each possible frozen kernel alone does not establish the induction after
data-dependent selection.

Consequently the old runs can supply the explicit estimator
`Qhat_far = exp(logZ) * mean(q >= 5)`. Here every archived other endpoint
passes this terminal filter, so its realized value equals the saved other
normalizer. Its **defined estimand** is nevertheless now the far-region mass,
not the whole `q > 1` mass. Combine it, on the linear mass scale, with new
independent estimates for disjoint regions `q <= 1` and the entire
`1 < q < 5` interval. The latter includes both table bins `1 < q < 2` and
`2 <= q < 5`. Bound/unbound coverage must agree across all terms. Do not add
a new shoulder estimate to an estimator still labelled as the complete
old-other target: that would double count in expectation.

This is a permissible exploratory combination, **not evidence that the far
mass has converged**. Retain all prespecified independent populations and
zero estimates; do not select only populations whose endpoints happened to
be far. Rare, very large population normalizers can dominate an unbiased
estimator while remaining absent from eight observed runs. Whole independent
populations are the replication units. Thresholds and estimator choices
suggested by these same data should be frozen and checked with fresh
replication before presenting a confirmatory physical conclusion.

A configuration-only way to target the far region directly, with unchanged
archived executable and explicit reference controls, is prepared in
[the explicit-far SMC plan](/home/xvg/tetramer-mc/docs/explicit-far-smc-reference-plan.md). No production
result is implied by preparing those configurations. After fresh analytic
controls and an independent geometry comparison passed, four site-0
512-particle populations were launched; the plan records their live handle.
