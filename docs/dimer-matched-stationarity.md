# Matched proposal versus bath stationarity control

This is a new, separately frozen reference allocation following the retained
primary rejection of defensive independent/world-Poisson sampling in the
[earlier one-step control](dimer-one-step-stationarity.md). It neither enlarges
that allocation nor replaces its rejection. Its source/proposal pairs are fresh;
historical data are used only to plan precision, never pooled into the new tests.

Eight independent populations each contain 65,536 exact-IID equilibrium sources
and independent defensive proposals: **524,288 pairs**. Every pair has the same
source, candidate, complete proposal correction and acceptance uniform for three
separate endpoint decisions: analytic depletion, world-Poisson depletion, and
singleton-path depletion. The two bath streams are independent. No accepted
endpoint is passed to another arm or to the next pair, and there is no angular
refresh. All invalid proposals and rejected outcomes stay in the denominators.
The three decisions per pair are correlated; different pairs are independent in
the mathematical random-source model.

The physical target, source rejection envelope, four identity-covariance Cayley
charts, uniform probability 0.5, relative cube half-width 2.5, auxiliary intensity
24 and geometry budget are unchanged. The source is two core-radius-0.2 spheres
with exclusion radius 1, activity 1.5, root ball radius 2 and relative separation
0.4–2.5. The frame is virtual and the child has no separate wall. The radius law
is exactly proportional to \(d^2\exp[zV(d)]\) by rejection from uniform shell
volume; positions and normalized-normal Haar quaternions have the same independent
factorization as before. Exactness describes the mathematics, not a floating-point
or pseudorandom-generator certificate.

The RNG domain remains `dimer-one-step-stationarity-v1` for transparent reuse of
the source audit. New, distinct master seeds 610027000 through 610027007 separate
this allocation from prior controls. Each pair has SHA256-derived roles `source`,
`proposal`, `world_gate`, `path_gate` and `mh`. The complete config, sources,
dependency closure, binary and toolchain identification are frozen before draws.

## Five primary tests

Let \(D_a=f(Y_a)-f(X)\), with \(f=\mathbf1[d<1]\). The family contains
\(D_{analytic},D_{world},D_{path},D_{world}-D_{analytic}\), and
\(D_{path}-D_{analytic}\). Every variable belongs to \(\{-1,0,1\}\): all arms use
the same proposed endpoint and otherwise retain the same source. Under the
respective stationarity null, positive and negative changes have equal
probability. Each exact two-sided binomial sign test conditions on the number of
nonzero changes. The five tests use Bonferroni family alpha 0.05; dependence
between arms does not invalidate that correction. Zeros remain in reported
means and variances. Estimated simultaneous normal intervals and conservative
Hoeffding intervals are also reported without replacing the primary rule.

The three direct tests detect failure to preserve equilibrium. Matched bath-minus-
analytic contrasts separate bath-specific discrepancies from a proposal/source
sample shared by all arms. A contrast consistent with zero is not proof that
both kernels are correct. All 16 secondary pose observables, their paired
changes and matched contrasts are retained. The separate 19-check source family
uses the same moments, radial histogram and geometric rejection-count reference
as the preceding control. Source tests are asymptotic checks, not guarantees.

With 524,288 independent pairs, the observed planning variances imply standard
errors about 0.000318 for analytic contact drift, 0.000292 for world contact
drift, and 0.000198 for world-minus-analytic contact drift. At the Bonferroni
threshold, approximate 80% power corresponds to effects 0.00109, 0.00100 and
0.00068, respectively. These are prelaunch planning estimates, not a revised
threshold or a guarantee of detecting smaller or higher-order errors. There is
no extension or repeated seed selection based on the result.

## Implementation and validation scope

The [standalone Rust example](../examples/dimer_matched_stationarity.rs) stores
one record per pair: a common source/proposal/acceptance-uniform record and three
ordered endpoint outcomes. Each outcome retains its complete gate record, MH
ratio, decision and state. A failure retains partial outcomes and stops; nothing
is silently retried. The output directory must be new. The production executable
and assembly kernels are untouched.

The [independent analyzer](../tools/analyze_dimer_matched_stationarity.py) imports
the two earlier independent analyzer modules; all three are part of the archived
closure. It reconstructs each defensive pose from recorded branch coins, uniform
variates or Gaussian label/latent coordinates, including the child's position
relative to the new root. It independently checks full proposal densities,
source generation, physical predicates, all three gate/MH decisions, copied path
intermediates, and retained states. It does not replay rejected source uniforms
or all depletant points. An unexplained numerical-null generation blocks the
audited conclusion; no missing attempt is filtered out.

Three Rust contract tests verify allocation/stream separation, source replay,
and identical zero-activity decisions under the shared uniform. Five Python
synthetic tests cover both generation branches, matching and branch/pose/density/
gate tampering, complete arm coverage, exact signed tests and JSON serialization.

Any result is a toy-kernel diagnosis, separate from mixing efficiency and protein
assembly. The earlier primary rejection, earlier trajectory flags and all
failed analysis receipts remain part of the evidence regardless of this result.

## Completed matched allocation

The frozen run at `results/dimer-matched-stationarity-20261002` completed exactly
524,288 independent pairs and 1,572,864 endpoint decisions in 58.598 process CPU
seconds. All attempted pairs are retained; there were no numerical proposal
nulls or runtime errors. The independent all-row audit reconstructed complete
proposal corrections within \(1.25\times10^{-14}\), generated translation
coordinates within \(1.07\times10^{-14}\), and quaternion coordinates up to sign
within \(1.84\times10^{-15}\).

| Primary contact change | Mean | Standard error | Exact p | Estimated simultaneous normal interval |
|---|---:|---:|---:|---:|
| Analytic | +0.0000381 | 0.0003166 | 0.9089 | [−0.000777, +0.000854] |
| World Poisson | +0.0001163 | 0.0002898 | 0.6930 | [−0.000630, +0.000863] |
| Singleton path | +0.0001717 | 0.0003119 | 0.5862 | [−0.000632, +0.000975] |
| World minus analytic | +0.0000782 | 0.0002029 | 0.7069 | [−0.000444, +0.000601] |
| Path minus analytic | +0.0001335 | 0.0001354 | 0.3311 | [−0.000215, +0.000482] |

None of the five predeclared primary tests rejects; all Bonferroni-adjusted
p-values are 1. There are no flags in the 19-check source family, and none of
the 80 fully reported secondary paired mean changes or contrasts exceeds 3 SE.
The artifact also retains the more conservative Hoeffding intervals, complete
per-population statistics and all source checks.

This new sample **does not reproduce the earlier positive primary drift** and
does not isolate a world- or path-bath discrepancy relative to analytic
acceptance. The earlier primary rejection is preserved explicitly in the plan
and completed receipt; historical samples are not pooled into the new tests.
The new executable includes the shared map/scorer coefficient-closure fix for
ill-conditioned atlases. The identity-covariance atlas used here has identical
factors in those paths, so this comparison is not evidence that the coefficient
fix repaired the earlier toy flag.
The result constrains errors at the declared precision, while leaving smaller
errors, unmeasured observables, floating-point obligations and evolving-chain
mixing uncertainty open. It is not evidence for protein assembly.

`completed-review.json` authenticates the 494 archived source/dependency files,
config and executable, eight unchanged physical streams, manifest and analysis.
It also verifies that every binding in the earlier completed receipt, including
its serialization failure and recovered primary rejection, remains unchanged.
