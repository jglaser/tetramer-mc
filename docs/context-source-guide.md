# Source-centered conditional contact control

The [saved-candidate audit](context-candidate-bank.md) found no proposals retaining
the source's complete secondary-contact pattern. This control asks whether a
normalized proposal centered on that known pose can cover the missing pocket
before we spend on absolute depletion weights. It is a source-informed diagnostic
with 263 fixed spectators from the historical 500 μM configuration, not a
template-free assembly experiment or a thermodynamic conclusion at 106.8 μM.

## Proposal and physical measure

For the same full atomic-wall domain and fixed anchor 16, draw independently from

\[
q_*(x)=\tfrac12 U(x)+\tfrac14G_C(x)+\tfrac14g_S(x).
\]

`G_C` is the previously validated context-weighted reciprocal atlas. `g_S` is one
Gaussian pushed through the existing Cayley pose chart, centered on saved source
body 77 relative to that anchor. Its covariance is fixed before draws and never
fitted to successful proposals. No validity-conditioned redraws are permitted.

All densities use translation volume times normalized Haar measure on SO(3).
For angular coordinate `ell*c`, where `c=q_vector/q_scalar`, the chart Jacobian
is `det(L)/(ell^3*pi^2*(1+|c|^2)^2)`. Reciprocal pose inversion and composition with
the fixed anchor preserve this measure. Every candidate is scored under the
complete three-component mixture, regardless of its generating branch. The
uniform component supplies support over the full physical domain; it does not
guarantee useful finite-sample coverage of all high-weight pockets.

The source covariance is diagonal: translational entries `sigma_t^2` and angular
entries `[ell*tan(theta/2)]^2`. Thus theta describes the small-angle scale of each
angular component; it is not an exact standard deviation of the total rotation
angle. There is no covariance ridge, optimizer or feasibility retry.

## First allocation

The geometry-only pilot fixes four widths, four independent streams per width,
and 512 unconditional draws per population: **8,192 new proposals total**.

| Translation scale (Å) | Angular scale (degrees) |
|---:|---:|
| 0.1 | 0.5 |
| 0.2 | 1 |
| 0.4 | 2 |
| 0.8 | 4 |

The sixteen jobs use at most four workers, one thread each. Each has ceilings of
600 CPU seconds, 1,200 wall seconds and 8 GiB. Every attempted draw, hard-invalid
zero and failure prefix is retained. Component, label, latent, uniform and both
prospective cloud roles have separate SHA-derived seeds bound to the entire
configuration, population and ordinal. A later change of mode/configuration
therefore creates fresh streams.

The complete T16/A/B/Other/unbound regions remain frozen as in the saved-bank
audit. The independent Python audit reconstructs all 8,192 source densities,
complete mixture sums and recorded latent decodes. Its fixed 512-pose panel also
reconstructs the full context-atlas density and physical geometry. Neither the
geometry pilot nor its audit draws depletant clouds.

## Absolute weights and failure semantics

The runner also implements an optional cloud mode, which is **not allocated by
this geometry pilot**. For a valid pose, the overlap envelope and two independent
Poisson clouds estimate the absolute many-body weight:

\[
W=e^{zL}(1+z/\lambda)^K,
\qquad K\sim\mathrm{Poisson}(\lambda[O_C-L]).
\]

Here `O_C` is overlap with the union of all fixed exclusion regions, rather than
a pairwise sum. The regional contribution is `mean(W0,W1)/q*`, divided by every
attempted draw; hard-invalid candidates contribute zero. There is no additional
Jacobian and no reuse of acceptance-gate ratio estimators.

The new `sample_with_envelope_bounded` API journals a sampled Poisson count before
budget admission or point generation. Limits admit the whole count or fail.
Callbacks allow periodic CPU/wall checks. Progress is retained on failure and
cannot expose a usable partial weight. The caller charges planned/processed
counts once and stops without clipping, retries or replacement draws. The
existing public sampler delegates to the same arithmetic with a no-op observer.

The Poisson expectation identity concerns the intended uncapped cloud law.
Conditioning on random budget success can change its distribution. Consequently,
an incomplete population must not be turned into an estimate by keeping only
completed clouds or restarting until a budget-successful run occurs. A resource
failure remains an unresolved calculation with its recorded prefix.

The caller remains responsible for binding the envelope to the same immutable
environment; the envelope itself checks its pose. Geometry predicates remain
guarded floating-point implementations. These obligations are not removed by
the algebraic Poisson identity.

## Validation and next decision

The isolated validation includes twelve overlap tests (four existing analytic
and geometric references plus eight bounded-sampler tests) and nine focused
runner tests. A copy of the previous point loop supplies an independent RNG
continuation oracle for the shared-core refactor. Eight deterministic Python
tests cover source density/Jacobian, frame and reciprocal transforms, complete
mixture weights, role identities, rejected zeros and journal completeness.

Run the relevant Rust checks without compiling unrelated binary targets:

```sh
cargo test --release --offline --locked --example absolute_overlap_reference -- --test-threads=1
cargo test --release --offline --locked --example context_source_guide source_guide_tests:: -- --test-threads=1
```

The first isolated integration-target build reached its compiler CPU cap before
any test ran. It is preserved in `results/context-source-guide-build-20261005`.
The example wrapper runs the same twelve test functions while avoiding Cargo's
automatic compilation of unrelated executables. The corrected build passed all
21 Rust tests in `results/context-source-guide-build-20261005-attempt02-wrapper`,
using 95.580 child CPU seconds and 81.185 wall seconds. The protected production
binary and earlier snapshots remained unchanged.

## Completed geometry pilot

All sixteen populations completed: 8,192 unconditional proposals, 239.75 CPU
seconds, 74.95 wall seconds and no depletant clouds. The independent Python
audit passed 59,968 scalar comparisons, all 8,192 recorded pose decodes and its
predeclared 512-pose atlas-density/geometry panel. Maximum pose reconstruction
errors were 8.53e-14 Å in position and 1.12e-15 in quaternion components. The
audit used 10.12 CPU seconds. Every producer and audit child drained.

| Source width (Å / degrees) | Complete source-pattern hits in each population | Hits / all draws | Source-branch hard-valid / source draws |
|---|---|---:|---:|
| 0.1 / 0.5 | 3, 3, 1, 1 | 8 / 2,048 | 8 / 510 |
| 0.2 / 1 | 1, 0, 0, 0 | 1 / 2,048 | 1 / 505 |
| 0.4 / 2 | 0, 0, 0, 0 | 0 / 2,048 | 0 / 507 |
| 0.8 / 4 | 0, 0, 0, 0 | 0 / 2,048 | 0 / 511 |

Every hard-valid source-branch draw retained the full T16 pattern in region A.
Thus this pilot's source-guide failure is primarily hard feasibility, rather
than valid source draws losing the recorded pattern. Context-atlas and uniform
draws supplied other valid configurations, but no complete T16 hits. The
narrowest guide recovers a pocket absent from the earlier 36,864 saved global
proposals; the two allocations have different proposals and sizes and do not
establish a mixing speedup.

The four narrowest-guide hard-only region-mass estimates range from 2.08e-11
to 1.41e-10 Å³ times normalized Haar measure. The next width has a single
population estimate of 2.48e-9 and three zero-hit populations. These estimates
are not converged. Nine total hits cannot satisfy the required regional
importance ESS of 200. Zero-hit estimates are not upper bounds on region mass.

![Source-guide coverage and prospective cloud cost](../results/context-source-guide-independent-audit-20261005/figures/source-guide-coverage.png)

The prospective two-cloud allocation would require about 49.86 million raw
points in expectation. It has **not** been launched: the pose coverage already
fails before that expense. The next diagnostic reuses source-start local
trajectories to measure translation–rotation covariance, preserving rejected
residence. Any resulting correlated guide must be frozen and evaluated with
fresh draws and the complete mixture density. Those trajectories need not be
equilibrated to define a valid proposal, but their fitted covariance would not
establish equilibrium fluctuations or template-free discovery.

Evidence:

- Geometry controller: `results/context-source-guide-geometry-20261005`,
  receipt SHA256 `cb91ff5444b5a7c64b0ef6cbd4f3d2612cc846d45b0c90a5f0fbe9b6e771245a`.
- Independent audit: `results/context-source-guide-independent-audit-20261005/result/report.json`,
  SHA256 `f35bb4d6ad768b7ac8d224ce201c8dabb83a9f130283b9a4a9e47d4a00cbec71`.
- Plot and its input/source hashes:
  `results/context-source-guide-independent-audit-20261005/figures/provenance.json`.

Conditional hard-only coverage does not establish depletion free energies,
finite-system assembly or instability. The original-condition regional physical
weight sensitivity campaign continues separately; the finite-system assembly
bank remains behind its physical convergence and coverage gates.

Implementation: [runner](../examples/context_source_guide.rs),
[geometry bridge](../examples/support/context_source_geometry.rs),
[bounded cloud kernel](../src/overlap_weight.rs),
[independent audit](../tools/audit_context_source_guide.py).

The subsequent [full-versus-diagonal covariance control](context-covariance-guide.md)
adds an optional frozen correlated chart while preserving this pilot's legacy
input format and proposal law.
