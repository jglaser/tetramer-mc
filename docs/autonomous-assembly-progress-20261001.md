# Autonomous assembly investigation: 2026-10-01

**Verdict: unresolved.** This continuation improves proposal coverage and produces
one accepted change from a misregistered to a native inter-tetramer contact.
It does not establish faster equilibrium mixing, template-free finite-system
assembly, or instability of the model. Existing simulations were left running;
the live production executable was not replaced.

## Keep the two ensembles separate

| Work | Physical parameters | What is measured |
|---|---|---|
| Proposal controls | 264 mobile tetramers, 500 μM, depletant radius 1.4 Å, activity 0.0275 Å⁻³; archived sweep 60,900 | Conditional transition opportunities in independent reset phases |
| Contact-weight pilot | Repaired rigid tetramer, radius 1.5 Å, activity 0.035 Å⁻³; original R4 and fixed two-neighbor scaffold | Unconditional regional integration weights, not an assembly chemical potential |
| Eventual decision ensemble | N=12/24, approximately 106.8 μM, radius 1.5 Å, activity 0.035 Å⁻³ | Finite-system assembly and stability, with size, preparation and boundary controls |

## Proposal changes and controls

The [constant-rate singleton channel](singleton-contact-control.md) permits a
tetramer to change its own contacts without carrying a misregistered partner.
The new retained, ordered contact-weighted anchor pool chooses actual neighbors
more reliably: both contacting neighbors occur in approximately 80–82% of
eligible learned trials, compared with about 19% using the old nearest-secondary
pool. Its complete reverse/forward selection probability is included once.
The same selected pool and spectator context are retained across the move.

The optional [finite-cap independent redraw](hard-conditioned-independent-redraw.md)
reuses that context and tries at most eight independent complete-mixture poses,
stopping at the first hard-valid candidate. It still applies the full guide,
pool and many-body depletion corrections. Failure at the physical acceptance
step is a rejection, not permission to try again. The unknown hard-valid
normalizer cancels between endpoints; every exhausted cap is a self-loop.
The default leaves existing correlated transport and random streams unchanged.

The [frozen comparison](hard-conditioned-redraw-control.md) completed all
2,560 reset phases across cap 1, cap 8 and local controls, reusing the completed
correlated-transport populations. Every raw trial and rejected endpoint is
retained. The local control uses the original 0.2 Å / 1° proposal scales.

| Result | Interpretation |
|---|---|
| Hard-valid independent events increased from about 6% at cap 1 to 35–36% at cap 8 | Reusing the context buys more valid candidates, not improved success per raw draw |
| Cap-8 fused-arm candidate throughput was 5.46 per CPU second, versus 1.43 at cap 1 | Catalogue construction is amortized; this is not independent contact ESS |
| Most valid candidates still fail the depletion gate | Steric conditioning alone does not solve destination selection |
| Most accepted patch changes were small | Accepted counts and coarse patch crossings do not establish reorganization efficiency |
| One accepted native registration change occurred in the cap-8 fused arm | A large contact rearrangement is accessible; actual successful target was an ordinary chart with **zero fused components** |

In that last event, tetramer 213 remained in contact with partner 70 while
moving 61.76 Å and rotating 94.47°. All 14 old surface-patch tokens were replaced;
the instantaneous native observer gained motif 7. The first seven raw proposals
clashed, and the eighth succeeded. This was an isolated dimer, not attachment to
the seeded aggregate. An outcome in the fused *arm* must not be attributed to
fusion when its conditional catalogue contained no fused components. Nor can one
event establish a statistically significant advantage of eight trials.

The successful ordinary branch 5 is the reciprocal of merged component 2,
refined from original discovery slot 1. Its lineage reaches an FFT search
rotation/translation followed by hard repair and overlap scoring, with no
native inter-tetramer template. The proposal atlas is geometry/depletion-only;
the benchmark's seed-informed starting state and passive native observer are
separate inputs. Native intratetramer geometry remains supplied throughout.
The [branch-provenance audit](native-registration-event-provenance.md) verifies
608 frozen input files and records the remaining historical archive limitation.

![Accepted contact rearrangement in the unchanged partner's frame](../runs/singleton-independent-control-20261001/native-registration-event/native-registration-event.png)

The complete [comparison and population breakdown](../runs/singleton-independent-control-20261001/comparison/report.md)
includes guide/pool/Poisson corrections, CPU times, patch changes, partner
exchanges, native gains and losses, replay checks and raw-trial accounting.
No reset-phase statistic is presented as a trajectory ESS or equilibrium weight.

## Independent physical evidence

A frozen expansion from 84 to 92 geometric integration components completed
four populations of 16,384 draws per arm: **131,072 attempted draws**. It preserves
50% uniform defensive support, the original R4, full native classifier,
unconditional denominators, invalid zeros and two independent Poisson clouds
per valid pose. Auxiliary intensity was 128 times the depletant activity.
Historical samples only designed the proposal; they were not pooled into these
fresh estimates.

The targeted native-complement and competing tails improved importance ESS per
physical CPU by 3.65× and 3.75×. Aggregate conditional native-minus-competing
free energies agree: −18.7926 ± 0.1744 and −18.7881 ± 0.1323 kBT (population-based
95% half-widths). These observed precision intervals are not missing-mode bounds
and do not include a finite-system assembly entropy.

The convergence gate remains closed. Native-complement ESS is only 126/166,
largest contributions exceed 2%, and 14 of 34 material stratum comparisons fail.
A newly better-observed competing family contacts both scaffold neighbors close
to the hard surface, without native registry. Its apparent mass differs sharply
between the two proposals and still has only 7.16 importance ESS. Some dominant
rows originated in an old component: this is incomplete coverage of a narrow
cooperative family, not evidence that the added components alone discovered a
new equilibrium phase.

See the [complete physical report](../runs/contact-tail-pilot-review-v2-20261001/report.md)
and [historical evidence reconciliation](autonomous-physical-evidence-status.md).
All eight physical jobs and independent audits completed; no allocation was
extended after seeing its results. Full-vessel and finite-system production
remain behind their original gates.

## Validation and next decision

The new kernels passed finite-state balance, ordered-pool normalization,
independent density/coordinate reconstruction, disabled-mode compatibility,
all-failure self-loops, deterministic restart and analytic two-/four-sphere
depletion controls from distinct preparations. The combined reference exercised
fused targets and nonzero pool corrections. The [validation receipts](../results/hard-conditioned-validation-20261001/implementation-and-balance.md)
retain the initial rejected test fixture as well as corrected checks. Twenty-one
analysis/controller tests also passed. The protein audit reconstructed every
attempted integration draw, with maximum log-density discrepancy 1.42×10⁻¹⁴.

The [Lean bridge](../formal/ReversibleSampling/CappedIndependent.lean) proves the
finite-cap renewal identity, empty-success case and shared accepted-flow factor
cancellation. The full build and 70 recorded axiom audits pass with the existing
pinned dependencies. Geometry, thinning, Jacobians, random independence and
floating-point execution remain implementation obligations; Lean does not
certify those by proving the algebra.

The immediate priorities are better coverage of the narrow two-neighbor
competing family and better depletion-compatible destination selection. A
larger assembly run using the same poor proposals would not resolve either.
The successful native rearrangement is retained as an exact reproducible
example, while broader destination-quality controls must include competing and
unbound environments. Subsequent evolving-trajectory comparisons must measure
initialization agreement, completed environment exchanges and contact ESS per
CPU, including repeated rejected states. Thermodynamically negligible regions
need not have frequent returns. A negative conclusion still requires converged
finite-system physical evidence.
