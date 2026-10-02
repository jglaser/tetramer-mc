# Autonomous assembly investigation: 2026-10-01

**Verdict: unresolved.** This continuation improves proposal coverage and produces
one accepted change from a misregistered to a native inter-tetramer contact.
It does not establish faster equilibrium mixing, template-free finite-system
assembly, or instability of the model. Existing simulations were left running;
the live production executable was not replaced.

The fixed **786,432-draw contact campaign is now complete**. In the
[larger stage](hard-free-line-population-size-results.md), every aggregate
proposal and population-size mass comparison passes. The guided conditional
native/competing contrast is −18.7204 ± 0.0494 kBT (population-based 95%
half-width), and importance ESS/CPU improves 1.68–1.88×. Two material strata
still disagree between proposals and eleven fail the size comparison. Together
with the earlier sensitivity failures, these leave the convergence gate closed.
A [fresh four-population SMC bridge control](hard-free-protein-smc-control.md)
has completed physical sampling. Two independent audits passed and two stopped
on narrow-interval density discrepancies; the queued contact analysis correctly
did not start. A [saved-data diagnosis](smc-density-conditioning-diagnosis.md)
reproduces both failures and isolates geometric roundoff amplified by small
conditional Normal masses. High-precision endpoint and mass checks support that
diagnosis. The optional supplemental audit passed 25 focused tests and all 128
saved diagnostic poses; its two full saved-data audits are now running, with
the previous successful audits reused. The original failed receipts remain
intact. No SMC region-weight result is available yet, and all earlier
convergence failures remain.

The [larger-stage variance decomposition](hard-free-line-population-size-noise.md)
now reuses the saved moments for all failed strata. The worst baseline competing
bin has importance ESS 3.8 and a single draw carrying 49.5% of its mass, while
only 1.6% of its observed row variance is attributed to cloud noise. Native bin
55 instead has substantially larger row-moment than four-population SE. These
diagnostics favor pose coverage and uncertainty calibration; the running SMC
allocation and all convergence failures remain unchanged.

## Latest continuation: feasibility-only conditioning

The [fresh line-guide comparison](hard-free-line-fresh-results.md) completed
2,048 new protein proposals and independently reconstructed every output.
Keeping the Gaussian measure and conditioning only on hard-free translation
intervals increased native proposals from 66 to 153 per 1,024 attempts, and
valid contacts from 211 to 329. Proposal-loop CPU increased 9.75-fold, so this
does **not** yet establish a useful physical-sampling speedup. No Poisson clouds
were generated in that geometry comparison.

The [production integration](hard-free-line-production-integration.md) passed
analytic sphere/depletion references, exact disabled-guide controls, complete
independent reconstruction of 16,704 reference/control rows and attempted-draw
failure checks. One test-helper mutation was corrected and recorded separately;
the validated physical outputs were reused without rerunning them.

The [fixed physical pilot](hard-free-line-physical-pilot.md) is complete:
131,072 physical draws, all eight independent audits and one classification
pass, at the original 1.5 Å / 0.035 Å⁻³ conditions. The conditioned guide improves
importance ESS per sampler CPU by **1.63× for native mass, 1.11× for competing
contacts and 2.05× for the native remainder outside old R5**. All four conditioned
regional aggregate quality checks pass; all four between-proposal region-mass
comparisons pass. Eight material strata still fail the absolute 0.2 log-mass
criterion (all are within three combined population SE). The gate therefore
remains closed. This is a physical integration improvement, not a demonstrated
assembly-mixing speedup.

Conditional native-minus-competing free energies are −18.6933 ± 0.1721 kBT for
the baseline and −18.6950 ± 0.1237 kBT for conditioning (population-based 95%
half-widths). These are fixed-scaffold R4 weights; they omit the full-vessel
remainder and finite-system association entropy. The
[complete report](../results/hard-free-line-physical-review-20261001/report.md)
preserves all failed strata, individual population masses and original gates.

![Completed regional physical comparison](../results/hard-free-line-physical-review-20261001/hard-free-line-physical.png)

Live status:
`/vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001/status.json`.
Protocol SHA-256:
`1b788b08df239319ac69126627e891893b214c35204e337c95f4a89ede23c050`.
The sections below preserve the earlier stages of this investigation.

The [matching-SMC bridge](hard-free-line-smc-bridge.md) completed against all
twenty historical populations in five controls. The broad unrestricted SMC
estimate has only 0.216× the conditioned native-remainder mass, a difference of
5.34 combined linear SE, while its old-R5 native mass agrees. Narrow unrestricted
SMC agrees in total native mass but remains sensitive in that remainder.
Restricted narrow and larger-population SMC agree on competing-contact mass;
the broad restricted control remains discrepant. Every observed hard-only
comparison passes. This localizes the mismatch to weighted coverage rather
than establishing a measure/geometry mismatch. All historical failures remain;
no trajectories or classifiers were replayed. The
[independent review](hard-free-line-population-size-review.md) supports the
next separately frozen population-size comparison with the same guide and binary.

The [saved-data bridge diagnostic](smc-bridge-bottleneck.md) identifies a
candidate mechanism: under the old SMC bridge, the conditioned estimate of
the remaining-native fraction falls from 8.85% to roughly 0.619%, then rises
to 47.05%. Both IID arms show the dip but fail concentration checks near its
minimum; its depth is not resolved. Using each frozen IID proposal as the
initial bridge law removes the observed dip: the conditioned alternative
stays between 29.09% and 51.17%, and every sampled stage passes aggregate
checks for the four nonempty decision regions. Endpoint weights are identical
by construction. This is an exploratory SMC design diagnostic, not independent
physical confirmation or a measured speedup. It adds no poses, clouds or
classifications and does not open the full-vessel or assembly gates.

An [optional SMC initial-guide mode](hard-free-smc-initial-guide.md) now
implements that path with complete physical q/J, deterministic density
caching and fresh auxiliary clouds. Five Rust tests, six independent-auditor
tests and three reference-preparation tests pass. The eleven sphere/Haar
references have now completed: all six analytic mass checks, ten independent
guided-law audits, the empty-target test and the same-stream uniform control
pass. The largest analytic discrepancy is 2.89 population SE, within the frozen
software tolerance. Actual protein SMC improvement remains unmeasured. All new
build work uses the isolated validation target, leaving the live assembly
executable unchanged.

The accompanying [pair-flow theorem](../formal/ReversibleSampling/PairFlowPoisson.lean)
now covers singular translation-only and rotation-only proposals without
inventing a six-dimensional proposal density. Four new Lean results build and
pass the focused axiom audit with unchanged dependencies. Concrete Gaussian/Haar
symmetry, geometry, thinning and floating-point execution remain obligations.
The exact-reference executor is frozen separately and enforces one physical
child at a time, independent auditing, capacity checks and no retries.

The new [protein SMC control](hard-free-protein-smc-control.md) launched after
the references and fixed IID controls completed. It retains M=262,144 initial
draws, N=2,048, 128 stages and four mutations per stage. Independent full-guide
auditing could cost tens to over a hundred CPU-hours; that estimate is a
planning scenario, not a measured SMC result. The normalized initial/intermediate
density changes; the broad local steps, endpoint and allocation remain fixed.
The frozen controller passes seven tests and enforces two physical/four audit
workers under the global limits. Controller PID 667281 runs at
`/vast/xvg/tetramer-mc-runs/hard-free-protein-smc-control-20261002`.

The [larger stage](hard-free-line-population-size-stage.md) completed physical
sampling, all eight independent audits and once-only classification at
`/vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001`: four fresh
65,536-draw populations per arm, 524,288 attempted draws in total. Protocol hash:
`cbcf1400a38365f2fa24585110da389b64174fc7e3a563e26cf1618a161c588e`.
Controller PID 541958 started two physical workers after checking the six
existing user simulations. Fourteen controller/comparison tests and an
independent source review passed. Every new draw was audited and classified,
then independent linear population means were compared between stages; no pilot
populations are pooled into the new estimates.

The separate [probability/intensity controls](hard-free-line-sensitivity-stage.md)
have completed: four 16,384-draw populations at α=0.2/λ128 and four at α=0.5/λ64,
each compared with the completed conditioned pilot. They preserve the same
components and physical target. Twenty-two tests and the additional α=0.2
analytic sphere references with full independent geometry audits pass.
The completed controller was PID 611099 at
`/vast/xvg/tetramer-mc-runs/hard-free-line-sensitivity-20261001`, protocol hash
`100fdb3b68b2a1bfdf9de6fdce61a7759894d0818f7affa6a9b9e7d0ba977cc6`.
The [completed result](hard-free-line-sensitivity-results.md) fails the fixed
sensitivity gates despite agreement of all regional Qz masses and native–competing
contrasts. It retains one hard-volume discrepancy, two λ64 native weight-tail
failures and all 11 unstable material strata. α=.2 improves importance efficiency
only 3–10%; λ64 increases the observed cloud share of native variance and lowers
native importance efficiency. The primary α=.5/λ128 setting remains unchanged.
The comparison includes the native–competing free-energy contrast itself,
since separate mass checks can miss opposite shifts. No new physical jobs
were launched while the eight-job limit was occupied.

The [retained-index alternative](hard-free-line-index-estimator.md) now has an
independently reviewed derivation, six arithmetic/provenance tests and four
additional Lean-checked identities. The complete 74-declaration axiom audit uses
only standard Lean axioms. This estimator preserves the current draw law but
could avoid the complete new-mixture density evaluation. Its retrospective
variance/cost estimates remain provisional. The completed
[paired retrospective](../results/hard-free-line-physical-index-retrospective-20261001/report.md)
retains 85–89% of full-density importance ESS on the same draws and clouds.
An index-only sampler would need to save roughly 11–15% of runtime to break
even on these observed diagnostics. That saving is unmeasured. The indexed
native-total estimate is also 1.80% lower: 3.83 estimated paired population SE,
but 1.55 paired-row SE. An [independent arithmetic review](hard-free-line-index-discrepancy-review.md)
found no label/mass error and unusually small dispersion of the four population
differences. Both uncertainty diagnostics and their weighted-tail limitations
remain visible. The frozen physical assessment and next size stage retain
their original full-density weights.

The [full-vessel wrapper](hard-free-vessel-guide.md) now accepts this normalized
guide while preserving its probability outside the source R4/capture domain.
It evaluates both mixture densities in world coordinates, divides the latent
density by the Jacobian once, and retains the original wall and vessel support.
Thirteen Rust coordinate/support/normalization tests and nine independent
Python tests passed. A fixed new sphere allocation retained 17,024 attempts;
its two analytic references and disabled-guide/legacy controls passed, followed
by independent reconstruction of all 16,640 new-format rows. These reference
checks do not launch or authorize a protein full-vessel campaign. The cheaper
indexed regional estimator cannot be reused unchanged outside its support;
this extension therefore retains complete-mixture weighting.

The new [full-vessel streaming audit](hard-free-vessel-streaming-audit.md)
processes at most 64 full traces at a time and retains geometry labels for the
later native partition. Twelve deterministic tests pass. Its reductions reuse
the 16,640 already audited reference draws and reproduce all six region/variance
summaries within 5.60×10⁻¹⁵ scaled difference, without repeating geometry,
density scoring, sampling or native classification. Baseline support now also
passes independent auditing. Three 128-draw logging controls reproduce the
archived baseline, Gaussian and hard-free sample streams byte for byte while
adding complete attempt journals; the old executable was not rerun.

The once-only native/pocket partition is implemented and tested for both
streaming audit formats. It preserves every attempted denominator, explicit
outside-pocket native/contact/unbound contributions and all fixed regional
strata. Six deterministic tests pass, including source-support boundaries and
failure preservation. No protein native classifier or physical reference was
replayed. The larger campaign has passed all audits and is classifying its new
poses; the sensitivity campaign is complete with unresolved failures. A new full-vessel campaign and stage
comparison remain to be prepared after the regional evidence is assessed;
no production gate has opened.

A deterministic exact-rational domain bound places the entire R4 chart and
every atom sphere at least 38.96 Å inside the planned vessel wall. This validates
the geometric inclusion needed to compare regional and full-vessel R4 masses,
without new sampling. It does not establish statistical convergence or certify
floating-point execution.

The [new full-vessel preparation](hard-free-streaming-vessel-stage.md) freezes
the requested four-population 65,536- and 262,144-draw stages for both arms.
It uses the current 92-component guide and λ/z=128 in both arms, retaining the
original target and explicit full-domain coverage. Preparation and eight new
allocation/statistics tests pass. No vessel protein jobs have started; the
completed regional campaign still fails convergence checks. The authenticated completed-stage reader now passes three
additional lineage/manifest controls and preserves all saved statistics without
geometry or native-classifier replay. The bounded dispatcher remains required.

The [angular extension](hard-free-pose-line-guide.md) conditions any raw Cayley
coordinate using quadratic sphere-overlap intervals. All 1,536 toy attempts
passed independent reconstruction; six geometry, 22 shared-guide and eleven
Python tests passed. It remains confined to the proposal-audit runner and is
not part of these physical results or the population-size comparison.

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

A further bounded preparation then fitted four components for the newly
identified cooperative competing family using only the original training
populations. It retained 50% uniform support and at least 90% of the preceding
guide density everywhere. On the archived bank/protected holdouts, its targeted
paired second-moment ratios were 0.418/0.548, while merely reallocating existing
components achieved 0.195/0.320 (smaller is better). Those moment estimates had
effective counts of only about five. The new geometry failed its prespecified
recommendation gates; no additional physical draws were launched. This is a
negative result for that guide expansion, with no thermodynamic implication.
The [cooperative-contact report](cooperative-contact-guide.md) records the
geometry, frozen densities, failed holdout checks and inert prospective commands.

### Exact translation conditioning: validated, but not yet efficient

The next implementation conditions one raw translation coordinate on contact
with both scaffold unions, subtracting all hard-overlap intervals. It retains
the other five Gaussian coordinates and evaluates the complete normalized
mixture, including empty-line fallback. Its 50% uniform / 25% original Gaussian /
25% conditioned Gaussian law bounds the density below by half the old guide.
The [derivation and numerical obligations](line-conditioned-gaussian-guide.md)
make explicit why this is a normalized integration proposal and why it cannot
fix missing orientations. The original center-capture domain, classifier and
physical measure are unchanged; no assembly kernel was modified.

Eight interval-geometry tests and nine guide reference tests passed. The latter
also passed in the production release profile; 32 existing latent-region,
physical-guide and SMC regression tests passed against that same executable
(one fixture-writing helper remains intentionally ignored). Independent Python reconstruction
then passed on all 512 fresh proposal draws and 284 archived probe queries in a
separately frozen, one-worker diagnostic. It generated **no new Poisson clouds**.

The fixed-x variant did not justify a physical campaign. Of 65 selected
conditional branches, 34 had no hard-free line segment and 21 had no simultaneous
contact interval at the selected width; only ten could be conditioned. There
were no mass-floor fallbacks. Valid endpoints were 53/256 versus 61/256 for the
baseline; simultaneous contacts within 0.1 Å were 15/256 in each arm. These small
counts do not establish a difference, while proposal/geometry CPU increased
3.45-fold. This timing excludes Poisson work and is not a physical-sampling
speed ratio. The archived critical-tail second-moment estimates also disagree
between their two original source arms and have very low effective counts.
See the [passive results](contact-line-passive-results.md).

This negative result identifies a restriction of the fixed line, not of the
protein model. A separate deterministic inspection of all three raw translation
axes reused 462 saved endpoints, without new draws or clouds. At 0.5 Å, 135 of
the 256 baseline poses have a usable line on at least one axis, versus 88 on x.
But uniformly choosing an axis does not select that union: on the 139 original
Gaussian draws, averaged over the three widths, the usable conditional fraction
changes only from 25.66% to 26.06%. Its concentrated archived weight diagnostics
also show no reliable advantage. The
[axis diagnostic](../results/contact-line-axes-20261001/analysis.json) therefore
does not justify tripling geometric traversal work in another physical pilot.

The next [candidate construction](two-distance-contact-guide-design.md) replaces
all three translations using two chosen atomic contact distances and an azimuth,
while retaining the Gaussian angular marginal. Its polygon area and Jacobian
give a normalized density. A localized azimuth with a uniform component can
retain useful covariance information. This remains a design, not an implemented
or validated improvement; it first needs bounded passive reference and protein
checks. No new physical campaign is justified by these diagnostics alone.

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
