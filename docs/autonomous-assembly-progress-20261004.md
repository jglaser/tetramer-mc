# Assembly progress, 4 October 2026

**The finite-system thermodynamic verdict remains unresolved.** Completed
calculations establish native accessibility and expose limitations of the
tested proposals. They do not establish equilibrium assembly or instability.

The decision conditions remain the repaired rigid tetramer, depletant radius
1.5 Å, activity 0.035 Å⁻³ and approximately 106.8 μM. The diagnostics below
have different physical conditions and must remain separate.

## Completed original-condition regional weights

The independent importance calculation now estimates native-minus-competing
free energy at **−18.777 ± 0.054** and **−18.772 ± 0.083 kBT** (population-based
95% half-widths) for its two proposal arms. This is one mobile tetramer against
two fixed neighbors, restricted to the unchanged R4 region. Main importance
ESS, contribution concentration, relative error and proposal-agreement checks
pass. Material strata and new-guide sensitivity studies remain unresolved;
coverage outside R4 is still required. Both the 81-stage campaign and its
two-stage postrun completed. The [results and figure](native-class-physical-results-20261004.md)
locate the remaining SMC discrepancy mainly in the native R4 complement of the old reference region,
and preserve the existing deterministic bound on unbound mass inside R4.

## New contact-catalogue finding

A completed supplemental diagnostic changes the interpretation of the old
growth comparison. Twenty additional hard-valid native monomer-orbit placements
are missing from the original fourteen-template observer. In the final
geometry-only endpoint they add 70 contact pairs: 29 → 99, with 40 → 155
tetramers participating, seed-connected size 8 → 9, and largest separate
component 2 → 4. The informed endpoint adds 34 pairs: 248 → 282, with seed
size sixteen and largest separate component ten unchanged. Every expanded
nontrivial component admits an exact injective monomer-site embedding.
All 113 components also fit the measured member centers with RMS error below
0.95 Å using one rigid alignment each. The geometry-only seed component has
0.225 Å RMS error; its four separate four-tetramer components have 0.164–0.228 Å
RMS errors. No individual-body relaxation or fitted cell strain was allowed.

The [supplemental report and figure](expanded-native-contact-diagnostic-20261004.md)
preserve the original counts and all new labels separately. This establishes
previously missed native registration, not equilibrium assembly. It also means
that the old physical region “contact without native entry” must not be read
as proof of nonnative monomer geometry. Running analyses keep their frozen
region definitions.

The full expanded histories have now completed. Both began with no supplemental
or solution contacts. The geometry-only history forms native-registered
solution oligomers up to four tetramers, but its last quarter contains only
one newly observed solution pair. The many observed threshold returns do not
demonstrate physical exchanges. The [history figure and report](expanded-native-contact-diagnostic-20261004.md#completed-histories-accessibility-and-growth-are-different)
separate this positive accessibility evidence from the remaining growth limit.

## Completed growth comparison

At radius 1.4 Å, activity 0.04 Å⁻³ and 500 μM, two completed N=264 histories
share initial poses and the RNG seed. Under the original fourteen-template
observer, native-informed proposals grew the registered seed from eight to
sixteen bodies; the tested native-blind FFT proposal retained eight. Separate
registered components reached size ten and two, respectively. The expanded
endpoint counts above supersede those geometric interpretations without
altering the original history records. Both histories lasted 100,000 sweeps. Neither is an
independent-start convergence test; neither estimates equilibrium occupancy.

The [growth report](completed-growth-tail-20261004.md) and
[figure](../results/growth-z004-native-figure-20261004/native-growth.png)
retain the full histories, sampler CPU and classifier provenance. Their final
exhaustive pair checks agree with the accelerated observer.

A broader exact symmetry calculation resolves an important interpretation
error. All 64 final registered components embed injectively in the native
**monomer** symmetry orbit. The 28 failures under one four-type tetramer
covering do not imply nonnative registry. The blind history's two observed
families already generate three independent translations when the full orbit
is allowed. This does not itself construct a nonoverlapping infinite packing,
fit the measured coordinates, or establish stability. See the
[unrestricted-orbit report](../results/growth-z004-unrestricted-native-orbit-20261004/REPORT.md).

## Completed conditional sampling comparison

The flexible surrogate benchmark has completed 24 chains and its full contact
observer. Each chain has 512 warmup and 4,096 production blocks. It moves pair
27/132 against 262 fixed spectators at radius 1.4 Å and activity 0.0275 Å⁻³,
using the saved 500 μM environment. Forty completed controls were reused.

Guided eight-step updates cost less CPU than the matched unguided eight-step
updates, but have lower whole-contact-fingerprint apparent ESS/CPU in six of
eight start/stream comparisons. All 24 flexible chains have zero external
contact occupancy. All source/prepared internal-patch distribution comparisons
for the flexible arms have total variation one. Faster internal patch returns
therefore do not establish broader contact coverage.

The [comparison and figures](surrogate-internal-native-comparison.md) preserve
each stream, rejected-state residence, undefined ESS and rare-presence caveats.
The completed sphere reference tests support the corrected kernel's physical
target; they do not establish protein mixing. These results do not justify
promoting the current flexible kernel into assembly production.

## Physical weights and completed native observer

The independent physical-weight campaign remains active. The native observer
has now completed:

- The independent physical-weight campaign at
  `/vast/xvg/tetramer-mc-runs/native-class-physical-primary-20261004` uses the
  original 1.5 Å / 0.035 Å⁻³ conditions. Its allocation is sixteen independent
  populations of 16,384 attempted draws: eight per proposal arm, two Poisson
  clouds per valid pose. Invalid draws retain zero weight. All 81 producer and
  audit stages have now completed and drained. The corrected postrun adapter
  passed 30 tests and completed both analysis stages. Its failed preparations
  created no analysis jobs or new physical samples. The completed matching
  comparison and remaining convergence issues are reported above.
- The completed internal-native observer at
  `/vast/xvg/tetramer-mc-runs/surrogate-internal-native-20261004` classifies all
  retained endpoints of 48 completed rigid/flexible chains and reuses sixteen
  completed native controls. All 24 new prepared-start histories have zero
  production native occupancy; new source-start native observations reach only
  motif 6. Guided flexible m8 beats flat8 in native-presence ESS/CPU in only one
  of four source streams. The reused nonlocal m4 reaches additional motifs from
  some prepared starts. It has no external-native or cycle observer and
  generates no physical samples. Its purpose is to distinguish patch motion
  from native registration, with full sampler CPU as denominator.

The allocations, failure rules and limits remain fixed. The user's three
ongoing growth simulations and their production executable are unchanged.

## Next decisions

The current native-class contact guide can now be evaluated in the full vessel,
with source capture retained as a proposal property. Seven Rust controls,
thirteen Python controls, a 160-attempt sphere CLI reference, and independent
reconstruction of every saved CLI row passed. This removes a format/geometry
adapter gap; it does not change the integration target or authorize a physical
conclusion. The subsequent 16,384-attempt analytic sphere reference completed
but failed its contact point-accuracy check at nonzero activity. Exact and
Poisson weights share the discrepancy, and contact importance ESS is only about
74 per activity. A saved-row diagnostic locates most variability outside R4,
where a small number of uniform-cube draws carry large weights. The
[reference report](native-class-sphere-normalization-20261004.md) preserves the
failed gate. An [optional normalized wall-envelope proposal](wall-envelope-normalizer.md)
has now been implemented and checked with nine Rust tests, independent Python
reconstruction and 80 retained toy CLI attempts. A separately calculated
131,072-attempt reference has a conservative point-error bound and has now
completed all seventeen stages, passing all 47 checks. Its largest log-weight
discrepancy was 0.022; the nonzero-activity contact mass was 17.605 ± 0.193
(population SE) versus the analytic 17.781. The preparation/reduction tools
passed thirteen additional checks. The [completed comparison](wall-envelope-reference-result-20261004.md)
keeps the earlier failed result separate. Proposal and allocation both changed,
so this is not an isolated speedup measurement. No finished allocation is extended. The regional campaign's explicit admission
checks also remain. See the
[implementation and validation record](native-class-full-vessel-guide.md).

The completed native audit strengthens the case for a long-range change of the
pair's relative pose: rigid relocation alone preserves it, while the tested
local flexible chains did not bridge initializations. The next optional kernel
now implemented applies a single-tetramer reversible atlas map anchored
to its other mobile partner, mixed with the same local moves. The partner is
unchanged during each coordinate proposal, so its conditional forward/reverse
anchor is identical. This is a hypothesis, not a speedup result or production
default. Six focused and nine existing kernel tests pass in an isolated build.
The independent sphere stationarity reference completed 32,768 calls using all
8,192 cached IID sources: all 292 checks and the Python journal audit passed.
Deliberately omitting or reversing the proposal correction produced measurable
stationarity failures. This [validation](partner-atlas-reference.md#completed-result-and-independent-audit)
supports the next conditional protein test; it does not measure mixing per CPU.
The conditional example now supports direct, guided m1, guided m8 and flat8
partner-atlas arms. An isolated release build passed all fifteen kernel tests
and all 23 example tests, including restarts and preservation of the original
local schedule. The bound model is the existing geometry-only FFT atlas, with
zero transport correlation and its defensive uniform branch; it supplies no
new native prior. The matched 32-chain campaign has now completed at
`/vast/xvg/tetramer-mc-runs/partner-atlas-dimer-context0-20261004`, with one
worker, two reused initializations and four streams per arm. Eight completed
local controls are the primary comparison; 32 additional completed controls
remain contextual. All sixteen observer/preparation tests passed. The
147,488 retained endpoints receive contact and separate instantaneous
native-label analyses after the full allocation. Those postrun analyses are
being completed before reporting a protein efficiency result.
The native-label adapter has passed eight synthetic tests, including complete
inventory admission, rejected residence, multilabel caching, full-CPU metrics
and failure-prefix preservation. It requires the completed contact/arithmetic
observer first and reuses sixteen cached native controls without reading their
old trajectories. The contact-analysis handoff passed fifteen tests and refuses
incomplete allocations. The subsequent native handoff passed fourteen tests
and refuses dispatch before that contact analysis completes. The handoff
watcher has now dispatched the completed-campaign contact analysis.
The [revised construction](mixed-atlas-surrogate-design.md#decision-after-the-completed-native-audit)
records the balance argument and retains a direct physical control.

A reviewed handoff watcher now waits for completion and controller exit before
starting the two already validated analyses serially. Twelve synthetic
lifecycle/deadline tests passed. Its frozen record is
`results/partner-atlas-autonomous-handoff-20261004/freeze.json`, SHA256
`fabbcefb49f568ea462bb26cab6acd9bd3a1c334569b216fb5d6eeb438e73457`.
It adds no physical draws, refuses retries, preserves failures, and waits at
most six hours; the analysis workers retain their own frozen resource limits.

A standalone defensive pair selector and a checked Lean selection theorem are
ready for later all-mobile integration. State-dependent pair selection enters
the **same final physical decision** through its reverse/forward probability
ratio; it cannot be added after committing an endpoint. The
[integration design](flexible-surrogate-all-mobile-design.md) records its
six-test validation, reference limits and remaining implementation obligations.
The production sampler does not yet use this selector.

The completed postrun preserves every failed stratum and compares matching
linear masses with prior estimates. Passing its primary checks alone does not complete the
required population-size/cloud/defensive sensitivity checks or bound the
unmeasured vessel contribution. The existing N=12/N=24 input bank remains
prepared, not validated assembly production. Finite-system stability requires
the independent-start, size, boundary and physical-weight evidence specified
in the original plan.

The three missing sensitivity arms are now running in a separately frozen
24-population follow-up, with the completed primary controls external and
unchanged. Fourteen focused tests and the actual source/metadata preflight
passed. The existing 32-chain partner-atlas simulation also completed; its
previously validated contact and native observers are dispatched by the
existing handoff watcher. No production assembly kernel has been promoted
on accepted-move throughput alone.
