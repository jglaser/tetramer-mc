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

Exact enumeration now finds 12–16 vacant cooperative catalogue placements on
each of the four blind solution quartets and 33 on its nine-body seed
component. Thus the retained ideal structures are not terminal through a lack
of site-disjoint native extensions. The
[extension figure and scope](expanded-native-contact-diagnostic-20261004.md#vacant-cooperative-extension-sites)
separate this geometric result from the still-untested atomic clearance and
binding weights in the surrounding fluid.

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

The primary physical-weight campaign and its postrun have completed; the new
sensitivity follow-up is active. The earlier native observer also completed:

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
147,488 retained endpoints have completed contact and instantaneous
native-label analyses. A separately validated expanded observer has also
completed all 221,232 endpoints from the 32 new chains and sixteen reused
controls, adding the twenty missing native placements without altering the
old labels or sampling kernels.

The [expanded comparison and figure](partner-atlas-expanded-native-comparison.md)
show native accessibility hidden by the old catalogue: guided m8 has mean
native fractions of 69.4% and 72.5% from its two starts, versus 47.4% and zero
for local moves. However, its placement histograms still differ (descriptive
TV 0.2271), and one prepared stream remains in a single native placement for
all production. Generic contact-fingerprint apparent ESS/CPU is below local
in all eight m8 comparisons, with median ratio 0.4273. High native occupancy
and similar means therefore do not establish equilibrium mixing or an
assembly speedup. The m4 control also reaches supplemental native placements
frequently but includes trapped streams.
The native-label adapter has passed eight synthetic tests, including complete
inventory admission, rejected residence, multilabel caching, full-CPU metrics
and failure-prefix preservation. It requires the completed contact/arithmetic
observer first and reuses sixteen cached native controls without reading their
old trajectories. The contact-analysis handoff passed fifteen tests and refuses
incomplete allocations. The subsequent native handoff passed fourteen tests
and refuses dispatch before that contact analysis completes. The handoff
watcher completed both previously declared analyses successfully.
The [revised construction](mixed-atlas-surrogate-design.md#decision-after-the-completed-native-audit)
records the balance argument and retains a direct physical control.

A reviewed handoff watcher waited for completion and controller exit before
starting the two already validated analyses serially. Twelve synthetic
lifecycle/deadline tests passed. Its frozen record is
`results/partner-atlas-autonomous-handoff-20261004/freeze.json`, SHA256
`fabbcefb49f568ea462bb26cab6acd9bd3a1c334569b216fb5d6eeb438e73457`.
It added no physical draws, refused retries and preserved failures. The
separate expanded-observer handoff also completed; its 23 observer/preparation
tests and six dispatch checks passed. That observer retained all rejected
residence and classified both new histories and reused controls consistently.

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
previously validated contact and native observers, including the supplemental
catalogue, have completed. No production assembly kernel has been promoted
on accepted-move throughput alone.

The N=12/N=24 bank's archived application sources were recovered exactly from
commit `9f2ae102ed100df68ad582d52043accd16490646`, including all 399 pinned
vendor files. Its 49 archived application/source files match the prepared
bank. The [isolated release build](finite-assembly-baseline-release-20261004.md)
passed the same forty integration and one configuration references, with all
children drained and source bindings rechecked. The existing production
executable is unchanged. This closes an executable-provenance prerequisite;
it does not open the physical convergence gate or launch the assembly bank.

The [expanded starting-state audit](finite-start-expanded-native-20261004.md)
has now completed all 48 assets and 8,208 pairs. No supplemental native
placements were found: the competing and dispersed starts remain without
native contacts, while the supplied eight-tetramer seeds retain thirteen
contacts. All exact graph checks completed without unresolved components or
periodic winding. The existing preparations were retained without redraws.

A separate analytic bound also excludes all twenty supplemental native
placements from the frozen R4 contact-weight region, against both fixed
neighbors and in either pair order. An independent calculation agrees with
the geometric margins to 2.85e-14 Å. Thus the new trajectory labels do not
reclassify the completed regional mass contrast. This resolves a definition
ambiguity, not the remaining convergence or full-vessel mass questions; see
the [updated regional report](native-class-physical-results-20261004.md).

The [native-extension clearance audit](native-extension-clearance-20261004.md)
completed all 113 components and 7,207 site-disjoint points, with no Monte Carlo
draws. Only one of 52 cooperative quartet points is hard-clear; 51 clash with
their own observed quartet. Both seeds retain open cooperative points. The
sub-Ångström gaps motivate a controlled ideal-component comparison and nearby
pose feasibility checks, not a physical instability conclusion. Nineteen
synthetic tests and an independent full-output arithmetic reduction passed.
