# Assembly progress, 4 October 2026

**The finite-system thermodynamic verdict remains unresolved.** Completed
calculations establish native accessibility and expose limitations of the
tested proposals. They do not establish equilibrium assembly or instability.

The decision conditions remain the repaired rigid tetramer, depletant radius
1.5 Å, activity 0.035 Å⁻³ and approximately 106.8 μM. The diagnostics below
have different physical conditions and must remain separate.

**6 October update:** the [competing-cage comparison](context-competing-cage.md#completed-prospective-comparison-6-october)
completed all 32,768 fresh attempts and 15,968 independent depletion clouds.
Separately centered contact guides expose approximately 224 times more
competing-contact weight than the matched baseline, improving its observed
importance ESS/CPU by 5.75 times. This is a coverage correction for the same
conditional target, not a thermodynamic change. Source-contact sampling
worsens and intermediate contact strata remain unobserved; neither arm
passes the convergence gates. The [saved-data variance diagnostic](context-poisson-noise-budget.md)
now separates cloud noise from pose-weight variance using a seven-theorem
Lean identity and fifteen synthetic controls. Its descriptive cost
forecasts favor better pose coverage over a substantial cloud-intensity
increase; their ratio uncertainties remain unbounded. These
calculations still concern one mobile tetramer in a frozen 500 μM
environment. They do not open the finite-assembly production gate.

The subsequent [frozen bridge pilot](context-bridge-guides.md) completed
16,384 geometry-only attempts and 512 independent saved-pose checks.
Coupled translation–rotation interpolation between the fitted source and
competing means did not recover any valid configuration retaining 5–15 of
the 16 source-contact tokens. Direct bridge draws mostly encountered core
collisions. A broadened endpoint did discover an original-contact tail, but
one pose supplied 87.9% of its pooled hard-volume estimate, leaving that
estimate unconverged. No depletion clouds were allocated to this failed
coverage pilot. This rules out promoting this particular bridge unchanged;
it does not rule out other pose routes or finite-system assembly.

The [subsequent saved-bank depletion extension](context-bridge-physical-extension.md)
changes the tail interpretation: that same pose supplies only **0.1683%**
of the observed A_T physical weight, despite its 87.9% hard-volume share.
All 3,727 valid poses received two independent clouds, retaining all 16,384
attempts. Source-region estimates differ by 0.185 log units, while competing
contact estimates still differ by 1.186 log units (3.27-fold); both remain
unconverged. This postinspection diagnostic redirects effort toward competing
physical contact coverage rather than broadening solely to capture hard
volume. Direct reversible endpoint transport does not require an interpolating
hard-free path. Separately, the new [native-class streaming vessel audit](native-class-vessel-streaming-audit.md)
passed synthetic and saved-fixture controls; its partition/loader registration
and the physical admission gate remain future integration work.

**5 October update:** the [completed physical comparison](context-prior-physical-control.md)
and [matched correlation control](context-correlated-control.md) supersede the
pending fixed-context-prior steps recorded below. All 40 chains completed,
with 460,800 total attempted moves and all rejected residence retained.
Context weighting improves geometric feasibility; retaining latent coordinates
at rho=.95 further increases accepted learned moves from 2 to 13 in its
matched comparison. Neighbor identities agree much better across starts, but
surface-patch histories remain disjoint and apparent contact ESS per CPU does
not improve consistently. These are conditional 500 micromolar neighborhood
tests, not all-mobile assembly results. The next priority is independent
physical contact weights and patch-region coverage, rather than another
correlation sweep. The original-condition sensitivity campaign remains active;
the N=12/N=24 assembly bank remains held behind its convergence gates.

The [completed saved-candidate audit](context-candidate-bank.md) now isolates a
specific coverage failure. All 36,864 independent global proposals were retained
and rescored, with a separate 512-pose density/geometry check. Context weighting
increases contacted candidates from 103 to 613, but neither arm ever recovers the
saved source's full secondary-contact pattern; the best context candidate retains
only four of its sixteen patch tokens. This remains true outside the coarse A
neighbor region. No depletion-cloud allocation is made on this bank: approximately
200 million expected auxiliary points would not repair the missing pose coverage.
The minimal next control is a frozen normalized source-centered component with
fresh independent draws and complete mixture weights. It is a conditional,
source-informed control, not geometry-only assembly or evidence against assembly.

The newest controlled geometric result is a **5.08-fold increase in proposed
hard-clear native extensions**, from 2.95% to 15.00%, after small adjustments
of incoming-particle Gaussian centers. Four independent populations, identical
kernel widths and all failed draws were retained. This is native-informed
proposal coverage, not an accepted-move or equilibrium speedup. The
[coverage report and figure](native-extension-gaussian-coverage-20261004.md)
show the full allocation, quartet heterogeneity and independent density audit.

A first geometry-only construction retained the complete blind atlas and
added normalized narrow children centered using the outside core geometry.
It completed in 190 CPU seconds but changed centers containing only 0.571%
of source probability mass: 85.945% encountered the static pair-list cap
before an objective evaluation. Thus the favorable native-informed result
has not yet generalized to the blind proposal. The
[construction and balance argument](context-relaxed-atlas.md) separate this
work-limit failure from physical steric impossibility. The subsequent
[actual-center diagnostic](context-core-score.md) completed all 2,066 scheduled
scores without candidate caps in 41.37 CPU seconds. Its 473 independently
checked saved objectives/gaps agree with the earlier implementation. However,
2,003 of 2,048 parent centers really clash; only 1.832% of source label mass
has clear centers, increasing to 2.290% in the exported children. These are
fixed-center label weights, not finite-width proposal probabilities.
The exported model's density evaluation has passed 524 frozen-panel comparisons
between production Rust and independent Python, with maximum log-density
disagreement `2.27e-13`. This checks finite-panel arithmetic, not mixing or
physical occupancy; reciprocal-origin center coverage remains incomplete.

A completed atomic blocker attribution resolves the earlier occupied-origin
signal. Only **0.2845%** of parent label weight clashes with the anchor itself;
**80.4301%** clashes with other members of the quartet and **51.9467%** with
remaining fixed bodies. The latter groups overlap. All 2,066 queries completed,
and an independent audit confirmed unchanged global scores and complete body
attribution. These pair centers are therefore mostly usable in isolation;
occupied surroundings dominate this particular destination-feasibility failure.
The next proposal should account for the outside geometry with its exact
forward/reverse selection correction. A practical reversible construction can
refit only two selected charts per attempt, using the original atlas for label
selection. Its [standalone Rust implementation](context-selected-transport.md)
now passes ten unit tests and an independent one-step stationarity control:
16,384 IID sources, 49,152 paired-arm transitions and all 131,072 journal events
were retained. Neither correct arm fails any of the 24 predeclared primary
tests; the deliberately incorrect shortcut fails ten of twelve diagnostics.
Eight focused Lean statements check the selection/accepted-flow algebra under
explicit implementation obligations. This is a Gaussian-target validation,
not a production protein kernel or an observed mixing gain.

An optional [fixed-context branch prior](context-branch-prior.md) now provides
a simpler unchanged-chart control. It retains every reciprocal label and a
10% original-prior component, using the same new probabilities in source
selection, destination selection and density evaluation. A frozen asset for
moving body 77 against the other 263 bodies is prepared from saved overlap
counts. It still needs source/domain certification and runner/checkpoint
binding before a physical comparison. Twenty-eight joint release tests pass.
The same validated change set removes a redundant complete-pair support scan:
the [fixed microbenchmark](atlas-pair-support-cost.md) measured approximately
0.879 ms per old check at 2,048 charts. The new check is constant-time, but no
end-to-end sampling or assembly speedup has yet been measured.

An exact-rational displacement bound, conditional on the saved gap error being
at most 0.01 Å, screens 95.736% of parent label mass as unrepairable within the
current 1 Å / 2° box. All previously capped labels screen. Smaller declared
gap allowances give nearly the same result; the allowance itself is not a
rigorously established floating-point error bound. Thus the next iteration
should address destination selection/initialization, rather than increase
the work budget of the same small search. Larger moves and cooperative
reorganization remain open, and this is not evidence of thermodynamic failure.

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
separate this ideal geometric result from the subsequent fixed-point and
incoming-adjustment checks described below. Binding weights in the surrounding
fluid remain unresolved.

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

The separately allocated same-52-point ideal-component control also completed:
all 52 points are hard-clear against ideal quartets, versus one against the
measured quartets. Ideal minimum gaps are only 0.0369–0.0426 Å. This establishes
the role of the measured distortions at those fixed positions, without
excluding nearby incoming poses. Nine synthetic tests and an independent
52-record audit passed.

The [incoming-only search](native-extension-incoming-search-20261004.md) has
now completed for the same 52 points. With all existing bodies fixed, small
incoming adjustments make 50/52 points quartet-clear and 34/52 entirely clear,
retaining every intended native contact. Median motion is 0.377 Å / 0.590°;
maximum motion is 0.937 Å / 1.894°. Sixteen quartet-clear points remain blocked
by spectators. Both remaining quartet failures report numerical optimizer
success, correctly overridden by their strict atomic clash certificates.
Twenty-three synthetic tests and an independent audit of every slot and
objective evaluation passed. The solve used 2.38 CPU seconds.

This weakens the hypothesis that extension usually requires simultaneous
motion of the existing quartet at these historical endpoints. It does not
measure allowed proposal volume or physical binding weight.

The [paired Gaussian control](native-extension-gaussian-coverage-20261004.md)
has subsequently completed 26,624 candidates: four populations of 64 paired
draws at each of the same 52 slots. At unchanged kernel width, full-environment
hard-clear intended-native coverage rises from **2.95% to 15.00%**, a 5.08-fold
increase. The paired difference is +12.05 percentage points with a four-population
95% t half-width of 0.83 points. Quartet-only coverage rises from 4.09% to 24.05%.
The 17 initially spectator-blocked slots have no full-environment successes
in either arm. Every invalid draw is retained; the full independent replay of
26,624 candidates, 93,184 journal events, poses, densities and scalar predicates
passed. The coverage calculation used 257.30 CPU seconds, plus the previously
measured 2.38 CPU seconds for center optimization. This is a geometric proposal
gain, not a physical acceptance or contact-mixing speedup.

A [geometry-only context-relaxed atlas exporter](context-relaxed-atlas.md) is
now implemented and passed 15 synthetic tests with independent review. It
preserves all original Gaussians and exact reciprocal branches, appending
explicitly normalized ordinary Gaussian children whose centers are adjusted
using outside atomic geometry. No native classifier or current moving pose
enters that adjustment. The physical sampler must retain the complete proposal
correction and defensive component. The native-informed coverage result does
not establish this exporter's protein performance. Its completed blind-atlas
application, density audit and blocker attribution are summarized above and
show why occupied destinations must be addressed. Original-condition
physical-weight sensitivity calculations remain separate and active.
