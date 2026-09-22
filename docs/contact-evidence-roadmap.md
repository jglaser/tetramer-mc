# Compression, physical contact weights, and native registry

The useful outcome is either assembly without supplied inter-tetramer registry
information, or a compact proposal representation that permits reliable
comparison of competing equilibrium contact arrangements. Compression is an
algorithmic diagnostic; it does not by itself measure a contact free energy.

With the present rigid bodies, template-free means discovering **inter-tetramer**
arrangements without native docking information. Native intratetramer geometry
is already supplied by the body shape. This experiment cannot establish
template-free crystallization from flexible monomers.

## Current objective — revised 2026-09-21

Determine whether the specified hard protein-shape plus ideal-depletant model
supports native inter-tetramer assembly, using validated reversible sampling
and independent physical free-energy evidence. Retain the involutive transport
kernel as a tested building block, and distinguish inefficient proposals from
thermodynamically rare contact environments before further compression or
large assembly campaigns.

The immediate milestone is a controlled estimate of the relative statistical
weights of native and competing contact regions at radius 1.5 Å and activity
0.035 Å⁻³. Keep the physical domain and region definitions fixed; include the
remaining configuration space explicitly. Validate normalizers with independent
populations, proposal and population-size sensitivity, and reference limits.
Resolve disagreement with earlier SMC estimates rather than treating either
the new one-way trajectories or the older estimates as equilibrium truth.

Use those weights to set the sampling target. Improve effective independent
contact samples per CPU and agreement between different initial conditions;
measure completed exchanges where both regions have appreciable weight.
Equal forward and reverse transition rates are not required when equilibrium
occupancies differ. The observed 5.1-fold accepted-move rate increase is a
secondary diagnostic, not completion of this milestone.

Then compress and discover contact basins while retaining the physical marginal
and measurable coverage of important regions. Test geometry-only inter-tetramer
assembly against a native-informed positive control, extending conditional
contact calculations to cooperative neighborhoods and assembly stability.
An alternative successful outcome requires converged physical evidence that
the specified conditions fail to stabilize the necessary arrangements. Failure
of a proposal, a short assembly trajectory, or one isolated pair calculation
does not establish that conclusion. The overall objective remains unfinished.

In parallel, formalize the balance argument in Lean with pinned dependencies
and a reproducible checked build. On a measurable state space, use a target
probability measure, a Markov proposal kernel, and a measurable acceptance
function of the old and proposed states. Symmetry of the accepted probability
flow should imply detailed balance and invariance of the completed kernel,
including rejection as a self-transition. The raw proposal need not preserve
the physical target. Extend the result to normalized auxiliary distributions
and recovery of the physical marginal. Keep the theorem's mathematical
assumptions explicit and connect them to the implemented moves; the general
theorem alone does not verify floating-point geometry or executable code.

## Approved finite-system endpoint and execution order

The first decisive endpoint is equilibrium native assembly in a finite system
at approximately **106.8 μM tetramers**, with N=12/N=24 and equal-volume boundary
checks. A bulk crystal-stability conclusion is separate. Keep the repaired shape,
1.5 Å depletants, activity 0.035 Å⁻³ and physical measure unchanged.

The [conditional-ray campaign](conditional-ray-reference.md) implements the
predeclared 786,432-draw, six-arm regional comparison, with two independent
Poisson clouds per valid pose. Its frozen R4 is a fixed conditional subset.
Every proposal and invalid zero retains the unconditional denominator; the
complete native classifier remains independent of proposal construction.
Full-vessel importance comparisons and subsequent production assembly runs are
gated on regional convergence. A failed fixed campaign leaves the thermodynamic
question unresolved and identifies unstable strata rather than changing the
region or enlarging the allocation after seeing results.

Once that gate passes, compare matched local, redraw and transport kernels by
contact-fingerprint ESS per CPU, initial-condition agreement and completed
environment exchanges. Freeze the native-blind memory-derived model before
held-out coverage/compression evaluation. Then compare all-mobile dispersed,
competing-aggregate and native-seeded starts at N=12/N=24, four streams per
preparation and proposal arm. Retain native-informed controls and compare
spherical versus periodic boundaries with the common valid move set.

The optional [frozen cluster-size bias](frozen-assembly-bias.md) is implemented
and reference-tested for later reversible assembly free energies. It corrects
each elementary kernel and stores exp(B) physical reweighting factors; no
protein biased-production result is implied. The [Lean project](../formal/README.md)
now has 38 checked theorem audits, preserving the original 19 and adding the
Poisson estimator and count-gate bridges. None of these correctness results
establishes equilibrium sampling by itself.

## Saved-data coverage diagnosis — 2026-09-22

A [matching-target review](../runs/mobile-conditional-ray-extremes-review-20260922/report.md)
shows that all 5,064 contributing poses in the independent R5 native-pocket
reference also satisfy current R4. The original 131,072-attempt subset estimate
is log Qz=60.6821, with observed population RSE 7.97% and ESS 169.8. It exceeds
the large-uniform estimate for the entire current native region by 4.815 log
units. The shape, scaffold, measure, full native observer and original
normalizers agree. This is evidence of missed native weight, not a new full-R4
estimate or a rigorous lower bound.

A [separate six-dimensional guide](contact-bank-reference.md) uses the completed
campaign only for proposal training. Its first frozen 48-component version
improved held-out scores on those campaign data but still missed the measured
native pocket, so no physical pilot was launched from it. A separately prepared
56-component version adds observed pocket anchors before fresh validation.
Both versions retain 50% uniform R4 coverage and full untruncated mixture
corrections. Positive training scores do not pass the physical convergence gate;
full-vessel and assembly production remain gated.

The [fresh 56-component pilot](contact-bank-reference-results.md) has now completed
131,072 new attempts, eight populations and two audits. The tighter guide gives
native/no-entry ESS 88.7/323.2, a substantial improvement, but fails native
precision and dominant-contribution criteria. The individual total-region log
weights differ by about 0.14 between widths; their free-energy contrasts differ
by 0.284 kBT. More decisively, the separately declared known-pocket split gives
59.7% versus 32.2% of native weight inside R5, with a 0.760-log-unit discrepancy
(4.02 combined observed population SE). Native-complement ESS is only 17.6/20.5.
Compensating regional errors therefore prevent declaring the total converged.
No dependent full-vessel or assembly production was launched.

## Fresh contact-weight confirmation — 2026-09-22

The saved-pose [continuation diagnostic](../runs/contact-complement-geometry-20260922/report.md)
locates the unresolved weight in recurring, tightly touching A7/B4 arrangements
outside the old R5 ellipsoid. It does not prove these contact regions are
connected. The [new guide and fixed confirmation](contact-refinement-confirmation.md)
retain the old 56 components and add 24 components, one for each of eight
training populations in each of three named regions. The declared held-out
selection favors weighted neighborhood means with 128 neighbors; its physical
second-moment estimates remain uncertain and omit new Poisson-cloud noise.

A **separate 2,228,224-attempt confirmation is now running**, with five arms and
four fresh populations per arm. The main allocation is 131,072 attempts per
population; a 32,768-attempt control tests population size. Other controls vary
all covariances by four, the uniform defensive fraction from 0.5 to 0.2, and the
auxiliary intensity ratio from 128 to 256. Old calculations are not extended
or pooled into these estimates. Full-native R5 intersection and native complement
are now separate decision quantities, alongside complete native and no-entry
weights and the original spatial/angular strata. The frozen workflow runs
independent density reconstruction and full classification after the physical
jobs finish. No result from this campaign is yet claimed.

The independent [R4 SMC control](smc-r4-control.md) provides a different estimator
and mutation mechanism. A geometric proposal-density bridge is needed because
resampling directly into the hard-fluid measure would almost always discard
old-pocket ancestors at the planned population size. Its physical endpoint
remains unchanged. Analytic toy tests validate this implementation; protein
SMC and full-vessel calculations remain separately controlled work. None of
these preparations changes the unresolved finite-system assembly verdict.

## Latest controlled results — 2026-09-21

The [conditional-ray comparison](conditional-ray-reference-results.md) has now
completed all 786,432 draws, 24 populations, six independent audits and the
unchanged native classification. It fixes out-of-region proposal waste but
fails the predeclared weight-convergence gate. Larger guided native/no-entry
ESS is 1.1/2.7; their largest draws supply 95.0%/60.4% of estimated mass.
The larger guided/uniform free-energy contrasts differ by 5.38 kBT, and
radial/angular contributions remain unstable. The full-vessel comparison,
physical mixing benchmark and N=12/N=24 production therefore remain gated.
This is an explicit unresolved sampling limitation, not finite-system
instability. The optional bias and 38-theorem Lean bridge are implemented
and validated independently of that unresolved physical result.

| Goal milestone | Present evidence | Remaining requirement |
|---|---|---|
| Correct reversible moves | Checked balance theorem, complete proposal corrections, reference limits and trajectory audits | These do not by themselves establish mixing |
| Access native growth | A free fourth tetramer registers in 3/4 native-informed runs; all four preattached controls remain connected | Sustained growth and independent returns |
| Remove inter-tetramer templates | Matched geometry-only controls and a portable shape-derived atlas | No new native attachment in its 2 free-start runs |
| Compare physical contact weights | Independently confirmed cooperative native pockets and an explicit remainder decomposition | Competing-contact weights and global tails remain poorly resolved |
| Decide the model question | Positive controlled accessibility at the selected bath | Converged thermodynamic evidence and larger assembly controls |

The present milestone is **native accessibility with a validated,
native-informed proposal**, not equilibrium assembly. The exact reciprocal
control reaches registry in four of four small mobile runs and allows registry
excursions, but it has not produced a neighbor exchange. Portable
[twelve-body free and seeded examples](../examples/README.md) now bundle this
model with local moves, GCA and center shifts. Both pass short execution and
trajectory-readback checks; their long-run assembly outcome is for measurement.

The [full D170 integration](mobile-full-capture-results.md) now finds a second
cooperative native pocket on the exact observed scaffold. Its dominant saved
poses register with both neighbors; they belonged to the old q>1 remainder
because that score described only the original native site. However, total
importance ESS is only 1.2–2.7, and both global proposal arms completely miss
the independently measured original native R4. These are useful discoveries,
not converged equilibrium occupancies. The subsequent
[independent regional reference](mobile-native-pocket-reference-results.md)
confirms substantial weight in the new pocket without a global mixture
denominator. A larger independent repeat gives log Qz=60.68 for R5 with
7.67% observed relative error and ESS170; its finite-region log ratio against
the original native R4 is about 24.60. Independent outer strata extend to
R32, with log Qz=61.03 for the cumulative finite region. They add about 5.65%
of observed mass beyond R8 but do not bound unmeasured tails. The next
discriminating uncertainty is the entire competing-contact remainder in
the original physical wall, followed by mobile growth beyond the known
triangle. Another core cutoff is secondary to those tests.
The [domain coverage review](mobile-domain-coverage.md)
also identifies wall-valid native sites outside D170; full-vessel weights
remain unresolved.

The [full-vessel comparison](mobile-wall-contact-results.md) has now completed
eight populations (131,072 poses) and two independent audits. A broader frozen
guide recovers the original R4 and alternative R32 reference regions, although
its R5 estimate remains low. Dominant observed weight still belongs to the
registered native triangle. The guide improves native ESS to 12.5, but its
unregistered-contact remainder has ESS approximately one. The conditional
point estimates favor native contacts strongly; their missing-tail coverage
is insufficient to call the global free-energy comparison converged.

The [four-body growth control](mobile-four-body-growth-results.md) is now
complete: three of four free starts attach a fourth tetramer in a
catalogue-consistent native network, and all four preattached controls remain
connected. All eight retain the original triangle. The broader atlas produces
both of its free-start attachments, versus one of two with the original atlas;
two seeds cannot establish a speedup. The preattached fourth bodies change
from six-bond motif 8 to two-bond motif 7, demonstrating access to other native
interfaces without proving an equilibrium preference. There are no complete
attachment–detachment round trips. The native-informed runs remain the positive
control for geometry-only discovery, while independent contact-weight work
resolves the remaining global uncertainty.
The [geometry-only control](geometry-only-growth-control.md) is now complete.
Its 96 unchanged shape-derived charts use the same reciprocal construction and
physical move schedule. Both free starts form transient exclusion contacts but
neither reaches native registry in 2,000 sweeps; both preattached starts retain
their initial six-bond interface. All four retain ABC. The native-informed
control accesses interfaces this geometry atlas has not reached, so its failure
to grow cannot establish a thermodynamic absence of native assembly.

The [fresh threshold-shoulder reference](mobile-threshold-reference-results.md)
shows that the largest no-entry contacts are near the existing native-entry
boundary. Its fixed finite R4 has a well-resolved hard-only volume, but the
depletion-weighted no-entry estimate still has ESS 2.7 and 61% observed relative
error. It is 2.67 log units above the broader global proposal's estimate on
the identical R4. Native entry dominates the observed weight, yet these
disagreements and unresolved tails still prevent a converged global
free-energy claim. Pose coverage, rather than Poisson noise alone, is the
main observed limitation.

An [exact orientation-marginal/member-shell proposal](entry-shell-reference.md) targets
that threshold neighborhood while preserving the full R4 integration domain.
It sums all overlapping proposal shells and keeps a 20% uniform branch;
every out-of-domain or hard-invalid draw remains a zero. Independent Python
density reconstruction agrees with 32,768 Rust sphere draws within 7.2e-15
in log density, and both analytic hard-sphere/depletion integral controls pass
their predeclared statistical checks. The protein reference has also completed:
all 65,536 unconditional draws pass the density audit, but total depletion
weight still has ESS 2.6 and 92% between-population relative error. It does not
resolve convergence. Only 831 of 52,320 guided draws enter R4; the uniform branch
supplies 99.72% of observed no-entry weight. The completed comparison therefore
does not demonstrate a sampling gain from these shells. This is regional
integration, separate from the assembly
moves supplied in the examples.

The [regional weight ledger](ab-regional-weight-status.md) now combines
independent native, shoulder, intermediate and far controls for the original
AB capture target. Current point estimates favor native over the combined
other regions by 10.65–10.74 kBT. This is an envelope of controlled estimates,
not a confidence interval; remaining competing-region uncertainty is explicit.
Missing native mass would strengthen the one-sided comparison. A separately
measured native subset already provides evidence in that direction, without
needing every native tail resolved first. Formation and motion of the AB
neighbors are absent from this conditional calculation.

The [conditional shoulder benchmark](shoulder-docking-benchmark.md) therefore
tests regions with appreciable conditional weight rather than demanding
frequent exits from an overwhelmingly populated native region. Its independent
40,000-cycle repeat yields 3, 202 and 508 direct–geometry roundtrips for local,
posterior redraw and correlated transport. Transport retains a 1.92-fold
observed roundtrip-per-CPU advantage over redraw, with initialization-mean
TV 0.030. Half-record discrepancies persist, so apparent ESS is not promoted
to a stationary speedup. Every attempted move and every post-burn contact
frame was audited; all runs and repeats remain in the comparison.

The [all-mobile native-informed control](mobile-posterior-pilot.md) is now
complete: twelve 2,000-sweep runs retain the full-mixture capture kernel,
posterior transport, known one-neighbor competitors, native charts and AB
shoulder charts. Five of six dispersed starts reached a native-connected
three-tetramer group. All six preassociated starts remained native-connected.
Four native pair edges were lost and three returned, but these changes left
the exclusion contacts intact. No partner exchange or exclusion-contact
detachment occurred. In particular, the conditional transport gain does not
yet establish improved mobile-cluster equilibration.

The next discriminating comparison must resolve the competing attachment
seen in the remaining dispersed run, retain independent initial conditions,
and count contact-environment changes rather than whole-cluster motion. A
geometry-only control must remove native-derived charts throughout the atlas,
not merely its separately allocated native block. Current successful access
uses supplied inter-tetramer registry; it is not completion of the discovery
or equilibrium objectives.

The [persistent mobile contact](mobile-competing-contact.md) is now located
explicitly: it lies outside the original 18 Å capture ball, at 96.17 Å in the
aligned AB frame. The observed scaffold has a small deformation, retained
exactly in the new finite-region references. The atlas has concentrated
support for this contact in one anchor ordering and only broad defensive
support in the other. This supplies a concrete coverage hypothesis for the
matched proposal control, independently of the earlier conditional shoulder
speedup. That eight-run control is now complete: all four augmented runs and
three of four legacy runs reach native connectivity, without a uniform
first-attachment advantage. One augmented run detaches from its initial
neighbor, attaches to the other and eventually regains the initial contact;
other native-edge changes retain exclusion contact. This identifies an
accessible route, not stationary efficiency or whole-basin equilibrium.
Independent finite-region integration on the exact observed scaffold now
extends through competitor R12. Its measured native-R4/competitor-R12 log
ratio is 15.33, but the two new outer shells have ESS only 6.69 and 3.01;
their high concentration and all coverage outside R12 remain unresolved.
The [outer-shell diagnostic](mobile-outer-importance-design.md) motivates a
frozen defensive importance guide with exact weighting and independent
fresh samples on these same regions. An [exact reciprocal proposal representation](reciprocal-contact-proposal-design.md)
is implemented and reference-tested. Its eight-run control reaches native
registry in all four reciprocal and three of four legacy preparations, with
two body-0 registry excursions but no exclusion-contact loss or neighbor
exchange. A return to an unregistered pair lies at geometric chart radius
18.84, outside the present R12 integration; its other neighbor is mobile.
Thus both domain coverage and stationary exchange remain unresolved. Both
atlases remain native-informed; neither target weights nor new contact fits
are supplied by symmetrizing the proposal.

The fresh [guided outer-shell control](mobile-outer-importance-design.md)
has now completed all eight populations and both audits. Observed relative
errors fall to about 19–21%, from 39–58%, at the same attempted sample count.
The outer shell's log weight rises from 19.30 to 20.29, and effective sample
sizes remain only 23–28. Thus the guide helps measured precision, while the
physical comparison still needs remainder coverage and sensitivity controls.

## Earlier evidence and the coverage problems it revealed

The independent normalization work has exposed a substantial coverage
problem in the earlier one-neighbor estimates. Recentered frozen proposals
give much steadier native and shoulder estimates, but fresh SMC also found
a compact distant contact that the previous guides almost entirely missed.
Independent importance sampling assigns substantial weight to a fixed region
around that contact; its full surrounding mass remains unresolved. The four
fresh far-region SMC populations span log Q from 14.16 to 27.27 and are
dominated by one population. The earlier native probabilities below are
historical estimates, not a settled physical reference. See the
[proposal and independent-normalizer controls](smc-normalizer-proposals.md)
and [explicit-far reference assessment](explicit-far-smc-reference-plan.md).

A separate [uniform latent-region integration](latent-region-normalizer.md)
now estimates log Q=21.195 for the predeclared radius-three ellipsoid, from
65,536 unconditional draws in four independent populations. Its observed
relative standard error is 5.6% (6.8% using the four population means).
This control avoids Gaussian-mixture importance denominators and retains the
exact translation/Haar Jacobian. It confirms consequential weight inside
that fixed region; it does not confirm the complete far normalizer or exclude
important mass outside the region.

The new contact is hard-valid and has appreciable exclusion overlap in
independent geometry checks. It is not an internal relabeling of the tetramer
or one of the inspected nearby crystal placements; see the
[endpoint audit](explicit-far-endpoint-audit.md). However, all 512 inspected
discovery endpoints clash when a second prescribed native neighbor is added.
Most native and shoulder snapshots also clash. This
[unweighted geometric screen](contact-neighborhood-survival.md) motivates a
cooperative-neighborhood free-energy test; it is not an equilibrium population
comparison or a bound on the entire continuous contact region.

A subsequent [cell-cover certificate](frozen-deep-region-clash-certificate.md)
now excludes the **entire** frozen latent balls of radii 3, 5 and 8 against
that second prescribed neighbor, with zero unresolved cells. Explicit atomic
witnesses and global Cayley displacement bounds replace inference from
snapshots. This is an analytic enclosure checked in floating point, not a
formally rounded interval proof. It remains local to these regions and does
not account for forming the prescribed neighbors or for other competitors.

The [MIS-refined guide](mis-refined-normalizer.md), trained on independent
regional controls and evaluated with fresh global draws, now gives log Q=21.830
for the fixed radius-eight region with 4.0% observed relative uncertainty.
The middle shell is much better resolved; the outer shell and remaining
far-region mass still need coverage checks. An independent
[uniform cover of the complete native region](native-region-cover.md) now gives
log Q=15.260 from 33,554,432 unconditional draws in eight fresh populations.
Its observed relative error is 29.7% and importance ESS only 11.3, with one
pose contributing 22.4%. It is consistent with the guided estimate near 14.98
but remains sensitive to rare poses. The smaller preliminary cover campaign
is kept separate. Complete geometric support does not certify convergence.

The point ratio of the known radius-eight competing weight to this native
estimate is about 710 (log ratio 6.57). This is a comparison of two specified
regions in the one-neighbor environment, not a global native probability.
The [comparison figure](../runs/cooperative-contact-evidence-20260920-v2/regional-contact-evidence.png)
shows independent population variation and the uniform-versus-guided controls.
The continuous hard-clash certificate eliminates this same competing region
when the second prescribed neighbor is present; the surviving native mass and
other competing arrangements in that environment still require integration.

The first [whole-native factorial screen](native-cooperativity-plan.md) has
now run the empty, second-neighbor-only and both-neighbor cases, reusing the
independent first-neighbor result. The second-neighbor estimate has observed
ESS 81, but the both-neighbor estimate has ESS 1.03 and one pose contributes
98.5%. The geometric hard volumes are much steadier. This identifies severe
weight concentration in the two-neighbor native region; it does not yet
resolve thermodynamic cooperativity. The fixed-neighbor formation cost and
the remaining competing configurations are still separate questions.

The [posterior-source transport](posterior-chart-involution.md) is now
implemented and has passed proposal-law, exact physical-reference and runner
replay tests. It removes the static source-label penalty while retaining the
physical correction. In the [matched protein pilot](posterior-docking-pilot.md),
four 5,000-cycle runs at correlation 0.9 complete three round trips between
the native core and the frozen competing radius-three region; four matched
zero-correlation redraw runs complete none. Only three events are observed.
The transport arm changes poses 1.87 times faster per sampler CPU second,
but its finite-record contact ESS per CPU is lower, and one trajectory spends
much of its time outside the known radius-eight region with different contact
statistics. This demonstrates reversible native accessibility, not a settled
mixing speedup or equilibrium coverage. That excursion supplies another
specific coverage test. Both arms use the same separate uniform branch.
The [frozen extension-region integrals](outside-r8-local-region.md) are now
complete. The prolonged excursion occupied a specified region much more
often than its independently estimated mass permits relative to old R8.
Recorded return proposals identify a strong full-proposal-density penalty.
A [frozen covariance extension](outside-r8-atlas-extension.md) then shortened
first returns from 69–891 cycles to 1–16 in four matched short controls.
This is a measured repair of that contact-coverage problem; the chains remain
too short to establish equilibrium contact statistics or global assembly.

Centered [nested native covers](nested-native-cover-mixture.md) did not resolve
the two-neighbor mass even after a larger independent run: one sample still
contributed 47.7%. The [frozen covariance-guided cover](native-guided-cover.md)
uses those observations only for proposal fitting, retains full geometric
support, and tests shifted translation–rotation covariance on fresh draws.
Its independent guide-width and population-size controls target this specific
integration bottleneck. They do not remove the need to measure competing
regions or the cost of forming the prescribed neighbors.

The [Lean subproject](../formal/README.md) now checks measurable-state
accept/reject balance, asymmetric Metropolis–Hastings, normalized conditional
auxiliary marginals, reference-preserving deterministic involutions, and
invariance under kernel composition. This provides
a reproducible mathematical foundation while leaving concrete implementation,
ergodicity, and sampling convergence as separate obligations.

## What the existing evidence establishes

The [matched atlas pilot](atlas-transport-pilot.md) uses a 28-component,
native-informed atlas, depletant radius 1.5 Å and activity 0.035 Å⁻³. Its four
arms form 34, 37, 37 and 28 native body-pair bonds, respectively, and none
detach during the 400-sweep runs. This establishes accessibility with the
atlas, not equilibrium native occupancy or contact stability.

At **the same bath parameters**, an earlier one-mobile/one-fixed tetramer
calculation offers a useful independent reference:

| Fixed neighborhood | Conditional SMC native probability | Competing-start MC reaching native in 5,000 cycles |
|---|---:|---:|
| site0-m1 | 0.494, interval [0.358, 0.658] | 0/8 trajectories |
| site1-m1 | 0.610, interval [0.433, 0.781] | 0/8 trajectories |

Native-start trajectories in both environments crossed the native boundary in
both directions, while competing starts remained outside. These SMC estimates
and the start dependence initially motivated the sampling-obstruction test.
The newer coverage findings above prevent treating them as a settled
thermodynamic reference. The SMC intervals
are conditional on explored populations; competing-basin normalizer variation
and, at site1, domination by one population remain limitations. These values
are conditional on an 18 Å capture domain and prescribed fixed neighbor,
not native probabilities in the dilute assembly system. See the
[matched MC and SMC report](/home/xvg/protein-nucleation/results/coordination-mc/narrow-water/report.md)
and [conditional-normalizer protocol](/home/xvg/protein-nucleation/COORDINATION_REGISTRATION_TEST.md).

There is also evidence for geometry-only proposal accessibility. A shell
proposal retaining 128 geometry-derived orientations generated 11 and 9
native candidates at the two sites using 15° caps and 2,097,152 raw draws per
site. Native coordinates were used only for diagnostics. Overlap weighting
increased their selection frequency, but this was conditional docking with a
supplied neighbor and capture domain, not equilibrium assembly. See the
[proposal qualification](/home/xvg/protein-nucleation/results/configurational-bias-depletion/native-proposal-qualification/qualification-report.md).

The older nominal native penalty of 10.225 kBT concerns two mobile tetramers
near a fixed seed at **radius 6.3 Å, activity 0.0008430997064 Å⁻³**. Its competing
normalizer was dominated by one population, and the full gap was explicitly
unresolved. It cannot rule out native assembly at the current bath. See the
[qualified comparison](/home/xvg/protein-nucleation/results/tetramer-native-adsorbed/comparison/report.md).

The subsequent [involutive docking pilot](involution-docking-pilot.md) integrates
the fixed-chart map with the exact depletion gate. Its covered, five-component
conditional atlas gives native access from competing starts, but none of the
20 runs of 5,000 cycles completes a return between the contact cores. The
`gamma=0.9` arm produces about five times more accepted pose changes per CPU
than independent redraw; its two native-start departures reach only `q≈2`,
not the distant competing region near `q≈30`. This is a movement-rate gain,
not a demonstrated contact-mixing gain. The physical basin-weight and
compression milestones therefore remain open.

## Keep three different quantities separate

1. **Proposal coverage:** probability of proposing a hard-valid pose in a
   registered or competing contact region, and the corresponding accepted
   transitions per CPU second. Component activation, learned mixture weight,
   held-out likelihood and compression ratio describe this representation.
2. **Physical basin weight:** an integral over translation and proper rotation
   of the hard-shape plus many-body depletion density. It must come from a
   validated physical chain or a corrected normalizer estimator.
3. **Registry role:** whether a contact family participates in a consistent
   multi-body arrangement, constrains relative registration, or supplies a
   missing connection in the native contact graph. This is not determined by
   the contact's isolated equilibrium population.

These categories overlap: a contact may be both common in equilibrium and
important for registry. A rare contact may be necessary for a particular
crystal graph yet unfavorable in isolation. Several Gaussians may describe
one physical basin, while one broad Gaussian may cover several basins.

Deleting a proposal component and observing fewer native events shows lost
proposal coverage. It does **not** establish that the corresponding physical
contact is unstable. The uniform defense maintains formal support but may be
far too slow to replace a deleted narrow component during a finite run.

The earlier [sparse basin fit](/home/xvg/protein-nucleation/results/hierarchical-pose/cayley-line/basin-discovery-pilot/assessment/report.md)
already found that a common-metric density-height penalty can penalize narrow
native clouds much more strongly than broad competitors. Its empirical peak
factors were 7.511×10⁴ versus 17.76. Preserve the s=0 neutral control and track
rare-region coverage separately from aggregate likelihood when compressing.

## Minimum controlled physical test

Retain the current radius and activity, and first return to the two existing
one-neighbor capture environments. Use matched native, competing-contact and
locally unbound starts. Compare the full atlas with its compressed versions
and with geometry-only discovery after establishing the physical basin-weight
reference and a useful exchange baseline, retaining the same physical kernel,
proposal widths, capture domain and CPU accounting. Separate crossing of a
native threshold from a completed return between well-separated contact
cores; frequent local recrossings do not establish global mixing.

Freeze a geometry-based partition into contact regions using separate pilot
data: relative poses together with residue-patch contact fingerprints, and an
explicit unbound/remaining region. Fix the partition before production.
The partition need not coincide with Gaussian labels. Native coordinates can
then annotate the regions after discovery. For comparison with older work,
also retain the established native criterion as a separate observable and
report its tolerance sensitivity.

For fixed neighbor set S and relative pose x, use

\[
C_S(x)=|E(x)\cap\bigcup_{j\in S}E_j|,\qquad
Q_b(S,z)=\int_{\mathcal D}dx\,H_S(x)1_b(x)e^{zC_S(x)}.
\]

Here E is the depletant exclusion union, H enforces hard nonoverlap, and dx
is translation volume times normalized SO(3) Haar measure. The capture domain
and basin definitions remain fixed. This expression uses the union directly,
so multiple-neighbor overlaps are counted once; no pairwise depletion
approximation is needed. Report Q_b or ratios of Q_b, not fitted Gaussian
weights. A ratio against the unbound region is a conditional association
free energy; converting it to a concentration-dependent binding probability
requires the specified reference volume or concentration.

An exact, nonnegative importance estimator is available. For any normalized
full-support pose proposal g, draw x from g and an independent Poisson cloud
of intensity λ in a conservative envelope of the overlap. Thin by exact
membership to obtain K distributed as Pois(λ C_S(x)). Then

\[
\widehat Q_b=\frac{H_S(x)1_b(x)}{g(x)}
                 \left(1+\frac z\lambda\right)^K,
\qquad E[\widehat Q_b]=Q_b.
\]

Hard-invalid and out-of-basin draws carry zero weight and stay in the sample.
For fixed x the relative variance of the Poisson factor is
\(\exp[z^2C_S(x)/\lambda]-1\). This identity permits a cost/variance diagnostic
before a long calculation. The pose-weight variance may remain severe even
if the Poisson variance is small. Exponentiating an ordinary unbiased volume
estimate is not this estimator and introduces bias.

The existing conditional SMC implementation is a more practical first choice
when the direct estimator has large variance: it already uses this positive
Poisson representation, known proposal densities, fixed annealing schedules,
independent populations and the correct conditional cloud intensity. Its
normalizers are averaged on the linear scale, including zero estimates, before
taking logarithms. Reuse its hard-only sphere and constant-density/binomial
controls; inspect population-size sensitivity, independent-population weight
concentration and proposal sensitivity. Bootstrap intervals alone do not
certify absence of unvisited high-weight modes.

Only after the one-neighbor check, add compatible fixed neighbors using the
existing site0 and site1 ladders, keeping the same domain. Measure

\[
\Delta F_{N,A}(S)=-\log[Q_N(S)/Q_A(S)].
\]

A reduction when adding a neighbor measures conditional registration
enhancement. The full factorial contrast using empty, first-only, second-only
and both-neighbor environments measures conditional free-energy cooperativity.
It includes pose correlations and hard-support changes and can be nonzero
even for additive pair potentials. The simple one-to-two-neighbor difference
also includes ordinary added binding.

To isolate depletion nonadditivity itself, use an additional pair-additive
reference on the **same** both-neighbor hard support:

\[
Q_b^{\rm pair}(A,B)=\int_{\mathcal D}dx\,H_A(x)H_B(x)1_b(x)
                  e^{z[C_A(x)+C_B(x)]}.
\]

At each pose the exact union volume obeys
`C_AB=C_A+C_B-|E(x)∩E_A∩E_B|`. Consequently
`-log[Q_b(A,B)/Q_b^pair(A,B)] >= 0`: the triple-overlap correction reduces
the weight compared with the pair-additive model. It can still change relative
registry preference because the reduction differs between contact regions.
Fixed-neighborhood results omit the assembly cost and fluctuations of the
prescribed neighbors. They diagnose conditional selectivity, not a bulk
nucleation barrier.

## Decision criteria

Compression succeeds when a substantially smaller active representation
retains useful hard-valid proposal coverage and completed environment returns
per CPU time, while physical conditional basin ratios remain consistent with
independent normalizer estimates. At this point an ablation can distinguish
components needed to represent narrow registry regions from components
describing common broad contacts; physical interpretation follows from Q_b,
not from whether the component survived the sparsity penalty.

Geometry-only assembly requires that native docking references never enter
initial proposal centers, component birth, ranking or weights. Keep the native
catalogue only in a separate diagnostic path. Multiple independent dispersed
starts, an increasing ordered cluster and registry-consistent multi-body
contacts supply stronger evidence than one native pair event. Irreversible
growth over the observed horizon still does not establish equilibrium.

If assembly remains absent, the result is informative only after the sampler
can exchange competing environments and conditional free energies converge
under independent controls. Otherwise the outcome remains a sampling limit,
not evidence that the hard-shape plus ideal-depletion model prevents assembly.
