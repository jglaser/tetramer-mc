# Compression, physical contact weights, and native registry

The useful outcome is either assembly without supplied inter-tetramer registry
information, or a compact proposal representation that permits reliable
comparison of competing equilibrium contact arrangements. Compression is an
algorithmic diagnostic; it does not by itself measure a contact free energy.

With the present rigid bodies, template-free means discovering **inter-tetramer**
arrangements without native docking information. Native intratetramer geometry
is already supplied by the body shape. This experiment cannot establish
template-free crystallization from flexible monomers.

## Current objective — revised 2026-09-20

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
both directions, while competing starts remained outside. The SMC reference
and the start dependence support a sampling obstruction. The SMC intervals
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
enhancement. To call an effect nonadditive, evaluate the full factorial
contrast using empty, first-only, second-only and both-neighbor environments;
the simple one-to-two-neighbor difference includes ordinary added binding.
Fixed-neighborhood results still omit the assembly cost and fluctuations of
those neighbors. They diagnose registry selectivity, not a bulk nucleation
barrier.

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
