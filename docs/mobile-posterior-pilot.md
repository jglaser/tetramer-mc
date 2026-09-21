# Mobile neighbors after the conditional contact benchmark

The conditional shoulder calculation demonstrated sustained contact exchange
with two fixed neighbors. It did not test the formation or rearrangement of
that neighborhood. This pilot releases all three tetramers and compares
dispersed and preassociated preparations at the same depletant radius 1.5 Å
and activity 0.035 Å⁻³.

## Fixed physical and proposal design

Three identical repaired-1LYZ tetramers move inside a protein-only spherical
wall of radius 223.326176723784 Å. This scales the previous 12-tetramer radius
354.508207863371 Å by `(3/12)^(1/3)`, preserving nominal number density. The
fraction of volume affected by the protein wall changes with system size;
this is a matched three-body control, not a finite-size extrapolation of
the earlier 12-body trajectories. The ideal bath
permeates the wall. There is no capture ball, registration window, fixed body,
or native bias in the physical target. Native geometry is supplied in the
rigid tetramers and in proposal charts, which remains an explicit limitation.

Each sweep visits every body once in randomized order, then attempts the
existing spherical GCA and common center shift. Local moves retain 0.2 Å
translation and 1° small-angle scales. Global slots have probability 0.5;
the separate defensive branch has probability 0.1. All three arms use the
same endpoint gate, `lambda/z=64`, 2,047 cells, depth 14 and minimum width
0.5 Å. The complete spectator exclusion union remains in the bath calculation.

| Arm | Local slots | Full-mixture capture slots | Posterior slots | Posterior correlation |
|---|---:|---:|---:|---:|
| Capture only | 0.5 | 0.5 | 0 | — |
| Redraw | 0.5 | 0.25 | 0.25 | 0 |
| Transport | 0.5 | 0.25 | 0.25 | 0.9 |

The primary matched comparison is redraw versus transport. Capture only is
a contextual control because it allocates twice as many attempts to capture.
The full-mixture capture correction includes its uniform reverse-density
floor; the posterior Gaussian correction uses `G(old)/G(new)` and has a
separately selected uniform branch. Mixing these individually valid kernels
with fixed probabilities preserves the physical target. Details and tests
are in [the implementation note](frozen-posterior-assembly.md).

The common 150-component atlas preserves all components from three existing
models: 50% allocation to the 28-chart native atlas, 40% to the 117-chart
native/one-neighbor competitor/outer-extension atlas, and 10% to the five-chart
AB shoulder atlas. These allocations are specified proposal choices, not
physical populations. No means, covariances or component counts are fitted
to the new trajectories. Positive normalized weights and covariance rank are
checked. The exact physical-density mixture identity, including normalized
Haar Jacobians, agrees to `3.55e-15` in log density on 332 test poses.

## Preparations and fixed budget

The dispersed preparation takes the first three positions from the archived
12-body dispersed configuration, scales their centers with the container
radius, and leaves their orientations unchanged. Every atomic sphere pair
and wall atom passes the independent check; there are initially no exclusion
contacts. The nearest wall clearance is 1.1347 Å. The recorded fallback
preparation was unnecessary.

The preassociated preparation releases the original A and B neighbors and
the identity native tetramer, then subtracts their centroid. Their orientations
and mutual geometry remain unchanged. All three interbody atomic checks pass;
the smallest core-surface gap is 0.0369184 Å. No geometry is repaired. Both
preparations are deliberate initial conditions, not equilibrium samples.

The inert [preparation plan](../runs/mobile-posterior-pilot-preparation-20260921/plan.json)
fixes **12 runs**: two preparations × two independent seeds × three kernels.
Each run has 2,000 sweeps, burn 400, and every sweep retained. Seeds are
`115501010 + 1009*i`, with at most 12 workers. Across the campaign this is
72,000 single-body attempts and 48,000 collective attempts. All arms and
replicates reuse the same pose preparation within each start type.

The preparation hash is
`e7c53e81d479a2adc72c92fca696c52d8b848de369b8a3d8ea5fcec8ccf3ee08`;
the model hash is
`d0f3f82960c218ecbb0617b78412bb8072a3f33a726e128465d9418e0690af68`.
Production must separately freeze the reviewed executable, its actual embedded
source bundle, all inputs and the observer. The preparation itself does not
select or launch a binary.

The production campaign is
`runs/mobile-posterior-pilot-12x2000-20260921`, with manifest SHA-256
`3fdb650f183b39607cce9b910b8eb4ee5d2bb264ae2e4890b811a0be42642a4f`.
Its separately built release executable has SHA-256
`aa70ea4dfda9b798341838495a6c55206bb2fed102f89329a3524d542994d7e5`.
All 31 embedded source files match the reviewed source tree. The archived
observer hash is
`bbfd9e757e94401b9851ced603461a3c2a6db8d7488be131bc801ebd053eb81b`.
The controller journal is `runs/mobile-posterior-pilot-launch-20260921` and
the original automatic assessment destination is
`runs/mobile-posterior-pilot-assessment-20260921`.

All twelve physical jobs exited successfully. That first assessment stopped
at startup: the frozen observer data omitted the authoritative atom records
needed to identify residues. Its empty output directory and failure journal
are preserved. No physical trajectory was rerun or changed.

The separate recovery package
`runs/mobile-posterior-reference-recovery-20260921` preserves every original
reference script and data file and adds exactly the shape's declared coordinate
source, SHA-256
`41179e5291c675c6c572004b0753f821495148277dc3938e46cbf22a59b729e7`.
The unchanged reference helper verifies all 1,001 atom rows and 129 residue
labels, with zero coordinate discrepancy after recentering. The revised
observer explicitly binds the original and replacement observer identities;
it does not present this as a successful original automatic audit.
The completed [separate assessment](../runs/mobile-posterior-pilot-assessment-v2-20260921/analysis.json)
has SHA-256
`cd84eda4e29023891c9434278a79a69831769769780b4d65b09740a056c47fbd`.
Future freezes include this coordinate dependency and validate it before launch.

`tools/run_mobile_posterior_pilot.py freeze` requires explicit executable and
embedded-source-bundle paths, validates them against the reviewed tree, and
archives the observer's complete local import closure and native reference
data. `preflight` executes no kernels. `run` reserves the campaign once,
refuses existing outputs, drains started jobs after any failure and never
retries them. It invokes the archived observer only after all twelve physical
jobs finish successfully. A completed observer must identify all twelve
matching runs and the exact campaign/source hashes.

Validation comprises the five focused Rust configuration/integration and
spherical-runner checks, plus Python batches for preparation (5 tests),
controller failure handling and reference-data closure (13), observer arithmetic,
inverse maps and bound recovery (10),
and graph-history metrics (11). The [stage record](../runs/mobile-posterior-stage-validation-20260921/validation.json)
also binds the independently completed conditional contact repeat and its
comparison artifacts. That stage record predates the recovery checks.
These software checks do not establish physical
assembly or equilibrium.

## Observations and interpretation

The frozen analysis keeps native registration separate from nonspecific
exclusion contact. Native bonds require the existing rigid-tetramer motif
and shared external residue-patch criterion; intrinsic intratetramer contacts
never count. Entry and retention use the existing hysteresis thresholds.
A motif-label change on an already bonded body pair is distinct from
detachment. Native references affect this observation and the declared
proposal, never the target support or acceptance weight.

The analysis retains every update, self-loop, saved frame and initial bond.
It reports attachment/detachment and partner exchanges with body identities,
residence and censored intervals, native/nonspecific components, initialization
and half-record occupancy differences, and kernel-attributed events per full
sampling CPU. A constant graph descriptor has unresolved apparent ESS.
Graph-based decorrelation is not atomic-contact or crystal-registration
equilibration. The original shape and full physical bath remain unchanged
between arms.

## Completed pilot

The separate assessment passed all twelve runs: 120,000 attempted-update
records, 24,012 atomic frames, 23,848 full-mixture corrections, 10,833 posterior
Gaussian corrections, 1,302 posterior uniform checks and 1,059 explicit map
reconstructions. Sampling consumed 1,023.76 CPU seconds. The independent
observer checks stored bath-count arithmetic; it does not regenerate every
unrecorded bath point or acceptance uniform. The tiny-system Rust replay
separately tests that wiring, including non-anchor shielding.

![All twelve native and exclusion-contact histories](../runs/mobile-posterior-pilot-figure-20260921/mobile-posterior-assembly.png)

Replicate numbers below are one-based. A native connected component needs
only two edges, so connectivity alone does not certify a complete native
triangle or a consistent crystal lattice. Loss/return counts refer to native
body-pair edges; a registry label switch on a retained edge is counted separately.

| Preparation | Kernel | Replicate | First native-connected 3 (sweep) | Native edges lost / returned* | Final native edges | Sampling CPU s |
|---|---|---:|---:|---:|---:|---:|
| Dispersed | Capture | 1 | 84 | 0 / 0 | 3 | 88.07 |
| Dispersed | Capture | 2 | 195 | 0 / 0 | 3 | 88.28 |
| Dispersed | Redraw | 1 | 24 | 0 / 0 | 3 | 83.91 |
| Dispersed | Redraw | 2 | 13 | 0 / 0 | 3 | 92.59 |
| Dispersed | Transport | 1 | 165 | 0 / 0 | 3 | 86.12 |
| Dispersed | Transport | 2 | Not reached | 0 / 0 | 1 | 58.99 |
| Preassociated | Capture | 1 | Initial | 0 / 0 | 3 | 86.67 |
| Preassociated | Capture | 2 | Initial | 1 / 1 | 3 | 85.75 |
| Preassociated | Redraw | 1 | Initial | 1 / 0 | 2 | 85.54 |
| Preassociated | Redraw | 2 | Initial | 0 / 0 | 3 | 89.79 |
| Preassociated | Transport | 1 | Initial | 1 / 1 | 3 | 89.54 |
| Preassociated | Transport | 2 | Initial | 1 / 1 | 3 | 88.51 |

*The initial formation of edges from a dispersed start is not a return.

Five of six dispersed starts reached the native-connected three-body state
before the fixed burn at sweep 400. All six reached a three-body exclusion
contact component. The sixth retained one native edge and a second,
nonspecific edge through the end. All six preassociated preparations remained
native-connected. There were **zero exclusion-contact detachments** and zero
sequential or same-update partner exchanges across the entire campaign.
The four native edge losses therefore describe registration changes within
an aggregate, not release to solution. Three regained the same native pair;
fifteen additional retained-pair registry label changes were recorded.

GCA moved the whole preassociated aggregate on every attempt: all six such
runs had zero partial components. Common center shifts also leave relative
contacts unchanged. Thus collective spatial motion here does not establish
contact relaxation. Most postburn graphs are constant; the three finite
apparent joint-edge ESS values reflect rare registry changes and are not
evidence of equilibrium mixing. Zero half-record discrepancy in a constant
trace is likewise uninformative about convergence.

Both redraw and transport permit native-informed association. Two replicates
per preparation and no contact detachments cannot establish a mobile-system
efficiency ranking. The 1.92-fold conditional shoulder advantage remains a
separate result. The next comparison should include the observed competing
attachment as an independent preparation and resolve its relative physical
weight and exchanges before scaling up. A geometry-only atlas must remove
native-derived information from every constituent model; removing just the
50% native block would leave other native charts supplied.

This is a native-informed association and rearrangement test with mobile
neighbors. Even a persistent three-body native cluster would not establish
template-free assembly, equilibrium stability, a nucleation barrier, or a
physical attachment rate. A failed short trajectory would likewise not show
that the model excludes native assembly.
