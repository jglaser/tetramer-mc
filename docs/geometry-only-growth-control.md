# Geometry-only proposal control for four-body growth

The matched four-body control is complete. Both free starts make transient
exclusion contacts but neither reaches native registry in 2,000 sweeps. Both
preattached starts retain their initial interface. The original ABC triangle
retains registry throughout all four runs. The native-informed positive controls
reach native attachment in three of four free starts under the same physical
conditions. This demonstrates a limitation of the tested geometry proposal;
it does not establish that the physical model prevents growth.

The 96-chart geometry-only atlas has no supplied inter-tetramer docking geometry
in its centers or covariances. Native intratetramer structure remains supplied,
as does the initial ABC triangle. This is **seeded growth without native proposal
charts**, not de novo assembly.

## Completed result

| Initial D | Repeat | First exclusion contact | First native entry | Final D native monomer bonds | CPU s |
|---|---:|---:|---:|---:|---:|
| Free | 0 | Sweep 94 | Not observed | 0 | 109.94 |
| Free | 1 | Sweep 378 | Not observed | 0 | 110.49 |
| Preattached motif 8 | 0 | Initial | Initial | 6 | 146.09 |
| Preattached motif 8 | 1 | Initial | Initial | 6 | 140.48 |

The preattached controls retain motif 8, whereas all four native-informed
preattached controls move to motif 7. Neither retention nor the number of
monomer contacts establishes an equilibrium preference. Both initially free
geometry runs detach repeatedly from nonspecific contacts; there are no native
attachment events from which to infer native return rates.

The [twelve-run comparison](../runs/mobile-geometry-four-body-growth-comparison-20260921/report.md)
retains every run, proposal arm, start and accepted/rejected update. Its analysis
SHA256 is `d14d279e06b739448d320b82ef6cfd1251b510da319225f82c94e252a7ac2920`.
All 48,000 new move records passed the observer audit. All observed native
graphs admit consistent ideal catalogue transforms around their cycles; this
does not certify a complete crystal or equilibrium sampling.

![Matched geometry and native-informed controls](../runs/mobile-geometry-four-body-growth-comparison-20260921/geometry-growth.png)

The prepared reciprocal model SHA256 is
`e90a7c85c5bdb2a587071949f0ae6527dba080594229434080d0761e4125983b`.
The [preparation](../tools/prepare_geometry_reciprocal_proposal.py) verifies
shape-only generation and unchanged full covariances, and the
[controller](../tools/run_geometry_four_body_growth.py) binds the reviewed
assembly executable, input closures and independent seeds before launch.
Four preparation tests and seven controller tests passed, together with ten
four-body and twelve preceding three-body observer regressions. No assembly
kernel change was needed. The following sections retain the construction and
predeclared comparison specifications.

## Existing atlas and provenance

Use [geometry.json](../runs/free-tetramer-atlases/geometry.json) unchanged as the
base model. Identical bytes are archived as `provenance/model.json` in both
`runs/contact-memory-free-1000` and `runs/contact-memory-frozen-1000`.
The [generation record](../runs/free-tetramer-atlases/geometry.provenance.json)
contains arguments, atomic shape, all construction metadata, and the historical
source bundle. Its generator entry matches the current
[src/bin/contact_atlas.rs](../src/bin/contact_atlas.rs) exactly.

| Binding | SHA-256 |
|---|---|
| Geometry model | `1a39c8cc0d2977016d5dcd2072328de99612495eab76f6aac8a4aca5086e1519` |
| Generation provenance JSON | `0e18a3511b8e2c2e6ab9c79001e70d7d885263315353c2b5dd32ba5e1ee16161` |
| Embedded generator source, `src/bin/contact_atlas.rs` | `b2e08464bb4ec5e2688906e46c775d38e7c1a117b186676fd923e2ed372b010d` |
| Atomic shape, identical to current N4 control | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |
| Historical generator executable, recorded in provenance | `ec83a9e02fc8cc97b594f9c47eadce9177aa564e141aea36a102789ad80e49e5` |
| Historical generator source-bundle bytes, recorded in provenance | `5f3f455ab4ad551af3ababe9d916ad3f005dce0173f6735f00e549572d6de010` |

Seed `2026091907` generated independent Haar orientations and isotropic
separation rays. Each center is the last atomic-core exit along its ray plus
0.25 Å and a numerical outward guard. All 96 rays succeeded. Every latent mean
is zero and each component has equal weight. Covariance uses independent 1 Å
translation plus a 3° small-angle rotation about the atom-derived contact pivot;
its rolling cross terms come from that pivot, not a native contact Hessian.
The optional angular-length argument was **null**: the value
55.02283113084895 Å was computed from the shape RMS. No native atlas, docking
catalogue, registration label, depletion score or fitted native covariance was
an input. The stored covariances are positive definite; the smallest eigenvalue
across the 96 components is approximately 0.21844 in stored coordinates.

This inspection checked saved hashes, parameters and generator source; it did
not regenerate the atlas or rerun the historical generator executable. The
recorded historical executable and bundle hashes identify that construction,
while the next campaign must separately bind its actual reviewed executable.

## Exact reciprocal conversion and matched experiment

Create a fresh fail-closed `reciprocal-pose-mixture-v1` envelope with the **entire
unchanged geometry model** as `base_model` and exactly 96 true
`reciprocal_components` flags. This gives 96 stored Gaussians and 192 virtual
branches, each pair splitting its original component mass equally:

\[
G_{\rm sym}(T)=\tfrac12[G(T)+G(T^{-1})],\qquad
T^{-1}=(-R^Tt,R^T).
\]

Do not refit, resize, duplicate by linearized inverse covariance, or blend in
any of the native-informed 150/178 components. Full independent-capture density
and correlated source responsibilities must use the exact reciprocal law,
including the unchanged 10% laboratory-uniform floor. Preserve invalid proposal
rejections rather than conditioning the Gaussian draws on contact or entry.
The existing reciprocal implementation supplies these operations.

Match the [completed N4 control](mobile-four-body-growth-results.md): four
mobile bodies, the same two starts (ABC plus contact-free D; ABC plus initially
attached D), two independent fresh streams per start, 2,000 sweeps, burn 400,
and one saved frame per sweep. Preserve wall radius 223.32617672378387 Å,
rd=1.5 Å, z=0.035 Å⁻³, lambda/z=64, local scales 0.2 Å/1°, global probability
0.5, frozen posterior probability 0.5/correlation 0.9, and the same GCA and
center-shift schedule. Do not add memory, RJ, conditional fitting, capture/q
constraints or fixed bodies. Proposal model and stream seeds are the intended
changes. Keep every start and failed run; freeze the allocation before launch.

The source starts are
[recommended-poses.json](../runs/mobile-four-body-design-20260921/recommended-poses.json),
SHA `994d860f28d0fe1e94fefff5ee6b818d524210153d0cae6ebc4f492ecdc53dd2`.
The completed positive-control
[protocol](../runs/mobile-four-body-growth-campaign-20260921/protocol.json) has
SHA `1a07fce8ac75e787f8ba778db5584968fe6bf912b72b34b0c9de2294db4f8d75`.
Retain its original and coverage results as separately identified positive
controls; a new geometry arm must not be relabelled as either old model.
The four-body observer now explicitly supports the geometry campaign schema
while retaining its original/coverage contracts. Its schema and provenance
extension was tested before this arm was frozen. The assembly kernels are
unchanged.

Report strict entry versus retained registry, actual exclusion detachment and
return, attachment kernel, ABC survival, competing contacts, and source/parity
proposal statistics with full sampler CPU. Different component counts change
density-evaluation cost. Short one-way attachment or failure to attach cannot
establish equilibrium, independent-contact efficiency, or a thermodynamic
absence of native assembly. The outermost-contact atlas also need not represent
deep interlocking pockets. Earlier geometry-only failures in
[rj-assembly.md](rj-assembly.md) used different auxiliary proposals and are not a
result for this reciprocal N4 control.

The current coverage atlas does not qualify: it retains native-derived charts,
adds prescribed catalogue sites, and selects some centers using the no-native-
entry label. Removing explicit labels after fitting, or extracting new centers
from the current native-informed trajectories, does not remove that provenance.

## Frozen native-blind pair-memory proposal

The all-slot baseline is now prepared as a separate, inert proposal asset in
[runs/native-blind-memory-proposal-preparation-20260921](../runs/native-blind-memory-proposal-preparation-20260921/manifest.json).
The self-contained `model.json` has SHA
`b6d06b0a076d7f3cd4f591d116a77799b2dc4caa3ea6c21081ad5da8a45b3e9b`.
It retains all 64 terminal slots in replicate/slot order, including duplicates
and any unbound poses, with equal weights and exact reciprocal symmetrization
(128 virtual branches). No native label, fitted covariance, likelihood,
production pose or source weight selected or altered any slot.

The prescribed ordinary Gaussian widths are 1 Å translation and 3° small-angle
rotation, with no cross terms. Their full-rank covariance uses the atom-derived
angular coordinate scale 55.02283113084892 Å. These widths are a baseline, not a
fit or demonstrated coverage optimum. Retain the 10% uniform proposal component
when evaluating it with the existing assembly runner.

The [exporter](../tools/prepare_native_blind_memory_proposal.py) archives complete
historical checkpoints, input configurations, source bundle, executable,
geometry model and its own Python dependency closure. All 64 centers passed an
independent atomic hard-overlap check. Four tests check every slot, ordering,
equal weights, independence from production poses, retention of duplicates,
failures instead of filtering, and the exact reciprocal law. The reciprocal
density identity and symmetry errors were below 8e-14; all 128 chart round trips
were below 5e-14. No new physical sampling or depletant clouds were generated,
and no assembly production run uses this asset yet. Contact-weight convergence
remains the gate for the planned controlled assembly comparison.

### Historical source and fixed selection rule

A separate later proposal can use all 64 terminal poses from the four historical
`runs/contact-memory-free-1000/runs/free-r0{0,1,2,3}-rj-memory-on/checkpoint.json`
files, in replicate/slot order. Each contains 16 slots at sweep 1000:

| Replicate | Checkpoint SHA-256 |
|---|---|
| 00 | `c01fa74db7e23604a2c9f81802a2240ab2f23011b735b28c5a738fe2cd4e689a` |
| 01 | `ff4e22386d523fa111a1989e1bbc3fa3395d7ae3ef1ceae1904eda7de5cf5a98` |
| 02 | `91ebc9029cb295a7948bba54ca2aff51dc3ce7cc8c562c68ba5ccb10de9dddb7` |
| 03 | `64b54beef1df47269f92b7e9aadef28377ab17274e98ffc395725e9896eefe3c` |

Nondependence on production configurations was checked in the **historical
embedded source**, not inferred solely from today's code. The r00-on
`provenance/source-bundle.json` hash is
`da9aaeb5401113b4cbaba5c0c6cfa17b3d0ddb1c5d49470daef1ed6654381746`;
its exact bytes are embedded in archived executable
`runs/contact-memory-free-1000/provenance/tetramer-mc`, SHA
`9ea570c1918321f90781856a0e4dd7d484dd19ce6b1542187b081d3c7a9e50e5`.
The bundle's `src/contact_memory.rs` is
`03ac0571dee123e3b5cabd07c25f16e68ea966f889e32dcb562a3e51152f6d7f`;
`src/simulation.rs` is
`11ef30f90cd76e0c3011e49cb493682a742b0db6d1f89292643b0f1e08677499`.
Their stored text hashes were checked. The simulation uses a named
`contact-memory` stream and calls `state.update(&tree, m, endpoint_gate, rng)`.
The update receives no production poses or native catalogue, constructs one
isolated pair environment, and excludes other slots and production particles
from its spectator geometry. Initialization is shape-only radial contact.
The bank bath is rd=1.5 Å/z=0.035 Å⁻³, with lambda/z=16 and a fixed separation
ball of radius 102.22404380707819 Å. This is a **native-blind, physically learned**
source, rather than a purely geometric source or an equilibrium sample of N4.

If this branch is pursued, use every terminal slot without native-label,
cloud-weight or production-success filtering. Prescribe equal weights and
full-rank geometric translation/Cayley widths before any new production; do not
borrow native covariance. Freeze those ordinary charts before applying exact
reciprocal symmetrization. Treat training cost and historical nonequilibrium
bank coverage separately, as described in
[contact-memory-pilot.md](contact-memory-pilot.md) and
[contact-memory-balance.md](contact-memory-balance.md). The export described above uses this complete source. It performs no fit or
new sampling, and has not launched a production assembly comparison.
