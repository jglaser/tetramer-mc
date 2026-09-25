# Oligomer-conditioned proposal learning

Status: design and code audit, 2026-09-25. No production kernel or running job is
changed by this note. The current rigid-subset phase still uses a frozen
single-handle atlas.

## What the conditioning must contain

The labeled subset `S` identifies the carried bodies; its size alone is not a
sufficient description. Choose a handle `h` in `S` and a retained external anchor
`a` outside `S`. With body poses `g_i`, define

\[
 \xi_{S,h}=\{g_h^{-1}g_i:i\in S\},\qquad
 E_a=\{g_a^{-1}g_j:j\notin S\}.
\]

The proposal should model the six-dimensional whole-oligomer pose
`g = g_a^{-1}g_h`, conditional on `xi`, the spectator configuration `E_a`, and
the wall geometry in that same external-anchor frame. Under the carried move
`g_i' = g_h' g_h^{-1}g_i`, both `xi` and `E_a` remain unchanged. Labels, member
ordering, shape identity, bath parameters and the retained anchor are also fixed
within the attempt. The wall is fixed within this elementary kernel.

A normalized model `G(g | xi, E_a, wall; theta)` can therefore be reconstructed
from this context at the start of each attempt and used identically on reversal.
The existing posterior-source involution then retains the correction

\[
 \log R_{\rm map}=\log G(g_{\rm old}\mid C;\theta)
                  -\log G(g_{\rm new}\mid C;\theta).
\]

There is one collective pose and one correction, regardless of subset size.
The existing internal-rate phase construction still handles subset selection.
Full moving-union hard/depletion checks remain unchanged.

This argument fails if the fitter also uses the current handle pose, the current
external-contact fingerprint, or a neighbor list cropped around that moving
handle. Such inputs generally change on reversal. A crop defined solely from
the fixed context is admissible. Treating the currently measured contact pattern
as an immutable cluster class would specifically obstruct attachment/detachment.

## What to learn

A useful fit must represent cooperative feasible docking basins, their
translation/rotation covariance, and their relative proposal allocation. It
should include broad or unbound poses and competing adsorbed arrangements, not
only sharp native-like maxima. Native labels remain evaluation-only for a
geometry-only arm. Proposal weights need not equal physical basin probabilities;
acceptance corrects the approximation. Accepted-event frequencies alone are not
unbiased training estimates of equilibrium weights.

A finite normalized GMM is usable even if its fit is approximate: evaluate its
complete density and all coordinate Jacobians. Retain defensive support and
hard-invalid null proposals. Truncating a Gaussian to a hard-valid fraction
introduces a normalizer that the existing map does not account for.

The measured dimer diagnostic illustrates the objective: seven of nine valid
attachment trials map a broad source to a narrow target with covariance-volume
log ratio -33.97. Adding narrow destination components without representing the
source environments need not improve acceptance. A successful fit must support
both directions and improve completed exchanges or contact ESS per CPU.

## Online reconstruction and persistent learning are different

A deterministic finite-budget fit depending only on the invariant context is an
exact route to on-demand conditional proposals. Approximate fitting accuracy
affects efficiency, not balance, provided the returned normalized model is the
same on reversal. A fresh randomized construction also needs its construction
randomness in the auxiliary trace and a context-dependent law that cancels.
A cache may accelerate an identical reconstruction; replacing a deterministic
model with a history-dependent fit is a different algorithm.

For persistent basin memory, declare the retained auxiliary state `A` and a
normalized joint law. Two already established patterns are:

1. `Pi(X,A) = pi(X) rho(A)`, with reversible auxiliary exploration independent of
   production `X`. At fixed `A`, construct the conditional oligomer model and use
   the existing map. Auxiliary updates preserve `rho`; physical updates preserve
   `pi` at fixed `A`. The pair-memory implementation is an example, but its data
   contain no cooperative oligomer environment.
2. `Pi(X,K,eta) = pi(X) p_K(X) phi(eta)`, with normalized finite `p_K(X)` and a
   deterministic current-geometry parameter reconstruction. This is the existing
   full-GMM conditional closure. Retained-auxiliary physical updates require the
   count-law ratio; changing endpoint models also requires the appropriate
   reverse proposal. The fixed-chart involution formula cannot be transplanted
   unchanged when reconstruction changes its charts during the move.

An unnormalized physical docking law conditioned on `xi` is not an automatically
valid auxiliary law: its partition function can depend on `xi` and on `X`.
Likewise, unrestricted accumulation of production history between individually
correct frozen-model kernels does not establish invariance of the adaptive chain.
Any copy, replacement, compression, split or merge must have a declared update
law, or be confined to a training stage followed by frozen evaluation. Checkpoints
must retain all actual memory and its update state.

## Existing code and minimum integration contract

- `src/cluster_phase.rs` currently owns one `DockingProposal` and chooses the
  handle before calling it. A conditional provider needs `S`, `h`, `a` and the
  invariant context. Anchor choice must be exposed before model construction;
  currently `DockingProposal::propose` chooses the external anchor internally.
- Reuse `RigidSubset` for carrying members and the many-body physical gate.
  The conditional atlas remains a normalized distribution over one six-D pose.
- Preserve reciprocal chart branches. The active coverage atlas uses them;
  `simulation.rs`, `AtlasEngine::new` and the conditional importer currently
  reject adaptation with reciprocal models. Removing compatibility guards
  alone would not preserve that representation.
- Existing `auxiliary_transport`, `atlas_transport`, `conditional_closure`,
  `contact_memory` and RJ implementations provide validated pieces. They do not
  currently compose with `cluster_phase` or frozen posterior transport.

The first correctness test is equality of the constructed context/model at both
ends of a carried move that changes external contacts, followed by inverse trace
and full-density checks. Then compare fixed versus reconstructed proposals using
reference sphere/dumbbell systems and matched protein trials. Persistent memory
adds joint-stationarity and exact-checkpoint tests, including other elementary
kernels; old frozen-mode trajectories must remain unchanged when disabled.

Physical contact-weight evidence remains a separate test. A more efficient
conditional proposal cannot create thermodynamic stabilization absent from the
hard-shape/depletion model.

Related: [rigid subset phase](rigid-subset-phase.md),
[posterior transport](posterior-chart-involution.md),
[conditional closure](conditional-closure.md),
[reversible pair memory](contact-memory-balance.md), and
[measured dimer penalties](../runs/cluster-selection-diagnostic-20260925/report.md).
