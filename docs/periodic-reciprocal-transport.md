# Periodic reciprocal capture and posterior transport

This note gives the balance argument and validation requirements for extending
the frozen learned proposal to periodic boundaries. It does not change the
physical target, establish a sampling advantage, or open the assembly gate.
The open-space Gaussian charts remain the underlying charts; periodicity adds
a canonical image convention and explicit null outcomes.

## Canonical image and retained anchor

For box lengths `L`, define the half-open world-frame displacement cube
`D = product_i [-L_i/2, L_i/2)`. With retained anchor pose `(a,A)` and moving
pose `(x,R)`, use

`d = minimum_image(x-a,L)`, `t = A^T d`, `S = A^T R`.

The body-frame translation domain is `D_A = {t : A t belongs to D}`. It is a
rotated box, not generally an axis-aligned box. The anchor is fixed during the
elementary move and chosen uniformly among the other bodies; its selection
probability `1/(N-1)` cancels between forward and reverse paths.

Apply the existing open-space chart map, obtaining `(t',S')`. If `A t'` is
outside `D`, retain the old physical state as a recorded null. Do not redraw.
Otherwise export

`x' = wrap(a+A t',L)`, `R' = A S'`.

Canonicalization of this accepted endpoint recovers exactly the same relative
pose almost everywhere. The inverse map therefore uses the existing swapped
labels and inverse auxiliary noise. A crossing of the coordinate cell face is
allowed; it is the relative displacement leaving the unique image cube that
creates a null. Naively wrapping all open-space proposals would merge multiple
preimages and would require image sums or an additional image state.

## Restricted-kernel balance

Write `G_j(u)` for the full open-space Gaussian chart mixture density relative
to translation volume times normalized rotational Haar measure, using anchor
`j`. It includes ordinary and reciprocal virtual components with their original
weights. For the posterior Gaussian branch, the existing involutive construction
chooses source component `a` with probability `w_a g_a(u)/G_j(u)`, target
component `b` with probability `w_b`, and standard Gaussian auxiliary noise.
The inverse swaps `a,b` and uses the returned inverse noise. Its label,
auxiliary-density and physical-Jacobian factors give the correction

`log G_j(u_old) - log G_j(u_new)`.

Equivalently, the Gaussian proposal kernel `P_j` is reversible with respect to
`G_j(u) dμ(u)` before physical acceptance. Restricting it to `D_A` gives

`P_D(u,dv) = 1_DA(v) P_j(u,dv) + P_j(u,D_A^c) δ_u(dv)`, for `u` in `D_A`.

The off-diagonal reversible flow is simply multiplied by the symmetric factor
`1_DA(u) 1_DA(v)`. Null mass contributes only on the diagonal. Hence `P_D` is
reversible with respect to the truncated measure `1_DA G_j dμ`, without needing
its normalizer. Accepted physical moves use the same proposal correction with
the physical target ratio. This restriction does not replace or approximate
the existing many-body depletion gate.

For independent capture, the continuous proposal subdensity on the torus is

`Q_j(u) = eta/V + (1-eta) G_j(u)` at the unique image.

Its integral can be below one; the remaining Gaussian mass is the recorded
null probability. The MH correction is `log Q_j(old)-log Q_j(new)`, including
the complete uniform/Gaussian mixture. The posterior sampler instead selects
its uniform and Gaussian kernels separately with fixed probabilities: its
uniform torus branch has correction zero, and its Gaussian branch uses `G_j`
alone. A posterior correlation of zero therefore remains different from an
independent full-mixture capture proposal.

Proper anchor rotation and translational wrapping have unit absolute volume
Jacobian. The existing latent-coordinate and SO(3) Haar Jacobians are already
included in the map correction; no second angular or image-volume factor is
introduced. Reciprocal chart inversion `(t,S) -> (-S^T t,S^T)` preserves the
same physical measure. It need not preserve `D_A`: the final endpoint test and
null rule handle that restriction while retaining both virtual branches.

The argument conditions on all spectator poses. Complete periodic hard checks
and spectator exclusion unions are still necessary. It does not justify
replacing the environment with the selected anchor, filtering reciprocal
components by box fit, or adapting chart parameters during the move.

## New reference tests

`tests/periodic_transport.rs` contains independent small-system controls:

- All sixteen pairs of reciprocal virtual labels, at five correlations,
  recover the canonical pose and auxiliary noise across coordinate-cell
  crossings. Known diagonal Gaussian scales and the analytic Cayley Haar
  factor independently determine each extended physical Jacobian.
- A ninety-degree anchor rotation in an anisotropic cell distinguishes a
  body-frame-outside/world-frame-inside output from the reverse situation.
  Outside-image outcomes remain individual null attempts with their trace.
- The periodic uniform branch is checked against uniform coordinate moments
  and normalized Haar rotation moments over the whole cell.
- Integer-cell changes of the coordinate representatives and capture-center
  gauge preserve capture and posterior proposals under matched RNG streams.
- A two-Gaussian toy target is initialized independently using direct Gaussian
  draws conditioned on the unique image cube. Deterministic one-dimensional
  product quadrature gives its truncated moments and component masses. The
  complete uniform-plus-posterior transition is checked against that reference
  while retaining null and rejected states, at three correlations. The chosen
  reference loses approximately 41% of its open-space Gaussian mass; using
  untruncated moments would give a different mean.
- A separate clipped reciprocal-mixture target uses direct Gaussian draws,
  independent fair reciprocal branches and rotated-box rejection only for
  reference initialization. Paired translation, rotation and mixed observables
  check the complete transition at four correlations, retaining every null
  and rejected endpoint. This covers orientation-dependent reciprocal clipping.

All six tests passed. Six additional controls in
`tests/periodic_reciprocal_proposal.rs` check independently reconstructed complete
densities, analytic null mass, reciprocal/world-frame cutoff order, half-open
faces, chart seams, exclusion of lattice image sums and unchanged legacy random
streams. Three tests in `tests/periodic_posterior_assembly.rs` exercise the actual
all-mobile three-sphere runner: independent 27-image geometry, every retained
hard/depletion acceptance, reciprocal traces and anchors, null outcomes, cell
crossings, input provenance and exact checkpoint continuation. These use a toy
bath, not the protein production conditions.

The targeted integration suite contains 40 passing tests including the existing
open-space proposal, transport and restart regressions. The
[validation receipt](../runs/periodic-reciprocal-validation-20260924/receipt.json)
records source and dependency identities and the completed checks. Periodic
proposals still reject incompatible adaptive modes, and spherical GCA and center
shifts remain unavailable with periodic boundaries. No protein campaign has been
launched by this extension.

The runner now records a boundary-specific coordinate description. The observer
recognizes exactly the corrected periodic description and the old archived
periodic description; spherical descriptions remain strict. Its other geometry
and provenance checks are unchanged. Fresh source-bound observer definitions
are required for this version; archived definitions and results are preserved.

Half-open faces, roundoff near those faces and chart seams remain explicit
floating-point obligations beyond the measure-theoretic argument. These checks
support the implemented restriction and balance construction, not a formal
proof of floating-point execution or evidence of protein mixing or stability.
