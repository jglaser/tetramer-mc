# Retaining the calibrated involution with configuration-dependent parameters

Design derivation and implementation bridge, 2026-09-25. This is not yet a
production option; existing runs and sampler code remain unchanged.

The existing `atlas_transport` already supplies normalized current-geometry
parameter laws and reference-relative covariance updates, but its physical
proposal is not the current reciprocal posterior-chart cluster involution.
The bridge below preserves that involution exactly during each trial.

## Normalized conditional parameters

Keep the imported atlas as an immutable reference: component count/order,
anchors, means, full covariances, weights, reciprocal flags and defensive
support. Let `u` contain active unconstrained corrections (initially covariance
coordinates only; means and logits can follow). Define

\[
 u=m(X)+L(X)\eta,\quad \eta\sim N(0,I),\qquad
 \rho_X(u)=\frac{\phi(L_X^{-1}[u-m_X])}{|\det L_X|}.
\]

`L` is full rank on the active coordinate block. Then

\[
 \Pi(X,u)=\pi(X)\rho_X(u),\qquad
 \widetilde\Pi(X,\eta)=\pi(X)\phi(\eta).
\]

Both representations give exactly the original physical marginal. The choice
of normalized `rho` is a proposal-design decision; it need not be a Bayesian
posterior on the true physical fluid. No unknown configuration-dependent
partition function is introduced.

Decode covariance changes in the calibrated component's own units:

\[
 C_j(u)=B_{0j}T_j(u)T_j(u)^T B_{0j}^T,
 \qquad C_{0j}=B_{0j}B_{0j}^T,\quad T_j(0)=I.
\]

Log diagonal coordinates make `T` nonsingular. The existing bounded decoder
also controls extreme proposals without replacing narrow calibrated scales by
an absolute isotropic floor. Preserve reciprocal branch tying: both directions
of a base Gaussian share the same updated parameters.

## Physical transport: hold the realized map, transport the residual

Given `(X,eta)`, reconstruct `u`. Select the internally connected subset, handle,
external anchor and chart trace as before. **Keep u fixed** and apply the
existing involution, obtaining candidate `Y`. Compute

\[
 \eta'=L_Y^{-1}[u-m_Y].
\]

Starting from `(Y,eta')`, reconstruction gives precisely the same `u`, so the
inverse chart trace recovers `X`. The existing map correction remains
`log G_u(old) - log G_u(new)` for its retained anchor and context. Add

\[
 \Delta_{\rm aux}
 =\log\rho_Y(u)-\log\rho_X(u)
 =-\tfrac12(\|\eta'\|^2-\|\eta\|^2)
   +\log|\det L_X|-\log|\det L_Y|.
\]

For the current unbiased physical target the complete acceptance log is

\[
 \min\{0,\log W_{\rm depletion}+\Delta_{\rm map}+\Delta_{\rm aux}\}.
\]

The existing conditional Poisson construction supplies `W_depletion`; it is not
replaced by an exponentiated noisy overlap estimate. Existing assembly-bias
corrections, when supported, are additional declared factors. A rejection must
retain both the old physical state and the old residual. A numerical failure
must not trigger an unaccounted redraw.

The determinant above follows by changing coordinates `(X,eta)` to `(X,u)`,
applying the fixed-u map, and changing back. It is `|L_X|/|L_Y|` in addition to
the old chart/noise Jacobian. It is not a second covariance/Haar correction.
No derivative of `m` or `L` with respect to `X` is needed: the substitutions are
conditionally affine at fixed `X` and `Y`.

Simply holding `eta` while reconstructing different charts at `Y` does not give
the original deterministic map's inverse. A new reverse-density expression
alone cannot repair that construction. Holding `u` avoids this problem.

If the model also depends on the invariant oligomer context
`C=(xi_S, spectators in retained external-anchor coordinates, wall)`, hold that
context fixed within the trial as described in
[oligomer-conditioned-learning.md](oligomer-conditioned-learning.md).
If `m,L` themselves depend only on that invariant context, then `eta'=eta` and
`Delta_aux=0`. For a single persistent global residual, exact calibration across
all possible contexts requires either zero context-dependent shift initially
or a compatible common shift; separately cancelling each context would require
explicitly represented per-context auxiliary state.

## Auxiliary updates and the other physical kernels

At fixed `X`, an independent standard-normal redraw of `eta` is an exact Gibbs
update. A partial Gaussian refresh

\[
 \eta'=\sqrt{1-\beta^2}\eta+\beta\zeta,\quad \zeta\sim N(0,I),
\]

is reversible for the same conditional, retaining more auxiliary correlation.
Use a fixed `beta`, or one depending only on `X` held constant during that
auxiliary update; arbitrary eta-dependent step sizes require their own balance
correction. The resulting `u'` is a reversible parameter update given `X`.

A physical-only reversible kernel (ordinary local move, spherical GCA, center
shift) may instead hold **eta** fixed. Its target in these stored coordinates
is the product `pi(X) phi(eta)`, so it needs no added auxiliary rejection.
Reconstruct its implied parameters after moving. Thus covariance auxiliaries
need not destroy the rejection-free physical GCA. This differs deliberately
from the calibrated chart move, which holds `u` and transports `eta`.

The rigid-subset phase rates depend on internal contact geometry, unchanged by
the carried move. They do not depend on `eta`, so the same fixed-duration rate
argument applies on the augmented state space once each elementary map has the
joint-target acceptance above.

## What MSD or curvature can contribute

The physical distribution `pi(X)` is independent of the proposal parameters.
There is no physical Hessian with respect to `u` to optimize. In a locally smooth,
approximately Gaussian pose basin, covariance can approximate inverse curvature
of the pose free energy. Hard overlap boundaries are nonsmooth, so an exact
Hessian is not generally available for this model.

Local translation/rotation residual scatter can instead define `m(X)` and
uncertainty scales `L(X)`, with shrinkage toward the reference atlas. Because
`rho` is explicitly normalized, those fits can be approximate without changing
the marginal. They still need to improve proposal efficiency in practice.
Short-lag step MSD depends on the proposal scale, rejected moves and temporal
correlations; it is not automatically the basin covariance. Historical scatter
or a rolling training set must be retained in the auxiliary state with a valid
update law, or be used only during a training stage followed by freezing. The
current-geometry fit does not silently authorize irreversible history fitting.

## Controlled migration and validation

1. Preserve the calibrated atlas and fixed component count. Extend parameter
   reconstruction to retain reciprocal metadata instead of dropping it.
2. Introduce covariance-only active coordinates with declared `m,L`. Keep the
   old code path when disabled. With all gains/noises disabled, return the
   original model; do not invert a singular `L`.
3. For a global conditional fit, start with `u0=0` by storing
   `eta0=-L(X0)^{-1}m(X0)`. This recovers the calibrated atlas exactly. It is a
   prepared auxiliary state, not an equilibrium draw. Perform the first refresh
   after the first physical block, as in existing reference initialization.
4. Add the fixed-u/residual-transport wrapper to the existing chart trial. Keep
   local/GCA/shift kernels in the fixed-eta representation. Checkpoint all retained
   auxiliaries and named RNG streams.
5. Validate reference recovery and disabled-run identity; reciprocal density
   reconstruction; inverse recovery of both physical state and eta; the expanded
   Jacobian/conditional-density identity; joint equilibrium in analytic sphere
   and finite-state references; rejection rollback and exact continuation.
6. Compare fixed and adaptive models with matched starts/schedules, including
   dimer/trimer cooperative docking, full contact fingerprints and CPU cost.
   Better accepted-move throughput alone is not the objective.

The relevant existing implementation pieces are `src/atlas_transport.rs`,
`src/proposal.rs`, `src/docking.rs`, `src/cluster_phase.rs`, and
`src/rigid_subset.rs`. Current compatibility guards must remain until reciprocal
reconstruction and the augmented inverse/acceptance path are implemented and
validated; removing the guards alone is not the transition.
