# Spherical GCA and reversible learned proposals

The existing spherical cluster implementation is in the research project at
`protein-nucleation/src/spherical_depletion.rs`. Its target confines proteins
with an exact hard spherical wall; the homogeneous ideal depletant bath
permeates that wall. A state-independent, centered **proper half-turn** supplies
the common isometry. Hard cross-overlaps and conditional Poisson hyperedges
recruit components, followed by independent fair component flips. Every owner
of a shared exclusion point participates, so depletion is many-body.

This is a physical, model-independent GCA kernel G reversible with respect to

\[
\pi_R(X)\propto H_R(X)H_{\mathrm{pairs}}(X)
\exp[-z|\cup_i E_i(X)|].
\]

The following derives the composition and lifting. Spherical GCA, center shifts
and a fixed-K Gaussian-mean auxiliary lift are now implemented in Python and
Rust; see [implementation and validation](spherical-ensemble.md). Periodic
production remains available as a separate boundary choice.

## Frozen learned proposal

Adapt the learned kernel S to the same spherical target: use ordinary relative
displacements, a normalized uniform support proposal on a fixed enclosing
position domain with Haar orientations, and exact atomic wall rejection.
Learned draws outside the allowed domain are rejected without retrying or
renormalizing by an unknown accessible volume. The periodic minimum-image
wrapper cannot be reused unchanged.

If G and S are individually reversible for π_R, a constant-probability mixture
`aG + (1−a)S` is reversible. Fixed alternation also preserves π_R but need not
itself satisfy detailed balance. Choose the algorithm schedule independently
of the current state unless its selection factor is explicitly corrected.

G already includes the depletion bath in its cluster law. Do not apply a second
single-body Poisson acceptance gate to its output. Each kernel refreshes the
auxiliary cloud prescribed by its own valid construction.

## A persistent auxiliary mixture law

For a normalized law ρ(dθ|X), require

\[
\widetilde\pi(dX,d\theta)=\pi_R(dX)\rho(d\theta\mid X).
\]

An arbitrary π_R-invariant operation on X alone need not preserve this joint
law. If θ is held fixed and Y is proposed using the reversible physical GCA,
the additional acceptance probability is

\[
\alpha=\min\{1,\rho(\theta\mid Y)/\rho(\theta\mid X)\}.
\]

The physical π_R ratio has already canceled against the reverse GCA transition.
For a product law ρ(θ) independent of X, this factor is one.

## Gaussian transport can retain rejection-free GCA

Write θ=(k,v), where v holds the continuous mixture coordinates. Define

\[
\rho(k,v\mid X)=p_k\,
\frac{\varphi(\eta)}{|\det L_k(X)|},\qquad
\eta=L_k(X)^{-1}[v-F_k(X)],
\]

with normalized, **X-independent** p_k and nonsingular L_k(X). In coordinates
(X,k,η), the target is simply `π_R(X) p_k φ(η)`.

Execute a complete model-independent physical GCA update X→Y, retain (k,η),
and reconstruct

\[
v'=F_k(Y)+L_k(Y)L_k(X)^{-1}[v-F_k(X)].
\]

The Gaussian conditional ratio and parameter-map Jacobian cancel exactly:

\[
\frac{\rho(k,v'\mid Y)}{\rho(k,v\mid X)}
\left|\frac{\partial v'}{\partial v}\right|=1.
\]

Thus GCA remains rejection-free on the joint ensemble. Its reversibility is
immediate in (X,k,η): G acts only on X and preserves π_R. The fitter need not be
invertible in X or differentiable; it must define the same reproducible F_k and
L_k whenever the same state is evaluated. Invertibility is needed in η. Mixture
parameters can be reconstructed lazily before the next learned move.

If the count law is p(k|X), retaining k leaves the factor p(k|Y)/p(k|X), which
requires a correction. Birth/death updates can instead occur separately with
the documented RJ rule. The learned physical-transport kernel must use the
transported model in its reverse proposal density.

The simple cancellation above requires G to be independent of the model and
latent variables. Using the mixture to choose the GCA axis or component changes
the forward/reverse probabilities; it is a different, biased construction
requiring its own selection correction.

## An exact redraw alternative

For any tractable normalized conditional ρ, execute Y~G(X,·) and draw
θ′~ρ(·|Y) exactly. The joint flux is

\[
\pi_R(dX)\rho(d\theta\mid X)G(X,dY)\rho(d\theta'\mid Y),
\]

which is symmetric by physical GCA detailed balance. There is no extra rejection,
even for an X-dependent count law. An approximate posterior fit or an arbitrary
number of RJ steps is not an exact conditional redraw. A few RJ steps can still
be used as separate joint-target-preserving updates, but cannot replace this
redraw argument without accounting for their actual kernel.

## What the combined sampler would test

Spherical GCA can carry a recruited oligomer without breaking its internal
contacts. Learned moves may then change registration at encounters. This
addresses collective mobility but does not guarantee the narrow pose required
for an oligomer merger. Strong depletion may still recruit one large component,
whose whole-system rotation changes no internal configuration.

Centered rotations alone preserve each body's radius and body-frame radial
vector. Ordinary local and learned translations/rotations break that obstruction;
the earlier global center-shift move can be included under the same target.

Compare learned+local against learned+local+GCA in the **same spherical domain**,
with identical physical parameters and equal computational budgets. Record
inter-oligomer relative displacement/rotation, native bond formation between
previously separate oligomers, component-size distributions, and environment
round trips per CPU second. Separate global rotations and rigid oligomer
transport from changes in internal order. Compare frozen versus transported
auxiliary mixtures only after this physical collective-move control.

Neither this invariance argument nor rejection-free cluster flips prove
ergodicity, rapid mixing, or crystal stability. Spherical and periodic ensembles
also have different finite-size boundary effects; they are not interchangeable.
