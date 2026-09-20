# Initializing the normalized conditional GMM from a learned proposal

A previously learned Gaussian proposal can supply the **initial auxiliary
state** of the normalized conditional sampler. It need not become another
physical interaction or an unnormalized likelihood factor. Exact recovery of
that proposal requires compatible coordinate charts and parameters inside the
conditional model's representable range. This note specifies that version and
the alternatives when charts differ.

## Initial state and invariant law

Let the current-state contact data be `D(X)`, and write the conditional closure
in unconstrained coordinates as

\[
 K\sim p(K\mid X),\qquad \eta\sim\mathcal N(0,I_{d_K}),\qquad
 v=F_K(D(X))+L_K(D(X))\eta,\qquad
 d_0=0,\quad d_K=28K-1\ (K\geq1).
\]

The decoder converts `v` to `K` Gaussian means, positive-definite covariances,
and positive normalized weights. The joint target in stored coordinates is

\[
 \widetilde\pi(X,K,\eta)=\pi(X)\,p(K\mid X)\,\phi_{d_K}(\eta).
\]

Fix a physical starting configuration `X₀`. If a validated learned model has
`K₀` compatible components, encode them as `v_import` and set

\[
 \boxed{\eta_0=L_{K_0}(D(X_0))^{-1}
       \left[v_{\rm import}-F_{K_0}(D(X_0))\right].}
\]

Reconstruction at `X₀` then recovers the imported parameters. The prototype's
scalar `residual_scale = ρ` reduces the solve to `(v_import − F)/ρ`. A general
non-diagonal `L` requires a linear solve. No determinant or likelihood weight
is added to the physical target by this initialization.

`K₀` must belong to the configured supported count range `0,…,k_max`.
`K=0` has an empty residual vector and a purely uniform proposal. A nonempty
imported Gaussian model starts at `K₀≥1`. The direct import
described here retains the component count and a declared correspondence
between components and slots. Reordering entire components preserves their
mixture density, but charts and every parameter block must be permuted
consistently. Merging, dropping, or refitting distinct components generally
produces a different initial proposal. A pure covariance map without
means, weights, and chart definitions does not identify a complete GMM; supply
these explicitly or describe the reconstruction as an approximation.

This is a prepared initial state, not an equilibrium conditional draw. A large
`η₀` is allowed when finite but can take time to relax. Recovering an old proposal
does not imply an equilibrated physical configuration or a mixing improvement.
The learned input is consulted once to choose `(K₀,η₀)`; future fits use only the
current `D(X)` and the fixed closure definition.

## Exact parameter conversion

The proposal export uses the anchor-body-relative chart

\[
 z_j(t,R)=\big(t-a_j,\;\ell\,c(RA_j^T)\big),
\]

where `(a_j,A_j)` is the component chart anchor, `c` is the Cayley coordinate,
and `ℓ` is `angular_length`. The learned Gaussian has mean `μ_j` and full
covariance `C_j` in this six-dimensional coordinate. The closure uses the
dimensionless coordinate `z_j / τ`, with `τ = translation_scale_A` in the
closure configuration. Thus import
uses

\[
 \bar\mu_j=\mu_j/\tau,\quad
 \bar C_j=C_j/\tau^2,\quad
 \epsilon=\texttt{covariance_floor}/16,\quad
 B_j=\operatorname{chol}(\bar C_j-\epsilon I),\quad B_{j,ii}>0.
\]

The decoder defines `C_j = τ²(B_j B_jᵀ + εI)`. Its fixed ridge is part of the
declared parameter map; import subtracts that same ridge before encoding and
recovers the original covariance. Both `C_j` and `C_j/τ²−εI` must be strictly
positive definite. All 21 independent covariance parameters are retained,
including coupling between translation and rotation. Cholesky factorization
must succeed without silently adding extra jitter. Gaussian `scales` are covariance matrices in the
existing export; a Student-t scale matrix is not a Gaussian covariance import.

The ordinary unbounded representation would pack the six means, the 21 lower
Cholesky entries with logged positive diagonal, and `K−1` reference logits
`log(w_j/w_K)`. The implemented closure additionally bounds these physical
parameters with smooth maps. Its unconstrained `v` therefore needs the inverse
of those maps:

\[
 \begin{aligned}
 v_{\mu_{j,i}}&=64\operatorname{atanh}(\bar\mu_{j,i}/64),\\
 v_{B_{j,ii}}&=h\operatorname{atanh}
     \left[(\log B_{j,ii}-m)/h\right],\\
 v_{B_{j,ik}}&=b\operatorname{atanh}(B_{j,ik}/b),\quad i>k,\\
 v_{w_j}&=24\operatorname{atanh}\left[\log(w_j/w_K)/24\right],\quad j<K.
 \end{aligned}
\]

Here `σ_low = sqrt(covariance_floor)/2`,
`σ_high = 2 sqrt(covariance_ceiling)`,
`m = (log σ_low + log σ_high)/2`,
`h = (log σ_high − log σ_low)/2`, and `b = σ_high`. These are bounds on Cholesky
coordinates; they should not be confused with eigenvalue bounds on every
decoded covariance. The eigenvalue clipping used by the deterministic fitter
and the coordinate bounds are separate operations. The scaled map
`B_b(v)=b tanh(v/b)` has unit slope at zero, so the Gaussian residual scale is
expressed in these parameter coordinates rather than multiplied by the bound.

For each component, pack six mean entries followed by the 21 Cholesky entries
in lower-triangle row order; append the `K−1` weight coordinates after all
component blocks. For `K=1` there is no weight coordinate. Every inverse-map
argument must be strictly between `−1` and `1`. A model on or beyond a bound
cannot be imported exactly with finite `v`. Reject it with a component and
coordinate diagnostic, or explicitly select and record a modified closure or
an approximate projection. Silent clipping would invalidate the exact-recovery
claim.

The library hook is `ConditionalEngine::state_from_model(&fit, &model)`, followed
by `engine.model(&fit, &state)` for reconstruction. It verifies the source
shape digest, open-space convention, angular metric, defensive cube and uniform
weight, matching fitted chart anchors, and parameter bounds. The lower-level
`state_from_components(&fit, &components)` accepts raw component parameters;
these lack shape and angular-metric metadata, so those checks belong to its
caller. Obtain such parameters with
`FrozenRelativePoseProposal::component_parameters()` and perform any exact
metric conversion explicitly. These hooks are not a command-line importer
for arbitrary existing atlases.

Recovery concerns the learned component mixture `G`. The complete physical
proposal also uses the closure's configured uniform branch, enclosing cube,
and open-space/null-move convention. Reproducing a previous complete proposal
requires those settings to match as well; a periodic wrapper is not the same
proposal as the spherical one.

Validate finite values, the physical shape digest, proper anchor rotations,
the relative-pose convention, and the chart/component ordering. Weights must
be positive and normalized. When an upstream format provides positive
unnormalized masses, normalize them explicitly before creating the existing
proposal-format input; the existing loader expects an already normalized
weight vector. Exact zero weights cannot be represented by finite logits.

## Chart compatibility is a real restriction

The formula above is exact when each imported component and its closure slot
use the same chart and metric. Some differences admit an exact affine repair:

- With equal anchor rotations, changing only a translation anchor gives
  `μ′_translation = μ_translation + a_old − a_new`; the covariance is unchanged.
- With unchanged charts and different angular lengths, set
  `S = diag(1,1,1,ℓ_new/ℓ_old,ℓ_new/ℓ_old,ℓ_new/ℓ_old)`,
  then `μ′ = S μ` and `C′ = S C Sᵀ`. The existing
  `tools/blend_contact_atlas.py` applies this metric conversion. The Gaussian
  coordinate determinant cancels the changed angular-to-Haar density factor.
- A general verified affine coordinate change `z′ = A z + b` gives
  `μ′ = A μ + b` and `C′ = A C Aᵀ`, provided the associated chart Jacobian is
  evaluated consistently.

Changing a rotational Cayley anchor is generally nonlinear. Matching a mean
pose and transporting its covariance with a local Jacobian gives an
approximation, not the same Gaussian density. A full Gaussian pushed through
that chart change is generally non-Gaussian. Therefore an exact shared-chart
import must reject incompatible rotational anchors. Equal physical means are
insufficient evidence of compatibility.

If the current-state fitter chooses its own anchors, an arbitrary existing
atlas will often fail this compatibility test. Three clearly distinct versions
are available:

1. **Initial-state import into the existing closure.** Keep its chart rule.
   Import only compatible components, or explicitly refit the learned law in
   those charts to obtain an approximate initial proposal. The result changes
   the starting state, not the specified conditional law.
2. **A closure with immutable imported chart context.** Before sampling, fix
   the imported chart metadata `A`, define `F_K(D(X);A)` in these charts, and
   use the same normalized conditional construction. At each supported count,
   declare a deterministic slot-to-chart rule, or give any additional discrete
   chart labels an explicit normalized law and reversible updates. Exact
   recovery at `K₀` is possible. The charts remain part of the closure definition,
   even after the imported continuous parameters have relaxed. This changes
   the auxiliary law/context, while its normalized physical marginal is still
   `π(X)`. It is not strictly an initial-state-only use of the atlas.
3. **A fixed proposal branch or permanent fitting reference.** An immutable
   learned mixture can remain in an explicitly evaluated mixture proposal, or
   serve as a fixed regularization target/seed inside every evaluation of
   `F_K(D(X);A)`. Both require a documented fixed rule. The proposal branch must
   appear in the whole forward and reverse density. A fitting reference alters
   the conditional law and possibly its count scores. Neither variant should
   be described as merely initializing the production state.

An arbitrary chart conversion may also be kept exactly as a non-Gaussian
pushforward with its full Jacobian. That would extend the implemented
Gaussian-component family and its evaluator; copying a covariance does not
implement it.

## Corrections after import

At a physical proposal `X→Y`, retain `(K,η)` and reconstruct the model from the
fit at **each** endpoint. The acceptance includes

\[
 \log q_{Y,K,\eta}(X)-\log q_{X,K,\eta}(Y)
 +\log p(K\mid Y)-\log p(K\mid X),
\]

in addition to the existing exact physical/Poisson correction. The transported
reverse model is required even if the initial forward model came from a learned
file. The continuous conditional-density ratio and transport Jacobian cancel
in stored `η` coordinates. The count probability does not cancel when it
depends on `X`.

For count scores `s_K(X)`, use
`log p(K|X) = s_K(X) − logsumexp_J s_J(X)` over the entire configured count
support at each endpoint. Keeping only a difference of unnormalized scores is
incorrect. The same rule applies if an immutable fitting reference affects
these scores.

A model-independent, physical reversible GCA update retaining `(K,η)` requires
the extra factor `min(1,p(K|Y)/p(K|X))` for a state-dependent count law. An exact
conditional redraw of both the count and residuals after a physical GCA update
is a different valid construction; a few approximate RJ updates cannot be
substituted for an exact redraw. See [the composition derivation](gca-composition.md).
The same fixed-residual count correction applies to other physical kernels;
for transformations that preserve `D(X)` and all count scores it is exactly one.

Within fixed physical `X`, residual refresh, count moves, or other model moves
must preserve the declared conditional law. Initial import does not authorize
subsequent uncorrected covariance learning, pruning, or trajectory accumulation.
Restart checkpoints store the realized `(K,η)` and the exact closure settings;
if a fixed imported context is used, serialize it or pin and verify its content
digest. Do not reinitialize from the learned file on resume.

## Scheduling and recovery checks

An imported state should survive until its first intended physical proposal.
Refreshing `η` from its standard-normal law immediately before that proposal
erases the supplied means, covariances, and weights. A production integration
can use a fixed schedule that performs its first physical block from the
imported state, then resumes the ordinary conditional updates. A checkpoint
written before that block must preserve whether the initial block is pending.
For a library initializer, this scheduling remains the caller's responsibility.

The older mean-only `auxiliary_transport` mode can import only compatible means
while preserving its fixed chart covariances and weights. With base covariance
factor `B_j`, its exact initialization is

\[
 \eta_{0,j}=\left[B_j^{-1}(\mu_{\rm import,j}-\mu_{\rm base,j})
                    -f_j(X_0)\right]/s_j(X_0).
\]

It cannot reproduce an arbitrary learned full-parameter GMM by changing `η`
alone. Its existing per-sweep residual redraw must also be considered before
calling an initial-state import useful.

The independent, standard-library check can be run with:

```bash
python3 research/validate_conditional_initialization.py
```

It tests the bounded-coordinate inverse, recovery of correlated full covariance
matrices and nonuniform weights, a non-diagonal initialization/transport solve,
physical density preservation under angular metric conversion, rejection of
invalid inverse coordinates and singular covariance, and a concrete
non-Gaussian rotational recharting example. Its 60 initialization cases give
maximum mixture log-density error `1.43e−14`. These are mathematical
construction checks, not evidence of physical equilibration.

`cargo test --locked --test conditional_initialization` tests the actual Rust
library import hook: full learned and physical proposal density recovery,
nonuniform weights, correlated covariances, retained residuals after a new fit,
and rejection of incompatible charts, source metadata, invalid covariances,
and out-of-bound coordinates. A future runner import option additionally needs
first-proposal and exact-restart tests.
