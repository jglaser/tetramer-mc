# Configuration-dependent parameters around the calibrated map

Status: mathematical construction and implementation contract, 2026-09-25.
This note does not enable a new production mode. The existing calibrated
reciprocal atlas and current running jobs remain unchanged.

## Normalized target and exact reference

Retain the supplied atlas as fixed reference: component identities, chart
anchors, means mu0, full covariance factors B0, weights, reciprocal branch flags,
and defensive support. Let u be unconstrained parameter deviations, and declare

\[
 \Pi(X,u)=\pi(X)\rho(u\mid X),\qquad
 \rho(u\mid X)=\mathcal N(m(X),L(X)L(X)^T).
\]

The fit m and positive-definite factor L are deterministic functions of the
current configuration and fixed settings. Their normalizing constants are
explicit. Integrating u gives exactly the physical pi(X), regardless of fit
quality. This is a chosen normalized auxiliary law, not a claim to sample an
intractable likelihood posterior over physical basins.

Decode parameters relative to the calibrated components, for example

\[
 \mu_j=\mu_{0j}+B_{0j}a_j(u),\quad
 C_j=B_{0j}T_j(u)T_j(u)^TB_{0j}^T,\quad
 w_j=\operatorname{softmax}_j(\log w_{0j}+\delta_j(u)).
\]

Use a(0)=0, T(0)=I and delta(0)=0; return the original atlas object in that exact
case. This preserves its anisotropy and narrow registration widths. The first
stage can activate only covariance scales, keeping means, weights and component
count fixed. It need not replace the calibrated map or discover new components
at the same time.

For an oligomer-conditioned predictor, these decoder functions may additionally
depend on the invariant context C=(internal member poses, all external poses and
wall in the retained anchor frame). Require G(u=0,C)=G0 for every context. Hold
both u and C fixed within a rigid-subset proposal. A current contact fingerprint
or a crop around the moving handle is not invariant context.

## Learned physical move: retain the parameters

At a learned single-body or rigid-subset trial X->Y, retain u. This keeps the
actual chart map fixed, so the already validated posterior-source involution
and its reciprocal wrappers can be reused. With g the handle pose in the
retained external-anchor frame, use

\[
 \log R=\log R_{\rm depletion}
       +\log G_{u,C}(g_X)-\log G_{u,C}(g_Y)
       +\log\rho(u\mid Y)-\log\rho(u\mid X).
\]

The physical term is the existing exact conditional Poisson auxiliary factor,
with the same hard/wall zeros and no retries. Combine the deterministic
auxiliary-density ratio once, not once per carried body. The new fit at Y
is used to evaluate rho(u|Y), not to replace the held atlas halfway through the
involution. Selection of the handle and anchor remains as already validated;
the internal-rate clock argument applies to the joint target as well.

A static u=0 initially gives exactly the calibrated proposal. If rho depends on
X, the extra auxiliary acceptance factor can still change the trajectory.
Exact old trajectories require the disabled mode or a configuration-independent
auxiliary law held at reference, not just reference initialization.

## Model-independent moves: transport the parameters instead

For a local kernel, ordinary physical GCA, or center shift whose proposal does
not depend on u, retain the standardized residual

\[
 \eta=L(X)^{-1}[u-m(X)],\qquad
 u'=m(Y)+L(Y)\eta.
\]

For fixed X,Y the auxiliary Jacobian is det L(Y)/det L(X), and

\[
 \frac{\rho(u'\mid Y)}{\rho(u\mid X)}
 \left|\frac{\partial u'}{\partial u}\right|=1.
\]

Equivalently, in (X,eta) coordinates the joint target is pi(X)phi(eta), and the
physical kernel acts only on X. This preserves its original physical acceptance;
a rejection retains both X and u. For constant L, transport is simply
u'=u+m(Y)-m(X). Thus GCA need not acquire an auxiliary rejection penalty.

The two choices are compatible: learned moves hold u fixed and pay the explicit
conditional ratio; model-independent moves hold eta fixed and transport u.
They are different invariant kernels on the same joint ensemble. Their ordered
composition is invariant without needing the whole sweep to be reversible.
Do not silently substitute a changing-chart map into the fixed-u learned step.

## Reuse the existing stored residual representation

The implementation can retain the existing `AtlasTransportState.eta` rather than
introducing a second parameter bank. With u=m(X)+L(X)eta, the stored target is
pi(X)phi(eta). For a learned trial, reconstruct u once, hold it fixed in the
existing map, and propose

\[
 \eta'=L(Y)^{-1}[u-m(Y)].
\]

The additional log factor is

\[
 -\tfrac12(\|\eta'\|^2-\|\eta\|^2)
 +\log|\det L(X)|-\log|\det L(Y)|,
\]

which is exactly log rho(u|Y)-log rho(u|X). On rejection retain both old X and
old eta. On reversal the proposed Y,eta' reconstructs the identical u and map.
Model-independent moves instead retain eta and reconstruct u only after the
physical move. This is the same joint construction in different coordinates;
it reuses existing checkpoint state rather than storing redundant u and eta.
No derivative of m(X) with respect to X is required.

L must be invertible on each active, configuration-dependent parameter block.
Inactive constant blocks can be omitted. A zero-noise deterministic parameter
that changes with X cannot use the held-parameter bridge without a separate
argument; do not silently divide by zero or change its reverse chart.

## Parameter update at fixed configuration

At fixed X, Gibbs sampling is u'=m(X)+L(X)xi for xi~N(0,I). A persistent alternative
is the Gaussian reversible update

\[
 u'=m(X)+c[u-m(X)]+\sqrt{1-c^2}\,L(X)\xi,
 \qquad -1<c<1.
\]

It needs no Metropolis correction because it preserves the declared conditional
Gaussian exactly. This gives persistent auxiliary parameter state while retaining
tractable normalization. Arbitrary additions to a training archive are not
licensed by this update; any additional retained memory needs its own law.

## MSD information without a hard-particle Hessian

At fixed physical X, pi(X) has no dependence on proposal parameters. A physical
pose Hessian is a different object and is problematic at hard boundaries.
Proposal residuals supply a cheaper and explicitly defined scale signal.

For n deterministic observations assigned to a fixed reference chart, let
z_i=B0^{-1}(v_i-mu0), M=sum_i ||z_i||^2 and d=6. A shrinkage estimate of the scalar
covariance multiplier is

\[
 \widehat s^2=\frac{\kappa+M/d}{\kappa+n},\qquad
 m_{\log s}=\frac\gamma2\log\widehat s^2.
\]

Here kappa>0 shrinks toward the calibrated scale, gamma controls fit strength,
and empty observations give m=0. Reciprocal observations must first be expressed
in the appropriate base chart. Counts and squared residuals are derived from the
current state, not accepted-event frequencies or hidden historical caches.
An oligomer-conditioned fit must use whole-oligomer pose observations and context;
pooled single-tetramer residuals alone do not provide that conditioning.

For comparison, the Gaussian residual log likelihood as a function of u=log s is
ell(u)=-dn*u-M*exp(-2u)/2+constant, with curvature
ell''(u)=-2M*exp(-2u). This may guide a Gaussian auxiliary scale, but the chosen
rho must remain explicitly normalized. A Gaussian approximation to such a
parameter objective still defines an exact physical-marginal construction when
used as the declared rho. It is not an exact posterior claim. Conditional
variance can initially stay fixed to keep the transition simplest.

## Implementation stages and validation

1. Add parameter replacement preserving the same base-component identities and
   reciprocal envelope. Current `with_component_parameters` deliberately rejects
   reciprocal charts; merely removing that guard would lose branch semantics.
2. Introduce the held-u update using explicit u or the existing eta state, a
   reference decoder and normalized conditional-law evaluation. Initialize u=0 and refresh only after the initial reference stage.
   First enable small covariance-scale changes; preserve zero-change clone path.
3. Wrap learned events with the endpoint rho ratio; wrap model-independent events
   with deterministic auxiliary transport. Checkpoint the chosen retained coordinates (u or eta), decoder settings, fit
   settings, model hashes and all RNG continuation state.
4. Supply an oligomer-aware current-state fit/provider, preserving invariant
   context within the learned trial. Update sufficient statistics incrementally;
   the existing all-pairs fitter is O(N^2 K), unsuitable for unconditional full
   rebuilding after every production trial.
5. Test exact disabled-mode trajectories, reciprocal densities, physical and
   auxiliary stationarity, endpoint correction and no-correction negative
   controls, deterministic restart, and matched fixed/adaptive contact exchange
   and ESS per CPU. Mean, covariance, weight and component-count changes are
   separate controls. Parameter mobility alone is not evidence of better fluid
   sampling or native assembly.

The independent mathematical checker is
`research/validate_conditional_parameter_bridge.py`, with its result archived in
`runs/conditional-parameter-bridge-20260925/validation.json`. It checks reference
flow identities; it is not a Rust/Poisson/geometry or production-stationarity
validation.

Related: [reference atlas transport](atlas-transport.md),
[oligomer-conditioned proposal requirements](oligomer-conditioned-learning.md),
[posterior chart involution](posterior-chart-involution.md).
