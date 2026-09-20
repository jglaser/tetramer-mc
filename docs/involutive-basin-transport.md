# Involutive transport between Gaussian pose basins

An explicit inverse map can structurally prevent a proposed transition from
having a missing reverse construction. It does **not** generally make ordinary
pose-space proposal probabilities equal: unequal basin volumes and auxiliary
probabilities still enter acceptance. The minimum useful control is a fixed
Gaussian-chart map whose inverse trace and complete Jacobian are known.

This construction uses the existing physical target, spherical boundary,
retained active mask `A`, and retained atlas latent state `eta`. It changes the
proposal kernel, not the physical interactions. Native information in supplied
reference charts remains supplied native information.

## Gaussian charts and their physical volume

Hold a moving body label `i` and a distinct spectator anchor `j` fixed during a
trial. Express the moving body's relative pose as `(t,R)` in the anchor's body
frame. For chart `a`, retain its reference position `t_a`, proper rotation
`R_a`, six-vector mean `mu_a`, lower Cholesky factor `L_a`, and angular length
`ell`. Define the chart map `G_a` by

\[
 v_a=\mu_a+L_a z,\qquad
 t=t_a+(v_a)_{1:3},\qquad
 c=(v_a)_{4:6}/\ell,\qquad
 R=\operatorname{Cayley}(c)R_a.
\]

The inverse uses a triangular solve after the inverse Cayley transform. Each
chart covers all translations and all rotations except its measure-zero pi
seam. The map to laboratory coordinates through the fixed spectator is a
proper rigid transformation with unit volume factor.

Relative to `d^3 t` and normalized Haar measure on `SO(3)`, the chart volume is

\[
 dx=J_a(z)\,d^6z,\qquad
 J_a(z)=\frac{\det L_a}{\ell^3\pi^2(1+\lVert c\rVert^2)^2}.
\]

This matches the Haar convention in the existing proposal density evaluator.
Its common constant cancels from all chart-to-chart ratios. In particular,
`q_a(x)=phi(z)/J_a(z)` for the chart's normalized Gaussian density.

## A reversible trace with adjustable latent perturbation

Choose an ordered chart pair `(a,b)` from the active mask with a probability
`r(a,b)` satisfying `r(a,b)=r(b,a)`, independently of the moving pose. Uniform
selection of distinct labels is one option. The one-label case can use `a=b`;
the empty case must explicitly fall back to another valid kernel or make a
null move. Retain these labels through the trial.

Choose a fixed correlation `gamma` in `[0,1]`, let
`s=sqrt(1-gamma^2)`, and draw `u ~ N(0,I_6)`. Encode `z=G_a^{-1}(x)` and apply

\[
 \begin{pmatrix}z'\\u'\end{pmatrix}
 =
 \begin{pmatrix}\gamma I&sI\\sI&-\gamma I\end{pmatrix}
 \begin{pmatrix}z\\u\end{pmatrix},\qquad
 y=G_b(z').
\]

The reverse trace is `(b,a,u')`. The displayed matrix is orthogonal and squares
to the identity. Reapplying the map with the reverse trace therefore recovers
`x,u`. The determinant of the **full extended-state map** in physical-pose and
auxiliary measures is

\[
 |\det DT|=\frac{J_b(z')}{J_a(z)}.
\]

This is not the derivative of the proposed pose with respect to the old pose
while holding noise fixed. That conditional derivative has an additional
`|gamma|^6` factor and is singular at `gamma=0`, whereas the full extended map
remains invertible.

At `gamma=1`, transport preserves the complete standardized residual:
`z'=z`, `u'=-u`. It directly maps corresponding locations in two basins.
At `gamma=0`, it exchanges the old latent coordinate with a fresh Gaussian:
`z'=u`, `u'=z`. Intermediate values retain partial within-basin information.

## Acceptance with the unchanged physical target

The trial extends the target with normalized label and noise variables:

\[
 \widetilde\Pi(X,A,\eta,a,b,u)
 =\pi(X)\rho(A)\phi(\eta)\,r(a,b)\phi(u).
\]

The physical move retains `A,eta` and the spectator. For the symmetric static
pair law, its nonphysical log acceptance correction is

\[
 \Delta_{\rm map}
 =\frac12\left(\lVert u\rVert^2-\lVert u'\rVert^2\right)
   +\log J_b(z')-\log J_a(z).
\]

Orthogonality also gives the useful identity
`Delta_map = log q_a(x) - log q_b(y)` for every correlation. These are the two
individual chart densities, not full-mixture densities. This supplies an
independent check on the Jacobian and auxiliary bookkeeping.

The ideal exact-energy expression is

\[
 \alpha=\min\{1,\exp[\log\pi(Y)-\log\pi(X)+\Delta_{\rm map}]\}.
\]

For this implementation, reject hard-core or spherical-wall violations first.
For valid endpoints, provide `Delta_map` to the existing exact conditional
Poisson acceptance mechanism at the same location where it currently receives
the learned proposal-density correction. Its sampled depletion log factor and
`Delta_map` are combined in the established acceptance rule. Do not replace
this by a newly invented exponential of an estimated overlap volume, and do
not add an independent-redraw Gaussian proposal ratio a second time.

The Poisson point process is part of the existing exact auxiliary construction;
its stochastic log factor is not a deterministic physical free-energy
difference. Correctness of that gate must be retained when composing kernels.

The reverse map negates the complete log correction. Refreshing or discarding
the ephemeral pair/noise variables after the accepted/rejected trial is valid
because their normalized conditional law was used to construct the trial.
The persistent mask and atlas latent state continue to use their own valid
refresh kernels.

Minimal trial pseudocode:

```text
choose moving body i and spectator anchor j with state-independent probabilities
obtain fixed charts, active labels, and an exchange-symmetric chart-pair law
draw (a,b), then draw u from six independent standard normals
z = encode(chart[a], relative_pose(X[i], X[j]))
z_next = gamma*z + s*u
u_next = s*z - gamma*u
Y[i] = absolute_pose(decode(chart[b], z_next), X[j])
reverse_trace = (b,a,u_next)
map_log_ratio = (norm2(u)-norm2(u_next))/2 + logJ(b,z_next)-logJ(a,z)
if endpoint is hard/wall invalid: reject
otherwise: apply the existing exact depletion gate with map_log_ratio
record pair labels, correlation, both latent coordinates, noise trace,
       Jacobian contribution, auxiliary contribution, and acceptance
```

## Adapting to the environment without losing the inverse

The existing atlas decoder uses a deterministic fit `F(X)` and retained `eta`.
Blindly building the forward map from `F(X)` and its reverse from `F(Y)` changes
the charts. The simple triangular-solve inverse and Jacobian above then cease
to describe that algorithm. Reusing only the labels does not solve this.

A practical exact continuation is a **spectator-only fit**

\[
 F_{-i}=F(X\setminus\{i\}).
\]

Exclude the moving body from every fit observation and from any neighbor
selection, normalization, or statistics that determine the charts. Decode the
charts from `F_-i` and the retained `eta`. All spectators remain fixed during
the single-body move, so `F_-i`, chart parameters, and active-mask contents are
identical in its two directions. The fit may still depend on the anchor and
other spectators. The full derivative with respect to unchanged spectator and
latent coordinates is block triangular, so the same conditional extended-map
Jacobian applies.

This allows environment-dependent chart placement while keeping an explicit
inverse. Any simultaneous update of additional bodies must exclude all moved
bodies from the fit or supply a new inverse derivation. For the first control,
using immutable reference charts alone is simpler and isolates the map itself.

## Source selection can help, but its probability must travel with the trace

Uniform source labels often encode the old pose very far from a component's
center. Large standardized coordinates can produce wall/core rejections or
large corrections. Selecting a source by its responsibility is a possible
improvement, not a symmetric-probability free operation.

For any normalized state-dependent pair selector `r_X(a,b)`, add

\[
 \log r_Y(b,a)-\log r_X(a,b)
\]

to `Delta_map`. This is the complete pair-selection correction; it does not
require implementing a separate inverse independent-redraw kernel. Reverse
labels and the retained trace specify exactly which probability to evaluate.

For example, with fixed chart mixture `Q(x)=sum_a w_a q_a(x)`, select the source
with `w_a q_a(x)/Q(x)` and the destination independently with probability `w_b`,
allowing equal labels. Orthogonality gives

\[
 \Delta_{\rm map}+\log\frac{r_Y(b,a)}{r_X(a,b)}
 =\log Q(x)-\log Q(y).
\]

Thus this informed transport is reversible with respect to the reference
mixture, not uniform physical pose volume. It retains correlated latent
positions but does not eliminate a mismatch between `Q` and the physical
target. Excluding equal labels by renormalizing the destination distribution
introduces an additional selection factor and must be handled explicitly.

Similarly, drawing a source with weight `w_a` and then a distinct destination
with `w_b/(1-w_a)` is not exchange symmetric. Drawing an unordered pair from a
known law and choosing its direction with probability one half avoids that
specific asymmetry.

## What is and is not structurally protected

### A strictly symmetric alternative

For an ordinary symmetric proposal, choose a rigid transform `H` in `SE(3)`
from a law independent of the moving pose, then choose `H` or `H^-1` with a
fair coin. Apply the selected transform by left multiplication to the pose,
using a fixed spectator frame if desired. Each transform preserves translation
volume times rotational Haar measure, and the transformation distribution is
invariant under inversion. Consequently the ordinary proposal kernel is
symmetric and its nonphysical acceptance correction is zero. Wall and core
violations become rejections; valid endpoints still require the physical
depletion acceptance step.

More flexible learned maps can retain this property if built from
measure-preserving shears or a measure-preserving conjugacy, with inverse
directions chosen symmetrically and all map parameters independent of the
moving pose. A general Gaussian covariance map does not meet that restriction.
A volume-preserving bijection cannot compress a broad set of physical poses
into a smaller registration volume without a compensating expansion elsewhere;
introducing additional coordinates instead requires their target factors to be
accounted for. Strict symmetry is thus a useful protected control, but may
sacrifice the covariance adaptation needed to connect differently sized basins.

Finally, detailed balance equates **probability fluxes**,
`pi(X) K(X,dY) = pi(Y) K(Y,dX)`, not transition probabilities themselves.
Unequal equilibrium occupancies generally require unequal transition rates.
Symmetric proposals and paired inverse maps cannot remove that physical
requirement.

The construction guarantees a paired reverse trace and a computable Jacobian.
It prevents the accidental absence of an implemented return transformation.
It does not force equal physical transition rates, which need not be equal
between unequally probable configurations, or guarantee useful acceptance.

Even equal Gaussian covariance determinants do not generally give unit
physical Jacobian: the `SO(3)` Haar factors may differ. Ordinary symmetric
pose proposals require stronger restrictions. For example, at `gamma=1`, if
the chart-to-chart map preserves physical pose volume and the ordered-pair
law is exchange symmetric, the auxiliary correction is zero. General
anisotropic covariance transport does not satisfy this condition.

Projection onto a wall, overlap repair, greedy optimization after mapping, or
redrawing until valid destroys the stated inverse unless that additional step
has its own reversible construction. Chart poles, nonfinite arithmetic, or
numerically singular covariance factors must yield declared null proposals.
Any extra validity bounds should treat the forward and inverse trace
symmetrically; one-sided clipping or truncation changes the kernel.

Validation should check both applications of the map, antisymmetry of the full
correction, Jacobians in the correct physical measure, and exact equilibrium
stationarity on a small known target. A paired benchmark must separately
report hard-valid proposals, auxiliary/Jacobian penalties, accepted native
transitions, and transitions out of registered contacts. Successful round trips
in the map alone do not demonstrate equilibrium mixing or crystal assembly.

## Implemented control

`src/basin_involution.rs` implements immutable charts and an unordered weighted
pair law followed by a fair direction coin. The same `apply` operation returns
the new pose, its inverse trace, the auxiliary-density contribution, and the
extended translation/Haar Jacobian. Asymmetric directional pair weights cannot
be supplied to this constructor. Correlations from -1 to 1 are supported.

Four independent checks in `tests/basin_involution.rs` pass:

- Applying the map twice recovers pose and noise within 4.44e-15; the correction
  agrees with the separately implemented component-density ratio within 1.55e-14.
- A numerical 12-dimensional Jacobian agrees with the chart/Haar formula,
  including the zero-correlation case where the fixed-noise pose map is singular.
- 16,000 independent uniform-ball/Haar equilibrium starts, each followed by
  five mapped attempts, retain the known physical moments. Deliberately omitting
  the extended Jacobian shifts the radial second moment by 9.95 standard errors.
- Pair selection is exchange symmetric, and malformed parameters and traces
  are rejected.

This is a library prototype. It has not yet been integrated into the assembly
runner or benchmarked for protein mixing; its physical equilibrium control uses
the uniform-ball/Haar target. The separate masked-mixture campaign still uses
the original full forward/reverse density evaluation.
