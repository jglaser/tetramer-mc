# Posterior-selected Gaussian-chart involution

The Gaussian source chart can be selected from its posterior responsibility
at the current pose, eliminating the large correction caused by choosing an
unrelated source chart. With an appropriate destination law, the complete
nonphysical acceptance correction becomes **the full Gaussian-mixture density
ratio**, for every fixed latent correlation. This construction is reversible
with respect to the fitted mixture; the physical MH/depletion correction
still remains necessary. The conditional Rust docking runner now implements
this law as `--method posterior-involution`. Validation is summarized below;
it does not establish a protein contact-mixing speedup.

## Extended target and cancellation

Hold a spectator anchor, chart parameters, component weights and all other
auxiliary variables fixed during the physical move. Let

\[
 G(x)=\sum_a w_a q_a(x),\qquad
 r_a(x)=\frac{w_aq_a(x)}{G(x)}.
\]

Here `q_a` is the normalized Gaussian chart density with respect to physical
translation and normalized SO(3) Haar measure. If `x=T_a(z)`, then
`q_a(x)=phi(z)/J_a(z)`, including the chart determinant and Cayley/Haar factor.
The source is drawn as `a ~ r(x)`, the destination independently as `b ~ w`,
and `u ~ N(0,I)`. These are normalized auxiliary variables, giving joint target

\[
 \widetilde\pi(x,a,b,u)=\pi(x)r_a(x)w_b\phi(u).
\]

Its physical marginal is exactly `pi(x)`, regardless of how accurately `G`
approximates the physical distribution. Apply the existing orthogonal map,
with a fixed `c` and `s=sqrt(1-c^2)`:

\[
 z'=cz+su,\qquad u'=sz-cu,\qquad y=T_b(z'),
 \qquad (a,b)\longmapsto(b,a).
\]

The full map is an involution. Its extended Jacobian is `J_b(z')/J_a(z)`.
The complete reverse/forward acceptance ratio is

\[
 R=\frac{\pi(y)}{\pi(x)}
   \frac{r_b(y)w_a}{r_a(x)w_b}
   \frac{\phi(u')}{\phi(u)}\frac{J_b(z')}{J_a(z)}.
\]

The label ratio simplifies to

\[
 \frac{r_b(y)w_a}{r_a(x)w_b}
 =\frac{q_b(y)}{q_a(x)}\frac{G(x)}{G(y)}.
\]

Orthogonality preserves `|z|²+|u|²`, so the remaining Gaussian/Jacobian
factor is `q_a(x)/q_b(y)`. Hence

\[
 \boxed{R=\frac{\pi(y)G(x)}{\pi(x)G(y)}}.
\]

Thus `G(x)K(x,dy)=G(y)K(y,dx)` for the uncorrected chart-transport kernel.
The physical correction is `log G(x)-log G(y)`. With the existing exact
depletion gate, this replaces the previous selected-chart correction; it
must **not** be added on top of that correction, which would double count.
Hard/capture-invalid poses remain null attempts, without retrying until valid.

The cancellation can also be audited numerically: take the existing map's
`log_auxiliary_ratio + log_extended_jacobian`, add
`log r_b(y)+log w_a-log r_a(x)-log w_b`, and compare against the full `log G`
difference. Implementing the simplified density ratio avoids cancellation
between extremely large component-specific terms, while retaining the full
trace supplies an independent balance check.

## Destination graphs and the redraw limit

More generally, choose `b ~ P[a,:]` for a frozen transition matrix reversible
with respect to the mixture weights:

\[
 w_aP_{ab}=w_bP_{ba}.
\]

The same cancellation holds. Destination selection `P_ab=w_b` is the minimum
construction and has symmetric flows `w_a w_b`. A graph may retain identity
transitions to satisfy its row sums. Simply excluding `a=b` and renormalizing
the remaining `w_b` generally breaks this weighted graph reversibility; its
extra reverse/forward factor cannot be dropped.

At `c=0`, `z'=u`; with `P_ab=w_b`, the destination is an independent draw
from the complete Gaussian mixture `G`, and acceptance is exactly the ordinary
Gaussian-mixture independence correction. With a general destination graph,
the destination still depends on the posterior source label, so `c=0` need
not be an independent redraw. At `c=1`, latent residuals are preserved and
equal source/destination labels produce an identity. Intermediate correlations
refresh residuals without losing the explicit inverse.

The construction works for a fixed `c` in `[-1,1]`. A symmetric frozen
pair-dependent `c_ab=c_ba` is also compatible with the inverse construction.
Choosing correlation from the current pose introduces additional issues and
is not covered by this argument.

## Uniform defense and the required matched control

The simplest defense remains a separate uniform cube/Haar kernel, chosen
with fixed probability `epsilon`; the Gaussian transport kernel is chosen
with probability `1-epsilon`. Each kernel separately preserves the physical
target after correction, so their mixture also does. Within the capture ball
contained by the uniform cube, the uniform branch has unit proposal ratio.

For this construction, `G` and its responsibilities contain **only the
normalized Gaussian atlas**, including any broad Gaussian component. They
exclude the separate uniform cube branch. The source density is therefore
the existing `relative_log_density`, not `log_density` which includes the
uniform floor.

The exact `c=0` control is a **branch-separated** Gaussian-independent/uniform
kernel with the same branch probability. It need not agree with the existing
`DockingMethod::Mixture` kernel at nonzero `epsilon`: that kernel uses
`q=(1-epsilon)G+epsilon U` in its acceptance ratio for both branches. Both are
valid physical kernels, but they marginalize the branch label differently.
At `epsilon=0` their Gaussian redraw limits coincide. A benchmark should keep
this distinction explicit rather than silently claiming identical acceptance.

## Anchors, masks and adaptation

For the current fixed-neighborhood experiment, an anchor `j` can be selected
independently of the moving pose and retained in the reverse trace. Use
`G_j`, source responsibilities conditioned on `j`, and correction
`log G_j(x)-log G_j(y)`. Each anchor-conditional corrected kernel preserves
the physical conditional distribution, so mixing those kernels is valid.
No average over anchors is required in that MH correction.

This differs from the independent normalizer estimator: its pose proposal
law marginalizes the randomly selected anchor, so its importance denominator
must average anchor densities. An alternative transport can also marginalize
anchors by treating `(anchor,chart)` as the mixture label, selecting its full
posterior and an independent destination joint label; then the correction
uses the global anchor-averaged `G`. That is a different, explicit design.

If anchor eligibility or selection probability depends on the moving pose,
the retained-anchor probability may differ on reversal and must be included.
The simple proof likewise requires fixed chart parameters during the move.
Spectator-only fits are admissible because the spectators do not move.
Refitting from the current moving pose, changing active masks inside the
attempt, or updating retained proposal parameters requires the already
specified auxiliary-ensemble target and its corresponding reverse map.
Posterior chart selection alone does not validate online learning.

## Connection to the checked involution theorem

The [Lean involution theorem](../formal/ReversibleSampling/Involution.lean)
has a direct abstract interpretation for the independent-destination
construction. In latent coordinates use the reference probability measure

\[
 d\mu(a,b,z,u)=w_a w_b\phi(z)\phi(u)\,dz\,du
\]

and target density, relative to that reference,

\[
 p(a,b,z,u)=\frac{\pi(T_a(z))}{G(T_a(z))}.
\]

The label swap preserves the symmetric factor `w_a w_b`; the orthogonal
latent map preserves the joint standard Gaussian law. The extended map is
therefore an involution preserving this reference measure. Its Metropolis
ratio is `p(b,a,z',u')/p(a,b,z,u)`, exactly the boxed correction above.
This avoids confusing preservation of the **reference** measure with
preservation of the physical target by the raw proposal.

Pushing the extended target onto x gives
`[pi(x)/G(x)] sum_a w_a q_a(x) = pi(x)`. Hard and capture exclusions belong
in `pi`; invalid endpoints have zero target density. The chart seam has zero
physical measure and needs a consistent measurable definition on that null
set when instantiating the theorem.

Lean checks the general implication from a measurable reference-preserving
involution and finite measurable target density to completed-kernel balance
and invariance. The argument identifying this particular Gaussian/chart map
with those hypotheses is still mathematical documentation, not a checked
Lean specialization or Rust verification. The implicit Poisson acceptance
gate additionally requires its own augmented-state balance argument; a noisy
substitution for the exact target ratio is not licensed by this theorem.

The current proof uses proper rotations and an open capture domain. Reusing
it with periodic wrapping needs a consistent image convention, inverse map
and proposal density; an unrestricted map followed by many-to-one wrapping
cannot inherit this Jacobian argument automatically. Numerical chart seams
or unrepresentable coordinates should produce a null move, not an unaccounted
retry or source-label substitution.

## Existing implementations and minimum validation

The original `--method involution` in `src/docking.rs` retains static pair
weights `w_a w_b`, `FixedBasinInvolution::draw_trace`, and its selected-component
correction. The new `--method posterior-involution` uses source log weights
`log w_a + log q_a(x)` and independent destination weights `w_b`. A Gumbel-max
draw avoids exponentiating very small responsibilities or applying a component
cutoff. Ordinary finite random-number and floating-point precision remain
implementation limits.

Both methods reuse the same deterministic map and inverse trace. The new
method **replaces** the old correction by `log G(x)-log G(y)`. Logs retain the
old map factor, forward and reverse source responsibilities, label correction,
and independently expanded result for auditing. The uniform branch is
separate and has unit proposal ratio; chart responsibilities never include
that branch. At correlation one, an identical source and destination gives
the exact unchanged input pose and zero correction.

The earlier Python prototype already implements the deterministic `c=1`
version with posterior source labels and a weight-reversible destination graph:

- `/home/xvg/protein-nucleation/scripts/mixture_pose_transport.py`:
  `MixturePoseTransport.sample`, `apply`, and `reversible_graph`.
- `/home/xvg/protein-nucleation/scripts/moment_pose_transport.py`:
  `FrozenMomentMixture.responsibilities` and normalized physical chart densities.
- `/home/xvg/protein-nucleation/scripts/validate_mixture_pose_transport.py`:
  source-label sampling, inverse maps, density-ratio cancellation and accepted
  flux checks.

That Python implementation keeps additional corrections when mapping between
different radial families. The simple orthogonal Gaussian proof above does
not transfer unchanged to unequal Student-t families. The noisy Gaussian
extension for `c<1` is now implemented by the new Rust method; the older
Python result is not used as evidence of its correctness.

Release validation now includes:

- `tests/posterior_docking.rs`: unequal mixture weights, coupled anisotropic
  covariances, nontrivial chart/spectator rotations, posterior-source
  frequencies, and the independent destination/normal-latent law at c=0.
  Independent exact draws from G remain distributed as G after the Gaussian
  branch at c=−0.6, 0, 0.9 and 1. The largest paired observable discrepancy is
  2.12 observed standard errors; the expanded/simplified correction agrees
  within 1.8×10⁻¹⁴. Target-G acceptance factors are one to numerical precision.
- `tests/involution_depletion.rs`: 6,000 independent exact physical starts
  per arm, three updates each, c=0 and c=0.9. One-neighbor sphere results use
  the analytic depletion lens law; the two-neighbor reference samples the
  full exclusion union, including triple overlap. The largest paired change
  is 2.12 observed standard errors. This checks the actual new proposal
  together with the existing conditional Poisson gate, including its uniform
  branch; it is not a comparison to fitted Gaussian weights.
- `tests/docking_runner.rs`: actual asymmetric-dumbbell runs with two fixed
  neighbors replay the source/label/map/depletion factors and reproduce
  uninterrupted trajectories exactly after restart, for c=0, 0.9 and 1.

```bash
cargo test --locked --release --test posterior_docking \
  --test docking_runner --test involution_depletion
```

For a matched performance comparison, use `--method posterior-involution`
in both arms and change only `--correlation 0` versus `--correlation 0.9`.
Keep the complete frozen atlas, local proposals, branch probabilities,
physical target and starts identical. This c=0 arm is an independent
Gaussian redraw with the same separate uniform kernel; the existing
`--method mixture` is a distinct control at nonzero uniform probability.

Separate coverage and contact-exchange tests remain necessary. Eliminating
wrong-source penalties cannot supply missing contact volume, guarantee
physical acceptance of an unfavorable destination, or make unequal physical
occupancies yield equal transition rates.
