# Taking the flexible pair kernel into an all-mobile system

This is a source audit and integration design, not an implemented production
option or a performance result. The fixed-label protein comparison must finish
before promotion. Its whole-contact and internal-native observers measure
different questions; neither decides finite-system assembly.

## Preserve the conditional kernel, correct selection

The current `FlexibleSurrogateKernel` moves two fixed labels, with all other
poses fixed during the call. Its random-scan inner chain is reversible under
the hard-supported reference measure `exp(S) mu`. One final decision combines
the fair-order two-singleton physical bath and `S(old) - S(new)`. It permits
changes to the selected pair's internal contact and relative pose.

Selecting that pair from the current configuration adds another probability.
For a retained selection label `a` with probability `p_a(X)`, the final gate
must instead use

\[
\log A_{\rm raw}=\log F_{\rm bath}+S_a(X)-S_a(Y)
                 +\log p_a(Y)-\log p_a(X).
\]

Here the reverse proposal retains the same pair, cloud, fixed spectators and
inner settings. The selection factor belongs in the **same final decision**.
It cannot be applied after a call that already committed its endpoint. Zero
reverse probability gives an ordinary, recorded rejection, not a fatal
nonfinite-log error or permission to draw another pair.

This is distinct from the existing rigid-subset clock. Its internally defined
eligibility is unchanged by an allowed move. Even the older contact-preserving
flexible proposal can retain that cancellation: changing relative geometry is
not itself the problem. Allowing attachment/detachment changes the eligibility
bit. Selecting only contacting pairs then gives zero reverse probability for a
detached endpoint. Keeping the guard preserves balance but prevents that route
to detachment; dropping both guard and correction is incorrect.

## A minimal intensive discrete selector

For `N >= 2`, choose an ordered root `i` uniformly and a partner `j != i`.
Define a fixed finite-radius center neighborhood and its degree `d_i(X)`.
With a frozen defensive probability `eta` in `(0,1)`, use the **complete**
partner mixture

\[
g_i(j\mid X)=
\begin{cases}
 (1-\eta)\,1_{j\in\mathcal N_i(X)}/d_i(X)+\eta/(N-1),&d_i(X)>0,\\
 1/(N-1),&d_i(X)=0.
\end{cases}
\]

The empty-neighborhood rule is part of the declared law. Retain the ordered
root in the journal and add `log g_i(j|Y) - log g_i(j|X)` at the outer gate;
the common `1/N` cancels. The branch used to draw the partner does not replace
the complete mixture density. Recompute the root's degree at the actual trial
endpoint, including both changed member poses.

If instead discarding the root and retaining only an unordered pair, the
selection law is `[g_i(j|X) + g_j(i|X)]/N`. Its correction is different.
Retaining the root avoids that additional summation. A separate cutoff-only
selector with no global fallback would reject endpoints outside its reverse
neighborhood. A neighborhood broader than exclusion contact can permit contact
breaking within that support, but a hard cutoff still restricts that channel's
reach. The mixture above has positive support everywhere, including the
zero-degree uniform fallback. Other elementary moves remain relevant to global
exploration.

A frozen schedule of order `N` attempts per sweep supplies a constant expected
number of selected roots per particle. One attempt per whole-system sweep
does not. At fixed density, a finite-radius neighborhood has ordinarily bounded
mean degree. Uniformly choosing from all `N(N-1)/2` pairs with only `O(N)`
attempts would instead make a particular local pair increasingly unlikely.
The defensive component maintains support; it does not remove the proposal
penalty for an endpoint supported only by its global tail.

An alternative retains the existing fixed-duration event clock and bounded
pair rates `a_ij(X)`. Its correction is `log a_ij(Y) - log a_ij(X)`, with no
total-rate ratio: state-dependent holding times supply that part. A fixed
number of draws normalized by the total rate requires the normalization ratio.
These two schedules must not be interchanged. Both rejected clock events and
rejected discrete trials consume their prescribed opportunity; runtime caps
must not silently truncate a phase into a valid endpoint.

## The existing point cloud already follows the particles correctly

The quadrature points are frozen in the **body frame**, not around an old
world-space pair pose. `DimerDepletionSurrogate::evaluate` transforms the same
points with each candidate member pose. Its score estimates

\[
S=z\{\lvert E_0\cap E_1\rvert+
       \lvert(E_0\cup E_1)\cap B\rvert\},
\]

where `B` is the fixed spectator union during this call. Spectator shielding
and the unshielded internal contribution give this union expression without
discarding many-body overlap. The finite quadrature defines a deterministic
function of each trial pair. Freezing that **function** does not require its
transformed points to remain at fixed world positions.

The cloud can therefore be reused for different selected pairs without growing
with vessel volume. Its body coordinates, retained indices, original raw-count
volume weight and guidance strength must remain bound to the configuration.
Do not refresh or optimize it conditional on the current pair while retaining
the same fixed-surrogate acceptance argument. No new world-space point field
or cloud-transport construction is needed for this integration.

## Intensive attempt counts are not yet intensive runtime

There is no body-level spatial neighbor index in the current production code.
The `SphereTree` accelerates atoms within a body. `ContactGraph::build` tests
all body pairs and stores dense adjacency; its incremental update still clones
that dense structure. The flexible kernel validates/copies spectators and its
score filters them by a conservative center bound. These scans add roughly
`O(m N)` overhead per pair call, even when the expensive point predicates only
involve nearby bodies. `O(N)` calls per sweep can therefore retain quadratic
overhead.

A later implementation can rebuild a deterministic body cell list once after
the other move phases, then update the moved labels on acceptance. Selection,
hard endpoint tests, score evaluation and both bath legs each need conservative
candidate queries. Gaussian translations have unbounded support: a spectator
list made only around the old poses is unsafe. Query the actual trial geometry
or a proven envelope, and keep the independent exhaustive path as a control.
Measure neighborhood size, scan/copy cost, geometry work and total sampler CPU
separately before claiming volume-independent performance.

## Integration and proof boundaries

Initially add a separate, default-disabled phase after existing collective
moves, for the spherical physical target with frozen proposals and no assembly
bias. Other auxiliary targets and bias tables require their own density terms.
Use dedicated versioned RNG roles keyed by absolute sweep, so disabled mode
does not alter existing streams. Checkpoints must bind selector law, cutoff,
schedule, cloud, scales, guidance and bath, plus accumulated resource counters.
A deterministic spatial cache can be reconstructed. Persist begun/outcome
records and drain partial bath records on failure before propagating the error.

The general checked theorem `twoLeg_poisson_correct` supplies the relevant
measure argument. For a fixed label, take its reference measure to be
`exp(S_a) mu` and its `base` to be `p_a exp(-S_a)`. The corrected kernel is then
reversible with respect to `p_a pi`. Summing these selected-label flows gives
reversibility under `pi`; null selections/rejections supply the diagonal.
The component kernel need not independently preserve `pi`. For event clocks,
replace `p_a` by the bounded rate `a_a` and use the rate-weighted flow.

This substitution and mixture connection are an integration argument, not a
new separately checked Lean corollary. The older
`measurable_internal_rate_reversible` theorem assumes rate equality along
transitions and cannot directly justify changing eligibility. Concrete
geometry, Poisson thinning, inner-kernel reversibility, RNG, floating-point
execution, selection support and clock execution remain implementation
obligations.

Source references:

- `src/flexible_surrogate_chain.rs`: fixed-label kernel and final gate.
- `src/depletion_surrogate.rs`: frozen body cloud, union score and pruning.
- `src/cluster_phase.rs`: contact graph, eligibility guard and event clock.
- `src/simulation.rs`: phase composition and checkpoint boundary.
- `formal/ReversibleSampling/TwoLegPoisson.lean`: general weighted gate.
- `formal/ReversibleSampling/ClusterMeasureRates.lean`: invariant-rate premise.
- [Conditional native comparison](surrogate-internal-native-comparison.md):
  completed-control evidence, prepared observers and interpretation limits.

No production kernel, campaign allocation, physical target, or active
executable changed in this audit.
