# Two-neighbor singleton control

`cluster_phase.singleton_rate` adds a constant-rate event channel for each
mobile tetramer. A selected tetramer can use the existing member-chart mixture
and its deterministic two-contact Gaussian fusion, without carrying a second
misregistered tetramer. It changes the proposal, not the hard shape, ideal-bath
weight, native classifier, or physical measure.

The default rate is zero. No singleton channels, counters, extra random draws,
or serialized configuration fields are introduced in that case. Existing
checkpoints without `singleton_events` deserialize with a zero count. Old
all-mobile validation remains in force: a nonempty `fixed_body_indices` is
rejected, rather than silently moving a fixed body. This does not introduce
fixed-scaffold support for the other production kernels.

## Matched proposal control

For an isolated transport-only singleton phase:

```json
{
  "cluster_phase": {
    "duration": 0.01,
    "singleton_rate": 1.0,
    "dimer_rate": 0.0,
    "trimer_rate": 0.0,
    "transport_probability": 1.0,
    "correlation": 0.9,
    "transport_charts": "members",
    "anchor_count": 2,
    "anchor_contact_uniform_probability": 0.1,
    "oligomer": {}
  }
}
```

The two-neighbor unfused control uses the same settings with `oligomer` absent
(or null). Both arms retain the same two-anchor pool and the same single-contact
components; only the fused components differ. `anchor_count: 1` is a separate
one-anchor control, not the matched fusion comparison. Keep local moves,
ordinary global moves, GCA, center shifts, proposal size/correlation, physical
parameters and observation cadence equal. The existing dimer/trimer phase can
also remain enabled identically in both arms; the example disables it only to
isolate singleton events.

By default the primary anchor is selected by the existing defensive contact law. The
secondary is the nearest spectator to that primary, using spectator coordinates
alone. It is not required to be an existing neighbor of the moving tetramer.
Conditioning the second anchor on the moving pose would require a different
reverse selection correction. If fewer than two spectators exist or no
compatible pair of components survives deterministic screening, the mixture
falls back to its unfused components. The defensive uniform branch is retained
with its existing probability. We do not retry until a useful fusion is found.

Every singleton, including isolated particles, has the same event rate. Filtering
channels to particles with two current contacts would turn their rate off after
a detachment and violate the invariant-rate argument below. The mixture may
contain both fused and unfused labels, and fused labels are selected by their
normalized weights, not forced at each event.

## Balance argument

Write the physical density as

\[
\pi(X)\propto \mathbf1_{\rm hard}(X)\exp[-zV_{\rm excl}(X)]
\]

with respect to the existing positional and proper-orientation measure. An
optional frozen assembly bias multiplies this density by its documented bias
weight; its existing correction is applied to each accepted elementary trial.

For a singleton label \(i\), \(\lambda_i=\lambda_1\) is constant. It is unchanged
when the tetramer forms, breaks or changes contacts. The label does not mean a
maximal component of the contact graph. Dimer and trimer rates still depend only
on the carried internal contacts, which are preserved by their rigid move.

Conditional on the sampled primary anchor \(a\), the anchor pool, spectators,
wall and singleton internal geometry are invariant during the proposal. The
single-member internal key is the identity. Thus the same deterministic fused
catalogue and normalized density \(G_a\) are reconstructed in the reverse move.
Existing fusion explicitly permits two different anchors on the same member;
no new Gaussian fit or fusion rule is needed.

The posterior-source, prior-destination involution has the existing net
proposal ratio

\[
R_q=\frac{G_a(x)}{G_a(y)}
    \frac{p(a\mid Y,i)}{p(a\mid X,i)}.
\]

All chart Jacobians, normalized label probabilities and latent noise factors are
included in this expression; densities use the existing Haar convention.
The independently selected uniform branch has its existing symmetric law and
does not inherit an unused contact-anchor factor. The exact gained/lost Poisson
gate is applied to the full many-body change in exclusion union, not a sum of
two pair energies. The implementation retains its acceptance construction for
the physical density and this proposal ratio. Hard-invalid trials are retained
as rejections.

Let \(K_i\) denote this resulting reversible elementary kernel, including its
rejection mass. Then

\[
\pi(dX)\lambda_i(X)K_i(X,dY)
 =\pi(dY)\lambda_i(Y)K_i(Y,dX).
\]

The same identity holds for each existing rigid-subset channel. Their sum is a
reversible continuous-time generator. Sampling its semigroup at the fixed
configured phase duration therefore preserves the target. Exponential waiting
times include rejected and null events; the total event rate is recomputed when
the graph changes. There is no additional total-rate acceptance factor and no
conditioning on a fixed number of accepted events.

## Validation and observables

Targeted tests cover:

- Constant singleton channel availability in disconnected and connected graphs,
  including invalid-rate guards.
- A singleton's exact fusion of two distinct neighbors, inverse reconstruction,
  catalogue equality after a move, and agreement of expanded and marginal
  proposal corrections.
- Deterministic full-versus-resumed production trajectories with nonzero
  depletion, assembly bias, fused labels and contact-anchor corrections.
- Implicit-zero versus explicit-zero singleton rates: identical physical checkpoint state and
  random sequences (raw configuration hashes intentionally differ), plus old
  counter deserialization.
- Independent analytic sphere/depletion equilibrium from dispersed and compact
  preparations. The depletant radius excludes triple intersection of exclusion
  spheres in any hard-valid state, so the exact pair-lens weight is an independent
  reference. Measured contacts, aggregate size, pair distance and Haar orientation
  moments include all repeated rejected states.

Run the focused tests without replacing any production binary:

```sh
cargo test --target-dir target-validation-singleton --locked --offline -j2 --test cluster_phase_clock \
  --test cluster_phase_runner --test oligomer_proposal singleton -- --test-threads=2
cargo test --target-dir target-validation-singleton --locked --offline -j2 --test cluster_phase_stationarity \
  singleton_two_neighbor -- --test-threads=2 --nocapture
```

The isolated target directory matters: Cargo may also build production binaries
while building integration tests. These checks establish the new channel's construction and tested reference
behavior. Protein sampling efficiency remains an empirical question. Report
source coverage, hard validity, full acceptance corrections, completed changes
between contact environments, permutation-insensitive contact statistics and
CPU-normalized independent samples. A larger accepted-move count alone is not
an improvement in reorganization.

## Frozen-state protein pilot

`tools/benchmark_singleton_guidance.py` freezes a matched unfused/fused pilot
from the archived sweep-60,900 snapshot. Its defaults are four independent
populations of 128 **reset phases** per arm, rate 1, duration 0.01,
transport probability 1, two anchors and contact-anchor uniform probability
0.1. Both arms use the same bath, snapshot and correlation. The source bundle
must match the compiled `contact-anchor-benchmark` binary. The driver archives
the inputs and itself, checks their hashes before continuation, and caps workers
at two.

```sh
python3 tools/benchmark_singleton_guidance.py freeze \
  --out /path/to/new-pilot \
  --binary target/release/contact-anchor-benchmark \
  --bundle /path/to/matching/source-bundle
python3 /path/to/new-pilot/provenance/benchmark_singleton_guidance.py run \
  --out /path/to/new-pilot
python3 /path/to/new-pilot/provenance/benchmark_singleton_guidance.py analyze \
  --out /path/to/new-pilot
```

The optional `--native-definition` is an observer input only. A reset-phase
pilot measures proposal opportunity and retained contact changes from a fixed
starting environment; it cannot measure equilibrium occupancies, trajectory
mixing time, or assembly stability. Subsequent evolving trajectories must test
those separately after source coverage and proposal cost have been assessed.

On 2026-10-01 the five focused tests passed. The independent sphere control used
250,000 uniform reference attempts (87,123 hard-valid, importance ESS 82,673).
After 3,000 burn-in sweeps, each production chain retained 24,000 observations
in 48 blocks. The phase included 70% transport and 30% local singleton events,
composed with the existing uniform single-body kernel; the runner restart test
separately exercises 100% transport phases.

| Observable | Independent reference | Dispersed start | Compact start |
|---|---:|---:|---:|
| Exclusion contacts | 0.50787 ± 0.00248 | 0.51025 ± 0.00638 | 0.51796 ± 0.00637 |
| Largest component | 1.50284 ± 0.00238 | 1.50583 ± 0.00628 | 1.51292 ± 0.00623 |
| Mean squared pair distance | 9.60459 ± 0.00876 | 9.62260 ± 0.02381 | 9.57058 ± 0.02475 |
| Mean quaternion scalar squared | 0.25021 ± 0.00050 | 0.24738 ± 0.00178 | 0.24912 ± 0.00187 |

Errors are standard errors (importance sampling for the reference, block means
for trajectories). The two chains accepted 116 and 112 moves involving fused
labels, and 174 and 157 involutions with a nonzero anchor correction. All four
reference comparisons and initialization-agreement checks passed their
predeclared five-combined-standard-error tolerances. This is an implementation
reference check, not a proof of rapid mixing or a result for proteins.

## Optional retained contact pool

Set `cluster_phase.anchor_pool_selection` to `"contact_without_replacement"`
to choose every anchor from the moving subset's current contacts with a
uniform defensive component. This requires `transport_charts: "members"` and
`anchor_contact_uniform_probability: epsilon` with `0 < epsilon <= 1`.
The default `"nearest"` preserves the earlier algorithm, serialized defaults
and random stream. This option changes the proposal law only. It is not a
claim of improved protein sampling.

For a fixed labeled moving subset S, let c_a(X) count its exclusion-contact
edges to spectator a. Keep an **ordered** tuple A = (a_1,...,a_k), where
k = min(anchor_count, number of spectators). At step j, remove previously
selected labels, denote the remaining set R_j, and draw

\[
p(a_j\mid X,S,A_{<j}) =
\begin{cases}
\epsilon/|R_j|+(1-\epsilon)c_{a_j}(X)/\sum_{b\in R_j}c_b(X),
 &\sum_{b\in R_j}c_b(X)>0,\\
1/|R_j|, &\text{otherwise}.
\end{cases}
\]

The primary draw is unchanged. Each additional anchor is drawn exactly once;
there is no retry until fusion succeeds. All contacts, including nonnative
ones, contribute. If all remaining contacts have already been selected, the
remaining spectators are sampled uniformly. Fewer than k contacting neighbors
is not a null event. Pool length depends only on the fixed body counts.

Record the joint log probability

\[
\log p(A\mid X,S)=\sum_j \log p(a_j\mid X,S,A_{<j}).
\]

During the trial, retain A with its order. Its spectator poses, wall, fixed
spectators and the guarded rounded internal geometry of S are unchanged.
Consequently the conditional chart catalogue G_A is identical when reconstructed
at Y. The reverse law evaluates the probability of **the same A** at Y;
it does not redraw a pool, sort its labels or substitute newly attached neighbors.
The complete proposal correction is

\[
\log R_q=\log G_A(X)-\log G_A(Y)
  +\log p(A\mid Y,S)-\log p(A\mid X,S).
\]

To see balance, condition the forward and reverse extended involution on A.
Its chart/noise factors reduce to G_A(X)/G_A(Y) by the existing transport
identity. The remaining auxiliary density ratio is exactly p(A|Y,S)/p(A|X,S).
Applying the existing gained/lost-count Poisson acceptance rule with their
product gives symmetric accepted flow on that extended state. Summation over
all ordered A and addition of rejection recover the physical invariant kernel.
The constant singleton/internal-contact subset rates and fixed-duration phase
argument remain unchanged. Neither the proposal nor the anchor law must
separately preserve the physical distribution.

Positive epsilon guarantees reverse support even if a trial loses every selected
contact. This has a cost: losing a strongly preferred anchor can substantially
penalize a proposal. Therefore report the pool correction, hard validity,
physical gate and completed contact changes separately; finding more fused
components is not itself a speedup. The uniform pose branch remains an
independent symmetric kernel and receives no unused pool-selection correction.

For auditability `anchor_forward_probability` and `anchor_reverse_probability`
still report the primary marginal. In this mode `pool_forward_log_probability`
and `pool_reverse_log_probability` report the full ordered law, and
`anchor_log_reverse_forward` is their difference (not an additional primary
factor). Products are accumulated in log space. Hard-invalid trials retain the
forward tuple and leave reverse fields null. The tuple is move auxiliary state;
it is not persistent learning or a new physical state variable.

Validation includes exhaustive ordered-pool normalization for empty and weighted
contact graphs, contact exhaustion, duplicate-label guards, reverse support after
contact exchange, sampled conditional frequencies and exact count-one RNG
compatibility. The full runner independently reconstructs pool probabilities
from sphere-center distances while checking accepted/rejected-state replay,
assembly-bias composition and checkpoint continuation. A four-sphere depletion
reference uses two selected anchors among three available spectators; unlike a
three-sphere singleton test, it exercises nontrivial secondary-anchor selection.
The existing fused-chart test reconstructs an identical conditional catalogue
and inverse map from the retained anchor tuple.
