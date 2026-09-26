# Rigid-subset moves at fixed algorithmic duration

Status (2026-09-25): implemented as an optional production kernel in `src/cluster_phase.rs`, with the whole-union gate in `src/rigid_subset.rs`. Clock, geometry, runner, restart, bias and independent physical equilibrium checks pass. The first short protein pilot has no completed cluster contact exchanges and establishes no sampling speedup. See [validation and pilot results](rigid-subset-validation.md).

![Rigid-subset phase schematic](../runs/rigid-subset-schematic-20260925/rigid-subset-phase.png)

The [SVG](../runs/rigid-subset-schematic-20260925/rigid-subset-phase.svg) is editable, and [the plotting script](../tools/plot_rigid_subset_phase.py) regenerates both formats. Each four-lobed icon represents one tetramer. The geometry is schematic, not a measured trajectory.

## What changes

A tetramer dimer or trimer can be translated and rotated as a rigid unit during one trial. It remains able to split or rearrange through the existing single-tetramer moves and later, differently selected subset moves. There is no permanent oligomer identity and no auxiliary bond memory.

The eligibility rule concerns only the selected labels' internal geometry. In particular, an eligible dimer remains eligible after one of its members attaches to a larger aggregate. Selecting only maximal connected components would lose this property: its reverse selection could then recruit the entire newly joined aggregate.

For each unordered subset of labels `S` of size two or three, define

\[
\lambda_S(X)=\kappa_{|S|}\,\mathbf1\{G(X_S)\text{ is connected}\}.
\]

Here `G` is the exclusion-contact graph of the selected bodies. Contact means that the two atomic sphere unions, with each atom radius increased by the configured depletant radius, overlap. It is a geometric union predicate, not a native label, a center-distance surrogate, or a depletion energy approximation. For triples, any two edges making a connected graph suffice; triangles are counted once. Labels outside `S` have no role in this connectivity test. Rates are algorithmic parameters, not physical kinetic rates.

A common rigid motion preserves internal relative poses, so

\[
\lambda_S(X)=\lambda_S(Y)
\]

for each proposed rigid-subset endpoint. There is no factor for the number of newly made external contacts. The number of *other* eligible subsets can nevertheless change, and the event clock accounts for that change.

## One handle supplies six proposal coordinates

Choose a handle `h` from `S` using a label-based rule that is the same in both directions. A local rigid pose perturbation or the existing frozen transport proposal supplies a new handle pose `g_h'`. Carry all selected labels with

\[
H=g_h'g_h^{-1},\qquad g_i'=Hg_i\quad(i\in S).
\]

The internal coordinates `u_i = g_h^{-1}g_i` are unchanged. In the coordinates `(g_h, {u_i})`, the move acts only on the six-dimensional handle pose. Consequently the transport uses one chart Jacobian and one proposal-ratio correction, **not** one factor per carried body. The covariance is the same handle-pose proposal used for a single tetramer; it need not already describe an optimal oligomer docking basin.

The frozen atlas's complete proposal law must be used, including its defensive component and any forward/reverse channel or anchor selection. An anchor for a subset transport must lie outside the moving subset; choosing it uniformly from the fixed complementary labels makes that label-selection factor cancel. A state-dependent anchor shortlist would require its own reverse-selection correction. The implementation must reject unsupported adaptive-model modes rather than reuse a frozen-model argument for them.

All carried atoms must remain inside the physical wall and avoid every stationary body. A collision gives a self-loop. Pair distances within the carried subset are already preserved by the common rigid map, subject to floating-point implementation checks.

## The whole exclusion union enters the physical gate

Let `B_X` and `B_Y` be the old and new unions of the selected bodies' exclusion regions, and let `E` be the stationary bodies' exclusion union. Common rigid motion preserves `|B|`. For the existing ideal-depletant target and hard-valid endpoints,

\[
\begin{aligned}
|E\cup B_Y|-|E\cup B_X|
&=-|E\cap B_Y|+|E\cap B_X|,\\
\log\frac{\pi(Y)}{\pi(X)}
&=z\,[|E\cap B_Y|-|E\cap B_X|].
\end{aligned}
\]

Thus the subset carries its internal depletion stabilization with it. External overlap remains a **many-body union** calculation. Summing pair overlap volumes would double count intersections covered by multiple neighbors.

This identity explains what the existing exact Poisson gate must represent; it does not authorize replacing that gate by an exponentiated noisy volume estimate. The gate needs the changed *whole-system* union or an exactly equivalent gained/lost construction. Proposal, map, and any deliberately imposed frozen assembly-bias corrections still apply. In this target, the spherical wall excludes protein atoms but is not a solvent-depletion surface.

## Why fixed algorithmic duration is required

At state `X`, sum the rates of all eligible subsets:

\[
\Lambda(X)=\sum_S\lambda_S(X).
\]

A phase of prescribed duration `tau` proceeds as follows:

1. Build the unique eligible pair/triple list and its total rate. If the total is zero, retain `X` for the rest of the phase.
2. Draw an exponential waiting time with rate `Lambda(X)`.
3. If it extends past the phase endpoint, retain the current state through that endpoint and finish. Do not execute the event beyond the endpoint.
4. Otherwise advance the clock, choose `S` with probability `lambda_S(X)/Lambda(X)`, and attempt its corrected rigid move.
5. Keep both accepted and rejected outcomes, recompute eligibility if the state changes, and continue to the same fixed endpoint.

Rejections, hard-invalid proposals, and unavailable proposal channels consume their sampled waiting time. An event log is useful diagnostic output, but an unweighted histogram of event states generally is not the physical equilibrium histogram. Use states at fixed algorithmic times, such as the phase endpoints, or correctly account for residence times.

There is no extra `Lambda(X)/Lambda(Y)` factor at an event: that compensation is supplied by the state-dependent holding time. Selecting among eligible subsets and then stopping after a fixed *number* of those events would lose it. Stopping the physical phase after a CPU-time budget is likewise not the prescribed invariant kernel. An event-count safety cutoff must not silently return a truncated phase as a valid endpoint; finite `N` and finite rates give a nonexplosive mathematical process, but runtime can still become large.

A minimal example shows the bias. Let two states have equal physical weight and a rate-one channel exchange them in either direction. Add nine units of null-event rate in the second state only. The total event rates are then one and ten. The event-indexed chain spends `10/11` of its recorded events in the second state, even when rejected/null events are retained. The fixed-time process still spends half its time in each state. State-dependent null rates are enough to demonstrate why recording events and recording physical equilibrium states are different.

The fixed duration does not make these moves physical dynamics. Existing MC sweeps, this phase, and the GCA each define sampling kernels, not a calibrated assembly timescale.

## Detailed balance and the fixed-time kernel

Let `K_S` be the complete Markov kernel for a *specified* subset: retained proposal variables, hard constraints, the existing exact depletion gate, the proposal/Jacobian corrections, and rejection on the diagonal. Assume its accepted flow is symmetric with respect to the desired target. This is an assumption about the combined kernel, not a claim that the proposal or acceptance separately preserves the target.

Because `K_S` only changes `S` by a common rigid motion,

\[
\pi(dX)\lambda_S(X)K_S(X,dY)
=\pi(dY)\lambda_S(Y)K_S(Y,dX).
\]

Summing over subsets gives the reversible generator

\[
Lf(X)=\sum_S\lambda_S(X)\left[\int f(Y)K_S(X,dY)-f(X)\right].
\]

For finite `N`, a state-independent rate bound is

\[
M=\kappa_2\binom{N}{2}+\kappa_3\binom{N}{3}.
\]

When `M > 0`, the uniformized kernel is

\[
P(X,dY)=\sum_S\frac{\lambda_S(X)}M K_S(X,dY)
+\left(1-\frac{\Lambda(X)}M\right)\delta_X(dY).
\]

Its off-diagonal flows are symmetric and its rows sum to one; it is reversible and invariant. Therefore

\[
P_\tau=e^{-M\tau}\sum_{n=0}^\infty\frac{(M\tau)^n}{n!}P^n
\]

is invariant and reversible as well. If `M = 0`, it is the identity. Removing the null events of this constant-rate construction yields the state-dependent exponential event algorithm above. This supplies the bridge from invariant rates to a fixed-time sampler, without assuming a fixed total rate in the physical configuration.

Composing a fixed-duration phase with the existing invariant kernels preserves their common target. A fixed-order composition need not itself satisfy detailed balance, even when each individual kernel does. That distinction does not obstruct invariance.

## Configuration and initial scope

The opt-in configuration is

```json
"cluster_phase": {
  "duration": 0.01,
  "dimer_rate": 1.0,
  "trimer_rate": 0.25,
  "transport_probability": 0.5,
  "correlation": 0.9,
  "local_translation_std_A": 0.2,
  "local_small_angle_std_degrees": 1.0
}
```

Optional `"transport_charts": "members"` with `"anchor_count": 4` replaces the
single-handle transport charts by the member/anchor mixture described in
[oligomer-conditioned learning](oligomer-conditioned-learning.md#implemented-member-charts-no-fitting).
It needs a nonperiodic posterior-involution model. The portable example is
[spherical-cluster-members.json](../examples/spherical-cluster-members.json). Omitting both keys keeps
the handle charts and the existing random stream.

These values are a starting point for tests, not a tuned recommendation. Expected attempt count is approximately `duration * Lambda` over a slowly changing state, rather than a prescribed number per phase. Doubling every rate is equivalent to doubling the duration for this isolated phase. Separate size rates control the relative rate of dimers and trimers, while the local/transport mixture controls how a selected subset is moved.

The implemented initial scope is spherical boundaries and frozen proposal models. A local-only rigid branch can operate without an atlas. The phase runs after the ordinary single-body move, GCA, and center-shift schedules and before any auxiliary refresh. It uses separate named streams and archives configuration, clock diagnostics and all attempted events when move logging is enabled; checkpoint continuation is deterministic. Assembly bias corrects each elementary subset move with the existing instantaneous cluster-size observable. Cluster bias counts live under `counts.cluster_phase` (`physical_accepted`, `bias_rejected`, `accepted`), separately from the original `counts.assembly_bias` per-kernel counters. Unsupported periodic and adaptive-model combinations fail validation. The portable configuration is [spherical-cluster-phase.json](../examples/spherical-cluster-phase.json), with a command in the [example guide](../examples/README.md#experimental-dimertrimer-transport-phase).

Spherical GCA, center shifts, and this phase can complement each other. GCA retains its own valid collective law; this phase can move an internally chosen small subset independently of the much larger physical aggregate to which it currently belongs. No native label is needed for recruitment, though a native-informed frozen atlas remains a distinct proposal control.

## Implementation choices to revisit

| Choice | Initial reason | What would justify changing it |
|---|---|---|
| Connected pairs and triples only | Small exhaustive catalogue, direct test of observed solution oligomers | Evidence that larger oligomers dominate blocked productive moves |
| Constant rate per subset size | Internal invariance is transparent | A cheap internal-geometry score predictive of useful accepted exchanges |
| Existing single-handle covariance | Reuses the validated six-dimensional map | Multiple carried bodies consistently narrow or rotate the useful docking basin. Opt-in member charts now let any member dock and rotate about it; covariances and weights are still single-body |
| Exclusion contact for connectivity | Native-blind predicate with exact sphere-union queries | Better geometric recruitment score with an equally explicit reverse law |
| Spherical boundaries first | Matches the currently stalled seeded run | A separate periodic rigid-subset transform and image-consistency audit |
| Rebuild eligible list after changes | Simple correctness baseline | Profiling shows local graph updates or caching are needed |
| Exact whole-union Poisson gate | Retains the many-body physical marginal | Certified geometric bounds can accelerate the same gate without changing its law |
| Fixed duration per phase | Composes with existing invariant sweeps | A validated alternative clock/uniformization implementation is measurably cheaper |

## Validation obligations and interpretation

The useful controls include: internal relative-pose preservation; unique pair/triple enumeration; a dimer selectable before and after external attachment; exact reverse map/Jacobian; full-union depletion limits; deterministic continuation; zero-rate behavior; retained rejected attempts; finite-state stationary distributions with changing total rates; and comparison with the deliberately wrong fixed-event-count control.

A theorem over ideal real-valued states does not certify the sphere-tree predicate, Poisson thinning, transport implementation, or floating-point execution. In particular, a rigid floating-point update can move a contact precisely at threshold across the predicate. The implementation compares the internal exclusion-contact graph at the old and new endpoints and rejects a trial if those predicates change, so a numerically changed eligibility rule cannot silently omit a rate correction. This guard does not prove bitwise geometric reversibility; that remains an implementation obligation. The [Lean development](../formal/README.md) proves both finite-state rates and the general measurable-state bounded-rate, uniformization and Poisson-mixture construction. All 65 audited results pass with the standard Lean axioms only. Reversibility of each fixed-subset physical kernel, bounded measurable rates invariant along its transitions, and the appropriate finite measure are explicit hypotheses. The equivalence of the optimized exponential-event race to the proved Poisson-uniformization construction remains outside Lean, as do executable geometry, random-number generation and floating-point arithmetic.

Performance assessment must count completed contact exchanges, retained internal geometry, contact-fingerprint ESS per CPU, and agreement across independent initial conditions. Accepted oligomer translations alone do not establish improved mixing. A restarted growth run can test whether the new moves address sequestration, but short growth or non-growth alone cannot decide thermodynamic stability.

## Passive subset diagnostic

New event logs include a pre-move `subset_context`: selected size, sorted full
exclusion-component membership, whether the subset is that whole component,
internal and external contact counts, and the number of distinct external
neighbors. This metadata is computed only when move recording is enabled. It
uses the existing contact graph and consumes no random numbers. It never enters
selection, the event clock, or acceptance. The parent can be much larger than the
selected subset. A seed-connected exclusion component need not be a native
registered assembly.

The standalone replay works with earlier logs as well:

```bash
cargo build --release --offline --locked --bin cluster-phase-diagnostic
target/release/cluster-phase-diagnostic \
  --config /path/to/frozen/config.json \
  --trajectory /path/to/frozen/trajectory.jsonl \
  --moves /path/to/frozen/moves.jsonl \
  --out /path/to/new/diagnostic-output
```

Supply complete prefixes ending at the same saved sweep, including the initial
frame and every attempted move. The diagnostic validates old and retained poses,
wall-center changes, every saved snapshot, phase rates, and cumulative cluster
counters. It fails on missing/unsupported move kinds rather than silently
reconstructing an incomplete path. It currently covers the spherical frozen
production move set. It leaves input runs unchanged.

The output separates whole oligomers from embedded subsets, solution components
from those connected to original seed labels, dimers from trimers, and local from
transport trials. `events.jsonl` retains all attempts and both acceptance factors;
`phase-census.jsonl` measures component sizes and eligible channels at each phase
start. Phase-start counts use the fixed sweep schedule; an event-selected
component histogram alone would be weighted by its attempt rate. Proposed
contact changes are scored only for hard-valid candidates with preserved internal
connectivity. Accepted and rejected candidates both remain in denominators.

A larger whole-oligomer supply motivates larger internally connected subsets,
but does not alone establish that their docking proposals will be efficient.
Whole-component status is an observation, not a permissible new eligibility
restriction under the present proof: docking changes that status. Higher-size
channels should retain a symmetric internally determined rate and eligibility in
both directions, including when embedded in a larger aggregate.
