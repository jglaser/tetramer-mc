# Prospective atlas transport inside a flexible surrogate chain

This records the original rigid-atlas-plus-local design and the decision from
the now-completed internal-native audit. **The rigid-only atlas mixture below
has not been promoted or launched.** The revised next candidate is a
[partner-anchored coordinate atlas](#decision-after-the-completed-native-audit),
which directly changes the pair's relative pose. The contact comparison shows no
consistent guided-versus-flat flexible ESS/CPU gain and no external contact in
the 24 flexible chains. It does not establish whether internal native registry
improved. The completed native observer now supplies that separate evidence.

The proposed intervention changes spatial reach: mix a cooperative atlas move
of the current pair with the existing single-member inner moves. Keep the
two-mobile conditional target, fixed spectators and physical endpoint gate.
There is no production integration, new pair selection or change to the
rd=1.4 Å, z=.0275 Å^-3 conditional target. The original finite-system decision
conditions remain rd=1.5 Å, z=.035 Å^-3 and approximately 106.8 μM.

## Why another local mixture is insufficient

The completed rigid comparison observed returns to nonempty external contact
sets: 100 for guided eight-step motion and 97 for flat eight-step motion across
their eight chains. Direct changes between nonempty environments were strongly
concentrated in one stream. Guidance did not consistently improve ESS/CPU.
These are contact-accessibility observations, not independent basin visits.

A random mixture of rigid Gaussian and single-member Gaussian moves could
combine transport and internal flexibility, but remains local. Ignoring
rotations and rejections, eight independent translation increments with
component standard deviation .2 Å give a pair-center displacement standard
deviation of .566 Å per axis for rigid moves, .283 Å for single-member moves,
and .447 Å for a half-rigid/half-single mixture. Rotations add lever-arm motion;
rejections add self-loops, and geometry/score selection can bias the net motion.
Gaussian support is unbounded, so these are unfiltered translation scales,
not bounds on the accepted displacement or its variance.
They offer no reason to expect reliable travel between separated environments
in eight steps. More local steps or higher throughput alone do not resolve
that limitation.

Nor should this repeat the earlier changes of root or retry effort. The
completed two-root proposal accepted fewer collective moves than m4 and made
no nonempty external returns. The earlier m4 redraw changes the root and child
edges separately, restricts that channel to internal contact, and can incur
large depletion penalties despite feasible endpoints. Fixed-singleton fusion
also encountered collisions with the other, unmoved mobile body. Those results
motivate a cooperative relocation mode, without predicting its success.

## Concrete proposed inner kernel

Freeze the pair labels, handle slot zero, all spectator poses, atlas parameters,
defensive branch weight, uniform cube/center, body-frame score cloud and score
strength throughout an outer call. Use the same fixed spectator anchor pool
at both endpoints of every atlas step. The simplest source-independent pool is
the complete set of 262 fixed spectators in label order; it requires no
current-contact selection. Its density-scoring cost must be measured rather
than called intensive. A smaller pool would require a separate frozen choice,
shared across the compared initializations, with its limited coverage declared.

At every inner attempt, independently select a mode with fixed probabilities:

- 1/4: `DockingProposal::propose_members`, using the **current** pair, handle
  slot zero and the immutable spectator pool;
- 3/8: the existing symmetric local proposal on member zero;
- 3/8: the existing symmetric local proposal on member one.

The atlas step carries both bodies rigidly and can place either member against
an atlas contact at a different anchor. It preserves their current relative
pose during that step. Local steps before and after it can change that relative
pose. Reconstruct a proposed pair with `transport_members` from the current
inner pair, using local indices `[0,1]`. Reusing the rigid kernel's outer-source
fiber would erase accepted internal rearrangements and is incorrect here.

Keep the local scales at .2 Å / 1° and the horizon fixed in advance. Every
proposed endpoint is checked against both mobile cores, every fixed spectator
and the wall. Score hard-valid candidates with the same full
`DimerDepletionSurrogate`, including internal overlap and spectator shielding.
The frozen body-frame points define a function of trial poses; their transformed
world positions need not remain fixed.

## Exact proposal correction and common reference measure

`propose_members` selects a state-independent uniform or learned-involution
branch. Treat these as components of the inner kernel. Their fixed branch
probabilities cancel in each paired reverse move. **The helper's learned
correction is not the density ratio of a marginal uniform-plus-learned
independence proposal.** Do not replace it with that different ratio.

For the learned component, `propose_members_learned` selects a source
member/anchor/chart label from its posterior responsibility, selects the target
member and anchor uniformly and its chart from the fixed component weights,
then applies the correlated latent map. `apply_member_trace` combines the
responsibility, destination-label and chart-Jacobian terms. Its returned
`log_reverse_forward` is `log G_h(old) - log G_h(new)`, where `G_h` sums all
member/anchor/chart densities on the current rigid fiber. Use this complete
recorded correction once; a selected Gaussian component's density alone is
insufficient. Preserve the expanded trace for independent arithmetic checks.

The uniform component returns correction zero only when the current and
proposed handle poses lie in the same immutable uniform cube support. The
future wrapper must either prove that this cube covers every hard-valid handle
center allowed by the container or explicitly check reverse support. A source
outside that cube has zero reverse probability for the uniform draw and must
produce a recorded rejection, not be assigned a zero log correction. Freeze
the cube center as well as its side lengths.

Let `mu` be translation Lebesgue measure times normalized rotational Haar
measure for one pose, and let

\[
\nu(dx)=1_D(x)\exp(S(x))\,\mu(dg_0)\mu(dg_1).
\]

This is a measure on the full twelve-dimensional pair pose space. Write
`h = g_0^{-1} g_1`. The change of coordinates from `(g_0,g_1)` to `(g_0,h)`
has unit measure Jacobian: SE(3) is unimodular and left/right rigid transport
preserves its translation/Haar measure. For fixed `h`, the atlas component
acts on the six-dimensional handle fiber and preserves its conditional measure
`1_D(g_0,g_0 h) exp(S(g_0,g_0 h)) mu(dg_0)` after the MH decision

\[
\log A_{\rm inner}=S(y)-S(x)+c_{\rm proposal}(x,y).
\]

Integrating the fiberwise detailed-balance identity over `h` gives reversibility
on the full pair measure, despite the atlas transition being singular in that
space. Each symmetric single-member MH component preserves the same `nu` while
changing `h`. Therefore their state-independent random mixture `K`, and its
fixed power `K^m`, are reversible for `nu`. Neither a deterministic alternating
composition nor stopping after a successful mode is the claimed kernel. This
is a measure argument; floating-point inverse maps and RNG execution retain
their existing implementation obligations.

After the fixed inner horizon, use one `SingletonPath` and
`bounded_path(..., explicit_order=None)` for the actual outer source and final
endpoint. The final decision is

\[
\log A_{\rm outer}=\log F_{\rm bath}+S(x_0)-S(x_m).
\]

Do not add the inner proposal corrections again at the outer gate. The inner
chain already includes them in its transition law. Use the flexible two-leg
bath even if the last inner step was rigid: earlier local steps can change the
pair's internal exclusion union. Do not hard-filter the copied bath
intermediate. Keep the physical state unchanged until this final decision.

If a later all-mobile implementation selects the pair with a nonuniform law
`p_a(X)`, retain its ordered label and add `log p_a(Y)-log p_a(X)` at that same
outer physical decision. Recompute the law at the actual trial endpoint.
The standalone defensive pair selector supplies that term, but is not part of
this fixed-label proposal. State-dependent anchor-pool selection would need its
own corresponding correction; it is deliberately absent from this design.

## Fixed horizon, nulls and failures

Hard rejections, inner MH rejections and documented proposal self-loops consume
one inner attempt. No retries to achieve a successful atlas step, best-state
selection or adaptive horizon are allowed. An identity final endpoint remains
a self-loop without a physical bath.

The existing docking helper turns some map/decoding failures into `None` with a
null trace. Preserve that declared behavior and trace rather than silently
redrawing; its numerical reversibility is an inherited obligation, not a new
proof supplied here. Unexpected wrapper errors, nonfinite score/arithmetic,
inconsistent endpoint geometry and resource exhaustion are fatal. Preserve
the partial begun/step/bath records, leave physical state unchanged and stop
the allocation. Zero reverse proposal support is an ordinary rejection and
must be distinguished from such failures.

## Prospective controls and decision criteria

If the completed audit motivates implementation, compare a direct physical
decision with guided m1, guided m8 and flat m8, all using the same mode mixture.
The direct control makes one candidate from that mixture and applies
`log F_bath + c_proposal` in one decision, with no surrogate filter. Use the
same flexible bath engine so a change of bath factorization does not confound
this control. It is not the existing m4 independent edge redraw.
Freeze identical mode
weights, atlas, pool, defensive weight, local scales, starts and cloud families.
Retain the existing four local physical attempts before the collective slot.
The m8/flat8 contrast tests guidance at equal inner horizon and proposal law;
the m1/m8 contrast also changes the number of local/atlas opportunities and
must report those counts and full CPU. Reuse completed flexible and rigid
controls as contextual references without replaying their old geometry.
Rigid controls use a different physical bath factorization, so their CPU or
acceptance difference does not isolate the inner proposal alone.

The guided m1 arm is a delayed-acceptance control. For a fixed candidate and a coupled
physical-bath realization, its inner and outer acceptance factors obey
`min(1,exp(a))*min(1,exp(b)) <= min(1,exp(a+b))`, with
`a = delta S + c_proposal` and `b = log F_bath - delta S`. It cannot inherently
increase acceptance over directly applying the complete physical decision to
that same proposal. Its possible benefit is avoiding expensive bath work on
poor candidates. At m8, the endpoint proposal itself changes, so there is no
corresponding pointwise acceptance ordering. Neither result guarantees better
sampling per CPU. With an asymmetric atlas, even a flat surrogate still needs
the inner proposal correction. Its inner rejection can remove a docking trial
whose proposal penalty the physical bath would otherwise compensate. Thus a
guided-versus-flat improvement alone would not establish an advantage over
the direct physical control.

The falsifiable claim is improved travel between distinct external contact
environments while preserving useful internal reorganization. Judge it using
nonempty external partner-set changes and returns, occupancy/residence and
initialization disagreement, keeping starts and streams separate. Repeated
breakage/reformation of one edge, empty-state agreement, accepted-move counts
and contact-fingerprint ESS alone are insufficient. Internal native labels
must be evaluated separately after the current audit, rather than inferred
from coarse patch motion.

Keep source-chart support, complete inner proposal correction, hard-invalid
destination counts, score changes, physical correction and CPU separated by
mode. Atlas reach can still fail because the source lies in proposal tails,
the current rigid pair cannot fit candidate destinations, the finite pool
misses useful environments, or the score poorly predicts the physical bath.
Internal local steps remain short and may not repair a deeply misregistered
pair. These are reasons to reject the hypothesis if the future fixed test
fails, rather than automatically increase caps or horizon. No new benchmark
allocation or physical-reference campaign has been frozen or launched. Define
and validate that allocation separately before collecting new samples.

## Source and completed evidence

- [`flexible_surrogate_chain.rs`](../src/flexible_surrogate_chain.rs): random
  member scan, common score, rejected-step accounting and one flexible bath.
- [`rigid_surrogate_chain.rs`](../src/rigid_surrogate_chain.rs) and
  [`rigid_subset.rs`](../src/rigid_subset.rs): handle proposal and rigid transport;
  the former's fixed outer-source fiber cannot be copied into mixed steps.
- [`docking.rs`](../src/docking.rs): `propose_members`,
  `propose_members_learned`, `apply_member_trace`, `draw_member_uniform` and
  `members_log_density` define the proposal semantics above.
- [`depletion_surrogate.rs`](../src/depletion_surrogate.rs): frozen body-frame
  quadrature and full internal/spectator score.
- [`evolving_dimer_benchmark.rs`](../examples/evolving_dimer_benchmark.rs):
  conditional labels, four-local schedule, score policy, RNG roles and journals.
- [`evolving_dimer.rs`](../src/evolving_dimer.rs) and
  [`factorized_dimer.rs`](../src/factorized_dimer.rs): existing m4 correction,
  contact-domain and finite-cap behavior.
- [Completed rigid comparison](rigid-surrogate-benchmark-results-20261004.md),
  [completed evolving-dimer diagnosis](evolving-dimer-benchmark.md),
  [completed two-root/score evidence](dimer-surrogate-inner-chain.md), and
  [internal-native audit scope](surrogate-internal-native-comparison.md).

## Decision after the completed native audit

The [completed audit](surrogate-internal-native-comparison.md#completed-native-audit-and-decision)
finds no native production occupancy in all 24 newly audited prepared-start
rigid/flexible histories. The source-start observations in those new arms
reach only motif 6; guided flexible m8 improves native-presence ESS per CPU over
flat8 in only one of four streams. The reused nonlocal m4 reaches motifs 3, 6
and 7 from some prepared starts. These are accessibility and initialization
results, not equilibrium basin weights.

A common rigid map preserves \(g_0^{-1}g_1\); adding it alone cannot directly
repair the observed internal-registry gap. The smaller, targeted next baseline
is therefore a **partner-anchored coordinate map**. Each inner attempt chooses
one of the two tetramers with probability 1/2, then chooses an atlas proposal
with fixed probability 1/4 or the existing local proposal with probability 3/4.
The atlas call is `propose_members([current[slot]], handle=0, pool=[current[other]])`.
It replaces only the selected tetramer's pose. Both tetramers retain their
supplied rigid intratetramer geometry.

The other tetramer is unchanged during this coordinate move. Consequently its
anchor pose is identical in the paired reverse move even though it changes
between inner steps. Conditional detailed balance at each fixed partner pose
therefore extends to detailed balance on the full twelve-dimensional pair
measure after integrating over the partner. No additional anchor-selection
factor arises for this one-element pool. A pool chosen using the moving
tetramer's current neighborhood would require a different argument and is not
part of this baseline.

Use the helper's complete learned branch correction in
\(\Delta S+\log q_{\rm rev}/q_{\rm fwd}\), inside each inner MH decision.
For its separately labeled uniform branch, explicitly check source and target
support in the same frozen cube. A source outside that cube has zero reverse
support and produces a recorded rejection, with no retry. Null proposals, hard
rejections and MH rejections all consume one step. The random mode/coordinate
mixture and a fixed horizon preserve reversibility of the common
\(1_D\exp S\) target. The final flexible two-singleton bath still receives only
\(S_0-S_m\); no inner proposal correction is added twice.

Retain a **direct physical coordinate-map control**: one proposal from exactly
the same atlas/local/slot mixture followed by one physical decision with bath
weight plus that proposal correction. This control omits the surrogate filter.
Compare direct, guided m1, guided m8 and flat8 with the same atlas, local scales,
starts, spectator geometry and outer local schedule. Guidance has shown no
consistent efficiency advantage, so guided-versus-flat alone is insufficient.
The existing pointwise delayed-acceptance inequality still applies to m1.

The optional methods `FlexibleSurrogateKernel::step_partner_atlas` and
`step_partner_atlas_direct` are implemented. Production defaults, examples,
executable and run allocations are unchanged. No protein benchmark has been
launched. Six focused tests and all nine existing flexible-kernel tests pass:
the conditional map's inverse trace and asymmetric proposal correction,
uniform reverse support, hard/score/null accounting, fixed horizons, fatal-state
atomicity, direct-versus-delayed corrections and default RNG replay are covered.
The direct control also passes with unusable score quadrature, verifying that it
never evaluates that guide or consumes its acceptance RNG. An initial test-only
private-helper import failed at compilation; its source and failed receipt were
retained before replacing it with an independent local-move reconstruction.
[Passing receipt](../results/partner-atlas-validation-20261004/attempt02/validation.json).

The next independent stationarity reference will reuse every one of the 8,192
previously audited IID sphere source poses, without selecting by prior move
outcome or redrawing sources. Its direct, guided m1, guided m8 and flat8 arms
will use fresh proposal/bath streams and retain every rejection. This reference
is under preparation, not a completed validation or efficiency measurement.
After that check, assess
native occupancy and motif exchanges separately by initialization, with full
CPU and absent-state diagnostics. Atlas coverage, hard destination failures,
source-tail penalties and noisy bath costs can still defeat this proposal.
