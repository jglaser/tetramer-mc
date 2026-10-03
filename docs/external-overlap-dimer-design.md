# External-overlap guidance for the conditional dimer benchmark

The minimal optional single-anchor/root proposal helper is implemented in
`src/factorized_dimer.rs`; the union-guided and singleton extensions below remain
designs. No production runner has enabled the new helper and no new simulation
allocation follows. The design preserves the benchmark's **two mobile labeled
tetramers and 262 fixed spectators**, repaired shape, 1.4 Å depletant radius,
and activity 0.0275 Å⁻³.
It does not concern the original finite-system assembly decision conditions.

## What the completed benchmark motivates

The frozen 96-chain analysis is
`results/evolving-dimer-analysis-final-20261003/analysis/analysis.json`, SHA256
`46ced1e7cd2e4e4c928e4eff76a146d87fc92267d4d02c58479dae1aa3db85d2`.
The guided m4 arm improves apparent fingerprint ESS per CPU relative to unguided
in every paired whole-dimer stream, but remains below local in every paired
comparison with finite fingerprint ESS. Initializations still disagree.
In `embedded_9_24/source`, both collective arms accept zero dimer moves and
reproduce local's contact observables while adding substantial CPU. In
`embedded_11_246/source`, m4 produces one partner-environment passage per stream
and no return. These observer results identify unresolved sampling; the
subsequent scalar journal attribution below identifies the dominant recorded
acceptance factor but not the particular geometric interface responsible.

The 61 partner passages and 60 returns in
`embedded_9_24/proposal_prepared/stream0/m4` only alternate between edge `[9,24]`
and no edge. They are same-pair contact formation/breakage, not exchanges to a
different binding partner. Neither these threshold events nor apparent ESS
establish equilibrium or native registry.

The completed [scalar journal attribution](../results/evolving-dimer-rejection-diagnostic-20261003/report.md)
uses the **already recorded** elementary journals separately by context,
initialization and stream. It reports root-cap,
internal-cap and joint-final failures; recorded child/spectator and wall
failures; and the complete-F, internal auxiliary and bath contributions at
physical decisions. Charge all attempts and preserve missing/unmeasured values.
This needs no new poses or clouds. The bath factor alone cannot separate
internal from external volume loss. New point-count diagnostics on saved
candidates would require a separately frozen diagnostic, not an unrecorded
search or a claim inferred from accepted endpoints alone. Its analysis SHA256
is `d4ca76538d144e5c7f628b6307d353e37901ff1806d9ab5e21606054990f9c89`.
For `embedded_9_24/source/m4`, 7,640 of 16,384 production proposals reach a
feasible candidate. Their median full-F correction is −8.80, internal
auxiliary correction +0.061, and bath factor −81.58. The realized sum of
acceptance probabilities is 2.79×10⁻³². Removing only the proposal correction
at these same candidates/clouds leaves 2.61×10⁻²⁴; removing the bath factor
gives 348.7. Those are conditional factor-removal diagnostics, not valid
alternative kernels or predicted physical acceptance rates. They motivate a
controlled overlap-guidance test, but do **not** isolate external overlap loss
from internal overlap loss or identify the fixed anchor as responsible.

## The omitted geometric term is an external union intersection

Let S be the fixed spectator exclusion union and E₀,E₁ the mobile exclusion
sets, each with fixed volume v. Then exactly

\[
 |S\cup E_0\cup E_1|
 = |S|+2v-|E_0\cap E_1|-|S\cap(E_0\cup E_1)|.
\]

Consequently the configuration-dependent ideal-depletant weight is proportional
to exp[z(V_internal + V_external)], where

\[
 V_{\rm internal}=|E_0\cap E_1|,\qquad
 V_{\rm external}=|S\cap(E_0\cup E_1)|.
\]

The current root-body cloud scores internal overlap only. Summing mobile–
spectator pair overlaps is not the external term: it double-counts multiply
covered regions. Both spectator and mobile coverage must be Boolean unions.

Freeze world-coordinate guidance points P inside S independently of both
mobile poses. Define

\[
 K_e(X)=\sum_{p\in P}\mathbf1\{p\in E_0(X)\ \mathrm{or}\ p\in E_1(X)\}.
\]

Every point counts at most once. Uniform or Poisson points in S give an overlap
volume proxy, but guidance correctness holds for **any fixed P**. A fixed
anchor-local window may reduce setup/query cost; then its score measures only
that window, not the full external volume. Its location, intensity, cloud and
spectator union must be fixed before the trajectory, without filtering for
favorable old-pose overlap. Points drawn separately per spectator require a
union construction if the score is to represent union volume. Duplicate points
are mathematically permitted in the auxiliary score but are not a uniform
union-volume estimator.

The external cloud stays in the world frame. Moving it with the root would
define a different score and cannot silently inherit the fixed-world argument.
It is guidance, not the physical bath cloud; the existing many-body Poisson
acceptance estimator remains unchanged.

## Shared normalized threshold law

For integer m≥1, draw m independent uniform integers on 0,…,K(X), and retain
their maximum k once for the entire proposal. Its conditional mass is

\[
 a_m(k\mid K)=\frac{(k+1)^m-k^m}{(K+1)^m},\quad 0\leq k\leq K.
\]

The masses sum to one, so π(X)a_m(k|K(X)) has marginal π. A candidate must have
K(Y)≥k. The exact additional log correction is

\[
 \log R_a=m[\log(K(X)+1)-\log(K(Y)+1)].
\]

If internal and external thresholds are both used, refresh them independently
conditional on the source, retain both across all retries, and add both log
corrections. Their clouds and m values are frozen context. Discard the
thresholds after the one physical decision; unrestricted local moves require
no retained-threshold correction. Existing threshold normalization and
correction lemmas apply; they do not prove the new geometry implementation.

## Option A: external union filter at the complete endpoint

This most directly addresses the missing geometric term while preserving the
existing coordinate map and edge laws. Keep

\[
 h_0=A^{-1}g_0,\quad h_1=g_0^{-1}g_1,\quad
 f_C(X)=F_0(h_0)F_1(h_1),
\]

with the same fixed anchor A, labels, spectators, full defensive densities and
unit translation/Haar Jacobian. Keep `root_stage` filtering root wall/core
validity and `internal_stage` filtering relative core/contact validity and
the existing internal threshold. Add K_e(Y)≥k_e only after assembling both
successful edges and passing the full hard/wall endpoint check. On failure,
**discard both edges**, consume one joint attempt, and start another complete
factorized attempt. Physical rejection ends the move; it does not trigger
another proposal or cloud.

At fixed C,k, the edge success factors are
A_k=s_K₀(p₀)s_K₁(p₁), where each pᵢ depends only on its own edge and the retained
context. Let D_k include all final predicates and d_k=∫f_C 1_D_k. The successful
subdensity after J joint attempts is

\[
 Q_k^{\rm success}(X,dY)
 = A_k s_J(A_k d_k) f_C(Y)\mathbf1_{D_k}(Y)dY.
\]

Its unknown scalar factor is identical in both directions and cancels. The
complete proposal/auxiliary correction is

\[
 \log f_C(X)-\log f_C(Y)+\log R_{a,i}+\log R_{a,e}.
\]

Use it with the existing fair-order two-singleton bath path and **one** MH
decision in `FixedLabelUpdates::dimer`. Do not hard-filter the copied path
intermediate or replace the physical bath by the guidance counts.

**J matters.** At current J=1, a new final-only filter does not search for a
replacement destination. It principally adds a screening/acceptance stage;
there is no basis to expect a sampling improvement from destination search.
For a fixed unaltered candidate law, augmenting with this threshold can only
reduce the marginal acceptance relative to the corresponding unaugmented MH
decision. It may avoid expensive bath work, but useful conditional resampling
requires a predeclared J>1 and must pay for every discarded pair. Compare to an
unguided arm with the same caps, not to a cheaper J=1 control alone.

An optional conservative root-only upper bound can screen impossible endpoints
without breaking factorization. If every exclusion set lies within radius B
of its pose center, internal exclusion contact implies the child center is
within 2B of the root center. All points covered by either mobile body are then
within 3B of the root center. Thus

\[
 K_e(Y)\leq U(h_0)=\#\{p\in P:\|p-g_0.position\|\leq3B\}.
\]

The root filter U(h₀)≥k_e depends on h₀ and fixed context only. Its normalizer
still cancels. This bound may be loose; conservative floating-point handling
must pass ambiguous cases. It is not necessary for the minimum implementation.

## Option B: a smaller root-only external guide

The smaller factorable implementation uses

\[
 K_r(h_0)=\sum_{p\in P}\mathbf1\{p\in E_0(h_0)\}
\]

and its **own** normalized threshold. Add K_r≥k_r to `root_stage` after root
hard validity, retaining the existing root cap 32 and complete-mixture redraw
on every retry. Unlike an endpoint-only J=1 filter, this immediately uses the
existing root-stage retries to search for another qualifying root.
Its success factor s_K₀(p₀(C,k_r)) is independent of h₁ and cancels as before.
Add m_r[log(K_r(X)+1)−log(K_r(Y)+1)] to the existing complete correction.

This preserves the physical conditional target but guides only the root's
overlap with the fixed spectator union. It omits child external contact and
does not equal the exact external union term. Do not use a threshold drawn
from K_e(X) with a root-only lower score: the source may fail that filter.
Either keep fixed root/child labels for a matched control, or select the root
with a fixed state-independent probability retained in the reverse move.
Selecting the currently better-contacting root requires an extra selection
ratio. A root-only result cannot establish that the union-guided move works.

### Smallest reuse control: root overlap with the single fixed anchor

An even smaller implementation can reuse `AuxiliaryOverlapThreshold` without
a new geometry scorer. Freeze a cloud P_A in the **fixed anchor's body-frame**
exclusion set and define K_A as the count also covered by the root. The shape
is shared by anchor and mobile bodies, and the existing methods already do
exactly this: `count_relative(h_0)` with h₀=A⁻¹g₀, and the independent world
check `count_world([A,g_0])`. Use `check_frame` with raw/recovered root edges
and `[A,old_root]` / `[A,reconstructed_root]`; retain the corresponding raw
root-stage count at a proposed endpoint.

This requires a second guide/threshold record on the root stage, separate from
the existing root–child internal guide. Its cloud, fixed anchor identity and
frame must be explicit. Refresh the two thresholds independently, retain both
through every cap/joint attempt, and sum their corrections exactly once. The
existing root cap 32 then searches K_A≥k_A with the same cancellation as Option
B. A fixed anchor-body cloud is equivalent to a fixed world cloud because A
does not move in this conditional benchmark; this is not a cloud transported
with the mobile root.

This scores **one external interface only**. It is a control/proxy, not the
union of all fixed spectators and not a substitute for the many-body physical
gate. Its relevance depends on the recorded rejection diagnostic identifying
this anchor interface as important. If the old root has K_A=0, then k_A=0 and
the threshold filters no root proposals; there is no conditional-search gain
at that source, although the auxiliary correction still applies when K_A(Y)>0.
Do not silently treat that case as guidance-off at only one endpoint. The
original fixed nearest-spectator anchor need not be the important contact in
an evolved environment.

The optional helper API is

```rust
proposal.propose_with_guides(
    proposal_rng,
    &context,
    old,
    Some(&anchor_root_guide),
    Some(root_threshold_rng),
    Some(&root_child_guide),
    Some(internal_threshold_rng),
)
```

Either guide and its corresponding stream may be omitted. Existing `propose`
and `propose_guided` remain the unguided and internal-only interfaces. With no
root guide, new optional fields are omitted from serialization and no new
random draws or count queries occur. Enabled thresholds refresh root first,
then internal, once after source eligibility and nonzero-cap checks and before
any capped draw. Missing enabled RNGs are fatal when that refresh is needed;
an absent guide does not consume a supplied stream. Callers must provide
independent proposal, root-threshold and internal-threshold streams, with
distinct recorded seed/role derivations. Merely passing two mutable RNG objects
cloned or initialized with the same state does not establish the conditional
independence used by the product auxiliary target. Freeze/bind the actual
cloud bytes and their frames separately from this random-stream obligation.

The trace adds `guidance_count` to root draws and `root_guidance` to source/
endpoint frames and the outcome. Existing `guidance` fields retain their
internal-edge meaning. Source and endpoint checks independently compare
relative, recovered, world and reconstructed counts for `[anchor,root]`.
`complete_log_correction()` validates and sums full-F, internal and optional
root auxiliary contributions once; callers must not separately add either
threshold correction again. No physical bath or acceptance logic changed.

## The unsafe tempting shortcut

Putting child–spectator hard validity or external union guidance inside a
retrying `internal_stage` while keeping a newly sampled root creates a success
probability p₁(h₀). The endpoint density then contains
s_K₁(p₁(h₀)), which is different at the old and new roots. The existing full-F
ratio omits it. Discarding the root only after this child cap exhausts does
**not** remove that root-dependent factor.

Safe alternatives are: test coupled predicates once at the joint endpoint and
discard both edges; use one internal draw per root for those coupled tests
(s₁=1) within a fixed joint cap; or derive and evaluate the missing normalizer.
Do not estimate an unknown cap factor with an uncorrected noisy acceptance
ratio. Caps, stage order, cloud, atlas and retry rules cannot adapt to the old
mobile pose, retained candidate, elapsed CPU or acceptance history under the
current cancellation proof.

## Matched two-neighbor single-mobile control

There is also a concrete control that changes only one mobile body at a time.
For moving label i, hold the other mobile label j fixed during this elementary
kernel. Use the retained anchor pool `[j, A]`, where A is the benchmark's
fixed spectator anchor. Both anchor poses are unchanged in the reverse
singleton move. Build the existing `OligomerMixture` with the single member
`[i]`, all other bodies as spectators, and these two anchors. A singleton's
internal offset is identity, so the fused catalogue is a function of the
retained spectators and proposal settings, not the moving pose.

Use matched fused and unfused versions with the same pool, complete defensive
law, local schedule, independent redraw size and finite cap. The frozen
single-anchor atlas and two-anchor single-contact labels are shared; only
the deterministic fused components differ. A fixed label schedule such as
i,j is acceptable: each elementary kernel preserves π even though the ordered
composition need not satisfy detailed balance.

For the minimum capped version, independently redraw the **complete** density
G_C (including its declared defensive component) until all-body hard/wall
validity or a fixed cap. The successful density is
s_K(p_C)G_C(Y)1_hard(Y), giving log G_C(X)−log G_C(Y); no unknown cap normalizer
remains. There is no requirement to preserve the i–j contact. Apply the existing
single-body many-body bath and one acceptance decision. A separately mixed
symmetric uniform kernel is also valid, as in production, but must be held
identical in both arms and documented as a kernel mixture rather than silently
inserted into a different density formula.

The alternative existing correlated involution can be used **once**, with its
validated proposal correction; do not retry correlated proposals until they
pass hard or contact conditions and keep the independent-cap cancellation.
Do not directly invoke the all-mobile production phase with fixed spectators:
its interface rejects that configuration. A small standalone fixed-label
wrapper should reuse the proposal and bath components. Rebuild the singleton
catalogue after the other mobile body moves; it is fixed only within its own
elementary kernel. No current-contact anchor selection or native-label filter
is needed.

## Degeneracies, implementation seams and minimum validation

- K=0 gives k=0 and a well-defined correction. An empty cloud is a constant
  uninformative guide, not evidence of absent physical overlap. Large m can
  suppress downward-count moves; it is not a guarantee of faster exchange.
  Guidance-off is a separate explicit baseline, not the undefined m=0 formula.
- Zero caps give identity. Finite cap exhaustion is a retained self-loop.
  Sources outside the dimer's internal-contact domain remain identity under
  either dimer extension; local/singleton moves connect that sector. Neither
  dimer extension alone is claimed ergodic.
- A successful candidate must pass the same support predicates as its source.
  Old zero full-F density means zero reverse flow and rejection, not a retry.
  Frame/count disagreements, overflow and scoring failures remain fatal with
  partial records, never selectively retried geometric nulls.
- For a spectator-union guide, reuse the integer threshold and correction
  algebra from `auxiliary_overlap_threshold.rs` with a distinct fixed-world
  scorer. The single-anchor control instead reuses its existing geometry
  directly with `[anchor,root]`, with that frame explicitly recorded. Do not
  repurpose a current mobile-root cloud by silently changing its meaning.
  In `factorized_dimer.rs`, record separate old/new external counts, threshold,
  cloud identity, query costs and final-filter dispositions. Extend the single
  `complete_log_correction` accessor rather than applying a second physical
  acceptance. `evolving_dimer.rs` can retain its physical gate unchanged.
- `FixedDimerContext` already snapshots spectators excluding both mobile
  labels. A cloud binding needs that same spectator identity and wall, with
  union-membership preparation separate from state evolution. The scores
  never replace full endpoint hard validity.
- Before any protein experiment, exhaustively check a small discrete joint
  model with nonuniform F and physical weights, both thresholds, zero/full
  support, finite caps and both stage orders. Verify marginal accepted-flow
  symmetry and demonstrate the incorrect retained-root child-retry factor.
  Synthetic sphere tests should cover multiply covered spectator/mobile
  points (count once), K=0, source support, frame boundaries and fatal replay.
  Singleton tests must verify catalogue identity after moving i, and rebuild
  after moving j, alongside the fused/unfused matched density ratios.

The scalar rejection decomposition motivates the implemented single-anchor
root guide as the smallest factorable **proxy control**, without asserting that
its external interface explains the bath penalty. If joint external loss
dominates, use the union filter with a matched, fixed J>1. If child–spectator
constraints dominate, the two-neighbor singleton control tests whether changing
one body within the surrounding environment is more effective than redrawing
both. A large full-F penalty instead points to proposal coverage, which a
count filter does not automatically repair. None of these designs currently
has a measured protein speedup or an assembly conclusion.

## Completed correctness checks

The four exact-rational finite-state checks in
`tools/test_external_overlap_balance.py` passed, with their source-bound receipt
at `results/external-overlap-balance-validation-20261003/validation.json`.
They cover both normalized auxiliary guides, independent threshold refresh,
both stage orders, joint caps and zero/disabled cases, and demonstrate the
failure of the incorrect retained-root child-retry law. This is an exact
discrete analogue, not a proof of floating-point geometry or Rust execution.

All 26 targeted Rust tests also passed: nine new tests in
`tests/factorized_root_guidance.rs`, eight existing factorized-dimer tests and
nine existing internal-threshold tests. The new checks exercise positive root
threshold rejection through every cap, both nonzero auxiliary corrections,
empty/zero-count guides, zero caps and ineligible-source RNG preservation,
exact disabled-path serialization/RNG equivalence, transformed anchor frames,
and fatal shape/frame/missing-stream failures. The source-bound receipt is
`results/factorized-root-guidance-validation-20261003/rust-tests02.json`.
The initial test-only range-literal compilation failure remains recorded in
`rust-tests01.json` and its log; fixing that fixture syntax and adding the
positive-threshold regression did not change the proposal implementation.
The build used isolated `target-validation-line-guide`, four Cargo jobs and
one test thread. No protein queries, physical bath sampling, production runner
changes or assembly runs were involved.

The existing checked Lean building blocks also apply: use
`weight_normalized` / `physical_mass_marginal` and
`weight_ratio` / `log_weight_ratio` for each of the two independently refreshed
thresholds, with auxiliary state U=ℕ×ℕ. Apply
`CappedIndependent.shared_factor_accepted_flow` **once** to the resulting joint
augmented mass, then the existing `accept_reject_correct` and
`conditional_auxiliary_update_preserves_marginal` constructions. This is a
derived paired-guide corollary of the checked building blocks, not a newly
Lean-checked declaration or a proof of the Rust implementation. No new Lean
build or dependency change was performed for this extension.

## Matched anchor-guided extension

The `root_m4` arm in `examples/evolving_dimer_benchmark.rs` now composes the
root and internal guides through `FixedLabelUpdates::dimer_with_guides`.
There is one two-member physical bath evaluation and one MH decision for an
assembled candidate. Both auxiliary corrections enter that decision once.
The old `dimer` entry point passes no root guide and preserves its records
and random streams. Twelve additional integration tests passed, including
disk checkpoint continuation from both kinds of prepared start; their
receipt is `results/evolving-dimer-root-integration-validation-20261003/rust-tests02.json`.

The fixed comparison adds 32 chains against the 32 completed `m4` controls
from `results/evolving-dimer-campaign-20261003`. It reuses the exact same four
contexts, two starts per context, four streams, 32 fixed point banks and
16 prepared starts. Preparation identities remain those of the original
campaign: no replacement starts, new point draws or repeated thinning.
Each new chain contains 512 warmup and 4096 production blocks, with four
local attempts and one collective attempt per block. Thus the new allocation
is 589,824 local and 147,456 collective attempts, including every failure
and self-loop. At most three new chains run concurrently.

The physical setting is the saved 264-body growth configuration at radius
1.4 Å and activity 0.0275 Å⁻³. Only the two specified members move; the other
262 bodies stay fixed. This tests conditional contact reorganization. It
does not test the original 1.5 Å / 0.035 Å⁻³ finite-system assembly objective.

For each block, `root_m4` shares the completed control's `m4/proposal`,
`m4/threshold`, `m4/bath` and `m4/accept` streams and the same local streams.
Only `root_m4/root_threshold` is new. The two threshold streams are distinct;
the same frozen body-frame point cloud is intentionally reused in the
fixed-anchor frame for the root and in the moving-root frame for the child.
The CPU limit rises from 1800 to 3600 seconds per new chain to allow for
the additional count work; the number of attempts and all physical point
budgets remain fixed. A limit failure is retained, without replacement.

This is a single-anchor proxy. It cannot preserve all exclusion overlap
with several spectators, nor guarantee an accessible root destination.
The full many-body physical correction still determines acceptance. The
test asks whether this inexpensive factorable guide improves actual
contact-environment sampling enough to justify extending the geometry.

### Evaluation and independent checks

The new observer classifies the initial state and all 4608 retained block
endpoints in each new chain. Completed control observations and metrics are
reused by hash. No control geometry is recomputed. In addition to the original
patch and partner fingerprints, it reports an external-only edge set that
excludes the moving pair's internal edge. Empty external sets and rejected
residence remain in the sequence. Repeated attachment and detachment of the
same moving pair therefore cannot masquerade as external-contact exchange.
Even an external set change can be simple contact gain/loss; it is not
automatically a new partner, metastable-basin transition or native registry.

Compare external and patch ESS per actual full sampler CPU, passages and
completed returns, and occupancy agreement between starts. Constant
observables have undefined ESS. Keep contexts separate and show all four
streams; their fixed spectator environments define different conditional
targets. Higher acceptance alone is insufficient. The source and deliberately
displaced starts are not independent equilibrium preparations.

The separate Python audit checks threshold support, capped stopping, count
accounting and the complete correction for every collective attempt. Its
independent geometric checks use exactly the first scheduled collective
event after four local attempts in each of the 32 chains, including null
events. It never substitutes the first successful event. This fixed budget
is at most 2560 count queries and 41,943,040 point-membership tests. The
remaining events receive scalar replay checks, not independent geometric
certification. Existing preparation audits authenticate the reused clouds;
they are not rerun. The 12 synthetic auditor controls passed, with receipt
`results/evolving-dimer-root-audit-validation-20261003/tests03.json`.

The preparer, exclusive bounded dispatcher and observer are
`tools/prepare_evolving_dimer_root_extension.py`,
`tools/run_evolving_dimer_root_extension.py` and
`tools/analyze_root_guided_dimer_benchmark.py`. Their frozen campaign records
bind source, executable, inherited preparations, completed controls and
evaluation allocation before physical execution.
