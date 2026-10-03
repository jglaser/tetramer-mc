# Flexible dimer integration: design retained, current experiment stops here

**Do not integrate this proposal into production or launch a trajectory from the
present evidence.** The [physical replay](factorized-dimer-physical.md) completed
all 1,536 frozen reset trials: 971 feasible candidates, 565 geometric nulls and
**zero accepted moves**. The sum of conditional acceptance probabilities was
0.002007806, conditional on the saved candidates and sampled auxiliary clouds.
This is not an equilibrium acceptance estimate or a free-energy measurement.
Across feasible candidates, the median proposal correction was −15.78, the median
Poisson depletion log factor −38.56, and the median combined log ratio −57.59.
The faster geometric search therefore did not produce a useful physical move in
this allocation. No extra clouds, poses or trajectory allocation follow from
these results.

This conclusion concerns the present proposal and growth-diagnostic contexts
(1.4 Å, 0.0275 Å⁻³, 500 μM). It neither rejects the physical model nor resolves
the original 1.5 Å, 0.035 Å⁻³, approximately 106.8 μM assembly question. The notes
below preserve a possible implementation route **if a later, separately
validated proposal earns it**. They describe no currently enabled production
feature.

## Existing subset law can retain a zero selection correction

Production constructs one channel per unordered contacting pair and one per
unique connected triple in
[`ContactGraph::channels`](../src/cluster_phase.rs#L495). A pair has rate

\[
\lambda_{ij}(X)=\kappa_2\,\mathbf1\{\text{exclusion contact between }i,j\}.
\]

It is not selected only when it is an isolated or maximal component. External
attachments can therefore change without changing this pair's eligibility.
The [fixed-duration phase](rigid-subset-phase.md) uses exponential holding times
and selects a channel proportional to its rate; see
[`ClusterPhase::run`](../src/cluster_phase.rs#L854). All rejections consume event
time. The event-indexed chain is not the equilibrium sampling schedule.

The flexible proposal preserves the selected dimer's **internal contact bit**:
its source and every successful endpoint are hard-valid and internally in
exclusion contact. Consequently

\[
\lambda_{ij}(X)=\lambda_{ij}(Y),
\]

although the internal relative pose, patch contacts and native registry can
change. Rigid internal-pose preservation is sufficient for this equality but is
not necessary. A fixed, state-independent probability of choosing a flexible
branch within the dimer channel also cancels in the reverse law. The existing
rigid and trimer branches can remain separate kernels.

Choose the root uniformly from the two retained labels and, initially, one
anchor uniformly from the other N−2 labels. Retain that ordered root/child/anchor
triple when evaluating the reverse move. These probabilities are identical at
both endpoints, giving zero log selection correction. A deterministic label-only
root ordering would also cancel. Do not choose the root by its current contact
quality without adding the corresponding reverse probability.

The sum of *all* eligible channel rates can change after external contacts
change. The fixed-time clock accounts for this; do not add a total-rate MH factor
on top. Conversely, a fixed-event-count sampler choosing uniformly among the
currently eligible pairs would need the ratio of eligible-pair counts. Do not
replace the phase with a fixed event or CPU budget and keep claiming the clock's
balance argument. A cutoff must be a retained fatal failure, not a truncated
physical phase.

## Anchor-dependent uniform draws still need the anchor correction

Production's optional contact-aware law is implemented in
[`anchor_probabilities`](../src/cluster_phase.rs#L346):

\[
q_A(X)=\epsilon/(N-2)+(1-\epsilon)c_A(X)/\sum_B c_B(X),
\]

with a uniform fallback when no boundary contacts exist. Its positive defensive
part keeps a departed partner available in the reverse move. Retaining the same
anchor requires adding

\[
\log q_A(Y)-\log q_A(X)
\]

to the complete proposal correction. The existing production implementation does
this in its learned member branch at
[`anchor_correction`](../src/cluster_phase.rs#L1127).

That production branch also has a separate *world-coordinate uniform kernel*
which draws no anchor. Its omission of an anchor factor cannot be copied to the
new proposal. In `DefensiveDimerProposal`, the root's uniform translation cube is
relative to the chosen anchor, so **every mixture branch uses that anchor**.
The anchor correction must therefore be included even if one or both edges were
drawn from the uniform part. The first integration should use a single uniform
anchor and avoid this extra state-dependent selection altogether.

A retained multi-anchor pool would require the joint probability of the same
ordered pool at both endpoints. The present factorized API takes one anchor; it
does not implement that extension or a source-dependent pool catalogue.

## Carry the flexible endpoint through a flexible physical gate

[`FixedDimerContext`](../src/capped_dimer.rs#L84) snapshots the fixed spectators,
excluding both selected labels. Its anchor is one of those physical spectators,
not a particle to delete from the bath. Rebuild this context after any kernel
moves a spectator. Holding it fixed during one proposal is compatible with
spectators moving during other kernels.

The endpoint consists of independently changed tree coordinates

\[
h_0=g_A^{-1}g_i,\qquad h_1=g_i^{-1}g_j,
\qquad g_i'=g_Ah_0',\quad g_j'=g_i'h_1'.
\]

The complete full-mixture correction is

\[
\log R_F=\log F(h_0)+\log F(h_1)
-\log F(h_0')-\log F(h_1'),
\]

with unit product translation/Haar coordinate Jacobian. The unknown conditioning
constants cancel only at fixed labels, anchor, spectators, atlas, cube and caps;
see [the conditioning argument](factorized-dimer-conditioning.md).

The production call to
[`RigidSubset::new`](../src/cluster_phase.rs#L1087) accepts only a new handle pose
and carries all members with a common isometry. Passing the flexible candidate
through it would overwrite the proposed internal pose. Its whole-subset bath
also cancels the selected union's own volume, which is constant only under a
rigid isometry. That cancellation is invalid when the internal arrangement
changes.

Use the actual two returned poses with
[`SingletonPath::new`](../src/singleton_path.rs#L99), followed by its
[`sample`](../src/singleton_path.rs#L180) method: a fair order coin and two fresh
leg clouds, combined into **one** physical MH decision. The reverse path uses
the reversed order and the same copied intermediate. The intermediate may
violate hard cores or the wall; only the complete physical endpoints are
required to be valid. Do not apply intermediate hard filtering or two separate
MH acceptances. Do not deterministically choose a cheaper path order without
its reverse-selection correction. `FlexibleSubset` provides the alternative
full-union gate, with its different computational cost.

For the single-anchor construction, the proposed physical decision would be

\[
\log U < \min\left(0,\log R_F
+\log\frac{q_A(Y)}{q_A(X)}
+\sum_{\ell=1}^2\log W_\ell\right).
\]

The anchor term vanishes for the uniform-label rule. The path's fair-order
probabilities cancel. There is no extra internal depletion correction, noisy
volume exponentiation, one-factor-per-rigid-member rule or total-rate factor.

The graph's
[`internal_equal`](../src/cluster_phase.rs#L546) check can retain the dimer's sole
contact edge. The rigid fused-oligomer
[`internal_key` equality guard](../src/cluster_phase.rs#L1107) must **not** apply:
changing these offsets is the purpose of this new branch. Its immutable
singleton atlas is different from the existing rigid oligomer catalogue built
from carried internal offsets. That catalogue cannot be reused with an unchanged
normalizer or mixture proof. This argument covers two selected bodies; flexible
trimers would need their own coordinates, graph conditions and proof.

## Composition and the smallest future trajectory control

The existing [frozen assembly bias](frozen-assembly-bias.md) can compose after
each reversible physical dimer decision, as currently done at
[`cluster_phase.rs`](../src/cluster_phase.rs#L1155). Recompute its instantaneous
whole-configuration graph score and restore both selected poses and graph/bias
caches if the bias rejects. Applying a bias only after an entire ordered sweep
does not follow from that elementary-kernel argument. Keep the current
[spherical and frozen-model restrictions](../src/simulation.rs#L291); adding
adaptive mixtures, memory or reversible-jump state would require the full
expanded target and auxiliary correction.

If a later proposal produces sufficiently useful accepted contact changes, the
safest first trajectory would remain standalone: fixed ordered mobile labels,
one fixed anchor, fixed spectators, a frozen atlas/caps, and identical local
single-body schedules in whole-joint and factorized arms. Use independent
streams and predeclared different feasible initial contacts. Other local moves
are needed to break and reform the internal contact: a factorized-only channel
self-loops outside contact and cannot connect that sector by itself.

Measure internal patch or relative-pose basin changes and external contact
fingerprints, retain rejected states, and compare independent initializations
and effective contact samples per total CPU. The internal i–j contact bit is
constant under the flexible branch, so its autocorrelation would not measure
internal reorganization. Instantaneous native registry, if evaluated, remains a
separate observer and must not filter geometry-only proposals.

Whole-joint and factorized samplers have the same success-conditioned endpoint
law, proportional to the full product F on the feasible domain. Staging changes
computational cost and null frequency, not a native-registration preference.
The current physical result provides no basis to proceed to that trajectory;
improving accepted physical contact changes comes first.
