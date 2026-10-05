# Absolute physical weights from the fixed-context candidate bank

This is a research note, not a sampling allocation or a convergence result. No new poses, geometry queries, Poisson clouds, or physical estimates were generated for this note.

**Implementation update:** the separate [saved-candidate preflight](context-candidate-bank.md)
is now implemented and complete. All 36,864 candidates and a 512-pose independent
panel passed; neither proposal covers the source secondary-contact pocket.
The absolute-cloud stage remains unimplemented and unallocated. The design below
is retained to explain the density and estimator; its prospective implementation
statements refer to that original design stage.

The proposed check concerns one mobile tetramer, fixed spectators `C`, and fixed anchor 16, at depletant radius 1.5 Å and activity 0.035 Å⁻³. The neighborhood comes from the historical 500 μM snapshot. It is not the all-mobile 106.8 μM assembly system. Its A/B labels are the exact instantaneous neighbor sets `{16,217}` and `{16,56}`; Other contains every remaining physically valid pose, including unbound poses. None of these labels asserts native registry.

## Why the zero-correlation candidates are independent draws

The frozen pilot uses the posterior-involution proposal with correlation `rho=0`. For a fixed context and anchor, let `p_b(C)` be the normalized prior of virtual chart `b`, including reciprocal labels, and let `g_b` be its normalized density in physical pose measure. Define

\[
G_C(x)=\sum_b p_b(C)g_b(x).
\]

The learned branch draws a source label with probability `p_i g_i(x)/G_C(x)`, a destination label independently with probability `p_j`, and independent standard Gaussian noise `epsilon`. The map computes

\[
z_y=\rho z_x+\sqrt{1-\rho^2}\,\epsilon.
\]

At zero correlation, `z_y=epsilon`: the destination depends only on its label and fresh noise. Summing the source posterior gives one. Therefore its candidate density is `G_C(y)`, independently of the current pose and source label. Encoding the old pose remains necessary for the inverse trace and acceptance bookkeeping, but does not affect this mathematical destination law.

With the separate uniform coin, the unconditional tested-slot candidate density is

\[
Q_C(y)=\tfrac12\,\mathbf1_{\mathrm{cube}}(y)/V_{\mathrm{cube}}
       +\tfrac12\,G_C(y).
\]

This statement requires the unchanged context, fixed anchor/prior, open-space chart convention, and zero correlation. It does not apply to the local slots, nonzero-correlation maps, or a periodic implementation that discards exterior images. Floating-point source-chart or density failures remain explicit failures, never a reason to redraw or silently manufacture a zero contribution.

The kernel deliberately accepts its uniform and learned branches separately. Its stored learned acceptance correction is `log G_C(old)-log G_C(new)`; the uniform correction is zero. That valid mixture of reversible kernels does **not** supply `log Q_C` for importance sampling. Every bank candidate must be scored under the complete `Q_C`, irrespective of the branch that generated it.

## Measure, reciprocal branches, and density reconstruction

Use `d³t` times **normalized Haar measure** on SO(3). If a map chart decodes standard Gaussian latent `z` through its actual lower factor `L`, angular scale `ell`, and Cayley vector `u`, its volume element is

\[
J_b(z)=\frac{|\det L_b|}{\ell^3\pi^2(1+|u|^2)^2},
\qquad g_b(y)=\phi_6(z)/J_b(z).
\]

A Cayley chart covers SO(3) except its half-turn seam, which has Haar measure zero; it does not restrict the proposal to a small angular neighborhood. Reciprocal branches apply exact SE(3) inversion, preserving translation volume times Haar measure. Composition with the fixed anchor also has unit Jacobian. Quaternion double covering does not introduce another factor of two.

Reconstruct the **compiled map's factors** once, with its frozen covariance-loading convention. Do not replace them with the direct model's separately refactored factors or an ordinary Gaussian centered at a reciprocal mean. `DockingProposal::member_chart_parts()` exposes the map, reciprocal flags and effective priors; `FixedBasinInvolution::checked_log_density()` distinguishes an exact seam from numerical failures. The saved learned-branch `full_new_gaussian_log_density` can be cross-checked, but uniform candidates also require a learned-density evaluation. Once `Q_C` is expressed in physical measure, there is no additional Jacobian in `W/Q_C`.

## Fixed inventory and statistical units

The pilot fixes eight chains per proposal arm: two certified starts and four streams, each with 2,304 cycles. Original-prior and context-prior arms each contribute exactly **18,432 tested-slot candidates**, or **36,864 total**. All 256 warmup cycles can be retained here: candidate independence does not require equilibration of the accepted trajectory. The local control's tested slot has a state-dependent proposal and is outside this bank.

Freeze the complete declared candidate inventory, input hashes, region definitions and new cloud-role seeds before inspecting physical weights. Keep all candidates from every declared tested slot, including rejected and hard-invalid proposals. Never select accepted poses, successful docking events, favorable streams, or only contact candidates. Local slots are excluded by the proposal definition, not by observed geometry. A missing/fatal numerical proposal or incomplete declared bank is unresolved; it is not a hard-invalid zero and must not be replaced.

The runner seeds each proposal/bath/acceptance role separately by master seed, cycle and slot. Original and context arms share the same start/stream seeds, so their banks are paired, not independent populations. The eight start/stream seeds are distinct. Preserve that pairing in comparisons and uncertainty estimates; do not concatenate trajectories into fictitious independent replicates. Future absolute clouds need a new, frozen namespace indexed by arm, start, stream, cycle and cloud replicate. The existing acceptance clouds and uniforms are not reused.

Conditional on a frozen bank, new clouds remove only cloud noise. They do not remove its pose-integration error. This is an independent physical-weight calculation on reused proposal data, not an entirely independent proposal dataset.

## Absolute many-body depletion estimator

For moving exclusion region `E(y)` and the union of all fixed exclusions, define

\[
O_C(y)=|E(y)\cap\bigcup_{c\in C}E(c)|.
\]

The conditional physical weight is `h_C(y)=1_wall,core exp(z O_C(y))`. This uses the same fixed-shape baseline as the existing normalizers: factors involving the fixed exclusion union and the constant isolated moving exclusion volume cancel from conditional probabilities. Such constants cannot be discarded when comparing different contexts or shapes without accounting for them.

The reusable implementation is [`src/overlap_weight.rs`](../src/overlap_weight.rs): `OverlapEnvelope::build` and `sample_with_envelope`. A disjoint geometric decomposition supplies an inner overlap volume `L` and an uncertain envelope. Exact point thinning gives

\[
K\sim\operatorname{Poisson}(\lambda[O_C-L]),\qquad
W=e^{zL}(1+z/\lambda)^K,
\]
\[
\mathbb E[W\mid y]=e^{zO_C(y)},\qquad
\frac{\operatorname{Var}(W\mid y)}{\mathbb E[W\mid y]^2}
=e^{z^2[O_C(y)-L]/\lambda}-1.
\]

At `lambda=64z`, the exponent is `z(O_C-L)/64`. Averaging `m` independent complete clouds divides this conditional relative variance by `m`. Better certified lower volumes can reduce cloud variance without changing the expectation. The existing geometric certification uses guarded floating-point predicates, not a formal interval-arithmetic proof.

For region `R`, use the linear mass estimator

\[
\widehat M_R=\frac1N\sum_{n=1}^N
 \frac{\mathbf1_{\mathrm{wall,core}}(y_n)\mathbf1_R(y_n)\overline W_n}{Q_C(y_n)}.
\]

Hard-invalid or wall-invalid proposals contribute exactly zero and remain in `N`; no cloud is needed for them. A/B/Other partition all valid configurations and their masses sum to the total. Compute free-energy differences from these linear masses, never from an average of `log W`. The gained/lost acceptance-gate weight is an auxiliary **ratio** and is unsuitable as an absolute Boltzmann estimator.

## What can be reused, and what is missing

The absolute kernel is already used by [`src/normalizer.rs`](../src/normalizer.rs), [`src/native_region.rs`](../src/native_region.rs), and [`src/latent_region.rs`](../src/latent_region.rs). Their cloud-averaging and unconditional-denominator conventions are useful precedents. The current `basin-normalizer` runner is not a ready-made bank scorer: it generates poses using the direct-model law and carries capture/native-region machinery. Those conventions must not silently replace this bank's map law, full atomic-wall domain, or A/B/Other definitions.

A future small adapter would need immutable bank extraction/reconstruction, strict full-mixture density auditing, the complete fixed exclusion union, the existing instantaneous neighbor classifier, independent absolute clouds, and population summaries. It must add bounded planned-cloud work and retained failure prefixes: `sample_with_envelope` itself currently draws a count and runs its point loop without the new physical runner's per-leg/campaign budget protocol. Oversized or partial clouds must not be clipped, dropped, or retried. This note adds neither the adapter nor a campaign allocation.

## Coverage and variance limitations

The immutable cube encloses every atomic-wall-valid origin, so the uniform component supplies positive density throughout the physical domain. Gaussian tails outside the wall remain proposal mass with zero target weight. This guarantees support, not useful finite-sample coverage.

For one arm with `N` IID candidates and `m` clouds per valid candidate, the variance includes

\[
\frac1N\left[\int_D\frac{\mathbf1_R e^{2zO_C}}{Q_C}
 \left(1+\frac{e^{z^2(O_C-L)/\lambda}-1}{m}\right)d\mu-M_R^2\right].
\]

Increasing cloud intensity or replication addresses the bracket's cloud multiplier. It cannot fix a small proposal density on a high-weight contact basin. A single-anchor atlas and a vessel-wide uniform draw may miss relevant pockets around other spectators. Other can therefore remain poorly estimated even if A/B are sampled well. Zero hits do not bound a region's physical weight without additional geometric/weight bounds.

Inspect independent population spread, largest contributions, importance ESS, cloud-replicate variation and complete A/B/Other contributions. Finite-bank consistency alone cannot certify unseen modes. This check could distinguish rare conditional environments from ineffective accepted-state proposals; it cannot establish finite-system assembly or crystal stability.

The completed pilot adds a specific coverage concern: trajectories can reach
the same A neighbor IDs while retaining different surface-patch contacts.
Any future absolute-weight calculation must retain patch-resolved coverage
diagnostics within A/B/Other, including the strongly retained saved-source
patch environment. Agreement of coarse A/B masses alone would not resolve
that missing within-region coverage. A region with no sampled contribution
cannot be declared absent on that basis.

## Possible refinement: accept the full proposal mixture

At zero correlation, the same candidate generator also permits an alternative exact kernel: accept **every global candidate** using `log Q_C(old)-log Q_C(new)` plus the existing bath factor, instead of retaining separate uniform/learned acceptance labels. Candidate generation, reciprocal maps and the sampled physical gate remain unchanged. This marginalizes the proposal labels for acceptance; it does not add another Jacobian or source-label correction.

The existing singleton gate depends on the fixed context and its two physical endpoints, not on the uniform/learned label. Given gained/lost overlap volumes `v+`, `v-`, its thinned counts are independent Poisson variables with means `lambda v+` and `(lambda+z) v-`. Swapping counts and endpoints gives the augmented target/proposal ratio

\[
\kappa=(1+z/\lambda)^{k_+-k_-}.
\]

The physical overlap-volume exponential cancels the ratio of the Poisson normalization factors. Zero-volume cases use their degenerate count laws, without division by a volume or zero probability. This is the identity formalized in [`ConditionalPoisson.lean`](../formal/ReversibleSampling/ConditionalPoisson.lean). Its application still requires correct thinning, geometry and complete clouds; resource failures cannot become ordinary rejections.

For fixed valid endpoints `x,y` and fixed gained/lost counts, write `q_U=U`, `q_G=G_C`, and `w_U=w_G=1/2`. After factoring out the common forward physical density and forward count probability, the branch-separated accepted flow is

\[
F_{\mathrm{separate}}=\sum_b w_b\min\{q_b(y),\kappa q_b(x)\},
\]

whereas the marginalized kernel gives

\[
F_{\mathrm{full}}=\min\{\sum_b w_bq_b(y),\kappa\sum_b w_bq_b(x)\}
\ \geq\ F_{\mathrm{separate}}.
\]

The inequality is the sum-of-minima bound. It remains valid with zero component support. Summing over the common endpoint-only count law proves off-diagonal accepted-flow dominance; the same Poisson balance identity proves reversibility of both kernels. Thus the full-mixture **elementary global kernel** Peskun-dominates the separate-branch kernel, with the usual stationary asymptotic-variance interpretation where applicable. It does not imply a better decision for every individually realized branch.

For physical endpoints inside this runner's uniform cube, `U(x)=U(y)=u`. Improvement is strict precisely when the two nonzero branch ratios `kappa` and `kappa G_C(x)/G_C(y)` lie strictly on opposite sides of one. In particular, when `kappa=1`, the total off-diagonal flows are **equal**, even though individual branch acceptance decisions can change. A small learned density at the source is therefore not sufficient evidence of a net improvement. Favorable depletion canceled by an excessive learned reverse-density penalty is one relevant regime; a dominant unfavorable bath factor may remain limiting.

This refinement is specific to the independent candidate law. Nonzero correlation requires the actual conditional transition density. General use must include the uniform cube indicator in `Q`; here all wall-valid endpoints lie inside the enclosing cube and invalid endpoints retain zero physical weight. If an anchor were randomized and marginalized too, its probabilities and all anchor densities would also enter `Q`.

The current `physical_step` has no correction-override mode. A future explicit, checkpointed zero-correlation mode could retain the original trace correction and record the full-mixture correction separately. Learned traces already store both endpoint `log G_C` values: with `a=log G_C(x)-log u`, `b=log G_C(y)-log u`, the new correction is `softplus(a)-softplus(b)`. Saved complete bath counts and uniforms permit a one-step counterfactual at the saved source. Uniform candidates require additional map-density evaluations to measure their compensating changes. Such a replay is not a new trajectory or a mixing estimate.

Extra scoring cost and the deterministic four-local/one-global schedule matter: Peskun ordering of the reversible global kernel alone does not establish ordering of that full cycle composition or improvement per CPU. Saved correction/bath diagnostics should precede implementation and a matched benchmark. No refinement has been implemented or allocated by this note.

## Minimum bank adapter and preflight gates — original design

The smallest useful implementation is a separate bank scorer, with no change
to the production sampler. It would consume all 36,864 global candidates from
the completed **rho=0** original/context runs, including warmup and rejected
proposals. The local and rho=0.95 runs are not candidate sources. Admission
must authenticate every chain's completion receipt, compiled source bundle,
config, prior, model, shape, fixed context, events and exact 2,304-slot
inventory. Missing candidates or numerical failures stop the calculation;
they do not become zero weights or replacement draws.

The existing APIs are sufficient for density reconstruction:
[`DockingProposal::member_chart_parts`](../src/docking.rs) supplies the actual
map, virtual reciprocal flags and effective priors;
[`FixedBasinInvolution::checked_log_density`](../src/basin_involution.rs)
distinguishes the legitimate Cayley seam from arithmetic failure. Compose
each saved pose with the inverse fixed anchor, apply each branch's reciprocal
wrapper, and sum its weighted physical density. Include the immutable cube
indicator when forming `log Q = logaddexp(log(U/2), log(G/2))`, for **both**
generating branches. Check saved learned `log G(new)` against reconstruction.
The score already includes the map factor and normalized-Haar Jacobian: the
mass contribution is `mean(W)/Q`, with no extra Jacobian or source-label factor.

Use two stages, with their inventories and resource ceilings frozen before
each stage:

1. **Whole-bank geometry and coverage preflight, without Poisson clouds.**
   Preserve one record per slot, reconstruct Q, check the full atomic wall
   and rigid core against the saved verdict, and classify every valid
   candidate with the existing native-blind patch observer. Keep all fixed
   bodies in the physical definition. A conservative body-bound cutoff may
   omit provably distant spectators from point tests, using the guarded
   `2*(tree.bound + rd)` construction already used in
   [`RigidSubset::new`](../src/rigid_subset.rs); validate this optimization
   against the unpruned union in synthetic tests. Build one
   `OverlapEnvelope` per valid candidate and record its lower and uncertain
   volumes and construction cost. An empty overlap has W=1 exactly.
2. **Independent absolute weights, only after reviewing the complete
   preflight.** Use that same pose/context/envelope for two independent
   clouds at lambda=64z. An explicit new SHA-256 seed namespace must include
   the bank/protocol digest, arm, start, stream, cycle and cloud index 0/1.
    Neither old acceptance clouds nor proposal RNG state is reused. Retain
   both individual estimates and their linear average, represented with
   log-sum-exp to avoid overflow. A preflight decision may stop the whole
   calculation as insufficiently covered; it may not remove unfavorable
   slots or populations from a subsequent estimate.

Patch regions should be fixed from the **certified initial source geometry**,
not from tokens chosen because another trajectory missed them. Let T be that
source pose's complete secondary-neighbor-217 patch-token set. Within A,
record `c=|P217(y) intersection T|/|T|` in the exhaustive bins
`[0,.25)`, `[.25,.5)`, `[.5,.75)`, `[.75,1)`, and `{1}`; the last is the
source-secondary-contact inclusion region and permits additional tokens.
Also retain the Jaccard similarity, complete sorted tokens and full
fingerprint. B and Other remain complete, with Other split into contacted
and unbound poses. These definitions must be serialized before examining
bank coverage. They are source-informed retrospective diagnostics on reused
candidate data, not a prospectively independent discovery experiment.
Confirmatory region choices informed by these bank results would require
fresh candidates.

Zero bank hits in the source-secondary region would be a decisive
**coverage failure**, not an upper bound on its physical mass. Sparse hits,
large importance tails, or disagreement between starts/populations likewise
prevent coarse A/B agreement from resolving the missing patch pocket. Report
every patch stratum and the full complement; do not substitute neighbor-ID
agreement for surface registration. The current trajectories' disagreement
in secondary-neighbor patches makes this failure mode plausible.

The only necessary sampling addition is a small bounded counterpart of
[`sample_with_envelope`](../src/overlap_weight.rs). It should preserve the
existing Poisson draw, cell selection, membership predicates and arithmetic
bit for bit when no limit is reached. Journal the cloud start and planned
Poisson count **before** its point loop; reject an over-budget allocation
fatally before processing points. Track processed/retained counts and CPU
periodically, preserve partial progress on failure, and publish a usable W
only for a complete cloud. Set explicit per-cloud, per-candidate and total
raw/retained-count limits, plus CPU/wall/memory limits. No clipping, early
acceptance, retries or partial-cloud weights are allowed. A sibling wrapper
keeps currently pinned production sources untouched; its equivalence tests
must compare both output and RNG continuation with the existing sampler.
The envelope's pose binding is private in the current API. A sibling adapter
must therefore own construction together with its immutable pose and context,
rather than accept arbitrary external envelopes. If the two stages run in
separate processes, rebuild deterministically, check against the preflight's
bound metadata, and share that checked envelope between the two clouds.

Useful new tests are the bounded/unbounded equivalence and failure-prefix
cases, two-cloud role separation and deterministic continuation, complete
bank/invalid-zero accounting, full-Q cube boundaries and reciprocal/Haar
density reconstruction, and patch-partition completeness. Reuse the existing
analytic lens first/second-moment, duplicate-neighbor union, rotated-dumbbell,
zero-activity and empty-overlap checks in
[`tests/overlap_weight.rs`](../tests/overlap_weight.rs); do not mistake the
acceptance-ratio reference for an absolute-weight reference. The same bank
also yields a hard-only `1/Q` control without new clouds.

Keep eight distinct start/stream populations per arm, each with denominator
2,304. Arms share proposal-role seeds and must be compared as paired
populations; clouds use independent roles. Cloud-replicate disagreement
measures only conditional cloud noise. Report population spread, importance
ESS, largest contributions, linear masses and their free-energy ratios,
without treating the two clouds as independent pose populations.

Cost is measurable before cloud scoring. Full density reconstruction requires
75,497,472 virtual-chart evaluations. At two clouds and lambda=2.24 Å^-3,
expected raw-point work is `4.48 * sum(uncertain_volume)` over valid poses;
the two copies reuse each envelope. The existing point loop stores no cloud
and short-circuits membership in the fixed union. However,
`Environment::contains` scans its retained spectator list, so conservative
spectator pruning matters. Acceptance-gate throughput is not a validated
runtime prediction for absolute envelopes. Record density, geometry, envelope
and point-loop costs separately. The preflight's volume totals and variance
bounds, rather than a guessed runtime or favorable subset, should determine
whether a separately frozen full-bank cloud allocation is worthwhile.
