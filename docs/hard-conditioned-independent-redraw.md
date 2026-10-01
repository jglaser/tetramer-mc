# Finite-cap hard-conditioned independent redraw

This optional control targets the finite-width clashes diagnosed in the
singleton benchmark. It changes the proposal, not the protein shape, ideal-bath
target, native classifier or physical measure. The implementation is initially
restricted to singleton channels; no protein efficiency result is implied.

Enable it with `cluster_phase.singleton_independent_max_trials: 8` (or another
positive fixed cap), `singleton_rate > 0`, `dimer_rate: 0`, `trimer_rate: 0`,
`transport_charts: "members"` and a non-null
`anchor_contact_uniform_probability`. Both spectator-nearest and retained
contact-without-replacement anchor pools are supported. Omitting the option, or
setting it to null, preserves the existing proposal and random stream. Uniform
and local branches remain separate. Counters record raw trials, cap exhaustion
and finite-density hard-valid candidates; recorded events retain every raw pose,
target label, latent vector and hard-valid/null result.
The existing `correlation` setting does not enter the independent latent draw;
it continues to describe the correlated transport when that mode is selected.

## Conditional proposal and finite-cap balance

Retain a moving subset, its internal geometry, handle and **ordered anchor
pool** A. Let C include those labels, all spectator poses, wall and frozen atlas.
The same C must be reconstructible at both endpoints. For a singleton this is
particularly direct: its spectators are unchanged by its move.

Let G_C(g) be the complete normalized learned mixture density for the rigid pose
g, with respect to the existing translation × normalized-Haar measure. Include
all member, anchor, reciprocal and fused branches. Let H_C(g) indicate the exact
protein hard-core and atomic-wall constraints, and define

\[
Z_C=\int G_C(g)H_C(g)\,dg.
\]

At most T times, independently draw a new mixture label and a new six-dimensional
standard normal, decode the resulting pose, and check H_C. Return the **first
hard-valid** pose; if none succeeds, return the old state. T is fixed in advance,
or is a deterministic function of invariant C. Do not redraw the pool between
these trials.

Writing

\[
B_T(Z)=\sum_{k=0}^{T-1}(1-Z)^k,
\]

the candidate subdensity and all-failed atom are

\[
q_T(g\mid x,C)=B_T(Z_C)G_C(g)H_C(g),\qquad
p_{\rm fail}=(1-Z_C)^T.
\]

The polynomial form covers Z_C=0 without division: B_T(0)=T, but the candidate
measure has total mass zero. For Z_C>0,
B_T(Z_C)=[1-(1-Z_C)^T]/Z_C. At Z_C=1 the first trial always succeeds. T=0 is the
identity kernel.

The exact renewal mass identity, its probability bounds, the empty-success-set
case and cancellation of a shared accepted-flow factor are checked in
[`CappedIndependent.lean`](../formal/ReversibleSampling/CappedIndependent.lean).
This is an algebraic bridge to the existing balance theorems, not a proof that
the Rust geometry, mixture sampler or floating-point implementation satisfies
all their premises. In particular, identical context and independent complete
mixture trials remain essential implementation obligations.

Both physical endpoints are hard-valid, and **the same B_T(Z_C) cancels** in
their reverse/forward candidate ratio. If p(A|X) is the retained ordered-pool
selection law, the deterministic proposal correction is

\[
\log R_q=\log G_C(x)-\log G_C(y)
 +\log p(A\mid Y)-\log p(A\mid X).
\]

Combine this correction with the existing exact many-body Poisson auxiliary
acceptance construction and optional frozen assembly-bias correction. The proof
does not replace the random Poisson factor by its expectation or interpret its
log as a physical free-energy difference. Conditional accepted flow is symmetric;
summing over pool labels and all auxiliary outcomes, and adding rejection and
all-failed atoms, gives an invariant physical kernel.

For a hard-only target, a valid candidate is still subject to the G_C and pool
corrections. Accepting every hard-valid candidate would sample a conditioned
guide rather than the required physical distribution. Thus conditioning is
neither a hard-only physical bias nor a modification of the target ensemble.

The ordered pool must be retained literally: use the existing
`ContactGraph::anchor_pool_log_probability` at X and Y, including every
without-replacement factor. Do not sort it, rebuild a destination pool, or
substitute the primary-anchor factor alone. For the spectator-nearest pool the
existing primary-anchor correction remains sufficient.

## Uniform defense and invalid variants

The smallest change preserves the existing split: select the constant-probability
uniform branch once, before choosing an anchor. That branch keeps its current
symmetric proposal and does not acquire an unused anchor correction. Only the
learned branch uses the retained pool and capped conditioning. The fixed mixture
of these invariant kernels preserves the physical marginal. Local moves and
the validated involutive transport remain separate kernels.

Alternatively one can condition the complete density
η U+(1−η)G_C, redrawing that branch coin on every trial. This is also valid, but
then the **complete mixture density** must appear in both endpoint corrections,
and the retained pool law belongs to the entire conditional kernel, including
its uniform candidates. It must not inherit the existing special zero-correction
uniform shortcut. The separate-branch design is simpler and preserves current
semantics.

The following shortcuts invalidate the cancellation or change the proposal:

- Retrying within one selected Gaussian. Its unknown hard-valid normalizer
  depends on the component, so the original mixture weights no longer describe
  the conditional mixture.
- Retrying the current correlated transport. Its endpoint law depends on the
  source, hence its hard-valid probability generally does not cancel.
- Continuing after a valid candidate fails the depletion/MH decision. The loop
  must stop at the first hard-valid candidate and retain the resulting rejection.
- Choosing the trial cap from the moving pose, previous failures, observed
  acceptance or elapsed CPU time without deriving the changed stopping law.
- Renormalizing statistics by successful events, or omitting exhausted caps.

## Minimal concrete implementation route

Available APIs already supply most of the construction:

1. `FixedBasinInvolution::decode(chart, z)` returns a pose for a chosen chart and
   independent standard-normal z. `DockingProposal::member_chart_parts()` supplies
   the virtual charts, reciprocal flags and normalized branch log weights.
2. `DockingProposal::members_log_density(members, pool)` evaluates the full
   unfused mixture. `OligomerMixture::log_density(members)` and `log_weights()`
   provide the fused mixture density and selection weights.
3. `OligomerMixture` has private `chart` and `chart_to_g0` helpers. Its new
   `draw_singleton_independent` method chooses a **target only**, decodes its independent latent,
   and transform to the subset's first-member pose. Its stored offsets convert
   that pose to the chosen handle. The unfused `DockingProposal::draw_singleton_independent` counterpart uses the existing
   reciprocal and member/anchor coordinate transforms.
4. Build the catalogue once per retained context, evaluate log G_C(x) once,
   and reuse both across all raw trials. Each raw trial samples fresh labels and
   latents and performs only decode plus exact hard geometry. Evaluate log G_C(y),
   the reverse pool law, graph changes and Poisson gate only for the first valid
   candidate. Cache spectator placements as context, not physical state updates.
5. Expose this as an explicitly named independent-redraw mode with a finite trial
   limit, initially for the constant-rate singleton channel. Default disabled.
   Add counters for raw trials, hard failures, cap exhaustion and valid candidates;
   retain every raw attempt or a reproducible complete trace plus rejected endpoint.

Calling the existing full transport with correlation zero is not the preferred
shortcut. It still draws/encodes a posterior source and constructs an unnecessary
inverse auxiliary trace. Source-dependent numerical failures must not become
hidden state-dependent exclusions in a rejection sampler. Direct target decoding
both removes that ambiguity and avoids repeated full source-density work.

### Rigid oligomers and numerical guards

For an oligomer, carry its actual internal geometry with the existing
`RigidSubset`/`transport_members` implementation. The first-member/handle change
of rigid-pose coordinates has unit Haar × translation Jacobian. Preserve the
rounded internal key that makes the fused catalogue identical at both endpoints;
also preserve the subset's internal contact graph and the current fixed-duration
constant-rate channel construction.

The existing internal-key and internal-graph equality checks are symmetric
endpoint guards. Keep them **after the first hard-valid candidate**, as an
ordinary self-loop on failure. Do not silently classify their failures as another
retry: those checks involve the old endpoint and would require an additional
normalizer argument. A more ambitious design could condition on an invariant
canonical geometry class, but it is unnecessary for the singleton control.

Exact geometry/Haar conventions, reciprocal decoding, catalogue equality,
floating-point carry consistency and representational failures remain explicit
implementation obligations. A direct decode failure caused solely by C and the
new random latent is an invariant proposal-null outcome; source-dependent
failures must not be hidden inside the inner conditioning loop.

## Expected cost and what this cannot fix

For single-trial hard-valid probability p,

\[
P(\text{candidate})=1-(1-p)^T,\qquad
E[\text{raw trials}]=\frac{1-(1-p)^T}{p}.
\]

For illustration, p=0.07 gives:

| T | Candidate probability | Mean raw trials |
|---:|---:|---:|
| 1 | 0.070 | 1.00 |
| 4 | 0.252 | 3.60 |
| 8 | 0.440 | 6.29 |
| 16 | 0.687 | 9.81 |
| 32 | 0.902 | 12.89 |

The recorded represented-source transport rates are about 6.5–7.1%, but these
are **not estimates of p for independent redraw**: its latent distribution is
different. A bounded independent pilot must measure p directly. The table is a
cost scenario, not a speedup prediction.

Repeated hard tests alone do not improve candidates per hard test: that ratio
remains p. The opportunity is amortization. In the fused pilot, catalogue build
timers summed to approximately **56.8 wall seconds**, while the whole kernel
consumed 78.9 process CPU seconds. These are different clocks, so their ratio is
not a measured CPU fraction, although it identifies catalogue construction as a
substantial cost in these single-worker jobs.
Building once and trying several cheap decodes/hard checks can avoid repeating
that expense. Do not rebuild fusion or evaluate the entire mixture per raw draw.
T=1 versus T=8 or T=16 is a bounded initial comparison; larger caps require
measured CPU benefit.

Conditioning removes protein overlaps, not poor depletion overlap or unfavorable
G_C(x)/G_C(y). The recorded one-for-one exchanges often had strongly unfavorable
Poisson factors. This kernel can therefore increase valid exchanges without
increasing accepted physical exchanges. Accepted contact-fingerprint changes and
independent samples per CPU remain the decisive sampling metrics.

An invariant center mask is complementary but weaker: multiply label weights by
a deterministic function of spectator geometry and component centers, renormalize
the **complete** resulting mixture, then apply the same derivation. It can remove
obviously infeasible centers cheaply; it cannot cure the many finite-width clashes
around valid centers. Keep defensive support, and treat an empty retained mask
as an invariant null/fallback decision. No mask may depend on the old moving pose
without an additional reverse construction.

## Validation before a protein efficiency claim

- Finite-state independent reconstruction of the capped kernel for T=0, 1, 8 and
  large T, including Z=0/1, nonuniform guide weights and state-dependent ordered
  pools. Verify accepted-flow symmetry and row sums.
- Independent direct-decode/full-mixture density tests for single, reciprocal,
  member and fused charts; verify chart and pool reconstruction at both endpoints.
- Sphere/depletion and hard-only reference distributions, deterministic restart,
  cap exhaustion, rejected-state retention and disabled-mode compatibility.
- A fixed-allocation singleton comparison at one frozen state, reporting raw
  trials, useful hard-valid destinations, exact physical acceptance and CPU.
  Proceed to trajectory mixing only if that control improves accepted contact
  reorganization. No conditional pilot can establish equilibrium assembly.

### Completed implementation checks (2026-10-01)

Seven focused nonstationary checks passed: explicit configuration guards and old
counter decoding; finite-state capped balance including zero and unit valid
volume; source-independent direct branch/latent draws and reciprocal density;
fused-label density and mixture weights; complete raw-candidate geometry and
ordered-pool reconstruction with bias and deterministic restart; omitted/null
disabled compatibility; and exhausted-cap self-loops with recording-independent
random streams. The all-failure fixture initially requested an invalid zero
defensive weight; its corrected strictly positive fixture passed without a
production change. Both initial and final receipts are retained.

The analytic two-sphere test passed from dispersed and compact starts both with
and without depletion. A separate four-sphere test exercises the complete new
composition: cap 8, fused components, two contact-weighted anchors sampled without
replacement, and the existing Poisson bath. The depletant radius excludes triple
exclusion intersections, giving an independent analytic pair-lens reference.

| Four-sphere observable | Independent reference | Dispersed start | Compact start |
|---|---:|---:|---:|
| Exclusion contacts | 1.03034 ± 0.00309 | 1.03308 ± 0.01075 | 1.04100 ± 0.01116 |
| Largest component | 1.95844 ± 0.00276 | 1.96046 ± 0.00912 | 1.97067 ± 0.01017 |
| Mean squared pair separation | 9.77668 ± 0.00558 | 9.76610 ± 0.02485 | 9.76798 ± 0.02136 |
| Mean quaternion scalar squared | 0.24963 ± 0.00038 | 0.24967 ± 0.00177 | 0.24707 ± 0.00201 |

The chains accepted 1,164 and 1,167 independent moves, including 116 and 100 fused
targets and 579 and 590 moves with nonzero ordered-pool corrections. All four
reference comparisons and between-start comparisons passed their predeclared
five-combined-standard-error tolerances. These are implementation references,
not evidence of protein assembly or a speedup claim. Exact source/executable
hashes and logs are saved in `results/hard-conditioned-validation-20261001`.

Use an isolated target directory for integration-test builds: Cargo also builds
other binaries, even when an individual integration-test target is selected.
The live `target/release/tetramer-mc` executable was left unchanged.
