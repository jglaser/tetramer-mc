# Independent full-R4 SMC control

This new executable integrates the **same original R4**, repaired shape, fixed
physical neighbors, capture support and bath as its supplied frozen regional
configuration. It uses uniform R4 or an explicitly corrected two-chart
initialization, a fixed activity schedule and local physical-pose mutation. It does not change any native definition, fit a
Gaussian guide, or launch a protein experiment. No protein allocation is frozen
by the CLI defaults or this implementation document.

The executable retains the physical-activity path as its default and offers the
proposal-density path described below through `--bridge proposal-density`.

The purpose is an independent check on native mass, its old-R5 intersection and
its native complement. Shared Gaussian centers can have shared coverage blind
spots even when their fresh importance samples use independent streams. SMC
can move from its initial poses along hard-valid contact continuations. Neither
method establishes coverage of unseen disconnected regions by itself.

## Target, initialization and exact-arithmetic identity

Write the fixed physical measure as `mu(dx)=d3t dHaar(R)`, with normalized proper
SO(3) Haar measure. Set `h=I_capture I_R4 H_hard` and retain every physical
neighbor in the exclusion union defining `C(x)`. For a fixed, prespecified
schedule `0=z_0<...<z_T=z`, the unnormalized measures are

```
gamma_t(dx) = h(x) exp(z_t C(x)) mu(dx).
```

At zero final activity every physical activity is zero; the prespecified
fraction schedule still increases strictly from zero to one.

Draw **M unconditional** original latent coordinates `u~q=1/V6(4)`. The chart
returns `x(u)` and

```
J(u) = det(L) / [ell^3 pi^2 (1+|Cayley|^2)^2],
a(u) = h(x(u)) J(u)/q(u),
Zhat_0 = sum_(j=1)^M a_j / M.
```

Invalid attempts have zero weight and remain in M. Systematic resampling draws
N particles from the positive a's; an initial family ID is the original
unconditional draw index. Conditional on the original draws,

```
E[Zhat_0 eta_0^N(f)] = (1/M) sum_j a_j f(x_j).
```

Consequently its unconditional expectation is `gamma_0(f)`. If every a is zero,
the population finishes with Z=0 and no terminal particles. There is no refill,
restart-until-hit or replacement seed. A zero-hit result is a valid zero
estimator, not proof of zero physical mass.

At each stage, for each current particle, sample fresh independent nonnegative
Poisson factors with conditional mean `G_t(x)=exp((z_t-z_(t-1)) C(x))` using
[`overlap_weight`](../src/overlap_weight.rs). The configured number of factors
are averaged in **linear** weight space. With `Ghat_i` their means, update

```
Zhat_t = Zhat_(t-1) * mean_(i=1)^N Ghat_i,
```

then systematically resample with probabilities proportional to Ghat. Each
offspring receives the same fixed mutation-kernel distribution and number of
moves, with independent fresh random streams. The kernel `K_t` preserves
`gamma_t`; it does not depend on the potential noise. Offspring kernels must
not be chosen from observed weights or empirical population statistics without
an additional argument. A fixed finite composition of invariant kernels is
sufficient; perfect mixing is unnecessary for this expectation identity.

For a test function f, the conditional expectation after resampling and
mutation is

```
E[Zhat_t eta_t^N(f) | prior particles, potential draws]
    = (Zhat_(t-1)/N) sum_i Ghat_i K_t f(x_i).
```

This uses the unbiased **offspring counts** of systematic resampling. It does
not claim each ordered offspring separately has the weighted parent law.
Averaging the fresh potential draws, applying induction to the deterministic
function `G_t K_t f`, and using invariance gives

```
E[Zhat_t eta_t^N(f)]
    = gamma_(t-1)(G_t K_t f)
    = gamma_t(K_t f)
    = gamma_t(f).
```

This is a random-potential Feynman–Kac construction. It requires no persistent
potential-noise state because the physical mutation kernel targets gamma_t
independently. It is not Metropolis acceptance using a newly regenerated noisy
ratio of absolute weights. The normalizer and terminal empirical measure must
be used **jointly**: a fixed region B is estimated by
`Zhat_T * mean(1_B(terminal_pose))`, including zero populations in comparisons.
A terminal fraction alone, or a ratio of estimated masses, is not an unbiased
physical probability.

The derivation assumes exact geometry, correct RNG laws, exact arithmetic and
unbiased resampling. The current Lean project proves importance-sampling and
MCMC bridges, **not this SMC identity**. The derivation and executable reference
checks supplement rather than extend the checked Lean theorem count.

## Optional independent geometric initializer

`--initial-reference-region PATH` enables a mixture with the complete unfiltered
ball of that reference chart. `--initial-current-probability` defaults to 0.5
when a reference is supplied and one otherwise; it must remain positive. The
reference shape and physical neighbors are checked and its exact bytes hashed.
Its outer radius and chart define the proposal. Its old q thresholds, capture
support and inner radius **do not** restrict or renormalize the proposal.

For the two chart coordinates and Jacobians at the same physical pose,

```
p_initial(x) = alpha I_current_ball(x)/(V4 J_current(x))
             + (1-alpha) I_reference_ball(x)/(Vref J_reference(x)),
a(x) = H_current_capture,hard,R4(x) / p_initial(x).
```

Both component densities enter every weight regardless of the chosen branch.
There is no additional J multiplier when dividing by this physical density.
A reference draw outside current R4/capture or hard support remains a zero in
the original M. No reference q failure is a reason to reject a draw or alter its
weight. At a chart's Cayley seam that finite ball has zero component density;
other inverse-map numerical failures stop the run. Each initialization record
includes the selected chart, both inverse coordinates/Jacobians when defined,
both ball indicators and the full physical log density.

This proposal uses the unchanged geometric reference chart rather than fitted
Gaussian centers. It introduces initial attempted poses in the known tiny
hard-volume pocket while leaving current R4 support complete. **Under the default
physical-activity path, immediate H/g resampling mostly removes those ancestors
again.** The observed R5 hard fraction is approximately 2.4e-6 of hard R4, so
N=2048 gives only about 0.0049 expected retained R5 descendants under that initial
physical target. The mixture improves Z0 estimation; it does not by itself solve
annealing coverage. The optional density bridge below retains proposal influence
early and removes it by the physical endpoint.
The initialization identity above applies with `a=H/p_initial` in physical
coordinates. The independent two-chart sphere test uses shifted means, unequal
translation/angular scales and angular lengths, explicitly incompatible old
q/capture/inner-radius masks, and exterior-current-R4 zero attempts.

## Optional fixed proposal-density bridge

For `--bridge proposal-density`, keep the normalized **full physical** geometric
mixture density g frozen throughout. Use the fixed reference measure
`mu=d3t dHaar` with translation in Å. All density powers and logarithms refer to
that same numerical measure convention; no latent density or partial component
density can replace g.

```
gamma_beta(dx) = h(x) g(x)^(1-beta) exp(beta*z*C(x)) mu(dx),
0=beta_0<...<beta_T=1.
```

The initial target is h*g. Unconditional draws from g have initial weights h,
so `Zhat_0=hits/M`, and equal-weight resampling among valid initial draws retains
the proposal allocation conditionally on target validity. Exterior-current-R4,
capture-invalid and hard-invalid attempts still remain in M. The initial file's
`log_initial_weight` is the actual resampling weight (zero for each hit in log
space); `log_hard_weight` separately retains the physical H/g diagnostic.

The random incremental factor is

```
Ghat_beta(x) = g(x)^(-delta_beta) * mean_cloud What(delta_beta*z*C(x)).
```

Its conditional expectation is exactly the ratio of the next and previous
unnormalized densities. The same fixed-schedule unnormalized-measure induction
above therefore applies. The terminal density at beta=1 is the original physical
h*exp(zC), independently of the initial geometric proposal.

Each symmetric physical-coordinate mutation uses the combined log acceptance

```
min(0, gained_lost_log_gate_at_beta_z
       + (1-beta)*(log g(proposed)-log g(retained))).
```

The deterministic factor tilts the existing physical Poisson count gate to the
bridge target; the old log g cache changes only on acceptance. Mutation scales,
mixture density, fraction schedule and sweep counts stay fixed. At **zero final
activity**, clouds and physical gates are one, but the g^(-delta_beta) incremental
factors and (1-beta) mutation corrections still apply. Skipping those stages
would retain h*g instead of the physical hard target.

Each potential row records log g and its deterministic correction. This mode
also records every hard/capture/R4-valid mutation's old/proposed pose and log g,
deterministic correction, full gained/lost gate, log uniform and acceptance.
Invalid proposals are retained in aggregate rejection counters and never acquire
an acceptance weight. The zero-activity two-chart test independently recomputes
both physical densities and every saved mutation correction; positive-activity
sphere tests also check the resulting physical normalizer and a nonconstant
terminal indicator.

The density bridge addresses loss of the geometric pocket during initialization,
but it can still lose families or fail to explore disconnected continuations.
Only fresh independent populations can assess its numerical precision and
coverage. Neither bridge authorizes protein sampling or advances the assembly
gates by implementation alone.

## Physical-coordinate mutation

The driver reuses [`docking::local`](../src/docking.rs): an unwrapped symmetric
translation or a symmetric left-composed Cayley rotation. The rotation law is
symmetric relative to Haar measure, including its even Cayley Jacobian.
Capture-invalid, R4-invalid and hard-invalid candidates reject. The existing
[`depletion::sample`](../src/depletion.rs) gained/lost-count gate then targets
the current physical activity. Its auxiliary intensity comes from the original
configuration's `poisson_lambda_ratio`; incremental potential clouds instead
use `lambda_ratio * delta_activity`.

There is **no additional Jacobian multiplier in physical-coordinate mutation**;
the optional density bridge instead includes the explicitly specified full-g ratio.
A symmetric random walk in whitened u would need the ratio of physical
Jacobians; that is not the implemented proposal. The existing docking runner's
move arithmetic remains unchanged; only crate visibility of its helper changes.

## Records and operational behavior

`latent-region-smc` and `latent_region::smc::run` produce a fresh output directory:

- `provenance/` binds the original config, region, shape and build-embedded Rust
  source bundle. The manifest hashes the executable and every physical input.
- `initialization.jsonl` retains every attempted pose, selected/reference/current latent coordinates,
  proposal/Jacobian factor, hard/capture flag and zero or positive hard weight.
- `stages.jsonl` retains initial parent draw IDs and particles; subsequent stages
  contain every pre-resampling pose, independent cloud/count record, incremental
  weight, normalizer increment, systematic offset/parent array, mutated endpoint
  and inherited initial ancestor. Initial-family counts and concentration are
  retained at every stage.
- `summary.json` includes the terminal poses and normalizer, genealogy and hashes
  of the raw records. `status.json` distinguishes completion, a completed zero
  estimate, and a failed/interrupted calculation through completion flags and
  the summary. Existing output directories are refused; there is no resume path.

These records permit independent reconstruction of initialization normalization,
all incremental factors, resampling ancestry, normalizer products, endpoint
support and later fixed terminal indicators. Under the default physical-activity path mutation counts are aggregate records;
individual proposal/gained-lost acceptance events are not saved. The density
bridge additionally saves every target-valid proposal, density correction and
gate decision. Invalid proposals retain aggregate rejection counts.

RNG streams are indexed by seed, stage, offspring/draw index, cloud/sweep index
and distinct roles. Two offspring with one ancestor receive different fresh
streams. An underflowed positive scaled weight causes explicit numerical failure;
it is not silently turned into an invalid zero. Unrepresentable linear Z is null
while finite log Z remains available. Floating-point CDF rounding and other
ordinary arithmetic still lie outside the exact-arithmetic identity.

One population is single-threaded. A controller can run independently seeded
populations concurrently while respecting the experiment's process/worker cap.
Fixed budgets, seeds, scales and the complete source closure must be frozen
before protein sampling. The executable itself neither adapts nor retries.
The command-line bridge default remains `physical-activity` for an explicit
baseline; it is not a recommendation to allocate protein sampling to that path.

## Focused validation and progression

[`tests/latent_region_smc.rs`](../tests/latent_region_smc.rs) compares positive
activity one-sphere normalizers and a nonconstant terminal contact indicator
with independent radial/Haar quadrature across eight populations. M differs from
N, the Haar Jacobian varies strongly, the capture boundary is active and clouds
are stochastic. The tests reconstruct every first-population potential,
normalizer and resampling parent from saved records. Separate zero-activity,
zero-hit, fixed-target rejection and API/CLI deterministic-record controls are
included. A small resampling test checks zero-weight boundary behavior and
unbiased offspring counts over an exact offset grid.

Complete native classification remains an external unchanged observer. Total
R4/native SMC can help check both native pieces without first porting that
observer into every mutation. The tiny observed no-entry/native ratio means an
empty no-entry terminal count is expected at modest N; independent class-balanced
importance sampling must continue to estimate that mass.

Track R5 initial ancestors and replenishment, both native pieces' whole-population
mass errors and family concentration. A resampling weight ESS near N or many
terminal descendants is not independent contact ESS. Uniform initialization
can miss the tiny hard-volume part of a high-weight pocket, and local moves can
fail to repair that loss. Agreement between independent methods is useful
regional evidence, not a bound on undiscovered full-vessel contacts.

The [regional roadmap](contact-evidence-roadmap.md) still gates the separate
full-vessel mixture, its outside-R4 remainder and the later all-mobile finite
assembly controls. No old physical run, executable, classifier or completed
audit is changed or replayed by this implementation.

## Completed implementation checks

The six new integration tests passed with
`cargo test --offline --release --test latent_region_smc -- --nocapture`.
They ran only synthetic sphere controls and deterministic record checks; no
protein SMC population or existing physical audit was launched.

| Path and toy activity | Total estimate ± independent-population SE | Exact total | Terminal contact estimate ± population SE | Exact contact |
|---|---:|---:|---:|---:|
| Physical activity, z=4 | 34.8580 ± 0.5665 | 34.8595 | 15.8206 ± 0.3900 | 15.5604 |
| Proposal density, z=4 | 34.0198 ± 1.1310 | 34.8595 | 15.4653 ± 0.9056 | 15.5604 |
| Two-chart proposal density, z=0 | 25.8084 ± 0.9710 | 26.7696 | 7.3030 ± 0.2952 | 7.4705 |

Each line uses eight independently seeded SMC populations. The test tolerances
are reference-check tolerances, not protein convergence criteria. The separate
nonphysical systematic-resampling test covers the exact zero-offset boundary,
expected offspring counts over a fixed uniform-offset grid and rejection of
positive scaled-weight underflow.
