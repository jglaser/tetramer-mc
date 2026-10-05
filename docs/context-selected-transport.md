# Transport using two context-adjusted charts

`src/context_transport.rs` supplies a proposal-correction building block. It
does not change production kernels, optimize protein poses, sample labels, or
evaluate physical interactions. Its input consists of two already prepared
ordinary Gaussian pose charts and their original atlas labels.

## Conditional construction and balance

Write the physical configuration as `(C,x)`, where `C` contains every unchanged
outside body and the retained coordinate anchor. Let the frozen original atlas
have normalized branch priors `p_b`, densities `g_b⁰`, and
`G⁰(x) = sum_b p_b g_b⁰(x)`. These are the exact virtual branch densities,
including the reciprocal coordinate transformation when present. Select

\[
s(i,j\mid x,C)=\frac{p_i g_i^0(x)}{G^0(x)}p_j.
\]

For each selected label construct a chart `h_b(C)` from the same outside
context and fixed branch initializer. The construction must be independent of
the current moving pose, noise, source/destination role, and unrecorded history.
The same label denotes the same chart in either direction. Unsuccessful
optimization must have a declared deterministic fallback, not a deleted label.
Current-pose warm starts, outcome-filtered labels, and wall-time-dependent
fallbacks are not covered by this construction.

In chart coordinates `x = Ψ_i(z)`, draw a six-dimensional standard normal `ξ`.
The existing `cross_chart_step` applies

\[
(z',\xi')=(\rho z+\sqrt{1-\rho^2}\xi,
           \sqrt{1-\rho^2}z-\rho\xi),\qquad y=\Psi_j(z').
\]

Its reverse swaps labels and supplies `ξ'`. The latent transformation is an
orthogonal involution. If `J_b` is the chart volume relative to translation
volume times normalized rotational Haar measure, its extended correction is

\[
\log R_{\rm map}
=\log[J_j(z')/J_i(z)]+\log[\phi(\xi')/\phi(\xi)].
\]

The full proposal correction is

\[
\boxed{\log R_q=\log R_{\rm map}
 +\log g_j^0(y)-\log g_i^0(x)
 +\log G^0(x)-\log G^0(y).}
\]

The priors cancel between the two ordered label laws. Multiplying the reverse
augmented target by the extended Jacobian gives the forward augmented target
times `exp(log π(y|C) - log π(x|C) + log R_q)`. Taking the smaller of those
two flows establishes accepted-flow symmetry; rejection preserves the remaining
mass. Integrating out labels and noise preserves the physical marginal.

There is no Jacobian for the parameter optimizer: it defines `Ψ_b` conditional
on unchanged `C`, rather than projecting a sampled pose. Even when `C` varies
between MC steps, the full transformation leaves it fixed during this step, so
its Jacobian is block triangular. This argument does not justify adaptive
memory or changing the context during a move.

When `h_b = g_b⁰`, the map and individual branch terms cancel, leaving the
existing `log G⁰(x) - log G⁰(y)` shortcut. Changed charts generally do not permit
that cancellation. No adjusted full-mixture normalizer is needed.

## API and conventions

`apply_selected_charts((source_map, i), (target_map, j), old, noise)` requires
exactly one prepared chart in each map; local chart index zero is internal.
The returned `SelectedChartStep` preserves the global labels and inverse noise.
Its fields cannot be independently replaced. When `i == j`, pass the same
borrowed map object to guard against role-dependent construction.

Call `step.correction(log_priors, weighted_old, weighted_new)` with the complete
ordered ORIGINAL label inventory. A weighted entry means `log(p_b g_b⁰)`, not
`log g_b⁰`. Priors have strictly positive support and finite logs. The method
normalizes them; common offsets are harmless. Weighted scores may be negative
infinity for genuine zero support. Unsupported forward selections, NaN,
positive infinity, and unrepresentable arithmetic are errors. Zero reverse
support returns `OriginalLabelCorrection::ReverseZero`, an explicit rejection
whose serialized record contains no infinity or misleading null density.
Callers must handle that variant before finite-only physical acceptance APIs.

`DockingProposal::branch_log_densities` exposes the original map-factor scores.
The direct-model factor path and exported/refactored map path are distinct;
the actual selector and its scored law must agree. This utility cannot check
whether caller-supplied scores correspond to the reported endpoint poses.

Adjusted children follow the exporter convention: an ordinary physical-pose
chart can originate from either an ordinary or inverse original branch. Its
original global label retains that association, but the adjusted map does not
apply another reciprocal wrapper. An inverse Gaussian is not approximated by
inverting its mean. A new ordinary chart is a deliberately different proposal.

## Integration obligations and scope

Retain the existing separate uniform proposal and all rejected states. Particle,
anchor, and carried-subset selection must be symmetric or contribute their own
ratios. Exclude every moving body from `C`. Preserve the full physical hard/wall
domain and use the existing validated auxiliary depletion acceptance with the
complete `log R_q`; substituting an arbitrary noisy energy estimate is invalid.
Periodic unique-image restrictions, chart seams, geometry predicates, and
floating-point inverse consistency remain implementation obligations.

Only two selected charts need construction, with exact context-keyed reuse when
available. Original mixture scores still cost `O(K)` at each endpoint. Original
responsibilities may poorly match adjusted charts, so lower preparation cost or
better point clearance does not establish higher acceptance or mixing speed.

Ten synthetic unit controls cover pose/noise inversion and correction antisymmetry,
global labels, correlation endpoints, unchanged-chart cancellation, explicit
reciprocal original scores with ordinary children, zero reverse support, and
invalid inputs, and normalization under common log offsets of magnitude `1e300`.
Log probabilities subtract the common maximum before the scaled normalizer;
forming an absolute log normalizer first would lose the normalization at such
offsets. A pointwise-flow control uses `π = G⁰`: the incorrect old
shortcut would accept every changed-map endpoint and violates the augmented
flow identity. All ten tests passed in the isolated archive
`results/context-transport-validation-20261004/attempt02/`, which also compiled
the held `examples/context_transport_stationarity.rs` executable. The receipt
is SHA-256 `1a1d6465c966be37f8a783a48d54718669e60c7e505f21c2f9cf8813cc84ee35`.
Attempt 01 omitted three existing compile-time test fixtures; that failed build
and its logs are retained. Both attempts together used 151.828 CPU and 111.948
wall seconds, below the declared 300/600 limits. The protected production
executable remained unchanged. These build validations ran no statistical draws
or protein queries.

The completed statistical control supplied four independent populations of 4,096
fresh original-mixture sources, three paired move arms, and twelve fixed binary
observables. The valid arms use full corrections with adjusted and unadjusted
ordinary child centers; the inverse-origin unadjusted child is not claimed to
equal its original reciprocal density. The negative arm deliberately uses the
old shortcut. Every source and arm attempt was retained. Numerical residuals
use predeclared scale-aware checks, with raw errors and allowances recorded;
these are diagnostics, not rigorous roundoff bounds.

## Completed synthetic stationarity control

The independently reviewed v3 allocation ran exactly once. Its 16,384 IID
sources yielded 49,152 paired arm attempts and 131,072 journal events. An
independent Python reconstruction checked source draws, exact original reciprocal
branch densities, chart factors and Jacobians, auxiliary noise, ordered label
probabilities, acceptance decisions, rejected-state retention, and every
population tally. All numerical comparisons passed their frozen criteria.

| Arm | Accepted / 16,384 | Prespecified observable rejections / 12 | Smallest paired-test p-value |
|---|---:|---:|---:|
| Adjusted charts, full correction | 3,599 (21.97%) | 0 | 0.13283 |
| Unadjusted ordinary children, full correction | 12,141 (74.10%) | 0 | 0.07467 |
| Adjusted charts, deliberately wrong old shortcut | 16,384 (100%) | 10 | 1.45 × 10⁻⁴⁶ |

The two valid arms jointly used the predeclared Bonferroni threshold
`0.05 / 24 = 0.00208333`. Each test compares paired up/down counts of a fixed
binary observable, conditional on their sum. All four populations and all twelve
negative-control tests remain in the report. The independent audit also found
the missing-factor pointwise discrepancy in every one of the 16,384 wrong-arm
traces. This control therefore distinguishes the implemented correction from
the tempting original-mixture shortcut.

The largest producer inverse-recovery residual was `1.76e-12`; its largest
adjusted/full accepted-flow residual was `9.31e-10`. Some extremely small branch
densities near chart seams have large log magnitudes: independent branch-log
absolute differences reached `0.00772`, while derived correction differences
reached `2.15e-7`. These passed the predeclared magnitude-dependent tolerances;
the result is not a claim of uniformly small absolute log-density error.

Acceptance is reported to expose the different kernels, not as a speedup. The
arbitrary context adjustments were not optimized to this synthetic physical
target. Finite diagnostic nonrejection does not prove invariance, establish
mixing efficiency, or test protein assembly. The formal argument and the
separate implementation obligations above remain necessary.

The exact held plan is
[`context-transport-stationarity-preparation-20261004-v3/execution-plan.json`](../results/context-transport-stationarity-preparation-20261004-v3/execution-plan.json),
SHA-256 `2647eeeb8752e18d054089f6c6e53a3e6fc4f2983d2b15203bcedcf7d1687b30`.
Its [independent report](../results/context-transport-stationarity-preparation-20261004-v3/independent.json)
has SHA-256 `20a1a317f2cacc49871fcaf218f2b9baa2ad212aa9da48c86117d69189780fe4`;
the [complete event journal](../results/context-transport-stationarity-preparation-20261004-v3/producer/events.jsonl)
has SHA-256 `2a4d656ba1e9e3adb58b394bb6a64d0110283b8f03a9f7a3e37163cd9aedf749`.
The earlier v1/v2 preparations remain unlaunched; their pre-data reference
refinements changed neither this allocation nor its numerical thresholds.

Both stages completed and all process groups drained. The host receipt records
80.129 CPU seconds, 81.363 wall seconds and 105 MiB maximum RSS, within the
separate 240 CPU / 420 wall-second allocation. All 684 frozen input bindings
were reverified, and the protected production binary remained unchanged.
The [drain receipt](../results/context-transport-stationarity-preparation-20261004-v3/host-drained.json)
has SHA-256 `0530b1d9d35ca4801f5dde2e7b9ae65979a0f9f915dd7c9d30fe8b22d961c931`.
No protein geometry, bath estimator or assembly trajectory was evaluated.

## Checked algebra bridge

[`SelectedChart.lean`](../formal/ReversibleSampling/SelectedChart.lean) contains
eight checked statements: ordered-label normalization and nonnegativity, prior
cancellation, the full physical/noise/Jacobian/selection ratio, the conditional
matched-chart shortcut, zero-reverse rejection, Jacobian-weighted accepted-flow
symmetry, and kernel completion conditional on integrated balance. The isolated
compile and axiom audit in `results/selected-chart-lean-20261004/attempt02/`
passed using 6.737 CPU and 6.956 wall seconds. The audit reports only `propext`,
`Classical.choice`, and `Quot.sound`. The earlier cache-path failure is retained.

These finite algebraic identities do not prove floating-point inversion,
measurability or change of variables for the implemented charts, physical
geometry, or the Poisson bath gate. The kernel theorem explicitly requires
integrated balance as an assumption; it does not infer it from the Rust source.
