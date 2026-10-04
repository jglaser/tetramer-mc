# Fixed-context relaxed atlas: preparation prototype

`tools/context_relaxed_atlas.py` prepares a normalized proposal asset from an
existing reciprocal atlas and a **fixed outside environment**. It keeps every
original Gaussian and reciprocal branch, then appends one ordinary Gaussian
for each virtual branch. A deterministic core-overlap minimization moves each
new chart's center. The optimizer never receives the current moving pose or a
native-contact classifier.

This is a preparation prototype, not a production MC change. Synthetic
validation and one fixed protein-context construction are complete. That
construction moved only a small fraction of the blind atlas's probability
mass; no MC or assembly claim follows from it. The objective uses core
geometry; it does not optimize depletion volume.

## The exported law

Let the original atlas have base weights `w_k`, reciprocal flags, and virtual
branches `b`. An ordinary parent contributes one branch of weight `p_b=w_k`;
a reciprocal parent contributes its forward and inverse branches, each with
`p_b=w_k/2`. Every branch is retained, including duplicate or unsuccessful
initializers. With the default child mass `m=0.5`, the output law is

\[
 G_C=(1-m)G_{\mathrm{source}}+m\sum_b p_b H_{b,C}.
\]

The old Gaussian parameters and reciprocal flags remain unchanged. Their
weights receive the same factor `1-m`. The new `H` components have zero mean
in their newly centered charts, explicitly specified covariance, and
`reciprocal_components=false`. We **do not** approximate an inverse Gaussian
by another Gaussian: the exact nonlinear inverse branches remain in the
parent mixture. There is one ordinary child for each such branch.

With a separately retained 50% uniform defensive proposal,

\[
q_C=\tfrac12 U+\tfrac14G_{\mathrm{source}}+\tfrac14 H_C
\geq \tfrac12 q_{\mathrm{baseline}}.
\]

This inequality protects proposal support, not physical occupancy or mixing.
The exporter itself does not add the uniform component; the sampling kernel
must retain it. At `child_mass=0` the source model is returned unchanged and
no context geometry is used.

The initial center of a branch is the existing chart evaluated at its
Gaussian latent mean. For inverse branches the tool applies the exact rigid
inverse `(R.T, -R.T @ t)` after decoding. This center is not generally the mode
of the density with respect to physical translation and Haar measure.

The fixed child covariance is diagonal:

| Coordinates | Variance at `ell=55.02283113084892 Å` |
|---|---:|
| Three translations | `0.0625 Å²` each |
| Three scaled Cayley coordinates | `0.014409934431278726 Å²` each |

Thus the translation standard deviation is `0.25 Å`; the nominal angular
width is `0.25°` (small-angle standard deviation approximately
`0.2500003966°`). This is the existing generic narrow-width baseline, not a
fit to successful centers. For another source `ell`, the angular covariance
is multiplied by `(ell/55.02283113084892)^2` to preserve the physical width.
The values match component 150 of
`examples/frozen-coverage-reciprocal-mixture.json`, SHA-256
`feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e`.
No component-to-native-label association is used.

## Balance and the fixed-context requirement

The decoder follows the production convention:

\[
t=t_0+v_t,\qquad R=\operatorname{Cayley}(v_r/\ell)R_0,
\qquad
J(v)=\frac{\ell^{-3}}{\pi^2(1+|v_r/\ell|^2)^2}.
\]

Here `J` is relative to `d³t` times normalized Haar measure, so an ordinary
Gaussian chart has physical density `phi_6(v)/J(v)`. Exact rigid inversion
preserves this physical measure. All chart and virtual-branch densities must
be included when evaluating the mixture. The optimizer needs no Jacobian:
it constructs fixed distribution parameters from `C` and fixed branch labels;
it does not project a drawn pose or transform the old state into a new one.

For a single moving body with the same outside context and anchor at both
endpoints, an independent draw uses

\[
\alpha(x,y)=\min\!\left(1,
 \frac{\pi(y\mid C)q_C(x)}{\pi(x\mid C)q_C(y)}\right).
\]

The accepted flow is symmetric. Existing validated auxiliary depletion
acceptance must still implement its own complete correction; a noisy weight
cannot simply be substituted into this expression. The validated involutive
transport may likewise use the prepared mixture, retaining its exact chart,
branch and auxiliary-density factors. The exporter does not implement or
validate that MC integration.

The context excludes **every** moving body. If choosing an anchor, moving
body or rigid subset depends on state, that selection's forward/reverse ratio
still belongs in acceptance unless a separately proved symmetric construction
cancels it. Cluster membership must not silently enter the fixed context.
The input schema checks disjoint outside and excluded-moving labels; the
caller remains responsible for truthful labels and atom/pose provenance.

Amortizing a precomputation in a conditional fixed-scaffold test is not a
demonstration of all-mobile economics. When outside bodies move, either use a
valid deterministic reconstruction/cache for the new context at both
directions of the elementary move, or freeze the entire proposal model and
evaluate its actual density. Reusing a stale context-derived model according
to unrecorded history is **not** covered by this argument. Continuous learning
requires its own auxiliary-state balance construction.

A reconstruction/cache key must include canonical outside coordinates and
labels, excluded moving labels, anchor identity, shape and source-model
hashes, all settings, exporter source, numerical runtime, and any boundary
convention. The report records Python/NumPy/SciPy versions; production
integration must bind the actual numerical runtime and its deterministic
behavior more fully. Here coordinates are nonperiodic. No periodic image or
atomic-wall treatment is inferred.

## Geometry and bounded failure behavior

`OutsideContext` transforms fixed bodies into a fixed anchor's frame, builds
one shared KD-tree, and caches maximum fixed radius and center norm.
`RigidIncomingProblem` minimizes

\[
\sum_{i,j}\left[\max\{0,\delta-(|a_i-b_j|-r_i-r_j)\}\right]^2,
\qquad \delta=0.002\,\mathrm{Å}.
\]

Translation and exponential rotation increments have separate unit-ball
constraints, with default scales `1 Å` and `2°`. SLSQP's trial points may
leave these balls while remaining inside its coordinate box. Therefore the
static pair list covers the **whole** `[-1,1]^6` box. Its motion envelope for
each initially rotated atom vector `a_i` is

\[
\sqrt3\,s_t+2|a_i|\sin(\sqrt3\,s_r/2).
\]

Using the actual initially rotated vector norm matters even when the input
rotation is only very slightly nonorthogonal within floating-point admission
tolerance. The search includes a conservative numerical guard and outward
`nextafter` rounding. The guard is an implementation safeguard, not a formal
floating-point proof. Exact geometry predicates remain an integration
obligation.

Each branch has default caps of 32,768 retained pairs, 64 objective evaluations
and 64 optimizer iterations. A pair cap, evaluation cap or unsuccessful
optimizer status preserves the **original center**, with the same child
weight. A successful search selects the lowest objective among its admissible
evaluations, resolving ties by earliest index. A center need not be hard
clear to receive a child: failures and unfavorable branches are never filtered
by a native or clearance label. Full physical hard-wall, core and depletion
checks remain the sampler's responsibility.

Unexpected numerical errors abort the exporter. The CLI flushes a
`child_begun` record before each branch construction and `objective_begun`
before each objective geometry calculation, retaining the attempted prefix.
It writes `failure.json` on catchable failures. External termination can leave
only the journal prefix; a supervising campaign driver must drain and record
the process. Pair caps limit retained objective work, not the temporary result
size of one KD-tree query; the address-space cap also matters in dense cases.

## API and prepared execution contract

The public functions are `branches`, `OutsideContext.from_records`,
`RigidIncomingProblem`, and `build_model`. Input is deliberately restricted to
the explicit `reciprocal-pose-mixture-v1` envelope with an
`anchor-body-relative` base model. Plain model dictionaries are rejected.

The outside-context JSON has exactly:

```json
{
  "schema": "fixed-outside-context-v1",
  "anchor_label": 2,
  "excluded_moving_labels": [99],
  "bodies": [
    {"label": 2, "pose": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]}}
  ]
}
```

Quaternion order is `w,x,y,z`; a `rotation` matrix may replace `orientation`.
The moving shape is supplied separately; the context has no current-moving
pose field. Every outside body uses that same rigid shape. A fictitious
incoming body's new label belongs in `excluded_moving_labels` while all
existing bodies remain outside.

The CLI requires user-supplied `--model`, `--shape`, `--context`, `--allocation`
and a fresh `--out`. The allocation schema is
`fixed-context-relaxed-atlas-allocation-v1`, with exact file SHA-256 fields
`model_sha256`, `shape_sha256`, `context_sha256`, `exporter_sha256`, the complete
`virtual_branches` count, all `Settings` values, and integer positive limits
`workers=1`, `threads=1`, `cpu_seconds`, `wall_seconds`, `memory_bytes`.
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` must be set to
`1` before starting Python. The source model must bind the supplied shape
file hash. Inputs are checked again before successful output.

Outputs are `model.json`, `report.json` and flushed `attempts.jsonl`.
`uses_native_classifier=false` describes this algorithm, not arbitrary input
provenance. A user can supply a native-informed atlas. The proposed blind
source is `examples/frozen-blind-contact-mixture-512.json`, SHA-256
`c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08`:
1024 original components, all reciprocal, hence 2048 virtual branches.
Its full unchanged provenance remains attached. Append-only expansion would
give 3072 stored components and 4096 virtual branches. Existing production
transport preparation can scale quadratically in virtual branch count; this
cost must be measured separately before interpreting any per-step speedup.

## Validation and the next decision

The frozen synthetic validation in
`results/context-relaxed-atlas-preparation-20261004/validation-attempt02/`
passed **15 tests**, using 0.531 CPU seconds and 0.575 wall seconds under
30 CPU seconds, 60 wall seconds, 2 GiB and one-thread caps. The subprocess
was reaped and its process group drained. Attempt 01's earlier 13-test result
is retained. The tests cover exact reciprocal decoding, all branch weights
and flags, independent exported-density reconstruction, zero-child identity,
context label exclusion and deterministic ordering, shared-tree reuse,
cap fallback, finite-difference gradients, full-box pruning, the nonorthogonal
motion-bound edge, synthetic sphere clearance, and failure-prefix retention.
The independent density reader is frozen alongside the tested sources.

These tests establish a small asset-construction implementation, not an MC
stationarity test. The separately frozen protein construction below used all
2048 virtual branches without success filtering.

Before a conditional efficiency benchmark, validate the exported model through
the production loader and matched sphere/depletion references, with full
proposal and physical-density corrections. Any conditional docking capture
ball must lie within the declared atomic-wall domain; it must not be silently
identified with the full vessel. Further benchmarks need matched physical
reference weights. Core clearance improvements alone do not establish
finite-volume basin weight, native binding, or assembly.

Revisitable choices are the core-only objective versus a cheap depletion
guide, the fixed width, trust radii, per-branch budgets and parent/child mass.
Changing them may improve practical coverage; none removes the requirement
for an explicit normalized proposal and the correct physical acceptance law.

## First protein construction: mostly budget-limited

`results/context-relaxed-atlas-quartet1-preparation-20261004-v2/` contains the
completed allocation, journal and exported model. It used all 264 fixed bodies
of the saved blind-growth endpoint, anchor 16, and a new incoming label 264.
The endpoint is from the historical 1.4 Å, activity 0.04 Å⁻³, 500 μM run;
this is not an original-condition physical calculation. Selection of this
quartet context was native-informed. The actual center construction received
only the blind atlas and core geometry, with no native labels or constraints.

Construction took **190.12 CPU seconds / 191.09 wall seconds**, with 16,744
objective evaluations. All original parents were retained and all 2048
children were exported. The independent arithmetic/event audit is in
`results/context-relaxed-atlas-quartet1-audit-20261004/`.

| Branch result | Number | Source probability mass |
|---|---:|---:|
| Pair-list cap; original center retained | 1651 | 85.945% |
| Optimizer status fallback | 197 | 6.959% |
| Evaluation cap fallback | 124 | 4.278% |
| Completed optimizer | 76 | 2.817% |
| Of these, changed/improved centers | 18 | 0.571% |

A cap is a work-limit observation, not a certificate of a hard clash. Fourteen
other branches have better admissible intermediate objectives recorded in
their histories but were returned to their original centers by the declared
failure policy; these represent another 0.571% of source mass. The completed
model has not been retrospectively changed to use those evaluations.

The unchanged-center control matters because adding narrow children changes
the proposal even when no center moves. Let `H_0` use identical child weights
and widths but all original centers, and let `H_1` be the actual export. Since
only 18 centers changed, coupling the identical component labels gives

\[
\|H_1-H_0\|_{\rm TV}\leq 0.0057100863.
\]

With half parent retention and half uniform defense, the corresponding full
proposal difference is at most **0.0014275216**, or **0.143 percentage points**
in any event's absolute proposal probability. This is an upper bound relative
to the same-width unrelaxed-child control, not relative to the original atlas.
It does not bound relative improvement of rare events, mixing, or physical
weights. The audit's `changed-center-mass.json` records the exact weight
arithmetic and matching center checks.

The preceding native-informed incoming-center control achieved a 5.08-fold
geometric coverage gain (see
[its report](native-extension-gaussian-coverage-20261004.md)). This blind
construction has not established an analogous gain. Density checks and a
diagnostic separating conservative pair-list work from actual center clashes
come before promotion into an assembly sampler.
