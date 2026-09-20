# Normalized current-geometry GMM closure

This is the minimal tractable online baseline: a deterministic fit to the
**current** pair poses, a normalized finite component-count law, and persistent
Gaussian residual coordinates. It replaces the earlier likelihood posterior
whose configuration-dependent evidence was unavailable. It does not claim to
sample that posterior, to have identified metastable basins, or to have proved
crystal assembly. There is no separate contact bank or irreversible archive.

## State and fitting rule

The physical state `X` contains every mobile rigid body. For spherical boundaries,
`D(X)` is every ordered body-relative pair pose whose center separation is below
one fixed cutoff (default `2 * body_bound + 10 Å`). It is recalculated from `X`;
it is not an auxiliary memory variable. Native contacts are not an input.

For every `K=0,...,K_max`, a reproducible finite-budget fitter returns chart
anchors and full Gaussian means, covariances, and weights. It uses farthest-point
seeds, a fixed number of hard assignment/refit iterations, quaternion averaging,
bounded Cayley fitting residuals, six unit-variance pseudo-observations per
component, and covariance eigenvalue clipping. Empty data/clusters have defined
broad defaults. These are explicit baseline choices; replacing them with a
better deterministic fitter leaves the construction valid. Never warm-start
from hidden history.

In a fixed six-dimensional metric, compute

\[
 S_K(X)=K\log z_b-\log K!+
 \tau\,\overline{\log q_{\hat\Theta_K}(D(X))}
 -\alpha R_s(\hat\Theta_K),\qquad
 p_K(X)=\frac{e^{S_K(X)}}{\sum_{j=0}^{K_{\max}}e^{S_j(X)}}.
\]

`z_b` is an **auxiliary basin activity**, independent of the physical depletant
activity `reservoir_density`. The average log density is zero for empty data.
The score density uses a uniform relative-translation cutoff ball times Haar
rotation, mixed with the fitted Gaussian density; `K=0` is uniform only. Both
terms are normalized on full pose space. This score law is distinct from the
physical proposal's uniform lab-frame cube defense. The score always evaluates
the full Gaussian mixture with its rotation Jacobian, even though fitting is
approximate.

The implemented volume penalty is

\[
 R_s=\sum_j w_j\exp\left[-\frac{s}{12}\log\det\bar C_j\right],
 \quad \bar C_j=C_j/\texttt{translation_scale_A}^2.
\]

`s=6` is the default. `s=0` explicitly **disables** this penalty, including any
constant offset between empty and nonempty models. This determinant proxy is
not an implementation of a complete Besov norm; it distinguishes covariance
volume but not aspect ratio at fixed determinant. Covariance bounds and the
chosen metric still affect fits at `s=0`. Sparse counts arise from the count
law, likelihood and scale penalty together, not from GMM weight normalization.

For `K>0`, the unconstrained coordinate `v` has `28K-1` entries: six means and
21 lower-Cholesky parameters per component, followed by `K-1` weight logits.
Smooth bounded maps and an explicit positive ridge keep decoded covariances
numerically positive definite. Fit eigenvalue bounds and decoder coordinate
bounds are different; see [initialization details](conditional-initialization.md).

Store only `(K,η)`, with

\[
 v=F_K(D(X))+\rho\eta,\qquad
 \eta\sim N(0,I_{28K-1}),\qquad
 \widetilde\pi(X,K,\eta)=\pi(X)p_K(X)\phi(\eta).
\]

`K=0` has no continuous coordinates. Summing over `K` and integrating over `η`
gives exactly `π(X)`. Chart anchors can change deterministically with `X`; the
defined target lives in the retained latent coordinates. This is not an affine
transport between fixed physical rotation charts.

## Balance and scheduling

At a learned physical proposal, retain `K,η` and fit at **both** endpoints. The
log correction is

\[
 \log q_{Y,K,\eta}(X\mid j)-\log q_{X,K,\eta}(Y\mid j)
 +\log p_K(Y)-\log p_K(X).
\]

The spectator anchor `j` is a retained, uniformly sampled move label. Evaluate
the full mixture at each endpoint. An independent uniform pose branch provides
support, and hard/wall invalid proposals are rejected without retries. Combine
this correction with the existing exact conditional Poisson gate for many-body
depletion. This is auxiliary detailed balance, not Metropolis with a noisy
energy estimate.

Local physical proposals also require the count ratio. Physical GCA is
reversible for `π`, so an additional Metropolis test `min(1,p_K(Y)/p_K(X))`
makes it reversible for the retained-latent joint law. Rejected GCA candidates
are rolled back. Center shifts use the same wrapper; relative pair data are
translation-invariant, so their factor is one apart from floating-point error.
There is no extra depletion gate on GCA or on a common center shift.

At each sweep's **end**, with a fixed configured probability, draw `K` from
the finite softmax and redraw all `η` independently from standard normals.
This is exact Gibbs sampling of the declared conditional closure, including
changes of dimension. It needs no approximate RJ acceptance or unknown
evidence. A birth/death kernel is optional future work, not a requirement for
this finite baseline. Composing the individually invariant kernels preserves
the joint law; the entire ordered sweep need not be reversible.

`refresh_probability=0` freezes `K,η`. Each physical kernel still preserves
the joint law, but an arbitrarily initialized fixed auxiliary state generally
does **not** yield the physical marginal along one trajectory. Use positive
refresh probability for ordinary production, and check physical mixing. An
arbitrary prepared initial state and correct invariance do not establish
equilibrium at finite time.

## Running and initialization

```bash
cargo build --locked --release --bin tetramer-mc
target/release/tetramer-mc run \
  --config examples/spherical-conditional.json --method learned \
  --out runs/conditional --sweeps 1000 --sample-every 10
```

No `--model` is supplied. `conditional_closure` is mutually exclusive with the
older `auxiliary_transport`, `reversible_jump`, and `contact_memory` modes.
It currently requires a spherical boundary larger than the body bounding
radius. Useful settings are `k_max`,
`basin_activity`, `score_temperature`, `penalty_strength`,
`covariance_exponent`, `residual_scale`, `fit_iterations`, and `pair_cutoff_A`.
The physical local-move scale, depletant radius/activity and collective
scheduling retain their existing configuration fields.

Initialization choices:

- `random`: conditional Gibbs draw, default.
- `mode_zero`: modal count and zero residuals, exactly the fitted center.
- `zero`: `K=0`, empty residuals.
- `fixed_k_zero`: configured `initial_k` and zero residuals.

The first refresh follows the first physical sweep, so nonrandom initialization
is used. Checkpoints retain `K,η`, physical poses and wall-origin bookkeeping;
resume rebuilds the fit from checkpoint geometry. A compatible previously
learned GMM can initialize the library API through `state_from_components`.
General learned-file CLI import is not included: arbitrary rotational charts
need a fixed-context extension or an explicitly approximate refit. See the
[worked initialization construction and tests](conditional-initialization.md).
The separate [reference-atlas transport mode](atlas-transport.md) now implements
that fixed-context alternative at fixed component count, including CLI model
loading and covariance updates relative to the original learned widths.

Frames and refresh logs record normalized count probabilities, raw scores,
covariance penalties, data counts, and realized auxiliary state. Move logs
separate reverse-proposal and count corrections. GCA/shift records include
`accepted`: their result payload describes the **proposed** transformation,
which replay tools must not apply after rejection. Summary costs include fit
time/calls and initialization time separately.

## Validation and matched pilot

```bash
cargo test --locked --release
python3 research/validate_conditional_initialization.py
python3 tools/run_conditional_campaign.py --out runs/conditional-pilot \
  --replicates 4 --workers 4 --sweeps 200 --sample-every 10
python3 tools/analyze_conditional_campaign.py --campaign runs/conditional-pilot
```

The launcher needs NumPy. It pairs four methods from identical dispersed starts:
closure with `s=0`, closure with `s=6`, local only, and local plus uniform redraws.
Every arm also uses the same GCA/shift schedule. The physical settings are
`rd=1.5 Å`, `z=0.035 Å⁻³`; native information is reserved for post-hoc classification.

Tests cover exact-start AO-sphere stationarity against analytic expectations,
joint latent/count moments, a finite-state negative control that omits count
corrections, actual-runner reverse-density replay, rollback, deterministic
initialization, and exact checkpoint continuation. These establish targeted
correctness checks. A short tetramer pilot measures feasibility and cost;
acceptance or native detections alone are not mixing-time measurements.

See the [completed pilot and validation results](conditional-pilot.md).
