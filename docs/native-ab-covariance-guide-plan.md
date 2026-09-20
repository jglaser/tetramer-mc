# Prepared reuse path for an AB covariance guide

This is a design-only follow-up to the completed independent AB confirmation.
No Gaussian has been fitted by this task and no new physical simulation is
launched. The present centered nested-cover mixture should not receive
another larger budget unchanged.

The largest confirmation contribution is just outside the scale-.2
translation ball: `|t|≈0.441 Å` versus its 0.4 Å radius. Crossing that boundary
removes a high-density component and increases `1/g` by roughly 64 times.
A shifted smooth proposal with translation/rotation covariance can place
density along the observed contact-compatible direction without this
internal cover boundary. It remains an importance proposal, not a change
to the model or to any recorded weight.

## Existing code to reuse

`tools/prepare_importance_guided_normalizer_atlas.py` already implements
importance-weighted six-dimensional means, full covariance, an additive
positive-definite floor, atlas serialization and frozen-fit provenance.
`tools/prepare_smc_normalizer_atlas.py` supplies `arrays`, `relative_poses`,
`convert_model`, and an independent NumPy/SciPy `Density` sampler/evaluator.
The Rust normalized chart law is `FrozenRelativePoseProposal` in
`src/proposal.rs`. `src/normalizer.rs` and `basin-normalizer` already evaluate
the physical full-neighbor exclusion union and divide every contribution
by the full proposal density.

The existing importance fitter cannot be invoked unchanged:

1. It reads `jobs[].directory`, whereas the new confirmation uses
   `jobs[].output`. New population size comes from each population manifest;
   the top-level generic runner jobs do not contain a `samples` field. Read
   physical config and shape from the first job's `provenance/config.json`
   and `provenance/shape.json`, checking all population signatures against
   them; this generic campaign does not put those two filenames at its
   campaign-level `provenance/` path.
2. It selects `hard_valid` rows with `1<q<5`. The native data instead encode
   invalid rows with a `zero` field and use the original `q≤1` target.
   Every one of the **18,577 nonzero native rows** must participate; selecting
   only high weights or only the successful population is not the design.
3. It assumes exactly one fixed neighbor and an existing source atlas/chart.
   This fit must accept AB, keep both fixed poses in the physical signature,
   and construct a specified native chart rather than silently reuse its
   default shoulder chart.
4. Its `candidate_audit` currently checks only `fixed_poses[0]`. An AB
   geometry diagnostic must check both atom unions. This is a limitation of
   that Python diagnostic, not of the Rust full-union normalizer.
5. `tools/run_basin_normalizer_campaign.py` has a historical
   `len(fixed_poses)==1` assertion. The Rust normalizer supports multiple
   fixed neighbors; the wrapper must validate the full supplied AB list
   instead of rejecting it.

A dedicated thin adapter, for example
`tools/prepare_native_confirmation_atlas.py`, can reuse these functions
without changing the existing shoulder-fitting semantics. Its sole input
campaign would be
`runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920`, with its
successful streaming assessment and all original source hashes required.

## Coordinate convention and normalized proposal

Stored quaternions are scalar first `(w,x,y,z)`; SciPy requires `(x,y,z,w)`.
For chosen fixed reference A at `(t_A,R_A)`, the existing atlas convention is

`t_rel=R_Aᵀ(t−t_A)`, `R_rel=R_AᵀR`.

Use the original native reference `(t₀,R₀)` to specify the chart anchor

`t_c=R_Aᵀ(t₀−t_A)`, `R_c=R_AᵀR₀`.

Then the six latent coordinates are

`x=(t_rel−t_c, ell*c)`,
`c=quat_vector(R_rel R_cᵀ)/quat_scalar(R_rel R_cᵀ)`.

This is a **left Cayley residual**: generation uses
`R_rel=Cayley(c) R_c`, not `R_c Cayley(c)`. Existing atlases use
`ell=55.02283113084892 Å`, which puts their scaled Cayley coordinates in
length units. Use that exact value if reusing their covariance routines.
Angles satisfy `|c|=tan(θ/2)`; ell is a chart scale, not an angle in radians.

The native reference relative to A is near a half turn because A itself is
rotated. An identity chart in the A frame would therefore be a poor or
singular choice. The chart above cancels A's reference rotation and is at
most the original 15° native metric limit from every training orientation,
well away from the π seam. Native chart position and orientation define
coordinates; the fitted mean remains free to shift in all six directions.

For a Gaussian density φ₆ in x, the density relative to translation volume
and normalized Haar measure is

`G(t,R)=φ₆(x; μ,Σ) ell³ π² (1+|c|²)²`.

The Haar Jacobian is `dR=dc/[π²(1+|c|²)²]`. The existing Rust and SciPy
evaluators already implement this factor. Do **not** multiply the original
training importance weights by another Jacobian: they already estimate the
physical target measure, and mapping those weighted poses into x gives the
required weighted latent sample.

An equivalent construction fits in a laboratory chart centered at `(t₀,R₀)`
and then uses `convert_model`. It rotates the six-vector and its full
covariance with `diag(R_Aᵀ,R_Aᵀ)`. Translation–rotation cross blocks must be
kept; diagonal-only fitting would discard the desired coupling.

## Fit and regularization

Read all eight original populations and retain their unconditional counts.
With equal population sizes, normalize all original `log_importance_weight`
values jointly using log-sum-exp. Their common `−log N` factor cancels.
Do not renormalize each population to equal weight: that would discard the
information in their unequal estimated masses. More generally an explicitly
defined pooled estimator must account for unequal budgets.

For normalized weights w_i, fit

`μ=Σ_i w_i x_i`,
`Σ_raw=Σ_i w_i (x_i−μ)(x_i−μ)ᵀ`.

These are proposal-fitting moments. Their self-normalization and finite
sample concentration prevent them from being a reliable equilibrium
covariance estimate. The effective training size is only **3.97**, despite
18,577 nonzero rows, and the largest training weight is 47.72%. Regularization
is therefore essential. Use a stated positive covariance floor or shrinkage,
preserve all six-dimensional cross terms, and report raw/final eigenvalues
and condition number. The existing fitter's additive-floor machinery is
reusable; its old shoulder covariance is not automatically an appropriate
floor for AB. A new physical-unit floor must be declared explicitly rather
than inherited unnoticed.

A leave-one-population-out weighted predictive-density check would expose
how much the dominant population controls μ and Σ. It is a training
diagnostic, not an independent mass estimate. Freeze the final all-population
fit and its floor before any new normalization draws. Tempered weights or
additional broad components would be separate declared proposal choices;
they must not replace the original physical weights.

## Runtime reuse and defensive support

The least Rust work is to feed the frozen Gaussian atlas to the existing
`basin-normalizer`. Its positive uniform centered cube/Haar branch is already
normalized and covers the original capture sphere; retain it. The estimator
will still classify the original native region and evaluate the complete
AB exclusion union. The Python fitter/launcher changes above are required,
but the underlying physical estimator needs no change.

There is a proposal-efficiency issue: the existing normalizer chooses an
anchor uniformly from all physical neighbors and averages their full
densities. An atlas fitted relative to A will therefore also be applied at
B, generally proposing a different world pose. This is correct but can waste
roughly half the learned attempts. The smallest optional Rust refinement is
an explicit proposal-anchor index/list independent of the physical neighbor
list. Proposal generation and density evaluation must use exactly that same
list; `Environment.fixed` must remain the complete AB list. The default
all-neighbor behavior should remain unchanged and be replay-tested.

For this native-only target, the existing outer native cover would be a
more efficient defensive law than the entire 36 Å cube with unrestricted
orientation. A later composition can use

`g_new = δ g_cover + (1−δ) g_existing_Gaussian_plus_cube`, `0<δ<1`.

Both branches are independently normalized, so this directly reuses the
existing Gaussian sampler and full density without exposing its private
draw internals or setting its required uniform weight to zero. All containing
branches enter the final denominator, including the small cube branch inside
the existing model. Draws outside the original q≤1 target remain zeros; do
not condition the Gaussian on that target using an unknown truncation
constant. This composition requires a small proposal adapter in the native
normalizer plus a matching independent density audit; it has not been
implemented here.

Before a fresh fixed-budget comparison, test chart round trips, lab/anchor
frame density equality, finite positive covariance, component and defensive
mixture normalization, sphere known-volume controls, original-q preservation,
and full AB hard checks. The confirmation serves as **training only** for that
new guide. Fresh seeds and a newly frozen budget must assess mass, contribution
concentration and cost. No old high-weight draw is recalculated or replaced.
This remains a native-informed integration control, not template-free basin
discovery or an assembly demonstration.
