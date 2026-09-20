# Frozen native AB covariance guide

The offline adapter `tools/prepare_native_confirmation_atlas.py` now freezes
a regularized Gaussian guide from the completed independent AB confirmation.
All **18,577 original nonzero observations from all eight populations** enter
the fit with their original importance weights. No new physical simulation,
depletant clouds, or replacement of selected old weights was performed by
this fitting task. The present centered nested-cover mixture should not
receive another larger budget unchanged.

The frozen output is `runs/native-ab-covariance-guide-20260920`. Its
`predeclared-fit-protocol.json` was written before the fit, and `provenance/`
contains the executed source, all original weighted training rows, source
campaign/assessment manifests, physical config, and an exact byte copy of
the physical shape. `report.json` records the input population hashes,
geometry checks, coordinate audit, and model hashes. `cross-validation.json`
records the eight leave-one-population-out fits and selection diagnostics.

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

A dedicated thin adapter,
`tools/prepare_native_confirmation_atlas.py`, reuses these functions
without changing the existing shoulder-fitting semantics. Its sole input
campaign is
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

A leave-one-population-out weighted predictive-density check exposes
how much the dominant population controls μ and Σ. It is a training
diagnostic, not an independent mass estimate. Freeze the final all-population
fit and its floor before any new normalization draws. Tempered weights or
additional broad components would be separate declared proposal choices;
they must not replace the original physical weights.

The implemented floor is additive,

`F=diag(0.05² I₃, [ell tan(0.1°/2)]² I₃)`.

The angular value specifies a nominal axis-angle scale in this local chart;
it is not a claim that the nonlinear angle variance is exactly 0.1° squared.
The two predeclared candidate families are `N(μ,Σ_raw+F)` and
`0.8 N(μ,Σ_raw+F) + 0.2 N(μ,4(Σ_raw+F))`. The latter is selected only if
at least six of eight heldout gains are positive, their mean after dropping
the largest gain is at least 0.1 nat, and the pooled-mass-weighted gain is
positive. This is a deterministic conservative design rule, not a statistical
significance test. The common single-Gaussian branch bounds every pointwise
mixture gain below by `log(0.8)`; the code checks this invariant.

The observed gains, in population order, were
`[-0.07310, 0.09496, -0.01331, -0.01134, 0.43419, 0.36883, 0.26312, 2.61556]`
nats. Five of eight were positive, so the frozen selected model remains
**one full-covariance Gaussian**. The drop-best mean was +0.15191 nat and
the pooled-mass gain +0.07821 nat; neither overrides the failed six-fold rule.

The all-population latent mean is
`[0.232240, -0.181965, 0.197286, -0.078320, -0.135429, 0.398485] Å`
in the specified deployment chart. The raw eigenvalues were
`[0.000436, 0.000555, 0.002410, 0.007910, 0.011083, 0.041316] Å²`,
and after the floor they are
`[0.002849, 0.003018, 0.004883, 0.010281, 0.013418, 0.043678] Å²`,
with condition number 15.33. All translation–rotation cross terms remain.
The floor materially affects the two narrowest directions; ESS=3.97 still
precludes claiming that this is a well-determined equilibrium covariance.

A second frozen file, `model-wide.json`, multiplies **every selected
covariance by four**, including the floor, without changing any mean,
anchor, component weight or original model. This is a prespecified width
sensitivity control for fresh integration, not a new data-dependent fit.

| File | SHA256 |
| --- | --- |
| `model.json` | `86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667` |
| `model-wide.json` | `3fe2a1280d8f3965fb7582f81b429a9902a3ef5f59b5e3766098d5a3ba9de952` |

## Runtime reuse and defensive support

One Rust reuse path is to feed the frozen Gaussian atlas to the existing
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

For this native-only target, the new runtime composition uses the existing
outer native cover as the primary defense, while retaining the positive
36 Å cube/Haar branch of the existing atlas:

`g_new = δ g_cover + (1−δ) g_existing_Gaussian_plus_cube`, `0<δ<1`.

Both branches are independently normalized, so this directly reuses the
existing Gaussian sampler and full density without exposing its private
draw internals or setting its required uniform weight to zero. All containing
branches enter the final denominator, including the small cube branch inside
the existing model. Draws outside the original q≤1 target remain zeros; do
not condition the Gaussian on that target using an unknown truncation
constant. The parent task owns the Rust adapter and independent runtime
density audit. These fitter outputs use the existing atlas file format;
there is no fitter-specific Rust serialization convention. The intended
runtime choices are `δ=0.25`, inner cube weight `ε=0.05`, selected anchor
index 0 (A), and the original full-size outer geometric cover. Both fixed
neighbors remain in the physical hard and exclusion-union calculation.

Before a fresh fixed-budget comparison, test chart round trips, lab/anchor
frame density equality, finite positive covariance, component and defensive
mixture normalization, sphere known-volume controls, original-q preservation,
and full AB hard checks. The confirmation serves as **training only** for that
new guide. Fresh seeds and a newly frozen budget must assess mass, contribution
concentration and cost. No old high-weight draw is recalculated or replaced.
This remains a native-informed integration control, not template-free basin
discovery or an assembly demonstration.

## Validation and cost screen completed

Run the seven independent fitter controls with

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xvg/protein-nucleation/.venv/bin/python -B tools/test_native_confirmation_atlas.py
```

They cover pooled original weights and cross covariance, rank-zero positive
floor with physical units, a noncommuting left-Cayley round trip, Haar
density under a rotated/translated frame, heldout-population exclusion,
the conservative family-selection rule, both-neighbor unequal-radius atom
checks against brute force, and the original max-member native metric.
All pass. On all 18,577 actual training poses, the laboratory and deployment
frame log densities agree within `8.6e-13`; the independent SciPy density
agrees to the same tolerance. The maximum training chart angle is 4.015°,
well away from the Cayley seam. The original native q metric is recomputed
independently for every training pose.

A separate exact atomic KD-tree predicate checks every moving atom against
each fixed radius class, separately for both A and B. It confirms the
128 uniformly chosen training poses and 16 largest-weight poses, with
smallest gaps 0.001196 Å to A and 0.000420 Å to B. These are static
predicate checks, not re-estimates of physical weights.

The independent proposal geometry screen used 512 new Gaussian draws per
model, checking both neighbors and original native/capture criteria. All
draws met native q and capture; hard support limited success:

| Frozen guide | Native/hard-valid | Fraction (binomial SE) | Hybrid valid fraction forecast | Four × 8192 CPU forecast |
| --- | ---: | ---: | ---: | ---: |
| Selected covariance | 136/512 | 0.2656 (0.0195) | 0.1893 | 306 s |
| Covariance × 4 | 44/512 | 0.08594 (0.01239) | 0.06129 | 99 s |

The forecast combines the geometric-cover hard-volume estimate with the
Gaussian checks and the original cube density. It scales the previous
AB confirmation's mean CPU per valid pose, so changed cloud cost and fixed
overheads may alter it. It predicts work, not accuracy or importance ESS.
The wider guide is cheaper here because more draws are hard-invalid, not
because it has demonstrated better sampling. The entire fitter and static
screen completed in 18.3 s.

The separate [runtime implementation and fresh validation](native-guided-cover.md)
are now complete, including two frozen guide widths and a larger independent
wider-guide confirmation. The wider estimate stabilizes near log Q=35.74 with
3.8% observed error, while an 18% difference from the narrow pilot remains.
The fitted source data above remain training only; none of their original
weights was replaced or pooled with those validation campaigns.

The next authorized integration must use fresh fixed budgets and seeds,
retain every zero, evaluate the full hybrid denominator, and compare the
two frozen widths without pooling their outcomes into the training fit.
The proposed control budget is four independent populations × 8192 draws
per model, λ/z=64 and two independent clouds, pending the runtime tests and
separate production launch. No convergence claim follows from the fit or
these static geometry results.
