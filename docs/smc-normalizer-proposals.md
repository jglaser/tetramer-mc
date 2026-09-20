# Frozen SMC-cloud guides for independent normalization

The original five-component docking atlas gives little importance-sampling
coverage to some distant contact configurations retained by earlier SMC.
`tools/prepare_smc_normalizer_atlas.py` makes an alternative proposal using
those configurations. It does not reuse their SMC partition estimates as
physical weights.

For the selected site at radius 1.5 Å and activity 0.035 Å⁻³, each of the eight
completed SMC populations contributes one full-covariance Gaussian. All final
endpoints contribute equally within that population, including descendants
sharing ancestry. This is a description of a correlated empirical cloud,
not an assertion that the population has found one physical basin.

Each chart uses an observed orientation that maximizes the smallest absolute
quaternion dot product to all endpoints. This minimizes the largest Cayley
radius among sampled candidate references. Translation is centered on the
same observed pose; the fitted mean retains the actual cloud displacement.
No endpoint is trimmed. An exact chart seam causes an error. The report records
the worst chart angle and Cayley radius, so a very broad or poorly charted
population remains visible.

Coordinates are `(translation in Å, ell × Cayley vector)`, with `ell` inherited
from the unchanged base atlas. The empirical covariance uses divisor N.
Eigenvalues are floored at the larger of `(0.02 Å)^2` and `10^-8` times the
largest raw eigenvalue. The latter bounds the condition number by 10⁸.
These are explicit proposal regularizers, not physical interaction parameters.
The earlier base fit used a different, whitened coordinate regularization;
its numerical eigenvalue floor cannot be transferred directly into Å units.

`--coordinate-std-floor` and `--relative-eigenvalue-floor` change those floors.
`--endpoint-std-scale` multiplies the new standard deviations after flooring;
the base atlas stays unchanged. The separate runtime normalizer option
`--covariance-scale` scales every Gaussian in whichever frozen model it reads.
Every change requires a fresh output directory and independent normalization
draws.

The laboratory charts are transformed into the fixed neighbor's body frame:

\[
a_t'=R_f^T(a_t-t_f),\quad a_R'=R_f^Ta_R,\quad
\mu'=B\mu,\quad\Sigma'=B\Sigma B^T,
\qquad B=\mathrm{diag}(R_f^T,R_f^T).
\]

This is an exact coordinate change with unit physical Jacobian. The Gaussian
pose density includes the Cayley-to-normalized-Haar factor. Fresh Gaussian
draws check complete mixture densities before and after conversion.

Three models are written under `models/`:

- `fold-a.json`: the original five components plus populations 0–3.
- `fold-b.json`: the original five components plus populations 4–7.
- `all.json`: the original five components plus all eight populations.

By default the unchanged base components retain their relative weights and
receive half the total Gaussian weight. The new components divide the other
half equally by population. These weights guide sampling only. The runtime
normalizer adds its independent uniform cube/Haar branch and divides by the
**complete** mixture density, including every overlapping component. Hard,
capture, and numerical rejects remain zero contributions in the fixed budget.

The two folds test sensitivity to which correlated snapshots supplied the new
guide. Neither is a blind physical discovery test. The report separates
own-cloud and other-fold density coverage, reports ancestry and covariance
diagnostics, and tests fresh candidates with an independent atomic-distance
hard-overlap predicate. That candidate check uses a KD tree only to enumerate
potential atom pairs; final decisions use each pair's actual radii.
Coverage of known endpoints does not bound unobserved contact-region mass.
The Gaussian-only density improvements can be enormous where the old atlas
assigned negligible Gaussian mass. They must not be confused with improvement
in the complete proposal: its uniform branch supplies a finite density floor.
`--diagnostic-uniform-probability` (default 0.1) reports both versions; it does
not set the later normalizer's runtime option.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_smc_normalizer_atlas.py \
  --site 0 --out runs/smc-normalizer-atlases/site0
```

`--site 1` selects the corresponding neighbor and SMC populations. A relative
base atlas remains a valid normalized proposal for either site; its utility
can differ. `--base-model` selects another already frozen, five-component,
body-relative base atlas with the same shape hash.

The output freezes source configurations, final populations, physical shape,
environment, base model, fitting script, and input hashes. The source run
configurations specify the actual bath: the environment JSON contains older
bath metadata. `report.json` records that distinction explicitly.

## Optional local geometric guides

`--local` replaces each entire-population fit by a collection of small
geometric fits, retaining entire-population components as fallbacks. The
existing default and its frozen models remain controls.

For each selected fold, pool the endpoint poses and measure

\[
d(x,y)^2=\frac{|t_x-t_y|^2}{(4\,\mathrm{\AA})^2}
       +\frac{\theta(R_xR_y^{-1})^2}{(15^\circ)^2}.
\]

The radii are configurable. Deterministic farthest-first selection starts
from endpoint zero and adds the endpoint farthest from all selected observed
centers, until every endpoint has distance at most one to a center. Each
endpoint belongs to its nearest selected center. This bounds each assigned
group's radius and avoids arbitrarily long chains of pairwise-close poses.
Ancestry labels do not enter either distance or grouping.

Groups with at least eight endpoints receive full-covariance local Cayley
Gaussians with the same explicit covariance floors. Groups with fewer
endpoints contribute their proposal mass to the original broad component
of their source population. They are not discarded. The selected populations
have equal total input weights; local and fallback weights are the sums of
their assigned endpoint weights. Their combined proposal mass remains one
before applying the overall default 50% base / 50% new-guide split.

The cap is 128 qualifying local components **per model**, excluding the five
base and up to eight fallback components. Exceeding it raises an error; the
preparer never silently drops regions to fit the cap. `report.json` records
the number of cover centers, retained groups, fallback fractions, ancestry
mixing, chart angles, and every local component's proposal mass. These counts
are still properties of a proposal fit, not equilibrium occupations.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_smc_normalizer_atlas.py \
  --site 0 --local --translation-cover-radius 4 \
  --rotation-cover-radius-deg 15 --minimum-cluster-size 8 \
  --maximum-local-components 128 \
  --out runs/smc-normalizer-local-atlases/site0
```

Local mode audits approximately 512 fresh component candidates per model;
`--candidate-probes-total` changes this budget. The audit allocates an equal
integer number of trials per component and reports all counts, rather than
presenting them as draws weighted by the overall mixture.

The first ancestry audit explains why localization is useful but does not
certify generalization: families with at least eight endpoints account for
85.8% of the site0 data and have median angular diameter 9.25°. Some individual
families nevertheless span 132–175°, demonstrating why ancestry cannot define
basins. Independent folds also contain distant environments: the median
nearest-other-fold distance in a translation/Cayley-scaled geodesic metric
is 13–18 Å. Own-cloud fit improvement is consequently reported separately
from other-fold coverage. The uniform branch and fallback components remain
essential. Full-proposal coverage reports fractions exceeding 10, 100, and
1,000 times the uniform contribution, in addition to log-density summaries.

## First local-guide screen

The site0 geometric covers retain 47, 44, and 96 local components for fold A,
fold B, and all populations. The corresponding fallback endpoint fractions
are 15.5%, 17.5%, and 16.3%. No component cap is reached. Local chart radii
remain below 9.1° and complete density conversion errors below 3.3×10⁻¹³.
Independent component-wise candidate checks find approximately 51–55%
hard-and-capture-valid local draws, compared with a few percent for most
entire-population components.

Four independent 512-draw normalizers were run for each model at λ/z=64,
with two independent clouds per pose and unchanged physical parameters:

| Guide | log Q native | log Q shoulder, bound | log Q distant, bound | Distant ESS | Largest distant weight fraction |
|---|---:|---:|---:|---:|---:|
| Local fold A | 14.538 | 15.719 | 12.499 | 18.04 | 0.125 |
| Local fold B | 14.653 | 12.996 | 13.247 | 6.59 | 0.320 |
| Local all | 14.937 | 16.070 | 13.368 | 6.73 | 0.354 |

These are exploratory estimates, not converged physical weights. Shoulder
estimates have only 1–5 nonzero contributions, and distant masses retain
substantial importance-weight concentration. Approximately 14–16% of the
observed distant-weight variance comes from the Poisson clouds; pose sampling
dominates the remaining observed variance. The local fits recover much more
distant contribution than entire-population guides, while cross-fold
coverage remains weak. This establishes a useful direction for proposal
refinement without resolving the earlier SMC discrepancy.

The independent reports are under
`runs/basin-normalizer-local-{fold-a,fold-b,all}-512-l64/assessment/`.
Follow-up models widen only the new local and fallback standard deviations
by factors two and four. Their original five components, chart anchors,
means, and weights remain exactly unchanged; the new covariances equal four
and sixteen times the original matrices, respectively. Production
normalizers must use global `--covariance-scale 1` for this comparison.

The larger width control used four fresh populations of 8,192 unconditional
draws for each frozen all-population guide (32,768 draws per row), with λ/z=64
and two independent clouds. Models were analyzed separately:

| New-guide standard-deviation factor | log Q native | log Q shoulder, bound | log Q distant, bound | Distant ESS | Largest distant weight fraction | Hard-valid intermediate draws |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 15.088 | 16.505 | 13.753 | 11.18 | 0.268 | 2 |
| 2 | 15.121 | 16.692 | 13.948 | 31.83 | 0.083 | 2 |
| 4 | 14.854 | 15.934 | 14.209 | 6.77 | 0.271 | 1 |

Native estimates are much steadier than the shoulder estimates. Distant
estimates increase with width, but their observed relative standard errors
remain 30%, 18%, and 38%, respectively. The intermediate region is unresolved:
only one or two valid contributions occur, and some contributions dominate
their estimate almost completely. No intermediate-unbound pose is observed;
that is not an upper bound on its physical mass.

Width two currently has the best observed distant-region importance
efficiency. This does not establish complete contact coverage or a trajectory
mixing speedup. The three width campaigns use exactly the same executable;
independent Python reconstruction of 128 valid pose densities per width
agrees with stored Rust full-mixture densities within 1.9×10⁻¹³.
Reports are under `runs/basin-normalizer-local-width{1,2,4}-8192-l64/assessment/`.

## Native-tail proposal control

`tools/prepare_native_tail_normalizer_atlas.py` retains all 109 components of
the width-two guide at half their previous weights, and adds two copies of
original native-informed component zero at standard-deviation factors two
and four, with weights 0.25 each. The original 109 means, anchors, and
covariances remain unchanged. This explicitly native-informed control tests
whether poor sampling in the shoulder and intermediate regions comes from
the original native Gaussian's narrow tails. It changes only the known
normalized proposal, never the physical target or region definitions.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_native_tail_normalizer_atlas.py
```

The default output is `runs/smc-normalizer-local-nativewide/site0/model.json`,
with source and transformation provenance beside it. Use global covariance
scale one and retain the normalizer's uniform cube/Haar branch.

The 32,768-draw native-tail pilot found 200 shoulder and 107 intermediate
hard-valid poses, compared with 69–74 shoulder and 1–2 intermediate poses in
the width controls. Accessibility improved, but the shoulder normalizer
remains dominated by one observation (84% of its estimated mass; ESS 1.42).
Its log Q is 17.229. The intermediate estimate has log Q 13.477 and ESS 3.53;
the distant estimate has log Q 14.452 and ESS 6.34. These are still unresolved
normalization estimates, not converged free-energy differences.

## Frozen importance-guided recentering

`tools/prepare_importance_guided_normalizer_atlas.py` addresses that remaining
coverage problem by fitting two new proposal means and covariances. Training
uses only completed, independently seeded width-one, width-two, width-four,
and native-tail campaigns. It selects the 529 hard-valid poses with original
registration `1 < q < 5` from their 131,072 unconditional draws. Original
coordinates, region definitions, and physical interactions stay fixed.

All selected poses are encoded in original component zero's fixed Cayley
chart, after transforming to the fixed neighbor's body frame. No chart anchor
is learned. Nonfinite coordinates or a failed positive-definite covariance
check stop preparation. The observed maximum chart angle is 54.57°, safely
away from the chart's half-turn seam.

For each tempering power τ in `{1, 0.5}`, normalize training weights
proportional to `exp[τ(log importance_weight − log campaign_draws)]`, then
compute the weighted six-dimensional mean and covariance. Here every source
campaign has the same number of unconditional draws, so its denominator is
a common constant. The final covariance is

\[
\Sigma_{\mathrm{guide}}=\Sigma_{\mathrm{weighted\ fit}}
                         +0.25\,\Sigma_{\mathrm{original\ component\ 0}}.
\]

The additional covariance explicitly preserves broad directions and positive
definiteness. There is no hidden eigenvalue clipping. The τ=1 training ESS is
7.10 and its largest normalized weight is 35.2%; for τ=0.5 these are 89.66 and
6.26%. These are training diagnostics, **not** production ESS or physical
basin weights. The covariance condition numbers are 98.4 and 97.5.

All 111 previous components retain half their weights, without changing any
means, anchors, or covariance matrices. The two new components each receive
weight 0.25, giving a 113-component normalized Gaussian guide. Independent
atomic-distance checks found 28/128 and 45/128 valid new-component candidates.

The fitter freezes the source model, source manifest/sample hashes, selected
training poses, raw/final covariances, actual physical signature, and its own
implementation. **Only new, independently seeded production draws assess the
result.** The training observations are not recycled as fresh evidence. This
is an explicitly native-informed physical-normalization control, not
template-free basin discovery or an equilibrium interpretation of GMM weights.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_importance_guided_normalizer_atlas.py
```

The default output is `runs/smc-normalizer-importance-guided/site0/model.json`.
Its independent evaluation uses four populations of 16,384 unconditional
draws at λ/z=64 and two independent clouds per valid pose, with global
covariance scale one. The previously frozen executable is reused.

The independent 65,536-draw evaluation completed in 442.3 summed CPU seconds:

| Region | Valid contributions | log Q | Importance ESS | Largest weight fraction |
|---|---:|---:|---:|---:|
| Native, q≤1 | 2,032 | 14.977 | 357.48 | 0.0173 |
| Shoulder, 1<q<2 | 7,769 | 16.887 | 214.33 | 0.0557 |
| Intermediate, 2≤q<5 | 574 | 14.991 | 8.33 | 0.3367 |
| Distant, q≥5, bound | 4,447 | 13.233 | 21.60 | 0.1756 |

Recentring greatly improves observed shoulder precision: its four independent
population log Q values are 16.767, 17.049, 16.844, and 16.864, compared with
the previous native-tail estimate's 84% contribution from one pose. The
paired-cloud variance diagnostic attributes about 15% of the new shoulder
estimator's variance to Poisson noise. The intermediate region remains
poorly estimated despite having more valid proposals.

The distant estimate declines relative to the earlier guides' values near
14–14.45, although all models have full support. This unresolved dependence
on the proposal is evidence that observed ESS and agreement among a few
replicates do not establish tail coverage. The observed native fraction
11.19% and log Q(other)=17.049 are provisional ratios of these independent
normalizer estimates, not a converged physical conclusion. No intermediate
unbound contribution was observed; its mass is not thereby bounded.

Independent SciPy reconstruction of the complete 113-component mixture and
uniform-defense density agrees with 128 fresh Rust production rows within
9.1×10⁻¹⁴ in log density. Reports, exhaustive region bins, per-population
values, and the density audit are under
`runs/basin-normalizer-importance-guided-16384-l64/assessment/`.

## Independent control of a newly discovered far-contact region

A fresh flat-reference SMC population at original `q>5` discovered a compact
contact cloud with SMC log Z 27.273. All 512 endpoints have distinct poses,
but descend from one ancestry family. This is exploratory discovery; neither
that estimate nor the endpoint count establishes an equilibrium basin weight.
The SMC run rescales registration internally by five. Its stored q range
5.187–5.648 therefore corresponds to **original q=25.936–28.238**, independently
recomputed with the unchanged 2 Å and 15° metric.

`tools/prepare_deep_far_normalizer_atlas.py` fits the equally weighted endpoints
in one compact Cayley chart. The maximum rotation from its sampled maximin
anchor is 4.657°, its covariance condition number is 64.7, and the requested
absolute/relative covariance floors do not activate. Lab-to-body conversion
preserves Gaussian log densities within 1.5×10⁻¹⁴. A guard stops preparation
if the cloud exceeds the configured chart extent; it does not silently fit
a chart spanning a half turn.

Three separate frozen models retain the complete 113-component parent at
total weight 0.5 and add the discovered-cloud Gaussian at weight 0.5. Only
the new Gaussian's standard deviations change, by factors 1, 2, and 4. SMC
log Z and ancestry never set these proposal weights. The full parent proposal
has median log density −12.976 on these endpoints; the new width-one proposal
has median 13.977. This 26.96-nat increase diagnoses a region that the earlier
proposal almost entirely missed. Independent candidate geometry checks found
105/256, 34/256, and 12/256 valid draws from the three new components.

Before fresh production, `tools/analyze_fixed_normalizer_region.py freeze`
recorded a fixed region: original `q≥5` and Mahalanobis radius at most three
in the **width-one** discovered Gaussian chart. Its mean, covariance, anchor,
and neighbor frame are immutable across all three proposal widths. The
definition is archived as
`runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json`, SHA-256
`01ee36aeff8f55260ae23fb67f8744c11fb9182d5d7606a7ad1be9a269d08789`.
This extra region leaves the primary q partition unchanged. Analysis reports
its mass, the remaining far-region mass, and the remaining capture-domain
mass using the same full-density importance weights and all unconditional
zero draws. Agreement for this fixed region would establish a local control,
not completeness of all far-contact configurations.

The independent campaigns each contain four populations of 8,192 draws with
λ/z=64 and two clouds per valid pose. They reuse the same frozen executable,
but have fresh seeds and no recycled SMC observations. Model construction
archives source summary/configuration/geometry and fitting code; production
archives the exact model and executable. This control inherits the parent's
native-informed components and is not a template-free assembly test.

All three fresh controls completed (32,768 unconditional draws each):

| New-guide width | Fixed ellipsoid log Q | Ellipsoid ESS | Largest ellipsoid fraction | Far complement log Q | Far complement ESS |
|---|---:|---:|---:|---:|---:|
| 1 | 21.382 | 77.40 | 0.0693 | 23.397 | 1.11 |
| 2 | 20.816 | 34.71 | 0.1044 | 19.878 | 47.20 |
| 4 | 22.322 | 2.12 | 0.6572 | 19.264 | 18.46 |

The ellipsoid receives 5,734, 478, and 17 valid contributions, respectively.
Widths one and two still differ by 0.566 in log Q; widening to four wastes
most proposals outside this fixed region and leaves its estimate dominated
by two observations. Thus even its local mass needs further precision and
coverage checks. The native estimates stay near log Q=15 (15.118, 15.191,
14.916), while the independently observed fixed far region is much heavier
than the previously sampled contacts. These results establish a material
coverage failure in the older guides, without confirming the exploratory
SMC log Z=27.273 or the complete far-region normalizer.

Width one's far-complement estimate is 94.7% due to a single pose: population
`scale-1-r00`, draw 403, original q=28.131, Mahalanobis radius 4.233. Its
log proposal density is 7.026; independent cloud log weights 41.071 and
40.326 give log importance weight 33.740. It lies near the discovered
contact cloud but outside the frozen ellipsoid. The companion cloud confirms
that this is not solely a large Poisson fluctuation; the proposal allocates
too little density to a consequential nearby pose. This observation remains
exploratory and cannot become fresh confirmation by fitting it retrospectively.

Independent full-proposal reconstruction agrees with 128 new poses per
width within 6.6×10⁻¹⁴ in log density. The fixed-region timestamp precedes all
campaign creation times, and the region/complement estimates sum to the
unchanged total normalizers. Complete results and top-contributor records
are under `runs/basin-normalizer-deep-far-std{1,2,4}-8192-l64/assessment/`;
`fixed-region.json` contains the additional fixed-domain estimates.

The follow-up [direct latent-region integrator](latent-region-normalizer.md)
uses uniform six-ball draws and the exact physical pose Jacobian for the
same frozen region. It provides an independent control without Gaussian
tail importance weights, plus prespecified radius-five/eight shell checks.
