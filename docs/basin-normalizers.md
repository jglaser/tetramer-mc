# Independent physical basin normalizers

The `basin-normalizer` executable estimates physical contact-region weights
independently of the trapped docking trajectories. It retains the exact same
hard sphere-union shapes, fixed neighbors, center-capture ball and ideal bath.
Its integration measure is center volume in Å³ times normalized rotational
Haar measure. Native coordinates classify the output and may inform a frozen
proposal; neither introduces an attractive interaction.

## Estimator

For a pose x, let C(x) be its exclusion overlap with the **union** of fixed
exclusions. A conservative body-frame cell traversal supplies disjoint known
inside cells of total volume L(x), plus uncertain cells containing the rest of
the overlap. At any finite traversal budget, exact leaf membership thins a
Poisson cloud on the uncertain envelope to

\[
 K\mid x\sim\mathrm{Pois}[\lambda(C(x)-L(x))],\qquad
 W(x,K)=e^{zL(x)}(1+z/\lambda)^K.
\]

Thus E[W|x]=exp[zC(x)] and its conditional relative variance is
exp[z²(C−L)/λ]−1. Known interior volume contributes deterministically. Cell
budgets change cost and variance, not the expectation. Geometry uses the same
guarded FP64 predicates as the MC code, not formal interval arithmetic.

Draw a fixed number N of independent poses from the normalized density g,
without retrying invalid draws. For each region b estimate

\[
 \widehat Q_b=\frac1N\sum_{i=1}^N
 \frac{H(x_i)1_D(x_i)1_b(x_i)}{g(x_i)}
 \left[\frac1m\sum_{j=1}^m W(x_i,K_{ij})\right].
\]

Each cloud is independent of pose selection and the other clouds. Hard-invalid
and out-of-capture draws remain zero contributions in the fixed denominator N.
Legacy all-anchor proposals also record finite-precision null draws as zeros;
their frequency must be inspected. Selected-anchor and active reciprocal
proposals instead stop on numerical nulls, since censoring their open Gaussian
tails changes the proposal law. Cloud and population averages are taken on the
linear weight scale before any logarithm or ratio. A ratio of unbiased
normalizers need not itself be unbiased.

The proposal is the complete frozen Gaussian mixture plus a positive uniform
cube/Haar density. The cube circumscribes the capture ball. If there is more
than one fixed neighbor, its uniformly chosen anchor is marginalized in g:
g(x)=sum_j g(x|j)/number_of_anchors. Gaussian component and anchor labels are
never substituted for this full denominator. `--covariance-scale s` multiplies
Gaussian standard deviations, so Σ becomes s²Σ. Means and chart anchors stay
fixed. These changes affect importance efficiency but not the physical target.

An exact reciprocal envelope is also supported with `--covariance-scale 1`.
It uses the original base mixture and its nonlinear inverse branches without
exporting them as Gaussian approximations. The full `F(T) = [G(T)+G(T^-1)]/2`
law (or the explicitly flagged partial mixture) remains inside each anchor's
uniform-plus-learned density. Both fixed-anchor and marginalized-anchor
estimation are supported. Rescaling an active reciprocal model is rejected;
nonrepresentable reciprocal draws stop the calculation instead of silently
changing the proposal law. Reciprocal populations use manifest schema 3 with
base/virtual component counts and inversion flags.

## Exhaustive region definitions

Using the same native-registration score q as the docking test, partition the
entire supported pose space into q≤0.8, 0.8<q≤1, 1<q<2, 2≤q<5 and q≥5.
Split each interval into bound and unbound according to exact intersection
of depletant-inflated atomic unions. This gives ten disjoint regions, including
the near-native shoulder and the remaining configuration space.

The legacy native normalizer sums q≤1, and legacy other sums **all** q>1.
They are reported separately from distant bound contact weights. The
[earlier SMC endpoint audit](previous-smc-region-audit.md) explains why this
distinction and shoulder coverage require explicit checks. Config metadata
fixes the native reference poses, member centers and tolerances before drawing
any samples; the minimum score over supplied equivalent references is used.

## Diagnostics and validation

Every unconditional draw records its physical pose, full proposal density,
region, hard/capture indicators, independent cloud counts, overlap bounds and
importance weight. Summary ESS, maximum weight fraction and relative standard
error refer to the observed importance sample. They do not prove the absence
of high-weight unvisited configurations. A region with no nonzero observations
has unresolved mass, not a zero-width confidence interval.

At fixed x, two clouds distinguish point noise from pose coverage:
E[W₁W₂|x]=exp[2zC(x)], while E[(W₁−W₂)²/2|x]=Var(W|x). Use these alongside
independent populations, covariance-scale and uniform-weight sensitivity, and
longer fixed sample counts. Increasing λ reduces cloud noise; it cannot
repair missing pose regions. Do not use the same cloud both to choose λ and
to estimate its weight without an additional argument.

Independent tests compare normalizers and angular/contact subregions against
radial quadrature for hard spheres with ideal depletants. They verify normalized
Haar measure, nonzero capture centers, multiple proposal anchors, full mixture
density, zero-activity hard volumes, duplicated Gaussian components and
covariance scaling. Separate overlap-estimator tests check both first and
second moments, traversal budgets, anisotropic dumbbells and the fixed-neighbor
union. These validate the estimator, not convergence for proteins.

The reciprocal extension passes seven targeted Rust checks (including the
existing target guards) and eleven Python auditor checks. A separate
[256-draw sphere integration](../runs/reciprocal-normalizer-cross-language-20260921/validation.json)
checks actual serialized Rust output against the Python auditor with shifted
capture, two rotated anchors and a partially reciprocal mixture. All 256
rows are reconstructed, including 162 invalid zeros and 17 inverse-branch
proposals; the maximum full-density difference is 3.56e-15. This validates
the integration/auditor interface, not whole-domain protein coverage.

Highly elongated proposal Gaussians also require auditing the **effective
factor used to draw them**. In the full D170 control, covariance condition
numbers near 10⁹ produced about 10⁻⁷ differences in log proposal density
between scalar Cholesky and NumPy/LAPACK. Both factors were backward-stable,
but represented slightly different floating-point Gaussian laws. The original
audit stopped at its unchanged 2×10⁻⁸ tolerance; saved physical weights must
not be replaced by weights from the other law.

`normalizer_proposal_density.py` reconstructs the source-bound scalar factor
used for both draws and densities, checks its covariance residual with exact
rational arithmetic against bounds derived from the individual floating-point
operations, and retains independent SciPy pose geometry and density evaluation.
This correction applies to active reciprocal normalizer models, whose factors
are retained directly. Legacy covariance-replacement paths are separate.
The [saved-row diagnosis](../runs/mobile-full-capture-density-math-20260921/analysis.json)
quantifies the difference between the two factors. It changes the observer,
not the proposal, target, samples, weights or acceptance rule.

## Running

For the full spherical vessel, `--wall-radius R --wall-center x y z`
enables the same atomic protein-wall predicate used by assembly. Every atom
sphere must fit; a conservative body bound is only an early acceptance test.
The ideal bath remains unbounded and permeates the protein wall. Thus there
is no extra solvent-wall overlap or clipped-cloud term in the estimator.

In this mode, the config's capture ball must enclose **all** feasible body
centers. The implementation requires its radius to be at least
`distance(capture_center, wall_center) + wall_radius + shape_bound`, with a
floating-point guard. This sufficient enclosure also works for offset and
concave sphere unions. A smaller capture, or a fixed neighbor outside the
atomic wall, fails before output is created. The physical indicator is the
full atomic wall; the enlarged capture is a proposal support enclosure.
Changing the capture without specifying the wall would define a different
target and is not an equivalent calculation.

Wall populations use schema 4 and retain the underlying proposal schema.
Every draw records `wall_valid`; rejected poses contribute zeros to the
same unconditional denominator. The independent Python auditor reconstructs
transformed atom spheres and verifies the enclosure, fixed scaffold and
unchanged permeable bath. No-wall runs retain their earlier schemas and
sampling laws. Sphere tests compare zero and positive activity integrals
with independent radial quadrature and normalized Haar rotations; a dumbbell
test checks wall-valid orientations beyond the conservative inner ball.

```bash
cargo build --locked --release --bin basin-normalizer
target/release/basin-normalizer \
  --config PATH_TO_DOCKING_CONFIG.json --model PATH_TO_FROZEN_ATLAS.json \
  --out runs/normalizer-new --samples 4096 --seed 193840217 \
  --covariance-scale 1 --cloud-replicates 2
```

The input is the existing docking configuration with native-region metadata.
`--activity` can override z for a matched hard-only reference, and
`--uniform-probability` can vary the defensive proposal weight up to one.
The positive Poisson scale uses the configuration's `poisson_lambda_ratio`.
Each output directory must be new. Input files, source bundle and executable
hashes are archived; named random streams separate pose and cloud draws.
`samples.jsonl` contains all proposals, `progress.json` the completed count,
and `summary.json` the fixed-budget estimates and CPU decomposition. A failed
partial run is not a completed population estimate.
