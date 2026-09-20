# Independent complete-native-region reference

The native normalizer can be estimated without a Gaussian atlas. The region
is defined by the existing registration score, with positional tolerance
`a=2 Å` on every rigid-member center and a proper rotation tolerance of 15°.
This calculation concerns the complete specified native region; it does not
claim that this geometric definition is an equilibrium basin.

## Finite-volume geometric cover

For member centers `b_i`, write their centroid as `c` and covariance as
`S=mean[(b_i-c)(b_i-c)^T]`. Relative to a native pose `(t0,R0)`, define
`w=t-t0+(R-R0)c`. The exact identity is

\[
 \frac1m\sum_i\|t-t_0+(R-R_0)b_i\|^2
 =\|w\|^2+\operatorname{tr}[(R-R_0)S(R-R_0)^T].
\]

If every member error is at most `a`, the left side is at most `a²`.
For relative rotation `R0^T R` of angle θ about axis u, the second term is

\[
 4\sin^2(\theta/2)\,[\operatorname{tr}S-u^TSu]
 \ge 4\sin^2(\theta/2)\,[\operatorname{tr}S-\lambda_{max}(S)].
\]

Consequently the entire native region lies inside `|w|≤a` and

\[
 \theta\le\theta_c=\min\left\{15^\circ,
 2\arcsin\min\left(1,\frac{a}{2\sqrt{L}}\right)\right\},
 \qquad L=\operatorname{tr}S-\lambda_{max}(S).
\]

When a usable positive lower bound on L is unavailable, retain the nominal
15° cap. To sample the cover, draw an isotropic axis, draw θ with CDF
`(θ−sin θ)/(θc−sin θc)`, and draw `w` uniformly in a 3-ball of radius `a`.
Set `R=R0 Rrelative` and `t=t0−(R−R0)c+w`. Translation compensation has
Jacobian one in the product translation/Haar measure: at each rotation it
is just a shift of the translation variable.

With normalized Haar measure on SO(3), the cover volume is

\[
 V_{cover}=\frac{4\pi a^3}{3}\frac{\theta_c-\sin\theta_c}{\pi}.
\]

The angle CDF uses a small-angle series and bisection, avoiding loss of
precision in `θ−sin θ`. Actual `q≤1`, hard-core, and capture indicators are
applied after drawing. No failed draw is retried or removed from the count.

At a valid pose the existing independent positive Poisson weight satisfies
`E W=exp(z C)`, with C the overlap with the **union** of all fixed exclusions.
The full-region estimator is

\[
 \widehat Q_{native}=\frac{V_{cover}}{N}\sum_{j=1}^N
 1_{q_j\le1}\,1_{capture,j}\,1_{hard,j}\,\overline W_j.
\]

Multiple cloud weights are averaged in linear space. Logarithms are only a
stable representation for sums. A correct finite cover eliminates support
holes from an atlas; finite samples can still miss important weight
concentration inside that cover.

## References and coordinate conventions

The site-0 configuration has exactly one native reference, four member
centers, and centroid zero. The Python pilot directly compared old SMC
metadata with the new configuration: native poses, member poses, fixed
neighbors, both metric tolerances, capture ball, shape SHA, and bath parameters
all match. `src/normalizer.rs` uses scalar-first quaternions and the maximum
member-center error combined with proper quaternion angular distance. Member
orientations are not additional terms in this metric. The shape contains
4004 atom spheres; the fixed-neighbor list has one tetramer.

The covariance eigenvalues are approximately `(0,161.382981,399.865612) Å²`.
The tight exact-arithmetic angular cap is 9.0296995°, giving cover volume
0.00695008506 Å³. The source data still use the original 2 Å / 15° metric for
every native classification.

The Python pilot used an FP64 eigensystem with a small outward margin. The
Rust implementation instead bounds the largest eigenvalue using

\[
 \lambda_{max}(S)\le U=\min(\|S\|_F,\max_i\sum_j|S_{ij}|),\qquad
 L_{lower}=\max(0,\operatorname{tr}S-U).
\]

This is analytically conservative and gives a slightly wider cover. FP64
outward slack is applied to the upper norm and lower trace; unusable moment
bounds fall back to the nominal cap. The implementation is not an interval
arithmetic certificate. The actual Rust cap is recorded separately in each
run's `cover.json`; it must not be confused with the tighter Python pilot cap.

For several native references, sample a mixture of their covers and divide
by the **full** density `sum_j p_j 1_cover_j/V_j`. Equal-volume, equal-weight
covers give weight `M V/m(x)`, where m(x) counts covers containing the pose.
Adding separate region masses without an overlap correction double-counts.
The Rust MVP rejects metadata with more than one native reference.

## Geometry pilot and independent checks

Run artifacts are in
[native-cover-geometry-pilot-20260920](/home/xvg/tetramer-mc/runs/native-cover-geometry-pilot-20260920).
The exact command was:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python -B \
  /home/xvg/tetramer-mc/tools/native_cover_pilot.py \
  --config /home/xvg/tetramer-mc/runs/basin-normalizer-local-nativewide-8192-l64/runs/scale-1-r01/config.json \
  --out /home/xvg/tetramer-mc/runs/native-cover-geometry-pilot-20260920 \
  --draws 32768 --seed 202609201700
```

| Screen | Count / 32,768 |
|---|---:|
| Mean-square necessary condition | 3,276 |
| Actual q≤1 | 735 |
| Capture-valid | 32,768 |
| Native, capture-valid, hard-valid | 24 |
| Hard-valid native core / shell | 8 / 16 |

Hard checks use an independent scipy KD-tree for each distinct fixed-atom
radius, taking the nearest atom in each equal-radius group. This exactly
reduces the variable-radius collision test to constant-radius nearest-point
queries. Accepted poses check all groups. A conservative body-ball bound
excludes nonzero periodic images throughout the capture region by more than
204 Å, so the open fixed-neighbor calculation matches the archived box.

The pilot took 1.99 s wall/CPU, including 1.77 s of atomic checks. The native
hard-valid rate was 0.07324% of the cover, or 3.265% given actual q≤1. The
zero-activity native pose volume was `(5.09±1.04)×10^-6 Å³` (sampling SE).
This is not a depletion normalizer. Scaling this tight-cover geometry screen
alone suggests about one minute and 730 valid poses per million trials;
the wider Rust cover and Poisson work change that cost.

`tools/check_native_cover_pilot.py` independently recomputed the quaternion
rotation formula and every q label, verified the mean-square identity, and
compared six poses against all 16,032,016 atom pairs per pose. All checks
passed. Maximum identity error was 1.78×10^-14 Å²; maximum q difference was
4.22×10^-15. The angular inverse-CDF residual was below 4.3×10^-14. Provenance,
full unweighted draws, source hashes, and validation results are saved with
the pilot.

## Rust validation and first physical reference

The implementation is `src/native_region.rs`, exposed by
`src/bin/native-region-normalizer.rs`. Three release tests passed in 0.29 s:
the exact no-obstacle sphere volume at zero activity, capture clipping and
retention of all zero draws, the normalized-Haar angular CDF including small
caps, nonzero member centroids, rotations of the world frame, actual member
metric clipping, and rejection of multiple-reference metadata.

The actual Rust moment bound is `L_lower=130.044495 Å²`. Its angular cap is
10.0615483° and cover volume 0.00961245591325 Å³. All 735 native Python pilot
draws and all 4096 archived site-0 native endpoints lie inside this cover;
the records are in
[native-region-cover-validation-20260920](/home/xvg/tetramer-mc/runs/native-region-cover-validation-20260920).
These checks supplement the analytic cover argument; a finite endpoint check
alone would not prove complete support.

The authorized physical reference campaign used four independent populations
of 262,144 unconditional draws, seeds `98581010+1009*i`, activity 0.035 Å⁻³,
depletant radius 1.5 Å, Poisson intensity `lambda=64*z`, and two independent
clouds per accepted geometry. The shape, fixed neighbor, capture region, and
native metric are unchanged. The exact runner invocation was:

```bash
cd /home/xvg/tetramer-mc
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/run_native_region_reference.py \
  --root runs/native-region-reference-4x262144-l64-20260920 \
  --config runs/basin-normalizer-local-nativewide-8192-l64/runs/scale-1-r01/config.json \
  --binary target/release/native-region-normalizer
```

The campaign is **complete**. Do not rerun this command against its existing
output directory. It archives the executable, configuration, exact commands,
seeds, source bundles, hashes, runner status, and all unconditional rows.
Invalid rows contain draw index, q, and zero reason; valid rows also contain
the full pose, both cloud counts/weights, envelope bounds, and final weight.
Per-draw random streams permit reproduction without storing a pose for every
invalid draw.

| Population | Hard-native poses | log Qnative | Observed weight ESS |
|---|---:|---:|---:|
| r00 | 154 | 11.87434 | 2.61 |
| r01 | 166 | 15.07877 | 1.61 |
| r02 | 142 | 10.16645 | 5.42 |
| r03 | 166 | 14.52165 | 1.41 |

The four populations took 4.36–5.06 s each, about 18.90 CPU seconds in total.
There were 628 valid poses and 18.89 million raw cloud points. Combining all
1,048,576 trials on the linear scale gives **log Qnative=14.17539**, with
observed point-weight ESS **3.08**, relative sampling SE **57.0%**, and a
largest single contribution of **47.5%**. The independent-population relative
SE is 58.5%. This is not a converged native normalizer.

All retained rows were checked for contiguous draw indices, correct count
accounting, linear averaging of the two cloud weights, and consistency with
`z*L+K*log1p(1/64)`. The unweighted geometry estimate
`(5.757±0.230)×10^-6 Å³` agrees with the earlier Python screen. The
[assessment](/home/xvg/tetramer-mc/runs/native-region-reference-4x262144-l64-20260920/assessment.json)
records these checks and the largest original weights without replacing them
after selection. Complete proposal support removes atlas truncation as a
failure mode; concentration of physical weight inside the native region
remains the source of poor precision.

## Prespecified 32-fold draw-budget sensitivity

A fresh campaign, kept separate from the pilot, used eight populations of
4,194,304 draws (33,554,432 total), seeds `98611010+1009*i`, eight workers,
the **same frozen executable**, the same configuration, lambda/z=64, and two
clouds. No result-dependent stopping was used. The prior pilot's sample files
occupied 51.92 MB; projected output was 1.66 GB, so the full existing row
encoding was retained. Actual new sample output was 1.700 GB.

```bash
cd /home/xvg/tetramer-mc
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/run_native_region_reference.py \
  --root runs/native-region-reference-8x4194304-l64-20260920 \
  --config runs/basin-normalizer-local-nativewide-8192-l64/runs/scale-1-r01/config.json \
  --binary runs/native-region-reference-4x262144-l64-20260920/provenance/native-region-normalizer \
  --replicates 8 --samples 4194304 --seed-base 98611010 --workers 8
```

This campaign is complete. Its eight population log normalizers were
`14.84365, 14.54691, 15.84038, 15.35912, 14.93658, 15.08772, 14.02504,
16.00487` in replicate order. All 33,554,432 rows and their file hashes passed
the streaming count/weight audit. Analysis held only moments and a small
top-weight heap in memory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/analyze_native_region_reference.py \
  runs/native-region-reference-8x4194304-l64-20260920 --workers 8
```

The fresh campaign alone gives:

| Quantity | Result |
|---|---:|
| Hard-native poses | 20,041 |
| log Qnative | 15.26037 |
| Point-weight ESS | 11.32 |
| Observed point relative SE | 29.72% |
| Independent-population relative SE | 22.49% |
| Largest point contribution | 22.43% |
| log Qnative core / shell | 14.61877 / 14.51287 |
| Zero-activity volume / Å³ | `(5.7412±0.0405)×10^-6` |
| Production CPU / maximum population wall time | 595.77 s / 79.17 s |
| Raw cloud points | 602.36 million |

The observed native mass increased by 1.085 log units relative to the separate
small pilot, while the unweighted geometric volume remained stable. This is
evidence of substantial concentration of Boltzmann weight within the native
region. The complete-cover estimator provides an independent check without
Gaussian support assumptions; the remaining uncertainty still precludes a
convergence claim or a global native probability without an adequate competing
region normalizer. Neither campaign has been pooled with the other.

The complete streaming result, per-population moments, original highest-weight
poses, and provenance are in
[assessment-streaming.json](/home/xvg/tetramer-mc/runs/native-region-reference-8x4194304-l64-20260920/assessment-streaming.json).

## Regional competitor comparison and reusable pose records

For the **same one-fixed-tetramer environment**, the independent refined
estimate of the already frozen competing region `R≤8, original q≥5` is
log Q_R8=21.83045, with observed relative SE 4.03%. This is a regional
normalizer, not the total nonnative mass. Compared with the fresh complete
native estimate 15.26037, its mass ratio is approximately **713**, or
`F_R8−F_native=−6.570 kBT`. Combining the two observed relative errors gives
about 30% error on the ratio by first-order propagation; it is not a bound on
unseen native weight or a rigorous confidence interval. The original direct
uniform-R8 control is noisier (log Q_R8=21.12644) and gives ratio about 353.
Neither comparison estimates a global native probability. Additional crystal
neighbors change hard support and can exclude this particular competitor.
The refined regional source is
[deep-shells.json](/home/xvg/tetramer-mc/runs/basin-normalizer-mis-refined-16384-l64/assessment/deep-shells.json).

Every valid original row already contains the full scalar-first quaternion
pose, q, both independent cloud counts/log weights, envelope lower/upper
volumes, and the final importance weight. It is sufficient for subsequent
proposal refinement, while full zero accounting remains in the original
files. A convenient derived index extracts all **20,041 valid poses** into
`valid-native-poses.jsonl` (13.93 MB), retaining replicate/draw identifiers and
original source paths. `manifest.json` records all jobs, seeds and input
hashes; each `runs/rXX/manifest.json` records the cover, target metric,
activity, cloud intensity and shape hash. `provenance/native-region-normalizer`
is the frozen executable. The extraction does not constitute a new normalizer.

The full observed distribution and leading poses are in
[concentration.json](/home/xvg/tetramer-mc/runs/native-region-reference-8x4194304-l64-20260920/concentration.json),
produced by `tools/native_region_diagnostics.py`. The top one/five/twenty
poses carry 22.43% / 54.06% / 76.70% of the measured mass. The native shell
`0.8<q≤1` contains 69.5% of valid geometry but 47.4% of weight; the core
contains the remaining 52.6% of weight. About 91.0% of weight lies at center
displacements 0.75–1.5 Å, and 68.0% at proper angles 2–4°.

| Replicate / draw | q | Center displacement / Å | Proper angle | Observed mass fraction |
|---|---:|---:|---:|---:|
| r07 / 3224755 | 0.9558 | 1.3257 | 3.5723° | 22.43% |
| r02 / 2535251 | 0.4952 | 0.8535 | 1.2011° | 13.64% |
| r03 / 3115420 | 0.7537 | 1.2242 | 2.1906° | 10.86% |
| r04 / 3215126 | 0.9406 | 1.4816 | 2.8072° | 3.60% |
| r05 / 239544 | 0.7367 | 1.0952 | 2.5146° | 3.53% |

These are sample poses around the single original native reference, not new
definitions of the native region. Their concentration supports investigating
a broad, regularized proposal refinement, but the effective weighted sample
size is only about eleven. Any learned proposal needs new frozen-proposal
validation draws; these original weights must not be retrospectively changed.

Four small analyzer controls also pass: an all-zero end-to-end campaign
reports unresolved/null uncertainty, one nonzero draw reports no variance
estimate, mixed zero/nonzero population means retain their zeros, and pooling
rejects mismatched physical targets or covers. The original analyzer source
matching the recorded assessment hash was archived before these robustness
changes. Existing nonzero population statistics and replicate uncertainty are
bit-identical under the updated formulas; all eight physical manifests and
covers match. See `analyzer-edge-controls.json` in the large campaign.

## Optional mixture of smaller geometric covers

The integrator now also accepts `--cover-scales 0.1,0.2,0.4,1` and optional
positive `--cover-weights`. The original native metric remains the target.
Each draw divides by the sum of the densities of every cover containing
that pose; the scale-1 component preserves complete target support. The
default single cover reproduces the original sample records byte for byte.

The [nested-cover implementation and controlled pilot](nested-native-cover-mixture.md)
describe the density, reference-limit tests, independent audit, and results.
That pilot reduced the observed concentration of the two-neighbor native
integral, while performing poorly for the broader single-neighbor target.
It is a targeted integration option, not an established general sampling
speedup. Earlier uniform estimates and files remain separate.
