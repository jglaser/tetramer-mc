# Independent local controls for intermediate contacts

The complete intermediate reference was dominated by one pose, while fresh
Gaussian-guided integration remained concentrated on rare contributions.
These controls measure a fixed neighborhood of that training maximum. They
retain both AB neighbors, the original interval **2 ≤ q < 5**, capture radius
18 Å, depletant radius 1.5 Å, and activity 0.035 Å⁻³. They do not change the
physical definition of the intermediate region or replace its complement.

## Fixed regions and independent draws

The chart is the existing weighted Gaussian in
`runs/ab-intermediate-guide-preparation-20260920/model-weighted.json`, hash
`2e5a193540b54a855dc55ff88ea98ee77dc419a73f126d9ea1b9aeaa85626e79`.
It has its original nonzero mean and full translation–rotation covariance.
No new fit is performed. With Cholesky factor L, mean μ, and pose coordinates
x, define u = L⁻¹(x − μ). The two local regions are the original intermediate
region intersected with **|u| ≤ 4** and **|u| ≤ 1**, respectively.

Radius four contains 98.625% of this Gaussian's probability, but no such
physical coverage is assumed. Uniform six-ball sampling allocates only
1/4096 of its draws to the radius-one core: about 16 of 65,536 draws before
hard/q rejection. The separate radius-one control was therefore frozen and
launched before inspecting radius-four outcomes. It contains 1.439% of the
Gaussian's probability. Its purpose is to resolve the central neighborhood,
not to adapt the target to a new observed maximum.

Each control has four independent populations of 16,384 unconditional draws,
with λ/z = 64 and two independent Poisson clouds per valid pose. The seed
bases are 99971010 for radius four and 99981010 for radius one, incremented
by 1009 per population. Counts and seeds are fixed before sampling. Each
plan also declares four equal-width radial bins within its own ball.

| Radius | Region SHA-256 | Plan |
| --- | --- | --- |
| 4 | `67e30f37418c382b89505309329190cad4a4983d61e393f67b903b943f8186b5` | `runs/ab-intermediate-weighted-r4-plan-20260920` |
| 1 | `3d37e4bd9bb558a143f5a379e7f4a0b398a144e91a7827730223fd6bb8aa568a` | `runs/ab-intermediate-weighted-r1-plan-20260920` |

The campaigns append `-reference-4x16384-l64-20260920` to the corresponding
`ab-intermediate-weighted-r4` or `ab-intermediate-weighted-r1` run prefix.

## Estimator and scope

The unchanged `latent-region-normalizer` draws u uniformly in the six-ball.
The estimator is

\[
\widehat Q_R=\frac{V_6(R)}{N}\sum_i
I_{\mathrm{capture}}I_{\mathrm{hard,AB}}I_{2\le q<5}
\frac{|\det L|}{\ell^3\pi^2(1+|c_i|^2)^2}\,\overline W_i,
\qquad c_i=(\mu_{\mathrm{rot}}+(Lu_i)_{\mathrm{rot}})/\ell.
\]

The measure is translation volume times normalized SO(3) Haar measure.
The two positive cloud weights are averaged on the linear scale and have
conditional expectation exp(zC), with C the intersection of the moving
exclusion union with the **union** of both fixed neighbors. All failed
draws retain zero weight in N. The Gaussian defines the coordinates; its
density does not multiply or divide this uniform-reference estimator.

An independent geometry review checked the complete embedded model, both
neighbors, exact q endpoints, bath, shape and executable hashes. For R=4,
conservative bounds place the entire chart ball within 8.543 Å of the capture
origin and below 26.12° residual rotation, away from the Cayley singularity.
Its physical pose volume before hard/q masks lies between 1.11169×10⁻⁶ and
1.23456×10⁻⁶ Å³. These are chart-volume bounds, not bounds on the
depletion-weighted integral.

The fresh local draws are independent conditional on the frozen chart and
regions. The chart itself was trained on the earlier reference; historical
guide restrictions are retrospective diagnostics. Neither training data nor
historical guide estimates are pooled with these fresh local references.
The guide comparison retains each campaign's original full density and N,
and reports the complementary region explicitly. The local reference does
not sample that complement. Disjoint radial bins from the same draws are
correlated; whole-ball variance is computed from whole-row weights, not by
summing bin variances.

## Reproduction

`tools/prepare_intermediate_local_region.py` verifies the frozen source and
archives the exact region, configuration, plan and source hashes without
drawing any poses. For example:

```bash
python tools/prepare_intermediate_local_region.py --out runs/fresh-r4-plan
python tools/run_latent_region_campaign.py \
  --out runs/fresh-r4-reference \
  --config runs/fresh-r4-plan/config.json \
  --region runs/fresh-r4-plan/region.json \
  --binary runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer \
  --samples 16384 --replicates 4 --workers 4 \
  --seed 99971010 --lambda-ratio 64 --cloud-replicates 2
python tools/analyze_latent_region.py --root runs/fresh-r4-reference
```

Use `--radius 1 --seed 99981010` when preparing the inner control and the same
seed in its launcher. Fresh output directories are required. The archived
executable is pinned to
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`;
the physical kernel is unchanged.

The remaining-space integral, complete intermediate mass and eventual
assembly verdict remain separate requirements. A successful local control
alone cannot establish their convergence.

## Completed local estimates

Both fixed-N campaigns completed and passed all-row independent pose,
Jacobian, original-q, cloud-law and provenance checks. The direct estimator
and an independently reconstructed original-density guide restriction give:

| Region | Fresh uniform log Q | Row / population relative SE | Weight ESS | Largest draw | Historical mixture log Q (row SE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| r ≤ 1 | −1.890440 | 0.920% / 1.686% | 10,017.0 | 0.0613% | −1.826968 (12.15%) |
| r ≤ 4 | 4.733195 | 3.518% / 3.815% | 798.1 | 1.075% | 4.717803 (14.68%) |

The radius-one reference has 45,514 valid draws and costs 1,414.00 sampler
CPU seconds. Its four population log estimates are −1.876642, −1.873362,
−1.942218 and −1.871284. Radius four has 31,233 valid draws and costs 954.18
CPU seconds; population logs are 4.638994, 4.779421, 4.807512 and 4.698026.
Paired clouds account for 19.78% and 25.19% of observed estimator variance,
respectively. These are importance-sampling diagnostics, not trajectory
mixing or accepted-move rates.

The radius-four sample's independently retained r≤1 restriction gives
log Q = −1.934299 with 56.59% row error and only eight valid observations.
The dedicated inner control resolves the same region much better. Hard
volumes also agree with the historical mixture: direct/mixture log Q₀ are
−22.302858/−22.293664 for r≤1 and −14.354662/−14.371769 for r≤4. Direct hard
volume errors are 0.259% and 0.409%. The geometry-only historical guide has
no hits in either ball; its local mass is unresolved, not zero.

![Independent local physical and hard-volume controls](../runs/ab-intermediate-local-reference-figure-labeled-20260920/local-reference.png)

The plotted error bars transform Q±one observed standard error onto the log
axis. They are not bounds on unvisited tails. Chart radii are dimensionless
and are separate from the 1.5 Å depletant radius.

Radius four remains an analysis region, not an identified isolated basin.
Its outer shell 3<r≤4 supplies **74.43%** of its observed physical mass.
The independent radius-one/radius-four mass point ratio is only **0.133%**.
There is no radial mass plateau, and a Gaussian's 98.625% probability
coverage must not be substituted for physical coverage.

The restricted mixture's complement has log Q = **7.710796**, row relative
error **73.37%**, and largest contribution **71.26%**. The geometry guide's
whole-window estimate, log Q = 6.628094, lies in that complement in its
observed sample and remains poorly controlled. These full-window guide
values reproduce the earlier audit exactly. The new references therefore
validate local mass while leaving the remaining intermediate space open.

The independent comparison artifacts are
`runs/ab-intermediate-weighted-r4-comparison-20260920/comparison.json` and
`runs/ab-intermediate-weighted-r1-comparison-20260920/comparison.json`.
`tools/compare_intermediate_local_reference.py` computes the original-density
ball, four radial bins and complement separately, preserving all original
zeros and denominators. Ten tests across the local and earlier intermediate
comparison suites check partitions, coupled chart geometry, cloud weights,
physical masks and provenance qualifications. Historical restrictions
remain retrospective controls rather than held-out region discovery.

## What the original outlier meant

The prior pointwise audit places the training maximum (population r02,
draw 118546, q = 3.26781) at weighted radius **0.233583**, inside both local
controls. Its individual contribution to the earlier full-cover sample
mean had log value **12.065418**. That is about **1,529 times** the fresh
estimate for the entire radius-four region containing it. This compares a
single realized importance contribution with a later regional estimate;
it is not a calibrated post-selection significance test. It demonstrates
how a rare hit under a diffuse proposal can make the observed integral
much too large without an error in its normalization formula.

A separate fixed-pose check draws 256 fresh independent Poisson clouds at
each of the training maximum and two leading new mixture extremes. It
uses the unchanged public overlap-weight kernel, the exact original poses
and envelope geometry, and λ/z=64. The arithmetic mean W is the unbiased
pointwise estimator; its logarithm is not unbiased.

| Fixed pose | Fresh log(mean W) | Observed relative SE | Original two-cloud mean / fresh mean |
| --- | ---: | ---: | ---: |
| Training maximum | 21.94153 | 3.89% | 1.835 |
| Leading mixture extreme | 16.87130 | 3.22% | 0.713 |
| Second mixture extreme | 15.71200 | 3.14% | 0.762 |

Cloud noise inflates the old training observation modestly and does not
explain the large local-integral discrepancy. The two fresh mixture
extremes were not upward cloud outliers in this check. Selection of all
three original points is acknowledged; empirical percentiles are descriptive,
not calibrated significance tests. No pose density or integrated regional
mass is inferred from these pointwise estimates.

The control is archived at
`runs/ab-intermediate-fixed-pose-clouds-20260920/analysis.json`.
`tools/run_intermediate_fixed_pose_clouds.py` builds a thin standalone Rust
driver, verifies relevant source identity against the archived production
kernel and seals its executable, poses, independent streams and counts.
All four existing overlap-weight API tests passed, covering analytic moments,
many-body union geometry, rotated frames and empty/zero limits. Source and
executable identity do not constitute formal verification of floating-point
geometry; the Lean proofs remain abstract mathematical guarantees.

These results support keeping the measured region and directing the next
reference effort to its outside, including the large-radius contacts already
seen by the broad guide. The full q≥5 remainder and assembly thermodynamics
also remain unresolved. Increasing cloud counts alone would not address the
dominant remaining pose-coverage problem.
