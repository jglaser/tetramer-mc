# Competing contacts with two prescribed neighbors

Adding the second native neighbor excludes the previously certified strong
one-neighbor competitor, but leaves other non-native contacts geometrically
accessible. A bounded screen now supplies a concrete two-contact candidate
for an independent physical integral. These observations concern one mobile
rigid tetramer in the unchanged 18 Å capture ball, proper rotations, depletant
radius 1.5 Å and activity 0.035 Å⁻³. They do not establish equilibrium contact
probabilities or the cost of assembling the fixed neighbors.

## Geometry evidence

The reproducible screen and every sampled pose are archived in
`runs/ab-competing-geometry-screen-20260920`. Exactly 2,048 poses were checked
against both physical neighbors with atom-radius inequalities and explicit
clash witnesses. Inflated-radius contact checks only diagnose whether a
depletion overlap exists; no Poisson clouds or depletion weights were used.

| Cohort | Poses | Hard/capture valid in AB | Valid poses contacting both neighbors |
| --- | ---: | ---: | ---: |
| Existing atlas, stratified over charts, frame A | 512 | 28 | 8 |
| Same chart panel, frame B | 512 | 0 | 0 |
| Uniform capture-ball/Haar | 512 | 18 | 7 |
| Historical other-region endpoints | 256 | 85 | 23 |
| Fresh far-SMC endpoints | 128 | 44 | 31 |
| Previously recorded shoulder poses | 128 | 2 | 2 |

The atlas panel deliberately visits every component; its counts are not
draws from the original component weights. Historical endpoint counts are
not independent equilibrium observations. The shoulder panel deliberately
includes both already known AB survivors. None of these survival fractions
is an equilibrium population estimate.

The most useful compact cohort comes from fresh far-SMC replicate 2:
30 of its 32 screened endpoints survive, all contacting both neighbors,
at original q=14.19–15.89. They are 30 distinct poses from one ancestry
family. Replicate 3 supplies 14 additional survivors. Replicates 0 and 1
supply none in this fixed panel. Thus the continuous exclusion certificate
for the old radius-eight region cannot be generalized to all competitors.

## Frozen proposal and finite comparison region

`tools/prepare_ab_competing_atlas.py` fits a full six-dimensional covariance
to all 30 surviving replicate-2 poses, with equal descriptive weights and
an explicit 0.05 Å floor on latent eigenvalue standard deviations. The
floor changes no eigenvalues in this fit. Its orientation chart spans
4.10°, and the covariance condition number is about 368. Neither the fit
nor endpoint ancestry determines an equilibrium mixture weight.

The Gaussian is transformed to the proper relative frame of neighbor A.
The independent lab/frame-A density comparison agrees within 9.5×10⁻¹⁴.
The frozen 119-component guide allocates:

- 0.5 to the separately refined AB native Gaussian;
- 0.25 to the unchanged 117-component previous atlas;
- 0.25 to the new two-contact competitor Gaussian.

Within each retained family, the original means, covariances and relative
weights are unchanged. The normalized global proposal is

`g(x)=0.1 U_cube,Haar(x) + 0.9 G_A(x)`.

The uniform cube has side 36 Å and contains the entire physical capture
ball. Gaussian draws remain untruncated. The complete Gaussian sum and
the cube contribution enter every denominator, irrespective of which
component generated the draw. Hard-invalid and capture-invalid draws
remain zeros in the unconditional sample count. Both neighbors remain in
the physical hard predicate and the full exclusion union.

The frozen files are in `runs/ab-competing-frozen-atlas-20260920`:

- `model-mixture.json`: SHA-256
  `5226ea364a07b30be07b48e4d76f4f8b0daa9b5965bd5a93d09edc16fb4b95a0`;
- `model-competitor.json`: SHA-256
  `2a63ae8950bb512bb7659b6bb74e63f6c80020ebd7cf17c696b7c1abc5da6e1e`;
- `region-competitor-r3.json`: the complete radius-three ellipsoid in this
  fixed chart, intersected with original q≥5, physical capture, and AB hard
  support. Its hash is
  `e4b8cf45567ccc8a8106e4853fc5cdfab3b56f21dcc9abba86145acb3d223a98`.

This finite region supports two independent integrations of exactly the
same physical quantity: global importance sampling with its full proposal
density, and uniform six-dimensional chart-ball sampling with its exact
translation/Haar Jacobian. The global calculation also reports the entire
complement. Agreement for the finite region does not establish coverage of
other competing regions.

## Selected proposal anchor

`basin-normalizer --proposal-anchor-index 0` chooses A only as the proposal
coordinate reference. It retains the complete fixed-neighbor list for hard
and depletion calculations. The default still averages the atlas over all
fixed neighbors with its original draw streams. In the screen, none of the
512 B-reference chart draws even entered capture; removing this proposal
choice avoids measured wasted work without removing physical interactions.

The selected-anchor path stops on a numerical null instead of counting an
unrepresentable pose as an ordinary zero contribution. An explicit control
uses finite Gaussian parameters whose translated center exceeds f64 pose
range; the old path records numerical nulls, while the selected path fails
before writing a completed estimate. There is no retry.

The independent schema-2 auditor reconstructs the full Gaussian/Haar/cube
density and original q/capture labels for every actual pose, checks complete
physical inputs and hashes, and rejects numerical-null rows. It currently
supports Gaussian components only and rejects non-Gaussian degrees of
freedom explicitly. It does not infer an accepted-pose distribution or use
valid-only normalization.

Three Rust reference tests cover exact sphere depletion and hard-region
integrals, a harmless proposal anchor with all physical interaction supplied
by the other neighbor, both anchor selections, CLI/API agreement, and the
numerical failure. Default replay against the previous frozen binary is
byte identical for 4,096 two-neighbor sphere draws and 1,024 protein draws
at positive depletion. Seven Python controls check preparation, density and
input corruption, coupled chart inversion, and unconditional zero handling.

## Bounded next calculation

A separately authorized 512-pose geometry forecast draws 256 candidates
from each newly emphasized Gaussian. It finds 62 valid native candidates
and 93 valid distant candidates; all 512 enter capture. Combining these
rates with the earlier chart and uniform probes predicts roughly 19.3%
hard/capture-valid draws under the complete proposal. The measured-cost
projection for four populations of 8,192 draws is approximately 268–458
sampler CPU seconds. This extrapolates cloud cost from earlier configurations
and provides no guarantee of precision, variance, or elapsed runtime.

The completed pilot used seed base 99151010, as archived in
`runs/ab-cooperative-weight-pilot-20260920/protocol.json`. The preparation
form of that command is shown below. Its output directory now contains
completed results; use a fresh output path for any new campaign.

```bash
/home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/run_basin_normalizer_campaign.py \
  --out /home/xvg/tetramer-mc/runs/ab-global-contact-4x8192-l64-20260920 \
  --config /home/xvg/tetramer-mc/runs/ab-competing-frozen-atlas-20260920/config.json \
  --model /home/xvg/tetramer-mc/runs/ab-competing-frozen-atlas-20260920/model-mixture.json \
  --binary /home/xvg/tetramer-mc/target/release/basin-normalizer \
  --scales 1 --replicates 4 --samples 8192 --cloud-replicates 2 \
  --lambda-ratio 64 --proposal-anchor-index 0 --seed 99151010 \
  --workers 4 --prepare-only
```

`tools/analyze_global_fixed_region.py` extracts the frozen R3 contribution
using every original global importance weight and all unconditional zeros.
It reconstructs every pose's latent coordinates, reports the complementary
mass, per-population estimates, observed RSE/ESS, largest contributions, and
the paired-cloud diagnostic. Its optional independent uniform-region
comparison requires identical region hashes and disjoint production seeds.
The statistical errors remain observed-sample diagnostics; missing modes
cannot be excluded by a finite favorable result.

## Completed independent pilot

All four global populations and four uniform-R3 populations completed and
passed their independent audits. Each campaign contains 32,768 unconditional
draws, with Poisson intensity/activity 64 and two clouds per valid pose.

| Physical contribution | log Q | Observed RSE | Importance ESS | Largest contribution |
| --- | ---: | ---: | ---: | ---: |
| Native, global proposal | 35.75204 | 6.03% | 272.3 | 2.85% |
| All other poses, global proposal | 15.29501 | 28.28% | 12.50 | 18.34% |
| Frozen competitor R3, global proposal | 13.95199 | 25.63% | 15.22 | 23.86% |
| Same frozen R3, uniform chart-ball reference | 14.43700 | 14.72% | 46.08 | 12.52% |

The global native estimate agrees with the separately measured native-region
mass. The R3 constructions differ by 0.485 log units, or 1.78 combined
observed standard errors. Their population log estimates are respectively
14.4273, 13.7984, 13.7195, 13.6527 and
14.8453, 14.1807, 14.3791, 14.1860. These remain moderately noisy regional
controls, not evidence of global convergence.

The observed native-versus-other difference is 20.457 kBT. However, the
global campaign has **zero valid shoulder observations** (1<q<2) and only
**one intermediate observation** (2≤q<5). Its near-unity estimated native
fraction is therefore not a converged global probability. Positive uniform
support cannot substitute for measured coverage of this missing region.

About 74% of the observed other-region weight lies outside the frozen R3
ball. The leading two other contributions, both from the new competitor
Gaussian, lie at chart radii 3.49 and 3.05 and carry 18.34% and 17.67% of
that estimate. This identifies another tail-coverage issue. A contribution
from old component 97 has a substantially different pose; fitted Gaussian
labels should not be interpreted as a complete physical basin partition.
The 16 largest original other contributions are preserved, with independent
atomic checks, in `assessment/top-other-contributors.json`.

All 32,768 global poses passed independent full-density, q and capture
reconstruction. Independent matrix and quaternion chart radii agree to
1.61×10⁻¹¹ relative error. A separate check of 128 randomly selected
unconditional poses plus the 16 largest global contributors inspected 144
distinct poses against both complete atom unions; all physical labels
agreed. The additional top-other check also agreed. These audits leave
sampling coverage and rare-weight uncertainty as separate questions.

The global campaign cost 270.37 sampler CPU seconds, within the geometric
forecast; the uniform-R3 reference cost 289.30 CPU seconds. They establish
usable two-neighbor normalizers and corroborate the native contribution,
while motivating broader frozen-proposal controls and explicit coverage of
near-native competing poses. No proposal was refitted to these observations.

## Width controls expose missing competing weight

The completed controls in `runs/ab-global-widths-2-4-4x16384-l64-20260920`
multiply every Gaussian standard deviation by two or four. They retain
the same 119 means, relative mixture weights, selected A frame, 0.1 uniform
defense, physical domain and full AB target. There are four independent
populations of 16,384 draws at each width: **131,072 new unconditional
draws**, separate from the nominal pilot. The seed base is 99221010;
scale-index and population offsets are 100003 and 1009, respectively.

| Proposal width | log Q native | Native ESS | log Q other | Other ESS | Shoulder observations | Intermediate observations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, original pilot | 35.7520 | 272.3 | 15.2950 | 12.50 | 0 | 1 |
| 2 | 35.6648 | 62.92 | 16.5561 | 18.95 | 0 | 6 |
| 4 | 36.4828 | 2.01 | 26.2576 | 2.55 | 66 | 9 |

Width four finds substantial near-native competing contributions that the
other proposals missed. Its observed other mass rises by almost 11 log
units relative to the nominal pilot. **The earlier 20.46 kBT gap is therefore
not robust to proposal coverage.** The width-four estimates themselves
are highly concentrated: observed RSE is 70.5% for native and 62.6% for
other mass. These results establish consequential missed physical weight at finite
sampling budgets, not a replacement converged gap.

All 66 shoulder poses passed direct atomic hard/contact checks against both
neighbors, original-q reconstruction and independent full-density evaluation.
Their q values span 1.00068–1.79701. Sixty-one were generated by the widened
native component; five came from retained older components. The three
largest shoulder contributions carry 57.79%, 18.88% and 14.23% of the
observed shoulder weight. All three contact both neighbors, at q=1.06176,
1.00078 and 1.17684.

At those three poses the recorded mean log Poisson factors are 47.03,
45.89 and 41.51, and full log proposal densities are 10.23, 10.21 and 6.12.
The widened proposal increases the density at these same poses by 4.60,
6.55 and 6.59 log units relative to the nominal complete proposal. These
are same-pose proposal comparisons, not alternative physical weights.
The paired-cloud diagnostic attributes approximately 55% of the observed
shoulder variance to cloud noise; pose coverage remains an additional
limitation.

Broadening also loses efficiency in a previously identified finite region.
The identical frozen R3 integral gives log Q=14.1646 at width two and
10.5852 at width four, with ESS 2.75 and 5.17. The independent uniform-R3
reference is 14.4370. Width four underestimates this known regional mass
by 3.85 log units, differing by 6.64 combined observed standard errors.
No unknown global competitor is needed to demonstrate this coverage failure.
Each width is analyzed separately using the full original denominator.

All newly observed shoulders, the 16 largest other contributions and the
16 largest native contributions per width are preserved in
`assessment/contact-geometry-audit/`, with their original poses, cloud
records, proposal densities and independent atomic checks. These observations
are frozen discovery evidence for the next controlled calculation; they
have not been used to refit a proposal. The controls cost 166.88 and 60.77
sampler CPU seconds at widths two and four, respectively.

The next required reference is the **complete original 1<q<2 region**,
retaining the original classification while constructing a conservative
geometric cover of q≤2. A guided local region can be a useful control, but
cannot replace that remaining-region coverage requirement. The physical
model and assembly conclusion remain open.
