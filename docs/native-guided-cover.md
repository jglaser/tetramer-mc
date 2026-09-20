# Frozen covariance guide with a complete geometric cover

The native-region normalizer can now mix a frozen full-covariance pose atlas
with its geometric cover. The purpose is to place more proposal density
along coupled translations and rotations learned from previously weighted
poses, while retaining support over the entire original native region.
The physical target, native metric, and fixed-neighbor list are unchanged.

## Sampling law and correctness

Write a pose as `x=(t,R)` and use translation volume times **normalized**
SO(3) Haar measure. The target integrand is

`f(x)=1_capture(x) 1_original_q<=1(x) 1_hard(x) exp[z C(x)]`,

where `C` is overlap with the **union of all fixed-neighbor exclusion
regions**, not a sum of pair overlaps. Both neighbors in the AB experiment
participate in the hard and depletion tests.

The optional proposal is

`g(x)=(1-beta) g_cover(x) + beta [(1-epsilon) G(x) + epsilon U(x)]`.

`g_cover` is the existing normalized mixture of geometric covers, including
the full outer cover. `G` is the normalized frozen Gaussian atlas including
its Cayley-to-Haar Jacobian. `U` is uniform translation in the cube of side
`2*capture_radius` centered at `capture_center`, and uniform Haar orientation.
The cube is only a proposal component: capture still uses the original
sphere. Gaussian draws are not truncated or resampled after a hard clash.
Every sample uses the **sum of all applicable density terms**, regardless
of the family or Gaussian component that generated it.

One explicitly selected fixed neighbor defines the guide's relative-pose
coordinate frame. It does not remove other physical neighbors. In that frame,
the atlas uses the left Cayley residual with scaled coordinates
`c=ell*quat_vector(R_rel R_anchor^T)/quat_scalar(R_rel R_anchor^T)`.
Its physical density includes `ell^3*pi^2*(1+|c/ell|^2)^2`.
The cube remains centered in laboratory coordinates. The implementation
reuses the existing open-boundary atlas sampler and evaluates its density
after shifting the capture center; these shifts cancel in relative geometry.

For each valid draw the existing independent Poisson clouds give a positive
estimator `Wbar` with conditional expectation `exp[z C(x)]`. The recorded
contribution is `Y=1_capture 1_native 1_hard Wbar/g(x)`. Invalid draws
contribute zero and remain in the fixed unconditional sample count `N`.
Thus, conditional on any already frozen training data and model,

`E[(1/N) sum Y | frozen model] = integral f(x) dx`.

Fitting or choosing the frozen model from earlier samples does not change
this identity for fresh independent draws. It does not make the earlier
training samples independent validation or establish adequate coverage at a
finite budget. Logs and ratios of estimated
normalizers are not themselves unbiased. This is an independent importance
integrator, not a new adaptive Markov-chain invariance result. The abstract
measure-theoretic importance identity is also available in the Lean
formalization; that proof does not verify floating-point geometry or this
concrete sampler implementation.

In this bounded native target, the outer cover also supplies a simple
finite-variance argument. If its total proposal weight is `delta>0` and its
volume is `V`, then `g>=delta/V` throughout the target. A finite upper bound
`Cmax` on the mobile exclusion volume bounds every overlap. For the present
Poisson factor,
`E[W^2|x]=exp[2*z*C+z^2*(C-L)/lambda]`, with `0<=L<=C<=Cmax`.
Consequently `E[Y^2]<=V^2/delta * exp[(2*z+z^2/lambda)*Cmax]`;
averaging independent clouds can only reduce this second moment. This bound
is finite but generally far too loose to certify useful precision here.
Observed errors still require independent coverage and scaling controls.

## Implementation and checks

`native-region-normalizer` accepts `--model`, `--model-weight`,
`--model-uniform-probability`, and `--model-anchor-index`. The default
without `--model` retains its original sampling streams and sample encoding.
Schema 3 archives the model and records the complete guide law, selected
families, and full density on every draw, including zeros. Numerical nulls
fail the run instead of retrying until a valid pose appears.

The generic campaign runner freezes the executable and guide model and
records their hashes. The independent Python auditor reconstructs the
relative quaternion geometry, Cholesky density, uniform cube, and geometric
covers, checks every recorded density and importance weight, and uses
importance weights for the hard-volume estimate even with a single cover.

Validation includes:

- Independent analytic density evaluation with shifted means, coupled
  covariance, rotated anchors, and a nonzero capture center.
- Known-volume integration with capture rejection and the original native
  metric.
- A sphere depletion reference where the guide anchor is harmless but the
  second physical neighbor creates all hard and depletion interactions.
- Default sphere and nested-cover protein replay against the previous
  frozen executable: 16,384 and 4,096 rows respectively, byte identical.
- A 512-draw real-binary launcher smoke with independent scalar and
  vectorized density checks; maximum log-density disagreement `4.49e-14`.
- Ten Rust native-region tests and 31 Python native-analysis/fitter tests.

The replay records are in
`runs/native-hybrid-validation-20260920/final-replay.json`; the launcher
smoke is in `runs/native-hybrid-launcher-smoke-20260920`.

## Frozen protein comparison

The training procedure and its limitations are in
[native-ab-covariance-guide-plan.md](native-ab-covariance-guide-plan.md).
It uses all 18,577 original nonzero rows from eight prior populations, with
their original importance weights. Their effective size is only 3.97.
The stated validation rule selected one Gaussian with a covariance floor;
a separate model multiplies its covariance by four. Neither covariance is
claimed to be a converged equilibrium covariance.

The independent comparison was frozen before production in
`runs/native-hybrid-validation-20260920/fresh-protocol.json`:

- Original AB native region; radius 1.5 Å and activity 0.035 Å^-3.
- Four populations of 8,192 unconditional draws per model.
- `beta=.75`, `epsilon=.05`, one full geometric cover, reference neighbor A.
- Poisson intensity/activity 64 and two independent clouds per valid pose.
- Selected-model seeds `99011010+1009*i`; wide-model seeds `99021010+1009*i`.
- Fixed budgets, no replacement or outcome-dependent stopping, and no
  pooling of training observations with fresh validation.

Campaign paths are `runs/native-ab-guided-selected-4x8192-l64-20260920`
and `runs/native-ab-guided-wide-4x8192-l64-20260920`. The prespecified
diagnostics are population agreement, weight concentration, proposal-width
sensitivity, conditional cloud noise, and observed precision per CPU time.
These are native-region integration diagnostics; they do not by themselves
establish competing-region coverage, Markov-chain mixing, or assembly.

## Fresh results and larger confirmation

All eight pilot populations completed normally. Every row, full proposal
density, original native metric, cloud weight, sample hash, and frozen input
passed the independent streaming audit. An additional exact atomic audit of
144 poses per pilot (128 uniform row selections plus the 16 largest original
weights, with duplicates removed) agreed with all hard, capture and native
labels for both fixed neighbors.

The guide-width pilot gave similar observed precision per CPU, so a separate
eight-population wide-guide confirmation was frozen before new outcomes:
32,768 draws per population, seeds `99051010+1009*i`, and exactly the same
model, physical target, mixture weights and cloud settings. This is eight
times the pilot's total draws and four times its population size. Its protocol
is `runs/native-hybrid-validation-20260920/wide-confirmation-protocol.json`;
raw outputs are
`runs/native-ab-guided-wide-confirmation-8x32768-l64-20260920`.

| Campaign | Draws | log Q | Observed row / population RSE | ESS | Largest contribution | Sampling CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source nested covers, training only | 1,048,576 | 36.5467 | 50.18% / 47.92% | 3.97 | 47.72% | 915.6 |
| Selected Gaussian pilot | 32,768 | 35.5758 | 5.25% / 1.80% | 358.65 | 2.44% | 293.0 |
| Twice-width Gaussian pilot | 32,768 | 35.6988 | 9.32% / 14.40% | 114.70 | 4.36% | 85.9 |
| Fresh twice-width confirmation | 262,144 | 35.7424 | 3.81% / 2.51% | 688.49 | 1.31% | 708.1 |

The confirmation's eight population log weights range from 35.6234 to
35.8094. Its mean is 4.45% above the independent wider pilot, a difference
of 0.44 combined observed linear-scale standard errors. The larger run
therefore supports the wider guide's pilot estimate with improved observed
precision. The full 262,144-row audit passed, including every frozen seed,
budget, executable/model/configuration hash and original importance weight.

The confirmation is **18.12% above the narrow-guide pilot**, however. This
is 2.62 combined observed pointwise standard errors on the linear scale,
or 5.23 using the less stable four/eight-population error estimates. These
are diagnostics, not calibrated significance tests; the residual proposal
sensitivity should not be hidden by pooling the campaigns. The small
narrow-pilot population scatter did not establish accuracy. No sample was
reweighted under a newly fitted model or removed as an outlier.

Paired independent cloud differences attribute about 41%, 34%, and 31% of
the observed variance to Poisson noise in the selected pilot, wider pilot,
and wider confirmation respectively. Pose variability remains the larger
observed contribution. The independent noise audit and comparison tool use
the original two clouds, preserve all zero draws, and agree after accounting
for the stated finite-N sample-variance convention.

There is also a distinct target limitation. These guides concentrate on
large **depletion-weighted** contributions, so their simultaneous zero-activity
hard-volume estimates are poor: hard ESS is 1.00, 3.29, and 14.57 respectively.
In the confirmation, 15 outer-cover hits carry 95.8% of estimated hard volume.
Use the earlier geometric controls for the hard-volume factorial contrast;
do not substitute these noisier estimates. Likewise, the confirmation's
`0.8<q<=1` shell has only ESS 1.45, despite contributing little observed
physical mass. Neither shell absence nor global convergence is certified.

The archived pilot comparison is
`runs/native-ab-guided-comparison-20260920`; the separate larger comparison
is `runs/native-ab-guided-confirmation-comparison-20260920`. Both preserve
the training/validation distinction, all source hashes, population estimates,
original-q histograms, guide/cover labels, and the executed analysis code.

![Independent guide-width and population-size controls](../runs/native-ab-guided-confirmation-comparison-20260920/native-guided-comparison.png)

This resolves a substantial part of the **integration-efficiency** problem:
coupled covariance proposals distribute weight over many more fresh poses.
It does not yet establish a fully controlled native mass, coverage of all
competing environments, contact mixing, or template-free self-assembly.

## Locating the remaining width discrepancy

A post hoc diagnostic partitions all original contributions using the same
selected-Gaussian Mahalanobis radius
`r=|L^-1(x-mu)|`, where `L L^T=Sigma` and `x` is the exact scaled left-Cayley
deployment coordinate. This six-dimensional radius is a proposal diagnostic,
not a new physical basin definition. Both pilot campaigns and the larger
confirmation retain their original denominators and full unconditional N;
there is no refit, reweighting or additional simulation.

For **r<=4**, selected and wide-confirmation log weights are **35.53786 and
35.54056**. The observed central-region estimates are nearly identical.
The region **r>4 accounts for 98.6% of their total observed mass difference**.
It supplies 3.72% of the selected estimate, 12.24% of the wider pilot, and
18.27% of the larger wider estimate.

| Selected-chart radius | Confirmation fraction of estimated native mass | Band ESS | Observed band RSE |
| --- | ---: | ---: | ---: |
| 4 < r <= 5 | 8.87% | 94.5 | 10.3% |
| 5 < r <= 8 | 8.98% | 21.4 | 21.6% |
| r > 8 | 0.42% | 4.18 | 48.9% |

The selected pilot had no nonzero observations in the 5–8 band. That empty
sample cannot certify absent physical mass. The measured residual therefore
points specifically to coverage of the Gaussian tails, rather than a
disagreement in the central fitted region. It also localizes the remaining
uncertainty: a stable pooled estimate does not imply equally precise tails.
The 98.6% attribution is arithmetic on these samples, not a confidence bound
on missing mass.

Every source checksum was verified again. Independent scalar quaternion/
Cholesky and matrix/SciPy evaluators agree over all 327,680 recorded poses
plus 65 controls, with maximum relative latent-coordinate difference
`2.42e-11`. The archived script, per-band and per-population estimates, and
gap decomposition are in
`runs/native-hybrid-validation-20260920/native-mahalanobis-tail-diagnostic.*`.
The next proposal refinement should address this measured tail coverage,
keeping fresh validation and the geometric support component.
