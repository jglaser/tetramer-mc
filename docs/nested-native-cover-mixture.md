# Nested native-cover mixture: independent sampling and review

This proposal concentrates independent integration draws near the native
reference while retaining support for the **entire original native region**.
It changes the proposal only. The native metric, capture sphere, hard shape,
fixed-neighbor poses, and depletant bath remain unchanged.

The fixed first screen uses scales `(0.1, 0.2, 0.4, 1)` with equal component
probabilities. No Gaussian model or learned density enters this proposal.

## Measure and density

For reference `(t₀,R₀)` and member centroid c, use

`w=t−t₀+(R−R₀)c`, `θ=angle(R₀⁻¹R)`.

The existing conservative outer cover has `|w|≤a` and `θ≤θ_max`. Component
i has `a_i=s_i a` and `θ_i=s_i θ_max`. Its volume under ordinary translation
measure and normalized SO(3) Haar measure is

`V_i=(4 a_i³/3)(θ_i−sin θ_i)`.

The translation change of variables has Jacobian one at fixed rotation;
left multiplication by R₀ preserves Haar measure. Sample the axis uniformly,
the angle from CDF `(θ−sin θ)/(θ_i−sin θ_i)`, and w uniformly in its ball.
Multiplying a draw from the original angular cap by s_i would not give the
correct new Haar-cap law. Construct the smaller cap first, then invert its
CDF. Stable small-angle evaluation of `θ−sin θ` is required.

Choose a component with fixed probability α_i. The resulting density is

`g(x)=Σ_i α_i 1_(C_i)(x)/V_i`.

Every pose uses this complete sum, including all containing covers. The
selected component's individual density is not the proposal marginal.
The scale-1 component has α=1/4, so `g≥1/(4 V_outer)` throughout the original
target. In particular the geometric inverse-density factor is bounded by
`4 V_outer`; narrow components cannot remove outer support.

For `h=1_(q≤1) 1_capture H`, the physical and hard-only estimators are

`Q̂_z=(1/N)Σ h(x_n) W̄(x_n)/g(x_n)`,
`Q̂_0=(1/N)Σ h(x_n)/g(x_n)`.

The original metric q is always evaluated. Small covers may contain points
outside scaled native subregions; this does not affect the argument. All N
draws, including every rejection zero, remain in each denominator. The
positive overlap estimator satisfies `E[W̄|x]=exp[z C(x)]` for the full
fixed-neighbor exclusion union, so conditioning on x proves unbiasedness
of each mass estimate in the idealized sampling model. Their logarithms
and ratios are not unbiased.

## Support conventions and independent reconstruction

The previous `NativeCover::contains` was a conservative validation helper:
it expanded radius by `1+1e−12` and angle by `1e−12`. Such a predicate cannot
be reused to define g with the nominal volumes. A positive-thickness
expansion changes the normalization, even if the numerical effect is tiny.
The density uses uninflated `|w|≤a_i` and `θ≤θ_i`; equality is immaterial for
the ideal continuous measure. An actual support expansion would have to
change both sampling and V_i consistently.

For normalized quaternions, `|q₀·q|≥cos(θ_i/2)` is a convenient ordinary-cap
test, but loses small-angle resolution near one. The implemented independent
audit forms the relative quaternion and uses
`θ=2 atan2(|vector part|, |scalar part|)`. It is invariant under quaternion
sign and resolves small caps without adding a support tolerance. Geometry
uses normalized quaternions, consistently with the Rust shape transform.
These are ordinary FP64 checks, not formal interval certificates.

Schema-2 mixture rows record pose, selected component and full log g for
every draw. Valid rows also record `log_hard_weight=−log g`. The Python
streaming audit reconstructs the metric, support membership, mixture
density, cloud weights, physical and hard moments, and their cross moment.
It checks hashes and every unconditional row. Using `V_outer*valid/N` would
be wrong for this mixture and is restricted to the legacy uniform path.

Single-scale `[1]` keeps the old sampling law and row encoding. A separate
component stream leaves the pose and cloud streams unchanged; default-mode
bit replay is a required Rust control. The updated analyzer preserves the
previous physical statistics exactly for an archived 1,048,576-row AB
population. Its old source is archived before modification.

## Independent test requirements

For nested covers ordered by size, set V₀=0 and
`G_j=Σ_(i≥j) α_i/V_i`. On shell `C_j \ C_(j−1)`, g is constant at G_j.
A one-member sphere with an unrestricted capture sphere has target exactly
the outer cover. Consequently,

`Σ_j (V_j−V_(j−1)) G_j = 1`,
`E[1/g] = V_outer`,
`E[(1/g)²] = Σ_j (V_j−V_(j−1))/G_j`.

These supply exact mean and variance references for a stochastic Rust test
whose proposal components overlap. For the present protein cap and four
scales, the full-cover sphere test has per-draw relative variance 2.98366.
This variance is expected: the mixture spends three quarters of its effort
near a point even when the target is uniform.

Additional controls should include:

- A centered capture ball smaller than the translation support: its target
  volume scales as the capture-radius cube while the orientation factor is
  unchanged. This exercises explicit zero contributions and weighted hard
  volume.
- A hard exclusion with an analytic sphere-lens volume, as a second
  rejection test independent of capture clipping. For a unit translation
  ball excluding a unit ball whose center is distance one away, the allowed
  translation volume is `11π/12`; unrestricted orientations have Haar mass
  one. Physical sphere radii can be chosen so their summed core radius is one.
- Off-center member centroid, rotated/translated reference frames,
  quaternion sign, and narrow-cap support. The full density must remain
  invariant under a common rigid transformation.
- Unchanged original q, including mixture-generated points near target
  boundaries. Deliberately altered target labels must fail the audit.
- Normalized mixture weights, component support, volume reconstruction,
  `[1]` row replay, empty/all-zero cases, and rejection of invalid scales or
  a mixture lacking a positive outer-cover component.

Thirteen current Python controls pass, covering the density shell identities,
support boundaries and tiny caps, frame changes, original-q reconstruction,
hard/physical covariance, legacy zero/one-population handling and pooling
guards. The factorial regression also fails if a nonuniform hard volume is
incorrectly replaced by `V_outer*valid/N`.

Seven Rust controls passed before launch, including exact shell normalization,
strict/tiny-cap support, unchanged legacy RNG, weighted radial/angular/joint
moments, world-frame invariance, capture zeros, the analytic AO sphere
integral, and the existing cover checks. The release executable SHA-256 is
`2816158e83b5d91e4790a1ac76455b36215645b57aed155d84c9a96f3b9d6669`.
A default-mode protein replay produced **32,768 byte-identical original
rows**, sample SHA-256
`c57b28c5f58089529cf1b1c7eed081595671c6def0fc657440989ec0275d1a96`;
see [replay.json](/home/xvg/tetramer-mc/runs/native-cover-mixture-validation-20260920/replay.json).

The complete launcher/preflight/streaming pipeline also passed a separate
synthetic sphere test with distant, noninteracting fixed neighbors. Its
explicitly reduced test budget was eight populations of 128 draws; every
audit passed, physical and hard weights agreed, and each target estimate
agreed with the exact geometric volume. The test took 3.9 seconds and is
recorded in
[smoke-result.json](/home/xvg/tetramer-mc/runs/native-cover-mixture-runner-smoke-20260920/smoke-result.json).
This fixture is not included in the protein estimates.

## Variance accounting

The first implementation chooses components randomly and independently.
With `X=hW̄/g`, `Y=h/g`, ordinary fixed-N sample variance and covariance apply.
For M conditionally independent clouds,

`Var(Q̂_z) = [∫ h(F²+Var(W|x)/M)/g dμ − Q_z²]/N`,

where `F=exp(zC)`. Average the clouds within each pose before computing
moments and ESS. Two clouds are not two independent pose draws. Hard and
physical contributions are paired; the log-enhancement variance includes
their covariance. The audit records the cross sum `Σ XY` for that purpose.

Fixed component counts would be a different design. If `β_i=N_i/N`, a
pooled average must use g_β, or else combine component means with the α_i
weights appropriate to g_α. Its variance must be computed within strata:
`Σ_i β_i² s_i²/N_i`. The ordinary pooled IID ESS-to-error identity is not
that stratified variance. Deterministic stratification is not used in the
first screen.

Observed ESS and sample/replicate errors remain concentration diagnostics.
They do not certify that rare high-weight regions were seen. The mixture
has complete geometric support, but finite integration budgets can still
miss important physical weight.

## Declared first screen and asymmetric expectations

The immutable plan is
[predeclared-pilot.json](/home/xvg/tetramer-mc/runs/nested-native-cover-review-20260920/predeclared-pilot.json).
For each of empty, A, B and AB it specifies four independent populations of
16,384 draws: 262,144 draws total, maximum eight workers, λ/z=64 and two
clouds. Seed bases are 98911010 / 98921010 / 98931010 / 98941010 in that order,
plus 1009 times the population index. No result-dependent stopping or
replacement is allowed. Previous uniform estimates remain separate controls.

A retrospective membership screen of existing valid poses found:

| Target | Valid old poses in scales .1 / .2 / .4 / 1 | Original weight inside scale .1 | Original weight inside scale .4 |
|---|---|---:|---:|
| A | 2 / 157 / 4421 / 20041 | 0.0031% | 9.28% |
| AB | 1 / 6 / 120 / 872 | 98.52% | 99.65% |

This motivates AB as the focused target and A as a degradation control.
More than 90% of A's observed weight lies outside the scale-.4 cover, where
the equal mixture has only one quarter of the old proposal density. Complete
support permits improvement but does not ensure it. B supplies the other
single-neighbor control, and empty checks pure geometry without cloud noise.

Extrapolating old valid counts predicts mixed hard-valid fractions around
4.1% for A and 6.7% for AB, roughly a few hundred CPU seconds for those two
groups at the fixed budget. The narrowest-cover estimates use only two and
one old events respectively; they are uncertain cost estimates, not
precision forecasts. Report per-component occupancy, Q_z/Q_0 separately,
core/shell weights, largest contributions, ESS, independent-population
variation, cloud count, and CPU time. A change in observed ESS per second
alone is not a validated speedup when the reference mass is unresolved.

The runner defaults to preparation only. After Rust tests and release-binary
validation, the explicit execution command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/run_native_cover_mixture_campaign.py \
  --binary target/release/native-region-normalizer \
  --out runs/native-cover-mixture-4x16384-l64-20260920 --execute
```

The execution directory must not already exist. It freezes the executable,
configuration and analyzer sources, repeats the independent geometry and
physical-identity checks, records process handles and statuses, and streams
all four target audits after simulation. Preparation itself launches no
physical production.

The authorized protein pilot was launched after these controls, using the
exact release hash above. Its separate execution manifest and owner handle
are under
[native-cover-mixture-4x16384-l64-20260920](/home/xvg/tetramer-mc/runs/native-cover-mixture-4x16384-l64-20260920).
The workload guard is 262,144 fixed draws and at most eight simulation
workers, followed by one four-worker target audit at a time. There is no
wall-time/CPU cutoff; conditional Poisson counts have unbounded support.
The runtime estimates above are forecasts, not guaranteed limits.

## Completed pilot: AB improves, the comparison remains unresolved

All sixteen populations completed their prescribed draws and all four
streaming audits passed. Owner session 69794 was verified terminal with exit
code zero. Total simulation cost was **312.19 CPU seconds**. The new data are
kept separate from every earlier uniform campaign.

Each new target used 65,536 unconditional draws. The observed physical
normalizers and concentration diagnostics are:

| Target | New log Q(z) | New ESS | New largest weight | New CPU s | Previous uniform log Q(z) | Previous ESS |
|---|---:|---:|---:|---:|---:|---:|
| Empty | −8.76360 | 337.69 | 0.38% | 0.50 | −8.70790 | 18,029 |
| A | 13.19663 | 3.88 | 47.36% | 99.14 | 15.26037 | 11.32 |
| B | 12.31836 | 4.31 | 47.53% | 156.79 | 12.27129 | 80.99 |
| AB | 35.07230 | 4.98 | 40.45% | 55.76 | 35.84604 | 1.03 |

The previous columns used different, larger fixed budgets: empty 1,048,576,
A 33,554,432, B and AB 4,194,304 each. Their original costs and concentration
are in [the uniform factorial report](native-factorial-screen.md). Comparing
raw draw counts alone would misrepresent work: the new AB run has 64 times
fewer proposal draws, but 1,119 valid cloud evaluations versus 872 before,
and costs 55.76 rather than 46.24 CPU seconds.

AB's largest contribution fell from 98.52% to 40.45%, and its observed ESS
rose from 1.03 to 4.98. Its point-based relative SE is still 44.8%, while the
independent-population relative SE is 69.9%. Its four population log masses
are 33.4632, 36.1965, 34.5010 and 33.2226. This is a useful reduction in
observed concentration, **not** a resolved normalizer or demonstrated
rare-tail convergence.

A's new mass is about 2.06 log units below its previous estimate. This is
consistent with the anticipated loss of effort in the broad region carrying
over 90% of A's observed old weight. Complete support alone does not make the
small new budget adequate there. B's mass agrees with the older result but
its new concentration and cost are worse. The equal centered mixture is
therefore not a general-purpose improvement for all neighboring environments.

The independently weighted hard volumes are:

| Target | New Q(0), Å³ | Observed point SE, Å³ | Previous uniform Q(0), Å³ |
|---|---:|---:|---:|
| Empty | 1.56320×10⁻⁴ | 8.48476×10⁻⁶ | 1.65275×10⁻⁴ |
| A | 6.55975×10⁻⁶ | 1.76076×10⁻⁶ | 5.74122×10⁻⁶ |
| B | 1.27148×10⁻⁵ | 2.48967×10⁻⁶ | 1.12229×10⁻⁵ |
| AB | 1.40053×10⁻⁶ | 8.30016×10⁻⁷ | 1.99844×10⁻⁶ |

These use `mean I_valid/g`, not the much larger fraction of valid centered
draws multiplied by the outer volume. The empty result is consistent with
the earlier independent geometric estimate, providing a useful full-target
normalization control. The new hard-volume uncertainties are larger than
the earlier uniform ones, as expected when effort is concentrated in a small
core instead of spread across the geometric target.

Using only the four new, independent target groups, the factorial point
estimate is `log C(z)=0.79370`, with observed population delta-method
SE **1.02345**. The new hard-only result is `log C(0)=0.96503` with SE 0.86142;
their difference is −0.17133 with SE 1.40734. Paired physical/hard population
covariance is included, and the empty term cancels from that difference.
The factorial sign and magnitude remain unresolved. This contrast still
includes shared-pose correlations and is not the isolated triple-overlap
depletion penalty.

All new estimates, actual per-population hard weighting and error propagation
are preserved in
[factorial-assessment.json](/home/xvg/tetramer-mc/runs/native-cover-mixture-4x16384-l64-20260920/factorial-assessment.json).
Individual target `assessment-streaming.json` files contain every audited
population, proposal-component counts, hard and physical moments/covariance,
and leading original rows. No weights were replaced. A separately authorized
AB-only confirmation was subsequently declared and launched as described
below; it does not pool with this pilot.

## Independent AB-only confirmation

The pilot justified an independent precision check for AB only. Its
[prelaunch plan](/home/xvg/tetramer-mc/runs/nested-native-cover-review-20260920/ab-confirmation-plan.json)
fixes eight populations of 131,072 draws, seeds `98951010+1009*i`, maximum
eight workers, the same four equal-weight covers, λ/z=64 and two clouds.
It uses the exact same executable and AB configuration; total unconditional
budget is 1,048,576. The 892-CPU-second forecast is an extrapolation, not a
time limit or stopping criterion.

The reusable `tools/run_native_region_reference.py` now accepts optional
`--cover-scales`, `--cover-weights`, and `--expected-binary-sha256` arguments.
An independent 256-draw sphere/far-neighbor smoke test passed before launch,
including streaming reconstruction and the exact geometric-volume check.
Omitting the new proposal options retains the legacy command law.

The confirmation execution is separate under
[native-cover-mixture-ab-confirmation-8x131072-l64-20260920](/home/xvg/tetramer-mc/runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920).
The immutable plan, runner and analyzer sources are archived there. No A,
B or empty confirmation runs are included; the pilot did not establish a
benefit for those environments. Every original draw and population is
retained, with no weight replacement or pooling with previous data.

### Confirmation result: increasing this mixture's budget did not resolve the mass

All eight confirmation populations completed, and the frozen Python analyzer
validated every one of the 1,048,576 rows, original q labels, mixture densities,
cloud weights, moments and hashes. The result is

`log Q_AB = 36.54666`, point ESS **3.97**, largest draw **47.72%**,
observed point relative SE **50.18%**, and independent-population relative
SE **47.92%**. The hard volume is `(2.32075 ± 0.27201)×10⁻⁶ Å³`, where the
uncertainty is the observed point SE. Cost was **915.60 CPU seconds**, with
18,577 valid poses, 1.155 billion raw cloud points and 318 MB of archived rows.

| Confirmation population | log Q_AB | Within-population ESS | Largest within-population contribution | Share of the total confirmation mass |
|---|---:|---:|---:|---:|
| r00 | 37.98259 | 1.21 | 90.82% | 52.54% |
| r01 | 35.92830 | 7.96 | 22.84% | 6.74% |
| r02 | 36.76852 | 2.63 | 54.27% | 15.60% |
| r03 | 36.59422 | 1.29 | 87.82% | 13.11% |
| r04 | 34.74644 | 5.79 | 40.52% | 2.07% |
| r05 | 35.13724 | 2.90 | 58.12% | 3.05% |
| r06 | 35.32401 | 7.44 | 30.05% | 3.68% |
| r07 | 35.18612 | 10.22 | 20.95% | 3.21% |

Compared with the 65,536-draw mixture pilot, this independent run has sixteen
times the budget but a **1.47436 higher log mass** (about 4.37 times the point
estimate) and slightly lower ESS, 3.97 versus 4.98. It discovered heavier
contributions instead of establishing stable precision. The approximately
half-total leading contribution is r00/draw45335 at q=0.33145. The next two
are at q=0.21452 and q=0.18550. No contribution was replaced or resampled.

This is a negative scaling result for the present equal centered mixture.
It does not distinguish equilibrium model limitations from insufficient
importance coverage and does not establish a thermodynamic cooperativity
sign. The hard-volume comparison remains consistent with the independent
uniform geometry, so there is no indication here of a target-normalization
failure. Repeating a still larger budget with the identical mixture is not
the next planned action.

The leading pose lies just beyond the scale-.2 translation ball, where
removing that component makes the inverse mixture density jump sharply.
A shifted proposal with coupled translation/rotation covariance is a
possible next design. It must use all original training weights, preserve
a normalized defensive component, and be judged on fresh independent
draws; changing selected old weights would invalidate the original result.

The final [streaming assessment](/home/xvg/tetramer-mc/runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920/assessment-streaming.json)
and [population/leading-row concentration](/home/xvg/tetramer-mc/runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920/population-concentration.json)
retain the complete confirmation-only audit. Simulation owner session 32605
and analyzer session 36209 both terminated successfully. These data remain
separate from the pilot and the original uniform campaign.

![Independent AB confirmation and its concentration](/home/xvg/tetramer-mc/runs/native-cover-mixture-4x16384-l64-20260920/comparison/AB-native-confirmation.png)

![Registration-resolved AB weight across the separate runs](/home/xvg/tetramer-mc/runs/native-cover-mixture-4x16384-l64-20260920/comparison/AB-registration-weight.png)

The confirmation places 80.20% of its observed weight at q between 0.2 and
0.4. The paired-cloud difference attributes about 63.12% of the observed
variance to conditional cloud noise, itself a noisy descriptive decomposition
that cannot establish how much unseen geometric variation remains. The
[prepared covariance reuse design](native-ab-covariance-guide-plan.md)
addresses the shifted geometric concentration; it is not a replacement
for these original observations and includes no new production.
