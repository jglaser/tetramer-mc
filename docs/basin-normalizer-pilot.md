# Independent contact-region normalizer pilot

The independent calculation finds substantial weight in the near-native
shoulder `1 < q < 2`, a region absent from the previous SMC final populations.
It also exposes serious proposal-width and Poisson-weight variance. **The
pilot does not establish equilibrium occupancies or resolve the SMC
discrepancy.** The next calculation needs both reduced cloud noise and broader
coverage of competing contacts.

The physical system matches the involutive docking benchmark: one mobile
rigid tetramer, one fixed site0 neighbor, an 18 Å capture ball, depletant
radius 1.5 Å, and activity 0.035 Å⁻³. The same frozen five-component
conditional atlas supplies a normalized pose proposal. This is a native-informed
control, not template-free discovery. There is no Markov trajectory, annealing,
resampling, or online adaptation in this estimator.

For each of four Gaussian standard-deviation multipliers (0.5, 1, 2, 4),
four independent populations make 512 **unconditional** pose draws. Means,
component weights, atlas anchors and the 10% uniform cube/Haar branch stay
fixed; covariance matrices are multiplied by the square of the stated
multiplier. Invalid poses contribute zero, and all 512 draws remain in the
denominator. There are 8,192 independent pose draws in total. Two independent
Poisson clouds evaluate each hard-valid pose at auxiliary intensity
`lambda = 16 z`; their weights are averaged linearly.

The [campaign manifest](../runs/basin-normalizer-pilot-512/manifest.json)
archives the binary, source inputs, scripts and SHA-256 hashes. All sixteen
jobs completed, using approximately 65.5 CPU seconds. Detailed per-population
results, the ten exhaustive region estimates, source-component contributions,
and diagnostics are in
[analysis.json](../runs/basin-normalizer-pilot-512/assessment/analysis.json).

## Estimator and region definitions

The integral is `Q_b = integral H(x) I_capture(x) I_b(x) exp(z C(x)) dx`,
where `C` is mobile exclusion overlapping the union of fixed-neighbor
exclusions. Let `L` be an inner overlap volume and `U` an uncertain envelope
covering the remainder. Exact membership thinning gives

\[
 K\sim\operatorname{Poisson}(\lambda(C-L)),\qquad
 \widehat B=e^{zL}(1+z/\lambda)^K,\qquad
 E[\widehat B\mid x]=e^{zC(x)}.
\]

With two independent clouds, a valid pose contributes
`W = (B_1 + B_2)/(2 g(x))`; invalid poses contribute zero. Here `g` is the
**full normalized mixture density**, including the uniform branch and the
anchor average, with the translation/Haar Jacobian. The fixed-budget average
`sum W / N` estimates `Q`. Averaging `log W`, normalizing only by valid draws,
or dividing by just the selected Gaussian density would estimate a different
quantity. Ratios of estimated normalizers are finite-sample plug-in estimates;
they are not themselves unbiased.

The fixed exhaustive registration bins are `q <= 0.8`, `0.8 < q <= 1`,
`1 < q < 2`, `2 <= q < 5`, and `q >= 5`. Each is split by exact exclusion
contact into bound and unbound, giving ten regions. Native uses `q <= 1`,
matching the old SMC definition. Its complement includes shoulder,
intermediate, distant, and unbound configurations. Selected Gaussian labels
are auxiliary proposal labels, not physical basin definitions.

## Observed weights and sensitivity

| Std multiplier | log Q native | log Q other | Native fraction | Shoulder fraction | Distant fraction | Total observed ESS / 2048 | Top 1% weight share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 16.016 | 9.571 | 0.9984 | unobserved | 0.001585 | 2.87 | 92.73% |
| 1 | 15.112 | 15.274 | 0.4596 | 0.5398 | 0.000623 | 8.14 | 92.63% |
| 2 | 13.146 | 13.765 | 0.3501 | 0.5249 | 0.01352 | 3.15 | 99.55% |
| 4 | 11.560 | 10.649 | 0.7132 | 0.03129 | 0.1112 | 2.01 | 99.15% |

Fractions above use the summed observed normalizers,
not equally weighted accepted samples. A zero bin has no inferred upper
bound on its physical mass. The top 1% refers to 21 of the 2,048 unconditional
draws in each arm, including invalid zeros in that count.

At multiplier 1, only ten hard-valid shoulder draws supply 53.98% of the
observed total weight. Their ESS is 4.78. The native estimate has ESS 3.41
and one draw carries 53.0% of its weight. At multiplier 2, one of six shoulder
draws contributes 97.3% of that region's observed weight. Four independent
populations and these ESS values cannot support narrow uncertainty claims.

![Normalizer and variance pilot](../runs/basin-normalizer-pilot-512/assessment/normalizer-pilot.png)

The original-width native estimate, log Q = 15.112, is compatible with the
previous SMC value 15.079 within its large observed noise. However, its
distant estimate is only 8.509, whereas the previous other estimate was
15.103 and all stored other endpoints were distant. The shoulder result
therefore does not explain away the earlier distant weight. Possible missed
competing regions remain a central concern. The current pilot deliberately
keeps a narrow, previously learned atlas; nonzero full support does not imply
useful coverage at this sample count.

At multiplier 1, selected component 0 supplies all observed native/shoulder
weight; the other selected components and uniform branch supply the remaining
0.0623%. These are proposal-channel contributions, not probabilities of
physical basins. Broader multipliers increase rejection and still fail to
give stable competing-region normalizers.

## Separating cloud variance from pose variance

For independent cloud estimates `W1,W2` at each pose, `E[W1 W2 | x]` equals
the square of the exact importance weight, whereas
`E[(W1-W2)^2 | x]/2` is the variance of one cloud. Therefore

\[
 V_{\rm cloud}=\frac1N\sum_i\frac{(W_{i1}-W_{i2})^2}{4}
\]

estimates the cloud contribution to the per-draw variance of their average.
The sample variance of `(W1+W2)/2` minus this quantity estimates the remaining
pose contribution. The analyzer retains negative residual estimates if they
occur; finite noisy data need not produce a nonnegative decomposition.
Invalid pose weights remain zero in both calculations.

The cloud shares of observed total weight variance are 76.2%, 47.3%, 64.4%,
and 70.3% at multipliers 0.5, 1, 2, and 4. Native shares are 76.2%, 71.8%,
55.4%, and 70.1%; distant shares are much smaller, 2.7%–12.9%. These are noisy
diagnostics, not converged variance budgets. Increasing auxiliary intensity
can reduce native/shoulder cloud noise but cannot repair missing competing
pose coverage.

The conditional second-moment identity is

\[
 \frac{E[\widehat B^2\mid x]}{E[\widehat B\mid x]^2}
 =e^{z^2(C-L)/\lambda}\le e^{z^2|U|/\lambda}.
\]

The analyzer stores quantiles of this geometric upper bound in logarithmic
form. Typical log bounds are approximately 17 at original width, so the
envelope bound is too loose to predict observed cloud variance closely.
The paired clouds give the more informative diagnostic. Envelope construction
accounts for roughly 86% of measured runtime, cloud generation for about
12.5%; increasing intensity is therefore cheaper than scaling total runtime
by the same factor.

## Larger original-atlas control at lambda/z = 64

A separately frozen follow-up retained the original-width five-component
atlas, increased auxiliary intensity to `lambda = 64 z`, and used four new
independent populations of 4,096 unconditional draws each. The physical
target, ten regions and two-cloud averaging stayed fixed. All 16,384 draws
completed in 206.2 CPU seconds; [the complete report](../runs/basin-normalizer-baseline-4096-l64/assessment/report.md)
and [per-population diagnostics](../runs/basin-normalizer-baseline-4096-l64/assessment/analysis.json)
are archived separately from the first pilot.

| Region | log Q | Positive draws | Observed ESS | Relative SE | Largest weight share |
|---|---:|---:|---:|---:|---:|
| Native `q <= 1` | 15.2983 | 627 | 21.65 | 21.5% | 14.9% |
| Shoulder `1 < q < 2` | 15.3984 | 71 | 19.51 | 22.6% | 12.4% |
| Distant `q >= 5` | 10.5180 | 6,072 | 8.48 | 34.3% | 25.8% |
| All configurations | 16.0467 | 6,771 | 41.06 | 15.6% | 7.06% |

The observed native/shoulder/distant fractions are 47.31%, 52.29%, and 0.3971%.
The four shoulder log-normalizer estimates are 15.040, 15.499, 15.456, and
15.526. The near-native shoulder thus remains important in independent
populations at larger sample count and lower cloud noise. Its physical mass
is still estimated imprecisely. The intermediate region `2 <= q < 5` has only
one positive draw and remains especially poorly covered.

The estimated cloud share of observed variance falls to 27.9% for the total,
43.7% for native, 16.2% for shoulder, and 6.89% for distant. Nevertheless,
the top 1% of draws still carry 91.94% of total weight. The distant log Q
remains far below the old SMC other estimate, and rises from 8.509 in the
smaller original-width pilot to 10.518 here. These differences reinforce
the need for independent proposal coverage controls. They do not establish
a converged preference for native contact over the full competing domain.
The two campaigns changed both sample count and auxiliary intensity, so
their ESS/CPU difference is not an isolated measurement of intensity benefit.

## Short controls using broad SMC-derived guides

Three further frozen atlases augment the original five components with
Gaussian guides built from earlier SMC competing populations: two separate
four-population folds add four guides each, and the combined atlas adds eight.
Each received four new independent 512-draw populations at original covariance
width and `lambda/z = 64`, with two independent clouds per pose. The thirteen-
component combined atlas and the two nine-component fold atlases have distinct
model hashes; results are never pooled across them.

| Frozen guide atlas | log Q native | log Q shoulder | log Q distant | Distant ESS / 2048 | Shoulder positive draws |
|---|---:|---:|---:|---:|---:|
| [Fold A](../runs/basin-normalizer-smc-fold-a-512-l64/assessment/report.md) | 15.670 | 15.204 | 9.498 | 7.38 | 6 |
| [Fold B](../runs/basin-normalizer-smc-fold-b-512-l64/assessment/report.md) | 15.339 | 13.642 | 9.305 | 7.98 | 3 |
| [Both folds](../runs/basin-normalizer-smc-all-512-l64/assessment/report.md) | 15.510 | 11.064 | 9.544 | 16.71 | 1 |

All twelve jobs completed in approximately 41.8 CPU seconds. The new guide
branches were selected for 952, 905 and 911 of their respective 2,048 draws,
but supplied only 0.0385%, 0.0934% and 0.0822% of observed total importance
weight. These are auxiliary proposal-channel contributions, not estimates of
SMC physical basin probabilities. The largest single distant weight still
carries 34.5%, 24.3% and 14.0% of its estimated region mass.

These short controls do not recover the old distant normalizer or improve
the basis for an equilibrium conclusion. The broad guides allocate substantial
proposal probability while leaving very few shoulder draws, so their native
versus all-other ratios are particularly unstable. A Gaussian fitted across
multiple distinct contact environments may poorly represent each of them;
localized guides are a separate construction to test. A larger run of these
same broad guides is not supported by this screen alone. The original-width
baseline at 4,096 draws per population remains the larger comparison, but its
different sample budget prevents a direct attribution of differences to the
guide atlas.

## Reproduction and validation

The analyzer independently recomputes every importance weight from the recorded
cloud counts, full proposal density and fixed sample budget, verifies each
stored region assignment, and verifies exhaustive-region sums against the
Rust totals. Per-population and combined log normalizers use linear averages.
Basic analytic arithmetic controls checked invalid-zero retention, importance
ESS, the paired-cloud variance identity, and all-zero unresolved behavior.
These checks concern aggregation; physical reference-limit tests are supplied
with the Rust estimator and remain distinct from protein convergence.

From `/home/xvg/tetramer-mc`, use a fresh output directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/run_basin_normalizer_campaign.py \
  --out runs/basin-normalizer-pilot-512-reproduction --samples 512 --replicates 4 \
  --scales 0.5 1 2 4 --cloud-replicates 2 --workers 8
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_basin_normalizers.py \
  --root runs/basin-normalizer-pilot-512-reproduction
```

`--prepare-only` freezes inputs without launching; `--run-prepared DIRECTORY`
launches that preparation after checking hashes. Existing job directories
are never overwritten or silently restarted. `--lambda-ratio` changes only
the auxiliary Poisson intensity, leaving the physical radius/activity fixed.
For the larger baseline use `--scales 1 --samples 4096 --replicates 4
--lambda-ratio 64 --seed 492087331` and a fresh output directory. Each campaign
is analyzed separately; the analyzer checks model/config hashes and auxiliary
intensities before pooling its independent populations.
