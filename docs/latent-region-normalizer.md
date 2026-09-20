# Direct physical integration of a frozen pose region

`latent-region-normalizer` measures one immutable region of pose space. It
does **not** estimate the complete far-contact region, native fraction, or
global physical normalizer. Its input is the previously frozen chart
ellipsoid used by the Gaussian-guide controls.

For chart mean μ, Cholesky factor L, angular scale ℓ, and radius R, draw

\[
u=R U^{1/6}\frac{n}{\lVert n\rVert},\qquad
n\sim N(0,I_6),\quad U\sim\mathrm{Uniform}(0,1).
\]

This is uniform in the six-ball. Set x=μ+Lu, use its first three coordinates
as the anchor-relative translation, and c=x₍rot₎/ℓ for the Cayley rotation.
The proper fixed-neighbor frame transform has unit Jacobian. Relative to
center volume in Å³ and **normalized** SO(3) Haar measure,

\[
J(u)=\frac{|\det L|}{\ell^3\pi^2(1+|c|^2)^2},\qquad
V_6(R)=\frac{\pi^3R^6}{6}.
\]

The estimator is

\[
\widehat Q_R=\frac{V_6(R)}{N}\sum_{i=1}^N
H_{\rm capture}(X_i)H_{\rm hard}(X_i)
I(q(X_i)\ge q_{\min}) J(u_i)\overline W_i.
\]

Here independent positive Poisson weights have conditional expectation
E[W|X]=exp(zC(X)). The certified-inner-volume plus uncertain-envelope
implementation in `overlap_weight` is reused unchanged, with two clouds per
accepted pose in this control. Cloud weights are averaged on the linear
scale. Every attempted latent draw counts in N; hard, capture, and q
rejections contribute zero. No Gaussian density enters the integration
weight. Physical energy variations within the region can still cause large
variance.

The region file pins its mean, covariance, anchor, radius, original-q
threshold, physical registration metric, fixed neighbor, hard shape, capture
domain, and bath. Changing the physical activity requires a different region
definition; the CLI only changes auxiliary Poisson intensity. Each pose is
encoded back into latent coordinates and checked against its original draw.
Numerical chart failures abort; they are not silently censored or redrawn.

The runtime records each latent draw, radius, lab pose, backmapped coordinate,
physical Jacobian, original q, validity indicators, and independent cloud
weights. Configuration, region, shape, executable hash, and exact compiled
source bundle are archived. The campaign launcher also freezes its executable
and analysis implementation before starting independent populations.

## Validation

`tests/latent_region_normalizer.rs` uses a sphere whose fixed neighbor is
remote, so at zero activity the answer is a pure physical pose volume. With
block-diagonal covariance (translation standard deviation a and rotation
coordinate standard deviation b), its independent reference is

\[
Q=\frac{16a^3b^3}{3\ell^3}
\int_0^R
\frac{\rho^2(R^2-\rho^2)^{3/2}}
{(1+(b\rho/\ell)^2)^2}\,d\rho.
\]

The clipped reference replaces the translation-section radius by the capture
radius and applies the analytic angular cutoff. Rotations reach approximately
148°, making the Haar factor consequential. Two sets of 32,768 unconditional
draws give 8.5469±0.0708 against quadrature 8.5396 and 1.3654±0.0281 against
1.3384. These quoted uncertainties are observed standard errors. The tests
also check radial second/fourth moments, a marginal coordinate variance,
cross moment, inner-ball probability, lab/body/latent inversion, original q,
rejected-zero accounting, exact output aggregation, and CLI/API identity.

The production analyzer independently reconstructs J using SciPy rotations
and the identity log J=log φ₆(u)−log g(X), where g is the normalized Gaussian
chart density. This is an audit identity, not the direct integration weight.

```bash
cargo test --locked --release --test latent_region_normalizer
/home/xvg/protein-nucleation/.venv/bin/python tools/run_latent_region_campaign.py \
  --out runs/latent-region-uniform-ball-16384-l64 \
  --samples 16384 --replicates 4 --workers 4 \
  --seed 98531010 --lambda-ratio 64 --cloud-replicates 2
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_latent_region.py \
  --root runs/latent-region-uniform-ball-16384-l64
```

Use a fresh output directory. The frozen region hash for this control is
`01ee36aeff8f55260ae23fb67f8744c11fb9182d5d7606a7ad1be9a269d08789`:
width-one Mahalanobis radius≤3 and original q≥5. It is exactly the region
used by the previous Gaussian-guide width comparison.

## First independent regional result

The four fresh populations completed 65,536 unconditional draws, of which
19,754 contributed nonzero weight. Their combined regional log Q is
**21.19496**, observed importance ESS 314.7, largest contribution 4.11%, and
observed relative standard error 5.62%. The four independent log Q values
are 21.35215, 21.24147, 21.05953, and 21.09937; population-level relative
standard error is 6.81%. The paired-cloud diagnostic attributes 53.3% of
observed estimator variance to Poisson noise. Total sampler CPU is 599.1 s.

The previous Gaussian-guide width-one and width-two estimates for exactly
this region were 21.38161 and 20.81573, with ESS 77.4 and 34.7 from 32,768
draws each. The width-four control had ESS 2.1. Direct integration improves
precision for this regional check; it does not establish global far-contact
coverage or validate the exploratory SMC log Z=27.273. Independent SciPy
reconstruction agrees with 128 production pose Jacobians within 6.8×10⁻¹⁴;
maximum latent backmap error is 3.6×10⁻¹⁴.

The same draws give **log Q₀=−15.25897** for the hard-valid region under
uniform physical pose measure, hence **log(Qz/Q₀)=36.45393**. The small
accessible pose volume and large depletion enhancement are distinct effects.
This is a correlated ratio from the same poses; it is reported as a point
value without independent-error bars. It is not obtained by exponentiating
the mean overlap volume.

![Independent estimates of the same fixed region](../runs/latent-region-uniform-ball-16384-l64/assessment/method-comparison.png)

The plot uses observed standard errors, not rigorous confidence bounds.
Efficiency is regional importance ESS per CPU second, not a trajectory
mixing speedup. Its data and standalone PNG/SVG/PDF are archived in the
campaign's `assessment` directory.

## Prespecified surrounding-shell controls

Two new region files enlarge only the frozen latent radius to five and eight;
the radius-three file is unchanged. Their chart, physical metric, hard
geometry, capture domain, and bath are identical. Before launching, the
plan in `runs/latent-region-shell-plan-20260920/plan.json` fixed the boundaries
3/5/8, independent seeds, and file hashes. Each campaign uses four new
populations of 16,384 uniform-ball draws, two Poisson clouds, and λ/z=64.
Two workers per campaign keep four sampling workers in total.

`tools/analyze_latent_region_shells.py` computes physical and hard-region
mass for r≤3, 3<r≤5, and 5<r≤8 with every unconditional draw retained in each
supported estimator's denominator. Each ratio Qz/Q₀ uses the same poses.
The radius-five campaign cannot estimate the outer shell; that entry is
explicitly outside its domain, not a zero estimate. An unobserved supported
shell is unresolved/null. Shells sum to their campaign's regional total;
campaigns with different radii are never pooled.

The controls probe the previously observed high-weight pose at radius 4.23
and its neighborhood. They do not estimate mass outside the largest frozen
region or establish global convergence.

Both larger-radius controls completed, with 65,536 unconditional draws each:

| Draw domain | Whole-domain log Qz | ESS | log Q₀ | log(Qz/Q₀) | CPU seconds |
|---|---:|---:|---:|---:|---:|
| r≤3 | 21.195 | 314.71 | −15.259 | 36.454 | 599.1 |
| r≤5 | 22.060 | 6.38 | −13.121 | 35.180 | 234.0 |
| r≤8 | 21.126 | 13.34 | −11.106 | 32.232 | 103.2 |

The separately estimated shell contributions are:

| Draw domain | r≤3 log Qz (ESS) | 3<r≤5 log Qz (ESS) | 5<r≤8 log Qz (ESS) |
|---|---:|---:|---:|
| r≤3 | 21.195 (314.71) | Outside domain | Outside domain |
| r≤5 | 21.101 (36.87) | 21.576 (2.49) | Outside domain |
| r≤8 | 20.277 (5.02) | 20.398 (6.11) | 18.713 (19.17) |

The radius-five campaign reproduces the inner region reasonably, but its
middle-shell estimate is 62.9% due to one pose. This is population `r01`,
draw 8934, radius 3.1267, original q=28.0764. Its paired cloud log weights
43.2877 and 43.1016 are close. Only 1.33% of the middle-shell estimator's
observed variance is assigned to Poisson noise. Therefore a narrow high
physical-weight part of pose space remains consequential even under direct
uniform integration; this problem is no longer attributable solely to a
Gaussian proposal's tail factor.

The radius-eight campaign allocates only 47 valid observations to the inner
region, versus 19,754 in the direct radius-three campaign. Its smaller total
point estimate does not establish that adding volume reduces mass: exact
regional masses are monotone under inclusion, while these independent
finite-sample estimates are noisy. Neither a mass plateau nor complete
contact coverage has been established. The hard-valid counts are 7,824 and
3,500 for radii five and eight, so their lower runtime comes with substantial
loss of resolution near the core.

The production executable is identical across all three radii. Independent
Jacobian checks agree within 2.2×10⁻¹³, shell contributions sum to the
unchanged regional total, the shell plan precedes both launches, and the
original radius-three region hash remains unchanged. An actual 16-draw
zero-weight Rust run also verified that total and supported-shell analysis
return unresolved/null without exception, while unsupported shells remain
explicitly outside the domain. That regression artifact is under
`runs/latent-region-zero-analysis-validation-20260920` and is separate from
the production estimates.

Full reports, shell statistics including Q₀, and the eight largest
contributors per shell are in each campaign's `assessment` directory under
`runs/latent-region-radius{5,8}-16384-l64`. No additional production runs were
performed or combined with these estimates.
