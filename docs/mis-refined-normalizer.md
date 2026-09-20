# Frozen guides from overlapping regional samples

`tools/prepare_mis_deep_normalizer_atlas.py` fits two additional proposal
Gaussians using the completed, independent uniform-latent-ball campaigns
with radii 3, 5, and 8. These campaigns used the same deep-contact chart,
physical configuration, and original-q≥5 restriction. They each contain
65,536 unconditional draws. The 196,608 observations are **training only**;
31,078 are hard/capture/region valid and have positive weights.

## The overlap correction

The source proposals overlap, so their original regional importance weights
cannot simply be pooled as if they sampled disjoint domains. With equal
allocation and ball volumes Vᵣ=π³r⁶/6, the complete latent proposal density is

\[
g_{\mathrm{mix}}(u)=\frac13\sum_{r\in\{3,5,8\}}
\frac{I(\lVert u\rVert\le r)}{V_r}.
\]

If J is the exact chart-to-physical Jacobian, the physical density is
g_mix/J. Each valid training observation receives the balance-heuristic MIS
weight

\[
a_i=\overline W_i J_i/g_{\mathrm{mix}}(u_i).
\]

Here W is the existing positive unbiased Poisson estimate of exp(zC), and
independent clouds are averaged on the linear scale. Invalid draws have
zero weight. Equivalently, the corrected log weight is the recorded regional
log weight minus log V_source minus log g_mix. Every source's full draw
budget, including invalid draws, is retained; the fitter requires equal
source budgets for the stated allocation.

An independent control integrates the known latent shell volumes using all
unconditional draws and weights 1/g_mix. All three shells and the full
radius-eight ball agree with analytic volumes within 1.56 observed standard
errors. The piecewise mixture density integrates analytically to one. These
controls check the overlapping-domain correction without using physical
overlap weights as reference truth.

## Frozen fit

All poses are encoded again from lab coordinates into the original deep
component's body-relative Cayley chart. The chart anchor is unchanged. The
reconstructed coordinates agree with the recorded latent affine maps within
3.6×10⁻¹⁵; an independent Gaussian-density/Jacobian identity agrees within
2.7×10⁻¹³. The maximum chart angle is 8.01°. Body-to-lab Gaussian density
conversion agrees within 3.6×10⁻¹⁵.

For powers τ=1 and 1/2, normalize aᵢ^τ and compute a weighted mean and full
covariance. The new covariance is

\[
\Sigma_{\mathrm{new}}=\Sigma_{\mathrm{weighted}}+
\tfrac14\Sigma_{\mathrm{original\ deep}}.
\]

There is no hidden covariance clipping. Cholesky positivity is checked.
Tempering noisy weights is a proposal-design choice, not a claim that the
result is the physical equilibrium distribution.

| Weight power | Training ESS | Largest normalized training weight | Covariance condition number | Covariance eigenvalues relative to original |
|---|---:|---:|---:|---:|
| 1 | 8.01 | 34.96% | 79.34 | 0.54–2.09 |
| 1/2 | 4,581.93 | 0.874% | 71.73 | 1.12–2.42 |

Independent geometry probes found 68/256 and 115/256 hard-valid candidates
from the new components. These are proposal diagnostics, not physical
weights. The sharp fit remains vulnerable to its small effective training
sample. The tempered fit has broad empirical coverage but has not been
validated by training likelihood alone. No observations outside the
radius-eight training domain are supplied by this fit.

The complete 114-component width-one parent is retained at total weight
0.5, without changing its means, anchors, or covariances. Each new guide has
weight 0.25, producing 116 normalized Gaussian components. The global
normalizer retains its 0.1 uniform cube/Haar branch. Thus the physical
target and full proposal support remain unchanged. New means follow the
weighted pose data rather than native coordinates, but the retained parent
contains native-informed guides; this is not a template-free assembly test.

Model, source hashes, selected training observations, raw/final covariance
matrices, fitting code, and validation plan are frozen under
`runs/smc-normalizer-mis-refined/site0`. Model SHA-256:
`5be4efd1710a542f52a982036780ccd05a07fb27ba04b403f36a650b9e740afd`.

## Independent validation

The authorized fresh campaign uses four populations of 16,384 unconditional
global importance draws, λ/z=64, two clouds, global covariance scale one,
and seeds 98571010+1009i. It reuses the frozen executable from the preceding
global controls. Training observations are never counted as new evidence.

Primary original-q bins remain unchanged. The additional
`tools/analyze_global_latent_shells.py` uses the same frozen chart and
boundaries 3/5/8 to classify far-contact poses. It also measures the remaining
far region outside radius eight and all non-far poses. These disjoint classes
sum to the global estimate. Every class uses the complete global proposal
density and every unconditional draw in its denominator; no acceptance or
shell-count conditioning is introduced. Qz/Q₀ is a correlated same-pose
ratio, reported as a point value.

The campaign is under `runs/basin-normalizer-mis-refined-16384-l64`. Its main
report assesses the physical original-q partition; `deep-shells.json`
assesses the unchanged geometric regions. Observed ESS and agreement among
replicates cannot exclude unobserved high-weight modes.

## Fresh validation result

The complete 65,536-draw global campaign took 604.4 sampler CPU seconds.
All 116 components and the uniform branch entered every proposal density.
Independent SciPy reconstruction agrees with 128 fresh complete proposal
densities within 3.4×10⁻¹⁴ in log density.

| Fixed region | Fresh log Qz | Importance ESS | Observed relative SE | Largest contribution |
|---|---:|---:|---:|---:|
| Deep r≤3 | 21.274 | 775.02 | 3.57% | 1.01% |
| Deep 3<r≤5 | 20.810 | 138.53 | 8.49% | 4.15% |
| Deep 5<r≤8 | 19.122 | 15.97 | 25.02% | 11.63% |
| All deep r≤8 | 21.830 | 610.14 | 4.03% | 1.49% |
| Far outside r=8 | 15.386 | 2.28 | 66.27% | 65.09% |

The four independent r≤3 estimates are 21.223, 21.306, 21.350, and 21.209.
Its combined value agrees with the separate uniform-ball estimate 21.195
(ESS 314.7 and observed relative SE 5.62%). At similar production CPU cost,
the new full-support guide has about 2.4 times the observed regional ESS.
This compares **normalization precision**, excludes prior training/discovery
cost, and does not measure Markov-chain mixing or assembly speed.

For the middle shell, the four independent log Q values are 20.936, 20.744,
20.668, and 20.868. Precision is much better than the previous uniform-R5
control, whose middle-shell ESS was 2.49 and largest contribution 62.9%.
The previously dominant training pose was not reused in this fresh estimate.
Paired-cloud noise contributes 28.2% and 31.6% of the observed core and
middle-shell estimator variance, respectively.

The unchanged primary-q estimates are log Q(native)=14.982, shoulder=16.983,
and intermediate=14.638. The entire observed far region has log Q=21.832
and ESS 611.7; the global estimate is 21.842 with ESS 623.3. These global
point estimates remain provisional because the region outside the training
neighborhood is weakly sampled. Low observed mass there is not an upper
bound on unobserved competitors.

Hard-region mass is also proposal dependent in precision. The core log Q₀
is −15.262 with ESS 5,332, agreeing with direct integration −15.259, and its
regional log(Qz/Q₀) is 36.535. Middle-shell log Q₀ is −13.238 with ESS 284.
In the outer shell the hard-region ESS is only 1.45 and one contribution
accounts for 82.9% of Q₀; its depletion-enhancement ratio is correspondingly
unresolved despite being present as a point value in the JSON report.

The global shell classifier was also checked against an earlier Gaussian
campaign: it reproduces the previous frozen r≤3 estimate exactly and its
disjoint classes sum to the original global normalizer. No further
production runs were launched for this bounded proposal control.
