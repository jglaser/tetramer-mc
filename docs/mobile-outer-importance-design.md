# A controlled extension for the unresolved outer contact shells

**Implemented and reference-tested; protein performance not yet measured.**
The optional `--importance-guide` path now exists in the Rust normalizer and
Python campaign runner. Its independent auditor checks the full latent
density, pose/Jacobian reconstruction, immutable guide binding and retained
zeros. [Ten targeted Rust checks](../runs/mobile-outer-importance-validation-20260921/validation.json)
pass, including an analytic sphere depletion/Haar integral, unequal Gaussian
weights, the uniform limit and an all-zero target. Fourteen focused Python
auditor tests and two launcher integration tests pass.

The [end-to-end sphere check](../runs/latent-importance-cross-language-20260921/report.md)
also passes: the actual Rust executable generated two fresh 256-draw
populations, followed by one archived Python audit. All 512 rows were
reconstructed, including 250 outside-shell zeros. Maximum density, latent
coordinate and log-Jacobian differences are 3.56e-15. This checks the
launcher/serialization/auditor boundary; it is not a protein efficiency test.

The two outer shells need better pose coverage. Increasing the number of
Poisson clouds alone cannot resolve their dominant uncertainty. This note
uses the already audited 65,536 unconditional rows in
`runs/mobile-competing-outer-weights-20260921`; it launches no physical
calculations and does not rerun an audit. The saved diagnostic and two frozen
guide specifications are in `runs/mobile-competing-outer-diagnostic-20260921`.

## What dominates the saved integrals

| Shell | ESS | Top four rows' share | Pose part of observed variance | Cloud part |
|---|---:|---:|---:|---:|
| 5 < rho <= 8 | 6.69 | 60.2% | 58.5% | 41.5% |
| 8 < rho <= 12 | 3.01 | 74.4% | 92.9% | 7.1% |

These are low-ESS descriptive statistics, not converged estimates of the
underlying variances. The decomposition uses the two independent Poisson
replicates at each saved pose, including zero weights for invalid draws. It
separates the variance of the conditional mean from conditional cloud noise;
it is not a comparison of only accepted or high-weight poses.

The two largest 5–8 contributions both occur in population r03. They supply
27.0% and 26.7% of its pooled four-population integral. Their translations
from the original competing pose are 0.54 and 0.97 Å, rotations 1.33 and
1.68 degrees, and maximum tetramer-member displacements 0.97 and 1.26 Å.
The largest row's two bath log factors are 40.45 and 38.09, so there is a
substantial positive cloud fluctuation in this row. The second has factors
40.17 and 39.33. Both are small changes to the competing registration,
with original native metric q about 58.56.

The largest 8–12 row is population r01, draw 4149. It supplies 56.2% of the
measured shell integral. Its center translates 1.07 Å and rotates 2.73
degrees; the maximum member displacement is 1.87 Å. Both independent bath
log factors are large, 37.71 and 37.21. The larger cloud supplies only 62.2%
of their average, which supports the diagnosis of rare favorable poses
rather than a single enormous Poisson fluctuation. It is still a competing
pose: q = 58.53. Its radius, 11.74, is close to the outer radius 12, which
is an additional reason not to interpret that radius as a basin boundary.

For the four largest contributions in each shell, angular coordinates
supply 75–95% and 54–92% of squared latent radius. A proposal must preserve
translation/rotation combinations rather than sample only a radial scalar.
The measured covariance eigenvalues span factors of about 108 and 335,
but ESS 3–7 is too low to treat those covariance matrices as reliable basin
fits. The full rows, translations, rotations, member displacement and
individual cloud factors are retained in the diagnostic artifact.

At fixed pose count, completely eliminating Poisson noise would reduce
observed relative SE only from 38.7% to 29.6% in 5–8, and from 57.6% to
55.5% in 8–12. Four clouds instead of two give approximately 34.4% and
56.6%, before accounting for their added cost. Increasing cloud intensity
can be a later control, but is not the first remedy for the outermost
shell.

## Minimal frozen importance proposal

Keep exactly the same frozen chart, shell, hard shape, two-neighbor
scaffold, wall-safe capture sphere, native-q restriction and physical bath.
Let u be the original six-dimensional whitened chart coordinate. Use

\[
q_\alpha(u)=\alpha\,\frac{\mathbf1_S(u)}{|S|}
 +(1-\alpha)\sum_k a_k\,\mathcal N_6(u;m_k,\Sigma_k),
\qquad \alpha=\tfrac12.
\]

The Gaussian branch is **untruncated**. Draw its latent point once. If it
falls outside the original shell, contributes an invalid hard pose, or
fails the original q restriction, record a zero. Do not condition on
success, redraw it, or infer a Gaussian truncation constant. The complete
mixture density is evaluated at every proposed point, independently of
which branch generated it. The uniform branch gives positive density
everywhere in the target shell.

With exact pose Jacobian J and fresh independent Poisson clouds,

\[
\widehat Q_z=\frac1N\sum_i
 \frac{\mathbf1_S H_{\rm capture}H_{\rm hard}I_{q>1}
        J(u_i)\,\overline W_i}{q_\alpha(u_i)},
\qquad
\widehat Q_0=\frac1N\sum_i
 \frac{\mathbf1_S H_{\rm capture}H_{\rm hard}I_{q>1}
        J(u_i)}{q_\alpha(u_i)}.
\]

The existing Jacobian remains
`det(L)/(ell^3*pi^2*(1+|Cayley|^2)^2)`, for center volume times normalized
SO(3) Haar measure. The conditional cloud identity is unchanged. For a
frozen guide these estimators target the same integrals as the uniform
shell calculation. No learned mixture normalizer enters the physical
weight.

Two guide JSON files have been frozen for a first implementation control.
Each uses four centers from each of the four completed populations,
ranked by the **smaller** of its two bath log factors and greedily separated
by one latent unit within that population. Every selected center gets
equal mass, with isotropic latent standard deviations 0.5 and 1.5 as two
equally weighted scales: 32 components in total. This deliberately avoids
estimating a single narrow covariance from three effective observations.
It also prevents the one winning population from taking all guide mass.
The smaller-cloud score is only a heuristic for choosing proposal centers;
it is never substituted for the unbiased physical estimator.

The guide is a normalized proposal design, not a measured thermodynamic
density. No claim is made that its centers cover every important part of
the shell. The saved rows used to build it must not be recycled into its
new importance estimate. The original design JSON retains `implemented: false`
as a historical record. The [new immutable preparation](../runs/mobile-competing-importance-preparation-20260921/plan.json)
converts those specifications to `defensive-latent-shell-guide-v1` by stripping
descriptive metadata only. All 32 component weights, means and covariances,
the defensive mass 0.5, and the region hashes are unchanged. Two converter
tests check exact parameter retention and rejection of malformed designs.

## Implementation and controls

`src/latent_region.rs` retains `run(options)` for uniform latent ball/shell
draws and adds `run_with_importance(options, guide_path)`. The CLI and
`tools/run_latent_region_campaign.py` accept `--importance-guide`. A guide
selects the alternative draw step and saves `log_proposal_density`, branch
and component labels, and `shell_valid`. It replaces `log_volume + log_jacobian` by
`log_jacobian - log_proposal_density` in both physical and hard weights.
The legacy no-guide RNG and records are unchanged. Gaussian samples outside
the shell have zero contribution. The backmap check scales with the drawn
radius rather than asserting that every draw lies below the target radius.
Failure of the chart inverse remains an error, not an invalid sample to censor.

The records bind guide bytes and their hash, component count, mixture mass,
branch selection, latent point, proposal log density and unchanged physical
inputs. The independent auditor reconstructs the complete mixture density,
latent coordinates and exact Jacobian, and verifies each shell flag. The
unconditional sample count remains fixed.

The reference tests check normalized Gaussian-mixture evaluation,
uniform-branch recovery, shell boundary flags and known ball/shell integrals.
They include a Gaussian branch that frequently leaves the shell, an
independently reconstructed density and the zero-activity hard-volume limit.
For a protein control, use separate new seeds, four independent populations of 8192 draws
per shell, two clouds, and lambda/z = 64, with the guides frozen before
launch. Compare to the existing uniform-shell estimates with their large
uncertainty explicitly retained. A later independent defensive-mass or
bandwidth control is needed before treating apparent gains as robust.

The new mobile reciprocal trajectory also supplies an unregistered return at
original chart radius 18.84 (2.31 Å and 4.10° from the initial relative pose).
Its other neighbor is mobile, so it is not a fixed-scaffold weight estimate.
It nevertheless reinforces that R12 does not exhaust the competing geometry.
The first guide comparison must keep its original regions fixed; any extension
beyond R12 needs a separately declared domain and independent weights.

## Benefit and cost bounds

For any fixed guide with uniform fraction alpha, the second moment of
the unbiased random-weight estimator obeys

\[
 E_{q_\alpha}[Y^2]\leq\alpha^{-1}E_{\rm uniform}[Y_U^2],
\]

because `q_alpha >= alpha/|S|` throughout the fixed target shell, including
the unchanged conditional Poisson noise. At alpha = 1/2, the second
moment is at most twice that of uniform sampling. This is a support and
second-moment guarantee, **not** a guarantee of variance reduction or
CPU speedup. It does not bound missing mass outside the shell.

Observed-moment extrapolation suggests uniform sampling would require
about 122,400 and 271,800 total draws to reach 20% relative SE in the two
shells, or 489,600 and 1,087,100 for 10%. The corresponding combined CPU
estimates are about 2137 and 8546 seconds, versus 371 seconds for the
completed 65,536-row campaign. These estimates can fail if another rare
high-weight pose is found; they are planning figures, not confidence or
convergence bounds.

The proposed pilot retains the current total number of attempted poses.
Evaluating 32 small Gaussian components should be measured separately
from hard and cloud work. A guide can also raise the fraction of valid
poses, so equal attempted counts do not imply equal CPU. With alpha =
1/2, the valid fraction cannot exceed `0.5*p_uniform + 0.5`; using the
observed uniform valid fractions as a planning approximation gives at
most about 2.59 and 2.89 times as many cloud evaluations. Cloud cost per
valid pose can change as well. Poisson counts are unbounded, so this is
not a hard runtime bound. Report uncertainty per CPU, maximum weight
share, independent-population dispersion and retained defensive coverage
before deciding whether the guide is useful.
