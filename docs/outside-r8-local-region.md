# Frozen regional check of the outside-R8 contact excursion

Two new integration regions are prepared around the contact configurations visited by the correlated docking run `site0-m1-deep-r01-c09`. **No production normalizer has been launched.** The purpose is to measure physical mass outside the previously integrated competitor region, independently of the finite trajectory's dwell time.

Preparation is stored under `runs/outside-r8-local-region-20260920/site0`. The 32 chronological cohort observations contain30 distinct poses; their selection was determined by equally spaced indices among all retained frames outside the old R8 ellipsoid, without physical-weight selection. Their counts, duplication, and uneven time spacing are not equilibrium evidence. Every source-pose lookup and source-file hash is checked before fitting.

## Fixed construction

The coordinate chart keeps the **original deep Cayley anchor**, fixed-neighbor body frame, and angular length55.02283113084892 Å. In its six translation/Cayley coordinates, the new mean is the equally weighted cohort mean, and

\[
\Sigma_{\rm new}=\frac1{32}\sum_i(x_i-\bar x)(x_i-\bar x)^T
                 +0.25\Sigma_{\rm original}.
\]

The additive covariance is an explicit geometric regularizer. Neither the empirical covariance nor this floor is a thermodynamic fluctuation estimate. Repeated observations remain in the fit. No physical weight, trajectory occupancy correction, SMC normalizer, or native pose enters the new mean.

The new regions are \(\lVert L_{\rm new}^{-1}(x-\bar x)\rVert\le3\) and≤5, both restricted to capture/hard-valid poses with the original q≥5. Their covariance and center are identical; only radius differs. The old R3/R5/R8 files remain unchanged. Here “new R3/R5” always refers to these new ellipsoids; “old R8” refers to the earlier chart mean and covariance.

All physical fields remain unchanged: fixed site0 neighbor, 18 Å capture ball, all proper rotations, depletant radius1.5 Å, activity0.035 Å⁻³, atomic hard shape, and original registration metric. The configuration and region files preserve the exact metadata equality required by the existing Rust integrator.

## Preparation checks

The covariance is positive definite, with eigenvalues0.00380–0.25167 Å² and condition number66.19. Generalized eigenvalues relative to the original covariance are0.324–0.863, above the imposed0.25 floor. The new R3 contains30/32 selected observations; new R5 contains all32. This describes training coverage only.

Pose backmaps agree to3.6×10⁻¹⁵ Å and2.3×10⁻¹⁶ in rotation-matrix entries. Uniform-ball probe backmaps have maximum latent error5.9×10⁻¹⁴. The Gaussian-density/Jacobian identity agrees within2.1×10⁻¹³, and body/laboratory density invariance within4.7×10⁻¹⁴. An independent central finite-difference calculation of the six-dimensional pose derivative, using local rotation-vector measure divided by8π² for normalized Haar, agrees with the analytic log Jacobian within2.2×10⁻¹⁰ across32 probes.

The exact physical Jacobian used by the existing integrator is

\[
J(u)=\frac{|\det L_{\rm new}|}
 {\ell^3\pi^2(1+|c(u)|^2)^2},
\]

relative to translation volume times normalized SO(3) Haar measure. The fixed proper body-to-laboratory transform has unit Jacobian.

There were1024 independent **geometry-only** uniform-ball probes per new region. None evaluated depletion weights or contributed to a physical mass estimate.

| Probe classification | New R3 | New R5 |
|---|---:|---:|
| Capture-valid | 1024 | 1024 |
| Capture + hard + original q≥5 valid | 200 (19.53%) | 95 (9.28%) |
| Valid and inside old R8 | 30 (2.93% of all draws) | 26 (2.54%) |
| Valid and outside old R8 | 170 (16.60% of all draws) | 69 (6.74%) |
| Outside old R8 among valid probes | 85.0% | 72.63% |

These fractions measure proposal geometry, not equilibrium probabilities. Both new regions overlap the old R8 domain, so their complete masses cannot simply be added to the old R8 mass.

## Prespecified independent calculation

`validation-plan.json` requests four populations of16,384 unconditional draws for each new radius:131,072 fresh draws overall. The Poisson auxiliary intensity ratio is64 with two independent clouds per valid pose. Seeds are98621010+1009i for new R3 and98631010+1009i for new R5, i=0,…,3. Two workers per campaign give four workers total. No refitting or region adjustment is allowed during confirmation.

The existing archived `latent-region-normalizer` executable is sufficient; no Rust change is required. It draws uniformly from each new latent6-ball and estimates only that region. Afterward `analyze_local_region_overlap.py` evaluates the old chart at each sampled pose and directly accumulates

\[
\widehat Q_{\pm}=\frac1N\sum_{i=1}^{N}
  \mathbf1_{\mathrm{inside/outside\ old\ R8}}(X_i)
  V_6(R)J(u_i)H_{\rm capture}(X_i)H_{\rm hard}(X_i)
  \mathbf1_{q\ge5}(X_i)\,\overline W_i.
\]

Here E[W|X]=exp(zC(X)), V₆(R)=π³R⁶/6, and **N includes every draw**, including invalid poses and the opposite subgroup. The two nonnegative estimates sum to the new region's total. Outside mass is never obtained by subtracting two noisy, nearly equal estimates or by conditioning the denominator on accepted poses. Q0 is accumulated with the same geometry and unit bath weight; log(Qz/Q0) is a correlated same-pose ratio, reported as a point value without independent-error assumptions.

The analysis reports per-population agreement, observed standard errors, ESS, maximum weight, paired-cloud variance, and hard-region mass. New R3 and new R5 are kept separate. A zero-observation group is marked unresolved, not a physical zero or upper bound. Synthetic controls with weights[2,0,8,0] reproduce total2.5, inside0.5, and outside2.0 using N=4, and cover entirely zero or empty subgroups.

The previous original-R3 integration cost599 CPU seconds for65,536 draws. Provision roughly1200 summed CPU seconds (about5 minutes ideal wall time on four workers), acknowledging different hard-valid fractions and gate costs. The frozen plan calls for stopping and reporting if cost exceeds four times that reference budget. This is a budget estimate, not a performance guarantee.

Prepared artifacts:

- `region-r3.json`, SHA256 `109ed19f45c1fc573c4716a080c09c62df80be0153042a30ce9e964469f44bca`
- `region-r5.json`, SHA256 `ca34578a816ef7a821daf0aa337e1a2618cad54af19bdb724bf2241c30022ac1`
- `physical-config.json`, `report.json`, `validation-plan.json`, and `analysis-plan.json`
- Archived cohort, source trajectory/configuration/summary, old chart, atomic shape, fit code, analysis code, and previously validated Rust executable

Run `tools/prepare_outside_r8_local_region.py --out <fresh-directory>` to reproduce preparation. To run the planned confirmation, pass the frozen config, region, binary, and seeds to `tools/run_latent_region_campaign.py`; then run `tools/analyze_latent_region.py --root <campaign>` for the complete regional integral and `tools/analyze_local_region_overlap.py --root <campaign> --plan <validation-plan.json>` for the direct old-R8 partition.

Even agreement between the new radii would establish only the mass of these specified regions. It would not establish global coverage, equilibrium of the discovery trajectory, a native fraction, or an assembly mechanism.
