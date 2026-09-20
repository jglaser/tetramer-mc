# Independent regional check of the outside-R8 contact excursion

The independent calculations are complete. The correlated docking run spends **54.39% of its retained frames in a fixed region whose population bound, evaluated at independently estimated masses, is only 0.516%**. Enlarging the region covers 58.77% of the trajectory and gives an estimated population bound of 0.819%. Independent integrations of the common region agree. This identifies a specific long-residence sampling problem; the occupation is not explained by a comparably large equilibrium mass. Unseen high importance weights and the selection of the region from this trajectory remain explicit limitations.

Two regions were frozen around contact configurations visited by `site0-m1-deep-r01-c09`, before generating the fresh integration draws. Their physical masses are measured independently of the discovery trajectory's dwell time. No model, region, or sample budget was changed during that production. A subsequent, separately authorized [matched escape control](outside-r8-atlas-extension.md) tests the proposal mismatch diagnosed below.

Preparation is stored under `runs/outside-r8-local-region-20260920/site0`. The 32 chronological cohort observations contain 30 distinct poses; their selection was determined by equally spaced indices among all retained frames outside the old R8 ellipsoid, without physical-weight selection. Their counts, duplication, and uneven time spacing are not equilibrium evidence. Every source-pose lookup and source-file hash is checked before fitting.

## Fixed construction

The coordinate chart keeps the **original deep Cayley anchor**, fixed-neighbor body frame, and angular length 55.02283113084892 Å. In its six translation/Cayley coordinates, the new mean is the equally weighted cohort mean, and

\[
\Sigma_{\rm new}=\frac 1{32}\sum_i(x_i-\bar x)(x_i-\bar x)^T
                 +0.25\Sigma_{\rm original}.
\]

The additive covariance is an explicit geometric regularizer. Neither the empirical covariance nor this floor is a thermodynamic fluctuation estimate. Repeated observations remain in the fit. No physical weight, trajectory occupancy correction, SMC normalizer, or native pose enters the new mean.

The new regions are \(\lVert L_{\rm new}^{-1}(x-\bar x)\rVert\le 3\) and≤5, both restricted to capture/hard-valid poses with the original q≥5. Their covariance and center are identical; only radius differs. The old R3/R5/R8 files remain unchanged. Here “new R3/R5” always refers to these new ellipsoids; “old R8” refers to the earlier chart mean and covariance.

All physical fields remain unchanged: fixed site 0 neighbor, 18 Å capture ball, all proper rotations, depletant radius 1.5 Å, activity 0.035 Å⁻³, atomic hard shape, and original registration metric. The configuration and region files preserve the exact metadata equality required by the existing Rust integrator.

## Preparation checks

The covariance is positive definite, with eigenvalues 0.00380–0.25167 Å² and condition number 66.19. Generalized eigenvalues relative to the original covariance are 0.324–0.863, above the imposed 0.25 floor. The new R3 contains 30/32 selected observations; new R5 contains all 32. This describes training coverage only.

Pose backmaps agree to 3.6×10⁻¹⁵ Å and 2.3×10⁻¹⁶ in rotation-matrix entries. Uniform-ball probe backmaps have maximum latent error 5.9×10⁻¹⁴. The Gaussian-density/Jacobian identity agrees within 2.1×10⁻¹³, and body/laboratory density invariance within 4.7×10⁻¹⁴. An independent central finite-difference calculation of the six-dimensional pose derivative, using local rotation-vector measure divided by 8π² for normalized Haar, agrees with the analytic log Jacobian within 2.2×10⁻¹⁰ across 32 probes.

The exact physical Jacobian used by the existing integrator is

\[
J(u)=\frac{|\det L_{\rm new}|}
 {\ell^3\pi^2(1+|c(u)|^2)^2},
\]

relative to translation volume times normalized SO(3) Haar measure. The fixed proper body-to-laboratory transform has unit Jacobian.

There were 1024 independent **geometry-only** uniform-ball probes per new region. None evaluated depletion weights or contributed to a physical mass estimate.

| Probe classification | New R3 | New R5 |
|---|---:|---:|
| Capture-valid | 1024 | 1024 |
| Capture + hard + original q≥5 valid | 200 (19.53%) | 95 (9.28%) |
| Valid and inside old R8 | 30 (2.93% of all draws) | 26 (2.54%) |
| Valid and outside old R8 | 170 (16.60% of all draws) | 69 (6.74%) |
| Outside old R8 among valid probes | 85.0% | 72.63% |

These fractions measure proposal geometry, not equilibrium probabilities. Both new regions overlap the old R8 domain, so their complete masses cannot simply be added to the old R8 mass.

## Prespecified independent calculation

`validation-plan.json` requests four populations of 16,384 unconditional draws for each new radius:131,072 fresh draws overall. The Poisson auxiliary intensity ratio is 64 with two independent clouds per valid pose. Seeds are 98621010+1009i for new R3 and 98631010+1009i for new R5, i=0,…,3. Two workers per campaign give four workers total. No refitting or region adjustment is allowed during confirmation.

The existing archived `latent-region-normalizer` executable is sufficient; no Rust change is required. It draws uniformly from each new latent 6-ball and estimates only that region. Afterward `analyze_local_region_overlap.py` evaluates the old chart at each sampled pose and directly accumulates

\[
\widehat Q_{\pm}=\frac 1N\sum_{i=1}^{N}
  \mathbf 1_{\mathrm{inside/outside\ old\ R8}}(X_i)
  V_6(R)J(u_i)H_{\rm capture}(X_i)H_{\rm hard}(X_i)
  \mathbf 1_{q\ge 5}(X_i)\,\overline W_i.
\]

Here E[W|X]=exp(zC(X)), V₆(R)=π³R⁶/6, and **N includes every draw**, including invalid poses and the opposite subgroup. The two nonnegative estimates sum to the new region's total. Outside mass is never obtained by subtracting two noisy, nearly equal estimates or by conditioning the denominator on accepted poses. Q0 is accumulated with the same geometry and unit bath weight; log(Qz/Q0) is a correlated same-pose ratio, reported as a point value without independent-error assumptions.

The analysis reports per-population agreement, observed standard errors, ESS, maximum weight, paired-cloud variance, and hard-region mass. New R3 and new R5 are kept separate. A zero-observation group is marked unresolved, not a physical zero or upper bound. Synthetic controls with weights[2,0,8,0] reproduce total 2.5, inside 0.5, and outside 2.0 using N=4, and cover entirely zero or empty subgroups.

The previous original-R3 integration cost 599.103 CPU seconds for 65,536 draws. The estimate was roughly 1200 summed CPU seconds for both new campaigns. `run_outside_r8_plan.py` revalidated the authoritative worktree and all frozen input/binary hashes, refused existing output paths, and supervised two workers per campaign. Its combined CPU guard was 2396.410 seconds, exactly four times the original single-R3 cost. A failure or guard violation would terminate both process groups and retain all partial/failed outputs without treating them as fixed-N estimates. All eight jobs completed normally in 506.164 summed CPU seconds: 347.172 for new R3 and 158.992 for new R5. The historical preparation plan remains immutable; completed status is recorded separately in `runs/outside-r8-production-20260920/summary.json`.

## Fresh regional weights

Every row in this table has 65,536 unconditional draws in its denominator. No invalid pose or opposite-subgroup draw is discarded. Q is measured relative to translation volume times normalized orientation Haar measure.

| New domain and subgroup | Nonzero | log Qz | ESS | Observed relative SE | Largest weight fraction | Paired-cloud variance fraction |
|---|---:|---:|---:|---:|---:|---:|
| R3, total | 12,009 | 16.87268 | 1090.99 | 3.00% | 0.734% | 23.55% |
| R3, inside old R8 | 1,541 | 15.53083 | 238.00 | 6.47% | 2.810% | 19.85% |
| R3, outside old R8 | 10,468 | 16.56974 | 866.58 | 3.37% | 0.884% | 25.03% |
| R5, total | 5,544 | 18.18517 | 19.92 | 22.40% | 21.12% | 39.63% |
| R5, inside old R8 | 1,431 | 17.80521 | 9.82 | 31.91% | 30.88% | 41.28% |
| R5, outside old R8 | 4,113 | 17.03350 | 38.87 | 16.04% | 14.05% | 9.05% |

The four independent outside-old-R8 population log Q values are:

- New R3: 16.66415, 16.58854, 16.47047, 16.54596.
- New R5: 16.85288, 16.71204, 17.12604, 17.32855.

The larger region's total uncertainty is concentrated in its overlap with old R8. Its outside mass is also less precise than new R3, with a 14% maximum contribution; rare unsampled weights remain a material limitation. Paired-cloud variance accounts for 25% of the new-R3 outside estimator's observed variance and 9% for new R5, so pose-weight variation is the larger uncertainty source.

| Region | log Q0 | log(Qz/Q0) |
|---|---:|---:|
| New R3, total | −17.82694 | 34.69962 |
| New R3 outside old R8 | −17.96465 | 34.53439 |
| New R5, total | −15.53310 | 33.71827 |
| New R5 outside old R8 | −15.83269 | 32.86619 |

These correlated ratios are point values, not independent-error estimates or exponentials of a mean overlap volume. Positive Poisson weight arithmetic was independently checked for every generated cloud, with zero observed log-weight discrepancy. Independent pose/Jacobian reconstruction errors were below 1.9×10⁻¹³.

There is also a direct common-domain control. Restricting **all new-R5 draws** to the already frozen new R3, while retaining the full 65,536 denominator, gives outside-old-R8 log Q 16.58952, compared with 16.56974 from direct new-R3 sampling. Their difference is 0.080 observed combined standard errors. The complete common-R3 mass agrees within 0.232 observed combined standard errors. These checks compare the same physical integrals and do not pool the two campaigns. They support the observed estimates but do not certify tail coverage.

## Residence versus a disjoint-region bound

Let A be the old R8 physical region and B be a new ellipsoid intersected with the outside of A. They are disjoint, so for the true physical normalizers

\[
\pi(B)=\frac{Q_B}{Q_{\rm total}}\le\frac{Q_B}{Q_B+Q_A}.
\]

No global normalizer or enumeration of all other contacts is required. Additional physical regions can only enlarge the denominator. The existing independent old-R8 estimate is log Q_A=21.83045 with 4.03% observed relative SE. The inequality itself is exact; evaluating it with estimated masses gives an estimated diagnostic, **not a certified upper confidence bound**.

| B | Estimated bound | Max-B-population/min-A-population sensitivity | Observed ±2SE sensitivity | Deep start 1, c=0.9 residence |
|---|---:|---:|---:|---:|
| New R3 outside old R8 | 0.5165% | 0.6007% | 0.5992% | 2720/5001 = 54.389% |
| New R5 outside old R8 | 0.8187% | 1.1608% | 1.1719% | 2939/5001 = 58.768% |

The sensitivity columns are descriptive calculations, not calibrated confidence limits. Unknown large weights in B could increase its mass estimate. The R3 calculation is the stronger diagnostic because its observed ESS is 866.6 and its largest weight contributes less than 0.9%.

Every retained trajectory frame, including cycle 0, is classified in the original and new charts. New B3/B5 contain 92.52%/99.97% of the 2940 outside-old-R8 frames in the prolonged excursion. The other six trajectories have zero occupancy in both B regions. Native start 1 at c=0.9 is the only other visitor, with 49/5001=0.980% in B3 and 177/5001=3.539% in B5. Per-run first/second-half fractions, complete dwell intervals, and hashes are retained in `assessment/mass-and-residence.json`.

![Residence compared with independent regional masses](../runs/outside-r8-production-20260920/assessment/residence-versus-regional-mass.png)

This comparison quantifies how unrepresentative the long-residence record is of the estimated target populations. It does not imply a violation of balance, a physical kinetic lifetime, or a calibrated rejection of stationarity: the region was selected from the trajectory, and a stationary but strongly correlated finite record can also have a rare, long excursion. The current evidence supports investigating escape from this contact neighborhood rather than interpreting its large empirical occupancy as a large equilibrium basin weight.

## What the recorded escape attempts show

The attempt logs identify a concrete proposal-density mismatch. While the current pose is in B5, the correlated learned branch proposes **1,013 returns to old R8**. Of these, **261 pass the geometric checks**, but only **two are accepted**. The median fresh Poisson log factor for the valid returns is **+0.606**, whereas the median proposal correction, log G(old)−log G(new), is **−10.844**. These candidates do reach the previously measured region; their large negative proposal corrections suppress acceptance.

Here G is the full 116-component Gaussian proposal density relative to translation volume and normalized Haar measure, without the separate uniform branch. Its correction is required for balance. A favorable target-region mass does not by itself guarantee useful acceptance when the current contact lies in a poorly represented part of G.

| Attempts starting in B5 | Proposed | Hard-valid | Accepted | Median gate log factor, valid endpoints | Median proposal correction, valid endpoints |
|---|---:|---:|---:|---:|---:|
| Learned, any destination | 2628 | 557 | 36 | −0.727 | −4.933 |
| Learned, outside B5 | 1705 | 381 | 2 | −0.667 | −8.603 |
| Learned, old R8 | 1013 | 261 | 2 | +0.606 | −10.844 |
| Local, outside B5 | 4417 | 629 | 3 | −25.947 | 0 |
| Uniform, outside B5 | 313 | 12 | 0 | −35.981 | 0 |

Geometric rejection also remains substantial. Local and uniform escapes have unfavorable recorded depletion factors. The learned old-R8 candidates provide the cleanest separation: their typical gate factor is favorable, but the full-density correction is strongly unfavorable. Medians in the table summarize separate quantities and must not be added to reconstruct a median acceptance probability.

Source selection in B5 uses only component 114 (552 attempts) and component 115 (2076), the untempered and tempered MIS guides. Destination selection still spans the frozen mixture. The original deep guide, component 113, receives 631 proposals; all lie in old R8,186 are hard-valid, and none is accepted. For those 186 valid endpoints the median gate log factor is+1.031 and the median proposal correction is−11.856. Their recorded acceptance probabilities sum to only 0.0305. All 36 accepted learned moves target 114 or 115;32 retain the same component. The two learned returns occur at cycles 1734 and 2602, both within the same component.

The final uninterrupted B5 residence starts at cycle 2770 and continues through 5000. During this interval the learned branch proposes 721 old-R8 returns,146 hard-valid, with zero accepted. Their median gate factor is+0.0303, median proposal correction−11.795, and recorded acceptance probabilities sum to 0.352. The particle nevertheless moves within the region: this interval contains 66 accepted local moves and 18 learned moves. Across all B5 visits there are 136 distinct runs of unchanged retained poses; the longest lasts 181 frames. Long basin residence therefore differs from remaining at one fixed pose.

![Recorded return proposals and their acceptance factors](../runs/outside-r8-production-20260920/assessment/escape-proposal-factors.png)

There is **no persistent auxiliary Poisson weight in this kernel**. Every geometrically valid attempt draws a fresh conditional pair cloud, computes log(1+z/λ)(gained−lost), uses the result once, and discards it. The retained state contains the pose and contact observable, not a cloud or estimated absolute likelihood. Classical sticking caused by retaining an overestimated pseudo-marginal likelihood is therefore not the implementation here. One cloud per candidate cannot determine how much additional rejection comes from stochastic fluctuations rather than the underlying depletion change; that would require an independent controlled comparison.

The matched c=0 trajectory never enters B5, so it supplies no direct escape-rate control there. The present finding concerns these recorded candidates and supports testing improved proposal coverage; it is not an equilibrium mixing-time estimate or proof of a unique trapping mechanism. No acceptance correction should be removed.

That targeted test is now [complete](outside-r8-atlas-extension.md). Adding the unchanged frozen geometric Gaussian at 1% proposal weight reduces the first old-R8 return from 69–891 cycles to 1–16 in four matched short comparisons. Both models retain the required full-density correction. The test uses new source selection and fresh chains, not the old candidate log with an altered denominator.

`tools/analyze_trapped_docking_moves.py` reconstructs all retained states in both 15000-attempt logs, checks every gate/proposal acceptance arithmetic, classifies all old/proposed endpoints, and independently evaluates the full Gaussian density on 256 deterministic endpoint pairs per run. Maximum independent log-density discrepancy is below 7.3×10⁻¹⁴. Counts include hard-invalid proposed destinations. The full branch/component tables and all boundary crossings are in `assessment/attempt-audit.json`.

## Prescribed second-neighbor exclusion

A separate deterministic geometric check excludes the **entire new R3 and R5 latent balls** in the presence of the prescribed second native neighbor. This set includes all their capture/hard/q-restricted physical subsets. New R3 needs one collision witness for its root cell, with penetration lower bound 0.28156 Å. New R5 needs 61 visited cells and 31 collision leaves, with minimum certified penetration 0.008624 Å. Both have zero unresolved cells. The check and numerical audits completed in 0.40 seconds.

The certificate uses analytic displacement bounds, conservative floating-point slack, and independently recomputed atom-pair witnesses. It is not a formally rounded interval-arithmetic proof. It applies only to this fixed second-neighbor arrangement; it does not include the free-energy cost of forming that neighbor or exclude other competitor regions. Artifacts are in `runs/outside-r8-local-region-neighbor2-certificate`.

## Artifacts

Prepared artifacts:

- `region-r3.json`, SHA256 `109ed19f45c1fc573c4716a080c09c62df80be0153042a30ce9e964469f44bca`
- `region-r5.json`, SHA256 `ca34578a816ef7a821daf0aa337e1a2618cad54af19bdb724bf2241c30022ac1`
- `physical-config.json`, `report.json`, `validation-plan.json`, and `analysis-plan.json`
- Archived cohort, source trajectory/configuration/summary, old chart, atomic shape, fit code, analysis code, and previously validated Rust executable

Run `tools/prepare_outside_r8_local_region.py --out <fresh-directory>` to reproduce preparation. Production used `tools/run_outside_r8_plan.py` with the frozen validation plan; its archived supervisor, commands, hash validation, guard, and completion records are in `runs/outside-r8-production-20260920`. Existing production paths are never overwritten. Reanalyze with `tools/analyze_latent_region.py --root <campaign>` for the complete regional integral and `tools/analyze_local_region_overlap.py --root <campaign> --plan <validation-plan.json>` for the direct old-R8 partition. `tools/analyze_outside_r8_residence.py` reproduces the common-region control, exact trajectory classifications, mass-bound sensitivity calculations, and figure.

These results establish only estimates for the specified regions. They do not establish global coverage, equilibrium of the discovery trajectory, a native fraction, or an assembly mechanism. The normalizer budget is complete. The subsequent short docking control is documented separately; no longer extension or additional normalizer run has been started.
