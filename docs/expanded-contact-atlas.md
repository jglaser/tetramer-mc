# Complete intermediate-window test with two measured contact neighborhoods

The [previous local reference](atlas-peak-neighborhood.md) resolved finite
weight around the maximum discovered by the broad atlas. This next comparison
adds that contact to a frozen proposal, then measures the **complete original
2 ≤ q < 5 window** with fresh populations. The two local neighborhoods remain
fixed analysis masks; all space outside them is included.

Both AB neighbors, depletant radius 1.5 Å, activity 0.035 Å⁻³, capture radius
18 Å, and normalized SO(3) Haar measure remain unchanged. Q is expressed in
the same Å³ translational-volume convention as the references. These are
conditional pose integrals: assembling the fixed neighbors is a separate cost.

## Frozen proposal

The 36 old geometric centers retain their order. The new broad maximum is
added explicitly at index 36, followed by deterministic farthest-first
selection until all 104 observed candidate poses lie within 0.5 Å member
RMS of a center. This gives **49 geometric centers**. The candidates comprise
the original 64 poses, 16 original atlas extremes and 24 new local-reference
extremes. Source poses, weights, seeds and hashes are checked. The maximum
nearest-center distance is 0.493729254 Å. This covers the observed set;
it makes no claim about unobserved physical basins.

One Gaussian is fitted to all 7,723 positive-weight poses in the new radius-0.5
reference, using their original importance weights. The source has ESS 120.26
and largest weight share 5.08%. The same additive covariance floor as the old
fit is used: 0.05 Å translation and 0.1° angular-axis scale. Leaving out each
population shifts the mean by 0.0088–0.0195 Å-equivalent and gives covariance
generalized eigenvalues 0.890–1.135 relative to the pooled fit. Held-out
conditional log-density gains are 2.83–3.32 nats over geometric width 0.2
and 5.59–5.96 over width 0.4. The moments describe the truncated reference
neighborhood, not an entire basin. Original population masses are retained;
populations are not separately renormalized before pooling the fit.
The additive floor contributes about 2.6%–46.0% of the final covariance
along its generalized directions, so the stability is partly regularized.

Each atlas contains 53 Gaussian components:

- 60% split equally among the 49 geometric centers;
- 16% old fitted Gaussian plus 4% with its covariance multiplied by four;
- 16% new fitted Gaussian plus 4% with its covariance multiplied by four.

The narrow and broad variants differ only in the geometric standard
deviation, 0.2 versus 0.4 Å-equivalent. Both local fits, centers and weights
remain identical. The complete proposal remains

\[
g=0.25g_{\rm complete\ product}
 +0.75\left(0.95g_{\rm atlas}+0.05g_{\rm cube/Haar}\right).
\]

Every contribution divides by this entire physical density, including all
components and Haar Jacobians. Rejected hard, q and capture draws stay zero
in the unconditional denominator. This mixture retains complete target
support, but useful statistical coverage must still be measured.

## Fixed allocation and validation

Each width uses eight independent populations of 32,768 draws: **524,288
new unconditional draws** across both widths. Seeds are 102101010+1009i
and 102201010+1009i for i=0,…,7. Two independent positive Poisson factors at
λ/z=64 are averaged linearly per valid pose. No refitting, population
replacement or adaptive stopping occurs during production.

Before physical sampling, 256 cloud-free probes per arm gave 75 narrow and
66 broad valid original-window poses. Independent scalar/vectorized proposal
densities agreed within 2.14 × 10⁻¹⁴. The reviewed physical executable is
unchanged, SHA-256
`3a6a2dbba0ec5234c36cf66d7c40877336f22358a6a1ea91ea8366b344ee027b`.

Analysis reconstructs every original full proposal density, q and cloud
factor. It measures the old and new geometric balls, as well as the disjoint
partition old radius-two ball / new radius-two ball / outside both, retaining
same-row covariance. Reference comparisons use exactly matched physical
regions; those historical references contributed to proposal construction
and are labeled calibration data.

The protocol also fixes first-8,192, first-16,384 and full-32,768 prefixes
inside each population. These are correlated sensitivity checks. Under
the fixed IID law with finite variance,

\[
\operatorname{Cov}(\bar W_m,\bar W_N)=\operatorname{Var}(\bar W_N),\qquad
\operatorname{Var}(\bar W_m-\bar W_N)
 =\left(\frac{N}{m}-1\right)\operatorname{Var}(\bar W_N).
\]

The observed full-stream variance estimates this difference variance; the
prefixes are not treated as independent replications. Row and population
errors, maximum contribution, width sensitivity and complete remaining-space
coverage all inform the interpretation. Small observed errors cannot bound
unseen tails. None of these independent-draw calculations measures MCMC
mixing or assembly rates.

## Physical results

All sixteen physical populations completed successfully. The independent
audit reconstructs all 524,288 original rows and all predeclared masks and
prefixes. All sixteen selected highest-weight poses pass independent atom
checks; the smallest checked gap is 0.005934 Å.

| Complete original 2 ≤ q < 5 | log Q | Row / population relative SE | Weight ESS | Largest draw | Sampler CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Narrow | 16.781336 | 6.33% / 6.47% | 249.61 | 2.95% | 2,092.32 |
| Broad | 16.893890 | 13.47% / 11.92% | 55.06 | 8.00% | 1,646.17 |

The broad mean is 11.91% higher, a difference of 0.73 combined observed
row standard errors on Q. Thus the widths are compatible at this precision.
The narrow proposal has about **4.5 times lower observed variance × sampler
CPU** than the broad one, using absolute variance of Q. This comparison
excludes discovery, fitting and auditing. It compares these two fixed
importance proposals; it is not a measured MCMC mixing speedup. Weight
concentration is much lower than in the earlier atlas, but neither small
observed errors nor cross-width agreement proves all important poses sampled.

The direct disjoint partition is:

| Physical region | Narrow log Q / relative SE | Broad log Q / relative SE | Estimated fraction, narrow / broad |
| --- | ---: | ---: | ---: |
| Old radius-two ball | 13.532404 / 6.25% | 13.679631 / 22.96% | 3.88% / 4.02% |
| New radius-two ball | 16.716615 / 6.72% | 16.822175 / 14.41% | 93.73% / 93.08% |
| Outside both balls | 13.045576 / 22.42% | 13.354087 / 30.84% | 2.39% / 2.90% |

Every outside-both contribution is measured directly. That observed small
share is not an upper bound on unexplored weight. All three pieces include
the original hard, capture and q masks, and their same-row covariance
reconstructs the full estimator. The log hard-volume estimates are −2.68599
and −2.30497 with 18.89% and 15.44% row errors; the existing complete
geometric reference remains substantially more precise for hard volume.

The tightly measured local controls also reproduce the earlier references:

| Matched ball | Narrow log Q / relative SE | Broad log Q / relative SE | Historical reference log Q / relative SE |
| --- | ---: | ---: | ---: |
| Old R=0.5 | 12.075699 / 1.81% | 12.100915 / 1.90% | 12.054798 / 4.16% |
| New R=0.5 | 14.776872 / 2.71% | 14.721858 / 2.36% | 14.721964 / 9.11% |
| New R=1 | 16.117394 / 4.73% | 16.104648 / 10.09% | 16.091928 / 18.17% |

The radius-one reference uses the earlier disjoint multiscale sum. Its
radius-two sum, log Q=16.416008 with 19.52% error, lies below both new
atlas estimates by roughly 1.6–1.7 combined observed row errors. This
is not a decisive discrepancy, but the wider neighborhood still merits
an independent higher-budget check. The small-ball reference does not
validate all of its outer shells.

## Sample-count sensitivity remains

| Draws per population | Narrow full log Q / relative SE | Broad full log Q / relative SE |
| --- | ---: | ---: |
| 8,192 | 16.557542 / 7.51% | 16.674604 / 20.42% |
| 16,384 | 16.626893 / 6.00% | 16.715571 / 18.15% |
| 32,768 | 16.781336 / 6.33% | 16.893890 / 13.47% |

The narrow half-run mean is 14.31% below its full-run mean, with an
observed difference standard error of 6.33% after accounting for shared
rows. That is a 2.26-SE shift. The broad half-run difference is −16.33%
with 13.47% difference SE. These are correlated diagnostics, not independent
confirmations or optional-stopping tests. No numeric pass/fail thresholds
were frozen for these statistics.

The largest narrow and broad contributions lie 1.368 and 1.772 Å-equivalent
from the new center, with original q=2.116 and 2.146, respectively. The
larger samples are finding weight beyond the fitted radius-0.5 core.
An unchanged-proposal, larger independent population comparison is the
next useful convergence control. Increasing the number of fitted centers
again would confound that sample-count test.
The subsequent [larger independent repeat](contact-atlas-repeat.md) preserves
these exact model bytes and uses four times the total draws per width.

![Expanded atlas, prefixes and disjoint physical regions](../runs/ab-intermediate-expanded-atlas-figure-20260920/expanded-contact-atlas.png)

The full intermediate estimate is substantially better supported than the
earlier single-draw-dominated result. The upward prefix sensitivity means
convergence is not established. Native and shoulder regions, original q≥5,
neighbor-assembly costs and reversible physical mixing remain separate
requirements before an assembly verdict.

## Reproduction

- `tools/extend_contact_centers.py`: preserve old centers and cover new observed poses.
- `tools/fit_peak_reference_proposal.py`: audit the fixed reference and fit its conditional covariance.
- `tools/prepare_expanded_contact_atlas.py`: freeze both models, analysis masks, seeds and budgets.
- `tools/run_native_region_reference.py`: execute each declared full-window campaign.
- `tools/analyze_expanded_contact_atlas.py`: independent full-row and correlated-prefix audit.

Artifacts are under `runs/ab-intermediate-expanded-contact-centers-20260920`,
`runs/ab-intermediate-atlas-peak-local-fit-20260920`,
`runs/ab-intermediate-expanded-atlas-preparation-20260920`,
`runs/ab-intermediate-expanded-atlas-{narrow,broad}-8x32768-l64-20260920`
and `runs/ab-intermediate-expanded-atlas-audit-20260920`.
Each preparation, campaign and analysis uses a fresh output directory and
archives its inputs and executable or scripts. No historical raw estimator
is replaced by fitted weights or by a selected-pose correction.
The frozen preparation protocol SHA-256 is
`074c30a2ca41657b90e57eb287a6b0954158fdb1482c7a916688e676794b339e`.
The figure, SVG and archived inputs are in
`runs/ab-intermediate-expanded-atlas-figure-20260920`.

Fourteen targeted fitting, selection, masked-moment and correlated-prefix
tests pass. Physical kernels are unchanged; full-row numerical audits and
independent selected-pose atom checks validate this campaign's outputs.

The [SMC reconciliation](smc-reference-reconciliation.md) separates the
earlier one-neighbor comparison from this AB problem. Its matched native
estimates agree; the missing one-neighbor shoulder and far-contact weight
still require a controlled SMC comparison. The current expanded atlas does
not by itself resolve that different environment.
The subsequent [compact SMC control](smc-shoulder-control.md) recovers the
A-only shoulder, with about 30% population uncertainty in native plus
shoulder weight. That result leaves the far region separate.
