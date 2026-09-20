# Frozen guides for the newly observed shoulder

The width-four global proposal found 66 valid poses in the unchanged
`1 < q < 2` shoulder. All contact both prescribed neighbors A and B. Their
full importance weights have ESS 2.546 and largest fraction 57.79%, so they
provide useful proposal locations but a poorly resolved physical integral.
The next calculation separates discovery from fresh integration.

`tools/prepare_shoulder_guides.py` freezes three proposals using every one
of those 66 poses:

- **Weighted:** maximum-likelihood mean and full covariance using the
  original full importance weights. There is no clipping, trimming or
  separate normalization within populations.
- **Geometry:** the same calculation with equal weight per observed pose.
  This fits the distribution of discovered geometry; it does not claim
  equal equilibrium weights.
- **Mixture:** a prespecified 50/50 mixture of these two Gaussians. The
  allocation remains fixed after the geometry probes and defines the
  primary guide. Each single Gaussian remains available as a control.

The source has four independent, equal-size populations of 16,384 draws;
their shoulder counts are 9, 21, 20 and 16. Preparation rescans all 65,536
raw draws, checks their hashes against the independent geometry audit,
and requires exact agreement of every selected pose and original weight.
It independently rechecks all 66 atomic hard contacts and original q.

## Coordinates and regularization

The chart anchor is the original native reference expressed relative to
the body frame of A, physical neighbor index 0. For relative translation
`t` and rotation `R`, use

```
x = (t − t_anchor, ell × Cayley(R R_anchorᵀ)),
ell = 55.02283113084892 Å.
```

The covariance retains all translation–rotation cross terms. The common
additive covariance floor is

```
diag(.05² I₃, [ell tan(.1° / 2)]² I₃).
```

This is a floor in deployment coordinates, with a nominal small angular
scale per axis. It is not an eigenvalue truncation or an assertion that
the fitted covariance has converged. The original weights give covariance
condition number 474.7; equal weights give 80.6. The largest observed
rotation relative to this chart is 6.576°, far from its Cayley seam.

The Gaussian density is with respect to translation times **normalized
SO(3) Haar**, including the factor
`ell³ π² (1 + |Cayley|²)²`. Independent SciPy chart evaluation agrees with
the production-format density to `1.26e-11` in log density; changing
between A body coordinates and laboratory coordinates agrees to
`5.7e-14`. The mixture evaluates the sum of both component densities.

## Frozen geometric probes

The preparation output is
`runs/ab-shoulder-guide-preparation-20260920`. It archives inputs, fitter
and helper sources, model/config/protocol hashes, all probe poses and
complete physical metadata. The model fixes neither the physical region
nor its normalization: A and B, capture radius 18 Å, depletant radius
1.5 Å and activity 0.035 Å⁻³ remain unchanged.

Each guide had exactly 256 fresh independent draws, without retry or
depletant clouds. The mixture draws its actual component labels with
probability 1/2, rather than forcing an equal count.

| Guide | Hard-valid poses | Valid original shoulder poses | Shoulder fraction |
| --- | ---: | ---: | ---: |
| Weighted | 110 | 32 | 12.50% ± 2.07% |
| Geometry | 134 | 64 | 25.00% ± 2.71% |
| Mixture | 128 | 45 | 17.58% ± 2.38% |

Every probe fell inside the capture sphere. Errors are observed binomial
standard errors for these geometric counts, not physical-weight errors.
The old calculation used approximately 0.0341 CPU seconds per valid pose,
which gives about 196 CPU seconds for 4 × 8,192 pure-mixture draws at the
observed target-valid fraction. This is only a conditional cost estimate:
the complete-cover allocation, proposal overhead and overlap-cloud volume
can change the cost.

## Required fresh comparison

Use the frozen guide inside a normalized proposal with a complete cover
of the original `q ≤ 2` set, retain `1 < q < 2` as the target mask, and
evaluate the full mixture density on every unconditional draw. Invalid
poses contribute zero; they remain in the sample denominator. The guide
alone neither resolves unseen modes nor proves a sampling improvement.
Fresh independent populations, guide controls, and a reference on the
same fixed finite region must establish whether the shoulder mass is
stable. These preparation results do not revise its free energy.

Six controls in `tools/test_shoulder_guides.py` check original-weight
retention, full covariance and coordinate changes, exact mixture density,
strict shoulder endpoints, preservation of hard-invalid draws, and an
active second physical neighbor independent of the proposal anchor. A
broad chart fails explicitly instead of silently dropping observations.
The additional analysis control checks the unconditional denominator when
a band contains zero-weight draws.

## Fresh fixed-budget integration

The protocol in `runs/ab-shoulder-window-pilot-20260920/protocol.json`
fixed four independent populations per method before their outcomes were
available. Each guided method used 65,536 total unconditional draws from
`0.25 cover + 0.75 (0.95 guide + 0.05 cube/Haar)`. The independent pure
cover used 524,288 draws. All runs retained the original strict shoulder
window, both neighbors, two Poisson clouds and auxiliary intensity/activity
ratio 64. None of the 66 fitting observations enters these estimates.

| Proposal | log Q(shoulder) | Importance ESS | Observed row RSE | Population RSE | CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted Gaussian | 24.6895 | 111.2 | 9.47% | 5.42% | 227.3 |
| Geometry Gaussian | 24.7860 | 38.4 | 16.13% | 25.96% | 527.0 |
| 50/50 mixture | 25.1992 | 47.9 | 14.44% | 16.00% | 371.5 |
| Pure complete cover | 15.5310 | 1.13 | 94.24% | 92.77% | 6.0 |

The pure-cover calculation establishes support but is too poorly sampled
to be a useful independent mass reference at this budget. One observation
carries 94.18% of its weight. Its low CPU cost and nonzero formal support
do not establish efficiency or convergence.

The broader guides identify where the weighted Gaussian lacks practical
coverage. Every band below retains the **original full draw count** as its
denominator. Adjacent endpoints use a half-open partition inside `1<q<2`.

| Original q band | Weighted log Q | Geometry log Q | Mixture log Q |
| --- | ---: | ---: | ---: |
| 1–1.1 | 24.4853 | 24.4535 | 24.8397 |
| 1.1–1.25 | 22.9783 | 23.2574 | 23.8527 |
| 1.25–1.5 | 19.1791 | 21.7831 | 21.9000 |
| 1.5–2 | 2.2255 | 20.6777 | 19.8895 |

In the 1.25–1.5 band, the two broader proposals agree approximately and
give 13–15 times the weighted-guide estimate. That weighted band has ESS
2.76; the geometry and mixture bands have ESS 24.3 and 14.8. The final
1.5–2 band remains imprecise even in the broader guides, with ESS 1.84
and 5.07. The weighted guide observes only three cover-generated poses
there. Better total importance ESS per CPU for the weighted guide therefore
does not establish better coverage of the physical region.

Within the mixture, the weighted and geometric source components supply
51.55% and 48.45% of its observed total importance sum. These are **source
diagnostics**, not equilibrium component occupancies. The geometric
component produces all three largest weights and accounts for most of
the variance: its component-contribution ESS is 12.4 versus 135.5 for
the weighted component. The mean paired-cloud variance fractions are
approximately 25–28% across guided campaigns; more pose coverage is still
needed even if the cloud noise were reduced.

`tools/analyze_shoulder_guides.py` independently rescans all 720,896 new
rows, verifies archived input and sample hashes, and requires identical
full physical configurations despite different archived shape paths. It
recomputes the unchanged scalar q on every valid pose (maximum error
`2.7e-15`), checks cloud averaging and full importance denominators, and
rejects reuse of any fitting seed or exact fitting pose. The result is
`runs/ab-shoulder-independent-comparison-20260920/comparison.json`.

The source covariance is a **frozen proposal memory**: fitting locations
from previous runs does not itself define a stationary physical ensemble.
Conditional on the frozen fits, fresh corrected importance draws target
the same physical integral. The limited source ESS, disagreement between
guide totals and the band-level coverage failure prevent a converged
shoulder free-energy claim. Future comparisons should preserve this
evidence and the broader guide support when refining the proposal; the
weighted guide's apparently smaller error must not select the success
criterion. These calculations do not validate continuously adapting an
uncorrected proposal or establish inter-tetramer assembly.

## Independent Cayley-cover reference

The conservative RMS/Cayley ellipsoid described in
[the separate cover derivation](rms-quaternion-cover.md) supplied a new
geometric reference with four independent populations of 262,144 draws.
It preserves the original open shoulder window and uses the existing
latent-ball sampler with its full Jacobian. The frozen calculation is
`runs/ab-shoulder-cayley-reference-4x262144-l64-20260920`.

The hard shoulder volume agrees across independent covers:

| Cover | Unconditional draws | Valid poses | Hard volume, Å³ | Observed RSE |
| --- | ---: | ---: | ---: | ---: |
| Product | 524,288 | 119 | 0.000140618 | 9.17% |
| Cayley ellipsoid | 1,048,576 | 3,224 | 0.000143271 | 1.76% |

The ellipsoid increases hard-volume ESS per draw by **13.55 times**.
With the depletion clouds still evaluated on valid poses, the improvement
in hard-volume ESS per CPU second is only 1.13 times: it spends 145.0 CPU
seconds, compared with 6.0 for the product reference. These are importance
integration metrics, not a measured speedup in physical contact mixing.

The physical integral still has a severe weight tail. The Cayley estimate
is log Q=21.7477, ESS=1.88 and observed RSE=72.84%. Its four population
log Q values are 18.5820, 21.1589, 20.8928 and 22.8388. The largest pose
contributes 71.84% of the total. Retrospective complementary halves give
log Q=20.1922 and 22.3293 even though their hard-volume logs agree at
−8.86767 and −8.83416. No observations are discarded from the full result.

The q bands distinguish volume from physical weight:

| Original q band | Cayley valid poses | Fraction of sampled hard volume | Fraction of sampled physical weight |
| --- | ---: | ---: | ---: |
| 1–1.1 | 32 | 0.99% | 75.13% |
| 1.1–1.25 | 88 | 2.73% | 4.03% |
| 1.25–1.5 | 335 | 10.40% | 11.80% |
| 1.5–2 | 2,769 | 85.88% | 9.04% |

Within the innermost band, one observation supplies 95.62% of the sampled
physical weight. Its q is 1.09666; its two independent cloud log weights
are 37.8302 and 38.6829. Consequently this reference has improved geometric
coverage and measures the hard volume precisely, but it has not resolved
the rare high-depletion contacts identified by the guides. Its lower
physical estimate is not evidence against those contacts. The weighted
guide has the complementary failure: it strongly favors deep contacts but
underestimates the innermost band's hard volume by roughly 150 times,
whereas the broader geometric guide and Cayley reference agree there
within their observed errors.

`tools/compare_shoulder_covers.py` independently rescans all 1,769,472
draws from these five campaigns, checks hashes, unchanged physical metadata,
original q and cloud averaging, and keeps every original denominator. Its
report in `runs/ab-shoulder-cayley-independent-comparison-20260920` contains
physical and hard bands, separate populations and complementary-half
diagnostics. Agreement of hard volumes supports the implementation control;
completeness itself rests on the separate geometric argument. Neither
agreement establishes convergence of exponentially weighted contact mass.

A separate retrospective check of the existing largest contributions
supports retaining the geometric guide while gathering more independent
data. The two largest Cayley observations have weighted-guide Mahalanobis
radii 5.36 and 9.67, but geometric-guide radii 2.05 and 2.24. The fourth
largest is at radii 6.16 and 1.31. These are six-dimensional proposal-chart
distances, not a new q metric or one-dimensional sigma count. The broader
guide already places these observed contacts within ordinary proposal
reach, so their discovery does not itself require a new fitted component.
`top-pose-guide-densities.json` in the comparison directory preserves every
pose and both frozen density evaluations; no new samples or fits enter it.

## Larger independent confirmations

The frozen mixture and geometry guides were each tested in a separate
four-population campaign of 65,536 draws per population. The same compiled
kernel, complete product-cover defense, original q window and physical
model were retained. These 262,144-draw confirmations do not incorporate
the pilot observations and were not used to refit either guide.

| Frozen guide | Pilot log Q / ESS | Fresh confirmation log Q / ESS | Confirmation row RSE | Population RSE | Largest fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50/50 mixture | 25.1992 / 47.9 | 25.0069 / 118.6 | 9.18% | 12.19% | 7.56% |
| Geometry | 24.7860 / 38.4 | 25.2362 / 13.9 | 26.84% | 27.12% | 25.65% |

The confirmations used 1,489 and 2,106 CPU seconds, respectively. Their
point estimates are compatible given their observed errors, but that
agreement does not establish coverage of rare weights. In particular,
quadrupling the geometry-guide sample count reduced its observed ESS:
new large contributions increased the measured second moment. Its
complementary halves give log Q=24.8917 and 25.4920, with ESS 44.4 and
6.03. The paired-cloud variance fraction is only 5.56%, so improving
cloud precision would not remove this observed pose-weight concentration.
The mixture confirmation's paired-cloud fraction is 14.83%.

| Original q band | Mixture confirmation log Q (ESS) | Geometry confirmation log Q (ESS) |
| --- | ---: | ---: |
| 1–1.1 | 24.6671 (72.8) | 24.9811 (8.42) |
| 1.1–1.25 | 23.5725 (41.6) | 23.4843 (49.0) |
| 1.25–1.5 | 21.9491 (21.4) | 22.2175 (28.8) |
| 1.5–2 | 19.1402 (18.0) | 19.3886 (8.06) |

The middle bands retain the additional mass missed by the weighted-only
pilot. The innermost band still dominates the total and its uncertainty.
The independent comparison artifact is
`runs/ab-shoulder-confirmation-with-cover-comparison-20260920/comparison.json`.
It keeps all seven campaigns separate, rescans 2,293,760 unconditional
rows, and records physical and hard bands, every population, complementary
halves, cloud-noise diagnostics, and archived hashes.

A further independent cover restricted to **the same original inner
band**, `1<q<1.1`, gives log Q=26.0943 with ESS 1.77, observed error 75.18%
and largest fraction 73.04%. Its point estimate exceeds the whole-shoulder
guided estimates. Exact integrals satisfy `Q(inner) <= Q(shoulder)`;
independent noisy estimates need not preserve that ordering. This reference
is too imprecise to certify or refute the guided estimates.

The source's 16 largest contributions were independently checked at the
atom-sphere level against both fixed neighbors. Every pose remains hard
valid and contacts both neighbors; the smallest surface gap across all
32 neighbor checks is 0.00959 Å. The largest contribution has q=1.06510,
gaps 0.01109 and 0.03753 Å, and cloud log weights 46.4971 and 46.1250.
It does not arise from an almost-zero numerical penetration allowance.

Crucially, that pose is already in a region of appreciable frozen-guide
density. Its weighted and geometric Mahalanobis radii are 3.31 and 2.00.
At this exact pose, the full weighted, geometric and mixture campaign
proposal densities exceed the direct inner-cover density by factors
2,866, 188 and 1,527. At the second-largest contribution the factors are
17,338, 325 and 8,832. Thus the unusually large direct-reference estimate
does **not** by itself establish a missing guided mode. The reference can
receive a large importance contribution at a pose where guide density is
already much higher. Pointwise density is not integrated basin probability
and does not establish adequate sampling of the surrounding high-weight
region. Statistical noise can explain the direct-versus-guided discrepancy.
Some smaller contributions lie far
outside the weighted fit but remain within the geometric fit, reinforcing
the reason to retain broader coverage.

The independently reconstructed guide densities agree between scalar
and vector implementations within `1.43e-14` in log density. The complete
pose, atom-gap, cloud and frozen-guide record is
`runs/ab-inner-shoulder-top-pose-audit-20260920/analysis.json`, generated by
`tools/analyze_inner_shoulder_top_poses.py` from the immutable same-band
comparison. No original weight is replaced or reinterpreted as a new mass
estimate. These diagnostics support the geometry and density checks while
leaving independent normalizer convergence unresolved. Beyond the
demonstrated middle-band limitation of the weighted-only guide, these
new extremes do not establish an additional guide-coverage failure. They
do not establish native assembly or justify selecting a new model from
the same extremes.
