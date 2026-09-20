# Independent neighborhood measurement of the atlas maximum

The broad [contact atlas](intermediate-contact-atlas.md) found a pose that
supplied 93.53% of its whole-window estimate. Fresh clouds confirmed a large
pointwise weight, but that check could not measure the surrounding pose volume.
This calculation freezes a neighborhood around that pose and measures its
integral with new independent samples.

The physical target remains both fixed AB neighbors, depletant radius 1.5 Å,
activity 0.035 Å⁻³, capture radius 18 Å and normalized SO(3) Haar measure.
Every neighborhood is intersected with the original **2 ≤ q < 5** window
and both hard exclusions. These conditional integrals omit the cost of
assembling the fixed neighbors.

## Frozen center and geometry

The selected historical pose is broad-atlas population r01, draw 14669,
seed 101502019. Its q is 2.391404325. Preparation checks the original source
pose, importance weight, population manifest, seed and archived physical
model. The independent analysis rechecks all 65,536 source rows and confirms
that this is the maximum over the selected arm, not merely its shortlist.
Selection uses historical data; all neighborhood measurements below use
fresh random streams after fixing the center and geometry.

The chart is the same geometric construction as the
[previous neighborhood reference](peak-neighborhood-reference.md). For
centered member positions, write A = tr(M)I − M, rotated into the chart frame.
With translation δt and Cayley vector c, the radius is

\[
\rho^2=|\delta t|^2+4c^T A c,
\qquad
d_{\rm member\,RMS}^2=|\delta t|^2+
\frac{4c^T A c}{1+|c|^2}\le\rho^2.
\]

Thus each chart ball is contained in a rigid-member RMS ball; it is not a
complete RMS cover or a definition of a metastable basin. The physical
Jacobian from the whitened six-dimensional coordinates is

\[
J=\frac{1}{8\pi^2\sqrt{\det A}(1+|c|^2)^2}.
\]

The old and new centers have member RMS distance **5.097889670 Å**.
Their radius-two chart balls are therefore disjoint, with at least
1.097889670 Å between their enclosing RMS balls. This permits a partition
into the old ball, the new ball and the region outside both. It does not
justify dropping that complement.

For these four member centers, the largest generalized eigenvalue of
(|rᵢ|²I − rᵢrᵢᵀ, A) is two. Hence an individual member moves by at most
√3ρ. The original registration metric obeys |q−q₀| ≤ √3ρ/2 here,
including its weaker angular term. Even the radius-0.5 bound reaches below
q=2, so the original lower mask is essential. Radius two cannot reach q=5
or the capture boundary; those checks remain in the sampler as well.
These are reconstructed numerical geometry bounds, not interval arithmetic.

## Sampling and audit

Before probes or physical draws, the protocol fixes radii 0.5, 1 and 2 Å,
four independent populations of 16,384 draws at each radius, λ/z=64 and
two independent Poisson clouds per valid pose. Population seed bases are
101701010, 101801010 and 101901010, with increments of 1009.
There are **196,608 unconditional physical draws** in total. The executable
is unchanged, SHA-256
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.

Cloud-free probes accepted 60, 30 and 24 of 512 poses at the three radii.
Pose reconstruction, member RMS identities and the Haar Jacobian agreed
within 1.5 × 10⁻¹⁴. Every physical contribution is the latent-ball volume
times J times the arithmetic mean of the two positive cloud factors.
Hard, q and capture failures remain zero in the full original denominator.

Nested full-ball estimates are compared separately. To form one cumulative
local estimate, the already-declared disjoint bands [0,0.5], (0.5,1] and
(1,2] each use their smallest covering reference campaign. These campaigns
have independent seeds, so their mean variances add. Same-source masks
retain covariance. No weight is clipped or substituted.

## Fresh results

All twelve populations completed successfully. The all-row audit reproduces
the density, original q, masks, Poisson factors and geometric identities;
maximum independent RMS/radius/Jacobian reconstruction error is 6.7 × 10⁻¹⁵.
All 24 selected extreme poses pass independent atom-union checks, with
smallest checked gap 0.000371 Å.

| Direct ball radius | Valid / 65,536 | log Q | Row / population relative SE | Largest draw | Sampler CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 7,723 | 14.721964 | 9.11% / 13.17% | 5.08% | 301.41 |
| 1 | 4,317 | 16.021414 | 20.44% / 15.88% | 15.79% | 159.51 |
| 2 | 3,287 | 16.830854 | 53.21% / 52.93% | 46.74% | 104.31 |

The radius-0.5 populations give log Q values 14.52928, 14.40203, 14.94790
and 14.90059. Its weight ESS is 120.26. This resolves finite local mass
around the selected contact beyond the earlier pointwise cloud test.
The finite-sample errors still do not bound unseen tails.

Broader uniform balls undersample the narrow inner region: the radius-one
source has 135 valid draws inside radius 0.5, and the radius-two source has
only two. Their estimates of that same inner mass have 33.31% and 83.20%
row uncertainty. The problem is therefore not just whether poses are hard-valid;
precision also depends on finding the small regions carrying large weight.

| Independent band used in local sum | log Q | Row relative SE | Fraction of radius-two sum |
| --- | ---: | ---: | ---: |
| 0 ≤ ρ ≤ 0.5, from R=0.5 | 14.721964 | 9.11% | 18.38% |
| 0.5 < ρ ≤ 1, from R=1 | 15.798743 | 24.16% | 53.94% |
| 1 < ρ ≤ 2, from R=2 | 15.131573 | 52.17% | 27.68% |

The cumulative disjoint estimate is **log Q = 16.091928** for radius one
(18.17% row, 10.38% population-group uncertainty), and **16.416008** for
radius two (19.52% row, 19.05% population-group uncertainty). The latter
uses 565.23 sampler CPU seconds and its largest single contribution is 13.70%.
It is a local estimate with better observed precision than the direct large
ball, not evidence of improved trajectory mixing. The noisy outer shell
still needs better coverage.

## What the historical atlas was measuring

The old atlas rows can be masked around the newly frozen center without
changing their original proposal densities or unconditional denominators:

| New-center radius | Historical narrow log Q / row SE | Historical broad log Q / row SE |
| --- | ---: | ---: |
| 0.5 | No hits | 18.192008 / 100% |
| 1 | 12.8057 / 54.3% | 18.2055 / 98.7% |
| 2 | 14.7443 / 31.7% | 18.2426 / 95.1% |

The broad radius-0.5 estimate contains exactly the selected maximum.
Its contribution is **32.14 times** the fresh local mean. This ratio is
a retrospective diagnostic after selecting the historical maximum, not an
independent significance test or a reason to replace one weight in the
old full-window estimate. The narrow arm never hit this smallest region.
Together the checks identify a high-weight contact with poor historical
proposal coverage; they do not validate either historical full-window number.

The new radius-two local estimate is about 17.25 times the earlier peak's
independent multiscale radius-two estimate, log Q=13.568313. Both quantities
refer to finite disjoint chart neighborhoods, with roughly 20% observed
row errors each, rather than complete physical basins.

Historical direct estimates outside **both** radius-two balls are log Q
12.7672 (34.0% error) and 13.2414 (23.2%) for the narrow and broad arms.
Their old-ball, new-ball and outside-both partitions reproduce the original
totals and same-row variances within 2.3 × 10⁻¹⁴. These data helped select
the new center, so the complement estimates remain retrospective. No
historical complement is added to the new references as a converged global
normalizer. A fresh complete-support calculation is still required.

![Fresh neighborhood references and historical atlas masks](../runs/ab-intermediate-atlas-peak-figure-20260920/atlas-peak-neighborhood.png)

## Consequence for the next calculation

The high pointwise weight survives as substantial finite local mass, but
the original dominant draw overstated the smallest neighborhood's measured
weight. A proposal around this contact should cover the 0.5–1 shell as well
as the center. The useful next comparison is a frozen expanded atlas against
these local references, retaining explicit coverage of the region outside
both measured neighborhoods and separate independent populations.

The complete intermediate integral, other registration regions, earlier
SMC discrepancies, reversible contact mixing and assembly verdict remain
open. These conditional AB results alone cannot distinguish successful
template-free assembly from a failure of the physical model.

## Reproduction

`tools/prepare_peak_neighborhood.py` now accepts `--source-audit`,
`--source-arm` and `--probe-seed`, while retaining its earlier default
center. Preparation and all derived reports require fresh directories.
`tools/run_latent_region_campaign.py` launches each frozen region with
its prescribed seed and budget. `tools/analyze_peak_neighborhood.py`
checks all original rows, and `tools/combine_peak_neighborhood.py` forms
the disjoint local sum. `tools/diagnose_atlas_peak_regions.py` applies the
new masks retrospectively to the unchanged historical atlas rows.

- Preparation: `runs/ab-intermediate-atlas-peak-geometry-20260920`
- Physical campaigns: `runs/ab-intermediate-atlas-peak-reference-20260920/r{0p5,1,2}`
- Independent audit: `runs/ab-intermediate-atlas-peak-reference-audit-20260920`
- Disjoint local sum: `runs/ab-intermediate-atlas-peak-multiscale-20260920`
- Historical masks: `runs/ab-intermediate-atlas-peak-historical-masks-20260920`
- Geometry separation: `runs/ab-intermediate-atlas-peak-separation-20260920`
- Selected atom checks: `runs/ab-intermediate-atlas-peak-reference-extremes-20260920`
- Figure and archived inputs: `runs/ab-intermediate-atlas-peak-figure-20260920`

The preparation protocol SHA-256 is
`1d76321550eb8bb6c4d823e07400dfce3fdb59ef6a6c3c687af2afeb4e719696`.
Its archived preparer predates an additional fail-fast source-manifest/seed
check in the working tool; frozen geometry, proposal laws and budgets are
unchanged. The physical analysis separately verifies source selection.

Validation: 21 targeted tests pass across preparation geometry, full-N
neighborhood masks, source-selection provenance, historical three-way
partitions and geometric separation. All physical launchers and derived
audits exit successfully; archived input and figure hashes were rechecked.
