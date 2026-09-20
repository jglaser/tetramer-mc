# Contact-centered proposals for the complete intermediate region

The new atlas combines contacts found by the complete reference calculation
with the measured local covariance around its dominant new pose. Fresh
independent populations test the full **2 ≤ q < 5** integral. The objective
is to recover narrow contact weight without losing the rest of this domain.
The physical target remains both fixed AB neighbors, depletant radius
1.5 Å, activity 0.035 Å⁻³, capture radius 18 Å and normalized Haar measure.
These conditional weights exclude the cost of assembling the fixed neighbors.

## Observed contact centers

The candidate set consists of the 40 atom-checked extremes from the
[complete intermediate allocation](intermediate-complete-partition.md)
and 24 checked extremes from the [new local references](peak-neighborhood-reference.md).
Their distance is the RMS displacement of actual transformed tetramer member
centers. Starting at the previously selected remainder maximum, deterministic
farthest-first selection stops when every candidate is within 0.5 Å of a
selected center. Exact ties use candidate order.

This gives **36 centers**, with maximum nearest-center RMS distance
0.493729254 Å. Reconstructing the coordinates as actual transformed members
independently verifies this cover. This is a cover of 64 observed poses,
not a cover of every important contact or a classification of metastable basins.
Equal center weights are proposal allocations, not equilibrium occupancies.

Each center defines the earlier geometry-scaled Cayley chart. Let
A = tr(M)I − M for the centered rigid-member moment M, rotated into neighbor
A's frame at that center. A Gaussian with latent standard deviation σ has
covariance σ² diag(I, ℓ²A⁻¹/4) and zero mean. The arbitrary angular length
ℓ cancels from the physical law when the covariance changes consistently.
Every component has its own center, orientation and Haar Jacobian.

The farthest observed pose from its nearest center is only slightly farther
in the Cayley metric: 0.493729859 Å-equivalent. Its standardized distance is
2.46865 for σ = 0.2 and 1.23432 for σ = 0.4. Those distances characterize
known-point proposal coverage; they do not bound unobserved physical weight.

## A measured local component

The radius-0.5 local reference supplies all 14,547 valid poses from 65,536
unconditional draws. One full covariance Gaussian is fitted in the fixed
peak-centered Cayley chart using the original positive importance weights.
The additive covariance floor corresponds to 0.05 Å translation and a
0.1° angular-axis scale. Source weight ESS is 571.7 and the largest share
is 1.28%; no weight is clipped or discarded.

Leaving out one of four populations moves the fitted mean by only
0.0040–0.0075 Å-equivalent. Generalized covariance eigenvalues relative to
the pooled fit span 0.923–1.066. Held-out conditional log-density gains are
2.60–2.82 nats over a centered geometric Gaussian at σ = 0.2, and
5.23–5.38 nats over σ = 0.4. The fits' overlapping training sets do not
provide independent significance tests. This validates a local proposal
fit, not a full-basin covariance, physical normalizer or sampling speedup.
Older and broader contact extrema still need other components.

The diagnostic model and its source reconstruction are archived in
`runs/ab-intermediate-local-fit-diagnostic-20260920`. Center selection,
distance matrices and its executable script are archived in
`runs/ab-intermediate-contact-centers-20260920`.

## Two frozen proposal arms

Both atlases have 38 components:

- 80% split equally among 36 geometric components;
- 16% assigned to the fitted local covariance;
- 4% assigned to that same local Gaussian with covariance multiplied by four.

The narrow and broad arms differ **only** in the geometric components'
σ = 0.2 versus 0.4. Both local components, all centers, means and weights
are identical. Their component Gaussian densities are combined first.
The complete physical proposal is

\[
g(x)=0.25g_{\rm product}(x)+
0.75\,[0.95g_{\rm atlas}(x)+0.05g_{\rm cube/Haar}(x)].
\]

The product component covers the original q ≤ 5 outer domain. The cube
contains the capture sphere and uses normalized Haar orientation. Every
contribution divides by the **full mixture density**, irrespective of the
branch that generated it; no density is conditioned on hard validity or q.
Original q, capture and both AB hard masks are retained. Invalid draws remain
zeros in the full fixed N. Two independent positive Poisson factors at
λ/z = 64 are averaged linearly for each valid pose.

These are independent importance-sampling calculations. They do not yet
measure reversible trajectory mixing or assembly rates. All source-center
and covariance decisions precede the new draws; historical local references
participated in proposal construction and are labeled calibration data.
The new arms remain separate estimates conditional on their frozen proposals.

`tools/prepare_contact_atlas.py` freezes the two models, physical inputs,
campaign budgets and analysis charts. The immutable preparation is
`runs/ab-intermediate-contact-atlas-preparation-20260920`, protocol SHA-256
`43a151a549f4c0156c4533d88af37974f29c916c4b393657d32901711128dc5b`.
Model hashes are in its `freeze.json`. Production uses the unchanged native
normalizer executable, SHA-256
`3a6a2dbba0ec5234c36cf66d7c40877336f22358a6a1ea91ea8366b344ee027b`.

| Arm | Geometric σ | Cloud-free valid / 256 | Populations × draws | Seed base |
| --- | ---: | ---: | ---: | ---: |
| Narrow | 0.2 | 73 | 4 × 16,384 | 101401010 |
| Broad | 0.4 | 54 | 4 × 16,384 | 101501010 |

Population seeds increment by 1009. Geometry probes draw from the raw atlas;
they exclude the product and cube branches. Scalar and vectorized guide
densities agree to within 2.4 × 10⁻¹⁴ at these probe poses. The production
audit also checks the complete outer cover-plus-guide density.

## Fresh physical results

All eight fixed populations completed: **131,072 unconditional draws** in
total. Every row passes independent reconstruction of the original q, full
38-component hybrid density, Poisson factors and denominator. Direct
poststratification retains all invalid and off-region zeros. The peak-centered
partition includes its entire outside region; the old weighted-chart partition
also covers the complete domain. Same-row covariance reconstructs each full
sum. All 16 highest-weight poses pass independent atom-union hard checks;
the smallest checked gap is 0.001771 Å.

| Full 2 ≤ q < 5 estimate | log Q | Row / population relative SE | Weight ESS | Largest draw | Sampler CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Narrow | 15.122254 | 22.00% / 21.77% | 20.65 | 15.75% | 457.36 |
| Broad | 18.258918 | 93.54% / 95.08% | 1.14 | 93.53% | 310.12 |

The broad estimate is dominated by one pose, so the difference between arms
does **not** establish a larger true mass or a converged full-window result.
The narrow populations have log estimates 14.79927, 15.53586, 14.54968 and
15.30431. Broad populations give 15.82791, 19.60757, 14.71435 and 14.78275.
Hard-volume estimates are −2.68492 and −2.53367 with 37.73% and 34.68%
row errors. This targeted atlas is a poor hard-volume reference compared
with the existing complete geometric controls, whose percent-level precision
remains useful. Positive complete support alone does not imply useful precision.

## Where the proposal helps

Both widths reproduce the earlier peak neighborhood's local mass, while
still sampling outside it:

| Peak-centered region | Narrow log Q / relative SE | Broad log Q / relative SE |
| --- | ---: | ---: |
| ρ ≤ 0.5 | 11.998190 / 3.54% | 12.089494 / 4.02% |
| ρ ≤ 1 | 13.176239 / 6.59% | 13.239111 / 9.98% |
| ρ ≤ 2 | 13.607364 / 7.87% | 13.612296 / 15.37% |
| ρ > 2 | 14.874008 / 28.12% | 18.249278 / 94.45% |

The earlier independent uniform references gave log Q = 12.054798 with
4.16% error in ρ ≤ 0.5. Their disjoint multiscale sum gave 13.568313 with
20.30% error in ρ ≤ 2, using 701.53 sampler CPU seconds. For this latter
fixed region, the observed **absolute variance × sampler CPU** improves by
9.45 times for the narrow atlas and 3.61 times for the broad atlas. This
uses variance on Q, not log Q, and accounts for the differing estimated means.
Preparation, training, historical discovery and audit costs are excluded.

This is a local integration benchmark with training-used calibration data.
It is not a measured full-window speedup, a test of untouched holdout
environments, or a measurement of MCMC contact mixing or crystal assembly.
The broader components expose an unresolved high-weight pose;
the narrow arm's local efficiency does not certify global coverage in practice.

## The unresolved contact

The broad maximum is r01, draw 14669, seed 101502019. It was drawn from
geometric atlas component 28, through the sampler's `learned` branch label.
That label refers to the Gaussian proposal family; this particular component's
covariance comes from geometry, not from the local fit. It has q = 2.391404,
old weighted radius 23.341481, and peak-centered radius 5.102887 Å-equivalent.
Thus it is outside the newly measured radius-two neighborhood and inside
the old 16 < r ≤ 32 partition piece. Both AB atomic gaps are positive,
0.082098 and 0.062321 Å.

Its original log two-cloud mean weight is 34.477778 and its full proposal
log density is 5.195415, producing log importance weight 29.282363. The two
cloud log weights are 34.310765 and 34.620849. The strongest narrow draw,
r01/11901, likewise lies in the old 16–32 piece and outside the current
local neighborhood, at q = 2.356108. Similar scalar q or old-chart radius
does not imply the same geometric neighborhood.

An independent density comparison reconstructs all 16 original extreme-pose
densities to within 7.2 × 10⁻¹⁵. At the broad maximum the narrow and broad
full proposal log densities are −4.924517 and 5.195415: a factor **24,833**
in favor of the broad proposal. The nearest geometric center is the old
R16 pose `r16:r01:68775`, 1.217842 Å member RMS away. Its standardized
Cayley distance is 6.090 for the narrow component and 3.045 for the broad
one. The actual generating center is slightly farther away. The earlier
0.5 Å cover applied to observed training poses and never enclosed this
newly discovered pose. This is a concrete local proposal-density gap, not
a claim about a finite region's integrated occupancy.

At the narrow maximum, broad/narrow proposal density is 0.55345 and the
nearest-center distance is 0.582944 Å. Thus the two strongest events test
different aspects of the coverage even though their q and old-chart radii
are similar.

### Independent clouds at the selected maxima

The existing fixed-pose executable was run with 256 fresh clouds at each
maximum, seeds 101601010 and 101602019. Shape, AB poses, activity, radius,
λ/z and the exact original exclusion-volume envelopes were preserved.
Compiled physical-kernel source hashes match the original production.

| Selected maximum | Fresh log arithmetic mean W | Relative SE | Original / fresh arithmetic mean W |
| --- | ---: | ---: | ---: |
| Narrow | 32.85004 | 4.54% | 2.318 |
| Broad | 33.99969 | 4.53% | 1.613 |

The broad pose's conditional Poisson-count 95% interval for log E[W] is
[33.92875, 34.10743]. Its large pointwise weight survives resampling, while
the original two clouds did inflate the estimate. The cloud test is
conditional on a selected fixed pose and numerical geometry. It measures
no neighborhood volume. Original regional estimators are unchanged; selected
cloud means are not retrospectively substituted into the full-window sum.

The current evidence supports transferring a well-measured local covariance
into a larger exact proposal mixture. It also exposes another contact
neighborhood needing controlled integration. The whole intermediate mass,
other registration regions, earlier SMC discrepancies, physical mixing and
assembly conclusion remain unresolved.

![Complete-region and local-reference comparison](../runs/ab-intermediate-contact-atlas-figure-20260920/contact-atlas.png)

## Artifacts

- Preparation: `runs/ab-intermediate-contact-atlas-preparation-20260920`
- Fresh campaigns: `runs/ab-intermediate-contact-atlas-{narrow,broad}-4x16384-l64-20260920`
- Full audit: `runs/ab-intermediate-contact-atlas-audit-20260920/analysis.json`
- Cross-density diagnosis: `runs/ab-intermediate-contact-atlas-extreme-densities-20260920/analysis.json`
- Independent clouds: `runs/ab-intermediate-contact-atlas-peak-clouds-20260920/analysis.json`
- Figure, SVG and archived inputs: `runs/ab-intermediate-contact-atlas-figure-20260920`

`tools/run_native_region_reference.py` launches the declared configurations
and models with `--q-min 2 --q-max 5 --q-upper-open --cover-scales 1`,
`--model-weight .75 --model-uniform-probability .05 --model-anchor-index 0`,
four populations and the counts/seeds in the protocol. Use fresh output
directories; never replace a failed or extreme population according to its
outcome. `tools/analyze_contact_atlas.py` validates the preparation and
campaigns, reconstructs all original rows, and performs the masked analysis.
Its finite-law tests cover full-N importance weights, exact boundary masks,
disjoint and physical/hard covariance, empty supported regions and merging
independent population moments.

Twenty-four relevant Python tests pass, including the five new atlas-audit
tests and the existing geometric/local-reference checks. Frozen analytic
sphere and overlap controls remain unchanged; no physical Rust kernel was
modified. No training update, fit or new physical pose enters a completed
campaign retrospectively.
