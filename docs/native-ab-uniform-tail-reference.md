# Independent uniform-shell checks of the native AB tails

The guide-width comparison left a small but unresolved tail in the original
Gaussian chart. Uniform six-dimensional shell sampling provides a separate
importance law for checking that tail. The original fitted Gaussian now
defines **coordinates only**; its Gaussian density does not weight or select
these reference poses.

The frozen chart is
`runs/native-ab-covariance-guide-20260920/model.json`, SHA-256
`86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667`.
The regions are `4<r<=5`, `5<r<=8` and `8<r<=12`, each intersected with
the **original** native interval `0<=q<=1`, the capture sphere and hard
validity against both fixed neighbors A and B. Radius 1.5 Å, activity
0.035 Å^-3, all fixed poses, and the registration metadata are unchanged.

For a standard-normal vector `v` in six dimensions and independent uniform
`u`, sample

`r=[a^6+u(b^6-a^6)]^(1/6),  y=r v/|v|`.

This is uniform in the shell `a<|y|<=b`, whose volume is
`V6=pi^3(b^6-a^6)/6`. Decode `x=mu+L y`, with `L L^T=Sigma`, using the
same left-Cayley rotation chart and fixed-neighbor coordinate frame as the
original proposal. Its physical Jacobian relative to translation volume
and normalized SO(3) Haar measure is

`J(y)=det(L)/(ell^3*pi^2*(1+|c|^2)^2)`.

Consequently `V6 * 1_capture * 1_native * 1_hard * J * Wbar` has expectation
equal to the shell's physical statistical weight when the independent
positive Poisson estimator has conditional mean `exp(z*C)`. Here `C`
uses overlap with the union of **all** fixed exclusions. Rejected poses
remain zero contributions in the full fixed sample count. No Gaussian
likelihood factor, success conditioning or fit correction is needed.

`tools/prepare_native_tail_regions.py` freezes the regions, source chart,
physical inputs and preparation protocol, then makes 256 independent
static geometry probes per region. Its five controls check the radial
inverse CDF, uniform volume and direction moments, translated/rotated
chart roundtrips, exact physical identity, and an independent finite
difference calculation of the full six-dimensional Jacobian. The latter
checks the normalized Haar factor via local rotation-vector volume
`domega/(8*pi^2)`. An independent agent also reviewed these conventions
and the complete AB identity checks.

Preparation is archived in
`runs/native-ab-tail-reference-preparation-20260920`:

| Region | Target-valid geometric probes | Forecast CPU seconds for 4 × 8,192 draws | Interval from geometry counts only |
| --- | ---: | ---: | ---: |
| Native 4–5 | 15 / 256 | 97 | 55–156 |
| Native 5–8 | 11 / 256 | 71 | 36–125 |
| Native 8–12 | 7 / 256 | 45 | 18–92 |
| Frozen competing R3 control | 72 / 256 | 464 | 375–562 |

The cost forecast uses 0.05036 historical sampling CPU seconds per valid
pose from the refined native campaign. The intervals propagate exact
binomial geometry intervals only; they omit changes in cloud cost and
overhead and do not predict physical precision. The preparation evaluated
no depletant clouds or physical statistical weights.

The optional competing R3 region was copied **byte unchanged** from
`runs/ab-competing-frozen-atlas-20260920/region-competitor-r3.json`, retaining
SHA-256 `e4b8cf45567ccc8a8106e4853fc5cdfab3b56f21dcc9abba86145acb3d223a98`.
It has original `q>=5` and its own previously frozen chart. Its physical
geometry and metadata match the native reference configuration; its
coordinate region is a separate competitor control.

The actual fixed budget was four independent populations of 8,192 draws
per native shell, with intensity/activity ratio 64 and two independent
clouds per valid pose. The executable, inputs and seeds were pinned before
production in `runs/ab-cooperative-weight-pilot-20260920/protocol.json`.
The latent-region executable SHA-256 is
`450a06a675215822d0055a1a253addb7a243516ed04624d849a578e87816dfdc`.
Seed bases are 99171010, 99181010 and 99191010 for the three respective
shells, incremented by 1009 per population.

## Independent reference results

The derived `compare_tail_references.py` uses exactly these shell
boundaries to partition the original hybrid contributions. Each method
retains its original unconditional denominator and all zero draws.

| Original-chart shell | Refined selected log Q (observed RSE) | Refined wide log Q (observed RSE) | Uniform reference log Q (observed RSE) | Uniform ESS |
| --- | ---: | ---: | ---: | ---: |
| 4–5 | 33.34403 (5.94%) | 33.45978 (13.26%) | 33.30200 (8.87%) | 126.7 |
| 5–8 | 33.07980 (10.64%) | 33.08739 (16.00%) | 32.65693 (12.56%) | 63.3 |
| 8–12 | 27.06243 (80.06%) | 30.27847 (17.70%) | 30.28155 (22.39%) | 19.9 |

The 4–5 reference agrees with both guides. In the 8–12 region, the wider
guide and the independent uniform reference agree closely, whereas the
selected guide missed most of their observed mass. This confirms a
specific coverage limitation of the narrower proposal at its fixed budget.
The agreement does not certify all remaining mass: uniform and wider
estimates still have appreciable uncertainty, and neither controls the
region beyond radius 12.

The 5–8 pilot was 34.5% below the refined selected estimate, a discrepancy
of 2.56 combined observed linear-scale standard errors. That diagnostic
motivated a **separate**, fixed-budget confirmation under the unchanged
uniform-shell law, rather than replacing any pilot population.

![Same shell under three independent proposal laws](../runs/native-ab-tail-reference-comparison-20260920/uniform-tail-comparison.png)

## Larger 5–8 confirmation

The confirmation used eight populations of 32,768 draws, fresh seeds
`99201010+1009*i`, and the same region, executable, intensity/activity
ratio and two-cloud estimator. Its plan was frozen before new outcomes in
`runs/ab-cooperative-weight-pilot-20260920/tail-5-8-confirmation-protocol.json`.
Pilot and confirmation observations remain separate.

| Uniform 5–8 campaign | Draws | log Q | Row / population RSE | ESS | Largest contribution | CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original pilot | 32,768 | 32.65693 | 12.56% / 14.27% | 63.3 | 5.33% | 61.75 |
| Fresh confirmation | 262,144 | 33.19840 | 15.94% / 20.88% | 39.4 | 13.78% | 486.15 |

The larger estimate is compatible with both refined-guide estimates: its
difference from the selected guide is 0.60 combined observed standard
errors. It is nevertheless 71.9% above the smaller uniform pilot, and its
concentration worsens despite eight times as many draws. One population
has log Q 34.08554, while the other seven lie between 32.81421 and
33.26393. The largest point carries 13.78% of the confirmation estimate.
Every population and contribution was retained.

The paired-cloud diagnostic assigns 69.4% of the confirmation's observed
variance to conditional Poisson noise, versus 52.1% in the pilot. This
decomposition is itself noisy. The results support the guide estimates'
scale and explain why agreement between two guides alone was insufficient,
but they do **not** establish a precise 5–8 shell mass. Larger sample counts
can reveal rare contributions that were absent from a smaller error
estimate. Reported errors and comparisons are diagnostics rather than
certified confidence bounds.

![Separate fixed-budget 5–8 confirmation](../runs/native-ab-tail-5-8-confirmation-comparison-20260920/five-eight-confirmation.png)

All **360,448 uniform-reference rows** passed the independent physical
pose/Jacobian, native-metric, original-weight and sample-hash analysis.
The comparison reclassified all 196,608 original hybrid draws using the
same old chart and checked scalar/matrix coordinate agreement. Separate
protocol audits verify actual archived binaries, shapes, configurations,
regions, seed schedules, cloud settings and fixed sample counts against
the frozen plans. Exact atom-union checks on 128 randomly selected
unconditional rows plus the 16 largest contributions per reference
campaign agreed with both-neighbor hard, capture and native flags:
**576 independently checked poses** across the three pilots and one
confirmation. These subset geometry audits do not claim to recompute
every atomic predicate in production.

The original three-shell comparison is in
`runs/native-ab-tail-reference-comparison-20260920`; the separate larger
5–8 comparison is in
`runs/native-ab-tail-5-8-confirmation-comparison-20260920`. Both include
protocol/geometry audit sidecars and retain their source hashes. Native
mass beyond original-chart radius 12 remains outside these uniform
references and cannot be silently treated as zero. These are finite
native-region checks, not global equilibrium or assembly results.
