# Independent local calibration of the inner shoulder

The direct reference on the original **1<q<1.1** interval is dominated by
three observations from different populations. Together they contribute
95.9039% of its log Q=26.09425 estimate. They lie within a geometric radius
of 0.25 Å around its largest contribution, whose original q is 1.06510.
This is a specific finite region we can measure independently.

The historical [shoulder analysis](shoulder-proposal-guides.md) also shows
that these poses already have appreciable density under the frozen guides.
Their large direct-reference weights do not establish a missing mode. The
new calculation tests their integrated neighborhood weight, rather than
substituting a selected pose or a proposal density for that integral.

Both fixed AB neighbors, capture radius 18 Å, depletant radius 1.5 Å,
activity 0.035 Å⁻³ and the original 2 Å/15° metric remain unchanged.
The local target is intersected with the **strict original inner interval**,
capture and full AB hard validity. It is not the full shoulder or an entire
physical basin.

## A geometry-defined chart

Let the rigid-member positions have zero centroid, second moment M and
rotational moment A=tr(M)I−M. Express poses relative to physical neighbor A,
and center a left-Cayley rotation chart on the selected historical pose.
With relative translation δt and Cayley vector c, define

\[
\rho^2=|\delta t|^2+4c^T A_{\rm anchor}c.
\]

The actual squared rigid-member RMS displacement from the peak is

\[
d_{\rm RMS}^2=|\delta t|^2+
\frac{4c^T A_{\rm anchor}c}{1+|c|^2}\le\rho^2.
\]

Thus each chart ball is contained in the corresponding RMS ball; it is not
a complete RMS cover. The covariance-shaped coordinate transform comes
from rigid-member geometry and has no fitted weights or widths. Its
translation/normalized-Haar Jacobian is

\[
J=\frac{1}{8\pi^2\sqrt{\det A}\,(1+|c|^2)^2}.
\]

Sampling a uniform six-ball of radius R therefore gives the unbiased
fixed-region contribution `V6(R) J × I_inner × I_capture × I_hard × Wbar`,
where `V6(R)=π³R⁶/6` and `Wbar` is the linear average of two independent
positive Poisson factors. The exclusion calculation includes the union of
both fixed neighbors. Invalid draws remain zeros in the full fixed count.

## Frozen calculation

`tools/prepare_shoulder_peak_reference.py` verifies the exact source row,
sample hashes, fixed physical configuration, atom gaps and previously
recorded geometric chart before freezing two independent campaigns:

| Radius | Independent populations | Draws per population | Seed schedule |
| --- | ---: | ---: | --- |
| 0.25 Å | 16 | 16,384 | 109001010+1009i |
| 0.5 Å | 16 | 16,384 | 109101010+1009i |

There are **524,288 new unconditional draws**, with 16 workers per radius,
32 workers total. Both campaigns use the unchanged reviewed latent-region
executable, intensity/activity ratio 64 and two independent clouds.
The budgets, regions and streams are fixed before any physical outcomes.

Separate 256-pose probes found 14 and seven target-valid poses at the two
radii. These probes evaluate geometry only. Independent coordinate,
rigid-member and Haar-Jacobian checks passed; the probe results did not
change either production allocation.

Preparation, archived dependencies, exact commands and probe records are in
`runs/ab-inner-shoulder-peak-reference-preparation-20260921`, with protocol
SHA-256 `113221a9ba9f3b34a718e81d451b1cf4175a2c7e227f10c7bd348fe1195f0ca8`.
Original production goes into separate `r0p25` and `r0p5` directories under
`runs/ab-inner-shoulder-peak-reference-20260921`. Launch records are retained
in `runs/ab-inner-shoulder-peak-reference-launch-20260921`.

## Same-region comparisons and remaining space

The whole-ball estimates overlap and must remain separate. The prespecified
radius-0.5 combined estimate adds only the radius-0.25 campaign's [0,0.25]
contribution and the independent radius-0.5 campaign's (0.25,0.5] shell,
with their independent variances. It does not add the two whole balls.

The historical direct inner reference and the two complete-shoulder guided
confirmations are also classified using this identical chart and the strict
inner q interval. Their original proposal densities and original full N
are retained: 1,048,576 draws for the direct reference and 262,144 for each
guided confirmation. Guided draws in 1.1≤q<2 are zeros for the inner
comparison, rather than being removed from its denominator.

The inner mass outside each ball remains explicit, with covariance for
pieces formed from the same rows. The historical observations helped select
this neighborhood, so comparisons with them are discovery/calibration
diagnostics rather than independent hypothesis tests. They are not pooled
with the fresh references or stitched into a new claimed whole-shoulder
estimate. Agreement here would settle a finite local question; the
remaining shoulder, native tail and far-contact checks would still matter
for the [complete AB comparison](ab-regional-weight-status.md).

## Completed reference and matched-history results

All 32 physical populations finished successfully. Both independent
original-density auditors returned zero. The derived analysis checks and
classifies **524,288 fresh rows and 1,572,864 historical rows**, preserving
their original unconditional denominators. Forty extreme poses pass
separate atomic checks against both neighbors. Fourteen new controls check
mask boundaries, support, sample laws, covariance, independent shell
allocation and failed-auditor handling; 15 existing geometry and boundary
controls also pass.

| Fresh reference | log Q | Row / population RSE | Weight ESS | Sampler CPU s |
| --- | ---: | ---: | ---: | ---: |
| Radius 0.25 | 21.950635 | 1.60% / 1.17% | 3854.8 | 1314.2 |
| Radius 0.5, whole ball | 23.234404 | 6.17% / 4.98% | 262.8 | 345.2 |
| Radius 0.5, prespecified disjoint sum | 23.238833 | 5.36% / 4.82% | — | 1659.5 |

The radius-0.5 stream independently estimates the smaller radius-0.25
region at log Q=21.93448 with 11.17% row error, agreeing with the dedicated
small-ball reference within 0.14 combined observed standard errors.
Its (0.25,0.5] shell has log Q=22.91619 with 7.37% error. The combined
radius-0.5 estimate uses that shell and the dedicated small-ball estimate;
it shares rows with the whole radius-0.5 control and is not independent
of it.

The decisive comparison uses identical finite regions:

| Original sampling law | Radius ≤0.25 log Q (row RSE) | Radius ≤0.5 log Q (row RSE) |
| --- | ---: | ---: |
| Historical direct inner cover | 26.05243 (78.35%) | 26.06301 (77.53%) |
| Historical mixture guide | 22.05869 (23.75%) | 23.00028 (13.91%) |
| Historical geometry guide | 21.25108 (49.87%) | 23.02341 (28.71%) |
| Fresh geometric reference | 21.95063 (1.60%) | 23.23883 (5.36%) |

The historical direct estimate assigns **60.45 times** the freshly measured
weight to the radius-0.25 region. It contains only three nonzero rows there,
the same three that selected the region. This calibrates the formerly
dominant event; its large original importance contribution is not evidence
of an equally large integrated physical neighborhood. The historical
estimate and its error are retained, and the data-dependent selection
prevents interpreting this ratio as an independent significance test.

The mixture guide agrees in scale for the smaller region. Its estimate
of the (0.25,0.5] shell is lower than the fresh reference by 2.49 combined
observed row standard errors. This finite-sample discrepancy remains
visible rather than being hidden by the whole-ball comparison. Neither
historical guide has high precision on all local pieces.

![Same local regions and the explicitly uncalibrated remainder](../runs/ab-inner-shoulder-peak-reference-figure-20260921/inner-shoulder-local-reference.png)

Most observed guided inner-shoulder weight is outside radius 0.5:
log Q=24.45785 for the mixture and 24.82887 for the geometry guide,
with 14.08% and 39.86% row errors. Those are 81.1% and 85.9% of their
respective inner totals. The independent local calculation therefore
resolves the conspicuous direct-reference outlier without supplying a
new whole-shoulder normalizer. The remaining guided maxima provide
concrete locations for further independent checks.

The assessment and executed source archive are in
`runs/ab-inner-shoulder-peak-reference-assessment-20260921/analysis.json`,
SHA-256 `a5b341daa73b8eabd2a9f9752c52055aeedfd0b96d26be5210699ba86999fb65`.
The figure and its archived data are in
`runs/ab-inner-shoulder-peak-reference-figure-20260921`.

The subsequent [guided-peak controls](shoulder-guide-peak-references.md)
measure both remaining recorded maxima with the same geometric construction.
They form a disjoint three-neighborhood union using explicit exclusions,
while preserving a separate outside-union estimate and the unchanged full
shoulder target.
