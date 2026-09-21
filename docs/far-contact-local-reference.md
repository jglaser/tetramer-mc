# Independently measuring the displaced far contact

The historical far-region estimates depended strongly on proposal width.
Their two largest width-four contributions were only 0.320476 Å apart in
rigid-member RMS, but both lay outside the old fitted competitor R3 region.
Fresh geometric neighborhoods establish that this contact has finite weight;
they also quantify why uniform capture sampling scarcely visits it.

The physical target is unchanged: both AB neighbors fixed, rigid mobile
tetramer, depletant radius 1.5 Å, activity 0.035 Å⁻³, capture radius 18 Å,
original 2 Å/15° metric, and normalized proper SO(3) Haar measure. The
[complete support proof](far-capture-reference.md) gives q < 37 throughout
capture, so the far mask is exactly 5 ≤ q < 37.

## Freeze geometry before drawing new weights

The [96 historical candidates](far-contact-candidates.md) preserve their
original rows, proposal densities, and selection provenance. The local chart
is centered on the largest original width-four far contribution: population
`scale-4-r03`, draw 15662, seed 99324040. Selecting that center does not turn
its historical weight into an independent reference measurement.

Let the four centered rigid-member positions have second moment M, and
let A = tr(M)I − M, expressed in the chart's proper orientation frame.
For translation δt and Cayley rotation c relative to the selected pose,
define

\[
\rho^2=|\delta t|^2+4c^T A c,
\qquad
\mathrm{RMS}^2=|\delta t|^2+\frac{4c^T A c}{1+|c|^2}\le\rho^2.
\]

The geometric chart makes a six-dimensional Euclidean ball tractable.
Its physical center/Haar Jacobian is

\[
J(c)=\frac{1}{8\pi^2\sqrt{\det A}(1+|c|^2)^2}.
\]

A ρ-ball is contained in the corresponding member-RMS ball. It is neither
the whole RMS ball nor an asserted thermodynamic basin. No covariance is
fitted to the new physical weights. The second historical maximum lies at
ρ = 0.320477 Å, inside the smallest frozen ball.

`tools/prepare_far_peak_reference.py` freezes radii 0.5, 1 and 2 Å, with four
independent populations of 16,384 unconditional draws per radius. Seed bases
are 106001010, 106101010 and 106201010, incremented by 1009 per population.
The exact latent-region executable is unchanged; its SHA-256 is
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.
The artifact protocol records the executable digest authoritatively.

Uniform six-ball volume is V₆(R) = π³R⁶/6. Each original draw contributes

\[
Y=V_6(R)J(c)\,H_{AB}\,I_{\rm capture}\,I_{5\le q<37}
\frac{W_1+W_2}{2}.
\]

The two positive Poisson estimators use the full exclusion union and
λ/activity = 64. Invalid and masked poses contribute zero to the original
fixed-N average. This includes many-body depletion without replacing it by
pairwise overlap sums.

## Separate nested balls from disjoint shells

Before physical sampling, `analysis-plan.json` specified using the R=0.5
campaign for [0,0.5], R=1 for (0.5,1], and R=2 for (1,2]. These disjoint
shell estimates have independent random streams. Their means and variances
are added to estimate the R≤2 neighborhood. Nested whole-ball estimates
remain separate controls and are never added.

Every ball and shell is also partitioned into its intersection with the
unchanged old fitted R3 region and its complement. These positive masks use
the same rows, so their negative covariance is retained. The old fitted
Mahalanobis radius is not replaced by the geometric radius. The full old R3
reference overlaps this neighborhood and cannot simply be added to it.

| Fresh source | log Q | Row relative SE | Population relative SE |
| --- | ---: | ---: | ---: |
| Whole ρ≤0.5 ball | 15.23539 | 3.85% | 5.02% |
| Whole ρ≤1 ball | 16.64899 | 10.36% | 14.76% |
| Whole ρ≤2 ball | 17.97999 | 44.75% | 60.86% |
| Prespecified disjoint-shell sum, ρ≤2 | **17.42075** | **22.78%** | **22.37%** |
| Shell sum intersecting old R3 | 14.01240 | 23.88% | 13.05% |
| Shell sum outside old R3 | 17.38710 | 23.54% | 23.10% |

The three selected shells separately give log Q 15.23539, 16.44761 and
16.74676, with row relative errors 3.85%, 11.46% and 43.87%. Their contributions
show why a tiny neighborhood around the recorded point is insufficient.
Conversely, one R2 draw at ρ=0.5506 supplies 39% of its whole-ball estimate;
the predeclared shell allocation measures that inner shell more densely.
About 96.69% of the shell-sum weight lies outside the old fitted R3 region.
This is a measured fraction with sampling uncertainty, not an exact mass ratio.

## Why complete flat support is insufficient

The shell-sum hard pose volume is
**V = (2.1840 ± 0.0467) × 10⁻⁵ Å³**, using one observed row SE.
The corresponding physical weight is Q ≈ 3.6791×10⁷ Å³. Their ratio has
log(Q/V) ≈ 28.15: this is an average Boltzmann enhancement over a finite hard
region, not a single-pose energy or an assembly free energy.

The independently tested flat proposal has constant density
g = 2.122502717×10⁻⁵ Å⁻³ everywhere inside capture. Thus its estimated
probability of a hard-valid visit to this particular neighborhood is
gV = 4.6356×10⁻¹⁰ per unconditional draw. At the completed budget N=262,144,

\[
\mathbb E[N_{\rm visits}]=NgV\simeq0.0001215,
\qquad P(N_{\rm visits}=0)=(1-gV)^N\simeq0.9998785.
\]

These are plug-in sampling-rate estimates from the measured hard volume,
not rigorous probability bounds. The corresponding mean waiting count is
about 2.16 billion draws. It is a property of this independent flat proposal,
not a measured MCMC correlation time or a physical attachment rate.

There is also a geometric upper bound that needs no sampled hard volume.
Since J(c) ≤ J(0), even the entire unmasked chart ball has volume at most
V₆(2)J(0) = 0.00069603 Å³. Consequently the flat campaign expects at most
0.003873 visits to the entire ball, with no-visit probability at least
0.99613. Removing hard-invalid poses only makes visits less likely. The
inequality is analytic; these reported values are evaluated in binary64,
not with a formal interval proof. This bound concerns visiting this specific
neighborhood and places no upper bound on its depletion-weighted mass.

The flat full-far estimate, log Q=12.81466 with 59.1% observed row error,
therefore cannot resolve this local contribution at its current budget.
Its error bars do not cover weight it never encountered. The historical
guided estimates also remain width-sensitive. Complete support is necessary
for correctness, but sampling its high-weight part requires contact guidance.

## Compare identical local regions in the old runs

`tools/compare_far_local_history.py` remasks all 163,840 historical draws
without changing their original densities, Poisson factors, or unconditional
denominators. It also remasks the 262,144 flat draws, reusing their completed
density audit. This separates the local neighborhood from all remaining
far poses; neither an all-other weight nor a whole-far weight is substituted
for the local integral.

| Same ρ≤0.5 region | Unconditional draws | Positive draws | log Q | Row relative SE |
| --- | ---: | ---: | ---: | ---: |
| Historical width 1 | 32,768 | 2 | 12.51005 | 73.06% |
| Historical width 2 | 65,536 | 11 | 15.04292 | 41.39% |
| Historical width 4 | 65,536 | 2 | 18.86956 | 70.92% |
| Fresh local reference | 65,536 | 6,822 | 15.23539 | 3.85% |

The two width-four observations give 37.87 times the fresh local mean and
supply 93.48% of that historical whole-far estimate. Its large observed error
makes the linear difference only 1.37 combined row standard errors. Moreover,
the region was selected using those historical extremes, so this comparison
is exploratory after selection; it is not a nominal significance test.

The flat campaign has exactly zero geometric visits to any of the three
local balls, even before hard restrictions. Its local integral is therefore
unresolved, not measured to be physically zero. This is consistent with both
the hard-volume rate estimate and the independent geometric upper bound.

The positive outside-R2 historical estimates give log Q 11.83039, 12.16530
and 13.98854 for widths 1, 2 and 4, with row errors 40.98%, 19.24% and 50.95%.
Those remaining-space estimates are still proposal-sensitive; their observed
small contributions do not bound undiscovered far contacts. The comparison
retains the full covariance matrix for nested balls and disjoint shells.

All matched results and visit-rate formulas are recorded in
`runs/ab-far-local-matched-history-20260920/analysis.json`. The single successful
read-only pass follows a documented validator correction for reporting-only
metadata; no physical samples were changed or regenerated.

## Validation and next use

All 196,608 local rows passed independent original q, proposal/Jacobian,
Poisson, full-N and provenance reconstruction. Independent member geometry
agreed with the chart identities; log-Jacobian errors were at most 3.56×10⁻¹⁵.
The top eight rows per radius passed independent atom-union checks, with
minimum gap +0.0001733 Å. The three campaigns consumed about 389.5 sampler CPU
seconds. These checks supplement, rather than formally verify, the numerical
collision implementation.

Preparation: `runs/ab-far-peak-reference-preparation-20260920`.
Production: `runs/ab-far-peak-reference-20260920`.
Full audit: `runs/ab-far-peak-reference-assessment-20260920/analysis.json`.
Figure: `runs/ab-far-contact-reference-figure-20260920/far-contact-references.png`.
The campaign and analysis plans, source files, hashes, original draws, and
terminal statuses are preserved. `tools/analyze_far_peak_reference.py` reports
the full radial/old-R3 partitions and their covariance.

The next useful proposal should allocate explicit probability to the newly
measured contact neighborhood while retaining all other known contacts and a
complete-domain component. It should be frozen before independent populations
test local agreement and the remaining-space contribution. No full-far
convergence, reversible mixing speedup, or assembly/model verdict follows
from this finite-region calculation.
