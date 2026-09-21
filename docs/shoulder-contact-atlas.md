# Complete-shoulder contact atlas

This experiment tests whether three calibrated contact neighborhoods improve
importance sampling of the **complete original shoulder, strict `1 < q < 2`**.
The AB environment, capture region, hard shape, ideal-depletant parameters,
and original geometric definition of `q` remain fixed. This is independent
importance sampling, not a Markov-chain relaxation or assembly experiment.

Both independent width controls and their all-row audits are complete. Their
full-shoulder estimates agree within 0.24 combined observed row standard
errors, with about 2% observed error each. The broad arm gives 10.8 times the
previous mixture guide's observed importance-weight ESS per sampling CPU
second. This measures integration efficiency; it is not a trajectory mixing
speedup or an assembly result.

## Fits from the original reference laws

Each of the `direct`, `mixture`, and `geometry` contacts supplies exactly one
six-dimensional Gaussian. Each fit uses its own original **whole radius-.5
reference stream: 16 populations × 16,384 draws = 262,144 draws**. The fitting
domain is that geometric ball intersected with capture, hard validity and the
strict inner shoulder `1 < q < 1.1`.

All positive original importance weights enter once, without clipping or
equal-population renormalization. Invalid draws retain their original places
in the physical-normalizer denominator. The radius-.25 reference streams,
historical discovery streams, and priority-assigned subregions supply no fit
rows. In particular, the independent shell sums used to allocate proposals
are separate from the whole-ball streams used to fit them.

The coordinates use the frozen geometric center in the A-relative frame:
translation plus the Cayley orientation coordinate scaled by the unchanged
angular length. The fitted covariance is the original-weight scatter plus

\[
\operatorname{diag}\left(
  0.05^2 I_3,\;[\ell\tan(0.1^\circ/2)]^2 I_3
\right).
\]

The Gaussian is normalized globally in pose space, but its learned moments
describe only the conditional source neighborhood. It is not a fitted density
for an entire basin.

| Center | Positive fit rows | Original weight ESS | Largest normalized weight |
|---|---:|---:|---:|
| Direct | 7,305 | 262.83 | 0.03077 |
| Mixture | 5,584 | 213.36 | 0.02372 |
| Geometry | 5,258 | 192.34 | 0.03439 |

Each fit has 16 leave-one-population-out diagnostics. The held-out weighted
log-density scores compare it with the original geometric chart at standard
deviation factors .1 and .2. Every fold improved those conditional scores.
The overlapping training sets and weighted held-out scores do not provide
physical integration errors, independence of the folds, or evidence that
unseen contact configurations have been covered. Independent chart and
laboratory-frame density evaluations agreed within `5e-14` in log density.

The deterministic fit outputs, original log-normalizer reconstructions,
population diagnostics, and source/archive hashes are in
[the frozen fit directory](../runs/ab-shoulder-local-guides-20260921/report.json).

## Five-component proposal and width control

Let `g_old` be the unchanged two-component proposal and `g_D`, `g_M`, `g_G`
the three new single-Gaussian fits. The learned proposal is

\[
g=\tfrac12 g_{\rm old}
 +\tfrac12\left(a_Dg_D+a_Mg_M+a_Gg_G\right),
\qquad
(a_D,a_M,a_G)=(0.305468,\;0.166701,\;0.527831).
\]

These allocation weights normalize the independently estimated physical
masses of three disjoint, strict-inner regions. Priority is direct first,
then mixture excluding the direct radius-.5 ball, then geometry excluding
both earlier balls. Each assigned mass uses its prespecified independent
radius-.25 inner piece and radius-.5 outer shell. The allocation changes
proposal frequencies only; it does not assign, truncate or reweight fit rows,
and it does not substitute finite-region masses for the new full normalizer.

There are two frozen arms:

| Arm | New fitted covariances | Old two-component family |
|---|---|---|
| Narrow | Unchanged | Unchanged |
| Broad | Multiplied by 4, so standard deviations double | Unchanged |

Both retain the same component means, centers and allocation weights. Each
arm has 16 independent populations of 32,768 draws, with disjoint streams and
fixed diagnostic prefixes of 8,192, 16,384 and 32,768 draws per population.

The actual proposal is the complete hybrid

\[
s(X)=0.99\,[0.95\,g(X)+0.05\,u(X)]+0.01\,c(X).
\]

Here `u` is uniform in the 36 Å cube centered on the capture center, with Haar
orientations. The geometry-derived product cover `c` contains the original
`q <= 2` domain. Capture and strict-shoulder restrictions are applied after
the draw. The full proposal probabilities are therefore .9405 learned,
.0495 cube/Haar, and .01 complete cover.

**Every importance denominator uses the sum of all applicable components and
branches**, including their overlap. The probability of only the selected
component is not an admissible denominator. All out-of-window, capture-invalid
and hard-invalid draws remain zero contributions to the original full `N`.

The frozen models, complete-cover check, launch commands and hashes are in
[the preparation protocol](../runs/ab-shoulder-contact-atlas-preparation-20260921/protocol.json).

## What the analysis measures

The archived density auditor reconstructs the original `q`, complete hybrid
denominator, cloud factors and unconditional counts before new masks are
applied. The shoulder decomposition is

\[
Q_{1<q<2}=Q_{1<q<1.1}+Q_{1.1\le q<2},
\]

and the inner part is the union of the three finite balls plus its directly
sampled complement. The exact boundary `q = 1.1` belongs to the outer part.
Geometric ball boundaries are closed at .25 and .5; priority removes overlap
without changing densities.

All masks share the same original rows and denominator. The analysis retains
their signed sample covariance: overlapping masks have their intersection
term, while disjoint masks retain the negative covariance of their sample
means. It reconstructs both the mass and variance of each disjoint partition.
Nested sample prefixes also share rows; their differences use the fixed-IID
prefix covariance relation, rather than treating prefixes as independent
repetitions. Narrow and broad arms remain separate.

Previous finite references calibrate only identical strict-inner masks:
the matching whole radius-.25 and radius-.5 balls, the priority-assigned
radius-.5 regions, and their union. They are never compared as substitutes
for the full `1 < q < 2` normalizer. No historical remainder is added to new
finite-region masses. The previous reference data informed proposal fitting
and allocation, so observed-error contrasts are calibration diagnostics,
not held-out significance tests or unseen-tail bounds.

## Production results

| Diagnostic | Narrow | Broad |
|---|---|---|
| Full shoulder log normalizer | 25.077182 | 25.070578 |
| Full shoulder row / population RSE | 1.95% / 1.43% | 1.94% / 1.65% |
| Strict-inner log normalizer | 24.725647 | 24.681958 |
| Inner finite-union log normalizer | 24.414727 | 24.389103 |
| Inner outside-union log normalizer | 23.405990 | 23.311022 |
| Outside-union row / population RSE | 7.45% / 5.52% | 4.80% / 3.37% |
| Outer-shoulder log normalizer | 23.861112 | 23.937399 |
| Largest row / population weight fractions | 1.10% / 7.09% | 0.55% / 7.08% |
| Full importance-weight ESS | 2,604.3 | 2,649.8 |
| Sampling CPU seconds | 5,548.2 | 3,080.9 |
| Importance-weight ESS / sampling CPU second | 0.4694 | 0.8601 |

Each arm retains all **524,288** original draws. The whole-shoulder width
ratio is 1.00663. The inner, outer, finite-union and outside-union width
contrasts are respectively 1.30, -1.60, 0.79 and 1.05 combined observed row
standard errors. Thus their agreement is not restricted to the whole total.
The outside-union contribution is still appreciable: 26.72% and 25.39% of
the respective inner estimates, or 18.80% and 17.21% of the full shoulder.

The independent finite-union reference has log Q=24.424744. Its contrasts
with the narrow and broad arms are small. Within the disjoint regions, the
narrow direct-contact estimate is 11.05% below the reference, a 1.99 combined
observed row-SE difference; the corresponding broad difference is 1.17 SE.
The narrow mixture region is 10.50% above its reference, or 1.45 SE. These
remaining calibration differences are reported rather than removed by
substituting reference means.

The broad arm's full log Q changes 25.08508 → 25.06934 → 25.07058 at the
fixed prefixes. The narrow sequence is 25.17845 → 25.11662 → 25.07718.
Its earliest prefix is 10.66% above the final estimate, or 3.15 estimated
same-stream difference standard errors using the final row variance.
That concentration remains a reason to retain an independent repeat check;
the close final width estimates alone do not establish convergence.

The previous complete-support mixture guide had 0.07962 importance-weight
ESS per CPU second, with 9.18% observed row error at 262,144 draws. The new
narrow and broad ratios are **5.90 and 10.80** relative to that guide.
This comparison excludes the cost of learning and independently calibrating
the contact neighborhoods; that is an upstream cost to amortize across future
use. It also compares complete proposal designs: the old mixture used hybrid
model probability 0.75, while both new arms use 0.99 (each retains a 0.05
cube branch within the model branch). Thus 10.8× is not an isolated effect of
adding the three contact fits. The narrow-versus-broad comparison does keep
that allocation fixed. The broad arm is cheaper because it triggers fewer expensive valid-pose
cloud evaluations, while preserving almost the same full-weight precision.

![Width controls, finite masks and fixed sample prefixes](../runs/ab-shoulder-contact-atlas-figure-20260921/shoulder-contact-atlas.png)

The audit is
`runs/ab-shoulder-contact-atlas-assessment-20260921/analysis.json`.
It reconstructs the full original density on **1,048,576** rows, matches
every stream hash, preserves all same-row partitions, and independently
checks the eight largest physical contributions from each arm against both
neighbors at atomic resolution. All 16 checks are hard-valid. The reviewed
implementation passes 84 shoulder-related tests, including the new
original-density, dense-covariance and comparison-direction controls.

Concentration, rare large weights, dependence on the width arm, and unresolved
complements remain qualifications on the reduction in observed variance.
Weight ESS and fitted-density scores are not MCMC mixing measures.
Agreement at finite sample size cannot establish that all relevant contact
arrangements have been found or that protein assembly has become observable.

Implementation: [fitter](../tools/fit_shoulder_local_guides.py),
[preparation](../tools/prepare_shoulder_contact_atlas.py),
[analysis](../tools/analyze_shoulder_contact_atlas.py), and
[mask/covariance accumulator](../tools/shoulder_atlas_moments.py).

## Independent repeat with twice the draw count

The repeat retains **1,048,576 unconditional draws per arm**: 16 populations
of 65,536. Configuration and five-component model files are byte-identical
to the original run, including the full `.99/.05` hybrid law, both physical
neighbors, strict `1 < q < 2` target and all 23 analysis masks. Seeds were
frozen before sampling as `114101010 + 1009*i` for narrow and
`114201010 + 1009*i` for broad, with `i=0,...,15`; they are disjoint from
original and calibration streams. The repeat budget was selected after the
original results. Original and repeat estimates remain separate.

Parentheses in the normalizer rows below give observed row RSE.

| Repeat diagnostic | Narrow | Broad |
|---|---|---|
| Full shoulder log normalizer | 25.094457 (1.67%) | 25.123489 (1.57%) |
| Full shoulder population RSE | 1.39% | 1.34% |
| Strict-inner log normalizer | 24.725690 (1.41%) | 24.756958 (2.07%) |
| Inner finite-union log normalizer | 24.419675 (1.03%) | 24.459799 (2.55%) |
| Inner outside-union log normalizer | 23.392461 (4.50%) | 23.398567 (3.25%) |
| Outer-shoulder log normalizer | 23.918143 (4.40%) | 23.942141 (2.10%) |
| Full importance-weight ESS | 3,574.5 | 4,032.7 |
| Sampling CPU seconds | 11,079.1 | 6,160.2 |
| Importance-weight ESS / sampling CPU second | 0.3226 | 0.6546 |

Relative to its original estimate, narrow increases **1.74%**, or 0.67
combined observed row standard errors. Broad increases **5.43%**, or 2.13
row SE and 2.50 population SE. Broad's change is concentrated in the inner
region (+7.79%, 2.29 row SE); its outer estimate changes only +0.48%
(0.14 row SE). Within the repeat, narrow/broad full weight is 0.97139,
a −1.27 row-SE difference. The corresponding finite-union, outside-union
and outer differences are −1.44, −0.11 and −0.50 row SE. These are
observed-variance diagnostics, not Gaussian significance claims.

The broad priority-assigned mixture-contact region remains a discrepancy:
it is **28.38% above** the original broad estimate (2.14 row SE) and
**26.28% above** its finite reference (2.38 row SE). Its own row RSE is
7.52% and weight ESS is 176.9. Narrow's assigned direct-contact estimate
remains 9.05% below its reference (1.65 row SE). The complete finite-union
reference contrasts are smaller: −0.16 row SE for narrow and +0.91 for
broad. No reference mean replaces any repeat contribution.

At the frozen 16,384 → 32,768 → 65,536 draws-per-population prefixes,
full log Q is **25.10558 → 25.09496 → 25.09446** for narrow and
**25.12966 → 25.13647 → 25.12349** for broad. Their earliest-versus-final
differences are 0.39 and 0.23 same-stream SE, using the final row variance
and prefix covariance. Narrow's earliest inner outside-union estimate is
still 15.86% above its final value, a 2.03
same-stream SE contrast. Nested prefixes and contrasts sharing the original
estimate are correlated; they are not additional independent runs.

The observed full N-times-variance-of-the-mean
increases by factors 1.51 and 1.46 relative to the respective original
campaigns, and observed ESS per CPU second decreases in both arms. These
changes and the component discrepancy remain qualifications even though
the repeat's full-prefix diagnostics are quieter. Neither the original
nor repeat is pooled, declared globally converged, or interpreted as an
MCMC speedup.

The [repeat assessment](../runs/ab-shoulder-contact-atlas-repeat-assessment-20260921/analysis.json)
preserves all full-N denominators, same-row partitions and within-run
prefix covariance. Its unchanged base analyzer validates all **2,097,152**
new rows and checks the eight largest contributions per arm against both
neighbors at atomic resolution.
