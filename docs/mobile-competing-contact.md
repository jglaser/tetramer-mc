# The competing attachment in the mobile pilot

The mobile pilot's persistent competing attachment was outside the earlier
fixed-neighborhood integration domain. After mapping body 2 onto original
neighbor A, body 0 lies 96.169 Å from the original capture center, compared
with the old radius of 18 Å. Its original registration score is 58.6604.
Consequently, the earlier native/other weight ratio in that capture ball
does not compare native binding with this attachment.

The observed native pair is the original crystallographic AB arrangement:
body 2 maps to A and body 1 to B. However, its final geometry differs slightly
from the ideal scaffold (B's member RMS displacement is 0.0840 Å). The
following calculation retains that exact observed deformation. It does not
replace the old AB calculation or reuse its statistical weights.

The observed body 0 touches A with a minimum atomic gap of 0.0393 Å; its
gap to B is 15.044 Å. The mapped original native pose also remains feasible,
with gaps 0.0371 and 0.0505 Å and 21.79 Å of actual atomic wall clearance.
Thus both placements are geometrically available in the same observed
neighborhood. Neither minimum gap measures their integrated statistical weight.

The exact snapshots, common proper isometry and inverse, coordinate origins,
wall-center transformation and source hashes are retained in
[the geometry evidence](../runs/mobile-competing-geometry-20260921/analysis.json).
Only twelve distinct relative translations of body 0 against body 2 occur
after burn, once common rotations and center shifts are removed.

![The old capture domain and two alternative positions of the same tetramer](../runs/mobile-competing-geometry-figure-20260921/mobile-competing-geometry.png)

## Why valid native candidates were rejected

Every postburn hard-valid new-native candidate for body 0 was retained in
[the acceptance diagnostic](../runs/mobile-competing-acceptance-20260921/analysis.json).
There are 34: 23 posterior transport proposals and 11 full-mixture capture
proposals. Twenty-three would produce two native body-pair edges. None was
accepted. Fourteen have a positive sampled bath log factor, but their
reverse/forward proposal factors overwhelm it. The largest recorded
acceptance probability is 0.000215; the sum over this particular candidate
list is 0.000216. These are conditional probabilities for the recorded
proposals and auxiliary clouds, not stationary rates.

![Bath factors and reverse-density penalties for every retained candidate](../runs/mobile-competing-acceptance-20260921/competing-acceptance.png)

The final competing pose has log Gaussian-mixture density −22.226 relative
to A. It is supported primarily by a deliberately broad defensive component,
not a fitted contact chart. Its Mahalanobis radius in that component is only
0.704: the low density is due to the component's breadth, not an extreme
tail excursion. A narrow native destination can therefore be proposed while
the reverse density at the competing source remains tiny.

The recorded bath factor is a random auxiliary factor, not an exact energy
difference. This diagnosis identifies a proposal mismatch but cannot establish
whether the total native or competing basin has greater physical weight.
Adding a frozen chart at an observed competing contact is a testable correction
to coverage; its normalization and reverse density must enter the same full
mixture formulas as the existing charts.

There is a more specific directional mismatch. For the reciprocal relative
pose, with body 0 as anchor and body 2 as moving body, the legacy log density
is already **14.0363**. Components 141–143, from the previously discovered
outside-R8 contact atlas, supply nearly all of that density. Changing which
particle serves as anchor is a proper relative-pose inversion, not a spatial
reflection or change of chirality. The contact is represented in one ordering
but only diffusely in the other. This is a proposal-coverage defect, not a
violation of the existing Metropolis balance correction.

The [frozen atlas control](../runs/mobile-competing-atlas-preparation-20260921/plan.json)
retains the original 150 components with total weight 0.8, then adds the
geometric contact chart and a reciprocal chart with weight 0.1 each. Both
centers are exact; the reciprocal covariance uses a checked first-order
pose Jacobian. Its Gaussian is a new normalized proposal component, not the
exact nonlinear pushforward of the forward Gaussian. The full mixture and
Haar corrections remain mandatory. This raises the original-direction log
density to 17.0151. The legacy atlas still contains native priors; this control
does not become geometry-only by adding observed contact charts.

## Independent local weight comparison

Freeze the final body 1 and body 2 poses in their original laboratory frame.
The conditional density for mobile body 0 is, up to constants independent of
its pose,

\[
 H(x;S)H_{\rm wall}(x)\exp[z C(x;S)],\qquad
 C(x;S)=|E(x)\cap(E_1\cup E_2)|.
\]

The two fixed exclusions remain a union. Pairwise overlap sums would
overcount their shared depletion volume. The single-body exclusion volume
and the fixed scaffold's weight cancel only in this conditional comparison;
the cost of making the scaffold remains outside it.

A laboratory-frame capture ball of radius 170 Å is a technical enclosing
domain for the two finite regions. The rigid atomic shape fits inside a
49.4871 Å body-centered sphere, so every captured pose lies inside the
223.3262 Å protein wall, regardless of orientation. This conservative bound
makes the existing wall-free regional integrator exact for these regions;
it does not remove the wall from the full assembly target. The native center
has radius 163.761 Å, and its q≤1 translation neighborhood also fits.

Use the unchanged original native Gaussian chart to define its radius-4
latent ball intersected with native q≤1. At the observed competing pose,
define a new zero-mean Gaussian chart with translation scale 0.25 Å and
Cayley scale `ell*tan(0.25°/2)`, without fitting to bath weights. Freeze its
radius-3 ball and a separate radius-3-to-5 shell. These disjoint finite
regions are explicit integration targets, not assertions that either basin
has been completely covered. The first comparison samples four independent
populations of 8,192 unconditional draws in each core, with two independent
Poisson clouds per valid pose and lambda/z=64.

Uniform six-dimensional ball sampling has a known volume, and the existing
translation/Cayley map supplies the physical translation/Haar Jacobian.
Every hard-invalid draw stays in the denominator. Positive Poisson weights
estimate `exp(z*C)`; the same poses also estimate the hard-only regional
volume. This separates local available volume from regional depletion
enhancement. Additional shells, independent sampling and shape/scale controls
are required before interpreting the result as a complete basin preference.

## Completed regional integrals

All twelve independent 8,192-draw populations and their saved-row audits
completed: four each for native R4, competitor R3 and the disjoint competitor
3–5 shell. The [comparison report](../runs/mobile-competing-reference-comparison-20260921/report.md)
checks the common physical configuration, wall certificate, frozen region
definitions, source/binary identities, unique seeds and saved audit hashes.
It sums the independently sampled core and shell integrals, rather than
pooling their rows as if drawn from one distribution.

| Finite region | log Q with depletion | Observed row relative SE | Hard-only log Q0 |
|---|---:|---:|---:|
| Native R4, q≤1 | 36.0852 | 7.89% | −20.7854 |
| Competing R3 | 18.1864 | 4.35% | −17.5943 |
| Competing shell 3–5 | 19.2868 | 13.07% | −14.8035 |
| Competing R5, core plus shell | 19.5740 | 9.87% | −14.7439 |

For these specified regions, native R4 has **16.511 kBT lower free energy**
than competing R5 (observed row SE 0.126 kBT). The point estimate separates
into a hard-only accessible-volume contribution favoring competing R5 by
6.042 kBT and a regional depletion enhancement favoring native R4 by
22.553 kBT. Enhancement is the ratio of paired region integrals, not the
exponential of mean overlap. The analysis keeps their covariance when
reporting ratio errors.

The outer shell supplies **75.03%** of the measured competitor R5 weight.
It therefore matters greatly for coverage even though including it does
not reverse this finite-region comparison. Nothing here bounds the weight
outside R5 or at other attachment sites. Four population means can also
understate variability: the shell's population relative SE is 2.89%, compared
with 13.07% from its observed rows. Both are retained; neither overrides
concentration or establishes convergence of a whole basin.

The result supports testing the proposal-coverage defect, while preserving
the distinction between the thermodynamics of specified regions and global
equilibrium. It cannot be substituted into a mobile crystal growth rate or
used to declare all competing arrangements disfavored.

## Completed all-mobile control

The [matched campaign](../runs/mobile-competing-atlas-benchmark-20260921/protocol.json)
completed all eight 2,000-sweep runs and both four-run audits. Each run starts
from the identical trapped snapshot; all three tetramers move. Physical
parameters, local moves, GCA, center shifts and branch fractions are unchanged.
The two atlases are tested at correlations 0 and 0.9 with two independent
seeds each. There is no fitting during production. All 80,000 attempted
updates and 16,008 stored endpoints were checked.

| Atlas | Correlation | Replicate | First native attachment of body 0, sweep | First three native edges, sweep | Total sampler CPU, s |
|---|---:|---:|---:|---:|---:|
| Legacy | 0 | 0 | Not observed by 2000 | Not observed | 59.15 |
| Legacy | 0 | 1 | 1216 | 1216 | 69.13 |
| Legacy | 0.9 | 0 | 416 | 416 | 74.03 |
| Legacy | 0.9 | 1 | 456 | 456 | 78.75 |
| Augmented | 0 | 0 | 419 | 580 | 83.90 |
| Augmented | 0 | 1 | 1822 | 1843 | 71.11 |
| Augmented | 0.9 | 0 | 775 | 775 | 79.38 |
| Augmented | 0.9 | 1 | 99 | 99 | 81.40 |

The 1–2 native body-pair edge remains present, although its registry can
change, so the first native attachment also connects all three bodies.
Three edges is a stricter graph criterion.
Neither graph criterion identifies the particular native R4 region of the
fixed-scaffold integral above. Actual motif IDs and every registry transition
are retained in the [comparison](../runs/mobile-competing-atlas-comparison-20260921/comparison.json).

The augmented atlas changes accessible routes but does not demonstrate a
uniform first-attachment speedup in this small control. All four augmented
runs and three of four legacy runs reach native connectivity. Legacy
successes overcome proposal corrections of −42.51, −41.54 and −44.21 with
sampled bath factors of +50.79, +46.98 and +53.75. Thus the old coverage
penalty is severe, not an absolute obstruction. Augmented moves can instead
accept intermediate registered contacts with corrections near zero, and
subsequently change registry. These sampled bath factors are not exact energies.

There is also one actual contact-environment exchange: augmented c=0,
replicate 1 loses the initial 0–2 exclusion contact at sweep 9, attaches to
body 1 at sweep 11, and regains contact with body 2 at sweep 1822. Every
other run retains its initial exclusion contacts. Other native-edge losses
are registration changes while physical contact remains. This single event
is useful evidence of an accessible route, not a stationary exchange-rate
estimate or a completed return to the original full contact environment.

The recorded detachment uses the new source chart 150 and existing broad
destination chart 27. Its proposal log correction is +37.980 and sampled
bath log factor −37.768, giving unit acceptance. The later attachment at
sweep 11 has correction −35.640 and bath factor +35.629, giving acceptance
probability about 0.989. These are the actual recorded auxiliary factors,
not counterfactual energy evaluations. They illustrate how resolving the
narrow source contact can balance the proposal cost of moving between a
contact and a broad environment without changing the physical target.

![Every matched run, with native and exclusion-contact histories](../runs/mobile-competing-atlas-comparison-20260921/mobile-competing-atlas-comparison.png)

This controlled test remains native-informed and starts from one selected
non-equilibrium state. It establishes neither equilibrium populations nor
template-free assembly. The next representation test can address the
directional coverage defect without adding a separately fitted reciprocal
Gaussian for each contact.

The [completed validation record](../runs/mobile-competing-completed-validation-20260921/validation.json)
binds the artifacts, audited totals and 37 passing Python tests for geometry,
paired statistics, reciprocal covariance, campaign allocation/lifecycle and
event summaries. No Rust physical kernel changed in this stage. The exact
reciprocal representation below was not part of those executable tests.

## Exact reciprocal representation

For identical rigid particles, swapping which one is the anchor replaces the
relative pose `T=(t,R)` by `I(T)=(-R^T t,R^T)`. This is an involution with unit
absolute Jacobian in translation volume times normalized rotational Haar
measure: the translation block is a proper rotation with an overall sign,
and rotational inversion preserves Haar measure. The cross derivative of
translation with respect to rotation does not change that block-triangular
Jacobian argument. Both rotations remain proper; chirality is unchanged.

Thus a normalized pair proposal can be made exactly reciprocal without
fitting a second Gaussian:

\[
 G_{\rm sym}(T)=\tfrac12G(T)+\tfrac12G(I(T)).
\]

An independent draw samples G and then, with probability one half, inverts
the resulting relative pose. A correlated chart implementation would retain
an additional inversion label per source/destination chart, invert before
encoding or after decoding, and evaluate the full symmetric mixture in its
posterior source responsibilities and Hastings correction. In general the
inverted component is not Gaussian in the original coordinates, so merely
inverting the mean and covariance is not this exact construction.

This stores each discovered contact once, with an analytic reciprocal
branch. It is now implemented for frozen spherical capture and posterior
proposals, with composed chart maps and full densities checked by independent
reference tests. Symmetrizing a pair proposal does not symmetrize
the physical many-body environment or remove any depletion correction.

The [implementation design and reference tests](reciprocal-contact-proposal-design.md)
spell out the required joint component/inversion responsibilities, inverse
trace and full-mixture density evaluation. The source inversion label cannot
be chosen by an independent fair coin.

The new [eight-run reciprocal control](../runs/mobile-reciprocal-atlas-benchmark-20260921/protocol.json)
compares the original 150-component atlas with exact symmetrization of every
component. It adds no fitted centers or covariances. At the initial trapped
contact, the two ordered legacy log densities are −22.22573 and 14.03630;
the reciprocal log density is 13.34315 in both directions. A preflight checks
this identity across 156 probes, including every ordered physical pair.
These are proposal densities, not physical statistical weights.

All eight 2,000-sweep runs and both independent four-run audits completed.
All 80,000 attempted updates and 16,008 stored endpoints passed the checks,
including the exact virtual-branch density and trace reconstruction on the
protein shapes. The complete [comparison artifact](../runs/mobile-reciprocal-atlas-comparison-20260921/comparison.json)
retains motif IDs, rejected candidates, individual events and CPU times.

| Atlas | Correlation | Replicate | First native attachment of body 0 | First native triangle | Sampler CPU, s |
|---|---:|---:|---:|---:|---:|
| Legacy | 0 | 0 | Not observed by 2000 | Not observed | 60.22 |
| Legacy | 0 | 1 | 1997 | 1997 | 56.90 |
| Legacy | 0.9 | 0 | 277 | 277 | 76.40 |
| Legacy | 0.9 | 1 | 81 | 81 | 79.31 |
| Reciprocal | 0 | 0 | 373 | 1063 | 83.34 |
| Reciprocal | 0 | 1 | 637 | 718 | 71.81 |
| Reciprocal | 0.9 | 0 | 358 | 358 | 71.69 |
| Reciprocal | 0.9 | 1 | 14 | 14 | 88.66 |

First native attachment occurs in all four reciprocal and three of four
legacy runs. Two reciprocal runs lose and regain body 0's native registry;
none of the legacy runs does. Another reciprocal run loses and regains
the scaffold's native registry while body 0 stays registered. However,
**none of these eight runs loses the original 0–2 exclusion contact or
exchanges neighbors**. Registry changes must not be counted as detachment.
These few independent preparations do not establish a uniform speedup.

The reciprocal c=0.9, replicate 1 event at sweep 14 uses the inverse branch
of existing base component 141 as source and ordinary component 146 as
destination. Its log proposal correction is +3.2140 and sampled bath factor
−3.0698, giving unit acceptance. The exact wrapper makes a useful narrow
source chart available without fitting another Gaussian or changing the bath.

Reciprocal c=0, replicate 1 supplies an explicit registry excursion. It
registers at sweep 637 (proposal correction −0.4148, bath factor +1.9845),
loses registry at 679 (−0.0305, −0.6512, acceptance probability about 0.506),
and forms a native triangle at 718 (−8.6834, +51.5979). These are recorded
auxiliary factors, not free energies. The [saved-pair diagnostic](../runs/mobile-reciprocal-contact-return-diagnostic-20260921/analysis.json)
places the unregistered return at original competing-chart radius 18.84:
only 2.31 Å translation and 4.10° rotation from the initial relative pose,
but already outside R12. The other neighbor is mobile, so this is not a
return to the fixed-scaffold integration region. It reinforces the unresolved
coverage issue rather than contradicting its finite-region weight estimate.

![Exact reciprocal and legacy proposals, all eight runs](../runs/mobile-reciprocal-atlas-comparison-20260921/mobile-reciprocal-atlas-comparison.png)

This establishes a useful representation and additional registry routes.
It does not settle equilibrium exchange efficiency, compression, geometry-only
discovery, or crystal growth. The original native-informed atlas is still
supplied in both arms.

## Extending the competing region to radius 12

Eight further independent 8,192-draw populations integrate the disjoint
5–8 and 8–12 shells on the same observed two-neighbor scaffold. The
[comparison](../runs/mobile-competing-outer-comparison-20260921/report.md)
sums these with the completed R5 reference; it does not rerun previous draws
or combine different proposal populations as though they shared a density.

| Competing region | log Q with depletion | Observed row relative SE | Population relative SE |
|---|---:|---:|---:|
| Shell 5–8 | 19.9859 | 38.65% | 53.73% |
| Shell 8–12 | 19.2953 | 57.60% | 64.43% |
| R12, core and all shells | 20.7577 | 22.50% | 28.98% |

The measured native-R4/competing-R12 log ratio is **15.3275**, with observed
row SE 0.2385 and population SE 0.2968. These errors describe the sampled
weights; they do not account for undiscovered modes. The new shells supply
69.4% of measured R12 weight, but their effective sample sizes are only
6.69 and 3.01. One outer-shell row supplies 56.2% of its integral, and lies
near radius 12. Neither convergence nor a basin boundary is established.

The [saved-row diagnostic and importance design](mobile-outer-importance-design.md)
attribute about 59% and 93% of observed variance to pose variation in the
two shells. Additional Poisson clouds alone cannot resolve the dominant
outer-shell uncertainty. The next control keeps the regions fixed and uses
a frozen Gaussian guide with a uniform defensive component, exact proposal
density and fresh independent physical weights. Native R4, all unvisited
regions outside R12, and other contact environments remain separate coverage
questions.
