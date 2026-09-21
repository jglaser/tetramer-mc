# Native and competing contacts in the full vessel

The independent finite-region controls established appreciable weight near a
second registered native pocket. The next comparison includes the entire
original atomic-wall domain, including native sites outside the former D170
capture sphere and every valid unregistered-contact pose. It retains the
same two observed fixed tetramers, repaired hard shape, depletant radius
1.5 Å and activity 0.035 Å⁻³.

The [completed comparison](../runs/mobile-wall-contact-comparison-20260921/report.md)
finds native registry in the dominant observed weight in both proposal arms.
The broader guide substantially improves the native estimate, but the
unregistered-contact estimate remains dominated by individual observations.
These results motivate continued native-growth tests; they do not close the
equilibrium-coverage milestone.

## Completed results

All eight physical populations and both independent audits passed, covering
131,072 unconditional poses and using 559.0 sampler CPU seconds. The
[validation ledger](../runs/mobile-wall-contact-validation-20260921/validation.json)
records the frozen inputs, source identities and 22 focused tests. The raw
audits independently reconstruct every proposal density and atomic-wall test.
The subsequent classifier leaves no native-entry/unbound inconsistency.

| Proposal | Region | log Qz | Observed row RSE | Population RSE | ESS | Largest row |
|---|---|---:|---:|---:|---:|---:|
| Original | All registered native entries | 63.8745 | 97.68% | 97.86% | 1.0 | 97.68% |
| Coverage | All registered native entries | 60.8650 | 28.26% | 20.01% | 12.5 | 21.09% |
| Original | Contact without native entry | 39.4934 | 38.44% | 13.80% | 6.8 | 24.96% |
| Coverage | Contact without native entry | 40.0994 | 99.47% | 99.34% | 1.0 | 99.46% |

The coverage arm recovers the original native R4 witness at log Qz=36.3104,
versus the independent regional 36.0852 (log difference 0.2252±0.1971 observed
row SE). The original atlas misses that region again. The coverage arm's
alternative R32 estimate is 60.8650 versus 61.0335 independently
(−0.1685±0.2930). Its alternative R5 estimate is lower, 60.1985 versus
60.6821 (−0.4836±0.2158). Thus even known-region agreement is incomplete;
the R5 discrepancy is a reason to retain proposal and sample-size sensitivity.
The original arm's R32 estimate is 2.8410 log units above its independent
reference and is almost entirely one weighted observation.

The [saved-outlier inspection](../runs/mobile-wall-contact-outliers-20260921/report.md)
places that row, original/r02/draw6611, at radius 5.4475 in the alternative
chart: it belongs to the already measured 5–8 shell. All eight largest rows
inspected across the two arms match the same motif7/4 triangle. The original
outlier's unusually small proposal density amplifies its importance weight;
it is not a newly discovered native site. Its brighter Poisson cloud also
contributes 94.9% of the two-cloud mean. No rows are removed or reweighted.

The observed native/contact-without-entry log ratios are 24.38 and 20.77
for the original and coverage arms. Those are conditional point estimates,
not converged free-energy differences or bounds on unseen competitors.
The coverage arm's unregistered-contact estimate has ESS approximately one;
its four population estimates vary substantially. Adding geometric coverage
has not yet made that remainder reliable. Native weight remains dominated by
the registered motif7/4 triangle inside D170. Sites outside D170 are included
in the physical target and reporting even when their observed contribution
is small.

Paired Poisson clouds attribute about 81% of the original total's observed
weight variance to cloud noise, versus 8.5% in the coverage arm. For the
unregistered-contact remainder, these fractions are about 55% and 47%.
These are finite-sample decompositions, not asymptotic variance certificates.
Increasing cloud intensity alone would not resolve the coverage arm's pose
sampling problem.

The subsequent [independent threshold reference](mobile-threshold-reference-results.md)
finds that the largest no-entry rows lie just beyond the native registration
cutoff in the same motif7/4 environment. On an identical finite R4, its fresh
no-entry estimate is 2.67 log units above the coverage arm, while still having
ESS 2.7. The global comparison therefore remains unresolved; the new result
does not justify reclassifying those rows or discarding the discrepancy.

![Separate full-wall estimates and population variation](../runs/mobile-wall-contact-comparison-20260921/wall-contact-comparison.png)

The subsequent mobile test measures growth beyond the known three-tetramer
triangle and retains a preassembled control. A converged conditional native
pocket does not include the free-energy cost of forming its scaffold.
The [four-body geometric design](../runs/mobile-four-body-design-20260921/recommendation.md)
finds hard-valid motif3 and motif8 interfaces on the exposed third tetramer,
each with six supported native monomer contacts. The recommended matched
starts are the same triangle plus a free fourth tetramer or one attached at
motif8, with all four bodies mobile. The
[completed eight-run control](mobile-four-body-growth-results.md) now produces
catalogue-consistent four-body attachment in three of four free starts, while
all four preattached controls stay connected and change docking motif. All
traces pass the generalized four-body observer. This is a native-informed
accessibility result, not a measured association free energy or converged
assembly test.

The physical wall has radius 223.32617672378387 Å, centered at the origin.
Every atomic sphere of the mobile tetramer must fit inside it. The ideal
depletant bath permeates this protein wall. The computational capture radius
is 273 Å, conservatively enclosing every wall-valid center; this capture
sphere is not an additional physical boundary.

## Frozen comparison

[The controller](../tools/mobile_wall_contact_campaign.py) fixes two proposal
arms, each with four independent populations of 16,384 unconditional poses,
two Poisson weight estimates per valid pose and auxiliary intensity λ/z=64.
It launches at most eight physical workers. Invalid draws retain zero weight;
there is no retry, valid-only normalization, adaptive extension or pooling
across proposal laws.

The original arm uses the existing 150-component reciprocal atlas. The
coverage arm retains those components and adds ordinary Gaussian guides at
saved, geometrically diverse unregistered contacts, the two measured native
sites and known wall-valid native sites outside D170. Covariances are copied
from existing charts and scaled by fixed factors. The guide is frozen before
fresh sampling, and noisy bath weights do not select or fit its centers.
Both arms marginalize the two anchor frames and include a 10% uniform
cube/Haar floor. The complete normalized proposal density enters every
importance weight.

The physical classes are exhaustive: registered native entry at any catalogue
site; exclusion contact without native entry; and unbound without native
entry. Here, “without native entry” refers to the fixed catalogue and geometric
thresholds, rather than a proof that a pose has no native structural feature.
The original single-site q score remains a separate historical diagnostic.
Reporting includes each class inside and outside D170, native registration
with one or both neighbors, closed registry triangles and motif identities.

The original native R4 and alternative native R5/R32 regions are additional
witnesses. Their global importance estimates are compared with the completed
independent uniform-volume references. These estimates are not added to the
full-wall integrals. The two proposal arms, their population dispersions,
effective sample sizes and largest-weight shares stay visible separately.

Full proposal support and exact weighting define the intended integral, but
do not certify finite-sample coverage. Agreement on known witnesses is a
necessary check; it cannot exclude an unseen competing pocket. These are
conditional one-mobile-body integrals, not evidence of bulk equilibrium or
continued crystal growth. The portable
[assembly examples](../examples/README.md) remain a separate all-mobile,
native-informed test.
