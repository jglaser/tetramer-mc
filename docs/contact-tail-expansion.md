# Frozen geometric expansion for the unresolved R4 contact tails

The new guide adds **eight six-dimensional Gaussian components** to the existing
84-component integration guide. It targets the two previously identified weak
subdivisions: native entry outside historical R5 in orthant 55, and contact
without native entry in orthant 63. The physical R4 region, repaired shape,
1.5 Å depletants, 0.035 Å⁻³ activity, complete native classifier, translation/Haar
measure and Poisson estimator are unchanged. This is **native-label-aware
integration design**, separate from blind assembly proposals.

The [frozen preparation](../runs/contact-tail-expansion-20261001/plan.json) uses
only r00/r01 from the bank and protected arms of the completed million-draw
campaign for fitting: four populations and 4,194,304 attempted draws. All were
already generated and classified. Another four previously inspected populations
(r02/r03, 4,194,304 attempts) provide retrospective diagnostics after the model
freeze. They are disjoint from this fit, but are not pristine prospective data.

## Construction and preserved support

Each fitting population supplies one anchor in each critical subdivision: its
largest contribution to the paired-cloud physical second moment under the
frozen 84-component guide. Around each anchor, select the 128 nearest **distinct**
poses of the same class from the four training populations, without restricting
neighbors to that orthant. Ties follow stable population/draw order. Estimate
the full translation–rotation covariance about the actual anchor, with the
existing 0.05 Å / 0.1° physical covariance floor. The existing capped-weight
fit limits one neighbor to 1/16 of the fitting mass; physical importance weights
are never clipped in evaluation. Source contributions use their original
attempted population sizes.

Keep 50% uniform R4 probability. Within the remaining Gaussian half, assign 80%
to the unchanged original components and 20% equally to the eight additions.
All Gaussians remain normalized on full R6, including their out-of-region tails.
The target's R4 and hard indicators, rather than a truncated proposal, remove
invalid contributions. Thus

\[
q_{92}=0.5u_{R4}+0.4g_{84}+0.1g_{8},\qquad
q_{92}\ge0.8q_{84}.
\]

This exact inequality preserves support and limits any regional second-moment
increase to 25%. It does not guarantee a gain, resolve unknown modes, or produce
a free-energy bound.

An offline allocation-only control assigns the same additional 20% to the old
components with greatest posterior density at the eight training anchors.
It keeps every old mean/covariance. Both controls were frozen before opening
held-out arrays, so their difference tests new local geometry rather than
simply increasing allocation to existing components.

## Retrospective results

Entries below are paired-cloud physical second-moment ratios against the old
84-component guide, pooled within each source arm only. Smaller is favorable.
Individual populations, all classes and all original radial/angular/orthant
subdivisions remain in [the full analysis](../runs/contact-tail-expansion-20261001/retrospective-analysis.json).

| Region | New geometry: bank / protected holdouts | Allocation only: bank / protected holdouts |
|---|---:|---:|
| Native inside R5 | 1.250 / 1.250 | 1.148 / 1.155 |
| Native outside R5 | 0.548 / 0.210 | 0.768 / 0.529 |
| All contact without entry | 1.018 / 1.019 | 1.001 / 1.037 |
| Native outside R5, orthant 55 | 0.294 / 0.056 | 0.867 / 0.481 |
| Contact without entry, orthant 63 | 0.084 / 0.029 | 0.181 / 0.184 |

The observed improvement in both targeted subdivisions exceeds the allocation
control. Whole competing-contact coverage is essentially unchanged, while the
R5 region pays the expected allocation cost. Orthant 58, relevant to the
restricted-SMC comparison, becomes approximately 22–24% more expensive; it is
retained explicitly, not removed from evaluation.

These moment ratios are **not sampling speedups or new physical mass estimates**.
In bank-source orthant 63, the expanded proposal's second-moment contributions
still have ESS only 2.97 and a largest term of 55.6%. The corresponding protected
values are 50.3 and 5.3%. Original-source second moments are also concentrated.
The marked difference between source arms is a reason for fresh physical
validation, not evidence that all tails are covered. No historical failed gate
has been changed.

## Checks and prepared prospective pilot

Five focused tests cover exact mixture identity and support retention,
untruncated anisotropic covariances, the allocation-only control, invalid
allocation, distinct-neighbor fitting and covariance regularization.
Across 329 arbitrary/anchor coordinates, the new density agrees with independent
SciPy normalized Gaussians within 2.18e−11 in log density and the existing
independent output auditor within 1.82e−12. The analytical proposal integral is
exactly one. At this preparation stage existing physical reference receipts were
reused, with no candidate Rust execution or physical draw. Preparation and all
saved-row diagnostics took 38.4 seconds on one scientific worker.

The [prospective pilot preparation](../runs/contact-tail-expansion-20261001/prospective-pilot-plan.json)
declares two arms (old 84 and new 92), four fresh independent populations per
arm, 16,384 attempts per population: **131,072 attempts total**, two independent
clouds per valid pose, intensity ratio 128, and at most two simultaneous physical
jobs within the existing eight-job/32-worker limit. It preserves every attempted
draw, all original strata, and the complete native classification. Seeds are
checked against all discovered historical protocol declarations and training
populations. It neither extends nor pools old physical estimates.

The candidate's [Rust sphere reference](../runs/contact-tail-sphere-reference-v2-20261001/validation.json)
subsequently passed at activities 0 and 2 with 16,384 attempted draws each. The
exact same 92 means, covariances and weights were rebound only to the analytic
sphere region. Independent reconstruction of every Rust proposal density agrees
within 7.11e−15; AO/Haar physical quadrature differs by at most 0.41 observed SE.
The first audit wrapper used the conditional-ray campaign schema rather than
the Gaussian schema and failed before auditing. Its artifacts remain intact;
the corrected audit reused all original zero-activity draws without resampling.
The activity-2 draw allocation and seed were unchanged.

The [frozen pilot](../runs/contact-tail-pilot-20261001/protocol.json) was launched
after six controller tests, the five guide tests, independent code review and
its source/input preflight passed. Its [status](../runs/contact-tail-pilot-20261001/status.json)
records every physical job and the subsequent independent density/weight audit
and one-pass complete native classification. The controller starts at most two
physical jobs, stops new launches on failure and drains all started children;
the original eight-job/32-worker machine limits remain enforced. Its independent
auditor and native classifier are the existing validated implementations.

The pilot measures whether the tail proposal merits a new fixed confirmation.
Its small allocation cannot substitute for the decision-driving convergence
campaign or decide finite-system assembly. No optional stopping, extension,
resampling of failed populations, or automatic full-vessel launch is allowed.

## Completed prospective pilot

The [completed review](../runs/contact-tail-pilot-review-v2-20261001/report.md)
contains the plot, every failed material stratum comparison, and bound source
hashes. All 131,072 attempted draws completed, with both independent audits and
all eight complete-native classifications. Runtime was 26.01 minutes wall and
2,438.01 physical sampler CPU seconds. Every raw row remains in its original
population; there was no resampling or extension.

The targeted native-complement orthant 55 and competing orthant 63 improve
observed importance ESS per physical CPU by **3.65× and 3.75×**. Whole-native
efficiency changes by 1.09× and whole-competing efficiency by 0.87×. These are
importance-weight diagnostics, not Markov-chain independent contact samples.
Aggregate native-minus-competing free energies agree: **−18.7926 ± 0.1744** and
**−18.7881 ± 0.1323 kBT**, with population-based 95% delta-interval half-widths.
They remain conditional on the fixed scaffold and R4 region.

**Convergence still fails:** native-complement ESS is 126/166 with largest
contributions 4.09%/4.71%, and 14 of 34 materially contributing stratum comparisons
fail the frozen tolerances. Competing orthant 55, distinct from targeted native
orthant 55, has 18.7 times greater observed mass under the expanded guide; even
there its ESS is only 7.16. This exposed coverage issue should be inspected
before designing another fixed confirmation. The pilot neither resolves older
SMC discrepancies nor bounds the vessel remainder. Full-vessel and assembly
decision gates remain closed; the finite-system verdict is unresolved.

The follow-up [cooperative-contact inspection and frozen guide test](cooperative-contact-guide.md)
archives the problematic competing-55 geometry. Four further covariance fits
failed their retrospective recommendation checks and were inferior to a
geometry-preserving allocation control. No additional physical pilot was launched.

- [Frozen 92-component guide](../runs/contact-tail-expansion-20261001/expanded-guide.json)
- [Model freeze before holdouts](../runs/contact-tail-expansion-20261001/model-freeze.json)
- [Anchors, neighbors and covariance diagnostics](../runs/contact-tail-expansion-20261001/fit.json)
- [Independent density checks](../runs/contact-tail-expansion-20261001/guide-validation.json)
- [Source/input/output manifest](../runs/contact-tail-expansion-20261001/completion.json)
- [Preparation helper](../tools/prepare_contact_tail_expansion.py)

Reproduce into a fresh directory; this command launches no physical simulations:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_contact_tail_expansion.py \
  --out /path/to/new-contact-tail-preparation
```
