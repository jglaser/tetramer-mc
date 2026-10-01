# Cooperative competing contacts: saved geometry and a bounded guide test

The new local covariance fits **do not justify another physical campaign**.
They improve a difficult tail in retrospective scoring, but simple reallocation
of existing components performs better, and the moment estimates remain dominated
by a few saved rows. No physical jobs or geometry/classifier reruns were launched.
The finite-system assembly question remains unresolved.

## What the saved configurations show

The [geometry archive](../runs/contact-tail-pilot-review-v2-20261001/geometry-inspection-v2/inspection.json)
preserves every competing-contact orthant-55 row in the completed pilot: four
from the 84-component arm and 74 from the 92-component arm. It includes original
population/seed/draw identities, unconditional denominators, full latent poses,
both Poisson log weights, proposal densities, Jacobians, complete stored labels,
and hashes of the raw outputs, label files and scored caches. All 78 contact both
fixed scaffold tetramers and have zero native-anchor matches. **Competing orthant
55 is distinct from native-complement orthant 55.**

| Pilot proposal branch | Rows | Fraction of observed competing-55 mass |
|---|---:|---:|
| 84-component guide, component 41 | 4 | 100% |
| 92-component guide, original component 41 | 9 | 49.3% |
| 92-component guide, added component 85 | 11 | 4.5% |
| 92-component guide, added component 87 | 54 | 46.2% |

The two largest expanded-arm weights still come from original component 41.
The expanded arm's leading five weighted poses have minimum surface gaps to each scaffold
between approximately 0.002 and 0.160 Å. This is a cooperative near-core contact
family reached sparsely by existing components, rather than evidence that all
its weight comes from a newly discovered basin. These descriptive fractions are
not converged physical occupancies.

The archived original training split contains another 732 competing-55 poses;
all also contact both scaffolds. Those original poses alone train the next
candidate. None of the fresh pilot's 78 poses enters its fit.

## A normalized four-component correction

The [frozen preparation](../runs/cooperative-contact-guide-preparation-v2-20261001/plan.json)
adds four full-covariance Gaussian components, one anchor from each original
bank/protected r00/r01 population. The anchor maximizes that population's paired
physical second-moment contribution under the frozen 92-component guide among
hard-valid competing-55 poses contacting both scaffolds. Covariance uses the
128 nearest distinct eligible training poses, original source importance
weights, the existing 1/16 fitting-weight cap, and the existing physical floor
of 0.05 Å and 0.1°. The covariance is centered about the actual anchor and
retains translation–rotation correlations. No new covariance scale is tuned.

The four anchors' paired scaffold gaps are (0.0116, 0.0051), (0.0734, 0.0246),
(0.0023, 0.0012), and (0.0264, 0.0134) Å. Conditioning the **fit data** on both
contacts lets its covariance describe the joint constraint. It does not make
subsequent Gaussian draws satisfy either contact or hard validity exactly.

The complete proposal is

\[
q_{96}=0.5U_{R4}+0.45G_{92}+0.05G_4,
\qquad q_{96}\ge0.9q_{92}.
\]

Here each Gaussian is normalized over **all of R6**, not truncated to R4,
contact configurations, or hard-valid poses. The uniform half is normalized on
R4. Gaussian draws outside R4 and hard-invalid draws retain zero target weight
in the original attempted-draw denominator. The estimator remains
\(J\,[W_1+W_2]/(2q)\) with the complete mixture density. The pointwise retention
bound limits any regional second-moment increase to 1/0.9, or 11.1%, even for
an unobserved region. It is not a bound on that region's physical mass.

The allocation-only control keeps all 92 existing means and covariances and
uses

\[
q_{\rm alloc}=0.5U_{R4}+0.45G_{92}
 +0.05(0.25g_{41}+0.75g_{87}).
\]

The old-component assignments follow their posterior density at the four
training anchors. This control has the same normalization and retention bound
and tests whether adding geometric detail earns its cost.

## Split, validation and result

The plan, fitting rules, covariances, allocation control and candidate-density
checks were frozen before this preparation read the original r02/r03 holdouts.
Those four holdouts were inspected in earlier work, so this is **retrospective
validation**, not a claim of pristine unseen data. The completed fresh pilot
is scored separately as diagnostic-only data; it affects neither fitting nor
the declared launch-recommendation rule. All original radial, angular and
orthant strata remain in the saved scoring output.

| Source holdouts | Competing-55 paired second moment, new geometry / baseline | Allocation-only / baseline | New-geometry moment-contribution ESS | Largest contribution |
|---|---:|---:|---:|---:|
| Original bank r02/r03 | 0.418 | 0.195 | 5.85 | 35.3% |
| Original protected r02/r03 | 0.548 | 0.320 | 5.06 | 32.7% |

Lower second moment is favorable. Neither source arm passes the declared
recommendation checks: a twofold target improvement, improvement over allocation
alone, moment-contribution ESS≥20 and largest contribution≤20%. The latter two
are diagnostic support checks, not guarantees of hidden-mode coverage.
Whole competing-contact second moments worsen by 6.2% and 9.7%; native-region
costs approach the 11.1% retention bound. The completed pilot also favors the
allocation-only control, but its moment contributions are too concentrated to
predict a reliable speedup.

Three focused tests cover joint-contact eligibility, the exact complete mixture
density inside and outside R4, and the geometry-preserving allocation control.
Across 325 arbitrary/anchor probes, independent SciPy normalized densities agree
within 2.18e−11 in log density and the existing independent auditor within
1.82e−12. All covariances are positive definite. Candidate-specific Rust
execution and new physical reference sampling have **not** been performed.

An initial preparation reused a validation-margin field measured against the
older 0.8 bound. The corrected v2 preparation measures that metadata against
the actual 0.9 bound. Both initial artifacts remain archived; the candidate
and allocation densities, anchors and covariances are unchanged. This revision
only reevaluated saved data and did not repeat physical or classifier work.

## Minimum next design, without launching

The [inert prospective plan](../runs/cooperative-contact-guide-preparation-v2-20261001/prospective-plan.json)
contains exact raw commands and distinct reserved seeds for **four populations
of 16,384 attempts per arm**, baseline92 versus candidate96: 131,072 attempts,
two clouds, intensity ratio128, at most two physical workers within the
eight-job/32-worker cap. It explicitly records `run_recommendation: false`.
This is the smallest matched feasibility allocation considered, **not a
recommended launch** or a promise of convergence. It requires new candidate
Rust reference receipts and a bound, failure-draining dispatcher before any
future use. There is no automatic retry, enlargement or substitution of the
allocation-only arm after seeing these results.

The original physical gates remain unchanged: population relative error≤10%,
importance ESS≥200, maximum weight≤2%, comparisons within three combined SE
and 0.2 kBT, a ≤0.5 kBT 95% half-width, and stable material strata. Matching SMC,
population-size/cloud-intensity controls and the vessel remainder still need
resolution. No small guide pilot can open the assembly gate by itself.

A better next direction is a tractable conditional proposal for the joint
near-core geometry, or explicit stratified integration with a separately
measured complement. Along a fixed orientation and translation ray, sphere-pair
overlap predicates give quadratic intervals; the intersection of two
scaffold-contact unions minus all core-overlap intervals could be sampled with
an explicitly normalized radial law and empty-ray fallback. Its complete
orientation/ray-selection density must remain tractable, and bounding trees
would be needed to avoid enumerating every atom pair. It may still miss angular
registration. This is a design direction, not an implemented or validated
sampler. Merely adding more Gaussian components or multiplying sample counts
is not supported by the current held-out moment diagnostics.

Reproduce preparation and inspection into fresh directories; this starts no
physical jobs:

```bash
cd /home/xvg/tetramer-mc
/home/xvg/protein-nucleation/.venv/bin/python tools/test_cooperative_contact_guide.py
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_cooperative_contact_guide.py \
  --out /path/to/new-cooperative-guide-preparation \
  --inspection-out /path/to/new-cooperative-geometry-inspection
```

- [Frozen candidate](../runs/cooperative-contact-guide-preparation-v2-20261001/candidate96.json)
- [Frozen allocation control](../runs/cooperative-contact-guide-preparation-v2-20261001/allocation-control92.json)
- [Full retrospective scoring](../runs/cooperative-contact-guide-preparation-v2-20261001/retrospective-analysis.json)
- [Anchors, neighbors and source provenance](../runs/cooperative-contact-guide-preparation-v2-20261001/fit.json)
- [Preparation source](../tools/prepare_cooperative_contact_guide.py)
