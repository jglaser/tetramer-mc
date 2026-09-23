# Independent validation of the protected contact guide

The completed fresh pilot improved sampling of native pockets but did not pass
the original convergence checks. Rare contributions in the contact-without-entry
region and disagreement between subdivisions still prevent a physical conclusion.
The next calculation tests a frozen reallocation of the existing Gaussian
components, before introducing a nonlinear transfer map.

The guide keeps the 84 means and covariances of the previous SMC-informed guide.
Only their normalized weights change, using the completed exploratory
`protected_minimax` fit. A 50% uniform component remains. Previously inspected
holdouts are design evidence, not independent validation. The new populations
are independent of that design and are not pooled with earlier campaigns.

See [the tail-allocation diagnostic](contact-tail-allocation.md),
[the previous pilot](smc-guide-pilot-results.md), and
[the separate nonlinear transfer prototype](kernel-transfer-prototype.md).

## Frozen allocation

| Arm | Components | Populations × attempted draws | Auxiliary intensity / activity |
|---|---:|---:|---:|
| `bank` | 80 | 4 × 1,048,576 | 128 |
| `protected` | 84 | 4 × 1,048,576 | 128 |
| `small` | 84 | 4 × 262,144 | 128 |
| `intensity256` | 84 | 4 × 262,144 | 256 |

The total is **10,485,760 attempted draws**, with two independent Poisson clouds
per valid pose. All arms use the repaired rigid tetramer, radius 1.5 Å, activity
0.035 Å⁻³, unchanged scaffold, R4 region, physical measure and complete native
classifier. Auxiliary intensity changes estimator variance, not physical activity.

The four mandatory comparisons are `bank/protected`, `protected/small`,
`protected/intensity256`, and the matched-size `small/intensity256`. Standalone
quality and free-energy precision of the small arm are diagnostic; they remain
decision criteria for the other three arms. Every pair comparison is mandatory.

All original convergence thresholds and subdivisions are retained: population
relative error ≤10%, importance ESS ≥200, largest contribution ≤2%, both 0.2
log units and three combined population standard errors for agreement, and a
population-based 95% free-energy interval half-width ≤0.5 kBT. Analysis covers
the original radial and angular bins and all 64 orthants, not only the 21 groups
used to fit the proposal. No new Pareto-improvement or second-moment gate is added.

## Estimator and validation

Each proposal is evaluated under its complete, untruncated mixture density.
The estimator remains `H_R4 H_capture H_hard J W / q`. Exterior and hard-invalid
draws retain zero contributions in the unconditional denominator. Empty or
unobserved regions are not silently removed. The full classifier is unchanged.

Before production, 131,072 fresh proposal-only draws checked the old and protected
guides. All eight population means and both arm means passed a simultaneous
Hoeffding bound for `E[H_R4/(V_R4 q)] = 1`; exterior zeros were retained. Arm means
were 1.008419 and 1.002806, against a simultaneous arm tolerance of 0.022650.
Independent density implementations differed by at most 1.14e-13. These tests
check the proposal, not the unknown protein partition functions.

The exact previously verified Rust executable and its analytic hard-only/finite
depletion sphere reference are reused. Source bundles, Python dependency closure,
input and classifier hashes, references, and prior declared seeds are archived.
The new preparation, controller, analyzer and workflow have focused tests for
normalization, frozen allocations, intensity controls, all denominators, tamper
rejection, worker limits and failure draining. Completed older physical populations,
audits and Lean checks are not rerun.

## Execution and review

The controller permits at most eight simultaneous physical jobs and checks the
shared 32-worker budget. Audits follow physical completion. First-pass classification
then uses at most 16 single-threaded workers, reserving room for parent management
threads within the same overall budget. BLAS, OpenMP and Rayon thread counts are
one. An interrupted or failed stage drains started work and records the failure;
the workflow does not retry or overwrite populations.

The archived workflow runs from
`runs/protected-guide-validation-20260923/common/run_protected_guide_workflow.py`,
with the absolute campaign path and expected protocol SHA256. Its status and
launch receipt are stored in that campaign. The controller refuses an existing
run, so the launch command must not be repeated.

Observed previous costs suggest roughly 52–61 CPU-hours of physical sampling,
about an overnight wall-clock run with postprocessing, and approximately 11 GB
of new records. These are estimates, not stopping rules. There is no adaptive
allocation, early success stopping or automatic assembly launch.

A regional diagnostic pass still requires review against earlier evidence and
independent coverage of important remaining regions. Full-vessel weights and
finite-system assembly remain separate requirements. Failure means unresolved
sampling; it is not evidence that native assembly is thermodynamically impossible.
