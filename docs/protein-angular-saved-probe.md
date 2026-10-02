# Saved protein poses: angular versus translational conditioning

Eight saved protein queries completed and passed independent reconstruction.
They establish that the rotational conditioner evaluates the intended geometry
and density on these poses. They show mixed density changes at higher geometry
cost, and do **not** yet justify a physical angular-guide campaign.

The allocation selected the first two hard-valid poses by draw order in four
predeclared groups from the completed larger conditioned population r00:
competing-contact orthants 22 and 62, old-R5 native orthant 55, and the remaining
native part of orthant 55. These are purposive queries from one population,
not independent or representative samples. No importance-weight ranking,
new pose draws, Poisson clouds or native classifications were used.

The frozen 92 components, defensive probability 0.5 and full hard-free
conditioning are unchanged. qT averages the three raw translation-axis laws;
qR averages the three length-scaled Cayley-axis laws. q6=(qT+qR)/2. All densities
refer to the same six-dimensional whitened R4 chart measure; the physical
Jacobian remains in any later importance estimator.

| Group | Archived draw | qR/qT | q6/qT |
|---|---:|---:|---:|
| Competing 22 | 518 | 0.290 | 0.645 |
| Competing 22 | 1003 | 1.110 | 1.055 |
| Competing 62 | 4006 | 0.983 | 0.992 |
| Competing 62 | 5233 | 1.527 | 1.263 |
| Old native 55 | 584 | 1.093 | 1.047 |
| Old native 55 | 599 | 0.239 | 0.619 |
| Remaining native 55 | 16 | 0.686 | 0.843 |
| Remaining native 55 | 17 | 1.356 | 1.178 |

The mixture raises density at four queried poses and lowers it at four. Its
largest increase here is 26.3%. These ratios are not native-generation
probabilities, physical weights, contact transitions or sampling speedups.
An average over these selected rows would not estimate any of those quantities.

Measured Rust geometry CPU over the eight queries was 0.04784 s for the
translation axes and an additional 0.10524 s for the angular axes. Evaluating
all six axes therefore cost **3.20 times the translation geometry alone**;
complete six-axis density evaluation took 0.16562 s. This excludes physical
Poisson sampling and cannot be interpreted as a physical sampler timing ratio.
The independent Python geometry/density audit took 2,048.7 CPU seconds.

The largest independent discrepancies were 1.74e-12 in complete log density,
2.58e-13 angstrom in interval endpoints and zero at recorded precision in the
log physical Jacobian. The reconstructed translation-only density also agrees
with the archived proposal to 8.88e-16 in log units. These bounded reference
checks do not certify arbitrary floating-point geometry or unseen contact modes.

Artifacts are under
`results/protein-angular-probe-preparation-20261002`:

- [Allocation](../results/protein-angular-probe-preparation-20261002/allocation.json),
  SHA-256 `e2cb8edcf3b5aeac7a5f7c3424aec94f8ffe6ba80f542e593b3b5dec0d317e8f`.
- [Independent audit](../results/protein-angular-probe-preparation-20261002/independent-audit.json),
  SHA-256 `b2e1ad9a914965271a79b5eab528cb75f73a4de577d14ed59a26bbb07cf49d35`.
- [Complete density/cost table](../results/protein-angular-probe-preparation-20261002/density-comparison.json)
  and [terminal receipt](../results/protein-angular-probe-preparation-20261002/independent-audit-execution-receipt.json).

The audit terminated successfully after all eight queries; there were no
retries. Earlier preparer corrections remain archived. The production
normalizers continue to reject the new angular schema, and the running SMC
control retains its original translation guide. A future extension needs a
separately frozen generation/physical-efficiency control; this diagnostic
neither launches one nor changes any convergence or assembly verdict.
