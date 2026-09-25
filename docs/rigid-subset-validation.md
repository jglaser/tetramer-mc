# Rigid-subset phase: validation and first protein pilot

Implemented on 2026-09-25. The optional fixed-duration phase is a working production kernel, with spherical boundaries, local rigid moves or frozen posterior transport, exact many-body ideal depletion, per-event assembly-bias correction, named RNG streams and deterministic continuation. Selection uses unique internally connected dimers/trimers, including subsets of a larger aggregate. No native label enters eligibility. The supplied coverage atlas remains native-informed.

[Construction and figure](rigid-subset-phase.md) · [Portable example](../examples/spherical-cluster-phase.json) · [Run command](../examples/README.md#experimental-dimertrimer-transport-phase)

## Correctness checks

Fourteen new Rust integration tests pass:

| Suite | Tests | Evidence |
|---|---:|---|
| `tests/rigid_subset.rs` | 6 | Rigid inverse and internal geometry; invalid labels and hard/wall guards; zero bath/isolated/identity limits; conservative disjoint cover; analytic two-sphere count/accepted-flow check; independent axial three-body union integral, detecting a pairwise surrogate |
| `tests/cluster_phase_clock.rs` | 4 | Exact two-state finite-time transition law; deliberately biased event-indexed control; zero-rate identity; unique connected subsets inside larger aggregates; incremental contact geometry |
| `tests/cluster_phase_runner.rs` | 3 | Full production learned-map/bias replay; exact checkpoint continuation; absent/null/zero-duration legacy compatibility; configuration guards |
| `tests/cluster_phase_stationarity.rs` | 1 | Independent equilibrium reference, two local-rigid initializations and a nonphysical anisotropic learned-proposal control |

Eleven relevant existing tests also pass: four assembly-bias runner tests, three conditioned-GCA runner tests, and four frozen-posterior assembly tests. Logs and source hashes are archived in `runs/cluster-phase-validation-20260925/`. The release executable builds offline with the pinned lockfile. The user-edited `examples/spherical-seeded-transport.json` is unchanged.

The phase records every attempted event, including proposal nulls, hard rejections, physical rejections and bias rejections, and consumes the sampled holding time for all of them. Output trajectories are written at sweep endpoints after the complete fixed-duration phase. Event-indexed move records are diagnostics, not unweighted equilibrium observations. An internal exclusion-graph mismatch under floating-point rigid transport causes a declared null event. This guard does not prove exact machine arithmetic reversibility.

## Independent physical reference

Three spheres have core radius 1, depletant radius 0.14, wall radius 3.5 and activity 5 in consistent units. Because the exclusion radius is below `2/sqrt(3)` times the core radius, no triple exclusion intersection is possible in a hard-valid state. The exact ideal-depletion weight is therefore computable from analytic pair lenses. This supplies an independent reference for contact, separation, cluster and rotational observables; the separate axial gate test exercises a substantial genuine three-body correction.

A reference calculation uses 250,000 independent uniform wall configurations, with 87,123 hard-valid samples and weighted ESS 82,673. Each Markov arm has 3,000 burn-in sweeps and 24,000 production sweeps in 48 blocks. Both single-body and subset physical decisions use the actual Poisson gate.

| Observable | Independent reference | Local-rigid dispersed start | Local-rigid aggregated start | Learned-map arm |
|---|---:|---:|---:|---:|
| Contact count | 0.5079 ± 0.0025 | 0.5189 ± 0.0068 | 0.5078 ± 0.0068 | 0.5016 ± 0.0060 |
| Largest cluster | 1.5028 ± 0.0024 | 1.5140 ± 0.0066 | 1.5028 ± 0.0066 | 1.4966 ± 0.0058 |
| Mean pair separation squared | 9.6046 ± 0.0088 | 9.6035 ± 0.0213 | 9.6486 ± 0.0230 | 9.6623 ± 0.0250 |
| Mean quaternion scalar squared | 0.2502 ± 0.0005 | 0.2493 ± 0.0016 | 0.2510 ± 0.0020 | 0.2514 ± 0.0016 |

Errors are reference standard errors and block standard errors, not guarantees of convergence. All declared reference and initialization comparisons pass. The learned arm uses unequal Gaussian weights, anisotropic covariances, translation/rotation correlations and distinct angular centers, with 70% transport slots and a 15% uniform floor within that proposal. It attempts 14,079 involutions with nonzero posterior correction and accepts 153 of them. Collective contact exchanges occur in all arms (67, 48 and 31), along with attachments and detachments. These observations validate sampling in this reference system; they do not establish a protein mixing advantage.

## Lean-to-implementation bridge

The complete project now audits **65 results** with pinned Lean 4.24.0 and mathlib commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. No `sorry` or new axioms are used; the axiom audit reports only `propext`, `Classical.choice` and `Quot.sound`.

- [`ClusterRates.lean`](../formal/ReversibleSampling/ClusterRates.lean): finite-state weighted flows, subset sums, uniformization, stochastic powers, normalized count/Poisson mixtures, composition and event-chain rate bias.
- [`ClusterMeasureRates.lean`](../formal/ReversibleSampling/ClusterMeasureRates.lean): general measurable-state stationary-flow symmetry, bounded invariant rate weighting (including singular kernels), finite-subset uniformization, kernel iteration and fixed-duration Poisson-mixture invariance.
- Existing accepted-flow, involution and conditional-Poisson count theorems are retained.

These are conditional mathematical guarantees: each fixed-subset kernel must obey balance, rates must be bounded/measurable and unchanged almost everywhere along its transitions, and the stated measure assumptions must hold. Equivalence of the optimized exponential-event implementation to Poisson uniformization is explained mathematically and tested against a finite-state reference, but is not formalized in Lean. Sphere-tree predicates, the single-handle coordinate/Jacobian implementation, random-number generation, Poisson thinning and floating-point execution remain explicit implementation obligations. See [`formal/validation.json`](../formal/validation.json).

## Frozen stalled-protein pilot

Allocation was fixed before execution: two independent master seeds per arm, 12 sweeps each, control versus added cluster phase. The two arms share the same starting configuration and paired master seeds. They start from the frozen sweep-27,000 configuration of the 264-tetramer run at 500 μM, depletant radius 1.4 Å, activity 0.0275 Å⁻³, Poisson intensity ratio 64. The eight original seed labels and full environment are retained. Live jobs were not changed. At most two pilot processes ran concurrently.

The added phase uses duration 0.01, dimer rate 1, trimer rate 0.25, local/transport probability 1/2, correlation 0.9, and the existing 0.2 Å / 1° local scales. It does not refit the atlas.

| New cluster branch, pooled across two streams | Attempted | Hard valid | Accepted | Completed contact exchanges |
|---|---:|---:|---:|---:|
| Local rigid | 47 | 4 | 4 | 0 |
| Posterior transport | 41 | 8 | 0 | 0 |
| Uniform defensive | 6 | 1 | 0 | 0 |
| Total | 94 | 13 | 4 | 0 |

The 94 events contain 67 dimer and 27 trimer attempts. Accepted cluster moves preserve the external contact graph in this pilot. Both controls also accept zero ordinary global moves; ordinary local accepted counts are 3 and 4 in both respective arms. The new phase costs 2.01 CPU seconds pooled; total CPU is 59.26 seconds with it versus 58.00 without it. These tiny paired timings are integration diagnostics, not a speedup estimate. Most new proposals fail hard geometry, and all nine hard-valid learned/uniform proposals are rejected by their remaining physical/proposal gate.

Full fixed allocation, input/source/executable hashes, commands, logs, complete trajectories, move records and analysis are in `runs/cluster-phase-protein-pilot-20260925/`. All four runs complete successfully. No contact-exchange improvement or protein assembly conclusion follows from 94 attempts. Longer comparisons should report contact-fingerprint ESS per CPU, completed environment exchanges and initial-condition agreement, retaining the fixed physical conditions and exact clock accounting.

## Passive selection diagnostic (2026-09-25)

The first 600 sweeps of the ongoing 264-tetramer, 500 μM growth run were frozen
and replayed without changing the simulation. Its bath is 1.4 Å / 0.0275 Å⁻³.
The replay distinguishes whole oligomers from embedded subsets and audits all
attempts and saved states. See the [diagnostic report](../runs/cluster-selection-diagnostic-20260925/report.md)
and [figure](../runs/cluster-selection-diagnostic-20260925/cluster-selection-diagnostic.png).
This is a descriptive one-stream growth comparison, not a mixing-time or
equilibrium test. Earlier physical stationarity checks above were reused.

Three new tests in `tests/cluster_subset_context.rs` validate graph context,
reject malformed subsets, and compare logging on/off with identical physical
trajectories, counts and all five random streams. All three pass. The three
existing `cluster_phase_runner` tests also pass after the metadata addition,
including checkpoint continuation and complete replay. The exact command and
output are in `runs/cluster-selection-diagnostic-20260925/metadata-tests.log`.
The standalone replay has separate independent graph-classification checks.
