# Cached body geometry for production depletion gates

The production single-body gate now reuses a lazy body-frame envelope tree.
The ordinary local, independent-redraw, and correlated-transport branches use it
without an input-format or command-line change. The frozen proposal, bath,
endpoint envelope, and Poisson thinning algorithm are unchanged.

## What is reused, and why the trajectory is preserved

The rigid particle and depletant radius determine the root AABB and every
longest-axis dyadic subdivision. Each cached node stores its cell, center,
circumscribed radius, width, and the coverage classification against the moving
body. Those calculations previously repeated for every proposed endpoint.
Spectator coverage at the old and new poses is still evaluated on every move.

The original breadth-first traversal order, per-query created-cell budget,
termination tests, retained-cell order, and floating-point volume sums remain
identical. Point generation and conditional thinning draws retain their
original order. The cache borrows the immutable shape and binds the exact
depletant radius; foreign shapes or radii are rejected. The subsequent
[cell-certificate shortcut](depletion-cell-certificates.md) additionally reuses
those classifications within the point loop and adds about 1% in the first
matched benchmark. The timings below describe the original body-cache stage.

Only visited subdivisions are stored. The default cap is 65,535 nodes;
exhausting it uses uncached traversal and does not change the envelope budget.
Default depth 14 permits at most 32,767 nodes. Cache contents are performance
state: reconstruction on checkpoint continuation consumes no random numbers.
A different bath radius constructs a new cache at startup.

The independent `Envelope::build` and `depletion::sample` implementations remain
available as regression oracles and for other calculation paths. This change
does not optimize spherical GCA, hard-overlap checks, the environment scan,
contact-weight calculations, or proposal-density evaluation.

## Measured results, 2026-09-24

These are single-thread, release-build **fixed-endpoint microbenchmarks**, not
assembly trajectories or an independent-sample efficiency measurement. Frozen
checkpoint poses supply the neighborhoods. Synthetic local displacements use
each configuration's production translation and rotation widths. Each tested
body gets at most 128 attempts to find a hard-valid endpoint; every attempt is
recorded. Successful endpoints receive five repeated timings with common random
seeds and alternating implementation order. No physical configuration evolves.

| Saved configuration | Radius / activity (A / A^-3) | Eligible endpoints / tested bodies | Envelope speedup | Complete ordinary gate speedup |
|---|---|---:|---:|---:|
| Native-contact N=12, sweep 2,000 | 1.5 / 0.035 | 3 / 12 | 1.87x | 1.50x |
| Aggregated N=48, sweep 29,300 | 1.4 / 0.025 | 18 / 24 | 1.90x | 1.52x |
| Seeded N=264, sweep 5,300 | 1.4 / 0.0275 | 17 / 32 | 1.88x | 1.57x |

All eligible endpoints had nonempty envelopes. No gate was skipped for the
point budget. Across 190 complete gate evaluations per implementation,
4,718,531 raw Poisson points per implementation produced bitwise-identical
counts, weights, envelopes, and subsequent RNG outputs. Warm caches held
3,995, 4,781 and 5,985 nodes, respectively. Initialization cost about 1.7 ms.
On first visits while growing a shared cache, envelope times were 0.092,
0.373 and 0.292 CPU seconds versus reference times of 0.198, 0.655 and 0.481.
The speedup table uses the repeated warm timings; it does not hide first-use
costs, which are retained separately in each report.

The tested labels are the first 12, 24 and 32 bodies, respectively, rather than
a random sample of all bodies. Hard-valid conditioning measures gates that can
actually execute; it does not estimate proposal acceptance. Failed attempts
and bodies with no valid endpoint remain in the reports. The N=264 preparation
contains eight seed bodies plus 256 free bodies. Bath conditions differ across
these snapshots, so the table is not a physical finite-size comparison.

About 1.5x faster gate evaluation is a useful CPU improvement. Whole-run speedup
also depends on the fraction spent in GCA, geometry, proposals and output; it
has not been measured here. The optimization changes neither pose accessibility
nor mixing per attempted move. Running processes retain their existing binary;
new runs and checkpoint continuations use the rebuilt executable.

## Reproduction and validation

Benchmark binary: `src/bin/envelope-cache-benchmark.rs`.
Frozen inputs, checkpoint/config hashes, all proposed endpoints, CPU timings,
source and executable hashes, and equality checks are under
`runs/body-envelope-cache-benchmark-20260924/`. Those run artifacts are local;
the benchmark also works with a supplied portable configuration.

```bash
cargo build --locked --release --bin tetramer-mc --bin envelope-cache-benchmark

target/release/envelope-cache-benchmark \
  --config runs/body-envelope-cache-benchmark-20260924/aggregate48-config.json \
  --out runs/body-envelope-cache-repeat.json --cases 24 --repeats 5
```

A dispersed configuration may give empty envelopes and is not a useful
contact-workload timing control. The benchmark reports this explicitly and
requires a fresh output path.

Eight cache tests passed, covering synthetic spheres/dumbbells, repaired
protein geometry, swapped endpoints, rotations, periodic crossings, traversal
budgets, caps of 1/3/17 nodes, invalid inputs, empty environments, identity moves,
zero activity, and RNG continuation. Another 25 production integration tests
passed, including uncached spectator/gate replay and cold-cache restarts:

```bash
cargo test --locked --offline --test depletion_cache
cargo test --locked --offline \
  --test frozen_posterior_assembly --test periodic_posterior_assembly \
  --test runner_restart --test bath_overrides --test seeded_free_tetramers \
  -- --test-threads=2
```

The cached and reference traversals should be kept together in future reviews;
changes to splitting arithmetic or predicate semantics require updating and
rerunning the bitwise equivalence tests.
