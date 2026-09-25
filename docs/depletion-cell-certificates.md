# Reusing envelope cell classifications in the depletion point loop

Implemented on 2026-09-24. The production cached gate now retains the body,
old-environment, and new-environment coverage classifications for each retained
envelope cell. A known label supplies the point-containment result directly;
only `Unknown` invokes the corresponding BVH query. The first matched protein
benchmark shows a small additional benefit: about 1% less complete-gate CPU
than the previous body-geometry cache.

## Correctness and scope

A cell's circumscribed ball covers its sampled positions. The existing
`classify_ball` predicates return `Inside` or `Outside` only with their
conservative geometric guards. Rigid transforms normalize ingress quaternions,
and environment classification includes a coordinate-dependent guard before
inverse transformation. These remain ordinary FP64 guards, not a formal
interval-arithmetic proof.

Certificates are private and live only with the envelope for the current
endpoints and environment. They are recomputed every call, including when the
persistent body cache is warm. Every retained-cell path appends its own three
classifications; there is no parent-to-child inheritance. If cumulative-volume
arithmetic triggers the reference root-envelope fallback, certificates become
`Unknown` and point predicates run normally.

The envelope, cell-selection draws, three coordinate draws per point,
conditional gained-point thinning draws, accumulated counts, and acceptance
weight retain their original order. Even fully classified cells consume the
original coordinate draws. This implements query bypass only: there is no
whole-gate early acceptance/rejection and no Poisson-count aggregation.

`BodyEnvelopeCache::sample` uses the new path automatically.
`sample_unclassified` retains the previous cached gate for comparison;
`depletion::sample` remains the uncached oracle. `sample_profiled` returns
`ContainmentQueries` counters. Compile-time specialization removes those
counters from ordinary production calls. Counters represent full body or
environment containment calls, not internal BVH node visits.

## Fixed-endpoint measurements

The same frozen snapshots, endpoint-generator seeds, proposal widths, traversal
budgets, and five repetitions from the body-cache benchmark were reused. Endpoint
hashes match all three prior reports. The current benchmark runs three arms with
cycled order: uncached reference, previous cached gate, and cached gate with
certificates. Each arm uses the same Poisson seed. A separate profiled cloud,
outside the timing loops, measures skipped queries and is independently checked
against the reference. No assembly state evolves.

| Frozen snapshot | Eligible endpoints | Avoided point containment queries | Previous cached gate CPU (s) | With certificates CPU (s) | CPU reduction |
|---|---:|---:|---:|---:|---:|
| Native-contact N=12, sweep 2,000 | 3 | 3,184 / 151,526 (2.10%) | 0.4435 | 0.4368 | 1.51% |
| Aggregate N=48, sweep 29,300 | 18 | 14,197 / 1,031,135 (1.38%) | 2.9800 | 2.9518 | 0.95% |
| Seeded N=264, sweep 5,300 | 17 | 10,010 / 597,549 (1.68%) | 2.0008 | 1.9787 | 1.10% |

Baths are unchanged: N=12 uses radius 1.5 A/activity 0.035 A^-3; N=48 uses
1.4/0.025; N=264 uses 1.4/0.0275. The eligible endpoints come from the first
12, 24, and 32 body labels, with up to 128 hard-validity attempts per label;
failed attempts remain in the reports. These are conditional local-endpoint
microbenchmarks, not physical size comparisons or assembly-efficiency estimates.

All 190 timed gate evaluations per arm reproduce both other arms and the prior
executable's archived gate results and RNG continuations. That covers 4,718,531
raw Poisson points per timed arm. All 38 separate profiling comparisons also
match. No case was omitted due to the expected-point budget.

Most retained cells still have uncertain membership at the production traversal
budget, so 97.9–98.6% of point queries remain. The incremental timing difference
is small and comes from one benchmark batch, rather than an established
end-to-end speedup. Together with body caching, full gates remain about
1.51–1.57x faster than the uncached reference on these workloads. This shortcut
does not address proposal quality or change mixing per attempted move.

## Validation and reproduction

40 targeted tests passed: seven new certificate tests, eight cache tests, and
25 production integration/restart tests. Tests compare all gate fields and
subsequent RNG words, exercise body and both endpoint bypasses, and cover
spheres, dumbbells, the repaired tetramer, rotations, periodic images, caps of
1/3/17 cached nodes, hard-only/empty/identity paths, tangencies, cell corners and
faces, adjacent floating-point values, and global coordinates shifted by 1e8.
Production tests replay the gate independently and verify cold-cache restarts.

```bash
cargo build --locked --release --bin tetramer-mc --bin envelope-cache-benchmark
cargo test --locked --offline --test depletion_certificates --test depletion_cache
cargo test --locked --offline \
  --test frozen_posterior_assembly --test periodic_posterior_assembly \
  --test runner_restart --test bath_overrides --test seeded_free_tetramers \
  -- --test-threads=2

target/release/envelope-cache-benchmark \
  --config runs/body-envelope-cache-benchmark-20260924/aggregate48-config.json \
  --out runs/cell-certificates-repeat.json --cases 24 --repeats 5
```

Local artifacts: `runs/body-envelope-cache-benchmark-20260924/` contains
`*-certificates-result.json` (schema v2) and `certificate-comparison.json`.
Reports include all proposed endpoints, counts, timings, skipped-query
denominators, and input/source/executable hashes. Query profiling is separated
from timed production code. Reproduction requires a fresh output filename.

New runs and checkpoint continuations use the rebuilt executable. Already
running processes continue with their existing code and RNG streams.
