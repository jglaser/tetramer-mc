# Saved circles contain missed hard-free arcs, but a sampler is not established

The independent Rust/Python comparison agrees for **all 240 saved conditional
circles**, with no native-label filtering, new pose draws or Poisson clouds.
There is useful geometric support that the previous azimuth draws missed:
**111 circles have positive hard-free arc length**, whereas only 25 of their
original endpoints were hard-valid. This motivates further proposal design;
it does not establish an improved normalized sampler, native yield, physical
weight or assembly stability.

## What the complete saved allocation contains

| Source proposal | Saved circles | Positive hard-free arc length | Original hard-valid endpoints | Mean hard-free probability under original azimuth law |
|---|---:|---:|---:|---:|
| Uniform azimuth | 118 | 54 | 4 | 2.09% |
| Localized azimuth | 122 | 57 | 21 | 15.74% |

The 21 localized original endpoints are the same valid poses in the completed
[passive comparison](contact-distance-passive-results.md), not additional
events. The remaining 65 localized circles have no positive hard-free arc.
The mean uniform-angle mass on the localized circles is only 2.83%; the
localized law already concentrates substantially toward their allowed angles.

Positive-circle counts by the originally selected width are:

| Gap width | Uniform azimuth | Localized azimuth |
|---|---:|---:|
| 0.02 Å | 12 / 47 | 12 / 41 |
| 0.1 Å | 19 / 38 | 22 / 44 |
| 0.5 Å | 23 / 33 | 23 / 37 |

Every original population remains represented: the positive-circle counts are
17/35, 9/23, 14/31 and 14/29 for uniform azimuth, and 12/33, 18/34, 12/27 and
15/28 for localized azimuth. These are descriptive diagnostics on the saved
conditional radii/orientations, not independent physical populations.

**R4 and capture restrictions were not applied along the arcs.** Their old
Boolean flags describe only each previously drawn endpoint. Therefore 57
nonempty localized circles are not 57 demonstrated in-domain or native poses.
Nor is 57/21 a sampling speedup. A proposed conditional law would still need
correct empty-circle handling, complete normalization and the density summed
over every contributing component/width label.

## Cost and independent checks

Rust's arc geometry took **0.549 CPU seconds for all 240 circles**, averaging
2.29 ms per circle (2.63 ms uniform-source, 1.96 ms localized-source). Total
Rust diagnostic CPU, including witnesses and output, was 0.940 seconds. The
independent Python geometry took 11.695 CPU seconds; total sequential execution
wall time was 13.917 seconds. These are geometry-reference costs, not timings
of a full corrected proposal or physical sampling kernel.

The Rust tree visited 6,161,684 node pairs, pruned 2,697,229 and tested 383,719
atomic pairs. Full coverage enabled 23 early exits. A future density evaluation
may need several such circles per pose, so the single-circle cost cannot be
substituted for complete proposal cost.

The comparator checked exact ordered identities and attempt journals, interval
counts and endpoint closures, allowed/forbidden complement partitions, all
saved original predicates and **2,622 direct-predicate witnesses**. No topology
or membership mismatch occurred. Maximum discrepancies were:

| Quantity | Maximum absolute difference |
|---|---:|
| Interval endpoint | 5.72e−13 rad |
| Allowed arc length | 5.95e−13 rad |
| Original azimuth-law probability mass | 5.99e−13 |
| Uniform azimuth probability mass | 9.46e−14 |

All are below the fixed **1e−10 absolute** comparison tolerance. The full
report preserves every case and discrepancy, including zero-length/empty
sets; no case was dropped for being outside the original domain. Floating-point
agreement is a reproducibility check, not a formal real-arithmetic guarantee.
Before execution, [seven Rust geometry tests](../results/circle-geometry-validation-20261001/final-test-output.log)
and [eleven independent Python reference tests](../results/contact-circle-reference-validation-20261001/validation.json)
passed, including strict tangencies, interval topology, analytic masses and
direct atomic witnesses. Each saved-protein geometry evaluation executed once
per implementation.

## Provenance and next gate

An initial controller failed before either geometry job started because a
scheduling field was absent. Its [failed status](../runs/contact-circle-feasibility-20261001/status.json)
is preserved. The corrected v2 controller evaluated the unchanged 240 cases
once per implementation; there was no repeated geometric evaluation or new
sampling.

The [independent comparison](../runs/contact-circle-feasibility-comparison-20261001/analysis.json)
binds 88 source/output files and has SHA256
`a5516f7d3929d34007767ff74aab662c9dac9915ef99366a01e9089c060097a2`.
The frozen cases SHA256 is
`dc4743a8dc06ea51f86e508cc1a3a631371a844ff2c50c28d3983b5c0236ca85`;
the [v2 execution protocol](../runs/contact-circle-feasibility-v2-20261001/protocol.json)
SHA256 is `b04da2995332408e1c9bf6f64347ef10c6db0175e39e1f93d057c8de93d4ebe7`.

[Comparison code](../tools/compare_contact_circle_feasibility.py) and
[five focused record checks](../tools/test_compare_contact_circle_feasibility.py)
perform only interval algebra and saved-output comparison. They do not rerun
atomic geometry or draw poses.

The appropriate next gate is a normalized conditional-azimuth proposal with
an independently checked complete density and bounded passive cost/retention
comparison. These saved-circle results alone do not justify a physical-weight
campaign or alter the unresolved finite-system assembly conclusion.

## Complete-density workload before a new sampler

A separate [read-only count of existing density traces](../results/contact-circle-full-density-workload-20261001/report.md)
finds that sharing circle geometry across identical atom-pair labels reduces
the maximum from 276 component/width branches to **43 distinct circles**.
It does not make density evaluation a one-circle calculation:

| Saved query class | Distinct eligible circles: median / mean / maximum |
|---|---:|
| Valid fresh localized draws | 8 / 15.48 / 43 |
| Critical competing-tail probes | 14 / 13.54 / 18 |
| Native R5 breadth probes | 43 / 41.63 / 43 |
| Native-complement breadth probes | 21.5 / 24.28 / 43 |
| Competing breadth probes | 0 / 2.16 / 12 |

These counts use already saved polygon/density support, without new geometry,
sampling or fitting. A hard-valid queried pose is itself a witness on each
eligible circle, so its full-density queries generally cannot use the fully
blocked-circle early exit. The 240-circle timing included blocked circles and
must not be multiplied by these counts to claim a measured full-proposal cost.
The next implementation should first score the complete corrected mixture on
the archived critical and breadth queries, with geometry shared across labels,
before spending a fresh proposal or physical-weight allocation.
