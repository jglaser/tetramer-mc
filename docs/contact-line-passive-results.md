# Exact line conditioning passed correctness checks but did not earn a physical pilot

**Do not launch a physical contact-weight campaign for this fixed-x variant.**
The frozen passive allocation completed without retries: four 64-draw streams
per arm, 156 critical-pose queries and 128 breadth queries, all independently
audited. It generated **zero Poisson clouds** and took 9.96 seconds wall time.
The original physical target, failed convergence gates and unresolved
finite-system assembly verdict are unchanged.

![Passive contact-line results](../runs/contact-line-passive-review-20261001/contact-line-passive.png)

The guide conditions one raw translation coordinate while retaining the other
five-coordinate Gaussian marginal. It preserves 50% uniform R4 support and
mixes conditioning with the original Gaussian draw. Empty geometry or a
component mass at or below 1e−12 uses the original conditional Normal.
[The frozen preparation](contact-line-passive-preparation.md) describes the
complete normalized density and the historically inspected versus fresh data.

## Retention and cost

| Quantity | Baseline92 | Line92 |
|---|---:|---:|
| Fresh attempts | 256 | 256 |
| Hard-valid, inside R4 and capture | 61 | 53 |
| Both scaffold gaps ≤0.02 Å, also hard-valid/in-domain | 0 | 2 |
| Both gaps ≤0.1 Å, also hard-valid/in-domain | 15 | 15 |
| Both gaps ≤0.5 Å, also hard-valid/in-domain | 43 | 34 |
| Fresh proposal-only CPU per attempt | 1.255 ms | 4.330 ms |
| Both gaps ≤0.5 Å per fresh CPU second | 133.8 | 30.7 |

For the 0.5 Å criterion, the fractions are 16.80% ±2.05% and 13.28% ±3.03%,
where ± denotes the observed standard error across four populations. The
difference is approximately one combined SE; this small pilot shows no gain
in broad contacts, rather than establishing a precise loss per attempted draw.
Two tight contacts versus zero are too few to establish a useful speedup; zero
observations do not establish zero baseline probability.

The **3.45× cost ratio is for this proposal-only audit**, including complete
density and endpoint geometry checks. It is not a measured physical-simulation
slowdown. The earlier 92-component protein/depletion sampler averaged
**17.92 ms per attempt**, including its Poisson work. That workload was not
rerun here. The report separately preserves draw/density CPU, startup overhead,
archived-query CPU and independent-auditor CPU; no physical speedup follows by
substituting the passive timing ratio.

## Why most conditional branches did nothing

Of 65 selected conditional branches, **55 fell back**:

| Cause | 0.02 Å | 0.1 Å | 0.5 Å | Total |
|---|---:|---:|---:|---:|
| No positive hard-free line inside R4/capture | 10 | 12 | 12 | 34 |
| Hard-free line exists, but no simultaneous contact interval at selected width | 16 | 4 | 1 | 21 |
| Positive interval below conditional-mass floor | 0 | 0 | 0 | 0 |
| Successful conditional draw | 1 | 2 | 7 | 10 |

Thus the immediate limitation is geometric coverage of the retained outer
five coordinates, rather than the chosen numerical mass floor. In 34 cases,
moving along the entire allowed x-line cannot escape hard overlap. Changing
the one-dimensional radial law cannot repair those outer configurations.
This motivates a future proposal that changes additional translation/rotation
coordinates or integrates explicitly over contact strata. It does not prove
that every such extension would work, or that the physical model cannot assemble.

## Weight diagnostics and coverage

For the 78 archived competing-55 poses, the observed paired second-moment
ratio is **1.232** using the original 84-component source and **0.727** using
the expanded 92-component source. The conditioned contributions have ESS only
**1.61 and 2.75**, with largest terms 77.1% and 45.8%. These source-dependent,
concentrated estimates cannot predict a reliable physical-weight improvement.
They are reused-cloud diagnostics, not new normalizers.

The separate breadth control selected 32 rows per class uniformly from the
original held-out populations. All 96 previously valid native-R5,
native-complement and competing rows remained valid; all 32 invalid rows
remained invalid. Median log-density changes were +0.171, +0.187 and 0.000 for
the three valid classes, respectively. The minimum observed density ratio
respected the declared 0.5 baseline-retention bound. These finite samples do
not bound missing contact regions or establish equilibrium coverage.

## Correctness receipts

All nine output populations/probe jobs passed the independent Python audit.
Maximum absolute errors were **2.49e−14** in full log density,
**5.33e−15** in baseline log density, **0** in the recorded log Jacobian,
and **1.23e−15** in selected inverse-CDF probability. The audit reconstructs
every Gaussian component's conditional mass and fallback, and validates physical
pose, support and recorded geometry predicates. Separate reference tests cover
interval completeness, Normal integration, normalized sampling and sphere
hard/depletion limits. Floating-point execution remains an implementation
obligation, not a formal exact-arithmetic guarantee.

- [Completed comparison](../runs/contact-line-passive-20261001/analysis.json)
- [All job and audit receipts](../runs/contact-line-passive-20261001/status.json)
- [Bound review and timings](../runs/contact-line-passive-review-20261001/summary.json)
- [Frozen source/input protocol](../runs/contact-line-passive-20261001/protocol.json)
- [Vector figure](../runs/contact-line-passive-review-20261001/contact-line-passive.svg)

No further physical job, enlarged allocation or proposal retuning was launched.

A separate [posthoc three-axis replay](../results/contact-line-axes-20261001/report.md)
subsequently inspected 462 unchanged poses, with zero draws or clouds. On the
139 saved Gaussian draws, uniform axis selection changes conditional availability
from 25.66% to 26.06%, while full density queries cost approximately three times
as much on matching probes. The union of useful directions is larger, but its
size is not the success probability of uniformly selecting an axis. This does
not justify an all-axis physical campaign either. A
[two-distance translation construction](two-distance-contact-guide-design.md)
is documented as an unimplemented next candidate.
