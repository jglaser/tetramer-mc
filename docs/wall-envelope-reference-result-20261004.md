# Completed wall-envelope sphere reference

The revised reference completed all **17 stages** and passed all **47 predeclared
checks**, with every child drained. Its **131,072 attempted draws** cover the full
atomic-wall target. The largest absolute log error among the 30 exact, Poisson
and hard regional estimates was **0.0220**, below the fixed 0.1 threshold.

![Regional masses divided by their analytic values, with one population standard error](../results/wall-envelope-completed-20261004/comparison.png)

| Reference | Four populations at z = 0 | Four populations at z = 0.4 | Total attempts | Largest absolute log error | Checks passed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier cube | 2,048 each | 2,048 each | 16,384 | 0.1647 | 44/47 |
| Wall envelope | 12,288 each | 20,480 each | 131,072 | 0.0220 | 47/47 |

The proposal and allocation changed together. This is a successful new fixed
reference, not an isolated speedup measurement or an extension of the earlier
failed allocation. The cube result remains failed: its three contact estimates
at z = 0.4 missed the point-error requirement, although their discrepancies
were within the separately reported statistical uncertainty.

At z = 0.4, the Poisson contact mass divided by its analytic value was
**0.99009 ± 0.01083**, compared with **1.17549 ± 0.19386** for the earlier cube.
The new outside-source ratio was **0.99986 ± 0.00424**. Errors here are one
standard error from four independent population means. Populations and regions
are not pooled; these bars are not simultaneous confidence intervals.

Both references use the same sphere fixture: core radius 0.3, depletant radius
0.5, exclusion radius 0.8 and wall radius 3. The valid center separation is
0.6–2.7, contact ends at 1.6, and source capture ends at 1.4. The five displayed
regions include the space outside the source guide. The complete physical
proposal density and unconditional attempted denominators were retained.

All ten Poisson-minus-exact residual checks passed. The deliberate extra-Jacobian
estimator failed full-target normalization at both activities, giving ratios
about 0.000123. Source censoring also failed the full-target check, with ratios
0.12982 and 0.14465, while correctly matching the inside-source target. These
negative controls passed their intended detection checks.

This establishes a bounded analytic implementation reference for the optional
wall-envelope proposal. It does not establish protein contact-weight convergence,
mixing efficiency or finite-system assembly. The [prospective design and
assumptions](wall-envelope-normalizer.md) remain the scope of the statistical
assurance.

The [plotted values](../results/wall-envelope-completed-20261004/plotted-values.json),
[SVG](../results/wall-envelope-completed-20261004/comparison.svg) and
[plot receipt](../results/wall-envelope-completed-20261004/receipt.json) preserve
the exact inputs and outputs. Rendering used one frozen execution and only
completed scalar summaries and terminal metadata, authenticated before and
afterward; no raw rows, geometry or sampling were read or generated.
The new analysis SHA256 is
`6164ede77f8a7456eabc38db69e0644f976d3eeb14caac79eae21a01ef1d8366`;
the earlier failed analysis is
`95f91bf96576bf69d2339fe2a5f5af6c8528c8c3569acaf3d4ab5f4876710c51`.
