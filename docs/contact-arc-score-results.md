# Hard-free arc conditioning: complete-density result

**The correction is numerically validated, but it does not justify a physical weight campaign.** It improves the earlier two-distance guides on the inspected critical contacts, while showing weak or no improvement over the original 92-component Gaussian guide. The physical contact-weight and finite-assembly questions remain unresolved.

The fixed calculation scored all 206 saved poses under both azimuth laws: 78 critical competing orthant 55 poses and 128 breadth poses, including all 32 invalid controls. It generated no new poses, depletant clouds, physical masses or assembly trajectories. These previously inspected data are retrospective diagnostics, not independent validation of sampling efficiency. The [preparation](contact-arc-score-preparation.md) preserves the original shape, scaffold, region and guide labels.

The bound physical configuration retains depletant radius 1.5 Å and activity
0.035 Å⁻³. This is separate from the running 1.4 Å growth simulations.

![Retrospective contact-score ratios, breadth coverage and full-mixture cost](../results/contact-arc-score-review-20261001/contact-arc-score.png)

## Critical contact scores

The table gives paired second-moment ratios relative to old92; smaller is favorable on these saved rows. The original sampling sources are kept separate, with their paired-cloud weights and unconditional attempted-draw denominators retained.

| Original source | Rows | Old distance, uniform φ | Old distance, localized φ | Arc, uniform φ | Arc, localized φ |
|---|---:|---:|---:|---:|---:|
| 84-component source | 4 | 1.875 | 1.593 | 1.559 | 1.486 |
| 92-component source | 74 | 1.936 | 1.208 | 1.093 | 0.976 |

For the localized arc law, contribution ESS is only **1.58 and 2.54**, and the largest terms contribute **78.7% and 58.5%**, respectively. The 2.4% reduction on the 74-row subset is therefore insufficient for a reliable scale-up prediction. These are contribution-concentration diagnostics for selected rows, not contact-sampling ESS, population uncertainty estimates or converged regional integrals.

## Breadth and computational cost

All four predeclared breadth classes retain their 32 poses. Median log-density changes relative to old92 are:

| Saved class | Uniform arc | Localized arc |
|---|---:|---:|
| Native R5 | −0.196 | +0.006 |
| Native complement | −0.291 | −0.073 |
| Competing | approximately 0 | approximately 0 |
| Invalid | −0.265 | −0.00005 |

The defensive bound $q\geq0.5q_{\rm old92}$ holds for every scored pose. Higher density at a saved pose is a proposal property; these values are not equilibrium occupancies or proof of coverage elsewhere.

The complete mixtures required **3,611 distinct circle geometries**, averaging 17.5 per query after sharing geometry across components, widths and arms. Joint full-density scoring took **8.401 CPU seconds**, including **8.117 seconds** of geometry: **40.78 ms per saved pose** for both laws together. A single-circle feasibility calculation therefore understates the cost of evaluating the complete mixture.

Uniform φ was evaluated first and paid the shared geometry cost; localized φ reused it. Their raw timers cannot be interpreted as separate sampler speed comparisons. The measured nongeometry costs were 0.150 and 0.134 seconds. The independently reconstructed Python calculation used 319.48 child CPU seconds; it is a validation expense, not production-kernel timing. No fresh-draw throughput or effective samples per CPU was measured.

## Validation and decision

Independent Rust/Python reconstruction covered all **412 full densities**, **113,712 component-width branches** and **3,611 circle geometries**. Every fallback decision, endpoint predicate and zero-density branch agreed. Maximum full log-density discrepancy was $2.24\times10^{-12}$, arc-mass discrepancy $2.17\times10^{-12}$, and interval-endpoint discrepancy $1.53\times10^{-12}$ radians. Rust also reproduced the archived old92/distance densities and physical Jacobians exactly. The reference suite had 6 Rust tests, 13 Python tests and 60 synthetic cross-language queries; 8 preparation/controller checks passed before execution.

No physical campaign is ready with this variant. A more promising next construction is to retain the original Gaussian conditional translation measure and condition only on the physical hard-free domain, rather than replace its radial law with a narrow selected-contact law. That is a proposed direction, not a validated algorithmic improvement. The present result gives no evidence that the physical model prevents native assembly.

## A cheaper auxiliary-index alternative

The [explicit-index construction](contact-arc-explicit-index-design.md) retains
one Gaussian-component index and uses an exactly normalized auxiliary target.
Its importance estimator is unbiased, with a per-draw variance bound relative
to the old **two-distance** guide. This bound is not relative to old92 and does
not establish faster Markov mixing.

A [passive reconstruction of all 206 rows](../results/contact-arc-index-retrospective-20261001/report.md)
finds localized-index second-moment penalties of only 1.007 and 1.018 relative to
the full arc mixture on the separate 4/74-row critical subsets. Thus it retains
most of that correction, but does not repair its weak performance relative to
old92. The archived-query geometry accounting falls from 17.53 full circles to
0.488 prior-weighted circles per query, or approximately 0.71 when weighted by
the old component responsibilities. These are cost proxies on fixed poses,
not measured sampler speedups. The construction is retained as a possible
building block; it does not yet warrant another physical allocation.

The completed execution is `runs/contact-arc-score-20261001`; the controller ran once and both jobs plus the independent audit completed. Protocol SHA-256: `2a05ad8a092515943482d9cbc56f8f890e16707c87af3f42b43801446d73d94b`. Analysis SHA-256: `85830269088161b570a1652fb8e47a42e6aceb1647691ecbebd3b90dc1a3bf8e`. Independent audit SHA-256: `e02e504446aee94165e0ca85cf6faa857acf60fcca4c6b39ae0308e47402ff20`. The score journal preserves all 206 attempted query identities.
