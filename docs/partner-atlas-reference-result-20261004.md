# Partner-atlas one-step reference: completed result

All **292 frozen checks passed** for the four optional partner-atlas arms,
and the independent saved-arithmetic audit passed. This is evidence that the
implemented corrected moves preserve the tested physical source observables
over one step. It is not a proof of joint stationarity or evidence of protein
sampling efficiency or assembly.

![One-step reference observables, correction controls and candidate costs](../results/partner-atlas-reference-completed-20261004/layout02/figure.png)

Each arm started afresh from every one of the same 8,192 cached IID sources
(four independent streams of 2,048). Sources were not selected by their old
or new acceptance outcomes. The target has two mobile spheres and one fixed
sphere, core radius 0.1, depletant radius 0.9, activity 0.5 and wall radius
1.6. The bath correction includes the many-body exclusion geometry. There
were 32,768 outer calls and 147,456 candidate attempts, with no new sources,
replacements or extensions. Rejected states and identities remain in every
denominator.

The largest absolute endpoint-minus-source mean divided by its empirical
standard error, across the 59 predefined observables, was 2.40 for direct,
2.46 for guided one-trial, 2.69 for guided eight-trial and 2.17 for flat
eight-trial. The frozen gate was six standard errors plus a small numerical
allowance. These correlated diagnostic checks are not a simultaneous
confidence guarantee.

The direct proposal's bounded learned-density drift was
**0.000172 ± 0.000246** (one standard error). Omitting the proposal correction
gave **0.010254 ± 0.000566**; reversing it gave **0.012613 ± 0.000606**.
The controls reuse the same direct candidates, baths and final uniforms.
They demonstrate sensitivity to these two balance errors without generating
additional physical samples.

| Arm | Candidate attempts | Accepted endpoints | Accepted internal lens changes |
|---|---:|---:|---:|
| Direct | 8,192 | 4,584 | 4,226 |
| Guided, one trial | 8,192 | 4,273 | 3,924 |
| Guided, eight trials | 65,536 | 6,607 | 6,187 |
| Flat, eight trials | 65,536 | 5,688 | 5,239 |

Every row has 8,192 outer calls. Eight trials increased accepted changes per
outer call while using eight times as many candidates. Neither those counts
nor the short reference runtime measure mixing, effective independent
samples per CPU, or the ability to reorganize protein contacts.

The independent audit checked source identities, decision arithmetic,
retained poses, counters and observable reductions. It did not replay the
RNG, reevaluate proposal densities or recompute geometry. The figure reads
only the completed scalar summary and its authenticated reference/audit
receipts; no journal or pose data were read during plotting.

- [Complete physical reference receipt](../results/partner-atlas-reference-20261004/reference/receipt.json)
- [Independent audit](../results/partner-atlas-reference-audit-20261004/attempt01/audit.json)
- [Plotted values](../results/partner-atlas-reference-completed-20261004/layout02/plotted-values.json)
- [Render receipt with unchanged input hashes](../results/partner-atlas-reference-completed-20261004/layout02/receipt.json)
- [Vector figure](../results/partner-atlas-reference-completed-20261004/layout02/figure.svg)

The first render and its successful input-authentication receipt remain in
the parent result directory. `layout02` corrects overlapping captions only;
the inputs and numerical plotted values are unchanged.
