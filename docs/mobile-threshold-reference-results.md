# Independent reference across the native-entry threshold

The largest unregistered-contact observations in the full-wall comparison
belong to the shoulder of the already known motif 7/4 native environment.
Their largest member-position errors are 2.015–2.408 Å, just outside the fixed
2 Å entry threshold, while angular errors remain within the entry criterion.
They are not evidence for a clearly separate competing basin. The classifier
and its thresholds remain unchanged.

The [fresh independent calculation](../runs/mobile-threshold-reference-comparison-20260921/report.md)
integrates an unchanged radius-four latent ball around ordinary atlas component
146, anchored at the second fixed tetramer B. Both physical neighbors remain
present, in their original order. The center enters native motifs 7/4, while
the region spans both sides of the native-entry threshold. There is no native,
contact or effective q filter in the integration target.

## Result and practical limit

Four independent populations of 16,384 unconditional draws completed, with
two Poisson clouds per contributing pose at λ/z=64. All four physical jobs and
the frozen raw audit passed, using 440.44 sampler CPU seconds. Of 65,536 draws,
8,136 are hard valid. The physical bath is unchanged: radius 1.5 Å and activity
0.035 Å⁻³. All invalid draws remain zero in the denominator.

| Region within this fixed R4 | log Qz | Observed row RSE | Population RSE | ESS | Largest row |
|---|---:|---:|---:|---:|---:|
| Entire region | 58.9515 | 77.93% | 69.13% | 1.65 | 73.23% |
| Native entry | 58.9515 | 77.93% | 69.13% | 1.65 | 73.23% |
| Exclusion contact without native entry | 42.7699 | 60.68% | 44.11% | 2.72 | 54.22% |

The hard-only volume is much better resolved: log Q0=−11.8958 with 1.04%
observed relative error. Its native-entry part has log Q0=−14.0634 (3.26%) and
the no-entry contact part has log Q0=−12.0173 (1.11%). No unbound contributions
were observed, which is not an upper bound or proof of zero volume.

The independent no-entry estimate is 2.67 log units above the broader global
proposal's estimate **restricted to this identical R4**. The original global
proposal is lower by 3.28 log units. All three estimates remain separate;
the older rows selected this reference region and are exploratory comparisons.
Their disagreement and low effective sample sizes preclude treating the
full-wall native-versus-contact gap as converged.

The fresh finite-region native/no-entry log ratio is 16.18, with observed
paired delta-method row SE 0.99. Because both weighted estimates depend on
very few observations, that SE is not a reliable missing-tail guarantee or a
converged free-energy conclusion. About 12% of observed total variance and
14% of no-entry variance are attributed to the paired Poisson clouds; pose
coverage is the larger observed source of uncertainty. More clouds at the
same poses would leave that problem largely intact.

The old full-wall proposals also disagree on parts of the hard-only volume.
For example, the original atlas gives log Q0=−12.7399 for this R4, versus the
independent −11.8958. Thus coverage of the geometric denominator deserves
attention alongside rare high-depletion-weight poses. Formal full support
does not guarantee useful finite-sample coverage.

![Separate estimates on an identical finite region](../runs/mobile-threshold-reference-comparison-20260921/finite-reference.png)

This calculation narrows the outstanding problem: refine the statistical
weight near the registration boundary and retain explicit coverage of other
contacts. It does not show that contacts outside the catalogue are absent,
nor that a conditional native preference suffices for bulk assembly. The
[four-body mobile control](mobile-four-body-growth-results.md) addresses a
different question: whether an additional tetramer can attach through the
validated moves.

## Reproducibility

[The generic finite-region controller](../tools/run_latent_reference_campaign.py)
archives the supplied region without modifying its means, covariance or
anchor. It permits any fixed neighbor as the chart anchor while verifying the
entire physical scaffold. It binds the reviewed executable, embedded Rust
source and eight-file Python audit closure; it uses fixed independent seeds,
retains invalid zeros, drains started workers on failure and prohibits retries.
The entire R4 lies inside D170 and has a certified 39.68 Å atomic-wall
clearance bound, so the physical wall is redundant on this finite target.

[The downstream classifier](../tools/analyze_mobile_threshold_reference.py)
uses the frozen native definition and independently checks exact
variable-radius sphere-union exclusion contact. It records zero
native-entry/unbound anomalies and zero near-negative core-gap cases. It
retains paired Qz/Q0 covariance, four-population dispersion, cloud-noise
diagnostics and the original unconditional counts. Old wall rows are only
geometrically restricted to the same R4; their proposal denominators are
unchanged, with zeros outside the region.

Nine controller tests and nine classifier/statistics tests passed.
[The validation ledger](../runs/mobile-threshold-reference-validation-20260921/validation.json)
binds these checks, terminal statuses, reviewed sources and completed analysis.
