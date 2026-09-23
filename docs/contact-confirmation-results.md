# Contact confirmation: aggregate agreement, unresolved concentration and subdivisions

The [completed confirmation](../runs/contact-confirmation-comparison-20260922/analysis.json)
contains **2,228,224 attempts in 20 independent populations**, with every
attempt retained. The five frozen arms and their audits completed successfully.
No old calculation was extended or pooled. The prescribed regional convergence
gate **fails**, so the full-vessel and assembly campaigns remain unlaunched.
This is an unresolved sampling limitation, not finite-system instability.

The comparison artifact SHA256 is
`963b2b32d8b4dc5fe4acbaf3aed4b4a9afdf8ec224d61306dd1ab55e059c0b05`;
its frozen protocol SHA256 is
`410f12940f3b98dd16b728d3875bc29bdc886edd052bfc7d91e503e02aa33bdb`.

## What improved

All aggregate regional comparisons pass the declared 0.2-log-unit and three
combined-population-SE rules. This includes proposal width, population size,
defensive probability and Poisson intensity controls. The direct paired
native/no-entry free-energy comparisons and their ≤0.5 kBT interval-halfwidth
requirement also pass. Classifier/contact consistency and native partition sums
pass.

| Arm | Conditional ΔF(native − no-entry), kBT | Approximate 95% halfwidth |
|---|---:|---:|
| bank | −18.7175 | 0.2513 |
| wide | −18.7344 | 0.1602 |
| small population | −18.6719 | 0.1628 |
| defensive probability 0.2 | −18.6310 | 0.2059 |
| auxiliary intensity ratio 256 | −18.6694 | 0.0238 |

These compare fixed regions **within the frozen R4 domain on the fixed
two-neighbor scaffold**. They are not total native-versus-adsorbed basin weights
in the vessel, a mobile-system chemical potential, or an assembly conclusion.
The intervals use the delta method and Student-t with three degrees of freedom
on four independent linear-mass estimates, retaining native/no-entry covariance.
They do not guarantee coverage of unseen mass. In particular, favorable
covariance makes the intensity256 ratio interval unusually narrow; it cannot
override the other failed diagnostics.

## What still fails

Main-arm quality requires population relative SE ≤10%, importance ESS ≥200 and
largest draw ≤2%. The following region/arm combinations fail at least one:

| Arm / region | Population RSE | Importance ESS | Largest draw |
|---|---:|---:|---:|
| bank / native | 5.05% | 341.8 | 5.13% |
| bank / contact without entry | 3.57% | 614.8 | 3.20% |
| bank / native outside old R5 | 10.43% | 83.6 | 10.57% |
| wide / native | 3.98% | 266.9 | 2.30% |
| wide / native inside old R5 | 6.48% | 122.3 | 4.32% |
| wide / native outside old R5 | 8.19% | 152.9 | 3.56% |
| defensive 0.2 / contact without entry | 5.63% | 271.1 | 5.89% |

All intensity256 standalone regions pass quality. That arm still fails seven
mandatory subdivision comparisons. The small arm's standalone quality is
diagnostic; its comparison against the larger bank allocation remains mandatory.

There are **30 failures among 133 decision-relevant subdivision comparisons**:
29 involve latent-coordinate orthants and one involves angular bin 1. All radial
comparisons pass. There are two failures against wide, nine against small,
twelve against defensive0.2 and seven against intensity256.

The clearest recurring discrepancies are contact-without-entry orthants 58 and
50. Bank assigns 4.23% of its no-entry mass to orthant58 versus 0.91–1.27% in
the controls, a difference of 1.216–1.582 log units. Native orthant55, including
the native complement of old R5, also differs across several controls. The
old-R5/native orthant19 bank-versus-defensive0.2 comparison is the one failure
of the three-SE rule (−0.0914 log units with SE0.0280). The other 29 exceed the
absolute 0.2 threshold. Large errors do not excuse failure of the frozen rule,
which requires both agreement conditions.

Bank's remaining-native population r03 has ESS9.34 and one draw contributing
32.38% of its estimate. Its estimated cloud variance fraction is only1.27%;
the aggregate subset has 97.3% residual pose variance. Increasing the Poisson
intensity alone therefore does not target the main observed source of instability.
Defensive0.2's no-entry r03 also concentrates weight (ESS24.12, largest draw20.15%).

No unbound pose was observed in this R4 calculation. The archived deterministic
bound `log Q_unbound ≤ −9.807904` applies to this finite region only. It says
nothing about the full-vessel unbound contribution.

## Consequence for the next calculation

The independent broad/narrow R4 SMC controls have now completed under their
frozen allocation; all eight populations and both matching-target comparisons
were authenticated. The [completed SMC reconciliation](smc-contact-reconciliation.md)
corroborates the historical R5 intersection and observed hard-only volumes, but
leaves native weight outside R5 dependent on mutation scale. Neither SMC arm has
terminal no-entry observations, so it cannot independently estimate that contrast.
All region masses use normalizer times terminal indicator, not endpoint fractions
or unmatched capture domains. The regional gate remains failed. No thresholds,
regions or allocations are relaxed retrospectively.

The [full-vessel comparison is prepared](full-vessel-comparison-preparation.md)
but unlaunched. It explicitly measures the complement of all named pockets.
Physical contact-mixing and N=12/N=24 assembly conclusions still require their
own mobile-system evidence. Aggregate agreement here is progress toward those
measurements; concentration and subdivision failures leave the model question
open.
