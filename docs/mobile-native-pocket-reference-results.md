# Independent weight of the alternative native pocket

Fresh uniform-volume integration confirms substantial weight near the new
native motif-7/motif-4 pocket. The result removes the global Gaussian-mixture
denominator as an explanation for that signal. It does not establish the
whole site's weight or the equilibrium of mobile tetramers.

The larger independent repeat gives **log Qz=60.6821**, with observed relative
SE 7.67% (7.97% from eight population means), ESS 169.8 and largest-row share
3.84%. It agrees with the initial core within their observed uncertainties.
Measured coverage now extends to R32; the cumulative estimate is log Qz=61.0335
with 7.75% observed relative SE. The outer-shell contributions remain noisy,
so these errors do not bound unseen weight. See the
[completed follow-up comparison](../runs/mobile-native-pocket-coverage-comparison-20260921/report.md).

![Original and alternative native placements](../runs/mobile-native-pocket-geometry-final-20260921/native-pockets.png)

The figure uses the same fixed scaffold and two orthographic projections.
Red lines mark catalogue-prescribed monomer contacts. The alternative is an
independently checked, hard-valid saved pose. Its minimum hard-surface gap
to the original mobile placement is 46.24 Å: these two illustrative placements
could coexist without mutual overlap. This geometric observation does not
measure their joint association free energy.

## Completed fixed experiment

The [campaign](../runs/mobile-native-pocket-campaign-20260921/protocol.json)
uses the previously declared [ideal-centered chart](mobile-native-pocket-reference-design.md),
with unchanged covariance, original q>1, both exact fixed neighbors, D170,
depletant radius 1.5 Å and activity 0.035 Å⁻³. Four independent populations
of 8,192 unconditional draws each estimate the R5 core; another four estimate
the disjoint 5<ρ≤8 shell. Both use two independent Poisson factors per valid
pose at λ/z=64. Every invalid draw contributes zero. No scaffold repair,
covariance fit, optional extension or valid-only resampling was performed.

| Finite region | log Qz | log Q0 | Observed relative SE | Importance ESS | Largest row |
|---|---:|---:|---:|---:|---:|
| Alternative native R5 | 60.4999 | −24.8390 | 17.10% | 34.2 | 13.77% |
| Alternative native shell 5–8 | 59.6054 | −23.0573 | 20.97% | 22.7 | 12.33% |

The four independent core estimates of log Qz are 60.2558, 60.3964,
60.8283 and 60.4233. The shell estimates are 59.0128, 59.9318, 59.7693
and 59.4811. These are appreciably steadier than the original global
importance estimate, whose effective sample size was only 1–3. They remain
finite-sample estimates with consequential uncertainty and no bound on
unobserved contributions.

The [core audit](../runs/mobile-native-pocket-campaign-20260921/r5/assessment/report.md)
and [shell audit](../runs/mobile-native-pocket-campaign-20260921/shell5to8/assessment/report.md)
independently reconstruct all 65,536 poses and their physical Haar Jacobians.
Maximum log-Jacobian discrepancy is 5.97×10⁻¹². All eight jobs and both audits
completed successfully, using 125.8 sampler CPU seconds. The
[validation record](../runs/mobile-native-pocket-validation-20260921/validation.json)
also binds the unchanged inputs and the controller's four failure/identity
tests. No Rust implementation change was needed.

## Interpretation and remaining coverage

The original native R4, measured independently on this same fixed scaffold,
has log Qz≈36.0852 and log Q0≈−20.7854. Thus the new **finite core** has
about 24.60±0.11 more log units of physical weight, using the larger repeat
and observed row delta uncertainty. Its smaller hard-accessible measure
contributes −4.03 log units, while the difference in regional depletion
enhancement contributes +28.63. These contributions sum to the
physical ratio. The enhancement is log(Qz/Q0), not the mean overlap volume
or the free energy of an individual pose.

The catalogue assigns four external monomer contacts at the new site versus
three at the original site. That geometry is consistent with a stronger
depletion preference, but it does not isolate causality or energetic
nonadditivity. These finite-region comparisons omit the cost of assembling
the prescribed neighbors and cannot establish a bulk nucleation barrier.

In the initial experiment, the outer shell supplies about 29% of measured R8 weight.
Its observed contribution does not decay convincingly toward the outer
boundary. A previously saved pose with the same registered triangle lies
at radius 20.62 in this chart. R8 therefore leaves demonstrated site support
outside the initial measured region. This motivated the completed follow-up:
eight fresh core populations of 16,384 draws and four fresh populations of
8,192 draws in each of 8–12, 12–16, 16–24 and 24–32. All 24 jobs and five
independent audits passed, using 421.5 sampler CPU seconds. The controller
kept at most eight children active and used no physical retries or extensions.

The new/initial core log ratio is 0.1822±0.1874 observed row SE. The cumulative
estimate uses the new core **once**, the old independent 5–8 shell, and the
four new outer shells; it does not add or pool the initial core. R8–32
contributes 5.65±2.28 percentage points of observed R32 mass. The newest
24–32 shell contributes about 0.006%, but is dominated by one observation;
that small estimate is not an upper bound. Native-label diagnostics remain
separate from the integration target throughout.

The [scope review](../runs/mobile-native-pocket-scope-review-20260921/recommendation.md)
identifies the next consequential uncertainty as other contact environments
in the full original physical wall. The new result resolves the misleading
interpretation of q>1 as nonnative: its dominant newly discovered weight
belongs to another registered native triangle. It does not reconcile numerical
values for the earlier one-neighbor SMC, ideal AB and observed AB problems,
which have different physical targets. Their remaining coverage problems
must retain those distinctions.

The next reference should compare the union of registered sites with the
entire unregistered-contact remainder, retain inside/outside-D170 reporting,
and test proposal/population sensitivity. A matched four-mobile-tetramer
attachment challenge can then test growth beyond the known triangle.
Template-free discovery still requires an independently clean geometry-only
proposal; the successful atlas remains native-informed.

The [combined validation ledger](../runs/mobile-native-pocket-validation-20260921/completed-validation.json)
records 32 completed physical populations, 327,680 unconditional draws and
seven successful audits across both campaigns, with 547.3 sampler CPU seconds.
The current controller and downstream analyzer pass seven and sixteen tests,
respectively. Each comparison archives its own code and source inputs;
the follow-up reuses the old independent shell without replaying its audit
or native classifier.
