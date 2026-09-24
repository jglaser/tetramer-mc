# Frozen nonlinear contact-guide comparison

This comparison extends the [existing invertible kernel prototype](kernel-transfer-prototype.md)
using the [fixed conditional fit](kernel-shear-fitting.md). The original Gaussian
affine parameters, mixture allocation and 50% uniform component are retained;
only the kernel displacement is fitted. Mixture weights are unchanged to
floating-point normalization roundoff (maximum observed difference 1.4e-17).
No production move, physical target or convergence gate is changed.

## Pilot held-out comparison

One fit used 262,144 archived attempts from four fitting populations. Its model
was frozen before opening the four held-out arrays, containing another 262,144
attempts. These populations had been inspected previously, so this is a
retrospective held-out diagnostic. Sixty-one of the 84 components met the fixed
responsibility-ESS threshold and received a 16-center shear; the others retained
their affine form. All original attempted rows, including zeros, remain in the
evaluation archive and denominators.

The paired-cloud physical second-moment ratios are:

| Contact region | Bank-source holdouts | SMC-guide-source holdouts |
|---|---:|---:|
| Native inside historical R5 | 0.898 | 0.871 |
| Native outside historical R5 | 0.890 | 0.814 |
| Contact without native entry | 0.786 | 0.797 |

A ratio below one favors the fitted proposal at the observed points. All twelve
individual population/region ratios are below one. This does not establish a
mixing or CPU speedup. The competing-contact second-moment estimate remains
heavy-tailed: in the bank-source holdouts its contribution ESS is only 3.79 and
one draw supplies 50.2% of the sum. Those are second-moment-contribution
diagnostics, not the physical weight ESS.

![Pilot held-out comparison](../runs/fitted-kernel-shear-pilot-20260924/heldout-comparison.png)

The original problematic subdivisions limit the conclusion:

- Native-complement orthant 55 has ratios 1.013 and 0.912. Improvement is
  inconsistent between the two source arms.
- Competing-contact orthant 63 is unchanged to numerical precision. Its dominant
  components, 11 and 41 (zero-based), were not fitted: their training
  responsibility ESS values were 15.48 and 17.26, below the predeclared 64.
  Together they supply approximately 96.5–99.2% of physical-weighted baseline
  proposal responsibility there. This is a data-sufficiency limitation of this
  fit, not evidence that the fitted kernels extrapolate badly.

An independent read-only review authenticated 63 files, reconstructed sampled
baseline and warped densities within 6.3e-15, and reproduced the original-N
likelihood and both second-moment formulas for a held-out competing-contact
population. It found no missing draw or denominator change. All 11 fitting tests
and 8 diagnostic-controller tests passed before the fit.

- [Frozen pilot design](../runs/fitted-kernel-shear-pilot-20260924/plan.json)
- [Full result, including every subdivision](../runs/fitted-kernel-shear-pilot-20260924/analysis.json)
- [Frozen fitted model](../runs/fitted-kernel-shear-pilot-20260924/fitted-model.json)

## Independent-population evaluation

A separate no-refit evaluation uses the exact same frozen candidate and baseline
on eight later populations: four bank-source and four protected-source
populations of 1,048,576 attempts each. Its 8,388,608 records are disjoint from
fitting and pilot holdouts. Existing physical summaries had already been
inspected, so this remains retrospective. The candidate is fixed before reading
these arrays; every radial, angular and orthant group, including residual and
zero contributions, is retained. Important-group flags use fitting-population
mass only and do not create or replace a physical convergence gate.

The evaluation completed without refitting. Its aggregate paired-cloud physical
second-moment ratios, warped proposal divided by the original affine proposal,
are:

| Contact region | Bank-source populations | Protected-source populations |
|---|---:|---:|
| Native inside historical R5 | 0.874 | 0.858 |
| Native outside historical R5 | 0.739 | 0.724 |
| Contact without native entry | 0.811 | 0.649 |

All 24 individual population/contact-region ratios are below one. Across the
two source arms this amounts to observed second-moment reductions of 13–14%,
26–28% and 19–35%, respectively. This is a proposal-density diagnostic evaluated
on archived physical weights, not measured Markov-chain mixing or a speedup.

![Independent-population evaluation](../runs/frozen-kernel-shear-evaluation-20260924/independent-comparison.png)

The larger evaluation improves the evidence for native-complement orthant 55:
its ratios are 0.796 and 0.700. Its second-moment contributions still have ESS
values of only 9.03 and 2.50. Competing-contact orthant 63 remains unchanged to
approximately 1e-12 relative precision. It carries about 2.6% and 2.3% of the
respective competing-contact masses, so the unchanged coverage matters. The
aggregate protected-source native-complement second moment also has contribution
ESS only 7.14. None of these observations resolves the existing physical
convergence failures or bounds an unseen contribution.

The archived-density evaluation took 295 seconds on one CPU. Density scoring
alone cost 31.3 versus 82.2 seconds for the bank source and 31.4 versus 82.5
seconds for the protected source: approximately 2.63 times the affine cost.
These timings exclude the physical Monte Carlo kernel; they cannot establish
effective contact samples per CPU. All seven evaluator tests passed before
this calculation.

- [Frozen evaluation design](../runs/frozen-kernel-shear-evaluation-20260924/plan.json)
- [Completed evaluation and every subdivision](../runs/frozen-kernel-shear-evaluation-20260924/analysis.json)

## Analytic Gaussian moment control

Keeping the *affine parameters* fixed does not keep the warped distribution's
actual moments fixed. A nonlinear displacement can also improve a distribution
by changing its mean or covariance. The new control replaces each fitted
component by the Gaussian with exactly the same first two moments, computed
analytically from Gaussian expectations of its fixed radial-basis functions.
It retains the mixture weights, coordinate transformation and 50% uniform
component. No data fitting, random sampling or covariance regularization is
used. The model was frozen before evaluating the eight original pilot arrays;
fitting and held-out results are reported separately.

Here q0 denotes the original Gaussian mixture, qm the moment-matched Gaussian
mixture, and qs the full nonlinear mixture. Entries are ratios of the paired
physical second-moment estimates, M2(q_new)/M2(q_old), on the four held-out pilot
populations:

| Contact region | Bank: qm / q0 | Bank: qs / qm | SMC guide: qm / q0 | SMC guide: qs / qm |
|---|---:|---:|---:|---:|
| Native inside historical R5 | 0.956 | 0.940 | 0.936 | 0.931 |
| Native outside historical R5 | 0.917 | 0.970 | 0.886 | 0.919 |
| Contact without native entry | 0.827 | 0.950 | 0.851 | 0.937 |

Changes in the first two moments account for much of the pilot improvement.
The nonlinear shape provides an additional observed 3–8% reduction beyond this
control. The pilot's heavy-tail limitations still apply. The moment Gaussian
minimizes forward KL from its corresponding fitted component among Gaussians;
it is not necessarily the best Gaussian for the physical target or the best
possible affine transport. Thus this comparison isolates the cost of discarding
higher moments of the fitted component, not an optimal nonlinear-versus-linear
algorithm comparison. Competing-contact orthant 63 remains unresolved in both.

All seven analytic-moment tests passed, including independent Gaussian
quadrature and the zero-shear limit. The complete control took 9.3 seconds and
preserves all 524,288 pilot attempts and their original denominators.

- [Moment-control construction](../tools/kernel_shear_moments.py)
- [Frozen control and all subdivisions](../runs/kernel-moment-control-20260924/analysis.json)

## Interpretation limits

This experiment is native-informed proposal design. It does not yet establish
geometry-only contact discovery, physical trajectory efficiency, full-vessel
weights or finite-system assembly. The separate restricted-SMC protein control
continues with its original frozen design.
