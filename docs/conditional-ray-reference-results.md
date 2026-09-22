# Conditional-ray reference: unresolved contact weights

The fixed **786,432-draw campaign completed**, including all 24 independent
populations, six independent density/weight audits and classification of every
saved draw. The predeclared regional convergence gate **failed**. This leaves
finite-system native stability and template-free sampling unresolved; it is
not evidence that the model prevents assembly.

Consequently the dependent full-vessel comparisons, matched physical mixing
benchmark, compression evaluation and N=12/N=24 assembly production were not
launched. Their stopping condition was specified before sampling. The physical
parameters remain 1.5 Å and 0.035 Å⁻³, with the same repaired tetramer, fixed
observed scaffold, R4 and native definition.

## Completed estimates

Each arm has four independent populations. Pilot, larger-population,
defensive-fraction and cloud-intensity estimates remain separate. Every
hard-invalid attempt contributes a zero to the unconditional denominator.

| Arm | Draws per population | Native log Qz | No-entry contact log Qz | Native ESS | No-entry ESS |
|---|---:|---:|---:|---:|---:|
| Pilot uniform | 16,384 | 56.3085 | 42.9558 | 1.1 | 2.3 |
| Pilot ray, α=0.5 | 16,384 | 55.3494 | 41.9002 | 3.2 | 1.6 |
| Larger uniform | 65,536 | 55.8667 | 44.4066 | 1.8 | 1.1 |
| Larger ray, α=0.5 | 65,536 | 59.4138 | 42.5778 | 1.1 | 2.7 |
| Ray, α=0.2 | 16,384 | 55.9348 | 42.3076 | 1.3 | 6.4 |
| Ray, λ/z=128 | 16,384 | 57.1322 | 44.9309 | 2.4 | 2.1 |

These logarithms are logarithms of arithmetic mean **linear weights**, not
averages of log weights. They are unconverged estimates. “No-entry contact”
means an exclusion contact failing the unchanged complete native-entry
classifier; it does not identify a separate physical basin.

In the larger ray arm, native and no-entry population relative SEs are **95.2%**
and **56.3%**, versus the required 10%. Their largest draws contribute **95.0%**
and **60.4%**, versus the required 2%; importance ESS is far below 200.
The larger uniform arm also fails all three quality requirements.

The paired-population estimates of β(F_native−F_no-entry) are −11.46±4.48
for uniform and −16.84±1.26 for the ray guide, where the quoted half-widths
are approximate Student-t3 95% delta intervals. Both exceed the permitted
0.5 kBT half-width. Their point estimates differ by **5.38 kBT**. These
intervals do not bound unseen mass and cannot support a thermodynamic verdict.
None of the six prescribed depletion-weight comparisons passes its full
agreement requirement. Doubling cloud intensity does not resolve the failure.

![Separate populations, regional weights and failed concentration diagnostics](../runs/mobile-conditional-ray-comparison-20260921/conditional-ray-reference.png)

## Where the estimates remain unstable

The larger uniform estimate places 99.91% of observed native weight in latent
radius 2–3; the larger ray estimate places 94.97% in radius 3–4. For no-entry
contacts, the larger uniform estimate places 98.44% of its observed weight
in angular projected radius squared a²<4, whereas the ray estimate places
76.98% in 4≤a²<9. These changes are driven by very few observations, not
established equilibrium redistribution.

For the direct larger-arm comparison, native disagreement affects radial
bins 2–3 and 3–4, angular bin a²<4 and latent orthants 19, 23 and 51.
No-entry disagreement affects all three radial bins, both angular bins
a²<4 and 4≤a²<9, and orthants 18, 26, 30, 42 and 59. Across all six
predeclared comparisons, 96 significant stratum comparisons fail. The
[complete report](../runs/mobile-conditional-ray-comparison-20260921/report.md)
lists every failed bin; `analysis.json` retains all 64 orthants, including
zero-weight and unvisited subsets separately.

![Radial and angular contributions disagree across proposals](../runs/mobile-conditional-ray-comparison-20260921/conditional-ray-strata.png)

The guide eliminates the earlier out-of-R4 loss: **all 786,432 proposals
remain inside R4**. In the larger ray arm, 94,884 of 130,709 guided attempts
(72.59%) use the prescribed same-ray fallback. Boundary-interval draws supply
29.51% of its observed no-entry weight. The dominant native draw comes from
the fallback, and boundary-interval draws supply no observed native weight.
This is consistent with targeting the outer member-error boundary while
retaining the original angular/directional law. It does not demonstrate an
importance-sampling or physical mixing speedup.

Cloud noise remains consequential in some estimates: its observed fraction
of larger-ray no-entry variance is 84.4%, versus 5.9% in the larger uniform
arm and 2.3% in the λ/z=128 arm. These noisy decompositions and independent
pose populations cannot isolate a reliable cloud-intensity speedup. The
remaining pose-weight concentration is severe in every arm.

## Reference limits and remaining space

The independent Python/Rust proposal-density discrepancy is at most
1.32×10⁻⁹ in log density for protein rows. Pose reconstruction errors are
below 1.36×10⁻¹³ in latent coordinates. Analytic sphere/depletion controls,
pure-uniform identity, mixture normalization and interval/fallback controls
passed before the protein campaign.

Hard-only total log volumes span −11.9115 to −11.8893. Despite that small
spread, the predeclared population-based three-SE check also fails for total
and no-entry hard volumes in two comparisons: larger ray versus larger
uniform, and larger uniform versus pilot uniform. These failures remain in
the gate; they are not replaced with more favorable row-level errors.

All hard-valid poses are partitioned into native entry, contact without
entry and unbound without entry. No unbound draw was observed. Independently
of that absence, the geometry gives the finite-region bound

\[
 Q_{\rm unbound}(R4)\le V_6(4)\max J,\qquad
 \log Q_{\rm unbound}(R4)\le -9.8079042,
\]

because an unbound pose has zero depletion-overlap enhancement. This bound
applies only inside the measured R4. The full atomic-wall contact remainder
outside R4 is still unresolved and has no new converged bound here.

Earlier SMC values retain the distinctions in the existing
[same-target reconciliation](smc-reference-reconciliation.md). No old SMC
population was rerun, and no single-neighbor capture normalizer or endpoint
fraction was substituted for the present two-neighbor region mass. The old
completed uniform R4 comparison was reused by frozen hashes, without raw
re-auditing or pooling its observations into this campaign.

## Implemented assets and validation

The [conditional-ray implementation](conditional-ray-reference.md) preserves
the earlier guide schemas. The optional [frozen assembly bias](frozen-assembly-bias.md)
is implemented with elementary-kernel corrections, rollback, exact checkpoint
continuation and physical reweighting tests. The
[native-blind memory export](geometry-only-growth-control.md) retains all
64 historical slots as 128 reciprocal branches, without native filtering;
it is frozen and has not been used for production here.

The [Lean project](../formal/README.md) now has **38 checked theorem audits**,
including the previous 19, Poisson mean/variance and the implemented count gate
with zero-volume cases. Only standard Lean axioms occur. The code-to-theorem
table keeps geometric thinning, predicates, Jacobians and floating-point
execution explicit obligations.

The [completed-stage ledger](conditional-ray-stage-validation.json) binds the
[software checks](../runs/conditional-ray-implementation-validation-20260921/validation.json),
physical protocol, raw audits, final classification and frozen model by hashes.
Validation records cover 40 focused Rust cases, 61 focused Python cases,
the sphere controls, all 24 protein populations and all six audits. Physical
sampling used **5,820.52 CPU seconds**, with at most eight simultaneous jobs.
No retry, adaptive allocation, additional physical sampling or assembly
production followed the failed gate.

The final outcome of this fixed plan is **an explicit unresolved sampling
limitation**. Neither finite-system native stability nor instability has
been established.
