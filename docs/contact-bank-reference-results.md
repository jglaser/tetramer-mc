# Contact-bank pilot: improved sampling, unresolved weights

The fresh 131,072-draw pilot completed all eight populations and both independent
audits. It improves the native and no-entry importance estimates substantially,
but **does not pass convergence**. The known-pocket diagnostic exposes
compensating regional discrepancies that agreement of the total weights hides.
Full-vessel and finite-assembly production remain gated.

The repaired shape, two-neighbor scaffold, R4 domain, radius 1.5 Å, activity
0.035 Å⁻³ and complete native classifier are unchanged. Native information is
used in this integration guide; this is not geometry-only assembly evidence.
The [construction](contact-bank-reference.md) and [frozen preparation](../runs/reference-contact-bank-preparation-20260922/plan.json)
record every training source and the two guide widths. The failed original
48-component preparation was retained without launching physics from it.

## Independent fresh populations

Each arm contains four populations of 16,384 attempts, including every exterior
and hard-invalid zero, with two independent clouds at λ/z=128. The full Gaussian
mixture is untruncated and normalized globally. The narrower `bank` arm uses
the fitted covariance plus the declared floor; `wide` multiplies all covariances
by four.

| Arm | Region | log Qz | Population RSE | Importance ESS | Largest draw |
|---|---|---:|---:|---:|---:|
| bank | Native entry | 61.2616 | 17.81% | 88.7 | 8.72% |
| bank | Contact without entry | 42.5428 | 3.38% | 323.2 | 2.75% |
| wide | Native entry | 61.1195 | 10.99% | 38.6 | 9.26% |
| wide | Contact without entry | 42.6846 | 22.78% | 25.3 | 13.02% |

The paired-population approximate 95% intervals for β(F_native−F_no-entry)
are −18.7188±0.6696 and −18.4349±1.0029. These conditional, unconverged
contrasts are not finite-system assembly free energies. The reporting class
"contact without entry" does not identify a distinct competing physical basin.

![Fresh population and importance diagnostics](../runs/contact-bank-pilot-comparison-20260922/contact-bank-pilot.png)

The tighter arm's no-entry ESS exceeds 200 and its observed population error
is below 10%, but its largest contribution exceeds 2%. Native ESS, population
error and largest-draw requirements fail. Thirteen significant original
radial/angular/orthant comparisons fail the declared agreement check.

The frozen v1 analyzer reports agreement of individual region log masses:
their width differences are about 0.142 each. Those differences have opposite
signs, however, so the free-energy contrasts differ by **0.2839 kBT**, exceeding
the requested 0.2 kBT tolerance. The working v2 analyzer now checks the contrast
itself as well, with a regression test. The completed v1 artifact is unchanged;
its overall verdict was already unresolved and remains so.

## The total conceals a coverage disagreement

A [separately frozen diagnostic](../runs/contact-bank-reference-partition-plan-20260922/protocol.json),
declared before examining the fresh weights, partitions complete native R4
mass into the old R5 intersection and the rest of native R4. It uses the exact
old chart, strict original metric q>1 and original capture support. Both parts
retain the same physical weights and every attempted-draw denominator.

| Arm | Native subset | log Qz | Population RSE | ESS | Native mass fraction |
|---|---|---:|---:|---:|---:|
| bank | Old R5 intersection | 60.7458 | 14.30% | 175.8 | 59.70% |
| bank | Remaining native R4 | 60.3528 | 27.00% | 17.6 | 40.30% |
| wide | Old R5 intersection | 59.9855 | 12.38% | 30.0 | 32.18% |
| wide | Remaining native R4 | 60.7313 | 21.90% | 20.5 | 67.82% |

The fresh intersection estimates differ by 0.7603 log units, or 4.02 combined
observed population standard errors. Their native complements shift in the
opposite direction. This explains why near agreement of the total is
insufficient. The tighter arm's intersection estimate is close to the saved
reference value 60.6821; that old reference remains separate and was used for
guide training. It is not fresh validation of the proposal.

The complement is explicitly measured here, not assumed negligible. Its poor
ESS and the subset disagreement leave R4 native weight unresolved. The
full-vessel complement outside R4 still has no converged estimate. No unbound
R4 contribution was observed; the previous deterministic bound
log Q_unbound≤−9.8079 remains valid inside R4 only.

## What improved and what remains expensive

Sampler CPU was 1,167.8 seconds for `bank` and 639.2 seconds for `wide`.
The eight-job campaign and audits finished in about 386 seconds wall time.
There were 597 and 4,560 exterior Gaussian attempts, respectively; all remain
zero-weight attempts in the estimates. The proposal and Jacobian audits passed.

The tighter arm's observed paired-cloud noise accounts for about 9.2% of
native variance and 29.5% of no-entry variance. Native pose allocation therefore
remains the main observed problem. Only 726 of 8,308 tighter-arm proposals from
the eight reference components contributed, although that branch supplied
about 91.1% of the estimated native weight. Its deliberately conservative
geometric floor supplies 80–89% of fitted covariance trace in original u.
This identifies covariance refinement and coverage of the native complement
as useful next targets; it does not prove a particular smaller floor will work.

The gain in observed importance ESS is useful integration progress. It is not
a measured MCMC contact-mixing speedup, a stationary contact ESS per CPU, or
evidence that a mobile finite system reaches equilibrium. Larger independent
populations, proposal/intensity checks, important-region coverage and the
later assembly controls are still required. No model-negative conclusion is
supported by the failed screening gate.

## Reproducibility

The [primary comparison](../runs/contact-bank-pilot-comparison-20260922/report.md)
and [pocket partition](../runs/contact-bank-reference-partition-20260922/report.md)
retain separate populations, source hashes, original denominators and all
reporting definitions. Old physical campaigns and their completed audits were
reused without rerunning them. The involutive transport, assembly kernels,
optional cluster-size bias and 38 checked Lean theorems remain unchanged.

The new fitter, controller and observer work has 51 focused tests, including
the later strict contrast check and native-partition observer. The conclusion
at this stage is **an explicit unresolved sampling limitation**. Neither
finite-system native stability nor finite-system instability is established.
