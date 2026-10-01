# Fresh hard-free line proposal result

The feasibility-conditioned guide produced **2.32× as many native proposals per attempt**, while complete proposal evaluation cost **9.75× as much CPU**. This validates improved geometric retention, not faster physical sampling. A bounded physical-weight comparison remains useful because it measures the importance weights and Poisson cost that this proposal-only test omits.

![Fresh proposal retention, native geometry and CPU](../results/hard-free-line-fresh-review-20261001/hard-free-line-fresh.png)

Both frozen arms completed four independent populations of 256 draws: 1,024 attempts each, 2,048 total. The repaired shape, scaffold, R4 region, center-capture domain, 92 Gaussian components and 50% uniform component were unchanged. The conditioned arm used equal x/y/z selection, beta=1 and mass floor 1e-12; the control used beta=0. No Poisson cloud was sampled. See the [frozen preparation](hard-free-line-fresh-preparation.md).

All fractions below retain every attempted draw; errors are standard errors across the four population fractions. Counts are observational geometric labels and are not equilibrium weights or accepted moves.

| Event | Original Gaussian92: count, fraction ± SE | Conditioned xyz: count, fraction ± SE | Events / proposal CPU second: original → conditioned |
|---|---:|---:|---:|
| Hard-valid inside R4 and capture | 211; 20.61 ± 0.49% | 329; 32.13 ± 1.52% | 433.3 → 69.3 |
| Any exclusion contact | 211; 20.61 ± 0.49% | 329; 32.13 ± 1.52% | 433.3 → 69.3 |
| Contact with both scaffold tetramers | 211; 20.61 ± 0.49% | 329; 32.13 ± 1.52% | 433.3 → 69.3 |
| Any native entry | 66; 6.45 ± 0.99% | 153; 14.94 ± 1.06% | 135.5 → 32.2 |
| Contact without native entry | 145; 14.16 ± 0.77% | 176; 17.19 ± 0.70% | 297.8 → 37.1 |
| Both native anchors | 59; 5.76 ± 0.92% | 142; 13.87 ± 1.13% | 121.2 → 29.9 |
| Cycle-consistent native triangle | 59; 5.76 ± 0.92% | 142; 13.87 ± 1.13% | 121.2 → 29.9 |

There were no valid unbound proposals in either arm. The remaining unconditional partition was 810 versus 691 in-domain hard-invalid attempts and three versus four exterior attempts. This regional proposal test does not measure missing equilibrium mass elsewhere in the vessel. Native counts by population were `[18, 9, 20, 19]` and `[38, 34, 35, 46]`; all populations were retained.

Of 546 selected conditional branches, **272 succeeded and all 272 were feasible**. The other 274 had no positive-length hard-free segment; each fell back to its original Gaussian scalar coordinate without redrawing the other five coordinates. Every fallback endpoint was hard-invalid. There were no additional numerical mass-floor fallbacks. The conditioned arm's 478 uniform draws supplied another 57 valid poses. The control's 518 original-Gaussian draws supplied 139 valid poses, and its 506 uniform draws supplied 72. The observed improvement therefore comes from repairing feasible translation lines; conditioning still cannot repair lines blocked at fixed outer coordinates.

The measured proposal loop took 0.487 versus 4.747 CPU seconds for 1,024 attempts, or 0.476 versus 4.636 ms per attempt. Conditioned scalar generation used 0.624 s and complete density evaluation used 3.571 s. Full-density evaluation is the dominant new cost. These measurements include diagnostic serialization and endpoint checks; child-process startup is recorded separately (total proposer-child CPU 0.860 versus 5.226 s). Independent reconstruction consumed 12.421 versus 106.376 s; classifier evaluation consumed 2.243 versus 3.574 s, with startup recorded separately. These validation costs are not folded into a production speedup claim.

All eight populations passed independent reconstruction of every complete density, whole-line interval, pose/Jacobian and selected inverse-CDF draw. Maximum discrepancies were 3.79e-10 in log q, 4.12e-13 Å in interval endpoints, 1.97e-13 in inverse-CDF probability and 3.56e-15 in log Jacobian. Each of the 540 valid new poses received exactly one complete native classification and one full sphere-union exclusion-contact classification; invalid/exterior attempts retained null labels. There were no native-without-exclusion-contact anomalies. The empty `width_contacts` sentinel was not used. All 2,048 attempt records remain archived.

The complete execution is `/vast/xvg/tetramer-mc-runs/hard-free-line-fresh-20261001`, protocol SHA `fdd202123ad34bcdf67f5d6623a9e20513a11fee715e340ff403f66f4f40144b`. The [figure and its input manifest](../results/hard-free-line-fresh-review-20261001/manifest.json) were produced only from the frozen `analysis.json`, completed status and protocol; no poses were redrawn or reclassified.

The next justified check is the fixed independent physical-weight pilot, using the full J W/q estimator and all hard-invalid zeros. This result does not pass the original regional convergence gates, establish equilibrium contact weights, measure MCMC acceptance/mixing, or support either assembly stability or instability.
