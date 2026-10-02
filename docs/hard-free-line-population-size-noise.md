# What limits the completed larger contact integral?

The saved two-cloud moments suggest that better pose coverage is the main
remaining opportunity in the worst competing-contact strata. Increasing the
Poisson intensity alone would not address their observed concentration of
weight. This is a finite-sample diagnostic, not a proof that the unseen tails
or true conditional cloud variance are small.

This calculation reuses only the completed pilot/larger summary files and
their frozen population-size comparison. It evaluates no geometry, native
classifier, new pose or cloud. Both within-stage failures and all eleven
population-size failures remain in the
[complete diagnostic](../results/hard-free-line-population-size-noise-20261002/analysis.json).

For two independent per-pose cloud weights W1 and W2, the stored quantity
(W1 − W2)²/4 estimates the conditional variance of their average. Averaging
over unconditional attempts, including invalid zeros, gives an estimate of
the cloud contribution to row variance. Subtracting it from the observed total
leaves a residual pose contribution. Ratios of these estimated variances are
noisy; negative residuals or cloud fractions above one would be retained.

![Observed variance decomposition](../results/hard-free-line-population-size-noise-20261002/noise.png)

For the conditioned guide, the observed cloud fractions of aggregate row
variance are 32.6% for total native mass, 27.1% for competing contacts, 33.4%
for old-R5 native and 31.4% for the native remainder. These are substantial
noise contributions, but they do not dominate the observed totals.

The two competing-contact pose-sign bins that fail the larger proposal
comparison show stronger concentration:

| Bin / arm | Fraction of competing mass | Importance ESS | Largest draw | Cloud fraction |
|---|---:|---:|---:|---:|
| 62, baseline | 1.019% | 3.8 | 49.47% | 1.59% |
| 62, conditioned | 0.212% | 40.6 | 6.16% | 22.47% |
| 22, baseline | 1.585% | 10.5 | 28.89% | 40.45% |
| 22, conditioned | 1.122% | 28.4 | 15.42% | 6.92% |

The complete mixture improves the observed importance ESS in these bins, but
neither is established as converged. Bin 62's large relative disagreement
concerns roughly 0.2–1.0% of observed competing mass. That helps explain why
aggregate masses can agree while its own convergence check fails. It does not
bound a missed contribution elsewhere.

The native bin-55 failures have a different signature. In the conditioned
larger stage, total-native bin 55 has importance ESS 1,343 and largest draw
0.51%; its remaining-native subset has ESS 814 and largest draw 0.80%. Their
cloud fractions are approximately 30%. However, the pooled row-moment SE is
3.51× and 5.05× the respective four-population SE. The total remaining-native
region also has a row/population SE ratio of 2.43.

These are two finite-sample estimates of uncertainty, not two independent
physical experiments. With only four populations, the between-population
variance has three degrees of freedom. The discrepancy identifies uncertainty
calibration as an issue to investigate; it does not prove either SE estimate
correct, or demonstrate an implementation bias. The frozen tests remain based
on the declared population estimator.

For intuition only, replacing noisy clouds by exact conditional expectations
at the same pose law would give the plug-in SE factor sqrt(1 − f_cloud).
The conditioned aggregate fractions correspond to factors approximately
0.82–0.85, and competing bin 22 to 0.965. These are neither rigorous improvement
bounds nor CPU speedups: the fractions themselves are estimated, and obtaining
exact cloud expectations is not a free operation.

The operational decision is to keep λ/z=128 and the running SMC allocation
unchanged. The [independent SMC bridge control](hard-free-protein-smc-control.md)
tests the large historical native-remainder discrepancy. If a subsequent IID
campaign is needed, the present evidence favors addressing sparse pose coverage
and using more independent streams to assess uncertainty, rather than simply
increasing points per pose. Redistributing a fixed total number of draws into
more streams can improve variance calibration; it does not by itself increase
the number of effective sampled poses. No follow-up physical allocation is
launched by this diagnostic.
