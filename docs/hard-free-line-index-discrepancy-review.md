# Retained-index discrepancy: independent arithmetic review

The conditioned retained-index estimate of native mass is **1.80024% below**
the full-mixture estimate on the same saved draws. That difference is a
sampling warning and must accompany the reported ESS tradeoff. This review
found no error in the saved-label correction or its arithmetic, but cannot
establish the absence of a sampler implementation error or unseen tails.

## Two uncertainty estimates tell different parts of the story

All 65,536 conditioned attempts, including invalid zeros, were retained. The
relative differences below are divided by the observed full native mean;
that denominator makes them ratio diagnostics. The underlying paired linear
difference is the quantity whose expectation is zero.

| Paired uncertainty estimate | Relative SE | Difference / SE | Approximate 95% interval for relative difference |
|---|---:|---:|---:|
| All attempted rows | 1.16503% | −1.54524 | [−4.08370%, +0.48321%] |
| Four independent population means | 0.46972% | −3.83256 | [−3.29511%, −0.30538%] |

The population interval uses Student's t with **three degrees of freedom**;
3.83 here is not a Gaussian “3.83 sigma” significance claim. The four
within-population relative differences are −2.76364%, −1.00945%, −1.04644%
and −2.31167%. They are all negative, and their dispersion is small compared
with the paired-row variance. The population variance estimate is 16.26% of
the row-based estimate of the variance of the mean.

The native subregions explain this unusually small aggregate dispersion:
their four population differences have a sample correlation of **−0.96373**.
Individually, the old-R5 native intersection differs by −0.22522%
(−0.177 population SE), and the remaining-R4 native region by −3.57311%
(−1.552 population SE). Their differences nearly cancel in how they vary
between these four populations. This observed correlation is not a physical
mechanism or a property established for future samples.

Neither uncertainty calculation should replace the other after seeing its
answer. Four populations give a noisy variance estimate, while the row
estimate also depends on the observed weighted tails: the 12 largest absolute
native differences account for **43.13% of the sum of squared differences**.
Both intervals are approximate and neither bounds unobserved contributions.
Total and native mass are nearly identical here, so their similar warnings
are not independent findings.

## Arithmetic and label checks

For primitive label \(\ell\), the unchanged law has prior probability
\(\eta_\ell\), old component density \(g_\ell\), actual density
\(h_\ell\), and complete densities
\(Q_0=\sum\eta_\ell g_\ell\),
\(Q_1=\sum\eta_\ell h_\ell\). On the physical R4 support,
\(h_\ell=g_\ell/Z_\ell\), with effective \(Z_\ell=1\) for uniform
or fallback labels. The old-responsibility construction gives

\[
W_I=\widehat F\,Z_\ell/Q_0,
\qquad
W_I/W_F=(Q_1/Q_0)Z_\ell.
\]

The conditional label probability under the actual sampler is
\(\eta_\ell h_\ell/Q_1\). Thus, assuming the validated proposal and
label-independent conditional cloud law,
\(E[W_I\mid x,\mathrm{clouds}]=W_F\). This is an expectation over labels,
not an identity for a finite realized sample. The full-mixture estimator is
the Rao–Blackwell average and remains the primary estimator.

The independent review reproduced the saved paired estimates by separately
forming \(\log\widehat F+\log Z_\ell-\log Q_0\). Maximum discrepancy
between the two arithmetic orders was **1.42×10⁻¹⁴ in log weight**. It also
reintegrated the selected conditional Normal over the **saved** interval
endpoints: maximum absolute mass discrepancy was **3.33×10⁻¹⁶**. No intervals,
poses, contact labels or physical clouds were regenerated.

All successful native Gaussian draws used non-fallback conditioning; no
native contribution came from mass-floor fallback. Axes 0 and 2 contributed
−0.84253 and −0.86335 percentage points of the native discrepancy; axis 1
contributed −0.09588, and the uniform branch +0.00151. These are contributions
to one observed difference, not independently expected-zero axis estimates.
There is no evidence here of a missing fallback or uniform term.

The hard-only native difference is +0.13224% (0.489 population SE); the
hard-only total difference is +0.01556% (0.308 population SE). The physical
competing-contact difference is −0.35454% (−0.431 population SE). These
controls provide no corresponding normalization warning.

Using each saved cloud separately gives native differences of −0.69316%
and −2.88381%; they share poses and labels and are not independent campaigns.
The paired-cloud calculation attributes about **31.22%** of the observed
paired-difference variance to cloud noise. The remaining noise includes pose
and retained-label variation; this decomposition is itself an estimate.

## Consequence

Report the population warning alongside the row interval and weighted-tail
concentration. The arithmetic supports treating it as an unresolved sampling
diagnostic, rather than claiming an index-estimator bug or dismissing the
offset as harmless. The fixed larger-population stage continues with the
**same full-mixture primary weights, binary and guide**. No index estimator
is substituted, and no full-vessel or assembly gate opens from this result.

The separate immutable arithmetic receipt is
`results/hard-free-line-index-discrepancy-review-20261001/analysis.json`.
Its adjacent script binds the original retrospective, completed population
summary, all four raw conditioned streams and classified array archives.
It performed zero new geometry queries, pose draws, Poisson clouds or
classifier calls. The original retrospective and its archived sources were
left unchanged.
