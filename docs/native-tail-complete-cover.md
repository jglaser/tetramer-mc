# Complete native support and the unresolved old-chart tail

This reference measures the part of the original native AB region beyond
radius 12 in the old fitted chart. It samples a **complete geometric cover
of native poses**, then applies old-chart masks to the recorded contributions.
It neither extrapolates the fitted Gaussian nor samples an enormous old-chart
radius-52 ball. The earlier independent finite-shell checks remain separate
calculations; see [the native tail reference](native-ab-uniform-tail-reference.md).

The physical target retains both fixed AB neighbors, capture radius 18 Å,
depletant radius 1.5 Å, activity 0.035 Å⁻³ and normalized SO(3) Haar measure.
The original registration interval is **inclusive**: `0 <= q <= 1`.
In particular, `q=1` is retained. The native intratetramer geometry and
registration metadata are unchanged.

## A complete geometric proposal

For exactly centered rigid-member coordinates, let
\(A=\langle |r_i|^2I-r_ir_i^T\rangle\). Relative to the native orientation,
the mean squared member displacement is

\[
\mathrm{RMS}^2=|\Delta t|^2+
\frac{4c^TA_Ac}{1+|c|^2},
\]

where \(c\) is the left Cayley coordinate in neighbor A's frame.
The original `q<=1` constraint implies RMS ≤ 2 Å. Exact rational LDL
factorization of the archived decimal geometry gives \(A>160I\), hence
\(1/(1+|c|^2)\ge159/160\) throughout this region.

The geometry-derived covariance defines a zero-mean six-dimensional chart
\(x=Lu\), with \(x=(\Delta t,\ell c)\). The emitted decimal covariance
passes a second exact rational matrix comparison:

\[
G^{-1}\preceq
\operatorname{diag}\!\left(I,
\frac{4(159/160)A_A}{\ell^2}\right).
\]

Consequently every exact native pose has \(|u|\le2\). The construction
uses no observed pose weights or learned covariance. Its mapped physical
cover volume is between **0.000695419 and 0.000704132 Å³**; this is the
cover volume before hard and native masks, not the allowed native volume.
The exact certificate concerns real arithmetic on archived JSON decimal
geometry. It does not certify floating-point encoder rounding or bound
depletion-weighted statistical mass.

The separately archived old-chart certificate proves `r_old < 52` for
every exact native pose. This is a support diagnostic only. **No upper
old-chart cutoff is imposed on the estimator.** A recorded native pose
violating that certificate stops the audit for investigation; it is never
silently removed.

## Direct masked weights with every original draw retained

The unchanged Rust executable samples a uniform six-ball of radius 2.
For each draw the contribution is

\[
Y=V_6J(u)\,1_{\rm capture}\,1_{\rm hard,AB}\,1_{0\le q\le1}
\frac{W_1+W_2}{2},\qquad
J=\frac{\det L}{\ell^3\pi^2(1+|c|^2)^2}.
\]

The two independent Poisson estimators have mean \(e^{zC}\), with
\(C\) the overlap against the **union** of both fixed exclusion regions.
Here \(\lambda/z=64\). Their arithmetic mean is used, not the mean of
their logarithms. Invalid poses contribute zero and remain in the original
unconditional sample count.

The analyzer reports the full native integral and these disjoint old-chart
pieces: `r<=4`, `4<r<=5`, `5<r<=8`, `8<r<=12`, and `r>12`.
Each estimate is the direct average of `Y * 1_piece` over the same full N.
Their sum reconstructs the complete native estimate, including same-row
covariances. There is no subtraction of noisy independently estimated totals,
conditioning on success, or pooling with historical finite-shell runs.

The existing kernel cannot apply a second chart before cloud generation.
It therefore evaluates clouds for every valid native draw; the old-chart
restriction is applied in analysis. This preserves the existing executable
and also supplies the finite-shell comparisons without additional sampling.

## Frozen preparation and geometry probes

Preparation: `runs/native-tail-reference-preparation-20260921`.
Its protocol SHA-256 is
`a6d4c95f7625d108dad48255d511b8a5a8b5359b7f338cecf2f76af3b5ed5618`.
The proposed physical allocation is **four populations of 32,768 draws**,
four workers, seeds `113501010 + 1009*i`, two clouds per valid pose and
\(\lambda/z=64\). Geometry, masks and this allocation were frozen before
the 2,048 independent geometry probes at seed `113601010`.

| Geometry predicate | Observed probes / 2,048 |
| --- | ---: |
| Inside capture sphere | 2,048 |
| Original native q interval | 510 |
| Hard-valid against both neighbors | 25 |
| Capture, native and hard-valid together | 9 |
| Valid, old `8<r<=12` | 1 |
| Valid, old `r>12` | 8 |
| Valid, old `r<=8` | 0 |

At the proposed fixed N, these frequencies project approximately 576
valid contributions, including 512 beyond old radius 12. The two-sided
binomial geometry interval projects roughly 221–1,007 tail contributions.
These estimates concern occupied rows only; they predict neither physical
weight precision nor runtime. Zero core or 5–8 probe hits do not imply
zero mass. This complete-cover law is expected to sample the concentrated
old-chart core poorly; the tail is its intended reference calculation.
Independent coordinate and density reconstructions differ by less than
\(8\times10^{-15}\) in the recorded probe checks.

## Explicit analyzer and launcher amendment

The base preparation remains byte unchanged. Review added checks for
nonnegative integer Poisson counts, `overlap_points <= raw_points`, finite
consistent lower/upper/uncertain volumes, shared envelopes for the two
clouds, and terminal launcher success for every population.

Review also identified a missing helper that the frozen launcher copies
by filename: `analyze_latent_region_shells.py`. This was detected before
native production. The complete runner dependency bundle and strengthened
analyzer are frozen in
`runs/native-tail-analyzer-amendment-v2-20260921`.
Its amendment SHA-256 is
`3089d8d5ab47c7ba0064a4578bdef74c442ffb7e89c47c4cfd42a246022ca73f`.
The earlier validation-only amendment is preserved and explicitly superseded.

Use `physical_launch_command_argv` and `analysis_command_argv` from the
V2 `amendment.json`. The replacement runner is byte-identical to the base
runner; **only its script path changes**. Every physical CLI argument,
executable, region, budget and seed remains unchanged. Preflight verifies
all statically declared copied inputs, runner/analyzer `--help` options and
hashes without executing sampling or analysis. The amended analyzer verifies
its own frozen bundle and its link to the original preparation at runtime.

The physical executable SHA-256 remains
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.
The amended analyzer SHA-256 is
`9358949e0fd36dd52343a56dd967f28f4c478d17e092d284cb9403979b8d514d`.
Six tests cover the exact emitted-cover check and its failure case,
inclusive native endpoints, old-radius boundaries, full-N masked variance,
Poisson input validation and terminal-status validation.

## Physical results

All four populations and the amended all-row audit completed successfully.
The pilot retained **131,072** draws and required **21.70 sampling CPU
seconds**. It found 394 valid native poses. All rows passed the independent
coordinate, Jacobian, original-q, positive-Poisson-weight and source-hash
checks; the eight largest contributions also passed independent AB atomic
checks. Maximum native old-chart radius was 43.80, below the support
certificate without a runtime cutoff.

| Original-chart mask | Positive rows | log Q | Row / population RSE | Weight ESS | Largest contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| r≤4 | 1 | 32.12789 | 100% / 100% | 1.00 | 100% |
| 4<r≤5 | 0 | Unobserved | — | 0 | — |
| 5<r≤8 | 3 | 32.45564 | 92.26% / 89.84% | 1.17 | 91.93% |
| 8<r≤12 | 11 | 28.65657 | 66.63% / 54.31% | 2.25 | 48.31% |
| r>12 | 379 | 25.47842 | 71.53% / 70.37% | 1.95 | 69.56% |

The complete proposal reaches the tail, but **its physical weight is not
controlled by this pilot**. Hundreds of valid poses do not imply hundreds
of effective weight samples. Paired-cloud noise accounts for 29.1% of the
tail's observed variance; geometric variation in the conditional mean is
also important. The hard tail volume, by contrast, has 5.13% observed row
error. This separates accessible geometric support from highly concentrated
depletion-weighted mass.

The pilot's complete-native log Q=33.01174 has 67.13% row error and only
2.22 weight ESS. Its one core observation and missing 4–5 contribution are
inadequate to replace the fitted-guide native estimates near log Q=35.8.
Likewise, the 5–8 and 8–12 pilot values do not replace the earlier targeted
uniform-shell references (33.19840 and 30.28155). The 8–12 pilot is **80.3%
below** its targeted reference, a failed finite-shell recovery diagnostic:
the difference is -3.09 combined observed linear standard errors, with only
11 positive pilot rows and weight ESS 2.25. A Gaussian significance
interpretation is unreliable at that concentration. The 5–8 pilot is 52.4%
below its reference (-1.12 observed combined SE), with only three positive
rows. These failures expose poor sampling of concentrated contributions;
they do not establish that the earlier mass disappears.

The new r>12 point estimate is approximately 3.2–3.4×10⁻⁵ of the earlier
full-native estimates. That is a ratio of point estimates, not a bound on
tail mass. It is not added to a full-native estimate that already includes
this region. A larger independent repeat under the same complete law is
inexpensive enough to be the next control; the pilot variance predicts
neither unseen events nor guaranteed precision.

The physical campaign is
`runs/native-tail-reference-4x32768-l64-20260921`; its analysis is
`runs/native-tail-reference-audit-20260921/analysis.json`.
These calculations do not establish native assembly or global convergence.

## Independent 32-fold repeat

The unchanged geometric law was repeated with **16 independent populations
of 262,144 draws**, for N=4,194,304. Seeds were fixed at
`114501010 + 1009*i`. The first 4, 8 and 16 whole populations were declared
as nested prefixes before sampling. No shape, chart, mask, cloud intensity,
executable or source-audit definition changed. Preparation and comparison
plans are in `runs/native-tail-repeat-preparation-20260921` and
`runs/native-tail-repeat-comparison-plan-20260921`.

All original draws passed the frozen all-row audit. Sampling required
**729.77 CPU seconds** and produced 12,063 hard-valid native contributions.
The repeat results are:

| Original-chart mask | Positive rows | log Q | Row / population RSE | Weight ESS | Largest contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| r≤4 | 6 | 32.72376 | 54.59% / 73.39% | 3.36 | 42.87% |
| 4<r≤5 | 6 | 33.48945 | 63.95% / 67.69% | 2.45 | 50.12% |
| 5<r≤8 | 102 | 32.55471 | 32.23% / 39.74% | 9.63 | 18.91% |
| 8<r≤12 | 648 | 30.40058 | 38.33% / 34.19% | 6.81 | 32.99% |
| r>12 | 11,301 | 26.10900 | 24.64% / 26.90% | 16.48 | 21.37% |

The 8–12 shell now agrees with the independent targeted reference: its
ratio is 1.126, a difference of 0.26 combined observed row SE. The 5–8
repeat is still only 0.525 of its targeted reference, with a difference
of −2.04 observed row SE (−1.61 population SE). These highly concentrated
weights do not support Gaussian significance claims. Agreement in one
shell does not establish global convergence.

The remote-tail estimate is 1.879 times the pilot estimate, despite the
smaller relative error. Its first 4, 8 and 16 population estimates were
log Q=25.68635, 25.63644 and 26.10900. Thus doubling the final prefix did
not simply narrow its error: it found new weight and increased row RSE
from 22.75% to 24.64%. Paired-cloud noise supplies only **8.34%** of the
final observed tail variance. Further reduction of Poisson noise alone
would leave most of this variance intact.

The tail point estimate is approximately **6×10⁻⁵** of the earlier fitted
full-native estimates. This is not an upper bound and is not added to an
estimate that already covers the tail. The repeat full-native estimate
has log Q=34.13335, 36.76% row RSE and weight ESS 7.40; its six core hits
remain insufficient to replace the fitted-guide estimates near 35.8.

The comparison reconstructs the original sufficient moments from the
audited equal-size population summaries, including off-mask zeros, paired
noise and maximum contributions. It checks all final moments against the
all-row audit and uses shared-draw covariance for prefix differences.
Previously completed shell audits are verified by archived hashes and
target identity, rather than replayed. No estimates are pooled across
different proposal laws or historical populations.

Artifacts:

- `runs/native-tail-repeat-audit-20260921/analysis.json`: all-row audit.
- `runs/native-tail-repeat-comparison-20260921/analysis.json`: independent
  pilot, finite-shell and fixed-prefix comparisons.
- `runs/native-tail-repeat-figure-v2-20260921/native-tail-repeat.png`: shell
  controls and nested remote-tail estimates.
