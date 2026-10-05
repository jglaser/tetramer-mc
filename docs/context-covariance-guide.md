# Frozen correlated source-guide control

The [isotropic source guide](context-source-guide.md) recovered nine complete
source-contact patterns in 8,192 unconditional draws. Every valid source-branch
draw retained that pattern, but almost all source draws clashed. The
[saved local trajectories](source-cage-covariance.md) supply a narrower and
correlated description of the occupied pocket. Their poor mixing prevents an
equilibrium interpretation, but does not prevent using their moments to define
a frozen importance proposal.

## Matched comparison

Both arms retain exactly the same proposal mixture,

\[
q(x)=\tfrac12 U(x)+\tfrac14G_C(x)+\tfrac14g_S(x),
\]

full atomic-wall domain, fixed neighbors, source-centered Cayley chart and
T16/A/B/Other/unbound partition. Only the covariance of `g_S` differs. Both
arms use the same mean and coordinate variances. The full arm retains all
translation–rotation and within-coordinate-block correlations; the diagonal
arm removes those correlations. No feasibility-conditioned retries occur.

The mean and empirical covariance use all 20,480 production retained states
from training streams 0–1, including rejected residence. Streams 2–3 do not
enter the fit. In coordinates scaled by

\[
S=\operatorname{diag}(0.1,0.1,0.1,
\ell\tan(\pi/720),\ell\tan(\pi/720),\ell\tan(\pi/720)),
\]

let \(\Sigma\) be the empirical covariance with denominator N. The frozen rule is

\[
\epsilon=\max(10^{-10},10^{-6}\operatorname{tr}(\Sigma)/6),\qquad
C_F=S(\Sigma+\epsilon I)S,\qquad C_D=\operatorname{diag}(C_F).
\]

There is no bandwidth search. This small, explicit ridge belongs to the
proposal definition; the sampler adds no covariance repair. The
[fit plan](../results/context-covariance-guide-preparation-20261005/fit-plan.json)
was saved before reviewing the reduced moments. The resulting
[full](../results/context-covariance-frozen-guides-20261005/full.json) and
[diagonal](../results/context-covariance-frozen-guides-20261005/diagonal.json)
guide assets record their training evidence and fit-rule hashes.

A subsequent descriptive [held-out moment score](../results/context-covariance-guide-heldout-moments-20261005/report.json)
uses these already frozen parameters without refitting. For a chart mean \(m\)
and covariance \(C\), the mean squared Mahalanobis distance is
`tr(C^-1 Cov(x)) + (E[x]-m)^T C^-1 (E[x]-m)`. It is about 6 in the training
sample and 25.71 / 26.13 in the full / diagonal held-out controls. The full
source component improves mean held-out log density by 1.53 nats per retained
pose (1.32 in training). The common coordinate-to-Haar Jacobian cancels in this
difference. This is a component score, not the complete defensive-mixture
score or a physical free energy. Retained observations remain correlated, and
the large held-out distances still expose limited coverage. This score changes
neither frozen guide nor the prospective allocation.

The fresh allocation is four independent populations of 2,048 draws per arm,
16,384 draws total. The independent audit reconstructs every recorded source
density, complete mixture sum and latent decode, plus a fixed 512-pose panel
of full atlas densities and atomic geometry. No depletant clouds are allocated.
This comparison tests coverage and hard-only importance weights, not physical
free energies or Markov-chain mixing.

## Implementation and correctness

`context_source_guide` accepts an optional versioned `explicit_gaussian` object
with a six-dimensional mean and provenance string. The covariance remains a
full stored 6×6 matrix. Explicit charts omit legacy isotropic widths; legacy
charts still require those widths. Ambiguous inputs, nonfinite values and
non-positive-definite covariances fail rather than being repaired.

The unchanged map uses direct strict Cholesky factorization and the normalized
Haar Jacobian. Every draw is scored under the complete mixture. The hard-only
regional estimator is the sum of `H_wall H_core / q`, divided by **all attempted
draws**. Hard-invalid draws remain zero. A future physical estimator must
include the absolute depletion weight in the numerator and retain the same
unconditional denominator.

Twelve focused Rust tests passed in an isolated build, including exact legacy
JSON/seed/density compatibility, nonzero-mean correlated transforms and analytic
Haar density checks. Nine Python density/legacy-audit checks and eight new
fit/audit checks also passed. The new audit rejects changed guide assets,
changed populations, missing attempts and unallocated clouds. The protected
production executable and previous source/build snapshots remain unchanged.

The Rust build used 104.98 CPU seconds including its controller, 76.34 wall
seconds, and drained both child groups. The covariance reducer reused saved
data; it made no geometry or cloud calls. These validations establish the
implemented proposal law, not adequate coverage of unknown important regions.

Implementation: [runner](../examples/context_source_guide.rs),
[guide exporter](../tools/prepare_context_covariance_guides.py),
[independent density](../tools/source_guide_reference.py),
[independent audit](../tools/audit_context_covariance_guide.py).

## Completed comparison and decision

All 16,384 proposals completed. The independent audit passed 73,722 scalar
checks, every recorded pose decode, and the fixed 512-pose atlas/geometry panel.
The full pattern was retained by every hard-valid source-branch proposal in
both arms. No other branch produced that complete pattern in this allocation.

| Diagnostic | Full covariance | Diagonal control |
|---|---:|---:|
| Valid source proposals / source draws | 1,093 / 2,081 (52.52%) | 588 / 2,129 (27.62%) |
| Complete-pattern hits / all draws | 1,093 / 8,192 | 588 / 8,192 |
| Geometry CPU seconds | 73.50 | 83.25 |
| Population hard-only importance ESS range | 4.35–13.57 | 1.29–28.67 |
| Largest regional contribution range | 14.75–45.99% | 11.57–87.89% |
| Mean hard-only region mass, Å³ × normalized Haar | 1.02e-11 | 3.68e-11 |
| Between-population relative standard error | 6.20% | 52.32% |

Correlations improve hard feasibility by 1.90× with the same center and
marginal widths. The complete-pattern proposal rate per CPU rises by 2.11×.
These are proposal-coverage results, not accepted physical transitions or
effective independent contact samples. Every population fails both the
regional ESS ≥200 and largest-contribution ≤2% checks. The apparently stable
full-arm population means therefore do not establish convergence. The two
mean geometric masses differ by a factor of 3.61, with the diagonal arm too
uncertain to interpret this as a statistically resolved bias.

![Coverage improves while importance-weight concentration remains poor](../results/context-covariance-guide-independent-audit-20261005/figures/covariance-guide-coverage.png)

Evaluating two clouds at every valid pose in this saved allocation would
require about 1.07 billion raw points in expectation under the current
envelopes. **No such cloud allocation was launched.** The geometric weight
concentration already demands better coverage. The next saved-data diagnostic
combines only these two simultaneously frozen proposal arms with their exact
equal-mixture denominator and locates the contributions in predeclared
Mahalanobis-radius strata. This asks whether their complementary coverage
helps and identifies tails for a later frozen proposal. It generates no new
poses or geometry and cannot supply physical free energies.

Evidence:

- [Geometry completion](../results/context-covariance-guide-geometry-20261005-v2/controller/receipt.json),
  SHA256 `9ec7c4c5bed3eb0cc32a6e99e20957e6c21d2b808b2a2d6cc224ce011b77bbb1`.
- [Independent audit](../results/context-covariance-guide-independent-audit-20261005/result/report.json),
  SHA256 `9d00efd1ef282dd8bad3db0f036d431eb0ade704f9b18d7367fd9cddf76a0e40`.
- [Derived per-population diagnostics](../results/context-covariance-guide-independent-audit-20261005/figures/report.json)
  retain their source and audit hashes; [plot/reducer](../tools/plot_context_covariance_guide.py).

Geometry used 156.75 CPU seconds and 46.95 controller wall seconds; the
independent audit used 13.79 CPU seconds. All thirteen producer groups and
the independent audit child drained. A prelaunch scheduler check caught a
legacy requirement for sixteen jobs. The first preparation remains unlaunched;
version 2 changed only the two job-count admission assertions to eight jobs
in four lanes. Correct, duplicate and incomplete inventory checks passed
before dispatch. No scientific draw was retried or discarded.

The result advances source-pocket access but leaves its statistical weight
unresolved. It neither establishes nor refutes native finite-system assembly
at 106.8 μM. The original-condition physical contact campaign remains separate.
