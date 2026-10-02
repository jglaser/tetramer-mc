# SMC density audit: narrow-interval conditioning

The four physical populations in
`/vast/xvg/tetramer-mc-runs/hard-free-protein-smc-control-20261002`
finished without retries. Independent full audits passed for r00 and r01;
r02 and r03 stopped during initialization-density reconstruction. The
controller drained its children and the queued contact analysis did not start.
No SMC region-weight conclusion is available from this execution yet.

A saved-data diagnostic scanned all 1,048,576 initialization records and
reconstructed the 32 smallest positive interval-width poses in each population.
Selection used geometry, with draw index breaking ties. It generated no poses,
clouds or native labels. This targeted diagnostic reproduced both failure types:

| Witness | Failing comparison | Difference in log density | Largest endpoint difference on that axis |
|---|---|---:|---:|
| r02, draw 133859 | Translation axis 0 | +2.4732×10⁻⁸ | 8.71×10⁻¹⁴ Å |
| r03, draw 247143 | Complete three-axis mixture | −2.3444×10⁻⁸ | 4.23×10⁻¹³ Å on axis 0 |

The SMC auditor's absolute log-density tolerance is 2×10⁻⁸, with its original
small relative allowance. The independent endpoint comparison already permits
1×10⁻⁹ Å. These tolerances are not equivalent when the conditional interval is
very narrow. For Normal interval mass \(M\),

\[
\delta\log M\simeq
\frac{\phi(z_b)\delta b-\phi(z_a)\delta a}{\sigma M}
\simeq\frac{\delta b-\delta a}{b-a}.
\]

Recomputing the Normal masses on the **same saved intervals** reduces the
two failing axes' Rust/Python differences to 3.20×10⁻¹⁴ and −2.10×10⁻¹³.
Across the 128 selected queries, the diagnostic separately records arithmetic
and geometry-induced differences; it is not a full replacement audit.

An independent Decimal calculation imports the saved binary64 values exactly
and normalizes the frozen line quaternions in high precision. Each of the four
controlling endpoints has a unique nearby atom-pair candidate. The 80- and
160-digit roots agree within 6.3×10⁻⁷⁷ Å. Rust's endpoints differ from these
frozen-line references by at most 1.59×10⁻¹³ Å. Independent thin-interval Normal
integrals, checked at both precisions and two series orders, differ from the
Python masses by at most 6.15×10⁻¹³ relatively for the selected dominant
components. Both witnesses also pass the unchanged generation-versus-scoring
checks, with inverse-CDF errors below 5.82×10⁻¹².

This identifies sensitivity to geometric roundoff in the observed failures.
It does not certify the complete BVH, pose-to-chart conversion, floating-point
execution, global normalization error or statistical convergence.

The diagnostic source, allocations, selected rows, hashes and oracle results
are archived in
[`results/smc-density-failure-diagnosis-20261002`](../results/smc-density-failure-diagnosis-20261002).
The original failed logs and frozen campaign remain intact. A subsequent
implementation audit must still check every attempted row and the full
accepted/rejected history before the predeclared contact analysis can proceed.

## Supplemental audit

[`conditioned_density_audit.py`](../tools/conditioned_density_audit.py) now checks
each complete density record in three parts. It retains the existing 1×10⁻⁹ Å
geometry tolerance and 2×10⁻⁸ log-arithmetic tolerance. For matched intervals,
intersect corresponding endpoints to obtain a lower Normal-mass bound and
take their enclosing intervals for an upper bound. Union these sets before
integration. Positivity of the Gaussian/axis/defensive mixture and monotonicity
of \(1/M\) give bounds on its complete density. The bounds describe sensitivity
between the two supplied floating-point geometries, not the unknown exact-real
geometry. A separate 2×10⁻⁷ log-envelope cap uses the preexisting physical
integration reference tolerance; it does not replace the tighter arithmetic
check or certify a global normalization error.

Both coordinate reconstructions must agree on R4 support. Every axis must agree
on query membership, and every Gaussian component must agree on fallback.
An envelope crossing a support boundary or fallback threshold fails. Jacobians,
complete mixture denominators, saved arithmetic and generation/inverse-CDF
checks remain explicit. The default SMC audit is unchanged; the new mode is
selected by `--conditioned-density-audit`.

The optional SMC audit independently certifies the saved-law scalar density
before using it in the bridge/history checks. Certificates are tied to exact
poses; inconsistent duplicates fail. They remain available through all accepted
and rejected mutations, then are pruned to retained endpoints after the stage
has been verified. No physical weight, move, cloud or trajectory is modified.
Failures now include the population phase and draw/stage identity.

All 25 focused tests pass, including support/fallback discontinuities, tampered
Jacobians/densities, missing components, overlapping interval hulls, stale
certificates and stage pruning. All 128 archived diagnostic poses pass the new
check. Their largest same-input log-arithmetic difference is 4.26×10⁻¹²; the
largest propagated log-envelope width is 3.73×10⁻⁸. See the
[witness receipt](../results/smc-conditioned-audit-validation-20261002/witnesses.json).
These purposively selected poses do not measure sampling efficiency or a
population-wide error bound.

Two supplemental full audits are running in
[`results/hard-free-protein-smc-supplemental-audits-20261002`](../results/hard-free-protein-smc-supplemental-audits-20261002).
Plan SHA-256:
`0f0e0d61104f677b14405deb1aa5371aa7e3adf152d5a0aca37273f81c646cdf`.
The frozen source passed all 25 tests before launch; the audit source contract
binds 26 files. Controller PID 964060 started audit PIDs 964061/964062, each
observed live with one thread. Read its `status.json` for current state.
It reuses successful audits r00/r01 and repeats only the incomplete audits of
saved r02/r03 data. Original failed receipts remain untouched. No physical
job or classifier is launched, and this receipt alone cannot release any
scientific convergence gate. The first launcher preflight counted its own
unlimited numerical-library threads and exited before creating a child;
the launch receipt records that error and the corrected import-time limits.
