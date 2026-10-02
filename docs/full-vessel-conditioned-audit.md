# Optional conditioning audit for the full vessel

The independent vessel auditor now supports `--conditioned-density-audit` in
both the ordinary and streaming entry points. The option checks the saved
conditional-interval law before handing its density to the unchanged outer
mixture and importance-weight checks. It does not substitute the independently
rebuilt interval law into the physical denominator. Generation and inverse-CDF
checks continue to use the independently rebuilt geometry.

This extends the [regional numerical audit](smc-density-conditioning-diagnosis.md)
to the full physical domain. The default strict auditor and every frozen
campaign remain unchanged. No physical sampling or native classification was
run for this extension.

The new domain cases are:

- Exact orientation-chart seams retain undefined latent coordinates and a
  zero guide contribution. The full vessel proposal supplies its own support.
- Outside the reference ball, a negative-infinite Gaussian log is allowed
  only when its multiplier is zero. Every active component must still have
  a finite log. An underflowed positive tail cannot be called a structural zero.
- Large finite latent coordinates use the existing vessel coordinate tolerance,
  `2e-8 + 2e-11 |u|`, while density, Jacobian, support, interval endpoint and
  fallback checks remain separate. The regional default keeps its existing
  absolute coordinate tolerance.
- Streaming reduction sums every attempted check and takes every reported
  numerical maximum across batches; it rejects unhandled result fields.

The numerical checks still distinguish same-saved-law arithmetic, coordinate
sensitivity, and interval-geometry sensitivity. The last uses intersections
and hulls of the two supplied interval families, with the existing 2e-7 absolute
log-envelope cap. This is not a rigorous enclosure of exact-real geometry.

## Completed validation

Forty-five focused tests pass from a frozen source closure. Five new tests
exercise exterior zero/fallback behavior, exact and near seams, rejection of
nonfinite active components, retained invalid zeros, use of the saved law in
the outer importance denominator, and complete streaming aggregation across
two batch sizes. A deliberately perturbed narrow interval passes the split
arithmetic/geometry check while the default strict comparison still rejects.
The synthetic streaming fixture is explicitly an inert binary/hash fixture,
not a physical simulation.

A [separate five-test supplement](../tools/test_full_vessel_conditioned_extremes.py)
also passes: it checks mixed finite-active/negative-infinite-inactive Gaussian
components, rejects activation of an unrepresentable component even when the
total mixture remains finite, and exercises actual enabled-conditioner Gaussian
tails close to the chart seam. Coordinate perturbations that pass the coordinate
tolerance but fail density sensitivity, Jacobian tampering and active log
overflow are rejected. These tests use the unchanged frozen auditor sources;
their [receipt](../results/full-vessel-conditioned-audit-20261002/extreme-tests/validation.json)
is separate from the original 45 tests and saved-data audits.

The optional auditor then checked all **16,640 saved Rust-generated sphere
draws** from the existing vessel references, without rerunning a sampler:

| Saved population | Draws | Valid outside R4 | Structural guide zeros | Audit CPU seconds |
|---|---:|---:|---:|---:|
| Activity 0 | 8,192 | 2,126 | 19 | 27.21 |
| Activity 0.4 | 8,192 | 2,173 | 16 | 27.24 |
| Pure uniform guide | 128 | 15 | 55 | 0.30 |
| Unconditioned Gaussian limit | 128 | 34 | 0 | 0.32 |

Every audit completed, preserving all original attempted denominators and hard
zeros. The new checks retain the previously reported large near-seam coordinate
sensitivity: one component log differs by 4.61e-5, within the existing
magnitude-relative log tolerance. Its same-input log error is 5.69e-14.
Interval endpoints differ by at most 1.39e-13 Å; the largest interval-geometry
log envelope is 1.43e-13. These are numerical observations on the saved sample,
not bounds on unseen protein poses. Exact seams and nonfinite inactive-component
cases are covered by deterministic tests, not claimed to occur in these random
reference draws.

The [frozen plan](../results/full-vessel-conditioned-audit-20261002/plan.json)
binds 34 Python files, original inputs and executable hashes. The
[completion receipt](../results/full-vessel-conditioned-audit-20261002/completed.json)
reauthenticates the sources and all historical inputs after execution. Plan
SHA256 is `f2bc3af7c86e9d3b2b42fb324de016c894b311cfd6699d81bb2ed8f767b0df40`;
completion SHA256 is `27290caf80a4b5d560a01730ef564045c4d7fa8d983191cf405fbd4a47bccaa4`.
The frozen test log also records the expected warning from an intentionally
invalid synthetic Gaussian. Earlier development import/mock-call failures
occurred before this closure and are recorded in the plan.

The inert protein vessel preparation still pins its earlier strict auditor.
Using this option requires a new authenticated execution closure and commands;
this report does not authorize or open that dispatch. Regional convergence,
the matching SMC discrepancy and unresolved contact-region coverage remain
scientific prerequisites. The running saved-data protein audits use their
unchanged archived code.
