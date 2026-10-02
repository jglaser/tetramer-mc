# Full-vessel comparison with the hard-free guide

The new inert preparation is
`/vast/xvg/tetramer-mc-runs/hard-free-vessel-preparation-20261002`.
Its plan SHA-256 is
`5a449d483f6791d288e46ef814ae61ad3105d189d527f5e6142f75792a863fec`.
No physical jobs have been launched. The previous 80-component preparation is
unchanged. The active regional campaigns must finish their independent audits
and sensitivity assessments, and the matching SMC discrepancy remains part of
the decision about proceeding.

The target remains the repaired rigid tetramer, two fixed neighbors, depletant
radius 1.5 Å and activity 0.035 Å⁻³, the original 223.3261767 Å atomic wall,
273 Å center capture, and a wall-permeable ideal bath. The proposal compares the
original vessel law against a normalized half-vessel/half-guide mixture. The
guide has the frozen 92 components, 50% defensive component and complete xyz
hard-free conditioning. Source R4/capture conditions the guide, not the target.

Two independent clouds are used per valid pose, with λ/z=128 in **both arms**,
matching the principal regional comparison. This differs explicitly from the
unlaunched historical vessel preparation's 64. It changes the auxiliary
estimator's noise, not the physical bath. The currently running λ64 regional
control retains its original allocation and purpose.

| Stage | Populations per arm | Draws per population | Total draws |
|---|---:|---:|---:|
| Standard | 4 | 65,536 | 524,288 |
| Large, fresh streams | 4 | 262,144 | 2,097,152 |

All 16 streams and both allocations are frozen before sampling. Stages are
analyzed separately, without pooling. Global limits remain eight physical jobs,
four independent audit workers and 32 total scientific workers. A future
dispatcher must respect other live simulations, preserve every attempted draw,
stop launching on failure and drain started children without retries.

The preparation reuses the completed analytic references and exact journal
compatibility checks. It freezes the checked executable, its embedded source
bundle, 37 Python source files, the complete native observer, all four reporting
pockets and the exact-rational R4 containment bound. Preparation validation
passed without executing a sampler. Three deterministic tests cover allocation,
CLI roles, no-overwrite behavior and target/proposal/budget drift.

## Statistical accounting

[`compare_streaming_vessel_statistics.py`](../tools/compare_streaming_vessel_statistics.py)
reuses the existing global mass covariance and paired native–competing contrast.
It validates the new streaming partition's sufficient statistics, all original
attempt denominators, regional native decomposition and every fixed stratum.
Four independent *linear* population means determine each arm's uncertainty.
Between-arm mass comparisons require both ≤3 combined linear standard errors
and ≤0.2 difference of log means. Free-energy contrasts retain paired
within-population covariance and Student-t intervals with three degrees of
freedom.

Every radial, angular and orthant bin is retained, including empty bins.
Materiality means an observed contribution ≥1% of its own regional parent in
either arm. A large native mass elsewhere in the vessel cannot hide an unstable
within-R4 stratum by diluting that fraction. Outside-R4 and outside-pocket native,
contact and unbound contributions retain their own uncertainty and comparison
diagnostics. Unobserved remainders remain explicitly unresolved.

The existing RSE≤10%, importance ESS≥200, largest contribution≤2%, contrast
half-width≤0.5 and agreement thresholds remain visible. Material remainder
contributions receive their own quality/agreement checks. Individual stratum
quality diagnostics are reported; the predeclared stratum stability check uses
between-arm agreement. Passing these observed diagnostics does not certify
absence of unseen modes or establish finite-system assembly.

Five deterministic mass-table tests cover exhaustive sums and bins, unconditional
zeros, linear rather than logarithmic SE comparisons, covariance cancellation,
independent streams, separate stages and the regional materiality denominator.
This module is arithmetic only: it does not authenticate file provenance.
An authenticated loader and a reviewed bounded dispatcher are still needed
before this preparation can run. `dispatch_ready` therefore remains false.
