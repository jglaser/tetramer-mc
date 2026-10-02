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
An authenticated loader is now available as described below. A reviewed bounded
dispatcher and the outstanding regional evidence are still needed before this
preparation can run. `dispatch_ready` therefore remains false.

[`streaming_vessel_execution_plan.py`](../tools/streaming_vessel_execution_plan.py)
now supplies a pure scheduling adapter for that future dispatcher. It preserves
all 16 frozen physical, audit and partition commands and destinations. Each
stage contains eight physical jobs, eight audits, eight partitions and one
authenticated stage-reader call, in that order. The declared worker limits are
8 physical, 4 audit and 32 overall; both predeclared stages remain separate.
The reader entry point is pinned under a future workflow's `common/` directory
by its source hash, and its command includes the explicit preparation hash.

This adapter has no launch entry point or gate authority and writes no files.
It checks the existing preparation contract and exact schedule, but does not
authenticate the declared hashes or assess scientific evidence. Deterministic
tests cover unchanged commands, independent populations, stage completeness,
budgets, source/hash pins and absence of file or process side effects. A future
dispatcher must freeze and authenticate the reader's source closure, apply the
regional prerequisites, check global live-worker capacity and drain started
children on failure. The frozen preparation remains unchanged and inert.

Two execution details remain explicit: log parent directories must be created
only after the execution claim and prerequisite checks, leaving fresh child
output directories absent; live-worker accounting must recognize Python jobs
under `/vast` as well as repository paths. The existing
`prepare_hard_free_line_sensitivity_reference.capacity` function handles those
paths and rejects a private PID namespace. Declared group limits alone cannot
enforce the global limit alongside other simulations.

### Numerical readiness of the vessel audit

The frozen vessel auditor still uses the strict single-axis density comparison
in `physical_hard_free_line_vessel.audit_trace`. The
[SMC conditioning diagnosis](smc-density-conditioning-diagnosis.md) shows that
this comparison can fail on narrow intervals even when interval endpoints and
same-input density arithmetic agree. Its supplemental helper is not yet a
validated drop-in replacement for the full vessel.

The vessel permits exact exterior support zeros, positive fallback tails and
the exact Cayley seam. Its scorer requires finite Gaussian logs only for active
components; the supplemental helper currently requires all such logs to be
finite and uses an absolute latent-coordinate tolerance. An optional extension
would need separate checks for inactive negative-infinite logs, large
near-seam coordinates and exact-seam zero handling. The existing sphere fixtures
cover exterior/fallback/invalid cases, but the 128 completed narrow-interval
witnesses all lie inside R4 and do not cover these additional cases.

Any later extension must verify saved-law q/J before passing it to the unchanged
outer-mixture and weight checks, preserve the independently rebuilt geometry
for generation checks, and retain every invalid zero. It must not replace the
physical importance denominator by an independently reconstructed geometry law.
The streaming batch reducer must explicitly aggregate the additional checks.
No tolerance, source file, job command or gate in the frozen vessel preparation
has been changed; completed references remain reusable.

## Authenticated completed-stage reader

[`analyze_streaming_vessel_stage.py`](../tools/analyze_streaming_vessel_stage.py)
requires the preparation hash explicitly. It checks the exact executed proposal,
shape, wall, cloud law, journal and binary; audit/partition completion hashes;
all four frozen pocket definitions; complete native-observer inputs; and each
archived source closure. Shared statistical code must match the preparation's
frozen version. It reduces the already saved sufficient statistics without
repeating geometry, native classification or prior-stage calculations.

The output includes both full-domain and regional estimates, all original
strata, whole-population uncertainty, sampler/audit/classification CPU costs and
importance ESS per sampler CPU. This ESS is not an autocorrelation-based contact
mixing time. The reader sets no physical-stability or unseen-mode certificate.

Three deterministic tests cover both streaming formats, completed artifact
lineage, corrupt/missing input rejection and the protein manifest contract.
The successful lineage controls use the seven-pose sphere fixture with its
synthetic native observer; protein-specific constants are tested separately,
not falsely attributed to those toy poses. No new physical reference jobs run.

The eventual invocation is:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/analyze_streaming_vessel_stage.py \
  --preparation /vast/xvg/tetramer-mc-runs/hard-free-vessel-preparation-20261002 \
  --preparation-sha256 5a449d483f6791d288e46ef814ae61ad3105d189d527f5e6142f75792a863fec \
  --stage standard --out /absolute/fresh/standard-comparison
```

This command requires all eight populations, audits and partitions of the
selected stage to have completed. The future execution preparation must freeze
this reader and its new source files before dispatch. It cannot be used to
analyze unfinished populations or to authorize launching them.

## Analytic bound for the unbound remainder

For an unbound pose the relative ideal-depletion weight is exactly one. With
normalized SO(3) Haar measure, its integral cannot exceed the accessible center
volume. This includes the unbound contribution outside every measured pocket.

For body-frame atom centers c_a and radii r_a, the atomic wall implies

\[
\|t-c_{\mathrm{wall}}\|
\le R_{\mathrm{wall}}-\overline r+\|\overline c\|.
\]

This follows by averaging the atomic containment inequalities and applying
convexity of the norm. It requires no convexity of the sphere-union particle.
Intersecting with the center-capture domain bounds the volume by the smaller
of their enclosing-ball volumes, even if their centers differ.

[`bound_vessel_unbound_mass.py`](../tools/bound_vessel_unbound_mass.py) evaluates
this bound using exact rational input arithmetic, outward norm bounds and
π<22/7. For the frozen vessel, **Q_unbound ≤45,654,774.77 Å³**, corresponding to a
display log bound of **17.636619**. The exact rational bound is stored in
`results/streaming-vessel-stage-validation-20261002/unbound-mass-bound.json`.
It is 53.55% of the looser capture-only volume bound. The decimal/log values are
for display; the rational artifact is the upper bound.

Two tests cover centered and offset spheres, a nonconvex unequal-radius union,
capture intersections and an empty-domain case. No poses or point clouds were
drawn. This does not bound adsorbed competing contacts or provide a rigorous
lower confidence bound on native weight. It therefore does not decide assembly.
As with the R4 inclusion certificate, floating-point execution is separate from
this exact-real geometric statement.
