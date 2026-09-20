# Archived SMC Poisson count audit

The actual archived `SingleBodyDepletion::sample_overlap_count` agrees with
independent overlap integrals at three representative protein poses. This
closes a gap in the earlier geometry audit, which compiled the archived
membership predicates but did not call its complete Poisson count sampler.
It does not validate SMC population coverage or establish its normalizer.

The diagnostic compiles the archived `geometry.rs` and
`single_body_depletion.rs` **unchanged**, with only a minimal `mirror` module
providing the original pose/vector aliases. Dependency sources were checked
byte for byte against the old vendor tree. Each independent cloud has its own
recorded RNG seed. Original SMC datasets and previous audit artifacts remain
unchanged.

For overlap volume C, the source constructs a PPP in the moving body's
exclusion bounding ball and retains points inside both the mobile exclusion
union and the fixed exclusion union. Coverage by multiple fixed neighbors
counts once. Consequently K is Poisson with mean λC, and M independent clouds
give C-hat=sum(K)/(Mλ), with Poisson standard error sqrt(sum(K))/(Mλ).
The source's uniform rejection sampler fills the ball, and its Poisson sum
splits the requested mean into independent means no larger than twenty.
Neither construction introduces an apparent geometric approximation.

The first test used 64 clouds at λ=0.125 Å⁻³, exposure8 per pose. Its third
pose differed from the common-box/current estimates by2.31/2.74 combined
standard errors. Because runtime was only3.52 CPU seconds, an independent,
prespecified confirmation used256 clouds at the same λ for **all three**
poses, rather than retaining only favorable results. This took13.43 CPU
seconds. Both tests and all individual counts are archived.

| Pose | Common-box C ± SE / Å³ | Current-envelope C ± SE / Å³ | Archived count C ± SE, initial / Å³ | Archived count C ± SE, confirmation / Å³ |
|---|---:|---:|---:|---:|
| site0 native-07, p263 |1041.83 ±6.79|1031.58 ±5.36|1047.50 ±11.44|1045.28 ±5.72|
| site0 other-01, p463 |598.25 ±5.30|613.70 ±4.14|608.13 ±8.72|609.56 ±4.36|
| site1 other-02, p352 |268.35 ±4.11|267.13 ±2.73|285.13 ±5.97|267.53 ±2.89|

Confirmation sample-variance/mean ratios are1.004,0.997,0.971. All differences
from either independent reference are at most1.75 combined standard errors.
The initially large third-pose residual did not reproduce. At z=0.035 Å⁻³,
the confirmation comparison uncertainties are0.14–0.31 kBT. No multi-kBT
count-law disagreement was detected at these fixed poses. Approximate
chi-square dispersion diagnostics are retained as descriptive checks; they
are not proofs of the count distribution or of full SMC correctness.

Artifacts: [initial assessment](../runs/archived-smc-count-audit/assessment.json),
[confirmation assessment](../runs/archived-smc-count-confirmation/assessment.json),
and each directory's `provenance.json`, input, exact copied source, executable,
raw counts and SHA-256 artifact manifest. The previous references are in
[the physical geometry audit](old-new-geometry-audit.md).

Reproduce in a fresh directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/audit_archived_smc_counts.py \
  --out runs/archived-smc-count-reproduction --clouds 256 --seed 901759331
```

The implementation is [the isolated Rust probe](../tools/archived_smc_count_probe.rs)
and [its launcher](../tools/audit_archived_smc_counts.py). The launcher enforces
a90-second CPU limit for the numerical subprocess. It does not modify any
production sampler.
