# Deterministic proposal-density probe

`examples/proposal_density_probe.rs` loads an atlas through the production
`FrozenRelativePoseProposal` parser and scores a fixed list of relative poses.
It performs no random draws, particle geometry queries, depletion calculation
or MC updates. It supports mixed ordinary and exact reciprocal components.

Each probe is an entry in a JSON array:

```json
[
  {"label": "declared-0", "pose": {
    "position": [2.0, 0.1, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]
  }}
]
```

Positions and rotations are already relative to the fixed anchor. Quaternion
order is `w,x,y,z`. Labels must be unique. Freeze the entire panel and its
selection rule before scoring; neither this tool nor its comparison script
selects successful poses.

```bash
/path/to/proposal_density_probe \
  --model model.json --expected-shape-sha256 SHAPE_SHA256 \
  --probes probes.json --expected-probes 131 \
  --expected-components 3072 --expected-branches 4096 \
  --transport-factors --out fresh-report.json
```

The example counts above describe the expanded atlas; substitute the actual
declared counts for another model. `--cube-side` and `--uniform-probability`
default to `1000` and `0.5`, satisfying the production open-loader contract.
The reported density is **learned `G` alone**, relative to translation volume
times normalized rotational Haar measure. It excludes the uniform component
and every physical factor, so its value is independent of these two options.
The expected shape hash binds atlas provenance without constructing a shape.

The report preserves every supplied probe and records input hashes, normalized
component weights, reciprocal flags and virtual branch inventory. Each score
has a `finite`, `negative_infinity` or `error` status; a null JSON number is
never silently interpreted as zero log density. Completed per-pose errors
produce a complete failed report and nonzero exit status. Invalid input or
loader failure may terminate before a report exists; the supervising runner
must preserve stderr and record/drain the process.

## Optional transport-factor check

Direct-model and transport scores are separate. The latter reconstructs the
model's effective covariance through `component_parameters()`, then prepares
the actual `FixedBasinInvolution` factors. This extra factorization can matter
for ill-conditioned inherited covariances.

The probe constructs all **base** charts with just one dummy self-pair. It
then evaluates all virtual branches, applying exact rigid inversion to the
pose for reciprocal branches. Duplicate virtual charts would have identical
factors, so reuse of their base chart reproduces the full production map's
density without its quadratic pair table. The self-pair is never drawn. This
does **not** validate the production pair-selection distribution or an MC
acceptance rule.

`tools/audit_proposal_density_probe.py` compares baseline and exported Rust
reports against independently evaluated Python densities. It uses the existing
scalar direct-factor reconstruction and exact-rational factor residual
certificates, followed by the ordered `LLᵀ` reconstruction and map-specific
Cholesky recurrence for the transport reference. SciPy supplies the independent
pose/Cayley and triangular-solve calculations. Source excerpts are bound to
the reviewed Rust build. Plain LAPACK covariance refactorization is not used
as a substitute for either implemented law.

```bash
python -B tools/audit_proposal_density_probe.py \
  --baseline-model baseline.json --exported-model exported.json \
  --baseline-report baseline-report.json --exported-report exported-report.json \
  --probes probes.json --expected-probes 131 \
  --source-bundle /path/to/held/source-bundle.json \
  --absolute-tolerance 0.000002 --out fresh-comparison.json
```

Freeze the tolerance before execution. Every discrepancy is retained; failure
does not license a wider threshold. Agreement at a finite panel is not a
global error bound, a normalization quadrature or an equilibrium certificate.

## Held build and validation

The initial build is in
`results/proposal-density-probe-validation-20261004/attempt01/`, with a complete
frozen source closure, manifest, logs and execution receipt. It used a fresh
isolated target, offline locked dependencies, two CPU affinity slots, Cargo
jobs two and one codegen unit. No protected production target was written.

Three synthetic Rust tests passed: explicit mixed-reciprocity density
reconstruction, agreement of the linear-storage map probe with a tiny full
`DockingProposal`, and invalid-pose/nonfinite reporting. A two-pose synthetic
CLI smoke test also passed. The build and checks used **74.6 CPU seconds and
54.0 wall seconds**, with sampled peak process-group RSS **1.32 GB**; all
children were reaped and process groups drained. The limits were 600 CPU
seconds, 1200 wall seconds and 4 GiB. This receipt does not cover a later
scientific probe panel. The independent Python comparator has a separate
synthetic validation in
`results/proposal-density-reference-validation-20261004/attempt01/`: two tests
passed in 0.568 CPU seconds and 0.636 wall seconds, with the process group
drained. They check the completed Rust fixture, deliberate density mismatch,
substituted probe labels, finite/negative-infinity/error statuses and the fixed
tolerance. No scientific panel was evaluated by either validation.

Executable SHA-256:
`78d89c9d588c08bfd445a3ffd4a78e9d2ddc4d3c54568081be37d94a0f8f2c74`.
Receipt SHA-256:
`467f33bd06c3033b6d2d5d88e27d6163cff59a80d1392518ee3d43fddced5b98`.

The loader/density path uses linear storage. In contrast,
`DockingProposal::new` would create 8,390,656 unordered pairs for 4096 virtual
branches: 256 MiB of logical pair/CDF payload before allocation capacity,
charts and temporary duplicate-detection storage. This probe avoids that
precomputation; it makes no claim about the cost of subsequent MC sampling.
