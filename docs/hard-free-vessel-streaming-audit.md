# Full-vessel audit with bounded batches

[`audit_hard_free_vessel_streaming.py`](../tools/audit_hard_free_vessel_streaming.py)
audits a completed schema-6 hard-free vessel population in batches. It preserves
the physical measure, complete proposal density, original wall, unconditional
attempt denominator and invalid zeros. It launches no sampler and performs no
native classification. The existing auditor and frozen campaign closures remain
unchanged.

The original density/Jacobian, conditional-interval, generation-metadata and
Poisson-count routines process every batch. Temporary row copies change only the
local index required by those routines; original draw IDs are checked against
the attempted-draw journal before batching. Rejection and point counters are
summed and compared with the original whole-population summary. The direct
transformed-atom wall predicate loads the shape once, then checks every pose.
Core and exclusion-contact checks retain the existing exact/certified geometry
implementation and its strict boundary convention.

The new streaming reducers preserve importance ESS, relative standard error,
largest contribution, top-one-percent contribution and two-cloud variance
diagnostics. All attempts enter every region's denominator. Online rescaling
avoids exponentiating absolute log weights; the central-moment update avoids
subtracting two large second moments. Negative estimated residual pose variance
is retained. Unobserved regions remain unresolved, not certified zero mass.

Memory holds at most `--batch-size` full reconstruction traces (default 64),
plus two heaps of `ceil(N/100)` scalar log weights for each of six reporting
regions. The heaps retain the exact observed top-one-percent statistic. Thus
memory is not strictly independent of N, but full geometry traces no longer
accumulate with N. Shape, model, source and executable buffers are independent
of population size. This is a structural bound, not a measured runtime speedup.

The output directory contains `geometry.jsonl`, one record per attempted draw,
for reuse by a later once-only native classifier. `analysis.json` binds these
records, the input rows and journal, source closure, executable and embedded
source bundle. Incomplete audits retain their partial geometry records and a
failed status; they emit no successful analysis. A subsequent campaign must
still bind the receipt to its predeclared executable, physical inputs and
allocation.

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/audit_hard_free_vessel_streaming.py \
  --directory /absolute/completed/population \
  --binary /absolute/frozen/basin-normalizer \
  --out /absolute/fresh/audit --batch-size 64
```

Six reducer tests cover array-estimator equivalence, extreme logs, missing or
extra attempts, sparse/all-zero contributions and negative residual variance.
Six audit tests cover ordered journals, batch dispatch and global counter
corruption, anisotropic union walls, exterior/invalid accounting, complete
entry-point equivalence across batch sizes, and failure preservation. The
entry-point fixture uses seven hand-authored sphere poses and zero-activity
counts; its inert binary tests provenance plumbing and is never executed.

The supplemental reduction check reused all **16,640** already independently
audited toy draws, across both analytic populations and both implementation
controls. All six reporting regions and paired-cloud statistics agree with the
saved array reductions; maximum scaled numerical difference is **5.60×10⁻¹⁵**.
No density, geometry, native classifier, sampler or Poisson calculation was
repeated. The result is
`results/hard-free-vessel-streaming-validation-20261002/reduction.json`.

This validates the batching and reductions. It does not certify random-stream
independence, exact thinning/envelope construction, floating-point execution or
missing-mode coverage. The regional convergence and full-vessel production
gates remain closed. A schema-4 baseline adapter and native/pocket partition
integration were initially outstanding; baseline support is now available as
described below. Native/pocket partition integration remains outstanding.

## Baseline arm and matched logging

[`audit_vessel_baseline_streaming.py`](../tools/audit_vessel_baseline_streaming.py)
applies the same batching, wall/core/contact predicates and streaming moments
to schema 4. It accepts `--region` solely as a reporting chart, without loading
or evaluating a regional proposal guide. Changing this chart or its capture
radius cannot change the baseline density or weights. Its geometry output
omits the guide-specific `latent_zero` label.

All full-wall Rust normalizers now record the attempted-draw journal, completion
hashes and handled failures. Full-wall sample records are flushed per draw;
plain non-wall normalizers keep their previous behavior. There is no change to
proposal or random-stream code and no restart/power-loss recovery claim.

The frozen logging comparison in
`results/normalizer-wall-journal-parity-20261002` completed three new 128-draw
controls using intentional same-seed replay of the updated executable. The
old executable was not rerun. Baseline, Gaussian-guide and hard-free sample
files are each byte-identical to their archived counterparts, including every
pose, invalid zero, density, weight and auxiliary count. The new journals and
summary hashes also pass. This is a compatibility check, not three additional
independent physical estimates.

The new baseline control then passed one complete independent audit of all
128 attempts with batch size 32. Three deterministic tests check reporting-chart
independence, missing journals/mismatched shapes, and rejection of an extra
weight factor while preserving earlier records. The isolated executable is
SHA-256 `5ebc7599e2bfe457da0aa46ccf99ab0fc789eb21b604b5fcc31226036bf979d6`,
with embedded source bundle
`0d012425681497200a3d60967c5fd9a652e6e4e5fc0b79ba5d2d9b8053128859`.
The running assembly executable is unchanged.
