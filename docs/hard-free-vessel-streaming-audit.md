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
integration are still needed for the planned protein comparison.
