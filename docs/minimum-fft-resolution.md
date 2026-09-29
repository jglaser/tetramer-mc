# How much FFT resolution is useful?

`tools/assess_fft_resolution.py` audits angular resolution using the completed
FFT cache. It does not repeat any FFT, physical simulation, repair, or contact
refinement. The first audit processed all 20,779,513 peaks in 34 seconds on one
CPU, using memory-mapped arrays and bounded chunks.

The practical recommendation is to **reuse the already discovered contacts
while fixing the proposal fits**. For another coarse discovery pass, a nested
75,000-orientation subset is a reasonable candidate budget to test with exact
repair and refinement, with the existing 300,000-orientation cache retained as
a reference. It is not established as the minimum sufficient discovery budget.
There is no evidence here to justify a finer translation grid than the existing
1 Å. A coarser translation grid was not tested.

## What was measured

The reference is the frozen, native-blind list of 768 union-map starting poses.
A cached peak covers a reference if either the reference or its physical inverse
is within **4 Å translation and 10° rotation** of that peak. Both constraints are
checked, including quaternion sign equivalence. This neighborhood measures a
possible repair/refinement starting point, not membership of a physical basin.

One seeded random permutation of the 300,000 existing orientation indices gives
nested budgets of 10,000, 30,000, 75,000, 150,000, and 300,000. Every cached peak
for each retained rotation is considered, without selecting by native labels.
For each reference and budget, the diagnostic records the largest nearby cached
grid-overlap score.

The strict criterion, fixed before this audit, requires at least 95% of
measurable references to retain a peak with at least 90% of their full-cache
best score, and at least 90% retention within every quartile of full-cache best
scores. Native labels are evaluated only after that decision; they neither
select the budget nor reweight any reference.

| Rotations | References with a nearby peak | Retain ≥90% of full-cache score | Random-probe p95 nearest angle |
|---:|---:|---:|---:|
| 10,000 | 733 / 768 | 18.8% | 10.18° |
| 30,000 | 767 / 768 | 42.4% | 6.94° |
| 75,000 | 768 / 768 | 71.6% | 4.97° |
| 150,000 | 768 / 768 | 90.8% | 3.71° |
| 300,000 | 768 / 768 | 100% | 2.73° |

The strict score-preservation rule selects 300,000 among these tested budgets.
This is more conservative than retaining starting points: a near-clashing raw
FFT peak can score better than a useful hard-valid contact. Therefore one cannot
infer from a lower grid score alone that exact refinement will fail. Conversely,
the existence of a nearby peak does not prove that refinement will recover the
same contact.

The 75,000 subset retains possible starting points near every measured contact,
which supports testing a coarse-search-plus-local-refinement workflow. The next
necessary check is matched **repaired** contact recovery, rather than more FFT
precision. The two direct native-entry poses already in the reference list also
retain their score criterion from 30,000 onward; that tiny post-hoc check is not
a validation of native-family completeness.

## Geometric interpretation and limitations

A translation lattice with spacing `h` has a nearest-grid positional error at
most `sqrt(3) h / 2`: 0.866 Å at 1 Å spacing. A rotation error `theta` displaces
any atom center by at most `2 R sin(theta/2)`, where `R` is the largest atom-center
lever arm. The tool reports these displacements using the actual shape. Neither
bound controls loss of overlap in a narrow interlocking pocket. The sampled
maximum nearest angle uses 8,192 independent Haar probes; it is not a proved
covering radius.

These references were themselves found using the full cache. The audit is
retrospective candidate retention, not independent discovery validation or a
bound on unobserved contacts. The cache already excludes peaks below its
1,000 Å³ grid-score threshold; this audit cannot recover the omitted weak
interfaces. Activity-dependent basin weights and covariance validity are
separate questions.

The audited subsets are selected **from** `super_fibonacci(300000)`. Generating
`super_fibonacci(75000)` directly gives a different grid. To reproduce the subset,
use `nested_ranks(300000, 20260928001) < 75000` and evaluate
`super_fibonacci(300000, selected_indices)`. The tool writes the chosen strict
criterion's index set as `selected-rotation-indices.npy` in new runs.

## Reproduction and tests

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/assess_fft_resolution.py \
  --scan runs/fft-depletion-docking-gpu-20260926 \
  --references runs/fft-union-starts-768-20260926/initial-poses.json \
  --native-definition runs/mobile-native-region-definition-20260921/definition.json \
  --export-budget 75000 \
  --out runs/fft-resolution-audit-<fresh>

/home/xvg/protein-nucleation/.venv/bin/python -m unittest discover \
  -s tools -p test_assess_fft_resolution.py -v
```

All seven tests pass: deterministic nested indices, sign/inverse equivalence,
separate angular and translation constraints, streaming equivalence, multiple
reference coverage, missing-reference accounting, score-stratum protection, and
monotonic nearest-angle diagnostics. The audit checks complete manifests and
hashes the shape, peak shards, plans, reference poses, script, and optional native
definition. Outputs refuse overwrite; no training data or production kernels
are changed.

Completed results: `runs/fft-resolution-audit-20260928/{plan.json,report.json,
best-grid-overlap.npy,executed-assess_fft_resolution.py}`. The archived executed
script preserves the exact code of the first result; the current tool also
archives itself and writes selected indices automatically.

## Export for the next coarse-to-exact test

`--export-budget 75000` also writes `subset-basin-starts.json`, with one slot for
every reference basin. The slot contains the highest-grid-score raw peak **within the reference
translation/rotation neighborhood** in the audited subset, or explicit nulls if
none exists. This is not nearest-peak selection: the reference restricts the
candidate pool, then grid overlap chooses within it. These are **unvalidated raw starts**: an FFT
clash threshold on shrunken cores does not imply exact hard-core validity.
The export keeps all references, has no native-label filtering, records the
actual rotation indices, and is suitable for a separate exact-repair/recovery
test. It must not be passed as if it were a hard-valid proposal atlas.

The completed export is at
`runs/fft-resolution-audit-export-20260928/subset-basin-starts.json`.
It contains all 768 slots; no FFT or protein trajectory was repeated. This
reference-conditioned export tests candidate recovery, not independent blind
basin discovery: the reference centers entered the retrospective audit.

## Bounded exact-repair check (fixed allocation)

`tools/validate_fft_subset_recovery.py` separates preparation from execution.
The prepared allocation selects **32 slots at indices 0, 24, …, 744**, before
reading any native labels. These slots inherit the reference-conditioned peak
selection above; the existing reference centers came from the full 300k scan.
Thus a successful check would establish repair from those conditional starts,
not independent discovery of the same basins by a new 75k scan. The repair itself
receives only the raw candidate, shape, and seed, never the reference pose. Each slot receives the existing geometric repair
with 400 attempted steps and an independent seed. A successfully repaired pose
then receives the exact Rust hard-core check and one fresh 8,192-point union-overlap
estimate. No additional optimization or FFT is performed.

For comparison, the plan freezes each slot's archived optimized center and its
two independent 65,536-point validation estimates. It verifies the original
source hash, shape and bath radius, and equality of the source's initial pose
to the reference used in the FFT audit. The repair program, scoring binary,
controller, shape and inputs are copied with hashes before execution. The native
classifier is not an input to this workflow. Native labels, if later desired,
require a separate post-hoc report.

The summary retains all 32 allocated slots, including missing peaks, failed
repairs, and execution errors. Failed geometric repairs count as zero recovered
overlap in the unconditional overlap ratio; execution errors invalidate a
completed diagnostic. The tool records both Python-worker and child-scorer CPU.
It also reports the descriptive count reaching at least 90% of archived overlap;
this is neither a confidence gate nor a basin-volume comparison. Restoring peak
overlap is weaker than demonstrating viable proposal covariances.

The frozen plan is
`runs/fft-subset-recovery-32-20260928/plan.json`. Its execution command was
(the completed output directory refuses a second run):

```bash
/home/xvg/protein-nucleation/.venv/bin/python \
  runs/fft-subset-recovery-32-20260928/inputs/controller.py run \
  --prepared /home/xvg/tetramer-mc/runs/fft-subset-recovery-32-20260928 \
  --workers 4
```

The three additional unit tests cover fixed slot selection, quaternion-sign
matching, failed-repair denominators, uncertainty aggregation, execution-error
status and total CPU accounting. Preparation runs no physical jobs.

## Figure

`tools/plot_fft_resolution_audit.py` produces the static coverage/score-retention
figure from an already completed audit. It reads no pose data or native labels,
executes no physical jobs, and archives its source report, plan, script and hashes.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/plot_fft_resolution_audit.py \
  --report runs/fft-resolution-audit-export-20260928/report.json \
  --out runs/fft-resolution-figure-<fresh>
```

Completed PNG/SVG:
`runs/fft-resolution-figure-20260928/fft-resolution.{png,svg}`.
The annotations distinguish the provisional 75k candidate budget from the
300k strict score-retention result, and explicitly exclude a claim of hard-valid
basin recovery.

## Reproduce the audited subset in a future scan

Both FFT runners now accept `--rotation-indices FILE.npy`. The array must be
one-dimensional, nonempty, integer-valued, unique, and within `[0, --rotations)`.
The runners preserve global indices and calculate quaternions on the original
`--rotations` grid. They sort the selected indices and distribute that ordered
list across workers/devices, so each selected orientation runs exactly once.
Without the option, the original `arange(rank, rotations, parts)` scheduling and
quaternion arithmetic are unchanged.

The exact input `.npy` bytes and their SHA256 are archived in each plan. CPU
scans write `rotation-indices.npy`; GPU processes independently write
`rotation-indices-gpu<device>.npy`. The plan also records the selection count,
original input path, ascending execution order, and index-helper source hash.
The selected indices in peak outputs retain their original global identity.

The verified 75k file is
`runs/fft-resolution-audit-export-20260928/exported-subset-rotation-indices.npy`
(SHA256 `ca18bf72eec24ea919f8d20898aeb8f35074ad2c1cf354688d5a2a7bc2f8cef1`).
It contains 75,000 unique indices, from 7 through 299,998. **Keep
`--rotations 300000`** when using it.

For a future coarse scan with the same grid and peak policy as the audit:

```bash
FFT_PY=/vast/xvg/venvs/tetramer-gpu/bin/python
FFT_SHAPE=runs/native-blind-depletion-discovery-20260926/inputs/shape.json
FFT_SUBSET=runs/fft-resolution-audit-export-20260928/exported-subset-rotation-indices.npy
FFT_OUT=runs/fft-coarse-75k-new-run  # choose a fresh directory
for device in 0 1; do
  "$FFT_PY" tools/fft_depletion_docking_gpu.py \
    --shape "$FFT_SHAPE" --out "$FFT_OUT" \
    --rotations 300000 --rotation-indices "$FFT_SUBSET" \
    --rd 1.4 --spacing 1 --core-shrink .5 --separation 3 \
    --min-overlap 1000 --clash-tolerances 2 --batch 8 \
    --device "$device" --devices 2 &
done
wait
```

This command has **not** been executed; completed audits still use the original
cache. CPU scans accept the same index option but retain their existing top-
`--peaks` selection policy, which differs from the GPU all-local-maxima cache.
The new option changes neither policy. Six focused CPU tests cover invalid
arrays, exact archive/hash preservation, complete disjoint sharding, unchanged
no-option ordering, and bit-for-bit equality of subset quaternions to entries in
the full grid. No GPU scan or physical simulation was needed for those tests.

## Completed repair check and stage audit

The 32-case repair check is now complete: **32/32 hard-valid**, but **0/32** attain
90% of the final optimized reference overlap (mean ratio **0.432**). Those
references had already received two 2,000-step optimization passes, whereas the
new coarse candidates received only clash repair. That initial comparison does
not isolate FFT resolution.

The read-only lineage audit in `tools/analyze_fft_recovery_stages.py` retrieves
matching archived **initial** validation scores for the original full-scan
repaired starts. Against that earlier stage, the mean ratio is **0.493**, with
**1/32** above 90%. Thus unmatched optimization explains some, but not most, of
the gap. Full-scan ranking/NMS and GPU fine-grid perturbation preprocessing remain
unmatched, and the coarse candidate export remains reference-conditioned.
**The 75k end-to-end recovery budget is not validated.**

All 32 slots are retained. No additional physical jobs ran during the lineage
audit. Its source hashes, stage-wise comparisons and interpretation are in
`runs/fft-subset-recovery-stage-audit-20260928/{report.json,report.md}`.
