# Actual-center core scoring

`tools/context_core_score.py` evaluates the core geometry of existing pose
centers against a fixed outside environment. It provides a reusable
`ExactPointScorer` and a bounded campaign CLI. The implementation was promoted
from the completed October 4 diagnostic with only its module imports changed.
It does not run an optimizer or change any pose, proposal or physical state.

## Observable and coordinate conventions

For moving atomic centers `a_i`, fixed centers `b_j` and their physical radii,
the diagnostic uses

\[
g_{ij}=|a_i-b_j|-r_i-r_j,\qquad
E=\sum_{ij}[\max(0,\delta-g_{ij})]^2,
\qquad\delta=0.002\;\mathrm{Å}.
\]

This is a geometric penalty in Å², not a depletion energy. Strict core overlap
uses `distance_squared < (r_i+r_j)^2`; the margin does not change that predicate.
The returned gradient is with respect to Cartesian translation and a left
infinitesimal rotation, in radians, about the moving body origin. It is not a
gradient with respect to an arbitrary finite rotation-vector parameterization.
An optimizer using one must apply its rotation Jacobian and coordinate scales.
Coincident atomic centers make the derivative undefined. Their count is
explicit, and their directional gradient contribution is set to zero by
convention.

`OutsideContext` and chart decoding come from
`tools/context_relaxed_atlas.py`. The context contains only fixed outside
bodies, with an explicit anchor label and excluded moving labels. Fixed atoms
are transformed into that anchor's frame. Incoming poses use the same frame.
The caller must truthfully exclude every moving body. The scorer implements
neither periodic images nor a wall predicate.

## KD queries, caps and results

One KD-tree holds the fixed atoms. At each actual pose, batches of at most 64
moving atoms query radii `r_i + max_fixed_radius + delta`, enlarged only by an
outward floating-point guard. A count-only query precedes list materialization.
The entire batch must fit the remaining 65,536-candidate budget for that pose.
Individual fixed radii enter the actual distance, core and hinge calculations.
No tree for a full translation/rotation trust region is constructed.

A completed score contains:

- strict core-overlap, active-hinge and individual-margin pair counts;
- the squared-hinge objective and both gradients;
- the smallest gap found within the individual margin;
- materialized candidates, distance tests, count/list query points and CPU
  timings, plus an explicit coincident-center count.

If no pair lies within the margin, `minimum_local_gap_A` is null and
`minimum_gap_censored=true`, with censor threshold `0.002 Å`. This establishes
only the corresponding lower bound within the implemented predicates. It does
not report infinity or measure a larger gap.

On `status="candidate_cap"`, `complete_geometry=false`: all full geometry
observables and `hard_clear` are null. `partial_values` retains the scored
prefix separately. An incomplete sum is never reported as the full objective,
and a cap is neither a clash certificate nor a clearance certificate.

The count precheck bounds candidate-list materialization. The supervisor's
time and address-space limits still cover KD traversal, fixed arrays and all
other work. Floating-point guards are implementation safeguards, not a formal
proof of exact real arithmetic.

## Identities, caching and failure records

The CLI reads every virtual parent from a reciprocal source atlas, then the
corresponding ordinary child from the exported atlas. It does not select
branches by native labels, clearance, optimizer status or weight.

It evaluates all parents first. A child reuses only its corresponding parent's
result when their serialized center records agree exactly; otherwise it is
evaluated. No tolerance or cross-parent deduplication is used. Cached cap
outcomes remain capped. Every identity retains its source label weight.

`result/attempts.jsonl` is flushed at context start/completion and at each
`pose_begun`/`pose_complete`. A failure therefore preserves completed records
and the identity being attempted. An external supervisor must drain failed or
timed-out processes. There is no resume or automatic retry.

`performed_geometry` and `cache_of` distinguish computation from reuse. Cached
records carry the original score and its timings; aggregate cost includes
only records with `performed_geometry=true`. A lifecycle report can pass with
retained cap outcomes: inspect `all_geometry_complete` and the per-arm cap
counts before interpreting geometry.

## Completed fixed-context diagnostic

The repaired shape has 4,004 atoms. All 264 saved outside bodies were retained,
with anchor 16 and fictitious incoming label 264 excluded. The source contains
2,048 virtual parents and the export supplies 2,048 corresponding children.
All 4,096 identities were recorded: 2,048 parent evaluations, 18 changed-child
evaluations and 2,030 corresponding-parent reuses. The 2,066 scheduled
evaluations are not a count of distinct physical poses across parents.

All evaluations completed without candidate caps:

| Centers | Core-clear labels | Sum of source label weights |
|---|---:|---:|
| Parent | 45/2,048 | 1.83245% |
| Exported child | 58/2,048 | 2.28964% |

These fractions weight fixed centers by their source labels. They are **not
finite-width proposal coverage**, accepted-move probabilities, basin volumes
or equilibrium occupancies. The outside context was selected through a native
diagnostic; conditional use of geometric inputs does not make this an
independent blind-discovery test. No wall, native-registry or depletion test
was performed here.

The scorer used 41.37 CPU seconds and 41.59 wall seconds. It tested 21,888,886
candidate distances; query and filtering work used 27.93 and 10.51 CPU seconds.
Average score cost was 19.60 ms. Maximum candidate and active-pair counts were
35,535 and 31,553. Thus the earlier full-box candidate caps did not prevent
actual-center evaluation, although 2,003 original centers fail the
implemented core predicate. These data do not establish optimizer efficiency.

Independent saved-scalar validation passed all **473 comparisons**: 397 old
initial objectives and 76 selected-child objectives. Maximum objective
differences were `2.73e-12 Å²` and `2.27e-13 Å²`, respectively; uncensored gap
differences were at most `6.44e-15 Å`. The independent audit also verified all
4,096 identities, 2,066 evaluations, 2,030 cache links and 8,194 journal events.
Its tolerances were frozen before the geometry run; it issued no new atomic
queries.

A subsequent saved-coordinate reduction examines nearest fixed **body
origins**, without atomic queries. Twelve parent labels carrying 13.8776% of
source weight lie within 2 Å of bodies 77 or 237; all twelve previously hit
the static pair-list cap. Across the full atlas, the nearest origin belongs
to anchor 16 for 15.3741% of label mass, another quartet member for 60.9253%,
and another body for 23.7006%. The 125 labels within 10 Å of any fixed origin
carry 19.0643% of source weight. This suggests occupied locations contribute
to the problem. It does not identify the atoms causing a clash, assign native
registry, or imply that the corresponding isolated pair center is invalid.

The reduction retains all 2048 labels, exact-distance tie handling, fixed bins
`[0, 2, 5, 10, 20, 40, infinity] Å` and the previous-cap strata. It completed
and drained in 0.196 CPU seconds; its report is
`results/context-parent-origin-proximity-20261004/result/report.json`, SHA-256
`47ef9b369bb08123ef59388ae392c6d4aadea8274137099bf95c36f407ca9ea5`.
Per-body atomic attribution is still needed before choosing an occupied-site
avoidance rule or blaming Gaussian centering.

## Can the original small relaxation box repair these centers?

A further saved-data reduction bounds the displacement of **every** incoming
atom under translation norm at most 1 Å and rotation angle at most 2°.
It computes an exact-rational upper bound `U=47.9370222092 Å` from the stored
body coordinates and all parent rotation matrices. Using `sin(u) <= u` and
`pi < 22/7` gives

\[
M=1+\frac{22}{630}U=2.67399125175\;\mathrm{Å}.
\]

If a saved minimum gap has error at most `epsilon`, then
`saved_gap + epsilon + M < 0` implies that at least one atom-pair overlap
persists throughout this trust region. This conditional geometric statement
does not require another atom-pair query. It also does **not** establish a
rigorous floating-point error bound for the saved gaps.

| Predeclared gap-error allowance | Screened parent labels | Source label mass |
|---|---:|---:|
| `1e-6 Å` | 1936/2048 | 95.7525% |
| `1e-4 Å` | 1936/2048 | 95.7525% |
| `1e-2 Å` | 1935/2048 | 95.7357% |

All 1651 previously capped parents screen at every allowance. Therefore the
measured cheaper objective does not justify simply repeating the same small
repair box at a larger work budget. The next construction needs different
destination selection, different initialization, or a wider geometric search.
This does not exclude a finite-width Gaussian's tails, larger pose moves,
cooperative rearrangements or physical assembly.

The complete fixed-allowance reduction, including all 2048 labels, is in
`results/context-relaxed-center-displacement-screen-20261004/execution01/`.
It completed and drained in 0.532 CPU seconds, with no new atom-pair geometry
or pose transforms. Analysis SHA-256:
`6807d3d07c851d512c6ed7b8952f34730ed802f114d5a452466103e289b5322b`.

Artifacts:

- [Fixed allocation and protocol](../results/context-relaxed-center-score-preparation-20261004/protocol.json)
- [Complete scores](../results/context-relaxed-center-score-preparation-20261004/result/report.json)
- [Drained execution](../results/context-relaxed-center-score-preparation-20261004/host-drained.json)
- [Independent scalar audit](../results/context-relaxed-center-scalar-review-20261004/execution01/analysis.json)
- [Independent audit receipt](../results/context-relaxed-center-scalar-review-20261004/execution01/receipt.json)

The score report SHA-256 is
`b352d1ab29977872dde9199743304698bdb2e85f1886460fb073f95f8d4eef97`.

## CLI reproduction

The CLI requires `--source-model`, `--exported-model`, `--shape`, `--context`,
`--allocation` and a fresh `--out`. Allocation schema
`actual-center-core-score-allocation-v1` binds the four input hashes, scorer
and geometry-helper hashes, all identity/query/atom counts, and `settings`
(`margin`, `max_candidates`, `batch_size`). The count fields must describe the
whole input table. The completed allocation is a concrete example.

Limits in the allocation are provenance; the CLI relies on its supervisor to
enforce CPU, wall and memory limits. Production campaigns should use a frozen
execution plan with `tools/run_native_class_physical_campaign.py`, as in the
archived run. For an explicit reproduction from the repository root, this
bounded command uses the same inputs and settings with the promoted scorer:

```bash
export SCORE_RUN=$(mktemp -d /tmp/context-core-score.XXXXXX)
export SCORE_ARCHIVE="$PWD/results/context-relaxed-center-score-preparation-20261004"
export SCORE_EXPORT="$PWD/results/context-relaxed-atlas-quartet1-preparation-20261004-v2"
export SCORE_PYTHON=/home/xvg/protein-nucleation/.venv/bin/python

"$SCORE_PYTHON" -B - <<'PY'
import hashlib, json, os
from pathlib import Path
allocation = json.loads((Path(os.environ['SCORE_ARCHIVE']) / 'allocation.json').read_bytes())
for key, path in [('scorer_sha256', 'tools/context_core_score.py'),
                  ('geometry_sha256', 'tools/context_relaxed_atlas.py')]:
    allocation[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
with (Path(os.environ['SCORE_RUN']) / 'allocation.json').open('x') as out:
    json.dump(allocation, out, indent=2)
PY

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  timeout --kill-after=2s 1200s \
  prlimit --cpu=600:601 --as=4294967296:4294967296 \
  "$SCORE_PYTHON" -B tools/context_core_score.py \
    --source-model "$SCORE_EXPORT/inputs/model.json" \
    --exported-model "$SCORE_EXPORT/result/model.json" \
    --shape "$SCORE_EXPORT/inputs/shape.json" \
    --context "$SCORE_EXPORT/inputs/context.json" \
    --allocation "$SCORE_RUN/allocation.json" --out "$SCORE_RUN/result"
```

This command starts a new geometry calculation; it was not rerun during tool
promotion. It leaves the archived allocation, sources and results untouched.
For strict replay, also retain the archived interpreter/library pins and
supervisor receipts; a current-runtime reproduction alone is not such a replay.

The same 14 synthetic tests cover exhaustive sphere/dumbbell references,
noncommuting frame transforms, finite-difference gradients, tangencies,
coincidences, radius filtering, censored gaps, caps, caching and failure
prefixes. The promoted code passed all 14 with one thread and limits of
30 CPU seconds, 60 wall seconds and 2 GiB; the child process was drained.
[Promotion validation receipt](../results/context-core-score-promotion-validation-20261004/validation-attempt01/receipt.json)
(SHA-256 `6075553dd1cd687362b983705b8d91cd68c492b0940da7188f57362e6c1e14f2`).
They can be run without protein inputs:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  timeout --kill-after=2s 60s \
  prlimit --cpu=30:30 --as=2147483648:2147483648 \
  /home/xvg/protein-nucleation/.venv/bin/python -B tools/test_context_core_score.py
```
