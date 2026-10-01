# Fixed passive check of cooperative translation-line conditioning

The [prepared allocation](../runs/contact-line-passive-preparation-v3-20261001/plan.json)
tests the sampling law, geometric support and CPU cost before spending on new
depletion estimators. It launches no jobs by itself and contains **zero new
Poisson clouds**. It cannot estimate physical contact weights or decide assembly.

The original repaired shape, two fixed scaffold poses, R4 chart, radius 1.5 Å,
activity 0.035 Å⁻³ and physical Jacobian are pinned unchanged. The guide retains
50% uniform R4 probability. In the Gaussian half it mixes the original draw
with exact one-coordinate conditioning with equal probability. The complete
density must therefore retain at least half the old92 density everywhere.
The conditional law is described in [the design note](line-conditioned-gaussian-guide.md).

## Geometry used to choose the first axis

All 78 previously archived competing-55 configurations were reconstructed from
their latent coordinates. Distances for the stored nearest atom pair to each
scaffold agree with their archived values to 2.91e−14 Å. The chosen **raw chart
x coordinate** translates along the fixed-neighbor body's x axis; its world
direction is approximately (−0.0059903, 0.7565771, 0.6538771). It is neither an
arbitrary whitened latent coordinate nor the world's x axis. Holding the other
five raw coordinates fixed preserves the particle's orientation.

For the axis diagnostic only, exact quadratic shell intervals for the two
stored atom pairs were intersected with the R4 line. These omit every other
core pair, and another atom pair could replace a saved pair as the closest
contact. Consequently they are **neither lower nor upper bounds on the full
sphere-union feasible intervals**. They help choose a direction but do not
validate the production geometry predicate.

| Axis / near-core width | Observed weighted fraction with positive two-pair interval | Observed weighted interval length |
|---|---:|---:|
| raw x / 0.02 Å | 41.7% | 0.00460 Å |
| raw x / 0.10 Å | 78.2% | 0.09276 Å |
| raw x / 0.50 Å | 100% | 0.72005 Å |
| raw y / 0.10 Å | 76.8% | 0.09699 Å |
| raw z / 0.10 Å | 81.9% | 0.09108 Å |

These summaries use the completed 92-component pilot's observed weights,
which remain unconverged. Full per-row geometry and both source arms, also split
into earlier/later populations, are preserved in
[axis-diagnostics.json](../runs/contact-line-passive-preparation-v3-20261001/axis-diagnostics.json).
All those rows were inspected historically; the splits are descriptive rather
than pristine holdouts. The fixed-axis choice precedes the new proposal-only
draws. No guide mean, covariance or weight was fitted here.

## Frozen pilot

Use raw axis x only, with widths **0.02, 0.1 and 0.5 Å equally likely**. They
describe the minimum hard-surface gap to each entire scaffold union, not bands
around the native-registration classifier's 2 Å threshold. Full 3 Å contact
intervals appear only in the saved-pair diagnostic; no fourth width or fresh
third arm is included.

| Arm | Conditional fraction within Gaussian half | Fresh streams × attempts | Archived probe queries |
|---|---:|---:|---:|
| baseline92 | 0 | 4 × 64 | 78, in r00 only |
| line92 | 0.5 | 4 × 64 | 78, in r00 only |

This gives **512 fresh proposal draws and 156 archived critical queries**, at most one
CPU worker. A separately declared breadth control adds **128 probe queries** under
the conditioned guide only, with its baseline density reconstructed directly.
It selects eight rows uniformly within each of four classes from each original
bank/protected r02/r03 population: native R5, native complement, competing, and
unbound if at least eight exist, otherwise invalid. No weight filtering is used.
All four populations use the invalid fallback, giving 32 rows per class. The
extra zero-fresh-draw job does not change the original allocation. Eight distinct fresh-stream seeds, a separate coverage-job seed and exact binary commands are in the
plan. Every attempted draw remains in the output, including shell/hard failures.
No retry, optional stopping or automatic extension is permitted. The executable
and its source closure must be bound before diagnostic execution.

For each fixed outer five-coordinate value, production geometry must intersect
both scaffold contact unions and R4, then subtract **all** hard-core overlap
intervals. Empty lines fall back to the original conditional Normal. The fixed
component-wise conditional-mass threshold is 1e−12; its implementation convention
and every resulting fallback must match sampling and complete-density scoring.
Neither fallback redraws the outer coordinates. The full density sums every
Gaussian component and width, including fallback branches.

Validation should independently reconstruct the raw pose, Jacobian, each
component's conditional Normal, stable interval probability and complete q from
saved interval endpoints. It must verify the baseline against the original92
density and the pointwise retention bound. Reference interval predicates and
boundary tests remain implementation obligations. A selected component's
density alone is insufficient.

The outcome is hard-valid, shell and joint-contact retention **per total CPU**,
with between-stream variation, fallback counts, geometry traversal counts and
construction/draw/full-density costs reported separately. Archived Poisson
weights may support separately labeled retrospective second-moment diagnostics;
new proposal frequencies are not physical occupancies. Existing regional
convergence failures and the finite-system assembly verdict remain unchanged.

Reproduce the inert preparation into a fresh directory:

```bash
cd /home/xvg/tetramer-mc
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_contact_line_passive.py \
  --out /path/to/new-contact-line-passive-preparation
```

- [Minimal id/latent probe bank](../runs/contact-line-passive-preparation-v3-20261001/probes.jsonl)
- [Baseline guide](../runs/contact-line-passive-preparation-v3-20261001/guides/baseline92.json)
- [Conditioned guide](../runs/contact-line-passive-preparation-v3-20261001/guides/line92.json)
- [Frozen file hashes](../runs/contact-line-passive-preparation-v3-20261001/freeze.json)
- [Preparation helper](../tools/prepare_contact_line_passive.py)

The first two preparations remain archived. Revision v3 adds the explicit
`contact_neighbor_indices: [0, 1]` required by the implementation and the separately
declared breadth control; no earlier plan was executed. A source-bound sequential
controller is available in [run_contact_line_passive.py](../tools/run_contact_line_passive.py).
The fixed allocation has now completed, with exact binary/source binding and
independent density audits passing for all nine jobs. The
[completed result](contact-line-passive-results.md) finds frequent geometric
fallback and no useful broad-contact gain at the higher passive cost; no
physical-weight campaign was launched for this variant. Its frozen execution
protocol is [archived here](../runs/contact-line-passive-20261001/protocol.json).
