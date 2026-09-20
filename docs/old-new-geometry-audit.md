# Archived SMC versus current normalizer geometry

No shape, radius, coordinate-frame or overlap-volume mismatch was detected
at the twelve tested archived SMC endpoints. **The archived and current
overlap predicates agreed on all 12,582,912 common random points.** Independent
Poisson overlap estimates from the current normalizer also agreed with the
common-point integral. The observed normalizer disagreement therefore cannot
be attributed to a detected multi-kBT physical-geometry change at these poses.
This result does not establish global pose coverage or normalizer convergence.

The [numerical results](../runs/old-new-geometry-audit-v2/results.json),
[selected poses](../runs/old-new-geometry-audit-v2/input.json),
[current/archived config comparison](../runs/old-new-geometry-audit-v2/current-config-comparison.json),
and [source provenance](../runs/old-new-geometry-audit-v2/provenance.json)
are retained. Original SMC datasets were not modified.

## Independent implementations and matched inputs

The diagnostic compiles the **actual archived** `HardUnion` implementation
from
`/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src/geometry.rs`
beside the current `tetramer_mc::geometry::SphereTree`. It uses each one's
own rigid-coordinate transformation. The old fixed-body query applies the
nearest periodic image as in archived `SingleBodyDepletion`; the current
query uses the open fixed neighborhood. The diagnostic verifies that remote
periodic images cannot intersect the compact exclusion domains, so this
boundary-condition difference is irrelevant at the selected poses.

The input checks establish:

- Identical 4,004-sphere shape bytes, SHA-256
  `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9`.
- Identical actual-run depletant radius 1.5 Å and activity 0.035 Å⁻³.
- Identical fixed poses, 18 Å capture ball and center for both site0 and site1
  current docking configurations versus their archived SMC configurations.
- Identical native registration errors after independent recomputation using
  both configurations' member geometry, reference poses and tolerance scales.
- Hard validity under both archived and current hard-overlap predicates.

Some old environment files retain bath metadata for an earlier radius/activity.
Those fields are not the operative parameters: the archived SMC run configs
set the actual values, as does the current normalizer. The diagnostic uses
those configs and rejects a discrepancy.

The archived hoomd vector, microstate and interaction dependencies are
byte-identical to the pinned current snapshot. The isolated diagnostic crate
uses the common snapshot to avoid duplicate package identities. The first
preparation failed at Cargo resolution because two copies of the same package
name/version had different paths; it produced no numerical result. Its failed
build is retained in `runs/old-new-geometry-audit/`. The successful run and
its archived executable are in `runs/old-new-geometry-audit-v2/`.

There is a boundary convention difference: archived point exclusion uses an
open sphere, while the current leaf predicate includes its boundary. Sphere
surfaces have zero volume in these continuous integrals. No tested random
point differed because of this convention.

## Selection and common-point volume check

For each site, the audit selects the native population with the largest
reported normalizer and the two largest-normalizer other populations. It
tests the minimum-q and maximum-q final endpoint in each selected population,
giving four native and eight distant poses. This deterministic selection
includes high-contribution populations and geometrically separated poses;
it is not an equilibrium sample or an exhaustive audit of their descendants.

For each pose, the integration box is the intersection of atomic exclusion
AABBs for the moving body and fixed union, constructed directly from atom
centers, inflated radii and pose rotations. Each implementation classifies
the **same** 1,048,576 independent uniform world-coordinate points. Both
individual-union membership and complete overlap membership are compared.

| Site / SMC population / endpoint | Common C / Å³ | Common standard error / Å³ | Current envelope C / Å³ | Difference / combined SE |
|---|---:|---:|---:|---:|
| 0 native-07 / 177 | 869.41 | 6.24 | 863.73 | −0.72 |
| 0 native-07 / 263 | 1041.83 | 6.79 | 1031.58 | −1.18 |
| 0 other-01 / 463 | 598.25 | 5.30 | 613.70 | +2.30 |
| 0 other-01 / 037 | 425.30 | 6.41 | 427.93 | +0.36 |
| 0 other-05 / 374 | 280.90 | 3.95 | 284.54 | +0.75 |
| 0 other-05 / 494 | 543.98 | 5.93 | 544.48 | +0.07 |
| 1 native-00 / 504 | 860.04 | 5.54 | 861.58 | +0.21 |
| 1 native-00 / 269 | 806.11 | 5.50 | 798.21 | −1.09 |
| 1 other-02 / 362 | 480.64 | 4.84 | 476.53 | −0.68 |
| 1 other-02 / 352 | 268.35 | 4.11 | 267.13 | −0.25 |
| 1 other-06 / 127 | 45.53 | 1.34 | 45.45 | −0.04 |
| 1 other-06 / 059 | 436.35 | 3.59 | 439.17 | +0.56 |

The current-envelope column averages sixteen independent Poisson counts at
`lambda = 64 z = 2.24 Å⁻³`. It estimates `C = L + mean(K)/lambda` using the
same `OverlapEnvelope` and `sample_with_envelope` functions as the current
normalizer. Its Poisson standard error and the independent common-box binomial
standard error are added in quadrature. The largest discrepancy is 2.30
combined standard errors among twelve comparisons; all others are at most
1.18. Common-integral uncertainty corresponds to 0.047–0.238 kBT in `z C`.

The *paired* old/new comparison is stronger than comparing independent volume
estimates: all overlap indicators are equal. Conditional on these fixed boxes
and independent uniform-point sampling, zero disagreements give a one-sided
binomial upper bound `1 - (0.05/12)^(1/N)` on disagreement probability in each
box, simultaneously at 95% by a union bound. Even the largest box then has
at most 0.533 Å³ disagreement volume, corresponding to 0.0187 kBT. This is
a Monte Carlo bound at the twelve selected poses, not a formal floating-point
proof or a bound over all possible poses. The exact-match result should not
be extrapolated to unseen geometry without further checks.

## Reproduction

From `/home/xvg/tetramer-mc`, use a fresh output directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/audit_old_new_geometry.py \
  --out runs/old-new-geometry-reproduction --points 1048576
```

The script creates an isolated Cargo crate, compiles the archived geometry
beside the current library offline, archives the executable and inputs, and
runs all cases. It does not edit the sampler or old data. To recheck only
current-versus-archived config agreement for the existing numerical audit:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/audit_old_new_geometry.py \
  --verify-existing runs/old-new-geometry-audit-v2
```

The standalone Rust diagnostic is
[old_new_geometry_probe.rs](../tools/old_new_geometry_probe.rs). The original
normalizer algorithms still require separate validation; this audit isolates
the physical shape, coordinate transformation, point exclusion, and current
overlap-envelope calculation from proposal-density and population effects.
