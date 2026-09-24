# Matched initial geometries for finite-system assembly

The new preparation supplies **24 independently seeded canonical starting
configurations and 48 boundary-specific state files**: N=12 and N=24, dispersed,
competing-aggregate and native-seeded starts, four streams each, exported to both
spherical and periodic vessels. Every saved state passes strict atomic core/wall
checks and its prescribed instantaneous contact-graph checks. No Monte Carlo
trajectory or depletant cloud was sampled.

![Matched initial preparations](../runs/finite-assembly-starts-report-20260924/initial-preparations.png)

The figure uses stream 0 without selecting a favorable example. It projects
every fourth atomic center onto XY; apparent projected overlaps can be separated
in depth. The dashed circle is the spherical wall projection and the dotted
square the equal-volume periodic cell. Green marks the supplied eight-tetramer
native fragment and its instantaneous native edges. Teal lines mark exclusion
contacts in the competing aggregate. All contacts come from the saved complete
validation records, not image inspection.

## Construction and physical identity

The repaired rigid shape is unchanged. Both sizes have concentration 106.8 μM,
depletant radius 1.5 Å and activity 0.035 Å⁻³. N=24 has twice the volume of N=12.
Their sphere radii are 354.4787133 and 446.6151926 Å; corresponding cubic periodic
side lengths are 571.4168336 and 719.9400970 Å.

Each canonical configuration fits both vessels. Periodic exports only wrap the
same coordinates into the cell; orientations and relative physical geometry
are preserved. The paired-boundary identity has been checked directly using the
canonical pose hashes and wrapping operation. Restricting initial placements to
a common valid domain is a preparation choice, not a uniform equilibrium law
for either vessel.

- **Dispersed:** independently oriented tetramers with all exclusion bounding
  spheres separated. Geometrically rejected placement attempts remain recorded.
- **Competing aggregate:** append each randomly oriented tetramer just 0.1 Å
  beyond the last core intersection along a ray through the entire existing
  atomic union. This works for a nonconvex union. A geometric extent check keeps
  the aggregate inside the common preparation domain. Native labels never guide
  construction; the completed aggregate is checked independently afterward.
- **Native-seeded:** preserve the first eight prescribed bodies from the pinned
  original seeded configuration, apply one proper rigid transformation to the
  entire fragment, and add separated free tetramers. There are four free bodies
  at N=12 and sixteen at N=24. The native intratetramer and seed structures are
  supplied, not discovered by this preparation.

All bodies remain mobile in the state assets. The plan, physical inputs, seed
source and source closure were archived before construction. Every accepted and
rejected construction event was flushed to its log. A failed independent check
would preserve the state and validation report and stop preparation; it would
not redraw a configuration based on its native label.

## Saved validation across all streams

The following ranges cover four streams and both boundary exports at each size.
Small differences between boundary exports arise from floating-point wrapping.

| N | Preparation | Largest exclusion component | Native edges | Native component containing contacts | Smallest queried core gap, Å | Rejected construction trials |
|---:|---|---:|---:|---:|---:|---:|
| 12 | Dispersed | 1 | 0 | none | bound-certified separation | 21 |
| 12 | Competing aggregate | 12 | 0 | none | 0.035875 | 0 |
| 12 | Native-seeded | 8 | 13 | 8 | 0.036918 | 3 |
| 24 | Dispersed | 1 | 0 | none | bound-certified separation | 42 |
| 24 | Competing aggregate | 24 | 0 | none | 0.020405 | 0 |
| 24 | Native-seeded | 8 | 13 | 8 | 0.036918 | 20 |

Every native seed has six independent catalogue cycles with a consistent ideal
pose assignment. All free bodies remain isolated in the exclusion graph. Every
competing aggregate is fully exclusion-connected, with zero complete native-entry
matches. Absence of an entry label does not prove that all structural similarity
to native arrangements is absent. Empty native graphs contain singleton vertices;
the figure reports zero bodies in a nontrivial native component in those cases.

The smallest spherical atomic-wall clearance is 13.7844 Å. There are 470 recorded
construction events, including 86 rejected geometric trials, counted once per
canonical preparation rather than twice for its two exports. These events are
not accepted/rejected physical Monte Carlo updates. Construction and independent
validation completed in 16.26 seconds.

The independent static validator uses guarded bounding-sphere pruning and
KD-tree candidates followed by strict squared-distance atom-pair inequalities.
Native classification occurs only after strict physical validity succeeds.
Periodic pairs use minimum images under the existing sufficient box-size
condition; native graphs must admit an ordinary-space image lift. The separate
eight-body source preflight is also preserved. Twenty-five focused tests cover
the ray construction, atomic thresholds, rotations, walls, periodic images,
disconnected or registered competing starts, native seed/remainder checks,
inconsistent cycles and periodic winding.

## Artifacts and limits

- [Frozen construction plan](../runs/finite-assembly-starts-20260924/plan.json)
- [All 48 state assets and their saved checks](../runs/finite-assembly-starts-20260924/manifest.json)
- [All-stream report and authenticated input hashes](../runs/finite-assembly-starts-report-20260924/analysis.json)
- [Independent eight-body source preflight](../runs/finite-assembly-seed-preflight-20260924/validation.json)
- [Inert campaign contract](finite-assembly-contract.md)

The report authenticates all frozen preparation files, checks the exact allocation,
retains each saved validation and verifies paired boundary coordinates. It does
not repeat atom geometry, native classification, or physical sampling.

These assets fill the initial-geometry requirement. They do not supply production
windows, observer/environment definitions, proposal assignments or fresh runtime
MC seeds. The geometry-only memory model and native-informed controls remain
separate proposal assets. Physical contact-weight/full-vessel gates and production
launch prerequisites are unchanged. An imposed aggregate or native seed supplies
no evidence of equilibrium assembly, mixing efficiency or finite-system stability.

To reproduce this report in a fresh directory without rerunning construction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  tools/summarize_finite_assembly_starts.py \
  --root runs/finite-assembly-starts-20260924 --out /absolute/fresh-report
```
