# Native extension points in the observed oligomer geometries

**At the original fixed positions, only one of 52 cooperative extension points
around the four geometry-only quartets is hard-clear.** A subsequent
[incoming-only pose search](native-extension-incoming-search-20261004.md) clears
50 of the same 52 points against the measured quartet and 34 against the full
environment. Thus most fixed-point clashes can be resolved without moving the
existing quartet. These are point-accessibility results, not basin volumes or
equilibrium weights.

This diagnostic uses the historical N=264 growth endpoints at **1.4 Å
depletant radius, 0.04 Å⁻³ activity and 500 μM**. It is separate from the
decision conditions of 1.5 Å / 0.035 Å⁻³ / 106.8 μM. There are two correlated
growth histories, not independent equilibrium populations.

![Clearance of fixed cooperative extension points](../results/native-extension-clearance-report-20261004/render02/clearance.png)

## Completed comparison

The earlier exact enumeration found site-disjoint native placements, using
each retained ideal embedding. The present calculation maps each such point
through its **already computed single rigid fit** into the actual endpoint.
All generating paths must agree. Existing particles stay at their measured
poses; no new fit, pose search or relaxation is performed.

| Component | Tetramer IDs | Ideal cooperative points | Clear of own component | Fully clear, with ≥2 native neighbors |
| --- | --- | ---: | ---: | ---: |
| Q1 | 16, 77, 217, 237 | 12 | 0 | 0 |
| Q2 | 42, 184, 194, 206 | 16 | 0 | 0 |
| Q3 | 62, 67, 80, 134 | 12 | 0 | 0 |
| Q4 | 69, 89, 91, 229 | 12 | 1 | 1 |
| Geometry-only seed, 9 bodies | Original seed plus body 258 | 33 | 5 | 4 |
| Native-informed seed, 16 bodies | Original seed plus eight attachments | 62 | 10 | 10 |

The denominators are **site-disjoint** cooperative points. The raw output also
retains candidates that share an occupied ideal monomer site; those must not
be added to the table's denominator. Cooperative means contact to at least
two distinct existing tetramers, not two independent geometric constraints.

For the 52 quartet points, the median minimum gap to the quartet is
−0.1944 Å. The deepest penetration is 0.5737 Å, and the shallowest negative
gap is −0.00483 Å. Fifty points penetrate by at least 0.01 Å, 37 by at least
0.1 Å, and five by at least 0.5 Å. Seventeen also clash with spectators;
34 clash with their own quartet while spectators are clear. None of these
52 points is wall-blocked. The one fully clear point has a minimum own-component
gap of +0.0221 Å and supports two native neighbors.

Every intended neighbor of all 52 points passes the coarse native pose and
residue-patch criteria. All 69 own-component blocking-body instances are
intended catalogue neighbors, rather than other members of the quartet.
This makes the distinction concrete: passing the native registration gates
does not certify that a proposed insertion avoids atomic overlap.

Both seed components retain several open cooperative points. Thus absence
of any such point does not explain the seed's stall at this endpoint. The
quartet results motivate a more specific distinction: can the incoming body
adjust its pose enough, or must existing members move as well?

## Scope and next discriminating check

This is a **native-informed diagnostic**, not a prior supplied to the blind
proposal. The complete calculation preserves all 113 nontrivial components,
9,952 candidate records and 7,207 site-disjoint point queries. Among all
site-disjoint cooperative points, the geometry-only endpoint has 55 fully
clear native points out of 577; the informed endpoint has 100 out of 865.
These are spatial inventories, not statistical weights or effective samples.

An open point does not establish an entry path or appreciable basin volume.
A blocked point does not exclude a nearby allowed pose. The separately
allocated ideal-component control below resolves whether the underlying ideal
packing permits these same points. The completed bounded incoming-pose search
then resolves 49 of the 51 initially blocked points with the measured component
fixed; the initially clear point stays unchanged. The two remaining searches
are unresolved, despite numerical optimizer success. Neither proves that
cooperative component rearrangement is required. No model-refutation or
equilibrium-stability claim follows from these pointwise diagnostics.

## Completed ideal-component control

**All 52 points are clear against the corresponding ideal quartet**, with the
same repaired atomic shape, candidate poses, exact embedding witnesses and
previously fitted global rigid frame. Only the existing four bodies are moved
from their measured coordinates to their fitted ideal coordinates. Spectators
and walls are omitted from this isolated-component comparison in both columns:

| Quartet | Ideal component: clear / tested | Measured component: clear / tested |
| --- | ---: | ---: |
| Q1 | 12 / 12 | 0 / 12 |
| Q2 | 16 / 16 | 0 / 16 |
| Q3 | 12 / 12 | 0 / 12 |
| Q4 | 12 / 12 | 1 / 12 |

The ideal minimum atomic gaps range from **0.036918 to 0.042645 Å**, with median
0.037131 Å and no near-zero flags. Thus departures of these observed quartets
from ideal geometry cause the obstruction at the particular tested points.
That does not establish obstruction of every nearby incoming pose, or exclude
physical growth through another route. These gaps are not basin widths.

The fixed 52-query control completed in 1.39 CPU / 1.49 wall seconds with its
child drained. Nine synthetic tests passed. Before querying atoms, the source
checked all 34 catalogue labels and 70 ideal monomer-contact witnesses; every
slot's intended operators and its unchanged candidate pose were independently
reconstructed. A separate arithmetic audit reconciled all 52 records and 104
begin/end events without additional geometry queries.

Artifacts are in `results/native-extension-ideal-control-20261004/`:

- Execution plan: `66d6daa9ad91ae608532cc25aeca750e20ea95cf15d9ad24aa9c50c25f53edb8`.
- Observer source: `b024518c1ad39fa962197536d30743113ad5fe6c0ab2763b47fdf341dfeddbb9`.
- Completed receipt: `b547efeb90f34b395c9f2ceffda6d55acc02d128cb774a50f907d5b19e2e1cf9`.

The [ideal-control procedure](../results/native-extension-ideal-control-preparation-20261004/PROCEDURE.md)
records the fixed body-ID mapping, omitted wall and spectator terms, and
floating-point implementation obligations.

## Validation and artifacts

Nineteen synthetic tests passed, including noncommuting transforms, all-path
agreement, ambiguous fits, strict tangencies and shallow overlaps, wall
boundaries, unequal radii, pruned-versus-exhaustive queries and complete output
denominators. The actual repaired shape supplies strict hard and wall geometry;
its 4,004 atom centers and radii matched the compiled member mapping exactly.
The observer uses the stored sphere-centered frame.

All 7,207 atom-neighborhood queries completed in **86.16 CPU / 109.42 wall
seconds** on one worker. Its controller drained successfully. The independent
arithmetic reduction reconciled every component/arm summary and all 19,904
begin/end attempt records. No point had a near-zero flag or a different
strict-versus-1e−8-tolerance overlap classification. Floating-point predicate
execution remains an implementation obligation; this is not an exact-real
arithmetic certificate.

The [independent reduction](../results/native-extension-clearance-independent-review-20261004/analysis-note.md)
retains all per-point gap values and verifies the intended-neighbor membership
of every quartet blocker without additional geometry queries.

The frozen calculation is
`results/native-extension-clearance-20261004-v2/`:

- Execution plan: `985d1c2f34a41bb7583e0d30aa17b8f4ff329068c7e51251621d497a99a7045c`.
- Observer source: `120356b1fa62e9b4b9be409df2d686405154eb4b7e8fc4a4d53a81b8eef51739`.
- Completed receipt: `74dff85ee55e657fe38599756fc02863a8ef4a67b11880bfb82e91c115e3d2b6`.
- Synthetic validation: `11236bc30ea4af8cf68316a96028a3009a8d6474cbc1dc18c08679b3b13570e4`.

The [procedure](../results/native-extension-clearance-preparation-20261004/PROCEDURE.md)
describes the guarded atomic tree queries, censored gap bounds, fixed native
criteria and numerical limits. The original v1 preparation was superseded
before any scientific execution; its frozen files remain intact. The first
figure is retained but superseded by the figure above, whose title avoids
attributing all obstruction to thermal deformation before the ideal-component
control is checked.
