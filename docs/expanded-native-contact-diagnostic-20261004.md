# Native contacts beyond the original tetramer catalogue

**The completed geometry-only growth run contains substantially more native
registration than the original observer reported.** A supplemental catalogue
raises its final contact count from 29 to 99, with 155 rather than 40 tetramers
participating in those contacts. This corrects an interpretation of the saved
growth history; it does not establish equilibrium crystal stability.

The two historical N=264 runs use depletant radius **1.4 Å**, activity
**0.04 Å⁻³**, concentration **500 μM**, spherical boundaries, and an initial
eight-tetramer native seed. They share initial poses and an RNG seed and are
not independent equilibrium preparations. The original decision conditions
remain **1.5 Å / 0.035 Å⁻³ / approximately 106.8 μM**.

## Why the observer missed these contacts

The original fourteen tetramer placements came from a selected covering of
the native monomer lattice. That catalogue is narrower than the set obtained
by placing any of four members against any other member using the seven
authenticated directed native monomer contacts:

`H = M_i C_k inverse(M_j)`.

The 112 constructions give 43 distinct exact operators. Nine reuse a monomer
site, leaving 34 site-disjoint placements, including all fourteen old motifs.
All twenty additional placements pass exhaustive hard-atom checks without
relaxation. All twenty also fail the old catalogue's finite body-entry gates
at their exact poses. They are absent contacts, not aliases already covered
by the old entry tolerances. The [construction and checks](exact-native-embedding.md)
give their provenance and limits.

## Completed endpoint diagnostic

The additional twenty placements were frozen before evaluating the two final
endpoints at sweep 100,000. Every one of the 69,432 unordered body pairs was
considered. The observer reused the original compiled residue-reference patches
and the original entry tolerances, requiring at least one qualifying prescribed
monomer contact. It retained every matching label and reused all old labels
unchanged. No simulation, reference-patch reconstruction, or old-label
classification was repeated.

| Final endpoint | Original contact pairs | Additional pairs | Expanded pairs | Tetramers with contacts | Seed-connected component | Largest separate component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native-informed proposal | 248 | 34 | 282 | 228 → 228 | 16 → 16 | 10 → 10 |
| Geometry-only proposal | 29 | 70 | 99 | 40 → 155 | 8 → 9 | 2 → 4 |

All 47 informed and 66 geometry-only nontrivial components admit exact,
cycle-consistent embeddings with distinct monomer sites. The graph calculation
retains all observed pair constraints and alternatives; it does not choose a
convenient subgraph. All positive witnesses were independently checked using
integer composition. Neither endpoint exhausted the fixed search budget.

![Original and expanded contact graphs in the sphere frame](../results/growth-z004-supplemental-native-figure-20261004-v2/contact-graphs.png)

Blue edges are the original labels; orange edges are supplemental labels.
Only body centers are drawn. Grey bodies can have other physical contacts.
The saved trajectory poses already use the sphere-centered frame. The first
plot incorrectly subtracted `coordinate_wall_center` again; it is preserved
and marked superseded. The figure above uses the authenticated serialization
convention. Classification depends on relative poses and was unaffected.

## What this changes

The geometry-only proposal can reach native-monomer-registered oligomers and
an additional seed attachment. Its endpoint cannot be described as showing
only the original sixteen separate dimers and an unchanged eight-body seed.
The native-informed endpoint still has larger components and more contacts.
These two trajectories do not establish a mixing speedup or equilibrium
occupancy for either proposal.

“Contact without native entry” in existing physical-weight reports retains
its literal frozen meaning: contact outside the original fourteen-template
entry region. It must not be equated with the absence of native monomer
registry. The ongoing weight and benchmark analyses keep that definition.
Any wider-region weights or trajectory diagnostics require a separately
declared observer and full preservation of the original results.

The measured coordinates also agree closely with the ideal lattice. For each
of all 113 components, the already selected exact embedding was aligned to
the observed member centers using one global proper rigid transformation.
There was no per-body relaxation, fitted cell strain, scale change, or
selection of an embedding to improve the fit.

| Endpoint | Components | Median component RMS error (Å) | Largest component RMS error (Å) | Seed component RMS error (Å) |
| --- | ---: | ---: | ---: | ---: |
| Native-informed | 47 | 0.177 | 0.943 | 0.125 (16 tetramers) |
| Geometry-only | 66 | 0.123 | 0.795 | 0.225 (9 tetramers) |

The four separate geometry-only four-tetramer components have RMS errors of
0.164–0.228 Å. Maximum individual member-center errors across the endpoints
are 2.40 Å and 1.40 Å, respectively; a component RMS is not a bound on every
member error. Maximum body-orientation errors are 3.70° and 2.23°. These fits
support interpreting the classified endpoint components as native-lattice
arrangements. A separately frozen full-history observer is needed to
distinguish their initial presence, formation, loss, and reformation.

Finite-system stability remains unresolved pending physical weights,
independent-start agreement, reversible exchanges, and size/boundary checks.

## Reproducibility

- Supplemental classifier: 13 synthetic tests passed, including direction,
  residue identity, nonidentity coordinate frames, all-label retention,
  thresholds, query exhaustion, and periodic-image safeguards. Actual endpoints
  are spherical. The completed calculation used 289 monomer contact queries,
  33.04 CPU seconds and 71.40 wall seconds on one worker/thread.
- Endpoint receipt:
  `results/growth-z004-supplemental-native-20261004/receipt.json`, SHA256
  `e2f0333058b0b4b13b1b373c1099c2f7ecd16d209f7b872143486675af652bf8`.
- Exact union-graph receipt:
  `results/growth-z004-supplemental-native-graph-20261004/receipt.json`, SHA256
  `6e45b3c2a813612990fa38a6ac8b5e59c76e5e48f3b6803687ca4a7ebbc573c0`.
  The previously validated solver used 181/89 attempted assignments, below
  the one-million-per-endpoint cap; no geometry was recalculated.
- Corrected figure and completed journal audit:
  `results/growth-z004-supplemental-native-figure-20261004-v2/receipt.json`,
  SHA256 `701428406a71728bda645790b84c8e056eebe8985690cbe13d79716f4919b597`.
- Fixed-witness coordinate fits: eight synthetic controls passed, including
  reflections, rank deficiency, reversed edges, all-cycle closure, and known
  strain. All 113 fits completed in 0.36 CPU seconds with no new geometry or
  embedding search. Receipt:
  `results/growth-z004-supplemental-native-fit-20261004/receipt.json`, SHA256
  `dc6a089a47a2441cf4d0ed03145128f08349c8ec79ea852b9c25070bb5c01ea6`.
- Conservative center pruning: nine synthetic tests and both completed
  endpoint oracles passed. The 91.6393 Å center bound retains 729/34,716 and
  547/34,716 pairs, preserving every observed supplemental body-gate pass and
  contact. This reduces potential classification calls by factors of 47.6 and
  63.5; it is not an end-to-end sampler speedup. Receipt:
  `results/supplemental-center-pruning-20261004/receipt.json`, SHA256
  `4af3fc492cb5dbb690e25beb71c10467180ddbbc0fce4bc930dda41722090ff1`.
