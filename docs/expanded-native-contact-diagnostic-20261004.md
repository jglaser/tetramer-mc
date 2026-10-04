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
arrangements. The separately frozen full-history observer has now completed,
distinguishing initial presence from later observations without changing the
original labels.

Finite-system stability remains unresolved pending physical weights,
independent-start agreement, reversible exchanges, and size/boundary checks.

## Completed histories: accessibility and growth are different

Both runs began with thirteen original seed bonds, **zero supplemental
contacts, and zero solution contacts**. All seventy supplemental labels in the
geometry-only final frame appeared after initialization. The final solution
inventory is 53 dimers, eight trimers and four quartets of tetramers. Its only
seed addition first appears at sweep 5,200; its largest solution component
first reaches four at sweep 15,500 and never exceeds that size.

![Expanded contact histories](../results/growth-z004-supplemental-history-figure-20261004/contact-history.png)

These historical runs use 1.4 Å / 0.04 Å⁻³ / 500 μM, N=264, a supplied
eight-tetramer seed and correlated initializations. In the last 25,000 sweeps,
the geometry-only history finds only **one new solution pair**, versus 34
returns of previously seen pairs across the saved entry threshold. Its seed
stays at nine and solution components stay at most four. The informed history
finds thirteen new solution pairs and 37 returns, while ending with a seed
size of sixteen and maximum solution component of ten.

The supplemental geometry-only labels are usually persistent at the saved
cadence: 144 of 185 observed episodes span at least 1,000 sweeps, 85 span at
least 10,000, and seventy are still present at the final frame. Only twelve
appear at one saved frame. These are consecutive saved observations, with
explicit censoring, not physical residence times. Threshold flicker can
produce a loss and return within the same physical contact; these counts do
not establish completed attachment/detachment or environment exchanges.

The new evidence establishes formation of native-registered solution
oligomers under this proposal. It also locates the observed stagnation in
growth beyond those oligomers. The
[full history report](../results/growth-z004-supplemental-history-figure-20261004/report.md)
retains original, supplemental and union counts, baseline labels, every saved
frame, persistence episodes and descriptive slopes. No equilibrium conclusion
follows from these two trajectories.

## Vacant cooperative extension sites

The four geometry-only solution quartets are **not terminal under the exact
34-placement catalogue**. Keeping each already selected ideal embedding fixed,
enumeration finds 12, 16, 12 and 12 vacant placements contacting at least two
distinct existing tetramers. Each quartet has four or six three-neighbor
placements. The nine-body geometry-only seed component has 33 cooperative
placements, including one with four neighbors; the sixteen-body informed seed
has 62, including one with five neighbors.

![Exact catalogue extensions and an example ideal quartet](../results/native-oligomer-extension-design-20261004/figure/extension-slots.png)

For every existing body operator H and catalogue placement M, the diagnostic
constructs HM, deduplicates the complete operator, independently checks every
relative contact, and rejects candidates sharing any occupied monomer site.
It retains all 113 nontrivial components and their original embedding
witnesses. All 13,022 products were examined, producing 9,952 distinct candidate
operators before occupied-site rejection. Aliases never inflate the number of
neighboring bodies. A separate output check found no duplicate unordered
monomer-site sets among the vacant operator slots.

This is a native-informed **diagnostic**, not a prior supplied to the blind
proposal. The slots describe isolated ideal components. Site disjointness is
not an atomic hard-overlap test, surrounding-particle clearance, finite basin
volume, or binding free energy. The result rules out a simple catalogue/site
exhaustion explanation for these retained ideal oligomers. The subsequent
clearance and incoming-adjustment tests below address point accessibility;
statistical weights remain necessary before attributing the growth stall to
sampling or thermodynamics.

The subsequent [fixed-point clearance audit](native-extension-clearance-20261004.md)
has now tested all 7,207 site-disjoint slots in the measured endpoints. Only
one of the four quartets' 52 cooperative points is fully hard-clear; the other
51 clash with the quartet itself, with sub-Ångström penetration. Both seeds
retain several open points. A separate ideal-component control then found
all 52 quartet points clear, with minimum gaps of 0.0369–0.0426 Å. The completed
[incoming-only adjustment](native-extension-incoming-search-20261004.md) now
clears 50/52 points against the measured quartet and 34/52 against the full
environment, while retaining every intended native contact. The median
adjustment is 0.377 Å / 0.590°. Sixteen quartet-clear points remain spectator
blocked. The two unresolved searches do not prove cooperative rearrangement
necessary. Point counts do not measure basin weights.

The [procedure and source](../results/native-oligomer-extension-design-20261004/PROCEDURE.md)
record the predeclared inventory and limits. Twelve synthetic controls passed,
including an independent finite-grid oracle, inverse composition, aliases,
gauge changes and occupied-site blocking. Enumeration completed in 1.52 CPU
seconds without atom queries. Receipt SHA256:
`d79ea6d00f6f745d85f5807726054e4eb722171d30e5099430da47e748c8ee9e`.

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
- Full-history observer: twelve synthetic tests passed; all 2,002 frames
  completed in 163.01 CPU / 194.57 wall seconds, using 262,538 monomer contact
  queries. Conservative center pruning retained 1,200,437 new pair queries
  from a complete 69,432,000-pair inventory for the 2,000 noncached frames.
  Both completed final endpoints and every old label were reused. Receipt:
  `/vast/xvg/tetramer-mc-runs/growth-z004-supplemental-history-20261004/analysis/receipt.json`,
  SHA256 `24b6658c85ed9b9bfea8416dd05b8284a6b8bb4c41dad4034c03939025082f86`.
- The independent history reduction checked every union label/pair episode,
  including timing and censoring. Plot receipt SHA256:
  `55c641c42118a941a7976cbbb83e4675daf6925120c33d3bc0a2eaacecdbf6a1`.
