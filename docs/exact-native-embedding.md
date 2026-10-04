# Exact native component embeddings

`tools/exact_native_embedding.py` adds a supplemental graph diagnostic. It does
not change the frozen pair classifier, the finite-assembly region definitions,
or any running sampler or observer.

The existing cycle checker asks whether the retained pair labels admit ideal
relative poses. A tree passes that check even if the propagated arrangement
assigns two monomers to the same native lattice site. That is insufficient for
an injective crystal arrangement. The completed unrestricted-orbit calculation
already checked injectivity for 64 final growth components, all successfully;
those results are reused, not recalculated here.

## Construction

An affine operation is `(R,t/d)`, with a proper signed-permutation matrix `R`,
integer translation numerator `t`, and one positive common denominator `d`.
This representation supports the current tetragonal crystal. It is not a
generic representation for every crystallographic point group.

For pair label `m` on the ordered pair `i<j`, require `G_j=G_i H_m`.
The four monomers in body `i` occupy operations `G_i M_k`. Exact equality of
these operations identifies a repeated monomer site. This uses the asymmetric
monomer's trivial stabilizer; it does not implement a symmetry quotient for a
different body. Native group membership and the correspondence of the supplied
operators to the rigid body must be authenticated by the caller.

The deterministic search places one additional body at a time. It retains all
labels on each pair as alternatives and checks every closed edge and occupied
site during search. A collision prunes that branch, allowing another label to
succeed. Testing injectivity only after accepting the first cycle-consistent
labeling would incorrectly reject some embeddable graphs.

Each connected component has its own root gauge. The result therefore makes no
claim about the relative arrangement of disconnected components. Components
without an ordinary-space periodic lift remain unresolved; this implementation
does not decide quotient-space crystal embeddings.

## Results and work limit

`check(body_count, native_keys, max_search_nodes=..., non_lifted_components=...)`
returns a certificate for every component:

- `embedding_found`: an injective assignment exists. The output includes the
  exact body operators, occupied sites, and one permitted label on every edge.
- `no_admissible_embedding`: all alternatives were exhausted. Diagnostic prune
  counts do not identify a unique cause for failure across different branches.
- `search_budget_exhausted` or `non_lifted_periodic_component`: unresolved,
  with `embeddable=null` rather than false.

The global search budget counts attempted non-root assignments, including
rejected candidates. It does not bound input parsing, candidate construction,
or serialization cost; callers still need ordinary resource limits. There is
no growing cache. Isolated bodies need no search and have independent gauges.

Successful ideal-label embedding is not a fit of the measured coordinates,
an atomic nonoverlap certificate, a completion to an infinite packing, or an
equilibrium claim. A failed diagnostic must not retroactively alter the old
region definitions or discard a physical sample.

## Validation

Fourteen tests passed, including comparison with an independent exhaustive
edge-label enumeration on 252 small graphs. Tests cover reverse traversal,
noncommuting screw cycles, cycle conflicts, nonadjacent monomer collisions in
trees, alternative-label rescue and rollback, exhausted searches, independent
component gauges, periodic unknowns, strict records, and permutation/duplicate
invariance. Every successful test witness is checked independently.

The bounded test receipt is
`results/exact-native-embedding-validation-20261004/attempt01/validation.json`,
SHA256 `38949e7e5af7ba27d5f1f17f4bf0c58b7d681489c21fadb70e249f7000f18e88`.
No protein geometry, trajectories, or physical samples were evaluated by this
validation. Current running campaigns retain their original implementations.

## Contact-catalogue coverage

A separate finite calculation enumerates
`H = M_i C_k inverse(M_j)` for four source members, the seven authenticated
monomer contact operations, and four target members: exactly 112 constructions.
After deduplication these give 43 operations. Nine reuse a monomer site, leaving
34 site-disjoint placements. The existing 14 motif operators are a subset;
20 additional placements are absent from that catalogue.

The result is saved in
`results/unrestricted-native-catalogue-coverage-20261004/analysis.json`.
It reuses the completed unrestricted-orbit authority and does not redo its
group discovery, growth classification, or physical calculations. The receipt
SHA256 is `fc39a357eae8fb4f018c5803643a3d2e7a1120bc23f87828d5664dcbdc73b7db`.

This finite coverage statement is relative to the seven supplied monomer
contact classes, not every possible contact definition. Absence of shared sites
alone does not establish atomic hard validity. The existing 14-label native
weights retain their original meaning; additional candidates belong to a
separately defined supplemental region.

![Exact contact construction coverage](../results/unrestricted-native-catalogue-coverage-20261004/figure-v2/catalogue-coverage.png)

Cells enumerate all member/class/member witnesses; repeated numbers denote the
same exact operation. Cell counts are not statistical weights. The plot reads
only the completed coverage output. Its first layout is retained separately;
the second changes spacing without changing the plotted values.

### Completed atomic checks

All 20 additional placements pass the strict hard-atom predicate without
relaxing their poses, coordinates, or radii. Their minimum surface gaps range
from 0.036918 to 0.911126 Å, and each has atomic contacts within 2 Å
(17–851 atom pairs). Sparse spatial-tree queries and exhaustive calculations
agree on every recorded threshold count. The exhaustive calculation evaluated
320,640,320 atomic pair distances across the 20 placements; existing catalogue
geometries were reused rather than reevaluated.

The tetramer reconstruction matches the physical shape exactly. All 112
member/contact/member constructions agree with their Cartesian transforms;
the 14 existing catalogue poses serve as coordinate controls. The calculation
uses serialized proper quaternions and explicitly checks matrix roundoff.

The completed result is
`results/unrestricted-native-catalogue-geometry-20261004/analysis.json`, with
receipt SHA256
`5780f0a8f45b53a7d73f72ad3883c0be7522512d5ea656d6104248ad74c4bda6`.
The bounded run used 6.421 CPU seconds, one worker and one thread.

All 20 are also necessarily excluded by the old classifier at their exact
poses. A separate metadata-only assessment compared every placement with all
14 old templates using the unchanged 15° body-orientation gate and the maximum
of the four member-center errors (at most 2 Å). None passes both gates for any
template. All 280 comparisons are retained in
`results/unrestricted-native-entry-body-gates-20261004/analysis.json`; receipt
SHA256 is `c12e3fad69683a6ba35f7f1c95d199ad5018c53853ad1256fc7bf75751d846f2`.
Five synthetic controls passed, including a case distinguishing the maximum
member error from center-of-mass or mean error. This step made no atom queries.

These are structurally allowed native-monomer contacts, not measured basin
weights or evidence of equilibrium stability. The completed
[supplemental endpoint diagnostic](expanded-native-contact-diagnostic-20261004.md)
finds 34/70 additional contact pairs in the informed/geometry-only endpoints,
using the unchanged supporting native-residue predicates. All 113 expanded
components admit exact injective embeddings and have measured member-center
RMS error below 0.95 Å after one rigid alignment per component. This does not
prove a complete tetramer packing or equilibrium assembly. Running observers
and physical-weight regions retain their frozen definitions.
