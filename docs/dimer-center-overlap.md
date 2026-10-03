# Saved center overlap and proposal excursions

The blind memory atlas has poor overlap coverage even at its Gaussian centers on
this fixed-cloud diagnostic. All 128 archived centers are core-valid, but none
reaches half the saved source's guidance count. FFT and native-informed atlases
already contain core-valid centers whose counts match or exceed the source.
Their saved learned draws include excursions from those high-count centers into
core collisions. Thus missing center coverage and loss of feasibility away from
centers are distinct observed problems; the evidence does not support treating
every atlas as a width-only problem.

This is a descriptive extension of the completed [center-clearance
audit](dimer-mean-clearance.md) and [passive guidance
probe](auxiliary-overlap-probe.md). It reused every archived decoded center pose
and its cached core/contact predicates. No center was decoded again, no hard
geometry was rerun, and no points, poses, proposals, or physical baths were drawn.

Before counting, the new method and its local Python import closure, input
hashes, allocation, and cloud selection were frozen. Selection was fixed to
`case_index=0, attempt=0` for each atlas: cloud IDs **0, 256, 512**. The selected
source is the same root 13 / child 215 context; the three atlases use different
saved clouds containing **4,687 / 4,726 / 4,789** retained points. The cached source
counts are **74 / 68 / 82**, agreeing between m=1, m=4, and the prior independent
analysis. Cloud coordinates, root membership, and raw/transformed/retained
hashes were independently reconstructed with the existing `CountOracle`.

| Atlas | Core-valid / all branches | Core-valid center count: median / maximum | Center/source ratio: median / maximum | Core-valid branches at least source | Their center-label weight |
|---|---:|---:|---:|---:|---:|
| Blind memory, all 64 slots | 128 / 128 | 2.5 / 31 | 0.0338 / 0.4189 | 0 | 0% |
| Blind FFT, 512 slots | 2,042 / 2,048 | 32 / 81 | 0.4706 / 1.1912 | 8 | 11.3716% |
| Native-informed, 178 components | 266 / 328 | 21 / 88 | 0.2561 / 1.0732 | 6 | 18.5127% |

These are finite-cloud counts, not exact overlap volumes. The median treats
each virtual branch equally; the last column sums stored normalized mixture
label weights over qualifying core-valid centers, without renormalizing after
the validity restriction. Those weights are neither Gaussian proposal success
probabilities nor physical or equilibrium weights. At least half the source
count is reached by **0 / 805 / 22** core-valid branches, carrying **0% / 64.1677% /
41.2011%** of total center-label weight. All 128 memory centers lie below that
threshold on this cloud; this does not prove a bound on their true overlap
volumes or on nearby Gaussian draws.

Every center remains in the ledger, including all **68 core-colliding** centers
and all reciprocal branches. Invalid centers cannot be interpreted as usable
high-overlap contacts: the 62 colliding native-informed centers have a maximum
count of 4,297, or 52.40 times the source count. Their large values inflate the
unrestricted distribution. The analysis separately reports all centers,
core-valid centers, core-valid exclusion contacts, and core collisions. Of the
core-valid centers, 76 memory, 2,042 FFT, and 264 native-informed centers have
cached exclusion contact. A real exclusion contact can still receive zero hits
from a finite cloud.

The directly matched draw comparison reuses only the selected clouds' saved
internal draw prefixes. Each m=4 prefix contains 32 draws and includes its m=1
prefix; common draws are counted once. All **96** raw draws, including uniform
draws and geometry failures, are retained in the join. The following table
isolates learned draws because only they have a selected atlas-center label.

| Atlas | Learned draws | Core collisions | No exclusion contact without a core collision | Internal geometric passes with cached counts |
|---|---:|---:|---:|---:|
| Blind memory | 19 | 3 | 11 | 5 |
| Blind FFT | 16 | 16 | 0 | 0 |
| Native-informed | 17 | 8 | 1 | 8 |

All 16 FFT learned draws started from labels with core-valid centers and ended
in core collisions. Three selected centers also had counts at least as large
as the source. Native-informed draws likewise include three excursions from
core-valid centers at or above the source count into core collisions. Seven of
its eight learned core collisions came from core-valid centers; the remaining
one came from a colliding center. These are exact joins of cached predicates,
not additional geometry calculations.

For memory's five geometrically valid learned draws, every cached draw count
was below its own center count: the paired change ranged from −7 to −3 points,
with median −4. Native-informed has eight such pairs: four lower, three equal,
and one higher, with median change −1. FFT has no geometrically valid draws in
this selected prefix, so it has no paired draw-count comparison. Counts were
not recorded on rejected geometry; missing values remain missing, not zero.

The selected m=1 candidates have counts **4 / unavailable / 37** for memory /
FFT / native-informed. All three selected m=4 attempts exhausted their caps;
their thresholds were **53 / 67 / 62**. These six selected attempts should not be
confused with the full passive screen: over 256 outers per atlas and arm, the
cached m=1 candidate counts were **16 / 52 / 142**, and m=4 counts were **0 / 10 /
104**. Full-screen candidate-conditional median retention was **0.083 / 0.287 /
0.833** at m=1 and **unavailable / 0.621 / 0.987** at m=4. Those broader draws use
their own clouds and source contexts; their conditional summaries are not
matched estimates of a center-to-draw loss.

The stopped prefixes condition on root-stage success, geometric filtering,
thresholds, and caps. Their counts cannot estimate full Gaussian failure mass
or identify which translation, rotation, correlation, or curvature direction
caused the losses. Likewise, reciprocal centers can have different counts on
one finite cloud even when their exact overlap volumes agree. Clouds differ
between atlases, and counts sharing a cloud have dependent sampling errors.
There was no native-label selection or atlas modification. These observations
do not establish physical acceptance, assembly efficiency, or a favorable
Metropolis correction.

The allocated **2,504 center queries and three cloud reconstructions** completed
once with one scientific worker in **5.32 CPU seconds**. A separate ledger-only
review passed **8,136 checks**, including exact preservation of all archived
center rows, independent summary arithmetic, cached draw joins, input and output
hashes, and all invalid centers. It made no count or geometry queries.

Reproducible artifacts are the [frozen allocation and
hash receipt](../results/dimer-center-overlap-20261003/freeze.json),
[configuration](../results/dimer-center-overlap-20261003/config.json),
[all center counts](../results/dimer-center-overlap-20261003/execution/centers.jsonl),
[saved raw-draw join](../results/dimer-center-overlap-20261003/execution/saved-raw-draw-join.jsonl),
[analysis](../results/dimer-center-overlap-20261003/execution/analysis.json),
[terminal receipt](../results/dimer-center-overlap-20261003/execution/terminal.json),
and [ledger review](../results/dimer-center-overlap-20261003/review.json).
The [reusable method](../tools/analyze_dimer_center_overlap.py) freezes its own
import closure and refuses to overwrite an existing run or execution directory.

| Binding | SHA-256 |
|---|---|
| Frozen configuration | `633c39dd942cd9836a510715fa58627716e7034c66a2ee821946186406e42d42` |
| Center-count ledger | `4cb4af036be7751f2233e071fd532ef9f5b44a3ed2f574dc5b79c9fbd20d70a5` |
| Saved raw-draw join | `2ba8abe63fbbdfda5bd171fb19b51044607a9ac78ecce8095855cb4432dd2a3e` |
| Analysis | `e895424784c3f38135990f9afe5a59928b486952e813fa1cb5c400343ae96481` |
