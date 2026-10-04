# Completed growth trajectory: mobility without further native seed growth

The geometry-only growth run now has its full 100,000-sweep analysis. The native
seed remains at 11 tetramers throughout the newly analyzed 60,900–100,000
interval. All 391 added frames were classified; the previous 610 classifications
and exclusion graphs were reused. The final exhaustive 34,716-pair check agrees
with the accelerated native observer.

These are diagnostic conditions: 264 mobile tetramers, initially eight in the
native seed, depletant radius 1.4 Å, activity 0.0275 Å⁻³ and 500 μM in the
geometric spherical volume. They differ from the decision conditions of 1.5 Å,
0.035 Å⁻³ and approximately 106.8 μM. The loaded atlas is native-blind, with
SHA256 `24fd1d1a2a9da387f671f771e9b2a3bbc870a951c5d730934f50ba6083e5253c`.
The seed and analysis classifier use supplied native geometry; nested
native-informed template metadata does not identify the loaded atlas.

| Observable | Sweep 60,900 | Sweep 100,000 |
| --- | ---: | ---: |
| Cycle-consistent native seed size | 11 | 11 |
| Seed-associated exclusion aggregate size | 16 | 19 |
| Registered pair contacts, whole system | 47 | 51 |
| Exclusion contacts, whole system | 244 | 249 |
| Cumulative transfers satisfying both interface-persistence requirements | 36 | 65 |

The seed has no member entries or exits in the added interval. Thirteen
previously unseen labeled native pairs appear elsewhere, with 17 new
pair-and-motif combinations; these counts are not counts of structural basins.
The same three inverse-motif families, 2/9, 10/12 and
11/13, remain absent outside the original seed. Registered contacts therefore
continue to turn over without demonstrating further seed incorporation.

The fixed transfer diagnostic requires an unchanged internal exclusion graph
throughout the observed free interval and old and new interfaces each present
for at least three consecutive saved endpoints. It gives 29 additional
transfers, none with a registered new interface. The 65 full-history episodes
involve 48 distinct memberships and are not independent samples. The earlier
apparent acceleration does not continue: counts are 7.41 and 7.44 per 10,000
sweeps in the two long comparison windows. Matched late windows contain 13
versus eight episodes in 10,900 sweeps each. Normalizing size-2–5 events by
their component inventory also gives a lower late count. These are descriptive
Monte Carlo counts at a 100-sweep observation cadence, not physical rates.

The persistence filter must not be mistaken for a test of whether any native
association occurred. Five episodes in the complete history have specifically
registered new pairs persisting for at least three saved endpoints; all five
fail only the old-interface requirement. Generic exclusion-contact persistence
is also insufficient: one such contact persists for 452 endpoints, but its
native registry persists for only 57. The separate table and plot are in
`results/growth-tail-native-interface-addendum-20261004/final`.

One additional late episode, involving the group
{16, 36, 125, 156}, acquires native pair 125–149 near sweep 88,300. That pair
remains registered for 118 saved endpoints through sweep 100,000, spanning
11,700 sweeps; its lifetime is right-censored. Its previous interface was present
for only one saved endpoint, so the event fails the fixed transfer filter.

The selected replay resolves its mechanism. At **sweep 88,205**, the learned
`full-mixture-capture` independent proposal moves **body 149 alone**, using body
125 as its anchor and reciprocal atlas component 3. The group does not move
during this update. Exact exclusion graphs of the recorded before/after poses
show an isolated body 149 joining the isolated four-body group through its sole
new edge, 125–149. This demonstrates free single-tetramer docking onto an
oligomer, not native docking by an oligomer move.

The proposal logratio is −29.82568 and the realized Poisson contribution is
+28.52770, giving log acceptance −1.29798. These are one recorded proposal and
auxiliary realization, not an association free energy. The group's two existing
native pairs and two unregistered internal contacts are unchanged. Adding body
149 extends one native chain; it does not register the entire oligomer.

Replay covers all 81,027 records in sweeps 88,001–88,300, including 774 accepted
spatial updates. All 81,268 logged old-pose checks pass. Three saved endpoints
reconcile poses, full native labels and per-kernel attempted/accepted counters.
The four-body group's relative centers vary by at most 1.69 × 10⁻¹² Å over the
accepted-update sequence. This establishes rigid geometry over the replayed
updates; no intervening exclusion graphs were computed. The 118-endpoint
native residence remains a saved-frame observation rather than an every-move
residence measurement over the remaining 11,700 sweeps.

## Evidence and continuation

The complete endpoint analysis is
`/vast/xvg/tetramer-mc-runs/growth-tail-final-20261004`; its execution receipt
SHA256 is `bb6ccb3bf23a625ca5e5aa9845c19851f570894bbd5bbaa2d42dedc22afca8aa`.
The plot, tables and fixed-window definitions are in
`results/growth-tail-figure-20261004`.

The selected replay is
`/vast/xvg/tetramer-mc-runs/growth-native-event-replay-20261004`, with receipt
SHA256 `df0cb95a8d7708f1c44e736d744058466952d2e6e28def13eced833a8c2f0c62`.
Its frozen 104,105,778-byte move excerpt and source closure are under
`results/growth-native-event-preparation-20261004`; no full move-log scan was
needed. The separate two-frame exact contact-context check is under
`results/growth-native-event-context-20261004`.

Preparation, bookkeeping and reporting retain their source-bound inputs and
validation under `results/growth-tail-{preparation,continuation,reporting}-20261004`.
The initial merge stopped because nested JSON object keys were serialized in
a different order. All 610 prefix observations agree under canonical JSON
comparison, including numeric types and array order. A separate continuation
restored the literal cached prefix, preserved the original failure record and
reused every completed geometry calculation. Cached topology, native joins and
all 6,345 historical episode records reproduce the old results. All 61 episodes
that were right-censored at the old cutoff are included in the extension.

This trajectory demonstrates mobility and a persistent native seed, not
equilibrium finite-system assembly or instability. The independent contact-mass
calculation and [corrected surrogate-chain comparison](dimer-surrogate-inner-chain.md)
remain separate requirements. No production kernel or live simulation was
changed by this analysis.

## Completed z = 0.04 proposal comparison

Two other completed 100,000-sweep histories now have the same strict native
observer applied to every saved frame. These use radius 1.4 Å, activity
0.04 Å⁻³, 264 mobile tetramers and 500 μM, with identical initial poses, eight
native seed bodies and the same RNG seed. They are a historical proposal
comparison, not independent replicates or an activity-controlled comparison
with the preceding trajectory. The original 1.5 Å / 0.035 Å⁻³ / 106.8 μM
decision conditions remain separate.

| Completed history | Native-informed coverage | Native-blind FFT 512 |
| --- | ---: | ---: |
| Cycle-certified seed component, initial → final | 8 → 16 | 8 → 8 |
| Maximum seed component | 18 | 8 |
| Final registered edges involving at least one original solution body | 236 | 16 |
| Largest separate cycle-certified native component, final | 10 | 2 |
| Saved seed-member entries / exits | 16 / 8 | 0 / 0 |
| Final-quarter edge gains / losses, whole system | 53 / 50 | 16 / 16 |
| Recorded sampler CPU, seconds | 379,436 | 430,983 |

![Completed native-registry comparison](../results/growth-z004-native-figure-20261004/native-growth.png)

The geometry-only history reaches native pairs but its separate registered
components never exceed dimers. Outside the supplied seed, it visits only
families 3/8 and 6/7; family 4/5 is absent throughout all 1,001 saved frames.
The native-informed history has 60 final outside-seed edges in family 4/5,
53 in 3/8 and 118 in 6/7. It forms larger native oligomers, yet its seed component
has no net growth in the last 25,000 sweeps. Family 2/9 never appears outside
the original seed in either history. These are observed family gaps, not
estimates of zero equilibrium mass or proofs of which families are necessary
for three-dimensional connectivity.

The contrast motivates testing exchange between competing contacts and access
to additional contact families. It does not distinguish a proposal barrier from
thermodynamic preference: the histories have no independent-start convergence
test. Native-edge turnover also does not establish oligomer attachment, which
requires the separate exclusion-interface and move-replay diagnostics used
above. No such geometry or move-log pass was added for this comparison.

Both final exhaustive 34,716-pair checks agree with the accelerated observer.
All 2,002 saved frames, including unchanged states, are retained; no unresolved
or frustrated native components occur. This last statement is a classifier
result, not certification that every exclusion aggregate is a native crystal.
Between-frame excursions remain unresolved, and reported sweep/CPU changes
are not physical rates.

The authenticated analysis root is
`/vast/xvg/tetramer-mc-runs/growth-z004-native-20261004`, execution-plan SHA256
`911b312e323c1e56135b89cd6dfb3141bfc86138b2551eff50eba0ead66aa724`.
The actual coverage and blind model hashes are respectively
`feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e`
and `c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08`.
The native manifests are `e992788eaf76bad0548077f70848784ba7db483d90098f23268e7c071fed6428`
and `835dbc3c30aa18b4f1774866a06db35a263c548c177c340e8f7aa4f02e03be55`.
The complete numeric summary and PNG/SVG figures are under
`results/growth-z004-native-figure-20261004`; presentation receipt SHA256 is
`12d6c42a423e26c98fe7368872116da7c5e7aa5ad99a7b70c2bf75642ef2ae7e`.
The snapshot preparer passed eight synthetic checks and the plotter six;
analysis reused the unchanged archived native observer and definition.

## Exact native embeddings beyond one reference covering

**Every one of the 64 final registered components admits an exact, injective
embedding of its retained labels into the native monomer symmetry orbit.**
This broader calculation resolves all failures of the earlier four-type
tetramer-cover test, including the enlarged informed seed component. Those
failures constrain one selected covering; they do not demonstrate loss of
native monomer registry. The frozen classifier and its original counts remain
unchanged.

| Final component class | Components | Tetramers | Exact monomer-orbit embeddings |
| --- | ---: | ---: | ---: |
| Informed, separate from the original seed | 46 | 212 | 46 |
| Informed, touching the original seed | 1 | 16 | 1 |
| Blind, separate from the original seed | 16 | 32 | 16 |
| Blind, touching the original seed | 1 | 8 | 1 |

The calculation reconstructs eight native basis representatives from seven
previously checked monomer contact operators with integer rotations and exact
rational translations. It verifies all 56 monomer contact relations, all 64
basis products and eight inverses, the prototype's 32 member-site identities
under eight anchor placements, all 36 tetramer-motif occurrences and all 30
prescribed member contacts. Each motif therefore has an exact crystal-group
operator. Propagating those operators through a saved component gives exact
cycle closure and distinct occupied **monomer** sites in every case. No
floating-point transform composition or new geometric query enters this check.

The broader symmetry orbit also changes the interpretation of family
connectivity. The blind history's two observed families, **3/8 and 6/7, already
generate rank-three primitive translations**. Additional contact families
cannot be declared generally necessary for three-dimensional native assembly
from the restricted-cover calculation.

| Available inverse-motif families | Translation rank using only reference-cover occurrences | Translation rank in the unrestricted generated subgroup |
| --- | ---: | ---: |
| 3/8 and 6/7 | 1 | 3 |
| 3/8, 4/5 and 6/7 | 2 | 3 |
| All seven families | 3 | 3 |

For these three unrestricted subgroups, the rotation subgroup has order four
and the translation lattice has index one in primitive `(a,b,c)` coordinates.
Their index in the full native space group is two. This is distinct from the
four-type covering, whose translation cell is `(a,b,2c)`.

Group generation does **not** construct a nonoverlapping infinite packing.
For example, motif 3 squared is translation by c, while the prototype already
contains monomers at the same basis with images zero and c. Occupying every
generated tetramer placement would reuse monomer sites. Conversely, subgroup
index two does not imply that half the monomer lattice is inaccessible, because
the prototype contains several monomer symmetry classes. The finite-component
site-injectivity checks pass; infinite packing and equilibrium stability remain
separate questions.

The [unrestricted-orbit report](../results/growth-z004-unrestricted-native-orbit-20261004/REPORT.md)
and [execution receipt](../results/growth-z004-unrestricted-native-orbit-20261004/receipt.json)
preserve all component assignments and all 128 family-subgroup calculations.
Receipt SHA256 is
`8c1973443cc02302b3697af13e7b3a738c85aba53b4c4343ded8849582a24e5e`;
full-result SHA256 is
`25bf85f703bb281756c8dab6044a3a327af984c4f0d70e0a573ff7a28c9ea886`.
Affine inverse, screw-power, translation-conjugation, lattice-index, reverse
traversal, cycle-conflict and monomer-collision controls passed. This is an
ideal-label embedding for each component, not a fit of the measured perturbed
coordinates to one global crystal, a joint system-wide embedding or an
equilibrium conclusion.

### Preserved diagnostics for the restricted covering

The earlier calculation used the frozen catalogue's four block types and
integer covering-image shifts. Its spanning-forest cycle voltages, ranks and
indices are valid for that selected graph. All 128 family subsets, six lattice
controls, a disconnected-component control and 384 direct modular-cover checks
are retained in the [restricted-topology report](../results/growth-z004-contact-topology-20261004/REPORT.md)
and [receipt](../results/growth-z004-contact-topology-20261004/receipt.json), SHA256
`40b1c1452350eac8ae9cba2b2035fb0beaf6e17646a9311e25da79f3a22d5f8f`.
Its rank-one and rank-two statements do not bound the unrestricted native
symmetry orbit.

Within that restricted graph, 41 informed nonseed components/200 bodies have
rank-two-capable family inventories; five smaller components contain only
6/7. These counts remain useful descriptions of the observed families, without
implying a dimensional limit on general native assembly. The blind history has
16 dimers/32 bodies. The 36 informed and 224 blind bodies with no registered
edge are not thereby physically free; no exclusion graph was used. The
[component inventory](../results/growth-z004-component-topology-20261004/REPORT.md)
and [receipt](../results/growth-z004-component-topology-20261004/receipt.json), SHA256
`64cb5b8bf996504367ce80fb847801a38de2cd40e8353fe04d7238e5edd6bc4d`,
preserve the exact joins.

The subsequent four-root type/image embedding test admitted 19 informed
nonseed components (58 bodies) and rejected 27 (154 bodies), plus the enlarged
16-body seed component. All blind dimers and its eight-body seed passed.
Every rejected root had an unavailable typed motif. The
[restricted embedding report](../results/growth-z004-integer-embedding-20261004/REPORT.md)
and [receipt](../results/growth-z004-integer-embedding-20261004/receipt.json), SHA256
`5a5a8bac689b8d40c77aa6428e9a31520c7bc4fb6e226489a12505f5a9ac5092`,
preserve these obstructions. **All of those component failures are resolved
by the unrestricted native-monomer embedding above.** They must not be
presented as failures of native registry, evidence that missing families
prevent three-dimensional assembly, or evidence of thermodynamic instability.
