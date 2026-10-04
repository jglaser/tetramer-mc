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
