# Two-distance conditioning finds native geometry but does not beat old92

**Do not launch a physical contact-weight campaign with these laws.** The
frozen three-arm passive comparison and the separately declared native observer
completed successfully: **512 fresh poses per arm**, four independent streams
each, plus 618 archived density queries. All 15 proposal jobs passed independent
audits. There were **zero Poisson clouds** and no new physical mass estimates.
The original thermodynamic convergence failures and assembly question remain
unresolved.

![Two-distance passive comparison](../runs/contact-distance-passive-review-20261001/contact-distance-passive.png)

## What improved, and what did not

| All attempted poses retained | Old92 | Uniform azimuth | Localized azimuth |
|---|---:|---:|---:|
| Fresh attempts | 512 | 512 | 512 |
| Hard-valid inside R4/capture | 96 | 61 | 75 |
| Complete native entry | 32 | 18 | 33 |
| Contact without native entry | 64 | 43 | 42 |
| Cycle-consistent entry to both scaffolds | 25 | 16 | 33 |
| Both surface gaps ≤0.02 Å | 2 | 1 | 9 |
| Both surface gaps ≤0.1 Å | 20 | 8 | 30 |
| Both surface gaps ≤0.5 Å | 66 | 36 | 52 |
| Fresh proposal-only CPU per attempt | 0.940 ms | 1.609 ms | 1.662 ms |
| Native events / fresh CPU second | 66.5 | 21.8 | 38.8 |
| Competing-contact events / fresh CPU second | 133.0 | 52.2 | 49.4 |

The exact polygon construction removed the previous empty-line problem:
**none of the 118 uniform-azimuth or 122 localized-azimuth conditional draws
used a geometric fallback.** But satisfying two selected atomic contacts does
not ensure survival against the remaining atoms or the original domain:

| Selected conditional branch outcome | Uniform azimuth | Localized azimuth |
|---|---:|---:|
| Outside R4/capture | 103 | 37 |
| Inside domain, hard overlap | 11 | 64 |
| Valid native entry | 2 | 17 |
| Valid competing contact | 2 | 4 |
| Total conditional attempts | 118 | 122 |

Thus conditional survival improves from **4/118 to 21/122**. Localizing the
azimuth preserves much more of the old Gaussian's translation information,
and all 17 native conditional survivors have cycle-consistent two-scaffold
registry. Hard overlaps become the main remaining loss within the domain.
Nevertheless, the complete localized mixture yields essentially the same
number of native poses as old92 at **1.77× passive cost**. Its observed native
events/CPU are 0.58× baseline, and competing-contact events/CPU are 0.37×.
These are geometry/cost observations, not estimates of physical mixing speed.

The tightest-contact count, 9 versus 2, is worth recording but remains
exploratory: corresponding fractions are **1.76% ±0.67% versus 0.39% ±0.23%**,
where ± is the descriptive SE across four populations. This small count does
not establish a useful speedup or greater physical weight. The figure retains
that uncertainty rather than treating all nine as independent evidence of a
new equilibrium basin.

## Full native labels and coverage

The observer applied the unchanged complete native definition only to valid
fresh poses, including member displacement, orientation, shared residue-patch
contacts and cycle consistency. Every invalid/outside attempt remains in the
denominator with null native labels. Full exclusion contact uses the original
**surface gap <3 Å** condition; it is separate from the near-core guide widths.
No valid unbound pose or native-without-contact anomaly occurred in this small
allocation. That does not establish zero unbound or unseen-region mass.

This remains a **native-informed integration guide** inherited from old92.
Geometrically choosing atom labels from its means does not establish blind
discovery. Neither native labels nor physical weights filtered component
selection in this experiment.

The same 128 breadth probes were scored by all three arms. The analytic
**q≥0.5 q_old** bound was respected, retaining full original support. For
localized azimuth, median log-density changes are −0.291 in native R5,
−0.290 in the native complement and approximately zero in competing contacts.
Individual queried competing poses approach the allowed factor-two loss.
Support preservation prevents outright removal; it does not guarantee good
allocation or bound unobserved physical mass.

## Archived weight diagnostics argue against escalation

The previously measured critical competing-55 rows remain separated by their
original sampling source:

| Original source | Saved rows | Uniform-azimuth paired M2 / old92 | Localized-azimuth paired M2 / old92 | Localized contribution ESS | Largest localized term |
|---|---:|---:|---:|---:|---:|
| 84-component pilot arm | 4 | 1.875 | 1.593 | 1.72 | 74.7% |
| Expanded 92-component pilot arm | 74 | 1.936 | 1.208 | 3.01 | 49.3% |

Lower M2 would be favorable. Both candidates worsen these observed second
moments, and their tiny contribution ESS makes extrapolation unreliable.
These are reused-cloud retrospective ratios, **not new normalizers**. Together
with lower broad/native efficiency, they do not justify a physical campaign.
This negative proposal result is not evidence against the physical model.

## Validation and reproducibility

Maximum independent reconstruction errors were **2.99e−13** in full log density,
**3.42e−13** in old92 log density, **3.56e−15** in log Jacobian,
**2.45e−15** in retained angular coordinates and **1.78e−14** in azimuth inverse
replay. Radius samples also replayed, with maximum error 4.45e−16 Å. The reference
checks passed before execution; all attempted/probe identities and immutable
source hashes were verified.

The passive run took 83.25 seconds wall time, including independent auditing.
Native postprocessing used 3.46 CPU seconds. The table's fresh CPU includes
proposal generation, full density, endpoint geometry and output; startup,
archived probes, independent audits and classification are recorded separately.
These timings do not imply a physical-simulation slowdown or speedup.

- [Bound review, all costs and 105 source hashes](../runs/contact-distance-passive-review-20261001/summary.json)
- [Completed proposal/audit analysis](../runs/contact-distance-passive-20261001/analysis.json)
- [All-attempt native classifications](../runs/contact-distance-native-diagnostic-20261001/analysis.json)
- [Frozen preparation and diagnostic declaration](contact-distance-passive-preparation.md)
- [Reproducible report helper](../tools/report_contact_distance_passive.py)
- [Vector figure](../runs/contact-distance-passive-review-20261001/contact-distance-passive.svg)

No additional draws, allocation extension, physical campaign or classifier
retuning followed these results.
