# Incoming-only adjustment recovers native extension points

**Moving only the incoming tetramer finds hard-clear native points for 50 of
52 predefined slots against their measured quartet, and 34 of 52 in the full
saved environment.** The quartet and every spectator remain fixed. The median
adjustment is **0.377 Å translation and 0.590° rotation**. Thus the original
fixed-point clashes usually do not require the quartet to rearrange before a
nearby incoming pose can fit.

This is a native-informed accessibility calculation at the historical
**1.4 Å depletant radius, 0.04 Å⁻³ activity, 500 μM, N=264 spherical endpoint**.
It does not establish template-free assembly or physical stability at the
separate decision conditions of 1.5 Å / 0.035 Å⁻³ / 106.8 μM.

![Incoming-only native extension accessibility](../results/native-extension-incoming-report-20261004/render01/incoming-accessibility.png)

## Three controls on the same 52 slots

The [original clearance audit](native-extension-clearance-20261004.md) found
only **one** fixed candidate point clear of its measured quartet. A subsequent
control placed the existing quartets at their retained ideal poses, under the
same frozen fits: **all 52** candidate points were clear of those ideal
components. That control omitted spectators and walls.

The present calculation returns the quartets to their measured poses and
adjusts only the incoming body. Each original slot receives one bounded
six-dimensional search, initialized at the original point, with translation
norm ≤1 Å and rotation angle ≤2°. A fixed initially passing catalogue label
for each intended neighbor constrains the search. There are no new initializers,
retries, replacement slots or changes to native criteria.

| Quartet | Tetramer IDs | Slots | Originally clear | Adjusted: quartet clear and native | Adjusted: full environment clear and native |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | 16, 77, 217, 237 | 12 | 0 | 12 | 5 |
| Q2 | 42, 184, 194, 206 | 16 | 0 | 14 | 7 |
| Q3 | 62, 67, 80, 134 | 12 | 0 | 12 | 10 |
| Q4 | 69, 89, 91, 229 | 12 | 1 | 12 | 12 |
| **Total** | | **52** | **1** | **50** | **34** |

All 52 selected points remain inside the restricted search domain and retain
all intended native entry and residue-patch predicates. Of the 50 points clear
of their quartet, **16 remain blocked by spectators**. No point is wall-blocked.
There are 17 spectator-blocked points in total, including one of the two
remaining quartet failures; spectator blockage status is unchanged for every
slot. The spectator-limited points have penetrations of approximately
2.57–3.28 Å, compared with sub-Ångström original quartet clashes.

The maximum adjustment is 0.937 Å and 1.894°. The initially clear point is
retained exactly. The displacements are successful witnesses, not estimates
of the minimum movement needed.

## Strict geometry decides success

All **52 numerical optimizer calls report success**, but two final points
still overlap their quartet:

| Q2 slot | Final minimum quartet gap | Translation | Rotation |
| --- | ---: | ---: | ---: |
| 16 | −0.04330 Å | 0.3315 Å | 0.3417° |
| 41 | −0.006856 Å | 0.5078 Å | 0.9281° |

These two searches remain unresolved. Numerical convergence does not prove
geometric feasibility, and failure to find a point does not prove an empty
basin or a need for collective motion.

The objective penalizes penetration with a +0.002 Å numerical margin. That
margin does not change the physical hard core. Final certification uses an
independent complete neighborhood query with the original strict atomic
squared-distance, wall and native-entry predicates. The 50 quartet-clear
witnesses have minimum quartet gaps from approximately +0.0020 to +0.4015 Å.
There are no near-zero flags or strict-versus-1e−8 Å tolerance disagreements.

The search used **399 objective evaluations and 52 final environment queries**,
completing in **2.382 CPU / 3.122 wall seconds** on one worker. It drew no Monte
Carlo samples. The [independent scalar review](../results/native-extension-incoming-independent-review-20261004/analysis-note.md)
verified all slot identities, 104 slot-attempt records, 798 objective begin/end
records, selected-evaluation rules, strict certificates and drained execution.
Its reduction added no geometry queries.

## What remains to test

This result establishes nearby **point accessibility** with the incoming body
alone for most of the specified slots. It does not measure proposal probability,
basin volume, an entry path, equilibrium contact weights or attachment rates.
It neither supplies a reversible transition kernel nor demonstrates blind
contact discovery. The spectator-limited results identify a separate obstruction
in this saved environment, without proving that spectator motion is necessary
for every nearby incoming pose.

The [controlled Gaussian-coverage comparison](native-extension-gaussian-coverage-20261004.md)
has now completed all 26,624 candidates with the same frozen covariance and
paired latent draws at both centers. All 52 slots, invalid draws and
unconditional denominators are retained. Full-environment intended-native
coverage increases from **393/13,312 (2.95%) to 1,997/13,312 (15.00%)**;
quartet-only coverage rises from 4.09% to 24.05%. An independent coordinate,
density and accounting audit passed. This measures probability under the
specified Gaussian kernel, not the depletant-weighted physical ensemble or MC
acceptance. Geometry-only construction and a corrected physical sampling
comparison remain necessary.

## Reproducible artifacts

- Search: `results/native-extension-incoming-search-20261004/`; terminal receipt
  `554070184274c5d8472bd6dd85602dca621a13b8a9e9f0cf8a5d152f9c105ee7`.
- Ideal-component control: `results/native-extension-ideal-control-20261004/`;
  terminal receipt `b547efeb90f34b395c9f2ceffda6d55acc02d128cb774a50f907d5b19e2e1cf9`.
- Independent review: `results/native-extension-incoming-independent-review-20261004/`;
  receipt `3a23a9d491383c7d7391daa231e95ec08856ccc5bef2a4ee38ab6d88952b0c40`.
- [Figure, plotted values and rendering receipt](../results/native-extension-incoming-report-20261004/render01/),
  generated only from authenticated saved summaries. Rendering was limited to
  30 CPU seconds, 60 wall seconds and 2 GiB on one thread; it completed and
  drained in 0.875 wall seconds with no geometry queries or draws.
- [Search procedure](../results/native-extension-incoming-search-preparation-20261004/PROCEDURE.md)
  documents the finite trust region, atom-pair pruning, saved trials and final
  certification. Floating-point transforms and geometry execution remain
  implementation obligations rather than exact-real certificates.
