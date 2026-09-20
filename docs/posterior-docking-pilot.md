# Posterior-source transport: bounded docking pilot

Correlated transport completed three native-core–competitor roundtrips in this pilot; the matched independent redraw completed none. These are also roundtrips to the **previously frozen R3 competitor region**, rather than only crossings of a distant-registration threshold. The broader contact diagnostic does **not** show an overall speedup. One correlated trajectory instead found a persistent contact region outside the integrated R8 domain, exposing a remaining coverage question.

## Matched experiment and balance

The physical target is unchanged: one moving rigid tetramer, one fixed site0 neighbor, an 18 Å capture ball, all proper orientations, depletant radius 1.5 Å, and activity 0.035 Å⁻³. The frozen 116-component proposal is
`runs/smc-normalizer-mis-refined/site0/model.json`, SHA256
`5be4efd1710a542f52a982036780ccd05a07fb27ba04b403f36a650b9e740afd`.
Its retained parent includes native information. This is a controlled sampling benchmark, not template-free assembly or online learning.

Both methods use `posterior-involution`, a separate 10% uniform branch, and exactly two local attempts per cycle: translation scales 0.2/2 Å, rotation scales 1.5/15°, and rotation probability 0.5. The endpoint Poisson gate uses λ/z=16, with 2047 maximum cells, depth14, and minimum width0.5 Å. Only the latent correlation differs, c=0 versus c=0.9. The c=0 method is an independent redraw **within the Gaussian branch**; it is not the older full-mixture-MH control.

For Gaussian density G(x)=Σₐwₐgₐ(x), the source label is drawn with probability wₐgₐ(x)/G(x); the destination has fixed probability w_b. In chart latents, z_b=c z_a+√(1−c²)η and reverse noise η′=√(1−c²)z_a−cη. Swapping labels reverses this map. Combining the chart Jacobian, Gaussian-noise density ratio, and forward/reverse label probabilities gives log G(x)−log G(y). The uniform branch retains its separate, symmetric correction. The existing exact implicit-depletion gate supplies the physical correction. Proposal labels and their weights are auxiliary; they are not physical basin probabilities.

There are eight runs of 5000 cycles: two selected native starts and two selected deep SMC endpoints, each paired across c. The four seeds are 98591010, 98592019, 98691010, and 98692019. Native initial q values are 0.54225 and 0.96533; deep initial values are 27.75773 and 27.78346. These starts are not claimed to be equilibrated. Binary, model, shape, source configurations, selected starts, and relevant source files were copied and hashed before launch. The archived executable predates the source commit but contains the tested posterior kernel.

Run root: `runs/posterior-docking-mis-5000`.

## Results and exact event records

| Observable, four runs per method | c=0 | c=0.9 |
|---|---:|---:|
| Global attempts | 20,000 | 20,000 |
| Accepted global pose changes | 749 | 1,562 |
| Accepted native global candidates, q≤1 | 0 | 5 |
| Native-core↔strict-far roundtrips | 0 | 3 |
| Native-core↔frozen-R3 roundtrips | 0 | 3 |
| Sampler CPU seconds | 363.443 | 405.885 |
| Accepted global changes / sampler CPU second | 2.061 | 3.848 |
| Apparent contact ESS, summed across records | 253.47 | 214.27 |
| Apparent contact ESS / sampler CPU second | 0.6974 | 0.5279 |

Native core means q≤0.8; strict far means q≥5. Both require depletant-exclusion contact. The q≤1 native region is used separately for occupancy and dwell intervals. The second native start begins outside the stricter core, so its departure is not counted as a native-core departure. Every actual accepted native-core entry was a global transport move.

| c=0.9 trajectory | Native-core/far visit sequence, cycles | Native q≤1 dwell after return | Far-arrival R3 radius |
|---|---|---|---|
| native start0 | native0 → far7 → native471 | 471–478, 7 cycles | far7: 1.993 |
| native start0 | native471 → far478 → native3193 | 3193–3195, 2 cycles | far478: 2.614; far3195: 1.717 |
| deep start0 | R3 at14 → native-core1022 → far1028 | native-region entry1019, exit1028: 9 cycles | R3 at14: 2.380; far1028: 1.459 |

Deep start0 initially lies outside R8 (radius9.604), but enters R3 at cycle14, attempt1, by a local move before its native return; its R3 tracker independently records the completed R3→native→R3 roundtrip. A further one-cycle native-shell visit at 3497–3498 does not enter the native core. Full attempt indices, CPU timestamps, native dwell episodes, and R3/R8 classifications are retained in each run's `analysis.json`. The three core-return cycles are marked in the figure.

![Matched registration trajectories](../runs/posterior-docking-mis-5000/assessment/registration-trajectories.png)

Native occupancy over all retained cycles is [0.0016,0.0014,0,0] for c=0 and [0.0022,0.0002,0.0020,0] for c=0.9, ordered native0/native1/deep0/deep1. These short records are not an equilibrium estimate, and three versus zero events is not a precise rate comparison. First/second-half occupancies and q statistics are reported separately for every trajectory.

## Contact relaxation and start dependence

The contact descriptor is a binary vector identifying mobile and fixed atoms whose depletant-expanded spheres overlap any sphere in the other particle. It detects changes beyond the single neighbor-contact bit. It retains redundant internal atom spheres and is neither a crystallographic patch classification nor an overlap-volume estimator. Exact sphere-pair geometry is evaluated every10 cycles, giving 501 observations per run. The earlier provisional 50-cycle analysis is superseded by this finer cadence.

| Start | Replicate | c=0 contact ESS, whole / last half | c=0.9 contact ESS, whole / last half |
|---|---:|---:|---:|
| Native | 0 | 54.04 / 38.23 | 105.02 / 58.10 |
| Native | 1 | 83.11 / 38.49 | 37.68 / 15.64 |
| Deep | 0 | 87.30 / 51.61 | 62.37 / 27.11 |
| Deep | 1 | 29.02 / 16.70 | 9.21 / 23.01 |

These are **apparent, descriptor-specific** effective counts. They use sample-centered vector autocovariance, summed across atom components, then the standard Geyer positive monotone pairs (ρ₀+ρ₁),(ρ₂+ρ₃),… and τ=−1+2Σ pairs. Summation stops at nonpositive pairs or half the record, with τ floored at1. The code also reports the unfloored value and last included lag. Sample initialization, changing contact statistics, unvisited modes, and sampling cadence prevent an equilibrium ESS or mixing-time claim. Moreover, the fixed local/local/global cycle is invariant but need not itself be reversible; the reversible-chain spectral guarantees of this truncation are not asserted for that composition. The calculation is checked against IID and positive-AR(1) controls, constant records, and a deterministic mixed negative/slow-positive autocovariance that exposes the incorrect shifted-pair truncation. It does not mutate the q input record.

Last-half mean-contact weighted Jaccard distances between native/deep starts span 0.0268–0.0746 for c=0 and 0.0215–0.1456 for c=0.9. The latter discrepancy is dominated by deep start1. Its contact statistics change over the run, with substantial differences between its two halves. Agreement has therefore not been established even though all runs spend most time at q≈28. The 1.87× accepted-pose-change rate does not translate into an overall contact-ESS improvement; the observed ESS/CPU ratio is about0.757.

## A newly visited region outside R8

The frozen R3/R8 ellipsoids use the original deep chart, not the refitted proposal covariance. Seven trajectories spend 94.8–99.98% of cycles in R8; c=0.9 deep start1 spends only41.2%. It has 2940 retained frames outside R8, with q27.626–28.252 and latent radius8.028–11.570. Outside-R8 dwell intervals are 1723–1733, 1912–2601, 2650–2654, 2763–2765, and 2770–5000. The final pose has q28.0436 and radius9.4503.

![Frozen competitor-region coverage](../runs/posterior-docking-mis-5000/assessment/deep-region-trajectories.png)

The deterministic cohort in `assessment/outside-R8-cohort.json` contains 32 equally spaced chronological selections from those outside frames, including the final frame; 30 poses are distinct. Selection did not use physical weights. Each includes the pose, cycle, original q, frozen latent radius, complete Gaussian log density, exact contact atom IDs/counts, and hard-validity check. Cohort Gaussian log densities range from −3.774 to3.510. Relative to the frozen deep chart's mean pose, these selected poses differ by 2.71–3.54 Å in center and 6.69–9.61° in proper orientation. This is a local geometric extension of the known contact arrangement; its physical mass has not been measured by the earlier R8 integration. A persistent finite trajectory does not by itself prove a thermodynamic basin or determine its weight.

The [independent regional check](outside-r8-local-region.md) is now complete. The prolonged excursion spends 54.39% of its retained frames in a region whose disjoint-set population bound, evaluated at independently estimated masses, is 0.516%. Common-region integrations agree; rare-weight and selected-region limitations remain explicit. The same report audits attempted returns: 261 geometrically valid learned proposals reach old R8, but their median full-G proposal correction is −10.844 while the median fresh Poisson factor is +0.606. No Poisson weight persists in the chain state. This identifies a proposal-coverage issue worth testing directly.

The earlier option of 16 runs of 25,000 cycles (400,000 cycles, about 2.1 summed CPU hours) **remains unlaunched**. A separate [eight-run, 2000-cycle frozen-atlas escape control](outside-r8-atlas-extension.md) is complete: adding the already frozen local Gaussian moves first old-R8 returns from 69–891 cycles to 1–16 at the same two initial configurations. This is a targeted proposal-coverage test, not the longer rate extension or an equilibrium mixing claim.

## Validation, cost, and reproduction

The independent analyzer replays all120,000 attempts and checks all35,980 learned forward/reverse full-G densities and posterior label corrections. It reconstructs2018 deterministically selected chart maps, checks acceptance arithmetic for every gated proposal, reconciles retained states and counts with checkpoints, and checks exact sphere geometry at4008 trajectory frames plus32 cohort poses. Maximum independent full-G discrepancies are 5.4×10⁻¹⁴ for old states and5.3×10⁻¹³ for candidates; posterior-label and expanded/collapsed corrections differ by less than2×10⁻¹². No numerical-map exception was waived.

Total sampler cost is769.327 CPU seconds and200.916 campaign wall seconds on four workers. Only2.001 CPU seconds are charged to proposal construction;759.976 seconds are charged to the implicit-depletion gate. Offline analysis costs about37.8 summed process CPU seconds, with the contact observer taking about1.8 wall seconds per run. Observer and plotting costs are excluded from sampler efficiency denominators. Sampling CPU includes runtime diagnostics; it is not total end-to-end workflow cost.

The two new tools accept the body-relative frozen atlas with its actual component count; they do not preserve the older hardcoded5/28-component assumption. To reproduce into a fresh directory:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B tools/run_posterior_docking_pilot.py \
  --out runs/posterior-docking-new --cycles 5000 --workers 4
/home/xvg/protein-nucleation/.venv/bin/python -B tools/analyze_posterior_docking_pilot.py \
  runs/posterior-docking-new --workers 4
```

The launcher refuses to overwrite a nonempty directory. Production used the binary archived under the original campaign's `provenance/`; a later rebuilt executable will have a different provenance. `manifest.json`, `assessment/analysis.json`, `assessment/outside-R8-cohort.json`, per-run reports, and PNG/SVG figures contain the reviewable inputs and outputs.
