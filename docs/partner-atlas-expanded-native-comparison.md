# Expanded native labels in the partner-atlas benchmark

The completed expanded observer changes the native-accessibility interpretation, but does not establish a sampling speedup. Many states called non-native by the original 14-placement catalogue belong to the additional 20 valid native placements. The existing `m4` control spends most of these short trajectories in that additional set. The partner-atlas `m8` arm reaches native placements from both starts, but its generic contact-fingerprint apparent ESS per CPU is below local moves in all eight matched comparisons.

This is a **conditional two-mobile-tetramer benchmark**, pair 27/132 against 262 frozen spectators, at **depletant radius 1.4 Å, activity 0.0275 Å⁻³ and 500 μM**. It is not the decision system at 1.5 Å / 0.035 Å⁻³ / approximately 106.8 μM. Neither these conditional contacts nor these short chains establish finite-system assembly or equilibrium instability.

![Expanded native occupancy and sampling diagnostics](../results/partner-atlas-expanded-native-report-20261004/render04/comparison.png)

The report uses all 48 completed chains: six arms, two initializations, four independent streams per initialization and arm. Every retained rejected state is included. There are 512 warmup blocks followed by 4096 production blocks per chain. The existing local and `m4` controls are reused. The two starts are deliberately different, not independent equilibrium preparations; equal stream indices do not imply paired random numbers across old and new arm families.

## Occupancy and initialization dependence

Entries are arithmetic means of the four separate chain occupancies. They are descriptive finite-record fractions, not equilibrium probabilities. The three columns remain separate definitions; the union is evaluated from the complete saved classification rather than inferred by changing the classifier during sampling.

| Arm | Start | Original 14 | Additional 20 | Union 34 |
|---|---|---:|---:|---:|
| Local | Source | 47.39% | 0.00% | 47.39% |
| Local | Prepared | 0.00% | 0.00% | 0.00% |
| m4 control | Source | 18.12% | 78.69% | 96.80% |
| m4 control | Prepared | 7.80% | 86.02% | 93.82% |
| Partner direct | Source | 41.94% | 37.85% | 79.80% |
| Partner direct | Prepared | 1.39% | 31.72% | 33.11% |
| Partner m1 | Source | 47.07% | 19.00% | 66.07% |
| Partner m1 | Prepared | 5.05% | 0.00% | 5.05% |
| Partner m8 | Source | 21.09% | 48.28% | 69.37% |
| Partner m8 | Prepared | 19.33% | 53.20% | 72.53% |
| Partner flat8 | Source | 31.28% | 46.94% | 78.22% |
| Partner flat8 | Prepared | 0.00% | 11.91% | 11.91% |

The near equality of the `m8` union means is encouraging as an accessibility diagnostic, but conceals incomplete placement-level agreement. Its individual union fractions are:

- Source: 0.5171, 0.9324, 0.3538, 0.9717.
- Prepared: 1.0000, 0.9490, 0.2949, 0.6575.

The between-stream standard errors of those means are 0.1530 and 0.1621. Observed source support is `{empty, 6, 7, 1016, 1029}`; prepared support is `{empty, 7, 1016, 1029}`. Source mean occupancy of placement 6 is 19.34%, whereas prepared chains never observe it. Prepared mean occupancy of placement 7 is 19.33%, versus 1.75% from the source start. Their mean placement-fingerprint histograms therefore have descriptive total-variation distance **0.2271**, despite the union means differing by only 0.0316.

Projecting placements onto catalogue monomer-family signatures reduces this discrepancy to **0.05133**: placements 6 and 7 both map to `C4`. Both starts also observe `C4/C5` and `C3/C4/C5`. This coarse agreement must not replace the more discriminating placement comparison. These family signatures describe the catalogue witnesses of a matched placement, not all atomic bonds currently supported by a noisy pose.

There is also incomplete within-chain exchange. For `m8`, prepared stream 0 remains at placement 1029 throughout production; stream 1 enters it once and has no native-label return. For `m4`, source stream 0 and prepared streams 0 and 2 remain at 1029 throughout production. The other `m4` streams explore a broader placement set, including prepared-only placement 3; the source/prepared mean placement-histogram distance is 0.1799. High native occupancy can coexist with trapping in one native pocket.

## Apparent efficiency, with undefined values retained

The following are medians of **defined per-chain** apparent ESS per full sampler CPU second; `(n/4)` states how many chains have a defined value. Constant descriptors have undefined ESS, not zero or infinite ESS. Medians over different defined subsets are not fair standalone rankings. Sampler CPU includes warmup; observer CPU is separate. The complete per-stream values, fingerprints, counts, source hashes and initialization histograms are in [the numeric report](../results/partner-atlas-expanded-native-report-20261004/render04/numeric-summary.json).

| Arm | Start | Union label-presence ESS/CPU | Union fingerprint ESS/CPU | Generic contact fingerprint ESS/CPU |
|---|---|---:|---:|---:|
| Local | Source | 0.07944 (4/4) | 0.07944 (4/4) | 0.27460 |
| Local | Prepared | undefined (0/4) | undefined (0/4) | 0.27380 |
| m4 control | Source | 0.03285 (3/4) | 0.03398 (3/4) | 0.06912 |
| m4 control | Prepared | 0.04617 (2/4) | 0.04377 (2/4) | 0.09311 |
| Partner direct | Source | 0.03883 (4/4) | 0.04505 (4/4) | 0.09812 |
| Partner direct | Prepared | 0.02313 (2/4) | 0.02267 (2/4) | 0.31852 |
| Partner m1 | Source | 0.03596 (4/4) | 0.04182 (4/4) | 0.13551 |
| Partner m1 | Prepared | 0.08656 (1/4) | 0.08656 (1/4) | 0.29407 |
| Partner m8 | Source | 0.07023 (4/4) | 0.07293 (4/4) | 0.14901 |
| Partner m8 | Prepared | 0.11016 (3/4) | 0.11016 (3/4) | 0.08768 |
| Partner flat8 | Source | 0.02901 (4/4) | 0.03771 (4/4) | 0.08853 |
| Partner flat8 | Prepared | 0.12219 (2/4) | 0.12219 (2/4) | 0.08679 |

For the generic contact fingerprint, `m8`/local ratios in stream order are `[0.595776, 0.653345, 0.323473, 0.429287]` from source and `[0.629848, 0.318245, 0.276896, 0.425374]` from prepared. All eight are below one; the median ratio is **0.42733**. This is a finite-record descriptor comparison, not a proven equilibrium mixing-time ratio. Local moves have a high apparent contact-fingerprint efficiency while failing to find any native placement from the prepared start, illustrating why efficiency must be judged together with coverage and initialization agreement.

Saved native returns can be threshold flicker within one physical contact. They do not establish completed exchanges between physical binding environments. In particular, the `m8` mobile pair never detaches internally in these records; external contact is absent in seven of its eight chains. The source stream 3 has an external-contact fraction of 0.8376. The native observer measures the internal pair only and cannot convert these events into external oligomer docking or growth rates.

The strongest supported update is therefore native accessibility that was hidden by the original catalogue. There is no demonstrated general partner-atlas efficiency advantage, and neither `m8` nor `m4` has passed an equilibrium coverage test here. Further work should preserve the expanded placement labels and the generic contact descriptor, diagnose transitions between the occupied native pockets and competing environments, and test destination feasibility and many-body overlap preservation against matched controls. This comparison alone does not justify promoting an arm to a production assembly campaign or changing any physical acceptance rule.

## Reproducibility and scope

Both observers completed successfully with their child processes drained, no active or unstarted jobs, and no retries or replacements. The report independently authenticates their execution plans, terminal-output hashes, frozen observer code and upstream terminal summaries. It reads completed JSON summaries only: no raw history decoding, geometry calls or new physical samples. It does not repeat the underlying scientific computation or claim that lifecycle success establishes scientific convergence.

The expanded observer retained 221,232 endpoints across all 48 chains, including 196,608 production endpoints. It reused original-14 classifications and added the supplemental-20 observation without changing proposals or the target measure. Source and output bindings are in the [report receipt](../results/partner-atlas-expanded-native-report-20261004/render04/receipt.json). The [bounded execution receipt](../results/partner-atlas-expanded-native-report-20261004/render04/run-receipt.json) records a 30 CPU-second, 60 wall-second, 2 GiB, one-thread limit and successful draining; execution took 1.93 seconds wall time.

The initial report and `render02`/`render03` are preserved as superseded renderings. The numerical summary is unchanged. `render04` also corrects a clipped generic-efficiency point (flat8, prepared stream 3, ratio 2.240104): its axis limit follows the maximum of all 48 plotted ratios with 10% headroom. The final plot was visually inspected; all observed occupancies and native-efficiency values lie inside the other panel limits.

| Artifact | SHA-256 |
|---|---|
| Expanded observer summary | `58c189242bb45246520dd3fe00b065251092ee66f1d4cefc6db954936e1daf38` |
| Generic contact observer summary | `6863c14e9e671e9b9014aaf0f704451522f2a358c832e5c4744b2aa8fe2c6e3a` |
| Final report source | `e09000cb9e2d3df8ee88b8373a8d6f72bd093266d465daaf5375b30883491877` |
| Numeric summary | `2af4f651312d77a24910d3a03a4f020c8c71b45c1934e3a8f90569def58156f9` |
| PNG figure | `919e1ccd96289bd4391de7dd8714401cf25a3b5d3f2c31dbdd1cf4ac2208162d` |
| Final report receipt | `c7939bf26c48dbae2cfe9587ce298aa04e071c58e53f0c0db3ee27bc7ca784ed` |
