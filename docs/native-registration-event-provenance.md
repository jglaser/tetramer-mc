# Provenance of the capped-redraw native registration event

The accepted `fused-p2`, phase 11, event 3 move uses a **geometry/depletion-only inter-tetramer proposal**, not an inherited native-informed contact atlas. This statement concerns the proposal. The benchmark starts from a seeded assembly snapshot, the rigid tetramer's internal native geometry is supplied, and native contact definitions classify the result afterward. Calling the entire experiment blind assembly would be incorrect.

The read-only audit and file hashes are in [the receipt](../results/hard-conditioned-validation-20261001/native-event-provenance.json). It verifies the identical model SHA-256 `24fd1d1a2a9da387f671f771e9b2a3bbc870a951c5d730934f50ba6083e5253c`, exact stored-pose/component identities, and all 608 files listed by the merged, base-fit and refined-fit freeze receipts. No physical sampling or rebuilding was performed.

## Selected branch

The move takes body 213 relative to partner 70. Raw trials 1–7 fail hard validity; trial 8 is the first valid candidate and is accepted. Its label is ordinary single-member branch 5; this event's catalogue contains **zero fused components**. The name of the population, `fused-p2`, must not be interpreted as evidence that fusion produced this event.

| Stage | Exact lineage |
|---|---|
| Recorded virtual branch | 5: reciprocal inversion of base component 2 |
| Merged atlas | `contact-refinement-continuation-v2-20260928/merged-atlas`: original slot 1, replacement rows 1 and 65 |
| Selected replacement | Refined row 1, `contact-refinement-20260928/populations/adaptive-pivot-r00-g00`, local slot 1, refinement stream 13941612840218095269 |
| Refinement data | 512 retained states, 440 unique poses; adaptive burn is discarded before fitting |
| Original reference pose | Optimized slot 1 from `fft-union-discovery-768-20260926/populations/r00` |
| Union start | Exactly the optimized slot 1 pose from `fft-seeded-discovery-512-converged-20260926/populations/r00` |
| Initial FFT contact | Exactly selected entry 1 of `fft-contact-starts-512-20260926/selected.json` |
| FFT grid antecedent | Rotation index 187 of 300,000 super-Fibonacci rotations; translation `[21, -38, -1]` Å, clash class 0; 1 Å grid |

That grid candidate underwent exact hard-core repair, overlap-volume scoring and pose/inverse nonmaximum suppression. Subsequent search and pair refinement used shape and depletion geometry. Original slot 1 has proposal mass 0.06070952268380827; its two replacement components split that mass equally, and reciprocal expansion splits each component equally again. The selected virtual branch has mass approximately 0.01517738067095205. These are proposal masses, not equilibrium basin probabilities.

## Native-input checks and limits

The original FFT and selection script bytes were recovered from commit `98bd117dc6071e198aaef8db26a0329cbec8a1c8` and match the recorded SHA-256 hashes (`f36df6…` and `5250cd…`). Their input paths read the rigid shape, scan peaks and exact overlap scores; neither reads an inter-tetramer native motif, atlas or production pose. The archived discovery/refinement sources likewise read shape plus the supplied initial poses. The exact initial-pose chain above establishes where those poses originated, rather than relying solely on `native_information: false` metadata.

The refinement campaign selected the first 64 recorded slots, with no native-label or score filter, and preserved every selected chain. The merged model retains all 768 original slots: 64 receive refinements and 704 retain their previous charts. Native scaffold files in the continuation campaign belong to later audits and assembly evaluations, not the refinement fit. The merged atlas inherits the shape-only FFT atlas `175cf0ec…`, not `frozen-coverage-reciprocal-mixture.json` or another native-informed atlas.

There is a provenance limitation: the union-start selection has a recorded rule and saved outputs but no archived merge script. For this event the antecedent identity is nevertheless recovered by exact equality of stored poses. This audit follows the selected branch all the way to its FFT row; it does not independently reconstruct every other branch of the atlas or every historical human choice of algorithm parameters.

The supported description is **one accepted native registration event using a geometry-only inter-tetramer guide in a seeded conditional benchmark**. It establishes accessibility. It does not establish spontaneous assembly, equilibrium stability, independent contact sampling or a benefit from fused two-contact proposals.
