# Full captured-domain integration and an alternative native pocket

The full-support calculation found an alternative native pocket with registered
contacts to both fixed neighbors, outside the original registration score.
It did **not** converge the physical
normalizer. This improves our contact inventory and identifies the next
regional reference; it is not evidence against the model or a demonstration
of equilibrium assembly.

## Fixed target and independent populations

The [frozen campaign](../runs/mobile-full-capture-campaign-20260921/protocol.json)
keeps the exact observed two-neighbor scaffold, the 170 Å capture ball,
depletant radius 1.5 Å and activity 0.035 Å⁻³. It draws four independent
populations of 8,192 poses under each of two proposal laws, with uniform
fractions 0.1 and 0.5. Both marginalize the two proposal anchors and use the
same reciprocal 150-base-component atlas. Two independent Poisson factors
per valid pose use λ/z=64. Invalid draws retain zero weight and their place
in the unconditional denominator. No population was extended or discarded.

The original partition is exhaustive: original q≤1; q>1 within competing
chart R12; and the remaining q>1 poses. A separately frozen
[instantaneous native-contact classifier](native-contact-regions.md) tests
all catalogue sites, with no persistence or trajectory-dependent labels.
The q>1 remainder can therefore contain native placements.

| Uniform fraction | Observed total log Qz | Row relative SE | Population relative SE | Importance ESS | Largest row |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 58.973 | 61.0% | 48.6% | 2.7 | 49.1% |
| 0.5 | 60.650 | 92.2% | 96.4% | 1.2 | 92.1% |

These are unconverged estimates, not accepted free energies. The full
[comparison](../runs/mobile-full-capture-comparison-20260921/report.md) and
[machine-readable analysis](../runs/mobile-full-capture-comparison-20260921/analysis.json)
retain every population, paired-cloud noise, hard-reference weights,
shared-draw ratio covariances, and region subdivisions. The eight physical
populations used about 130 CPU seconds, excluding analysis.

Both proposal laws miss the previously measured native R4 entirely. Its
independent regional estimate is log Qz≈36.085, whereas the globally sampled
original q≤1 estimates are only 28.079 and 34.039. This is a direct coverage
warning: full support does not make a finite sample complete. Neither a
missing region nor its tiny estimated occupancy can be treated as zero.

## What the new poses establish

The dominant observed contributions lie in the q>1 remainder and satisfy
native entry against both fixed neighbors, including a catalogue-consistent
triangle. The [saved-row geometry diagnostic](../runs/mobile-native-remainder-diagnostic-20260921/analysis.json)
identifies directed motifs 7 and 4 against original bodies 2 and 1,
respectively. The original native site uses motifs 4 and 10. The new ideal
site is about 87.15 Å away with a 180° relative orientation difference;
it was not a small fluctuation of the originally scored site.
The catalogue prescribes four external monomer contacts there (two C4 from
body 2 and two C3 from body 1), compared with three at the original site
(two C3 from body 2 and one C4 from body 1).
This is a geometric distinction; the saved cloud factors alone do not
establish the resulting free-energy difference or its mechanism.

Independent full-atom checks of the two largest contributing rows in every
population confirm hard validity and clearance from the original spherical
wall. The ideal catalogue triangle closes, but the two ideal placements
composed from the slightly deformed observed anchors differ by about
0.136 Å and 0.072°. Each ideal placement itself slightly clashes with the
other fixed neighbor. The sampled nearby poses are valid. A regional
integral must preserve the actual scaffold and count these hard-invalid
parts as zeros, rather than repair the scaffold to favor this site.

The weight concentration prevents assigning an equilibrium probability to
this pocket. A few high-weight observations also cannot justify fitting a
new covariance as if it represented the basin. The next
[independent regional design](mobile-native-pocket-reference-design.md)
uses an existing well-conditioned chart, an ideal-site center and uniform
latent-volume integration, avoiding the global mixture denominator.
Region-size sensitivity and continued remainder coverage remain necessary.
That [independent core/shell calculation is now complete](mobile-native-pocket-reference-results.md)
and confirms the signal, while demonstrating consequential mass at the
outer boundary of the newly measured region.

## Audit correction and reproducibility

All eight physical populations completed successfully. The first automatic
density audit failed because its NumPy/LAPACK Cholesky factor differed from
the sampler's scalar Cholesky for ill-conditioned atlas covariances. Both
were backward-stable, but only the latter is the factor actually used to
draw the saved poses. The [diagnosis](../runs/mobile-full-capture-density-math-20260921/analysis.json)
checks the resulting density change and rounding bounds; this was not a
reason to change the physical weights or relax the tolerance.

A source-bound factor reconstruction now uses that actual draw factor while
retaining independent matrix-based geometry and density calculations.
The [separate recovered audit](../runs/mobile-full-capture-audit-recovery-20260921/status.json)
passes all 65,536 saved draws, with maximum log-density disagreement
5.18×10⁻¹¹, below the unchanged 2×10⁻⁸ tolerance. Original outputs, failed
audit and controller status remain untouched. The comparison explicitly
validates the recovered audit and its input hashes. No physical rows were
regenerated for recovery.

The corrected-observer suite passes 34 tests; the campaign controller has
10 tests and the stateless classifier has nine. The separate atomic-wall
normalizer extension passes 11 Rust controls and independent Python checks,
including an analytic sphere reference whose depletion lens crosses the
wall. That implementation enables full-vessel integration, but no full-wall
protein normalizer has yet been measured. D170 still excludes known
wall-valid native sites; see [the domain review](mobile-domain-coverage.md).
