# Fixed-pose overlap at the near-native shoulder

The largest selected shoulder importance weight is supported by a real,
hard-valid high-overlap pose. Fresh counts estimate its overlap volume as
**1082.43 ± 1.94 Å³**, corresponding to `z*C=37.885 ± 0.068` at
depletant radius 1.5 Å and activity 0.035 Å⁻³. Its original two-cloud
Boltzmann estimate was approximately 1.49 times the new 128-cloud mean.
That fluctuation contributes to the original large weight, but does not
explain it away.

This is a conditional diagnostic of selected poses, not a new estimate of
basin mass or an equilibrium contact comparison. The original importance
weights in the normalizer campaign have **not** been modified or selectively
replaced.

## Selected poses and independent geometry checks

The three requested shoulder poses came from
[/home/xvg/tetramer-mc/runs/basin-normalizer-local-nativewide-8192-l64](/home/xvg/tetramer-mc/runs/basin-normalizer-local-nativewide-8192-l64):
population `scale-1-r01`, draw 3650; `scale-1-r00`, draw 3609; and
`scale-1-r02`, draw 4972. They were selected after observing large importance
weights, so their noise and geometry are not representative of all shoulder
poses. For context, the audit also uses a native pose near the registration
boundary and a far pose from the previous SMC archive. These two references
are individual selected configurations, neither basin means nor optima.

All cases use the same 4004-atom sphere union, fixed site-0 neighbor, bath,
and original native metric `q=max(member displacement/2 Å, angle/15°)`.
The fresh diagnostic uses 128 independent clouds at **lambda/z=64**, hence
`lambda=2.24 Å⁻³`, and 1,048,576 independent common world-coordinate points
per pose. The envelope options remain 2047 maximum cells, depth 14, and
minimum width 0.5 Å. No fully interior cells were certified in these cases,
so the deterministic lower volume is zero.

| Pose | Original q | Maximum member displacement / Å | Proper angular error | C / Å³, ± SE | zC, ± SE |
|---|---:|---:|---:|---:|---:|
| Shoulder r01 / 3650 | 1.4587 | 2.9175 | 4.990° | 1082.43 ± 1.94 | 37.885 ± 0.068 |
| Shoulder r00 / 3609 | 1.4815 | 2.9631 | 5.929° | 984.23 ± 1.85 | 34.448 ± 0.065 |
| Shoulder r02 / 4972 | 1.3837 | 2.7673 | 5.009° | 952.85 ± 1.82 | 33.350 ± 0.064 |
| Native `native-07`, particle 263 | 0.9994 | 1.9988 | 3.081° | 1041.13 ± 1.91 | 36.440 ± 0.067 |
| Far `other_adsorbed-01`, particle 463 | 10.2241 | 20.4481 | 39.110° | 609.73 ± 1.46 | 21.341 ± 0.051 |

Thus the highest-overlap shoulder example has approximately 1.45 kBT more
depletion log weight than the selected native-boundary example. The other
two shoulder examples have less overlap than that native reference. This
does not establish a maximum attainable native overlap or a shoulder-versus-
native free-energy difference: the available pose volumes still matter.

The archived `HardUnion` and current `SphereTree` returned identical exclusion
membership on all **5,242,880 common world points**, corresponding to
10,485,760 individual moving/fixed-union comparisons. Their hard-validity
checks agreed for all five poses. Independent common-point estimates and
the current Poisson-envelope estimates differed by at most 2.11 combined
standard errors. This is an observed paired predicate agreement, not a
formal proof of equality at every possible floating-point coordinate.

An additional BVH-independent calculation checked all 16,032,016 interbody
atomic pairs per pose, using transformed atom centers and periodic minimum
images. Minimum core-surface separations were:

| Pose | Minimum atomic gap / Å |
|---|---:|
| Shoulder r01 / 3650 | 0.0324344 |
| Shoulder r00 / 3609 | 0.0221916 |
| Shoulder r02 / 4972 | 0.0724807 |
| Native boundary reference | 0.000406688 |
| Far reference | 0.0609020 |

All margins are positive. This supplies a direct atomic hard-validity check
under ordinary FP64 conventions, separate from both tree traversals; it is
not outward-rounded interval arithmetic.

## Original two-cloud fluctuations

For overlap `C`, deterministic inner volume `L`, and Poisson intensity
`lambda`, each positive estimator is

\[
 K\sim\operatorname{Pois}(\lambda(C-L)),\qquad
 \widehat W=e^{zL}(1+z/\lambda)^K.
\]

The new overlap estimate uses `C_hat=L+sum(K)/(128*lambda)`, with Poisson
standard error `sqrt(sum(K))/(128*lambda)`. The independently estimated
Boltzmann weight is the **linear mean** of the 128 positive weights,
evaluated by log-mean-exp. It is not the exponential of a noisy volume
estimate. Quoted relative errors below use those independent cloud weights.

| Shoulder draw | Original counts | Original log mean W | Fresh log mean W | Fresh relative SE | Original / fresh mean W |
|---|---|---:|---:|---:|---:|
| r01 / 3650 | 2504, 2375 | 38.2563 | 37.8583 | 7.90% | 1.489 |
| r00 / 3609 | 2268, 2268 | 35.1635 | 34.4591 | 7.82% | 2.023 |
| r02 / 4972 | 2203, 2250 | 34.5850 | 33.3134 | 7.07% | 3.567 |

Relative to the fresh overlap estimates, the original *combined count* in
each pair is +0.42, +1.89, and +2.80 standard deviations respectively,
including uncertainty in the fresh count mean. These are fixed-pose noise
diagnostics, **not post-selection significance tests**. Selecting unusually
large importance weights preferentially selects favorable cloud fluctuations
as well as favorable geometry and small proposal density.

Keeping the original full proposal density for the first pose,
`log g=10.8055847`, the fresh conditional estimate gives
`log(mean W/g)=27.0528`, compared with the original `27.4507`. The large
importance contribution therefore persists after a much more precise
fixed-pose depletion calculation. Its contribution to the basin integral
still requires adequate proposal coverage and independent integration;
this calculation does not supply that missing pose-volume factor.

## Artifacts and reproduction

The complete
[analysis](/home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/analysis.json),
[selected input rows](/home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/selected-poses.json),
[direct atomic gaps](/home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/direct-atomic-gaps.json),
and [input/source/binary hashes](/home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/provenance.json)
are retained. The copied probe executable has SHA-256
`5042bfb1ddd79fba7a812d63a30a7096c5dcad9c0e5199cb2d53555d88311daf`.
It compiles the archived geometry beside the current tree; its source is
[old_new_geometry_probe.rs](/home/xvg/tetramer-mc/tools/old_new_geometry_probe.rs).

The recorded invocation was:

```bash
/home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/probe \
  /home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/input.json \
  /home/xvg/tetramer-mc/runs/shoulder-contact-diagnostic-20260920/results.json
```

All new counts use fresh recorded seeds. Selection and repeated evaluation
occur only in this diagnostic dataset; the existing importance-sampling
normalizer records remain unchanged.
