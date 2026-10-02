# Completed larger contact-weight comparison

All 524,288 new attempts, eight independent geometry/density audits and the
once-only contact classification are complete. Together with the 131,072-draw
pilot and 131,072-draw sensitivity controls, this finishes the fixed
**786,432-draw campaign**. Stages remain separate; no populations were pooled.

All four aggregate Qz comparisons between the larger baseline and conditioned
arms pass. All sixteen larger-versus-pilot regional comparisons also pass:
two arms, four regions, both physical Qz and hard-only Q0. The conditional
native/competing contrast is stable:

| Larger arm | β(F_native − F_competing) | Population 95% half-width |
|---|---:|---:|
| Baseline | −18.6815 | 0.1061 |
| Hard-free conditioned | −18.7204 | 0.0494 |

These are paired-population delta intervals using Student-t with three degrees
of freedom. Their precision does not bound unseen weight. The calculation
concerns the fixed two-neighbor R4 integral, not an association free energy in
the whole vessel or finite-system assembly stability.

The conditioned proposal passes aggregate quality checks for every decision
region. Its observed importance ESS per sampler CPU improves by 1.68–1.88×
relative to the baseline. This measures integration efficiency, not trajectory
mixing or assembly speed.

| Region | Conditioned importance ESS | Largest draw fraction | ESS/CPU ratio |
|---|---:|---:|---:|
| Total native | 5,559.5 | 0.4973% | 1.779 |
| Competing contact | 3,906.4 | 0.4854% | 1.684 |
| Old-R5 native intersection | 3,591.8 | 0.2528% | 1.675 |
| Remaining native | 2,177.5 | 1.0545% | 1.881 |

The baseline's remaining-native region fails the largest-contribution check:
one draw supplies 2.0529%, exceeding the frozen 2% threshold. That failure is
retained even though the excess is small.

![Larger-population comparison](../results/hard-free-line-population-size-review-20261002/hard-free-line-physical.png)

## Convergence failures retained

Two material competing-contact strata disagree between proposals: pose-space
orthants 22 and 62, with log baseline/conditioned ratios 0.4071 and 1.6323.
Both pass the statistical-SE check but fail the absolute 0.2 criterion.
These orthants are sign bins of the six-dimensional latent pose chart, not
bins of orientation alone. Their observed weights remain noisy.

Eleven material strata fail the predeclared population-size comparison:

| Arm | Region | Bin | Log larger/pilot | Failed criterion |
|---|---|---|---:|---|
| Baseline | Competing | radial 2 | +0.3491 | Absolute |
| Baseline | Competing | orthant 22 | +0.3745 | Absolute |
| Baseline | Competing | orthant 26 | +0.2497 | Absolute |
| Baseline | Competing | orthant 62 | +1.4743 | Absolute |
| Baseline | Competing | orthant 63 | −0.3401 | Absolute |
| Baseline | Old-R5 native | orthant 55 | −0.2298 | Absolute |
| Baseline | Remaining native | orthant 19 | +0.2106 | Absolute |
| Conditioned | Total native | orthant 55 | +0.08893 | Three combined linear SE |
| Conditioned | Competing | orthant 22 | −0.3748 | Absolute |
| Conditioned | Competing | orthant 39 | −1.8780 | Absolute |
| Conditioned | Remaining native | orthant 55 | +0.1707 | Three combined linear SE |

All original strata and four independent population masses are retained in
`/vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001/population-size-comparison.json`.
The earlier probability/intensity failures also remain part of the
[completed campaign](hard-free-line-sensitivity-results.md). The complete fixed
campaign therefore **does not pass its convergence gate**. No full-vessel
calculation was launched on the strength of these results.

## Next independent check

The larger conditioned log native-remainder mass is 60.52958, close to the
pilot's 60.53275. Increasing the population fourfold does not explain the
historical broad-SMC remainder estimate of approximately 0.216× the pilot.
The discrepancy still calls for a separate check of SMC coverage and its
annealing path.

The [new protein SMC bridge control](hard-free-protein-smc-control.md) retains
the old broad-mutation allocation and changes the initial/intermediate density
using the validated normalized hard-free guide. It tests the specific
intermediate-population bottleneck hypothesis. Even agreement will not waive
the failed strata, establish full-vessel coverage or decide assembly.

## Provenance

Campaign: `/vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001`.
Protocol SHA-256:
`cbcf1400a38365f2fa24585110da389b64174fc7e3a563e26cf1618a161c588e`.
Completed classification summary:
`a05fc50b2d86bf32d1913388e1a05eff8397b734a6a05ce7932c5ac9bc07b39f`.
Population-size comparison:
`cf658c26171ec267712aee249b72eac301832688a3503b363c78dabe4e0ea49b`.
The [display report](../results/hard-free-line-population-size-review-20261002/report.md)
reuses completed aggregate evidence without repeating geometry, clouds or labels.
