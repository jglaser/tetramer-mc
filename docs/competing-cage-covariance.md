Two pilot-selected competing contact cages are represented by one Gaussian
proposal each. The fit uses all production residence states from local streams
0 and 1. Streams 2 and 3 are held out: they evaluate the frozen fit and cannot
change its mean, covariance, bandwidth, or regularization. This is a conditional
proposal construction, not a measurement of equilibrium basin widths.

The [chart construction and comparison figure](context-competing-cage.md)
describe how these fits extend the existing proposal without changing the
original contact reference.

Three poses or references have distinct roles:

| Input | Role |
|---|---|
| `cfg.initial_pose` | Physical starting configuration of that training chain. |
| `source_chart.chart_center` | Fixed origin of the Gaussian's local pose coordinates; here it is the selected cage seed. |
| Original `source_state` and patch reference | Immutable physical provenance, anchor and contact-region definitions shared with earlier calculations. |

The chart center is a saved world pose in the sphere-center frame. With anchor
rotation \(R_a\), center \((t_c,R_c)\), and pose \((t,R)\), the six coordinates are
\(x_t=R_a^T(t-t_c)\) and \(x_r=\ell q_v/q_w\), where \(q\) represents
\((R_a^TR)(R_a^TR_c)^T\). No display shift or native registration adjustment is
applied. A Cayley seam is retained and makes the affected coordinate summary
undefined; it is never removed from the trajectory.

Each cage has four independent streams of 2,304 cycles, with five local attempts
per cycle. The first 256 cycles are archived and summarized separately. Every
retained production state, including repeated rejected states, contributes to
the residence-weighted mean and covariance. Thus each cage has 20,480 training
states and 20,480 held-out states. These counts are not independent sample sizes.

The fixed fit uses the covariance with denominator \(N\). Set
\(S=\operatorname{diag}(0.1\,\AA,0.1\,\AA,0.1\,\AA,
\ell\tan(0.25^\circ),\ell\tan(0.25^\circ),\ell\tan(0.25^\circ))\),
\(\Sigma=S^{-1}CS^{-1}\), and
\(\epsilon=\max(10^{-10},10^{-6}\operatorname{tr}\Sigma/6)\).
The exported covariance is \(S(\Sigma+\epsilon I)S\), with the observed mean.
There is no bandwidth search or additional within-cage mixture component.

The report keeps the unregularized rank, eigenvalues, fraction of variance
introduced by the ridge, unique accepted and retained states, movement ranges,
and per-stream coordinate autocorrelation. Constant observables have undefined
ESS, not ESS equal to the trajectory length. A ridge-only or rank-deficient fit
remains explicitly labeled even though the resulting Gaussian is normalized.

Held-out likelihood uses the Gaussian coordinate density divided once by the
normalized-Haar Cayley Jacobian
\(J=\ell^{-3}\pi^{-2}(1+\|x_r/\ell\|^2)^{-2}\).
Squared Mahalanobis distances and nominal Gaussian coverage are approximation
diagnostics. The held-out streams remain serially correlated and start from
the same selected cage; these scores do not establish equilibrium, independent
coverage of competing environments, or a physical contact free-energy gap.

`tools/prepare_competing_cage_covariance.py` checks all eight completed training
receipts and the controller's 13 drained process groups, then snapshots the
reducer and its imports. The fixed plan runs ten synthetic controls and the
saved-state reduction, with no atomic geometry, new poses, or cloud draws.
The fit specification is frozen at
`results/competing-cage-covariance-preparation-20261005-v2/fit-specification.json`.
The original unexecuted specification remains archived; v2 corrects the nested
schema used to authenticate the immutable patch reference and tests that case.

The separate contact observer retains full patch-fingerprint residence as well
as neighbor identities. Its inventory must agree before interpreting the fit
and contacts together. Neighbor agreement alone cannot identify the same cage.
The physical scope remains one mobile tetramer in a fixed neighborhood from
the historical 500 μM system, at depletant radius 1.5 Å and activity
0.035 Å⁻³; this is not an all-mobile assembly result.

The first completed fit retained all 92,160 states and passed ten synthetic
controls. Both empirical covariances have rank six; the ridge contributes
approximately one millionth of their scaled variance trace. The fits therefore
contain observed movement, but that movement remains strongly correlated.

| Diagnostic | Cage 0: partial source contact | Cage 1: other neighbor environment |
|---|---:|---:|
| Unique training coordinate states | 268 | 997 |
| Per-stream production vector ESS | 4.53–9.48 | 7.85–16.25 |
| Mean training squared Mahalanobis distance | 6.00 | 6.00 |
| Mean held-out squared Mahalanobis distance | 25.75 | 9.19 |
| Held-out coverage inside nominal 95% Gaussian ellipsoid | 39.32% | 86.01% |
| Training minus held-out mean log density | 9.88 nats | 1.59 nats |

Cage 1 generalizes better within this local control. Cage 0's held-out mismatch
is substantial and remains visible in the frozen proposal; no held-out fitting
or component addition follows from this result. The training value near six is
largely a consequence of estimating a six-dimensional covariance on those same
training samples, not a separate validation. All production neighbor sets
remain respectively `{16,217}` and `{16,56}`. Full patch observations are a
separate completion gate for interpreting contact identity.

That [contact audit has now passed](context-competing-cage.md#contact-audit):
fingerprints change within each neighbor environment, but no chain changes
its neighbor set. Rare minority fingerprints make some apparent ESS values
unstable. These findings do not supply equilibrium basin weights.

The report and two normalized guide assets are in
`results/competing-cage-covariance-20261005/result/`. Report SHA-256:
`f28437dd572a9b98996ce25d6967dfd99d96fdaee3e135910d6a920578561fc1`.
The saved-data reduction used 3.88 CPU seconds and generated no poses, atom
queries, or depletant clouds.
