# Independent contact weights with two prescribed crystal neighbors

The calculation compares native registration and competing arrangements of
one tetramer next to two fixed tetramers A and B. Both neighbors remain in
the hard-overlap test and in the **union** of depletant exclusion regions.
The bath is unchanged: depletant radius 1.5 Å, activity 0.035 Å^-3, and an
18 Å capture sphere. Rotations use normalized SO(3) Haar measure. This is a
conditional docking calculation; it does not include the cost of assembling
the two fixed neighbors.

For a pose x and a fixed region Ω, the target integral is

```
Q(Ω) = ∫Ω H(x; A,B) exp[z C(x; A,B)] d³t dHaar(R),
C(x; A,B) = |E(x) ∩ (E(A) ∪ E(B))|.
```

The original native metric is unchanged: the maximum of rigid-member
displacement divided by 2 Å and rotation error divided by 15°, minimized
over the supplied reference poses. Native means q≤1; distant means q≥5.
Proposal components and their weights do not define these physical regions.

## A proposal frame is independent of the physical environment

`basin-normalizer --proposal-anchor-index 0` uses A to express all learned
relative poses. It does not delete B. This avoids spending half of the
Gaussian proposals in the wrong reference frame: the geometry screen found
that all 512 B-anchored draws missed the capture sphere. The default, which
averages the proposal over all fixed anchors, is preserved for replay.

The selected-anchor proposal is a normalized mixture of an untruncated
Gaussian pose atlas and a uniform cube/rotation proposal. For every draw,
including capture and hard-invalid draws, the denominator is the full
mixture density. Independent cloud weights satisfy

```
E[W | x] = exp[z C(x)],
Y = H(x) 1Ω(x) mean(W₁,W₂) / g(x).
```

There is no retry and no normalization by the number of valid poses.
Selected-anchor runs stop on an unrepresentable numerical proposal, instead
of silently assigning a zero to an event whose mathematical density is
nonzero. Older default runs keep their existing behavior for compatibility.
Independent Python analysis reconstructs the full Gaussian/cube density,
capture and registration labels on every actual selected-anchor pose, and
rejects numerical nulls and changed archived inputs.

## A separate uniform reference within a frozen region

`latent-region-normalizer` now supports a complete physical-neighbor list,
a radial shell in a frozen Gaussian chart, and lower/upper q cutoffs. A
single `fixed_neighbor` defines the chart frame; `physical_fixed_neighbors`
defines the environment and must equal the configuration's full list.

Write the chart coordinates as `v = μ + L u`, where the last three
coordinates are `ell × Cayley(rotation residual)`. For `a≤|u|≤b`, draw an
isotropic six-dimensional direction and a radius with

```
r = [a⁶ + U(b⁶ − a⁶)]^(1/6),   U uniform on [0,1),
V = π³ (b⁶ − a⁶) / 6,
J(u) = det(L) / [ell³ π² (1 + |Cayley|²)²],
Y = V J(u) H(x(u)) 1Ω(x(u)) mean(W₁,W₂).
```

The implementation evaluates the shell difference with `log1p` and
`expm1` to preserve thin-shell precision. With a=0 it retains the old
draw order and radial expression exactly. The reference integrates only
its declared region. A Gaussian guide can independently estimate the same
region by masking its original full importance weights, with the original
unconditional draw count as denominator.

For the native tail checks, the chart is the **original** single Gaussian
(`86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667`).
It is not replaced by the more recent refined guide. The fixed shells are
4–5, 5–8 and 8–12, each intersected with q≤1 and the same physical support.
A Mahalanobis radius in six dimensions is not a one-dimensional sigma
interval.

## Frozen competitor and global proposal

A geometry-only screen found 30 distinct old SMC endpoint poses that remain
valid with both A and B and contact both. All descend from one SMC ancestor;
their covariance is a proposal construction, not an equilibrium estimate.
The resulting competitor chart defines a fixed radius-3 region intersected
with q≥5. A new global proposal combines the refined native Gaussian (50%),
the unchanged 117-component atlas (25%) and this competitor Gaussian (25%),
plus a 10% outer uniform proposal. These percentages allocate sampling
effort and are corrected by the complete proposal density.

The frozen files and preparation diagnostics are in
`runs/ab-competing-frozen-atlas-20260920`; native reference shells and
independent static geometry probes are in
`runs/native-ab-tail-reference-preparation-20260920`. The latter also
preserves the competitor region byte for byte.

## Validation and interpretation

The sphere controls independently integrate the hard geometry and analytic
spherical depletion lens. They test broad rotations, a nonzero capture
center, unequal translational and rotational chart scales, shell radial
moments, both q cutoffs, and an active second neighbor that is not the
proposal anchor. Tests also reject mismatched neighbor lists, inverted
regions and numerical-null importance draws. Python controls independently
differentiate the six-dimensional pose map to check its Haar Jacobian.
The legacy latent normalizer reproduced 1,024 sample rows byte for byte.

All protein estimates retain independent populations and invalid zeros.
Observed ESS and standard errors measure the sampled contributions; they
cannot certify unseen modes. Agreement on the finite competitor region or
native shells tests the proposal correction on those regions. It does not
establish global coverage, equilibrium assembly, or the cost of creating
the prescribed neighbors.

## Completed fixed-budget comparisons

The frozen experiment protocol is
`runs/ab-cooperative-weight-pilot-20260920/protocol.json`. Each initial
calculation used four independent populations of 8,192 unconditional draws,
two Poisson clouds per valid pose, and auxiliary intensity/activity ratio
64. Subsequent fixed-budget confirmation and width controls have separate
protocols in the same directory; no original observations were removed.

The refined native guide gives log Q(native)=35.77224, observed relative
standard error 2.20%, and importance ESS 2,036.7 in its separate 131,072-draw
validation. The nominal global proposal independently gives 35.75204,
6.03% and ESS 272.3. On the **same finite competing R3 region**, the global
estimate is 13.95199 (25.6% observed error) and the uniform reference is
14.43700 (14.7%). Their difference is 1.78 combined observed standard
errors. This is a regional correction check, not proof of global coverage.

The native tails expose the difference between an apparently stable total
and adequate coverage. For the fixed original-chart 8–12 shell, the wider
guide gives log Q=30.27847 and uniform integration gives 30.28155, while
the narrow guide gives 27.06243. In the 5–8 shell, a smaller uniform pilot
gave 32.65693; a fresh eight-population, 262,144-draw confirmation gives
33.19840, compatible with both guided estimates near 33.08. Its observed
error remains 15.9%, and a few large contributions prevent a precise
reference claim. See the detailed tail report for full population and
cloud diagnostics.

More significantly, widening the **unchanged** global proposal discovers
previously unsampled competing arrangements:

| Gaussian standard-deviation multiplier | Draws | Native log Q (ESS) | Other log Q (ESS) | Valid shoulder poses, 1<q<2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 35.752 (272.3) | 15.295 (12.5) | 0 |
| 2 | 65,536 | 35.665 (62.9) | 16.556 (18.9) | 0 |
| 4 | 65,536 | 36.483 (2.0) | 26.258 (2.5) | 66 |

The width-four competing estimate is almost entirely shoulder weight, and
its observed relative error is 62.6%. Its native estimate is also poorly
concentrated. These values cannot be read as a converged free-energy gap.
They establish a concrete coverage failure in the narrower proposal: a
new sampled region changes the competing estimate by approximately eleven
natural-log units. The physical shape, two neighbors, bath, capture and
registration partition did not change.

![Independent proposal-width controls reveal previously missed shoulder poses](../runs/ab-cooperative-weight-pilot-20260920/proposal-width-coverage.png)

Every one of the 66 shoulder poses passed an independent atomic hard/contact
check against both neighbors. Broadening also loses integration accuracy:
the width-four proposal substantially underestimates the already measured
finite R3 competitor region. Discovery and accurate integration therefore
need separate controls; one broad Gaussian scale does not solve both.

The next controlled calculation should target the shoulder directly,
using a complete geometric cover for q≤2 with the **original** q retained
for the 1<q<2 mask, plus a frozen guide from the newly discovered poses.
The existing cover builder can derive its bounds from twice the metric
tolerances; the physical metric and regional labels must stay unchanged.
An independent finite-region reference can then check that guide before
using its proposal in trajectories. The current data do not establish
native assembly or rule it out.
