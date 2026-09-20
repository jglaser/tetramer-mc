# Complete geometric covers for an original-q window

This note reviews the geometric proposal needed to integrate the original
metric over a larger, fixed window such as `1 < q <= 2`. It changes the
integration region, not the metric or the physical interactions. In the AB
calculation both fixed proteins still enter the hard and union-depletion
tests. Selecting A as a Gaussian coordinate anchor does not remove B.

The statements below are exact real-arithmetic identities. The implemented
spectral bound has a conservative floating-point guard, not a formal
interval-arithmetic proof. The existing abstract Lean importance theorem
does not prove the geometry or its floating-point implementation.

## Coverage and normalized sampling

Let the reference pose be `(t0,R0)`, the rigid member positions be `r_j`, and
their arithmetic centroid and covariance be

`c = mean(r_j)`, `M = mean[(r_j-c)(r_j-c)^T]`.

Write `delta` for the original member-error scale and `alpha` for the
original angular scale in radians. There is one native reference in this
implementation. For several native references one would need a normalized
union-cover mixture with the full summed density; a cover of just one
reference is not complete for the minimum-over-references metric.

For an upper threshold `b>0`, every pose with original `q<=b` satisfies
`|e_j|<=a=b*delta` for every member and orientation angle
`theta<=min(b*alpha,pi)`. Introduce compensated displacement

`w = t-t0 + (R-R0)c`.

If `R0^T R` is a rotation of angle `theta` about unit axis `u`, then

```
mean |e_j|^2
  = |w|^2 + tr[(R-R0) M (R-R0)^T]
  = |w|^2 + 4 sin(theta/2)^2 [tr M - u^T M u]
  <= a^2.
```

Let `L` be any nonnegative lower bound on `tr M-lambda_max(M)`. Consequently
every such pose lies in the product cover

```
|w| <= a,
theta <= h_b,
h_b = min(pi, b*alpha,
          2 asin(min(1, a/(2 sqrt(L)))))  when L>0,
h_b = min(pi, b*alpha)                   when L=0.
```

This is a necessary-condition cover. It usually contains poses with
`q>b`; the original metric must still reject them. The cover radius and
cap are computed from `b`, but the stored original metric scales must not
be multiplied by `b` when labeling samples.

The SO(3) angle CDF under normalized Haar measure is

`F(theta) = (theta-sin(theta))/pi`, for `0<=theta<=pi`.

The cover volume in translation volume times normalized Haar measure is

`V_b = (4*pi*a^3/3) F(h_b) = (4*a^3/3)(h_b-sin(h_b))`.

Draw an isotropic axis, an angle satisfying
`theta-sin(theta) = U*(h_b-sin(h_b))`, an isotropic direction `v`, and
`w=a*U2^(1/3)*v`. Set

`R=R0*Rot(axis,theta)`, `t=t0-(R-R0)c+w`.

This law is uniform on the cover. At each orientation, the change from
`w` to `t` is a translation with unit Jacobian. The full change of variables
is triangular in translation and orientation, and Haar measure is invariant
under multiplication by `R0`. In particular, a nonzero centroid does not
create an additional Jacobian. Omitting its compensating translation would
move the proposal support and break the coverage proof.

When `b*alpha>pi`, the angular restriction saturates at pi. This is an
ordinary, valid case: it means the original angular part of the metric no
longer constrains that upper-q region. An angle above pi must not be passed
to the cap inverse CDF; nor should the widened target be rejected solely
because its nominal angular tolerance exceeds pi.

## Window estimator and endpoints

For a lower threshold `l` and upper threshold `b`, use the requested original
window, for example `l<q<=b`, as a separate indicator. Boundaries have zero
Lebesgue-Haar measure, but exact endpoint conventions should remain explicit
in the schema and independent auditor. The legacy native path should retain
its original endpoint convention and random stream.

If `g` is the complete normalized cover/guide mixture and `Wbar` is the
independent positive Poisson estimator, each unconditional draw contributes

`Y = 1_capture * 1_hard * 1_window * Wbar / g`.

All invalid draws remain zero in the original fixed count `N`. With
`E[Wbar|pose]=exp[z*C(pose)]`, where `C` is overlap with the **union** of all
physical neighbors' exclusion regions, `mean(Y)` estimates the desired
window integral without changing its physical measure. Proposal family
labels do not restrict which density terms enter `g`. A positive outer-cover
weight gives complete window support even when a fitted guide misses a
basin. This support statement is not a finite-budget convergence claim.

Validate finite positive `b`, finite nonnegative `l<b`, finite scaled radius,
and finite positive volume. The inverse-angle CDF should use the existing
small-angle series for `theta-sin(theta)`. Normalized-pose and zero-measure
cap endpoints require the same care as the original cover. Guarding the
covariance subtraction makes the derived angular cap wider, never narrower,
when the spectral lower bound is uncertain; falling back to the nominal
cap is safe. Extremely large coordinates or underflowed volumes should fail
explicitly rather than silently alter the proposal law.

## Why subtracting the smaller cover can lose shoulder mass

The covers `C_l` and `C_b` share `(c,t0,R0)` and are nested. Their difference
can be sampled exactly. It is the disjoint union of

- A translation shell: `a_l<|w|<=a_b`, with `theta<=h_b`, of volume
  `(4*pi/3)*(a_b^3-a_l^3)*F(h_b)`.
- An angular shell: `|w|<=a_l`, with `h_l<theta<=h_b`, of volume
  `(4*pi/3)*a_l^3*(F(h_b)-F(h_l))`.

Choose these two pieces in proportion to those volumes. The radial inverse
CDF is `(a_l^3+U*(a_b^3-a_l^3))^(1/3)`; the angular shell uses a uniform
CDF coordinate between `F(h_l)` and `F(h_b)`.

For a general rigid member set, however, `C_l` is a **superset** of
`{q<=l}`. It can contain important poses with `q>l`. Sampling only
`C_b\C_l` would drop those poses and bias a shoulder integral. A normalized
mixture with a positive inner-cover branch is valid:

`g = p*1_C_l/V_l + (1-p)*1_(C_b\C_l)/(V_b-V_l)`, with `0<p<1`.

For a concrete counterexample, take members at `(+-1,0,0)`, reference identity,
`delta=1`, `alpha=90 degrees`, translation `(.9,0,0)`, and a 60-degree rotation
about z. The covariance has `L=0`, so this pose is strictly inside `C_1`.
Its member errors have lengths `sqrt(.91)` and `sqrt(2.71)`, giving
`q=sqrt(2.71)` in the shoulder. Continuity gives a nonzero-volume neighborhood
with the same property; this is not a boundary-only objection.

This is a possible later variance allocation, not necessary for the initial
complete-window implementation. If there is only one metric member, the
cover is the exact original-q ball (before capture or hard constraints), so
the cover difference alone is then a complete window proposal. More
generally one may safely remove only a proven **inscribed** part of
`{q<=l}`. For example, with `rho=max_j|r_j-c|`, the condition
`|w|+2*rho*sin(theta/2)<=l*delta` and `theta<=min(l*alpha,pi)` is sufficient.
It does not justify removing the larger circumscribed cover.

## Independent reference controls

An unobstructed singleton-member metric has an exact window volume
`V_b-V_l`. Use a nonzero member centroid, a rotated and translated reference,
and a capture sphere large enough to contain all compensated translations.
This checks the coupled translation/orientation construction without a
protein geometry oracle. Test both an unsaturated angular cap and a case
with `b*alpha>pi`.

For a physical depletion reference, let the metric member be at body origin
and put the reference center at `c0`. The mobile shape is a sphere of core
radius `R`, exclusion radius `S=R+rd`. Put physical neighbor 0 far away and
physical neighbor 1 at `c0`; use neighbor 0 as the guide anchor. A capture
sphere centered at `c0` has radius `Rc`. All physical hard and depletion
effects must then come from neighbor 1.

Define `H_s=F(min(s*alpha,pi))`. The exact window integral is the one-dimensional
reference

```
Q = 4*pi * integral from 2*R to min(b*delta,Rc):
        r^2 exp[z*C_sphere(r)] [H_b - 1_(r<=l*delta)*H_l] dr,

C_sphere(r) = pi*(4*S+r)*(2*S-r)^2/12   for r<2*S,
              0                       otherwise.
```

Split numerical quadrature at `l*delta` and `2*S`, where the formula changes.
For example, `delta=.8`, `alpha=110 degrees`, `l=1`, `b=2`, `R=.2`,
`rd=.35`, and `Rc=1.4` exercise the pi cap, hard rejection, capture rejection,
and positive depletion simultaneously. A positive activity and independent
Poisson clouds check the physical weight; setting activity to zero gives a
separate hard-volume reference. At every radius the angular factor follows
directly from the original max metric, independently of the geometric-cover
implementation.

These controls can validate implementation and normalization while leaving
the AB protein window weights, unresolved remaining-space mass, and eventual
assembly stability as separate empirical questions.
