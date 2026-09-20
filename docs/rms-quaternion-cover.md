# A complete quaternion RMS proposal for a fixed original-q window

The exact rigid-member RMS error gives a substantially smaller complete
proposal than the product translation-ball/angular-cap cover. It couples
translation and orientation through one six-dimensional ball and retains a
tractable density. This note derives the construction for independent
contact-region integration; it is not a claim about adaptive proposal
learning, equilibrium mixing, or assembly.

All formulas below use exact real arithmetic and normalized SO(3) Haar
measure. Numerical eigenvalues and volume bounds reported for the AB metric
are floating-point evaluations of those formulas, not outward-rounded
interval certificates.

## Geometry and inverse map

For body-frame rigid-member positions `r_j`, define

```
c = mean(r_j),
M = mean[(r_j-c)(r_j-c)^T],
A = tr(M) I - M.
```

Let `(t0,R0)` be the native reference. The relative rotation convention is
`Q=R0^T R`. Choose its quaternion representative `(q0,v)` on the hemisphere
`q0>=0`, so `q0=sqrt(1-|v|^2)` and `v=sin(theta/2)*u`. The vector `v` is
expressed in the reference **body** coordinates, matching `A`. A quaternion
for `R R0^T` would instead express its vector in laboratory coordinates and
would require rotating `A` as well.

Define the compensated displacement

`w=t-t0+(R-R0)c`.

The exact mean squared member displacement is

```
mean_j |t+R*r_j-t0-R0*r_j|^2
    = |w|^2 + 4 v^T A v.
```

If `A` is positive definite, set `eta=2*A^(1/2)*v`. The right-hand side is
then `|w|^2+|eta|^2`. Hence every pose satisfying original `q<=b` maps inside
the six-dimensional ball of radius `a=b*delta`, where `delta` is the
original member-error scale. The angular part of the metric may restrict
this set further, but cannot invalidate its coverage.

The draw is a uniform vector `(w,eta)` from that ball. Decode

```
v = (1/2)*A^(-1/2)*eta,
q0 = sqrt(1-|v|^2),
R = R0 * Rot(q0,v),
t = t0 - (R-R0)c + w.
```

It is unnecessary to compute a symmetric matrix square root. If a Cholesky
factorization gives `A=L L^T`, use `eta=2 L^T v` and solve
`L^T v=eta/2`. Using `2 L v` with that lower-triangular convention would give
the wrong quadratic form.

Uniform six-ball draws use six independent standard normals for a uniform
direction and radius `a*U^(1/6)`. Do not draw translation and orientation
independently: their shared radius constraint supplies the compression.

## Jacobian and density

The unit three-sphere has area `2*pi^2`; one hemisphere parametrizes SO(3)
almost everywhere and has area `pi^2`. The hemisphere vector coordinates
therefore give normalized Haar measure

`dH(R) = d^3v / [pi^2 sqrt(1-|v|^2)]`.

Since `d^3eta=8*sqrt(det A)*d^3v`, and the compensated translation has a
unit, triangular Jacobian at fixed rotation,

```
d^3t dH(R) = J(w,eta) d^3w d^3eta,

J = 1 / [8*pi^2*sqrt(det A)*sqrt(1-|v|^2)].
```

There is no additional factor of two, `sin(theta)`, or centroid Jacobian.
Left multiplication by `R0` preserves Haar measure. The two quaternion
hemispheres represent the same rotations; retaining both would double
count the physical orientation space.

Write `V6=pi^3*a^6/6`. For every representable draw, the physical proposal
density is

`g(pose)=1/[V6*J(pose)]`

inside the mapped six-ball, and zero outside. To evaluate this density on
an arbitrary pose, form `R0^T R`, choose the quaternion sign with `q0>=0`,
recover `w` and `eta`, and test the same six-ball support. The plane `q0=0`
and exact ball boundaries have zero measure, but endpoint conventions must
be declared consistently in implementation and audit.

The desired target is still the original max-error/angle window with capture,
hard validity, and depletion from the union of **all AB neighbors**. Its
conditional importance contribution is

`Y=V6*J*1_window*1_capture*1_hard*Wbar`.

As before, `E[Wbar|pose]=exp[z*C(pose)]`. The radius `a` controls proposal
support and does not rescale the q values used for region membership.

## Representability, null draws, and angular caps

If `a<2*sqrt(lambda_min(A))`, every latent draw has `|v|<1`, because

`|v| <= a/[2*sqrt(lambda_min(A))]`.

The map is then one-to-one almost everywhere onto the complete RMS-error
region and `g` integrates to one. A tighter nominal angular limit
`theta<=min(b*alpha,pi)` may be applied as a target mask, without changing
the unconditioned proposal density or redrawing a failed point.

For larger `a`, some latent points satisfy `|v|>=1` and do not encode a
physical rotation. They can remain explicit null outcomes with zero
contribution. In this case `g` integrates to a mass `m<=1`. The **full**
proposal is normalized on the extended space consisting of physical poses
and a null atom of mass `1-m`. With the original unconditional count `N`,

```
E[Y]
 = integral_representable (1/V6)*(V6*J*f(pose)) d(w,eta)
 = integral_target f(pose) d^3t dH(R).
```

Thus no unknown representable-volume normalization is required. Refilling
null draws until `N` physical poses have appeared would change the law and
would require that missing normalization. When this component is mixed
with a guide, use its unconditional subdensity in the full summed physical
mixture density, and keep its null events in the count. Calling that
physical subdensity a normalized density would be inaccurate even though
the extended proposal and estimator are normalized correctly.

Correct expectation does not guarantee finite variance. The Jacobian
diverges at `|v|=1`. If the target reaches a positive-area part of this
boundary with a nonzero allowed translation volume, `J^2` produces a
logarithmic divergence in the latent second moment. A strict nominal cap
below pi, or the sufficient all-representable bound above, keeps the target
away from that singularity. A complete positive-weight product-cover
component also bounds the full importance denominator on a bounded target.
The small-angle AB application is comfortably inside the nonsingular case.

## Positive definiteness and numerical choices

The eigenvalues of `A` are `tr(M)-lambda_i(M)`. It is positive definite
exactly when the member positions have at least two nonzero centered
covariance directions. A planar but noncollinear set is allowed. A
singleton or collinear set makes `A` singular because rotation about its
unconstrained direction can have zero member displacement.

For a singular or numerically unresolved `A`, the simple safe fallback is
the existing complete product cover. Adding an arbitrary positive diagonal
ridge is not a coverage-preserving repair: it penalizes rotations that the
actual RMS error does not constrain and can exclude valid target poses.
A positive-definite matrix `A_tilde<=A` in the Loewner order is a valid
conservative replacement, with its own determinant and inverse used
consistently. Such a replacement is impossible if `A` is truly singular.

A production implementation should archive the matrix, factor, radius,
relative-rotation convention, determinant, and representability rule. Its
guard must control both the covariance factorization and compensated
centroid arithmetic. A conservative lower matrix with a guarded radius is
one possible approach; a nominally computed positive eigenvalue or an
arbitrary covariance floor is not an outward coverage certificate. The
existing product-cover mixture can be retained as an independent support
control while numerical coverage is validated. None of these floating-point
choices should be confused with the exact real-arithmetic derivation.

## Size for the actual AB metric

Using the unchanged metric in
`runs/ab-shoulder-guide-preparation-20260920/config.json` gives eigenvalues
of `A` approximately

`[161.3829812143, 399.8656117739, 561.2485929881] A^2`,

with `sqrt(det A)=6018.1572023 A^3`. At `b=2`, `a=4 A`, so the largest
possible quaternion-vector norm is `0.157435` and the largest angle is
`18.1161 degrees`. This is below the original upper-window nominal cap of
30 degrees. There are no representability nulls, and the Jacobian changes
by at most 1.263% over the whole six-ball.

If `J0=1/[8*pi^2*sqrt(det A)]` and
`vmax=a/[2*sqrt(lambda_min(A))]`, then the exact physical image volume obeys

`V6*J0 <= V_RMS <= V6*J0/sqrt(1-vmax^2)`.

For `b=2`, these evaluated bounds are `0.0445456` to `0.0451082 A^3`,
compared with the current product cover's `0.619534 A^3`. Their ratio
implies at least **13.73 times** smaller covering volume in exact arithmetic.
At `b=1`, the corresponding ratio is at least **13.77**. The near-constant
Jacobian also makes the latent-uniform proposal nearly uniform in physical
measure in these cases.

These are geometric comparisons before the original max-member-error,
hard, and capture indicators. They suggest a substantially cheaper complete
coverage control, but do not establish a gain in depletion-weighted ESS,
independent contact samples, or assembly success. Those remain measured
comparisons against the validated product cover and frozen guides.

## Minimal implementation with the existing Cayley chart integrator

The same idea can already be approximated conservatively using the existing
uniform-latent-ball sampler. Let `h<pi` be the complete product cover's
angular cap for `q<=b`, and let `s` be the right relative Cayley vector, so
`v=s/sqrt(1+|s|^2)`. On the target,

```
mean |e_j|^2 = |w|^2 + 4 s^T A s/(1+|s|^2),
1/(1+|s|^2) >= cos(h/2)^2 = k.
```

Consequently `|w|^2+4*k*s^T A s<=a^2` is another complete outer ellipsoid.
For an exactly zero metric centroid, `w=t-t0`, so this is an ordinary
Gaussian-chart ellipsoid with no translation/rotation cross block. The
distribution to sample is **uniform in its latent six-ball**, not Gaussian.

The existing chart uses the *left* Cayley residual in physical neighbor A's
body frame. With neighbor orientation `Rf` and reference orientation `R0`,
its vector is `s_chart=Rf^T R0*s`. Thus

```
A_chart = Rf^T R0 A R0^T Rf,
Sigma_translation = I,
Sigma_rotation = ell^2 * A_chart^(-1)/(4*k),
Sigma_cross = 0.
```

The chart anchor is the unchanged native pose expressed relative to that
neighbor, the mean is zero, and the latent radius is `a`. The original
physical window and full AB neighbor list remain separate. A positive
definite guarded matrix below `A` widens this cover and uses the same
formula with its own inverse. As in the quaternion construction, the
floating-point guard is a numerical margin, not a formal interval proof.

Its Jacobian is already supplied by the latent integrator:

`J_Cayley = 1/[8*pi^2*k^(3/2)*sqrt(det A)*(1+|s_chart|^2)^2]`.

It has no finite-coordinate representability nulls or hemisphere-boundary
singularity. A uniform finite latent ball maps through the existing Cayley
chart to a finite physical region, and all importance weights still use
the exact pointwise Jacobian.

For the current AB `q<=2` cover the resulting physical-volume bounds are
approximately `0.0443838` to `0.0466828 A^3`, implying at least 13.27 times
less volume than the product cover. The largest possible angle is 18.1709
degrees and the Jacobian ratio is at most 1.0518. This retains nearly all
of the quaternion construction's geometric compression without requiring
a new pose-chart kernel.

A nonzero centroid cannot simply be ignored. One safe loose extension is
to increase the latent radius from `a` to
`a+2*sin(h/2)*|c|`: the six-dimensional triangle inequality bounds the
change from `w` to `t-t0` by `|(R-R0)c|`. Increasing it by `2*|c|` is also
safe but looser. The initial preparation instead requires exactly zero
centroid, making this restriction explicit and avoiding an unnecessary
enlargement.

The frozen preparation is
`runs/ab-shoulder-cayley-cover-preparation-20260920`. It records the matrix
construction, guard, body-frame transformation, model and region hashes,
and independent coordinate/density/RMS-identity checks. Its geometry-only
probes do not estimate depletion-weighted statistical mass. A preflight
audit error that double-counted the Cholesky determinant was corrected
before those checks passed or any atomic probes began; the failed
preparation was retained separately. The geometry model and existing
sampler were unaffected by that audit-only error.

The reusable constructor and geometry auditor are now
`tools/prepare_cayley_rms_cover.py`, with independent tests in
`tools/test_cayley_rms_cover.py`. The constructor recomputes the complete
angular cap for each requested upper q bound. It rejects singular moments,
nonzero centroids, multiple native references, and caps reaching pi instead
of silently substituting a restrictive covariance. Its default window is
open at both ends. The auditor can reconstruct the frozen model and all
probe coordinates without rerunning atomic checks or changing the inputs:

```
python tools/prepare_cayley_rms_cover.py \
  --audit-preparation runs/ab-shoulder-cayley-cover-preparation-20260920 \
  --out runs/cayley-cover-tool-validation-new
```

The first frozen protocol's short derivation string wrote `RMS>=...` next
to a squared-length expression. This is a notation error: the left side
is **RMS squared**, as the full derivation and executed matrices specify.
The frozen input is preserved. The promoted constructor corrects that
description; its numerical model reconstructs the frozen one exactly.

## A smaller complete window and an exact proposal-density ratio

The same constructor can target `1<q<1.1` by recomputing its angular bound
at `a=2.2 A`. It must not reuse an angular cap already derived for a
different upper threshold. The resulting proposal remains complete for
this narrower band; it no longer estimates the rest of the shoulder.

Let `b` denote the upper q threshold and `k_b=cos(h_b/2)^2`. For any pose
inside both geometric covers, the physical proposal density has factors
proportional to

`g_b(pose) proportional to b^(-6) * k_b^(3/2) * (1+|s_chart|^2)^2`.

The coordinate anchor, original metric, and guarded rotational matrix are
the same. Thus throughout the entire fixed band `B={1<q<1.1}`,

```
g_1.1 / g_2 = (2/1.1)^6 * (k_1.1/k_2)^(3/2)
            = 37.33243162495188.
```

This is a constant density increase, beyond the plain sixth-power factor
36.1263. With unchanged conditional cloud law and target indicator,
`E[Y_1.1^2]=E[Y_2^2]/37.33243162495188` for estimators of **this same band**.
The means remain equal. This identity concerns true moments and does not
turn observed importance moments into converged estimates. In particular,
an earlier finite run can have missed the high-weight poses that determine
the second moment.

The geometry-only narrow preparation is
`runs/ab-inner-shoulder-cayley-cover-preparation-20260920`: seed 99621010,
1,024 unconditional draws, with no clouds. It found zero jointly valid
band poses, consistent with the previous 32-hit estimate predicting about
1.17. All 22 previously observed source poses in the new band lie inside
the new radius-2.2 chart. Its cap is 11.0707 degrees, and evaluated physical
cover-volume bounds are `0.00123175` to `0.00125046 A^3`.

The subsequent independent four-population physical test retained
1,048,576 unconditional draws. The matching-band comparison is:

| Proposal | Band-valid draws | log Q for 1<q<1.1 | Physical importance ESS | Row / population RSE | Total CPU s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upper bound 2, evaluate only the matching band | 32 | 21.461721 | 1.09 | 95.7% / 94.3% | 145.04 |
| Upper bound 1.1, same band | 1,320 | 26.094252 | 1.77 | 75.2% / 67.2% | 90.99 |

The hard band volumes agree within their observed errors, while the
physical estimate reveals much larger contributions and remains dominated
by individual draws. This is a coverage finding, not convergence or an
equilibrium contradiction. It also shows why the exact density improvement
cannot be assessed solely from the earlier observed physical ESS.

`tools/compare_cayley_qwindow.py` independently rescans both campaigns,
retains every zero in its original denominator, checks shape/input/sample
hashes and physical definitions, and validates the geometric models before
reporting the density ratio. It requires equal population budgets when
using unweighted population scatter. Results are in
`runs/ab-inner-shoulder-matching-comparison-audited-20260920`; the full
regional progression is documented in `docs/shoulder-proposal-guides.md`.

The independent top-pose audit adds an important qualification. The largest
direct contribution has weighted/geometry guide radii 3.3097/1.9963, and
the frozen mixture assigns it about **1,527 times** the direct inner-cover
proposal density. The second-largest has a ratio near 8,832. The largest
pose's two cloud log weights are 46.4971 and 46.1250; its full AB geometry
passes. These events therefore do not demonstrate missing guided support
or a newly discovered contact basin. A proposal that visits a region
rarely can give a large importance weight to a pose that a guide already
samples efficiently. The direct/guide discrepancy still needs an
independent weight control; refitting merely because these direct weights
are large is not justified. That audit is preserved at
`runs/ab-inner-shoulder-top-pose-audit-20260920/analysis.json`.
