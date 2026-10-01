# Candidate: two contact distances and an azimuth

**Unimplemented and unvalidated.** This is a normalized integration-guide
construction to examine with passive geometry checks before considering new
physical sampling. It changes all three translation coordinates at fixed
orientation, addressing the fixed-x guide's frequent blocked lines. It does
not resolve the failed contact-weight gates or establish assembly stability.

Keep the repaired shape, original R4/capture domain, complete classifier,
1.5 Å radius, 0.035 Å⁻³ activity and physical Jacobian unchanged. This proposal
may use frozen integration-training contact labels; it is not a claim of
native-blind assembly discovery. Retain the old defensive proposal throughout.

## Exact three-dimensional translation law

At a fixed proper orientation R, choose a **frozen label** containing two atomic
contacts, one to each scaffold. For moving atom b_j and world fixed atom f_j,
define

\[
c_j=f_j-Rb_j,\qquad d_j=R^{\rm atom}_{b_j}+R^{\rm atom}_{f_j},
\qquad D=|c_2-c_1|.
\]

The two selected distances are r_j=|t-c_j|. For a frozen width w, use

\[
P(D)=[d_1,d_1+w]\times[d_2,d_2+w]
\cap\{r_1+r_2\ge D,\ r_1-r_2\le D,\ r_2-r_1\le D\}.
\]

This is a convex polygon with area A(D). Uniform sampling is elementary:
clip the rectangle by the three half-planes, triangulate the polygon, choose
a triangle in proportion to its area, and use square-root barycentric
sampling. The widths 0.02, 0.1 and 0.5 Å concern **atomic surface gaps**;
they do not change the native-registration threshold or depletion radius.

Let e=(c_2-c_1)/D and choose a deterministic orthonormal transverse basis
(e_1,e_2). For a sampled interior radius pair,

\[
h=\frac{r_1^2-r_2^2+D^2}{2D},\quad
\rho=\sqrt{r_1^2-h^2},\quad
t=c_1+he+\rho(e_1\cos\phi+e_2\sin\phi).
\]

Apart from measure-zero degeneracies, this parameterization is one-to-one for
0≤φ<2π and

\[
d^3t=\frac{r_1r_2}{D}\,dr_1\,dr_2\,d\phi.
\]

For uniform φ, the normalized translation density is therefore

\[
f(t\mid a,p,w)=\mathbf1_{(r_1,r_2)\in P(D)}
\frac{D}{2\pi A(D)r_1r_2}.
\]

This is uniform in the two radii, **not** uniform in translation volume.
Neither is required to equal the physical equilibrium density. Other atom
pairs can still overlap; whole-union core, R4 and capture failures contribute
zeros to the original unconditional estimator. Do not retry a hard-invalid
draw or truncate this density to R4 without its conditional normalizer.

## Angular marginal, mixture and fallback

Transform each old latent Gaussian to the raw chart x=(t_raw,a)=m₀+L₀u.
Its three-dimensional angular marginal p_k(a), and conditional translation
Normal n_k(t_raw|a), follow from the existing full covariance. World translation
differs from t_raw by a rigid transformation and has unit volume determinant.

Choose component k with its unchanged weight π_k, then an explicit finite
pair-label/width distribution η_{p,w|k}. Draw the old angular marginal once.
If D is below a predeclared positive floor, or P(D) has zero area, draw the
original conditional translation Normal instead. Any numerical area threshold
must also be fixed and evaluated identically in sampling and density scoring.
The fallback depends only on orientation and the frozen labels, never on the
selected endpoint. No angular redraw is permitted.

Writing f for the geometric law or its declared fallback, the latent density is

\[
H(u)=|\det L_0|\sum_{k,p,w}\pi_k\eta_{p,w|k}
p_k(a)f_{k,p,w}(t\mid a),
\qquad
q=0.5U_{R4}+0.25G_{92}+0.25H.
\]

Every conditional translation law integrates to one, so H is normalized on
all R6 and preserves the old Gaussian angular marginal. In particular
q≥0.5q_old. This is not the angular marginal after selecting hard-valid R4
endpoints. The estimator remains the complete J W/q with hard-invalid zeros.
Sum **all** labels, components, widths and fallback branches at every queried
pose; using only the generating label's density is incorrect. Distinct labels
may represent overlapping translation sets and their densities must add.

Coincident effective centers, zero-area polygons, zero widths and tangent
circles require explicit cases. Exact tangencies have zero continuous mass;
exclude their coordinate singularities from inversion tests. Use a stable
triangle-area expression for ρ near tangency and a cancellation-aware polygon
area calculation. Numerical errors are implementation obligations, not a
reason to silently redraw. A deterministic basis convention may jump with
orientation; it remains valid if generation and density use the same basis.

## Optional localized azimuth, with a uniform control

Uniform φ is the simplest exact control but discards useful localization.
For each component, project its conditional **world-translation mean** onto
the plane of the sampled circle. Its polar angle gives φ₀; the tangent there
is v=−e_1 sinφ₀+e_2 cosφ₀. Transform the conditional covariance into the same
world frame, obtaining S_k, and define an angular scale such as

\[
\gamma=\operatorname{clip}\left(
\frac{\sqrt{v^TS_kv}}{\max(\rho,\rho_{\min})},
\gamma_{\min},\gamma_{\max}\right).
\]

All floors, caps and the mixture fraction must be frozen before evaluation.
This scale is a **guide choice**, not a physical law or an exact conditional
Gaussian on the circle. A vanishing projected mean can use uniform φ by a
declared projection threshold; that decision must enter the density too.

With η=exp(−γ), a wrapped Cauchy density and its inverse sampler are

\[
c(\phi)=\frac{1-\eta^2}
 {2\pi[1+\eta^2-2\eta\cos(\phi-\phi_0)]},\qquad
\phi=\phi_0+2\arctan\!\left[
 \tanh(\gamma/2)\tan\{\pi(U-1/2)\}\right]\pmod{2\pi}.
\]

Use a fixed mixture p_φ=(1−b)/(2π)+b c(φ), with 0≤b<1. Replace 1/(2π)
in f by this **complete** p_φ. It is normalized conditionally on every
(a,r_1,r_2,k,p,w), so the angular-marginal argument is unchanged. The b=0
control remains uniform azimuth. Stable half-angle/log-density evaluation and
open-interval uniforms would need reference checks before implementation use.

## What the saved geometry suggests

The 78 previously inspected competing-55 rows contain eight distinct tuples
of the two nearest atomic contacts; none uses the same moving atom twice.
Their effective-center separations D range from 2.202 to 5.832 Å, and their
intersection-circle radii range from 1.441 to 3.012 Å, with median 2.319 Å.
At fixed orientation, the existing raw translation conditional principal
standard deviations are:

| Old component | Smallest / largest SD, Å |
|---|---:|
| 41 | 0.06879 / 0.09373 |
| 85 | 0.06095 / 0.07568 |
| 87 | 0.06001 / 0.06922 |

A uniform azimuth traverses a circle of circumference 9–19 Å, much wider
than these conditional distributions. It may replace fixed-line blockage
with very low whole-union or R4 survival. The small observed label catalogue
suggests that azimuth/core geometry deserves attention before label
proliferation in this particular family. It does not establish coverage of
unseen contacts. These 78 rows are exploratory diagnostics, not permitted
new training data for a purported independent test; pair-label selection
should use the original designated training split and freeze before scoring.

Unlike the implemented line guide, this construction imposes only **two
selected atomic** contacts and performs no all-core interval exclusion.
Unlike the earlier conditional-ray member guide, it redraws all translation
coordinates rather than keeping a sampled ray, and it targets physical
surface gaps rather than a native-classifier boundary. It cannot repair
missing angular support, and it does not condition on the entire many-body
depletion weight. Its complete q needs only a small polygon and distance
calculation per distinct label, without a BVH interval traversal per density
query. Whole-union endpoint checks remain necessary. Actual CPU benefit and
retention must be measured, not inferred from this operation count.

## Provenance and minimum validation

Read-only calculations above use the following immutable sources:

| Source | SHA256 |
|---|---|
| [Saved geometry](../runs/contact-line-passive-preparation-v3-20261001/axis-diagnostics.json) | `56e02a2241e1b71d4fb54e74f783f72bde8e66ea9bb7c92e8a9d9752918649c1` |
| [Old92 guide](../runs/contact-line-passive-preparation-v3-20261001/common/original92.json) | `a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0` |
| [Region/chart](../runs/contact-line-passive-preparation-v3-20261001/common/region.json) | `924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02` |
| [Hard shape](../runs/contact-line-passive-preparation-v3-20261001/common/shape.json) | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |

Before a physical job, validate polygon areas/sampling, inverse reconstruction,
the translation Jacobian, label exchange symmetry, normalized full mixtures,
degenerate fallbacks, the unchanged angular marginal, uniform-azimuth limit,
and independent Python/Rust densities. Then freeze a bounded **zero-Poisson**
comparison of old92, uniform azimuth and localized azimuth, with fresh streams,
archived target/breadth probes, every attempted draw retained and total CPU
reported. Measure whole-union validity and joint-contact retention, not just
the guaranteed selected-pair contacts. No allocation or launch is frozen by
this design note. A favorable passive result would justify further validation;
it would not itself establish physical contact weights or assembly.
