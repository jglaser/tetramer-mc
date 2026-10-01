# Conditioning a Gaussian guide on a rotational coordinate

This extends the normalized [translation conditioner](line-conditioned-gaussian-guide.md)
to any of the six **raw chart coordinates**. It retains the fixed integration
region, physical measure and complete mixture weighting. The new schema
`defensive-hard-free-pose-line-guide-v1` uses `raw_pose_axes`, with indices 0–2
for translation and 3–5 for length-scaled Cayley coordinates. The old schema
still accepts translation axes only. The extension is initially available in
`contact-line-guide-audit`, which generates no Poisson clouds or physical mass
estimates; the production normalizers reject the new schema.

The motivation is specific. Translation conditioning cannot repair a poorly
proposed orientation. Angular conditioning can remove hard-forbidden portions
of an angular conditional Gaussian, but it still retains the other five raw
coordinates. It cannot guarantee discovery of a missing contact basin. A
useful gain must ultimately appear in independently validated physical
importance ESS per CPU, including competing and remaining regions.

## Geometry

Write the Cayley vector as \(c=v+s e_j\), with \(v_j=0\); the sampled raw
coordinate is \(x_{3+j}=\ell s\). The center stays fixed. For a moving atom
offset \(a=R_{\rm anchor}a_{\rm body}\), express a fixed atom relative to
the moving center in the chart frame as
\(b=R_{\rm fixed}^{T}(y-t)\). With contact radius \(r\), set

\[
\begin{aligned}
K&=|a|^2+|b|^2-r^2,\\
N_0&=(1-|v|^2)a+2v(v\cdot a)+2v\times a,\\
N_1&=2[e_j(v\cdot a)+v a_j+e_j\times a],\\
N_2&=-a+2e_j a_j.
\end{aligned}
\]

Multiplying the squared-distance inequality by the positive denominator
\(1+|v|^2+s^2\) gives the strict hard-overlap condition

\[
(K-2b\cdot N_2)s^2-2(b\cdot N_1)s
+K(1+|v|^2)-2b\cdot N_0<0.
\]

Thus each atom pair contributes quadratic intervals. Downward-opening,
linear, constant and tangent cases matter; overlap is not always between two
roots. Union the forbidden leaf intervals and subtract them from the finite
R4 chord. Center capture is a single Boolean for an angular line. Enclosing
spheres in the body-frame BVH only rule out possible atom overlaps; their
overlap cannot establish atomic overlap.

An equivalent construction explains the geometry:

\[
h=\sqrt{1+|v|^2},\quad k=(e_j+v\times e_j)/h,\qquad
\operatorname{Cayley}(v+s e_j)
=\operatorname{Rot}(k,2\arctan(s/h))\operatorname{Cayley}(v).
\]

Every atom follows a circle about a common axis. The raw parameter is nonlinear
in the physical rotation angle. Analytic intervals are exact for the stated
sphere-union model; floating-point coefficients, roots, frame transforms and
BVH pruning remain implementation obligations.

## Normalized conditional law

The existing Gaussian conditional machinery works after placing any selected
raw coordinate last in the covariance factorization. Draw the frozen mixture
component and its six-dimensional Gaussian; retain five coordinates; replace
the selected coordinate by its Normal distribution restricted to the allowed
interval union. Its normalization is the sum of interval CDF differences.
If that mass is at or below the declared floor, retain the original coordinate.
Do not redraw the outer five coordinates.

The same normalization and fallback must be evaluated for every component and
advertised axis when scoring the complete proposal. Uniform defensive mass is
retained. Each conditional branch integrates to one on all R6; its fallback
may generate an endpoint outside the physical integration region, which
contributes zero with its attempted-draw denominator retained.

Unlike a translation line, an angular line has a varying physical Jacobian:

\[
J(u)=\det L_0/[\ell^3\pi^2(1+|c|^2)^2].
\]

It belongs in the physical importance weight \(J\widehat W/q\), **not** in
the conditional Normal mass. This proposal preserves the five-coordinate
Gaussian marginal; it does not preserve the full original angular marginal.
No state-dependent axis selection or unaccounted move-rate correction is used.

## Relation to earlier rotation experiments

The prototype in
`/home/xvg/protein-nucleation/src/cayley_line.rs` already established quadratic
pure-rotation geometry (and quartic coupled affine translation/rotation).
Its [contact-motion audit](/home/xvg/protein-nucleation/results/hierarchical-pose/cayley-line/contact-motion-audit.md)
found improvements in local motion without competing-to-native entry. Those
moves began at the current physical pose, integrated a cloud-conditioned line
law and corrected a state-dependent direction proposal. Their small capture
domains and one-/three-neighbor scaffolds differ from the present R4,
two-neighbor weight calculation.

Here the line conditions a global frozen Gaussian integration proposal on hard
geometry only. The depletion weight and Haar Jacobian remain outside that
conditional proposal. Earlier local-motion gains therefore neither validate
this new density nor predict its efficiency.

## Validation allocation

The [preparation script](../tools/prepare_pose_line_toy_validation.py) freezes
512 unconditional draws each for a sphere, dumbbell and asymmetric 12-sphere
union, using correlated chart covariances, rotated frames and all six axes.
The [allocation](../results/pose-line-toy-validation-20261001/allocation.json)
is separate from every protein population and contains no physical cloud
sampling. All 1,536 attempted draws completed and passed independent Python
reconstruction of geometry, complete densities, Jacobians, retained coordinates
and conditional CDFs. Maximum discrepancies were 2.75×10⁻¹⁴ in log density,
1.78×10⁻¹⁵ in log Jacobian, 5.17×10⁻¹⁴ in interval endpoints and 2.30×10⁻¹⁵ in
the inverse CDF. No draw was retried.

Six new Rust geometry tests and 22 shared guide tests passed (the latter include
five new pose-line checks), as did eleven independent Python reference tests.
The geometry tests include 13,878 direct atom-predicate witnesses across shifted
and rotated frames, strict tangencies, disconnected intervals and extreme
finite chords. The guide checks include unchanged legacy translation streams,
fixed-center capture, conditional normalization and the varying Jacobian.
See the [geometry receipt](../results/cayley-axis-geometry-validation-20261001/validation.json),
[guide regression log](../results/pose-line-toy-validation-20261001/guide-regression.log)
and [independent reconstruction receipt](../results/pose-line-toy-validation-20261001/independent-validation.json).

These tests validate a bounded numerical reference, not a certified
floating-point implementation or protein performance. The next physical stage
keeps the existing translation guide fixed for population-size sensitivity.
No angular protein draws or assembly moves are included here.
