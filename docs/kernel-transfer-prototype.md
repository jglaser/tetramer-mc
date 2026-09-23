# Nonlinear kernel transfer within the existing reversible construction

A covariance map is affine in its local six-dimensional chart. It matches
orientation, scale and linear dependence, but cannot in general turn one curved
translation–rotation relation into a different curved relation. A Mercer kernel
can supply nonlinear features; it is not itself a normalized transition density
or a reversible Markov kernel.

There is a direct precedent in [Kernel Adaptive Metropolis–Hastings](https://proceedings.mlr.press/v32/sejdinovic14.html):
feature-space covariance yields a location-dependent Gaussian proposal with an
evaluable forward and reverse density. For this repository, another useful
construction preserves the explicit inverse used by `basin_involution.rs`.
It uses an additive coupling transformation of the kind introduced in
[NICE](https://arxiv.org/abs/1410.8516), with a finite kernel expansion as the
coupling function. This particular prototype is our implementation of that
specialization, not the KAMH algorithm.

![Covariance transfer versus an invertible kernel warp](../runs/kernel-shear-schematic-20260923/covariance-versus-kernel-transfer.png)

The figure uses known normalized two-dimensional distributions. The middle
transfer uses full means and covariances evaluated by Gaussian quadrature;
the right transfer uses the known nonlinear inverse and forward maps. It
illustrates representational capacity, not a trained protein model or measured
sampling improvement. Points are proposals before physical acceptance.

## Smallest invertible extension

Split standardized coordinates into disjoint groups A and B. Define

\[
 f(z_A)=\sum_j c_j\exp\{-\|z_A-r_j\|^2/(2h^2)\},\qquad
 S(z_A,z_B)=(z_A,z_B+f(z_A)).
\]

The coefficients describe a displacement and need not be positive. The inverse
subtracts the same displacement because the conditioning coordinates do not
change. The Jacobian is triangular with diagonal one. The normalized chart

\[
 x=F_i(z)=\mu_i+L_iS_i(z),\qquad z\sim N(0,I)
\]

therefore has the exactly evaluable Euclidean density

\[
 q_i(x)=\phi\!\left[S_i^{-1}(L_i^{-1}(x-\mu_i))\right]/|\det L_i|.
\]

With orientation-like coordinates in A and translation-like coordinates in B,
the kernel describes a nonlinear displacement needed to maintain contact while
rotating. Other splits, such as five tangential coordinates conditioning one
normal displacement, are possible. One additive layer has fixed conditional
width and leaves its conditioning marginal unchanged; it is not a universal
six-dimensional model by itself. Alternating splits or positive, variable
conditional scales could extend it, with the corresponding determinant included.

## Reverse move and correction

Encode the source pose with the inverse of its frozen chart. Use the same latent
involution as the existing implementation:

\[
 (z,\xi)\longmapsto
 (cz+s\xi,\;sz-c\xi),\qquad s=\sqrt{1-c^2}.
\]

Decode the first result using the destination chart. Swap source and destination
labels and retain the second result as reverse noise. Applying the same operation
again recovers both the original position and noise. The extended Euclidean
log correction is

\[
 \log|\det L_b|-\log|\det L_a|
 +\tfrac12(\|\xi\|^2-\|\xi'\|^2).
\]

No independently learned reverse map is needed. This remains valid at c=0,
where the fixed-noise pose derivative is singular but the extended map is
invertible, and at c=±1. For a fixed symmetric chart-pair law, adding this
correction to the exact target-density log ratio gives the usual involutive
Metropolis rule. A state-dependent pair-selection law needs its additional
forward/reverse factor. A Mercer similarity alone does not cancel that factor.

The new prototype is **Euclidean only**. Applying it to poses still requires
the existing rotation-chart/Haar volume factors, chart seams and support checks,
physical hard/depletion acceptance, frozen parameters during each move, and
explicit accounting for the pair-selection law. Online learning requires a
separate valid adaptive or auxiliary-state construction. The existing Lean
theorems are not a proof that these new implementation obligations have been met.

## Implementation and checks

`tools/kernel_shear.py` supplies immutable kernel shears, warped Gaussian charts,
exact density evaluation and the extended pair map. It does not modify Rust,
production guide formats, physical kernels or any convergence gate.

The 13 tests in `tools/test_kernel_shear.py` cover inverses, the Gaussian limit,
independent quadrature of a curved density in output coordinates, finite-
difference Jacobians of both charts and extended moves, endpoint correlations,
correction antisymmetry, parameter immutability and unit acceptance when the
toy target exactly equals the normalized chart distributions. The last check
is explicitly about that toy target, not a physical protein target.

Run them with:

```sh
/home/xvg/protein-nucleation/.venv/bin/python -B -m unittest discover \
  -s tools -p test_kernel_shear.py -v
```

`tools/plot_kernel_shear.py --out FRESH_DIRECTORY` produces the schematic and
a provenance receipt. The source/destination reconstruction and reverse-map
errors are retained in the receipt.

## Connection to the current contact-weight failure

The separate [cached tail diagnostic](../runs/contact-tail-geometry-diagnostic-20260923/analysis.json)
examined 2,752,512 already-completed attempts in 28 populations, without new
geometry predicates, bath queries or native classification. Its selected
high-contribution poses are generally within a few Mahalanobis units of existing
Gaussians. Only 0–2 of the 32 selected weak-tail rows per source arm are
uniform-dominated. Their cached atom-surface gaps often lie near 0.01 Å, which
suggests thin contact layers worth investigating. This is a selection-biased
description of observed tails, not a measurement of the full equilibrium layer.

The pointwise envelope `0.5 U + 0.5 max_j G_j` is often much larger than the
current mixture density at those rows. It bounds each saved-row second-moment
contribution attainable by weight changes from below, but is not normalized,
need not be attainable by one global weight vector, and gives no bound on unseen
physical mass. Thus the evidence points to component allocation as an immediate
issue; it does not yet demonstrate that a nonlinear warp is required.

A useful subsequent test would freeze a geometry-trained warp and compare it
with the affine map at matched budgets, retaining all rejected states and the
full physical correction. Held-out density coverage, failed contact subdivisions,
and effective contact samples per CPU remain the criteria. The schematic and
unit tests alone establish neither protein efficiency nor finite-system assembly.
