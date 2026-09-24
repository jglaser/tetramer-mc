# Fitting conditional curvature while preserving the affine control

The [existing prototype](kernel-transfer-prototype.md) supplies an explicit inverse
and density for an additive kernel shear. `tools/fit_kernel_shear.py` adds a
bounded fitting operation. It changes no physical kernel, guide format, running
campaign, region definition or convergence criterion.

The comparison keeps the 84-component protected guide's weights, means,
covariances and 50% uniform component fixed. Only the conditional kernel
displacements are learned. This separates representational curvature from a
new allocation of mixture weight or a new covariance fit.

## Coordinates and density

Let `u` be the original six-dimensional R4 coordinates, with outer pose-chart
Cholesky factor `L0`. Define `y = T u`, where `T` consists of the rows of `L0`
reordered as `[3,4,5,0,1,2]`. Its first three coordinates are the angular chart
coordinates up to their fixed offset. For each component, transform its mean and
covariance into `y`, then take the Cholesky factor in this angular-first order.
The first three whitened coordinates `a` therefore depend only on orientation;
the last three `b` are conditional translation residuals.

The shear is `S(a,b) = (a,b+f(a))`. Its inverse subtracts the same function and
its determinant is one. For component `k`, the density in original `u` is

```
q_k(u) = phi(S_k^-1(L_k^-1(Tu - mean_k))) * abs(det T) / det L_k.
```

The complete normalized proposal remains

```
q(u) = 0.5 * Uniform_R4(u) + 0.5 * sum_k weight_k * q_k(u).
```

Gaussian components retain their full support, including points outside the
original R4. The uniform component is supported on the original `u` ball, not a
new ball in transformed or component coordinates. Neither hard-valid-only
conditioning nor redrawing until inside R4 is used. Those operations would
require a new, generally unknown normalizer.

For physical poses, the proposal density still needs division by the original
outer chart/Haar Jacobian. The transform determinant above is separate from that
Jacobian. In a same-chart affine-versus-warped density ratio, the physical
Jacobian cancels. A future physical transport move must retain the existing
auxiliary-noise, pair-selection and physical acceptance corrections as well.

## One fixed fit

The offline diagnostic uses the existing pilot's r00/r01 populations in both
proposal arms for fitting: 262,144 original attempts. The r02/r03 populations
are held out from this fit. They were inspected in earlier work, so the new
comparison is retrospective rather than pristine prospective validation.
No old classifiers, Poisson clouds or physical audits are rerun.

Training assigns one third of its objective to each complete observed contact
class: native inside the old R5, native outside it, and contact without native
entry. Within a class, the weights are the saved positive linear importance
weights, including the physical Jacobian and the average of two Poisson-cloud
estimators. Training normalizes these weights using fitting populations only.
This region-balanced objective designs a proposal; it does not replace the
physical target or define a new physical normalizer. The current diagnostic is
native-informed and cannot establish geometry-only assembly.

Each component uses its soft responsibility under the *complete original
mixture*, including the uniform contribution in the denominator. Sampling branch
or component IDs are not treated as basin labels. With fixed responsibilities,
maximum likelihood becomes weighted regression of conditional translation `b`
on the angular kernel features:

```
coefficients = (K.T W K + ridge I)^-1 K.T W b.
```

`W` is normalized within that component's responsibility mass. Thus the global
surrogate's ridge penalty is weighted by the component responsibility mass.
The defaults are fixed before fitting: 16 weighted kmeans++ kernel centers,
bandwidth 1, coefficient ridge 0.01 and minimum responsibility-weight ESS 64.
Components below that ESS retain their affine form and their original weight.
All components remain in the complete density. There is one fit, with no
holdout-based parameter choice.

The fitting code reuses immutable `KernelShear` and `WarpedGaussianChart` objects.
`frozen-kernel-shear-mixture-v1` is a diagnostic serialization; it is not an
operational Rust guide. Frozen parameter use is compatible with the existing
reversible construction, but online adaptation and physical-kernel integration
are separate obligations.

## Validation and useful readouts

Eleven focused tests independently check complete densities, nonorthogonal
coordinate Jacobians, conditional orientation preservation, mixture
responsibilities and regression normal equations, deterministic serialization,
zero-weight invariance, curved held-out gain and an exact Gaussian-null loss.
For an exactly Gaussian target, any frozen nonzero shear must have expected
log-density change `-0.5 E[|f(A)|^2]`, which is nonpositive. This catches leakage
or misleading training-only gains.

On 1,024 archived fitting poses, the affine limit reproduces the existing
protected protein guide within 1.43e-14 in log density. This validates the
coordinate/density connection; it is not a sampling-speed measurement.

The diagnostic reports paired affine-versus-warped log densities and two saved
sample second moments, retaining every original attempt in the denominator:

```
noisy estimator M2: mean_R exp(2*z + log_q_source - log_q_candidate)
physical M2:        mean_R exp(pair_log_weight_1 + pair_log_weight_2
                             + log_q_source - log_q_candidate)
```

Here `z` is the log of the saved two-cloud average importance weight, not the
average of log cloud weights. The second expression uses independent clouds to
remove estimator-noise inflation of the physical second moment. Both are
reported separately for each source population and each region, including all
radial, angular and 64 orthant subdivisions. An unobserved contribution stays
unresolved. Neither a density improvement nor a retrospective second-moment
improvement proves unseen-mode coverage, faster reversible mixing or assembly.

One additive layer changes the conditional mean while keeping its Gaussian
width and the conditioning marginal fixed. Failure of this specific fit would
not rule out variable conditional scales or additional invertible layers.
