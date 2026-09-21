# Integrating the contact region near the native-entry boundary

The current proposal targets a known coverage problem in the fixed R4 contact
integral. It changes where test poses are drawn, while preserving the physical
domain, hard geometry, bath and strict native-entry classifier. This is an
independent importance calculation; it is not an assembly move or an adaptive
production kernel.

The [saved-row diagnostic](../runs/mobile-threshold-coverage-diagnostic-20260921/analysis.json)
finds that 623 of the 65,536 uniform draws lie within a small body-registration
margin and carry 96.46% of the observed unregistered-contact weight. The three
largest contributions carry 92.49% and occupy distinct latent directions.
The old global proposal misses much of this neighborhood. These are empirical
coverage diagnostics, not upper bounds on weight elsewhere. Repeating Poisson
clouds alone cannot repair missed pose volume.

## Normalized proposal

Use the same ordinary six-dimensional chart as the uniform reference,
\(x=m+L\xi\), with \(\|\xi\|\le R=4\). Split its raw coordinates into
translation and scaled Cayley rotation, \(x=(x_t,x_\omega)\). The exact angular
marginal of a uniform six-ball is

\[
p_\omega(x_\omega)=
\frac{8(R^2-a^2)^{3/2}}{\pi^2R^6\sqrt{\det\Sigma_{\omega\omega}}}
\,\mathbf1_{a^2<R^2},\qquad
a^2=(x_\omega-m_\omega)^T\Sigma_{\omega\omega}^{-1}
(x_\omega-m_\omega).
\]

Here \(\Sigma=LL^T\). All six columns of the angular rows of \(L\) enter the
marginal covariance; using only its lower-right Cholesky block would give the
conditional covariance and the wrong density. The implementation draws a full
uniform six-ball vector and retains its angular coordinates.

At that orientation \(R_b\), a selected moving member \(m_k\) has world
displacement \(t+R_bm_k-y_k\) from target member \(y_k\). Consequently its
registration-error shell is a sphere centered on
\(c_k(R_b)=y_k-R_bm_k\) in translation space. Choose one of 24 fixed entries
and draw an isotropic direction and a radius uniformly in \(r^3\).
The entries are the four members of each of two nearby native interfaces,
with inner error radius 2 Å and outer radii 2.02, 2.1 and 2.5 Å. Their weights
are equal. These guide shells do not replace atomic overlap checks or the full
native classifier.

The normalized conditional translation density sums every shell containing
the pose, including overlaps and nested shells:

\[
h(t\mid R_b)=\sum_k\frac{w_k}{V_k}
\mathbf1_{r_{k,-}\le\|t-c_k(R_b)\|\le r_{k,+}}.
\]

With a 20% uniform branch, the complete density in the original whitened
coordinates is

\[
q_\xi(\xi)=\frac{\alpha}{V_6(R)}\mathbf1_{\|\xi\|\le R}
+(1-\alpha)\det(L)\,p_\omega(x_\omega)h(t\mid R_b),
\qquad\alpha=0.2.
\]

The factor \(\det L\) converts the raw chart density back to whitened
coordinates. The physical translation/normalized-Haar Jacobian remains

\[
J_\xi=\frac{\det L}{\pi^2\ell^3(1+\|x_\omega/\ell\|^2)^2}.
\]

Each unconditional draw contributes \(\mathbf1_{\rm target}J_\xi W/q_\xi\),
where \(W\) is the existing unbiased many-body depletion estimator. Guided
draws outside R4 or inside atomic cores contribute zero; they are never
retried. The positive uniform branch preserves support throughout the target
even if the guide misses a relevant interface. It does not guarantee a small
variance or establish coverage outside this finite region.

## Implementation and validation

[The Rust implementation](../src/latent_region/entry_shell.rs) has an explicit
guide schema and retains the old Gaussian proposal unchanged. The separate
[NumPy/SciPy density](../tools/entry_shell_proposal.py) reconstructs all proposal
terms from saved poses. The auditor checks the full summed density and selected
shell support, retaining the original unconditional denominator.

Fifteen Rust tests pass, including angular marginal checks, overlapping-shell
density, radial moments, the uniform limit and existing normalizer regressions.
Six new Python tests additionally cover marginal normalization, duplicate-shell
invariance and global isometries; sixteen previous latent-region tests pass.
Fourteen controller tests cover frozen-source bindings, output identity,
failure draining, worker limits and retry refusal.

The [independent sphere validation](../runs/entry-shell-sphere-validation-20260921/validation.json)
checks 32,768 Rust draws against exact hard-sphere and ideal-depletion integrals
using one-dimensional quadrature with the full Haar measure. Both predeclared
statistical checks pass. The zero-activity estimate differs by 0.008 observed
standard errors; the nonzero-activity estimates differ by 2.58 (depletion) and
2.65 (hard volume). Independent log proposal densities agree within 7.2e-15.
These finite statistical controls complement the density identities; they are
not a proof that every future integral has converged.

## Fixed protein comparison

The [prepared guide](../runs/mobile-threshold-entry-shell-preparation-20260921/plan.json)
freezes the existing B-anchored R4, fixed two-neighbor scaffold, repaired hard
shape, depletant radius 1.5 Å and activity 0.035 Å⁻³. All original-q values are
included. The [controller](../tools/run_entry_shell_reference_campaign.py)
allocates four fresh independent populations of 16,384 draws, two independent
clouds per pose and auxiliary intensity ratio 64. It runs at most four workers
and preserves every launched population on failure.

The [completed comparison](../runs/mobile-entry-shell-reference-comparison-20260921/report.md)
uses the earlier equal-N uniform reference on the same region, without pooling
discovery and validation samples. All four populations and the independent raw
audit completed. All 65,536 proposal densities agree with the separate Python
calculation within 9.6e-14 in log density. Eight downstream analysis tests pass.
The [implementation validation ledger](../runs/mobile-entry-shell-reference-validation-20260921/validation.json)
binds the completed physical test, reference checks and source hashes.

| Quantity | Entry-shell guide | Earlier uniform |
|---|---:|---:|
| Hard-region log Q0 | −11.91099 | −11.89577 |
| Hard-region observed relative error | 2.37% | 1.04% |
| Native-entry log Qz | 49.01402 | 58.95151 |
| Contact without entry log Qz | 41.10992 | 42.76993 |
| Contact without entry ESS | 1.39 | 2.72 |
| Contact without entry row relative error | 84.82% | 60.68% |
| Contact without entry population relative error | 80.02% | 44.11% |
| Sampler CPU seconds | 98.55 | 440.44 |

The hard volumes agree within their observed errors, but the weighted estimates
do not establish convergence. The new guide does not demonstrate better contact
sampling. Of 52,320 guided attempts, only 831 land inside R4 and 169 contribute
after hard tests; 160 are contacts without native entry. These contribute only
0.28% of the realized no-entry weight. The uniform branch supplies the remaining
99.72%. Most shell directions miss the narrow translation/rotation constraints
of the chart. Its lower runtime mainly accompanies fewer contributing poses
and fewer Poisson evaluations, and must not be called a sampling speedup.

![Separate estimates on the unchanged target](../runs/mobile-entry-shell-reference-comparison-20260921/entry-shell-reference.png)

This rejects the present guide as a demonstrated efficiency improvement while
retaining its normalized density and reference checks. A refinement would need
to target the intersection with the conditional translation ellipsoid, including
its proposal normalization, rather than sampling whole member-error shells.
Population agreement and missing-region coverage remain necessary before any
global thermodynamic conclusion. These results do not change the validated
assembly move model supplied in the examples.
