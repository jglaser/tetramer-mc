# A bridge bottleneck and an endpoint-proposal alternative

The completed importance-sampling pilot identifies a plausible mechanism for
the earlier SMC discrepancy. Under the original annealing bridge, native entry
outside the exact old-R5 intersection loses most of its probability midway
through the path, then regains substantial endpoint weight. Replacing the
initial density with the already frozen endpoint-oriented proposal removes
that dip in the saved-data diagnostic. This is a reason to test a different
SMC bridge, **not a demonstrated SMC speedup or a physical assembly result**.

Both calculations retain the repaired shape, fixed two-neighbor scaffold, R4
domain, complete native classifier, Cartesian translation × normalized SO(3)
Haar measure, radius 1.5 Å and endpoint activity 0.035 Å⁻³. Neither expands the
sampled domain. Both reuse all 131,072 completed pilot attempts, including hard
zeros, two saved clouds per valid pose and the original independent populations.
No new poses, Poisson clouds or classifications are generated.

## Original bridge

Write the physical bridge density as

\[
f_\beta(x)=H(x)g(x)^{1-\beta}\exp[\beta z C(x)],
\qquad 0\le\beta\le1,
\]

where the original normalized proposal \(g\) is the 50/50 mixture of the
current chart ball and the full unfiltered old chart ball. Old-native,
capture and hard constraints must not be applied to that second proposal
component. The target indicator \(H\) carries the physical region and hard
constraints. Intermediate densities are computational bridges, not physical
baths at a rescaled activity.

For an IID latent draw with density \(q\), physical Jacobian \(J\), certified
overlap lower bound \(L_r\), and saved count
\(K_r\sim\operatorname{Poisson}[\lambda(C-L_r)]\), use

\[
\widehat W_{\beta,r}=e^{\beta z L_r}(1+\beta z/\lambda)^{K_r},
\qquad
\widehat Q_\beta=\frac1N\sum_i
H_i\frac{J_i}{q_i}g_i^{1-\beta}
\frac{\widehat W_{\beta,1,i}+\widehat W_{\beta,2,i}}2.
\]

The Poisson generating function gives
\(E\widehat W_{\beta,r}=e^{\beta zC}\). Raising a noisy endpoint weight to
\(\beta\) would not give this identity. All attempted draws remain in \(N\).
At zero the integral is proposal feasibility \(\int Hg\), not hard volume;
at one it is the original physical mass.

![Original bridge diagnostic](../results/smc-bridge-bottleneck-20261001/figures-v2/smc-bridge-bottleneck.png)

The conditioned IID estimate of the remaining-native fraction falls from
8.85% to 0.619% at the smallest of nine sampled grid values (β=0.375), then
rises to 47.05%. The independent baseline shows the same qualitative dip.
**Both arms fail concentration checks near their minima**, so the depth is
not converged. These observations support an intermediate bottleneck, without
certifying its precise magnitude or coverage of unseen modes.

The old broad SMC populations have only 2, 2, 5 and 4 initial families
represented in this class at their minima. Later increases mean surviving
lineages entering the class, not restoration of extinct initial ancestry.
Broad and narrow SMC recover respectively 0.216 and 0.773 times the conditioned
IID endpoint remaining-native mass. Normalizer-times-class-indicator masses,
not endpoint fractions alone, define these comparisons. The
[complete report](../results/smc-bridge-bottleneck-20261001/figures-v2/README.md)
retains every regional and stratum disagreement.

## Alternative initial density

For each IID arm separately, take its own normalized physical proposal
\(g_{\rm new}=q/J\). Its defensive uniform R4 component guarantees support on
this regional target. This support argument does not automatically extend to
the full vessel. The new contribution is

\[
H(J/q)^\beta\,\overline{\widehat W_\beta}.
\]

With \(h=\log J-\log q\), it can be reconstructed from the first diagnostic:

\[
\log w_{\rm new,\beta}
=\log w_{\rm old,\beta}-(1-\beta)(h+\log g_{\rm old}).
\]

Invalid rows are masked before arithmetic. Valid zero-stage weights are
exactly one after checking cancellation; endpoint weights are preserved
directly. The companion reads only authenticated derived arrays and saved
classification records, without rereading raw poses or cloud rows. Every
endpoint class and stratum moment is identical to the first diagnostic.

| Initial proposal | Remaining-native fraction at β=0 | Range over the nine new-bridge stages | β=1 |
|---|---:|---:|---:|
| Baseline IID proposal | 22.94% | 22.94–42.85% | 42.59% |
| Conditioned IID proposal | 29.09% | 29.09–51.17% | 47.05% |

For the conditioned bridge, all four nonempty decision regions pass the
aggregate population relative-SE, importance-ESS and largest-contribution
diagnostics at every sampled stage. Remaining-native ESS ranges from 6,132
at zero to 642 at one. All original strata are retained. The baseline still
fails its known endpoint tail diagnostics. Unobserved unbound contribution
remains unresolved by these samples; the separately derived R4 bound is not
a full-vessel bound.

![Alternative bridge diagnostic](../results/endpoint-proposal-bridge-20261001/figures-v2/endpoint-proposal-bridge.png)

The two new bridges have **different intermediate targets**, because their
initial proposals differ. There is no intermediate cross-arm agreement test.
Old/new fraction changes use paired whole-population covariance, and all β
values share saved draws/clouds. Endpoint equality within each arm is an
algebraic identity, not independent confirmation. The proposal remains native
informed. This calculation does not establish geometry-only discovery.

## Implementation and next test

The two tools are
[`analyze_smc_bridge_bottleneck.py`](../tools/analyze_smc_bridge_bottleneck.py)
and [`analyze_endpoint_proposal_bridge.py`](../tools/analyze_endpoint_proposal_bridge.py).
They preserve immutable source/input hashes and independent population
statistics. Eight specific arithmetic tests cover the Poisson expectation,
its difference from a powered estimator, zero activity, invalid attempts,
exact endpoints, a discrete unequal-volume reference and paired uncertainty.
Seventeen existing controller/bridge tests also passed in the same test run.
An independent code review found no remaining blocker.

A future SMC implementation must change its initial draw law, incremental
proposal-density correction and mutation acceptance together. Density
evaluation cost and mutation mixing still need a matched benchmark. This
retrospective alone measures neither. The fixed larger IID campaign and
defensive-probability/cloud-intensity controls retain priority and their
original allocations. No full-vessel or assembly gate is opened by this work.

Artifacts:

- [Original bridge analysis](../results/smc-bridge-bottleneck-20261001/analysis.json)
- [Alternative bridge analysis](../results/endpoint-proposal-bridge-20261001/analysis.json)
- [Alternative bridge report](../results/endpoint-proposal-bridge-20261001/figures-v2/README.md)
- [Validation receipt](../results/smc-bridge-diagnostics-validation-20261001/validation.json)

The finite-system conclusion remains **unresolved**.
