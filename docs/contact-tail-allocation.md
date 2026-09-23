# Observed contact tails and subdivision-aware mixture allocation

Two bounded diagnostics follow the [completed pilot](smc-guide-pilot-results.md).
They use only archived poses, classifications and Poisson weights. They generate
no new physical draws, geometry predicates or operational guides and change no
convergence gate.

## Allocation versus missing component geometry

The [tail analysis](../runs/contact-tail-geometry-diagnostic-20260923/analysis.json)
reuses 2,752,512 attempts from 28 populations: the two fresh arms and five older
confirmation arms. Cohorts and cloud intensities remain separate. It preserves
all attempted denominators and independently reconstructs each source's second
moment within 2.84×10⁻¹⁴ in log units.

Selected high-contribution competing-contact rows generally lie a few
Mahalanobis units from an existing Gaussian. Median nearest-component distances
among selected weak-tail rows range from 2.04 to 2.95 across source arms. Only
0–2 of 32 selected rows per arm have proposal density dominated by the uniform
component. Representative rows could have roughly 18–89 times higher density
under the pointwise envelope

\[
 q_{\max}(u)=\tfrac12 U_{R4}(u)+\tfrac12\max_j G_j(u).
\]

For fixed component geometry, every normalized weight mixture is at most this
envelope. Substituting the envelope in the denominator therefore gives a lower
bound on **saved-sample** second-moment estimates attainable by reweighting.
The envelope is not a normalized proposal, may not be attainable by one weight
vector, and gives no bound on the true integral or unseen tails.

For competing contacts, envelope/original-bank moment ratios are about 0.0277
on fresh-bank data, 0.0370 on fresh-new-guide data and 0.0281 on old-bank data.
Thus existing components can put appreciable density on the observed troublesome
poses. Simple absence of nearby component geometry is not established as the
main cause of their large weights.

The selected rows' cached minimum atom-surface gaps have medians near
0.010–0.016 Å. This motivates studying thin, curved contact layers, including
the [nonlinear kernel-map prototype](kernel-transfer-prototype.md). These
selection-biased tail descriptions are not equilibrium layer measurements.

## Protect subdivisions while fitting weights

The previous three-region minimax objective improved aggregates while worsening
some important orthants. A separate [predeclared fit](../runs/contact-guide-stratum-reweighting-20260923/plan.json)
keeps the same 84 component geometries, 50% uniform component and positive
Gaussian weight floor. It protects:

- The three complete reporting classes: native inside historical R5, native
  outside R5, and contact without native entry.
- Every class orthant with at least 1% of its class's estimated **training**
  physical mass.
- Competing orthants 30, 34, 42, 50, 58, 62 and 63, whether or not they meet that
  threshold. All were observed in training; none was silently dropped.

This gives 21 protected groups, frozen before optimization. Training uses only
r00/r01 from each fresh arm, totaling 262,144 attempts. Previously inspected
r02/r03 supply 262,144 fit-holdout attempts; they are not pristine validation.
The fit minimizes the worst protected-group second-moment ratio against the
original bank and starts once from the floored original weights. Neither
holdout results nor the earlier aggregate-only fit enter training.

SLSQP converged in 49 iterations. The normalized training maximum is 0.758275,
and the four added SMC components receive 4.39% of Gaussian mass. The results
on populations held out from fitting are:

| Aggregate region | Fitted/original M₂, bank-source holdouts | Fitted/original M₂, new-guide-source holdouts |
|---|---:|---:|
| Native inside R5 | 0.735 | 0.793 |
| Native outside R5 | 0.766 | 0.488 |
| Contact without native entry | 0.582 | 0.809 |

All three aggregate ratios are below one in each of the four individual
holdout populations. The worst protected-group ratio improves from 1.863 to
1.212 on bank holdouts and from 1.769 to 1.137 on new-guide holdouts, compared
with the previous aggregate-only fit. Remaining regressions involve native-
complement orthant 19 on bank data and competing orthants 30/42 on new-guide data.

A moment ratio above one is a proposal tradeoff, **not an additional physical
convergence gate**. The original plan does not require every candidate to
outperform the baseline in every subdivision. Conversely, improved retrospective
ratios do not establish convergence or open the full-vessel/assembly gates.
This candidate is a basis for considering separately frozen fresh validation;
no operational guide or physical campaign was launched by this diagnostic.

The [full analysis](../runs/contact-guide-stratum-reweighting-20260923/analysis.json)
retains all protected groups, weights, normalization, optimizer status, both
cloud-moment forms, individual-population results and source hashes. The
[tail verification](../runs/contact-tail-geometry-diagnostic-20260923/verification.json)
binds its own plan, script and outputs. Completed physical evidence remains
unchanged, with an unresolved sampling limitation.
