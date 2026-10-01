# Retaining the conditioning index

The current physical pilot still evaluates the complete proposal density. This
alternative could remove its three all-component geometry evaluations without
changing which poses are drawn. It is not enabled in that pilot or in assembly.

Write the old guide as `Q = Σ p_b g_b`. The label `b` is either the uniform
branch or a Gaussian component and translation axis. Conditioning replaces
`g_b` by a normalized `h_b`, while retaining the same label probabilities.
The uniform branch is unchanged. The target `F` includes the region, hard
constraints, physical Jacobian and depletion weight.

Split the target over labels using the **old full-mixture responsibilities**:

```
F_b(x) = F(x) p_b g_b(x) / Q(x)
draw density = p_b h_b(x)
weight = F(x) g_b(x) / [Q(x) h_b(x)]
```

Summing `F_b` gives `F` pointwise. For a successfully conditioned line,
`h_b = g_b / Z_b` on its allowed intervals, so its weight is `F Z_b / Q`.
The uniform and empty-line/mass-floor fallback branches have `Z_eff = 1`.
Hard-invalid and exterior attempts still contribute zero to the unconditional
denominator. One must not replace `Q` by the selected component's density.

On valid physical R4 poses each conditioned component satisfies `h_b ≥ g_b`.
Writing `a_b = p_b g_b` and `b_b = p_b h_b`, the three pointwise second-moment
contributions satisfy

```
F² / Σ b_b  ≤  F² / Q² × Σ (a_b² / b_b)  ≤  F² / Q.
 complete             retained index               old guide
```

The left inequality is conditional averaging over the index. The right follows
from `b_b ≥ a_b ≥ 0`. With the same label-independent conditional Poisson law,
replace `F²` by the conditional second moment of the noisy integrand. These
inequalities do not establish finite integrated variance, coverage or speed.

This construction is restricted to the original regional target. It cannot be
copied unchanged into the full-vessel calculation: outside the source domain,
some conditioned components have zero density even though their old Gaussian
responsibilities are positive. Keeping those responsibilities would then lose
target mass. The full-vessel implementation therefore retains complete-mixture
weighting. Any future indexed vessel estimator needs a new partition of the
target that respects each component's support, including the vessel branch.

The [Lean module](../formal/ReversibleSampling/IndexedImportance.lean) checks
the indexed mean, upper second-moment inequality, weighted-variance identity
and lower second-moment inequality for finite labels. Its assumptions are
pointwise real arithmetic, positive branch densities and a positive old mixture.
The previously checked nonnegative importance-integral theorem supplies the
integration framework; identifying these expressions with the actual geometry,
normalized proposal and random streams remains an implementation obligation.
This is not a proof of the Rust executable or of convergence.

The [retrospective report](../results/hard-free-line-index-retrospective-20261001/report.md)
uses only 206 previously scored poses. On its two selected critical sets,
retaining the primitive index increases the estimated second moment by 4.5%
and 15.5% relative to full-density weighting. Those selected sets do not estimate
the complete variance reliably. The proposal-only timing gives a 4.04-fold
zero-replacement-cost ceiling; physical cloud generation and the remaining
old-mixture evaluation reduce any practical gain. No physical sampling speedup
has been measured for this estimator.

Grouping a copy of the uniform branch with each Gaussian/axis gives another
conditional average, with second moment between the primitive and complete
estimators. On the saved critical sets its extra benefit is small. A uniform
draw would need a separately drawn component/axis label with the original
probabilities; a data-selected label would define a different law.
