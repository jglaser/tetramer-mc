# Revised guide: retained density and second moment

Keep the normalized uniform law `U`, the old normalized conditional mixture
`C_old`, and add a normalized nonnegative law `C_added`. In exact arithmetic,

\[
q_{\rm old}=\tfrac12 U+\tfrac12 C_{\rm old},\qquad
q_{\rm new}=\tfrac12 U+\tfrac38 C_{\rm old}+\tfrac18 C_{\rm added}.
\]

Then

\[
q_{\rm new}-\tfrac34q_{\rm old}
=\tfrac18(U+C_{\rm added})\geq0.
\]

This retains every old positive-density point. It does not establish that the
old guide covered all scientifically important regions adequately, or that the
added components improve coverage.

## Conditional estimator noise

For either guide, draw `u` from its complete density and auxiliary randomness
from the same conditional law `eta(d noise | u)`. Write the estimator as
`Y_q = J(u) W(u, noise) / q(u)`, with physical validity and observable indicators
included in the numerator as appropriate. Invalid and exterior attempts remain
zeros; there is no replacement or conditioning on success. Define

\[
A(u)=\mathbb E[(J(u)W(u,\mathrm{noise}))^2\mid u],\qquad
M_2(q)=\int \frac{A(u)}{q(u)}\,d\mu(u).
\]

The common nonnegative numerator and density domination give

\[
M_2(q_{\rm new})\leq\tfrac43 M_2(q_{\rm old}).
\]

The hypotheses include support wherever `A` is nonzero, measurability, and a
finite old second moment. Changing the Poisson noise law, including its
intensity, requires establishing a new comparison of the conditional second
moments; having the same unbiased first moment alone is insufficient.

For the same nonzero mean `Z` and positive finite second moments, define the
population moment ESS fraction `e(q)=Z^2/M_2(q)`. Then

\[
e(q_{\rm new})\geq\tfrac34e(q_{\rm old}).
\]

This is a relative retention bound, not an absolute ESS fraction of 0.75.
For independent identically distributed weights with finite second moment,
the usual empirical weight-ESS fraction converges to this population quantity;
that strong-law step is not formalized in the file below. The result does not
bound realized finite-sample ESS, autocorrelation ESS, ESS per CPU, or unseen
mode mass. It also does not give a multiplicative `4/3` bound on variance:
subtracting the common `Z^2` changes that comparison.

## Source audit and implementation obligations

The source audit found no dependence of an existing component's conditional
law or fallback decision on its outer mixture weight:

- [ImportanceGuide::from_bytes](../src/latent_region.rs) normalizes each
  Gaussian weight by the total. Mean, covariance, Cholesky factor and Gaussian
  normalization are computed independently of that weight.
- [ContactLineGuide](../src/latent_region/contact_line.rs) constructs each
  `ConditionalNormal` from its component and the fixed chart. In class mode,
  `class_law` compares the **unweighted** class and hard-free Normal masses to
  `minimum_conditional_mass`. Its class-to-hard-free-to-unconditional fallback
  does not use component mixture probabilities. The hard-free-only path uses
  the same component-local unweighted mass test.
- Class probabilities are separately normalized. Density evaluation sums
  channel multipliers and axis probabilities for each component, then inserts
  the component's mixture weight. The independent Python reference in
  [native_class_line_reference.py](../tools/native_class_line_reference.py)
  has the same separation in `conditional_branch` and `density`.

Consequently, adding Gaussian components and reducing the old bank's total
mass preserves its component laws **provided** the old means/covariances,
region chart, physical geometry, axes, conditional probability, class-channel
probabilities and orthants, compiled native definition, and conditional mass
floor remain unchanged. This includes retaining unconditional fallback mass
outside the target. No renormalization onto observed-valid or selected-class
poses is permitted.

The ideal learned-bank weights are `0.75 * old normalized weights` and
`0.25 * added normalized weights`. The Rust parser subsequently normalizes all
input weights in binary64. Therefore exact retention of `3/4` is an explicit
implementation obligation: writing decimal scaled weights alone is not a
proof of that exact floating-point inequality. If necessary, the exact
coefficient bound for the implemented normalized weights is the minimum of
one and all retained-component new/old weight ratios. This observation does
not certify floating-point density evaluation, inverse-CDF sampling, line
geometry, class topology or the Poisson estimator.

## Isolated formal check

[RevisedGuideSecondMoment.lean](../formal/RevisedGuideSecondMoment.lean) proves:

1. Exact pointwise mixture domination.
2. Pointwise second-moment quotient domination.
3. New integrability and the integral bound from a finite old moment, allowing
   zero proposal density where the numerator is zero.
4. The population moment-ESS retention inequality.

The integral theorem accepts `A` as the shared nonnegative conditional second
moment; it does not construct or validate the physical conditional law. The
four axiom audits contain only `propext`, `Classical.choice` and `Quot.sound`.
The pinned Lean 4.24.0 / mathlib
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7` check used cached dependencies, one
thread and one CPU core, taking 4.584 CPU seconds. It did not rebuild the
project, modify existing formal targets, or perform physical queries.
Two redundant tactic warnings do not affect the checked proofs.

Receipt: `results/revised-guide-second-moment-lean-20261004/validation.json`.
