# Two-leg implicit depletion: checked balance bridge

`ReversibleSampling/TwoLegPoisson.lean` closes the mathematical two-leg
specialization left open by the earlier fixed-length-surrogate theorem. It
does not replace or rerun the existing mathematical or numerical references.
The state space may be uncountable and the proposal may be singular.

The auxiliary trace is

```
(order, (gained₁, lost₁), (gained₂, lost₂)).
```

Reversal changes the order, reverses the legs, and swaps gained/lost counts
on each leg. This is an involution. Conditional on the endpoints and order,
the two count pairs are independent, with each gained count distributed as
Poisson(λ V⁺) and each lost count as Poisson((λ+z) V⁻). The fair order has
probability one half. `twoLegLaw_normalized` proves that the entire trace law
sums to one, including degenerate zero-volume laws.

For the implemented coordinate replacements, the two paths are
`x → (y₀,x₁) → y` and `x → (x₀,y₁) → y`. Reversing the endpoints and order
preserves the copied midpoint. A lost volume on a forward leg is a gained
volume on the opposite reverse leg. The formal volume function explicitly
uses this pairing; applying it to the actual geometry remains an obligation.
For each order, its assumed volume identity is

\[
C(y)-C(x)=(V_1^+-V_1^-)+(V_2^+-V_2^-).
\]

There is no midpoint hard-support factor. The copied midpoint may violate
protein cores or the protein wall. It is used only to define exclusion sets,
not as a physical state accepted by a separate move.

The two Poisson flow identities telescope. `twoLeg_count_factor` and
`twoLeg_gate_term` show that one augmented MH decision is exactly the
implemented count rule, including impossible count pairs and zero volumes.
The factor is

\[
(1+z/\lambda)^{g_1+g_2-l_1-l_2},
\]

represented with natural powers and their ratio, without division by a
volume or count probability. `twoLegFactor_aggregate` proves that adding
the counts equals multiplying the two factors. There is **one** outer
minimum with one, not a product of independently accepted leg moves.

The generic `traceGate_correct` first proves accepted-flow symmetry after
summing any normalized countable trace law with an involutive reverse. It
then applies the existing accepted-flow-plus-rejection theorem. Neither
the bare proposal nor the acceptance function is assumed separately to
preserve the physical target.

Finally, `fixed_surrogate_twoLeg_correct` applies this result to `K^m`, where
`K` is a Markov kernel reversible for the finite measure `exp(S) μ` and `m`
is fixed before the path. Taking the base factor `exp(-S)` yields the
acceptance log factor

\[
(g_1+g_2-l_1-l_2)\log(1+z/\lambda)+S(x)-S(y),
\]

and leaves `exp(z C) μ` invariant. Hard support is encoded in `μ` at the
physical endpoints. Positive λ and nonnegative z are explicit premises;
zero activity and zero changed volumes are included. A finite nonzero
physical normalizer is needed separately to call this a probability law.

The theorem still assumes inner-kernel reversibility. Fair independent
random scan of two reversible coordinate kernels supplies it by linearity;
the concrete Gaussian/proper-rotation proposal and its implementation have
not been formally refined into that premise. The proof also does not certify
geometry predicates, exact thinning, independent random draws, pseudorandom
execution, floating point, interruption/resource caps, mixing or assembly.
The completed independent physical reference and retained-trajectory audits
provide separate finite implementation evidence.

## Validation

The focused build and all **18** declaration audits passed with only
`propext`, `Classical.choice`, and `Quot.sound` allowed. Dependencies retain
Lean 4.24.0 and mathlib commit
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. The build used cached dependencies
and one CPU. The new module's project-local import closure was hashed before
and after the run. Earlier main imports, audits and receipts are unchanged.

Evidence: `two-leg-poisson-validation.json`, `two-leg-poisson-build.log`,
and `two-leg-poisson-audit.txt` in this directory. To reproduce:

```sh
ELAN_HOME="$PWD/.elan" .elan/bin/lake build ReversibleSampling.TwoLegPoisson
ELAN_HOME="$PWD/.elan" .elan/bin/lake env lean -j1 TwoLegPoissonAudit.lean
```
