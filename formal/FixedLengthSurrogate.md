# Fixed-length surrogate chain: formal scope

`ReversibleSampling/FixedLengthSurrogate.lean` proves that any fixed number
`m : ℕ` of applications of the **same** reversible kernel remains reversible.
If that kernel is Markov, the power is Markov and preserves the reference
measure. This result uses arbitrary measurable state spaces and arbitrary
reference measures; it is not a finite-state model. Power zero is included.
The proof first shows that reversing a composition reverses the order of its
two reversible factors. Commuting factors, in particular powers of one
kernel, therefore preserve reversibility. Arbitrary ordered compositions of
different reversible kernels are not covered by that conclusion.

Each inner transition is a completed accept/reject kernel. Both hard rejection
and surrogate MH rejection stay at the current state and consume one of the
fixed `m` steps. A procedure that stops after `m` accepted moves, retries
rejected proposals without counting them, changes the kernel along the path,
or chooses its length from the realized path does not instantiate this proof.

The endpoint theorem `fixed_surrogate_poisson_correct` assumes:

- A measurable state space with reference measure `μ`. Hard support can be
  represented by restricting `μ` to the admissible domain.
- A measurable score `S : X → ℝ` and a **finite surrogate measure**
  `ν = μ.withDensity (exp S)`. Finite mass is an explicit assumption, not a
  claim that a pointwise finite score alone is integrable. The power theorem
  does not need this assumption; the conversion to a symmetric pair measure
  uses it.
- A Markov inner kernel `K` reversible for `ν`, fixed throughout the chain.
- A measurable physical overlap functional `C`, measurable gained/lost
  volumes `v`, positive `λ`, nonnegative activity `z`, and the geometric
  identity `C(y) - C(x) = v(x,y) - v(y,x)`.

It proves that proposing `y` from `K^m(x,dy)` and applying one exact
conditional-Poisson endpoint gate is Markov, reversible, and invariant for
`μ.withDensity (exp(z C))`. This is an unnormalized target; finite nonzero
physical mass is needed separately for a probability distribution.
The proposal may be singular and its endpoints need not be independent.
No density for `K^m` against a coordinate-space measure is assumed.

The gate uses the base factor `exp(-S)`. The separately checked
`surrogate_correction_ratio` proves

`exp(-S(y)) / exp(-S(x)) = exp(S(x) - S(y))`.

Thus the executable endpoint correction is `S_old - S_new`. The existing
conditional-Poisson count-flow theorem supplies the gained/lost count factor,
including zero-volume cases, and `surrogate_target_cancel` removes the
surrogate from the final target. The theorem allows `z = 0` while retaining
its explicit `λ > 0` premise.

For an identity endpoint `y = x`, acceptance and rejection both return `x`.
Skipping the bath at that endpoint has the same physical-state output. This
observation does not identify random-stream consumption or prove equivalence
of an extended state containing a pseudorandom generator.

The following application obligations remain outside this proof:

- Establish the actual handle-Gaussian/Cayley proposal's reference symmetry,
  measurable rigid geometry, inverse and any Jacobian factors; the proof
  assumes inner-kernel reversibility rather than certifying that proposal.
- Show hard-support checks, the surrogate score, fixed proposal parameters,
  rejection accounting, and the endpoint construction implement the assumed
  kernel on the appropriate fixed rigid-body fiber.
- Establish the geometric gained-minus-lost identity and exact Poisson draw
  law, including correct envelopes/thinning and random-stream laws.
- Relate ideal real/ENNReal arithmetic to Rust floating-point arithmetic,
  numerical comparisons, and error or resource-budget paths. No such code
  refinement or statement about a success-conditioned execution is proved.
- Prove irreducibility, ergodicity, mixing, effective sample size, or scientific
  efficiency separately. Reversibility and invariance alone do not prove them.

`FixedLengthSurrogateAudit.lean` audits all 13 proof declarations in this
extension, including its Markov instance. The separate validation receipt
records the pinned dependencies and source hashes. The main theorem import
list, earlier 65-declaration audit, and previous validation receipts are
unchanged.

Reproduce from `formal/` using the pinned toolchain:

```sh
ELAN_HOME="$PWD/.elan" .elan/bin/lake build ReversibleSampling.FixedLengthSurrogate
ELAN_HOME="$PWD/.elan" .elan/bin/lake env lean -j1 FixedLengthSurrogateAudit.lean
```

The focused validation run additionally restricted the build and its children
to one available CPU, retained cached dependencies, and recorded the source
hashes of this module's complete project-local import closure.
