# Measurable-state balance and importance-weight proofs

This subproject proves that an acceptance rule with symmetric accepted flow,
completed by rejection at the current state, preserves its target. It also
constructs that symmetry for the usual Metropolis–Hastings rule, including
asymmetric proposals. The raw proposal is **not** required to preserve the
target. It also proves exact nonnegative importance-weight expectation and
conditional auxiliary-weight marginal identities. All results use general
measurable spaces, not a finite-state model.

## Proven statements

`ReversibleSampling/Balance.lean` starts with any measurable kernel
`A(x,dy)` of mass at most one and defines

\[
 K(x,dy)=A(x,dy)+[1-A(x,X)]\delta_x(dy).
\]

`complete_markov` proves normalization. `complete_reversible` proves detailed
balance when `π(dx) A(x,dy)` is symmetric, using the fact that the rejection
flow lies on the diagonal. `complete_invariant` concludes `πK=π`.

`accept_reject_correct` specializes this construction to a Markov proposal
`Q` and jointly measurable acceptance `α` valued in `[0,1]`, with
`A(x,dy)=Q(x,dy) α(x,y)`. Symmetry is expressed by equality of accepted flows
between every pair of measurable sets. Neither countability nor a Euclidean
coordinate system is assumed. The target may be an unnormalized measure;
normalization is needed separately if it is to be called a probability law.

`ReversibleSampling/MetropolisHastings.lean` discharges the symmetry premise
for a concrete acceptance rule. It assumes an s-finite reference measure `μ`,
measurable finite nonnegative densities `p(x)` and `q(x,y)`, and a Markov
proposal satisfying `Q(x,dy)=q(x,y) μ(dy)`. Then

\[
 \alpha(x,y)=\min\left(1,
 \frac{p(y)q(y,x)}{p(x)q(x,y)}\right),\qquad
 p(x)q(x,y)\alpha(x,y)
 =\min\{p(x)q(x,y),p(y)q(y,x)\}.
\]

`mh_accepted_flow_symmetric` proves the resulting flow symmetry using
Tonelli's theorem, and `metropolis_hastings_correct` proves the completed
kernel Markov, reversible, and invariant for `μ.withDensity p`. Densities
may be zero, including hard exclusions. The zero-forward-density acceptance
uses Lean's extended-nonnegative-real division convention; it cannot change
accepted stationary flow. Infinite densities are excluded by explicit
hypotheses. Singular deterministic proposals are covered by the separate
involution result below, without pretending that a Dirac proposal has an
ordinary density with respect to the physical continuous reference measure.

`ReversibleSampling/Involution.lean` proves balance for a measurable map `T`
that is an involution (`T(T(x))=x`) and preserves a reference measure `μ`.
For finite measurable nonnegative `p`, its deterministic proposal and
acceptance are

\[
 Q(x,dy)=\delta_{T(x)}(dy),\qquad
 \alpha(x,T(x))=\min\{1,p(T(x))/p(x)\}.
\]

`deterministic_accepted_flow_density` identifies accepted flow from measurable
sets `S` to `B` as

\[
 \int_{S\cap T^{-1}(B)}\min\{p(x),p(T(x))\}\,\mu(dx).
\]

`involution_accepted_flow_symmetric` proves symmetry by changing variables
with `T`, using both involutivity and reference-measure preservation.
`involution_metropolis_correct` then proves the completed kernel Markov,
reversible, and invariant for `μ.withDensity p`. This theorem needs no
s-finiteness assumption and allows zero target density and fixed points of
the map.

The assumption `MeasurePreserving T μ μ` is explicit. An arbitrary invertible
map does not satisfy it automatically and need not have unit Jacobian. The
result applies on an augmented state space, but proving that our particular
Gaussian-coordinate, chart, or Poisson-cloud map is measurable, involutive,
and reference-measure-preserving remains separate work. It does not silently
absorb omitted Jacobian or auxiliary-law factors into the acceptance rule.

`ReversibleSampling/ImportanceSampling.lean` proves

\[
 \int \frac{f(x)}{g(x)}\,(\mu.withDensity\ g)(dx)
 =\int f(x)\,\mu(dx)
\]

for measurable `f,g : X → ℝ≥0∞`, assuming that, for `μ`-almost every `x`,
`f(x) ≠ 0` implies `g(x) ≠ 0` and `g(x) ≠ ∞`.
`importance_sampling_lintegral` needs neither s-finiteness nor normalization.
Zero target values and zero proposal values outside target support are allowed.
`normalized_importance_sampling` additionally assumes `∫g dμ = 1` and proves
that `μ.withDensity g` is a probability law with that exact expectation.
The equality is in the extended nonnegative reals; a finite target integral is
needed for a finite first moment. No finite variance or sampling convergence
follows. Applying this result to a geometric proposal still requires proving
its actual density, normalization, support, and any coordinate/Jacobian factors.

`unbiased_auxiliary_weight_marginal` assumes an s-finite base measure `μ`, a
normalized measurable auxiliary kernel `η`, and a jointly measurable
nonnegative weight `w(x,u)` satisfying `∫w(x,u) η(x,du) = f(x)` for `μ`-almost
every `x`. It proves

\[
 \bigl[(\mu(dx)\,\eta(x,du))\,w(x,u)\bigr].fst
 =\mu.withDensity\ f.
\]

This is the marginal identity for an unbiased nonnegative auxiliary weight.
Correct conditional mean is an explicit premise, not a proved property of a
particular Poisson or geometric estimator. Preserving this joint law remains
an obligation for any proposed transition, covered abstractly by the balance
results above. Strict positivity of the weight is unnecessary for this
marginal identity; nonnegativity is built into its type.

`randomized_importance_sampling` combines conditional averaging and importance
weighting: with the same normalized `g` and support condition, and
`E_η[w(x,U)] = f(x)` almost everywhere, it proves

\[
 \int \frac{w(x,u)}{g(x)}\,
       (\mu.withDensity\ g)(dx)\,\eta(x,du)
 =\int f(x)\,\mu(dx).
\]

For the intended positive Poisson estimator, take `w(x,u)=h(x) W̄(x,u)`
and `f(x)=h(x) exp[z C(x)]` after separately establishing the conditional
mean `E[W̄|x]=exp[z C(x)]`. For the nested cover mixture, the proposal is the
full marginal density `g(x)=Σ_i α_i 1_(C_i)(x)/V_i`; this complete sum belongs
in the denominator. These substitutions connect the two abstract steps.
The concrete Poisson conditional-mean formula, geometric cover volumes,
mixture normalization, support, and floating-point implementation are still
outside this Lean proof. The theorem proves an expectation identity and
does not imply convergence or accurate estimates from a finite sample.

Additional results establish:

- preservation of a specified physical marginal by an invariant joint kernel;
- the physical marginal of `π × ρ` when the auxiliary reservoir `ρ` is normalized;
- the physical marginal of `π(dx) η(x,dθ)` for any normalized, measurable,
  configuration-dependent auxiliary kernel, and its preservation by a joint
  invariant update;
- invariance under composition of two kernels preserving the same joint target.

The ordered composition need not itself be reversible. A projected physical
trajectory need not be Markov. An arbitrary online fit is not licensed by the
marginal theorem: its update must preserve the declared joint law. In particular,
retuning the defining conditional kernel `η` during the chain changes that law
and is not justified by this theorem. A variable component count can be part of
the auxiliary measurable space, but no particular reversible-jump birth/death
rule is verified here. No claim
of ergodicity, mixing speed, convergence from a chosen initialization, or
correctness of the Rust floating-point implementation follows from these
proofs. Connecting a specific transport, Poisson estimator, or reversible-jump
implementation to the hypotheses remains separate work.

## Pinned environment and reproducible check

- Lean `v4.24.0`, compiler commit
  `797c613eb9b6d4ec95db23e3e00af9ac6657f24b`.
- mathlib commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`
  (release `v4.24.0`).
- Transitive revisions are recorded in `lake-manifest.json`.
- The local optional Elan installation is `v4.1.2`; its artifacts and the
  toolchain/cache are ignored by Git. No system toolchain was modified.

With an existing Lean/Elan installation, run from this directory:

```bash
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
lake exe cache get Mathlib.Probability.Kernel.Invariance Mathlib.Probability.Kernel.CompProdEqIff
lake build
lake env lean Audit.lean
```

`lake update` checks out the pinned dependency; the first cache request fetches
only the 2,425 mathlib dependency files needed here. On this machine, the exact
verified local commands are:

```bash
cd /home/xvg/tetramer-mc/formal
ELAN_HOME="$PWD/.elan" .elan/bin/lake build
ELAN_HOME="$PWD/.elan" .elan/bin/lake env lean Audit.lean
```

`python3 check.py` runs both checks, verifies that every printed axiom list is
within the standard trusted set, and writes commands, outputs, and source
SHA-256 hashes to `validation.json`.

The system CA bundle is `/etc/pki/tls/certs/ca-bundle.crt`. The cache downloader
on this host needed `CURL_CA_BUNDLE` set to that path. Download staging used
`MATHLIB_CACHE_DIR=/tmp/tetramer-formal-mathlib-cache` to limit home-directory
disk use; these settings affect downloads, not proof checking.

## Trust boundary and sources

`Audit.lean` prints transitive axiom dependencies for the main theorems. The
successful audit contains only the standard Lean axioms `propext`,
`Classical.choice`, and `Quot.sound`; no `sorryAx` or project-defined axiom
appears. Proof terms are checked by Lean's kernel. The trust boundary includes
the Lean implementation and the pinned dependency artifacts. Definitions use
classical/noncomputable mathematics; this is not an executable sampler.

The proof uses mathlib's definitions of
[`Kernel.IsReversible` and `Kernel.Invariant`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/Invariance.html),
[`Kernel.withDensity`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/WithDensity.html),
and its measure-theoretic Tonelli theorem. The version-pinned local sources
under `.lake/packages/mathlib/Mathlib/` are authoritative for this build;
the linked generated documentation may track a newer revision.
