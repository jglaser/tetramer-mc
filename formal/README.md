# Measurable-state balance and importance-weight proofs

This subproject proves that an acceptance rule with symmetric accepted flow,
completed by rejection at the current state, preserves its target. It also
constructs that symmetry for the usual Metropolis–Hastings rule, including
asymmetric proposals. The raw proposal is **not** required to preserve the
target. It also proves exact nonnegative importance-weight expectation and
conditional auxiliary-weight marginal identities, the concrete Poisson depletion
mean/variance, and the gained/lost-count acceptance gate. Physical states use
general measurable spaces; auxiliary counts live in the natural numbers.

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
The new Poisson specialization below proves the conditional-mean formula.
Geometric thinning into the declared count law, cover volumes, mixture
normalization, support, and floating-point implementation remain separate
obligations. The theorem proves an expectation identity and does not imply
convergence or accurate estimates from a finite sample.

## Concrete Poisson bridge

`ReversibleSampling/Poisson.lean` proves the generating-function identity
under mathlib's actual `poissonMeasure`, starting from the exponential series.
For finite nonnegative overlap volume `C`, activity `z`, and positive auxiliary
intensity `lam`, it proves

\[
K\sim\operatorname{Pois}(\lambda C),\qquad
E[(1+z/\lambda)^K]=e^{zC},
\]

\[
E[((1+z/\lambda)^K)^2]=e^{2zC+z^2C/\lambda},\qquad
\frac{E[W^2]}{E[W]^2}-1=e^{z^2C/\lambda}-1.
\]

Zero activity and zero overlap are included. The relative-variance statement
is expressed through the proved first and second moments. The concrete
`poisson_auxiliary_weight_marginal` and `poisson_importance_sampling` theorems
instantiate the previous abstract results with this count law and a
nonnegative finite base weight, including hard exclusions, physical region
indicators and coordinate factors. They require the auxiliary kernel to
actually generate `Pois(lam*C(x))`; they do not prove that a geometric
thinning implementation meets that requirement. There is no added
finite-variance claim for the overall pose-importance estimator.

`ReversibleSampling/CountGate.lean` marginalizes a normalized law on two
natural-number counts. Reversal swaps the old/new states and both counts.
It proves pointwise accepted-flow symmetry, then applies the existing
measurable-state theorem to obtain a rejection-completed Markov kernel that
is reversible and invariant. Zero target, proposal and count probabilities
are permitted; an impossible count pair contributes zero flow.

`ReversibleSampling/ConditionalPoisson.lean` specializes this construction.
Let `v(x,y)` be the gained overlap volume and `v(y,x)` the lost volume, with

\[
 C(y)-C(x)=v(x,y)-v(y,x).
\]

The theorem uses the exact forward count law

\[
G\sim\operatorname{Pois}(\lambda v(x,y)),\qquad
L\sim\operatorname{Pois}((\lambda+z)v(y,x)),
\]

with independent counts, and target density
`base(x)*exp(z*C(x))`. It proves that the implemented simplified acceptance

\[
\alpha(x,y;G,L)=\min\!\left\{1,
 \frac{\mathrm{base}(y)q(y,x)}{\mathrm{base}(x)q(x,y)}
 (1+z/\lambda)^{G-L}\right\}
\]

has symmetric accepted flow after averaging over the counts. The proof
first establishes a multiplication identity valid even for zero volumes,
then cancels the positive exponential factor only on nonzero count support.
It never divides by a changed-region volume or requires all count pairs to
have positive probability. The exponential normalization of the reversed
count law cancels the physical depletion ratio exactly. This is an auxiliary
MH construction, not insertion of independent noisy absolute weights into
an ordinary MH ratio.

`conditional_poisson_metropolis_correct` proves the completed kernel
Markov, reversible, and invariant. Its explicit hypotheses are a common
s-finite physical reference measure, measurable finite proposal/base
densities, a Markov proposal with that density, measurable finite gained
volumes, the gained-minus-lost identity, and `lam>0`. No pair-additive
approximation is assumed. An actual exclusion-union geometry must establish
the volume identity and the independent count laws through correct thinning.
The separate zero-activity theorem covers `lam=0` as well. Empty gained and
lost regions reduce to the ordinary base/proposal MH correction.

This density-based specialization does not by itself cover singular
deterministic proposals such as the `c=1` chart map or GCA. The existing
involution theorem remains the appropriate starting point for those maps;
its concrete Gaussian/chart measure-preservation specialization and a
singular-kernel Poisson-gate extension are separate work. These proofs do
not certify numerical geometry, RNGs, mixing, or finite-sample convergence.

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
proofs. Connecting a specific transport, geometric Poisson thinning implementation,
or reversible-jump implementation to the hypotheses remains separate work.

## Internal-geometry subset rates and fixed-duration cluster phases

`ReversibleSampling/ClusterRates.lean` supplies a new **finite-state,
exact-arithmetic** bridge for the collective proposal. It leaves the earlier
general measurable-state theorems unchanged. It is not a formal discretization
of the continuous protein-pose problem or a proof of the Rust program.

For each fixed labelled subset `S`, let `K_S` be the complete physical
accepted/rejected kernel, already reversible for `π`. Its rate `a_S(x)` must
be unchanged on every transition of `K_S` carrying nonzero stationary flow.
A sufficient geometric construction is `a_S(x) = κ_S w(D_S(x))`, where
`D_S` consists only of internal relative geometry and the proposal moves all
members rigidly. The subset remains eligible after attachment to a larger
aggregate: membership is not redefined as the entire physical component.
The proofs establish

\[
R(x,y)=\sum_S a_S(x)K_S(x,y),\quad
\Lambda(x)=\sum_S a_S(x),\quad
\pi(x)R(x,y)=\pi(y)R(y,x).
\]

For a fixed positive finite global bound `M ≥ Λ(x)`, the uniformized kernel

\[
Q(x,y)=R(x,y)/M+[1-\Lambda(x)/M]\mathbf1_{x=y}
\]

is stochastic, reversible, and invariant. Its diagonal includes failed
physical proposals as well as clock thinning. Every power is invariant,
and every **state-independent normalized count mixture** is invariant.
In particular, a fixed algorithmic duration `t` gives

\[
P_t=\sum_{n\ge0}\Pr\{\mathrm{Pois}(Mt)=n\}Q^n,
\qquad \pi P_t=\pi.
\]

`internal_subset_phase_correct` combines these steps. Zero duration and
zero local total rate are covered by this uniformized construction.
`compose_invariant` proves that a phase of fixed algorithmic duration can
be composed with other invariant steps; the ordered hybrid need not be
reversible. A duration selected from CPU time, successful-move count, or
a configuration-dependent stopping criterion is not justified here.

`event_chain_rate_bias` proves a useful contrast: normalizing event selection
at the current state gives `J(x,y)=R(x,y)/Λ(x)`, whose reversible invariant
measure is `π(x)Λ(x)`, assuming positive finite `Λ`. Dropping residence times
therefore generally changes equilibrium. This event chain counts rejected
attempts; an accepted-jump-only chain has a different escape rate again.

A production implementation may skip uniformization null events using
exponential residence times of rate `Λ(x)`, select subsets with probabilities
`a_S(x)/Λ(x)`, and stop at fixed elapsed algorithmic time. Its equivalence to
the Poisson uniformization formula is a standard mathematical construction
but is **not formalized in this file**. Exact exponential races, refreshing
rates after accepted moves, preserving null-event time, and correct handling
of an overshooting final waiting time remain implementation obligations.
The general measurable-state extension below proves the same uniformization
and fixed-time mixture construction without a finite-state restriction.
Neither version establishes ergodicity, mixing efficiency, or physical assembly.

`ReversibleSampling/ClusterMeasureRates.lean` strengthens the rate and
uniformization bridge to **arbitrary measurable state spaces**, including
continuous poses and singular deterministic proposal kernels. For a finite
stationary measure `π` and a finite reversible kernel `K`, it first proves
that stationary pair flow `μ = π(dx)K(x,dy)` is invariant under swapping its
two coordinates. If `a` is measurable, uniformly bounded, and

\[
a(x)=a(y)\quad\text{for }\mu\text{-almost every }(x,y),
\]

then `K.withDensity (fun x _ => a x)` is reversible for the original `π`.
No common Lebesgue or pose-coordinate density is assumed. This covers
accepted subkernels and complete accepted/rejected fixed-subset kernels,
provided their balance and internal-rate invariance have been established.

`measurable_subset_uniformization_correct` adds finitely many such weighted
kernels with normalized rates `b_S = a_S/M` and `Σ_S b_S(x) ≤ 1`, then applies
the existing rejection-completion theorem. It proves that the resulting
continuous-state `Q` is Markov, reversible, and invariant.
`measurable_count_mixture_correct` proves invariance for any state-independent
PMF mixture of invariant Markov kernels; `measurable_poisson_phase_correct`
specializes this to the Poisson mixture of `Q`'s iterates. This directly
covers the fixed-duration uniformized phase on the intended continuous
state space. The earlier `invariant_hybrid` theorem then allows composition
with the other equilibrium kernels. These results do not require finite
physical state cardinality. The separate finite-state file retains the
explicit event-chain `πΛ` calculation and a convenient matrix reference.

The remaining scheduling gap is precisely the mathematical equivalence
between the optimized state-dependent exponential race used in production
and this uniformized Poisson-mixture definition, followed by its numerical
implementation. Neither equivalence nor random-clock floating-point
execution is claimed to be Lean-checked here.

Other implementation obligations are exact contact predicates and subset
enumeration, internal-rate invariance under the actual rigid transform,
fixed-subset proposal balance (including anchor selection, Jacobians, chart
restrictions, auxiliary laws and many-body depletion), finite bounded rates,
RNG laws, and floating-point execution. In particular, cancelling the subset
rate does not cancel the physical Metropolis correction.

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
lake exe cache get Mathlib.Probability.Kernel.Invariance Mathlib.Probability.Kernel.CompProdEqIff Mathlib.Probability.Distributions.Poisson Mathlib.MeasureTheory.Integral.Lebesgue.Countable
lake build
lake env lean Audit.lean
```

`lake update` checks out the pinned dependency; the cache request fetches
only the pinned dependency closure of the named modules. On this machine, the exact
verified local commands are:

```bash
cd /home/xvg/tetramer-mc/formal
ELAN_HOME="$PWD/.elan" .elan/bin/lake build
ELAN_HOME="$PWD/.elan" .elan/bin/lake env lean Audit.lean
```

`python3 check.py` runs both checks, verifies that every printed axiom list is
within the standard trusted set, and writes commands, outputs, and source
SHA-256 hashes to `validation.json`. The current audit contains 65 theorems.
`validation-before-poisson-20260921.json` preserves the preceding successful
19-theorem record. No physical campaign is rerun by this check.

The system CA bundle is `/etc/pki/tls/certs/ca-bundle.crt`. The cache downloader
on this host needed `CURL_CA_BUNDLE` set to that path. Download staging used
`MATHLIB_CACHE_DIR=/tmp/tetramer-formal-mathlib-cache` to limit home-directory
disk use; these settings affect downloads, not proof checking.

## Concrete implementation obligations

| Checked mathematical bridge | Rust implementation | Obligation still outside Lean |
|---|---|---|
| `poisson_depletion_mean`, second moment and relative variance | [`overlap_weight::sample_with_envelope`](../src/overlap_weight.rs) | Envelope coverage, disjoint volume accounting, exact overlap predicates and thinning must give the stated Poisson law. |
| `poisson_importance_sampling` | [`latent_region`](../src/latent_region.rs), [`conditional_ray`](../src/latent_region/conditional_ray.rs) | The implemented full proposal density must be normalized with complete support; the chart/Haar Jacobian and every unconditional zero must be correct. Two independent cloud weights are averaged in linear weight space. |
| `poisson_gate_term_correction`, `conditional_poisson_metropolis_correct` | [`depletion::sample_with_envelope`](../src/depletion.rs), physical acceptance in [`simulation`](../src/simulation.rs) | Lost points must have intensity λ+z and gained points λ, with the gained-minus-lost overlap-volume identity. The base/proposal correction must match the selected kernel. |
| Existing involution and completed-flow theorems | Frozen chart transport and [`spherical`](../src/spherical.rs) kernels | Concrete augmented maps, reference-measure preservation and singular-kernel auxiliary count laws require their own instantiation. The common-density count theorem does not establish this automatically. |

The gained and lost counts are independent because they are thinnings of a
Poisson process into disjoint gained/lost regions, assuming correct geometric
sampling. This geometric-to-count-law implication is an implementation
obligation here, not a hidden additional theorem. Every table row also requires
correct RNG behavior and floating-point execution. Executable reference tests
and independent numerical reconstruction support these obligations but do not
turn them into kernel-checked proofs of the Rust program.

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
