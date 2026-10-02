# Exact finite-state check of factorized dimer conditioning

The proposed [factorized sampler](factorized-dimer-conditioning.md) passes an
independent exact-rational enumeration. A deliberately different sampler that
keeps the root after a failed final joint check fails detailed balance when it
uses the same full-product guide correction. This distinguishes the required
retry rule from an apparently similar biased implementation.

The standalone [test](../tools/test_factorized_dimer_balance.py) uses Python
`Fraction`, exact equality and no RNG. It enumerates finite raw edge tapes up to
the cap, integrates their unused suffixes, then enumerates outer trial tapes
after exactly summing the edge histories. It does not generate protein poses,
call a physical bath or measure mixing.

Two nonuniform pairs of three-state edge distributions are tested, including
zero guide weights. Preliminary masks restrict each edge separately; the final
mask depends on both root and child. Every combination of edge caps K0,K1 and
joint cap J in {0,1,2,3} is tested, for both stage orders and with/without
short-circuiting the unused second stage after first-stage exhaustion. This
checks **512 joint proposal laws**. Three physical targets per law—including
nonuniform positive weights, an interior zero weight, and zero weight outside
the conditioned domain—give **1,536 complete physical kernels**.

For each kernel, every row sums to one, every directed accepted flow matches its
reverse, and summing one step preserves the complete target. The checks retain
all exhaustion and MH rejection mass and use the original full F0*F1 correction.
They also verify identity steps outside the conditioning domain. Additional
controls cover 32 preliminary edge laws, 108 empty/full-region cap combinations,
and the fact that a fixed short-circuit policy changes trace consumption without
changing the successful endpoint law.

For p0,p1 preliminary success masses, d the final mass under F0*F1, and
s_K(p)=sum_{j<K}(1-p)^j, the independently enumerated masses equal

\[
a=s_{K_0}(p_0)s_{K_1}(p_1),\qquad q=ad,\qquad
q_{success}(y)=a s_J(q)F_0(y_0)F_1(y_1)\mathbf1_D(y),
\quad q_{null}=(1-q)^J.
\]

This includes zero caps and zero/full feasible mass without division by p0, p1
or d. The common prefactor cancels only for a fixed reversible context and the
specified independent retry rule. It does not remove anchor-selection or
physical acceptance factors.

## Retaining the root is a measurable counterexample

Let F0=(1/2,1/2), F1=(1/3,2/3), and allow final pairs
D={(0,0),(1,0),(1,1)}. Draw the root once, then try the child twice while retaining
the root after failure. The resulting masses are:

| Outcome | Proposal mass |
|---|---:|
| (0,0) | 5/18 |
| (1,0) | 1/6 |
| (1,1) | 1/3 |
| Exhaustion | 2/9 |

The full-product guide has equal density 1/6 at (0,0) and (1,0), but the actual
proposal does not. For a target uniform on all four states, with the excluded
state receiving the specified identity step, the accepted flows are **1/24**
from (0,0) to (1,0) and **5/72** in reverse. One step changes the mass at (0,0)
from 1/4 to **11/36**. Row sums remain exactly one; stochasticity alone therefore
does not establish invariance. Replacing the guide correction with the actual
altered proposal ratio restores balance in this toy. In the protein geometry,
the missing root-dependent normalization is generally unknown.

## Focused Lean bridge

The new [FactorizedIndependent.lean](../formal/ReversibleSampling/FactorizedIndependent.lean)
imports the existing capped-retry results and checks **11 scalar theorems**:
nonnegative edge and endpoint factors; q in [0,1] from 0<=d<=p0*p1; complete
successful/exhaustion mass and its bounds; zero root, child and joint caps;
zero final feasible mass; and preservation of deterministic or already
count-marginalized accepted-flow symmetry by the common factor.

The [focused axiom audit](../formal/FactorizedIndependentAudit.lean) contains only
`propext`, `Classical.choice` and `Quot.sound`. Dependency pins are unchanged.
The final per-file checks explicitly used `lean -j2`, reusing the existing
compiled pinned dependencies. No main library imports or prior proofs were
modified. The [formal receipt](../formal/factorized-independent-validation.json)
binds sources, commands, dependency hashes and final logs.

The five finite-model tests pass. Their [receipt](../results/factorized-dimer-balance-20261002/finite-validation.json)
and [log](../results/factorized-dimer-balance-20261002/finite-tests.log) are separate
from the [Lean axiom output](../results/factorized-dimer-balance-20261002/lean-final-axioms.log).
Run the finite check with:

```sh
/home/xvg/protein-nucleation/.venv/bin/python tools/test_factorized_dimer_balance.py
```

The finite enumeration verifies these selected matrices; the Lean results prove
the stated scalar identities under explicit premises. Neither instantiates the
SE(3) measure, body/world geometric equivalence, actual code's stopping history,
Poisson thinning, RNG independence or floating-point execution. Those remain
implementation and reference-test obligations. The planned J=1, K0=K1=32
benchmark is not a new physical validation or an assembly result.
