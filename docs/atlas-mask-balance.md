# Controlled active subsets of the learned atlas

The active-mask extension tests whether fewer simultaneously available Gaussian
components preserve the successful atlas's native proposal coverage. It retains
the original reference centers, rotational charts, covariances and weights, and
the full continuous atlas-transport latent vector. It therefore compresses the
active proposal, not stored parameters, and does not discover new basins.

## A count law with an interpretable activity

For a reference dictionary of size `J`, let `A` be a sorted subset of labels and
`K=|A|`. With configured bounds `K_min <= K <= K_max`, define

\[
 P(K=k)=\frac{z^k/k!}{\sum_{h=K_{\min}}^{K_{\max}}z^h/h!}.
\]

The positive finite activity `z` is the untruncated Poisson parameter; it is not
necessarily the mean count after truncation. Setting equal lower and upper
bounds fixes the count. The optional upper bound defaults to `J`.

At each fixed count, the conditional subset probability is

\[
 P(A\mid K=k)=\frac{\prod_{j\in A}b_j}{e_k(b_1,\ldots,b_J)},
 \qquad
 e_k(b)=\sum_{|B|=k}\prod_{j\in B}b_j.
\]

The `uniform` label law has `b_j=1`; the `atlas_weight` law uses the positive
reference weights. These are immutable weights, not the current fluctuating
mixture weights. Rescaling all `b_j` leaves every conditional law unchanged.
The cardinality normalizer `e_k` is essential: it ensures that changing label
bias does not change the specified count distribution through combinatorial
multiplicity. This differs from independent Bernoulli activations.

Suffix elementary symmetric polynomials obey

\[
 e_k(b_i,\ldots,b_J)=e_k(b_{i+1},\ldots,b_J)
       +b_i e_{k-1}(b_{i+1},\ldots,b_J).
\]

The engine computes this table in log space in `O(J^2)` time and storage. First
it draws `K`, then scans labels. With `r` labels still needed, label `i` is
included with probability

\[
 \frac{b_i e_{r-1}(b_{i+1},\ldots,b_J)}
      {e_r(b_i,\ldots,b_J)}.
\]

The resulting subset draw is exact up to floating-point arithmetic, with no
rejection, subset enumeration, or estimated normalizer. The zero-count subset
is empty and the full-count subset contains all labels. Fixed empty/full count
draws consume no random variates.

## Physical marginal and balance

Let `eta` be the retained full atlas-transport latent state. The joint target is

\[
 \widetilde\pi(X,A,\eta)=\pi(X)\rho(A)\phi(\eta),
 \qquad \rho(A)=P(K=|A|)P(A\mid K).
\]

Both auxiliary factors are normalized and independent of physical coordinates,
so summing over subsets and integrating `eta` recovers exactly `pi(X)`.
Current geometry still determines the reproducible atlas fit `F(X)`. The
physical proposal retains `A,eta`, selects its usual spectator anchor, and
uses only the active components of the decoded model. Their positive weights
are renormalized within the subset. The usual uniform defense remains present;
the empty mask gives the uniform-only proposal.

For a proposal from `X` to `Y`, the proposal part of the physical acceptance
ratio is

\[
 \log q_{Y,A,\eta}(X\mid\text{anchor})
 -\log q_{X,A,\eta}(Y\mid\text{anchor}).
\]

The reverse model must use `F(Y)` with the same retained mask and latent state.
The reference subset law cancels exactly. No mask, fit-score, sparsity penalty,
or count factor enters the physical energy. The existing exact conditional
Poisson gate supplies the physical depletion acceptance factor.

Local moves, model-independent GCA, and common center shifts preserve this
joint target without additional mask corrections. Independently redrawing the
mask from `rho` is a Gibbs update. The refresh scheduling probability is a
constant configuration parameter and is applied by the runner; the engine's
`refresh` method always performs the exact redraw. Independent normal refreshes
of the retained atlas latent state remain Gibbs updates.

The optional `initial_full` preparation starts with all labels only when that
count is allowed; otherwise initialization samples the prescribed law. A
prepared full mask need not be a stationary auxiliary draw. It should survive
the first physical sweep, with the first scheduled refresh after that sweep.
Checkpoints must retain the actual labels and full latent state. Changing
auxiliary initialization does not change the invariant physical marginal.

## Interpretation and controls

Subset occupancy is an algorithmic auxiliary statistic, not an equilibrium
contact probability. A removed component can matter because it generates a
useful candidate, because it supplies reverse probability at the old pose, or
both. Native labels belong in held-out diagnostics, not this subset law.

In particular, sparse masks can be balanced yet slow. For two equally probable
physical basins, components `(0.99,0.01)` and `(0.01,0.99)` form an equal mixture
that crosses basins with probability `0.5`. Selecting one component uniformly
and retaining its mask during MH instead crosses with probability `0.01`:
the missing reverse coverage makes it 50 times slower. Merely retaining a
component that proposes native candidates is insufficient.

The first comparison should preserve all other physical settings and examine
forward and reverse proposal density, native candidate counts, acceptance,
and bidirectional binding-environment transitions per CPU time. Fixed-full
support must recover the complete-atlas kernel. Fixed-zero support supplies
the uniform-only endpoint. Variable-count controls should compare the same
Poisson count law with uniform and atlas-weight label selection.

The core tests enumerate small dictionaries to verify normalization at every
cardinality, compare exact weighted-subset probabilities and empirical draws,
check count-law independence from label bias, exercise endpoint and malformed
states, and demonstrate the reverse-coverage counterexample. The runner tests
replay both densities, verify all count endpoints, reproduce the full-atlas
control, and check exact checkpoint continuation. A separate equilibrium test
starts 5,000 independent AO sphere configurations and composes three rounds of
masked transport, local moves, GCA and center shifts. Physical and auxiliary
cross moments remain consistent; finite-state negative controls detect using
the wrong reverse mask. The [protein pilot](atlas-mask-pilot.md) independently
audits the actual transported densities and retained subsets.
