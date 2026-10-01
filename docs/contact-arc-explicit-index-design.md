# Explicit component index for hard-free-arc guidance

Status: a derived alternative, not a production sampler. The frozen complete-density scorer and archived-query campaign remain unchanged. This construction becomes worth implementing only if complete arc-density evaluation is too expensive.

## The normalized component laws

Let `u` denote the existing six-dimensional latent pose coordinate, with the existing chart Jacobian included in the physical integrand. The Gaussian weights are fixed positive numbers `ρ_k` summing to one. For each component, define densities on the **whole latent space**:

\[
 q_{o,k}=\tfrac12 U_{R4}+\tfrac14 g_k+\tfrac14 h_{o,k},\qquad
 q_{n,k}=\tfrac12 U_{R4}+\tfrac14 g_k+\tfrac14 h_{n,k}.
\]

Here `h_o,k` is the existing two-distance guide averaged over all three widths, and `h_n,k` changes only its conditional azimuth. The Gaussian angular marginal, selected atom labels, radius-polygon distribution and coordinate Jacobian remain the same. At fixed angle and radii, with allowed azimuth set `A`, old density `p_k(φ)` and `Z_k=∫_A p_k`, the new law is

\[
 p_{n,k}(\phi)=
 \begin{cases}
 p_k(\phi)1_A(\phi)/Z_k,&Z_k>10^{-12},\\
 p_k(\phi),&Z_k\le10^{-12}.
 \end{cases}
\]

The floor is part of the normalized law, including when the allowed set is empty. The existing center-distance and polygon-area fallbacks also remain the original conditional Gaussian; they never redraw the outer angular coordinates. Each `q_o,k` and `q_n,k` therefore integrates to one. The nondegenerate Gaussian term gives every component full support on latent R6; the uniform term protects the fixed integration ball. No proposal is renormalized after rejecting a hard-invalid or out-of-domain pose.

Write `Q_o=Σρ_k q_o,k` and `Q_n=Σρ_k q_n,k`. **`Q_o` is the full old two-distance density, not the Gaussian-only baseline.** Define the inexpensive auxiliary conditional

\[
 \psi_k(u)=\frac{\rho_k q_{o,k}(u)}{Q_o(u)}.
\]

It requires all old component densities but no hard-free-circle geometry. It is exactly normalized; approximate or truncated responsibilities would need their own normalization and would lose the bound below.

## Exact importance sampling and its variance bound

Draw `k∼ρ` and `u∼q_n,k`. For a physical regional integral `I=∫f(u)du`, retain the index and use

\[
 \widehat I=f(u)\frac{\psi_k(u)}{\rho_k q_{n,k}(u)}
 =\frac{f(u)}{Q_o(u)}\frac{q_{o,k}(u)}{q_{n,k}(u)}.
\]

Summing the expectation over `k` gives `I` because `Σψ_k=1`. All attempts belong in the denominator. Hard-invalid and out-of-region states have `f=0`, including attempts using a fallback. The same construction estimates each region separately; it does not turn ratios of noisy masses into unbiased free energies.

At every hard-valid pose, against **all the fixed bodies used in the circle geometry**, each supported component circle contains that pose's azimuth. Thus `h_n,k≥h_o,k`: conditioning multiplies a supported term by `1/Z_k≥1`, while floor and outer fallbacks leave it unchanged. Unsupported width or axis-null terms remain zero. Consequently `q_n,k≥q_o,k` wherever `f≠0`.

The pointwise second-moment multiplier is bounded by

\[
 \frac1{Q_n}
 \;\le\;
 S(u):=\sum_k\frac{\psi_k^2}{\rho_k q_{n,k}}
 =\frac1{Q_o^2}\sum_k\rho_k\frac{q_{o,k}^2}{q_{n,k}}
 \;\le\;\frac1{Q_o}.
\]

The right inequality follows termwise from hard-valid monotonicity. The left inequality is Cauchy–Schwarz; equality requires the optimal conditional index distribution `ψ*_k=ρ_k q_n,k/Q_n`. More precisely,

\[
 Q_n S=1+\chi^2(\psi\,\|\,\psi^*).
\]

Therefore this indexed estimator has no larger variance per attempted draw than sampling from the old marginalized `Q_o`, when the corresponding moments exist. Marginalized `Q_n` remains optimal among these index weights, but requires the expensive full density. The χ² term quantifies exactly what the cheap index costs in statistical efficiency.

For an independent unbiased Poisson estimator `F(u)` of the physical integrand, the proof uses `E[F(u)^2|u]` in place of `f(u)^2`. It still holds if the conditional cloud law is unchanged across indices and comparisons, and is zero on hard-invalid poses. An index-dependent cloud allocation would require a separate second-moment analysis. None of these statements establishes adequate coverage or convergence, and fewer CPU seconds per effective sample does not follow from a per-draw variance bound.

## The corresponding Markov kernel

An alternative use is the auxiliary target `Π(u,k)=π(u)ψ_k(u)`. Refresh `k` from its exact conditional `ψ(u)`, then independently propose `v∼q_n,k` while holding the index fixed. Its acceptance is

\[
 \alpha((u,k),(v,k))=
 \min\!\left[1,
 \frac{\pi(v)\psi_k(v)q_{n,k}(u)}
      {\pi(u)\psi_k(u)q_{n,k}(v)}\right].
\]

Accepted-flow symmetry gives detailed balance for this step, and the index refresh preserves `Π`; summing over `k` recovers the physical marginal `π`. The auxiliary factor must be included once in any existing exact depletion acceptance construction. This note does not implement or validate that composition. The importance-sampling variance bound does **not** imply a Markov-chain mixing-time improvement.

## Work and retained diagnostics

All old component evaluations remain necessary for `Q_o` and `ψ`; these are Gaussian, two-center and clipped-polygon calculations without whole-union arc traversal. The expensive arc density requires just the retained component at a scored pose. Its three widths share the same circle whenever they support that pose; the component law and normalizer must remain component-specific. A new independent proposal can reuse its generated circle in its density. A Markov step generally needs the selected component's circles at both endpoints, with possible cached source reuse.

The expected savings must include old-density arithmetic, proposal generation, geometric cache construction and target evaluation. They cannot be estimated by multiplying selected-circle timing by the number of labels: hard-valid queries always carry an allowed witness and preclude the whole-circle-blocked early exit.

The already saved complete score traces permit a future **deterministic retrospective** diagnostic: reconstruct each `q_o,k` and `q_n,k`, verify both mixture sums, then report `Q_n S` and the old-reference ratio `Q_o S` by the predeclared probe classes, alongside geometry cost. Such probes do not estimate integrated variance by themselves. This design stage draws no additional protein poses or Poisson clouds and applies no selection-dependent filtering.

## Validation

`tools/test_contact_arc_index_design.py` checks the finite-state identities exactly to floating-point tolerance, normalization and second moments with and without Poisson estimator noise, the optimal-responsibility equality, and the necessity of restricting monotonicity to hard-valid support. It also passively reads all 60 deterministic sphere-reference queries and verifies componentwise hard-valid monotonicity, full-circle/empty-circle/floor behavior and that a blocked query can lose conditional density. These tests neither alter the frozen scorer nor rerun geometry.
