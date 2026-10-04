# Posterior context selection with a retained hard-trial cap

The isolated [Lean module](../formal/ReversibleSampling/PosteriorContext.lean)
checks finite-context exact arithmetic. For fixed spectators, let the context
catalogue, nonnegative weights `w_s`, normalized full proposal densities `G_s`,
hard-feasible domain and trial caps `K_s` stay identical in both directions.
Write

\[
Q(x)=\sum_s w_sG_s(x),\qquad
p_s(x)=\frac{w_sG_s(x)}{Q(x)},\qquad
C_s=\sum_{j=0}^{K_s-1}(1-z_s)^j,
\]

where `z_s` is the hard-feasible probability of one independent full draw from
`G_s`. At points with finite positive `Q`, the posterior context probabilities
sum to one. Select the context once and retain it while trying at most `K_s`
independent draws. Only hard infeasibility is retried. On the common feasible
domain, the successful path density is `p_s(x) G_s(y) C_s`. Its integrated mass
is `p_s(x) z_s C_s`; the remaining mass is the all-failed self-loop.

The full-mixture correction is `Q(x)/Q(y)`. For nonnegative target density `π`,
the accepted path flow factors as

\[
\pi(x)p_s(x)G_s(y)C_s\,
\min\!\left(1,\frac{\pi(y)/Q(y)}{\pi(x)/Q(x)}\right)
=w_sG_s(x)G_s(y)C_s\,
\min\!\left(\frac{\pi(x)}{Q(x)},\frac{\pi(y)}{Q(y)}\right).
\]

The right side is symmetric. Summing finitely many contexts therefore preserves
pointwise accepted-flow symmetry. No division by a context weight, context
density or retry factor is needed: individual zero weights/densities and
`K_s=0` are included. When `z_s=0`, the integrated successful mass is zero and
`C_s=K_s`; a density need not vanish pointwise on a null set. Target-zero
endpoints also satisfy the algebraic flow identity. The familiar ratio
`(π(y)/π(x)) (Q(x)/Q(y))` is proved equivalent where `π(x)>0`.
When `Q(x)=0`, posterior selection is undefined as a probability law and the
wrapper must self-loop; the zero-extended scalar candidate path is proved
zero. A context with `w_s G_s(x)=0` likewise contributes no posterior path.
Prior/local context selectors are outside this posterior theorem and must
retain their own source-zero reverse-flow self-loop guard.

The separate `weighted_gate_path_symmetric` theorem accepts an arbitrary gate
`A(x,y)` with the explicit premise
`(π(x)/Q(x)) A(x,y) = (π(y)/Q(y)) A(y,x)`. Multiplying this identity by the shared
`w_s G_s(x) G_s(y) C_s` proves path balance, and a second theorem sums contexts.
For the physical auxiliary count gate, its count law must supply this weighted
identity after the appropriate auxiliary integration. The ideal marginal MH
factorization above is a separate sufficient rule; it is not substituted for
the implemented random gate, nor claimed to prove that gate's count law.

The module reuses `CappedIndependent` for the retry mass identity and bounds.
Its final corollary applies the existing `Balance` rejection completion theorem
only after an actual measurable accepted kernel, its subprobability bound and
integrated balance are supplied. Normalization of the concrete `G_s`, the
common reference measure, unchanged spectators/catalogue, physical hard
predicates, iid complete-mixture draws and the implementation's physical
acceptance law remain premises. A Poisson count gate requires the existing
count-law/overlap-volume obligations; the proof does not replace that gate by
an expectation of a random log acceptance. It proves no Rust, floating-point,
geometric, mixing or performance claim.

The standalone [axiom entry](../formal/PosteriorContextAudit.lean) audits sixteen
theorems. It is separate from the pinned project's existing default library and
audit entry; neither dependency pins nor previous theorem files are changed.
