# Factorized finite-cap dimer conditioning

This proposed extension is sound in exact arithmetic when both edge filters depend only on their own edge and a fixed reversible context. It can avoid assembling obviously infeasible pairs, while retaining the complete independent proposal density ratio. It is a design review, not an implementation or a change to the frozen [whole-joint capped screen](capped-dimer-conditioning.md). No proposals or physical jobs were run for this review.

Fix context C: selected labels, every spectator pose, spectator anchor A, shapes, wall, both complete edge densities F₀ and F₁, and finite nonnegative caps K₀, K₁ and J. The context must be identical on the reverse move. In translation/Haar tree coordinates,

\[
h_0=A^{-1}g_0,\qquad h_1=g_0^{-1}g_1,\qquad
f_C(g_0,g_1)=F_0(h_0)F_1(h_1).
\]

The coordinate Jacobian is one. Each F includes the full defensive uniform/learned mixture, every reciprocal branch and the prepared-map Jacobian, regardless of which branch generated the pose.

Define B₀(h₀) as root hard validity against fixed spectators and the root wall. Exclude **both selected labels** from those spectators; the old child cannot enter this filter. Define B₁(h₁) as internal core validity and internal exclusion contact, evaluated with the root at identity. It does not test child spectators or wall, which depend on the newly drawn root. The final predicate T(h₀,h₁) includes child–spectator validity and **child wall validity**, plus complete endpoint revalidation. Then

\[
D_C=\{B_0(h_0)B_1(h_1)T(h_0,h_1)=1\}.
\]

In exact arithmetic this is the intended full hard-valid, internally contacting domain. A hard-invalid source is an error; a physical source outside D_C self-loops in this channel. The old source must satisfy the same preliminary predicates, not merely an approximately equivalent world-space contact test. The channel cannot create contact from a contact-free source without another reversible mechanism.

For each of at most J joint attempts, draw a complete fresh F₀ mixture up to K₀ times, retaining its first B₀ success. Independently do the same with F₁, K₁ and B₁. If either edge cap exhausts, consume this joint attempt and discard its successful partner, if any. Short-circuiting the other edge when the first exhausts is permitted; record it as unattempted. If both edges succeed, assemble the child relative to the **new** root and test T. On failure, discard both edges and start a fresh joint attempt. Root-first or child-first execution has the same successful law; freeze the order and RNG policy before running.

Write

\[
s_K(p)=\sum_{j=0}^{K-1}(1-p)^j,\quad
p_i=\int F_i(h)B_i(h)\,dh,\quad
a=s_{K_0}(p_0)s_{K_1}(p_1),\quad
d=\int f_C(y)\mathbf1_{D_C}(y)\,dy.
\]

Each edge block has successful subdensity s_Kᵢ(pᵢ)FᵢBᵢ and exhaustion atom (1−pᵢ)^Kᵢ. Therefore one joint attempt has successful subdensity a f_C(y)1_D_C(y), with success probability q=ad. Because d≤p₀p₁, q≤[p₀s_K₀(p₀)][p₁s_K₁(p₁)]≤1. After the finite joint cap,

\[
Q^{\rm success}_{C}(x,dy)
=a\,s_J(q)f_C(y)\mathbf1_{D_C}(y)\,dy,
\qquad Q^{\rm exhaustion}_{C}(x,\{x\})=(1-q)^J,
\quad x\in D_C.
\]

The mass identity q s_J(q)+(1−q)^J=1 avoids division and covers zero caps or zero success probabilities. Outside D_C the kernel is identity. Exhaustion is a diagonal atom, not a pose density. On successful endpoint pairs, the common factor cancels:

\[
\log R=\log F_0(h_0^{old})+\log F_1(h_1^{old})
-\log F_0(h_0^{new})-\log F_1(h_1^{new}).
\]

An old zero F density gives zero reverse flow and rejection. Selection probabilities for labels/anchor remain separate; they only cancel when their reverse law establishes that. Caps and filters cannot adapt to an old selected pose, a retained draw, acceptance history or elapsed runtime under this proof.

Multiplying an already balanced physical/count accepted flow by the symmetric factor 1_D_C(x)1_D_C(y)a s_J(q) preserves balance. One final physical gate follows the first complete success; gate rejection ends the outer move. The existing `capped_mass_identity`, `capped_mass_bounds` and `shared_factor_preserves_balance` in [CappedIndependent.lean](../formal/ReversibleSampling/CappedIndependent.lean) cover the scalar steps without a new theorem. Their use does not certify the geometry, SE(3) implementation or floating-point generator.

Keeping a root after final child–spectator failure is a different law: repeated child trials introduce a root-dependent success factor. The displayed full-F ratio then generally omits that factor. In contrast, keeping the root only while executing the single prescribed B₁ block is valid because B₁ and its cap do not depend on the root.

Implementation and validation obligations are:

- Preserve every executed raw edge draw, mixture coin, label/latent, pose, predicate and cap-exhaustion record. Distinguish skipped edges, geometric failure, joint failure and fatal numerical errors. Decoder/scoring errors are not automatically invariant null atoms. Even source-independent failure can remove selected latent preimages and change the surviving density; retrying it needs a separate density proof. A proven extra context-only null with surviving edge subdensity bᵢFᵢ would instead contribute bᵢs_Kᵢ(bᵢpᵢ), but that extension is not assumed here.
- Address frame arithmetic explicitly. A body-frame prefilter on raw h₁ and a world/recovered-coordinate test can disagree near a strict boundary. Define identical source/destination domain predicates and validate raw→world→relative reconstruction. A mismatch must remain visible and stop validation; silently retrying, adding an arbitrary tolerance, or merely checking both frames does not prove the displayed exact proposal density. Conservative early rejection needs certified containment, with ambiguous cases passed to the authoritative endpoint predicate. Full floating-point reversibility remains a separate limitation.
- Test a small finite model exhaustively: candidate probabilities, null masses and accepted flow for nonuniform F and nonunit physical ratios; include zero/full success sets, zero caps, and root-dependent final feasibility. A retained-root retry variant should demonstrate the omitted factor. Exercise both stage orders and verify source-ineligible identity behavior.
- Test atomic tangencies, internal overlap/contact, separate root/child wall failures, fixed spectators and frame round trips. Validate full mixture densities with the current prepared factors. Seed replay must reproduce all stage boundaries and failures; a serialized checkpoint is a separate feature. Freeze caps, stage order and random-stream indexing before any paired comparison.

The [center audit](dimer-mean-clearance.md) motivates removing infeasible within-chart draws before expensive work. This factorization may reduce wasted geometry by combining separately feasible edges. Its success-conditional destination law remains f_C1_D_C/d, exactly the same as whole-joint conditioning for the same D_C. It does not by itself change destination quality, establish a favorable MH/bath factor, or resolve the retained physical stationarity discrepancy. Any efficiency claim needs raw-edge, geometry, CPU and final-gate accounting from a separately frozen test.
