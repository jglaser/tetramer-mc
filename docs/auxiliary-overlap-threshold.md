# A reversible auxiliary overlap threshold

A root-frame guidance cloud can condition the internal dimer edge toward retaining exclusion overlap without changing the physical marginal. The correction is essential: threshold conditioning alone biases equilibrium. This is a design and exact finite-state check, not a protein implementation, a sampled cloud experiment, or evidence of improved acceptance.

Fix the selected root/child labels, spectator anchor, all other poses, shapes, full edge densities, and caps as in [factorized dimer conditioning](factorized-dimer-conditioning.md). Write the two edge coordinates as

\[
h_0=A^{-1}g_0,\qquad h_1=g_0^{-1}g_1,\qquad
F(g_0,g_1)=F_0(h_0)F_1(h_1).
\]

The translation/Haar Jacobian is one. Both densities include every defensive and reciprocal mixture component and the actual prepared-map Jacobian.

## Auxiliary variables and the physical marginal

Let \(P\sim\rho\) be a finite cloud expressed in the rigid root's body frame, with points inside its fixed inflated shape. A Poisson cloud with a fixed guidance intensity is one possible \(\rho\); the intensity need not equal the bath intensity. A fixed cloud, as a point-mass law, also satisfies the algebra. The cloud law must not depend on the current internal pose or on which points happen to overlap.

Define

\[
K_X(P)=\sum_{p\in P}\mathbf1\{p\text{ lies inside the child exclusion shape at }h_1(X)\}.
\]

For a fixed integer \(m\ge1\), introduce \(k\in\{0,\ldots,K_X\}\) with

\[
a_m(k\mid K)=\frac{(k+1)^m-k^m}{(K+1)^m}.
\]

The numerator telescopes, so this is normalized, including \(K=0\), where \(k=0\) has probability one. Its CDF is \(\Pr(k\le j\mid K)=((j+1)/(K+1))^m\). The preferred exact discrete construction is to draw \(m\) independent uniform integers from \(\{0,\ldots,K\}\) and take their maximum. Its CDF is precisely the displayed power, including \(K=0\); it needs an unbiased bounded-integer sampler and valid integer ranges, but no floating-point inverse power. For comparison, \(\lfloor(K+1)U^{1/m}\rfloor\), with \(U\in(0,1)\), has the same law in real arithmetic and introduces avoidable integer-boundary rounding questions in floating point.

The extended target is

\[
\widetilde\pi(dX,dP,k)=\pi(dX)\rho(dP)a_m(k\mid K_X(P)).
\]

Summing over \(k\) and integrating \(P\) gives \(\pi\). A joint auxiliary refresh conditional on \(X\) is exactly: draw \(P\sim\rho\), then draw \(k\sim a_m(\cdot\mid K_X(P))\). Do not redraw a cloud to require \(K_X>0\), condition it on favorable overlap, or replace the probabilistic threshold by \(k=K_X\).

## Fixed-cloud pose block

Hold \((P,k)\) fixed for the entire capped proposal. Keep the root-only filter \(B_0(h_0)\), which tests root spectators and wall while excluding both selected labels. Replace the internal filter by

\[
B_{1,P,k}(h_1)=B_1(h_1)\mathbf1\{K(h_1;P)\ge k\},
\]

where \(B_1\) still requires internal hard validity and exclusion contact. In particular, \(k=0\) does not waive those requirements. Test child spectators and child wall only after assembling both edges, in the final endpoint predicate. A guidance count does not replace hard geometry, the contact predicate, or the physical many-body bath.

Let \(D_{P,k}\) be the complete final domain including the threshold. For fixed context, \(P\), and \(k\), define

\[
p_0=\int F_0B_0,\quad p_1=\int F_1B_{1,P,k},\quad
s_K(p)=\sum_{j=0}^{K-1}(1-p)^j,\quad
a=s_{K_0}(p_0)s_{K_1}(p_1),\quad
d=\int F\mathbf1_{D_{P,k}},\quad q=ad.
\]

Independently cap each edge block, then discard **both** edges after any final failure. At joint cap \(J\), the successful subdensity is

\[
Q_{P,k}(x,dy)=a\,s_J(q)F(y)\mathbf1_{D_{P,k}}(y)\,dy,
\qquad x\in D_{P,k},
\]

with null atom \((1-q)^J\). The same division-free mass identity covers zero caps, zero edge mass and zero final mass. A physical source outside the complete channel domain self-loops; a hard-invalid source is a validation error. The common coefficient cancels between endpoints because it depends on fixed context and \((P,k)\), not on the old selected pose.

For a successful pair, the additional auxiliary target ratio is

\[
\frac{a_m(k\mid K_Y)}{a_m(k\mid K_X)}
=\left(\frac{K_X+1}{K_Y+1}\right)^m.
\]

Thus the one final MH log ratio is

\[
\log F(X)-\log F(Y)
+m\log\frac{K_X+1}{K_Y+1}
+\log R_{\mathrm{physical\ bath}}
+\log R_{\mathrm{selection}},
\]

plus any separately justified bias correction. For the current fixed-label context, the selection ratio is one. The existing exact auxiliary bath gate can supply its count factor; the guidance cloud is a separate auxiliary variable and is not reused as that physical cloud. A physical-gate rejection ends the outer move. There is no resampling until acceptance.

## Why moving the root does not change the normalizer

The count is a function of \(h_1\) and \(P\), not \(h_0\). A new root pose transports the cloud rigidly along with the root in world space while its body-frame coordinates stay unchanged. The root's exclusion shape and its cloud measure remain fixed; the isometry has unit Jacobian. Therefore \(p_1(P,k)\) is independent of the newly selected root. This is the required separation that makes the capped normalizers cancel.

A world-fixed cloud, a child-wall prefilter, or a spectator-dependent internal filter can make \(p_1\) root dependent. Retaining the root while retrying under that changed filter generally introduces an omitted root-dependent factor. Likewise, selecting or changing labels/anchor from contact-dependent lists requires its own reverse selection law. A chirality change or a change of root shape is not covered by the fixed-cloud-measure argument.

For an initial implementation, refresh and discard \((P,k)\) within each outer move. Integrating the fixed-auxiliary accepted-flow equality over \((P,k)\) proves that this collapsed physical kernel is reversible with respect to \(\pi\). It can then be mixed with already valid local/GCA/center-shift kernels without making those kernels handle persistent guidance variables.

If the auxiliary variables are retained as memory, each Gibbs refresh and conditional pose block preserves \(\widetilde\pi\), but their ordered composition need not itself be reversible on the joint state. A rejected pose still retains the correctly refreshed auxiliaries in that composition; reverting auxiliaries only on pose rejection is not this proof. Other physical moves must then preserve or update the auxiliary conditional law, rather than silently ignoring it.

## Exact finite checks and limitations

[test_auxiliary_overlap_threshold.py](../tools/test_auxiliary_overlap_threshold.py) enumerates finite edge/outer tapes using rational arithmetic. It covers two cloud states, root-dependent final feasibility, nonuniform edge laws and physical weights, \(m=1,4\), \(K_X=0\), and all three caps in \(\{0,1,2\}\). It checks null masses, each fixed-auxiliary accepted flow, the collapsed physical kernel, and the distinction between joint invariance and joint reversibility. A separate exhaustive integer-tape test verifies the maximum construction for \(m=1,4\) and \(K=0,1,2,3\), without drawing random numbers.

A two-state counterexample has \(K_0=0,K_1=1\), uniform physical and proposal probabilities, and one raw attempt. Omitting the auxiliary ratio gives equilibrium flows \(1/4\) versus \(1/2^{m+2}\): already \(1/4\) versus \(1/8\) at \(m=1\). The correct ratio restores equality. No overlap-conditioned redraw can be called reversible merely because its hard filters are symmetric.

This proposal does not add an overlap attraction to the physical model. Large \(m\) favors thresholds closer to the source count, while the explicit auxiliary ratio compensates for their source dependence. It may help preserve an existing internal contact during docking, but can also reduce access to a better state or add another proposal penalty. Finite clouds only approximate overlap volume; correctness relies on their explicitly defined count variable, not on that approximation being accurate. CPU cost and final physical acceptance remain open questions.

Before protein use, require identical source/destination frame predicates, strict point-boundary conventions, source eligibility, complete draw/null/error records, fixed caps and guidance law, and tests of the full forward/reverse generator. Numerical errors must be fatal during validation rather than unaccounted retries. The existing scalar capped-factor Lean results can be reused conditionally on \((P,k)\); this note does not formalize the cloud measure, geometric frame transport, or floating-point implementation in Lean.
