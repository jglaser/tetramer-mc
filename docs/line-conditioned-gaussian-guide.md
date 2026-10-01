# Line-conditioned Gaussian guide for the fixed-scaffold normalizer

This design changes the integration proposal only. It keeps the repaired shape,
the fixed scaffold, the frozen R4 region, the existing center-capture condition,
the complete native classifier, and the translation/Haar physical measure.
It does not change an assembly kernel. The existing Gaussian guide was fitted
for integration with native-region information; this is not a blind assembly
proposal.

![Recorded protein line, exact interval construction and conditional density](../results/contact-line-guide-figure-20261001/contact-line-guide.png)

The figure uses an actual recorded protein line, with the illustrated Gaussian
component explicitly identified. Independent atom-quadratic reconstruction of
all three widths agrees with the saved interval sets within 1.24e−13 Å of
symmetric-difference length. It is a geometric illustration, not another
sampled event or physical-weight estimate. See the
[reconstruction receipt](../results/contact-line-guide-figure-20261001/interval-reconstruction.json)
and [plotting source](../tools/plot_contact_line_guide.py).

The implementation supports any fixed subset of the three raw chart translation
axes, with equal axis probabilities, and near-core contact widths 0.02, 0.1 and
0.5 Å with equal width probabilities. The first frozen passive diagnostic uses
only raw chart x (axis 0), following the saved-contact geometry and cost review;
it does not evaluate a fresh all-axis arm. Half of the Gaussian proposal remains unchanged.
The uniform R4 defensive probability remains 0.5. A deterministic conditional
mass floor of 1e-12 selects a documented fallback; it is not a physical cutoff.

## Coordinates and the conditional Gaussian

The existing region map uses whitened coordinates u and raw chart coordinates

\[
x=m_0+L_0u=(t_1,t_2,t_3,a_1,a_2,a_3).
\]

The angular coordinates are length-scaled Cayley coordinates. In the current
implementation `Chart::decode` maps them to the proper rotation and supplies
the physical Jacobian

\[
J(u)=\frac{\det L_0}{\ell^3\pi^2(1+|a/\ell|^2)^2}.
\]

Condition a **raw translation coordinate** s=x_j, j∈{0,1,2}, while retaining
v=x_{−j}. Varying a whitened coordinate u_j would generally also vary the
angular coordinates through the region Cholesky factor and would not produce
a straight translation at fixed orientation.

For an existing latent Gaussian g_k=N(μ_k,Σ_k), the raw-coordinate Gaussian has
mean M_k=m_0+L_0μ_k and covariance S_k=L_0Σ_kL_0ᵀ. Its conditional law is

\[
s\mid v,k\sim N\left(M_{k,s}+S_{k,sv}S_{k,vv}^{-1}(v-M_{k,v}),
 S_{k,ss}-S_{k,sv}S_{k,vv}^{-1}S_{k,vs}\right).
\]

Precompute a Cholesky factor after placing the retained five coordinates first
and s last. Its upper-left block generates the retained marginal, its last row
provides the conditional regression, and its last diagonal element is the
conditional standard deviation. This avoids explicitly inverting a covariance
or repeatedly subtracting nearly equal terms in a Schur complement.

An equally valid implementation first draws the complete existing six-dimensional
Gaussian, maps it to x, and then replaces only s. The resulting v already has
the required five-dimensional marginal. On fallback, retaining the original s
preserves its original conditional Gaussian law. No retained coordinate is
redrawn. Translation–rotation correlations are retained in the conditional mean.

This preserves each Gaussian's angular marginal **on all of R6**. It does not
preserve the conditional angular distribution obtained after selecting only
hard-valid R4 endpoints. It also cannot repair an angular direction that the
old guide almost never proposes.

## A common geometric line and its contact intervals

For fixed v, set the selected raw translation coordinate to zero. Let u₀ be its
backmapped latent coordinate and d=L₀⁻¹e_j. Then u(s)=u₀+sd, so the original
R4 condition is the quadratic inequality |u₀+sd|²≤R4². The existing spherical
**center-capture** condition supplies another quadratic interval. This is the
normalizer's existing capture region, not a new atomic-wall condition.

The moving rotation is fixed. Every moving atom follows p_i(s)=p_i(0)+s e,
where e is the selected chart axis rotated to the world frame. For scaffold h,
define

\[
C_h(w;v)=\bigcup_{i\in\mathrm{moving},\,l\in h}
 \{s:|p_i(0)+se-p_l|<r_i+r_l+w\}.
\]

These intervals mean that the **minimum surface gap of the two sphere unions**
is at most w. They are not bands around the older 2 Å native-registration
threshold, and they do not require particular atom pairs or native labels.
The two-scaffold contact set is an intersection of unions, not the union of
pairwise simultaneous contacts chosen in advance.

Let H(v) be the union of core-overlap intervals against **every** physical
fixed neighbor. The selected interval set is

\[
I_{j,w}(v)=I_{R4}(v)\cap I_{\mathrm{capture}}(v)
 \cap C_A(w;v)\cap C_B(w;v)\setminus H(v).
\]

Merge overlapping intervals before computing their masses. Tangencies and
isolated points have zero continuous mass. A final physical hard-validity check
remains authoritative; numerical geometry predicates remain implementation
obligations. A 2r_d width is a useful separate control corresponding to the full
exclusion-contact range. It is not part of the chosen three-width mixture.

For one axis at a queried pose, every Gaussian component shares the same v and
the same physical line. Build its geometry once, not once per Gaussian.
One traversal with the largest width can supply candidate atom pairs for all
three widths and all core exclusions. The width interval sets are nested.
Three axes require three different lines at that pose; they cannot share one
line's interval answer.

## Normalization, the mass floor and the complete density

For the conditional Gaussian n_{k,j}(s|v), define

\[
Z_{k,j,w}(v)=\int_{I_{j,w}(v)}n_{k,j}(s|v)\,ds.
\]

The component's latent density is

\[
h_{k,j,w}(u)=g_k(u)B_{k,j,w}(u),\qquad
B_{k,j,w}(u)=
\begin{cases}
1,&Z_{k,j,w}(v)\le\tau,\\
\mathbf1\{s\in I_{j,w}(v)\}/Z_{k,j,w}(v),&Z_{k,j,w}(v)>\tau,
\end{cases}
\quad\tau=10^{-12}.
\]

An empty set has Z=0. A nonempty Gaussian interval has strictly positive
mathematical mass; invoking the positive floor is an additional deliberate
proposal choice, not evidence that its geometric volume vanished. The floor
depends only on the retained v and the fixed component/axis/width label. The
same decision must appear when evaluating **every** component's density,
regardless of which component generated the sample.

For every v, both conditional branches integrate to one in s. Integrating over
the unchanged five-dimensional Gaussian marginal proves ∫h du=1. On nonempty,
usable lines the new draw lies inside the selected contact set; on fallback
lines it retains the original Gaussian conditional and can lie outside R4.
Thus h is normalized on **all R6**, not globally truncated to R4. Rejecting
fallback endpoints outside R4 and redrawing their retained coordinates would
change this law and require an unavailable outer-marginal normalizer.

With π_k the unchanged normalized Gaussian weights, α=1/2, β=1/2, and B=3|axes|
equally weighted axis/width labels (B=3 in the fixed-x diagnostic; B=9 for xyz),

\[
G(u)=\sum_k\pi_k g_k(u),\qquad
H(u)=\frac1B\sum_{j,w,k}\pi_k h_{k,j,w}(u),
\]

\[
q(u)=\alpha U_{R4}(u)+(1-\alpha)[(1-\beta)G(u)+\beta H(u)].
\]

Consequently q≥(1−β)q_old=0.5q_old everywhere. Any regional second moment can
increase by at most a factor of two due solely to reallocation; this is not a
bound on the physical mass of an unseen region. The density must sum all declared
axis/width labels and all Gaussians, including fallback contributions. It is not the
selected label's conditional density.

The raw-coordinate change introduces no extra factor in h/g: it cancels when
both are expressed relative to du. Equivalently q_u=det(L₀)q_x. The existing
estimator remains J(u)·(W₁+W₂)/(2q_u(u)), with all existing target indicators,
hard-invalid zeros and unconditional attempted-draw denominators. The Poisson
clouds, complete native classifier, and target regions do not enter the guide
selection.

## Stable interval masses and sampling

For every merged interval [a,b], standardize its endpoints with the conditional
mean and standard deviation. Its mass is Φ(b′)−Φ(a′). Use reflected erfc tail
expressions so that a positive-tail interval never subtracts two values rounded
to one. Select an interval proportional to its Gaussian mass, then invert its
conditional CDF by monotone bisection on that finite interval. This is not
uniform sampling in interval length.

The 1e-12 floor prevents attempting severely underflowed normalizers. It does
not by itself guarantee relative accuracy of differences between very close
central endpoints. Test close endpoints independently against high-precision
integration; if necessary, use a controlled small-width Gaussian-integral
series or quadrature for those differences. NaN, negative mass and inconsistent
interval ordering are errors, not ordinary floor fallback. A sampled endpoint
must satisfy the selected interval up to a declared floating-point convention;
silently clipping a draw into the region creates an unintended atom.

Draw and density evaluation must reconstruct the same retained v. Prefer a
single shared raw-coordinate geometry routine and record reconstruction errors.
The final stored density should be evaluated from the actual returned latent
point, as in the existing normalizer. Near-zero interval endpoints, the mass
floor and coordinate round trips require explicit regression probes.

## Required checks and passive diagnostics

Before physical weights are generated, check:

1. Correlated Gaussian reconstruction: latent/raw densities differ by the
   constant determinant, conditional parameters agree with independent Python,
   and conditioning preserves the outer five coordinates and orientation.
2. Interval algebra: overlap, nesting, differences, disconnected sets,
   tangencies, empty R4 projections, capture clipping, and a third physical
   neighbor blocking a two-contact interval. Compare trees against brute atom
   pairs on small asymmetric sphere unions.
3. Normalization at fixed v for two-sphere examples: positive-length intervals,
   exact tangencies, empty geometry and positive-but-below-floor mass. Include
   zero-activity hard-only limits and the existing analytic depletion checks.
4. Complete mixtures: all axes, widths and components; selected versus
   unselected fallback; β=0 reproduces the old Gaussian law and α=1 reproduces
   the old uniform stream; q≥0.5q_old inside and outside R4.
5. Tail precision: both signs, intervals crossing zero, narrow intervals near
   zero and near ±8 or farther standard deviations, and Z immediately above
   and below the predeclared floor. Check finite-interval inversion against
   independent numerical CDFs and retain any failed draw in the audit.

The proposed saved-pose/proposal-only diagnostic should report, separately by
axis and width: empty-line frequency, positive-mass floor fallback, interval
count, conditional mass, true hard-valid fraction, both-scaffold contact rate,
R4/capture retention, complete proposal density and total CPU. Reconstruct all
saved weighted poses under the full mixture to measure retrospective physical
second-moment changes without generating new clouds. Include every old region
and material radial/angular stratum, not only the targeted competing tail.

These diagnostics can reject a costly or ineffective geometry guide before a
new physical allocation. A larger nonempty-line fraction is not a physical
weight or evidence of equilibrium assembly. If outer-five-dimensional coverage
is the limiting factor, exact one-dimensional conditioning will leave that
limitation visible rather than remove it.

## Existing Lean results and remaining obligations

The general importance identity is already checked in
[`normalized_importance_sampling`](../formal/ReversibleSampling/ImportanceSampling.lean#L30).
Use six-dimensional Lebesgue measure, proposal density `g=q`, and target
`f=I_R4 I_capture I_hard I_q J exp(z C)`. Its hypotheses require a measurable,
normalized proposal that is positive and finite wherever the target is nonzero.
The defensive uniform component supplies positivity throughout R4; the positive
mass floor bounds each conditional multiplier by `1/tau`. The conditional
normalization argument in this document supplies the mathematical justification
for `integral q = 1`. **The concrete Gaussian/interval construction has not itself
been formalized in Lean**: its implementation is checked by the independent
[debug and release reference tests](../results/contact-line-guide-validation-20261001/release-validation.json).

For the random depletion estimator, the existing
[`randomized_importance_sampling`](../formal/ReversibleSampling/ImportanceSampling.lean#L60)
adds conditional averaging, and
[`poisson_importance_sampling`](../formal/ReversibleSampling/Poisson.lean#L111)
provides the Poisson generating-function specialization. Deterministic overlap
contributions can enter the base factor; averaging two fresh unbiased clouds
preserves the same expectation by linearity. These results prove expectation
identities under their stated hypotheses, not finite variance, convergence or
coverage of unmeasured regions.

The implementation obligations remain:

- The raw-coordinate map, conditional covariance and Jacobian must agree with
  the stated measures. Geometry sets and the floor decision must depend only
  on the five retained coordinates and fixed component/axis/width labels.
- Exact sphere-union leaf predicates, conservative BVH pruning, interval
  algebra, Gaussian masses and inverse CDFs must implement that conditional
  law. Floating-point execution, including coordinate round trips and boundary
  cases, is not certified by the Lean identities; draws must not be clipped or
  silently retried after a numerical failure.
- The density must include every mixture branch and fallback. The existing
  R4, center-capture, hard-core and complete native-classifier indicators, all
  attempted-draw denominators, and the fresh Poisson estimator remain unchanged.

At `beta=1`, normalization still holds, but some baseline Gaussian tails outside
R4 need no longer be covered. The uniform R4 component still covers the target
integration domain; a global `E_new[q_old/q_new]=1` claim would require additional
support and is not used in that test.

This is an integration proposal, so it does not need to preserve the physical
target as a Markov update by itself. A future MH-corrected assembly kernel could
reuse [`metropolis_hastings_correct`](../formal/ReversibleSampling/MetropolisHastings.lean#L76)
with its complete forward/reverse densities. No new assembly kernel or new Lean
theorem is claimed here.
