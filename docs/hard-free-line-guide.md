# Gaussian conditioning on the feasible line

This diagnostic preserves the old 92-component Gaussian guide and conditions
one raw translation coordinate only on physical feasibility. It does not select
contact labels or contact widths. The previous [contact-line guide](line-conditioned-gaussian-guide.md)
also required simultaneous close contacts, and the [two-distance/arc guide](contact-arc-score-results.md)
changed the contact-distance distribution. Both could reduce proposal density
at valid target poses. This construction has a stronger pointwise property.

The repaired shape, fixed scaffold, radius 1.5 Å, activity 0.035 Å⁻³, frozen R4
region, center-capture domain and full native classifier remain unchanged. The
Gaussian guide is historically native-informed; this is not template-free
contact discovery. Following the saved-score audit, the production normalizer
also accepts this schema through a separate v6 interface; see the
[physical validation plan](hard-free-line-physical-pilot.md). Assembly kernels
and the main production executable remain unchanged.

## Conditional law and target support

Write the existing raw coordinates as `x=m0+L0 u`. Choose a raw translation axis
`j`, retaining the other five coordinates `v`. As in the validated contact-line
implementation, a full Gaussian draw provides the correct marginal of `v`.
Its remaining conditional coordinate is a Normal distribution `n_k,j(s|v)`.
This retains the Gaussian's translation–rotation correlations and angular
marginal on the full coordinate space.

At fixed `v`, let

\[
I_j(v)=I_{R4}(v)\cap I_{\rm capture}(v)\setminus
\bigcup_{h,i,l}\{s:\lvert p_i(s)-p_{h,l}\rvert<r_i+r_l\}.
\]

The union includes every moving atom and every atom in every physical fixed
neighbor. The two domain chords and each atom-pair exclusion interval are
quadratic. A sphere tree prunes separated pairs, but the leaf inequalities and
interval union define the result. Core exclusions are open; the domain chords
are closed. Isolated points have zero Normal probability.

Define `Z_k,j(v)=∫_I n_k,j(s|v) ds`, with the fixed floor `tau=1e-12`. The new
component density, relative to the same latent measure, is

\[
h_{k,j}(u)=g_k(u)
\begin{cases}
\mathbf1\{s\in I_j(v)\}/Z_{k,j}(v),&Z_{k,j}(v)>\tau,\\
1,&Z_{k,j}(v)\leq\tau.
\end{cases}
\]

Each conditional branch integrates to one. Empty lines and small-mass lines
retain the original `s` at the same `v`; there is no retry of the retained
coordinates. Interval selection uses Normal probability mass, followed by
inverse conditional CDF sampling. Nothing is sampled uniformly in interval
length or clipped to an endpoint.

For the declared primary control, `alpha=0.5` and `beta=1`:

\[
q_j=\tfrac12 U_{R4}+\tfrac12\sum_k\rho_k h_{k,j},\qquad
q_{xyz}=\tfrac13(q_x+q_y+q_z).
\]

On every feasible target pose, its coordinate belongs to each corresponding
`I_j`. Thus each nonfallback term is multiplied by `1/Z>=1`, while a fallback
is unchanged. Consequently **`q_j>=q_old92` and `q_xyz>=q_old92` on physical
target support**. At other poses the density may decrease; at some points
outside R4 it can be zero. There is no global half-old92 support claim for
`beta=1`. Uniform support remains throughout R4.

For a regional physical integrand `f` supported on feasible R4/capture poses,

\[
\int f^2/q_{\rm new}\leq\int f^2/q_{\rm old92}.
\]

The same inequality holds with the unchanged, index-independent Poisson
estimator's conditional second moment replacing `f²`. When the moments exist,
the importance estimator therefore has no greater variance **per attempted
draw**. This does not prove a CPU speedup, converged contact weights, coverage
of undiscovered basins, or a faster Markov chain. All attempted draws and
hard-invalid zeros remain in the estimator denominator.

## Implementation and audit

The diagnostic schema is `defensive-hard-free-line-guide-v1`. It retains the
Gaussian components, `raw_translation_axes`, `conditional_probability`,
`minimum_conditional_mass`, and the defensive uniform probability. It forbids
`contact_widths_A` and `contact_neighbor_indices` to prevent ambiguous controls.
The executable is the existing `contact-line-guide-audit`; its output schema
for this law is `hard-free-line-guide-audit-v1`.

At each pose, each axis's hard-free geometry is shared by every component.
The trace stores the complete interval union; each component's unweighted
Gaussian log density, conditional mean, standard deviation, mass and fallback;
the complete single-axis density; and geometry CPU time. The xyz density must
equal the arithmetic mean of the three complete single-axis densities. Thus a
single xyz scoring run supplies all four diagnostic laws without repeated
geometry. The [frozen scoring allocation](hard-free-line-score-preparation.md)
retains all 206 previously selected critical and breadth queries.

Every attempted query is journaled before work. A failure records its ordinal,
kind, identity and error; completed rows remain on disk. Fresh draws must have
positive finite proposal density. Saved outside-domain queries may have zero
density, represented by a null log density, and must not be discarded. New
draw traces retain the original Gaussian point, axis, interval-selection
uniform and inverse-CDF uniform so the unchanged five coordinates and all
fallback decisions can be checked independently.

Sixteen Rust tests passed: nine existing line controls and seven new controls
covering actual-density normalization over disconnected intervals, pointwise
dominance and axis mixing, retained coordinates and inversion, empty/small-mass
fallbacks, disabled limits and outside-domain zero density, schema separation,
and failure journaling. Independent Python reconstruction and fixed toy draws
must pass before the saved protein scoring run. The analytic argument does not
certify floating-point geometry predicates, interval endpoints or Jacobians;
those remain explicit numerical obligations.
