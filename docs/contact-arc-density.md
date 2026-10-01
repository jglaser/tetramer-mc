# A normalized hard-free contact-circle guide

This is a **candidate density evaluator**, not a new production move. It extends
the [two-distance guide](two-distance-contact-guide-design.md) by conditioning
its azimuth on atomic non-overlap. The existing sampler, physical model and
region classifier are unchanged.

The diagnostic retains the historically native-informed 92-component guide.
Computing the free arcs uses only geometry, but this does not turn its inherited
component locations or labels into a template-free discovery method.

At fixed orientation, ordered atom-pair labels and two sampled contact distances,
the possible translations lie on a circle. Let `p_k(phi)` be the original
component-specific azimuth density, `S` its hard-free arcs against the complete
fixed scaffold, and

\[
Z_k=\int_S p_k(\phi)\,d\phi.
\]

For the frozen floor `epsilon = 1e-12`, replace only this conditional density:

\[
p_k^*(\phi)=
\begin{cases}
p_k(\phi)\mathbf 1_S(\phi)/Z_k,&Z_k>\epsilon,\\
p_k(\phi),&Z_k\leq\epsilon.
\end{cases}
\]

Both branches integrate to one. An empty or extremely small allowed set retains
the original azimuth law **at the same orientation, labels and radii**. Retrying
those outer variables would introduce another success-probability normalizer;
this construction does not retry them. The pre-existing coincident-center and
zero-area radius-polygon fallbacks also retain their original probabilities.

The complete six-dimensional density sums every Gaussian component and every
width. Each translated circle term contains the radius-coordinate Jacobian
`D / (polygon_area * r1 * r2)` and the latent-to-raw chart determinant.
Its angular Gaussian density remains unchanged. For the fixed diagnostic,

\[
q^*=\tfrac12 U_{R4}+\tfrac14 G_{92}+\tfrac14 H_{\rm arc}.
\]

In particular, `q* >= 0.5 q_old92`: the defensive component and old Gaussian
coverage remain present. Hard-free conditioning does not impose the R4 boundary,
capture domain or native classifier along the circle. A later importance draw
would still retain every attempted pose and assign zero physical contribution
to an invalid endpoint, with the complete `J W / q*` weight for valid endpoints.

## Geometry and cost

The [circle geometry implementation](contact-circle-feasibility.md) forms the
union of strict atomic-overlap angular intervals, using the sphere tree only to
prune separated node pairs. Its complement gives `S`; analytic interval masses
give `Z_k` for either uniform or wrapped-Cauchy/uniform azimuth mixtures.

At a queried pose, the same ordered effective contact centers produce the same
circle. Geometry is cached for that exact pose and scaffold, across widths,
components and the two diagnostic arms. **Probability mass is component-specific**:
different Gaussian means and covariances can produce different `Z_k` on shared
arcs. A cache cannot cross query poses.

A selected-circle timing understates the cost of the complete proposal density.
The complete density may need many distinct circles. The score report therefore
records every required geometry, its construction time, and complete mixture
evaluation. The second diagnostic arm reuses geometry built by the first;
comparing their raw timers would not be a standalone-arm speed comparison.

## Validation and reproducibility

Six Rust reference tests cover the two-sphere limit, a partially blocking
spectator, excluded queries, empty-circle and floor fallbacks, shared geometry
with different component masses, disabled conditioning, stale-cache rejection,
and deterministic normalization quadrature with a non-unit chart determinant.
An independent Python implementation reconstructs labels, Schur complements,
circle intervals, component probabilities and the full mixture density.

The dedicated `contact-arc-density-audit` executable reads frozen query poses.
It has no pose-draw or Poisson-cloud path. It archives input and executable
hashes, its source bundle, a begin record for each query, and every completed
score; a failure retains its completed prefix and records the failed identity.
The [fixed saved-pose comparison](contact-arc-score-preparation.md) evaluates
coverage and retrospective second moments before implementing a draw routine.
These diagnostics cannot establish physical convergence or assembly stability.

Analytic normalization is relative to the sphere-union geometry and stated
coordinate map. Floating-point predicates, interval endpoints and Jacobians
remain numerical implementation obligations, checked independently rather than
assumed from the normalization argument.
