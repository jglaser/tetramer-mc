# Fixed-context branch priors

`DockingProposal::with_virtual_branch_log_prior` is an opt-in library adapter
for `PosteriorInvolution`. It accepts one normalized, finite log probability
per virtual branch. Every branch must remain positive. The serialized model,
Gaussian factors, reciprocal inversion flags, correlation, defensive uniform
branch and map support are unchanged. Direct-model and static-pair methods
are rejected because their label drawers use different stored distributions.
No production runner or checkpoint format currently enables this option.

The shared vector enters ordinary posterior-source selection, independent
destinations, single-body densities, independent singleton draws, member
densities and member transport. Derived oligomer catalogues obtain the same
vector through `member_chart_parts`. An inverse virtual branch can receive a
different probability from its forward partner; its physical density remains
the exact inverse of the original chart, not a Gaussian at the inverse mean.

## A finite geometric prior

The helper `defensive_virtual_branch_log_prior` implements

\[
 w_j(C)=\epsilon p_j+(1-\epsilon)
        \frac{p_j b_j(C)}{\sum_k p_k b_k(C)}.
\]

Here `p` is a normalized original virtual-branch prior, `0 < epsilon <= 1`,
and `b` is a Boolean center diagnostic. If every or no branch is eligible,
or epsilon is one, the original log vector is returned unchanged. Computation
uses log probabilities, including when a selected probability is too small
to represent in linear arithmetic. Finite RNG precision remains an execution
obligation; a positive mathematical probability is not a guarantee that a
rare branch will occur in a finite run.

The proposed first diagnostic uses epsilon `0.1`, moving label `77`, anchor
`16`, and the saved fixed neighborhood. Eligibility is zero strict core
overlaps against all other bodies at that branch center. Existing per-body
scores can supply this Boolean after removing label 77's contributions.
Every label is retained. The score does not claim a finite-width feasible
volume, account for the wall, or measure depletion energy. The complete
physical endpoint and bath tests remain necessary. This saved neighborhood
was chosen using native diagnostics; this construction alone is not blind
contact discovery or an assembly result.

The finite sum in the denominator is explicitly computable. There is no
unknown configuration-volume normalization and no need to refit charts.
The completed saved-data construction is recorded below. It does not supply a
physical sampling allocation or complete the runner/checkpoint integration.

## Balance and context

Write `g_j` for the unchanged normalized physical-pose chart density and
`G_C(x) = sum_j w_j(C) g_j(x)`. Select the source with probability
`w_i g_i(x)/G_C(x)` and the destination with probability `w_j`. The existing
latent involution and auxiliary Gaussian ratio contribute `g_i(x)/g_j(y)`.
Together with the reversed label probabilities this gives

\[
 \frac{g_i(x)}{g_j(y)}
 \frac{[w_j g_j(y)/G_C(y)]w_i}{[w_i g_i(x)/G_C(x)]w_j}
 =\frac{G_C(x)}{G_C(y)}.
\]

The physical Metropolis ratio includes this proposal correction and the
unchanged physical target ratio (or its validated auxiliary-bath rule).
The separately selected uniform kernel retains its existing correction.
An independent redraw from a *combined* defensive density must instead score
that complete density, as the existing defensive redraw implementations do.

The prior must be identical in the forward and reverse move. Therefore `C`
excludes every moving body and cannot depend on its current pose, selected
noise or proposed endpoint. Fixed anchor/pool labels must be invariant or
have their own explicit selection correction. Geometry-conditioned priors
must be rebuilt after relevant spectators move, or recovered from a cache
whose key proves that the full construction inputs are identical. Stale
context priors cannot be silently described as recomputed priors.

For changed chart parameters, the selected-chart correction in
`context_transport` is still required; the shortcut above is justified here
precisely because the charts generating and scoring `g_j` remain unchanged.

## Required frozen binding before runner integration

A future context-prior manifest and checkpoint binding must include:

- original model and physical-shape hashes, ordered virtual labels and exact
  reciprocal flags;
- fixed spectator pose bytes, excluded moving labels, anchor/pool labels,
  coordinate convention and boundary configuration;
- scoring rule/version, original log priors, every eligibility flag, floor,
  selected mass, fallback status and complete resulting log-prior vector;
- hashes of the saved score source and label-to-body inventory, when reused;
- proposal method, correlation, uniform law and source/binary provenance.

Resume must reject a changed vector or construction context even when the
original model hash is unchanged. The adapter does not infer these facts or
alter existing runner checkpoints; callers must bind them before enabling it.

## Matched evaluation

The smallest useful comparison keeps local step sizes, schedules, map
correlation, covariance widths, uniform component and physical bath identical,
changing only the original prior versus the frozen context prior. Retain all
rejected states and compare contact-environment ESS per total CPU, agreement
between distinct valid initial contact environments, and completed changes
of contact partners. Constant traces or absent exchanges remain unresolved.
Include prior construction cost and report its amortization explicitly.

Increasing a hard-feasibility retry cap alone does not change the accepted
candidate distribution. For independent raw feasibility probability `p`,
a cap `K` yields success `1-(1-p)^K` and expected raw draws
`[1-(1-p)^K]/p`. Improvements then concern setup amortization, not a new
geometric proposal. The cap cancellation requires fixed context and a source
in the same feasible set; it does not justify retrying correlated transports
whose feasibility probability depends on the starting pose.

The saved neighborhood is a conditional test, not an all-mobile assembly
benchmark. Any subsequent all-mobile claim must include context rebuilding
cost and the actual physical parameters, density and boundary conditions.

## Completed frozen prior asset

The reviewed saved-data reduction completed all 2,048 parent labels and 4,098
journal events. After removing moving label 77 from the outside context, 51
centers have no remaining strict core overlap. They carry **3.00447%** of the
original learned prior and **90.30045%** of the new prior. The latter follows
from the declared mixture rule; it is not measured finite-width feasibility,
acceptance, native coverage or equilibrium occupancy. Every branch retains
positive support. An independent direct-probability reconstruction of all
weights agrees within `1.39e-17`.

The [prior asset](../results/context-branch-prior-preparation-20261004-v3/result/prior.json)
has SHA256 `e189bba9838a9b90924ee4e629ff71b5b8fd5381db9edf4652c7a32736b7639e`.
Its [report](../results/context-branch-prior-preparation-20261004-v3/result/report.json)
has SHA256 `8baf53ba1fa54a92750f447bca5a80e9e481b365028278f3525122e596297134`.
The fixed 263-body context and source pose are copied without transformations
from the bound endpoint. **Source hard/wall validity has not been rechecked**;
the saved center scores do not certify that pose. The reduction used 2.682 CPU
seconds including its controller, with no geometry queries, new poses or
physical draws. All process groups drained.

The asset is ready for a bound conditional control after source/domain
certification and runner/checkpoint integration. Existing production runners
do not load it. The original snapshot came from different density/bath
conditions than the assembly decision conditions, and was selected using
native diagnostics. Neither that selection nor the prior reduction supplies
finite-system thermodynamic evidence.

## Validation

The seven adapter tests passed in the isolated release validation at
`results/context-transport-prior-support-validation-20261004/attempt02`,
alongside ten selected-chart tests and eleven basin-involution tests. The
receipt SHA256 is
`91cf5bfed2b7123f6c7a6e2bce6194e3b975ca54880bd671ebde5eff9fef7708`.
Tests cover the normalized support floor and empty/all-eligible limits,
extreme selected log probabilities, invalid methods/inputs, unchanged model
and uniform draws, exact reciprocal branches, and matching source/destination
selection, densities and reversed member corrections.

The retained first attempt failed to compile because this rand version's
`StdRng` does not implement `Clone`. The tests were corrected to instantiate
both streams with the same fixed seed. No production arithmetic changed in
that correction. These checks establish the tested implementation identities;
they do not establish improved contact sampling or assembly.
