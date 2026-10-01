# Hard-free sectors on an existing contact circle

This is a **completed geometry diagnostic, not an implemented assembly move**.
The [results and independent audit](contact-circle-feasibility-results.md)
find positive hard-free sectors on 57/122 localized circles, compared with
21 originally valid endpoints. This supports testing a normalized conditional
azimuth law, not a claim of proposal speedup. The
[two-distance pilot](contact-distance-passive-results.md) showed that selected
atom contacts often clash with the rest of the protein. Before changing the
proposal again, measure whether the same sampled circles have allowed sectors.
Every one of its 240 conditioned circles is retained, without native-label,
validity, component or width filtering. No new pose draws or Poisson clouds are
needed. The original repaired shape and two-scaffold domain remain fixed.

## Geometry and a possible normalized proposal

At fixed orientation, radii and atom-pair labels, write a moving atom's position
as `a + rho*(e1*cos(phi) + e2*sin(phi))`. For a fixed atom at `b`, its hard-overlap
condition is

\[
2\rho[(a-b)\cdot e_1\cos\phi+(a-b)\cdot e_2\sin\phi]
< (R_a+R_b)^2-|a-b|^2-\rho^2.
\]

Each pair therefore excludes an analytic angular interval, possibly empty,
wrapping, or the whole circle. Union the forbidden intervals over every atom
and scaffold; their complement `S` is the hard-free set. Tangent points use the
original strict overlap inequality. Bounding spheres only eliminate pairs
that cannot overlap anywhere on the circle; their overlaps never replace
atomic tests.

Let `p(phi)` be the already normalized uniform/wrapped-Cauchy mixture. Its
allowed mass

\[
Z=\int_S p(\phi)\,d\phi
\]

is a sum of analytic interval probabilities. For a declared positive floor,
use `p(phi)*1_S/Z` when `Z` exceeds that floor. Otherwise retain the original
unconditioned `p(phi)` on the **same** circle. Never redraw the radii,
orientation, component or labels after inspecting `S`. Every conditional
azimuth law then integrates to one; no unknown radial success probability
enters the proposal. An empty circle remains an unconditional zero-weight
attempt. The old Gaussian/uniform defensive mixture would remain present.

Sampling the truncated mixture would not require inverting a sum of CDFs.
First select its uniform or wrapped-Cauchy branch in proportion to that
branch's mass on `S`; then select an interval by its branch-specific mass.
Use a uniform draw inside that interval for the uniform branch, or the
wrapped-Cauchy inverse between its endpoint CDF values for the other branch,
splitting at the periodic cut as necessary. The scored density must still be
the full mixture `p(phi)/Z`, regardless of the selected latent branch.

A complete future implementation must calculate this normalizer for **all**
contributing labels/components/widths at a queried pose, not only its generating
label. Circle geometry is shared across widths for an identical queried pose
and atom-pair label, while the Cauchy law can differ between components. The
physical estimator remains `J*W/q`, with R4/capture failures retained as zeros.
This construction conditions on hard geometry, not on the many-body depletion
weight or the native classifier. It does not repair missing angular support.

## Frozen feasibility question

The [preparation helper](../tools/prepare_contact_circle_feasibility.py) verifies
all audited source hashes and freezes the exact 118 uniform-azimuth and
122 localized-azimuth circles. The cases file SHA256 is
`dc4743a8dc06ea51f86e508cc1a3a631371a844ff2c50c28d3983b5c0236ca85`.
Cases live at
`/vast/xvg/tetramer-mc-runs/contact-circle-feasibility-preparation-20261001`.

Measure empty/nonempty circles, allowed angular length, probability under the
original azimuth law, and BVH traversal CPU/counters, separated by proposal
arm and width. Verify the original hard-valid predicate and deterministic
interior witnesses. An independent Python implementation uses atom KD-tree
candidates and wider-precision leaf geometry to check the Rust sphere BVH.
Neither implementation may use approximate quadrature to determine the
allowed intervals or their probability mass.

A nonempty circle is an opportunity, not a demonstrated sampling improvement:
its allowed sector may leave R4, and scoring all components may cost too much.
The completed diagnostic justifies implementing a normalized guide and an
independent fixed proposal comparison. No physical-weight or assembly
convergence gate is opened by this calculation.

## Execution obligations

Analytic boundaries do not certify floating-point evaluation. Tangencies,
periodic cuts, constant predicates, small interval masses and conservative
tree pruning require explicit checks. The diagnostic records a case identity
before evaluation, preserves completed rows, and stops with a failure record
on geometry errors. It performs no retries or automatic allocation extension.
This journal does not claim power-loss durability. Full simulation-side
failure journaling remains a separate implementation task.
