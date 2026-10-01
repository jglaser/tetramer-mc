# Hard-free guide in the full atomic-wall integral

The vessel wrapper accepts `defensive-hard-free-line-guide-v1` with the same
frozen Gaussian components, chart and source geometry as the regional guide.
The new law is separately recorded as manifest schema 6,
`full-vessel-hard-free-line-half-mixture-v1`. Existing Gaussian and vessel-only
paths retain their schemas, proposal streams and density conventions.

For retained coordinates `v`, let `F_a(v)` be the hard-free interval union for
axis `a`, clipped to the source R4 ball and source center-capture ball. For each
Gaussian component, its conditional Normal is normalized on this union if the
mass exceeds the frozen floor; otherwise it is retained unchanged. Keeping the
original marginal of `v` makes each conditional law normalized over all six
latent coordinates, including fallback draws outside R4. It is not a proposal
conditioned on entering the regional physical domain.

The outer proposal and physical estimator are

\[
p(x)=\tfrac12p_{\mathrm{vessel}}(x)
    +\tfrac12\frac{q_D(u(x))}{J(u(x))},\qquad
\widehat Z=\frac1N\sum_i
  H_{\mathrm{wall}}(x_i)H_{\mathrm{hard}}(x_i)
  \frac{(W_{i,1}+W_{i,2})/2}{p(x_i)}.
\]

Both complete densities are evaluated at every world pose. The physical
Jacobian appears once in the latent-to-world density conversion. No additional
Jacobian or source-region indicator multiplies the physical weight. Invalid
attempts remain zeros in `N`. The full vessel proposal, including all reciprocal
branches, anchors and its cube/Haar floor, supplies support everywhere inside the
atomic wall.

## Distinct geometric roles

For the protein inputs the source capture radius is 170 Å, while the vessel
capture is 273 Å and encloses the atomic wall of radius 223.3261767 Å. The wrapper
constructs conditional intervals using the **source** capture and the bound
fixed-neighbor geometry. Neither that capture nor R4 restricts the vessel target.
The depletant bath continues to permeate the protein wall.

At full conditioning, the latent component can have exact zero density at a
finite exterior pose. The checked scorer establishes this from interval
membership and the component-specific fallback decisions. It does not infer a
support zero from numerical underflow. Every active Gaussian term must have a
representable log density; failures stop the calculation. The exact Cayley seam
has zero guide density, with no deleted finite neighborhood around it.

## Runtime and records

The hard-free branch uses a draw-only method and then scores the world pose once.
The support check reuses the compact interval trace; it performs no second BVH
traversal. The old scored Gaussian draw API remains available.

New records extend `latent_density` with `hard_free_line_density` and
`structural_zero`. Latent-generated records also retain `hard_free_line_draw`.
An attempted-draw journal precedes each attempt. Completed rows are flushed
individually, and handled failures preserve the failed identity and earlier
rows without retry. Completion binds the sample/journal hashes. These are audit
records, not a continuation or power-loss checkpoint protocol.

The independent adapter is
[`tools/physical_hard_free_line_vessel.py`](../tools/physical_hard_free_line_vessel.py).
It reconstructs the complete density and geometry while keeping the physical
wall target separate from source conditioning. Frozen historical auditors and
campaign inputs are unchanged.

## Validation scope

The new wrapper tests cover global conditional normalization, transformed
coordinates, complete axis mixtures, genuine exterior zeros, fallback tails,
near-seam numerical failures and exact disabled-guide RNG compatibility.
`tests/basin_normalizer_hard_free_line.rs` declares new sphere hard-volume and
depletion references over the full atomic wall, two clouds per valid pose,
exterior contributions, and matched legacy/no-guide controls. Its fixed
allocation is recorded in
`results/hard-free-vessel-validation-20261001/allocation.json`; outputs are
preserved and execution must respect the global physical-worker budget.

The frozen toy allocation completed: 16,384 primary sphere attempts and 640
matched control attempts, with no retries. Both hard-volume/depletion integrals
and the normalized-Haar moment agree with their analytic references within the
predeclared test tolerances. The two primary jobs retain 1,661 and 1,743 valid
poses outside both source capture and R4, respectively. Disabled conditioning
and pure-uniform controls reproduce the old Gaussian rows exactly after removing
only the new diagnostics; the matched vessel-only stream preserves poses and
clouds. These small references validate the estimator and support accounting,
not convergence for proteins. The exact CLI, typed-control harness and embedded
source bundle are preserved in
`results/hard-free-vessel-validation-20261001/archive/`. All four independent
Python audits passed, reconstructing every one of the 16,640 new-law attempts,
including invalid zeros and both cloud records. Interval endpoints agree to
1.4e-13 Å and the selected inverse CDF to 8.7e-15. The final receipt is
`results/hard-free-vessel-validation-20261001/validation.json`.

The [numerical-scale follow-up](../results/hard-free-vessel-validation-20261001/numerical-scale.md)
separates component and complete-mixture errors. Its largest component-log
difference is about 4.6e-5 at log density −80 million on a physically invalid
pose. The complete mixture is unchanged there; its maximum log-density error
over either primary reference is 1.03e-14. This follow-up reused the independently
audited intervals and performed no new geometry, sampling or classification.

This registration does not open the regional convergence, full-vessel production
or assembly gates. The old 80-component vessel preparation cannot be reused by
changing its frozen guide in place. A later campaign requires a new preparation,
matching regional evidence, independent full-vessel auditing and explicit
outside-pocket contribution checks. The finite-system assembly verdict remains
unresolved.
