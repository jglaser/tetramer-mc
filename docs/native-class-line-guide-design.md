# Next contact-weight calculation: exact class-conditioned translation lines

This is a design for the next implementation and validation stage, **not an implemented sampler or a committed physical allocation**. No new protein pose, geometry query, native classification, or depletant draw was evaluated for this document. The evolving dimer benchmark and existing growth runs remain unchanged.

The physical target stays the repaired rigid shape, the frozen two-neighbor scaffold, the complete original R4 region, the existing translation/Haar measure, depletant radius **1.5 Å**, and activity **0.035 Å⁻³**. This conditional integral remains distinct from a finite-system assembly decision.

The [completed SMC reconciliation](hard-free-smc-reconciliation.md) supports the principal native masses, while material-stratum checks remain unresolved. The [saved variance decomposition](hard-free-line-population-size-noise.md) identifies competing-contact orthants 22 and 62 as poorly covered: baseline importance ESS 10.5 and 3.8, with largest contributions 28.89% and 49.47%, respectively. Native remainder orthant 55 also remains important to uncertainty calibration. Increasing Poisson intensity alone does not address these pose-coverage limitations.

## Construction and existing implementation

Use the existing hard-free **raw translation coordinate** guide in `src/latent_region/contact_line.rs`, rather than changing the integration region or adding another SMC calculation. At fixed other five raw coordinates, orientation is fixed and translation is affine in the selected scalar. Existing code already supplies Gaussian conditional means and variances, stable Normal interval masses, numerical inverse CDFs, empty-line behavior, all-axis/all-component proposal densities, and physical Jacobians.

`src/line_geometry.rs` supplies endpoint-aware interval unions, intersections and differences, sphere-union hard-overlap intervals, atomic spherical-wall intervals, and contact intervals. The independent implementations in `tools/hard_free_line_reference.py`, `tools/analyze_contact_line_audit.py`, and `tools/physical_hard_free_line_vessel.py` already reconstruct geometry, Gaussian conditioning, and full-vessel physical densities. The earlier `tools/conditional_ray_proposal.py` gives an alternative uniform-ball radial conditional with exact angular marginal and uniform density in radius cubed. Its A7/B4 member shells do **not** implement the complete native-entry predicate and must not be substituted for it.

For each raw translation axis, let H be the existing hard-free line intervals within the unchanged R4 and source-capture constraints. Construct additional interval sets C contained in H. Candidate channels are:

- All exclusion contacts with no complete native entry.
- The same competing-contact class restricted to frozen orthant 22.
- The same competing-contact class restricted to frozen orthant 62.
- Complete native entry in the original native remainder, restricted to orthant 55.

Keep a broad hard-free channel as well as the **50% uniform R4 defensive component**. Channel choices and probabilities must be frozen before fresh proposals; labels select a quadrature proposal, never a replacement physical target. All other classes, all 64 orthants, and the existing radial/angular strata remain reported.

## Exact native-entry line intervals

The authoritative predicate is `NativeEntry::classify_pair` in `src/native_entry.rs`, independently matched by `NativeContactRegions.classify_pair` in `tools/native_contact_regions.py`. A registered motif requires:

1. Body orientation error at most 15°, constant on the line.
2. **All four** body-member position errors at most 2 Å.
3. **At least one**, not every, prescribed supporting member bond. A bond requires monomer orientation error at most 20°, monomer position error at most 3 Å, and at least one shared native residue pair having an atom-pair gap at most 2 Å.

Native entry is the union over **every frozen motif and both fixed anchors**. Body and monomer position bounds produce closed quadratic intervals. For a prescribed monomer bond, form the union of contact intervals over atom pairs whose ordered residue IDs belong to that reference's frozen native residue-pair set. Intersect this union with the bond's monomer displacement interval, provided its constant angular predicate passes. Union the qualifying bonds, then intersect with the motif's four body-member intervals. Finally union all motifs and anchors. Motif and bond caches may share identical geometric subproblems, without removing labels or changing the union.

Call the resulting native set N. The competing set is H intersected with the union of exclusion-contact intervals, minus N. Exclusion contact uses the physical depletant radius; its total pair gap is **2 r_d**, whereas the native reference contact criterion uses a **2 Å total gap**. These must not be confused. Core overlap and exclusion overlap are strict predicates; native contact and registration thresholds are inclusive. Closed/tangent endpoints have zero continuous mass, but their flags must still agree with the reporting classifier.

The native remainder is not merely “outside old R5.” Its existing reporting support is `old_chart_radius <= 5 AND original_q > 1 AND old_capture_valid`; the remainder is native entry minus that support. With fixed orientation, the old-chart coordinates are affine in the translation scalar, so its radius and capture constraints are quadratic. The original q≤1 reference condition is likewise a finite union of member-distance intersections after its constant orientation tests. Preserve this exact Boolean expression and frozen charts.

For orthants, reconstruct the **original whitened latent coordinates** u(s)=u₀+d s. Intersect six affine sign constraints, with zero assigned to the nonnegative bit as before. Changing a raw translation coordinate can change several whitened coordinates through covariance coupling, even though physical orientation is fixed. Never substitute signs of raw translations or independently truncate each coordinate. Radial strata are additional quadratic interval constraints; the existing angular-projection stratum is constant on a fixed-orientation line.

## Normalized law and the fallback guarantee

For Gaussian component g and axis a, let f(s | x₋ₐ) be its ordinary conditional Normal density and let

    Z_H = integral over H of f(s | x₋ₐ) ds
    Z_C = integral over C of f(s | x₋ₐ) ds, with C contained in H.

The existing hard-free law is f_H=f 1_H/Z_H when its mass exceeds the frozen numerical floor, otherwise it uses f. The new class law is f_C=f 1_C/Z_C when Z_C exceeds that same floor; otherwise **fall back to f_H**, without redrawing component, axis, orientation, or any of the other five coordinates. If Z_H is below the floor, both branches use the original f. Numerical failure is an explicit failure, not an empty-set certificate or an occasion to retry.

Every conditional integrates to one; hence each original five-coordinate Gaussian marginal is retained. On a single specified class C, f_C/f_H=Z_H/Z_C≥1 when conditioned, and equals one on fallback. This pointwise density dominance survives averaging over axes and Gaussian components for **that same class**. It is a proposal-density statement, not an ESS or CPU improvement theorem.

A mixture of **different** classes does not in general dominate the old guide on their union: other nonempty class channels may give zero density at a pose in C. Include every channel in q and account for its fixed probability. A broad hard-free channel supplies its own positive lower bound; the 50% uniform R4 component retains coverage independently. Empty-class fallback must be evaluated independently for every axis/component/channel combination, not only the branch that generated the point.

For the latent guide,

    q(u) = 0.5 U_R4(u)
         + 0.5 sum_g w_g phi_g(u) sum_a p_a sum_h p_h R_gah(u),

where R is the conditioned-to-unconditioned Normal ratio, including its hard-free/original fallback. No unknown multidimensional class normalizer is required. Regional weights remain J(u) W(u)/q(u), with both independent Poisson-cloud estimates per valid pose, unconditional attempted-draw denominators, and explicit hard-invalid/exterior zeros. Class labels remain the original independently evaluated reporting predicates.

For a later full-vessel stage, reuse the normalized outer 50/50 vessel/guide mixture and q_physical=q_latent/J exactly once. The source R4/capture constraints restrict only the guide. The original atomic-wall vessel and everything outside measured pockets retain their physical contribution. That stage still requires its existing gates; this proposal does not open them.

## Minimum implementation and checks

Add a native interval method alongside the current native observer, using the same frozen compiled definition. Expose only the interval primitives needed for member/residue constraints; keep the classifier itself unchanged. Extend the line guide with explicitly named class channels, the old-R5 reference binding, fixed channel probabilities, and per-channel interval/mass/fallback traces. Reuse its Gaussian component law and add the new schema as an alternative, preserving old guide formats.

Before protein sampling:

- Compare interval membership with the complete classifier on toy motifs covering **ANY bond**, all-member intersections, wrong angular registration, native residue versus unrelated residue contacts, both anchors, overlapping motifs, and native complements. Include inclusive tangencies, zero-length sets and strict core/exclusion boundaries.
- Independently reconstruct complete unions from leaf atom pairs in Python. Compare pruned versus brute-force interval topology and endpoints, not only classifier values at a few points. Protein development diagnostics must use a frozen saved-pose inventory and preserve every selected query and failure.
- Test orthant zero-bit assignment, correlated raw/latent transforms, old-R5/native-q/capture Boolean complements, radial boundaries, constant predicates and empty lines.
- Verify conditional normalization, full component/axis/channel mixture density, hard-free fallback dominance on a single class, and a counterexample to union-wide dominance for different class channels. Test alpha=1 and disabled class conditioning against the original law, inverse-CDF reconstruction, floor transitions and exact endpoint handling.
- Compare independent Rust/Python q and J. Verify sphere hard-only and analytic depletion limits, all-attempt zero accounting, independent cloud roles, deterministic continuation and fatal-trace draining. Exact geometry means analytic model geometry; floating-point roots and endpoint classification remain explicit implementation obligations.

The expensive new part is residue-restricted contact interval construction. Perform the constant body-angle test and intersect the four member bounds first. Only surviving motif intervals need monomer checks or residue atom-pair searches. Native residue-pair atom lists can be built once from the frozen definition; cache shared member/class tests within a line. Conservative projected bounds may prune pairs, while exact leaf inequalities decide membership. Full mixture q needs all channels, but they can share H, N, exclusion-contact and old-reference sets per axis.

## Proposed staged evaluation, not yet an allocation

First freeze a proposal-only diagnostic that reports class/orthant feasibility, empty-line and numerical-floor fallback rates, conditional masses, full q, and CPU. Preserve all source/axis/channel rows; do not report only successful lines. Saved rows selected from the completed 22/62/55 strata are development probes, not an independent validation population. Include the rest of the domain and then a fresh fixed-size proposal-only check before selecting a physical allocation.

A candidate physical comparison is **eight independent populations of 16,384 unconditional attempts per arm**: the existing hard-free guide versus the new class-balanced guide. Keep alpha=0.5, λ/z=128, two independent clouds per valid pose, and identical target/strata. More populations address the poorly calibrated four-population uncertainty without pretending that repartitioning alone increases pose ESS. This count, channel probabilities, numerical floor, seeds and analysis must be reviewed and frozen before launching; they are suggestions here, not authorization to draw or extend an existing campaign. A larger fresh stage should be specified in that new protocol, not appended after looking at its favorable bins.

Use the existing population-based mass, ESS, largest-contribution, 0.2 kBT/three-SE agreement and native/competing free-energy interval checks. Keep all material strata and historical failures visible, and analyze stages separately. Reuse the completed aggregate SMC evidence without rerunning it. Evaluate improvements by useful effective weighted contact samples per CPU, not native proposal fraction alone.

The principal limitation is unchanged: a translation-line proposal preserves five raw-coordinate marginals and **cannot recover a missing angular mode merely by conditioning its scalar translation**. High class-specific fallback rates would identify that limitation before an expensive physical campaign. Even successful regional convergence still needs full-domain coverage and the prescribed finite-system assembly, initialization, size, and boundary evidence.
