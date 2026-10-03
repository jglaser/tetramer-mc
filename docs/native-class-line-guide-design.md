# Next contact-weight calculation: exact class-conditioned translation lines

The translation-line implementation is under validation in `src/native_entry/line.rs` and `src/latent_region/contact_line.rs`, with an independent Python reference in `tools/native_class_line_reference.py`. It is **not yet validated on proteins or allocated a physical-weight campaign**. The evolving dimer benchmark and existing growth runs remain unchanged.

The physical target stays the repaired rigid shape, the frozen two-neighbor scaffold, the complete original R4 region, the existing translation/Haar measure, depletant radius **1.5 Å**, and activity **0.035 Å⁻³**. This conditional integral remains distinct from a finite-system assembly decision.

The [completed SMC reconciliation](hard-free-smc-reconciliation.md) supports the principal native masses, while material-stratum checks remain unresolved. The [saved variance decomposition](hard-free-line-population-size-noise.md) identifies competing-contact orthants 22 and 62 as poorly covered: baseline importance ESS 10.5 and 3.8, with largest contributions 28.89% and 49.47%, respectively. Native remainder orthant 55 also remains important to uncertainty calibration. Increasing Poisson intensity alone does not address these pose-coverage limitations.

## Construction and existing implementation

Use the existing hard-free **raw translation coordinate** guide in `src/latent_region/contact_line.rs`, rather than changing the integration region or adding another SMC calculation. At fixed other five raw coordinates, orientation is fixed and translation is affine in the selected scalar. Existing code already supplies Gaussian conditional means and variances, stable Normal interval masses, numerical inverse CDFs, empty-line behavior, all-axis/all-component proposal densities, and physical Jacobians.

`src/line_geometry.rs` supplies endpoint-aware interval unions, intersections and differences, sphere-union hard-overlap intervals, atomic spherical-wall intervals, and contact intervals. The independent implementations in `tools/hard_free_line_reference.py`, `tools/analyze_contact_line_audit.py`, and `tools/physical_hard_free_line_vessel.py` already reconstruct geometry, Gaussian conditioning, and full-vessel physical densities. The earlier `tools/conditional_ray_proposal.py` gives an alternative uniform-ball radial conditional with exact angular marginal and uniform density in radius cubed. Its A7/B4 member shells do **not** implement the complete native-entry predicate and must not be substituted for it.

For each raw translation axis, let H be the existing hard-free line intervals within the unchanged R4 and source-capture constraints. Construct additional interval sets C contained in H. Candidate channels are:

- All exclusion contacts with no complete native entry.
- The same competing-contact class restricted to frozen orthant 22.
- The same competing-contact class restricted to frozen orthant 62.
- Complete native entry restricted to orthant 55, including both old-R5 support and its remainder.

Keep a broad hard-free channel as well as the **50% uniform R4 defensive component**. The first prototype assigns these five channels equal probability. Channel choices and probabilities must be frozen before fresh proposals; labels select a quadrature proposal, never a replacement physical target. All other classes, all 64 orthants, and the existing radial/angular strata remain reported. The old-R5 Boolean support described below remains an independent reporting definition; conditioning directly on its complement is deferred. The native55 proposal must not be described as a remainder-only channel.

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

For the frozen five-channel prototype, the broad channel has probability 0.2, so the complete new density is at least **0.2 times the old hard-free density** everywhere. This follows by retaining that channel's contribution and the common uniform component; the other terms are nonnegative. It is a coverage bound, not a guarantee of improved variance or runtime.

For the latent guide,

    q(u) = 0.5 U_R4(u)
         + 0.5 sum_g w_g phi_g(u) sum_a p_a sum_h p_h R_gah(u),

where R is the conditioned-to-unconditioned Normal ratio, including its hard-free/original fallback. No unknown multidimensional class normalizer is required. Regional weights remain J(u) W(u)/q(u), with both independent Poisson-cloud estimates per valid pose, unconditional attempted-draw denominators, and explicit hard-invalid/exterior zeros. Class labels remain the original independently evaluated reporting predicates.

For a later full-vessel stage, reuse the normalized outer 50/50 vessel/guide mixture and q_physical=q_latent/J exactly once. The source R4/capture constraints restrict only the guide. The original atomic-wall vessel and everything outside measured pockets retain their physical contribution. That stage still requires its existing gates; this proposal does not open them.

## Minimum implementation and checks

Add a native interval method alongside the current native observer, using the same frozen compiled definition. Expose only the interval primitives needed for member/residue constraints; keep the classifier itself unchanged. Extend the line guide with explicitly named class channels, fixed channel probabilities, and per-channel interval/mass/fallback traces. Reuse its Gaussian component law and add the new schema as an alternative, preserving old guide formats. The first schema binds the compiled native definition, shape, scaffold, capture domain and depletant radius; no old-R5 restriction is applied by this schema.

Before protein sampling:

- Compare interval membership with the complete classifier on toy motifs covering **ANY bond**, all-member intersections, wrong angular registration, native residue versus unrelated residue contacts, both anchors, overlapping motifs, and native complements. Include inclusive tangencies, zero-length sets and strict core/exclusion boundaries.
- Independently reconstruct complete unions from leaf atom pairs in Python. Compare pruned versus brute-force interval topology and endpoints, not only classifier values at a few points. Protein development diagnostics must use a frozen saved-pose inventory and preserve every selected query and failure.
- Test orthant zero-bit assignment, correlated raw/latent transforms, radial boundaries, constant predicates and empty lines. Any future old-R5-remainder channel additionally requires testing its complete native-q/capture Boolean complement.
- Verify conditional normalization, full component/axis/channel mixture density, hard-free fallback dominance on a single class, and a counterexample to union-wide dominance for different class channels. Test alpha=1 and disabled class conditioning against the original law, inverse-CDF reconstruction, floor transitions and exact endpoint handling.
- Compare independent Rust/Python q and J. Verify sphere hard-only and analytic depletion limits, all-attempt zero accounting, independent cloud roles, deterministic continuation and fatal-trace draining. Exact geometry means analytic model geometry; floating-point roots and endpoint classification remain explicit implementation obligations.

The expensive new part is residue-restricted contact interval construction. Perform the constant body-angle test and intersect the four member bounds first. Only surviving motif intervals need monomer checks or residue atom-pair searches. Native residue-pair atom lists can be built once from the frozen definition; cache shared member/class tests within a line. Conservative projected bounds may prune pairs, while exact leaf inequalities decide membership. Full mixture q needs all channels, but they can share H, N, exclusion-contact and old-reference sets per axis.

The class-guide loader also verifies that the compiled member/monomer spheres describe the physical rigid body. It reconstructs every sphere and finds a deterministic bijection, allowing atom permutations. The setup tolerances are 1e-10 Å for centers and 1e-12 Å for radii, with a reported bound on the resulting pair-gap discrepancy that must fit inside the original observer's 1e-8 Å overlap tolerance. This check changes no geometry. Its implication assumes proper rotations in real arithmetic; subsequent floating-point transforms remain a separate obligation.

Original-latent orthants require a precise numerical chart convention. The Rust chart validates/factorizes the covariance, reconstructs it from the factor, and factorizes again. Direct SciPy factorization can differ at approximately 1e-17 in nominally zero entries, which can change an exact sign test on an all-zero boundary probe. The independent reference therefore reconstructs this declared FP64 chart convention explicitly, without zero snapping or relaxed sign tests. Conditional Gaussian calculations, Normal integrals, atom-pair roots and physical labels remain independently implemented. Failed toy-audit records are retained; correcting the chart reconstruction does not redraw the Rust samples.

## Proposed staged evaluation, not yet an allocation

First freeze a proposal-only diagnostic that reports class/orthant feasibility, empty-line and numerical-floor fallback rates, conditional masses, full q, and CPU. Preserve all source/axis/channel rows; do not report only successful lines. `tools/prepare_native_class_line_probe.py` declares four populations of 128 unconditional fresh draws per arm (hard-free and class-conditioned; 1,024 total), plus 40 saved development probes: the eight already inspected 22/62/55 poses and eight per class from the existing breadth inventory. The saved probes are not an independent validation population. Every output receives an independent geometry/density audit, with no depletant clouds or physical-weight estimates. The preparation command only freezes inputs; the exact executable and reference source must be bound after synthetic tests and before execution.

The H-only arm uses the new schema with one hard-free channel. It is a density/coverage control; it still constructs the full class geometry for audit, so its runtime is not the optimized legacy H runtime. No speedup claim should use that comparison. The independent audits traverse all leaf pairs in bounded arrays and may take substantially longer than the Rust queries. They use one worker, preserve per-row progress and failures, and have fixed resource limits with no automatic retries. The dispatcher binds exact commands as well as metadata, and the summary verifies the audited input hashes before reading proposal fractions. These fractions are not physical occupancies.

A candidate physical comparison is **eight independent populations of 16,384 unconditional attempts per arm**: the existing hard-free guide versus the new class-balanced guide. Keep alpha=0.5, λ/z=128, two independent clouds per valid pose, and identical target/strata. More populations address the poorly calibrated four-population uncertainty without pretending that repartitioning alone increases pose ESS. This count, channel probabilities, numerical floor, seeds and analysis must be reviewed and frozen before launching; they are suggestions here, not authorization to draw or extend an existing campaign. A larger fresh stage should be specified in that new protocol, not appended after looking at its favorable bins.

Use the existing population-based mass, ESS, largest-contribution, 0.2 kBT/three-SE agreement and native/competing free-energy interval checks. Keep all material strata and historical failures visible, and analyze stages separately. Reuse the completed aggregate SMC evidence without rerunning it. Evaluate improvements by useful effective weighted contact samples per CPU, not native proposal fraction alone.

The principal limitation is unchanged: a translation-line proposal preserves five raw-coordinate marginals and **cannot recover a missing angular mode merely by conditioning its scalar translation**. High class-specific fallback rates would identify that limitation before an expensive physical campaign. Even successful regional convergence still needs full-domain coverage and the prescribed finite-system assembly, initialization, size, and boundary evidence.

## Implementation and launch, 2026-10-03

- 21 native-interval/classifier/shape-compatibility tests and 33 class/legacy line-guide tests passed. The latter include analytic sphere hard-volume and pair-depletion references.
- 33 Python reference, allocation, binding, dispatcher and summary tests passed, including child draining on timeout, interruption and SIGTERM.
- Five frozen synthetic CLI cases retained 256 fresh draws and 40 fixed queries. Independent complete geometry/density/Jacobian/inverse-CDF checks passed for all 296 rows. Maximum log-density discrepancy was 1.96e-14. Earlier audit failures and their corrections remain archived; no Rust draws were repeated.
- The fixed protein diagnostic is bound by execution-plan SHA `7f1bfeb01f8c0d371a5e1f14fbc66371b68fb485452be5fe25f5a2ca72fc246e`, using the isolated audit executable SHA `f9afc90000004062b67e5cc872124cd8203e2acd6bb462df2eb4835b330474bc`. Its 1,024 fresh proposals and 40 saved queries completed; independent audits are running with one worker. All 4,004 compiled native atoms matched the physical shape exactly in the setup check. No physical Poisson weights were sampled.

Artifacts: [Rust tests](../results/native-class-line-validation-20261003/class-validation-final.json), [Python tests](../results/native-class-line-validation-20261003/python-validation-final.json), [synthetic CLI audit](../results/native-class-line-synthetic-20261003/cli-validation-final.json), and [frozen protein execution plan](../results/native-class-line-proposal-preparation-20261003/execution-plan.json). The source-independent protein audit must finish before interpreting these proposal diagnostics. These checks do not open the physical-weight or assembly gates. The production assembly executable remains unchanged.

## Compact physical estimator and analytic control

The v7 normalizer now retains every class/axis interval, raw coordinates and the
complete per-component multiplier, while omitting redundant component/channel
records from production JSON. An independent reference reconstructs every branch
mass and fallback from the saved intervals and frozen guide. The full diagnostic
trace is unchanged. Compact/full q and random draws are bit-identical in tests;
a synthetic 92-component trace shrank from 270,128 to 7,124 bytes. This reduces
storage; it does not establish improved sampling. The isolated normalizer passed
94 tests, with one deliberately ignored fixture emitter. Available pose, q/J,
validity and completed-cloud fields are retained on a later attempt failure;
internal progress of an unfinished cloud is explicitly unavailable.

A frozen analytic sphere control used **16 independent populations of 2,048
attempts**: H-only and class proposals, four streams each, at z=0 and z=0.3,
core radius 1 and depletant radius 0.5. Four coincident member spheres represent
exactly one union sphere. The six-dimensional R4 chart couples translation and
orientation; the analytic reference integrates the normalized Haar cap rather
than treating the two as an independent product. These are test parameters, not
the protein decision conditions.

| Test | H-only estimate ± population SE | Class estimate ± population SE | Analytic |
|---|---:|---:|---:|
| Hard-only mass | 2.43936 ± 0.01413 | 2.41252 ± 0.01611 | 2.42201 |
| Depletion mass, z=0.3 | 2.83151 ± 0.02818 | 2.86610 ± 0.03051 | 2.83080 |

All 32,768 attempted poses received independent geometry, complete-q, Jacobian,
inverse-map and physical-estimator audits. The unconditional denominator retains
5,464 hard-invalid and 2,154 exterior flags (these categories may overlap).
There were 25,156 contributing poses and two cloud estimates per contributing
pose. At z=0 the estimator legitimately skips Poisson point generation.
Checks of counts, conditional weight moments, replica differences/products,
proposal agreement and analytic masses passed the fixed five-SE reference
criteria. These are validation diagnostics, not a proof of convergence.

The fixed physical execution and all independent audits completed without a
retry. Its frozen toy controller drained timeouts but lacked general signal/
exception draining; that limitation is preserved with the execution record.
The working controller now drains exceptions and signals as well, including
the process-launch ownership window; eight focused controller tests passed.
Those changes did not alter or repeat the completed allocation. The normalizer
and auditors themselves retain their attempted-row journals.

Artifacts: [Rust validation](../results/native-class-line-physical-validation-20261003/rust-validation.json),
[frozen sphere allocation](../results/native-class-line-physical-sphere-final-20261003/allocation.json),
[complete analytic result](../results/native-class-line-physical-sphere-final-20261003/analytic-validation.json).
The [next-campaign review](native-class-physical-campaign-review.md) describes
fresh protein controls and the unresolved coverage checks. No new physical
protein allocation has been launched by this sphere validation.
