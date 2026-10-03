# Review of the next physical contact calculation and frozen dimer analysis

Review date: 2026-10-03. This document records a proposed allocation and a launch recipe. It does **not** allocate new protein queries or launch an analysis. The review used source, frozen protocols, saved aggregate evidence and controller metadata; it did not sample poses, classify protein geometry, fit proposals or evaluate depletion weights.

## What remains unresolved

The decision target remains the repaired rigid tetramer at depletant radius **1.5 Å**, activity **0.035 Å⁻³**, and eventually an all-mobile finite system near **106.8 μM tetramers**. The current R4 calculation is conditional on a fixed two-neighbor scaffold. Neither its favorable native weights nor failure to sample a competing region decides finite-system assembly.

The completed [hard-free population-size campaign](hard-free-line-population-size-results.md) has useful aggregate precision. Its larger, four-population hard-free result gives native-minus-competing free energy −18.7204 kBT with a population-based 95% interval half-width 0.0494 kBT. The original Gaussian-plus-uniform proposal gives −18.6815 ± 0.1061 kBT. The historical arm named `baseline` is a **92-component Gaussian mixture plus 50% uniform R4**, with line conditioning disabled; it is not pure uniform R4 sampling.

These aggregate numbers do not close the original convergence checks:

- Competing-contact orthants 22 and 62 have poor pose-weight coverage. In the larger baseline, their importance ESS are 10.5 and 3.8, with largest individual contributions 28.89% and 49.47%. The corresponding hard-free ESS are 28.4 and 40.6. Increasing Poisson intensity cannot remedy missing poses.
- Material proposal/population-size discrepancies remain in the original radial and orthant strata. Native and native-remainder orthant 55 also expose uncertain calibration of four-population standard errors, even with apparently reasonable row ESS.
- The [defensive/intensity sensitivities](hard-free-line-sensitivity-results.md) have consistent aggregate physical weights but fail some hard-volume and material-stratum checks. The lower-intensity arm has larger single contributions; λ/z=64 is not an established improvement over 128.
- The [matched SMC reconciliation](hard-free-smc-reconciliation.md) supports total native, old-R5-intersection native and native-remainder masses. It uses each population's normalizer times its terminal indicator, not the terminal fraction alone. Its missing nonnative descendants do not establish zero competing mass or provide an upper bound. Older mismatched or discrepant SMC results remain visible.

Reuse all completed results without rerunning or reclassifying them. In particular, retain the original 786,432-draw campaign, its failures, and the new SMC analysis as separate evidence. The new class guide was designed using those failures, so the historical populations are development evidence rather than an independent prospective test of that guide.

## Readiness of the new proposal

The [class-line construction](native-class-line-guide-design.md) changes the normalized proposal, not the target, native predicate, R4 region or physical measure. Its five equally weighted conditional channels are broad hard-free support H, competing contacts, competing orthant 22, competing orthant 62 and **all native orthant 55**. The last channel includes both the old-R5 intersection and its remainder; it is not remainder-only. Keep the defensive R4 probability 0.5, numerical conditional-mass floor 1e-12, existing 92 components and the frozen complete classifier.

The current fixed proposal diagnostic contains 1,024 fresh unconditional proposals and 40 saved development probes, with independent geometry/density audits. It contains **no physical Poisson weights**. Its H-only arm still constructs all class geometry, and consequently is a density control rather than an optimized runtime baseline. Its full independent audit, the v7 physical normalizer checks and the independently allocated sphere hard-volume/depletion reference must pass before a protein weight campaign is bound.

Review of the physical estimator found the intended structure intact: J times the mean of two independent cloud estimates divided by the complete proposal density; hard-invalid and exterior attempts remain zero in their unconditional denominators. Indexed cloud RNG roles are distinct. New class-only failure records should also preserve a sampled pose, q/J, validity flags and completed cloud records when a later cloud fails. A begun attempt must never vanish or be replaced. A numerical failure must not be interpreted as an empty conditional set.

## Candidate fresh physical allocations, not frozen

Freeze all stages, seeds, proposal files, executables, reference closure, classifiers, time limits and comparisons before the first protein draw. The table is one candidate complete design for a **new** protocol, not a frozen million-draw allocation or a modification of an archived allocation. It remains gated on the fixed proposal diagnostic, independent audits and measured cost.

| Stage | Proposal | Independent populations × attempts | α | λ/z | New attempts |
|---|---|---:|---:|---:|---:|
| Primary control | Optimized existing H guide | 8 × 16,384 | 0.5 | 128 | 131,072 |
| Primary candidate | Frozen five-channel class guide | 8 × 16,384 | 0.5 | 128 | 131,072 |
| Larger candidate | Same class guide, fresh streams | 8 × 65,536 | 0.5 | 128 | 524,288 |
| Defensive sensitivity | Same class guide | 8 × 16,384 | 0.2 | 128 | 131,072 |
| Cloud sensitivity | Same class guide | 8 × 16,384 | 0.5 | 64 | 131,072 |
| **Total** | **40 independent populations** | | | | **1,048,576** |

Use two independent clouds for every valid pose. This candidate's absolute upper allocation is 2,097,152 cloud realizations; invalid/exterior draws legitimately use none. Preserve every attempt, including failed and invalid attempts. At most eight physical jobs and 32 scientific threads may be active globally, including existing growth/dimer jobs and independent geometry audits. A controller must use available capacity rather than treating eight as eight additional slots.

A smaller alternative freezes only the two primary arms: **262,144 attempts, 16 populations, at most 524,288 clouds**. It answers whether the new guide improves prospective pose coverage and cost, but cannot itself close the new population-size and sensitivity checks. Any later stage would require a new protocol, with the pilot retained as development evidence and fresh streams used for confirmation. The complete table has the advantage that those stages and decisions are fixed before looking at physical pilot results; the smaller alternative avoids committing expensive class-density calculations before their benefit is measured.

Eight populations improve estimation of between-population variation; dividing a fixed sample into more streams does not create more pose ESS. The fresh optimized-H arm provides an independent control for the data-informed class construction and comparable runtime. The completed four-population H-large stage remains a secondary population-size/reference comparison. Repeating H-large is not necessary for this economical candidate. A symmetric fresh H-large stage would cost another **524,288** attempts and must be declared before its draws, not an automatic response to an inconvenient result. If unresolved uncertainty remains in that control, report it rather than claiming closure from the candidate alone.

The saved large H arm used 9,120.96 sampler CPU seconds for 262,144 attempts, approximately **0.03479 CPU seconds/attempt**. At unchanged performance the fresh H primary arm would cost approximately 4,560.48 CPU seconds (1.27 CPU hours). This is historical calibration, not a budget for the more expensive class guide. If per-attempt times are t_H, t_C, t_C20 and t_C64, the table's estimated sampler CPU is `131072*t_H + 655360*t_C + 131072*t_C20 + 131072*t_C64`. Independent geometry/density auditing, classification and setup are additional. Obtain the class geometry cost from the fixed diagnostic after its audits finish; do not use its deliberately unoptimized H-only runtime as a speedup denominator or infer Poisson-cloud cost from that cloud-free diagnostic.

Analyze all five stages separately. The larger candidate is fresh, not a continuation containing the primary candidate; neither stage is pooled with historical rows. Compare the two fresh primary arms, class-large versus class-primary, each class sensitivity versus class-primary, and class-large versus archived H-large as a clearly labeled historical comparison. Retain the aggregate matched SMC comparison. Class sensitivity must be measured afresh: old H sensitivities do not validate a changed proposal.

This allocation is deliberately limited to the existing five-channel law. If the proposal-only audit shows mostly empty class lines in angular strata of concern, or runtime renders useful ESS/CPU implausible, pause before binding this physical allocation and document that evidence. Conditioning one raw translation coordinate preserves the other five raw marginals; it cannot repair an absent angular mode.

## Proposed audit budget and independent-reference optimization

The present `native_class_line_physical_reference.audit` independently rebuilds every atom-derived line set on every attempted row. At the live proposal audit's approximate **26 CPU seconds per row**, 262,144 rows would require about **79 CPU days**, and 1,048,576 about **316 CPU days**, before other analysis. This cannot be hidden inside the sampler's cost. The following alternatives are design proposals only; neither changes the running frozen audit nor authorizes additional geometry queries.

First consider preserving all-row independent geometry by improving the independent reference. `hard_free_line_reference.hard_free_intervals` already has a projected 2D `cKDTree` candidate search followed by long-double leaf quadratics; the class reference currently calls it with `use_tree=False`. The new exclusion/native `leaf_contact_intervals` routine is fully unpruned. For a line direction d, a necessary condition for an atom-pair interval is that the center separation projected perpendicular to d is at most the sum of radii plus the relevant gap. Query projected fixed centers using `moving_radius + maximum_fixed_radius + gap`, then apply the original pair-specific long-double residual, root, clipping and endpoint rules. Use gap zero for hard cores, 2rd for exclusion contacts, and the original native residue-pair whitelist and gap for supporting bonds. Strict/inclusive leaf predicates remain distinct.

The scaffold and its raw translation-axis directions are fixed, so transverse bases and fixed-atom projected trees can be cached per axis/anchor using a common projection origin. No Rust BVH or saved Rust candidate list is needed. `cKDTree` queries must be exact (`eps=0`) and their radius must include a justified outward guard for coordinate projection, basis orthogonality and search rounding; the existing constant 1e-9 guard is not automatically a general error bound. Ambiguous or unrepresentable bounds fall back to all pairs, not an empty set. Validate endpoint topology and inclusions against the frozen unpruned implementation on predeclared synthetic tangencies, large-coordinate cases and a separately bounded saved-protein allocation. Do not reuse already completed protein validation as if it tested a new pruning path, and do not repeat the current allocation silently. Measure the resulting cost before deciding whether all-row independent geometry becomes affordable.

A second possible design separates two audit layers:

1. **Every attempted row:** independently reconstruct raw/latent/physical maps, J, all Gaussian conditional means/variances, and all axis/component/channel Normal interval masses from the saved complete endpoint-aware interval sets. Rebuild channel Boolean intersections/differences and original-latent orthant clipping, fallback decisions, full mixture q and selected inverse-CDF arithmetic. Compare independently recomputed multipliers with saved ones; never accept the saved component multipliers as the reconstruction. Check selected-line geometry consistency between generation and scoring, unchanged other five raw coordinates, original-q/capture/R4 algebra, all-attempt IDs and denominators, invalid zeros, both cloud-count weight formulas and their arithmetic mean, and source/output bindings. Stream records instead of retaining a million full traces in memory. Preserve draw-wise audit progress and failure context.
2. **A separately declared full-geometry allocation:** rebuild H/native/exclusion intervals from original atoms without Rust interval input, compare complete topology/endpoints and classifier membership, and check the original geometry-dependent validity/label predicates. Reuse the completed 1,024 fresh proposal validations and 40 development probes under their original implementation hashes; they are not fresh production checks or a validation of later changed geometry code.

For a concrete bounded option, preselect **16 unconditional draw IDs per population** using a separate frozen audit seed before physical draws. Add, diagnostically, at most the single largest contributor in each of the four decision regions per population, with ties settled by lowest draw ID and no replacement if a region is unobserved. Deduplicate overlapping IDs. This caps new full-geometry checks at **320 rows for a 16-population primary-only campaign** or **800 rows for the 40-population candidate**: approximately 2.3 or 5.8 CPU hours at 26 seconds/row. Counts, IDs/selection rule, source hashes, CPU/wall limits and worker capacity must be bound before observing the selected geometry. Largest-contribution checks are explicitly outcome-selected diagnostics; they do not form an unbiased validation sample or authorize removing an awkward weight. A discrepancy fails the audit and preserves the population; it never triggers dropping, replacing or silently extending audited rows.

The trust boundary must be visible in receipts and conclusions. All-row saved-interval reconstruction establishes the **proposal/estimator algebra conditional on those recorded sets and geometry-dependent flags**. It does not independently prove that every recorded set matches physical atoms, that every hard-valid/native label is correct, or that the depletant envelope and thinning were exact. Cloud counts and distinct RNG keys do not prove independence or Poisson spatial sampling. Independent physical hard/contact/native classification for production regions should remain a separate budgeted pass wherever feasible; if reduced, that additional reliance on the pinned geometry implementation must be stated. Full independent geometry on a sample provides implementation evidence, not a guarantee against rare geometry bugs or unseen thermodynamic modes. Receipts must distinguish `all_rows_algebra` from `full_geometry_selected_rows`; they must not retain the current claim that every geometry row was independently reconstructed.

Adopt either optimization or split only in a new validated reference/protocol. Neither computational convenience nor the class guide's defensive component justifies silently weakening an already frozen all-row audit. Full physical convergence and remaining-support requirements remain separate from this audit-budget choice.

The first bounded optional-pruning assessment can use **exactly 56 already saved poses**: fresh IDs 0–15 of `hard_free-r00`, plus all 40 `saved` development queries in the frozen proposal pilot. This is a new, separately bound evaluation of the optional reference path; it draws no poses/clouds and does not rerun the unpruned reference. Construct the current independent `Reconstructor(..., use_tree=True)`, call its density reconstruction once per selected row, then compare the full saved density trace, including every H/native/exclusion interval union and channel, q, J and coordinate maps. For the 16 fresh rows also run the existing inverse-CDF/draw-trace check using the reconstructed axes; saved probes must have `draw=None`. H-only q is insensitive to native/exclusion interval errors, so checking q alone would be inadequate. Endpoint topology and open/closed flags are mandatory. Three axes imply at most 168 line reconstructions; draw verification reuses them. No extra direct point-classifier pass is needed solely to test this candidate-pruning change.

Observer initialization is an additional, explicit geometry allocation: seven directed reference-contact queries and one fixed-scaffold classifier, whose catalogue contains 30 distinct member/class triples and therefore permits at most 30 further contact queries. Bind that catalogue and scaffold, journal each begun/completed setup query, and report setup CPU separately. These at most 37 contact queries and one classifier call are outside the 56 density reconstructions; initialization is not geometry-free. Keep the existing observer checks intact.

The first 16 unpruned checks are already complete inside the original audit journal. Bind an immutable byte-for-byte snapshot of its first **33 newline-terminated records**: the `started` event followed by the ordered `begin`/`complete` pairs for ordinals 0–15. Bind the original execution plan, the `started.input_sha256` closure, complete immutable Rust row files, and selected-row identities/hashes. Require the live journal to retain that exact prefix before and after the optional assessment; its later appended records must not invalidate the binding. Do not hash the whole changing journal as a purported immutable predecessor. Its original reference SHA is `1e154ca3b335be2ce233d460554bde66811f3a63b047143959ea9ac6a663ec80`. The journal records completion/error summaries rather than complete independently computed interval sets, so the comparison establishes that both reference paths agree with the same frozen Rust trace within declared tolerances; it is not a direct bitwise comparison of old and new independently generated endpoints.

The 40 saved queries' original unpruned audits are still pending. Optional results can be recorded under the fixed 56-row allocation, but their final independent-equivalence status must remain pending until the existing unpruned completion evidence is available and linked without replay. Preserve all failures and unfinished predecessors; use no replacement IDs. Report matched-row CPU and candidate counts with scope limited to these selected fresh/development rows. Keep hard-core reconstruction unpruned for this first assessment. Only after its 56-row checks and predecessor evidence are complete should measured remaining hard-core cost motivate a separate pruning change, synthetic validation and independently declared protein check. Combining both optimizations now would obscure the source of any mismatch and change the comparison being validated.

This assessment was prepared and launched on 2026-10-03 in
`results/native-class-projected-protein-preparation-20261003`. Its allocation SHA
is `d546082a71fe1e9ef70028e09b1481c9733292c835f12a6759973a2a567586b6`;
freeze SHA is `ce83ad08021c44ab7c5e7103bbb42a9d51c25cf4b37ac161be04f99b32471ee1`.
`tools/prepare_native_class_projected_probe.py` binds the exact saved inputs and
separate root review, then launches the archived worker once. The worker journals
setup and row attempts, drains on failure, and has no retry mode. Its limits are
one scientific worker, 3,600 CPU seconds, 7,200 wall seconds and 16 GiB.
The [10-test helper receipt](../results/native-class-projected-probe-validation-20261003/validation.json)
and existing 38-test projected-reference receipt are linked by `root-review.json`.
The host preflight reserved the eighth scientific worker alongside three growth
jobs, up to three dimer jobs and the original one-worker audit. This launch creates
no physical draws and leaves the full independent audit and physical campaign
gates closed. Consult its terminal receipt for completion; launch is not a pass.

The [56-row terminal receipt](../results/native-class-projected-protein-preparation-20261003/completion.json)
now reports a pass with no retries. All 168 axis reconstructions agreed with the
saved Rust intervals and full densities: maximum endpoint discrepancy
2.30e-13 Å, log-density discrepancy 1.36e-12, Jacobian discrepancy zero, and
inverse-CDF discrepancy 6.67e-15. Setup used seven reference contacts and six
scaffold contacts inside one classifier call. The
[cost assessment](../results/native-class-projected-protein-preparation-20261003/cost-assessment.json)
uses only completed counters: 918.72 row CPU seconds, or 16.41 per row; the fresh
16-row and saved 40-row means are 16.15 and 16.51 seconds respectively.
Exclusion-pair candidates fell from 5,386,757,376 to 154,844,507 (97.13% removed),
but 5,386,757,376 hard-core pairs remained. Hard cores account for 97.20% of the
remaining evaluated pairs. Native-whitelist pairs were 535,128 possible and
533,113 evaluated. Pair fractions are not CPU fractions. The original audit
also ran point observers, so these measurements do not define a matched-workload
speedup ratio. The 40 saved queries still await their original unpruned audits;
the complete pilot and physical campaign gates remain closed.

### Geometry-free proposal algebra foundation

`tools/native_class_line_algebra_reference.py` now supplies
`AlgebraLaw(region, guide, config)`, `evaluate_saved(u, compact_trace)` and
`verify_draw(draw, u, evaluation)`. Its initializer builds coordinate and Gaussian
parameters without atom trees or a native observer. The evaluator independently
reconstructs the raw/physical maps and Jacobian, R4/capture chords, class Boolean
sets and original orthants, all conditional Gaussian masses, fallback decisions
and the full component/axis/channel density. Serialized channel intervals and
component multipliers are checked after reconstruction, not used as answers.
The draw verifier reuses these results and reports numerically ambiguous interval
selection using the existing 2e-14 cumulative-mass tolerance. Exact serialized
interval identity distinguishes nearby narrow intervals before tolerance matching.

The [12-test receipt](../results/native-class-algebra-validation-20261003/validation.json)
includes comparison with full toy geometry, 92 components × three axes × five
channels, coordinate/Jacobian inverses, every fallback, disabled conditioning,
exterior and tangent cases, deliberate derived-field corruption, and guards
against geometry calls. A deliberately self-consistent incorrect base union can
pass this algebra audit while disagreeing with atoms; the result explicitly sets
`geometry_certified=false`. Sixteen evaluations on one fixed 92-component toy
trace took 0.05445 CPU seconds (3.40 ms/evaluation). This is neither protein timing
nor a comparison against the full atom audit. A production streaming wrapper,
physical estimator/label pass and frozen independent geometry allocation remain
to be implemented and bound; this module does not launch or certify a campaign.

The distinction between proposal geometry and physical geometry is useful.
In exact arithmetic, write the raw chart as `x=T(u)=(y,s)`, with `s=x[a]` and
`y` the other five coordinates. For Gaussian component k, any measurable
effective set `E_kac(y)` with positive Normal conditional mass `M_kac(y)` defines
a normalized component density
`g_k(u) * 1_E(s) / M_kac(y)`. Integrating over s first recovers the Gaussian
marginal of y. The class → H → whole-line fallback also has this property when
its decisions depend only on y; the whole-line mass is one. Fixed component,
axis, channel and defensive mixture probabilities therefore preserve
normalization. These sets need not coincide with physical contacts.

The Rust `class_geometry` construction sets `raw[axis]=0` before constructing
its frame, chords and sets; its Normal conditional excludes that coordinate.
Draw and density paths use the same construction. This supplies a source-level
real-arithmetic argument, not a proof from isolated saved rows or of floating-point
execution. No hidden random/cache/state dependence may change the sets between
generation and scoring. Masses, inverse draws, full q, fallbacks, uncensored
attempts and support must remain correct. Raw/latent roundoff near topology and
mass-floor boundaries remains a numerical obligation. With those conditions,
imperfect **proposal** contact sets affect efficiency rather than the physical
importance integral. The physical hard predicate, target domain/Jacobian,
conditional expectation of the depletion estimator, and complete native/contact
labels must still be correct independently. This observation does not change any
running audit, frozen allocation or launch gate.

## Analysis requirements and stop decisions

Two existing wrappers explicitly require four populations: `analyze_contact_confirmation.summarize_arm` and `analyze_conditional_ray_campaign.free_energy_interval`. The new arithmetic adapter documented below supports eight populations without modifying those wrappers or archived outputs; binding it to production inputs remains a future campaign obligation. It uses each population's linear mean importance weight to calculate between-population uncertainty, covariance and ratios. The new paired-population delta interval uses Student-t with **7 degrees of freedom**, while historical four-population intervals retain 3. Its synthetic validation includes unequal population counts, constant weights, zero mass, covariance and partition checks.

Preserve the original numerical thresholds. The history contains two statistical implementations: the older generic `analyze_conditional_ray_campaign.compare_mass` uses a log-delta comparison, while the newer `analyze_hard_free_line_population_size.compare_linear` and `analyze_hard_free_line_sensitivity.compare_linear` compare **linear masses**. Archived outputs must retain their original rule. The new preregistration should explicitly use the latter linear comparison, consistent with the requested linear-mass evidence:

- Decision-driving regional physical masses: between-population relative SE ≤10%, importance ESS ≥200 and largest contribution ≤2%.
- Proposal, population-size and intensity agreement: **|mean Q₁ − mean Q₂| ≤3 sqrt(SE₁²+SE₂²)** and, independently, **|log mean Q₁ − log mean Q₂| ≤0.2**. Evaluate linear means and SE on a shared safe scale, allowing only the existing 32 machine-epsilon times maximum scaled mean rounding tolerance. Compute each SE as sample SD divided by sqrt(number of independent populations), including comparisons with historical four-population arms. Log-delta SE is diagnostic, not the new statistical pass criterion. Hard-only Q0 checks remain reported alongside Qz.
- Native/competing free-energy interval: population-based 95% half-width ≤0.5 kBT, including the covariance of the two masses within each population.
- Original radial edges `[0,2,3,4]`, angular-projection-squared edges `[0,4,9,16]`, and all 64 original-latent sign orthants, with zero assigned positive. A stratum is material when it contributes at least 1% of its own parent regional physical mass in either arm of the comparison. Historical failed strata remain listed even if a fresh estimate falls below that threshold.

Use the complete native / contact-without-entry / unbound partition and independently partition native mass into the exact old-R5 support and its complement. The old support is `old_chart_radius <= 5 AND original_q > 1 AND old_capture_valid`. Keep hard-invalid/exterior counts, branch contributions and channel/fallback diagnostics. Empty observed bins are neither known zero mass nor bounds. Compare row and population SE, the two-cloud noise decomposition, and the largest-contribution locations. Preserve sampled selected-class/H conditional mass summaries to distinguish scarce class support from bad Poisson noise.

Report importance ESS per measured sampler CPU, all audit/observer CPU separately, and setup-inclusive cost when it is measured. This is an importance-quadrature efficiency diagnostic. It is not stationary contact-fingerprint ESS, an MCMC speedup or evidence of reversible environment exchanges.

Correctness/provenance failure stops dispatch and drains every started job. Preserve all attempted draw IDs, partial outputs and failure records; no replacement draws, reseeding or reruns to complete a favorable table. Statistical failure at the primary stage is not, by itself, grounds for data-dependent extension or early success: the larger/sensitivity stages may run only as already declared. If the fixed allocation ends with unstable material strata, missing significant mass, failed comparisons or unresolved population calibration, the thermodynamic conclusion remains **unresolved sampling**. Successful aggregate checks do not erase those failures and unsuccessful sampling is not evidence against assembly.

## Remaining support and the assembly boundary

Uniform R4 support gives proposal coverage but does not prove unseen modes have negligible mass. Regional success must include every material class/stratum and a sufficiently useful measurement or bound for its remainder. The native55 guide does not explicitly cover an old-R5-complement channel, and all contact modes outside R4 remain separate.

The inert [full-vessel preparation](hard-free-streaming-vessel-stage.md) specifies the original atomic-wall sphere (radius 223.3261767 Å), capture 273 Å and wall-permeable depletant bath. The source guide's R4/capture restriction is not the vessel target. A future full-vessel comparison is the normalized vessel proposal versus 50% vessel + 50% class-guide physical density, with `q_physical=q_latent/J` exactly once. The existing prepared H guide must not be edited in place or assumed to support the new schema: bind a new class-compatible adapter, validate exterior-zero/seam/normalization/J behavior and prepare a new provenance closure first.

The predeclared full-vessel sizes are four populations of 65,536 per arm, followed by four fresh populations of 262,144 per arm: **2,621,440** attempts in total. Do not launch them before the regional gates. Report native, contact-without-entry and unbound weights everywhere, including outside measured pockets and outside R4. The existing exact unbound upper bound Q ≤45,654,774.77 Å³ (log Q≈17.636619) is useful because W=1 there. It does not bound unseen adsorbed/contact mass. Do not dilute a material regional discrepancy by dividing by the much larger whole-vessel mass.

Only after contact/domain evidence is adequate should contact-sampling benchmarks and finite-system N=12/N=24 assembly tests decide the original endpoint. Keep equal concentration, independent dispersed/competing/native-seeded starts, geometry-only frozen proposals, native-informed positive controls, matched collective schedules and the valid common move set for equal-volume boundary controls. Native intratetramer structure is already supplied. A conditional dimer improvement, an isolated favorable native contact or a short assembly trajectory is insufficient to decide native intertetramer stability.

## Frozen evolving-dimer final-analysis recipe

The running campaign is `results/evolving-dimer-campaign-20261003`. It concerns **two mobile labeled tetramers with 262 fixed spectators**, rd=1.4 Å, z=0.0275 Å⁻³, and the inherited 500 μM source. Its four contexts are different conditional targets. It is not the original assembly decision system. Do not restart its controller or any growth controller. At this review's metadata snapshot the 96-chain campaign was incomplete; no final analysis was launched.

Use the already frozen scientific observer in `common/source/tools`, not an evolving working-tree copy. No new scientific source preparation is required for the declared native-blind patch/contact metrics. Before the first new geometry query, create a separate metadata-only execution binding and exclusive claim that fixes: this observer closure, Python executable/version/package identities, exact argv, fresh output directory, resource limits, and the complete terminal/trajectory hash inventory. The existing analyzer verifies its frozen plan/source, journals and input hashes but does not supply that final controller or resumable query ledger.

Preflight requires **all 96 unique planned jobs** to be terminal, successful with return code zero, and bound to the current immutable config/run-binding. Dispatch must be complete and passed, with no active or unstarted job. Require exact job equality among config, dispatch plan and completed records; validate the dispatch status's plan digest, plan's config/binding/executable/controller/review digests, and each terminal's exact job, `complete=true`, `conditional_target=true`, 4,608 blocks, config and binding digests. Require each trajectory to be its job's `execution/job-NNN/trajectory.jsonl`, with the terminal's digest; terminal digests must match dispatch records. Verify `freeze.json` and every `files`/original-input binding, the protocol's archived Python source closure, run-binding's prelaunch binding/prepared manifest, and every prepared-file digest. Freeze that full inventory before observing any endpoint. A successful prefix or currently running trajectory is not sufficient.

| Frozen artifact | SHA-256 |
|---|---|
| `config.json` | `c96c938fe33fb6ade9d2a984fbcd790daa0885ffdd496d9cd21ebb7bebb62054` |
| `run-binding.json` | `eedbf181270ef96b9991c08038e12fd205ee2496ea6f4ea20972327dc27cfaed` |
| `analysis-plan.json` | `6a76f6dc0cd3999de99ad38798fd5365caa950e7884532a68e4d990eefefc3e6` |
| `protocol.json` | `5d0f6f69a4c21746dd754c6cadf98cff739fedfa178b0a6098eba0d41708e7fb` |
| `dispatch/plan.json` | `9330f6b343ed227c38ad77dc5e26b08d023c29e60811c470075c315a3713ec0d` |
| `common/source/tools/analyze_evolving_dimer_benchmark.py` | `bce32d6c54e7f3655134e59f9e35838adc06e84a6205e7257189ce8492293d24` |
| `common/source/tools/analyze_contact_efficiency.py` | `c64ce8b62d9a2c99f6933343b7800a81ee5afeff4d4add8cede386487ac49594` |
| `common/source/tools/mobile_posterior_metrics.py` | `9dbbdf18fecd7c080530011c0d5bb1fdd3c423b68a63ac028b0e6adc629990b4` |
| `common/source/tools/test_analyze_evolving_dimer_benchmark.py` | `41ee2be95746d78be7f47ec024b398782ae9ebc40e1bd705d6841b9840405b2d` |
| `common/source/tools/prepare_evolving_dimer_benchmark.py` | `2de958e3f80514570262f7183a8821dd804f872cf0f29b969fe006cd65f2fd1b` |

The frozen allocation is 4 contexts × 3 arms × 2 initializations × 4 streams. Each chain contains 512 warmup and 4,096 production blocks, plus its initial endpoint. The analysis will:

- Replay **1,769,472 local and 294,912 dimer elementary attempts**, including rejections and self-loops, without rerunning moves or generating new physical draws.
- Classify **442,464 retained/initial endpoints**, of which 393,216 are production and 49,248 are initial/warmup. These are new geometry observations under the predeclared observer.
- Consider at most **525 distinct mobile-related pairs per endpoint**, hence **232,293,600 pair classifications** before conservative center pruning and exact-pose cache reuse. Actual calls and cache hits are reported. The separate mobile-wall check transforms 2×4,004 atomic centers per endpoint: **3,543,251,712** atom checks. Pair classification may include additional internal atomic work; these are counts at specified API levels, not CPU-operation estimates.
- Use one observer process/CPU worker with OpenMP/OpenBLAS/MKL each limited to one thread; no concurrent duplicate analysis. Proposed outer ceilings to freeze are **86,400 CPU seconds, 172,800 wall seconds and 16 GiB address space**. These are safety budgets, not runtime predictions. They may be chosen differently before launch, but must not be extended silently after an incomplete prefix.

The scientific command below must only be run through the bound, failure-draining launcher after preflight. The proposed output directory must not already exist:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python -B \
  /home/xvg/tetramer-mc/results/evolving-dimer-campaign-20261003/common/source/tools/analyze_evolving_dimer_benchmark.py \
  --base /home/xvg/tetramer-mc/results/evolving-dimer-campaign-20261003 \
  --output /home/xvg/tetramer-mc/results/evolving-dimer-analysis-final-20261003/analysis
```

`tools/run_evolving_dimer_analysis.py` supplies that outer launcher. Its preparation
command refuses to create an execution directory until all 96 chains pass the
metadata checks. It binds every trajectory, prepared input, frozen source,
interpreter/package identity and the limits above without evaluating geometry:

```sh
/home/xvg/protein-nucleation/.venv/bin/python tools/run_evolving_dimer_analysis.py \
  --base results/evolving-dimer-campaign-20261003 \
  --root results/evolving-dimer-analysis-final-20261003
```

After preparation, execute its copied `runner.py --run --root <absolute execution
directory>` in a persistent process. The wrapper claims the allocation exclusively,
runs one frozen observer, rechecks the input/plan hashes, and records either a
complete manifest or a retained failure. It does not alter the scientific observer
or add a resume mode. Tests exercise incomplete/duplicate job inventories, changed
inputs, launch failure, nonzero exit, timeout draining and refusal of a second run.

The outer launcher must set the chosen resource limits, save stdout/stderr and a terminal receipt, drain/reap the process on timeout, interruption or SIGTERM, and preserve its claim on failure. The observer's own output creation is exclusive; it has **no resume mode**. Its JSONL observation files are buffered and it emits final `analysis.json`/`manifest.json` only after all chains. A failure may therefore leave incomplete output and some unflushed geometry observations. Do not erase them, restart in another directory, or call the missing results zero. Record the failed allocation as incomplete; any later salvage/continuation must have a new explicit protocol that avoids silently repeating the observed prefix. This limitation is a reason to use generous predeclared limits, not to claim resumability.

The journal replay validates stored acceptance decisions and retained states; it does not independently recompute every original Poisson cloud or every depletion acceptance factor. The new observer is explicitly native-blind (`native=None`). It reports patch/partner/fingerprint occupancies, apparent contact ESS, passages/returns and Jaccard changes with rejected residence retained. Constant observables have null ESS, not infinite speed. Summaries keep contexts and trajectories separate, compare paired arm streams and initializations, and charge full sampler CPU including warmup/null moves while reporting observer CPU separately. These results can guide reversible oligomer proposals; they cannot by themselves establish equilibrium or native assembly.

On 2026-10-03 the campaign completed all **96/96** chains successfully, with no
active or unstarted chain and no failure draining. The metadata-only final
preparation bound 845 files and the original frozen observer. Its execution plan
is `results/evolving-dimer-analysis-final-20261003/execution-plan.json`, SHA
`255a1fc59a96cb19f733a0385eaefc637ad428f646b9b1cbd04fba00f72a4607`.
The one-worker analysis was launched under the bounds above; its host receipt and
resource preflight are in the same directory. Five scientific workers were
reserved in total: three growth simulations, the original class-proposal audit
and this observer, leaving three single-thread test slots. No physical draws
were added. Sampling completion is distinct from analysis completion; inspect
the analysis terminal receipt before interpreting efficiency or agreement.


## Population-statistics API and synthetic validation

`tools/contact_population_statistics.py` provides two pure arithmetic entry points:
`summarize_populations(declaration, records)` and
`compare_population_masses(left_declaration, left_records, right_declaration, right_records, region)`.
A declaration fixes the arm, population count, equal attempted-draw budget per
population, unconditional total, ordered region names, and every population ID
and seed. Each record supplies the corresponding ID, seed, attempted draws,
unconditional denominator, and regional log masses. These must be logs of
linear population means; null or minus infinity denotes an observed zero.
Four and eight populations are supported, with unequal counts and draw budgets
allowed between independent comparison arms. Missing populations, changed
denominators, repeated seeds, and invalid log masses are rejected.

Each declaration also requires `target_and_regions_sha256`. The caller must
bind this immutable identity to the exact shape, scaffold, domain, physical
measure, depletant radius/activity, classifier, and complete region definitions.
The helper requires identical identities for cross-arm comparisons; it does
not load geometry, establish target equivalence, or verify the original row
estimator. Those remain obligations of the upstream audited input preparation.

The summary averages linear masses, retains paired population covariance, and
reports the native/competing delta-method interval using Student t with P−1
degrees of freedom. Independent-arm comparisons use linear-Q standard errors
with each arm's own population count and the separate 0.2-log-unit condition.
All-zero classes retain their observed counts but supply no precision claim,
finite free-energy interval, physical-zero conclusion, or upper bound. Scaled
linear quantities and finite log means distinguish numerical display underflow
from unobserved mass. Row importance ESS and largest individual contributions
cannot be inferred from population means and are not reported by this helper.

The [source-bound validation](../results/contact-population-statistics-validation-20261003/validation.json)
contains 16 arithmetic-only tests. Synthetic partition fixtures check that
native-old plus native-remainder equals native, and native plus competing plus
unbound equals total, including the covariance sums and arbitrary region and
record order. Other fixtures cover four/eight populations, unequal population
counts, paired correlations, zeros, constant observations, extreme log scales,
independent seeds, denominator checks, and incompatible target identities.
No old scientific inputs were read or recomputed, and no physical allocation or
sampling was launched. Archived four-population analyses remain unchanged;
these approximate uncertainty diagnostics do not establish unseen-mode coverage.

## Streaming physical-weight algebra audit

`tools/native_class_line_physical_algebra_audit.py` now provides the arithmetic
half of a layered audit for v7 records. It streams every original attempted
index and reconstructs the coordinate map, Jacobian, complete proposal density,
conditional draw, capture/original-q/shell predicates, two-cloud arithmetic and
unconditional denominator. The single-row implementation is
`tools/native_class_line_weight_row.py`. Rejected poses retain zero weights;
the two cloud estimates are averaged in linear weight, not log weight.

Only four numeric arrays are retained for population statistics. Attempt and
sample streams must contain exactly the declared rows, in order, with newline
termination and no duplicate JSON keys or nonfinite constants. A durable journal
records each begun and completed row, byte offsets and hashes. Source, runtime,
input and consumed-stream hashes are checked; exceptions and SIGTERM preserve
a failed receipt and do not trigger retries. An uncatchable termination can only
leave the fsynced journal. Existing producers and complete geometry auditors
are unchanged.

This audit conditions on saved H/native/exclusion interval unions. It does not
query atomic overlaps, establish the physical hard-valid flag, independently
classify native contacts, or prove Poisson spatial sampling and RNG execution.
Static atom-bijection validation is distinguished from geometry queries. The
receipt explicitly reports `geometry_certified=false` and zero independently
reconstructed geometry rows. The retained-five-coordinate dependence required
by the proposal normalization argument remains a source-level obligation.

The row validator passes 12 synthetic tests, with receipt
`results/native-class-row-weight-validation-20261003/validation.json` (SHA256
`b7e83caea9baa5d1911e8d302ff9962dcd35fc3ca25b8b727fde46d72dc05d2e`).
The streaming wrapper passes 13 tests including tampering, malformed/truncated
streams, impossible beta=1 lineage and interruption handling; receipt
`results/native-class-physical-algebra-validation-20261003/wrapper-tests02.json`
has SHA256 `68aa45a810d55e1f8b3cfc3bb9bf9ca0f27f657fe44dff0cc0406b863f93ea3d`.
The earlier 12-case wrapper receipt is retained; the final extra case followed
an independent review finding, rather than a repeated sampling allocation.
These tests used no real scientific rows, atom queries or physical draws.

The dimer final observer has also completed successfully. Its 96-chain analysis
and implications are documented in [the benchmark report](evolving-dimer-benchmark.md#completed-comparison).
The original unpruned class-proposal controller remains live: all nine query
jobs and the first two independent audits have passed. It must finish its
remaining frozen audits before the proposal pilot is declared complete.
