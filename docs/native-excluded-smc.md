# An independent SMC control for contacts without native entry

The completed importance campaigns estimate the native and competing weights
inside the same fixed R4 region. The existing independent SMC controls produce
native terminal configurations but no competing-region endpoints. A vanishing
terminal fraction cannot independently validate the competing denominator.

This extension targets a **separate, explicitly restricted integral**. It does
not change the model, classifier, assembly kernels, or original regional target.
The results must be kept separate from unrestricted SMC and combined only through
matching, disjoint region integrals. A restricted calculation alone cannot
establish vessel coverage or finite-system assembly.

## Target and invariance

Let `h(x)` be the original hard/capture/R4 indicator, `n(x)` the complete
instantaneous native-entry indicator, and `C(x)` the original many-body exclusion
overlap with both physical neighbors. For the physical-activity bridge use

```
h_minus(x) = h(x) [1 - n(x)]
gamma_t_minus(dx) = h_minus(x) exp(z_t C(x)) dmu(x).
```

Here `dmu` remains Cartesian translation times normalized proper rotational Haar
measure. For the existing proposal-density bridge, use
`h_minus(x) g(x)^(1-beta_t) exp(beta_t z C(x)) dmu(x)`, with the same fixed,
fully evaluable initial physical proposal density `g`.

An unconditional initializer `x ~ g` retains every attempted draw. The
physical-activity bridge uses initial weight `h_minus/g`; the proposal-density
bridge starts with weight `h_minus`. The restricted hard-volume diagnostic always
uses `h_minus/g`. A native pose is a **target rejection**, not a hard overlap.
Geometric and target validity are therefore recorded separately. A zero-hit
population remains a zero estimator; it is never replenished until successful.

The existing exact mutation kernel satisfies accepted-flow symmetry for the
unrestricted stage density. Multiplying its accepted flow by
`[1-n(x)][1-n(y)]` preserves that symmetry. Equivalently, for a current state in
the restricted domain, reject a native candidate before the bath gate and retain
the current state. Rejections complete the Markov kernel and preserve the
restricted stage distribution. The proposal-density bridge keeps its existing
deterministic proposal-density factor as well as its physical bath acceptance.

Fresh independent Poisson potential factors retain their original conditional
expectations on this smaller domain. The same resampling and invariant-mutation
argument as in [the unrestricted derivation](smc-r4-control.md) then yields

```
E[Zhat_T eta_T(f)] = integral h_minus exp(z C) f dmu.
```

For the competing contact weight, `f` is the original exclusion-contact indicator.
For the unbound remainder it is its complement. Report `Zhat_T` times each
terminal indicator mean, including zero populations. The normalizer alone
includes both classes. Neither an endpoint fraction alone nor an SMC ancestry
count establishes independent contact samples or convergence.

## Complete native predicate

`tools/compile_native_entry.py` materializes the verified frozen Python
definition into explicit member transforms, monomer atoms/residue IDs, native
residue-pair patches, and directed motif transforms. The actual catalogue has
14 tetramer motifs and seven directed monomer classes, with 1,001 monomer atoms,
129 residues and four rigid members. All original source hashes are retained.

A match requires the complete body-position and body-orientation gates and at
least one prescribed member bond satisfying monomer position, orientation and
native residue-patch entry. All matching motifs and anchors are retained. The
entry definition has no hysteresis, R5 restriction, or nearest-motif substitution.
Catalogue cycle consistency remains a separate assembly diagnostic; it is not
part of `native_any`.

`src/native_entry.rs` evaluates this predicate. `native-entry-probe` accepts
one JSON request per line with a moving pose, optional explicit anchor, and an
explicit `hard_valid: true` precondition. It returns all motif and supporting
residue-patch matches, or an error. Full atomic hard, capture and wall checks
remain caller obligations. The SMC integration calls the predicate only after
its physical hard/capture/R4 checks.

`tools/check_native_entry_parity.py` compares the probe against the independent
Python predicate on exact motifs, reversed directions, quaternion signs, global
proper isometries, member-threshold perturbations, and fixed-scaffold poses.
It independently checks complete tetramer atomic hard validity before asserting
that precondition. Clashing test candidates are retained in the fixture record
and excluded from valid-pose parity. These are geometric tests, not physical
normalizer samples or a replay of completed population classifications.

## Audit and production requirements

Restricted SMC uses distinct output schemas and archives its compiled definition.
It binds that definition to the current tetramer shape and fixed scaffold.
Initialization retains geometric validity, target validity, native decisions,
both hard-volume weights and the operative initial weight. Mutation records retain
current and candidate decisions, including all native-rejected candidates before
the bath gate. Every rejected state remains part of the transition history.

The independent `tools/analyze_native_excluded_smc.py` reconstructs this membership
and the complete normalizer recursion. The existing unrestricted auditor does not
accept restricted outputs. Zero/finite-activity reference integrals, all-zero
initialization, default-regression checks, a frozen independent allocation and
explicit contact-versus-unbound accounting precede protein sampling. Predicate
parity alone is insufficient to launch or interpret a protein SMC campaign.

The previously checked balance and Poisson theorems supply the mathematical
structure. Compiled-predicate fidelity, geometric predicates, floating-point
threshold execution and the extended record/audit implementation remain explicit
obligations; no new Lean proof is claimed for this code.


## Completed implementation checks

The protein predicate agrees with the independent Python definition on all 97
hard-valid geometric fixtures: 82 with native entry and 15 without it. All 14
motifs, reversed contacts, quaternion signs, transformed coordinate frames,
member-threshold perturbations and the two-anchor scaffold are covered. Full
motif, anchor and supporting residue-pair labels agree exactly. The largest
position/gap difference is 2.83e-14 Å; the largest near-zero acos angle difference
is 1.21e-6 degrees. All 119 attempted geometric fixtures are recorded, including
22 atomic clashes excluded from valid-pose parity. Missing or false hard-valid
assertions are rejected by the probe.

- [Cross-language receipt](../runs/native-entry-parity-20260924/validation.json)
- [Tested probe and source bundle](../runs/native-entry-build-20260924/validation.json)
- [Compiled protein definition](../runs/native-entry-compiled-20260924/compilation.json)

A nontrivial sphere reference isolates the restriction from protein geometry.
For a one-sphere body of core radius 0.3 and a single reference at `(1.5,0,0)`,
a sufficiently narrow rotational chart makes native entry exactly the translation
ball `|t-(1.5,0,0)| <= 2`. On a shell of radius `r`, the remaining direction
fraction is

```
f_minus(r) = clamp([1 + (r*r + 1.5*1.5 - 4)/(3*r)]/2, 0, 1).
```

Multiplying the independent radial/Haar integrand by this fraction yields exact
reference integrals (numerical quadrature), with depletant radius 0.4:

| Activity | Restricted total | Restricted contact | Unbound |
|---|---:|---:|---:|
| 0 | 0.004462706426615 | 0.001569924700053 | 0.002892781726561 |
| 4 | 0.005747109017931 | 0.002854327291370 | 0.002892781726561 |

Eight independent small populations per activity and bridge give 32 sphere
reference populations. Both physical-activity and proposal-density bridges
pass the numerical reference tolerance of six estimated population standard
errors. These are bounded implementation tests, not the stricter protein
convergence gate. Every initial native classification is separately compared with the analytic
ball inequality. The unchanged unbound integral and increased contact integral
also check the distinction between restricted normalization and contact mass.

These tests established the sampler prerequisite. The subsequent independent
output audit and fixed protein control are described below.

## Independent output audit

The new auditor reconstructs every attempted initial pose, including geometry
flags saved as false, both H/g and Hminus/g volumes, unconditional initialization
denominators, the physical proposal density and chart/Haar Jacobian. It compares
all operative native certificates with the original complete Python classifier,
reconstructs mutation weights, rejections, resampling and normalizer recursion,
and reports terminal region mass as `Zhat * terminal_indicator_fraction`.
Population errors use the four independent whole populations; terminal
particle descendants are not independent observations.

Seven focused tests cover both bridge laws, mixture initialization, zero cases,
strict near-zero geometry, 22 resealed history corruptions and exact reconstruction
of the compiled protein predicate from its original archived inputs. The fresh
release executable then generated 16 sphere populations plus one all-zero
fixture specifically for validating this new auditor. All 17 audits passed,
retaining all 65,792 initial attempts and checking 245,005 native certificates.
Total, contact and unbound integrals pass the predeclared implementation tolerance
of six population standard errors plus 1e-6. In particular, the z=0 unbound check
is 3.63 standard errors from its reference: this implementation tolerance is
explicitly broader than the scientific protein comparison criterion.

- [Frozen reference plan](../runs/native-excluded-smc-reference-20260924/plan.json)
- [Independent audited reference receipt](../runs/native-excluded-smc-reference-20260924/auditor-validation/receipt/validation.json)

Recorded hard/capture/R4-rejected mutation attempts retain counters rather than
candidate poses. The auditor therefore cannot geometrically replay those
unrecorded candidates. It verifies their accounting and all retained states;
the correctness of those skipped predicates remains an implementation obligation.
This limitation is explicit in every audit. Initial draws do preserve their poses,
and all initial validity flags are independently checked.

## Fixed restricted protein control

`tools/run_native_excluded_smc_campaign.py` freezes and executes three independent
arms with unchanged shape, scaffold, R4 region, native definition and physical
measure at radius 1.5 Å and activity 0.035 Å^-3:

| Arm | Populations | Particles | Translation steps (Å) | Rotation steps (degrees) |
|---|---:|---:|---|---|
| narrow | 4 | 2048 | 0.05 | 0.1 |
| large | 4 | 4096 | 0.05 | 0.1 |
| broad | 4 | 2048 | 0.2, 2.0 | 1.5, 15.0 |

Each population uses 262,144 attempted initialization draws from the same
normalized 50/50 mixture of the complete R4 and reference R5 chart balls.
Reporting masks on the reference chart do not filter that proposal. There are
128 bridge stages, four local attempts per particle per stage, two independent
clouds per potential and incremental cloud intensity ratio 128. Mutation-gate
intensity ratio remains 64, as in the existing physical configuration.

The fixed allocation has 3,145,728 initialization draws, 4,194,304 particle
potential evaluations and 16,777,216 local attempts. It cannot extend, replace
or retry populations. At most eight physical processes and 32 scientific
threads may run; subsequent independent audits use at most four workers.
Sources, executable and embedded source bundle, Python runtime, all inputs,
reference evidence and a prior-seed inventory are pinned before sampling.
The controller drains started children after any failure and starts no further
work; existing outputs and logs cannot be overwritten.

All original radial, angular and 64 orthant strata remain in the report. Contact
masses and hard-only contact volumes are compared with cached matching importance
estimates, without reclassifying or pooling old samples. The two comparison
requirements are separate: at most 0.2 difference in log mass and at most three
combined population standard errors in **linear mass**. Reported population
relative error must be at most 10% for the total restricted and contact weights.
A missing unbound observation remains unresolved and is accompanied by the
existing deterministic finite-R4 bound; it is not interpreted as zero mass.
Native mass is absent by construction in this control and is never used to
infer physical rarity. Importance ESS requirements remain with the original
importance campaign; SMC descendant counts cannot satisfy them.

Nine controller tests cover allocation, unconditional linear averages including
zero populations, both comparison gates, retained strata, prerequisite and
output authentication, resealed tampering, no overwrite and failure stopping.
The common resource-aware executor retains its previously validated worker-limit
and child-draining implementation. An independent review found no launch blocker.

The campaign launched from its frozen runner and is still an independent
fixed-scaffold denominator control:

- [Frozen control plan](../runs/native-excluded-smc-control-20260924/plan.json)
- [Launch receipt](../runs/native-excluded-smc-control-20260924/launch-receipt.json)
- [Current execution status](../runs/native-excluded-smc-control-20260924/status.json)

Completion does not authorize full-vessel or assembly production. Even agreement
of this denominator control cannot repair all native-region strata failures in
the completed importance campaign. The finite-system thermodynamic conclusion
remains unresolved.
