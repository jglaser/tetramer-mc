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

An independent restricted-output auditor must reconstruct this membership and
the complete normalizer recursion. The existing unrestricted auditor must not
silently accept restricted outputs. Before protein production, the extension also
needs zero/finite-activity reference integrals, all-zero initialization and
unrestricted-default regression checks, a frozen independent allocation, and
explicit analysis of contact versus unbound mass. Passing a predicate test is
not sufficient to launch or interpret a protein SMC campaign.

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

These tests establish a usable implementation prerequisite. No restricted
protein population has been launched. An independent production-output auditor
and a separately frozen protein allocation remain necessary before that step.
