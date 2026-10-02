# Capped joint conditioning of dimer destinations

This construction targets the failure measured in the
[dimer destination screen](dimer-destination-probe.md): many independent draws
clash, while most hard-valid defensive draws discard all exclusion contacts.
Conditioning can avoid evaluating depletion at those destinations. It does not
by itself improve proposals per raw geometry test, establish physical acceptance,
or resolve the [failed stationarity control](dimer-one-step-stationarity.md).
This document specifies a fixed-context building block, not production assembly
validation.

## Fixed context, density and feasible set

Fix the selected root and child labels, every spectator pose, one spectator
anchor A, the wall, repaired shape, exclusion radii, atlas, defensive probability,
relative uniform cube and nonnegative integer cap K. Denote this context C.
None of these values may change within a retry sequence. For the reverse move,
the same labels and spectator configuration reconstruct the same C.

For the two physical poses x=(g1,g2), use

\[
h_0=A^{-1}g_1,\qquad h_1=g_1^{-1}g_2,\qquad
f_C(x)=F_0(h_0)F_1(h_1).
\]

Each F is the complete normalized frozen mixture, optionally
`alpha * U_L + (1-alpha) * G`. The two mixtures may differ if their choice is
fixed by C. U_L uses the relative translation cube and normalized Haar
orientation. Both branches enter every density regardless of the generated
branch. Product translation/Haar measure obeys
`dg1 dg2 = dh0 dh1`; the child is reconstructed relative to the **new root**.
The existing chart-to-Haar Jacobians remain inside each F.

Define D_C as the set of complete hard-valid endpoints whose selected root and
child have an exclusion contact. Hard validity includes both moving particles,
every spectator and the atomic wall. Internal exclusion contact uses the same
Boolean union-of-inflated-atomic-spheres predicate as the diagnostics. It is not
a native label, contact area or many-body overlap volume. The spectator part of
the physical configuration remains fixed throughout this kernel.

The source must also belong to D_C. A physical source outside D_C receives an
identity step in this channel. This is essential: otherwise a contact-free
source could enter D_C but have zero reverse probability. Other reversible
kernels must create or break the internal contact. Requiring a particular
external anchor contact as well would similarly exclude sources lacking that
contact; it cannot silently turn this construction into an attachment kernel.

## Complete retry law and balance

Write `p_C = integral_D_C f_C(x) dx`. Its value is unknown and need not be
estimated. For x in D_C, independently redraw **both** edges and all their branch
labels and latent variates up to K times. Return the first finite proposal in
D_C. After that first success, make one physical acceptance decision. If every
trial fails the geometric predicate, retain x. Do not keep a feasible first
edge while retrying its child; that is a different conditional generator.

Define the polynomial

\[
s_K(p)=\sum_{j=0}^{K-1}(1-p)^j.
\]

For x in D_C, the successful candidate subdensity and exhaustion probability are

\[
q_{C,K}^{\rm success}(x,y)=s_K(p_C)f_C(y)\mathbf1_{D_C}(y),
\qquad q_{\rm exhaustion}=(1-p_C)^K.
\]

The mass identity `p_C s_K(p_C) + (1-p_C)^K = 1` holds without division.
Consequently p_C=0 and K=0 are well-defined identity kernels. At p_C=1, any
positive cap succeeds on its first trial. On a forward/reverse pair in D_C,

\[
\frac{q_{C,K}^{\rm success}(y,x)}{q_{C,K}^{\rm success}(x,y)}
=\frac{f_C(x)}{f_C(y)}.
\]

The unknown retry factor cancels. Old zero proposal density is allowed: the
candidate then has zero reverse probability and must be rejected. A generated
candidate with nonpositive or unrepresentable evaluated density is a validation
error, not permission to change the stopping rule.

With a deterministic physical weight W, acceptance is
`min(1, W(y) f_C(x) / (W(x) f_C(y)))`, including any explicit selection or frozen
bias factors. For many-body depletion, use the existing exact auxiliary count
acceptance instead of evaluating or estimating this deterministic ratio naively.
For the full-union bath construction, its factor is

\[
\exp\!\left[\log f_C(x)-\log f_C(y)
 +(N_g-N_l)\log(1+z/\lambda)\right].
\]

Gained and lost solvent counts have their prescribed forward rates lambda and
lambda+z. The fair-order two-leg bath construction may also be used when its
reverse trace exchanges the legs and preserves the same intermediate state.
The retry success must precede either bath calculation, and bath/MH rejection
ends the outer move. Retrying after a rejected bath decision would condition on
a different, generally unknown law. No intermediate physical acceptance or
hard rejection may be inserted into the two-leg bath path.

The proof concerns **accepted flow**, not separate invariance of the proposal
and acceptance. Start with the already balanced accepted flow for independent
f_C draws and the correct physical/count gate. Multiply it by

\[
c_C(x,y)=\mathbf1_{D_C}(x)\mathbf1_{D_C}(y)s_K(p_C).
\]

This factor is symmetric, so accepted flow remains symmetric after integrating
all auxiliary counts. Its row mass is at most `p_C s_K(p_C) <= 1`. Completing
that subkernel with its missing diagonal mass gives a Markov kernel reversible
for the full physical target. Sources outside D_C simply contribute identity
steps. This argument does not require the target normalized within D_C, or
compute the probability of D_C under equilibrium.

## Numerical nulls are not automatically geometric failures

The narrow safe implementation retries only after a successful finite raw draw
fails the declared hard/contact predicate. Record all raw variates, decoded
poses, predicates and failed trials. Unexpected decoder, pose, density,
Jacobian or geometry errors must terminate the validation run with its trace;
they must not silently become another trial or a completed valid observation.
Continuation from an incomplete event requires the saved RNG/event state and
an explicit repaired-executable provenance, not a replacement draw.

Even terminating that outer move as a self-loop is not a general balance proof
for arbitrary numerical errors. A source-dependent failure can remove outgoing
moves while reverse arrivals remain possible. A symmetric endpoint-pair guard
can legitimately add rejection, but symmetry must be established rather than
inferred from an error message.

A documented invariant null atom is a possible extension, subject to stronger
requirements. If a raw generator has successful subdensity `a_C f_C(y)` and a
null probability `1-a_C`, where a_C depends only on invariant C, then a null may
count as a failed raw trial. Its unknown common factors still cancel. But a
source-independent decoder failure alone does **not** establish this form: if
failure discards only some latent preimages, the surviving density can be
`f_C(y) r_C(y)` with an endpoint-dependent r_C. The scored density must then
include r_C, unless failure is exactly a deterministic endpoint predicate
included in D_C for both source and destination. Neither arbitrary missing
components nor source-tail encoding failures may be hidden in this exception.

The current independent drawer does not use the old pose to generate new
edges, which is helpful, but its complete wrapper also calculates source and
endpoint densities. Keeping decode, geometry classification and density errors
distinct is therefore an implementation obligation. The
[prepared-factor correction](docking-factor-closure.md) now shares the actual
map coefficients between generation and scoring; floating-point coordinate
round trips remain a separate obligation.

## Implemented fixed-context kernel

[`capped_dimer.rs`](../src/capped_dimer.rs) implements this channel without a
physical acceptance or assembly runner integration. Its context owns an immutable
spectator snapshot and constructs the atomic wall from the same core shape.
It verifies that the supplied exclusion tree encloses the corresponding core
atoms. Hard-invalid sources are errors; physical sources without internal
contact self-loop. Cap exhaustion and successful candidates have distinct status
values. Every whole-joint raw trial is retained, including failed geometry.

Decoder, scoring and unsupported-arithmetic failures return a distinct fatal
error containing completed trial records. A runner must persist it together
with context/configuration identity and seed or RNG provenance, then stop.
Such errors are never silently converted into a retry or an MH rejection.
The current implementation evaluates full proposal densities on every raw draw;
amortizing repeated source scoring remains an optional runtime optimization.

All eight [focused tests](../tests/capped_dimer.rs) passed: full hard/contact/wall
predicates and tangencies, source-outside-contact and zero-cap self-loops,
complete cap exhaustion, exact raw-prefix stopping and density ratios,
fatal numerical-null trace retention, invalid inputs, seed-prefix replay with
immutable spectators, and finite-state accepted flow with rejection completion.
Replay reconstructs the random stream from its seed and prefix; this standalone
module does not supply a serialized RNG checkpoint or production restart format.
An initial test compilation used an unavailable `StdRng::clone`; the corrected
test uses seed replay and changes no scientific allocation or kernel behavior.

## Production selection still needs a reverse law

The fixed-label test has selection ratio one. It does not license selecting an
arbitrary current connected component and treating that selection as fixed.
External contacts and component membership may change on this move.

Two practical routes are available:

- Choose root, child and anchor from an endpoint-independent distribution, with
  self-loops for sources outside D_C. This is simple but may waste selections.
- Use a constant-rate channel for each labeled internally contacting pair. Its
  internal-contact eligibility remains true on every successful endpoint, so a
  rate `kappa * 1_internal_contact` is unchanged along that channel. The existing
  fixed-duration subset-rate argument can then apply. Pair rates depending on
  contact area, graph degree, full component membership or detailed internal
  registration do not inherit this cancellation.

With either route, retain the selected anchor or ordered anchor pool literally.
If its probability depends on the current pose, include the same label's
`log p(anchor | y) - log p(anchor | x)`, with all without-replacement factors.
The retry factor remains common conditional on that retained context. Redrawing
the anchor between failed attempts produces a different mixture with
anchor-dependent success normalizers and requires another derivation. The
complete defensive mixture does not have a special selection-free uniform
branch unless the actual generator was separately constructed that way.

A state-dependent total event rate must retain its residence times and fixed
algorithmic phase duration. Counting a fixed number of such selected events
instead generally weights the invariant measure by the total rate. Native
labels and observed acceptance must not alter geometry-only selection.

## Existing Lean bridge and its limits

No new algebra theorem is needed for this fixed-context extension. The checked
[`CappedIndependent.lean`](../formal/ReversibleSampling/CappedIndependent.lean)
results instantiate directly:

- `capped_mass_identity` and `capped_mass_bounds`: substitute z=p_C and t=K.
- `empty_success_set`: covers zero feasible proposal mass without division.
- `shared_factor_accepted_flow`: use c=s_K(p_C) for ordinary deterministic MH
  flow on a pair in D_C; directed masses include the physical and selection
  factors as appropriate.
- `shared_factor_preserves_balance`: applies to the already count-marginalized
  accepted-flow identity. Off D_C, both directed accepted flows are zero.

Integration then supplies the reversible accepted subkernel required by
[`complete_markov`, `complete_reversible`, and `complete_invariant`](../formal/ReversibleSampling/Balance.lean).
Do not apply the density-based MH theorem to the whole capped proposal by
pretending its exhaustion atom has a translation/Haar density; the atom is
singular. The completion theorems handle that diagonal mass directly.

The existing [focused receipt](../formal/capped-independent-validation.json)
and [axiom audit](../formal/capped-independent-audit.txt) were reused. The
current capped source and audit hashes agree with that receipt:

- Source SHA256: `3f524a987f62d1738ee37e1c2bfd1285c76b98f6244105fec40ca56c921c0caa`.
- Audit SHA256: `2bcff85584d2cce3eaa52bfe9c207db3dbb79d100373061d0567ff4c44a9f286`.

Only `propext`, `Classical.choice`, and `Quot.sound` occur in its listed axiom
dependencies. This reuse proves the scalar retry and flow algebra. It does not
instantiate the actual SE(3) measure, measurable sphere-union geometry, Rust
retry loop, exact Poisson thinning, normalized Gaussian generator, complete
Jacobian, floating-point arithmetic or RNG. No new Lean build was required or
claimed by this review.

## What would count as a benefit

For identical independent raw trials with feasible probability p_C, success
probability is `1-(1-p_C)^K` and expected trial count is s_K(p_C). The ratio of
successful candidates to raw trials remains p_C. Gains can come from amortizing
context preparation or avoiding expensive bath calls at contact-free endpoints;
large caps alone do not improve the proposal distribution conditional on success.
The pooled panel has different contexts and does not justify inserting its
pooled feasible fraction into this formula as if every context had the same p.

Validation should first check finite-state balance, zero/full success sets,
whole-joint stopping, deliberate asymmetric-error failures, restart trace
retention, complete proposal density and reference sphere/depletion laws. A
matched cap comparison must retain every outer rejection and report raw trials,
useful destination types, exact bath cost and completed contact-environment
changes per CPU. Native-informed and frozen geometry-only atlases remain
separate arms. The existing stationarity failure must remain visible rather
than being erased by this new control.

This channel can reorganize and dock an existing dimer while preserving its
internal exclusion contact. It does not force many-body overlap preservation,
native registry, external attachment or thermodynamic stability. Only later
contact ESS, independent initialization agreement and converged finite-system
physical evidence can establish those outcomes.
