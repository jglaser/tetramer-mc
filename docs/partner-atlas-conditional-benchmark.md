# Conditional partner-atlas benchmark

The optional integration in `examples/evolving_dimer_benchmark.rs` adds
`partner_atlas_direct`, `partner_atlas_m1`, `partner_atlas_m8`, and
`partner_atlas_flat8`. This document defines the conditional comparison; the final section records
its validated launch. Existing arms, their RNG roles,
and their serialized contracts retain their previous behavior.

## Fixed proposal and target

Each new arm requires `partner_atlas_policy` with schema
`partner-atlas-policy-v1` and `proposal_scales: {source: local}`. The arm fixes
the horizon and strength: direct `(1,1)`, guided m1 `(1,1)`, guided m8 `(8,1)`,
and flat8 `(8,0)`. The direct method does not evaluate a surrogate score or draw
an inner acceptance coin; its nominal strength is unused. Unknown policy fields
and policies attached to another arm are rejected. The example also supports an
explicit frozen scale override for bounded synthetic validation; the proposed
protein comparison keeps the original local scales.

Use context0, mobile labels `[27,132]`, all 262 other tetramers fixed, the original
shape/scaffold/wall, `rd=1.4 Å`, activity `0.0275 Å^-3`, and lambda/activity 64.
These are the conditional growth controls, not the original decision conditions
`rd=1.5 Å`, activity `0.035 Å^-3`. No tetramer is internally deformed: internal
rearrangement here means changing the relative pose of the two selected rigid
tetramers.

The actual proposal is the existing **native-blind FFT atlas**, not a
native-informed proposal control. The bound conditional `config.json` points to
`results/evolving-dimer-campaign-20261003/common/inputs/atlas.json`, SHA256
`f6be7889418ae39c078719c64af989aaa7c7bbbe004989badaa56d34121b8e49`.
It is the exact tau=.25 covariance scaling of the original atlas, SHA256
`c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08`;
`prepare_evolving_dimer_benchmark.verify_covariance_transform` defines that
transform. The reciprocal model has 1024 base components and 2048 virtual charts,
with angular length 55.02283113084892. This proposal provenance does not imply
that the historical scaffold/source generation was native-blind.

`Geometry::load` constructs the open, nonperiodic `PosteriorInvolution` model
with correlation 0, uniform branch probability .1, center `[0,0,0]`, and cube
side lengths `2*(wall_radius+core.bound)`. `SphereTree::new` defines `core.bound`
as `max(norm(atom.center)+atom.radius)`. The initial contract records the exact
constructor cube/center/uniform weight and the live model's method, correlation,
chart count, angular length, and periodic flag; no rounded geometry bound is used
to generate that contract.

## Kernel and record contract

Each block retains the four original local attempts `[0,1,0,1]`, with
`.2 Å / 1°` scales and unchanged `local/{attempt}/{proposal,bath,accept}` roles.
One additional attempt follows. The new roles
`partner_atlas/{proposal,inner_accept,bath,accept}` are shared across the four
new arms and separate from previous collective families. Restart reconstructs
the same role streams by block; the arms are not independent replicates.

Every inner attempt selects a slot fairly, then chooses atlas with probability
.25 or local with probability .75. The atlas call proposes only the selected
body against the **current** unselected body. The retained uniform/involution
branch uses the helper's complete correction; the learned branch includes its
complete chart density and expanded-map/label correction. This is not an
unconditional uniform-plus-learned mixture-density ratio. The kernel checks
immutable uniform reverse support. Null, zero-reverse-support, hard and MH
rejections consume their attempted slot and retain residence.

Delayed arms apply `S(new)-S(old)+log(q_reverse/q_forward)` at each inner gate;
their sole final physical gate adds `S(source)-S(endpoint)` to the fair-order
two-singleton bath. Direct applies one mode/slot candidate and
`bath.log_weight+log(q_reverse/q_forward)` without an inner filter. The copied
two-leg midpoint is not hard-filtered. Identity endpoints spend no bath; fatal
errors preserve the full attempted record and leave the physical state unchanged.

The initial row has `partner_atlas_contract`, schema
`evolving-dimer-partner-atlas-v1`, including the bound atlas, reused preparation,
body-frame cloud, full quadrature weight and effective config. Each block has
seven rows: four `local`, durable `partner_atlas_attempt_begun`, durable
`flexible_partner_atlas_direct` or `flexible_partner_atlas_chain`, then
`retained_block`. Begun records distinguish `inner_filter`; outcomes include the
full library trace plus block, physical retained pair, CPU and any fatal error.
The common dimer counters count outer attempts/acceptance/self-loops. Inner mode,
slot, null/rejection, proposal, and bath-work counters remain in the full trace.

## Proposed allocation and observation boundary

The intended matched allocation is 32 new chains: four arms × two existing start
types × four streams, 512 warmup plus 4096 production blocks. Reuse four prepared
starts and eight family clouds; generate no replacement starts or clouds. The
eight cached local chains are the primary historical baseline. Other completed
cached controls can be shown once as context, with their original provenance
and CPU, without pooling them into extra independent replicates.

Every retained endpoint, including rejections and absent states, is required.
For each stream/start report actual internal native motif-set occupancy,
nonempty transitions/returns and initialization disagreement, alongside internal
contact/patch-set reorganization and external exchanges. Keep all matching
native motif labels. Relative-pose arithmetic can reduce
`inverse(X27)*X132` from the validated retained poses, with fixed initial and
block 512 references and quaternion-sign-invariant angles; patch changes alone
do not establish such motion or native registration. Native classification is a
separate frozen classifier operation, never a proposal or sampler decision.

Use complete sampler CPU, candidate and mode opportunities, hard/null/inner/outer
rejections, and raw/retained PPP work when comparing efficiency. An m8 block has
eight opportunities, not one matched-cost proposal. Constant descriptors have
undefined ESS; rare occupancy must accompany any apparent ESS/CPU. This finite
conditional comparison cannot establish equilibrium weights, full-vessel
assembly, or physical rates.

Legacy readers deliberately reject the new family. The new
`analyze_partner_atlas_benchmark.py` provides an explicit adapter for the
contract and begun/outcome kinds, extending the roles of
`analyze_flexible_surrogate_benchmark.{effective_config,validate_inventory,
surrogate_contract,validate_surrogate,canonical_rows,validate_journal}`. It reuses
the path, score, local-residence and checkpoint arithmetic with the new arm
semantics. The separate `analyze_partner_atlas_native.py` now supplies the
new-family endpoint and admission adapter; its old seven-row reader is not
interchangeable merely because row counts match.
Independent partner-reference audit helpers supply expanded proposal correction
checks, with global labels mapped only after validating the selected pair.

The example has 23 focused tests (20 existing plus 3 new). The new tests cover
explicit fixed policy, separate deterministic RNG roles, all four arms and both
starts, continuous versus disk-checkpoint equality, unchanged local-control
records, actual model bindings, current partner locality, direct versus delayed
correction placement, and fatal/uncheckpointed-tail retention. The revised
example was validated in an isolated target before freezing the
producer/observer/preparer allocation. No protected release binary or
production sampler is changed by this integration.

## Validation and launched allocation

The isolated release validation passed six partner-atlas tests, nine existing
flexible-kernel tests and all 23 example tests, then built the example. The
first build failed only at JSON macro expansion; its source and receipt were
preserved before increasing the example's macro recursion limit. No physical
reference or protein sample was generated by that failed compile. The passing
receipt is
`results/partner-atlas-benchmark-validation-20261004/attempt02/validation.json`.
All sixteen Python observer/preparation tests passed under
`results/partner-atlas-benchmark-tools-validation-20261004/attempt01/validation.json`.
The nine observer tests include independent cube/chart metadata reconstruction,
direct/delayed corrections, malformed inverse traces, nulls, uniform reverse
support, retained residence and relative-pose diagnostics.

The allocation above was frozen and launched at
`/vast/xvg/tetramer-mc-runs/partner-atlas-dimer-context0-20261004`:

- protocol SHA256: `8f34cbcff8fecfcb21579a32e52a9cad2513882f830d69d138dc075f53c50d7d`;
- execution-plan SHA256: `e69b87a7b4b46657e5cef6a70da049c6c259e7688c444268eaa075e9565bef93`;
- executable SHA256: `3050f605ad6ca04c5ae89df0169a0b7d58fbf290791787ef8f5ea1fdee2f6fab`.

There are exactly 589,824 local attempts, 147,456 additional outer calls and
663,552 inner candidates, with all self-loops counted. One worker runs one
thread; each chain has a 1,800 CPU-second, 3,600 wall-second and 8 GiB cap. A
fatal cap preserves the partial chain and stops the queue, with no replacement.
The launcher checked global capacity and the unchanged production executable.
Its first host preflight encountered an unreadable executable symlink for
`sd-pam`, before creating a controller or sampling claim; the retained second
preflight used readable command metadata and launched the same frozen plan.

The contact observer and native classification definition are frozen separately
from proposal generation. The saved native plan binds the existing classifier,
compiled motifs, atom-identity witness and all fifteen dependencies. No native
label is used for retaining a trajectory, selecting a proposal or choosing a
sample. Interpretation waits for the complete allocation and its audits.

The native adapter passed eight synthetic tests under
`results/partner-atlas-native-validation-20261004/attempt01/validation.json`
(SHA256 `091289bd7801294721cfd05d2f2a8712287458993d875dfb2931450f8f80ce34`).
They check the 32-new/16-cached join, all four record families, rejected and null
residence, all matching motif labels, exact-pose caching, query caps, full CPU
and undefined ESS, completed lifecycle admission, and preservation of failed
prefixes. The test run made no protein geometry queries. The adapter requires
the completed 72-chain contact/arithmetic report first; it observes only the
internal pair and makes no external-native or cycle-consistency claim.

## Decisions after completion

Show every stream and both starts. Repeated native entry, exit and return from
the prepared starts would address the earlier zero-native limitation; a single
occupied frame or patch flicker would not. Motif-set transitions and relative
poses distinguish exchanges of registry from repeated visits to the same motif.
Persistent initialization disagreement remains a sampling limitation.
Missing returns alone cannot distinguish a poor proposal from negligible
equilibrium weight. Do not require equal directional rates or frequent returns
to a region whose physical weight is negligible; that interpretation needs the
independent weight calculation on a matching target and domain.

Compare native residence and nonconstant apparent ESS per **full sampler CPU**
against the eight local controls. Compare direct with guided m1 to isolate the
inner filter, m8 with flat8 to assess guidance, and m8 with direct to assess the
additional chain work. Include failed candidates and Poisson work. Favor the
simpler direct move if the inner chain adds no useful sampling. These arms share
RNG roles and are not additional independent populations.

Consistent internal-registry improvement could justify optional all-mobile
integration with the existing pair-selection correction. It would not establish
external contact exchange, assembly, or equilibrium stability. Neither an
unsuccessful pilot nor a rare return refutes the physical model; the independent
physical-weight and finite-system checks remain necessary.
