# Flexible surrogate benchmark design

The implementation, reference checks, and fixed allocation are complete. The
prepared campaign is `/vast/xvg/tetramer-mc-runs/flexible-surrogate-dimer-context0-20261004`;
its execution receipt is authoritative for launch/completion status. It keeps the
context0 conditional target: mobile labels `[27,132]`, all 262 other labels fixed,
the original scaffold and shape, `rd=1.4 Å`, activity `0.0275 Å^-3`, and the original
spherical wall. It does not address full-system assembly or physical kinetics.

## Matched allocation and integration

The implemented arms are `flexible_m1`, `flexible_m8`, and `flexible_flat8`: respectively one/eight inner
steps with guidance strength one, and eight with strength zero. Use the original
two initialization types and four streams: 24 new chains, each with 512 warmup
and 4096 production blocks. Reuse the four prepared starts and eight family cloud
banks (16384 raw points each); generate no new starts or clouds. Keep the original
`.2 Å / 1°` proposal scales and four local attempts `[0,1,0,1]` before one flexible
attempt per block.

`examples/evolving_dimer_benchmark.rs` requires an explicit
`flexible_surrogate_policy` with schema `flexible-surrogate-policy-v1` and
`proposal_scales: {source: local}`. Reuse the existing scale type, cloud loader,
raw-count quadrature weight, checkpoint machinery, and seven-row block layout.
Dispatch `FlexibleSurrogateKernel` without a rigid handle. Preserve every old arm
and RNG role. New roles `flexible_surrogate/{proposal,inner_accept,bath,accept}`
are shared across the three new arms; local roles remain unchanged. Global draws
are not matched to rigid arms: random scan and the two-leg bath consume different
random variables.

Labels remain fixed throughout each chain. Each inner attempt chooses one member
with probability one half; no contact-dependent eligibility or pair selection is
introduced. Hard/MH rejections consume a step. The final nonidentity endpoint gets
one fair-order two-singleton bath and one decision with
`bath.aggregate.log_weight + S(old) - S(new)`. The copied bath intermediate may
violate hard cores or the protein wall and must not be filtered.

## What the cached observations actually contain

`ConditionalObserver` in `tools/analyze_evolving_dimer_benchmark.py` saves
`patch_tokens`, `partner_edges`, `internal_contact`, `external_edges`,
`partners_by_member`, and a whole-token-set fingerprint, plus block/CPU metadata.
A completed two-root cache was checked for this exact schema. It contains no body
poses, relative transforms, or native motif labels.

The tokens are `(global_i, global_j, patch_i, patch_j)`, not just particle-pair
edges. The frozen `body-frame-atom-patch-map-v1` assigns all 4004 atoms to 32 labels:
four monomer members times eight octants in each archived monomer body frame.
For example, a token can be
`[27,132,"member0:octant4","member3:octant4"]`. The octant rule is
`4*(x>=0)+2*(y>=0)+(z>=0)`, with no recentering or native-label filtering. A token
exists if at least one associated atom pair has strictly overlapping inflated
spheres. See `build_patch_map` in `tools/prepare_finite_assembly_observer.py` and
`ContactObserver.classify` in `tools/analyze_contact_efficiency.py`.

Therefore the cached internal token set can change while the binary internal
edge remains present. It supports **coarse internal contact-patch reorganization**
metrics without old geometry queries. It does not uniquely determine relative
pose, contact multiplicity, native registration, or motion within unchanged
patches. Threshold flicker alone can change the set; these are not established
binding basins or native-registry transitions.

## Implemented observation and possible extensions

The implemented observer reuses immutable score/count helpers and retained-state
replay, and validates flexible scan labels, unchanged nonselected members, fixed
horizon, both bath legs/order/copied intermediate, aggregate counts, and the
single outer correction. Keep fatal/null/rejected records and authenticate complete
terminal and checkpoint closure. Do not apply the rigid reconstruction check.

For every retained endpoint, partition cached tokens into the internal pair and
external contacts. Report internal edge occupancy, attachment/detachment, internal
patch-set occupancy and direct nonempty changes/returns separately from external
nonempty changes/returns and whole-fingerprint apparent ESS. Include block512 as
the transition baseline, preserve all production residence, use full sampler CPU,
and keep constant-descriptor ESS null.

Continuous relative-pose metrics remain a possible extension, not part of this
frozen contact observer. The smallest such measure uses the
already validated new poses: `g = inverse(X27) * X132`. Save its translation and
proper rotation, then translation displacement and quaternion-sign-invariant
rotation angle relative to fixed initial and block512 references. These require
pose arithmetic only. They detect rearrangements hidden by coarse patch labels,
but still do not identify native registry. Old contact caches cannot supply these
pose metrics; obtaining them for controls requires a separately declared saved
journal pass with its cost recorded. Avoid assigning basin thresholds after seeing outcomes.

If the decision requires **native registry**, use the separately validated frozen
native classifier and explicit shape/catalogue bindings. One internal-pair call
per new retained endpoint is a bounded possible extension (at most 110616 calls),
with all matching motif labels retained. It is not included in this contact observer;
cycle consistency requires the corresponding external
native graph as well.

## Reuse and comparison scope

Authenticate completed rigid/contact analysis receipts before consuming caches.
Reuse the 24 rigid and eight local chains as primary matched controls; the eight
cached m4 chains can remain a secondary comparator. Include each historical chain
once, yielding 24 new plus 40 cached chains. Classify only the 110616 new retained
endpoints, with a worst-case cap of 58073400 moving-related pair classifications.
No old geometry, physical draws, or control resampling is needed for contact
metrics. Keep four streams per arm/start, initialization disagreements, shared-RNG
dependence, and actual CPU costs visible; do not concatenate trajectories or infer
equilibrium from apparent ESS or contact returns.

Before any protein allocation, freeze the new build, policy and complete plan;
require the independent flexible stationarity reference and its journal audit,
focused observer mutations, and actual compiled synthetic CLI/checkpoint replay.
Use the existing single-worker lifecycle and explicit fatal resource caps. The
rigid precedent uses 1800 CPU seconds/3600 wall seconds/8 GiB per chain; the new
plan must bind its limits before execution, not relax them after observing a tail.

## Frozen validation and campaign

The release build passed nine kernel tests and twenty compiled example tests,
including all new arms, both starts, restart equality, fatal tails, and unchanged
local proposals. Seventeen Python tests cover the observer and preparer. The
independent physical-reference run and journal audit were reused without new
reference draws. That reference used the debug profile; the unchanged runtime
source was tested and compiled separately in release. No production executable
was replaced.

- Release validation: `results/flexible-surrogate-benchmark-validation-20261004/validation.json`,
  SHA256 `40643c8ff369725cf5d4ad225d1ffca6986fde6409f338d470bd602da75ef148`.
- Python validation: `results/flexible-surrogate-python-validation-20261004/validation.json`,
  SHA256 `a2b0c5ad6ba74cf938416796cb369783aa73c6c160ee3064e0837b19d9940864`.
- Frozen execution plan under the campaign root: SHA256
  `7f47cf66893284b048e4b1edbbd61cbc4361be1948c90317be95ce403499deb9`.
- Protocol: SHA256 `b8a709c6d21443634650055b48acd40eb0dc970c4d494f39cd01b26ab8a924ad`.

One worker executes the 24 chains serially. Each chain has fixed caps of 1800 CPU
seconds, 3600 wall seconds and 8 GiB. The full observer runs only after the complete
inventory passes; failure stops the campaign without replacing a chain. The new
observer measures internal patch changes and external environments, but performs
no native-registry classification. Protein efficiency remains unmeasured until
that observer completes.
