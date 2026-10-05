# Correlated transport in the occupied neighborhood

This completed control tests whether retaining within-chart coordinates improves
physical contact-environment sampling. The completed independent-redraw pilot
already supplies the local and correlation-zero baselines. It need not be
repeated. The frozen design and its validation are retained below, followed by
the results. No further protein allocation follows from this document.

The [completed comparison](../results/context-prior-physical-comparison-20261005/comparison.json)
found 13 hard-valid learned proposals out of 8,226 production proposals with
the original prior, and 467 out of 8,226 with the context prior. Those yielded
one and two accepted learned moves, respectively. There were no completed
A-to-B-to-A returns. Context weighting reduced disagreement between starts
for neighbor sets, but complete patch fingerprints remained disjoint. Thus
center availability alone did not produce a consistent improvement in
independent contact samples per CPU.

The hypothesis is that independent redraws lose useful deviations from a
Gaussian center. A physical pose constrained by several neighbors can occupy
atypical coordinates within its selected pair-contact chart. Correlated
transport may carry those coordinates into a competing chart. Whether those
coordinates have compatible geometric meaning across charts is an empirical
question: a Cholesky basis is not a learned correspondence between surfaces.
No refit or native reference is introduced by this control.

## Existing move and balance

For fixed spectators `C`, let `g_i` be the existing normalized pose-chart
density and let `G_C(x) = sum_i w_i(C) g_i(x)`. The original and context
priors each retain all 2,048 reciprocal virtual labels. The moving body is
excluded from `C`; the anchor and prior remain fixed during a move.

Select the source label from `w_i g_i(x)/G_C(x)` and the destination label
from `w_j`. Encode the source pose as whitened coordinates `u`, draw
`eta ~ N(0,I_6)`, and use

```text
s = sqrt((1-rho)(1+rho))
v = rho*u + s*eta
eta_reverse = s*u - rho*eta
y = decode_j(v)
```

Swap the labels for the reverse move. This map on `(u,eta)` is an
orthogonal involution. Its Gaussian norm identity, physical chart Jacobians,
and posterior label probabilities give the same correction
`log G_C(x) - log G_C(y)` at any permitted correlation. Exact reciprocal
pose wrappers remain part of each virtual chart. The existing Poisson
gained/lost-count acceptance factor remains unchanged.

At `rho=0.95`, `s≈0.31225`, and
`E[||v||² | u] = 0.9025 ||u||² + 0.585`. The proposal retains much more of
the source residual than a redraw. It is still stochastic. An inter-chart
move can still be large, and favorable proposal geometry does not guarantee
favorable physical acceptance. The learned transition at nonzero correlation
is not an independent draw from `G_C`; an independent-mixture density ratio
for `0.5 U + 0.5 G_C` must not replace its branch-specific correction.

The production runner already supports this correlation and binds it in
checkpoints. The held production executable can be reused byte-for-byte.
Its correlated continuation and failure tests exercised `rho=0.65`; generic
involution checks cover other correlations and reciprocal labels. Those
completed checks remain evidence and should not be rerun merely to change
the parameter.

## Smallest matched protein control

Prepare 16 new chains: original and context priors, each from both certified
starts with four streams. Set only the tested transport correlation to 0.95.
Reuse the existing repaired shape, atlas, positive context prior, fixed
263-body environment, anchor 16 and moving label 77. The two starts remain
the saved pose with neighbors `{16,217}` (A) and certified branch 32 with
neighbors `{16,56}` (B).

Copy the eight existing start/stream seeds into the new configurations, using
the same seed for the two new arms and their completed controls. These are
matched random streams, not identical trajectories or independent arms;
hard decisions and accepted moves can cause subsequent random consumption
and state histories to diverge. Streams within an arm remain independent.

- 256 warmup and 2,048 production cycles per chain.
- Four unchanged local slots and one tested slot per cycle; local translation
  standard deviation 0.2 Å, rotation standard deviation 1°, and rotation
  probability 0.5.
- Defensive uniform probability 0.5, with the same full atomic spherical
  wall and origin-centered cube. The uniform branch is unchanged.
- Depletant radius 1.5 Å, activity 0.035 Å⁻³, and auxiliary intensity ratio 64.
- 11,520 attempted moves per chain, 184,320 new attempts in total; every
  rejection and any fatal prefix retained.
- Proposed unchanged per-chain limits: 1,200 CPU seconds, 2,400 wall seconds,
  8 GiB, one thread, and the previous raw/retained-point caps. At most four
  new physical jobs run concurrently. No retries, replacement streams,
  outcome-dependent extension, or additional correlation sweep.

Do not rerun local controls, correlation-zero controls, the source/alternative
certification, prior construction, or completed body-attribution calculations.
New manifests must reference their immutable results and hashes. New identity
metadata and output directories distinguish correlation 0.95 without changing
any old configuration or checkpoint. The neighborhood is the historical
500 μM conditional context, not the 106.8 μM all-mobile assembly target.

## Missing validation before launch

1. **Independent correlated arithmetic.** Exercise the exact 0.95 map using
   ordinary and reciprocal branches, noncommuting orientations, and moderate
   and atypical finite latent vectors. Reconstruct `v` and `eta_reverse`,
   recover the old physical pose, verify Gaussian norm conservation and the
   scaled Jacobian/noise/label cancellation. Retain the existing floating-point
   seam and fatal-error obligations. This adds the untested parameter case;
   it does not require a new production kernel or a repetition of old tests.
2. **Actual-bath reference.** The completed sphere reference hardcodes zero
   correlation. Use the same 8,192 cached IID sources, with a fresh,
   predeclared role-seed namespace, for the two correlation-0.95 prior arms:
   16,384 new one-step physical calls and no new source sampling. Preserve
   all sources and hard-invalid proposals as rejections with unchanged
   retained source, with the same 12 binary observables
   and the predeclared 24-test familywise criterion. An omitted-proposal-ratio
   control can reuse each candidate, cloud and acceptance uniform; report its
   sensitivity without demanding a detectable effect from an underpowered
   negative control. Freeze allocation and thresholds before these calls.
3. **A new observer contract.** The old observer explicitly requires
   correlation zero and must stay frozen. A new analyzer must authenticate
   0.95 in invocation, summary and checkpoint; retain the same all-state
   geometry and metrics; and audit the correlated forward and reverse noise
   formulas. The independent reference must decode the candidate from `v`,
   not from fresh noise alone. Hard rejections retain their proposal ratio;
   only bath-derived fields are absent. Bind this schema to the held compiled
   shared-step source, as in the corrected existing observer validation.
4. **Frozen comparison admission.** Validate 16 unique new identities and
   pair them with the exact completed baseline identities. Reject mixed
   correlations, changed contexts or filtered attempts. Check controller
   inventory, fixed resource limits and continuation compatibility before
   launch. Existing validated lifecycle code should be reused; only genuinely
   new inventory or audit paths require additional tests.

The new reference example and analyzers should have separate filenames and
isolated validation/build artifacts. No held source, old observer snapshot,
production binary or running contact-weight input may be modified. Reference
failure leaves the protein control unlaunched; it does not trigger replacement
sources or new thresholds.

## Decision criteria

Reuse the [complete metric specification](../results/context-prior-physical-pilot-20261005/metric-spec.json)
and [A/B exchange definition](../results/context-prior-physical-pilot-20261005/exchange-addendum.json).
Every production endpoint retains its unconditional physical weight. Observe
full atom-patch contact fingerprints, patch-token and neighbor presence, and
both elementary-attempt and cycle-endpoint cadences. Warmup contributes to
CPU cost but not production occupancy. Constant traces keep undefined ESS.

Report every stream's apparent ESS per full invocation CPU and paired ratios
against its same-prior zero-correlation control and the existing local control.
Report undefined pairs without imputation. Compare all cross-start support
and occupancy distributions, alongside within-start variation. Keep full
fingerprint disagreement visible even if neighbor-set disagreement improves.
Preparation, reference and offline-observer CPU are separate costs; no
unmeasured all-mobile amortization is assumed.

Count five-cycle-confirmed A/B passages through any intermediate neighbor
sets, directional passages and completed returns. A successful directional
transfer is useful evidence of accessibility, but not convergence. Unequal
equilibrium weights need not give equal transition rates or frequent returns.
No missing reverse passage can decide between physical rarity and inefficient
proposals without independent conditional weights.

As mechanism diagnostics, retain source/target latent norms, physical
translation and rotation changes, endpoint feasibility, proposal correction,
bath weight, and accepted versus rejected contact changes. Saved zero-
correlation traces can supply their side of these diagnostics without new
geometry or sampling. Better clearance or smaller steps alone do not pass the
test: the question is whether full contact-environment sampling per CPU and
initialization agreement improve. Any finding remains conditional sampling
evidence, not native-registry, equilibrium assembly or model-instability
evidence.

## Validation completed and control admitted

The isolated build passed both new Rust tests, including all 54 forward and
54 inverse map cases and 162 prior-correction identities. It used 88.15 total
CPU seconds and left the production executables and reused cache unchanged.
The new Python reference/observer contracts passed 21 tests. One initial
synthetic fixture used a NumPy scalar where the saved-JSON schema requires a
native scalar; that failed attempt was retained and the corrected JSON-roundtrip
fixture passed. No physical samples were replaced. Three further comparator
tests bind declared seeds to the actual executed configuration hashes.

The [correlated reference audit](../results/context-correlated-reference-preparation-20261005/audit.json)
then passed all 24 predeclared equilibrium checks using all 8,192 cached IID
sources and 16,384 new one-step physical calls. The smallest paired-test
p-value was 0.0410, above the 0.002083 per-test cutoff. The separate omitted-
proposal-ratio control failed 6 of 12 checks. Independent reconstruction
passed 286,862 numerical checks, with maximum absolute error `4.12e-9` and
maximum tolerance fraction `0.00012945`. All 49,152 events and all rejected
states were retained; both execution stages drained normally.

Audit SHA256:
`afc226f4986ec6382ebed122dea5c7a046b55da95c6ecd3968b863497e5d377d`.
The [frozen protein controller](../results/context-correlated-control-execution-20261005/controller-manifest.json)
has SHA256
`3c633364c0163f08a61467ce2fb6f6018fbe9546a2d7335c013d4cd14f720e7e`.
All 493 bindings, all 16 actual configuration comparisons, and the eight
paired seeds were independently checked. Four new controller tests passed,
covering the changed 16-job inventory and failure/termination cleanup. The
16-chain control was admitted after these checks; its outcomes remain a
separate sampling-efficiency question.

## Completed result: better access, unresolved patch mixing

All 16 physical chains and all 16 observers completed and drained without
retries or replacement streams. The physical allocation retained all 184,320
attempts. The [comparison](../results/context-correlated-control-comparison-20261005/comparison.json)
has SHA256
`a6b369fc7638557d71d29f9c1cf4b2231f86069aa916ed2967056b94fbe4a803`.
It reuses the 24 completed controls once, without new observations or physical
draws for them.

| Production learned branch | Original, rho=0 | Original, rho=.95 | Context, rho=0 | Context, rho=.95 |
| --- | ---: | ---: | ---: | ---: |
| Attempted proposals | 8,226 | 8,226 | 8,226 | 8,226 |
| Hard-core and wall valid | 13 | 41 | 467 | 1,165 |
| Accepted | 1 | 3 | 2 | 13 |
| Confirmed B-to-A passages | 1 | 3 | 2 | 2 |
| Confirmed A-to-B passages / roundtrips | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The context-prior correlation control improves feasibility by 2.49 times
relative to its independent-redraw control. All four B-start streams reach
the A neighbor set, including two during warmup. The two production passages
therefore do not count all initial relaxation. Between-start neighbor-set
total variation decreases from 0.2711 to 0.0477. This is useful accessibility
evidence for the supplied atlas in this fixed neighborhood.

However, the exact patch-fingerprint supports observed from the two starts
remain disjoint in every arm (total variation approximately one). Sharing
neighbors `{16,217}` does not mean sharing their surface contacts. No learned
move is accepted in any A-start stream, and its paired retained contact
history is unchanged by correlation. These facts prevent interpreting the
neighbor occupancy agreement as contact equilibration or native registry.

The [per-neighbor reduction](../results/context-correlated-patch-neighbor-reduction-20261005/report.json)
locates the discrepancy. In the correlated context arm, the mean absolute
patch-occupancy difference between starts is 0.0469 for the common anchor
16, but 0.8012 for neighbor 217. Expected active patch-token counts are
9.00 versus 9.45 at the anchor, and 17.29 versus 2.87 at neighbor 217.
Ten secondary-contact tokens are present throughout the A-start production
histories and never observed from B starts. The calculation retains all
40,960 production endpoints per start, including rejected residence and
poses without that neighbor; it does not condition on having the contact.
All arms and tokens are retained in the report (SHA256
`a09f4e3f9e1880139c528cef7710000575298afc913355a4eebcd962f4a0ca23`).
These coarse patch indicators are neither physical contact area nor exclusion
overlap volume, depletion free energy, or a native-registry classification.
They suggest a concrete missing capability: preserving or discovering the
second-neighbor registration while satisfying the anchor contact.

The predeclared efficiency measure does not show a consistent benefit:

| rho=.95 / rho=0, same prior | Median patch ESS/CPU ratio | Median fingerprint ESS/CPU ratio | Defined pairs |
| --- | ---: | ---: | ---: |
| Original, start A | 0.998 | 0.998 | 4 / 4 |
| Original, start B | 0.433 | 0.497 | 4 / 4 |
| Context, start A | 0.880 | 0.880 | 4 / 4 |
| Context, start B | 0.533 | 0.343 | 3 / 4 |

These are descriptive paired medians, not confidence intervals. In the last
row the fourth baseline trace is constant, so its ESS and ratio remain
undefined. The B-start ratios are heterogeneous, with large improvements in
one stream and decreases in others. Finite-record ESS estimates from
non-equilibrated starts cannot establish an equilibrium speedup.

![Matched contact occupancy, apparent ESS per CPU, and runtime](../results/context-correlated-control-comparison-20261005/figures/correlated-contact-comparison.png)

The new sampler invocations used 1,923.32 CPU seconds; including supervision,
the physical execution used 1,964.17 CPU seconds and 536.19 wall seconds.
Offline observers used 84.52 CPU seconds inside their analysis sections and
187.07 CPU seconds including process startup and supervision. The final
comparison used 7.44 CPU seconds. These costs are separate from preparation
and reference validation. More valid proposals also incur more bath work;
the gain in accepted count is not free. Dynamic context-prior construction
in an all-mobile simulation remains unmeasured.

The decision is to retain the validated correlated involution as a building
block, but not promote rho=.95 as a demonstrated contact-sampling speedup or
start a larger correlation sweep. Independent physical weights and important
patch-region coverage are now more discriminating. The existing native/contact
weight follow-up continues separately. For this occupied neighborhood,
[reusing the complete rho=0 candidate bank](context-candidate-importance.md)
can supply a second, explicitly conditional importance calculation once its
absolute overlap estimator and full proposal density are validated. The
rho=.95 proposals are state dependent and cannot be added to that independent
bank using a rho=0 density.

The assembly question remains unresolved. This conditional 500 micromolar
neighborhood test neither establishes nor refutes finite-system assembly at
the goal's 106.8 micromolar concentration.
