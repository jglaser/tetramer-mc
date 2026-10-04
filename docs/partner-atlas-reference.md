# Partner-anchored atlas: cached-source physical reference

This completed reference tests the optional partner-atlas methods in
`FlexibleSurrogateKernel`. It does not launch a protein campaign or establish
efficiency, native accessibility, or full-vessel assembly. The six focused
partner-atlas tests and nine existing flexible-kernel tests passed under
`results/partner-atlas-validation-20261004/attempt02/validation.json`; the first
compile-only failure remains preserved under `attempt01`.

`tests/partner_atlas_stationarity.rs` is a new ignored numerical test. It does
not call a source sampler. Its inputs are all four streams of 2,048 previously
accepted IID physical source poses, without selecting on any old kernel
acceptance or new outcome. Every arm resets to each source. There are exactly
32,768 new outer calls and 147,456 candidate attempts:

| Arm | Candidates per source | Inner score | Final correction |
|---|---:|---|---|
| `direct` | 1 | No filter or score evaluation | Helper proposal correction |
| `m1_guided` | 1 | Full fixed surrogate | S(source) − S(endpoint) |
| `m8_guided` | 8 | Full fixed surrogate | S(source) − S(endpoint) |
| `flat8` | 8 | Zero score, including proposal MH correction | Zero |

All arms use fresh fair member selection and fixed .25 partner-atlas/.75 local
mode probabilities. The learned or uniform helper branch is retained. Atlas
proposals move only the selected tetramer and condition on the other current
member. The learned helper correction enters the inner MH filter, or the direct
physical gate, exactly once. Each final physical decision uses the same
fair-order two-singleton bath engine. Hard failures, map nulls, MH rejections
and identities remain in the allocation and statistical denominators.

## Cached source authority and fixed target

The old authority is `results/flexible-surrogate-reference-20261004`:

| Artifact | SHA256 |
|---|---|
| `validation.json` | `84d0b14661f5b137cd84b130add0b6cb5fac6c691eb35d57c4af30d158f2428f` |
| `reference/receipt.json` | `d5aac345bd46dabf525b59e7b5c309caf29dc1eb19734bfdc1f7529361c4801d` |
| `audit-validation.json` | `98e4657314153617ab1a1277c0f9827c295752112b65eafbafe40004dd722146` |
| `journal-audit.json` | `2f22d5e47fcf717299ca140b1b9beb15748c7eef5c424f8a899e0e5e46098c79` |
| `reference/kernel-attempts.jsonl` | `b8218bb4bb658587fa70540f2a423b620cdfd600a083d179b70a82a983d454c5` |
| `reference/protocol.json` | `15c9293a6d7aafdd92246eaef8b8c81d407060c1a4955bf84ace40b42ee78165` |

The independent audit already joined every old kernel source to the exact
direct-sphere/PPP source sampler. The extractor takes every `m1_guided`
`kernel_begin.old` pair and checks the corresponding source identities across
all old arms. It does not rerun old geometry or decode the larger PPP source
journal. Cache output is
`results/partner-atlas-reference-sources-20261004/{sources.jsonl,receipt.json}`.
Rows have exactly `kind`, `stream`, `source_index`, `source_attempt`, and `old`
(two complete poses); all 8,192 identities occur in stream/index order.

The consumer requires explicit expected SHA256 values for both cache files,
the exact prospective allocation, and the bound original protocol. It checks
the physical target, cloud hash, complete inventory and direct sphere support.
The target remains core radius .1, depletant radius .9, activity .5, lambda 2,
protein wall radius 1.6, and one fixed sphere at the origin. The ideal bath
permeates the protein wall. The body cloud remains the original 8³ midpoint
grid with 280 retained points and raw-volume weight .015625; its hash is
`c68700a110eba1324d1d50fb5498a1be724b1a4f5693bef880cb2fb9ae8fd482`.

The synthetic two-component reciprocal atlas is fixed in source before new
physical execution. Its full model, weights, covariance matrices and rotations
are serialized in `protocol.json`. Correlation is .6, the separate uniform
weight is .1, and the immutable world cube is side 3 centered at zero. It
contains all valid source centers; encoded candidates are still checked.
Local scales are .25 translation and 30 degrees rotation. Nothing is fitted
to the cached poses or previous kernel outcomes.

## Fixed diagnostics and failure semantics

The original 38 direct-sphere and orientation observables are unchanged. The
new fixed bounded orientation panel adds all nine entries of R0ᵀR1, their
nine squares, R0_00², R1_00², and x0·R0_00. Analytic source means are zero for
entries/cross-products, one third for squared matrix entries, and one quarter
for the original quaternion-square observables. Each arm reports paired
endpoint-minus-source moments for every source, per stream and over all 8,192
sources. All paired means must be within 6 empirical SE plus 1e-11 of zero.
These checks are diagnostics, not a simultaneous confidence guarantee or a
proof of joint stationarity.

Two negative controls reuse the actual direct candidate, bath and final
uniform, omitting or reversing its proposal correction. Hard/null/identity
self-loops remain unchanged. There is no hypothetical replay of a different
multi-step inner path and no additional bath draw. Their diagnostic is
accepted × [atan(log G_new) − atan(log G_old)]/π for learned direct candidates,
zero for all other source rows. G is the complete learned member-chart
density, excluding the separately retained uniform branch. The correct
direct drift must be within 6 SE of zero; each negative drift must exceed
max(.0005, 6 SE). Failure means unresolved sensitivity, not permission to
retune or replace samples.

Each arm must exercise both slots and bath orders (>100 each), learned
proposals (>100), uniform proposals (>10), nonzero learned corrections (>100),
hard rejections, >500 physical decisions, >200 final acceptances, and >100
accepted changes in internal lens volume. More than ten physical decisions
must contain a retained learned move with |correction| >1e-9, including at
least one final acceptance. Each arm must process >100 raw bath points and
nonzero gained and lost counts. These are explicit implementation-exercise
gates; local-mode success alone cannot satisfy them.

All begun/outcome records are flushed, with full kernel traces and negative
retained poses. The complete statistical summary and every check are written
even when a diagnostic fails. Resource/numerical failures preserve the prefix
and stop; there are no retries, replacements, extensions or fresh sources.
The kernel budget is 600 process CPU seconds and 10 million raw/retained bath
points, with 100,000 per leg/outer gate. The supervising one-off runner must
add wall/memory/process-drain bounds and bind the executable/source closure.

Compilation and execution remain separate. The ignored test is named
`partner_atlas_preserves_cached_physical_sources`. Its required environment
variables are `PARTNER_ATLAS_REFERENCE_OUT` (fresh absolute directory),
`PARTNER_ATLAS_SOURCE_CACHE`, `PARTNER_ATLAS_SOURCE_CACHE_SHA256`,
`PARTNER_ATLAS_SOURCE_RECEIPT`, and `PARTNER_ATLAS_SOURCE_RECEIPT_SHA256`.
Fresh role seeds hash namespace
`partner-atlas-cached-iid-physical-reference-v1`, stream, arm and role. No old
RNG continuation or source stream is reused.

## Completed result and independent audit

The fixed allocation completed: 32,768 outer calls, 147,456 candidate attempts,
790,544 raw bath points and 122,933 retained bath points. All **292 predeclared
checks passed**. Every arm processed all 8,192 cached sources; there were no new
source draws, retries, replacements or allocation extensions. The numerical
summary reports 7.37 CPU seconds. These independently reset source updates are
a stationarity diagnostic, not a trajectory mixing or speedup benchmark.

The proposal-correction control was sensitive to the intended mistakes. For
the bounded learned-density observable defined above, the correct direct
kernel gave mean 0.000172 with SE 0.000246. Omitting the correction gave
0.010254 ± 0.000566; reversing it gave 0.012613 ± 0.000606. The latter two
deviations are 18.1 and 20.8 empirical standard errors from zero. Both negative
controls reused the direct arm's candidates, bath and final uniform.

An independent Python audit reconstructed all 32,768 outer records and 147,456
candidate records, including source identity and residence, inner and final
acceptance arithmetic, work counts, saved-observable reductions, the negative
controls and all 292 checks. It passed. This audit did not independently
recompute geometry, proposal densities, or random-number generation. Together
with the focused inverse-map and kernel tests, it supports taking this
implementation to a conditional protein efficiency test. It does not prove
floating-point detailed balance or establish protein accessibility.

| Completed artifact | SHA256 |
|---|---|
| `results/partner-atlas-reference-20261004/validation.json` | `7b7e110cef6547c28c385b88c372a9c4ab6ee776899448c04ca1e7b2b849155e` |
| `results/partner-atlas-reference-20261004/reference/summary.json` | `0dbe756408dc0f3c202738abb454115c199f9e830feeb377bf97cc3058ffd3e4` |
| `results/partner-atlas-reference-20261004/reference/receipt.json` | `a1ea1bdfe88ee57f78ab76c0c5f1a564f3e632fe2e948b140164bb20d5d47eca` |
| `results/partner-atlas-reference-audit-20261004/attempt01/audit.json` | `a410ce80b184d6c5017c4263f69f31d641fb606183afaeea89bdb22e67df1f14` |
| `results/partner-atlas-reference-audit-20261004/attempt01/validation.json` | `84d8070078c2da23f0b6dd4f2ddea0a2947f2f10ba04875f6b2a5c81a4174467` |

The audit's nine synthetic tests also passed, including strict rejection of
boolean/integer substitutions. Compile-only failures remain archived in
`results/partner-atlas-reference-build-20261004` before the successful
`attempt03`; they generated no numerical samples. The executed source and
prospective protocol are retained in the reference archive. Production
executables and existing simulation defaults remain unchanged.
