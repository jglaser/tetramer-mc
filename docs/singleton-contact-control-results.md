# Two-neighbor singleton control: completed conditional comparison

The new control does **not** show a general sampling advantage from fusing the
two neighbors' Gaussian guidance. Single-tetramer moves do enable some external
contact returns absent under local moves, but neither the fused nor unfused
version resolves the tightly embedded dimer or agreement across initial states.
These are conditional sampling results, not evidence against native assembly.

The completed comparison contains 64 new singleton chains and 64 cached local
and rigid-dimer (`m4`) controls: four environments, two preparations, four streams
and four proposal arms. Each chain has 512 warmup blocks and 4,096 production
blocks, including rejected residence. The target is **two mobile tetramers with
262 frozen spectators**, depletant radius **1.4 Å**, activity **0.0275 Å⁻³**,
from the 500 μM source. It differs from the assembly decision conditions of
1.5 Å, 0.035 Å⁻³ and approximately 106.8 μM.

![Per-stream contact diagnostics](../results/two-neighbor-singleton-contact-review-20261004/contact-efficiency.png)

Each dot is one stream. The contact-fingerprint ESS is an apparent finite-record
autocorrelation estimate, not established equilibrium ESS. Its denominator is
the full sampler CPU, including warmup and rejections. Undefined estimates remain
undefined. Contexts are different conditional targets and are not pooled.
Fused/unfused runs share proposal RNG roles; they are paired controls, not two
independent confirmations.

## What changed, and what did not

- **Environment 27/132, source start:** fused and unfused single-tetramer moves
  each produced external-return counts `[0, 0, 4, 9]`; the rigid-dimer control
  produced `[0, 1, 3, 0]`, and local moves `[0, 0, 0, 0]`. The singleton chains
  cost approximately 397–448 CPU seconds each, versus 180–249 for the rigid
  dimer. More observed returns therefore do not alone establish a speedup.
- **Environment 32/110:** source-start returns were `[0, 0, 0, 6]` for fused and
  `[0, 0, 0, 2]` for unfused guidance. This difference is confined to one stream;
  the prepared starts provide no consistent fusion advantage.
- **Embedded environment 9/24:** all four source-start streams of every arm
  had zero external contact changes. The external-environment distributions
  from the two starts remain disjoint (total variation 1). Faster fluctuations
  of internal contact fingerprints do not resolve this obstruction.
- **Embedded environment 11/246:** prepared-start returns were `[0, 0, 11, 0]`
  for fused and `[0, 3, 11, 0]` for unfused guidance. The shared successful stream
  is useful accessibility evidence; it does not establish initialization
  agreement. External-distribution total variation between equally weighted
  four-stream means is 0.803 for both singleton arms, versus 0.345 for `m4`.

Local moves have larger apparent contact-fingerprint ESS per CPU in this test,
but **all 32 local chains have constant external contact sets**. In the whole
27/132 and 32/110 environments those sets are empty. Thus the large local ESS
mostly describes changes within the observed internal-contact history; it is
not a measure of sampling competing external environments. Conversely, contact
returns under the broader moves can include gain/loss of the same partner.
They need not be changes of basin or native registry.

The native-blind contact observer finished successfully in 330.35 wall seconds,
with 322.05 analysis CPU seconds. The original new-chain geometry setup cost
24.89 CPU seconds and remains separate from observation. The observer classified
18,488 previously uncached pairs across 294,976 new
retained endpoints; old geometry and MC draws were not repeated. Sampling costs
for the new 64 chains were 28,591.58 CPU seconds; the inherited 64 controls cost
10,586.72 CPU seconds. These sums describe expenditure, not a pooled efficiency.

## Consequence for the next step

Do not promote Gaussian fusion as a default on this evidence. Keep the unfused
single-tetramer control and rigid-dimer proposal while resolving what prevents
the embedded environment from changing contacts. Native registration must be
measured independently of external-contact turnover.

The saved aggregate counts narrow the trapped source case: every singleton
stream in both arms has 4,608 attempts, zero accepted moves and zero proposal
self-loops. Each attempt reached a hard/wall-valid candidate and was rejected
by the joint proposal-density and Poisson-bath acceptance gate. This rules out
cap exhaustion or zero source support as the final rejection cause for these
attempts. Those aggregate counts alone do not separate a proposal penalty from
a bath penalty. Rejected intermediate
geometric trials may still contribute cost before the feasible candidate.

The passed initial-state reference identifies a native internal motif for 9/24
and two native external contacts incident to 24. Persistence of this state may
therefore reflect strong physical binding, rather than unwanted misregistration.
Zero departures alone do not diagnose inefficient sampling: transition rates
need not be equal between unequally occupied regions. Independent conditional
weights and the saved gate factors are needed to distinguish rare physical
returns from poor proposals. The different preparations have not established
equilibrium agreement.

### Saved acceptance factors

The subsequent bounded reduction completed successfully on all 16 context-2
singleton chains: **73,728 attempts and 221,810 inner trials**, with no new
geometry, bath draws or MC replay. The context was selected after observing its
lack of accepted source moves; every chain and attempt within that follow-up
allocation was then fixed before journal inspection. This is a development
diagnostic, not an independent equilibrium estimate.

Write the previously audited saved identity as `T = C + B`, where
`C = log G(old) - log G(new)` is the complete proposal correction and `B` is
the realized Poisson-bath log factor. Positive values favor the trial. The
following ranges are **per-stream production medians**, not physical free energies
or confidence intervals:

| Moving member | Proposal `C` | Bath `B` | Combined `T` |
|---|---:|---:|---:|
| 9 | +19.5 to +23.3 | −54.4 to −51.8 | −34.9 to −29.1 |
| 24 | about +26.9 | −101.2 to −98.4 | −74.4 to −71.6 |

The proposal correction is favorable for 92.4% and 99.0% of source-production
candidates for members 9 and 24, respectively. Every source-production bath
factor is negative. Learned inner draws pass geometry only 4.87% and 1.73% of
the time; consequently 92.1% and 97.1% of final candidates come from the uniform
branch. These mostly uniform escape attempts lose favorable exclusion overlap.
An adverse proposal-density correction is not their dominant obstruction.

No fused target label was drawn in any of the 221,810 inner trials. All 32
matched fused/unfused stratum summaries are identical. The saved schema does not
contain catalogue screening or eligible-fusion counts, so this does not identify
why fusion was unused. Missing compatible basins and rejection of approximate
component means remain hypotheses, not findings.

This result does not motivate an immediate correlated-transport campaign in
this native-bound context. Retain it as a native-retention control and use the
other contexts' native-registry audit to identify an appropriate reorganization
target. Any future comparison of independent and correlated proposals must use
matched kernels; repeating correlated trials until feasible generally needs an
additional state-dependent normalization correction.

The reduction used 71.34 CPU seconds and 129.02 worker wall seconds (131.06
seconds including the enclosing child lifecycle). Five synthetic tests passed
before the frozen run. The
[complete factor summary](../results/trapped-singleton-gate-diagnostic-20261004/analysis/summary.json)
has SHA `fcac797822e76837ef91813f6bf74fe14ebf74b45624a9482a3cd61b854d717e`;
the [per-stream interpretation](../results/trapped-singleton-gate-diagnostic-20261004/supplemental-result.json)
has SHA `a6a4dce8bd89e7a88d68c6eb3da6cd0631786cd38a784ec22e5ee68167bb7fb7`.

The predeclared native audit was launched on all 128 completed trajectories at
`/vast/xvg/tetramer-mc-runs/conditional-native-registry-audit-20261004`, with one
worker, 36,000 CPU seconds, 72,000 wall seconds and 16 GiB memory. It retains all
589,952 endpoints, reuses the authenticated fixed native graph, and includes
256 preselected unpruned checks of omitted pairs. Its execution-plan SHA is
`7e131d6565c173b39f2f6b4f776a8fc5f6939f4cfcf488c344a4539c19a2a12f`.
Launch is not completion. The earlier home-directory preparation is inert.

The independent physical-weight audit continues separately. Even successful
conditional contact reorganization cannot determine finite-system stability;
failed proposals cannot refute it.

## Reproducible saved evidence

- Completed contact analysis:
  `results/two-neighbor-singleton-postrun-analysis-20261004/analysis/analysis.json`,
  SHA `2d956ca548b1c3f2abd563d5d6473f6f766f5e05117002603ac9c899e29c98ff`.
- Full per-stream table and initialization distances:
  [table](../results/two-neighbor-singleton-contact-review-20261004/table.md),
  [bound reduction](../results/two-neighbor-singleton-contact-review-20261004/review.json),
  SHA `1e170da169d5d0b999aeb635a035bc4eedd4e8be35f8294a6590c3658f8d8459`.
- [Plot source](../results/two-neighbor-singleton-contact-review-20261004/plot.py)
  reads only the bound completed per-chain summaries. No trajectory parsing,
  geometry, new sampling or ESS recomputation is required to render the figure.
- Construction, balance and reference checks:
  [singleton control](singleton-contact-control.md) and
  [native audit](conditional-native-registry-audit.md).
