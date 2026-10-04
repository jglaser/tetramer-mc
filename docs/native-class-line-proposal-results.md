# The class-conditioned line guide passed its audit but missed critical support

All 18 jobs in the original proposal diagnostic completed successfully. The
independent audits validate the saved proposal calculations; they do not show
that the guide covers the troublesome contact regions well enough for a new
physical-weight campaign.

The completed pilot has four independent populations of 128 draws per arm,
plus 40 previously saved development probes. It evaluates proposals and their
densities only: **zero new Poisson clouds or physical region weights**.

| Requested class channel | Selected fresh lines | Empty class lines |
|---|---:|---:|
| Competing, orthant 22 | 35 | 35 |
| Competing, orthant 62 | 73 | 73 |
| Native, orthant 55 | 51 | 47 |

An empty class line falls back to the hard-free line, or to the unconditional
Gaussian if that line is also empty. This keeps the proposal normalized and
preserves support through the defensive component, but does not provide the
intended targeted sampling. The empty sets here are observed geometric outcomes,
not evidence that the corresponding six-dimensional regions have zero mass.

Across 512 fresh class-guide draws, two endpoints were competing orthant 22
and none were competing orthant 62. The hard-free arm found neither. Native
orthant 55 appeared 19 times under the class guide and 13 under hard-free.
Overall native/competing proposal counts were 70/92 for the class guide and
85/89 for hard-free. These are unweighted proposal frequencies, not equilibrium
occupancies or free-energy estimates.

The declared condition to reconsider mostly empty critical lines therefore
applies. The proposed 262,144-draw physical comparison remains **unprepared and
unlaunched**. Its inert resource/input drafts are preserved in
`results/native-class-physical-primary-budget-20261004`; a new physical campaign
should first have a guide with demonstrated access to the missing regions.

## Saved-pose diagnosis and a revised guide

The bounded diagnosis of the 40 already audited saved poses is complete. Each
of the four known competing-22/62 endpoints has a positive-width target interval
containing that endpoint on all three translation axes: 12 of 12 lines. Their
lengths range from 0.0414 to 0.4867 Å. This rules out globally empty line geometry
as the explanation for the pilot's empty competing lines. The frozen prior
instead misses useful neighborhoods in the five coordinates retained during a
line draw. These selected points do not distinguish orientation error from
error in the other two translations.

The proposed revision preserves all 92 old Gaussian means and covariances, with
75% of their previous mixture mass, and assigns the remaining 25% equally to
24 isotropic latent Gaussians. These use eight already known critical poses as
centers, each at standard deviations 0.05, 0.15 and 0.45. The uniform defensive
probability remains 0.5. The five class channels, their equal probabilities,
three translation axes, mass floor and class-to-hard-free-to-unconditional
fallback remain unchanged. Thus, in exact arithmetic,

\[
q_{\mathrm{new}}=\tfrac12 U+\tfrac38 C_{\mathrm{old}}
                  +\tfrac18 C_{\mathrm{added}}
\geq \tfrac34 q_{\mathrm{old}}.
\]

The old conditioned component laws do not depend on their mixture weights.
This is why retaining their parameters also retains their contribution to
coverage. The full new mixture density must still be evaluated for every draw,
including fallbacks and points outside the integration region. No target,
region, native definition or integration measure changes.

With the same conditional estimator noise and a finite old second moment,
this density bound implies a new importance-weight second moment at most
4/3 of the old one. The [checked Lean argument and implementation
obligations](revised-guide-second-moment-bound.md) distinguish this population
bound from finite-record ESS, runtime efficiency and unseen-mode coverage.

All eight critical centers are training data; there are **zero independent
critical holdout poses**. Higher density at those centers is not evidence of
generalization or physical mass. A new, frozen proposal-only pilot will use
four fresh populations of 128 draws. All 512 attempts receive algebra and
endpoint checks; 16 unconditional, independently selected IDs per population
receive the expensive unpruned geometry reference. This selection reduces
reference cost without selecting favorable outcomes. The historical pilot is
development context, not a newly randomized matched control.

If either troublesome competing region remains unobserved or its selected
critical lines remain uniformly empty, the physical-weight campaign stays on
hold. Conversely, observing access does not establish convergence or bound
unseen physical mass. The pilot has no Poisson clouds or physical-weight
estimates and has no automatic sample-size extension.

The completed diagnosis, including the training/holdout distinction, is in
[findings](../results/native-class-saved-support-diagnosis-20261004/findings-v2.json).
The revised reference allocation is in the
[selected-64 pilot recommendation](../results/native-class-saved-support-diagnosis-20261004/pilot-recommendation-selected-v2.json);
the earlier full-reference recommendation is retained as a superseded draft.

### Implementation and the first launch

The 116-component guide was materialized and all 18 focused synthetic tests
passed. Independent source review checked the mixture accounting, fixed audit
selection and preservation of failed attempts. The
[frozen pilot](../results/native-class-support-pilot-20261004/protocol.json)
has four 128-draw populations, 64 full references and no Poisson clouds. Its
execution-plan SHA is
`61613b40549091e46d5496a9e72559841f613753b34fc6b937419cd212e0be76`.

The first population generated all 128 proposals successfully. The new algebra
checker then stopped on its first row because it expected `width_contacts=[]`.
The existing Rust class guide uses one sentinel width and no contact-neighbor
entries, so its genuine output is `[[]]`. Both saved log densities were finite;
this was an adapter-format error, not a proposal-density disagreement. No
independent geometry or endpoint-classification query had begun. The original
outputs, frozen code and failure receipt are preserved.

The correction changes that exact format check and its synthetic fixture. All
24 tests, including the six continuation tests, passed. The separately reviewed
[continuation](../results/native-class-support-recovery-20261004/protocol.json)
was launched with one worker. Its first stage successfully adopted the complete
first population byte for byte; it generates only the remaining 384 draws,
with the original seeds, guide and unconditional audit IDs. The failed run's
source and receipts remain unchanged. All 17 continuation jobs have now
completed cleanly; the physical-weight calculation is a separate next stage.

The continuation execution-plan SHA is
`a847304317aae29967c1aaacc04a12388b34ecbb41af5d4da304e5a8c6d19cee`.
Its launch controller was PID 1925272, process birth tick 199206093. Current
completion is authenticated by `execution/summary.json`, SHA
`48a5d298b01959df747980fdb05e84973fab7b76a03d71a792108d4264c7f2a2`.
The [validation receipt](../results/native-class-support-recovery-validation-20261004/validation.json)
has SHA `8e6000f96ade5b4f5c75cde4f514be790390f39739014a04dec1f2c68c0fd471`.

The initial sandbox-local launcher ended before creating any controller claim
or scientific output. The subsequent host launch is separately recorded in
the pilot directory; it is the launch that generated the 128 proposals. No
scientific draws were repeated by that launcher correction.

### Completed revised-guide access result

All 512 proposal draws passed the algebra and endpoint checks; all 64
preselected rows passed the independent full geometry check. The complete
[statistics](../results/native-class-support-recovery-20261004/analysis/statistics.json)
have SHA `503a7f0e18e81f601ec74e487e09773920caab7cbe7dacd92b66b442450d5ec6`.

| Population | Native | Competing | Competing22 | Competing62 | Native55 |
|---|---:|---:|---:|---:|---:|
| r00 | 17 | 25 | 0 | 2 | 6 |
| r01 | 18 | 28 | 1 | 1 | 6 |
| r02 | 20 | 19 | 1 | 2 | 7 |
| r03 | 20 | 34 | 3 | 2 | 9 |

The other 331 attempts were invalid or exterior, retained in the unconditional
denominator. No unbound endpoint was observed. Usable selected class lines
were 1/59, 3/57 and 10/58 for the three critical channels. The first channel's
single hit occurred in r02; it remains a fragile route. All competing22/62
selected-line hits came from the added bank. The corresponding original-bank
counts were 0/43 and 0/41.

The predeclared access screen passes. This demonstrates access under the
revised proposal, not physical mass or independent generalization. All eight
critical training centers were used in construction; there are no critical
holdouts. Empty requested class intervals still occurred in 207/257 learned
selections (80.5%): 89 fell back to the hard-free line and 118 to the
unconditional law. Thus the retained five coordinates still frequently miss
the requested class.

Producer work totaled 9.21 CPU seconds (about 18 ms per proposal), without
Poisson weight estimation. Independent selected geometry cost 1,704 CPU
seconds, about 26.6 seconds per reference row. Algebra and endpoint checks cost
61.4 and 12.7 CPU seconds. These costs support reviewing the already bounded
primary physical allocation, but do not estimate its unknown cloud cost or
assembly efficiency. The guide stays frozen; the physical comparison must
retain all original strata, coverage and sensitivity obligations.

## Receipts and the reporting correction

The original auditor reports ordered `rows` and a `counts` object. The old
summary expected nonexistent top-level `samples` and `probes` fields. This was
caught before the first summary execution. A
[compatible successor](../tools/summarize_native_class_line_probe_completed.py)
joins the actual audit rows/counts to the separately bound query manifest and
checks every attempt identity. Original working and archived sources remain
unchanged. Six focused synthetic tests passed.

- [Proposal summary](../results/native-class-line-proposal-summary-20261004.json),
  SHA `990644192945028974fb21d0a373158d46d43b368cb6a78207e9ce15bccd3201`.
- Original 18-job execution summary SHA
  `6b104d1436f947d4d16b5e316752ebcc130c563b617b1bd8c879a1130998e436`.
- [Once-only reduction completion](../results/native-class-line-proposal-summary-execution-20261004/completion.json),
  SHA `71969045213b377eec0652807803a4dd5e59a3aac3055d5e051f9d4cb0ba94a6`.
  The bounded reduction exited successfully in 3.88 wall seconds.
- [Correction validation](../results/native-class-line-summary-schema-correction-20261004/validation.json),
  SHA `f8ee20a00af160b328fb17aee2be37f74eb0f906f48f116fc3822d590d52442b`.

Neither the original guide's coverage failure nor the revised guide's access
success decides finite-system native assembly. The thermodynamic question
remains unresolved.
