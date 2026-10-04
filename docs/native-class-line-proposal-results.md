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

The next bounded development diagnosis uses the 40 already audited saved poses
to distinguish failure of the five fixed coordinates to reach useful lines
from failure of the one-dimensional conditional geometry. It can assess whether
additional normalized Gaussian components around those known poses are a useful
proposal change. These poses are development data; any resulting guide must be
frozen and evaluated on independent draws with the full density and defensive
component. Coverage of unseen modes remains a separate obligation.

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

No conclusion about finite-system native assembly follows from this failed
coverage improvement. The required thermodynamic question remains unresolved.
