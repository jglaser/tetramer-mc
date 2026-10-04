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
