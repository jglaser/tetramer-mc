# Completed matching-target SMC reconciliation

All eight independent SMC populations and both importance comparisons completed.
Authentication checked 231 bound files, the terminal command ledgers, archived
sources, physical inputs, and the exact broad/narrow protocols. No simulation,
classification or previous audit was rerun. The physical conclusion remains
**unresolved sampling limitation**; full-vessel and assembly production remain
unlaunched.

The independent methods agree inside the historical R5 witness. The main
unresolved contribution is native entry **outside that witness but inside the
fixed R4 domain**. Smaller SMC mutation steps improve its representation, but
do not meet the prescribed precision and subdivision checks.

![Completed matching-region masses](../runs/smc-completed-evidence-review-20260923/report/matching-region-masses.png)

The figure reports each estimate's diagnostic population interval, shifted by the
bank point estimate. It does not include the bank uncertainty in each difference
bar; formal comparisons below use combined errors. Axes have different scales.

## Matched physical masses

The shape, two-neighbor scaffold, R4 domain, native classifier and physical
translation/Haar measure match. Each SMC region estimator is
`Zhat_terminal * mean(terminal_region_indicator)`. The table takes arithmetic
means of four independent **linear masses**, retaining zero region estimates.
It is not a comparison of terminal fractions, average log normalizers, or
unmatched capture domains.

| Region | Importance bank log Qz | Broad SMC log Qz | Narrow SMC log Qz |
|---|---:|---:|---:|
| Native entry in R4 | 61.330618 | 60.814299 | 61.167489 |
| Native entry inside old R5 | 60.666634 | 60.636459 | 60.639967 |
| Native entry outside old R5 | 60.607432 | 58.999828 | 60.275732 |
| Contact without native entry | 42.613114 | Unobserved | Unobserved |

Old-R5 native mass passes both the 0.2-log-unit and three-combined-SE criteria
against all five importance arms for both SMC controls. Broad minus narrow is
only −0.003508 log units there. All **50 observed Q0 comparisons** also pass;
their largest absolute log difference is 0.037316. Q0 uses unresampled initial
`H/g` weights, not the proposal-weighted bridge initializer. This is positive
evidence for the observed reference geometry and normalization, not a bound on
unseen weighted mass. The unbound reference remains unobserved.

For total native mass, broad minus narrow is −0.353190 with combined SE0.183098.
For native outside R5, it is −1.275903 with SE0.692079. Both fail the fixed
absolute-agreement rule while remaining within three combined SEs. They are
failed prescribed agreement, not decisive rejection based on those aggregate
errors alone. Narrow total/native estimates pass against all importance arms;
narrow complement passes against only the small-population arm.

## Where the SMC calculation still fails

Both controls use 2,048 particles, 128 fixed annealing stages and four local
mutation opportunities per particle per stage. Broad translation/rotation scales
are `[0.2, 2] Å` / `[1.5, 15]°`; narrow scales are `0.05 Å` / `0.1°`.
This is a proposal-sensitivity control, not the matched-proposal efficiency
benchmark required later.

The complement's population relative standard error (RSE) is
**67.35% broad and 15.93% narrow**.
Broad population r03 supplies 74.88% of its complement mass sum. At the terminal
stage, complement descendants represent only 12–22 initial families per broad
population, versus 189–200 for narrow. Ancestral families are correlated; these
counts and initial-family ESS are not independent physical contact samples.

The complement/native point ratios are 16.29% broad, 40.99% narrow and about
48.52% importance bank. This localizes the disagreement to a substantial native
subset. Narrow used roughly three times the sampler CPU per population, so the
improvement is not a demonstrated contact-mixing speedup.

All ten SMC/importance comparisons fail the frozen subdivision requirement.
Among 27 decision-relevant primary-class/bin comparisons per pair:

| Importance arm | Broad failures | Narrow failures |
|---|---:|---:|
| bank | 19 | 9 |
| wide | 19 | 8 |
| small | 19 | 4 |
| defensive 0.2 | 21 | 3 |
| intensity 256 | 19 | 7 |

These are overlapping class/bin comparisons, not independent hypothesis tests.
For the native complement's outer radial bin, broad minus narrow is
**−2.173339 with SE0.374615**, failing both agreement criteria. That bin supplies
stronger evidence of mutation-dependent sampling than the total normalizer alone.
The ordinary importance calculation independently still fails its original
concentration and subdivision checks; it is not established ground truth.

## Why missing no-entry endpoints do not settle the question

All 16,384 terminal descendants across the eight SMC populations are native.
Neither control independently estimates the physical no-entry mass or the
native/no-entry free-energy difference. Its absence is not assigned zero physical
mass, a confidence bound, or an artificial epsilon.

The importance bank's conditional no-entry fraction is about `7.4e-9`. If that
unconverged estimate were correct, missing no-entry endpoints would be expected
at this allocation. Thus those missing visits alone are not evidence of a bad
return kernel. Intermediate annealing profiles contain no-entry configurations,
but they sample different bridge measures and cannot substitute for final Qz.
The importance estimate near −18.7 kBT remains a conditional-scaffold estimate
without independent SMC validation of its denominator.

## Consequence

The calculation partially reconciles earlier estimates: the common historical
R5 intersection is reproduced, while estimates over the enlarged native region
depend on how its complement is sampled. It has not fully resolved the
discrepancy or supplied the full-vessel remainder. No threshold, region, source
population or allocation is relaxed or discarded after observing the result.

The current full-vessel workflow correctly refuses to launch. The required
finite-system N=12/N=24, initialization, boundary, geometry-only/native-informed,
contact-exchange and assembly-stability evidence is still missing. This result
cannot establish that the protein model supports or prevents native assembly.

Artifacts: [authenticated completed closure](../runs/smc-completed-evidence-review-20260923/authentication.json),
[reproducible summary](../runs/smc-completed-evidence-review-20260923/report/summary.json),
[broad comparison](../runs/smc-importance-bridge-workflow-20260922/broad-comparison/analysis.json),
[narrow comparison](../runs/smc-importance-bridge-workflow-20260922/narrow-comparison/analysis.json),
[mutation-scale comparison](../runs/smc-r4-controls-workflow-20260922/comparison/comparison.json).

The summary and figure are regenerated by
`tools/summarize_completed_smc_evidence.py` using explicit completed-artifact and
authentication paths and a fresh output directory. It launches no physical work.


A subsequent [bounded proposal-design diagnostic](smc-terminal-geometry-guides.md)
reuses terminal geometry to improve native-region proposal coverage. It preserves
this failed physical convergence result and launches no new simulations.
