# From FFT candidates to usable contact proposals

The current change reuses the completed FFT search and repairs the proposal-fit
workflow. It does not change the hard shape, depletion acceptance, production
move schedule, native classifier, or thermodynamic interpretation.

## Resolution evidence

The cached 1 Å scan was thinned to nested sets of 10k, 30k, 75k, 150k and 300k
orientations. The 75k subset has a raw peak within 4 Å and 10° of every one of
the 768 existing blind-search references. It retains at least 90% of the
full-cache peak score for only 71.6% of those references. Only 300k passes the
predeclared stricter score-preservation criterion. No coarser spatial grid was
tested.

Thus 75k is a **candidate budget for exact-repair testing**, not a demonstrated
minimum resolution. This retrospective check uses existing references to find
nearby peaks. It does not establish independent discovery or coverage of unseen
contacts. A fixed 32-reference repair test completed without any new FFT;
its results below leave the sufficient angular budget unresolved.
See [the resolution audit](minimum-fft-resolution.md).

![Cached angular-resolution audit](../runs/fft-resolution-figure-20260928/fft-resolution.png)

## Addressed failure and remaining questions

Many previous contact covariances were fitted to essentially stationary traces.
Their widths came from the regularizer, which need not follow the narrow
hard-valid contact geometry. The new opt-in refinement uses a frozen material
pivot and tunes its local step size during discarded warmup. Both remain fixed
for retained sampling. The [balance argument and tests](contact-refinement-pivot.md)
describe the symmetric proposal and unchanged exact Poisson acceptance.

The fitter now distinguishes observed scatter from regularizer-only covariance.
Requested exploration gates reject an entire fit without dropping slots. The
pilot requires at least 32 distinct poses and empirical rank six in every chain.
Within-trace PCA ESS, chronological holdout coverage, and independent-chain
agreement remain diagnostics; passing the fit gates is not equilibration.

If the new fit passes, the merger replaces only the first 64 source slots,
representing 41.54% of the original atlas mass. Their original total weights
are preserved exactly, split equally across two independent fitted chains per
slot. The other 704 slots are unchanged. The uniform production branch remains
in the sampler. No native labels select, fit, filter, or reweight the contacts.

The subsequent geometry audit measures unconditional proposal draws against
one partner and against each of the eight supplied seed-anchor neighborhoods.
The seed is evaluation context only. Every null and hard rejection remains in
the denominator. This separates a poorly fitted pair proposal from obstruction
by additional neighbors. The production benchmark then applies the full
many-body acceptance rule, so geometric validity alone cannot count as success.

This change does not supply neighborhood-conditioned contact weights. Pair
peak scores are also not integrated basin masses. Rare proposal selection,
reverse-proposal penalties, and competing multi-neighbor contacts can therefore
remain limiting even after Gaussian validity improves.

## Fixed calculation and unattended continuation

The matched refinement has 64 source contacts, two independent streams each,
two arms, 512 discarded and 4,096 retained-phase attempts per chain, retaining
every eighth attempted state including rejections. Both arms use radius 1.4 Å
and activity 0.0275 Å⁻³, matching the existing atlas and assembly comparison.
This is distinct from the separate 1.5 Å / 0.035 Å⁻³ contact-weight campaign.

Completed calculation:
`runs/contact-refinement-20260928`.

The frozen continuation was launched on 2026-09-28 and completed every stage:
`runs/contact-refinement-continuation-v2-20260928`.
It ran these stages with at most four additional single-thread physical jobs:

1. Analyze all declared chains, then fit both arms with unchanged gates.
2. If the complete adaptive fit passes, merge all original slot masses and
   audit the old/new updated charts: pair plus all eight seed neighborhoods,
   256 draws per component per environment.
3. If validation succeeds, compare the old and merged full atlases from the
   identical eight-seed-plus-256-free preparation at 500 μM tetramers, two streams
   per arm and 200 sweeps per stream. All elementary schedules and physical
   parameters match. Concentration uses the full geometric vessel volume.
4. Independently execute the frozen 32-slot coarse-FFT repair check, including
   failed repairs in its denominator, even if the fit gate rejects the model.

The controller archives code, inputs, binaries, seeds, gates and stage receipts.
It refuses duplicate starts and drains children after failure. Do not relaunch
the completed controller. Its `status.json`, `completion.json`, and `stages/` files
record outcomes. The earlier unlaunched continuation package is
preserved with an external supersession receipt: v2 corrects interpreter-path
handling so the scientific Python virtual environment is retained.

An explicitly partial snapshot is saved at
`/vast/xvg/tetramer-mc-runs/contact-refinement-partial-20260928`.
Among its first 24 matched source chains, the fit gates pass for 24 adaptive
chains versus 15 legacy chains; median distinct retained poses are 438.5 versus
57.5. Median per-chain elapsed time rises from 17.7 to 85.5 seconds, while median
minimum PCA ESS rises only from 5.8 to 7.2. This is evidence of more observed
scatter, **not an efficiency or convergence result**.

The full allocation and proposal benchmark subsequently completed. The adaptive
fit passed all 128 chains; the legacy fit failed 30. Accepted learned redraws
increased from 18 to 203, with 5.78 times as many accepts per production CPU
second. Final registered pairs outside the supplied seed increased from 1/0
to 14/9 across the two streams. One cycle-certified seed reached nine bodies;
the other stayed eight. Most new contacts were isolated registered dimers.
These are accessibility results, not contact ESS or evidence of equilibrium
mixing or sustained crystal growth. The refinement chains still disagree and
their median minimum PCA ESS remains about six.

All four initial/final frames match their input configurations/checkpoints.
An exhaustive 34,716-pair classification of the most ordered final frame
reproduces all 27 registered edges from the accelerated observer. Native labels
are instantaneous and seed components are checked for catalogue cycle closure;
single monomer-patch contacts are not substituted for complete tetramer registry.

See the [completed report and figure](../runs/contact-refinement-results-20260928/report.md)
and the [full posthoc assembly assessment](../runs/contact-refinement-assembly-assessment-v2-20260928/assessment.json).
The initial direct-center-only assessment is also preserved; version two labels
both the original reference center and its physical inverse. This affects
chart-label counts, not the native classification of saved configurations.

The coarse FFT repair test completed too: all 32 poses are hard-valid, but mean
overlap is only 49.3% of archived original repaired full-scan starts, and 1/32
reaches 90%. Because selection/preprocessing differ, the minimum sufficient FFT
budget remains unresolved. The assembly improvement used the existing full-scan
centers, not these coarse candidates.
