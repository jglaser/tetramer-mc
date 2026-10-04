# Assembly progress, 4 October 2026

**The finite-system thermodynamic verdict remains unresolved.** Completed
calculations establish native accessibility and expose limitations of the
tested proposals. They do not establish equilibrium assembly or instability.

The decision conditions remain the repaired rigid tetramer, depletant radius
1.5 Å, activity 0.035 Å⁻³ and approximately 106.8 μM. The diagnostics below
have different physical conditions and must remain separate.

## Completed growth comparison

At radius 1.4 Å, activity 0.04 Å⁻³ and 500 μM, two completed N=264 histories
share initial poses and the RNG seed. Native-informed proposals grew the
registered seed from eight to sixteen bodies; the tested native-blind FFT
proposal retained eight. Separate registered components reached size ten and
two, respectively. Both histories lasted 100,000 sweeps. Neither is an
independent-start convergence test; neither estimates equilibrium occupancy.

The [growth report](completed-growth-tail-20261004.md) and
[figure](../results/growth-z004-native-figure-20261004/native-growth.png)
retain the full histories, sampler CPU and classifier provenance. Their final
exhaustive pair checks agree with the accelerated observer.

A broader exact symmetry calculation resolves an important interpretation
error. All 64 final registered components embed injectively in the native
**monomer** symmetry orbit. The 28 failures under one four-type tetramer
covering do not imply nonnative registry. The blind history's two observed
families already generate three independent translations when the full orbit
is allowed. This does not itself construct a nonoverlapping infinite packing,
fit the measured coordinates, or establish stability. See the
[unrestricted-orbit report](../results/growth-z004-unrestricted-native-orbit-20261004/REPORT.md).

## Completed conditional sampling comparison

The flexible surrogate benchmark has completed 24 chains and its full contact
observer. Each chain has 512 warmup and 4,096 production blocks. It moves pair
27/132 against 262 fixed spectators at radius 1.4 Å and activity 0.0275 Å⁻³,
using the saved 500 μM environment. Forty completed controls were reused.

Guided eight-step updates cost less CPU than the matched unguided eight-step
updates, but have lower whole-contact-fingerprint apparent ESS/CPU in six of
eight start/stream comparisons. All 24 flexible chains have zero external
contact occupancy. All source/prepared internal-patch distribution comparisons
for the flexible arms have total variation one. Faster internal patch returns
therefore do not establish broader contact coverage.

The [comparison and figures](surrogate-internal-native-comparison.md) preserve
each stream, rejected-state residence, undefined ESS and rare-presence caveats.
The completed sphere reference tests support the corrected kernel's physical
target; they do not establish protein mixing. These results do not justify
promoting the current flexible kernel into assembly production.

## Work still running

Two separately frozen jobs are active at this checkpoint:

- The independent physical-weight campaign at
  `/vast/xvg/tetramer-mc-runs/native-class-physical-primary-20261004` uses the
  original 1.5 Å / 0.035 Å⁻³ conditions. Its allocation is sixteen independent
  populations of 16,384 attempted draws: eight per proposal arm, two Poisson
  clouds per valid pose. Invalid draws retain zero weight. All 81 producer and
  audit stages must finish before the declared postrun comparison. No partial
  scientific result is reported here.
- The internal-native observer at
  `/vast/xvg/tetramer-mc-runs/surrogate-internal-native-20261004` classifies all
  retained endpoints of 48 completed rigid/flexible chains and reuses sixteen
  completed native controls. It has no external-native or cycle observer and
  generates no physical samples. Its purpose is to distinguish patch motion
  from native registration, with full sampler CPU as denominator.

Both allocations, failure rules and limits remain fixed. The user's three
ongoing growth simulations and their production executable are unchanged.

## Next decisions

The current native-class contact guide can now be evaluated in the full vessel,
with source capture retained as a proposal property. Seven Rust controls,
thirteen Python controls, a 160-attempt sphere CLI reference, and independent
reconstruction of every saved CLI row passed. This removes a format/geometry
adapter gap; it does not change the integration target or authorize a physical
conclusion. A larger analytic normalization reference and the current regional
campaign's explicit admission checks remain. See the
[implementation and validation record](native-class-full-vessel-guide.md).

The native observer must first establish what the completed surrogate trials
actually sampled. A mixture of rigid and single-member *local* steps would
remain local at the current scales; it is not a demonstrated solution to
exchange between separated environments. The next candidate under source
review instead combines an existing long-range, reversible member-chart map
with local internal rearrangements inside one corrected surrogate chain.
That is a hypothesis, not a speedup result or an implemented production default.
Its [prospective construction](mixed-atlas-surrogate-design.md) records the
inner and outer balance corrections and the useful matched controls.

A standalone defensive pair selector and a checked Lean selection theorem are
ready for later all-mobile integration. State-dependent pair selection enters
the **same final physical decision** through its reverse/forward probability
ratio; it cannot be added after committing an endpoint. The
[integration design](flexible-surrogate-all-mobile-design.md) records its
six-test validation, reference limits and remaining implementation obligations.
The production sampler does not yet use this selector.

After the full physical campaign completes, reuse the validated postrun
analysis, preserve every failed stratum, and compare matching linear masses
with prior estimates. Passing its primary checks alone would not complete the
required population-size/cloud/defensive sensitivity checks or bound the
unmeasured vessel contribution. The existing N=12/N=24 input bank remains
prepared, not validated assembly production. Finite-system stability requires
the independent-start, size, boundary and physical-weight evidence specified
in the original plan.
