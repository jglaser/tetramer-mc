# Implementation validation — 2026-09-19

The final Rust release build passes **15 tests**:

- 3 unit tests: rigid rotations, body-BVH versus direct atomic geometry, and
  GSD writing/reading including FP64 box lengths.
- 6 proposal tests: 77 SciPy density/Hastings fixtures from the actual mixture,
  half-turn charts and periodic images, full-covariance Gaussian/Haar moments,
  uniform anchor selection, explicit periodic null mass, invalid-model rejection,
  and matching quaternion ingress tolerances.
- 5 depletion tests: analytic sphere Poisson means/variances, nonzero-Hastings
  accepted flux, a many-neighbor union with material pairwise-overcount error,
  periodic face crossing, conservative envelopes at three budgets, zero activity,
  and invalid-input checks.
- 1 runner test: every move and final checkpoint agrees for uninterrupted versus
  resumed trajectories for learned and uniform methods, including guarded legacy
  continuation. Changed physical shapes/models and output overwrites are rejected.

`cargo test --offline --release` and package formatting checks pass. Clippy is
not installed in the available Rust toolchain and was not run successfully.

The [eight-run Rust campaign audit](rust-validation.md) independently checks
38,400 move records, sixteen atomic endpoints, and 328 GSD frames. It identifies
the historical executable that produced that campaign. Subsequent changes add
provenance/checkpoint protections, consistent ingress validation, segment counts,
and an FP64 box chunk; they do not change the frozen physical parameters.

The final release also continued one real protein checkpoint from sweep 400 to
410. Its three stored frames pass independent atomic audits and independent GSD
decoding, including the added box chunk. All build-embedded application sources
match their hashes and the current files. See the
[final-build check](final-build-validation.json).

## Fixed-endpoint performance

Sixteen recorded protein endpoint pairs (eight seeded, eight fluid), each with
three independent cloud draws, compare the same physical changes:

| Cost | Python reference | Rust |
|---|---:|---:|
| Poisson planning and sampling | 6.856 CPU s | 2.143 CPU s |
| Environment construction and hard checks | 0.844 CPU s | 0.011 CPU s |
| Generated points | 1,178,987 | 1,177,597 |

The measured gate speedup is **3.20×**, or **3.57×** including the listed setup.
There is one timing batch with repeated endpoint draws; no timing confidence
interval is estimated. Partitions and RNG streams differ, while exact thinning
targets the same changing coverage regions. This measures implementation cost,
not mixing or full-application speedup. Source/binary/case hashes are in
[port-benchmark.json](port-benchmark.json).

## Scientific scope

The [longer frozen Python runs](long-python-runs.md) demonstrate persistent
registered incorporation but no contact-environment or unbound returns. Fluid
tetramers become distributed among several small native oligomers. Their
coalescence/rearrangement remains unresolved, as does equilibrium sampling.

The [RJ checks](rjmcmc.md) validate finite-state physical marginals, birth-map
Jacobians, and auxiliary transport flux. They are not a protein RJ benchmark.
Continual model updating is not enabled in the production sampler.

The browser viewer passes syntax, mock-DOM/control, and independent geometry
checks. No browser executable was available, so its WebGL rendering was not
visually verified. It is an optional view of the authoritative saved trajectories.
