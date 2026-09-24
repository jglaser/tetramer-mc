# Runtime evidence and boundary compatibility before finite assembly

This is a read-only review of existing summaries, configurations, provenance and
Rust source. It launches no simulation, reruns no audit, and selects no production
window. The [review receipt](../runs/finite-assembly-runtime-review-20260924/receipt.json)
binds the specific inputs, extracted timing records, current source snippets and
their full-file hashes. Historical trajectories and proposal models are unchanged.

## Relevant runtime evidence

| Existing calculation | Runs and sweeps per run | Sampler CPU seconds per sweep | Differences from the new finite-system design |
|---|---:|---:|---|
| `reciprocal-seeded-2000` | 1 × 2,000; N=12 | **0.153855** | Correct coverage atlas and bath; older supplied preparation, not the newly constructed native-eight-plus-isolated-free preparation |
| N=3 mobile posterior pilot | 12 × 2,000 | 0.029493–0.046295; mean **0.042657** | Different size and 150-component atlas; capture-only, posterior-correlation-zero and correlated arms are pooled here only for a runtime range |
| N=4 coverage growth | 4 × 2,000 | 0.053718–0.063316; mean **0.060083** | Correct coverage atlas, but **142.364 μM**, not approximately 106.8 μM |
| N=12 free fixed-atlas controls | 4 × 400 | 0.074346–0.085286; mean **0.081247** | Earlier atlas; auxiliary cloud intensity ratio **16**, not 64; these are the frozen arms only |

All rows use the repaired rigid tetramer shape, depletant radius 1.5 Å and
activity 0.035 Å⁻³. The old N=12 sphere radius is 354.50820786337056 Å and
its nominal concentration is **106.773345 μM**. The N=3 pilot scales that
volume with particle count and has the same nominal concentration. This lies
within the contract's 0.05 μM tolerance, but the new preparations use exactly
106.8 μM: their domains are not byte-identical to the historical ones. The N=4
calculation retained the N=3 sphere instead of enlarging its volume.

The strongest N=12 cost reference is its
[summary](../runs/reciprocal-seeded-2000/summary.json),
[configuration](../runs/reciprocal-seeded-2000/config.json), and
[manifest](../runs/reciprocal-seeded-2000/manifest.json). It consumed
307.710147672 sampler CPU seconds. Its schedule was:

- Local widths: 0.2 Å translation and 1° small-angle rotation.
- Global probability 0.5; half of global slots use posterior transport with
  correlation 0.9. The other half use independent full-mixture capture.
- Uniform defensive weight 0.1; auxiliary Poisson intensity ratio 64.
- Endpoint budget: 2,047 cells, maximum depth 14, minimum width 0.5 Å.
- One GCA opportunity and one center-shift opportunity after every sweep,
  each with probability 1. All twelve bodies remain mobile.

There were 80 accepted local attempts out of 12,035, and five accepted global
attempts out of 11,965. Every one of the 2,000 GCA and 2,000 center-shift attempts
completed. These counters do not say whether internal contacts relaxed. The
summary combines capture and posterior attempts in its global counter; this
review does not scan move logs to split them.

Reported cost components were 212.5265 seconds for the bath gate, 78.9383 for
GCA, 12.5788 for geometry, 1.3454 for proposals and 0.4245 for center shifts.
At this observed rate, 100,000 N=12 sweeps would cost approximately **4.27 CPU
hours**. That is a linear planning illustration, not a bound or a convergence
estimate. Different starts, late aggregate configurations, proposal arms,
hardware contention, collective schedules, native-blind models and N=24 can
change cost. No matching N=24 timing or matched pure-local/redraw/transport
runtime comparison was found in the bounded review.

The exact native-blind 64-slot model has not yet been used for assembly
production according to its [export record](geometry-only-growth-control.md).
The older shape-ray geometry atlas and auxiliary/RJ runs are different
proposals and cannot supply a performance result for that frozen bank.

## Executable and model identity

The N=12 reference binds these SHA-256 identities:

| Input | SHA-256 |
|---|---|
| Repaired shape | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |
| Coverage proposal | `feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e` |
| Embedded source bundle | `2adfcc4ec6ad884c68d2c25bf595906b480bc550810f8a1596a943fa797e96e6` |
| Executable | `1304fe2eadb2dddc7a89c5757c202eb6338569c9a5f3bbb6481d57feafa292ec` |
| Summary | `a0883e32b31076212d748fff4334f03d72b3e5f884c0182e01d49179a919080b` |

The executable bytes remain available under
`runs/mobile-four-body-growth-campaign-20260921/coverage/provenance/tetramer-mc`
and were hashed directly. The current `target/release/tetramer-mc` instead hashes
to `fa0563ce31ef4c7cc480a13672e2d7ae8b52d735c00e84956304e8cb59da258c` at
review time. Historical timings therefore do not authenticate the current
executable. A production freeze must bind its actual binary and embedded source
bundle separately. Descriptive example metadata can name an older atlas; the
runtime manifest and archived model bytes are authoritative.

The current [CLI](../src/main.rs) takes `run --config ... --model ... --method
learned --out ... --sweeps ... --sample-every ...`. Defaults are 400 sweeps and
cadence 10. These are invocation options; putting them in configuration metadata
does not override the CLI. `--no-gsd` and `--no-moves` control output, and
`--resume` changes the invocation history. An independent production stream
must not be silently substituted with a resumed segment.

## A missing capability in the inert boundary contract

The [432-stream inert contract](finite-assembly-contract.md) assumes local,
independent learned redraw and correlated transport arms in both spherical and
periodic boundary blocks. **The current Rust implementation cannot execute that
periodic learned comparison with the two pinned reciprocal models.**

- `src/simulation.rs`, `Config::validate`: frozen posterior transport requires a
  spherical boundary.
- `src/proposal.rs`, `from_json_str_impl`: reciprocal model envelopes require
  an open proposal; `simulation::run` also rejects reciprocal models without a
  spherical wall.
- `src/docking.rs`, `DockingProposal::new`: the transport constructor requires
  an open-space atlas.

Disabling GCA and center shifts is therefore insufficient. Both pinned model
families have reciprocal components, so the independent-redraw arm is blocked
as well. This is a compatibility finding, not a physical or sampling result.
The existing contract, allocation and scientific gates are left unchanged.

The immediately valid common boundary move set is local MC, optionally mixed
with the existing uniform-global kernel, with no learned atlas, GCA or center
shift. A separately declared local-only boundary study could use that set.
Keeping the currently declared learned boundary arms instead requires a
validated periodic extension before configuration and invocation binding.
Either choice must be recorded explicitly; passing the inert schema cannot
stand in for executable support.

## Candidate periodic extension and balance obligations

Let the retained anchor pose be `(a,A)`, the moving pose `(x,R)`, and the box
lengths be `L`. Define the half-open **world-frame** image cube
`D = product_i [-L_i/2, L_i/2)`, and the canonical relative pose

`d = minimum_image(x-a,L)`, `t = A^T d`, `S = A^T R`.

In anchor-body coordinates, the allowed translation cell is the rotated box
`{t : A t in D}`. Checking an axis-aligned cube directly in body coordinates
would be wrong when the anchor is rotated. The anchor is a spectator during
this elementary move, so this cell is fixed for both directions.

An extension can apply the existing open-space involution to `(t,S)` and its
retained labels/noise. If its output `(t',S')` has `A t'` outside `D`, the attempt
must be a recorded null, without redrawing. Otherwise set
`x'=wrap(a+A t',L)` and `R'=A S'`. Canonicalization then recovers the same output
relative pose, so swapping labels and using the inverse noise can recover the
input. This is a proposed construction; it has not been implemented or tested.

Naively wrapping arbitrary open-space outputs merges multiple image preimages.
Their omitted probabilities would invalidate the existing correction. The
unique-image null convention avoids a periodized Gaussian sum and avoids
needing the Gaussian mass inside the rotated box. Ordinary nonreciprocal
periodic capture already uses this convention in `proposal::propose`.

The derivation must establish the following points for the actual code:

1. Canonicalization and wrapping are inverse charts almost everywhere on the
   torus; their translational Jacobians are one. Anchor rotation has determinant
   one, and the existing SO(3) Haar and latent-coordinate Jacobians remain in
   the transport correction. Half-open faces need one reproducible convention.
2. Each retained anchor has the same state-independent probability `1/(N-1)`
   in both directions. All spectators still enter the physical depletion gate.
   Anchor selection cannot silently become neighbor- or contact-weighted.
3. For posterior Gaussian attempts, source responsibility is proportional to
   `w_a g_a(t,S)` and destination labels use the existing weights. On the valid
   unique-image domain, the existing auxiliary cancellation should yield
   `log G(old)-log G(new)`, with **no second Jacobian or image-volume factor**.
   The null complement restores the full Markov kernel; no conditioning on a
   successful in-box draw is allowed.
4. Independent capture uses the continuous off-diagonal subdensity
   `eta/V + (1-eta) G(t,S)` at the unique image, plus its null mass. Its correction
   uses that complete mixture. The separate uniform posterior branch has
   correction zero and must generate a normalized uniform torus pose.
5. Reciprocal charts implement rigid-pose inversion
   `(t,S) -> (-S^T t,S^T)`, which preserves translation volume times Haar measure.
   The image-domain test applies after the full reciprocal/map output; its
   support must remain paired with the inverse trace. Reciprocal branches and
   mixture weights must not be filtered or renormalized by box fit.
6. Periodic hard checks and the many-body gained/lost Poisson gate must receive
   compatible endpoints and all required neighbor images. Boundary-crossing
   replay must verify that a valid proposal construction does not omit a
   spectator or count its exclusion overlap twice.

## Concrete source and test touchpoints

| Source | Required future work |
|---|---|
| `src/proposal.rs`: reciprocal parsing, `log_density`, `propose` | Permit an explicitly validated periodic reciprocal law; retain world-frame image nulls and complete density evaluation |
| `src/docking.rs`: constructor, relative-pose construction and final endpoint | Introduce canonical-image transport support without altering the validated open-space branch |
| `src/simulation.rs`: validation, model construction, `posterior_for_body`, endpoint recording | Wire the boundary-aware law, uniform branch, named streams, null reasons and global anchor indices |
| `src/math.rs`: `minimum_image`, `wrap` | Reuse and test one half-open convention; do not independently invent a second image rule |

Existing tests provide starting points, not a certificate for the extension:

- `tests/proposal_reference.rs::half_open_periodic_boundaries_are_null_without_raw_wrapping`
  and `python_reference_density_hastings_and_periodic_images`: add reciprocal
  envelopes, rotated anchors, anisotropic cells and boundary-adjacent poses.
- `tests/reciprocal_transport.rs::all_virtual_label_inverses_recover_pose_noise_and_physical_jacobian`:
  verify accepted periodic traces and explicit out-of-domain nulls for every
  virtual label, including reverse reconstruction across cell faces.
- `tests/posterior_docking.rs` and the reciprocal stationary-mixture test:
  derive an independent bounded periodic reference with its exact null law;
  do not reuse untruncated open-space moment expectations after clipping.
- `tests/frozen_posterior_assembly.rs`: extend actual-runner trace and bath
  replay with periodic wrapping, non-anchor shielding and restart equality.
  Keep separate coverage of uniform, capture and posterior branches.
- `tests/reciprocal_proposal.rs::reciprocal_envelope_fails_closed_and_rejects_periodic_or_malformed_models`
  and the configuration test in `simulation.rs` currently require rejection.
  Their intended capability assertions need explicit replacement only after
  the new law is implemented; malformed-model and incompatible-mode guards
  must remain.
- `tests/runner_restart.rs`: retain exact deterministic continuation and all
  attempted/null/rejected-state accounting under the new boundary mode.

None of these planning observations resolves contact-weight convergence or
finite-system native stability. The physical evidence gates remain in force.
