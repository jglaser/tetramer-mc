# Hard-free line guide in the physical region normalizer

`latent-region-normalizer --importance-guide guide.json` accepts the frozen
`defensive-hard-free-line-guide-v1` schema. It has a separate dispatch variant and
manifest schema, `importance-latent-region-normalizer-v6`; the existing guide
formats and the assembly kernels retain their previous proposal laws.

The guide keeps the original Gaussian mixture and its five-dimensional marginal
for each selected raw translation axis. It replaces the remaining conditional
Normal only on the exact union of hard-free intervals inside the frozen region
ball and center-capture ball. Empty intervals and component conditional mass at
or below the declared floor use the original conditional Normal. Other retained
coordinates are never redrawn. There is no contact-width or native-label
restriction. See [the guide derivation](hard-free-line-guide.md) for its law.

The production estimator remains

\[
\widehat Z = N^{-1}\sum_{i=1}^N
 I_{\rm region}(u_i)H_{\rm capture}(u_i)H_{\rm hard}(u_i)
 \frac{J(u_i)}{q(u_i)}\frac{W_{i,1}+W_{i,2}}2.
\]

The complete normalized mixture density includes every component, selected-axis
mixture, deterministic fallback and defensive uniform term. Outside-region and
hard-invalid attempts remain zero in the unconditional denominator. The physical
Jacobian, cloud estimator, target metric and independently indexed latent/cloud
RNG streams are unchanged. Production still rejects any generated sample with an
unrepresentable proposal density instead of dropping or retrying it.

## Records and failures

Each v6 sample includes `hard_free_line_draw` and
`hard_free_line_density`. The latter retains all axes' exact feasible intervals,
axis proposal densities and baseline density, enough for independent full-mixture
reconstruction from the archived guide. It omits large component-by-component
diagnostic tables and runtime-dependent timing fields. These records do not
change RNG consumption.

`attempts.jsonl` is appended and flushed **before** each requested draw, with
`{"draw": i, "state": "begin"}`. Successful sample rows are flushed after each
draw. An error writes `failure.json`, identifying the draw and reporting
`complete: false` and `retry_performed: false`, then exits. Earlier samples and the
failed attempt's identity remain available. Completion synchronizes both files
and binds the journal hash in `summary.json`. Output directories must be empty.

This is an attempt journal, not a power-loss checkpoint or a resume protocol.
There is no resume option and no automatic retry. Deterministically repeating a
seed in a separately declared reference control verifies reproducibility; it is
not a replacement for preserving a failed production population.

## Validation

`tests/latent_region_hard_free_line.rs` checks the production CLI against an
independent sphere hard-volume and Asakura–Oosawa integral, using 8,192 attempts
at each of activities 0 and 2, two independent clouds per valid pose, a fixed
six-dimensional ball, and all three translation axes. The tests preserve
hard-invalid and outside-ball zeros, check unconditional volume normalization,
and verify the `J/q` and two-cloud formulas on every recorded row. An exact
64-row CLI/API prefix comparison checks indexed-stream reproducibility.

Small matched controls check that zero conditioning and the pure-uniform limit
preserve complete physical rows and cloud streams. Binding/schema rejection and
nonempty-output refusal are also covered. A unit test injects a failed attempt
and checks the journal precedes work, exactly one attempt is made, and the failure
receipt is retained. The shared guide's conditional-Normal, geometry,
normalization and compact-trace tests remain separate.

To retain integration artifacts at a chosen new path:

```sh
TETRAMER_HARD_FREE_REFERENCE_OUT=/absolute/new/reference/path \
  cargo test --release --target-dir target-validation-line-guide -j2 \
  --test latent_region_hard_free_line -- --nocapture --test-threads=1
```

The reference allocation validates implementation and simple physical limits. It
does not establish protein contact-weight convergence or finite-system assembly.

The preserved 2026-10-01 production references passed: the activity-zero hard
integral was `16.6628 ± 0.2298`, against `16.6911`; the activity-two depletion
integral was `18.6091 ± 0.2382`, against `18.5593` (one standard error). Independent
Python reconstruction checked all 16,704 new-guide rows, including the 64-row
repeated prefix, with maximum log-density discrepancy `1.16e-14`. The zero-guide
and pure-uniform controls reproduced the old physical records exactly.

The receipt and archived candidate are in
`results/hard-free-line-production-validation-20261001`. A separate, recorded
test-only correction avoids mutating a JSON object while comparing disabled
compact traces; it changes no production code or physical output. The receipt
binds this `cfg(test)` overlay separately from the executable's embedded sources.
