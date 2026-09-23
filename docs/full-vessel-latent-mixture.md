# Full-vessel latent proposal mixture

`basin-normalizer` can integrate the unchanged atomic protein-wall target with a normalized 50/50 mixture of the complete existing vessel proposal and a frozen latent guide. This is a proposal change, not a restriction to a contact pocket. Protein execution remains gated by regional convergence and matching independent-method checks. The full-vessel remainder is the next measurement, not a prerequisite that can already be assumed.

The library entry point is:

```rust
normalizer::run_with_wall_and_latent_guide(options, wall,
    NormalizerLatentGuideFiles { region, guide })
```

The CLI adds paired `--latent-region REGION.json --latent-guide GUIDE.json`; both require `--wall-radius`. Existing `NormalizerOptions` literals and the `run`/`run_with_wall` entry points are unchanged. Existing no-guide schemas and pose/cloud streams retain their previous behavior.

For world pose x, the estimator uses

```
p_mix(x) = 0.5 p_vessel(x) + 0.5 q_D(u(x))/J_D(u(x))
weight(x) = H_wall(x) H_hard(x) mean(W_1,W_2,...)/p_mix(x)
```

The physical measure is center volume times normalized SO(3) Haar. `p_vessel` is the complete atlas/reciprocal/uniform law, averaged over every selected proposal anchor, evaluated after recentering the world pose by the current capture center. For legacy models, covariance scaling still multiplies standard deviations; active reciprocal models still require scale 1. The latent component uses the separately tested `PhysicalLatentGuide` API. Its uniform component is restricted to the source R4 ball; its Gaussians remain untruncated. The source R4 ball and source capture do not restrict the target. In particular, valid exterior-R4 and exterior-source-capture poses contribute normally. Both full component densities are evaluated on every pose, including target-invalid poses, before wall/hard checks.

The outer coin uses the independent `outer-mixture` stream: a uniform value below 0.5 selects the latent branch. Latent generation uses `latent-pose`; vessel generation retains `pose`; cloud generation retains `cloud`. All are derived by the original `tetramer-normalizer-independent-v1` seed/draw/role hash. Draws are unconditional and never retried. A numerical null or unrepresentable positive density in the mixed path fails explicitly instead of silently conditioning the proposal. Invalid physical outcomes remain zero-weight rows in the original denominator.

The guarded capture enclosure and every fixed-body atomic wall check remain mandatory. The wall tests every protein-core atom; it does not clip the ideal-depletant bath. Valid rows have `log_hard_weight = -log_proposal_density`; no additional latent Jacobian belongs in this weight because the physical mixture already contains q_D/J_D. The existing native metric and output partition are unchanged, and the independent complete native classifier remains necessary for the physical contact comparison.

## Versioned output

The manifest has integer `schema: 5` and `outer_mixture_schema: "full-vessel-latent-half-mixture-v1"`. `pose_proposal_schema` retains 1/2/3 for the underlying vessel generation law. Existing atomic-wall metadata are retained. New manifest bindings include exact guide/region hashes, latent component count and defensive probability, source capture metadata marked as not restricting the target, density measure and stream names. The original vessel proposal description is retained as `vessel_proposal`.

Exact bytes are archived as `provenance/latent-region.json` and `provenance/latent-guide.json`, alongside the existing input configuration, shape, vessel model and embedded source bundle.

New fields on every mixed row:

| Field | Meaning |
|---|---|
| `outer_branch` | `vessel` or `latent` |
| `log_vessel_proposal_density` | Complete anchor-averaged vessel density at the world pose |
| `log_latent_physical_density` | Complete q_D/J_D at that same pose |
| `log_proposal_density` | The full outer 50/50 mixture, always finite on a generated row |
| `latent_density.latent` | Inverted world-pose latent coordinates; null only at the exact Cayley seam |
| `latent_density.in_reference_ball` | Reporting-only R4 membership |
| `latent_density.log_latent_density` | Complete latent mixture log q_D |
| `latent_density.log_physical_jacobian` | Log J_D |
| `latent_density.coordinate_chart_seam` | Explicit exact-seam flag |
| `proposal` | Existing vessel generation object on vessel rows; null on latent rows |
| `latent_proposal` | Generating latent coordinates, radius and selected Gaussian index on latent rows; null on vessel rows |

Within **component log-density fields**, JSON null means negative infinity, hence zero density. This includes the legitimate exterior-R4 zero of an alpha=1 latent guide. At the exact Cayley seam, its latent coordinates and Jacobian are absent and its physical component density is zero. This convention does not turn missing generated-row mixture density or missing valid-row weight into zero; those values must be finite. There is no finite seam exclusion band.

## Validation and remaining obligations

The new integration tests use synthetic analytic spheres only. They test full-wall hard and depletion integrals against independent radial quadrature, normalized Haar observables, both outer branches, shifted capture and rotated anchor frames, all-anchor and selected-anchor density evaluation, reciprocal and legacy covariance-scale laws, valid latent tails beyond R4/source capture, invalid zeros and original N, alpha=1 component zeros, deterministic repeats and unchanged vessel/cloud streams, source identity failure, and CLI option guards.

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --test basin_normalizer_latent_mixture -- --test-threads=1 --nocapture
```

The separate Python full-vessel audit must reconstruct both physical densities on every world row and verify branch-specific generation only for the appropriate branch. It must not apply the old assumption that every row was vessel-generated. This integrated reference validation does not establish converged protein weights, full-vessel coverage or finite-system assembly. Geometry predicates, exact thinning, floating-point execution, the complete native observer and population-level convergence remain explicit obligations.

For a separate cross-language audit, the ignored export test writes six fresh 128-draw fixtures: legacy, reciprocal and alpha=1 latent guides, each with all-anchor and selected-anchor vessel proposals. It preserves every fixture input, output, embedded source bundle and the test executable; `fixture-index.json` binds manifests and sample files. Use a new destination and wait for a free physical worker slot before executing it:

```bash
TETRAMER_FULL_VESSEL_REFERENCE_OUT=/home/xvg/tetramer-mc/runs/full-vessel-latent-reference-validation-20260922 \
CARGO_BUILD_JOBS=2 cargo test --offline --test basin_normalizer_latent_mixture \
  export_integrated_reference_fixtures -- --ignored --exact --test-threads=1 --nocapture
```

This explicit export is separate from the already completed radial/Haar calculation. Its small fixture draws are for arithmetic and provenance reconstruction, not a second precision estimate. Exact seam behavior remains covered by the mathematical physical-guide tests; an ordinary continuous random proposal almost surely never generates the exact seam.

The subsequent independent reconstruction is complete: six fresh128-row
fixtures covering legacy/reciprocal/uniform-source laws and selected/all
anchors passed [`audit_full_vessel_latent.py`](../tools/audit_full_vessel_latent.py).
See the [integrated validation](../runs/full-vessel-reference-dispatch-20260922/validation.json)
and [independent-method documentation](r4-independent-method-comparison.md).
This does not authorize a protein-vessel or assembly conclusion.
