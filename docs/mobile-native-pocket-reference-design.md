# Independent finite reference for the alternative native pocket

This design was frozen before the [completed independent integration](mobile-native-pocket-reference-results.md).
The original design and its limits are retained below. The
[saved full-D170 comparison](../runs/mobile-full-capture-comparison-20260921/analysis.json)
places almost all observed high weight in an alternative registered triangle:
body 2 (A)→moving motif 7 and body 1 (B)→moving motif 4. Its original q is
about 46, so it lies outside the single original q≤1 native reference.
The two global estimates, log Q≈58.97 and 60.65, have ESS≈2.7 and 1.2;
they identify a candidate region, not a converged basin weight.

## Freeze the region independently of the sampling density

Keep the exact observed two-neighbor scaffold, capture radius 170 Å,
depletant radius 1.5 Å, activity 0.035 Å⁻³ and hard shape from the
[existing physical config](../runs/mobile-competing-reference-preparation-20260921/config.json).
Use fixed body 2 as the chart frame. Define a single ordinary chart by:

- Relative translation anchor: motif 7's ideal
  `[21.124896616345055, 39.55, -0.5015618770058232]` Å.
- Relative rotation anchor: motif 7, quaternion `[0,0,1,0]` in `[w,x,y,z]`
  order, equivalently `diag(-1,1,-1)`.
- Mean: six zeros. Angular length: `55.02283113084892` Å.
- Covariance: all 36 entries of **base component 19** from the frozen model,
  unchanged. Its condition number is 28.78, and its translation marginal
  standard deviations are approximately 0.06136, 0.03667 and 0.09088 Å.
- Primary finite region: `||u||≤5`, original `q>1`, hard-valid against both
  exact fixed neighbors, and capture-valid. Preserve the original q metric.

Component 19's anchor, mean, covariance and angular length exactly match
component 19 in the earlier 28-component native model. No covariance is
estimated from the current few high-weight observations. Its angular anchor
already equals motif 7's rotation, so ideal centering changes only the affine
mean in the same chart. If `L Lᵀ=Σ19`, the displacement between the old and
ideal centers has whitened norm 0.853976. Therefore

`old component-19 R4 ⊂ ideal-centered R(4+0.853976) ⊂ ideal-centered R5`.

The unchanged covariance is a geometric coordinate choice, not a physical
Gaussian prior. Freeze the final single-component chart bytes, region file,
map/Jacobian convention and hashes before drawing new samples. The final
region-file hash did not yet exist at this design stage. The executed R5
region has SHA-256 `76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae`;
the separate 5–8 shell has `2d9a8e704415e89958d8b9a987a98b26edacb68a8532cf4f3c42e27d8949e88e`.

## Estimate with uniform latent volume

Use the existing `latent-region-normalizer` uniform-ball route, without an
importance guide, reciprocal wrapper or global atlas density. Draw each
six-dimensional `u` uniformly from R5 once. Its physical weight is

`Hhard Hcapture I(q>1) V6(R5) J(u) (W1+W2)/2`,

where `V6(R)=π³R⁶/6` and
`J=det(L)/(ell³ π² (1+|c|²)²)`. The current global component/anchor mixture
denominator is absent. Every invalid draw remains zero in the fixed sample
denominator. Estimate Q0 from the same poses with the bath factor set to one;
report Qz, Q0 and their correlated Qz/Q0 uncertainty.

A minimal first allocation is four independent populations of 8,192 draws,
two independent clouds per valid pose and λ/z=64, with new seeds. Do not
reuse discovery rows as reference samples. Within those same unconditional
draws, retain nested R3/R4/R5 contributions, the 4–5 shell, and the frozen
stateless motif-7/motif-4 triangle label. The label is an additional diagnostic;
it does not silently change the specified physical region.

## Limits, geometric checks and cost

The [saved geometry diagnostic](../runs/mobile-native-remainder-diagnostic-20260921/analysis.json)
finds that the exact A→motif-7 center clashes with B by 0.0247233 Å; the
alternative B→motif-4 ideal placement clashes with A by 0.0225916 Å. The
observed scaffold is slightly deformed. An invalid coordinate center does
not invalidate a surrounding integration region: retain hard-invalid zeros
and **do not repair the scaffold or fit the center to accepted high weights**.

R5 allows at most about 0.5141 Å translation from its center near laboratory
radius 131 Å, safely inside D170 and its existing atomic-wall certificate.
The strongest narrow-chart observations have old component-19 radii 1.06–3.36.
However, a saved broader same-triangle contribution, epsilon-0.1/r03 draw 827,
has radius 20.97. This R5 control tests a named core, not complete site coverage.

Require independent-population agreement, weight-concentration and paired-cloud
diagnostics, plus a prespecified larger-region check before claiming local
coverage. Nested radii alone cannot bound mass outside R5. A separate fixed
R5–8 control can test immediate outer mass; any significant boundary contribution
requires a broader coverage design, not an equilibrium verdict. Neither this
finite reference nor global importance ESS establishes mobile equilibrium.

The previous direct native-R4 control used about 242 sampler CPU seconds for
four populations of 8,192 draws. A few hundred CPU seconds is a planning scale,
not a promised runtime or precision. The new site's saved clouds have roughly
53,000 raw Poisson points each and substantial cloud noise. The current global
populations cannot supply a reliable covariance fit: across both proposal laws,
only 93 draws selected ordinary component 19 at A, of which 12 were hard-valid.

## Input identities

| Frozen input | SHA-256 |
|---|---|
| [Exact physical config](../runs/mobile-competing-reference-preparation-20260921/config.json) | `2ed28d5b618209d51dd8a0381c95a8e58853a7e9780f66d53fb027ec07318cf1` |
| [Hard shape](../runs/mobile-competing-reference-preparation-20260921/provenance/shape.json) | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |
| [Chart source: frozen envelope, base component 19](../runs/mobile-full-capture-campaign-20260921/epsilon-0p1/provenance/model.json) | `dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3` |
| [Earlier 28-component native source](../runs/mobile-posterior-pilot-preparation-20260921/provenance/model-native.json) | `2e534634e6e2fe83da969a2867504c293af8e90a0cf3c4330cdc9e38a6770064` |
| [Native pair catalogue](../runs/mobile-native-region-definition-20260921/inputs/native-pair-motifs.json) | `4c0309b21bf7c31e12bb2f11fb103b20c5772a8472bfc59c2cc0ff8c6ff51282` |
| [Stateless native definition](../runs/mobile-native-region-definition-20260921/definition.json) | `5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9` |

Component 19's covariance alone has SHA-256
`09749b57217199442de7c79b85a1efbf54e14b942b8a9d70110964955134940d`
when serialized by Python `json.dumps(C, separators=(',', ':'), allow_nan=False)`
with no trailing newline. This identifies the proposed unchanged covariance;
the source-model hash above remains authoritative. No integration, new weights
or native-observer replay was performed for this design.
