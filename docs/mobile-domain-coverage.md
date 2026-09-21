# Fixed-scaffold domain coverage

The finite-region weights motivated a full-support importance campaign over
the remaining captured pose space. That [campaign is now complete](mobile-full-capture-results.md):
it discovers a consequential alternative native pocket, but its estimates
remain dominated by a few rows. It integrates an **inner capture domain**,
not the entire physical spherical vessel. The original coverage review and
frozen design below explain its scope.

## Frozen target and existing evidence

Use the exact observed scaffold in
[config.json](../runs/mobile-competing-reference-preparation-20260921/config.json):
fixed neighbors are original body 2 (A) and body 1 (B), moving body 0,
capture center zero, capture radius 170 Å, depletant radius 1.5 Å and
activity 0.035 Å⁻³. The slightly deformed observed scaffold must remain
unchanged. Its frame mapping is preserved in
[fixed-snapshot.json](../runs/mobile-competing-geometry-20260921/fixed-snapshot.json).

The completed native-R4 and competing-R3/3–5/5–8/8–12 integrals use this
target; see the [finite-region comparison](../runs/mobile-competing-reference-comparison-20260921/report.md)
and [outer comparison](../runs/mobile-competing-outer-comparison-20260921/report.md).
The configuration inventory at this review found no completed whole-domain
`basin-normalizer` estimate on this exact scaffold. All 198 archived
`basin-normalizer` configuration copies inspected described the older
one-neighbor, capture-radius-18 problem. Its
[global shell analysis](mis-refined-normalizer.md) supplies an analysis pattern,
not transferable weights.

## Contacts omitted by capture radius 170

The physical wall has radius 223.32617672378387 Å. The frozen atom union
has orientation-independent bound 49.48702190349181 Å, so every pose with
center norm at most 170 Å clears the atomic wall by at least
3.83915482029206 Å. This is a sufficient inward bound, not an exhaustive
description of wall-valid poses.

A deterministic geometric check already performed during this review found
these hard-valid, wall-valid native contact placements outside capture:

| Fixed anchor | Frozen motif ID | Center norm, Å | Minimum atomic wall clearance, Å | Minimum hard-atom gap to A / B, Å |
|---|---:|---:|---:|---:|
| A, original body 2 | 0 | 175.2558701873374 | 6.9238835897530 | 0.0426451350061 / 20.4759504511492 |
| B, original body 1 | 6 | 180.0004213325064 | 7.1710901060667 | 21.8151382981887 / 0.0369183579553 |
| B, original body 1 | 13 | 175.2580487298912 | 14.5076752693510 | 21.8435881534363 / 0.0369183579553 |

These compose the catalogue's relative pose onto the corresponding exact
fixed anchor: `t = t_anchor + R_anchor t_motif`,
`R = R_anchor R_motif`. For the first witness, the laboratory pose is

```json
{
  "position": [-110.48995177833392, 136.0113205273436, -2.7406713685702364],
  "orientation": [0.6882360353297444, -0.1701385672833947, 0.5760388146466761, -0.4068947180989217]
}
```

Quaternion order is `[w,x,y,z]`. The check transformed every frozen atom,
evaluated sphere-to-wall clearances and minimum interbody atomic sphere
gaps against **both** fixed neighbors. A positive hard gap below 3 Å
implies overlap of the exclusions inflated by 1.5 Å. These witnesses prove
omitted contact support; they provide no estimate of its physical weight.
The numbers above preserve the completed static check; documenting them
did not rerun geometry or sampling.

Exact inputs for that check:

| Archived input | SHA-256 |
|---|---|
| [Physical config](../runs/mobile-competing-reference-preparation-20260921/config.json) | `2ed28d5b618209d51dd8a0381c95a8e58853a7e9780f66d53fb027ec07318cf1` |
| [Hard atom union](../runs/mobile-competing-reference-preparation-20260921/provenance/shape.json) | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |
| [Native pair catalogue](../runs/mobile-posterior-pilot-preparation-20260921/provenance/native-pair-motifs.json) | `4c0309b21bf7c31e12bb2f11fb103b20c5772a8472bfc59c2cc0ff8c6ff51282` |
| [Observed snapshot mapping](../runs/mobile-competing-geometry-20260921/fixed-snapshot.json) | `4f74942180010463fbefb51c6b116fad675428c27533d6558e8bb0b87e1029b9` |

## An exhaustive partition within the unchanged capture

The subsequently executed pilot uses four fresh populations of 8,192 unconditional
draws, two independent Poisson clouds per valid pose, and λ/z=64. Freeze the
existing exact reciprocal 150-component proposal at covariance scale one,
with uniform cube/Haar fractions 0.1 and 0.5 in separate arms, and marginalizes
over both fixed proposal anchors. The [normalizer estimator](basin-normalizers.md)
then uses the full physical pose density, not the sampled component or anchor
density. No Jacobian is added again to that physical-density denominator.

Prespecify three disjoint indicators covering every hard-valid captured pose:

1. Original registration score q≤1.
2. q>1 and original competing-chart radius ρ≤12.
3. q>1 and ρ>12, assigning any exact chart seam to this remainder.

Native R4 versus its remaining q≤1 region can be an additional subdivision.
Keep bound/unbound contact labels. Every estimate retains every unconditional
draw in its denominator, including all invalid zeros. Compare the new R12
estimate with the independent regional result; do not pool rows with different
proposal densities as though they shared a sampling law. Report paired-cloud
noise, population dispersion and weight concentration for the remainder
separately. Full support and an exhaustive partition do not certify numerical
coverage or bound unobserved high-weight modes.

Here q refers to the single original native reference in config metadata.
Consequently q>1 is **not** synonymous with nonnative attachment: it includes
other legitimate crystallographic sites. Preserve this original partition for
comparison, and add a separate fixed, stateless native-contact classifier.
The archived [TetramerOrder observer](../runs/mobile-reciprocal-atlas-benchmark-20260921/reciprocal/provenance/reference/scripts/tetramer_order.py)
and its frozen motif/classification data provide the entry criteria. Use
instantaneous entry tests rather than history-dependent registry persistence
for independent integration poses; retain motif identities and any multiple
matches. This classification changes reporting, not the physical target.

## Mobile returns and the full physical wall

The [saved mobile return](../runs/mobile-reciprocal-contact-return-diagnostic-20260921/analysis.json)
at ρ=18.84 lies outside R12, despite being only 2.31 Å and 4.10° from the
initial relative pose. Its other neighbor had moved. It therefore cannot
supply a same-fixed-scaffold weight or demonstrate return to a fixed region.
Its relative pose could inform a separately frozen proposal chart after
composition onto the original anchor and checks against the original other
neighbor and wall. Fresh weights would still be required.

To integrate the entire physical vessel, add the explicit atomic-wall
indicator and a proposal support covering every feasible body center, while
retaining the same fixed scaffold and bath. A conservative cube with half-side
`wall_radius + shape_bound` suffices for support; merely enlarging capture
without the atomic-wall indicator changes the target. Keep inside-D170 and
outside-D170 contributions separate. The full-vessel protein experiment
remains unperformed. The whole-D170 pilot was executed after this initial
review; its [results and limits](mobile-full-capture-results.md) are separate.

The completed schema3 Python audit controls are recorded in
[validation.json](../runs/basin-reciprocal-python-audit-validation-20260921/validation.json).

The [stateless native-entry classifier](native-contact-regions.md) is now
implemented and its criteria and inputs are frozen. The full atomic-wall
extension of [the normalizer](basin-normalizers.md) also passes independent
sphere and dumbbell controls, including actual Rust/Python serialization and
unchanged no-wall output. These establish available tools; full-vessel protein
statistical weights still require a separately declared campaign.
