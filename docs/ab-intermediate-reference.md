# Complete reference for the original AB intermediate region

This reference integrates the **entire original interval `2<=q<5`** for
one mobile tetramer with the same two prescribed neighbors A and B. The
capture radius is 18 Å, depletant radius 1.5 Å, and activity 0.035 Å⁻³.
Both neighbors remain in the hard predicate and the union of exclusion
regions. Translation volume and normalized SO(3) Haar measure are the
physical integration measure. The original 2 Å / 15° registration metric
is unchanged.

The region is a conditional docking test. It does not integrate `q>=5`,
determine the cost of assembling the fixed neighbors, or establish an
assembly outcome.

## Why this band needs a reference

The previous global proposal had support throughout capture:
`0.1 cube/Haar + 0.9 frozen atlas`, with the full density used for every
draw. Its independent width controls nevertheless show substantial
proposal sensitivity. Combining the disjoint intermediate and distant
contributions gives:

| Gaussian width | Unconditional draws | log Q(q>=2) | Importance ESS | Row / population RSE | Valid 2<=q<5 poses | log Q(2<=q<5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 15.29501 | 12.50 | 28.3% / 9.1% | 1 | 1.18030 |
| 2 | 65,536 | 16.55607 | 18.95 | 23.0% / 24.5% | 6 | 1.34252 |
| 4 | 65,536 | 18.93701 | 2.27 | 66.4% / 52.7% | 9 | 5.09236 |

These are separately estimated historical populations, not pooled
production. Sources are
`runs/ab-global-contact-4x8192-l64-20260920/assessment/analysis.json`
and `runs/ab-global-widths-2-4-4x16384-l64-20260920/assessment/analysis.json`.
The roughly 38-fold change in the `q>=2` estimate is separate from the
previously discovered shoulder mass at `1<q<2`. The very few intermediate
observations do not bound its mass.

The earlier uniform competitor R3 reference measures only a finite
ellipsoid restricted to `q>=5`. Its log Q=14.4370 is a useful regional
control, but does not cover this intermediate band or the whole remainder.
See [ab-competing-contact-proposals.md](ab-competing-contact-proposals.md).

## Complete cover without a fitted basin

The existing `derive_model` construction uses the four original rigid
member centers. Their centroid is exactly zero. Let `M` be their covariance
and `A=tr(M)I-M`. For displacement `t-t0` and the body-frame Cayley
coordinate `c` of `R0ᵀR`, the exact member mean squared error is

`RMS² = |t-t0|² + 4 cᵀ A c/(1+|c|²)`.

On `q<=5`, every member error is at most `a=5*2=10 Å`. Recomputing the
complete angular bound gives `h=52.01056121794498°`, below the nominal
75° cap. Therefore `k=cos²(h/2)=0.8077581062002825` and

`|t-t0|² + 4 k cᵀ(A-kappa I)c <= a²`

contains the entire target. Here `kappa=2.045448465607961e-9 Å²` is the
existing outward numerical margin. The guarded rotational moment is
positive definite, with smallest eigenvalue 161.38298121220961 Å².
The geometric lower bound and cap follow the same norm/subtraction formula
as `NativeCover` in `src/native_region.rs`; the physical metric is not
rescaled. The margin is a floating-point safeguard, not a formally rounded
interval certificate. Full conventions and preconditions are in
[rms-quaternion-cover.md](rms-quaternion-cover.md).

The Gaussian-shaped parameter object specifies coordinates only. Its
mean is zero, translation covariance is the identity, rotation covariance
is `ell² A_chart⁻¹/(4k)`, and `ell=55.02283113084892 Å`. A uniform latent
six-ball of radius 10 is decoded in neighbor A's proper frame. Its exact
Jacobian is

`J(u)=det(L)/[ell³*pi²*(1+|c|²)²]`.

With `V6=pi³*10⁶/6`, the unconditional proposal density is `1/(V6*J)`.
Every outside-window, hard-invalid, or capture-invalid draw remains zero:

`Q_hat = (V6/N) sum 1_[2<=q<5] 1_capture 1_hard J(u) Wbar`.

The physical cover volume lies between 10.547066757639008 and
14.980389996780092 Å³, compared with 46,656 Å³ for the cube/Haar domain.
This geometric compression does not by itself guarantee useful physical
variance. No smaller outer cover is subtracted: such subtraction could
remove valid intermediate poses.

## Frozen preparation and geometry checks

The preparation is
`runs/ab-intermediate-cayley-cover-preparation-20260920`.
Its top-level `region.json` is the canonical production input, with lower
endpoint **closed** and upper endpoint **open**. The source configuration
was `runs/ab-shoulder-cayley-cover-preparation-20260920/config.json`.

| File | SHA-256 |
| --- | --- |
| `region.json` | `ed05ecd2208c666b1c6264019a160e81e95bd90e89b66c06494df1ce94520ac2` |
| `model.json` | `6a8c37f5f2af7d464a5810af6f444efb25403afc96b06e026a6ce4c8f633d110` |
| `config.json` | `60d3d3eda0ce7084b88a114036b532d35d4a4d894b19e776994920f71094c55e` |
| `protocol.json` | `3ae8df0155b7321d3deb206f95d8080f4c421a46f419f138e4a9f736a881301a` |
| Physical shape | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |

The archived driver is `provenance/prepare_ab_intermediate_reference.py`;
its source dependencies, original configuration, shape and relevant Rust
geometry source are archived alongside it. `sha256.json` seals the entire
preparation. The constructor used no fitted samples.

All 16 historical hard-valid intermediate poses were extracted with their
original rows and source hashes. They are all inside the new cover; the
largest latent radius is 6.893052819223061, below 10. This check tests
known-pose coverage; completeness comes from the geometric enclosure.

Exactly **2,048** independent uniform-latent poses were generated with
seed **99901010**, with no Poisson clouds:

| Geometry predicate | Count |
| --- | ---: |
| Capture-valid | 2,048 |
| Hard-valid against AB | 90 |
| Original q in the intermediate interval | 346 |
| Hard/capture/window-valid | 14 |
| Valid and contacting both neighbors | 8 |

The maximum independent density reconstruction error was 4.98e-14.
Replaying every coordinate, original q and Jacobian gave zero difference.
The Python atom-union checks took 24.30 CPU seconds. This is geometry-audit
cost and must not be extrapolated as Rust physical-kernel cost.

The existing generic preparation tool uses both endpoints open. Its
unchanged output is retained under `open-window-probe/`, with its replay
under `open-window-replay-audit/`. The canonical region was then frozen
separately with `minimum_original_q_inclusive=true` and
`maximum_original_q_inclusive=false`. All probe rows were reclassified
under those flags. None hit either boundary, so the counts agree exactly.
Adjacent floating-point values around 2 and 5 pass the expected predicate
controls. Under the continuous translation/Haar proposal, the `q=2`
surface has measure zero: it lies in a finite union of member-distance
level surfaces and an angular level surface. The convention does not
alter the integral, but the production classifier matches `[2,5)` exactly.

For an optional fresh geometry-only probe using the existing tool, the
following independent seed is reserved. This command does not run the
physical normalizer; its original-q probe mask is open at both endpoints
as described above:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xvg/protein-nucleation/.venv/bin/python -B \
  /home/xvg/tetramer-mc/tools/prepare_cayley_rms_cover.py \
  --config /home/xvg/tetramer-mc/runs/ab-intermediate-cayley-cover-preparation-20260920/config.json \
  --out /home/xvg/tetramer-mc/runs/ab-intermediate-independent-geometry-probe-20260920 \
  --q-min 2 --q-max 5 --anchor-index 0 --probes 2048 --seed 99902010
```

## Prespecified physical control

The completed control used four independent populations of 131,072 draws
each, using seeds `99911010+1009*i`, Poisson intensity/activity ratio 64,
and two clouds per valid pose. The campaign is
`runs/ab-intermediate-cayley-reference-4x131072-l64-20260920`.
Every population and the independent full-row audit completed successfully.

The pinned executable SHA-256 is
`d5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d`.
The actual launch used:

```bash
/home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/run_latent_region_campaign.py \
  --out /home/xvg/tetramer-mc/runs/ab-intermediate-cayley-reference-4x131072-l64-20260920 \
  --config /home/xvg/tetramer-mc/runs/ab-intermediate-cayley-cover-preparation-20260920/config.json \
  --region /home/xvg/tetramer-mc/runs/ab-intermediate-cayley-cover-preparation-20260920/region.json \
  --binary /home/xvg/tetramer-mc/runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer \
  --samples 131072 --replicates 4 --workers 4 --seed 99911010 \
  --lambda-ratio 64 --cloud-replicates 2
```

The output is already allocated; any later control must use a fresh output
and independent seeds. The geometry rate predicts about 3,584 valid cloud
evaluations at this fixed budget, with substantial count uncertainty. It
does not predict physical ESS, tail coverage or precision. Compare the
result only with the historical **same `2<=q<5` band**, retaining all
unconditional zeros. The `q>=5` remainder remains a separate open question.

## Completed reference and same-band comparison

The 524,288 unconditional draws contain 3,335 valid intermediate poses.
Their physical estimate is **log Q = 12.118911**, with observed relative
SE 94.92%, weight ESS 1.110, and one draw supplying 94.79% of the total.
The four independent population log estimates are 6.03523, 6.74429,
13.50262 and 6.43986. Generation cost is 80.75 CPU seconds.
The hard-region integral is substantially better resolved:
**log Q₀ = −2.364224**, with observed relative SE 1.726%.

The independent comparison preserves each historical global proposal's
original density and all its unconditional zeros. It does not reuse those
draws under a new mixture denominator or pool different widths:

| Original q band | Complete reference log Q | Largest reference weight | Global width 1 / 2 / 4 log Q |
| --- | ---: | ---: | --- |
| 2–2.5 | 1.29541 | 54.17% | unresolved / −0.21264 / 0.61959 |
| 2.5–3 | 2.48452 | 69.95% | 1.18030 / unresolved / 4.66677 |
| 3–4 | 12.06610 | 99.93% | unresolved / −0.07966 / 3.99229 |
| 4–5 | 9.15026 | 96.17% | unresolved / 0.74042 / −0.95918 |
| Entire 2–5 band | 12.11891 | 94.79% | 1.18030 / 1.34252 / 5.09236 |

Each interval includes its lower endpoint and excludes its upper endpoint.
There are no exact q=2 observations in any compared campaign. Every
subband uses the original full campaign denominator. “Unresolved” means
no contributing observations, not zero physical weight. All rows and
original densities were checked: 524,288 reference and 163,840 global
draws, with maximum historical density reconstruction error 1.25e-12.

The new reference finds important contributions in bands poorly sampled
by the old global proposals, but it cannot supply a reliable replacement
normalizer. Its observed paired-cloud variance fraction is 12.68%; most
of its estimated variance comes from rare poses. Increasing cloud counts
alone would therefore leave the main observed problem unresolved.
Inspect the new high-weight contacts and the existing frozen guides before
committing a larger uniform budget. A favorable pointwise guide density
would support a targeted control, not certify finite-sample convergence.

The complete comparison, including hard weights, source hashes, population
errors and cloud diagnostics, is
`runs/ab-intermediate-reference-comparison-20260920/comparison.json`.
Reproduce it into a fresh derived output with:

```bash
/home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/compare_intermediate_reference.py \
  --reference /home/xvg/tetramer-mc/runs/ab-intermediate-cayley-reference-4x131072-l64-20260920 \
  --out /home/xvg/tetramer-mc/runs/fresh-intermediate-comparison
```

The comparison helper also accepts additional separately audited original
hybrid-guide campaigns via `--guided-root`. Such fresh controls must retain
the same physical band, domain and AB interactions. Neither this reference
nor the finite R3 integral resolves the whole q≥5 remainder or establishes
native assembly.

## Existing shoulder guides do not cover the newly observed contacts well

The completed diagnostic in
`runs/ab-intermediate-frozen-guide-diagnostic-20260920/analysis.json`
evaluates the unchanged shoulder mixture and equal-geometry guide at all
16 historical intermediate poses and the 16 largest new reference
contributions. It uses their **full** prospective hybrid density, including
the product cover recomputed at q_max=5 and the cube term.

At every one of the 16 direct-reference extremes, either guide has only
0.02246–0.02335 times the complete Cayley reference density: approximately
43 times less. The largest contribution is at q=3.26781, with full g/c
0.0230193 and Gaussian Mahalanobis radii 75.26 (weighted shoulder fit) and
9.16 (geometry shoulder fit). The fallback terms supply nearly all of the
prospective guided density there. These differ from the inner-shoulder
extremes where the existing guide was already dense.

Independent atom-union checks pass for all 32 inspected poses. The largest
reference contribution has minimum gaps 0.140696 Å and 0.006410 Å to A and
B, respectively; both depletion contacts are present. The smallest gap
across the panel is 0.000582 Å. The diagnostic reuses the original cloud
weights and performs no new physical sampling.

The same reference rows predict only about 10 or 19 hard-valid target
poses for proposed unchanged-guide budgets of 4×16,384 or 4×32,768 draws.
This observed geometric importance diagnostic does not bound unseen guide
mass. Together with the pointwise density checks it gives no reason to
spend those budgets on the unchanged shoulder guides. The next control
will instead fit separate intermediate proposals, freeze them, and sample
fresh populations with the original region, physical numerator and full
proposal correction. Earlier reference draws remain separate training and
comparison evidence; they will not be retrospectively pooled with the new
estimates.
