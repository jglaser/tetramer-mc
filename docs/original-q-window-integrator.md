# Integration over the original registration coordinate

The broad global proposal discovered contacts just outside the native region
that narrower proposals had missed. The integrator now accepts a finite window
of the **original** registration coordinate, so those contacts can be measured
without changing the physical metric or optimizing a new region around a
selected pose. The first application is the strict shoulder `1 < q < 2`.

The physical conditions remain depletant radius 1.5 Å, activity 0.035 Å⁻³,
capture radius 18 Å and both fixed neighbors A and B. A is a coordinate anchor;
both neighbors contribute to hard exclusion and to the depletion overlap
`C = |E(pose) ∩ (E(A) ∪ E(B))|`. This is a conditional docking integral. It
does not include the free-energy cost of assembling the prescribed neighbors.

## Target and complete proposal

For an original metric `q(pose)` and declared endpoints, the target is

```
Q_window = ∫ I_window(q) I_capture I_hard exp(z C) d³t dH(R),
```

where `H` is normalized SO(3) Haar measure. Numerical volumes use Å³.
`QWindow` stores the bounds and their open/closed conventions. The default
is the previous inclusive native region `0 <= q <= 1`.

To cover an upper endpoint `b`, the product-cover constructor receives member
displacement tolerance `b*delta` and angle tolerance `min(b*alpha, pi)`.
It then recomputes its conservative geometric angular cap. Scaling the cap
previously derived at `b=1` would be incorrect because its arcsine dependence
is nonlinear. The target predicate and every reported q still use the original
metric. The [coverage derivation](qwindow-cover-proof.md) also explains why
subtracting the inner **cover** would incorrectly remove part of the shoulder.

An optional frozen Gaussian guide retains its exact translation/Haar density.
The denominator is the full summed density of the cover, guide and cube/Haar
branches, including contributions from branches other than the one drawn.
For every unconditional draw the contribution is

```
Y = I_window I_capture I_hard Wbar / g_full(pose).
```

The independent positive Poisson clouds satisfy
`E[Wbar | pose] = exp(z C)`. Rejected draws have `Y=0` and remain in the
original denominator `N`. No cloud outcomes are used to retry or select a
pose. Conditional on a frozen guide this estimates the declared physical
integral; it does not require the guide's training distribution to be at
equilibrium. It also does not establish the validity of an uncorrected,
continuously adapting Markov chain.

## Commands and records

For an already prepared physical configuration and frozen guide:

```bash
target/release/native-region-normalizer \
  --config path/to/config.json --out runs/fresh-shoulder \
  --model path/to/model.json --model-weight .75 \
  --model-uniform-probability .05 --model-anchor-index 0 \
  --q-min 1 --q-max 2 --q-lower-open --q-upper-open \
  --samples 16384 --cloud-replicates 2 --lambda-ratio 64 --seed 99501010
```

Omitting the model produces the pure geometric reference. Independent campaigns
use `tools/run_native_region_reference.py`, which accepts the same q-window
options and archives the shape, effective configuration, executable, model
and analysis tools before launching. Use a fresh output directory and fresh
seeds for a new estimate.

A nondefault window writes schema 4 with both the original and cover metrics,
exact endpoint conventions and a pose/full-density record for every draw.
Its summary fields are `region` and `hard_region`, rather than `native`.
The optional `region_core`/`region_shell` diagnostic continues to split at
**original q=0.8**; it is not a new subdivision of each supplied interval.
Every shoulder observation therefore belongs to `region_shell`.

The default schema 2/3 paths retain their previous random streams and rows.
On the physical AB configuration, 512 pure-cover and 512 guided draws replayed
byte for byte against the earlier executable. This tests compatibility, not
physical convergence.

## A tighter independent reference

The mean squared displacement of the rigid-member centers bounds the original
maximum displacement from below. Its exact quadratic form couples translation
and rotation. The [moment-cover derivation](rms-quaternion-cover.md) turns that
necessary condition into a complete outer ellipsoid in the existing Cayley
chart for this zero-centroid, noncollinear metric. Its uniform latent sampler
has a known physical Jacobian and does not use fitted contact locations.

The frozen preparation is
`runs/ab-shoulder-cayley-cover-preparation-20260920`. Its radius-four six-ball
is a geometric construction, not a Gaussian probability contour. The mapped
cover has volume at most 0.046683 Å³, versus 0.619534 Å³ for the product cover:
at least 13.27 times less proposal volume in the real-arithmetic construction.
The floating-point guard is conservative by design but is not an interval
arithmetic certificate. The full max-error, angle, capture and atomic hard
predicates remain target masks; satisfying the moment bound alone is not a
valid contact.

`latent-region-normalizer` now accepts optional boolean fields
`minimum_original_q_inclusive` and `maximum_original_q_inclusive`; both default
to true for older regions. This permits exactly the same open shoulder in the
geometric reference. A generic latent ellipsoid need not cover a whole q-window;
completeness here follows from the stated construction and its preconditions.

The reusable preparer requires NumPy and SciPy, one reference, an exactly zero
member centroid, a positive definite guarded rotational moment and an angular
cap below pi. Unsupported cases fail explicitly and can use the product-cover
integrator instead. Preparation records geometry probes without Poisson clouds:

```bash
python tools/prepare_cayley_rms_cover.py \
  --config path/to/config.json --out runs/fresh-moment-cover \
  --q-min 1 --q-max 2 --probes 4096 --seed 99511010

python tools/run_latent_region_campaign.py \
  --config runs/fresh-moment-cover/config.json \
  --region runs/fresh-moment-cover/region.json \
  --binary target/release/latent-region-normalizer \
  --out runs/fresh-moment-reference --samples 262144 \
  --replicates 4 --workers 4 --seed 99521010 \
  --lambda-ratio 64 --cloud-replicates 2

python tools/analyze_latent_region.py --root runs/fresh-moment-reference
```

Changing `--q-max` recomputes the geometric cap and enclosing ellipsoid. For
example, an upper endpoint 1.1 gives an independent reference on the innermost
shoulder band. Its result must be compared with that same band of the broader
calculation, not with the whole shoulder. The original full-window preparation
and each new band remain separate, immutable records.

## Validation and interpretation

Analytic sphere controls check nonzero metric centroids, angular saturation,
the nonlinear cover bound, open endpoints, full guide/cube/cover densities,
hard-volume and depletion-weight integrals, and an active second physical
neighbor distinct from the coordinate anchor. Independent Python audits check
every recorded pose, original q, density/Jacobian, unconditional sample count
and archived input identity. A real-binary synthetic smoke campaign checks
runner-to-analyzer compatibility for pure and guided windows.

The [fresh shoulder comparison](shoulder-proposal-guides.md) shows why total
importance ESS alone cannot choose a proposal: the narrow weighted guide has
better apparent efficiency but misses weight seen by broader guides farther
into the shoulder. A complete cover proves support, not that a finite sample
has reached its important poses. Population and proposal sensitivity remain
necessary, and importance ESS per CPU is not a Markov-chain mixing rate.
