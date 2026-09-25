# Supplied physical inputs

## Offline assembly quick start

Use the broader **native-informed coverage atlas** with either supplied
12-tetramer start. From the repository root, with Rust and the locked registry
dependencies already cached:

```bash
cargo build --offline --locked --release --bin tetramer-mc

target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --out runs/coverage-free-10000 --sweeps 10000 --sample-every 10

target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-seeded.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --out runs/coverage-seeded-10000 --sweeps 10000 --sample-every 10
```

Every invocation needs a fresh output directory. All runtime assets are
included in `examples/`; no campaign files or original protein-nucleation
checkout are needed. These commands write JSON/GSD trajectories, move records,
checkpoints, and input/source provenance. Add `--no-gsd` for JSON output only.

Both starts use the same 354.50820786337056 Å sphere, depletant radius 1.5 Å,
activity 0.035 Å⁻³, and auxiliary Poisson intensity ratio 64. Every tetramer
moves, including the initial seed. Local steps are 0.2 Å and 1 degree. Half
of ordinary attempts select a global move; half of those select frozen
posterior transport with correlation 0.9. The mixture has a 10% uniform floor.
Spherical GCA and sphere-center shifts run after each ordinary sweep.
There is no online fitting, contact-memory update, or reversible-jump update
in these two configurations.

The newer entry-shell importance guide measures a fixed contact-region
integral. It is a separate analysis sampler and is **not an assembly move**
enabled by these commands.

### Experimental dimer/trimer transport phase

`examples/spherical-cluster-phase.json` adds a fixed-duration event phase to the
seeded preparation. It selects contact-connected pairs/triples, applies a local
rigid move or the existing frozen covariance map to one handle, and carries the
other members. The physical gate uses the entire moving exclusion union. It
retains the ordinary local/global, GCA and center-shift schedules.

```bash
target/release/tetramer-mc run \
  --config examples/spherical-cluster-phase.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --free-tetramers 256 --tetramer-concentration-um 500 --seed 20260926 \
  --depletant-radius 1.4 --depletant-activity 0.0275 \
  --out runs/cluster-phase-seed8-free256 --sweeps 10000 --sample-every 100
```

The `cluster_phase` object sets duration, per-subset size rates, local/transport
mixture, correlation and local move sizes. Duration is an **algorithmic clock**,
not CPU time or physical kinetics. Endpoint trajectories retain the physical
measure; event-indexed records must not be treated as equilibrium samples.
The frozen atlas remains native-informed. Removing this object, setting it to
null, or setting duration to zero preserves the existing physical RNG sequence.
The initial implementation supports spherical boundaries and frozen proposals;
assembly bias is corrected per elementary cluster attempt. This is an experimental
sampling control, not a demonstrated protein assembly speedup. See the
[figure, balance derivation, tests and limitations](../docs/rigid-subset-phase.md).

### Experimental contact-conditioned GCA axes

Use `--config examples/spherical-conditioned-axis.json` with the same coverage
model to enable geometry-guided half-turn selection. The selector applies its
own reversible correction before any existing auxiliary or assembly-bias gate.
It is optional and has not demonstrated a protein contact-exchange speedup;
see [construction, settings and benchmark](../docs/conditional-half-turn.md).
Removing its `gca_axis` object restores the original isotropic GCA sequence.

### Choose count and concentration

The runner can generate a new free preparation without editing the JSON:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --free-tetramers 24 --tetramer-concentration-um 106.8 --seed 20260924 \
  --out runs/coverage-free-N24-106p8uM --sweeps 100000 --sample-every 100
```

Both count and concentration are required together. `--concentration-um` is an
alias for `--tetramer-concentration-um`. Concentration counts **tetramers**,
not monomers or depletants. A requested concentration `c` in μM gives number
density `c * 6.02214076e-10` Å⁻³ and full vessel volume
`(seed_count + free_count) / number_density`.
A spherical template uses radius `(3 V / (4 pi))^(1/3)`; a periodic template
uses a cubic box of side `V^(1/3)`. The latter still requires the existing
periodic-cell bound `L > 4 * (body_bound + depletant_radius)`.

The bodies listed in `seed_labels` are retained, with their orientations and
seed geometry preserved. The requested count is the number of **new free
bodies**, not the total: existing non-seed bodies are replaced. Retained seed
bodies appear first in original index order; labels are remapped accordingly.
All bodies, including the seed, remain mobile during MC. `--discard-seed`
explicitly requests an all-free replacement and requires the count/concentration
flags. Without initialization flags, the original poses are unchanged.

For an eight-tetramer seed plus 248 free tetramers at total concentration 500 μM:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-seeded.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --free-tetramers 248 --tetramer-concentration-um 500 --seed 20260924 \
  --depletant-radius 1.4 --depletant-activity 0.025 \
  --out runs/seed8-free248-500uM-rd1p4-z0p025 \
  --sweeps 100000 --sample-every 100
```

In a spherical vessel seed positions stay exactly as supplied; initialization
fails if the resized wall would exclude them. A compact periodic seed can be
unwrapped using the original cell and translated together to fit the new cell.
Every old and new minimum-image pair displacement is checked; a winding,
ambiguous or oversized seed is rejected rather than having its contacts altered.
Duplicate and invalid seed labels are errors. Seed-seed depletion contacts are
retained, while hard-core overlaps remain invalid.

Shape and move parameters remain those of the template; bath parameters may be
overridden separately as described below. Changing the total number of particles
does not resize a frozen assembly-bias table; that table must remain valid for
the requested system.

New free orientations are independent Haar rotations. Their positions are
placed sequentially with separated **exclusion bounds**, including the depletant
radius, against every seed body and earlier free body. Thus new free bodies
have no initial depletion contacts; supplied seed contacts are preserved. In a sphere, the centers are drawn
uniformly inside radius `R - body_bound`; the depletant bath still permeates the
atomic wall. Periodic separation uses minimum images. This conservative
nonequilibrium preparation can fail at concentrations where atomic-core packing
would still be possible. Placement stops with an error after 100,000 candidate
attempts; lower the concentration or change the seed instead of treating failure
as evidence about physical stability. At most 100,000 free bodies may be
requested, and the total count must be at least two. Zero or one free body is
allowed when the retained seed supplies the remaining particles.

`--seed` overrides the template master seed for both preparation and MC, using
separate RNG streams. The generated configuration, positions, actual volume,
concentration, seed and placement certificate are saved in
`provenance/input-config.json`; the original file is archived separately as
`provenance/template-config.json`. Old preparation metadata is nested under
`template_metadata`, and the output manifest identifies the actual model.

To continue this generated run, use its **archived input configuration**, with
no count, concentration, discard-seed or random-seed flags. For example, to reach 200,000 total sweeps:

```bash
target/release/tetramer-mc run \
  --config runs/coverage-free-N24-106p8uM/provenance/input-config.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --resume runs/coverage-free-N24-106p8uM/checkpoint.json \
  --out runs/coverage-free-N24-106p8uM-continued \
  --sweeps 200000 --sample-every 100
```

The generated archive uses resolved absolute asset paths. Those files must remain
available for continuation. `config.json` is the effective segment snapshot;
it is not the byte-identical input used by the checkpoint hash.

### Choose depletant radius and activity

`--depletant-radius` (Å) and `--depletant-activity` (Å⁻³) independently override
the input JSON's `depletant_radius` and `reservoir_density`. Both must be finite
and nonnegative. Activity zero gives the hard-only limit. These flags work with
the stored poses or together with a newly generated free start:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --free-tetramers 24 --tetramer-concentration-um 106.8 --seed 20260924 \
  --depletant-radius 1.4 --depletant-activity 0.04 \
  --out runs/coverage-N24-rd1p4-z0p04 --sweeps 100000 --sample-every 100
```

The supplied frozen map, its weights and covariance matrices stay unchanged.
Exclusion geometry, the Poisson acceptance gate, and collective moves use the
new bath. At positive activity, the auxiliary intensity remains
`poisson_lambda_ratio * activity`.
The proposal correction remains applicable; efficiency at the new parameters
must be measured separately. This command does not refit the map.

With a generated start, the new radius also sets the initial exclusion-bound
separation. Without count/concentration flags, positions and vessel size are
retained, and only the bath changes. A larger radius can violate the existing
periodic-cell bound, in which case the runner reports an error.

As for count/concentration overrides, the effective configuration is archived
at `provenance/input-config.json`, with the original under
`provenance/template-config.json`. Older preparation metadata is retained as
provenance rather than a contact guarantee at the new radius. Continue with
the archived input and **no override flags**, using a fresh output directory;
changing a bath parameter is a new run, not checkpoint continuation.

### Choose the proposal model

The same two configuration files support all three frozen models via `--model`:

| Model file | Base Gaussians / virtual branches | Role |
|---|---:|---|
| `frozen-coverage-reciprocal-mixture.json` | 178 / 328 | Broader native-informed coverage; quick start above |
| `frozen-reciprocal-mixture.json` | 150 / 300 | Original native-informed reference |
| `frozen-geometry-reciprocal-mixture.json` | 96 / 192 | Geometry-only inter-tetramer proposal control |

The coverage atlas preserves both reciprocal branches of the original 150
components and adds 28 ordinary charts for known native and competing contact
geometries. It reuses existing covariances without a new fit. Its weights
specify the proposal, not equilibrium contact probabilities. In the short
four-body control, coverage reached native-connected attachment in 2/2
free-start runs, versus 1/2 for the original atlas. This establishes
accessibility, not an established speedup, equilibrium preference, or
template-free assembly. The twelve-body examples are different initial
conditions from that control.

The geometry-only model is an exact copy of the validated 96-component
reciprocal control. Its centers come from shape-only contact rays and its
covariances from contact-pivot geometry; no supplied inter-tetramer registry
was used. The protein shape still contains a native tetramer, and the seeded
configuration supplies a native initial scaffold. In the four-body control,
neither of its two free-start runs reached native attachment within 2,000
sweeps; the two initially attached runs retained their native contacts. These
finite runs do not establish a thermodynamic obstruction. See the
[geometry-only control](../docs/geometry-only-growth-control.md) and
[current evidence summary](../docs/contact-evidence-roadmap.md).

For a free-start geometry-only proposal control:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-geometry-reciprocal-mixture.json \
  --out runs/geometry-free-10000 --sweeps 10000 --sample-every 10
```

Use `spherical-reciprocal-seeded.json` and a fresh output path for its seeded
counterpart. To reproduce the original reference, select
`examples/frozen-reciprocal-mixture.json` instead. Reciprocal inversion swaps
which proper relative pose describes a contact; it does not invert protein
chirality.

`--model` selects the actual proposal. The shared configurations' descriptive
metadata still describes the original 150-component atlas, including its
native-informed label. When overriding the model, use the output manifest's
`model_sha256` to identify the proposal; the descriptive input metadata is not
rewritten. The geometry model's historical `construction.shape_path` is also
provenance only; runtime shape loading uses the configuration's relative path.

Exact model SHA256 values:

- Coverage: `feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e`.
- Original: `dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3`.
- Geometry-only: `e90a7c85c5bdb2a587071949f0ae6527dba080594229434080d0761e4125983b`.

Both configurations passed eight-sweep checks with the original atlas,
including JSON/GSD readback, and two-sweep checks with coverage, including
model binding, checkpoint/final-frame agreement, and atomic overlap/wall
checks. The geometry-only model passed its separate four-body campaign;
these twelve-body configurations have not been smoke-tested with that model.
No completed test establishes crystal assembly or equilibration.

### Continue, compare, and view

To continue the coverage free run to a **total** of 20,000 sweeps, retain its
configuration and model and write to a fresh directory:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --resume runs/coverage-free-10000/checkpoint.json \
  --out runs/coverage-free-20000 --sweeps 20000 --sample-every 10
```

The sweep count is the cumulative endpoint, not an additional count. For an
independent replicate, make a configuration copy with a fresh `seed` and start
a fresh run. For the matched independent-redraw control, remove
`frozen_posterior` (or set it to `null`), preserving `global_probability`, the
frozen model, the uniform component, local step sizes and collective-move
schedule. Setting only `frozen_posterior.correlation` to `0.0` retains a
posterior-component move and is a different control. Changing radius or activity
defines a different physical run; efficiency of an existing atlas at those
conditions is not established.

Export a self-contained offline viewer with Python's standard library:

```bash
python3 tools/export_viewer.py --run runs/coverage-free-10000 \
  --out runs/coverage-free-10000/viewer.html --native-bonds off
```

Open `viewer.html` in a browser. The default frame is sphere-centered; the
coordinate-origin selector displays accumulated sphere-center shifts.
This portable command disables the optional native-bond classifier, whose
research reference data are separate. The earlier `frozen-relative-mixture.json`
and earlier configurations remain available for historical controls.

## Earlier pilot inputs

These files are copied from the existing repaired-1LYZ frozen-proposal pilot:
`protein-nucleation/results/learned-tetramer-assembly/production/inputs/`.
The two config files adapt that pilot's replicate-zero initial configurations
to portable relative paths. The shape is the existing repaired hard model,
not a new atomic geometry optimization.

- Frozen model SHA256:
  `2e534634e6e2fe83da969a2867504c293af8e90a0cf3c4330cdc9e38a6770064`.
- Tetramer shape SHA256:
  `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9`.
- Four proteins per rigid body; 4,004 atomic spheres per tetramer.
- Twelve bodies in a 571.4643787085516 Å periodic cube; total tetramer
  concentration about 106.8 μM. Seeded and dispersed starts have the same total
  N and volume, but different initial free-particle inventories.
- Depletant radius 1.5 Å, activity 0.035 Å⁻³. The auxiliary Poisson intensity
  ratio of 16 affects sampling efficiency, not the physical bath.

The Gaussian model uses 24 fitted components and four broad Gaussian floors;
the production wrapper adds a separate 10% uniform pose branch. It was fitted
to neighborhood poses from seeded and dispersed exploration trajectories,
with trajectory-held-out model selection. It therefore contains supplied
crystallographic information. Its weights are proposal weights, not measured
equilibrium basin probabilities.

`native-pair-motifs.json` and `monomer-shape.json` are optional post-hoc analysis
inputs. They never enter the production proposal density or acceptance gate.

## Spherical starts

`spherical-seeded.json` and `spherical-oligomers.json` preserve assembled
fragments from the 400-sweep Rust learned pilot. Exclusion-bound neighborhoods
were unwrapped, then re-placed without hard overlap inside an equal-volume
sphere of radius 354.50820786337056 Å. Input metadata records source checkpoint
hashes and fragment labels. These are new preparations, not equilibrium maps
from periodic to spherical boundaries. Both examples enable GCA and common
center shifts after every ordinary learned/local sweep. The default mixture
is frozen; add `"auxiliary_transport": {}` for the normalized Gaussian-mean law.

`spherical-free.json` and `spherical-free-transport.json` start with twelve
independently oriented, separated tetramers. Every pair of exclusion bounding
spheres is disjoint. Metadata records the preparation seed, pose hash, and
separation certificate. The starts are prepared quenches, not equilibrium
fluid samples. The second config enables auxiliary transport; both enable GCA
and center shifts. Add `"reversible_jump": {}` to the transport config for
variable proposal-component counts. The [campaign guide](../docs/rj-assembly.md)
explains a matched test with and without explicit inter-tetramer native charts.

`contact-memory-defaults.json` is an options object for the campaign launcher's
`--memory-json` flag, not a complete simulation input. Equivalently, add
`"contact_memory": {}` to a spherical learned-proposal config. The fixed bank
uses geometry-only pair preparations and reversible pair updates; this does
not remove native information from any supplied base atlas. See the
[balance argument and matched controls](../docs/contact-memory-balance.md).

`spherical-conditional.json` uses the same separated preparation as
`spherical-free.json`, with the normalized current-geometry GMM closure and
`s=6`. Run with `--method learned` and no model file. It fits means, full
covariances, weights and a finite count law without a separate memory bank;
see [the closure guide](../docs/conditional-closure.md).

### Experimental oligomer-conditioned guide

The optional [joint guide](../docs/oligomer-guide.md) evaluates every carried member
in a fixed external-anchor neighborhood. Try it with the same physical settings
as the cluster control:

```bash
target/release/tetramer-mc run \
  --config examples/spherical-oligomer-guide.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --free-tetramers 256 --tetramer-concentration-um 500 \
  --depletant-radius 1.4 --depletant-activity 0.0275 --seed 20260926 \
  --out runs/oligomer-guide-seed8-free256 --sweeps 10000 --sample-every 100
```

`cluster_phase.guide.steps` controls fixed inner work (default example: 4),
`anchor_count` the spectator-only pool size (4), and `score_power` the joint guide
strength (1). Remove `guide` or set `steps` to zero for the existing cluster
proposal. This is an experimental sampling control, not a recommendation based
on a demonstrated protein mixing speedup.
