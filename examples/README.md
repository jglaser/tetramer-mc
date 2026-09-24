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
