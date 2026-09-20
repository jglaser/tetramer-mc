# tetramer-mc

Rust Monte Carlo for rigid sphere-union particles in an implicit ideal depletant
bath, with periodic or spherical boundaries. A full-covariance Gaussian mixture proposes relative translations
and orientations; a complete Hastings factor and a conditional Poisson gate
preserve the hard-particle, many-body depletion target. Crystal information is
allowed in training examples but never substitutes for physical acceptance.

The supplied example is twelve repaired-1LYZ tetramers at depletant radius
1.5 Å and reservoir activity 0.035 Å⁻³. The seeded preparation has eight native
tetramers and four dispersed tetramers; the fluid preparation has twelve
dispersed tetramers. Every body moves. This is a sampler, not a physical dynamics
model or a determination of the crystallization phase diagram.

## Build and run

Requires Rust 1.95 or later and Linux for the process CPU clock. Registry
dependencies and the RNG version are pinned in `Cargo.lock`. The vendored
`hoomd-gsd` source writes trajectories without Python.

```bash
cargo build --locked --release --bins
cargo test --locked --release

target/release/tetramer-mc run \
  --config examples/seeded.json \
  --model examples/frozen-relative-mixture.json \
  --out runs/seeded --sweeps 400 --sample-every 10

target/release/tetramer-mc run \
  --config examples/seed-free.json --method local-uniform \
  --out runs/fluid-control --sweeps 400 --sample-every 10
```

Add `--offline` to Cargo commands when dependencies are already cached. Configure
radius, activity, local steps, mixture floor, and Poisson planning budget in the
input JSON. Changing those inputs defines a new run; the example mixture may
be inefficient at other physical conditions, but its correction remains valid.

A matched campaign can run in the background:

```bash
mkdir -p runs
nohup python3 tools/run_campaign.py --out runs/frozen-4000 \
  --sweeps 4000 --replicates 4 --workers 8 \
  > runs/frozen-4000.log 2>&1 &
```

The launcher repeats the supplied initial configurations with independent paired
RNG seeds. It does not create new equilibrated fluid preparations. This campaign
uses a frozen mixture. See `progress.json` and per-job logs.

For spherical GCA, common center shifts, and optional reversible transport of
a conditional Gaussian model, see [the spherical ensemble guide](docs/spherical-ensemble.md).
Use `examples/spherical-seeded.json` or `examples/spherical-oligomers.json`
with the same Rust command. Both collective kernels are enabled in those examples.
For initially isolated tetramers and native-on/off reversible-jump campaigns,
see [the free-assembly guide](docs/rj-assembly.md).

## Output and continuation

- `trajectory.gsd`: standard HOOMD atom-sphere positions/diameters for external
  visualization, plus FP64 body positions and scalar-first orientations in
  `log/tetramer_mc/body_position` and `body_orientation`, and FP64 periodic
  lengths in `log/tetramer_mc/box_lengths`.
- `trajectory.jsonl`: lossless rigid-body frames at fixed sweep cadence,
  including residence after rejections. Physical lengths are Å.
- `moves.jsonl`: proposed and retained poses, component/anchor labels, full
  density correction, Poisson counts, and acceptance. `--no-moves` disables it.
- `checkpoint.json`: complete sweep-boundary state, counts, and input hashes.
- `config.json`, `manifest.json`, `provenance/`: effective settings, frozen input
  copies, model/shape/executable hashes, and the exact application source bundle
  embedded at build time; `summary.json` records CPU cost.

After continuation, `counts` is cumulative; CPU/wall times cover the new segment.
Use the separate `segment_counts` with those timings for throughput estimates.

The standard GSD atom coordinates are FP32 for interoperability. FP64 body
chunks and JSONL are authoritative for precision-sensitive geometry. The GSD
body-id log is a display grouping, not HOOMD's rigid-body center-index field.

Resume into a **new empty directory**, using the same original config/model:

```bash
target/release/tetramer-mc run \
  --config examples/seeded.json --model examples/frozen-relative-mixture.json \
  --resume runs/seeded/checkpoint.json --out runs/seeded-next \
  --sweeps 4000 --sample-every 10
```

`--sweeps` is the absolute final sweep. Named RNG streams are derived from the
master seed and absolute sweep, so a resumed run reproduces the uninterrupted
trajectory with the same build and inputs. The test suite checks every move
and checkpoint for exact continuation. Changed models/configs are rejected.

## Browser viewer

```bash
python3 tools/export_viewer.py --run runs/seeded --out runs/seeded/viewer.html
```

Open the resulting standalone HTML locally. It contains actual atom-sphere
geometry and retained body poses, with playback, rotation, zoom, and periodic
focus controls, a spherical frame selector (sphere center by default), and
optional native coordination bonds. It needs no server or network. This lightweight viewer is not
`hoomd-bevy`; the latter remains an optional future adapter. The physical kernel
and trajectory format do not depend on a rendering engine.

## Modules and invariance

| Module | Responsibility |
|---|---|
| `math` | FP64 proper rotations, Cayley coordinates, periodic representatives |
| `geometry` | Reusable body-frame sphere BVH, exact atomic hard tests, union coverage |
| `proposal` | Immutable Gaussian atlas, full Haar density, uniform support, periodic null trials |
| `depletion` | Conservative disjoint endpoint envelope and exact Poisson thinning |
| `spherical` | Exact atomic wall, implicit many-body GCA, common-translation chord |
| `auxiliary` | Normalized state-dependent mixture-mean law and latent transport |
| `rj` | Reversible ordered component births/deaths under a truncated Poisson prior |
| `simulation` | Scheduling, corrected acceptance, independent RNG streams, checkpoints |
| `trajectory` | GSD atom display and FP64 rigid-body pose chunks |

For the exclusion union U(X), the physical target is
`π(X) ∝ hard(X) wall(X) exp[-z |U(X)|]`, with translation volume and normalized Haar
orientation measure. It includes all simultaneous exclusions, not a sum of
pair overlaps. A spectator anchor is uniformly selected and retained as a move
label; every proposal evaluates the **whole** mixture at both endpoints.

For periodic boundaries, the learned raw displacement must lie in the unique lab-frame minimum-image
cell. Draws outside it are null moves, without retries or Gaussian wrapping.
The uniform pose branch supplies reverse support. For fixed spectators and
independently chosen endpoints, the fresh body-coordinate cloud gives
`G ~ Poisson(λ |new-only coverage|)` and
`L ~ Poisson((λ+z) |old-only coverage|)` within the moving exclusion union.
Acceptance is `min(1, exp(log q_old - log q_new) (1+z/λ)^(G-L))`, after the hard
test. This follows from auxiliary detailed balance, not from substituting an
arbitrary unbiased energy estimate into Metropolis.

At finite pruning depth, unresolved cells are retained and exact membership
thinning follows. Budget controls efficiency rather than the target. Ordinary
guarded FP64 is used, not formal interval arithmetic. Orthorhombic boxes must
satisfy every `L > 4 (body bounding radius + depletant radius)`; unsupported
small periodic cells and fixed particles are explicitly rejected. Spherical
coordinates have no minimum-image convention; their ideal bath is not clipped
at the protein wall.

The default model is frozen. The optional spherical `auxiliary_transport`
mode changes mixture means under a normalized conditional Gaussian law,
using the transported reverse model. Optional `reversible_jump` adds component
births/deaths from a finite atlas, with all labels and latent residuals included
in checkpoints. This preserves the physical marginal; it does not prove
equilibration or accumulate training data. See the [implemented RJ law and
native-prior controls](docs/rj-assembly.md), and the broader [continuing-learning
constructions](docs/rjmcmc.md).

Mixture fitting remains in the existing Python research workflow. This Rust
port loads a fixed base Gaussian export; it does not accumulate training data or
include the experimental Student-t proposal variant.

## Validation and scope

Tests include independent SciPy mixture densities/Hastings ratios, Gaussian and
Haar sampling moments, periodic null mass, BVH-vs-brute-force hard geometry,
conservative envelope coverage, analytic-sphere Poisson means, a genuine
many-neighbor union, forward/reverse accepted flux, zero activity, GSD I/O,
and exact continuation. The Python fixtures are archived with their generator.
The Rust RNG differs from NumPy; matching target/proposal laws does not imply
matching individual trajectories across implementations.

`tools/benchmark_port.py --reference /path/to/protein-nucleation --out results/bench`
compares the gate on identical recorded endpoints. This optional developer
benchmark needs the original NumPy/SciPy reference environment. Its speedup is
not automatically the end-to-end sampling or equilibration speedup.

The fitted mixture contains native examples and still has limited fluid
generalization. Short accepted assembly and persistent contacts do not determine
equilibrium basin populations, reversibility in practice, or nucleation rates.

See [third-party provenance](vendor/README.md), [input provenance](examples/README.md),
the [validation and benchmark report](docs/validation.md), and the
[RJ derivation](docs/rjmcmc.md). This repository is local; nothing has
been published remotely.
