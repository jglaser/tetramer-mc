# Supplied physical inputs

## Latest frozen reciprocal assembly examples

`spherical-reciprocal-free.json` and `spherical-reciprocal-seeded.json` use
`frozen-reciprocal-mixture.json`, the exact validated reciprocal envelope of
the current 150-component atlas. Each base component has two reciprocal
branches, giving 300 virtual branches without fitting another Gaussian.
The model SHA256 is
`dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3`.
All runtime inputs are included here with relative paths; no campaign data
or original protein-nucleation directory is needed to run these examples.

Both configurations preserve the corresponding existing spherical example's
12 initial tetramer poses, hard shape, and 354.50820786337056 Å sphere. Every
tetramer is mobile, including those labelled as the initial seed. They use
depletant radius 1.5 Å and activity 0.035 Å⁻³, local steps of 0.2 Å and
1 degree, global-move probability 0.5, frozen-posterior probability 0.5
within the global branch with correlation 0.9, and a 10% uniform proposal
floor. The auxiliary Poisson intensity ratio is 64. GCA and sphere-center
shift attempts run after each ordinary sweep. Each example has its own RNG
seed; change `seed` to obtain an independent replicate.

For the matched independent-redraw control, set only
`frozen_posterior.correlation` to `0.0`. Keep its probability, the global and
local proposals, GCA, and center shifts unchanged. The supplied latest
configuration uses `0.9`.

This atlas remains **native-informed**. Reciprocal inversion exchanges the
roles of the two proper relative poses; it does not invert a protein's
chirality. The example is a frozen reversible sampling control, not a claim
of template-free discovery or a guarantee that a crystal will assemble.
There is no adaptive training, auxiliary mixture learning, or reversible-jump
update in these two configurations. The separate importance sampler for
fixed contact-region integrals is not an assembly move and is not enabled
here.

From the repository root, with Rust and the locked dependencies available
locally:

```bash
cargo run --offline --locked --release --bin tetramer-mc -- run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-reciprocal-mixture.json \
  --out runs/reciprocal-free-10000 --sweeps 10000 --sample-every 10

cargo run --offline --locked --release --bin tetramer-mc -- run \
  --config examples/spherical-reciprocal-seeded.json \
  --model examples/frozen-reciprocal-mixture.json \
  --out runs/reciprocal-seeded-10000 --sweeps 10000 --sample-every 10
```

The commands retain `trajectory.gsd`, `trajectory.jsonl`, `moves.jsonl`,
checkpoints, and input/source provenance. Use a fresh output directory for
each invocation. To extend the free run to a **total** of 20,000 sweeps:

```bash
cargo run --offline --locked --release --bin tetramer-mc -- run \
  --config examples/spherical-reciprocal-free.json \
  --model examples/frozen-reciprocal-mixture.json \
  --resume runs/reciprocal-free-10000/checkpoint.json \
  --out runs/reciprocal-free-20000 --sweeps 20000 --sample-every 10
```

Resume keeps the original stream and physical configuration; a larger
`--sweeps` value is the new cumulative endpoint, not an additional count.

Export a self-contained offline browser viewer with Python's standard
library:

```bash
python3 tools/export_viewer.py --run runs/reciprocal-free-10000 \
  --out runs/reciprocal-free-10000/viewer.html --native-bonds off
```

Open the resulting `viewer.html` in a browser. The default is the
**sphere-centered frame**. The coordinate-origin selector shows accumulated
center shifts when their saved history is available. This portable command
disables the optional native-bond classifier, whose research reference
inputs are separate from the bundled assembly inputs.

The earlier `frozen-relative-mixture.json` and earlier example configurations
are retained for reproducing their historical controls.

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
