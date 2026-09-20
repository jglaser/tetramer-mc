# Supplied physical inputs

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
