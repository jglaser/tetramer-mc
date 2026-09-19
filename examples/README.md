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
