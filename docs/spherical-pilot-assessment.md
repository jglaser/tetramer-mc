# Spherical physical pilot: independent audit

All 12 runs pass saved-frame interbody hard-core and explicit atomic spherical-wall checks, move replay, unchanged native-motif classification, and archived shape/model/source checks.

The physical target uses a protein-only spherical wall and a homogeneous ideal bath that permeates it. Radius 354.5082 Å, depletant radius 1.5 Å, activity 0.035 Å⁻³; 12 mobile tetramers, 40 sweeps per run. These starts are reembedded preassembled fragments from earlier periodic runs, not fresh dispersed nucleation experiments.

| Engine | Arm | Largest registered component | Local/global accepted | GCA partial/none/all | GCA motif formation/breakage | CPU s |
|---|---|---:|---:|---:|---:|---:|
| python | seed-free-frozen | 4 → 4 | 0/0 | 0/0/0 | 0/0 | 11.16 |
| python | seed-free-frozen-gca-shift | 4 → 4 | 0/0 | 39/1/0 | 0/0 | 34.19 |
| python | seed-free-transport-gca-shift | 4 → 4 | 0/1 | 37/2/1 | 0/0 | 88.71 |
| python | seeded-frozen | 9 → 9 | 24/0 | 0/0/0 | 0/0 | 21.58 |
| python | seeded-frozen-gca-shift | 9 → 10 | 13/2 | 23/11/6 | 0/0 | 55.49 |
| python | seeded-transport-gca-shift | 9 → 9 | 24/0 | 26/9/5 | 0/0 | 104.75 |
| rust | seed-free-frozen | 4 → 4 | 0/0 | 0/0/0 | 0/0 | 2.61 |
| rust | seed-free-frozen-gca-shift | 4 → 4 | 0/0 | 39/0/1 | 0/0 | 4.00 |
| rust | seed-free-transport-gca-shift | 4 → 4 | 0/0 | 39/0/1 | 0/0 | 3.87 |
| rust | seeded-frozen | 9 → 9 | 21/1 | 0/0/0 | 0/0 | 2.64 |
| rust | seeded-frozen-gca-shift | 9 → 9 | 21/1 | 31/5/4 | 0/0 | 3.67 |
| rust | seeded-transport-gca-shift | 9 → 9 | 21/1 | 31/5/4 | 0/0 | 3.77 |

## Interpretation

Across the eight collective-move runs, 265/320 GCA updates flip a proper subset of bodies (33 flip none; 22 flip all). There are zero GCA registered-motif formations or breakages. The sole registry-changing update is a global learned proposal in the Python seeded frozen+GCA+shift arm at sweep 21: body 8 adds motif (4,8,7), growing the ordered component from 9 to 10. One event cannot establish an advantage over controls or latent transport.

GCA components usually reproduce the existing physical aggregates: the seed-free state most often splits as [4,2,2,2,2], and the seeded state as [9,2,1]. Additional links sometimes join separate aggregates transiently into one move component. This is effective aggregate relocation while registry exchange remains unresolved.

Partial GCA flips move subsets of bodies and can reposition intact aggregates relative to others; a completed rejection-free move may also flip zero bodies or every body. Common center shifts preserve every interbody relative pose exactly. Neither common motion nor rotation alone establishes exchange between competing registered environments.

The registered descriptor requires a certified native tetramer relative pose plus its prescribed external monomer residue-patch contact. It excludes intrinsic tetramer contacts. A connected registered component is an ordered-association measure; this audit does not certify a perfect global lattice or thermodynamic crystal stability.

Only one physical preparation per seeded/seed-free arm is used here. The Python and Rust generators differ, and the Rust arms share named random streams. Counts and raw CPU times are diagnostics, not mixing-speedup, equilibrium, or independent-replication estimates. Base atlas was not retrained; transported arms used the specified instantaneous conditional fit.

### Registry changes by accepted move

- python/seed-free-frozen: 0 changing updates; retained 8/8 original registered motifs.
- python/seed-free-frozen-gca-shift: 0 changing updates; retained 8/8 original registered motifs.
- python/seed-free-transport-gca-shift: 0 changing updates; retained 8/8 original registered motifs.
- python/seeded-frozen: 0 changing updates; retained 16/16 original registered motifs.
- python/seeded-frozen-gca-shift: 1 changing updates; retained 16/16 original registered motifs.
  - Sweep 21, global, body 8: formed [(4, 8, 7)]; broken [].
- python/seeded-transport-gca-shift: 0 changing updates; retained 16/16 original registered motifs.
- rust/seed-free-frozen: 0 changing updates; retained 8/8 original registered motifs.
- rust/seed-free-frozen-gca-shift: 0 changing updates; retained 8/8 original registered motifs.
- rust/seed-free-transport-gca-shift: 0 changing updates; retained 8/8 original registered motifs.
- rust/seeded-frozen: 0 changing updates; retained 16/16 original registered motifs.
- rust/seeded-frozen-gca-shift: 0 changing updates; retained 16/16 original registered motifs.
- rust/seeded-transport-gca-shift: 0 changing updates; retained 16/16 original registered motifs.

### Provenance

Each per-run analysis records hashes of archived inputs, model, source, logs, and the historical Rust executable. Python config files retain inherited sweep/sample defaults; manifest CLI arguments and actual records specify the 40-sweep, every-5-sweep pilot. Current source/build may include later parser or numerical-input validation guards; it is not silently substituted for archived campaign code.

Replay checks every local/global accepted endpoint, half-turn axis/selected subset, common shift, saved frame, and final checkpoint. Rotation matrices are compared instead of quaternion sign conventions. Full native classification at every saved frame checks the partial pair updates used during replay. The atomic audit uses the unchanged reference in an analysis-only box of side 8(R+body_bound), whose minimum-image mapping is the identity for all possible physical pairs, plus an explicit wall check.
