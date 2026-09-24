# Window-free finite-assembly configuration bank

`tools/prepare_finite_assembly_config_bank.py` prepares the complete **432-job input
bank** without selecting a production window or running Monte Carlo. The bank
has twelve blocks: N=12 and N=24, the native-blind and native-informed frozen
models, the primary spherical study and the spherical/periodic boundary control.
Each block has local, independent-redraw and correlated-transport arms, three
starting preparations and four independent runtime seeds per arm/preparation.

All 48 starting-state assets and their completed geometry checks are retained.
Matched arms/families/studies reuse the same prescribed starting coordinates;
their runtime seeds differ. These are controlled initial configurations, not
independent equilibrium samples. The native seed and intratetramer structures
remain supplied.

The bank's schema is `finite-assembly-config-bank-v1`, deliberately distinct from
the [full campaign contract](finite-assembly-contract.md). Its `window`,
`executable` and `dispatcher` are null; both `production_ready` and
`production_authorized` are false. The existing regional/full-vessel convergence
gates remain unresolved and unchanged. Preparing inputs neither estimates
equilibrium weights nor establishes assembly or finite-system instability.

## Frozen inputs and checks

The preparer authenticates and copies the completed
`runs/finite-assembly-starts-20260924` archive, the periodic-compatible observer
archive `runs/finite-assembly-observer-periodic-20260924`, the complete 64-slot
native-blind memory export, and the pinned native-informed coverage model.
It reuses saved hard/wall/native graph checks without atom-geometry replay,
contact classification, proposal fitting or physical draws. Observer target
identities must equal the starting-state identities for each N/boundary.

The native-blind model has 64 base/128 virtual reciprocal components. The
coverage model has 178 base components with 150 reciprocal flags, giving 328
virtual components. They are different proposal controls under the same physical
target; equal component counts are not required. Local controls are repeated
with independent streams in every block.

Every job records separate byte hashes and canonical JSON digests. In particular,
the observer binding carries the definition file's byte hash, while
`observer_definition_sha256` is the canonical digest of the parsed definition and
`region_definition_sha256` is the canonical digest of its region rules. These
serve different purposes and must not be substituted for each other.

All output configurations bind the unchanged 4,004-atom repaired tetramer,
depletant radius 1.5 Å, activity 0.035 Å⁻³ and all-mobile status. The existing
starts supply exactly 106.8 μM and matched spherical/periodic geometric volumes.
No coordinates, radii, orientation measures or native labels are refitted.

## Explicit proposal schedule

The default reuses the documented historical N=12 schedule; it does not infer a
production length from that trajectory:

| Setting | Value |
|---|---:|
| Local translation standard deviation | 0.2 Å |
| Local angular standard deviation | 1° |
| Global branch probability, learned arms | 0.5 |
| Uniform defensive probability | 0.1 |
| Posterior branch probability within global updates, transport arm | 0.5 |
| Transport correlation | 0.9 |
| Auxiliary Poisson intensity/activity | 64 |
| Endpoint maximum cells/depth/minimum width | 2047 / 14 / 0.5 Å |
| Primary spherical GCA and center-shift probability per sweep | 1 each |
| Boundary-comparison GCA and shift probability, both boundaries | 0 each |

The local arm sets global probability to zero and supplies no model. Independent
redraw has no `frozen_posterior`; it is not a zero-correlation posterior update.
The transport arm keeps the independent-capture branch. Adaptive modes,
compression, nonlinear kernel shear and assembly bias are absent. Optional
frozen bias needs its separate later contract and reweighted observer.

An explicit `--schedule` JSON may override this entire schedule, with exactly
`common`, `global_probability`, `learned_uniform_weight` and `frozen_posterior`
keys. `common` has the six exhaustive fields from the matched benchmark contract.
The preparer rejects missing/unknown settings, zero defensive support, zero
transport correlation, missing independent capture, or a primary schedule
without both collective moves. Boundary controls always disable both collective
moves. A one-sweep synthetic window is used only inside the existing pure
schedule validator; it is never emitted as a scientific timing choice.

## Seeds, source provenance and unresolved execution

Runtime seeds are deterministic uint64 values: the first eight SHA-256 bytes,
big-endian, of `finite-assembly-config-bank-v1:runtime:MASTER:JOB_ID`. All 432
values must be distinct and absent from the historical declaration inventory;
collisions stop preparation before creating the bank. The inventory scans run
protocols, plans, manifests, declarations, seed inventories, named configuration
files and configuration directories, plus all example JSON. It collects scalar
`seed`/`*_seed` and list `seeds`/`*_seeds` fields recursively. Declaration bytes
are archived by content hash. Raw trajectories and physical samples are not read.
The inventory scope is explicit; a dispatcher must repeat reservation checks
at launch to catch newly created concurrent declarations.

The bank archives current application sources, dependency lock and the completed
periodic correctness receipt. Their hashes must match that receipt. It does
**not** bind the old `target/release/tetramer-mc`, which predates periodic learned
support. A later release build must bind its executable and embedded source
bundle before any performance campaign. The bank does not claim that its source
snapshot is itself a tested release executable.

Each job has a structured `argv_template`. Literal arguments and bound
configuration/model paths are fixed; typed unresolved tokens identify the
executable, output root, end sweep and sample cadence. The templates are data,
not executable shell strings. They omit `--resume`, `--no-moves` and `--no-gsd`.
Actual method, sweep count and cadence are CLI arguments; putting those values
only in input metadata would not configure the runner. Burn-in belongs to the
future analysis plan, not the runtime command.

The later binder must choose and freeze a common observation window, resolve
the typed arguments, emit twelve complete benchmark contracts and 432 analysis
plans, authenticate the release executable, and enforce the original scientific
and resource gates. At most eight physical jobs and 32 total workers remain the
resource ceilings. This preparation provides no dispatcher or gate bypass.

## API and validation

```python
prepare(repository, starts, observers, out,
        master_seed=152101010, schedule_path=None)
validate_bank(out, check_external=True)
```

Preparation requires a fresh output directory. Partial failures retain a failure
record and freeze; no overwrite, redraw or retry occurs. Validation authenticates
the archives, all 432 configurations, exact pose/model/observer bindings, the
historical seed union and command templates. By default it also checks original
external source paths. `check_external=False` verifies the archived bank only;
it does not establish launch-time seed freshness or authorize production.

Focused tests use synthetic targets and mock trust anchors plus completed
geometry-audit results. They exercise the full 432-job allocation, matched
starting states, independent seeds, explicit schedule and typed unresolved
arguments, configuration/source corruption, archive escape, seed collision,
failure preservation and no-overwrite behavior. They generate no physical data.

The preparer creates inputs only. To reproduce them in a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  tools/prepare_finite_assembly_config_bank.py \
  --starts runs/finite-assembly-starts-20260924 \
  --observers runs/finite-assembly-observer-periodic-20260924 \
  --out /absolute/fresh-config-bank
```

## Completed input preparation

The [prepared bank](../runs/finite-assembly-config-bank-20260924/manifest.json)
contains all 432 configurations and distinct runtime seeds. Its
[freeze inventory](../runs/finite-assembly-config-bank-20260924/freeze.json)
binds 3,178 files. The historical inventory authenticated 4,513 declarations
(2,440 distinct byte contents) and reserved 1,123 previously declared seed
values. None of the new runtime seeds collides with that inventory. All 48
completed starting-state geometry records were reused without physical draws
or classifier replay.

The complete bank validator passed, including original external-source hashes.
Manifest SHA-256 is
`3b1a3492654e0eb95252fd087a6995f91098644fa442fabe2032356d5d8814ff`.
Nine focused implementation tests had passed before preparation. Executable,
observation window and dispatcher remain unresolved; no assembly runs were
started. This prepares the requested size, initial-condition, proposal-family
and boundary controls without claiming their equilibrium behavior.
