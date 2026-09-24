# Inert finite-system assembly contract

`tools/finite_assembly_contract.py` validates the proposed assembly design and
its declared input bytes. It never creates configurations, starts a sampler,
fits a model, classifies a contact, or opens a production gate. Existing
single-target `matched-contact-kernel-benchmark-v1` checks remain unchanged.

The design has 12 separate blocks:

| Study | Sizes | Boundaries | Frozen model families | Blocks |
|---|---|---|---|---:|
| Primary spherical | 12, 24 | spherical | native-blind memory, native-informed coverage | 4 |
| Boundary comparison | 12, 24 | spherical, periodic | the same two families | 8 |

Each block contains local, independent-redraw and correlated-transport arms;
dispersed, competing-aggregate and native-seeded preparations; and four streams
per arm/preparation. Thus the complete design reserves **432 distinct seeds**.
Local controls are explicitly duplicated between model families and studies,
using different seeds. Seed uniqueness is checked within this manifest only;
a future dispatcher must additionally exclude seeds from the historical
campaign inventory. There is no shared-control or paired-stream inference.
These are declared allocations, not launched or completed simulations.

The primary study keeps one fixed positive GCA/center-shift schedule. The
boundary study sets both probabilities to zero in *both* boundary conditions,
since these implementations require a sphere. Local widths, global branch
probabilities, the uniform floor, endpoint work budget, cloud intensity,
transport correlation and observation window otherwise remain matched across
the design. Independent redraw has no `frozen_posterior`; zero-correlation
posterior transport is a different proposal and is rejected as that control.

Each size/boundary pair retains its own physical identity and observer/region
identity. Those identities must match across model families and studies, but
are never identified with a different size or boundary. The existing trajectory
comparator can subsequently compare the arms *within* each block. This new
contract does not pool occupancies or define finite-size extrapolation.

## Fixed physical constants and inputs

The schema requires the repaired rigid tetramer shape, radius 1.5 Å and activity
0.035 Å⁻³. Every body is mobile. Tetramer concentration must be within **0.05 μM
of 106.8 μM**; the tolerance itself is fixed by this schema and cannot be relaxed
in a manifest. With Avogadro's exact constant,

`concentration_uM = N / volume_A3 * 1e33 / 6.02214076e23`.

At a given N, the sphere's geometric volume and periodic cell volume must agree
within relative tolerance `1e-12`; N=24 must have twice the N=12 volume. Sphere
display-box dimensions do not determine its physical volume. Equal geometric
volume does not assert equal orientation-dependent accessible volume at a wall.
The atomic wall and wall-permeable ideal bath retain their existing meanings.

The required `geometry-only` family is the **native-blind, physically learned
64-slot pair-memory export**, not the 96-component shape-ray atlas. Its pinned
model is `b6d06b0a076d7f3cd4f591d116a77799b2dc4caa3ea6c21081ad5da8a45b3e9b`.
The pinned native-informed model is the coverage atlas
`feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e`.
Both retain supplied intratetramer geometry. A seeded initial preparation also
supplies intertetramer registry; the proposal family and preparation are distinct
controls.

The file validator authenticates the existing memory export's pinned manifest,
freeze and every archived file, including its historical checkpoint/source
closure. It checks the exact four-replicate, sixteen-slot order and 64/128
base/reciprocal component inventory. It reuses the reviewed export and its
already completed checks; it does not repeat atom geometry or infer native
blindness from an arbitrary `native_informed: false` flag. Changing either model,
compressing the bank or adopting a new source requires a separately reviewed
schema/preparation, not changing a claimed provenance label.

## API and manifest

Public Python interfaces are:

```python
validate_contract(plan)                 # pure design checks, no file reads
validate_configuration(config, block, job)  # pure effective-law/pose checks
validate_files(plan, base_dir)           # read-only input authentication
```

The strict top-level schema has exactly these fields:

```json
{
  "schema": "finite-assembly-campaign-v1",
  "physical": {
    "shape": {"path": "shape.json", "sha256": "REPAIRED_SHAPE_SHA256"},
    "concentration_uM": 106.8,
    "concentration_tolerance_uM": 0.05,
    "volume_relative_tolerance": 1e-12,
    "depletant_radius": 1.5,
    "reservoir_density": 0.035
  },
  "models": {
    "geometry-only": {
      "model": {"path": "memory/model.json", "sha256": "PINNED_MEMORY_MODEL_SHA256"},
      "provenance": {
        "manifest": {"path": "memory/manifest.json", "sha256": "PINNED_MANIFEST_SHA256"},
        "freeze": {"path": "memory/freeze.json", "sha256": "PINNED_FREEZE_SHA256"}
      }
    },
    "native-informed": {
      "model": {"path": "coverage-model.json", "sha256": "PINNED_COVERAGE_MODEL_SHA256"},
      "provenance": null
    }
  },
  "primary_common_schedule": "EXISTING_CONTRACT_COMMON_SCHEDULE_OBJECT",
  "window": {"burn_sweep": 1000, "end_sweep": 100000, "cadence_sweeps": 10},
  "blocks": ["ALL_TWELVE_BLOCKS"]
}
```

This is a schema illustration, not an executable scientific allocation. The
window is illustrative; its value must be chosen and frozen before production.
All actual hashes are exported as constants in the module. Paths resolve against
the manifest directory; a bound configuration's shape path resolves against
that configuration. Each block has exactly `id`, `study`, `bodies`, `boundary`,
`model_family`, `physical_identity`, `benchmark_contract`, and `jobs`.
`study` is `primary-spherical` or `boundary-comparison`; `boundary` is the string
`spherical` or `periodic`. The physical identity uses the existing
`analyze_contact_efficiency.physical_identity` representation, including the
full boundary object and the unchanged physical measure string.

Every block embeds its own complete existing benchmark contract. Preparation
labels are exactly `dispersed`, `competing-aggregate`, `native-seeded`, in that
order. Arm names are arbitrary nonempty IDs, with exactly the three existing
kernel roles. The full set of 36 jobs is required, not a reduced sample of a
future campaign. Job IDs and uint64 master seeds must be globally unique.

Every job has exactly `id`, `arm`, `preparation`, `stream`, `seed`, `invocation`,
`config`, and `initial_poses_sha256`. `invocation` records `method`, `sweeps`,
`sample_every`, and `model_sha256`, matched exactly to its nested arm and window.
These are CLI controls; putting these fields in the input configuration does
not set them. The declaration does not prove which command a dispatcher executes. Burn and end sweeps must align with
saved cadence. Every job binds its input configuration and canonical
initial-pose hash. Set
**both** fields to `null` for a genuinely missing preparation. A config binding
is `{ "path": "...", "sha256": "..." }`; the pose hash is the existing
canonical JSON `digest(config["initial_poses"])`, not the file's byte hash.
For the same N/boundary/preparation/stream, all arms and families use identical
initial poses while their simulation RNG seeds differ. A stream index is one
of 0, 1, 2, 3. Common starting coordinates are prepared controls, not independent
equilibrium starting samples.

## What a successful check means

`complete_design: true` means the declared allocation and parameter relationships
are complete. `input_bindings_complete` says whether every job has a config/pose
binding. `bound_files_verified` says all *declared* files passed the byte and configuration
checks. `input_files_verified` additionally requires all 432 configuration
bindings; it stays false while any preparation is unfilled.
The report always retains `geometry_validated: false` and
`production_authorized: false`. It lists missing scientific/operational
obligations instead of treating a passed design check as production readiness.

Bound configurations are checked for matching physical identity, frozen arm,
schedule, seed, all-mobile status, initial pose count and unit quaternions. The
version-one contract supports unbiased frozen proposals only. Adaptive/auxiliary
models and assembly bias are rejected; the already validated frozen-bias kernels
and reweighted observer need their own explicit later campaign contract.

Still required are initial hard/wall geometry, evidence that preparation labels
describe their poses, observer definition/source closure, compatible executable
and exact invocation/model bindings, the physical convergence gates, and any
launch authorization. This validator does not invent meaningful environments
from hashes or resolve periodic native graphs with nonzero winding. No finite
trajectory, design check, or conditional scaffold integral establishes assembly.

```bash
python tools/finite_assembly_contract.py /path/to/inert-manifest.json
python tools/finite_assembly_contract.py /path/to/inert-manifest.json --verify-files
```

Both commands only print a report. No `run`, resume, retry or sampler-dispatch
interface exists.

## Completed implementation checks

Twenty focused synthetic tests pass under both system Python 3.9.25 and the
project's Python 3.13.12. They cover the complete matrix, independent volume
calculations, mismatched physics and schedules, seed reuse, invocation arguments,
complete versus partial input bindings, ordered all-slot provenance, archive
containment and streaming hashes for empty and multi-block files.

The [real-asset integration exercise](../runs/finite-assembly-contract-validation-v2-20260924/validation.json)
authenticates 39 existing files, including both frozen models and the native-blind
export, and binds one existing N=12 configuration. Its other 431 configuration
slots remain unfilled. It correctly reports `bound_files_verified: true`,
`input_bindings_complete: false`, `input_files_verified: false`, and
`production_authorized: false`. Its observer hashes, seeds and observation window
are explicitly synthetic or illustrative. This is an input-schema exercise, not
a frozen scientific allocation or a claim that the preparations are ready.

The [first attempt](../runs/finite-assembly-contract-validation-20260924/failure.json)
is preserved: Python 3.9 lacked `hashlib.file_digest`. The portable implementation
now hashes bounded chunks with `hashlib.sha256`; the successful retry uses a fresh
directory. Neither attempt generated physical draws or repeated geometry audits.
