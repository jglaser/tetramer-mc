# Frozen finite-assembly observer

The observer distinguishes exclusion-driven aggregation from instantaneous native registry in the all-mobile N=12 and N=24 systems. It measures the spherical and periodic systems independently. Its definitions do not prescribe a production window, establish equilibrium, or authorize a physical campaign.

`tools/prepare_finite_assembly_observer.py` builds immutable assets from the completed `runs/finite-assembly-starts-20260924` archive. It authenticates all 158 frozen files and the complete 48-state allocation before using saved results. The preparation performs no physical draws, pair-overlap calculations, or native classifications. It repeats graph calculations using the already saved exact contact and motif keys, preserving all 48 records.

## Patches fixed by the rigid shape

The repaired tetramer contains four consecutive blocks of 1,001 atom spheres. For each block, preparation independently reconstructs the atom centers from the archived monomer and that member's rigid transform. The maximum coordinate difference must be at most `1e-10` Å. Radii must agree exactly; preparation cannot repair or resize the shape.

Each atom gets an ID `memberM:octantO`, where `M` is its member index and

```text
O = 4*(x >= 0) + 2*(y >= 0) + (z >= 0)
```

Here `(x,y,z)` is the atom center in the archived monomer body frame, with its original origin. No centroid, surface fitting, reference contact, or native label enters this construction. Zero belongs to the nonnegative side. All 32 IDs appear in the dictionary, including any empty patches; every atom retains one ID. Member rotations and translations cannot change an atom's patch assignment.

These deliberately coarse patches provide a fixed contact fingerprint. Their apparent ESS can miss rearrangement within an octant, so native motif fingerprints and the component statistics are also retained.

## Instantaneous regions

Let `cX` be the largest exclusion-connected component and `cN` the largest component of the complete instantaneous native-contact graph. Singletons have size one. A native component is certified only when its full set of alternative catalogue labels admits a consistent assignment and its periodic edge images, when present, admit an ordinary-space lift. Certification never selects a favorable spanning tree while ignoring conflicting edges.

| Region | Complete-frame criterion |
| --- | --- |
| `dispersed` | All native components resolved, no native edges, and `cX = 1` |
| `contact_no_entry` | All native components resolved, no native edges, and `cX >= 2` |
| `registered_small` | All native components resolved and `2 <= cN <= 7` |
| `registered_eight` | All native components resolved and `cN = 8` |
| `registered_growth` | All native components resolved and `cN >= 9` |
| `remaining` | At least one native component has a frustrated catalogue assignment or nonzero periodic winding |

These regions are a disjoint, exhaustive partition of a physically valid observed frame. Eight is the supplied seed-fragment size, not an inferred critical nucleus. A seeded component of eight plus isolated tetramers is therefore `registered_eight`. A cluster can grow in exclusion connectivity without growing in native registry.

Trees have no independent cycles; their consistency check cannot supply cycle evidence. `registered_growth` means instantaneous registered size at least nine. The label alone establishes neither a growth event nor crystal stability.

Image lifting precedes catalogue testing. Winding components have `catalogue_consistent = null`; winding alone is not a catalogue contradiction or evidence of a nonnative phase. A frame with winding or frustration remains in every occupancy denominator as `remaining`. The observer also retains the full raw component sizes and any separately certified components in that frame. The largest certified whole component is a lower bound on supported registry, not a search for the largest compatible subgraph or proof of global crystallinity.

The output preserves exclusion and native component lists, component number/size statistics, edge counts, raw and certified largest sizes, per-component image/cycle status, and all instantaneous native labels. Region and component-size measurements are invariant to relabeling bodies. Tagged patch and native-motif fingerprints answer a different relaxation question and are reported separately; their apparent ESS is not a permutation-invariant assembly occupancy.

## Assets and provenance

Preparation writes:

- `patch-map.json` and the per-member reconstruction report.
- `regions.json` with the exact six selectors and measurement scope.
- Four `definitions/N{12,24}-{spherical,periodic}.json` files using schema `finite-assembly-observer-definition-v1`.
- The full native-definition archive, repaired shape, and transitive Python source closure, including the new trajectory adapter and preparation tests.
- `starting-state-graphs.jsonl`, retaining all 48 cached graph projections and their originating state/validation hashes.
- A preparation plan, manifest, and a hash inventory of every output. Failed preparations retain their partial outputs, a failure record, and a final hash inventory.

Each definition binds the existing physical-identity dictionary and its canonical SHA-256, the patch map, the complete native definition, all observer source files, the six region rules, and reference size eight. The identity includes particle count, boundary and vessel size, shape, radius, activity, fixed-body indices, and the physical measure. Source paths are relative to each definition. Runtime Python, NumPy, and SciPy versions are recorded. Changing the measurement law or source closure requires new assets rather than overwriting old definitions.

The saved starting-state projections should give 16 `dispersed`, 16 `contact_no_entry`, and 16 `registered_eight` records. Those are deliberately constructed preparations; their counts are not physical probabilities. No sampling efficiency, free-energy, or assembly-stability conclusion follows from them.

After all observer sources and tests are final, prepare a fresh directory with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python tools/prepare_finite_assembly_observer.py \
  --starts runs/finite-assembly-starts-20260924 \
  --out runs/finite-assembly-observer-preparation-20260924
```

The separate `analyze_finite_assembly.py` adapter authenticates this definition against trajectory inputs and runtime source files and uses an explicitly supplied analysis window. It currently supports unbiased trajectories. Existing biased-occupancy tooling remains separate; these assets do not establish support for applying it to the new six-region observer. Production windows, executable invocation, physical gates, and campaign allocation remain separate obligations. Historical frozen observers and analyses are unchanged.

For a future compatible completed run, the invocation is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python tools/analyze_finite_assembly.py \
  --run /path/to/completed-run \
  --definition runs/finite-assembly-observer-preparation-20260924/definitions/N12-spherical.json \
  --plan /path/to/frozen-analysis-window.json \
  --out /path/to/fresh-observation-output
```

The window plan must bind the exact definition SHA-256 and provide the declared preparation, proposal arm, burn sweep, and end sweep. This example deliberately supplies no invented production-window values. An optional bound benchmark contract connects the report to the existing matched-kernel validation.

## Completed preparation

The [asset manifest](../runs/finite-assembly-observer-preparation-20260924/manifest.json) contains four definitions and all 48 cached graph projections: 16 `dispersed`, 16 `contact_no_entry`, and 16 `registered_eight`. All four 1,001-atom blocks reconstructed with zero numerical difference in centers and radii, and all 32 patches are occupied. The [freeze inventory](../runs/finite-assembly-observer-preparation-20260924/freeze.json) binds 47 files, including 16 source files. Manifest SHA-256 is `acc94abd8984f76c1421fd5689a3a0f1d4d2d07041ac4535c8d18d8e1e074aa9`.

Validation comprises 22 graph/series observer tests, 10 trajectory-adapter tests, and [10 preparation tests](../runs/finite-assembly-observer-test-receipt-20260924/validation.json). The asset preparation used Python 3.13.12, NumPy 2.5.3, and SciPy 1.18.1. It launched no physical jobs and replayed no native or interbody geometry classifications.
