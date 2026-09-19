# Independent Rust campaign validation

The eight-run, 400-sweep campaign in `runs/rust-validation-400` passed an
independent structural and output audit. It used twelve mobile rigid tetramers,
depletant radius 1.5 Å, activity 0.035 Å⁻³, and the unchanged frozen proposal
model `2e534634e6e2fe83da969a2867504c293af8e90a0cf3c4330cdc9e38a6770064`.

| Initial preparation | Method | Final largest registered components | Sampler CPU seconds |
|---|---|---|---|
| Seeded | Local + learned | 9, 9 | 38.498, 32.745 |
| Seeded | Local + uniform | 8, 8 | 90.803, 91.035 |
| Seed-free | Local + learned | 4, 3 | 32.406, 31.639 |
| Seed-free | Local + uniform | 1, 1 | 6.913, 6.599 |

The two repeats per preparation share its initial configuration and use fresh
paired random seeds. There are **two distinct initial configurations**, not eight
independent equilibrium samples. The timings include GSD writing. These runs
demonstrate accepted ordered association and working integration; they do not
measure equilibrium mixing, a physical nucleation rate, or a Rust/Python sampling
speedup. Rust and NumPy use different RNGs, and the earlier Python campaign used
different preparations for some repeats. A benchmark of identical endpoint
geometry is a separate kernel measurement.

## Checks performed

- The unchanged Python `AtomicAssembly` independently audited all sixteen initial
  and final states using periodic atomic sphere geometry. All passed. Its existing
  numerical audit tolerance is 2×10⁻⁸ Å; production uses strict hard geometry.
- The unchanged `TetramerOrder` classified all 328 stored frames. A registered
  edge requires a certified full-tetramer relative pose and a prescribed external
  native monomer contact, including its residue patch. Internal tetramer contacts
  do not count. Histories use individual motif entry/stay hysteresis at the
  ten-sweep recording cadence; events between frames can be missed.
- All 38,400 move records replay exactly to every stored frame and final
  checkpoint. Old/proposed/retained poses, permutation scheduling, and aggregate
  acceptance counters are consistent. This validates state bookkeeping, not an
  independent reevaluation of every acceptance probability.
- Shape, model, configuration and executable hashes agree with the campaign
  manifest and archived input copies. The campaign executable hash is
  `5ad6e86d75991fcc485782035c8a6c3c04d28404bd76ed44b1a7f46663772dfc`.
- A separate Python `struct`/NumPy reader decoded the emitted GSD2 subset without
  calling Rust GSD or installing a Python GSD package. Every FP64 body pose equals
  JSON exactly. Across 15,759,744 atom positions, standard FP32 coordinates,
  periodic images, diameters, types and display body IDs agree with reconstruction.
  Maximum FP32 position error was 1.52588×10⁻⁵ Å. This is not a general external
  viewer compatibility certification.

The audit hashes its Python reference modules and inspected Rust/GSD format
sources separately. Inspection-time source hashes do not alone identify the
source revision used to build an executable. The executable hash above identifies
the binary that produced these results.

## Reproduction

`tools/analyze_rust_campaign.py` is an optional developer analysis tool. It takes
the original Python reference project explicitly; that project and NumPy/SciPy
are not simulation runtime dependencies.

```bash
/path/to/protein-nucleation/.venv/bin/python tools/analyze_rust_campaign.py \
  --reference /path/to/protein-nucleation \
  --campaign runs/rust-validation-400 \
  --out results/rust-validation --workers 4
```

The output directory contains the aggregate analysis, per-run contact histories,
hashes, an audit report, and an exact copy of the analysis script.

The README's distinction between frozen-target invariance and observed mixing,
its FP64/FP32 output description, and its warnings about repeated initial states
agree with this campaign audit. Unit-test claims—including exact restart and
forward/reverse flux—are covered by the repository tests rather than inferred
from these assembly trajectories.
