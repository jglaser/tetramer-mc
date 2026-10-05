# Atlas pair-support lookup

`FixedBasinInvolution::apply` now checks support in constant time when its constructor has certified a complete pair set. The previous implementation searched the stored unordered-pair table on every call; an atlas with 2,048 virtual charts stores 2,098,176 pairs.

The constructor already rejects noncanonical, out-of-range and duplicate pairs. A validated subset with exactly `K*(K+1)/2` entries therefore contains every legal unordered pair. Checked integer arithmetic computes that cardinality, dividing before multiplying; overflow cannot grant the certificate. `apply` checks both label bounds before using the certificate. Sparse pair sets retain the previous scan, including reversed-label handling.

Pair weights, cumulative probabilities, random draws, chart transformations, densities and acceptance corrections are unchanged. Seven focused tests check the count arithmetic, exhaustive dense support, sparse holes, invalid constructors, invalid labels, bit-identical map outputs against a clone forced to use the old scan, and identical RNG consumption from matching seeds. The joint isolated release validation passed **28 tests**: these seven, four existing basin tests, seven prior-adapter tests and ten selected-chart tests. The ignored benchmark was excluded from that validation. An earlier compilation failure involving unsupported `StdRng::clone` calls is retained; the corrected tests initialize two RNGs from the same seed.

## Fixed support-only benchmark

The separately frozen benchmark made 256 predetermined ordered-pair checks per method at each of three sizes: **1,536 checks total**, with all answers and 18 journal events retained. It ran once, with no geometry, pose updates or random draws. Independent saved-data review authenticated all 638 bound files, completion/draining, the complete query inventory and every answer.

| Virtual charts | Stored pairs | Constructor wall time | Old scan, 256 checks | Certified check, 256 checks | Old scan per check |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,080 | 1.633 ms | 0.6963 ms | 5.974 µs | 2.720 µs |
| 256 | 32,896 | 20.935 ms | 5.8749 ms | 3.373 µs | 22.949 µs |
| 2,048 | 2,098,176 | 389.791 ms | 225.0505 ms | 0.948 µs | 879.104 µs |

At the current 2,048-chart scale, the old support lookup cost approximately **0.879 ms per check** for this fixed inventory. The certified path's entire 256-check measurement was below one microsecond, and its measured CPU time was zero at all sizes. Timer resolution, bookkeeping and allocation overhead prevent a reliable speedup factor; the decreasing tiny timings are not evidence of improved scaling with larger atlases. No timing repetitions were added.

This removes an avoidable scan from `apply`. It does not measure complete move cost, physical Monte Carlo speed, mixing or assembly. Construction still materializes an **O(K²)** pair table, and complete mixture-density scoring remains **O(K)**. The constructor times above are separate; the optimization does not remove those costs or establish the economics of rebuilding maps as an environment changes.

## Reproducible artifacts

- Corrected implementation: `src/basin_involution.rs`, SHA256 `8c118d2ec8f8164be249184d0749c4e3b6cc2f709964c9ea6f93f690a99254ef`.
- Joint release validation: `results/context-transport-prior-support-validation-20261004/attempt02/receipt.json`, SHA256 `91cf5bfed2b7123f6c7a6e2bce6194e3b975ca54880bd671ebde5eff9fef7708`. Production binaries and prior held targets were unchanged.
- Benchmark allocation: `results/basin-pair-support-preparation-20261004-v2/allocation.json`, SHA256 `05f2a3a9fbb699c9c45e13bccce91602e41a24dddc07f5570ed0887c148b6f73`.
- Benchmark plan: `results/basin-pair-support-benchmark-preparation-20261004/execution-plan.json`, SHA256 `066d6fbf33221f6377ad9c537334570c86e62be6fddf5dbd084dde3599c97ad5`. Separate limits were 30 CPU seconds, 60 wall seconds, 2 GiB and one thread.
- Completed report: `results/basin-pair-support-benchmark-preparation-20261004/benchmark/report.json`, SHA256 `cecf741b7784647d3e038a97cb1b58498a3e7e3cf2c294608f2b5b2274c36af1`.
- Drained receipt: `results/basin-pair-support-benchmark-preparation-20261004/host-drained.json`, SHA256 `7737cb0f0fc0dbecb6699b313f606ef25722fdd3f638d52759c53ee85b33da26`.
- Independent saved-data audit: `results/basin-pair-support-benchmark-preparation-20261004/independent-review.py` and `independent-review.json`. No benchmark or geometry was rerun by this audit.
