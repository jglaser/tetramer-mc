# Recovered and validated finite-assembly baseline

The saved N=12/N=24 input bank now has an **isolated release executable built
from its exact archived application source**. All forty integration references
and the one posterior-configuration check pass in release mode. This resolves
the missing executable prerequisite; it does not launch assembly or satisfy the
regional/full-vessel physical convergence gates.

The source is commit `9f2ae102ed100df68ad582d52043accd16490646`. All 49 source,
lock, build, test and documentation files archived by the bank match it
byte-for-byte. The remaining archived file is the separately retained historical
validation receipt, which was never a Git source file. The complete Git archive
supplies the other original tests and all 399 unchanged vendored hoomd-rs files;
it contains 1,106 files. Dependency versions remain pinned by the original
Cargo.lock. The build used Cargo/Rust 1.95.0 with `--locked --offline`, two Cargo
jobs, four-core affinity and single-threaded test execution.

## Completed checks

| Existing test target | Passed |
| --- | ---: |
| Frozen posterior assembly wiring | 4 |
| Periodic posterior assembly wiring | 3 |
| Periodic reciprocal proposal | 6 |
| Posterior docking | 2 |
| Independent proposal references | 6 |
| Reciprocal proposal | 9 |
| Reciprocal transport stationarity/inversion | 3 |
| Exact runner checkpoint continuation | 1 |
| Periodic transport | 6 |
| Posterior configuration constraints | 1 |
| **Total** | **41** |

These are the previously established Gaussian/sphere references and toy runner
controls, now exercised against the rebuilt release implementation. They cover
Haar/Jacobian corrections, reciprocal labels and mixture densities, periodic
unique-image nulls, stationarity, runner/bath wiring and deterministic restart.
They generate no protein trajectories and do not measure protein mixing.

The integration build and tests took 269.50 wall seconds; the configuration
check took 21.80 seconds, and the final runner build took 0.28 seconds. All
children drained, all expected test counts matched, source bytes were rechecked,
and the validation controller exited. Embedded source-bundle entries match the
archived build source. The production executable remains unchanged at SHA256
`94c987694af7cc347fdf076ff121425464329b35ae76c80b9e0415592410010a`.

## Artifacts and remaining work

The isolated directory is
`results/finite-assembly-baseline-build-20261004/`. Its executable is
`target/release/tetramer-mc` **inside that directory**, not the repository's
production target. Its SHA256 is
`e18efb4b5d44772fee4d788a6fa46e0da886b60b74d116b5eac28f1d314937a4`.

- Source freeze: `3cc4b37d84e706f91881a966339b868ab61058ed4220e79346dea2aa83932527`.
- Completed validation: `7af586ade7c55c0464c95e169f86ee5f16b3504c49e9d6df5e5b2d3f9e6816a3`.
- Embedded bundle: `d7f6abec7f8201a261aa20803f4ddf8e041e7bd1eb0511563684ed9be8834a0b`.

A metadata-preparation failure is retained: system Python lacked
`hashlib.file_digest`. It occurred after archive extraction but before any
build or test. The project interpreter then verified every extracted file
against the preserved archive before freezing it. No scientific allocation was
rerun or changed.

The original 432-configuration bank is untouched. It still needs an explicit
production observation window, launch-time seed verification, executable/source
binding, bounded dispatch and analysis contracts. The separate expanded-native
observer must describe the same 48 starting states without redrawing those that
acquire supplemental native labels. Regional sensitivity and remaining-vessel
mass remain unresolved. Finite-system stability needs the planned independent
starts, size and boundary comparisons; successful reference tests do not
substitute for that evidence.
