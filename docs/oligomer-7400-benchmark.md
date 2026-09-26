# Two-contact oligomer charts on the sweep-7400 state

HEAD `f2e33c34106a2d588dd750e8a576572905640ea0` was tested without modifying
production code. The proposal produced one substantial oligomer attachment,
but a reproducible catalogue-reconstruction failure prevents treating the
results as validation of an invariant sampler or a sampling speedup.

## Frozen comparison

The saved state contains 264 all-mobile tetramers at 500 micromolar in a
spherical vessel, with depletant radius 1.4 Angstrom and activity
0.0275 Angstrom^-3. Each arm used four population seeds and 128 reset phases
per population, duration 0.01 per phase, at most two concurrent physical jobs.
The only configuration difference was `cluster_phase.oligomer: {}`. Both arms
retained contact-aware anchors, the same frozen atlas, local/transport schedule,
defensive uniform fraction, correlation and full many-body Poisson gate.

| Conditional count or cost | Member charts | Two-contact charts |
|---|---:|---:|
| Learned attempts | 714 | 709 |
| Hard-valid learned trials | 74 | 142 |
| Accepted learned moves | 0 | 13 |
| Accepted contact-changing moves | 0 | 1 |
| Accepted embedded contact exchanges | 0 | 0 |
| Kernel CPU seconds | 17.60 | 77.49 |

The twelve contact-preserving acceptances were small embedded refinements,
with member RMS displacement about 0.011--0.074 Angstrom. They do not represent
docking. The single contact-changing event moved the free dimer of tetramers
`[27,132]` onto the existing dimer `[179,228]`, forming a four-tetramer aggregate.
Both moving members gained a contact to spectator 228; no contacts were lost.
The move's member RMS displacement was 123.65 Angstrom and its rotation was
172.53 degrees. Its fused destination constrained two different carried members
against that same spectator.

For that event, the logged factors were map -52.04398470, anchor selection
+5.46340799 and realized Poisson depletion +48.80717922, giving an unclipped
log acceptance factor +2.22660250. These are proposal/auxiliary factors for one
event, not basin free energies. Native registry was not established.

The control's dynamical events and phase endpoints reproduced the earlier
contact-anchor benchmark exactly for all four seeds. Every attempted event,
including rejections and null proposals, was retained and replay-checked.
Construction increased phase CPU by a factor of 4.40. One observed attachment
from one frozen state cannot establish an efficiency advantage, mixing time,
equilibrium stability or assembly success.

## Correctness issue in unchanged HEAD

The targeted `oligomer_proposal` suite passed three tests and failed
`docked_pair_fuses_and_context_rebuilds_identically`; a filtered rerun reproduced
the failure. The inverse-map and normalized-mixture checks passed with a fixed
catalogue, but rebuilding the catalogue after a rigid carry is a separate
requirement.

The failing scene retains only 32 components. Two distinct fitted basins have
nearly equal residuals (about 13.0416257636645 and 13.0416257636630). Roundoff
changes which survives the cap: pair `(7,15)` is replaced by `(4,12)`. Sorting
retained components by semantic labels cannot recover an already discarded
component. The rebuilt densities differ by about 1.7875 in log density at the
affected centers, a factor of about 5.97. This is not merely an index permutation.
Removing the cap in this diagnostic retains the same 78 label pairs.

A second audit reconstructed catalogues for 21 actual protein contexts: the
first two hard-valid learned trials per population plus all 13 accepted trials.
One rejected protein trial also changed retained label pairs. All 13 accepted
trials retained the same labels; their largest checked endpoint log-density
difference was about 3.94e-8. For the substantial attachment, the reverse log
correction residual was about 1.85e-9 and reverse handle-position error about
7.40e-7 Angstrom. These checks support the interpretation of that particular
event, but do not repair the general reconstruction failure.

The next correctness work is to make budgeted catalogue construction robust to
near-tied fits, or account explicitly for different endpoint catalogues in the
reverse proposal. Then repeat validation before making production sampling
claims. Merely relaxing the reconstruction test would not resolve the issue.

## Resolution

Construction now reads internal offsets rounded to a fixed grid (1e-5 Å,
quaternion components 1e-9), and the cluster phase rejects an oligomer trial
whose rounded key changes. The guard is symmetric in the endpoints, and when it
holds all construction inputs are bitwise identical, so the catalogue is too.
Details and tests are in [the oligomer chart doc](oligomer-proposal.md#balance).
The reconstruction test now asserts bitwise equality, and a new capped stress
test fails with unrounded offsets and passes with the key.

The oligomer arm was rerun on the same frozen input, seeds, 4 x 128 phases and
two concurrent jobs (`runs/oligomer-7400-keyed-20260926`, with the base commit,
uncommitted diff and executable hash):

| | `f2e33c3` | Rounded key and guard |
|---|---:|---:|
| Learned attempts | 709 | 707 |
| Hard-valid learned trials | 142 | 140 |
| Accepted learned moves | 13 | 12 |
| Accepted contact-changing moves | 1 | 1 |
| Guard rejections | | 0 |
| Kernel CPU seconds | 77.49 | 77.9 |

The contact-changing move is the same attachment: dimer `[27,132]` onto 228,
gaining contacts 27-228 and 132-228 and losing none. The other accepted moves
are again small embedded refinements. The counts differ slightly because the
rounded offsets change fits in their last digits, which changes later random
draws. The earlier caveats stand: one attachment from one frozen state is not an
efficiency, mixing or assembly result.

## Artifacts

- [Full conditional report](../runs/oligomer-7400-benchmark-20260926/report.md)
- [Comparison figure](../runs/oligomer-7400-benchmark-20260926/oligomer-7400-comparison.png)
- [Validation status](../runs/oligomer-7400-benchmark-20260926/validation-status.json)
- [Toy reconstruction witness](../runs/oligomer-7400-benchmark-20260926/reconstruction-audit.log)
- [Actual protein catalogue audit](../runs/oligomer-7400-benchmark-20260926/catalogue-rebuild-audit.json)
- [Member/anchor interface decomposition](../runs/oligomer-7400-benchmark-20260926/fused-interface-audit.md)

The result directory contains frozen inputs, executable and compiled-source
hashes, all event/phase records, and standalone analysis scripts. The untracked
`oligomer_fusion.rs` design was not linked or used. Existing production jobs and
all production Rust source files were left unchanged.
