# Frozen passive comparison of two-distance contact guides

**The fixed proposal allocation and all independent audits have completed.** This tests the
[two-distance proposal](two-distance-contact-guide-design.md) at the original
repaired shape, R4/scaffold, 1.5 Å radius and 0.035 Å⁻³ activity. It adds no
Poisson clouds or physical-weight estimates. Execution used the reviewed
reference tests, independent full-density audit and source-bound draining
controller. The [execution protocol](../runs/contact-distance-passive-20261001/protocol.json)
and [completed status](../runs/contact-distance-passive-20261001/status.json)
are separate from the immutable preparation below.

## Label construction without held-out pose fitting

Each of the unchanged 92 Gaussian means is decoded to its physical pose. For
each scaffold, choose the atomic pair minimizing center distance minus the
two hard radii. Four fixed-radius KD trees make this exact: within a radius
group the nearest center minimizes surface gap; the minimum across all moving
atoms and radius groups is the global surface minimum.

The strict floating-point minimum wins, with lexicographic atom indices for
exact ties. All near ties within 1e−10 Å are recorded separately without changing
the selected minimum. Independent exhaustive small-system checks cover varying
radii, exact/near ties and the radius-polygon geometry. No native labels,
contact filter or validity filter selects components. The 78 critical and 128
breadth poses are copied only after all labels and guides are written.

The old92 integration model itself was already fitted with regional/native
information; deriving labels geometrically from it does not make this a blind
assembly proposal. This preparation adds no new fit or use of held-out poses.

The [complete label archive](../runs/contact-distance-passive-preparation-20261001/component-labels.json)
records all means, raw/world coordinates, atomic witnesses, gaps, ties,
conditional translation scales and polygon areas:

| Deterministic mean-pose diagnostic | Count |
|---|---:|
| Components retained | 92 |
| Distinct two-contact label tuples | 43 |
| Hard-valid means inside R4 and capture | 92 |
| Both scaffold gaps ≤0.02 / 0.1 / 0.5 Å | 10 / 55 / 79 |
| Positive radius polygons at each of the three widths | 92 / 92 / 92 |
| Exact or near-tied scaffold queries | 0 / 184 |

The smallest effective-center separation is 1.6545 Å. Direct recomputation and
KD distances differ by at most 4.45e−16 Å. These are facts about component
means, not expected proposal acceptance or equilibrium occupancies.

## Fixed law and allocation

Each component has one frozen pair-pair label, with scaffold indices `[0,1]`.
The three widths **0.02, 0.1 and 0.5 Å** are equally likely. Keep the uniform
R4 probability at 0.5 and every old Gaussian mean, covariance and weight exactly
unchanged. Here β selects distance conditioning within the Gaussian half;
b selects the localized wrapped-Cauchy part of its azimuth mixture.

| Arm | β | b | Fresh streams × attempts | Critical queries | Breadth queries |
|---|---:|---:|---:|---:|---:|
| baseline92 | 0 | 0 | 4 × 128 | 78 | 128 |
| uniform_phi92 | 0.5 | 0 | 4 × 128 | 78 | 128 |
| localized_phi92 | 0.5 | 0.9 | 4 × 128 | 78 | 128 |

Total: **1,536 fresh draws and 618 archived queries**, in 15 sequential jobs
with one CPU worker. Each arm's critical probes accompany r00; each breadth
control is a separate zero-fresh-draw job. Fifteen distinct seeds were checked
against 143 existing protocol/plan declarations. No retry, optional stopping
or automatic extension is allowed. Preserve every attempted draw.

The fixed effective-center-distance and polygon-area floors are 1e−8 Å and
1e−16 Å². Azimuth parameters are radius floor 1e−8 Å, projected-mean floor
1e−10 Å and angular scale bounds [0.01, π]. Empty/small geometry uses the old
conditional translation Normal at the **same angular draw**; a degenerate
azimuth projection uses the declared uniform law. No hard-valid retry occurs.
The complete proposal sums all component, width, fallback and azimuth branches,
including the chart determinant; its density retains at least half of old92.

Evaluate whole-union hard validity and simultaneous contacts per **total CPU**,
alongside R4/capture losses, selected-pair contacts, fallback reasons and
between-stream variation. Merely satisfying the two selected atom pairs is
insufficient. Archived weighted diagnostics remain retrospective and separated
by their original source. This allocation cannot open the original physical
convergence gates or decide finite-system assembly.

## Reproducible preparation

The helper and four focused checks are
[prepare_contact_distance_passive.py](../tools/prepare_contact_distance_passive.py)
and [test_prepare_contact_distance_passive.py](../tools/test_prepare_contact_distance_passive.py).
Preparation used 1.60 CPU seconds. Its [frozen plan](../runs/contact-distance-passive-preparation-20261001/plan.json)
has SHA256 `c1cedd40062e5277434652411defc611aeeb477d3479159aa223a12b7eb658eb`;
the [file manifest](../runs/contact-distance-passive-preparation-20261001/freeze.json)
binds all 14 preparation files. The plan contains the exact proposed binary
commands, but does not bind or execute a binary.

```bash
cd /home/xvg/tetramer-mc
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_contact_distance_passive.py \
  --out /path/to/new-contact-distance-passive-preparation
```

Running this command only prepares a new immutable directory. No physical or
proposal sampling is launched by the helper.

The separate [sequential controller](../tools/run_contact_distance_passive.py)
provides `freeze`, `preflight` and one-shot `run` stages. Its seven
[focused checks](../tools/test_contact_distance_passive_controller.py) verify
the fixed allocation and commands, attempted-row/probe identity checks,
partial-output preservation, no retries and draining a started child after
interruption. It preserves twelve fresh populations and three breadth jobs
separately, invoking the independent density audit after each. Frozen
execution protocol SHA256:
`7faa707bfcd96a1c10b269810488788335b4d2d9106bbc030d34b5a13308a501`.

## Separately declared native-geometry diagnostic

A [separate declaration](../runs/contact-distance-native-diagnostic-declaration-20261001/declaration.json)
was frozen before any fresh proposal sampling, without modifying the allocation.
It binds the exact native classifier used in the prior contact-weight pilot,
including member displacement, orientation, shared residue-patch and
cycle-consistency criteria. Loading and validating its original physical
inputs took 0.019 CPU seconds; per-pose costs remain to be measured.

After all passive outputs pass their audits, classify every new hard-valid pose
inside R4/capture exactly once. Retain every invalid or outside-domain attempt
with null native labels and its original flags. The existing full-union
`ExclusionContact` additionally identifies contact at gap **<3 Å**; the passive
audit's maximum 0.5 Å criterion alone cannot separate unbound poses from all
competing exclusion contacts. Preserve native-without-contact anomalies rather
than changing labels.

Report native, contact-without-native and unbound geometry by arm, population
and proposal branch, including joint-contact and cycle-consistent subcounts.
These are proposal frequencies, not equilibrium probabilities. Archive
classifier CPU separately and retain the original unconditional denominator.
No archived physical rows are reclassified and no new Poisson clouds are used.

The declaration has SHA256
`cb7c4389e84a4ab380de379118ed4c2672d0582dc62c861625f9dce36b427be5`;
its [manifest](../runs/contact-distance-native-diagnostic-declaration-20261001/freeze.json)
binds all 26 files. The [one-pass adapter](../tools/analyze_contact_distance_native.py)
is now implemented; five [bookkeeping checks](../tools/test_contact_distance_native.py)
pass, including skipped invalid rows, complete unconditional counts and
preservation of native-without-contact anomalies. Its separately
[frozen execution protocol](../runs/contact-distance-native-diagnostic-20261001/protocol.json)
passes preflight, with SHA256
`1c3454fa8f2012f974cf7710e6dadda47dbfbfb48e01955dd0f5a15829e0362f`.
No classifier or contact rule changed. All 1,536 fresh rows are now accounted
for in the completed observer. The [results](contact-distance-passive-results.md)
keep this native-geometry diagnostic separate from proposal-density correctness
and physical contact weights.
