# Growth beyond the native triangle

The four-body positive control produces native attachment from a free fourth
tetramer in three of four trajectories. All four preattached controls retain
native connectivity, and the original three-body scaffold retains its registry
in all eight runs. This demonstrates accessible growth with the validated,
native-informed proposal. It does not establish template-free assembly,
equilibrium stability, or a physical attachment rate.

The [complete comparison](../runs/mobile-four-body-growth-comparison-v2-20260921/report.md)
retains every run and accepted/rejected update. The earlier comparison directory
is preserved; v2 fixes a clipped plot axis without changing any trajectory or
reported statistic.

## Matched experiment

All four tetramers are mobile. The initial ABC triangle is identical in every
run. D starts either free of all exclusion contacts or attached to C at native
motif 8. Two independent seeds per start compare the original 150-component
reciprocal atlas with the broader 178-component coverage atlas. These are both
native-informed proposals. Physical shape, bath and move parameters are fixed.

Each run has 2,000 sweeps and saves every sweep: four single-body attempts, one
spherical GCA attempt and one center-shift attempt per sweep. There are 96,000
audited move records in total, using 920.22 sampler CPU seconds. Both observer
audits passed. The exact archived controller is in
`runs/mobile-four-body-growth-campaign-20260921/provenance/`.

The depletant radius is 1.5 Å, activity 0.035 Å⁻³ and auxiliary intensity λ/z=64.
The physical spherical wall has radius 223.32617672378387 Å and permeates the
ideal depletant bath. Local translation and rotation scales remain 0.2 Å and
1 degree. Half the single-body proposals are global, equally split between
independent mixture proposals and correlated involutions at correlation 0.9;
the uniform floor is 10%. GCA and center shifts use the existing exact kernels.
Four bodies in this wall have 4/3 the body density of the preceding three-body
control. The two proposal arms have identical density; comparisons with the
three-body campaign are not density-matched.

| Proposal | Initial D | Repeat | First native attachment | Final D native monomer bonds |
|---|---|---:|---:|---:|
| Original | Free | 0 | Not observed | 0 |
| Original | Free | 1 | Sweep 177 | 6 |
| Coverage | Free | 0 | Sweep 2 | 2 |
| Coverage | Free | 1 | Sweep 43 | 2 |
| Original | Preattached | 0 | Initial state | 2 |
| Original | Preattached | 1 | Initial state | 2 |
| Coverage | Preattached | 0 | Initial state | 2 |
| Coverage | Preattached | 1 | Initial state | 2 |

The original free repeat 0 forms an exclusion contact at sweep 34 but never
passes the native-entry criterion. Original free repeat 1 attaches through the
correlated involution; both coverage attachments use independent mixture
proposals. Two seeds per arm cannot establish a speedup. The successful
attachment proposals have large unfavorable proposal-density corrections
(−47.39, −30.44 and −46.15 log units), compensated by positive sampled depletion
gate log weights (62.62, 46.36 and 59.92). These individual stochastic gate
values are not free-energy differences.

## Connectivity and registry are separate observations

Every observed native graph admits a compatible assignment of the ideal
catalogue transforms around its cycles, to 1e−6 Å and 1e−6 radians. Entry labels
and hysteretic retained labels are checked separately. A connected four-body
graph need not be a complete six-edge clique. The cycle check tests catalogue
compatibility; it does not require the measured configuration to be an exact
crystal or replace the hard-overlap test. A tail attached to a valid triangle
does not create an additional cycle constraint.

All four preattached runs leave their initial six-monomer-contact motif 8 and
end in the two-contact motif 7 on C. Coverage free repeat 1 does the same.
Coverage free repeat 0 first forms the original two-neighbor native pocket
(A motif 4 / B motif 10), then changes to motif 6 on A. Original free repeat 1
instead makes the motif sequence 8 → 3 → 8. Thus retained connectivity does
not mean retention of the initial docking interface, and more counted native
monomer contacts cannot be treated as a free-energy ordering.

![All eight trajectories](../runs/mobile-four-body-growth-comparison-v2-20260921/four-body-growth.png)

There are no full D attachment–detachment round trips. One coverage trajectory
loses one of its two native neighbor edges while remaining attached to the
other. The original triangle never loses registry. The retained motif changes
demonstrate reversible-kernel access to distinct registered interfaces, but
these short, unequilibrated trajectories do not measure their equilibrium
weights or independent-contact sampling efficiency.

The next comparison should keep the native-informed control while testing
geometry-only discovery. Independent integration must also resolve the
contact region just outside the native-entry threshold and the remaining
full-vessel weight. Neither a fourth-body attachment nor a conditional
one-body integral includes the thermodynamic cost of forming ABC.

## Implementation and validation

[The controller](../tools/run_mobile_four_body_growth.py) freezes both arms,
starts and source closures, launches at most eight physical workers, drains
every started child on failure, and prohibits retries or overwrite. It binds
the reviewed executable to its embedded source bundle. The only differences
from the previously tested assembly executable are in normalizer code; the
assembly kernels are unchanged.

[The observer](../tools/analyze_mobile_posterior_pilot.py) explicitly supports
four bodies while retaining the older three-body contracts. It checks all six
pairs and all twelve moving-body/anchor roles, replayed proposals and accepted
states, strict entry versus retained monomer support, collective crossing
pairs, initial-state censoring and attachment/registry histories.
[The separate cycle checker](../tools/native_graph_consistency.py) backtracks
over alternative motif labels to test compatible SE(3) assignments.

Five campaign tests, ten four-body observer tests, five cycle-consistency
tests and the 23 existing observer/graph tests passed. The all-atom preflight
confirmed both initial geometries and their strict monomer support without
sampling. [The validation ledger](../runs/mobile-four-body-growth-validation-20260921/validation.json)
binds those checks to the archived campaign and completed analyses.
