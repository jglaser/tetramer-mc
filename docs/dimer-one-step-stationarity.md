# Independent-source one-step stationarity control

This is a separately allocated diagnostic of the residual contact-occupancy
deficit in the completed sphere trajectories. It does not extend or replace
those trajectories. Every attempted move starts from a fresh equilibrium source
configuration; rejected moves remain paired observations with zero change.
There is **no orientation refresh after the physical move**.

The physical target is unchanged: two hard spheres of core radius 0.2 and
exclusion radius 1, activity 1.5, root center uniformly confined to a ball of
radius 2, and relative separation \(0.4\le d\le2.5\). The child has no separate
wall. The anchor is an identity coordinate frame, not a third physical sphere.

## Exact source construction

Draw \(u\in(0,1)\) and set

\[
d=[0.4^3+u(2.5^3-0.4^3)]^{1/3}.
\]

This baseline already supplies the radial \(d^2\) measure. Accept it with

\[
\exp\{z[V(d)-V(0.4)]\},\qquad
V(d)=\frac{\pi(4+d)(2-d)^2}{12}\quad(d<2),
\]

and \(V(d)=0\) otherwise. Sphere-lens overlap decreases monotonically, so this
is a rejection envelope with no estimated normalization or density adjustment.
The accepted distance therefore has density proportional to \(d^2e^{zV(d)}\).
Draw the root uniformly in its ball, an independent relative direction, and two
independent normalized four-normal quaternions. Directions use normalized
three-normal draws. These factors are independent in the target measure.

Every physical row retains the source rejection count, accepted source variates,
both full endpoint poses, proposal/auxiliary diagnostics, MH uniform, and outcome.
Source/proposal/bath/MH streams have separate per-attempt SHA256-derived seeds.
The source seed and attempt index permit deterministic replay, including rejected
radial trials. The Python audit reconstructs accepted source geometry and checks
source statistics; it does not independently replay Rust's random generator or
every auxiliary point. Exactness here describes the mathematical construction;
FP64, geometry predicates and RNG execution remain implementation obligations.

## Frozen allocation and decisions

The [configuration](../examples/dimer-one-step-stationarity.json) contains five
arms: correlated tree proposal with analytic, world-Poisson, or singleton-path
depletion; defensive independent redraw with world or singleton-path depletion.
Each arm has four independent populations of 16,384 source/move pairs: **327,680
attempts** total. There is no burn-in, adaptation, continuation or extension.
The atlas, correlation 0.7, defensive probability 0.5, relative cube half-width
2.5, auxiliary intensity 24 and envelope budget match the preceding controls.

The primary observable is \(f(X)=\mathbf1[d(X)<1]\). For each arm, count positive
and negative paired changes \(f(Y)-f(X)\). Under stationarity these probabilities
are equal, so, conditional on the total number of changes, the positive count
has an exact Binomial(·, 1/2) null distribution. The five two-sided tests form
one Bonferroni family with alpha 0.05 (0.01 per arm). Rejected and unchanged moves
remain in the mean change and variance denominator. Report estimated simultaneous
normal intervals and conservative simultaneous Hoeffding intervals as well;
the latter retain nonzero width even when no changes occur.

All 16 secondary paired observables are reported without favorable filtering:
distance, root squared radius, all eight root/child quaternion-component squares,
root/child/relative sums of quaternion fourth powers, the product of root and
child scalar-component squares, contact times root fourth-power sum, and squared
alignment of the relative separation with the root's rotated body-z axis. All
are invariant to either quaternion's sign. Their target means follow from the
factorized isotropic measure: component squares 1/4, fourth-power sums 1/2,
the scalar-square product 1/16, contact-fourth-power product \(P(d<1)/2\), and
axis alignment 1/3. Source radial moments use independent one-dimensional
quadrature, not the source sampler.

Source checks form a separate declared family: all 17 observable means pooled
over the 20 IID source populations, a seven-bin radial histogram with edges
0.4, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, and the mean geometric rejection count.
Moment/histogram intervals are asymptotic diagnostics and are labeled as such.
Per-population source, endpoint and paired moments remain visible. Source-check
failure prevents interpreting a paired null result as evidence of correctness.

## Power and limitations

There are 65,536 independent pairs per arm. The worst-case standard error of
the mean contact change is 0.003906. The earlier world-gate flip frequency near
0.0865 suggests about 0.00115 instead; at that variance, the corrected test has
approximately 80% power for a one-step drift of 0.0039. This estimate is a
planning calculation, not a revised threshold after observing results.

A roughly 0.01 stationary-occupancy error can correspond to only about 0.0017
one-step drift when relaxation is slow. This allocation may miss such a drift.
Failure to reject the primary nulls does not prove the kernels correct, settle
the earlier pooled deficit, or establish protein equilibrium/assembly. Successful
checks complement map inverses, accepted-flow identities and finite trajectory
evidence. Any flags are retained; the allocation is not enlarged in response.

The [Rust example](../examples/dimer_one_step_stationarity.rs) and
[independent analyzer](../tools/analyze_dimer_one_step_stationarity.py) are new
standalone files. The analyzer imports quaternion/density helpers from the prior
independent analyzer; that file must be included in the frozen source closure.
Before invocation, archive config, sources, dependency pins and binary hash in a
new plan. The executable creates a new output directory and fails rather than
overwriting one. No production executable or assembly kernel is replaced.

## Completed control and retained failure

The frozen allocation completed once at
`results/dimer-one-step-stationarity-20261002`: 327,680 attempts, 20 populations,
24.149 process CPU seconds. The independent all-row audit reconstructed the
complete proposal correction to a maximum absolute log error of
\(1.78\times10^{-14}\), and checked source reconstruction, gate counters,
MH decisions, and retained endpoints. No orientation refresh occurred.

| Arm | Mean paired contact change | Standard error | Exact p | Bonferroni p |
|---|---:|---:|---:|---:|
| Tree + analytic | +0.0005341 | 0.0012495 | 0.6780 | 1.0000 |
| Tree + world Poisson | −0.0015717 | 0.0011557 | 0.1781 | 0.8904 |
| Tree + singleton path | +0.0010071 | 0.0012063 | 0.4110 | 1.0000 |
| Defensive independent + world Poisson | **+0.0026245** | 0.0008267 | **0.001596** | **0.007979** |
| Defensive independent + singleton path | −0.0012817 | 0.0008741 | 0.1474 | 0.7369 |

The defensive independent/world arm **fails the predeclared primary test**:
1,554 contact gains versus 1,382 losses. Its estimated simultaneous normal
interval is [0.000495, 0.004754]; the more conservative Hoeffding interval
includes zero. These different intervals do not replace the exact primary
decision. No predeclared source-family check flags. Among the fully reported
secondary diagnostics, the same arm has mean-distance change −3.15 SE and
contact-times-quartic change +3.28 SE; these are correlated diagnostics, not
independent confirmations. No other secondary paired mean exceeds 3 SE.

The positive one-step flag has the opposite sign from the earlier evolving-chain
pooled contact deficit. Neither is discarded, and neither establishes the sign
or existence of a persistent bias by itself. There is no allocation extension.

The first frozen analysis encountered a **report serialization failure**, after
its row checks and statistics had returned: a SciPy p-value comparison produced
a NumPy boolean unsupported by JSON. Its log and partial output remain intact.
A separately frozen supplemental analyzer converts that p-value to a Python
float; its entire AST is otherwise identical. Five regression tests passed,
and only read-only analysis was repeated. Use `analysis-recovered.json`, not the
partial `analysis.json`. `completed-review.json` authenticates all 489 original
archived inputs, the supplemental import closure, the unchanged 20 physical
streams, and manifest/config/source/binary agreement. This receipt records the
scientific primary rejection separately from successful arithmetic/provenance
checks. The frozen prelaunch documentation is preserved in the archive.

## Exploratory saved-trial diagnosis

An explicitly post hoc scan of the flagged arm's existing 65,536 rows is saved
under `exploratory-saved-row-diagnosis/`. It applies the analytic depletion MH
decision to the **same old poses, proposals and acceptance uniforms**, without
new samples. This yields +133 net contacts, compared with +172 recorded with the
Poisson gate: paired difference +39/65,536 = 0.000595 ± 0.000560 SE.

Integrating over just the acceptance uniform for each saved trial gives predicted
mean changes 0.001783 ± 0.000757 for analytic depletion and
0.001746 ± 0.000701 for the saved Poisson cloud. Their paired difference is
−0.000037 ± 0.000348. Thus this saved proposal sample also has positive drift
under analytic acceptance; it does not isolate a Poisson-only discrepancy.
This comparison cannot erase the primary rejection or prove either kernel
correct.

The four populations' recorded net changes are +30, +1, +57 and +84; their
analytic counterparts are +48, −8, +27 and +66. Complete branch, source-contact,
and radial-source strata remain in the diagnostic artifact. Forward branch
conditioning does not itself preserve equilibrium: the complete mixture enters
the acceptance ratio, so branch-specific drifts need not vanish. These strata
are decompositions, not additional pass/fail tests or replacement estimates.
