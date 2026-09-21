# Frozen-atlas contact exchange within the AB shoulder

This sampling benchmark tests one moving rigid
tetramer conditional on both fixed AB neighbors, the unchanged 18 Å capture
ball, and the strict original registration window `1 < q < 2`.

The physical density is

\[
\pi_{\mathrm{shoulder}}(x)\propto
 H(x;A,B)\,\mathbf1_{|t|\le18}\,\mathbf1_{1<q(x)<2}
 \exp\{z\,|E(x)\cap(E(A)\cup E(B))|\}.
\]

The original q uses maximum member displacement divided by 2 Å and proper
rotation error divided by 15°. Neither tolerances nor region boundaries are
rescaled by fitted chart widths. The bath has depletant radius 1.5 Å and
activity 0.035 Å⁻³. Both neighbors remain in the complete hard and depletion
union even when only A is used as a proposal anchor.

## Fixed comparison

Every cycle has the same first two local attempts. Its third slot is either
another local attempt, a posterior-source Gaussian redraw (`c=0`), or
posterior-source correlated chart transport (`c=0.9`). All three use the same
translation step list `[0.2, 2.0]` Å, rotation list `[1.5, 15.0]` degrees and
rotation probability 0.5. `--method local` counts all three slots as local.

The transport/redraw arms use the immutable broad five-component model from
`runs/ab-shoulder-contact-atlas-preparation-20260921/model-broad.json`, fixed
anchor index 0, and a separately selected uniform cube/Haar branch with
probability 0.05. The atlas weights are proposal allocations, not physical
contact probabilities. The earlier integrator's 0.99 hybrid allocation and
geometric cover are not part of this MCMC kernel.

The Gaussian branch selects its source from posterior chart responsibilities
and its destination independently from the fixed weights. Its proposal
correction is `log G(old) - log G(new)`, including the full Gaussian mixture
and the normalized Haar chart Jacobian. At c=0 this is a matched independent
Gaussian redraw. The existing `--method mixture` uses a different treatment
of the defensive branch and is not substituted for that control.

The conditional-Poisson endpoint gate is unchanged. For gained/lost covered
volumes it draws `G ~ Pois(lambda V+)` and `L ~ Pois((lambda+z) V-)` and
accepts using `min(1, exp(proposal correction) (1+z/lambda)^(G-L))`.
The benchmark fixes `lambda/z=64`. It does not insert fresh absolute
importance-weight estimates into a Metropolis ratio.

The bounded protocol has two archived starts (direct and geometry contacts),
two independent seeds per start per arm, and 5,000 cycles per run:
**12 runs, 180,000 attempted moves**. Seeds are `115101010 + 1009*i`,
`i=0,...,11`. Burn is fixed at 1,000 cycles, all later retained endpoints
including repeats have equal statistical weight, and concurrency is at most
four. Starts are deliberately different, not independently equilibrated.

## Support and diagnostics

`DockingConfig.target_region` explicitly contains the original `NativeMetric`
and a `QWindow` with both inclusion flags. A missing region preserves the
previous unrestricted captured target. Boundary-invalid proposals are
self-loops; there is no retry-until-valid. Initial and restarted states must
also satisfy the window. `proposal_anchor_index` defaults to the old random
anchor rule and changes only proposal coordinates.

The optional `DockingConfig.target_region` constraint is implemented by
`docking-mc` only. The absolute `basin-normalizer` and
`latent-region-normalizer` share the configuration type but now reject a
non-null `target_region` before geometry, region-file reads or output creation,
rather than silently integrating a different target. Their historical absent
or null configurations retain their previous behavior. Use those integrators'
own explicit region definitions when preparing independent normalizers.

Five physical labels use the frozen geometric masks: within `1<q<1.1`,
assign the radius-0.5 direct, mixture and geometry balls in that priority;
then label their complement as inner remainder. The complete `1.1<=q<2`
region is outer shoulder. Labels retain overlap priority and exact original
boundaries; selected Gaussian chart labels do not replace them.

The observer replays every attempt, audits the proposal/gate arithmetic,
and measures physical-label transitions and contact incidence separately
for A and B. It retains self-loops and compares initializations and trajectory
halves. Integrated autocorrelation and effective counts are descriptive until
stationarity and coverage are supported. A constant indicator has unresolved
ESS, not infinite efficiency. Contact observer cost is reported separately
from sampler CPU.

Individual component kernels are reversible; their ordered cycle need not
be. Flux diagnostics should therefore retain the attempted kernel/slot.
Comparable transition rates in opposite directions are not required when
physical populations differ. In the unrestricted AB target, the current
other/native ratio is about 2.2e-5; absence of native exits in this small pilot
would not diagnose a sampling failure. The present conditional shoulder
benchmark cannot establish full-target native escape, assembly, or physical
kinetic rates.

## Preparation and validation

`tools/prepare_shoulder_docking_benchmark.py --out NEW_DIRECTORY` writes an
inert 12-run plan, immutable input copies, hashes and configuration files.
Without an explicitly supplied reviewed `--binary`, its shell plan stops
without running. It never builds or launches a simulation and never infers
the potentially older executable in `target/release`.

To build a separate review executable without replacing that release binary:

```bash
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target/shoulder-docking-review \
  /home/xvg/.cargo/bin/cargo build --locked --release --bin docking-mc
```

After review, create a fresh preparation with `--binary` pointing to that
executable. Its `commands.json` contains exact commands. The observer is
`tools/analyze_shoulder_docking_benchmark.py CAMPAIGN --out NEW_REPORT`.

`tools/run_shoulder_docking_benchmark.py --campaign CAMPAIGN --journal NEW_JOURNAL
--assessment NEW_REPORT` executes the frozen plan with its recorded worker count,
then invokes the archived observer. `--preflight` verifies inputs without
launching a kernel. Existing job outputs, logs or controller statuses prevent
a restart. Failed children are drained and preserved, and no observer starts
until all twelve physical jobs terminate successfully.

Preparation also accepts fixed `--cycles`, `--burn-cycles`, `--seed-base`
and `--workers` arguments. Defaults reproduce the original pilot allocation.
At least two retained endpoints are required for half diagnostics; worker
counts are limited to 1–32 and the observer never launches more processes
than there are jobs. Exact CLI budgets, method allocations and distinct seed
offsets are validated before execution. These options do not alter a run
after outcomes become available.

The executed preparation is `runs/ab-shoulder-docking-pilot-12x5000-20260921`.
Its manifest SHA-256 is
`b5613f96f4bd91f3d2ec062813d3b98f3ac61e305f6990552a70d64912fc62fb`;
the separately built executable SHA-256 is
`c9165404d38021f979931b2dc36a23aba09e60ba041c60264c5050b3a651738f`.
The runtime source bundle is checked against this preparation's archived
source bytes. This binary predates the later fail-fast guards in the two
other integrator entry points; its docking kernel is the reviewed kernel
described here. Those guards do not change the running experiment.

## Frozen longer independent control

The independent repeat in `runs/ab-shoulder-docking-repeat-12x40000-20260921`
uses **12 runs of 40,000 cycles**, with fixed burn 8,000, seeds
`115201010 + 1009*i`, and 12 workers: **1,440,000 attempted moves**.
The original model, physical reference, starts, metric, bath and move sizes
are unchanged. Every configuration is compared against its original pilot
counterpart, allowing only the fresh seed and archived shape-file location
to differ. Neither Gaussian weights nor covariance maps are refitted.

The comparison plan was frozen before any repeat trajectory in
`runs/ab-shoulder-docking-repeat-comparison-plan-20260921/plan.json`
(SHA-256 `097798cebf3bc5caa8d6c0c0c69f5f29f15c7228e00523f032b92410aa769b20`).
It retains all runs, self-loops, half and initialization comparisons, label
and atomic-contact diagnostics, and CPU-normalized roundtrips. It forbids
optional stopping and pooling chain boundaries into a single autocorrelation.
The repeat's manifest SHA-256 is
`b27ab410f2fcad7d6d7d1056b517816ba8d46e443c89d2386f702a220dc463ea`.
Its rebuilt executable includes the two integrator guards in the embedded
source archive; the docking kernel is unchanged. All twelve runs and the
automatic independent observer have now completed successfully.

## Independent 40,000-cycle results

The longer, independently seeded control retains 32,000 endpoints per run
after its fixed burn. The comparison uses the original physical reference,
not a reference chosen after seeing these trajectories.

| Mode | Direct–geometry roundtrips (four runs) | Full post-burn sampling CPU s | Roundtrips / CPU s | Initialization-mean TV | Largest run half TV |
|---|---|---:|---:|---:|---:|
| Local | 3 (3, 0, 0, 0) | 1267.63 | 0.002367 | 0.13113 | 0.62525 |
| Posterior c=0 | 202 (51, 57, 46, 48) | 1672.72 | 0.120761 | 0.07248 | 0.15419 |
| Posterior c=0.9 | 508 (138, 106, 128, 136) | 2190.84 | 0.231875 | 0.02964 | 0.12831 |

Correlated transport retains **1.920 times** the observed roundtrips per
full post-burn sampling CPU of independent redraw, versus 2.107 in the
short pilot. All four transport runs contribute exchanges. These are additive
event counts divided by summed CPU, not a pooled-chain autocorrelation or
an equilibrium kinetic rate. Total sampling cost, including burn, is
6,410.47 CPU seconds across all twelve runs.

The stronger check is the improvement in coverage: transport's difference
between initialization means falls from 0.10963 to 0.02964, and its largest
within-run half difference falls from 0.333 to 0.12831. Nevertheless, this is
not a stationarity certificate. Individual transport occupancies differ
from the frozen importance reference by TV 0.039–0.088, and the reference
itself has finite-sample uncertainty. Local sampling remains strongly
heterogeneous; its smaller initialization-mean discrepancy hides one run
with half TV 0.62525 and three runs with no direct–geometry roundtrip.

Individual apparent joint atomic-contact ESS/CPU ranges are 0.049–0.129
for local, 0.190–0.471 for redraw and 0.251–0.531 for transport. These are
lower than several short-pilot values as slower modes enter the longer
records. Their ranges overlap between redraw and transport, and they remain
descriptive. The demonstrated gain is sustained contact exchange per CPU
with improved occupancy agreement, not a certified stationary ESS speedup.

The [completed repeat audit](../runs/ab-shoulder-docking-repeat-assessment-20260921/analysis.json)
replays all **1,440,000 attempts**, independently checks all **303,878 learned
full-mixture corrections**, checks 2,044 representative chart maps, and
audits **384,000 post-burn atomic-contact frames** against both neighbors.
The [frozen-plan comparison](../runs/ab-shoulder-docking-repeat-comparison-20260921/comparison.md)
retains all 24 separate short/long records and the slot/kernel flux matrices.
`tools/compare_shoulder_docking_repeat.py --out NEW_DIRECTORY` reproduces
the comparison from completed audits and source identities, without
rerunning trajectories or physical audits. Its synthetic tests cover missing
runs, changed targets, unequal CPU denominators and constant descriptors.

![Independent longer contact benchmark](../runs/ab-shoulder-docking-repeat-figure-20260921/shoulder-docking-benchmark.png)

This supports testing the kernel with mobile neighbors. It does not establish
the cost of forming the AB neighborhood, native escape in the unrestricted
target, or assembly. The next all-mobile control retains the full-mixture
capture kernel alongside frozen posterior transport; the Gaussian-only
transport correction by itself lacks the capture kernel's uniform reverse
density floor at dilute configurations.

`tests/docking_region.rs` checks strict/closed support boundaries, unchanged
default configuration semantics, spectator-only anchor effects, both physical
neighbors, self-loops, two-direction exchange, matched local slots and exact
restart on a synthetic two-neighbor system. Existing posterior-chart,
conditional-Poisson reference and asymmetric-runner tests remain applicable.

## Original 5,000-cycle pilot results

All twelve runs completed. After the fixed 1,000-cycle burn, the four runs
per mode retain 16,000 cycle endpoints in total, including every repeat.
The following counts use every post-burn attempted move. A direct–geometry
roundtrip is a nonoverlapping direct→geometry→direct visit sequence or its
reverse; intervening labels are allowed. These are returns between the
frozen physical masks, not between selected proposal-chart labels.

| Mode | Direct–geometry roundtrips (four runs) | Full-run sampling CPU s | Full post-burn sampling CPU s | Roundtrips / post-burn CPU s |
|---|---|---|---|---|
| Local | 0 (0, 0, 0, 0) | 189.32 | 148.60 | 0 |
| Posterior c=0 | 23 (5, 8, 5, 5) | 270.25 | 216.45 | 0.10626 |
| Posterior c=0.9 | 66 (23, 16, 12, 15) | 362.94 | 294.78 | 0.22389 |

Within parentheses, runs are ordered direct-start replicates 1–2, then
geometry-start replicates 1–2. CPU totals include all sampling slots,
rejections, gate work and recorded sampling overhead, summed over all four
runs; the independent observer's cost is separate. Correlated transport
produces **2.107 times** the observed roundtrips per post-burn CPU second
of the matched c=0 redraw. Local proposals produce no direct–geometry
roundtrip in this record. This is a finite contact-exchange result, not a
stationary ESS speedup or a physical transition rate.

| Mode | Initialization-mean occupancy TV | Largest run half-occupancy TV | Individual joint-contact ESS / post-burn CPU s | Individual reference-occupancy TV |
|---|---|---|---|---|
| Local | 0.78275 | 0.3750 | 0.308–0.545 | 0.232–0.638 |
| Posterior c=0 | 0.13088 | 0.3475 | 0.364–0.697 | 0.094–0.218 |
| Posterior c=0.9 | 0.10963 | 0.3330 | 0.563–0.845 | 0.043–0.151 |

TV is half the sum of absolute differences across the five physical-label
occupancies. Initialization TV compares the two-run mean from each start;
half TV compares the two retained halves within a run. Reference TV uses
the importance estimate frozen before this pilot, with its finite-sample
uncertainty; no threshold turns these distances into a convergence test.

The atomic-contact ESS values are individual, sample-centered trace
statistics of joint A+B mobile/fixed atom incidence. They are not pooled
or averaged into a speedup. Local trajectories illustrate the limitation:
they have finite apparent contact ESS while retaining strong initialization
dependence and never completing the specified contact roundtrip. Transport
improves observed exchange, but half-occupancy TV remains as large as 0.333
and the four runs do not establish equilibrium or unseen-mode coverage.

The [completed assessment](../runs/ab-shoulder-docking-assessment-20260921/analysis.json)
audits all **180,000 attempted moves**, the original-q support, retained
self-loops, proposal and gate arithmetic, and all **48,000 post-burn
contact frames** against both neighbors. The scope remains one mobile
tetramer in the conditional shoulder. Full-target native equilibration,
protein assembly and physical kinetics are not inferred.
