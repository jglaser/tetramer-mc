# Physical control for a neighborhood-dependent contact prior

The question is whether directing learned proposals toward available contact
destinations improves **physical contact-environment sampling per CPU**. The
Gaussian charts, their exact reciprocal branches, local move sizes, and
defensive uniform probability stay fixed. Only the learned destination prior
changes. This is a conditional experiment with one moving tetramer and 263
fixed tetramers, not an assembly or crystal-stability test.

The new `context-docking-mc` runner uses the full atomic spherical wall.
It has no capture ball, native window, or imposed contact constraint. The
uniform proposal cube contains every physically allowed body origin. The ideal
depletant bath permeates the wall, as in the existing physical model.

The original atlas is compared with the positive
[fixed-context prior](context-branch-prior.md). For fixed spectators `C`, its
weights define `G_C(x) = sum_j w_j(C) g_j(x)`. Selecting the source chart from
its posterior and the destination from `w(C)` gives the proposal correction
`log G_C(old) - log G_C(new)`. The context excludes the moving particle, so
the same prior applies in both directions. The separate uniform branch has
its own symmetric proposal correction. Both use the existing exact auxiliary
Poisson gained/lost-count acceptance kernel.

The runner authenticates the shape, original atlas, fixed poses, source
metadata, ordered virtual branches, and complete prior vector. Checkpoints
also bind the method, correlation, schedule, executable, and resource limits.
An alternative initial pose does not change the prior construction context.
Attempt decisions are journaled before contact observation; every rejection
remains in the trajectory. An incomplete cloud or unexpected numerical map
failure stops the invocation and preserves its prefix.

The frozen allocation is
[`results/context-prior-physical-pilot-20261005/allocation.json`](../results/context-prior-physical-pilot-20261005/allocation.json):

- Local, original-prior, and context-prior arms, with correlation zero.
- Four local slots plus one tested slot per cycle, matched across arms.
- Two certified starting environments, four streams each: 24 chains.
- 256 warmup and 2,048 production cycles, retaining all 276,480 attempts.
- Radius 1.5 Å, activity 0.035 Å⁻³, auxiliary intensity ratio 64.

The second start is chosen by a frozen geometric rule from all 51 eligible
branch centers. Invalid centers and failure to find an alternative remain
reported outcomes. Selection uses no native labels or depletion outcomes.
The historical neighborhood itself was selected using native diagnostics
and comes from a 500 μM snapshot, so this is not a native-blind discovery
result or the 106.8 μM all-mobile decision system.

The evaluation retains instantaneous patch labels and reports contact-pattern
ESS per full sampler CPU, agreement between starts, and persistent partner
exchanges. Constant traces have undefined ESS. Label flicker, accepted-move
throughput, and the number of available branch centers cannot establish
equilibrium contact weights. Context preparation and diagnostic costs are
reported separately; an all-mobile implementation would also need to pay for
updating the context prior as spectators move.

Launch requires source/full-wall certification, an independent equilibrium
sphere reference using the actual bath kernel, and deterministic continuation
and failure-prefix checks. The allocation alone does not imply those checks
or the protein sampling have completed.

## Integration validation

The first isolated release build passed all eight new integration tests,
including identical physical decisions and Poisson counters for uninterrupted
versus split/resumed runs. Tests also exercised the off-center atomic wall,
ordered prior bindings, alternative initialization, complete candidate
certification, and a fatal point-limit prefix with no partial-cloud acceptance.

The [build receipt](../results/context-docking-build-20261005/receipt.json)
has SHA256
`c13681e9f101284a960d584468a4d65f8699a51a555f1afaba2cbb988d1a1345`.
Compilation and unit execution used 153.75 child CPU seconds, plus 28.34
controller CPU seconds; all child groups drained. The production executable
and the reused dependency cache were unchanged. This validates the exercised
integration cases; the independent physical stationarity check and protein
pilot remain separate gates.

The subsequent [geometry certificate](../results/context-branch-prior-start-certification-20261005/certification/summary.json)
validated all 263 fixed bodies, the saved source, and all 51 eligible branch
centers. Forty-seven centers have a different exclusion-neighbor set from the
source. The frozen rule selected virtual branch 32, with contacts `{16,56}`;
the source has contacts `{16,217}`. The
[selection record](../results/context-branch-prior-start-certification-20261005/selection.json)
retains every candidate. Certification and selection used 1.49 CPU seconds;
neither performed bath sampling or equilibrium inference. This clears the
two-start feasibility gate.

## Independent physical reference

The shared physical step completed 16,384 calls from all 8,192 cached IID
two-sphere configurations, updating one sphere while conditioning on the
other. Both original and context-prior arms passed all 24 predeclared paired
binary stationarity tests at familywise level 0.05. The smallest p-value was
0.0228, above the per-test threshold 0.002083. An intentionally incorrect
control reused the same candidate, bath and acceptance uniform while omitting
the proposal ratio; it failed 10 of its 12 tests.

The [independent audit](../results/context-prior-reference-preparation-20261005/audit.json)
also passed 260,584 reconstructions of densities, reciprocal charts,
Jacobians, auxiliary noise/label factors, count arithmetic, endpoint predicates
and decisions. Maximum absolute numerical discrepancy was `2.30e-9`, within
the predeclared scaled tolerances. The run used 186,550 raw bath points;
physical execution cost 1.98 CPU seconds and the saved-record audit 20.56 CPU
seconds. All process groups drained, without replacement sources or retries.

Audit SHA256:
`48ef84099b810f084b872248750541de8f18cc59e5765a9fad6bf597a8437711`.
The [host receipt](../results/context-prior-reference-preparation-20261005/execution-v4/receipt.json)
has SHA256
`098afd04da962de2dc790cf6da80b021d7050951354a4bf6cddb6ee90a571d5d`.
Six deterministic tests of the independent analyzer also passed. Earlier
controller versions were revised before execution to ensure child-process
cleanup on interruption; they made no scientific draws. This check supports
the implemented physical kernel in the tested reference limit. Protein mixing
efficiency and equilibrium assembly remain separate questions.

## Protein pilot

The fixed 24-chain allocation was launched after those gates and independent
review of all configurations and child-process cleanup. Its
[controller manifest](../results/context-prior-physical-pilot-execution-20261005/controller-manifest.json)
has SHA256
`6ccc5dc13e1d6e565e47269f2a4b69c6bd235f8cfa79d212df5eb3d556cf72d2`.
Four lanes each run six chains, with at most four new physical jobs at once.
All 119 input/source/binary bindings and paired stream seeds were checked.
Four synthetic controller tests passed before launch, including child failure
and termination of nested process groups. No retries or replacement chains
are permitted by this allocation.

The [metric specification](../results/context-prior-physical-pilot-20261005/metric-spec.json)
keeps every instantaneous retained state. A
[pre-launch addendum](../results/context-prior-physical-pilot-20261005/exchange-addendum.json)
also fixes the two starting neighbor sets as environments A and B. It records
five-cycle-confirmed passages between them through any intermediate environment,
alongside unconditional A/B/other occupancies. Thus a sequence such as
`{16,217} -> {16} -> {16,56}` can count as a completed A-to-B passage without
discarding the intermediate states from physical statistics. These are contact
descriptors, not native-registry or metastability definitions.

All 24 chains subsequently completed their full allocation: 276,480 attempted
moves, with no retries, replacements, null proposals, or failed chains. The
[host receipt](../results/context-prior-physical-pilot-execution-20261005/host-drained.json)
records 2,629.88 CPU seconds and 739.66 wall seconds, with all child groups
drained. This establishes execution completeness, not convergence.

The saved-state observer passed ten synthetic tests, including global versus
compact particle labels, every rejected residence, persistent autocorrelation,
and A-to-B passage through other contact sets. The final
[validation](../results/context-prior-observer-validation-20261005/attempt02/report.json)
also authenticates the compiled shared-step schema. An earlier observer draft
incorrectly expected the top-level proposal ratio to be absent on hard
rejections; the held sampler records it before geometry. That observer draft
and its unlaunched plan were retained and superseded before any protein
observation. The physical sampler and its completed draws were unchanged.

The corrected observer
[manifest](../results/context-prior-physical-observers-20261005-v2/controller-manifest.json)
binds all 24 chains and 758 source/input hashes. Each analysis has a 300 CPU
second, 600 wall second, 4 GiB cap, using at most four single-thread workers.
The all-stream comparator separately passed three synthetic tests and requires
all 24 completed analyses. It preserves undefined ESS, individual stream
ratios, all cross-start support comparisons, and missing shared construction
costs instead of imputing them.

## Completed comparison

All 24 saved-state observers passed, retaining all 276,480 attempted endpoints.
The [comparison](../results/context-prior-physical-comparison-20261005/comparison.json)
has SHA256 `9ef364c36de0d10c4099e37df9cbae56dec2831f9932c8dfa6914009d57fb397`.
The observers and their controllers used 311.46 CPU seconds and 109.40 wall
seconds; the final comparison used 0.83 CPU seconds. No new physical draws,
replacement chains, or accepted-state filtering entered the analysis.

| Production diagnostic | Original prior | Context prior |
|---|---:|---:|
| Learned proposals | 8,226 | 8,226 |
| Hard- and wall-valid learned proposals | 13 (0.158%) | 467 (5.677%) |
| Accepted learned proposals | 1 | 2 |
| Confirmed B-to-A passages | 1 | 2 |
| Confirmed A-to-B passages / round trips | 0 / 0 | 0 / 0 |
| Between-start neighbor-set total variation | 0.893 | 0.271 |
| Between-start full-fingerprint total variation | 1.000 | 1.000 |

The context prior improves learned-proposal feasibility **35.9-fold**. One
additional context stream reaches neighbor environment A during warmup; three
of four alternative-start context streams therefore visit A, versus one of
four original-prior streams. These are arrivals at the same **neighbor IDs**,
not the same patch contacts. The two initializations have disjoint observed
full-fingerprint supports in every arm, and some patch marginals differ by
one. Local trajectories also remain initialization-dependent. No equilibrium
contact weights or meaningful return-rate comparison follow from these runs.

There is no consistent contact-sampling efficiency gain. From the saved A
start, context and original priors have identical retained patch histories
within each paired stream; context's extra cost reduces patch and fingerprint
ESS per CPU to a median 0.896 times the original. From B, context/original
patch ESS per CPU has median ratio 2.26 among three defined streams, but one
context trace is constant and undefined; the fingerprint ratio is only 0.953.
The corresponding context/local patch ratio is 0.765 among the same three
defined streams. These finite-record numbers cannot rank equilibrium mixing
while contact coverage remains inconsistent.

Most geometrically valid learned proposals still fail the physical acceptance
step. For rejected valid context proposals, individual-stream median auxiliary
depletion log factors are approximately −92 to −97 from the saved A start,
and −26 to −47 from the B start. These are sampled **acceptance factors**,
not integrated basin free energies. They suggest that clearance alone does
not preserve enough exclusion overlap. A separate saved-trace diagnostic
will compare proposal and bath terms on exactly the same gated candidates.

![Matched contact-sampling comparison](../results/context-prior-physical-comparison-20261005/figures-v2/contact-prior-comparison.png)

The next sampling control is to retain correlated within-chart coordinates
during transfer, using the existing reversible transport, rather than fitting
another center-only prior. Independently, the full-domain absolute-weight
route in [the candidate-bank note](context-candidate-importance.md) could
separate poor proposals from thermodynamically rare environments. Its positive
proposal floor does not certify unseen pockets. The main independent
native/contact-region weight campaign continues separately; neither this
conditional pilot nor a future candidate-bank calculation decides finite-system
assembly stability.
