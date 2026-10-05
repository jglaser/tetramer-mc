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
