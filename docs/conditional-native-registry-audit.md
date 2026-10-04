# Native registry after the matched reorganization benchmark

The running two-neighbor singleton comparison measures contact reorganization
with all rejected states retained. Its frozen contact observer deliberately has
no native classifier. A change of contact partners need not improve native
registry; conversely, registration can change while the partner set stays fixed.
The separate native audit therefore must not be conditional on a favorable
contact-exchange result.

This is a conditional diagnostic at **1.4 Å, 0.0275 Å⁻³**, with two mobile
tetramers and 262 frozen spectators from the 500 μM source. It does not replace
the original **1.5 Å, 0.035 Å⁻³, approximately 106.8 μM** assembly question.
No extra sampling, proposal adaptation or bath calculation belongs to this audit.

## Fixed coverage and interpretation

After all 64 new chains and their unchanged contact observer finish, include all
128 matched chains: fused singleton, unfused singleton, local and `m4`; four
contexts, two initializations and four streams per arm. Observe every initial
endpoint and all 4,608 retained blocks: **589,952 endpoints**, including residence
repeats and warmup. Production uses blocks 513–4,608, with block 512 retained as
the baseline for detecting a change into the production window.

Use the completed contact-analysis receipts to authenticate the original
journals, then extract their retained poses. The new endpoint adapter must not
repeat the already completed proposal/bath arithmetic replay. It still checks
endpoint order, complete cadence, pose validity, production flags and CPU values.
An accepted-only, event-selected or native-selected subset is not this audit.
The [endpoint and metrics adapter](../tools/conditional_native_metrics.py) now
implements those pure reductions. Its reader requires prior receipt/byte
authentication by the production caller; it is not a standalone journal audit.

Report external native partner sets and external motif labels separately from
the mobile-mobile native bond. Report entries, exits, direct changes between
nonempty native environments and completed observed returns separately. Preserve
cycle-inconsistent states and all alternative labels. A catalogue-consistent
isolated body is not a native attachment. Use full sampler CPU denominators and
separate observer/setup cost. Constant traces have undefined ESS; different
contexts remain different conditional targets and must not be pooled.

## A conservative geometric filter

Let `b_k` be the four member centers in a tetramer's body frame, `b̄` their mean,
and `(d_m, R_m)` an ideal directed motif. The unchanged native classifier requires

\[
\max_k\|R b_k+d-(R_m b_k+d_m)\|\leq\varepsilon,
\qquad \varepsilon=2\;\text{Å}.
\]

Taking the mean and applying the triangle inequality gives the necessary condition

\[
\|d+R\bar b-(d_m+R_m\bar b)\|\leq\varepsilon.
\]

In world coordinates, the moving member centroid `p_j + R_j b̄` must lie near
one of the anchor's predicted centroids `p_i + R_i (d_m + R_m b̄)`.
[The candidate filter](../tools/native_pair_candidates.py) queries these points
in a spatial tree. It preserves the original directed `i < j` convention rather
than assuming every catalogue contains reciprocal motifs.

The implementation uses a closed, outward-inflated infinity-norm neighborhood,
which contains the necessary Euclidean ball. Centered extended-precision
construction and a scale-dependent floating-point guard reduce numerical risk;
unsafe arithmetic falls back to all requested pairs. This is not a formal proof
of the floating-point implementation or SciPy. False positives are intentional.
Every candidate still passes through the unchanged full native pair classifier,
including member, orientation and residue-contact checks. The filter creates
no native labels and uses neither depletion weights nor contact-edge labels.
The first version supports open/spherical geometry only; periodic image
conventions require a separate implementation and validation.

## Cached conditional observer

[The passive adapter](../tools/conditional_native_observer.py) keeps the spectator
native graph once per conditional context and recomputes only mobile-related
labels. Reuse one adapter per context across its 32 chains. An exact pair-pose
cache avoids repeating the full classifier. A one-entry endpoint cache also
avoids rebuilding candidate searches after rejection, while still producing
every observation and updating its observation count. The graph cache is bounded.

Fixed edges participate in cycle checks. The report retains the full graph and
also checks components touching either mobile body, so a separate frustrated
spectator component cannot erase a valid mobile component. No inconsistent edge
is deleted to manufacture a compatible subgraph. The graph checker's actual
default angular tolerance is `1e-6` **radians**; it is not the degree-valued
scaffold-triangle criterion.

The geometry adapter requires a hard/wall-valid retained state and an already
authenticated frozen native definition. It is not a substitute for those checks.
Use the exact benchmark shape and the existing 4,004-atom identity witness in
`results/native-class-line-proposal-preparation-20261003/queries/saved/manifest.json`,
whose SHA256 is `d04145f5d458233bd138c384e2b244c5888f44cd9158dc34ccf969507b6ca653`.
The matching native definition is in that preparation's
`execution-code/native-definition/definition.json`, SHA256
`5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9`.
Bind its complete input closure; a similarly named file is insufficient.
Only the geometric pair predicate transfers from the old definition, not its
R4 integration domain or fixed-scaffold physical target.

## Validation and remaining production work

The candidate filter passed 12 synthetic tests, including comparison to an
independent unpruned all-member inequality, noncentered shapes, directed motifs,
rigid transformations, boundary cases, empty/duplicate catalogues and numerical
fallback. The conditional adapter passed nine tests against unpruned labels and
graphs, including exact residence caching, multiple matching labels, isolated
bodies, internal bonds, frustrated cycles and fatal classifier errors.
The endpoint/metrics adapter passed 14 further synthetic tests. One integration
test connects endpoint extraction, the full toy classifier, candidate/cache and
cycle handling, and metrics: it distinguishes a direct native-partner change
from detachment and a return through an empty state, retaining rejection
residence and using full sampler CPU. All 36 tests passed; no protein trajectory
was read by these tests.

Source-bound receipts:

- `results/native-pair-candidates-validation-20261004/validation.json`, SHA256
  `c150090469cceefe27ecbed06ebe9cc708aaec42303d985706a5fa5feb71bc20`.
- `results/conditional-native-observer-final-validation-20261004/validation.json`,
  SHA256 `939fb9c83ead2208757378f8d396bdae8a8b2a43467839069e57c54d017e14d6`.
- `results/conditional-native-metrics-validation-20261004/validation.json`, SHA256
  `60842f62edbd67546f2cc5be5509b283d82b40cf12b5e5972c4221596b3b8a1f`.
- `results/conditional-native-pipeline-validation-20261004/validation.json`, SHA256
  `9278d0dde946e732a78d612efee851d841cf9abbec5ee40fdcad139b95cc994c`.

These tests used no protein observations. The native audit has not run. Before
production, bind the completed contact receipts, freeze the endpoint/metrics
adapter and one-worker resource/query allocation, and include classifier setup
and fixed-graph queries in durable attempt accounting. Leave partial outputs
visible and do not replace failed chains. The unpruned pair-query ceiling is
309,724,800 mobile-related calls plus 136,764 fixed-pair calls over four contexts;
the candidate and exact-pose caches should reduce actual calls, but their protein
cost has not been measured. Constructor reference/scaffold work is additional
and uses the existing bounded setup instrumentation.

The result can show whether a move accesses distinct native registrations more
efficiently in these conditional environments. It cannot establish converged
equilibrium weights, finite-system assembly stability or physical kinetic rates.
