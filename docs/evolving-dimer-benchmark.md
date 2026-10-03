# Evolving conditional dimer benchmark

The narrowed FFT proposal accepted physical moves in the reset-state pilot, including one external contact gain. That is evidence of accessibility, not contact equilibration or native growth. The next comparison follows actual retained trajectories, including every rejection, rather than repeatedly proposing from the same source.

This benchmark uses the existing repaired tetramer at **1.4 Å, activity 0.0275 Å⁻³**, in the saved sweep-7400 spherical configuration. It moves exactly two labeled tetramers against the fixed remaining bodies. It is a conditional relaxation test, not an all-mobile assembly test. The original decision conditions remain **1.5 Å, 0.035 Å⁻³, approximately 106.8 μM**.

The four contexts are selected without native labels from the archived eligible-pair table. The entire parent components used in the preceding width pilot are excluded. The remaining pairs are 27/132 and 32/110 (initially whole dimers), and 9/24 and 11/246 (initially embedded in larger components). Each anchor is the nearest nonmember to the root in the frozen source, with label tie-breaking; these labels never change during the trajectory. Contexts are different conditional targets and their occupancies must not be pooled.

Each context has four streams from the source and four from separately proposal-prepared poses. The latter are the first hard-valid, contact-preserving unguided candidates that move at least one center by 5 Å or rotate one body by 10°. Preparation has 256 outer attempts, records every attempt, and uses no energy or native filter. These are deliberately different starting configurations, not equilibrium samples. Failure to prepare a start does not trigger replacement.

| Arm | Four local attempts per block | Extra attempt per block |
|---|---|---|
| Local | Member order 0,1,0,1 | None |
| Unguided | Identical proposal sizes and streams | Factorized FFT redraw |
| Guided | Identical proposal sizes and streams | Same redraw with overlap threshold, m=4 |

Local moves use the production Gaussian translation and Cayley rotation law, with 0.2 Å and 1° parameters. They may detach the pair. The extra proposal has covariance multiplier 0.25², a 50% uniform relative-cube/Haar component with half-width 160 Å, root/internal caps 32/32, and one joint attempt. It is the identity when the selected pair is not in exclusion contact. There are no GCA or center-shift attempts in this fixed-scaffold comparison. Those kernels remain available in production all-mobile simulations.

The fixed allocation is 96 chains, each with 512 warmup and 4,096 production blocks: 1,769,472 local attempts and 294,912 extra dimer attempts. Three physical workers are allowed for this campaign, subject to the global eight-job/32-worker limit. No extra trajectories replace failures. Matched arms share their local RNG roles and starting poses; the four streams within each arm are independent, but cross-arm comparisons are paired.

## Balance and the fixed cloud

For each chain family, prepare a cloud P independently of all poses, by uniform draws in the root-body exclusion AABB and exact root-membership thinning. Freeze this cloud for the trajectory. If K(X,P) counts points also covered by the child, refresh k with

\[
a_m(k\mid K)=\frac{(k+1)^m-k^m}{(K+1)^m},\qquad 0\leq k\leq K.
\]

For **each fixed P**, the joint density \(\Pi_P(X,k)=\pi(X)a_m(k\mid K(X,P))\) has physical marginal \(\pi\), because the finite sum telescopes to one. The cloud need not be refreshed to preserve that marginal. Refresh k once per attempt and retain it throughout the capped retries. The acceptance correction is the complete forward/reverse edge-density ratio plus

\[
m\{\log[K(X,P)+1]-\log[K(Y,P)+1]\}.
\]

The factorized feasible sets and cap law are fixed during the attempt. Full mixture densities include both branches. The final endpoint receives the validated fair-order two-singleton Poisson bath path and **one** MH decision. The copied intermediate is not hard-filtered; the final endpoint is. Any arithmetic, geometry-consistency or resource failure terminates the chain with its partial record, rather than becoming a rejection or triggering a retry.

After that attempt the threshold is discarded. Unrestricted local moves can therefore change K or leave the contact domain without an auxiliary correction. Each elementary kernel preserves the same physical conditional target. Their ordered composition preserves that target, although the whole ordered block need not itself satisfy detailed balance. This argument concerns exact mathematics; geometry predicates, floating-point arithmetic and RNG execution remain implementation obligations.

The threshold normalization, marginalization and correction reuse the existing [fifteen checked Lean lemmas](../formal/ReversibleSampling/AuxiliaryOverlapThreshold.lean) and [their axiom audit](../formal/auxiliary-overlap-threshold-validation.json). No new continuous-state or Rust correctness theorem is claimed.

## Records and evaluation

`examples/evolving_dimer_benchmark.rs` writes every elementary attempt and every retained block. Named streams depend on the absolute block and role, so graceful checkpoint continuation does not require serializing opaque RNG state. A checkpoint must match the bound journal's final retained poses, counters, block, row count, byte length and hash. Uncheckpointed tails and completed/fatal-run resumes are refused, preserving existing evidence.

The observer uses instantaneous surface-patch and neighbor fingerprints, including internal and external contacts separately. Constant traces yield unresolved ESS, not perfect efficiency. Compare contact-fingerprint effective samples per actual CPU, occupancies between preparations, completed environment passages and returns. Patch-threshold crossings do not by themselves establish a distinct basin or native registry. Setup, preparation and observer CPU are reported separately. Retain the uniform component and report all unvisited regions.

Preparation, executable binding, independent preparation audit and dispatch are separate stages. `tools/prepare_evolving_dimer_benchmark.py` freezes inputs before any geometry search. `tools/run_evolving_dimer_benchmark.py` requires a successful bound review and drains already-started jobs if one fails. The existing production executable and user trajectories are not modified.

No efficiency or physical-stability result is claimed until the resulting trajectories and convergence diagnostics have been evaluated.

## Frozen execution, 3 October 2026

The campaign completed all 96 chains under `results/evolving-dimer-campaign-20261003`, using at most three workers. The configuration hash is `c96c938fe33fb6ade9d2a984fbcd790daa0885ffdd496d9cd21ebb7bebb62054`; the run-binding hash is `eedbf181270ef96b9991c08038e12fd205ee2496ea6f4ea20972327dc27cfaed`. `dispatch/status.json` records the complete, passed allocation. Do not dispatch this allocation again.

Validation includes 17 Rust tests, 24 preparer/observer tests, nine independent-auditor tests, and 1,560 independent checks of synthetic sphere trajectories. The actual protein preparation audit checks 32 clouds (524,288 raw points), all 16 selected starts and all 21 attempted preparations, including five nulls. All 11,792 checks pass; maximum reconstructed complete-density correction error is 3.43e−13.

The original preparation auditor incorrectly assumed an unpadded atomic AABB. Its failed receipt is preserved. The corrected auditor reconstructs the existing explicit conservative padding; it changes neither the sampled cloud nor strict sphere membership. The v2 audit uses the same saved data without redraws. The first sandbox background launcher exited before claiming work; an authoritative process check confirmed no controller or sampler, after which the identical frozen controller was launched on the host. Both launch records remain preserved and no scientific attempt was repeated.

## Completed comparison

The frozen observer completed all 96 chains and 442,464 initial/retained endpoints
without adding sampling attempts. Its terminal receipt is
`results/evolving-dimer-analysis-final-20261003/summary.json`; the complete analysis
SHA256 is `46ced1e7cd2e4e4c928e4eff76a146d87fc92267d4d02c58479dae1aa3db85d2`.
Sampler CPU totals 26,799.7 seconds; setup-inclusive sampling CPU is 26,841.3
seconds and observer CPU is 529.1 seconds. The analysis preserves all attempted
moves and rejected residence. Different conditional targets are not pooled.

![Per-stream contact efficiency and initialization agreement](../results/evolving-dimer-report-final-20261003/contact-efficiency.png)

The m=4 overlap threshold generally reduces the overhead of unguided dimer
redraws, but does not establish equilibration. In both whole-dimer contexts,
from both starts, every matched stream improves whole-fingerprint apparent
ESS/CPU over unguided by 1.14–6.95 times. All finite paired comparisons still
favor local moves for this descriptor. Local can explore its current contact
region cheaply while never changing binding partners; its higher apparent ESS
does not imply greater environment coverage.

The embedded 9/24 source provides the clearest failure case. Neither collective
arm accepts a dimer move in any stream. All contact time-series statistics and
local acceptance counts match local exactly, while guided execution costs
24.1–32.4 times as much. All arms retain maximum patch- and partner-occupancy
differences of one between the source and prepared starts in every stream.
Those starts have not equilibrated.

One prepared 9/24 guided stream records 61 partner-set passages and 60 returns.
Its only observed pair is 9/24 itself: these events alternate between that same
pair being in contact and separated, with 10.18% contact occupancy. They are
attachment/detachment events, not exchanges with a different partner. For
embedded 11/246 from the source, guided moves produce one partner-set passage
per stream and no return. Threshold crossings and finite ESS therefore cannot
be treated as evidence of completed exchange between competing basins.

The next proposal should address constraints from **external contacts** while
preserving the same conditional target, rather than only increasing the
internal-pair threshold. Before choosing the update, scalar rejection records
can distinguish capped destination failure, proposal-density penalties and
depletion loss without generating new physical draws. Allowing spectators to
move would be a different benchmark and must be identified as such.

The plot is reproducible with `tools/plot_evolving_dimer_analysis.py` and its
source-bound receipt; `chains.csv` retains every stream separately. Native
registry was not measured. These are conditional sampling results at 1.4 Å and
0.0275 Å⁻³, not finite-system thermodynamics or evidence against assembly at the
original decision conditions.
