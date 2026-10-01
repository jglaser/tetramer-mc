# Feasibility-only line conditioning: saved-score result

**This candidate warrants a bounded fresh validation.** Preserving the original Gaussian measure while conditioning on feasible translation intervals improves the inspected critical-contact scores at substantially lower cost than the earlier full arc-mixture calculation. These retrospective diagnostics do not establish physical convergence or native assembly.

![Feasibility-only line scores, breadth controls and computational cost](../results/hard-free-line-score-review-20261001/hard-free-line-score.png)

The [frozen comparison](hard-free-line-score-preparation.md) evaluated the xyz mixture on all 206 existing poses and derived the individual x/y/z laws from the same saved component terms. All 78 critical competing orthant 55 poses and all 128 breadth poses were retained. No fresh protein pose, depletant cloud or physical mass estimate was generated. Shape, scaffold, original Gaussian components, physical Jacobian and target domain remained unchanged.

## Critical scores and coverage

Paired second-moment ratios relative to old92 are shown below. The original source groups remain separate, with their paired cloud terms and unconditional attempted-draw denominators preserved. The xyz mixture was declared primary before scoring.

| Original source | Rows | x | y | z | xyz, primary |
|---|---:|---:|---:|---:|---:|
| 84-component source | 4 | 0.804 | 0.884 | 0.885 | **0.8325** |
| 92-component source | 74 | 0.521 | 0.592 | 0.464 | **0.5031** |

The primary law's contribution ESS is only **1.70 and 2.54**; the largest terms contribute **74.4% and 45.9%**. The reduction on these selected rows is encouraging, but those concentration measures prevent reliable uncertainty or campaign-size predictions. They are not independent contact samples or population-level convergence diagnostics. The favorable sign of the change is a mathematical consequence of $q_{\rm new}\geq q_{\rm old92}$ on feasible target poses; its magnitude and computational cost are the relevant measurements here.

All 174 feasible saved poses preserved or increased proposal density. Every breadth class retained all 32 rows. For the primary xyz law, median log-density changes relative to old92 were **+0.536** in native R5, **+0.525** in the native complement, and approximately zero in competing and invalid controls. Some invalid poses lost density substantially; the largest reduction was −7.365 in log density. This is permissible, and the uniform R4 component retained nonzero density for every saved control. Proposal density changes are not equilibrium occupancy changes or evidence that unseen regions are covered.

## Cost and independent validation

The entire xyz density calculation took **0.750 CPU seconds**, or **3.642 ms per query**. Exact line geometry accounted for **0.658 seconds** across all three axes. Whole-probe processing, including additional checks and output, took 0.884 seconds. The x/y/z secondary densities were derived algebraically without repeating geometric queries. These measurements concern complete proposal-density evaluation; they do not measure fresh proposal survival or sampling speed.

The independent Python reconstruction checked all 206 poses, all 618 axis cases and all 56,856 component-axis branches. Maximum full log-density discrepancy was $1.92\times10^{-12}$; maximum interval-endpoint discrepancy was $2.69\times10^{-13}$ Å. The original Jacobians agreed exactly. Reconstructing xyz as the arithmetic mean of the three complete axis densities agreed to $7.11\times10^{-15}$ in log density. The independent audit used 23.58 CPU seconds, separate from production scoring cost.

The reference work included 16 Rust tests, 8 independent reference tests, six corrected toy arms with 768 toy draws and 66 fixed probes, and 8 preparation/controller/report checks. The initial toy fixture's pre-query shape-hash failure remains archived separately; no protein calculation was retried.

## Next decision

A bounded fresh proposal and weight validation is now justified, with independent streams, the predeclared xyz law, complete attempted-draw bookkeeping, retained invalid zeros and the original physical convergence gates. It must measure actual proposal survival, total CPU including depletant evaluation, and physical weight concentration. The individual-axis diagnostics do not authorize selecting a favorable axis and treating it as independently validated.

No regional weight gate has passed, and no new evidence decides finite-system stability. This result supports continuing the sampling calculation; it does not support either a native-assembly or a model-instability conclusion.

Completed execution: `runs/hard-free-line-score-20261001`. Protocol SHA-256: `66e97c9c03b903956e2d5c6b33b7fc3a2f679c3e4367378d95dfa8e10d267124`. Analysis SHA-256: `557ab9fc1592111810302f677c46be85f2681a034faa1de24145c830ba7fdd24`. Independent audit SHA-256: `47dcfc173dc302e2a8569af02e4ce8d41837edc17bee717ad0340be7f7ccbb76`. The [figure archive](../results/hard-free-line-score-review-20261001/summary.json) binds 34 source files; its summary SHA-256 is `a21966f1473924632b92bf7e8bd78fda460ecbd07e09c9b3553686f2b9f79020`.
