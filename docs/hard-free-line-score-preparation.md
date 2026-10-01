# Feasibility-only Gaussian line guide: frozen score comparison

This candidate keeps the original 92 Gaussian components and conditions only on geometric feasibility. It introduces no contact labels, contact widths or selected atomic pairs. The first diagnostic evaluates the same 206 saved poses; it draws no new poses or Poisson clouds and estimates no new physical mass.

At a selected raw translation axis, the other five raw coordinates retain each component's original Gaussian marginal. The scalar conditional Normal is restricted to the intersection of its hard-free intervals, the original R4 region and the original center-capture domain. If that interval set has Gaussian probability at most $10^{-12}$, the original conditional Normal is retained on the same outer coordinates. There is no outer-coordinate redraw or hard retry.

With component density $g_k(u)$, weight $p_k$, conditional feasible mass $Z_{ka}$ and query membership $I_a$, the complete axis density is

\[
q_a(u)=\tfrac12 U_{R4}(u)+\tfrac12\sum_k p_k g_k(u)c_{ka}(u),
\qquad
c_{ka}=\begin{cases}
1,&Z_{ka}\leq10^{-12},\\
\mathbf1_{I_a}(u)/Z_{ka},&Z_{ka}>10^{-12}.
\end{cases}
\]

The **xyz mixture is the predeclared primary comparison**, $q_{xyz}=(q_x+q_y+q_z)/3$. The three individual axes are secondary diagnostics. Every Gaussian, axis and fallback branch enters the denominator; no winning axis is selected after inspecting scores.

For valid poses in the physical integration domain, every axis satisfies $q_a\geq q_{\rm old92}$, as does the xyz mixture. Consequently the importance-estimator second moment cannot increase per attempted draw for a target supported on that feasible domain, provided the conditional estimator is unchanged. The diagnostic must measure the size of any reduction and its computational cost. This property does not establish faster MCMC mixing or converged physical integrals. With conditioning probability one, there is **no global $q\geq0.5q_{\rm old92}$ bound**: invalid poses may lose density. The 50% uniform R4 component preserves complete target-region support. Legitimate zero densities outside R4 must remain explicit null log densities, rather than causing rows to disappear.

## Fixed allocation and trace

The immutable package is `runs/hard-free-line-score-preparation-20261001`. Plan SHA-256: `7e7e1679bd666f39c7000b8d01c136f162e6caa18e74e4ff792d8a77d4b8dd1b`. Probe SHA-256: `55310d72d9a97a241e3e41c23cea007dd556d4db5ef6a793d78f326600a97967`, unchanged from the arc comparison.

- All 78 critical competing orthant 55 poses remain, with original source groups of 4 and 74 kept separate.
- All 128 breadth poses remain: 32 each native R5, native complement, competing and invalid.
- All 206 poses are inside the original R4 and capture domain; 174 are hard-valid and 32 hard-invalid.
- Only the xyz guide is scored geometrically: 206 pose queries, 618 axis requests and 56,856 component-axis branches. Individual-axis densities are reconstructed algebraically from the same trace, yielding 824 reported densities in total.
- Alpha is 0.5, conditioning probability is 1, and the conditional-mass floor is $10^{-12}$. The hard shape, original physical Jacobian and depletant parameters remain unchanged.

The diagnostic trace supplies the three complete axis log densities and, for each axis and component, the unweighted Gaussian log density, conditional mean and standard deviation, feasible mass, fallback decision and query membership. It also preserves interval sets and geometry CPU. These fields suffice to derive all four proposal densities without repeating geometric queries. An independent reference reconstructs the conditional laws and entire-line geometry before the result is interpreted.

`tools/prepare_hard_free_line_score.py` creates and verifies the frozen inputs. `tools/report_hard_free_line_score.py` binds the completed independent audit, reconstructs the four densities, checks old92/Jacobian/pose identity, and reports paired second moments separately by original source, all breadth classes and full-mixture CPU. `tools/run_hard_free_line_score.py` prepares a single xyz scoring job with zero fresh samples and an unused fixed seed of zero, then audits and reports without repeating the geometry job. It preserves all attempted query identities and drains its child on failure. Eight preparation/algebra/controller tests cover allocation, unchanged components, absence of contact restrictions, complete-axis averaging, empty/floor fallback, explicit zero density, the zero-draw command and failure handling; the receipt is `results/hard-free-line-score-controller-validation-20261001/validation.json`.

The existing poses have already informed development, so their scores are retrospective. No independent-population uncertainty, fresh-draw efficiency, physical convergence or finite-system assembly conclusion follows. Before any protein scoring, the exact executable and reference must pass production validation and a one-worker, no-retry controller must bind their sources and every attempted query. No protein score or physical campaign was launched during preparation.

## Validated execution package

The execution package is `runs/hard-free-line-score-20261001`, protocol SHA-256 `66e97c9c03b903956e2d5c6b33b7fc3a2f679c3e4367378d95dfa8e10d267124`. Archived receipts cover 16 Rust tests, the exact release build, 8 controller/preparation/report tests, and the independent reference's 8 tests plus six toy arms (768 toy draws and 66 fixed toy probes). The exact archived controller passed preflight. None of those reference draws were proteins.

An initial toy fixture had an inconsistent nested shape-hash binding and failed before any query. Its failed inputs and logs were preserved; a separate versioned fixture corrected the binding. The successful reference receipt is `results/hard-free-line-reference-validation-v2-20261001/validation.json`. This was a preparation error, not a failed geometric or physical test.

The exact one-shot command is recorded for provenance below. An existing launch is never repeated; check `status.json` first.

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  /vast/xvg/tetramer-mc-runs/hard-free-line-score-20261001/common/run_hard_free_line_score.py run \
  --out /vast/xvg/tetramer-mc-runs/hard-free-line-score-20261001 \
  --expected-protocol-sha256 66e97c9c03b903956e2d5c6b33b7fc3a2f679c3e4367378d95dfa8e10d267124
```
