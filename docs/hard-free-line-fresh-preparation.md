# Fresh feasibility-line proposal pilot

The favorable [saved-pose score](hard-free-line-score-results.md) now has a fixed fresh-proposal check. It compares the original 92-component Gaussian guide (`beta=0`) with exact hard-free translation-line conditioning (`beta=1`, equal x/y/z selection). Both retain the same 50% uniform R4 component, Gaussian parameters, repaired shape, two-tetramer scaffold, R4 region and capture sphere. Depletant parameters remain 1.5 Å and 0.035 Å⁻³, but **this pilot samples no Poisson clouds and estimates no physical weights**.

Allocation was frozen before generating any new poses:

| Arm | Independent populations | Draws per population | Total |
|---|---:|---:|---:|
| Original Gaussian92 | 4 | 256 | 1,024 |
| Feasibility-conditioned xyz | 4 | 256 | 1,024 |

There are no saved-pose probes, retries, optional stopping or automatic extensions. The controller runs one child at a time: generate a population, independently reconstruct all its densities and geometry, then classify its valid new poses. An error drains the active child and prevents later launches. Every attempted draw and partial output remains recorded.

The native observer uses the complete frozen definition from the earlier contact-distance diagnostic, SHA `5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9`. Each hard-valid pose inside R4 and capture is classified exactly once for native registration and exactly once for physical exclusion contact. The latter uses the full sphere-union minimum surface gap below `2 rd = 3 Å`, retaining the gap to each scaffold separately. Invalid and exterior attempts remain in the denominator with null contact/native labels. The old Rust `width_contacts` sentinel is never interpreted as physical contact. Native labels are observers and never alter proposals.

The report separates uniform, original-Gaussian, conditioned-success and conditioned-fallback branches, selected axes, fallback reasons, hard/domain retention, native/contact fractions, both-scaffold contacts and cycle-consistent native triangles. Fractions use all 256 attempts in each population; uncertainty comes from the four independent population fractions. Proposal generation, density evaluation, full proposal-loop CPU, independent audit and classifier CPU are reported separately. No event-per-CPU statistic here measures equilibrium mixing, accepted transitions or physical contact weights.

Preparation and execution archives:

- `/vast/xvg/tetramer-mc-runs/hard-free-line-fresh-preparation-20261001`; plan SHA `78ab51fb40262b0197191a72542eec40d1cbfd4149c25c2bd72c258df0d3a1be`.
- `/vast/xvg/tetramer-mc-runs/hard-free-line-fresh-20261001`; protocol SHA `fdd202123ad34bcdf67f5d6623a9e20513a11fee715e340ff403f66f4f40144b`.
- The existing completed score executable was copied, SHA `937a8f999759eae16db0541344f632d461ee92b251c682a4d617907d56d8736d`, with its exact embedded source bundle. Current production-source edits are excluded.
- The archived two-file independent reference closure is unchanged. Frozen receipts include 16 Rust reference tests, eight independent Python tests and six cross-language toy arms, plus eight new controller/bookkeeping tests in [the fresh validation receipt](../results/hard-free-line-fresh-controller-validation-20261001/validation.json). The earlier reference fixture's pre-query shape-hash failure remains preserved; no failed protein query is being replaced.
- A single metadata inventory examined 26,958 declaration files across five `/vast` campaign roots, finding 7,219 previously declared distinct seeds. None matched the eight predeclared new seeds. The inventory and every inspected declaration hash are archived; it is not repeated during execution.

The frozen-controller preflight passed without generating a pose. The one-shot launch command is:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B -E \
  /vast/xvg/tetramer-mc-runs/hard-free-line-fresh-20261001/common/run_hard_free_line_fresh.py run \
  --out /vast/xvg/tetramer-mc-runs/hard-free-line-fresh-20261001 \
  --expected-protocol-sha256 fdd202123ad34bcdf67f5d6623a9e20513a11fee715e340ff403f66f4f40144b
```

Completion requires all eight populations, all independent audits and all once-only classifier receipts. This check can justify a subsequent bounded physical-weight comparison; it cannot pass the original thermodynamic convergence gates or decide assembly stability.
