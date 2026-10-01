# Hard-free arc guide: fixed saved-pose comparison

This stage evaluates the complete proposal density on the **same 206 saved poses** used by the completed two-distance diagnostic. It draws no new pose or depletant cloud and estimates no new physical contact weight. The original physical convergence requirements remain unmet.

The allocation is fixed before scoring: all 78 critical competing orthant 55 poses, followed by all 128 breadth poses (32 each native R5, native complement, competing and invalid). The original critical sources remain separate: 4 poses from the original 84-component source and 74 from the expanded 92-component source. These probes have already informed development; their comparison is retrospective.

Both unchanged frozen two-distance guides are scored, in order `uniform_phi92`, then `localized_phi92`. Geometry is shared across arms: 206 unique pose queries produce 412 complete candidate densities. Every one of the 92 components and all three equally weighted widths 0.02, 0.1, 0.5 Å enters each mixture density. The 50% uniform defensive component, 50% conditioning probability within the Gaussian component, atomic labels, region, shape and scaffold stay fixed.

The candidate changes only the conditional azimuth law: condition on whole-union hard-free arcs when their mass under that arm's azimuth law exceeds the fixed threshold 1e−12; otherwise retain the original azimuth distribution at the same orientation and two contact distances. The floor is applied separately for each component/width/arm. There is no hard retry or redraw of the outer variables. The full density includes all branch probabilities and fallback contributions. R4 and capture support are unchanged and are not imposed along the arcs.

`tools/prepare_contact_arc_score.py` verifies the old completed execution, independent audit receipts, all pose identities, guide hashes and saved density agreement. It archives the original score files and builds a compact table containing old 92-component and old distance densities, saved geometric flags, paired cloud terms and unconditional source denominators. No old physical row is reclassified. The exact executable and independent reference must pass production validation and be bound to the controller before scoring.

Reports must retain all four breadth classes, report full-density CPU and shared geometry cost, and compare critical paired second moments separately by original source. A reduced moment based on 4 or 74 previously selected critical rows is diagnostic; it cannot establish physical convergence, independent contact samples per CPU, assembly stability or absence of unobserved basins. A score-only benefit would justify a later fixed passive sampling test, not an immediate physical campaign.

The immutable preparation is `runs/contact-arc-score-preparation-20261001`: plan SHA-256 `7c45d2b63cd371c21b751d5189d077b4ca2068e91f94d09abccd7412e4647244`; combined probe SHA-256 `55310d72d9a97a241e3e41c23cea007dd556d4db5ef6a793d78f326600a97967`. It binds 42 source files. The score controller preserves all attempted query identities and drains its child on failure. Four preparation and four controller/moment tests pass.

Preparation command (creates a fresh immutable directory):

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_contact_arc_score.py \
  --preparation /vast/xvg/tetramer-mc-runs/contact-distance-passive-preparation-20261001 \
  --passive /vast/xvg/tetramer-mc-runs/contact-distance-passive-20261001 \
  --out /vast/xvg/tetramer-mc-runs/contact-arc-score-preparation-20261001
```

One CPU, no retries, no optional stopping and no automatic extension are allowed. Existing calculations and the complete physical target remain unchanged.

The reviewed execution package is `runs/contact-arc-score-20261001`, with protocol SHA-256 `2a05ad8a092515943482d9cbc56f8f890e16707c87af3f42b43801446d73d94b`. It archives the exact release binary, embedded Rust source, Python import closure, and three validation receipts: 6 Rust tests, 13 independent reference tests plus 60 synthetic cross-language queries, and 8 preparation/controller checks. Its two jobs run sequentially: Rust full-density scoring, then independent Python reconstruction of the same queries. The controller compares completed records and produces the retrospective summary; it cannot generate new poses or depletants.

The exact one-shot launch command is recorded below for provenance. Check `status.json` first: an existing launch is never repeated.

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  /vast/xvg/tetramer-mc-runs/contact-arc-score-20261001/common/run_contact_arc_score.py run \
  --out /vast/xvg/tetramer-mc-runs/contact-arc-score-20261001 \
  --expected-protocol-sha256 2a05ad8a092515943482d9cbc56f8f890e16707c87af3f42b43801446d73d94b
```
