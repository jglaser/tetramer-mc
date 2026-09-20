# Involutive docking pilot

The controlled atlas restores native access, but **none of the five methods
completes a native–competing–native contact roundtrip in 5,000 cycles**. The
correlated map at γ=0.9 accepts substantially more changed poses per CPU, mostly
within the same broad contact environment. This pilot does not demonstrate
faster equilibrium exchange between the competing basins.

The experiment moves one rigid tetramer near one fixed tetramer. The moving
center lies in an 18 Å capture ball, with all proper orientations allowed. The
ideal depletant radius is 1.5 Å, activity 0.035 Å⁻³, and Poisson oversampling
ratio 16. Every cycle has two local attempts and one global attempt. The local
translation steps [0.2, 2] Å, rotation steps [1.5°, 15°], and rotation probability
0.5 are identical in every arm. The separate uniform cube/Haar branch has
probability 0.1. There is no learning, mask update, or cluster move in this test.
The [balance argument and runner](involution-docking.md) describe the exact
physical target and composition with the Poisson gate.

## Coverage control and starting configurations

The initial 28-component assembly atlas gave zero nonidentity global accepts
in a 20-run, 200-cycle smoke test. The selected SMC endpoints were supported
mostly by its broad defensive components. Exact native reference poses do have
narrow atlas support, so this result is a mismatch between the supplied atlas
and the sampled conditional basins, rather than evidence that native poses
have no proposal support.

The main control instead retains the existing **five-component conditional
atlas** from `protein-fit/frozen-pose-mixture.json` in the earlier
`basin-discovery-pilot`: four fitted components plus its 2% broad component.
Its shape and site0 environment hashes exactly match this target. No component
is refitted or pruned. To express the old laboratory-frame model relative to
the fixed neighbor `(t_f,R_f)`, the launcher applies

\[
 a_t'=R_f^T(a_t-t_f),\quad a_R'=R_f^T a_R,\quad
 \mu'=B\mu,\quad\Sigma'=B\Sigma B^T,
 \qquad B=\operatorname{diag}(R_f^T,R_f^T).
\]

The Cayley vector transforms by the same proper rotation. Ten fresh poses,
two from every component, verify the complete translation/Haar proposal
density before and after conversion; the maximum log-density difference is
4.98×10⁻¹⁴.

| Start | Original source | MC seed | Initial native error q | Atlas log density | Static source probability ‖z‖≤6 |
|---|---|---:|---:|---:|---:|
| Native0 | narrow-water site0 native SMC00 | 1849372091 | 0.542 | 12.618 | 0.283 |
| Native1 | narrow-water site0 native SMC01 | 1849372092 | 0.965 | 13.781 | 0.283 |
| Competing0 | heldout-r2-f0.5 | 1849372099 | 29.723 | 5.658 | 0.565 |
| Competing1 | heldout-r3-f0.5 | 1849372100 | 29.548 | 13.369 | 0.737 |

These are selected SMC endpoints and previously inspected conditional-basin
snapshots, not equilibrium samples or a blind generalization test. Each start
and seed is reused in all five modes. Both source files and the original and
converted atlas are archived with hashes. A 200-cycle coverage smoke passed
before the 20 runs of 5,000 cycles, using four workers.

## Contact exchange

For each supplied native reference, the registration error is the maximum of
corresponding tetramer-member displacements divided by 2 Å and proper angular
error divided by 15°. The native error q is the minimum over those references.
Primary cores are q≤0.8 and q≥2; the stricter competing core is q≥5. Both
endpoints must have an exact overlap of the inflated atomic sphere unions.
Unbound transit is permitted between visits but is reported separately;
all saved frames in this pilot remain bound. Two successive core passages
returning to the first visited core make one nonoverlapping roundtrip. Every
attempt is replayed, including events between saved cycle frames.

| Method | Strict competing→native entries, out of two starts | Strict native→competing exits | Completed roundtrips | Useful global changes | Changed source/target labels | Total CPU s |
|---|---:|---:|---:|---:|---:|---:|
| Full-mixture redraw | 2 | 0 | 0 | 82 | n/a | 405.07 |
| Involution γ=0 | 1 | 0 | 0 | 16 | 2 | 405.82 |
| Involution γ=0.5 | 2 | 0 | 0 | 51 | 2 | 222.88 |
| Involution γ=0.9 | 2 | 0 | 0 | 205 | 8 | 199.05 |
| Involution γ=1 | 1 | 0 | 0 | 22 | 22 | 210.26 |

The primary q≥2 criterion additionally counts two exits for γ=0.9, one from
each native start. They first reach q=2.013 and 2.051, through a local move and
an involution respectively, and never reach q≥5. These small registration
excursions do not establish exchange with the original competing configurations
near q=30. No primary or strict roundtrip is completed in any trajectory.

The full-mixture control first reaches the native core from the two competing
starts at cycles 3 and 29; γ=0.5 does so at 51 and 132. These are individual
first-entry observations, not estimates of a mean passage time. Some entries
finish through a local move after a preceding global jump lands outside the
q≤0.8 core.

For γ=0.9, accepted nonidentity globals per CPU rise from 0.202 to 1.030,
about 5.1× the full-mixture rate. Of its 205 accepted changed poses, 197 use
equal source and destination chart labels; labels alone do not classify a
physical basin. The γ=1 arm also has 4,515 accepted exact identities, which
are excluded from useful changes. A faster accepted-pose rate is not a
roundtrip-speedup claim.

![Controlled conditional docking comparison](../runs/involution-docking-conditional-5000/assessment/involution-docking.png)

The [SVG](../runs/involution-docking-conditional-5000/assessment/involution-docking.svg)
and [PDF](../runs/involution-docking-conditional-5000/assessment/involution-docking.pdf)
are standalone exports. The figure's primary-core exits are qualified by the
stricter diagnostic above.

## Why direct reverse proposals fail

Hard-valid proposals from the bound native core to q≥5 are present in every
arm. None is accepted:

| Method | Hard/capture-valid direct N→O candidates | Median proposal log ratio | Median sampled Poisson log factor |
|---|---:|---:|---:|
| Full mixture | 1,526 | −0.76 | −13.16 |
| γ=0 | 763 | −6,988.98 | −12.19 |
| γ=0.5 | 605 | −2.56 | −14.31 |
| γ=0.9 | 605 | −1.80 | −12.43 |
| γ=1 | 931 | −2.23 | −12.49 |

The candidate classification uses q≥5 as a competing-pose proxy: depletion
contact is not recomputed for every rejected candidate. Observed accepted
core passages always require contact. These medians condition on valid
geometry and an evaluated gate; they are **not free-energy differences**.
The [diagnostic data](../runs/involution-docking-conditional-5000/assessment/core-candidate-diagnostics.json)
also contain both directions, q≥2 controls, and 10/50/90% quantiles.

At γ=0, source labels are still sampled from the static reference weights.
Selecting a chart that assigns negligible density to the old pose adds a
large negative selected-component correction. The full-mixture redraw sums
over chart labels in its acceptance density and avoids that particular
penalty, although the aggregate candidate law is the same. Correlation reduces
this problem for many valid proposals, but the sampled physical gate remains
strongly unfavorable for the direct reverse candidates that were drawn.

The runtime difference mostly comes from the depletion gate: its CPU falls
from 401.46 s in the full-mixture arm to 196.20 s for γ=0.9. Proposal evaluation
itself costs only 0.085–0.098 s per arm. Different geometry rejection rates and
retained trajectories change how many expensive gates are evaluated; the
result is not an acceleration of Gaussian arithmetic.

## Reproduction and validation

```bash
cargo build --locked --release --bin docking-mc
/home/xvg/protein-nucleation/.venv/bin/python tools/run_involution_docking_campaign.py \
  --out runs/involution-docking-conditional-5000 \
  --sites site0-m1 --replicates 2 --cycles 5000 --workers 4 \
  --conditional-atlas /home/xvg/protein-nucleation/results/hierarchical-pose/cayley-line/basin-discovery-pilot/protein-fit/frozen-pose-mixture.json \
  --covered-starts /home/xvg/protein-nucleation/results/hierarchical-pose/cayley-line/basin-discovery-pilot/protein-campaign/inputs
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_involution_docking_campaign.py \
  --campaign runs/involution-docking-conditional-5000 --workers 4
```

Use a new output directory when rerunning. All 300,000 physical attempts replay
exactly to their saved configurations and checkpoints. The independent audit
checks 72,312 involutive maps, selected and full Gaussian densities, Haar
Jacobians, 60,000 full-mixture density values/ratios, and 52,265 acceptance
decompositions. The maximum full Gaussian log-density discrepancy is
5.48×10⁻⁸; mapped physical poses agree to 2.85×10⁻¹⁴ Å and 8.04×10⁻¹⁴ in
rotation-matrix entries. All strict conditional-atlas map audits pass without
warnings. Selected frames independently pass atom-union hard/contact checks;
a geometric clearance bound excludes remote periodic-image interactions
throughout the capture ball. The runner's 85 release tests include independent
depletion stationarity, missing-Jacobian controls, and restart checks.

The original 28-component smoke retains four rejected, hard-invalid extreme
tail traces whose reconstructed source rotations exceeded the strict audit
tolerance. They never moved the physical state. Its report records those
warnings explicitly and makes no blanket numerical-map validation claim.

The [full report](../runs/involution-docking-conditional-5000/assessment/report.md)
and [complete audit](../runs/involution-docking-conditional-5000/assessment/analysis.json)
preserve per-start results, contact visits, candidate corrections, source
labels, runtime, and provenance. The result separates improved motion within
represented basins from the still unresolved problem of repeated physical
exchange. It cannot determine equilibrium basin weights or conclude that
the protein model prevents crystallization.
