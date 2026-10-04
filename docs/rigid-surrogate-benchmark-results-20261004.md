# Completed rigid-pair comparison

The fixed eight-step rigid moves revisit occupied external contact environments
far more often than the earlier proposal control in this conditional test. The
matched unguided eight-step arm performs similarly overall, so this comparison
does not establish a benefit from the overlap guidance itself. It motivates
testing internal flexibility, rather than treating guided rigid motion as a
solution to assembly.

All 24 new chains and 16 unique cached controls completed. The analysis retained
every rejected state and 4,096 production blocks per chain, after 512 warmup
blocks. Only labels 27 and 132 move; 262 spectator tetramers remain fixed.
Parameters are `rd=1.4 Å`, `z=0.0275 Å^-3`, with the inherited growth-state wall
and concentration. These are not the decision conditions `1.5 Å`, `0.035 Å^-3`,
and approximately 106.8 μM.

## Observed contact motion

Each bracket lists streams 0–3 separately. A return means revisiting a previously
exited nonempty external edge set; it need not exchange every partner or identify
a binding basin. CPU includes warmup and rejected work. These counts are
descriptive, correlated trajectory events, not physical rates or independent
equilibrium samples.

| Move arm | Returns: source start | Returns: prepared start | Full CPU, all eight chains (s) |
|---|---|---|---:|
| Local | [0, 0, 0, 0] | [0, 0, 0, 0] | 728.4 |
| Earlier `m4` proposal | [0, 0, 1, 0] | [0, 0, 0, 1] | 1694.5 |
| Guided rigid, one step | [0, 44, 4, 0] | [0, 0, 0, 0] | 1244.0 |
| Guided rigid, eight steps | [0, 1, 43, 30] | [9, 14, 3, 0] | 1470.6 |
| Unguided rigid, eight steps | [1, 25, 11, 7] | [3, 48, 2, 0] | 1472.1 |

The guided and unguided eight-step totals are 100 and 97, respectively. All 24
direct nonempty external-set changes in the guided arm occur in source stream 2;
the unguided arm has two such changes, also in source stream 2. The guided stream
visits tetramer 132 in contact with 179, with 228, and with both. Production
fractions for those three sets are 9.155%, 0.562%, and 0.952%. This is useful
contact accessibility, but its concentration in one stream prevents a general
mixing claim. The observation does not classify native registry.

## Efficiency and convergence limits

Guided/unguided eight-step whole-contact-fingerprint ESS per CPU ratios range
from 0.625 to 1.064 across matched starts and streams; guidance is higher in only
two of eight comparisons. The controls are correlated through shared random
number roles. These ratios are descriptive, not confidence intervals. There is
also no consistent advantage over local-only fingerprint ESS per CPU.

The exceptionally high external-edge-presence ESS/CPU (31.12) in guided source
stream 0 comes from exactly **one occupied production frame out of 4,096**.
It cannot be interpreted as 31 independent contact samples per CPU second.
Whole fingerprints and edge-presence indicators must be read alongside actual
occupancies and returns; none of these finite-record diagnostics certifies
independent basins.

The guided eight-step between-start external-environment total-variation
differences range from 0.013 to 0.182. Local-only differences are zero because
both starts remain externally unbound throughout production. The earlier `m4`
arm has differences of one for two matched stream pairs. Small differences
dominated by the empty environment do not establish convergence of contact
weights or of internal relative geometry.

The next fixed comparison therefore retains both eight-step arms and adds
independent inner motion of the two tetramers. Its principal new measurements
are internal patch-set rearrangements and residence, alongside external contact
exchange and initialization agreement. The corrected flexible kernel has passed
its independent physical reference; its protein efficiency is unmeasured.

## Evidence

- Completed analysis:
  `/vast/xvg/tetramer-mc-runs/rigid-surrogate-dimer-analysis-20261004/analysis/analysis.json`,
  SHA256 `bd1640d6689378c5c3ddf058fb91fad0d473f9e5169c526d840b0f69582faeef`.
- Successful drained controller:
  `execution/summary.json` under that root, SHA256
  `2a06f9895db44e3be38ed029de04d676613b5713c38f02cc372c6b2d7f3f2315`.
- New analysis CPU: 572.3 seconds; 110,616 retained endpoints. Cached control
  geometry was reused, with zero old geometry queries and zero new physical
  draws in the analysis.
- Plots and copied numeric values:
  `results/rigid-surrogate-contact-figure-20261004-v2/`. The first rendering is
  retained separately; v2 fixes overlapping undefined-value labels and uses a
  logarithmic axis for external-edge ESS. Numeric values are unchanged.

![Per-stream contact efficiency](../results/rigid-surrogate-contact-figure-20261004-v2/contact-efficiency.png)

![Initialization agreement](../results/rigid-surrogate-contact-figure-20261004-v2/initialization-agreement.png)

Neither finite-system native stability nor instability follows from this
conditional comparison. The independent physical-weight campaign remains open.
