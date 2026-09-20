# Static and evolving contact-memory controls

4 matched, genuinely dispersed preparations of 12 mobile tetramers; 1,000 sweeps each. Every arm uses the same geometry-only base atlas, physical hard/depletion target, local/GCA/shift schedule, RJ count prior, and conditional mean transport. The frozen and evolving banks start from exactly the same 16 relative poses. A frozen bank has zero pair updates; an evolving bank has 2 per sweep. Neither supplies native docking charts.

| Start | Arm | Largest registered / nonspecific component | Bank accepted | Global accepted (base/memory/uniform) | CPU s |
|---|---|---:|---:|---:|---:|
| 0 | memory-off | 1 / 2 | 0 | 0/0/112 | 42.75 |
| 0 | memory-frozen | 1 / 2 | 0 | 0/0/96 | 40.46 |
| 0 | memory-on | 1 / 2 | 1042 | 0/0/75 | 64.19 |
| 1 | memory-off | 1 / 1 | 0 | 0/0/131 | 30.67 |
| 1 | memory-frozen | 1 / 2 | 0 | 0/0/110 | 36.17 |
| 1 | memory-on | 1 / 4 | 816 | 0/0/109 | 50.72 |
| 2 | memory-off | 1 / 3 | 0 | 0/0/104 | 31.07 |
| 2 | memory-frozen | 1 / 2 | 0 | 0/0/108 | 29.80 |
| 2 | memory-on | 1 / 3 | 880 | 0/0/95 | 43.02 |
| 3 | memory-off | 1 / 4 | 0 | 0/0/128 | 30.77 |
| 3 | memory-frozen | 1 / 3 | 0 | 0/0/128 | 30.37 |
| 3 | memory-on | 1 / 2 | 826 | 0/0/96 | 50.02 |

## Interpretation

Native registration and bank-native discovery remain separate observables. Physical non-native contact changes establish movement between neighborhoods, not discovery of a native pose. Global acceptance, when dominated by uniform draws, cannot be credited to the learned memory charts. Total CPU includes the bank work. These four correlated, matched experiments do not establish equilibrium or a mixing-time advantage.

| Arm | Total CPU s | Bank update CPU s | Global accepted / CPU s | Native formations / breakages | Bank-native slots ever |
|---|---:|---:|---:|---:|---:|
| memory-off | 135.26 | 0.00 | 3.512 | 0/0 | 0 |
| memory-frozen | 136.81 | 0.03 | 3.231 | 0/0 | 0 |
| memory-on | 207.96 | 33.02 | 1.803 | 0/0 | 0 |

## Proposal diagnostic

The table separates geometric rejection from acceptance after the exact conditional Poisson gate. Expected acceptance counts are the sum of final acceptance probabilities over sampled auxiliary clouds. They are descriptive Monte Carlo diagnostics, not independent free-energy measurements; actual accept/reject decisions are reported separately.

| Arm | Source | Proposed | Hard valid | Accepted |
|---|---|---:|---:|---:|
| memory-off | base | 21537 | 9631 | 0 |
| memory-off | memory | 0 | 0 | 0 |
| memory-off | uniform | 2357 | 569 | 475 |
| memory-frozen | base | 10990 | 4887 | 0 |
| memory-frozen | memory | 10547 | 4420 | 0 |
| memory-frozen | uniform | 2357 | 568 | 442 |
| memory-on | base | 10990 | 4821 | 0 |
| memory-on | memory | 10547 | 4906 | 0 |
| memory-on | uniform | 2357 | 572 | 375 |

For memory-chart draws that pass the hard test:

| Start | Arm | Hard valid / proposed | Median log(q reverse/q forward) | Median final log acceptance | Expected accepted draws |
|---|---|---:|---:|---:|---:|
| 0 | memory-frozen | 1056/2722 | -25.46 | -25.24 | 2.37e-06 |
| 0 | memory-on | 1362/2722 | -25.35 | -26.62 | 1.05e-06 |
| 1 | memory-frozen | 1125/2621 | -25.50 | -25.15 | 3.31e-06 |
| 1 | memory-on | 1105/2621 | -25.44 | -25.18 | 5.63e-06 |
| 2 | memory-frozen | 1163/2637 | -25.49 | -24.27 | 2.44e-05 |
| 2 | memory-on | 1293/2637 | -25.48 | -25.57 | 2.62e-06 |
| 3 | memory-frozen | 1076/2567 | -25.36 | -24.14 | 6.95e-06 |
| 3 | memory-on | 1146/2567 | -25.37 | -25.69 | 3.62e-06 |

## Independent verification

The contributing audits reconstruct 224,000 logged updates to 1,212 frames/checkpoints, including physical, RJ and contact-bank moves. Every saved physical state and every retained bank pose is checked independently for hard geometry; the physical wall is checked atom by atom. Native templates are post-hoc analysis only.

The baseline/evolving campaign and later frozen control archive separate executable hashes because the frozen control required allowing a zero bank-update budget. The initial bank equality and all other resolved parameters are explicitly checked. Raw CPU differences should not be interpreted as a proven speedup.

- Campaign /home/xvg/tetramer-mc/runs/contact-memory-free-1000: binary `9ea570c1918321f90781856a0e4dd7d484dd19ce6b1542187b081d3c7a9e50e5`
- Campaign /home/xvg/tetramer-mc/runs/contact-memory-frozen-1000: binary `c54e0de8f0ad6149f75ae84bbbf7a299c146df4e28e05006d7d1f799ab92dd1d`
