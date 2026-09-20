# Independent larger-sample repeat of the frozen contact atlas

The [expanded atlas](expanded-contact-atlas.md) gave compatible complete
intermediate-window estimates across two widths, but both estimates rose
with sample count. This repeat changes only independent random streams and
the fixed allocation. It tests that sensitivity without adding centers,
refitting covariances or changing the physical integration domain.

The target remains both fixed AB neighbors, depletant radius 1.5 Å,
activity 0.035 Å⁻³, capture radius 18 Å, normalized proper SO(3) Haar measure
and **original 2 ≤ q < 5**. The old and new radius-two neighborhoods and
their common complement retain their previous definitions.

## Unchanged models, larger independent allocation

Both 53-component model files are byte-identical to the previous campaign:

| Arm | Model SHA-256 |
| --- | --- |
| Narrow | `ceb8ce76f2f3574203b5ff247c03971d992d117ed323e1b36152773518865ef3` |
| Broad | `9f31798c711a5928b4d3e35f1df29b9797fbd2a563055f12517a0a1ce34ce93f` |

The original model paths are supplied to the runner, preserving their
embedded link to the original training protocol. The repeat preparation
also archives identical copies for inspection. The complete product-cover,
Gaussian-atlas and uniform-cube mixture, full-density denominator and
two-cloud λ/z=64 estimator are unchanged. Only the archived shape path in
the new configuration is relocated; its bytes and physical parameters match.

Each arm now has **16 independent populations × 65,536 draws**, four
times its previous total N. Both arms together request **2,097,152
unconditional draws**. The seed bases are 103101010 and 103201010, with
increments of 1009. Original and repeated population seeds are disjoint.
The two arm runners use 16 workers each.

The protocol fixes three prefixes per population:

| Prefix per population | Total per arm | Relative to the previous complete arm |
| --- | ---: | ---: |
| 16,384 | 262,144 | 1× |
| 32,768 | 524,288 | 2× |
| 65,536 | 1,048,576 | 4× |

Thus the first fresh prefix gives an equal-total-budget check using new
draws, while the complete repeat tests a larger allocation. Prefixes within
the repeat remain correlated. Their errors use the shared-row covariance;
they are not additional independent replications. Original and repeat
campaigns remain separate during validation, and independent-difference
variances are used only after checking model identity, physical masks,
input hashes and disjoint streams.

All invalid and off-region draws stay zero in their original unconditional
denominators. No population is restarted, discarded or replaced according
to its observed physical weight. The protocol specifies no adaptive stop
or numerical pass/fail threshold. Comparisons must consider both widths,
population spread, maximum contribution, fixed-region calibration and the
directly sampled outside-both contribution. Observed small errors alone do
not establish absence of unseen important poses.

## Completed independent repeat

All 32 physical populations completed successfully. The audit reconstructs
all **2,097,152** original densities, q values, cloud factors and unconditional
denominators. The sixteen retained highest-weight poses pass independent
atom-union checks; the smallest checked gap is 0.004027 Å.

| Proposal | Original log Q / row SE | Repeat log Q | Repeat row / population SE | Repeat weight ESS | Largest repeat draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Narrow | 16.781336 / 6.33% | **16.699758** | **3.46% / 2.40%** | 834.60 | 1.62% |
| Broad | 16.893890 / 13.47% | **16.775340** | **7.33% / 6.60%** | 186.15 | 4.39% |

The earlier upward sample-count trend does **not** recur in this independent
repeat. Narrow and broad means are respectively 7.83% and 11.18% below their
original estimates, differences of −1.11 and −0.75 combined observed row
standard errors on linear Q. The two repeat widths differ by 7.85%, or
0.91 combined observed SE. These comparisons retain each campaign's
unconditional estimator; no samples or proposal arms are pooled.

Observed variance multiplied by N changes by factors 1.016 and 0.933 from
original to repeat, consistent with ordinary inverse-N variance reduction
over these tested allocations. This is an observed scaling diagnostic, not
a guarantee against missed tails. Sampler CPU is 8,477.89 s narrow and
6,691.82 s broad. Narrow again has lower variance × CPU, by a factor of
**4.12**, excluding discovery, fitting and audit costs. Paired-cloud noise
accounts for about 25.4% and 35.7% of observed full-estimator variance;
these cloud pairs share poses and are not independent pose samples.

The fixed repeat prefixes give:

| Total unconditional draws per arm | Narrow log Q / row SE | Broad log Q / row SE |
| --- | ---: | ---: |
| 262,144 | 16.719014 / 8.26% | 16.775787 / 13.08% |
| 524,288 | 16.710203 / 5.45% | 16.698866 / 8.40% |
| 1,048,576 | 16.699758 / 3.46% | 16.775340 / 7.33% |

For narrow, the quarter/full and half/full differences are respectively
+1.94% ± 5.99% and +1.05% ± 3.46% observed difference SE. For broad they
are +0.045% ± 12.69% and −7.36% ± 7.33%. The errors include shared-row
covariance. These are the predeclared correlated prefixes, not independent
replications or a retrospective acceptance test.

## Fixed neighborhoods and their complement

| Disjoint original-window region | Narrow log Q / row SE | Broad log Q / row SE | Estimated share, narrow / broad |
| --- | ---: | ---: | ---: |
| Old radius-two neighborhood | 13.627398 / 3.03% | 13.589556 / 4.11% | 4.63% / 4.13% |
| New radius-two neighborhood | 16.618812 / 3.72% | 16.678947 / 7.85% | 92.22% / 90.81% |
| Outside both neighborhoods | 13.240216 / 14.22% | 13.790492 / 33.80% | 3.14% / 5.05% |

The dominant contact remains the newer one. Tight old-contact calibration
gives log Q=12.067129 and 12.060369 in its R=0.5 ball, with 0.93% and
0.94% row errors, consistent with the historical log Q=12.054798 reference
(4.16% row SE). The new R=1 estimates are log Q=16.109578 and 16.176045
with 3.42% and 9.33% row errors, compatible with the historical multiscale
reference log Q=16.091928 (18.17% row SE). Historical reference rows
informed training; they remain calibration
data, not independent held-out tests of the fitted model.

The outside-both estimate remains less precise. Its largest broad draw
supplies 32.55% of that part, or about 1.65% of the full estimate. It is
`r06`, seed 103207064, draw 55424, at new-chart radius **2.002596 Å-equivalent**:
only 0.002596 beyond the frozen R=2 cutoff. Its original q is 2.08156 and
independent atom gaps are 0.03829 and 0.04135 Å. Its position relative to
the cutoff identifies a contribution at the measured neighborhood's edge;
distance alone does not establish connectedness of a thermodynamic basin.
A separate geometric check contracts the Cayley coordinate radially to R=2.
The whole segment moves every atom center by at most 0.0052401 Å, leaving
at least **0.0330530 Å hard clearance**. Its bounded original q remains
between 2.079617 and 2.083504, and its capture distance stays below 2.778 Å.
Thus there is a short hard-valid path across this chosen cutoff. It does
not measure the statistical weight or mixing time along that path.
The exact same cutoff and contribution are retained. The small observed
complement fraction is not an upper bound on its unknown mass.

![Independent repeat and fixed neighborhood weights](../runs/ab-intermediate-expanded-atlas-repeat-figure-20260920/contact-atlas-repeat.png)

This repeat supplies stronger evidence for a stable **fixed-AB intermediate
integral** at a few-percent observed precision with the narrow proposal.
The complete q≥5 region, native/shoulder tails, neighbor-formation costs,
and reversible trajectory mixing still require their own evidence. This
importance result alone establishes neither an assembly mechanism nor a
model failure.

## Reproduction

`tools/prepare_contact_atlas_repeat.py` freezes the new allocation and
verifies unchanged model bytes. The existing full-window runner executes
the two declared arms. `tools/analyze_expanded_contact_atlas.py` validates
the frozen prefix schedule and repeat lineage before auditing the rows.
Independent arm audits may use separate processes; this changes analysis
wall time, not the sampled estimator.

- Preparation: `runs/ab-intermediate-expanded-atlas-repeat-preparation-20260920`
- Production: `runs/ab-intermediate-expanded-atlas-{narrow,broad}-16x65536-l64-20260920`
- Audit: `runs/ab-intermediate-expanded-atlas-repeat-audit-20260920`
- Independent comparison: `runs/ab-intermediate-expanded-atlas-repeat-comparison-20260920`
- PNG, SVG and figure provenance: `runs/ab-intermediate-expanded-atlas-repeat-figure-20260920`
- Original-row outer-peak and continuous boundary-path checks: `runs/ab-intermediate-expanded-atlas-repeat-outer-peak-20260920`

`tools/compare_expanded_contact_atlas_repeat.py` verifies source hashes,
unchanged models/physical masks and disjoint seeds before computing
independent-difference variances. `tools/plot_contact_atlas_repeat.py`
renders the archived comparison and original/repeat audit summaries.
Ten targeted prefix/lineage/comparison tests pass; the physical executable
is unchanged. The figure was rendered and visually inspected.

Recorded SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Frozen repeat protocol | `ea5f61727ec10b37b52208fc7ba64ffa77062c4c229696b1b7ef16a973e4ae57` |
| Complete repeat audit | `e3e7e4b8bee9207167bfee68e2e9d31ec84876bdd69786adea3f4ff831b308e2` |
| Independent comparison | `b3542b59fbbc1ce7d12bb40d2fd8942fc6d4902e3e4609fd55cd70e2a31c12d1` |
| Continuous boundary-path bound | `3f30e5464836c52972126c4fa9977d1b5f1e326a30e43810fe2a7bb0b47f407c` |

This is still a fixed-AB importance-sampling reference. It cannot measure
reversible trajectory mixing, the cost of assembling A and B, the other
original-q regions, or a crystal nucleation rate. The separate
[one-neighbor SMC control](smc-shoulder-control.md) addresses a
different matched-target discrepancy.
