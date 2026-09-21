# A calibrated proposal for the complete far region

The [independent local reference](far-contact-local-reference.md) measured a
displaced competing contact that broad capture sampling almost never visits.
This experiment uses that reference to construct a frozen proposal, then
measures **all captured original q≥5 poses**, including every pose outside
the local neighborhood. It does not replace the full target by the training
region.

The pilot and a larger independent repeat are complete. The calibrated
small neighborhood is reproducible, but the repeat reveals rare weight
outside it. Agreement of the two full totals does not establish convergence:
their regional contributions differ, and both totals rise from the pilot.

Both AB neighbors, depletant radius 1.5 Å, activity 0.035 Å⁻³, capture radius
18 Å, original 2 Å/15° metric, and normalized proper SO(3) Haar measure remain
fixed. The [support proof](far-capture-reference.md) establishes q<37 everywhere
inside capture. Statistical weights Q retain the same Å³ center-volume units.

## Local covariance calibration

`tools/fit_far_local_guide.py` uses all 6,822 positive observations from the
radius-0.5 geometric reference, which contained 65,536 unconditional draws
in four independent populations. Their original importance weights are
normalized once across all populations. No population is given equal mass
artificially; no weights are clipped or trimmed.

The weighted scatter receives the existing additive floor: 0.05 Å on each
translation coordinate and ell·tan(0.1°/2) on each Cayley coordinate, with
ell = 55.02283113 Å. The resulting Gaussian is globally normalized in the
same A-relative chart; its fitted moments describe only the truncated
reference region, not an entire physical basin.

The source has weight ESS 666.31 and largest normalized weight 0.699%.
Leaving out each population shifts the mean by 0.0055–0.0076 Å in the
geometric coordinates. Covariance generalized eigenvalues relative to the
pooled fit range from 0.921 to 1.060. Held-out original-weight averages of
log proposal density improve by 2.186–2.210 nats over the geometric width-0.2
control and 4.975–5.067 over width 0.4. These are conditional ratio diagnostics
on the source region; overlapping training folds are not independent tests.
The additive floor contributes to this stability.

Independent chart and laboratory-frame density calculations agree within
2.52×10⁻¹³. Fitting requires no new physical draws and took about 1.34 CPU
seconds, excluding the original source sampling and its audit.

## Cover every recorded candidate geometrically

`tools/prepare_far_contact_centers.py` combines the original 96 far candidates,
the 24 largest local-reference observations, and the eight largest flat-cover
observations. The previously selected peak remains the first center.
Deterministic farthest-first selection adds centers until every recorded pose
is within 0.5 Å rigid-member RMS of a selected center.

There are **59 centers**: 47 historical, four from the new local references,
and eight from flat sampling. The maximum nearest-center distance among all
128 candidates is 0.474489 Å. All 59 centers pass independent AB atom checks.
This covers the recorded candidate set; it is not a cover of all physical
basins. The flat-discovered geometries remain represented even though their
observed mass was small.

## Frozen mixture and width control

Each atlas G has 180 Gaussian components:

- 20% of its probability preserves the full old 119-component proposal;
- 60% is shared equally among the 59 geometric centers;
- 16% uses the new fitted local Gaussian;
- 4% uses the same local Gaussian with its covariance multiplied by four.

The narrow and broad arms differ only in the geometric standard deviation,
0.2 versus 0.4 Å-equivalent. Their old components, centers, fitted means,
fitted covariances, and weights are identical. Independent artifact checks
verify that precisely the 59 geometric covariances change by a factor of four.

The full physical proposal is

\[
g(x)=0.01g_{74\,\mathrm{Å}\ \mathrm{ball/Haar}}(x)
 +0.99\left[0.95G(x)+0.05g_{36\,\mathrm{Å}\ \mathrm{cube/Haar}}(x)\right].
\]

The complete ball has full proper Haar orientations. Every importance weight
divides by the entire normalized mixture, with all body-frame Jacobians.
Hard, q, and capture rejections remain zeros in the original unconditional
denominator. Poisson factors use the full fixed exclusion union, retaining
many-body depletion; two independent cloud factors at λ/activity=64 are
averaged linearly.

The 256 cloud-free model probes per width gave 77 narrow and 44 broad valid
far poses. Independent scalar and vectorized full hybrid log densities agree
within 7.64×10⁻¹⁴. These probes do not measure physical acceptance or the
normalizer, and their results do not change the frozen production allocation.
The reviewed Rust executable is unchanged.

## Independent production and analysis

Each arm uses eight populations of 32,768 unconditional draws: **524,288 new
draws** in total. Seeds are 107101010+1009i and 107201010+1009i, i=0,…,7.
The models are frozen before these streams start. No refitting, population
replacement, or result-dependent stopping occurs during production.

The protocol fixes prefixes 8,192, 16,384 and 32,768 within every population.
The shorter prefixes share rows with the full population. Their comparisons
therefore retain the IID prefix covariance rather than treating estimates
as independent. Entire widths remain separate estimates of the same target.

The all-row audit reconstructs the original full density, q, hard/zero masks,
Poisson factors, and hashes. Analysis retains nested local balls at radii
0.5, 1 and 2, plus the disjoint shells [0,0.5], (0.5,1], (1,2], and outside2.
The four disjoint pieces sum to the complete far integral. Their covariance
is retained; nested balls are not added. Original local references provide
calibration comparisons, not unseen-tail guarantees.

Row and population errors, width and prefix sensitivity, largest weights,
paired-cloud noise, and the complete outside2 contribution determine whether
precision has improved. Importance ESS per CPU is an integration diagnostic;
it does not establish MCMC mixing, physical attachment rates or assembly.

Artifacts:

- `runs/ab-far-local-guide-20260921`: fit, holdout diagnostics, frozen models.
- `runs/ab-far-contact-centers-20260921`: candidate cover and atom checks.
- `runs/ab-far-contact-atlas-preparation-20260921`: protocol, models, probes,
  support proof, exact runner commands and provenance.
- `runs/ab-far-contact-atlas-{narrow,broad}-8x32768-l64-20260921`: original
  fixed-budget physical populations.
- `runs/ab-far-contact-atlas-assessment-20260921`: independent full-density
  audit, same-row partitions, fixed-prefix comparisons, and atom checks.
- `runs/ab-far-contact-atlas-figure-20260921`: the figure below and its
  archived inputs.

`commands.json` in the preparation contains exact invocations for fresh
production directories. The existing runner refuses to overwrite a campaign.

## Completed fixed-budget results

Both eight-population campaigns finished successfully. The independent audit
checked all **524,288 original rows**, including full proposal densities and
Poisson factors. Sixteen largest-weight poses passed separate atom-union
checks. Geometric and physical calculations preserve all original draws,
including zeros.

| Complete far region | Narrow | Broad |
| --- | ---: | ---: |
| log Q | 17.446139 | 17.430035 |
| Observed row RSE | 5.62% | 10.07% |
| Independent-population RSE | 3.35% | 10.01% |
| Importance-weight ESS | 316.7 | 98.6 |
| Largest row fraction | 3.48% | 8.67% |
| Sampler CPU seconds | 2002.3 | 1441.2 |

The difference is 0.14 combined observed row standard errors. The broad
proposal's observed variance times CPU is 2.24 times the narrow proposal's.
This compares importance integration of the same Q and excludes discovery,
fitting and auditing costs; it is not a trajectory-mixing speedup.

The successive complete-far log Q estimates are
17.4609, 17.4456, 17.4461 for the narrow arm and
17.6280, 17.4560, 17.4300 for the broad arm. These fixed prefixes share draws;
the accompanying analysis uses their covariance. Their apparent stability
does not bound mass in regions that neither stream has adequately visited.

| Same geometric region | Narrow log Q (row RSE) | Broad log Q (row RSE) |
| --- | ---: | ---: |
| Radius ≤0.5 | 15.26457 (2.05%) | 15.24477 (1.74%) |
| Radius ≤2 | 17.41728 (5.75%) | 17.37782 (10.51%) |
| Outside radius 2 | 13.88628 (21.50%) | 14.45159 (26.64%) |

The radius-0.5 estimates agree with the independent local calibration
log Q=15.23539 (3.85% row RSE). The radius-2 estimates agree with the
predeclared sum of independent local shells, log Q=17.42075 (22.78% RSE).
The exact same masks and unconditional denominators are used for these
comparisons. The outside-radius-2 contribution is positive and remains in
the full answer: 2.84% and 5.09% of the respective observed totals. It is
less precisely sampled than the main neighborhood and needs further
checking. The complete proposal has support there, but support alone is
not a convergence test.

![Complete far-region convergence and disjoint local contributions](../runs/ab-far-contact-atlas-figure-20260921/far-contact-atlas.png)

The error bars show observed linear-scale standard errors transformed onto
the logarithmic plot. The four radial pieces are disjoint. This calculation
supports a reproducible estimate at the pilot's measured precision, not a
certified unseen-tail bound or a result about assembly.

## Larger independent repeat

After the pilot audit, a separate repeat was frozen with **16 populations of
65,536 draws per width**, 2,097,152 new draws in total. It uses the exact same
model and physical-config bytes, centers, local fit, executable, cover and
region definitions. The only changes are the independent seed schedules
108101010+1009i and 108201010+1009i, population count and draw allocation.
Prefixes 16,384, 32,768 and 65,536 are fixed before production. Sixteen
workers per arm give 32 workers in total.

The original and repeat estimates remain separate; no refitting or
result-dependent replacement is allowed. Their common local references
are calibration data rather than independent new training. Preparation
is in `runs/ab-far-contact-atlas-repeat-preparation-20260921`, with protocol
SHA-256 `a7626df99adafbee3256646b9293a5cebe03988cf2280e2c41b2093477ddca8c`.
Launch records are in `runs/ab-far-contact-atlas-repeat-launch-20260921`.
The two physical campaigns use fresh
`runs/ab-far-contact-atlas-{narrow,broad}-16x65536-l64-20260921` directories.

An earlier preparation with unsupported launcher flags is preserved in the
sibling `*-superseded-cli` directory with a `do_not_launch` marker. It never
launched physical draws. The corrected preparation checks its flags against
the archived launcher's actual help output and pins the model lineage.

## The repeat reveals remaining concentration

All 32 repeat populations finished successfully. Both independent density
auditors returned zero; the derived audit checked and remasked all
**2,097,152 original rows**. It verifies exact model bytes, original embedded
provenance, commands, runtime configurations, streams and allocation.
Sixteen largest-weight poses passed separate atom-union checks.

| Complete far region | Narrow repeat | Broad repeat |
| --- | ---: | ---: |
| log Q | 17.684573 | 17.678709 |
| Observed row RSE | 11.23% | 6.66% |
| Independent-population RSE | 11.20% | 4.98% |
| Importance-weight ESS | 79.2 | 225.2 |
| Largest row fraction | 9.97% | 3.95% |
| Sampler CPU seconds | 8087.5 | 5891.1 |

The repeat totals differ by only 0.59%, but rise 26.9% and 28.2% from their
respective pilots. Those changes are 1.76 and 2.14 combined observed row
standard errors. The narrow proposal's ESS falls despite four times as
many draws. Its largest contribution lies at geometric radius 2.21535,
q=13.39827, just outside the calibrated radius-two neighborhood. A second
pose at q=31.37618 supplies another 4.28% of its total and is far from
that chart center. Both pass independent checks against A and B.

The full-total agreement conceals different regional estimates:

| Same region | Narrow repeat log Q (row RSE) | Broad repeat log Q (row RSE) |
| --- | ---: | ---: |
| Radius ≤0.5 | 15.25660 (0.89%) | 15.24830 (0.91%) |
| Radius ≤2 | 17.48402 (3.45%) | 17.62511 (6.93%) |
| Outside radius 2 | 15.97930 (59.84%) | 14.72585 (21.91%) |

The narrow outside contribution is 18.17% of its observed total, versus
5.22% for the broad arm. A single row accounts for 54.86% of the narrow
outside estimate. Conversely, the broad estimate of the radius-one-to-two
shell is higher: log Q=17.09188 versus 16.83738. These contributions remain
separate in the report; their compensation does not establish regional
agreement.

The radius-0.5 calibration remains stable and is now measured at about
0.9% observed row error. The full far integral and its outside tail are
less settled. The pilot's apparent efficiency ranking reverses in the
repeat, so there is no demonstrated stable efficiency winner for the full
integral. More samples have identified a concrete limitation of the
current atlas rather than removed the need for coverage checks.

![Larger repeat retains uncertainty in the positive outside contribution](../runs/ab-far-contact-atlas-repeat-figure-20260921/far-contact-atlas.png)

The repeat's full-density audit, correlated prefixes, separate baseline
comparisons and top poses are in
`runs/ab-far-contact-atlas-repeat-assessment-20260921/analysis.json`, SHA-256
`925a233b627bac1836bfe56cd512e0ceabb049025975e2f56b0e3f8ca44304a2`.
The figure and archived sources are in
`runs/ab-far-contact-atlas-repeat-figure-20260921`. No pilot population was
replaced or pooled into the repeat. The
[regional evidence ledger](ab-regional-weight-status.md) preserves these
qualifications alongside the remaining native and shoulder checks.
