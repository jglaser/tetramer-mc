# Controlled transport around the successful frozen atlas

This pilot keeps the information and resolution of the successful frozen
proposal, then enables reversible changes to its parameters in stages. It does
not compare a learned atlas against a newly fitted, much broader mixture.

The campaign reuses the four dispersed preparations and MC seeds from
`runs/free-tetramer-fixedk-400`. Each preparation contains 12 mobile rigid
tetramers with no initial interbody exclusion contact. Every arm has the same
28-component native-informed atlas, spherical boundary of radius
354.50820786337056 Å, depletant radius 1.5 Å, activity 0.035 Å⁻³, local
translation and rotation proposals, 50% global-proposal probability, spherical
GCA, and common center shifts. These preparations are not equilibrium samples.

| Arm | Transported parameters | Gain | Auxiliary noise |
|---|---|---:|---:|
| Frozen | None | 0 | 0 |
| Means | Six-dimensional component means | 0.25 | 0.1 |
| Covariance | Means and full component covariances | 0.25 | 0.1 |
| Weights | Means, covariances, and mixture weights | 0.25 | 0.1 |

Disabled parameter groups have zero gain and noise. All transported arms use
the reference initialization, so the initial proposal recovers the supplied
atlas. The normalized auxiliary residual is refreshed after each sweep.
Component count, chart anchors, and the reference atlas remain fixed. Current
pair geometry supplies conditional fit statistics; no growing trajectory
archive enters the update.

The fit uses an assignment cutoff of six reference standard deviations,
residual clipping at three, and shrinkage eight. Parameter coordinates are
relative to each reference component, which preserves the previously learned
narrow native-registration scales. The [transport target and balance argument](atlas-transport.md)
are documented with the implementation; these pilot trajectories by themselves
do not validate the equilibrium marginal.

Run the matched campaign and its independent audit with:

```bash
cargo build --release
/home/xvg/protein-nucleation/.venv/bin/python tools/run_atlas_transport_campaign.py \
  --out runs/atlas-transport-400 --sweeps 400 --workers 4
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_atlas_transport_campaign.py \
  --campaign runs/atlas-transport-400 --workers 4
```

The launcher archives the executable, shape, atlas, native catalogue, source
configurations, and hashes. It refuses a nonempty output directory. An optional
`--atlas-json` applies common parameter overrides after the arm presets; use
this only when the resulting comparisons remain intentional.

The analyzer replays every accepted physical update to all saved frames and the
final checkpoint. Every hard-valid global candidate is classified, including
rejections. Native motif formation and breakage use the established registered
tetramer-pose plus external residue-patch criterion, with its entry and
retention thresholds. Native candidates and accepted transitions are separated
by the Gaussian and uniform branches. Multiple motif labels can represent one
body pair, so motif transitions and unique bond counts need not coincide.

Every saved frame additionally receives independent atom-union hard-overlap
and spherical-wall checks. Atomic validation at saved frames is not an audit
of every intermediate hard-core calculation.

Reusing the old starts provides a paired comparison; it does not create four
new independent preparations. Native information is supplied in every arm.
Assembly events therefore establish accessibility with that information, not
native discovery, crystal growth, equilibrium, or a mixing speedup.

## Results

All 16 runs completed 400 sweeps. Full move replay and the independent saved-
frame checks passed. The frozen control exactly reproduced all 164 saved
physical configurations from the earlier four frozen runs. This directly
checks that the control retained the successful proposal and random streams.

| Arm | Gaussian moves accepted | Native Gaussian candidates accepted/proposed | Native bonds formed/broken | Native registry changes | Total sampler CPU s |
|---|---:|---:|---:|---:|---:|
| Frozen | 42 | 37/428 | 34/0 | 0 | 130.06 |
| Means | 48 | 40/435 | 37/0 | 2 | 133.18 |
| Covariance | 46 | 41/428 | 37/0 | 1 | 135.64 |
| Weights | 58 | 33/375 | 28/0 | 0 | 133.63 |

Each arm made 8,595 Gaussian attempts, pooled over the four paired starts.
Native candidates here are hard-valid global candidates passing the native
entry screen; rejected candidates are included in the denominator. Candidate
counts and accepted transitions are correlated trajectory observations, not
independent Bernoulli trials. A move can form more than one native body-pair
bond, while an accepted native candidate can preserve an existing bond.

The essential control succeeds: reversible mean and covariance changes retain
native accessibility when the supplied atlas keeps its centers, rotational
charts, covariance scales, and resolution. Their native candidate counts and
accepted counts remain comparable to the frozen sampler. Fit and reverse-model
accounting takes 2.7–3.1% of total sampler CPU time in the transported arms.

Adding weight transport increases total Gaussian acceptance but reduces native
candidate and bond counts in this pilot. This is a reason to keep total
acceptance separate from useful native transitions, not evidence for a reliable
ranking from four starts. No arm demonstrates a mixing speedup.

All native bond formations and the three registry changes arose from Gaussian
proposals. The apparent motif breakages are changes of registration label on a
body pair that remains native-bonded; **no native body-pair bond detaches**.
Final largest native components contain two to four tetramers. The experiment
recovers assembly accessibility with a native-informed proposal but does not
resolve oligomer trapping or establish reversible bulk attachment/detachment.

The full [machine-readable audit](../runs/atlas-transport-400/assessment/analysis.json)
contains per-run native proposals, forward/reverse and Poisson diagnostics,
every native transition, replay errors, and source hashes. The compact
[campaign report](../runs/atlas-transport-400/assessment/report.md) includes final
native bond counts by start.

![Matched atlas transport pilot](../runs/atlas-transport-400/assessment/atlas-transport-comparison.png)
