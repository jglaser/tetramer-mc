# Frozen proposal extension for the measured contact-escape bottleneck

A frozen proposal with 117 components substantially improves escape in the matched short pilot: all four extended-atlas runs first reach old R8 in **1–16 cycles**, compared with **69–891 cycles** for the original atlas at the same starts and seeds. These are eight deliberately initialized 2000-cycle records, not equilibrium rate estimates. One extended run later makes a 145-cycle revisit to the contact extension, so the improvement does not remove all residence problems.

The proposal adds the already frozen outside-R8 geometric Gaussian to the current 116-component atlas:

\[
G_{\rm new}(x)=0.99G_{\rm parent}(x)+0.01g_{\rm extension}(x).
\]

The new component copies the existing cohort mean, full covariance, original Cayley anchor, and angular length exactly. There is no new fit, occupancy weighting, covariance adjustment, or physical-parameter change. The coefficient 0.01 is a specified proposal-control choice; it is not an estimate of the physical region probability. The parent contains native information, so this remains a controlled sampling experiment rather than template-free discovery.

This follows the [recorded escape diagnostic](outside-r8-local-region.md#what-the-recorded-escape-attempts-show). In the prolonged B5 excursion, learned proposals repeatedly reach the previously measured old-R8 region, but a strongly unfavorable full-G correction rejects them. No persistent Poisson weight is present. The new component tests whether representing this known contact neighborhood directly improves that particular mismatch.

## Density and support

All 117 components are full, untruncated Gaussians in their 6D translation/Cayley charts, transformed with the exact Jacobian to translation volume times normalized SO(3) Haar measure. The new covariance is positive definite, with eigenvalues 0.00380–0.25167 Å² and condition 66.19. Positive weights sum to one. The old 116 component geometries remain byte-for-value identical and their relative weights remain unchanged.

Capture and hard constraints are still imposed by rejection against the physical target. The density is never normalized by a hard-valid or captured fraction. The parent support is retained, and the independent uniform branch remains unchanged at probability 0.1. The posterior-involution correction remains log G(old)−log G(new), evaluated with the complete new Gaussian mixture. The c=0 branch is its matched Gaussian redraw, with c=0.9 providing the correlated transport control.

Preparation independently checks the mixture identity to 1.8×10⁻¹⁵ in log density, and a laboratory/body-frame coordinate conversion to 8.6×10⁻¹⁴ on 256 selected poses. Reconstructing the old proposal corrections at every selected return pair agrees with the recorded values to 1.5×10⁻¹⁴. All physical configuration fields are checked against the frozen region.

## Fixed-candidate diagnostic

The table reevaluates the same 261 hard-valid return pairs and their same recorded Poisson factors. It measures only the change in the complete proposal-density correction.

| Recorded candidate subset | Count | Old median correction | New median correction | Old sum of conditional acceptance probabilities | New sum using the same candidates/clouds |
|---|---:|---:|---:|---:|---:|
| All B5 → old-R8 returns |261|−10.844|+1.378|3.110|242.038|
| Destination original deep component 113 |186|−11.856|+1.123|0.0305|171.009|
| Final residence, cycles 2770–5000 |146|−11.795|+1.469|0.352|135.641|

These are **not fresh acceptances or predicted acceptance rates**. The added component changes source selection as well as the density correction. Across all 2939 B5 retained frames, its posterior source responsibility is at least 0.99769, with median 0.9999983. Thus an actual run would select different charts and produce different destinations. The diagnostic shows that a frozen representation of this contact can remove the measured density penalty for existing useful pairs; only a new controlled trajectory can show whether its actual proposal law produces useful escape candidates.

The new component raises log G across B5 frames by a median 13.297. This large increase from a 1% mixture weight reflects a concentrated local density in a small pose volume. It does not add or alter the physical depletion free energy.

## Matched escape control

The completed experiment has eight runs: parent/extension × c=0/0.9 × two identical initial poses. Starts are the previously frozen chronological cohort entries 8 and 24, at source cycles 2819 and 4336. Both are geometrically valid B5 poses; selection does not optimize a new weight or covariance. These two starts belong to the known region used to construct the geometric guide. They test neither held-out generalization nor stationary initial conditions.

Each run has 2000 cycles and exactly two unchanged local attempts per cycle, the separate uniform branch at 0.1, λ/z=16, and the same 2047-cell gate budget. All physical fields remain fixed: one native site 0 neighbor, 18 Å capture ball, atomic hard shape, depletant radius 1.5 Å, and activity 0.035 Å⁻³. Seeds 98641010 and 98642019 are paired across all four variants for each start. Four workers used 303.987 summed sampler CPU seconds, against the preparation estimate of about 310. All eight jobs completed without a timeout, failure, refit, or budget change.

Prespecified outputs are first B5 exit and first old-R8 entry with censoring at 2000 cycles, region crossings, full retained-frame residence, candidate validity and gate/correction factors, source/target component concentration, and CPU cost. The same-start old-model control is necessary because the earlier c=0 record never entered B5 and therefore did not measure escape from it.

| Atlas | Cohort | c | First old-R8 entry: cycle, move | B5 frames / 2001 | B5 reentries | Accepted contact changes | Sampler CPU seconds |
|---|---:|---:|---|---:|---:|---:|---:|
| Parent |8|0|592, local|37.431%|4|100|35.27|
| Parent |8|0.9|418, local|21.639%|1|181|38.22|
| Parent |24|0|891, local|85.707%|2|49|32.63|
| Parent |24|0.9|69, learned|3.448%|0|236|40.83|
| Extension |8|0|4, learned|0.350%|1|175|35.51|
| Extension |8|0.9|2, learned|0.300%|1|249|43.81|
| Extension |24|0|16, learned|8.696%|2|188|37.43|
| Extension |24|0.9|1, learned|0.050%|0|254|40.29|

Residence includes the initial frame. B5 reentries count entries from any outside region, including an excursion outside both B5 and old R8; they are not all complete old-R8 roundtrips. Contact changes are exact changes in the binary mobile/fixed atom-incidence descriptor for overlapping depletant-expanded atom spheres, counted on accepted moves. They are not independent samples or necessarily changes of a crystallographic patch.

All four extension first entries use the new component 116 as the source and an existing deep guide as destination. Reverse passages are also observed: the extension runs at cohort 8, c=0 and c=0.9 return to B5 at cycles 1109 and 1384, then leave at 1112 and 1388. At cohort 24, c=0 the two revisits last 13 and 145 cycles, from 141–153 and 285–429. The latter accounts for much of that run's 8.696% B5 residence. These observations show both directions remain accessible; they do not establish correct equilibrium occupation in these short records.

![Matched retained-state region trajectories](../runs/outside-r8-escape-2000/assessment/matched-escape-trajectories.png)

This fresh experiment uses the changed posterior source-selection law. It is distinct from the earlier fixed-candidate calculation: actual first-return corrections range from −3.021 to +1.017, not the fixed-candidate median +1.378. The geometric chart gives the trapped states a useful source representation, and the first returns actually use it. The observed improvement therefore goes beyond changing a diagnostic denominator on an unchanged set of endpoints.

## Replay and cost

Independent analysis checks all 48,000 attempted transitions, all 14,336 learned full-mixture densities and posterior label ratios, and every one of those 14,336 latent maps, reverse noises, selected densities, and Haar Jacobians. It independently tests atomic hard geometry for **all 46,949 captured candidate endpoints**, including rejected proposals, and replays contact incidence on every accepted pose. Noncaptured candidates have zero physical weight and do not require a hard test. Initial and retained states, checkpoints, acceptance arithmetic, and original metric classifications all agree.

Maximum log-density discrepancies are 2.3×10⁻¹⁴ for old states and 2.6×10⁻¹³ for proposals; posterior-label discrepancies are below 1.6×10⁻¹². Gate/count and acceptance arithmetic agree exactly. The smallest absolute core gap encountered by the independent geometry audit is 4.5×10⁻⁷ Å; no numerical exception was waived.

Analysis used 108.775 summed CPU seconds, including 79.362 for the complete geometry/contact audit. This is separate from 303.987 sampler CPU seconds: 146.948 for the parent and 157.039 for the extension. The full audit is intentionally more extensive than the stride 10 contact observer used in the earlier pilot. The descriptive contact correlations retained in per-run analysis are finite-record diagnostics; no equilibrium ESS is inferred.

No global equilibrium claim, assembly claim, precise escape-rate estimate, or general speedup follows from this small control. It tests two configurations of one already discovered contact region under a native-informed parent. The earlier 16×25,000-cycle extension remains unlaunched and separate. No additional production is running.

## Frozen artifacts

`runs/outside-r8-atlas-extension-20260920/site0` contains the new `model.json`, `report.json`, 261 `fixed-candidates.jsonl` records, the immutable preproduction `benchmark-plan.json`, and archived source chart/model/configuration/cohort/code files. The new model SHA256 is `54423aef3a1317e2fe8369f9cc8e6dad8effad4c72ff18a76fdcc5f58566458d`.

`tools/prepare_outside_r8_atlas_extension.py --out <fresh-directory>` reproduces preparation and refuses an existing output directory. It launches no trajectory and leaves previous models and results unchanged.

Production and analysis are under `runs/outside-r8-escape-2000`, with separate parent/extension configurations and results. `tools/run_outside_r8_escape_control.py` archives the already validated docking binary and all frozen input hashes before launching. `tools/analyze_outside_r8_escape_control.py` reproduces the complete replay, geometry checks, region/contact diagnostics, and figure. The preparation report remains a record of what was known before production; the production summary records completion.
