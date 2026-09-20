# Explicit far-region SMC reference

Fresh configurations are prepared in
[/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920).
All four site-0, 512-particle production populations completed successfully after
the analytic controls and independent protein geometry audit passed.
Their [terminal runner handle](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/runner-handle.json)
records exec session `60661` with owner-verified exit status 0; status and
per-population logs are retained. No job remains running.
The
[manifest](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/manifest.json)
records exact commands, seeds, configuration hashes, source hashes, and
historical runtime measurements. There are four independent populations for
each combination of sites 0/1 and particle counts 512/2048. The 2048-particle
configurations are prepared for a later sensitivity decision.

## Configuration-only restriction

Keep the archived executable's `basin="other_adsorbed"`, which accepts its
internal registration score above one. Set

```json
{
  "member_error_scale": 10.0,
  "angle_error_scale_deg": 75.0,
  "uniform_probability": 1.0,
  "proposal_components": []
}
```

Both score denominators have been multiplied by five, so

\[
 q_{\rm internal}(x)
 =\min_r\max\{\Delta_r(x)/10\,\mathrm{\AA},\theta_r(x)/75^\circ\}
 =q_{\rm original}(x)/5.
\]

The executable therefore targets original `q > 5`, equivalent to `q >= 5`
apart from a zero-measure boundary. This changes the integration region and
its reflecting rejection boundary; it does not change shape, interactions,
or the reference measure. Native reporting remains **original `q <= 1`**.
Saved internal scores must be multiplied by five or independently recomputed
using the original 2 Å / 15° metric. Never relabel internal `q <= 1` as the
original native region.

All protein configurations retain the archived 4004-sphere tetramer shape,
fixed-neighbor poses, 18 Å capture sphere, 360 Å periodic box, 1.5 Å depletant
radius, and activity 0.035 Å⁻³. The uniform capture-ball/full-Haar proposal
has constant density `g=1/(4*pi*18^3/3)` on the target. Initial draw budget
1,048,576, 128 fixed annealing increments, two mutation sweeps per stage,
one thread per population, and local/line/arc/global move settings are
unchanged. Only the score denominators, independent seeds, requested
population size, and explanatory metadata differ from the source config.

The resulting `exp(logZ)` directly estimates far-region mass. Compare it with
the archived terminal-filter estimator
`exp(logZ_other)*mean(original_q>=5)`, whose
[unnormalized-measure argument](/home/xvg/tetramer-mc/docs/previous-smc-region-audit.md)
does not require completed contact roundtrips. Agreement would be a useful
independent check; it would not by itself prove coverage of all far basins.

## Archived executable and reference checks

Use the frozen binary at
[/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/coordination_smc](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/coordination_smc).
Do not silently substitute the currently built project executable.

| Artifact | SHA-256 |
|---|---|
| Archived executable | `9422a048f80476845510de71b1e8b97813846e521a1d10040d8a4efa1c4421c9` |
| Archived `coordination_smc.rs` | `f216b5885e063c4a3c14109424b1ddeb071114b3b43f71cc355b79f292d33487` |
| Archived `single_body_depletion.rs` | `0388e24a6c68b54874f6200e6278dc1fda607c6d5f42558a73740ea70b1680b7` |
| Archived `rigid_pose_sampling.rs` | `927c077af994a5b5000212f82ba407955bf0a644e25f59dc7dc52adbadad4264` |
| Tetramer shape | `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9` |

The historical kernel-validation record predates the frozen optimized binary;
its main-source and executable hashes differ. Fresh analytic controls therefore
use this exact archived executable, including flat geometry, envelope clouds,
line moves, and arc moves. The new control configs compare two constructions:
an unrestricted hard-shell SMC with a terminal `q>=5` indicator, and an explicit
far-region SMC using the rescaled metric above.

The artificial system has core spheres of radius 1 Å, depletant radius 0.5 Å,
and capture radius 3 Å, with the fixed sphere and reference at the origin.
Hard-valid positions satisfy `2<r<3`. Original `q=max(r/2,theta/15deg)`, so
the far region is exactly the orientation complement `theta>75deg`. Thus

\[
 Q_{\rm far}=
 \left[1-\frac{\alpha-\sin\alpha}{\pi}\right]
 \int_2^3 4\pi r^2
 \exp\!\left[z\,\frac{\pi(6+r)(3-r)^2}{12}\right]\,dr,
 \qquad \alpha=75^\circ.
\]

The reference values are `70.89587903570556 Å³` at `z=0` and
`92.16821118538877 Å³` at `z=0.4 Å⁻³`. There are four independent populations
per construction/activity, each with 512 particles, 16 fixed increments, and
8192 initialization draws. Additional checks reconstruct every final score
from sphere position/quaternion, enforce hard-shell support, and verify at
zero activity that `Zhat=hits/(M*g)` directly. These controls address the
region-normalizer construction and reference measure; they do not establish
agreement between the protein geometry engines.

All 16 fresh controls passed. The
[validation record](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/control-validation.json)
contains exact commands, configuration/summary hashes, and all population
estimates. The largest discrepancy from an exact integral was 2.19 estimated
standard errors. Total child CPU was approximately 95.27 seconds; the final
15-control batch took 27.35 seconds wall time on four workers. The first
positive control was run separately to measure cost. Far-mass comparisons are:

| Construction | Activity / Å⁻³ | Mean Qfar / Å³ | Independent-population SE | Exact Qfar / Å³ |
|---|---:|---:|---:|---:|
| Terminal indicator | 0 | 71.3833 | 0.2647 | 70.8959 |
| Explicit region | 0 | 70.8895 | 0.2891 | 70.8959 |
| Terminal indicator | 0.4 | 93.0839 | 0.4176 | 92.1682 |
| Explicit region | 0.4 | 91.8874 | 0.7070 | 92.1682 |

## Commands and production decision

The exact command for one prepared site-0 population is below. Its output
directory belongs to the completed campaign; do not execute it again.

```bash
/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/coordination_smc \
  --config /home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/configs/site0-far5-r1.5-z0.035-n512-r0.json \
  --out /home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r0
```

The following documents the equivalent four-population launch procedure.
It checks the recorded input hashes and refuses existing output directories.
The completed campaign was launched with
`/home/xvg/protein-nucleation/.venv/bin/python -u /home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/run-site0.py`.
That persisted runner additionally records child PIDs, poll timestamps, exit
codes, and final summary hashes. Neither launcher should be invoked a second
time against the completed outputs.

```bash
/home/xvg/protein-nucleation/.venv/bin/python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, subprocess
root = Path('/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920')
manifest = json.loads((root/'manifest.json').read_text())
for path, expected in manifest['source_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
jobs = [j for j in manifest['jobs'] if j['site'] == 0 and j['population'] == 512]
assert len(jobs) == 4
for job in jobs:
    assert not Path(job['output']).exists(), job['output']
    assert hashlib.sha256(Path(job['config']).read_bytes()).hexdigest() == job['config_sha256']
def run(job):
    output = Path(job['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('x') as log:
        subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, check=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(run, jobs))
PY
```

Retain every completed population and every zero estimate, and combine masses
on the linear scale. Failed processes need diagnosis; they must not be silently
replaced according to their physical result. A genuine interrupted run can use
the archived `--resume` option with its unchanged config/binary, whose
fingerprints are checked by the executable.

## Runtime evidence and limitations

The old 0.035 Å⁻³ other runs provide eight one-thread measurements per site:

| Site | Historical 512-particle mean wall time | Range | Fixed initialization mean | 2048-particle forecast |
|---|---:|---:|---:|---:|
| 0 | 17.07 min | 16.91–17.23 min | 17.81 s | 67.38 min |
| 1 | 30.43 min | 30.20–30.61 min | 12.91 s | 121.09 min |

The forecast keeps initialization fixed and multiplies the remaining work by
four. It is an extrapolation, not a measured explicit-far runtime. Excluding
the near region, different concurrent load, and sampled cloud volumes can
change the cost. Four populations run concurrently need approximately one
listed wall interval if comparable resources are available; their aggregate
work is four times a single population. Historical measurements came from a
32-worker campaign and are recorded individually in the manifest.

Four fresh 512-particle populations are a bounded independent reference,
not a convergence certificate. Comparison with the prepared 2048-particle
populations, independent starts/seeds, mass concentration across populations,
and coverage of high-weight competing contacts remain necessary. Direct far
SMC only supplies the `q>=5` contribution; complete contact weights also need
the entire `q<5` complement with compatible bound/unbound definitions.


## Completed pilot: high variance, no convergence claim

The four fresh site-0, N=512 populations took 15.3–15.8 minutes each. Their
log far-region masses were 17.5310, 27.2731, 17.5823, and 14.1615. Combining
the four estimates on the linear scale gives log(mean Qfar)=25.8869, but one
population contributes 99.9877% of that mean (replicate mass ESS 1.00025/4).
The observed relative standard error is about 100%, so this pilot does not
establish a converged far-region normalizer.

The dominant population ends in original q=25.94–28.24 and retains one
initial family. Its final-cloud mean overlap estimate is about 1055 Å³,
which motivates a separate fixed-pose geometry/overlap audit and improved
coverage of competing basins. Those audits do not replace or resample the
original mass estimate. The complete record is
[initial-production-assessment.json](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/initial-production-assessment.json).
N=2048 and site-1 configurations remain prepared and unlaunched.
