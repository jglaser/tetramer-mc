# A-neighbor native-plus-shoulder SMC control

This configuration-only control tests whether the archived SMC can recover the
missing shoulder, `1 < q <= 2`, when the complete `q <= 2` target is explicitly
covered at initialization. Its physical neighborhood is **A only**, matching
the old site-0 SMC and the one-neighbor direct integrals. The AB environment,
`q > 2`, and assembly remain separate questions.

The frozen plan is [protocol.json](/home/xvg/tetramer-mc/runs/smc-shoulder-control-20260920/protocol.json).
Preparation does not launch a simulation. Protocol status is a freeze-time
record; execution and terminal assessments are separate artifacts.

## Physical region and normalized proposal

The original metric remains
`q = max(maximum member displacement / 2 Å, proper rotation error / 15°)`.
The executable receives scales 4 Å and 30°, so `q_internal = q / 2` and its
`basin="native"` filter enforces original `q <= 2`. Every reported endpoint
is independently evaluated with the original metric. Native still means
original `q <= 1`.

Exactly centered rigid members imply that maximum member displacement at most
4 Å bounds the center displacement by 4 Å. The original reference is unchanged.
Replacing the **explicit** `proposal_components` override by a product of an
outward-rounded radius-4 Å ball and Haar cap of 30° therefore covers the whole
target, including its hard-valid portions. The positive uniform branch is a
capture **ball**, as implemented by the archived executable. It is not a cube.
Both unconditioned mixture components have constant density on the target:

```
g = .99 / [V_ball(nextafter(4, +infinity)) H_cap(30°)] + .01 / V_ball(18)
  = 0.491617546504177 Å^-3,
H_cap(alpha) = (alpha - sin(alpha)) / pi.
```

No initialization retries condition away the hit probability. Shape, A pose,
18 Å capture sphere, normalized proper SO(3) Haar measure, 1.5 Å depletant
radius and 0.035 Å^-3 physical activity are unchanged.

## Why terminal filtering has the correct normalizer

For fixed B = {original q <= 2}, define

```
gamma_t(x) = H(x) 1_capture(x) 1_B(x) g(x)^(1-t) exp[z_t C_A(x)],
lambda = reference_activity * smc_lambda_ratio,
c = 1 + z/lambda,
z_t = lambda * (c^t - 1).
```

A fresh conditional count `K ~ Poisson(lambda*c^t*C_A(x))` before advancing
by dt gives weight `exp[dt*(K*log(c)-log(g(x)))]`. Its Poisson expectation
recovers `gamma_(t+dt)(x)/gamma_t(x)`. Initialization contributes `hits/M`,
where M is the fixed number of unconditional g draws. Unbiased systematic
resampling and invariant mutation preserve the unnormalized empirical-measure
identity `E[Zhat_t * eta_hat_t(f)] = integral gamma_t(x) f(x) dx` for the fixed
schedule in exact arithmetic. This concerns the weighted measure; unweighted
finite-population endpoint fractions need not be unbiased physical probabilities.

Each reversible physical mutation is filtered through fixed basin/capture
restrictions. The remaining `g^(1-t)` tilt is constant on this target; its
correction is still evaluated by the original code. Global proposals keep
their original proposal and Poisson corrections. No adaptive region or schedule
is introduced. This extends the [archived argument](/home/xvg/tetramer-mc/docs/previous-smc-region-audit.md)
without changing the mutation implementation.

The reported masses are `Zhat * mean(q <= 1)` and
`Zhat * mean(1 < q <= 2)`. Their sum equals the population Zhat. Average
independent populations on the linear scale, retain zero populations, and
include native/shoulder covariance in the total standard error. Terminal
filtering yields a valid estimator even without basin roundtrips; convergence
and mixing still need evidence.

## Frozen budget and reference checks

Four N=512 populations retain 1,048,576 unconditional initialization draws,
128 fixed increments, two mutation sweeps per stage and one thread each.
Seeds are `104001010 + 1009*i`, i=0..3. Four N=2048 populations with independent
seeds `104101010 + 1009*i` are prepared for later population-size sensitivity;
they are not automatically launched. The exact archived executable hash is
`9422a048f80476845510de71b1e8b97813846e521a1d10040d8a4efa1c4421c9`.

The old matched N=512 native runs took 764.27–772.67 s, mean 767.73 s, including
about 3 s initialization. Four concurrent pilot populations are thus a roughly
13-minute forecast, not a measured q<=2 runtime. The new region may change
cloud costs. Four populations are half the original eight-population budget
and do not by themselves certify convergence.

Before positive-activity protein production, the runner requires:

- Four zero-activity and four positive-activity analytic sphere populations.
  Core radius .5 Å, depletant radius .5 Å, capture radius 3 Å, and the same
  4 Å / 30° proposal give native and shoulder positive volume. Overlap for
  1<r<2 is `pi*(4+r)*(2-r)^2/12`, vanishing for r>=2. Integration over r and
  exact Haar caps gives native, shoulder and total references at z=0 and .4.
- A deliberately impossible sphere basin, which must remain a zero estimator.
- A zero-activity protein control at the full initialization budget. Its exact
  identity is `Zhat = hits / (M*g)`. This checks normalization and the rescaled
  predicate, not convergence of the hard protein volume.

Audits reconstruct saved stage weights, cumulative normalizers, conditional
Poisson intensities, family propagation and systematic resampling consistency.
They check final original q and target density; sphere endpoints get direct
hard-shell/capture checks. Independent whole-population errors, family diversity
and matched native/shoulder masses are the production diagnostics. Unit tests
cover the explicit override, arbitrary-pose metric rescaling, exact sphere
integrals, zero estimates, and covariance-sensitive sums.

## Commands

Preparation uses `tools/prepare_smc_shoulder_control.py prepare --out DIR`.
The output above already exists; never repeat preparation into it. From the
repository root, with `/home/xvg/protein-nucleation/.venv/bin/python`:

```
tools/prepare_smc_shoulder_control.py run --out DIR --group sphere-z0.0
tools/prepare_smc_shoulder_control.py run --out DIR --group sphere-z0.4
tools/prepare_smc_shoulder_control.py run --out DIR --group zero-hit
tools/prepare_smc_shoulder_control.py run --out DIR --group protein-zero
tools/prepare_smc_shoulder_control.py analyze --out DIR --controls
tools/prepare_smc_shoulder_control.py run --out DIR --group protein-n512
tools/prepare_smc_shoulder_control.py analyze --out DIR --group protein-n512
```

Each group refuses existing outputs and preserves child commands, PIDs, exit
codes and logs. A zero statistical result is never replaced with a new seed.

## Completed controls and matched independent references

All ten fresh executable controls completed with exit code zero. The
[control audit](/home/xvg/tetramer-mc/runs/smc-shoulder-control-20260920/control-validation.json)
passes the saved-stage, lineage, normalization and geometric-region checks.
The largest discrepancy among the six sphere regional comparisons is 1.98
independent-population standard errors.

| Activity / Å^-3 | Region | Measured Q ± population SE / Å³ | Exact Q / Å³ |
| --- | --- | ---: | ---: |
| 0 | Native | 0.028787 ± 0.002391 | 0.027817 |
| 0 | Shoulder | 0.789785 ± 0.004595 | 0.790274 |
| 0 | Total q<=2 | 0.818572 ± 0.004672 | 0.818091 |
| .4 | Native | 0.032653 ± 0.001928 | 0.031906 |
| .4 | Shoulder | 0.813453 ± 0.002588 | 0.818564 |
| .4 | Total q<=2 | 0.846106 ± 0.003524 | 0.850470 |

The protein zero-activity control produced 214 hits in 1,048,576 unconditional
draws. Its `log Qhard = -7.7869133869` agrees with the exact hit-count identity;
maximum final q-reconstruction error is 1.44e-15. This is a normalization
control, not a precise independent hard-volume determination.

The [matched direct-reference audit](/home/xvg/tetramer-mc/runs/smc-shoulder-control-20260920/matched-independent-references.json)
re-masks 131,072 archived independent draws, preserving original proposal
densities, unconditional sample counts and zeros. It verifies raw and frozen
input hashes, original q, positive cloud factors, and native/shoulder covariance.
The two proposal sources remain separate:

| Source | Native log Q | Shoulder log Q | Total q<=2 log Q | Total row RSE | Total population RSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Importance guide | 14.977288 | 16.886678 | 17.024848 | 5.98% | 6.12% |
| MIS refined | 14.982290 | 16.983372 | 17.110171 | 8.48% | 8.19% |

The native/shoulder row errors are respectively 5.27%/6.82% for the first
source and 15.51%/9.40% for the second. The native/shoulder population errors
are 7.28%/6.18% and 10.03%/8.15%. These observed errors are not missing-mode
bounds. The historical direct densities are reused; this audit reconstructs
the physical mask and Poisson weights, not a new proposal-density implementation.

The four positive-activity N=512 protein populations completed with exit code
zero after all controls passed. Exec session **85931** is terminal; child
records and exit codes remain archived. The complete audit is
[assessment-protein-n512.json](/home/xvg/tetramer-mc/runs/smc-shoulder-control-20260920/assessment-protein-n512.json).
N=2048 remains unlaunched.


## The missing shoulder is recovered, with substantial uncertainty

| Region | Fresh SMC log Q | Independent-population RSE | Direct importance log Q | Direct MIS log Q |
| --- | ---: | ---: | ---: | ---: |
| Native q<=1 | 15.042769 | 21.81% | 14.977288 | 14.982290 |
| Shoulder 1<q<=2 | 16.971723 | 33.02% | 16.886678 | 16.983372 |
| Total q<=2 | 17.107390 | 30.05% | 17.024848 | 17.110171 |

All six SMC/direct regional comparisons are within 0.29 combined observed
standard errors. This recovers a positive shoulder contribution with the
archived executable, using a target and proposal that explicitly cover it.
The result supports a coverage explanation for the old missing shoulder;
the larger cap and restricted target both change exploration, so it does not
isolate initialization as the sole cause. Agreement with these noisy estimates
is not a convergence certificate.

The four SMC total log normalizers are 17.73367, 16.39431, 16.88455 and 16.93182.
Their contributions to the mean are 46.77%, 12.25%, 20.01% and 20.97%, giving
population-mass ESS 3.15/4. The run took 687.92–701.92 s per population, about
2790 one-thread sampler-seconds in total. These are normalizer diagnostics,
not effective independent physical contact samples per CPU.

![Matched SMC contact masses and annealing coverage](/home/xvg/tetramer-mc/runs/smc-shoulder-control-figure-20260920/smc-shoulder-control.png)

The [SVG](/home/xvg/tetramer-mc/runs/smc-shoulder-control-figure-20260920/smc-shoulder-control.svg)
and [plot input/output hashes](/home/xvg/tetramer-mc/runs/smc-shoulder-control-figure-20260920/provenance.json)
are retained. Bars use one observed SE; direct-reference bars use independent
rows while SMC bars use whole populations. Small orange points show individual
SMC population estimates.

| Population | Initial native endpoints / families | First native stage | Final native endpoints / families | Final family ESS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 / 0 | 1 | 43 / 14 | 3.01 |
| 1 | 2 / 1 | 0 | 76 / 19 | 15.91 |
| 2 | 0 / 0 | 1 | 121 / 20 | 10.43 |
| 3 | 6 / 3 | 0 | 54 / 15 | 10.96 |

Both populations initially missing native poses replenish them at the first
annealing stage. These are correlated descendants at changing physical
activities. Native-fraction curves describe coverage during annealing, not
equilibrium transition rates, roundtrips or independent terminal samples.
The normalizer-weighted final native fraction is 0.12687; the large spread
between populations remains relevant even though every population reaches
both regions.

All 516 saved stage records pass independent reconstruction; the largest
cumulative log-normalizer discrepancy is 2.46e-13. Thirty-two selected final
poses also pass an independent atom-union distance audit, with the smallest
checked gap 0.0002047 Å. This selected check does not certify every possible
pose or the floating-point implementation as a whole.

## Why uniform initialization missed the shoulder

The [confirmed initialization diagnostic](/home/xvg/tetramer-mc/runs/smc-shoulder-control-20260920/initialization-coverage-confirmed.json)
checks all eight actual old **other** configurations: each has
`uniform_probability=1`, `proposal_components=[]`, and every saved endpoint
has `log_g=-log(Vcapture)`. The .01 mixture belongs to the old **native**
configurations and is not the old competing-region proposal.

The new zero-activity control estimates hard q<=2 volume as 0.00041513 Å³,
with a binomial 95% confidence interval 0.00036138–0.00047463 Å³. Uniformly
drawing 1,048,576 poses in the 18 Å capture ball/full-Haar domain would thus
produce only **0.01782 q<=2 hits per population** (propagated interval
0.01551–0.02037). At that volume estimate, eight populations have an 86.7%
probability of no initial q<=2 hits. Excluding native poses only makes shoulder
initialization sparser. Subsequent mutations could still enter the region;
this calculation concerns initialization alone.

The new cap raises proposal density there by about 12,010 times. This is a
proposal-density ratio, not a demonstrated mixing or CPU speedup. Raising
N from 512 to 2048 with the same fixed initialization budget also leaves the
number of independently hit initial poses largely unchanged; it supplies
more mutation descendants. A later initialization-budget sensitivity test
would address that distinction separately. The present pilot leaves the full
q>2 competing space, AB neighborhood, and assembly conclusions unresolved.
