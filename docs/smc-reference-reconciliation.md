# SMC and independent contact integrals: what is comparable

The earlier site-0 SMC and the current AB calculation have different physical
neighborhoods. SMC used **one fixed tetramer A**; the current native and
shoulder references use **A and B**. The change from a native log Q near 15
to 35.77 is therefore not a same-target discrepancy. The unresolved SMC
disagreement concerns competing contacts in the **one-neighbor** problem.
This reconciliation reads existing configurations, summaries and source; it
does not rerun physical simulations or certify their convergence.

## Physical target and measure

For a fixed physical neighborhood S, all comparisons should refer to

```
Q_S(B) = integral_capture H(x; S) 1_B(x) exp[z C_S(x)] d³t dHaar(R),
C_S(x) = |E(x) intersect union_{j in S} E(j)|.
```

The inspected operative configurations agree on the 4004-sphere shape
(SHA256 `c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9`),
depletant radius 1.5 Å, activity 0.035 Å⁻³, capture center and radius 18 Å,
rigid-member geometry, native references and original 2 Å / 15° q metric.
Both the importance-guided and MIS-refined one-neighbor configurations have
exactly the archived SMC fixed-pose list. The AB configuration retains A and
adds B. Its native reference and q scales remain unchanged. These checks use
the [actual archived SMC config](/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/configs/site0-m1-r1.5-z0.035-native-00.json),
[one-neighbor importance config](/home/xvg/tetramer-mc/runs/basin-normalizer-importance-guided-16384-l64/provenance/config.json),
[MIS config](/home/xvg/tetramer-mc/runs/basin-normalizer-mis-refined-16384-l64/provenance/config.json)
and [AB config](/home/xvg/tetramer-mc/runs/native-ab-refined-selected-8x16384-l64-20260920/runs/r00/provenance/config.json).

The measure is center volume in Å³ times **normalized proper SO(3) Haar**.
For the current Cayley chart `x=(translation, ell*c)`, the proposal density
contains `ell³ π² (1+|c|²)²`; a uniform latent reference instead multiplies
its contribution by the inverse factor and its affine determinant. The old
SMC rotational cap uses Haar fraction `(alpha-sin(alpha))/pi`. No extra
`8π²` factor is required between these conventions. Current original
importance weights already include the proposal/Jacobian correction.
[Chart formula and independent checks](/home/xvg/tetramer-mc/docs/native-ab-covariance-guide-plan.md).

The archived `reference_activity=0.0008430997064 Å⁻³` sets auxiliary count
intensities; it is not the final bath activity. In the archived SMC path,
`lambda=smc_lambda_ratio*reference_activity`, `c=1+z/lambda`, and the
intermediate physical activity is `lambda*(c^t-1)`, which equals z at t=1.
The new direct estimates use `lambda=64*z` and average two independent
positive Poisson weights. These different auxiliary laws target the same
`exp(z*C)` in exact arithmetic. [Archived path and count argument](/home/xvg/tetramer-mc/docs/previous-smc-region-audit.md).

## Same-neighbor numerical comparisons

The table uses **A only** throughout. Errors are observed relative standard
errors, not bounds on unseen weight. The SMC errors use independent whole
populations; the importance errors use original independent draws.

| Fixed region / estimand | Archived SMC | Independent estimate | Interpretation |
| --- | ---: | ---: | --- |
| Native q≤1 | log Q 15.07941; 5.16% | 14.97729; 5.27% | Compatible; using both methods' population errors, the difference is 1.16 combined standard errors on the linear Q scale. |
| Shoulder 1<q<2 | Zero observed terminal contribution | 16.88668; 6.82% | Every archived other endpoint misses this region. A second independent proposal gives 16.98337; 9.40%. |
| Whole other q>1 | 15.10263; 34.97% | MIS full-support estimate 21.84060; 3.99% | Full other mass remains unresolved; the independent proposal's observed error cannot bound omitted contacts. |
| Frozen far-contact region r≤8, original q≥5 | No directly matched archived estimate reported | 21.83045; 4.03% | A fixed subset, not the whole other domain. Its point estimate is 835 times the old whole-other estimate. |

The [archived population audit](/home/xvg/tetramer-mc/runs/normalizer-reference-audit/audit.json)
contains eight 512-particle populations per site/basin. The independent
values above come from the [importance-guided assessment](/home/xvg/tetramer-mc/runs/basin-normalizer-importance-guided-16384-l64/assessment/analysis.json),
[MIS assessment](/home/xvg/tetramer-mc/runs/basin-normalizer-mis-refined-16384-l64/assessment/analysis.json)
and its [fixed far-region assessment](/home/xvg/tetramer-mc/runs/basin-normalizer-mis-refined-16384-l64/assessment/deep-shells.json).
The complete uniform native cover independently gives log Q=15.26037 with
29.72% observed error; it is compatible with both native estimates and remains
noisy. [Native cover assessment](/home/xvg/tetramer-mc/runs/native-region-reference-8x4194304-l64-20260920/assessment-streaming.json).

The shoulder estimate alone is 5.95 times the old whole-other estimate.
Exact masses must obey subset monotonicity; noisy estimates need not.
Together with the unrepresented terminal region, this identifies a material
coverage problem in the old reference. It does not supply a converged
replacement for the entire competing domain. In particular, an empty
terminal histogram is not a zero physical mass or an upper bound.

Fresh explicit-far SMC, with original q>5 enforced throughout, gives
population log Q values **17.5310, 27.2731, 17.5823, 14.1615**. Their linear
mean has log Q=25.88689 and 99.98% observed relative error; one population
supplies 99.9877% of its weight and retains one ancestry family.
[Completed assessment](/home/xvg/tetramer-mc/runs/explicit-far-smc-plan-20260920/initial-production-assessment.json).
This is neither agreement with the old SMC result nor a converged new far
normalizer. The fixed r≤8 estimate is only a subset and need not equal it.

The old `other_adsorbed` label means every q>1 pose in capture, including
unbound poses. It is not the later q≥2 or q≥5 contact-core definition.
For a fixed region B, terminal filtering yields the unnormalized estimator
`Qhat_B=Zhat*mean(1_B)`, not just the endpoint fraction. Applied to q≥5 it
happens to equal each old other normalizer because all saved other endpoints
pass. Relabeling that filtered estimand as far permits adding independently
estimated disjoint q<5 regions; adding them to a quantity still labeled
whole-other would double count in expectation. A region selected from the
same endpoints also needs selection qualifications or fresh validation.
[Exact fixed-schedule argument](/home/xvg/tetramer-mc/docs/previous-smc-region-audit.md).

## What the AB results establish separately

The new AB native estimate is log Q=35.77224 with 2.20% observed error, while
the AB shoulder mixture confirmation gives 25.0069 with 9.18% observed
error. Neither number is a direct check of the A-only SMC figures above.
The full AB competing space, native/shoulder tails, and the cost of forming
the fixed neighbors remain separate requirements. The old deep-contact
region's exclusion by B is a physical support change, not a normalization
correction. [AB native assessment](/home/xvg/tetramer-mc/runs/native-ab-refined-selected-8x16384-l64-20260920/assessment-streaming.json),
[AB shoulder controls](/home/xvg/tetramer-mc/docs/shoulder-proposal-guides.md),
[continuous clash enclosure](/home/xvg/tetramer-mc/docs/frozen-deep-region-clash-certificate.md).

The geometry audit found no mismatch between archived/current membership
predicates on 12,582,912 common random points at twelve selected endpoints.
The subsequent archived Poisson-count confirmation agrees with independent
overlap estimates at three selected poses, within 1.75 combined standard
errors. These checks make a detected multi-kBT geometry/count-law change an
unsupported explanation at those poses. They do not validate the complete
SMC normalizer or every mutation kernel. [Geometry audit](/home/xvg/tetramer-mc/docs/old-new-geometry-audit.md),
[count-law audit](/home/xvg/tetramer-mc/docs/archived-smc-count-audit.md).

## Smallest next quantitative control

A configuration-only **A-neighbor q≤2 SMC** can directly test the missing
shoulder before another unrestricted far search. The archived executable
already supports the needed native cap, basin filter and terminal estimator:

1. Keep shape, A, capture, z, depletant radius and all physical interactions.
   Set `basin="native"`, `member_error_scale=4` and
   `angle_error_scale_deg=30`; internal q is then original q/2. Preserve the
   original metric separately for reporting.
2. Replace the explicit `proposal_components` override with a single
   unchanged-reference product ball of outward-rounded radius 4 Å and
   Haar cap 30°, retaining the positive uniform capture-ball branch. Changing
   only `proposal_position_radii` would have no effect while the explicit
   override remains present. Exactly centered rigid members make this
   product an envelope of the entire original q≤2 region. With uniform
   probability .01, g is constant on that target, approximately
   0.491618 Å⁻³, so proposal flattening is especially simple.
3. Use fresh independent populations and a frozen schedule/budget. Report
   `Zhat*mean(original_q≤1)` and `Zhat*mean(1<original_q<2)`, retaining
   zero-hit populations. The point sum from the existing matched independent
   source is log Q(q≤2)=17.02485; its two pieces are correlated because
   they use the same original rows, so error propagation must retain that
   covariance.
4. First validate the rescaled mask and zero-activity normalization with the
   archived analytic controls. Then compare whole-population variance,
   initial family diversity and a larger population size. Agreement for
   this compact target would test the missing-shoulder explanation; it
   would still leave q≥2, the AB environment and assembly unresolved.

This reconciliation originally proposed the control. The subsequent
[implementation and validation](smc-shoulder-control.md) freezes its
configurations and records the analytic controls and independent protein
populations. The capture-uniform branch in the archived executable is a
**ball**, whereas the current atlas's uniform branch is a cube; their
normalized densities are evaluated according to their actual proposal laws.
The control avoids a new region-predicate implementation while testing a
specific same-target discrepancy. A later frozen latent-region SMC control
could compare directly with the independently measured far-contact region,
but would require a new chart-region predicate or carefully qualified
terminal filtering.

## Outcome of the compact control

The [completed A-only q≤2 pilot](smc-shoulder-control.md) recovers both pieces:
native log Q=15.04277 (21.81% independent-population SE), shoulder
16.97172 (33.02%), and total 17.10739 (30.05%). Its four N=512 populations
agree with both independent proposal references in all three regions,
within 0.29 combined observed standard errors on linear Q. The shoulder
deficit therefore does not persist when that compact target is explicitly
covered. Four populations and substantial ancestry loss leave this SMC
estimate much less precise than the direct integration.

The actual old `other_adsorbed` configurations use uniform probability one
and no explicit components: initialization is uniform over the capture ball
and full Haar orientations. The zero-activity control estimates that only
about 0.01782 q≤2 poses would be hit per old population's 1,048,576 draws,
even before native poses are excluded. The new cap raises density there
by a factor of about 12,010. This gives a quantitative initialization-coverage
explanation; it does not prove how much subsequent mutation can repair
missing regions. The controlled pilot supplies that additional evidence
for q≤2. The full competing capture region, the AB environment and
equilibrium mixing remain separate requirements.
