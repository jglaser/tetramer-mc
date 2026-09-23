# Fresh guide pilot: better native coverage, unresolved competing contacts

The frozen pilot completed **524,288 attempted draws in eight independent
populations**, with all density/Jacobian audits and first-pass classifications.
It compares the original 80-component proposal with the frozen 84-component
proposal containing geometry learned from earlier SMC. Both retain 50% uniform
sampling, two independent clouds per valid pose and intensity/activity ratio
128. The repaired shape, 1.5 Å depletants, activity 0.035 Å⁻³, R4 domain,
two-anchor scaffold, full native classifier and physical measure are unchanged.
The new calculation is importance sampling, not a new SMC trajectory.

**The prescribed convergence checks still fail.** The original confirmation is
not superseded, and full-vessel and finite-system assembly production remain
unlaunched. The physical conclusion is an unresolved sampling limitation.

![Observed ESS and same-pose second-moment comparisons](../runs/smc-guide-pilot-completed-review-20260923/report/guide-efficiency-diagnostic.png)

## What the independent pilot establishes

The following are logarithms of arithmetic means of four independent linear
mass estimates. Invalid/exterior draws retain zero weight in the original
unconditional denominators. The historical R5 intersection and its complement
are reporting subsets of the unchanged complete native region in R4.

| Region | Original proposal log Qz | New proposal log Qz | Observed ESS/CPU ratio, new/original |
|---|---:|---:|---:|
| Complete native entry | 61.288579 | 61.246771 | 1.908 |
| Native inside historical R5 | 60.679645 | 60.657651 | 1.681 |
| Native outside historical R5 | 60.503470 | 60.437506 | 2.014 |
| Contact without native entry | 42.564907 | 42.557959 | 1.315 |

All observed aggregate depleted-mass comparisons pass the 0.2-log-unit and three-combined-
population-SE criteria. The native complement accounts for about 45.6% and
44.5% of observed native weight. All significant native-complement subdivisions
agree between the fresh arms. These observations support improved coverage of
the subset that previously caused trouble.

The table's efficiency ratios are observed **importance-weight ESS per sampler
CPU**. They are neither contact-trajectory ESS nor demonstrated mixing speedups.
Their interpretation needs a second diagnostic because rare, large weights
can reverse an observed ESS ranking.

For a source proposal q_s and candidate q_t, reuse the saved physical estimator
to estimate the candidate's second moment:

\[
 M_2(q_t)=E_{q_s,\omega}\left[
 I_R\,\widehat w_s^2\frac{q_s}{q_t}\right],
 \qquad \widehat w_s=J\overline W/q_s.
\]

This compares candidates on identical source poses and clouds. Replacing
the square of the two-cloud mean with the product of the two independent
cloud weights estimates the physical second moment without cloud-noise bias
in expectation. Neither form removes finite-sample tail uncertainty.

| Region | M₂(new)/M₂(original), original-source data | Same ratio, new-source data |
|---|---:|---:|
| Native inside R5 | 0.689 | 0.713 |
| Native outside R5 | 0.598 | 0.407 |
| Complete native entry | 0.630 | 0.494 |
| Contact without native entry | 1.969 | 1.970 |

Both sources support the native benefit. Both also indicate that the new
proposal **sacrifices competing-contact coverage**. Independent-cloud products
give competing-contact ratios 1.972 and 1.964. The absolute competing-contact
second moment evaluated for the same original proposal differs by 2.45× between
source arms; one source draw supplies 41.7% of that moment estimate. Thus the
observed 1.315× competing-contact ESS/CPU gain cannot establish an algorithmic
improvement. The ratios above remain estimates, not rigorous variance bounds.

## Failed checks remain failed

Three checks fail:

1. Original-proposal competing-contact quality: the largest draw supplies
   **2.164%**, exceeding the fixed 2% limit. Importance ESS is 889.5 and
   population RSE is 3.07%; those passing quantities do not override the failure.
2. Hard-only native mass: the log difference is **0.05447**, or **4.33 combined
   population SE**, exceeding the three-SE criterion. The native complement
   has the same failure.
3. **Seven of 35 significant subdivision comparisons** fail: competing-contact
   orthants 30, 34, 42, 50, 62 and 63, and historical-R5/native orthant 51.
   Competing orthant 30 fails both the absolute and three-SE rules. The other
   six fail the absolute rule. The [generated report](../runs/smc-guide-pilot-completed-review-20260923/report/report.md)
   preserves each difference and uncertainty.

The hard-only discrepancy also appears when using only the uniform proposal
branch with the Horvitz–Thompson contribution
`I_uniform * I_valid * J * V_R4 / alpha`, divided by the **original attempted
draw count**. All other-branch and invalid rows remain zero. This control does
not use Gaussian-density weights. Its log difference is 0.05859, or 4.66
population SE. Corresponding row-based SEs make the ordinary and uniform-only
differences 1.70 and 1.78 SE, respectively. The four population means happen to
have much less scatter than the individual-draw variance suggests. This makes
ordinary sampling fluctuation a plausible explanation; it does not prove
absence of bias or authorize replacing the prescribed population criterion.

The conditional native-minus-competing free energies are −18.7237 and
−18.6888 kBT, with approximate paired-population 95% halfwidths 0.0838 and
0.0708 kBT. Those small intervals do not settle the failed tail and subdivision
checks. No unbound pose was observed; only the existing finite-R4 bound applies,
not a bound on the full-vessel complement.

## Relation to earlier SMC and the next calculation

The new proposal's native-complement log mass, 60.437506, is closer to the
matching narrow-SMC estimate, 60.275732, than the earlier importance estimate.
Their difference, 0.161775 with combined SE 0.160240, passes both aggregate
criteria. Significant complement orthant 51 still differs by 0.288502 log units.
The narrow-SMC control has no terminal competing-contact observations, so it
cannot independently confirm that denominator. Agreement on a subset is not
resolution of the full free-energy contrast.

A restricted SMC control could prevent native dominance by targeting

\[
 \gamma_\beta(dx)=H_{R4,\,hard,\,capture}(x)
 [1-I_{\rm native}(x)]g(x)^{1-\beta}e^{\beta z C(x)}d\mu.
\]

Its initializer would retain all attempted draws and multiply by the exact
non-native predicate. Mutations would reject native entry before the depletion
gate. The denominator would be the SMC normalizer times the terminal exclusion-
contact indicator; unbound contributions would remain explicit. This requires
the **complete** native classifier inside the sampler, parity with the independent
Python classifier and recording/auditing every operative membership decision.
Rust's current body-registration metric is not an equivalent predicate.

This is a possible independent control, not an implemented or validated speedup.
Existing narrow SMC cost about 118,259 CPU seconds, versus 4,964 for the fresh
original-proposal importance arm. Those workloads differ and do not predict
restricted-SMC cost. A cheaper immediate design diagnostic is to reallocate the
existing Gaussian components using saved samples, retaining the defensive
component and exact density. Any such fitted candidate needs a separately
frozen fresh evaluation; retrospective optimization cannot repair this result.

The subsequent [fixed-component reweighting diagnostic](contact-guide-reweighting-diagnostic.md)
improves the three aggregate moment estimates on populations held out from
fitting, but worsens several unstable competing-contact strata. No operational
proposal or new physical campaign was promoted from that diagnostic.

## Reproducibility

- [Frozen pilot protocol](../runs/smc-geometry-guide-pilot-20260923/protocol.json):
  `03778e707ca08759c1401ad5949a52bfe9c1857f1964eacddecaf9ab9dbdbab5`.
- [Completed primary analysis](../runs/smc-geometry-guide-pilot-20260923/comparison/analysis.json):
  `41264f95c98e3dce7d6748ee32bb80aee666b82e8e0fd8b1e351d875e0c846be`.
- [Authentication of 444 files and terminal process records](../runs/smc-guide-pilot-completed-review-20260923/authentication.json):
  `07e3c0bee824524c687f4c497960d4c6dacd64fad85fedc37e4e8acb28929c1b`.
- [Saved-sample moment and uniform-branch diagnostic](../runs/fresh-guide-noise-diagnostic-20260923/analysis.json):
  `291b15fde0706349e914e280049923f8200e502667c51681d2ab78df092c241b`.
- [Machine-readable figure/report summary](../runs/smc-guide-pilot-completed-review-20260923/report/summary.json)
  binds inputs, code, displayed numbers and outputs.

`tools/summarize_smc_guide_pilot.py` generates the report from these completed
artifacts into a fresh directory. Its input checks authenticate the moment
diagnostic and its sources against the completed primary analysis. It launches
no physical work and replays no classification or density audit. The original
campaigns, failed gates and user configuration are preserved.
