# Reweighting existing components: aggregate gains do not cure weak strata

The [fresh guide pilot](smc-guide-pilot-results.md) improved native coverage but
allocated too little proposal weight to competing contacts. A bounded offline
diagnostic tests whether changing **only mixture weights** can improve that
tradeoff. It reuses saved poses and clouds; it generates no new physical samples,
classifications, operational proposal files or campaign allocations.

Keep all 84 Gaussian means and covariances and the 50% uniform component:

\[
q_a(u)=\tfrac12 U_{R4}(u)+\tfrac12\sum_{j=1}^{84}a_jG_j(u),
\qquad \sum_j a_j=1,\quad a_j\ge10^{-5}.
\]

The Gaussians are untruncated. Outside-region attempts retain zero physical
weight. All components retain positive mass, and every original attempted draw
remains in the denominator. The changed proposal is normalized; physical
importance weights would still use its complete density and the original
Jacobian and target.

The fit uses r00/r01 from each fresh source arm, 262,144 original attempts in
total. The other four populations, also 262,144 attempts, are held out from the
fit. **Their earlier diagnostics were already inspected**: this is a design
control, not pristine independent validation. New production would require a
separately frozen proposal and fresh samples.

For each of the three regions (native inside R5, native outside R5, and contact
without native entry), estimate candidate second moments with the original
source-density correction. Minimize the largest candidate/original moment ratio:

\[
\min_a\max_R
 \frac{\widehat M_{2,R}^{\rm train}(q_a)}
      {\widehat M_{2,R}^{\rm train}(q_{80})}.
\]

Each numerator is a nonnegative weighted sum of reciprocals of affine positive
densities, so the objective is convex in the weights. Normalizing both moments
by a training-only estimate of the region mass squared gives the same ratio.
No holdout statistics enter the fit. The weak orthants are report-only checks.

## Saved-sample result

A preliminary grid varied the total weight of the four new Gaussians while
preserving relative weights within the old and new groups. No grid value
improved all three aggregate regions in training or either holdout source.

Allowing all 84 weights to vary did. SLSQP converged in 69 iterations, with
training moment ratios approximately 0.66275 for each region. The fit assigns
**7.55% of Gaussian mass** to the four new components, compared with the pilot's
50%. Thus simple group allocation alone misses useful redistribution among the
original components.

| Region | Fitted/original M₂, original-proposal holdouts | Fitted/original M₂, new-proposal holdouts |
|---|---:|---:|
| Native inside R5 | 0.641 | 0.697 |
| Native outside R5 | 0.847 | 0.500 |
| Contact without native entry | 0.511 | 0.949 |

These aggregate estimates improve in both reserved source groups. They are not
variance guarantees or measured trajectory speedups. One individual new-source
holdout population has competing-contact ratio 1.028; the improvement is not
uniform across populations.

More importantly, several previously unstable subdivisions become worse:

| Competing-contact orthant | Fitted/original M₂, original-source holdouts | Fitted/original M₂, new-source holdouts |
|---|---:|---:|
| 30 | 1.089 | 1.661 |
| 34 | 1.006 | 0.620 |
| 42 | 1.626 | 0.761 |
| 50 | 0.298 | 0.875 |
| 62 | 1.863 | 1.769 |
| 63 | 0.515 | 0.557 |

For new-source orthant 62, one draw supplies about 99.2% of the fitted proposal's
estimated second moment (moment-contribution ESS about 1.02). These are
diagnostics of **squared-weight concentration**, not ordinary importance ESS.
They do not establish that unseen tails are bounded. No further fitting was
performed after examining the holdouts.

The result supports including component allocation in proposal design. It also
shows why optimizing three aggregate regions alone is insufficient for the
declared subdivision requirement. This candidate is **not promoted to a
physical campaign**. The failed regional gate and unresolved thermodynamic
conclusion are unchanged.

## Artifacts

The [pre-execution diagnostic plan](../runs/contact-guide-reweighting-diagnostic-20260923/plan.json),
[reproducible script](../runs/contact-guide-reweighting-diagnostic-20260923/diagnose.py)
and [complete results](../runs/contact-guide-reweighting-diagnostic-20260923/analysis.json)
retain the training/holdout split, all source hashes, reference-density checks,
optimizer status, all 84 fitted weights, individual-population results and
both two-cloud and independent-cloud-product moment diagnostics. They are
proposal-design artifacts; they do not alter completed physical evidence.

The [verification receipt](../runs/contact-guide-reweighting-diagnostic-20260923/verification.json)
binds the pre-execution plan and analysis and records explicitly normalized
weights. The optimizer simplex residual was 4.73×10⁻¹¹; normalization changes
reported moment ratios by less than 4.73×10⁻¹¹. Independent reconstruction
reproduced the unnormalized held-out ratios within 2.1×10⁻¹⁴ and checked the
objective derivatives. This is numerical verification of the diagnostic, not
validation of an operational proposal or a bound on physical unseen mass.

A subsequent [tail and protected-subdivision allocation diagnostic](contact-tail-allocation.md)
shows that existing components cover the observed troublesome rows and tests
a training-only objective protecting important orthants as well as aggregates.
It remains an offline design experiment, with no change to physical gates.
