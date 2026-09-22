# A separate six-dimensional contact-guide pilot

The completed [conditional-ray campaign](conditional-ray-reference-results.md)
failed its fixed convergence criteria. Its 786,432 attempts remain immutable;
they are not extended, restarted, or pooled with a new estimator. The physical
question remains unresolved. The next experiment uses those saved poses as
training data for a different, explicitly frozen importance proposal.

The repaired shape, both fixed neighbors, radius 1.5 Å, activity 0.035 Å⁻³,
R4 region, physical measure, and complete native observer are unchanged.
This is a native-informed integration reference, not geometry-only discovery
or an assembly trajectory. It does not alter the validated assembly kernels.

## Why couple all six coordinates?

The ray guide improves allocation only along a translation ray after choosing
orientation and direction. Its larger-population native estimate was dominated
by a fallback draw. The saved native extremes occupy the cooperative A7/B4
geometry, while the strongest contact-without-entry examples lie just beyond
the two 2 Å member-error gates. These are descriptive selected-pose observations,
not demonstrations that a contact family is connected or that its weight has
converged.

A separate saved calculation measures an independently defined R5 native
pocket. Restricting its original attempts to the present R4 gives a subset
estimate of log Qz = 60.6821, observed population RSE 7.97%, and importance
ESS 169.8. All 5,064 contributing native rows satisfy the present R4 predicate;
the denominator remains all 131,072 original attempts. Its mass exceeds the
large-uniform estimate for the entire current native region by 4.815 log units.
This indicates missed native weight. The finite-sample subset estimate is
uncertain; it is neither a rigorous numerical lower bound nor the whole R4
integral. No assumption that the entire geometric R5 lies inside R4 is needed.

The [saved-pose diagnostic](../runs/mobile-conditional-ray-extremes-20260922/report.md)
records the actual intersection, member distances, and Poisson-cloud noise.
The new proposal is also checked against these saved native-pocket poses
before its fresh physical evaluation. Such checks remain training/coverage
diagnostics and do not count as fresh validation.

## First construction: frozen without launching physics

Use original whitened latent coordinates u from the exact R4 chart. Retain
one observed hard-valid anchor for each of the 24 source populations and each
of the two complete classes: native entry and contact without native entry.
The anchor is that population's largest original importance-weight pose in
the class. This gives 48 components, with no source-population slot deleted.

For each anchor a, select the 64 nearest distinct same-class source poses in
Euclidean u distance. Use fitting log weights log w minus log N_source, then
water-fill the normalized weights to a maximum of 1/16. The cap prevents a
single noisy source observation from determining the fitted covariance. It
does not modify any physical estimator or held-out scoring weight.

Keep a as the Gaussian mean. If the capped neighbor mean and covariance are
μ and Σ, use

\[
\Sigma_a=\Sigma+(\mu-a)(\mu-a)^T+L^{-1}F_{\rm raw}L^{-T}.
\]

Here L is the original chart's Cholesky factor. The physical geometric floor
uses 0.05 Å translation standard deviation and the existing 0.1° angle-axis
scale convention. Transforming this floor into u is essential: raw translation
and angular coordinates are correlated under the original chart.

The full normalized proposal is

\[
q(u)=\frac{1}{2}\frac{1_{R4}(u)}{V_6(4)}
     +\frac{1}{2}\frac{1}{48}\sum_{j=1}^{48}
       \mathcal N_6(u;a_j,\Sigma_j).
\]

The native and no-entry banks each receive 25% of proposal probability. The
second arm uses the same centers and weights with every covariance multiplied
by four. Both widths are retained regardless of leave-one-population-out scores.
Gaussian components are globally normalized and untruncated. An exterior draw
is a zero-weight attempt, never a reason to retry or renormalize the proposal.

The saved-reference coverage check failed: the weighted log-density gain over
uniform was 4.46/3.64 for the two widths, but their densities remained 7.29/8.12
log units below the old pocket proposal. Its noisy, optimistic second-moment
proxy predicted fewer than 0.1 effective native-pocket contributions per
100,000 attempts. Positive leave-one-population-out scores on the ray campaign
therefore did not establish useful coverage of a known important region.
The [48-component preparation](../runs/contact-bank-guide-preparation-20260922/README.md)
is retained unchanged and **no physical pilot was launched from it**.

## Separate reference-augmented construction

Before any fresh physical validation, add eight observed native-pocket anchors,
one per independent R5 source population. These are additional training data,
not new normalizer observations. Use the same 64-neighbor, capped-weight,
anchored-covariance construction on saved R5 contributors expressed in the
current original u coordinates. Remove a whole R5 population from its anchor
and neighbor pools for each held-out diagnostic; the old 48-component bank
uses independent source data and can stay fixed during those folds.

The new proposal still allocates 50% uniformly over R4. It assigns 12.5% to
the 24 original native components, 25% to the 24 no-entry components, and
12.5% to the eight measured-pocket components. Their normalized weights
inside the Gaussian half are respectively 1/96, 1/48 and 1/32. The 56-component
bank and wide guides share centers and allocations; wide multiplies every
covariance by four. The old preparation is not edited or retrospectively
called successful. This augmentation is a separate, documented response to
its coverage failure, made before any fresh pilot outcomes.

For each width, the augmented density satisfies the pointwise bound
q_aug(u) ≥ q_old(u)/2. Thus its importance second moment per draw is at most
twice the original guide's for any fixed reporting region and unchanged
conditional cloud law, whenever that moment exists. This is an exact allocation
bound, unlike a correlation-based coverage score. It neither proves that the
original guide was adequate nor bounds CPU cost or unseen physical mass.

## Correct estimator and diagnostic separation

After freezing the proposal from the source data D, draw fresh u independently
from q_D, and two independent nonnegative Poisson estimators W1 and W2 with
conditional expectation exp(z C(u)). For each fixed physical region B, use

\[
\widehat Q_B=\frac1N\sum_{i=1}^N
 1_{R4}1_{\rm capture}H1_B\frac{J(u_i)}{q_D(u_i)}
 \frac{W_{i1}+W_{i2}}2.
\]

Conditional on D this has the required expectation. Integrating over D also
preserves that expectation. No learned mixture weight is interpreted as a
physical region mass. Every exterior or hard-invalid zero stays in N.
The independent Python audit reconstructs the full mixture q and physical J
from each saved pose, including exterior draws.

Leave-one-population-out fitting removes the entire held-out population from
both anchors and neighbor selection. Its original physical importance weights
are used without clipping for likelihood/coverage diagnostics. Low source ESS
makes these ratio diagnostics uncertain. The independent cloud product permits
the additional second-moment diagnostic

\[
 E_{u\sim q_{old},W_1,W_2}
 \left[\frac{1_BJ^2W_1W_2}{q_{old}q_{new}}\right]
 =\int_B\frac{J^2e^{2zC}}{q_{new}}\,du.
\]

This removes within-pose Poisson noise from the expectation of that numerator;
it does not remove finite-source pose variance or certify unseen-mode coverage.

## Separate fixed allocation

The reference-augmented arms receive four fresh populations of 16,384 unconditional draws:
131,072 new attempts in total. Seeds are 132101010 + 1009 i for i = 0,…,7.
Each valid pose receives two independent clouds at auxiliary intensity
λ/z = 128. At most eight single-thread physical jobs run simultaneously.
The two archived independent audits follow completion of all eight jobs.

The controller preserves the existing Gaussian-guide and population formats,
freezes the executable/source/input closure, rejects overwrites, drains all
launched jobs after failure, and checks every attempted-draw denominator.
The analyzer uses the unchanged full native classifier and exact exclusion
contact predicate. It retains original radial bins [0,2), [2,3), [3,4], angular
squared-radius bins [0,4), [4,9), [9,16], all 64 latent orthants, and explicit
exterior attempt counts.

Apply the prior regional precision thresholds as screening diagnostics, and
compare the two widths as separate estimators. Even a successful pilot still
requires fresh larger populations, intensity/defensive-floor sensitivity,
and adequate remaining-domain coverage. It cannot automatically authorize
the full-vessel or finite-assembly production gates. A failed pilot is another
sampling limitation, not evidence that the physical model prevents assembly.

## Additional named pocket/complement diagnostic

Before examining the fresh weights, a [separate observer specification](../runs/contact-bank-reference-partition-plan-20260922/protocol.json)
was frozen to split native R4 mass into the old R5 intersection and its native
complement. It uses the exact old radius-five chart, original physical metric
q>1 and D170 support, and retains every fresh attempt with its existing J/q
weight. There is no second Jacobian multiplier and no valid-only denominator.
This supplements the primary radial/angular strata; it does not change their
frozen gates. Both original R5 and ray data have been used in guide design.

## Later full-vessel mixture: the density contract

If the regional checks pass, the normalized physical mixture can use

\[
p_{mix}(x)=\tfrac12 p_{vessel}(x)
            +\tfrac12 q_D(u(x))/J(u(x)),
\]

where p_vessel includes every anchor, reciprocal branch and uniform component
of the existing proposal. The Gaussian guide is globally normalized: only its
uniform R4 term vanishes outside R4. For the full-vessel target, a hard-valid
exterior-R4 guide draw inside the atomic wall contributes normally. Wall-invalid
and hard-invalid attempts remain zeros. Do not truncate this guide or divide
by an unknown R4 probability. The physical importance weight is
H_wall H_hard Wbar/p_mix; the R4 indicator is only a reporting restriction.
This requires a separate implementation and fresh validation, and remains gated.
