# Full-wall sphere reference: failed precision gate

The fixed reference completed all **16,384 attempted poses** and all eight
independent saved-row audits. It **failed** the predeclared contact point-accuracy
check at nonzero activity. This does not admit the new class-to-vessel adapter
to the protein vessel campaign, detect a statistically resolved bias, or provide
evidence against protein assembly.

The reference uses a toy union sphere, not protein conditions: four coincident
radius-0.3 atoms, depletant radius 0.5, activities 0 and 0.4, a wall of radius 3,
and a source capture of radius 1.4. Each activity has four independent populations
of 2,048 attempts and two Poisson clouds per valid pose. Normalized rotational
Haar measure is retained even though the physical sphere is isotropic. The source
R4 region does not restrict the target.

The eight populations retained 10,966 valid poses, 5,418 invalid zeros and 21,932
clouds. Of the valid poses, 4,313 lie outside R4. Sampler CPU totaled 3.87 seconds;
that excludes the separate Python audits and analysis. No retry, replacement,
extension, or new protein draw was performed.

## Analytic comparison

For center separation \(r\), the exclusion-overlap volume is

\[
V(r)=\frac{\pi(3.2+r)(1.6-r)^2}{12}\quad(r<1.6),
\qquad V(r)=0\quad(r\ge1.6).
\]

The full wall permits \(0.6\le r\le2.7\), so each radial region has mass
\(Q_A(z)=4\pi\int_A r^2e^{zV(r)}dr\). Hard-only and unbound masses use closed
forms; depletion integrals use split Simpson quadrature with a real-arithmetic
fourth-derivative remainder bound and a separately declared floating-point
allowance. An independent quadrature comparison is among the six passing tool
tests. The original criteria require both four-population-SE compatibility and
an absolute point log discrepancy at most 0.1.

| Contact estimator, z=0.4 | Estimated mass | Analytic mass | Population SE | Absolute log discrepancy |
| --- | ---: | ---: | ---: | ---: |
| Exact sphere weight | 20.8787 | 17.7815 | 3.4200 | 0.16057 |
| Mean of two Poisson weights | 20.9019 | 17.7815 | 3.4470 | 0.16168 |
| Hard-only companion on the same poses | 19.1626 | 16.2525 | 3.0395 | 0.16471 |

These are the three failed checks, and they are strongly correlated. Each
discrepancy is below one population SE, but that does not satisfy the fixed
point-accuracy requirement. All other original mass checks pass. Passing a
point check by chance does not establish high precision: even the z=0 contact
estimate has low importance ESS.

The noisy-minus-exact contact estimate is 0.02322, with conditional SE 0.12803
calculated from the Poisson generating function. Thus Poisson noise is small
compared with the observed pose-integration variability. All original paired
Poisson residual checks pass. Independent row reconstruction, envelope bounds,
unconditional denominators and negative controls also pass. These checks do
not exclude every possible generator defect.

![Completed reference and posthoc strata](../results/native-class-vessel-sphere-diagnostic-20261004/sphere-normalization.png)

Small dots are independent population estimates; error bars are **one**
population SE, not confidence intervals. The green band shows the separate
fixed point-discrepancy criterion. Lower panels are posthoc diagnostics, not
additional campaign gates.

## Where the variance occurs

A bounded posthoc analysis reused the saved exact-weight companions and pose
metadata. Its radial/angular bins and calculations were recorded before reading
the individual contributions. It draws no poses or clouds and makes no geometry
queries. The entire original input ledger was reauthenticated.

At z=0.4, 8,192 attempts give contact importance ESS **73.7**. The largest
contribution supplies **3.77%** of the contact sum; the ten largest supply
**31.5%**. The same descriptors at z=0 are 74.4, 3.80% and 35.1%.

Only **29** z=0.4 contact poses come from the vessel's uniform branch. They
supply **43.9%** of the estimated contact mass and **91.7%** of the sum of squared
contact weights. Outside R4, the exact-weight contact contribution is
18.1835 ± 3.4297 population SE; inside R4 it is 2.6951 ± 0.0406. These are
contributions to one unconditional integral, not separately normalized targets.

The 120–180° physical-rotation-angle stratum carries 98.6% of the squared contact
weights. Its mass estimate is 14.4124 ± 3.3758, versus analytic 10.8289. The
1.4–1.6 separation annulus contributes 7.9053 ± 1.7399, versus analytic 5.6978.
Other angular bins also disagree with their analytic masses: in particular,
the 60–120° estimate is 5.3684 ± 0.0858 versus 5.9272. Small between-population
error in one sparsely covered stratum cannot rule out a shared missing tail.
These observations locate weak coverage; they do not prove the generator is
unbiased or turn the failed reference into a pass.

The source of the poor defensive coverage is explicit. The original uniform
law draws translations from a **cube of side 10**, not the wall's allowed ball.
Its total mixture weight is 0.5 × 0.4 = 0.2, giving the physical density floor
\(q_{\min}=0.0002\). The expected number of uniform-branch contact draws per
activity is only \(8192(0.2)Q_{\rm contact}(0)/1000=26.63\), consistent with
the observed 29. Most orientations near the poorly covered part of the atlas
therefore rely on a small number of high-weight defensive draws.

For an exact regional estimator \(Y=1_A e^{zV}/q\),

\[
\mathbb E[Y^2]\le Q_A(2z)/q_{\min},\qquad
\frac{\mathrm{Var}(\overline Y)}{Q_A(z)^2}
\le\frac{Q_A(2z)/(q_{\min}Q_A(z)^2)-1}{n}.
\]

For hard contact the relative-variance factor bound is 306.65, giving a 19.35%
relative-SE bound at 8,192 attempts. This allocation never had a strong
analytic assurance of the 0.1-log requirement. That explains a limitation of
the reference design; it does not relax its recorded failure.

## Next intervention

Preserve this allocation and its failed gate. A useful new proposal intervention
is a wall-supported defensive envelope, rather than repeating the same seeds,
increasing the finished allocation, or sampling until a pass occurs.

For a fixed body-frame atom center \(a\) and radius \(r_a\), draw a proper Haar
rotation \(R\), then draw \(y\) uniformly in the ball of radius
\(R_w-r_a\), and set

\[
t=c_w-Ra+y.
\]

Its normalized physical density is the reciprocal ball volume when the selected
atom is inside the wall, and zero otherwise. Conditional translation has unit
Jacobian. Every fully wall-valid protein pose satisfies that atom's constraint,
so this proposal covers the entire target even for offset, concave sphere unions.
It still needs all other atom-wall and core predicates and retains invalid
zeros. The complete mixture density must be evaluated on every draw, including
draws from the atlas outside this envelope. A deterministic atom choice can be
frozen before sampling; choosing a largest-radius atom minimizes this simple
envelope's volume.

For the coincident-sphere fixture the envelope is exactly the center ball of
radius 2.7, reducing the uniform proposal volume by a factor about 12.1 without
changing the physical target. For proteins it is an enclosing proposal, not
uniform sampling of the true allowed region. Any implementation remains optional,
preserves the old cube law and must have independent density/support checks
and a separately fixed precision-aware allocation. It is not implemented or
launched by this report. The current protein populations and their frozen
proposal laws remain unchanged.

## Reproducibility and lifecycle

- Preparation: `tools/prepare_native_class_vessel_sphere.py`.
- Analysis: `tools/analyze_native_class_vessel_sphere.py`.
- Six passing tests:
  `results/native-class-vessel-sphere-tools-validation-20261004/attempt02/validation.json`.
- Completed root:
  `/vast/xvg/tetramer-mc-runs/native-class-vessel-sphere-20261004-v2`.
- Execution plan SHA256:
  `a9c755c78fb09c2750c0fc589e2ac5e31a0640d45c35a84c70cfdcf40a5d4856`.
- Authoritative `analysis/summary.json` SHA256:
  `95f91bf96576bf69d2339fe2a5f5af6c8528c8c3569acaf3d4ab5f4876710c51`.
- Posthoc source, plan, scalar table, figures and receipt:
  `results/native-class-vessel-sphere-diagnostic-20261004`.

All sixteen producer/audit stages succeeded. The final statistics worker
completed and drained with process exit code zero, writing `complete:true,
passed:false`. The controller correctly recorded a failed final contract
(`Terminal did not pass`), with no active or unstarted job. This is a failed
scientific reference gate, not a missing calculation or a crashed sampler.
The initial preparation without the `-v2` suffix was superseded before launch
to fix status-snapshot provenance and contains zero physical draws.

The isolated optimized build preserves the Rust closure of the passing CLI
fixture. The production executable and the active native observer's executable
were not rebuilt. These records establish what was tested and what remains
unresolved; they make no finite-system assembly claim.
