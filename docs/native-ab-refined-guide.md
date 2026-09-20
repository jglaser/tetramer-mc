# Refitting the native AB guide from the larger independent population

The covariance guide has been refitted from **all 14,121 nonzero original
importance contributions** in the eight-population wider-guide confirmation,
out of 262,144 unconditional draws. The source effective weight count is
688.49 and its largest contribution is 1.312%. These data were independent
validation of the previous guide; they are now training data for this new
one. They cannot independently validate its integration performance.

The source is
`runs/native-ab-guided-wide-confirmation-8x32768-l64-20260920` and the frozen
refit is `runs/native-ab-refined-covariance-guide-20260920`.
The existing `tools/prepare_native_confirmation_atlas.py` was reused
**unchanged**. No original sample, weight, model, or physical definition was
modified, and the refit generated no depletant clouds.

## Preserved physical and statistical definitions

The target still includes both fixed neighbors A and B, the original
`q<=1` member-displacement/orientation definition, the same capture sphere,
depletant radius 1.5 Å and activity 0.035 Å^-3. Choosing A as the guide's
coordinate reference does not remove B from either hard exclusion or the
many-body depletion calculation.

Fitting pools the **original physical importance weights** from all eight
equal-size populations. It does not equalize population masses, discard
large weights, replace proposal denominators, or multiply by another Haar
Jacobian. The six-dimensional deployment coordinates are translation and
the scaled **left** Cayley rotation residual, with angular length
55.02283113084892 Å. The additive covariance floor is unchanged: translation
standard deviation 0.05 Å and rotational scale
`ell*tan(0.1 degrees/2)` in each axis.

The same prespecified candidate comparison is retained: one shifted full
covariance Gaussian, versus an 80/20 mixture with the same mean and
covariance standard-deviation multipliers 1 and 2. The broad-tail candidate
must improve at least six of eight leave-one-population-out scores, improve
their mean by at least 0.1 nat after excluding the largest gain, and improve
the pooled mass-weighted score. The folds share training sets, and these
criteria are proposal-selection heuristics rather than significance tests.

## The guide changes shape rather than merely widening uniformly

The refitted single Gaussian beats the broad-tail candidate in **all eight
held-out populations**. Broad-tail minus single weighted log-density gains
range from -0.1645 to -0.0529 nat, with mean -0.1176 nat. Its one-Gaussian
fit is therefore selected by the unchanged rule. Comparing each refit fold
to the previous frozen single Gaussian gives positive gains in all eight
populations, averaging **1.4199 nat**.

The translational mean shifts by 0.06395 Å. The scaled rotational mean
shift corresponds to approximately 0.2332° in the small-angle limit. Along
the generalized covariance directions, standard deviations change by
factors **1.072, 1.080, 1.113, 1.341, 1.548 and 2.074**. An ellipsoid of a
fixed Mahalanobis radius therefore has 5.55 times the previous volume in
the translation/scaled-Cayley chart coordinates. The corresponding
whitened ball's volume is unchanged; physical Haar volume also includes
the nonconstant Cayley Jacobian. Translation/rotation cross covariance
remains included.

As a training diagnostic, 18.27% of the original observed mass lay outside
radius 4 in the old single-Gaussian chart; only 2.07% lies outside radius 4
in the refined chart. The denominator and original weights in this
comparison are unchanged. This reorganization addresses the previously
identified tail direction, but it is not a new mass estimate or evidence
that unobserved contributions are small.

These results do **not** establish that one Gaussian completely describes
the physical basin. The comparison considers only a common-center scaled
tail, not distinct means, nonlinear coordinates, or other basins. Hard
exclusion sharply truncates the target; an untruncated Gaussian necessarily
proposes clashes. Fresh guide-width and independent geometric-reference
checks remain necessary.

## Frozen outputs, checks, and cost forecast

`model.json` contains the selected Gaussian. `model-wide.json` preserves its
mean and weights and multiplies its covariance by four as a prespecified
sensitivity control. Their SHA-256 hashes are:

| File | SHA-256 |
| --- | --- |
| `model.json` | `cbfd4a5b164199f0215c64cf5bea75ca381929f65e688b03321469622b4df590` |
| `model-wide.json` | `bb18ecec258f9548c93fc1e991a243f7dcb31366c57d9aef5998c8b886ddde78` |

The source campaign hashes, all eight sample hashes, fixed geometry, original
metric and complete sample counts passed the unchanged loader audit. Native
metrics were recomputed for every nonzero pose. Body-frame/laboratory and
independent SciPy density calculations agree within `3.7e-13` in log density.
Exact atom-union checks on 128 randomly selected nonzero rows plus the 16
largest contributions agree with hard validity against **both** neighbors.
The seven existing fitter unit controls also pass.

Independent static probes gave the following hard-valid, native and
capture-valid rates. They did not evaluate depletion weights.

| Model | Valid geometric probes | Gaussian valid fraction | Forecast CPU s per 4 × 8,192 hybrid draws |
| --- | ---: | ---: | ---: |
| Selected | 117 / 512 | 22.85% ± 1.86% observed SE | 268 |
| Covariance × 4 | 38 / 512 | 7.42% ± 1.16% observed SE | 87 |

The forecasts assume guide weight 0.75, inner uniform weight 0.05, and the
previous valid-pose cloud cost. They also use the source's imprecise
hard-volume estimate for the much smaller geometric-defense contribution;
neither the runtime nor physical precision is guaranteed.

`report.json`, `cross-validation.json`, and `fit-comparison.json` preserve
the diagnostics. The latter is reproduced by the archived `compare_fit.py`,
which uses a matrix Cayley inverse independently of the fitter's quaternion
quotient. No new physical normalizer or convergence claim follows from
these fitting diagnostics.

## Fresh physical validation of the frozen refit

The subsequent campaign was frozen in
`runs/native-ab-refined-validation-20260920/fresh-protocol.json` before new
outcomes: eight selected-model populations and four covariance-times-four
populations, each with 16,384 unconditional draws. The complete cover,
guide weight 0.75, inner uniform weight 0.05, Poisson intensity/activity
ratio 64 and two independent clouds were unchanged. Selected seeds were
`99111010+1009*i`; wider seeds were `99121010+1009*i`. The executable hash
was unchanged from the earlier validated hybrid normalizer.

| Campaign | Draws | log Q | Observed row / population RSE | ESS | Largest contribution | Sampling CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier wide confirmation, now training | 262,144 | 35.74235 | 3.81% / 2.51% | 688.49 | 1.312% | 708.10 |
| Fresh refined single Gaussian | 131,072 | 35.77224 | 2.20% / 2.68% | 2,036.66 | 0.601% | 1,187.43 |
| Fresh refined covariance × 4 | 65,536 | 35.83594 | 17.62% / 16.62% | 32.19 | 14.399% | 153.25 |

The selected estimate is 3.03% above the training-source estimate, a
difference of 0.68 combined observed linear-scale standard errors. The new
width control is 6.58% above the selected estimate but has substantially
larger error; the difference is 0.35 combined observed standard errors.
The selected population log estimates range from 35.64771 to 35.88117.
These are useful independent agreement checks, not certified tail bounds
or calibrated significance tests.

Observed importance ESS per sampling CPU second rises from 0.972 for the
training-source guide to 1.715 for the refined selected guide, a factor of
1.76 in this comparison. Its squared row error times CPU also decreases,
from 1.026 to 0.574. The population-based error diagnostic does not show the
same improvement because the source's population scatter was smaller than
its row error. These finite-population results should not be advertised as
a converged efficiency ratio or a Markov-chain mixing speedup.

Every one of the **196,608 fresh rows** passed the independent scalar
proposal-density, branch, original-weight and checksum audit. A separate
protocol/cloud audit verified all frozen models, executable, configuration,
seeds, draw budgets, mixture settings and cloud settings. Paired cloud
differences attribute 36.56% of the selected run's observed variance and
22.64% of the wider run's variance to conditional cloud noise. The
normalization convention retains all zero draws. An independent exact
atom-union check of 128 uniformly selected unconditional poses plus the
16 largest original contributions per campaign—288 poses total—agreed with
the recorded native, capture and both-neighbor hard classifications.

Hard-volume estimation remains inadequate: the simultaneous zero-activity
ESS values are only 3.07 and 1.55. The refined guide concentrates on the
depletion-weighted target, so these runs do not replace the geometric
hard-volume controls. Nor do these native-region integrals establish
coverage of all competing environments, actual contact mixing, or assembly.

The independent analysis and rendered comparison are in
`runs/native-ab-refined-comparison-20260920`; protocol, cloud and geometry
audit artifacts are in `runs/native-ab-refined-validation-20260920`.

![Frozen refit and independent guide-width validation](../runs/native-ab-refined-comparison-20260920/native-guided-comparison.png)

## The remaining outer tail is small but still unresolved

Applying the **original, pre-refit** Mahalanobis chart to every fresh pose
retains the original importance denominators and full unconditional sample
counts. The earlier problematic `5<r<=8` band now agrees between the two
fresh proposals: selected log Q **33.07980** (10.64% observed RSE, ESS 88.3)
and wider log Q **33.08739** (16.00%, ESS 39.0). The selected `4<r<=5` band
also has improved concentration, with log Q 33.34403 and ESS 283.2.

The outer `r>8` band remains proposal-sensitive. The selected estimate has
log Q **27.06244**, ESS 1.56 and 80% observed RSE; it supplies only 0.0165%
of that run's observed total. The wider estimate has log Q **30.29869**,
ESS 32.89 and 17.43% RSE, contributing 0.3937%, close to the previous
source's observed 0.4235%. The small contribution limits its observed
effect on the total, but the selected proposal's failure to reproduce it
cannot be treated as convergence or as evidence of negligible unseen mass.
An independent geometric shell calculation can test this residual.

That independent check is now available in
[native-ab-uniform-tail-reference.md](native-ab-uniform-tail-reference.md).
The uniform 8–12 reference reproduces the wider guide's estimate of that
exact shell. A separate larger uniform 5–8 run agrees with the guide
estimates' scale but reveals increased weight concentration; the shell
precision and coverage beyond radius 12 remain unresolved.

The complete diagnostic is
`runs/native-ab-refined-validation-20260920/original-chart-tail-comparison.json`.
It reuses the archived scalar classifier and independently checks matrix
coordinates on all 196,608 fresh poses plus 65 synthetic controls; maximum
relative latent-coordinate disagreement is `5.1e-12`. These bands remain
proposal diagnostics rather than newly defined physical basins.
