# Paired-cloud allocation gives only a small additional gain

Changing the fixed-dictionary allocation objective from the noisy second moment
to the physical second moment reduces aggregate heldout second moments by only
about **0–4%** relative to the previous optimized allocation. Important tails
remain poorly sampled. This diagnostic does not justify a new physical campaign
or change the unresolved contact-weight conclusion.

![Independent heldout moment ratios](../runs/paired-guide-allocation-report-20260924-v2/heldout-moment-ratios.png)

Each point is one independent heldout population. The denominator is the frozen
historical noisy-moment optimum, not the original unoptimized dictionary. Ratios
below one favor the new allocation. There are only two populations per source
arm, so the figure intentionally has no uncertainty bars. The `smc` label names
the proposal dictionary used to generate these independent importance samples;
the plotted observations are not correlated SMC terminal particles.

The fit reused all 84 Gaussian means and covariances, the original 21 protected
groups, the 50% uniform defensive component and the component-weight floor
`1e-5`. Only the weights changed. The previously completed noisy-moment optimum
was not refitted. Training used the original `r00/r01` populations from both
source arms; the candidate was frozen before opening `r02/r03` arrays. These
holdouts had been inspected in earlier work, so this remains retrospective
proposal diagnosis, not pristine validation.

For independent cloud estimators at the same pose, let
\(I_k=J\widehat W_k/q_s\). The new objective uses

\[
\widehat M_{2,R}(q)=\frac1N\sum_{i\in R}I_{i1}I_{i2}
\frac{q_s(u_i)}{q(u_i)},\qquad
\mathbb E\widehat M_{2,R}(q)=\int_R\frac{J(u)^2 W(u)^2}{q(u)}\,du.
\]

The old noisy objective replaces the product by
\(((I_1+I_2)/2)^2\), which also contains conditional Poisson-estimator variance.
The new fit minimizes the worst physical-moment ratio to the original `q80`
reference over the same protected groups. Both objectives retain the complete
source-density correction and unconditional attempted denominator. Hard-invalid
attempts contribute zero; they are not removed from that denominator.

Pooling the two heldout populations **within each source arm** gives the
following descriptive ratios. Every independent outcome is retained in the
[report](../runs/paired-guide-allocation-report-20260924-v2/report.md).

| Region | Physical M₂, bank | Physical M₂, smc | Noisy M₂, bank | Noisy M₂, smc |
|---|---:|---:|---:|---:|
| Native inside R5 | 0.9979 | 0.9952 | 0.9999 | 0.9946 |
| Native outside R5 | 0.9826 | 0.9626 | 0.9790 | 0.9657 |
| Contact without native entry | 0.9736 | 0.9984 | 0.9739 | 0.9967 |
| Native outside R5, orthant 55 | 0.9747 | 0.9857 | 0.9738 | 0.9745 |
| No-entry contact, orthant 63 | 0.9675 | 0.9611 | 0.9673 | 0.9620 |

“Native outside R5” is the native complement within the fixed R4 integration
domain; it is not the competing-contact class. The complete native classifier,
region, Jacobian and physical measure remain unchanged.

The small ratios must be read alongside contribution concentration. For
no-entry orthant 63, physical-M₂ contribution ESS is only **1.26–2.47 per heldout
population**, with the largest term carrying **60–89%** of its sum. Even pooling
within source gives ESS only **3.14 and 2.75**. Native-complement orthant 55 has
ESS **2.61** in `bank/r02`; its ratio worsens to **1.0254** in `bank/r03`.
These are effective counts for second-moment contributions, distinct from
ordinary importance-weight ESS and trajectory ESS. Similar ratios can be stable
while both underlying moments miss the same tail. They neither establish
coverage nor demonstrate independent contact samples per CPU.

The one fit took **47 optimizer iterations**; fitting plus frozen evaluation
took **15.47 s**. All **524,288 original attempts** were preserved: 262,144 for
training and 262,144 for heldout evaluation. Training included 212,909 known
invalid zeros, including outside-region/support and hard-invalid attempts.
No physical samples, geometry/classifier evaluations or
physical audits were rerun. The earlier **22 passing tests** (16 objective and
6 controller checks) are recorded in the
[validation receipt](../runs/paired-guide-allocation-tests-20260924/validation.json);
they were not repeated to produce this report.

The [reporting script](../tools/report_paired_guide_weights.py) authenticates all
33 files in the pinned diagnostic archive. It hashes sample files without
decoding them; every plotted or tabulated number comes from `analysis.json`.
Its [frozen report](../runs/paired-guide-allocation-report-20260924-v2/report.json)
and [CSV](../runs/paired-guide-allocation-report-20260924-v2/all-population-strata.csv)
preserve all 355 original class/radial/angular/orthant groups for all eight
populations, including empty groups. The five displayed groups are a readable
subset, not replacement convergence gates.

Reproduce only the presentation into a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  tools/report_paired_guide_weights.py \
  --source runs/paired-physical-guide-allocation-20260924 \
  --out /tmp/paired-guide-report-reproduction
```

Pinned plan SHA256:
`f4d1b7722b8390bc6c853d8b5996b85caf857ec282edc11da8ff05c2f4d9f79a`.
Pinned analysis SHA256:
`e3d83b46749d6b7ae3031d85f7e97777aacfa684f0a782c523ae7436dcf4cf64`.
Validation receipt SHA256:
`0e35cc3e34c65e2ae22aeabf245436867e31198d004bc49f66b4450fc557c917`.
