# Completed regional contact weights

**Two independent importance proposals agree on a strong native preference
inside the fixed R4 region: approximately −18.77 kBT relative to contact
without native entry.** This is a conditional result for **one mobile tetramer
and two fixed neighbors**, at depletant radius **1.5 Å** and activity
**0.035 Å⁻³**. It is not the free energy of assembling those neighbors, a
full-vessel contact comparison, or evidence of finite-system equilibrium.

All 81 producer/audit stages and both postrun stages completed and drained.
The implementation admission passed, including 304 selected full-geometry
rows. The calculation retains all 262,144 attempted draws, including hard-invalid
zeros, with eight independent populations of 16,384 draws per proposal arm and
two independent Poisson clouds per valid pose. It uses the original repaired
shape, physical measure, R4 domain and fourteen-template native classifier.

![Regional weights and matching SMC comparisons](../results/native-class-physical-report-20261004-v2/regional-weights.png)

## Physical estimates

The contrast is `−log(Q_native / Q_competing)`, formed from independent
**linear population masses**, with their paired covariance. Intervals below
are approximate population-based Student-t/delta-method 95% intervals. Their
precision does not rule out unobserved contributions.

| Proposal | Native − competing, depletion (kBT) | 95% half-width | Native importance ESS | Competing importance ESS | Largest native / competing contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard-free guide | −18.7771 | 0.0539 | 3,300 | 1,494 | 0.311% / 1.504% |
| Class guide | −18.7718 | 0.0828 | 2,408 | 2,437 | 0.444% / 0.473% |

All four decision regions—native, competing, native in the old R5 intersection, and
remaining native R4—pass their aggregate ESS, largest-contribution and
population-relative-error checks for both proposals and both physical weights
and hard-only volumes. For native and competing depletion weights, relative
population errors are 2.2–2.5%. Their proposal comparisons pass both the
three-combined-SE and 0.2 kBT criteria.

The old R5 intersection retains all original restrictions: old-chart radius
at most 5, original `q > 1`, old capture validity, and native entry. “Remaining
native R4” is its complement within native R4, not solely a radial shell.

Without depletion, the same native/competing hard-volume contrasts are
**+2.0628 ± 0.0649** and **+2.0151 ± 0.1058 kBT** (95% half-widths).
Thus the depletion factor strongly changes the relative weight in this
particular two-neighbor pocket. It does not follow that an isolated tetramer
will find or create that neighborhood.

Sampler-only importance ESS per CPU second is 0.718/0.325 for native/competing
with the hard-free guide, versus 0.394/0.399 with the class guide. The class
guide improves competing-region efficiency here by about 23%, while the
hard-free guide is better for native mass. These are importance-integral
efficiencies; they exclude audit cost and do not measure trajectory mixing.

## What remains unresolved

The aggregate results do **not** open the convergence gate:

- The report retains 223 unresolved stratum comparisons across physical
  weights and hard-only volumes. Many concern previously problematic,
  currently tiny strata. Of the physical-weight entries, 18 are material
  (at least 1% observed parent mass in either arm); two fail proposal agreement.
  Both are competing-region latent orthants: bin 62 contributes 1.754% versus
  0.407%, and bin 34 contributes 1.634% versus 1.286%. Their log-mass differences
  are 1.482 and 0.260, exceeding 0.2; neither exceeds three combined SE.
- Material radial bins also fail individual quality checks despite proposal
  agreement. Examples include native radius 2–3 (about 3.1% of native mass),
  competing radius 0–2 (24–26%), and competing radius 3–4 (7–8%). Nearly 97%
  of observed native weight is at latent radius 3–4. That is a reason to
  investigate the boundary and remaining space, not to discard these draws.
- New class-guide population-size, defensive-probability and cloud-intensity
  studies have not run. The **original conditional-ray allocation of 786,432
  draws already completed** and retains its failed convergence result; it will
  not be repeated or relabeled as this follow-up.
- The atomic-wall domain outside R4 remains unresolved. A tiny competitor
  weight inside this native-centered region does not exclude a much more
  important competing basin elsewhere.

There were no unbound observations and no native-entry/unbound classifier
inconsistencies. Absence is not a zero-mass proof. Independently, the previously
completed calculation provides the deterministic bound

`log Q_unbound(R4) ≤ −9.8079042`

for both hard-only and depletion weights. Its region bytes, repaired shape and
Cayley/Haar convention match this calculation exactly. An unbound pose has
zero exclusion-overlap enhancement, so its mass is bounded by the latent
6-ball volume times the maximum Jacobian. This bound addresses the **unbound
part of R4 only**; it neither bounds contact outside R4 nor changes the saved
zero-observation comparison failures. The derivation and original result are
preserved in [the conditional-ray report](conditional-ray-reference-results.md).

The [expanded contact catalogue](expanded-native-contact-diagnostic-20261004.md)
also matters for interpretation: “competing” here means outside the original
frozen entry definition. It is not a general assertion of nonnative monomer
registry. No labels or integration regions were changed after sampling.

## Matching the earlier SMC estimates

The comparison uses the same shape, two-neighbor scaffold, domain, measure and
regions, including the SMC normalizer times terminal-region indicator. No
endpoint fraction or differently sized capture domain is substituted.

| Historical estimate | Matching quantity | Current class guide minus SMC, log mass | Combined SE units | Both checks pass? |
| --- | --- | ---: | ---: | --- |
| Unrestricted broad | Native | +0.487 | +3.56 | No |
| Unrestricted broad | Remaining native R4 | +1.549 | +5.36 | No |
| Unrestricted narrow | Native | +0.134 | +2.05 | Yes |
| Unrestricted narrow | Remaining native R4 | +0.273 | +1.90 | No |
| Restricted narrow | Competing | −0.064 | −1.85 | Yes |
| Restricted large | Competing | +0.059 | +0.75 | Yes |
| Restricted broad | Competing | −0.523 | −2.20 | No |

The hard-free arm gives the same qualitative comparison. Both current arms
agree with both unrestricted SMC variants on native mass in the old R5 intersection.
The broad unrestricted SMC discrepancy is concentrated in the **remaining
native R4 mass**. Restricted narrow and larger-population SMC agree with the
current competing mass; restricted broad SMC still disagrees by the absolute
criterion. Unrestricted SMC's lack of terminal competing observations is not
a zero-mass estimate with resolved uncertainty.

This narrows the discrepancy to particular regions and earlier sampling
variants. It does not retroactively establish their convergence or establish
coverage outside R4.

## Reproducibility and continuation

- Original campaign:
  `/vast/xvg/tetramer-mc-runs/native-class-physical-primary-20261004`.
  Statistics SHA256:
  `942b4f7fbe8bc8e163eb8c88a342c2e98432edfcece282a8e63374c87a4fb72c`.
- Completed postrun:
  `/vast/xvg/tetramer-mc-runs/native-class-physical-postrun-20261004-v3`.
  Execution-plan SHA256:
  `9abc6361d7c43e89f5a4a7bc79baad6954a7096f40ccb2325e043b7ee5d04709`.
  Admission SHA256:
  `7a15e180263b8bf5d19b5f9056a74a149b16e128d37e3ab7bfafe6c64ec70613`.
- Thirty synthetic postrun tests passed. A metadata preflight authenticated all
  115 source/input maps, including the 17 documented filename namespaces.
  Earlier failed preparations are preserved; neither created analysis jobs
  or altered scientific output. The final postrun parsed no scientific rows
  and performed no new geometry or sampling.
- Plot and machine-readable display of all regional/SMC/stratum results:
  `results/native-class-physical-report-20261004-v2/`. The report reuses the
  completed finite-R4 unbound bound and records every input hash.
  The original figure is preserved; v2 corrects two labels to “remaining
  native R4” to avoid describing the reference-region complement as a purely
  radial shell. Numerical results are unchanged.

The next sensitivity allocation uses three new independent class-guide arms,
with completed primary controls reused externally. In parallel, the broader
growth-history observer and the partner-atlas benchmark test accessibility
and contact exchange. Finite-system assembly and instability both remain
unresolved.
