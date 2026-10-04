# Native-class guide in the full vessel

The full-vessel normalizer now accepts the same frozen native-class conditional
line guide as the regional calculation. This closes a proposal-format gap; it
does not launch the vessel campaign or establish convergence of contact masses.
The assembly kernels and their running executables are unchanged.

For a world pose \(x\), its proposal is

\[
 p(x)=\tfrac12 p_{\rm vessel}(x)
      +\tfrac12 q_{\rm class}(u(x))/J(u(x)).
\]

Both terms are evaluated at every attempted world pose. A valid attempt has
weight \(\widehat W(x)/p(x)\), with the existing arithmetic mean of two
independent Poisson estimators. Invalid wall/core/capture attempts retain zero
in the original draw denominator. The Jacobian appears exactly once, in the
conversion of latent density to physical density.

The source R4 chart and its 170 Å capture govern conditional proposal intervals.
They do not replace the 273 Å vessel capture or the 223.3261767 Å atomic wall.
The complete class guide includes every Gaussian component, translation axis
and class channel, including class → hard-free → unconditional fallbacks.
Gaussian tails remain untruncated. Native geometry is used here to improve a
physical integral's proposal; this is not a native-blind assembly proposal.

## Implementation and independent checks

- [Rust wrapper](../src/latent_region/physical_guide.rs) and
  [checked class scorer](../src/latent_region/contact_line.rs): certify positive
  support from retained intervals and fallback predicates. No second geometry
  traversal is needed. An active Gaussian or multiplier that cannot be
  represented is an error; it is not converted to an exterior zero. The exact
  Cayley seam has zero component density, with no finite exclusion band.
- [Normalizer](../src/normalizer.rs): full-vessel schema **7**, outer schema
  `full-vessel-native-class-line-half-mixture-v1`, separate class draw/density
  traces, archived compiled-native bytes and a complete physical-shape atom
  bijection. The source shape, scaffold and depletant radius are authenticated.
  Existing Gaussian and hard-free output schemas are retained.
- [Independent Python audit](../tools/physical_native_class_line_vessel.py):
  rebuilds interval geometry and the complete mixture, checks full-vessel
  weights/zeros/wall/cloud accounting, and checks the original native definition
  for protein data. The computational chart factor has an independent
  high-precision covariance-residual check. Generation is audited at saved
  generated coordinates; world-density scoring is audited at re-encoded world
  coordinates. Those are different floating-point operations near an orthant
  boundary.

The new wrapper rejects class categorical weights whose cumulative intervals
are narrower than one f64 `Open01` spacing, or whose increments disappear in
floating-point accumulation. This admission check does not change
the existing regional generator, its random sequence, or its frozen inputs.
It is one implementation check, not a proof of arbitrary floating-point random
sampling.

Seven Rust proposal-only controls passed in a fresh isolated target:
[receipt](../results/native-class-vessel-validation-20261004/attempt03/validation.json).
They check complete axis/channel mixing, transformed charts, source/vessel
separation, provenance, fallbacks, seams, numerical failures and disabled-law
random-stream parity. A prelaunch Python-version failure is retained as attempt01;
it started no child or sample.

Thirteen independent Python toy checks passed:
[receipt](../results/native-class-vessel-python-validation-20261004/attempt02/validation.json).
They include tampered density/trace/shape negative controls, a lost-positive-channel
case, and full-mixture rows with exterior, seam and hard-invalid zeros. They
generate no physical draws.

The [CLI reference](../tests/basin_normalizer_native_class_line.rs) completed
160 saved sphere attempts: 64 hard-only class, 64 depletion class, and a paired
16-draw Gaussian/disabled-class control. The two active arms produced 36
conditional draws, 52 valid poses outside the source capture, 35 valid poses
outside R4, and 41 explicit invalid zeros. The disabled pair matches every
physical row exactly after removing class-only diagnostics.
[Execution receipt](../results/native-class-vessel-cli-validation-20261004/attempt01/validation.json).

Independent Python reconstruction passed for **all 160 saved rows**, without
new draws. Maximum reconstructed log-density disagreement was 8.19×10⁻¹²;
maximum inverse-CDF disagreement was 4.45×10⁻¹⁵. This includes separately
reconstructing the conditional generation traces and world-pose densities.
[Audit receipt](../results/native-class-vessel-cli-validation-20261004/audit-attempt01/validation.json).

## Admission to a physical campaign

The independent full-wall sphere/depletion **normalization** reference completed
16,384 attempts and failed its fixed nonzero-activity contact precision gate.
Exact, Poisson-weighted and hard companions share the discrepancy; the saved
contributions identify sparse defensive coverage outside R4. All individual-row
audits passed. The [completed result and variance diagnosis](native-class-sphere-normalization-20261004.md)
retain the failure, allocation and proposed geometric improvement. The adapter
is not yet admitted to the protein vessel campaign. The 160-attempt wiring test
has no stochastic mass-precision gate. Existing completed hard-free vessel references
are reused as evidence for their original implementation; they do not test the
new class-to-vessel wiring.

The legacy full-vessel dispatcher and stage aggregator already exist. The newer
streaming path requires an explicit authenticated integration and a reviewed
admission predicate for the current regional campaign's linear-space population
uncertainties. Renaming the new campaign's fields to satisfy a historical gate
would not be valid. The pending primary comparison also cannot substitute for
the declared population-size, defensive-mixture and cloud-intensity checks.

No protein vessel population has been started by this change. Native, competing
contact and unbound contributions—including regions outside all measured
pockets—remain necessary physical evidence. A successful fixed-scaffold
comparison alone cannot decide finite-system assembly.
