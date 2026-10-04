# Matching completed regional masses to historical SMC

`tools/compare_native_class_physical_smc.py` is a summary-only adapter. It requires
the completed current campaign admission; it cannot consume a partial campaign
or produce new poses, Poisson clouds, labels or geometry. It leaves the physical,
full-vessel and assembly gates closed, including when the reported comparisons
agree. Historical verdicts remain separate, unchanged output fields.

Its `native-class-physical-smc-comparison-plan-v1` input contains these bound JSON
references, each `{path: absolute_path, sha256: digest}`:

```text
current:
  admission, admission_plan
unrestricted:
  authentication, review
  analyses: {broad, narrow}
restricted:
  plan, status, analysis
```

The current admission resolves its statistics plan/result and scientific
protocol. The historical restricted analysis resolves all twelve population
audits and their bound physical configurations. Only JSON summary metadata is
opened. Previous all-row audits and execution receipts are inherited evidence;
their large scientific input streams are not rehashed or replayed here. Source
closure and every metadata file actually consumed are recorded in the output.

All targets must match the repaired shape, fixed two-neighbor scaffold, original
R4 region, 170 Å capture sphere, complete native definition, old R5 reporting
region, 1.5 Å depletants, activity 0.035 Å⁻³, and translation times normalized
proper rotational Haar measure. Old R5 membership remains its original radius,
original-q and capture conjunction. A matching name or capture radius alone is
insufficient. No population can be removed after seeing its weight.

| Current region | Historical region |
| --- | --- |
| `total` | `total` |
| `native` | `registered_native_entry` |
| `native_old_r5` | `old_R5_intersection_native` |
| `native_remainder` | `remaining_R4_native` |
| `competing` | `contact_no_native_entry` |
| `unbound` | `unbound_no_native_entry` |

Unrestricted SMC Qz uses each independent population's terminal normalizer times
its terminal region-indicator fraction. Q0 uses its initial unconditional H/g
estimate, never its proposal-density bridge normalizer. Zero population estimates
are retained. The native-excluded calculation supplies only `competing` and
`unbound`: its `total` excludes native and cannot replace unrestricted total.
Nor does its structural exclusion demonstrate zero physical native weight.

The adapter uses `contact_population_statistics.compare_population_masses` to
compare linear means with independent population errors, allowing eight current
versus four historical populations. It reports both the three-combined-linear-SE
criterion and the 0.2-log-unit criterion. Historical log-delta comparisons are
preserved under their original verdict fields rather than rewritten. Every Qz
radial, angular and orthant comparison is retained, with observed parent-mass
fractions. Historical Q0 stratum estimates were not archived and are explicitly
unavailable. All-zero estimates remain unresolved, without an artificial epsilon
or missing-mass upper bound. No mixed unrestricted/restricted SMC free-energy
estimate is silently formed.

After completion and a separately bound plan, the command is:

```sh
python -B tools/compare_native_class_physical_smc.py \
  --plan /absolute/frozen-smc-bridge-plan.json \
  --out /absolute/new-smc-bridge-result.json
```

The output must be new. This documentation creates no real plan and launches no
comparison. Synthetic tests are in `tools/test_compare_native_class_physical_smc.py`.

All nine focused synthetic tests passed in the final source-bound validation at
[`results/native-class-smc-bridge-validation-20261004-v2/validation.json`](../results/native-class-smc-bridge-validation-20261004-v2/validation.json).
They cover an eight-versus-four-population metadata join, retained historical
failures, linear rather than logarithmic averaging, zero populations, physical
H/g rather than bridge weights, target/scaffold/measure mismatches, missing
populations, reused seeds, corrupted metadata and restricted-region semantics.
They also distinguish the terminal profile marker from annealing stage 128,
require the actual beta bridge, and bind historical bin definitions and the
native partition before assigning matching current stratum identities.

Independent review caught a terminal-schema mismatch after the initial six
synthetic tests: archived terminal profiles use `stage="terminal"`, while
annealing diagnostics carry the integer stage. The adapter and fixtures were
corrected; the original validation remains archived. Metadata-only inspection
checked all twenty completed historical populations and their source/target
bindings. Legitimate zero initializations retain a zero estimate without
fabricating annealing stages. No current physical campaign results were read or
compared, and no scientific sample rows were replayed.
