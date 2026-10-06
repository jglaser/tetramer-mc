# Native-class guide integration for the vessel measurement

The full-vessel measurement needs to account for native, contact-without-entry
and unbound configurations, including everything outside R4 and the measured
pockets. This integration prepares that measurement; it supplies no new protein
weights and does not establish finite-system assembly or instability.

The new [preparer](../tools/prepare_native_class_streaming_vessel_comparison.py)
keeps the original shape, atomic spherical wall, scaffold, physical measure,
depletant radius 1.5 Å and activity 0.035 Å⁻³. It declares two independent
proposal arms: the full-vessel proposal with the validated wall-envelope uniform
component, and its normalized 50/50 mixture with the frozen 116-component
native-class regional guide. This is an explicit proposal change from the old
cube baseline. It changes neither the target nor the complete native classifier.
The regional guide retains its 50% uniform defensive component.

Both arms use two independent Poisson clouds per valid pose, at auxiliary
intensity ratio 128. Four populations of 65,536 draws per arm precede four
fresh populations of 262,144 per arm: 2,621,440 attempted draws in total. All
hard-invalid zeros and unconditional denominators remain. The stages are
analyzed separately. Preparing these declarations starts no worker.

The [baseline streaming auditor](../tools/audit_vessel_baseline_streaming.py)
now supports the schema-8 wall-envelope law as well as the original schema-4
law. It reconstructs the complete density without relabeling schema-8 input,
checks the envelope against the shape, and retains the arithmetic mean of the
two cloud estimators. Ten synthetic controls passed. Two audits of sixteen
archived sphere poses, at batch sizes 3 and 64, agreed with zero maximum density
discrepancy. They generated no poses or clouds. The
[frozen validation](../results/vessel-baseline-streaming-validation-20261006/attempt02/)
records the checks and drained execution groups.

The [partition](../tools/partition_vessel_streaming.py) and
[stage reader](../tools/analyze_streaming_vessel_stage.py) add a distinct
native-class path, preserving the legacy preparation. They bind the compiled
native bytes to the original definition, its archived inputs, and the complete
shape correspondence. Saved guide labels do not replace the independent
physical classifier. Source closures, geometry records and every attempted
denominator are authenticated before reduction.

Twenty [integration controls](../results/native-class-streaming-integration-validation-20261006/attempt04/report.json)
passed, including legacy paths, class schema 7/8, baseline schema 8, corrupted
bindings and completed-output lifecycle checks. Early attempts contained
incomplete synthetic fixtures and two overstrict exact-float assertions; those
failed logs are retained. The final reduction checks use the existing 10⁻¹²
tolerance. No physical result or gate was changed to obtain a pass.

The actual protein preparation is now frozen at
[attempt01](../results/native-class-streaming-vessel-preparation-20261005/attempt01/plan.json),
SHA-256 `ac70051e13ebc23d3d90d7d95ef3433fad558ea983ad947e1f0168f2edaf3482`.
Its archived validator passes. Only declared job output trees may grow beneath
the preparation; workflow execution, admission and aggregate analysis belong
in a separate output root. `dispatch_ready` remains false. The existing
declaration-only execution planner and old scientific admission schema cannot
be used to bypass the remaining reviewed-dispatcher work.

## Regional prerequisites remain separate

The new [prerequisites adapter](../tools/admit_native_class_vessel_measurement.py)
consumes the completed sensitivity comparison and the already reviewed SMC
comparison. It reads JSON metadata, not scientific sample streams. It checks
the terminal 121-phase lifecycle and its admission/statistics links, retains
the sixteen completed primary populations, and verifies all twenty-four fresh
populations. No populations are pooled across proposal arms or stages.

It recomputes statistics from independent linear population masses, including
the paired native/competing free-energy contrast. It retains **both** the
original proposal comparison and the three sensitivity comparisons. Stratum
mass fractions and per-population partition sums are checked before materiality
is evaluated. A changed materiality flag cannot hide a failed stratum. The
original error, ESS, largest-contribution and agreement thresholds remain.

The SMC report and its plan have explicit reviewed hashes. Its independent
linear-SE and absolute-log criteria determine three distinct states:
corroborated, unresolved, and material contradiction. Missing terminal
observations are unresolved. Restricted competing-contact estimates are never
substituted for an unrestricted total. Resolving the present discrepancies
requires new reviewed evidence; editing a pass flag cannot change admission.

Unbound mass needs special care. The existing geometric R4 upper bound is
5.501502852 × 10⁻⁵ for both Qz and Q0. The adapter verifies its derivation from
the fixed region and retains it as an **absolute** bound. It does not turn
unobserved contributions into zeros. Ratios to estimated physical contact
weights look very small, but this bound is not small relative to the observed
hard-only weights. Neither a rigorous relative bound nor complete Q0
normalization follows. The adapter therefore names its scope as contact-region
prerequisites for the next vessel measurement and preserves the unresolved
unbound comparisons explicitly.

Passing these prerequisites would permit assessing the next measurement's
admission; it would not establish convergence of the unmeasured vessel
remainder. The adapter has no launch entry point, always records zero physical
jobs launched, and leaves the assembly conclusion unresolved. The production
sampler and ongoing simulations are unchanged.

Nineteen [prerequisite controls](../results/native-class-vessel-prerequisites-validation-20261005/attempt02/report.json)
passed, including complete/missing execution phases, rehashed mismatched
receipts, population denominators, classifier/contact consistency, altered
materiality, and the absolute-bound scope. A saved-metadata check also
reconstructed both completed primary arms and retained their 223 unstable
stratum comparisons. It preserved 38 material-contradiction entries and 148
unresolved entries in the matching SMC report; these are overlapping diagnostic
entries, not independent experiments. These validation results are not a
follow-up campaign verdict. The validation pins the Python executable and
NumPy/SciPy initializer files, not every shared-library byte; the future
scientific dispatcher still needs its full runtime/resource contract.

The follow-up comparison must complete before an actual prerequisites report
can be authenticated. Its running allocation is unchanged. The full-vessel
measurement and finite-system assembly campaign have not been launched.
