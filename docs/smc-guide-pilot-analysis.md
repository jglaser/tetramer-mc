# Bounded fresh SMC geometry guide pilot

The pilot compares the frozen 80-component bank with the 84-component `r5-cov1`
guide using four independent populations of 65,536 attempted draws per arm.
Both retain a 50% uniform component, two independent Poisson clouds per valid
pose and auxiliary intensity/activity ratio 128. Shape, radius 1.5 Å, activity
0.035 Å⁻³, two-anchor scaffold, R4 domain, measure and complete native classifier
remain unchanged. This is 524,288 fresh attempts, with no automatic extension.

The serialized workflow runs physical sampling, both independent density audits,
and first-pass classification automatically. It loads only the frozen archived
modules, preserves the terminal physical status while analyzing, and refuses
existing physical or classification output.

For manual use after a standalone controller run finishes, invoke the archived
analyzer once (never while the serialized workflow is active):

```sh
/home/xvg/protein-nucleation/.venv/bin/python -B CAMPAIGN/common/analyze_smc_guide_pilot.py \
  --campaign CAMPAIGN --out FRESH_ANALYSIS_DIRECTORY --workers 4
```

The analyzer reuses the validated first-pass classification and region
summarization from `analyze_contact_confirmation.py`. Every attempted row remains
in the saved compressed records, including hard-invalid and exterior zero
weights. Every contributing pose receives exactly one full native classification
and one exclusion-contact classification. The same pass saves the historical
R5 intersection of full-native R4 and its full-native complement. This measured
intersection retains the original metric `q > 1` and capture-domain predicates;
it is distinct from the unfiltered R5 ball used to group training geometry.

All native, contact-without-entry and unbound contributions, both native subsets,
all original radial and angular strata, and all 64 orthants are reported. Each
arm retains four separate populations, paired-cloud variance diagnostics,
unconditional denominators, proposal-branch accounting and physical sampler CPU
time. Native/no-entry free-energy intervals reuse population covariance and the
Student-t distribution with three degrees of freedom.

The original thresholds remain unchanged: population RSE ≤10%, importance ESS
≥200, largest contribution ≤2%, agreement within both 0.2 log units and three
combined standard errors, and paired 95% interval half-width ≤0.5 kBT. These
checks apply to both arms; native-complement improvement cannot hide a
contact-without-entry precision failure. Improved no-entry ESS/CPU is **not** a
pass requirement. The descriptive efficiency ratio measures importance sampling,
not trajectory mixing or completed environment exchanges.

Even if every pilot diagnostic passes, `passed`, `full_vessel_gate_open` and
`assembly_stability_established` stay false. The earlier failed confirmation is
not superseded. New-proposal size/intensity sensitivity, independent region
coverage, remaining-vessel contributions and finite-system controls remain
separate requirements. Unobserved contributions are unresolved unless the
existing finite-R4 geometric bound applies; they are not assigned physical zero.

The completed controller, raw results, density audits, full local Python source
closure and classifier definitions are authenticated before classification and
again before final output. Labels and records are hashed and frozen. Output
directories must be fresh; partial analyses are never silently replayed. If a
classification worker fails, the executor drains all already-submitted work,
including queued tasks, retains its output and records a failed terminal status.
A scoped SIGTERM handler converts termination into the same draining path and
restores the prior handler afterward. Worker processes defer SIGTERM/SIGINT to
the parent; status snapshots use atomic replacement.
Neither old audit nor old classifier passes are replayed.

Validation:

```sh
python -m unittest discover -s tools -p test_analyze_smc_guide_pilot.py -v
```


## Frozen execution and references

The campaign is [smc-geometry-guide-pilot-20260923](../runs/smc-geometry-guide-pilot-20260923/protocol.json),
with protocol SHA256
`03778e707ca08759c1401ad5949a52bfe9c1857f1964eacddecaf9ab9dbdbab5`.
It uses fresh seeds `137101010 + 1009*i`, i=0,…,7. No earlier population is
restarted or pooled, and there is no automatic larger allocation. The numerical
proposal selection follows the previous offline diagnostics; these new physical
populations provide the independent test of that fixed choice.

The same archived Rust executable (`b0e051638276de13336090c21b4f2c8def693d5b6037eac1e2591dba29225094`)
has completed analytic sphere/depletion and Haar-measure references, reused
without audit replay. A new proposal-only normalization control uses
`X = 1_R4 / (V_R4 q)` with exact expectation 1 and bounds 0≤X≤2. Four independent
16,384-draw populations per proposal gave 1.001292 (bank) and 1.003950 (SMC).
Both satisfy the simultaneous Hoeffding halfwidth 0.021539 at family failure
probability 10⁻⁶. Complete-density implementations agree within 1.14×10⁻¹³,
including Gaussian centers and points on both sides of the R4 boundary. This
reference is Python proposal validation, not Rust execution or new protein data.
The physical pilot audits every new row's Rust density and Jacobian independently.

The controller, analyzer and serialized workflow have 31 passing new tests,
including full denominator accounting, immutable guide/target binding,
source/runtime rejection, failure draining, and actual SIGTERM during synthetic
classification. Eight physical jobs and 32 scientific workers are the caps;
classification uses four workers. Every raw attempted draw is retained.

The already-launched serialized command is:

```sh
/home/xvg/protein-nucleation/.venv/bin/python -B \
  /home/xvg/tetramer-mc/runs/smc-geometry-guide-pilot-20260923/common/run_smc_guide_workflow.py \
  --campaign /home/xvg/tetramer-mc/runs/smc-geometry-guide-pilot-20260923 \
  --expected-protocol-sha256 03778e707ca08759c1401ad5949a52bfe9c1857f1964eacddecaf9ab9dbdbab5
```

This command is recorded for reproduction, not for relaunching an active or
completed campaign. [Workflow status](../runs/smc-geometry-guide-pilot-20260923/workflow-status.json)
and [physical/audit status](../runs/smc-geometry-guide-pilot-20260923/status.json)
record the actual stage and process birth identities. The workflow completed
and wrote `comparison/analysis.json` and `comparison/report.md`.
The [completed pilot results](smc-guide-pilot-results.md) retain its failed
convergence checks and distinguish native proposal gains from competing-contact
tail uncertainty.

This allocation tests candidate utility. Previous noisy-moment diagnostics
suggest its no-entry ESS may remain below the fixed precision requirement.
Both regions remain reported in both arms; a native improvement cannot be
converted into evidence of assembly or used to discard a competing-region
precision failure.
