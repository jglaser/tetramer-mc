# Independent R4 controls and the full-vessel bridge

The regional thermodynamic question remains unresolved. The five-arm contact
confirmation has a fixed allocation of 2,228,224 fresh attempts; its population
weights will be inspected after the declared campaign and first classification
pass complete. No training observations are pooled into those estimates.

## Dependent SMC workflow

[`run_r4_smc_controls.py`](../tools/run_r4_smc_controls.py) now schedules the two
already-frozen [SMC controls](smc-r4-control.md). The immutable controls retain
their original commands, seeds, populations, initialization counts, beta
schedule and move scales. Their original `frozen_unlaunched` fields describe
preparation time, not live state. Live execution belongs to the separate
`runs/smc-r4-controls-workflow-20260922/status.json`.

The workflow was queued behind
`runs/contact-confirmation-workflow-20260922`. It waits for both prerequisite
commands to finish successfully, checks the original process birth identity,
then runs at most four SMC populations at once. Broad and narrow populations
are interleaved. Each uses one numerical thread; subsequent independent audits
use four workers. A failed dependency stops dispatch. A physical failure stops
new launches and drains started children without retry, replacement or output
overwrite. All zero-hit populations remain part of the declared estimator.

Scientific convergence failure in the importance experiment does **not** block
this independent diagnostic; it does continue to block full-vessel production.
The two controls do not depend on any partially observed importance weights.
Six synthetic scheduler tests cover the dependency gate, recycled-process
identity, exact interleaving, overwrite refusal and failure draining. They
launch no protein simulation.

The scheduler archives its complete local Python closure, pins its Python
runtime and protocol/input/binary hashes, and uses exclusive creation for its
status and logs. It is deliberately not a resume/retry implementation. No
existing campaign, classifier, or completed audit is rerun by the dispatcher.

## Cross-method estimator

[`compare_r4_smc_importance.py`](../tools/compare_r4_smc_importance.py) reads
completed, frozen analysis artifacts, validates matching shape, scaffold, full
R4 domain, normalized physical measure, complete native definition and old-R5
reporting chart, and requires disjoint population seeds. Only declared local
translation/rotation proposal scales may differ in physical configuration
comparison. It reuses saved class masses without rereading configurations or
replaying geometry/density audits.

| Quantity | Importance population | SMC population |
|---|---|---|
| Physical `Qz(B)` | All-attempt mean of `H_B J W/q` | Terminal `Zhat * count_B/N` |
| Hard-only `Q0(B)` | All-attempt mean of `H_B J/q` | Unresampled initial all-attempt mean of `H_B/g_physical` |
| Intermediate beta measure | Not used | Bridge diagnostics only |

For each arm, the comparison independently reconstructs the arithmetic mean
and covariance of four whole-population vectors, including zero vectors. It
checks the two partition sums in every importance population and reconstructs
each terminal SMC mass from the normalizer and indicator. SMC initial bridge
mass `H`, endpoint fractions alone, geometric means of population estimates,
and descendant IID errors are not physical normalizer substitutes.

Class agreement requires both an absolute log-mass difference at most 0.2 and
at most three combined population standard errors. The native/no-entry contrast
also receives a direct comparison using its within-population covariance.
Predeclared radial, angular and orthant strata retain all zero bins; a primary
class stratum is decision-relevant when either method observes at least 1% of
that class's mass there. Unobserved mass has no epsilon replacement and does
not become a zero physical mass or upper bound. The initial hard-only stratum
estimator was not archived, so only terminal physical `Qz` strata are compared.

After both analyses complete, use a fresh output directory for each control:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/compare_r4_smc_importance.py \
  --smc runs/smc-r4-controls-workflow-20260922/broad-analysis/analysis.json \
  --importance runs/contact-confirmation-comparison-20260922/analysis.json \
  --out runs/r4-smc-broad-versus-importance-20260922
```

The narrow comparison changes `broad-analysis` to `narrow-analysis` and uses
another fresh output path. These cross-method commands are **not yet queued**;
they require completed analyzed artifacts. Eight nonphysical tests cover
normalizer-times-indicator reconstruction, arithmetic population means,
retained zero populations, physical-hard/bridge separation, domain/observer/
seed mismatches, unconditional denominators, exact stratum definitions and mass sums, and direct contrast disagreement.

Neither agreement nor failure here decides finite-system assembly. Agreement
inside R4 still needs the unmeasured-vessel contribution, physical contact
mixing and all-mobile finite-system evidence. A failed fixed control identifies
a sampling limitation rather than instability.

## Full-vessel proposal and independent auditor

The [Rust integration](full-vessel-latent-mixture.md) adds the normalized
`0.5 p_vessel + 0.5 q_D/J_D` proposal. The previous vessel law and no-guide
streams remain available. Full-vessel execution is still gated on regional
validation. Four synthetic Rust tests passed, including analytic sphere
hard-volume/depletion references and orientation moments; no protein vessel
population has been run.

[`audit_full_vessel_latent.py`](../tools/audit_full_vessel_latent.py) explicitly
dispatches the new schema. It evaluates both component densities on all world
poses, independently reconstructs latent coordinates/Jacobians, checks original
attempted denominators and arithmetic cloud means, and checks physical atomic
walls and hard/contact geometry. R4 membership restricts only the guide's
uniform term: every Gaussian remains untruncated. The audit reports valid
contributions both inside and outside R4. With a complete frozen native
definition it also reports native, contact-without-entry and unbound masses
without using the historical scalar registration score as the full classifier.

The normalizer summary does not contain a sample-file hash; the audit binds the
raw bytes at audit time and says so explicitly. An independent pre-execution
campaign freeze must bind the executable and input law. Cloud count algebra and
envelope consistency do not prove exact geometric thinning or floating-point
execution. Those remain implementation obligations.

All six integrated cross-language fixtures now passed, with 128 fresh draws
per fixture (768 total): legacy covariance scaling, reciprocal branches and a
pure-uniform latent guide, each with all anchors and with a selected anchor.
All 162 valid exterior-R4 poses were retained across the six fixtures. The
largest reconstructed component or mixture log-density difference was
1.31e-10. The [reference validation](../runs/full-vessel-reference-dispatch-20260922/validation.json)
contains the per-fixture audit hashes and geometry-pruning counts.

The original queued reference job was stopped before any step when the
confirmation reached 16 completed populations and four remaining workers.
The identical frozen export and audit commands then ran sequentially in that
released capacity. Only dispatch timing changed: executable, inputs, seeds,
768 draws and output paths stayed fixed. This change is recorded in the
separate dispatch journal; the earlier waiting workflow is marked superseded.
The SMC controls remain queued behind the complete confirmation workflow.

The [earlier stage validation](../runs/r4-independent-bridge-stage-validation-20260922/validation.json)
records 28 passing focused tests (4 Rust, 10 vessel-audit, 8 cross-method,
6 scheduler), source hashes and the then-live process state. It also records
that the initial short Rust toy-test batch briefly added a ninth physical
process alongside the eight protein jobs; subsequent references were deferred
until capacity was available. No protein allocation or physical model changed.

These are synthetic reference results. Full-vessel protein production remains
gated, and finite-system native assembly remains unresolved.
