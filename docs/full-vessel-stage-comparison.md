# Separate full-vessel stage comparison

`tools/compare_full_vessel_stage.py` consumes the eight already completed
population audits and contact partitions for exactly one frozen stage. It does
not run a sampler or repeat geometry/native classification. The preparation
directory, stage and a fresh output directory are explicit:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/compare_full_vessel_stage.py \
  --preparation runs/full-vessel-comparison-preparation-20260922 \
  --stage standard --out runs/full-vessel-standard-comparison-20260922
```

The standard stage has four independent populations of 65,536 unconditional
attempts in each of `vessel` and `half_mixture`. The separate large stage has
four of 262,144 per arm. Neither stages nor arms are combined into an estimator;
the planned allocation is unchanged. A completed comparison can have unresolved
precision. Precision diagnostics do not adapt, skip or authorize an allocation.

The reader verifies the actual preparation freeze, exact jobs, runtime, binary,
embedded Rust source bundle, input hashes, manifests, raw bytes and completed
summaries. Each consumed population file must be hash-bound by its frozen audit;
the existing sampler does not itself emit a population freeze. Audit and
partition freezes must cover their complete files, including labels and archived
Python source closures. Those closures must match the prepared common sources.
The complete native definition, runtime and definition inputs, current support,
historical supports, physical config, wall, measure, proposal, seed and attempted
denominator must match. A `complete` or `artifact_verified` flag alone is not
accepted as identity evidence. Validation failure creates no comparison output.

All saved classes are retained for both depletion mass Qz and hard mass Q0:
native entry, contact without native entry and unbound without native entry;
inside/outside current R4; inside/outside the Boolean union of the four measured
pocket witnesses; the corresponding primary-class intersections and individual
witness diagnostics. Overlapping witnesses are never added to form the union.
Counts, linear mass sums, squared-weight sums and maxima must reconcile across
each exhaustive partition. Invalid attempts retain zero weight in the original
unconditional denominator.

For each class, the reported mass is the arithmetic mean of four independent
linear population estimates. Covariance of the mean is sample covariance divided
by four, retained with separate logarithmic scales for each class. The delta
method gives log-mass SE and native/contact-noentry free energy
`beta(F_native - F_contact_noentry) = log Q_contact_noentry - log Q_native`.
The latter includes covariance between the two masses in each population.
Intervals use Student t with three degrees of freedom. A separate explicitly
labeled native versus all-noentry contrast adds contact and unbound masses in
each population before uncertainty propagation.

Each mass reports population RSE <= 10%, observed importance ESS >= 200 and
largest individual draw fraction <= 2%. ESS/max use the combined weight moments
of the four equally sized populations as weight diagnostics; the uncertainty
sample size remains four. The native/contact-noentry interval must have 95%
half-width <= 0.5 kBT. Between-arm log-mass and direct free-energy comparisons
report both absolute difference <= 0.2 and difference <= three combined SEs.
One passing bound is unresolved, while failing both is a material observed
disagreement. Every remainder contribution is reported even when tiny or
unobserved; no epsilon replaces zero observations and no upper bound on unseen
mass is inferred.

`analysis.json` records the preparation/protocol SHA, exact population job
bindings, all authenticated input hashes, source closure, sufficient statistics,
covariance, per-class diagnostics and comparisons. Its own sources and result
are frozen in the new output. `complete` means the fixed-stage accounting and
comparison finished. Even passing all primary diagnostics leaves
`full_vessel_unseen_modes_certified` and `assembly_stability_established` false.
These observed-weight checks cannot certify absence of unseen modes or finite
system assembly stability.

Synthetic tests require no physical execution:

```bash
PYTHONPATH=tools /home/xvg/protein-nucleation/.venv/bin/python -m unittest \
  tools/test_compare_full_vessel_stage.py
```
