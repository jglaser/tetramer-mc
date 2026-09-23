# Passive contact efficiency observer

`tools/analyze_contact_efficiency.py` implements the measurement prerequisite for
the frozen-proposal assembly benchmark. It does **not** launch simulations or
decide whether assembly is thermodynamically stable. Validation uses synthetic
histories only; existing protein histories were read only to inspect their file
schema. No completed physical audit or contact calculation was repeated.

## What the observer measures

The input is one completed invocation of Rust `tetramer-mc`: its effective
`config.json`, `manifest.json`, `summary.json`, `checkpoint.json`, lossless
`trajectory.jsonl`, and archived input configuration, shape, source bundle and
optional frozen proposal. These are the actual `simulation.rs` schemas. A resumed
invocation has its own initial sweep and CPU clock; it is not concatenated with
another invocation. The effective configuration is reconciled with the archived
input and the Rust serialization defaults. The explicit override list permits CLI
method/sweep/cadence settings, canonicalized shape/catalogue paths, periodic
wrapping or declared checkpoint restoration of initial poses, and recorded derived
coordinate/proposal fields. Physical bath, boundary, seed and proposal settings
cannot silently depart from the archived input. The manifest and summary must
agree with those settings. Archived shape bytes define geometry; historical
canonical path strings are not reopened.

Every scheduled saved frame must be present, including repeated retained poses.
The analyzer checks the initial poses, terminal checkpoint, cumulative and segment
move dispositions, one selected update per body per sweep, monotone CPU, monotone
counts, and exact saved-frame schedule. The production window must start and end
on saved endpoints and have constant spacing. An off-cadence terminal frame is
audited but must lie outside the explicitly selected regular window. The baseline
frame is used to establish the first interval; subsequent endpoints are samples.
No rejected state is removed, no gaps are interpolated and no extra samples are
created by repeating a sparse frame for hypothetical intermediate sweeps.

The contact fingerprint is the vector of indicators
`(body_i, body_j, patch_i, patch_j)`, with `i < j`. A token is present if any pair
of its atom spheres has surface gap strictly less than twice the depletant radius.
Hard-overlap checks and spherical per-atom wall checks accompany classification.
Every atom has a frozen body-frame patch ID. A residue/member partition is useful
for proteins; atom IDs also work but describe finer and noisier surface motion.
The analyzer never filters patches according to a native label or observed weight.
All tokens observed anywhere in the declared window enter the diagnostic; unseen
tokens have no estimated mixing time and remain an explicit coverage limitation.

This distinguishes a change of contacting surface patches from a rigid rotation
that preserves the contact environment. It reports a trace apparent ESS and
region-indicator apparent ESS using the existing FFT/Geyer positive-monotone
finite-record estimator from `mobile_posterior_metrics.py`. The CPU denominator is
the actual sampler CPU difference between the window boundaries, including local,
collective and rejected work. Passive observer CPU is reported separately. The
estimator is capped at the number of saved samples. A constant descriptor gives
`null` ESS, not an independent sample at each frame or infinite efficiency.

The estimator is a diagnostic, not a general convergence theorem. In particular,
the composed sweep need not itself be reversible, and Geyer's reversible-chain
positive-sequence justification is not established for every schedule. Short
records, common trapping, nonstationarity or unseen modes can give misleading
apparent ESS. No equilibrium error bar is manufactured from that ESS. Independent
run means supply the between-population uncertainty used in comparisons.

## Instantaneous registry and environments

Optional `native_definition` loads the frozen `NativeContactRegions` definition,
checks its shape and every archived `input_sha256` binding, and calls `classify_pair`
for every mobile pair. It does not invoke its fixed-scaffold `classify` method.
All directed entry motif matches are retained. `NativeGraphConsistency` then
checks whether their alternatives admit one consistent ideal catalogue pose
assignment. Hysteretic entry/retention labels are never read or used as physical
regions. Catalogue closure is a local registry diagnostic, not a global crystal
or free-energy certificate. Every native dependency is recorded in the analysis
and rechecked after observation, including archived reference/source files not
read by pair classification. Input paths must remain inside the definition's
`inputs` directory.

Periodic contacts use a unique minimum image, requiring each cell length to
exceed `4 * (shape_bound + depletant_radius)`. The native observer reimages each
pair and checks that the native graph admits an ordinary-space image lift. A
native graph winding around a periodic cycle fails closed: the existing catalogue
checker has no lattice-winding closure law. A finite crystallite crossing a
boundary is supported when its contact graph has a consistent lift. A future
periodic crystal observer must supply the correct quotient-space cycle test.

The plan declares disjoint named environments through required/forbidden patch
tokens and optionally `native_consistent_connected: true/false`. `remaining`
always contains every other configuration. Observed overlapping environment
predicates fail closed. The predicates should be designed as a genuine partition
before sampling; checking observed points does not prove disjointness everywhere.

An exchange is a saved-frame passage from one declared environment to a different
declared environment; any number of `remaining` observations may intervene.
Subsequent visits to the same environment do not create an exchange. For each
pair, forward and reverse passages and nonoverlapping observed roundtrips are
separate outputs. A third declared environment interrupts that pair's roundtrip.
The initial and unfinished final episodes are explicitly censored. Saved episode
spans do not assert uninterrupted residence or resolve transient events between
frames. MC sweeps/CPU are algorithmic effort, not physical attachment kinetics.

An unobserved return means `unassessed_equilibrium_weights` unless an independent
physical reference is supplied. No zero observation becomes a thermodynamic
zero. An independently certified negligible upper probability bound removes a
requirement for frequent returns. If both independent lower bounds exceed the
declared appreciability threshold, no observed passage in one direction flags a
finite-run sampling limitation. Unequal occupancies do not impose equal rates or
equal numbers of observed passages.

## Frozen input plan

Prepare the patch file before the benchmark:

```json
{
  "schema": "body-frame-atom-patch-map-v1",
  "shape_sha256": "SHA256_OF_ARCHIVED_SHAPE",
  "atom_patch_ids": ["member0:residue0", "member0:residue0", "member0:residue1"]
}
```

There must be exactly one ID per atom; the three-element list illustrates the
format and is not a protein patch definition. The benchmark must freeze its
actual complete map.

An analysis plan has this form:

```json
{
  "schema": "frozen-contact-efficiency-plan-v1",
  "physical_identity_sha256": "DIGEST_FROM_physical_identity",
  "patch_map": {"path": "patches.json", "sha256": "SHA256"},
  "native_definition": {"path": "native-region/definition.json", "sha256": "SHA256"},
  "preparation_id": "dispersed",
  "proposal_arm": "correlated-transport",
  "burn_sweep": 1000,
  "end_sweep": 100000,
  "appreciable_probability": 0.01,
  "environments": [
    {"id": "native", "native_consistent_connected": true},
    {"id": "competing-patch", "native_consistent_connected": false,
     "required_tokens": [[0, 1, "member0:residue0", "member1:residue1"]]}
  ]
}
```

These are **illustrative**, not scientifically selected production regions or
burn lengths. `physical_identity(config, shape_sha)` binds shape, N, boundary,
cell dimensions, bath parameters, mobility and physical measure;
`digest(identity)` is canonical SHA256. The two optional native/reference keys
can be omitted. Relative file bindings resolve against the plan directory.

An optional `physical_reference` binding points to
`independent-contact-region-probability-bounds-v1`, with:

- matching `physical_identity_sha256` and `region_definition_sha256` (the latter
  is `digest(plan["environments"])`), plus `observer_definition_sha256` binding
  those predicates together with the exact patch-map and native-definition hashes;
- `coverage_resolved: true` and `population_diagnostics_passed: true`;
- `probability_bounds` mapping every named environment **and remaining** to
  `[lower, upper]` probabilities, with `sum(lower) <= 1 <= sum(upper)` for
  the exhaustive partition;
- `independent_of_run_sha256` identifying this trajectory and an `evidence`
  `{path, sha256}` binding to the independent physical calculation.

This is an explicit audited-evidence attestation interface. It does not prove
the reference's mathematical derivation or independence from a JSON flag. The
campaign owner must establish those properties. Conditional R4/scaffold region
weights cannot be substituted for a mobile finite-system reference.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONOPTIMIZE=0 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/analyze_contact_efficiency.py \
  --run /absolute/path/to/completed-run \
  --plan /absolute/path/to/frozen-plan.json \
  --out /absolute/path/to/fresh-analysis
```

Output includes every frame's instantaneous patch/native labels, region
occupancies and half-window diagnostics, observed passages and censored episodes,
initialization seed/pose hash/preparation identity, proposal identity and schedule,
source bindings and separate observer CPU. Dependencies are rechecked after
observation. `freeze.json` binds both output files and the observer/estimator
implementation digests. This is an integrity record, not external proof that the
plan was declared before sampling.

## Independent initialization and proposal comparisons

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONOPTIMIZE=0 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/analyze_contact_efficiency.py \
  --compare /absolute/path/to/run1/analysis.json /absolute/path/to/run2/analysis.json \
  --out /absolute/path/to/fresh-comparison
```

The comparator requires identical physical target, complete region definition,
patch/native observer, cadence, window, local/collective schedule and endpoint
noise/cost settings (`poisson_lambda_ratio` and the full `endpoint_gate`). It
verifies each report's sibling `freeze.json`, both frozen output hashes and the
observer/ESS implementation bindings, and requires identical implementation
digests across reports. Comparison inputs are rechecked after aggregation.
Grouping is
by the predeclared proposal arm and preparation. All master seeds must differ;
paired-stream inference and joined continuation segments are deliberately
unsupported. It reports independent-population mean and standard error of each
occupancy and ESS/CPU descriptor, plus observed initialization/proposal contrasts.
Any unresolved constant-trace ESS keeps the group's efficiency unresolved. Fewer
than four streams per group are flagged as incomplete relative to the assembly
plan. Agreement within three observed standard errors does not establish
coverage; a zero observed variance explicitly remains a warning sign.

The strict comparator also requires equal `global_probability`. Its current
baseline is **local+uniform-global**, compared with redraw/transport arms using
the same global slot probability. A pure-local arm (`global_probability: 0`)
cannot be compared with a positive-global-probability arm through this interface.
Such a comparison needs a separately predeclared arm probability contract; the
current implementation does not infer or relax one after seeing results.

## Remaining obligations

Before a production efficiency comparison, freeze meaningful protein patch and
environment definitions and independently resolve their relevant equilibrium
weights. Existing regional contact calculations alone do not supply those mobile
system occupancies. Matched proposal scales and collective schedules, preparation
streams, sufficient trajectory cadence and long enough observation windows are
campaign responsibilities.

This initial analyzer fails closed for biased or configuration-dependent
auxiliary/adaptive proposal modes. Frozen correlated transport is supported.
Physical occupancies under a frozen assembly bias require self-normalized
reweighting and a correlated weighted-observable variance estimator; an ordinary
unweighted-chain ESS is inappropriate. This observer does not replace the
separate reversible kernel/bias implementation audits.

Focused validation command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONOPTIMIZE=0 \
  /home/xvg/protein-nucleation/.venv/bin/python -m unittest discover \
  -s /home/xvg/tetramer-mc/tools -p test_analyze_contact_efficiency.py -v
```

The tests cover repeated rejected-state residence, attempted-count/CPU closure,
missing/duplicate frames, irregular terminal windows, constant/unseen contacts,
patch changes with unchanged partners, independent stream uncertainty,
instantaneous native entry/cycle integration, periodic contacts and wall checks,
exchange censoring and unequal/unknown/negligible region occupancies. Regression
checks also cover effective-input tampering, legitimate Rust defaults/overrides,
complete native dependency closure and mutation, report/observation/source freezes,
matched noise/cost schedules and impossible exhaustive probability intervals.
They do not
run a sampler or establish protein mixing.
