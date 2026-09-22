# Auditing the independent R4 SMC controls

`tools/analyze_r4_smc_control.py` audits one completed, immutable control at a
time. It never starts the SMC executable. Both September 22 control protocols
remain inert: `runs/smc-r4-density-bridge-control-20260922/protocol.json` and
`runs/smc-r4-density-bridge-narrow-control-20260922/protocol.json`.

Each fixes four independent populations, N=2048 particles, M=262144
unconditional initialization attempts, 128 beta stages, four local proposals per
particle and stage, two incremental clouds, and incremental Poisson ratio 128.
Both use the same frozen executable, original R4 physical target, complete native
observer, and normalized 50/50 current-ball plus **unfiltered** old-chart-ball
proposal. Original old-R5 q/capture restrictions apply only to reporting its
intersection with native entry; they do not restrict the geometric proposal.

The original mutation scales are translations [0.2, 2] Angstrom and rotations
[1.5, 15] degrees. These substantially exceed the observed thin-contact geometry.
The separate narrow protocol predeclares translations [0.05] Angstrom and
rotations [0.1] degrees using existing config fields, with no executable change.
Both retain rotation probability 0.5, mutation Poisson ratio 64 and the original
endpoint envelope settings. Narrow scales can improve local acceptance while
slowing travel across pose space; neither setting guarantees mixing. The second
protocol has independent seeds and output directories. Neither protocol is an
adaptive extension of the other, and the original frozen protocol is unchanged.

## Commands after the physical populations have completed

These commands are documentation only and have not been executed on protein
outputs. They require all four terminal status records before creating analysis
output. Each output path must be new; failed analyses also require a new path.

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_r4_smc_control.py analyze \
  --protocol runs/smc-r4-density-bridge-control-20260922/protocol.json \
  --out runs/smc-r4-density-bridge-analysis-20260922 \
  --workers 4 --stage-stride 16

/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_r4_smc_control.py analyze \
  --protocol runs/smc-r4-density-bridge-narrow-control-20260922/protocol.json \
  --out runs/smc-r4-density-bridge-narrow-analysis-20260922 \
  --workers 4 --stage-stride 16

/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_r4_smc_control.py compare \
  --left runs/smc-r4-density-bridge-analysis-20260922/analysis.json \
  --right runs/smc-r4-density-bridge-narrow-analysis-20260922/analysis.json \
  --out runs/smc-r4-density-bridge-comparison-20260922
```

The parent scheduler must keep the total physical-job cap; these commands do not
supply launch authorization. Native classification uses at most four processes
with numerical library threads limited to one each.

## Evidence and reconstruction

The audit checks the protocol freeze, implementation-validation freeze, binary
hash, literally embedded source bundle, input and complete-observer hashes, exact
CLI options, executed manifest, provenance copies, terminal summary/status chain
and both raw JSONL hashes. Sources and raw files are rechecked after processing.
The analysis copies its local Python source closure and freezes every output.
Comparison checks both analysis freezes, original physical identity and the
complete observer. Only explicitly documented translation/rotation proposal
scales may differ in configs; bath, geometry, target support, observer,
rotation probability and mutation Poisson settings may not change silently.

Every one of the M initialization rows is read and labeled, including explicit
invalid zeros and old-chart proposals outside current R4. The independent
inverse charts reconstruct each support indicator, Jacobian and full physical
mixture density. For the proposal-density bridge, the operative initial weight
is H, whereas H/g is a separate physical-hard-target diagnostic. Both class-mass
sums retain denominator M. No selected-component density or additional current
Jacobian is substituted for g.

Every stage reconstructs the beta path, fresh-cloud count factors, arithmetic
cloud mean, density correction, normalizer product, systematic parents and
original ancestor counts. Every saved valid mutation record is checked against
the combined gained/lost and deterministic density correction; accepted
trajectory endpoints and move-type counts must match. Genealogy refers to
original unconditional draw IDs, including through duplicate descendants.

Complete native and independent atom-union hard/contact classification applies
to contributing initial poses, all terminal particles, and stages
0,16,32,48,64,80,96,112,128 by default. The fixed profile is written before
classification. Numeric audits still cover every stage. A different positive
stride must be declared before analysis; labels at all 128 stages are not needed
for the terminal estimator. Invalid initial rows retain explicit zero labels
without expensive native classification.

Raw data do not record atom coordinates for each Poisson thinning event, nor
invalid mutation proposals. The audit reconstructs saved counts and algebra; it
does not independently regenerate each random cloud or verify the geometry of
every rejected pose. Intermediate poses outside the declared native profile do
not receive fresh atom-union checks. Frozen source review and the independent
analytic Rust reference controls supply separate evidence for those mechanisms.

## Population estimates and limits

For each fixed class B and population r, the terminal estimator is
`Qhat_B,r = Zhat_r * count_B,r/N`. The classes are total R4, full registered native
entry, native intersection with exact old R5, remaining native R4, contact with
no native entry, and unbound with no native entry. They retain the same radial,
angular and 64-orthant strata as the importance analysis. Intermediate
`Zhat_beta * count_B/N` estimates the **bridge** measure, not the final physical
class mass.

A completed zero-hit population contributes zero to every estimator with its
original M retained. It is never replaced or extended. Zero observed mass is
reported as unresolved, with no epsilon, finite free energy, or implied upper
bound. Other populations retain it in the arithmetic mean.

Uncertainty is the sample covariance of the independent population mass vectors
divided by the original population count. The output retains a log scale plus
scaled covariance matrix, relative covariance and class RSE. Native/no-entry
free-energy contrast uses their paired covariance within a population. Its SE
uses the delta method; the displayed 95% halfwidth uses Student t with R-1
degrees of freedom. Four populations give limited tail information. Neither
particle-count ESS nor family concentration is treated as an IID sample size.
Family counts, family concentration and largest population contributions are
diagnostics only.

The comparison preserves both allocations and every original denominator,
reports class and stratum log-mass differences, and separately checks the direct
native/no-entry free-energy contrast. Diagnostic agreement requires both
absolute difference at most 0.2 and difference at most three combined
whole-population SEs. Opposite shifts in two class masses cannot silently pass
the direct contrast check. Unobserved contributions remain unresolved; stratum
checks are diagnostic and have no automatic claim of simultaneous coverage.

No automatic full-vessel or finite-assembly decision is implemented. A validated
and analyzed independent SMC control is still needed before that decision, and
agreement inside R4 does not bound missing disconnected components or mass
outside R4. The separate class-balanced importance estimate remains necessary
when no-entry terminal particles are absent.

## Nonphysical tests

`tools/test_analyze_r4_smc_control.py` uses tiny synthetic JSON records and a toy
observer. It exercises all-M/exterior-zero accounting, zero-hit preservation,
positive cloud arithmetic, mutation history, corrupt-record/hash rejection,
independent chart measure and unfiltered reference support, population covariance
including zeros, proposal-only identity exclusions and the direct-contrast
opposite-shift failure. It launches no sampler and reruns no historical audit.
The analytic physical toy tests for the Rust driver are recorded separately in
`runs/smc-r4-implementation-validation-20260922/validation.json`.
