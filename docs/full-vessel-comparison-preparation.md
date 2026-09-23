# Frozen full-vessel comparison, awaiting the regional gate

The [preparation](../runs/full-vessel-comparison-preparation-20260922/plan.json)
freezes **16 independent populations and 2,621,440 attempted draws**. It is
inert: no protein population has been launched. The completed regional
confirmation still fails its predeclared precision/concentration and subdivision
checks, so the vessel comparison must not start on the strength of aggregate
mass agreement alone. The finite-system assembly question remains unresolved.

| Stage | Existing vessel arm | Half-mixture arm |
|---|---:|---:|
| standard | 4 × 65,536 | 4 × 65,536 |
| large | 4 × 262,144 | 4 × 262,144 |

Both stages have fresh, disjoint seeds. They are separate estimates. The second
stage follows successful execution and auditing of the first; its allocation
does not adapt to observed weights. The preparation provides commands and
immutable inputs, not an executable production dispatcher or a final stage
aggregator. Those integrations remain required before launch.

## Identical physical measure and complete proposals

The baseline is the existing coverage atlas, with 178 base components and 328
virtual components, averaged over both fixed anchors. Its cube/Haar defensive
probability is 0.1 and its covariance standard-deviation multiplier is 1. The
other arm samples exactly

`0.5 p_vessel(x) + 0.5 q_bank(u(x))/J(u(x))`.

The frozen bank has 80 components and internal uniform probability 0.5. Its
Gaussian tails are untruncated. Every attempted draw uses the complete physical
proposal density, independent of the generating branch. Invalid attempts remain
zeros in the original denominator; two independent clouds are averaged in
linear weight, and no extra Jacobian is multiplied into `1/p_mix`.

The repaired shape, two fixed neighbors, normalized proper-rotation Haar measure,
radius 1.5 Å and activity 0.035 Å⁻³ are unchanged. Auxiliary intensity ratio is
64. The atomic wall has center zero and radius 223.32617672378387 Å; capture
radius 273 Å encloses the wall domain. The ideal bath permeates the wall.
Historical capture-related text in metadata is retained for identity, while
the actual domain is fixed by these explicit predicates.

The release binary and its exact embedded Rust source closure are archived.
The preparation verifies that closure against the already checked integrated
sphere references. It does not claim that compilation alone tested optimized
execution. The [subsequent release reference](../runs/full-vessel-release-reference-20260922/validation.json) now supplies that record: four fresh sequential sphere populations, 256 attempts, three baseline variants and one half-mixture, all independently audited. Its binary/source hashes match the preparation. The largest reconstructed log-density error is 1.36e-12. A future dispatcher must bind this separate record; the earlier immutable preparation is not rewritten.

## One physical classification, exhaustive reporting

[The baseline auditor](../tools/audit_full_vessel_baseline.py) reconstructs the
unchanged schema-4 density. [The mixture auditor](../tools/audit_full_vessel_latent.py)
reconstructs both complete component laws and their schema-5 mixture. They use
the same frozen complete native observer and independently check wall/core
predicates, original metric, primitive cloud counts, arithmetic weights,
generation metadata, invalid zeros and unconditional denominators.

[The partition pass](../tools/vessel_contact_partition.py) reuses those saved
native/core labels; it does not repeat expensive atom-pair classification. It
reports native entry, contact without native entry, and unbound without native
entry, crossed with:

- inside versus outside the **current** measured R4 support;
- inside versus outside the union of every measured pocket witness.

The witnesses are current R4, the older original-native R4, alternative R5 and
alternative R32. The two objects called R4 are different charts. Each witness
retains its original source capture and original-q cut. Witness overlap is
combined by Boolean union and never by summing overlapping masses. Capture,
q and chart radii are reporting predicates, not restrictions of the vessel
target. Ball-only reporting flags in the original auditors are distinguished
from these complete witness supports.

Every cross contribution uses the original number of attempts. Zero observations
remain an unresolved contribution, not an upper bound. Population statistics
average **linear** masses, retain zero populations, and use paired whole-population
covariance for the native/competing free-energy contrast. Importance ESS and
largest-draw contribution remain separate concentration diagnostics. An explicit
remainder estimate still needs convergence assessment; its mere inclusion does
not prove unseen modes are absent.

## Evidence and execution contract

`prepare_full_vessel_comparison.py` only implements `freeze` and `validate`.
Validation checks archive bytes and runtime and never starts physics. Its
numerical gate requires `convergence.confirmation_passed` and every declared
regional check, rather than the deliberately false downstream `passed` flag.
The authenticated gate helper also binds the exact confirmation artifact and
distinct broad/narrow SMC protocols with disjoint population seeds.

An observed independent matching mass corroborates an importance estimate when
its difference is both ≤0.2 log units and ≤3 combined population SE. A material
contradiction exceeds both; intermediate or unobserved cases remain unresolved.
The gate does not demand equal occupancies, rare no-entry terminal hits, an
IID ESS for resampled descendants, or observations in every subdivision. Audit
or provenance failures still stop progression. A material independent
contradiction must be resolved before dispatch.

A future dispatcher must bind the frozen preparation, release-reference record,
completed regional and both matching SMC comparisons; enforce eight physical
jobs and 32 total workers; stop launches and drain children on failure; preserve
all raw outputs without retry; and invoke the frozen audits/partition per job.
No regional, vessel or scaffold result alone settles mobile finite-system
assembly.

The new baseline, accounting and gate tests use synthetic records only. They
include original versus mixed density, selected/all anchors, hard-invalid zeros,
arithmetic cloud averaging, overlapping witnesses, captured exterior support,
population covariance, immutable inputs, duplicated controls, and changed
confirmation identities. Existing completed protein audits are reused.
