# Retaining either root of the conditional dimer proposal

The experimental `two_root_m4` arm in `examples/evolving_dimer_benchmark.rs`
chooses which of the two mobile tetramers is the root relative to a fixed
spectator anchor. It moves both tetramers and can change their relative pose,
using the existing factorized dimer kernel. The original `m4` arm always uses
the first mobile label as root; `two_root_m4` selects either order with probability
one half at each collective slot.

```mermaid
flowchart TD
    A[Four unchanged local updates in canonical label order] --> B{One independent root-choice bit}
    B -- probability 1/2 --> C[Root: first member; child: second member]
    B -- probability 1/2 --> D[Root: second member; child: first member]
    C --> E[Retain selected order through all capped trials]
    D --> E
    E --> F[Same full proposal ratio and m4 auxiliary correction]
    F --> G[One physical depletion acceptance decision]
    G --> H[Keep accepted or rejected state in canonical label order]
```

For fixed spectators, let `K0` and `K1` denote the two existing reversible
conditional kernels, including null trials and rejections. The new kernel is
`K = (K0 + K1)/2`. Summing their accepted-flow balance identities proves balance
for the mixture. The selected-root correction is still
`log F_r(old) - log F_r(proposed)` plus the existing auxiliary correction.
The one-half selection probabilities cancel. Relabeling the order used to store
two poses has unit Jacobian in the same product pose measure; physical particle
labels and the target density do not change.

This argument requires retaining the selected root for the entire kernel call.
It does not authorize choosing a new root after a failed inner trial, selecting
a root based on its apparent acceptance, or omitting a state-dependent reverse
selection factor. The [context-selector balance analysis](context-selector-balance.md)
and its existing finite-state tests cover that distinction. No new physical
balance identity is needed for a state-independent mixture of the unchanged
kernels; their geometry, floating-point and physical-bath obligations remain.

The cached guidance points are body-coordinate data for the identical rigid
shapes. The selected engine evaluates them in its root's frame, with `m=4`
internal guidance and no root guidance. Proposal, threshold, bath and acceptance
random streams retain `m4` roles. Only root selection uses the additional
`two_root_m4/root_order` role. Local updates, retained endpoints and checkpoint
pose arrays remain in canonical label order.

The new arm requires its own exact policy:

```json
{
  "two_root_policy": {
    "schema": "two-root-factorized-m4-policy-v1",
    "root_probabilities": [0.5, 0.5],
    "selection": "once_per_global_slot",
    "internal_threshold_m": 4
  }
}
```

Trial records retain the selected order and add canonical/selected labels,
root slot/probability, and canonical old/proposed projections. Fatal and null
attempts retain the root metadata. Observers must validate these fields and map
by particle label; treating a reversed proposal array as canonical is incorrect.
The existing Python observer/preparer are frozen inputs of an ongoing native
registry audit and have not been modified. This arm is therefore an experimental
implementation, not yet an admitted trajectory campaign or a production
assembly option. A subsequent observer integration and frozen campaign are
required before scientific use.

The intended first comparison is eight context-0 chains: four streams from the
same source plus one from each of the four existing proposal-prepared states.
Each has 512 warmup plus 4,096 production blocks, with four locals and one
collective attempt per block. Existing local and `m4` controls can be reused. The conditional
conditions remain 1.4 Å, activity 0.0275 Å⁻³, two mobile tetramers and frozen
spectators; they are distinct from the original finite-system decision target.

The diagnostic must keep both chosen roots and their failures, complete source
and destination proposal scores, auxiliary/bath contributions and completed
external-contact exchanges. Reversing the root can worsen source support.
Acceptance alone cannot establish useful sampling: compare contact-environment
ESS per full CPU, initialization agreement and nonempty exchanges, with native
registry assessed independently.

This control follows the [completed fusion diagnostic](singleton-fusion-diagnostic.md):
all 34 admitted fitted centers collide, including 24 whose first blocker is the
other mobile member. That result motivates investigating cooperative geometry,
but does not predict that changing root will solve it. Neither this conditional
control nor the collision diagnostic establishes physical assembly or instability.


All thirteen example tests passed, including five new tests for forced-root
record/state/RNG equivalence, correct body-cloud frame, null/fatal outcomes,
finite-state balance and disk continuation from both preparation types.
Independent source review passed. The offline release build uses the separate
`target-validation-two-root` directory. Source and production-executable hashes
were unchanged throughout validation. Evidence is in
`results/two-root-validation-20261004`; the authoritative clarified build witness
is `build-witness-protein-scope.json`. The clarification distinguishes synthetic
test draws from the zero new protein-campaign draws; original logs and receipts
are preserved. No protein trajectory was launched for this arm.
