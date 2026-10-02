# Staged feasible dimer proposals

[`FactorizedDimerProposal`](../src/factorized_dimer.rs) implements the
[finite-cap construction](factorized-dimer-conditioning.md) as a standalone
proposal. It has no bath, state-selection law or production assembly option.
The original whole-joint kernel and its random draw law remain available.

At fixed selected labels, anchor and spectators, the sampler independently
draws two complete defensive-mixture edges:

1. The anchor-to-root edge is redrawn until the root clears fixed spectators
   and its atomic wall, or the root cap is exhausted.
2. The root-to-child edge is drawn in the identity root frame until the pair
   has no core overlap and does have exclusion contact, or its cap is exhausted.
3. The two passing edges are composed and the complete endpoint is checked,
   including the child's wall and spectator constraints. A joint failure
   discards both edges. A fresh joint attempt is permitted only within the
   separately fixed joint cap.
4. The first complete success is scored with the full old/new product-mixture
   density ratio. The conditioning factors are constant for this fixed context
   and cancel. The caller must subsequently make one physical decision.

Both execution orders are supported and frozen per sampler. A failed first
stage skips the second; all consumed draws remain in the trace. Any zero cap
is the identity limit. Hard-valid sources outside internal exclusion contact
also self-loop in this channel. Other reversible channels are needed to create
or break that internal contact.

The new `draw_edge` API delegates to the unchanged independent edge drawer and
defers mixture scoring until a complete feasible endpoint exists. It redraws
the mixture coin and component on every trial. Retaining a rejected component
would change the law. The root filter excludes both selected labels from its
fixed obstacles.

This reference implementation additionally checks source and final endpoint
frame consistency: recovered internal predicates, raw internal predicates,
world predicates, root constraints and recomposed endpoint predicates must
agree. Arithmetic errors, null decodes, scoring errors and predicate mismatches
are fatal records with all completed draws attached. They are never converted
to ordinary rejections or retries. These checks expose discrepancies; they do
not prove floating-point reversibility or certify that every rejected raw
edge has an exact-real predicate.

Eight new Rust tests passed together with eight capped-kernel and seven
defensive-kernel tests. They check edge/RNG compatibility, both orders, stopping,
zero caps, source exclusion, child-wall and spectator failures, discarding both
edges after a final failure, full reverse corrections, coordinate composition,
fatal nulls, an explicit frame-mismatch witness and deterministic seed-prefix
continuation. No serialized RNG checkpoint is claimed.

The [independent finite and Lean checks](factorized-dimer-balance-check.md)
cover exact rational accepted flow, cap exhaustion and the shared scalar
factor, including zero probabilities. They do not certify the Rust geometry,
random generator, SE(3) implementation or a configuration-dependent label
selection law. Those remain separate obligations.

The prepared passive comparison uses root/internal caps 32/32 and joint cap 1
against whole-joint cap 32. Both permit at most 64 raw edge draws per outer
attempt. Independent method streams and a separate execution-order coin are
frozen for every source/slot. Equal maximum budgets do not imply equal work or
time. A physical test must include bath cost and rejections; this proposal alone
does not establish faster mixing or native assembly.
