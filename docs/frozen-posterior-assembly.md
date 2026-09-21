# Frozen posterior transport in the all-mobile runner

The spherical assembly runner can mix the existing full-mixture capture move
with the frozen posterior-chart move validated in the docking benchmark. Every
tetramer remains mobile. The physical target is unchanged:

`hard(X) wall(X) exp[-z |union_i E_i(X)|]`.

The spherical wall constrains the protein atoms only; the ideal depletant bath
permeates it. There is no docking capture sphere, q window, or contact-region
conditioning in this mode.

## Configuration and scheduling

Use the ordinary `tetramer-mc run --method learned --model ATLAS --config CONFIG`
command, with an immutable open-space atlas and this optional configuration:

```json
"frozen_posterior": {
  "probability": 0.5,
  "correlation": 0.9
}
```

`probability` is the state-independent fraction of global update slots assigned
to posterior transport. It must be in `[0,1)`, leaving the original full-mixture
capture kernel present. `correlation` must be in `[-1,1]`. For example,
`global_probability=0.5` and posterior probability `0.5` give 50% local,
25% full-mixture capture, and 25% posterior attempts. Each global kernel uses
the existing `learned_uniform_weight` to choose its uniform or Gaussian branch.
Both kernels use the same supplied atlas. No atlas fitting occurs.

Omitting this option preserves the previous proposal streams and serialized
configuration schema. Probability zero gives the previous physical trajectory
for the same inputs and seed, while recording the enabled configuration and
kernel names. Posterior slot selection and posterior proposal draws use separate
named streams, `posterior-choice` and `posterior-proposal`. The usual sweep,
body order, local/global slot, gate, and acceptance streams retain their names.

This first integration requires a spherical boundary and the learned method.
It rejects combinations with auxiliary transport, reversible jumps, contact
memory, conditional closure, atlas transport, or atlas masks. A fitted chart
depending on the moving body would require a different inverse construction;
this implementation does not silently reuse its frozen-chart formula.

## Retained anchor and acceptance

For each posterior attempt on body `i`, select anchor `j` uniformly among all
other body indices. These spectators are fixed for this attempt only. Convert
the old pose into the selected anchor's body frame and use `DockingProposal`
with source responsibility `w_a g_a(x)/G(x)`, destination weight `w_b`, and
standard-normal auxiliary noise. Equal source and destination labels remain
allowed. The inverse swaps labels and retains the returned inverse noise.

The Gaussian proposal correction is

`log G_j(old) - log G_j(new)`.

`G` includes the normalized Gaussian chart mixture and physical SO(3) Haar
density conversion. It excludes the separately selected uniform branch.
The uniform branch correction is zero. The source-label probability ratio,
auxiliary-density ratio, and extended map Jacobian are already included in the
Gaussian correction; they must not be added a second time.

The anchor stays fixed during the update, and its `1/(N-1)` selection probability
cancels. State-dependent nearest-anchor or contact-weighted selection is not
implemented. Such a selector would need its reverse/forward probability ratio
and a preserved reverse support.

All spectators remain in the physical calculation. The runner's existing
environment constructor retains every body within exclusion-overlap reach of
either endpoint, using symmetric conservative bounds. This is independent of
the proposal anchor. Hard/wall-invalid candidates are rejected. Valid candidates
use the unchanged conditional Poisson gate with the union of all spectator
exclusion regions, and acceptance is

`min(1, exp(proposal_correction + gate.log_weight))`.

Thus overlapping spectator exclusion regions are counted once. Reducing the
environment to the selected anchor would change the many-body marginal.
Global pose maps, collective GCA, and center shifts retain their original
physical acceptance rules. No projection, overlap repair, or retry-until-valid
step is introduced.

## Why capture remains separate

The existing independent capture proposal uses
`Q_j(x)=eta/V + (1-eta)G_j(x)` and its full `log Q(old)-log Q(new)` correction.
The posterior Gaussian branch uses `G` alone. In a dispersed configuration,
`G(old)` can be very small; its separately selected uniform branch does not
provide a reverse-density floor inside a Gaussian attempt. Retaining the
full-mixture kernel preserves the existing route from the dispersed state into
the atlas neighborhoods.

At `correlation=0`, the posterior Gaussian candidate is an independent draw
from `G`; at `0.9`, it retains correlated standardized chart coordinates.
The matched c=0 and c=0.9 comparison must keep the capture kernel, slot
probabilities, atlas, bath, boundary, and local/collective schedule identical.
The old capture-only sampler is a useful third arm, but is not the branch-matched
c=0 control.

## Audit trail and validation

With this mode enabled, ordinary move records name `local`,
`full-mixture-capture`, or `frozen-posterior` in `proposal.kernel`.
Posterior records retain global `moving_index` and `anchor_index`, correlation,
and the unchanged physical gate and acceptance records. Gaussian attempts also
retain source/target labels, forward and inverse traces, component/mixture log
densities, expanded correction, and final correction. Null proposals retain
their reason and anchor.
The resolved option is recorded in configuration, manifest, and summary.

Focused validation uses tiny sphere systems: configuration incompatibilities,
absence/zero-probability compatibility, actual runner scheduling and restart,
global-anchor trace reconstruction, inverse/correction identities, and full
spectator gate replay. A comparison against an anchor-only gate must expose
non-anchor shielding in a posterior attempt. Existing posterior-docking and
involution/depletion tests separately cover exact reference stationarity and
the chart construction. These checks validate integration assumptions; they
do not establish tetramer assembly, equilibration, or a sampling speedup.

The focused checks can be run in the development profile:

```sh
cargo test --locked --offline --lib frozen_posterior_config
cargo test --locked --offline --test frozen_posterior_assembly --test spherical_runner
```

The three-sphere replay found 24 and 32 posterior attempts whose sampled bath
factor differed when the non-anchor spectator was incorrectly removed, for
c=0 and c=0.9 respectively. These are discriminating test witnesses, not
measurements of tetramer performance.

The next physical control should release all three tetramers in the AB
preparation and include independent dispersed starts, with no q restriction.
Count native attachment/detachment, partner exchange, competing-contact
transitions, and events per CPU, classified by the move kernel. Frozen native
charts remain supplied proposal information; their use is not native discovery.
No physical run is launched by enabling or documenting this implementation.
