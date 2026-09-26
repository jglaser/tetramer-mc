# Contact-aware anchor selection for rigid oligomers

This optional extension changes the primary spectator selection in the member
transport branch. It leaves the frozen atlas, physical hard shapes, depletant
bath, internal subset rates and ordinary simulation kernels unchanged. It uses
instantaneous geometric exclusion contacts without native labels.

## Motivation and scope

In the existing member-chart protein run, 642 of 675 hard-valid learned trials
on embedded subsets selected a four-anchor pool disjoint from the subset's
entire current exclusion-connected aggregate. Those pools necessarily lacked
an existing external contact partner. Both handle and member-chart runs had zero
accepted learned cluster transports through the matched sweep 7,400. See the
[matched-run audit](../runs/member-mixture-comparison-20260926/report.md).

Member charts can represent contacts through any carried member, but only to
spectators present in the selected pool. Contact-aware selection increases the
chance that this pool represents the current interface. It does not fit new
basins, narrow their covariances, remove association entropy, or guarantee that
the atlas represents a misregistered source contact.

## Selection law

Let S be the selected internally connected subset. For each external tetramer
label a, count the current exclusion-contact edges from S to a:

\[
 c_a(X,S)=\sum_{i\in S}1\{E_i(X)\cap E_a(X)\ne\varnothing\},\quad
 C(X,S)=\sum_{a\notin S}c_a(X,S),\quad M=N-|S|.
\]

Contacts use the existing atomic sphere-union overlap predicate with every atom
radius inflated by the depletant radius. If C>0, select a primary anchor with

\[
 q_a(X,S)=\frac\epsilon M+(1-\epsilon)\frac{c_a(X,S)}{C(X,S)}.
\]

If C=0, use q_a=1/M. Require 0<epsilon<=1 and representable positive probabilities
for all spectators. A selected primary remains a valid reverse label even if
all its contacts disappear. Counted edges weight selection only; the physical
energy is still the full many-body exclusion union, not a pairwise contact sum.

Build A(a) from the primary spectator and its nearest spectator neighbors, as
in the current member implementation. The spectators do not move during this
trial, so the selected pool A(a) is identical on reversal. The *distribution of
the primary label* changes with the state; this requires an explicit correction.

## Balance argument

For retained subset S, primary label a and pool A(a), let G_{S,A} be the existing
normalized member/anchor mixture over one collective six-dimensional pose. Its
posterior-source map contributes

\[
 \Delta_{\rm map}=\log G_{S,A}(X)-\log G_{S,A}(Y).
\]

The new term is

\[
 \Delta_{\rm anchor}=\log q_a(Y,S)-\log q_a(X,S).
\]

The learned trial combines, once per rigid subset,

\[
 \log\alpha=\min\{0,\log W_{\rm depletion}
                 +\Delta_{\rm map}+\Delta_{\rm anchor}\}.
\]

W_depletion is the existing gained/lost-count auxiliary factor. No noisy
energy estimate is exponentiated in its place. Hard-invalid poses have zero
acceptance. Existing assembly bias, when enabled, retains its separate
per-elementary-kernel correction.

To see the selection correction directly, regard a as a retained proposal label.
At fixed S and a, let Q_a denote the member-map proposal with its auxiliary trace.
The accepted forward flow is the minimum of
pi(X) q_a(X) Q_a(X,dY) and its reversed labeled flow
pi(Y) q_a(Y) Q_a(Y,dX), with the existing auxiliary/Jacobian factors included.
These minima are equal under reversal. Summing over a and adding rejection
self-loops gives a pi-invariant fixed-S kernel. The anchor distribution does not
need to preserve pi separately.

The internal subset clock uses only contacts within S, which rigid transport
preserves. Its earlier fixed-duration argument therefore applies unchanged.
There is no new subset-selection correction; the nonzero anchor-selection term
belongs to the elementary fixed-S proposal. Selecting primaries by contact while
omitting this term would bias attachment and detachment.

The defensive uniform branch is chosen **before** the contact-aware primary.
It has no retained primary label and keeps its existing zero proposal correction.
Local rigid moves also remain unchanged. A failed candidate is not redrawn.
Every null/rejected attempt consumes its clock time.

## Configuration, records and continuation

Add the following key to an existing member-chart cluster phase:

```json
"cluster_phase": {
  "transport_charts": "members",
  "anchor_count": 4,
  "anchor_contact_uniform_probability": 0.1
}
```

Other phase settings retain their configured values. The value 0.1 means 10% of
the *primary-anchor distribution* is uniform. This is separate from the existing
`learned_uniform_weight`, which chooses a uniform *pose proposal*. The portable
[spherical-cluster-contact-anchors.json](../examples/spherical-cluster-contact-anchors.json)
contains the full configuration.

Omitting this option or setting it to null preserves the old branch order and
random stream. The option is valid only for member charts. An explicit value 1
has the uniform primary law, but uses the new branch order and therefore need not
reproduce the old bitstream. Enabled settings are recorded in configuration and
checkpoint provenance; changing a resumed configuration is subject to the usual
resume validation.

Enabled learned records retain `primary_anchor`, `anchor_pool`,
`anchor_forward_probability`, `anchor_reverse_probability`,
`anchor_log_reverse_forward` and `map_log_reverse_forward`. After a valid endpoint
is evaluated, `proposal.log_reverse_forward` is their complete map-plus-anchor
correction. Hard-invalid/null candidates may have no evaluated reverse
probability. The outer `selection_log_correction: 0` still refers to the internal
subset clock, not to primary-anchor selection.

## Validation and performance interpretation

The new tests cover probability normalization and support, no-contact fallback,
attachment, loss of the selected anchor, loss of all contacts, and changes to the
contact-count denominator. A finite-state balance check must fail when the anchor
correction is deliberately omitted. The production phase is checked against an
independent analytic sphere/depletion reference, including accepted moves with
nonzero anchor corrections, and through complete runner replay and checkpoint
continuation. Existing inverse/member-chart checks remain applicable.

The first protein comparison uses a frozen 264-tetramer configuration from sweep
7,400 of the current member-chart growth run, at 500 μM, radius 1.4 Å and activity
0.0275 Å⁻³. Four independent populations per arm each run 128 fixed-duration
cluster phases, resetting to the common configuration before each phase. The
allocation is frozen before sampling. This measures conditional proposal
performance, not equilibrium, contact ESS, native assembly or a growth trajectory.
Accepted exclusion-contact changes are distinguished from accepted motions with
unchanged partners and from native registration. Every attempted event is kept.

### Completed validation

All 15 targeted tests passed: four selection/balance tests, six runner tests,
four existing member-chart tests, and the new production sphere-fluid check.
The runner checks cover restart, assembly bias, independent reconstruction of
anchor probabilities, and unchanged trajectories when the option is absent or
null. The finite-state negative control detects the bias from omitting the
selection correction.

For the independent four-sphere depletion reference, the mean contact count was
1.03034 +/- 0.00309. Production chains from dispersed and aggregated starts gave
1.05400 +/- 0.01113 and 1.02513 +/- 0.01125, respectively, including 113 and 133
accepted learned moves with nonzero anchor corrections. These are Monte Carlo
standard errors. See the archived
[validation manifest](../runs/contact-anchor-validation-20260926/validation.json)
for commands, source hashes and logs. Existing Lean results are reused; no new
Lean theorem is claimed for this extension.

### Completed protein diagnostic

The predeclared eight populations completed without expanding the allocation.

| Conditional diagnostic | Uniform primary | Contact-aware primary |
|---|---:|---:|
| Embedded learned trials whose pool contains a current contact partner | 18/641 (2.8%) | 575/629 (91.4%) |
| Hard-valid learned trials | 82/727 | 74/714 |
| Accepted learned moves | 0/727 | 0/714 |
| Median map correction, hard-valid embedded trials | -38.26 | -0.162 |
| Broad fallback sources, hard-valid embedded trials | 69/69 | 16/68 |

The enabled arm's median complete proposal correction, including the primary
selection term, was -6.77. Its median sampled depletion log-factor was -80.49.
These are separate medians of acceptance factors, not free energies, and should
not be added as though they described one trial. Of the 52 hard-valid embedded
trials using non-fallback source components, 50 lost at least one existing
external contact. Source recognition improved substantially, but the proposed
whole-oligomer destinations usually did not compensate the lost interface.

There is **no demonstrated docking or sampling speedup** in this experiment.
The result supports improving destination proposals for the entire moving
oligomer; it does not establish physical instability or equilibrium contact
weights. The option remains opt-in. See the
[full report and acceptance decomposition](../runs/contact-anchor-benchmark-20260926/report.md)
and [comparison figure](../runs/contact-anchor-benchmark-20260926/contact-anchor-comparison.png).

The benchmark executable is `contact-anchor-benchmark`. Its frozen inputs,
launcher, analysis and every attempted event are retained under
`runs/contact-anchor-benchmark-20260926/`. The reusable plotting command is:

```sh
python tools/plot_contact_anchor_benchmark.py \
  --benchmark runs/contact-anchor-benchmark-20260926
```
