# Conditional docking with an involutive Gaussian map

`docking-mc` combines the fixed-chart inverse map with the exact conditional
Poisson depletion gate. It isolates exchange between contact environments:
one rigid sphere-union body moves among fixed neighbors of the same shape.
It is a conditional docking experiment, not an assembly run.

## Physical target and boundaries

Let `S` denote the fixed neighbors, `E(x)` the moving body's exclusion union
inflated by the depletant radius, and `U(S)` the union of the fixed exclusions.
With translation volume and normalized rotational Haar measure, the target is

\[
 \pi(x\mid S)\propto
 \mathbf1_{\mathrm{hard}(x,S)}\,
 \mathbf1_{\lVert t-c\rVert\le R}\,
 \exp\{z\,|E(x)\cap U(S)|\}.
\]

The ball restricts the **moving center**, rather than all atomic spheres.
The ideal depletant bath is unbounded. All fixed neighbors are checked for
hard-core validity before the run starts. Simultaneous exclusions are treated
as a union; there is no pairwise-additive depletion approximation.

## Matched kernels

Every cycle consists of the configured number of local attempts followed by
one global attempt. Each local attempt selects translation or proper rotation
and a step size from the corresponding fixed list. The same lists and schedule
are used in every benchmark arm. No line, arc, GCA, fitting, or mask update is
included in this experiment.

The global controls are:

* `mixture`: independent redraw from the full normalized frozen Gaussian
  mixture plus a defensive uniform cube/Haar component. The correction uses
  the complete mixture density at both endpoints.
* `involution`: select a fixed neighbor uniformly, then a Gaussian chart pair
  independently of the moving pose. The directed probability is `w_a w_b`,
  implemented by unordered weights `w_a²` or `2 w_a w_b` and a fair direction
  coin. Equal labels are allowed. With a separately selected defensive
  probability, use a uniform cube/Haar proposal instead.

The uniform cube is centered on the capture ball and has side `2R`. Proposals
outside the ball or overlapping a hard core are rejected without retries.
Gaussian coordinates are always relative to the selected neighbor's body
frame; recentering the laboratory frame only relocates the defensive cube.

For the Gaussian branch, let `z` encode the old pose in source chart `a`,
draw `u ~ N(0,I_6)`, and set

\[
 z'=\gamma z+\sqrt{1-\gamma^2}\,u,\qquad
 u'=\sqrt{1-\gamma^2}\,z-\gamma u,\qquad
 y=G_b(z').
\]

The reverse uses `(b,a,u')` and the same map. With `J_a` the chart's physical
volume Jacobian, the required correction is

\[
 \Delta_{\rm map}=
 \tfrac12(\lVert u\rVert^2-\lVert u'\rVert^2)
 +\log J_b(z')-\log J_a(z)
 =\log q_a(x)-\log q_b(y).
\]

This is the **selected component** ratio. Adding the full mixture ratio again
would be incorrect. See the [map derivation](involutive-basin-transport.md)
for the Haar Jacobian and restrictions on changing the chart parameters.
Here the entire atlas stays immutable.

At `gamma=0`, the aggregate candidate law equals independent redraw, but the
acceptance mechanisms differ: the involution retains the source/destination
labels, while `mixture` sums their densities before acceptance. At `gamma=1`,
equal labels produce identities; these are accepted but excluded from useful
pose-change counts. Numerical chart failures are logged null moves.

## Composition with the Poisson gate

Pull old and new neighbor coverage into the moving body's fixed body frame.
Let `V_+` be coverage gained and `V_-` coverage lost, both restricted to its
inflated exclusion union. The existing gate samples

\[
 G\sim\operatorname{Pois}(\lambda V_+),\qquad
 L\sim\operatorname{Pois}((\lambda+z)V_-)
\]

using conservative BVH envelopes followed by exact membership thinning.
For valid endpoints it accepts with

\[
 \min\{1,\exp(\Delta_{\rm map})(1+z/\lambda)^{G-L}\}.
\]

For a fixed pair of endpoints the forward and reverse Poisson laws obey

\[
 \frac{P_y(L,G)}{P_x(G,L)}
 =\exp[-z(V_+-V_-)](1+z/\lambda)^{G-L}.
\]

Multiplication by the physical target ratio cancels the exponential volume
factor. The remaining factor, together with the inverse-map correction, is
therefore the extended-state Metropolis ratio. Fresh point clouds are valid
here because endpoint selection does not use the gate's cloud. This argument
does not justify substituting an arbitrary noisy energy estimate into
Metropolis. The zero-activity path has unit physical correction.

Each component kernel preserves the same physical target. Their scheduled
composition preserves it too; the whole ordered cycle need not itself satisfy
detailed balance. Finite trajectories from selected starting poses are not
automatically equilibrium samples.

## Running and continuation

```bash
cargo build --locked --release --bin docking-mc
target/release/docking-mc \
  --config PATH_TO_DOCKING_CONFIG.json --model PATH_TO_FROZEN_ATLAS.json \
  --out runs/docking-c05 --cycles 5000 --sample-every 1 \
  --method involution --correlation 0.5
```

Use `--method mixture` for the independent control. Configuration fields,
including physical bath parameters, local steps and capture geometry, are
declared in `DockingConfig` in `src/docking.rs`. The campaign launcher
`tools/run_involution_docking_campaign.py` writes complete configurations
from archived conditional docking inputs; its `--help` describes dataset
selection and preparation-only operation.

Resume into a fresh directory with `--resume OLD_RUN/checkpoint.json` and the
same original config, model, method and correlation. `--cycles` is the absolute
final cycle. Proposal, gate and acceptance RNG streams depend on seed, cycle
and attempt, so continuation exactly reproduces uninterrupted sampling with
the same build. Source, executable and input hashes are archived in each run.

`moves.jsonl` includes proposed and retained poses, inverse traces, corrections,
gate counts and acceptance. `trajectory.jsonl` retains residence after
rejection. `summary.json` separates accepted proposals from actual pose changes
and reports proposal, geometry, gate and diagnostic CPU time. Timings cover
the current segment; cumulative counts should be differenced against
`initial_counts` when analyzing a continuation.

## Validation and interpretation

The integration tests check analytic one-neighbor sphere equilibrium, an
independently sampled two-neighbor target with genuine triple exclusion
overlap, and a deliberately missing-Jacobian negative control. An asymmetric
dumbbell runner test independently replays density corrections, laboratory
frames, hard/capture checks and Poisson gates, and checks exact restart for
mixture and three correlations.

The protein benchmark separately measures native registration and whether the
moving exclusion union overlaps any neighbor exclusion. Only exchanges between
well-separated **bound** native and competing cores count as contact
roundtrips. Raw threshold recrossings and excursions into locally unbound
space are reported separately. Source/destination labels and atlas weights
are proposal diagnostics, not thermodynamic basin probabilities.

The conditional atlas control is retrospective and native-informed. It tests
whether a represented pair of basins can exchange under the physical gate;
it does not establish template-free discovery or an equilibrium free-energy
difference. The scientific endpoint remains repeated physical contact exchange
per CPU, followed by a separately converged basin-weight calculation.
