# Spherical ensemble with learned proposals and collective moves

Both the Python prototype and Rust runner now support all-mobile rigid sphere
unions inside an exact atomic spherical wall, with a homogeneous ideal bath
that **permeates the wall**. The target is

\[
\pi_R(X)\propto H_R(X)H_{\rm pairs}(X)\exp[-z|\cup_i E_i(X)|].
\]

This adds no wall-depletion potential. The periodic and spherical systems are
different finite-volume ensembles; the new spherical starts preserve the old
volume and preassembled fragments, not the old equilibrium distribution.

## Kernels

- Local translation and proper rotation use the existing symmetric proposals,
  with ordinary Cartesian coordinates and exact atom-wall rejection.
- Learned moves choose a uniform other-body anchor. Every density uses the full
  mixture and ordinary relative displacement. A uniform cube/Haar branch gives
  full support. Its cube half-width is `R + body_bound`, sufficient even for an
  off-center body-frame origin. No successful-draw retry or accessible-volume
  normalization is used.
- Spherical GCA applies a common centered proper half-turn. Hard cross-overlaps
  join components. A Poisson process at activity `z` on multiply covered old
  exclusion volume outside the transformed union joins **all** owners of a
  point. Pair-lens envelopes with lexicographic ownership avoid multiple
  counting of triple and higher coverage. Independent fair component flips
  complete the move; no second endpoint depletion gate follows it.
- Center shift translates **all bodies together** along a uniform isotropic
  direction, drawing uniformly on the complete feasible atomic-wall chord.
  It preserves every relative pose and the exclusion-union volume. It moves
  the assembly relative to the wall, which changes the geometry of subsequent
  centered GCA rotations.

For a unit direction `u`, atom world center `c`, and radius `a`, the shift
constraint is

\[
t\in[-u\cdot c-\sqrt{(R-a)^2-|c|^2+(u\cdot c)^2},\;
       -u\cdot c+\sqrt{(R-a)^2-|c|^2+(u\cdot c)^2}].
\]

Intersect over every atom. If the draw is `t`, the reverse interval is exactly
`I(X)-t`, with the same length. Thus the common shift is reversible and needs
no acceptance decision. Rust uses stable quadratic roots and conservative
whole-body pruning. Numerical geometry failures stop with an error; neither
implementation silently substitutes an endpoint rejection for these kernels.

GCA alone preserves body-frame radial vectors. Center shifts change this
invariant; local/learned moves also allow internal rearrangement. These facts
are not an ergodicity or mixing-time proof.

## Optional transported Gaussian auxiliary model

The implemented first case is a **fixed number of components with fluctuating
means**. The frozen base covariances, chart anchors and weights stay fixed.
This is a useful subcase of the general normalized auxiliary construction;
it is not RJ component birth/death or unlimited accumulation of training data.

For every ordered pair in the current configuration, compute its standardized
residual in each frozen component chart. Assign the smallest squared residual,
retaining it only below `cutoff`. Clip its length at `clip`. For component `k`,
with `n_k` assignments and residual sum `b_k`, define

\[
f_k(X)=\frac{g\,b_k}{h+n_k},\qquad
s_k(X)=\frac{\sigma}{\sqrt{h+n_k}},\qquad
a_k=f_k(X)+s_k(X)\eta_k,
\quad\eta_k\sim N(0,I_6).
\]

The actual mixture mean is `mu_k = mu_base,k + L_base,k a_k`. This explicitly
normalized Gaussian conditional can depend on the current physical state
without an unknown evidence integral. Assignment labels, clipping, shrinkage,
and sums use deterministic current-state data only. Defaults are `gain=.25`,
`noise=.1`, `cutoff=6`, `clip=3`, `shrinkage=1`; they are exploratory choices,
not optimized parameters or inferred physical basin probabilities.

In latent coordinates, the joint target is `pi_R(X) product phi(eta_k)`.
Each sweep starts with an independent Gaussian refresh of eta. Every physical
move retains it. The candidate model is reconstructed at Y and the learned
Hastings ratio uses

\[
q_{\theta(Y,\eta)}(X\mid Y) / q_{\theta(X,\eta)}(Y\mid X).
\]

The existing exact conditional Poisson gate supplies the physical factor.
The Gaussian conditional ratio cancels the transport Jacobian, including
changes in `s_k(X)`. Local moves, model-independent GCA and center shifts need
no auxiliary correction. For this relative-pose fitter, common shifts even
leave `f,s` unchanged. No model-guided GCA axis selection is implemented.

The runner performs a random-permutation sweep of single-body moves, then
independent state-independent Bernoulli GCA and center-shift attempts, in that
order. Each component kernel is reversible; this fixed composition preserves
the target but need not itself satisfy detailed balance.

## Running

The Rust runner has no Python dependency:

```bash
cargo build --locked --release --bins
cargo test --locked --release

target/release/tetramer-mc run \
  --config examples/spherical-oligomers.json \
  --model examples/frozen-relative-mixture.json \
  --out runs/spherical-oligomers --sweeps 4000 --sample-every 10
```

`examples/spherical-seeded.json` supplies the other preparation. Both start
from previously assembled fragments re-placed in an equal-volume sphere;
all 12 tetramers remain mobile. The sphere radius is 354.50820786337056 Å,
depletant radius 1.5 Å and activity 0.035 Å^-3.

Relevant input fields:

```json
{
  "boundary": {"kind": "spherical", "radius": 354.50820786337056},
  "gca_probability": 1.0,
  "center_shift_probability": 1.0,
  "auxiliary_transport": {"gain": 0.25, "noise": 0.1,
                          "cutoff": 6.0, "clip": 3.0, "shrinkage": 1.0}
}
```

Omit `auxiliary_transport` for frozen proposals. Empty `{}` enables its defaults.
The two collective probabilities are separately configurable; zero disables
that kernel. Ordinary local/learned sweeps still occur in all these cases.
Omitting `boundary` keeps the previous periodic behavior. Collective kernels
and this auxiliary prototype currently require a spherical boundary.

Checkpoints and JSONL frames include eta, and exact continuation validates its
presence and dimension. The GSD file records direct atom coordinates, zero
periodic images and `log/tetramer_mc/periodic=[0,0,0]`; wall radius is logged.
GSD atom positions remain FP32 display data. JSONL/checkpoints retain the full
joint state. The browser viewer uses open distances and displays the wall.

The Python reference requires NumPy/SciPy and imports the existing research
geometry/proposal/gate modules. Set `PROTEIN_NUCLEATION_ROOT` if they are not at
`/home/xvg/protein-nucleation`:

```bash
/home/xvg/protein-nucleation/.venv/bin/python research/spherical_ensemble.py \
  --config examples/spherical-oligomers.json \
  --model examples/frozen-relative-mixture.json \
  --out runs/python-spherical --sweeps 40 --gca-every 1 --shift-every 1 \
  --auxiliary
```

Python uses fixed integer collective cadences; Rust uses Bernoulli probabilities.
Every-sweep/disabled controls agree. The underlying geometric and conditional
laws agree; RNG streams, numerical implementations and null-move bookkeeping
differ. Python archives its final RNG/state but has no resume command.

## Validation

[Python results](../research/spherical-validation.json) include:

- Exact two-sphere AO distance law in the spherical container, including the
  midpoint distribution and Haar orientations. 3,000 independent equilibrium
  starts per kernel test stationarity after one update; this avoids treating
  a correlated trajectory as independent equilibrium samples.
- Seven kernel combinations, including actual frozen and transported learned
  steps and transported learned + GCA + shift.
- An exact eight-state three-sphere orbit with genuine triple exclusion volume,
  at zero and nonzero activity; 6,000 trials per activity. Stationary weights
  and resolved forward/reverse fluxes agree. Pair-additive weights serve as a
  deliberately incorrect comparison.
- Atomic-wall chord/reverse-interval checks for an asymmetric shape.
- Actual learned reverse densities checked independently with SciPy/Haar
  factors; maximum log-ratio error about 2e-14. Conditional-density/Jacobian
  cancellation includes nonzero log determinants. A stale reverse model
  differs by up to roughly 9 log units in the diagnostic.

All 25 Rust tests pass. The new checks add atomic BVH/geometry references, independent chord intervals,
a 128,000-transition exact triple-overlap orbit check, 12 cross-language
full-covariance auxiliary fixtures, actual runner reverse-density replay,
GCA-only/shift-only/combined scheduling, unwrapped GSD output and exact
continuation of both frozen and transported joint state. Existing periodic
checks remain in the suite.

The short matched tetramer pilot has two assembled starts and three arms:
frozen, frozen + GCA + shift, transported + GCA + shift, each for 40 sweeps
in both languages. It is a functionality/geometry pilot. Comparing its raw CPU
cost or acceptance counts does not establish a mixing-speed advantage or
crystal stability. See the generated pilot assessment for component motion and
native-registration changes.

## Pilot outcome

All 12 runs, 108 saved configurations, and full move-log replays passed the
independent atomic-wall/hard-overlap audit. Of 320 GCA updates, 265 flipped a
proper subset of bodies. All original registered motifs survived. One learned
move in the Python seeded frozen+GCA+shift arm formed a new external native
motif at sweep 21, increasing its registered component from 9 to 10 tetramers.
Every other arm retained its initial largest component (9 seeded, 4 unseeded).
GCA itself changed no registered motifs in this pilot. This demonstrates
collective mobility plus one learned incorporation, not a measured improvement
in mixing time. [Detailed audit](spherical-pilot-assessment.md) and
[machine-readable costs/provenance](spherical-pilot-validation.json).
