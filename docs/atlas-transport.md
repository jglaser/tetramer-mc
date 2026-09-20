# Controlled reversible updates around a learned atlas

`atlas_transport` preserves the successful imported proposal as an immutable
reference, including all component identities, chart conventions, full
covariances, weights, and narrow registration scales. The actual proposal may
fluctuate around this reference. The controlled stages enable mean, covariance,
and weight updates separately; the component count is fixed throughout.

This addresses a confound in the geometry-only conditional pilot: replacing
the trained contact model, its data selection, and its regularization at the
same time as changing the update rule. The present test retains the supplied
native-informed atlas. It measures whether reversible parameter changes retain
or improve its usefulness, not whether native basins can be discovered blindly.

## Reference coordinates and conditional fit

For reference component `j`, retain chart metadata, Gaussian mean `μ₀ⱼ`, lower
Cholesky factor `B₀ⱼ`, and weight `w₀ⱼ`. Each ordered current pair is expressed
in every reference chart and standardized using its covariance. Assign the
nearest component if its whitened residual norm is at most 6; clip each
residual coordinate to ±3. These defaults are configurable. There is no
center-distance surrogate for contact and no accumulating observation bank.
Assignments are to reference components, never to the fluctuating model.

With assigned residuals `rₐ` and shrinkage `h=8`, define

\[
 m_j=\frac{\sum_a r_a}{h+n_j},\qquad
 S_j=\frac{h(I+m_jm_j^T)+\sum_a(r_a-m_j)(r_a-m_j)^T}{h+n_j}.
\]

The second expression is a covariance estimate including zero-mean,
unit-covariance reference pseudo-observations. Encode `mⱼ`, lower Cholesky
entries of `Sⱼ` (log diagonal), and `K−1` relative log-weight corrections from
counts shrunk toward `h w₀ⱼ`. Unobserved components have identity covariance
corrections and zero mean corrections; no component disappears.

This defines a reproducible vector `F(X)` with `28K−1` coordinates. Let

\[
 u=G F(X)+\Sigma\eta,\qquad \eta\sim N(0,I),
\]

where diagonal `G` and `Σ` contain independently configurable gains and noise
scales for mean, covariance and weight blocks. Default active-block gain is
0.25 and noise is 0.1. These are dimensionless in each reference Gaussian's own
units; they are not Å-scale covariance floors.

Decode bounded mean offsets `a`, lower triangular matrices `T` and logit
offsets `δ` using smooth scaled tanh maps:

\[
 \mu_j=\mu_{0j}+B_{0j}a_j,\qquad
 C_j=B_{0j}T_jT_j^TB_{0j}^T,\qquad
 w_j=\operatorname{softmax}_j(\log w_{0j}+\delta_j).
\]

Mean-coordinate bounds are ±3 reference standard deviations. Diagonal `T`
entries lie between 1/2 and 2, and off-diagonal entries between −1/2 and 1/2.
Logit offsets are bounded by ±4, with the last component the fixed reference
logit. These element bounds are not eigenvalue bounds. They ensure numerical
control without replacing narrow reference scales by an absolute isotropic
ridge. The existing full pose-density evaluator includes the rotation/Haar
Jacobian and uniform defense.

## Balance and memory

The retained joint target is

\[
 \widetilde\pi(X,\eta)=\pi(X)\phi(\eta).
\]

All reference metadata are fixed context, not additional random state. The
count `K` is fixed. Current pair data and `F(X)` are derived quantities; `η` is
the persistent auxiliary state. Integrating it out recovers exactly the hard
shape plus many-body depletion target `π(X)`.

A physical proposal retains `η` and the selected spectator anchor `j`:

\[
 \log R_q=\log q_{Y,\eta}(X\mid j)-\log q_{X,\eta}(Y\mid j).
\]

The candidate model must be reconstructed from `F(Y)`. Add this ratio to the
existing exact conditional Poisson gate. Do not insert a fit objective into
the physical energy. In parameter coordinates, the normalized conditional-law
ratio and transport Jacobian cancel; working in stored latent coordinates
makes that cancellation explicit. Disabled coordinate blocks can be
deterministic; the retained standard-normal latent law remains normalized.

Local moves, model-independent GCA and common center shifts remain unchanged:
they preserve `π(X)` while retaining `η`. No count correction or extra collective
rejection is required. At the end of each sweep, independently redraw `η` from
its standard-normal law with the configured refresh probability. This is an
exact conditional Gibbs update. Individual kernels preserve the joint target;
the scheduled whole sweep need not itself be reversible.

The fit uses only current geometry. This is a tractable conditional auxiliary
model, not a Bayesian posterior over basins or an accumulating training history.
Successful native proposals do not alone establish equilibration or a speedup.

## Initialization and exact frozen limit

`initialization: "reference"` chooses

\[
 \eta_0=-\Sigma^{-1}G F(X_0),
\]

on active noisy coordinates, recovering the imported atlas at `X₀`. If a
coordinate has zero noise and a nonzero fitted shift, this initialization is
rejected rather than silently changing the reference. Zero gain on inactive
coordinates avoids that case. `zero` sets `η=0` and uses the current fitted
center; `random` draws the conditional auxiliary state.

Reference initialization is deliberately prepared, not a stationary auxiliary
draw. It survives until the first physical sweep; the first Gibbs refresh is
after that sweep. Checkpoints retain the realized `η`, verify the imported
model hash, and rebuild `F` from checkpoint geometry on resume.

With **all gains and noises zero**, the engine returns the original proposal
object. The tests verify identical random draws, densities, physical move
decisions, and complete trajectories versus the frozen runner, even after
auxiliary refreshes. This is a direct regression control for the preserved
atlas, beyond an approximate comparison of its parameters.

## Running the stages

Use the normal spherical learned command with `--model`. Add `atlas_transport`
to its JSON. It is mutually exclusive with the older auxiliary, RJ, contact-bank
and geometry-only conditional modes. The current implementation is spherical.

```json
{
  "atlas_transport": {
    "mean_gain": 0.25, "mean_noise": 0.1,
    "covariance_gain": 0.0, "covariance_noise": 0.0,
    "weight_gain": 0.0, "weight_noise": 0.0,
    "assignment_cutoff": 6.0, "residual_clip": 3.0, "shrinkage": 8.0,
    "initialization": "reference", "refresh_probability": 1.0
  }
}
```

This enables means only. Set the covariance gain/noise to 0.25/0.1 for the next
stage, then enable the weight block for the final stage. Omit `atlas_transport`
for the frozen control. All four stages use the same model file, physical
parameters and initial configurations.

```bash
cargo build --locked --release --bin tetramer-mc
python3 tools/run_atlas_transport_campaign.py --out runs/my-atlas-control \
  --source-campaign runs/free-tetramer-fixedk-400 --sweeps 400 --workers 4
python3 tools/analyze_atlas_transport_campaign.py --campaign runs/my-atlas-control --workers 4
```

The launcher reuses the earlier successful campaign's exact starts and seeds,
archives its executable/model/inputs, and changes only the declared parameter
update blocks. Analysis requires the existing Python reference classifier. It
replays every accepted physical move and distinguishes native proposals,
accepted native candidates, motif formation, breakage, and uniform fallbacks.
The [pilot report](atlas-transport-pilot.md) records the measured outcome.

Tests include exact-start AO-sphere physical/auxiliary stationarity, finite-state
balance checks with stale-reverse negative controls, exact frozen-limit draws,
reference initialization with the actual narrow protein atlas, full reverse
model replay in the runner, and exact restart. These complement the established
hard-geometry and many-body depletion tests.
