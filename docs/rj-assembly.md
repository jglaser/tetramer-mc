# Free tetramers and reversible proposal-component births

The free-start campaign asks whether twelve initially isolated rigid tetramers
can associate under the same many-body depletion target as the seeded runs.
The new auxiliary process changes the number of proposal components during
sampling. Neither its component-count prior nor a native-informed proposal is
an extra interaction in the physical model.

## Physical preparation and matched controls

All bodies are mobile in a sphere of radius 354.50820786337056 Å, with depletant
radius 1.5 Å and activity 0.035 Å⁻³. The ideal bath permeates the spherical wall.
The four preparations have independent Haar orientations and center separations
strictly greater than twice the tetramer exclusion bounding radius. Thus no
inter-tetramer core or exclusion overlap is present initially. Sequential
placement under this separation constraint is a prepared quench, not a sample
from the interacting fluid equilibrium distribution.

`examples/spherical-free.json` uses the frozen model;
`examples/spherical-free-transport.json` adds the normalized conditional mean
law. Both include one implicit GCA and one common-center shift per sweep,
in addition to twelve single-body updates. The matched 400-sweep campaign,
`runs/free-tetramer-fixedk-400`, uses identical physical starts and MC seeds
within each pair. All eight runs use the existing **native-informed** atlas.

Largest registered components across the four repetitions were 3, 3, 4, 4 for
the frozen proposal and 3, 3, 4, 5 for transported means. All 68 registered motif
formations came from learned Gaussian proposals. No registered motif broke.
All 328 saved frames passed independent hard-core/wall checks, and all 44,800
move records replayed correctly. This establishes accessibility of oligomers
with this atlas, not equilibrated association or a transport speedup.

## Implemented variable-dimensional target

Let `D` be a finite, frozen atlas, `w_c` its normalized label weights, and
`c=(c₁,…,c_K)` an **ordered list with replacement**. Duplicate labels are
different slots. The model has equal active mixture weights `1/K`, the selected
atlas covariances and charts, and six continuous mean residuals per slot.
Store independent standard-normal latent residuals `η_j`, and reconstruct
physical means using the existing deterministic, current-state fit:

\[
 a_j=f_j(X,c)+s_j(X,c)\eta_j,\qquad
 \mu_j=\mu_{c_j}+L_{c_j}a_j.
\]

Every selected component participates in the fit at both endpoints. Inserting
a duplicate label can change assignments and therefore several reconstructed
means; all are reconstructed, rather than carrying old fitted means forward.

The target relative to ordered-label counting measure and Lebesgue measure on
the `6K` latent coordinates is

\[
 \widetilde\pi(X,K,c,\eta)=\pi(X)\,p(K)
       \prod_{j=1}^K w_{c_j}\,\phi_6(\eta_j),
 \qquad
 p(K)=\frac{\lambda^K/K!}{\sum_{k=K_{\min}}^{K_{\max}}\lambda^k/k!}.
\]

Summing/integrating the auxiliary coordinates gives exactly `π(X)`. In physical
mean coordinates this is a normalized conditional law depending on `X`.
The Jacobian of mean transport cancels its conditional-density change. In
the stored latent coordinates that cancellation is already built in.

At fixed `X`, choose birth or death with probability one half. Keep null choices
at the count limits; do not force the other direction.

- Birth: draw a label from `w`, draw six independent standard normals, and insert
  into a uniformly chosen slot among `K+1` slots.
- Death: choose one of the `K` slots uniformly and remove it.

The matching birth/death insertion and removal probabilities cancel. So do the
new label and normal densities. The latent insertion map has Jacobian one:

\[
 A_b=\min\{1,\lambda/(K+1)\},\qquad
 A_d=\min\{1,K/\lambda\}.
\]

There is no additional factorial for these ordered slots. The factorial already
in `p(K)` expresses the chosen count prior. Sorting, merging duplicates, or
discarding a low-weight component would require a different transition rule.

For a physical move, retain `K,c,η`. Propose using the whole reconstructed
mixture at `X`; compute its reverse density using the mixture reconstructed at
`Y`. Combine `log q_Y(X) - log q_X(Y)` with the existing exact conditional
Poisson gate. Local moves, implicit GCA, and common-center shifts leave these
stored auxiliary variables fixed and retain their previous corrections.
Refresh the residuals from their standard-normal law before each sweep's RJ
attempts. Named random streams and full joint checkpoints preserve exact resume.

The implementation is a genuine birth/death process with continuous dimension
`6K`. It is deliberately narrower than unrestricted mixture discovery:
covariances and chart labels come from a fixed finite atlas, and the stationary
count distribution is prescribed. It neither accumulates training history nor
infers the optimal number of physical basins from the production trajectory.

## With and without explicit inter-tetramer native information

`contact_atlas` reads only the rigid sphere-union shape. It samples independent
Haar orientations and separation directions, uses a body-frame sphere BVH to
find the last forbidden atomic interval along each ray, then places a chart
0.25 Å outside that contact. Means are hard-valid. A full-rank Gaussian combines
1 Å translation noise with a 3° small-angle fluctuation about the witness
contact; `--uncoupled` removes that linearized rolling covariance. No rejection
conditioning of this Gaussian occurs in production: the usual hard test and
full proposal density apply.

The 96-chart native-off atlas has no inter-tetramer crystal geometry, native
training examples, or depletion-energy ranking. The supplied rigid tetramer
itself remains the native intratetramer scaffold in both arms. These outermost
surface contacts do not enumerate deep interlocking pockets. A negative result
with this finite atlas is therefore not evidence that the model forbids native
assembly.

Native-on mixes the same 96 charts with the 28 existing native-informed charts,
with prior mass one half for each source. `blend_contact_atlas.py` rescales
angular latent means and covariances if their Cayley length scales differ;
this preserves their physical proposal densities. Native templates remain
postprocessing information in both arms; they never enter physical acceptance.

The campaign preparer requires NumPy; the analyzer and native-bond exporter
additionally require the existing NumPy/SciPy research environment. Commands
below use that local virtual environment; the Rust simulator itself does not.

```bash
cargo build --locked --release --bins
target/release/contact_atlas --shape examples/tetramer-shape.json \
  --out runs/my-atlases/geometry.json --count 96
python3 tools/blend_contact_atlas.py \
  --geometry runs/my-atlases/geometry.json \
  --native examples/frozen-relative-mixture.json \
  --out runs/my-atlases/native-on.json

# First make four independent, initially contact-free starts and controls.
/home/xvg/protein-nucleation/.venv/bin/python tools/run_free_tetramer_campaign.py --out runs/my-fixedk-400 \
  --sweeps 400 --replicates 4 --workers 8

# Reuse those exact starts and paired MC seeds for both RJ proposal priors.
/home/xvg/protein-nucleation/.venv/bin/python tools/run_free_tetramer_campaign.py --out runs/my-rj-1000 \
  --reuse-starts runs/my-fixedk-400 --reversible-jump \
  --atlas native-on=runs/my-atlases/native-on.json \
  --atlas native-off=runs/my-atlases/geometry.json \
  --sweeps 1000 --sample-every 10 --workers 8
```

For a direct run, add `"auxiliary_transport": {}` and
`"reversible_jump": {}` to a spherical learned-proposal config. RJ defaults are
`min_components=1`, `max_components=24`, `initial_components=8`,
`poisson_mean=8`, and `attempts_per_sweep=4`. The initial fixed count is not an
equilibrium auxiliary-count draw; the chain relaxes to its specified prior.
If `min_components=max_components`, births and deaths are all null and labels
remain fixed; this edge case does not mix the full label prior. The campaign
uses a genuinely variable count range.

## Validation and interpretation

Tests verify exact count transition flux including null boundary choices,
Gaussian and label priors, atomic radial-contact geometry against brute-force
intervals, the actual variable-K runner's forward/reverse densities, and exact
continuation of the whole joint state. All 32 Rust tests pass. Physical marginal stationarity is checked
separately against 6,000 independent analytic two-sphere depletion equilibrium
draws, followed by three composed rounds. All 15 predeclared observables pass;
maximum paired standardized change is 2.245. An independent noise-averaged
Poisson flux check has relative defect 1.2e-15 for the correct reverse model
and 44% for the deliberately stale one. The sampled stale-control means alone
are underpowered. See [the actual validation output](rj-stationarity-validation.txt).

Compare native and nonspecific component histories, accepted formation and
breakage events, and total sampler CPU cost. Proposal priors change efficiency,
not the equilibrium physical distribution. Short quenches cannot establish
relative equilibrium populations or a nucleation rate; in particular, persistent
oligomers without breakup leave reversible exchange unresolved.

## Native-on/off pilot outcome

All eight 1,000-sweep runs completed. The independent audit replays all
144,000 records, including 32,000 model jumps, to all 808 saved frames and
checkpoints; each saved frame passes hard-core and atomic-wall checks. These
reuse the four exact free starts
above; they are not eight new independent preparations.

| Proposal prior | Final largest native components, four starts | Native motif formations / breakages |
|---|---|---|
| Geometry + explicit native charts | 2, 2, 3, 3 | 25 / 0 |
| Geometry-only charts | 1, 1, 1, 1 | 0 / 0 |

Replaying active labels attributes all 25 registered formations to supplied
native charts and none to geometry-only charts. Geometry-only sampling did
form and lose nonspecific contacts, with sampled
aggregates reaching four tetramers. Its lack of registered motifs is an
accessibility result for this outer-contact atlas. Native-on sampling formed
persistent dimers and trimers but did not demonstrate reversible native
exchange. Cluster motion therefore complements initial capture without yet
solving bound-oligomer reorganization.

K visited 1–19 overall, with per-start means 8.65, 8.15, 7.69, 8.01; paired arms
have identical K paths because their named RJ streams and count priors are
identical. Native chart labels differ. The auxiliary process has no physical
acceptance feedback into its prescribed label/count prior. These correlated
histograms are not the independent physical-stationarity validation.

This comparison isolates supplied native proposal information at the same
physical target and RJ law. Comparing against the earlier fixed-K pilot does
**not** isolate the effect of RJ: dictionary composition, active weights, and
run length changed. Total CPU was 500.01 s across eight jobs, 96.31 s wall time.

The [compact audit](free-assembly-validation.json) records both campaigns.
Full report: `runs/free-tetramer-rj-1000/assessment/report.md`.
The paired component-history figure is
`runs/free-tetramer-rj-1000/assessment/free-assembly-components.png`.
Open `runs/free-tetramer-rj-1000/native-on-viewer.html` or
`native-off-viewer.html` for actual atom-sphere trajectories. Both offer
sphere-center/default versus coordinate-origin frames and optional native
monomer coordination bonds. The origins coincide initially. The corrected
viewer accumulates common center shifts to show wall motion in coordinate-origin
view; sphere-center view keeps the wall fixed. Re-export old HTML files to use
this correction. See [the coordinate convention](spherical-ensemble.md#display-coordinates).
The overlay tests individual monomer contacts; its bond count is distinct from
the stricter registered tetramer motifs tabulated above.
