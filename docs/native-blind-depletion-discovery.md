# Native-blind depletion contact discovery and frozen proposals

The [discovery implementation](../src/contact_discovery.rs) builds pair-contact
proposal data from the rigid sphere-union shape, depletant bath, explicit search
budgets, and random seeds. It first improves a finite overlap score with an
offline geometric heuristic, then refines each endpoint with the existing exact
anchored-pair Monte Carlo kernel. The [fitter](../tools/fit_depletion_contact_atlas.py)
freezes a normalized proposal before any production trajectory starts.

![Shape-only discovery, independent validation, exact pair refinement, frozen fitting, and physical acceptance](figures/native-blind-depletion-discovery.svg)

No inter-particle native motif, registration label, fitted native covariance,
native atlas, production trajectory, or production starting pose enters discovery
or fitting. The input rigid body may itself be a native tetramer; its internal
shape is the physical particle geometry. Native labels may be used only after
freezing, for evaluation. A supplied native seed in a later physical benchmark
does not turn its initialization into blind assembly.

**Geometry and heuristic search.** Each independent slot uses
`MemoryState::initialize` to draw a Haar orientation and isotropic separation ray,
find its outermost hard contact, and move outward by an explicit gap plus a
floating-point guard. This preparation is geometry-only and is not an equilibrium
draw. Contact witnesses record the tangent atom pair, surface point, and mobile
lever for the later rolling covariance.

For a relative pair pose \(A=(t,R)\), let \(E_A\) and \(E_0\) be its inflated
atomic **unions**, and define

\[
C(A)=|E_A\cap E_0|.
\]

`OverlapEnvelope` partitions the overlap into a certified interior of volume
\(L\) and a disjoint uncertain envelope of volume \(U\). With a fixed allocation
of \(n\) unconditional samples in that envelope and \(k\) exact union-membership
hits, the score is

\[
\widehat C=L+U k/n,\qquad
\widehat{\mathrm{SE}}=U\sqrt{(k/n)(1-k/n)/n}.
\]

The output also records a binomial Wilson interval transformed to volume units,
the geometric bounds \([L,L+U]\), and point/cell counts. Zero-hit scores retain
their full denominator; a zero estimated standard error does not imply zero
uncertainty when \(U>0\). Exact empty or fully certified envelopes need no random
point queries. Membership counts overlapping atomic spheres only once. A sum of
atomic lens volumes, or `log(overlap_weight)/z`, is not this score.

Each attempted search step independently selects orientation-only, ray-only, or
combined perturbation with equal probabilities, and a 1°, 4°, or 12° per-axis
small-angle scale for its Gaussian Cayley increment with equal probabilities.
It projects the proposed
orientation/ray back to an outermost radial contact with the same gap, rejects
invalid endpoints, and retains a candidate only when its score strictly improves.
Each slot reuses fixed pseudorandom score uniforms mapped through the candidate's
own envelope. This makes the finite objective reproducible; it can still favor
sampling noise. Projection and greedy acceptance are **offline optimization**,
with no equilibrium or reversible-search interpretation. At positive activity,
maximizing \(C\) also maximizes the pointwise pair depletion factor \(e^{zC}\). The
search itself has no activity-dependent Metropolis acceptance.

**Validation and exact refinement.** Initial and optimized poses each receive
two fresh fixed-size validation populations. Four distinct validation streams
are separate from initialization, search moves, search scores, and refinement.
SHA256-derived master-seed/slot/stream keys prevent budget-dependent stream
shifts and unintended stream reuse for adjacent job seeds. Holdout scores are
reporting data: they never select an endpoint, remove a slot, alter its weight,
or choose a covariance. Changing only the validation allocation leaves search
and refinement bitwise unchanged.

The optimized pose then initializes a one-slot `MemoryState`. Its exact target,
with respect to Cartesian translation and normalized proper Haar rotation, is

\[
\rho(A)\propto
\mathbf 1_{|t|<R_{\rm mem}}\mathbf 1_{\rm hard}(A)
\exp[-z_{\rm mem}|E_0\cup E_A|]
\propto
\mathbf 1_{|t|<R_{\rm mem}}\mathbf 1_{\rm hard}(A)e^{z_{\rm mem}C(A)}.
\]

The finite ball makes the pair law proper. Its default radius is
\(2(b+r_d)+g+2\epsilon_{\rm geom}\), where \(b\) is the shape bound and \(g\)
the initialization gap. It bounds the mobile origin, not every atom or the
ideal depletant bath. Other slots and production particles are absent from this
two-body environment.

`MemoryState::update` supplies the existing symmetric local translation/Cayley
proposal, hard/domain rejection, and exact conditional Poisson acceptance.
Global refresh has probability zero in this local refinement. The default
translation standard deviation is 0.2 Å, the per-axis small-angle parameter is
1°, and
\(\lambda=64z_{\rm mem}\) at positive activity. At zero activity every valid
local proposal is accepted. The approximate search score never enters this
physical transition. See [pair-memory balance](contact-memory-balance.md) for
the conditional gain/loss construction.

There are 256 attempted refinement steps per slot by default. Steps 1–64 are
excluded as preparation, and steps 68, 72, …, 256 are retained: 48 poses per slot.
Sampling is by attempted step, including unchanged poses after rejection. The
full move trace is archived. These finite, correlated traces are proposal-design
data; the burn allocation does not demonstrate stationarity, and retained
counts are not effective sample sizes or basin masses.

**All-slot fitting and the matched initial control.** Every slot contributes
one initial contact chart and one fitted chart. With \(M\) slots and frozen
initial mass \(\alpha=0.5\), their base weights are \(\alpha/M\) and
\((1-\alpha)/M\). No native score, overlap score, accepted-move count, or sample
count changes this allocation. The initial-only control uses those same initial
contact charts at weight \(1/M\), with identical widths and reciprocal
representation. It isolates the added search/refinement information from the
particular random outer contacts used to prepare this campaign.

The chart scale is shape-derived:
\(\ell=2\max(\mathrm{RMS}\text{ atom-center distance about their mean},
\max_a r_a)\). Initial charts have 1 Å independent translation width and 3°
per-axis angular width, with the existing contact-pivot rolling coupling. The
fit uses all retained poses in translation/Cayley coordinates about the optimized
contact, including repeated rejected states. Let \(B=LL^T\) be the corresponding
shape-contact rolling covariance, and let \(S\) be the retained population
scatter, with denominator \(n\). If
\(L^{-1}SL^{-T}=V\operatorname{diag}(d)V^T\), the frozen covariance is

\[
\Sigma=LV\operatorname{diag}\!\left(
\max\{0.75\max(d,0)+0.25,\;0.01\}\right)V^TL^T.
\]

The shrinkage 0.25 and dimensionless variance floor 0.01 are fixed proposal
regularizers. They are not measured thermodynamic stiffnesses. The fitter fails
the complete fit if a retained pose lies at its chart's Cayley seam; it does not
trim that pose or omit its slot. All Gaussian tails remain untruncated.

**Frozen density and physical balance.** For a chart anchored at \((t_k,R_k)\),
write \(c_k=\mathrm{Cayley}^{-1}(RR_k^T)\) and
\(x_k=(t-t_k,\ell c_k)\). The normalized Haar density in Cayley coordinates is

\[
J_H(c)=\frac{1}{\pi^2(1+|c|^2)^2},\qquad
g_k(t,R)=\varphi_6(x_k;m_k,\Sigma_k)\frac{\ell^3}{J_H(c_k)}.
\]

Thus each complete Gaussian chart integrates to one under
\(d^3t\,dHaar(R)\), and \(G(T)=\sum_k w_k g_k(T)\) is normalized. The fitter
stores the exact reciprocal mixture

\[
F(T)=\tfrac12G(T)+\tfrac12G(I(T)),\qquad
I(t,R)=(-R^Tt,R^T).
\]

Physical inversion preserves this measure. Both virtual branches contribute
their exact densities; transforming a Gaussian covariance is not a replacement
for the inverse branch. With four eight-slot populations, the discovered model
has 64 base/128 virtual components, and the initial-only model has 32 base/64
virtual components. These are predetermined allocations, not discovered basin
counts.

For an independent capture move using selected anchor \(j\), production adds
its configured uniform Cartesian/Haar branch of mass \(\eta>0\):

\[
Q_j(x)=\eta U_{\rm lab}(x)+(1-\eta)F(T_j(x)).
\]

The Hastings factor is the **complete** \(Q_j(x_{old})/Q_j(x_{new})\), including
every overlapping component and both reciprocal branches. Invalid raw draws
are rejected without redraw, clipping, or hard conditioning of \(Q_j\).
For the existing posterior-transport branch, the labelled involution instead
has the full learned-density correction \(F(T_{old})/F(T_{new})\), with the
runner's separate uniform branch unchanged; see the
[reciprocal proposal derivation](reciprocal-contact-proposal-design.md).

After endpoint selection, the unchanged production kernel evaluates hard cores,
the physical wall, and the exact many-body ideal-depletant interaction. Its
spectator region is the **union of all other bodies' exclusions**, not a sum of
pair scores. In the moving body's coordinates, let \(v_g\) and \(v_l\) be the
gained and lost spectator-overlap volumes. The auxiliary gate samples
\(N_g\sim\mathrm{Pois}(\lambda v_g)\) and
\(N_l\sim\mathrm{Pois}((\lambda+z)v_l)\) by exact membership thinning, and uses

\[
a=\min\left[1,
\exp(\Delta\log q)\left(1+z/\lambda\right)^{N_g-N_l}\right].
\]

Here \(\Delta\log q\) is the appropriate full proposal correction above.
The tilted lost-count law supplies auxiliary detailed balance. This is not a
noisy plug-in overlap energy. Fixed proposal information cancels from the target;
there is no search projection Jacobian or fitted basin normalization in
production. Approximate discovery, unequal success among slots, and an imperfect
Gaussian fit can affect efficiency while leaving this frozen physical target
unchanged.

**Reproduction and audit.** Run from `/home/xvg/tetramer-mc`. Every output directory
must be fresh. The standalone binary defaults to 8,192 samples per validation
cloud; the four-population pilot controller explicitly increases that fixed
allocation to 65,536. Its other search/refinement settings match the defaults.

```bash
cargo build --release --bin depletion-contact-discovery --bin tetramer-mc
python3 tools/run_depletion_contact_discovery.py \
  --shape examples/tetramer-shape.json \
  --out runs/native-blind-depletion-discovery-reproduction \
  --populations 4 --workers 4 --starts 8 \
  --search-steps 128 --search-points 2048 --validation-points 65536 \
  --refine-steps 256 --burn 64 --save-every 4 \
  --rd 1.4 --activity 0.0275 --seed 20260926301

python3 tools/fit_depletion_contact_atlas.py \
  --shape examples/tetramer-shape.json \
  --discovery runs/native-blind-depletion-discovery-reproduction/populations/r00 \
              runs/native-blind-depletion-discovery-reproduction/populations/r01 \
              runs/native-blind-depletion-discovery-reproduction/populations/r02 \
              runs/native-blind-depletion-discovery-reproduction/populations/r03 \
  --initial-weight 0.5 --shrinkage 0.25 --covariance-floor 0.01 \
  --translation-width 1 --angle-width-degrees 3 \
  --out runs/native-blind-depletion-discovery-fit-reproduction

python3 tools/benchmark_depletion_contact_atlas.py \
  --config runs/cluster-oligomer-seed8-free256-geometry/config.json \
  --initial-model runs/native-blind-depletion-discovery-fit-reproduction/model-initial-only.json \
  --discovered-model runs/native-blind-depletion-discovery-fit-reproduction/model.json \
  --out runs/native-blind-depletion-discovery-benchmark-reproduction \
  --streams 2 --sweeps 200 --sample-every 100 --workers 4 \
  --seed 20260926101
```

The fitter requires NumPy and SciPy. The discovery controller archives the input
shape and executable before launching four bounded workers. Each population
writes its exact source bundle, configuration, stream keys, attempts, scores,
contact witnesses, refinement trace, and retained samples. Its completion
manifest is written last and hashes every payload. The fitter verifies those
hashes, archives the complete inputs and source closure, validates chart
normalization and inversion, and writes immutable models plus fit/preflight
receipts. An incomplete directory is not a usable discovery population.

The matched benchmark launches two fresh streams per arm from the same supplied
seed8/free256 configuration. It holds bath, physical start, schedule, and budget
fixed while changing the frozen proposal file. Matching arm seeds do not imply
identical random-number consumption or independent paired errors. The supplied
seed is evaluation context only; its poses never feed discovery or fitting.
No native classifier is used to choose a move or accept it.

Focused [Rust controls](../tests/depletion_contact_discovery.rs) check analytic
sphere lens volumes, union counting with duplicate atoms, zero radius and zero
activity, deterministic replay, adjacent-job stream separation, holdout-budget
independence, preservation of every slot/repeated state, exact pair-kernel replay,
and CLI hash/completion/no-overwrite behavior. Run them with
`cargo test --test depletion_contact_discovery`. For an anchored hard sphere of
radius \(a\), the exact refinement reference is
\(p(r)\propto r^2e^{zC(r)}\), \(2a\le r<R_{\rm mem}\), with
\(C(r)=\pi(4A+r)(2A-r)^2/12\) for \(r<2A\), zero otherwise, and \(A=a+r_d\).
Existing [pair-memory stationarity tests](../tests/contact_memory_stationary.rs)
check that law independently of a burn-in assumption.

This bounded protocol does not enumerate all contact basins, estimate their
equilibrium weights, establish native accessibility, prove faster mixing, or
measure assembly stability or physical kinetics. Outermost-ray search can miss
interlocking pockets; short local exact refinement can remain trapped. Stronger
pointwise pair overlap is distinct from a larger basin mass, and compatibility
with a third neighbor requires the later many-body physical checks. Discovery,
validation, fitting, and production costs should be reported separately.

**2026-09-26 pilot results.** The fixed 32-start search at 1.4 Å and
0.0275 Å⁻³ improved fresh-cloud overlap estimates in 31 of 32 starts; all 31
improvements exceeded twice their pointwise integration standard error.
Mean union overlap increased from 86.84 to 260.91 Å³. Thus the average
fixed-pose pair log-weight gain was 4.79, not an integrated association free
energy. All starts were retained, including the one whose heldout score fell.
The four discovery processes used 95.3 summed wall seconds (about 25 seconds
in parallel); exact refinement accepted 2,013 of 8,192 attempts and retained
1,536 scheduled states, including repeated states. The frozen atlas contains
64 base / 128 reciprocal components. Rust/Python density agreement was within
7.11e-15 in log density and the Jacobian check within 1.78e-15. Ten Rust tests,
13 fitter tests, and a production ingress smoke passed.

In the matched 200-sweep, two-stream many-body comparison, both arms accepted
zero learned independent redraws out of 23,897 attempts. Correlated learned
transport accepted zero out of 23,864 initial-atlas attempts and 23,817
discovered-atlas attempts. The average of the two run medians for the redraw's
log depletion factor improved from 0.81 to 1.66, while the proposal-density
penalty remained about 24.6–24.9. These summaries concern hard-valid proposals;
medians of terms do not add to the median total acceptance ratio. The stronger
pair contacts therefore did not yet produce useful capture moves. Sampler CPU
times summed to 245.2 seconds for the initial control and 248.1 seconds for the
discovered arm; this is not a mixing-speed comparison.

Native geometry entered only the separate, subsequent diagnostics. None of the
32 initial or 32 optimized centers satisfied the existing complete native-entry
definition; all 14 ideal native templates passed its positive control. At saved
sweeps 0, 100, and 200, every many-body run retained 13 native tetramer bonds
within the original eight-body seed and had zero native bonds involving the
initially free bodies. Intermediate transient events were not classified.
The result supports successful discovery of stronger non-native pair contacts,
but neither useful native-blind assembly sampling nor physical instability.
The next proposal-design target is broad coverage of extended contact basins
and their accessible pose volume, beyond outermost-ray overlap maximization.

Artifacts under `runs/native-blind-depletion-discovery-20260926/` include
`frozen-fit/`, `validation.json`, `posthoc-native-coverage/`,
`benchmark-after-storage-recovery/`, and `benchmark-native-diagnostics/`.
The first benchmark allocation in `benchmark/` stopped at filesystem exhaustion
and has no completed populations. It remains preserved and excluded. A separate
fixed four-population allocation with fresh seeds was recorded after moving
data to `/vast`; it is the comparison reported here. Original run paths remain
valid through symlinks. The benchmark controller now checks a conservative
output-size estimate before launching; concurrent writers can still consume
space after that check.

The same disk failure interrupted two existing growth jobs. Their archived
source bundles rebuilt to byte-identical original executables. One complete
sweep from each saved checkpoint reproduced all non-profiling move fields
(266 rows at sweep 67,501; 272 rows at sweep 4,001). Both were resumed toward
their original absolute target of 100,000 sweeps, with unchanged models,
configurations, RNG checkpoints, and output cadence, in fresh directories:
`runs/coverage-seeded-256-500uM-d00275-r14-2-resume67500/` and
`runs/cluster-oligomer-seed8-free256-2-resume4000/`. Original partial tails were
preserved and not appended to. Binary, replay, launch, and checkpoint receipts
are archived at `/vast/xvg/tetramer-mc-runs/storage-recovery-20260926/`.
