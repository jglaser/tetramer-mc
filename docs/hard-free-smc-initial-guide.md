# Optional SMC bridge initialized from the frozen hard-free guide

The [retrospective bridge diagnostic](smc-bridge-bottleneck.md) suggests that
the old initialization density suppresses an important native remainder in
intermediate SMC stages. The new optional mode uses the already normalized
hard-free guide as its initial density. Its physical endpoint and fixed R4
target remain unchanged. This implementation still needs exact-binary sphere
references and an independent protein benchmark before it can be credited
with improved normalizer estimates or sampling efficiency.

## Law and balance

Let \(q(u)\) be the complete frozen latent guide, including its uniform
defensive branch, every Gaussian component and empty-line fallbacks. Let
\(J(u)\) convert the original chart to Cartesian translation times normalized
SO(3) Haar measure. The normalized physical proposal is \(g(x)=q(u(x))/J(u(x))\).
Positive uniform probability gives support throughout the regional target.
The SMC path is

\[
\gamma_\beta(x)=H(x)g(x)^{1-\beta}e^{\beta zC(x)}.
\]

Initialization draws the entire proposal without clipping or refilling
invalid attempts. Each valid draw contributes one and each invalid draw
zero, with all \(M\) attempts in the denominator. An empty population is a
recorded zero estimate, never permission to retry. Initial resampling starts
from \(Hg\); the physical endpoint is \(He^{zC}\).

For an increment \(\Delta\beta\), use

\[
g(x)^{-\Delta\beta}
\frac12\sum_{r=1}^2 e^{\Delta\beta zL_r}
\left(1+\frac{\Delta\beta z}{\lambda}\right)^{K_r},
\quad
K_r\sim\operatorname{Poisson}[\lambda(C-L_r)].
\]

The two incremental clouds are fresh and independent. Unlike the retrospective
plot, a live run does not reuse them across stages. The conditional expected
potential is \(\gamma_{\beta+\Delta\beta}/\gamma_\beta\), so the existing
random-potential SMC normalizer argument applies. Terminal region masses
remain the normalizer multiplied by the terminal indicator average.

Local mutations are symmetric in physical pose measure. The existing
gained/lost-count gate at activity \(\beta z\) acquires the deterministic
factor

\[
\log\alpha=\min\{0,\log R_{\rm bath}
 +(1-\beta)[\log g(y)-\log g(x)]\}.
\]

The factor belongs inside the combined acceptance rule. The Poisson
forward/reverse auxiliary identity gives accepted-flow symmetry for
\(\gamma_\beta\); rejection completes the invariant kernel. No additional
Jacobian appears because the local proposal is already symmetric in physical
translation/Haar measure. This is not an acceptance ratio of freshly drawn
absolute noisy partition-function estimates. At β=1 the proposal-density
correction vanishes, and at zero activity the density corrections still
transform \(Hg\) into the hard-volume target.

The existing Lean balance results apply at the algebraic accepted-flow level;
this extension does not add a formal theorem for the complete SMC program.
Geometry predicates, normalized guide construction, thinning, independent
random streams and floating-point execution remain implementation obligations.

## Interface and records

`latent-region-smc --initial-guide PATH` requires `--bridge proposal-density`.
The first version accepts only `defensive-hard-free-line-guide-v1` with all
three raw translation axes and positive uniform probability. It rejects a
simultaneous reference-chart initializer, explicit initial-current-probability
override, native-excluded target or angular-guide schema. The public Rust
entry point is `run_with_initial_guide`; existing `run` and `SmcOptions` callers
retain their interface.

The distinct manifest schema is
`latent-region-smc-hard-free-initial-guide-v1`. It binds the complete guide,
source region, shape, physical measure, executable and source bundle. Each
initial draw records its generation trace and separately evaluated physical
density. Each scored local candidate records its complete density and bath
gate. Rejected states remain in the retained trajectory.

Only deterministic \(\log g\) is cached, bound to the exact retained pose.
Resampling clones the cache; an accepted mutation replaces it. No auxiliary
count or stochastic weight is cached. Compared with scoring retained poses
at both potential evaluation and mutation entry, this removes \(2NT\)
redundant guide evaluations: 524,288 at \(N=2048,T=128\). This call count is
not a measured runtime saving.

The new mode writes a flushed `attempts.jsonl` entry before every initial
draw, incremental potential and elementary mutation. A failure records its
last attempted identity and retains partial outputs. There is no restart or
retry mechanism. Geometry-rejected mutation candidates have their attempted
identities and rejection counts, but their proposed coordinates are not part
of the independent accepted-history audit.

## Validation scope

Five focused Rust tests check the summed Poisson accepted-flow identity for
nonuniform \(g\), zero-volume cases, endpoint/telescoping identities, cache
binding, attempt-journal behavior and strict option checks. They generate no
protein or SMC reference populations. The first compile error and corrected
test logs are preserved in
`results/hard-free-smc-initial-guide-validation-20261001`.

[`audit_hard_free_smc.py`](../tools/audit_hard_free_smc.py) independently
reconstructs the complete physical guide from geometry, checks every
initial attempt, deterministic cache, accepted/rejected history, incremental
weight, resampling parent and terminal normalizer, and authenticates the
supplied binary and its embedded source bundle. Analytic sphere-lens bounds
are checked when applicable. A campaign controller must additionally bind
the options to its predeclared allocation. Adversarial Python tests cover
truncated schedules, impossible generation branches, stale caches, an extra
Jacobian, altered linear normalizers and zero-hit accounting.

The separate reference allocation is frozen at
`results/hard-free-smc-reference-preparation-20261001`, with plan SHA-256
`2e63ba15b3b97ca42d9d1acd802a1341817d26f7b6625346aa5123bddad1b65d`.
All 165 archived input/source files match their hashes. The eleven jobs
declare 33,536 unconditional initial attempts: four populations at each of
z=0 and z=4, a forced zero-hit case, and a deliberately paired pure-uniform
control. The analytic references include total, contact and unbound masses
with the exact sphere lens and normalized Haar cap. Three preparation tests
and six auditor tests passed. The archived isolated binary is
`92553e2ed79801811008460c4e725cf73a6a5f21fb785c3e09c42312507103dd`.

Preparation starts no jobs. References must pass independent audits before
new protein SMC work. Existing protein campaigns and the production assembly
binary remain unchanged. Finite-system assembly and instability remain
unresolved.
