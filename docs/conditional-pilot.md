# Conditional closure: correctness and short free-start pilot

The conditional implementation passes the exact-start stationarity check and
the deterministic count-correction negative control. The short assembly pilot
exercises variable counts and accepted learned proposals, but shows no evidence
of improved sampling or registered native assembly.

## Independent correctness check

`tests/conditional_stationarity.rs` uses 6,000 independent exact equilibrium
starts for two hard spheres in a spherical wall, with the analytic AO pair
weight at positive depletant activity. Each replicate applies three rounds of
conditional Gibbs refresh, learned and LOCAL proposals with the exact Poisson
gate, count-corrected GCA, and common center shifts. Fourteen predeclared
observables include radial and Haar moments, the residual
`K - E[K | D(X)]`, and Gaussian residual moments. Paired errors use independent
replicates, with no burn-in or trajectory independence assumption.

For the final implementation including the full-support uniform cube, the
largest absolute paired z score was **1.303**, below the predeclared 5.5 bound.
There were 11,925 count changes, 1,286 accepted global moves, 9,133 accepted LOCAL
moves, 1,067,315 Poisson gate points, 17,008 GCA probes, and 417 GCA rejections
from the conditional count correction. The maximum LOCAL count log correction
was 1.1223; reconstructing the reverse model changed the log proposal ratio by
up to 4.2900. Counts 0–3 were all exercised.

A separate exact finite-state transition matrix uses the actual fitted count
probabilities. Its corrected stationary residual is **2.776 × 10⁻¹⁷**. Omitting
the conditional count correction produces a physical marginal drift of
**0.00294033 in one step**. This negative control has no sampling-power
assumption. These checks supplement the core parameter-transport and runner
restart tests; they are not mixing or ergodicity proofs.

## Matched pilot

Four independent preparations of 12 dispersed tetramers were reused exactly
across four arms, with paired master streams and 200 sweeps per arm. Initial
orientations are Haar; every start has a certified separation exceeding twice
the exclusion bound. All bodies move. The spherical wall radius is 354.5082 Å,
`rd=1.5 Å`, and `z=0.035 Å⁻³`; GCA and center shifts follow every sweep.

The conditional arms use no supplied model or native docking data. They use the
default deterministic closure: `Kmax=4`, basin activity 2, score temperature 1,
penalty strength 1, three fit iterations, covariance bounds `[0.04,4]`, residual
scale 0.05, uniform defense 0.1, and an exact conditional refresh after every
sweep. Only the covariance exponent changes between `s=0` and `s=6`. Both use a
0.5 global-attempt probability. The LOCAL control uses probability 0; the uniform
control uses 0.5. Native templates enter only the independent post-hoc classifier.

| Arm | LOCAL accepted/attempted | Global accepted/attempted | Accepted learned / uniform branches | Total sampler CPU s | Final nonspecific bonds by start |
|---|---:|---:|---:|---:|---|
| Conditional s=0 | 4390/4847 | 91/4753 | 10 / 81 | 14.62 | 1, 1, 2, 1 |
| Conditional s=6 | 4566/4847 | 129/4753 | 8 / 121 | 11.77 | 1, 0, 1, 1 |
| LOCAL | 8695/9600 | 0/0 | — | 13.02 | 0, 1, 3, 0 |
| Uniform | 4474/4847 | 918/4753 | 0 / 918 | 10.93 | 3, 0, 0, 0 |

All **336 saved frames** pass independent atom-union overlap and atomic-wall
checks. There are **zero registered native bonds in every saved frame**.
Nonspecific bonds mean an interbody atomic gap at most `2rd`; they do not imply
native registration. These are observations every ten sweeps, not a replay of
transient formation or residence between saved frames.

The 800 Gibbs count draws per conditional arm have histograms for `K=0,…,4`:

| Arm | Count histogram | Mean K | GCA count rejections/attempts | Conditional fit CPU s |
|---|---|---:|---:|---:|
| s=0 | 20, 23, 169, 316, 272 | 2.996 | 155/800 | 0.521 |
| s=6 | 51, 66, 235, 254, 194 | 2.593 | 169/800 | 0.463 |

The count law remains variable. These correlated occupancies do not validate
the physical target. Most accepted global moves in the conditional arms came
from the uniform defense; only 18 learned-branch moves were accepted across
both arms. Four short preparations and acceptance counts cannot establish an
efficiency gain, equilibrium, crystal growth, or a ranking of the algorithms.
The table is a baseline for improving the instantaneous fit and proposal scale.

## Artifacts and reproduction

The full local campaign is in
`results/conditional-pilot-20260919/`, including archived executable and inputs,
per-run configurations, move logs, trajectories, source bundles, and independent
analysis at `assessment/analysis.json`. The final stationarity output is
`stationarity.txt`. The 16 runs used 50.34 total sampler CPU seconds and 15.26
seconds of supervisor wall time with four workers. GSD output was disabled.

Executable SHA-256:
`1c209180fea1e12299c1dd0c1a75e8c8ec1ac975e2ada87d33ff92b9e4417ff2`.
Embedded source-bundle SHA-256:
`56ee995416ffe7c0e07d1cb2398839293387ced6f42269770a36df8e10c3576c`.

From the repository root, using a Python environment with NumPy and SciPy:

```sh
cargo test --release --test conditional_stationarity -- --nocapture
cargo build --release --bin tetramer-mc
python tools/run_conditional_campaign.py --out results/conditional-pilot-repeat --sweeps 200 --replicates 4 --workers 4
python tools/analyze_conditional_campaign.py --campaign results/conditional-pilot-repeat --workers 4
```

The classifier additionally uses the existing `protein-nucleation` reference
checkout, configurable through the analyzer's `--reference` option. Classifier
source hashes, input hashes, per-refresh count probabilities, and detailed
acceptance/cost records are retained in the machine-readable assessment.
