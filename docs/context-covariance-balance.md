# Equal-mixture reduction of the frozen covariance controls

This calculation reuses all 16,384 saved poses from the full-covariance and diagonal source-guide controls. It generates no poses, performs no atom queries, and samples no depletants. The target is the **hard-only volume of the defined A_T contact region**, with translation volume times normalized Haar orientation measure. A_T means exact exclusion-neighbor labels `{16,217}` and complete inclusion of the sixteen fixed source217 patch tokens. It is not a native-registry or equilibrium-energy classifier.

Each arm was frozen before its draws:

\[
q_F=\tfrac12U+\tfrac14G_C+\tfrac14g_F,\qquad
q_D=\tfrac12U+\tfrac14G_C+\tfrac14g_D.
\]

Pairing the two arms' stream \(i\), each with \(N=2048\) attempts, gives the fixed deterministic-mixture estimator

\[
\bar q=\tfrac12(q_F+q_D)
 =\tfrac12U+\tfrac14G_C+\tfrac18g_F+\tfrac18g_D,
\qquad
\widehat V_{A_T}=\frac1{2N}\sum_{a\in\{F,D\}}\sum_{j=1}^N
 \frac{H_{A_T}(X_{aj})}{\bar q(X_{aj})}.
\]

Here \(H_{A_T}\) includes the atomic-wall and hard-core indicators. Invalid draws contribute zero and remain in the denominator. The balance-heuristic expectation identity is

\[
\mathbb E\widehat V_{A_T}
=\tfrac12\int q_F\frac{H_{A_T}}{\bar q}\,d\mu
 +\tfrac12\int q_D\frac{H_{A_T}}{\bar q}\,d\mu
=\int H_{A_T}\,d\mu.
\]

The shared uniform component covers the physical vessel. The argument is conditional on the already frozen, externally trained charts; there is no fitting to these poses or selection of draws by outcome. Earlier isotropic-guide draws are excluded. This is an importance identity, not a new Markov-chain balance argument.

The reducer independently reconstructs both source-Gaussian densities and their Cayley/Haar Jacobian for every saved pose. It authenticates the previously audited atlas/uniform densities, poses, hard verdicts and regions through the completed independent audit's hashes. Each original source density and full proposal density is checked again against the reconstruction. Six synthetic controls cover normalization and the expectation identity, hard-invalid zeros and unconditional denominators, noncommuting anisotropic charts, radial partition closure, population-based uncertainty, and deliberate density mismatches.

Four paired-stream estimates provide the reported standard error: sample standard deviation divided by \(\sqrt4\). Importance ESS \((\sum w)^2/\sum w^2\) and the largest normalized contribution describe weight concentration. They are not effective independent contact samples. A paired population contains twice the draws of an individual arm, so an uncertainty reduction alone does not establish a sampling speedup.

For A_T hits, squared Mahalanobis radii under **each** frozen source chart use the predeclared bins `[0,6)`, `[6,12)`, `[12,24)`, `[24,48)`, and `[48,infinity)`. Each chart separately partitions the entire region weight, including subdivisions by generating arm and proposal branch. The two chart families must not be summed together. All original region classes and hard-invalid attempts are retained.

The frozen calculation is in `results/context-covariance-balance-20261005`, with a single 30 CPU second / 60 wall second / 2 GiB / one-thread allocation for six synthetic tests followed by the saved-data reduction. Results remain hard-only conditional evidence at the fixed historical 500 μM context and the specified 1.5 Å / 0.035 Å⁻³ model settings. No depletion-weight or finite-system assembly conclusion follows from this reduction.

Poor hard-only importance ESS does **not** imply poor physical-weight ESS. The source guide was fitted to physical local-trajectory residence, so geometric-volume tails may have negligible weight after multiplication by the depletion factor `exp(z O)`. The saved radial counts can support a separately frozen, small core-versus-tail overlap comparison before deciding whether broader guidance or a full physical-weight campaign is warranted. No such overlap comparison is part of this reduction, and broadening is not selected from its hard-volume diagnostics alone.

## Completed saved-data result

All six synthetic tests and the one fixed reduction passed; the child process drained. All 16,384 attempts and 1,681 A_T hits were retained, with zero Cayley seams and no geometry/cloud calls. Both independently reconstructed source laws agreed with the saved generating densities under the frozen tolerance.

| Estimator | Attempts per population | Mean hard A_T volume (Å³) | Population RSE | Per-population importance ESS | Largest contribution per population |
|---|---:|---:|---:|---|---|
| Full arm | 2,048 | 1.0212e−11 | 6.20% | 5.94, 4.35, 11.80, 13.57 | 32.37%, 45.99%, 18.79%, 14.75% |
| Diagonal arm | 2,048 | 3.6843e−11 | 52.32% | 8.39, 1.29, 28.67, 1.79 | 30.39%, 87.89%, 11.57%, 74.31% |
| Equal deterministic mixture | 4,096 | 3.8212e−11 | 51.94% | 15.61, 1.38, 36.88, 1.92 | 20.49%, 84.93%, 9.99%, 71.83% |

The equal mixture retains severe hard-volume weight concentration. Across all four populations its importance ESS is 3.24, and one contribution accounts for 52.23% of the total. The full arm's smaller between-population error estimate does not certify that rare high-weight geometric tails were covered.

The predeclared radial partitions expose the concentration directly. Counts below are A_T hits, split by the **generating arm**; the weight share uses the combined estimator. All A_T hits came from the source-Gaussian branch, with zero from the uniform or context-atlas branches.

| Radius chart | Squared radius interval | Full-arm hits | Diagonal-arm hits | Combined region-weight share |
|---|---|---:|---:|---:|
| Full | [0,6) | 812 | 306 | 2.16% |
| Full | [6,12) | 258 | 200 | 7.34% |
| Full | [12,24) | 23 | 76 | 18.13% |
| Full | [24,48) | 0 | 6 | 72.37% |
| Full | [48,infinity) | 0 | 0 | 0% |
| Diagonal | [0,6) | 839 | 456 | 3.76% |
| Diagonal | [6,12) | 221 | 122 | 11.87% |
| Diagonal | [12,24) | 33 | 10 | 84.37% |
| Diagonal | [24,48) | 0 | 0 | 0% |
| Diagonal | [48,infinity) | 0 | 0 | 0% |

The six poses in the full-chart `[24,48)` stratum are therefore a small, explicit saved set for a separately designed core-versus-tail physical-overlap control. Their importance for **hard volume** does not establish their importance after depletion weighting. No such control or broader proposal has been executed or selected here.

Evidence: `results/context-covariance-balance-20261005/result/report.json`, SHA256 `10d587d440368fcdae24425250846d8546abecd1e2d2f37bef2d79fc0374cf2b`; complete per-attempt density/weight records in `result/contributions.jsonl`, SHA256 `93d9a35b7f2fbdf17cb3390c351078afdbf4cde6e64e1df58569190e5a990092`. The passed execution status is SHA256 `37bb196118c42f679b7cecd4fa802d6b49437760e7e930dadf446c5cb7cdcac2`. The reduction took 1.00 CPU second after input authentication; tests took 0.006 CPU seconds.
