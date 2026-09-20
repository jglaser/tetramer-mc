# Active atlas mask pilot

This matched pilot changes the number of components evaluated in a proposal
while retaining the successful atlas and its reversible mean/covariance
transport. It is **active evaluation compression**, not compression of the
stored parameters or a procedure for discovering basins.

The same 28-component native-informed atlas, complete 783-dimensional Gaussian
auxiliary state, and all-component current-geometry fit remain in every arm.
The active subset changes after each sweep. The physical configuration and
Gaussian residual stay fixed during that refresh. A physical proposal retains
the same subset and residual in its forward and candidate-dependent reverse
density evaluation.

For a subset A with K labels, the auxiliary distribution is

\[
\rho(A)=
\frac{8^K/K!}{\sum_{k=K_{\min}}^{K_{\max}}8^k/k!}
\frac{\prod_{j\in A}w_j}{e_K(w)},
\]

where the weights are the fixed reference-atlas probabilities and e_K is their
elementary symmetric polynomial. Normalizing the subset law separately at each
K is essential; otherwise the label weights also change the intended count
law. Production mixture weights retain their relative values inside each
selected subset. An empty subset uses the defensive uniform proposal.
The [balance argument](atlas-mask-balance.md) explains why the physical marginal
is unchanged and no mask probability appears in physical-move acceptance.

| Arm | Count support | Initialization |
|---|---|---|
| Full | K=28 | All labels |
| Fixed16 | K=16 | Exact conditional subset draw |
| Fixed8 | K=8 | Exact conditional subset draw |
| Poisson8 | K=0,…,28, activity 8 | All labels initially, then exact Gibbs refreshes |

All four controls reuse the four dispersed starts and MC seeds from the
previous covariance-arm campaign. Each run contains 12 mobile rigid tetramers,
with no initial interbody exclusion contact, spherical radius
354.50820786337056 Å, depletant radius 1.5 Å, activity 0.035 Å⁻³, identical local
moves, spherical GCA, and center shifts. Mean and covariance gain/noise are
0.25/0.1; weight gain/noise are zero. The Gaussian residual refreshes after each
sweep, followed by the mask refresh. These finite assembly runs begin outside
equilibrium; the deliberately full initial Poisson8 mask is also outside its
auxiliary equilibrium law.

Run and audit the campaign with:

```bash
cargo build --locked --release --bin tetramer-mc
/home/xvg/protein-nucleation/.venv/bin/python tools/run_atlas_mask_campaign.py \
  --out runs/atlas-mask-400 --sweeps 400 --workers 4
/home/xvg/protein-nucleation/.venv/bin/python tools/analyze_atlas_mask_campaign.py \
  --campaign runs/atlas-mask-400 --workers 4
```

The launcher archives the executable, model, shapes, source configurations, and
hashes. The analyzer independently reconstructs every physical update and mask
refresh to the saved frames and final checkpoint. It also recomputes the full
deterministic fit, both transported densities, masked mixture normalization,
and each mask's auxiliary probability using NumPy. The full-mask arm must
exactly reproduce the prior covariance-arm physical trajectories.

Every hard-valid global candidate receives the established native-pose and
external residue-patch classification, including rejected candidates. Native
body-pair bond changes are separated from switches of native motif label on a
pair that remains bonded. Saved frames independently receive atom-union
hard-core and spherical-wall checks. These checks do not replace the separate
equilibrium and balance tests.

Additional diagnostics compare the masked and full-atlas reverse densities
using the same candidate fit. A reverse density is called atlas-supported when
its Gaussian fraction exceeds one half. Loss of support means that the full
atlas meets that condition while the mask leaves more than 99% of the density
in the uniform floor. These thresholds classify diagnostics only.

The joint availability of reference labels (14,19), (21,22), and (7,7) is
reported because an earlier post-hoc audit identified their reciprocal contact
coverage. They do not enter mask construction. Their exact inclusion
probabilities follow from the subset law; empirical occupancy excludes the
initial mask and uses all post-refresh masks.

## Results

All 16 runs completed 400 sweeps. Every move replay passed, and all 164 saved
full-mask configurations exactly reproduced the preceding covariance control.
The independent reconstruction checked 100,258 density values or ratios,
7,056 fit records, and 6,400 mask probabilities. Maximum absolute errors were
2.84×10⁻⁷ in log density, 3.78×10⁻¹⁵ in fit coordinates, and 2.49×10⁻¹⁴ in
mask log probability. The density audit tolerance is 2×10⁻⁶.

| Arm | Mean active K | Native Gaussian candidates accepted/proposed | Native bonds formed/broken | Total sampler CPU s |
|---|---:|---:|---:|---:|
| Full | 28.000 | 41/428 | 37/0 | 136.47 |
| Fixed16 | 16.000 | 38/358 | 29/0 | 130.48 |
| Fixed8 | 8.000 | 33/253 | 29/0 | 126.29 |
| Poisson8 | 7.925 | 32/232 | 27/0 | 122.25 |

Native accessibility survives masking, but this pilot shows fewer native
events for the saved runtime. Relative to the full control, fixed8 forms 22%
fewer native bonds with 7% less sampler CPU time; Poisson8 forms 27% fewer with
10% less CPU time. These are descriptive totals from four paired assembly
trajectories, not estimates of a mixing-time ratio. No arm exhibits a native
body-pair detachment. The isolated motif breakages are registration changes on
pairs that remain bonded.

The masks sometimes remove useful reverse coverage:

| Arm | Proposals with full-atlas reverse support | Such proposals reduced to the uniform floor by masking | Native candidates among those losses |
|---|---:|---:|---:|
| Full | 566 | 0 | 0 |
| Fixed16 | 504 | 103 | 12 |
| Fixed8 | 453 | 250 | 34 |
| Poisson8 | 430 | 227 | 22 |

All these comparisons use the same candidate-dependent parameter fit, so they
isolate the density support removed by a particular mask. They do not establish
how the complete trajectory would change under a different mask policy.

The observed joint availability of labels (14,19) is 1.000, 0.664, 0.234, and
0.216 across the four arms. For labels (21,22) it is 1.000, 0.842, 0.469, and
0.436. The [full report](../runs/atlas-mask-400/assessment/report.md) compares these
observations with their exact auxiliary-prior probabilities. Independent
label compression can therefore separate components providing reciprocal
coverage even when individual frequently used labels are favored.

The direct proposal cost is small: total proposal CPU falls from 0.327 s for
the full atlas to 0.286 s for Poisson8, while all-component fit/reverse-model
cost stays around 3.7–3.9 s. Most of the change in total CPU comes from the
depletion gates (84.5 versus 70.2 s), along trajectories with different contact
configurations. The wall-clock difference consequently should not be described
as the speedup of masked Gaussian evaluation alone.

The [machine-readable audit](../runs/atlas-mask-400/assessment/analysis.json)
contains every native proposal and transition, original selected component
labels, refreshed masks, inclusion statistics, independent-density checks,
and source hashes.

![Matched active-mask comparison](../runs/atlas-mask-400/assessment/atlas-mask-comparison.png)

The baseline is exact and retains some useful native proposals, but the present
independent subset law offers no demonstrated sampling advantage. It tests
retention of accessibility with supplied native information. Four reused starts
and 400 sweeps cannot establish equilibrium, native discovery, or crystal
growth.
