# Physical overlap scores for saved core and tail poses

The [fixed-mixture reduction](context-covariance-balance.md) found that six
far-tail poses carry 72.4% of the estimated hard-only A_T volume. This does not
establish their physical importance: the depletion factor can change those
relative weights. The next diagnostic scores a small fixed panel before
allocating clouds to the full bank or broadening the proposal.

This remains the historical fixed environment at 500 μM, with one mobile
tetramer and 263 fixed bodies. Depletants have radius 1.5 Å and activity
0.035 Å⁻³. A_T is the complete source contact pattern with exclusion neighbors
{16,217}; it is not a native-registry classifier. The finite-system assembly
decision at approximately 106.8 μM remains separate.

## Frozen panel and physical measure

The full-covariance chart divides the saved A_T poses at squared Mahalanobis
radii 0, 6, 12, 24, 48 and infinity. These strata contain 1,118, 458, 99, 6 and
0 poses. Six poses per occupied stratum are selected by a predeclared SHA256
ranking of arm, stream and ordinal. Thus all six far-tail poses are included.
The panel records every selected saved pose, its generating row and density,
the complete ranking proof, and the input hashes. Hash priorities are
deterministic once the salt is fixed; the recorded `min(6,M)/M` probabilities
are nominal under a uniform-hash model. They are not used to infer masses.

The scorer verifies each exact saved pose against its authenticated original
row, recomputes atomic-wall and hard-core validity, and checks the full patch
classification. It reconstructs the same overlap envelope and verifies its
volume bounds, cell counts and fixed-body labels. The fixed physical geometry
and measure are unchanged. No new poses are generated.

For moving exclusion region A and the union B of fixed exclusions, let
O = |A ∩ B|. The existing many-body kernel partitions the overlap into a
certified lower volume L and an uncertain envelope of volume U. Two independent
Poisson clouds at intensity λ = 2.24 Å⁻³ yield

\[
K_0+K_1\sim\operatorname{Poisson}(2\lambda(O-L)),\qquad
\widehat S=zL+\frac{z(K_0+K_1)}{2\lambda}.
\]

Then E[Ŝ] = zO, and

\[
\operatorname{Var}(\widehat S)=\frac{z^2(O-L)}{2\lambda}
\leq\frac{z^2U}{2\lambda}.
\]

This estimates the overlap benefit in kBT. The independent Python analysis
uses exact Poisson count intervals, including nonzero upper uncertainty when
no hits occur. Bonferroni intervals cover all 24 fixed-pose scores
simultaneously at at least 95% under the uncapped Poisson law. They describe
point-cloud noise only. They do not include uncertainty from missed poses.

The positive Boltzmann estimator is a distinct quantity:

\[
W_i=e^{zL}(1+z/\lambda)^{K_i},\qquad E[W_i]=e^{zO}.
\]

Neither log((W₀+W₁)/2) nor exp(Ŝ) can be substituted for the corresponding
unbiased estimator. The scorer records both quantities separately. Comparing
Ŝ − log q̄ shows whether depletion suppresses the large inverse-density
contributions seen in the geometric reduction. It does not integrate the
region: this panel is deliberately stratified and small.

## Execution and validation

The allocation is exactly 24 poses and 48 independent clouds, with about
14.025 million expected raw points. Caps are 500,000 raw/processed points per
cloud, one million per pose and 24 million in total, with 600 CPU seconds,
1,200 wall seconds, 4 GiB and one thread. A whole Poisson count is admitted
before point work. Every begun pose, cloud, sampled count and processed prefix
is journaled. The first failure invalidates panel completion; there are no
retries, clipped counts, skipped poses or replacements. Finite-cap success is
not claimed to preserve unconditional estimator identities by itself.

The standalone Rust example is `fixed_saved_pose_overlap`; production kernels,
assembly runners and production binaries are unchanged. Tests cover score
versus positive-weight identities, deterministic zero-volume clouds, nonzero
uncertainty after zero hits, source/provenance mismatches, independent role
seeds, and whole-count failure accounting. The Python reconstruction adds
exact count intervals and checks all completed cloud and score arithmetic.
Previously validated overlap geometry and sphere-reference controls are reused.

Selection evidence is in
`results/context-overlap-panel-selection-20261005`; its twelve initial tests
passed. The frozen panel SHA256 is
`b912baacd2326d89ad236dfabdd777c43a001b01c3d28b1457a750e4d012d680`.
The isolated build and subsequent physical allocation use separate directories
`results/context-overlap-panel-build-20261005-v2` and
`results/context-overlap-panel-physical-20261005`. The original selection-stage
code snapshots remain preserved; the subsequent count audit adds explicit
handling and tests for the deterministic empty-envelope API representation.
The first build stopped before tests on an ambiguous float literal in a
synthetic test. Its drained failure record remains in the original build
directory. A test-only type annotation fixed compilation; the second build
passed all eight Rust controls and preserved the old cache and production
binary. The embedded source bundle covers the library; the example and its
helpers are additionally pinned in the immutable build snapshot.

Floating-point geometry, exact thinning, pose/environment binding and RNG
execution remain implementation obligations. This diagnostic is not a proof
of coverage, an equilibrium contact-weight estimate, or a test of finite-system
crystal stability.

## Completed result

All 24 poses and 48 clouds completed: 14,023,059 raw points, all processed,
336,251 overlap hits, and 5.384 CPU seconds in the physical scorer. The eight
subsequent Python score/accounting tests and the independent audit passed.
All three execution stages drained without retries or replacements.

The two largest hard-volume tail contributions survive physical scoring.
Among the six tested core poses, the largest corrected score Ŝ − log q̄ is
86.994 and the largest simultaneous upper bound is 89.888.

| Saved tail | Full-chart r² | Estimated zO | Estimated zO − log q̄ | Simultaneous lower bound minus the largest tested-core upper bound |
|---|---:|---:|---:|---:|
| diagonal stream 1, ordinal 1029 | 45.318 | 108.242 | 93.309 | +0.613 |
| diagonal stream 3, ordinal 1784 | 31.754 | 110.305 | 94.326 | +1.603 |

The observed effect comes from inadequate proposal density at those poses,
not higher physical density. Compared with the core pose having the largest
corrected score, their depletion benefits are lower by 2.547 and 0.484 kBT,
while their log proposal densities are lower by 8.862 and 7.817. Depletion
does not compensate for this proposal-density deficit. The simultaneous
interval comparison concerns these two tail poses and the six tested core
poses only; it does not rank every core pose or integrate any stratum.

The [plot](../results/context-overlap-panel-physical-20261005/figures-readable/core-tail-overlap.png)
separates the physical overlap benefit from the correction for proposal
density. The original plotted data and figure remain in the frozen execution;
`figures-readable` changes only spacing and label size.

Evidence hashes:

- Physical summary: `e1af675b16c2586dd4967d2f51bbe503a983ba99702785de567b608916c44ca6`.
- Independent score audit: `77ea6140c224988c79adb652a1247d8f40f704cfba34fec95b935cf2b9dbeef6`.
- Passed execution status: `3c38e69413c4d3244bce8affc3456c845b5d0d347c65e97cb7912a72ef508e7e`.

This supports a prospective broader proposal control, not an assembly
conclusion. A minimal next source mixture retains the full and diagonal charts
and adds a chart with the same mean and covariance 4C_full:

\[
q_{new}=\tfrac12U+\tfrac14G_C+
\tfrac14(0.375g_F+0.375g_D+0.25g_{4C_F}).
\]

It obeys q_new ≥ 0.75 q̄ pointwise, so existing probability support is retained
and no existing importance weight can increase by more than 4/3 solely from
the proposal change. The broad component's expected squared radius in the
full chart is 24. Fresh independent populations must assess physical weight
concentration, population agreement and CPU cost; the present panel motivated
this choice and cannot serve as its validation.

The broader guide and allocation are now frozen in
`results/context-broadened-frozen-guides-20261005/result/mixture-manifest.json`
(SHA256 `d74a4d564fd71547a52292c5d7a9a84ab14cf967e8ce0072389f3958b7bee827`).
Five synthetic controls passed. This preparation generates no poses, clouds
or fitted parameters and does not modify an assembly kernel.

The prospective comparison has four fresh 4,096-attempt populations per arm,
32,768 attempts total. Each baseline population combines 2,048 draws from the
existing F runner law and 2,048 from D. Each broadened population combines
1,536 F, 1,536 D and 1,024 B. Each runner law already includes its uniform and
context branches; these counts are not promised source-branch hits. Reweight
every draw by its arm's full deterministic-mixture density and divide by all
4,096 attempts. No old draws are pooled into the new comparison. The unchanged
runner generated these strata. The [broader-guide comparison](context-broadened-guide.md)
implements the complete-mixture reduction, independent density and geometry
audit, frozen seeds and limits, and two-cloud scoring at every hard-valid draw.

The completed physical panel's final host audit verified every owned process
group had drained and the production binary and three pre-existing simulation
workers were unchanged:
`results/context-overlap-panel-physical-20261005/host-drained.json`, SHA256
`b91fc00b1d50a77e1f6d0c79eb445e3be1bd64466fc2100a2333864529ddc420`.
