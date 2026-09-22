# Physical-pose latent guide API and gated vessel integration

The new [physical guide module](../src/latent_region/physical_guide.rs) exposes the existing checked `Chart` and Gaussian `ImportanceGuide` as a normalized proposal on world poses. It does not sample protein configurations, evaluate overlap, draw Poisson clouds, apply a wall, or change an existing sampler. Its only shared source change is the `pub mod physical_guide` export in [latent_region.rs](../src/latent_region.rs). Full-vessel physical execution remains gated by the regional confirmation and independent coverage results.

`PhysicalLatentGuide::from_files(region, guide, expected_shape_sha256)` binds exact region/guide bytes and the shape identity. It requires the full radius-4 source chart and the untruncated Gaussian guide schema, and retains all physical fixed neighbors and the source capture metadata. `validate_physical_context` checks shape and every fixed body. The source capture radius is metadata: the enclosing vessel capture must be established independently and may be larger.

The main methods are:

- `draw(&mut StdRng)` returns one unconditional world pose, original u and radius, selected Gaussian component or uniform branch, and its complete density. Exterior-R4 and exterior-source-capture outcomes are retained.
- `decode(u)` returns a world pose and the original chart log Jacobian for any representable finite u.
- `evaluate(world_pose)` returns u, the reporting-only R4 flag, latent log density, log Jacobian, and physical log density. `log_density(world_pose)` returns the latter alone.
- `half_mixture_log_density(log_vessel, log_latent_physical)` performs the outer 50/50 log-sum after the caller has evaluated both complete densities at the same world pose. A query outside both supports returns log density −∞; a generated-row weight caller must separately require positive density.

The density measure is center volume times **normalized** SO(3) Haar. For the original chart,

`log q_D^physical(x) = log q_D(u(x)) − log J_D(u(x))`.

The R4 indicator restricts only the uniform term in q_D. Every Gaussian contributes globally. For normalized chart-relative quaternion scalar w, the evaluator uses

`log J_D = log det L − 3 log ell − 2 log pi + 4 log |w|`.

This avoids squaring enormous Cayley coordinates near the seam. Only the exact `w == 0` seam receives density zero, an arbitrary convention on a measure-zero set. There is no epsilon exclusion band. Non-seam inverse or density calculations outside representable FP64 range return an error; they never silently turn a positive Gaussian component into a rejected pose. A pure-uniform guide is also supported and legitimately has zero density outside its ball.

Seven [math-only Rust tests](../src/latent_region/physical_guide_tests.rs) passed. They check q/J inside and outside R4 in two distinct full-covariance charts, physical normalization using independent radial quadrature, unconditional synthetic guide draws outside source capture, source identity checks, the outer mixture algebra, exact seam handling, finite near-seam evaluations through w=10⁻⁵⁰, and explicit failure for an unrepresentable non-seam Gaussian log density. These tests use no atoms, environment, overlap estimator or physical simulation:

```bash
cd /home/xvg/tetramer-mc
CARGO_BUILD_JOBS=1 cargo test --offline --lib physical_guide::tests -- --test-threads=1
```

The vessel loop is deliberately not integrated yet. Its coordinated patch should follow these steps:

1. Add a separate optional guide-files type and `run_with_wall_and_latent_guide(options, wall, guide_files)` wrapper in [normalizer.rs](../src/normalizer.rs). Thread an optional loaded guide through the internal implementation while preserving the existing `None` path, RNG order, schemas and public option literals. Add paired `--latent-region`/`--latent-guide` CLI options requiring an atomic wall. The source chart is a proposal and reporting coordinate system, not a new target restriction.
2. Keep the current vessel proposal intact. Its density is the **average over all selected fixed anchors** of the full model density, including the original cube/Haar uniform floor and every atlas or reciprocal virtual branch. Preserve selected-anchor mappings and the distinction between physical neighbors and proposal anchors. Active reciprocal models retain their exact branches and covariance scale 1; legacy covariance scaling multiplies standard deviations and therefore covariance by the square.
3. Choose the outer branch with probability 1/2 in a separately named deterministic stream. A vessel outcome is converted from capture-centered coordinates to a world pose. A latent-guide outcome is already a world pose. Neither branch retries numerical nulls or exterior outcomes. For either branch, evaluate

   `log q_mix(x) = logaddexp(log p_vessel(x), log q_D^physical(x)) − log 2`.

   Evaluate both densities before any target validity check, including for invalid poses and poses outside R4. `p_vessel` must use the world pose recentered by the vessel capture center and the complete original anchor average; a selected branch's logged density is insufficient. The pointwise bound is `q_mix >= 0.5 p_vessel`, so the existing global defensive support remains available even on the latent chart seam.
4. Retain `run_with_wall`'s full atomic-wall and hard predicates. Preserve the guarded capture enclosure `capture_radius >= |wall_center − capture_center| + wall_radius + shape.bound + guard` and validation that all fixed bodies fit. Wall feasibility checks every physical-core atom. R4, original metric q and native class remain reporting labels only. The depletion calculation still uses the union of all fixed-neighbor exclusion bodies with a wall-permeable ideal bath; the wall does not clip cloud support or change depletant radii. Valid rows use `log_hard_weight = −log q_mix`; all invalid attempts remain zeros in the original N.
5. Introduce an explicit new outer-mixture manifest/schema and archive the exact regional guide, chart, source hashes and old vessel model. Record outer branch, both component log densities, full mixture log density, and branch-specific generation metadata. Keep reciprocal/anchor outcome metadata on vessel rows and u/J/component metadata on latent rows. Preserve all attempted rows, including exterior-R4 contributions that are valid in the vessel.
6. Extend the independent Python auditor before physical execution. Separate the all-pose vessel-density evaluator from the vessel-generated-outcome checker; the latter applies only to vessel rows. Independently reconstruct q_D/J from world poses on every row, including invalid/exterior rows, then verify the complete outer mixture and hard/cloud arithmetic. Reuse the atomic-wall/enclosure audit after explicit new-schema dispatch. Existing schema-4 parsing and assumptions that every row has a vessel `proposal` object must be updated explicitly. Do not relax tolerances or use the reciprocal direct-factor auditor for the legacy covariance-refactor law.

The independent [Python physical-density helper](../tools/physical_latent_guide.py) now passes seven mathematical tests. It reconstructs eight Rust-generated world-pose records in two charts, including exterior and near-seam cases; the largest scaled coordinate/density error is 4.60e-16. The [validation archive](../runs/physical-latent-guide-math-validation-20260922/python-validation.json) retains tests, fixture bindings and the independent audit. These are synthetic coordinate/density checks, not an audit of an integrated full-vessel sampling run. Integration still needs the branch-aware full-vessel auditor and physical reference tests after worker capacity permits them. Protein execution remains gated on the regional checks. No full-vessel importance run, coverage conclusion or assembly inference has been produced by this API work.
