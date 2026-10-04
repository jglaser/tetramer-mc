//! Oligomer destination charts for rigid-subset transport.
//!
//! The member/anchor mixture places one carried member in one atlas basin
//! against one pool anchor. An embedded subset usually has several external
//! contacts, and a one-contact destination gives most of them up. This module
//! adds deterministic two-contact components: pairs of single member/anchor
//! labels whose implied subset poses agree are fused by a Gauss-Newton fit of
//! both latent residuals, and the fused Gaussian (inverse summed information)
//! becomes one more chart of the pose of the first listed member, g0.
//!
//! Construction reads only invariant context: the internal offsets
//! u_i = g0^{-1} g_i, the fixed pool anchors, the fixed spectators and the
//! wall. It never reads g0 itself. The offsets recomputed from carried
//! floating-point poses differ in the last bits, and any cutoff (component
//! cap, check budget, mismatch threshold) can turn that into a different
//! catalogue. So construction uses offsets rounded to a fixed grid, and the
//! caller rejects any trial whose rounded offsets change (`internal_key`).
//! That condition is symmetric in the two endpoints, so it is a valid guard,
//! and when it holds every construction input is bitwise identical: the
//! same catalogue is rebuilt at both endpoints, near-tied fits included. The
//! posterior-source map and its correction log G_O(old) - log G_O(new) then
//! follow the member-chart argument exactly.
use crate::{
    basin_involution::{BasinPair, BasinStep, FixedBasinInvolution, cross_chart_step},
    docking::{DockingProposal, draw_log_category},
    geometry::{Placed, SphereTree},
    math::*,
    proposal::GaussianComponentParameters,
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Vec6 = [f64; 6];

/// Rounding grids of the catalogue's internal offsets: translation in
/// angstrom, and components of the sign-canonical unit quaternion. Offsets
/// are recomputed with errors near 1e-13 angstrom, so a rounding boundary is
/// crossed with probability of order 1e-7 per coordinate and move.
pub const OFFSET_POSITION_GRID: f64 = 1e-5;
pub const OFFSET_QUATERNION_GRID: f64 = 1e-9;

/// Integer key of the rounded internal offsets g0^{-1} g_i, first member as
/// g0, and the offset poses rebuilt from it. The catalogue is a function of
/// the key, the pool, the spectators and the wall.
pub fn internal_key(members: &[Pose]) -> (Vec<i64>, Vec<Pose>) {
    let g0_inverse = invert_relative_pose(members[0]);
    let mut key = Vec::with_capacity(7 * members.len());
    let mut offsets = Vec::with_capacity(members.len());
    for &g in members {
        let u = compose(g0_inverse, g);
        let mut q = u.orientation;
        if q.iter().find(|x| **x != 0.).is_some_and(|x| *x < 0.) {
            q = q.map(|x| -x);
        }
        let kt = u
            .position
            .map(|x| (x / OFFSET_POSITION_GRID).round() as i64);
        let kq = q.map(|x| (x / OFFSET_QUATERNION_GRID).round() as i64);
        key.extend(kt);
        key.extend(kq);
        let q = kq.map(|k| k as f64 * OFFSET_QUATERNION_GRID);
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        offsets.push(Pose {
            position: kt.map(|k| k as f64 * OFFSET_POSITION_GRID),
            orientation: q.map(|x| x / n),
        });
    }
    (key, offsets)
}
type Mat6 = [[f64; 6]; 6];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OligomerConfig {
    /// Proposal mass of the fused two-contact components when any exist.
    pub multi_contact_mass: f64,
    /// Largest summed squared whitened residual of both fused labels.
    pub max_mismatch: f64,
    /// Prefilter on the implied subset positions of two labels.
    #[serde(rename = "pair_distance_A")]
    pub pair_distance_a: f64,
    /// Prefilter on the implied subset orientations of two labels.
    pub pair_angle_degrees: f64,
    /// Gauss-Newton fits per attempt, in prefilter order.
    pub max_candidates: usize,
    /// Hard-core checks of fused centers per attempt, in mismatch order.
    pub max_hard_checks: usize,
    /// Fused components retained per attempt.
    pub max_components: usize,
}
impl Default for OligomerConfig {
    fn default() -> Self {
        Self {
            multi_contact_mass: 0.8,
            max_mismatch: 12.,
            pair_distance_a: 8.,
            pair_angle_degrees: 60.,
            max_candidates: 4096,
            max_hard_checks: 256,
            max_components: 32,
        }
    }
}
impl OligomerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.multi_contact_mass.is_finite() && (0. ..1.).contains(&self.multi_contact_mass),
            "oligomer multi-contact mass must lie in [0,1)"
        );
        for v in [
            self.max_mismatch,
            self.pair_distance_a,
            self.pair_angle_degrees,
        ] {
            ensure!(v.is_finite() && v > 0., "invalid oligomer fusion threshold");
        }
        ensure!(
            self.max_candidates > 0 && self.max_hard_checks > 0 && self.max_components > 0,
            "oligomer component limits must be positive"
        );
        Ok(())
    }
}

/// Single labels are (member * anchors + anchor) * branches + branch; fused
/// labels follow them in construction order.
#[derive(Clone, Debug, Serialize)]
pub struct FusedComponent {
    pub first: usize,
    pub second: usize,
    pub mismatch: f64,
    pub center: Pose,
}

/// Diagnostic comparison only. Production entry points retain EarlyAbort.
/// FullIterations removes only the existing large-residual early exit; all
/// iteration, line-search, screening and component limits remain identical.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionFitPolicy {
    #[default]
    EarlyAbort,
    FullIterations,
}

/// JSON-safe observation of existing arithmetic, without converting infinities
/// or NaNs into nulls or changing how the fitter handles them.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FusionFitValue {
    Finite(f64),
    PositiveInfinity,
    NegativeInfinity,
    Nan,
}
impl From<f64> for FusionFitValue {
    fn from(value: f64) -> Self {
        if value.is_finite() {
            Self::Finite(value)
        } else if value.is_nan() {
            Self::Nan
        } else if value.is_sign_positive() {
            Self::PositiveInfinity
        } else {
            Self::NegativeInfinity
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionFitTermination {
    Begun,
    InitialEncodeFailure,
    JacobianEncodeFailure,
    FinalJacobianEncodeFailure,
    NormalCholeskyFailure,
    EarlyResidualCutoff,
    LineSearchStalled,
    TinyStep,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionFitDisposition {
    Begun,
    ReturnedNone,
    AboveThreshold,
    Usable,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionFitStage {
    Initial,
    Jacobian,
    LineSearch,
    FinalJacobian,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionFitEncodeFailure {
    pub stage: FusionFitStage,
    /// Zero-based optimizer iteration; absent outside the iteration loop.
    pub iteration: Option<usize>,
    /// Finite-difference coordinate and sign, if applicable.
    pub coordinate: Option<usize>,
    pub direction: Option<i8>,
    /// Zero-based attempted line-search trial, if applicable.
    pub trial: Option<usize>,
    pub label: usize,
    pub error: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionFitDiagnostic {
    pub first: usize,
    pub second: usize,
    pub start_chi2: Option<FusionFitValue>,
    pub final_chi2: Option<FusionFitValue>,
    /// Iterations whose Jacobian was begun; the early cutoff is checked first.
    pub iterations: usize,
    pub accepted_steps: usize,
    /// Actual halvings, including the last unsuccessful line-search trial.
    pub backtracks: usize,
    pub line_search_trials: usize,
    pub residual_evaluations: usize,
    pub jacobian_evaluations: usize,
    pub encode_failures: usize,
    /// Includes recoverable trial failures, without storing an unbounded log.
    pub last_encode_failure: Option<FusionFitEncodeFailure>,
    pub nonfinite_objectives: usize,
    pub nonfinite_jacobians: usize,
    pub nonfinite_gradients: usize,
    pub nonfinite_steps: usize,
    /// Normal loop stop, retained even if the final Jacobian then fails.
    pub iteration_stop: Option<FusionFitTermination>,
    pub termination: FusionFitTermination,
    pub disposition: FusionFitDisposition,
}
impl FusionFitDiagnostic {
    fn new(first: usize, second: usize) -> Self {
        Self {
            first,
            second,
            start_chi2: None,
            final_chi2: None,
            iterations: 0,
            accepted_steps: 0,
            backtracks: 0,
            line_search_trials: 0,
            residual_evaluations: 0,
            jacobian_evaluations: 0,
            encode_failures: 0,
            last_encode_failure: None,
            nonfinite_objectives: 0,
            nonfinite_jacobians: 0,
            nonfinite_gradients: 0,
            nonfinite_steps: 0,
            iteration_stop: None,
            termination: FusionFitTermination::Begun,
            disposition: FusionFitDisposition::Begun,
        }
    }
}

/// Passive accounting of the existing catalogue construction, not a search
/// for additional components. No random numbers or extra geometry are used.
/// A caller-owned sink preserves this prefix when construction returns an error.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionBuildDiagnostics {
    pub complete: bool,
    pub fit_policy: FusionFitPolicy,
    pub single_labels: usize,
    pub decoded_means: usize,
    pub mean_decode_failures: usize,
    /// All unordered label pairs on different (member, anchor) interfaces.
    pub possible_interface_pairs: usize,
    /// Such pairs with two successfully decoded means.
    pub decoded_interface_pairs: usize,
    /// Includes pairs skipped by the sorted x-coordinate distance bound.
    pub distance_rejected_pairs: usize,
    /// Evaluated only after the full distance test passes, as in `build`.
    pub angle_rejected_pairs: usize,
    pub candidate_pairs_before_cap: usize,
    pub candidates_truncated: usize,
    pub fit_attempts: usize,
    /// One record per attempted fit, never more than max_candidates. Recorded
    /// only by build_with_fit_diagnostics; the existing APIs remain passive
    /// aggregate-only diagnostics with their original construction policy.
    pub fit_records: Vec<FusionFitDiagnostic>,
    /// `fuse` returned None: includes its early residual cutoff, chart encode
    /// failures and linear-algebra failures. Not a numerical-failure count.
    pub fit_returned_none: usize,
    pub fits_above_threshold: usize,
    pub usable_fits: usize,
    pub centers_checked: usize,
    /// Center-level counts are exclusive first failures in traversal order.
    pub invalid_pose_rejections: usize,
    pub wall_rejections: usize,
    pub core_rejections: usize,
    pub core_overlap_calls: usize,
    pub information_cholesky_failures: usize,
    pub covariance_cholesky_failures: usize,
    pub retained_components: usize,
    pub requested_fused_mass: f64,
    pub effective_fused_mass: f64,
    pub unvisited_fits: usize,
    pub component_cap_reached: bool,
    pub hard_check_cap_reached: bool,
    /// None means all usable fits were visited (including the empty set).
    pub stopped_by: Option<FusionBuildCap>,
    /// One record per visited center, never more than max_hard_checks.
    pub records: Vec<FusionCenterDiagnostic>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionBuildCap {
    Components,
    HardChecks,
    ComponentsAndHardChecks,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionCenterOutcome {
    Begun,
    InvalidPose,
    OutsideWall,
    CoreOverlap,
    InformationNotPositiveDefinite,
    CovarianceNotPositiveDefinite,
    Retained,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionCenterDiagnostic {
    pub first: usize,
    pub second: usize,
    pub mismatch: f64,
    pub center: Pose,
    pub outcome: FusionCenterOutcome,
    /// Offset/member index of the first geometric failure, if any.
    pub first_blocking_member: Option<usize>,
    /// Index in the input spectator slice, not a global particle label.
    pub first_blocking_spectator: Option<usize>,
    pub core_overlap_calls: usize,
}

fn cross_interface_pairs(counts: impl IntoIterator<Item = usize>) -> Result<usize> {
    let (mut previous, mut pairs) = (0usize, 0usize);
    for count in counts {
        pairs = previous
            .checked_mul(count)
            .and_then(|n| pairs.checked_add(n))
            .context("Fusion diagnostic pair count overflow")?;
        previous = previous
            .checked_add(count)
            .context("Fusion diagnostic label count overflow")?;
    }
    Ok(pairs)
}

#[derive(Clone, Debug, Serialize)]
pub struct OligomerStep {
    pub handle: Pose,
    pub source: usize,
    pub target: usize,
    pub step: BasinStep,
    pub identity: bool,
    pub full_old_log_density: f64,
    pub full_new_log_density: f64,
    pub label_log_reverse_forward: f64,
    pub expanded_log_reverse_forward: f64,
    pub log_reverse_forward: f64,
}

pub struct OligomerMixture<'p> {
    map: &'p FixedBasinInvolution,
    inverted: Vec<bool>,
    offsets: Vec<Pose>,
    key: Vec<i64>,
    anchors: Vec<Pose>,
    fused_map: Option<FixedBasinInvolution>,
    pub fused: Vec<FusedComponent>,
    log_weights: Vec<f64>,
    singles: usize,
    fit_candidates: usize,
    hard_checks: usize,
    /// Wall-clock seconds: label means and pair search, fits, hard checks.
    pub build_seconds: [f64; 3],
}

fn compose(a: Pose, b: Pose) -> Pose {
    let ra = rotation(a.orientation);
    Pose {
        position: add(a.position, matvec(ra, b.position)),
        orientation: quaternion(matmul(ra, rotation(b.orientation))),
    }
}
fn perturb(center: Pose, delta: Vec6) -> Pose {
    Pose {
        position: std::array::from_fn(|k| center.position[k] + delta[k]),
        orientation: quaternion(matmul(
            cayley([delta[3], delta[4], delta[5]]),
            rotation(center.orientation),
        )),
    }
}
fn rotation_angle(a: Pose, b: Pose) -> f64 {
    let d: f64 = a
        .orientation
        .iter()
        .zip(&b.orientation)
        .map(|(x, y)| x * y)
        .sum();
    2. * d.abs().min(1.).acos()
}
fn cholesky6(a: &Mat6) -> Option<Mat6> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let r = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !(r > 0. && r.is_finite()) {
                    return None;
                }
                l[i][i] = r.sqrt();
            } else {
                l[i][j] = r / l[j][j];
            }
        }
    }
    Some(l)
}
fn cholesky_solve(l: &Mat6, b: Vec6) -> Vec6 {
    let mut y = [0.; 6];
    for i in 0..6 {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.; 6];
    for i in (0..6).rev() {
        x[i] = (y[i] - (i + 1..6).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

/// The original bounded Gauss-Newton arithmetic, factored only so synthetic
/// residuals can test exit accounting without invoking protein geometry.
fn fit_pose_residual(
    start: Pose,
    limit: f64,
    policy: FusionFitPolicy,
    mut diagnostic: Option<&mut FusionFitDiagnostic>,
    residual: impl Fn(Pose) -> std::result::Result<[f64; 12], (usize, String)>,
) -> Option<(Pose, Mat6, f64)> {
    let evaluate = |g: Pose,
                    stage: FusionFitStage,
                    iteration: Option<usize>,
                    coordinate: Option<usize>,
                    direction: Option<i8>,
                    trial: Option<usize>,
                    diagnostic: &mut Option<&mut FusionFitDiagnostic>| {
        if let Some(d) = diagnostic.as_deref_mut() {
            d.residual_evaluations += 1;
        }
        match residual(g) {
            Ok(r) => Some(r),
            Err((label, error)) => {
                if let Some(d) = diagnostic.as_deref_mut() {
                    d.encode_failures += 1;
                    d.last_encode_failure = Some(FusionFitEncodeFailure {
                        stage,
                        iteration,
                        coordinate,
                        direction,
                        trial,
                        label,
                        error,
                    });
                }
                None
            }
        }
    };
    let chi2 = |r: &[f64; 12]| r.iter().map(|x| x * x).sum::<f64>();
    let jacobian = |c: Pose,
                    stage: FusionFitStage,
                    iteration: Option<usize>,
                    diagnostic: &mut Option<&mut FusionFitDiagnostic>| {
        if let Some(d) = diagnostic.as_deref_mut() {
            d.jacobian_evaluations += 1;
            // This reason applies only if one of the existing encodes fails.
            d.termination = if stage == FusionFitStage::FinalJacobian {
                FusionFitTermination::FinalJacobianEncodeFailure
            } else {
                FusionFitTermination::JacobianEncodeFailure
            };
        }
        let mut j = [[0.; 6]; 12];
        for k in 0..6 {
            let h = if k < 3 { 1e-5 } else { 1e-6 };
            let mut delta = [0.; 6];
            delta[k] = h;
            let plus = evaluate(
                perturb(c, delta),
                stage,
                iteration,
                Some(k),
                Some(1),
                None,
                diagnostic,
            )?;
            delta[k] = -h;
            let minus = evaluate(
                perturb(c, delta),
                stage,
                iteration,
                Some(k),
                Some(-1),
                None,
                diagnostic,
            )?;
            for r in 0..12 {
                j[r][k] = (plus[r] - minus[r]) / (2. * h);
            }
        }
        if let Some(d) = diagnostic.as_deref_mut() {
            d.nonfinite_jacobians += usize::from(j.iter().flatten().any(|x| !x.is_finite()));
        }
        Some(j)
    };
    let normal = |j: &[[f64; 6]; 12]| -> Mat6 {
        std::array::from_fn(|a| std::array::from_fn(|b| (0..12).map(|r| j[r][a] * j[r][b]).sum()))
    };
    let result = (|| {
        let mut center = start;
        if let Some(d) = diagnostic.as_deref_mut() {
            d.termination = FusionFitTermination::InitialEncodeFailure;
        }
        let mut r = evaluate(
            center,
            FusionFitStage::Initial,
            None,
            None,
            None,
            None,
            &mut diagnostic,
        )?;
        let mut value = chi2(&r);
        if let Some(d) = diagnostic.as_deref_mut() {
            d.start_chi2 = Some(value.into());
            d.final_chi2 = Some(value.into());
            d.nonfinite_objectives += usize::from(!value.is_finite());
        }
        let mut stop = FusionFitTermination::IterationLimit;
        for iteration in 0..16 {
            // This is the only arithmetic/control-flow difference between
            // policies. The original production guard remains unchanged.
            if policy == FusionFitPolicy::EarlyAbort && iteration >= 2 && value > 10. * limit {
                if let Some(d) = diagnostic.as_deref_mut() {
                    d.termination = FusionFitTermination::EarlyResidualCutoff;
                }
                return None;
            }
            if let Some(d) = diagnostic.as_deref_mut() {
                d.iterations += 1;
            }
            let j = jacobian(
                center,
                FusionFitStage::Jacobian,
                Some(iteration),
                &mut diagnostic,
            )?;
            if let Some(d) = diagnostic.as_deref_mut() {
                d.termination = FusionFitTermination::NormalCholeskyFailure;
            }
            let lower = cholesky6(&normal(&j))?;
            let gradient: Vec6 =
                std::array::from_fn(|a| -(0..12).map(|k| j[k][a] * r[k]).sum::<f64>());
            let mut delta = cholesky_solve(&lower, gradient);
            if let Some(d) = diagnostic.as_deref_mut() {
                d.nonfinite_gradients += usize::from(gradient.iter().any(|x| !x.is_finite()));
                d.nonfinite_steps += usize::from(delta.iter().any(|x| !x.is_finite()));
            }
            let mut improved = false;
            for trial_index in 0..12 {
                let trial = perturb(center, delta);
                if let Some(d) = diagnostic.as_deref_mut() {
                    d.line_search_trials += 1;
                }
                if let Some(tr) = evaluate(
                    trial,
                    FusionFitStage::LineSearch,
                    Some(iteration),
                    None,
                    None,
                    Some(trial_index),
                    &mut diagnostic,
                ) {
                    let trial_value = chi2(&tr);
                    if let Some(d) = diagnostic.as_deref_mut() {
                        d.nonfinite_objectives += usize::from(!trial_value.is_finite());
                    }
                    if trial_value <= value {
                        center = trial;
                        r = tr;
                        // Preserve the original second evaluation and order.
                        value = chi2(&tr);
                        improved = true;
                        if let Some(d) = diagnostic.as_deref_mut() {
                            d.accepted_steps += 1;
                            d.final_chi2 = Some(value.into());
                        }
                        break;
                    }
                }
                delta = delta.map(|x| 0.5 * x);
                if let Some(d) = diagnostic.as_deref_mut() {
                    d.backtracks += 1;
                }
            }
            if !improved {
                stop = FusionFitTermination::LineSearchStalled;
                break;
            }
            if delta.iter().map(|x| x * x).sum::<f64>() < 1e-24 {
                stop = FusionFitTermination::TinyStep;
                break;
            }
        }
        if let Some(d) = diagnostic.as_deref_mut() {
            d.iteration_stop = Some(stop);
        }
        let j = jacobian(center, FusionFitStage::FinalJacobian, None, &mut diagnostic)?;
        if let Some(d) = diagnostic.as_deref_mut() {
            d.termination = stop;
        }
        Some((center, normal(&j), value))
    })();
    if let Some(d) = diagnostic {
        d.disposition = match result {
            None => FusionFitDisposition::ReturnedNone,
            Some((_, _, mismatch)) if mismatch <= limit => FusionFitDisposition::Usable,
            Some(_) => FusionFitDisposition::AboveThreshold,
        };
    }
    result
}

impl<'p> OligomerMixture<'p> {
    /// `members` in fixed label order (the first is g0), `pool` the fixed
    /// anchors, `spectators` every non-member body (hard-core screen only).
    pub fn build(
        proposal: &'p DockingProposal,
        tree: &SphereTree,
        wall: &Container,
        members: &[Pose],
        spectators: &[Pose],
        pool: &[Pose],
        config: &OligomerConfig,
    ) -> Result<Self> {
        Self::build_impl(
            proposal,
            tree,
            wall,
            members,
            spectators,
            pool,
            config,
            FusionFitPolicy::EarlyAbort,
            false,
            None,
        )
    }

    /// The same construction with bounded passive accounting. The sink is
    /// reset before validation and remains incomplete on any returned error.
    /// This does not change screening, fit initialization, traversal or RNG.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_diagnostics(
        proposal: &'p DockingProposal,
        tree: &SphereTree,
        wall: &Container,
        members: &[Pose],
        spectators: &[Pose],
        pool: &[Pose],
        config: &OligomerConfig,
        diagnostics: &mut FusionBuildDiagnostics,
    ) -> Result<Self> {
        *diagnostics = FusionBuildDiagnostics::default();
        Self::build_impl(
            proposal,
            tree,
            wall,
            members,
            spectators,
            pool,
            config,
            FusionFitPolicy::EarlyAbort,
            false,
            Some(diagnostics),
        )
    }

    /// Opt-in bounded fitter comparison for the standalone diagnostic. This
    /// records existing fit evaluations without additional geometry or RNG.
    /// Production build entry points do not expose the alternative policy.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_fit_diagnostics(
        proposal: &'p DockingProposal,
        tree: &SphereTree,
        wall: &Container,
        members: &[Pose],
        spectators: &[Pose],
        pool: &[Pose],
        config: &OligomerConfig,
        policy: FusionFitPolicy,
        diagnostics: &mut FusionBuildDiagnostics,
    ) -> Result<Self> {
        *diagnostics = FusionBuildDiagnostics {
            fit_policy: policy,
            ..FusionBuildDiagnostics::default()
        };
        Self::build_impl(
            proposal,
            tree,
            wall,
            members,
            spectators,
            pool,
            config,
            policy,
            true,
            Some(diagnostics),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_impl(
        proposal: &'p DockingProposal,
        tree: &SphereTree,
        wall: &Container,
        members: &[Pose],
        spectators: &[Pose],
        pool: &[Pose],
        config: &OligomerConfig,
        fit_policy: FusionFitPolicy,
        record_fit_diagnostics: bool,
        mut diagnostics: Option<&mut FusionBuildDiagnostics>,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            !members.is_empty() && !pool.is_empty(),
            "Oligomer charts need members and anchors"
        );
        let (map, inverted, log_branch) = proposal.member_chart_parts();
        let nb = log_branch.len();
        let na = pool.len();
        let (key, offsets) = internal_key(members);
        let mut mixture = Self {
            map,
            inverted,
            offsets,
            key,
            anchors: pool.to_vec(),
            fused_map: None,
            fused: Vec::new(),
            log_weights: Vec::new(),
            singles: members.len() * na * nb,
            fit_candidates: 0,
            hard_checks: 0,
            build_seconds: [0.; 3],
        };
        let clock = std::time::Instant::now();
        // Implied g0 at every single label's chart mean.
        let means: Vec<Option<Pose>> = (0..mixture.singles)
            .map(|l| {
                let y = map.decode(l % nb, [0.; 6]).ok()?;
                Some(mixture.single_to_g0(l, y))
            })
            .collect();
        if let Some(d) = diagnostics.as_deref_mut() {
            d.requested_fused_mass = config.multi_contact_mass;
            d.single_labels = mixture.singles;
            d.decoded_means = means.iter().filter(|m| m.is_some()).count();
            d.mean_decode_failures = d.single_labels - d.decoded_means;
            d.possible_interface_pairs = cross_interface_pairs(means.chunks(nb).map(|c| c.len()))?;
            d.decoded_interface_pairs = cross_interface_pairs(
                means
                    .chunks(nb)
                    .map(|c| c.iter().filter(|m| m.is_some()).count()),
            )?;
        }
        let mut order: Vec<_> = (0..mixture.singles)
            .filter(|&l| means[l].is_some())
            .collect();
        order.sort_by(|&a, &b| {
            means[a].unwrap().position[0]
                .total_cmp(&means[b].unwrap().position[0])
                .then(a.cmp(&b))
        });
        let angle = config.pair_angle_degrees.to_radians();
        let mut candidates = Vec::new();
        for (k, &l1) in order.iter().enumerate() {
            let m1 = means[l1].unwrap();
            for &l2 in &order[k + 1..] {
                let m2 = means[l2].unwrap();
                if m2.position[0] - m1.position[0] > config.pair_distance_a {
                    break;
                }
                // Two contacts need distinct (member, anchor) interfaces.
                if l1 / nb == l2 / nb {
                    continue;
                }
                let d = norm(sub(m1.position, m2.position));
                if d < config.pair_distance_a {
                    if rotation_angle(m1, m2) < angle {
                        candidates.push((d, l1.min(l2), l1.max(l2)));
                    } else if let Some(diagnostics) = diagnostics.as_deref_mut() {
                        diagnostics.angle_rejected_pairs += 1;
                    }
                }
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        if let Some(d) = diagnostics.as_deref_mut() {
            d.candidate_pairs_before_cap = candidates.len();
            d.distance_rejected_pairs =
                d.decoded_interface_pairs - d.angle_rejected_pairs - candidates.len();
            d.candidates_truncated = candidates.len().saturating_sub(config.max_candidates);
        }
        candidates.truncate(config.max_candidates);
        mixture.fit_candidates = candidates.len();
        mixture.build_seconds[0] = clock.elapsed().as_secs_f64();
        let mut fits = Vec::new();
        for &(_, l1, l2) in &candidates {
            if let Some(d) = diagnostics.as_deref_mut() {
                d.fit_attempts += 1;
            }
            let fit_diagnostic = if record_fit_diagnostics {
                diagnostics.as_deref_mut().map(|d| {
                    d.fit_records.push(FusionFitDiagnostic::new(l1, l2));
                    d.fit_records.last_mut().unwrap()
                })
            } else {
                None
            };
            match mixture.fuse(
                l1,
                l2,
                means[l1].unwrap(),
                config.max_mismatch,
                fit_policy,
                fit_diagnostic,
            ) {
                Some((center, information, mismatch)) if mismatch <= config.max_mismatch => {
                    fits.push((mismatch, l1, l2, center, information));
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.usable_fits += 1;
                    }
                }
                Some(_) => {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.fits_above_threshold += 1;
                    }
                }
                None => {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.fit_returned_none += 1;
                    }
                }
            }
        }
        fits.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        mixture.build_seconds[1] = clock.elapsed().as_secs_f64() - mixture.build_seconds[0];
        let fixed: Vec<_> = spectators.iter().map(|&p| Placed::new(p)).collect();
        let scale = proposal.angular_length();
        let mut parameters = Vec::new();
        let mut fused_logs = Vec::new();
        for (mismatch, l1, l2, center, information) in fits {
            if mixture.fused.len() >= config.max_components
                || mixture.hard_checks >= config.max_hard_checks
            {
                if let Some(d) = diagnostics.as_deref_mut() {
                    d.stopped_by = Some(
                        match (
                            mixture.fused.len() >= config.max_components,
                            mixture.hard_checks >= config.max_hard_checks,
                        ) {
                            (true, true) => FusionBuildCap::ComponentsAndHardChecks,
                            (true, false) => FusionBuildCap::Components,
                            _ => FusionBuildCap::HardChecks,
                        },
                    );
                }
                break;
            }
            mixture.hard_checks += 1;
            if let Some(d) = diagnostics.as_deref_mut() {
                d.centers_checked += 1;
                d.records.push(FusionCenterDiagnostic {
                    first: l1,
                    second: l2,
                    mismatch,
                    center,
                    outcome: FusionCenterOutcome::Begun,
                    first_blocking_member: None,
                    first_blocking_spectator: None,
                    core_overlap_calls: 0,
                });
            }
            let bodies: Vec<_> = mixture
                .offsets
                .iter()
                .map(|&u| compose(center, u))
                .collect();
            let valid = bodies.iter().enumerate().all(|(member, &p)| {
                if p.validate().is_err() {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.invalid_pose_rejections += 1;
                        let record = d.records.last_mut().unwrap();
                        record.outcome = FusionCenterOutcome::InvalidPose;
                        record.first_blocking_member = Some(member);
                    }
                    return false;
                }
                if !wall.contains(p) {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.wall_rejections += 1;
                        let record = d.records.last_mut().unwrap();
                        record.outcome = FusionCenterOutcome::OutsideWall;
                        record.first_blocking_member = Some(member);
                    }
                    return false;
                }
                let placed = Placed::new(p);
                !fixed.iter().enumerate().any(|(spectator, s)| {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.core_overlap_calls += 1;
                        d.records.last_mut().unwrap().core_overlap_calls += 1;
                    }
                    let overlap = tree.overlaps(&placed, s);
                    if overlap && let Some(d) = diagnostics.as_deref_mut() {
                        d.core_rejections += 1;
                        let record = d.records.last_mut().unwrap();
                        record.outcome = FusionCenterOutcome::CoreOverlap;
                        record.first_blocking_member = Some(member);
                        record.first_blocking_spectator = Some(spectator);
                    }
                    overlap
                })
            });
            if !valid {
                continue;
            }
            // Chart coordinates are (t - c, L * Cayley), so scale rotations.
            let Some(lower) = cholesky6(&information) else {
                if let Some(d) = diagnostics.as_deref_mut() {
                    d.information_cholesky_failures += 1;
                    d.records.last_mut().unwrap().outcome =
                        FusionCenterOutcome::InformationNotPositiveDefinite;
                }
                continue;
            };
            let mut covariance = [[0.; 6]; 6];
            for j in 0..6 {
                let mut e = [0.; 6];
                e[j] = 1.;
                let column = cholesky_solve(&lower, e);
                let sj = if j < 3 { 1. } else { scale };
                for (i, row) in covariance.iter_mut().enumerate() {
                    let si = if i < 3 { 1. } else { scale };
                    row[j] = si * column[i] * sj;
                }
            }
            let covariance: Mat6 = std::array::from_fn(|i| {
                std::array::from_fn(|j| 0.5 * (covariance[i][j] + covariance[j][i]))
            });
            if cholesky6(&covariance).is_none() {
                if let Some(d) = diagnostics.as_deref_mut() {
                    d.covariance_cholesky_failures += 1;
                    d.records.last_mut().unwrap().outcome =
                        FusionCenterOutcome::CovarianceNotPositiveDefinite;
                }
                continue;
            }
            parameters.push(GaussianComponentParameters {
                anchor_position: center.position,
                anchor_rotation: rotation(center.orientation),
                mean: [0.; 6],
                covariance,
                weight: 1.,
            });
            fused_logs.push(log_branch[l1 % nb] + log_branch[l2 % nb] - 0.5 * mismatch);
            mixture.fused.push(FusedComponent {
                first: l1,
                second: l2,
                mismatch,
                center,
            });
            if let Some(d) = diagnostics.as_deref_mut() {
                d.retained_components += 1;
                d.records.last_mut().unwrap().outcome = FusionCenterOutcome::Retained;
            }
        }
        if let Some(d) = diagnostics.as_deref_mut() {
            d.unvisited_fits = d.usable_fits - d.centers_checked;
            d.component_cap_reached = mixture.fused.len() >= config.max_components;
            d.hard_check_cap_reached = mixture.hard_checks >= config.max_hard_checks;
        }
        mixture.build_seconds[2] =
            clock.elapsed().as_secs_f64() - mixture.build_seconds[0] - mixture.build_seconds[1];
        // Mismatch order only spends the check budget. Index components by
        // label pair so near-tied mismatches cannot permute chart indices
        // between the two endpoints' reconstructions.
        let mut order: Vec<_> = (0..mixture.fused.len()).collect();
        order.sort_by_key(|&k| (mixture.fused[k].first, mixture.fused[k].second));
        let parameters: Vec<_> = order.iter().map(|&k| parameters[k].clone()).collect();
        let fused_logs: Vec<_> = order.iter().map(|&k| fused_logs[k]).collect();
        mixture.fused = order.iter().map(|&k| mixture.fused[k].clone()).collect();
        let mass = if parameters.is_empty() {
            0.
        } else {
            mixture.fused_map = Some(FixedBasinInvolution::new(
                parameters,
                scale,
                proposal.correlation(),
                vec![BasinPair {
                    first: 0,
                    second: 0,
                    weight: 1.,
                }],
            )?);
            config.multi_contact_mass
        };
        let single_offset = (1. - mass).ln() - ((members.len() * na) as f64).ln();
        mixture.log_weights = (0..mixture.singles)
            .map(|l| single_offset + log_branch[l % nb])
            .collect();
        let fused_total = log_sum(&fused_logs);
        mixture
            .log_weights
            .extend(fused_logs.iter().map(|w| mass.ln() + w - fused_total));
        if let Some(d) = diagnostics {
            d.effective_fused_mass = mass;
            d.complete = true;
        }
        Ok(mixture)
    }

    /// Rounded internal geometry this catalogue was built from.
    pub fn key(&self) -> &[i64] {
        &self.key
    }
    pub fn fit_candidates(&self) -> usize {
        self.fit_candidates
    }
    pub fn hard_checks(&self) -> usize {
        self.hard_checks
    }
    pub fn label_count(&self) -> usize {
        self.log_weights.len()
    }
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    fn nb(&self) -> usize {
        self.inverted.len()
    }
    fn to_single(&self, label: usize, g0: Pose) -> Pose {
        let nb = self.nb();
        let (member, anchor, branch) = (
            label / (self.anchors.len() * nb),
            label / nb % self.anchors.len(),
            label % nb,
        );
        let relative = compose(
            invert_relative_pose(self.anchors[anchor]),
            compose(g0, self.offsets[member]),
        );
        if self.inverted[branch] {
            invert_relative_pose(relative)
        } else {
            relative
        }
    }
    fn single_to_g0(&self, label: usize, y: Pose) -> Pose {
        let nb = self.nb();
        let (member, anchor, branch) = (
            label / (self.anchors.len() * nb),
            label / nb % self.anchors.len(),
            label % nb,
        );
        let relative = if self.inverted[branch] {
            invert_relative_pose(y)
        } else {
            y
        };
        compose(
            compose(self.anchors[anchor], relative),
            invert_relative_pose(self.offsets[member]),
        )
    }
    fn chart(&self, label: usize) -> (&FixedBasinInvolution, usize) {
        if label < self.singles {
            (self.map, label % self.nb())
        } else {
            (
                self.fused_map.as_ref().expect("fused label without charts"),
                label - self.singles,
            )
        }
    }
    fn to_chart(&self, label: usize, g0: Pose) -> Pose {
        if label < self.singles {
            self.to_single(label, g0)
        } else {
            g0
        }
    }
    fn chart_to_g0(&self, label: usize, y: Pose) -> Pose {
        if label < self.singles {
            self.single_to_g0(label, y)
        } else {
            y
        }
    }

    /// Minimize both labels' squared whitened residuals over g0, starting at
    /// the first label's implied mean. Default policy preserves the original
    /// early large-residual exit; the diagnostic policy changes only that exit.
    fn fuse(
        &self,
        l1: usize,
        l2: usize,
        start: Pose,
        limit: f64,
        policy: FusionFitPolicy,
        diagnostic: Option<&mut FusionFitDiagnostic>,
    ) -> Option<(Pose, Mat6, f64)> {
        let nb = self.nb();
        fit_pose_residual(start, limit, policy, diagnostic, |g| {
            let z1 = self
                .map
                .encode(l1 % nb, self.to_single(l1, g))
                .map_err(|error| (l1, error.to_string()))?;
            let z2 = self
                .map
                .encode(l2 % nb, self.to_single(l2, g))
                .map_err(|error| (l2, error.to_string()))?;
            Ok(std::array::from_fn(
                |k| if k < 6 { z1[k] } else { z2[k - 6] },
            ))
        })
    }

    /// Per-label log terms w_l q_l(g0); their log-sum is log G_O(g0).
    pub fn label_logs(&self, g0: Pose) -> Vec<f64> {
        self.log_weights
            .iter()
            .enumerate()
            .map(|(l, &w)| {
                if w == f64::NEG_INFINITY {
                    return w;
                }
                let (map, chart) = self.chart(l);
                w + map.log_density(chart, self.to_chart(l, g0))
            })
            .collect()
    }

    /// Strict per-label terms for callers that must retain numerical failures.
    /// Zero-weight labels and exact chart seams remain legitimate zero terms.
    pub fn checked_label_logs(&self, g0: Pose) -> Result<Vec<f64>> {
        g0.validate()?;
        self.log_weights
            .iter()
            .enumerate()
            .map(|(label, &weight)| {
                if weight == f64::NEG_INFINITY {
                    return Ok(weight);
                }
                ensure!(
                    weight.is_finite(),
                    "Nonfinite oligomer label {label} weight"
                );
                let (map, chart) = self.chart(label);
                let density = map
                    .checked_log_density(chart, self.to_chart(label, g0))
                    .with_context(|| format!("Cannot score oligomer label {label}"))?;
                if density == f64::NEG_INFINITY {
                    return Ok(density);
                }
                let term = weight + density;
                ensure!(
                    term.is_finite(),
                    "Unrepresentable oligomer label {label} term"
                );
                Ok(term)
            })
            .collect()
    }

    pub fn log_density(&self, members: &[Pose]) -> f64 {
        log_sum(&self.label_logs(members[0]))
    }

    /// Read-only coverage diagnostic using precisely the production chart,
    /// including member offset, anchor, reciprocal inversion or fused chart.
    pub fn source_chart_coordinates(&self, label: usize, g0: Pose) -> Result<Vec6> {
        ensure!(label < self.label_count(), "Invalid oligomer source label");
        let (map, chart) = self.chart(label);
        map.encode(chart, self.to_chart(label, g0))
    }

    /// Target-only independent draw from this complete fused/unfused mixture.
    /// Restricted to a singleton so there is no source-dependent carry or
    /// internal-offset guard inside a caller's hard-conditioning loop.
    pub fn draw_singleton_independent(&self, rng: &mut StdRng) -> Result<(Option<Pose>, Value)> {
        ensure!(
            self.offsets.len() == 1,
            "independent mixture draws require a singleton"
        );
        let target = draw_log_category(rng, &self.log_weights)?
            .context("No independent destination component")?;
        let latent: Vec6 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let mut trace =
            json!({"target":target,"target_label":self.describe(target),"target_latent":latent});
        let (map, chart) = self.chart(target);
        let decoded = map.decode(chart, latent).and_then(|y| {
            let p = self.chart_to_g0(target, y);
            p.validate()?;
            Ok(p)
        });
        match decoded {
            Ok(p) => Ok((Some(p), trace)),
            Err(error) => {
                trace["null_reason"] = json!(error.to_string());
                Ok((None, trace))
            }
        }
    }

    /// Posterior source label, independent destination label and noise.
    pub fn propose(
        &self,
        rng: &mut StdRng,
        members: &[Pose],
        handle: usize,
    ) -> Result<(Option<Pose>, Value)> {
        let old = self.label_logs(members[0]);
        let Some(source) = draw_log_category(rng, &old)? else {
            return Ok((
                None,
                json!({"branch":"involution","charts":"oligomer",
                    "null_reason":"No finite Gaussian source density"}),
            ));
        };
        let target =
            draw_log_category(rng, &self.log_weights)?.context("No destination component")?;
        let noise: Vec6 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let context = json!({"fused_components":self.fused.len(),
            "fit_candidates":self.fit_candidates,"hard_checks":self.hard_checks,
            "build_seconds":self.build_seconds,
            "source_label":self.describe(source),"target_label":self.describe(target)});
        match self.apply(members, handle, source, target, noise) {
            Ok(step) => {
                let mut info = serde_json::to_value(&step)?;
                info["branch"] = json!("involution");
                info["charts"] = json!("oligomer");
                info["source_law"] = json!("posterior");
                info["oligomer"] = context;
                Ok((Some(step.handle), info))
            }
            Err(error) => Ok((
                None,
                json!({"branch":"involution","charts":"oligomer","oligomer":context,
                    "noise":noise,"null_reason":error.to_string()}),
            )),
        }
    }

    pub fn describe(&self, label: usize) -> Value {
        if label < self.singles {
            let nb = self.nb();
            json!({"kind":"single","member":label / (self.anchors.len() * nb),
                "anchor":label / nb % self.anchors.len(),"branch":label % nb})
        } else {
            let f = &self.fused[label - self.singles];
            json!({"kind":"fused","index":label - self.singles,"first":self.describe(f.first),
                "second":self.describe(f.second),"mismatch":f.mismatch})
        }
    }

    /// Deterministic map for given labels and noise. The inverse applies the
    /// swapped labels and `step.inverse_trace.noise` at the carried endpoint.
    pub fn apply(
        &self,
        members: &[Pose],
        handle: usize,
        source: usize,
        target: usize,
        noise: Vec6,
    ) -> Result<OligomerStep> {
        ensure!(
            handle < members.len()
                && members.len() == self.offsets.len()
                && source < self.label_count()
                && target < self.label_count(),
            "Invalid oligomer label or members"
        );
        let old = self.label_logs(members[0]);
        let full_old = log_sum(&old);
        ensure!(full_old.is_finite(), "Nonfinite oligomer density");
        let step = cross_chart_step(
            self.chart(source),
            self.chart(target),
            self.to_chart(source, members[0]),
            noise,
        )?;
        if self.map.correlation() == 1. && source == target {
            return Ok(OligomerStep {
                handle: members[handle],
                source,
                target,
                step,
                identity: true,
                full_old_log_density: full_old,
                full_new_log_density: full_old,
                label_log_reverse_forward: 0.,
                expanded_log_reverse_forward: 0.,
                log_reverse_forward: 0.,
            });
        }
        let moved = self.chart_to_g0(target, step.pose);
        moved.validate()?;
        let reference = members[0];
        let delta = matmul(
            rotation(moved.orientation),
            transpose(rotation(reference.orientation)),
        );
        let carry = |p: Pose| Pose {
            position: add(
                moved.position,
                matvec(delta, sub(p.position, reference.position)),
            ),
            orientation: quaternion(matmul(delta, rotation(p.orientation))),
        };
        let proposed = carry(members[handle]);
        proposed.validate()?;
        let new = self.label_logs(carry(members[0]));
        let full_new = log_sum(&new);
        ensure!(full_new.is_finite(), "Nonfinite oligomer density");
        let labels = (new[target] - full_new) - (old[source] - full_old) + self.log_weights[source]
            - self.log_weights[target];
        Ok(OligomerStep {
            handle: proposed,
            source,
            target,
            identity: false,
            full_old_log_density: full_old,
            full_new_log_density: full_new,
            label_log_reverse_forward: labels,
            expanded_log_reverse_forward: step.log_correction + labels,
            log_reverse_forward: full_old - full_new,
            step,
        })
    }
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod checked_label_tests {
    use super::*;

    const ORIGIN: Pose = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };

    fn map(charts: usize) -> FixedBasinInvolution {
        FixedBasinInvolution::new(
            (0..charts)
                .map(|_| GaussianComponentParameters {
                    anchor_position: [0.; 3],
                    anchor_rotation: IDENTITY,
                    mean: [0.; 6],
                    covariance: std::array::from_fn(|i| {
                        std::array::from_fn(|j| if i == j { 1. } else { 0. })
                    }),
                    weight: 1. / charts as f64,
                })
                .collect(),
            1.,
            0.,
            vec![BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            }],
        )
        .unwrap()
    }

    fn mixture(map: &FixedBasinInvolution) -> OligomerMixture<'_> {
        OligomerMixture {
            map,
            inverted: vec![false, true],
            offsets: vec![ORIGIN],
            key: vec![],
            anchors: vec![ORIGIN],
            fused_map: Some(self::map(1)),
            fused: vec![FusedComponent {
                first: 0,
                second: 1,
                mismatch: 0.,
                center: ORIGIN,
            }],
            log_weights: vec![0.3_f64.ln(), 0.4_f64.ln(), 0.3_f64.ln()],
            singles: 2,
            fit_candidates: 0,
            hard_checks: 0,
            build_seconds: [0.; 3],
        }
    }

    #[test]
    fn two_neighbor_singleton_checked_labels_cover_single_reciprocal_and_fused() {
        let map = map(2);
        let mixture = mixture(&map);
        let p = Pose {
            position: [1., -2., 0.5],
            orientation: quaternion(cayley([0.2, 0.4, -0.1])),
        };
        let terms = mixture.checked_label_logs(p).unwrap();
        assert_eq!(terms.len(), 3);
        assert!(terms.iter().all(|v| v.is_finite()));
        assert_eq!(terms, mixture.label_logs(p));
        let seam = Pose {
            orientation: [0., 1., 0., 0.],
            ..ORIGIN
        };
        assert_eq!(
            mixture.checked_label_logs(seam).unwrap(),
            vec![f64::NEG_INFINITY; 3]
        );
    }

    #[test]
    fn two_neighbor_singleton_checked_labels_skip_zero_weight_and_propagate_errors() {
        let map = map(2);
        let mut mixture = mixture(&map);
        // The world-space fused chart remains representable. The singles'
        // huge fixed anchor instead makes their Gaussian quadratics overflow.
        mixture.anchors[0].position[0] = 1e200;
        assert!(mixture.checked_label_logs(ORIGIN).is_err());
        mixture.log_weights[0] = f64::NEG_INFINITY;
        mixture.log_weights[1] = f64::NEG_INFINITY;
        let terms = mixture.checked_label_logs(ORIGIN).unwrap();
        assert_eq!(terms[..2], [f64::NEG_INFINITY; 2]);
        assert!(terms[2].is_finite());
        let invalid = Pose {
            orientation: [0.; 4],
            ..ORIGIN
        };
        assert!(mixture.checked_label_logs(invalid).is_err());
        for weight in [f64::NAN, f64::INFINITY] {
            mixture.log_weights[0] = weight;
            assert!(mixture.checked_label_logs(ORIGIN).is_err());
        }
    }

    #[test]
    fn two_neighbor_singleton_checked_labels_reject_weighted_sum_overflow() {
        let map = map(2);
        let mut mixture = mixture(&map);
        mixture.log_weights[0] = -f64::MAX;
        let tail = Pose {
            position: [1e154, 0., 0.],
            ..ORIGIN
        };
        assert!(map.checked_log_density(0, tail).unwrap().is_finite());
        assert_eq!(mixture.label_logs(tail)[0], f64::NEG_INFINITY);
        assert!(mixture.checked_label_logs(tail).is_err());
    }
}

#[cfg(test)]
mod fusion_diagnostic_tests {
    use super::*;
    use crate::{
        docking::DockingMethod,
        geometry::{Atom, Shape},
        proposal::FrozenRelativePoseProposal,
    };
    use rand::{RngExt, SeedableRng};

    const ORIGIN: Pose = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };

    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            ..ORIGIN
        }
    }

    fn parameters(n: usize) -> Vec<GaussianComponentParameters> {
        (0..n)
            .map(|i| GaussianComponentParameters {
                anchor_position: [0.05 * i as f64, 0., 0.],
                anchor_rotation: IDENTITY,
                mean: [0.; 6],
                covariance: std::array::from_fn(|i| {
                    std::array::from_fn(|j| if i == j { 1. } else { 0. })
                }),
                weight: 1. / n as f64,
            })
            .collect()
    }

    fn proposal(parameters: Vec<GaussianComponentParameters>) -> DockingProposal {
        let hash = "a".repeat(64);
        DockingProposal::new(
            FrozenRelativePoseProposal::from_components_open(
                parameters, 1., [100.; 3], 0.1, &hash, &hash,
            )
            .unwrap(),
            DockingMethod::PosteriorInvolution,
            0.6,
            [0.; 3],
        )
        .unwrap()
    }

    fn tree() -> SphereTree {
        SphereTree::new(Shape {
            name: "synthetic asymmetric union".into(),
            volume: 0.,
            atoms: vec![
                Atom {
                    center: [0.; 3],
                    radius: 0.1,
                },
                Atom {
                    center: [0.23, 0.04, 0.],
                    radius: 0.07,
                },
            ],
        })
        .unwrap()
    }

    fn check_accounting(d: &FusionBuildDiagnostics, config: &OligomerConfig) {
        assert!(d.complete);
        assert_eq!(d.single_labels, d.decoded_means + d.mean_decode_failures);
        assert_eq!(
            d.decoded_interface_pairs,
            d.distance_rejected_pairs + d.angle_rejected_pairs + d.candidate_pairs_before_cap
        );
        assert_eq!(
            d.candidate_pairs_before_cap,
            d.fit_attempts + d.candidates_truncated
        );
        assert_eq!(
            d.fit_attempts,
            d.fit_returned_none + d.fits_above_threshold + d.usable_fits
        );
        assert_eq!(d.usable_fits, d.centers_checked + d.unvisited_fits);
        assert_eq!(
            d.centers_checked,
            d.invalid_pose_rejections
                + d.wall_rejections
                + d.core_rejections
                + d.information_cholesky_failures
                + d.covariance_cholesky_failures
                + d.retained_components
        );
        assert_eq!(d.centers_checked, d.records.len());
        assert!(d.records.len() <= config.max_hard_checks);
        assert_eq!(
            d.core_overlap_calls,
            d.records
                .iter()
                .map(|r| r.core_overlap_calls)
                .sum::<usize>()
        );
        assert!(
            d.records
                .iter()
                .all(|r| r.outcome != FusionCenterOutcome::Begun)
        );
        assert_eq!(
            d.effective_fused_mass,
            if d.retained_components == 0 {
                0.
            } else {
                config.multi_contact_mass
            }
        );
    }

    #[test]
    fn fusion_diagnostics_preserve_catalogue_density_proposals_and_rng() {
        let proposal = proposal(parameters(3));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let config = OligomerConfig::default();
        for members in [vec![ORIGIN], vec![ORIGIN, pose(0.8)]] {
            let pool = [pose(-0.2), pose(0.25)];
            let spectators = [pose(10.), pose(-10.)];
            let ordinary = OligomerMixture::build(
                &proposal,
                &tree,
                &wall,
                &members,
                &spectators,
                &pool,
                &config,
            )
            .unwrap();
            let mut d = FusionBuildDiagnostics::default();
            let measured = OligomerMixture::build_with_diagnostics(
                &proposal,
                &tree,
                &wall,
                &members,
                &spectators,
                &pool,
                &config,
                &mut d,
            )
            .unwrap();
            check_accounting(&d, &config);
            assert!(!ordinary.fused.is_empty());
            assert_eq!(ordinary.log_weights(), measured.log_weights());
            assert_eq!(ordinary.key(), measured.key());
            assert_eq!(
                serde_json::to_value(&ordinary.fused).unwrap(),
                serde_json::to_value(&measured.fused).unwrap()
            );
            assert_eq!(ordinary.fit_candidates(), measured.fit_candidates());
            assert_eq!(ordinary.hard_checks(), measured.hard_checks());
            for p in [
                pose(-0.4),
                ORIGIN,
                Pose {
                    position: [0.2, -0.1, 0.3],
                    orientation: quaternion(cayley([0.1, 0.2, -0.1])),
                },
            ] {
                assert_eq!(
                    ordinary.checked_label_logs(p).unwrap(),
                    measured.checked_label_logs(p).unwrap()
                );
            }
            let (mut a, mut b) = (StdRng::seed_from_u64(173), StdRng::seed_from_u64(173));
            for _ in 0..24 {
                if members.len() == 1 {
                    assert_eq!(
                        ordinary.draw_singleton_independent(&mut a).unwrap(),
                        measured.draw_singleton_independent(&mut b).unwrap()
                    );
                }
                let (pa, mut ta) = ordinary
                    .propose(&mut a, &members, members.len() - 1)
                    .unwrap();
                let (pb, mut tb) = measured
                    .propose(&mut b, &members, members.len() - 1)
                    .unwrap();
                // Timing is observational; every pose, label, latent and MH term must match.
                ta["oligomer"]
                    .as_object_mut()
                    .unwrap()
                    .remove("build_seconds");
                tb["oligomer"]
                    .as_object_mut()
                    .unwrap()
                    .remove("build_seconds");
                assert_eq!(pa, pb);
                assert_eq!(ta, tb);
            }
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }

    #[test]
    fn fusion_diagnostics_partition_distance_angle_and_missing_means() {
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let config = OligomerConfig::default();
        let proposal = proposal(parameters(2));
        for (pool, distance, angle) in [
            ([ORIGIN, pose(20.)], 4, 0),
            (
                [
                    ORIGIN,
                    Pose {
                        orientation: [0., 1., 0., 0.],
                        ..ORIGIN
                    },
                ],
                0,
                4,
            ),
        ] {
            let mut d = FusionBuildDiagnostics::default();
            OligomerMixture::build_with_diagnostics(
                &proposal,
                &tree,
                &wall,
                &[ORIGIN],
                &[],
                &pool,
                &config,
                &mut d,
            )
            .unwrap();
            check_accounting(&d, &config);
            assert_eq!(d.possible_interface_pairs, 4);
            assert_eq!(d.decoded_interface_pairs, 4);
            assert_eq!(d.distance_rejected_pairs, distance);
            assert_eq!(d.angle_rejected_pairs, angle);
            assert_eq!(d.centers_checked, 0);
        }
        let mut bad = parameters(1);
        bad[0].anchor_position[0] = 1e308;
        bad[0].mean[0] = 1e308;
        let proposal = self::proposal(bad);
        let mut d = FusionBuildDiagnostics::default();
        OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &[],
            &[ORIGIN; 2],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert_eq!(d.mean_decode_failures, 2);
        assert_eq!(d.possible_interface_pairs, 1);
        assert_eq!(d.decoded_interface_pairs, 0);
        assert_eq!(d.fit_attempts, 0);
    }

    #[test]
    fn fusion_diagnostics_record_candidate_truncation_and_both_caps() {
        let proposal = proposal(parameters(3));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        for (candidates, components, checks, stopped) in [
            (1, 32, 256, None),
            (4096, 1, 256, Some(FusionBuildCap::Components)),
            (4096, 32, 1, Some(FusionBuildCap::HardChecks)),
            (4096, 1, 1, Some(FusionBuildCap::ComponentsAndHardChecks)),
        ] {
            let config = OligomerConfig {
                max_candidates: candidates,
                max_components: components,
                max_hard_checks: checks,
                ..OligomerConfig::default()
            };
            let mut d = FusionBuildDiagnostics::default();
            OligomerMixture::build_with_diagnostics(
                &proposal,
                &tree,
                &wall,
                &[ORIGIN],
                &[],
                &[ORIGIN; 2],
                &config,
                &mut d,
            )
            .unwrap();
            check_accounting(&d, &config);
            assert_eq!(d.candidate_pairs_before_cap, 9);
            assert_eq!(d.candidates_truncated, if candidates == 1 { 8 } else { 0 });
            assert_eq!(d.centers_checked, 1);
            assert_eq!(d.stopped_by, stopped);
            assert_eq!(d.unvisited_fits, if candidates == 1 { 0 } else { 8 });
        }
    }

    #[test]
    fn fusion_diagnostics_preserve_first_blocker_and_wall_short_circuit() {
        let proposal = proposal(parameters(1));
        let tree = tree();
        let wall = Container::new(1., &tree).unwrap();
        let config = OligomerConfig::default();
        let mut d = FusionBuildDiagnostics::default();
        let spectators = [pose(10.), ORIGIN, ORIGIN];
        let mixture = OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &spectators,
            &[ORIGIN; 2],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert!(mixture.fused.is_empty());
        assert_eq!(d.core_rejections, 1);
        assert_eq!(d.core_overlap_calls, 2);
        assert_eq!(d.records[0].first_blocking_member, Some(0));
        assert_eq!(d.records[0].first_blocking_spectator, Some(1));
        OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &spectators,
            &[pose(3.); 2],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert_eq!(d.wall_rejections, 1);
        assert_eq!(d.core_overlap_calls, 0);
        assert_eq!(d.records[0].outcome, FusionCenterOutcome::OutsideWall);
        assert_eq!(d.records[0].first_blocking_spectator, None);
    }

    #[test]
    fn fusion_diagnostics_separate_returned_none_from_returned_bad_mismatch() {
        let proposal = proposal(parameters(1));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let mut d = FusionBuildDiagnostics::default();
        let config = OligomerConfig {
            max_mismatch: 0.1,
            ..OligomerConfig::default()
        };
        OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &[],
            &[ORIGIN, pose(1.)],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert_eq!(d.fits_above_threshold, 1);
        assert_eq!(d.fit_returned_none, 0);
        let config = OligomerConfig {
            pair_angle_degrees: 181.,
            ..OligomerConfig::default()
        };
        OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &[],
            &[
                ORIGIN,
                Pose {
                    orientation: [0., 1., 0., 0.],
                    ..ORIGIN
                },
            ],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert_eq!(d.fit_returned_none, 1);
        assert_eq!(d.fits_above_threshold, 0);
    }

    #[test]
    fn fusion_diagnostics_zero_mass_does_not_skip_construction_and_invalid_config_resets_sink() {
        let proposal = proposal(parameters(2));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let config = OligomerConfig {
            multi_contact_mass: 0.,
            ..OligomerConfig::default()
        };
        let mut d = FusionBuildDiagnostics::default();
        let mixture = OligomerMixture::build_with_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &[],
            &[ORIGIN; 2],
            &config,
            &mut d,
        )
        .unwrap();
        check_accounting(&d, &config);
        assert_eq!(d.retained_components, 4);
        assert_eq!(d.effective_fused_mass, 0.);
        assert!(
            mixture.log_weights()[mixture.singles..]
                .iter()
                .all(|v| *v == f64::NEG_INFINITY)
        );
        let invalid = OligomerConfig {
            max_hard_checks: 0,
            ..config
        };
        assert!(
            OligomerMixture::build_with_diagnostics(
                &proposal,
                &tree,
                &wall,
                &[ORIGIN],
                &[],
                &[ORIGIN; 2],
                &invalid,
                &mut d
            )
            .is_err()
        );
        assert!(!d.complete);
        assert_eq!(d.centers_checked, 0);
        assert!(d.records.is_empty());
    }

    #[test]
    fn fusion_fit_diagnostics_preserve_default_catalogue_density_and_seeded_draws() {
        let proposal = proposal(parameters(3));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let config = OligomerConfig::default();
        assert_eq!(FusionFitPolicy::default(), FusionFitPolicy::EarlyAbort);
        assert_eq!(
            serde_json::to_value(FusionFitPolicy::default()).unwrap(),
            "early_abort"
        );
        assert!(serde_json::from_str::<FusionFitPolicy>("\"unbounded\"").is_err());
        for members in [vec![ORIGIN], vec![ORIGIN, pose(0.8)]] {
            let pool = [pose(-0.2), pose(0.25)];
            let spectators = [pose(10.), pose(-10.)];
            let mut old = FusionBuildDiagnostics::default();
            let ordinary = OligomerMixture::build_with_diagnostics(
                &proposal,
                &tree,
                &wall,
                &members,
                &spectators,
                &pool,
                &config,
                &mut old,
            )
            .unwrap();
            let mut diagnostic = FusionBuildDiagnostics::default();
            let traced = OligomerMixture::build_with_fit_diagnostics(
                &proposal,
                &tree,
                &wall,
                &members,
                &spectators,
                &pool,
                &config,
                FusionFitPolicy::EarlyAbort,
                &mut diagnostic,
            )
            .unwrap();
            check_accounting(&diagnostic, &config);
            assert!(old.fit_records.is_empty());
            assert_eq!(diagnostic.fit_records.len(), diagnostic.fit_attempts);
            assert!(diagnostic.fit_records.len() <= config.max_candidates);
            let mut old_json = serde_json::to_value(&old).unwrap();
            let mut traced_json = serde_json::to_value(&diagnostic).unwrap();
            old_json.as_object_mut().unwrap().remove("fit_records");
            traced_json.as_object_mut().unwrap().remove("fit_records");
            assert_eq!(old_json, traced_json);
            for (disposition, count) in [
                (
                    FusionFitDisposition::ReturnedNone,
                    diagnostic.fit_returned_none,
                ),
                (
                    FusionFitDisposition::AboveThreshold,
                    diagnostic.fits_above_threshold,
                ),
                (FusionFitDisposition::Usable, diagnostic.usable_fits),
            ] {
                assert_eq!(
                    diagnostic
                        .fit_records
                        .iter()
                        .filter(|r| r.disposition == disposition)
                        .count(),
                    count
                );
            }
            for record in &diagnostic.fit_records {
                assert_ne!(record.termination, FusionFitTermination::Begun);
                assert!(record.iterations <= 16 && record.backtracks <= 16 * 12);
                assert!(record.residual_evaluations <= 1 + 16 * (12 + 12) + 12);
            }
            assert_eq!(ordinary.log_weights(), traced.log_weights());
            assert_eq!(
                serde_json::to_value(&ordinary.fused).unwrap(),
                serde_json::to_value(&traced.fused).unwrap()
            );
            for p in [ORIGIN, pose(0.4)] {
                assert_eq!(
                    ordinary.checked_label_logs(p).unwrap(),
                    traced.checked_label_logs(p).unwrap()
                );
            }
            let (mut a, mut b) = (StdRng::seed_from_u64(273), StdRng::seed_from_u64(273));
            for _ in 0..12 {
                if members.len() == 1 {
                    assert_eq!(
                        ordinary.draw_singleton_independent(&mut a).unwrap(),
                        traced.draw_singleton_independent(&mut b).unwrap()
                    );
                }
                let (pa, mut ta) = ordinary
                    .propose(&mut a, &members, members.len() - 1)
                    .unwrap();
                let (pb, mut tb) = traced.propose(&mut b, &members, members.len() - 1).unwrap();
                ta["oligomer"]
                    .as_object_mut()
                    .unwrap()
                    .remove("build_seconds");
                tb["oligomer"]
                    .as_object_mut()
                    .unwrap()
                    .remove("build_seconds");
                assert_eq!(pa, pb);
                assert_eq!(ta, tb);
            }
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }

    fn fit_test_residual(p: Pose) -> [f64; 12] {
        let mut r = [0.; 12];
        r[..3].copy_from_slice(&p.position);
        for k in 0..3 {
            r[k + 3] = p.orientation[k + 1] / p.orientation[0];
        }
        r
    }

    #[test]
    fn fusion_fit_diagnostics_early_cutoff_full_iterations_and_no_extra_evaluations() {
        // Smooth full-rank local model with an irreducible squared residual
        // 15²=225>120. The x² residual keeps improving beyond two GN steps;
        // no policy can produce an acceptable <=12 fit for this model.
        for policy in [FusionFitPolicy::EarlyAbort, FusionFitPolicy::FullIterations] {
            let calls = std::cell::Cell::new(0usize);
            let residual = |p| {
                calls.set(calls.get() + 1);
                let mut r = fit_test_residual(p);
                r[0] *= r[0];
                r[6] = 15.;
                Ok(r)
            };
            let plain = fit_pose_residual(pose(1.), 12., policy, None, residual);
            let plain_calls = calls.replace(0);
            let mut record = FusionFitDiagnostic::new(2, 5);
            let traced = fit_pose_residual(pose(1.), 12., policy, Some(&mut record), residual);
            assert_eq!(
                serde_json::to_value(plain).unwrap(),
                serde_json::to_value(traced).unwrap()
            );
            assert_eq!(calls.get(), plain_calls);
            assert_eq!(record.residual_evaluations, plain_calls);
            assert_eq!(record.start_chi2, Some(FusionFitValue::Finite(226.)));
            assert_eq!(record.encode_failures, 0);
            assert_eq!(record.backtracks, 0);
            match policy {
                FusionFitPolicy::EarlyAbort => {
                    assert!(traced.is_none());
                    assert_eq!(
                        record.termination,
                        FusionFitTermination::EarlyResidualCutoff
                    );
                    assert_eq!(record.disposition, FusionFitDisposition::ReturnedNone);
                    assert_eq!(record.iterations, 2);
                    assert_eq!(plain_calls, 1 + 2 * (12 + 1));
                }
                FusionFitPolicy::FullIterations => {
                    assert!(traced.unwrap().2 >= 225.);
                    assert_eq!(record.termination, FusionFitTermination::IterationLimit);
                    assert_eq!(record.disposition, FusionFitDisposition::AboveThreshold);
                    assert_eq!(record.iterations, 16);
                    assert_eq!(plain_calls, 1 + 16 * (12 + 1) + 12);
                }
            }
        }
    }

    #[test]
    fn fusion_fit_diagnostics_encode_stages_and_recoverable_backtracks() {
        // A zero-residual affine model takes one zero step. Inject failures at
        // independently counted existing evaluator calls, not extra probes.
        for (first_failure, last_failure, termination, evaluations) in [
            (1, 1, FusionFitTermination::InitialEncodeFailure, 1),
            (2, 2, FusionFitTermination::JacobianEncodeFailure, 2),
            (15, 15, FusionFitTermination::FinalJacobianEncodeFailure, 15),
            (14, 25, FusionFitTermination::LineSearchStalled, 37),
        ] {
            let calls = std::cell::Cell::new(0usize);
            let mut record = FusionFitDiagnostic::new(2, 5);
            let result = fit_pose_residual(
                ORIGIN,
                12.,
                FusionFitPolicy::EarlyAbort,
                Some(&mut record),
                |p| {
                    calls.set(calls.get() + 1);
                    if (first_failure..=last_failure).contains(&calls.get()) {
                        Err((5, "synthetic encode failure".into()))
                    } else {
                        Ok(fit_test_residual(p))
                    }
                },
            );
            assert_eq!(record.termination, termination);
            assert_eq!(record.residual_evaluations, evaluations);
            assert_eq!(calls.get(), evaluations);
            assert_eq!(record.encode_failures, last_failure - first_failure + 1);
            let failure = record.last_encode_failure.unwrap();
            assert_eq!(failure.label, 5);
            assert_eq!(failure.error, "synthetic encode failure");
            if termination == FusionFitTermination::LineSearchStalled {
                assert!(result.is_some());
                assert_eq!(record.disposition, FusionFitDisposition::Usable);
                assert_eq!(record.backtracks, 12);
                assert_eq!(failure.stage, FusionFitStage::LineSearch);
                assert_eq!(failure.trial, Some(11));
            } else {
                assert!(result.is_none());
                assert_eq!(record.disposition, FusionFitDisposition::ReturnedNone);
            }
        }
    }

    #[test]
    fn fusion_fit_diagnostics_cholesky_and_nonfinite_values_are_explicit_json() {
        let mut record = FusionFitDiagnostic::new(2, 5);
        assert!(
            fit_pose_residual(
                ORIGIN,
                12.,
                FusionFitPolicy::EarlyAbort,
                Some(&mut record),
                |_| Ok([f64::MAX; 12])
            )
            .is_none()
        );
        assert_eq!(
            record.termination,
            FusionFitTermination::NormalCholeskyFailure
        );
        assert_eq!(record.start_chi2, Some(FusionFitValue::PositiveInfinity));
        assert_eq!(record.final_chi2, record.start_chi2);
        assert_eq!(record.nonfinite_objectives, 1);
        assert_eq!(record.residual_evaluations, 13);
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["start_chi2"]["kind"], "positive_infinity");
        for (value, name) in [
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "negative_infinity"),
            (f64::INFINITY, "positive_infinity"),
        ] {
            let json = serde_json::to_value(FusionFitValue::from(value)).unwrap();
            assert_eq!(json["kind"], name);
            assert!(json.get("value").is_none());
        }
    }

    #[test]
    fn fusion_fit_diagnostics_actual_chart_seam_retains_error_and_labels() {
        let proposal = proposal(parameters(1));
        let tree = tree();
        let wall = Container::new(50., &tree).unwrap();
        let config = OligomerConfig {
            pair_angle_degrees: 181.,
            ..OligomerConfig::default()
        };
        let mut diagnostic = FusionBuildDiagnostics::default();
        OligomerMixture::build_with_fit_diagnostics(
            &proposal,
            &tree,
            &wall,
            &[ORIGIN],
            &[],
            &[
                ORIGIN,
                Pose {
                    orientation: [0., 1., 0., 0.],
                    ..ORIGIN
                },
            ],
            &config,
            FusionFitPolicy::EarlyAbort,
            &mut diagnostic,
        )
        .unwrap();
        check_accounting(&diagnostic, &config);
        assert_eq!(diagnostic.fit_records.len(), 1);
        let record = &diagnostic.fit_records[0];
        assert_eq!((record.first, record.second), (0, 1));
        assert_eq!(
            record.termination,
            FusionFitTermination::InitialEncodeFailure
        );
        assert_eq!(record.start_chi2, None);
        assert_eq!(record.residual_evaluations, 1);
        let failure = record.last_encode_failure.as_ref().unwrap();
        assert_eq!(failure.label, 1);
        assert!(failure.error.contains("Cayley half-turn seam"));
        assert_eq!(diagnostic.centers_checked, 0);
        assert_eq!(diagnostic.core_overlap_calls, 0);
    }
}
