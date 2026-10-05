//! Positive, unbiased Boltzmann weights for ideal many-body depletion.
//!
//! For a moving exclusion region A and the UNION B of fixed exclusions, let
//! C=|A intersection B|. A disjoint cell decomposition finds a certified inner
//! volume L and an uncertain envelope U covering the remainder. A Poisson
//! process of intensity lambda on U, thinned by exact leaf membership, gives
//! K~Poisson(lambda*(C-L)). Therefore
//!
//! ```text
//! W = exp(z*L) * (1 + z/lambda)^K,
//! E W = exp(z*C),
//! E W^2 / (E W)^2 = exp(z*z*(C-L)/lambda).
//! ```
//!
//! This estimates an absolute overlap weight, not an acceptance probability.
//! In particular, averaging log(W) would NOT estimate log(E W). Construction
//! budgets change cost and variance, never the mathematical expectation.
//! "Certified" refers to the guarded FP64 geometry predicates used elsewhere
//! in this crate, not to formal outward-rounded interval arithmetic. Boundary
//! membership uses the same closed exclusion spheres as Environment::contains.
use crate::{
    depletion::GateOptions,
    geometry::{Cell, Coverage, Environment, Placed},
    math::{Pose, norm},
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Immutable geometric preprocessing for repeated independent weight samples.
/// Reuse only with the same environment and pose that constructed it.
#[derive(Clone, Debug)]
pub struct OverlapEnvelope {
    /// Disjoint cells requiring exact point membership tests.
    pub cells: Vec<Cell>,
    pub cumulative: Vec<f64>,
    pub lower_volume: f64,
    pub uncertain_volume: f64,
    pub created: usize,
    pub certified_cells: usize,
    pose: Pose,
}

impl OverlapEnvelope {
    pub fn upper_volume(&self) -> f64 {
        self.lower_volume + self.uncertain_volume
    }

    pub fn build(env: &Environment, pose: Pose, opts: GateOptions) -> Result<Self> {
        opts.validate()?;
        pose.validate()?;
        ensure!(
            env.rd.is_finite() && env.rd >= 0.,
            "invalid depletant radius"
        );
        let mut envelope = Self {
            cells: Vec::new(),
            cumulative: Vec::new(),
            lower_volume: 0.,
            uncertain_volume: 0.,
            created: 0,
            certified_cells: 0,
            pose,
        };
        if env.fixed.is_empty() {
            return Ok(envelope);
        }
        let root = env.tree.bounds(env.rd);
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid root AABB"
        );
        let moving = Placed::new(pose);
        let mut queue = VecDeque::from([(root, 0)]);
        envelope.created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let c = cell.center();
            let r = cell.radius();
            let body = env.tree.classify_ball(c, r, env.rd);
            if body == Coverage::Outside {
                continue;
            }
            // Guard the moving-body transformation in addition to the guards
            // supplied by Environment when converting to each fixed body.
            let guard = 1024. * f64::EPSILON * (1. + norm(c) + norm(moving.position) + r);
            let fixed = env.classify_ball(moving.apply(c), r + guard);
            if fixed == Coverage::Outside {
                continue;
            }
            if body == Coverage::Inside && fixed == Coverage::Inside {
                let next = envelope.lower_volume + cell.volume();
                if !next.is_finite() || next <= envelope.lower_volume {
                    return Ok(Self::root_fallback(root, pose, envelope.created));
                }
                envelope.lower_volume = next;
                envelope.certified_cells += 1;
            } else if depth >= opts.max_depth
                || envelope.created > opts.max_cells.saturating_sub(2)
                || cell.hi[cell.longest()] - cell.lo[cell.longest()] <= opts.min_width
            {
                envelope.cells.push(cell);
            } else if let Some((left, right)) = cell.split() {
                queue.push_back((left, depth + 1));
                queue.push_back((right, depth + 1));
                envelope.created += 2;
            } else {
                envelope.cells.push(cell);
            }
        }
        for cell in &envelope.cells {
            let next = envelope.uncertain_volume + cell.volume();
            if !next.is_finite() || next <= envelope.uncertain_volume {
                return Ok(Self::root_fallback(root, pose, envelope.created));
            }
            envelope.uncertain_volume = next;
            envelope.cumulative.push(next);
        }
        ensure!(
            envelope.upper_volume().is_finite(),
            "nonfinite overlap bound"
        );
        Ok(envelope)
    }

    fn root_fallback(root: Cell, pose: Pose, created: usize) -> Self {
        Self {
            cells: vec![root],
            cumulative: vec![root.volume()],
            lower_volume: 0.,
            uncertain_volume: root.volume(),
            created,
            certified_cells: 0,
            pose,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct OverlapWeight {
    pub log_weight: f64,
    pub lower_volume: f64,
    pub upper_volume: f64,
    pub uncertain_volume: f64,
    pub raw_points: u64,
    pub overlap_points: u64,
    pub retained_cells: usize,
    pub created_cells: usize,
    pub certified_cells: usize,
}

/// Remaining work allowed for one complete cloud. Callers pass the minimum
/// of their per-cloud cap and every remaining outer/campaign allowance.
/// Both count limits admit the WHOLE sampled Poisson count before point work;
/// processed_points may be less than the planned count only after a fatal error.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudLimits {
    pub raw_points: u64,
    pub processed_points: u64,
    /// Positive number of completed membership tests between callbacks.
    pub callback_interval: u64,
}

/// Durable caller-owned progress, including failures before any point work.
/// A sampled count is Some(n), including n=0; None means no Poisson draw.
/// log_weight is published only after the final callback succeeds. No partial
/// progress record supplies a Boltzmann estimate or an MCMC rejection.
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct CloudProgress {
    pub begun: bool,
    pub planned_points: Option<u64>,
    pub processed_points: u64,
    pub overlap_points: u64,
    pub complete: bool,
    pub log_weight: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CloudEvent {
    Begun,
    /// Emitted before cap checks and before any cell/point random numbers.
    CountDrawn,
    Progress,
    /// All arithmetic is done, but no usable result has yet been published.
    Finishing,
}

/// Bounded counterpart of sample_with_envelope, with identical RNG/arithmetic
/// whenever it completes. This is an absolute-weight estimator, not a gate.
///
/// Supply fresh default progress and an observer that journals begun/count
/// events and checks the caller's CPU/wall budget. Any observer or count-limit
/// failure is fatal; retain progress and stop, without retries or replacement.
/// On either result, charge planned_points.unwrap_or(0) and processed_points
/// exactly ONCE to caller-owned totals (never once per callback).
///
/// The caller must bind the immutable environment to the envelope and pose;
/// the envelope checks its pose but does not authenticate its environment.
#[allow(clippy::too_many_arguments)]
pub fn sample_with_envelope_bounded<F>(
    rng: &mut StdRng,
    env: &Environment,
    pose: Pose,
    lambda: f64,
    z: f64,
    envelope: &OverlapEnvelope,
    limits: CloudLimits,
    progress: &mut CloudProgress,
    mut observer: F,
) -> Result<OverlapWeight>
where
    F: FnMut(CloudEvent, &CloudProgress) -> Result<()>,
{
    ensure!(limits.callback_interval > 0, "zero cloud callback interval");
    sample_with_envelope_observed(
        rng,
        env,
        pose,
        lambda,
        z,
        envelope,
        limits.callback_interval,
        progress,
        &mut |event, state| {
            // Publish the complete sampled count even when it cannot be admitted.
            observer(event, state)?;
            if event == CloudEvent::CountDrawn {
                let number = state.planned_points.expect("count event has a count");
                ensure!(
                    number <= limits.raw_points,
                    "fatal planned raw-point budget exceeded"
                );
                ensure!(
                    number <= limits.processed_points,
                    "fatal planned processed-point budget exceeded"
                );
            }
            Ok(())
        },
    )
}

/// Draw an independent positive estimate of exp(z*C). The environment must be
/// the same as used to build `envelope`; no hard-core validity is imposed here.
/// The caller multiplies by its hard-core and domain indicators separately.
pub fn sample_with_envelope(
    rng: &mut StdRng,
    env: &Environment,
    pose: Pose,
    lambda: f64,
    z: f64,
    envelope: &OverlapEnvelope,
) -> Result<OverlapWeight> {
    sample_with_envelope_observed(
        rng,
        env,
        pose,
        lambda,
        z,
        envelope,
        1024,
        &mut CloudProgress::default(),
        &mut |_, _| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_with_envelope_observed<F>(
    rng: &mut StdRng,
    env: &Environment,
    pose: Pose,
    lambda: f64,
    z: f64,
    envelope: &OverlapEnvelope,
    callback_interval: u64,
    progress: &mut CloudProgress,
    observer: &mut F,
) -> Result<OverlapWeight>
where
    F: FnMut(CloudEvent, &CloudProgress) -> Result<()>,
{
    ensure!(
        *progress == CloudProgress::default(),
        "cloud progress must be fresh"
    );
    progress.begun = true;
    observer(CloudEvent::Begun, progress)?;
    validate_intensity(lambda, z)?;
    pose.validate()?;
    ensure!(
        pose == envelope.pose,
        "overlap envelope belongs to a different pose"
    );
    let mut result = OverlapWeight {
        lower_volume: envelope.lower_volume,
        upper_volume: envelope.upper_volume(),
        uncertain_volume: envelope.uncertain_volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        certified_cells: envelope.certified_cells,
        ..Default::default()
    };
    if z == 0. {
        return finish_cloud(result, progress, observer);
    }
    result.log_weight = z * envelope.lower_volume;
    ensure!(
        result.log_weight.is_finite(),
        "nonfinite deterministic log weight"
    );
    if envelope.uncertain_volume == 0. {
        return finish_cloud(result, progress, observer);
    }
    let mean = lambda * envelope.uncertain_volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    if mean == 0. {
        // Silently treating an underflowed intensity as exactly zero would
        // discard a nonempty unknown overlap and lose unbiasedness.
        anyhow::bail!("underflowed Poisson mean");
    }
    result.raw_points = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    progress.planned_points = Some(result.raw_points);
    observer(CloudEvent::CountDrawn, progress)?;
    let moving = Placed::new(pose);
    for index in 0..result.raw_points {
        let target = rng.random::<f64>() * envelope.uncertain_volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let p =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        if env.tree.contains(p, env.rd) && env.contains(moving.apply(p)) {
            result.overlap_points += 1;
        }
        progress.processed_points = index + 1;
        progress.overlap_points = result.overlap_points;
        if progress.processed_points % callback_interval == 0 {
            observer(CloudEvent::Progress, progress)?;
        }
    }
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    result.log_weight += result.overlap_points as f64 * coefficient;
    ensure!(
        result.log_weight.is_finite(),
        "nonfinite sampled log weight"
    );
    finish_cloud(result, progress, observer)
}

fn finish_cloud<F>(
    result: OverlapWeight,
    progress: &mut CloudProgress,
    observer: &mut F,
) -> Result<OverlapWeight>
where
    F: FnMut(CloudEvent, &CloudProgress) -> Result<()>,
{
    observer(CloudEvent::Finishing, progress)?;
    progress.log_weight = Some(result.log_weight);
    progress.complete = true;
    Ok(result)
}

pub fn sample(
    rng: &mut StdRng,
    env: &Environment,
    pose: Pose,
    lambda: f64,
    z: f64,
    opts: GateOptions,
) -> Result<OverlapWeight> {
    validate_intensity(lambda, z)?;
    pose.validate()?;
    opts.validate()?;
    if z == 0. {
        return Ok(OverlapWeight::default());
    }
    let envelope = OverlapEnvelope::build(env, pose, opts)?;
    sample_with_envelope(rng, env, pose, lambda, z, &envelope)
}

fn validate_intensity(lambda: f64, z: f64) -> Result<()> {
    ensure!(
        lambda.is_finite()
            && lambda >= 0.
            && (z == 0. || lambda > 0.)
            && z.is_finite()
            && z >= 0.
            && (lambda + z).is_finite(),
        "invalid bath intensity"
    );
    Ok(())
}
