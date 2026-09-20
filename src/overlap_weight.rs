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
        return Ok(result);
    }
    result.log_weight = z * envelope.lower_volume;
    ensure!(
        result.log_weight.is_finite(),
        "nonfinite deterministic log weight"
    );
    if envelope.uncertain_volume == 0. {
        return Ok(result);
    }
    let mean = lambda * envelope.uncertain_volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    if mean == 0. {
        // Silently treating an underflowed intensity as exactly zero would
        // discard a nonempty unknown overlap and lose unbiasedness.
        anyhow::bail!("underflowed Poisson mean");
    }
    result.raw_points = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    let moving = Placed::new(pose);
    for _ in 0..result.raw_points {
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
