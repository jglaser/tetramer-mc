//! Exact conditional Poisson acceptance after independent endpoint selection.
//! Unknown cells remain in a conservative, disjoint XOR envelope at any depth.
use crate::{
    geometry::{Cell, Coverage, Environment, Placed},
    math::Pose,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GateOptions {
    pub max_cells: usize,
    pub max_depth: u32,
    pub min_width: f64,
}
impl Default for GateOptions {
    fn default() -> Self {
        Self {
            max_cells: 2047,
            max_depth: 14,
            min_width: 0.5,
        }
    }
}
impl GateOptions {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.max_cells > 0 && self.min_width.is_finite() && self.min_width >= 0.,
            "invalid envelope options"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Envelope {
    pub cells: Vec<Cell>,
    pub cumulative: Vec<f64>,
    pub volume: f64,
    pub created: usize,
}
impl Envelope {
    pub fn build(env: &Environment, old: Pose, new: Pose, opts: GateOptions) -> Result<Self> {
        opts.validate()?;
        old.validate()?;
        new.validate()?;
        if env.fixed.is_empty() || old == new {
            return Ok(Self {
                cells: Vec::new(),
                cumulative: Vec::new(),
                volume: 0.,
                created: 0,
            });
        }
        let root = env.tree.bounds(env.rd);
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid root AABB"
        );
        let old = Placed::new(old);
        let new = Placed::new(new);
        let mut queue = VecDeque::from([(root, 0)]);
        let mut cells = Vec::new();
        let mut created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let c = cell.center();
            let r = cell.radius();
            let body = env.tree.classify_ball(c, r, env.rd);
            if body == Coverage::Outside {
                continue;
            }
            let a = env.classify_ball(old.apply(c), r);
            let b = env.classify_ball(new.apply(c), r);
            if a == b && a != Coverage::Unknown {
                continue;
            }
            let known =
                body == Coverage::Inside && a != Coverage::Unknown && b != Coverage::Unknown;
            if known
                || depth >= opts.max_depth
                || created + 2 > opts.max_cells
                || cell.hi[cell.longest()] - cell.lo[cell.longest()] <= opts.min_width
            {
                cells.push(cell);
            } else if let Some((left, right)) = cell.split() {
                queue.push_back((left, depth + 1));
                queue.push_back((right, depth + 1));
                created += 2;
            } else {
                cells.push(cell);
            }
        }
        let mut cumulative = Vec::new();
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
                return Ok(Self {
                    cells: vec![root],
                    cumulative: vec![root.volume()],
                    volume: root.volume(),
                    created,
                });
            }
            volume = next;
            cumulative.push(volume);
        }
        Ok(Self {
            cells,
            cumulative,
            volume,
            created,
        })
    }
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct GateResult {
    pub gained: u64,
    pub lost: u64,
    pub raw_points: u64,
    pub retained_points: u64,
    pub log_weight: f64,
    pub envelope_volume: f64,
    pub retained_cells: usize,
    pub created_cells: usize,
}

pub fn sample_with_envelope(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    new: Pose,
    lambda: f64,
    z: f64,
    envelope: &Envelope,
) -> Result<GateResult> {
    ensure!(
        lambda.is_finite() && lambda > 0. && z.is_finite() && z >= 0. && (lambda + z).is_finite(),
        "invalid bath intensity"
    );
    old.validate()?;
    new.validate()?;
    let mut result = GateResult {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        ..Default::default()
    };
    if z == 0. || envelope.volume == 0. {
        return Ok(result);
    }
    let mean = (lambda + z) * envelope.volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    result.raw_points = number;
    let old = Placed::new(old);
    let new = Placed::new(new);
    for _ in 0..number {
        let target = rng.random::<f64>() * envelope.volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let p =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        if !env.tree.contains(p, env.rd) {
            continue;
        }
        let a = env.contains(old.apply(p));
        let b = env.contains(new.apply(p));
        if a && !b {
            result.lost += 1;
        }
        if b && !a && rng.random::<f64>() < lambda / (lambda + z) {
            result.gained += 1;
        }
    }
    result.retained_points = result.gained + result.lost;
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    result.log_weight = coefficient * (result.gained as f64 - result.lost as f64);
    Ok(result)
}

pub fn sample(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    new: Pose,
    lambda: f64,
    z: f64,
    opts: GateOptions,
) -> Result<GateResult> {
    ensure!(
        lambda.is_finite() && lambda > 0. && z.is_finite() && z >= 0. && (lambda + z).is_finite(),
        "invalid bath intensity"
    );
    old.validate()?;
    new.validate()?;
    opts.validate()?;
    if z == 0. {
        return Ok(GateResult::default());
    }
    let envelope = Envelope::build(env, old, new, opts)?;
    sample_with_envelope(rng, env, old, new, lambda, z, &envelope)
}
