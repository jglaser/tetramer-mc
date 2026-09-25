//! Frozen-endpoint volume diagnostics for geometric cluster recruitment.
//!
//! These independent, fixed-size Monte Carlo estimates are diagnostics only.
//! Inserting their noisy volumes into an exponential bond probability would
//! generally bias a Markov chain. This module does not modify a configuration
//! or participate in production acceptance.
use crate::{
    geometry::{Cell, Coverage, Environment, Placed, SphereTree},
    math::{Vec3, norm, sub},
};
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::collections::VecDeque;

/// Every count has the same unconditional denominator `draws`, including
/// samples outside every counted overlap region. No hit-conditioned denominator
/// is used.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Population {
    pub seed: u64,
    pub draws: usize,
    pub old_hits: usize,
    pub cross_hits: usize,
    pub lost_hits: usize,
    pub reverse_hits: usize,
    pub shielded_lost_hits: usize,
    pub triple_old_hits: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PairEstimate {
    pub envelope_volume: f64,
    pub retained_cells: usize,
    pub created_cells: usize,
    pub populations: Vec<Population>,
}

struct PairEnvelope {
    cells: Vec<Cell>,
    cumulative: Vec<f64>,
    volume: f64,
    created: usize,
}

impl PairEnvelope {
    /// Cover A intersect (B union D) using conservative ball certificates.
    /// The body-i AABB and every partition remain in A's body coordinates.
    fn build(
        tree: &SphereTree,
        a: Placed,
        b: Placed,
        d: Placed,
        rd: f64,
        max_cells: usize,
    ) -> Result<Self> {
        let root = tree.bounds(rd);
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid root AABB"
        );
        let make_env = |body| Environment {
            tree,
            fixed: vec![body],
            labels: vec![],
            rd,
        };
        let b_env = make_env(b);
        let d_env = make_env(d);
        let mut queue = VecDeque::from([(root, 0_u32)]);
        let mut cells = Vec::new();
        let mut created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let center = cell.center();
            let radius = cell.radius();
            let body = tree.classify_ball(center, radius, rd);
            if body == Coverage::Outside {
                continue;
            }
            let world = a.apply(center);
            // Environment::classify_ball also guards the frame transform.
            let b_status = b_env.classify_ball(world, radius);
            let d_status = d_env.classify_ball(world, radius);
            if b_status == Coverage::Outside && d_status == Coverage::Outside {
                continue;
            }
            let known_inside = body == Coverage::Inside
                && (b_status == Coverage::Inside || d_status == Coverage::Inside);
            let axis = cell.longest();
            if known_inside
                || depth >= 18
                || created > max_cells.saturating_sub(2)
                || cell.hi[axis] - cell.lo[axis] <= 0.25
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
        let mut cumulative = Vec::with_capacity(cells.len());
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
                // A rare summation failure falls back to the original cover;
                // it must never silently remove a positive-volume cell.
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

    fn sample(&self, rng: &mut StdRng) -> Vec3 {
        let coordinate = rng.random::<f64>() * self.volume;
        let index = self
            .cumulative
            .partition_point(|&v| v <= coordinate)
            .min(self.cells.len() - 1);
        let cell = self.cells[index];
        std::array::from_fn(|k| cell.lo[k] + rng.random::<f64>() * (cell.hi[k] - cell.lo[k]))
    }
}

/// Measure pair overlap and pointwise lost-overlap volumes for a fixed common
/// isometry. Let A=old[i], B=old[j], C=shadow[i], D=shadow[j]. We count:
///
/// * old: A intersect B;
/// * cross: A intersect D (isometric to C intersect B for an involution);
/// * lost: A intersect B minus (C union D);
/// * reverse: A intersect D minus (B union C);
/// * shielded lost: lost and outside every other shadow body;
/// * triple old: old and inside at least one additional old body.
///
/// The caller must supply matching old/shadow configurations obtained with one
/// common involution to interpret these as GCA forward/reverse quantities.
/// No isometry assumption is needed for the raw set-volume estimates.
#[allow(clippy::too_many_arguments)]
pub fn estimate_pair(
    tree: &SphereTree,
    old: &[Placed],
    shadow: &[Placed],
    i: usize,
    j: usize,
    rd: f64,
    draws_per_population: usize,
    populations: usize,
    seed: u64,
    max_cells: usize,
) -> Result<PairEstimate> {
    ensure!(
        old.len() == shadow.len(),
        "old/shadow configuration lengths differ"
    );
    ensure!(
        i < old.len() && j < old.len() && i != j,
        "invalid distinct pair indices"
    );
    ensure!(rd.is_finite() && rd >= 0., "invalid depletant radius");
    ensure!(
        draws_per_population > 0 && populations > 0,
        "positive fixed sampling allocations required"
    );
    ensure!(max_cells > 0, "positive cell budget required");
    ensure!(
        old.iter().chain(shadow).all(|p| p
            .position
            .iter()
            .chain(p.rotation.iter().flatten())
            .all(|x| x.is_finite())),
        "nonfinite placed geometry"
    );
    let [a, b, c, d] = [old[i], old[j], shadow[i], shadow[j]];
    let envelope = PairEnvelope::build(tree, a, b, d, rd, max_cells)?;
    // Only these bodies can cover a point in A. The generous outward guard
    // also includes the positions used by the rigid-frame transforms.
    let near_a = |body: &&Placed| {
        let reach = 2. * (tree.bound + rd);
        let guard = 4096. * f64::EPSILON * (1. + reach + norm(a.position) + norm(body.position));
        norm(sub(body.position, a.position)) <= reach + guard
    };
    let old_others: Vec<_> = old
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != i && *k != j)
        .map(|(_, p)| p)
        .filter(near_a)
        .collect();
    let shadow_others: Vec<_> = shadow
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != i && *k != j)
        .map(|(_, p)| p)
        .filter(near_a)
        .collect();
    let contains = |body: &Placed, world: Vec3| tree.contains(body.unapply(world), rd);
    let mut seed_rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::with_capacity(populations);
    for _ in 0..populations {
        let population_seed = seed_rng.random::<u64>();
        let mut rng = StdRng::seed_from_u64(population_seed);
        let mut result = Population {
            seed: population_seed,
            draws: draws_per_population,
            old_hits: 0,
            cross_hits: 0,
            lost_hits: 0,
            reverse_hits: 0,
            shielded_lost_hits: 0,
            triple_old_hits: 0,
        };
        if envelope.volume > 0. {
            for _ in 0..draws_per_population {
                let local = envelope.sample(&mut rng);
                if !tree.contains(local, rd) {
                    continue;
                }
                let world = a.apply(local);
                let in_b = contains(&b, world);
                let in_d = contains(&d, world);
                if !in_b && !in_d {
                    continue;
                }
                // C matters only for exclusive old/cross hits.
                let in_c = (in_b != in_d) && contains(&c, world);
                result.old_hits += usize::from(in_b);
                result.cross_hits += usize::from(in_d);
                let lost = in_b && !in_c && !in_d;
                result.lost_hits += usize::from(lost);
                result.reverse_hits += usize::from(in_d && !in_b && !in_c);
                if lost && !shadow_others.iter().any(|p| contains(p, world)) {
                    result.shielded_lost_hits += 1;
                }
                if in_b && old_others.iter().any(|p| contains(p, world)) {
                    result.triple_old_hits += 1;
                }
            }
        }
        // An empty certified envelope has exactly zero mass. Retain the full
        // declared denominator and seed even though no point draws are needed.
        results.push(result);
    }
    Ok(PairEstimate {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        populations: results,
    })
}
