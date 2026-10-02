//! Proposal-independent bath gate for arbitrary selected-body endpoints.
//!
//! Unlike `RigidSubset`, selected relative poses may change. All predicates use
//! the WORLD-coordinate Boolean unions U_old = spectators union selected_old
//! and U_new = spectators union selected_new. Gained solvent is U_old \ U_new;
//! lost solvent is U_new \ U_old. Their counts have rates lambda and lambda+z,
//! respectively, and the log factor is (gained-lost)*log1p(z/lambda).
//!
//! The endpoint-symmetric cover includes changes of the selected union's own
//! volume, even with no spectators. The bath permeates the protein-only wall.
//! This module neither proposes nor accepts moves, selects members, retries,
//! nor supplies proposal/map/rate corrections. It supports nonperiodic poses
//! of identical rigid bodies; relative motion between those bodies is free.
//! Exactness refers to the union/Poisson construction, not certified FP64.
use crate::{
    depletion::{Envelope, GateOptions, GateResult},
    geometry::{Cell, Coverage, Environment, Placed, SphereTree},
    math::{Pose, Vec3, norm, sub},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use std::collections::{BTreeSet, VecDeque};

pub struct FlexibleSubset<'a> {
    tree: &'a SphereTree,
    members: Vec<usize>,
    proposed: Vec<Pose>,
    old: Vec<Placed>,
    new: Vec<Placed>,
    environment: Environment<'a>,
    rd: f64,
    root: Cell,
    identity: bool,
}

impl<'a> FlexibleSubset<'a> {
    /// `proposed[i]` replaces `state[members[i]]`; spectators remain fixed.
    /// Numeric inputs are checked, but old/intermediate hard validity is not
    /// required to define the positive bath density. Call `hard_valid` on the
    /// final endpoint before a physical acceptance decision.
    pub fn new(
        tree: &'a SphereTree,
        state: &[Pose],
        members: &[usize],
        proposed: &[Pose],
        rd: f64,
    ) -> Result<Self> {
        ensure!(!members.is_empty(), "empty flexible subset");
        ensure!(
            members.len() == proposed.len(),
            "proposed/member count differs"
        );
        ensure!(
            members.iter().all(|&i| i < state.len()),
            "member index out of range"
        );
        ensure!(
            members.iter().copied().collect::<BTreeSet<_>>().len() == members.len(),
            "duplicate flexible subset member"
        );
        ensure!(rd.is_finite() && rd >= 0., "invalid depletant radius");
        for pose in state.iter().chain(proposed) {
            pose.validate()?;
        }
        let old: Vec<_> = members.iter().map(|&i| Placed::new(state[i])).collect();
        let new: Vec<_> = proposed.iter().copied().map(Placed::new).collect();
        let own = tree.bounds(rd);
        let center = own.center();
        let half: Vec3 = std::array::from_fn(|k| (own.hi[k] - own.lo[k]) * 0.5);
        let mut root = Cell {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        };
        // Each endpoint contributes its own guarded world AABB. No summation
        // depends on which endpoint is called old: swapping them gives the same
        // root, traversal order, refinement decisions, and cumulative volumes.
        for placed in old.iter().chain(&new) {
            let c = placed.apply(center);
            let h: Vec3 = std::array::from_fn(|k| {
                (0..3).map(|j| placed.rotation[k][j].abs() * half[j]).sum()
            });
            let guard = 2048. * f64::EPSILON * (1. + norm(c) + norm(h));
            for k in 0..3 {
                root.lo[k] = root.lo[k].min(c[k] - h[k] - guard);
                root.hi[k] = root.hi[k].max(c[k] + h[k] + guard);
            }
        }
        ensure!(
            root.lo.iter().chain(&root.hi).all(|v| v.is_finite())
                && root.volume().is_finite()
                && root.volume() > 0.,
            "invalid flexible subset world bounds"
        );
        // Keep every spectator. This simple reference has no neighbor shortlist
        // whose omission could erase shielding or proposed hard collisions.
        let labels: Vec<_> = (0..state.len())
            .filter(|i| !members.contains(i))
            .map(|i| (i, [0; 3]))
            .collect();
        let fixed = labels.iter().map(|&(i, _)| Placed::new(state[i])).collect();
        let environment = Environment {
            tree,
            fixed,
            labels,
            rd,
        };
        Ok(Self {
            tree,
            members: members.to_vec(),
            proposed: proposed.to_vec(),
            old,
            new,
            environment,
            rd,
            root,
            identity: members.iter().zip(proposed).all(|(&i, p)| state[i] == *p),
        })
    }

    pub fn members(&self) -> &[usize] {
        &self.members
    }
    /// Returned in `members()` order, not full configuration order.
    pub fn proposed_poses(&self) -> &[Pose] {
        &self.proposed
    }
    pub fn spectator_count(&self) -> usize {
        self.environment.fixed.len()
    }
    pub fn root_bounds(&self) -> Cell {
        self.root
    }

    /// Proposed selected-pair, spectator and full atomic-wall checks. The fixed
    /// spectators' mutual validity is the caller's state invariant. The wall is
    /// never used by the bath membership or Poisson cover.
    pub fn hard_valid(&self, wall: Option<&Container>, wall_center: Vec3) -> bool {
        if wall.is_some() && !wall_center.iter().all(|v| v.is_finite()) {
            return false;
        }
        for i in 0..self.new.len() {
            for j in 0..i {
                if self.tree.overlaps(&self.new[i], &self.new[j]) {
                    return false;
                }
            }
        }
        self.proposed.iter().all(|&pose| {
            if let Some(wall) = wall {
                let shifted = Pose {
                    position: sub(pose.position, wall_center),
                    ..pose
                };
                if !wall.contains(shifted) {
                    return false;
                }
            }
            self.environment.hard_valid(pose)
        })
    }

    /// Full (old,new) EXCLUSION membership at a world point. A spectator-covered
    /// point returns (true,true), including all triple/higher overlaps. Gained
    /// solvent therefore corresponds to (true,false), not (false,true).
    pub fn excluded_indicators(&self, point: Vec3) -> (bool, bool) {
        if self.environment.contains(point) {
            return (true, true);
        }
        let contains = |poses: &[Placed]| {
            poses
                .iter()
                .any(|p| self.tree.contains(p.unapply(point), self.rd))
        };
        (contains(&self.old), contains(&self.new))
    }

    fn classify_selected(&self, poses: &[Placed], center: Vec3, radius: f64) -> Coverage {
        let mut result = Coverage::Outside;
        for pose in poses {
            // Enlarging the query ball makes BOTH Inside and Outside proofs
            // harder, so roundoff guard bands retain uncertainty conservatively.
            let guard = 2048.
                * f64::EPSILON
                * (1. + norm(center) + norm(pose.position) + self.tree.bound + self.rd + radius);
            match self
                .tree
                .classify_ball(pose.unapply(center), radius + guard, self.rd)
            {
                Coverage::Inside => return Coverage::Inside,
                Coverage::Unknown => result = Coverage::Unknown,
                Coverage::Outside => (),
            }
        }
        result
    }

    /// Disjoint adaptive world cover of the whole-union symmetric difference.
    /// Every unresolved cell is retained when any budget is exhausted.
    pub fn envelope(&self, opts: GateOptions) -> Result<Envelope> {
        opts.validate()?;
        if self.identity {
            return Ok(Envelope {
                cells: vec![],
                cumulative: vec![],
                volume: 0.,
                created: 0,
            });
        }
        let union = |a, b| match (a, b) {
            (Coverage::Inside, _) | (_, Coverage::Inside) => Coverage::Inside,
            (Coverage::Outside, Coverage::Outside) => Coverage::Outside,
            _ => Coverage::Unknown,
        };
        let mut queue = VecDeque::from([(self.root, 0)]);
        let mut cells = Vec::new();
        let mut created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let c = cell.center();
            let r = cell.radius();
            let spectators = self.environment.classify_ball(c, r);
            if spectators == Coverage::Inside {
                continue;
            }
            let a = union(spectators, self.classify_selected(&self.old, c, r));
            let b = union(spectators, self.classify_selected(&self.new, c, r));
            if a == b && a != Coverage::Unknown {
                continue;
            }
            let known = a != Coverage::Unknown && b != Coverage::Unknown;
            if known
                || depth >= opts.max_depth
                || opts.max_cells.saturating_sub(created) < 2
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
        let mut cumulative = Vec::with_capacity(cells.len());
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
                // A representability failure enlarges the cover to its original
                // symmetric box; it never silently drops a small-volume cell.
                return Ok(Envelope {
                    cells: vec![self.root],
                    cumulative: vec![self.root.volume()],
                    volume: self.root.volume(),
                    created,
                });
            }
            volume = next;
            cumulative.push(volume);
        }
        Ok(Envelope {
            cells,
            cumulative,
            volume,
            created,
        })
    }

    /// One fresh auxiliary draw, after independently choosing the endpoints.
    /// Sampling lambda+z on the cover and thinning only gained-solvent points
    /// gives the exact two Poisson count laws. Hard-invalid intermediate states
    /// still define this bath factor; physical endpoint checks belong to caller.
    pub fn sample(
        &self,
        rng: &mut StdRng,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<GateResult> {
        ensure!(
            lambda.is_finite()
                && lambda > 0.
                && z.is_finite()
                && z >= 0.
                && (lambda + z).is_finite(),
            "invalid bath intensity"
        );
        opts.validate()?;
        if z == 0. {
            return Ok(GateResult::default());
        }
        let envelope = self.envelope(opts)?;
        let mut result = GateResult {
            envelope_volume: envelope.volume,
            retained_cells: envelope.cells.len(),
            created_cells: envelope.created,
            ..Default::default()
        };
        if envelope.volume == 0. {
            return Ok(result);
        }
        let mean = (lambda + z) * envelope.volume;
        ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
        let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
        result.raw_points = number;
        for _ in 0..number {
            let target = rng.random::<f64>() * envelope.volume;
            let k = envelope
                .cumulative
                .partition_point(|&v| v <= target)
                .min(envelope.cells.len() - 1);
            let cell = envelope.cells[k];
            let point = std::array::from_fn(|j| {
                cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j])
            });
            let (old, new) = self.excluded_indicators(point);
            if new && !old {
                result.lost += 1;
            }
            if old && !new && rng.random::<f64>() < lambda / (lambda + z) {
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
}
