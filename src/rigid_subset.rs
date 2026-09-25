//! Exact implicit depletion for a rigidly carried subset of identical bodies.
//!
//! Coordinates are expressed in the old handle's frame. The moving exclusion
//! region is the BOOLEAN UNION of its member shapes in that frame; it is never
//! counted once per member. Since a common proper isometry preserves this union's
//! volume, only its overlap with the fixed environment changes. The bath crosses
//! the protein-only spherical wall, just as in `spherical`.
//!
//! This module does not select subsets, choose proposal maps, or accept a trial.
//! It returns the same conditional gained/lost-count likelihood ratio as the
//! single-body gate. Its log weight must be combined with all map/auxiliary/bias
//! corrections before the caller's Metropolis decision. It supports ordinary
//! nonperiodic poses only; periodic image enumeration is deliberately absent.
use crate::{
    depletion::{Envelope, GateOptions, GateResult},
    geometry::{Cell, Coverage, Environment, Placed, SphereTree},
    math::{Pose, Vec3, add, matmul, matvec, norm, quaternion, rotation, sub, transpose},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use std::collections::{BTreeSet, VecDeque};

fn validate_members(state: &[Pose], members: &[usize], handle: usize) -> Result<()> {
    ensure!(!members.is_empty(), "empty rigid subset");
    ensure!(
        members.iter().all(|&i| i < state.len()),
        "rigid subset index out of range"
    );
    ensure!(
        members.contains(&handle),
        "rigid subset does not contain its handle"
    );
    ensure!(
        members.iter().copied().collect::<BTreeSet<_>>().len() == members.len(),
        "duplicate rigid subset member"
    );
    for pose in state {
        pose.validate()?;
    }
    Ok(())
}

/// Carry each selected member with g'_h g_h^-1, in exactly `members` order.
/// Untouched bodies are omitted. Internal relative poses are invariant in exact
/// arithmetic; floating-point quaternion transformations have the same obligation
/// as the existing single-body geometry. An identity move is bitwise unchanged.
pub fn transport_members(
    state: &[Pose],
    members: &[usize],
    handle: usize,
    proposed_handle: Pose,
) -> Result<Vec<Pose>> {
    validate_members(state, members, handle)?;
    proposed_handle.validate()?;
    let old = state[handle];
    if proposed_handle == old {
        return Ok(members.iter().map(|&i| state[i]).collect());
    }
    let delta = matmul(
        rotation(proposed_handle.orientation),
        transpose(rotation(old.orientation)),
    );
    let result = members
        .iter()
        .map(|&i| {
            if i == handle {
                proposed_handle
            } else {
                Pose {
                    position: add(
                        proposed_handle.position,
                        matvec(delta, sub(state[i].position, old.position)),
                    ),
                    orientation: quaternion(matmul(delta, rotation(state[i].orientation))),
                }
            }
        })
        .collect::<Vec<_>>();
    for pose in &result {
        pose.validate()?;
    }
    Ok(result)
}

/// Whole candidate configuration convenience wrapper; all spectators are copied.
pub fn carry(
    state: &[Pose],
    members: &[usize],
    handle: usize,
    proposed_handle: Pose,
) -> Result<Vec<Pose>> {
    let selected = transport_members(state, members, handle, proposed_handle)?;
    let mut candidate = state.to_vec();
    for (&i, p) in members.iter().zip(selected) {
        candidate[i] = p;
    }
    Ok(candidate)
}

/// One rigid trial. Member geometry is held in the old handle frame. The fixed
/// environment includes every body that can intersect any old/new member's
/// excluded volume, using an outward-guarded center bound only for pruning.
/// Exact membership is delegated to the original sphere-union tree.
pub struct RigidSubset<'a> {
    tree: &'a SphereTree,
    members: Vec<usize>,
    proposed: Vec<Pose>,
    body: Vec<Placed>,
    environment: Environment<'a>,
    old: Placed,
    new: Placed,
    identity: bool,
    rd: f64,
    root: Cell,
}
impl<'a> RigidSubset<'a> {
    pub fn new(
        tree: &'a SphereTree,
        state: &[Pose],
        members: &[usize],
        handle: usize,
        proposed_handle: Pose,
        rd: f64,
    ) -> Result<Self> {
        validate_members(state, members, handle)?;
        ensure!(rd.is_finite() && rd >= 0., "invalid depletant radius");
        let proposed = transport_members(state, members, handle, proposed_handle)?;
        let old = Placed::new(state[handle]);
        let new = Placed::new(proposed_handle);
        let inverse_old = transpose(old.rotation);
        let body = members
            .iter()
            .map(|&i| {
                if i == handle {
                    Placed::new(Pose {
                        position: [0.; 3],
                        orientation: [1., 0., 0., 0.],
                    })
                } else {
                    Placed::new(Pose {
                        position: old.unapply(state[i].position),
                        orientation: quaternion(matmul(
                            inverse_old,
                            rotation(state[i].orientation),
                        )),
                    })
                }
            })
            .collect::<Vec<_>>();
        let own_bounds = tree.bounds(rd);
        let own_center = own_bounds.center();
        let own_half: Vec3 = std::array::from_fn(|k| (own_bounds.hi[k] - own_bounds.lo[k]) * 0.5);
        let mut root = Cell {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        };
        for member in &body {
            let center = member.apply(own_center);
            let half: Vec3 = std::array::from_fn(|k| {
                (0..3)
                    .map(|j| member.rotation[k][j].abs() * own_half[j])
                    .sum()
            });
            let guard = 2048. * f64::EPSILON * (1. + norm(center) + norm(half));
            for k in 0..3 {
                root.lo[k] = root.lo[k].min(center[k] - half[k] - guard);
                root.hi[k] = root.hi[k].max(center[k] + half[k] + guard);
            }
        }
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid rigid subset bounds"
        );
        let reach = 2. * (tree.bound + rd);
        ensure!(reach.is_finite(), "invalid rigid subset exclusion reach");
        let labels = state
            .iter()
            .enumerate()
            .filter_map(|(j, spectator)| {
                if members.contains(&j) {
                    return None;
                }
                let near = members.iter().zip(&proposed).any(|(&i, new_pose)| {
                    [state[i].position, new_pose.position]
                        .into_iter()
                        .any(|center| {
                            let guard = 2048.
                                * f64::EPSILON
                                * (1. + reach + norm(center) + norm(spectator.position));
                            norm(sub(center, spectator.position)) <= reach + guard
                        })
                });
                near.then_some((j, [0; 3]))
            })
            .collect::<Vec<_>>();
        let fixed = labels.iter().map(|&(j, _)| Placed::new(state[j])).collect();
        let environment = Environment {
            tree,
            fixed,
            labels,
            rd,
        };
        Ok(Self {
            tree,
            members: members.to_vec(),
            proposed,
            body,
            environment,
            old,
            new,
            identity: state[handle] == proposed_handle,
            rd,
            root,
        })
    }

    pub fn members(&self) -> &[usize] {
        &self.members
    }
    /// In `members()` order, NOT in whole-configuration order.
    pub fn proposed_poses(&self) -> &[Pose] {
        &self.proposed
    }
    pub fn spectator_count(&self) -> usize {
        self.environment.fixed.len()
    }
    pub fn root_bounds(&self) -> Cell {
        self.root
    }

    /// Core and full atom-wall checks. Internal validity is preserved by a
    /// common isometry in exact arithmetic, but is also checked explicitly to
    /// guard floating-point endpoint geometry. The wall's center is subtracted
    /// only for the wall query, never for physical contacts.
    pub fn hard_valid(&self, wall: Option<&Container>, wall_center: Vec3) -> bool {
        for i in 0..self.proposed.len() {
            for j in 0..i {
                if self.tree.overlaps(
                    &Placed::new(self.proposed[i]),
                    &Placed::new(self.proposed[j]),
                ) {
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

    /// Exact Boolean membership in the moving subset's canonical excluded union.
    pub fn contains_reference_point(&self, point: Vec3) -> bool {
        self.body
            .iter()
            .any(|member| self.tree.contains(member.unapply(point), self.rd))
    }
    /// (old,new) overlap indicators, counted at most once in each union.
    pub fn overlap_indicators(&self, point: Vec3) -> (bool, bool) {
        if !self.contains_reference_point(point) {
            return (false, false);
        }
        (
            self.environment.contains(self.old.apply(point)),
            self.environment.contains(self.new.apply(point)),
        )
    }

    fn classify_body(&self, center: Vec3, radius: f64) -> Coverage {
        let mut result = Coverage::Outside;
        for member in &self.body {
            let guard = 2048.
                * f64::EPSILON
                * (1. + norm(center) + norm(member.position) + self.tree.bound + self.rd + radius);
            match self
                .tree
                .classify_ball(member.unapply(center), radius + guard, self.rd)
            {
                Coverage::Inside => return Coverage::Inside,
                Coverage::Unknown => result = Coverage::Unknown,
                Coverage::Outside => (),
            }
        }
        result
    }

    /// Disjoint dyadic cover of the gained/lost overlap, in the moving frame.
    /// Stopping early enlarges the cover; it never approximates the target.
    pub fn envelope(&self, opts: GateOptions) -> Result<Envelope> {
        opts.validate()?;
        if self.identity || self.environment.fixed.is_empty() {
            return Ok(Envelope {
                cells: vec![],
                cumulative: vec![],
                volume: 0.,
                created: 0,
            });
        }
        let mut queue = VecDeque::from([(self.root, 0)]);
        let mut cells = Vec::new();
        let mut created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let c = cell.center();
            let r = cell.radius();
            let body = self.classify_body(c, r);
            if body == Coverage::Outside {
                continue;
            }
            let a = self.environment.classify_ball(self.old.apply(c), r);
            let b = self.environment.classify_ball(self.new.apply(c), r);
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
        let mut cumulative = Vec::with_capacity(cells.len());
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
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

    /// Sample the conditional Poisson gate. Old-only overlap points have rate
    /// lambda+z and new-only points rate lambda; the likelihood factor is
    /// ((lambda+z)/lambda)^(gained-lost). Averaging *accepted auxiliary flow*, not
    /// exponentiating a noisy energy estimate, gives exact depletion balance.
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
        self.sample_envelope(rng, lambda, z, &envelope)
    }

    /// Prepared envelopes are private to this trial; callers cannot provide an
    /// undersized cover or accidentally reuse one from a different rigid move.
    fn sample_envelope(
        &self,
        rng: &mut StdRng,
        lambda: f64,
        z: f64,
        envelope: &Envelope,
    ) -> Result<GateResult> {
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
            let (old, new) = self.overlap_indicators(point);
            if old && !new {
                result.lost += 1;
            }
            if new && !old && rng.random::<f64>() < lambda / (lambda + z) {
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
