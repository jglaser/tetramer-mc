//! Exact leaf geometry for finite translation lines through sphere unions.
//!
//! At fixed proper orientation the moving center is `origin.position + s*d`.
//! Atomic hard overlaps are open quadratic intervals; atomic wall constraints
//! are closed intervals. BVH spheres only prune impossible pairs or certify
//! complete wall containment. They never replace the atomic predicates.
//!
//! "Exact" refers to the sphere-union model and analytic interval construction,
//! not certified real arithmetic. FP64 rotation, distances, roots and endpoint
//! comparisons remain implementation obligations. No tolerance is applied to
//! leaf predicates or interval unions; conservative padding is used only when
//! pruning tree nodes. Unrepresentable arithmetic returns an error.
use crate::{
    geometry::{Placed, SphereTree},
    math::*,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
    pub lower_closed: bool,
    pub upper_closed: bool,
}

impl Interval {
    pub fn closed(lower: f64, upper: f64) -> Result<Self> {
        ensure!(
            lower.is_finite() && upper.is_finite() && lower <= upper,
            "invalid finite closed interval"
        );
        Ok(Self {
            lower,
            upper,
            lower_closed: true,
            upper_closed: true,
        })
    }

    pub fn contains(&self, x: f64) -> bool {
        (x > self.lower || (x == self.lower && self.lower_closed))
            && (x < self.upper || (x == self.upper && self.upper_closed))
    }

    pub fn length(&self) -> f64 {
        self.upper - self.lower
    }

    fn nonempty(&self) -> bool {
        self.lower < self.upper
            || (self.lower == self.upper && self.lower_closed && self.upper_closed)
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        let result = Self {
            lower,
            upper,
            lower_closed: (lower != self.lower || self.lower_closed)
                && (lower != other.lower || other.lower_closed),
            upper_closed: (upper != self.upper || self.upper_closed)
                && (upper != other.upper || other.upper_closed),
        };
        result.nonempty().then_some(result)
    }
}

/// Sorted, disjoint intervals. A missing tangent point is never filled in.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct IntervalSet {
    intervals: Vec<Interval>,
}

impl IntervalSet {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn from_intervals(intervals: Vec<Interval>) -> Result<Self> {
        ensure!(
            intervals
                .iter()
                .all(|v| v.lower.is_finite() && v.upper.is_finite() && v.lower <= v.upper),
            "invalid finite interval"
        );
        Ok(Self::normalized(intervals))
    }

    pub fn segment(segment: [f64; 2]) -> Result<Self> {
        Self::from_intervals(vec![Interval {
            lower: segment[0],
            upper: segment[1],
            lower_closed: true,
            upper_closed: true,
        }])
    }

    fn normalized(mut intervals: Vec<Interval>) -> Self {
        intervals.retain(Interval::nonempty);
        intervals.sort_by(|a, b| {
            a.lower
                .total_cmp(&b.lower)
                .then_with(|| b.lower_closed.cmp(&a.lower_closed))
        });
        let mut result: Vec<Interval> = Vec::with_capacity(intervals.len());
        for next in intervals {
            if let Some(last) = result.last_mut() {
                if next.lower < last.upper
                    || (next.lower == last.upper && (next.lower_closed || last.upper_closed))
                {
                    if next.upper > last.upper {
                        last.upper = next.upper;
                        last.upper_closed = next.upper_closed;
                    } else if next.upper == last.upper {
                        last.upper_closed |= next.upper_closed;
                    }
                    continue;
                }
            }
            result.push(next);
        }
        Self { intervals: result }
    }

    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }
    pub fn contains(&self, x: f64) -> bool {
        self.intervals.iter().any(|v| v.contains(x))
    }
    pub fn length(&self) -> f64 {
        self.intervals.iter().map(Interval::length).sum()
    }

    pub fn union(&self, other: &Self) -> Self {
        Self::normalized(
            self.intervals
                .iter()
                .chain(&other.intervals)
                .copied()
                .collect(),
        )
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let (mut i, mut j) = (0, 0);
        let mut result = Vec::new();
        while i < self.intervals.len() && j < other.intervals.len() {
            let a = self.intervals[i];
            let b = other.intervals[j];
            if let Some(v) = a.intersection(b) {
                result.push(v);
            }
            // Advancing just one on equal endpoints preserves a possible
            // singleton intersection with the next open-ended interval.
            if a.upper <= b.upper {
                i += 1;
            } else {
                j += 1;
            }
        }
        Self::normalized(result)
    }

    pub fn difference(&self, other: &Self) -> Self {
        let mut pieces = self.intervals.clone();
        for cut in &other.intervals {
            let mut next = Vec::new();
            for piece in pieces {
                if let Some(hit) = piece.intersection(*cut) {
                    let left = Interval {
                        lower: piece.lower,
                        upper: hit.lower,
                        lower_closed: piece.lower_closed,
                        upper_closed: piece.contains(hit.lower) && !hit.lower_closed,
                    };
                    let right = Interval {
                        lower: hit.upper,
                        upper: piece.upper,
                        lower_closed: piece.contains(hit.upper) && !hit.upper_closed,
                        upper_closed: piece.upper_closed,
                    };
                    if left.nonempty() {
                        next.push(left);
                    }
                    if right.nonempty() {
                        next.push(right);
                    }
                } else {
                    next.push(piece);
                }
            }
            pieces = next;
            if pieces.is_empty() {
                break;
            }
        }
        Self::normalized(pieces)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SphericalWall {
    pub center: Vec3,
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineCounts {
    pub node_pairs_visited: u64,
    pub node_pairs_pruned: u64,
    pub leaf_pairs_tested: u64,
    pub wall_nodes_visited: u64,
    pub wall_nodes_contained: u64,
    pub wall_atoms_tested: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LineGeometry {
    /// Positive-volume core intersections: |ci(s)-cj| < ri+rj.
    pub hard_overlap: IntervalSet,
    /// Optional |ci(s)-cj| < ri+rj+gap. `gap` is the total pair gap;
    /// a depletant radius is not automatically multiplied by two.
    pub contact_overlap: Option<IntervalSet>,
    /// Complete atomic-wall constraint, or the full segment without a wall.
    pub wall_valid: IntervalSet,
    pub counts: LineCounts,
}

/// Analytic interval for |offset + s*direction| < radius (or <= when closed).
/// Zero direction is a constant predicate; wall tangencies may be singletons.
pub(crate) fn ball_interval(
    offset: Vec3,
    direction: Vec3,
    radius: f64,
    segment: Interval,
    closed: bool,
) -> Result<Option<Interval>> {
    ensure!(
        offset.iter().chain(direction.iter()).all(|v| v.is_finite())
            && radius.is_finite()
            && radius >= 0.,
        "invalid line-ball geometry"
    );
    let speed = norm(direction);
    ensure!(speed.is_finite(), "unrepresentable line direction");
    if speed == 0. {
        let distance = norm(offset);
        ensure!(distance.is_finite(), "unrepresentable constant distance");
        return Ok((distance < radius || (closed && distance == radius)).then_some(segment));
    }
    let unit = direction.map(|v| v / speed);
    ensure!(
        unit.iter().all(|v| v.is_finite()),
        "unrepresentable normalized direction"
    );
    let along = dot(offset, unit);
    let cross: Vec3 = [
        offset[1].mul_add(unit[2], -offset[2] * unit[1]),
        offset[2].mul_add(unit[0], -offset[0] * unit[2]),
        offset[0].mul_add(unit[1], -offset[1] * unit[0]),
    ];
    let perpendicular = norm(cross);
    ensure!(
        along.is_finite() && perpendicular.is_finite(),
        "unrepresentable line projection"
    );
    if perpendicular > radius || (perpendicular == radius && !closed) {
        return Ok(None);
    }
    // This avoids cancellation of |offset|² - along² for distant lines.
    let half = if radius == 0. {
        0.
    } else {
        radius * ((1. - perpendicular / radius) * (1. + perpendicular / radius)).sqrt()
    };
    let lower = (-along - half) / speed;
    let upper = (-along + half) / speed;
    ensure!(
        !lower.is_nan() && !upper.is_nan(),
        "unrepresentable line roots"
    );
    Ok(segment.intersection(Interval {
        lower,
        upper,
        lower_closed: closed,
        upper_closed: closed,
    }))
}

/// Construct all hard/contact intervals and atomic spherical-wall intervals.
/// All inputs are immutable; fixed bodies may use a different sphere union.
/// This routine has no periodic-image policy: callers supply every image that
/// can intersect the finite segment, as ordinary fixed poses.
#[allow(clippy::too_many_arguments)]
pub fn translation_intervals(
    moving_tree: &SphereTree,
    origin: Pose,
    direction: Vec3,
    fixed_tree: &SphereTree,
    fixed_poses: &[Pose],
    segment: [f64; 2],
    contact_gap: Option<f64>,
    wall: Option<SphericalWall>,
) -> Result<LineGeometry> {
    origin.validate()?;
    ensure!(
        direction.iter().all(|v| v.is_finite()),
        "nonfinite line direction"
    );
    let whole = IntervalSet::segment(segment)?;
    let segment = whole.intervals[0];
    if let Some(gap) = contact_gap {
        ensure!(gap.is_finite() && gap >= 0., "invalid contact gap");
    }
    if let Some(w) = wall {
        ensure!(
            w.center.iter().all(|v| v.is_finite()) && w.radius.is_finite() && w.radius > 0.,
            "invalid spherical wall"
        );
    }
    let moving = Placed::new(origin);
    let mut result = LineGeometry {
        hard_overlap: IntervalSet::default(),
        contact_overlap: contact_gap.map(|_| IntervalSet::default()),
        wall_valid: whole.clone(),
        counts: LineCounts::default(),
    };
    let mut hard = Vec::new();
    let mut contact = Vec::new();
    for pose in fixed_poses {
        pose.validate()?;
        let fixed = Placed::new(*pose);
        let mut pending = vec![(0, 0)];
        while let Some((i, j)) = pending.pop() {
            result.counts.node_pairs_visited += 1;
            let a = moving_tree.line_node(i);
            let b = fixed_tree.line_node(j);
            let ac = moving.apply(a.center);
            let bc = fixed.apply(b.center);
            let extent = norm(direction) * segment.lower.abs().max(segment.upper.abs());
            let guard =
                1024. * f64::EPSILON * (1. + norm(ac) + norm(bc) + a.radius + b.radius + extent);
            ensure!(guard.is_finite(), "unrepresentable line BVH guard");
            if ball_interval(
                sub(ac, bc),
                direction,
                a.radius + b.radius + contact_gap.unwrap_or(0.) + guard,
                segment,
                true,
            )?
            .is_none()
            {
                result.counts.node_pairs_pruned += 1;
                continue;
            }
            if let (Some(ai), Some(bi)) = (a.atom, b.atom) {
                result.counts.leaf_pairs_tested += 1;
                let aa = &moving_tree.shape.atoms[ai];
                let bb = &fixed_tree.shape.atoms[bi];
                let offset = sub(moving.apply(aa.center), fixed.apply(bb.center));
                if let Some(v) =
                    ball_interval(offset, direction, aa.radius + bb.radius, segment, false)?
                {
                    hard.push(v);
                }
                if let Some(gap) = contact_gap {
                    if let Some(v) = ball_interval(
                        offset,
                        direction,
                        aa.radius + bb.radius + gap,
                        segment,
                        false,
                    )? {
                        contact.push(v);
                    }
                }
            } else if a.children.is_some() && (b.children.is_none() || a.radius >= b.radius) {
                let (l, r) = a.children.unwrap();
                pending.push((r, j));
                pending.push((l, j));
            } else {
                let (l, r) = b.children.unwrap();
                pending.push((i, r));
                pending.push((i, l));
            }
        }
    }
    result.hard_overlap = IntervalSet::normalized(hard);
    result.contact_overlap = contact_gap.map(|_| IntervalSet::normalized(contact));
    if let Some(wall) = wall {
        let mut pending = vec![0];
        while let Some(i) = pending.pop() {
            result.counts.wall_nodes_visited += 1;
            let node = moving_tree.line_node(i);
            let center = sub(moving.apply(node.center), wall.center);
            let guard = 1024. * f64::EPSILON * (1. + norm(center) + node.radius + wall.radius);
            ensure!(guard.is_finite(), "unrepresentable wall BVH guard");
            // The norm is convex along the line, so the two segment endpoints
            // suffice to certify containment of an entire enclosing sphere.
            let contained = [segment.lower, segment.upper].iter().all(|&s| {
                let distance = norm(add(center, scale(direction, s)));
                distance.is_finite() && distance + node.radius + guard < wall.radius
            });
            if contained {
                result.counts.wall_nodes_contained += 1;
                continue;
            }
            if let Some(ai) = node.atom {
                result.counts.wall_atoms_tested += 1;
                let atom = &moving_tree.shape.atoms[ai];
                if atom.radius > wall.radius {
                    result.wall_valid = IntervalSet::default();
                    break;
                }
                let offset = sub(moving.apply(atom.center), wall.center);
                let valid =
                    ball_interval(offset, direction, wall.radius - atom.radius, segment, true)?;
                result.wall_valid = result
                    .wall_valid
                    .intersection(&IntervalSet::normalized(valid.into_iter().collect()));
                if result.wall_valid.is_empty() {
                    break;
                }
            } else {
                let (l, r) = node.children.unwrap();
                pending.push(r);
                pending.push(l);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
