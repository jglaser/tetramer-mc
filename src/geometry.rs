//! Body-frame sphere BVHs with exact leaf predicates and conservative pruning.
//! Overlaps inside a rigid shape are allowed. External overlaps are forbidden.
use crate::math::*;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Atom {
    pub center: Vec3,
    pub radius: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shape {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub volume: f64,
    pub atoms: Vec<Atom>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    Outside,
    Unknown,
    Inside,
}
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub lo: Vec3,
    pub hi: Vec3,
}
impl Cell {
    pub fn center(self) -> Vec3 {
        std::array::from_fn(|k| 0.5 * self.lo[k] + 0.5 * self.hi[k])
    }
    pub fn radius(self) -> f64 {
        let c = self.center();
        norm(std::array::from_fn(|k| {
            (c[k] - self.lo[k]).max(self.hi[k] - c[k])
        }))
    }
    pub fn volume(self) -> f64 {
        (0..3).map(|k| self.hi[k] - self.lo[k]).product()
    }
    pub fn longest(self) -> usize {
        (0..3)
            .max_by(|&i, &j| (self.hi[i] - self.lo[i]).total_cmp(&(self.hi[j] - self.lo[j])))
            .unwrap()
    }
    pub fn split(self) -> Option<(Self, Self)> {
        let k = self.longest();
        let m = self.center()[k];
        if m <= self.lo[k] || m >= self.hi[k] {
            return None;
        }
        let mut a = self;
        let mut b = self;
        a.hi[k] = m;
        b.lo[k] = m;
        Some((a, b))
    }
}

#[derive(Clone, Debug)]
struct Node {
    center: Vec3,
    radius: f64,
    children: Option<(usize, usize)>,
    atom: Option<usize>,
}
#[derive(Clone, Debug)]
pub struct SphereTree {
    pub shape: Shape,
    nodes: Vec<Node>,
    pub bound: f64,
}

/// Last exit from the union of atomic hard-core intervals on t=s*direction,
/// s>=0, for a rotated copy against an identical copy at the origin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadialContact {
    pub distance: f64,
    pub direction: Vec3,
    /// Canonical proper orientation actually used by the radial query.
    pub orientation: [f64; 4],
    pub moving_atom: usize,
    pub fixed_atom: usize,
    pub witness_interval: [f64; 2],
    pub node_pairs_visited: u64,
    pub atomic_pairs_tested: u64,
    pub node_pairs_pruned: u64,
}

#[derive(Clone, Copy, Debug)]
struct RadialNodePair {
    upper: f64,
    moving: usize,
    fixed: usize,
}
impl PartialEq for RadialNodePair {
    fn eq(&self, other: &Self) -> bool {
        self.upper == other.upper && self.moving == other.moving && self.fixed == other.fixed
    }
}
impl Eq for RadialNodePair {}
impl PartialOrd for RadialNodePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RadialNodePair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper
            .total_cmp(&other.upper)
            .then_with(|| other.moving.cmp(&self.moving))
            .then_with(|| other.fixed.cmp(&self.fixed))
    }
}

impl SphereTree {
    pub fn new(shape: Shape) -> Result<Self> {
        ensure!(!shape.atoms.is_empty(), "empty rigid shape");
        ensure!(
            shape.atoms.iter().all(|a| a.radius.is_finite()
                && a.radius > 0.
                && a.center.iter().all(|v| v.is_finite())),
            "invalid atom geometry"
        );
        let bound = shape
            .atoms
            .iter()
            .map(|a| norm(a.center) + a.radius)
            .fold(0., f64::max);
        ensure!(bound.is_finite(), "nonfinite body bound");
        let mut tree = Self {
            shape,
            nodes: Vec::new(),
            bound,
        };
        let mut ids: Vec<_> = (0..tree.shape.atoms.len()).collect();
        tree.build(&mut ids);
        Ok(tree)
    }
    fn build(&mut self, ids: &mut [usize]) -> usize {
        let lo: Vec3 = std::array::from_fn(|k| {
            ids.iter()
                .map(|&i| self.shape.atoms[i].center[k])
                .fold(f64::INFINITY, f64::min)
        });
        let hi: Vec3 = std::array::from_fn(|k| {
            ids.iter()
                .map(|&i| self.shape.atoms[i].center[k])
                .fold(f64::NEG_INFINITY, f64::max)
        });
        let center = scale(add(lo, hi), 0.5);
        let raw = ids
            .iter()
            .map(|&i| norm(sub(self.shape.atoms[i].center, center)) + self.shape.atoms[i].radius)
            .fold(0., f64::max);
        let radius = raw + 256. * f64::EPSILON * (1. + norm(center) + raw);
        let n = self.nodes.len();
        self.nodes.push(Node {
            center,
            radius,
            children: None,
            atom: None,
        });
        if ids.len() == 1 {
            self.nodes[n].atom = Some(ids[0]);
        } else {
            let axis = (0..3)
                .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
                .unwrap();
            ids.sort_unstable_by(|&i, &j| {
                self.shape.atoms[i].center[axis].total_cmp(&self.shape.atoms[j].center[axis])
            });
            let mid = ids.len() / 2;
            let (a, b) = ids.split_at_mut(mid);
            let left = self.build(a);
            let right = self.build(b);
            self.nodes[n].children = Some((left, right));
        }
        n
    }
    pub fn bounds(&self, inflate: f64) -> Cell {
        let guard = 512. * f64::EPSILON * (1. + self.bound + inflate);
        Cell {
            lo: std::array::from_fn(|k| {
                self.shape
                    .atoms
                    .iter()
                    .map(|a| a.center[k] - a.radius - inflate - guard)
                    .fold(f64::INFINITY, f64::min)
            }),
            hi: std::array::from_fn(|k| {
                self.shape
                    .atoms
                    .iter()
                    .map(|a| a.center[k] + a.radius + inflate + guard)
                    .fold(f64::NEG_INFINITY, f64::max)
            }),
        }
    }
    pub fn contains(&self, p: Vec3, inflate: f64) -> bool {
        self.contains_at(0, p, inflate)
    }
    fn contains_at(&self, i: usize, p: Vec3, inflate: f64) -> bool {
        let n = &self.nodes[i];
        let d = sub(p, n.center);
        let r = n.radius + inflate;
        if dot(d, d) > r * r {
            return false;
        }
        if let Some(a) = n.atom {
            let a = &self.shape.atoms[a];
            let d = sub(p, a.center);
            return dot(d, d) <= (a.radius + inflate).powi(2);
        }
        let (l, r) = n.children.unwrap();
        self.contains_at(l, p, inflate) || self.contains_at(r, p, inflate)
    }
    pub fn classify_ball(&self, p: Vec3, radius: f64, inflate: f64) -> Coverage {
        let guard = 1024. * f64::EPSILON * (1. + norm(p) + self.bound + inflate + radius);
        self.classify_at(0, p, radius, inflate, guard)
    }
    fn classify_at(&self, i: usize, p: Vec3, radius: f64, inflate: f64, guard: f64) -> Coverage {
        let n = &self.nodes[i];
        if norm(sub(p, n.center)) > n.radius + inflate + radius + guard {
            return Coverage::Outside;
        }
        if let Some(a) = n.atom {
            let a = &self.shape.atoms[a];
            let d = norm(sub(p, a.center));
            if d + radius + guard < a.radius + inflate {
                Coverage::Inside
            } else if d > a.radius + inflate + radius + guard {
                Coverage::Outside
            } else {
                Coverage::Unknown
            }
        } else {
            let (l, r) = n.children.unwrap();
            let a = self.classify_at(l, p, radius, inflate, guard);
            if a == Coverage::Inside {
                return a;
            }
            let b = self.classify_at(r, p, radius, inflate, guard);
            if b == Coverage::Inside {
                b
            } else if a == Coverage::Outside && b == Coverage::Outside {
                Coverage::Outside
            } else {
                Coverage::Unknown
            }
        }
    }
    /// Strict atomic-interior overlap, with BVH spheres only excluding pairs.
    pub fn overlaps(&self, a: &Placed, b: &Placed) -> bool {
        self.overlap_nodes(0, 0, a, b)
    }
    fn overlap_nodes(&self, i: usize, j: usize, a: &Placed, b: &Placed) -> bool {
        let na = &self.nodes[i];
        let nb = &self.nodes[j];
        let delta = sub(a.apply(na.center), b.apply(nb.center));
        let r = na.radius + nb.radius;
        let guard = 512. * f64::EPSILON * (1. + norm(a.position) + norm(b.position) + self.bound);
        if dot(delta, delta) > (r + guard).powi(2) {
            return false;
        }
        if let (Some(ai), Some(bi)) = (na.atom, nb.atom) {
            let aa = &self.shape.atoms[ai];
            let bb = &self.shape.atoms[bi];
            let d = sub(a.apply(aa.center), b.apply(bb.center));
            return dot(d, d) < (aa.radius + bb.radius).powi(2);
        }
        if na.children.is_some() && (nb.children.is_none() || na.radius >= nb.radius) {
            let (l, r) = na.children.unwrap();
            self.overlap_nodes(l, j, a, b) || self.overlap_nodes(r, j, a, b)
        } else {
            let (l, r) = nb.children.unwrap();
            self.overlap_nodes(i, l, a, b) || self.overlap_nodes(i, r, a, b)
        }
    }

    /// Outermost radial contact, found by a best-first two-body BVH traversal.
    /// Bounding spheres supply conservative exit bounds; only atomic sphere
    /// intervals set the answer. A ray with no positive-length forbidden
    /// interval at s>=0 returns None. This does NOT enumerate internal pockets.
    ///
    /// Rotation ingress is checked and canonicalized through a unit quaternion.
    /// Ordinary FP64 guards support pruning; this is not interval arithmetic.
    pub fn outermost_radial_contact(
        &self,
        relative_rotation: Mat3,
        direction: Vec3,
    ) -> Result<Option<RadialContact>> {
        ensure!(
            relative_rotation.iter().flatten().all(|x| x.is_finite()),
            "nonfinite radial orientation"
        );
        let gram = matmul(transpose(relative_rotation), relative_rotation);
        ensure!((0..3).all(|i| (0..3).all(|j| (gram[i][j] - if i == j {1.} else {0.}).abs() <= 1e-10)), "radial orientation is not orthogonal");
        let r = relative_rotation;
        let determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        ensure!(
            (determinant - 1.).abs() <= 1e-10,
            "radial orientation must be proper"
        );
        let orientation = quaternion(relative_rotation);
        let relative_rotation = rotation(orientation);
        let length = norm(direction);
        ensure!(
            direction.iter().all(|x| x.is_finite()) && length.is_finite() && length > 0.,
            "invalid radial direction"
        );
        let direction = direction.map(|x| x / length);
        let aa = dot(direction, direction);
        let mut heap = BinaryHeap::new();
        if let Some(upper) = self.radial_node_upper(0, 0, relative_rotation, direction, aa)? {
            heap.push(RadialNodePair {
                upper,
                moving: 0,
                fixed: 0,
            });
        }
        let mut best: Option<RadialContact> = None;
        let mut visited = 0;
        let mut tested = 0;
        let mut pruned = 0;
        while let Some(pair) = heap.pop() {
            if best.as_ref().is_some_and(|b| pair.upper <= b.distance) {
                pruned += 1 + heap.len() as u64;
                break;
            }
            visited += 1;
            let moving = &self.nodes[pair.moving];
            let fixed = &self.nodes[pair.fixed];
            if let (Some(i), Some(j)) = (moving.atom, fixed.atom) {
                tested += 1;
                let a = &self.shape.atoms[i];
                let b = &self.shape.atoms[j];
                let d = sub(matvec(relative_rotation, a.center), b.center);
                let bb = dot(d, direction);
                let cc = dot(d, d) - (a.radius + b.radius).powi(2);
                let discriminant = bb.mul_add(bb, -aa * cc);
                ensure!(
                    discriminant.is_finite(),
                    "unrepresentable atomic radial discriminant"
                );
                if discriminant <= 0. {
                    continue;
                }
                let far_numerator = -bb - discriminant.sqrt().copysign(bb);
                ensure!(
                    far_numerator.is_finite() && far_numerator != 0.,
                    "unrepresentable atomic radial root"
                );
                let aroot = far_numerator / aa;
                let broot = cc / far_numerator;
                let lower = aroot.min(broot);
                let upper = aroot.max(broot);
                ensure!(
                    lower.is_finite() && upper.is_finite(),
                    "nonfinite atomic radial interval"
                );
                if upper > 0.
                    && best.as_ref().is_none_or(|b| {
                        upper > b.distance
                            || (upper == b.distance && (i, j) < (b.moving_atom, b.fixed_atom))
                    })
                {
                    best = Some(RadialContact {
                        distance: upper,
                        direction,
                        orientation,
                        moving_atom: i,
                        fixed_atom: j,
                        witness_interval: [lower, upper],
                        node_pairs_visited: 0,
                        atomic_pairs_tested: 0,
                        node_pairs_pruned: 0,
                    });
                }
                continue;
            }
            let children = if moving.children.is_some()
                && (fixed.children.is_none() || moving.radius >= fixed.radius)
            {
                let (a, b) = moving.children.unwrap();
                [(a, pair.fixed), (b, pair.fixed)]
            } else {
                let (a, b) = fixed.children.unwrap();
                [(pair.moving, a), (pair.moving, b)]
            };
            for (i, j) in children {
                if let Some(upper) =
                    self.radial_node_upper(i, j, relative_rotation, direction, aa)?
                {
                    if best.as_ref().is_none_or(|b| upper > b.distance) {
                        heap.push(RadialNodePair {
                            upper,
                            moving: i,
                            fixed: j,
                        });
                    } else {
                        pruned += 1;
                    }
                } else {
                    pruned += 1;
                }
            }
        }
        if let Some(contact) = &mut best {
            contact.node_pairs_visited = visited;
            contact.atomic_pairs_tested = tested;
            contact.node_pairs_pruned = pruned;
        }
        Ok(best)
    }

    fn radial_node_upper(
        &self,
        i: usize,
        j: usize,
        rotation: Mat3,
        direction: Vec3,
        aa: f64,
    ) -> Result<Option<f64>> {
        let a = &self.nodes[i];
        let b = &self.nodes[j];
        let d = sub(matvec(rotation, a.center), b.center);
        let center = -dot(d, direction) / aa;
        let perpendicular = add(d, scale(direction, center));
        let guard = 2048. * f64::EPSILON * (1. + self.bound + norm(d) + a.radius + b.radius);
        let radius = a.radius + b.radius + guard;
        let discriminant = radius * radius - dot(perpendicular, perpendicular);
        ensure!(
            center.is_finite() && discriminant.is_finite(),
            "unrepresentable radial node bound"
        );
        if discriminant < 0. {
            return Ok(None);
        }
        let upper = center + (discriminant / aa).sqrt() + guard;
        ensure!(upper.is_finite(), "unrepresentable radial upper bound");
        Ok((upper > 0.).then_some(upper))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub position: Vec3,
    pub rotation: Mat3,
    inverse: Mat3,
}
impl Placed {
    /// Construct from a validated pose. Invalid poses are programmer errors.
    pub fn new(pose: Pose) -> Self {
        assert!(pose.validate().is_ok(), "invalid rigid pose");
        let rotation = rotation(pose.orientation);
        Self {
            position: pose.position,
            rotation,
            inverse: transpose(rotation),
        }
    }
    pub fn apply(&self, p: Vec3) -> Vec3 {
        add(self.position, matvec(self.rotation, p))
    }
    pub fn unapply(&self, p: Vec3) -> Vec3 {
        matvec(self.inverse, sub(p, self.position))
    }
}

/// Symmetric old/new set of unique nearby periodic images of spectator bodies.
pub struct Environment<'a> {
    pub tree: &'a SphereTree,
    pub fixed: Vec<Placed>,
    pub labels: Vec<(usize, [i64; 3])>,
    pub rd: f64,
}
impl<'a> Environment<'a> {
    pub fn new(
        tree: &'a SphereTree,
        poses: &[Pose],
        moving: usize,
        old: Pose,
        new: Pose,
        lengths: Vec3,
        rd: f64,
    ) -> Result<Self> {
        ensure!(moving < poses.len(), "moving index out of range");
        ensure!(rd.is_finite() && rd >= 0., "invalid depletant radius");
        ensure!(
            lengths
                .iter()
                .all(|&l| l.is_finite() && l > 4. * (tree.bound + rd)),
            "box must satisfy L > 4*(body bound + depletant radius)"
        );
        old.validate()?;
        new.validate()?;
        for p in poses {
            p.validate()?;
        }
        for p in poses.iter().chain([&old, &new]) {
            ensure!(
                (0..3).all(|k| p.position[k] >= 0. && p.position[k] < lengths[k]),
                "environment poses must use canonical [0,L) centers"
            );
        }
        let reach = 2. * (tree.bound + rd);
        let guard = 512. * f64::EPSILON * (1. + reach + lengths.iter().copied().fold(0., f64::max));
        let mut labels = BTreeSet::new();
        for center in [old.position, new.position] {
            for (j, p) in poses.iter().enumerate() {
                if j == moving {
                    continue;
                }
                let image: [i64; 3] = std::array::from_fn(|k| {
                    (-((p.position[k] - center[k]) / lengths[k] + 0.5).floor()) as i64
                });
                let shifted = std::array::from_fn(|k| p.position[k] + image[k] as f64 * lengths[k]);
                if norm(sub(shifted, center)) <= reach + guard {
                    labels.insert((j, image));
                }
            }
        }
        let labels: Vec<_> = labels.into_iter().collect();
        let fixed = labels
            .iter()
            .map(|&(j, image)| {
                Placed::new(Pose {
                    position: std::array::from_fn(|k| {
                        poses[j].position[k] + image[k] as f64 * lengths[k]
                    }),
                    orientation: poses[j].orientation,
                })
            })
            .collect();
        Ok(Self {
            tree,
            fixed,
            labels,
            rd,
        })
    }
    pub fn contains(&self, p: Vec3) -> bool {
        self.fixed
            .iter()
            .any(|f| self.tree.contains(f.unapply(p), self.rd))
    }
    pub fn classify_ball(&self, p: Vec3, r: f64) -> Coverage {
        let mut answer = Coverage::Outside;
        for fixed in &self.fixed {
            let guard =
                1024. * f64::EPSILON * (1. + norm(p) + norm(fixed.position) + self.tree.bound + r);
            match self
                .tree
                .classify_ball(fixed.unapply(p), r + guard, self.rd)
            {
                Coverage::Inside => return Coverage::Inside,
                Coverage::Unknown => answer = Coverage::Unknown,
                Coverage::Outside => {}
            }
        }
        answer
    }
    pub fn hard_valid(&self, p: Pose) -> bool {
        let a = Placed::new(p);
        !self.fixed.iter().any(|b| self.tree.overlaps(&a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    #[test]
    fn bvh_matches_brute() {
        let mut rng = StdRng::seed_from_u64(9801);
        let atoms = (0..53)
            .map(|_| Atom {
                center: std::array::from_fn(|_| rng.random_range(-4.0..4.0)),
                radius: rng.random_range(0.2..1.4),
            })
            .collect();
        let tree = SphereTree::new(Shape {
            name: "test".into(),
            volume: 0.,
            atoms,
        })
        .unwrap();
        for _ in 0..4000 {
            let p = std::array::from_fn(|_| rng.random_range(-6.0..6.0));
            let r = rng.random_range(0.0..0.5);
            let brute = tree
                .shape
                .atoms
                .iter()
                .any(|a| norm(sub(p, a.center)) <= a.radius + 0.7);
            assert_eq!(tree.contains(p, 0.7), brute);
            let status = tree.classify_ball(p, r, 0.7);
            if status == Coverage::Outside {
                assert!(
                    tree.shape
                        .atoms
                        .iter()
                        .all(|a| norm(sub(p, a.center)) > a.radius + 0.7 + r)
                );
            }
            if status == Coverage::Inside {
                assert!(
                    tree.shape
                        .atoms
                        .iter()
                        .any(|a| norm(sub(p, a.center)) + r < a.radius + 0.7)
                );
            }
        }
        for _ in 0..100 {
            let a = Placed::new(crate::math::uniform_pose(&mut rng, [15.; 3]));
            let b = Placed::new(crate::math::uniform_pose(&mut rng, [15.; 3]));
            let brute = tree.shape.atoms.iter().any(|x| {
                tree.shape
                    .atoms
                    .iter()
                    .any(|y| norm(sub(a.apply(x.center), b.apply(y.center))) < x.radius + y.radius)
            });
            assert_eq!(tree.overlaps(&a, &b), brute);
        }
    }
}
