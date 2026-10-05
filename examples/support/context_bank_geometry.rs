//! Example-local patch enumeration. Production hard-core and envelope APIs stay authoritative.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};
use tetramer_mc::{
    geometry::{Placed, Shape},
    math::*,
    simulation::cpu_seconds,
};

pub type Token = (usize, usize, String, String);
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    pub patch_node_visits_per_candidate: u64,
    pub patch_leaf_tests_per_candidate: u64,
    pub patch_node_visits_total: u64,
    pub patch_leaf_tests_total: u64,
}
pub struct Work {
    pub limits: Limits,
    pub nodes: u64,
    pub leaves: u64,
    pub total_nodes: u64,
    pub total_leaves: u64,
    pub started_cpu: f64,
    pub started_wall: Instant,
}
impl Work {
    pub fn new(limits: Limits) -> Result<Self> {
        ensure!(
            limits.cpu_seconds.is_finite()
                && limits.cpu_seconds > 0.
                && limits.wall_seconds.is_finite()
                && limits.wall_seconds > 0.,
            "Invalid time limits"
        );
        ensure!(
            limits.patch_node_visits_per_candidate > 0
                && limits.patch_leaf_tests_per_candidate > 0
                && limits.patch_node_visits_total > 0
                && limits.patch_leaf_tests_total > 0,
            "Invalid patch count limits"
        );
        Ok(Self {
            limits,
            nodes: 0,
            leaves: 0,
            total_nodes: 0,
            total_leaves: 0,
            started_cpu: cpu_seconds(),
            started_wall: Instant::now(),
        })
    }
    pub fn check(&self) -> Result<()> {
        ensure!(
            cpu_seconds() - self.started_cpu <= self.limits.cpu_seconds
                && self.started_wall.elapsed().as_secs_f64() <= self.limits.wall_seconds,
            "Fatal preflight time limit"
        );
        Ok(())
    }
    pub fn begin(&mut self) -> Result<()> {
        self.nodes = 0;
        self.leaves = 0;
        self.check()
    }
    fn node(&mut self) -> Result<()> {
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Node count overflow"))?;
        self.total_nodes = self
            .total_nodes
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Total node count overflow"))?;
        ensure!(
            self.nodes <= self.limits.patch_node_visits_per_candidate
                && self.total_nodes <= self.limits.patch_node_visits_total,
            "Fatal patch node cap"
        );
        if self.nodes % 1024 == 0 {
            self.check()?;
        }
        Ok(())
    }
    fn leaf(&mut self) -> Result<()> {
        self.leaves = self
            .leaves
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Leaf count overflow"))?;
        self.total_leaves = self
            .total_leaves
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Total leaf count overflow"))?;
        ensure!(
            self.leaves <= self.limits.patch_leaf_tests_per_candidate
                && self.total_leaves <= self.limits.patch_leaf_tests_total,
            "Fatal patch leaf cap"
        );
        Ok(())
    }
}
#[derive(Clone)]
struct Node {
    center: Vec3,
    radius: f64,
    children: Option<(usize, usize)>,
    atom: Option<usize>,
}
pub struct PatchBvh {
    nodes: Vec<Node>,
    radii: Vec<f64>,
    patches: Vec<usize>,
    names: Vec<String>,
    bound: f64,
}
pub struct World {
    centers: Vec<Vec3>,
    position: Vec3,
}
impl PatchBvh {
    pub fn new(shape: &Shape, patches: &[String]) -> Result<Self> {
        ensure!(
            !shape.atoms.is_empty()
                && patches.len() == shape.atoms.len()
                && patches.iter().all(|p| !p.is_empty()),
            "Patch inventory mismatch"
        );
        let names: Vec<_> = patches
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let labels: BTreeMap<_, _> = names
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let mut tree = Self {
            nodes: Vec::new(),
            radii: shape.atoms.iter().map(|a| a.radius).collect(),
            patches: patches.iter().map(|s| labels[s]).collect(),
            names,
            bound: shape
                .atoms
                .iter()
                .map(|a| norm(a.center) + a.radius)
                .fold(0., f64::max),
        };
        tree.build(shape, (0..shape.atoms.len()).collect())?;
        Ok(tree)
    }
    fn build(&mut self, shape: &Shape, mut indices: Vec<usize>) -> Result<usize> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &i in &indices {
            for k in 0..3 {
                lo[k] = lo[k].min(shape.atoms[i].center[k]);
                hi[k] = hi[k].max(shape.atoms[i].center[k]);
            }
        }
        let center = std::array::from_fn(|k| 0.5 * lo[k] + 0.5 * hi[k]);
        let raw = indices
            .iter()
            .map(|&i| norm(sub(shape.atoms[i].center, center)) + shape.atoms[i].radius)
            .fold(0., f64::max);
        let radius = raw + 2048. * f64::EPSILON * (1. + norm(center) + raw + self.bound);
        ensure!(
            center.iter().all(|x| x.is_finite()) && radius.is_finite(),
            "Patch bound overflow"
        );
        let index = self.nodes.len();
        self.nodes.push(Node {
            center,
            radius,
            children: None,
            atom: None,
        });
        if indices.len() == 1 {
            self.nodes[index].center = shape.atoms[indices[0]].center;
            self.nodes[index].atom = Some(indices[0]);
        } else {
            let axis = (0..3)
                .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
                .unwrap();
            indices.sort_by(|&a, &b| {
                shape.atoms[a].center[axis]
                    .total_cmp(&shape.atoms[b].center[axis])
                    .then(a.cmp(&b))
            });
            let right = indices.split_off(indices.len() / 2);
            let left = self.build(shape, indices)?;
            let right = self.build(shape, right)?;
            self.nodes[index].children = Some((left, right));
        }
        Ok(index)
    }
    pub fn placed(&self, pose: Pose) -> Result<World> {
        pose.validate()?;
        let p = Placed::new(pose);
        let centers: Vec<_> = self.nodes.iter().map(|n| p.apply(n.center)).collect();
        ensure!(
            centers.iter().flatten().all(|x| x.is_finite()),
            "Patch world transform overflow"
        );
        Ok(World {
            centers,
            position: pose.position,
        })
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn contacts(
        &self,
        left: &World,
        right: &World,
        left_label: usize,
        right_label: usize,
        rd: f64,
        work: &mut Work,
    ) -> Result<BTreeSet<Token>> {
        ensure!(
            left_label < right_label && rd.is_finite() && rd >= 0.,
            "Invalid patch pair"
        );
        let mut pending = vec![(0, 0)];
        let mut pairs = BTreeSet::new();
        let guard = 4096.
            * f64::EPSILON
            * (1. + norm(left.position) + norm(right.position) + self.bound + rd);
        while let Some((i, j)) = pending.pop() {
            work.node()?;
            let a = &self.nodes[i];
            let b = &self.nodes[j];
            let d = sub(left.centers[i], right.centers[j]);
            let radius = a.radius + b.radius + 2. * rd + guard;
            ensure!(
                radius.is_finite() && dot(d, d).is_finite(),
                "Patch distance overflow"
            );
            if dot(d, d) > radius * radius {
                continue;
            }
            match (a.atom, b.atom) {
                (Some(ai), Some(bi)) => {
                    work.leaf()?;
                    let r = (self.radii[ai] + rd) + (self.radii[bi] + rd);
                    if dot(d, d) < r * r {
                        pairs.insert((self.patches[ai], self.patches[bi]));
                    }
                }
                _ => {
                    if a.children.is_some() && (b.children.is_none() || a.radius >= b.radius) {
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
        }
        Ok(pairs
            .into_iter()
            .map(|(a, b)| {
                (
                    left_label,
                    right_label,
                    self.names[a].clone(),
                    self.names[b].clone(),
                )
            })
            .collect())
    }
}

pub fn near(pose: Pose, fixed: Pose, bound: f64, rd: f64) -> bool {
    let reach = 2. * (bound + rd);
    let guard = 2048. * f64::EPSILON * (1. + reach + norm(pose.position) + norm(fixed.position));
    norm(sub(pose.position, fixed.position)) <= reach + guard
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::{
        depletion::GateOptions,
        geometry::{Atom, Environment, SphereTree},
        overlap_weight::OverlapEnvelope,
    };
    fn limits() -> Limits {
        Limits {
            cpu_seconds: 30.,
            wall_seconds: 60.,
            patch_node_visits_per_candidate: 1000000,
            patch_leaf_tests_per_candidate: 1000000,
            patch_node_visits_total: 10000000,
            patch_leaf_tests_total: 10000000,
        }
    }
    fn pose(x: f64, y: f64, angle: f64) -> Pose {
        Pose {
            position: [x, y, 0.],
            orientation: [(angle / 2.).cos(), 0., 0., (angle / 2.).sin()],
        }
    }
    fn shape() -> Shape {
        Shape {
            name: "toy".into(),
            volume: 1.,
            atoms: vec![
                Atom {
                    center: [-0.9, 0., 0.],
                    radius: 0.7,
                },
                Atom {
                    center: [0.9, 0., 0.],
                    radius: 0.4,
                },
                Atom {
                    center: [0., 0.8, 0.3],
                    radius: 0.3,
                },
            ],
        }
    }
    fn exhaustive(s: &Shape, a: Pose, b: Pose, patches: &[String], rd: f64) -> BTreeSet<Token> {
        let a = Placed::new(a);
        let b = Placed::new(b);
        let mut out = BTreeSet::new();
        for (i, x) in s.atoms.iter().enumerate() {
            for (j, y) in s.atoms.iter().enumerate() {
                let d = sub(a.apply(x.center), b.apply(y.center));
                let r = (x.radius + rd) + (y.radius + rd);
                if dot(d, d) < r * r {
                    out.insert((16, 77, patches[i].clone(), patches[j].clone()));
                }
            }
        }
        out
    }
    #[test]
    fn patches_match_exhaustive_rotated_union() {
        let s = shape();
        let labels = vec!["left".into(), "right".into(), "tip".into()];
        let tree = PatchBvh::new(&s, &labels).unwrap();
        let mut w = Work::new(limits()).unwrap();
        for k in 0..24 {
            let a = pose(-0.2, 0.1, 0.13 * k as f64);
            let b = pose(0.12 * k as f64 - 1.2, 0.3, -0.17 * k as f64);
            w.begin().unwrap();
            assert_eq!(
                tree.contacts(
                    &tree.placed(a).unwrap(),
                    &tree.placed(b).unwrap(),
                    16,
                    77,
                    0.2,
                    &mut w
                )
                .unwrap(),
                exhaustive(&s, a, b, &labels, 0.2)
            );
        }
    }
    #[test]
    fn strict_tangency_and_canonical_labels() {
        let s = Shape {
            name: "sphere".into(),
            volume: 1.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        };
        let tree = PatchBvh::new(&s, &["p".into()]).unwrap();
        let mut w = Work::new(limits()).unwrap();
        let a = tree.placed(pose(0., 0., 0.)).unwrap();
        for (x, hit) in [(2.4, false), (2.4 - 1e-10, true), (2.4 + 1e-10, false)] {
            w.begin().unwrap();
            let b = tree.placed(pose(x, 0., 0.)).unwrap();
            assert_eq!(
                !tree
                    .contacts(&a, &b, 16, 77, 0.2, &mut w)
                    .unwrap()
                    .is_empty(),
                hit
            );
        }
        assert!(tree.contacts(&a, &a, 77, 16, 0.2, &mut w).is_err());
    }
    #[test]
    fn coincident_centers_and_repeated_patches() {
        let s = shape();
        let labels = vec!["same".into(); 3];
        let tree = PatchBvh::new(&s, &labels).unwrap();
        let p = pose(1., 2., 0.7);
        let world = tree.placed(p).unwrap();
        let mut w = Work::new(limits()).unwrap();
        assert_eq!(
            tree.contacts(&world, &world, 16, 77, 0., &mut w)
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn node_and_leaf_caps_fail_without_truncation() {
        let s = shape();
        let labels = vec!["p".into(); 3];
        let tree = PatchBvh::new(&s, &labels).unwrap();
        let world = tree.placed(pose(0., 0., 0.)).unwrap();
        let mut l = limits();
        l.patch_node_visits_per_candidate = 1;
        let mut w = Work::new(l).unwrap();
        assert!(tree.contacts(&world, &world, 16, 77, 0.2, &mut w).is_err());
        assert_eq!(w.nodes, 2);
        let mut l = limits();
        l.patch_leaf_tests_per_candidate = 1;
        let mut w = Work::new(l).unwrap();
        assert!(tree.contacts(&world, &world, 16, 77, 0.2, &mut w).is_err());
        assert_eq!(w.leaves, 2);
    }
    #[test]
    fn conservative_pruning_matches_unpruned_overlap_predicate() {
        let tree = SphereTree::new(shape()).unwrap();
        let moving = pose(0.3, -0.2, 0.9);
        let bodies = [pose(1.1, 0., -0.7), pose(-2., 0.4, 0.6), pose(90., 0., 0.)];
        let rd = 0.4;
        let all = Environment {
            tree: &tree,
            fixed: bodies.iter().map(|p| Placed::new(*p)).collect(),
            labels: vec![(0, [0; 3]), (1, [0; 3]), (2, [0; 3])],
            rd,
        };
        let selected: Vec<_> = bodies
            .iter()
            .enumerate()
            .filter(|(_, p)| near(moving, **p, tree.bound, rd))
            .collect();
        assert_eq!(selected.len(), 2);
        let pruned = Environment {
            tree: &tree,
            fixed: selected.iter().map(|(_, p)| Placed::new(**p)).collect(),
            labels: selected.iter().map(|(i, _)| (*i, [0; 3])).collect(),
            rd,
        };
        let placed = Placed::new(moving);
        for i in -8..=8 {
            for j in -8..=8 {
                for k in -4..=4 {
                    let x = [i as f64 / 3., j as f64 / 3., k as f64 / 3.];
                    if tree.contains(x, rd) {
                        assert_eq!(
                            all.contains(placed.apply(x)),
                            pruned.contains(placed.apply(x))
                        );
                    }
                }
            }
        }
        let opts = GateOptions {
            max_cells: 255,
            max_depth: 8,
            min_width: 0.,
        };
        for env in [&all, &pruned] {
            let e = OverlapEnvelope::build(env, moving, opts).unwrap();
            assert!(
                e.lower_volume >= 0. && e.uncertain_volume >= 0. && e.upper_volume().is_finite()
            );
        }
    }
    #[test]
    fn pruned_and_unpruned_envelopes_bound_analytic_sphere_lens() {
        let tree = SphereTree::new(Shape {
            name: "sphere".into(),
            volume: 1.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })
        .unwrap();
        let moving = pose(0., 0., 0.);
        let rd = 0.5;
        let d = 2.;
        let radius: f64 = 1. + rd;
        let lens = std::f64::consts::PI * (4. * radius + d) * (2. * radius - d).powi(2) / 12.;
        let bodies = [pose(d, 0., 0.), pose(30., 0., 0.)];
        for count in [1, 2] {
            let env = Environment {
                tree: &tree,
                fixed: bodies[..count].iter().map(|p| Placed::new(*p)).collect(),
                labels: (0..count).map(|i| (i, [0; 3])).collect(),
                rd,
            };
            let e = OverlapEnvelope::build(
                &env,
                moving,
                GateOptions {
                    max_cells: 2047,
                    max_depth: 12,
                    min_width: 0.,
                },
            )
            .unwrap();
            assert!(e.lower_volume <= lens && lens <= e.upper_volume());
        }
    }
}
