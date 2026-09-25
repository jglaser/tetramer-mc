//! Proper common-isometry GCA in a protein-only spherical wall.
//!
//! The ideal bath permeates the wall. Its collapsed target is therefore
//! H_wall H_pairs exp(-z |union E_i|), without solvent-wall depletion.
//! Centered half-turns preserve chirality and the full atomic wall. Common
//! translations use the complete feasible wall chord and preserve all pair
//! interactions. Numerical certification failures are errors, never endpoint
//! rejection rules added to either rejection-free construction.
use crate::{
    geometry::{Atom, Placed, SphereTree},
    math::*,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::PI, time::Instant};

#[derive(Clone, Debug)]
pub struct Container {
    pub radius: f64,
    pub shape_bound: f64,
    atoms: Vec<Atom>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ShiftInterval {
    pub lower: f64,
    pub upper: f64,
    pub atom_constraints: usize,
    pub bodies_skipped: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ShiftOutcome {
    pub interval: ShiftInterval,
    pub displacement: Vec3,
    pub degenerate: bool,
}

impl Container {
    /// Owns its atom data; no lifetime is tied to the supplied immutable tree.
    pub fn new(radius: f64, tree: &SphereTree) -> Result<Self> {
        ensure!(
            radius.is_finite() && radius > 0. && (radius * radius).is_finite(),
            "invalid spherical wall radius"
        );
        ensure!(
            tree.shape.atoms.iter().all(|a| a.radius <= radius),
            "an atomic sphere exceeds the wall radius"
        );
        Ok(Self {
            radius,
            shape_bound: tree.bound,
            atoms: tree.shape.atoms.clone(),
        })
    }

    /// Every atom sphere must fit, including its physical core radius.
    pub fn contains(&self, pose: Pose) -> bool {
        if pose.validate().is_err() {
            return false;
        }
        let distance = norm(pose.position);
        let guard = 256. * f64::EPSILON * (1. + self.radius + self.shape_bound + distance);
        if distance + self.shape_bound + guard < self.radius {
            return true;
        }
        let center = matvec(transpose(rotation(pose.orientation)), pose.position);
        self.atoms.iter().all(|atom| {
            let available = self.radius - atom.radius;
            let c = add(center, atom.center);
            available >= 0. && dot(c, c) <= available * available
        })
    }

    pub fn clearance(&self, pose: Pose) -> f64 {
        if pose.validate().is_err() {
            return f64::NEG_INFINITY;
        }
        let center = matvec(transpose(rotation(pose.orientation)), pose.position);
        self.atoms
            .iter()
            .map(|a| self.radius - a.radius - norm(add(center, a.center)))
            .fold(f64::INFINITY, f64::min)
    }

    fn bounds_fit_shift(&self, pose: Pose, direction: Vec3, lower: f64, upper: f64) -> bool {
        let inner = self.radius - self.shape_bound;
        if inner <= 0. {
            return false;
        }
        [lower, upper].into_iter().all(|t| {
            let distance = norm(add(pose.position, scale(direction, t)));
            let guard = 256. * f64::EPSILON * (1. + self.radius + self.shape_bound + distance);
            distance.is_finite() && distance + guard < inner
        })
    }

    /// Intersect |R_i^-1 r_i + c_a + t R_i^-1 d|² <= (R-a)².
    /// A direction need not have unit length. No endpoint shrinkage is used.
    pub fn translation_interval(&self, state: &[Pose], direction: Vec3) -> Result<ShiftInterval> {
        self.translation_interval_impl(state, direction, true)
    }

    fn translation_interval_impl(
        &self,
        state: &[Pose],
        direction: Vec3,
        prune: bool,
    ) -> Result<ShiftInterval> {
        ensure!(
            !state.is_empty()
                && direction.iter().all(|v| v.is_finite())
                && dot(direction, direction).is_finite()
                && dot(direction, direction) > 0.,
            "invalid shift state/direction"
        );
        for pose in state {
            pose.validate()?;
        }
        let mut order: Vec<_> = state
            .iter()
            .enumerate()
            .map(|(i, p)| (i, norm(p.position)))
            .collect();
        order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut result = ShiftInterval {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            atom_constraints: 0,
            bodies_skipped: 0,
        };
        for (i, _) in order {
            let pose = state[i];
            if prune
                && result.lower.is_finite()
                && result.upper.is_finite()
                && self.bounds_fit_shift(pose, direction, result.lower, result.upper)
            {
                result.bodies_skipped += 1;
                continue;
            }
            let inverse = transpose(rotation(pose.orientation));
            let center = matvec(inverse, pose.position);
            let d = matvec(inverse, direction);
            let aa = dot(d, d);
            ensure!(
                aa.is_finite() && aa > 0.,
                "invalid body-frame direction for {i}"
            );
            for atom in &self.atoms {
                result.atom_constraints += 1;
                let available = self.radius - atom.radius;
                let c = add(center, atom.center);
                let bb = dot(c, d);
                let cc = dot(c, c) - available * available;
                ensure!(
                    available >= 0. && bb.is_finite() && cc.is_finite() && cc <= 0.,
                    "shift starts outside atomic wall or unsupported arithmetic: body {i}, C={cc}"
                );
                let discriminant = bb.mul_add(bb, -aa * cc);
                ensure!(
                    discriminant.is_finite() && discriminant >= 0.,
                    "invalid shift discriminant for {i}"
                );
                let far_numerator = -bb - discriminant.sqrt().copysign(bb);
                let (lower, upper) = if far_numerator == 0. {
                    ensure!(bb == 0. && cc == 0., "unresolved zero shift root for {i}");
                    (0., 0.)
                } else {
                    let far = far_numerator / aa;
                    let near = cc / far_numerator;
                    (far.min(near), far.max(near))
                };
                ensure!(
                    lower.is_finite() && upper.is_finite() && lower <= 0. && upper >= 0.,
                    "invalid finite shift roots for {i}"
                );
                result.lower = result.lower.max(lower);
                result.upper = result.upper.min(upper);
            }
        }
        ensure!(
            result.lower.is_finite()
                && result.upper.is_finite()
                && result.lower <= 0.
                && result.upper >= 0.
                && (result.upper - result.lower).is_finite(),
            "invalid common shift interval"
        );
        Ok(result)
    }

    /// Uniform conditional draw on the complete common-translation chord.
    /// The caller supplies an isotropic state-independent direction and open u.
    pub fn center_shift(
        &self,
        state: &mut [Pose],
        direction: Vec3,
        u: f64,
    ) -> Result<ShiftOutcome> {
        ensure!(
            u.is_finite() && u > 0. && u < 1.,
            "center shift requires open uniform u"
        );
        let interval = self.translation_interval(state, direction)?;
        if interval.lower == interval.upper {
            ensure!(
                interval.lower == 0.,
                "degenerate shift must retain its starting state"
            );
            return Ok(ShiftOutcome {
                interval,
                displacement: [0.; 3],
                degenerate: true,
            });
        }
        let t = interval.lower.mul_add(1. - u, interval.upper * u);
        let displacement = scale(direction, t);
        ensure!(
            t.is_finite() && displacement.iter().all(|x| x.is_finite()),
            "nonfinite common shift"
        );
        let next: Vec<_> = state
            .iter()
            .map(|pose| Pose {
                position: add(pose.position, displacement),
                ..*pose
            })
            .collect();
        for (i, &pose) in next.iter().enumerate() {
            ensure!(
                self.contains(pose),
                "numerical shift wall failure for {i}: clearance {}; no move committed",
                self.clearance(pose)
            );
        }
        state.copy_from_slice(&next);
        Ok(ShiftOutcome {
            interval,
            displacement,
            degenerate: false,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HalfTurn {
    matrix: Mat3,
}
impl HalfTurn {
    pub fn new(axis: Vec3) -> Result<Self> {
        let length = norm(axis);
        ensure!(
            axis.iter().all(|v| v.is_finite()) && length.is_finite() && length > 0.,
            "invalid half-turn axis"
        );
        // Direct division also handles a valid subnormal axis without an
        // overflowing reciprocal turning zero components into NaNs.
        let unit = axis.map(|v| v / length);
        Ok(Self {
            matrix: rotation([0., unit[0], unit[1], unit[2]]),
        })
    }
    pub fn point(&self, point: Vec3) -> Vec3 {
        matvec(self.matrix, point)
    }
    pub fn apply(&self, pose: Pose) -> Pose {
        Pose {
            position: self.point(pose.position),
            orientation: quaternion(matmul(self.matrix, rotation(pose.orientation))),
        }
    }
}

fn root(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = root(parent, parent[i]);
    }
    parent[i]
}
fn join(parent: &mut [usize], owners: &[usize]) {
    if let Some(&first) = owners.first() {
        for &i in &owners[1..] {
            let a = root(parent, first);
            let b = root(parent, i);
            parent[b] = a;
        }
    }
}
fn groups(parent: &mut [usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); parent.len()];
    for i in 0..parent.len() {
        let r = root(parent, i);
        out[r].push(i);
    }
    out.into_iter().filter(|g| !g.is_empty()).collect()
}

pub fn validate_state(tree: &SphereTree, wall: &Container, state: &[Pose]) -> Result<()> {
    ensure!(!state.is_empty(), "empty spherical state");
    for pose in state {
        pose.validate()?;
    }
    let placed: Vec<_> = state.iter().copied().map(Placed::new).collect();
    for (i, &pose) in state.iter().enumerate() {
        ensure!(wall.contains(pose), "body {i} is outside spherical wall");
        for j in i + 1..state.len() {
            ensure!(
                !tree.overlaps(&placed[i], &placed[j]),
                "spherical hard overlap {i},{j}"
            );
        }
    }
    Ok(())
}

/// Cross-overlap graph for a common involution. Edges constrain fair flips.
pub fn hard_components(
    tree: &SphereTree,
    state: &[Pose],
    transform: HalfTurn,
) -> Result<(Vec<Vec<usize>>, Vec<Pose>)> {
    ensure!(!state.is_empty(), "empty hard cluster state");
    for pose in state {
        pose.validate()?;
    }
    let shadow: Vec<_> = state.iter().map(|&p| transform.apply(p)).collect();
    let old: Vec<_> = state.iter().copied().map(Placed::new).collect();
    let new: Vec<_> = shadow.iter().copied().map(Placed::new).collect();
    let mut parent: Vec<_> = (0..state.len()).collect();
    for i in 0..state.len() {
        for j in i + 1..state.len() {
            // These two predicates agree in exact arithmetic. Checking both
            // avoids allowing a hard crossing if roundoff disagrees at contact.
            if tree.overlaps(&new[i], &old[j]) || tree.overlaps(&old[i], &new[j]) {
                join(&mut parent, &[i, j]);
            }
        }
    }
    Ok((groups(&mut parent), shadow))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GcaStats {
    pub hard_components: usize,
    pub components: usize,
    pub largest_component: usize,
    pub component_sizes: Vec<usize>,
    pub component_size_histogram: BTreeMap<usize, u64>,
    pub flipped_indices: Vec<usize>,
    pub flipped_particles: usize,
    pub partial_flip: bool,
    pub full_flip: bool,
    pub zero_flip: bool,
    pub poisson_probes: u64,
    pub owned_probes: u64,
    pub geometry_probes: u64,
    pub hyperedges: u64,
    pub max_hyperedge_size: usize,
    pub pair_envelopes: u64,
    pub skipped_envelopes: u64,
    pub hard_seconds: f64,
    pub sample_seconds: f64,
    pub total_seconds: f64,
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

struct Lens {
    midpoint: Vec3,
    axis: Vec3,
    u: Vec3,
    v: Vec3,
    half_length: f64,
    transverse_radius: f64,
    cover_radius: f64,
    volume: f64,
}
impl Lens {
    fn new(a: Vec3, b: Vec3, radius: f64) -> Result<Option<Self>> {
        let displacement = sub(b, a);
        let distance = norm(displacement);
        ensure!(distance.is_finite(), "nonfinite lens distance");
        if distance >= 2. * radius {
            return Ok(None);
        }
        let axis = if distance > 0. {
            scale(displacement, 1. / distance)
        } else {
            [1., 0., 0.]
        };
        let reference = if axis[0].abs() < 0.8 {
            [1., 0., 0.]
        } else {
            [0., 1., 0.]
        };
        let perpendicular = cross(axis, reference);
        let u = scale(perpendicular, 1. / norm(perpendicular));
        let v = cross(axis, u);
        let midpoint = add(scale(a, 0.5), scale(b, 0.5));
        let h = radius - 0.5 * distance;
        let rho = (h * (radius + 0.5 * distance)).sqrt();
        // Only the bounding lens determines ownership. Expanding its sampling
        // cylinder suppresses uncertain-boundary pruning; extra points thin out.
        let guard = 256. * f64::EPSILON * (1. + radius + norm(midpoint));
        let half_length = h + guard;
        let transverse_radius = rho + guard;
        let volume = 2. * PI * half_length * transverse_radius * transverse_radius;
        ensure!(
            volume.is_finite() && volume > 0.,
            "invalid lens envelope volume"
        );
        Ok(Some(Self {
            midpoint,
            axis,
            u,
            v,
            half_length,
            transverse_radius,
            cover_radius: rho + guard,
            volume,
        }))
    }
    fn draw(&self, rng: &mut StdRng) -> Vec3 {
        let s = (2. * rng.random::<f64>() - 1.) * self.half_length;
        let radial = rng.random::<f64>().sqrt() * self.transverse_radius;
        let angle = 2. * PI * rng.random::<f64>();
        add(
            add(self.midpoint, scale(self.axis, s)),
            add(
                scale(self.u, radial * angle.cos()),
                scale(self.v, radial * angle.sin()),
            ),
        )
    }
}

fn bounding_owners(point: Vec3, bodies: &[Placed], bound_squared: f64) -> Vec<usize> {
    bodies
        .iter()
        .enumerate()
        .filter_map(|(i, body)| {
            let d = sub(point, body.position);
            (dot(d, d) <= bound_squared).then_some(i)
        })
        .collect()
}

fn redundant_lens(
    lens: &Lens,
    first: usize,
    old: &[Placed],
    bound: f64,
    parent: &mut [usize],
) -> bool {
    let component = root(parent, first);
    old.iter().enumerate().all(|(i, body)| {
        if root(parent, i) == component {
            return true;
        }
        let guard = 256. * f64::EPSILON * (1. + bound + norm(body.position) + norm(lens.midpoint));
        norm(sub(body.position, lens.midpoint)) > bound + lens.cover_radius + guard
    })
}

fn contains(
    tree: &SphereTree,
    body: &Placed,
    point: Vec3,
    rd: f64,
    bound_squared: f64,
    stats: &mut GcaStats,
) -> bool {
    let d = sub(point, body.position);
    if dot(d, d) > bound_squared {
        return false;
    }
    stats.geometry_probes += 1;
    tree.contains(body.unapply(point), rd)
}

/// Collapsed explicit-bath GCA: reveal PPP(z) on old multiply-covered
/// exclusion regions outside the transformed union. Each point joins ALL old
/// owners. Lexicographic bounding-pair ownership prevents triple overcounting.
///
/// This kernel is reversible for each fixed axis. A caller may either select
/// an axis independently of the current state/model, or wrap state-dependent
/// selection in a proven forward/reverse correction (see conditional_axis).
/// Uncorrected favorable-axis selection is not valid. All particles are mobile.
/// The tree must be the same immutable shape used to construct the wall.
pub fn update(
    tree: &SphereTree,
    wall: &Container,
    state: &mut [Pose],
    transform: HalfTurn,
    rd: f64,
    activity: f64,
    rng: &mut StdRng,
) -> Result<GcaStats> {
    let start = Instant::now();
    ensure!(
        rd.is_finite() && rd >= 0. && activity.is_finite() && activity >= 0.,
        "invalid ideal bath"
    );
    validate_state(tree, wall, state)?;
    let hard_start = Instant::now();
    let (hard, shadow) = hard_components(tree, state, transform)?;
    let mut parent: Vec<_> = (0..state.len()).collect();
    for group in &hard {
        join(&mut parent, group);
    }
    let mut stats = GcaStats {
        hard_components: hard.len(),
        hard_seconds: hard_start.elapsed().as_secs_f64(),
        ..Default::default()
    };
    let sample_start = Instant::now();
    if activity > 0. && state.len() > 1 {
        let raw_bound = tree.bound + rd;
        let bound = raw_bound + 256. * f64::EPSILON * (1. + raw_bound + wall.radius);
        ensure!(
            bound.is_finite() && bound > 0. && (bound * bound).is_finite(),
            "invalid GCA exclusion bound"
        );
        let bound_squared = bound * bound;
        let old: Vec<_> = state.iter().copied().map(Placed::new).collect();
        let next: Vec<_> = shadow.iter().copied().map(Placed::new).collect();
        for i in 0..state.len() {
            for j in i + 1..state.len() {
                let Some(lens) = Lens::new(old[i].position, old[j].position, bound)? else {
                    continue;
                };
                stats.pair_envelopes += 1;
                if redundant_lens(&lens, i, &old, bound, &mut parent) {
                    stats.skipped_envelopes += 1;
                    continue;
                }
                let mean = activity * lens.volume;
                ensure!(
                    mean.is_finite() && mean < 9e15,
                    "unsupported GCA Poisson mean"
                );
                let count = if mean > 0. {
                    Poisson::<f64>::new(mean)?.sample(rng) as u64
                } else {
                    0
                };
                stats.poisson_probes += count;
                for _ in 0..count {
                    let point = lens.draw(rng);
                    let bounds = bounding_owners(point, &old, bound_squared);
                    if bounds.len() < 2 || bounds[0] != i || bounds[1] != j {
                        continue;
                    }
                    stats.owned_probes += 1;
                    if next
                        .iter()
                        .any(|body| contains(tree, body, point, rd, bound_squared, &mut stats))
                    {
                        continue;
                    }
                    let owners: Vec<_> = bounds
                        .into_iter()
                        .filter(|&k| contains(tree, &old[k], point, rd, bound_squared, &mut stats))
                        .collect();
                    if owners.len() >= 2 {
                        stats.hyperedges += 1;
                        stats.max_hyperedge_size = stats.max_hyperedge_size.max(owners.len());
                        join(&mut parent, &owners);
                    }
                }
            }
        }
    }
    stats.sample_seconds = sample_start.elapsed().as_secs_f64();
    let components = groups(&mut parent);
    stats.components = components.len();
    stats.largest_component = components.iter().map(Vec::len).max().unwrap();
    stats.component_sizes = components.iter().map(Vec::len).collect();
    for group in &components {
        *stats
            .component_size_histogram
            .entry(group.len())
            .or_default() += 1;
    }
    // A fixed N coin draws also reproduce the hard-only graph rule at z=0.
    let coins: Vec<bool> = (0..state.len()).map(|_| rng.random()).collect();
    let mut result = state.to_vec();
    for group in &components {
        if coins[group[0]] {
            for &i in group {
                result[i] = shadow[i];
                stats.flipped_indices.push(i);
            }
        }
    }
    stats.flipped_indices.sort_unstable();
    stats.flipped_particles = stats.flipped_indices.len();
    stats.zero_flip = stats.flipped_particles == 0;
    stats.full_flip = stats.flipped_particles == state.len();
    stats.partial_flip = !stats.zero_flip && !stats.full_flip;
    // This check is a numerical certificate. Failure aborts without mutating
    // the caller's state; it must not be converted into an endpoint veto.
    validate_state(tree, wall, &result)?;
    state.copy_from_slice(&result);
    stats.total_seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}
