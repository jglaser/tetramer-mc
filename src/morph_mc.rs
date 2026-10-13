//! Exact Metropolis sampler of identical rigid bodies in a spherical vessel with
//! morphometric solvation plus a pairwise atom square well:
//!   beta E = beta F_solv (exact union measures, `solvation`)
//!          - epsilon * #{atom pairs (i in A, j in B != A): |x_i - x_j| < r_i + r_j + width}
//! with exact atom hard cores (`SphereTree`) and an exact spherical wall (`Container`).
//! Moves of one non-fixed body chosen uniformly: a local move (Gaussian translation,
//! Cayley rotation with a symmetric Gaussian generator) or a uniform reinsertion (position
//! uniform in the vessel ball, Haar orientation). Both kernels are symmetric, so the
//! acceptance is min(1, exp(-beta dE)).
use crate::{
    geometry::{Placed, SphereTree},
    math::{
        Pose, Vec3, add, cayley, matmul, matvec, norm, quaternion, rotation, scale, sub, transpose,
    },
    morphometry::BallGrid,
    solvation::{SolvationModel, SolvationState},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Copy, Debug)]
pub struct SquareWell {
    /// kT per atom pair in range.
    pub epsilon: f64,
    /// Range beyond contact of the van der Waals spheres, A.
    pub width: f64,
}

/// A charged group of the body template: atom indices, sign and fractional charge.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ChargeGroup {
    pub atoms: Vec<usize>,
    pub sign: i8,
    pub fraction: f64,
}

/// Like-charge contact penalty:
///   + weight * sum f_g f_h over like-signed groups g, h on different bodies
///     whose closest atom centres are nearer than `cutoff`.
#[derive(Clone, Debug, Default)]
pub struct LikeCharge {
    /// kT per pair of full like charges in contact.
    pub weight: f64,
    /// Closest atom-centre distance for contact, A.
    pub cutoff: f64,
    pub groups: Vec<ChargeGroup>,
}

/// Finite memory: K auxiliary relative-pose slots m_k with the conditional law
///   q(m | x) = (1 - u) (1/N) sum_e kernel(m; g_e) + u floor(m)   (floor alone if N = 0),
/// where g_e runs over the relative poses of the N current contact entries (both orders),
/// kernel = normalized Gaussian in translation (sigma_translation) times a Gaussian in
/// exponential coordinates of the rotation (sigma_rotation <= 20 deg, so its truncation at pi
/// is below 1e-16), and floor = Gaussian in translation (floor_sigma) times uniform rotation.
/// The joint target is pi(x) prod_k q(m_k | x), whose x-marginal is pi. Physical moves carry
/// the factor prod_k q(m_k | x') / q(m_k | x); slots are refreshed by exact draws from
/// q(. | x) (Gibbs, always accepted); docking proposes from the library and the slots.
#[derive(Clone, Copy, Debug)]
pub struct MemorySettings {
    pub slots: usize,
    pub sigma_translation: f64,
    pub sigma_rotation: f64,
    pub uniform_weight: f64,
    pub floor_sigma: f64,
}

#[derive(Clone, Debug)]
struct Memory {
    s: MemorySettings,
    slots: Vec<Pose>,
    /// S_k = sum over current contact entries of kernel(m_k; g_e).
    sums: Vec<f64>,
    /// Number of contact entries N (twice the number of contacts).
    entries: usize,
}

/// Relative pose of `pj` in the frame of `pi`.
fn rel_pose(pi: Pose, pj: Pose) -> Pose {
    let ri = rotation(pi.orientation);
    Pose {
        position: matvec(transpose(ri), sub(pj.position, pi.position)),
        orientation: quaternion(matmul(transpose(ri), rotation(pj.orientation))),
    }
}

/// Rotation angle in [0, pi] of a rotation given by a (not necessarily canonical) quaternion.
fn rotation_angle(q: [f64; 4]) -> f64 {
    2. * q[0].abs().min(1.).acos()
}

impl MemorySettings {
    fn kernel(&self, m: Pose, g: Pose) -> f64 {
        let (st, sr) = (self.sigma_translation, self.sigma_rotation);
        let d = sub(m.position, g.position);
        let t2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if t2 > 144. * st * st {
            return 0.;
        }
        let delta = matmul(transpose(rotation(g.orientation)), rotation(m.orientation));
        let theta = rotation_angle(quaternion(delta));
        if theta > 12. * sr {
            return 0.;
        }
        let jac = if theta < 1e-8 {
            1.
        } else {
            theta * theta / (2. * (1. - theta.cos()))
        };
        let pi = std::f64::consts::PI;
        let c = (2. * pi * st * st).powf(-1.5) * 8. * pi * pi * (2. * pi * sr * sr).powf(-1.5);
        c * (-t2 / (2. * st * st) - theta * theta / (2. * sr * sr)).exp() * jac
    }
    fn floor(&self, m: Pose) -> f64 {
        let s2 = self.floor_sigma * self.floor_sigma;
        let t2 = m.position.iter().map(|x| x * x).sum::<f64>();
        (2. * std::f64::consts::PI * s2).powf(-1.5) * (-t2 / (2. * s2)).exp()
    }
    fn log_q(&self, m: Pose, sum: f64, entries: usize) -> f64 {
        if entries == 0 {
            self.floor(m).ln()
        } else {
            (self.uniform_weight * self.floor(m)
                + (1. - self.uniform_weight) * sum.max(0.) / entries as f64)
                .ln()
        }
    }
    /// Exact draw from q(. | x) given the contact entries of x.
    fn draw(&self, rng: &mut StdRng, entries: &[Pose]) -> Pose {
        let normal = |rng: &mut StdRng| -> f64 { StandardNormal.sample(rng) };
        if entries.is_empty() || rng.random::<f64>() < self.uniform_weight {
            let position: Vec3 = std::array::from_fn(|_| self.floor_sigma * normal(rng));
            let q: [f64; 4] = std::array::from_fn(|_| normal(rng));
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            return Pose {
                position,
                orientation: q.map(|x| x / n),
            };
        }
        let g = entries[rng.random_range(0..entries.len())];
        let w: Vec3 = loop {
            let w: Vec3 = std::array::from_fn(|_| self.sigma_rotation * normal(rng));
            if norm(w) < std::f64::consts::PI {
                break w;
            }
        };
        let theta = norm(w);
        let e = if theta < 1e-12 {
            [1., 0., 0., 0.]
        } else {
            let s = (theta / 2.).sin() / theta;
            [(theta / 2.).cos(), w[0] * s, w[1] * s, w[2] * s]
        };
        let position: Vec3 =
            std::array::from_fn(|k| g.position[k] + self.sigma_translation * normal(rng));
        Pose {
            position,
            orientation: quaternion(matmul(rotation(g.orientation), rotation(e))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoveSettings {
    pub translation: f64,
    pub rotation: f64,
    pub reinsertion_probability: f64,
}

/// Independent docking proposal: with probability `uniform_weight` a uniform pose in the
/// vessel ball (Haar orientation), otherwise a Gaussian about relative pose `components[k]`
/// (k uniform) in the frame of an anchor body chosen uniformly among the other bodies.
/// The full mixture density Q is evaluated at both endpoints (the anchor set is the same),
/// so the acceptance is min(1, exp(-beta dE) Q(old) / Q(new)).
///
/// With `learned`, G is instead a frozen learned relative-pose mixture (e.g. a blind contact
/// map): its exact independent draw and normalized density over the same anchors.
///
/// With `library` (requires `MorphMc::with_contact_library`), the components are instead the
/// relative poses of the current contacts between bodies other than the moved body b, in both
/// directions. They are a function of the other bodies only, which the move leaves unchanged,
/// so the same library enters Q at both endpoints and the move is Metropolis-within-Gibbs for
/// body b with an exact acceptance (formal/ReversibleSampling/FrozenContext.lean). An empty
/// library makes the proposal purely uniform.
#[derive(Clone)]
pub struct DockSettings {
    pub learned: Option<std::sync::Arc<crate::docking::DockingProposal>>,
    pub library: bool,
    pub components: Vec<Pose>,
    pub sigma_translation: f64,
    pub sigma_rotation: f64,
    pub uniform_weight: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MoveStats {
    pub dock_attempts: u64,
    pub dock_accepted: u64,
    pub attempts: u64,
    pub wall_rejected: u64,
    pub hard_rejected: u64,
    pub accepted: u64,
    pub reinsertion_attempts: u64,
    pub reinsertion_accepted: u64,
}

pub struct MorphMc {
    tree: SphereTree,
    wall: Container,
    solv: SolvationState,
    pub well: SquareWell,
    centers: Vec<Vec3>,
    vdw: Vec<f64>,
    template_grid: BallGrid,
    poses: Vec<Pose>,
    placed: Vec<Placed>,
    fixed: Vec<bool>,
    /// Square-well atom pairs between different bodies, each pair once.
    pair_count: i64,
    like: LikeCharge,
    /// Largest distance of a charged-group atom from the body origin.
    charge_extent: f64,
    /// sum f_g f_h over like-charge contacts between different bodies, each pair once.
    like_sum: f64,
    /// Contact library: atom-gap threshold, its grid, and the symmetric contact matrix.
    contact_gap: Option<f64>,
    contact_grid: Option<BallGrid>,
    contact: Vec<Vec<bool>>,
    memory: Option<Memory>,
    pub stats: MoveStats,
}

impl MorphMc {
    pub fn new(
        tree: SphereTree,
        wall: Container,
        model: SolvationModel,
        well: SquareWell,
        poses: Vec<Pose>,
        fixed: Vec<bool>,
    ) -> Result<Self> {
        ensure!(
            fixed.len() == poses.len(),
            "fixed flags and poses differ in length"
        );
        ensure!(
            well.epsilon.is_finite() && well.width >= 0.,
            "invalid square well"
        );
        let centers: Vec<Vec3> = tree.shape.atoms.iter().map(|a| a.center).collect();
        let vdw: Vec<f64> = tree.shape.atoms.iter().map(|a| a.radius).collect();
        let vmax = vdw.iter().copied().fold(0., f64::max);
        let template_grid = BallGrid::new(&centers, 2. * vmax + well.width);
        let solv = SolvationState::new(model, &centers, &vdw, &poses)?;
        let placed = poses.iter().map(|&p| Placed::new(p)).collect();
        let mut s = Self {
            tree,
            wall,
            solv,
            well,
            centers,
            vdw,
            template_grid,
            poses,
            placed,
            fixed,
            pair_count: 0,
            like: LikeCharge::default(),
            charge_extent: 0.,
            like_sum: 0.,
            contact_gap: None,
            contact_grid: None,
            contact: Vec::new(),
            memory: None,
            stats: MoveStats::default(),
        };
        for (b, &p) in s.poses.iter().enumerate() {
            ensure!(s.wall.contains(p), "body {b} outside the wall");
        }
        for a in 0..s.poses.len() {
            for b in a + 1..s.poses.len() {
                ensure!(
                    !s.tree.overlaps(&s.placed[a], &s.placed[b]),
                    "bodies {a} and {b} overlap"
                );
            }
        }
        let mut pairs = 0;
        for a in 0..s.poses.len() {
            pairs += s.well_pairs(a, s.poses[a]);
        }
        s.pair_count = (pairs / 2) as i64;
        Ok(s)
    }

    /// Switch on the like-charge contact penalty.
    pub fn with_like_charge(mut self, like: LikeCharge) -> Result<Self> {
        ensure!(
            like.weight.is_finite() && like.cutoff >= 0.,
            "invalid like-charge penalty"
        );
        for g in &like.groups {
            ensure!(
                !g.atoms.is_empty()
                    && g.atoms.iter().all(|&i| i < self.centers.len())
                    && (g.sign == 1 || g.sign == -1)
                    && g.fraction.is_finite(),
                "invalid charge group {g:?}"
            );
        }
        self.charge_extent = like
            .groups
            .iter()
            .flat_map(|g| g.atoms.iter().map(|&i| norm(self.centers[i])))
            .fold(0., f64::max);
        self.like = like;
        let mut sum = 0.;
        for a in 0..self.poses.len() {
            sum += self.like_pairs(a, self.poses[a]);
        }
        self.like_sum = sum / 2.;
        Ok(self)
    }

    /// Track contacts (some atom pair with |x_i - x_j| < r_i + r_j + gap) for library docking.
    pub fn with_contact_library(mut self, gap: f64) -> Result<Self> {
        ensure!(gap.is_finite() && gap >= 0., "invalid contact gap");
        let vmax = self.vdw.iter().copied().fold(0., f64::max);
        self.contact_grid = Some(BallGrid::new(&self.centers, 2. * vmax + gap));
        self.contact_gap = Some(gap);
        let n = self.poses.len();
        self.contact = vec![vec![false; n]; n];
        for a in 0..n {
            for c in a + 1..n {
                let k = self.in_contact(a, self.poses[a], c);
                self.contact[a][c] = k;
                self.contact[c][a] = k;
            }
        }
        Ok(self)
    }

    /// Whether body `a` placed at `p` is in contact with body `c` at its current pose.
    fn in_contact(&self, a: usize, p: Pose, c: usize) -> bool {
        let (Some(gap), Some(grid)) = (self.contact_gap, &self.contact_grid) else {
            return false;
        };
        let q = self.poses[c];
        if a == c || norm(sub(q.position, p.position)) > 2. * self.tree.bound + gap {
            return false;
        }
        let ra = rotation(p.orientation);
        let rct = transpose(rotation(q.orientation));
        self.centers.iter().enumerate().any(|(i, &x)| {
            let y = matvec(rct, sub(add(matvec(ra, x), p.position), q.position));
            norm(y) <= self.tree.bound + self.vdw[i] + gap
                && grid
                    .near(&y)
                    .any(|j| norm(sub(self.centers[j], y)) < self.vdw[i] + self.vdw[j] + gap)
        })
    }

    /// Refresh the contacts of body `b` after it moved.
    fn update_contacts(&mut self, b: usize) {
        if self.contact_gap.is_none() {
            return;
        }
        for c in 0..self.poses.len() {
            if c != b {
                let k = self.in_contact(b, self.poses[b], c);
                self.contact[b][c] = k;
                self.contact[c][b] = k;
            }
        }
    }

    /// Relative poses (j in the frame of i, both orders) of the current contacts between
    /// bodies other than `b`: a function of the other bodies only. With `b` out of range,
    /// all contacts.
    pub fn library(&self, b: usize) -> Vec<Pose> {
        let mut out = Vec::new();
        for i in 0..self.poses.len() {
            for j in i + 1..self.poses.len() {
                if i == b || j == b || !self.contact[i][j] {
                    continue;
                }
                out.push(rel_pose(self.poses[i], self.poses[j]));
                out.push(rel_pose(self.poses[j], self.poses[i]));
            }
        }
        out
    }

    /// Switch on finite memory (requires the contact library); slots start from exact draws
    /// of q(. | x).
    pub fn with_memory(mut self, s: MemorySettings, rng: &mut StdRng) -> Result<Self> {
        ensure!(self.contact_gap.is_some(), "memory needs with_contact_library");
        ensure!(
            s.slots > 0
                && s.sigma_translation > 0.
                && s.sigma_rotation > 0.
                && s.sigma_rotation <= 20f64.to_radians()
                && s.uniform_weight > 0.
                && s.uniform_weight < 1.
                && s.floor_sigma > 0.,
            "invalid memory settings"
        );
        let entries = self.library(usize::MAX);
        let slots: Vec<Pose> = (0..s.slots).map(|_| s.draw(rng, &entries)).collect();
        self.memory = Some(Memory {
            s,
            slots,
            sums: vec![0.; s.slots],
            entries: entries.len(),
        });
        self.recompute_memory();
        Ok(self)
    }

    /// Recompute the cached kernel sums exactly (bounds rounding drift).
    pub fn recompute_memory(&mut self) {
        let entries = self.library(usize::MAX);
        if let Some(mem) = &mut self.memory {
            mem.entries = entries.len();
            for (k, m) in mem.slots.iter().enumerate() {
                mem.sums[k] = entries.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            }
        }
    }

    /// Largest relative deviation of the cached kernel sums from a fresh evaluation.
    pub fn memory_sum_error(&self) -> f64 {
        let entries = self.library(usize::MAX);
        let Some(mem) = &self.memory else {
            return 0.;
        };
        let mut worst: f64 = if mem.entries == entries.len() { 0. } else { f64::INFINITY };
        for (k, m) in mem.slots.iter().enumerate() {
            let fresh: f64 = entries.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            worst = worst.max((fresh - mem.sums[k]).abs() / fresh.abs().max(1e-300));
        }
        worst
    }

    /// Gibbs refresh of one uniformly chosen slot: an exact draw from q(. | x).
    pub fn refresh_memory(&mut self, rng: &mut StdRng) {
        let entries = self.library(usize::MAX);
        if let Some(mem) = &mut self.memory {
            let k = rng.random_range(0..mem.slots.len());
            let m = mem.s.draw(rng, &entries);
            mem.slots[k] = m;
            mem.sums[k] = entries.iter().map(|&g| mem.s.kernel(m, g)).sum();
            mem.entries = entries.len();
        }
    }

    /// Memory slots (relative poses).
    pub fn memory_slots(&self) -> Option<&[Pose]> {
        self.memory.as_ref().map(|m| m.slots.as_slice())
    }

    /// Log of prod_k q(m_k | x') / q(m_k | x) for body `b` moving to `new`, with the
    /// updated sums, entry count and new contact partners of `b`. Without memory: zero.
    fn memory_delta(&self, b: usize, new: Pose) -> (f64, Vec<f64>, usize, Vec<usize>) {
        let Some(mem) = &self.memory else {
            return (0., Vec::new(), 0, Vec::new());
        };
        let old = self.poses[b];
        let partners: Vec<usize> = (0..self.poses.len())
            .filter(|&c| c != b && self.in_contact(b, new, c))
            .collect();
        let mut old_g = Vec::new();
        for c in 0..self.poses.len() {
            if c != b && self.contact[b][c] {
                old_g.push(rel_pose(old, self.poses[c]));
                old_g.push(rel_pose(self.poses[c], old));
            }
        }
        let mut new_g = Vec::new();
        for &c in &partners {
            new_g.push(rel_pose(new, self.poses[c]));
            new_g.push(rel_pose(self.poses[c], new));
        }
        let n_new = mem.entries + new_g.len() - old_g.len();
        let mut sums = mem.sums.clone();
        let mut d = 0.;
        for (k, m) in mem.slots.iter().enumerate() {
            let removed: f64 = old_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            let added: f64 = new_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            let s_new = (mem.sums[k] - removed + added).max(0.);
            d += mem.s.log_q(*m, s_new, n_new) - mem.s.log_q(*m, mem.sums[k], mem.entries);
            sums[k] = s_new;
        }
        (d, sums, n_new, partners)
    }

    /// Commit the contacts (and memory sums) of an accepted move of body `b`.
    fn commit_memory(&mut self, b: usize, sums: Vec<f64>, entries: usize, partners: &[usize]) {
        if let Some(mem) = &mut self.memory {
            mem.sums = sums;
            mem.entries = entries;
            for c in 0..self.poses.len() {
                if c != b {
                    let k = partners.contains(&c);
                    self.contact[b][c] = k;
                    self.contact[c][b] = k;
                }
            }
        } else {
            self.update_contacts(b);
        }
    }

    /// Number of contacts between different bodies (each pair once).
    pub fn contact_count(&self) -> usize {
        self.contact.iter().map(|r| r.iter().filter(|&&k| k).count()).sum::<usize>() / 2
    }

    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }
    pub fn like_energy(&self) -> f64 {
        self.like.weight * self.like_sum
    }

    /// sum f_g f_h over like-charge contacts between body `b` placed at `p` and all others.
    pub fn like_pairs(&self, b: usize, p: Pose) -> f64 {
        if self.like.weight == 0. || self.like.groups.is_empty() {
            return 0.;
        }
        let cut = self.like.cutoff;
        let reach = 2. * self.charge_extent + cut;
        let rb = rotation(p.orientation);
        let mut sum = 0.;
        for (c, q) in self.poses.iter().enumerate() {
            if c == b || norm(sub(q.position, p.position)) > reach {
                continue;
            }
            let rct = transpose(rotation(q.orientation));
            for g in &self.like.groups {
                // Atoms of group g of b in the frame of body c.
                let y: Vec<Vec3> = g
                    .atoms
                    .iter()
                    .map(|&i| matvec(rct, sub(add(matvec(rb, self.centers[i]), p.position), q.position)))
                    .collect();
                if y.iter().all(|&v| norm(v) > self.charge_extent + cut) {
                    continue;
                }
                for h in &self.like.groups {
                    if h.sign == g.sign
                        && y.iter().any(|&v| {
                            h.atoms
                                .iter()
                                .any(|&j| norm(sub(self.centers[j], v)) < cut)
                        })
                    {
                        sum += g.fraction * h.fraction;
                    }
                }
            }
        }
        sum
    }
    pub fn solvation_energy(&self) -> f64 {
        self.solv.energy()
    }
    pub fn well_energy(&self) -> f64 {
        -self.well.epsilon * self.pair_count as f64
    }
    pub fn pair_count(&self) -> i64 {
        self.pair_count
    }
    /// Change the well depth (replica exchange in epsilon); the configuration is kept.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.well.epsilon = epsilon;
    }

    /// Square-well atom pairs between body `b` placed at `p` and all other bodies.
    pub fn well_pairs(&self, b: usize, p: Pose) -> usize {
        let reach = 2. * self.tree.bound + self.well.width;
        let rb = rotation(p.orientation);
        let mut count = 0;
        for (c, q) in self.poses.iter().enumerate() {
            if c == b || norm(sub(q.position, p.position)) > reach {
                continue;
            }
            let rct = transpose(rotation(q.orientation));
            for (i, &x) in self.centers.iter().enumerate() {
                // Atom i of b in the frame of body c.
                let y = matvec(rct, sub(add(matvec(rb, x), p.position), q.position));
                if norm(y) > self.tree.bound + self.vdw[i] + self.well.width {
                    continue;
                }
                for j in self.template_grid.near(&y) {
                    if norm(sub(self.centers[j], y)) < self.vdw[i] + self.vdw[j] + self.well.width {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn hard_valid(&self, b: usize, p: Pose) -> bool {
        if !self.wall.contains(p) {
            return false;
        }
        let placed = Placed::new(p);
        self.poses.iter().enumerate().all(|(c, q)| {
            c == b
                || norm(sub(q.position, p.position)) > 2. * self.tree.bound
                || !self.tree.overlaps(&placed, &self.placed[c])
        })
    }

    /// One Metropolis step on a uniformly chosen non-fixed body. Returns true if accepted.
    pub fn step(&mut self, rng: &mut StdRng, mv: MoveSettings) -> Result<bool> {
        let free: Vec<usize> = (0..self.poses.len()).filter(|&b| !self.fixed[b]).collect();
        ensure!(!free.is_empty(), "no mobile body");
        let b = free[rng.random_range(0..free.len())];
        let old = self.poses[b];
        let reinsertion = rng.random::<f64>() < mv.reinsertion_probability;
        let new = if reinsertion {
            let r = self.wall.radius;
            let position = loop {
                let x: Vec3 = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * r);
                if norm(x) <= r {
                    break x;
                }
            };
            let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            Pose {
                position,
                orientation: q.map(|x| x / n),
            }
        } else {
            let d: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                z * mv.translation
            });
            let c: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                z * mv.rotation / 2.
            });
            Pose {
                position: add(old.position, d),
                orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
            }
        };
        self.stats.attempts += 1;
        self.stats.reinsertion_attempts += u64::from(reinsertion);
        if !self.wall.contains(new) {
            self.stats.wall_rejected += 1;
            return Ok(false);
        }
        if !self.hard_valid(b, new) {
            self.stats.hard_rejected += 1;
            return Ok(false);
        }
        let dp = self.well_pairs(b, new) as i64 - self.well_pairs(b, old) as i64;
        let d_well = -self.well.epsilon * dp as f64;
        let dl = self.like_pairs(b, new) - self.like_pairs(b, old);
        let update = self.solv.propose(&[(b, new)])?;
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        // Minus the log joint-target ratio: energy change minus the memory-conditional ratio.
        let de = update.delta + d_well + self.like.weight * dl - dm;
        ensure!(!de.is_nan(), "NaN energy change");
        if de <= 0. || rng.random::<f64>() < (-de).exp() {
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.commit_memory(b, sums, n_entries, &partners);
            self.stats.accepted += 1;
            self.stats.reinsertion_accepted += u64::from(reinsertion);
            return Ok(true);
        }
        Ok(false)
    }

    /// Log of the docking mixture density of pose `x` of body `b` (anchors: all others), with
    /// Gaussian components `comps` and uniform weight `u` (1 when there are no components).
    fn dock_log_density(&self, b: usize, x: Pose, d: &DockSettings, comps: &[Pose], u: f64) -> f64 {
        let r = self.wall.radius;
        let uniform = if norm(x.position) <= r {
            u.ln() - (4. / 3. * std::f64::consts::PI * r * r * r).ln()
        } else {
            f64::NEG_INFINITY
        };
        if let Some(m) = &d.learned {
            // G = (1/A) sum over all A anchors; anchors farther than 2 bound + 20 A carry
            // contributions below exp(-100) and are skipped, with the 1/A normalization kept.
            let all = self.poses.len() - 1;
            let cut = 2. * self.tree.bound + 20.;
            let near: Vec<Pose> = (0..self.poses.len())
                .filter(|&a| a != b && norm(sub(self.poses[a].position, x.position)) < cut)
                .map(|a| self.poses[a])
                .collect();
            let g = if near.is_empty() {
                f64::NEG_INFINITY
            } else {
                m.members_log_density_pruned(&[x], &near)
                    .unwrap_or(f64::NEG_INFINITY)
                    + (near.len() as f64).ln()
                    - (all as f64).ln()
            };
            let l = (1. - u).ln() + g;
            let hi = uniform.max(l);
            if !hi.is_finite() {
                return hi;
            }
            return hi + ((uniform - hi).exp() + (l - hi).exp()).ln();
        }
        let rx = rotation(x.orientation);
        let (st, sr) = (d.sigma_translation, d.sigma_rotation);
        let norm_t = -1.5 * (2. * std::f64::consts::PI * st * st).ln();
        let norm_r = (8. * std::f64::consts::PI * std::f64::consts::PI).ln()
            - 1.5 * (2. * std::f64::consts::PI * sr * sr).ln();
        let mut terms = Vec::new();
        let anchors = self.poses.len() - 1;
        if comps.is_empty() {
            return uniform;
        }
        let lw = (1. - u).ln() - ((anchors * comps.len()) as f64).ln();
        for (a, pa) in self.poses.iter().enumerate() {
            if a == b {
                continue;
            }
            let rat = transpose(rotation(pa.orientation));
            let trel = matvec(rat, sub(x.position, pa.position));
            let rrel = matmul(rat, rx);
            for c in comps {
                let dt = sub(trel, c.position);
                let t2 = dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2];
                if t2 > 100. * st * st {
                    continue;
                }
                let delta = matmul(transpose(rotation(c.orientation)), rrel);
                let q = quaternion(delta);
                let theta = 2. * q[0].abs().min(1.).acos();
                // Exponential coordinates: Haar factor theta^2 / (2 (1 - cos theta)).
                let jac = if theta < 1e-8 {
                    1.
                } else {
                    theta * theta / (2. * (1. - theta.cos()))
                };
                terms.push(
                    lw + norm_t - t2 / (2. * st * st) + norm_r - theta * theta / (2. * sr * sr)
                        + jac.ln(),
                );
            }
        }
        terms.push(uniform);
        let m = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return m;
        }
        m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln()
    }

    /// One docking Metropolis-Hastings step on a uniformly chosen non-fixed body.
    pub fn step_dock(&mut self, rng: &mut StdRng, d: &DockSettings) -> Result<bool> {
        let free: Vec<usize> = (0..self.poses.len()).filter(|&b| !self.fixed[b]).collect();
        ensure!(
            !free.is_empty()
                && self.poses.len() > 1
                && (d.learned.is_some() || d.library || !d.components.is_empty()),
            "invalid docking setup"
        );
        ensure!(
            !d.library || (self.contact_gap.is_some() && d.learned.is_none()),
            "library docking needs with_contact_library and no learned model"
        );
        ensure!(
            d.uniform_weight > 0. && d.uniform_weight < 1.,
            "docking uniform weight must lie in (0, 1)"
        );
        let b = free[rng.random_range(0..free.len())];
        let old = self.poses[b];
        self.stats.dock_attempts += 1;
        // Components: the contact library of the other bodies (unchanged by this move), or
        // the fixed list. Without components the proposal is uniform with probability one.
        let library = if d.library {
            let mut lib = self.library(b);
            if let Some(slots) = self.memory_slots() {
                lib.extend_from_slice(slots);
            }
            Some(lib)
        } else {
            None
        };
        let comps: &[Pose] = library.as_deref().unwrap_or(&d.components);
        let u = if d.learned.is_none() && comps.is_empty() { 1. } else { d.uniform_weight };
        let new = if rng.random::<f64>() < u {
            let r = self.wall.radius;
            let position = loop {
                let x: Vec3 = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * r);
                if norm(x) <= r {
                    break x;
                }
            };
            let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            Pose {
                position,
                orientation: q.map(|x| x / n),
            }
        } else if let Some(m) = &d.learned {
            let anchors: Vec<Pose> = (0..self.poses.len())
                .filter(|&a| a != b)
                .map(|a| self.poses[a])
                .collect();
            match m.draw_singleton_independent(rng, &anchors)?.0 {
                Some(p) => p,
                None => return Ok(false),
            }
        } else {
            let others: Vec<usize> = (0..self.poses.len()).filter(|&a| a != b).collect();
            let a = others[rng.random_range(0..others.len())];
            let c = comps[rng.random_range(0..comps.len())];
            let t: Vec3 = std::array::from_fn(|k| {
                let z: f64 = StandardNormal.sample(rng);
                c.position[k] + z * d.sigma_translation
            });
            let w: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                z * d.sigma_rotation
            });
            let theta = norm(w);
            if theta >= std::f64::consts::PI {
                return Ok(false);
            }
            let e = if theta < 1e-12 {
                [1., 0., 0., 0.]
            } else {
                let s = (theta / 2.).sin() / theta;
                [(theta / 2.).cos(), w[0] * s, w[1] * s, w[2] * s]
            };
            let rrel = matmul(rotation(c.orientation), rotation(e));
            let pa = self.poses[a];
            let ra = rotation(pa.orientation);
            Pose {
                position: add(pa.position, matvec(ra, t)),
                orientation: quaternion(matmul(ra, rrel)),
            }
        };
        self.stats.attempts += 1;
        if !self.wall.contains(new) {
            self.stats.wall_rejected += 1;
            return Ok(false);
        }
        if !self.hard_valid(b, new) {
            self.stats.hard_rejected += 1;
            return Ok(false);
        }
        let dp = self.well_pairs(b, new) as i64 - self.well_pairs(b, old) as i64;
        let d_well = -self.well.epsilon * dp as f64;
        let dl = self.like_pairs(b, new) - self.like_pairs(b, old);
        let update = self.solv.propose(&[(b, new)])?;
        let log_ratio =
            self.dock_log_density(b, old, d, comps, u) - self.dock_log_density(b, new, d, comps, u);
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        let a = -(update.delta + d_well + self.like.weight * dl) + dm + log_ratio;
        ensure!(!a.is_nan(), "NaN docking acceptance");
        if a >= 0. || rng.random::<f64>() < a.exp() {
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.commit_memory(b, sums, n_entries, &partners);
            self.stats.accepted += 1;
            self.stats.dock_accepted += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// Recompute both energies from scratch (for checks).
    pub fn recompute(&self) -> (f64, f64) {
        let mut pairs = 0.;
        for a in 0..self.poses.len() {
            pairs += self.well_pairs(a, self.poses[a]) as f64;
        }
        (self.solv.recompute(), -self.well.epsilon * pairs / 2.)
    }

    /// Recompute the like-charge energy from scratch (for checks).
    pub fn recompute_like(&self) -> f64 {
        let sum: f64 = (0..self.poses.len())
            .map(|a| self.like_pairs(a, self.poses[a]))
            .sum();
        self.like.weight * sum / 2.
    }
}

/// Uniform scale of a vector, re-exported for callers building start states.
pub fn scaled(v: Vec3, s: f64) -> Vec3 {
    scale(v, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Atom, Shape};
    use crate::morphometry::union_measures;
    use crate::solvation::GaussConvention;
    use rand::SeedableRng;

    /// One fixed ball at the centre and one mobile ball in a spherical vessel: the
    /// sampled centre distance must follow d^2 exp(-beta E(d)) on [2 r, R - r], with E
    /// from the exact two-ball union measures plus the square well.
    #[test]
    fn two_ball_distance_distribution_is_exact() {
        let r = 1.0;
        let probe = 1.4;
        let shape = Shape {
            name: "ball".into(),
            volume: 4. / 3. * std::f64::consts::PI,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        };
        let tree = SphereTree::new(shape).unwrap();
        let wall = Container::new(7.0, &tree).unwrap();
        let model = SolvationModel {
            probe,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 1.2,
            width: 0.6,
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [4., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let mut mc =
            MorphMc::new(tree, wall, model.clone(), well, start, vec![true, false]).unwrap();
        let energy = |d: f64| {
            let rr = r + probe;
            let one = union_measures(&[[0.; 3]], &[rr]).unwrap();
            let two = union_measures(&[[0.; 3], [d, 0., 0.]], &[rr, rr]).unwrap();
            let c = model.coefficients[0];
            let f = |m: &crate::morphometry::UnionMeasures| {
                model.pressure * m.volume + c[0] * m.area + c[1] * m.mean + c[2] * m.gauss
            };
            let sw = if d < 2. * r + well.width {
                -well.epsilon
            } else {
                0.
            };
            f(&two) - 2. * f(&one) + sw
        };
        let (lo, hi) = (2. * r, 7.0 - r);
        let bins = 20;
        // Expected bin probabilities by fine quadrature of d^2 exp(-E).
        let mut expect = vec![0.; bins];
        let fine = 4000;
        for k in 0..fine {
            let d = lo + (k as f64 + 0.5) * (hi - lo) / fine as f64;
            expect[((d - lo) / (hi - lo) * bins as f64) as usize] += d * d * (-energy(d)).exp();
        }
        let z: f64 = expect.iter().sum();
        expect.iter_mut().for_each(|e| *e /= z);
        let mut rng = StdRng::seed_from_u64(3);
        let mv = MoveSettings {
            translation: 0.6,
            rotation: 0.3,
            reinsertion_probability: 0.3,
        };
        let mut hist = vec![0u64; bins];
        let (burn, n) = (5_000, 400_000);
        for t in 0..burn + n {
            mc.step(&mut rng, mv).unwrap();
            if t >= burn {
                let d = norm(mc.poses()[1].position);
                hist[(((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1)] += 1;
            }
        }
        let (sf, sw) = mc.recompute();
        assert!((sf - mc.solvation_energy()).abs() < 1e-8 && (sw - mc.well_energy()).abs() < 1e-12);
        // Correlated samples: compare with a generous binomial tolerance.
        for k in 0..bins {
            eprintln!(
                "bin {k:2}: sampled {:.4} expected {:.4}",
                hist[k] as f64 / n as f64,
                expect[k]
            );
        }
        for k in 0..bins {
            let p = hist[k] as f64 / n as f64;
            let sd = (expect[k] * (1. - expect[k]) / n as f64).sqrt();
            let _ = sd;
            assert!(
                (p - expect[k]).abs() < 2.5e-3,
                "bin {k}: sampled {p:.4}, expected {:.4}",
                expect[k]
            );
        }
        assert!(
            expect[0] > 0.05,
            "contact bin must carry weight: {}",
            expect[0]
        );

        // Same target with docking proposals mixed in: Q at both endpoints must keep it exact.
        let dock = DockSettings {
            learned: None,
            library: false,
            components: vec![Pose {
                position: [2.4, 0., 0.],
                orientation: [1., 0., 0., 0.],
            }],
            sigma_translation: 0.3,
            sigma_rotation: 0.2,
            uniform_weight: 0.3,
        };
        let mut hist = vec![0u64; bins];
        for t in 0..burn + n {
            if t % 2 == 0 {
                mc.step_dock(&mut rng, &dock).unwrap();
            } else {
                mc.step(&mut rng, mv).unwrap();
            }
            if t >= burn {
                let d = norm(mc.poses()[1].position);
                hist[(((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1)] += 1;
            }
        }
        assert!(
            mc.stats.dock_accepted > 10_000,
            "docking must be accepted: {:?}",
            mc.stats
        );
        for k in 0..bins {
            let p = hist[k] as f64 / n as f64;
            eprintln!("dock bin {k:2}: sampled {p:.4} expected {:.4}", expect[k]);
            assert!(
                (p - expect[k]).abs() < 2.5e-3,
                "dock bin {k}: sampled {p:.4}, expected {:.4}",
                expect[k]
            );
        }
    }
    /// Two balls, each carrying one like-charged group: the penalty is a step
    /// weight f^2 [d < cutoff] on top of solvation and well; local and docking moves.
    #[test]
    fn like_charge_two_ball_distribution_is_exact() {
        let r = 1.0;
        let probe = 1.4;
        let shape = Shape {
            name: "ball".into(),
            volume: 4. / 3. * std::f64::consts::PI,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        };
        let tree = SphereTree::new(shape).unwrap();
        let wall = Container::new(7.0, &tree).unwrap();
        let model = SolvationModel {
            probe,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 1.2,
            width: 0.6,
        };
        let like = LikeCharge {
            weight: 2.5,
            cutoff: 3.4,
            groups: vec![ChargeGroup {
                atoms: vec![0],
                sign: 1,
                fraction: 0.8,
            }],
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [4., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let mut mc = MorphMc::new(tree, wall, model.clone(), well, start, vec![true, false])
            .unwrap()
            .with_like_charge(like.clone())
            .unwrap();
        let energy = |d: f64| {
            let rr = r + probe;
            let one = union_measures(&[[0.; 3]], &[rr]).unwrap();
            let two = union_measures(&[[0.; 3], [d, 0., 0.]], &[rr, rr]).unwrap();
            let c = model.coefficients[0];
            let f = |m: &crate::morphometry::UnionMeasures| {
                model.pressure * m.volume + c[0] * m.area + c[1] * m.mean + c[2] * m.gauss
            };
            let sw = if d < 2. * r + well.width {
                -well.epsilon
            } else {
                0.
            };
            let lc = if d < like.cutoff {
                like.weight * 0.64
            } else {
                0.
            };
            f(&two) - 2. * f(&one) + sw + lc
        };
        let (lo, hi) = (2. * r, 7.0 - r);
        let bins = 20;
        let mut expect = vec![0.; bins];
        let fine = 4000;
        for k in 0..fine {
            let d = lo + (k as f64 + 0.5) * (hi - lo) / fine as f64;
            expect[((d - lo) / (hi - lo) * bins as f64) as usize] += d * d * (-energy(d)).exp();
        }
        let z: f64 = expect.iter().sum();
        expect.iter_mut().for_each(|e| *e /= z);
        let mv = MoveSettings {
            translation: 0.6,
            rotation: 0.3,
            reinsertion_probability: 0.3,
        };
        let dock = DockSettings {
            learned: None,
            library: false,
            components: vec![Pose {
                position: [2.4, 0., 0.],
                orientation: [1., 0., 0., 0.],
            }],
            sigma_translation: 0.3,
            sigma_rotation: 0.2,
            uniform_weight: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(5);
        let mut hist = vec![0u64; bins];
        let (burn, n) = (5_000, 600_000);
        for t in 0..burn + n {
            if t % 2 == 0 {
                mc.step_dock(&mut rng, &dock).unwrap();
            } else {
                mc.step(&mut rng, mv).unwrap();
            }
            if t >= burn {
                let d = norm(mc.poses()[1].position);
                hist[(((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1)] += 1;
            }
        }
        assert!((mc.recompute_like() - mc.like_energy()).abs() < 1e-9);
        for k in 0..bins {
            let p = hist[k] as f64 / n as f64;
            eprintln!("like bin {k:2}: sampled {p:.4} expected {:.4}", expect[k]);
            assert!(
                (p - expect[k]).abs() < 2.5e-3,
                "like bin {k}: sampled {p:.4}, expected {:.4}",
                expect[k]
            );
        }
    }

    /// Three balls (one fixed, two mobile): library docking (proposals from the contact of the
    /// two bodies not being moved) must sample the same distribution as local moves alone.
    /// Compared: histograms of d(0,1), d(0,2) and d(1,2), and the contact fraction.
    #[test]
    fn library_docking_matches_local_moves() {
        let r = 1.0;
        let shape = Shape {
            name: "ball".into(),
            volume: 4. / 3. * std::f64::consts::PI,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        };
        let model = SolvationModel {
            probe: 1.4,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 1.5,
            width: 0.6,
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [3., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [-3., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let make = || {
            let tree = SphereTree::new(shape.clone()).unwrap();
            let wall = Container::new(5.0, &tree).unwrap();
            MorphMc::new(tree, wall, model.clone(), well, start.clone(), vec![true, false, false])
                .unwrap()
                .with_contact_library(0.6)
                .unwrap()
        };
        let mv = MoveSettings {
            translation: 0.5,
            rotation: 0.3,
            reinsertion_probability: 0.2,
        };
        let dock = DockSettings {
            learned: None,
            library: true,
            components: Vec::new(),
            sigma_translation: 0.3,
            sigma_rotation: 0.3,
            uniform_weight: 0.3,
        };
        let (bins, lo, hi) = (12usize, 2.0, 8.0);
        let observe = |mc: &MorphMc, h: &mut Vec<u64>, contacts: &mut u64| {
            let p = mc.poses();
            for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                let d = norm(sub(p[i].position, p[j].position));
                let bin = (((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1);
                h[k * bins + bin] += 1;
            }
            *contacts += mc.contact_count() as u64;
        };
        let (burn, n) = (10_000, 1_500_000);
        let mut out = Vec::new();
        for (seed, with_dock) in [(11u64, false), (12, true)] {
            let mut mc = make();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut h = vec![0u64; 3 * bins];
            let mut contacts = 0u64;
            for t in 0..burn + n {
                if with_dock && t % 2 == 0 {
                    mc.step_dock(&mut rng, &dock).unwrap();
                } else {
                    mc.step(&mut rng, mv).unwrap();
                }
                if t >= burn {
                    observe(&mc, &mut h, &mut contacts);
                }
            }
            if with_dock {
                assert!(
                    mc.stats.dock_accepted > 20_000,
                    "library docking must be accepted: {:?}",
                    mc.stats
                );
            }
            // The cached contact matrix must equal a fresh evaluation.
            let fresh = make();
            let mut check = fresh;
            check.poses = mc.poses.clone();
            check.placed = mc.placed.clone();
            let check = check.with_contact_library(0.6).unwrap();
            assert_eq!(check.contact, mc.contact, "cached contacts differ from recomputed");
            out.push((h, contacts as f64 / n as f64));
        }
        let (ha, ca) = &out[0];
        let (hb, cb) = &out[1];
        eprintln!("mean contacts: local {ca:.4}, with library docking {cb:.4}");
        for k in 0..3 * bins {
            let (pa, pb) = (ha[k] as f64 / n as f64, hb[k] as f64 / n as f64);
            eprintln!("bin {k:2}: local {pa:.4} library {pb:.4}");
            assert!(
                (pa - pb).abs() < 6e-3,
                "bin {k}: local {pa:.4} vs library docking {pb:.4}"
            );
        }
        assert!((ca - cb).abs() < 0.02, "contact fraction {ca:.4} vs {cb:.4}");
        assert!(*ca > 0.1, "contacts must carry weight: {ca}");
    }

    /// Finite memory: with slot refreshes and library+memory docking, the x-marginal must
    /// equal the local-move distribution (three balls), and the cached kernel sums must equal
    /// a fresh evaluation.
    #[test]
    fn memory_keeps_marginal() {
        let r = 1.0;
        let shape = Shape {
            name: "ball".into(),
            volume: 4. / 3. * std::f64::consts::PI,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        };
        let model = SolvationModel {
            probe: 1.4,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 1.5,
            width: 0.6,
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [3., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [-3., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let mem = MemorySettings {
            slots: 3,
            sigma_translation: 0.3,
            sigma_rotation: 10f64.to_radians(),
            uniform_weight: 0.3,
            floor_sigma: 3.,
        };
        let mv = MoveSettings {
            translation: 0.5,
            rotation: 0.3,
            reinsertion_probability: 0.2,
        };
        let dock = DockSettings {
            learned: None,
            library: true,
            components: Vec::new(),
            sigma_translation: 0.3,
            sigma_rotation: 0.3,
            uniform_weight: 0.3,
        };
        let (bins, lo, hi) = (12usize, 2.0, 8.0);
        let (burn, n) = (10_000, 1_000_000);
        let mut out = Vec::new();
        // Independent chains on each side; compare group means against the between-chain
        // spread (memory pins remembered contacts, so samples are more correlated).
        let chains = [(21u64, false), (26, false), (27, false), (22, true), (23, true), (24, true)];
        for (seed, with_memory) in chains {
            let tree = SphereTree::new(shape.clone()).unwrap();
            let wall = Container::new(5.0, &tree).unwrap();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut mc = MorphMc::new(tree, wall, model.clone(), well, start.clone(), vec![
                true, false, false,
            ])
            .unwrap()
            .with_contact_library(0.6)
            .unwrap();
            if with_memory {
                mc = mc.with_memory(mem, &mut rng).unwrap();
            }
            let mut h = vec![0u64; 3 * bins];
            let mut contacts = 0u64;
            for t in 0..burn + n {
                if with_memory {
                    match t % 3 {
                        0 => {
                            mc.step_dock(&mut rng, &dock).unwrap();
                        }
                        1 => {
                            mc.step(&mut rng, mv).unwrap();
                        }
                        _ => mc.refresh_memory(&mut rng),
                    }
                } else {
                    mc.step(&mut rng, mv).unwrap();
                }
                if t >= burn {
                    let p = mc.poses();
                    for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                        let d = norm(sub(p[i].position, p[j].position));
                        let bin = (((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1);
                        h[k * bins + bin] += 1;
                    }
                    contacts += mc.contact_count() as u64;
                }
            }
            if with_memory {
                assert!(
                    mc.memory_sum_error() < 1e-9,
                    "cached memory sums drift: {}",
                    mc.memory_sum_error()
                );
                assert!(mc.stats.dock_accepted > 10_000, "docking must be accepted: {:?}", mc.stats);
            }
            out.push((h, contacts as f64 / n as f64));
        }
        let stats = |group: &[usize], k: usize| -> (f64, f64) {
            let v: Vec<f64> = group.iter().map(|&c| out[c].0[k] as f64 / n as f64).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64;
            (m, (var / v.len() as f64).sqrt())
        };
        let (local, memory) = ([0usize, 1, 2], [3usize, 4, 5]);
        for c in 0..chains.len() {
            eprintln!("chain {c} (memory {}): bin0 {:.4} bin12 {:.4} contacts {:.4}", chains[c].1,
                out[c].0[0] as f64 / n as f64, out[c].0[12] as f64 / n as f64, out[c].1);
        }
        for k in 0..3 * bins {
            let ((ma, sa), (mb, sb)) = (stats(&local, k), stats(&memory, k));
            let tol = 4. * (sa * sa + sb * sb).sqrt() + 2e-3;
            eprintln!("bin {k:2}: local {ma:.4}+-{sa:.4} memory {mb:.4}+-{sb:.4} (tol {tol:.4})");
            assert!((ma - mb).abs() < tol, "bin {k}: local {ma:.4} vs memory {mb:.4}");
        }
        let ca = local.iter().map(|&c| out[c].1).sum::<f64>() / 3.;
        let cb = memory.iter().map(|&c| out[c].1).sum::<f64>() / 3.;
        eprintln!("mean contacts: local {ca:.4}, with memory {cb:.4}");
        assert!((ca - cb).abs() < 0.03, "contact fraction {ca:.4} vs {cb:.4}");
    }

    /// Docking from a learned relative-pose mixture (exact independent draw and density over
    /// all other bodies as anchors) keeps the analytic two-ball distance distribution.
    #[test]
    fn learned_docking_keeps_two_ball_distribution() {
        use crate::docking::{DockingMethod, DockingProposal};
        use crate::math::{IDENTITY, cayley};
        use crate::proposal::FrozenRelativePoseProposal;
        use crate::simulation::hash_bytes;
        let r = 1.0;
        let probe = 1.4;
        let shape = Shape {
            name: "ball".into(),
            volume: 4. / 3. * std::f64::consts::PI,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        };
        let tree = SphereTree::new(shape).unwrap();
        let sha = hash_bytes(&serde_json::to_vec(&tree.shape).unwrap());
        let cov = |d: [f64; 6]| -> [[f64; 6]; 6] {
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { d[i] * d[i] } else { 0. }))
        };
        let model = serde_json::json!({
            "angular_length": 1., "shape_sha256": sha, "coordinate_convention": "anchor-body-relative",
            "anchors": [{"position": [2.3, 0., 0.], "rotation": IDENTITY}, {"position": [0., 2.6, 0.], "rotation": cayley([0.3, -0.5, 0.2])}],
            "means": [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]],
            "covariances": [cov([0.3, 0.4, 0.35, 0.6, 0.5, 0.7]), cov([0.5, 0.3, 0.4, 0.4, 0.8, 0.6])],
            "weights": [0.4, 0.6]
        });
        let frozen =
            FrozenRelativePoseProposal::from_json_str_open(&model.to_string(), [14.; 3], 0.1, &sha)
                .unwrap();
        let learned =
            DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0.5, [0.; 3]).unwrap();
        let wall = Container::new(7.0, &tree).unwrap();
        let sm = SolvationModel {
            probe,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 1.2,
            width: 0.6,
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [4., 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let mut mc = MorphMc::new(tree, wall, sm.clone(), well, start, vec![true, false]).unwrap();
        let energy = |d: f64| {
            let rr = r + probe;
            let one = union_measures(&[[0.; 3]], &[rr]).unwrap();
            let two = union_measures(&[[0.; 3], [d, 0., 0.]], &[rr, rr]).unwrap();
            let c = sm.coefficients[0];
            let f = |m: &crate::morphometry::UnionMeasures| {
                sm.pressure * m.volume + c[0] * m.area + c[1] * m.mean + c[2] * m.gauss
            };
            let sw = if d < 2. * r + well.width {
                -well.epsilon
            } else {
                0.
            };
            f(&two) - 2. * f(&one) + sw
        };
        let (lo, hi) = (2. * r, 7.0 - r);
        let bins = 20;
        let mut expect = vec![0.; bins];
        let fine = 4000;
        for k in 0..fine {
            let d = lo + (k as f64 + 0.5) * (hi - lo) / fine as f64;
            expect[((d - lo) / (hi - lo) * bins as f64) as usize] += d * d * (-energy(d)).exp();
        }
        let z: f64 = expect.iter().sum();
        expect.iter_mut().for_each(|e| *e /= z);
        let dock = DockSettings {
            learned: Some(std::sync::Arc::new(learned)),
            library: false,
            components: Vec::new(),
            sigma_translation: 0.,
            sigma_rotation: 0.,
            uniform_weight: 0.3,
        };
        let mv = MoveSettings {
            translation: 0.6,
            rotation: 0.3,
            reinsertion_probability: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(9);
        let mut hist = vec![0u64; bins];
        let (burn, n) = (5_000, 1_200_000);
        for t in 0..burn + n {
            if t % 2 == 0 {
                mc.step_dock(&mut rng, &dock).unwrap();
            } else {
                mc.step(&mut rng, mv).unwrap();
            }
            if t >= burn {
                let d = norm(mc.poses()[1].position);
                hist[(((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1)] += 1;
            }
        }
        assert!(
            mc.stats.dock_accepted > 10_000,
            "learned docking must be accepted: {:?}",
            mc.stats
        );
        for k in 0..bins {
            eprintln!(
                "learned dock bin {k:2}: sampled {:.4} expected {:.4}",
                hist[k] as f64 / n as f64,
                expect[k]
            );
        }
        for k in 0..bins {
            let p = hist[k] as f64 / n as f64;
            assert!(
                (p - expect[k]).abs() < 2.5e-3,
                "learned dock bin {k}: sampled {p:.4}, expected {:.4}",
                expect[k]
            );
        }
    }
}
