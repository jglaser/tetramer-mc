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

/// Learned SE(3)-invariant interface chemistry residual model (Formulation 6b):
/// evaluates direct neutral and cationic H-bonds, bilateral-cationic anion-bridge triads,
/// water-bridged polar networks (gated by like-polar clashes), and unsatisfied buried polar
/// and charged desolvation penalties between interacting bodies.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct ChemistryModel {
    /// Per-atom 8-bit chemistry flags (length = number of body atoms, or empty if disabled).
    pub flags: Vec<u8>,
    /// Feature weights [w_hb, w_chb, w_cl, w_wb, w_clash, w_upol, w_uchg] in kT.
    pub weights: [f64; 7],
    /// Cap on raw water-bridge pair score before clash gating (default 5.0).
    pub wb_cap: f64,
}

const CHEM_CAT: u8 = 0x01;
const CHEM_ANI: u8 = 0x02;
const CHEM_NDON: u8 = 0x04;
const CHEM_NACC: u8 = 0x08;
const CHEM_TYR_OH: u8 = 0x10;
const CHEM_AMIDE_ND: u8 = 0x20;
const CHEM_PURE_ACC: u8 = 0x40;
const CHEM_PURE_DON: u8 = 0x80;

impl ChemistryModel {
    pub fn is_enabled(&self) -> bool {
        !self.flags.is_empty() && self.weights.iter().any(|&w| w != 0.)
    }

    #[inline]
    fn complementary(fi: u8, fj: u8) -> bool {
        ((fi & CHEM_NDON != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_NDON != 0))
            || ((fi & CHEM_CAT != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_CAT != 0))
            || ((fi & CHEM_CAT != 0) && (fj & CHEM_ANI != 0))
            || ((fi & CHEM_ANI != 0) && (fj & CHEM_CAT != 0))
            || ((fi & CHEM_ANI != 0) && (fj & CHEM_NDON != 0))
            || ((fi & CHEM_NDON != 0) && (fj & CHEM_ANI != 0))
    }

    /// Pairwise chemistry residual energy (kT) between body `p` and body `q` with template
    /// atom centres `centers` and bounding radius `bound`. Symmetric in `(p, q)`.
    pub fn pair_energy(&self, centers: &[Vec3], bound: f64, p: Pose, q: Pose) -> f64 {
        if !self.is_enabled() || norm(sub(q.position, p.position)) > 2. * bound + 5.25 {
            return 0.;
        }
        let n = centers.len();
        let rp = rotation(p.orientation);
        let rqt = transpose(rotation(q.orientation));
        let cut = 5.25;
        // Transform atoms of p into the frame of q, keeping those within bound + cut of origin.
        let mut active_a: Vec<(usize, Vec3)> = Vec::new();
        for (i, &x) in centers.iter().enumerate() {
            let y = matvec(rqt, sub(add(matvec(rp, x), p.position), q.position));
            if norm(y) <= bound + cut {
                active_a.push((i, y));
            }
        }
        if active_a.is_empty() {
            return 0.;
        }
        let rpt = transpose(rp);
        let rq = rotation(q.orientation);
        let mut active_b: Vec<usize> = Vec::new();
        for (j, &x) in centers.iter().enumerate() {
            let y = matvec(rpt, sub(add(matvec(rq, x), q.position), p.position));
            if norm(y) <= bound + cut {
                active_b.push(j);
            }
        }
        if active_b.is_empty() {
            return 0.;
        }
        let mut min_d_a = vec![99.0f64; n];
        let mut min_comp_a = vec![99.0f64; n];
        let mut min_d_b = vec![99.0f64; n];
        let mut min_comp_b = vec![99.0f64; n];

        let mut s_hb = 0.0f64;
        let mut s_chb = 0.0f64;
        let mut s_wb_raw = 0.0f64;
        let mut chb_mids: Vec<Vec3> = Vec::new();
        let mut tn_pairs: Vec<(Vec3, f64)> = Vec::new();
        let mut clash_candidates: Vec<(usize, usize, f64)> = Vec::new();

        for &(i, ya) in &active_a {
            let fi = self.flags[i];
            for &j in &active_b {
                let xb = centers[j];
                let d = norm(sub(ya, xb));
                if d >= cut {
                    continue;
                }
                if d < min_d_a[i] {
                    min_d_a[i] = d;
                }
                if d < min_d_b[j] {
                    min_d_b[j] = d;
                }
                let fj = self.flags[j];
                let comp = Self::complementary(fi, fj);
                if comp {
                    if d < min_comp_a[i] {
                        min_comp_a[i] = d;
                    }
                    if d < min_comp_b[j] {
                        min_comp_b[j] = d;
                    }
                    if (3.95..=5.25).contains(&d) {
                        let u = (d - 4.55) / 0.45;
                        s_wb_raw += (-u * u).exp();
                    }
                }
                if d < 3.9 {
                    let u = (d.max(2.7) - 2.95) / 0.45;
                    let g = (-u * u).exp();
                    if ((fi & CHEM_NDON != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_NDON != 0))
                    {
                        s_hb += g;
                    }
                    if ((fi & CHEM_CAT != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_CAT != 0))
                    {
                        s_chb += g;
                        if d < 3.6 {
                            chb_mids.push(scale(add(ya, xb), 0.5));
                        }
                    }
                }
                if (3.4..=4.5).contains(&d)
                    && (((fi & CHEM_TYR_OH != 0) && (fj & CHEM_AMIDE_ND != 0))
                        || ((fi & CHEM_AMIDE_ND != 0) && (fj & CHEM_TYR_OH != 0)))
                {
                    tn_pairs.push((scale(add(ya, xb), 0.5), d));
                }
                if d < 4.0
                    && (((fi & CHEM_PURE_ACC != 0) && (fj & CHEM_PURE_ACC != 0))
                        || ((fi & CHEM_PURE_DON != 0) && (fj & CHEM_PURE_DON != 0)))
                {
                    clash_candidates.push((i, j, d));
                }
            }
        }

        let mut s_cl = 0.0f64;
        if !tn_pairs.is_empty() && !chb_mids.is_empty() {
            let cats_a: Vec<Vec3> = centers
                .iter()
                .enumerate()
                .filter(|&(i, _)| self.flags[i] & CHEM_CAT != 0)
                .map(|(_, &x)| matvec(rqt, sub(add(matvec(rp, x), p.position), q.position)))
                .collect();
            let cats_b: Vec<Vec3> = centers
                .iter()
                .enumerate()
                .filter(|&(j, _)| self.flags[j] & CHEM_CAT != 0)
                .map(|(_, &x)| x)
                .collect();
            for (mid, d) in tn_pairs {
                let da = cats_a.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                let db = cats_b.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                if da < 8.0 && db < 8.0 {
                    let d_chb = chb_mids.iter().map(|&m| norm(sub(m, mid))).fold(99.0f64, f64::min);
                    if d_chb < 8.5 {
                        let u1 = (d - 4.05) / 0.40;
                        let u2 = (d_chb.max(5.0) - 5.0) / 3.0;
                        s_cl += (-u1 * u1 - u2 * u2).exp();
                    }
                }
            }
        }

        let mut s_clash = 0.0f64;
        for (i, j, d) in clash_candidates {
            if min_comp_a[i] >= 3.65 && min_comp_b[j] >= 3.65 {
                let u = (d.max(2.8) - 2.8) / 0.8;
                s_clash += (-u * u).exp();
            }
        }
        let s_wb = s_wb_raw.min(self.wb_cap) * (-1.5 * s_clash).exp();

        let mut unp_pol = 0.0f64;
        let mut unp_chg = 0.0f64;
        for &(i, _) in &active_a {
            let fi = self.flags[i];
            let dmin = min_d_a[i];
            let dcomp = min_comp_a[i];
            let is_chg = (fi & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fi & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 3.9 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.7;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.0 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.9;
                unp_chg += (-u * u).exp();
            }
        }
        for &j in &active_b {
            let fj = self.flags[j];
            let dmin = min_d_b[j];
            let dcomp = min_comp_b[j];
            let is_chg = (fj & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fj & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 3.9 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.7;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.0 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.9;
                unp_chg += (-u * u).exp();
            }
        }

        let w = &self.weights;
        -w[0] * s_hb - w[1] * s_chb - w[2] * s_cl - w[3] * s_wb
            + w[4] * s_clash
            + w[5] * unp_pol
            + w[6] * unp_chg
    }
}

/// Finite memory as a Poisson point process of relative poses with intensity
///   mu_x(m) = lambda_floor floor(m) + lambda_prior P(m) + lambda sum_e kernel(m; g_e),
/// where P(m) = sum_c w_c kernel(m; p_c) is an optional fixed prior (e.g. the centres of a
/// blind contact map, weights summing to one), independent of x,
/// where g_e runs over the relative poses of the N current contact entries (both orders),
/// kernel = normalized Gaussian in translation (sigma_translation) times a Gaussian in
/// exponential coordinates of the rotation (sigma_rotation <= 20 deg, so its truncation at pi
/// is below 1e-16), and floor = Gaussian in translation (floor_sigma) times uniform rotation.
/// Total intensity Lambda(x) = lambda_floor + lambda_prior + lambda N. The joint target is
/// pi(x) exp(-Lambda(x)) prod_{m in M} mu_x(m), whose x-marginal is pi. A physical move carries
/// exp(-lambda (N' - N)) prod_m mu_x'(m) / mu_x(m): a fixed cost per contact formed, and
/// coupling only to slots that sit on changed contacts. The memory is refreshed at fixed x by
/// Poisson thinning and superposition (each slot dies with probability `death`, then
/// Poisson(death Lambda) births from mu_x / Lambda), which is reversible for the Poisson law.
/// Docking proposes from the library and the slots.
#[derive(Clone, Copy, Debug)]
pub struct MemorySettings {
    pub lambda_floor: f64,
    pub lambda_prior: f64,
    pub lambda: f64,
    pub death: f64,
    pub sigma_translation: f64,
    pub sigma_rotation: f64,
    pub floor_sigma: f64,
}

#[derive(Clone, Debug)]
struct Memory {
    s: MemorySettings,
    /// Fixed prior centres and normalized weights.
    prior: Vec<(Pose, f64)>,
    /// P(m_k) for every slot (independent of x).
    prior_values: Vec<f64>,
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
    /// Log intensity at `m` given its prior value and kernel sum over the contact entries.
    fn log_mu(&self, m: Pose, prior: f64, sum: f64) -> f64 {
        (self.lambda_floor * self.floor(m) + self.lambda_prior * prior + self.lambda * sum.max(0.))
            .ln()
    }
    fn total(&self, entries: usize) -> f64 {
        self.lambda_floor + self.lambda_prior + self.lambda * entries as f64
    }
    fn prior_value(&self, prior: &[(Pose, f64)], m: Pose) -> f64 {
        prior.iter().map(|&(c, w)| w * self.kernel(m, c)).sum()
    }
    /// Number of births: Poisson with the given mean.
    fn poisson(rng: &mut StdRng, mean: f64) -> usize {
        if mean <= 0. {
            return 0;
        }
        let n: f64 = rand_distr::Poisson::new(mean).expect("finite mean").sample(rng);
        n as usize
    }
    /// Exact draw from mu_x / Lambda(x) given the contact entries of x and the prior.
    fn draw(&self, rng: &mut StdRng, entries: &[Pose], prior: &[(Pose, f64)]) -> Pose {
        let normal = |rng: &mut StdRng| -> f64 { StandardNormal.sample(rng) };
        let total = self.total(entries.len());
        let u = rng.random::<f64>() * total;
        let use_floor = u < self.lambda_floor || (entries.is_empty() && prior.is_empty());
        let use_prior = !use_floor && u < self.lambda_floor + self.lambda_prior && !prior.is_empty();
        if use_floor || (!use_prior && entries.is_empty()) {
            let position: Vec3 = std::array::from_fn(|_| self.floor_sigma * normal(rng));
            let q: [f64; 4] = std::array::from_fn(|_| normal(rng));
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            return Pose {
                position,
                orientation: q.map(|x| x / n),
            };
        }
        let g = if use_prior {
            let mut v = rng.random::<f64>();
            let mut pick = prior[prior.len() - 1].0;
            for &(c, w) in prior {
                if v < w {
                    pick = c;
                    break;
                }
                v -= w;
            }
            pick
        } else {
            entries[rng.random_range(0..entries.len())]
        };
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

/// Settings for Fortuin-Kasteleyn (FK) rigid cluster moves and proper half-turn GCA
/// (`formal/ReversibleSampling/FkClusterMove.lean`).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClusterSettings {
    /// Maximum cluster size `k_max` allowed for `step_cluster` (0 disables size cap).
    pub max_size: usize,
    /// Inverse-temperature scale `beta_fk` for FK bond activation `p_ab = 1 - exp(beta_fk * min(0, E_ab))`.
    /// `1.0` uses the exact pairwise surrogate energy; `0.0` disables multi-body recruitment (singleton only).
    pub fk_beta: f64,
    /// Probability that an FK cluster move proposes a rigid docking move (`step_cluster` dock branch).
    pub dock_probability: f64,
    /// Probability that a non-docking FK cluster move proposes a rigid uniform reinsertion of the handle.
    pub reinsertion_probability: f64,
    /// Base COM translation step size (Å) for local cluster diffusion (scaled by `|C|^-0.5` if `size_scaling` is true).
    pub translation: f64,
    /// Base COM rotation step size (radians) for local cluster diffusion (scaled by `|C|^-0.5` if `size_scaling` is true).
    pub rotation: f64,
    /// Scale local cluster translation/rotation step sizes by `|C|^-0.5` (`true` by default).
    pub size_scaling: bool,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            max_size: 6,
            fk_beta: 1.0,
            dock_probability: 0.5,
            reinsertion_probability: 0.2,
            translation: 0.5,
            rotation: 10f64.to_radians(),
            size_scaling: true,
        }
    }
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

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MoveStats {
    pub dock_attempts: u64,
    pub dock_accepted: u64,
    pub attempts: u64,
    pub wall_rejected: u64,
    pub hard_rejected: u64,
    pub accepted: u64,
    pub reinsertion_attempts: u64,
    pub reinsertion_accepted: u64,
    pub cluster_attempts: u64,
    pub cluster_accepted: u64,
    pub cluster_multibody_attempts: u64,
    pub cluster_multibody_accepted: u64,
    pub cluster_dock_attempts: u64,
    pub cluster_dock_accepted: u64,
    pub cluster_reinsertion_attempts: u64,
    pub cluster_reinsertion_accepted: u64,
    pub cluster_pinned_rejected: u64,
    pub cluster_cap_rejected: u64,
    pub gca_attempts: u64,
    pub gca_accepted: u64,
    pub gca_moved_bodies: u64,
    pub center_shift_attempts: u64,
    pub center_shift_accepted: u64,
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
    coordinate_wall_center: Vec3,
    /// Square-well atom pairs between different bodies, each pair once.
    pair_count: i64,
    like: LikeCharge,
    /// Largest distance of a charged-group atom from the body origin.
    charge_extent: f64,
    /// sum f_g f_h over like-charge contacts between different bodies, each pair once.
    like_sum: f64,
    /// Learned SE(3)-invariant interface chemistry residual model and cached total energy.
    chem: ChemistryModel,
    chem_sum: f64,
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
            coordinate_wall_center: [0.; 3],
            pair_count: 0,
            like: LikeCharge::default(),
            charge_extent: 0.,
            like_sum: 0.,
            chem: ChemistryModel::default(),
            chem_sum: 0.,
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

    /// Switch on the learned interface chemistry residual model.
    pub fn with_chemistry_model(mut self, chem: ChemistryModel) -> Result<Self> {
        ensure!(
            chem.flags.is_empty() || chem.flags.len() == self.centers.len(),
            "chemistry model flags length ({}) does not match body atoms ({})",
            chem.flags.len(),
            self.centers.len()
        );
        ensure!(
            chem.weights.iter().all(|w| w.is_finite()) && chem.wb_cap.is_finite() && chem.wb_cap >= 0.,
            "invalid chemistry model weights"
        );
        self.chem = chem;
        let mut sum = 0.;
        for a in 0..self.poses.len() {
            sum += self.chem_pairs(a, self.poses[a]);
        }
        self.chem_sum = sum / 2.;
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
        if self.contact.is_empty() {
            return Vec::new();
        }
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

    /// Switch on finite memory (requires the contact library); the memory starts from an
    /// exact draw of the Poisson process with intensity mu_x.
    pub fn with_memory(
        mut self,
        s: MemorySettings,
        prior: Vec<(Pose, f64)>,
        rng: &mut StdRng,
    ) -> Result<Self> {
        ensure!(self.contact_gap.is_some(), "memory needs with_contact_library");
        let wsum: f64 = prior.iter().map(|p| p.1).sum();
        ensure!(
            prior.iter().all(|p| p.1 >= 0. && p.1.is_finite())
                && (prior.is_empty() || wsum > 0.)
                && (s.lambda_prior == 0. || !prior.is_empty()),
            "invalid memory prior"
        );
        let prior: Vec<(Pose, f64)> = prior.into_iter().map(|(c, w)| (c, w / wsum)).collect();
        ensure!(
            s.lambda_floor > 0.
                && s.lambda_prior >= 0.
                && s.lambda >= 0.
                && s.death > 0.
                && s.death <= 1.
                && s.sigma_translation > 0.
                && s.sigma_rotation > 0.
                && s.sigma_rotation <= 20f64.to_radians()
                && s.floor_sigma > 0.,
            "invalid memory settings"
        );
        let entries = self.library(usize::MAX);
        let n0 = MemorySettings::poisson(rng, s.total(entries.len()));
        let slots: Vec<Pose> = (0..n0).map(|_| s.draw(rng, &entries, &prior)).collect();
        let prior_values = slots.iter().map(|&m| s.prior_value(&prior, m)).collect();
        self.memory = Some(Memory {
            s,
            prior,
            prior_values,
            sums: vec![0.; slots.len()],
            slots,
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

    /// Memory refresh at fixed configuration: Poisson thinning (each slot dies with
    /// probability `death`) and superposition (Poisson(death Lambda) births from mu_x /
    /// Lambda). Reversible for the Poisson process with intensity mu_x.
    pub fn refresh_memory(&mut self, rng: &mut StdRng) {
        let entries = self.library(usize::MAX);
        if let Some(mem) = &mut self.memory {
            let mut slots = Vec::with_capacity(mem.slots.len());
            let mut sums = Vec::with_capacity(mem.slots.len());
            let mut prior_values = Vec::with_capacity(mem.slots.len());
            for ((m, sum), pv) in mem.slots.iter().zip(&mem.sums).zip(&mem.prior_values) {
                if rng.random::<f64>() >= mem.s.death {
                    slots.push(*m);
                    sums.push(*sum);
                    prior_values.push(*pv);
                }
            }
            let births = MemorySettings::poisson(rng, mem.s.death * mem.s.total(entries.len()));
            for _ in 0..births {
                let m = mem.s.draw(rng, &entries, &mem.prior);
                sums.push(entries.iter().map(|&g| mem.s.kernel(m, g)).sum());
                prior_values.push(mem.s.prior_value(&mem.prior, m));
                slots.push(m);
            }
            mem.slots = slots;
            mem.sums = sums;
            mem.prior_values = prior_values;
            mem.entries = entries.len();
        }
    }

    /// Memory slots (relative poses).
    pub fn memory_slots(&self) -> Option<&[Pose]> {
        self.memory.as_ref().map(|m| m.slots.as_slice())
    }

    /// Log of the memory factor exp(-lambda (N' - N)) prod_m mu_x'(m) / mu_x(m) for body `b`
    /// moving to `new`, with the updated sums, entry count and new contact partners of `b`.
    /// Without memory: zero.
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
        let mut d = -mem.s.lambda * (n_new as f64 - mem.entries as f64);
        for (k, m) in mem.slots.iter().enumerate() {
            if old_g.is_empty() && new_g.is_empty() {
                break;
            }
            let removed: f64 = old_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            let added: f64 = new_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            if removed == 0. && added == 0. {
                continue;
            }
            let s_new = (mem.sums[k] - removed + added).max(0.);
            let pv = mem.prior_values[k];
            d += mem.s.log_mu(*m, pv, s_new) - mem.s.log_mu(*m, pv, mem.sums[k]);
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
    pub fn chem_energy(&self) -> f64 {
        self.chem_sum
    }

    /// Sum of pairwise chemistry residual energy between body `b` placed at `p` and all others.
    pub fn chem_pairs(&self, b: usize, p: Pose) -> f64 {
        if !self.chem.is_enabled() {
            return 0.;
        }
        let mut sum = 0.;
        for (c, &q) in self.poses.iter().enumerate() {
            if c != b {
                sum += self.chem.pair_energy(&self.centers, self.tree.bound, p, q);
            }
        }
        sum
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
        let dc = self.chem_pairs(b, new) - self.chem_pairs(b, old);
        let update = self.solv.propose(&[(b, new)])?;
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        // Minus the log joint-target ratio: energy change minus the memory-conditional ratio.
        let de = update.delta + d_well + self.like.weight * dl + dc - dm;
        ensure!(!de.is_nan(), "NaN energy change");
        if de <= 0. || rng.random::<f64>() < (-de).exp() {
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.chem_sum += dc;
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
        let max_anchor_dist = 2. * self.tree.bound + self.contact_gap.unwrap_or(6.) + 10. * st;
        for (a, pa) in self.poses.iter().enumerate() {
            if a == b || norm(sub(x.position, pa.position)) > max_anchor_dist {
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
        let dc = self.chem_pairs(b, new) - self.chem_pairs(b, old);
        let update = self.solv.propose(&[(b, new)])?;
        let log_ratio =
            self.dock_log_density(b, old, d, comps, u) - self.dock_log_density(b, new, d, comps, u);
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        let a = -(update.delta + d_well + self.like.weight * dl + dc) + dm + log_ratio;
        ensure!(!a.is_nan(), "NaN docking acceptance");
        if a >= 0. || rng.random::<f64>() < a.exp() {
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.chem_sum += dc;
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.commit_memory(b, sums, n_entries, &partners);
            self.stats.accepted += 1;
            self.stats.dock_accepted += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// Pairwise square-well atom contacts between pose `p` and pose `q`.
    pub fn well_pair_between(&self, p: Pose, q: Pose) -> usize {
        let reach = 2. * self.tree.bound + self.well.width;
        if norm(sub(q.position, p.position)) > reach {
            return 0;
        }
        let rb = rotation(p.orientation);
        let rct = transpose(rotation(q.orientation));
        let mut count = 0;
        for (i, &x) in self.centers.iter().enumerate() {
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
        count
    }

    /// Pairwise like-charge contact sum `sum f_g f_h` between pose `p` and pose `q`.
    pub fn like_pair_between(&self, p: Pose, q: Pose) -> f64 {
        if self.like.weight == 0. || self.like.groups.is_empty() {
            return 0.;
        }
        let cut = self.like.cutoff;
        let reach = 2. * self.charge_extent + cut;
        if norm(sub(q.position, p.position)) > reach {
            return 0.;
        }
        let rb = rotation(p.orientation);
        let rct = transpose(rotation(q.orientation));
        let mut sum = 0.;
        for g in &self.like.groups {
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
        sum
    }

    /// Pairwise chemistry residual energy between pose `p` and pose `q`.
    pub fn chem_pair_between(&self, p: Pose, q: Pose) -> f64 {
        self.chem.pair_energy(&self.centers, self.tree.bound, p, q)
    }

    /// Total pairwise surrogate energy `E_well + E_like + dG_chem` between pose `p` and pose `q`.
    pub fn pair_surrogate_energy(&self, p: Pose, q: Pose) -> f64 {
        let wp = self.well_pair_between(p, q) as f64;
        let lk = self.like_pair_between(p, q);
        let ch = self.chem_pair_between(p, q);
        -self.well.epsilon * wp + self.like.weight * lk + ch
    }

    /// Whether pose `p` and pose `q` are in contact according to `self.contact_gap`.
    fn in_contact_poses(&self, p: Pose, q: Pose) -> bool {
        let (Some(gap), Some(grid)) = (self.contact_gap, &self.contact_grid) else {
            return false;
        };
        if norm(sub(q.position, p.position)) > 2. * self.tree.bound + gap {
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

    /// Relative poses (j in the frame of i, both orders) of the current contacts between
    /// spectator bodies outside `in_cluster`: invariant under any move of `in_cluster`.
    pub fn library_excluding(&self, in_cluster: &[bool]) -> Vec<Pose> {
        if self.contact.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for i in 0..self.poses.len() {
            if in_cluster[i] {
                continue;
            }
            for j in i + 1..self.poses.len() {
                if in_cluster[j] || !self.contact[i][j] {
                    continue;
                }
                out.push(rel_pose(self.poses[i], self.poses[j]));
                out.push(rel_pose(self.poses[j], self.poses[i]));
            }
        }
        out
    }

    /// Log of the docking mixture density of pose `x` of handle `b` anchored only on
    /// spectator bodies outside `in_cluster`.
    fn dock_log_density_subset(
        &self,
        in_cluster: &[bool],
        x: Pose,
        d: &DockSettings,
        comps: &[Pose],
        u: f64,
    ) -> f64 {
        let r = self.wall.radius;
        let uniform = if norm(x.position) <= r {
            u.ln() - (4. / 3. * std::f64::consts::PI * r * r * r).ln()
        } else {
            f64::NEG_INFINITY
        };
        let spectators: Vec<usize> = (0..self.poses.len()).filter(|&a| !in_cluster[a]).collect();
        if spectators.is_empty() {
            return uniform;
        }
        if let Some(m) = &d.learned {
            let all = spectators.len();
            let cut = 2. * self.tree.bound + 20.;
            let near: Vec<Pose> = spectators
                .iter()
                .copied()
                .filter(|&a| norm(sub(self.poses[a].position, x.position)) < cut)
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
        if comps.is_empty() {
            return uniform;
        }
        let rx = rotation(x.orientation);
        let (st, sr) = (d.sigma_translation, d.sigma_rotation);
        let norm_t = -1.5 * (2. * std::f64::consts::PI * st * st).ln();
        let norm_r = (8. * std::f64::consts::PI * std::f64::consts::PI).ln()
            - 1.5 * (2. * std::f64::consts::PI * sr * sr).ln();
        let mut terms = Vec::new();
        let lw = (1. - u).ln() - ((spectators.len() * comps.len()) as f64).ln();
        let max_anchor_dist = 2. * self.tree.bound + self.contact_gap.unwrap_or(6.) + 10. * st;
        for &a in &spectators {
            let pa = self.poses[a];
            if norm(sub(x.position, pa.position)) > max_anchor_dist {
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

    /// Multi-body memory delta and contact refresh for a subset `moves: &[(usize, Pose)]`.
    fn memory_delta_multi(
        &self,
        in_cluster: &[bool],
        moves: &[(usize, Pose)],
    ) -> (f64, Vec<f64>, usize, Vec<(usize, usize, bool)>) {
        if self.contact_gap.is_none() {
            return (0., Vec::new(), 0, Vec::new());
        }
        let n = self.poses.len();
        let mut new_poses = self.poses.clone();
        for &(b, p) in moves {
            new_poses[b] = p;
        }
        let mut updates = Vec::new();
        let mut old_g = Vec::new();
        let mut new_g = Vec::new();
        for &(b, p_new) in moves {
            let p_old = self.poses[b];
            for c in 0..n {
                if c == b || in_cluster[c] {
                    continue;
                }
                let k_old = self.contact[b][c];
                let q_new = new_poses[c];
                let k_new = self.in_contact_poses(p_new, q_new);
                if k_old != k_new {
                    updates.push((b, c, k_new));
                }
                if self.memory.is_some() {
                    let q_old = self.poses[c];
                    if k_old {
                        old_g.push(rel_pose(p_old, q_old));
                        old_g.push(rel_pose(q_old, p_old));
                    }
                    if k_new {
                        new_g.push(rel_pose(p_new, q_new));
                        new_g.push(rel_pose(q_new, p_new));
                    }
                }
            }
        }
        let Some(mem) = &self.memory else {
            return (0., Vec::new(), 0, updates);
        };
        let n_new = mem.entries + new_g.len() - old_g.len();
        let mut sums = mem.sums.clone();
        let mut d = -mem.s.lambda * (n_new as f64 - mem.entries as f64);
        for (k, m) in mem.slots.iter().enumerate() {
            if old_g.is_empty() && new_g.is_empty() {
                break;
            }
            let removed: f64 = old_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            let added: f64 = new_g.iter().map(|&g| mem.s.kernel(*m, g)).sum();
            if removed == 0. && added == 0. {
                continue;
            }
            let s_new = (mem.sums[k] - removed + added).max(0.);
            let pv = mem.prior_values[k];
            d += mem.s.log_mu(*m, pv, s_new) - mem.s.log_mu(*m, pv, mem.sums[k]);
            sums[k] = s_new;
        }
        (d, sums, n_new, updates)
    }

    fn commit_memory_multi(
        &mut self,
        sums: Vec<f64>,
        entries: usize,
        updates: &[(usize, usize, bool)],
    ) {
        if let Some(mem) = &mut self.memory {
            mem.sums = sums;
            mem.entries = entries;
        }
        if self.contact_gap.is_some() {
            for &(b, c, k) in updates {
                self.contact[b][c] = k;
                self.contact[c][b] = k;
            }
        }
    }

    /// Fortuin-Kasteleyn (FK) rigid cluster step (`formal/ReversibleSampling/FkClusterMove.lean`):
    ///
    /// 1. Pick a random mobile seed body `b0` uniformly from free bodies.
    /// 2. Grow an FK cluster `C` via BFS using bond probability
    ///    `p_xy = 1 - exp(beta_fk * min(0, E_pair(x, y)))`.
    ///    If `C` recruits any fixed body (`fixed[c]`) or exceeds `cs.max_size`, retain the state
    ///    (exact self-loop by `complete_reversible`).
    /// 3. Propose a rigid SE(3) motion of `C` (either docking the handle `b0` onto spectator
    ///    anchors `V \ C`, uniform vessel reinsertion of `b0`, or local COM translation + rotation).
    /// 4. Accept with the exact FK-cancelled Metropolis ratio:
    ///    `min(1, (q_rev / q_fwd) * exp(-(dF_solv + dE_cut - dm) + beta_fk * (min(0, E_cut_new) - min(0, E_cut_old))))`.
    ///    When `beta_fk == 1`, all attractive cut energies cancel against the FK cut survival
    ///    probabilities, allowing pre-formed oligomers in bulk solution to diffuse and dock
    ///    onto the crystal seed without breaking their internal bonds.
    pub fn step_cluster(
        &mut self,
        rng: &mut StdRng,
        cs: &ClusterSettings,
        dock: Option<&DockSettings>,
    ) -> Result<bool> {
        let n = self.poses.len();
        let free: Vec<usize> = (0..n).filter(|&b| !self.fixed[b]).collect();
        ensure!(!free.is_empty(), "no mobile body");
        let b0 = free[rng.random_range(0..free.len())];
        self.stats.cluster_attempts += 1;

        let mut in_cluster = vec![false; n];
        in_cluster[b0] = true;
        let mut members = vec![b0];
        let mut queue = std::collections::VecDeque::from([b0]);
        let mut pinned = false;
        let max_reach = (2. * self.tree.bound + self.well.width.max(self.like.cutoff).max(6.0))
            .max(2. * self.charge_extent + self.like.cutoff);

        if cs.fk_beta > 0. {
            while let Some(u) = queue.pop_front() {
                let pu = self.poses[u];
                for v in 0..n {
                    if in_cluster[v] || norm(sub(self.poses[v].position, pu.position)) > max_reach {
                        continue;
                    }
                    let e_uv = self.pair_surrogate_energy(pu, self.poses[v]);
                    if e_uv < 0. {
                        let p_bond = 1. - (cs.fk_beta * e_uv).exp();
                        if rng.random::<f64>() < p_bond {
                            if self.fixed[v] {
                                pinned = true;
                                break;
                            }
                            in_cluster[v] = true;
                            members.push(v);
                            if cs.max_size > 0 && members.len() > cs.max_size {
                                break;
                            }
                            queue.push_back(v);
                        }
                    }
                }
                if pinned || (cs.max_size > 0 && members.len() > cs.max_size) {
                    break;
                }
            }
        }

        if pinned {
            self.stats.cluster_pinned_rejected += 1;
            return Ok(false);
        }
        if cs.max_size > 0 && members.len() > cs.max_size {
            self.stats.cluster_cap_rejected += 1;
            return Ok(false);
        }
        members.sort_unstable();
        let is_multibody = members.len() > 1;
        self.stats.cluster_multibody_attempts += u64::from(is_multibody);

        let spectators: Vec<usize> = (0..n).filter(|&a| !in_cluster[a]).collect();
        let use_dock = dock.is_some()
            && !spectators.is_empty()
            && cs.dock_probability > 0.
            && rng.random::<f64>() < cs.dock_probability;

        let mut is_reinsertion = false;
        let mut log_q_ratio = 0.;

        let moves: Vec<(usize, Pose)> = if use_dock {
            let d = dock.unwrap();
            self.stats.cluster_dock_attempts += 1;
            let library = if d.library {
                let mut lib = self.library_excluding(&in_cluster);
                if let Some(slots) = self.memory_slots() {
                    lib.extend_from_slice(slots);
                }
                Some(lib)
            } else {
                None
            };
            let comps: &[Pose] = library.as_deref().unwrap_or(&d.components);
            let u = if d.learned.is_none() && comps.is_empty() {
                1.
            } else {
                d.uniform_weight
            };
            let old_b0 = self.poses[b0];
            let new_b0 = if rng.random::<f64>() < u {
                let r = self.wall.radius;
                let position = loop {
                    let x: Vec3 = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * r);
                    if norm(x) <= r {
                        break x;
                    }
                };
                let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
                let qn = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                Pose {
                    position,
                    orientation: q.map(|x| x / qn),
                }
            } else if let Some(m) = &d.learned {
                let anchors: Vec<Pose> = spectators.iter().map(|&a| self.poses[a]).collect();
                match m.draw_singleton_independent(rng, &anchors)?.0 {
                    Some(p) => p,
                    None => return Ok(false),
                }
            } else {
                let a = spectators[rng.random_range(0..spectators.len())];
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
            log_q_ratio = self.dock_log_density_subset(&in_cluster, old_b0, d, comps, u)
                - self.dock_log_density_subset(&in_cluster, new_b0, d, comps, u);
            let carried = crate::rigid_subset::transport_members(&self.poses, &members, b0, new_b0)?;
            members.iter().copied().zip(carried).collect()
        } else if rng.random::<f64>() < cs.reinsertion_probability {
            is_reinsertion = true;
            self.stats.cluster_reinsertion_attempts += 1;
            let r = self.wall.radius;
            let position = loop {
                let x: Vec3 = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * r);
                if norm(x) <= r {
                    break x;
                }
            };
            let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
            let qn = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            let new_b0 = Pose {
                position,
                orientation: q.map(|x| x / qn),
            };
            let carried = crate::rigid_subset::transport_members(&self.poses, &members, b0, new_b0)?;
            members.iter().copied().zip(carried).collect()
        } else {
            let scale_f = if cs.size_scaling {
                (members.len() as f64).powf(-0.5)
            } else {
                1.0
            };
            let sigma_t = cs.translation * scale_f;
            let sigma_r = cs.rotation * scale_f;
            let mut com = [0.; 3];
            for &c in &members {
                com = add(com, self.poses[c].position);
            }
            com = scale(com, 1. / (members.len() as f64));
            let dt: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                z * sigma_t
            });
            let c_rot: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                z * sigma_r / 2.
            });
            let delta_r = cayley(c_rot);
            let new_com = add(com, dt);
            members
                .iter()
                .map(|&c| {
                    let old_p = self.poses[c];
                    let rel = sub(old_p.position, com);
                    (
                        c,
                        Pose {
                            position: add(new_com, matvec(delta_r, rel)),
                            orientation: quaternion(matmul(delta_r, rotation(old_p.orientation))),
                        },
                    )
                })
                .collect()
        };

        self.stats.attempts += 1;
        for &(_, p_new) in &moves {
            if !self.wall.contains(p_new) {
                self.stats.wall_rejected += 1;
                return Ok(false);
            }
        }
        let placed_moves: Vec<(usize, Placed)> =
            moves.iter().map(|&(b, p)| (b, Placed::new(p))).collect();
        for &(b, ref pb) in &placed_moves {
            for &d in &spectators {
                let q = self.poses[d];
                if norm(sub(q.position, pb.position)) <= 2. * self.tree.bound
                    && self.tree.overlaps(pb, &self.placed[d])
                {
                    self.stats.hard_rejected += 1;
                    return Ok(false);
                }
            }
            // Also guard against numerical drift creating intra-cluster hard overlaps.
            for &(b2, ref pb2) in &placed_moves {
                if b2 > b
                    && norm(sub(pb2.position, pb.position)) <= 2. * self.tree.bound
                    && self.tree.overlaps(pb, pb2)
                {
                    self.stats.hard_rejected += 1;
                    return Ok(false);
                }
            }
        }

        // Compute cut pairwise changes between C and V \ C, plus any floating-point
        // intra-cluster drift so cached pair_count/like_sum/chem_sum stay machine-exact.
        let mut dp: i64 = 0;
        let mut dl: f64 = 0.;
        let mut dc: f64 = 0.;
        let mut fk_log_survival_diff = 0.;

        for &(b, p_new) in &moves {
            let p_old = self.poses[b];
            for &d in &spectators {
                let q = self.poses[d];
                if norm(sub(p_old.position, q.position)) > max_reach
                    && norm(sub(p_new.position, q.position)) > max_reach
                {
                    continue;
                }
                let wp_old = self.well_pair_between(p_old, q) as i64;
                let wp_new = self.well_pair_between(p_new, q) as i64;
                let lk_old = self.like_pair_between(p_old, q);
                let lk_new = self.like_pair_between(p_new, q);
                let ch_old = self.chem_pair_between(p_old, q);
                let ch_new = self.chem_pair_between(p_new, q);
                dp += wp_new - wp_old;
                dl += lk_new - lk_old;
                dc += ch_new - ch_old;
                if cs.fk_beta > 0. {
                    let e_old =
                        -self.well.epsilon * (wp_old as f64) + self.like.weight * lk_old + ch_old;
                    let e_new =
                        -self.well.epsilon * (wp_new as f64) + self.like.weight * lk_new + ch_new;
                    fk_log_survival_diff += cs.fk_beta * (e_new.min(0.) - e_old.min(0.));
                }
            }
        }
        // Account for any tiny floating-point boundary shift on internal pairs (b1 < b2 in C).
        for (idx1, &(b1, p1_new)) in moves.iter().enumerate() {
            let p1_old = self.poses[b1];
            for &(b2, p2_new) in &moves[idx1 + 1..] {
                let p2_old = self.poses[b2];
                if norm(sub(p1_old.position, p2_old.position)) > max_reach {
                    continue;
                }
                dp += self.well_pair_between(p1_new, p2_new) as i64
                    - self.well_pair_between(p1_old, p2_old) as i64;
                dl += self.like_pair_between(p1_new, p2_new)
                    - self.like_pair_between(p1_old, p2_old);
                dc += self.chem_pair_between(p1_new, p2_new)
                    - self.chem_pair_between(p1_old, p2_old);
            }
        }

        let d_well = -self.well.epsilon * dp as f64;
        let update = self.solv.propose(&moves)?;
        let (dm, sums, n_entries, contact_updates) = self.memory_delta_multi(&in_cluster, &moves);
        let de_total = update.delta + d_well + self.like.weight * dl + dc;
        let log_acc = -de_total + fk_log_survival_diff + dm + log_q_ratio;
        ensure!(!log_acc.is_nan(), "NaN cluster move acceptance");
        if log_acc >= 0. || rng.random::<f64>() < log_acc.exp() {
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.chem_sum += dc;
            for (b, p_new) in moves {
                self.poses[b] = p_new;
                self.placed[b] = Placed::new(p_new);
            }
            self.commit_memory_multi(sums, n_entries, &contact_updates);
            self.stats.accepted += 1;
            self.stats.cluster_accepted += 1;
            self.stats.cluster_multibody_accepted += u64::from(is_multibody);
            if use_dock {
                self.stats.cluster_dock_accepted += 1;
            }
            if is_reinsertion {
                self.stats.cluster_reinsertion_accepted += 1;
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Proper centered half-turn GCA step (`spherical::HalfTurn` + Fortuin-Kasteleyn bonds +
    /// residual morphometric solvation gate, `formal/ReversibleSampling/FkClusterMove.lean`):
    ///
    /// 1. Draw an isotropic axis `u` on `S^2` and construct the chirality-preserving involution
    ///    `T_u(r) = 2 u (u . r) - r`.
    /// 2. Link pairs `(i, j)` that would hard-overlap across `old[i]` and `shadow[j]` (`p = 1`).
    /// 3. For non-overlapping pairs `(i, j)`, compute `dE_ij = E_pair(old[i], shadow[j]) - E_pair(old[i], old[j])`
    ///    and link `(i, j)` with FK probability `p_ij = max(0, 1 - exp(-fk_beta * dE_ij))`.
    /// 4. Partition into connected components. Any component containing a fixed seed body is
    ///    pinned (`flip = false`); each mobile component flips independently with probability 1/2.
    /// 5. Gate the collective move by the exact residual energy change (including many-body
    ///    morphometric solvation `self.solv.propose(&moves)` and Poisson memory `dm`).
    pub fn step_gca(&mut self, rng: &mut StdRng, fk_beta: f64) -> Result<bool> {
        let n = self.poses.len();
        let free: Vec<usize> = (0..n).filter(|&b| !self.fixed[b]).collect();
        ensure!(!free.is_empty(), "no mobile body");
        let b0 = free[rng.random_range(0..free.len())];
        self.stats.gca_attempts += 1;
        let axis: Vec3 = loop {
            let v: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
            let len = norm(v);
            if len > 1e-12 {
                break v.map(|x| x / len);
            }
        };
        let transform = crate::spherical::HalfTurn::new(axis)?;
        let shadow: Vec<Pose> = self.poses.iter().map(|&p| transform.apply(p)).collect();

        let hard_reach = 2. * self.tree.bound;
        let max_reach = (2. * self.tree.bound + self.well.width.max(self.like.cutoff).max(6.0))
            .max(2. * self.charge_extent + self.like.cutoff);

        let mut in_cluster = vec![false; n];
        in_cluster[b0] = true;
        let mut moves = vec![(b0, shadow[b0])];
        let mut queue = std::collections::VecDeque::from([b0]);
        let mut pinned = false;

        while let Some(u) = queue.pop_front() {
            let pu_old = self.poses[u];
            let pu_new = shadow[u];
            let placed_u_new = Placed::new(pu_new);
            for v in 0..n {
                if in_cluster[v] {
                    continue;
                }
                let pv_old = self.poses[v];
                let pv_new = shadow[v];
                let d_old = norm(sub(pu_old.position, pv_old.position));
                let d_cross = norm(sub(pu_old.position, pv_new.position));
                if d_old > max_reach && d_cross > max_reach {
                    continue;
                }
                let mut bonded = false;
                if d_cross <= hard_reach {
                    let placed_v_new = Placed::new(pv_new);
                    if self.tree.overlaps(&placed_u_new, &self.placed[v])
                        || self.tree.overlaps(&self.placed[u], &placed_v_new)
                    {
                        bonded = true;
                    }
                }
                if !bonded && fk_beta > 0. {
                    let e_old = self.pair_surrogate_energy(pu_old, pv_old);
                    let e_cross = self.pair_surrogate_energy(pu_old, pv_new);
                    let de = e_cross - e_old;
                    if de > 0. {
                        let p_bond = 1. - (-fk_beta * de).exp();
                        if rng.random::<f64>() < p_bond {
                            bonded = true;
                        }
                    }
                }
                if bonded {
                    if self.fixed[v] {
                        pinned = true;
                        break;
                    }
                    in_cluster[v] = true;
                    moves.push((v, pv_new));
                    queue.push_back(v);
                }
            }
            if pinned {
                break;
            }
        }

        if pinned {
            return Ok(false);
        }
        moves.sort_unstable_by_key(|&(i, _)| i);
        let spectators: Vec<usize> = (0..n).filter(|&a| !in_cluster[a]).collect();

        self.stats.attempts += 1;
        for &(_, p_new) in &moves {
            if !self.wall.contains(p_new) {
                self.stats.wall_rejected += 1;
                return Ok(false);
            }
        }

        let mut dp: i64 = 0;
        let mut dl: f64 = 0.;
        let mut dc: f64 = 0.;
        let mut fk_log_survival_diff = 0.;

        for &(b, p_new) in &moves {
            let p_old = self.poses[b];
            for &d in &spectators {
                let q_old = self.poses[d];
                if norm(sub(p_old.position, q_old.position)) > max_reach
                    && norm(sub(p_new.position, q_old.position)) > max_reach
                {
                    continue;
                }
                let wp_old = self.well_pair_between(p_old, q_old) as i64;
                let wp_new = self.well_pair_between(p_new, q_old) as i64;
                let lk_old = self.like_pair_between(p_old, q_old);
                let lk_new = self.like_pair_between(p_new, q_old);
                let ch_old = self.chem_pair_between(p_old, q_old);
                let ch_new = self.chem_pair_between(p_new, q_old);
                dp += wp_new - wp_old;
                dl += lk_new - lk_old;
                dc += ch_new - ch_old;
                if fk_beta > 0. {
                    let e_old =
                        -self.well.epsilon * (wp_old as f64) + self.like.weight * lk_old + ch_old;
                    let e_cross =
                        -self.well.epsilon * (wp_new as f64) + self.like.weight * lk_new + ch_new;
                    let de = e_cross - e_old;
                    // Forward survival contributed -fk_beta * max(0, de);
                    // Reverse survival contributes -fk_beta * max(0, -de).
                    // Ratio P_surv(rev) / P_surv(fwd) = exp(fk_beta * de).
                    fk_log_survival_diff += fk_beta * de;
                }
            }
        }
        for (idx1, &(b1, p1_new)) in moves.iter().enumerate() {
            let p1_old = self.poses[b1];
            for &(b2, p2_new) in &moves[idx1 + 1..] {
                let p2_old = self.poses[b2];
                if norm(sub(p1_old.position, p2_old.position)) > max_reach {
                    continue;
                }
                dp += self.well_pair_between(p1_new, p2_new) as i64
                    - self.well_pair_between(p1_old, p2_old) as i64;
                dl += self.like_pair_between(p1_new, p2_new)
                    - self.like_pair_between(p1_old, p2_old);
                dc += self.chem_pair_between(p1_new, p2_new)
                    - self.chem_pair_between(p1_old, p2_old);
            }
        }

        let d_well = -self.well.epsilon * dp as f64;
        let update = self.solv.propose(&moves)?;
        let (dm, sums, n_entries, contact_updates) = self.memory_delta_multi(&in_cluster, &moves);
        let de_total = update.delta + d_well + self.like.weight * dl + dc;
        let log_acc = -de_total + fk_log_survival_diff + dm;
        ensure!(!log_acc.is_nan(), "NaN GCA acceptance");
        if log_acc >= 0. || rng.random::<f64>() < log_acc.exp() {
            let moved_count = moves.len() as u64;
            self.solv.commit(update);
            self.pair_count += dp;
            self.like_sum += dl;
            self.chem_sum += dc;
            for (b, p_new) in moves {
                self.poses[b] = p_new;
                self.placed[b] = Placed::new(p_new);
            }
            self.commit_memory_multi(sums, n_entries, &contact_updates);
            self.stats.accepted += 1;
            self.stats.gca_accepted += 1;
            self.stats.gca_moved_bodies += moved_count;
            return Ok(true);
        }
        Ok(false)
    }

    /// Cumulative wall-center shift (`p_coordinate = p_sphere + coordinate_wall_center`).
    pub fn coordinate_wall_center(&self) -> Vec3 {
        self.coordinate_wall_center
    }

    /// Common-translation chord move (`Container::center_shift`) of all bodies (including the
    /// seed) inside the spherical vessel. Preserves all pairwise relative poses and energies
    /// exactly (`dE = 0`, rejection-free), shifting the seed off the sphere origin so subsequent
    /// centered `HalfTurn` GCA flips (`step_gca`) can transport oligomers radially.
    pub fn step_center_shift(&mut self, rng: &mut StdRng) -> Result<bool> {
        self.stats.center_shift_attempts += 1;
        let direction: Vec3 = loop {
            let v: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
            let len = norm(v);
            if len > 1e-12 {
                break v.map(|x| x / len);
            }
        };
        let open_u = ((rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
        let outcome = self.wall.center_shift(&mut self.poses, direction, open_u)?;
        if outcome.degenerate {
            return Ok(false);
        }
        for (b, &p) in self.poses.iter().enumerate() {
            self.placed[b] = Placed::new(p);
        }
        self.solv.translate_all(&self.poses);
        self.coordinate_wall_center = sub(self.coordinate_wall_center, outcome.displacement);
        self.stats.center_shift_accepted += 1;
        Ok(true)
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

    /// Recompute the chemistry residual energy from scratch (for checks).
    pub fn recompute_chem(&self) -> f64 {
        let sum: f64 = (0..self.poses.len())
            .map(|a| self.chem_pairs(a, self.poses[a]))
            .sum();
        sum / 2.
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
        // A fixed prior centre at contact distance exercises the prior term.
        let prior = vec![(
            Pose {
                position: [2.3, 0., 0.],
                orientation: [1., 0., 0., 0.],
            },
            1.,
        )];
        let mem = MemorySettings {
            lambda_floor: 2.,
            lambda_prior: 1.,
            lambda: 0.5,
            death: 0.3,
            sigma_translation: 0.3,
            sigma_rotation: 10f64.to_radians(),
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
                mc = mc.with_memory(mem, prior.clone(), &mut rng).unwrap();
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

    /// Chemistry residual model: exact SE(3) invariance, pair symmetry E(A, B) == E(B, A),
    /// and zero drift between incremental move updates (`chem_energy()`) and full recomputation
    /// (`recompute_chem()`) under both local and docking moves.
    #[test]
    fn chemistry_model_is_symmetric_and_consistent_under_moves() {
        let shape = Shape {
            name: "chem_dimer".into(),
            volume: 20.,
            atoms: vec![
                Atom { center: [-1.2, 0.0, 0.0], radius: 0.8 },
                Atom { center: [1.2, 0.0, 0.0], radius: 0.8 },
                Atom { center: [0.0, 1.3, 0.0], radius: 0.8 },
                Atom { center: [0.0, -1.3, 0.0], radius: 0.8 },
            ],
        };
        let tree = SphereTree::new(shape).unwrap();
        let wall = Container::new(10.0, &tree).unwrap();
        let model = SolvationModel {
            probe: 1.4,
            pressure: 0.,
            coefficients: vec![[0.; 3]],
            atom_types: vec![0; 4],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell { epsilon: 0.3, width: 1.0 };
        let chem = ChemistryModel {
            flags: vec![
                CHEM_CAT | CHEM_PURE_DON,
                CHEM_NACC | CHEM_TYR_OH,
                CHEM_NDON | CHEM_AMIDE_ND | CHEM_PURE_DON,
                CHEM_ANI | CHEM_PURE_ACC,
            ],
            weights: [1.1, 1.6, 10.5, 0.45, 1.8, 0.85, 1.8],
            wb_cap: 5.0,
        };
        let start = vec![
            Pose { position: [0., 0., 0.], orientation: [1., 0., 0., 0.] },
            Pose {
                position: [4.6, 0.2, -0.1],
                orientation: quaternion(cayley([0.1, -0.2, 0.15])),
            },
            Pose {
                position: [-4.8, 0.4, 0.3],
                orientation: quaternion(cayley([-0.25, 0.3, -0.1])),
            },
        ];
        let centers: Vec<Vec3> = tree.shape.atoms.iter().map(|a| a.center).collect();
        // Verify exact symmetry and SE(3) invariance on initial pairs.
        for i in 0..start.len() {
            for j in i + 1..start.len() {
                let e_ij = chem.pair_energy(&centers, tree.bound, start[i], start[j]);
                let e_ji = chem.pair_energy(&centers, tree.bound, start[j], start[i]);
                assert!(
                    (e_ij - e_ji).abs() < 1e-12,
                    "asymmetric pair chemistry energy: {e_ij} vs {e_ji}"
                );
            }
        }
        let mut mc = MorphMc::new(tree, wall, model, well, start, vec![true, false, false])
            .unwrap()
            .with_chemistry_model(chem)
            .unwrap();
        let mv = MoveSettings {
            translation: 0.4,
            rotation: 0.25,
            reinsertion_probability: 0.1,
        };
        let dock = DockSettings {
            learned: None,
            library: false,
            components: vec![Pose {
                position: [4.6, 0., 0.],
                orientation: [1., 0., 0., 0.],
            }],
            sigma_translation: 0.4,
            sigma_rotation: 0.2,
            uniform_weight: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for t in 0..2_000 {
            if t % 3 == 0 {
                mc.step_dock(&mut rng, &dock).unwrap();
            } else {
                mc.step(&mut rng, mv).unwrap();
            }
        }
        let diff = (mc.recompute_chem() - mc.chem_energy()).abs();
        assert!(
            diff < 1e-10,
            "cached chemistry energy {} differs from recomputed {} by {}",
            mc.chem_energy(),
            mc.recompute_chem(),
            diff
        );
    }

    /// FK rigid cluster moves (`step_cluster`: local COM diffusion, reinsertion, and library
    /// cluster docking) and proper half-turn GCA (`step_gca`) preserve the exact energy caches,
    /// contact matrix, and Poisson memory sums with zero drift across multi-body moves, and
    /// preserve the equilibrium radial/pair distance distribution.
    #[test]
    fn fk_cluster_and_gca_moves_preserve_caches_and_detailed_balance() {
        let r = 1.0;
        let probe = 1.4;
        let shape = Shape {
            name: "dimer_ball".into(),
            volume: 2. * 4. / 3. * std::f64::consts::PI,
            atoms: vec![
                Atom {
                    center: [-0.8, 0., 0.],
                    radius: r,
                },
                Atom {
                    center: [0.8, 0., 0.],
                    radius: r,
                },
            ],
        };
        let tree = SphereTree::new(shape.clone()).unwrap();
        let wall = Container::new(8.5, &tree).unwrap();
        let model = SolvationModel {
            probe,
            pressure: 0.209,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0, 0],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 0.8,
            width: 0.8,
        };
        let like = LikeCharge {
            weight: 0.2,
            cutoff: 3.2,
            groups: vec![ChargeGroup {
                atoms: vec![0],
                sign: 1,
                fraction: 0.5,
            }],
        };
        let chem = ChemistryModel {
            flags: vec![CHEM_CAT | CHEM_PURE_DON, CHEM_ANI | CHEM_PURE_ACC],
            weights: [0.5, 0.8, 1.5, 0.45, 0.8, 0.85, 0.8],
            wb_cap: 5.0,
        };
        let start = vec![
            Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [0., 4.2, 0.],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [0., 4.2, 2.25],
                orientation: [1., 0., 0., 0.],
            },
            Pose {
                position: [0., -4.2, 0.],
                orientation: [1., 0., 0., 0.],
            },
        ];
        let mut rng = StdRng::seed_from_u64(12345);
        let mem_s = MemorySettings {
            lambda_floor: 1.5,
            lambda_prior: 0.,
            lambda: 0.4,
            death: 0.2,
            sigma_translation: 0.3,
            sigma_rotation: 10f64.to_radians(),
            floor_sigma: 3.0,
        };
        let mut mc = MorphMc::new(
            tree,
            wall,
            model.clone(),
            well,
            start,
            vec![true, false, false, false],
        )
        .unwrap()
        .with_like_charge(like)
        .unwrap()
        .with_chemistry_model(chem)
        .unwrap()
        .with_contact_library(0.8)
        .unwrap()
        .with_memory(mem_s, Vec::new(), &mut rng)
        .unwrap();

        let mv = MoveSettings {
            translation: 0.4,
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
        let cs = ClusterSettings {
            max_size: 4,
            fk_beta: 1.0,
            dock_probability: 0.5,
            reinsertion_probability: 0.25,
            translation: 0.5,
            rotation: 0.25,
            size_scaling: true,
        };
        for t in 0..3_000 {
            if t % 10 == 0 {
                mc.step_center_shift(&mut rng).unwrap();
            }
            match t % 4 {
                0 => {
                    mc.step_cluster(&mut rng, &cs, Some(&dock)).unwrap();
                }
                1 => {
                    mc.step_gca(&mut rng, 1.0).unwrap();
                }
                2 => {
                    mc.step_dock(&mut rng, &dock).unwrap();
                }
                _ => {
                    mc.step(&mut rng, mv).unwrap();
                    if t % 12 == 11 {
                        mc.refresh_memory(&mut rng);
                    }
                }
            }
        }
        let (sf, sw) = mc.recompute();
        assert!((sf - mc.solvation_energy()).abs() < 1e-9, "solvation drift: {sf} vs {}", mc.solvation_energy());
        assert!((sw - mc.well_energy()).abs() < 1e-9, "well drift: {sw} vs {}", mc.well_energy());
        assert!((mc.recompute_like() - mc.like_energy()).abs() < 1e-9, "like-charge drift");
        assert!((mc.recompute_chem() - mc.chem_energy()).abs() < 1e-9, "chemistry drift");
        assert!(mc.memory_sum_error() < 1e-9, "memory sum drift: {}", mc.memory_sum_error());
        assert!(mc.stats.cluster_accepted >= 5, "cluster moves should be accepted: {:?}", mc.stats);
        assert!(mc.stats.cluster_multibody_accepted >= 5, "multi-body FK clusters should be accepted: {:?}", mc.stats);
        assert!(mc.stats.cluster_dock_accepted >= 1, "cluster dock moves should be accepted: {:?}", mc.stats);
        assert!(mc.stats.gca_accepted > 20, "half-turn GCA moves should be accepted: {:?}", mc.stats);
        assert!(mc.stats.center_shift_accepted > 200, "center_shift moves should be accepted: {:?}", mc.stats);
    }
}
