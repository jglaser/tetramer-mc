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
use std::sync::Arc;

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

/// Learned SE(3)-invariant interface chemistry + hydration-shell solvation model (Formulation 6b):
/// evaluates direct neutral and cationic H-bonds, bilateral-cationic anion-bridge triads,
/// water-bridged polar networks (gated by like-polar clashes), unsatisfied buried polar
/// and charged desolvation penalties, and optional 5-feature hydration-shell morphometric
/// solvation (`solv_weights`: `[w_p2, w_p3, w_a1, w_a2, w_a05]` for gap `g < 2.8 A`)
/// between interacting bodies.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct ChemistryModel {
    /// Per-atom 8-bit chemistry flags (length = number of body atoms, or empty if disabled).
    pub flags: Vec<u8>,
    /// Feature weights [w_hb, w_chb, w_cl, w_wb, w_clash, w_upol, w_uchg] in kT.
    pub weights: [f64; 7],
    /// Cap on raw water-bridge pair score before clash gating (default 5.0).
    pub wb_cap: f64,
    /// Optional 5-feature hydration-shell morphometric solvation weights
    /// `[w_p2, w_p3, w_a1, w_a2, w_a05]` in kT for pairwise gap `g = d - (r_i + r_j) < 2.8 A`.
    #[serde(default)]
    pub solv_weights: [f64; 5],
    /// Optional cap (`> 0` in kT) on attractive pairwise hydration-shell solvation (`e_solv >= -solv_cap`),
    /// preventing unbounded linear solvation runaway on oversized non-native patches (`0.0` = uncapped).
    #[serde(default)]
    pub solv_cap: f64,
    /// Optional smooth Hermite burial gate `[b_on, b_full]` (in kT of `-e_solv_raw`) multiplying
    /// attractive polar/ion-bridge chemistry (`s_hb, s_chb, s_cl, s_wb`); disabled when `b_full <= b_on`.
    #[serde(default)]
    pub burial_gate: [f64; 2],
    /// Optional 2-feature SE(3)-invariant Stillinger-Weber / Tersoff 3-body cooperative
    /// weights `[w_wedge, w_tri]` in kT (`>= 0` for attraction) across contacting surface
    /// patches `s_{ij}` on a shared vertex body `i`.
    #[serde(default)]
    pub coop_weights: [f64; 2],
    /// Half-saturation surface burial scale `s_0` (default `15.0`) for `phi(s) = s / (s + s_0)`.
    #[serde(default = "default_coop_s0")]
    pub coop_s0: f64,
    /// Learned 3-body Lawrence-Colman junction weights `[w_seal, w_sat]` in kT:
    /// `-w_seal * (W_ABC / (W_ABC + junc_w0)) * (1 - R_ABC^2)^2 + w_sat * W_ABC * R_ABC^2`,
    /// where `N_A = R_A sum_{i in A} u_i^(A,B) u_i^(A,C) n_i^body`,
    /// `W_ABC = ||N_A|| + ||N_B|| + ||N_C||`, and `R_ABC = ||N_A + N_B + N_C|| / W_ABC`.
    #[serde(default)]
    pub junc_weights: [f64; 2],
    /// Half-saturation scale `W_0` (default `4.0`) for `W_ABC / (W_ABC + W_0)`.
    #[serde(default = "default_junc_w0")]
    pub junc_w0: f64,
}

fn default_coop_s0() -> f64 {
    15.0
}

fn default_junc_w0() -> f64 {
    4.0
}

/// Precompute outward unit surface normals in the body frame for each template sphere `i`
/// from the negative gradient of the smooth molecular volume density field
/// `rho(x) = sum_j r_j^3 exp(-||x - x_j||^2 / sigma^2)` with `sigma = 3.5 A`.
pub fn compute_body_normals(centers: &[Vec3], vdw: &[f64]) -> Vec<Vec3> {
    let n = centers.len();
    let inv_sig2 = 1.0 / (3.5 * 3.5);
    let r3: Vec<f64> = vdw
        .iter()
        .map(|&r| {
            let r_pos = if r > 0.0 { r } else { 1.0 };
            r_pos * r_pos * r_pos
        })
        .collect();
    let mut normals = Vec::with_capacity(n);
    for i in 0..n {
        let xi = centers[i];
        let mut g = [0.0f64; 3];
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = sub(xi, centers[j]);
            let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if d2 > 1e-12 {
                let w = r3[j] * (-d2 * inv_sig2).exp();
                g[0] += w * dx[0];
                g[1] += w * dx[1];
                g[2] += w * dx[2];
            }
        }
        let gn = norm(g);
        if gn > 1e-12 {
            normals.push([g[0] / gn, g[1] / gn, g[2] / gn]);
        } else {
            let xn = norm(xi);
            if xn > 1e-12 {
                normals.push([xi[0] / xn, xi[1] / xn, xi[2] / xn]);
            } else {
                normals.push([1.0, 0.0, 0.0]);
            }
        }
    }
    normals
}

/// Directed interface patch summary on body `a` from its contact with body `b`.
#[derive(Clone, Debug, Default)]
pub struct PatchContact {
    /// Smooth 6.0 A surface burial sum `sum_{i in a} max_{j in b} max(0, 1 - g_{ij}/6.0)`.
    pub s: f64,
    /// Unit vector from body `a`'s origin toward the burial-weighted interface centroid in `a`'s frame.
    pub u: Vec3,
    /// Per-atom 6.0 A burial weights `u_i^(a,b)` on body `a` (present when `s > 0.5` and `has_junction()`).
    pub u6: Option<Arc<[f64]>>,
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
    pub fn has_solvation(&self) -> bool {
        self.solv_weights.iter().any(|&w| w != 0.)
    }

    pub fn has_junction(&self) -> bool {
        self.junc_weights.iter().any(|&w| w != 0.)
    }

    pub fn has_cooperativity(&self) -> bool {
        self.coop_weights.iter().any(|&w| w != 0.) || self.has_junction()
    }

    pub fn is_enabled(&self) -> bool {
        !self.flags.is_empty()
            && (self.weights.iter().any(|&w| w != 0.)
                || self.has_solvation()
                || self.has_cooperativity())
    }

    /// Stillinger-Weber / Tersoff bond-order saturation `phi(s) = s / (s + s_0)` for `s > 0.5`.
    #[inline]
    pub fn coop_phi(&self, s: f64) -> f64 {
        if s <= 0.5 {
            0.0
        } else {
            let s0 = if self.coop_s0 > 0.0 { self.coop_s0 } else { 15.0 };
            s / (s + s0)
        }
    }

    /// Angular orthogonality factor `Psi(u_1, u_2) = 1 - max(0, u_1 . u_2)^2` on a vertex body.
    #[inline]
    pub fn coop_psi(u1: Vec3, u2: Vec3) -> f64 {
        let dot = (u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2]).clamp(-1.0, 1.0);
        if dot > 0.0 {
            1.0 - dot * dot
        } else {
            1.0
        }
    }

    /// Combine raw hydration-shell solvation `e_solv_raw`, buried hydration-shell atom sum
    /// `a1_burial = sum(u_a) + sum(u_b)`, and pairwise chemistry features into the total 2-body
    /// residual energy (kT), applying optional `solv_cap` and `burial_gate`.
    #[inline]
    pub fn combine_pair_energy(
        &self,
        e_solv_raw: f64,
        a1_burial: f64,
        s_hb: f64,
        s_chb: f64,
        s_cl: f64,
        s_wb: f64,
        s_clash: f64,
        unp_pol: f64,
        unp_chg: f64,
    ) -> f64 {
        let e_solv = if self.solv_cap > 0.0 {
            e_solv_raw.max(-self.solv_cap)
        } else {
            e_solv_raw
        };
        let g_burial = if self.burial_gate[1] > self.burial_gate[0] {
            let t = ((a1_burial - self.burial_gate[0])
                / (self.burial_gate[1] - self.burial_gate[0]))
                .clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        } else {
            1.0
        };
        let w = &self.weights;
        e_solv - g_burial * (w[0] * s_hb + w[1] * s_chb + w[2] * s_cl + w[3] * s_wb)
            + w[4] * s_clash
            + w[5] * unp_pol
            + w[6] * unp_chg
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
            || ((fi & CHEM_TYR_OH != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_TYR_OH != 0))
    }

    fn direct_complementary(fi: u8, fj: u8) -> bool {
        ((fi & CHEM_NDON != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_NDON != 0))
            || ((fi & CHEM_CAT != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_CAT != 0))
            || ((fi & CHEM_CAT != 0) && (fj & CHEM_ANI != 0))
            || ((fi & CHEM_ANI != 0) && (fj & CHEM_CAT != 0))
            || ((fi & CHEM_TYR_OH != 0) && (fj & CHEM_NACC != 0))
            || ((fi & CHEM_NACC != 0) && (fj & CHEM_TYR_OH != 0))
    }

    /// Whether chemical flag byte `f` represents a polar or charged side-chain functional group
    /// (cationic, anionic, aromatic OH, amide ND2/NE2, or dual donor+acceptor hydroxyl Ser/Thr/Tyr OH),
    /// as opposed to pure backbone NH (`CHEM_NDON | CHEM_BB_NH`) or backbone CO (`CHEM_NACC | CHEM_BB_CO`).
    fn is_sidechain_polar(f: u8) -> bool {
        (f & (CHEM_CAT | CHEM_ANI | CHEM_TYR_OH | CHEM_AMIDE_ND)) != 0
            || (f & (CHEM_NDON | CHEM_NACC)) == (CHEM_NDON | CHEM_NACC)
    }

    /// Pairwise chemistry residual energy (kT) between body `p` and body `q` with template
    /// atom centres `centers` and bounding radius `bound`. Symmetric in `(p, q)`.
    pub fn pair_energy(&self, centers: &[Vec3], bound: f64, p: Pose, q: Pose) -> f64 {
        if !self.is_enabled() || norm(sub(q.position, p.position)) > 2. * bound + 5.25 {
            return 0.;
        }
        let grid = BallGrid::new(centers, 5.25);
        self.pair_energy_with_grid(centers, bound, &grid, p, q)
    }

    /// Pairwise interface solvation + chemistry energy (kT) using atom radii `vdw` and a
    /// prebuilt template `BallGrid` (`cell >= 2*vmax + 2.8` when `has_solvation()`, else `>= 5.25`).
    pub fn pair_energy_with_radii_and_grid(
        &self,
        centers: &[Vec3],
        vdw: &[f64],
        bound: f64,
        grid: &BallGrid,
        p: Pose,
        q: Pose,
    ) -> f64 {
        if !self.has_solvation() {
            return self.pair_energy_with_grid(centers, bound, grid, p, q);
        }
        let max_cut = 6.6;
        if !self.is_enabled() || norm(sub(q.position, p.position)) > 2. * bound + max_cut {
            return 0.;
        }
        let n = centers.len();
        let rp = rotation(p.orientation);
        let rqt = transpose(rotation(q.orientation));
        let r_rel = matmul(rqt, rp);
        let t_rel = matvec(rqt, sub(p.position, q.position));
        let rpt = transpose(rp);
        let q_in_p = matvec(rpt, sub(q.position, p.position));
        let pre_cut2 = (bound + max_cut) * (bound + max_cut);

        let mut active_a: Vec<(usize, Vec3)> = Vec::new();
        for (i, &x) in centers.iter().enumerate() {
            let dx = sub(x, q_in_p);
            if dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] <= pre_cut2 {
                active_a.push((i, add(matvec(r_rel, x), t_rel)));
            }
        }
        if active_a.is_empty() {
            return 0.;
        }

        let mut p2 = 0.0f64;
        let mut p3 = 0.0f64;
        let mut max_u_a = vec![0.0f64; n];
        let mut max_u_b = vec![0.0f64; n];

        let mut min_d_a = vec![99.0f64; n];
        let mut min_comp_a = vec![99.0f64; n];
        let mut min_d_b = vec![99.0f64; n];
        let mut min_comp_b = vec![99.0f64; n];

        let mut s_hb = 0.0f64;
        let mut s_chb = 0.0f64;
        let mut s_wb_raw = 0.0f64;
        let mut chb_mids: Vec<Vec3> = Vec::new();
        let mut tn_pairs: Vec<(usize, usize, Vec3, f64)> = Vec::new();
        let mut clash_candidates: Vec<(usize, usize, f64)> = Vec::new();

        for &(i, ya) in &active_a {
            let fi = self.flags[i];
            let ri = vdw[i];
            for j in grid.near(&ya) {
                let xb = centers[j];
                let r_sum = ri + vdw[j];
                let r_solv = r_sum + 2.8;
                let r_lim = r_solv.max(5.25);
                let diff = sub(ya, xb);
                let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if d2 >= r_lim * r_lim {
                    continue;
                }
                let d = d2.sqrt();
                let g = d - r_sum;
                if g < 2.8 {
                    let u = ((2.8 - g) * (1.0 / 2.8)).clamp(0.0, 1.0);
                    let u2 = u * u;
                    p2 += u2;
                    p3 += u2 * u;
                    if u > max_u_a[i] {
                        max_u_a[i] = u;
                    }
                    if u > max_u_b[j] {
                        max_u_b[j] = u;
                    }
                }
                if d >= 5.25 {
                    continue;
                }
                let fj = self.flags[j];
                if fi != 0 && d < min_d_a[i] {
                    min_d_a[i] = d;
                }
                if fj != 0 && d < min_d_b[j] {
                    min_d_b[j] = d;
                }
                if fi == 0 || fj == 0 {
                    continue;
                }
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
                    let gh = (-u * u).exp();
                    if ((fi & CHEM_NDON != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_NDON != 0))
                    {
                        s_hb += gh;
                    }
                    if ((fi & CHEM_CAT != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_CAT != 0))
                    {
                        s_chb += gh;
                        if d < 3.6 {
                            chb_mids.push(scale(add(ya, xb), 0.5));
                        }
                    }
                }
                if (3.4..=4.5).contains(&d)
                    && (((fi & CHEM_TYR_OH != 0) && (fj & CHEM_AMIDE_ND != 0))
                        || ((fi & CHEM_AMIDE_ND != 0) && (fj & CHEM_TYR_OH != 0)))
                {
                    tn_pairs.push((i, j, scale(add(ya, xb), 0.5), d));
                }
                if d < 4.0
                    && (((fi & CHEM_PURE_ACC != 0) && (fj & CHEM_PURE_ACC != 0))
                        || ((fi & CHEM_PURE_DON != 0) && (fj & CHEM_PURE_DON != 0)))
                {
                    clash_candidates.push((i, j, d));
                }
            }
        }

        let mut a1 = 0.0f64;
        let mut a2 = 0.0f64;
        let mut a05 = 0.0f64;
        for &u in max_u_a.iter().chain(max_u_b.iter()) {
            if u > 0.0 {
                a1 += u;
                a2 += u * u;
                a05 += u.sqrt();
            }
        }
        let sw = &self.solv_weights;
        let e_solv = sw[0] * p2 + sw[1] * p3 + sw[2] * a1 + sw[3] * a2 + sw[4] * a05;

        let mut s_cl = 0.0f64;
        if !tn_pairs.is_empty() && !chb_mids.is_empty() {
            let cats_a: Vec<Vec3> = centers
                .iter()
                .enumerate()
                .filter(|&(i, _)| self.flags[i] & CHEM_CAT != 0)
                .map(|(_, &x)| add(matvec(r_rel, x), t_rel))
                .collect();
            let cats_b: Vec<Vec3> = centers
                .iter()
                .enumerate()
                .filter(|&(j, _)| self.flags[j] & CHEM_CAT != 0)
                .map(|(_, &x)| x)
                .collect();
            let mut cl_cands: Vec<(usize, usize, f64)> = Vec::new();
            for (i, j, mid, d) in tn_pairs {
                let da = cats_a.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                let db = cats_b.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                if da < 8.0 && db < 8.0 {
                    let d_chb = chb_mids.iter().map(|&m| norm(sub(m, mid))).fold(99.0f64, f64::min);
                    if d_chb < 8.5 {
                        let u1 = (d - 4.05) / 0.40;
                        let u2 = (d_chb.max(5.0) - 5.0) / 3.0;
                        cl_cands.push((i, j, (-u1 * u1 - u2 * u2).exp()));
                    }
                }
            }
            cl_cands.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut used_a = Vec::new();
            let mut used_b = Vec::new();
            for (i, j, val) in cl_cands {
                if !used_a.contains(&i) && !used_b.contains(&j) {
                    used_a.push(i);
                    used_b.push(j);
                    s_cl += val;
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
        for (i, (&dmin, &dcomp)) in min_d_a.iter().zip(&min_comp_a).enumerate() {
            if dmin >= 4.25 {
                continue;
            }
            let fi = self.flags[i];
            let is_chg = (fi & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fi & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 4.15 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.75;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.25 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.95;
                unp_chg += (-u * u).exp();
            }
        }
        for (j, (&dmin, &dcomp)) in min_d_b.iter().zip(&min_comp_b).enumerate() {
            if dmin >= 4.25 {
                continue;
            }
            let fj = self.flags[j];
            let is_chg = (fj & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fj & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 4.15 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.75;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.25 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.95;
                unp_chg += (-u * u).exp();
            }
        }

        self.combine_pair_energy(
            e_solv, a1, s_hb, s_chb, s_cl, s_wb, s_clash, unp_pol, unp_chg,
        )
    }

    /// Pairwise chemistry residual energy (kT) using a prebuilt template `BallGrid` (cell >= 5.25 A).
    pub fn pair_energy_with_grid(
        &self,
        centers: &[Vec3],
        bound: f64,
        grid: &BallGrid,
        p: Pose,
        q: Pose,
    ) -> f64 {
        if !self.is_enabled() || norm(sub(q.position, p.position)) > 2. * bound + 5.25 {
            return 0.;
        }
        let n = centers.len();
        let rp = rotation(p.orientation);
        let rqt = transpose(rotation(q.orientation));
        let cut = 5.25;
        // Transform flagged atoms of p into the frame of q, keeping those within bound + cut.
        let mut active_a: Vec<(usize, Vec3)> = Vec::new();
        for (i, &x) in centers.iter().enumerate() {
            if self.flags[i] == 0 {
                continue;
            }
            let y = matvec(rqt, sub(add(matvec(rp, x), p.position), q.position));
            if norm(y) <= bound + cut {
                active_a.push((i, y));
            }
        }
        let rpt = transpose(rp);
        let rq = rotation(q.orientation);
        let mut active_b: Vec<(usize, Vec3)> = Vec::new();
        for (j, &x) in centers.iter().enumerate() {
            if self.flags[j] == 0 {
                continue;
            }
            let y = matvec(rpt, sub(add(matvec(rq, x), q.position), p.position));
            if norm(y) <= bound + cut {
                active_b.push((j, y));
            }
        }
        if active_a.is_empty() && active_b.is_empty() {
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
        let mut tn_pairs: Vec<(usize, usize, Vec3, f64)> = Vec::new();
        let mut clash_candidates: Vec<(usize, usize, f64)> = Vec::new();

        for &(i, ya) in &active_a {
            let fi = self.flags[i];
            for j in grid.near(&ya) {
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
                if fj == 0 {
                    continue;
                }
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
                    tn_pairs.push((i, j, scale(add(ya, xb), 0.5), d));
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
            let mut cl_cands: Vec<(usize, usize, f64)> = Vec::new();
            for (i, j, mid, d) in tn_pairs {
                let da = cats_a.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                let db = cats_b.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                if da < 8.0 && db < 8.0 {
                    let d_chb = chb_mids.iter().map(|&m| norm(sub(m, mid))).fold(99.0f64, f64::min);
                    if d_chb < 8.5 {
                        let u1 = (d - 4.05) / 0.40;
                        let u2 = (d_chb.max(5.0) - 5.0) / 3.0;
                        cl_cands.push((i, j, (-u1 * u1 - u2 * u2).exp()));
                    }
                }
            }
            cl_cands.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut used_a = Vec::new();
            let mut used_b = Vec::new();
            for (i, j, val) in cl_cands {
                if !used_a.contains(&i) && !used_b.contains(&j) {
                    used_a.push(i);
                    used_b.push(j);
                    s_cl += val;
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
        for &(j, yb) in &active_b {
            let fj = self.flags[j];
            let dcomp = min_comp_b[j];
            let is_chg = (fj & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fj & (CHEM_NDON | CHEM_NACC)) != 0);
            if (is_npol && dcomp >= 3.65) || (is_chg && dcomp >= 4.30) {
                let mut dmin = min_d_b[j];
                for i in grid.near(&yb) {
                    if self.flags[i] == 0 {
                        let d = norm(sub(yb, centers[i]));
                        if d < dmin {
                            dmin = d;
                        }
                    }
                }
                if is_npol && dmin < 3.9 {
                    let u = (dmin.max(2.9) - 2.9) / 0.7;
                    unp_pol += (-u * u).exp();
                }
                if is_chg && dmin < 4.0 {
                    let u = (dmin.max(3.0) - 3.0) / 0.9;
                    unp_chg += (-u * u).exp();
                }
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
    /// Cached monomer-monomer pair surrogate energy `E_pair(identity, m_k)` for every slot (independent of x).
    slot_energies: Vec<f64>,
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

// Helper quaternion operations for fast relative rotation checks.

/// Quaternion product `q1 * q2` for unit quaternions `[w, x, y, z]`.
#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Conjugate (inverse for unit quaternion) `[w, -x, -y, -z]`.
#[inline]
fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Absolute dot product `|q1 . q2| = cos(theta / 2)` between two unit quaternions.
#[inline]
fn quat_abs_dot(a: [f64; 4], b: [f64; 4]) -> f64 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .min(1.0)
}

/// Rotation matrix mapping unit vector `u` to unit vector `v`.
fn rotation_between_unit(u: Vec3, v: Vec3) -> [[f64; 3]; 3] {
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-10 {
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    }
    if dot < -1.0 + 1e-10 {
        let ortho = if u[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut ax = [
            u[1] * ortho[2] - u[2] * ortho[1],
            u[2] * ortho[0] - u[0] * ortho[2],
            u[0] * ortho[1] - u[1] * ortho[0],
        ];
        let n = norm(ax);
        ax = [ax[0] / n, ax[1] / n, ax[2] / n];
        return rotation([0.0, ax[0], ax[1], ax[2]]);
    }
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let q = [1.0 + dot, cross[0], cross[1], cross[2]];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    rotation([q[0] / n, q[1] / n, q[2] / n, q[3] / n])
}

impl MemorySettings {
    fn kernel(&self, m: Pose, g: Pose) -> f64 {
        let (st, sr) = (self.sigma_translation, self.sigma_rotation);
        let d = sub(m.position, g.position);
        let t2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if t2 > 144. * st * st {
            return 0.;
        }
        let cos_half = quat_abs_dot(m.orientation, g.orientation);
        let min_cos_half = (6. * sr).cos();
        if cos_half < min_cos_half {
            return 0.;
        }
        let theta = 2. * cos_half.acos();
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
    /// Exact draw from mu_x / Lambda(x) given the contact entries of x and the prior,
    /// rejection-sampling any draw that causes a monomer-monomer hard-core overlap with
    /// the anchor body at the origin (`tree.overlaps(&placed_id, &Placed::new(m))`).
    fn draw(
        &self,
        rng: &mut StdRng,
        entries: &[Pose],
        prior: &[(Pose, f64)],
        tree: &SphereTree,
    ) -> Pose {
        let normal = |rng: &mut StdRng| -> f64 { StandardNormal.sample(rng) };
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let placed_id = Placed::new(identity);
        let total = self.total(entries.len());
        let u = rng.random::<f64>() * total;
        let use_floor = u < self.lambda_floor || (entries.is_empty() && prior.is_empty());
        let use_prior = !use_floor && u < self.lambda_floor + self.lambda_prior && !prior.is_empty();
        if use_floor || (!use_prior && entries.is_empty()) {
            let mut last = identity;
            for _ in 0..64 {
                let position: Vec3 = std::array::from_fn(|_| self.floor_sigma * normal(rng));
                let q: [f64; 4] = std::array::from_fn(|_| normal(rng));
                let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cand = Pose {
                    position,
                    orientation: q.map(|x| x / n),
                };
                last = cand;
                if !tree.overlaps(&placed_id, &Placed::new(cand)) {
                    return cand;
                }
            }
            return last;
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
        let rg = rotation(g.orientation);
        let mut last = g;
        for attempt in 0..32 {
            // Progressively shrink perturbation scale after initial full-sigma attempts
            // so tight lock-and-key contacts (e.g. m1, m6 with ~0.04 A steric clearance)
            // sample inside their narrow H-bond/salt-bridge well rather than selecting
            // outward-displaced (+1.5 A) tail draws that break short-range H-bonds.
            let s_scale = if attempt < 2 {
                1.0
            } else {
                (0.75f64).powi((attempt - 1) as i32).max(0.08)
            };
            let w: Vec3 = loop {
                let w: Vec3 =
                    std::array::from_fn(|_| s_scale * self.sigma_rotation * normal(rng));
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
            let position: Vec3 = std::array::from_fn(|k| {
                g.position[k] + s_scale * self.sigma_translation * normal(rng)
            });
            let cand = Pose {
                position,
                orientation: quaternion(matmul(rg, rotation(e))),
            };
            last = cand;
            if !tree.overlaps(&placed_id, &Placed::new(cand)) {
                return cand;
            }
        }
        // Tight native contacts (e.g. m1, m6 with min_gap ~ 0.037 A) almost always clash under
        // isotropic 0.5 A / 5 deg perturbations: draw a finer perturbation and retract along the
        // outward separation vector until the monomer hard cores are clash-free.
        let w: Vec3 = loop {
            let w: Vec3 = std::array::from_fn(|_| 0.10 * self.sigma_rotation * normal(rng));
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
        let r_cand = matmul(rg, rotation(e));
        let q_cand = quaternion(r_cand);
        let pos0: Vec3 =
            std::array::from_fn(|k| g.position[k] + 0.10 * self.sigma_translation * normal(rng));
        let gn = norm(g.position);
        let n_out = if gn > 1e-8 {
            scale(g.position, 1.0 / gn)
        } else {
            [1.0, 0.0, 0.0]
        };
        for step in 0..=32 {
            let shift = 0.06 * (step as f64);
            let cand = Pose {
                position: add(pos0, scale(n_out, shift)),
                orientation: q_cand,
            };
            if !tree.overlaps(&placed_id, &Placed::new(cand)) {
                return cand;
            }
        }
        if !tree.overlaps(&placed_id, &Placed::new(g)) {
            return g;
        }
        last
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
/// directions (plus any Poisson `memory_slots()`). When `memory_only` is true (or `library` is
/// false while `memory_only` is true), only the Poisson `memory_slots()` are used as docking
/// components, omitting the raw instantaneous contact pairs. They are a function of the other
/// bodies / auxiliary memory only, which the move leaves unchanged, so the same components
/// enter Q at both endpoints and the move is Metropolis-within-Gibbs for body b with an exact
/// acceptance (formal/ReversibleSampling/FrozenContext.lean). An empty component list makes
/// the proposal purely uniform.
#[derive(Clone, Default)]
pub struct DockSettings {
    pub learned: Option<std::sync::Arc<crate::docking::DockingProposal>>,
    pub library: bool,
    pub memory_only: bool,
    pub library_beta: f64,
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

/// Sparse update of pairwise cache + optional patch contacts for a single moved body `b`.
struct SingleBodyDelta {
    dp_well: i64,
    dc_chem: f64,
    old_neighbors: Vec<usize>,
    new_pairs: Vec<(usize, u32, f64, PatchContact, PatchContact)>,
}

/// Cached pairwise interaction between body `a` and body `b` (`a != b`).
#[derive(Clone, Copy, Debug, Default)]
struct PairCache {
    well: u32,
    chem: f64,
    like: f64,
}

pub struct MorphMc {
    tree: SphereTree,
    wall: Container,
    solv: SolvationState,
    pub well: SquareWell,
    centers: Vec<Vec3>,
    vdw: Vec<f64>,
    body_normals: Vec<Vec3>,
    template_grid: BallGrid,
    poses: Vec<Pose>,
    placed: Vec<Placed>,
    fixed: Vec<bool>,
    /// Spatial hash of body centers (`cell = 2 * bound + max_cutoff`) for O(1) neighbor lookup.
    body_grid: BallGrid,
    /// Cached symmetric pairwise `(well_pairs, chem_pair, like_pair)` matrix `pair_cache[a][b]`.
    pair_cache: Vec<Vec<PairCache>>,
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
    chem_grid: Option<BallGrid>,
    chem_sum: f64,
    /// Directed patch contact matrix `patch[a][b]` (`s > 0.5` when bodies `a, b` share 6 A interface)
    /// used when `chem.has_cooperativity()` is enabled.
    patch: Vec<Vec<PatchContact>>,
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
        let body_normals = compute_body_normals(&centers, &vdw);
        let vmax = vdw.iter().copied().fold(0., f64::max);
        let template_grid = BallGrid::new(&centers, 2. * vmax + well.width);
        let solv = SolvationState::new(model, &centers, &vdw, &poses)?;
        let placed = poses.iter().map(|&p| Placed::new(p)).collect();
        let body_centers: Vec<Vec3> = poses.iter().map(|p| p.position).collect();
        let body_cell = (2. * tree.bound + well.width.max(12.0)).max(1.0);
        let body_grid = BallGrid::new(&body_centers, body_cell);
        let n_bodies = poses.len();
        let pair_cache = vec![vec![PairCache::default(); n_bodies]; n_bodies];
        let mut s = Self {
            tree,
            wall,
            solv,
            well,
            centers,
            vdw,
            body_normals,
            template_grid,
            poses,
            placed,
            fixed,
            body_grid,
            pair_cache,
            coordinate_wall_center: [0.; 3],
            pair_count: 0,
            like: LikeCharge::default(),
            charge_extent: 0.,
            like_sum: 0.,
            chem: ChemistryModel::default(),
            chem_grid: None,
            chem_sum: 0.,
            patch: Vec::new(),
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
            let pos_a = s.poses[a].position;
            for b in s.body_grid.near(&pos_a) {
                if b > a {
                    ensure!(
                        !s.tree.overlaps(&s.placed[a], &s.placed[b]),
                        "bodies {a} and {b} overlap"
                    );
                }
            }
        }
        let mut pairs = 0i64;
        for a in 0..n_bodies {
            let pos_a = s.poses[a].position;
            for b in s.body_grid.near(&pos_a) {
                if b > a {
                    let wp = s.well_pair_between(s.poses[a], s.poses[b]);
                    s.pair_cache[a][b].well = wp as u32;
                    s.pair_cache[b][a].well = wp as u32;
                    pairs += wp as i64;
                }
            }
        }
        s.pair_count = pairs;
        Ok(s)
    }

    fn rebuild_body_grid(&mut self) {
        let max_cut = self
            .well
            .width
            .max(self.like.cutoff)
            .max(self.contact_gap.unwrap_or(0.))
            .max(12.0);
        let cell = (2. * self.tree.bound + max_cut).max(1.0);
        let body_centers: Vec<Vec3> = self.poses.iter().map(|p| p.position).collect();
        self.body_grid = BallGrid::new(&body_centers, cell);
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
        let reach = 2. * self.charge_extent + like.cutoff;
        self.like = like;
        self.rebuild_body_grid();
        let n = self.poses.len();
        let mut sum = 0.;
        for a in 0..n {
            for b in 0..n {
                self.pair_cache[a][b].like = 0.;
            }
        }
        if self.like.weight != 0. && !self.like.groups.is_empty() {
            for a in 0..n {
                let pos_a = self.poses[a].position;
                for b in self.body_grid.near_within(&pos_a, reach) {
                    if b > a {
                        let lk = self.like_pair_between(self.poses[a], self.poses[b]);
                        self.pair_cache[a][b].like = lk;
                        self.pair_cache[b][a].like = lk;
                        sum += lk;
                    }
                }
            }
        }
        self.like_sum = sum;
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
            chem.weights.iter().all(|w| w.is_finite())
                && chem.solv_weights.iter().all(|w| w.is_finite())
                && chem.solv_cap.is_finite()
                && chem.solv_cap >= 0.
                && chem.burial_gate.iter().all(|w| w.is_finite())
                && chem.coop_weights.iter().all(|w| w.is_finite())
                && chem.coop_s0.is_finite()
                && chem.junc_weights.iter().all(|w| w.is_finite())
                && chem.junc_w0.is_finite()
                && chem.wb_cap.is_finite()
                && chem.wb_cap >= 0.,
            "invalid chemistry model weights"
        );
        let vmax = self.vdw.iter().copied().fold(0., f64::max);
        let cell = if chem.has_solvation() {
            (2. * vmax + 2.8).max(5.25)
        } else {
            5.25
        };
        self.chem_grid = if chem.is_enabled() {
            Some(BallGrid::new(&self.centers, cell))
        } else {
            None
        };
        self.chem = chem;
        let n = self.poses.len();
        for a in 0..n {
            for b in 0..n {
                self.pair_cache[a][b].chem = 0.;
            }
        }
        let has_coop = self.chem.has_cooperativity();
        if has_coop {
            self.patch = vec![vec![PatchContact::default(); n]; n];
        } else {
            self.patch.clear();
        }
        let mut sum = 0.;
        if self.chem.is_enabled() {
            for a in 0..n {
                let pos_a = self.poses[a].position;
                for b in self.body_grid.near(&pos_a) {
                    if b > a {
                        let (_, ch, pa, pb) =
                            self.well_chem_and_patch_between(self.poses[a], self.poses[b]);
                        self.pair_cache[a][b].chem = ch;
                        self.pair_cache[b][a].chem = ch;
                        if has_coop {
                            self.patch[a][b] = pa;
                            self.patch[b][a] = pb;
                        }
                        sum += ch;
                    }
                }
            }
        }
        self.chem_sum = sum + self.coop_total_from_matrix(&self.patch, &self.poses);
        Ok(self)
    }

    /// Track contacts (some atom pair with |x_i - x_j| < r_i + r_j + gap) for library docking.
    pub fn with_contact_library(mut self, gap: f64) -> Result<Self> {
        ensure!(gap.is_finite() && gap >= 0., "invalid contact gap");
        let vmax = self.vdw.iter().copied().fold(0., f64::max);
        self.contact_grid = Some(BallGrid::new(&self.centers, 2. * vmax + gap));
        self.contact_gap = Some(gap);
        self.rebuild_body_grid();
        let n = self.poses.len();
        self.contact = vec![vec![false; n]; n];
        for a in 0..n {
            let pos_a = self.poses[a].position;
            for c in self.body_grid.near(&pos_a) {
                if c > a {
                    let k = self.in_contact(a, self.poses[a], c);
                    self.contact[a][c] = k;
                    self.contact[c][a] = k;
                }
            }
        }
        Ok(self)
    }

    /// Whether body `a` placed at `p` is in contact with body `c` at its current pose.
    fn in_contact(&self, a: usize, p: Pose, c: usize) -> bool {
        let Some(gap) = self.contact_gap else {
            return false;
        };
        if a == c {
            return false;
        }
        self.tree.within_gap(p, self.poses[c], gap)
    }

    /// Precomputed body-frame indices of exposed surface atoms (atoms whose centroid along their
    /// outward body normal by `r_i + 1.4 A` is not buried inside any other atom's `r_j + 1.4 A` ball).
    pub fn surface_atoms(&self) -> Vec<usize> {
        let probe = 1.4;
        let mut out = Vec::new();
        for (i, (&xi, &ri)) in self.centers.iter().zip(&self.vdw).enumerate() {
            let ni = self.body_normals[i];
            let test_pt = add(xi, scale(ni, ri + probe));
            let mut exposed = true;
            for (j, (&xj, &rj)) in self.centers.iter().zip(&self.vdw).enumerate() {
                if i == j {
                    continue;
                }
                let d = sub(test_pt, xj);
                let cut = rj + probe - 0.35;
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cut * cut {
                    exposed = false;
                    break;
                }
            }
            if exposed {
                out.push(i);
            }
        }
        if out.is_empty() {
            (0..self.centers.len()).collect()
        } else {
            out
        }
    }

    /// Generate `count` monomer-only surface-normal-opposed relative poses `(Pose, weight)`
    /// (plus reciprocal inverse poses for symmetry) that are hard-core valid and within
    /// `contact_gap`, optionally weighted by `exp(-beta * E_pair(identity, rel))` (clamped to `<= 25`).
    ///
    /// When `self.chem` is enabled with per-atom chemical functional group flags, systematically
    /// aligns deduplicated complementary polar/charged surface atom pairs `(i, j)` across 12
    /// axial twist angles and a 5-point local normal tilt cone (`0 deg`, `+-10 deg`) with radial
    /// outward un-clashing (`0.25 A` steps up to `2.0 A`), then selects `count` diverse
    /// low-energy dimer modes ranked by `pair_surrogate_energy`.
    pub fn generate_surface_normal_prior(
        &self,
        rng: &mut StdRng,
        count: usize,
        beta: f64,
    ) -> Vec<(Pose, f64)> {
        if count == 0 {
            return Vec::new();
        }
        let surf = self.surface_atoms();
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let placed_id = Placed::new(identity);

        if self.chem.is_enabled() && self.chem.flags.len() == self.centers.len() {
            let mut pol_rep: Vec<usize> = Vec::new();
            for &i in &surf {
                let fi = self.chem.flags[i];
                if fi == 0 {
                    continue;
                }
                let xi = self.centers[i];
                if pol_rep.iter().all(|&k| {
                    self.chem.flags[k] != fi || norm(sub(xi, self.centers[k])) >= 2.5
                }) {
                    pol_rep.push(i);
                }
            }
            let sc_rep: Vec<usize> = {
                let v: Vec<usize> = pol_rep
                    .iter()
                    .copied()
                    .filter(|&i| ChemistryModel::is_sidechain_polar(self.chem.flags[i]))
                    .collect();
                if v.is_empty() {
                    pol_rep.clone()
                } else {
                    v
                }
            };
            let mut comp_pairs: Vec<(usize, usize)> = Vec::new();
            for &i in &sc_rep {
                let fi = self.chem.flags[i];
                for &j in &sc_rep {
                    if i != j {
                        let fj = self.chem.flags[j];
                        if ChemistryModel::direct_complementary(fi, fj) {
                            comp_pairs.push((i, j));
                        }
                    }
                }
                for &j in &pol_rep {
                    if !ChemistryModel::is_sidechain_polar(self.chem.flags[j]) {
                        let fj = self.chem.flags[j];
                        if ChemistryModel::direct_complementary(fi, fj) {
                            comp_pairs.push((i, j));
                        }
                    }
                }
            }
            if !comp_pairs.is_empty() {
                let mut surf_rep: Vec<usize> = Vec::new();
                for &i in &surf {
                    let xi = self.centers[i];
                    if surf_rep
                        .iter()
                        .all(|&k| norm(sub(xi, self.centers[k])) >= 2.2)
                    {
                        surf_rep.push(i);
                    }
                }
                let sr_centers: Vec<Vec3> = surf_rep.iter().map(|&idx| self.centers[idx]).collect();
                let sr_cut = 2.6;
                let sr_cut2 = sr_cut * sr_cut;
                let sr_bound_cut2 = (self.tree.bound + sr_cut) * (self.tree.bound + sr_cut);

                let pol_centers: Vec<Vec3> = pol_rep.iter().map(|&idx| self.centers[idx]).collect();
                let pol_flags: Vec<u8> = pol_rep.iter().map(|&idx| self.chem.flags[idx]).collect();
                let pol_normals: Vec<Vec3> =
                    pol_rep.iter().map(|&idx| self.body_normals[idx]).collect();
                let pol_cut = 6.5;
                let pol_cut2 = pol_cut * pol_cut;
                let motif_cut2 = 5.8 * 5.8;
                let hb_cut2 = 4.3 * 4.3;
                let pol_bound_cut2 = (self.tree.bound + pol_cut) * (self.tree.bound + pol_cut);

                let build_voxel_csr = |pts: &[Vec3], cut: f64, cell: f64| {
                    let half = self.tree.bound + cut + cell;
                    let dim = ((2.0 * half / cell).ceil() as usize).max(1);
                    let r_exp2 = (cut + 0.5 * 3.0f64.sqrt() * cell).powi(2);
                    let mut offsets = Vec::with_capacity(dim * dim * dim + 1);
                    let mut atoms = Vec::new();
                    offsets.push(0u32);
                    for ix in 0..dim {
                        let cx = -half + (ix as f64 + 0.5) * cell;
                        for iy in 0..dim {
                            let cy = -half + (iy as f64 + 0.5) * cell;
                            for iz in 0..dim {
                                let cz = -half + (iz as f64 + 0.5) * cell;
                                for (a_idx, &xa) in pts.iter().enumerate() {
                                    let dx = xa[0] - cx;
                                    let dy = xa[1] - cy;
                                    let dz = xa[2] - cz;
                                    if dx * dx + dy * dy + dz * dz <= r_exp2 {
                                        atoms.push(a_idx as u16);
                                    }
                                }
                                offsets.push(atoms.len() as u32);
                            }
                        }
                    }
                    (half, 1.0 / cell, dim as isize, offsets, atoms)
                };

                let (sr_half, sr_inv_cell, sr_dim, sr_offsets, sr_atoms) =
                    build_voxel_csr(&sr_centers, sr_cut, 2.6);
                let (pol_half, pol_inv_cell, pol_dim, pol_offsets, pol_atoms) =
                    build_voxel_csr(&pol_centers, pol_cut, 3.25);

                let sr_clashes = |r_rel: [[f64; 3]; 3], t_rel: Vec3| -> bool {
                    let r_rel_t = transpose(r_rel);
                    let q_in_p = scale(matvec(r_rel_t, t_rel), -1.0);
                    for &xb in &sr_centers {
                        let d_center = sub(xb, q_in_p);
                        if d_center[0] * d_center[0]
                            + d_center[1] * d_center[1]
                            + d_center[2] * d_center[2]
                            > sr_bound_cut2
                        {
                            continue;
                        }
                        let xb_w = add(matvec(r_rel, xb), t_rel);
                        let ix = ((xb_w[0] + sr_half) * sr_inv_cell).floor() as isize;
                        let iy = ((xb_w[1] + sr_half) * sr_inv_cell).floor() as isize;
                        let iz = ((xb_w[2] + sr_half) * sr_inv_cell).floor() as isize;
                        if ix < 0
                            || ix >= sr_dim
                            || iy < 0
                            || iy >= sr_dim
                            || iz < 0
                            || iz >= sr_dim
                        {
                            continue;
                        }
                        let v_idx = ((ix * sr_dim + iy) * sr_dim + iz) as usize;
                        let start = sr_offsets[v_idx] as usize;
                        let end = sr_offsets[v_idx + 1] as usize;
                        for &a_u16 in &sr_atoms[start..end] {
                            let dx = sub(xb_w, sr_centers[a_u16 as usize]);
                            if dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] < sr_cut2 {
                                return true;
                            }
                        }
                    }
                    false
                };

                let score_pol_comp = |r_rel: [[f64; 3]; 3], t_rel: Vec3| -> (usize, usize) {
                    let r_rel_t = transpose(r_rel);
                    let q_in_p = scale(matvec(r_rel_t, t_rel), -1.0);
                    let mut val_a = [0u8; 256];
                    let mut motif_score = 0isize;
                    let mut area_score = 0usize;
                    for (b_idx, &xb) in pol_centers.iter().enumerate() {
                        let d_center = sub(xb, q_in_p);
                        if d_center[0] * d_center[0]
                            + d_center[1] * d_center[1]
                            + d_center[2] * d_center[2]
                            > pol_bound_cut2
                        {
                            continue;
                        }
                        let xb_w = add(matvec(r_rel, xb), t_rel);
                        let ix = ((xb_w[0] + pol_half) * pol_inv_cell).floor() as isize;
                        let iy = ((xb_w[1] + pol_half) * pol_inv_cell).floor() as isize;
                        let iz = ((xb_w[2] + pol_half) * pol_inv_cell).floor() as isize;
                        if ix < 0
                            || ix >= pol_dim
                            || iy < 0
                            || iy >= pol_dim
                            || iz < 0
                            || iz >= pol_dim
                        {
                            continue;
                        }
                        let v_idx = ((ix * pol_dim + iy) * pol_dim + iz) as usize;
                        let start = pol_offsets[v_idx] as usize;
                        let end = pol_offsets[v_idx + 1] as usize;
                        if start == end {
                            continue;
                        }
                        let nb_w = matvec(r_rel, pol_normals[b_idx]);
                        let fb = pol_flags[b_idx];
                        let scb = ChemistryModel::is_sidechain_polar(fb);
                        let mut val_b = 0u8;
                        for &a_u16 in &pol_atoms[start..end] {
                            let a_idx = a_u16 as usize;
                            let dx = sub(xb_w, pol_centers[a_idx]);
                            let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                            if d2 > pol_cut2 {
                                continue;
                            }
                            let na = pol_normals[a_idx];
                            let dot_n = na[0] * nb_w[0] + na[1] * nb_w[1] + na[2] * nb_w[2];
                            if dot_n >= -0.25 {
                                continue;
                            }
                            let fa = pol_flags[a_idx];
                            let is_comp = ChemistryModel::direct_complementary(fa, fb);
                            if is_comp {
                                area_score += 1;
                            }
                            if d2 > motif_cut2 {
                                continue;
                            }
                            if d2 <= hb_cut2
                                && (((fa & CHEM_PURE_ACC != 0) && (fb & CHEM_PURE_ACC != 0))
                                    || ((fa & CHEM_PURE_DON != 0) && (fb & CHEM_PURE_DON != 0))
                                    || ((fa & CHEM_CAT != 0) && (fb & CHEM_CAT != 0))
                                    || ((fa & CHEM_ANI != 0) && (fb & CHEM_ANI != 0)))
                            {
                                motif_score -= 2;
                                continue;
                            }
                            if is_comp && val_b < 2 && a_idx < 256 && val_a[a_idx] < 2 {
                                let sca = ChemistryModel::is_sidechain_polar(fa);
                                if !sca && !scb && d2 > hb_cut2 {
                                    continue;
                                }
                                val_b += 1;
                                val_a[a_idx] += 1;
                                let mut w = if sca && scb { 2isize } else { 1isize };
                                if ((fa & CHEM_TYR_OH != 0)
                                    && (fb & (CHEM_AMIDE_ND | CHEM_CAT) != 0))
                                    || ((fb & CHEM_TYR_OH != 0)
                                        && (fa & (CHEM_AMIDE_ND | CHEM_CAT) != 0))
                                {
                                    w += 2;
                                } else if ((fa & (CHEM_CAT | CHEM_ANI)) != 0)
                                    || ((fb & (CHEM_CAT | CHEM_ANI)) != 0)
                                {
                                    w += 1;
                                }
                                motif_score += w;
                            }
                        }
                    }
                    (motif_score.max(0) as usize, area_score)
                };

                // Stage 1: Coarse complementary patch alignment across 12 twists at gap = 1.8 A.
                let min_com_dist = 0.95 * self.tree.bound;
                let mut stage1_motif: Vec<(usize, usize, usize, usize)> = Vec::new();
                let mut stage1_area: Vec<(usize, usize, usize, usize)> = Vec::new();
                for &(i, j) in &comp_pairs {
                    let ni = self.body_normals[i];
                    let nj = self.body_normals[j];
                    let target = [-ni[0], -ni[1], -ni[2]];
                    let r_align = rotation_between_unit(nj, target);
                    let sep1 = self.vdw[i] + self.vdw[j] + 1.8;
                    let xi_shift = add(self.centers[i], scale(ni, sep1));
                    let mut best_m: Option<(usize, usize)> = None;
                    let mut best_a: Option<(usize, usize)> = None;
                    for k in 0..12usize {
                        let psi = (k as f64) * (2.0 * std::f64::consts::PI / 12.0);
                        let half_psi = 0.5 * psi;
                        let q_twist = [
                            half_psi.cos(),
                            ni[0] * half_psi.sin(),
                            ni[1] * half_psi.sin(),
                            ni[2] * half_psi.sin(),
                        ];
                        let r0 = matmul(rotation(q_twist), r_align);
                        let t0 = sub(xi_shift, matvec(r0, self.centers[j]));
                        if norm(t0) < min_com_dist || sr_clashes(r0, t0) {
                            continue;
                        }
                        let (m_score, a_score) = score_pol_comp(r0, t0);
                        if m_score >= 14 && best_m.map_or(true, |(best_c, _)| m_score > best_c) {
                            best_m = Some((m_score, k));
                        }
                        if a_score >= 18 && best_a.map_or(true, |(best_c, _)| a_score > best_c) {
                            best_a = Some((a_score, k));
                        }
                    }
                    if let Some((m_score, k)) = best_m {
                        stage1_motif.push((m_score, i, j, k));
                    }
                    if let Some((a_score, k)) = best_a {
                        stage1_area.push((a_score, i, j, k));
                    }
                }
                stage1_motif.sort_by(|a, b| b.0.cmp(&a.0));
                stage1_area.sort_by(|a, b| b.0.cmp(&a.0));

                // Filter stage1 with per-atom diversity cap before Stage 2 un-clashing,
                // allocating half the Stage-2 budget to high-specificity motif contacts and
                // half to extensive polar-network area contacts.
                let mut top_stage1: Vec<(u8, usize, usize, usize, usize)> = Vec::new();
                let motif_stage2 = (count * 3).max(180);
                let area_stage2 = (count * 5).max(320);
                for (tag, pool, limit, caps) in [
                    (0u8, &stage1_motif, motif_stage2, [2usize, 4, 8]),
                    (1u8, &stage1_area, area_stage2, [2usize, 5, 8]),
                ] {
                    let mut s1_use: std::collections::HashMap<usize, usize> =
                        std::collections::HashMap::new();
                    let mut added = 0usize;
                    for cap in caps {
                        for &(c_score, i, j, k) in pool {
                            if added >= limit {
                                break;
                            }
                            if top_stage1.iter().any(|&(t_tag, _, ti, tj, tk)| {
                                t_tag == tag && ti == i && tj == j && tk == k
                            }) {
                                continue;
                            }
                            if *s1_use.get(&i).unwrap_or(&0) >= cap
                                || *s1_use.get(&j).unwrap_or(&0) >= cap
                            {
                                continue;
                            }
                            *s1_use.entry(i).or_insert(0) += 1;
                            *s1_use.entry(j).or_insert(0) += 1;
                            top_stage1.push((tag, c_score, i, j, k));
                            added += 1;
                        }
                    }
                }

                // Stage 2: Exact 5-tilt + radial retraction un-clashing with SphereTree::overlaps.
                let tilt_half = 0.5 * 10f64.to_radians();
                let mut by_motif: Vec<(usize, usize, f64, usize, usize, Pose)> = Vec::new();
                let mut by_area: Vec<(usize, usize, f64, usize, usize, Pose)> = Vec::new();
                for &(tag, s1_score, i, j, k) in &top_stage1 {
                    let ni = self.body_normals[i];
                    let nj = self.body_normals[j];
                    let target = [-ni[0], -ni[1], -ni[2]];
                    let r_align = rotation_between_unit(nj, target);
                    let psi = (k as f64) * (2.0 * std::f64::consts::PI / 12.0);
                    let half_psi = 0.5 * psi;
                    let q_twist = [
                        half_psi.cos(),
                        ni[0] * half_psi.sin(),
                        ni[1] * half_psi.sin(),
                        ni[2] * half_psi.sin(),
                    ];
                    let r_base = matmul(rotation(q_twist), r_align);
                    let ref_ax = if ni[0].abs() < 0.9 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    let u1_raw = [
                        ni[1] * ref_ax[2] - ni[2] * ref_ax[1],
                        ni[2] * ref_ax[0] - ni[0] * ref_ax[2],
                        ni[0] * ref_ax[1] - ni[1] * ref_ax[0],
                    ];
                    let u1 = scale(u1_raw, 1.0 / norm(u1_raw).max(1e-12));
                    let u2 = [
                        ni[1] * u1[2] - ni[2] * u1[1],
                        ni[2] * u1[0] - ni[0] * u1[2],
                        ni[0] * u1[1] - ni[1] * u1[0],
                    ];
                    let r_tilts = [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        cayley(scale(u1, tilt_half)),
                        cayley(scale(u1, -tilt_half)),
                        cayley(scale(u2, tilt_half)),
                        cayley(scale(u2, -tilt_half)),
                    ];
                    let base_sep = self.vdw[i] + self.vdw[j] + 0.6;
                    let mut best_s = 99usize;
                    let mut best_pose: Option<Pose> = None;
                    for r_tilt in &r_tilts {
                        let r_rel = matmul(*r_tilt, r_base);
                        let q_rel = quaternion(r_rel);
                        let n_dir = matvec(*r_tilt, ni);
                        let rj_rot = matvec(r_rel, self.centers[j]);
                        let t0 = sub(add(self.centers[i], scale(n_dir, base_sep)), rj_rot);
                        let t_norm = norm(t0);
                        if t_norm < 1e-6 {
                            continue;
                        }
                        let t_hat = scale(t0, 1.0 / t_norm);
                        let max_s = best_s.min(8);
                        let pos_max = add(t0, scale(t_hat, 0.25 * (max_s as f64)));
                        let pose_max = Pose {
                            position: pos_max,
                            orientation: q_rel,
                        };
                        if self.tree.overlaps(&placed_id, &Placed::new(pose_max)) {
                            continue;
                        }
                        for s in 0..=max_s {
                            let pos = add(t0, scale(t_hat, 0.25 * (s as f64)));
                            let cand = Pose {
                                position: pos,
                                orientation: q_rel,
                            };
                            if !self.tree.overlaps(&placed_id, &Placed::new(cand)) {
                                if s < best_s {
                                    best_s = s;
                                    best_pose = Some(cand);
                                }
                                break;
                            }
                        }
                    }
                    if let Some(pose) = best_pose {
                        let e_pair = self.pair_surrogate_energy(identity, pose);
                        if tag == 0 {
                            let (s2_m, _) =
                                score_pol_comp(rotation(pose.orientation), pose.position);
                            by_motif.push((s2_m, s1_score, e_pair, i, j, pose));
                        } else {
                            by_area.push((0, s1_score, e_pair, i, j, pose));
                        }
                    }
                }

                // Select top poses from both the motif-specificity ranking and the
                // extensive-network ranking so both compact motif contacts (m1) and
                // broad polar contacts (m6) are represented in the memory prior.
                by_motif.sort_by(|a, b| {
                    let sa = (a.0 as f64) - 0.15 * a.2.clamp(-6.0, 6.0);
                    let sb = (b.0 as f64) - 0.15 * b.2.clamp(-6.0, 6.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                by_area.sort_by(|a, b| {
                    let sa = 0.8 * (a.1 as f64) - a.2;
                    let sb = 0.8 * (b.1 as f64) - b.2;
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut selected: Vec<(Pose, f64)> = Vec::with_capacity(count);
                let rot_cut = 20f64.to_radians();
                let half_count = count / 2;
                for (pool, target_sel, caps) in [
                    (&by_motif, half_count, [1usize, 2, usize::MAX]),
                    (&by_area, count, [2usize, 5, usize::MAX]),
                ] {
                    let mut atom_use: std::collections::HashMap<usize, usize> =
                        std::collections::HashMap::new();
                    for cap in caps {
                        for &(_, _, e_pair, i, j, pose) in pool {
                            if selected.len() >= target_sel {
                                break;
                            }
                            if *atom_use.get(&i).unwrap_or(&0) >= cap
                                || *atom_use.get(&j).unwrap_or(&0) >= cap
                            {
                                continue;
                            }
                            let r_p = rotation(pose.orientation);
                            let r_inv = transpose(r_p);
                            let inv_pos = matvec(r_inv, pose.position).map(|x| -x);
                            let distinct = selected.iter().all(|&(q, _)| {
                                let r_q = rotation(q.orientation);
                                let same_direct = if norm(sub(pose.position, q.position)) < 3.5 {
                                    let r_diff = matmul(transpose(r_p), r_q);
                                    let tr = r_diff[0][0] + r_diff[1][1] + r_diff[2][2];
                                    ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos() < rot_cut
                                } else {
                                    false
                                };
                                let same_inv = if norm(sub(inv_pos, q.position)) < 3.5 {
                                    let r_diff = matmul(transpose(r_inv), r_q);
                                    let tr = r_diff[0][0] + r_diff[1][1] + r_diff[2][2];
                                    ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos() < rot_cut
                                } else {
                                    false
                                };
                                !same_direct && !same_inv
                            });
                            if distinct {
                                *atom_use.entry(i).or_insert(0) += 1;
                                *atom_use.entry(j).or_insert(0) += 1;
                                selected.push((pose, e_pair));
                            }
                        }
                    }
                }
                if !selected.is_empty() {
                    let mut out = Vec::with_capacity(2 * selected.len());
                    let pool_span = half_count.max(1);
                    for (idx, (pose, e_pair)) in selected.into_iter().enumerate() {
                        // Weight each selected mode by its chemical-complementarity rank in
                        // its pool (by_motif or by_area) combined with a tempered Boltzmann
                        // factor so un-relaxed lock-and-key H-bond contacts (whose 0.45 A
                        // directional wells are narrower than the coarse angular grid) are
                        // not exponentially suppressed relative to broad apolar patches.
                        let rank = (idx % pool_span) as f64;
                        let w_rank = 1.0 / (1.0 + 0.12 * rank);
                        let beta_eff = (0.12 * beta).clamp(0.0, 0.15);
                        let w_boltz = if beta_eff > 0.0 {
                            (-beta_eff * e_pair.clamp(-15.0, 5.0)).exp()
                        } else {
                            1.0
                        };
                        let w = w_rank * w_boltz;
                        let r_inv = transpose(rotation(pose.orientation));
                        let inv_pose = Pose {
                            position: matvec(r_inv, pose.position).map(|x| -x),
                            orientation: quaternion(r_inv),
                        };
                        out.push((pose, 0.5 * w));
                        out.push((inv_pose, 0.5 * w));
                    }
                    let wsum: f64 = out.iter().map(|&(_, w)| w).sum();
                    if wsum > 0.0 {
                        for item in &mut out {
                            item.1 /= wsum;
                        }
                    }
                    return out;
                }
            }
        }

        let gap_max = self.contact_gap.unwrap_or(2.8).min(2.5);
        let mut out = Vec::with_capacity(2 * count);
        let max_trials = (count * 50).max(500);
        let mut accepted = 0usize;
        for _ in 0..max_trials {
            if accepted >= count {
                break;
            }
            let i = surf[rng.random_range(0..surf.len())];
            let j = surf[rng.random_range(0..surf.len())];
            let ni = self.body_normals[i];
            let nj = self.body_normals[j];
            // Construct rotation R_align mapping nj -> -ni, followed by uniform twist psi around ni.
            let target = [-ni[0], -ni[1], -ni[2]];
            let r_align = rotation_between_unit(nj, target);
            let psi = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
            let half_psi = 0.5 * psi;
            let q_twist = [
                half_psi.cos(),
                ni[0] * half_psi.sin(),
                ni[1] * half_psi.sin(),
                ni[2] * half_psi.sin(),
            ];
            // Small normal tilt jitter (~8 deg) around ni.
            let tilt_sd = 8f64.to_radians();
            let w_tilt: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                tilt_sd * z
            });
            let r_rel = matmul(
                cayley(scale(w_tilt, 0.5)),
                matmul(rotation(q_twist), r_align),
            );
            let gap = rng.random::<f64>() * gap_max;
            let sep = self.vdw[i] + self.vdw[j] + gap;
            let n_dir = matvec(cayley(scale(w_tilt, 0.5)), ni);
            let rj_rot = matvec(r_rel, self.centers[j]);
            let pos = sub(add(self.centers[i], scale(n_dir, sep)), rj_rot);
            let mut pose = Pose {
                position: pos,
                orientation: quaternion(r_rel),
            };
            let placed_p = Placed::new(pose);
            if self.tree.overlaps(&placed_id, &placed_p) {
                continue;
            }
            let mut e_pair = self.pair_surrogate_energy(identity, pose);
            // Short greedy local contact relaxation (8 steps) under the monomer-monomer pair potential.
            for _ in 0..8 {
                let dt: Vec3 = std::array::from_fn(|_| {
                    let z: f64 = StandardNormal.sample(rng);
                    0.35 * z
                });
                let dw: Vec3 = std::array::from_fn(|_| {
                    let z: f64 = StandardNormal.sample(rng);
                    0.5 * 2.5f64.to_radians() * z
                });
                let cand = Pose {
                    position: add(pose.position, dt),
                    orientation: quaternion(matmul(cayley(dw), rotation(pose.orientation))),
                };
                if !self.tree.overlaps(&placed_id, &Placed::new(cand))
                    && self.in_contact_poses(identity, cand)
                {
                    let e_cand = self.pair_surrogate_energy(identity, cand);
                    if e_cand <= e_pair {
                        pose = cand;
                        e_pair = e_cand;
                    }
                }
            }
            let w = if beta > 0. {
                (-beta * e_pair).clamp(-25.0, 25.0).exp()
            } else {
                1.0
            };
            let r_inv = transpose(rotation(pose.orientation));
            let inv_pose = Pose {
                position: matvec(r_inv, pose.position).map(|x| -x),
                orientation: quaternion(r_inv),
            };
            out.push((pose, 0.5 * w));
            out.push((inv_pose, 0.5 * w));
            accepted += 1;
        }
        let wsum: f64 = out.iter().map(|&(_, w)| w).sum();
        if wsum > 0.0 {
            for item in &mut out {
                item.1 /= wsum;
            }
        }
        out
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

    fn weighted_memory_slots(&self, beta: f64) -> Vec<(Pose, f64)> {
        let Some(mem) = &self.memory else {
            return Vec::new();
        };
        if beta == 0.0 {
            mem.slots.iter().map(|&p| (p, 1.0)).collect()
        } else {
            mem.slots
                .iter()
                .zip(&mem.slot_energies)
                .map(|(&p, &e)| (p, (-beta * e).clamp(-25.0, 25.0).exp()))
                .collect()
        }
    }

    /// Weighted relative poses `(Pose, weight)` of the current contacts between bodies other than `b`
    /// weighted by `exp(-beta * E_pair(i, j))` (or `1.0` when `beta == 0`), plus any Poisson memory
    /// slots weighted by `exp(-beta * E_pair(identity, m))`. Depends only on spectator bodies `i, j != b`
    /// and memory slots.
    pub fn weighted_dock_components(&self, b: usize, d: &DockSettings) -> Vec<(Pose, f64)> {
        let beta = d.library_beta;
        if d.memory_only {
            return self.weighted_memory_slots(beta);
        }
        if !d.library {
            return d.components.iter().map(|&p| (p, 1.0)).collect();
        }
        let mut out = Vec::new();
        if !self.contact.is_empty() {
            for i in 0..self.poses.len() {
                if i == b {
                    continue;
                }
                for j in (i + 1)..self.poses.len() {
                    if j == b || !self.contact[i][j] {
                        continue;
                    }
                    let w = if beta != 0.0 {
                        let pc = self.pair_cache[i][j];
                        let e_pair = -self.well.epsilon * (pc.well as f64)
                            + self.like.weight * pc.like
                            + pc.chem;
                        (-beta * e_pair).clamp(-25.0, 25.0).exp()
                    } else {
                        1.0
                    };
                    out.push((rel_pose(self.poses[i], self.poses[j]), w));
                    out.push((rel_pose(self.poses[j], self.poses[i]), w));
                }
            }
        }
        out.extend(self.weighted_memory_slots(beta));
        out
    }

    /// Weighted relative poses `(Pose, weight)` of spectator contacts outside `in_cluster`.
    pub fn weighted_dock_components_excluding(
        &self,
        in_cluster: &[bool],
        d: &DockSettings,
    ) -> Vec<(Pose, f64)> {
        let beta = d.library_beta;
        if d.memory_only {
            return self.weighted_memory_slots(beta);
        }
        if !d.library {
            return d.components.iter().map(|&p| (p, 1.0)).collect();
        }
        let mut out = Vec::new();
        if !self.contact.is_empty() {
            for i in 0..self.poses.len() {
                if in_cluster[i] {
                    continue;
                }
                for j in (i + 1)..self.poses.len() {
                    if in_cluster[j] || !self.contact[i][j] {
                        continue;
                    }
                    let w = if beta != 0.0 {
                        let pc = self.pair_cache[i][j];
                        let e_pair = -self.well.epsilon * (pc.well as f64)
                            + self.like.weight * pc.like
                            + pc.chem;
                        (-beta * e_pair).clamp(-25.0, 25.0).exp()
                    } else {
                        1.0
                    };
                    out.push((rel_pose(self.poses[i], self.poses[j]), w));
                    out.push((rel_pose(self.poses[j], self.poses[i]), w));
                }
            }
        }
        out.extend(self.weighted_memory_slots(beta));
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
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let placed_id = Placed::new(identity);
        let entries = self.library(usize::MAX);
        let n0 = MemorySettings::poisson(rng, s.total(entries.len()));
        let slots: Vec<Pose> = (0..n0)
            .map(|_| s.draw(rng, &entries, &prior, &self.tree))
            .collect();
        let prior_values = slots.iter().map(|&m| s.prior_value(&prior, m)).collect();
        let slot_energies = slots
            .iter()
            .map(|&m| {
                if self.tree.overlaps(&placed_id, &Placed::new(m)) {
                    0.0
                } else {
                    self.pair_surrogate_energy(identity, m)
                }
            })
            .collect();
        self.memory = Some(Memory {
            s,
            prior,
            prior_values,
            slot_energies,
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
        let Some(mem) = &self.memory else {
            return;
        };
        let s = mem.s;
        let prior = mem.prior.clone();
        let mut slots = Vec::with_capacity(mem.slots.len());
        let mut slot_energies = Vec::with_capacity(mem.slots.len());
        let mut sums = Vec::with_capacity(mem.slots.len());
        let mut prior_values = Vec::with_capacity(mem.slots.len());
        for (((m, sum), pv), se) in mem
            .slots
            .iter()
            .zip(&mem.sums)
            .zip(&mem.prior_values)
            .zip(&mem.slot_energies)
        {
            if rng.random::<f64>() >= s.death {
                slots.push(*m);
                sums.push(*sum);
                prior_values.push(*pv);
                slot_energies.push(*se);
            }
        }
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let placed_id = Placed::new(identity);
        let births = MemorySettings::poisson(rng, s.death * s.total(entries.len()));
        for _ in 0..births {
            let m = s.draw(rng, &entries, &prior, &self.tree);
            sums.push(entries.iter().map(|&g| s.kernel(m, g)).sum());
            prior_values.push(s.prior_value(&prior, m));
            let e_slot = if self.tree.overlaps(&placed_id, &Placed::new(m)) {
                0.0
            } else {
                self.pair_surrogate_energy(identity, m)
            };
            slot_energies.push(e_slot);
            slots.push(m);
        }
        if let Some(mem) = &mut self.memory {
            mem.slots = slots;
            mem.slot_energies = slot_energies;
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
        let mut partners: Vec<usize> = self
            .body_grid
            .near(&new.position)
            .filter(|&c| c != b && self.in_contact(b, new, c))
            .collect();
        partners.sort_unstable();
        let mut old_g = Vec::new();
        for c in self.body_grid.near(&old.position) {
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
        if old_g.is_empty() && new_g.is_empty() {
            return (0., Vec::new(), n_new, partners);
        }
        let mut sums = mem.sums.clone();
        let mut d = -mem.s.lambda * (n_new as f64 - mem.entries as f64);
        for (k, m) in mem.slots.iter().enumerate() {
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

    /// Commit the contacts (and memory sums) of an accepted move of body `b` from `old_pos` to `new_pos`.
    fn commit_memory(
        &mut self,
        b: usize,
        old_pos: Vec3,
        sums: Vec<f64>,
        entries: usize,
        partners: &[usize],
    ) {
        if let Some(mem) = &mut self.memory {
            if !sums.is_empty() {
                mem.sums = sums;
            }
            mem.entries = entries;
            for c in self.body_grid.near(&old_pos) {
                if c != b {
                    self.contact[b][c] = false;
                    self.contact[c][b] = false;
                }
            }
            for &c in partners {
                self.contact[b][c] = true;
                self.contact[c][b] = true;
            }
        } else if self.contact_gap.is_some() {
            for c in self.body_grid.near(&old_pos) {
                if c != b {
                    self.contact[b][c] = false;
                    self.contact[c][b] = false;
                }
            }
            let new_pos = self.poses[b].position;
            let cand: Vec<usize> = self.body_grid.near(&new_pos).filter(|&c| c != b).collect();
            for c in cand {
                let k = self.in_contact(b, self.poses[b], c);
                self.contact[b][c] = k;
                self.contact[c][b] = k;
            }
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
        for c in self.body_grid.near(&p.position) {
            if c != b {
                sum += self.chem_pair_between(p, self.poses[c]);
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
        for c in self.body_grid.near_within(&p.position, reach) {
            let q = self.poses[c];
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
        let mut count = 0;
        for c in self.body_grid.near(&p.position) {
            if c != b {
                count += self.well_pair_between(p, self.poses[c]);
            }
        }
        count
    }

    /// Combined `(well_pairs, chem_energy)` between body `b` placed at `p` and all other bodies,
    /// evaluated in a single spatial-grid pass per interacting body pair (including any 3-body
    /// cooperative pocket energy when `self.chem.has_cooperativity()`).
    pub fn well_and_chem_pairs(&self, b: usize, p: Pose) -> (usize, f64) {
        if !self.chem.is_enabled() {
            return (self.well_pairs(b, p), 0.);
        }
        if !self.chem.has_cooperativity() {
            let mut wp = 0usize;
            let mut ch = 0.0f64;
            for c in self.body_grid.near(&p.position) {
                if c != b {
                    let (w_pq, c_pq) = self.well_and_chem_pair_between(p, self.poses[c]);
                    wp += w_pq;
                    ch += c_pq;
                }
            }
            return (wp, ch);
        }
        let mut wp = 0usize;
        let mut ch = 0.0f64;
        let mut new_pairs = Vec::new();
        for c in self.body_grid.near(&p.position) {
            if c != b {
                let (w_pq, c_pq, pb, pc) = self.well_chem_and_patch_between(p, self.poses[c]);
                wp += w_pq;
                ch += c_pq;
                if pb.s > 0.5 || pc.s > 0.5 {
                    new_pairs.push((c, pb, pc));
                }
            }
        }
        ch += self.coop_single_body_sparse(b, p, &new_pairs);
        (wp, ch)
    }

    /// Body-frame 3-body junction normal vector `sum_{i in v} u_i^(v,c1) u_i^(v,c2) n_i^body`
    /// on vertex body `v` from its contacts with `c1` and `c2`.
    #[inline]
    fn vertex_junction_normal(&self, p1: &PatchContact, p2: &PatchContact) -> Vec3 {
        let (Some(u1), Some(u2)) = (&p1.u6, &p2.u6) else {
            return [0.0; 3];
        };
        let mut g = [0.0f64; 3];
        for ((&a, &b), &nb) in u1.iter().zip(u2.iter()).zip(self.body_normals.iter()) {
            let w = a * b;
            if w > 0.0 {
                g[0] += w * nb[0];
                g[1] += w * nb[1];
                g[2] += w * nb[2];
            }
        }
        g
    }

    /// Triangle normal-cancellation stabilization `W_ABC (1 - R_ABC^2) [w_sat + w_seal (1 - R_ABC^2) / (W_ABC + W_0)]`
    /// subtracted when a closed 3-body triangle `(A, B, C)` seals its triple-junction normals (`N_A + N_B + N_C -> 0`).
    #[inline]
    fn triangle_junction_cancel_term(&self, n_a: Vec3, n_b: Vec3, n_c: Vec3) -> f64 {
        let w_abc = norm(n_a) + norm(n_b) + norm(n_c);
        if w_abc <= 1e-12 {
            return 0.0;
        }
        let sum_n = add(add(n_a, n_b), n_c);
        let r_abc = (norm(sum_n) / w_abc).clamp(0.0, 1.0);
        let one_m_r2 = 1.0 - r_abc * r_abc;
        if one_m_r2 <= 0.0 {
            return 0.0;
        }
        let w_seal = self.chem.junc_weights[0];
        let w_sat = self.chem.junc_weights[1];
        let w0 = if self.chem.junc_w0 > 0.0 {
            self.chem.junc_w0
        } else {
            4.0
        };
        w_abc * one_m_r2 * (w_sat + w_seal * one_m_r2 / (w_abc + w0))
    }

    /// Cooperative 3-body contribution of all wedges `(c1, b, c2)` and `(b, c1, d)` involving
    /// body `b` at `pose_b` given its sparse list of active patch contacts `contacts: &[(c, pb, pc)]`.
    fn coop_single_body_sparse(
        &self,
        b: usize,
        pose_b: Pose,
        contacts: &[(usize, PatchContact, PatchContact)],
    ) -> f64 {
        if !self.chem.has_cooperativity() || self.patch.is_empty() || contacts.is_empty() {
            return 0.0;
        }
        let w_wedge = self.chem.coop_weights[0];
        let w_tri = self.chem.coop_weights[1];
        let has_junc = self.chem.has_junction();
        let w_sat = self.chem.junc_weights[1];
        let r_b = if has_junc {
            rotation(pose_b.orientation)
        } else {
            [[0.0; 3]; 3]
        };
        let lookup_b = |d: usize| -> (f64, f64) {
            for &(c, ref pb, ref pc) in contacts {
                if c == d {
                    return (pb.s, pc.s);
                }
            }
            (0.0, 0.0)
        };
        let mut e_coop = 0.0f64;
        // 1. Vertex on body b and triangles {b, c1, c2}: pairs (c1, c2) both contacting b.
        for idx1 in 0..contacts.len() {
            let (c1, ref pb_c1, ref pc_c1) = contacts[idx1];
            let phi_bc1 = self.chem.coop_phi(pb_c1.s);
            let u_bc1 = pb_c1.u;
            for &(c2, ref pb_c2, ref pc_c2) in &contacts[idx1 + 1..] {
                let phi_bc2 = self.chem.coop_phi(pb_c2.s);
                let u_bc2 = pb_c2.u;
                let mut nb_body = [0.0; 3];
            let mut nb_computed = false;
            if phi_bc1 > 0.0 && phi_bc2 > 0.0 {
                let psi_b = ChemistryModel::coop_psi(u_bc1, u_bc2);
                let wedge_b = phi_bc1 * phi_bc2 * psi_b;
                let phi_c1c2 = 0.5
                    * (self.chem.coop_phi(self.patch[c1][c2].s)
                        + self.chem.coop_phi(self.patch[c2][c1].s));
                e_coop -= (w_wedge + w_tri * phi_c1c2) * wedge_b;
                if has_junc {
                    nb_body = self.vertex_junction_normal(pb_c1, pb_c2);
                    nb_computed = true;
                    e_coop += w_sat * norm(nb_body);
                }
            }
            if has_junc && (self.patch[c1][c2].s > 0.5 || self.patch[c2][c1].s > 0.5) {
                if !nb_computed {
                    nb_body = self.vertex_junction_normal(pb_c1, pb_c2);
                }
                let nc1_body = self.vertex_junction_normal(pc_c1, &self.patch[c1][c2]);
                let nc2_body = self.vertex_junction_normal(pc_c2, &self.patch[c2][c1]);
                let n_b = matvec(r_b, nb_body);
                let n_c1 = matvec(rotation(self.poses[c1].orientation), nc1_body);
                let n_c2 = matvec(rotation(self.poses[c2].orientation), nc2_body);
                e_coop -= self.triangle_junction_cancel_term(n_b, n_c1, n_c2);
            }
        }
        }
        // 2. Vertex on neighbor c: wedge (b, d) on vertex c for d != b, c.
        for &(c, _, ref pc_c) in contacts {
            let phi_cb = self.chem.coop_phi(pc_c.s);
            if phi_cb <= 0.0 {
                continue;
            }
            let u_cb = pc_c.u;
            let pos_c = self.poses[c].position;
            for d in self.body_grid.near(&pos_c) {
                if d == b || d == c {
                    continue;
                }
                let p_cd = &self.patch[c][d];
                if p_cd.s <= 0.5 {
                    continue;
                }
                let phi_cd = self.chem.coop_phi(p_cd.s);
                let psi_c = ChemistryModel::coop_psi(u_cb, p_cd.u);
                let wedge_c = phi_cb * phi_cd * psi_c;
                let (s_bd, s_db) = lookup_b(d);
                let phi_bd = 0.5 * (self.chem.coop_phi(s_bd) + self.chem.coop_phi(s_db));
                e_coop -= (w_wedge + w_tri * phi_bd) * wedge_c;
                if has_junc {
                    let nc_body = self.vertex_junction_normal(pc_c, p_cd);
                    e_coop += w_sat * norm(nc_body);
                }
            }
        }
        e_coop
    }

    /// Total 3-body cooperative energy across the full `n x n` patch matrix `mat` at body `poses`.
    fn coop_total_from_matrix(&self, mat: &[Vec<PatchContact>], poses: &[Pose]) -> f64 {
        if !self.chem.has_cooperativity() || mat.is_empty() {
            return 0.0;
        }
        let n = mat.len();
        let w_wedge = self.chem.coop_weights[0];
        let w_tri = self.chem.coop_weights[1];
        let has_junc = self.chem.has_junction();
        let w_sat = self.chem.junc_weights[1];
        let rots: Vec<[[f64; 3]; 3]> = if has_junc {
            poses.iter().map(|p| rotation(p.orientation)).collect()
        } else {
            Vec::new()
        };
        let mut e_coop = 0.0f64;
        for i in 0..n {
            let mut active: Vec<(usize, f64, Vec3)> = Vec::new();
            for j in 0..n {
                if j != i && mat[i][j].s > 0.5 {
                    active.push((j, self.chem.coop_phi(mat[i][j].s), mat[i][j].u));
                }
            }
            for idx1 in 0..active.len() {
                let (j, phi_ij, u_ij) = active[idx1];
                for &(k, phi_ik, u_ik) in &active[idx1 + 1..] {
                    let psi_i = ChemistryModel::coop_psi(u_ij, u_ik);
                    let wedge_i = phi_ij * phi_ik * psi_i;
                    let phi_jk =
                        0.5 * (self.chem.coop_phi(mat[j][k].s) + self.chem.coop_phi(mat[k][j].s));
                    e_coop -= (w_wedge + w_tri * phi_jk) * wedge_i;
                    if has_junc {
                        let ni_body = self.vertex_junction_normal(&mat[i][j], &mat[i][k]);
                        e_coop += w_sat * norm(ni_body);
                    }
                }
            }
        }
        if has_junc {
            for i in 0..n {
                for j in (i + 1)..n {
                    if mat[i][j].s <= 0.5 && mat[j][i].s <= 0.5 {
                        continue;
                    }
                    for k in (j + 1)..n {
                        if (mat[i][k].s <= 0.5 && mat[k][i].s <= 0.5)
                            || (mat[j][k].s <= 0.5 && mat[k][j].s <= 0.5)
                        {
                            continue;
                        }
                        let ni = matvec(rots[i], self.vertex_junction_normal(&mat[i][j], &mat[i][k]));
                        let nj = matvec(rots[j], self.vertex_junction_normal(&mat[j][i], &mat[j][k]));
                        let nk = matvec(rots[k], self.vertex_junction_normal(&mat[k][i], &mat[k][j]));
                        e_coop -= self.triangle_junction_cancel_term(ni, nj, nk);
                    }
                }
            }
        }
        e_coop
    }

    /// Evaluate sparse `SingleBodyDelta` for moving single body `b` from `old` to `new`
    /// using `self.body_grid` and cached `self.pair_cache[b]`.
    fn single_body_well_and_chem_delta(&self, b: usize, old: Pose, new: Pose) -> SingleBodyDelta {
        let has_coop = self.chem.has_cooperativity();
        let mut old_neighbors = Vec::new();
        let mut wp_old = 0usize;
        let mut ch_old_pair = 0.0f64;
        let mut old_coop_contacts = Vec::new();
        for c in self.body_grid.near(&old.position) {
            if c != b {
                old_neighbors.push(c);
                let pc = self.pair_cache[b][c];
                wp_old += pc.well as usize;
                ch_old_pair += pc.chem;
                if has_coop && (self.patch[b][c].s > 0.5 || self.patch[c][b].s > 0.5) {
                    old_coop_contacts.push((c, self.patch[b][c].clone(), self.patch[c][b].clone()));
                }
            }
        }
        let mut wp_new = 0usize;
        let mut ch_new_pair = 0.0f64;
        let mut new_pairs = Vec::new();
        let mut new_coop_contacts = Vec::new();
        for c in self.body_grid.near(&new.position) {
            if c != b {
                let (w_pq, c_pq, pb, pc) = self.well_chem_and_patch_between(new, self.poses[c]);
                wp_new += w_pq;
                ch_new_pair += c_pq;
                if has_coop && (pb.s > 0.5 || pc.s > 0.5) {
                    new_coop_contacts.push((c, pb.clone(), pc.clone()));
                }
                if w_pq > 0 || c_pq != 0.0 || (has_coop && (pb.s > 0.5 || pc.s > 0.5)) {
                    new_pairs.push((c, w_pq as u32, c_pq, pb, pc));
                }
            }
        }
        let coop_delta = if has_coop {
            self.coop_single_body_sparse(b, new, &new_coop_contacts)
                - self.coop_single_body_sparse(b, old, &old_coop_contacts)
        } else {
            0.0
        };
        SingleBodyDelta {
            dp_well: wp_new as i64 - wp_old as i64,
            dc_chem: (ch_new_pair - ch_old_pair) + coop_delta,
            old_neighbors,
            new_pairs,
        }
    }

    fn commit_single_body_delta(
        &mut self,
        b: usize,
        delta: SingleBodyDelta,
        old_like_neighbors: &[usize],
        new_likes: &[(usize, f64)],
    ) {
        let has_coop = !self.patch.is_empty();
        for &c in &delta.old_neighbors {
            self.pair_cache[b][c].well = 0;
            self.pair_cache[c][b].well = 0;
            self.pair_cache[b][c].chem = 0.0;
            self.pair_cache[c][b].chem = 0.0;
            if has_coop {
                self.patch[b][c] = PatchContact::default();
                self.patch[c][b] = PatchContact::default();
            }
        }
        for &c in old_like_neighbors {
            self.pair_cache[b][c].like = 0.0;
            self.pair_cache[c][b].like = 0.0;
        }
        for (c, wp, ch, pb, pc) in delta.new_pairs {
            self.pair_cache[b][c].well = wp;
            self.pair_cache[c][b].well = wp;
            self.pair_cache[b][c].chem = ch;
            self.pair_cache[c][b].chem = ch;
            if has_coop {
                self.patch[b][c] = pb;
                self.patch[c][b] = pc;
            }
        }
        for &(c, lk) in new_likes {
            self.pair_cache[b][c].like = lk;
            self.pair_cache[c][b].like = lk;
        }
    }

    fn single_body_like_delta(
        &self,
        b: usize,
        old: Pose,
        new: Pose,
    ) -> (f64, Vec<usize>, Vec<(usize, f64)>) {
        if self.like.weight == 0. || self.like.groups.is_empty() {
            return (0., Vec::new(), Vec::new());
        }
        let reach = 2. * self.charge_extent + self.like.cutoff;
        let mut lk_old = 0.0f64;
        let mut old_like_neighbors = Vec::new();
        for c in self.body_grid.near_within(&old.position, reach) {
            if c != b {
                let lk = self.pair_cache[b][c].like;
                if lk != 0.0 {
                    lk_old += lk;
                    old_like_neighbors.push(c);
                }
            }
        }
        let mut lk_new = 0.0f64;
        let mut new_likes = Vec::new();
        for c in self.body_grid.near_within(&new.position, reach) {
            if c != b {
                let lk = self.like_pair_between(new, self.poses[c]);
                if lk != 0.0 {
                    lk_new += lk;
                    new_likes.push((c, lk));
                }
            }
        }
        (lk_new - lk_old, old_like_neighbors, new_likes)
    }

    fn hard_valid(&self, b: usize, p: Pose) -> bool {
        if !self.wall.contains(p) {
            return false;
        }
        let placed = Placed::new(p);
        let reach = 2. * self.tree.bound;
        let r2 = reach * reach;
        for c in self.body_grid.near(&p.position) {
            if c == b {
                continue;
            }
            let d = sub(self.poses[c].position, p.position);
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2
                && self.tree.overlaps(&placed, &self.placed[c])
            {
                return false;
            }
        }
        true
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
        let delta = self.single_body_well_and_chem_delta(b, old, new);
        let d_well = -self.well.epsilon * delta.dp_well as f64;
        let (dl, old_likes, new_likes) = self.single_body_like_delta(b, old, new);
        let update = self.solv.propose(&[(b, new)])?;
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        // Minus the log joint-target ratio: energy change minus the memory-conditional ratio.
        let de = update.delta + d_well + self.like.weight * dl + delta.dc_chem - dm;
        ensure!(!de.is_nan(), "NaN energy change");
        if de <= 0. || rng.random::<f64>() < (-de).exp() {
            self.solv.commit(update);
            self.pair_count += delta.dp_well;
            self.like_sum += dl;
            self.chem_sum += delta.dc_chem;
            self.commit_single_body_delta(b, delta, &old_likes, &new_likes);
            self.body_grid.relocate(b, &old.position, &new.position);
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.commit_memory(b, old.position, sums, n_entries, &partners);
            self.stats.accepted += 1;
            self.stats.reinsertion_accepted += u64::from(reinsertion);
            return Ok(true);
        }
        Ok(false)
    }

    /// Log of the docking mixture density of pose `x` of body `b` (anchors: all others), with
    /// weighted Gaussian components `comps: &[(Pose, f64)]` and uniform weight `u` (1 when empty).
    fn dock_log_density(
        &self,
        b: usize,
        x: Pose,
        d: &DockSettings,
        comps: &[(Pose, f64)],
        w_sum: f64,
        u: f64,
    ) -> f64 {
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
            let near: Vec<Pose> = self
                .body_grid
                .near_within(&x.position, cut)
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
        let (st, sr) = (d.sigma_translation, d.sigma_rotation);
        let norm_t = -1.5 * (2. * std::f64::consts::PI * st * st).ln();
        let norm_r = (8. * std::f64::consts::PI * std::f64::consts::PI).ln()
            - 1.5 * (2. * std::f64::consts::PI * sr * sr).ln();
        let mut terms = Vec::new();
        let anchors = self.poses.len() - 1;
        if comps.is_empty() || w_sum <= 0. {
            return uniform;
        }
        let lw_base = (1. - u).ln() - ((anchors as f64) * w_sum).ln();
        let max_anchor_dist = 2. * self.tree.bound + self.contact_gap.unwrap_or(6.) + 10. * st;
        let max_d2 = max_anchor_dist * max_anchor_dist;
        let max_t2 = 100. * st * st;
        let inv_2st2 = 1. / (2. * st * st);
        let inv_2sr2 = 1. / (2. * sr * sr);
        for a in self.body_grid.near_within(&x.position, max_anchor_dist) {
            if a == b {
                continue;
            }
            let pa = self.poses[a];
            let dx = sub(x.position, pa.position);
            if dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] > max_d2 {
                continue;
            }
            let rat = transpose(rotation(pa.orientation));
            let trel = matvec(rat, dx);
            let qrel = quat_mul(quat_conj(pa.orientation), x.orientation);
            for &(c, w_c) in comps {
                let dt = sub(trel, c.position);
                let t2 = dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2];
                if t2 > max_t2 {
                    continue;
                }
                let cos_half = quat_abs_dot(c.orientation, qrel);
                let theta = 2. * cos_half.acos();
                // Exponential coordinates: Haar factor theta^2 / (2 (1 - cos theta)).
                let jac = if theta < 1e-8 {
                    1.
                } else {
                    theta * theta / (2. * (1. - theta.cos()))
                };
                terms.push(
                    lw_base + w_c.ln() + norm_t - t2 * inv_2st2 + norm_r
                        - theta * theta * inv_2sr2
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
                && (d.learned.is_some() || d.library || d.memory_only || !d.components.is_empty()),
            "invalid docking setup"
        );
        ensure!(
            !(d.library || d.memory_only) || (self.contact_gap.is_some() && d.learned.is_none()),
            "library/memory docking needs with_contact_library and no learned model"
        );
        ensure!(
            !d.memory_only || self.memory.is_some(),
            "memory_only docking needs with_memory"
        );
        ensure!(
            d.uniform_weight > 0. && d.uniform_weight < 1.,
            "docking uniform weight must lie in (0, 1)"
        );
        let b = free[rng.random_range(0..free.len())];
        let old = self.poses[b];
        self.stats.dock_attempts += 1;
        // Components: the contact library of the other bodies (unchanged by this move), or
        // the Poisson memory slots only when `memory_only` is set, or the fixed list.
        // Without components the proposal is uniform with probability one.
        let comps: Vec<(Pose, f64)> = if d.learned.is_some() {
            Vec::new()
        } else {
            self.weighted_dock_components(b, d)
        };
        let w_sum: f64 = comps.iter().map(|&(_, w)| w).sum();
        let u = if d.learned.is_none() && (comps.is_empty() || w_sum <= 0.) {
            1.
        } else {
            d.uniform_weight
        };
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
            let c = if d.library_beta == 0. {
                comps[rng.random_range(0..comps.len())].0
            } else {
                let mut v = rng.random::<f64>() * w_sum;
                let mut c = comps[comps.len() - 1].0;
                for &(pose_c, w_c) in &comps {
                    if v < w_c {
                        c = pose_c;
                        break;
                    }
                    v -= w_c;
                }
                c
            };
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
        let delta = self.single_body_well_and_chem_delta(b, old, new);
        let d_well = -self.well.epsilon * delta.dp_well as f64;
        let (dl, old_likes, new_likes) = self.single_body_like_delta(b, old, new);
        let update = self.solv.propose(&[(b, new)])?;
        let log_ratio = self.dock_log_density(b, old, d, &comps, w_sum, u)
            - self.dock_log_density(b, new, d, &comps, w_sum, u);
        let (dm, sums, n_entries, partners) = self.memory_delta(b, new);
        let a = -(update.delta + d_well + self.like.weight * dl + delta.dc_chem) + dm + log_ratio;
        ensure!(!a.is_nan(), "NaN docking acceptance");
        if a >= 0. || rng.random::<f64>() < a.exp() {
            self.solv.commit(update);
            self.pair_count += delta.dp_well;
            self.like_sum += dl;
            self.chem_sum += delta.dc_chem;
            self.commit_single_body_delta(b, delta, &old_likes, &new_likes);
            self.body_grid.relocate(b, &old.position, &new.position);
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.commit_memory(b, old.position, sums, n_entries, &partners);
            self.stats.accepted += 1;
            self.stats.dock_accepted += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// Pairwise square-well atom contacts between pose `p` and pose `q`.
    pub fn well_pair_between(&self, p: Pose, q: Pose) -> usize {
        let reach = 2. * self.tree.bound + self.well.width;
        let dpq = sub(q.position, p.position);
        if dpq[0] * dpq[0] + dpq[1] * dpq[1] + dpq[2] * dpq[2] > reach * reach {
            return 0;
        }
        let rb = rotation(p.orientation);
        let rct = transpose(rotation(q.orientation));
        let r_rel = matmul(rct, rb);
        let t_rel = matvec(rct, sub(p.position, q.position));
        let rbt = transpose(rb);
        let q_in_p = matvec(rbt, dpq);
        let mut active_a = Vec::new();
        self.tree
            .active_atoms_in_frame(&r_rel, &t_rel, &q_in_p, self.well.width, &mut active_a);
        let mut count = 0;
        for (i, y) in active_a {
            let ri = self.vdw[i];
            for j in self.template_grid.near(&y) {
                let r_cut = ri + self.vdw[j] + self.well.width;
                let diff = sub(self.centers[j], y);
                if diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] < r_cut * r_cut {
                    count += 1;
                }
            }
        }
        count
    }

    /// Fused single-pass evaluation of `(well_pairs, chem_energy)` between pose `p` and pose `q`.
    pub fn well_and_chem_pair_between(&self, p: Pose, q: Pose) -> (usize, f64) {
        let (w, c, _, _) = self.well_chem_and_patch_between_inner(p, q, false);
        (w, c)
    }

    /// Fused single-pass evaluation of `(well_pairs, chem_energy, patch_p_from_q, patch_q_from_p)`.
    pub fn well_chem_and_patch_between(
        &self,
        p: Pose,
        q: Pose,
    ) -> (usize, f64, PatchContact, PatchContact) {
        self.well_chem_and_patch_between_inner(p, q, self.chem.has_cooperativity())
    }

    /// Directed 2.8 A hydration-shell patch contacts `(patch_p_from_q, patch_q_from_p)` between pose `p` and pose `q`.
    pub fn patch_pair_between(&self, p: Pose, q: Pose) -> (PatchContact, PatchContact) {
        let reach = 2. * self.tree.bound + 2.8;
        let dpq = sub(q.position, p.position);
        if dpq[0] * dpq[0] + dpq[1] * dpq[1] + dpq[2] * dpq[2] > reach * reach {
            return (PatchContact::default(), PatchContact::default());
        }
        let n = self.centers.len();
        let rp = rotation(p.orientation);
        let rqt = transpose(rotation(q.orientation));
        let r_rel = matmul(rqt, rp);
        let t_rel = matvec(rqt, sub(p.position, q.position));
        let rpt = transpose(rp);
        let q_in_p = matvec(rpt, dpq);
        let mut active_a = Vec::new();
        self.tree
            .active_atoms_in_frame(&r_rel, &t_rel, &q_in_p, 2.8, &mut active_a);
        if active_a.is_empty() {
            return (PatchContact::default(), PatchContact::default());
        }
        let mut u6_a = Vec::new();
        let mut u6_b = Vec::new();
        let mut any = false;
        for (i, ya) in active_a {
            let ri = self.vdw[i];
            for j in self.template_grid.near(&ya) {
                let r_sum = ri + self.vdw[j];
                let r_cut = r_sum + 2.8;
                let diff = sub(ya, self.centers[j]);
                let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if d2 < r_cut * r_cut {
                    if !any {
                        u6_a.resize(n, 0.0);
                        u6_b.resize(n, 0.0);
                        any = true;
                    }
                    let g = (d2.sqrt() - r_sum).max(0.0);
                    let u6 = ((2.8 - g) * (1.0 / 2.8)).clamp(0.0, 1.0);
                    if u6 > u6_a[i] {
                        u6_a[i] = u6;
                    }
                    if u6 > u6_b[j] {
                        u6_b[j] = u6;
                    }
                }
            }
        }
        if !any {
            return (PatchContact::default(), PatchContact::default());
        }
        (
            self.build_patch_contact(&u6_a),
            self.build_patch_contact(&u6_b),
        )
    }

    fn build_patch_contact(&self, u6: &[f64]) -> PatchContact {
        let mut s = 0.0f64;
        let mut v = [0.0f64; 3];
        for (i, &u) in u6.iter().enumerate() {
            if u > 0.0 {
                s += u;
                let x = self.centers[i];
                v[0] += u * x[0];
                v[1] += u * x[1];
                v[2] += u * x[2];
            }
        }
        let vn = norm(v);
        if s > 0.5 && vn > 1e-12 {
            let u6_arc = if self.chem.has_junction() {
                Some(Arc::from(u6))
            } else {
                None
            };
            PatchContact {
                s,
                u: [v[0] / vn, v[1] / vn, v[2] / vn],
                u6: u6_arc,
            }
        } else {
            PatchContact::default()
        }
    }

    fn well_chem_and_patch_between_inner(
        &self,
        p: Pose,
        q: Pose,
        need_patch: bool,
    ) -> (usize, f64, PatchContact, PatchContact) {
        if !self.chem.is_enabled() {
            return (
                self.well_pair_between(p, q),
                0.,
                PatchContact::default(),
                PatchContact::default(),
            );
        }
        if self.well.width < 2.8 {
            let (pa, pb) = if need_patch {
                self.patch_pair_between(p, q)
            } else {
                (PatchContact::default(), PatchContact::default())
            };
            return (
                self.well_pair_between(p, q),
                self.chem_pair_between(p, q),
                pa,
                pb,
            );
        }
        let max_chem_cut = if self.chem.has_solvation() { 6.6 } else { 5.25 };
        let well_reach = 2. * self.tree.bound + self.well.width;
        let chem_reach = 2. * self.tree.bound + max_chem_cut;
        let max_reach = well_reach.max(chem_reach);
        let dpq = sub(q.position, p.position);
        let d2_pq = dpq[0] * dpq[0] + dpq[1] * dpq[1] + dpq[2] * dpq[2];
        if d2_pq > max_reach * max_reach {
            return (0, 0., PatchContact::default(), PatchContact::default());
        }
        if d2_pq > chem_reach * chem_reach && !need_patch {
            return (
                self.well_pair_between(p, q),
                0.,
                PatchContact::default(),
                PatchContact::default(),
            );
        }
        let n = self.centers.len();
        let rp = rotation(p.orientation);
        let rqt = transpose(rotation(q.orientation));
        let r_rel = matmul(rqt, rp);
        let t_rel = matvec(rqt, sub(p.position, q.position));
        let rpt = transpose(rp);
        let q_in_p = matvec(rpt, dpq);
        let has_solv = self.chem.has_solvation();
        let extra = self.well.width.max(max_chem_cut);
        let mut active_a = Vec::new();
        self.tree
            .active_atoms_in_frame(&r_rel, &t_rel, &q_in_p, extra, &mut active_a);
        if active_a.is_empty() {
            return (0, 0., PatchContact::default(), PatchContact::default());
        }

        let mut well_count = 0usize;
        let mut any_chem = false;
        let mut any_patch = false;
        let mut p2 = 0.0f64;
        let mut p3 = 0.0f64;
        let mut max_u_a: Vec<f64> = Vec::new();
        let mut max_u_b: Vec<f64> = Vec::new();
        let mut u6_a: Vec<f64> = Vec::new();
        let mut u6_b: Vec<f64> = Vec::new();

        let mut min_d_a: Vec<f64> = Vec::new();
        let mut min_comp_a: Vec<f64> = Vec::new();
        let mut min_d_b: Vec<f64> = Vec::new();
        let mut min_comp_b: Vec<f64> = Vec::new();

        let mut s_hb = 0.0f64;
        let mut s_chb = 0.0f64;
        let mut s_wb_raw = 0.0f64;
        let mut chb_mids: Vec<Vec3> = Vec::new();
        let mut tn_pairs: Vec<(usize, usize, Vec3, f64)> = Vec::new();
        let mut clash_candidates: Vec<(usize, usize, f64)> = Vec::new();

        let r_chem_bound2 = (self.tree.bound + max_chem_cut) * (self.tree.bound + max_chem_cut);
        for (i, ya) in active_a {
            let x = self.centers[i];
            let ri = self.vdw[i];
            let r_pre = self.tree.bound + (ri + self.well.width).max(max_chem_cut);
            let dx = sub(x, q_in_p);
            let dx2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if dx2 > r_pre * r_pre {
                continue;
            }
            let fi = self.chem.flags[i];
            let in_chem_a = dx2 <= r_chem_bound2;
            for j in self.template_grid.near(&ya) {
                let xb = self.centers[j];
                let r_sum = ri + self.vdw[j];
                let r_well = r_sum + self.well.width;
                let r_chem = if has_solv {
                    (r_sum + 2.8).max(5.25)
                } else {
                    5.25
                };
                let r_outer = r_well.max(r_chem);
                let diff = sub(ya, xb);
                let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                if d2 >= r_outer * r_outer {
                    continue;
                }
                let mut d_cached = -1.0f64;
                if d2 < r_well * r_well {
                    well_count += 1;
                    if need_patch {
                        let r_28 = r_sum + 2.8;
                        if d2 < r_28 * r_28 {
                            if !any_patch {
                                u6_a.resize(n, 0.0);
                                u6_b.resize(n, 0.0);
                                any_patch = true;
                            }
                            let d = d2.sqrt();
                            d_cached = d;
                            let g = (d - r_sum).max(0.0);
                            let u6 = ((2.8 - g) * (1.0 / 2.8)).clamp(0.0, 1.0);
                            if u6 > u6_a[i] {
                                u6_a[i] = u6;
                            }
                            if u6 > u6_b[j] {
                                u6_b[j] = u6;
                            }
                        }
                    }
                }
                if !in_chem_a || d2 >= r_chem * r_chem {
                    continue;
                }
                if !any_chem {
                    if has_solv {
                        max_u_a.resize(n, 0.0);
                        max_u_b.resize(n, 0.0);
                    }
                    min_d_a.resize(n, 99.0);
                    min_comp_a.resize(n, 99.0);
                    min_d_b.resize(n, 99.0);
                    min_comp_b.resize(n, 99.0);
                    any_chem = true;
                }
                let d = if d_cached >= 0.0 { d_cached } else { d2.sqrt() };
                if has_solv {
                    let g = d - r_sum;
                    if g < 2.8 {
                        let u = ((2.8 - g) * (1.0 / 2.8)).clamp(0.0, 1.0);
                        let u2 = u * u;
                        p2 += u2;
                        p3 += u2 * u;
                        if u > max_u_a[i] {
                            max_u_a[i] = u;
                        }
                        if u > max_u_b[j] {
                            max_u_b[j] = u;
                        }
                    }
                }
                if d >= 5.25 {
                    continue;
                }
                let fj = self.chem.flags[j];
                if fi != 0 && d < min_d_a[i] {
                    min_d_a[i] = d;
                }
                if fj != 0 && d < min_d_b[j] {
                    min_d_b[j] = d;
                }
                if fi == 0 || fj == 0 {
                    continue;
                }
                let comp = ChemistryModel::complementary(fi, fj);
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
                    let gh = (-u * u).exp();
                    if ((fi & CHEM_NDON != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_NDON != 0))
                    {
                        s_hb += gh;
                    }
                    if ((fi & CHEM_CAT != 0) && (fj & CHEM_NACC != 0))
                        || ((fi & CHEM_NACC != 0) && (fj & CHEM_CAT != 0))
                    {
                        s_chb += gh;
                        if d < 3.6 {
                            chb_mids.push(scale(add(ya, xb), 0.5));
                        }
                    }
                }
                if (3.4..=4.5).contains(&d)
                    && (((fi & CHEM_TYR_OH != 0) && (fj & CHEM_AMIDE_ND != 0))
                        || ((fi & CHEM_AMIDE_ND != 0) && (fj & CHEM_TYR_OH != 0)))
                {
                    tn_pairs.push((i, j, scale(add(ya, xb), 0.5), d));
                }
                if d < 4.0
                    && (((fi & CHEM_PURE_ACC != 0) && (fj & CHEM_PURE_ACC != 0))
                        || ((fi & CHEM_PURE_DON != 0) && (fj & CHEM_PURE_DON != 0)))
                {
                    clash_candidates.push((i, j, d));
                }
            }
        }

        let (pa, pb) = if any_patch {
            (
                self.build_patch_contact(&u6_a),
                self.build_patch_contact(&u6_b),
            )
        } else {
            (PatchContact::default(), PatchContact::default())
        };

        if !any_chem {
            return (well_count, 0.0, pa, pb);
        }

        let (e_solv, a1_burial) = if has_solv {
            let mut a1 = 0.0f64;
            let mut a2 = 0.0f64;
            let mut a05 = 0.0f64;
            for &u in max_u_a.iter().chain(max_u_b.iter()) {
                if u > 0.0 {
                    a1 += u;
                    a2 += u * u;
                    a05 += u.sqrt();
                }
            }
            let sw = &self.chem.solv_weights;
            (
                sw[0] * p2 + sw[1] * p3 + sw[2] * a1 + sw[3] * a2 + sw[4] * a05,
                a1,
            )
        } else {
            (0.0, 999.0)
        };

        let mut s_cl = 0.0f64;
        if !tn_pairs.is_empty() && !chb_mids.is_empty() {
            let cats_a: Vec<Vec3> = self
                .centers
                .iter()
                .enumerate()
                .filter(|&(i, _)| self.chem.flags[i] & CHEM_CAT != 0)
                .map(|(_, &x)| add(matvec(r_rel, x), t_rel))
                .collect();
            let cats_b: Vec<Vec3> = self
                .centers
                .iter()
                .enumerate()
                .filter(|&(j, _)| self.chem.flags[j] & CHEM_CAT != 0)
                .map(|(_, &x)| x)
                .collect();
            let mut cl_cands: Vec<(usize, usize, f64)> = Vec::new();
            for (i, j, mid, d) in tn_pairs {
                let da = cats_a.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                let db = cats_b.iter().map(|&c| norm(sub(c, mid))).fold(99.0f64, f64::min);
                if da < 8.0 && db < 8.0 {
                    let d_chb = chb_mids.iter().map(|&m| norm(sub(m, mid))).fold(99.0f64, f64::min);
                    if d_chb < 8.5 {
                        let u1 = (d - 4.05) / 0.40;
                        let u2 = (d_chb.max(5.0) - 5.0) / 3.0;
                        cl_cands.push((i, j, (-u1 * u1 - u2 * u2).exp()));
                    }
                }
            }
            cl_cands.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut used_a = Vec::new();
            let mut used_b = Vec::new();
            for (i, j, val) in cl_cands {
                if !used_a.contains(&i) && !used_b.contains(&j) {
                    used_a.push(i);
                    used_b.push(j);
                    s_cl += val;
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
        let s_wb = s_wb_raw.min(self.chem.wb_cap) * (-1.5 * s_clash).exp();

        let mut unp_pol = 0.0f64;
        let mut unp_chg = 0.0f64;
        for (i, (&dmin, &dcomp)) in min_d_a.iter().zip(&min_comp_a).enumerate() {
            if dmin >= 4.25 {
                continue;
            }
            let fi = self.chem.flags[i];
            let is_chg = (fi & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fi & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 4.15 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.75;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.25 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.95;
                unp_chg += (-u * u).exp();
            }
        }
        for (j, (&dmin, &dcomp)) in min_d_b.iter().zip(&min_comp_b).enumerate() {
            if dmin >= 4.25 {
                continue;
            }
            let fj = self.chem.flags[j];
            let is_chg = (fj & (CHEM_CAT | CHEM_ANI)) != 0;
            let is_npol = !is_chg && ((fj & (CHEM_NDON | CHEM_NACC)) != 0);
            if is_npol && dmin < 4.15 && dcomp >= 3.65 {
                let u = (dmin.max(2.9) - 2.9) / 0.75;
                unp_pol += (-u * u).exp();
            }
            if is_chg && dmin < 4.25 && dcomp >= 4.30 {
                let u = (dmin.max(3.0) - 3.0) / 0.95;
                unp_chg += (-u * u).exp();
            }
        }

        let e_chem = self.chem.combine_pair_energy(
            e_solv, a1_burial, s_hb, s_chb, s_cl, s_wb, s_clash, unp_pol, unp_chg,
        );
        (well_count, e_chem, pa, pb)
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
        if let Some(grid) = &self.chem_grid {
            self.chem.pair_energy_with_radii_and_grid(
                &self.centers,
                &self.vdw,
                self.tree.bound,
                grid,
                p,
                q,
            )
        } else {
            self.chem.pair_energy(&self.centers, self.tree.bound, p, q)
        }
    }

    /// Total pairwise surrogate energy `E_well + E_like + dG_chem` between pose `p` and pose `q`.
    pub fn pair_surrogate_energy(&self, p: Pose, q: Pose) -> f64 {
        let (wp, ch) = self.well_and_chem_pair_between(p, q);
        let lk = self.like_pair_between(p, q);
        -self.well.epsilon * (wp as f64) + self.like.weight * lk + ch
    }

    /// Whether pose `p` and pose `q` are in contact according to `self.contact_gap`.
    fn in_contact_poses(&self, p: Pose, q: Pose) -> bool {
        let Some(gap) = self.contact_gap else {
            return false;
        };
        self.tree.within_gap(p, q, gap)
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
        comps: &[(Pose, f64)],
        w_sum: f64,
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
            let near: Vec<Pose> = self
                .body_grid
                .near_within(&x.position, cut)
                .filter(|&a| !in_cluster[a] && norm(sub(self.poses[a].position, x.position)) < cut)
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
        if comps.is_empty() || w_sum <= 0. {
            return uniform;
        }
        let (st, sr) = (d.sigma_translation, d.sigma_rotation);
        let norm_t = -1.5 * (2. * std::f64::consts::PI * st * st).ln();
        let norm_r = (8. * std::f64::consts::PI * std::f64::consts::PI).ln()
            - 1.5 * (2. * std::f64::consts::PI * sr * sr).ln();
        let mut terms = Vec::new();
        let lw_base = (1. - u).ln() - ((spectators.len() as f64) * w_sum).ln();
        let max_anchor_dist = 2. * self.tree.bound + self.contact_gap.unwrap_or(6.) + 10. * st;
        let max_d2 = max_anchor_dist * max_anchor_dist;
        let max_t2 = 100. * st * st;
        let inv_2st2 = 1. / (2. * st * st);
        let inv_2sr2 = 1. / (2. * sr * sr);
        for a in self.body_grid.near_within(&x.position, max_anchor_dist) {
            if in_cluster[a] {
                continue;
            }
            let pa = self.poses[a];
            let dx = sub(x.position, pa.position);
            if dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] > max_d2 {
                continue;
            }
            let rat = transpose(rotation(pa.orientation));
            let trel = matvec(rat, dx);
            let qrel = quat_mul(quat_conj(pa.orientation), x.orientation);
            for &(c, w_c) in comps {
                let dt = sub(trel, c.position);
                let t2 = dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2];
                if t2 > max_t2 {
                    continue;
                }
                let cos_half = quat_abs_dot(c.orientation, qrel);
                let theta = 2. * cos_half.acos();
                let jac = if theta < 1e-8 {
                    1.
                } else {
                    theta * theta / (2. * (1. - theta.cos()))
                };
                terms.push(
                    lw_base + w_c.ln() + norm_t - t2 * inv_2st2 + norm_r
                        - theta * theta * inv_2sr2
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
        let max_chem_cut = if self.chem.has_solvation() { 6.6 } else { 5.25 };
        let max_reach = (2. * self.tree.bound
            + self
                .well
                .width
                .max(self.like.cutoff)
                .max(max_chem_cut)
                .max(6.0))
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
            let comps: Vec<(Pose, f64)> = if d.learned.is_some() {
                Vec::new()
            } else {
                self.weighted_dock_components_excluding(&in_cluster, d)
            };
            let w_sum: f64 = comps.iter().map(|&(_, w)| w).sum();
            let u = if d.learned.is_none() && (comps.is_empty() || w_sum <= 0.) {
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
                let c = if d.library_beta == 0. {
                    comps[rng.random_range(0..comps.len())].0
                } else {
                    let mut v = rng.random::<f64>() * w_sum;
                    let mut c = comps[comps.len() - 1].0;
                    for &(pose_c, w_c) in &comps {
                        if v < w_c {
                            c = pose_c;
                            break;
                        }
                        v -= w_c;
                    }
                    c
                };
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
            log_q_ratio = self.dock_log_density_subset(&in_cluster, old_b0, d, &comps, w_sum, u)
                - self.dock_log_density_subset(&in_cluster, new_b0, d, &comps, w_sum, u);
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

        let has_coop = self.chem.has_cooperativity();
        let mut new_patch = if has_coop {
            self.patch.clone()
        } else {
            Vec::new()
        };
        let mut cache_updates: Vec<(usize, usize, PairCache)> = Vec::new();

        for &(b, p_new) in &moves {
            let p_old = self.poses[b];
            for &d in &spectators {
                let q = self.poses[d];
                if norm(sub(p_old.position, q.position)) > max_reach
                    && norm(sub(p_new.position, q.position)) > max_reach
                {
                    if has_coop {
                        new_patch[b][d] = PatchContact::default();
                        new_patch[d][b] = PatchContact::default();
                    }
                    cache_updates.push((b, d, PairCache::default()));
                    continue;
                }
                let pc_old = self.pair_cache[b][d];
                let wp_old = pc_old.well as i64;
                let ch_old = pc_old.chem;
                let lk_old = pc_old.like;
                let (wp_new_u, ch_new, pb, pd) = self.well_chem_and_patch_between(p_new, q);
                if has_coop {
                    new_patch[b][d] = pb;
                    new_patch[d][b] = pd;
                }
                let wp_new = wp_new_u as i64;
                let lk_new = self.like_pair_between(p_new, q);
                cache_updates.push((
                    b,
                    d,
                    PairCache {
                        well: wp_new_u as u32,
                        chem: ch_new,
                        like: lk_new,
                    },
                ));
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
                    if has_coop {
                        new_patch[b1][b2] = PatchContact::default();
                        new_patch[b2][b1] = PatchContact::default();
                    }
                    cache_updates.push((b1, b2, PairCache::default()));
                    continue;
                }
                let (wp12_new, ch12_new, p_b1, p_b2) =
                    self.well_chem_and_patch_between(p1_new, p2_new);
                let pc_old = self.pair_cache[b1][b2];
                let wp12_old = pc_old.well as usize;
                let ch12_old = pc_old.chem;
                let lk12_old = pc_old.like;
                let lk12_new = self.like_pair_between(p1_new, p2_new);
                cache_updates.push((
                    b1,
                    b2,
                    PairCache {
                        well: wp12_new as u32,
                        chem: ch12_new,
                        like: lk12_new,
                    },
                ));
                if has_coop {
                    new_patch[b1][b2] = p_b1;
                    new_patch[b2][b1] = p_b2;
                }
                dp += wp12_new as i64 - wp12_old as i64;
                dl += lk12_new - lk12_old;
                dc += ch12_new - ch12_old;
            }
        }
        if has_coop {
            let mut trial_poses = self.poses.clone();
            for &(b, p_new) in &moves {
                trial_poses[b] = p_new;
            }
            dc += self.coop_total_from_matrix(&new_patch, &trial_poses)
                - self.coop_total_from_matrix(&self.patch, &self.poses);
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
            for (u, v, pc) in cache_updates {
                self.pair_cache[u][v] = pc;
                self.pair_cache[v][u] = pc;
            }
            if has_coop {
                self.patch = new_patch;
            }
            for (b, p_new) in moves {
                let old_pos = self.poses[b].position;
                self.body_grid.relocate(b, &old_pos, &p_new.position);
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
        let max_chem_cut = if self.chem.has_solvation() { 6.6 } else { 5.25 };
        let max_reach = (2. * self.tree.bound
            + self
                .well
                .width
                .max(self.like.cutoff)
                .max(max_chem_cut)
                .max(6.0))
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
                    let pc_old = self.pair_cache[u][v];
                    let e_old = -self.well.epsilon * (pc_old.well as f64)
                        + self.like.weight * pc_old.like
                        + pc_old.chem;
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

        let has_coop = self.chem.has_cooperativity();
        let mut new_patch = if has_coop {
            self.patch.clone()
        } else {
            Vec::new()
        };
        let mut cache_updates: Vec<(usize, usize, PairCache)> = Vec::new();

        for &(b, p_new) in &moves {
            let p_old = self.poses[b];
            for &d in &spectators {
                let q_old = self.poses[d];
                if norm(sub(p_old.position, q_old.position)) > max_reach
                    && norm(sub(p_new.position, q_old.position)) > max_reach
                {
                    if has_coop {
                        new_patch[b][d] = PatchContact::default();
                        new_patch[d][b] = PatchContact::default();
                    }
                    cache_updates.push((b, d, PairCache::default()));
                    continue;
                }
                let pc_old = self.pair_cache[b][d];
                let wp_old = pc_old.well as i64;
                let ch_old = pc_old.chem;
                let lk_old = pc_old.like;
                let (wp_new_u, ch_new, pb, pd) = self.well_chem_and_patch_between(p_new, q_old);
                if has_coop {
                    new_patch[b][d] = pb;
                    new_patch[d][b] = pd;
                }
                let wp_new = wp_new_u as i64;
                let lk_new = self.like_pair_between(p_new, q_old);
                cache_updates.push((
                    b,
                    d,
                    PairCache {
                        well: wp_new_u as u32,
                        chem: ch_new,
                        like: lk_new,
                    },
                ));
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
                    if has_coop {
                        new_patch[b1][b2] = PatchContact::default();
                        new_patch[b2][b1] = PatchContact::default();
                    }
                    cache_updates.push((b1, b2, PairCache::default()));
                    continue;
                }
                let (wp12_new, ch12_new, p_b1, p_b2) =
                    self.well_chem_and_patch_between(p1_new, p2_new);
                let pc_old = self.pair_cache[b1][b2];
                let wp12_old = pc_old.well as usize;
                let ch12_old = pc_old.chem;
                let lk12_old = pc_old.like;
                let lk12_new = self.like_pair_between(p1_new, p2_new);
                cache_updates.push((
                    b1,
                    b2,
                    PairCache {
                        well: wp12_new as u32,
                        chem: ch12_new,
                        like: lk12_new,
                    },
                ));
                if has_coop {
                    new_patch[b1][b2] = p_b1;
                    new_patch[b2][b1] = p_b2;
                }
                dp += wp12_new as i64 - wp12_old as i64;
                dl += lk12_new - lk12_old;
                dc += ch12_new - ch12_old;
            }
        }
        if has_coop {
            let mut trial_poses = self.poses.clone();
            for &(b, p_new) in &moves {
                trial_poses[b] = p_new;
            }
            dc += self.coop_total_from_matrix(&new_patch, &trial_poses)
                - self.coop_total_from_matrix(&self.patch, &self.poses);
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
            for (u, v, pc) in cache_updates {
                self.pair_cache[u][v] = pc;
                self.pair_cache[v][u] = pc;
            }
            if has_coop {
                self.patch = new_patch;
            }
            for (b, p_new) in moves {
                let old_pos = self.poses[b].position;
                self.body_grid.relocate(b, &old_pos, &p_new.position);
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
        self.rebuild_body_grid();
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
        let coop = if self.chem.has_cooperativity() {
            let n = self.poses.len();
            let mut mat = vec![vec![PatchContact::default(); n]; n];
            for a in 0..n {
                for b in a + 1..n {
                    let (pa, pb) = self.patch_pair_between(self.poses[a], self.poses[b]);
                    mat[a][b] = pa;
                    mat[b][a] = pb;
                }
            }
            self.coop_total_from_matrix(&mat, &self.poses)
        } else {
            0.0
        };
        sum / 2. + coop
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
            memory_only: false,
            library_beta: 0.,
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
            memory_only: false,
            library_beta: 0.,
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
            memory_only: false,
            library_beta: 0.,
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
            memory_only: false,
            library_beta: 0.,
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
            memory_only: false,
            library_beta: 0.,
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
        let well = SquareWell { epsilon: 0.3, width: 6.0 };
        let chem = ChemistryModel {
            flags: vec![
                CHEM_CAT | CHEM_PURE_DON,
                CHEM_NACC | CHEM_TYR_OH,
                CHEM_NDON | CHEM_AMIDE_ND | CHEM_PURE_DON,
                CHEM_ANI | CHEM_PURE_ACC,
            ],
            weights: [1.45, 2.0, 12.5, 0.45, 1.8, 1.15, 2.2],
            solv_weights: [0.4645, -0.7707, -0.2771, -0.7201, 0.5226],
            solv_cap: 11.5,
            burial_gate: [4.0, 8.0],
            coop_weights: [1.5, 10.0],
            coop_s0: 15.0,
            junc_weights: [4.2, 0.7],
            junc_w0: 4.0,
            wb_cap: 5.0,
        };
        let start = vec![
            Pose { position: [0., 0., 0.], orientation: [1., 0., 0., 0.] },
            Pose {
                position: [3.2, 1.8, -0.1],
                orientation: quaternion(cayley([0.1, -0.2, 0.15])),
            },
            Pose {
                position: [-3.2, 1.8, 0.1],
                orientation: quaternion(cayley([-0.25, 0.3, -0.1])),
            },
        ];
        let centers: Vec<Vec3> = tree.shape.atoms.iter().map(|a| a.center).collect();
        let vdw: Vec<f64> = tree.shape.atoms.iter().map(|a| a.radius).collect();
        let cgrid = BallGrid::new(&centers, 5.25);
        // Verify exact symmetry and SE(3) invariance on initial pairs.
        for i in 0..start.len() {
            for j in i + 1..start.len() {
                let e_ij = chem.pair_energy_with_radii_and_grid(&centers, &vdw, tree.bound, &cgrid, start[i], start[j]);
                let e_ji = chem.pair_energy_with_radii_and_grid(&centers, &vdw, tree.bound, &cgrid, start[j], start[i]);
                assert!(
                    (e_ij - e_ji).abs() < 1e-12,
                    "asymmetric pair chemistry+solvation energy: {e_ij} vs {e_ji}"
                );
            }
        }
        let mut mc = MorphMc::new(tree, wall, model, well, start.clone(), vec![true, false, false])
            .unwrap()
            .with_chemistry_model(chem)
            .unwrap();
        for i in 0..start.len() {
            for j in i + 1..start.len() {
                let (w_f, c_f) = mc.well_and_chem_pair_between(start[i], start[j]);
                let w_s = mc.well_pair_between(start[i], start[j]);
                let c_s = mc.chem_pair_between(start[i], start[j]);
                assert_eq!(w_f, w_s, "fused well count mismatch");
                assert!((c_f - c_s).abs() < 1e-12, "fused chem+solv mismatch: {c_f} vs {c_s}");
            }
        }
        let mv = MoveSettings {
            translation: 0.4,
            rotation: 0.25,
            reinsertion_probability: 0.1,
        };
        let dock = DockSettings {
            learned: None,
            library: false,
            memory_only: false,
            library_beta: 0.,
            components: vec![Pose {
                position: [4.6, 0., 0.],
                orientation: [1., 0., 0., 0.],
            }],
            sigma_translation: 0.4,
            sigma_rotation: 0.2,
            uniform_weight: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let cs = ClusterSettings {
            max_size: 3,
            fk_beta: 0.5,
            dock_probability: 0.1,
            reinsertion_probability: 0.1,
            translation: 0.4,
            rotation: 0.25,
            size_scaling: true,
        };
        for t in 0..2_000 {
            if t % 5 == 0 {
                mc.step_cluster(&mut rng, &cs, Some(&dock)).unwrap();
            } else if t % 3 == 0 {
                mc.step_dock(&mut rng, &dock).unwrap();
            } else {
                mc.step(&mut rng, mv).unwrap();
            }
            if t % 200 == 0 {
                let d = (mc.recompute_chem() - mc.chem_energy()).abs();
                assert!(
                    d < 1e-10,
                    "step {t}: cached chemistry energy {} differs from recomputed {} by {}",
                    mc.chem_energy(),
                    mc.recompute_chem(),
                    d
                );
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
            solv_weights: [0.4645, -0.7707, -0.2771, -0.7201, 0.5226],
            solv_cap: 11.5,
            burial_gate: [4.0, 8.0],
            coop_weights: [1.5, 10.0],
            coop_s0: 15.0,
            junc_weights: [4.2, 0.7],
            junc_w0: 4.0,
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
        let mut rng_prior = StdRng::seed_from_u64(999);
        let surf_prior = mc.generate_surface_normal_prior(&mut rng_prior, 8, 0.5);
        assert!(
            !surf_prior.is_empty(),
            "surface-normal-opposed prior should discover valid non-overlapping contacts"
        );
        let dock_comps: Vec<Pose> = surf_prior.into_iter().map(|(p, _)| p).collect();
        let dock = DockSettings {
            learned: None,
            library: true,
            memory_only: false,
            library_beta: 0.8,
            components: dock_comps,
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

    /// Three balls (one fixed, two mobile) with Boltzmann-weighted library docking
    /// (`library_beta > 0`): weighting spectator contact pairs by `exp(-beta * E_pair(i, j))`
    /// is invariant to the moved body and must preserve exact detailed balance.
    #[test]
    fn weighted_library_docking_preserves_detailed_balance() {
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
        let mut rng_p = StdRng::seed_from_u64(42);
        let prior_mc = make();
        let surf_prior = prior_mc.generate_surface_normal_prior(&mut rng_p, 4, 0.5);
        assert!(!surf_prior.is_empty());
        let wsum_surf: f64 = surf_prior.iter().map(|&(_, w)| w).sum();
        assert!(
            (wsum_surf - 1.0).abs() < 1e-12,
            "surface prior weights must be normalized to 1.0: {wsum_surf}"
        );
        let dock = DockSettings {
            learned: None,
            library: true,
            memory_only: false,
            library_beta: 0.8,
            components: surf_prior.into_iter().map(|(p, _)| p).collect(),
            sigma_translation: 0.3,
            sigma_rotation: 0.3,
            uniform_weight: 0.3,
        };
        let (bins, lo, hi) = (12usize, 2.0, 8.0);
        let (burn, n) = (10_000, 800_000);
        let mut out = Vec::new();
        for (seed, with_dock) in [(31u64, false), (32, true)] {
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
                    let p = mc.poses();
                    for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                        let d = norm(sub(p[i].position, p[j].position));
                        let bin = (((d - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1);
                        h[k * bins + bin] += 1;
                    }
                    contacts += mc.contact_count() as u64;
                }
            }
            if with_dock {
                assert!(mc.stats.dock_accepted > 15_000);
            }
            out.push((h, contacts as f64 / n as f64));
        }
        let (ha, ca) = &out[0];
        let (hb, cb) = &out[1];
        for k in 0..3 * bins {
            let (pa, pb) = (ha[k] as f64 / n as f64, hb[k] as f64 / n as f64);
            assert!(
                (pa - pb).abs() < 8e-3,
                "weighted dock bin {k}: local {pa:.4} vs weighted library {pb:.4}"
            );
        }
        assert!((ca - cb).abs() < 0.025, "contact fraction {ca:.4} vs {cb:.4}");
    }

    #[test]
    fn chemical_complementarity_surface_prior_discovers_monomer_native_contacts() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shape_path = root.join("examples/monomer-shape.json");
        let chem_path = root.join("examples/chemistry-model-monomer.json");
        if !shape_path.exists() || !chem_path.exists() {
            return;
        }
        let shape: Shape =
            serde_json::from_str(&std::fs::read_to_string(&shape_path).unwrap()).unwrap();
        let chem: ChemistryModel =
            serde_json::from_str(&std::fs::read_to_string(&chem_path).unwrap()).unwrap();
        let n = shape.atoms.len();
        let tree = SphereTree::new(shape).unwrap();
        let wall = Container::new(150.0, &tree).unwrap();
        let model = SolvationModel {
            probe: 1.4,
            pressure: 0.0,
            coefficients: vec![[0.0; 3]; n],
            atom_types: (0..n).collect(),
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let well = SquareWell {
            epsilon: 0.0,
            width: 0.6,
        };
        let identity = Pose {
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
        };
        let mc = MorphMc::new(tree, wall, model, well, vec![identity], vec![true])
            .unwrap()
            .with_chemistry_model(chem)
            .unwrap()
            .with_contact_library(2.8)
            .unwrap();
        let t0 = std::time::Instant::now();
        let mut rng = StdRng::seed_from_u64(42);
        let prior = mc.generate_surface_normal_prior(&mut rng, 64, 0.5);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("generate_surface_normal_prior(64) took {elapsed_ms:.1} ms, len={}", prior.len());
        assert_eq!(prior.len(), 128);
        let placed_id = Placed::new(identity);
        for &(p, w) in &prior {
            assert!(!mc.tree.overlaps(&placed_id, &Placed::new(p)));
            assert!(w > 0.0);
        }
        let nat_modes = [
            (
                "m1",
                Pose {
                    position: [19.47624995357571, 18.42510338365495, -9.475000000000003],
                    orientation: [0.7071067811865477, 0.0, 0.0, 0.7071067811865475],
                },
            ),
            (
                "m2",
                Pose {
                    position: [19.47624995357571, 18.42510338365495, 28.424999999999994],
                    orientation: [0.7071067811865477, 0.0, 0.0, 0.7071067811865475],
                },
            ),
            (
                "m4",
                Pose {
                    position: [-20.073750046424273, -20.07375004642428, -19.451561877005823],
                    orientation: [0.0, 0.7071067811865477, -0.7071067811865475, 0.0],
                },
            ),
            (
                "m6",
                Pose {
                    position: [21.124896616345055, -21.124896616345048, -0.5015618770058232],
                    orientation: [0.0, 0.7071067811865475, 0.7071067811865477, 0.0],
                },
            ),
        ];
        let mem_s = MemorySettings {
            lambda: 0.25,
            lambda_prior: 3.0,
            lambda_floor: 1e-4,
            death: 0.05,
            sigma_translation: 0.5,
            sigma_rotation: 5f64.to_radians(),
            floor_sigma: 20.0,
        };
        for (name, nat) in nat_modes {
            let r_nat = rotation(nat.orientation);
            let mut best_sc = 999.0f64;
            let mut best_dt = 999.0f64;
            let mut best_dth = 999.0f64;
            let mut best_e = 0.0f64;
            let mut best_w = 0.0f64;
            for &(p, w) in &prior {
                let dt = norm(sub(p.position, nat.position));
                let r_p = rotation(p.orientation);
                let r_diff = matmul(transpose(r_p), r_nat);
                let tr = r_diff[0][0] + r_diff[1][1] + r_diff[2][2];
                let dth = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees();
                let sc = ((dt / 2.0).powi(2) + (dth / 20.0).powi(2)).sqrt();
                if sc < best_sc {
                    best_sc = sc;
                    best_dt = dt;
                    best_dth = dth;
                    best_e = mc.pair_surrogate_energy(identity, p);
                    best_w = w;
                }
            }
            println!(
                "  {name}: best_sc={best_sc:.2} (dt={best_dt:.2} A, dth={best_dth:.1} deg, E={best_e:+.2} kT, w={best_w:.5}), E_nat={:+.2} kT",
                mc.pair_surrogate_energy(identity, nat)
            );
            if name == "m1" || name == "m6" {
                assert!(
                    best_sc < 1.8 && best_dt < 3.0 && best_dth < 20.0,
                    "expected {name} to be discovered in surface prior, got sc={best_sc:.2} (dt={best_dt:.2} A, dth={best_dth:.1} deg)"
                );
                assert!(
                    best_w > 0.002,
                    "expected {name} prior weight not to be Boltzmann-crushed, got w={best_w:.5}"
                );
                let mut sum_draw_e = 0.0f64;
                for _ in 0..64 {
                    let drawn = mem_s.draw(&mut rng, &[nat], &[], &mc.tree);
                    assert!(!mc.tree.overlaps(&placed_id, &Placed::new(drawn)));
                    sum_draw_e += mc.pair_surrogate_energy(identity, drawn);
                }
                let mean_draw_e = sum_draw_e / 64.0;
                println!("    {name} MemorySettings::draw mean_E = {mean_draw_e:+.2} kT");
                assert!(
                    mean_draw_e < -5.0,
                    "expected annealed MemorySettings::draw around {name} to stay in attractive basin, got mean_E={mean_draw_e:+.2}"
                );
            }
        }
    }
}
