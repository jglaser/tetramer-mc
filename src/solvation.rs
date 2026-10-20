//! Exact many-body morphometric solvation free energy of rigid bodies built from
//! atoms, with atom-type coefficients:
//!   beta F = p V + sum_atoms (sigma_t A_i + kappa_t M_i + kbar_t X_i)
//! where V, A_i, M_i, X_i are the exact union measures and per-atom parts of the
//! accessible union (atom radii + probe) of all bodies (`morphometry`). Units: A,
//! kT; sigma kT/A^2, kappa kT/A, kbar kT, p kT/A^3.
//!
//! `SolvationState` caches every atom's parts and returns the exact change of beta F
//! for a rigid move of any set of bodies. Only atoms with an overlap across bodies
//! before or after the move are recomputed: an atom's parts depend only on the balls
//! that overlap it. A moved atom without such an overlap keeps A, M and X; its
//! volume part (r^3 |E| + r^2 c . m)/3 changes only through its centre c and
//! the rotated moment m, so it is updated without geometry.
use crate::{
    math::{Mat3, Pose, Vec3, dot, matmul, matvec, norm, rotation, sub, transpose},
    morphometry::{BallGrid, BallMeasures, ball_measures},
};
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

/// Coefficients per atom type and the atom-type map of one body template.
#[derive(Clone, Debug)]
pub struct SolvationModel {
    pub probe: f64,
    pub pressure: f64,
    /// [sigma, kappa, kbar] per type.
    pub coefficients: Vec<[f64; 3]>,
    pub atom_types: Vec<usize>,
    /// Which local share of the Gaussian curvature each atom carries.
    pub gauss_convention: GaussConvention,
}

/// Local per-atom shares of the integrated Gaussian curvature. The union total is
/// 4 pi chi only for `PatchGaussBonnet` (faces, own arcs and corners); `Face` keeps
/// only the exposed patch |E_i|, and `FaceEdge` adds the own-arc geodesic term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaussConvention {
    #[default]
    PatchGaussBonnet,
    /// Akopyan and Edelsbrunner (2020), as implemented in AlphaMol (P. Koehl).
    AlphaMol,
    FaceEdge,
    Face,
}

/// Harano, Roth, Chiba, J. Comput. Chem. 34, 1969 (2013), Table 1: octanol -> water
/// transfer, kJ/mol with nm (kappa printed as nm^-2; their sphere formula needs nm^-1).
/// Order: CH, aCH, C=O (carbon), O=C (oxygen), OH, N, S.
pub const HARANO_TYPES: [&str; 7] = ["CH", "aCH", "C=O", "O=C", "OH", "N", "S"];
pub const HARANO_OCTANOL_WATER_KJ_NM: [[f64; 3]; 7] = [
    [6.629, 2.435, -1.301],
    [-1.44, -10.84, 1.885],
    [75.16, -24.44, 2.491],
    [-69.63, 11.00, -0.534],
    [-27.49, 7.438, -0.233],
    [-18.96, 10.19, -1.252],
    [49.62, -31.65, 6.718],
];

impl SolvationModel {
    /// Harano octanol -> water coefficients converted to kT and A at `temperature_k`.
    /// The volume term vanishes for a transfer (p = 0). Probe radius 1.4 A.
    pub fn harano_octanol_water(atom_types: Vec<usize>, temperature_k: f64) -> Result<Self> {
        ensure!(temperature_k > 0., "temperature must be positive");
        ensure!(
            atom_types.iter().all(|&t| t < 7),
            "Harano atom type out of range"
        );
        let rt = 8.314462618e-3 * temperature_k; // kJ/mol
        let coefficients = HARANO_OCTANOL_WATER_KJ_NM
            .iter()
            .map(|c| [c[0] * 0.01 / rt, c[1] * 0.1 / rt, c[2] / rt])
            .collect();
        Ok(Self {
            probe: 1.4,
            pressure: 0.,
            coefficients,
            atom_types,
            gauss_convention: GaussConvention::PatchGaussBonnet,
        })
    }
    fn ball_energy(&self, atom: usize, b: &BallMeasures) -> f64 {
        let c = self.coefficients[self.atom_types[atom]];
        let x = match self.gauss_convention {
            GaussConvention::PatchGaussBonnet => b.gauss,
            GaussConvention::AlphaMol => b.gauss_alphamol,
            GaussConvention::FaceEdge => b.gauss_face + b.gauss_edge,
            GaussConvention::Face => b.gauss_face,
        };
        c[0] * b.area + c[1] * b.mean + c[2] * x
    }
}

/// Cached per-atom state for a set of identical rigid bodies.
pub struct SolvationState {
    model: SolvationModel,
    template: Vec<Vec3>,
    radii: Vec<f64>,
    /// Same-body overlapping atoms per template atom (rigid, so fixed).
    intra: Vec<Vec<usize>>,
    poses: Vec<Pose>,
    rotations: Vec<Mat3>,
    world: Vec<Vec<Vec3>>,
    balls: Vec<Vec<BallMeasures>>,
    cell: f64,
    grid: BallGrid,
    energy: f64,
    /// Largest distance from a body origin to a dilated atom surface; bodies whose
    /// origins are farther apart than twice this cannot overlap.
    reach: f64,
}

/// A proposed move: new poses and recomputed atom parts; apply with `commit`.
pub struct SolvationUpdate {
    pub delta: f64,
    poses: Vec<(usize, Pose)>,
    balls: Vec<((usize, usize), BallMeasures)>,
}

impl SolvationState {
    pub fn new(
        model: SolvationModel,
        atom_centers: &[Vec3],
        atom_radii: &[f64],
        poses: &[Pose],
    ) -> Result<Self> {
        ensure!(
            atom_centers.len() == atom_radii.len() && model.atom_types.len() == atom_centers.len(),
            "template, radii and atom types differ in length"
        );
        ensure!(
            model.coefficients.len() > *model.atom_types.iter().max().unwrap_or(&0),
            "missing atom-type coefficients"
        );
        let radii: Vec<f64> = atom_radii.iter().map(|r| r + model.probe).collect();
        let n = radii.len();
        let intra = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| {
                        j != i && norm(sub(atom_centers[j], atom_centers[i])) < radii[i] + radii[j]
                    })
                    .collect()
            })
            .collect();
        let rmax = radii.iter().copied().fold(0., f64::max);
        let cell = 2. * rmax;
        let reach = atom_centers
            .iter()
            .zip(&radii)
            .map(|(c, r)| norm(*c) + r)
            .fold(0., f64::max);
        let mut s = Self {
            model,
            template: atom_centers.to_vec(),
            radii,
            intra,
            poses: poses.to_vec(),
            rotations: poses.iter().map(|p| rotation(p.orientation)).collect(),
            world: Vec::new(),
            balls: Vec::new(),
            cell,
            grid: BallGrid::new(&[], cell),
            energy: 0.,
            reach,
        };
        s.world = (0..poses.len()).map(|b| s.place(b, poses[b])).collect();
        s.rebuild_grid();
        let none = BTreeMap::new();
        let none_centers = BTreeMap::new();
        s.balls = (0..poses.len())
            .map(|b| (0..n).map(|i| s.measure(b, i, &none, &none_centers)).collect())
            .collect();
        s.energy = s.total();
        Ok(s)
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }
    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }

    fn place(&self, _b: usize, pose: Pose) -> Vec<Vec3> {
        self.template.iter().map(|&c| pose.apply(c)).collect()
    }

    fn flat(&self, b: usize, i: usize) -> usize {
        b * self.template.len() + i
    }

    fn rebuild_grid(&mut self) {
        let all: Vec<Vec3> = self.world.iter().flatten().copied().collect();
        self.grid = BallGrid::new(&all, self.cell);
    }

    fn total(&self) -> f64 {
        let mut f = 0.;
        let mut v = 0.;
        for row in &self.balls {
            for (i, b) in row.iter().enumerate() {
                f += self.model.ball_energy(i, b);
                v += b.volume;
            }
        }
        f + self.model.pressure * v
    }

    /// World position of atom (b, i), with moved bodies taken from `moved`.
    fn position(&self, b: usize, i: usize, moved: &BTreeMap<usize, Vec<Vec3>>) -> Vec3 {
        moved.get(&b).map_or(self.world[b][i], |w| w[i])
    }

    /// Atoms of other bodies overlapping atom (b, i) at position c in the configuration
    /// where bodies in `moved` sit at their new positions.
    fn inter_neighbours(
        &self,
        b: usize,
        i: usize,
        c: Vec3,
        moved: &BTreeMap<usize, Vec<Vec3>>,
        moved_centers: &BTreeMap<usize, Vec3>,
    ) -> Vec<(usize, usize)> {
        let n = self.template.len();
        let r = self.radii[i];
        let mut out: Vec<(usize, usize)> = self
            .grid
            .near(&c)
            .map(|k| (k / n, k % n))
            .filter(|&(bb, j)| {
                bb != b
                    && !moved.contains_key(&bb)
                    && norm(sub(self.world[bb][j], c)) < r + self.radii[j]
            })
            .collect();
        for (&bb, w) in moved {
            if bb == b || norm(sub(moved_centers[&bb], c)) >= self.reach + r {
                continue;
            }
            for (j, &cj) in w.iter().enumerate() {
                if norm(sub(cj, c)) < r + self.radii[j] {
                    out.push((bb, j));
                }
            }
        }
        out
    }

    fn measure(
        &self,
        b: usize,
        i: usize,
        moved: &BTreeMap<usize, Vec<Vec3>>,
        moved_centers: &BTreeMap<usize, Vec3>,
    ) -> BallMeasures {
        let c = self.position(b, i, moved);
        let mut nb: Vec<(Vec3, f64)> = self.intra[i]
            .iter()
            .map(|&j| (self.position(b, j, moved), self.radii[j]))
            .collect();
        for (bb, j) in self.inter_neighbours(b, i, c, moved, moved_centers) {
            nb.push((self.position(bb, j, moved), self.radii[j]));
        }
        ball_measures(c, self.radii[i], &nb)
    }

    /// Exact change of beta F for rigid moves of the listed bodies.
    pub fn propose(&self, moves: &[(usize, Pose)]) -> Result<SolvationUpdate> {
        let nb = self.poses.len();
        let mut moved: BTreeMap<usize, Vec<Vec3>> = BTreeMap::new();
        let mut moved_centers: BTreeMap<usize, Vec3> = BTreeMap::new();
        for &(b, p) in moves {
            ensure!(
                b < nb && !moved.contains_key(&b),
                "invalid or repeated moved body"
            );
            p.validate()?;
            moved.insert(b, self.place(b, p));
            moved_centers.insert(b, p.position);
        }
        let is_rigid = if let Some(&(b0, p0)) = moves.first() {
            let rel0 = matmul(rotation(p0.orientation), transpose(self.rotations[b0]));
            let shift0 = sub(p0.position, matvec(rel0, self.poses[b0].position));
            moves.iter().skip(1).all(|&(b, p)| {
                let rel = matmul(rotation(p.orientation), transpose(self.rotations[b]));
                let shift = sub(p.position, matvec(rel0, self.poses[b].position));
                norm(sub(shift, shift0)) < 1e-7
                    && (0..3).all(|r| (0..3).all(|c| (rel[r][c] - rel0[r][c]).abs() < 1e-7))
            })
        } else {
            true
        };
        let none = BTreeMap::new();
        let none_centers = BTreeMap::new();
        let n_atoms = self.template.len();
        // Atoms with a changed overlap across bodies before or after the move.
        // When `moves` is a single rigid SE(3) transformation of `moved`, intra-cluster
        // overlaps between bodies inside `moved` are strictly invariant, so only atoms
        // overlapping spectator bodies (`!moved.contains_key(&bb)`) before or after the
        // move can change their intrinsic BallMeasures.
        let mut affected: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (&b, w) in &moved {
            let (po, pn) = (self.poses[b].position, moved_centers[&b]);
            let near = (0..nb).any(|c| {
                if c == b || (is_rigid && moved.contains_key(&c)) {
                    return false;
                }
                let qo = self.poses[c].position;
                let qn = moved_centers.get(&c).copied().unwrap_or(qo);
                norm(sub(qo, po)) < 2. * self.reach || norm(sub(qn, pn)) < 2. * self.reach
            });
            if !near {
                continue;
            }
            if is_rigid {
                for i in 0..n_atoms {
                    let r = self.radii[i];
                    let co = self.world[b][i];
                    let cn = w[i];
                    let mut hit = false;
                    for (bb, j) in self
                        .grid
                        .near(&co)
                        .map(|k| (k / n_atoms, k % n_atoms))
                        .filter(|&(bb, j)| {
                            !moved.contains_key(&bb) && norm(sub(self.world[bb][j], co)) < r + self.radii[j]
                        })
                    {
                        hit = true;
                        affected.insert((bb, j));
                    }
                    for (bb, j) in self
                        .grid
                        .near(&cn)
                        .map(|k| (k / n_atoms, k % n_atoms))
                        .filter(|&(bb, j)| {
                            !moved.contains_key(&bb) && norm(sub(self.world[bb][j], cn)) < r + self.radii[j]
                        })
                    {
                        hit = true;
                        affected.insert((bb, j));
                    }
                    if hit {
                        affected.insert((b, i));
                    }
                }
            } else {
                for i in 0..n_atoms {
                    let old = self.inter_neighbours(b, i, self.world[b][i], &none, &none_centers);
                    let new = self.inter_neighbours(b, i, w[i], &moved, &moved_centers);
                    if !old.is_empty() || !new.is_empty() {
                        affected.insert((b, i));
                        affected.extend(old);
                        affected.extend(new);
                    }
                }
            }
        }
        let mut delta = 0.;
        let mut dv = 0.;
        let mut balls = Vec::new();
        for &(b, i) in &affected {
            let old = self.balls[b][i];
            let new = self.measure(b, i, &moved, &moved_centers);
            delta += self.model.ball_energy(i, &new) - self.model.ball_energy(i, &old);
            dv += new.volume - old.volume;
            balls.push(((b, i), new));
        }
        // Moved atoms without overlaps across bodies: rigid transport of the moment.
        for &(b, p) in moves {
            let rel = matmul(rotation(p.orientation), transpose(self.rotations[b]));
            for i in 0..self.template.len() {
                if affected.contains(&(b, i)) {
                    continue;
                }
                let old = self.balls[b][i];
                let r = self.radii[i];
                let mut new = old;
                new.moment = matvec(rel, old.moment);
                new.volume =
                    (r * r * r * old.area / (r * r) + r * r * dot(moved[&b][i], new.moment)) / 3.;
                dv += new.volume - old.volume;
                balls.push(((b, i), new));
            }
        }
        delta += self.model.pressure * dv;
        Ok(SolvationUpdate {
            delta,
            poses: moves.to_vec(),
            balls,
        })
    }

    pub fn commit(&mut self, update: SolvationUpdate) {
        let n = self.template.len();
        for &(b, p) in &update.poses {
            let new = self.place(b, p);
            for (i, (o, w)) in self.world[b].iter().zip(&new).enumerate() {
                self.grid.relocate(b * n + i, o, w);
            }
            self.poses[b] = p;
            self.rotations[b] = rotation(p.orientation);
            self.world[b] = new;
        }
        for ((b, i), m) in update.balls {
            self.balls[b][i] = m;
        }
        self.energy += update.delta;
    }

    /// Full recomputation from scratch (for checks).
    pub fn recompute(&self) -> f64 {
        let none = BTreeMap::new();
        let none_centers = BTreeMap::new();
        let mut f = 0.;
        let mut v = 0.;
        for b in 0..self.poses.len() {
            for i in 0..self.template.len() {
                let m = self.measure(b, i, &none, &none_centers);
                f += self.model.ball_energy(i, &m);
                v += m.volume;
            }
        }
        f + self.model.pressure * v
    }

    /// Flat atom index, for callers that keep per-atom data.
    pub fn atom_index(&self, b: usize, i: usize) -> usize {
        self.flat(b, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::uniform_pose;
    use crate::morphometry::union_measures;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn template(rng: &mut StdRng) -> (Vec<Vec3>, Vec<f64>, Vec<usize>) {
        let c: Vec<Vec3> = (0..14)
            .map(|_| std::array::from_fn(|_| (rng.random::<f64>() - 0.5) * 5.))
            .collect();
        let r: Vec<f64> = (0..14).map(|_| 0.5 + 0.4 * rng.random::<f64>()).collect();
        let t: Vec<usize> = (0..14).map(|k| k % 3).collect();
        (c, r, t)
    }

    #[test]
    fn incremental_delta_matches_full_recompute() {
        let mut rng = StdRng::seed_from_u64(7);
        let (c, r, t) = template(&mut rng);
        let model = SolvationModel {
            probe: 0.8,
            pressure: 0.21,
            coefficients: vec![[0.3, -0.2, 0.05], [-0.4, 0.3, -0.1], [0.1, 0.15, 0.2]],
            atom_types: t.clone(),
            gauss_convention: GaussConvention::PatchGaussBonnet,
        };
        let box_l = [13.; 3];
        let poses: Vec<Pose> = (0..6).map(|_| uniform_pose(&mut rng, box_l)).collect();
        let mut s = SolvationState::new(model.clone(), &c, &r, &poses).unwrap();
        // Independent check of the cached total: the per-type sum over a fresh union.
        let check = |s: &SolvationState| {
            let mut centers = Vec::new();
            let mut radii = Vec::new();
            let mut types = Vec::new();
            for p in s.poses() {
                for (i, &ci) in c.iter().enumerate() {
                    centers.push(p.apply(ci));
                    radii.push(r[i] + model.probe);
                    types.push(t[i]);
                }
            }
            let u = union_measures(&centers, &radii).unwrap();
            u.balls
                .iter()
                .zip(&types)
                .map(|(b, &ty)| {
                    let k = model.coefficients[ty];
                    k[0] * b.area + k[1] * b.mean + k[2] * b.gauss
                })
                .sum::<f64>()
                + model.pressure * u.volume
        };
        assert!((s.energy() - check(&s)).abs() < 1e-9);
        let mut nonzero = 0;
        for step in 0..200 {
            let nmove = 1 + step % 3;
            let mut moves = Vec::new();
            for k in 0..nmove {
                let b = (step + 2 * k) % 6;
                if moves.iter().any(|&(bb, _)| bb == b) {
                    continue;
                }
                let p = if step % 2 == 0 {
                    uniform_pose(&mut rng, box_l)
                } else {
                    let old = s.poses()[b];
                    let mut q = uniform_pose(&mut rng, [1.; 3]);
                    q.position =
                        std::array::from_fn(|d| old.position[d] + (q.position[d] - 0.5) * 2.);
                    q
                };
                moves.push((b, p));
            }
            let before = s.energy();
            let u = s.propose(&moves).unwrap();
            let d = u.delta;
            s.commit(u);
            let full = check(&s);
            assert!(
                (before + d - full).abs() < 1e-8 * (1. + full.abs()),
                "step {step}: {} vs {}",
                before + d,
                full
            );
            assert!((s.recompute() - full).abs() < 1e-8 * (1. + full.abs()));
            nonzero += usize::from(d.abs() > 1e-6);
        }
        assert!(nonzero > 50, "moves must change the energy: {nonzero}");
    }
}
