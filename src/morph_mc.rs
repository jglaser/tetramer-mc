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

#[derive(Clone, Copy, Debug)]
pub struct MoveSettings {
    pub translation: f64,
    pub rotation: f64,
    pub reinsertion_probability: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MoveStats {
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
    well_energy: f64,
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
            well_energy: 0.,
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
        let mut pairs = 0.;
        for a in 0..s.poses.len() {
            pairs += s.well_pairs(a, s.poses[a]) as f64;
        }
        s.well_energy = -s.well.epsilon * pairs / 2.;
        Ok(s)
    }

    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }
    pub fn solvation_energy(&self) -> f64 {
        self.solv.energy()
    }
    pub fn well_energy(&self) -> f64 {
        self.well_energy
    }

    /// Square-well atom pairs between body `b` placed at `p` and all other bodies.
    pub fn well_pairs(&self, b: usize, p: Pose) -> usize {
        if self.well.epsilon == 0. {
            return 0;
        }
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
        let d_well =
            -self.well.epsilon * (self.well_pairs(b, new) as f64 - self.well_pairs(b, old) as f64);
        let update = self.solv.propose(&[(b, new)])?;
        let de = update.delta + d_well;
        ensure!(de.is_finite(), "nonfinite energy change");
        if de <= 0. || rng.random::<f64>() < (-de).exp() {
            self.solv.commit(update);
            self.well_energy += d_well;
            self.poses[b] = new;
            self.placed[b] = Placed::new(new);
            self.stats.accepted += 1;
            self.stats.reinsertion_accepted += u64::from(reinsertion);
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
    }
}
