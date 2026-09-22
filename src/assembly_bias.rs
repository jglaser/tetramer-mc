//! A frozen bias on an instantaneous geometric observable, with no native labels.
//!
//! If P is reversible for pi, then P(x,dy) min(1,exp[B(x)-B(y)])
//! (plus rejection on x) is reversible for pi_B(dx) ∝ exp[-B(x)] pi(dx).
//! Apply this gate to each elementary reversible physical kernel, not a whole
//! ordered sweep. Physical observables reweight by exp[B]; store B in log form.
use crate::{
    geometry::{Placed, Shape, SphereTree},
    math::*,
};
use anyhow::{Result, ensure};
use rand::RngExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssemblyBiasConfig {
    /// Dimensionless B(1), ..., B(N). Entries are not probabilities.
    pub values: Vec<f64>,
}
impl AssemblyBiasConfig {
    pub fn validate(&self, bodies: usize) -> Result<()> {
        ensure!(
            bodies > 0 && self.values.len() == bodies,
            "assembly bias requires exactly N values for sizes 1..N"
        );
        ensure!(
            self.values.iter().all(|b| b.is_finite()),
            "nonfinite assembly bias"
        );
        let lo = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        ensure!(
            (hi - lo).is_finite(),
            "assembly bias differences must be representable"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssemblyBiasState {
    pub largest_component_size: usize,
    pub bias: f64,
    /// Physical expectations are sum(f exp(log_reweight))/sum(exp(log_reweight)).
    pub log_reweight: f64,
}

pub struct AssemblyBias {
    config: AssemblyBiasConfig,
    exclusion: SphereTree,
    lengths: Option<Vec3>,
}
impl AssemblyBias {
    pub fn new(
        config: AssemblyBiasConfig,
        shape: &Shape,
        rd: f64,
        bodies: usize,
        periodic_lengths: Option<Vec3>,
    ) -> Result<Self> {
        config.validate(bodies)?;
        ensure!(rd.is_finite() && rd >= 0., "invalid contact inflation");
        let mut inflated = shape.clone();
        for atom in &mut inflated.atoms {
            atom.radius += rd;
        }
        let exclusion = SphereTree::new(inflated)?;
        if let Some(lengths) = periodic_lengths {
            ensure!(
                lengths
                    .iter()
                    .all(|&l| l.is_finite() && l > 4. * exclusion.bound),
                "contact bias requires L > 4*(body bound + depletant radius)"
            );
        }
        Ok(Self {
            config,
            exclusion,
            lengths: periodic_lengths,
        })
    }
    pub fn score(&self, poses: &[Pose]) -> Result<AssemblyBiasState> {
        ensure!(
            poses.len() == self.config.values.len(),
            "bias body count mismatch"
        );
        for pose in poses {
            pose.validate()?;
        }
        let mut labels: Vec<_> = (0..poses.len()).collect();
        for i in 0..poses.len() {
            for j in i + 1..poses.len() {
                let mut neighbor = poses[j];
                if let Some(lengths) = self.lengths {
                    neighbor.position = std::array::from_fn(|k| {
                        neighbor.position[k]
                            - lengths[k]
                                * ((neighbor.position[k] - poses[i].position[k]) / lengths[k] + 0.5)
                                    .floor()
                    });
                }
                if self
                    .exclusion
                    .overlaps(&Placed::new(poses[i]), &Placed::new(neighbor))
                {
                    let old = labels[j];
                    let new = labels[i];
                    for label in &mut labels {
                        if *label == old {
                            *label = new;
                        }
                    }
                }
            }
        }
        let size = (0..poses.len())
            .map(|i| labels.iter().filter(|&&l| l == i).count())
            .max()
            .unwrap();
        let bias = self.config.values[size - 1];
        Ok(AssemblyBiasState {
            largest_component_size: size,
            bias,
            log_reweight: bias,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BiasDecision {
    pub before: AssemblyBiasState,
    pub proposed: AssemblyBiasState,
    pub log_acceptance: f64,
    pub accepted: bool,
}
/// Never consume RNG for unit acceptance, preserving constant-bias paths.
pub fn decide(
    rng: &mut impl rand::Rng,
    before: AssemblyBiasState,
    proposed: AssemblyBiasState,
) -> BiasDecision {
    let log_acceptance = (before.bias - proposed.bias).min(0.);
    let accepted =
        log_acceptance == 0. || rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_acceptance;
    BiasDecision {
        before,
        proposed,
        log_acceptance,
        accepted,
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BiasCounts {
    pub physical_accepted: u64,
    pub accepted: u64,
    pub rejected: u64,
}
impl BiasCounts {
    pub fn record(&mut self, decision: BiasDecision) {
        self.physical_accepted += 1;
        self.accepted += u64::from(decision.accepted);
        self.rejected += u64::from(!decision.accepted);
    }
    fn since(&self, old: &Self) -> Self {
        Self {
            physical_accepted: self.physical_accepted - old.physical_accepted,
            accepted: self.accepted - old.accepted,
            rejected: self.rejected - old.rejected,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AssemblyBiasCounts {
    pub local: BiasCounts,
    pub global: BiasCounts,
    pub gca: BiasCounts,
    pub center_shift: BiasCounts,
}
impl AssemblyBiasCounts {
    pub fn since(&self, old: &Self) -> Self {
        Self {
            local: self.local.since(&old.local),
            global: self.global.since(&old.global),
            gca: self.gca.since(&old.gca),
            center_shift: self.center_shift.since(&old.center_shift),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Atom;
    use rand::{SeedableRng, rngs::StdRng};
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn shape() -> Shape {
        Shape {
            name: "sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 0.5,
            }],
        }
    }

    #[test]
    fn instantaneous_components_strict_tangency_and_periodic_images() -> Result<()> {
        let config = AssemblyBiasConfig {
            values: vec![0., 2., 4.],
        };
        let open = AssemblyBias::new(config.clone(), &shape(), 0.5, 3, None)?;
        assert_eq!(
            open.score(&[pose(0.), pose(2.), pose(4.)])?
                .largest_component_size,
            1
        );
        assert_eq!(
            open.score(&[pose(0.), pose(1.999), pose(3.998)])?
                .largest_component_size,
            3
        );
        assert_eq!(
            open.score(&[pose(0.), pose(1.999), pose(9.)])?
                .largest_component_size,
            2
        );
        let periodic = AssemblyBias::new(config, &shape(), 0.5, 3, Some([10.; 3]))?;
        assert_eq!(
            periodic
                .score(&[pose(0.2), pose(9.8), pose(5.)])?
                .largest_component_size,
            2
        );
        // No native metadata enters this constructor or score.
        Ok(())
    }

    #[test]
    fn sphere_union_score_matches_brute_atomic_geometry() -> Result<()> {
        let shape = Shape {
            name: "dumbbell".into(),
            volume: 0.,
            atoms: vec![
                Atom {
                    center: [-0.7, 0., 0.],
                    radius: 0.35,
                },
                Atom {
                    center: [0.7, 0., 0.],
                    radius: 0.45,
                },
            ],
        };
        let engine = AssemblyBias::new(
            AssemblyBiasConfig {
                values: vec![0., 1.],
            },
            &shape,
            0.3,
            2,
            None,
        )?;
        let mut rng = StdRng::seed_from_u64(312441);
        for _ in 0..500 {
            let p = Pose {
                position: std::array::from_fn(|_| rng.random_range(-3.0..3.0)),
                orientation: quaternion(cayley(std::array::from_fn(|_| {
                    rng.random_range(-2.0..2.0)
                }))),
            };
            let q = pose(0.);
            let expected = shape.atoms.iter().any(|a| {
                shape.atoms.iter().any(|b| {
                    norm(sub(p.apply(a.center), q.apply(b.center))) < a.radius + b.radius + 0.6
                })
            });
            assert_eq!(
                engine.score(&[p, q])?.largest_component_size,
                if expected { 2 } else { 1 }
            );
        }
        Ok(())
    }

    #[test]
    fn constant_bias_consumes_no_randomness_and_tables_fail_closed() -> Result<()> {
        let state = AssemblyBiasState {
            largest_component_size: 1,
            bias: 7.,
            log_reweight: 7.,
        };
        let mut rng = StdRng::seed_from_u64(123);
        let mut original = StdRng::seed_from_u64(123);
        assert!(decide(&mut rng, state, state).accepted);
        assert_eq!(rng.random::<u64>(), original.random::<u64>());
        for values in [
            vec![],
            vec![0.],
            vec![0., f64::INFINITY],
            vec![-f64::MAX, f64::MAX],
        ] {
            assert!(AssemblyBiasConfig { values }.validate(2).is_err());
        }
        AssemblyBiasConfig {
            values: vec![0., -3.],
        }
        .validate(2)?;
        Ok(())
    }

    #[test]
    fn enumerated_reversible_elementary_kernels_preserve_biased_mass_and_reweight() {
        let pi = [0.2, 0.3, 0.5];
        let b: [f64; 3] = [0., 1.4, -0.7];
        let raw: Vec<_> = (0..3).map(|i| pi[i] * (-b[i]).exp()).collect();
        let norm: f64 = raw.iter().sum();
        let pb: Vec<_> = raw.iter().map(|x| x / norm).collect();
        let kernel = |flows: [[f64; 3]; 3]| {
            let mut p = [[0.; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        p[i][j] = flows[i][j] / pi[i] * (b[i] - b[j]).min(0.).exp();
                    }
                }
                p[i][i] = 1. - p[i].iter().sum::<f64>();
            }
            for i in 0..3 {
                for j in 0..3 {
                    assert!((pb[i] * p[i][j] - pb[j] * p[j][i]).abs() < 2e-15);
                }
            }
            p
        };
        let a = kernel([[0., 0.04, 0.02], [0.04, 0., 0.01], [0.02, 0.01, 0.]]);
        let c = kernel([[0., 0.01, 0.06], [0.01, 0., 0.03], [0.06, 0.03, 0.]]);
        for k in 0..3 {
            let after: f64 = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| pb[i] * a[i][j] * c[j][k])
                .sum();
            assert!((after - pb[k]).abs() < 2e-15);
        }
        let denominator: f64 = (0..3).map(|i| pb[i] * b[i].exp()).sum();
        for i in 0..3 {
            assert!((pb[i] * b[i].exp() / denominator - pi[i]).abs() < 2e-15);
        }
    }
}
