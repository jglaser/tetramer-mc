//! A normalized one-atom envelope of an atomic spherical-wall domain.
//!
//! Fix the first atom of largest radius, with body center `a` and radius `r`.
//! Draw independent Haar `R` and uniform `y` in the ball of radius `W-r`, then
//! set `t = c - R*a + y`. For each rotation this is a translation of that ball,
//! so its Jacobian in translation times normalized rotational Haar is one.
//! Every pose satisfying the full atomic wall satisfies this atom's wall.
//! Other atoms and fixed-body hard cores are deliberately not conditioned on.
//!
//! These are real-arithmetic support/normalization statements. FP64 transforms,
//! RNG execution and boundary classification remain implementation obligations.
//! A numerical failure or a generated point outside the computed support is
//! an error: callers must stop rather than redraw, clip or silently discard it.

use crate::{
    geometry::Shape,
    math::{Pose, Vec3, add, matvec, norm, rotation, scale, sub},
};
use anyhow::{Result, ensure};
use rand::{distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use std::f64::consts::PI;

/// Frozen, serializable witness of the chosen atom and normalized envelope.
/// Fields are private so construction is the only way to change the law.
#[derive(Clone, Debug, Serialize)]
pub struct AtomWallEnvelope {
    atom_index: usize,
    atom_center: Vec3,
    atom_radius: f64,
    wall_center: Vec3,
    wall_radius: f64,
    envelope_radius: f64,
    log_volume: f64,
}

impl AtomWallEnvelope {
    pub fn new(shape: &Shape, wall_center: Vec3, wall_radius: f64) -> Result<Self> {
        ensure!(!shape.atoms.is_empty(), "Empty wall-envelope shape");
        ensure!(
            wall_center.iter().all(|v| v.is_finite())
                && wall_radius.is_finite()
                && wall_radius > 0.,
            "Invalid wall-envelope wall"
        );
        let mut atom_index = 0;
        for (index, atom) in shape.atoms.iter().enumerate() {
            ensure!(
                atom.center.iter().all(|v| v.is_finite())
                    && atom.radius.is_finite()
                    && atom.radius > 0.,
                "Invalid wall-envelope atom {index}"
            );
            // Strict comparison preserves the first index on a radius tie.
            if atom.radius > shape.atoms[atom_index].radius {
                atom_index = index;
            }
        }
        let atom = &shape.atoms[atom_index];
        ensure!(
            wall_radius > atom.radius,
            "Wall envelope must have positive radius"
        );
        let envelope_radius = wall_radius - atom.radius;
        let log_volume = (4. * PI / 3.).ln() + 3. * envelope_radius.ln();
        ensure!(
            envelope_radius.is_finite() && envelope_radius > 0. && log_volume.is_finite(),
            "Unrepresentable wall-envelope volume"
        );
        Ok(Self {
            atom_index,
            atom_center: atom.center,
            atom_radius: atom.radius,
            wall_center,
            wall_radius,
            envelope_radius,
            log_volume,
        })
    }

    /// One unconditional draw. No atom-wall or hard-core retry is performed.
    pub fn draw(&self, rng: &mut StdRng) -> Result<Pose> {
        let raw: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let length = raw.iter().fold(0_f64, |total, &v| total.hypot(v));
        ensure!(
            raw.iter().all(|v| v.is_finite()) && length.is_finite() && length > 0.,
            "Invalid wall-envelope Haar normal length"
        );
        let orientation = raw.map(|v| v / length);
        let direction: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let length = norm(direction);
        ensure!(
            direction.iter().all(|v| v.is_finite()) && length.is_finite() && length > 0.,
            "Invalid wall-envelope direction normal length"
        );
        let uniform: f64 = Open01.sample(rng);
        let radius = self.envelope_radius * uniform.cbrt();
        ensure!(
            radius.is_finite() && radius > 0.,
            "Unrepresentable wall-envelope radial draw"
        );
        let displacement = scale(direction.map(|v| v / length), radius);
        ensure!(
            norm(displacement) > 0.,
            "Underflowed wall-envelope displacement"
        );
        let position = add(
            sub(
                self.wall_center,
                matvec(rotation(orientation), self.atom_center),
            ),
            displacement,
        );
        let pose = Pose {
            position,
            orientation,
        };
        pose.validate()?;
        ensure!(
            self.log_density(pose)?.is_finite(),
            "Generated wall-envelope pose lies outside FP64 support; do not redraw"
        );
        Ok(pose)
    }

    /// Density relative to `d^3t` times normalized SO(3) Haar, including the
    /// boundary. Exterior points have `-inf`; invalid arithmetic is an error.
    pub fn log_density(&self, pose: Pose) -> Result<f64> {
        pose.validate()?;
        let displacement = sub(pose.apply(self.atom_center), self.wall_center);
        let distance = norm(displacement);
        ensure!(
            displacement.iter().all(|v| v.is_finite()) && distance.is_finite(),
            "Unrepresentable wall-envelope support query"
        );
        Ok(if distance <= self.envelope_radius {
            -self.log_volume
        } else {
            f64::NEG_INFINITY
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geometry::Atom,
        math::{cayley, dot, quaternion},
    };
    use rand::SeedableRng;

    fn shape(atoms: &[(Vec3, f64)]) -> Shape {
        Shape {
            name: "wall-envelope-fixture".into(),
            volume: 0.,
            atoms: atoms
                .iter()
                .map(|&(center, radius)| Atom { center, radius })
                .collect(),
        }
    }

    fn pose(position: Vec3) -> Pose {
        Pose {
            position,
            orientation: [1., 0., 0., 0.],
        }
    }

    fn near(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 2e-12 * (1. + a.abs() + b.abs()),
            "{a} != {b}"
        );
    }

    #[test]
    fn largest_atom_ties_and_serialized_witness_are_stable() -> Result<()> {
        let s = shape(&[([0.; 3], 0.2), ([1., 2., 3.], 0.5), ([-2., 0., 0.], 0.5)]);
        let proposal = AtomWallEnvelope::new(&s, [7., -4., 3.], 3.)?;
        assert_eq!(proposal.atom_index, 1);
        let witness = serde_json::to_value(&proposal)?;
        assert_eq!(witness["atom_index"], 1);
        assert_eq!(witness["atom_center"], serde_json::json!([1., 2., 3.]));
        assert_eq!(witness["atom_radius"], 0.5);
        assert_eq!(witness["wall_center"], serde_json::json!([7., -4., 3.]));
        assert_eq!(witness["wall_radius"], 3.);
        assert_eq!(witness["envelope_radius"], 2.5);
        near(
            witness["log_volume"].as_f64().unwrap(),
            (4. * PI * 2.5_f64.powi(3) / 3.).ln(),
        );
        Ok(())
    }

    #[test]
    fn centered_sphere_has_exact_ball_density_and_boundary() -> Result<()> {
        let proposal = AtomWallEnvelope::new(&shape(&[([0.; 3], 0.5)]), [0.; 3], 3.)?;
        let logq = -(4. * PI * 2.5_f64.powi(3) / 3.).ln();
        for position in [[0.; 3], [2.5, 0., 0.], [0., -2.5, 0.], [0.2, 0.3, -0.7]] {
            near(proposal.log_density(pose(position))?, logq);
        }
        assert_eq!(
            proposal.log_density(pose([2.500001, 0., 0.]))?,
            f64::NEG_INFINITY
        );
        Ok(())
    }

    #[test]
    fn asymmetric_offset_shape_full_wall_is_a_subset() -> Result<()> {
        let s = shape(&[
            ([1., 0., 0.], 0.5),
            ([-1., 0.2, 0.1], 0.2),
            ([0.2, 1., -0.3], 0.3),
        ]);
        let center = [7., -4., 3.];
        let proposal = AtomWallEnvelope::new(&s, center, 3.)?;
        let mut valid_count = 0;
        for c in [[0.; 3], [0.2, -0.3, 0.4], [1., 0.5, -0.2]] {
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -3..=3 {
                        let p = Pose {
                            position: add(center, [x as f64 * 0.6, y as f64 * 0.6, z as f64 * 0.6]),
                            orientation: quaternion(cayley(c)),
                        };
                        let wall_valid = s
                            .atoms
                            .iter()
                            .all(|a| norm(sub(p.apply(a.center), center)) <= 3. - a.radius);
                        if wall_valid {
                            valid_count += 1;
                            assert!(proposal.log_density(p)?.is_finite());
                        }
                    }
                }
            }
        }
        assert!(valid_count > 100);
        // The proposal is an envelope, not a whole-shape wall conditioner.
        let p = pose(add(center, [-3., 0., 0.]));
        assert!(proposal.log_density(p)?.is_finite());
        assert!(norm(sub(p.apply(s.atoms[1].center), center)) > 3. - s.atoms[1].radius);
        Ok(())
    }

    #[test]
    fn rotation_dependent_translation_has_unit_volume_and_correct_offset() -> Result<()> {
        let proposal =
            AtomWallEnvelope::new(&shape(&[([1.5, -0.5, 0.25], 0.5)]), [3., -2., 1.], 4.)?;
        let y = [0.25, -0.5, 0.75];
        for c in [[0.; 3], [0.3, -0.4, 0.2], [2., -1., 0.5]] {
            let orientation = quaternion(cayley(c));
            let offset = sub(
                proposal.wall_center,
                matvec(rotation(orientation), proposal.atom_center),
            );
            let p = Pose {
                position: add(offset, y),
                orientation,
            };
            let recovered = sub(p.apply(proposal.atom_center), proposal.wall_center);
            for i in 0..3 {
                near(recovered[i], y[i]);
            }
            near(proposal.log_density(p)?, -proposal.log_volume);
            // At every fixed rotation, three unit y-edges remain unit t-edges.
            let edges: [Vec3; 3] = std::array::from_fn(|i| {
                let mut next = y;
                next[i] += 1.;
                sub(add(offset, next), p.position)
            });
            let cross = [
                edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1],
                edges[1][2] * edges[2][0] - edges[1][0] * edges[2][2],
                edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0],
            ];
            near(dot(edges[0], cross), 1.);
        }
        Ok(())
    }

    #[test]
    fn malformed_geometry_and_unrepresentable_queries_fail() -> Result<()> {
        assert!(AtomWallEnvelope::new(&shape(&[]), [0.; 3], 2.).is_err());
        for radius in [0., -0.1, f64::NAN, f64::INFINITY] {
            assert!(AtomWallEnvelope::new(&shape(&[([0.; 3], radius)]), [0.; 3], 2.).is_err());
        }
        let s = shape(&[([0.; 3], 0.5)]);
        for radius in [0., -1., 0.4, 0.5, f64::NAN, f64::INFINITY] {
            assert!(AtomWallEnvelope::new(&s, [0.; 3], radius).is_err());
        }
        assert!(AtomWallEnvelope::new(&s, [f64::NAN, 0., 0.], 2.).is_err());
        assert!(
            AtomWallEnvelope::new(
                &shape(&[([0.; 3], 0.5), ([f64::INFINITY, 0., 0.], 0.1)]),
                [0.; 3],
                2.
            )
            .is_err()
        );
        let proposal = AtomWallEnvelope::new(&s, [0.; 3], 2.)?;
        assert!(
            proposal
                .log_density(Pose {
                    orientation: [0.; 4],
                    ..pose([0.; 3])
                })
                .is_err()
        );
        assert!(proposal.log_density(pose([f64::NAN, 0., 0.])).is_err());
        let extreme = AtomWallEnvelope::new(&shape(&[([f64::MAX, 0., 0.], 0.5)]), [0.; 3], 2.)?;
        assert!(extreme.log_density(pose([f64::MAX, 0., 0.])).is_err());
        Ok(())
    }

    #[test]
    fn fixed_seed_ball_and_haar_moments_include_every_draw() -> Result<()> {
        let proposal =
            AtomWallEnvelope::new(&shape(&[([1., -0.5, 0.25], 0.5)]), [7., -4., 3.], 3.)?;
        let mut rng = StdRng::seed_from_u64(6100412001);
        const N: usize = 8192;
        let mut sums = [0.; 7];
        let mut squares = [0.; 7];
        for _ in 0..N {
            let p = proposal.draw(&mut rng)?;
            near(proposal.log_density(p)?, -proposal.log_volume);
            let y = scale(
                sub(p.apply(proposal.atom_center), proposal.wall_center),
                1. / proposal.envelope_radius,
            );
            let r = norm(y);
            let matrix = rotation(p.orientation);
            let r00 = matrix[0][0];
            let values = [
                y[0],
                y[0] * y[0],
                r * r,
                r * r * r,
                r00,
                r00 * r00,
                r * r * r00,
            ];
            for i in 0..values.len() {
                sums[i] += values[i];
                squares[i] += values[i] * values[i];
            }
        }
        // Analytic uniform-ball and Haar moments; no retained-draw conditioning.
        for (i, expected) in [0., 1. / 5., 3. / 5., 1. / 2., 0., 1. / 3., 0.]
            .into_iter()
            .enumerate()
        {
            let mean = sums[i] / N as f64;
            let variance = (squares[i] - sums[i] * sums[i] / N as f64) / (N - 1) as f64;
            let se = (variance / N as f64).sqrt();
            assert!(
                (mean - expected).abs() <= 6. * se + 1e-12,
                "moment {i}: {mean} vs {expected}, SE {se}"
            );
        }
        Ok(())
    }
}
