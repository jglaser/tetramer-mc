//! Normalized radial interval sampling inside the original conditional ellipsoid.
//! Angular coordinates and the whitened translation direction are never retried:
//! an empty interval set uses the original conditional uniform radius. Thus the
//! full density is the uniform six-ball density times an explicit radial ratio.
use super::{Chart, Result, StandardNormal, StdRng, draw_uniform};
use crate::math::*;
use anyhow::ensure;
use rand::RngExt;
use rand_distr::Distribution;
use serde::Deserialize;

type Interval = (f64, f64);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GuideFile {
    schema: String,
    region_sha256: String,
    defensive_uniform_shell_probability: f64,
    inner_radius: f64,
    widths: Vec<f64>,
    interfaces: Vec<Interface>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Interface {
    moving_members: Vec<Vec3>,
    target_world_members: Vec<Vec3>,
}

pub(super) struct ConditionalRayGuide {
    pub(super) alpha: f64,
    inner: f64,
    widths: Vec<f64>,
    interfaces: Vec<Interface>,
    angular_lower: Mat3,
    regression: Mat3,
    conditional_lower: Mat3,
}

struct Ray {
    x: [f64; 6],
    mean: Vec3,
    direction: Vec3,
    radius: f64,
    maximum: f64,
    origin_world: Vec3,
    vector_world: Vec3,
    rotation_world: Mat3,
}

fn lower_solve(lower: Mat3, rhs: Vec3) -> Vec3 {
    let mut x = [0.; 3];
    for i in 0..3 {
        x[i] = (rhs[i] - (0..i).map(|j| lower[i][j] * x[j]).sum::<f64>()) / lower[i][i];
    }
    x
}
fn chol(covariance: Mat3) -> Result<Mat3> {
    let mut lower = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let value = 0.5 * (covariance[i][j] + covariance[j][i])
                - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            lower[i][j] = if i == j {
                ensure!(
                    value.is_finite() && value > 0.,
                    "Nonpositive conditional covariance pivot"
                );
                value.sqrt()
            } else {
                value / lower[j][j]
            };
        }
    }
    Ok(lower)
}
fn union(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut result: Vec<Interval> = Vec::new();
    for (lo, hi) in intervals {
        if let Some(last) = result.last_mut()
            && lo <= last.1
        {
            last.1 = last.1.max(hi);
        } else {
            result.push((lo, hi));
        }
    }
    result
}
fn difference(outer: &[Interval], inner: &[Interval]) -> Vec<Interval> {
    let mut result = Vec::new();
    for &(lo, hi) in outer {
        let mut cursor = lo;
        for &(a, b) in inner {
            if b <= cursor || a >= hi {
                continue;
            }
            if a > cursor {
                result.push((cursor, a.min(hi)));
            }
            cursor = cursor.max(b);
            if cursor >= hi {
                break;
            }
        }
        if cursor < hi {
            result.push((cursor, hi));
        }
    }
    result
}
fn cubic_mass(intervals: &[Interval]) -> f64 {
    intervals
        .iter()
        .map(|&(lo, hi)| (hi - lo) * (hi * hi + hi * lo + lo * lo))
        .sum()
}
fn contains(intervals: &[Interval], r: f64) -> bool {
    intervals.iter().any(|&(lo, hi)| lo <= r && r <= hi)
}

impl ConditionalRayGuide {
    pub(super) fn from_bytes(
        raw: &[u8],
        region_hash: &str,
        chart: &Chart,
        inner_region: f64,
    ) -> Result<Self> {
        let p: GuideFile = serde_json::from_slice(raw)?;
        ensure!(
            p.schema == "defensive-conditional-ray-guide-v1",
            "Unknown conditional-ray guide schema"
        );
        ensure!(
            p.region_sha256 == region_hash,
            "Conditional-ray guide targets another region"
        );
        ensure!(
            inner_region == 0.,
            "Conditional rays require a complete latent ball"
        );
        ensure!(
            p.defensive_uniform_shell_probability.is_finite()
                && p.defensive_uniform_shell_probability > 0.
                && p.defensive_uniform_shell_probability <= 1.,
            "Uniform defensive mass must lie in (0,1]"
        );
        ensure!(
            p.inner_radius.is_finite() && p.inner_radius >= 0. && !p.widths.is_empty(),
            "Invalid conditional-ray radii"
        );
        for &w in &p.widths {
            ensure!(
                w.is_finite()
                    && w > 0.
                    && (p.inner_radius + w).is_finite()
                    && p.inner_radius + w > p.inner_radius
                    && (p.inner_radius + w).powi(2).is_finite(),
                "Unrepresentable conditional-ray width"
            );
        }
        ensure!(!p.interfaces.is_empty(), "Missing guide interfaces");
        for interface in &p.interfaces {
            ensure!(
                !interface.moving_members.is_empty()
                    && interface.moving_members.len() == interface.target_world_members.len(),
                "Guide member arrays must be nonempty and matched"
            );
            ensure!(
                interface
                    .moving_members
                    .iter()
                    .chain(&interface.target_world_members)
                    .flatten()
                    .all(|x| x.is_finite()),
                "Nonfinite guide members"
            );
        }
        let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| chart.lower[i][k] * chart.lower[j][k]).sum())
        });
        let angular_lower = chol(std::array::from_fn(|i| {
            std::array::from_fn(|j| covariance[i + 3][j + 3])
        }))?;
        let mut regression = [[0.; 3]; 3];
        for i in 0..3 {
            let tmp = lower_solve(angular_lower, std::array::from_fn(|j| covariance[i][j + 3]));
            for j in (0..3).rev() {
                regression[i][j] = (tmp[j]
                    - ((j + 1)..3)
                        .map(|k| angular_lower[k][j] * regression[i][k])
                        .sum::<f64>())
                    / angular_lower[j][j];
            }
        }
        let conditional = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                covariance[i][j]
                    - (0..3)
                        .map(|k| regression[i][k] * covariance[k + 3][j])
                        .sum::<f64>()
            })
        });
        let conditional_lower = chol(conditional)?;
        Ok(Self {
            alpha: p.defensive_uniform_shell_probability,
            inner: p.inner_radius,
            widths: p.widths,
            interfaces: p.interfaces,
            angular_lower,
            regression,
            conditional_lower,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.widths.len()
    }

    fn ray(&self, u: [f64; 6], chart: &Chart, outer: f64) -> Result<Ray> {
        let x = chart.coordinates(u);
        let angular = std::array::from_fn(|i| x[i + 3] - chart.mean[i + 3]);
        let a = lower_solve(self.angular_lower, angular);
        let remaining = outer * outer - dot(a, a);
        ensure!(
            remaining.is_finite() && remaining >= -1e-11 * (1. + outer * outer),
            "Pose outside conditional angular support"
        );
        let maximum = remaining.max(0.).sqrt();
        let mean = add(
            [chart.mean[0], chart.mean[1], chart.mean[2]],
            matvec(self.regression, angular),
        );
        let v = lower_solve(self.conditional_lower, sub([x[0], x[1], x[2]], mean));
        let radius = norm(v);
        ensure!(radius.is_finite(), "Nonfinite conditional radius");
        // A density version on the measure-zero radial origin.
        let direction = if radius > 0. {
            scale(v, 1. / radius)
        } else {
            [1., 0., 0.]
        };
        let fixed_r = rotation(chart.fixed.orientation);
        let origin_world = chart.fixed.apply(add(chart.anchor_position, mean));
        let vector_world = matvec(fixed_r, matvec(self.conditional_lower, direction));
        let rotation_world = matmul(
            fixed_r,
            matmul(
                cayley([x[3] / chart.ell, x[4] / chart.ell, x[5] / chart.ell]),
                chart.anchor_rotation,
            ),
        );
        Ok(Ray {
            x,
            mean,
            direction,
            radius,
            maximum,
            origin_world,
            vector_world,
            rotation_world,
        })
    }

    fn interface_intervals(&self, ray: &Ray, bound: f64) -> Result<Vec<Interval>> {
        let aa = dot(ray.vector_world, ray.vector_world);
        ensure!(
            aa.is_finite() && aa > 0.,
            "Invalid conditional ray direction"
        );
        let mut result = Vec::new();
        for interface in &self.interfaces {
            let (mut lo, mut hi) = (0_f64, ray.maximum);
            for (&moving, &target) in interface
                .moving_members
                .iter()
                .zip(&interface.target_world_members)
            {
                let displacement = sub(
                    add(ray.origin_world, matvec(ray.rotation_world, moving)),
                    target,
                );
                let center = -dot(displacement, ray.vector_world) / aa;
                // Project onto the line first to avoid subtracting two large
                // squared quantities in the quadratic discriminant.
                let closest = add(displacement, scale(ray.vector_world, center));
                let half_squared = (bound * bound - dot(closest, closest)) / aa;
                ensure!(
                    center.is_finite() && half_squared.is_finite(),
                    "Nonfinite quadratic interval"
                );
                if half_squared <= 0. {
                    hi = lo;
                    break;
                }
                let half = half_squared.sqrt();
                lo = lo.max(center - half);
                hi = hi.min(center + half);
                if hi <= lo {
                    break;
                }
            }
            if hi > lo {
                result.push((lo, hi));
            }
        }
        Ok(union(result))
    }
    fn intervals(&self, ray: &Ray, width: usize) -> Result<Vec<Interval>> {
        let inner = self.interface_intervals(ray, self.inner)?;
        let outer = self.interface_intervals(ray, self.inner + self.widths[width])?;
        Ok(difference(&outer, &inner))
    }
    #[cfg(test)]
    pub(super) fn selected_fallback(
        &self,
        u: [f64; 6],
        chart: &Chart,
        outer: f64,
        width: usize,
    ) -> Result<bool> {
        let ray = self.ray(u, chart, outer)?;
        Ok(cubic_mass(&self.intervals(&ray, width)?) == 0.)
    }
    pub(super) fn log_density(
        &self,
        u: [f64; 6],
        inside: bool,
        log_volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<f64> {
        if !inside {
            return Ok(f64::NEG_INFINITY);
        }
        if self.alpha == 1. {
            return Ok(-log_volume);
        }
        let ray = self.ray(u, chart, outer)?;
        let mut ratio = 0.;
        for i in 0..self.widths.len() {
            let intervals = self.intervals(&ray, i)?;
            let mass = cubic_mass(&intervals);
            ratio += if mass == 0. {
                1.
            } else if contains(&intervals, ray.radius) {
                ray.maximum.powi(3) / mass
            } else {
                0.
            };
        }
        let full_ratio = self.alpha + (1. - self.alpha) * ratio / self.widths.len() as f64;
        ensure!(
            full_ratio.is_finite() && full_ratio > 0.,
            "Unrepresentable conditional-ray density"
        );
        Ok(full_ratio.ln() - log_volume)
    }
    pub(super) fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        if self.alpha == 1. || rng.random::<f64>() < self.alpha {
            let (u, radius) = draw_uniform(rng, outer, 0., 1.)?;
            return Ok((u, radius, None, None));
        }
        // Keep the exact angular marginal; its discarded translation is not a
        // condition for retrying this draw or selecting a different direction.
        let (seed, _) = draw_uniform(rng, outer, 0., 1.)?;
        let mut ray = self.ray(seed, chart, outer)?;
        let normal: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let length = norm(normal);
        ensure!(
            length.is_finite() && length > 0. && ray.maximum > 0.,
            "Degenerate conditional draw; never silently retry"
        );
        ray.direction = scale(normal, 1. / length);
        ray.vector_world = matvec(
            rotation(chart.fixed.orientation),
            matvec(self.conditional_lower, ray.direction),
        );
        let component = rng.random_range(0..self.widths.len());
        let intervals = self.intervals(&ray, component)?;
        let mass = cubic_mass(&intervals);
        let fallback = mass == 0.;
        let uniform = rng.random::<f64>();
        let radial = if fallback {
            ray.maximum * uniform.cbrt()
        } else {
            let mut position = uniform * mass;
            let mut chosen = *intervals.last().unwrap();
            for (index, &(lo, hi)) in intervals.iter().enumerate() {
                let part = (hi - lo) * (hi * hi + hi * lo + lo * lo);
                if position < part || index + 1 == intervals.len() {
                    chosen = (lo, hi);
                    break;
                }
                position -= part;
            }
            // Protect only against accumulation roundoff at the last interval.
            let part = (chosen.1 - chosen.0)
                * (chosen.1 * chosen.1 + chosen.1 * chosen.0 + chosen.0 * chosen.0);
            (chosen.0.powi(3) + position.clamp(0., part)).cbrt()
        };
        let t = add(
            ray.mean,
            matvec(self.conditional_lower, scale(ray.direction, radial)),
        );
        ray.x[..3].copy_from_slice(&t);
        let mut u = [0.; 6];
        for i in 0..6 {
            u[i] =
                (ray.x[i] - chart.mean[i] - (0..i).map(|j| chart.lower[i][j] * u[j]).sum::<f64>())
                    / chart.lower[i][i];
        }
        let radius = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        ensure!(
            radius.is_finite() && radius <= outer,
            "Conditional draw left the latent ball; stop instead of censoring or retrying"
        );
        Ok((u, radius, Some(component), Some(fallback)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use serde_json::{Value, json};
    use std::f64::consts::PI;

    fn chart(correlated: bool) -> Chart {
        let mut lower: [[f64; 6]; 6] = [[0.; 6]; 6];
        for (i, row) in lower.iter_mut().enumerate() {
            row[i] = 1.;
        }
        if correlated {
            lower[0][0] = 0.7;
            lower[1][1] = 1.2;
            lower[2][2] = 0.8;
            lower[3][0] = 1.1;
            lower[3][1] = 0.2;
            lower[3][3] = 0.5;
            lower[4][1] = -0.7;
            lower[4][4] = 0.9;
            lower[5][0] = 0.4;
            lower[5][2] = 0.3;
            lower[5][5] = 1.1;
        }
        Chart {
            log_det: (0..6).map(|i| lower[i][i].ln()).sum(),
            lower,
            mean: if correlated {
                [0.3, -0.1, 0.2, 0.2, -0.1, 0.4]
            } else {
                [0.; 6]
            },
            anchor_position: if correlated {
                [1.2, -0.3, 0.7]
            } else {
                [0.; 3]
            },
            anchor_rotation: if correlated {
                cayley([-0.1, 0.4, 0.2])
            } else {
                IDENTITY
            },
            fixed: Pose {
                position: if correlated { [4., -2., 1.] } else { [0.; 3] },
                orientation: quaternion(if correlated {
                    cayley([0.2, -0.1, 0.3])
                } else {
                    IDENTITY
                }),
            },
            ell: 2.,
        }
    }
    fn data(alpha: f64) -> Value {
        json!({"schema":"defensive-conditional-ray-guide-v1","region_sha256":"target",
            "defensive_uniform_shell_probability":alpha,"inner_radius":1.,"widths":[0.5,1.,2.],
            "interfaces":[{"moving_members":[[0.,0.,0.]],"target_world_members":[[0.,0.,0.]]}]})
    }
    fn guide(raw: Value, chart: &Chart) -> Result<ConditionalRayGuide> {
        ConditionalRayGuide::from_bytes(raw.to_string().as_bytes(), "target", chart, 0.)
    }
    fn check_mean(values: &[f64], expected: f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let se = ((values.iter().map(|x| x * x).sum::<f64>() / n - mean * mean).max(0.) / (n - 1.))
            .sqrt();
        assert!(
            (mean - expected).abs() < 6.5 * se + 1e-10,
            "mean={mean}, expected={expected}, se={se}"
        );
    }
    #[test]
    fn interval_union_difference_and_cubic_mass() {
        let a = union(vec![(1., 3.), (2., 4.), (6., 7.), (0., 0.5)]);
        assert_eq!(a, vec![(0., 0.5), (1., 4.), (6., 7.)]);
        let b = union(vec![(0.25, 2.), (3., 6.5)]);
        let intervals = difference(&a, &b);
        assert_eq!(intervals, vec![(0., 0.25), (2., 3.), (6.5, 7.)]);
        assert!(
            (cubic_mass(&intervals) - (0.25_f64.powi(3) + 27. - 8. + 343. - 6.5_f64.powi(3))).abs()
                < 1e-12
        );
    }
    #[test]
    fn sphere_intervals_full_density_and_duplicate_interfaces() -> Result<()> {
        let chart = chart(false);
        let raw = data(0.3);
        let g = guide(raw.clone(), &chart)?;
        let radius: f64 = 3.;
        let log_volume = (PI.powi(3) * radius.powi(6) / 6.).ln();
        let u = [1.2, 0., 0., 0., 0., 0.];
        let ratio = 0.3 + 0.7 / 3. * (27. / (1.5_f64.powi(3) - 1.) + 27. / 7. + 27. / 26.);
        assert!(
            (g.log_density(u, true, log_volume, &chart, radius)? - (ratio.ln() - log_volume)).abs()
                < 1e-13
        );
        let ray = g.ray(u, &chart, radius)?;
        assert_eq!(g.intervals(&ray, 0)?, vec![(1., 1.5)]);
        let mut duplicate = raw;
        let interface = duplicate["interfaces"][0].clone();
        duplicate["interfaces"]
            .as_array_mut()
            .unwrap()
            .push(interface);
        let h = guide(duplicate, &chart)?;
        assert_eq!(
            g.log_density(u, true, log_volume, &chart, radius)?,
            h.log_density(u, true, log_volume, &chart, radius)?
        );
        let absent = [0.2, 0., 0., 0., 0., 0.];
        assert!(
            (g.log_density(absent, true, log_volume, &chart, radius)?
                - (0.3_f64.ln() - log_volume))
                .abs()
                < 1e-13
        );
        assert_eq!(
            g.log_density(u, false, log_volume, &chart, radius)?,
            f64::NEG_INFINITY
        );
        Ok(())
    }
    #[test]
    fn all_members_constrain_an_interface_and_interfaces_union() -> Result<()> {
        let chart = chart(false);
        let mut raw = data(0.5);
        raw["inner_radius"] = json!(0.1);
        raw["widths"] = json!([0.9]);
        raw["interfaces"] = json!([
            {"moving_members":[[0.,0.,0.],[0.,0.,0.]],"target_world_members":[[1.,0.,0.],[2.,0.,0.]]},
            {"moving_members":[[0.,0.,0.]],"target_world_members":[[5.,0.,0.]]}]);
        let g = guide(raw, &chart)?;
        let ray = g.ray([1., 0., 0., 0., 0., 0.], &chart, 7.)?;
        assert_eq!(g.interface_intervals(&ray, 1.)?, vec![(1., 2.), (4., 6.)]);
        assert_eq!(g.intervals(&ray, 0)?, vec![(1., 2.), (4., 4.9), (5.1, 6.)]);
        Ok(())
    }
    #[test]
    fn alpha_one_preserves_legacy_rng_and_pose_bits() -> Result<()> {
        let chart = chart(true);
        let g = guide(data(1.), &chart)?;
        let mut a = StdRng::seed_from_u64(77102);
        let mut b = StdRng::seed_from_u64(77102);
        for _ in 0..128 {
            let expected = draw_uniform(&mut a, 4., 0., 1.)?;
            let actual = g.draw(&mut b, &chart, 4.)?;
            assert_eq!(expected, (actual.0, actual.1));
            assert_eq!((actual.2, actual.3), (None, None));
            assert_eq!(a.random::<u64>(), b.random::<u64>());
            assert_eq!(g.log_density(actual.0, true, 2.7, &chart, 4.)?, -2.7);
        }
        Ok(())
    }
    #[test]
    fn empty_intervals_fallback_has_uniform_density_and_radial_moments() -> Result<()> {
        let chart = chart(true);
        let mut raw = data(0.2);
        raw["interfaces"][0]["target_world_members"] = json!([[1e5, 0., 0.]]);
        let g = guide(raw, &chart)?;
        let mut rng = StdRng::seed_from_u64(889013);
        let count = 20_000;
        let mut samples = Vec::new();
        let mut direction_moments = Vec::new();
        let volume = PI.powi(3) * 4_f64.powi(6) / 6.;
        for _ in 0..count {
            let (u, r, selected, fallback) = g.draw(&mut rng, &chart, 4.)?;
            assert!(r <= 4.);
            assert_eq!(
                g.log_density(u, true, volume.ln(), &chart, 4.)?,
                -volume.ln()
            );
            if selected.is_some() {
                assert_eq!(fallback, Some(true));
            }
            let ray = g.ray(u, &chart, 4.)?;
            samples.push((ray.radius / ray.maximum).powi(3));
            direction_moments.push(ray.direction[0].powi(2));
        }
        check_mean(&samples, 0.5);
        check_mean(&direction_moments, 1. / 3.);
        Ok(())
    }
    #[test]
    fn conditional_factorization_preserves_angular_marginal_and_volume() -> Result<()> {
        let chart = chart(true);
        let mut raw = data(0.5);
        raw["interfaces"][0]["target_world_members"] = json!([chart.fixed.apply(add(
            chart.anchor_position,
            [chart.mean[0], chart.mean[1], chart.mean[2]]
        ))]);
        let g = guide(raw, &chart)?;
        let mut rng = StdRng::seed_from_u64(111276);
        let radius: f64 = 4.;
        let volume = PI.powi(3) * radius.powi(6) / 6.;
        let mut weights = Vec::new();
        let mut angular_moments = Vec::new();
        let mut visited = 0;
        for _ in 0..40_000 {
            let (u, r, selected, fallback) = g.draw(&mut rng, &chart, radius)?;
            assert!(r <= radius);
            let ray = g.ray(u, &chart, radius)?;
            let a = lower_solve(
                g.angular_lower,
                [
                    ray.x[3] - chart.mean[3],
                    ray.x[4] - chart.mean[4],
                    ray.x[5] - chart.mean[5],
                ],
            );
            assert!((dot(a, a) + ray.radius.powi(2) - r.powi(2)).abs() < 1e-10);
            angular_moments.push(dot(a, a));
            if let Some(i) = selected {
                assert_eq!(fallback, Some(g.selected_fallback(u, &chart, radius, i)?));
                if fallback == Some(false) {
                    assert!(contains(&g.intervals(&ray, i)?, ray.radius));
                    visited += 1;
                }
            }
            let q = g.log_density(u, true, volume.ln(), &chart, radius)?.exp();
            assert!(q >= 0.5 / volume * (1. - 1e-12));
            weights.push(1. / q);
        }
        assert!(visited > 1000);
        check_mean(&weights, volume);
        check_mean(&angular_moments, 3. * radius.powi(2) / 8.);
        Ok(())
    }
    #[test]
    fn global_isometry_preserves_density_and_interval_geometry() -> Result<()> {
        let original = chart(true);
        let mut changed = chart(true);
        let transform = Pose {
            position: [-3., 5., 2.],
            orientation: quaternion(cayley([0.2, 0.4, -0.3])),
        };
        changed.fixed.position = transform.apply(changed.fixed.position);
        changed.fixed.orientation = quaternion(matmul(
            rotation(transform.orientation),
            rotation(changed.fixed.orientation),
        ));
        let raw = data(0.4);
        let mut moved = raw.clone();
        moved["interfaces"][0]["target_world_members"] = json!([transform.apply([0.; 3])]);
        let g = guide(raw, &original)?;
        let h = guide(moved, &changed)?;
        let mut rng = StdRng::seed_from_u64(93901);
        for _ in 0..1000 {
            let (u, _) = draw_uniform(&mut rng, 4., 0., 1.)?;
            assert!(
                (g.log_density(u, true, 5., &original, 4.)?
                    - h.log_density(u, true, 5., &changed, 4.)?)
                .abs()
                    < 2e-10
            );
        }
        Ok(())
    }
    #[test]
    fn invalid_guides_and_punctured_regions_fail_closed() {
        let chart = chart(false);
        for replacement in [json!(0.), json!(-0.1), json!(1.1)] {
            let mut raw = data(0.5);
            raw["defensive_uniform_shell_probability"] = replacement;
            assert!(guide(raw, &chart).is_err());
        }
        for replacement in [json!([]), json!([0.]), json!([-1.])] {
            let mut raw = data(0.5);
            raw["widths"] = replacement;
            assert!(guide(raw, &chart).is_err());
        }
        let mut raw = data(0.5);
        raw["interfaces"][0]["moving_members"] = json!([]);
        assert!(guide(raw, &chart).is_err());
        assert!(
            ConditionalRayGuide::from_bytes(data(0.5).to_string().as_bytes(), "target", &chart, 1.)
                .is_err()
        );
        assert!(
            ConditionalRayGuide::from_bytes(
                data(0.5).to_string().as_bytes(),
                "different",
                &chart,
                0.
            )
            .is_err()
        );
    }
}
