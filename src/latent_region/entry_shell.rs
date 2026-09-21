//! Exact orientation marginal of the original latent ball, followed by a
//! normalized mixture of member-centered translation shells. These shells
//! guide proposals only; exterior or hard-invalid draws remain target zeros.
use super::{Chart, PI, Pose, Result, StandardNormal, StdRng, draw_uniform, log_add};
use crate::math::*;
use anyhow::ensure;
use rand::RngExt;
use rand_distr::Distribution;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GuideFile {
    schema: String,
    region_sha256: String,
    defensive_uniform_shell_probability: f64,
    entries: Vec<EntryFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryFile {
    moving_member: Vec3,
    target_world_member: Vec3,
    inner_radius: f64,
    outer_radius: f64,
    weight: f64,
}

struct Entry {
    moving_member: Vec3,
    target_world_member: Vec3,
    inner_squared: f64,
    outer_squared: f64,
    outer: f64,
    inner_cubed_ratio: f64,
    shell_fraction: f64,
    weight: f64,
    log_volume: f64,
}

pub(super) struct EntryShellGuide {
    pub(super) alpha: f64,
    entries: Vec<Entry>,
    angular_lower: [[f64; 3]; 3],
    angular_log_det: f64,
}

impl EntryShellGuide {
    pub(super) fn from_bytes(
        raw: &[u8],
        region_hash: &str,
        chart: &Chart,
        inner_region_radius: f64,
    ) -> Result<Self> {
        let parsed: GuideFile = serde_json::from_slice(raw)?;
        ensure!(
            parsed.schema == "defensive-entry-shell-guide-v1",
            "Unknown entry-shell guide schema"
        );
        ensure!(
            parsed.region_sha256 == region_hash,
            "Entry-shell guide targets another region"
        );
        ensure!(
            inner_region_radius == 0.,
            "Entry-shell guide requires a latent ball, not a punctured shell"
        );
        let alpha = parsed.defensive_uniform_shell_probability;
        ensure!(
            alpha.is_finite() && alpha > 0. && alpha <= 1.,
            "Uniform defensive mass must lie in (0,1]"
        );
        ensure!(!parsed.entries.is_empty(), "Missing entry-shell components");
        let total: f64 = parsed.entries.iter().map(|e| e.weight).sum();
        ensure!(
            total.is_finite() && total > 0.,
            "Invalid entry-shell mixture mass"
        );
        let mut entries = Vec::new();
        for e in parsed.entries {
            ensure!(
                e.moving_member
                    .iter()
                    .chain(e.target_world_member.iter())
                    .all(|x| x.is_finite()),
                "Nonfinite member coordinates"
            );
            ensure!(
                e.inner_radius.is_finite()
                    && e.outer_radius.is_finite()
                    && e.inner_radius >= 0.
                    && e.outer_radius > e.inner_radius
                    && e.outer_radius.powi(2).is_finite(),
                "Invalid entry-shell radii"
            );
            ensure!(
                e.weight.is_finite() && e.weight > 0. && e.weight / total > 0.,
                "Invalid entry-shell weight"
            );
            let inner_cubed_ratio = (e.inner_radius / e.outer_radius).powi(3);
            let shell_fraction = 1. - inner_cubed_ratio;
            let log_volume = (4. * PI / 3.).ln() + 3. * e.outer_radius.ln() + shell_fraction.ln();
            ensure!(
                shell_fraction > 0. && log_volume.is_finite(),
                "Unrepresentable entry-shell volume"
            );
            entries.push(Entry {
                moving_member: e.moving_member,
                target_world_member: e.target_world_member,
                inner_squared: e.inner_radius.powi(2),
                outer_squared: e.outer_radius.powi(2),
                outer: e.outer_radius,
                inner_cubed_ratio,
                shell_fraction,
                weight: e.weight / total,
                log_volume,
            });
        }
        // All SIX columns matter: translation/angular correlation survives
        // marginalization. The bottom-right Cholesky block is conditional.
        let covariance: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| chart.lower[i + 3][k] * chart.lower[j + 3][k])
                    .sum()
            })
        });
        let mut angular_lower = [[0.; 3]; 3];
        for i in 0..3 {
            for j in 0..=i {
                let value = covariance[i][j]
                    - (0..j)
                        .map(|k| angular_lower[i][k] * angular_lower[j][k])
                        .sum::<f64>();
                angular_lower[i][j] = if i == j {
                    ensure!(
                        value.is_finite() && value > 0.,
                        "Nonpositive marginal angular covariance"
                    );
                    value.sqrt()
                } else {
                    value / angular_lower[j][j]
                };
            }
        }
        let angular_log_det = (0..3).map(|i| angular_lower[i][i].ln()).sum();
        Ok(Self {
            alpha,
            entries,
            angular_lower,
            angular_log_det,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    fn angular_log_density(&self, x: [f64; 6], chart: &Chart, radius: f64) -> f64 {
        let mut a = [0.; 3];
        for i in 0..3 {
            a[i] = (x[i + 3]
                - chart.mean[i + 3]
                - (0..i).map(|j| self.angular_lower[i][j] * a[j]).sum::<f64>())
                / self.angular_lower[i][i];
        }
        let squared = dot(a, a);
        if squared >= radius * radius {
            return f64::NEG_INFINITY;
        }
        // V3/V6 = 8/pi². This is the density in raw scaled-Cayley d³xω,
        // NOT Haar measure and NOT a Gaussian orientation marginal.
        8_f64.ln() - 2. * PI.ln() - 3. * radius.ln() - self.angular_log_det
            + 1.5 * (-squared / (radius * radius)).ln_1p()
    }

    fn translation_log_density(&self, pose: Pose) -> f64 {
        let r = rotation(pose.orientation);
        let mut log_density = f64::NEG_INFINITY;
        for e in &self.entries {
            let center = sub(e.target_world_member, matvec(r, e.moving_member));
            let delta = sub(pose.position, center);
            let squared = dot(delta, delta);
            if squared >= e.inner_squared && squared <= e.outer_squared {
                // Count EVERY overlapping shell, not just the selected one.
                log_density = log_add(log_density, e.weight.ln() - e.log_volume);
            }
        }
        log_density
    }

    pub(super) fn log_density(
        &self,
        u: [f64; 6],
        inside: bool,
        log_volume: f64,
        chart: &Chart,
        radius: f64,
    ) -> f64 {
        let uniform = if inside {
            self.alpha.ln() - log_volume
        } else {
            f64::NEG_INFINITY
        };
        if self.alpha == 1. {
            return uniform;
        }
        let x = chart.coordinates(u);
        let (pose, _) = chart.decode(u);
        // dx/dξ=det(L). The Haar factor cancels in qξ; it remains in the
        // physical integration Jacobian Jξ used by the unchanged estimator.
        let guided = (1. - self.alpha).ln()
            + chart.log_det
            + self.angular_log_density(x, chart, radius)
            + self.translation_log_density(pose);
        log_add(uniform, guided)
    }

    pub(super) fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        radius: f64,
    ) -> Result<([f64; 6], f64, Option<usize>)> {
        // Preserve the legacy uniform stream exactly in the alpha=1 limit.
        if self.alpha == 1. || rng.random::<f64>() < self.alpha {
            let (u, radial) = draw_uniform(rng, radius, 0., 1.)?;
            return Ok((u, radial, None));
        }
        let (angular_seed, _) = draw_uniform(rng, radius, 0., 1.)?;
        let (mut pose, _) = chart.decode(angular_seed);
        let selector = rng.random::<f64>();
        let mut index = self.entries.len() - 1;
        let mut cumulative = 0.;
        for (i, e) in self.entries.iter().enumerate() {
            cumulative += e.weight;
            if selector < cumulative {
                index = i;
                break;
            }
        }
        let e = &self.entries[index];
        let direction: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let norm = dot(direction, direction).sqrt();
        ensure!(
            norm.is_finite() && norm > 0.,
            "Invalid shell direction; never silently redraw"
        );
        let radial =
            e.outer * (e.inner_cubed_ratio + e.shell_fraction * rng.random::<f64>()).cbrt();
        let center = sub(
            e.target_world_member,
            matvec(rotation(pose.orientation), e.moving_member),
        );
        pose.position = add(center, direction.map(|x| x * radial / norm));
        pose.validate()?;
        let u = chart.encode(pose)?;
        let radius = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(
            radius.is_finite(),
            "Nonfinite shell backmap; never silently redraw"
        );
        Ok((u, radius, Some(index)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use serde_json::{Value, json};

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
        let log_det = (0..6).map(|i| lower[i][i].ln()).sum();
        Chart {
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
                cayley([0.; 3])
            },
            fixed: Pose {
                position: if correlated { [4., -2., 1.] } else { [0.; 3] },
                orientation: quaternion(if correlated {
                    cayley([0.2, -0.1, 0.3])
                } else {
                    cayley([0.; 3])
                }),
            },
            ell: 2.,
            log_det,
        }
    }

    fn raw(alpha: f64) -> Value {
        json!({"schema":"defensive-entry-shell-guide-v1", "region_sha256":"target",
            "defensive_uniform_shell_probability":alpha,
            "entries":[{"moving_member":[0.6,-0.8,0.2],"target_world_member":[1.,0.,0.5],
                "inner_radius":1.,"outer_radius":3.5,"weight":2.},
                {"moving_member":[-0.1,0.9,0.3],"target_world_member":[0.5,-0.2,0.1],
                "inner_radius":0.,"outer_radius":2.5,"weight":5.}]})
    }

    fn guide(data: Value, chart: &Chart) -> Result<EntryShellGuide> {
        EntryShellGuide::from_bytes(data.to_string().as_bytes(), "target", chart, 0.)
    }

    #[test]
    fn correlated_angular_marginal_uses_all_six_columns_and_nonzero_mean() -> Result<()> {
        let chart = chart(true);
        let guide = guide(raw(0.25), &chart)?;
        let radius: f64 = 3.;
        let mut rng = StdRng::seed_from_u64(77101);
        let mut mean = [0.; 3];
        let mut second = [[0.; 3]; 3];
        let count = 50_000;
        for _ in 0..count {
            let (u, _, _) = guide.draw(&mut rng, &chart, radius)?;
            let x = chart.coordinates(u);
            let d: Vec3 = std::array::from_fn(|i| x[i + 3] - chart.mean[i + 3]);
            for i in 0..3 {
                mean[i] += d[i];
                for j in 0..3 {
                    second[i][j] += d[i] * d[j];
                }
            }
        }
        for i in 0..3 {
            assert!(mean[i].abs() / (count as f64) < 0.025);
            for j in 0..3 {
                let covariance: f64 = (0..6)
                    .map(|k| chart.lower[i + 3][k] * chart.lower[j + 3][k])
                    .sum();
                let expected = radius.powi(2) * covariance / 8.;
                assert!((second[i][j] / count as f64 - expected).abs() < 0.055);
            }
        }
        // The erroneous conditional Cholesky block loses a large variance.
        assert!((guide.angular_lower[0][0].powi(2) - chart.lower[3][3].powi(2)) > 1.);
        Ok(())
    }

    #[test]
    fn complete_density_counts_overlapping_shells_and_raw_cayley_jacobian() -> Result<()> {
        let chart = chart(false);
        let mut data = raw(0.3);
        for e in data["entries"].as_array_mut().unwrap() {
            e["moving_member"] = json!([0., 0., 0.]);
            e["inner_radius"] = json!(0.);
        }
        data["entries"][0]["target_world_member"] = json!([0., 0., 0.]);
        data["entries"][0]["outer_radius"] = json!(2.);
        data["entries"][1]["target_world_member"] = json!([0.5, 0., 0.]);
        data["entries"][1]["outer_radius"] = json!(3.);
        let guide = guide(data, &chart)?;
        let radius: f64 = 3.;
        let volume = PI.powi(3) * radius.powi(6) / 6.;
        let u = [0.2, 0.1, 0., 0.3, 0.4, 0.];
        let p_omega = 8. / PI.powi(2) * (radius.powi(2) - 0.25).powf(1.5) / radius.powi(6);
        let h = (2. / 7.) / (4. * PI * 8. / 3.) + (5. / 7.) / (4. * PI * 27. / 3.);
        let expected = 0.3 / volume + 0.7 * p_omega * h;
        assert!(
            (guide.log_density(u, true, volume.ln(), &chart, radius) - expected.ln()).abs() < 2e-14
        );
        let (_, log_j) = chart.decode(u);
        let haar = 1. / (PI.powi(2) * chart.ell.powi(3) * (1. + 0.25 / chart.ell.powi(2)).powi(2));
        assert!((log_j.exp() - haar).abs() < 1e-15);
        assert!((expected / log_j.exp() - (0.3 / volume + 0.7 * p_omega * h) / haar).abs() < 1e-13);
        assert_eq!(
            guide.log_density(
                [0., 0., 0., radius, 0., 0.],
                false,
                volume.ln(),
                &chart,
                radius
            ),
            f64::NEG_INFINITY
        );
        assert_eq!(
            guide.log_density(
                [20., 0., 0., 0., 0., 0.],
                false,
                volume.ln(),
                &chart,
                radius
            ),
            f64::NEG_INFINITY
        );
        Ok(())
    }

    #[test]
    fn pure_uniform_is_bitwise_legacy_including_next_rng_value() -> Result<()> {
        let chart = chart(true);
        let guide = guide(raw(1.), &chart)?;
        let radius: f64 = 4.;
        let log_volume = (PI.powi(3) * radius.powi(6) / 6.).ln();
        let mut a = StdRng::seed_from_u64(331);
        let mut b = StdRng::seed_from_u64(331);
        for _ in 0..128 {
            let expected = draw_uniform(&mut a, radius, 0., 1.)?;
            let actual = guide.draw(&mut b, &chart, radius)?;
            assert_eq!(expected, (actual.0, actual.1));
            assert_eq!(actual.2, None);
            assert_eq!(
                guide.log_density(actual.0, true, log_volume, &chart, radius),
                -log_volume
            );
            assert_eq!(chart.decode(expected.0), chart.decode(actual.0));
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
        Ok(())
    }

    #[test]
    fn shell_radius_cubed_is_uniform_and_component_selection_has_correct_mass() -> Result<()> {
        let chart = chart(true);
        let guide = guide(raw(0.2), &chart)?;
        let mut rng = StdRng::seed_from_u64(60011);
        let mut counts = [0; 3];
        let mut sums = [[0.; 2]; 2];
        for _ in 0..50_000 {
            let (u, _, selected) = guide.draw(&mut rng, &chart, 3.)?;
            counts[selected.map_or(0, |i| i + 1)] += 1;
            if let Some(k) = selected {
                let (pose, _) = chart.decode(u);
                let e = &guide.entries[k];
                let center = sub(
                    e.target_world_member,
                    matvec(rotation(pose.orientation), e.moving_member),
                );
                let d = sub(pose.position, center);
                let r = dot(d, d).sqrt();
                let transformed = ((r / e.outer).powi(3) - e.inner_cubed_ratio) / e.shell_fraction;
                assert!((-1e-12..=1. + 1e-12).contains(&transformed));
                sums[k][0] += transformed;
                sums[k][1] += transformed * transformed;
            }
        }
        for (n, p) in counts.into_iter().zip([0.2, 0.8 * 2. / 7., 0.8 * 5. / 7.]) {
            assert!((n as f64 / 50_000. - p).abs() < 0.008);
        }
        for k in 0..2 {
            assert!((sums[k][0] / counts[k + 1] as f64 - 0.5).abs() < 0.008);
            assert!((sums[k][1] / counts[k + 1] as f64 - 1. / 3.).abs() < 0.008);
        }
        Ok(())
    }

    #[test]
    fn unconditional_fixed_n_recovers_known_ball_volume_with_bounded_weights() -> Result<()> {
        let chart = chart(false);
        let guide = guide(raw(0.4), &chart)?;
        let radius: f64 = 3.;
        let volume = PI.powi(3) * radius.powi(6) / 6.;
        let mut rng = StdRng::seed_from_u64(81211);
        let n = 80_000;
        let mut total = 0.;
        let mut squared = 0.;
        let mut outside = 0;
        for _ in 0..n {
            let (u, r, _) = guide.draw(&mut rng, &chart, radius)?;
            let inside = r <= radius;
            let log_q = guide.log_density(u, inside, volume.ln(), &chart, radius);
            assert!(log_q.is_finite());
            let w = if inside {
                (-log_q - volume.ln()).exp()
            } else {
                outside += 1;
                0.
            };
            assert!(w <= 1. / guide.alpha + 1e-12);
            total += w;
            squared += w * w;
        }
        let mean = total / n as f64;
        let se = ((squared / n as f64 - mean * mean) / (n - 1) as f64).sqrt();
        assert!((mean - 1.).abs() < 6. * se);
        assert!(outside > 10_000);
        // Valid-only normalization would falsely increase this integral.
        assert!(total / (n - outside) as f64 > 1.15);
        Ok(())
    }

    #[test]
    fn invalid_schema_target_shell_members_radii_and_mass_fail_closed() {
        let chart = chart(false);
        for (key, value) in [
            ("schema", json!("unknown")),
            ("region_sha256", json!("other")),
            ("defensive_uniform_shell_probability", json!(0.)),
            ("defensive_uniform_shell_probability", json!(1.1)),
            ("entries", json!([])),
            ("unexpected", json!(true)),
        ] {
            let mut data = raw(0.5);
            data[key] = value;
            assert!(guide(data, &chart).is_err());
        }
        for (key, value) in [
            ("weight", json!(0.)),
            ("weight", json!(-2.)),
            ("inner_radius", json!(-1.)),
            ("outer_radius", json!(1.)),
            ("moving_member", json!([0., 0.])),
            ("target_world_member", json!([null, 0., 0.])),
            ("extra", json!(1)),
        ] {
            let mut data = raw(0.5);
            data["entries"][0][key] = value;
            assert!(guide(data, &chart).is_err());
        }
        assert!(
            EntryShellGuide::from_bytes(raw(0.5).to_string().as_bytes(), "target", &chart, 0.1)
                .is_err()
        );
    }
}
