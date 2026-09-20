use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{f64::consts::PI, fs};
use tetramer_mc::{
    math::*,
    native_region::{self, NativeCover, NativeCoverMixture, NativeMetric, NativeRegionOptions},
};

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn metric() -> NativeMetric {
    NativeMetric {
        native_poses: vec![pose([0.; 3])],
        rigid_members: vec![pose([0.; 3])],
        member_error_scale: 2.,
        angle_error_scale_deg: 35.,
    }
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn axis_pose(theta: f64) -> Pose {
    Pose {
        position: [0.; 3],
        orientation: [(theta / 2.).cos(), (theta / 2.).sin(), 0., 0.],
    }
}

#[test]
fn nominal_support_resolves_boundaries_and_tiny_angles() -> Result<()> {
    let cover = NativeCover::new(&metric())?;
    let outside = pose([cover.ball_radius * (1. + 5e-13), 0., 0.]);
    assert!(cover.contains(outside));
    assert!(!cover.contains_support(outside));
    let outside = axis_pose(cover.angle_cap + 5e-13);
    assert!(cover.contains(outside));
    assert!(!cover.contains_support(outside));
    let mut tiny_metric = metric();
    tiny_metric.angle_error_scale_deg = 1e-8_f64.to_degrees();
    let tiny = NativeCover::new(&tiny_metric)?;
    for (factor, expected) in [(0.99, true), (1.01, false)] {
        let mut p = axis_pose(factor * tiny.angle_cap);
        assert_eq!(tiny.contains_support(p), expected);
        p.orientation = p.orientation.map(|v| -v);
        assert_eq!(tiny.contains_support(p), expected);
    }
    Ok(())
}

#[test]
fn full_density_has_exact_shell_normalization_and_legacy_replay() -> Result<()> {
    let cover = NativeCover::new(&metric())?;
    let mixture = NativeCoverMixture::new(&cover, vec![0.1, 0.2, 0.4, 1.], vec![1., 2., 3., 4.])?;
    let mut probability = 0.;
    let mut previous_volume = 0.;
    let mut previous_radius = 0.;
    for (j, c) in mixture.covers.iter().enumerate() {
        let expected: f64 = (j..mixture.covers.len())
            .map(|i| mixture.weights[i] / mixture.covers[i].volume)
            .sum();
        let p = pose([(c.ball_radius + previous_radius) / 2., 0., 0.]);
        near(mixture.log_density(p), expected.ln());
        probability += (c.volume - previous_volume) * expected;
        previous_volume = c.volume;
        previous_radius = c.ball_radius;
    }
    near(probability, 1.);
    assert_eq!(mixture.log_density(pose([3., 0., 0.])), f64::NEG_INFINITY);
    for (scales, weights) in [
        (vec![0.1, 0.4], vec![]),
        (vec![0., 1.], vec![]),
        (vec![1.1, 1.], vec![]),
        (vec![f64::NAN, 1.], vec![]),
        (vec![0.1, 1.], vec![1.]),
        (vec![0.1, 1.], vec![-1., 1.]),
        (vec![0.1, 1.], vec![0., 1.]),
    ] {
        assert!(NativeCoverMixture::new(&cover, scales, weights).is_err());
    }
    let legacy = NativeCoverMixture::new(&cover, vec![1.], vec![])?;
    let mut old_rng = StdRng::seed_from_u64(611902);
    let mut new_rng = StdRng::seed_from_u64(611902);
    let mut components = StdRng::seed_from_u64(311999);
    let mut unused_components = StdRng::seed_from_u64(311999);
    for _ in 0..1024 {
        let old = cover.sample(&mut old_rng);
        let (index, new) = legacy.sample(&mut new_rng, &mut components);
        assert_eq!(index, 0);
        assert_eq!(old, new);
        near(legacy.log_density(new), -cover.volume.ln());
    }
    assert_eq!(
        components.random::<u64>(),
        unused_components.random::<u64>()
    );
    Ok(())
}

#[test]
fn independent_weighted_moments_and_frame_invariance() -> Result<()> {
    let mut m = metric();
    m.rigid_members[0].position = [3., -2., 4.];
    m.native_poses[0] = Pose {
        position: [5., 1., -7.],
        orientation: quaternion(cayley([0.3, -0.2, 0.5])),
    };
    let cover = NativeCover::new(&m)?;
    let scales = vec![0.1, 0.2, 0.4, 1.];
    let mixture = NativeCoverMixture::new(&cover, scales.clone(), vec![])?;
    let world = Pose {
        position: [-9., 4., 2.],
        orientation: quaternion(cayley([-0.2, 0.6, 0.1])),
    };
    let transform = |p: Pose| Pose {
        position: world.apply(p.position),
        orientation: quaternion(matmul(rotation(world.orientation), rotation(p.orientation))),
    };
    let mut transformed_metric = m.clone();
    transformed_metric.native_poses[0] = transform(m.native_poses[0]);
    let transformed =
        NativeCoverMixture::new(&NativeCover::new(&transformed_metric)?, scales, vec![])?;
    let theta = cover.angle_cap;
    let r2_mean = 3. * cover.ball_radius.powi(2) / 5.;
    let cos_mean = (theta.sin() - theta / 2. - (2. * theta).sin() / 4.)
        / native_region::theta_minus_sin(theta);
    let mut sums = [0.; 4];
    let mut squares = [0.; 4];
    let mut counts = [0; 4];
    let mut rng = StdRng::seed_from_u64(98650192);
    let mut component_rng = StdRng::seed_from_u64(98670102);
    let n = 65536;
    for _ in 0..n {
        let (component, p) = mixture.sample(&mut rng, &mut component_rng);
        counts[component] += 1;
        assert!(mixture.covers[component].contains_support(p));
        assert!(m.q(p) <= 1. + 1e-12);
        let log_g = mixture.log_density(p);
        near(log_g, transformed.log_density(transform(p)));
        let w = add(
            sub(p.position, cover.reference.position),
            sub(
                matvec(rotation(p.orientation), cover.centroid),
                matvec(rotation(cover.reference.orientation), cover.centroid),
            ),
        );
        let r2 = dot(w, w);
        let cos = native_region::relative_angle(p.orientation, cover.reference.orientation).cos();
        // Centered observables check that the proposal's coupled scale label
        // disappears after importance weighting, including the joint moment.
        let values = [
            1.,
            r2 - r2_mean,
            cos - cos_mean,
            (r2 - r2_mean) * (cos - cos_mean),
        ];
        for (j, f) in values.iter().enumerate() {
            let y = (-log_g).exp() * f;
            sums[j] += y;
            squares[j] += y * y;
        }
    }
    let mut prev = 0.;
    let mut second_moment = 0.;
    for j in 0..4 {
        let v = mixture.covers[j].volume;
        let g: f64 = (j..4)
            .map(|i| mixture.weights[i] / mixture.covers[i].volume)
            .sum();
        second_moment += (v - prev) / g;
        prev = v;
    }
    let exact_se = ((second_moment - cover.volume.powi(2)) / n as f64).sqrt();
    assert!((sums[0] / n as f64 - cover.volume).abs() < 6. * exact_se);
    for j in 1..4 {
        let mean = sums[j] / n as f64;
        let se = ((squares[j] / n as f64 - mean * mean) / (n - 1) as f64).sqrt();
        assert!(mean.abs() < 6. * se, "observable {j}: {mean} / {se}");
    }
    for count in counts {
        assert!((count as f64 - n as f64 / 4.).abs() < 6. * (n as f64 * 0.25 * 0.75).sqrt());
    }
    Ok(())
}

#[test]
fn sphere_depletion_and_capture_have_independent_integral_references() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "native-cover-mixture-integrals-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let mut m = metric();
    m.member_error_scale = 4.;
    m.angle_error_scale_deg = 40.;
    for (name, fixed, capture, activity) in [
        ("capture", vec![], 1., 0.),
        ("ao", vec![pose([0.; 3])], 5., 0.5),
    ] {
        let config = root.join(format!("{name}.json"));
        fs::write(&config, json!({"shape":"shape.json","fixed_poses":fixed,"capture_center":[0.,0.,0.],"capture_radius":capture,"depletant_radius":0.5,"reservoir_density":activity,"poisson_lambda_ratio":16.,"endpoint_gate":{"max_cells":1,"max_depth":14,"min_width":0.5},"metadata":m}).to_string())?;
        let result = native_region::run(NativeRegionOptions {
            config,
            out: root.join(format!("{name}-output")),
            samples: 32768,
            seed: 98680021,
            cloud_replicates: 2,
            activity: None,
            lambda_ratio: None,
            cover_scales: vec![0.1, 0.2, 0.4, 1.],
            cover_weights: vec![],
        })?;
        let haar = native_region::theta_minus_sin(m.angle_error_scale_deg.to_radians()) / PI;
        let hard_expected = if activity == 0. {
            haar * 4. * PI / 3.
        } else {
            haar * 4. * PI / 3. * (64. - 8.)
        };
        let expected = if activity == 0. {
            hard_expected
        } else {
            // Independent radial AO integral: exclusion radius R=1.5,
            // hard center distance >=2, finite translation cover r<=4.
            let f = |r: f64| {
                let overlap = if r < 3. {
                    PI * (6. + r) * (3. - r).powi(2) / 12.
                } else {
                    0.
                };
                4. * PI * r * r * (activity * overlap).exp()
            };
            let n = 4096;
            let h = 2. / n as f64;
            let integral = (f(2.)
                + f(4.)
                + (1..n)
                    .map(|i| (if i % 2 == 0 { 2. } else { 4. }) * f(2. + h * i as f64))
                    .sum::<f64>())
                * h
                / 3.;
            haar * integral
        };
        for (label, target) in [("native", expected), ("hard_native", hard_expected)] {
            let estimate = result[label]["log_normalizer"].as_f64().unwrap().exp();
            let se = estimate
                * result[label]["observed_relative_standard_error"]
                    .as_f64()
                    .unwrap();
            assert!(
                (estimate - target).abs() < 6. * se,
                "{name}/{label}: {estimate}, target {target}, SE {se}"
            );
        }
        assert!(
            result[if activity == 0. {
                "capture_rejected"
            } else {
                "hard_rejected"
            }]
            .as_u64()
            .unwrap()
                > 1000
        );
        assert_eq!(result["native"]["unconditional_draws"], 32768);
        if activity > 0. {
            assert!(result["raw_cloud_points"].as_u64().unwrap() > 0);
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
