use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
};
use tetramer_mc::{
    math::*,
    native_region::{self, NativeCover, NativeMetric, NativeRegionOptions},
};

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-10 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn metric(c: Vec3) -> NativeMetric {
    NativeMetric {
        native_poses: vec![pose([0.; 3])],
        rigid_members: vec![pose(c)],
        member_error_scale: 2.,
        angle_error_scale_deg: 35.,
    }
}
fn fixture(path: &Path, c: Vec3, capture: f64, blocked: bool) -> Result<PathBuf> {
    fs::create_dir_all(path)?;
    fs::write(path.join("shape.json"),json!({"name":"sphere","volume":4.*PI/3.,"atoms":[{"center":[0.,0.,0.],"radius":if blocked{10.}else{1.}}]}).to_string())?;
    let cfg = json!({"shape":"shape.json","fixed_poses":if blocked{vec![pose([0.;3])]}else{vec![]},
        "capture_center":[0.,0.,0.],"capture_radius":capture,"depletant_radius":0.5,"reservoir_density":0.,
        "poisson_lambda_ratio":64.,"metadata":metric(c)});
    fs::write(path.join("config.json"), cfg.to_string())?;
    Ok(path.join("config.json"))
}
fn options(config: PathBuf, out: PathBuf, n: u64) -> NativeRegionOptions {
    NativeRegionOptions {
        config,
        out,
        samples: n,
        seed: 17280019,
        cloud_replicates: 2,
        activity: None,
        lambda_ratio: None,
        cover_scales: vec![1.],
        cover_weights: Vec::new(),
        model: None,
        model_weight: 0.75,
        model_uniform_probability: 0.05,
        model_anchor_index: 0,
        q_window: Default::default(),
    }
}

#[test]
fn no_obstacles_known_volume_and_rejection_zeros() -> Result<()> {
    let root = std::env::temp_dir().join(format!("native-region-control-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    let cfg = fixture(&root.join("full"), [3., -2., 4.], 100., false)?;
    let full = native_region::run(options(cfg, root.join("full-output"), 4096))?;
    let theta = 35_f64.to_radians();
    let expected = 4. * 2_f64.powi(3) / 3. * (theta - theta.sin());
    near(
        full["native"]["log_normalizer"].as_f64().unwrap().exp(),
        expected,
    );
    assert_eq!(full["native"]["nonzero"], 4096);
    assert_eq!(full["raw_cloud_points"], 0);
    let cfg = fixture(&root.join("clipped"), [0.; 3], 1., false)?;
    let clipped = native_region::run(options(cfg, root.join("clipped-output"), 32768))?;
    let estimated = clipped["native"]["log_normalizer"].as_f64().unwrap().exp();
    let se = expected * (0.125_f64 * 0.875 / 32768.).sqrt();
    assert!((estimated - expected / 8.).abs() < 6. * se);
    assert!(clipped["capture_rejected"].as_u64().unwrap() > 27000);
    let cfg = fixture(&root.join("blocked"), [0.; 3], 100., true)?;
    let zero = native_region::run(options(cfg, root.join("blocked-output"), 512))?;
    assert_eq!(zero["hard_rejected"], 512);
    assert!(zero["native"]["log_normalizer"].is_null());
    assert_eq!(zero["native"]["unconditional_draws"], 512);
    let rows: Vec<Value> = fs::read_to_string(root.join("blocked-output/samples.jsonl"))?
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows.len(), 512);
    assert!(rows.iter().all(|v| v["zero"] == "hard"));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn angular_haar_cdf_and_small_angle_inverse() -> Result<()> {
    for cap in [1e-8, 1e-4, 0.03, 0.2, PI] {
        for u in [0.001, 0.1, 0.5, 0.99] {
            let t = native_region::inverse_angle_cdf(u, cap);
            assert!(
                (native_region::theta_minus_sin(t) / native_region::theta_minus_sin(cap) - u).abs()
                    < 2e-12
            );
        }
    }
    let cover = NativeCover::new(&metric([0.; 3]))?;
    let mut rng = StdRng::seed_from_u64(641002);
    let n = 32768;
    let mut half = 0;
    let mut radial = 0.;
    for _ in 0..n {
        let p = cover.sample(&mut rng);
        assert!(cover.contains(p));
        half += usize::from(
            native_region::relative_angle(p.orientation, cover.reference.orientation)
                < cover.angle_cap / 2.,
        );
        radial += (norm(p.position) / cover.ball_radius).powi(3);
    }
    let t = cover.angle_cap;
    let expected = (t / 2. - (t / 2.).sin()) / (t - t.sin());
    assert!(
        (half as f64 / n as f64 - expected).abs()
            < 6. * (expected * (1. - expected) / n as f64).sqrt()
    );
    assert!((radial / n as f64 - 0.5).abs() < 6. * (1. / (12. * n as f64)).sqrt());
    Ok(())
}

#[test]
fn metric_clipping_centroid_compensation_and_rotated_frame() -> Result<()> {
    let centroid = [3., -2., 4.];
    let mut m = metric(centroid);
    m.member_error_scale = 0.8;
    m.angle_error_scale_deg = 35.;
    m.rigid_members = vec![
        pose(add(centroid, [4., 0., 0.])),
        pose(add(centroid, [-4., 0., 0.])),
        pose(add(centroid, [0., 3., 0.])),
        pose(add(centroid, [0., -3., 0.])),
    ];
    m.native_poses[0] = Pose {
        position: [2., -3., 7.],
        orientation: quaternion(cayley([0.2, -0.3, 0.4])),
    };
    let cover = NativeCover::new(&m)?;
    assert!(cover.angle_cap < cover.nominal_angle_cap);
    let exact_l = 4.5;
    assert!(cover.l_lower <= exact_l && cover.l_lower > 0.);
    let world = Pose {
        position: [-7., 6., 3.],
        orientation: quaternion(cayley([-0.6, 0.4, 0.1])),
    };
    let transform = |p: Pose| Pose {
        position: world.apply(p.position),
        orientation: quaternion(matmul(rotation(world.orientation), rotation(p.orientation))),
    };
    let mut transformed = m.clone();
    transformed.native_poses[0] = transform(m.native_poses[0]);
    let rotated_cover = NativeCover::new(&transformed)?;
    near(cover.volume, rotated_cover.volume);
    let mut loose = cover.clone();
    loose.angle_cap = cover.nominal_angle_cap;
    let mut rng = StdRng::seed_from_u64(873241);
    let (mut inside, mut outside) = (0, 0);
    for _ in 0..32768 {
        let p = loose.sample(&mut rng);
        let q = m.q(p);
        near(q, transformed.q(transform(p)));
        if q <= 1. {
            inside += 1;
            assert!(cover.contains(p));
            assert!(rotated_cover.contains(transform(p)));
        } else {
            outside += 1;
        }
    }
    assert!(inside > 0 && outside > inside);
    let mut shifted = metric([3., -2., 4.]);
    shifted.native_poses = m.native_poses.clone();
    let shifted_cover = NativeCover::new(&shifted)?;
    for _ in 0..1024 {
        let p = shifted_cover.sample(&mut rng);
        assert!(shifted.q(p) <= 1. + 1e-12);
    }
    let mut multiple = m.clone();
    multiple.native_poses.push(m.native_poses[0]);
    assert!(NativeCover::new(&multiple).is_err());
    Ok(())
}
