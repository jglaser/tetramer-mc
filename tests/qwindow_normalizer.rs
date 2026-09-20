use anyhow::Result;
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path};
use tetramer_mc::{
    math::*,
    native_region::{self, NativeCover, NativeMetric, NativeRegionOptions, QWindow},
    simulation::hash_file,
};

const CENTER: Vec3 = [2., -3., 4.];
const DELTA: f64 = 0.8;
const ALPHA: f64 = 110.;
const CAPTURE: f64 = 1.4;
const ELL: f64 = 1.1;
const BETA: f64 = 0.65;
const EPSILON: f64 = 0.1;
const RHO: f64 = 0.3;

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn window() -> QWindow {
    QWindow {
        minimum: 1.,
        maximum: 2.,
        lower_inclusive: false,
        upper_inclusive: false,
    }
}
fn metric(centroid: Vec3) -> NativeMetric {
    NativeMetric {
        native_poses: vec![pose(CENTER)],
        rigid_members: vec![pose(centroid)],
        member_error_scale: DELTA,
        angle_error_scale_deg: ALPHA,
    }
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn options(root: &Path, output: &str, samples: u64) -> NativeRegionOptions {
    NativeRegionOptions {
        config: root.join("config.json"),
        out: root.join(output),
        samples,
        seed: 99410319,
        cloud_replicates: 2,
        activity: None,
        lambda_ratio: None,
        cover_scales: vec![1.],
        cover_weights: vec![],
        model: None,
        model_weight: BETA,
        model_uniform_probability: EPSILON,
        model_anchor_index: 0,
        q_window: window(),
    }
}
fn write_config(root: &Path, m: &NativeMetric, fixed: &[Pose], capture: f64, z: f64) -> Result<()> {
    fs::create_dir_all(root)?;
    fs::write(
        root.join("shape.json"),
        json!({"atoms":[{"center":[0.,0.,0.],"radius":0.2}],"volume":4.*PI*0.2_f64.powi(3)/3.})
            .to_string(),
    )?;
    fs::write(root.join("config.json"), json!({"shape":"shape.json","fixed_poses":fixed,
        "capture_center":CENTER,"capture_radius":capture,"depletant_radius":0.35,"reservoir_density":z,
        "poisson_lambda_ratio":64.,"endpoint_gate":{"max_cells":31,"max_depth":10,"min_width":0.05},
        "metadata":m}).to_string())?;
    Ok(())
}

#[test]
fn original_window_endpoints_and_nonlinear_cover_bounds() -> Result<()> {
    let w = window();
    w.validate()?;
    assert!(!w.contains(1.) && w.contains(1.5) && !w.contains(2.) && !w.contains(f64::NAN));
    assert!(QWindow::default().contains(0.) && QWindow::default().contains(1.));
    for (minimum, maximum) in [(1., 1.), (-1., 2.), (0., f64::INFINITY), (f64::NAN, 2.)] {
        assert!(
            QWindow {
                minimum,
                maximum,
                ..w
            }
            .validate()
            .is_err()
        );
    }
    let original = metric([3., -2., 1.]);
    let scaled = w.cover_metric(&original)?;
    assert_eq!(original.member_error_scale, DELTA);
    assert_eq!(original.angle_error_scale_deg, ALPHA);
    assert_eq!(scaled.member_error_scale, 2. * DELTA);
    assert_eq!(scaled.angle_error_scale_deg, 180.);
    near(NativeCover::new(&scaled)?.angle_cap, PI);
    let mut multi = original.clone();
    multi.member_error_scale = 1.;
    multi.angle_error_scale_deg = 35.;
    multi.rigid_members = vec![
        pose([10., 0., 0.]),
        pose([-10., 0., 0.]),
        pose([0., 10., 0.]),
        pose([0., -10., 0.]),
        pose([0., 0., 10.]),
        pose([0., 0., -10.]),
    ];
    let small = NativeCover::new(&multi)?;
    let big = NativeCover::new(&w.cover_metric(&multi)?)?;
    assert!(
        big.angle_cap > 2. * small.angle_cap,
        "arcsin cap must be recomputed"
    );
    near(big.angle_cap, 2. * (2. / (2. * big.l_lower.sqrt())).asin());
    Ok(())
}

#[test]
fn shifted_centroid_window_has_independent_exact_volume() -> Result<()> {
    let root = std::env::temp_dir().join(format!("qwindow-centroid-{}", std::process::id()));
    write_config(&root, &metric([3., -2., 1.]), &[], 100., 0.)?;
    let result = native_region::run(options(&root, "run", 32768))?;
    let theta = ALPHA.to_radians();
    let outer = 4. * PI * (2. * DELTA).powi(3) / 3.;
    let inner = 4. * DELTA.powi(3) / 3. * (theta - theta.sin());
    let expected = outer - inner;
    let estimate = result["region"]["log_normalizer"].as_f64().unwrap().exp();
    let se = (inner / outer * (1. - inner / outer) / 32768.).sqrt() * outer;
    assert!((estimate - expected).abs() < 6. * se);
    assert!(result["native"].is_null());
    assert!(result["q_rejected"].as_u64().unwrap() > 0);
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("run/manifest.json"))?)?;
    assert_eq!(manifest["schema"], 4);
    assert_eq!(manifest["metric"]["angle_error_scale_deg"], ALPHA);
    assert_eq!(manifest["cover_metric"]["angle_error_scale_deg"], 180.);
    for line in fs::read_to_string(root.join("run/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let p: Pose = serde_json::from_value(row["pose"].clone())?;
        near(row["q"].as_f64().unwrap(), metric([3., -2., 1.]).q(p));
        near(row["log_proposal_density"].as_f64().unwrap(), -outer.ln());
        assert_eq!(row["proposal_family"], "cover");
        assert_eq!(row["proposal_component"], 0);
        assert_eq!(
            row.get("zero").is_none(),
            window().contains(row["q"].as_f64().unwrap())
        );
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

fn reference(z: f64) -> f64 {
    let lower_haar = (ALPHA.to_radians() - ALPHA.to_radians().sin()) / PI;
    let intervals = [
        (0.4, DELTA, 1. - lower_haar),
        (DELTA, 1.1, 1.),
        (1.1, CAPTURE, 1.),
    ];
    intervals
        .into_iter()
        .map(|(lo, hi, haar)| {
            let n = 2048;
            let h = (hi - lo) / n as f64;
            let f = |r: f64| {
                let overlap = if r < 1.1 {
                    PI * (2.2 + r) * (1.1 - r).powi(2) / 12.
                } else {
                    0.
                };
                4. * PI * haar * r * r * (z * overlap).exp()
            };
            (f(lo)
                + f(hi)
                + (1..n)
                    .map(|i| (if i % 2 == 0 { 2. } else { 4. }) * f(lo + i as f64 * h))
                    .sum::<f64>())
                * h
                / 3.
        })
        .sum()
}

#[test]
fn guided_sphere_window_retains_second_neighbor_and_full_density() -> Result<()> {
    let root = std::env::temp_dir().join(format!("qwindow-sphere-{}", std::process::id()));
    let anchor = pose(add(CENTER, [20., 0., 0.]));
    write_config(
        &root,
        &metric([0.; 3]),
        &[anchor, pose(CENTER)],
        CAPTURE,
        8.,
    )?;
    let mut covariance = [[0.; 6]; 6];
    for (i, row) in covariance.iter_mut().enumerate() {
        row[i] = 1.;
    }
    covariance[0][3] = RHO;
    covariance[3][0] = RHO;
    fs::write(root.join("model.json"),json!({"schema":"weighted-pose-mixture-v1",
        "shape_sha256":hash_file(&root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative",
        "angular_length":ELL,"anchors":[{"position":[-20.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]}).to_string())?;
    let mut o = options(&root, "run", 65536);
    o.model = Some(root.join("model.json"));
    let result = native_region::run(o)?;
    for (key, z) in [("region", 8.), ("hard_region", 0.)] {
        let v = &result[key];
        let mean = v["log_normalizer"].as_f64().unwrap().exp();
        let se = mean * v["observed_relative_standard_error"].as_f64().unwrap();
        let expected = reference(z);
        assert!(
            (mean - expected).abs() < 6.5 * se + 1e-9,
            "{key}: {mean} versus {expected}, SE {se}"
        );
    }
    for key in ["q_rejected", "capture_rejected", "hard_rejected"] {
        assert!(result[key].as_u64().unwrap() > 0);
    }
    let outer = 4. * PI * (2. * DELTA).powi(3) / 3.;
    for line in fs::read_to_string(root.join("run/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let p: Pose = serde_json::from_value(row["pose"].clone())?;
        let t = sub(p.position, CENTER);
        let c = [
            p.orientation[1] / p.orientation[0],
            p.orientation[2] / p.orientation[0],
            p.orientation[3] / p.orientation[0],
        ];
        let x = [t[0], t[1], t[2], ELL * c[0], ELL * c[1], ELL * c[2]];
        let squared = (x[0] * x[0] - 2. * RHO * x[0] * x[3] + x[3] * x[3]) / (1. - RHO * RHO)
            + x[1] * x[1]
            + x[2] * x[2]
            + x[4] * x[4]
            + x[5] * x[5];
        let normal = (-3. * (2. * PI).ln() - 0.5 * (1. - RHO * RHO).ln() - 0.5 * squared).exp();
        let gaussian = normal * ELL.powi(3) * PI.powi(2) * (1. + dot(c, c)).powi(2);
        let uniform = if t.iter().all(|v| *v >= -CAPTURE && *v < CAPTURE) {
            (2. * CAPTURE).powi(-3)
        } else {
            0.
        };
        let cover = if norm(t) <= 2. * DELTA {
            1. / outer
        } else {
            0.
        };
        let density = (1. - BETA) * cover + BETA * ((1. - EPSILON) * gaussian + EPSILON * uniform);
        near(row["log_proposal_density"].as_f64().unwrap(), density.ln());
        let q = (norm(t) / DELTA)
            .max(2. * p.orientation[0].abs().clamp(0., 1.).acos() / ALPHA.to_radians());
        near(q, row["q"].as_f64().unwrap());
        let valid = window().contains(q) && norm(t) <= CAPTURE && norm(t) >= 0.4;
        assert_eq!(row.get("zero").is_none(), valid);
        if valid {
            near(row["log_hard_weight"].as_f64().unwrap(), -density.ln());
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
