//! AO sphere overlap and Haar-volume references for a defensive latent mixture.
use anyhow::Result;
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path, process::Command};
use tetramer_mc::{
    latent_region::{self, LatentRegionOptions},
    simulation::hash_file,
};

const INNER: f64 = 1.2;
const OUTER: f64 = 2.4;
const ACTIVITY: f64 = 2.;

fn fixture(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    fs::write(root.join("shape.json"),json!({"name":"sphere","volume":4.*PI*0.3_f64.powi(3)/3.,"atoms":[{"center":[0.,0.,0.],"radius":0.3}]}).to_string())?;
    let fixed = json!({"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let native = json!({"position":[1.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let metadata = json!({"native_poses":[native],"rigid_members":[fixed],"member_error_scale":1.,"angle_error_scale_deg":15.});
    let cfg = json!({"shape":root.join("shape.json"),"fixed_poses":[fixed],"initial_pose":native,
        "capture_center":[0.,0.,0.],"capture_radius":4.,"depletant_radius":0.4,"reservoir_density":ACTIVITY,
        "poisson_lambda_ratio":64.,"translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":1,"metadata":metadata});
    fs::write(root.join("config.json"), cfg.to_string())?;
    let identity: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let shape_hash = hash_file(&root.join("shape.json"))?;
    let region = json!({"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],"capture_center":[0.,0.,0.],"capture_radius":4.,
        "shape_sha256":shape_hash,"activity":ACTIVITY,"depletant_radius":0.4,"physical_metric":metadata,
        "minimum_original_q":0.,"minimum_mahalanobis_radius":INNER,"mahalanobis_radius":OUTER,
        "gaussian_chart":{"shape_sha256":shape_hash,"angular_length":1.,"coordinate_convention":"anchor-body-relative",
            "anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[identity],"weights":[1.]}});
    fs::write(root.join("region.json"), region.to_string())?;
    let covariance: Vec<Vec<f64>> = identity
        .iter()
        .map(|r| r.iter().map(|x| x * 4.).collect())
        .collect();
    let guide = json!({"schema":"defensive-latent-shell-guide-v1","region_sha256":hash_file(&root.join("region.json"))?,
        "defensive_uniform_shell_probability":0.5,"gaussian_components":[{"weight":1.,"mean":[0.,0.,0.,0.,0.,0.],"covariance":covariance}]});
    fs::write(root.join("guide.json"), guide.to_string())?;
    Ok(())
}

fn reference(z: f64) -> f64 {
    // Independent 1D quadrature: integrate SO(3)'s radial Cayley coordinate
    // analytically for each physical center separation, then integrate dr.
    let f = |t: f64| {
        let hi = (OUTER * OUTER - t * t).max(0.).sqrt();
        let lo = (INNER * INNER - t * t).max(0.).sqrt();
        let primitive = |a: f64| 0.5 * (a.atan() - a / (1. + a * a));
        let overlap = if t < 1.4 {
            PI * (2.8 + t) * (1.4 - t).powi(2) / 12.
        } else {
            0.
        };
        16. * t * t * (primitive(hi) - primitive(lo)) * (z * overlap).exp()
    };
    let n = 32768;
    let step = (OUTER - 0.6) / n as f64;
    (f(0.6)
        + f(OUTER)
        + (1..n)
            .map(|i| {
                if i % 2 == 0 {
                    2. * f(0.6 + step * i as f64)
                } else {
                    4. * f(0.6 + step * i as f64)
                }
            })
            .sum::<f64>())
        * step
        / 3.
}

fn check(values: &[f64], want: f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se = ((values.iter().map(|v| v * v).sum::<f64>() / n - mean * mean) / (n - 1.)).sqrt();
    eprintln!("importance estimate {mean}, reference {want}, SE {se}");
    assert!((mean - want).abs() < 6.5 * se + 1e-7);
}

#[test]
fn importance_ao_reference_haar_jacobian_and_outside_zeros() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("tetramer-latent-importance-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fixture(&root)?;
    let options = LatentRegionOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        out: root.join("run"),
        samples: 32768,
        seed: 987651,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    };
    let summary = latent_region::run_with_importance(options.clone(), &root.join("guide.json"))?;
    assert_eq!(
        summary["manifest"]["schema"],
        "importance-latent-region-normalizer-v1"
    );
    assert_eq!(
        summary["manifest"]["importance_guide_sha256"],
        hash_file(&root.join("guide.json"))?
    );
    assert_eq!(
        fs::read(root.join("guide.json"))?,
        fs::read(root.join("run/provenance/importance-guide.json"))?
    );
    let volume = PI.powi(3) * (OUTER.powi(6) - INNER.powi(6)) / 6.;
    let mut mass = Vec::new();
    let mut hard = Vec::new();
    let mut latent_volume = Vec::new();
    let mut outside = 0;
    let mut hard_zeros = 0;
    for line in fs::read_to_string(root.join("run/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let u: Vec<f64> = serde_json::from_value(row["latent"].clone())?;
        let radius = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        let inside = radius > INNER && radius <= OUTER;
        assert_eq!(row["shell_valid"], inside);
        let gaussian = (-3. * (8. * PI).ln() - radius * radius / 8.).exp();
        let density = 0.5 * gaussian + if inside { 0.5 / volume } else { 0. };
        assert!((row["log_proposal_density"].as_f64().unwrap() - density.ln()).abs() < 1e-11);
        let angular2 = u[3..].iter().map(|x| x * x).sum::<f64>();
        let jacobian = 1. / (PI.powi(2) * (1. + angular2).powi(2));
        assert!((row["log_physical_jacobian"].as_f64().unwrap() - jacobian.ln()).abs() < 1e-11);
        latent_volume.push(if inside { 1. / density } else { 0. });
        let valid = inside
            && row["hard_valid"].as_bool().unwrap()
            && row["region_valid"].as_bool().unwrap();
        if valid {
            let h = jacobian / density;
            assert!((row["log_hard_weight"].as_f64().unwrap() - h.ln()).abs() < 1e-11);
            let clouds = row["clouds"].as_array().unwrap();
            assert_eq!(clouds.len(), 2);
            let w = clouds
                .iter()
                .map(|c| {
                    let expected = ACTIVITY * c["lower_volume"].as_f64().unwrap()
                        + c["overlap_points"].as_u64().unwrap() as f64 * (1. + 1_f64 / 64.).ln();
                    assert!((c["log_weight"].as_f64().unwrap() - expected).abs() < 1e-11);
                    expected.exp()
                })
                .sum::<f64>()
                / 2.;
            assert!((row["log_importance_weight"].as_f64().unwrap() - (h * w).ln()).abs() < 1e-11);
            mass.push(h * w);
            hard.push(h);
        } else {
            assert!(
                row["log_importance_weight"].is_null()
                    && row["log_hard_weight"].is_null()
                    && row["clouds"].as_array().unwrap().is_empty()
            );
            outside += usize::from(!inside);
            hard_zeros += usize::from(inside && !row["hard_valid"].as_bool().unwrap());
            mass.push(0.);
            hard.push(0.);
        }
    }
    assert_eq!(mass.len(), 32768);
    assert!(outside > 10000 && hard_zeros > 100);
    check(&mass, reference(ACTIVITY));
    check(&hard, reference(0.));
    check(&latent_volume, volume);

    // CLI uses the same API and fresh deterministic streams, including zero rows.
    let mut short = options;
    short.samples = 64;
    short.out = root.join("short-api");
    latent_region::run_with_importance(short, &root.join("guide.json"))?;
    let result = Command::new(env!("CARGO_BIN_EXE_latent-region-normalizer"))
        .args([
            "--config",
            root.join("config.json").to_str().unwrap(),
            "--region",
            root.join("region.json").to_str().unwrap(),
            "--out",
            root.join("short-cli").to_str().unwrap(),
            "--samples",
            "64",
            "--seed",
            "987651",
            "--importance-guide",
            root.join("guide.json").to_str().unwrap(),
        ])
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(root.join("short-api/samples.jsonl"))?,
        fs::read(root.join("short-cli/samples.jsonl"))?
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn importance_uniform_physical_stream_and_all_zero_target() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-latent-importance-controls-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fixture(&root)?;
    let guide = json!({"schema":"defensive-latent-shell-guide-v1","region_sha256":hash_file(&root.join("region.json"))?,
        "defensive_uniform_shell_probability":1.,"gaussian_components":[]});
    fs::write(root.join("uniform-guide.json"), guide.to_string())?;
    let mut options = LatentRegionOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        out: root.join("uniform"),
        samples: 128,
        seed: 733815,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    };
    latent_region::run(options.clone())?;
    options.out = root.join("guided-uniform");
    latent_region::run_with_importance(options.clone(), &root.join("uniform-guide.json"))?;
    let old = fs::read_to_string(root.join("uniform/samples.jsonl"))?;
    let new = fs::read_to_string(root.join("guided-uniform/samples.jsonl"))?;
    for (a, b) in old.lines().zip(new.lines()) {
        let old: Value = serde_json::from_str(a)?;
        let mut new: Value = serde_json::from_str(b)?;
        for field in [
            "shell_valid",
            "log_proposal_density",
            "proposal_branch",
            "proposal_component",
        ] {
            new.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(
            old, new,
            "Alpha1 must preserve all physical draws and cloud factors"
        );
    }
    let mut cfg: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
    let mut region: Value = serde_json::from_slice(&fs::read(root.join("region.json"))?)?;
    cfg["capture_radius"] = json!(0.001);
    region["capture_radius"] = json!(0.001);
    fs::write(root.join("config.json"), cfg.to_string())?;
    fs::write(root.join("region.json"), region.to_string())?;
    let mut guide: Value = serde_json::from_slice(&fs::read(root.join("guide.json"))?)?;
    guide["region_sha256"] = json!(hash_file(&root.join("region.json"))?);
    fs::write(root.join("guide.json"), guide.to_string())?;
    options.out = root.join("zero");
    let summary = latent_region::run_with_importance(options, &root.join("guide.json"))?;
    assert!(summary["estimates"]["region"]["logQ"].is_null());
    assert_eq!(summary["estimates"]["region"]["nonzero"], 0);
    let rows = fs::read_to_string(root.join("zero/samples.jsonl"))?;
    assert_eq!(rows.lines().count(), 128);
    for line in rows.lines() {
        let row: Value = serde_json::from_str(line)?;
        assert!(
            row["log_importance_weight"].is_null() && row["clouds"].as_array().unwrap().is_empty()
        );
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
