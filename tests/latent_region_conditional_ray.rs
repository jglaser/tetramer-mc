//! Independent AO-sphere/Haar references for the conditional-ray guide.
use anyhow::Result;
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path, process::Command};
use tetramer_mc::{
    latent_region::{self, LatentRegionOptions},
    simulation::hash_file,
};
const R: f64 = 2.4;
const Z: f64 = 2.;

fn fixture(root: &Path, alpha: f64) -> Result<()> {
    fs::create_dir_all(root)?;
    fs::write(root.join("shape.json"),json!({"name":"sphere","volume":4.*PI*0.3_f64.powi(3)/3.,"atoms":[{"center":[0.,0.,0.],"radius":0.3}]}).to_string())?;
    let fixed = json!({"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let native = json!({"position":[1.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let metadata = json!({"native_poses":[native],"rigid_members":[fixed],"member_error_scale":1.,"angle_error_scale_deg":15.});
    fs::write(root.join("config.json"),json!({"shape":root.join("shape.json"),"fixed_poses":[fixed],"initial_pose":native,
        "capture_center":[0.,0.,0.],"capture_radius":4.,"depletant_radius":0.4,"reservoir_density":Z,
        "poisson_lambda_ratio":64.,"translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":1,"metadata":metadata}).to_string())?;
    let identity: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let shape_hash = hash_file(&root.join("shape.json"))?;
    fs::write(root.join("region.json"),json!({"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],"capture_center":[0.,0.,0.],"capture_radius":4.,
        "shape_sha256":shape_hash,"activity":Z,"depletant_radius":0.4,"physical_metric":metadata,
        "minimum_original_q":0.,"minimum_mahalanobis_radius":0.,"mahalanobis_radius":R,
        "gaussian_chart":{"shape_sha256":shape_hash,"angular_length":1.,"coordinate_convention":"anchor-body-relative",
            "anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[identity],"weights":[1.]}}).to_string())?;
    fs::write(root.join("guide.json"),json!({"schema":"defensive-conditional-ray-guide-v1","region_sha256":hash_file(&root.join("region.json"))?,
        "defensive_uniform_shell_probability":alpha,"inner_radius":0.6,"widths":[0.1,0.8,1.8],
        "interfaces":[{"moving_members":[[0.,0.,0.]],"target_world_members":[[0.,0.,0.]]}]}).to_string())?;
    Ok(())
}
fn reference(z: f64) -> f64 {
    let f = |r: f64| {
        let a = (R * R - r * r).max(0.).sqrt();
        let angular = 0.5 * (a.atan() - a / (1. + a * a));
        let overlap = if r < 1.4 {
            PI * (2.8 + r) * (1.4 - r).powi(2) / 12.
        } else {
            0.
        };
        16. * r * r * angular * (z * overlap).exp()
    };
    let n = 32768;
    let h = (R - 0.6) / n as f64;
    (f(0.6)
        + f(R)
        + (1..n)
            .map(|i| f(0.6 + h * i as f64) * if i % 2 == 0 { 2. } else { 4. })
            .sum::<f64>())
        * h
        / 3.
}
fn check(values: &[f64], expected: f64) {
    let n = values.len() as f64;
    let m = values.iter().sum::<f64>() / n;
    let se = ((values.iter().map(|x| x * x).sum::<f64>() / n - m * m).max(0.) / (n - 1.)).sqrt();
    eprintln!("conditional-ray estimate={m}, reference={expected}, SE={se}");
    assert!((m - expected).abs() < 6.5 * se + 1e-7);
}
#[test]
fn conditional_ray_exact_density_ao_sphere_and_cli() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-conditional-ray-sphere-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fixture(&root, 0.5)?;
    let options = LatentRegionOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        out: root.join("run"),
        samples: 16384,
        seed: 3339107,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    };
    let summary = latent_region::run_with_importance(options.clone(), &root.join("guide.json"))?;
    assert_eq!(
        summary["manifest"]["schema"],
        "importance-latent-region-normalizer-v3"
    );
    assert_eq!(
        summary["manifest"]["guide_schema"],
        "defensive-conditional-ray-guide-v1"
    );
    assert_eq!(
        summary["manifest"]["proposal_kind"],
        "conditional-ray-interval-mixture"
    );
    assert_eq!(summary["shell_rejected"], 0);
    let volume = PI.powi(3) * R.powi(6) / 6.;
    let mut hard = Vec::new();
    let mut weighted = Vec::new();
    let mut volumes = Vec::new();
    let mut zeros = 0;
    let mut guided = 0;
    let mut fallbacks = 0;
    for line in fs::read_to_string(root.join("run/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let u: Vec<f64> = serde_json::from_value(row["latent"].clone())?;
        let t = u[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
        let a2 = u[3..].iter().map(|v| v * v).sum::<f64>();
        let s = (R * R - a2).sqrt();
        let mut factor = 0.;
        for width in [0.1, 0.8, 1.8] {
            let hi = s.min(0.6 + width);
            factor += if hi <= 0.6 {
                1.
            } else if t >= 0.6 && t <= hi {
                s.powi(3) / (hi.powi(3) - 0.6_f64.powi(3))
            } else {
                0.
            };
        }
        let density = (0.5 + 0.5 * factor / 3.) / volume;
        assert!((row["log_proposal_density"].as_f64().unwrap() - density.ln()).abs() < 2e-10);
        assert_eq!(row["shell_valid"], true);
        volumes.push(1. / density);
        if row["proposal_branch"] == "conditional-ray" {
            guided += 1;
            assert_eq!(row["selected_ray_fallback"], s <= 0.6);
            fallbacks += usize::from(s <= 0.6);
        } else {
            assert_eq!(row["proposal_branch"], "uniform-shell");
            assert!(row["selected_ray_fallback"].is_null());
        }
        if row["hard_valid"] == true {
            let jacobian = 1. / (PI.powi(2) * (1. + a2).powi(2));
            let h = jacobian / density;
            assert!((row["log_hard_weight"].as_f64().unwrap() - h.ln()).abs() < 2e-10);
            let clouds = row["clouds"].as_array().unwrap();
            assert_eq!(clouds.len(), 2);
            let w = clouds
                .iter()
                .map(|c| c["log_weight"].as_f64().unwrap().exp())
                .sum::<f64>()
                / 2.;
            assert!((row["log_importance_weight"].as_f64().unwrap() - (h * w).ln()).abs() < 2e-10);
            hard.push(h);
            weighted.push(h * w);
        } else {
            zeros += 1;
            assert!(row["clouds"].as_array().unwrap().is_empty());
            assert!(row["log_hard_weight"].is_null());
            assert!(row["log_importance_weight"].is_null());
            hard.push(0.);
            weighted.push(0.);
        }
    }
    assert!(guided > 7000 && fallbacks > 0 && zeros > 100);
    check(&volumes, volume);
    check(&hard, reference(0.));
    check(&weighted, reference(Z));
    let mut short = options;
    short.samples = 64;
    short.out = root.join("short-api");
    latent_region::run_with_importance(short, &root.join("guide.json"))?;
    let output = Command::new(env!("CARGO_BIN_EXE_latent-region-normalizer"))
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
            "3339107",
            "--importance-guide",
            root.join("guide.json").to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(root.join("short-api/samples.jsonl"))?,
        fs::read(root.join("short-cli/samples.jsonl"))?
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn conditional_ray_alpha_one_preserves_all_physical_rows() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-conditional-ray-uniform-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fixture(&root, 1.)?;
    let mut options = LatentRegionOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        out: root.join("uniform"),
        samples: 128,
        seed: 3339108,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    };
    latent_region::run(options.clone())?;
    options.out = root.join("guided");
    latent_region::run_with_importance(options, &root.join("guide.json"))?;
    let old = fs::read_to_string(root.join("uniform/samples.jsonl"))?;
    let new = fs::read_to_string(root.join("guided/samples.jsonl"))?;
    for (a, b) in old.lines().zip(new.lines()) {
        let a: Value = serde_json::from_str(a)?;
        let mut b: Value = serde_json::from_str(b)?;
        for key in [
            "shell_valid",
            "log_proposal_density",
            "proposal_branch",
            "proposal_component",
            "selected_ray_fallback",
        ] {
            b.as_object_mut().unwrap().remove(key);
        }
        assert_eq!(a, b);
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
