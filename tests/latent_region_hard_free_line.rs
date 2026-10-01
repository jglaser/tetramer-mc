//! Production integration: exact physical sphere reference, controls and journals.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tetramer_mc::{
    latent_region::{self, LatentRegionOptions},
    simulation::hash_file,
};

const OUTER: f64 = 2.4;
const SAMPLES: u64 = 8192;

fn root(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("TETRAMER_HARD_FREE_REFERENCE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!(
                "tetramer-hard-free-production-{}",
                std::process::id()
            ))
        });
    let path = base.join(name);
    anyhow::ensure!(
        !path.exists(),
        "Reference outputs already exist; no overwrite"
    );
    fs::create_dir_all(&path)?;
    Ok(path)
}
fn save(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn read(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn rows(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn fixture(path: &Path, activity: f64, alpha: f64, beta: f64) -> Result<()> {
    let shape = json!({"name":"sphere","volume":4.*PI*0.3_f64.powi(3)/3.,"atoms":[{"center":[0.,0.,0.],"radius":0.3}]});
    save(path.join("shape.json"), &shape)?;
    let fixed = json!({"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let native = json!({"position":[1.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let metric = json!({"native_poses":[native],"rigid_members":[fixed],"member_error_scale":1.,"angle_error_scale_deg":15.});
    let config = json!({"shape":path.join("shape.json"),"fixed_poses":[fixed],"initial_pose":native,
        "capture_center":[0.,0.,0.],"capture_radius":4.,"depletant_radius":0.4,"reservoir_density":activity,
        "poisson_lambda_ratio":64.,"translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":1,"metadata":metric});
    save(path.join("config.json"), &config)?;
    let identity: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let hash = hash_file(&path.join("shape.json"))?;
    let region = json!({"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],"capture_center":[0.,0.,0.],"capture_radius":4.,
        "shape_sha256":hash,"activity":activity,"depletant_radius":0.4,"physical_metric":metric,
        "minimum_original_q":0.,"minimum_mahalanobis_radius":0.,"mahalanobis_radius":OUTER,
        "gaussian_chart":{"shape_sha256":hash,"angular_length":1.,"coordinate_convention":"anchor-body-relative",
        "anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[identity],"weights":[1.]}});
    save(path.join("region.json"), &region)?;
    let components: Vec<Value> = [(0.35, 0., 4.), (0.65, 0.8, 0.6)]
        .into_iter()
        .map(|(weight, mean, variance)| {
            let covariance: Vec<Vec<f64>> = identity
                .iter()
                .map(|r| r.iter().map(|x| x * variance).collect())
                .collect();
            json!({"weight":weight,"mean":[mean,0.,0.,0.,0.,0.],"covariance":covariance})
        })
        .collect();
    let guide = json!({"schema":"defensive-hard-free-line-guide-v1","region_sha256":hash_file(&path.join("region.json"))?,
        "defensive_uniform_shell_probability":alpha,"gaussian_components":components,
        "conditional_probability":beta,"minimum_conditional_mass":1e-12,"raw_translation_axes":[0,1,2]});
    save(path.join("guide.json"), &guide)?;
    let mut old = guide;
    old["schema"] = json!("defensive-latent-shell-guide-v1");
    for key in [
        "conditional_probability",
        "minimum_conditional_mass",
        "raw_translation_axes",
    ] {
        old.as_object_mut().unwrap().remove(key);
    }
    save(path.join("old-guide.json"), &old)?;
    Ok(())
}
fn options(path: &Path, out: &str, samples: u64, seed: u64) -> LatentRegionOptions {
    LatentRegionOptions {
        config: path.join("config.json"),
        region: path.join("region.json"),
        out: path.join(out),
        samples,
        seed,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    }
}
fn reference(z: f64) -> f64 {
    let integrand = |r: f64| {
        let a = (OUTER * OUTER - r * r).max(0.).sqrt();
        let angular = 0.5 * (a.atan() - a / (1. + a * a));
        let overlap = if r < 1.4 {
            PI * (2.8 + r) * (1.4 - r).powi(2) / 12.
        } else {
            0.
        };
        16. * r * r * angular * (z * overlap).exp()
    };
    let n = 32768;
    let h = (OUTER - 0.6) / n as f64;
    (integrand(0.6)
        + integrand(OUTER)
        + (1..n)
            .map(|i| integrand(0.6 + i as f64 * h) * if i % 2 == 0 { 2. } else { 4. })
            .sum::<f64>())
        * h
        / 3.
}
fn check(values: &[f64], expected: f64) -> Value {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = (values.iter().map(|v| v * v).sum::<f64>() / n - mean * mean).max(0.);
    let se = (variance / (n - 1.)).sqrt();
    assert!(
        (mean - expected).abs() < 6.5 * se + 1e-7,
        "mean={mean} reference={expected} SE={se}"
    );
    json!({"mean":mean,"reference":expected,"standard_error":se,"passed":true,"attempted_denominator":values.len()})
}

#[test]
fn hard_free_production_two_cloud_hard_and_depletion_sphere_limits() -> Result<()> {
    let base = root("analytic")?;
    let mut receipts = Vec::new();
    for (activity, seed) in [(0., 610020201), (2., 610020202)] {
        let path = base.join(format!("activity{activity}"));
        fs::create_dir(&path)?;
        fixture(&path, activity, 0.5, 1.)?;
        let cfg = options(&path, "run", SAMPLES, seed);
        // Bind the primary reference to the production CLI, not the test binary.
        let output = Command::new(env!("CARGO_BIN_EXE_latent-region-normalizer"))
            .args([
                "--config",
                path.join("config.json").to_str().unwrap(),
                "--region",
                path.join("region.json").to_str().unwrap(),
                "--importance-guide",
                path.join("guide.json").to_str().unwrap(),
                "--out",
                path.join("run").to_str().unwrap(),
                "--samples",
                &SAMPLES.to_string(),
                "--seed",
                &seed.to_string(),
            ])
            .output()?;
        fs::write(path.join("stdout.log"), &output.stdout)?;
        fs::write(path.join("stderr.log"), &output.stderr)?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let summary = read(path.join("run/summary.json"))?;
        let manifest = &summary["manifest"];
        assert_eq!(manifest["schema"], "importance-latent-region-normalizer-v6");
        assert_eq!(
            manifest["guide_schema"],
            "defensive-hard-free-line-guide-v1"
        );
        assert_eq!(manifest["resume_supported"], false);
        assert_eq!(manifest["cloud_replicates"], 2);
        assert_eq!(
            manifest["importance_guide_sha256"],
            hash_file(&path.join("guide.json"))?
        );
        assert_eq!(
            fs::read(path.join("guide.json"))?,
            fs::read(path.join("run/provenance/importance-guide.json"))?
        );
        let records = rows(path.join("run/samples.jsonl"))?;
        let attempts = rows(path.join("run/attempts.jsonl"))?;
        assert_eq!(records.len(), SAMPLES as usize);
        assert_eq!(attempts.len(), records.len());
        assert_eq!(
            summary["attempts_sha256"],
            hash_file(&path.join("run/attempts.jsonl"))?
        );
        assert_eq!(summary["estimates"]["region"]["draws"], SAMPLES);
        let (mut mass, mut hard, mut volume) = (Vec::new(), Vec::new(), Vec::new());
        let (mut exterior, mut hard_zero, mut conditioned, mut different_clouds) = (0, 0, 0, 0);
        for (i, row) in records.iter().enumerate() {
            assert_eq!(row["draw"], i);
            assert_eq!(attempts[i], json!({"draw":i,"state":"begin"}));
            let q = row["log_proposal_density"].as_f64().unwrap();
            let j = row["log_physical_jacobian"].as_f64().unwrap();
            let inside = row["shell_valid"].as_bool().unwrap();
            let valid = inside && row["hard_valid"] == true && row["region_valid"] == true;
            volume.push(if inside { (-q).exp() } else { 0. });
            let trace = &row["hard_free_line_density"];
            let axes = trace["axes"].as_array().unwrap();
            assert_eq!(axes.len(), 3);
            for (axis, item) in axes.iter().enumerate() {
                assert_eq!(item["axis"], axis);
                assert!(item.get("hard_free_intervals").is_some());
                assert!(
                    item.get("components").is_none() && item.get("geometry_cpu_seconds").is_none()
                );
            }
            conditioned += usize::from(row["hard_free_line_draw"]["conditional"] == true);
            if valid {
                assert!(q + 1e-12 >= trace["baseline_log_density"].as_f64().unwrap());
                assert!((row["log_hard_weight"].as_f64().unwrap() - (j - q)).abs() < 1e-12);
                let clouds = row["clouds"].as_array().unwrap();
                assert_eq!(clouds.len(), 2);
                different_clouds += usize::from(clouds[0] != clouds[1]);
                let lambda = manifest["lambda"].as_f64().unwrap();
                for cloud in clouds {
                    let w = activity * cloud["lower_volume"].as_f64().unwrap()
                        + cloud["overlap_points"].as_u64().unwrap() as f64
                            * (activity / lambda).ln_1p();
                    assert!((w - cloud["log_weight"].as_f64().unwrap()).abs() < 1e-12);
                }
                let pair_mean = clouds
                    .iter()
                    .map(|c| c["log_weight"].as_f64().unwrap().exp())
                    .sum::<f64>()
                    / 2.;
                let h = (j - q).exp();
                hard.push(h);
                mass.push(h * pair_mean);
                assert!(
                    (row["log_importance_weight"].as_f64().unwrap() - (h * pair_mean).ln()).abs()
                        < 1e-12
                );
            } else {
                assert!(
                    row["log_importance_weight"].is_null()
                        && row["log_hard_weight"].is_null()
                        && row["clouds"].as_array().unwrap().is_empty()
                );
                exterior += usize::from(!inside);
                hard_zero += usize::from(inside && row["hard_valid"] == false);
                hard.push(0.);
                mass.push(0.);
            }
        }
        assert!(exterior > 10 && hard_zero > 10 && conditioned > 1000);
        if activity > 0. {
            assert!(different_clouds > 10);
        }
        let receipt = json!({"activity":activity,"seed":seed,"samples":SAMPLES,"cloud_replicates":2,"lambda_ratio":64.,
            "hard":check(&hard,reference(0.)),"depletion":check(&mass,reference(activity)),
            "latent_volume":check(&volume,PI.powi(3)*OUTER.powi(6)/6.),"outside_zeros":exterior,"hard_zeros":hard_zero,
            "conditioned_draws":conditioned,"different_cloud_records":different_clouds,
            "manifest":manifest,"samples_sha256":summary["samples_sha256"],"attempts_sha256":summary["attempts_sha256"]});
        save(path.join("reference.json"), &receipt)?;
        receipts.push(receipt);
        // Exact indexed streams preserve prefixes and CLI/API rows. This is
        // reproducibility, not a resume API or permission to retry a failure.
        if activity == 2. {
            let mut prefix = cfg;
            prefix.out = path.join("prefix-api");
            prefix.samples = 64;
            latent_region::run_with_importance(prefix, &path.join("guide.json"))?;
            assert_eq!(rows(path.join("prefix-api/samples.jsonl"))?, records[..64]);
        }
    }
    save(
        base.join("reference.json"),
        &json!({"complete":true,"reference":"sphere radial quadrature with normalized Haar measure",
        "primary_attempts":2*SAMPLES,"prefix_control_attempts":64,"jobs":receipts,"no_protein_jobs":true}),
    )?;
    eprintln!(
        "Preserved hard-free-line sphere references: {}",
        base.display()
    );
    Ok(())
}

#[test]
fn hard_free_production_disabled_and_uniform_preserve_physical_streams() -> Result<()> {
    let base = root("controls")?;
    for (name, alpha, beta) in [("beta0", 0.5, 0.), ("alpha1", 1., 1.)] {
        let path = base.join(name);
        fs::create_dir(&path)?;
        fixture(&path, 2., alpha, beta)?;
        let opts = options(&path, "old", 128, 610020301);
        if alpha == 1. {
            latent_region::run(opts.clone())?;
        } else {
            latent_region::run_with_importance(opts.clone(), &path.join("old-guide.json"))?;
        }
        let mut new = opts;
        new.out = path.join("new");
        latent_region::run_with_importance(new, &path.join("guide.json"))?;
        let old = rows(path.join("old/samples.jsonl"))?;
        let new = rows(path.join("new/samples.jsonl"))?;
        for (a, mut b) in old.iter().zip(new) {
            assert_eq!(b["hard_free_line_density"]["conditioning_disabled"], true);
            b.as_object_mut().unwrap().remove("hard_free_line_draw");
            b.as_object_mut().unwrap().remove("hard_free_line_density");
            if alpha == 1. {
                for key in [
                    "shell_valid",
                    "log_proposal_density",
                    "proposal_branch",
                    "proposal_component",
                ] {
                    b.as_object_mut().unwrap().remove(key);
                }
            } else if b["proposal_branch"] == "hard-free-line" {
                b["proposal_branch"] = json!("gaussian");
            }
            assert_eq!(*a, b, "Disabled limit changed physical row or cloud RNG");
        }
    }
    Ok(())
}

#[test]
fn hard_free_production_rejects_changed_bindings_and_nonempty_outputs() -> Result<()> {
    let path = root("rejections")?;
    fixture(&path, 0., 0.5, 1.)?;
    let mut bad = read(path.join("guide.json"))?;
    bad["region_sha256"] = json!("0".repeat(64));
    save(path.join("bad-guide.json"), &bad)?;
    assert!(
        latent_region::run_with_importance(
            options(&path, "bad-region", 1, 1),
            &path.join("bad-guide.json")
        )
        .is_err()
    );
    bad = read(path.join("guide.json"))?;
    bad["contact_widths_A"] = json!([0.1]);
    save(path.join("bad-guide.json"), &bad)?;
    assert!(
        latent_region::run_with_importance(
            options(&path, "bad-schema", 1, 1),
            &path.join("bad-guide.json")
        )
        .is_err()
    );
    let out = path.join("existing");
    fs::create_dir(&out)?;
    fs::write(out.join("retain.txt"), b"keep")?;
    assert!(
        latent_region::run_with_importance(
            options(&path, "existing", 1, 1),
            &path.join("guide.json")
        )
        .is_err()
    );
    assert_eq!(fs::read(out.join("retain.txt"))?, b"keep");
    Ok(())
}
