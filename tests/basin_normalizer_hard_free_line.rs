//! New hard-free vessel-law references; historical populations are not replayed.
//! These fixtures contain one or two analytic spheres, never protein inputs.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tetramer_mc::{
    math::*,
    normalizer::{self, NormalizerLatentGuideFiles, NormalizerOptions, NormalizerWall},
    simulation::hash_file,
};

const CENTER: Vec3 = [7., -4., 3.];
const CORE: f64 = 0.3;
const WALL: f64 = 3.;
const CAPTURE: f64 = 5.;
const ST: f64 = 0.4;
const SR: f64 = 0.8;
const ELL: f64 = 1.5;

fn root(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("TETRAMER_HARD_FREE_VESSEL_REFERENCE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("hard-free-vessel-{}", std::process::id()))
        });
    let path = base.join(name);
    anyhow::ensure!(
        !path.exists(),
        "Fresh reference path required; no overwrite"
    );
    fs::create_dir_all(&path)?;
    Ok(path)
}
fn fixture(root: &Path, second: bool, reciprocal: bool, alpha: f64) -> Result<()> {
    fs::write(
        root.join("shape.json"),
        json!({"atoms":[{"center":[0.,0.,0.],"radius":CORE}]}).to_string(),
    )?;
    let hash = hash_file(&root.join("shape.json"))?;
    let fixed = Pose {
        position: CENTER,
        orientation: quaternion(cayley([0.2, -0.1, 0.3])),
    };
    let mut neighbors = vec![fixed];
    if second {
        neighbors.push(Pose {
            position: add(CENTER, [0., 1.2, 0.]),
            orientation: quaternion(cayley([-0.3, 0.2, 0.1])),
        });
    }
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let metadata = json!({"native_poses":[fixed],"rigid_members":[identity],"member_error_scale":3.,"angle_error_scale_deg":90.});
    let diagonal = |t: f64, r: f64| {
        std::array::from_fn::<_, 6, _>(|i| {
            std::array::from_fn::<_, 6, _>(|j| {
                if i != j {
                    0.
                } else if i < 3 {
                    t * t
                } else {
                    r * r
                }
            })
        })
    };
    let base = json!({"shape_sha256":hash,"coordinate_convention":"anchor-body-relative","angular_length":1.,
        "weights":[1.],"anchors":[{"position":[1.,0.,0.],"rotation":IDENTITY}],"means":[([0.;6])],"covariances":[diagonal(1.,1.)]});
    let model = if reciprocal {
        json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true]})
    } else {
        base
    };
    fs::write(root.join("model.json"), model.to_string())?;
    fs::write(
        root.join("config.json"),
        json!({"shape":"shape.json","fixed_poses":neighbors,
        "initial_pose":Pose{position:add(CENTER,[1.5,0.,0.]),..identity},
        "capture_center":add(CENTER,[0.2,0.1,0.]),"capture_radius":CAPTURE,
        "depletant_radius":0.5,"reservoir_density":0.4,"poisson_lambda_ratio":16.,
        "translation_steps":[0.1],"rotation_steps_deg":[5.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.4,"seed":441,
        "endpoint_gate":{"max_cells":63,"max_depth":8,"min_width":0.1},"metadata":metadata})
        .to_string(),
    )?;
    // Source capture conditions only the guide. The physical wall remains larger.
    fs::write(root.join("region.json"),json!({"shape_sha256":hash,"fixed_neighbor":fixed,"physical_fixed_neighbors":neighbors,
        "capture_center":CENTER,"capture_radius":1.4,"minimum_original_q":0.,"mahalanobis_radius":4.,
        "gaussian_chart":{"shape_sha256":hash,"coordinate_convention":"anchor-body-relative","angular_length":ELL,
        "weights":[1.],"anchors":[{"position":[1.,0.,0.],"rotation":IDENTITY}],"means":[([0.;6])],"covariances":[diagonal(ST,SR)]}}).to_string())?;
    fs::write(root.join("guide.json"),json!({"schema":"defensive-latent-shell-guide-v1","region_sha256":hash_file(&root.join("region.json"))?,
        "defensive_uniform_shell_probability":alpha,"gaussian_components":[{"weight":1.,"mean":[4.,0.,0.,0.,0.,0.],"covariance":diagonal(1.,1.)}]}).to_string())?;
    Ok(())
}
fn options(root: &Path, out: &str, n: u64, z: f64) -> NormalizerOptions {
    NormalizerOptions {
        config: root.join("config.json"),
        model: root.join("model.json"),
        out: root.join(out),
        samples: n,
        seed: 818391,
        covariance_scale: 1.,
        uniform_probability: Some(0.4),
        proposal_anchor_index: None,
        cloud_replicates: 2,
        activity: Some(z),
    }
}
fn files(root: &Path) -> NormalizerLatentGuideFiles {
    NormalizerLatentGuideFiles {
        region: root.join("region.json"),
        guide: root.join("guide.json"),
    }
}
fn wall() -> NormalizerWall {
    NormalizerWall {
        center: CENTER,
        radius: WALL,
    }
}
fn read(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 2e-8 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn logzero(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NEG_INFINITY)
}
fn ladd(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        a.max(b) + (a.min(b) - a.max(b)).exp().ln_1p()
    }
}

// Exact sphere lens and normalized Haar measure integrated independently of
// the sampler, its clouds, its proposal law, and the source chart.
fn reference(z: f64) -> f64 {
    let lo = 2. * CORE;
    let hi = WALL - CORE;
    let n = 16384;
    let h = (hi - lo) / n as f64;
    let f = |r: f64| {
        let e = CORE + 0.5;
        let v = if r < 2. * e {
            PI * (4. * e + r) * (2. * e - r).powi(2) / 12.
        } else {
            0.
        };
        4. * PI * r * r * (z * v).exp()
    };
    (f(lo)
        + f(hi)
        + (1..n)
            .map(|i| if i % 2 == 0 { 2. } else { 4. } * f(lo + h * i as f64))
            .sum::<f64>())
        * h
        / 3.
}

fn save(path: impl AsRef<Path>, v: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
fn records(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn hard_free(root: &Path, beta: f64) -> Result<()> {
    let mut guide = read(root.join("guide.json"))?;
    save(root.join("old-guide.json"), &guide)?;
    guide["schema"] = json!("defensive-hard-free-line-guide-v1");
    guide["conditional_probability"] = json!(beta);
    guide["minimum_conditional_mass"] = json!(1e-12);
    guide["raw_translation_axes"] = json!([0, 1, 2]);
    save(root.join("guide.json"), &guide)
}
fn mass_check(values: &[f64], reference: f64) -> Value {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se =
        ((values.iter().map(|x| x * x).sum::<f64>() / n - mean * mean).max(0.) / (n - 1.)).sqrt();
    assert!(
        (mean - reference).abs() < 6.5 * se + 1e-7,
        "{mean} +/- {se} != {reference}"
    );
    json!({"mean":mean,"standard_error":se,"reference":reference,"passed":true,"attempted_denominator":values.len()})
}

#[test]
fn hard_free_vessel_cli_full_wall_hard_depletion_and_haar_reference() -> Result<()> {
    let root = root("analytic")?;
    fixture(&root, false, true, 0.5)?;
    hard_free(&root, 1.)?;
    let mut references = Vec::new();
    for (index, z) in [0., 0.4].into_iter().enumerate() {
        let seed = 610030501 + index as u64;
        let n = 8192;
        let out = root.join(format!("activity{z}"));
        let command = Command::new(env!("CARGO_BIN_EXE_basin-normalizer"))
            .args([
                "--config",
                root.join("config.json").to_str().unwrap(),
                "--model",
                root.join("model.json").to_str().unwrap(),
                "--latent-region",
                root.join("region.json").to_str().unwrap(),
                "--latent-guide",
                root.join("guide.json").to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
                "--samples",
                &n.to_string(),
                "--seed",
                &seed.to_string(),
                "--cloud-replicates",
                "2",
                "--activity",
                &z.to_string(),
                "--uniform-probability",
                "0.4",
                "--wall-radius",
                "3",
                "--wall-center",
                "7",
                "-4",
                "3",
            ])
            .output()?;
        fs::write(
            root.join(format!("activity{z}-stdout.log")),
            &command.stdout,
        )?;
        fs::write(
            root.join(format!("activity{z}-stderr.log")),
            &command.stderr,
        )?;
        assert!(
            command.status.success(),
            "{}",
            String::from_utf8_lossy(&command.stderr)
        );
        let summary = read(out.join("summary.json"))?;
        let manifest = &summary["manifest"];
        assert_eq!(manifest["schema"], 6);
        assert_eq!(
            manifest["outer_mixture_schema"],
            "full-vessel-hard-free-line-half-mixture-v1"
        );
        assert_eq!(manifest["latent_source_capture"]["radius"], 1.4);
        assert_eq!(manifest["latent_source_capture"]["restricts_target"], false);
        assert_eq!(manifest["latent_source_capture"]["conditions_guide"], true);
        assert_eq!(manifest["resume_supported"], false);
        let rows = records(out.join("samples.jsonl"))?;
        let journal = records(out.join("attempts.jsonl"))?;
        assert_eq!(rows.len(), n);
        assert_eq!(journal.len(), n);
        assert_eq!(
            summary["attempts_sha256"],
            hash_file(&out.join("attempts.jsonl"))?
        );
        let (mut weighted, mut hard, mut haar) = (Vec::new(), Vec::new(), Vec::new());
        let (
            mut outside_source,
            mut outside_r4,
            mut outside_both,
            mut invalid,
            mut tail,
            mut structural_zeros,
        ) = (0, 0, 0, 0, 0, 0);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row["draw"], i);
            assert_eq!(journal[i], json!({"draw":i,"state":"begin"}));
            let pose: Pose = serde_json::from_value(row["pose"].clone())?;
            let radius = norm(sub(pose.position, CENTER));
            let valid = (2. * CORE..=WALL - CORE).contains(&radius);
            assert_eq!(row["hard_valid"], valid);
            let density = row["log_proposal_density"].as_f64().unwrap();
            let vessel = logzero(&row["log_vessel_proposal_density"]);
            let latent = logzero(&row["log_latent_physical_density"]);
            near(density, ladd(vessel, latent) - 2_f64.ln());
            assert!(density >= vessel - 2_f64.ln() - 1e-12);
            structural_zeros += usize::from(row["latent_density"]["structural_zero"] == true);
            if latent.is_finite() {
                near(
                    latent,
                    logzero(&row["latent_density"]["log_latent_density"])
                        - row["latent_density"]["log_physical_jacobian"]
                            .as_f64()
                            .unwrap(),
                );
            }
            if row["outer_branch"] == "latent" {
                assert!(row["latent_proposal"].get("hard_free_line_draw").is_some());
            }
            let w = row["log_importance_weight"].as_f64().map_or(0., f64::exp);
            weighted.push(w);
            hard.push(row["log_hard_weight"].as_f64().map_or(0., f64::exp));
            haar.push(w * rotation(pose.orientation)[0][0]);
            if valid {
                near(row["log_hard_weight"].as_f64().unwrap(), -density);
                outside_source += usize::from(radius > 1.4);
                outside_r4 += usize::from(row["latent_density"]["in_reference_ball"] == false);
                outside_both += usize::from(
                    radius > 1.4 && row["latent_density"]["in_reference_ball"] == false,
                );
                tail += usize::from(
                    row["outer_branch"] == "latent"
                        && row["latent_density"]["in_reference_ball"] == false,
                );
                assert_eq!(row["clouds"].as_array().unwrap().len(), 2);
            } else {
                invalid += 1;
                assert!(
                    row["log_importance_weight"].is_null()
                        && row["log_hard_weight"].is_null()
                        && row["clouds"].as_array().unwrap().is_empty()
                );
            }
        }
        assert!(
            outside_source > 20 && outside_r4 > 20 && outside_both > 20 && invalid > 20 && tail > 0
        );
        let receipt = json!({"activity":z,"seed":seed,"samples":n,"cloud_replicates":2,"hard":mass_check(&hard,reference(0.)),
            "depletion":mass_check(&weighted,reference(z)),"haar_moment":mass_check(&haar,0.),"outside_source_valid":outside_source,
            "outside_R4_valid":outside_r4,"outside_both_valid":outside_both,"invalid_zeros":invalid,"latent_tail_valid":tail,
            "structural_zero_queries":structural_zeros,"manifest":manifest,"samples_sha256":summary["samples_sha256"],"attempts_sha256":summary["attempts_sha256"]});
        save(out.join("reference.json"), &receipt)?;
        references.push(receipt);
    }
    save(
        root.join("reference.json"),
        &json!({"complete":true,"primary_attempts":16384,"jobs":references,"no_protein_draws":true}),
    )?;
    Ok(())
}

#[test]
fn hard_free_vessel_disabled_controls_preserve_full_legacy_physics_rows() -> Result<()> {
    let root = root("controls")?;
    for (index, (name, alpha, beta)) in [("beta0", 0.5, 0.), ("alpha1", 1., 1.)]
        .into_iter()
        .enumerate()
    {
        let path = root.join(name);
        fs::create_dir(&path)?;
        fixture(&path, true, true, alpha)?;
        hard_free(&path, beta)?;
        let mut old = options(&path, "old", 128, 0.4);
        old.seed = 610030601 + index as u64;
        let mut new = old.clone();
        new.out = path.join("new");
        normalizer::run_with_wall_and_latent_guide(
            old,
            wall(),
            NormalizerLatentGuideFiles {
                region: path.join("region.json"),
                guide: path.join("old-guide.json"),
            },
        )?;
        normalizer::run_with_wall_and_latent_guide(new, wall(), files(&path))?;
        let a = records(path.join("old/samples.jsonl"))?;
        let b = records(path.join("new/samples.jsonl"))?;
        if name == "beta0" {
            let mut baseline = options(&path, "vessel-only", 128, 0.4);
            baseline.seed = 610030601;
            normalizer::run_with_wall(baseline, wall())?;
            let baseline = records(path.join("vessel-only/samples.jsonl"))?;
            let mut matched = 0;
            for (original, mixed) in baseline.iter().zip(&b) {
                if mixed["outer_branch"] != "vessel" {
                    continue;
                }
                matched += 1;
                for field in [
                    "pose",
                    "proposal",
                    "capture_valid",
                    "wall_valid",
                    "hard_valid",
                    "q",
                    "depletion_contact",
                    "region",
                    "clouds",
                ] {
                    assert_eq!(
                        original[field], mixed[field],
                        "No-guide vessel/cloud stream changed: {field}"
                    );
                }
                near(
                    original["log_proposal_density"].as_f64().unwrap(),
                    mixed["log_vessel_proposal_density"].as_f64().unwrap(),
                );
            }
            assert!(matched > 32);
            assert_eq!(read(path.join("vessel-only/manifest.json"))?["schema"], 4);
            assert!(!path.join("vessel-only/attempts.jsonl").exists());
        }
        for (a, mut b) in a.iter().zip(b) {
            b["latent_density"]
                .as_object_mut()
                .unwrap()
                .remove("hard_free_line_density");
            b["latent_density"]
                .as_object_mut()
                .unwrap()
                .remove("structural_zero");
            if let Some(p) = b["latent_proposal"].as_object_mut() {
                p.remove("hard_free_line_draw");
            }
            assert_eq!(
                a, &b,
                "Disabled law changed physical pose/density/cloud record"
            );
        }
        assert_eq!(read(path.join("old/manifest.json"))?["schema"], 5);
        assert!(!path.join("old/attempts.jsonl").exists());
    }
    Ok(())
}
