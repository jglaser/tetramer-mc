//! End-to-end synthetic references for the full-vessel outer proposal mixture.
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
    proposal::FrozenRelativePoseProposal,
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
    let root = std::env::temp_dir().join(format!("vessel-latent-{name}-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir_all(&root)?;
    Ok(root)
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
    // Deliberately tiny historical source capture: it is metadata, never a
    // restriction of the current full atomic-wall target or guide tails.
    fs::write(root.join("region.json"),json!({"shape_sha256":hash,"fixed_neighbor":fixed,"physical_fixed_neighbors":neighbors,
        "capture_center":CENTER,"capture_radius":0.1,"minimum_original_q":0.,"mahalanobis_radius":4.,
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
#[test]
fn sphere_mixture_preserves_full_wall_depletion_and_haar_reference() -> Result<()> {
    let root = root("reference")?;
    fixture(&root, false, true, 0.5)?;
    for (index, z) in [0., 0.4].into_iter().enumerate() {
        let name = format!("z{index}");
        let n = 24000;
        let mut opt = options(&root, &name, n, z);
        opt.seed += index as u64;
        let result = normalizer::run_with_wall_and_latent_guide(opt, wall(), files(&root))?;
        assert_eq!(result["manifest"]["schema"], 5);
        assert_eq!(result["manifest"]["pose_proposal_schema"], 3);
        let rows = fs::read_to_string(root.join(&name).join("samples.jsonl"))?;
        let mut sums = [0.; 2];
        let mut squares = [0.; 2];
        let mut outside = 0;
        let mut invalid = 0;
        let mut latent_tail = 0;
        for (i, line) in rows.lines().enumerate() {
            let row: Value = serde_json::from_str(line)?;
            assert_eq!(row["draw"], i);
            let pose: Pose = serde_json::from_value(row["pose"].clone())?;
            let r = norm(sub(pose.position, CENTER));
            let valid = (2. * CORE..=WALL - CORE).contains(&r);
            assert_eq!(row["hard_valid"], valid);
            if valid {
                near(
                    row["log_hard_weight"].as_f64().unwrap(),
                    -row["log_proposal_density"].as_f64().unwrap(),
                );
                if row["latent_density"]["in_reference_ball"] == false {
                    outside += 1;
                    if row["outer_branch"] == "latent" {
                        latent_tail += 1;
                    }
                }
            } else {
                invalid += 1;
                assert!(row["log_importance_weight"].is_null());
                assert!(row["clouds"].as_array().unwrap().is_empty());
            }
            let w = row["log_importance_weight"].as_f64().map_or(0., f64::exp);
            for (j, x) in [w, w * rotation(pose.orientation)[0][0]]
                .into_iter()
                .enumerate()
            {
                sums[j] += x;
                squares[j] += x * x;
            }
        }
        assert_eq!(rows.lines().count(), n as usize);
        assert!(outside > 0 && invalid > 0 && latent_tail > 0);
        for (j, exact) in [reference(z), 0.].into_iter().enumerate() {
            let mean = sums[j] / n as f64;
            let se = ((squares[j] / n as f64 - mean * mean) / (n - 1) as f64).sqrt();
            eprintln!("full vessel latent z={z}, observable {j}: {mean} +/- {se}, exact {exact}");
            assert!((mean - exact).abs() < 6.5 * se);
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn all_world_rows_use_both_complete_laws_with_selected_or_averaged_anchors() -> Result<()> {
    let root = root("density")?;
    for reciprocal in [false, true] {
        fixture(&root, true, reciprocal, 0.5)?;
        for selected in [None, Some(1)] {
            let name = format!("density-{reciprocal}-{selected:?}");
            let mut opt = options(&root, &name, 512, 0.);
            opt.proposal_anchor_index = selected;
            // Legacy scaling must retain its standard-deviation interpretation.
            opt.covariance_scale = if reciprocal { 1. } else { 1.3 };
            let result =
                normalizer::run_with_wall_and_latent_guide(opt.clone(), wall(), files(&root))?;
            let cfg = read(root.join("config.json"))?;
            let capture: Vec3 = serde_json::from_value(cfg["capture_center"].clone())?;
            let fixed: Vec<Pose> = serde_json::from_value(cfg["fixed_poses"].clone())?;
            let original = FrozenRelativePoseProposal::from_json_str_open(
                &fs::read_to_string(root.join("model.json"))?,
                [2. * CAPTURE; 3],
                0.4,
                &hash_file(&root.join("shape.json"))?,
            )?;
            let model = if reciprocal {
                original
            } else {
                let mut c = original.component_parameters();
                for v in c.iter_mut().flat_map(|c| c.covariance.iter_mut()).flatten() {
                    *v *= 1.3_f64.powi(2);
                }
                original.with_component_parameters(c)?
            };
            let mut branches = [0; 2];
            let mut distinct_density = 0;
            let mut invalid = 0;
            for line in fs::read_to_string(root.join(&name).join("samples.jsonl"))?.lines() {
                let row: Value = serde_json::from_str(line)?;
                let pose: Pose = serde_json::from_value(row["pose"].clone())?;
                let centered = Pose {
                    position: sub(pose.position, capture),
                    ..pose
                };
                let anchors: Vec<Pose> = selected.map_or_else(|| fixed.clone(), |i| vec![fixed[i]]);
                let mut lp = f64::NEG_INFINITY;
                for a in &anchors {
                    lp = ladd(
                        lp,
                        model.log_density(
                            &centered,
                            &Pose {
                                position: sub(a.position, capture),
                                ..*a
                            },
                        )?,
                    );
                }
                lp -= (anchors.len() as f64).ln();
                near(lp, logzero(&row["log_vessel_proposal_density"]));
                // Independent inverse chart with diagonal covariance; no
                // PhysicalLatentGuide methods enter the reconstruction.
                let inv = transpose(rotation(fixed[0].orientation));
                let t = matvec(inv, sub(pose.position, fixed[0].position));
                let r = matmul(inv, rotation(pose.orientation));
                let d = 1. + r[0][0] + r[1][1] + r[2][2];
                let c = [
                    (r[2][1] - r[1][2]) / d,
                    (r[0][2] - r[2][0]) / d,
                    (r[1][0] - r[0][1]) / d,
                ];
                let u: [f64; 6] = std::array::from_fn(|i| {
                    if i < 3 {
                        (t[i] - if i == 0 { 1. } else { 0. }) / ST
                    } else {
                        ELL * c[i - 3] / SR
                    }
                });
                let radius2 = u.iter().map(|v| v * v).sum::<f64>();
                let gaussian = -3. * (2. * PI).ln()
                    - 0.5
                        * u.iter()
                            .enumerate()
                            .map(|(i, v)| (v - if i == 0 { 4. } else { 0. }).powi(2))
                            .sum::<f64>();
                let ball = if radius2 <= 16. {
                    -(PI.powi(3) * 4_f64.powi(6) / 6.).ln()
                } else {
                    f64::NEG_INFINITY
                };
                let logq = ladd(gaussian, ball) - 2_f64.ln();
                let logj = 3. * ST.ln() + 3. * SR.ln()
                    - 3. * ELL.ln()
                    - 2. * PI.ln()
                    - 2. * dot(c, c).ln_1p();
                near(
                    logj,
                    row["latent_density"]["log_physical_jacobian"]
                        .as_f64()
                        .unwrap(),
                );
                near(
                    logq,
                    row["latent_density"]["log_latent_density"]
                        .as_f64()
                        .unwrap(),
                );
                near(logq - logj, logzero(&row["log_latent_physical_density"]));
                let mix = ladd(lp, logq - logj) - 2_f64.ln();
                near(mix, row["log_proposal_density"].as_f64().unwrap());
                assert!(mix + 2_f64.ln() >= lp - 1e-10);
                distinct_density += usize::from((lp - (logq - logj)).abs() > 1.);
                if row["outer_branch"] == "vessel" {
                    branches[0] += 1;
                    assert!(row["proposal"].is_object());
                    assert!(row["latent_proposal"].is_null());
                } else {
                    branches[1] += 1;
                    assert!(row["proposal"].is_null());
                    assert!(row["latent_proposal"].is_object());
                }
                invalid += usize::from(row["hard_valid"] == false);
            }
            assert!(branches.iter().all(|n| *n > 150));
            assert!(distinct_density > 100 && invalid > 0);
            assert_eq!(
                result["manifest"]["latent_region_sha256"],
                hash_file(&root.join("region.json"))?
            );
            assert_eq!(
                fs::read(root.join(&name).join("provenance/latent-guide.json"))?,
                fs::read(root.join("guide.json"))?
            );
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn pure_uniform_source_keeps_zero_tail_component_and_original_vessel_stream() -> Result<()> {
    let root = root("uniform")?;
    fixture(&root, false, true, 1.)?;
    normalizer::run_with_wall(options(&root, "old", 512, 0.), wall())?;
    normalizer::run_with_wall_and_latent_guide(
        options(&root, "mixed", 512, 0.),
        wall(),
        files(&root),
    )?;
    normalizer::run_with_wall_and_latent_guide(
        options(&root, "repeat", 512, 0.),
        wall(),
        files(&root),
    )?;
    assert_eq!(
        fs::read(root.join("mixed/samples.jsonl"))?,
        fs::read(root.join("repeat/samples.jsonl"))?
    );
    let old = fs::read_to_string(root.join("old/samples.jsonl"))?;
    let mixed = fs::read_to_string(root.join("mixed/samples.jsonl"))?;
    let mut zeros = 0;
    for (a, b) in old.lines().zip(mixed.lines()) {
        let a: Value = serde_json::from_str(a)?;
        let b: Value = serde_json::from_str(b)?;
        assert!(a.get("outer_branch").is_none());
        assert_eq!(read(root.join("old/manifest.json"))?["schema"], 4);
        if b["outer_branch"] == "vessel" {
            assert_eq!(a["pose"], b["pose"]);
            assert_eq!(a["proposal"], b["proposal"]);
            assert_eq!(a["clouds"], b["clouds"]);
        }
        if b["latent_density"]["in_reference_ball"] == false {
            zeros += 1;
            assert!(b["log_latent_physical_density"].is_null());
            near(
                b["log_proposal_density"].as_f64().unwrap(),
                b["log_vessel_proposal_density"].as_f64().unwrap() - 2_f64.ln(),
            );
        }
    }
    assert!(zeros > 0);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn mixture_identity_and_cli_guards_fail_before_output() -> Result<()> {
    let root = root("guards")?;
    fixture(&root, true, true, 0.5)?;
    let path = root.join("config.json");
    let mut cfg = read(&path)?;
    cfg["fixed_poses"].as_array_mut().unwrap().swap(0, 1);
    fs::write(&path, cfg.to_string())?;
    let err = normalizer::run_with_wall_and_latent_guide(
        options(&root, "bad", 1, 0.),
        wall(),
        files(&root),
    )
    .unwrap_err();
    assert!(err.to_string().contains("physical neighbors differ"));
    assert!(!root.join("bad").exists());
    for flags in [
        vec!["--latent-region", "region.json"],
        vec!["--latent-guide", "guide.json"],
        vec![
            "--latent-region",
            "region.json",
            "--latent-guide",
            "guide.json",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_basin-normalizer"))
            .args([
                "--config", "missing", "--model", "missing", "--out", "missing",
            ])
            .args(flags)
            .output()?;
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("required arguments"));
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Export fresh, small reference records for the independent Python audit.
/// Explicitly opt in only after a physical worker slot is available. This does
/// not rerun the 48,000-draw radial/Haar tests or invoke a protein shape.
#[test]
#[ignore = "explicit durable cross-language fixture export; requires free physical slot"]
fn export_integrated_reference_fixtures() -> Result<()> {
    let path = std::env::var_os("TETRAMER_FULL_VESSEL_REFERENCE_OUT").ok_or_else(|| {
        anyhow::anyhow!("Set TETRAMER_FULL_VESSEL_REFERENCE_OUT to a new directory")
    })?;
    let export = PathBuf::from(path);
    anyhow::ensure!(
        !export.exists(),
        "Reference export directory already exists"
    );
    fs::create_dir_all(&export)?;
    fs::copy(std::env::current_exe()?, export.join("test-executable"))?;
    let mut records = vec![];
    for (kind, reciprocal, alpha) in [
        ("legacy", false, 0.5),
        ("reciprocal", true, 0.5),
        ("uniform", true, 1.),
    ] {
        for selected in [None, Some(1)] {
            let name = format!(
                "{kind}-{}",
                if selected.is_some() {
                    "selected"
                } else {
                    "all"
                }
            );
            let target = export.join(&name);
            fs::create_dir_all(&target)?;
            fixture(&target, true, reciprocal, alpha)?;
            let mut opt = options(&target, "output", 128, 0.4);
            opt.covariance_scale = if reciprocal { 1. } else { 1.3 };
            opt.proposal_anchor_index = selected;
            let summary = normalizer::run_with_wall_and_latent_guide(opt, wall(), files(&target))?;
            records.push(
                json!({"name":name,"output":target.join("output"),"samples":128,
                "manifest_sha256":hash_file(&target.join("output/manifest.json"))?,
                "samples_sha256":hash_file(&target.join("output/samples.jsonl"))?,
                "hard_valid":summary["hard_valid"]}),
            );
        }
    }
    fs::write(
        export.join("fixture-index.json"),
        serde_json::to_vec_pretty(&json!({
        "schema":"full-vessel-latent-integrated-reference-fixtures-v1","complete":true,
        "physical_system":"two analytic spheres, shifted center, rotated fixed frames; no protein configuration",
        "draws":768,"fixtures":records,"test_executable_sha256":hash_file(&export.join("test-executable"))?}))?,
    )?;
    eprintln!(
        "exported integrated reference fixtures to {}",
        export.display()
    );
    Ok(())
}
