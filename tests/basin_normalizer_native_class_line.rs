//! CLI wiring/identity check only: 160 attempted poses, never a precision gate.
//! Four coincident rigid members represent exactly one union sphere. No protein
//! inputs or assembly states are used. Set TETRAMER_CLASS_VESSEL_TEST_ROOT to a
//! fresh absolute path to retain every fixture, journal and output for auditing.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tetramer_mc::{math::*, simulation::hash_file};

const CENTER: Vec3 = [7., -4., 3.];
const CORE: f64 = 0.3;
const SOURCE_CAPTURE: f64 = 1.4;
const WALL: f64 = 3.;
const CAPTURE: f64 = 5.;
const ATTEMPTS: usize = 160;

struct OutputRoot {
    path: PathBuf,
    retained: bool,
}
impl OutputRoot {
    fn fresh() -> Result<Self> {
        let specified = std::env::var_os("TETRAMER_CLASS_VESSEL_TEST_ROOT");
        let retained = specified.is_some();
        let path = specified.map(PathBuf::from).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("class-vessel-cli-{}", std::process::id()))
        });
        ensure!(
            path.is_absolute() && !path.exists(),
            "Fresh absolute CLI fixture directory required"
        );
        fs::create_dir_all(&path)?;
        Ok(Self { path, retained })
    }
}
impl Drop for OutputRoot {
    fn drop(&mut self) {
        if !self.retained {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
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
fn diagonal(t: f64, r: f64) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i != j {
                0.
            } else if i < 3 {
                t * t
            } else {
                r * r
            }
        })
    })
}
fn fixture(root: &Path) -> Result<()> {
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let fixed = Pose {
        position: CENTER,
        orientation: quaternion(cayley([0.2, -0.1, 0.3])),
    };
    save(
        root.join("shape.json"),
        &json!({"name":"four-coincident-members-one-union-sphere",
        "atoms":(0..4).map(|_|json!({"center":[0.,0.,0.],"radius":CORE})).collect::<Vec<_>>() }),
    )?;
    let shape_hash = hash_file(&root.join("shape.json"))?;
    let native_pose = Pose {
        position: fixed.apply([0.8, 0., 0.]),
        orientation: fixed.orientation,
    };
    let metadata = json!({"native_poses":[native_pose],"rigid_members":vec![identity;4],
        "member_error_scale":2.,"angle_error_scale_deg":15.});
    let base = json!({"shape_sha256":shape_hash,"coordinate_convention":"anchor-body-relative","angular_length":1.,
        "weights":[1.],"anchors":[{"position":[1.,0.,0.],"rotation":IDENTITY}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[diagonal(1.,1.)]});
    save(
        root.join("model.json"),
        &json!({"schema":"reciprocal-pose-mixture-v1",
        "base_model":base,"reciprocal_components":[true]}),
    )?;
    save(
        root.join("config.json"),
        &json!({"shape":"shape.json","fixed_poses":[fixed],
        "initial_pose":Pose{position:add(CENTER,[1.5,0.,0.]),..identity},
        "capture_center":add(CENTER,[0.2,0.1,0.]),"capture_radius":CAPTURE,
        "depletant_radius":0.5,"reservoir_density":0.4,"poisson_lambda_ratio":16.,
        "translation_steps":[0.1],"rotation_steps_deg":[5.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.4,"seed":441,
        "endpoint_gate":{"max_cells":63,"max_depth":8,"min_width":0.1},"metadata":metadata}),
    )?;
    save(
        root.join("region.json"),
        &json!({"shape_sha256":shape_hash,"fixed_neighbor":fixed,
        "physical_fixed_neighbors":[fixed],"capture_center":CENTER,"capture_radius":SOURCE_CAPTURE,
        "activity":0.4,"depletant_radius":0.5,"minimum_original_q":0.,"mahalanobis_radius":4.,
        "physical_metric":metadata,"gaussian_chart":{"shape_sha256":shape_hash,
        "coordinate_convention":"anchor-body-relative","angular_length":1.5,"weights":[1.],
        "anchors":[{"position":[1.,0.,0.],"rotation":IDENTITY}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[diagonal(0.4,0.8)]}}),
    )?;
    let native = json!({"schema":"native-entry-compiled-v1","source_definition_sha256":"0".repeat(64),
        "source_input_sha256":{"tetramer-shape.json":shape_hash},
        "criteria":{"body_member_position_entry_A":2.,"body_orientation_entry_deg":15.,
        "monomer_position_entry_A":3.,"monomer_orientation_entry_deg":20.,"contact_entry_A":2.,
        "native_reference_patch_gap_A":1.,"minimum_shared_native_residue_pairs":1.,
        "hard_overlap_tolerance_A":1e-8,"catalogue_cycle_position_tolerance_A":1e-6,
        "catalogue_cycle_angle_tolerance_deg":1e-6},"fixed_poses":[fixed],
        "members":(0..4).map(|_|json!({"position":[0.,0.,0.],"rotation":IDENTITY})).collect::<Vec<_>>(),
        "monomer_atoms":[{"center":[0.,0.,0.],"radius":CORE,"residue":0}],"residue_count":1,
        "references":[{"label":"toy","family":"toy","position":[0.8,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
        "motifs":[{"id":0,"position":[0.8,0.,0.],"rotation":IDENTITY,
        "member_contacts":(0..4).map(|i|json!({"member_i":i,"member_j":i,"directed_class":"toy"})).collect::<Vec<_>>()}]});
    save(root.join("compiled-native.json"), &native)?;
    let gaussian = json!({"schema":"defensive-latent-shell-guide-v1",
        "region_sha256":hash_file(&root.join("region.json"))?,"defensive_uniform_shell_probability":0.5,
        "gaussian_components":[{"weight":1.,"mean":[4.,0.,0.,0.,0.,0.],"covariance":diagonal(1.,1.)}]});
    save(root.join("gaussian-guide.json"), &gaussian)?;
    let mut guide = gaussian;
    guide["schema"] = json!("defensive-native-class-line-guide-v1");
    guide["raw_translation_axes"] = json!([0, 1, 2]);
    guide["conditional_probability"] = json!(1.);
    guide["minimum_conditional_mass"] = json!(1e-12);
    guide["class_channels"] = json!([{"class":"hard_free","probability":0.2},
        {"class":"native","probability":0.2},{"class":"contact_without_native","probability":0.2},
        {"class":"native","probability":0.2,"orthant":63},
        {"class":"contact_without_native","probability":0.2,"orthant":0}]);
    guide["compiled_native"] = json!({"path":root.join("compiled-native.json"),"sha256":hash_file(&root.join("compiled-native.json"))?});
    guide["shape_sha256"] = json!(shape_hash);
    guide["fixed_poses"] = json!([fixed]);
    guide["capture_center"] = json!(CENTER);
    guide["capture_radius"] = json!(SOURCE_CAPTURE);
    guide["depletant_radius"] = json!(0.5);
    save(root.join("class-guide.json"), &guide)?;
    guide["conditional_probability"] = json!(0.);
    save(root.join("disabled-class-guide.json"), &guide)
}

fn invoke(root: &Path, name: &str, guide: &str, n: usize, activity: f64, seed: u64) -> Result<()> {
    let arguments = vec![
        "--config".to_owned(),
        root.join("config.json").display().to_string(),
        "--model".into(),
        root.join("model.json").display().to_string(),
        "--latent-region".into(),
        root.join("region.json").display().to_string(),
        "--latent-guide".into(),
        root.join(guide).display().to_string(),
        "--out".into(),
        root.join(name).display().to_string(),
        "--samples".into(),
        n.to_string(),
        "--seed".into(),
        seed.to_string(),
        "--cloud-replicates".into(),
        "2".into(),
        "--activity".into(),
        activity.to_string(),
        "--uniform-probability".into(),
        "0.4".into(),
        "--wall-radius".into(),
        WALL.to_string(),
        "--wall-center".into(),
        CENTER[0].to_string(),
        CENTER[1].to_string(),
        CENTER[2].to_string(),
    ];
    save(
        root.join(format!("{name}-command.json")),
        &json!({"executable":env!("CARGO_BIN_EXE_basin-normalizer"),
        "argv":arguments,"attempts":n,"cloud_replicates":2,"seed":seed,"activity":activity,"retries":0}),
    )?;
    let child = Command::new(env!("CARGO_BIN_EXE_basin-normalizer"))
        .args(&arguments)
        .output()?;
    fs::write(root.join(format!("{name}-stdout.log")), &child.stdout)?;
    fs::write(root.join(format!("{name}-stderr.log")), &child.stderr)?;
    save(
        root.join(format!("{name}-process.json")),
        &json!({"complete":true,"child_drained":true,
        "success":child.status.success(),"exit_code":child.status.code()}),
    )?;
    ensure!(
        child.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    Ok(())
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
fn near(a: f64, b: f64) {
    assert!(
        a == b || (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

fn check_class_output(root: &Path, n: usize, active: bool) -> Result<Value> {
    let summary = read(root.join("summary.json"))?;
    let manifest = read(root.join("manifest.json"))?;
    assert_eq!(summary["complete"], true);
    assert_eq!(summary["numerical_nulls"], 0);
    assert_eq!(summary["manifest"], manifest);
    assert_eq!(summary["samples"], n);
    assert_eq!(manifest["schema"], 7);
    assert_eq!(manifest["outer_vessel_probability"], 0.5);
    assert_eq!(
        manifest["outer_mixture_schema"],
        "full-vessel-native-class-line-half-mixture-v1"
    );
    assert_eq!(
        manifest["latent_guide_schema"],
        "defensive-native-class-line-guide-v1"
    );
    assert_eq!(
        manifest["latent_source_capture"],
        json!({"center":CENTER,"radius":SOURCE_CAPTURE,
        "restricts_target":false,"conditions_guide":true})
    );
    assert_eq!(
        manifest["atomic_wall"],
        json!({"center":CENTER,"radius":WALL})
    );
    assert_eq!(manifest["bath_wall_permeable"], true);
    assert_eq!(
        manifest["latent_reference_ball_is_target_restriction"],
        false
    );
    assert_eq!(manifest["resume_supported"], false);
    for (file, key) in [
        ("input-config.json", "config_sha256"),
        ("shape.json", "shape_sha256"),
        ("model.json", "model_sha256"),
        ("latent-region.json", "latent_region_sha256"),
        ("latent-guide.json", "latent_guide_sha256"),
        ("source-bundle.json", "source_bundle_sha256"),
    ] {
        assert_eq!(
            manifest[key],
            hash_file(&root.join("provenance").join(file))?
        );
    }
    let native = read(root.join("provenance/compiled-native.json"))?;
    let provenance = &manifest["compiled_native"];
    assert_eq!(
        provenance["compiled_sha256"],
        hash_file(&root.join("provenance/compiled-native.json"))?
    );
    assert_eq!(
        provenance["source_definition_sha256"],
        native["source_definition_sha256"]
    );
    assert_eq!(
        provenance["source_input_sha256"],
        native["source_input_sha256"]
    );
    assert_eq!(provenance["shape_compatibility"]["compatible"], true);
    assert_eq!(provenance["shape_compatibility"]["matched_atoms"], 4);
    assert_eq!(
        provenance["shape_compatibility"]["hard_valid_implication_within_tolerance"],
        true
    );
    let samples = rows(root.join("samples.jsonl"))?;
    let attempts = rows(root.join("attempts.jsonl"))?;
    assert_eq!(samples.len(), n);
    assert_eq!(attempts.len(), n);
    assert_eq!(
        summary["samples_sha256"],
        hash_file(&root.join("samples.jsonl"))?
    );
    assert_eq!(
        summary["attempts_sha256"],
        hash_file(&root.join("attempts.jsonl"))?
    );
    let config = read(root.join("config.json"))?;
    assert_eq!(config["capture_radius"], CAPTURE);
    let capture_center: Vec3 = serde_json::from_value(config["capture_center"].clone())?;
    let (mut valid, mut outside_source, mut exterior, mut invalid, mut latent, mut vessel) =
        (0, 0, 0, 0, 0, 0);
    let mut conditional_generated = 0;
    for (i, row) in samples.iter().enumerate() {
        assert_eq!(row["draw"], i);
        assert_eq!(attempts[i], json!({"draw":i,"state":"begin"}));
        let pose: Pose = serde_json::from_value(row["pose"].clone())?;
        let radius = norm(sub(pose.position, CENTER));
        let is_valid = (2. * CORE..=WALL - CORE).contains(&radius);
        assert_eq!(
            row["capture_valid"],
            norm(sub(pose.position, capture_center)) <= CAPTURE
        );
        assert_eq!(row["wall_valid"], radius <= WALL - CORE);
        assert_eq!(row["hard_valid"], is_valid);
        let logq = row["log_proposal_density"].as_f64().unwrap();
        let qv = logzero(&row["log_vessel_proposal_density"]);
        let ql = logzero(&row["log_latent_physical_density"]);
        near(logq, ladd(qv, ql) - 2_f64.ln());
        if qv.is_finite() {
            assert!(logq >= qv - 2_f64.ln() - 1e-12);
        }
        let d = &row["latent_density"];
        assert!(d.get("hard_free_line_density").is_none());
        assert_eq!(d["structural_zero"], ql == f64::NEG_INFINITY);
        if ql.is_finite() {
            near(
                ql,
                d["log_latent_density"].as_f64().unwrap()
                    - d["log_physical_jacobian"].as_f64().unwrap(),
            );
        }
        let trace = &d["native_class_line_density"];
        if active {
            assert_eq!(trace["trace_format"], "class-line-compact-v1");
            assert_eq!(trace["axes"].as_array().unwrap().len(), 3);
            for axis in trace["axes"].as_array().unwrap() {
                assert_eq!(axis["channels"].as_array().unwrap().len(), 5);
            }
        } else {
            assert_eq!(trace["conditioning_disabled"], true);
        }
        if row["outer_branch"] == "latent" {
            latent += 1;
            let p = &row["latent_proposal"];
            assert!(
                p.get("native_class_line_draw").is_some() && p.get("hard_free_line_draw").is_none()
            );
            conditional_generated +=
                usize::from(p["native_class_line_draw"]["conditional"] == true);
            assert!(ql.is_finite());
        } else {
            vessel += 1;
            assert_eq!(row["outer_branch"], "vessel");
            assert!(row["latent_proposal"].is_null());
        }
        if is_valid {
            valid += 1;
            outside_source += usize::from(radius > SOURCE_CAPTURE);
            exterior += usize::from(d["in_reference_ball"] == false);
            near(row["log_hard_weight"].as_f64().unwrap(), -logq);
            let clouds = row["clouds"].as_array().unwrap();
            assert_eq!(clouds.len(), 2);
            let cloud_sum = clouds
                .iter()
                .map(|c| c["log_weight"].as_f64().unwrap())
                .fold(f64::NEG_INFINITY, ladd);
            near(
                row["log_importance_weight"].as_f64().unwrap(),
                cloud_sum - 2_f64.ln() - logq,
            );
            if manifest["activity"] == 0. {
                near(row["log_importance_weight"].as_f64().unwrap(), -logq);
            }
        } else {
            invalid += 1;
            assert!(row["log_hard_weight"].is_null() && row["log_importance_weight"].is_null());
            assert!(row["clouds"].as_array().unwrap().is_empty());
        }
    }
    assert_eq!(summary["hard_valid"], valid);
    for estimate in summary["estimates"].as_object().unwrap().values() {
        assert_eq!(estimate["unconditional_draws"], n);
    }
    if active {
        assert!(
            outside_source > 0
                && exterior > 0
                && invalid > 0
                && latent > 0
                && vessel > 0
                && conditional_generated > 0
        );
    }
    Ok(
        json!({"attempts":n,"valid":valid,"invalid_zeros":invalid,"valid_outside_source":outside_source,
        "valid_outside_R4":exterior,"latent_branch":latent,"vessel_branch":vessel,
        "conditional_generated":conditional_generated,
        "samples_sha256":summary["samples_sha256"],"attempts_sha256":summary["attempts_sha256"],
        "scope":"Wiring and same-row algebra only; no precision, mass-convergence or equilibrium conclusion"}),
    )
}

#[test]
fn native_class_vessel_cli_retains_full_target_provenance_and_disabled_parity() -> Result<()> {
    let root = OutputRoot::fresh()?;
    let path = &root.path;
    fixture(path)?;
    save(
        path.join("allocation.json"),
        &json!({"total_attempts":ATTEMPTS,"maximum_clouds":2*ATTEMPTS,
        "jobs":[{"name":"class-hard","attempts":64,"activity":0.,"seed":6100407201_u64},
        {"name":"class-depletion","attempts":64,"activity":0.4,"seed":6100407202_u64},
        {"name":"gaussian-control","attempts":16,"activity":0.4,"seed":6100407203_u64},
        {"name":"disabled-class","attempts":16,"activity":0.4,"seed":6100407203_u64}],
        "precision_gate":false,"retries":0,"protein_attempts":0}),
    )?;
    let mut checks = Vec::new();
    for (name, z, seed) in [
        ("class-hard", 0., 6100407201),
        ("class-depletion", 0.4, 6100407202),
    ] {
        invoke(path, name, "class-guide.json", 64, z, seed)?;
        checks.push(check_class_output(&path.join(name), 64, true)?);
    }
    invoke(
        path,
        "gaussian-control",
        "gaussian-guide.json",
        16,
        0.4,
        6100407203,
    )?;
    invoke(
        path,
        "disabled-class",
        "disabled-class-guide.json",
        16,
        0.4,
        6100407203,
    )?;
    checks.push(check_class_output(&path.join("disabled-class"), 16, false)?);
    let baseline = rows(path.join("gaussian-control/samples.jsonl"))?;
    let disabled = rows(path.join("disabled-class/samples.jsonl"))?;
    assert_eq!(baseline.len(), 16);
    assert_eq!(disabled.len(), 16);
    for (old, mut new) in baseline.iter().zip(disabled) {
        new["latent_density"]
            .as_object_mut()
            .unwrap()
            .remove("native_class_line_density");
        new["latent_density"]
            .as_object_mut()
            .unwrap()
            .remove("structural_zero");
        if let Some(p) = new["latent_proposal"].as_object_mut() {
            p.remove("native_class_line_draw");
        }
        assert_eq!(
            old, &new,
            "Disabled class changed pose/proposal/density/cloud record"
        );
    }
    assert_eq!(
        read(path.join("gaussian-control/manifest.json"))?["schema"],
        5
    );
    let old_attempts = rows(path.join("gaussian-control/attempts.jsonl"))?;
    assert_eq!(
        old_attempts,
        (0..16)
            .map(|i| json!({"draw":i,"state":"begin"}))
            .collect::<Vec<_>>()
    );
    save(
        path.join("reference.json"),
        &json!({"complete":true,"passed":true,"attempts":ATTEMPTS,
        "checks":checks,"disabled_control_paired_rows":16,"no_stochastic_mass_gate":true,
        "new_protein_draws":0,"physical_sphere_reference":"Analytic hard validity and same-row estimator algebra; not normalization precision"}),
    )?;
    Ok(())
}
