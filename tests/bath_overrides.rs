//! CLI bath changes must alter the physical target while leaving the frozen
//! proposal, initial poses and deterministic continuation under explicit control.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
use tetramer_mc::{
    geometry::{Shape, SphereTree},
    initialization::FreeTetramerStart,
    simulation::{Config, hash_file},
};

struct Fixture {
    root: PathBuf,
    template: Value,
}
impl Fixture {
    fn new(label: &str, periodic: bool) -> Result<Self> {
        let root =
            std::env::temp_dir().join(format!("tetramer-mc-bath-{label}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            json!({"name":"sphere", "atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
        )?;
        let covariance: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..6).map(|j| if i == j { 0.3 } else { 0. }).collect())
            .collect();
        fs::write(root.join("model.json"), json!({"angular_length":1.,"anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
            "means":[[2.3,0.,0.,0.,0.,0.]], "covariances":[covariance],"weights":[1.],
            "shape_sha256":hash_file(&root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative"}).to_string())?;
        let template = json!({
            "shape":"shape.json", "box_lengths":[20.,20.,20.],
            "boundary":if periodic {json!({"kind":"periodic"})} else {json!({"kind":"spherical","radius":10.})},
            "seed":13579,"depletant_radius":0.2,"reservoir_density":0.08,"poisson_lambda_ratio":4.,
            "global_probability":0.7,"learned_uniform_weight":0.1,
            "frozen_posterior":{"probability":0.5,"correlation":0.9},
            "gca_probability":if periodic {0.} else {1.},"center_shift_probability":if periodic {0.} else {1.},
            "local_translation_std_A":0.1,"local_small_angle_std_degrees":2.,
            "endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25},
            "seed_labels":[0,1],"metadata":{"source":"synthetic bath override test"},
            "initial_poses":[{"position":[-1.1,0.,0.],"orientation":[1.,0.,0.,0.]},
                {"position":[1.1,0.,0.],"orientation":[1.,0.,0.,0.]}]
        });
        fs::write(root.join("config.json"), template.to_string())?;
        Ok(Self { root, template })
    }
    fn invoke(&self, name: &str, config: &Path, sweeps: u64, extra: &[&str]) -> Result<Output> {
        Ok(Command::new(env!("CARGO_BIN_EXE_tetramer-mc"))
            .args(["run", "--config"])
            .arg(config)
            .arg("--model")
            .arg(self.root.join("model.json"))
            .args(["--method", "learned", "--out"])
            .arg(self.root.join(name))
            .args([
                "--sweeps",
                &sweeps.to_string(),
                "--sample-every",
                "1",
                "--no-gsd",
            ])
            .args(extra)
            .output()?)
    }
    fn run(&self, name: &str, extra: &[&str]) -> Result<()> {
        success(self.invoke(name, &self.root.join("config.json"), 2, extra)?);
        Ok(())
    }
    fn read(&self, name: &str, file: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(name).join(file),
        )?)?)
    }
    fn config(&self, name: &str) -> Result<Value> {
        self.read(name, "provenance/input-config.json")
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "runner failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn without_config_hash(mut state: Value) -> Value {
    state.as_object_mut().unwrap().remove("config_sha256");
    state
}
fn without_timings(mut value: Value) -> Value {
    match &mut value {
        Value::Object(fields) => {
            fields.retain(|key, _| !key.ends_with("_seconds"));
            for child in fields.values_mut() {
                *child = without_timings(child.take());
            }
        }
        Value::Array(rows) => {
            for child in rows {
                *child = without_timings(child.take());
            }
        }
        _ => {}
    }
    value
}
fn move_rows(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| Ok(without_timings(serde_json::from_str(line)?)))
        .collect()
}

#[test]
fn independent_flags_preserve_other_bath_parameter_and_configuration() -> Result<()> {
    let f = Fixture::new("independent", false)?;
    for (name, flags, radius, activity) in [
        ("radius", vec!["--depletant-radius", "0.4"], 0.4, 0.08),
        ("activity", vec!["--depletant-activity", "0.02"], 0.2, 0.02),
    ] {
        f.run(name, &flags)?;
        let cfg = f.config(name)?;
        assert_eq!(cfg["depletant_radius"], radius);
        assert_eq!(cfg["reservoir_density"], activity);
        for key in [
            "initial_poses",
            "seed_labels",
            "seed",
            "boundary",
            "box_lengths",
            "global_probability",
            "frozen_posterior",
            "poisson_lambda_ratio",
            "gca_probability",
            "center_shift_probability",
        ] {
            assert_eq!(cfg[key], f.template[key], "override changed {key}");
        }
        assert!(Path::new(cfg["shape"].as_str().unwrap()).is_absolute());
        assert_eq!(f.read(name, "provenance/template-config.json")?, f.template);
        assert_eq!(
            fs::read(
                f.root
                    .join(name)
                    .join("provenance/frozen-relative-model.json")
            )?,
            fs::read(f.root.join("model.json"))?
        );
    }
    Ok(())
}

#[test]
fn zero_radius_and_zero_activity_are_valid_limits() -> Result<()> {
    let f = Fixture::new("zero", false)?;
    for (name, flags, radius, activity) in [
        ("radius", vec!["--depletant-radius", "0"], 0., 0.08),
        ("activity", vec!["--depletant-activity", "0"], 0.2, 0.),
        (
            "both",
            vec!["--depletant-radius", "0", "--depletant-activity", "0"],
            0.,
            0.,
        ),
    ] {
        f.run(name, &flags)?;
        let cfg = f.config(name)?;
        assert_eq!(cfg["depletant_radius"], radius);
        assert_eq!(cfg["reservoir_density"], activity);
        assert_eq!(f.read(name, "summary.json")?["complete"], true);
    }
    Ok(())
}

#[test]
fn invalid_bath_parameters_and_resume_overrides_fail_before_output() -> Result<()> {
    let f = Fixture::new("invalid", false)?;
    let cases = [
        vec!["--depletant-radius=-1"],
        vec!["--depletant-activity=-1"],
        vec!["--depletant-radius", "NaN"],
        vec!["--depletant-activity", "NaN"],
        vec!["--depletant-radius", "inf"],
        vec!["--depletant-activity", "inf"],
        vec!["--depletant-radius=-inf"],
        vec!["--depletant-activity=-inf"],
        vec!["--depletant-radius", "0.4", "--resume", "missing.json"],
        vec!["--depletant-activity", "0.02", "--resume", "missing.json"],
    ];
    for (i, flags) in cases.iter().enumerate() {
        let name = format!("invalid-{i}");
        assert!(
            !f.invoke(&name, &f.root.join("config.json"), 2, flags)?
                .status
                .success(),
            "accepted {flags:?}"
        );
        assert!(
            !f.root.join(name).exists(),
            "failed override left an output directory: {flags:?}"
        );
    }
    Ok(())
}

#[test]
fn enlarged_depletants_revalidate_periodic_cell_reach() -> Result<()> {
    let f = Fixture::new("periodic", true)?;
    let out = f.invoke(
        "bad-cell",
        &f.root.join("config.json"),
        2,
        &["--depletant-radius", "5"],
    )?;
    assert!(
        !out.status.success(),
        "L=20 cannot hold four exclusion bounding radii of 6"
    );
    assert!(!f.root.join("bad-cell").exists());
    Ok(())
}

#[test]
fn free_initialization_uses_requested_depletion_envelopes() -> Result<()> {
    let f = Fixture::new("free", false)?;
    let flags = [
        "--depletant-radius",
        "2",
        "--depletant-activity",
        "0.02",
        "--free-tetramers",
        "5",
        "--discard-seed",
        "--concentration-um",
        "1000000",
        "--seed",
        "93",
    ];
    f.run("run", &flags)?;
    let cfg = f.config("run")?;
    let mut expected: Config = serde_json::from_value(f.template.clone())?;
    expected.depletant_radius = 2.;
    expected.reservoir_density = 0.02;
    expected.seed_labels.clear();
    let shape: Shape = serde_json::from_slice(&fs::read(f.root.join("shape.json"))?)?;
    let tree = SphereTree::new(shape)?;
    FreeTetramerStart {
        count: 5,
        concentration_um: 1_000_000.,
        seed: Some(93),
    }
    .apply(&mut expected, &tree)?;
    assert_eq!(
        cfg["initial_poses"],
        serde_json::to_value(&expected.initial_poses)?
    );
    assert_eq!(cfg["depletant_radius"], 2.);
    assert_eq!(cfg["reservoir_density"], 0.02);
    assert_eq!(cfg["seed_labels"], json!([]));
    assert_eq!(
        f.read("run", "provenance/template-config.json")?,
        f.template
    );
    for (i, a) in expected.initial_poses.iter().enumerate() {
        for b in expected.initial_poses.iter().take(i) {
            let d2: f64 = a
                .position
                .iter()
                .zip(b.position)
                .map(|(x, y)| (x - y).powi(2))
                .sum();
            assert!(
                d2 >= 36.,
                "free initialization used the old depletion envelope"
            );
        }
    }
    Ok(())
}

#[test]
fn fixed_learned_map_matches_equivalent_explicit_bath_config() -> Result<()> {
    let f = Fixture::new("manual", false)?;
    let flags = ["--depletant-radius", "0.8", "--depletant-activity", "0.3"];
    success(f.invoke("override", &f.root.join("config.json"), 12, &flags)?);
    let mut manual = f.template.clone();
    manual["depletant_radius"] = json!(0.8);
    manual["reservoir_density"] = json!(0.3);
    fs::write(f.root.join("manual.json"), manual.to_string())?;
    success(f.invoke("manual", &f.root.join("manual.json"), 12, &[])?);
    assert_eq!(
        without_config_hash(f.read("override", "checkpoint.json")?),
        without_config_hash(f.read("manual", "checkpoint.json")?)
    );
    assert_eq!(
        move_rows(&f.root.join("override/moves.jsonl"))?,
        move_rows(&f.root.join("manual/moves.jsonl"))?
    );
    for name in ["override", "manual"] {
        assert_eq!(
            f.read(name, "manifest.json")?["model_sha256"],
            hash_file(&f.root.join("model.json"))?
        );
        assert_eq!(
            fs::read(
                f.root
                    .join(name)
                    .join("provenance/frozen-relative-model.json")
            )?,
            fs::read(f.root.join("model.json"))?
        );
    }
    Ok(())
}

#[test]
fn altered_bath_archived_config_resumes_exactly() -> Result<()> {
    let f = Fixture::new("resume", false)?;
    let flags = ["--depletant-radius", "0.8", "--depletant-activity", "0.3"];
    success(f.invoke("full", &f.root.join("config.json"), 12, &flags)?);
    success(f.invoke("part", &f.root.join("config.json"), 5, &flags)?);
    let archived = f.root.join("part/provenance/input-config.json");
    let checkpoint = f.root.join("part/checkpoint.json");
    success(f.invoke(
        "resumed",
        &archived,
        12,
        &["--resume", checkpoint.to_str().unwrap()],
    )?);
    assert_eq!(
        f.read("full", "checkpoint.json")?,
        f.read("resumed", "checkpoint.json")?
    );
    assert_eq!(f.read("resumed", "summary.json")?["initial_sweep"], 5);
    let complete_moves = move_rows(&f.root.join("full/moves.jsonl"))?;
    let mut restarted_moves = move_rows(&f.root.join("part/moves.jsonl"))?;
    restarted_moves.extend(move_rows(&f.root.join("resumed/moves.jsonl"))?);
    assert_eq!(complete_moves, restarted_moves);
    Ok(())
}
