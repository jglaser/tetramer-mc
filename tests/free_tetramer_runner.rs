//! CLI initialization at a requested tetramer count and concentration. Tiny
//! sphere shapes isolate runner geometry/provenance from protein convergence.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const CONCENTRATION: f64 = 1_000_000.;
const UM_TO_A3: f64 = 6.02214076e-10;

struct Fixture {
    root: PathBuf,
    template: Value,
}
impl Fixture {
    fn new(label: &str, periodic: bool) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "tetramer-mc-free-runner-{label}-{}",
            std::process::id()
        ));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            json!({
                "name":"tiny sphere", "atoms":[{"center":[0.,0.,0.],"radius":1.}]
            })
            .to_string(),
        )?;
        let template = json!({
            "shape":"shape.json", "box_lengths":[20.,20.,20.],
            "boundary":if periodic { json!({"kind":"periodic"}) }
                else { json!({"kind":"spherical","radius":10.}) },
            "seed":13579, "depletant_radius":0.2, "reservoir_density":0.,
            "poisson_lambda_ratio":4., "global_probability":0.3,
            "gca_probability":if periodic {0.} else {1.},
            "center_shift_probability":if periodic {0.} else {1.},
            "local_translation_std_A":0.1,"local_small_angle_std_degrees":2.,
            "seed_labels":[], "metadata":{"source":"synthetic runner test"},
            "initial_poses":[
                {"position":[-2.,0.,0.],"orientation":[1.,0.,0.,0.]},
                {"position":[2.,0.,0.],"orientation":[1.,0.,0.,0.]}
            ]
        });
        fs::write(root.join("config.json"), template.to_string())?;
        Ok(Self { root, template })
    }
    fn invoke(&self, name: &str, config: &Path, sweeps: u64, extra: &[&str]) -> Result<Output> {
        Ok(Command::new(env!("CARGO_BIN_EXE_tetramer-mc"))
            .args(["run", "--config"])
            .arg(config)
            .args(["--method", "local-uniform", "--out"])
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
    fn initialized(&self, name: &str) -> Result<Value> {
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
fn check_geometry(config: &Value, count: usize, concentration: f64, periodic: bool) {
    let poses = config["initial_poses"].as_array().unwrap();
    assert_eq!(poses.len(), count);
    assert_eq!(config["seed_labels"], json!([]));
    let lengths: [f64; 3] = serde_json::from_value(config["box_lengths"].clone()).unwrap();
    let volume = if periodic {
        lengths.iter().product::<f64>()
    } else {
        4. * std::f64::consts::PI * config["boundary"]["radius"].as_f64().unwrap().powi(3) / 3.
    };
    assert!((volume * concentration * UM_TO_A3 / count as f64 - 1.).abs() < 2e-12);
    assert_eq!(lengths[0], lengths[1]);
    assert_eq!(lengths[1], lengths[2]);
    if periodic {
        assert!(lengths[0] > 4. * 1.2);
    }
    let positions: Vec<[f64; 3]> = poses
        .iter()
        .map(|p| serde_json::from_value(p["position"].clone()).unwrap())
        .collect();
    for (i, position) in positions.iter().enumerate() {
        let q: [f64; 4] = serde_json::from_value(poses[i]["orientation"].clone()).unwrap();
        assert!((q.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-12);
        if periodic {
            assert!(
                position
                    .iter()
                    .zip(lengths)
                    .all(|(p, l)| *p >= 0. && *p < l)
            );
        } else {
            let radius = config["boundary"]["radius"].as_f64().unwrap();
            assert!(position.iter().map(|x| x * x).sum::<f64>().sqrt() + 1. <= radius + 1e-12);
        }
        for other in positions.iter().take(i) {
            let distance2: f64 = (0..3)
                .map(|axis| {
                    let d = position[axis] - other[axis];
                    let d = if periodic {
                        d - (d / lengths[axis]).round() * lengths[axis]
                    } else {
                        d
                    };
                    d * d
                })
                .sum();
            assert!(
                distance2 >= (2_f64 * 1.2).powi(2),
                "initial depletion envelopes intersect"
            );
        }
    }
}

#[test]
fn spherical_free_count_density_and_provenance() -> Result<()> {
    let f = Fixture::new("sphere", false)?;
    f.run(
        "run",
        &[
            "--free-tetramers",
            "5",
            "--tetramer-concentration-um",
            "1000000",
        ],
    )?;
    let config = f.initialized("run")?;
    check_geometry(&config, 5, CONCENTRATION, false);
    assert_eq!(config["seed"], f.template["seed"]);
    let initialization = &config["metadata"]["free_tetramer_initialization"];
    assert_eq!(initialization["count"], 5);
    assert_eq!(initialization["tetramer_concentration_uM"], CONCENTRATION);
    assert_eq!(config["metadata"]["preparation_equilibrated"], false);
    assert_eq!(
        config["metadata"]["template_metadata"],
        f.template["metadata"]
    );
    for field in [
        "depletant_radius",
        "reservoir_density",
        "poisson_lambda_ratio",
        "global_probability",
        "gca_probability",
        "center_shift_probability",
        "local_translation_std_A",
        "local_small_angle_std_degrees",
    ] {
        assert_eq!(
            config[field], f.template[field],
            "template option changed: {field}"
        );
    }
    assert!(Path::new(config["shape"].as_str().unwrap()).is_absolute());
    assert_eq!(
        f.read("run", "provenance/template-config.json")?,
        f.template
    );
    assert_eq!(f.read("run", "summary.json")?["bodies"], 5);
    Ok(())
}

#[test]
fn periodic_density_and_minimum_image_separation() -> Result<()> {
    let f = Fixture::new("periodic", true)?;
    f.run(
        "run",
        &["--free-tetramers", "5", "--concentration-um", "1000000"],
    )?;
    check_geometry(&f.initialized("run")?, 5, CONCENTRATION, true);
    Ok(())
}

#[test]
fn preparation_seed_controls_initial_and_simulation_states() -> Result<()> {
    let f = Fixture::new("seed", false)?;
    for name in ["first", "same"] {
        f.run(
            name,
            &[
                "--free-tetramers",
                "4",
                "--concentration-um",
                "1000000",
                "--seed",
                "37",
            ],
        )?;
    }
    f.run(
        "different",
        &[
            "--free-tetramers",
            "4",
            "--concentration-um",
            "1000000",
            "--seed",
            "38",
        ],
    )?;
    let first = f.initialized("first")?;
    assert_eq!(first["seed"], 37);
    assert_eq!(
        first["initial_poses"],
        f.initialized("same")?["initial_poses"]
    );
    assert_ne!(
        first["initial_poses"],
        f.initialized("different")?["initial_poses"]
    );
    assert_eq!(
        f.read("first", "checkpoint.json")?,
        f.read("same", "checkpoint.json")?
    );
    Ok(())
}

#[test]
fn malformed_or_impossible_free_starts_fail_before_output_creation() -> Result<()> {
    let f = Fixture::new("invalid", false)?;
    let cases: &[&[&str]] = &[
        &["--free-tetramers", "4"],
        &["--concentration-um", "1000000"],
        &["--seed", "37"],
        &["--free-tetramers", "1", "--concentration-um", "1000000"],
        &["--free-tetramers", "4", "--concentration-um", "NaN"],
        &["--free-tetramers", "4", "--concentration-um", "inf"],
        &["--free-tetramers", "4", "--concentration-um", "0"],
        &["--free-tetramers", "4", "--concentration-um=-1"],
        &["--free-tetramers", "4", "--concentration-um", "1e15"],
        &[
            "--free-tetramers",
            "4",
            "--concentration-um",
            "1000000",
            "--resume",
            "missing.json",
        ],
    ];
    for (i, flags) in cases.iter().enumerate() {
        let name = format!("invalid-{i}");
        let output = f.invoke(&name, &f.root.join("config.json"), 2, flags)?;
        assert!(
            !output.status.success(),
            "invalid options accepted: {flags:?}"
        );
        assert!(
            !f.root.join(name).exists(),
            "failed initialization created an output directory: {flags:?}"
        );
    }
    Ok(())
}

#[test]
fn legacy_without_free_flags_preserves_initial_configuration() -> Result<()> {
    let f = Fixture::new("legacy", false)?;
    f.run("run", &[])?;
    let config = f.initialized("run")?;
    assert_eq!(config["initial_poses"], f.template["initial_poses"]);
    assert_eq!(config["seed_labels"], f.template["seed_labels"]);
    assert_eq!(config["boundary"], f.template["boundary"]);
    assert_eq!(config["box_lengths"], f.template["box_lengths"]);
    Ok(())
}

#[test]
fn generated_config_resumes_exactly_without_reinitialization() -> Result<()> {
    let f = Fixture::new("resume", false)?;
    let flags = [
        "--free-tetramers",
        "4",
        "--concentration-um",
        "1000000",
        "--seed",
        "91",
    ];
    success(f.invoke("full", &f.root.join("config.json"), 4, &flags)?);
    success(f.invoke("part", &f.root.join("config.json"), 2, &flags)?);
    let config = f.root.join("part/provenance/input-config.json");
    let checkpoint = f.root.join("part/checkpoint.json");
    success(f.invoke(
        "resumed",
        &config,
        4,
        &["--resume", checkpoint.to_str().unwrap()],
    )?);
    assert_eq!(
        f.read("full", "checkpoint.json")?,
        f.read("resumed", "checkpoint.json")?
    );
    assert_eq!(f.read("resumed", "summary.json")?["initial_sweep"], 2);
    Ok(())
}

/// Exercise the actual repaired protein geometry without starting a physical run.
#[test]
fn repaired_tetramers_fit_requested_concentration() -> Result<()> {
    use tetramer_mc::{
        geometry::{Shape, SphereTree},
        initialization::FreeTetramerStart,
        math::{norm, sub},
        simulation::Config,
        spherical::Container,
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let template = root.join("examples/spherical-reciprocal-free.json");
    let original: Config = serde_json::from_slice(&fs::read(&template)?)?;
    let shape: Shape =
        serde_json::from_slice(&fs::read(template.parent().unwrap().join(&original.shape))?)?;
    let tree = SphereTree::new(shape)?;
    for (count, expected_radius) in [(12, 354.478713257288), (24, 446.615192572506)] {
        let mut config = original.clone();
        FreeTetramerStart {
            count,
            concentration_um: 106.8,
            seed: Some(20260924),
        }
        .apply(&mut config, &tree)?;
        let radius = config.boundary.radius().unwrap();
        assert!((radius - expected_radius).abs() < 1e-9);
        let wall = Container::new(radius, &tree)?;
        assert_eq!(config.initial_poses.len(), count);
        for (i, &pose) in config.initial_poses.iter().enumerate() {
            assert!(wall.contains(pose), "atom exceeds wall");
            for other in config.initial_poses.iter().take(i) {
                assert!(
                    norm(sub(pose.position, other.position))
                        > 2. * (tree.bound + config.depletant_radius)
                );
            }
        }
    }
    Ok(())
}
