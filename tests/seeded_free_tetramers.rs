//! The requested free count augments only the labeled seed; concentration is
//! based on every body, and seed contacts are exempt from initial envelope separation.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const UM_TO_A3: f64 = 6.02214076e-10;
const CONCENTRATION: f64 = 1_000_000.;

struct Fixture {
    root: PathBuf,
    template: Value,
}
impl Fixture {
    fn new(label: &str, periodic: bool) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "tetramer-mc-seeded-free-{label}-{}",
            std::process::id()
        ));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            json!({"name":"sphere", "atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
        )?;
        let positions = if periodic {
            [
                [12., 12., 12.],
                [3., 3., 3.],
                [15., 15., 15.],
                [5.1, 3., 3.],
            ]
        } else {
            [[7., 0., 0.], [-1.05, 0., 0.], [-7., 0., 0.], [1.05, 0., 0.]]
        };
        let template = json!({
            "shape":"shape.json", "box_lengths":[20.,20.,20.],
            "boundary":if periodic {json!({"kind":"periodic"})}
                else {json!({"kind":"spherical","radius":10.})},
            "seed":13579, "depletant_radius":0.2, "reservoir_density":0.,
            "poisson_lambda_ratio":4., "global_probability":0.3,
            "gca_probability":if periodic {0.} else {1.},
            "center_shift_probability":if periodic {0.} else {1.},
            "local_translation_std_A":0.1, "local_small_angle_std_degrees":2.,
            "seed_labels":[3,1], "metadata":{"source":"seeded free preparation test"},
            "initial_poses":[
                {"position":positions[0],"orientation":[1.,0.,0.,0.]},
                {"position":positions[1],"orientation":[0.5,0.5,0.5,0.5]},
                {"position":positions[2],"orientation":[1.,0.,0.,0.]},
                {"position":positions[3],"orientation":[0.,1.,0.,0.]}
            ]
        });
        let fixture = Self { root, template };
        fixture.write_template()?;
        Ok(fixture)
    }
    fn write_template(&self) -> Result<()> {
        fs::write(self.root.join("config.json"), self.template.to_string())?;
        Ok(())
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
    fn fails_before_output(&self, name: &str, extra: &[&str]) -> Result<()> {
        let output = self.invoke(name, &self.root.join("config.json"), 2, extra)?;
        assert!(
            !output.status.success(),
            "invalid seeded preparation accepted: {name}"
        );
        assert!(
            !self.root.join(name).exists(),
            "failed preparation created output: {name}"
        );
        Ok(())
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
fn assert_total_density(config: &Value, total: usize, periodic: bool) {
    let volume = if periodic {
        config["box_lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .product()
    } else {
        4. * std::f64::consts::PI * config["boundary"]["radius"].as_f64().unwrap().powi(3) / 3.
    };
    assert!((volume * CONCENTRATION * UM_TO_A3 / total as f64 - 1.).abs() < 2e-12);
    assert_eq!(config["initial_poses"].as_array().unwrap().len(), total);
}
fn assert_retained_seed(config: &Value, template: &Value) {
    assert_eq!(config["seed_labels"], json!([0, 1]));
    // Unsorted, noncontiguous input indices are compacted in original body order.
    assert_eq!(config["initial_poses"][0], template["initial_poses"][1]);
    assert_eq!(config["initial_poses"][1], template["initial_poses"][3]);
}
fn assert_free_envelopes_separated(config: &Value, periodic: bool) {
    let poses = config["initial_poses"].as_array().unwrap();
    let length = config["box_lengths"][0].as_f64().unwrap();
    for i in 2..poses.len() {
        for j in 0..i {
            let distance2: f64 = (0..3)
                .map(|axis| {
                    let delta = poses[i]["position"][axis].as_f64().unwrap()
                        - poses[j]["position"][axis].as_f64().unwrap();
                    let delta = if periodic {
                        delta - (delta / length).round() * length
                    } else {
                        delta
                    };
                    delta * delta
                })
                .sum();
            assert!(
                distance2 > 2.4_f64.powi(2),
                "free body {i} overlaps envelope of body {j}"
            );
        }
    }
}

#[test]
fn preserves_noncontiguous_contacting_seed_adds_free_and_uses_total_density() -> Result<()> {
    let f = Fixture::new("preserve", false)?;
    f.run(
        "run",
        &["--free-tetramers", "3", "--concentration-um", "1000000"],
    )?;
    let config = f.initialized("run")?;
    assert_total_density(&config, 5, false);
    assert_retained_seed(&config, &f.template);
    assert_free_envelopes_separated(&config, false);
    // Seed separation is 2.1 A: hard cores do not overlap, depletion envelopes do.
    assert_eq!(f.read("run", "summary.json")?["bodies"], 5);
    assert_eq!(
        f.read("run", "provenance/template-config.json")?,
        f.template
    );
    assert_eq!(config["metadata"]["preparation_equilibrated"], false);
    let preparation = &config["metadata"]["free_tetramer_initialization"];
    assert_eq!(preparation["count"], 3);
    assert_eq!(preparation["seed_count"], 2);
    assert_eq!(preparation["total_count"], 5);
    assert_eq!(preparation["preserved_seed_source_indices"], json!([1, 3]));
    Ok(())
}

#[test]
fn zero_and_one_free_tetramers_are_valid_with_a_seed() -> Result<()> {
    let f = Fixture::new("small", false)?;
    for free in [0, 1] {
        let name = format!("free-{free}");
        f.run(
            &name,
            &[
                "--free-tetramers",
                &free.to_string(),
                "--concentration-um",
                "1000000",
            ],
        )?;
        let config = f.initialized(&name)?;
        assert_total_density(&config, 2 + free, false);
        assert_retained_seed(&config, &f.template);
        assert_free_envelopes_separated(&config, false);
    }
    Ok(())
}

#[test]
fn explicit_discard_replaces_seed_and_requires_free_count() -> Result<()> {
    let f = Fixture::new("discard", false)?;
    f.run(
        "run",
        &[
            "--free-tetramers",
            "3",
            "--discard-seed",
            "--concentration-um",
            "1000000",
        ],
    )?;
    let config = f.initialized("run")?;
    assert_total_density(&config, 3, false);
    assert_eq!(config["seed_labels"], json!([]));
    assert_ne!(config["initial_poses"][0], f.template["initial_poses"][1]);
    assert_eq!(
        f.read("run", "provenance/template-config.json")?,
        f.template
    );
    f.fails_before_output("missing-count", &["--discard-seed"])?;
    f.fails_before_output(
        "discard-too-few",
        &[
            "--free-tetramers",
            "1",
            "--discard-seed",
            "--concentration-um",
            "1000000",
        ],
    )?;
    Ok(())
}

#[test]
fn malformed_seed_labels_fail_before_output_creation() -> Result<()> {
    let mut f = Fixture::new("invalid-labels", false)?;
    for (name, labels) in [
        ("duplicate", json!([1, 1])),
        ("out-of-range", json!([1, 4])),
    ] {
        f.template["seed_labels"] = labels;
        f.write_template()?;
        f.fails_before_output(
            name,
            &["--free-tetramers", "3", "--concentration-um", "1000000"],
        )?;
    }
    Ok(())
}

#[test]
fn core_overlapping_or_outside_wall_seed_is_not_repaired_silently() -> Result<()> {
    let mut f = Fixture::new("invalid-geometry", false)?;
    f.template["initial_poses"][3]["position"] = json!([0., 0., 0.]);
    f.write_template()?;
    f.fails_before_output(
        "core-overlap",
        &["--free-tetramers", "2", "--concentration-um", "1000000"],
    )?;
    // Original sphere of radius 10 contains this seed. The requested radius for
    // two total bodies is smaller; preserving these coordinates must fail.
    f.template["initial_poses"][1]["position"] = json!([8.5, 0., 0.]);
    f.template["initial_poses"][3]["position"] = json!([6.4, 0., 0.]);
    f.write_template()?;
    f.fails_before_output(
        "new-wall",
        &["--free-tetramers", "0", "--concentration-um", "1000000"],
    )?;
    Ok(())
}

#[test]
fn periodic_seed_and_added_bodies_obey_the_new_cell_geometry() -> Result<()> {
    let f = Fixture::new("periodic", true)?;
    f.run(
        "run",
        &["--free-tetramers", "3", "--concentration-um", "1000000"],
    )?;
    let config = f.initialized("run")?;
    assert_total_density(&config, 5, true);
    assert_retained_seed(&config, &f.template);
    assert_free_envelopes_separated(&config, true);
    Ok(())
}

#[test]
fn seeded_preparation_checkpoint_resumes_exactly_without_reinitialization() -> Result<()> {
    let f = Fixture::new("resume", false)?;
    let flags = [
        "--free-tetramers",
        "3",
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
    assert_retained_seed(&f.initialized("full")?, &f.template);
    Ok(())
}

#[test]
fn periodic_seed_crossing_old_face_keeps_its_internal_minimum_image_geometry() -> Result<()> {
    let mut f = Fixture::new("periodic-crossing", true)?;
    f.template["initial_poses"][1]["position"] = json!([19., 3., 3.]);
    f.template["initial_poses"][3]["position"] = json!([1.2, 3., 3.]);
    f.write_template()?;
    f.run(
        "run",
        &["--free-tetramers", "1", "--concentration-um", "1000000"],
    )?;
    let config = f.initialized("run")?;
    assert_total_density(&config, 3, true);
    assert_eq!(config["seed_labels"], json!([0, 1]));
    for (i, old) in [(0, 1), (1, 3)] {
        assert_eq!(
            config["initial_poses"][i]["orientation"],
            f.template["initial_poses"][old]["orientation"]
        );
    }
    let length = config["box_lengths"][0].as_f64().unwrap();
    for axis in 0..3 {
        let delta = config["initial_poses"][1]["position"][axis]
            .as_f64()
            .unwrap()
            - config["initial_poses"][0]["position"][axis]
                .as_f64()
                .unwrap();
        let minimum_image = delta - (delta / length).round() * length;
        let expected = if axis == 0 { 2.2 } else { 0. };
        assert!((minimum_image - expected).abs() < 1e-12);
    }
    assert_free_envelopes_separated(&config, true);
    Ok(())
}

#[test]
fn periodic_winding_seed_is_rejected_instead_of_changing_its_geometry() -> Result<()> {
    let mut f = Fixture::new("periodic-winding", true)?;
    f.template["seed_labels"] = json!([0, 1, 3]);
    for (index, x) in [(0, 0.), (1, 7.), (3, 14.)] {
        f.template["initial_poses"][index]["position"] = json!([x, 3., 3.]);
    }
    f.write_template()?;
    f.fails_before_output(
        "winding",
        &["--free-tetramers", "3", "--concentration-um", "1000000"],
    )?;
    Ok(())
}

/// Check the real native fragment, whose enclosing spheres overlap strongly,
/// without running the physical dynamics or drawing a depletant cloud.
#[test]
fn repaired_native_seed_is_retained_at_requested_total_concentration() -> Result<()> {
    use tetramer_mc::{
        geometry::{Placed, Shape, SphereTree},
        initialization::FreeTetramerStart,
        math::{norm, sub},
        simulation::Config,
        spherical::Container,
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("examples/spherical-reciprocal-seeded.json");
    let original: Config = serde_json::from_slice(&fs::read(&path)?)?;
    let shape: Shape =
        serde_json::from_slice(&fs::read(path.parent().unwrap().join(&original.shape))?)?;
    let tree = SphereTree::new(shape)?;
    let mut config = original.clone();
    FreeTetramerStart {
        count: 4,
        concentration_um: 106.8,
        seed: Some(20260924),
    }
    .apply(&mut config, &tree)?;
    assert_eq!(config.initial_poses.len(), 12);
    assert_eq!(config.seed_labels, (0..8).collect::<Vec<_>>());
    assert_eq!(
        serde_json::to_value(&config.initial_poses[..8])?,
        serde_json::to_value(&original.initial_poses[..8])?
    );
    let radius = config.boundary.radius().unwrap();
    assert!((radius - 354.478713257288).abs() < 1e-9);
    let wall = Container::new(radius, &tree)?;
    for (i, &pose) in config.initial_poses.iter().enumerate() {
        assert!(wall.contains(pose));
        for &other in &config.initial_poses[..i] {
            assert!(!tree.overlaps(&Placed::new(pose), &Placed::new(other)));
            if i >= 8 {
                assert!(
                    norm(sub(pose.position, other.position))
                        > 2. * (tree.bound + config.depletant_radius)
                );
            }
        }
    }
    Ok(())
}
