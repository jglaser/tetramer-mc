//! The selector is an elementary physical kernel, before existing bias gates.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    geometry::{Shape, SphereTree},
    math::{Pose, Vec3},
    simulation::{Config, Method, RunOptions, run},
    spherical::{self, Container, HalfTurn},
};

fn scrub(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.retain(|k, _| !k.ends_with("seconds"));
            for v in map.values_mut() {
                scrub(v);
            }
        }
        Value::Array(a) => {
            for v in a {
                scrub(v);
            }
        }
        _ => {}
    }
}
struct Fixture {
    root: PathBuf,
    config: Value,
}
impl Fixture {
    fn new(name: &str) -> Result<Self> {
        let root =
            std::env::temp_dir().join(format!("conditional-axis-{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":0.35}]}).to_string(),
        )?;
        let pose = |x, y| json!({"position":[x,y,0.],"orientation":[1.,0.,0.,0.]});
        let config = json!({"shape":"shape.json","box_lengths":[6.,6.,6.],
            "boundary":{"kind":"spherical","radius":3.},"seed":812377,
            "depletant_radius":0.9,"reservoir_density":0.2,"poisson_lambda_ratio":4.,
            "global_probability":0.,"local_translation_std_A":0.1,"local_small_angle_std_degrees":1.,
            "gca_probability":1.,"center_shift_probability":1.,
            "gca_axis":{"score_floor":0.05,"max_candidates":8,"uniform_axis_weight":0.25},
            "endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25},
            "initial_poses":[pose(1.5,0.),pose(0.,1.5),pose(-1.5,0.),pose(0.,-1.5)]});
        Ok(Self { root, config })
    }
    fn run(&self, name: &str, sweeps: u64, resume: Option<&str>) -> Result<Value> {
        fs::write(self.root.join("config.json"), self.config.to_string())?;
        run(RunOptions {
            config: self.root.join("config.json"),
            model: None,
            method: Method::LocalUniform,
            out: self.root.join(name),
            sweeps,
            sample_every: 1,
            resume: resume.map(|n| self.root.join(n).join("checkpoint.json")),
            write_gsd: false,
            record_moves: true,
        })
    }
    fn read(&self, folder: &str, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(folder).join(name),
        )?)?)
    }
    fn rows(&self, folder: &str) -> Result<Vec<Value>> {
        fs::read_to_string(self.root.join(folder).join("moves.jsonl"))?
            .lines()
            .map(|s| Ok(serde_json::from_str(s)?))
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn selector_nulls_bias_replay_and_restart() -> Result<()> {
    let mut f = Fixture::new("replay")?;
    f.config["assembly_bias"] = json!({"values":[0.,1.,2.,3.]});
    let summary = f.run("full", 96, None)?;
    f.run("part", 40, None)?;
    f.run("resumed", 96, Some("part"))?;
    assert_eq!(
        f.read("full", "checkpoint.json")?,
        f.read("resumed", "checkpoint.json")?
    );
    assert_eq!(summary["counts"]["gca_axis"]["attempted"], 96);
    assert_eq!(summary["counts"]["gca"]["completed"], 96);
    let mut full: Vec<_> = f
        .rows("full")?
        .into_iter()
        .filter(|r| r["sweep"].as_u64().unwrap() > 40)
        .collect();
    let mut resumed = f.rows("resumed")?;
    for row in full.iter_mut().chain(resumed.iter_mut()) {
        scrub(row);
    }
    assert_eq!(full, resumed);
    let shape: Shape = serde_json::from_slice(&fs::read(f.root.join("shape.json"))?)?;
    let tree = SphereTree::new(shape)?;
    let wall = Container::new(3., &tree)?;
    let mut poses: Vec<Pose> = serde_json::from_value(f.config["initial_poses"].clone())?;
    let mut frames = fs::read_to_string(f.root.join("full/trajectory.jsonl"))?
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .collect::<Vec<_>>()
        .into_iter();
    assert_eq!(
        frames.next().unwrap()["poses"],
        serde_json::to_value(&poses)?
    );
    let mut failed = 0;
    let mut selector_rejected = 0;
    let mut exchanges = 0;
    for row in f.rows("full")? {
        match row["kind"].as_str().unwrap() {
            "local" | "global" => {
                let i = row["moving_index"].as_u64().unwrap() as usize;
                assert_eq!(serde_json::to_value(poses[i])?, row["old_pose"]);
                poses[i] = serde_json::from_value(row["retained_pose"].clone())?;
            }
            "gca" => {
                let stats = &row["axis_selector"];
                assert!(stats.is_object());
                if stats["forward"]["capped_failure"] == true
                    || stats["reverse"]["capped_failure"] == true
                {
                    failed += 1;
                    assert_eq!(row["accepted"], false);
                    assert_eq!(row["axis_selector_accepted"], false);
                    assert!(row["assembly_bias_decision"].is_null());
                }
                if row["axis_selector_accepted"] == false {
                    selector_rejected += 1;
                }
                if row["accepted"] == true {
                    let axis: Vec3 = serde_json::from_value(row["axis"].clone())?;
                    let turn = HalfTurn::new(axis)?;
                    for i in row["result"]["flipped_indices"].as_array().unwrap() {
                        let i = i.as_u64().unwrap() as usize;
                        poses[i] = turn.apply(poses[i]);
                    }
                    assert_eq!(row["axis_selector_accepted"], true);
                }
                assert_eq!(
                    row["retained_tag_contacts"],
                    if row["accepted"] == true {
                        stats["proposed_contacts"].clone()
                    } else {
                        stats["initial_contacts"].clone()
                    }
                );
                if row["accepted"] == false {
                    assert_eq!(row["retained_tag_lost"], json!([]));
                    assert_eq!(row["retained_tag_gained"], json!([]));
                }
                if row["completed_tag_exchange"] == true {
                    exchanges += 1;
                    assert_eq!(row["accepted"], true);
                }
            }
            "center_shift" => {
                if row["accepted"] == true {
                    let d: Vec3 = serde_json::from_value(row["result"]["displacement"].clone())?;
                    for pose in &mut poses {
                        for k in 0..3 {
                            pose.position[k] += d[k];
                        }
                    }
                }
                let frame = frames.next().unwrap();
                assert_eq!(frame["poses"], serde_json::to_value(&poses)?);
            }
            other => panic!("unexpected {other}"),
        }
        spherical::validate_state(&tree, &wall, &poses)?;
    }
    assert!(
        failed > 0 && selector_rejected > 0,
        "null attempts must be exercised"
    );
    assert_eq!(
        summary["counts"]["gca_axis"]["completed_tag_exchanges"],
        exchanges
    );
    assert_eq!(frames.next(), None);
    assert!(
        f.read("full", "manifest.json")?["gca_axis_protocol"]
            .as_str()
            .unwrap()
            .contains("exchange-v1")
    );
    Ok(())
}

#[test]
fn absent_and_null_setting_preserve_legacy_move_sequence() -> Result<()> {
    let mut f = Fixture::new("disabled")?;
    f.config.as_object_mut().unwrap().remove("gca_axis");
    f.run("absent", 16, None)?;
    f.config["gca_axis"] = Value::Null;
    f.run("null", 16, None)?;
    let mut a = f.rows("absent")?;
    let mut b = f.rows("null")?;
    for row in a.iter_mut().chain(b.iter_mut()) {
        scrub(row);
    }
    assert_eq!(a, b);
    assert!(
        f.read("absent", "checkpoint.json")?["counts"]
            .get("gca_axis")
            .is_none()
    );
    assert_eq!(
        f.read("absent", "checkpoint.json")?["poses"],
        f.read("null", "checkpoint.json")?["poses"]
    );
    Ok(())
}

#[test]
fn selector_requires_spherical_boundary_and_valid_options() -> Result<()> {
    let mut f = Fixture::new("validation")?;
    let mut config: Config = serde_json::from_value(f.config.clone())?;
    config.validate()?;
    config.boundary = tetramer_mc::simulation::Boundary::Periodic;
    config.gca_probability = 0.;
    config.center_shift_probability = 0.;
    assert!(config.validate().is_err());
    for (key, value) in [
        ("max_candidates", json!(0)),
        ("score_floor", json!(0.)),
        ("score_floor", json!(1.1)),
        ("uniform_axis_weight", json!(0.)),
        ("uniform_axis_weight", json!(-0.1)),
    ] {
        let old = f.config["gca_axis"][key].clone();
        f.config["gca_axis"][key] = value;
        assert!(
            serde_json::from_value::<Config>(f.config.clone())?
                .validate()
                .is_err()
        );
        f.config["gca_axis"][key] = old;
    }
    Ok(())
}
