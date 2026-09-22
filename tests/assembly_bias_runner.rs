//! Reconstruct actual retained configurations after every elementary update.
use anyhow::Result;
use hoomd_gsd::file_layer::{GsdFile, Mode};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    assembly_bias::{AssemblyBias, AssemblyBiasConfig, AssemblyBiasState},
    geometry::Shape,
    math::*,
    simulation::{Config, Method, RunOptions, hash_file, run},
    spherical::HalfTurn,
};

struct Fixture {
    root: PathBuf,
    config: Value,
    shape: Shape,
}
impl Fixture {
    fn new(label: &str) -> Result<Self> {
        let root =
            std::env::temp_dir().join(format!("assembly-bias-{label}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        let shape = json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":0.35}]});
        fs::write(root.join("shape.json"), shape.to_string())?;
        let p = |x, y| json!({"position":[x,y,0.],"orientation":[1.,0.,0.,0.]});
        let config = json!({"shape":"shape.json","box_lengths":[4.,4.,4.],
            "boundary":{"kind":"spherical","radius":2.},"seed":749902,
            "depletant_radius":0.5,"reservoir_density":0.2,"poisson_lambda_ratio":4.,
            "global_probability":0.5,"local_translation_std_A":0.35,
            "gca_probability":1.,"center_shift_probability":1.,
            "initial_poses":[p(-0.8,0.),p(0.,0.4),p(0.8,0.)]});
        Ok(Self {
            root,
            config,
            shape: serde_json::from_value(shape)?,
        })
    }
    fn run(&self, label: &str, sweeps: u64, resume: Option<&str>, sample: u64) -> Result<Value> {
        fs::write(self.root.join("config.json"), self.config.to_string())?;
        run(RunOptions {
            config: self.root.join("config.json"),
            model: self
                .root
                .join("model.json")
                .exists()
                .then(|| self.root.join("model.json")),
            method: if self.root.join("model.json").exists() {
                Method::Learned
            } else {
                Method::LocalUniform
            },
            out: self.root.join(label),
            sweeps,
            sample_every: sample,
            resume: resume.map(|p| self.root.join(p).join("checkpoint.json")),
            write_gsd: true,
            record_moves: true,
        })
    }
    fn read(&self, folder: &str, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(folder).join(name),
        )?)?)
    }
    fn rows(&self, folder: &str, name: &str) -> Result<Vec<Value>> {
        fs::read_to_string(self.root.join(folder).join(name))?
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
fn constant_bias_preserves_every_existing_physical_move_and_pose() -> Result<()> {
    let mut f = Fixture::new("constant")?;
    f.run("old", 64, None, 1)?;
    f.config["assembly_bias"] = json!({"values":[7.,7.,7.]});
    f.run("constant", 64, None, 1)?;
    let mut old = f.rows("old", "moves.jsonl")?;
    let mut new = f.rows("constant", "moves.jsonl")?;
    fn scrub(row: &mut Value) {
        match row {
            Value::Object(obj) => {
                obj.retain(|key, _| {
                    !key.ends_with("seconds")
                        && key != "physical_accepted"
                        && key != "assembly_bias_decision"
                });
                for value in obj.values_mut() {
                    scrub(value);
                }
            }
            Value::Array(rows) => {
                for row in rows {
                    scrub(row);
                }
            }
            _ => {}
        }
    }
    for row in &mut old {
        scrub(row);
    }
    for row in &mut new {
        scrub(row);
    }
    assert_eq!(old.len(), new.len());
    for (i, (a, b)) in old.iter().zip(&new).enumerate() {
        assert_eq!(a, b, "move {i}");
    }
    let a = f.read("old", "checkpoint.json")?;
    let b = f.read("constant", "checkpoint.json")?;
    assert_eq!(a["poses"], b["poses"]);
    assert_eq!(a["coordinate_wall_center"], b["coordinate_wall_center"]);
    let mut counts = b["counts"].clone();
    counts.as_object_mut().unwrap().remove("assembly_bias");
    assert_eq!(a["counts"], counts);
    assert!(a.get("assembly_bias").is_none());
    let covariance: Vec<Vec<f64>> = (0..6)
        .map(|i| (0..6).map(|j| if i == j { 0.4 } else { 0. }).collect())
        .collect();
    fs::write(f.root.join("model.json"),json!({"angular_length":1.,
        "anchors":[{"position":[0.95,0.,0.],"rotation":IDENTITY}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.],
        "shape_sha256":hash_file(&f.root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative"}).to_string())?;
    for (label, correlation) in [("independent", 0.), ("correlated", 0.9)] {
        f.config["frozen_posterior"] = json!({"probability":0.5,"correlation":correlation});
        f.config.as_object_mut().unwrap().remove("assembly_bias");
        f.run(&format!("{label}-old"), 64, None, 1)?;
        f.config["assembly_bias"] = json!({"values":[0.,0.,0.]});
        f.run(&format!("{label}-zero"), 64, None, 1)?;
        let mut original = f.rows(&format!("{label}-old"), "moves.jsonl")?;
        let mut biased = f.rows(&format!("{label}-zero"), "moves.jsonl")?;
        assert!(
            original
                .iter()
                .any(|row| row["proposal"]["kernel"] == "frozen-posterior"
                    && row["accepted"] == true)
        );
        for row in &mut original {
            scrub(row);
        }
        for row in &mut biased {
            scrub(row);
        }
        assert_eq!(original.len(), biased.len());
        for (i, (a, b)) in original.iter().zip(&biased).enumerate() {
            assert_eq!(a, b, "{label}, move {i}");
        }
    }
    Ok(())
}

#[test]
fn biased_elementary_replay_rejections_gsd_and_restart() -> Result<()> {
    let mut f = Fixture::new("replay")?;
    f.config["assembly_bias"] = json!({"values":[0.,2.,4.]});
    f.run("full", 192, None, 1)?;
    f.run("part", 80, None, 1)?;
    f.run("resume", 192, Some("part"), 1)?;
    assert_eq!(
        f.read("full", "checkpoint.json")?,
        f.read("resume", "checkpoint.json")?
    );
    let engine = AssemblyBias::new(
        AssemblyBiasConfig {
            values: vec![0., 2., 4.],
        },
        &f.shape,
        0.5,
        3,
        None,
    )?;
    let mut poses: Vec<Pose> = serde_json::from_value(f.config["initial_poses"].clone())?;
    let mut center = [0.; 3];
    let mut rejects = 0;
    let mut gca_rejects = 0;
    let frames = f.rows("full", "trajectory.jsonl")?;
    for row in f.rows("full", "moves.jsonl")? {
        let before = engine.score(&poses)?;
        let mut proposed = poses.clone();
        match row["kind"].as_str().unwrap() {
            "local" | "global" => {
                let i = row["moving_index"].as_u64().unwrap() as usize;
                assert_eq!(serde_json::to_value(poses[i])?, row["old_pose"]);
                if !row["proposed_pose"].is_null() {
                    proposed[i] = serde_json::from_value(row["proposed_pose"].clone())?;
                }
            }
            "gca" => {
                let axis: Vec3 = serde_json::from_value(row["axis"].clone())?;
                let turn = HalfTurn::new(axis)?;
                for i in row["result"]["flipped_indices"].as_array().unwrap() {
                    let i = i.as_u64().unwrap() as usize;
                    proposed[i] = turn.apply(proposed[i]);
                }
            }
            "center_shift" => {
                let displacement: Vec3 =
                    serde_json::from_value(row["result"]["displacement"].clone())?;
                for p in &mut proposed {
                    p.position = add(p.position, displacement);
                }
            }
            _ => panic!("unexpected kernel"),
        }
        if row["physical_accepted"] == true {
            let decision = &row["assembly_bias_decision"];
            assert_eq!(
                serde_json::from_value::<AssemblyBiasState>(decision["before"].clone())?,
                before
            );
            let next = engine.score(&proposed)?;
            assert_eq!(
                serde_json::from_value::<AssemblyBiasState>(decision["proposed"].clone())?,
                next
            );
            assert_eq!(
                decision["log_acceptance"],
                json!((before.bias - next.bias).min(0.))
            );
            assert_eq!(decision["accepted"], row["accepted"]);
            if row["accepted"] == false {
                rejects += 1;
                if row["kind"] == "gca" {
                    gca_rejects += 1;
                }
            }
        } else {
            assert!(row["assembly_bias_decision"].is_null());
            assert_eq!(row["accepted"], false);
        }
        if row["accepted"] == true {
            poses = proposed;
            if row["kind"] == "center_shift" {
                center = sub(
                    center,
                    serde_json::from_value(row["result"]["displacement"].clone())?,
                );
            }
        }
        if row["kind"] == "local" || row["kind"] == "global" {
            assert_eq!(
                row["retained_pose"],
                serde_json::to_value(poses[row["moving_index"].as_u64().unwrap() as usize])?
            );
        }
        if row["kind"] == "center_shift" {
            let frame = &frames[row["sweep"].as_u64().unwrap() as usize];
            assert_eq!(frame["poses"], serde_json::to_value(&poses)?);
            assert_eq!(frame["coordinate_wall_center"], json!(center));
            assert_eq!(
                frame["assembly_bias"],
                serde_json::to_value(engine.score(&poses)?)?
            );
        }
    }
    assert!(rejects > 10, "no useful bias rejections exercised");
    assert!(gca_rejects > 0, "GCA rollback not exercised");
    let gsd = GsdFile::open(&f.root.join("full/trajectory.gsd"), Mode::Read)?;
    let b: Vec<f64> = gsd
        .iter_scalars(192, "log/tetramer_mc/assembly_bias")?
        .collect();
    let rw: Vec<f64> = gsd
        .iter_scalars(192, "log/tetramer_mc/assembly_log_reweight")?
        .collect();
    assert_eq!(b, rw);
    assert_eq!(b, vec![engine.score(&poses)?.bias]);
    // Neither an altered table nor an invented cached score may continue.
    let mut damaged = f.read("part", "checkpoint.json")?;
    damaged["assembly_bias_state"]["bias"] = json!(123.);
    fs::write(f.root.join("part/checkpoint.json"), damaged.to_string())?;
    assert!(f.run("bad-resume", 192, Some("part"), 1).is_err());
    Ok(())
}

#[test]
fn unsupported_auxiliary_modes_fail_closed_before_run() -> Result<()> {
    let mut f = Fixture::new("config")?;
    f.config["assembly_bias"] = json!({"values":[0.,0.,0.]});
    serde_json::from_value::<Config>(f.config.clone())?.validate()?;
    for mode in [
        "auxiliary_transport",
        "reversible_jump",
        "contact_memory",
        "conditional_closure",
        "atlas_transport",
        "atlas_mask",
    ] {
        let mut config = f.config.clone();
        config[mode] = json!({});
        assert!(
            serde_json::from_value::<Config>(config)?
                .validate()
                .is_err(),
            "accepted {mode}"
        );
    }
    Ok(())
}

#[test]
fn two_sphere_contact_occupancy_and_reweighting_match_exact_wall_integral() -> Result<()> {
    let mut f = Fixture::new("stationarity")?;
    f.config["initial_poses"].as_array_mut().unwrap().remove(1);
    f.config["reservoir_density"] = json!(0.);
    f.config["global_probability"] = json!(1.);
    f.config["assembly_bias"] = json!({"values":[0.,1.5]});
    f.run("sample", 8192, None, 4)?;
    // The separation measure is 4πr² times the overlap of two center balls.
    // Constants cancel; hard exclusion removes r < 0.7.
    let w: f64 = 1.65;
    let primitive =
        |r: f64| 16. * w.powi(3) * r.powi(3) / 3. - 3. * w.powi(2) * r.powi(4) + r.powi(6) / 6.;
    let contact = (primitive(1.7) - primitive(0.7)) / (primitive(2. * w) - primitive(0.7));
    let expected = contact * (-1.5_f64).exp() / (1. - contact + contact * (-1.5_f64).exp());
    let frames = f.rows("sample", "trajectory.jsonl")?;
    let observed: Vec<f64> = frames
        .iter()
        .skip(129)
        .map(|row| {
            if row["assembly_bias"]["largest_component_size"] == 2 {
                1.
            } else {
                0.
            }
        })
        .collect();
    let average = observed.iter().sum::<f64>() / observed.len() as f64;
    let batches: Vec<f64> = observed
        .chunks_exact(64)
        .map(|v| v.iter().sum::<f64>() / 64.)
        .collect();
    let mean = batches.iter().sum::<f64>() / batches.len() as f64;
    let se = (batches.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (batches.len() * (batches.len() - 1)) as f64)
        .sqrt();
    println!(
        "sphere biased contact frequency={average:.6}; exact={expected:.6}; batch SE={se:.6}; physical contact={contact:.6}"
    );
    assert!(
        (average - expected).abs() < (6. * se).max(0.025),
        "observed {average}, expected {expected}, batch SE {se}"
    );
    let reweighted = average * 1.5_f64.exp() / (1. - average + average * 1.5_f64.exp());
    assert!(
        (reweighted - contact).abs() < 0.07,
        "physical reweight: {reweighted} vs {contact}"
    );
    Ok(())
}
