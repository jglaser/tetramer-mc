//! Full runner: inherited map, many-body gate, complete logging, bias and restart.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    geometry::{Shape, SphereTree},
    math::*,
    simulation::{Config, Method, RunOptions, hash_file, run},
    spherical::{self, Container},
};
struct Fixture {
    root: PathBuf,
    config: Value,
}
impl Fixture {
    fn new(name: &str) -> Result<Self> {
        let root =
            std::env::temp_dir().join(format!("cluster-phase-{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":0.35}]}).to_string(),
        )?;
        let cov = |v: f64| {
            (0..6)
                .map(|i| {
                    (0..6)
                        .map(|j| if i == j { v } else { 0. })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let atlas = json!({"angular_length":1.,"anchors":[{"position":[0.9,0.,0.],"rotation":IDENTITY},{"position":[0.,0.9,0.],"rotation":IDENTITY}],
            "means":vec![[0.;6];2],"covariances":[cov(0.2),cov(0.4)],"weights":[0.4,0.6],"shape_sha256":hash_file(&root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative"});
        fs::write(root.join("model.json"), atlas.to_string())?;
        let p = |x, y| json!({"position":[x,y,0.],"orientation":[1.,0.,0.,0.]});
        let config = json!({"shape":"shape.json","box_lengths":[6.,6.,6.],"boundary":{"kind":"spherical","radius":3.},
            "seed":938712,"depletant_radius":0.5,"reservoir_density":0.2,"poisson_lambda_ratio":4.,
            "global_probability":0.2,"learned_uniform_weight":0.1,"gca_probability":1.,"center_shift_probability":1.,
            "local_translation_std_A":0.1,"local_small_angle_std_degrees":3.,
            "cluster_phase":{"duration":1.,"dimer_rate":2.,"trimer_rate":1.,"transport_probability":0.6,"correlation":0.9,
                "local_translation_std_A":0.25,"local_small_angle_std_degrees":15.},
            "endpoint_gate":{"max_cells":31,"max_depth":4,"min_width":0.1},
            "initial_poses":[p(-0.8,0.),p(0.,0.1),p(0.8,0.),p(1.1,1.1)]});
        Ok(Self { root, config })
    }
    fn run(&self, name: &str, n: u64, resume: Option<&str>) -> Result<Value> {
        fs::write(self.root.join("config.json"), self.config.to_string())?;
        run(RunOptions {
            config: self.root.join("config.json"),
            model: Some(self.root.join("model.json")),
            method: Method::Learned,
            out: self.root.join(name),
            sweeps: n,
            sample_every: 1,
            resume: resume.map(|x| self.root.join(x).join("checkpoint.json")),
            write_gsd: false,
            record_moves: true,
        })
    }
    fn read(&self, name: &str, file: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(name).join(file),
        )?)?)
    }
    fn rows(&self, name: &str, file: &str) -> Result<Vec<Value>> {
        fs::read_to_string(self.root.join(name).join(file))?
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
fn scrub(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.retain(|k, _| !k.ends_with("seconds"));
            for x in m.values_mut() {
                scrub(x);
            }
        }
        Value::Array(a) => {
            for x in a {
                scrub(x);
            }
        }
        _ => {}
    }
}
#[test]
fn phase_retains_attempts_bias_and_exact_restart_and_replay() -> Result<()> {
    check_restart_and_replay(false)
}
#[test]
fn contact_anchor_phase_retains_bias_restart_and_independent_log_reconstruction() -> Result<()> {
    check_restart_and_replay(true)
}
fn check_restart_and_replay(contact_anchors: bool) -> Result<()> {
    let mut f = Fixture::new(if contact_anchors {
        "contact-restart"
    } else {
        "restart"
    })?;
    if contact_anchors {
        f.config["cluster_phase"]["transport_charts"] = json!("members");
        f.config["cluster_phase"]["anchor_count"] = json!(2);
        f.config["cluster_phase"]["anchor_contact_uniform_probability"] = json!(0.2);
    }
    f.config["assembly_bias"] = json!({"values":[0.,0.2,0.4,0.6]});
    let summary = f.run("full", 120, None)?;
    f.run("part", 47, None)?;
    f.run("resumed", 120, Some("part"))?;
    assert_eq!(
        f.read("full", "checkpoint.json")?,
        f.read("resumed", "checkpoint.json")?
    );
    let mut full: Vec<_> = f
        .rows("full", "moves.jsonl")?
        .into_iter()
        .filter(|r| r["sweep"].as_u64().unwrap() > 47)
        .collect();
    let mut resumed = f.rows("resumed", "moves.jsonl")?;
    for r in full.iter_mut().chain(resumed.iter_mut()) {
        scrub(r);
    }
    assert_eq!(full, resumed);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(
        f.root.join("shape.json"),
    )?)?)?;
    let wall = Container::new(3., &tree)?;
    let mut poses: Vec<Pose> = serde_json::from_value(f.config["initial_poses"].clone())?;
    let frames = f.rows("full", "trajectory.jsonl")?;
    let rows = f.rows("full", "moves.jsonl")?;
    let mut ev = 0;
    let mut rejects = 0;
    let mut transported = 0;
    let mut anchor_corrected = 0;
    let mut uniform = 0;
    let mut phase_times = vec![0.; 121];
    for (k, row) in rows.iter().enumerate() {
        let sweep = row["sweep"].as_u64().unwrap() as usize;
        match row["kind"].as_str().unwrap() {
            "local" | "global" => {
                let i = row["moving_index"].as_u64().unwrap() as usize;
                assert_eq!(json!(poses[i]), row["old_pose"]);
                poses[i] = serde_json::from_value(row["retained_pose"].clone())?;
            }
            "gca" => {
                if row["accepted"] == true {
                    let turn =
                        spherical::HalfTurn::new(serde_json::from_value(row["axis"].clone())?)?;
                    for i in row["result"]["flipped_indices"].as_array().unwrap() {
                        let i = i.as_u64().unwrap() as usize;
                        poses[i] = turn.apply(poses[i]);
                    }
                }
            }
            "center_shift" => {
                if row["accepted"] == true {
                    let d: Vec3 = serde_json::from_value(row["result"]["displacement"].clone())?;
                    for p in &mut poses {
                        p.position = add(p.position, d);
                    }
                }
            }
            "cluster_event" => {
                ev += 1;
                rejects += usize::from(row["accepted"] == false);
                let t = row["event_time"].as_f64().unwrap();
                assert!(t > phase_times[sweep] && t < 1.);
                phase_times[sweep] = t;
                assert_eq!(row["selection_log_correction"], 0.);
                let ids: Vec<usize> = serde_json::from_value(row["members"].clone())?;
                assert_eq!(
                    json!(ids.iter().map(|&i| poses[i]).collect::<Vec<_>>()),
                    row["old_poses"]
                );
                if let Some(a) = row["proposal"]["anchor_index"]
                    .as_u64()
                    .or_else(|| row["proposal"]["primary_anchor"].as_u64())
                {
                    assert!(!ids.contains(&(a as usize)));
                    transported += 1;
                }
                if contact_anchors {
                    let info = &row["proposal"];
                    if info["branch"] == "uniform" {
                        uniform += 1;
                        assert!(info.get("primary_anchor").is_none());
                        assert!(info.get("anchor_pool").is_none());
                        assert_eq!(info["log_reverse_forward"], 0.);
                    }
                    if let Some(primary) = info["primary_anchor"].as_u64() {
                        // Reconstruct from center distances, independently of
                        // ContactGraph::anchor_probabilities and graph updates.
                        let probability = |state: &[Pose]| {
                            let contacts: Vec<usize> = (0..state.len())
                                .map(|j| {
                                    if ids.contains(&j) {
                                        0
                                    } else {
                                        ids.iter()
                                            .filter(|&&i| {
                                                norm(sub(state[i].position, state[j].position))
                                                    < 1.7
                                            })
                                            .count()
                                    }
                                })
                                .collect();
                            let total: usize = contacts.iter().sum();
                            let m = (state.len() - ids.len()) as f64;
                            if total == 0 {
                                1. / m
                            } else {
                                0.2 / m + 0.8 * contacts[primary as usize] as f64 / total as f64
                            }
                        };
                        let qx = probability(&poses);
                        assert!(
                            (qx - info["anchor_forward_probability"].as_f64().unwrap()).abs()
                                < 1e-12
                        );
                        if row["hard_valid"] == true
                            && row["internal_contact_graph_preserved"] == true
                        {
                            let mut proposed = poses.clone();
                            for (&i, p) in ids.iter().zip(row["proposed_poses"].as_array().unwrap())
                            {
                                proposed[i] = serde_json::from_value(p.clone())?;
                            }
                            let qy = probability(&proposed);
                            assert!(
                                (qy - info["anchor_reverse_probability"].as_f64().unwrap()).abs()
                                    < 1e-12
                            );
                            let correction = qy.ln() - qx.ln();
                            anchor_corrected += usize::from(correction.abs() > 1e-8);
                            assert!(
                                (correction - info["anchor_log_reverse_forward"].as_f64().unwrap())
                                    .abs()
                                    < 1e-12
                            );
                            let total =
                                correction + info["map_log_reverse_forward"].as_f64().unwrap();
                            assert!(
                                (total - info["log_reverse_forward"].as_f64().unwrap()).abs()
                                    < 1e-12
                            );
                            let expected =
                                (total + row["gate"]["log_weight"].as_f64().unwrap()).min(0.);
                            assert!(
                                (expected - row["log_acceptance"].as_f64().unwrap()).abs() < 1e-12
                            );
                        } else {
                            assert!(info["anchor_reverse_probability"].is_null());
                            assert!(info["anchor_log_reverse_forward"].is_null());
                            assert!(info["log_reverse_forward"].is_null());
                        }
                    }
                }
                for (&i, p) in ids.iter().zip(row["retained_poses"].as_array().unwrap()) {
                    poses[i] = serde_json::from_value(p.clone())?;
                }
                if row["accepted"] == true {
                    assert_eq!(row["internal_contact_graph_preserved"], true);
                    assert_eq!(row["physical_accepted"], true);
                    assert!(row["assembly_bias_decision"].is_object());
                }
            }
            "cluster_phase_end" => {
                assert_eq!(row["duration"], 1.);
                assert_eq!(row["event_time"].as_f64().unwrap(), phase_times[sweep]);
            }
            other => panic!("unexpected {other}"),
        }
        spherical::validate_state(&tree, &wall, &poses)?;
        if k + 1 == rows.len() || rows[k + 1]["sweep"] != row["sweep"] {
            assert_eq!(frames[sweep]["poses"], json!(poses));
        }
    }
    assert!(ev > 100 && rejects > 0 && transported > 0);
    if contact_anchors {
        assert!(anchor_corrected > 0 && uniform > 0);
    }
    assert_eq!(summary["counts"]["cluster_phase"]["events"], ev);
    assert_eq!(summary["counts"]["cluster_phase"]["phases"], 120);
    assert_eq!(
        summary["counts"]["cluster_phase"]["horizon_stops"]
            .as_u64()
            .unwrap()
            + summary["counts"]["cluster_phase"]["zero_rate_stops"]
                .as_u64()
                .unwrap(),
        120
    );
    Ok(())
}
#[test]
fn absent_null_and_zero_duration_preserve_ordinary_rng_sequences() -> Result<()> {
    let mut f = Fixture::new("disabled")?;
    let cfg = f.config["cluster_phase"].clone();
    f.config.as_object_mut().unwrap().remove("cluster_phase");
    f.run("absent", 20, None)?;
    f.config["cluster_phase"] = Value::Null;
    f.run("null", 20, None)?;
    f.config["cluster_phase"] = cfg;
    f.config["cluster_phase"]["duration"] = json!(0.);
    f.run("zero", 20, None)?;
    let mut all = Vec::new();
    for name in ["absent", "null", "zero"] {
        let mut rows = f.rows(name, "moves.jsonl")?;
        for r in &mut rows {
            scrub(r);
        }
        all.push(rows);
    }
    assert_eq!(all[0], all[1]);
    assert_eq!(all[0], all[2]);
    assert!(
        f.read("zero", "checkpoint.json")?["counts"]
            .get("cluster_phase")
            .is_none()
    );
    Ok(())
}
#[test]
fn scope_and_parameter_guards_are_explicit() -> Result<()> {
    let f = Fixture::new("guards")?;
    let mut cfg: Config = serde_json::from_value(f.config.clone())?;
    cfg.validate()?;
    cfg.boundary = tetramer_mc::simulation::Boundary::Periodic;
    cfg.gca_probability = 0.;
    cfg.center_shift_probability = 0.;
    assert!(cfg.validate().is_err());
    for key in [
        "duration",
        "dimer_rate",
        "trimer_rate",
        "transport_probability",
        "local_translation_std_A",
    ] {
        let mut c = f.config.clone();
        c["cluster_phase"][key] = json!(-1.);
        assert!(serde_json::from_value::<Config>(c)?.validate().is_err());
    }
    Ok(())
}

#[test]
fn absent_and_null_contact_anchor_option_preserve_existing_member_rng_sequences() -> Result<()> {
    let mut f = Fixture::new("contact-disabled")?;
    f.config["cluster_phase"]["transport_charts"] = json!("members");
    f.config["cluster_phase"]["anchor_count"] = json!(2);
    f.run("absent", 30, None)?;
    f.config["cluster_phase"]["anchor_contact_uniform_probability"] = Value::Null;
    f.run("null", 30, None)?;
    // Raw-input provenance differs intentionally; all dynamical state and
    // counters must match after excluding only that source-document hash.
    let mut checkpoints = [
        f.read("absent", "checkpoint.json")?,
        f.read("null", "checkpoint.json")?,
    ];
    for checkpoint in &mut checkpoints {
        checkpoint.as_object_mut().unwrap().remove("config_sha256");
    }
    assert_eq!(checkpoints[0], checkpoints[1]);
    let mut a = f.rows("absent", "moves.jsonl")?;
    let mut b = f.rows("null", "moves.jsonl")?;
    for r in a.iter_mut().chain(b.iter_mut()) {
        scrub(r);
    }
    assert_eq!(a, b);
    Ok(())
}
#[test]
fn contact_anchor_parameter_and_chart_guards_are_explicit() -> Result<()> {
    let f = Fixture::new("contact-guards")?;
    for value in [0., -0.1, 1.01] {
        let mut c = f.config.clone();
        c["cluster_phase"]["transport_charts"] = json!("members");
        c["cluster_phase"]["anchor_contact_uniform_probability"] = json!(value);
        assert!(serde_json::from_value::<Config>(c)?.validate().is_err());
    }
    for value in [0.2, 1.] {
        let mut c = f.config.clone();
        c["cluster_phase"]["anchor_contact_uniform_probability"] = json!(value);
        assert!(
            serde_json::from_value::<Config>(c.clone())?
                .validate()
                .is_err(),
            "option must reject handle charts"
        );
        c["cluster_phase"]["transport_charts"] = json!("members");
        serde_json::from_value::<Config>(c)?.validate()?;
    }
    Ok(())
}
