//! Cross-language conditional-law and actual runner reverse-density checks.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::Path};
use tetramer_mc::{
    auxiliary::{self, AuxiliaryConfig},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::{Method, RunOptions, hash_file, run},
    spherical::HalfTurn,
};
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-10 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
#[test]
fn conditional_means_and_transported_densities_match_independent_python_fixture() -> Result<()> {
    let data: Value = serde_json::from_str(include_str!("fixtures/spherical_auxiliary.json"))?;
    let base = FrozenRelativePoseProposal::from_json_str_open(
        &data["model_data"].to_string(),
        [10.; 3],
        0.2,
        data["shape_hash"].as_str().unwrap(),
    )?;
    let config: AuxiliaryConfig = serde_json::from_value(data["auxiliary_settings"].clone())?;
    let extreme = AuxiliaryConfig {
        noise: f64::MIN_POSITIVE,
        shrinkage: 1e308,
        ..Default::default()
    };
    let probe: Vec<Pose> = serde_json::from_value(data["cases"][0]["poses_x"].clone())?;
    assert!(
        auxiliary::fit(&base, &probe, &extreme).is_err(),
        "conditional width must not underflow to zero"
    );
    let mut differing = 0;
    for case in data["cases"].as_array().unwrap() {
        let x: Vec<Pose> = serde_json::from_value(case["poses_x"].clone())?;
        let y: Vec<Pose> = serde_json::from_value(case["poses_y"].clone())?;
        let eta: Vec<[f64; 6]> = serde_json::from_value(case["eta"].clone())?;
        let fx = auxiliary::fit(&base, &x, &config)?;
        let fy = auxiliary::fit(&base, &y, &config)?;
        for (fit, name) in [(&fx, "fit_x"), (&fy, "fit_y")] {
            let ref_f: Vec<[f64; 6]> = serde_json::from_value(case[name]["f"].clone())?;
            let ref_s: Vec<f64> = serde_json::from_value(case[name]["s"].clone())?;
            assert_eq!(json!(fit.counts), case[name]["counts"]);
            for (a, b) in fit.center.iter().flatten().zip(ref_f.iter().flatten()) {
                near(*a, *b);
            }
            for (a, b) in fit.scale.iter().zip(ref_s) {
                near(*a, b);
            }
        }
        let ax = fx.coordinates(&eta)?;
        let ay = fy.coordinates(&eta)?;
        near(
            fx.log_density(&ax)?,
            case["conditional_logdensity_x"].as_f64().unwrap(),
        );
        near(
            fy.log_density(&ay)?,
            case["conditional_logdensity_y"].as_f64().unwrap(),
        );
        near(
            fx.transport_log_jacobian(&fy)?,
            case["transport_log_jacobian"].as_f64().unwrap(),
        );
        near(
            fy.log_density(&ay)? - fx.log_density(&ax)? + fx.transport_log_jacobian(&fy)?,
            0.,
        );
        let mx = auxiliary::model(&base, &x, &eta, &config)?;
        let my = auxiliary::model(&base, &y, &eta, &config)?;
        let i = case["moving_index"].as_u64().unwrap() as usize;
        let j = case["anchor_index"].as_u64().unwrap() as usize;
        let forward = mx.log_density(&y[i], &x[j])?;
        let reverse = my.log_density(&x[i], &y[j])?;
        near(forward, case["forward_logpdf"].as_f64().unwrap());
        near(reverse, case["reverse_logpdf"].as_f64().unwrap());
        near(
            reverse - forward,
            case["log_reverse_forward"].as_f64().unwrap(),
        );
        if (reverse - mx.log_density(&x[i], &x[j])?).abs() > 0.1 {
            differing += 1;
        }
        let shifted: Vec<_> = x
            .iter()
            .map(|p| Pose {
                position: add(p.position, [0.7, -0.2, 0.1]),
                ..*p
            })
            .collect();
        let fs = auxiliary::fit(&base, &shifted, &config)?;
        assert_eq!(fx.counts, fs.counts);
        for (a, b) in fx.center.iter().flatten().zip(fs.center.iter().flatten()) {
            near(*a, *b);
        }
    }
    assert!(differing >= 4, "fixture must detect stale reverse models");
    Ok(())
}
fn options(root: &Path, name: &str, sweeps: u64, resume: Option<&str>) -> RunOptions {
    RunOptions {
        config: root.join("config.json"),
        model: Some(root.join("model.json")),
        method: Method::Learned,
        out: root.join(name),
        sweeps,
        sample_every: 1,
        resume: resume.map(|n| root.join(n).join("checkpoint.json")),
        write_gsd: false,
        record_moves: true,
    }
}
#[test]
fn transported_runner_replays_reverse_models_and_resumes_full_joint_state() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-aux-runner-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir(&root)?;
    let fixture: Value = serde_json::from_str(include_str!("fixtures/spherical_auxiliary.json"))?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let mut model = fixture["model_data"].clone();
    model["shape_sha256"] = json!(hash_file(&root.join("shape.json"))?);
    fs::write(root.join("model.json"), model.to_string())?;
    let aux: AuxiliaryConfig = serde_json::from_value(fixture["auxiliary_settings"].clone())?;
    let config = json!({"shape":"shape.json","box_lengths":[8.,8.,8.],"boundary":{"kind":"spherical","radius":4.},
        "initial_poses":[{"position":[-1.1,0.,0.],"orientation":[1.,0.,0.,0.]},{"position":[1.1,0.,0.],"orientation":[1.,0.,0.,0.]}],
        "seed":98273,"depletant_radius":0.5,"reservoir_density":0.4,"gca_probability":1.,"center_shift_probability":1.,
        "learned_uniform_weight":0.2,"auxiliary_transport":aux,"endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25}});
    fs::write(root.join("config.json"), config.to_string())?;
    run(options(&root, "full", 32, None))?;
    run(options(&root, "part", 13, None))?;
    run(options(&root, "resume", 32, Some("part")))?;
    let read = |folder: &str| -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            root.join(folder).join("checkpoint.json"),
        )?)?)
    };
    assert_eq!(
        read("full")?,
        read("resume")?,
        "joint checkpoint including eta must continue exactly"
    );
    let frames: Vec<Value> = fs::read_to_string(root.join("full/trajectory.jsonl"))?
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut state: Vec<Pose> = serde_json::from_value(frames[0]["poses"].clone())?;
    let base = FrozenRelativePoseProposal::from_json_str_open(
        &model.to_string(),
        [10.; 3],
        0.2,
        model["shape_sha256"].as_str().unwrap(),
    )?;
    let mut verified = 0;
    let mut nontrivial = 0;
    for text in fs::read_to_string(root.join("full/moves.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(text)?;
        let sweep = row["sweep"].as_u64().unwrap() as usize;
        let eta: Vec<[f64; 6]> = serde_json::from_value(frames[sweep]["auxiliary_eta"].clone())?;
        match row["kind"].as_str().unwrap() {
            "gca" => {
                let transform = HalfTurn::new(serde_json::from_value(row["axis"].clone())?)?;
                for i in row["result"]["flipped_indices"].as_array().unwrap() {
                    let i = i.as_u64().unwrap() as usize;
                    state[i] = transform.apply(state[i]);
                }
            }
            "center_shift" => {
                let d: Vec3 = serde_json::from_value(row["result"]["displacement"].clone())?;
                for p in &mut state {
                    p.position = add(p.position, d);
                }
            }
            kind => {
                let i = row["moving_index"].as_u64().unwrap() as usize;
                if kind == "global" && !row["proposed_pose"].is_null() {
                    let new: Pose = serde_json::from_value(row["proposed_pose"].clone())?;
                    let j = row["proposal"]["anchor_index"].as_u64().unwrap() as usize;
                    let f = auxiliary::model(&base, &state, &eta, &aux)?;
                    let mut next = state.clone();
                    next[i] = new;
                    let r = auxiliary::model(&base, &next, &eta, &aux)?;
                    let log_ratio =
                        r.log_density(&state[i], &state[j])? - f.log_density(&new, &state[j])?;
                    near(
                        row["proposal"]["log_reverse_forward"].as_f64().unwrap(),
                        log_ratio,
                    );
                    if (r.log_density(&state[i], &state[j])?
                        - f.log_density(&state[i], &state[j])?)
                    .abs()
                        > 1e-4
                    {
                        nontrivial += 1;
                    }
                    verified += 1;
                }
                state[i] = serde_json::from_value(row["retained_pose"].clone())?;
            }
        }
    }
    assert!(
        verified > 20 && nontrivial > 5,
        "must exercise changing reverse model"
    );
    let final_state: Vec<Pose> = serde_json::from_value(read("full")?["poses"].clone())?;
    for (a, b) in state.iter().zip(final_state) {
        for d in 0..3 {
            near(a.position[d], b.position[d]);
        }
    }
    // An auxiliary checkpoint cannot silently lose the model state.
    let mut broken = read("part")?;
    broken["auxiliary_eta"] = Value::Null;
    fs::write(root.join("part/checkpoint.json"), broken.to_string())?;
    assert!(run(options(&root, "broken", 32, Some("part"))).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
