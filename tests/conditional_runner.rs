//! Audit the actual runner, including corrections on local/collective moves.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::Path};
use tetramer_mc::{
    conditional::{ConditionalConfig, ConditionalEngine, ConditionalState},
    math::*,
    simulation::{Method, RunOptions, hash_file, run},
    spherical::HalfTurn,
};

fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

fn options(root: &Path, name: &str, sweeps: u64, resume: Option<&str>) -> RunOptions {
    RunOptions {
        config: root.join("config.json"),
        model: None,
        method: Method::Learned,
        out: root.join(name),
        sweeps,
        sample_every: 1,
        resume: resume.map(|p| root.join(p).join("checkpoint.json")),
        write_gsd: false,
        record_moves: true,
    }
}

#[test]
fn runner_corrects_every_kernel_and_resumes_retained_latents() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-conditional-runner-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir(&root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let cfg = json!({
        "shape":"shape.json", "box_lengths":[12.,12.,12.],
        "boundary":{"kind":"spherical","radius":6.},
        "initial_poses":[
            {"position":[-2.2,0.,0.],"orientation":[1.,0.,0.,0.]},
            {"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]},
            {"position":[2.2,0.3,0.],"orientation":[1.,0.,0.,0.]}],
        "seed":983871, "depletant_radius":0.5,"reservoir_density":0.4,
        "gca_probability":1., "center_shift_probability":1., "global_probability":0.5,
        "local_translation_std_A":1.2, "local_small_angle_std_degrees":25.,
        "conditional_closure":{
            "k_max":3,"basin_activity":1.5,"covariance_exponent":6.,
            "penalty_strength":0.3,"covariance_floor":0.5,"covariance_ceiling":4.,
            "translation_scale_A":2.,"angular_length_A":1.,"pair_cutoff_A":4.8,
            "residual_scale":0.2,"uniform_weight":0.25,"fit_iterations":2,
            "initialization":"fixed_k_zero","initial_k":1},
        "endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25}});
    fs::write(root.join("config.json"), cfg.to_string())?;
    run(options(&root, "full", 64, None))?;
    run(options(&root, "part", 23, None))?;
    run(options(&root, "resume", 64, Some("part")))?;
    let read = |name: &str| -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            root.join(name).join("checkpoint.json"),
        )?)?)
    };
    assert_eq!(
        read("full")?,
        read("resume")?,
        "Full physical and auxiliary checkpoint must resume exactly"
    );
    let frames: Vec<Value> = fs::read_to_string(root.join("full/trajectory.jsonl"))?
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut poses: Vec<Pose> = serde_json::from_value(frames[0]["poses"].clone())?;
    let mut state: ConditionalState =
        serde_json::from_value(frames[0]["conditional_state"].clone())?;
    assert_eq!(state.k, 1);
    assert!(
        state.eta.iter().all(|e| *e == 0.),
        "Nonrandom initial model is retained until first sweep ends"
    );
    let settings: ConditionalConfig = serde_json::from_value(cfg["conditional_closure"].clone())?;
    let engine = ConditionalEngine::new(1., 6., &hash_file(&root.join("shape.json"))?, settings)?;
    let mut fit = engine.fit(&poses)?;
    let mut center = [0.; 3];
    let mut verified = [0; 4];
    let mut nonzero = 0;
    let mut refreshes = 0;
    for line in fs::read_to_string(root.join("full/moves.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let kind = row["kind"].as_str().unwrap();
        if kind == "conditional_refresh" {
            assert_eq!(json!(state), row["old_state"]);
            state = serde_json::from_value(row["state"].clone())?;
            let sweep = row["sweep"].as_u64().unwrap() as usize;
            assert_eq!(json!(state), frames[sweep]["conditional_state"]);
            assert_eq!(json!(poses), frames[sweep]["poses"]);
            for (a, b) in center.into_iter().zip(serde_json::from_value::<Vec3>(
                frames[sweep]["coordinate_wall_center"].clone(),
            )?) {
                near(a, b);
            }
            refreshes += 1;
            continue;
        }
        let mut next = poses.clone();
        let expected_q;
        let field;
        let bucket;
        if kind == "gca" {
            let turn = HalfTurn::new(serde_json::from_value(row["axis"].clone())?)?;
            for i in row["result"]["flipped_indices"].as_array().unwrap() {
                let i = i.as_u64().unwrap() as usize;
                next[i] = turn.apply(next[i]);
            }
            expected_q = 0.;
            field = &row;
            bucket = 2;
        } else if kind == "center_shift" {
            let d: Vec3 = serde_json::from_value(row["result"]["displacement"].clone())?;
            for p in &mut next {
                p.position = add(p.position, d);
            }
            if row["accepted"] == true {
                center = sub(center, d);
            }
            expected_q = 0.;
            field = &row;
            bucket = 3;
        } else {
            assert!(kind == "local" || kind == "global");
            let i = row["moving_index"].as_u64().unwrap() as usize;
            assert_eq!(json!(poses[i]), row["old_pose"]);
            if row["hard_valid"] != true {
                assert_eq!(json!(poses[i]), row["retained_pose"]);
                continue;
            }
            next[i] = serde_json::from_value(row["proposed_pose"].clone())?;
            field = &row["proposal"];
            assert_eq!(field["conditional_k"], json!(state.k));
            bucket = usize::from(kind == "global");
            expected_q = if kind == "global" {
                let j = field["anchor_index"].as_u64().unwrap() as usize;
                let forward = engine.model(&fit, &state)?;
                let reverse_fit = engine.fit(&next)?;
                let reverse = engine.model(&reverse_fit, &state)?;
                reverse.log_density(&poses[i], &poses[j])?
                    - forward.log_density(&next[i], &poses[j])?
            } else {
                0.
            };
            near(expected_q, field["log_reverse_forward"].as_f64().unwrap());
        }
        let next_fit = engine.fit(&next)?;
        let count = next_fit.log_probabilities[state.k] - fit.log_probabilities[state.k];
        near(
            count,
            field["conditional_count_log_ratio"].as_f64().unwrap(),
        );
        if bucket < 2 {
            near(
                expected_q + count,
                field["conditional_log_correction"].as_f64().unwrap(),
            );
            near(
                (expected_q + count + row["gate"]["log_weight"].as_f64().unwrap()).min(0.),
                row["log_acceptance"].as_f64().unwrap(),
            );
        }
        verified[bucket] += 1;
        nonzero += usize::from(count.abs() > 1e-5);
        if row["accepted"] == true {
            poses = next;
            fit = next_fit;
        }
    }
    assert!(
        verified.iter().all(|n| *n > 5),
        "all four kernel types must be audited: {verified:?}"
    );
    assert!(
        nonzero > 10,
        "fixture must exercise state-dependent count corrections"
    );
    assert_eq!(refreshes, 64);
    let final_checkpoint = read("full")?;
    assert_eq!(json!(poses), final_checkpoint["poses"]);
    assert_eq!(json!(state), final_checkpoint["conditional_state"]);
    let mut broken = read("part")?;
    broken["conditional_state"] = Value::Null;
    fs::write(root.join("part/checkpoint.json"), broken.to_string())?;
    assert!(
        run(options(&root, "broken", 64, Some("part"))).is_err(),
        "Missing latent state cannot silently restart"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
