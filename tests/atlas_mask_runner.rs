//! Actual masked runner: full-atlas control, both-density replay and exact restart.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::Path};
use tetramer_mc::{
    atlas_mask::{AtlasMaskConfig, AtlasMaskEngine, AtlasMaskState},
    atlas_transport::{AtlasTransportConfig, AtlasTransportEngine, AtlasTransportState},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::{Method, RunOptions, hash_file, run},
    spherical::HalfTurn,
};

fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn options(root: &Path, config: &str, name: &str, sweeps: u64, resume: Option<&str>) -> RunOptions {
    RunOptions {
        config: root.join(config),
        model: Some(root.join("model.json")),
        method: Method::Learned,
        out: root.join(name),
        sweeps,
        sample_every: 1,
        resume: resume.map(|s| root.join(s).join("checkpoint.json")),
        write_gsd: false,
        record_moves: true,
    }
}
fn rows(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn without_timings(mut value: Value) -> Value {
    if let Some(fields) = value.as_object_mut() {
        fields.retain(|k, _| !k.ends_with("_seconds"));
    }
    value
}

#[test]
fn full_mask_matches_control_and_sparse_masks_replay_and_resume() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("tetramer-atlas-mask-runner-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir(&root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let fixture: Value = serde_json::from_str(include_str!("fixtures/spherical_auxiliary.json"))?;
    let mut model = fixture["model_data"].clone();
    model["shape_sha256"] = json!(hash_file(&root.join("shape.json"))?);
    fs::write(root.join("model.json"), model.to_string())?;
    let mut cfg = json!({"shape":"shape.json","box_lengths":[8.,8.,8.],"boundary":{"kind":"spherical","radius":4.},
        "initial_poses":[{"position":[-1.1,0.,0.],"orientation":[1.,0.,0.,0.]},{"position":[1.1,0.,0.],"orientation":[1.,0.,0.,0.]}],
        "seed":98273,"depletant_radius":0.5,"reservoir_density":0.4,"gca_probability":1.,"center_shift_probability":1.,
        "learned_uniform_weight":0.2,"endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25}});
    cfg["atlas_transport"] = json!({"mean_gain":0.25,"covariance_gain":0.25,"weight_gain":0.25,
        "mean_noise":0.1,"covariance_noise":0.1,"weight_noise":0.1,"initialization":"reference"});
    fs::write(root.join("frozen.json"), cfg.to_string())?;
    cfg["atlas_mask"] = json!({"min_components":2,"max_components":2});
    fs::write(root.join("zero.json"), cfg.to_string())?;
    run(options(&root, "frozen.json", "frozen", 32, None))?;
    run(options(&root, "zero.json", "zero", 32, None))?;
    let frozen = rows(root.join("frozen/trajectory.jsonl"))?;
    let zero = rows(root.join("zero/trajectory.jsonl"))?;
    for (a, b) in frozen.iter().zip(&zero) {
        assert_eq!(
            a["poses"], b["poses"],
            "full mask must preserve each physical bit"
        );
        assert_eq!(a["coordinate_wall_center"], b["coordinate_wall_center"]);
        for key in ["local", "global", "gca", "center_shift"] {
            assert_eq!(a["counts"][key], b["counts"][key]);
        }
    }
    let old_moves = rows(root.join("frozen/moves.jsonl"))?;
    let new_moves = rows(root.join("zero/moves.jsonl"))?;
    let new_moves: Vec<_> = new_moves
        .iter()
        .filter(|m| m["kind"] != "atlas_mask_refresh")
        .collect();
    assert_eq!(old_moves.len(), new_moves.len());
    for (a, b) in old_moves.iter().zip(new_moves) {
        for key in [
            "kind",
            "proposed_pose",
            "retained_pose",
            "accepted",
            "gate",
            "log_acceptance",
        ] {
            assert_eq!(a[key], b[key], "full mask control mismatch for {key}");
        }
        assert_eq!(
            without_timings(a["result"].clone()),
            without_timings(b["result"].clone())
        );
    }
    cfg["atlas_transport"] = json!({"mean_gain":0.25,"covariance_gain":0.25,"weight_gain":0.25,
        "mean_noise":0.1,"covariance_noise":0.1,"weight_noise":0.1,"initialization":"reference"});
    cfg["atlas_mask"] = json!({"activity":0.9,"initial_full":false});
    fs::write(root.join("active.json"), cfg.to_string())?;
    run(options(&root, "active.json", "full", 192, None))?;
    run(options(&root, "active.json", "part", 19, None))?;
    run(options(&root, "active.json", "resume", 192, Some("part")))?;
    let checkpoint = |name: &str| -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            root.join(name).join("checkpoint.json"),
        )?)?)
    };
    assert_eq!(checkpoint("full")?, checkpoint("resume")?);
    let frames = rows(root.join("full/trajectory.jsonl"))?;
    let mut poses: Vec<Pose> = serde_json::from_value(frames[0]["poses"].clone())?;
    let mut state: AtlasTransportState = serde_json::from_value(frames[0]["atlas_state"].clone())?;
    let mut mask: AtlasMaskState = serde_json::from_value(frames[0]["atlas_mask_state"].clone())?;
    let base = FrozenRelativePoseProposal::from_json_str_open(
        &model.to_string(),
        [10.; 3],
        0.2,
        model["shape_sha256"].as_str().unwrap(),
    )?;
    let settings: AtlasTransportConfig = serde_json::from_value(cfg["atlas_transport"].clone())?;
    let mask_engine = AtlasMaskEngine::new(
        &base.component_weights(),
        serde_json::from_value::<AtlasMaskConfig>(cfg["atlas_mask"].clone())?,
    )?;
    let engine = AtlasTransportEngine::new(base.clone(), settings)?;
    let recovered = engine.model(&engine.fit(&poses)?, &state)?;
    for i in 0..12 {
        let mut p = poses[0];
        p.position = add(p.position, [0.01 * i as f64, 0., 0.]);
        near(
            base.log_density(&p, &poses[1])?,
            recovered.log_density(&p, &poses[1])?,
        );
    }
    let mut verified = 0;
    let mut nontrivial = 0;
    let mut refreshes = 0;
    let mut mask_refreshes = 0;
    let mut seen_counts = [0; 3];
    let mut center = [0.; 3];
    for row in rows(root.join("full/moves.jsonl"))? {
        match row["kind"].as_str().unwrap() {
            "atlas_mask_refresh" => {
                assert_eq!(json!(mask), row["old_state"]);
                mask = serde_json::from_value(row["state"].clone())?;
                mask_engine.validate_state(&mask)?;
                near(
                    mask_engine.log_probability(&mask)?,
                    row["log_probability"].as_f64().unwrap(),
                );
                let s = row["sweep"].as_u64().unwrap() as usize;
                assert_eq!(json!(mask), frames[s]["atlas_mask_state"]);
                mask_refreshes += 1;
                seen_counts[mask.labels.len()] += 1;
            }
            "atlas_refresh" => {
                assert_eq!(json!(state), row["old_state"]);
                state = serde_json::from_value(row["state"].clone())?;
                let s = row["sweep"].as_u64().unwrap() as usize;
                assert_eq!(json!(state), frames[s]["atlas_state"]);
                assert_eq!(json!(poses), frames[s]["poses"]);
                for (a, b) in center.into_iter().zip(serde_json::from_value::<Vec3>(
                    frames[s]["coordinate_wall_center"].clone(),
                )?) {
                    near(a, b);
                }
                refreshes += 1;
            }
            "gca" => {
                assert_eq!(row["accepted"], true);
                near(row["conditional_count_log_ratio"].as_f64().unwrap(), 0.);
                let turn = HalfTurn::new(serde_json::from_value(row["axis"].clone())?)?;
                for i in row["result"]["flipped_indices"].as_array().unwrap() {
                    let i = i.as_u64().unwrap() as usize;
                    poses[i] = turn.apply(poses[i]);
                }
            }
            "center_shift" => {
                assert_eq!(row["accepted"], true);
                let d: Vec3 = serde_json::from_value(row["result"]["displacement"].clone())?;
                for p in &mut poses {
                    p.position = add(p.position, d);
                }
                center = sub(center, d);
            }
            kind => {
                assert!(kind == "local" || kind == "global");
                let i = row["moving_index"].as_u64().unwrap() as usize;
                assert_eq!(json!(poses[i]), row["old_pose"]);
                if row["hard_valid"] == true {
                    let mut next = poses.clone();
                    next[i] = serde_json::from_value(row["proposed_pose"].clone())?;
                    let ratio = if kind == "global" {
                        let j = row["proposal"]["anchor_index"].as_u64().unwrap() as usize;
                        let forward = engine
                            .model(&engine.fit(&poses)?, &state)?
                            .weighted_subset(&mask.labels)?;
                        let reverse = engine
                            .model(&engine.fit(&next)?, &state)?
                            .weighted_subset(&mask.labels)?;
                        assert_eq!(row["proposal"]["atlas_mask_labels"], json!(mask.labels));
                        if let Some(label) = row["proposal"]["component_index"].as_u64() {
                            assert_eq!(
                                row["proposal"]["atlas_component_label"],
                                json!(mask.labels[label as usize])
                            );
                        }
                        let r = reverse.log_density(&poses[i], &poses[j])?
                            - forward.log_density(&next[i], &poses[j])?;
                        nontrivial += usize::from(
                            (reverse.log_density(&poses[i], &poses[j])?
                                - forward.log_density(&poses[i], &poses[j])?)
                            .abs()
                                > 1e-6,
                        );
                        verified += 1;
                        r
                    } else {
                        0.
                    };
                    near(
                        ratio,
                        row["proposal"]["log_reverse_forward"].as_f64().unwrap(),
                    );
                    near(
                        (ratio + row["gate"]["log_weight"].as_f64().unwrap()).min(0.),
                        row["log_acceptance"].as_f64().unwrap(),
                    );
                }
                poses[i] = serde_json::from_value(row["retained_pose"].clone())?;
            }
        }
    }
    assert!(
        verified > 5 && nontrivial > 3,
        "Need changing reverse fits: {verified}/{nontrivial}"
    );
    assert_eq!(refreshes, 192);
    assert_eq!(mask_refreshes, 192);
    assert!(
        seen_counts.iter().all(|&n| n > 2),
        "need all counts: {seen_counts:?}"
    );
    assert_eq!(json!(poses), checkpoint("full")?["poses"]);
    let mut broken = checkpoint("part")?;
    broken["atlas_mask_state"] = Value::Null;
    fs::write(root.join("part/checkpoint.json"), broken.to_string())?;
    assert!(run(options(&root, "active.json", "broken", 192, Some("part"))).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
