//! Actual conditional runner replay: lab/capture/anchor frames, full proposal
//! factors, exact Poisson acceptance, unchanged spectators and deterministic
//! restart. Independent equilibrium tests live in involution_depletion.rs.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    depletion,
    docking::{self, DockingConfig, DockingCounts, DockingMethod, DockingOptions},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::hash_file,
};

fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

fn matrix_error(a: Pose, b: Pose) -> f64 {
    rotation(a.orientation)
        .into_iter()
        .flatten()
        .zip(rotation(b.orientation).into_iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}

fn near_pose(a: Pose, b: Pose) {
    for (x, y) in a.position.into_iter().zip(b.position) {
        near(x, y);
    }
    assert!(matrix_error(a, b) < 3e-10);
}

fn rows(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}

fn read(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn physical(mut row: Value) -> Value {
    row.as_object_mut().unwrap().remove("sampler_cpu_seconds");
    row
}

fn rng(seed: u64, cycle: u64, attempt: usize, kind: &str) -> StdRng {
    let mut hash = Sha256::new();
    hash.update(b"tetramer-docking-rng-v1");
    hash.update(seed.to_le_bytes());
    hash.update(cycle.to_le_bytes());
    hash.update((attempt as u64).to_le_bytes());
    hash.update(kind.as_bytes());
    StdRng::from_seed(hash.finalize().into())
}

fn options(
    root: &Path,
    name: &str,
    method: DockingMethod,
    c: f64,
    cycles: u64,
    resume: Option<PathBuf>,
) -> DockingOptions {
    DockingOptions {
        config: root.join("input.json"),
        model: root.join("model.json"),
        out: root.join(name),
        cycles,
        sample_every: 1,
        method,
        correlation: c,
        resume,
    }
}

fn fixture(root: &Path) -> Result<(DockingConfig, Value)> {
    fs::create_dir_all(root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"asymmetric dumbbell","volume":3.0,
        "atoms":[{"center":[-0.45,0.,0.],"radius":0.65},
                 {"center":[0.5,0.,0.],"radius":0.8}]})
        .to_string(),
    )?;
    let fixture: Value = serde_json::from_str(include_str!("fixtures/spherical_auxiliary.json"))?;
    let mut model = fixture["model_data"].clone();
    model["shape_sha256"] = json!(hash_file(&root.join("shape.json"))?);
    fs::write(root.join("model.json"), model.to_string())?;
    let center = [7., -4., 3.];
    let config = json!({
        "shape":"shape.json",
        "fixed_poses":[
            Pose{position:add(center,[-1.4,0.,0.]),orientation:quaternion(cayley([0.3,0.4,-0.2]))},
            Pose{position:add(center,[2.,2.1,-0.2]),orientation:quaternion(cayley([-0.2,0.1,0.4]))}
        ],
        "initial_pose":Pose{position:add(center,[-0.3,-2.7,0.8]),orientation:quaternion(cayley([0.1,0.2,-0.3]))},
        "capture_center":center,"capture_radius":5.,"depletant_radius":0.6,
        "reservoir_density":0.4,"poisson_lambda_ratio":4.,
        "translation_steps":[0.1,0.5],"rotation_steps_deg":[5.,20.],
        "rotation_probability":0.5,"local_attempts_per_cycle":2,
        "uniform_probability":0.2,"seed":490863,
        "endpoint_gate":{"max_cells":31,"max_depth":5,"min_width":0.1},
        "metadata":{"purpose":"frame and balance regression"}
    });
    fs::write(root.join("input.json"), config.to_string())?;
    Ok((serde_json::from_value(config)?, model))
}

fn replay(
    root: &Path,
    name: &str,
    cfg: &DockingConfig,
    raw_model: &Value,
    method: DockingMethod,
    c: f64,
) -> Result<()> {
    let shape = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(
        root.join("shape.json"),
    )?)?)?;
    let fixed_before = cfg.fixed_poses.clone();
    let env = Environment {
        tree: &shape,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: (0..cfg.fixed_poses.len()).map(|i| (i, [0; 3])).collect(),
        rd: cfg.depletant_radius,
    };
    let model = FrozenRelativePoseProposal::from_json_str_open(
        &raw_model.to_string(),
        [2. * cfg.capture_radius; 3],
        cfg.uniform_probability,
        raw_model["shape_sha256"].as_str().unwrap(),
    )?;
    let weights = model.component_weights();
    let pairs = (0..weights.len())
        .flat_map(|a| {
            let weights = &weights;
            (a..weights.len()).map(move |b| BasinPair {
                first: a,
                second: b,
                weight: weights[a] * weights[b] * if a == b { 1. } else { 2. },
            })
        })
        .collect();
    let map = FixedBasinInvolution::new(
        model.component_parameters(),
        model.angular_length(),
        c,
        pairs,
    )?;
    let components = model
        .component_parameters()
        .into_iter()
        .map(|mut p| {
            p.weight = 1.;
            FrozenRelativePoseProposal::from_components_open(
                vec![p],
                model.angular_length(),
                model.box_lengths(),
                cfg.uniform_probability,
                model.shape_sha256(),
                model.shape_sha256(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let center_pose = |p: Pose| Pose {
        position: sub(p.position, cfg.capture_center),
        orientation: p.orientation,
    };
    let frames = rows(root.join(name).join("trajectory.jsonl"))?;
    let moves = rows(root.join(name).join("moves.jsonl"))?;
    let mut current: Pose = serde_json::from_value(frames[0]["pose"].clone())?;
    assert_eq!(current, cfg.initial_pose);
    let mut counts: [DockingCounts; 2] = Default::default();
    let mut uniform = 0;
    let mut mapped = 0;
    let mut identities = 0;
    let mut rejected = 0;
    let mut nonzero_gates = 0;
    let mut anchors = [0usize; 2];
    for row in moves {
        let cycle = row["cycle"].as_u64().unwrap();
        let attempt = row["attempt"].as_u64().unwrap() as usize;
        let is_global = row["kind"] == "global";
        assert_eq!(is_global, attempt == cfg.local_attempts_per_cycle);
        let index = usize::from(is_global);
        counts[index].attempted += 1;
        assert_eq!(json!(current), row["old_pose"]);
        let info = &row["proposal"];
        assert!(!info.as_object().unwrap().contains_key("native"));
        let candidate: Option<Pose> = serde_json::from_value(row["proposed_pose"].clone())?;
        let mut correction = 0.;
        if let Some(new) = candidate {
            new.validate()?;
            if is_global {
                let j = info["anchor_index"].as_u64().unwrap() as usize;
                anchors[j] += 1;
                let anchor = cfg.fixed_poses[j];
                if method == DockingMethod::Mixture {
                    let centered_anchor = center_pose(anchor);
                    let old_density = model.log_density(&center_pose(current), &centered_anchor)?;
                    let new_density = model.log_density(&center_pose(new), &centered_anchor)?;
                    assert_eq!(center_pose(anchor), centered_anchor);
                    assert_eq!(anchor, cfg.fixed_poses[j]);
                    near(info["old_log_density"].as_f64().unwrap(), old_density);
                    near(info["new_log_density"].as_f64().unwrap(), new_density);
                    correction = old_density - new_density;
                    if row["branch"] == "uniform" {
                        uniform += 1;
                    }
                } else if row["branch"] == "uniform" {
                    uniform += 1;
                    assert!(
                        sub(new.position, cfg.capture_center)
                            .into_iter()
                            .all(|x| x >= -cfg.capture_radius && x < cfg.capture_radius)
                    );
                    near(info["log_reverse_forward"].as_f64().unwrap(), 0.);
                } else {
                    assert_eq!(row["branch"], "involution");
                    mapped += 1;
                    let ar = rotation(anchor.orientation);
                    let relative = Pose {
                        position: matvec(transpose(ar), sub(current.position, anchor.position)),
                        orientation: quaternion(matmul(
                            transpose(ar),
                            rotation(current.orientation),
                        )),
                    };
                    let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
                    let step = map.apply(relative, &trace)?;
                    let reverse = map.apply(step.pose, &step.inverse_trace)?;
                    near_pose(reverse.pose, relative);
                    near(step.log_correction + reverse.log_correction, 0.);
                    let recorded: Pose = serde_json::from_value(info["step"]["pose"].clone())?;
                    near_pose(step.pose, recorded);
                    near(
                        step.log_correction,
                        info["step"]["log_correction"].as_f64().unwrap(),
                    );
                    let source = components[trace.source]
                        .relative_log_density(relative.position, rotation(relative.orientation))?;
                    let target = components[trace.target].relative_log_density(
                        step.pose.position,
                        rotation(step.pose.orientation),
                    )?;
                    near(
                        info["selected_source_log_density"].as_f64().unwrap(),
                        source,
                    );
                    near(
                        info["selected_target_log_density"].as_f64().unwrap(),
                        target,
                    );
                    // Remote Cayley-chart tails can have million-scale log
                    // densities with an order-one difference. The stable map
                    // correction must agree to subtraction precision, not to
                    // relative precision of the cancelled result alone.
                    assert!(
                        (step.log_correction - (source - target)).abs()
                            < 2e-9 * (1. + step.log_correction.abs())
                                + 128. * f64::EPSILON * (source.abs() + target.abs())
                    );
                    let identity = c == 1. && trace.source == trace.target;
                    assert_eq!(info["identity"], identity);
                    if identity {
                        identities += 1;
                        assert_eq!(current, new, "diagonal c=1 is bit-exact identity");
                        correction = 0.;
                    } else {
                        correction = if method == DockingMethod::PosteriorInvolution {
                            let old_g = model.relative_log_density(
                                relative.position,
                                rotation(relative.orientation),
                            )?;
                            let new_g = model.relative_log_density(
                                step.pose.position,
                                rotation(step.pose.orientation),
                            )?;
                            let log_source = weights[trace.source].ln() + source - old_g;
                            let log_reverse_source = weights[trace.target].ln() + target - new_g;
                            let labels = log_reverse_source + weights[trace.source].ln()
                                - log_source
                                - weights[trace.target].ln();
                            near(info["source_log_probability"].as_f64().unwrap(), log_source);
                            near(
                                info["inverse_source_log_probability"].as_f64().unwrap(),
                                log_reverse_source,
                            );
                            near(info["label_log_reverse_forward"].as_f64().unwrap(), labels);
                            near(step.log_correction + labels, old_g - new_g);
                            old_g - new_g
                        } else {
                            step.log_correction
                        };
                        near_pose(
                            new,
                            Pose {
                                position: add(anchor.position, matvec(ar, step.pose.position)),
                                orientation: quaternion(matmul(
                                    ar,
                                    rotation(step.pose.orientation),
                                )),
                            },
                        );
                    }
                }
            }
            near(correction, info["log_reverse_forward"].as_f64().unwrap());
            let capture = norm(sub(new.position, cfg.capture_center)) <= cfg.capture_radius;
            let hard = capture && env.hard_valid(new);
            assert_eq!(row["capture_valid"], capture);
            assert_eq!(row["hard_valid"], hard);
            if !capture {
                counts[index].capture_rejected += 1;
            } else if !hard {
                counts[index].hard_rejected += 1;
            } else {
                counts[index].hard_valid += 1;
                let gate = depletion::sample(
                    &mut rng(cfg.seed, cycle, attempt, "gate"),
                    &env,
                    current,
                    new,
                    cfg.poisson_lambda_ratio * cfg.reservoir_density,
                    cfg.reservoir_density,
                    cfg.endpoint_gate,
                )?;
                assert_eq!(
                    json!(gate),
                    row["gate"],
                    "replayed Poisson cloud must agree exactly"
                );
                counts[index].gate_raw_points += gate.raw_points;
                nonzero_gates += usize::from(gate.gained + gate.lost > 0);
                let alpha = (correction + gate.log_weight).min(0.);
                near(alpha, row["log_acceptance"].as_f64().unwrap());
                let accept = rng(cfg.seed, cycle, attempt, "accept").random::<f64>().ln() < alpha;
                assert_eq!(row["accepted"], accept);
                if accept {
                    counts[index].accepted += 1;
                    if norm(sub(new.position, current.position)) > 1e-10
                        || matrix_error(new, current) > 1e-12
                    {
                        counts[index].accepted_pose_changes += 1;
                    }
                }
            }
        } else {
            counts[index].numerical_nulls += 1;
        }
        if row["accepted"] == true {
            current = candidate.unwrap();
        } else {
            rejected += 1;
        }
        assert_eq!(json!(current), row["retained_pose"]);
        assert!(cfg.contains(current) && env.hard_valid(current));
        if is_global {
            assert_eq!(json!(current), frames[cycle as usize]["pose"]);
            assert_eq!(json!(counts), frames[cycle as usize]["counts"]);
        }
    }
    let checkpoint = read(root.join(name).join("checkpoint.json"))?;
    assert_eq!(checkpoint["pose"], json!(current));
    assert_eq!(checkpoint["counts"], json!(counts));
    assert_eq!(
        read(root.join(name).join("config.json"))?["fixed_poses"],
        json!(fixed_before)
    );
    assert_eq!(cfg.fixed_poses, fixed_before);
    assert!(anchors.iter().all(|&n| n > 20));
    assert!(uniform > 10 && rejected > 10 && nonzero_gates > 10);
    assert!(counts[0].accepted_pose_changes > 20 && counts[1].accepted_pose_changes > 2);
    if method != DockingMethod::Mixture {
        assert!(mapped > 80);
        if c == 1. {
            assert!(identities > 30);
            assert!(counts[1].accepted > counts[1].accepted_pose_changes + 30);
        } else {
            assert_eq!(identities, 0);
        }
    }
    eprintln!(
        "Docking replay {name}: global changed={}, identities={identities}, nonzero gates={nonzero_gates}",
        counts[1].accepted_pose_changes
    );
    Ok(())
}

#[test]
fn conditional_dumbbell_runner_replays_all_factors_and_restarts_exactly() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-docking-runner-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    let (cfg, model) = fixture(&root)?;
    for (name, method, c) in [
        ("mixture", DockingMethod::Mixture, 0.),
        ("c0", DockingMethod::Involution, 0.),
        ("c05", DockingMethod::Involution, 0.5),
        ("c1", DockingMethod::Involution, 1.),
        ("posterior-c0", DockingMethod::PosteriorInvolution, 0.),
        ("posterior-c09", DockingMethod::PosteriorInvolution, 0.9),
        ("posterior-c1", DockingMethod::PosteriorInvolution, 1.),
    ] {
        docking::run(options(&root, name, method, c, 192, None))?;
        let part = format!("{name}-part");
        let resumed = format!("{name}-resumed");
        docking::run(options(&root, &part, method, c, 37, None))?;
        docking::run(options(
            &root,
            &resumed,
            method,
            c,
            192,
            Some(root.join(&part).join("checkpoint.json")),
        ))?;
        assert_eq!(
            read(root.join(name).join("checkpoint.json"))?,
            read(root.join(&resumed).join("checkpoint.json"))?
        );
        let uninterrupted = rows(root.join(name).join("moves.jsonl"))?;
        let mut split = rows(root.join(&part).join("moves.jsonl"))?;
        split.extend(rows(root.join(&resumed).join("moves.jsonl"))?);
        assert_eq!(
            uninterrupted.into_iter().map(physical).collect::<Vec<_>>(),
            split.into_iter().map(physical).collect::<Vec<_>>()
        );
        replay(&root, name, &cfg, &model, method, c)?;
    }
    // Every rejection below must occur before writing any output artifact.
    let mut bad = options(&root, "bad-model", DockingMethod::Involution, 0.5, 10, None);
    let mut wrong_model = model.clone();
    wrong_model["shape_sha256"] = json!("0".repeat(64));
    fs::write(root.join("wrong-model.json"), wrong_model.to_string())?;
    bad.model = root.join("wrong-model.json");
    assert!(docking::run(bad).is_err());
    assert!(!root.join("bad-model").exists());
    let mut bad_config = read(root.join("input.json"))?;
    bad_config["initial_pose"] = bad_config["fixed_poses"][0].clone();
    fs::write(root.join("wrong-input.json"), bad_config.to_string())?;
    let mut bad = options(&root, "bad-input", DockingMethod::Mixture, 0., 10, None);
    bad.config = root.join("wrong-input.json");
    assert!(docking::run(bad).is_err());
    assert!(!root.join("bad-input").exists());
    let bad = options(
        &root,
        "bad-restart",
        DockingMethod::Involution,
        0.9,
        300,
        Some(root.join("c05/checkpoint.json")),
    );
    assert!(docking::run(bad).is_err());
    assert!(!root.join("bad-restart").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
