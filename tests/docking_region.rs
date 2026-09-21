//! Explicit conditional support, spectator-only anchor selection, and the
//! matched local third slot. These are synthetic controls, not protein jobs.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tetramer_mc::{
    docking::{self, DockingConfig, DockingMethod, DockingOptions, DockingProposal},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::hash_file,
};

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn temporary() -> PathBuf {
    std::env::temp_dir().join(format!(
        "tetramer-docking-region-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
fn fixture(root: &Path) -> Result<DockingConfig> {
    fs::create_dir_all(root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"region sphere", "volume":0.5235987755982988,
        "atoms":[{"center":[0.,0.,0.],"radius":0.5}]})
        .to_string(),
    )?;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                if i < 3 { 0.0225 } else { 0.0009 }
            } else {
                0.
            }
        })
    });
    let identity = rotation([1., 0., 0., 0.]);
    let zero_mean = [0.; 6];
    let model = json!({"schema":"weighted-pose-mixture-v1", "angular_length":1., "coordinate_convention":"anchor-body-relative",
        "shape_sha256":hash_file(&root.join("shape.json"))?,
        "anchors":[{"position":[-5.,3.,0.],"rotation":identity},
                   {"position":[-5.,-3.,0.],"rotation":identity}],
        "means":[zero_mean,zero_mean],"covariances":[covariance,covariance],"weights":[0.4,0.6]});
    fs::write(root.join("model.json"), model.to_string())?;
    let config = json!({"shape":root.join("shape.json"),
        "fixed_poses":[pose([5.,0.,0.]),pose([-1.1,-3.,0.])],"initial_pose":pose([0.,3.,0.]),
        "capture_center":[0.,0.,0.],"capture_radius":8.,"depletant_radius":0.3,
        "reservoir_density":0.4,"poisson_lambda_ratio":4.,
        "translation_steps":[0.15,1.2],"rotation_steps_deg":[1.5,20.],
        "rotation_probability":0.5,"local_attempts_per_cycle":2,"uniform_probability":0.1,
        "seed":7021551,"endpoint_gate":{"max_cells":31,"max_depth":5,"min_width":0.1},
        "proposal_anchor_index":0,"target_region":{
            "metric":{"native_poses":[pose([0.;3])],"rigid_members":[pose([0.;3])],
                "member_error_scale":2.,"angle_error_scale_deg":15.},
            "window":{"minimum":1.,"maximum":2.,"lower_inclusive":false,"upper_inclusive":false}}});
    fs::write(root.join("config.json"), config.to_string())?;
    Ok(serde_json::from_value(config)?)
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
        config: root.join("config.json"),
        model: root.join("model.json"),
        out: root.join(name),
        cycles,
        sample_every: 1,
        method,
        correlation: c,
        resume,
    }
}
fn rows(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}
fn without_timing(mut row: Value) -> Value {
    row.as_object_mut().unwrap().remove("sampler_cpu_seconds");
    row
}

#[test]
fn original_q_window_has_explicit_boundaries_and_defaults_preserve_capture() -> Result<()> {
    let root = temporary();
    let mut cfg = fixture(&root)?;
    cfg.validate()?;
    let region = cfg.target_region.as_mut().unwrap();
    for x in [2., 4.] {
        assert!(!region.contains(pose([x, 0., 0.])));
    }
    assert!(region.contains(pose([3., 0., 0.])));
    assert_eq!(region.metric.q(pose([3., 0., 0.])), 1.5);
    region.window.lower_inclusive = true;
    region.window.upper_inclusive = true;
    for x in [2., 4.] {
        assert!(region.contains(pose([x, 0., 0.])));
    }
    let mut rotated = pose([3., 0., 0.]);
    let half = 31_f64.to_radians() / 2.;
    rotated.orientation = [half.cos(), 0., 0., half.sin()];
    assert!(!region.contains(rotated));
    region.metric.member_error_scale = 0.;
    assert!(cfg.validate().is_err());
    cfg.target_region = None;
    cfg.proposal_anchor_index = None;
    cfg.validate()?;
    assert!(cfg.region_contains(pose([0.; 3])));
    let raw = serde_json::to_value(&cfg)?;
    assert!(raw.get("target_region").is_none() && raw.get("proposal_anchor_index").is_none());
    let restored: DockingConfig = serde_json::from_value(raw)?;
    assert!(restored.region_contains(pose([0.; 3])));
    cfg.proposal_anchor_index = Some(2);
    assert!(cfg.validate().is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn fixed_anchor_changes_only_proposal_and_keeps_other_physical_neighbor() -> Result<()> {
    let root = temporary();
    let cfg = fixture(&root)?;
    let shape: Shape = serde_json::from_slice(&fs::read(&cfg.shape)?)?;
    let tree = SphereTree::new(shape)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        &fs::read_to_string(root.join("model.json"))?,
        [16.; 3],
        0.1,
        &hash_file(&cfg.shape)?,
    )?;
    let proposal = DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0.9, [0.; 3])?
        .with_anchor_index(Some(0));
    let mut altered = cfg.fixed_poses.clone();
    altered[1] = pose([0., 7., 0.]);
    let mut a = StdRng::seed_from_u64(92071);
    let mut b = StdRng::seed_from_u64(92071);
    for _ in 0..64 {
        let first = proposal.propose(&mut a, cfg.initial_pose, &cfg.fixed_poses)?;
        let second = proposal.propose(&mut b, cfg.initial_pose, &altered)?;
        assert_eq!(first, second);
        assert_eq!(first.1["anchor_index"], 0);
    }
    let env = Environment {
        tree: &tree,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: vec![(0, [0; 3]), (1, [0; 3])],
        rd: cfg.depletant_radius,
    };
    let only_a = Environment {
        tree: &tree,
        fixed: vec![Placed::new(cfg.fixed_poses[0])],
        labels: vec![(0, [0; 3])],
        rd: cfg.depletant_radius,
    };
    let witness = cfg.fixed_poses[1];
    assert!(cfg.contains(witness) && cfg.region_contains(witness));
    assert!(only_a.hard_valid(witness));
    assert!(!env.hard_valid(witness));
    // B also remains in the exclusion union even though it cannot be selected
    // as a chart anchor.
    assert!(env.contains(witness.position));
    assert!(!only_a.contains(witness.position));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn conditional_arms_keep_self_loops_exchange_both_ways_and_restart() -> Result<()> {
    let root = temporary();
    let cfg = fixture(&root)?;
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(&cfg.shape)?)?)?;
    let env = Environment {
        tree: &tree,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: vec![(0, [0; 3]), (1, [0; 3])],
        rd: cfg.depletant_radius,
    };
    let mut common_first_slots = None;
    for (name, method, c) in [
        ("local", DockingMethod::Local, 0.),
        ("redraw", DockingMethod::PosteriorInvolution, 0.),
        ("transport", DockingMethod::PosteriorInvolution, 0.9),
    ] {
        let summary = docking::run(options(&root, name, method, c, 400, None))?;
        let all = rows(root.join(name).join("moves.jsonl"))?;
        assert_eq!(all.len(), 1200);
        let first: Vec<_> = all[..2].iter().cloned().map(without_timing).collect();
        if let Some(previous) = &common_first_slots {
            assert_eq!(&first, previous);
        } else {
            common_first_slots = Some(first);
        }
        let mut crossings = [0usize; 2];
        let mut out_of_region = 0usize;
        let mut gates = 0usize;
        let mut hard_b_rejections = 0usize;
        for (index, row) in all.iter().enumerate() {
            assert_eq!(row["cycle"].as_u64().unwrap(), (index / 3 + 1) as u64);
            assert_eq!(row["attempt"].as_u64().unwrap(), (index % 3) as u64);
            let old: Pose = serde_json::from_value(row["old_pose"].clone())?;
            let retained: Pose = serde_json::from_value(row["retained_pose"].clone())?;
            assert!(
                cfg.contains(retained) && cfg.region_contains(retained) && env.hard_valid(retained)
            );
            assert_eq!(
                row["retained_q"].as_f64().unwrap(),
                cfg.target_region.as_ref().unwrap().metric.q(retained)
            );
            let global = method != DockingMethod::Local && index % 3 == 2;
            assert_eq!(row["kind"], if global { "global" } else { "local" });
            if let Some(candidate) = row["proposed_pose"].as_object() {
                let proposed: Pose = serde_json::from_value(Value::Object(candidate.clone()))?;
                assert_eq!(row["region_valid"], cfg.region_contains(proposed));
                if cfg.contains(proposed) && !cfg.region_contains(proposed) {
                    out_of_region += 1;
                    assert_eq!(row["accepted"], false);
                    assert_eq!(old, retained);
                    assert!(row["gate"].is_null());
                }
                if cfg.contains(proposed)
                    && cfg.region_contains(proposed)
                    && !env.hard_valid(proposed)
                {
                    hard_b_rejections += 1;
                    assert_eq!(row["accepted"], false);
                }
            }
            if !row["gate"].is_null() {
                gates += 1;
            }
            if old.position[1] > 0. && retained.position[1] < 0. {
                crossings[0] += 1;
            }
            if old.position[1] < 0. && retained.position[1] > 0. {
                crossings[1] += 1;
            }
            if global {
                assert_eq!(row["proposal"]["anchor_index"], 0);
            }
        }
        assert!(out_of_region > 20 && gates > 20 && hard_b_rejections > 0);
        if method == DockingMethod::Local {
            assert_eq!(summary["counts"][0]["attempted"], 1200);
            assert_eq!(summary["counts"][1]["attempted"], 0);
        } else {
            assert_eq!(summary["counts"][0]["attempted"], 800);
            assert_eq!(summary["counts"][1]["attempted"], 400);
            assert!(
                crossings[0] > 2 && crossings[1] > 2,
                "{name}: {crossings:?}"
            );
        }
        let partial = format!("{name}-partial");
        let resumed = format!("{name}-resumed");
        docking::run(options(&root, &partial, method, c, 73, None))?;
        docking::run(options(
            &root,
            &resumed,
            method,
            c,
            400,
            Some(root.join(&partial).join("checkpoint.json")),
        ))?;
        let continued = rows(root.join(resumed).join("moves.jsonl"))?;
        assert_eq!(
            continued
                .into_iter()
                .map(without_timing)
                .collect::<Vec<_>>(),
            all[73 * 3..]
                .iter()
                .cloned()
                .map(without_timing)
                .collect::<Vec<_>>()
        );
    }
    let mut bad = cfg.clone();
    bad.initial_pose = pose([2., 0., 0.]);
    fs::write(root.join("bad-config.json"), serde_json::to_vec(&bad)?)?;
    let mut bad_options = options(&root, "bad-window-start", DockingMethod::Local, 0., 1, None);
    bad_options.config = root.join("bad-config.json");
    assert!(docking::run(bad_options).is_err());
    assert!(!root.join("bad-window-start").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
