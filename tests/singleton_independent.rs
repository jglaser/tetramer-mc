//! Finite-cap hard conditioning: direct densities, complete records and
//! restart, plus an independent analytic two-sphere/depletion reference.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    cluster_phase::{ClusterPhase, ClusterPhaseConfig, ClusterPhaseCounts},
    depletion::GateOptions,
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Shape, SphereTree},
    math::*,
    oligomer_proposal::{OligomerConfig, OligomerMixture},
    proposal::FrozenRelativePoseProposal,
    simulation::{Method, RunOptions, hash_file, run},
    spherical::Container,
};
const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
fn pose(p: Vec3) -> Pose {
    Pose {
        position: p,
        orientation: [1., 0., 0., 0.],
    }
}
fn model_raw(shape_hash: &str) -> Value {
    let cov = |s: f64| -> [[f64; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { s } else { 0. }))
    };
    json!({"schema":"reciprocal-pose-mixture-v1","reciprocal_components":[true,false],"base_model":{
        "coordinate_convention":"anchor-body-relative","shape_sha256":shape_hash,"angular_length":1.,
        "weights":[0.35,0.65],"anchors":[{"position":[0.9,0.,0.],"rotation":IDENTITY},{"position":[0.,0.9,0.],"rotation":IDENTITY}],
        "means":vec![[0.;6];2],"covariances":[cov(0.07),cov(0.19)]}})
}
fn proposal(correlation: f64, uniform: f64) -> Result<DockingProposal> {
    let m = FrozenRelativePoseProposal::from_json_str_open(
        &model_raw(SHA).to_string(),
        [6.; 3],
        uniform,
        SHA,
    )?;
    DockingProposal::new(m, DockingMethod::PosteriorInvolution, correlation, [0.; 3])
}
fn settings(cap: Option<usize>, fused: bool) -> ClusterPhaseConfig {
    serde_json::from_value(json!({"duration":0.7,"singleton_rate":1.,"dimer_rate":0.,"trimer_rate":0.,
        "transport_probability":0.85,"correlation":0.9,"transport_charts":"members","anchor_count":2,
        "anchor_contact_uniform_probability":0.2,"anchor_pool_selection":"contact_without_replacement",
        "singleton_independent_max_trials":cap,"oligomer":if fused{json!({})}else{Value::Null}})).unwrap()
}
fn tree(r: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 4. * std::f64::consts::PI * r.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: r,
        }],
    })
    .unwrap()
}
fn close(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 1e-11);
    assert!(
        (a.orientation
            .iter()
            .zip(b.orientation)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs()
            - 1.)
            .abs()
            < 1e-11
    );
}

#[test]
fn config_is_explicit_singleton_only_and_legacy_counts_deserialize() -> Result<()> {
    let c = settings(Some(8), true);
    c.validate()?;
    for mutation in ["dimer", "trimer", "rate", "chart", "anchor", "cap"] {
        let mut q = c.clone();
        match mutation {
            "dimer" => q.dimer_rate = 1.,
            "trimer" => q.trimer_rate = 1.,
            "rate" => q.singleton_rate = 0.,
            "chart" => q.transport_charts = tetramer_mc::cluster_phase::TransportCharts::Handle,
            "anchor" => q.anchor_contact_uniform_probability = None,
            _ => q.singleton_independent_max_trials = Some(0),
        }
        assert!(q.validate().is_err(), "{mutation}");
    }
    let legacy = serde_json::to_value(ClusterPhaseCounts::default())?;
    assert!(legacy.get("independent_raw_trials").is_none());
    let x: ClusterPhaseCounts = serde_json::from_value(legacy)?;
    assert_eq!(x.independent_cap_exhaustions, 0);
    assert!(
        serde_json::to_value(ClusterPhaseConfig::default())?
            .get("singleton_independent_max_trials")
            .is_none()
    );
    Ok(())
}

#[test]
fn finite_cap_kernel_balances_nonuniform_guides_and_pool_selection_including_zero_volume() {
    let pi = [0.2, 0.3, 0.5];
    let selection = [
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.6, 0.15, 0.15],
        [0.3, 0.25, 0.2, 0.25],
    ];
    let guide: [[f64; 4]; 4] = [
        [0.05, 0.2, 0.15, 0.6],
        [0.3, 0.1, 0.5, 0.1],
        [0., 0., 0., 1.],
        [0.1, 0.7, 0.2, 0.],
    ];
    for cap in [0, 1, 8, 32] {
        let mut kernel = [[0.; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                if x == y {
                    continue;
                }
                for a in 0..4 {
                    let z = guide[a][..3].iter().sum::<f64>();
                    let b = (0..cap).map(|k| (1. - z).powi(k)).sum::<f64>();
                    let forward = selection[x][a] * b * guide[a][y];
                    let reverse = selection[y][a] * b * guide[a][x];
                    if forward > 0. {
                        kernel[x][y] += forward * (pi[y] * reverse / (pi[x] * forward)).min(1.);
                    }
                }
            }
            kernel[x][x] = 1. - kernel[x].iter().sum::<f64>();
        }
        for x in 0..3 {
            assert!(kernel[x][x] >= 0.);
            assert!((kernel[x].iter().sum::<f64>() - 1.).abs() < 1e-14);
            for y in 0..3 {
                assert!((pi[x] * kernel[x][y] - pi[y] * kernel[y][x]).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn direct_draws_have_correct_labels_latents_reciprocity_and_no_source_dependence() -> Result<()> {
    let p = proposal(0.9, 0.1)?;
    let other = proposal(-0.4, 0.1)?;
    let anchors = [
        pose([1., 2., 0.]),
        Pose {
            position: [-2., 1., 0.],
            orientation: quaternion(cayley([0.2, 0.1, -0.3])),
        },
    ];
    let mut rng = StdRng::seed_from_u64(6543);
    let mut rng2 = StdRng::seed_from_u64(6543);
    let mut labels = [0usize; 3];
    let mut sum = [0.; 6];
    let mut sum2 = [0.; 6];
    let (map, inverse, weights) = p.member_chart_parts();
    for _ in 0..6000 {
        let (y, r) = p.draw_singleton_independent(&mut rng, &anchors)?;
        let (y2, r2) = other.draw_singleton_independent(&mut rng2, &anchors)?;
        assert_eq!(y, y2);
        assert_eq!(r, r2);
        let y = y.unwrap();
        let b = r["target_label"]["branch"].as_u64().unwrap() as usize;
        let a = r["target_label"]["anchor"].as_u64().unwrap() as usize;
        let z: [f64; 6] = serde_json::from_value(r["target_latent"].clone())?;
        let ar = rotation(anchors[a].orientation);
        let mut rel = Pose {
            position: matvec(transpose(ar), sub(y.position, anchors[a].position)),
            orientation: quaternion(matmul(transpose(ar), rotation(y.orientation))),
        };
        if inverse[b] {
            rel = invert_relative_pose(rel);
        }
        close(rel, map.decode(b, z)?);
        for k in 0..6 {
            sum[k] += z[k];
            sum2[k] += z[k] * z[k];
        }
        labels[b] += 1;
        let terms: Vec<_> = anchors
            .iter()
            .flat_map(|&a| p.branch_log_densities(y, a).unwrap())
            .collect();
        let m = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let independent =
            m + terms.iter().map(|v| (v - m).exp()).sum::<f64>().ln() - (anchors.len() as f64).ln();
        assert!((independent - p.members_log_density(&[y], &anchors)?).abs() < 1e-10);
    }
    for b in 0..3 {
        assert!((labels[b] as f64 / 6000. - weights[b].exp()).abs() < 0.025);
    }
    for k in 0..6 {
        assert!((sum[k] / 6000.).abs() < 0.05);
        assert!((sum2[k] / 6000. - 1.).abs() < 0.06);
    }
    Ok(())
}

#[test]
fn fused_direct_decode_recovers_recorded_latents_and_weighted_labels() -> Result<()> {
    let tree = tree(0.15);
    let wall = Container::new(3., &tree)?;
    let p = proposal(0.9, 0.1)?;
    let old = [pose([0.; 3])];
    let anchors = [pose([-0.9, 0., 0.]), pose([0., -0.9, 0.])];
    let mix = OligomerMixture::build(
        &p,
        &tree,
        &wall,
        &old,
        &anchors,
        &anchors,
        &OligomerConfig::default(),
    )?;
    assert!(!mix.fused.is_empty());
    let mut rng = StdRng::seed_from_u64(761543);
    let mut fused = 0;
    for _ in 0..3000 {
        let (y, raw) = mix.draw_singleton_independent(&mut rng)?;
        let y = y.unwrap();
        let target = raw["target"].as_u64().unwrap() as usize;
        let z: [f64; 6] = serde_json::from_value(raw["target_latent"].clone())?;
        let recovered = mix.source_chart_coordinates(target, y)?;
        for k in 0..6 {
            assert!((z[k] - recovered[k]).abs() < 1e-9);
        }
        assert!(mix.log_density(&[y]).is_finite());
        fused += usize::from(raw["target_label"]["kind"] == "fused");
    }
    assert!((fused as f64 / 3000. - 0.8).abs() < 0.035);
    Ok(())
}

struct Fixture {
    root: PathBuf,
    config: Value,
}
impl Fixture {
    fn new(name: &str, cap: Option<usize>, fused: bool) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "singleton-independent-{name}-{}",
            std::process::id()
        ));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        fs::write(
            root.join("shape.json"),
            serde_json::to_vec(&tree(0.35).shape)?,
        )?;
        fs::write(
            root.join("model.json"),
            model_raw(&hash_file(&root.join("shape.json"))?).to_string(),
        )?;
        let config = json!({"shape":"shape.json","box_lengths":[6.,6.,6.],"boundary":{"kind":"spherical","radius":3.},
            "seed":853164,"depletant_radius":0.15,"reservoir_density":1.2,"poisson_lambda_ratio":4.,
            "global_probability":0.,"learned_uniform_weight":0.2,"gca_probability":0.,"center_shift_probability":0.,
            "local_translation_std_A":0.08,"local_small_angle_std_degrees":3.,"cluster_phase":settings(cap,fused),
            "assembly_bias":{"values":[0.,0.2,0.4,0.6]},"endpoint_gate":{"max_cells":15,"max_depth":3,"min_width":0.1},
            "initial_poses":[pose([-0.8,0.,0.]),pose([0.,0.1,0.]),pose([0.8,0.,0.]),pose([1.2,1.2,0.])]});
        Ok(Self { root, config })
    }
    fn run(&self, name: &str, n: u64, resume: Option<&str>) -> Result<()> {
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
        })?;
        Ok(())
    }
    fn read(&self, name: &str, file: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(name).join(file),
        )?)?)
    }
    fn rows(&self, name: &str) -> Result<Vec<Value>> {
        fs::read_to_string(self.root.join(name).join("moves.jsonl"))?
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
                scrub(x)
            }
        }
        Value::Array(a) => {
            for x in a {
                scrub(x)
            }
        }
        _ => {}
    }
}

#[test]
fn independent_restart_records_all_raw_trials_and_full_endpoint_corrections() -> Result<()> {
    for fused in [false, true] {
        let f = Fixture::new(if fused { "fused" } else { "member" }, Some(4), fused)?;
        f.run("full", 90, None)?;
        f.run("part", 37, None)?;
        f.run("resume", 90, Some("part"))?;
        assert_eq!(
            f.read("full", "checkpoint.json")?,
            f.read("resume", "checkpoint.json")?
        );
        let mut full: Vec<_> = f
            .rows("full")?
            .into_iter()
            .filter(|r| r["sweep"].as_u64().unwrap() > 37)
            .collect();
        let mut resumed = f.rows("resume")?;
        for r in full.iter_mut().chain(&mut resumed) {
            scrub(r);
        }
        assert_eq!(full, resumed);
        let mut events = 0;
        let mut raws = 0;
        let mut exhausted = 0;
        let mut accepted = 0;
        let mut uniform = 0;
        let mut state: Vec<Pose> = serde_json::from_value(f.config["initial_poses"].clone())?;
        for r in f.rows("full")? {
            if r["kind"] == "local" || r["kind"] == "global" {
                let i = r["moving_index"].as_u64().unwrap() as usize;
                assert_eq!(json!(state[i]), r["old_pose"]);
                state[i] = serde_json::from_value(r["retained_pose"].clone())?;
            }
            if r["kind"] != "cluster_event" {
                continue;
            }
            let i = r["members"][0].as_u64().unwrap() as usize;
            assert_eq!(json!(state[i]), r["old_poses"][0]);
            let old_state = state.clone();
            state[i] = serde_json::from_value(r["retained_poses"][0].clone())?;
            let info = &r["proposal"];
            if info["branch"] == "uniform" {
                uniform += 1;
                assert!(info.get("anchor_pool").is_none());
                assert_eq!(info["log_reverse_forward"], 0.);
            }
            if info["branch"] != "independent_conditioned" {
                continue;
            }
            events += 1;
            let rows = info["raw_trials"].as_array().unwrap();
            let n = info["raw_trial_count"].as_u64().unwrap() as usize;
            assert_eq!(rows.len(), n);
            assert!((1..=4).contains(&n));
            raws += n;
            for (k, row) in rows.iter().enumerate() {
                assert_eq!(row["trial"], k + 1);
                if !row["candidate"].is_null() {
                    let p: Pose = serde_json::from_value(row["candidate"].clone())?;
                    let independent_hard = norm(p.position) + 0.35 <= 3.
                        && old_state
                            .iter()
                            .enumerate()
                            .all(|(j, s)| j == i || norm(sub(p.position, s.position)) >= 0.7);
                    assert_eq!(row["hard_valid"], independent_hard);
                }
                if k + 1 < n {
                    assert_eq!(row["hard_valid"], false);
                }
            }
            if info["cap_exhausted"] == true {
                exhausted += 1;
                assert_eq!(n, 4);
                assert!(rows.iter().all(|x| x["hard_valid"] == false));
                assert_eq!(r["accepted"], false);
                assert_eq!(r["old_poses"], r["retained_poses"]);
            } else {
                assert_eq!(rows.last().unwrap()["hard_valid"], true);
                assert_eq!(rows.last().unwrap()["candidate"], r["proposed_poses"][0]);
                assert_eq!(r["hard_valid"], true);
                let old = info["full_old_log_density"].as_f64().unwrap();
                let new = info["full_new_log_density"].as_f64().unwrap();
                let pool = info["pool_reverse_log_probability"].as_f64().unwrap()
                    - info["pool_forward_log_probability"].as_f64().unwrap();
                let anchors: Vec<usize> = serde_json::from_value(info["anchor_pool"].clone())?;
                let pool_probability = |poses: &[Pose]| {
                    let mut excluded = vec![i];
                    let mut result = 0.;
                    for &selected in &anchors {
                        let available: Vec<_> =
                            (0..poses.len()).filter(|j| !excluded.contains(j)).collect();
                        let contacting: Vec<_> = available
                            .iter()
                            .copied()
                            .filter(|&j| norm(sub(poses[i].position, poses[j].position)) < 1.)
                            .collect();
                        let prob = if contacting.is_empty() {
                            1. / available.len() as f64
                        } else {
                            0.2 / available.len() as f64
                                + if contacting.contains(&selected) {
                                    0.8 / contacting.len() as f64
                                } else {
                                    0.
                                }
                        };
                        result += prob.ln();
                        excluded.push(selected);
                    }
                    result
                };
                let mut proposed = old_state.clone();
                proposed[i] = serde_json::from_value(r["proposed_poses"][0].clone())?;
                assert!(
                    (pool_probability(&old_state)
                        - info["pool_forward_log_probability"].as_f64().unwrap())
                    .abs()
                        < 1e-12
                );
                assert!(
                    (pool_probability(&proposed)
                        - info["pool_reverse_log_probability"].as_f64().unwrap())
                    .abs()
                        < 1e-12
                );
                assert!(
                    (info["log_reverse_forward"].as_f64().unwrap() - (old - new + pool)).abs()
                        < 1e-10
                );
            }
            accepted += usize::from(r["accepted"] == true);
        }
        assert!(events > 80 && raws > events && exhausted > 0 && accepted > 0 && uniform > 0);
        let checkpoint = f.read("full", "checkpoint.json")?;
        let c = &checkpoint["counts"]["cluster_phase"];
        assert_eq!(c["independent_raw_trials"], raws);
        assert_eq!(c["independent_cap_exhaustions"], exhausted);
    }
    Ok(())
}

#[test]
fn omitted_and_explicit_disabled_modes_preserve_complete_stream() -> Result<()> {
    let mut f = Fixture::new("disabled", None, false)?;
    f.config["cluster_phase"]
        .as_object_mut()
        .unwrap()
        .remove("singleton_independent_max_trials");
    f.run("implicit", 25, None)?;
    f.config["cluster_phase"]["singleton_independent_max_trials"] = Value::Null;
    f.run("explicit", 25, None)?;
    let mut a = f.rows("implicit")?;
    let mut b = f.rows("explicit")?;
    for r in a.iter_mut().chain(&mut b) {
        scrub(r);
    }
    assert_eq!(a, b);
    let a = f.read("implicit", "checkpoint.json")?;
    let b = f.read("explicit", "checkpoint.json")?;
    assert_eq!(a["poses"], b["poses"]);
    assert_eq!(a["counts"], b["counts"]);
    Ok(())
}

#[test]
fn cap_exhaustion_is_a_retained_self_loop_and_recording_does_not_change_rng() -> Result<()> {
    let tree = tree(0.35);
    let wall = Container::new(2., &tree)?;
    let mut raw = model_raw(SHA);
    raw["base_model"]["anchors"][0]["position"] = json!([1000., 0., 0.]);
    raw["base_model"]["anchors"][1]["position"] = json!([0., 1000., 0.]);
    // The production model requires a strictly positive defensive weight.
    // This fixed RNG stream takes no uniform branch in the bounded test.
    let m = FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [6.; 3], 1e-12, SHA)?;
    let proposal = DockingProposal::new(m, DockingMethod::PosteriorInvolution, 0.9, [0.; 3])?;
    let mut cfg = settings(Some(3), false);
    cfg.transport_probability = 1.;
    cfg.duration = 4.;
    let engine = ClusterPhase::new(
        &tree,
        &wall,
        0.1,
        0.,
        1.,
        GateOptions {
            max_cells: 1,
            max_depth: 0,
            min_width: 0.,
        },
        cfg,
        Some(proposal),
    )?;
    let initial = vec![pose([-0.5, 0., 0.]), pose([0.5, 0., 0.])];
    let mut results = Vec::new();
    for record in [true, false] {
        let mut x = initial.clone();
        let mut clock = StdRng::seed_from_u64(21);
        let mut p = StdRng::seed_from_u64(22);
        let mut g = StdRng::seed_from_u64(23);
        let mut a = StdRng::seed_from_u64(24);
        let mut b = StdRng::seed_from_u64(25);
        let (counts, rows) = engine.run(
            &mut x, &mut clock, &mut p, &mut g, &mut a, &mut b, None, &mut None, record,
        )?;
        assert_eq!(x, initial);
        assert!(counts.events > 0);
        assert_eq!(counts.independent_cap_exhaustions, counts.events);
        assert_eq!(counts.independent_raw_trials, 3 * counts.events);
        assert_eq!(counts.accepted, 0);
        if record {
            for row in rows.iter().filter(|r| r["kind"] == "cluster_event") {
                assert_eq!(row["proposal"]["raw_trials"].as_array().unwrap().len(), 3);
                assert_eq!(row["old_poses"], row["retained_poses"]);
            }
        } else {
            assert!(rows.is_empty());
        }
        results.push((
            counts,
            [
                clock.random::<u64>(),
                p.random(),
                g.random(),
                a.random(),
                b.random(),
            ],
        ));
    }
    assert_eq!(results[0], results[1]);
    Ok(())
}

#[test]
fn sphere_stationarity_matches_independent_analytic_hard_and_depletion_reference() -> Result<()> {
    const CORE: f64 = 0.35;
    const RD: f64 = 0.15;
    const WALL: f64 = 2.5;
    const CENTER: f64 = WALL - CORE;
    let lens = |d: f64| {
        let r = CORE + RD;
        if d >= 2. * r {
            0.
        } else {
            std::f64::consts::PI * (4. * r + d) * (2. * r - d).powi(2) / 12.
        }
    };
    let tree = tree(CORE);
    let wall = Container::new(WALL, &tree)?;
    let mut chain_results = Vec::new();
    for z in [0., 20.] {
        // Analytic separation density of two independent uniform points in a
        // ball, followed by exact hard exclusion and two-sphere lens weight.
        let mut normw = 0.;
        let mut reference = [0.; 3];
        let bins = 200_000;
        for k in 0..bins {
            let d = 2. * CENTER * (k as f64 + 0.5) / bins as f64;
            if d < 2. * CORE {
                continue;
            }
            let p = 3. * d * d / CENTER.powi(3)
                * (1. - 3. * d / (4. * CENTER) + d.powi(3) / (16. * CENTER.powi(3)));
            let w = p * (z * lens(d)).exp();
            normw += w;
            reference[0] += w * f64::from(d < 2. * (CORE + RD));
            reference[1] += w * d * d;
        }
        reference[0] /= normw;
        reference[1] /= normw;
        reference[2] = 0.25;
        for compact in [false, true] {
            let mut cfg = settings(Some(8), false);
            cfg.duration = 1.;
            cfg.transport_probability = 1.;
            let engine = ClusterPhase::new(
                &tree,
                &wall,
                RD,
                z,
                if z > 0. { z } else { 1. },
                GateOptions {
                    max_cells: 7,
                    max_depth: 2,
                    min_width: 0.,
                },
                cfg,
                Some(proposal(0.9, 0.35)?),
            )?;
            let separation = if compact { 0.72 } else { 3. };
            let mut x = vec![
                pose([-separation / 2., 0., 0.]),
                pose([separation / 2., 0., 0.]),
            ];
            let seed = 358721 + u64::from(compact) * 19 + (z as u64) * 503;
            let mut clock = StdRng::seed_from_u64(seed);
            let mut p = StdRng::seed_from_u64(seed + 1);
            let mut g = StdRng::seed_from_u64(seed + 2);
            let mut a = StdRng::seed_from_u64(seed + 3);
            let mut b = StdRng::seed_from_u64(seed + 4);
            let burn = 2000;
            let size = 300;
            let blocks = 60;
            let mut blockvec = vec![[0.; 3]; blocks];
            let mut count = ClusterPhaseCounts::default();
            for step in 0..burn + size * blocks {
                let (c, _) = engine.run(
                    &mut x, &mut clock, &mut p, &mut g, &mut a, &mut b, None, &mut None, false,
                )?;
                if step < burn {
                    continue;
                }
                count.add(&c);
                let d = norm(sub(x[0].position, x[1].position));
                assert!(d >= 2. * CORE - 1e-12);
                let obs = [
                    f64::from(d < 2. * (CORE + RD)),
                    d * d,
                    (x[0].orientation[0].powi(2) + x[1].orientation[0].powi(2)) / 2.,
                ];
                for k in 0..3 {
                    blockvec[(step - burn) / size][k] += obs[k] / size as f64;
                }
            }
            let mean: [f64; 3] =
                std::array::from_fn(|k| blockvec.iter().map(|v| v[k]).sum::<f64>() / blocks as f64);
            let se: [f64; 3] = std::array::from_fn(|k| {
                (blockvec
                    .iter()
                    .map(|v| (v[k] - mean[k]).powi(2))
                    .sum::<f64>()
                    / (blocks * (blocks - 1)) as f64)
                    .sqrt()
            });
            println!(
                "z={z},compact={compact}: analytic={reference:?}, mean={mean:?}, block_se={se:?}, accepted={}, raw={}",
                count.accepted, count.independent_raw_trials
            );
            for k in 0..3 {
                assert!(
                    (mean[k] - reference[k]).abs() < 6. * se[k] + 0.001,
                    "z={z} compact={compact} observable{k}: {} vs {} ± {}",
                    mean[k],
                    reference[k],
                    se[k]
                );
            }
            assert!(count.accepted > 1000 && count.independent_raw_trials > count.events);
            chain_results.push((z, mean, se));
        }
    }
    for pair in chain_results.chunks(2) {
        for k in 0..3 {
            assert!(
                (pair[0].1[k] - pair[1].1[k]).abs() < 6. * pair[0].2[k].hypot(pair[1].2[k]) + 0.001
            );
        }
    }
    Ok(())
}
