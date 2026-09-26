//! End-to-end validation of fused oligomer proposals in the physical runner.
//! Restart and replay retain every attempted move; rejected states remain samples.
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
            std::env::temp_dir().join(format!("fused-cluster-phase-{name}-{}", std::process::id()));
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

fn enable_fusion(f: &mut Fixture) {
    f.config["cluster_phase"]["transport_charts"] = json!("members");
    f.config["cluster_phase"]["anchor_count"] = json!(2);
    f.config["cluster_phase"]["anchor_contact_uniform_probability"] = json!(0.2);
    f.config["cluster_phase"]["contact_fusion"] = json!({});
}

#[test]
fn fusion_runner_has_exact_restart_and_replay_with_bias() -> Result<()> {
    let mut f = Fixture::new("restart")?;
    enable_fusion(&mut f);
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
    let (mut events, mut rejects, mut uniform, mut fused, mut fused_accepted) = (0, 0, 0, 0, 0);
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
                events += 1;
                rejects += usize::from(row["accepted"] == false);
                let ids: Vec<usize> = serde_json::from_value(row["members"].clone())?;
                assert_eq!(
                    row["old_poses"],
                    json!(ids.iter().map(|&i| poses[i]).collect::<Vec<_>>())
                );
                let info = &row["proposal"];
                if info["branch"] == "uniform" {
                    uniform += 1;
                    assert!(info.get("primary_anchor").is_none());
                    assert_eq!(info["log_reverse_forward"], 0.);
                }
                let used_fusion = info["source_kind"] == "fused" || info["target_kind"] == "fused";
                fused += usize::from(used_fusion);
                fused_accepted += usize::from(used_fusion && row["accepted"] == true);
                if let Some(primary) = info["primary_anchor"].as_u64() {
                    let probability = |state: &[Pose]| {
                        let counts: Vec<usize> = (0..state.len())
                            .map(|j| {
                                if ids.contains(&j) {
                                    0
                                } else {
                                    ids.iter()
                                        .filter(|&&i| {
                                            norm(sub(state[i].position, state[j].position)) < 1.7
                                        })
                                        .count()
                                }
                            })
                            .collect();
                        let total: usize = counts.iter().sum();
                        let m = (state.len() - ids.len()) as f64;
                        if total == 0 {
                            1. / m
                        } else {
                            0.2 / m + 0.8 * counts[primary as usize] as f64 / total as f64
                        }
                    };
                    let qx = probability(&poses);
                    assert!(
                        (qx - info["anchor_forward_probability"].as_f64().unwrap()).abs() < 1e-12
                    );
                    if row["hard_valid"] == true && row["internal_contact_graph_preserved"] == true
                    {
                        let mut proposed = poses.clone();
                        for (&i, p) in ids.iter().zip(row["proposed_poses"].as_array().unwrap()) {
                            proposed[i] = serde_json::from_value(p.clone())?;
                        }
                        let qy = probability(&proposed);
                        let anchor_correction = qy.ln() - qx.ln();
                        assert!(
                            (anchor_correction
                                - info["anchor_log_reverse_forward"].as_f64().unwrap())
                            .abs()
                                < 1e-12
                        );
                        let total =
                            anchor_correction + info["map_log_reverse_forward"].as_f64().unwrap();
                        assert!(
                            (total - info["log_reverse_forward"].as_f64().unwrap()).abs() < 1e-12
                        );
                        let expected =
                            (total + row["gate"]["log_weight"].as_f64().unwrap()).min(0.);
                        assert!((expected - row["log_acceptance"].as_f64().unwrap()).abs() < 1e-12);
                    }
                }
                for (&i, p) in ids.iter().zip(row["retained_poses"].as_array().unwrap()) {
                    poses[i] = serde_json::from_value(p.clone())?;
                }
            }
            "cluster_phase_end" => {}
            other => panic!("unexpected record {other}"),
        }
        spherical::validate_state(&tree, &wall, &poses)?;
        if k + 1 == rows.len() || rows[k + 1]["sweep"] != row["sweep"] {
            assert_eq!(frames[sweep]["poses"], json!(poses));
        }
    }
    eprintln!(
        "fused runner events={events} rejected={rejects} uniform={uniform} fused={fused} fused_accepted={fused_accepted}"
    );
    assert!(events > 100 && rejects > 0 && uniform > 0 && fused > 0 && fused_accepted > 0);
    assert_eq!(summary["counts"]["cluster_phase"]["events"], events);
    Ok(())
}

#[test]
fn absent_and_null_fusion_preserve_existing_member_bitstream() -> Result<()> {
    let mut f = Fixture::new("disabled")?;
    enable_fusion(&mut f);
    f.config["cluster_phase"]
        .as_object_mut()
        .unwrap()
        .remove("contact_fusion");
    f.run("absent", 30, None)?;
    f.config["cluster_phase"]["contact_fusion"] = Value::Null;
    f.run("null", 30, None)?;
    let mut a = f.read("absent", "checkpoint.json")?;
    let mut b = f.read("null", "checkpoint.json")?;
    a.as_object_mut().unwrap().remove("config_sha256");
    b.as_object_mut().unwrap().remove("config_sha256");
    assert_eq!(a, b);
    let mut a = f.rows("absent", "moves.jsonl")?;
    let mut b = f.rows("null", "moves.jsonl")?;
    for r in a.iter_mut().chain(b.iter_mut()) {
        scrub(r);
    }
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn fusion_configuration_rejects_handle_charts_and_unknown_fields() -> Result<()> {
    let mut f = Fixture::new("guards")?;
    enable_fusion(&mut f);
    serde_json::from_value::<Config>(f.config.clone())?.validate()?;
    f.config["cluster_phase"]["transport_charts"] = json!("handle");
    assert!(
        serde_json::from_value::<Config>(f.config.clone())?
            .validate()
            .is_err()
    );
    f.config["cluster_phase"]["transport_charts"] = json!("members");
    f.config["cluster_phase"]["contact_fusion"]["typo_parameter"] = json!(0.5);
    assert!(serde_json::from_value::<Config>(f.config.clone()).is_err());
    Ok(())
}

mod physical_reference {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, StandardNormal};
    use serde_json::json;
    use std::f64::consts::PI;
    use tetramer_mc::{
        cluster_phase::{ClusterPhase, ClusterPhaseConfig, ClusterPhaseCounts, TransportCharts},
        depletion::GateOptions,
        docking::{DockingMethod, DockingProposal},
        geometry::{Atom, Shape, SphereTree},
        math::{IDENTITY, Pose, cayley, norm, sub},
        oligomer_fusion::FusionConfig,
        proposal::FrozenRelativePoseProposal,
        rigid_subset::RigidSubset,
        simulation::hash_bytes,
        spherical::Container,
    };

    const RD: f64 = 0.14;
    const Z: f64 = 5.;
    const LAMBDA: f64 = 5.;
    const WALL: f64 = 3.5;
    const CENTER_RADIUS: f64 = WALL - 1.;
    const OBS: usize = 4;

    fn pose(position: [f64; 3]) -> Pose {
        Pose {
            position,
            orientation: [1., 0., 0., 0.],
        }
    }
    fn uniform_pose(rng: &mut StdRng) -> Pose {
        let position = loop {
            let p = std::array::from_fn(|_| rng.random_range(-CENTER_RADIUS..CENTER_RADIUS));
            if norm(p) <= CENTER_RADIUS {
                break p;
            }
        };
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        Pose {
            position,
            orientation: q.map(|v| v / length),
        }
    }
    fn lens(distance: f64) -> f64 {
        let r = 1. + RD;
        if distance >= 2. * r {
            0.
        } else {
            PI * (4. * r + distance) * (2. * r - distance).powi(2) / 12.
        }
    }
    fn exact(poses: &[Pose]) -> Option<(f64, [f64; OBS])> {
        let n = poses.len();
        let mut weight = 0.;
        let mut contacts = 0;
        let mut distance2 = 0.;
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                i = parent[i];
            }
            i
        }
        for i in 0..n {
            for j in i + 1..n {
                let d = norm(sub(poses[i].position, poses[j].position));
                if d < 2. {
                    return None;
                }
                weight += Z * lens(d);
                if d < 2. * (1. + RD) {
                    contacts += 1;
                    let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                    parent[a] = b;
                }
                distance2 += d * d;
            }
        }
        let mut sizes = vec![0; n];
        for i in 0..n {
            sizes[root(&mut parent, i)] += 1;
        }
        let largest = *sizes.iter().max().unwrap();
        Some((
            weight.exp(),
            [
                contacts as f64,
                largest as f64,
                distance2 / (n * (n - 1) / 2) as f64,
                poses.iter().map(|p| p.orientation[0].powi(2)).sum::<f64>() / n as f64,
            ],
        ))
    }

    #[derive(Clone, Debug)]
    struct Reference {
        mean: [f64; OBS],
        se: [f64; OBS],
        ess: f64,
        valid: usize,
    }
    fn reference(draws: usize, bodies: usize) -> Reference {
        let mut rng = StdRng::seed_from_u64(10459359);
        let mut sumw = 0.;
        let mut sumw2 = 0.;
        let mut valid = 0;
        let mut sumwf = [0.; OBS];
        let mut sumw2f = [0.; OBS];
        let mut sumw2f2 = [0.; OBS];
        for _ in 0..draws {
            let poses: Vec<_> = (0..bodies).map(|_| uniform_pose(&mut rng)).collect();
            let Some((w, f)) = exact(&poses) else {
                continue;
            };
            valid += 1;
            sumw += w;
            sumw2 += w * w;
            for k in 0..OBS {
                sumwf[k] += w * f[k];
                sumw2f[k] += w * w * f[k];
                sumw2f2[k] += w * w * f[k] * f[k];
            }
        }
        let mean = sumwf.map(|v| v / sumw);
        let se = std::array::from_fn(|k| {
            ((sumw2f2[k] - 2. * mean[k] * sumw2f[k] + mean[k] * mean[k] * sumw2).max(0.)
                / (sumw * sumw))
                .sqrt()
        });
        Reference {
            mean,
            se,
            ess: sumw * sumw / sumw2,
            valid,
        }
    }

    fn learned_proposal(tree: &SphereTree) -> DockingProposal {
        let shape_hash = hash_bytes(&serde_json::to_vec(&tree.shape).unwrap());
        let covariance = |diagonal: [f64; 6], cross: f64| {
            let mut l = [[0.; 6]; 6];
            for k in 0..6 {
                l[k][k] = diagonal[k];
            }
            l[4][0] = cross;
            let matrix: [[f64; 6]; 6] = std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..6).map(|k| l[i][k] * l[j][k]).sum())
            });
            matrix
        };
        // Deliberately unequal weights, off-diagonal translation/rotation
        // covariance, angular centers and scales unrelated to the isotropic target.
        let model = json!({
            "angular_length":1.,"shape_sha256":shape_hash,
            "coordinate_convention":"anchor-body-relative",
            "anchors":[{"position":[2.1,0.,0.],"rotation":IDENTITY},
                {"position":[0.,2.7,0.],"rotation":cayley([0.3,-0.5,0.2])}],
            "means":[[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]],
            "covariances":[covariance([0.35,0.8,0.6,0.65,1.2,0.9],0.3),
                covariance([0.9,0.45,0.7,1.3,0.6,0.8],-0.25)],
            "weights":[0.35,0.65]
        });
        let frozen = FrozenRelativePoseProposal::from_json_str_open(
            &model.to_string(),
            [2. * (WALL + tree.bound); 3],
            0.15,
            &shape_hash,
        )
        .unwrap();
        DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0.45, [0.; 3]).unwrap()
    }

    #[derive(Debug)]
    struct Chain {
        mean: [f64; OBS],
        se: [f64; OBS],
        counts: ClusterPhaseCounts,
        fused_proposed: usize,
        fused_accepted: usize,
        mixed_accepted: usize,
        corrected_fused_accepted: usize,
    }
    fn chain(tree: &SphereTree, wall: &Container, mut poses: Vec<Pose>, seed: u64) -> Chain {
        let cfg = ClusterPhaseConfig {
            duration: 0.5,
            dimer_rate: 4.,
            trimer_rate: 1.,
            transport_probability: 0.7,
            local_translation_std_a: 0.5,
            local_small_angle_std_degrees: 20.,
            transport_charts: TransportCharts::Members,
            anchor_count: 2,
            anchor_contact_uniform_probability: Some(0.2),
            contact_fusion: Some(FusionConfig {
                fused_weight: 0.5,
                max_fused: 8,
                candidates_per_member_anchor: 4,
                max_pair_candidates: 32,
            }),
            ..ClusterPhaseConfig::default()
        };
        let gate = GateOptions {
            max_cells: 1,
            max_depth: 0,
            min_width: 0.,
        };
        let engine = ClusterPhase::new(
            tree,
            wall,
            RD,
            Z,
            LAMBDA,
            gate,
            cfg,
            Some(learned_proposal(tree)),
        )
        .unwrap();
        let mut clock = StdRng::seed_from_u64(seed + 1);
        let mut proposal_rng = StdRng::seed_from_u64(seed + 2);
        let mut gate_rng = StdRng::seed_from_u64(seed + 3);
        let mut accept_rng = StdRng::seed_from_u64(seed + 4);
        let mut bias_rng = StdRng::seed_from_u64(seed + 5);
        let mut single_rng = StdRng::seed_from_u64(seed + 6);
        let mut single_gate = StdRng::seed_from_u64(seed + 7);
        let mut bias_state = None;
        let (burn, blocks, block_size) = (2_000, 32, 500);
        let mut block_means = vec![[0.; OBS]; blocks];
        let mut counts = ClusterPhaseCounts::default();
        let (
            mut fused_proposed,
            mut fused_accepted,
            mut mixed_accepted,
            mut corrected_fused_accepted,
        ) = (0, 0, 0, 0);
        for sweep in 0..burn + blocks * block_size {
            for _ in 0..poses.len() {
                let i = single_rng.random_range(0..poses.len());
                let proposed = uniform_pose(&mut single_rng);
                let trial = RigidSubset::new(tree, &poses, &[i], i, proposed, RD).unwrap();
                if trial.hard_valid(Some(wall), [0.; 3]) {
                    let sampled = trial.sample(&mut single_gate, LAMBDA, Z, gate).unwrap();
                    if single_rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                        < sampled.log_weight.min(0.)
                    {
                        poses[i] = proposed;
                    }
                }
            }
            let (c, records) = engine
                .run(
                    &mut poses,
                    &mut clock,
                    &mut proposal_rng,
                    &mut gate_rng,
                    &mut accept_rng,
                    &mut bias_rng,
                    None,
                    &mut bias_state,
                    true,
                )
                .unwrap();
            let (_, obs) = exact(&poses).expect("production kernel created overlap");
            assert!(poses.iter().all(|&p| wall.contains(p)));
            if sweep < burn {
                continue;
            }
            counts.add(&c);
            for row in records {
                if row["kind"] != "cluster_event" {
                    continue;
                }
                let info = &row["proposal"];
                let source_fused = info["source_kind"] == "fused";
                let target_fused = info["target_kind"] == "fused";
                if !source_fused && !target_fused {
                    continue;
                }
                fused_proposed += 1;
                if row["accepted"] == true {
                    fused_accepted += 1;
                    mixed_accepted += usize::from(source_fused != target_fused);
                    corrected_fused_accepted += usize::from(
                        info["log_reverse_forward"]
                            .as_f64()
                            .is_some_and(|v| v.abs() > 1e-6),
                    );
                }
            }
            let block = (sweep - burn) / block_size;
            for k in 0..OBS {
                block_means[block][k] += obs[k] / block_size as f64;
            }
        }
        let mean =
            std::array::from_fn(|k| block_means.iter().map(|b| b[k]).sum::<f64>() / blocks as f64);
        let se = std::array::from_fn(|k| {
            (block_means
                .iter()
                .map(|b| (b[k] - mean[k]).powi(2))
                .sum::<f64>()
                / (blocks * (blocks - 1)) as f64)
                .sqrt()
        });
        Chain {
            mean,
            se,
            counts,
            fused_proposed,
            fused_accepted,
            mixed_accepted,
            corrected_fused_accepted,
        }
    }
    #[test]
    fn fused_cluster_phase_matches_independent_sphere_depletion_equilibrium() {
        assert!(RD < 2. / 3.0_f64.sqrt() - 1.);
        let tree = SphereTree::new(Shape {
            name: "analytic sphere".into(),
            volume: 4. * PI / 3.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })
        .unwrap();
        let wall = Container::new(WALL, &tree).unwrap();
        let reference = reference(1_000_000, 4);
        assert!(reference.valid > 100_000 && reference.ess > 80_000.);
        let starts = [
            vec![
                pose([-1.8, -1.1, 0.]),
                pose([1.8, -1.1, 0.]),
                pose([0., 2., 0.]),
                pose([0., 0., 2.2]),
            ],
            vec![
                pose([0., 0., 0.]),
                pose([2.05, 0., 0.]),
                pose([0., 2.05, 0.]),
                pose([0., 0., 2.05]),
            ],
        ];
        let chains: Vec<_> = starts
            .into_iter()
            .enumerate()
            .map(|(i, p)| chain(&tree, &wall, p, 849001 + i as u64))
            .collect();
        eprintln!("fused_sphere_reference: {reference:?}");
        for (i, c) in chains.iter().enumerate() {
            eprintln!("fused_sphere_chain_{i}: {c:?}");
            assert!(c.counts.transport_events > 3_000);
            assert!(
                c.fused_proposed > 500
                    && c.fused_accepted > 20
                    && c.mixed_accepted > 5
                    && c.corrected_fused_accepted > 20,
                "nonvacuous fusion coverage required: {c:?}"
            );
            assert!(c.counts.accepted_attachments > 50 && c.counts.accepted_detachments > 50);
            for k in 0..OBS {
                let tol = 5. * (c.se[k].powi(2) + reference.se[k].powi(2)).sqrt() + 0.002;
                assert!(
                    (c.mean[k] - reference.mean[k]).abs() < tol,
                    "chain {i}, observable {k}: {} +/- {} vs {} +/- {}",
                    c.mean[k],
                    c.se[k],
                    reference.mean[k],
                    reference.se[k]
                );
            }
        }
        for k in 0..OBS {
            let tol = 5. * (chains[0].se[k].powi(2) + chains[1].se[k].powi(2)).sqrt() + 0.002;
            assert!(
                (chains[0].mean[k] - chains[1].mean[k]).abs() < tol,
                "initialization discrepancy {k}"
            );
        }
    }
}
