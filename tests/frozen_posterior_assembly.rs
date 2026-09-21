//! Exercise the actual all-mobile runner, including its named RNG streams,
//! global anchor labels, capture mixture, posterior correction, and bath gate.
//! The generic conditional stationarity checks live in involution_depletion.rs;
//! this test instead detects integration errors in the runner's wiring.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    depletion::{self, Envelope, GateOptions},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::{Method, RunOptions, hash_file, run},
};

const CORE: f64 = 0.35;
const RD: f64 = 0.5;
const Z: f64 = 0.35;
const WALL: f64 = 2.;
const SEED: u64 = 104392003;
const SWEEPS: u64 = 512;
const GATE: GateOptions = GateOptions {
    max_cells: 1,
    max_depth: 0,
    min_width: 0.,
};

struct Fixture {
    root: PathBuf,
    config: Value,
    tree: SphereTree,
    model: FrozenRelativePoseProposal,
    reciprocal_flags: Vec<bool>,
}
impl Fixture {
    fn new(label: &str) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "tetramer-mc-frozen-posterior-{label}-{}",
            std::process::id()
        ));
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir(&root)?;
        let shape = json!({"name":"tiny sphere","atoms":[{"center":[0.,0.,0.],"radius":CORE}]});
        fs::write(root.join("shape.json"), shape.to_string())?;
        let hash = hash_file(&root.join("shape.json"))?;
        let covariance = |variance: f64| -> Vec<Vec<f64>> {
            (0..6)
                .map(|i| (0..6).map(|j| if i == j { variance } else { 0. }).collect())
                .collect()
        };
        let atlas = json!({
            "angular_length":1.,
            "anchors":[
                {"position":[0.95,0.,0.],"rotation":IDENTITY},
                {"position":[0.,0.95,0.],"rotation":cayley([0.15,-0.2,0.1])}
            ],
            "means":vec![[0_f64; 6]; 2],
            "covariances":[covariance(0.25),covariance(0.5)],
            "weights":[0.35,0.65],
            "shape_sha256":hash,
            "coordinate_convention":"anchor-body-relative"
        });
        fs::write(root.join("model.json"), atlas.to_string())?;
        let pose = |position: Vec3| json!({"position":position,"orientation":[1.,0.,0.,0.]});
        let config = json!({
            "shape":"shape.json", "box_lengths":vec![4.; 3],
            "boundary":{"kind":"spherical","radius":WALL},
            "seed":SEED,"depletant_radius":RD,"reservoir_density":Z,
            "poisson_lambda_ratio":4.,"endpoint_gate":GATE,
            "global_probability":0.5,"learned_uniform_weight":0.2,
            "local_translation_std_A":0.25,"local_small_angle_std_degrees":20.,
            "initial_poses":[pose([-0.8,0.,0.]),pose([0.,0.4,0.]),pose([0.8,0.,0.])]
        });
        let tree = SphereTree::new(serde_json::from_value::<Shape>(shape)?)?;
        let model = FrozenRelativePoseProposal::from_json_str_open(
            &atlas.to_string(),
            [2. * (WALL + CORE); 3],
            0.2,
            &hash,
        )?;
        Ok(Self {
            root,
            config,
            tree,
            model,
            reciprocal_flags: vec![false; 2],
        })
    }
    fn enable_reciprocal(&mut self, flags: Vec<bool>) -> Result<()> {
        let base: Value = serde_json::from_slice(&fs::read(self.root.join("model.json"))?)?;
        assert_eq!(flags.len(), base["weights"].as_array().unwrap().len());
        let wrapped = json!({"schema":"reciprocal-pose-mixture-v1", "base_model":base,
            "reciprocal_components":flags});
        fs::write(self.root.join("model.json"), wrapped.to_string())?;
        self.model = FrozenRelativePoseProposal::from_json_str_open(
            &wrapped.to_string(),
            [2. * (WALL + CORE); 3],
            0.2,
            &hash_file(&self.root.join("shape.json"))?,
        )?;
        self.reciprocal_flags = flags;
        Ok(())
    }
    fn save_config(&self) -> Result<()> {
        fs::write(self.root.join("config.json"), self.config.to_string())?;
        Ok(())
    }
    fn run(&self, name: &str, sweeps: u64, resume: Option<&str>) -> Result<Value> {
        run(RunOptions {
            config: self.root.join("config.json"),
            model: Some(self.root.join("model.json")),
            method: Method::Learned,
            out: self.root.join(name),
            sweeps,
            sample_every: 8,
            resume: resume.map(|folder| self.root.join(folder).join("checkpoint.json")),
            write_gsd: false,
            record_moves: true,
        })
    }
    fn read(&self, folder: &str, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(folder).join(name),
        )?)?)
    }
    fn moves(&self, folder: &str) -> Result<Vec<Value>> {
        fs::read_to_string(self.root.join(folder).join("moves.jsonl"))?
            .lines()
            .map(|line| {
                let mut row: Value = serde_json::from_str(line)?;
                row.as_object_mut().unwrap().remove("sampler_cpu_seconds");
                Ok(row)
            })
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn stream(sweep: u64, label: &str) -> StdRng {
    let mut hash = Sha256::new();
    hash.update(b"tetramer-mc-rng-v1");
    hash.update(SEED.to_le_bytes());
    hash.update(sweep.to_le_bytes());
    hash.update(label.as_bytes());
    StdRng::from_seed(hash.finalize().into())
}

fn relative(pose: Pose, anchor: Pose) -> Pose {
    Pose {
        position: anchor.inverse(pose.position),
        orientation: quaternion(matmul(
            transpose(rotation(anchor.orientation)),
            rotation(pose.orientation),
        )),
    }
}

fn assert_pose_near(actual: Pose, expected: Pose) {
    assert!(norm(sub(actual.position, expected.position)) < 2e-9);
    assert!(
        rotation(actual.orientation)
            .iter()
            .flatten()
            .zip(rotation(expected.orientation).iter().flatten())
            .all(|(a, b)| (a - b).abs() < 2e-9)
    );
}

// Independent test construction of virtual labels, rather than calling the
// production DockingProposal wrapper or trusting its virtual-branch table.
fn branches(fixture: &Fixture) -> Vec<(usize, bool, f64)> {
    fixture
        .model
        .component_weights()
        .into_iter()
        .zip(&fixture.reciprocal_flags)
        .enumerate()
        .flat_map(|(base, (weight, reciprocal))| {
            if *reciprocal {
                vec![(base, false, weight / 2.), (base, true, weight / 2.)]
            } else {
                vec![(base, false, weight)]
            }
        })
        .collect()
}

fn reciprocal_pose(value: Pose) -> Pose {
    // Matrix inversion is intentionally separate from invert_relative_pose,
    // used by the production proposal implementation.
    let inverse_rotation = transpose(rotation(value.orientation));
    Pose {
        position: scale(matvec(inverse_rotation, value.position), -1.),
        orientation: quaternion(inverse_rotation),
    }
}

fn involution(fixture: &Fixture, correlation: f64) -> Result<FixedBasinInvolution> {
    let model = &fixture.model;
    let virtual_branches = branches(fixture);
    let weights: Vec<_> = virtual_branches.iter().map(|b| b.2).collect();
    let base = model.component_parameters();
    let parameters = virtual_branches
        .iter()
        .map(|&(index, _, weight)| {
            let mut parameter = base[index].clone();
            parameter.weight = weight;
            parameter
        })
        .collect();
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
    FixedBasinInvolution::new(parameters, model.angular_length(), correlation, pairs)
}

fn replay(fixture: &Fixture, folder: &str, correlation: f64) -> Result<()> {
    let rows = fixture.moves(folder)?;
    assert_eq!(rows.len(), 3 * SWEEPS as usize);
    let mut poses: Vec<Pose> = serde_json::from_value(fixture.config["initial_poses"].clone())?;
    let map = involution(fixture, correlation)?;
    let virtual_branches = branches(fixture);
    let reported_branches = fixture.model.virtual_branches();
    assert_eq!(reported_branches.len(), virtual_branches.len());
    for (actual, &(index, inverted, weight)) in reported_branches.iter().zip(&virtual_branches) {
        assert_eq!(actual.component_index, index);
        assert_eq!(actual.inverted, inverted);
        assert_eq!(actual.weight, weight);
    }
    let base_components = fixture
        .model
        .component_parameters()
        .into_iter()
        .map(|mut parameter| {
            parameter.weight = 1.;
            FrozenRelativePoseProposal::from_components_open(
                vec![parameter],
                fixture.model.angular_length(),
                fixture.model.box_lengths(),
                fixture.model.uniform_weight(),
                fixture.model.shape_sha256(),
                fixture.model.shape_sha256(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let component_density = |label: usize, value: Pose| -> Result<f64> {
        let (base, inverted, _) = virtual_branches[label];
        let value = if inverted {
            reciprocal_pose(value)
        } else {
            value
        };
        base_components[base].relative_log_density(value.position, rotation(value.orientation))
    };
    let full_density = |value: Pose| -> Result<f64> {
        let logs = virtual_branches
            .iter()
            .enumerate()
            .map(|(label, &(_, _, weight))| Ok(weight.ln() + component_density(label, value)?))
            .collect::<Result<Vec<f64>>>()?;
        let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(maximum
            + logs
                .iter()
                .map(|value| (value - maximum).exp())
                .sum::<f64>()
                .ln())
    };
    let has_reciprocal = fixture.reciprocal_flags.iter().any(|&flag| flag);
    let mut posterior_attempts = [0_u64; 3];
    let mut posterior_accepted = [0_u64; 3];
    let mut anchor_pairs = [[0_u64; 3]; 3];
    let mut captures = 0;
    let mut posterior_gaussians = 0;
    let mut posterior_uniforms = 0;
    let mut shielding_witnesses = 0;
    let mut largest_correction = 0_f64;
    let mut inverse_sources = 0;
    let mut inverse_targets = 0;
    let mut inverse_captures = 0;
    for (index, sweep_rows) in rows.chunks_exact(3).enumerate() {
        let sweep = index as u64 + 1;
        let mut order = [0_usize, 1, 2];
        order.shuffle(&mut stream(sweep, "schedule"));
        let mut choice = stream(sweep, "choice");
        let mut posterior_choice = stream(sweep, "posterior-choice");
        let mut gate_rng = stream(sweep, "gate");
        // StdRng intentionally is not Clone. Keep a second stream synchronized
        // until the first valid posterior Gaussian in each sweep, then spend
        // that copy on the anchor-only negative control.
        let mut witness_rng = Some(stream(sweep, "gate"));
        let mut accept_rng = stream(sweep, "accept");
        for (update, row) in sweep_rows.iter().enumerate() {
            assert_eq!(row["sweep"], sweep);
            assert_eq!(row["update_in_sweep"], update);
            let moving = row["moving_index"].as_u64().unwrap() as usize;
            assert_eq!(moving, order[update]);
            let old: Pose = serde_json::from_value(row["old_pose"].clone())?;
            assert_eq!(old, poses[moving]);
            let global = choice.random::<f64>() < 0.5;
            let posterior = global && posterior_choice.random::<f64>() < 0.5;
            let info = &row["proposal"];
            assert_eq!(row["kind"], if global { "global" } else { "local" });
            assert_eq!(
                info["kernel"],
                if posterior {
                    "frozen-posterior"
                } else if global {
                    "full-mixture-capture"
                } else {
                    "local"
                }
            );
            let anchor = if global {
                let anchor = info["anchor_index"].as_u64().unwrap() as usize;
                assert!(anchor < 3 && anchor != moving);
                if posterior {
                    posterior_attempts[moving] += 1;
                    anchor_pairs[moving][anchor] += 1;
                    assert_eq!(info["moving_index"], moving);
                } else {
                    captures += 1;
                }
                Some(anchor)
            } else {
                None
            };
            let candidate: Option<Pose> = serde_json::from_value(row["proposed_pose"].clone())?;
            let Some(new) = candidate else {
                assert_eq!(row["accepted"], false);
                assert_eq!(row["retained_pose"], json!(old));
                continue;
            };
            let correction = info["log_reverse_forward"].as_f64().unwrap();
            if posterior && info["branch"] == "involution" {
                posterior_gaussians += 1;
                let anchor = poses[anchor.unwrap()];
                let old_relative = relative(old, anchor);
                let new_relative = relative(new, anchor);
                let old_g = full_density(old_relative)?;
                let new_g = full_density(new_relative)?;
                let expected = old_g - new_g;
                assert!((correction - expected).abs() < 2e-9);
                largest_correction = largest_correction.max(correction.abs());
                let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
                let inverse: BasinTrace =
                    serde_json::from_value(info["step"]["inverse_trace"].clone())?;
                let (source_base, source_inverse, source_weight) = virtual_branches[trace.source];
                let (target_base, target_inverse, target_weight) = virtual_branches[trace.target];
                assert_eq!(inverse.source, trace.target);
                assert_eq!(inverse.target, trace.source);
                if has_reciprocal {
                    assert_eq!(info["source_component_index"], source_base);
                    assert_eq!(info["target_component_index"], target_base);
                    assert_eq!(info["source_inverted"], source_inverse);
                    assert_eq!(info["target_inverted"], target_inverse);
                    inverse_sources += usize::from(source_inverse);
                    inverse_targets += usize::from(target_inverse);
                } else {
                    for key in [
                        "source_component_index",
                        "target_component_index",
                        "source_inverted",
                        "target_inverted",
                    ] {
                        assert!(info.get(key).is_none(), "legacy serialization gained {key}");
                    }
                }
                let input = if source_inverse {
                    reciprocal_pose(old_relative)
                } else {
                    old_relative
                };
                let forward = map.apply(input, &trace)?;
                let output = if target_inverse {
                    reciprocal_pose(forward.pose)
                } else {
                    forward.pose
                };
                assert_pose_near(output, new_relative);
                let reverse_input = if target_inverse {
                    reciprocal_pose(new_relative)
                } else {
                    new_relative
                };
                let back = map.apply(reverse_input, &inverse)?;
                let reverse_output = if source_inverse {
                    reciprocal_pose(back.pose)
                } else {
                    back.pose
                };
                assert_pose_near(reverse_output, old_relative);
                for (key, expected) in [
                    ("log_extended_jacobian", forward.log_extended_jacobian),
                    ("log_auxiliary_ratio", forward.log_auxiliary_ratio),
                    ("log_correction", forward.log_correction),
                ] {
                    assert!((info["step"][key].as_f64().unwrap() - expected).abs() < 2e-9);
                }
                for (key, expected) in [
                    ("source_latent", forward.source_latent),
                    ("target_latent", forward.target_latent),
                ] {
                    let actual: [f64; 6] = serde_json::from_value(info["step"][key].clone())?;
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| (a - b).abs() < 2e-9)
                    );
                }
                let source_log = component_density(trace.source, old_relative)?;
                let target_log = component_density(trace.target, new_relative)?;
                let source_probability = source_weight.ln() + source_log - old_g;
                let reverse_probability = target_weight.ln() + target_log - new_g;
                let label_correction = reverse_probability + source_weight.ln()
                    - source_probability
                    - target_weight.ln();
                for (key, expected) in [
                    ("selected_source_log_density", source_log),
                    ("selected_target_log_density", target_log),
                    ("full_old_gaussian_log_density", old_g),
                    ("full_new_gaussian_log_density", new_g),
                    ("source_log_probability", source_probability),
                    ("inverse_source_log_probability", reverse_probability),
                    ("label_log_reverse_forward", label_correction),
                ] {
                    assert!(
                        (info[key].as_f64().unwrap() - expected).abs() < 2e-9,
                        "{key}"
                    );
                }
                assert!(
                    (info["expanded_log_reverse_forward"].as_f64().unwrap() - correction).abs()
                        < 2e-9
                );
            } else if posterior {
                assert_eq!(info["branch"], "uniform");
                posterior_uniforms += 1;
                assert_eq!(correction, 0.);
            } else if global {
                let anchor = poses[anchor.unwrap()];
                let expected = fixture.model.log_density(&old, &anchor)?
                    - fixture.model.log_density(&new, &anchor)?;
                assert!((correction - expected).abs() < 2e-9);
                if has_reciprocal && info["branch"] == "learned" {
                    let base = info["component_index"].as_u64().unwrap() as usize;
                    let inverse = info["component_inverted"].as_bool().unwrap();
                    assert!(base < fixture.reciprocal_flags.len());
                    assert!(!inverse || fixture.reciprocal_flags[base]);
                    inverse_captures += usize::from(inverse);
                } else {
                    assert!(info.get("component_inverted").is_none());
                }
            } else {
                assert_eq!(correction, 0.);
            }
            // Independent single-sphere predicates check the physical wall and
            // every spectator, including the spectator not chosen as anchor.
            let valid = norm(new.position) + CORE <= WALL
                && poses
                    .iter()
                    .enumerate()
                    .all(|(j, p)| j == moving || norm(sub(new.position, p.position)) >= 2. * CORE);
            assert_eq!(row["hard_valid"], valid);
            if valid {
                let all = Environment {
                    tree: &fixture.tree,
                    fixed: poses
                        .iter()
                        .enumerate()
                        .filter_map(|(j, &p)| (j != moving).then_some(Placed::new(p)))
                        .collect(),
                    labels: (0..3)
                        .filter(|&j| j != moving)
                        .map(|j| (j, [0; 3]))
                        .collect(),
                    rd: RD,
                };
                // A distant spectator can change a conservative envelope's
                // subdivision without changing point membership. Match that
                // representation, then replay every point against ALL bodies.
                let relevant: Vec<_> = (0..3)
                    .filter(|&j| {
                        j != moving
                            && [old, new].iter().any(|p| {
                                norm(sub(poses[j].position, p.position)) <= 2. * (CORE + RD) + 1e-12
                            })
                    })
                    .collect();
                let envelope_env = Environment {
                    tree: &fixture.tree,
                    fixed: relevant.iter().map(|&j| Placed::new(poses[j])).collect(),
                    labels: relevant.iter().map(|&j| (j, [0; 3])).collect(),
                    rd: RD,
                };
                let envelope = Envelope::build(&envelope_env, old, new, GATE)?;
                let gate = depletion::sample_with_envelope(
                    &mut gate_rng,
                    &all,
                    old,
                    new,
                    4. * Z,
                    Z,
                    &envelope,
                )?;
                assert_eq!(
                    serde_json::to_value(gate)?,
                    row["gate"],
                    "full spectator replay: sweep {sweep}, body {moving}"
                );
                if posterior && info["branch"] == "involution" && witness_rng.is_some() {
                    let anchor = anchor.unwrap();
                    let anchor_only = Environment {
                        tree: &fixture.tree,
                        fixed: vec![Placed::new(poses[anchor])],
                        labels: vec![(anchor, [0; 3])],
                        rd: RD,
                    };
                    let wrong = depletion::sample_with_envelope(
                        &mut witness_rng.take().unwrap(),
                        &anchor_only,
                        old,
                        new,
                        4. * Z,
                        Z,
                        &envelope,
                    )?;
                    shielding_witnesses += usize::from(wrong.log_weight != gate.log_weight);
                } else if let Some(rng) = &mut witness_rng {
                    depletion::sample_with_envelope(rng, &all, old, new, 4. * Z, Z, &envelope)?;
                }
                let alpha = (correction + gate.log_weight).min(0.);
                assert_eq!(row["log_acceptance"], alpha);
                let accepted = accept_rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < alpha;
                assert_eq!(row["accepted"], accepted);
                if accepted {
                    poses[moving] = new;
                    if posterior && new != old {
                        posterior_accepted[moving] += 1;
                    }
                }
            } else {
                assert_eq!(row["accepted"], false);
                assert!(row["gate"].is_null());
            }
            assert_eq!(row["retained_pose"], json!(poses[moving]));
        }
    }
    assert_eq!(
        fixture.read(folder, "checkpoint.json")?["poses"],
        json!(poses)
    );
    assert!(posterior_attempts.iter().all(|&n| n > 10));
    assert!(posterior_accepted.iter().all(|&n| n > 0));
    assert!((0..3).all(|i| (0..3).all(|j| i == j || anchor_pairs[i][j] > 0)));
    assert!(captures > 20 && posterior_gaussians > 20 && posterior_uniforms > 10);
    assert!(largest_correction > 0.1);
    if has_reciprocal {
        assert!(inverse_sources > 0 && inverse_targets > 0 && inverse_captures > 0);
        eprintln!(
            "Reciprocal c={correlation}: inverted source/target/capture counts {inverse_sources}/{inverse_targets}/{inverse_captures}"
        );
    }
    assert!(
        shielding_witnesses > 0,
        "fixture must distinguish full spectator shielding from anchor-only depletion"
    );
    eprintln!(
        "All-mobile c={correlation}: posterior attempts {posterior_attempts:?}, accepted {posterior_accepted:?}, non-anchor shielding witnesses {shielding_witnesses}"
    );
    Ok(())
}

#[test]
fn posterior_assembly_replays_full_spectator_gate_and_resumes_exactly() -> Result<()> {
    let mut fixture = Fixture::new("replay")?;
    for (label, correlation) in [("c0", 0.), ("c09", 0.9)] {
        fixture.config["frozen_posterior"] = json!({"probability":0.5,"correlation":correlation});
        fixture.save_config()?;
        let summary = fixture.run(label, SWEEPS, None)?;
        assert_eq!(summary["all_bodies_mobile"], true);
        assert_eq!(
            summary["counts"]["selected_body_updates_by_body"],
            json!(vec![SWEEPS; 3])
        );
        replay(&fixture, label, correlation)?;
        if correlation == 0.9 {
            fixture.run("part", 17, None)?;
            fixture.run("resumed", SWEEPS, Some("part"))?;
            assert_eq!(
                fixture.read(label, "checkpoint.json")?,
                fixture.read("resumed", "checkpoint.json")?
            );
            assert_eq!(
                &fixture.moves(label)?[17 * 3..],
                fixture.moves("resumed")?.as_slice()
            );
        }
    }
    Ok(())
}

#[test]
fn zero_posterior_probability_preserves_existing_physical_draws() -> Result<()> {
    let mut fixture = Fixture::new("zero")?;
    fixture.save_config()?;
    fixture.run("absent", 12, None)?;
    fixture.config["frozen_posterior"] = json!({"probability":0.,"correlation":0.9});
    fixture.save_config()?;
    fixture.run("zero", 12, None)?;
    for name in ["config.json", "manifest.json", "summary.json"] {
        assert!(
            fixture
                .read("absent", name)?
                .get("frozen_posterior")
                .is_none()
        );
        assert_eq!(
            fixture.read("zero", name)?["frozen_posterior"],
            fixture.config["frozen_posterior"]
        );
    }
    let mut absent = fixture.read("absent", "checkpoint.json")?;
    let mut zero = fixture.read("zero", "checkpoint.json")?;
    absent.as_object_mut().unwrap().remove("config_sha256");
    zero.as_object_mut().unwrap().remove("config_sha256");
    assert_eq!(absent, zero);
    let absent_moves = fixture.moves("absent")?;
    let mut zero_moves = fixture.moves("zero")?;
    for row in &mut zero_moves {
        row["proposal"].as_object_mut().unwrap().remove("kernel");
    }
    assert_eq!(absent_moves, zero_moves);
    Ok(())
}

#[test]
fn mixed_reciprocal_assembly_replays_virtual_traces_and_resumes_exactly() -> Result<()> {
    let mut fixture = Fixture::new("reciprocal-replay")?;
    fixture.enable_reciprocal(vec![true, false])?;
    assert!(fixture.model.has_reciprocal_components());
    for (label, correlation) in [("reciprocal-c0", 0.), ("reciprocal-c09", 0.9)] {
        fixture.config["frozen_posterior"] = json!({"probability":0.5,"correlation":correlation});
        fixture.save_config()?;
        let summary = fixture.run(label, SWEEPS, None)?;
        assert_eq!(summary["all_bodies_mobile"], true);
        assert_eq!(
            summary["counts"]["selected_body_updates_by_body"],
            json!(vec![SWEEPS; 3])
        );
        replay(&fixture, label, correlation)?;
        let partial = format!("{label}-part");
        let resumed = format!("{label}-resumed");
        fixture.run(&partial, 17, None)?;
        fixture.run(&resumed, SWEEPS, Some(&partial))?;
        assert_eq!(
            fixture.read(label, "checkpoint.json")?,
            fixture.read(&resumed, "checkpoint.json")?
        );
        assert_eq!(
            &fixture.moves(label)?[17 * 3..],
            fixture.moves(&resumed)?.as_slice()
        );
    }
    Ok(())
}

#[test]
fn active_reciprocal_model_rejects_adaptive_atlas_before_sampling() -> Result<()> {
    let mut fixture = Fixture::new("reciprocal-adaptive-guard")?;
    fixture.enable_reciprocal(vec![true, false])?;
    fixture.config["atlas_transport"] = json!({"mean_gain":0.,"covariance_gain":0.,"weight_gain":0.,
        "mean_noise":0.,"covariance_noise":0.,"weight_noise":0.,"initialization":"reference"});
    fixture.save_config()?;
    let error = fixture.run("forbidden", 1, None).unwrap_err();
    assert!(
        error.to_string().contains(
            "Reciprocal contact proposals currently require immutable capture/posterior charts"
        ),
        "{error:#}"
    );
    assert!(!fixture.root.join("forbidden/moves.jsonl").exists());
    Ok(())
}
