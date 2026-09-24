//! Periodic all-mobile integration, independent periodic image enumeration,
//! reciprocal branch replay, and exact restart. This checks runner wiring,
//! not convergence of a protein assembly campaign.
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
const LENGTHS: Vec3 = [4.; 3];
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
            "tetramer-mc-periodic-posterior-{label}-{}",
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
            "boundary":{"kind":"periodic"},
            "seed":SEED,"depletant_radius":RD,"reservoir_density":Z,
            "poisson_lambda_ratio":4.,"endpoint_gate":GATE,
            "global_probability":0.5,"learned_uniform_weight":0.2,
            "local_translation_std_A":0.25,"local_small_angle_std_degrees":20.,
            "initial_poses":[pose([0.15,2.,2.]),pose([3.35,2.4,2.]),pose([0.95,2.,2.])]
        });
        let tree = SphereTree::new(serde_json::from_value::<Shape>(shape)?)?;
        let model =
            FrozenRelativePoseProposal::from_json_str(&atlas.to_string(), LENGTHS, 0.2, &hash)?;
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
        self.model = FrozenRelativePoseProposal::from_json_str(
            &wrapped.to_string(),
            LENGTHS,
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

fn periodic_delta(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|axis| {
        (-1..=1)
            .map(|image| a[axis] - b[axis] + image as f64 * LENGTHS[axis])
            .min_by(|x, y| x.abs().total_cmp(&y.abs()).then_with(|| x.total_cmp(y)))
            .unwrap()
    })
}

// Exhaustive 27-image enumeration is deliberately independent of the production
// nearest-image construction. The pair reach is less than half the box length.
fn periodic_environment<'a>(
    tree: &'a SphereTree,
    poses: &[Pose],
    moving: usize,
    endpoints: Option<[Pose; 2]>,
    only_anchor: Option<usize>,
) -> Environment<'a> {
    let mut fixed = Vec::new();
    let mut labels = Vec::new();
    let reach = 2. * (CORE + RD);
    let guard = 512. * f64::EPSILON * (1. + reach + LENGTHS[0]);
    for (j, pose) in poses.iter().enumerate() {
        if j == moving || only_anchor.is_some_and(|anchor| j != anchor) {
            continue;
        }
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let image = [x, y, z];
                    let shifted = Pose {
                        position: std::array::from_fn(|k| {
                            pose.position[k] + image[k] as f64 * LENGTHS[k]
                        }),
                        orientation: pose.orientation,
                    };
                    if endpoints.is_none_or(|pair| {
                        pair.iter()
                            .any(|p| norm(sub(shifted.position, p.position)) <= reach + guard)
                    }) {
                        fixed.push(Placed::new(shifted));
                        labels.push((j, image));
                    }
                }
            }
        }
    }
    Environment {
        tree,
        fixed,
        labels,
        rd: RD,
    }
}

fn relative(pose: Pose, anchor: Pose) -> Pose {
    Pose {
        position: matvec(
            transpose(rotation(anchor.orientation)),
            periodic_delta(pose.position, anchor.position),
        ),
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
    let mut nulls = 0;
    let mut posterior_nulls = 0;
    let mut wrapped_candidates = 0;
    let mut image_gates = 0;
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
                nulls += 1;
                posterior_nulls += usize::from(posterior);
                assert!(
                    info["null_reason"]
                        .as_str()
                        .is_some_and(|reason| !reason.is_empty())
                );
                assert_eq!(row["accepted"], false);
                assert_eq!(row["hard_valid"], false);
                assert!(row["gate"].is_null());
                assert_eq!(row["retained_pose"], json!(old));
                continue;
            };
            assert!((0..3).all(|k| new.position[k] >= 0. && new.position[k] < LENGTHS[k]));
            wrapped_candidates += usize::from(
                (0..3).any(|k| (new.position[k] - old.position[k]).abs() > 0.5 * LENGTHS[k]),
            );
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
                // Independently form the full capture mixture, including the
                // uniform density, from unwrapped virtual component densities.
                let capture_log = |pose: Pose| -> Result<f64> {
                    let uniform = 0.2_f64.ln() - LENGTHS.iter().product::<f64>().ln();
                    let learned = 0.8_f64.ln() + full_density(relative(pose, anchor))?;
                    let maximum = uniform.max(learned);
                    Ok(maximum + ((uniform - maximum).exp() + (learned - maximum).exp()).ln())
                };
                let expected = capture_log(old)? - capture_log(new)?;
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
            // Independent minimum-image core checks and exhaustive spectator
            // images include bodies across faces, not just the selected anchor.
            let valid = poses.iter().enumerate().all(|(j, p)| {
                j == moving || norm(periodic_delta(new.position, p.position)) >= 2. * CORE
            });
            assert_eq!(row["hard_valid"], valid);
            if valid {
                let all = periodic_environment(&fixture.tree, &poses, moving, None, None);
                let envelope_env =
                    periodic_environment(&fixture.tree, &poses, moving, Some([old, new]), None);
                image_gates += usize::from(
                    envelope_env
                        .labels
                        .iter()
                        .any(|(_, image)| *image != [0; 3]),
                );
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
                    let anchor_only =
                        periodic_environment(&fixture.tree, &poses, moving, None, Some(anchor));
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
    assert!(
        nulls > 0 && posterior_nulls > 0,
        "must retain out-of-cube nulls"
    );
    assert!(
        wrapped_candidates > 0 && image_gates > 0,
        "must exercise periodic crossings/images"
    );
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
fn periodic_zero_posterior_probability_preserves_existing_physical_draws() -> Result<()> {
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
fn periodic_reciprocal_assembly_replays_images_traces_nulls_and_resumes_exactly() -> Result<()> {
    let mut fixture = Fixture::new("reciprocal-replay")?;
    fixture.enable_reciprocal(vec![true, false])?;
    assert!(fixture.model.has_reciprocal_components());
    for (label, correlation) in [("reciprocal-c0", 0.), ("reciprocal-c09", 0.9)] {
        fixture.config["frozen_posterior"] = json!({"probability":0.5,"correlation":correlation});
        fixture.save_config()?;
        let summary = fixture.run(label, SWEEPS, None)?;
        let effective = fixture.read(label, "config.json")?;
        assert_eq!(effective["boundary"], json!({"kind":"periodic"}));
        assert_eq!(effective["uniform_proposal_cube_lengths"], json!(LENGTHS));
        assert!(
            effective["coordinate_frame_convention"]
                .as_str()
                .unwrap()
                .contains("canonical periodic centers")
        );
        let manifest = fixture.read(label, "manifest.json")?;
        assert_eq!(manifest["boundary"], json!({"kind":"periodic"}));
        assert_eq!(manifest["bath_wall_permeable"], false);
        assert_eq!(
            manifest["model_sha256"],
            hash_file(&fixture.root.join("model.json"))?
        );
        assert_eq!(
            manifest["frozen_posterior"],
            fixture.config["frozen_posterior"]
        );
        assert_eq!(
            fs::read(
                fixture
                    .root
                    .join(label)
                    .join("provenance/frozen-relative-model.json")
            )?,
            fs::read(fixture.root.join("model.json"))?
        );
        assert_eq!(
            fs::read(
                fixture
                    .root
                    .join(label)
                    .join("provenance/input-config.json")
            )?,
            fs::read(fixture.root.join("config.json"))?
        );
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
fn periodic_posterior_keeps_collective_and_adaptive_guards() -> Result<()> {
    let mut fixture = Fixture::new("forbidden-modes")?;
    fixture.enable_reciprocal(vec![true, false])?;
    fixture.config["frozen_posterior"] = json!({"probability":0.5,"correlation":0.9});
    let base = fixture.config.clone();
    for key in ["gca_probability", "center_shift_probability"] {
        fixture.config = base.clone();
        fixture.config[key] = json!(0.1);
        fixture.save_config()?;
        let error = fixture.run(key, 1, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("spherical GCA/center shift requires a spherical boundary"),
            "{error:#}"
        );
        assert!(!fixture.root.join(key).join("moves.jsonl").exists());
    }
    fixture.config = base;
    fixture.config["atlas_transport"] = json!({"mean_gain":0.,"covariance_gain":0.,"weight_gain":0.,
        "mean_noise":0.,"covariance_noise":0.,"weight_noise":0.,"initialization":"reference"});
    fixture.save_config()?;
    let error = fixture.run("atlas", 1, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("atlas transport requires a spherical boundary"),
        "{error:#}"
    );
    assert!(!fixture.root.join("atlas/moves.jsonl").exists());
    Ok(())
}
