//! Sequential, all-mobile periodic Monte Carlo. Every sweep has independent
//! named RNG streams derived from (master seed, absolute sweep, stream label),
//! allowing exact checkpoint continuation without serializing opaque RNG state.
use crate::{
    depletion::{self, GateOptions},
    geometry::{Environment, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    trajectory::Trajectory,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

fn default_lambda() -> f64 {
    16.
}
fn default_global() -> f64 {
    0.5
}
fn default_translation() -> f64 {
    0.2
}
fn default_angle() -> f64 {
    1.
}
fn default_floor() -> f64 {
    0.1
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub monomer_shape: Option<PathBuf>,
    pub shape: PathBuf,
    pub box_lengths: Vec3,
    pub initial_poses: Vec<Pose>,
    pub seed: u64,
    pub depletant_radius: f64,
    pub reservoir_density: f64,
    #[serde(default = "default_lambda")]
    pub poisson_lambda_ratio: f64,
    #[serde(default = "default_global")]
    pub global_probability: f64,
    #[serde(default = "default_translation", rename = "local_translation_std_A")]
    pub local_translation_std_a: f64,
    #[serde(default = "default_angle")]
    pub local_small_angle_std_degrees: f64,
    #[serde(default = "default_floor")]
    pub learned_uniform_weight: f64,
    #[serde(default)]
    pub endpoint_gate: GateOptions,
    #[serde(default)]
    pub seed_labels: Vec<usize>,
    #[serde(default)]
    pub fixed_body_indices: Vec<usize>,
    #[serde(default)]
    pub metadata: Value,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.initial_poses.len() >= 2,
            "at least two bodies required"
        );
        ensure!(
            self.fixed_body_indices.is_empty(),
            "only all-mobile systems supported"
        );
        ensure!(
            self.box_lengths.iter().all(|x| x.is_finite() && *x > 0.),
            "invalid periodic lengths"
        );
        ensure!(
            self.depletant_radius.is_finite()
                && self.depletant_radius >= 0.
                && self.reservoir_density.is_finite()
                && self.reservoir_density >= 0.,
            "invalid physical bath"
        );
        ensure!(
            self.poisson_lambda_ratio.is_finite() && self.poisson_lambda_ratio > 0.,
            "invalid auxiliary intensity ratio"
        );
        ensure!(
            self.global_probability.is_finite() && (0. ..=1.).contains(&self.global_probability),
            "invalid scheduling probability"
        );
        ensure!(
            self.local_translation_std_a.is_finite()
                && self.local_translation_std_a >= 0.
                && self.local_small_angle_std_degrees.is_finite()
                && self.local_small_angle_std_degrees >= 0.,
            "invalid local step"
        );
        ensure!(
            self.seed_labels
                .iter()
                .all(|&i| i < self.initial_poses.len()),
            "invalid seed label"
        );
        for p in &self.initial_poses {
            p.validate()?;
        }
        self.endpoint_gate.validate()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    LocalUniform,
    Learned,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Counts {
    pub attempted: u64,
    pub hard_valid: u64,
    pub accepted: u64,
    pub hard_rejected: u64,
    pub proposal_nulls: u64,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunCounts {
    pub local: Counts,
    pub global: Counts,
    pub selected_body_updates: u64,
    pub selected_body_updates_by_body: Vec<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema: u32,
    pub config_sha256: String,
    #[serde(default)]
    pub shape_sha256: String,
    pub model_sha256: Option<String>,
    pub method: Method,
    pub completed_sweeps: u64,
    pub poses: Vec<Pose>,
    pub counts: RunCounts,
    pub master_seed: u64,
    pub rng_protocol: String,
}

impl Counts {
    fn since(&self, old: &Self) -> Self {
        Self {
            attempted: self.attempted - old.attempted,
            hard_valid: self.hard_valid - old.hard_valid,
            accepted: self.accepted - old.accepted,
            hard_rejected: self.hard_rejected - old.hard_rejected,
            proposal_nulls: self.proposal_nulls - old.proposal_nulls,
        }
    }
}
impl RunCounts {
    fn since(&self, old: &Self) -> Self {
        Self {
            local: self.local.since(&old.local),
            global: self.global.since(&old.global),
            selected_body_updates: self.selected_body_updates - old.selected_body_updates,
            selected_body_updates_by_body: self
                .selected_body_updates_by_body
                .iter()
                .zip(&old.selected_body_updates_by_body)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn hash_file(path: &Path) -> Result<String> {
    Ok(hash_bytes(&fs::read(path)?))
}
pub fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    {
        let mut file = File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}
pub fn cpu_seconds() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: t is a valid writable timespec; CLOCK_PROCESS_CPUTIME_ID has no
    // lifetime requirements and the return code is checked.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) };
    assert_eq!(status, 0, "process CPU clock failed");
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}
fn stream(seed: u64, sweep: u64, label: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"tetramer-mc-rng-v1");
    h.update(seed.to_le_bytes());
    h.update(sweep.to_le_bytes());
    h.update(label.as_bytes());
    StdRng::from_seed(h.finalize().into())
}
fn jsonline(stream: &mut impl Write, row: &Value) -> Result<()> {
    serde_json::to_writer(&mut *stream, row)?;
    stream.write_all(b"\n")?;
    Ok(())
}

pub struct RunOptions {
    pub config: PathBuf,
    pub model: Option<PathBuf>,
    pub method: Method,
    pub out: PathBuf,
    pub sweeps: u64,
    pub sample_every: u64,
    pub resume: Option<PathBuf>,
    pub write_gsd: bool,
    pub record_moves: bool,
}

pub fn run(options: RunOptions) -> Result<Value> {
    ensure!(
        options.sweeps > 0 && options.sample_every > 0,
        "positive sweep counts required"
    );
    let raw_config = fs::read(&options.config)?;
    let config_sha = hash_bytes(&raw_config);
    let mut config: Config = serde_json::from_slice(&raw_config)?;
    config.validate()?;
    let shape_path = if config.shape.is_absolute() {
        config.shape.clone()
    } else {
        options
            .config
            .parent()
            .unwrap_or(Path::new("."))
            .join(&config.shape)
    };
    let shape_raw =
        fs::read(&shape_path).with_context(|| format!("shape {}", shape_path.display()))?;
    let shape_sha = hash_bytes(&shape_raw);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    config.shape = fs::canonicalize(&shape_path)?;
    if let Some(path) = &config.monomer_shape {
        let path = if path.is_absolute() {
            path.clone()
        } else {
            options.config.parent().unwrap_or(Path::new(".")).join(path)
        };
        config.monomer_shape = Some(fs::canonicalize(path)?);
    }
    if let Some(path) = config
        .metadata
        .get("native_pair_motifs")
        .and_then(Value::as_str)
    {
        let path = PathBuf::from(path);
        let path = if path.is_absolute() {
            path
        } else {
            options.config.parent().unwrap_or(Path::new(".")).join(path)
        };
        config.metadata["native_pair_motifs"] = json!(fs::canonicalize(path)?);
    }
    let model_raw = options.model.as_ref().map(fs::read).transpose()?;
    let model_sha = model_raw.as_ref().map(|raw| hash_bytes(raw));
    let proposal = match options.method {
        Method::Learned => Some(FrozenRelativePoseProposal::from_json_str(
            std::str::from_utf8(
                model_raw
                    .as_ref()
                    .context("learned method requires --model")?,
            )?,
            config.box_lengths,
            config.learned_uniform_weight,
            &shape_sha,
        )?),
        Method::LocalUniform => {
            ensure!(model_raw.is_none(), "uniform control does not use a model");
            None
        }
    };
    let mut poses = config.initial_poses.clone();
    for p in &mut poses {
        p.position = wrap(p.position, config.box_lengths);
    }
    let mut counts = RunCounts {
        selected_body_updates_by_body: vec![0; poses.len()],
        ..Default::default()
    };
    let mut completed = 0;
    if let Some(path) = &options.resume {
        let mut checkpoint: Checkpoint = serde_json::from_slice(&fs::read(path)?)?;
        if checkpoint.shape_sha256.is_empty() {
            // Support the first pre-release checkpoints only when BOTH archived
            // provenance records independently identify the same physical shape.
            let parent = path.parent().context("checkpoint has no parent")?;
            let archived = hash_file(&parent.join("provenance/shape.json"))?;
            let manifest: Value = serde_json::from_slice(&fs::read(parent.join("manifest.json"))?)?;
            ensure!(
                manifest.get("shape_sha256").and_then(Value::as_str) == Some(archived.as_str())
                    && archived == shape_sha,
                "legacy checkpoint shape provenance mismatch"
            );
            checkpoint.shape_sha256 = archived;
        }
        ensure!(
            checkpoint.schema == 1 && checkpoint.rng_protocol == "sha256-master-sweep-stream-v1",
            "checkpoint protocol mismatch"
        );
        ensure!(
            checkpoint.config_sha256 == config_sha
                && checkpoint.shape_sha256 == shape_sha
                && checkpoint.model_sha256 == model_sha
                && checkpoint.method == options.method
                && checkpoint.master_seed == config.seed,
            "checkpoint input/method/seed mismatch"
        );
        ensure!(
            checkpoint.poses.len() == poses.len()
                && checkpoint.counts.selected_body_updates_by_body.len() == poses.len(),
            "checkpoint body count mismatch"
        );
        poses = checkpoint.poses;
        counts = checkpoint.counts;
        completed = checkpoint.completed_sweeps;
    }
    ensure!(
        completed < options.sweeps,
        "target sweeps must exceed checkpoint sweep"
    );
    for (i, &p) in poses.iter().enumerate() {
        p.validate()?;
        let env = Environment::new(
            &tree,
            &poses,
            i,
            p,
            p,
            config.box_lengths,
            config.depletant_radius,
        )?;
        ensure!(env.hard_valid(p), "hard overlap in initial body {i}");
    }
    if options.out.exists() {
        ensure!(
            fs::read_dir(&options.out)?.next().is_none(),
            "output directory must be empty (resume into a NEW directory)"
        );
    } else {
        fs::create_dir_all(&options.out)?;
    }
    fs::create_dir(options.out.join("provenance"))?;
    let source_bundle = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(
        options.out.join("provenance/source-bundle.json"),
        source_bundle,
    )?;
    fs::write(options.out.join("provenance/input-config.json"), raw_config)?;
    fs::write(options.out.join("provenance/shape.json"), shape_raw)?;
    if let Some(raw) = &model_raw {
        fs::write(
            options.out.join("provenance/frozen-relative-model.json"),
            raw,
        )?;
    }
    let mut effective = serde_json::to_value(&config)?;
    effective["method"] = json!(options.method);
    effective["sweeps"] = json!(options.sweeps);
    effective["sample_every"] = json!(options.sample_every);
    effective["initial_poses"] = json!(poses);
    save(&options.out.join("config.json"), &effective)?;
    let executable_sha = hash_file(&std::env::current_exe()?)?;
    save(
        &options.out.join("manifest.json"),
        &json!({"schema":1,"config_sha256":config_sha,"shape_sha256":shape_sha,"model_sha256":model_sha,"executable_sha256":executable_sha,"source_bundle_sha256":hash_bytes(source_bundle.as_bytes()),"version":env!("CARGO_PKG_VERSION"),"resume":options.resume,"initial_sweep":completed,"rng":"sha256-master-sweep-stream-v1; rand pinned by Cargo.lock","physical_target":"hard(X) exp[-z * exclusion_union_volume(X)]","scope":"Frozen learned proposal; algorithmic MC time, not physical kinetics"}),
    )?;
    let mut trajectory = BufWriter::new(File::create(options.out.join("trajectory.jsonl"))?);
    let mut moves = if options.record_moves {
        Some(BufWriter::new(File::create(
            options.out.join("moves.jsonl"),
        )?))
    } else {
        None
    };
    let mut gsd = if options.write_gsd {
        Some(Trajectory::create(&options.out.join("trajectory.gsd"))?)
    } else {
        None
    };
    let start = Instant::now();
    let start_cpu = cpu_seconds();
    let start_sweep = completed;
    let initial_counts = counts.clone();
    let mut gate_points = 0_u64;
    let mut gate_cpu = 0.;
    let mut geometry_cpu = 0.;
    let mut proposal_cpu = 0.;
    let snapshot = |sweep: u64,
                    poses: &[Pose],
                    counts: &RunCounts,
                    trajectory: &mut BufWriter<File>,
                    gsd: &mut Option<Trajectory>|
     -> Result<()> {
        jsonline(
            trajectory,
            &json!({"sweep":sweep,"poses":poses,"seed_labels":config.seed_labels,"boundary":"periodic","sampler_cpu_seconds":cpu_seconds()-start_cpu,"counts":counts}),
        )?;
        trajectory.flush()?;
        if let Some(writer) = gsd {
            writer.append(sweep, poses, config.box_lengths, &tree)?;
            writer.sync()?;
        }
        Ok(())
    };
    snapshot(completed, &poses, &counts, &mut trajectory, &mut gsd)?;
    for sweep in (completed + 1)..=options.sweeps {
        let mut schedule = stream(config.seed, sweep, "schedule");
        let mut choice = stream(config.seed, sweep, "choice");
        let mut local = stream(config.seed, sweep, "local");
        let mut global = stream(config.seed, sweep, "global");
        let mut gate_rng = stream(config.seed, sweep, "gate");
        let mut accept = stream(config.seed, sweep, "accept");
        let mut order: Vec<_> = (0..poses.len()).collect();
        order.shuffle(&mut schedule);
        for (update, &i) in order.iter().enumerate() {
            let old = poses[i];
            let is_global = choice.random::<f64>() < config.global_probability;
            let stats = if is_global {
                &mut counts.global
            } else {
                &mut counts.local
            };
            stats.attempted += 1;
            let before = cpu_seconds();
            let mut correction = 0.;
            let mut proposal_info = json!({"branch":if is_global{"uniform"}else{"local"},"anchor_index":null,"null":false,"log_reverse_forward":0.});
            let candidate = if !is_global {
                Some(local_pose(
                    &mut local,
                    old,
                    config.box_lengths,
                    config.local_translation_std_a,
                    config.local_small_angle_std_degrees.to_radians() / 2.,
                ))
            } else if let Some(proposal) = &proposal {
                let result = proposal.propose(&mut global, &poses, i)?;
                correction = result.log_reverse_forward.unwrap_or(0.);
                let pose = result.candidate;
                proposal_info = serde_json::to_value(result)?;
                pose
            } else {
                Some(uniform_pose(&mut global, config.box_lengths))
            };
            proposal_cpu += cpu_seconds() - before;
            let mut valid = false;
            let mut accepted = false;
            let mut sampled = None;
            let mut log_alpha = None;
            if let Some(new) = candidate {
                new.validate()?;
                ensure!(correction.is_finite(), "nonfinite proposal ratio");
                let before = cpu_seconds();
                let env = Environment::new(
                    &tree,
                    &poses,
                    i,
                    old,
                    new,
                    config.box_lengths,
                    config.depletant_radius,
                )?;
                ensure!(
                    env.hard_valid(old),
                    "current body invalid during sweep {sweep}"
                );
                valid = env.hard_valid(new);
                geometry_cpu += cpu_seconds() - before;
                if valid {
                    stats.hard_valid += 1;
                    let before = cpu_seconds();
                    let lambda = if config.reservoir_density > 0. {
                        config.poisson_lambda_ratio * config.reservoir_density
                    } else {
                        1.
                    };
                    let result = depletion::sample(
                        &mut gate_rng,
                        &env,
                        old,
                        new,
                        lambda,
                        config.reservoir_density,
                        config.endpoint_gate,
                    )?;
                    gate_cpu += cpu_seconds() - before;
                    gate_points += result.raw_points;
                    let alpha = (correction + result.log_weight).min(0.);
                    accepted = accept.random::<f64>().max(f64::MIN_POSITIVE).ln() < alpha;
                    if accepted {
                        poses[i] = new;
                        stats.accepted += 1;
                    }
                    sampled = Some(result);
                    log_alpha = Some(alpha);
                } else {
                    stats.hard_rejected += 1;
                }
            } else {
                stats.proposal_nulls += 1;
            }
            counts.selected_body_updates += 1;
            counts.selected_body_updates_by_body[i] += 1;
            if let Some(writer) = &mut moves {
                jsonline(
                    writer,
                    &json!({"sweep":sweep,"update_in_sweep":update,"moving_index":i,"kind":if is_global{"global"}else{"local"},"old_pose":old,"proposed_pose":candidate,"retained_pose":poses[i],"proposal":proposal_info,"hard_valid":valid,"accepted":accepted,"gate":sampled,"log_acceptance":log_alpha,"sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                )?;
            }
        }
        completed = sweep;
        if sweep % options.sample_every == 0 || sweep == options.sweeps {
            snapshot(sweep, &poses, &counts, &mut trajectory, &mut gsd)?;
            if let Some(m) = &mut moves {
                m.flush()?;
            }
            let checkpoint = Checkpoint {
                schema: 1,
                config_sha256: config_sha.clone(),
                shape_sha256: shape_sha.clone(),
                model_sha256: model_sha.clone(),
                method: options.method,
                completed_sweeps: completed,
                poses: poses.clone(),
                counts: counts.clone(),
                master_seed: config.seed,
                rng_protocol: "sha256-master-sweep-stream-v1".into(),
            };
            save(&options.out.join("checkpoint.json"), &checkpoint)?;
            save(
                &options.out.join("progress.json"),
                &json!({"complete":sweep==options.sweeps,"completed_sweeps":sweep,"requested_sweeps":options.sweeps,"counts":counts,"cpu_seconds":cpu_seconds()-start_cpu,"wall_seconds":start.elapsed().as_secs_f64()}),
            )?;
        }
    }
    let summary = json!({"complete":true,"completed_sweeps":completed,"initial_sweep":start_sweep,"requested_sweeps":options.sweeps,"method":options.method,"bodies":poses.len(),"all_bodies_mobile":true,"boundary":"periodic","counts":counts,"initial_counts":initial_counts,"segment_counts":counts.since(&initial_counts),"timing_scope":"CPU, wall and cost cover this invocation only; pair them with segment_counts","sampler_cpu_seconds":cpu_seconds()-start_cpu,"wall_seconds":start.elapsed().as_secs_f64(),"cost":{"proposal_cpu_seconds":proposal_cpu,"geometry_cpu_seconds":geometry_cpu,"gate_cpu_seconds":gate_cpu,"gate_raw_points":gate_points},"model_sha256":model_sha,"shape_sha256":shape_sha,"config_sha256":config_sha,"initial_metadata":config.metadata});
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}
