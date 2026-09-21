//! Sequential, all-mobile periodic or spherical Monte Carlo. Every sweep has independent
//! named RNG streams derived from (master seed, absolute sweep, stream label),
//! allowing exact checkpoint continuation without serializing opaque RNG state.
use crate::{
    atlas_mask::{AtlasMaskConfig, AtlasMaskEngine, AtlasMaskState},
    atlas_transport::{
        AtlasTransportConfig, AtlasTransportEngine, AtlasTransportFit, AtlasTransportState,
    },
    auxiliary::{self, AuxiliaryConfig},
    conditional::{ConditionalConfig, ConditionalEngine, ConditionalFit, ConditionalState},
    contact_memory::{MemoryConfig, MemoryState, ResolvedMemoryConfig},
    depletion::{self, GateOptions},
    docking::{DockingMethod, DockingProposal},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    rj::{JumpCounts, RjConfig, RjState},
    spherical::{self, Container, HalfTurn},
    trajectory::Trajectory,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, StandardNormal};
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
/// Origin-centered hard atomic wall; the ideal depletant bath remains unbounded.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Boundary {
    #[default]
    Periodic,
    Spherical {
        radius: f64,
    },
}
impl Boundary {
    pub fn radius(self) -> Option<f64> {
        match self {
            Self::Periodic => None,
            Self::Spherical { radius } => Some(radius),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Periodic => "periodic",
            Self::Spherical { .. } => "spherical",
        }
    }
}

/// Fixed mixture of the existing capture kernel and frozen posterior transport.
/// `probability` is conditional on an already selected global update slot.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FrozenPosteriorConfig {
    pub probability: f64,
    pub correlation: f64,
}
impl FrozenPosteriorConfig {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.probability.is_finite() && (0. ..1.).contains(&self.probability),
            "frozen posterior probability must be in [0, 1), retaining the capture kernel"
        );
        ensure!(
            self.correlation.is_finite() && (-1. ..=1.).contains(&self.correlation),
            "frozen posterior correlation must be in [-1, 1]"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub monomer_shape: Option<PathBuf>,
    pub shape: PathBuf,
    pub box_lengths: Vec3,
    #[serde(default)]
    pub boundary: Boundary,
    /// Independent Bernoulli attempts after each ordinary single-body sweep.
    #[serde(default)]
    pub gca_probability: f64,
    #[serde(default)]
    pub center_shift_probability: f64,
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
    pub auxiliary_transport: Option<AuxiliaryConfig>,
    #[serde(default)]
    pub reversible_jump: Option<RjConfig>,
    #[serde(default)]
    pub contact_memory: Option<MemoryConfig>,
    #[serde(default)]
    pub conditional_closure: Option<ConditionalConfig>,
    #[serde(default)]
    pub atlas_transport: Option<AtlasTransportConfig>,
    #[serde(default)]
    pub atlas_mask: Option<AtlasMaskConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen_posterior: Option<FrozenPosteriorConfig>,
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
            "invalid display/periodic lengths"
        );
        for probability in [self.gca_probability, self.center_shift_probability] {
            ensure!(
                probability.is_finite() && (0. ..=1.).contains(&probability),
                "invalid collective scheduling probability"
            );
        }
        match self.boundary {
            Boundary::Periodic => ensure!(
                self.gca_probability == 0. && self.center_shift_probability == 0.,
                "spherical GCA/center shift requires a spherical boundary"
            ),
            Boundary::Spherical { radius } => ensure!(
                radius.is_finite() && radius > 0.,
                "invalid spherical wall radius"
            ),
        }
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
        if let Some(options) = &self.auxiliary_transport {
            options.validate()?;
            ensure!(
                self.boundary.radius().is_some(),
                "auxiliary transport currently requires a spherical boundary"
            );
        }
        if let Some(options) = &self.reversible_jump {
            options.validate()?;
            ensure!(
                self.auxiliary_transport.is_some(),
                "reversible jump requires auxiliary transport"
            );
        }
        if let Some(options) = &self.contact_memory {
            options.validate()?;
            ensure!(
                self.boundary.radius().is_some(),
                "contact memory currently requires a spherical/open proposal"
            );
        }
        if let Some(options) = &self.conditional_closure {
            options.validate()?;
            ensure!(
                self.boundary.radius().is_some(),
                "conditional closure requires a spherical boundary"
            );
            ensure!(
                self.auxiliary_transport.is_none()
                    && self.reversible_jump.is_none()
                    && self.contact_memory.is_none(),
                "conditional closure is a separate model mode; disable auxiliary_transport, reversible_jump and contact_memory"
            );
        }
        if let Some(options) = &self.atlas_transport {
            options.validate()?;
            ensure!(
                self.boundary.radius().is_some(),
                "atlas transport requires a spherical boundary"
            );
            ensure!(
                self.auxiliary_transport.is_none()
                    && self.reversible_jump.is_none()
                    && self.contact_memory.is_none()
                    && self.conditional_closure.is_none(),
                "atlas transport is a separate fixed-atlas mode; disable other auxiliary model modes"
            );
        }
        if let Some(options) = &self.atlas_mask {
            options.validate()?;
            ensure!(
                self.atlas_transport.is_some(),
                "atlas_mask requires atlas_transport"
            );
        }
        if let Some(options) = self.frozen_posterior {
            options.validate()?;
            ensure!(
                self.boundary.radius().is_some(),
                "frozen posterior transport requires a spherical boundary"
            );
            ensure!(
                self.auxiliary_transport.is_none()
                    && self.reversible_jump.is_none()
                    && self.contact_memory.is_none()
                    && self.conditional_closure.is_none()
                    && self.atlas_transport.is_none()
                    && self.atlas_mask.is_none(),
                "frozen posterior transport requires immutable charts; disable auxiliary, RJ, contact-memory, conditional and atlas adaptations"
            );
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
    #[serde(default)]
    pub gca: CollectiveCounts,
    #[serde(default)]
    pub center_shift: CollectiveCounts,
    #[serde(default)]
    pub model_jumps: JumpCounts,
    #[serde(default)]
    pub contact_memory: Counts,
    #[serde(default)]
    pub conditional_refreshes: u64,
    #[serde(default)]
    pub conditional_gca_rejected: u64,
    #[serde(default)]
    pub conditional_shift_rejected: u64,
    #[serde(default)]
    pub atlas_refreshes: u64,
    #[serde(default)]
    pub atlas_mask_refreshes: u64,
    pub selected_body_updates: u64,
    pub selected_body_updates_by_body: Vec<u64>,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct CollectiveCounts {
    pub attempted: u64,
    pub completed: u64,
    /// Bodies selected for a transformation; may include a numerical identity.
    pub transformed_bodies: u64,
}
impl CollectiveCounts {
    fn since(&self, old: &Self) -> Self {
        Self {
            attempted: self.attempted - old.attempted,
            completed: self.completed - old.completed,
            transformed_bodies: self.transformed_bodies - old.transformed_bodies,
        }
    }
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
    #[serde(default)]
    pub auxiliary_eta: Option<Vec<[f64; 6]>>,
    #[serde(default)]
    pub rj_state: Option<RjState>,
    #[serde(default)]
    pub contact_memory_state: Option<MemoryState>,
    #[serde(default)]
    pub conditional_state: Option<ConditionalState>,
    #[serde(default)]
    pub atlas_state: Option<AtlasTransportState>,
    #[serde(default)]
    pub atlas_mask_state: Option<AtlasMaskState>,
    /// Display bookkeeping only: r_coordinate = r_sphere + C. A common
    /// sphere-frame translation d is equivalently a wall shift C -> C-d.
    #[serde(default)]
    pub coordinate_wall_center: Vec3,
    #[serde(default)]
    pub coordinate_origin_sweep: u64,
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
            gca: self.gca.since(&old.gca),
            center_shift: self.center_shift.since(&old.center_shift),
            model_jumps: self.model_jumps.since(&old.model_jumps),
            contact_memory: self.contact_memory.since(&old.contact_memory),
            conditional_refreshes: self.conditional_refreshes - old.conditional_refreshes,
            conditional_gca_rejected: self.conditional_gca_rejected - old.conditional_gca_rejected,
            conditional_shift_rejected: self.conditional_shift_rejected
                - old.conditional_shift_rejected,
            atlas_refreshes: self.atlas_refreshes - old.atlas_refreshes,
            atlas_mask_refreshes: self.atlas_mask_refreshes - old.atlas_mask_refreshes,
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

/// The open bath uses ordinary spectator poses. Its exclusion volume is NOT
/// clipped to the spherical wall: only the protein atoms see that wall.
fn environment<'a>(
    tree: &'a SphereTree,
    poses: &[Pose],
    moving: usize,
    old: Pose,
    new: Pose,
    config: &Config,
) -> Result<Environment<'a>> {
    if config.boundary == Boundary::Periodic {
        return Environment::new(
            tree,
            poses,
            moving,
            old,
            new,
            config.box_lengths,
            config.depletant_radius,
        );
    }
    ensure!(moving < poses.len(), "invalid moving index");
    old.validate()?;
    new.validate()?;
    let reach = 2. * (tree.bound + config.depletant_radius);
    let guard = 1024. * f64::EPSILON * (1. + reach + norm(old.position) + norm(new.position));
    let mut fixed = Vec::new();
    let mut labels = Vec::new();
    for (j, &pose) in poses.iter().enumerate() {
        pose.validate()?;
        if j != moving
            && (norm(sub(pose.position, old.position)) <= reach + guard
                || norm(sub(pose.position, new.position)) <= reach + guard)
        {
            fixed.push(Placed::new(pose));
            labels.push((j, [0; 3]));
        }
    }
    Ok(Environment {
        tree,
        fixed,
        labels,
        rd: config.depletant_radius,
    })
}

fn spherical_local_pose(rng: &mut StdRng, old: Pose, dt: f64, dc: f64) -> Pose {
    let displacement: Vec3 = std::array::from_fn(|_| {
        let z: f64 = StandardNormal.sample(rng);
        dt * z
    });
    let c: Vec3 = std::array::from_fn(|_| {
        let z: f64 = StandardNormal.sample(rng);
        dc * z
    });
    Pose {
        position: add(old.position, displacement),
        orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
    }
}

/// The anchor is a retained uniform label among all other bodies. Spectators
/// are fixed only for this attempt; the physical environment is built separately.
fn posterior_for_body(
    proposal: &DockingProposal,
    rng: &mut StdRng,
    poses: &[Pose],
    moving: usize,
) -> Result<(Option<Pose>, Value)> {
    ensure!(moving < poses.len(), "invalid posterior moving index");
    let spectators: Vec<_> = poses
        .iter()
        .enumerate()
        .filter_map(|(j, &pose)| (j != moving).then_some(pose))
        .collect();
    let (candidate, mut info) = proposal.propose(rng, poses[moving], &spectators)?;
    let local_anchor = info["anchor_index"]
        .as_u64()
        .context("posterior proposal did not retain its anchor label")?
        as usize;
    ensure!(
        local_anchor < spectators.len(),
        "invalid posterior anchor label"
    );
    let anchor = if local_anchor < moving {
        local_anchor
    } else {
        local_anchor + 1
    };
    info["anchor_index"] = json!(anchor);
    info["moving_index"] = json!(moving);
    info["kernel"] = json!("frozen-posterior");
    Ok((candidate, info))
}

fn uniform_direction(rng: &mut StdRng) -> Vec3 {
    let v: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
    let length = norm(v);
    assert!(
        length.is_finite() && length > 0.,
        "unrepresentable random direction"
    );
    scale(v, 1. / length)
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

fn memory_dictionary(
    base: &Option<FrozenRelativePoseProposal>,
    state: &Option<MemoryState>,
    config: &Option<ResolvedMemoryConfig>,
) -> Result<Option<FrozenRelativePoseProposal>> {
    match (base, state, config) {
        (Some(base), Some(state), Some(c)) => Ok(Some(base.with_contact_components(
            &state.poses,
            c.proposal_mass,
            c.proposal_translation_std_a,
            c.proposal_small_angle_std_degrees,
        )?)),
        (base, None, None) => Ok(base.clone()),
        _ => anyhow::bail!("inconsistent contact memory/model state"),
    }
}

fn conditional_diagnostic(fit: &ConditionalFit) -> Value {
    json!({"data_count":fit.data_count,"log_probabilities":fit.log_probabilities,
        "log_scores":fit.log_scores,"covariance_penalties":fit.covariance_penalties,
        "mean_log_densities":fit.mean_log_densities})
}

fn atlas_diagnostic(fit: &AtlasTransportFit) -> Value {
    json!({"data_count":fit.data_count,"assigned_count":fit.assigned_count,
        "component_counts":fit.component_counts,"coordinates":fit.coordinates})
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
    let declared: Value = serde_json::from_slice(&raw_config)?;
    if let Some(radius) = declared.get("spherical_radius").filter(|v| !v.is_null()) {
        ensure!(
            radius.as_f64() == config.boundary.radius() && config.boundary.radius().is_some(),
            "legacy spherical_radius requires matching boundary: {{kind: spherical, radius: R}}"
        );
    }
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
    let wall = config
        .boundary
        .radius()
        .map(|radius| Container::new(radius, &tree))
        .transpose()?;
    let uniform_lengths = match config.boundary {
        Boundary::Periodic => config.box_lengths,
        Boundary::Spherical { radius } => [2. * (radius + tree.bound); 3],
    };
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
    let conditional_engine = config
        .conditional_closure
        .as_ref()
        .map(|c| {
            ConditionalEngine::new(
                tree.bound,
                config.boundary.radius().unwrap(),
                &shape_sha,
                c.clone(),
            )
        })
        .transpose()?;
    ensure!(
        conditional_engine.is_none() || (options.method == Method::Learned && model_raw.is_none()),
        "conditional closure uses --method learned without --model; it fits current geometry"
    );
    let proposal = match options.method {
        Method::Learned if conditional_engine.is_some() => None,
        Method::Learned => {
            let text = std::str::from_utf8(
                model_raw
                    .as_ref()
                    .context("learned method requires --model")?,
            )?;
            Some(if wall.is_some() {
                FrozenRelativePoseProposal::from_json_str_open(
                    text,
                    uniform_lengths,
                    config.learned_uniform_weight,
                    &shape_sha,
                )?
            } else {
                FrozenRelativePoseProposal::from_json_str(
                    text,
                    uniform_lengths,
                    config.learned_uniform_weight,
                    &shape_sha,
                )?
            })
        }
        Method::LocalUniform => {
            ensure!(model_raw.is_none(), "uniform control does not use a model");
            None
        }
    };
    let posterior_proposal = config
        .frozen_posterior
        .map(|settings| {
            DockingProposal::new(
                proposal
                    .as_ref()
                    .context("frozen posterior transport requires --method learned and --model")?
                    .clone(),
                DockingMethod::PosteriorInvolution,
                settings.correlation,
                [0.; 3],
            )
        })
        .transpose()?;
    let atlas_engine = config
        .atlas_transport
        .as_ref()
        .map(|settings| {
            AtlasTransportEngine::new(
                proposal
                    .as_ref()
                    .context("atlas transport requires --method learned and --model")?
                    .clone(),
                settings.clone(),
            )
        })
        .transpose()?;
    let atlas_mask_engine = config
        .atlas_mask
        .as_ref()
        .map(|settings| {
            AtlasMaskEngine::new(
                &proposal
                    .as_ref()
                    .context("atlas mask requires a model")?
                    .component_weights(),
                settings.clone(),
            )
        })
        .transpose()?;
    ensure!(
        config.auxiliary_transport.is_none() || proposal.is_some(),
        "auxiliary transport requires --method learned and a model"
    );
    ensure!(
        config.contact_memory.is_none() || proposal.is_some(),
        "contact memory requires --method learned and a model"
    );
    let memory_initialization_start = cpu_seconds();
    let resolved_contact_memory = config
        .contact_memory
        .as_ref()
        .map(|m| m.resolve(&tree, config.depletant_radius, config.reservoir_density))
        .transpose()?;
    let mut contact_memory_state = resolved_contact_memory
        .as_ref()
        .map(|m| {
            MemoryState::initialize(&tree, m, &mut stream(config.seed, 0, "contact-memory-init"))
        })
        .transpose()?;
    let memory_initialization_cpu = cpu_seconds() - memory_initialization_start;
    let mut dictionary =
        memory_dictionary(&proposal, &contact_memory_state, &resolved_contact_memory)?;
    let mut auxiliary_eta = config
        .auxiliary_transport
        .as_ref()
        .filter(|_| config.reversible_jump.is_none())
        .map(|_| {
            auxiliary::draw_eta(
                dictionary.as_ref().unwrap(),
                &mut stream(config.seed, 0, "auxiliary"),
            )
        });
    let label_weights = dictionary.as_ref().map(|p| p.component_weights());
    let mut rj_state = config
        .reversible_jump
        .as_ref()
        .map(|r| {
            RjState::new(
                r,
                label_weights.as_ref().unwrap(),
                &mut stream(config.seed, 0, "rj-init"),
            )
        })
        .transpose()?;
    let mut poses = config.initial_poses.clone();
    if wall.is_none() {
        for p in &mut poses {
            p.position = wrap(p.position, config.box_lengths);
        }
    }
    let conditional_initialization_start = cpu_seconds();
    let mut conditional_fit = conditional_engine
        .as_ref()
        .map(|e| e.fit(&poses))
        .transpose()?;
    let mut conditional_state = conditional_engine
        .as_ref()
        .map(|e| {
            e.initialize(
                conditional_fit.as_ref().unwrap(),
                &mut stream(config.seed, 0, "conditional-init"),
            )
        })
        .transpose()?;
    let mut conditional_initialization_cpu = cpu_seconds() - conditional_initialization_start;
    let atlas_initialization_start = cpu_seconds();
    let mut atlas_fit = atlas_engine.as_ref().map(|e| e.fit(&poses)).transpose()?;
    let mut atlas_state = atlas_engine
        .as_ref()
        .map(|e| {
            e.initialize(
                atlas_fit.as_ref().unwrap(),
                &mut stream(config.seed, 0, "atlas-init"),
            )
        })
        .transpose()?;
    let mut atlas_initialization_cpu = cpu_seconds() - atlas_initialization_start;
    let mut atlas_mask_state = atlas_mask_engine
        .as_ref()
        .map(|engine| engine.initialize(&mut stream(config.seed, 0, "atlas-mask-init")))
        .transpose()?;
    let mut counts = RunCounts {
        selected_body_updates_by_body: vec![0; poses.len()],
        ..Default::default()
    };
    let mut completed = 0;
    let mut coordinate_wall_center = [0.; 3];
    let mut coordinate_origin_sweep = 0;
    if let Some(path) = &options.resume {
        let checkpoint_json: Value = serde_json::from_slice(&fs::read(path)?)?;
        let mut checkpoint: Checkpoint = serde_json::from_value(checkpoint_json.clone())?;
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
        ensure!(
            checkpoint.auxiliary_eta.is_some()
                == (config.auxiliary_transport.is_some() && config.reversible_jump.is_none())
                && checkpoint.rj_state.is_some() == config.reversible_jump.is_some()
                && checkpoint.contact_memory_state.is_some() == config.contact_memory.is_some()
                && checkpoint.conditional_state.is_some() == config.conditional_closure.is_some(),
            "checkpoint auxiliary mode mismatch"
        );
        ensure!(
            checkpoint.atlas_state.is_some() == config.atlas_transport.is_some(),
            "checkpoint atlas mode mismatch"
        );
        ensure!(
            checkpoint.atlas_mask_state.is_some() == config.atlas_mask.is_some(),
            "checkpoint atlas mask mode mismatch"
        );
        if let Some(state) = &checkpoint.atlas_mask_state {
            atlas_mask_engine.as_ref().unwrap().validate_state(state)?;
        }
        atlas_mask_state = checkpoint.atlas_mask_state;
        if let Some(eta) = &checkpoint.auxiliary_eta {
            ensure!(
                eta.len() == dictionary.as_ref().unwrap().component_count()
                    && eta.iter().flatten().all(|x| x.is_finite()),
                "checkpoint auxiliary residual mismatch"
            );
        }
        auxiliary_eta = checkpoint.auxiliary_eta;
        if let Some(state) = &checkpoint.rj_state {
            state.validate(
                config.reversible_jump.as_ref().unwrap(),
                dictionary.as_ref().unwrap().component_count(),
            )?;
        }
        rj_state = checkpoint.rj_state;
        if let Some(state) = &checkpoint.contact_memory_state {
            state.validate(&tree, resolved_contact_memory.as_ref().unwrap())?;
        }
        contact_memory_state = checkpoint.contact_memory_state;
        dictionary = memory_dictionary(&proposal, &contact_memory_state, &resolved_contact_memory)?;
        poses = checkpoint.poses;
        if let Some(state) = &checkpoint.conditional_state {
            state.validate(config.conditional_closure.as_ref().unwrap())?;
        }
        conditional_state = checkpoint.conditional_state;
        let before = cpu_seconds();
        conditional_fit = conditional_engine
            .as_ref()
            .map(|e| e.fit(&poses))
            .transpose()?;
        conditional_initialization_cpu += cpu_seconds() - before;
        if let Some(state) = &checkpoint.atlas_state {
            state.validate(proposal.as_ref().unwrap().component_count())?;
        }
        atlas_state = checkpoint.atlas_state;
        let before = cpu_seconds();
        atlas_fit = atlas_engine.as_ref().map(|e| e.fit(&poses)).transpose()?;
        atlas_initialization_cpu += cpu_seconds() - before;
        counts = checkpoint.counts;
        completed = checkpoint.completed_sweeps;
        ensure!(
            checkpoint
                .coordinate_wall_center
                .iter()
                .all(|x| x.is_finite()),
            "invalid checkpoint coordinate-frame center"
        );
        coordinate_wall_center = checkpoint.coordinate_wall_center;
        coordinate_origin_sweep = if checkpoint_json.get("coordinate_wall_center").is_some() {
            checkpoint.coordinate_origin_sweep
        } else {
            // Legacy checkpoints did not preserve the display gauge. Start a
            // declared new display origin; never invent missing prior shifts.
            completed
        };
        ensure!(
            coordinate_origin_sweep <= completed,
            "coordinate origin is after checkpoint"
        );
    }
    ensure!(
        completed < options.sweeps,
        "target sweeps must exceed checkpoint sweep"
    );
    for (i, &p) in poses.iter().enumerate() {
        p.validate()?;
        let env = environment(&tree, &poses, i, p, p, &config)?;
        ensure!(
            wall.as_ref().is_none_or(|w| w.contains(p)),
            "initial body {i} violates spherical wall"
        );
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
    effective["uniform_proposal_cube_lengths"] = json!(uniform_lengths);
    effective["coordinate_origin_sweep"] = json!(coordinate_origin_sweep);
    effective["resolved_contact_memory"] = json!(resolved_contact_memory);
    effective["coordinate_frame_convention"] = json!(
        "Stored poses use sphere-centered coordinates; coordinate positions = stored positions + coordinate_wall_center; center shifts subtract their common displacement from the wall center. Origin established at coordinate_origin_sweep."
    );
    save(&options.out.join("config.json"), &effective)?;
    let executable_sha = hash_file(&std::env::current_exe()?)?;
    let mut manifest = json!({"schema":1,"config_sha256":config_sha,"shape_sha256":shape_sha,"model_sha256":model_sha,"executable_sha256":executable_sha,"source_bundle_sha256":hash_bytes(source_bundle.as_bytes()),"version":env!("CARGO_PKG_VERSION"),"resume":options.resume,"initial_sweep":completed,"rng":"sha256-master-sweep-stream-v1; rand pinned by Cargo.lock","physical_target":"hard(X) wall(X) exp[-z * exclusion_union_volume(X)]","boundary":config.boundary,"bath_wall_permeable":wall.is_some(),"collective_schedule":"after each single-body sweep: independent state-independent Bernoulli GCA, then center shift; dedicated RNG streams","auxiliary_transport":config.auxiliary_transport,"reversible_jump":config.reversible_jump,"contact_memory":resolved_contact_memory,"conditional_closure":config.conditional_closure,"atlas_transport":config.atlas_transport,"atlas_mask":config.atlas_mask,"scope":"Frozen atlas/contact-memory modes or normalized current-geometry conditional full-GMM closure; explicit auxiliary state and corrections; no unrecorded training history; algorithmic MC time, not physical kinetics"});
    if let Some(settings) = config.frozen_posterior {
        manifest["frozen_posterior"] = json!(settings);
    }
    save(&options.out.join("manifest.json"), &manifest)?;
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
    let mut gca_cpu = 0.;
    let mut shift_cpu = 0.;
    let mut memory_cpu = 0.;
    let mut memory_gate_points = 0_u64;
    let mut conditional_fit_cpu = 0.;
    let mut conditional_fit_calls = 0_u64;
    let mut atlas_fit_cpu = 0.;
    let mut atlas_fit_calls = 0_u64;
    let snapshot = |sweep: u64,
                    poses: &[Pose],
                    counts: &RunCounts,
                    auxiliary_eta: &Option<Vec<[f64; 6]>>,
                    rj_state: &Option<RjState>,
                    contact_memory_state: &Option<MemoryState>,
                    conditional_state: &Option<ConditionalState>,
                    conditional_fit: &Option<ConditionalFit>,
                    atlas_state: &Option<AtlasTransportState>,
                    atlas_fit: &Option<AtlasTransportFit>,
                    atlas_mask_state: &Option<AtlasMaskState>,
                    coordinate_wall_center: Vec3,
                    trajectory: &mut BufWriter<File>,
                    gsd: &mut Option<Trajectory>|
     -> Result<()> {
        jsonline(
            trajectory,
            &json!({"sweep":sweep,"poses":poses,"seed_labels":config.seed_labels,"boundary":config.boundary.name(),"spherical_wall_radius":config.boundary.radius(),"coordinate_wall_center":coordinate_wall_center,"coordinate_origin_sweep":coordinate_origin_sweep,"sampler_cpu_seconds":cpu_seconds()-start_cpu,"counts":counts,"auxiliary_eta":auxiliary_eta,"rj_state":rj_state,"contact_memory_state":contact_memory_state,"conditional_state":conditional_state,"conditional_fit":conditional_fit.as_ref().map(conditional_diagnostic),"atlas_state":atlas_state,"atlas_fit":atlas_fit.as_ref().map(atlas_diagnostic),"atlas_mask_state":atlas_mask_state}),
        )?;
        trajectory.flush()?;
        if let Some(writer) = gsd {
            let display_lengths = config
                .boundary
                .radius()
                .map_or(config.box_lengths, |r| [2. * r; 3]);
            writer.append_with_coordinate_frame(
                sweep,
                poses,
                display_lengths,
                &tree,
                config.boundary.radius(),
                coordinate_wall_center,
                coordinate_origin_sweep,
            )?;
            writer.sync()?;
        }
        Ok(())
    };
    snapshot(
        completed,
        &poses,
        &counts,
        &auxiliary_eta,
        &rj_state,
        &contact_memory_state,
        &conditional_state,
        &conditional_fit,
        &atlas_state,
        &atlas_fit,
        &atlas_mask_state,
        coordinate_wall_center,
        &mut trajectory,
        &mut gsd,
    )?;
    for sweep in (completed + 1)..=options.sweeps {
        if let Some(state) = &mut contact_memory_state {
            let before = cpu_seconds();
            let m = resolved_contact_memory.as_ref().unwrap();
            let mut rng = stream(config.seed, sweep, "contact-memory");
            for attempt in 0..m.attempts_per_sweep {
                let result = state.update(&tree, m, config.endpoint_gate, &mut rng)?;
                counts.contact_memory.attempted += 1;
                counts.contact_memory.hard_valid += u64::from(result.hard_valid);
                counts.contact_memory.accepted += u64::from(result.accepted);
                counts.contact_memory.hard_rejected += u64::from(!result.hard_valid);
                if let Some(gate) = &result.gate {
                    memory_gate_points += gate.raw_points;
                }
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"contact_memory","attempt":attempt,"result":result,"sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
            dictionary =
                memory_dictionary(&proposal, &contact_memory_state, &resolved_contact_memory)?;
            // Fixed slot identities and fixed mixture mass are essential to
            // retaining the existing independent RJ label prior.
            ensure!(
                dictionary.as_ref().unwrap().component_weights()
                    == *label_weights.as_ref().unwrap(),
                "contact memory changed the RJ label prior"
            );
            memory_cpu += cpu_seconds() - before;
        }
        if let Some(state) = &mut rj_state {
            state.refresh_eta(&mut stream(config.seed, sweep, "auxiliary"));
            let r = config.reversible_jump.as_ref().unwrap();
            let mut jump_rng = stream(config.seed, sweep, "model-jumps");
            for attempt in 0..r.attempts_per_sweep {
                let result = state.update(r, label_weights.as_ref().unwrap(), &mut jump_rng)?;
                counts.model_jumps.record(&result);
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"model_jump","attempt":attempt,"result":result,"sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
        } else if config.auxiliary_transport.is_some() {
            auxiliary_eta = Some(auxiliary::draw_eta(
                dictionary.as_ref().unwrap(),
                &mut stream(config.seed, sweep, "auxiliary"),
            ));
        }
        let mut schedule = stream(config.seed, sweep, "schedule");
        let mut choice = stream(config.seed, sweep, "choice");
        let mut local = stream(config.seed, sweep, "local");
        let mut global = stream(config.seed, sweep, "global");
        // Independent streams leave all preexisting streams untouched when the
        // option is absent (or its probability is zero).
        let mut posterior_choice = config
            .frozen_posterior
            .map(|_| stream(config.seed, sweep, "posterior-choice"));
        let mut posterior_rng = config
            .frozen_posterior
            .map(|_| stream(config.seed, sweep, "posterior-proposal"));
        let mut gate_rng = stream(config.seed, sweep, "gate");
        let mut accept = stream(config.seed, sweep, "accept");
        let mut order: Vec<_> = (0..poses.len()).collect();
        order.shuffle(&mut schedule);
        for (update, &i) in order.iter().enumerate() {
            let old = poses[i];
            let is_global = choice.random::<f64>() < config.global_probability;
            let is_posterior = is_global
                && config.frozen_posterior.is_some_and(|settings| {
                    settings.probability > 0.
                        && posterior_choice.as_mut().unwrap().random::<f64>() < settings.probability
                });
            let stats = if is_global {
                &mut counts.global
            } else {
                &mut counts.local
            };
            stats.attempted += 1;
            let before = cpu_seconds();
            let mut correction = 0.;
            let mut conditional_forward: Option<(usize, f64)> = None;
            let mut atlas_forward: Option<(usize, f64)> = None;
            let mut proposal_info = json!({"branch":if is_global{"uniform"}else{"local"},"anchor_index":null,"null":false,"log_reverse_forward":0.});
            let candidate = if !is_global {
                Some(if wall.is_some() {
                    spherical_local_pose(
                        &mut local,
                        old,
                        config.local_translation_std_a,
                        config.local_small_angle_std_degrees.to_radians() / 2.,
                    )
                } else {
                    local_pose(
                        &mut local,
                        old,
                        config.box_lengths,
                        config.local_translation_std_a,
                        config.local_small_angle_std_degrees.to_radians() / 2.,
                    )
                })
            } else if is_posterior {
                let (candidate, info) = posterior_for_body(
                    posterior_proposal.as_ref().unwrap(),
                    posterior_rng.as_mut().unwrap(),
                    &poses,
                    i,
                )?;
                if candidate.is_some() {
                    correction = info["log_reverse_forward"]
                        .as_f64()
                        .context("posterior proposal correction missing")?;
                }
                proposal_info = info;
                proposal_info["correlation"] = json!(config.frozen_posterior.unwrap().correlation);
                candidate
            } else if let Some(engine) = &conditional_engine {
                let forward = engine.model(
                    conditional_fit.as_ref().unwrap(),
                    conditional_state.as_ref().unwrap(),
                )?;
                let result = forward.propose(&mut global, &poses, i)?;
                let candidate = result.candidate;
                if candidate.is_some() {
                    conditional_forward = Some((
                        result.anchor_index,
                        result
                            .new_log_density
                            .context("conditional forward density missing")?,
                    ));
                }
                proposal_info = serde_json::to_value(result)?;
                proposal_info["conditional_closure"] = json!(true);
                proposal_info["conditional_k"] = json!(conditional_state.as_ref().unwrap().k);
                candidate
            } else if let Some(engine) = &atlas_engine {
                let forward =
                    engine.model(atlas_fit.as_ref().unwrap(), atlas_state.as_ref().unwrap())?;
                let forward = match &atlas_mask_state {
                    Some(mask) => forward.weighted_subset(&mask.labels)?,
                    None => forward,
                };
                let result = forward.propose(&mut global, &poses, i)?;
                let candidate = result.candidate;
                if candidate.is_some() {
                    atlas_forward = Some((
                        result.anchor_index,
                        result
                            .new_log_density
                            .context("atlas forward density missing")?,
                    ));
                }
                proposal_info = serde_json::to_value(result)?;
                proposal_info["atlas_transport"] = json!(true);
                if let Some(mask) = &atlas_mask_state {
                    proposal_info["atlas_mask_labels"] = json!(mask.labels);
                    proposal_info["atlas_component_label"] = proposal_info["component_index"]
                        .as_u64()
                        .map(|i| json!(mask.labels[i as usize]))
                        .unwrap_or(Value::Null);
                }
                candidate
            } else if let Some(base) = &dictionary {
                let selected = rj_state
                    .as_ref()
                    .map(|s| base.selected_components(&s.labels))
                    .transpose()?;
                let base = selected.as_ref().unwrap_or(base);
                let eta = rj_state.as_ref().map(|s| &s.eta).or(auxiliary_eta.as_ref());
                let forward_model = config
                    .auxiliary_transport
                    .as_ref()
                    .map(|a| auxiliary::model(base, &poses, eta.unwrap(), a))
                    .transpose()?;
                let forward = forward_model.as_ref().unwrap_or(base);
                let mut result = forward.propose(&mut global, &poses, i)?;
                correction = result.log_reverse_forward.unwrap_or(0.);
                let candidate = result.candidate;
                let mut reverse_log = None;
                if let (Some(new), Some(a)) = (candidate, config.auxiliary_transport.as_ref()) {
                    // The model is transported by preserving eta. Use its
                    // reconstruction at Y, never the stale forward model.
                    let mut next = poses.clone();
                    next[i] = new;
                    let reverse = auxiliary::model(base, &next, eta.unwrap(), a)?;
                    let value = reverse.log_density(&poses[i], &poses[result.anchor_index])?;
                    correction =
                        value - result.new_log_density.context("missing forward density")?;
                    reverse_log = Some(value);
                    result.log_reverse_forward = Some(correction);
                }
                proposal_info = serde_json::to_value(result)?;
                if contact_memory_state.is_some() {
                    let component = proposal_info["component_index"]
                        .as_u64()
                        .map(|c| c as usize);
                    let label = component.map(|c| rj_state.as_ref().map_or(c, |r| r.labels[c]));
                    let base_count = proposal.as_ref().unwrap().component_count();
                    proposal_info["dictionary_label"] = json!(label);
                    proposal_info["contact_memory_slot"] =
                        json!(label.and_then(|c| c.checked_sub(base_count)));
                }
                if config.auxiliary_transport.is_some() {
                    proposal_info["transported_reverse_log_density"] = json!(reverse_log);
                    proposal_info["auxiliary_transport"] = json!(true);
                }
                candidate
            } else {
                let mut pose = uniform_pose(&mut global, uniform_lengths);
                if wall.is_some() {
                    pose.position = sub(pose.position, scale(uniform_lengths, 0.5));
                }
                Some(pose)
            };
            if config.frozen_posterior.is_some() && !is_posterior {
                proposal_info["kernel"] = json!(if is_global {
                    "full-mixture-capture"
                } else {
                    "local"
                });
            }
            proposal_cpu += cpu_seconds() - before;
            let mut valid = false;
            let mut accepted = false;
            let mut sampled = None;
            let mut log_alpha = None;
            if let Some(new) = candidate {
                new.validate()?;
                ensure!(correction.is_finite(), "nonfinite proposal ratio");
                let before = cpu_seconds();
                let env = environment(&tree, &poses, i, old, new, &config)?;
                ensure!(
                    env.hard_valid(old) && wall.as_ref().is_none_or(|w| w.contains(old)),
                    "current body invalid during sweep {sweep}"
                );
                valid = wall.as_ref().is_none_or(|w| w.contains(new)) && env.hard_valid(new);
                geometry_cpu += cpu_seconds() - before;
                if valid {
                    stats.hard_valid += 1;
                    let mut next_conditional_fit = None;
                    let mut next_atlas_fit = None;
                    if let Some(engine) = &conditional_engine {
                        let before = cpu_seconds();
                        let mut next = poses.clone();
                        next[i] = new;
                        let fit = engine.fit(&next)?;
                        conditional_fit_calls += 1;
                        let state = conditional_state.as_ref().unwrap();
                        let count_ratio = fit.log_probabilities[state.k]
                            - conditional_fit.as_ref().unwrap().log_probabilities[state.k];
                        let mut q_ratio = 0.;
                        if let Some((anchor, forward_log)) = conditional_forward {
                            let reverse = engine.model(&fit, state)?;
                            let reverse_log = reverse.log_density(&old, &poses[anchor])?;
                            q_ratio = reverse_log - forward_log;
                            proposal_info["transported_reverse_log_density"] = json!(reverse_log);
                        }
                        correction = count_ratio + q_ratio;
                        ensure!(
                            correction.is_finite(),
                            "nonfinite conditional acceptance correction"
                        );
                        proposal_info["conditional_closure"] = json!(true);
                        proposal_info["conditional_k"] = json!(state.k);
                        proposal_info["conditional_count_log_ratio"] = json!(count_ratio);
                        proposal_info["log_reverse_forward"] = json!(q_ratio);
                        proposal_info["conditional_log_correction"] = json!(correction);
                        conditional_fit_cpu += cpu_seconds() - before;
                        next_conditional_fit = Some(fit);
                    }
                    if let Some(engine) = &atlas_engine {
                        let before = cpu_seconds();
                        let mut next = poses.clone();
                        next[i] = new;
                        let fit = engine.fit(&next)?;
                        atlas_fit_calls += 1;
                        if let Some((anchor, forward_log)) = atlas_forward {
                            let reverse = engine.model(&fit, atlas_state.as_ref().unwrap())?;
                            let reverse = match &atlas_mask_state {
                                Some(mask) => reverse.weighted_subset(&mask.labels)?,
                                None => reverse,
                            };
                            let reverse_log = reverse.log_density(&old, &poses[anchor])?;
                            correction = reverse_log - forward_log;
                            proposal_info["transported_reverse_log_density"] = json!(reverse_log);
                            proposal_info["log_reverse_forward"] = json!(correction);
                        }
                        ensure!(
                            correction.is_finite(),
                            "nonfinite atlas proposal correction"
                        );
                        proposal_info["atlas_transport"] = json!(true);
                        atlas_fit_cpu += cpu_seconds() - before;
                        next_atlas_fit = Some(fit);
                    }
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
                        if let Some(fit) = next_conditional_fit {
                            conditional_fit = Some(fit);
                        }
                        if let Some(fit) = next_atlas_fit {
                            atlas_fit = Some(fit);
                        }
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
        if let Some(wall) = &wall {
            // Separate named streams preserve all preexisting periodic draws.
            let mut gca_rng = stream(config.seed, sweep, "spherical-gca");
            if config.gca_probability > 0. && gca_rng.random::<f64>() < config.gca_probability {
                counts.gca.attempted += 1;
                let before = cpu_seconds();
                let previous = conditional_engine.as_ref().map(|_| poses.clone());
                let axis = uniform_direction(&mut gca_rng);
                let result = spherical::update(
                    &tree,
                    wall,
                    &mut poses,
                    HalfTurn::new(axis)?,
                    config.depletant_radius,
                    config.reservoir_density,
                    &mut gca_rng,
                )?;
                gca_cpu += cpu_seconds() - before;
                let mut count_ratio = 0.;
                let mut accepted = true;
                if let Some(engine) = &conditional_engine {
                    let before = cpu_seconds();
                    let fit = engine.fit(&poses)?;
                    conditional_fit_calls += 1;
                    let k = conditional_state.as_ref().unwrap().k;
                    count_ratio = fit.log_probabilities[k]
                        - conditional_fit.as_ref().unwrap().log_probabilities[k];
                    let mut rng = stream(config.seed, sweep, "conditional-gca-accept");
                    accepted =
                        rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < count_ratio.min(0.);
                    if accepted {
                        conditional_fit = Some(fit);
                    } else {
                        poses = previous.unwrap();
                        counts.conditional_gca_rejected += 1;
                    }
                    conditional_fit_cpu += cpu_seconds() - before;
                }
                counts.gca.completed += 1;
                if let Some(engine) = &atlas_engine {
                    let before = cpu_seconds();
                    atlas_fit = Some(engine.fit(&poses)?);
                    atlas_fit_calls += 1;
                    atlas_fit_cpu += cpu_seconds() - before;
                }
                if accepted {
                    counts.gca.transformed_bodies += result.flipped_indices.len() as u64;
                }
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"gca","axis":axis,"result":result,"accepted":accepted,"conditional_count_log_ratio":count_ratio,"sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
            let mut shift_rng = stream(config.seed, sweep, "spherical-center-shift");
            if config.center_shift_probability > 0.
                && shift_rng.random::<f64>() < config.center_shift_probability
            {
                counts.center_shift.attempted += 1;
                let before = cpu_seconds();
                let previous = conditional_engine.as_ref().map(|_| poses.clone());
                let direction = uniform_direction(&mut shift_rng);
                // Strictly interior 52-bit midpoint, representable even at the
                // upper endpoint (53-bit midpoint addition could round to one).
                let open_u = ((shift_rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
                let result = wall.center_shift(&mut poses, direction, open_u)?;
                shift_cpu += cpu_seconds() - before;
                let mut count_ratio = 0.;
                let mut accepted = true;
                if let Some(engine) = &conditional_engine {
                    let before = cpu_seconds();
                    let fit = engine.fit(&poses)?;
                    conditional_fit_calls += 1;
                    let k = conditional_state.as_ref().unwrap().k;
                    count_ratio = fit.log_probabilities[k]
                        - conditional_fit.as_ref().unwrap().log_probabilities[k];
                    let mut rng = stream(config.seed, sweep, "conditional-shift-accept");
                    accepted =
                        rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < count_ratio.min(0.);
                    if accepted {
                        conditional_fit = Some(fit);
                    } else {
                        poses = previous.unwrap();
                        counts.conditional_shift_rejected += 1;
                    }
                    conditional_fit_cpu += cpu_seconds() - before;
                }
                if accepted {
                    coordinate_wall_center = sub(coordinate_wall_center, result.displacement);
                }
                ensure!(
                    coordinate_wall_center.iter().all(|x| x.is_finite()),
                    "unrepresentable coordinate-frame center"
                );
                counts.center_shift.completed += 1;
                if let Some(engine) = &atlas_engine {
                    let before = cpu_seconds();
                    atlas_fit = Some(engine.fit(&poses)?);
                    atlas_fit_calls += 1;
                    atlas_fit_cpu += cpu_seconds() - before;
                }
                if accepted {
                    counts.center_shift.transformed_bodies += poses.len() as u64;
                }
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"center_shift","direction":direction,"result":result,"accepted":accepted,"conditional_count_log_ratio":count_ratio,"sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
        }
        // Refresh after physical moves so an explicit initial model is actually
        // used. This is a Gibbs draw from p(K|D(X)) times the Gaussian latent law.
        if let (Some(engine), Some(state)) = (&conditional_engine, &mut conditional_state) {
            let mut rng = stream(config.seed, sweep, "conditional-refresh");
            if rng.random::<f64>()
                < config
                    .conditional_closure
                    .as_ref()
                    .unwrap()
                    .refresh_probability
            {
                let previous = state.clone();
                engine.refresh(conditional_fit.as_ref().unwrap(), state, &mut rng)?;
                counts.conditional_refreshes += 1;
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"conditional_refresh",
                        "old_state":previous,"state":state,"fit":conditional_diagnostic(conditional_fit.as_ref().unwrap()),
                        "sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
        }
        if let (Some(engine), Some(state)) = (&atlas_engine, &mut atlas_state) {
            let mut rng = stream(config.seed, sweep, "atlas-refresh");
            if rng.random::<f64>() < config.atlas_transport.as_ref().unwrap().refresh_probability {
                let previous = state.clone();
                engine.refresh(state, &mut rng)?;
                counts.atlas_refreshes += 1;
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"atlas_refresh",
                        "old_state":previous,"state":state,"fit":atlas_diagnostic(atlas_fit.as_ref().unwrap()),
                        "sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
        }
        if let (Some(engine), Some(state)) = (&atlas_mask_engine, &mut atlas_mask_state) {
            let mut rng = stream(config.seed, sweep, "atlas-mask-refresh");
            if rng.random::<f64>() < config.atlas_mask.as_ref().unwrap().refresh_probability {
                let previous = state.clone();
                engine.refresh(state, &mut rng)?;
                counts.atlas_mask_refreshes += 1;
                if let Some(writer) = &mut moves {
                    jsonline(
                        writer,
                        &json!({"sweep":sweep,"kind":"atlas_mask_refresh",
                        "old_state":previous,"state":state,"log_probability":engine.log_probability(state)?,
                        "sampler_cpu_seconds":cpu_seconds()-start_cpu}),
                    )?;
                }
            }
        }
        completed = sweep;
        if sweep % options.sample_every == 0 || sweep == options.sweeps {
            snapshot(
                sweep,
                &poses,
                &counts,
                &auxiliary_eta,
                &rj_state,
                &contact_memory_state,
                &conditional_state,
                &conditional_fit,
                &atlas_state,
                &atlas_fit,
                &atlas_mask_state,
                coordinate_wall_center,
                &mut trajectory,
                &mut gsd,
            )?;
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
                auxiliary_eta: auxiliary_eta.clone(),
                rj_state: rj_state.clone(),
                contact_memory_state: contact_memory_state.clone(),
                conditional_state: conditional_state.clone(),
                atlas_state: atlas_state.clone(),
                atlas_mask_state: atlas_mask_state.clone(),
                coordinate_wall_center,
                coordinate_origin_sweep,
            };
            save(&options.out.join("checkpoint.json"), &checkpoint)?;
            save(
                &options.out.join("progress.json"),
                &json!({"complete":sweep==options.sweeps,"completed_sweeps":sweep,"requested_sweeps":options.sweeps,"counts":counts,"cpu_seconds":cpu_seconds()-start_cpu,"wall_seconds":start.elapsed().as_secs_f64()}),
            )?;
        }
    }
    let mut summary = json!({"complete":true,"completed_sweeps":completed,"initial_sweep":start_sweep,"requested_sweeps":options.sweeps,"method":options.method,"bodies":poses.len(),"all_bodies_mobile":true,"boundary":config.boundary.name(),"spherical_wall_radius":config.boundary.radius(),"bath_wall_permeable":wall.is_some(),"counts":counts,"initial_counts":initial_counts,"segment_counts":counts.since(&initial_counts),"timing_scope":"CPU, wall and cost cover this invocation only; pair them with segment_counts","sampler_cpu_seconds":cpu_seconds()-start_cpu,"wall_seconds":start.elapsed().as_secs_f64(),"cost":{"proposal_cpu_seconds":proposal_cpu,"geometry_cpu_seconds":geometry_cpu,"gate_cpu_seconds":gate_cpu,"gate_raw_points":gate_points,"gca_cpu_seconds":gca_cpu,"center_shift_cpu_seconds":shift_cpu,"contact_memory_cpu_seconds":memory_cpu,"contact_memory_gate_raw_points":memory_gate_points,"contact_memory_initialization_cpu_seconds":memory_initialization_cpu,"conditional_fit_cpu_seconds":conditional_fit_cpu,"conditional_fit_calls":conditional_fit_calls,"conditional_initialization_cpu_seconds":conditional_initialization_cpu,"atlas_fit_cpu_seconds":atlas_fit_cpu,"atlas_fit_calls":atlas_fit_calls,"atlas_initialization_cpu_seconds":atlas_initialization_cpu},"model_sha256":model_sha,"shape_sha256":shape_sha,"config_sha256":config_sha,"initial_metadata":config.metadata,"auxiliary_transport":config.auxiliary_transport,"reversible_jump":config.reversible_jump,"contact_memory":resolved_contact_memory,"conditional_closure":config.conditional_closure,"atlas_transport":config.atlas_transport,"atlas_mask":config.atlas_mask});
    if let Some(settings) = config.frozen_posterior {
        summary["frozen_posterior"] = json!(settings);
    }
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod frozen_posterior_tests {
    use super::*;

    fn config() -> Value {
        json!({
            "shape":"unused.json", "box_lengths":[12.,12.,12.],
            "boundary":{"kind":"spherical","radius":6.}, "seed":19,
            "depletant_radius":0.5, "reservoir_density":0.1,
            "initial_poses":[
                {"position":[-2.,0.,0.],"orientation":[1.,0.,0.,0.]},
                {"position":[2.,0.,0.],"orientation":[1.,0.,0.,0.]}
            ]
        })
    }

    #[test]
    fn frozen_posterior_config_requires_a_capture_branch_and_immutable_open_charts() -> Result<()> {
        for probability in [-0.1, 1., 1.1, f64::INFINITY, f64::NAN] {
            assert!(
                FrozenPosteriorConfig {
                    probability,
                    correlation: 0.9
                }
                .validate()
                .is_err()
            );
        }
        for correlation in [-1.1, 1.1, f64::INFINITY, f64::NAN] {
            assert!(
                FrozenPosteriorConfig {
                    probability: 0.5,
                    correlation
                }
                .validate()
                .is_err()
            );
        }
        for probability in [0., 0.5, 0.999] {
            for correlation in [-1., 0., 0.9, 1.] {
                FrozenPosteriorConfig {
                    probability,
                    correlation,
                }
                .validate()?;
            }
        }
        let absent: Config = serde_json::from_value(config())?;
        absent.validate()?;
        assert!(
            serde_json::to_value(&absent)?
                .get("frozen_posterior")
                .is_none()
        );
        let mut enabled = config();
        enabled["frozen_posterior"] = json!({"probability":0.5,"correlation":0.9});
        serde_json::from_value::<Config>(enabled.clone())?.validate()?;
        let mut periodic = enabled.clone();
        periodic["boundary"] = json!({"kind":"periodic"});
        assert!(
            serde_json::from_value::<Config>(periodic)?
                .validate()
                .is_err()
        );
        for mode in [
            "auxiliary_transport",
            "reversible_jump",
            "contact_memory",
            "conditional_closure",
            "atlas_transport",
            "atlas_mask",
        ] {
            let mut changed = enabled.clone();
            changed[mode] = json!({});
            assert!(
                serde_json::from_value::<Config>(changed)?
                    .validate()
                    .is_err(),
                "accepted {mode}"
            );
        }
        Ok(())
    }
}
