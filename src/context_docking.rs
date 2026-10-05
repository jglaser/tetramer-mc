//! One mobile body in an authenticated, immutable spherical context.
//! The atom wall is the physical domain; the defensive cube is a proposal only.
//! All elementary attempts, including self-loops and fatal partial clouds, are
//! journaled. No accepted-prefix retry or physical-target restriction is hidden.
use crate::{
    bounded_singleton_path::{self, Budget, Limits},
    depletion::{GateOptions, GateResult},
    docking::{self, DockingConfig, DockingMethod, DockingProposal},
    geometry::{Placed, Shape, SphereTree},
    math::*,
    proposal::{FrozenRelativePoseProposal, RelativePoseBranch},
    rigid_subset::RigidSubset,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const PROTOCOL: &str = "fixed-context-docking-rng-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedInputs {
    pub model: String,
    pub shape: String,
    pub fixed_context: String,
    pub source_state: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextDockingConfig {
    pub shape: PathBuf,
    pub fixed_context: PathBuf,
    pub source_state: PathBuf,
    pub expected_sha256: ExpectedInputs,
    pub expected_fixed_body_count: usize,
    pub initial_pose: Option<Pose>,
    pub depletant_radius: f64,
    pub reservoir_density: f64,
    pub poisson_lambda_ratio: f64,
    pub translation_steps: Vec<f64>,
    pub rotation_steps_deg: Vec<f64>,
    pub rotation_probability: f64,
    pub local_attempts_per_cycle: usize,
    pub uniform_probability: f64,
    pub seed: u64,
    pub endpoint_gate: GateOptions,
    pub limits: Limits,
    pub maximum_attempts: u64,
    pub wall_seconds: f64,
    /// Immutable run/arm/initialization/stream identifiers; not acceptance data.
    pub identity: Value,
    pub warmup_cycles: u64,
}

#[derive(Clone, Debug)]
pub struct ContextDockingOptions {
    pub config: PathBuf,
    pub model: PathBuf,
    pub prior: Option<PathBuf>,
    pub out: PathBuf,
    pub cycles: u64,
    pub sample_every: u64,
    pub method: DockingMethod,
    pub correlation: f64,
    pub resume: Option<PathBuf>,
    pub certify: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBody {
    pub label: usize,
    pub pose: Pose,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedContext {
    schema: String,
    anchor_label: usize,
    excluded_moving_labels: Vec<usize>,
    bodies: Vec<ContextBody>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceState {
    schema: String,
    moving_label: usize,
    pose: Pose,
    anchor_label: usize,
    anchor_pose: Pose,
    boundary: String,
    spherical_wall_radius: f64,
    coordinate_frame: String,
    endpoint_sha256: String,
}

#[derive(Deserialize)]
struct PriorBranch {
    virtual_label: usize,
    component_index: usize,
    inverted: bool,
    original_probability: f64,
    original_log_probability: f64,
    eligible: bool,
    log_probability: f64,
    probability: f64,
}

#[derive(Deserialize)]
struct PriorAsset {
    schema: String,
    complete: bool,
    passed: bool,
    method: String,
    moving_labels: Vec<usize>,
    anchor_label: usize,
    floor_probability: f64,
    virtual_branch_count: usize,
    branches: Vec<PriorBranch>,
    log_prior: Vec<f64>,
    base_log_prior: Vec<f64>,
    eligibility: Vec<bool>,
    input_sha256: BTreeMap<String, String>,
    fixed_context_sha256: String,
    source_state_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CertificationCandidate {
    Source {
        id: String,
    },
    VirtualCenter {
        id: String,
        virtual_branch: usize,
        original_probability: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub pose: Pose,
    pub wall_valid: bool,
    pub core_overlap_labels: Vec<usize>,
    pub exclusion_contact_labels: Vec<usize>,
}
impl Observation {
    pub fn physical_valid(&self) -> bool {
        self.wall_valid && self.core_overlap_labels.is_empty()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Counts {
    pub attempted: u64,
    pub completed: u64,
    pub accepted: u64,
    pub nulls: u64,
    pub wall_rejected: u64,
    pub core_rejected: u64,
    pub bath_rejected: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    protocol: String,
    bindings: BTreeMap<String, String>,
    method: DockingMethod,
    correlation: f64,
    sample_every: u64,
    completed_cycles: u64,
    completed_attempts: u64,
    pose: Pose,
    observation: Observation,
    counts: Counts,
    raw_points: u64,
    retained_points: u64,
    charged_cpu_seconds: f64,
    charged_wall_seconds: f64,
}

struct Prepared {
    cfg: ContextDockingConfig,
    context: FixedContext,
    source: SourceState,
    bindings: BTreeMap<String, String>,
    tree: SphereTree,
    contact_tree: SphereTree,
    wall: Container,
    fixed: Vec<Placed>,
    fixed_poses: Vec<Pose>,
    proposer: DockingProposal,
    branches: Vec<RelativePoseBranch>,
    anchor_index: usize,
    local: DockingConfig,
    cube_half_width: f64,
}

struct Journal {
    file: BufWriter<File>,
    next: u64,
}
impl Journal {
    fn emit(&mut self, mut value: Value) -> Result<()> {
        value["event_index"] = json!(self.next);
        serde_json::to_writer(&mut self.file, &value)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.next += 1;
        Ok(())
    }
}

fn resolve(base: &Path, value: &Path) -> PathBuf {
    if value.is_absolute() {
        value.to_path_buf()
    } else {
        base.join(value)
    }
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-12 * (1. + b.abs())
}
fn digest_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Any contained atomic sphere gives |origin| <= R + |body-frame center|.
/// Hence this immutable cube contains every physical origin, even for an
/// off-center body. The added outward guard is proposal support, not wall repair.
pub fn uniform_half_width(radius: f64, bound: f64) -> Result<f64> {
    ensure!(
        radius.is_finite() && radius > 0. && bound.is_finite() && bound > 0.,
        "Invalid uniform support inputs"
    );
    let half = radius + bound + 512. * f64::EPSILON * (1. + radius + bound);
    ensure!((2. * half).is_finite(), "Uniform support overflow");
    Ok(half)
}

fn stream(seed: u64, cycle: u64, slot: usize, role: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(PROTOCOL.as_bytes());
    h.update(seed.to_le_bytes());
    h.update(cycle.to_le_bytes());
    h.update((slot as u64).to_le_bytes());
    h.update(role.as_bytes());
    StdRng::from_seed(h.finalize().into())
}

/// Separate role streams are supplied explicitly so the physical reference and
/// the runner exercise the same elementary kernel without hidden RNG draws.
pub struct StepRngs {
    pub proposal: StdRng,
    pub gate: StdRng,
    pub accept: StdRng,
}

#[derive(Debug, Serialize)]
pub struct PhysicalStep {
    pub old_pose: Pose,
    pub proposed_pose: Option<Pose>,
    pub retained_pose: Pose,
    pub proposal: Value,
    pub accepted: bool,
    pub status: String,
    pub wall_valid: Option<bool>,
    pub core_valid: Option<bool>,
    pub gate: Option<GateResult>,
    pub log_proposal_reverse_forward: Option<f64>,
    pub raw_log_acceptance: Option<f64>,
    pub log_acceptance: Option<f64>,
    pub acceptance_uniform: Option<f64>,
    pub log_uniform: Option<f64>,
    pub proposal_cpu_seconds: f64,
    pub geometry_cpu_seconds: f64,
    pub gate_cpu_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct PhysicalStepFailure {
    pub reason: String,
    pub phase: String,
    pub partial_trace: PhysicalStep,
    pub partial_gate: Option<bounded_singleton_path::LegFailure>,
}
impl std::fmt::Display for PhysicalStepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "physical step failed in {}: {}", self.phase, self.reason)
    }
}
impl std::error::Error for PhysicalStepFailure {}

/// Full-wall singleton kernel shared with the cached physical reference.
/// The fixed context and proposal prior must remain frozen through this call.
/// All numerical proposal failures are fatal, never manufactured rejections.
#[allow(clippy::too_many_arguments)]
pub fn physical_step(
    tree: &SphereTree,
    wall: &Container,
    fixed: &[Pose],
    old: Pose,
    local: &DockingConfig,
    global_proposal: Option<&DockingProposal>,
    rngs: &mut StepRngs,
    budget: &mut Budget,
) -> std::result::Result<PhysicalStep, PhysicalStepFailure> {
    let mut trace = PhysicalStep {
        old_pose: old,
        proposed_pose: None,
        retained_pose: old,
        proposal: Value::Null,
        accepted: false,
        status: "begun".into(),
        wall_valid: None,
        core_valid: None,
        gate: None,
        log_proposal_reverse_forward: None,
        raw_log_acceptance: None,
        log_acceptance: None,
        acceptance_uniform: None,
        log_uniform: None,
        proposal_cpu_seconds: 0.,
        geometry_cpu_seconds: 0.,
        gate_cpu_seconds: 0.,
    };
    let mut phase = "source_validation";
    let mut partial_gate = None;
    let result = (|| -> Result<()> {
        old.validate()?;
        budget.check_cpu()?;
        ensure!(wall.contains(old), "Old pose is outside atomic wall");
        let old_placed = Placed::new(old);
        for (i, p) in fixed.iter().enumerate() {
            if i % 32 == 0 {
                budget.check_cpu()?;
            }
            p.validate()?;
            ensure!(
                !tree.overlaps(&old_placed, &Placed::new(*p)),
                "Old pose overlaps spectator {i}"
            );
        }
        if let Some(proposer) = global_proposal {
            ensure!(
                proposer.method() == DockingMethod::PosteriorInvolution && !proposer.is_periodic(),
                "Physical singleton requires open posterior proposal"
            );
            ensure!(
                proposer.member_uniform_contains(old)?,
                "Uniform branch lacks reverse support for physical old pose"
            );
        }
        phase = "proposal";
        let time = cpu_seconds();
        let (candidate, info) = if let Some(proposer) = global_proposal {
            proposer.propose(&mut rngs.proposal, old, fixed)?
        } else {
            (
                Some(docking::local(&mut rngs.proposal, old, local)),
                json!({"branch":"local","log_reverse_forward":0.}),
            )
        };
        trace.proposal_cpu_seconds = cpu_seconds() - time;
        trace.proposed_pose = candidate;
        trace.proposal = info;
        let new =
            candidate.with_context(|| format!("Uncertified proposal null: {}", trace.proposal))?;
        new.validate()?;
        let correction = trace.proposal["log_reverse_forward"]
            .as_f64()
            .context("Missing finite proposal correction")?;
        ensure!(correction.is_finite(), "Nonfinite proposal correction");
        trace.log_proposal_reverse_forward = Some(correction);
        if trace.proposal["branch"] == "uniform" {
            ensure!(
                global_proposal
                    .context("Uniform label without proposal")?
                    .member_uniform_contains(new)?,
                "Uniform destination outside its immutable support"
            );
        }
        budget.check_cpu()?;
        phase = "endpoint_geometry";
        let time = cpu_seconds();
        let wall_valid = wall.contains(new);
        trace.wall_valid = Some(wall_valid);
        if !wall_valid {
            trace.status = "wall_rejected".into();
            trace.geometry_cpu_seconds = cpu_seconds() - time;
            return Ok(());
        }
        let placed = Placed::new(new);
        let mut valid = true;
        for (i, p) in fixed.iter().enumerate() {
            if i % 32 == 0 {
                budget.check_cpu()?;
            }
            if tree.overlaps(&placed, &Placed::new(*p)) {
                valid = false;
                break;
            }
        }
        trace.core_valid = Some(valid);
        trace.geometry_cpu_seconds = cpu_seconds() - time;
        if !valid {
            trace.status = "core_rejected".into();
            return Ok(());
        }
        phase = "bath";
        budget.check_cpu()?;
        let time = cpu_seconds();
        let mut state = fixed.to_vec();
        let moving = state.len();
        state.push(old);
        let gate = RigidSubset::new(tree, &state, &[moving], moving, new, local.depletant_radius)?;
        let lambda = if local.reservoir_density > 0. {
            local.poisson_lambda_ratio * local.reservoir_density
        } else {
            1.
        };
        let sampled = bounded_singleton_path::bounded_singleton(
            &gate,
            &mut rngs.gate,
            lambda,
            local.reservoir_density,
            local.endpoint_gate,
            budget,
        );
        trace.gate_cpu_seconds = cpu_seconds() - time;
        let cloud = match sampled {
            Ok(value) => value,
            Err(error) => {
                let reason = error.to_string();
                partial_gate = Some(error);
                anyhow::bail!(reason);
            }
        };
        let raw = correction + cloud.log_weight;
        trace.gate = Some(cloud);
        trace.raw_log_acceptance = Some(raw);
        ensure!(raw.is_finite(), "Nonfinite physical acceptance");
        budget.check_cpu()?;
        phase = "acceptance";
        let clipped = raw.min(0.);
        let uniform = rngs.accept.random::<f64>();
        trace.log_acceptance = Some(clipped);
        trace.acceptance_uniform = Some(uniform);
        // A finite RNG can return zero. Save log(0) as null, with U=0 exact;
        // never serialize an infinity as if it were a finite likelihood.
        trace.log_uniform = (uniform > 0.).then(|| uniform.ln());
        trace.accepted = uniform.ln() < clipped;
        if trace.accepted {
            trace.retained_pose = new;
            trace.status = "accepted".into();
        } else {
            trace.status = "bath_rejected".into();
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(trace),
        Err(error) => Err(PhysicalStepFailure {
            reason: format!("{error:#}"),
            phase: phase.into(),
            partial_trace: trace,
            partial_gate,
        }),
    }
}

fn prepare(options: &ContextDockingOptions) -> Result<Prepared> {
    ensure!(
        matches!(
            options.method,
            DockingMethod::Local | DockingMethod::PosteriorInvolution
        ),
        "Only local and posterior-involution methods supported"
    );
    ensure!(
        options.method != DockingMethod::Local || options.prior.is_none(),
        "Local method cannot consume an unused prior"
    );
    ensure!(
        options.correlation.is_finite() && (-1. ..=1.).contains(&options.correlation),
        "Invalid correlation"
    );
    ensure!(
        options.sample_every > 0 && options.cycles > 0,
        "Positive run lengths required"
    );
    let config_raw = fs::read(&options.config)?;
    let cfg: ContextDockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.limits.validate()?;
    ensure!(
        cfg.wall_seconds.is_finite() && cfg.wall_seconds > 0. && cfg.maximum_attempts > 0,
        "Invalid invocation limits"
    );
    ensure!(
        cfg.local_attempts_per_cycle < usize::MAX,
        "Slot count overflow"
    );
    ensure!(cfg.identity.is_object(), "Run identity must be an object");
    let base = options.config.parent().unwrap_or(Path::new("."));
    let inputs = [
        (
            "model",
            options.model.clone(),
            cfg.expected_sha256.model.clone(),
        ),
        (
            "shape",
            resolve(base, &cfg.shape),
            cfg.expected_sha256.shape.clone(),
        ),
        (
            "fixed_context",
            resolve(base, &cfg.fixed_context),
            cfg.expected_sha256.fixed_context.clone(),
        ),
        (
            "source_state",
            resolve(base, &cfg.source_state),
            cfg.expected_sha256.source_state.clone(),
        ),
    ];
    let mut bindings = BTreeMap::from([("config".into(), hash_bytes(&config_raw))]);
    let mut raw = BTreeMap::new();
    for (name, path, expected) in &inputs {
        ensure!(digest_valid(expected), "Invalid expected digest: {name}");
        let bytes = fs::read(path).with_context(|| format!("Read {name}"))?;
        ensure!(
            hash_bytes(&bytes) == *expected,
            "Bound {name} bytes changed"
        );
        bindings.insert((*name).into(), expected.clone());
        raw.insert(*name, bytes);
    }
    let context: FixedContext = serde_json::from_slice(&raw["fixed_context"])?;
    let source: SourceState = serde_json::from_slice(&raw["source_state"])?;
    ensure!(
        context.schema == "fixed-outside-context-v1"
            && source.schema == "saved-source-state-metadata-v1",
        "Unknown context/source schema"
    );
    ensure!(
        source.boundary == "spherical"
            && source.coordinate_frame
                == "Saved spherical-center frame; no display offset, wrapping or pose transform.",
        "Only authenticated sphere-center coordinates are supported"
    );
    ensure!(
        context.bodies.len() == cfg.expected_fixed_body_count && !context.bodies.is_empty(),
        "Fixed body inventory mismatch"
    );
    ensure!(
        context.excluded_moving_labels == vec![source.moving_label]
            && context.anchor_label == source.anchor_label,
        "Moving/anchor labels mismatch"
    );
    let labels: BTreeSet<_> = context.bodies.iter().map(|b| b.label).collect();
    ensure!(
        labels.len() == context.bodies.len() && !labels.contains(&source.moving_label),
        "Duplicate or moving spectator label"
    );
    ensure!(
        context.bodies.windows(2).all(|p| p[0].label < p[1].label),
        "Fixed labels must be in canonical increasing order"
    );
    source.pose.validate()?;
    source.anchor_pose.validate()?;
    for body in &context.bodies {
        body.pose.validate()?;
    }
    if let Some(pose) = cfg.initial_pose {
        pose.validate()?;
    }
    let anchor_index = context
        .bodies
        .iter()
        .position(|b| b.label == source.anchor_label)
        .context("Missing fixed anchor")?;
    ensure!(
        context.bodies[anchor_index].pose == source.anchor_pose,
        "Anchor pose differs from source authority"
    );
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&raw["shape"])?)?;
    let wall = Container::new(source.spherical_wall_radius, &tree)?;
    let cube_half_width = uniform_half_width(wall.radius, tree.bound)?;
    let fixed_poses: Vec<_> = context.bodies.iter().map(|b| b.pose).collect();
    let local = DockingConfig {
        shape: resolve(base, &cfg.shape),
        fixed_poses: fixed_poses.clone(),
        initial_pose: cfg.initial_pose.unwrap_or(source.pose),
        capture_center: [0.; 3],
        capture_radius: cube_half_width,
        depletant_radius: cfg.depletant_radius,
        reservoir_density: cfg.reservoir_density,
        poisson_lambda_ratio: cfg.poisson_lambda_ratio,
        translation_steps: cfg.translation_steps.clone(),
        rotation_steps_deg: cfg.rotation_steps_deg.clone(),
        rotation_probability: cfg.rotation_probability,
        local_attempts_per_cycle: cfg.local_attempts_per_cycle,
        uniform_probability: cfg.uniform_probability,
        seed: cfg.seed,
        endpoint_gate: cfg.endpoint_gate,
        metadata: Value::Null,
        proposal_anchor_index: Some(anchor_index),
        target_region: None,
    };
    local.validate()?; // Its capture fields are used only to construct proposal support, never physical acceptance.
    let lambda = if cfg.reservoir_density > 0. {
        cfg.reservoir_density * cfg.poisson_lambda_ratio
    } else {
        1.
    };
    ensure!(
        lambda.is_finite() && lambda > 0. && (lambda + cfg.reservoir_density).is_finite(),
        "Invalid bath intensity"
    );
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&raw["model"])?,
        [2. * cube_half_width; 3],
        cfg.uniform_probability,
        &bindings["shape"],
    )?;
    let branches = model.virtual_branches();
    let mut proposer = DockingProposal::new(model, options.method, options.correlation, [0.; 3])?
        .with_anchor_index(Some(anchor_index));
    if let Some(path) = &options.prior {
        let bytes = fs::read(path)?;
        let asset: PriorAsset = serde_json::from_slice(&bytes)?;
        validate_prior(&asset, &branches, &bindings, &context, &source)?;
        proposer = proposer.with_virtual_branch_log_prior(&asset.log_prior)?;
        bindings.insert("prior".into(), hash_bytes(&bytes));
        raw.insert("prior", bytes);
    }
    bindings.insert(
        "ordered_virtual_branches".into(),
        hash_bytes(&serde_json::to_vec(&branches)?),
    );
    bindings.insert(
        "effective_log_prior".into(),
        hash_bytes(&serde_json::to_vec(proposer.virtual_branch_log_prior())?),
    );
    let mut contact_shape = tree.shape.clone();
    for atom in &mut contact_shape.atoms {
        atom.radius += cfg.depletant_radius;
    }
    let contact_tree = SphereTree::new(contact_shape)?;
    let fixed = fixed_poses.iter().copied().map(Placed::new).collect();
    let provenance = options.out.join("provenance");
    fs::create_dir(&provenance)?;
    for (name, bytes) in raw {
        fs::write(provenance.join(format!("{name}.json")), bytes)?;
    }
    fs::write(provenance.join("config.json"), config_raw)?;
    let bundle = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(provenance.join("source-bundle.json"), bundle)?;
    bindings.insert("source_bundle".into(), hash_bytes(bundle.as_bytes()));
    bindings.insert("executable".into(), hash_file(&std::env::current_exe()?)?);
    Ok(Prepared {
        cfg,
        context,
        source,
        bindings,
        tree,
        contact_tree,
        wall,
        fixed,
        fixed_poses,
        proposer,
        branches,
        anchor_index,
        local,
        cube_half_width,
    })
}

fn validate_prior(
    asset: &PriorAsset,
    branches: &[RelativePoseBranch],
    bindings: &BTreeMap<String, String>,
    context: &FixedContext,
    source: &SourceState,
) -> Result<()> {
    ensure!(
        asset.schema == "fixed-context-virtual-branch-prior-v1"
            && asset.complete
            && asset.passed
            && asset.method == "posterior_involution",
        "Prior asset incomplete or wrong method"
    );
    ensure!(
        asset.moving_labels == context.excluded_moving_labels
            && asset.anchor_label == context.anchor_label,
        "Prior labels mismatch"
    );
    ensure!(
        asset.fixed_context_sha256 == bindings["fixed_context"]
            && asset.source_state_sha256 == bindings["source_state"],
        "Prior context/source bytes mismatch"
    );
    for key in ["model", "shape"] {
        ensure!(
            asset.input_sha256.get(key) == bindings.get(key),
            "Prior {key} mismatch"
        );
    }
    ensure!(
        asset.input_sha256.get("endpoint") == Some(&source.endpoint_sha256),
        "Prior physical endpoint mismatch"
    );
    let n = branches.len();
    ensure!(
        asset.virtual_branch_count == n
            && asset.branches.len() == n
            && asset.log_prior.len() == n
            && asset.base_log_prior.len() == n
            && asset.eligibility.len() == n,
        "Prior branch inventory mismatch"
    );
    for (i, (actual, saved)) in branches.iter().zip(&asset.branches).enumerate() {
        ensure!(
            saved.virtual_label == i
                && saved.component_index == actual.component_index
                && saved.inverted == actual.inverted,
            "Prior branch identity mismatch at {i}"
        );
        ensure!(
            close(saved.original_probability, actual.weight)
                && close(saved.original_log_probability, actual.weight.ln())
                && close(asset.base_log_prior[i], actual.weight.ln()),
            "Prior original weight mismatch at {i}"
        );
        ensure!(
            saved.eligible == asset.eligibility[i]
                && close(saved.log_probability, asset.log_prior[i])
                && close(saved.probability, asset.log_prior[i].exp()),
            "Inconsistent prior record at {i}"
        );
    }
    let reconstructed = docking::defensive_virtual_branch_log_prior(
        &asset.base_log_prior,
        &asset.eligibility,
        asset.floor_probability,
    )?;
    ensure!(
        reconstructed
            .iter()
            .zip(&asset.log_prior)
            .all(|(a, b)| close(*a, *b)),
        "Prior construction does not match its saved rule"
    );
    Ok(())
}

impl Prepared {
    fn observe(&self, pose: Pose, budget: &Budget) -> Result<Observation> {
        pose.validate()?;
        budget.check_cpu()?;
        let moving = Placed::new(pose);
        let mut core = Vec::new();
        let mut contact = Vec::new();
        for (i, fixed) in self.fixed.iter().enumerate() {
            if i % 32 == 0 {
                budget.check_cpu()?;
            }
            if self.tree.overlaps(&moving, fixed) {
                core.push(self.context.bodies[i].label);
            }
            if self.contact_tree.overlaps(&moving, fixed) {
                contact.push(self.context.bodies[i].label);
            }
        }
        Ok(Observation {
            pose,
            wall_valid: self.wall.contains(pose),
            core_overlap_labels: core,
            exclusion_contact_labels: contact,
        })
    }
    fn audit_fixed(&self, budget: &Budget) -> Result<Value> {
        let mut pairs = Vec::new();
        let mut wall = Vec::new();
        for i in 0..self.fixed.len() {
            budget.check_cpu()?;
            if !self.wall.contains(self.fixed_poses[i]) {
                wall.push(self.context.bodies[i].label);
            }
            for j in 0..i {
                if j % 64 == 0 {
                    budget.check_cpu()?;
                }
                if self.tree.overlaps(&self.fixed[i], &self.fixed[j]) {
                    pairs.push([self.context.bodies[j].label, self.context.bodies[i].label]);
                }
            }
        }
        Ok(
            json!({"fixed_fixed_core_overlaps":pairs,"fixed_wall_invalid_labels":wall,"physical_context_valid":pairs.is_empty() && wall.is_empty()}),
        )
    }
    fn candidate(&self, spec: &CertificationCandidate) -> Result<(String, Pose, Value)> {
        match spec {
            CertificationCandidate::Source { id } => {
                Ok((id.clone(), self.source.pose, json!({"kind":"source"})))
            }
            CertificationCandidate::VirtualCenter {
                id,
                virtual_branch,
                original_probability,
            } => {
                let branch = self
                    .branches
                    .get(*virtual_branch)
                    .context("Invalid candidate virtual label")?;
                ensure!(
                    close(*original_probability, branch.weight),
                    "Candidate original probability mismatch"
                );
                let mut relative = self
                    .proposer
                    .member_chart_parts()
                    .0
                    .decode(*virtual_branch, [0.; 6])?;
                if branch.inverted {
                    relative = invert_relative_pose(relative);
                }
                let anchor = self.fixed_poses[self.anchor_index];
                let pose = Pose {
                    position: anchor.apply(relative.position),
                    orientation: quaternion(matmul(
                        rotation(anchor.orientation),
                        rotation(relative.orientation),
                    )),
                };
                pose.validate()?;
                Ok((
                    id.clone(),
                    pose,
                    json!({"kind":"virtual_center","virtual_branch":virtual_branch,"original_probability":original_probability,"component_index":branch.component_index,"inverted":branch.inverted}),
                ))
            }
        }
    }
}

fn checkpoint_matches(
    saved: &Checkpoint,
    p: &Prepared,
    options: &ContextDockingOptions,
) -> Result<()> {
    ensure!(
        saved.protocol == PROTOCOL
            && saved.bindings == p.bindings
            && saved.method == options.method
            && saved.correlation == options.correlation
            && saved.sample_every == options.sample_every,
        "Checkpoint inputs/proposal/schedule mismatch"
    );
    let slots = (p.cfg.local_attempts_per_cycle as u64)
        .checked_add(1)
        .context("Slot overflow")?;
    ensure!(
        saved.completed_cycles.checked_mul(slots) == Some(saved.completed_attempts)
            && saved.counts.completed == saved.completed_attempts
            && saved.counts.attempted == saved.completed_attempts,
        "Checkpoint is not a complete cycle prefix"
    );
    ensure!(
        saved.pose == saved.observation.pose && saved.observation.physical_valid(),
        "Invalid checkpoint state"
    );
    let classified = [
        saved.counts.accepted,
        saved.counts.nulls,
        saved.counts.wall_rejected,
        saved.counts.core_rejected,
        saved.counts.bath_rejected,
    ]
    .into_iter()
    .try_fold(0u64, |total, value| total.checked_add(value))
    .context("Checkpoint count overflow")?;
    ensure!(
        classified == saved.completed_attempts,
        "Checkpoint decision counters are inconsistent"
    );
    ensure!(
        saved.raw_points <= p.cfg.limits.raw_campaign
            && saved.retained_points <= p.cfg.limits.retained_campaign,
        "Checkpoint resource counters exceed budget"
    );
    ensure!(
        saved.charged_cpu_seconds.is_finite()
            && saved.charged_cpu_seconds >= 0.
            && saved.charged_wall_seconds.is_finite()
            && saved.charged_wall_seconds >= 0.,
        "Invalid checkpoint timing"
    );
    Ok(())
}

fn check_limits(
    budget: &Budget,
    clock: Instant,
    old_wall: f64,
    cfg: &ContextDockingConfig,
) -> Result<()> {
    budget.check_cpu()?;
    ensure!(
        old_wall + clock.elapsed().as_secs_f64() <= cfg.wall_seconds,
        "Fatal campaign wall budget exceeded"
    );
    Ok(())
}

/// New output directory per invocation, including deterministic continuations.
/// Resume only a completed prior invocation: partial failed clouds are never retried.
pub fn run(options: ContextDockingOptions) -> Result<Value> {
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    ensure!(!options.out.exists(), "Fresh output directory required");
    fs::create_dir_all(&options.out)?;
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(options.out.join("events.jsonl"))?;
    let mut journal = Journal {
        file: BufWriter::new(file),
        next: 0,
    };
    journal.emit(json!({"kind":"invocation_begun","protocol":PROTOCOL,"config":options.config,"model":options.model,"prior":options.prior,"method":options.method,"correlation":options.correlation,"resume":options.resume,"certify":options.certify}))?;
    let result = run_inner(&options, &mut journal, start, cpu_start);
    match result {
        Ok(value) => {
            save(&options.out.join("summary.json"), &value)?;
            Ok(value)
        }
        Err(error) => {
            let failed = json!({"kind":"fatal","complete":false,"passed":false,"error":format!("{error:#}"),"invocation_cpu_seconds":cpu_seconds()-cpu_start,"invocation_wall_seconds":start.elapsed().as_secs_f64(),"scope":"Fatal prefix retained; no partial cloud acceptance or retry."});
            journal.emit(failed.clone())?;
            save(&options.out.join("failure.json"), &failed)?;
            Err(error)
        }
    }
}

fn run_inner(
    options: &ContextDockingOptions,
    journal: &mut Journal,
    start: Instant,
    cpu_start: f64,
) -> Result<Value> {
    let p = prepare(options)?;
    let mut budget = Budget::new(p.cfg.limits)?;
    budget.started = cpu_start;
    check_limits(&budget, start, 0., &p.cfg)?;
    journal.emit(json!({"kind":"prepared","bindings":p.bindings,"identity":p.cfg.identity,"moving_label":p.source.moving_label,"fixed_labels":p.context.bodies.iter().map(|b|b.label).collect::<Vec<_>>(),"moving_state_index":p.fixed_poses.len(),"anchor_label":p.source.anchor_label,"anchor_fixed_index":p.anchor_index,"uniform_cube_half_width":p.cube_half_width,"wall_radius":p.wall.radius,"preparation_cpu_seconds":cpu_seconds()-cpu_start,"physical_target":"atomic-wall(x) hard(x,C) exp[z |E(x) intersect union E(C)|]; unbounded wall-permeable ideal bath","scope":"Fixed-context conditional test; historical snapshot preparation does not specify an equilibrated fluid at the run bath."}))?;
    let fixed = p.audit_fixed(&budget)?;
    journal.emit(json!({"kind":"fixed_context_audit","audit":fixed}))?;
    if let Some(path) = &options.certify {
        ensure!(
            options.resume.is_none(),
            "Certification cannot resume an MC checkpoint"
        );
        let bytes = fs::read(path)?;
        let candidates: Vec<CertificationCandidate> = serde_json::from_slice(&bytes)?;
        ensure!(
            !candidates.is_empty() && candidates.len() as u64 <= p.cfg.maximum_attempts,
            "Invalid certification allocation"
        );
        fs::write(options.out.join("provenance/candidates.json"), &bytes)?;
        let mut seen = BTreeSet::new();
        let mut records = Vec::new();
        for (ordinal, candidate) in candidates.iter().enumerate() {
            journal
                .emit(json!({"kind":"candidate_begun","ordinal":ordinal,"candidate":candidate}))?;
            check_limits(&budget, start, 0., &p.cfg)?;
            let (id, pose, metadata) = p.candidate(candidate)?;
            ensure!(seen.insert(id.clone()), "Duplicate candidate identity");
            let observation = p.observe(pose, &budget)?;
            let record = json!({"id":id,"ordinal":ordinal,"metadata":metadata,"pose":pose,"wall_valid":observation.wall_valid,"core_overlap_labels":observation.core_overlap_labels,"exclusion_contact_labels":observation.exclusion_contact_labels});
            journal.emit(json!({"kind":"candidate_complete","record":record}))?;
            records.push(record);
        }
        return Ok(
            json!({"schema":"fixed-context-certification-v1","complete":true,"passed":true,"bindings":p.bindings,"candidate_sha256":hash_bytes(&bytes),"fixed_context_audit":fixed,"records":records,"candidate_count":candidates.len(),"random_draws":0,"bath_calls":0,"preparation_and_observer_cpu_seconds":cpu_seconds()-cpu_start,"wall_seconds":start.elapsed().as_secs_f64(),"scope":"All listed exact point predicates only; invalid points do not exclude neighboring basins."}),
        );
    }
    ensure!(
        fixed["physical_context_valid"] == true,
        "Fixed context violates physical hard/wall support"
    );
    let mut pose = p.cfg.initial_pose.unwrap_or(p.source.pose);
    let mut completed_cycles = 0;
    let mut attempted = 0;
    let mut counts = Counts::default();
    let mut old_wall = 0.;
    let mut resume_hash = None;
    let mut resume_observation = None;
    if let Some(path) = &options.resume {
        let bytes = fs::read(path)?;
        let saved: Checkpoint = serde_json::from_slice(&bytes)?;
        checkpoint_matches(&saved, &p, options)?;
        let previous = path.parent().context("Checkpoint parent missing")?;
        let summary: Value = serde_json::from_slice(&fs::read(previous.join("summary.json"))?)?;
        ensure!(
            summary["complete"] == true
                && summary["passed"] == true
                && !previous.join("failure.json").exists(),
            "Resume requires a completed, nonfailed invocation"
        );
        ensure!(
            summary["checkpoint_sha256"] == hash_bytes(&bytes),
            "Checkpoint is not the completed predecessor's state"
        );
        pose = saved.pose;
        completed_cycles = saved.completed_cycles;
        attempted = saved.completed_attempts;
        counts = saved.counts;
        budget.raw = saved.raw_points;
        budget.retained = saved.retained_points;
        budget.started = cpu_start - saved.charged_cpu_seconds;
        old_wall = saved.charged_wall_seconds;
        resume_hash = Some(hash_bytes(&bytes));
        resume_observation = Some(saved.observation);
    }
    let slots = (p.cfg.local_attempts_per_cycle as u64) + 1;
    ensure!(
        completed_cycles < options.cycles
            && options
                .cycles
                .checked_mul(slots)
                .is_some_and(|n| n <= p.cfg.maximum_attempts),
        "Requested cycle allocation invalid or exceeds fixed attempt cap"
    );
    check_limits(&budget, start, old_wall, &p.cfg)?;
    let mut observation = p.observe(pose, &budget)?;
    if let Some(expected) = resume_observation {
        ensure!(
            observation == expected,
            "Checkpoint contact observation no longer matches geometry"
        );
    }
    ensure!(
        observation.physical_valid(),
        "Initial/checkpoint pose violates full physical support"
    );
    let preparation_cpu = cpu_seconds() - cpu_start;
    let sampler_start = cpu_seconds();
    let mut observer_cpu = 0.;
    let mut proposal_cpu = 0.;
    let mut geometry_cpu = 0.;
    let mut gate_cpu = 0.;
    journal.emit(json!({"kind":"initial_state","completed_cycles":completed_cycles,"completed_attempts":attempted,"pose":pose,"observation":observation,"counts":counts,"resume_sha256":resume_hash,"identity":p.cfg.identity,"preparation_cpu_seconds":preparation_cpu}))?;
    let mut frames = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(options.out.join("trajectory.jsonl"))?,
    );
    for cycle in completed_cycles + 1..=options.cycles {
        for slot in 0..=p.cfg.local_attempts_per_cycle {
            let is_global = slot == p.cfg.local_attempts_per_cycle
                && options.method == DockingMethod::PosteriorInvolution;
            let old = pose;
            let before_labels = observation.exclusion_contact_labels.clone();
            journal.emit(json!({"kind":"attempt_begun","cycle":cycle,"slot":slot,"attempt_index":attempted,"global":is_global,"old_pose":old,"identity":p.cfg.identity}))?;
            counts.attempted += 1;
            check_limits(&budget, start, old_wall, &p.cfg)?;
            let mut rngs = StepRngs {
                proposal: stream(p.cfg.seed, cycle, slot, "proposal"),
                gate: stream(p.cfg.seed, cycle, slot, "gate"),
                accept: stream(p.cfg.seed, cycle, slot, "accept"),
            };
            let result = physical_step(
                &p.tree,
                &p.wall,
                &p.fixed_poses,
                old,
                &p.local,
                if is_global { Some(&p.proposer) } else { None },
                &mut rngs,
                &mut budget,
            );
            let step = match result {
                Ok(value) => value,
                Err(error) => {
                    journal.emit(json!({"kind":"attempt_fatal","cycle":cycle,"slot":slot,"attempt_index":attempted,"failure":error}))?;
                    return Err(error.into());
                }
            };
            // Persist the physical decision BEFORE any observer or further
            // resource check can fail. A failed observer never erases a move.
            journal.emit(json!({"kind":"attempt_decision","cycle":cycle,"slot":slot,"attempt_index":attempted,"step":step}))?;
            pose = step.retained_pose;
            proposal_cpu += step.proposal_cpu_seconds;
            geometry_cpu += step.geometry_cpu_seconds;
            gate_cpu += step.gate_cpu_seconds;
            if step.accepted {
                counts.accepted += 1;
            }
            match step.status.as_str() {
                "wall_rejected" => counts.wall_rejected += 1,
                "core_rejected" => counts.core_rejected += 1,
                "bath_rejected" => counts.bath_rejected += 1,
                "accepted" => (),
                _ => anyhow::bail!("Unexpected completed step status"),
            }
            if pose != old {
                let t = cpu_seconds();
                observation = p.observe(pose, &budget)?;
                observer_cpu += cpu_seconds() - t;
                ensure!(
                    observation.physical_valid(),
                    "Accepted state failed retained-state observation"
                );
            }
            check_limits(&budget, start, old_wall, &p.cfg)?;
            counts.completed += 1;
            attempted += 1;
            journal.emit(json!({"kind":"attempt_complete","cycle":cycle,"slot":slot,"attempt_index":attempted-1,"global":is_global,"production":cycle>p.cfg.warmup_cycles,"old_pose":old,"proposed_pose":step.proposed_pose,"retained_pose":pose,"proposal":step.proposal,"accepted":step.accepted,"wall_valid":step.wall_valid,"core_valid":step.core_valid,"gate":step.gate,"log_proposal_reverse_forward":step.log_proposal_reverse_forward,"raw_log_acceptance":step.raw_log_acceptance,"log_acceptance":step.log_acceptance,"acceptance_uniform":step.acceptance_uniform,"log_uniform":step.log_uniform,"exclusion_contact_labels_before":before_labels,"exclusion_contact_labels":observation.exclusion_contact_labels,"counts":counts,"identity":p.cfg.identity,"sampler_cpu_seconds":cpu_seconds()-sampler_start,"invocation_cpu_seconds":cpu_seconds()-cpu_start}))?;
        }
        let saved = Checkpoint {
            protocol: PROTOCOL.into(),
            bindings: p.bindings.clone(),
            method: options.method,
            correlation: options.correlation,
            sample_every: options.sample_every,
            completed_cycles: cycle,
            completed_attempts: attempted,
            pose,
            observation: observation.clone(),
            counts: counts.clone(),
            raw_points: budget.raw,
            retained_points: budget.retained,
            charged_cpu_seconds: cpu_seconds() - budget.started,
            charged_wall_seconds: old_wall + start.elapsed().as_secs_f64(),
        };
        save(&options.out.join("checkpoint.json"), &saved)?;
        journal.emit(json!({"kind":"cycle_complete","cycle":cycle,"pose":pose,"observation":observation,"completed_attempts":attempted,"counts":counts,"sampler_cpu_seconds":cpu_seconds()-sampler_start}))?;
        if cycle % options.sample_every == 0 || cycle == options.cycles {
            serde_json::to_writer(
                &mut frames,
                &json!({"cycle":cycle,"pose":pose,"exclusion_contact_labels":observation.exclusion_contact_labels,"counts":counts,"sampler_cpu_seconds":cpu_seconds()-sampler_start}),
            )?;
            frames.write_all(b"\n")?;
            frames.flush()?;
        }
    }
    check_limits(&budget, start, old_wall, &p.cfg)?;
    Ok(
        json!({"schema":"fixed-context-docking-summary-v1","complete":true,"passed":true,"bindings":p.bindings,"identity":p.cfg.identity,"method":options.method,"correlation":options.correlation,"completed_cycles":options.cycles,"completed_attempts":attempted,"counts":counts,"pose":pose,"observation":observation,"raw_points":budget.raw,"retained_points":budget.retained,"checkpoint_sha256":hash_file(&options.out.join("checkpoint.json"))?,"preparation_cpu_seconds":preparation_cpu,"sampler_cpu_seconds":cpu_seconds()-sampler_start,"observer_cpu_seconds":observer_cpu,"proposal_cpu_seconds":proposal_cpu,"geometry_cpu_seconds":geometry_cpu,"gate_cpu_seconds":gate_cpu,"invocation_cpu_seconds":cpu_seconds()-cpu_start,"invocation_wall_seconds":start.elapsed().as_secs_f64(),"scope":"Full-wall one-mobile fixed-context physical sampling only. No equilibrium fluid, assembly stability, or native-registry inference."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Atom;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    const ID: Pose = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            ..ID
        }
    }
    fn limits() -> Limits {
        Limits {
            raw_per_leg: 100000,
            raw_per_outer: 100000,
            raw_campaign: 1000000,
            retained_per_leg: 100000,
            retained_per_outer: 100000,
            retained_campaign: 1000000,
            cpu_seconds: 30.,
        }
    }
    struct Fixture {
        root: PathBuf,
        cfg: ContextDockingConfig,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    impl Fixture {
        fn new() -> Result<Self> {
            let root = std::env::temp_dir().join(format!(
                "context-docking-tests-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root)?;
            let shape = Shape {
                name: "toy sphere".into(),
                volume: 0.,
                atoms: vec![Atom {
                    center: [0.; 3],
                    radius: 0.1,
                }],
            };
            save(&root.join("shape.json"), &shape)?;
            let shape_sha = hash_file(&root.join("shape.json"))?;
            let covariance: [[f64; 6]; 6] =
                std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 0.02 } else { 0. }));
            let means = [[0.; 6]; 2];
            let model = json!({"schema":"reciprocal-pose-mixture-v1","reciprocal_components":[true,false],
                "base_model":{"angular_length":1.3,"shape_sha256":shape_sha,"coordinate_convention":"anchor-body-relative",
                "anchors":[{"position":[0.6,0.,0.],"rotation":IDENTITY},{"position":[0.8,0.,0.],"rotation":IDENTITY}],
                "means":means,"covariances":[covariance,covariance],"weights":[0.7,0.3]}});
            save(&root.join("model.json"), &model)?;
            save(
                &root.join("context.json"),
                &json!({"schema":"fixed-outside-context-v1","anchor_label":4,"excluded_moving_labels":[77],"bodies":[{"label":4,"pose":ID}]}),
            )?;
            save(
                &root.join("source.json"),
                &json!({"schema":"saved-source-state-metadata-v1","moving_label":77,"pose":pose(0.6),"anchor_label":4,"anchor_pose":ID,"boundary":"spherical","spherical_wall_radius":3.,"coordinate_frame":"Saved spherical-center frame; no display offset, wrapping or pose transform.","coordinate_wall_center":[100.,-100.,300.],"endpoint_sha256":"e".repeat(64)}),
            )?;
            let expected_sha256 = ExpectedInputs {
                model: hash_file(&root.join("model.json"))?,
                shape: shape_sha,
                fixed_context: hash_file(&root.join("context.json"))?,
                source_state: hash_file(&root.join("source.json"))?,
            };
            let cfg = ContextDockingConfig {
                shape: root.join("shape.json"),
                fixed_context: root.join("context.json"),
                source_state: root.join("source.json"),
                expected_sha256,
                expected_fixed_body_count: 1,
                initial_pose: None,
                depletant_radius: 0.4,
                reservoir_density: 0.2,
                poisson_lambda_ratio: 2.,
                translation_steps: vec![0.02],
                rotation_steps_deg: vec![1.],
                rotation_probability: 0.5,
                local_attempts_per_cycle: 1,
                uniform_probability: 0.1,
                seed: 751,
                endpoint_gate: GateOptions {
                    max_cells: 31,
                    max_depth: 4,
                    min_width: 0.05,
                },
                limits: limits(),
                maximum_attempts: 100,
                wall_seconds: 60.,
                identity: json!({"job":"deterministic-toy"}),
                warmup_cycles: 1,
            };
            save(&root.join("config.json"), &cfg)?;
            Ok(Self { root, cfg })
        }
        fn options(&self, name: &str, cycles: u64) -> ContextDockingOptions {
            ContextDockingOptions {
                config: self.root.join("config.json"),
                model: self.root.join("model.json"),
                prior: None,
                out: self.root.join(name),
                cycles,
                sample_every: 1,
                method: DockingMethod::PosteriorInvolution,
                correlation: 0.65,
                resume: None,
                certify: None,
            }
        }
        fn update(&self) -> Result<()> {
            save(&self.root.join("config.json"), &self.cfg)
        }
        fn prior(&self, p: &Prepared) -> Result<PathBuf> {
            let base = p.proposer.virtual_branch_log_prior().to_vec();
            let eligible = vec![true, false, true];
            let logs = docking::defensive_virtual_branch_log_prior(&base, &eligible, 0.1)?;
            let branches:Vec<_>=p.branches.iter().enumerate().map(|(i,b)|json!({"virtual_label":i,"component_index":b.component_index,"inverted":b.inverted,"original_probability":b.weight,"original_log_probability":base[i],"eligible":eligible[i],"log_probability":logs[i],"probability":logs[i].exp()})).collect();
            let asset = json!({"schema":"fixed-context-virtual-branch-prior-v1","complete":true,"passed":true,"method":"posterior_involution","moving_labels":[77],"anchor_label":4,"floor_probability":0.1,"virtual_branch_count":3,"branches":branches,"log_prior":logs,"base_log_prior":base,"eligibility":eligible,"input_sha256":{"model":p.bindings["model"],"shape":p.bindings["shape"],"endpoint":"e".repeat(64)},"fixed_context_sha256":p.bindings["fixed_context"],"source_state_sha256":p.bindings["source_state"]});
            let path = self.root.join("prior.json");
            save(&path, &asset)?;
            Ok(path)
        }
        fn prepared(&self, name: &str) -> Result<Prepared> {
            let options = self.options(name, 1);
            fs::create_dir(&options.out)?;
            prepare(&options)
        }
    }
    fn events(path: &Path) -> Result<Vec<Value>> {
        fs::read_to_string(path)?
            .lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }

    #[test]
    fn cube_covers_offcenter_shape_and_atomic_wall_is_the_domain() -> Result<()> {
        let tree = SphereTree::new(Shape {
            name: String::new(),
            volume: 0.,
            atoms: vec![Atom {
                center: [10., 0., 0.],
                radius: 1.,
            }],
        })?;
        let wall = Container::new(3., &tree)?;
        let half = uniform_half_width(3., tree.bound)?;
        for angle in [0., 0.4, 1.] {
            let r = cayley([0., 0., angle]);
            let p = Pose {
                position: scale(matvec(r, [10., 0., 0.]), -1.),
                orientation: quaternion(r),
            };
            assert!(norm(p.position) > wall.radius && wall.contains(p));
            assert!(p.position.iter().all(|x| x.abs() < half));
        }
        assert!(wall.contains(pose(-8.)));
        assert!(!wall.contains(pose(-7.99)));
        Ok(())
    }

    #[test]
    fn completed_restart_replays_identical_physical_decisions_and_budgets() -> Result<()> {
        let f = Fixture::new()?;
        let all = run(f.options("all", 4))?;
        run(f.options("first", 2))?;
        let mut continuation = f.options("second", 4);
        continuation.resume = Some(f.root.join("first/checkpoint.json"));
        let second = run(continuation)?;
        for key in [
            "pose",
            "counts",
            "raw_points",
            "retained_points",
            "completed_attempts",
        ] {
            assert_eq!(all[key], second[key], "{key}");
        }
        let decisions = |name: &str| -> Result<Vec<Value>> {
            Ok(events(&f.root.join(name).join("events.jsonl"))?
                .into_iter()
                .filter(|v| v["kind"] == "attempt_decision")
                .map(|mut v| {
                    v.as_object_mut().unwrap().remove("event_index");
                    for k in [
                        "proposal_cpu_seconds",
                        "geometry_cpu_seconds",
                        "gate_cpu_seconds",
                    ] {
                        v["step"].as_object_mut().unwrap().remove(k);
                    }
                    v
                })
                .collect())
        };
        let full = decisions("all")?;
        let mut split = decisions("first")?;
        split.extend(decisions("second")?);
        assert_eq!(full, split);
        let retained = events(&f.root.join("all/events.jsonl"))?;
        let rows: Vec<_> = retained
            .iter()
            .filter(|x| x["kind"] == "attempt_complete")
            .collect();
        assert_eq!(rows.len(), 8);
        assert_eq!(rows.iter().filter(|x| x["production"] == false).count(), 2);
        Ok(())
    }

    #[test]
    fn mismatch_and_failed_prefix_cannot_resume_or_silently_change_inputs() -> Result<()> {
        let mut f = Fixture::new()?;
        run(f.options("first", 1))?;
        f.cfg.seed += 1;
        f.update()?;
        let mut mismatch = f.options("mismatch", 2);
        mismatch.resume = Some(f.root.join("first/checkpoint.json"));
        assert!(run(mismatch).is_err());
        assert!(f.root.join("mismatch/failure.json").exists());
        let e = events(&f.root.join("mismatch/events.jsonl"))?;
        assert!(!e.iter().any(|v| v["kind"] == "attempt_begun"));
        f.cfg.expected_sha256.shape = "0".repeat(64);
        f.update()?;
        assert!(run(f.options("changed-shape", 1)).is_err());
        let e = events(&f.root.join("changed-shape/events.jsonl"))?;
        assert_eq!(e.first().unwrap()["kind"], "invocation_begun");
        assert_eq!(e.last().unwrap()["kind"], "fatal");
        Ok(())
    }

    #[test]
    fn prior_binds_every_branch_and_context_but_not_initialized_moving_pose() -> Result<()> {
        let mut f = Fixture::new()?;
        let p = f.prepared("prepare-original")?;
        let prior = f.prior(&p)?;
        f.cfg.initial_pose = Some(pose(-0.7));
        f.update()?;
        let mut options = f.options("changed-start", 1);
        options.prior = Some(prior.clone());
        run(options)?;
        let mut asset: Value = serde_json::from_slice(&fs::read(&prior)?)?;
        asset["branches"][1]["inverted"] = json!(false);
        save(&prior, &asset)?;
        let mut wrong = f.options("wrong-label", 1);
        wrong.prior = Some(prior.clone());
        assert!(run(wrong).is_err());
        let mut local = f.options("unused-prior", 1);
        local.method = DockingMethod::Local;
        local.prior = Some(prior);
        assert!(run(local).is_err());
        Ok(())
    }

    #[test]
    fn source_core_wall_and_fixed_inventory_are_checked_before_sampling() -> Result<()> {
        let mut f = Fixture::new()?;
        for (name, initial) in [("source-core", ID), ("source-wall", pose(4.))] {
            f.cfg.initial_pose = Some(initial);
            f.update()?;
            assert!(run(f.options(name, 1)).is_err());
            assert!(
                !events(&f.root.join(name).join("events.jsonl"))?
                    .iter()
                    .any(|v| v["kind"] == "attempt_begun")
            );
        }
        f.cfg.initial_pose = None;
        f.cfg.expected_fixed_body_count = 2;
        f.update()?;
        assert!(run(f.options("missing-spectator", 1)).is_err());
        Ok(())
    }

    #[test]
    fn certification_retains_every_fixed_candidate_and_ignores_display_offset() -> Result<()> {
        let f = Fixture::new()?;
        let p = f.prepared("cert-fixture")?;
        let mut candidates = vec![CertificationCandidate::Source {
            id: "source77".into(),
        }];
        for (i, b) in p.branches.iter().enumerate() {
            candidates.push(CertificationCandidate::VirtualCenter {
                id: format!("branch-{i}"),
                virtual_branch: i,
                original_probability: b.weight,
            });
        }
        let path = f.root.join("candidates.json");
        save(&path, &candidates)?;
        let mut options = f.options("certificate", 1);
        options.certify = Some(path);
        let result = run(options)?;
        assert_eq!(result["candidate_count"], 4);
        assert_eq!(
            result["records"][0]["pose"],
            serde_json::to_value(p.source.pose)?
        );
        assert_eq!(result["random_draws"], 0);
        assert_eq!(result["bath_calls"], 0);
        assert_eq!(
            result["fixed_context_audit"]["physical_context_valid"],
            true
        );
        let e = events(&f.root.join("certificate/events.jsonl"))?;
        assert_eq!(
            e.iter().filter(|v| v["kind"] == "candidate_begun").count(),
            4
        );
        assert_eq!(
            e.iter()
                .filter(|v| v["kind"] == "candidate_complete")
                .count(),
            4
        );
        Ok(())
    }

    #[test]
    fn point_limit_preserves_partial_gate_and_never_accepts_a_partial_cloud() -> Result<()> {
        let mut f = Fixture::new()?;
        f.cfg.reservoir_density = 1.;
        f.cfg.poisson_lambda_ratio = 1e9;
        f.cfg.rotation_probability = 0.;
        f.cfg.limits.raw_per_leg = 0;
        f.cfg.limits.raw_per_outer = 0;
        f.cfg.limits.raw_campaign = 0;
        f.update()?;
        let mut options = f.options("bounded-failure", 1);
        options.method = DockingMethod::Local;
        assert!(run(options).is_err());
        let e = events(&f.root.join("bounded-failure/events.jsonl"))?;
        let failed = e
            .iter()
            .find(|v| v["kind"] == "attempt_fatal")
            .context("Missing partial failure")?;
        assert!(
            failed["failure"]["partial_gate"]["failed_progress"]["gate"]["raw_points"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(
            failed["failure"]["partial_gate"]["failed_progress"]["processed_points"],
            0
        );
        assert_eq!(failed["failure"]["partial_trace"]["accepted"], false);
        assert!(!e.iter().any(|v| v["kind"] == "attempt_complete"));
        assert!(!f.root.join("bounded-failure/checkpoint.json").exists());
        Ok(())
    }

    #[test]
    fn physical_step_uses_full_context_prior_ratio_in_one_bath_decision() -> Result<()> {
        let f = Fixture::new()?;
        let mut p = f.prepared("step-fixture")?;
        p.proposer = p.proposer.with_virtual_branch_log_prior(&[
            0.1_f64.ln(),
            0.6_f64.ln(),
            0.3_f64.ln(),
        ])?;
        let mut budget = Budget::new(limits())?;
        let mut checked = 0;
        for index in 0..16 {
            let mut rngs = StepRngs {
                proposal: StdRng::seed_from_u64(804 + index),
                gate: StdRng::seed_from_u64(1804 + index),
                accept: StdRng::seed_from_u64(2804 + index),
            };
            let step = physical_step(
                &p.tree,
                &p.wall,
                &p.fixed_poses,
                p.source.pose,
                &p.local,
                Some(&p.proposer),
                &mut rngs,
                &mut budget,
            )?;
            if let Some(gate) = step.gate {
                let correction = step.log_proposal_reverse_forward.unwrap();
                assert!(close(
                    step.raw_log_acceptance.unwrap(),
                    correction + gate.log_weight
                ));
                assert_eq!(
                    step.accepted,
                    step.acceptance_uniform.unwrap().ln() < step.log_acceptance.unwrap()
                );
                if step.proposal["branch"] == "involution" {
                    let logsum = |v: Vec<f64>| {
                        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
                    };
                    let old = logsum(
                        p.proposer
                            .branch_log_densities(step.old_pose, p.source.anchor_pose)?,
                    );
                    let new =
                        logsum(p.proposer.branch_log_densities(
                            step.proposed_pose.unwrap(),
                            p.source.anchor_pose,
                        )?);
                    assert!(close(correction, old - new));
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
        Ok(())
    }
}
